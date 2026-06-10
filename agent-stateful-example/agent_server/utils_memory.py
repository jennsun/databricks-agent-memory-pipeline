import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional

from databricks_langchain import AsyncCheckpointSaver, AsyncDatabricksStore
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.store.base import BaseStore
from mlflow.types.responses import ResponsesAgentRequest

from agent_server.utils import _is_databricks_app_env

logger = logging.getLogger(__name__)

MEMORY_ROOT = "/memories/"
USER_NAMESPACE_PREFIX = "user_memories"
AGENT_NAMESPACE_PREFIX = "agent_memories"


@dataclass(frozen=True)
class LakebaseConfig:
    """Configuration for a single Lakebase connection."""
    autoscaling_project: Optional[str] = None
    autoscaling_branch: Optional[str] = None
    autoscaling_endpoint: Optional[str] = None
    instance_name: Optional[str] = None
    embedding_endpoint: str = "databricks-gte-large-en"
    embedding_dims: int = 1024
    memory_schema: Optional[str] = None

    @property
    def description(self) -> str:
        return self.autoscaling_endpoint or self.instance_name or f"{self.autoscaling_project}/{self.autoscaling_branch}"


def init_user_lakebase_config() -> LakebaseConfig:
    """Initialize Lakebase config for user-scoped memory."""
    return LakebaseConfig(
        autoscaling_project=os.getenv("LAKEBASE_USER_PROJECT"),
        autoscaling_branch=os.getenv("LAKEBASE_USER_BRANCH"),
        autoscaling_endpoint=os.getenv("LAKEBASE_USER_ENDPOINT"),
        embedding_endpoint=os.getenv("DATABRICKS_EMBEDDING_ENDPOINT", "databricks-gte-large-en"),
        memory_schema=os.getenv("LAKEBASE_AGENT_MEMORY_SCHEMA"),
    )


def init_agent_lakebase_config() -> LakebaseConfig:
    """Initialize Lakebase config for agent-scoped memory (read-only)."""
    return LakebaseConfig(
        autoscaling_project=os.getenv("LAKEBASE_AGENT_PROJECT"),
        autoscaling_branch=os.getenv("LAKEBASE_AGENT_BRANCH"),
        autoscaling_endpoint=os.getenv("LAKEBASE_AGENT_ENDPOINT"),
        embedding_endpoint=os.getenv("DATABRICKS_EMBEDDING_ENDPOINT", "databricks-gte-large-en"),
        memory_schema=os.getenv("LAKEBASE_AGENT_MEMORY_SCHEMA"),
    )


def get_user_id(request: ResponsesAgentRequest) -> Optional[str]:
    custom_inputs = dict(request.custom_inputs or {})
    if "user_id" in custom_inputs:
        return custom_inputs["user_id"]
    if request.context and getattr(request.context, "user_id", None):
        return request.context.user_id
    return None


def get_lakebase_access_error_message(lakebase_description: str) -> str:
    if _is_databricks_app_env():
        app_name = os.getenv("DATABRICKS_APP_NAME")
        return (
            f"Failed to connect to Lakebase '{lakebase_description}'. "
            f"The App Service Principal for '{app_name}' may not have access.\n\n"
            "To fix this:\n"
            "1. Go to the Databricks UI and navigate to your app\n"
            "2. Click 'Edit' -> 'App resources' -> 'Add resource'\n"
            "3. Add your Lakebase instance as a resource\n"
            "4. Grant the necessary permissions."
        )
    else:
        return (
            f"Failed to connect to Lakebase '{lakebase_description}'. "
            "Please verify:\n"
            "1. The configuration is correct\n"
            "2. You have the necessary permissions\n"
            "3. Your Databricks authentication is configured correctly"
        )


@asynccontextmanager
async def lakebase_context(config: LakebaseConfig):
    """Yield (checkpointer, store) for short-term and long-term memory."""
    async with AsyncCheckpointSaver(
        instance_name=config.instance_name,
        autoscaling_endpoint=config.autoscaling_endpoint,
        project=config.autoscaling_project,
        branch=config.autoscaling_branch,
        schema=config.memory_schema,
    ) as checkpointer, AsyncDatabricksStore(
        instance_name=config.instance_name,
        autoscaling_endpoint=config.autoscaling_endpoint,
        project=config.autoscaling_project,
        branch=config.autoscaling_branch,
        embedding_endpoint=config.embedding_endpoint,
        embedding_dims=config.embedding_dims,
        embedding_fields=["content"],
        schema=config.memory_schema,
    ) as store:
        yield checkpointer, store


@asynccontextmanager
async def agent_store_context(config: LakebaseConfig):
    """Yield a read-only store for agent-scoped memory."""
    async with AsyncDatabricksStore(
        instance_name=config.instance_name,
        autoscaling_endpoint=config.autoscaling_endpoint,
        project=config.autoscaling_project,
        branch=config.autoscaling_branch,
        embedding_endpoint=config.embedding_endpoint,
        embedding_dims=config.embedding_dims,
        embedding_fields=["content"],
        schema=config.memory_schema,
    ) as store:
        yield store


async def run_lakebase_setup(user_config: LakebaseConfig, agent_config: LakebaseConfig) -> None:
    """Run database migrations for checkpoint and store tables."""
    async with lakebase_context(user_config) as (checkpointer, store):
        await checkpointer.setup()
        await store.setup()
    logger.info("User Lakebase setup complete")
    async with agent_store_context(agent_config) as store:
        await store.setup()
    logger.info("Agent Lakebase setup complete")


def _normalize_path(path: str) -> str:
    """Normalize a memory file path. Returns the cleaned path or raises ValueError."""
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be a non-empty string")
    p = path.strip()
    if not p.startswith("/"):
        p = "/" + p
    if not p.startswith(MEMORY_ROOT):
        raise ValueError(f"path must start with '{MEMORY_ROOT}' (got: {path!r})")
    if not p.endswith(".md"):
        raise ValueError(f"path must end with '.md' (got: {path!r})")
    if "//" in p or p.endswith("/"):
        raise ValueError(f"path contains empty segments (got: {path!r})")
    return p


def _normalize_directory(directory: str) -> str:
    """Normalize a directory prefix. Always returns a string ending in '/'."""
    if not isinstance(directory, str) or not directory.strip():
        return MEMORY_ROOT
    d = directory.strip()
    if not d.startswith("/"):
        d = "/" + d
    if not d.startswith(MEMORY_ROOT) and d != MEMORY_ROOT.rstrip("/"):
        # silently accept and coerce to /memories/<dir>/ for friendliness
        d = MEMORY_ROOT + d.lstrip("/")
    if not d.endswith("/"):
        d = d + "/"
    return d


def _value_field(value: Any, field: str, default: Any = "") -> Any:
    """Safely extract a field from a stored value (which should be a dict)."""
    if isinstance(value, dict):
        return value.get(field, default)
    return default


def _format_search_result(item: Any, snippet_chars: int = 400) -> str:
    """Format a single SearchItem as a path + content snippet."""
    content = _value_field(item.value, "content", "")
    description = _value_field(item.value, "description", "")
    snippet = content if len(content) <= snippet_chars else content[:snippet_chars] + "..."
    score = getattr(item, "score", None)
    score_str = f" (score={score:.3f})" if isinstance(score, float) else ""
    desc_str = f"\n_{description}_\n" if description else ""
    return f"## {item.key}{score_str}{desc_str}\n{snippet}"


def _format_listing(items: list[Any]) -> str:
    """Format a listing of items as a tree with description (if present) and size."""
    if not items:
        return "(empty)"
    lines = []
    for item in items:
        content = _value_field(item.value, "content", "")
        description = _value_field(item.value, "description", "")
        flag = " [always-loaded]" if _value_field(item.value, "startup_load", False) else ""
        desc = f" — {description}" if description else ""
        lines.append(f"- `{item.key}` ({len(content)} chars){flag}{desc}")
    return "\n".join(lines)


def _build_memory_map(items: list[Any], heading: str) -> str:
    """Build a markdown 'map' of a namespace's files (paths + descriptions)."""
    if not items:
        return f"## {heading}\n\n(empty)\n"
    items = sorted(items, key=lambda it: it.key)
    lines = [f"## {heading}", ""]
    for item in items:
        description = _value_field(item.value, "description", "")
        flag = " [always-loaded]" if _value_field(item.value, "startup_load", False) else ""
        desc = f" — {description}" if description else ""
        lines.append(f"- `{item.key}`{flag}{desc}")
    return "\n".join(lines) + "\n"


def _build_startup_load_section(items: list[Any], heading: str) -> str:
    """Inline the full content of any file marked startup_load=true."""
    always_on = [it for it in items if _value_field(it.value, "startup_load", False)]
    if not always_on:
        return ""
    always_on.sort(key=lambda it: it.key)
    lines = [f"## {heading}", ""]
    for item in always_on:
        content = _value_field(item.value, "content", "")
        lines.append(f"### `{item.key}`\n\n{content}".rstrip())
        lines.append("")
    return "\n".join(lines) + "\n"


async def build_memory_preamble(
    user_store: Optional[BaseStore],
    user_namespace: Optional[tuple[str, str]],
    agent_store: Optional[BaseStore],
) -> str:
    """Build a per-session preamble appended to the system prompt.

    Includes:
      1. A `memory.md` map: every path in the user + agent stores with descriptions.
      2. The full content of any file marked `startup_load: true` in either store.

    The agent sees this on every turn, so it always knows what files exist and has
    the most-critical content inlined without needing a tool call to fetch it.
    """
    sections: list[str] = ["# Your memory snapshot at session start", ""]

    # --- User-scoped memory ---
    user_items: list[Any] = []
    if user_store is not None and user_namespace is not None:
        try:
            user_items = await user_store.asearch(user_namespace, limit=500)
        except Exception as e:
            logger.warning("Failed to list user memory for preamble: %s", e)
    sections.append(_build_memory_map(user_items, "Your user memory files"))

    # --- Agent-scoped memory ---
    agent_items: list[Any] = []
    if agent_store is not None:
        try:
            agent_items = await agent_store.asearch((AGENT_NAMESPACE_PREFIX,), limit=500)
        except Exception as e:
            logger.warning("Failed to list agent memory for preamble: %s", e)
    sections.append(_build_memory_map(agent_items, "Shared agent knowledge files"))

    # --- Always-on file contents ---
    user_startup = _build_startup_load_section(user_items, "Always-loaded: your user memory")
    if user_startup:
        sections.append(user_startup)
    agent_startup = _build_startup_load_section(agent_items, "Always-loaded: shared agent knowledge")
    if agent_startup:
        sections.append(agent_startup)

    return "\n".join(sections).rstrip() + "\n"


def memory_tools():
    """Returns filesystem-style memory tools.

    User-scoped (read/write): ls_memories, read_memory, write_memory, edit_memory,
    delete_memory, search_user_memories.
    Agent-scoped (read-only): ls_agent_memories, read_agent_memory, search_agent_memories.

    Paths must live under /memories/ and end with .md. By convention, organize as:
      /memories/episodic/    — events and what happened (timestamps, conversations)
      /memories/semantic/    — facts, preferences, identity (timeless)
      /memories/procedural/  — how-to workflows and rules
    """

    def _get_user_namespace(config: RunnableConfig) -> Optional[tuple[str, str]]:
        user_id = config.get("configurable", {}).get("user_id")
        if not user_id:
            return None
        return (USER_NAMESPACE_PREFIX, user_id)

    def _get_user_store(config: RunnableConfig) -> Optional[BaseStore]:
        return config.get("configurable", {}).get("user_store")

    def _get_agent_store(config: RunnableConfig) -> Optional[BaseStore]:
        return config.get("configurable", {}).get("agent_store")

    AGENT_NAMESPACE = (AGENT_NAMESPACE_PREFIX,)

    # ------------------------------------------------------------------
    # User-scoped tools (read/write)
    # ------------------------------------------------------------------

    @tool
    async def ls_memories(directory: str, config: RunnableConfig) -> str:
        """List your memory files under a directory.

        Args:
            directory: Path prefix to list, e.g. "/memories/", "/memories/semantic/".
                Use "/memories/" to see everything you've saved.

        Returns markdown listing of paths and content size. Use read_memory(path) to
        open any one of them.
        """
        namespace = _get_user_namespace(config)
        if namespace is None:
            return "Memory not available — no user_id provided."
        store = _get_user_store(config)
        if not store:
            return "Memory not available — user store not configured."

        prefix = _normalize_directory(directory)
        items = await store.asearch(namespace, limit=200)
        matching = [it for it in items if it.key.startswith(prefix)]
        if not matching:
            return f"No memory files found under {prefix}"
        matching.sort(key=lambda it: it.key)
        return f"Files under {prefix} ({len(matching)} total):\n" + _format_listing(matching)

    @tool
    async def read_memory(path: str, config: RunnableConfig) -> str:
        """Read the full content of a memory file by exact path.

        Args:
            path: Full path including the .md extension, e.g.
                "/memories/semantic/coding_preferences.md".

        Returns the markdown body of the file, or an error if the file doesn't exist.
        """
        namespace = _get_user_namespace(config)
        if namespace is None:
            return "Memory not available — no user_id provided."
        store = _get_user_store(config)
        if not store:
            return "Memory not available — user store not configured."

        try:
            normalized = _normalize_path(path)
        except ValueError as e:
            return f"Invalid path: {e}"

        item = await store.aget(namespace, normalized)
        if item is None:
            return f"File not found: {normalized}"
        content = item.value.get("content", "") if isinstance(item.value, dict) else ""
        return f"# {normalized}\n\n{content}"

    @tool
    async def write_memory(
        path: str,
        content: str,
        config: RunnableConfig,
        description: str = "",
        always_load: bool = False,
    ) -> str:
        """Create a new memory file or completely overwrite an existing one.

        Args:
            path: Full path under /memories/, ending with .md. Examples:
                "/memories/semantic/coding_preferences.md"
                "/memories/episodic/events_log.md"
                "/memories/procedural/review_pr.md"
            content: The full markdown body to store. This REPLACES the file —
                use edit_memory for surgical changes to a long file.
            description: Optional one-line summary of what this file contains.
                Shown in the memory map at session start so the agent (you) and
                tools can decide whether to read the full file. Keep under 100 chars.
            always_load: If True, the full content of this file is automatically
                injected into your system prompt at the start of EVERY future
                session for this user. Use sparingly — only for absolutely
                critical, frequently-referenced facts (e.g., a strict allergy, a
                non-negotiable formatting rule). Default False.

        By convention, use:
          /memories/episodic/     for things that happened (events, conversations)
          /memories/semantic/     for facts/preferences/identity (timeless)
          /memories/procedural/   for how-to workflows and rules
        """
        namespace = _get_user_namespace(config)
        if namespace is None:
            return "Cannot save memory — no user_id provided."
        store = _get_user_store(config)
        if not store:
            return "Cannot save memory — user store not configured."

        try:
            normalized = _normalize_path(path)
        except ValueError as e:
            return f"Invalid path: {e}"
        if not isinstance(content, str):
            return "Invalid content: must be a string."

        existing = await store.aget(namespace, normalized)
        value: dict[str, Any] = {"content": content}
        if description:
            value["description"] = description
        if always_load:
            value["startup_load"] = True
        await store.aput(namespace, normalized, value)
        action = "Overwrote" if existing is not None else "Created"
        suffix = " [always-loaded]" if always_load else ""
        return f"{action} {normalized} ({len(content)} chars){suffix}."

    @tool
    async def edit_memory(
        path: str, old_text: str, new_text: str, config: RunnableConfig
    ) -> str:
        """Make a surgical edit to an existing memory file by exact string replacement.

        Args:
            path: Full path of the file to edit.
            old_text: Exact substring to find. Must appear exactly once in the file.
                Include enough surrounding context to make it unique.
            new_text: Replacement text. Use "" to delete the old_text.

        Use this instead of write_memory when you want to append/update part of a
        long file without rewriting the whole thing.
        """
        namespace = _get_user_namespace(config)
        if namespace is None:
            return "Cannot edit memory — no user_id provided."
        store = _get_user_store(config)
        if not store:
            return "Cannot edit memory — user store not configured."

        try:
            normalized = _normalize_path(path)
        except ValueError as e:
            return f"Invalid path: {e}"

        item = await store.aget(namespace, normalized)
        if item is None:
            return f"Cannot edit — file not found: {normalized}. Use write_memory to create it."
        content = item.value.get("content", "") if isinstance(item.value, dict) else ""

        occurrences = content.count(old_text)
        if occurrences == 0:
            return f"old_text not found in {normalized}. The file content has not been changed."
        if occurrences > 1:
            return (
                f"old_text appears {occurrences} times in {normalized}. "
                "Provide more surrounding context so it matches exactly once."
            )

        new_content = content.replace(old_text, new_text, 1)
        new_value: dict[str, Any] = dict(item.value) if isinstance(item.value, dict) else {}
        new_value["content"] = new_content
        await store.aput(namespace, normalized, new_value)
        return f"Edited {normalized} ({len(content)} → {len(new_content)} chars)."

    @tool
    async def delete_memory(path: str, config: RunnableConfig) -> str:
        """Delete a memory file by exact path.

        Args:
            path: Full path of the file to delete.
        """
        namespace = _get_user_namespace(config)
        if namespace is None:
            return "Cannot delete memory — no user_id provided."
        store = _get_user_store(config)
        if not store:
            return "Cannot delete memory — user store not configured."

        try:
            normalized = _normalize_path(path)
        except ValueError as e:
            return f"Invalid path: {e}"

        await store.adelete(namespace, normalized)
        return f"Deleted {normalized}."

    @tool
    async def search_user_memories(query: str, config: RunnableConfig) -> str:
        """Semantic search across all your memory files.

        Args:
            query: Natural-language description of what you want to find, e.g.
                "user's preferred programming language" or "how to format money".

        Returns the top 5 files most relevant to the query, with path and a snippet.
        Use this as your first action on any user turn to look up relevant context.
        """
        namespace = _get_user_namespace(config)
        if namespace is None:
            return "Memory not available — no user_id provided."
        store = _get_user_store(config)
        if not store:
            return "Memory not available — user store not configured."

        results = await store.asearch(namespace, query=query, limit=5)
        if not results:
            return "No memories found matching your query."
        formatted = "\n\n".join(_format_search_result(item) for item in results)
        return f"Top {len(results)} matches for {query!r}:\n\n{formatted}"

    # ------------------------------------------------------------------
    # Agent-scoped tools (read-only, shared across users)
    # ------------------------------------------------------------------

    @tool
    async def ls_agent_memories(directory: str, config: RunnableConfig) -> str:
        """List files in the agent's shared knowledge base under a directory.

        Args:
            directory: Path prefix to list, e.g. "/memories/" or "/memories/procedural/".

        These files apply to all users and are managed by admins (read-only at runtime).
        """
        store = _get_agent_store(config)
        if not store:
            return "Agent memory not available — agent store not configured."

        prefix = _normalize_directory(directory)
        items = await store.asearch(AGENT_NAMESPACE, limit=200)
        matching = [it for it in items if it.key.startswith(prefix)]
        if not matching:
            return f"No agent-memory files found under {prefix}"
        matching.sort(key=lambda it: it.key)
        return f"Agent files under {prefix} ({len(matching)} total):\n" + _format_listing(matching)

    @tool
    async def read_agent_memory(path: str, config: RunnableConfig) -> str:
        """Read the full content of a file in the agent's shared knowledge base.

        Args:
            path: Full path including the .md extension, e.g.
                "/memories/procedural/money_formatting.md".
        """
        store = _get_agent_store(config)
        if not store:
            return "Agent memory not available — agent store not configured."

        try:
            normalized = _normalize_path(path)
        except ValueError as e:
            return f"Invalid path: {e}"

        item = await store.aget(AGENT_NAMESPACE, normalized)
        if item is None:
            return f"Agent file not found: {normalized}"
        content = item.value.get("content", "") if isinstance(item.value, dict) else ""
        return f"# {normalized}\n\n{content}"

    @tool
    async def search_agent_memories(query: str, config: RunnableConfig) -> str:
        """Semantic search across the agent's shared knowledge base.

        Args:
            query: Natural-language description of what you want to find.

        Returns the top 5 most-relevant shared knowledge files. These rules apply
        to all users — always run this at the start of a turn to surface relevant
        formatting rules, currency rules, tone preferences, etc.
        """
        store = _get_agent_store(config)
        if not store:
            return "Agent memory not available — agent store not configured."

        results = await store.asearch(AGENT_NAMESPACE, query=query, limit=5)
        if not results:
            return "No agent memories found matching your query."
        formatted = "\n\n".join(_format_search_result(item) for item in results)
        return f"Top {len(results)} agent matches for {query!r}:\n\n{formatted}"

    return [
        ls_memories,
        read_memory,
        write_memory,
        edit_memory,
        delete_memory,
        search_user_memories,
        ls_agent_memories,
        read_agent_memory,
        search_agent_memories,
    ]

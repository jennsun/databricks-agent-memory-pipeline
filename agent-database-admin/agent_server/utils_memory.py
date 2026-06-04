import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional

from databricks.sdk import WorkspaceClient
from databricks_langchain import AsyncCheckpointSaver, AsyncDatabricksStore
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.store.base import BaseStore
from mlflow.types.responses import ResponsesAgentRequest

from agent_server.utils import _is_databricks_app_env

logger = logging.getLogger(__name__)

# ---- memory model constants ----
MEMORY_ROOT = "/memories/"
AGENT_NAMESPACE_PREFIX = "agent_memories"
AGENT_NAMESPACE: tuple[str, ...] = (AGENT_NAMESPACE_PREFIX,)
EMBEDDING_FIELDS: list[str] = ["content"]


@dataclass(frozen=True)
class LakebaseConfig:
    instance_name: Optional[str]
    autoscaling_endpoint: Optional[str]
    autoscaling_project: Optional[str]
    autoscaling_branch: Optional[str]
    embedding_endpoint: str = "databricks-gte-large-en"  # override via DATABRICKS_EMBEDDING_ENDPOINT
    embedding_dims: int = 1024
    memory_schema: Optional[str] = None

    @property
    def description(self) -> str:
        return self.autoscaling_endpoint or self.instance_name or f"{self.autoscaling_project}/{self.autoscaling_branch}"


def init_lakebase_config() -> LakebaseConfig:
    endpoint = os.getenv("LAKEBASE_AUTOSCALING_ENDPOINT") or None
    raw_name = os.getenv("LAKEBASE_INSTANCE_NAME") or None
    project = os.getenv("LAKEBASE_AUTOSCALING_PROJECT") or None
    branch = os.getenv("LAKEBASE_AUTOSCALING_BRANCH") or None

    has_autoscaling = project and branch
    if not endpoint and not raw_name and not has_autoscaling:
        raise ValueError(
            "Lakebase configuration is required but not set. "
            "Please set one of the following in your environment:\n"
            "  Option 1 (autoscaling endpoint): LAKEBASE_AUTOSCALING_ENDPOINT=<your-endpoint-name>\n"
            "  Option 2 (autoscaling): LAKEBASE_AUTOSCALING_PROJECT=<project> and LAKEBASE_AUTOSCALING_BRANCH=<branch>\n"
            "  Option 3 (provisioned): LAKEBASE_INSTANCE_NAME=<your-instance-name>\n"
        )

    # Priority: endpoint > project+branch > instance_name (mutually exclusive in the library)
    if endpoint:
        instance_name = None
        project = None
        branch = None
    elif has_autoscaling:
        instance_name = None
        endpoint = None
    else:
        instance_name = resolve_lakebase_instance_name(raw_name)
        endpoint = None
        project = None
        branch = None

    embedding_endpoint = os.getenv("DATABRICKS_EMBEDDING_ENDPOINT", "databricks-gte-large-en")
    memory_schema = os.getenv("LAKEBASE_AGENT_MEMORY_SCHEMA") or None
    return LakebaseConfig(
        instance_name=instance_name,
        autoscaling_endpoint=endpoint,
        autoscaling_project=project,
        autoscaling_branch=branch,
        embedding_endpoint=embedding_endpoint,
        memory_schema=memory_schema,
    )


def _is_lakebase_hostname(value: str) -> bool:
    """Check if the value looks like a Lakebase hostname rather than an instance name."""
    # Hostname pattern: instance-{uuid}.database.{env}.cloud.databricks.com
    return ".database." in value and value.endswith(".com")


def resolve_lakebase_instance_name(
    instance_name: str, workspace_client: Optional[WorkspaceClient] = None
) -> str:
    """Resolve a Lakebase instance name from a hostname if needed."""
    if not _is_lakebase_hostname(instance_name):
        return instance_name

    client = workspace_client or WorkspaceClient()
    hostname = instance_name

    try:
        instances = list(client.database.list_database_instances())
    except Exception as exc:
        raise ValueError(
            f"Unable to list database instances to resolve hostname '{hostname}'. "
            "Ensure you have access to database instances."
        ) from exc

    for instance in instances:
        rw_dns = getattr(instance, "read_write_dns", None)
        ro_dns = getattr(instance, "read_only_dns", None)
        if hostname in (rw_dns, ro_dns):
            resolved_name = getattr(instance, "name", None)
            if not resolved_name:
                raise ValueError(
                    f"Found matching instance for hostname '{hostname}' "
                    "but instance name is not available."
                )
            logging.info(f"Resolved Lakebase hostname '{hostname}' to instance name '{resolved_name}'")
            return resolved_name

    raise ValueError(
        f"Unable to find database instance matching hostname '{hostname}'. "
        "Ensure the hostname is correct and the instance exists."
    )


async def run_lakebase_setup(config: LakebaseConfig) -> None:
    """Run database migrations for checkpoint and store tables. Call once at app startup."""
    async with lakebase_context(config) as (checkpointer, store):
        await checkpointer.setup()
        await store.setup()
    logger.info("Lakebase setup complete")


def get_user_id(request: ResponsesAgentRequest) -> Optional[str]:
    custom_inputs = dict(request.custom_inputs or {})
    if "user_id" in custom_inputs:
        return custom_inputs["user_id"]
    if request.context and getattr(request.context, "user_id", None):
        return request.context.user_id
    return None


def get_lakebase_access_error_message(lakebase_instance_name: str) -> str:
    """Generate a helpful error message for Lakebase access issues."""
    if _is_databricks_app_env():
        app_name = os.getenv("DATABRICKS_APP_NAME")
        return (
            f"Failed to connect to Lakebase instance '{lakebase_instance_name}'. "
            f"The App Service Principal for '{app_name}' may not have access.\n\n"
            "To fix this:\n"
            "1. Go to the Databricks UI and navigate to your app\n"
            "2. Click 'Edit' → 'App resources' → 'Add resource'\n"
            "3. Add your Lakebase instance as a resource\n"
            "4. Grant the necessary permissions on your Lakebase instance."
        )
    else:
        return (
            f"Failed to connect to Lakebase instance '{lakebase_instance_name}'. "
            "Please verify:\n"
            "1. The instance name is correct\n"
            "2. You have the necessary permissions to access the instance\n"
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
        embedding_fields=EMBEDDING_FIELDS,
        schema=config.memory_schema,
    ) as store:
        yield checkpointer, store


# =============================================================================
# Filesystem memory: helpers
# =============================================================================

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
        d = MEMORY_ROOT + d.lstrip("/")
    if not d.endswith("/"):
        d = d + "/"
    return d


def _value_field(value: Any, field: str, default: Any = "") -> Any:
    if isinstance(value, dict):
        return value.get(field, default)
    return default


def _format_search_result(item: Any, snippet_chars: int = 400) -> str:
    content = _value_field(item.value, "content", "")
    description = _value_field(item.value, "description", "")
    snippet = content if len(content) <= snippet_chars else content[:snippet_chars] + "..."
    score = getattr(item, "score", None)
    score_str = f" (score={score:.3f})" if isinstance(score, float) else ""
    desc_str = f"\n_{description}_\n" if description else ""
    return f"## {item.key}{score_str}{desc_str}\n{snippet}"


def _format_listing(items: list[Any]) -> str:
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


async def build_memory_preamble(store: Optional[BaseStore]) -> str:
    """Per-session preamble: memory map + always-loaded file contents."""
    sections: list[str] = ["# Your memory snapshot at session start", ""]
    items: list[Any] = []
    if store is not None:
        try:
            items = await store.asearch(AGENT_NAMESPACE, limit=500)
        except Exception as e:
            logger.warning("Failed to list agent memory for preamble: %s", e)
    sections.append(_build_memory_map(items, "Agent memory files (shared across all users)"))
    startup = _build_startup_load_section(items, "Always-loaded: shared agent knowledge")
    if startup:
        sections.append(startup)
    return "\n".join(sections).rstrip() + "\n"


# =============================================================================
# Filesystem memory tools — read/write on the agent_memories namespace
# =============================================================================

def memory_tools():
    """Filesystem-style tools for the SHARED agent knowledge base.

    Unlike agent-stateful-example (where the runtime is read-only on agent
    memories), this admin app exposes the full read/write/edit/delete surface.

    Paths live under /memories/ and end with .md. By convention:
      /memories/episodic/    — what happened (incidents, audit entries)
      /memories/semantic/    — timeless facts and definitions
      /memories/procedural/  — how-to workflows and rules (the most common bucket
                                for this admin app)
    """

    def _get_store(config: RunnableConfig) -> Optional[BaseStore]:
        return config.get("configurable", {}).get("store")

    @tool
    async def ls_agent_memories(directory: str, config: RunnableConfig) -> str:
        """List files in the shared agent memory under a directory.

        Args:
            directory: Path prefix to list, e.g. "/memories/" or "/memories/procedural/".

        Returns markdown listing of paths with descriptions and content size.
        """
        store = _get_store(config)
        if not store:
            return "Agent memory not available — store not configured."

        prefix = _normalize_directory(directory)
        items = await store.asearch(AGENT_NAMESPACE, limit=200)
        matching = [it for it in items if it.key.startswith(prefix)]
        if not matching:
            return f"No agent-memory files found under {prefix}"
        matching.sort(key=lambda it: it.key)
        return f"Files under {prefix} ({len(matching)} total):\n" + _format_listing(matching)

    @tool
    async def read_agent_memory(path: str, config: RunnableConfig) -> str:
        """Read the full content of a shared agent memory file by exact path.

        Args:
            path: Full path including the .md extension, e.g.
                "/memories/procedural/expense_report_workflow.md".
        """
        store = _get_store(config)
        if not store:
            return "Agent memory not available — store not configured."

        try:
            normalized = _normalize_path(path)
        except ValueError as e:
            return f"Invalid path: {e}"

        item = await store.aget(AGENT_NAMESPACE, normalized)
        if item is None:
            return f"Agent file not found: {normalized}"
        content = _value_field(item.value, "content", "")
        return f"# {normalized}\n\n{content}"

    @tool
    async def write_agent_memory(
        path: str,
        content: str,
        config: RunnableConfig,
        description: str = "",
        always_load: bool = False,
    ) -> str:
        """Create a new shared agent memory file or completely overwrite an existing one.

        Args:
            path: Full path under /memories/, ending with .md. Examples:
                "/memories/procedural/expense_report_workflow.md"
                "/memories/semantic/team_glossary.md"
                "/memories/episodic/2026-06-01_policy_update.md"
            content: The full markdown body to store. REPLACES the file — use
                edit_agent_memory for surgical changes to a long file.
            description: Optional one-line summary of what this file contains.
                Shown in the memory map at session start. Keep under 100 chars.
            always_load: If True, the full content is injected into every agent's
                system prompt at session start. Use sparingly — only for
                critical, frequently-referenced rules that apply to ALL users.

        By convention:
          /memories/episodic/    for events, audit entries, dated decisions
          /memories/semantic/    for facts, definitions, glossaries
          /memories/procedural/  for how-to workflows and rules (most common)
        """
        store = _get_store(config)
        if not store:
            return "Cannot save agent memory — store not configured."

        try:
            normalized = _normalize_path(path)
        except ValueError as e:
            return f"Invalid path: {e}"
        if not isinstance(content, str):
            return "Invalid content: must be a string."

        existing = await store.aget(AGENT_NAMESPACE, normalized)
        value: dict[str, Any] = {"content": content}
        if description:
            value["description"] = description
        if always_load:
            value["startup_load"] = True
        await store.aput(AGENT_NAMESPACE, normalized, value)
        action = "Overwrote" if existing is not None else "Created"
        suffix = " [always-loaded]" if always_load else ""
        return f"{action} {normalized} ({len(content)} chars){suffix}."

    @tool
    async def edit_agent_memory(
        path: str, old_text: str, new_text: str, config: RunnableConfig
    ) -> str:
        """Make a surgical edit to an existing shared agent memory file.

        Args:
            path: Full path of the file to edit.
            old_text: Exact substring to find. Must appear exactly once.
                Include enough surrounding context to make it unique.
            new_text: Replacement text. Use "" to delete the old_text.

        Use this instead of write_agent_memory when updating part of a long file.
        Preserves the file's description and always_load flag.
        """
        store = _get_store(config)
        if not store:
            return "Cannot edit agent memory — store not configured."

        try:
            normalized = _normalize_path(path)
        except ValueError as e:
            return f"Invalid path: {e}"

        item = await store.aget(AGENT_NAMESPACE, normalized)
        if item is None:
            return f"Cannot edit — file not found: {normalized}. Use write_agent_memory to create it."
        content = _value_field(item.value, "content", "")

        occurrences = content.count(old_text)
        if occurrences == 0:
            return f"old_text not found in {normalized}. The file has not been changed."
        if occurrences > 1:
            return (
                f"old_text appears {occurrences} times in {normalized}. "
                "Provide more surrounding context so it matches exactly once."
            )

        new_content = content.replace(old_text, new_text, 1)
        new_value: dict[str, Any] = dict(item.value) if isinstance(item.value, dict) else {}
        new_value["content"] = new_content
        await store.aput(AGENT_NAMESPACE, normalized, new_value)
        return f"Edited {normalized} ({len(content)} → {len(new_content)} chars)."

    @tool
    async def delete_agent_memory(path: str, config: RunnableConfig) -> str:
        """Delete a shared agent memory file by exact path.

        Args:
            path: Full path of the file to delete.
        """
        store = _get_store(config)
        if not store:
            return "Cannot delete agent memory — store not configured."

        try:
            normalized = _normalize_path(path)
        except ValueError as e:
            return f"Invalid path: {e}"

        existing = await store.aget(AGENT_NAMESPACE, normalized)
        if existing is None:
            return f"Nothing to delete — {normalized} does not exist."
        await store.adelete(AGENT_NAMESPACE, normalized)
        return f"Deleted {normalized}."

    @tool
    async def search_agent_memories(query: str, config: RunnableConfig) -> str:
        """Semantic search across the shared agent memory.

        Args:
            query: Natural-language description of what you want to find, e.g.
                "how to write an expense report" or "money formatting rules".

        Returns the top 5 files most relevant to the query, with path and snippet.
        Use this when you don't know the exact path of the file you need.
        """
        store = _get_store(config)
        if not store:
            return "Agent memory not available — store not configured."

        results = await store.asearch(AGENT_NAMESPACE, query=query, limit=5)
        if not results:
            return "No agent memories found matching your query."
        formatted = "\n\n".join(_format_search_result(item) for item in results)
        return f"Top {len(results)} matches for {query!r}:\n\n{formatted}"

    return [
        ls_agent_memories,
        read_agent_memory,
        write_agent_memory,
        edit_agent_memory,
        delete_agent_memory,
        search_agent_memories,
    ]

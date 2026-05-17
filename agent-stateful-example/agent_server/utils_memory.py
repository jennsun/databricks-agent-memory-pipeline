import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional

from databricks_langchain import AsyncCheckpointSaver, AsyncDatabricksStore
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.store.base import BaseStore
from mlflow.types.responses import ResponsesAgentRequest

from agent_server.utils import _is_databricks_app_env

logger = logging.getLogger(__name__)


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


def memory_tools():
    """Returns user-scoped memory tools (get, save, delete) and agent-scoped memory tool (read)."""

    @tool
    async def get_user_memory(query: str, config: RunnableConfig) -> str:
        """Search for relevant information about the user from long-term memory."""
        user_id = config.get("configurable", {}).get("user_id")
        if not user_id:
            return "Memory not available - no user_id provided."

        store: Optional[BaseStore] = config.get("configurable", {}).get("user_store")
        if not store:
            return "Memory not available - user store not configured."

        namespace = ("user_memories", user_id)
        results = await store.asearch(namespace, query=query, limit=5)

        if not results:
            return "No memories found for this user."

        memory_items = [f"- [{item.key}]: {json.dumps(item.value)}" for item in results]
        return f"Found {len(results)} relevant memories:\n" + "\n".join(memory_items)

    @tool
    async def save_user_memory(memory_key: str, memory_data_json: str, config: RunnableConfig) -> str:
        """Save information about the user to long-term memory."""
        user_id = config.get("configurable", {}).get("user_id")
        if not user_id:
            return "Cannot save memory - no user_id provided."

        store: Optional[BaseStore] = config.get("configurable", {}).get("user_store")
        if not store:
            return "Cannot save memory - user store not configured."

        namespace = ("user_memories", user_id)

        try:
            memory_data = json.loads(memory_data_json)
            if not isinstance(memory_data, dict):
                return f"Failed: memory_data must be a JSON object, not {type(memory_data).__name__}"
            await store.aput(namespace, memory_key, memory_data)
            return f"Successfully saved memory '{memory_key}' for user."
        except json.JSONDecodeError as e:
            return f"Failed to save memory: Invalid JSON - {e}"

    @tool
    async def delete_user_memory(memory_key: str, config: RunnableConfig) -> str:
        """Delete a specific memory from the user's long-term memory."""
        user_id = config.get("configurable", {}).get("user_id")
        if not user_id:
            return "Cannot delete memory - no user_id provided."

        store: Optional[BaseStore] = config.get("configurable", {}).get("user_store")
        if not store:
            return "Cannot delete memory - user store not configured."

        namespace = ("user_memories", user_id)
        await store.adelete(namespace, memory_key)
        return f"Successfully deleted memory '{memory_key}' for user."

    @tool
    async def read_agent_memory(query: str, config: RunnableConfig) -> str:
        """Search the agent's shared knowledge base for relevant information.
        This memory contains curated knowledge that applies to all users."""
        agent_store: Optional[BaseStore] = config.get("configurable", {}).get("agent_store")
        if not agent_store:
            return "Agent memory not available - agent store not configured."

        namespace = ("agent_memories",)
        results = await agent_store.asearch(namespace, query=query, limit=5)

        if not results:
            return "No relevant agent memories found."

        memory_items = [f"- [{item.key}]: {json.dumps(item.value)}" for item in results]
        return f"Found {len(results)} relevant agent memories:\n" + "\n".join(memory_items)

    return [get_user_memory, save_user_memory, delete_user_memory, read_agent_memory]

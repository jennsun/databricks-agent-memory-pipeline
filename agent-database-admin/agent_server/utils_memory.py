import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional

from databricks.sdk import WorkspaceClient
from databricks_langchain import AsyncCheckpointSaver, AsyncDatabricksStore
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.store.base import BaseStore
from mlflow.types.responses import ResponsesAgentRequest

from agent_server.utils import _is_databricks_app_env

logger = logging.getLogger(__name__)


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
    """
    Resolve a Lakebase instance name from a hostname if needed.

    If the input is a hostname (e.g., from Databricks Apps value_from resolution),
    this will resolve it to the actual instance name by listing database instances.

    Args:
        instance_name: Either an instance name or a hostname
        workspace_client: Optional WorkspaceClient to use for resolution

    Returns:
        The resolved instance name

    Raises:
        ValueError: If the hostname cannot be resolved to an instance name
    """
    if not _is_lakebase_hostname(instance_name):
        # Input is already an instance name
        return instance_name

    # Input is a hostname - resolve to instance name
    client = workspace_client or WorkspaceClient()
    hostname = instance_name

    try:
        instances = list(client.database.list_database_instances())
    except Exception as exc:
        raise ValueError(
            f"Unable to list database instances to resolve hostname '{hostname}'. "
            "Ensure you have access to database instances."
        ) from exc

    # Find the instance that matches this hostname
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
            "4. Grant the necessary permissions on your Lakebase instance. "
            "See the README section 'Grant Lakebase permissions to your App's Service Principal' for the SQL commands."
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
        schema=config.memory_schema,
    ) as store:
        yield checkpointer, store


AGENT_MEMORY_NAMESPACE = ("agent_memories",)


def memory_tools():
    @tool
    async def get_agent_memory(query: str, config: RunnableConfig) -> str:
        """Search the agent-wide long-term memory for relevant information."""
        store: Optional[BaseStore] = config.get("configurable", {}).get("store")
        if not store:
            return "Memory not available - store not configured."

        results = await store.asearch(AGENT_MEMORY_NAMESPACE, query=query, limit=5)

        if not results:
            return "No agent memories found."

        memory_items = [f"- [{item.key}]: {json.dumps(item.value)}" for item in results]
        return f"Found {len(results)} relevant agent memories:\n" + "\n".join(memory_items)

    @tool
    async def save_agent_memory(memory_key: str, memory_data_json: str, config: RunnableConfig) -> str:
        """Save information to the agent-wide long-term memory."""
        store: Optional[BaseStore] = config.get("configurable", {}).get("store")
        if not store:
            return "Cannot save memory - store not configured."

        try:
            memory_data = json.loads(memory_data_json)
            if not isinstance(memory_data, dict):
                return f"Failed: memory_data must be a JSON object, not {type(memory_data).__name__}"
            await store.aput(AGENT_MEMORY_NAMESPACE, memory_key, memory_data)
            return f"Successfully saved agent memory '{memory_key}'."
        except json.JSONDecodeError as e:
            return f"Failed to save memory: Invalid JSON - {e}"

    @tool
    async def delete_agent_memory(memory_key: str, config: RunnableConfig) -> str:
        """Delete a specific memory from the agent-wide long-term memory."""
        store: Optional[BaseStore] = config.get("configurable", {}).get("store")
        if not store:
            return "Cannot delete memory - store not configured."

        await store.adelete(AGENT_MEMORY_NAMESPACE, memory_key)
        return f"Successfully deleted agent memory '{memory_key}'."

    return [get_agent_memory, save_agent_memory, delete_agent_memory]

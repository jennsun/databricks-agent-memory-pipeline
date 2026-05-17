import json
import logging
from datetime import datetime
from typing import Any, AsyncGenerator, Optional, Sequence, TypedDict

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks
from fastapi import HTTPException
from langchain.agents import create_agent
from langchain_core.messages import AnyMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.store.base import BaseStore
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    to_chat_completions_input,
)
from typing_extensions import Annotated

from agent_server.prompts import SYSTEM_PROMPT
from agent_server.utils import (
    _get_or_create_thread_id,
    get_user_workspace_client,
    load_mcp_tools,
    process_agent_astream_events,
)
from agent_server.utils_memory import (
    agent_store_context,
    get_lakebase_access_error_message,
    get_user_id,
    init_agent_lakebase_config,
    init_user_lakebase_config,
    lakebase_context,
    memory_tools,
)

logger = logging.getLogger(__name__)
mlflow.langchain.autolog()
logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.ERROR)
sp_workspace_client = WorkspaceClient()

LLM_ENDPOINT_NAME = "databricks-claude-sonnet-4"
USER_LAKEBASE_CONFIG = init_user_lakebase_config()
AGENT_LAKEBASE_CONFIG = init_agent_lakebase_config()


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().isoformat()


class StatefulAgentState(TypedDict, total=False):
    messages: Annotated[Sequence[AnyMessage], add_messages]
    custom_inputs: dict[str, Any]
    custom_outputs: dict[str, Any]


def _extract_tool_content(content) -> str:
    """Extract readable text from structured MCP tool responses.

    Genie MCP returns content blocks like ``[{"type":"text","text":"...","id":"..."}]``.
    The ``text`` field for Genie contains a JSON string with ``textAttachments``
    (human-readable summaries) and ``queryAttachments`` (SQL + raw data).
    We extract the text summaries so the LLM gets a clean, concise input.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                raw_text = block["text"]
                # Try to extract Genie textAttachments for a concise summary
                try:
                    parsed = json.loads(raw_text)
                    if isinstance(parsed, dict) and "content" in parsed:
                        inner = parsed["content"]
                        if "textAttachments" in inner and inner["textAttachments"]:
                            texts.extend(inner["textAttachments"])
                            continue
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass
                texts.append(raw_text)
            elif isinstance(block, str):
                texts.append(block)
        return "\n".join(texts) if texts else json.dumps(content)

    return json.dumps(content)


class _SanitizedChatDatabricks(ChatDatabricks):
    """ChatDatabricks wrapper that sanitizes ToolMessage content.

    Genie MCP returns structured content with extra fields (e.g. ``id``)
    that the LLM API rejects with 'Extra inputs are not permitted'.
    This extracts readable text from tool responses before sending to the LLM.

    All execution paths (_generate, _agenerate, _stream, _astream) are
    overridden because LangChain's _agenerate_with_cache may route through
    _astream for streaming-capable models, bypassing _agenerate.
    """

    def _sanitize(self, messages):
        sanitized = []
        for msg in messages:
            if hasattr(msg, "type") and msg.type == "tool" and not isinstance(msg.content, str):
                msg = msg.copy(update={"content": _extract_tool_content(msg.content)})
            sanitized.append(msg)
        return sanitized

    def _generate(self, messages, *args, **kwargs):
        return super()._generate(self._sanitize(messages), *args, **kwargs)

    async def _agenerate(self, messages, *args, **kwargs):
        return await super()._agenerate(self._sanitize(messages), *args, **kwargs)

    def _stream(self, messages, *args, **kwargs):
        yield from super()._stream(self._sanitize(messages), *args, **kwargs)

    async def _astream(self, messages, *args, **kwargs):
        async for chunk in super()._astream(self._sanitize(messages), *args, **kwargs):
            yield chunk


async def init_agent(
    user_store: BaseStore,
    workspace_client: Optional[WorkspaceClient] = None,
    checkpointer: Optional[Any] = None,
):
    wc = workspace_client or sp_workspace_client
    tools = [get_current_time] + memory_tools()
    tools.extend(await load_mcp_tools(wc))

    model = _SanitizedChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

    return create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
        store=user_store,
        state_schema=StatefulAgentState,
    )


@invoke()
async def invoke_handler(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    outputs = [
        event.item
        async for event in stream_handler(request)
        if event.type == "response.output_item.done"
    ]

    custom_outputs: dict[str, Any] = {}
    if user_id := get_user_id(request):
        custom_outputs["user_id"] = user_id
    return ResponsesAgentResponse(output=outputs, custom_outputs=custom_outputs)


@stream()
async def stream_handler(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    thread_id = _get_or_create_thread_id(request)
    mlflow.update_current_trace(metadata={"mlflow.trace.session": thread_id})

    user_id = get_user_id(request)
    if not user_id:
        logger.warning("No user_id provided - memory features will not be available")

    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    if user_id:
        config["configurable"]["user_id"] = user_id

    input_state: dict[str, Any] = {
        "messages": to_chat_completions_input([i.model_dump() for i in request.input]),
        "custom_inputs": dict(request.custom_inputs or {}),
    }

    try:
        async with lakebase_context(USER_LAKEBASE_CONFIG) as (checkpointer, user_store), \
                   agent_store_context(AGENT_LAKEBASE_CONFIG) as agent_store:
            config["configurable"]["user_store"] = user_store
            config["configurable"]["agent_store"] = agent_store

            agent = await init_agent(
                user_store=user_store,
                checkpointer=checkpointer,
            )

            async for event in process_agent_astream_events(
                agent.astream(input_state, config, stream_mode=["updates", "messages"])
            ):
                yield event
    except Exception as e:
        error_msg = str(e).lower()
        if any(
            keyword in error_msg
            for keyword in ["lakebase", "pg_hba", "postgres", "database instance"]
        ):
            logger.error("Lakebase access error: %s", e)
            raise HTTPException(
                status_code=503,
                detail=get_lakebase_access_error_message(USER_LAKEBASE_CONFIG.description),
            ) from e
        raise

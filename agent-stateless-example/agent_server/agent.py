import json
import logging
from datetime import datetime
from typing import AsyncGenerator, Optional

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks, DatabricksMCPServer, DatabricksMultiServerMCPClient
from langchain.agents import create_agent
from langchain_core.tools import tool
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    to_chat_completions_input,
)

from agent_server.utils import (
    get_databricks_host_from_env,
    get_session_id,
    get_user_workspace_client,
    process_agent_astream_events,
)

logger = logging.getLogger(__name__)
mlflow.langchain.autolog()
logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.ERROR)
sp_workspace_client = WorkspaceClient()

LLM_ENDPOINT_NAME = "databricks-claude-sonnet-4"

SYSTEM_PROMPT = """You are a helpful assistant with access to web search, code execution, and employee expense data tools.

- Use web search (you-com-search) to find up-to-date information from the internet.
- Use python_exec to run Python code for calculations, data analysis, or other programming tasks.
- Use the expense-data Genie space to query employee expense data — it supports natural language questions about spending patterns by employee, merchant, category, and date.
- Always cite your sources when using web search results.
"""


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().isoformat()


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


def _mcp_servers(workspace_client: WorkspaceClient) -> list[DatabricksMCPServer]:
    host_name = get_databricks_host_from_env()
    return [
        DatabricksMCPServer(
            name="system-ai",
            url=f"{host_name}/api/2.0/mcp/functions/system/ai",
            workspace_client=workspace_client,
            handle_tool_error=True,
        ),
        DatabricksMCPServer(
            name="you-com-search",
            url=f"{host_name}/api/2.0/mcp/external/you-com-search",
            workspace_client=workspace_client,
            handle_tool_error=True,
        ),
        DatabricksMCPServer(
            name="expense-data",
            url=f"{host_name}/api/2.0/mcp/genie/<your-genie-space-id>",
            workspace_client=workspace_client,
            handle_tool_error=True,
            timeout=60.0,
        ),
    ]


async def init_agent(workspace_client: Optional[WorkspaceClient] = None):
    wc = workspace_client or sp_workspace_client
    tools = [get_current_time]
    for server in _mcp_servers(wc):
        try:
            client = DatabricksMultiServerMCPClient([server])
            tools.extend(await client.get_tools())
            logger.info("Loaded MCP tools from '%s'", server.name)
        except Exception:
            logger.warning("Failed to fetch tools from MCP server '%s'. Skipping.", server.name, exc_info=True)
    return create_agent(
        tools=tools,
        model=_SanitizedChatDatabricks(endpoint=LLM_ENDPOINT_NAME),
        system_prompt=SYSTEM_PROMPT,
    )


@invoke()
async def invoke_handler(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    outputs = [
        event.item
        async for event in stream_handler(request)
        if event.type == "response.output_item.done"
    ]
    return ResponsesAgentResponse(output=outputs)


@stream()
async def stream_handler(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    if session_id := get_session_id(request):
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})

    agent = await init_agent()
    messages = {"messages": to_chat_completions_input([i.model_dump() for i in request.input])}

    async for event in process_agent_astream_events(
        agent.astream(input=messages, stream_mode=["updates", "messages"])
    ):
        yield event

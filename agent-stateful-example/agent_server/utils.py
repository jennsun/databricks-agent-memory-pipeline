import json
import logging
import os
import time as time_mod
from typing import Any, AsyncGenerator, AsyncIterator, Optional
from uuid import uuid4

import uuid_utils
from databricks.sdk import WorkspaceClient
from databricks_langchain import DatabricksMCPServer, DatabricksMultiServerMCPClient
from langchain_core.messages import AIMessageChunk, ToolMessage
from mlflow.genai.agent_server import get_request_headers
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentStreamEvent,
    create_function_call_item,
    create_function_call_output_item,
    create_text_output_item,
)


def _get_or_create_thread_id(request: ResponsesAgentRequest) -> str:
    # priority of getting thread id:
    # 1. Use thread id from custom inputs
    # 2. Use conversation id from ChatContext https://mlflow.org/docs/latest/api_reference/python_api/mlflow.types.html#mlflow.types.agent.ChatContext
    # 3. Generate random UUID
    ci = dict(request.custom_inputs or {})

    if "thread_id" in ci and ci["thread_id"]:
        return str(ci["thread_id"])

    if request.context and getattr(request.context, "conversation_id", None):
        return str(request.context.conversation_id)

    return str(uuid_utils.uuid7())


def _is_databricks_app_env() -> bool:
    """Check if running in a Databricks App environment."""
    return bool(os.getenv("DATABRICKS_APP_NAME"))


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
            url=f"{host_name}/api/2.0/mcp/genie/01f15172b4f911ffb116cfffb242a1ce",
            workspace_client=workspace_client,
            handle_tool_error=True,
            timeout=60.0,
        ),
    ]


def _wrap_tool_with_error_catch(tool):
    """Make a tool surface any exception as a tool-result string instead of raising.

    LangGraph's default ToolNode only catches ToolInvocationError. An uncaught
    McpError (PERMISSION_DENIED, transient MCP failure, etc.) propagates up
    and ends the turn, leaving the checkpoint with a `tool_use` block that
    has no matching `tool_result`. The next turn then 400s with
    "tool_use ids without tool_result blocks". Catching here keeps the
    checkpoint consistent and lets the agent recover.
    """
    original_coroutine = getattr(tool, "coroutine", None)
    if original_coroutine is None:
        return tool

    response_format = getattr(tool, "response_format", "content")

    async def safe_coroutine(*args, **kwargs):
        try:
            return await original_coroutine(*args, **kwargs)
        except Exception as e:
            logging.warning("Tool '%s' raised %s: %s", tool.name, type(e).__name__, e)
            error_msg = f"Tool '{tool.name}' failed: {type(e).__name__}: {e}"
            # When response_format='content_and_artifact', the coroutine must
            # return (content, artifact). Returning a bare string breaks LangChain
            # with "a two-tuple of the message content and raw tool output is expected".
            if response_format == "content_and_artifact":
                return error_msg, None
            return error_msg

    try:
        tool.coroutine = safe_coroutine
    except Exception:
        logging.warning("Could not wrap coroutine on tool '%s'", tool.name)
    return tool


async def load_mcp_tools(workspace_client: WorkspaceClient) -> list:
    """Load MCP tools per-server so individual failures don't break all tools."""
    tools = []
    for server in _mcp_servers(workspace_client):
        try:
            client = DatabricksMultiServerMCPClient([server])
            server_tools = await client.get_tools()
            tools.extend(_wrap_tool_with_error_catch(t) for t in server_tools)
            logging.info("Loaded MCP tools from '%s'", server.name)
        except Exception:
            logging.warning("Failed to fetch tools from MCP server '%s'. Skipping.", server.name, exc_info=True)
    return tools


def get_user_workspace_client() -> WorkspaceClient:
    token = get_request_headers().get("x-forwarded-access-token")
    return WorkspaceClient(token=token, auth_type="pat")


def get_databricks_host_from_env() -> Optional[str]:
    try:
        w = WorkspaceClient()
        return w.config.host
    except Exception as e:
        logging.exception("Error getting databricks host from env: %s", e)
        return None


_FAKE_ID_PREFIX = "resp_placeholder_"


def replace_fake_id(obj: Any, real_id: str) -> Any:
    """Recursively replace any resp_placeholder_* ID with real_id in dicts/lists/strings."""
    if isinstance(obj, dict):
        return {k: replace_fake_id(v, real_id) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_fake_id(item, real_id) for item in obj]
    elif isinstance(obj, str) and obj.startswith(_FAKE_ID_PREFIX):
        return real_id
    return obj


async def process_agent_astream_events(
    async_stream: AsyncIterator[Any],
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    response_id = f"{_FAKE_ID_PREFIX}{uuid4().hex[:16]}"
    in_turn = False
    turn_output_items: list[dict] = []
    output_index = 0
    active_text_item_id: str | None = None
    active_text_content = ""
    active_tool_calls: dict[int, dict] = {}

    def _response_obj(output: list[dict] | None = None) -> dict:
        return {
            "id": response_id,
            "created_at": time_mod.time(),
            "object": "response",
            "output": output or [],
            "status": None,
        }

    def _start_turn():
        nonlocal in_turn, turn_output_items
        in_turn = True
        turn_output_items = []

    def _end_turn():
        nonlocal in_turn, active_text_item_id, active_text_content
        in_turn = False
        active_text_item_id = None
        active_text_content = ""

    async for event in async_stream:
        if event[0] == "messages":
            try:
                chunk = event[1][0]
                if not isinstance(chunk, AIMessageChunk):
                    continue

                if not in_turn:
                    _start_turn()
                    yield ResponsesAgentStreamEvent(
                        type="response.created",
                        response=_response_obj(),
                    )

                # Tool call chunks
                if chunk.tool_call_chunks:
                    for tc_chunk in chunk.tool_call_chunks:
                        idx = tc_chunk.get("index", 0)
                        name = tc_chunk.get("name") or ""
                        tc_id = tc_chunk.get("id") or ""
                        args = tc_chunk.get("args") or ""

                        if idx not in active_tool_calls:
                            item_id = str(uuid_utils.uuid7())
                            active_tool_calls[idx] = {
                                "item_id": item_id,
                                "name": name,
                                "args": "",
                                "call_id": tc_id,
                                "output_index": output_index,
                            }
                            output_index += 1
                            yield ResponsesAgentStreamEvent(
                                type="response.output_item.added",
                                item={
                                    "type": "function_call",
                                    "id": item_id,
                                    "call_id": tc_id,
                                    "name": name,
                                    "arguments": "",
                                },
                                output_index=active_tool_calls[idx]["output_index"],
                            )
                        else:
                            tc_info = active_tool_calls[idx]
                            if name and not tc_info["name"]:
                                tc_info["name"] = name
                            if tc_id and not tc_info["call_id"]:
                                tc_info["call_id"] = tc_id

                        if args:
                            active_tool_calls[idx]["args"] += args
                            yield ResponsesAgentStreamEvent(
                                type="response.function_call_arguments.delta",
                                delta=args,
                                item_id=active_tool_calls[idx]["item_id"],
                                output_index=active_tool_calls[idx]["output_index"],
                            )

                # Text content
                elif chunk.content:
                    content = chunk.content
                    if not active_text_item_id:
                        active_text_item_id = str(uuid_utils.uuid7())
                        active_text_content = ""
                        yield ResponsesAgentStreamEvent(
                            type="response.output_item.added",
                            item={
                                "type": "message",
                                "id": active_text_item_id,
                                "role": "assistant",
                                "status": "in_progress",
                                "content": [],
                            },
                            output_index=output_index,
                        )
                        yield ResponsesAgentStreamEvent(
                            type="response.content_part.added",
                            item_id=active_text_item_id,
                            output_index=output_index,
                            content_index=0,
                            part={"type": "output_text", "text": "", "annotations": []},
                        )

                    active_text_content += content
                    yield ResponsesAgentStreamEvent(
                        type="response.output_text.delta",
                        delta=content,
                        item_id=active_text_item_id,
                        content_index=0,
                        output_index=output_index,
                    )

            except Exception as e:
                logging.exception(f"Error processing agent stream event: {e}")

        elif event[0] == "updates":
            for node_data in event[1].values():
                messages = node_data.get("messages", [])
                if not messages:
                    continue

                has_ai_message = False

                for i, msg in enumerate(messages):
                    if isinstance(msg, ToolMessage):
                        # Tool result — standalone event between turns
                        content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
                        item = create_function_call_output_item(
                            call_id=msg.tool_call_id,
                            output=content,
                        )
                        yield ResponsesAgentStreamEvent(
                            type="response.output_item.done",
                            item=item,
                        )

                    elif hasattr(msg, "tool_calls") and msg.tool_calls:
                        has_ai_message = True
                        if not in_turn:
                            _start_turn()
                            yield ResponsesAgentStreamEvent(
                                type="response.created",
                                response=_response_obj(),
                            )

                        for j, tc in enumerate(msg.tool_calls):
                            call_id = tc.get("id", "")
                            name = tc.get("name", "")
                            args = tc.get("args", {})
                            args_str = json.dumps(args) if isinstance(args, dict) else str(args)

                            # Match to active tool call by chunk index
                            tc_info = active_tool_calls.get(j)
                            if tc_info:
                                item_id = tc_info["item_id"]
                                matched_oi = tc_info["output_index"]
                            else:
                                item_id = str(uuid_utils.uuid7())
                                matched_oi = output_index
                                output_index += 1

                            item = create_function_call_item(
                                id=item_id,
                                call_id=call_id,
                                name=name,
                                arguments=args_str,
                            )
                            turn_output_items.append(item)
                            yield ResponsesAgentStreamEvent(
                                type="response.output_item.done",
                                item=item,
                                output_index=matched_oi,
                            )

                        active_tool_calls.clear()

                    elif hasattr(msg, "content") and msg.content:
                        has_ai_message = True
                        if not in_turn:
                            _start_turn()
                            yield ResponsesAgentStreamEvent(
                                type="response.created",
                                response=_response_obj(),
                            )

                        # When deltas already streamed the text, reuse the exact
                        # accumulated stream content so the client can dedupe the
                        # redundant output_item.done. msg.content may differ in
                        # formatting (list-of-blocks vs string) and break dedup.
                        if active_text_item_id and active_text_content:
                            text = active_text_content
                        else:
                            text = msg.content if isinstance(msg.content, str) else str(msg.content)
                        item_id = active_text_item_id or str(uuid_utils.uuid7())

                        if not active_text_item_id:
                            yield ResponsesAgentStreamEvent(
                                type="response.output_item.added",
                                item={
                                    "type": "message",
                                    "id": item_id,
                                    "role": "assistant",
                                    "status": "in_progress",
                                    "content": [],
                                },
                                output_index=output_index,
                            )
                            yield ResponsesAgentStreamEvent(
                                type="response.content_part.added",
                                item_id=item_id,
                                output_index=output_index,
                                content_index=0,
                                part={"type": "output_text", "text": "", "annotations": []},
                            )

                        yield ResponsesAgentStreamEvent(
                            type="response.content_part.done",
                            item_id=item_id,
                            output_index=output_index,
                            content_index=0,
                            part={"type": "output_text", "text": text, "annotations": []},
                        )

                        item = create_text_output_item(text=text, id=item_id)
                        item["status"] = "completed"
                        turn_output_items.append(item)
                        yield ResponsesAgentStreamEvent(
                            type="response.output_item.done",
                            item=item,
                            output_index=output_index,
                        )
                        output_index += 1
                        active_text_item_id = None
                        active_text_content = ""

                if has_ai_message and in_turn:
                    yield ResponsesAgentStreamEvent(
                        type="response.completed",
                        response=_response_obj(turn_output_items),
                    )
                    _end_turn()

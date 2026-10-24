"""ACP bridge: TCP server mapping ACP client sessions onto durable
AgentSession workflows.

    ACP client --tcp/NDJSON-RPC--> BridgeAgent --Temporal--> AgentSession

Mapping (session_id == Temporal workflow_id minus the configured prefix):
    initialize          -> capabilities (load_session supported)
    new_session         -> start_workflow(AgentSession, id="<prefix><sid>")
    load_session        -> verify workflow alive; replay offset resets to 0
    list_sessions       -> Temporal workflow-id query (prefix match)
    prompt              -> execute_update(prompt) then stream replay until
                           turn_end; returns PromptResponse(stop_reason)
    cancel (notif)      -> execute_update(cancel) — in-flight turn cancelled
    pr merged (signal)  -> session wakes with a system prompt (watcher.py)

Streaming: the workflow publishes every event (model deltas, tool/plan/turn
boundaries) on WorkflowStream topic ``acp``. The bridge subscribes from its
per-session offset — the durable log IS the replay source (no checkpointer).
Reconnect/load_session rewinds to 0 and the bridge re-sends everything as ACP
session/update notifications; the client renders the replay like live output.

Runs wherever the host points it (typically beside the worker): egress needs
the Temporal frontend only; ingress is one TCP port behind whatever network
boundary the host runs (tailnet, VPN, localhost).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import uuid
from typing import Any

import temporalio.client as tc
from acp.core import AgentSideConnection  # low-level; acp.run_agent is stdio-only
from acp.helpers import text_block, tool_content, update_plan
from acp.interfaces import Client
from acp.schema import (
    AgentCapabilities,
    AgentMessageChunk,
    AgentThoughtChunk,
    Implementation,
    InitializeResponse,
    ListSessionsResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PlanEntry,
    PromptResponse,
    SessionInfo,
    ToolCallProgress,
    ToolCallStart,
)
from temporalio.contrib.workflow_streams import WorkflowStreamClient

from ._config import configure as _configure
from ._config import get_default_spec, get_tool_builder
from .session import (
    STREAM_TOPIC,
    AgentSession,
    PromptIn,
    SessionStart,
    session_workflow_id,
)

BRIDGE_PORT = int(os.environ.get("ACP_BRIDGE_PORT", "8080"))
TASK_QUEUE = os.environ.get("ACP_BRIDGE_TASK_QUEUE", "durable-agents")
# Replay buffer bound: a loaded session replays at most this many events.
REPLAY_MAX_ITEMS = 5000


def configure(**kwargs: Any) -> None:
    """Re-export of :func:`acp_temporal_bridge.configure` for convenience."""
    _configure(**kwargs)


# ---------------------------------------------------------------------------
# Pure translation: stream-event dict -> ACP update objects. Unit-tested
# without Temporal or a socket (tests/test_bridge_translate.py).
# ---------------------------------------------------------------------------


def _chunk_text(dump: dict[str, Any]) -> str:
    """Extract delta text from a dumped ChatGenerationChunk."""
    out: list[str] = []
    for gen in dump.get("generations", []) or []:
        if not isinstance(gen, dict):
            continue
        text = gen.get("text")
        if text:
            out.append(text)
            continue
        message = gen.get("message") or {}
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            out.append(content)
    return "".join(out)


def _lc_message_text(dump: dict[str, Any]) -> str:
    """Extract text from a langchain-core-serialized message dump.

    Shape: {"lc": 1, "type": "constructor", "id": ["langchain", ..., "AIMessage"],
    "kwargs": {"content": "text" | [{"type": "text", ...}], ...}} — the plugin's
    streaming batches serialize message(-chunk) objects this way.
    """
    kwargs = dump.get("kwargs")
    if not isinstance(kwargs, dict):
        return ""
    content = kwargs.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def translate_stream_item(item: Any) -> list[Any]:
    """Map one WorkflowStreamItem (or its raw data) to 0..n ACP updates.

    Returns [] for anything unrecognizable — unknown events must never break
    the bridge (forward compatibility with newer workers).
    """
    data = getattr(item, "data", item)
    if not isinstance(data, dict):
        return []
    # Plugin model deltas: dumped ChatGenerationChunk (recognizable by the
    # "generations" key) or a serialized LC message(-chunk) dump ("lc"/"kwargs").
    if "generations" in data:
        text = _chunk_text(data)
        return (
            [AgentMessageChunk(session_update="agent_message_chunk", content=text_block(text))]
            if text
            else []
        )
    if "lc" in data and isinstance(data.get("kwargs"), dict):
        text = _lc_message_text(data)
        return (
            [AgentMessageChunk(session_update="agent_message_chunk", content=text_block(text))]
            if text
            else []
        )
    kind = data.get("kind")
    if kind in ("message", "thought"):
        text = data.get("text", "")
        if not text:
            return []
        if kind == "message":
            return [
                AgentMessageChunk(
                    session_update="agent_message_chunk", content=text_block(text)
                )
            ]
        return [
            AgentThoughtChunk(
                session_update="agent_thought_chunk", content=text_block(text)
            )
        ]
    if kind == "tool_start":
        return [
            ToolCallStart(
                session_update="tool_call",
                tool_call_id=data.get("id") or str(uuid.uuid4()),
                title=str(data.get("name", "tool")),
                kind="execute",
                status="in_progress",
            )
        ]
    if kind == "tool_end":
        output = str(data.get("output", ""))
        return [
            ToolCallProgress(
                session_update="tool_call_update",
                tool_call_id=data.get("id") or "",
                status="completed",
                content=[tool_content(text_block(output))] if output else [],
            )
        ]
    if kind == "plan":
        entries = [
            PlanEntry(
                content=str(todo.get("content", "")),
                priority=todo.get("priority", "medium"),
                status=todo.get("status", "pending"),
            )
            for todo in data.get("todos", [])
            if isinstance(todo, dict) and todo.get("content")
        ]
        return [update_plan(entries)] if entries else []
    # turn_end and unknown kinds: turn_end is consumed by prompt(), not sent.
    return []


# ---------------------------------------------------------------------------
# BridgeAgent: the ACP Agent implementation (one per TCP connection).
# ---------------------------------------------------------------------------


class BridgeAgent:
    """Implements the ACP Agent protocol against Temporal.

    Per-connection state: ACP client handle + in-memory stream offsets per
    session (workflow-id -> last-replayed offset). Stateless across
    connections by design; the durable state lives in the workflows.
    """

    def __init__(self, client: Client, temporal: tc.Client) -> None:
        self._client = client
        self._temporal = temporal
        # ACP notifications carry no response, so prompt() owns the offset
        # bookkeeping for its session; cancel needs the same map read-only.
        self._offsets: dict[str, int] = {}

    # -- ACP lifecycle -------------------------------------------------------

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: Any = None,
        client_info: Any = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        return InitializeResponse(
            protocol_version=protocol_version,
            agent_capabilities=AgentCapabilities(load_session=True),
            agent_info=Implementation(name="acp-temporal-bridge", version="0.1.0"),
        )

    async def new_session(
        self,
        cwd: str,
        additional_directories: Any = None,
        mcp_servers: Any = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        session_id = uuid.uuid4().hex[:12]
        start = SessionStart(run=get_default_spec(), session_id=session_id)
        # All three run-args must be passed on start: temporalio drops run
        # type hints entirely when payload count != signature length
        # (_convert_payloads), which silently decodes SessionStart as a dict.
        await self._temporal.start_workflow(
            AgentSession.run,
            args=[start, None, None],  # stream_state/transcript: fresh run
            id=session_workflow_id(session_id),
            task_queue=TASK_QUEUE,
        )
        self._offsets[session_id] = 0
        return NewSessionResponse(session_id=session_id)

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        additional_directories: Any = None,
        mcp_servers: Any = None,
        **kwargs: Any,
    ) -> LoadSessionResponse:
        handle = self._temporal.get_workflow_handle(session_workflow_id(session_id))
        await handle.describe()  # raises if gone
        # Replay the whole durable log on the next prompt/attach.
        self._offsets[session_id] = 0
        return LoadSessionResponse(session_id=session_id)

    async def list_sessions(self, cursor: Any = None, **kwargs: Any) -> ListSessionsResponse:
        sessions: list[SessionInfo] = []
        prefix = session_workflow_id("")
        async for wf in self._temporal.list_workflows(
            f"WorkflowId STARTS WITH {prefix}",
        ):
            sessions.append(SessionInfo(session_id=wf.id.removeprefix(prefix)))
            if len(sessions) >= 50:
                break
        return ListSessionsResponse(sessions=sessions)

    async def authenticate(self, method: str, **kwargs: Any) -> None:
        raise RuntimeError("no auth methods supported (network boundary is the gate)")

    async def set_session_mode(self, session_id: str, mode_id: str, **kwargs: Any) -> None:
        raise RuntimeError("session modes not supported")

    async def set_config_option(self, session_id: str, name: str, value: Any, **kwargs: Any) -> None:
        raise RuntimeError("config options not supported")

    # -- prompt / cancel -----------------------------------------------------

    async def prompt(self, session_id: str, prompt: list[Any], **kwargs: Any) -> PromptResponse:
        text = "".join(
            block.text
            for block in prompt
            if getattr(block, "type", None) == "text" and getattr(block, "text", "")
        )
        handle = self._temporal.get_workflow_handle(session_workflow_id(session_id))
        await handle.execute_update(AgentSession.prompt, PromptIn(text=text))
        offset = self._offsets.get(session_id, 0)
        stop_reason = "end_turn"
        stream = WorkflowStreamClient.create(
            self._temporal, session_workflow_id(session_id)
        )
        count = 0
        async for item in stream.subscribe(STREAM_TOPIC, from_offset=offset):
            count += 1
            if count > REPLAY_MAX_ITEMS:
                break
            data = getattr(item, "data", None)
            if isinstance(data, dict) and data.get("kind") == "turn_end":
                stop_reason = data.get("stop_reason") or "end_turn"
                self._offsets[session_id] = getattr(item, "offset", offset) + 1
                break
            for update in translate_stream_item(item):
                await self._client.session_update(session_id, update)
            self._offsets[session_id] = getattr(item, "offset", offset) + 1
        return PromptResponse(stop_reason=stop_reason)

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        """ACP notification: client asked to cancel the in-flight turn."""
        handle = self._temporal.get_workflow_handle(session_workflow_id(session_id))
        # Fire-and-forget semantics: prompt() sees turn_end(stop_reason=
        # "cancelled") on the stream and returns.
        asyncio.create_task(handle.execute_update(AgentSession.cancel))


# ---------------------------------------------------------------------------
# TCP serve loop (newline-delimited JSON-RPC frames — the SDK's stdio framing
# works unchanged over TCP StreamReader/StreamWriter pairs).
# ---------------------------------------------------------------------------


async def _handle_connection(
    temporal: tc.Client, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    conn = AgentSideConnection(
        lambda client: BridgeAgent(client, temporal), writer, reader
    )
    try:
        await conn.listen()
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except (ConnectionError, OSError):
            pass


async def serve() -> None:
    temporal = await tc.Client.connect(
        os.environ.get("TEMPORAL_HOST", "localhost:7233"),
        namespace=os.environ.get("TEMPORAL_NAMESPACE", "default"),
    )
    get_tool_builder()  # fail fast if the host forgot configure()
    server = await asyncio.start_server(
        lambda r, w: _handle_connection(temporal, r, w),
        host=os.environ.get("ACP_BRIDGE_BIND", "0.0.0.0"),
        port=BRIDGE_PORT,
    )
    async with server:
        # serve_forever never returns normally; suppress the shutdown noise.
        with contextlib.suppress(asyncio.CancelledError):
            await server.serve_forever()


def main() -> None:
    asyncio.run(serve())


if __name__ == "__main__":
    main()

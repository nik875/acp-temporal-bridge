"""ACP bridge: TCP server mapping ACP client sessions onto durable
AgentSession workflows.

    ACP client --tcp/NDJSON-RPC--> BridgeAgent --Temporal--> AgentSession

Mapping (session_id == Temporal workflow_id minus the configured prefix):
    initialize          -> capabilities (load_session supported)
    new_session         -> start_workflow(AgentSession, id="<prefix><sid>")
    load_session        -> verify workflow alive; render transcript history
                           as message chunks, then tail the stream live from
                           the head (in-flight turns + pending gates)
    list_sessions       -> Temporal workflow-id query (prefix match)
    prompt              -> execute_update(prompt) then stream replay until
                           turn_end; returns PromptResponse(stop_reason)
    cancel (notif)      -> execute_update(cancel) — in-flight turn cancelled
    pr merged (signal)  -> session wakes with a system prompt (watcher.py)

Streaming: the workflow publishes every event (model deltas, tool/plan/turn
boundaries) on WorkflowStream topic ``acp``. The bridge subscribes from its
per-session offset — the durable log IS the replay source (no checkpointer).
Reconnect/load_session renders the durable transcript (user + assistant
messages) to the client immediately, then tails the stream from the head in
the background, so a reattached client sees history AND a turn that was
mid-flight when the connection dropped.

Runs wherever the host points it (typically beside the worker): egress needs
the Temporal frontend only; ingress is one TCP port behind whatever network
boundary the host runs (tailnet, VPN, localhost).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import uuid
from typing import Any, get_args

import temporalio.client as tc
from acp.core import AgentSideConnection  # low-level; acp.run_agent is stdio-only
from acp.exceptions import RequestError
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
    PermissionOption,
    PlanEntry,
    PromptResponse,
    SessionInfo,
    StopReason,
    ToolCallProgress,
    ToolCallStart,
    ToolCallUpdate,
    UserMessageChunk,
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


def _bridge_version() -> str:
    """Real package version; the hardcoded string was forever stale."""
    try:
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as _v

        return _v("acp-temporal-bridge")
    except PackageNotFoundError:  # pragma: no cover - odd installs
        return "0.0.0"
TASK_QUEUE = os.environ.get("ACP_BRIDGE_TASK_QUEUE", "durable-agents")
# Replay buffer bound: a loaded session replays at most this many events.
REPLAY_MAX_ITEMS = 5000
# History render bound: a load_session replays at most this many transcript
# messages (oldest are dropped — the newest turns matter most on reattach).
REPLAY_TRANSCRIPT_MESSAGES = 200
# ACP stop reasons are a closed vocabulary; anything else on the durable
# turn_end (e.g. "error") is surfaced as a prompt failure, never silently
# mapped onto a fake success.
ACP_STOP_REASONS = frozenset(get_args(StopReason))


class ReplayOverflowError(RuntimeError):
    """A turn's stream produced more items than REPLAY_MAX_ITEMS without a
    turn_end.

    prompt() must NEVER return a fabricated successful completion in this
    case: the turn is either still running or the durable log is degenerate,
    and a bogus ``end_turn`` would lie to the client (observed in the field
    as a 10k-item flood that silently "completed" a dead turn).
    """


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


def _transcript_message(dump: dict[str, Any]) -> tuple[str, str]:
    """(role, text) from one durable-transcript message dump.

    The transcript is langchain's ``message_to_dict`` form:
    ``{"type": "human"|"ai"|"tool"|"system", "data": {"content": ...}}``
    (see session._serialize). Unknown shapes yield ("", "") and are skipped.
    """
    if not isinstance(dump, dict):
        return "", ""
    role = str(dump.get("type") or "")
    data = dump.get("data")
    content = data.get("content") if isinstance(data, dict) else None
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    else:
        text = ""
    return role, text


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
        # Per-session background tail tasks spawned by load_session: they
        # stream an in-flight turn to the reattached client until turn_end.
        self._tail_tasks: dict[str, asyncio.Task[None]] = {}

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
            agent_info=Implementation(
                name="acp-temporal-bridge",
                version=_bridge_version(),
            ),
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
        # A reattaching client (crash, pod roll, client restart) must see the
        # conversation NOW, not on its next prompt: until session/updates
        # arrive, the client renders an empty "start chatting with this
        # agent" session. Two replay sources, split to avoid double-render:
        #   1. the durable transcript (human + AI text, CAN-surviving)
        #      renders the full history as message chunks, then
        #   2. the stream log is tailed LIVE from the head — covers an
        #      in-flight turn (pod died mid-prompt; the turn kept running
        #      in Temporal) and any pending approval gate.
        # Offsets then point at the head, so the next prompt() replays
        # nothing old.
        await self._replay_transcript(session_id, handle)
        head = await self._stream_head(handle)
        self._offsets[session_id] = head
        self._spawn_tail_task(session_id, handle, head)
        return LoadSessionResponse(session_id=session_id)

    async def _replay_transcript(
        self, session_id: str, handle: tc.WorkflowHandle
    ) -> None:
        """Render durable transcript history as ACP message-chunk updates."""
        try:
            tail = await handle.query(
                AgentSession.transcript_tail, REPLAY_TRANSCRIPT_MESSAGES
            )
        except tc.RPCError:
            return  # gone / query timeout: nothing to render
        for dump in (tail or {}).get("transcript", []) or []:
            role, text = _transcript_message(dump)
            if not text:
                continue
            if role == "human":
                update: Any = UserMessageChunk(
                    session_update="user_message_chunk",
                    content=text_block(text),
                )
            elif role == "ai":
                update = AgentMessageChunk(
                    session_update="agent_message_chunk",
                    content=text_block(text),
                )
            else:
                continue  # tool/system messages: not client-renderable history
            await self._client.session_update(session_id, update)

    async def _stream_head(self, handle: tc.WorkflowHandle) -> int:
        try:
            return await handle.query(AgentSession.stream_head)
        except tc.RPCError:
            return 0

    def _spawn_tail_task(
        self, session_id: str, handle: tc.WorkflowHandle, from_offset: int
    ) -> None:
        """Tail the stream log live from `from_offset` in the background.

        One task per session per connection; a re-spawn cancels any
        previous tail (a second load_session means a fresh reattach).
        """
        existing = self._tail_tasks.get(session_id)
        if existing is not None and not existing.done():
            existing.cancel()
        self._tail_tasks[session_id] = asyncio.create_task(
            self._tail_stream(session_id, handle, from_offset)
        )

    async def _tail_stream(
        self, session_id: str, handle: tc.WorkflowHandle, offset: int
    ) -> None:
        stream = WorkflowStreamClient.create(
            self._temporal, session_workflow_id(session_id)
        )
        count = 0
        async for item in stream.subscribe(STREAM_TOPIC, from_offset=offset):
            count += 1
            if count > REPLAY_MAX_ITEMS:
                # Degenerate/flooded tail: give up quietly. The next
                # prompt() replays from the tracked offset and its own
                # overflow guard raises honestly.
                break
            data = getattr(item, "data", None)
            if isinstance(data, dict) and data.get("kind") == "turn_end":
                self._offsets[session_id] = getattr(item, "offset", offset) + 1
                return
            if isinstance(data, dict) and data.get("kind") == "approval_request":
                # A gate that was pending when the client dropped: it must
                # be re-asked now that the client is back.
                await self._resolve_gate(session_id, handle, data)
                self._offsets[session_id] = getattr(item, "offset", offset) + 1
                continue
            for update in translate_stream_item(item):
                await self._client.session_update(session_id, update)
            self._offsets[session_id] = getattr(item, "offset", offset) + 1

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
        # A load_session tail may still be streaming an in-flight turn to
        # this client; drain it (through turn_end) before subscribing, or
        # both loops would send the same items twice.
        tail = self._tail_tasks.get(session_id)
        if tail is not None and not tail.done():
            await tail
        offset = self._offsets.get(session_id, 0)
        stop_reason = "end_turn"
        stream = WorkflowStreamClient.create(
            self._temporal, session_workflow_id(session_id)
        )
        count = 0
        halting = False
        async for item in stream.subscribe(STREAM_TOPIC, from_offset=offset):
            count += 1
            data = getattr(item, "data", None)
            if count > REPLAY_MAX_ITEMS and not halting:
                # Emergency brake (v0.1.8): stopping this client's view is
                # NOT enough — the turn would keep streaming chunks into
                # durable history (SSD burn) while nobody watches. Halt the
                # turn workflow-side (the same update path as the client's
                # cancel notification: the model activity is cancelled and
                # the run loop records turn_end stop_reason="cancelled"),
                # then keep consuming THIS stream silently through the
                # halted turn's turn_end so the next prompt replays from
                # the post-turn boundary instead of tripping over a stale
                # end. The session stays alive; the user's next prompt
                # resumes it. Never fake a completion: the error below
                # still surfaces.
                halting = True
                await handle.execute_update(AgentSession.cancel)
                continue
            if halting:
                # Drain: advance offsets, surface nothing.
                self._offsets[session_id] = getattr(item, "offset", offset) + 1
                if isinstance(data, dict) and data.get("kind") == "turn_end":
                    break
                continue
            if isinstance(data, dict) and data.get("kind") == "turn_end":
                reason = data.get("stop_reason") or "end_turn"
                if reason not in ACP_STOP_REASONS:
                    # e.g. "error" from a failed turn: the durable record is
                    # honest, the ACP surface is an error (StopReason is a
                    # closed Literal — pydantic would reject "error").
                    raise RequestError.internal_error(                        {
                            "details": (
                                f"session {session_id}: turn ended with "
                                f"stop_reason={reason!r}"
                            ),
                            "stop_reason": reason,
                        }
                    )
                stop_reason = reason
                self._offsets[session_id] = getattr(item, "offset", offset) + 1
                break
            if isinstance(data, dict) and data.get("kind") == "approval_request":
                await self._resolve_gate(session_id, handle, data)
                self._offsets[session_id] = getattr(item, "offset", offset) + 1
                continue
            for update in translate_stream_item(item):
                await self._client.session_update(session_id, update)
            self._offsets[session_id] = getattr(item, "offset", offset) + 1
        if halting:
            # The halt drain ended without the halted turn's turn_end
            # (degenerate log even after cancel). Never report a fake
            # completion — surface the anomaly.
            raise ReplayOverflowError(
                f"session {session_id}: turn halted after {REPLAY_MAX_ITEMS} "
                f"stream items (from offset {offset}) but no turn_end "
                "followed the cancel — refusing to report end_turn"
            )
        return PromptResponse(stop_reason=stop_reason)

    async def _resolve_gate(
        self, session_id: str, handle: tc.WorkflowHandle, gate: dict[str, Any]
    ) -> None:
        """Surface a durable HITL gate as an ACP request_permission round-trip.

        Only asks the client when the workflow is actually paused on THIS
        gate (pending_approval query): on replay, an already-answered
        approval_request is skipped because the query returns None or a
        different (newer) gate.
        """
        try:
            pending = await handle.query(AgentSession.pending_approval)
        except tc.RPCError:
            pending = None  # workflow gone / query deadline: treat as answered
        same_gate = (
            pending is not None
            and pending.tool == gate.get("tool")
            and dict(pending.args or {}) == dict(gate.get("args") or {})
        )
        if not same_gate:
            return

        args = gate.get("args") or {}
        tool_call = ToolCallUpdate(
            tool_call_id=f"approval-{uuid.uuid4().hex[:12]}",
            title=f"{gate.get('tool') or 'tool'} {json.dumps(args, default=str)[:120]}",
            kind="execute",
            raw_input=args,
        )
        options = [
            PermissionOption(kind="allow_once", name="Allow", option_id="approve"),
            PermissionOption(kind="reject_once", name="Deny", option_id="respond"),
        ]
        resp = await self._client.request_permission(options, session_id, tool_call)
        out = resp.outcome
        if getattr(out, "outcome", "") == "selected" and getattr(out, "option_id", "") == "approve":
            payload = {"action": "approve", "response": None}
        elif getattr(out, "outcome", "") == "selected":
            payload = {"action": "respond", "response": f"denied via option {out.option_id}"}
        else:  # cancelled / closed request
            payload = {"action": "respond", "response": "denied: permission request cancelled"}
        await handle.execute_update(AgentSession.resolve_approval, payload)

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
    # listening=False: the constructor already starts the receive loop when
    # listening defaults True; calling listen() afterwards would run a SECOND
    # read loop on the same reader ("readuntil() called while another
    # coroutine is already waiting for incoming data" on the first prompt).
    # Same pattern as the SDK's own stdio runner.
    conn = AgentSideConnection(
        lambda client: BridgeAgent(client, temporal), writer, reader,
        listening=False,
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

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
import logging
import os
import time
import uuid
from datetime import timedelta
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
from temporalio.common import WorkflowIDReusePolicy
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
# Emergency brake against a flooding turn (the 2026-10-26 flood published
# 16,401 items in one turn). Raised 5k → 50k now that the flood sources are
# fixed upstream (no-op filter + tool-arg coalescing at the model seam):
# legitimate tool-dense turns were landing uncomfortably close to 2,292
# kept items, and this guard halts REAL turns when it fires.
# REMOVAL PLAN: this guard is scaffolding for the flood era — once a few
# weeks of production traffic show the flooding has actually stopped
# (stream items per turn staying well under this bound), delete it
# entirely; it exists to bound blast radius, not to be load-bearing.
REPLAY_MAX_ITEMS = 50000
# Interrupt cancel re-driver: session/cancel kills the turn cooperatively
# (AgentSession.cancel update -> turn task cancelled -> turn_end
# stop_reason="cancelled") and the durable session ALWAYS survives —
# terminating the workflow is forbidden (the 2026-10-28 incident: a v0.1.15
# watchdog escalated an interrupt to a server-side terminate during a
# worker Recreate rollout, destroying a 1211-event transcript). But an
# update only takes effect if a worker is alive to process it, and the
# deployments are Recreate — a rollout gap has NO worker. So: if the turn
# is still active CANCEL_RETRY_AFTER_SECS after the first update went out,
# RE-SEND the cancel update with backoff until it lands. Durable? The
# re-driver task lives in the bridge process — if the BRIDGE dies, paseo's
# client-cancel notification dies with the connection, and any reconnect
# re-drives the cancel (load_session shows the still-running turn). The
# invariant "interrupt kills only the turn" holds because the update, once
# delivered, is consumed by the workflow itself.
#
# Retries are SCOPED to the turn being cancelled: cancel() snapshots the
# session's turn_generation before the first update, and the re-driver
# re-sends only while the generation still matches. Without the scope a
# fast user re-prompt after the interrupt looks like "turn still active"
# and the retry ghost-cancels the NEW turn (found in live e2e 2026-10-28:
# a prompt 2s after an interrupt returned cancelled). An unknown
# generation (workflow too old for the query / query failure) degrades to
# v0.1.17 behavior: retry until turn_active reports False.
CANCEL_RETRY_AFTER_SECS = float(
    os.environ.get("ACP_BRIDGE_CANCEL_RETRY_AFTER_SECS", "5")
)
CANCEL_RETRY_BACKOFF_SECS = float(
    os.environ.get("ACP_BRIDGE_CANCEL_RETRY_BACKOFF_SECS", "5")
)
CANCEL_RETRY_MAX_SECS = float(
    os.environ.get("ACP_BRIDGE_CANCEL_RETRY_MAX_SECS", "300")
)
# UI-liveness heartbeat pacing (the workflow-side gate lives in session.py):
# the bridge heartbeats every attached session this often. Kept well under
# UI_STALE_SECS so a single lost beat never trips the gate.
UI_HEARTBEAT_INTERVAL_SECS = float(
    os.environ.get("ACP_BRIDGE_UI_HEARTBEAT_INTERVAL_SECS", "10")
)
# Workflow TASK timeout (set at workflow start; per-activation, not per-run).
# The 10s default fails a cold-cache session whose multi-MB history replays
# in 25-28s — see the session_workflow start path comment for the wedge chain.
# Env-overridable.
SESSION_TASK_TIMEOUT_SECS = float(
    os.environ.get("ACP_BRIDGE_SESSION_TASK_TIMEOUT_SECS", "120")
)


def _consume_update_exception(task: asyncio.Task[None]) -> None:
    """Done-callback for fire-and-forget update tasks.

    Retrieves the exception (suppressing it) so asyncio never logs
    "exception was never retrieved" — the cancel re-driver, not the
    update task, owns cancel retry/observability on a dead workflow.
    """
    with contextlib.suppress(Exception):
        task.result()


def _is_closed_workflow(exc: BaseException) -> bool:
    """True when an RPC failed because the target workflow is already closed.

    Temporal surfaces this as an RPC error whose message (possibly on a
    cause) is "workflow execution already completed". A session workflow
    closes three ways in production: a client-cancel raced the turn,
    idle expiry completed it, or a recreate replaced it. The bridge must
    treat the closed workflow as recoverable (recreate, v0.1.16), not a
    fatal protocol error — before this, every prompt to a closed session
    bounced off it forever with ACP internal_error -32603, bricking the
    session in Paseo until the pod restarted.
    """
    seen: set[int] = set()
    stack: list[BaseException | None] = [exc]
    while stack:
        cur = stack.pop()
        if cur is None or id(cur) in seen:
            continue
        seen.add(id(cur))
        if "workflow execution already completed" in str(cur).lower():
            return True
        stack.append(cur.__cause__)
        stack.append(cur.__context__)
    return False
# History render bound: a load_session replays at most this many transcript
# messages (oldest are dropped — the newest turns matter most on reattach).
REPLAY_TRANSCRIPT_MESSAGES = 200
# ACP stop reasons are a closed vocabulary; anything else on the durable
# turn_end (e.g. "error") is surfaced as a prompt failure, never silently
# mapped onto a fake success.
ACP_STOP_REASONS = frozenset(get_args(StopReason))

log = logging.getLogger(__name__)


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
        # Sessions this connection created / actually prompted: aclose()
        # cancels the created-but-never-prompted ones (clients — paseo
        # probes, reconnecting UIs — spawn sessions they never use, and
        # without this the workflow waits on its prompt signal forever).
        self._created: set[str] = set()
        self._prompted: set[str] = set()
        # Per-session background tail tasks spawned by load_session: they
        # stream an in-flight turn to the reattached client until turn_end.
        self._tail_tasks: dict[str, asyncio.Task[None]] = {}
        # Cancel re-drivers (cancel()): re-send the cancel update until the
        # turn reports stopped — no terminate path exists; the session
        # always survives an interrupt.
        self._cancel_watchdogs: set[asyncio.Task[None]] = set()
        # UI-liveness heartbeats (session.py's ui-gate): one task per
        # attached session, sending ui_heartbeat(epoch) on a cadence so the
        # workflow knows a live client is streaming-relevant. _ui_epochs is
        # the per-session epoch (time_ns: monotonic across bridge restarts,
        # so a RECONNECTED bridge's beats outrank a dead connection's).
        self._ui_tasks: dict[str, asyncio.Task[None]] = {}
        self._ui_epochs: dict[str, int] = {}

    async def aclose(self) -> None:
        """Shut down per-connection state; called when the ACP client drops.

        Cancels every live tail task — without this they keep tailing the
        workflow stream and pushing updates into a dead connection. Also
        cancels sessions this connection created but never prompted: they
        hold nothing (no transcript, no turn), so cancelling is lossless;
        prompted sessions survive (the client may reconnect and reattach —
        that is the durability promise).
        """
        for task in self._tail_tasks.values():
            task.cancel()
        if self._tail_tasks:
            await asyncio.gather(*self._tail_tasks.values(), return_exceptions=True)
        self._tail_tasks.clear()
        # Cancel re-drivers die with the connection: on disconnect the
        # durability contract wins (the client may reconnect and reattach
        # to the session — which still exists, terminate being gone).
        for task in self._cancel_watchdogs:
            task.cancel()
        if self._cancel_watchdogs:
            await asyncio.gather(*self._cancel_watchdogs, return_exceptions=True)
        self._cancel_watchdogs.clear()
        # UI heartbeats die with the connection, and the workflow is told:
        # ui_detached pauses any in-flight turn durably (its next model call
        # gates) until a reconnecting client's heartbeat re-arms the gate.
        # Tasks are stopped FIRST — a stray heartbeat landing after the
        # detach signal would silently un-detach the session.
        for task in self._ui_tasks.values():
            task.cancel()
        if self._ui_tasks:
            await asyncio.gather(*self._ui_tasks.values(), return_exceptions=True)
        self._ui_tasks.clear()
        for session_id, epoch in self._ui_epochs.items():
            with contextlib.suppress(tc.RPCError):
                await self._temporal.get_workflow_handle(
                    session_workflow_id(session_id)
                ).signal(AgentSession.ui_detached, epoch)
        self._ui_epochs.clear()
        for session_id in self._created - self._prompted:
            handle = self._temporal.get_workflow_handle(
                session_workflow_id(session_id)
            )
            # Already gone (completed/expired/cancelled) is fine.
            with contextlib.suppress(tc.RPCError):
                await handle.cancel()

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
            # Long-lived sessions (dozens of turns) reach multi-MB histories; a
            # worker-cache miss then forces a FULL history replay in one
            # workflow task. At 2MB+ that replay takes 25-28s and blows the 10s
            # default task timeout — the task times out, re-replays, times out
            # again (TMPRL1104 loop), and NO update/query/signal (including a
            # mid-turn steer's cancel) is processed until the workflow is
            # cached again: the turn wedges, Paseo's interrupt-and-replace
            # fallback races the eventual completion, and every later prompt
            # dies with "A foreground turn is already active" (observed
            # 2026-10-28, session sysadmin-76dd64a395ac: the cancel was
            # accepted ~18s after send). 2 min covers a 10MB replay with
            # headroom; the history-budget CAN in session.py keeps replays
            # small in the first place.
            task_timeout=timedelta(seconds=SESSION_TASK_TIMEOUT_SECS),
        )
        self._offsets[session_id] = 0
        self._created.add(session_id)
        # Arm the UI gate from the first moment of the session's life: the
        # first heartbeat lands before any prompt can possibly run.
        self._ensure_ui_heartbeat(session_id)
        return NewSessionResponse(session_id=session_id)

    async def _recreate_closed_session(self, session_id: str) -> tc.WorkflowHandle:
        """Start a fresh AgentSession under the SAME workflow id.

        For sessions whose workflow closed out from under the client
        (idle expiry, a lost cancel race). The ACP session
        id is client-visible state — Paseo keeps using it — so the
        replacement must land on the same id, not a new one. ALLOW_DUPLICATE
        (not FAILED_ONLY) because idle expiry COMPLETES the workflow
        cleanly and must still be recreatable; the only window where a
        live workflow could be duplicated is a lost-update race that the
        "already completed" error already rules out.

        Resets replay bookkeeping for the new run: offsets go to 0 (the
        new workflow's stream starts fresh) and any load_session tail
        task is cancelled (it was subscribed to the dead workflow).
        """
        start = SessionStart(run=get_default_spec(), session_id=session_id)
        await self._temporal.start_workflow(
            AgentSession.run,
            args=[start, None, None],  # stream_state/transcript: fresh run
            id=session_workflow_id(session_id),
            task_queue=TASK_QUEUE,
            id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE,
            task_timeout=timedelta(seconds=SESSION_TASK_TIMEOUT_SECS),
        )
        self._offsets[session_id] = 0
        tail = self._tail_tasks.pop(session_id, None)
        if tail is not None and not tail.done():
            tail.cancel()
        # The fresh run has no UI state; arm the gate immediately so the
        # prompt that triggered the recreate is covered.
        self._ensure_ui_heartbeat(session_id)
        return self._temporal.get_workflow_handle(session_workflow_id(session_id))

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        additional_directories: Any = None,
        mcp_servers: Any = None,
        **kwargs: Any,
    ) -> LoadSessionResponse:
        handle = self._temporal.get_workflow_handle(session_workflow_id(session_id))
        try:
            await handle.describe()
        except tc.RPCError:
            # Session workflow is gone (idle expiry, a cancel race, or a
            # recreate that lost the race). A reattaching client must still
            # get a usable session: return with an EMPTY replay — the client
            # renders a fresh session, and the next prompt() recreates the
            # workflow under the same id (see _recreate_closed_session).
            # Before v0.1.16 this raised, and Paseo showed the bricked
            # session as permanently unloadable.
            self._offsets[session_id] = 0
            return LoadSessionResponse(session_id=session_id)
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
        # Reattach = live client: (re)arm the heartbeat with a fresh epoch.
        self._ensure_ui_heartbeat(session_id)
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

    def _ensure_ui_heartbeat(self, session_id: str) -> None:
        """Start (or restart) the UI-liveness heartbeat for a session.

        A fresh epoch per attachment (time_ns: strictly increasing, so a
        newer connection's beats always outrank a stale one's — and a dead
        connection's late detach is ignored by the epoch check). Called on
        every path that (re)attaches a client to a workflow: new_session,
        load_session (alive workflow), _recreate_closed_session, and
        prompt() defensively.
        """
        existing = self._ui_tasks.get(session_id)
        if existing is not None and not existing.done():
            existing.cancel()
        epoch = time.time_ns()
        self._ui_epochs[session_id] = epoch
        self._ui_tasks[session_id] = asyncio.create_task(
            self._ui_heartbeat_loop(session_id, epoch)
        )

    async def _ui_heartbeat_loop(self, session_id: str, epoch: int) -> None:
        """Send ui_heartbeat(epoch) every UI_HEARTBEAT_INTERVAL_SECS.

        The first beat goes out immediately — it is what (re)arms the
        workflow's gate on attach. RPC errors are suppressed (the workflow
        may be mid-CAN, the cluster busy); a closed workflow ends the loop
        (nothing to heartbeat; a future prompt recreates + re-arms).
        """
        handle = self._temporal.get_workflow_handle(session_workflow_id(session_id))
        while True:
            try:
                await handle.signal(AgentSession.ui_heartbeat, epoch)
            except tc.RPCError as e:
                if _is_closed_workflow(e):
                    return  # workflow gone: nothing left to heartbeat
                # transient (worker gap, busy cluster): keep beating
            await asyncio.sleep(UI_HEARTBEAT_INTERVAL_SECS)

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
        self._prompted.add(session_id)
        try:
            await handle.execute_update(AgentSession.prompt, PromptIn(text=text))
        except tc.RPCError as e:
            if not _is_closed_workflow(e):
                raise
            # The session workflow is closed (idle expiry, cancel race).
            # Recreate it under the same id and re-send the prompt — the
            # client only ever sees a working session, never the -32603
            # brick.
            handle = await self._recreate_closed_session(session_id)
            await handle.execute_update(AgentSession.prompt, PromptIn(text=text))
        # Defensive re-arm: if the heartbeat loop died with a previously
        # closed workflow (recreate above), a fresh one must exist BEFORE
        # this turn starts — once the gate is armed, a turn must never run
        # headless just because no beat was flowing.
        self._ensure_ui_heartbeat(session_id)
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
        """ACP notification: client asked to cancel the in-flight turn.

        Cooperative-only, with a durable re-driver: the cancel update is
        re-sent with backoff until the turn reports inactive — scoped to
        THIS turn (generation snapshot below), so a fast re-prompt after
        the interrupt can never be ghost-cancelled by a retry. The session
        workflow is NEVER terminated — an interrupt kills only the turn;
        the session and transcript always survive (2026-10-28 decision).
        A worker gap (Recreate rollout) just delays the retry until a
        worker returns — exactly the case the old terminate escalation
        mishandled.
        """
        handle = self._temporal.get_workflow_handle(session_workflow_id(session_id))
        # Snapshot the turn identity BEFORE the first update: the re-driver
        # re-sends only while this turn is still the live one. Best-effort —
        # a too-old workflow (or a query hiccup) yields None and the
        # re-driver falls back to the unscoped turn_active semantics.
        generation: int | None = None
        try:
            generation = await handle.query(
                AgentSession.turn_generation, rpc_timeout=timedelta(seconds=5)
            )
        except Exception:  # noqa: BLE001 — unknown generation is a fallback, not an error
            log.debug("turn_generation query failed for %s (unscoped retry)", session_id)
        # Fire-and-forget semantics: prompt() sees turn_end(stop_reason=
        # "cancelled") on the stream and returns. The task's exception (if
        # any) is consumed by the done-callback — the re-driver below owns
        # retrying, and a closed workflow (idle expiry won the race) must
        # not spray "exception was never retrieved" warnings.
        update_task = asyncio.create_task(handle.execute_update(AgentSession.cancel))
        update_task.add_done_callback(_consume_update_exception)
        redriver = asyncio.create_task(self._cancel_redriver(handle, generation))
        self._cancel_watchdogs.add(redriver)
        redriver.add_done_callback(self._cancel_watchdogs.discard)

    async def _cancel_redriver(
        self, handle: tc.WorkflowHandle, generation: int | None
    ) -> None:
        """Re-send the cancel update until the interrupted turn is stopped.

        No terminate exists on this path — the session outlives every
        interrupt. Stop conditions: the turn reports inactive; the
        workflow closed (idle expiry — turn ended with it); or the turn
        generation moved on, meaning the interrupted turn ALREADY ended
        and a newer turn exists — re-sending now would ghost-cancel the
        user's fresh turn, so stop. Query failures (worker gap, wedged
        query) are "unknown": keep retrying. A Recreate rollout gap just
        delays the retry until a worker returns. Bounded total wait so a
        genuinely stuck loop gives up visibly instead of leaking a task
        forever; the turn itself remains durably cancellable by any later
        cancel.
        """
        delay = CANCEL_RETRY_BACKOFF_SECS
        deadline = asyncio.get_running_loop().time() + CANCEL_RETRY_MAX_SECS
        while True:
            await asyncio.sleep(min(CANCEL_RETRY_AFTER_SECS, delay))
            alive: bool | None = None
            try:
                alive = await handle.query(
                    AgentSession.turn_active, rpc_timeout=timedelta(seconds=5)
                )
            except tc.RPCError as e:
                if _is_closed_workflow(e):
                    return  # workflow closed (idle expiry): turn ended with it
                alive = None  # worker gap / wedged query: unknown, keep retrying
            except Exception:  # noqa: BLE001 — any query failure is "unknown"
                alive = None
            if alive is False:
                return  # cooperative cancel won: turn ended, session intact
            if generation is not None:
                # Scope check: is the live turn still the one we were asked
                # to cancel? A different generation means the interrupted
                # turn ended and the user moved on — NEVER re-send.
                try:
                    now_gen = await handle.query(
                        AgentSession.turn_generation, rpc_timeout=timedelta(seconds=5)
                    )
                except tc.RPCError as e:
                    if _is_closed_workflow(e):
                        return
                    now_gen = None  # unknown: fall through to the active check
                except Exception:  # noqa: BLE001
                    now_gen = None
                if now_gen is not None and now_gen != generation:
                    return  # interrupted turn is history; newer turn untouchable
            # Still active (same generation), or unknown. Re-send the
            # update: once a worker processes it, the turn dies.
            update_task = asyncio.create_task(handle.execute_update(AgentSession.cancel))
            update_task.add_done_callback(_consume_update_exception)
            if asyncio.get_running_loop().time() >= deadline:
                # Give up on DRIVING the cancel, never on the session: log
                # loudly (the turn remains durably alive and re-cancellable
                # by any later cancel / reconnect).
                log.warning(
                    "cancel re-driver gave up after %.0fs (turn_active=%s) — "
                    "turn remains durably alive; session NOT terminated",
                    CANCEL_RETRY_MAX_SECS,
                    alive,
                )
                return
            delay = min(delay * 2, CANCEL_RETRY_MAX_SECS)


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
    #
    # The SDK's context managers (spawn_client_process et al.) always pair the
    # connection with `await conn.close()` in a finally — so must we. Without
    # it, every connection leaks its supervised sender/dispatcher/handler
    # tasks (pending forever, one set per client reconnect) until GC destroys
    # them mid-flight ("Task was destroyed but it is pending" / "cannot reuse
    # already awaited coroutine"), which can sever a live turn's stream.
    holder: list[BridgeAgent] = []

    def make_agent(client: Client) -> BridgeAgent:
        agent = BridgeAgent(client, temporal)
        holder.append(agent)
        return agent

    conn = AgentSideConnection(make_agent, writer, reader, listening=False)
    try:
        await conn.listen()
    finally:
        if holder:
            await holder[0].aclose()
        await conn.close()
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

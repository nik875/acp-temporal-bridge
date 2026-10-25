"""Interactive session workflow: one durable workflow per ACP session.

The ACP bridge (bridge.py) maps client sessions onto THIS workflow:

    session/new   -> start_workflow(AgentSession, id="<prefix><session_id>")
    session/prompt -> execute_update("prompt", PromptIn)   (queued between turns)
    session/cancel -> execute_update("cancel")             (cancels the turn task)
    PR merged     -> signal("pr_merged", PrEvent)          (from the watcher)

Durability contract (DeepAgents-in-Temporal):
  - DeepAgents/LangGraph loop inside the workflow, replaying deterministically;
    every model/tool call is a Temporal activity. No durable checkpointer, ever
    (replay-unsafe). Temporal history IS the durable store.
  - Multi-turn memory lives in the in-run InMemorySaver; the canonical
    transcript is a plain message-dict list carried across continue-as-new
    (CAN) — turn-boundary only, so the in-memory checkpointer is disposable.
  - Streaming: the plugin's streaming activity (DeepAgentsPlugin
    streaming_topic=<STREAM_TOPIC>) pushes model deltas to the workflow's
    WorkflowStream; the workflow itself publishes tool/plan/turn events on the
    same topic. The bridge replays the log from a stream offset — that log,
    not a checkpointer, is the session-replay source.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.contrib.workflow_streams import WorkflowStream, WorkflowStreamState
from temporalio.exceptions import ApplicationError

with workflow.unsafe.imports_passed_through():
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        ToolMessage,
        message_to_dict,
        messages_from_dict,
    )
    from langgraph.types import Command
    from temporalio.deepagents import create_temporal_deep_agent

    # Import the seam + policy types here too: the sandbox must reuse the
    # main-process module objects (with their configured state), not re-import
    # fresh copies — configure() is called at host import time, before any
    # sandbox import, and a fresh sandbox copy would see None seams.
    from . import policy
    from ._config import SessionSpec, get_tool_builder

MODEL_START_TO_CLOSE_SECS = 600
MAX_GATES_PER_RUN = 10
# Workflow ids are prefixed so external components can distinguish session
# workflows from others on the queue.
WORKFLOW_PREFIX = os.environ.get("ACP_BRIDGE_WORKFLOW_PREFIX", "agent-")
STREAM_TOPIC = "acp"


def session_workflow_id(session_id: str) -> str:
    return f"{WORKFLOW_PREFIX}{session_id}"


@dataclasses.dataclass
class PromptIn:
    text: str
    kind: str = "user"  # user | system (e.g. pr_merged wakes)


@dataclasses.dataclass
class SessionStart:
    """Session input. `run` is the full agent spec, resolved CLIENT-side from
    the host's configuration (same audit-trail rationale as SessionSpec)."""

    run: SessionSpec
    session_id: str
    # Test/driver hook: continue-as-new deterministically after N turns.
    # History growth otherwise triggers CAN via is_continue_as_new_suggested().
    force_can_after_turns: int = 0


@dataclasses.dataclass
class PrEvent:
    """A merged PR, delivered to the session by the watcher."""

    repo: str
    number: int
    title: str
    url: str
    workflow_id: str  # target session workflow id (parsed from the PR marker)
    merged_at: str = ""


def _publish(stream: WorkflowStream, entry: dict[str, Any]) -> None:
    """Publish one event dict on the bridge-facing topic.

    Event vocabulary (the bridge translates these to ACP session/update):
      message/thought: {"kind", "text"}
      tool_start:      {"kind", "id", "name", "args"}
      tool_end:        {"kind", "id", "output"}
      plan:            {"kind", "todos": [{content, status, priority}]}
      approval_request: {"kind", "tool", "args", "description"} — published
                         just before the durable gate pause; the bridge asks
                         the ACP client (request_permission) and answers via
                         the resolve_approval update.
      approval_resolved: {"kind", "action", "response"} — published after the
                         decision lands (lets replays skip answered gates).
      turn_end:        {"kind", "stop_reason", "output"}
    Model token deltas arrive on the same topic from the plugin's streaming
    activity (ChatGenerationChunk dumps — distinguishable by their
    "generations" key; see bridge.translate_stream_item).
    """
    stream.topic(STREAM_TOPIC).publish(entry)


def _text_of(content: Any) -> str:
    """Extract plain text from LangChain message content (str or blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


@workflow.defn
class AgentSession:
    @workflow.init
    def __init__(
        self,
        start: SessionStart,
        stream_state: WorkflowStreamState | None = None,
        transcript: list[dict[str, Any]] | None = None,
    ) -> None:
        # MUST be constructed directly from @workflow.init (library-enforced).
        # Stream state is carried across CAN and typed exactly — see
        # WorkflowStream.__init__ docstring (Any-typed fields silently break).
        self._stream = WorkflowStream(prior_state=stream_state)
        self._transcript: list[dict[str, Any]] = list(transcript or [])
        self._queue: list[PromptIn] = []
        self._pending: policy.PendingApproval | None = None
        self._decision: policy.ApprovalDecision | None = None
        self._turn_task: asyncio.Task | None = None
        self._turn_count = 0
        self._gates = 0

    @workflow.run
    async def run(
        self,
        start: SessionStart,
        stream_state: WorkflowStreamState | None = None,
        transcript: list[dict[str, Any]] | None = None,
    ) -> None:
        run = start.run
        if not run.tools:
            raise ApplicationError("no tools registered", non_retryable=True)
        self._transcript = list(transcript or [])
        # First turn of a (re)started run must re-seed the fresh in-memory
        # checkpointer with the carried transcript; later turns are incremental.
        needs_seed = True
        while True:
            await workflow.wait_condition(lambda: bool(self._queue))
            prompt = self._queue.pop(0)
            self._turn_count += 1
            task = asyncio.create_task(self._turn(run, prompt.text, seed=needs_seed))
            self._turn_task = task
            try:
                await task
                stop_reason = "end_turn"
            except asyncio.CancelledError:
                stop_reason = "cancelled"
            finally:
                self._turn_task = None
            needs_seed = False
            _publish(self._stream, {"kind": "turn_end", "stop_reason": stop_reason})
            if (
                start.force_can_after_turns
                and self._turn_count >= start.force_can_after_turns
            ) or workflow.info().is_continue_as_new_suggested():
                # CAN only at turn boundaries. The stream log + transcript are
                # carried; the in-memory checkpointer is discarded by design.
                workflow.continue_as_new(
                    args=[start, self._stream.get_state(), self._transcript]
                )

    async def _turn(self, run: SessionSpec, text: str, *, seed: bool) -> None:
        """One agent turn: stream events to the bridge topic, keep transcript."""
        agent = create_temporal_deep_agent(
            model=run.model,
            system_prompt=run.system_prompt,
            tools=get_tool_builder()(list(run.tools)),
            interrupt_on=policy.build_interrupt_on(list(run.tools)),
            checkpointer=self._checkpointer(),
            activity_options={
                "start_to_close_timeout": timedelta(seconds=MODEL_START_TO_CLOSE_SECS)
            },
        )
        config = {"configurable": {"thread_id": workflow.info().workflow_id}}
        human = HumanMessage(content=text)
        base: list[BaseMessage] = []
        if seed and self._transcript:
            base = messages_from_dict(self._transcript)

        turn_messages: list[BaseMessage] = []
        interrupt: Any = None
        try:
            gen_input: Any = {"messages": base + [human]}
            while True:
                interrupt = None
                async for item in agent.astream(
                    gen_input,
                    config=config,
                    stream_mode=["messages", "updates"],
                    subgraphs=True,
                ):
                    _ns, mode, data = item
                    if mode == "messages":
                        self._on_message_chunk(data)
                    elif mode == "updates":
                        self._on_update(data, turn_messages)
                        if isinstance(data, dict) and "__interrupt__" in data:
                            interrupt = data["__interrupt__"]
                if interrupt is None:
                    break
                # HITL gate: durable pause, then resume with the decision.
                self._gates += 1
                if self._gates > MAX_GATES_PER_RUN:
                    raise ApplicationError(
                        "too many approval gates in one run", non_retryable=True
                    )
                self._pending = policy.pending_from_interrupt(interrupt[0])
                # Surface the gate on the durable stream BEFORE pausing, so
                # the bridge (live or replaying) sees what is being waited
                # on; the bridge asks the ACP client and answers via the
                # resolve_approval update.
                _publish(
                    self._stream,
                    {
                        "kind": "approval_request",
                        "tool": self._pending.tool,
                        "args": self._pending.args,
                        "description": self._pending.description,
                    },
                )
                await workflow.wait_condition(lambda: self._decision is not None)
                decision, self._decision, self._pending = self._decision, None, None
                _publish(
                    self._stream,
                    {
                        "kind": "approval_resolved",
                        "action": decision.action,
                        "response": decision.response,
                    },
                )
                gen_input = Command(resume=decision.to_resume_payload())
        finally:
            # On cancel, keep whatever of the turn already completed.
            self._transcript = self._transcript + _serialize([human] + turn_messages)

    def _checkpointer(self) -> Any:
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()

    def _on_message_chunk(self, data: tuple[Any, dict[str, Any]]) -> None:
        chunk = data[0] if isinstance(data, tuple) else data
        if not isinstance(chunk, AIMessage) and not hasattr(chunk, "content"):
            return
        text = _text_of(getattr(chunk, "content", ""))
        if text:
            _publish(self._stream, {"kind": "message", "text": text})

    def _on_update(self, data: Any, turn_messages: list[BaseMessage]) -> None:
        if not isinstance(data, dict):
            return
        for delta in data.values():
            if not isinstance(delta, dict):
                continue
            updates = delta.get("messages") or []
            if isinstance(updates, dict):
                updates = list(updates.values())
            for msg in updates:
                if not isinstance(msg, BaseMessage):
                    continue
                turn_messages.append(msg)
                if isinstance(msg, ToolMessage):
                    _publish(
                        self._stream,
                        {
                            "kind": "tool_end",
                            "id": msg.tool_call_id or "",
                            "output": _text_of(msg.content)[:4000],
                        },
                    )
                elif isinstance(msg, AIMessage):
                    for tc in msg.tool_calls or []:
                        if tc.get("name") == "write_todos":
                            _publish(
                                self._stream,
                                {
                                    "kind": "plan",
                                    "todos": (tc.get("args") or {}).get("todos") or [],
                                },
                            )
                        else:
                            _publish(
                                self._stream,
                                {
                                    "kind": "tool_start",
                                    "id": tc.get("id") or "",
                                    "name": tc.get("name") or "tool",
                                    "args": tc.get("args") or {},
                                },
                            )

    # -- client surface -------------------------------------------------------

    @workflow.update
    async def prompt(self, msg: PromptIn) -> None:
        """Bridge session/prompt -> queued between turns; returns on acceptance."""
        if not msg.text.strip():
            raise ApplicationError("empty prompt", non_retryable=True)
        self._queue.append(msg)

    @workflow.update
    async def cancel(self) -> None:
        """Bridge session/cancel -> cancel the in-flight turn task."""
        if self._turn_task is not None and not self._turn_task.done():
            self._turn_task.cancel()

    @workflow.signal
    def pr_merged(self, event: PrEvent) -> None:
        """Watcher: a PR authored by this session was merged. The agent wakes
        at the next loop iteration and reviews the rollout."""
        self._queue.append(
            PromptIn(
                text=(
                    f"[system] PR #{event.number} ({event.repo}) was merged: "
                    f"{event.title} ({event.url}). Review the effects of this "
                    "change with your read-only tools; open a follow-up PR if "
                    "anything needs fixing."
                ),
                kind="system",
            )
        )

    @workflow.query
    def transcript_tail(self, limit: int = 20) -> dict[str, Any]:
        return {
            "transcript": self._transcript[-limit:],
            "queued": [p.text for p in self._queue],
        }

    @workflow.query
    def pending_approval(self) -> policy.PendingApproval | None:
        return self._pending

    @workflow.update
    async def resolve_approval(self, decision: dict[str, str | None]) -> None:
        action = decision.get("action", "")
        if action not in ("approve", "respond"):
            raise ApplicationError(
                f"invalid action {action!r}: must be 'approve' or 'respond'",
                non_retryable=True,
            )
        self._decision = policy.ApprovalDecision(
            action=action, response=decision.get("response")
        )


def _serialize(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """LangChain messages -> plain dicts (the durable transcript form)."""
    return [message_to_dict(m) for m in messages]

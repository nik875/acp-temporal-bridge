"""AgentSession e2e: prompt → streamed turn → transcript carry across CAN → HITL gate.

Uses the plugin's offline FakeModel/MockTool — no network, no LLM. Streaming is
verified through the WorkflowStreamClient (the bridge's actual replay path).
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, ClassVar

from temporalio import workflow
from temporalio.contrib.workflow_streams import WorkflowStreamClient
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

with workflow.unsafe.imports_passed_through():
    from acp.schema import AgentMessageChunk
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from temporalio.deepagents import DeepAgentsPlugin
    from temporalio.deepagents.testing import MockTool, mock_model_provider

from acp_temporal_bridge import (
    AgentSession,
    PromptIn,
    SessionSpec,
    SessionStart,
    configure,
    session_workflow_id,
)
from acp_temporal_bridge.bridge import translate_stream_item

SESSION_ID = "e2e"


def _session_start(**kwargs: Any) -> SessionStart:
    return SessionStart(
        run=SessionSpec(model="mock", system_prompt="e2e", tools=["echo"]),
        session_id=SESSION_ID,
        **kwargs,
    )


def _configure(interrupt_tool: str | None = None) -> None:
    configure(
        tool_builder=lambda names: [MockTool(name="echo", result="ran")],
        interrupt_compiler=(
            (lambda names: {"echo": True}) if interrupt_tool else None
        ),
        default_spec=_session_start().run,
    )


def _plugin(responses: list[Any]) -> DeepAgentsPlugin:
    # The package's seam state (configure()) lives in module globals — the
    # sandbox must reuse the main-process modules, not re-import.
    return DeepAgentsPlugin(
        model_provider=mock_model_provider(responses),
        streaming_topic="acp",
        passthrough_modules=["acp_temporal_bridge"],
    )


async def _workflow_env():
    cli = os.environ.get("TEMPORAL_CLI_PATH")
    if cli is not None:
        return await WorkflowEnvironment.start_local(dev_server_existing_path=cli)
    return await WorkflowEnvironment.start_time_skipping()


def _contents(info: dict, prefix: str) -> list[str]:
    """Transcript contents starting with `prefix` (data field of message dicts)."""
    out = []
    for msg in info.get("transcript", []):
        content = msg.get("data", {}).get("content", "") if isinstance(msg, dict) else ""
        if isinstance(content, str) and content.startswith(prefix):
            out.append(content)
    return out


async def _collect_until_turn_end(
    client: Any, offset: int
) -> tuple[list[dict[str, Any]], list[Any], int]:
    """Subscribe from `offset` until turn_end; return (events, acp updates, offset)."""
    stream = WorkflowStreamClient.create(client, session_workflow_id(SESSION_ID))
    events: list[dict[str, Any]] = []
    updates: list[Any] = []
    async for item in stream.subscribe("acp", from_offset=offset):
        data = getattr(item, "data", None)
        if not isinstance(data, dict):
            continue
        events.append(data)
        updates.extend(translate_stream_item(item))
        if data.get("kind") == "turn_end":
            return events, updates, getattr(item, "offset", offset) + 1
        if len(events) > 200:
            break
    return events, updates, offset


async def _prompt_and_wait(
    client: Any, handle: Any, text: str, offset: int
) -> tuple[list[dict], list[Any], int]:
    await handle.execute_update(AgentSession.prompt, PromptIn(text=text))
    events, updates, end = await asyncio.wait_for(
        _collect_until_turn_end(client, offset), timeout=30
    )
    assert events and events[-1]["kind"] == "turn_end", f"no turn_end: {events}"
    return events, updates, end


async def test_prompt_streams_and_transcribes() -> None:
    """One turn: message event on the stream, transcript records both sides."""
    _configure()
    async with await _workflow_env() as env, Worker(
        env.client,
        task_queue="t",
        workflows=[AgentSession],
        plugins=[_plugin(["ok."])],
    ):
        handle = await env.client.start_workflow(
            AgentSession.run,
            args=[_session_start(), None, None],
            id=session_workflow_id(SESSION_ID),
            task_queue="t",
        )
        _, updates1, _offset = await _prompt_and_wait(
            env.client, handle, "hello", offset=0
        )
        assert any(
            isinstance(u, AgentMessageChunk) and "ok." in u.content.text
            for u in updates1
        ), updates1

        info = await handle.query(AgentSession.transcript_tail)
        assert _contents(info, "hello"), "human message missing from transcript"
        assert any(
            isinstance(m.get("data"), dict) and m["data"].get("type") == "ai"
            for m in info["transcript"]
        ), "ai message missing from transcript"


async def test_transcript_carries_across_continue_as_new() -> None:
    """force_can_after_turns=1: turn 2 still sees turn 1's transcript."""
    _configure()
    async with await _workflow_env() as env, Worker(
        env.client,
        task_queue="t",
        workflows=[AgentSession],
        plugins=[_plugin(["first reply", "second reply"])],
    ):
        handle = await env.client.start_workflow(
            AgentSession.run,
            args=[_session_start(force_can_after_turns=1), None, None],
            id=session_workflow_id(SESSION_ID),
            task_queue="t",
        )
        _, _, offset1 = await _prompt_and_wait(env.client, handle, "first", offset=0)
        # CAN happens at the turn boundary; the new run carries the transcript
        # and continues the stream offsets. Subscribe past turn 1's end.
        _, turn2, _ = await _prompt_and_wait(
            env.client, handle, "second", offset=offset1
        )
        assert any(
            isinstance(u, AgentMessageChunk) and "second reply" in u.content.text
            for u in turn2
        ), turn2

        for _ in range(20):
            try:
                info = await handle.query(AgentSession.transcript_tail)
                break
            except Exception:  # noqa: BLE001 — CAN transition window
                await asyncio.sleep(0.1)
        texts = [
            m.get("data", {}).get("content", "")
            for m in info["transcript"]
            if isinstance(m.get("data"), dict)
        ]
        assert "first" in texts and "second" in texts, texts
        assert any("first reply" in t for t in texts), texts


async def test_approval_gate_approve_resumes() -> None:
    """Gated tool call pauses the workflow; resolve_approval(approve) resumes it."""
    _configure(interrupt_tool="echo")
    # Script: tool call, then final text after the gate resumes.
    async with await _workflow_env() as env, Worker(
        env.client,
        task_queue="t",
        workflows=[AgentSession],
        plugins=[
            _plugin(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "echo", "args": {"x": "1"}, "id": "call1"}
                        ],
                    ),
                    "done.",
                ]
            )
        ],
    ):
        handle = await env.client.start_workflow(
            AgentSession.run,
            args=[_session_start(), None, None],
            id=session_workflow_id(SESSION_ID),
            task_queue="t",
        )
        await handle.execute_update(AgentSession.prompt, PromptIn(text="use the tool"))

        pending = None
        for _ in range(100):
            pending = await handle.query(AgentSession.pending_approval)
            if pending is not None:
                break
            await asyncio.sleep(0.05)
        assert pending is not None, "gate never surfaced"
        assert pending.tool == "echo", pending

        await handle.execute_update(
            AgentSession.resolve_approval, {"action": "approve", "response": None}
        )
        events, _, _ = await asyncio.wait_for(
            _collect_until_turn_end(env.client, 0), timeout=30
        )
        assert events[-1]["kind"] == "turn_end" and events[-1]["stop_reason"] == "end_turn"
        info = await handle.query(AgentSession.transcript_tail)
        assert (await handle.query(AgentSession.pending_approval)) is None, (
            "gate should be cleared"
        )
        del info  # transcript shape asserted in other tests


class _BoomModel(BaseChatModel):
    """Always-failing model (FakeModel can only script successes)."""

    @property
    def _llm_type(self) -> str:
        return "boom"

    def _generate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        raise RuntimeError("synthetic model failure")


async def test_failed_turn_surfaces_error_and_session_survives() -> None:
    """A turn whose model activity fails exhausts its bounded retries, records
    turn_end(stop_reason="error") on the durable stream, and leaves the
    session workflow alive for the next prompt (run() must not die with the
    turn — poison-pill containment)."""
    _configure()
    plugin = DeepAgentsPlugin(
        model_provider=lambda name: _BoomModel(),
        streaming_topic="acp",
        passthrough_modules=["acp_temporal_bridge"],
    )
    async with await _workflow_env() as env, Worker(
        env.client,
        task_queue="t",
        workflows=[AgentSession],
        plugins=[plugin],
    ):
        handle = await env.client.start_workflow(
            AgentSession.run,
            args=[_session_start(), None, None],
            id=session_workflow_id(SESSION_ID),
            task_queue="t",
        )
        await handle.execute_update(AgentSession.prompt, PromptIn(text="die"))
        events, _, end = await asyncio.wait_for(
            _collect_until_turn_end(env.client, 0), timeout=60
        )
        assert events[-1]["kind"] == "turn_end", events[-1]
        assert events[-1]["stop_reason"] == "error", events[-1]

        # The session survived: a follow-up prompt runs to another turn_end.
        await handle.execute_update(AgentSession.prompt, PromptIn(text="again"))
        events2, _, _ = await asyncio.wait_for(
            _collect_until_turn_end(env.client, end), timeout=60
        )
        assert events2[-1]["kind"] == "turn_end", events2[-1]
        assert events2[-1]["stop_reason"] == "error", events2[-1]


class _RecordingModel(BaseChatModel):
    """Records the human-visible text it was asked over, per model call."""

    calls: ClassVar[list[str]] = []  # class-level: models are rebuilt per activity

    @property
    def _llm_type(self) -> str:
        return "recording"

    def bind_tools(self, tools: Any, **kwargs: Any) -> _RecordingModel:
        # The activity binds the tool set before invoking; ignore it.
        return self

    def _generate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        type(self).calls.append("".join(str(m.content) for m in messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok."))])

    async def _agenerate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        # The streaming model activity drives the async path (see FakeModel).
        return self._generate(messages)


async def test_turn_two_sees_turn_one_memory() -> None:
    """Regression (Paseo E2E): turn 2's model context must include turn 1.

    The old seed-first-turn-only scheme rebuilt the agent with a fresh
    checkpointer each turn and passed base=[] on turn 2 — cross-turn memory
    silently broke unless continue-as-new re-seeded at the boundary.
    """
    _configure()
    _RecordingModel.calls = []
    provider_calls = _RecordingModel.calls
    plugin = DeepAgentsPlugin(
        model_provider=lambda name: _RecordingModel(),
        streaming_topic="acp",
        passthrough_modules=["acp_temporal_bridge"],
    )
    async with await _workflow_env() as env, Worker(
        env.client,
        task_queue="t",
        workflows=[AgentSession],
        plugins=[plugin],
    ):
        handle = await env.client.start_workflow(
            AgentSession.run,
            args=[_session_start(), None, None],
            id=session_workflow_id(SESSION_ID),
            task_queue="t",
        )
        await handle.execute_update(
            AgentSession.prompt, PromptIn(text="remember: XYZZY-marker")
        )
        _, _, end1 = await asyncio.wait_for(
            _collect_until_turn_end(env.client, 0), timeout=30
        )
        await handle.execute_update(
            AgentSession.prompt, PromptIn(text="what did I ask you to remember?")
        )
        _, _, _ = await asyncio.wait_for(
            _collect_until_turn_end(env.client, end1), timeout=30
        )

    assert len(provider_calls) >= 2, provider_calls
    turn2 = provider_calls[-1]
    assert "XYZZY-marker" in turn2, f"turn 2 forgot turn 1: {turn2[:500]!r}"

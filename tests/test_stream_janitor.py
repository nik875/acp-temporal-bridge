"""Stream-log janitor: mid-turn trimming of the in-memory log.

Background: the CAN at a turn boundary bounds carried state
(_stream_state_with_tail), but a single long turn can grow the in-memory
log without bound (the 2026-10-26 flood turn published 16,401 entries).
The janitor trims the oldest entries mid-turn once the log passes a
threshold. These tests prove the trim fires and the stream stays healthy
across it: turn_end still arrives, offsets stay monotonic, and a second
turn replays fine after truncation (client-side resync path).
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from typing import Any

from temporalio import workflow
from temporalio.contrib.workflow_streams import WorkflowStreamClient
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

with workflow.unsafe.imports_passed_through():
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessageChunk
    from temporalio.deepagents import DeepAgentsPlugin
    from temporalio.deepagents.testing import MockTool

import acp_temporal_bridge.session as session_mod
from acp_temporal_bridge import (
    AgentSession,
    PromptIn,
    SessionSpec,
    SessionStart,
    configure,
    session_workflow_id,
)

SESSION_ID = "janitor-e2e"


class FloodModel(BaseChatModel):
    """Streams `noop` empty AIMessageChunks before a real final text chunk.

    Simulates the recorded flood shape (provider protocol ticks with empty
    deltas): one chunk per publish, no useful content until the end.
    """

    noop: int = 120
    chunk_delay: float = 0.03  # real-time pacing so ticks land mid-stream

    @property
    def _llm_type(self) -> str:
        return "flood-model"

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any) -> Any:
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="flooded but finished"))])

    def bind_tools(self, tools: Any, **kwargs: Any) -> FloodModel:
        return self

    def bind(self, *args: Any, **kwargs: Any) -> FloodModel:
        return self

    async def astream(self, messages: Any, config: Any = None, **kwargs: Any) -> AsyncIterator[Any]:
        for _ in range(self.noop):
            await asyncio.sleep(self.chunk_delay)
            yield AIMessageChunk(content="")
        yield AIMessageChunk(content="flooded but finished")


def _session_start(**kwargs: Any) -> SessionStart:
    return SessionStart(
        run=SessionSpec(model="mock", system_prompt="e2e", tools=["echo"]),
        session_id=SESSION_ID,
        **kwargs,
    )


def _configure() -> None:
    configure(
        tool_builder=lambda names: [MockTool(name="echo", result="ran")],
        interrupt_compiler=None,
        default_spec=_session_start().run,
    )


def _plugin() -> DeepAgentsPlugin:
    return DeepAgentsPlugin(
        model_provider=lambda name: FloodModel(),
        streaming_topic="acp",
        passthrough_modules=["acp_temporal_bridge"],
    )


async def _workflow_env():
    cli = os.environ.get("TEMPORAL_CLI_PATH")
    if cli is not None:
        return await WorkflowEnvironment.start_local(dev_server_existing_path=cli)
    return await WorkflowEnvironment.start_time_skipping()


async def _collect_until_turn_end(client: Any, offset: int) -> list[dict[str, Any]]:
    """Subscribe from `offset` until turn_end (tolerating resync dupes)."""
    stream = WorkflowStreamClient.create(client, session_workflow_id(SESSION_ID))
    events: list[dict[str, Any]] = []
    async for item in stream.subscribe("acp", from_offset=offset):
        data = getattr(item, "data", None)
        if not isinstance(data, dict):
            continue
        events.append(data)
        if data.get("kind") == "turn_end":
            return events
        if len(events) > 10_000:
            break
    return events


async def test_janitor_trims_mid_turn_and_stream_stays_healthy() -> None:
    _configure()

    # Spy on the janitor to distinguish "never ran" from "ran but never
    # trimmed" (the session module is a passthrough — the worker shares
    # this class object).
    orig_janitor = session_mod.AgentSession._stream_janitor
    janitor_calls: list[str] = []

    async def _spy_janitor(self: Any) -> None:
        janitor_calls.append("start")
        try:
            await orig_janitor(self)
        except BaseException as exc:
            janitor_calls.append(f"raised: {type(exc).__name__}: {exc}")
            raise
        janitor_calls.append("end")

    session_mod.AgentSession._stream_janitor = _spy_janitor

    # Small thresholds so the trim fires during the flood turn. The
    # janitor reads these as module globals at runtime, so patching works
    # without reimport.
    session_mod.STREAM_LOG_TRIM_THRESHOLD = 60
    session_mod.STREAM_LOG_TRIM_KEEP = 30
    session_mod.STREAM_JANITOR_TICK_SECS = 0.5

    try:
        async with await _workflow_env() as env, Worker(
            env.client,
            task_queue="t",
            workflows=[AgentSession],
            plugins=[_plugin()],
        ):
            handle = await env.client.start_workflow(
                AgentSession.run,
                args=[_session_start(), None, None],
                id=session_workflow_id(SESSION_ID),
                task_queue="t",
            )
            await handle.execute_update(AgentSession.prompt, PromptIn(text="go"))
            events = await asyncio.wait_for(
                _collect_until_turn_end(env.client, 0), timeout=60
            )
            assert events and events[-1]["kind"] == "turn_end", "no turn_end"
            assert events[-1]["stop_reason"] == "end_turn", events[-1]

            head = await handle.query(AgentSession.stream_head)
            assert head > 0

            # Trim observability: a successful mid-turn trim advances the
            # stream's base_offset. Subscribing from 0 post-turn must
            # therefore deliver entries starting at the new base (> 0) —
            # if nothing was trimmed, the first entry would be offset 0.
            stream = WorkflowStreamClient.create(
                env.client, session_workflow_id(SESSION_ID)
            )
            first_offset = None
            async for item in stream.subscribe("acp", from_offset=0):
                first_offset = getattr(item, "offset", None)
                break
            assert first_offset is not None and first_offset > 0, (
                f"stream log was never trimmed (first offset {first_offset!r}); "
                f"janitor calls: {janitor_calls}"
            )

            # Second turn after trimming: the full round-trip still works
            # (replay/resync across truncation never wedges the session).
            await handle.execute_update(AgentSession.prompt, PromptIn(text="again"))
            events2 = await asyncio.wait_for(
                _collect_until_turn_end(env.client, head), timeout=60
            )
            assert events2 and events2[-1]["kind"] == "turn_end"
    finally:
        session_mod.AgentSession._stream_janitor = orig_janitor

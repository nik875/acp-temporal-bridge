"""UI-liveness gate: an agent may never continue without streaming to the UI.

2026-10-28 invariant decision. The bridge heartbeats every attached
session (ui_heartbeat(epoch)); ui_detached(epoch) on disconnect. Every
turn start/resume and every model call gates on liveness:

* gate INACTIVE until the first heartbeat (old histories, fresh runs —
  backward compatibility);
* UI detached  -> turn pauses durably (never runs headless);
* detached past UI_LOST_TURN_BUDGET_SECS -> turn fails cleanly with
  stop_reason="error" — the SESSION AND TRANSCRIPT ALWAYS SURVIVE;
* cancel during the pause -> stop_reason="cancelled", session survives;
* a newer epoch's heartbeat re-adopts (reconnect); a stale epoch's is
  ignored.

Uses the plugin's offline FakeModel/MockTool — no network, no LLM.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from temporalio import workflow
from temporalio.contrib.workflow_streams import WorkflowStreamClient
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

with workflow.unsafe.imports_passed_through():
    from temporalio.deepagents import DeepAgentsPlugin
    from temporalio.deepagents.testing import MockTool, mock_model_provider

import acp_temporal_bridge.session as session_mod
from acp_temporal_bridge import (
    AgentSession,
    PromptIn,
    SessionSpec,
    SessionStart,
    configure,
    session_workflow_id,
)

SESSION_ID = "ui-gate"


def _session_start(**kwargs: Any) -> SessionStart:
    return SessionStart(
        run=SessionSpec(model="mock", system_prompt="e2e", tools=["echo"]),
        session_id=SESSION_ID,
        **kwargs,
    )


def _configure() -> None:
    configure(
        tool_builder=lambda names: [MockTool(name="echo", result="ran")],
        default_spec=_session_start().run,
    )


def _plugin(responses: list[Any]) -> DeepAgentsPlugin:
    # passthrough_modules keeps the test's monkeypatched module constants
    # (UI_LOST_TURN_BUDGET_SECS) visible to the workflow code.
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


async def _collect_until_turn_end(
    client: Any, offset: int
) -> tuple[list[dict[str, Any]], int]:
    """Subscribe from `offset` until turn_end; return (event dicts, next offset)."""
    stream = WorkflowStreamClient.create(client, session_workflow_id(SESSION_ID))
    events: list[dict[str, Any]] = []
    async for item in stream.subscribe("acp", from_offset=offset):
        data = getattr(item, "data", None)
        if not isinstance(data, dict):
            continue
        events.append(data)
        if data.get("kind") == "turn_end":
            return events, getattr(item, "offset", offset) + 1
        if len(events) > 200:
            break
    return events, offset


async def _prompt_and_wait(
    client: Any, handle: Any, text: str, offset: int
) -> tuple[list[dict], int]:
    # Subscribe BEFORE the prompt: a gate-failed turn completes in pure
    # virtual time (no model activity), so the workflow may finish before
    # an update-first collector even attaches.
    collect = asyncio.create_task(_collect_until_turn_end(client, offset))
    await asyncio.sleep(0.1)
    await handle.execute_update(AgentSession.prompt, PromptIn(text=text))
    events, offset = await asyncio.wait_for(collect, timeout=30)
    assert events and events[-1]["kind"] == "turn_end", f"no turn_end: {events}"
    return events, offset


async def _wait_turn_active(handle: Any) -> None:
    """Poll until the turn task exists (the gate may hold it indefinitely)."""
    for _ in range(100):
        if await handle.query(AgentSession.turn_active):
            return
        await asyncio.sleep(0.05)
    raise AssertionError("turn never became active")


async def _transcript_texts(handle: Any) -> list[str]:
    info = await handle.query(AgentSession.transcript_tail)
    return [
        m.get("data", {}).get("content", "")
        for m in info.get("transcript", [])
        if isinstance(m.get("data"), dict)
    ]


async def test_no_heartbeat_turn_completes_backward_compat() -> None:
    """No heartbeat ever: the gate is INACTIVE (old histories / fresh run
    before the bridge's first beat) — turns complete as before."""
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
        events, _ = await _prompt_and_wait(env.client, handle, "hello", 0)
        assert events[-1]["stop_reason"] == "end_turn"


async def test_heartbeat_before_prompt_turn_completes() -> None:
    """Heartbeat armed + UI live: the gate passes, the turn streams."""
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
        await handle.signal(AgentSession.ui_heartbeat, 1)
        events, _ = await _prompt_and_wait(env.client, handle, "hello", 0)
        assert events[-1]["stop_reason"] == "end_turn"


async def test_detached_turn_fails_cleanly_session_survives(
    monkeypatch: Any,
) -> None:
    """UI detached before the turn: it pauses at the gate, trips the
    UI_LOST budget, and fails CLEANLY (stop_reason="error") — the session
    and transcript survive, and a reconnected UI resumes normal turns."""
    _configure()
    monkeypatch.setattr(session_mod, "UI_LOST_TURN_BUDGET_SECS", 1.0)
    async with await _workflow_env() as env, Worker(
        env.client,
        task_queue="t",
        workflows=[AgentSession],
        plugins=[_plugin(["ok.", "back."])],
    ):
        handle = await env.client.start_workflow(
            AgentSession.run,
            args=[_session_start(), None, None],
            id=session_workflow_id(SESSION_ID),
            task_queue="t",
        )
        await handle.signal(AgentSession.ui_heartbeat, 1)  # arm the gate
        await handle.signal(AgentSession.ui_detached, 1)
        events, offset = await _prompt_and_wait(env.client, handle, "lost me", 0)
        assert events[-1]["stop_reason"] == "error", events
        # The session SURVIVED: a reattached client (fresh heartbeat epoch)
        # prompts and gets a normal turn.
        await handle.signal(AgentSession.ui_heartbeat, 2)
        events2, _ = await _prompt_and_wait(env.client, handle, "back", offset)
        assert events2[-1]["stop_reason"] == "end_turn"
        texts = " ".join(await _transcript_texts(handle))
        assert "back" in texts


async def test_cancel_during_ui_pause_session_survives(monkeypatch: Any) -> None:
    """Interrupt a UI-paused turn: stop_reason="cancelled", session
    survives — interrupt kills only the turn."""
    _configure()
    monkeypatch.setattr(session_mod, "UI_LOST_TURN_BUDGET_SECS", 1.0)
    async with await _workflow_env() as env, Worker(
        env.client,
        task_queue="t",
        workflows=[AgentSession],
        plugins=[_plugin(["ok.", "after."])],
    ):
        handle = await env.client.start_workflow(
            AgentSession.run,
            args=[_session_start(), None, None],
            id=session_workflow_id(SESSION_ID),
            task_queue="t",
        )
        await handle.signal(AgentSession.ui_heartbeat, 1)
        await handle.signal(AgentSession.ui_detached, 1)
        await handle.execute_update(AgentSession.prompt, PromptIn(text="paused"))
        await _wait_turn_active(handle)
        await handle.execute_update(AgentSession.cancel)
        collect = asyncio.create_task(_collect_until_turn_end(env.client, 0))
        await asyncio.sleep(0.1)
        events, offset = await asyncio.wait_for(collect, timeout=30)
        assert events[-1]["stop_reason"] == "cancelled", events
        # Interrupt killed ONLY the turn: the next (reattached) turn works.
        await handle.signal(AgentSession.ui_heartbeat, 2)
        events2, _ = await _prompt_and_wait(env.client, handle, "resume", offset)
        assert events2[-1]["stop_reason"] == "end_turn"


async def test_stale_epoch_ignored_new_epoch_adopted(monkeypatch: Any) -> None:
    """Epoch discipline: a STALE heartbeat (older epoch, e.g. a dead
    connection's beat racing a reconnect) cannot un-detach the session; a
    NEWER epoch re-adopts and the turn completes."""
    _configure()
    monkeypatch.setattr(session_mod, "UI_LOST_TURN_BUDGET_SECS", 1.0)
    async with await _workflow_env() as env, Worker(
        env.client,
        task_queue="t",
        workflows=[AgentSession],
        plugins=[_plugin(["ok.", "reconnected."])],
    ):
        handle = await env.client.start_workflow(
            AgentSession.run,
            args=[_session_start(), None, None],
            id=session_workflow_id(SESSION_ID),
            task_queue="t",
        )
        await handle.signal(AgentSession.ui_heartbeat, 10)  # arm, epoch 10
        await handle.signal(AgentSession.ui_detached, 10)
        await handle.signal(AgentSession.ui_heartbeat, 3)  # stale: ignored
        events, offset = await _prompt_and_wait(env.client, handle, "still detached?", 0)
        assert events[-1]["stop_reason"] == "error", events
        # Newer epoch re-adopts: the turn completes normally.
        await handle.signal(AgentSession.ui_heartbeat, 11)
        events2, _ = await _prompt_and_wait(env.client, handle, "reconnected", offset)
        assert events2[-1]["stop_reason"] == "end_turn"


async def test_detach_mid_turn_pauses_then_readopt_completes() -> None:
    """Detach mid-turn: the turn's model calls pause durably; a reconnect
    heartbeat (new epoch) re-adopts and the turn completes."""
    _configure()
    async with await _workflow_env() as env, Worker(
        env.client,
        task_queue="t",
        workflows=[AgentSession],
        plugins=[_plugin(["mid-turn answer."])],
    ):
        handle = await env.client.start_workflow(
            AgentSession.run,
            args=[_session_start(), None, None],
            id=session_workflow_id(SESSION_ID),
            task_queue="t",
        )
        await handle.signal(AgentSession.ui_heartbeat, 1)
        await handle.execute_update(AgentSession.prompt, PromptIn(text="long turn"))
        await asyncio.sleep(0.2)  # let the turn get going (best-effort)
        await handle.signal(AgentSession.ui_detached, 1)
        await asyncio.sleep(0.2)
        await handle.signal(AgentSession.ui_heartbeat, 2)  # reconnect
        collect = asyncio.create_task(_collect_until_turn_end(env.client, 0))
        await asyncio.sleep(0.1)
        events, _ = await asyncio.wait_for(collect, timeout=30)
        assert events[-1]["stop_reason"] == "end_turn", events
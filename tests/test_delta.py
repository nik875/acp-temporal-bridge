"""Delta-protocol tests (Phase 1b): model-call activity inputs shrink to
content-addressed markers, expand worker-side, and fall back to the full
input on a cache miss."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from temporalio.contrib.workflow_streams import WorkflowStreamClient
from temporalio.deepagents import DeepAgentsPlugin
from temporalio.deepagents.testing import MockTool, mock_model_provider
from temporalio.exceptions import ApplicationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from acp_temporal_bridge import (
    AgentSession,
    PromptIn,
    SessionSpec,
    SessionStart,
    _delta,
    configure,
    session_workflow_id,
)

SESSION_ID = "e2e"


def _msg(role: str, text: str) -> dict[str, Any]:
    """A dumped (dumpd-form) message like _serde.dump_messages emits."""
    return {
        "lc": 1,
        "type": "constructor",
        "id": ["langchain", "schema", "messages", role],
        "kwargs": {"content": text, "id": f"run-{role}-{text[:8]}"},
    }


@pytest.fixture(autouse=True)
def _fresh_cache() -> None:
    _delta._CACHE = _delta._DeltaCache()


def _configure() -> None:
    configure(
        tool_builder=lambda names: [MockTool(name="echo", result="ran")],
        interrupt_compiler=None,
        default_spec=SessionStart(
            run=SessionSpec(model="mock", system_prompt="e2e", tools=["echo"]),
            session_id=SESSION_ID,
        ).run,
    )


def _plugin(responses: list[Any]) -> DeepAgentsPlugin:
    return DeepAgentsPlugin(
        model_provider=mock_model_provider(responses),
        streaming_topic="acp",
        passthrough_modules=["acp_temporal_bridge"],
    )


async def _collect_until_turn_end(
    client: Any, offset: int
) -> tuple[list[dict[str, Any]], list[Any], int]:
    stream = WorkflowStreamClient.create(client, session_workflow_id(SESSION_ID))
    events: list[dict[str, Any]] = []
    updates: list[Any] = []
    async for item in stream.subscribe("acp", from_offset=offset):
        data = getattr(item, "data", None)
        if not isinstance(data, dict):
            continue
        events.append(data)
        if data.get("kind") == "turn_end":
            return events, updates, getattr(item, "offset", offset) + 1
        if len(events) > 200:
            break
    return events, updates, offset


async def _model_inputs(handle: Any) -> list[dict[str, Any]]:
    """All recorded invoke_model_streaming inputs, in history order."""
    hist = await handle.fetch_history()
    out = []
    for evt in hist.events:
        if not evt.HasField("activity_task_scheduled_event_attributes"):
            continue
        attrs = evt.activity_task_scheduled_event_attributes
        if "invoke_model_streaming" not in attrs.activity_type.name:
            continue
        out.append(json.loads(attrs.input.payloads[0].data))
    return out


# ---------------------------------------------------------------------------
# shrink
# ---------------------------------------------------------------------------


def test_shrink_first_call_sends_full_and_seeds_chain() -> None:
    msgs = [_msg("HumanMessage", "hello"), _msg("SystemMessage", "x" * 40_000)]
    assert _delta.shrink("wf", msgs) is None  # full send, no marker
    digest, _ = _delta._digest(msgs)
    assert _delta._CACHE.get(digest) == msgs
    assert _delta._CACHE.get_chain("wf") == (digest, msgs)


def test_shrink_append_only_call_emits_marker() -> None:
    base = [_msg("HumanMessage", "hello"), _msg("SystemMessage", "x" * 40_000)]
    assert _delta.shrink("wf", base) is None  # seeds the chain

    grown = [*base, _msg("AIMessage", "reply")]
    marker = _delta.shrink("wf", grown)
    assert marker is not None and len(marker) == 1
    m = marker[0][_delta.DELTA_KEY]
    assert m["v"] == 1
    assert m["prefix_len"] == len(base)
    assert m["delta"] == grown[len(base) :]
    digest, _ = _delta._digest(grown)
    assert m["full_sha256"] == digest


def test_shrink_non_prefix_change_sends_full() -> None:
    base = [_msg("HumanMessage", "hello"), _msg("SystemMessage", "x" * 40_000)]
    _delta.shrink("wf", base)

    # A resumed turn can change earlier content (transcript re-seed) — the
    # chain must detect it and fall back to a full send, not emit a marker.
    changed = [_msg("HumanMessage", "hello"), _msg("SystemMessage", "y" * 40_000)]
    assert _delta.shrink("wf", changed) is None


def test_shrink_small_payloads_never_marked() -> None:
    small = [_msg("HumanMessage", "hi")]
    assert _delta.shrink("wf", small) is None
    assert _delta.shrink("wf", [*small, _msg("AIMessage", "ok")]) is None


# ---------------------------------------------------------------------------
# expand
# ---------------------------------------------------------------------------


def test_expand_roundtrips_marker() -> None:
    base = [_msg("HumanMessage", "hello"), _msg("SystemMessage", "x" * 40_000)]
    _delta.shrink("wf", base)
    grown = [*base, _msg("AIMessage", "reply")]
    marker = _delta.shrink("wf", grown)
    assert marker is not None
    assert _delta.expand(marker) == grown


def test_expand_passes_plain_lists_through() -> None:
    plain = [_msg("HumanMessage", "hello")]  # old-history / full input
    assert _delta.expand(plain) is plain


def test_expand_cache_miss_raises_non_retryable() -> None:
    marker = [
        {
            _delta.DELTA_KEY: {
                "v": 1,
                "prefix_sha256": "deadbeef",
                "prefix_len": 3,
                "delta": [_msg("AIMessage", "x")],
                "full_sha256": "cafe",
            }
        }
    ]
    with pytest.raises(ApplicationError) as exc:
        _delta.expand(marker)
    assert exc.value.type == _delta.DELTA_MISS_ERR
    assert exc.value.non_retryable


def test_install_is_idempotent() -> None:
    import temporalio.deepagents._plugin as plugin_mod
    import temporalio.deepagents.workflow as wf_mod

    _delta.install()
    first_call_model = wf_mod.call_model
    first_activities = plugin_mod.DeepAgentActivities
    _delta.install()
    assert wf_mod.call_model is first_call_model
    assert plugin_mod.DeepAgentActivities is first_activities
    assert getattr(first_call_model, "_bridge_delta_patch", False)
    assert getattr(first_activities, "_bridge_delta_patch", False)


# ---------------------------------------------------------------------------
# e2e: the recorded history actually shrinks / the miss fallback recovers
# ---------------------------------------------------------------------------


async def test_model_inputs_shrink_to_deltas_in_history() -> None:
    """Two turns: the first model call records the full message list; the
    second records a one-element delta marker an order of magnitude smaller.
    Session correctness (stream, transcript) is unchanged."""
    _configure()
    async with (
        await WorkflowEnvironment.start_time_skipping() as env,
        Worker(
            env.client,
            task_queue="t",
            workflows=[AgentSession],
            plugins=[_plugin(["first reply", "second reply"])],
        ),
    ):
        big = "pad " * 12_000  # keep every call over DELTA_MIN_BYTES
        start = _session_start()
        handle = await env.client.start_workflow(
            AgentSession.run,
            args=[start, None, None],
            id=session_workflow_id(SESSION_ID),
            task_queue="t",
        )
        await handle.execute_update(AgentSession.prompt, PromptIn(text=f"hello {big}"))
        events, _, end = await asyncio.wait_for(_collect_until_turn_end(env.client, 0), timeout=30)
        assert events[-1]["kind"] == "turn_end"

        await handle.execute_update(AgentSession.prompt, PromptIn(text="second"))
        events2, _, _ = await asyncio.wait_for(_collect_until_turn_end(env.client, end), timeout=30)
        assert events2[-1]["kind"] == "turn_end"

        inputs = await _model_inputs(handle)
        assert len(inputs) == 2, [i["messages"][:1] for i in inputs]
        first, second = inputs
        assert not any(isinstance(m, dict) and _delta.DELTA_KEY in m for m in first["messages"])
        second_msgs = second["messages"]
        assert len(second_msgs) == 1 and _delta.DELTA_KEY in second_msgs[0]
        marker = second_msgs[0][_delta.DELTA_KEY]
        # The delta is the new messages only — the turn-1 payload is referenced.
        assert marker["delta"] and "second" in json.dumps(marker["delta"])
        assert marker["prefix_len"] >= 1


async def test_cache_miss_falls_back_to_full_input() -> None:
    """Cold worker cache (rollout/restart): the marker-carrying activity
    attempt fails non-retryably, the workflow re-dispatches the FULL input,
    and the turn completes with exactly one turn_end."""
    _configure()
    async with (
        await WorkflowEnvironment.start_time_skipping() as env,
        Worker(
            env.client,
            task_queue="t",
            workflows=[AgentSession],
            plugins=[_plugin(["first reply", "second reply"])],
        ),
    ):
        big = "pad " * 12_000
        start = _session_start()
        handle = await env.client.start_workflow(
            AgentSession.run,
            args=[start, None, None],
            id=session_workflow_id(SESSION_ID),
            task_queue="t",
        )

        # Turn 1 runs normally and warms the delta chain.
        await handle.execute_update(AgentSession.prompt, PromptIn(text=f"hello {big}"))
        events, _, end = await asyncio.wait_for(_collect_until_turn_end(env.client, 0), timeout=30)
        assert events[-1]["kind"] == "turn_end"

        # Simulate a worker restart: every expand() misses.
        import acp_temporal_bridge._delta as delta_mod

        orig_get = delta_mod._DeltaCache.get

        def cold_get(self: Any, digest: str) -> Any:
            return None

        delta_mod._DeltaCache.get = cold_get  # type: ignore[method-assign]
        try:
            await handle.execute_update(AgentSession.prompt, PromptIn(text="second"))
            events2, _, _ = await asyncio.wait_for(
                _collect_until_turn_end(env.client, end), timeout=30
            )
        finally:
            delta_mod._DeltaCache.get = orig_get  # type: ignore[method-assign]

        assert events2[-1]["kind"] == "turn_end"
        assert events2[-1]["stop_reason"] == "end_turn"

        inputs = await _model_inputs(handle)
        # Attempt 1 of turn 2: marker. Retry: full message list. Both recorded.
        assert any(
            len(i["messages"]) == 1 and _delta.DELTA_KEY in i["messages"][0] for i in inputs
        ), [list(i) for i in inputs]
        assert any(
            len(i["messages"]) > 1 and "second" in json.dumps(i["messages"]) for i in inputs
        ), [list(i) for i in inputs]


def _session_start() -> SessionStart:
    return SessionStart(
        run=SessionSpec(model="mock", system_prompt="e2e", tools=["echo"]),
        session_id=SESSION_ID,
    )

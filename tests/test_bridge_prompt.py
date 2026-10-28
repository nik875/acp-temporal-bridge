"""Bridge prompt() failure surfaces — no Temporal, scripted stream items.

Regression for the field incident where a degenerate model stream flooded the
durable log (10k+ items, no turn_end) and prompt() silently returned a
fabricated ``end_turn``. Contract after the fix:

* overflow without turn_end          -> ReplayOverflowError (never fake success)
  AND the turn is halted workflow-side (cancel update) — stopping the
  client's view is not enough; the turn must stop burning durable history
* overflow with the halted turn's
  turn_end behind it                 -> ReplayOverflowError, offsets drained
  past the stale turn_end (next prompt replays from the post-turn boundary)
* turn_end with out-of-vocabulary
  stop_reason (e.g. "error")         -> RequestError.internal_error
* turn_end with a valid stop_reason  -> PromptResponse(stop_reason)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import pytest
from acp.exceptions import RequestError
from acp.schema import PromptResponse

from acp_temporal_bridge import bridge as bridge_mod
from acp_temporal_bridge.bridge import BridgeAgent, ReplayOverflowError


@dataclass
class _Item:
    data: Any
    offset: int


class _FakeHandle:
    updates: ClassVar[list[Any]] = []

    async def execute_update(self, *args: Any, **kwargs: Any) -> None:
        _FakeHandle.updates.append(args[0] if args else None)

    async def signal(self, *args: Any, **kwargs: Any) -> None:
        _FakeHandle.updates.append(("signal", args[0] if args else None))


class _FakeTemporal:
    def get_workflow_handle(self, workflow_id: str) -> _FakeHandle:
        return _FakeHandle()


class _FakeConn:
    def __init__(self) -> None:
        self.updates: list[Any] = []

    async def session_update(self, session_id: str, update: Any) -> None:
        self.updates.append(update)


def _agent() -> tuple[BridgeAgent, _FakeConn]:
    conn = _FakeConn()
    return BridgeAgent(conn, _FakeTemporal()), conn


def _use_stream(monkeypatch: pytest.MonkeyPatch, items: list[_Item]) -> None:
    class _Stream:
        @staticmethod
        def create(client: Any, workflow_id: str) -> _Stream:
            return _Stream()
        async def subscribe(self, topic: str, from_offset: int = 0) -> Any:
            for item in items:
                if item.offset >= from_offset:
                    yield item

    monkeypatch.setattr(bridge_mod, "WorkflowStreamClient", _Stream)


async def test_overflow_halts_turn_and_raises_not_fake_end_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A flooded log (no turn_end within REPLAY_MAX_ITEMS) must be halted
    workflow-side and error out — never a fabricated end_turn."""
    _use_stream(
        monkeypatch,
        [
            _Item({"kind": "message", "text": "x"}, i)
            for i in range(bridge_mod.REPLAY_MAX_ITEMS + 1)
        ],
    )
    agent, _conn = _agent()
    with pytest.raises(ReplayOverflowError):
        await agent.prompt("s1", [])
    # The halt actually reached the workflow (cancel update executed).
    assert any(u is bridge_mod.AgentSession.cancel for u in _FakeHandle.updates)


async def test_overflow_drains_past_halted_turn_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After the guard halts the turn, the drain consumes through the halted
    turn's turn_end so the NEXT prompt replays from the post-turn boundary
    instead of tripping over the stale end."""
    max_items = bridge_mod.REPLAY_MAX_ITEMS
    _use_stream(
        monkeypatch,
        [
            *(
                _Item({"kind": "message", "text": "x"}, i)
                for i in range(max_items + 1)
            ),
            _Item({"kind": "turn_end", "stop_reason": "cancelled"}, max_items + 1),
        ],
    )
    agent, _conn = _agent()
    with pytest.raises(ReplayOverflowError):
        await agent.prompt("s1", [])
    # Offsets advanced past the halted turn's turn_end...
    assert agent._offsets["s1"] == max_items + 2
    # ...and the halt reached the workflow.
    assert any(u is bridge_mod.AgentSession.cancel for u in _FakeHandle.updates)


async def test_unknown_stop_reason_raises_request_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """stop_reason="error" (failed turn) surfaces as an ACP request error."""
    _use_stream(monkeypatch, [_Item({"kind": "turn_end", "stop_reason": "error"}, 0)])
    agent, _conn = _agent()
    with pytest.raises(RequestError) as ei:
        await agent.prompt("s1", [])
    assert ei.value.data is not None
    assert ei.value.data.get("stop_reason") == "error"


@pytest.mark.parametrize("reason", ["end_turn", "cancelled", "refusal", "max_tokens"])
async def test_valid_stop_reason_passes_through(
    monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    """In-vocabulary stop reasons ride the normal PromptResponse path."""
    _use_stream(monkeypatch, [_Item({"kind": "turn_end", "stop_reason": reason}, 0)])
    agent, _conn = _agent()
    resp = await agent.prompt("s1", [])
    assert isinstance(resp, PromptResponse)
    assert resp.stop_reason == reason


async def test_missing_stop_reason_defaults_end_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """turn_end without stop_reason stays a successful end_turn."""
    _use_stream(monkeypatch, [_Item({"kind": "turn_end"}, 0)])
    agent, _conn = _agent()
    resp = await agent.prompt("s1", [])
    assert resp.stop_reason == "end_turn"


class _CancelRecordingHandle:
    def __init__(self, workflow_id: str, log: list[str]) -> None:
        self._workflow_id = workflow_id
        self._log = log

    async def execute_update(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def signal(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def cancel(self) -> None:
        self._log.append(self._workflow_id)


class _CancelRecordingTemporal:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.cancels: list[str] = []

    def get_workflow_handle(self, workflow_id: str) -> _CancelRecordingHandle:
        return _CancelRecordingHandle(workflow_id, self.cancels)

    async def start_workflow(self, *args: Any, **kwargs: Any) -> None:
        self.started.append(kwargs.get("id", "?"))


def _cancelling_agent() -> tuple[BridgeAgent, _FakeConn, _CancelRecordingTemporal]:
    from acp_temporal_bridge import SessionSpec, configure

    configure(
        default_spec=SessionSpec(model="mock", system_prompt="e2e", tools=["echo"])
    )
    conn = _FakeConn()
    temporal = _CancelRecordingTemporal()
    return BridgeAgent(conn, temporal), conn, temporal


async def test_aclose_cancels_never_prompted_sessions() -> None:
    """aclose() cancels sessions this connection created but never prompted:
    clients (paseo probes, reconnecting UIs) spawn sessions they never use —
    without this the workflow waits on its prompt signal forever."""
    agent, _conn, temporal = _cancelling_agent()
    await agent.new_session(cwd="~")
    await agent.new_session(cwd="~")
    assert len(temporal.started) == 2
    await agent.aclose()
    assert sorted(temporal.cancels) == sorted(temporal.started)


async def test_aclose_keeps_prompted_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prompted session survives disconnect — the client may reconnect and
    reattach (the durability promise). Only the empty probes get cancelled."""
    _use_stream(
        monkeypatch, [_Item({"kind": "turn_end", "stop_reason": "end_turn"}, 0)]
    )
    agent, _conn, temporal = _cancelling_agent()
    sid = await agent.new_session(cwd="~")
    await agent.prompt(sid.session_id, [])
    await agent.aclose()
    assert temporal.cancels == []

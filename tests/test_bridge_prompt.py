"""Bridge prompt() failure surfaces — no Temporal, scripted stream items.

Regression for the field incident where a degenerate model stream flooded the
durable log (10k+ items, no turn_end) and prompt() silently returned a
fabricated ``end_turn``. Contract after the fix:

* overflow without turn_end          -> ReplayOverflowError (never fake success)
* turn_end with out-of-vocabulary
  stop_reason (e.g. "error")         -> RequestError.internal_error
* turn_end with a valid stop_reason  -> PromptResponse(stop_reason)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    async def execute_update(self, *args: Any, **kwargs: Any) -> None:
        return None


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


async def test_overflow_raises_not_fake_end_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A flooded log (no turn_end within REPLAY_MAX_ITEMS) must error out."""
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

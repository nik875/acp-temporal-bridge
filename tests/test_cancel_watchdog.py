"""Cancel watchdog: interrupt must kill the turn, period.

Regression for the 2026-10-26 production wedge: Paseo's interrupt sent
session/cancel, the bridge fire-and-forgot the cooperative
AgentSession.cancel update, the update never took effect, and the
in-flight prompt() never returned — Paseo wedged the session as
"A foreground turn is already active" forever (its own force-cancel only
repairs the daemon's view, never the agent).

Contract after the fix (bridge.py cancel(), two-phase):

* cooperative update still goes out first (normal interrupts preserve
  the durable session and transcript);
* if the turn is STILL reported alive after CANCEL_KILL_AFTER_SECS
  (turn_active query == True)        -> workflow TERMINATED server-side
* if the turn_active query cannot be answered at all (wedged worker —
  the exact case the cooperative path cannot recover from)
                                     -> workflow TERMINATED server-side
* turn_active query == False         -> no terminate (cooperative won)
* connection teardown (aclose) cancels pending watchdogs: on disconnect
  the durability contract wins, a late terminate must never murder a
  legitimately resumed session.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from acp_temporal_bridge import bridge as bridge_mod
from acp_temporal_bridge.bridge import BridgeAgent


class _WatchHandle:
    """WorkflowHandle stand-in: scripted turn_active answer, recorded kills."""

    def __init__(self, query_result: bool | Exception) -> None:
        self.query_result = query_result
        self.terminated: list[str] = []
        self.updates: list[Any] = []

    async def execute_update(self, *args: Any, **kwargs: Any) -> None:
        self.updates.append(args[0] if args else None)

    async def query(self, *args: Any, **kwargs: Any) -> bool:
        if isinstance(self.query_result, Exception):
            raise self.query_result
        return self.query_result

    async def terminate(self, *args: Any, **kwargs: Any) -> None:
        reason = kwargs.get("reason")
        self.terminated.append(reason if isinstance(reason, str) else "")


class _WatchTemporal:
    def __init__(self, handle: _WatchHandle) -> None:
        self.handle = handle

    def get_workflow_handle(self, workflow_id: str) -> _WatchHandle:
        return self.handle


class _SilentConn:
    async def session_update(self, session_id: str, update: Any) -> None:  # pragma: no cover
        raise AssertionError("cancel watchdog must not stream updates")


def _agent(handle: _WatchHandle) -> BridgeAgent:
    return BridgeAgent(_SilentConn(), _WatchTemporal(handle))


async def _drain_watchdogs(agent: BridgeAgent) -> None:
    pending = list(agent._cancel_watchdogs)
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


@pytest.fixture(autouse=True)
def _fast_watchdog(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bridge_mod, "CANCEL_KILL_AFTER_SECS", 0.01)


async def test_surviving_turn_is_terminated() -> None:
    """Turn still alive at the deadline (query == True): the workflow is
    terminated — the 'interrupt kills the agent, full stop' contract."""
    handle = _WatchHandle(query_result=True)
    agent = _agent(handle)
    await agent.cancel("s1")
    await _drain_watchdogs(agent)
    # Phase 1 went out cooperatively first...
    assert bridge_mod.AgentSession.cancel in handle.updates
    # ...and the escalation actually fired.
    assert handle.terminated, "watchdog must terminate a surviving turn"


async def test_unanswerable_query_terminates() -> None:
    """A wedged worker can't answer queries either — that must escalate
    to terminate, not be treated as 'turn probably ended'."""
    handle = _WatchHandle(query_result=RuntimeError("query wedged"))
    agent = _agent(handle)
    await agent.cancel("s1")
    await _drain_watchdogs(agent)
    assert handle.terminated, "query failure must fail closed toward terminate"


async def test_completed_cooperative_cancel_spares_session() -> None:
    """Normal path: cooperative cancel won (turn_active == False) — no
    terminate, the durable session and transcript survive."""
    handle = _WatchHandle(query_result=False)
    agent = _agent(handle)
    await agent.cancel("s1")
    await _drain_watchdogs(agent)
    assert handle.terminated == []
    assert bridge_mod.AgentSession.cancel in handle.updates


async def test_aclose_cancels_pending_watchdog() -> None:
    """Disconnect kills the watchdog: reattach durability beats a late
    terminate. The watchdog here would only fire after 60s (patched
    constant in real runs) — aclose must cancel it before it can kill."""
    handle = _WatchHandle(query_result=True)
    agent = _agent(handle)
    await agent.cancel("s1")
    assert agent._cancel_watchdogs, "watchdog task must be tracked"
    await agent.aclose()
    await asyncio.sleep(0.05)  # done-callbacks (set discard) run on a later tick
    assert not agent._cancel_watchdogs
    assert handle.terminated == []

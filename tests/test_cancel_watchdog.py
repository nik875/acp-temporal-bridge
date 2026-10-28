"""Cancel re-driver: interrupt kills ONLY the turn — never the session.

Regression for the 2026-10-28 production incident (supersedes the
2026-10-26 wedge contract): Paseo interrupted a turn during a worker
Recreate rollout gap. The v0.1.15 cancel watchdog saw the turn_active
query fail (no worker alive to answer it), failed "closed" toward
terminate, and TERMINATED the session workflow — destroying a 1211-event
transcript. The invariant is now the opposite: an interrupt kills only
the turn; the session and transcript ALWAYS survive.

Contract after the fix (bridge.py cancel() -> _cancel_redriver):

* cooperative cancel update still goes out first (and is re-sent with
  backoff until it lands — a Recreate rollout gap has NO worker to
  process it, and the retry recovers when a worker returns);
* turn_active query == False  -> stop retrying (cooperative won)
* turn_active query == True   -> re-send the update, keep retrying
* query raises RPCError "workflow execution already completed"
                              -> stop (workflow closed: turn ended too)
* query raises anything else  -> treated as "unknown", keep retrying
* total retry window bounded (CANCEL_RETRY_MAX_SECS): give up DRIVING
  the cancel loudly — but NEVER terminate the workflow;
* connection teardown (aclose) cancels pending re-drivers.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest
import temporalio.client as tc

from acp_temporal_bridge import bridge as bridge_mod
from acp_temporal_bridge.bridge import BridgeAgent


class _WatchHandle:
    """WorkflowHandle stand-in: scripted query answers, recorded updates."""

    def __init__(self, query_result: bool | Exception) -> None:
        self.query_result = query_result
        self.updates: list[Any] = []
        self.terminated: list[str] = []  # records any terminate ATTEMPT

    async def execute_update(self, *args: Any, **kwargs: Any) -> None:
        self.updates.append(args[0] if args else None)

    async def query(self, *args: Any, **kwargs: Any) -> bool:
        if isinstance(self.query_result, Exception):
            raise self.query_result
        return self.query_result

    async def terminate(self, *args: Any, **kwargs: Any) -> None:
        # _cancel_redriver must NEVER call this; record it so any call
        # fails the test loudly.
        reason = kwargs.get("reason")
        self.terminated.append(reason if isinstance(reason, str) else "")


class _WatchTemporal:
    def __init__(self, handle: _WatchHandle) -> None:
        self.handle = handle

    def get_workflow_handle(self, workflow_id: str) -> _WatchHandle:
        return self.handle


class _SilentConn:
    async def session_update(self, session_id: str, update: Any) -> None:  # pragma: no cover
        raise AssertionError("cancel re-driver must not stream updates")


def _agent(handle: _WatchHandle) -> BridgeAgent:
    return BridgeAgent(_SilentConn(), _WatchTemporal(handle))


async def _drain_watchdogs(agent: BridgeAgent) -> None:
    pending = list(agent._cancel_watchdogs)
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


@pytest.fixture(autouse=True)
def _fast_redriver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bridge_mod, "CANCEL_RETRY_AFTER_SECS", 0.01)
    monkeypatch.setattr(bridge_mod, "CANCEL_RETRY_BACKOFF_SECS", 0.01)
    monkeypatch.setattr(bridge_mod, "CANCEL_RETRY_MAX_SECS", 0.05)


async def test_alive_turn_is_retried_never_terminated() -> None:
    """Turn still active: the cancel update is RE-SENT, and no terminate
    is ever attempted — the session always survives the interrupt."""
    handle = _WatchHandle(query_result=True)
    agent = _agent(handle)
    await agent.cancel("s1")
    await _drain_watchdogs(agent)
    # The cooperative update went out (initial + re-sends)...
    assert handle.updates.count(bridge_mod.AgentSession.cancel) >= 2
    # ...and the terminate escalation is gone for good.
    assert handle.terminated == [], "interrupt must never terminate the session"


async def test_unanswerable_query_retries_not_terminates() -> None:
    """A worker gap makes the query unanswerable — the OLD code escalated
    to terminate here (the incident). Now: treat as unknown, keep
    re-sending the update, never terminate."""
    handle = _WatchHandle(query_result=RuntimeError("query wedged"))
    agent = _agent(handle)
    await agent.cancel("s1")
    await _drain_watchdogs(agent)
    assert handle.updates.count(bridge_mod.AgentSession.cancel) >= 2
    assert handle.terminated == []


async def test_closed_workflow_ends_retries() -> None:
    """Workflow already closed (idle expiry won the race): the re-driver
    exits immediately instead of retrying into the void."""
    handle = _WatchHandle(
        query_result=tc.RPCError(
            "workflow execution already completed",
            tc.RPCStatusCode.INTERNAL,
            -1,
        )
    )
    agent = _agent(handle)
    await agent.cancel("s1")
    await _drain_watchdogs(agent)
    # Only the initial update; the closed-workflow check stopped the loop.
    assert handle.updates.count(bridge_mod.AgentSession.cancel) == 1
    assert handle.terminated == []


async def test_completed_cooperative_cancel_stops_retrying() -> None:
    """Normal path: cooperative cancel won (turn_active == False) — the
    re-driver stops, session and transcript survive untouched."""
    handle = _WatchHandle(query_result=False)
    agent = _agent(handle)
    await agent.cancel("s1")
    await _drain_watchdogs(agent)
    assert handle.updates.count(bridge_mod.AgentSession.cancel) == 1
    assert handle.terminated == []


async def test_retry_budget_gives_up_loudly_not_deadly(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Exhausted retry window: the re-driver gives up DRIVING the cancel
    (loud log) — and still never terminates the session."""
    handle = _WatchHandle(query_result=True)
    agent = _agent(handle)
    with caplog.at_level(logging.WARNING, logger="acp_temporal_bridge.bridge"):
        await agent.cancel("s1")
        await _drain_watchdogs(agent)
    assert any("NOT terminated" in r.message for r in caplog.records)
    assert handle.terminated == []


async def test_aclose_cancels_pending_redriver() -> None:
    """Disconnect kills the re-driver: the session remains as durable as
    ever (and re-cancellable by any later cancel / reconnect)."""
    handle = _WatchHandle(query_result=True)
    agent = _agent(handle)
    await agent.cancel("s1")
    assert agent._cancel_watchdogs, "re-driver task must be tracked"
    await agent.aclose()
    await asyncio.sleep(0.05)  # done-callbacks (set discard) run on a later tick
    assert not agent._cancel_watchdogs
    assert handle.terminated == []

"""Closed-workflow resilience: a killed session must not brick.

Regression for the 2026-10-26 follow-on to the interrupt wedge: the
v0.1.15 watchdog correctly TERMINATED the wedged session workflow, but
afterwards EVERY prompt to that session failed with
"workflow execution already completed" (ACP internal_error -32603) —
Paseo kept the session, so it was bricked until the pod restarted.

Contract after the fix (v0.1.16):

* prompt() to a closed session   -> silently recreates the workflow under
  the SAME id (ALLOW_DUPLICATE), re-sends the prompt, streams normally
* prompt() RPC errors that are NOT closed-workflow -> still raise
* load_session to a closed session -> empty replay, no raise (the next
  prompt() recreates)
* cancel() to a closed session   -> fire-and-forget update exception is
  consumed (no "exception was never retrieved" noise), watchdog's own
  query/terminate failures suppressed
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest
from acp.schema import LoadSessionResponse, PromptResponse
from temporalio.client import RPCError
from temporalio.common import WorkflowIDReusePolicy
from temporalio.service import RPCStatusCode

from acp_temporal_bridge import bridge as bridge_mod
from acp_temporal_bridge.bridge import BridgeAgent
from acp_temporal_bridge.session import AgentSession


@pytest.fixture(autouse=True)
def _configure_spec() -> None:
    """Recreate needs a default spec (new_session needs one in prod too)."""
    from acp_temporal_bridge import SessionSpec, configure

    configure(default_spec=SessionSpec(model="mock", system_prompt="e2e", tools=["echo"]))


def _closed_rpc_error() -> RPCError:
    return RPCError(
        "workflow execution already completed",
        RPCStatusCode.UNKNOWN,
        b"",
    )


def _unrelated_rpc_error() -> RPCError:
    return RPCError("worker wedged", RPCStatusCode.UNAVAILABLE, b"")


@dataclass
class _Item:
    data: Any
    offset: int


class _FlakyHandle:
    """WorkflowHandle stand-in: first update raises a scripted error,
    queries and terminate always raise (a closed workflow answers
    nothing)."""

    def __init__(self, first_error: Exception | None = None) -> None:
        self.first_error = first_error
        self.updates: list[Any] = []
        self.terminated: list[str] = []

    async def execute_update(self, *args: Any, **kwargs: Any) -> None:
        if self.first_error is not None:
            err, self.first_error = self.first_error, None
            raise err
        self.updates.append(args[0] if args else None)

    async def query(self, *args: Any, **kwargs: Any) -> Any:
        raise _closed_rpc_error()

    async def describe(self) -> None:
        if isinstance(self.first_error, RPCError):
            raise self.first_error

    async def terminate(self, *args: Any, **kwargs: Any) -> None:
        self.terminated.append(kwargs.get("reason", ""))
        raise _closed_rpc_error()


class _RecordingTemporal:
    """get_workflow_handle + start_workflow, recording recreate calls."""

    def __init__(self, handle: _FlakyHandle | None = None) -> None:
        self.handle = handle
        self.starts: list[dict[str, Any]] = []

    def get_workflow_handle(self, workflow_id: str) -> _FlakyHandle:
        assert self.handle is not None
        return self.handle

    async def start_workflow(self, *args: Any, **kwargs: Any) -> None:
        self.starts.append(kwargs)


class _Conn:
    def __init__(self) -> None:
        self.updates: list[Any] = []

    async def session_update(self, session_id: str, update: Any) -> None:
        self.updates.append(update)


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


async def test_prompt_recreates_closed_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The -32603 brick: prompt a watchdog-killed session -> the workflow
    is recreated under the SAME id, the prompt re-sent, turn streams."""
    _use_stream(monkeypatch, [_Item({"kind": "turn_end", "stop_reason": "end_turn"}, 0)])
    handle = _FlakyHandle(first_error=_closed_rpc_error())
    temporal = _RecordingTemporal(handle)
    agent = BridgeAgent(_Conn(), temporal)
    resp = await agent.prompt("s1", [])
    assert isinstance(resp, PromptResponse)
    assert len(temporal.starts) == 1
    # ALLOW_DUPLICATE (not FAILED_ONLY): idle expiry COMPLETES the
    # workflow cleanly, so it must be recreatable too.
    assert (
        temporal.starts[0]["id_reuse_policy"] is WorkflowIDReusePolicy.ALLOW_DUPLICATE
    )
    # The prompt reached the NEW workflow (first raise consumed, re-sent).
    assert handle.updates == [AgentSession.prompt]


async def test_prompt_reraises_unrelated_rpc_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only closed-workflow errors trigger recreate; everything else is
    still a failure (a wedged worker must NOT silently respawn sessions)."""
    handle = _FlakyHandle(first_error=_unrelated_rpc_error())
    temporal = _RecordingTemporal(handle)
    agent = BridgeAgent(_Conn(), temporal)
    with pytest.raises(RPCError):
        await agent.prompt("s1", [])
    assert temporal.starts == []


async def test_prompt_surfaces_failure_after_recreate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the re-sent prompt fails on the FRESH workflow too, that error
    surfaces honestly — no fake completion, no infinite loop."""
    _use_stream(monkeypatch, [])
    handle = _FlakyHandle(first_error=_closed_rpc_error())
    temporal = _RecordingTemporal(handle)
    agent = BridgeAgent(_Conn(), temporal)
    orig = handle.execute_update

    async def always_closed(*args: Any, **kwargs: Any) -> None:
        await orig(*args, **kwargs)
        raise _closed_rpc_error()

    handle.execute_update = always_closed  # type: ignore[method-assign]
    with pytest.raises(RPCError):
        await agent.prompt("s1", [])
    assert len(temporal.starts) == 1  # exactly one recreate, no loop


async def test_recreate_resets_offsets_and_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recreate must reset stream bookkeeping for the new run: offsets 0
    (fresh stream) and any load_session tail (subscribed to the DEAD
    workflow) cancelled."""
    _use_stream(monkeypatch, [_Item({"kind": "turn_end", "stop_reason": "end_turn"}, 0)])
    handle = _FlakyHandle(first_error=_closed_rpc_error())
    temporal = _RecordingTemporal(handle)
    agent = BridgeAgent(_Conn(), temporal)
    agent._offsets["s1"] = 4096  # stale offset from the dead workflow
    tail = asyncio.create_task(asyncio.sleep(999))
    agent._tail_tasks["s1"] = tail
    await agent.prompt("s1", [])
    assert agent._offsets["s1"] == 1  # advanced by the NEW run's turn_end
    assert "s1" not in agent._tail_tasks
    assert tail.cancelling() > 0  # cancel requested (delivery is async)


async def test_load_session_closed_workflow_returns_empty_replay() -> None:
    """Reattaching to a killed session: empty replay, no raise — the
    next prompt() recreates. Before v0.1.16 this raised and the client
    could not even reattach."""
    handle = _FlakyHandle(first_error=_closed_rpc_error())
    agent = BridgeAgent(_Conn(), _RecordingTemporal(handle))
    resp = await agent.load_session("/tmp", "s1")
    assert isinstance(resp, LoadSessionResponse)
    assert agent._offsets["s1"] == 0
    assert "s1" not in agent._tail_tasks  # no tail on a dead workflow


async def test_cancel_closed_workflow_is_silent() -> None:
    """cancel() on a closed session: the fire-and-forget update exception
    is consumed and the watchdog's query+terminate failures suppressed —
    nothing escapes, no un-retrieved-exception warnings."""
    handle = _FlakyHandle(first_error=_closed_rpc_error())
    agent = BridgeAgent(_Conn(), _RecordingTemporal(handle))
    await agent.cancel("s1")
    # First tick: the update task raises (consumed by the done-callback).
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    # Drain the watchdog (CANCEL_KILL_AFTER_SECS is 5s in prod; here the
    # sleep dominates — cancel it directly and check the outcome fields).
    for task in list(agent._cancel_watchdogs):
        task.cancel()
    await asyncio.gather(*agent._cancel_watchdogs, return_exceptions=True)
    # terminate() raised -> suppressed, and the update task left no noise.
    assert handle.terminated == []
    assert handle.first_error is None  # update consumed the scripted error

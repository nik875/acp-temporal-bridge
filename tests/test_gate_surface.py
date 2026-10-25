"""Unit: _resolve_gate maps the ACP permission outcome onto resolve_approval."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock

import pytest
from acp.schema import (
    AllowedOutcome,
    DeniedOutcome,
    RequestPermissionResponse,
)

from acp_temporal_bridge.bridge import BridgeAgent

GATE = {"kind": "approval_request", "tool": "file_change_request", "args": {"path": "x"}, "description": ""}


@dataclass
class FakePending:
    tool: str = "file_change_request"
    args: dict = field(default_factory=lambda: {"path": "x"})
    description: str = ""


def _mk_bridge(permission_result, pending="UNSET"):
    if pending == "UNSET":
        pending = FakePending()
    agent = object.__new__(BridgeAgent)
    agent._client = MagicMock()
    agent._client.request_permission = AsyncMock(return_value=permission_result)
    handle = MagicMock()
    handle.query = AsyncMock(return_value=pending)
    handle.execute_update = AsyncMock(return_value=None)
    return agent, handle


def _resp(outcome):
    return RequestPermissionResponse(outcome=outcome)


@pytest.mark.asyncio
async def test_allow_outcome_resolves_approve():
    agent, handle = _mk_bridge(_resp(AllowedOutcome(outcome="selected", option_id="approve")))
    await BridgeAgent._resolve_gate(agent, "s1", handle, GATE)
    handle.execute_update.assert_awaited_once()
    args = handle.execute_update.await_args.args
    assert args[1] == {"action": "approve", "response": None}


@pytest.mark.asyncio
async def test_reject_option_resolves_respond():
    agent, handle = _mk_bridge(_resp(AllowedOutcome(outcome="selected", option_id="respond")))
    await BridgeAgent._resolve_gate(agent, "s1", handle, GATE)
    args = handle.execute_update.await_args.args
    assert args[1]["action"] == "respond"
    assert "respond" in args[1]["response"]


@pytest.mark.asyncio
async def test_cancelled_outcome_resolves_deny():
    agent, handle = _mk_bridge(_resp(DeniedOutcome(outcome="cancelled")))
    await BridgeAgent._resolve_gate(agent, "s1", handle, GATE)
    args = handle.execute_update.await_args.args
    assert args[1]["action"] == "respond"
    assert "denied" in args[1]["response"].lower()


@pytest.mark.asyncio
async def test_replayed_gate_is_skipped():
    # pending_approval returns None -> the gate was already answered in the
    # durable log (replay); the client must NOT be asked again.
    agent, handle = _mk_bridge(_resp(AllowedOutcome(outcome="selected", option_id="approve")), pending=None)
    await BridgeAgent._resolve_gate(agent, "s1", handle, GATE)
    agent._client.request_permission.assert_not_awaited()
    handle.execute_update.assert_not_awaited()


@pytest.mark.asyncio
async def test_different_pending_gate_is_skipped():
    # A NEWER gate is pending; this (older) request_event was already answered.
    newer = FakePending(tool="other_tool", args={})
    agent, handle = _mk_bridge(_resp(AllowedOutcome(outcome="selected", option_id="approve")), pending=newer)
    await BridgeAgent._resolve_gate(agent, "s1", handle, GATE)
    agent._client.request_permission.assert_not_awaited()

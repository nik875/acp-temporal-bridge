"""The explicit history-byte budget trips the mid-turn CAN guard.

Background (2026-10-28 wedge): a long-lived session's history reached 2MB
with NO CAN trip — the server's continue-as-new suggestion fires far above
that — and a cold-cache worker then had to replay the full history inside
one 10s workflow task. The replay took 25-28s, the task timed out
(TMPRL1104 loop), and no update/query/signal was processed until the cache
held the workflow again: a mid-turn steer's cancel sat unprocessed ~18s,
Paseo's interrupt-and-replace fallback raced the eventual turn completion,
and every later prompt died with "A foreground turn is already active".

Fix under test: `_check_budget` also CANs at a model-call boundary once
history passes `HISTORY_CAN_BUDGET_BYTES`, keeping cold-cache replays fast.
"""

from __future__ import annotations

from unittest import mock

import pytest


def test_budget_trips_mid_turn_can():
    from temporalio import workflow

    import acp_temporal_bridge.session as s

    class _FakeInfo:
        def __init__(self, suggested: bool, size: int):
            self._suggested, self._size = suggested, size

        def is_continue_as_new_suggested(self) -> bool:
            return self._suggested

        def get_current_history_size(self) -> int:
            return self._size

    model = s._CanGuardModel(model="m")
    # Over budget, server suggestion not fired: the explicit budget trips.
    with (
        mock.patch.object(
            workflow, "info", return_value=_FakeInfo(False, s.HISTORY_CAN_BUDGET_BYTES + 1)
        ),
        pytest.raises(s.MidTurnContinueAsNew),
    ):
        model._check_budget()
    # Exactly at budget: no trip.
    with mock.patch.object(
        workflow, "info", return_value=_FakeInfo(False, s.HISTORY_CAN_BUDGET_BYTES)
    ):
        model._check_budget()
    # Server suggestion still trips on its own.
    with (
        mock.patch.object(workflow, "info", return_value=_FakeInfo(True, 0)),
        pytest.raises(s.MidTurnContinueAsNew),
    ):
        model._check_budget()

"""Unit tests for the TMPRL1103 wedge guard: bounded continue-as-new state.

Background: an unbounded CAN fails the workflow task deterministically on
every retry, permanently wedging the session (observed in production
2026-10-26 — a 4.1MB stream log of per-token entries wedged session
15a7e2cea9a9). These helpers truncate the carried state to a budget.
"""

from __future__ import annotations

import base64
import json
from typing import Any

from acp_temporal_bridge.session import (
    _bound_transcript,
    _cap_json,
    _stream_state_with_tail,
)


def _wire(data: dict[str, Any]) -> Any:
    from temporalio.contrib.workflow_streams._types import _WorkflowStreamWireItem

    return _WorkflowStreamWireItem(topic="acp", data=base64.b64encode(json.dumps(data).encode()).decode())


def _tool_msg(content: str, call_id: str = "c1") -> dict[str, Any]:
    return {"type": "tool", "data": {"content": content, "tool_call_id": call_id}}


class TestBoundTranscript:
    def test_caps_long_tool_output(self):
        t = [_tool_msg("x" * 20_000)]
        out = _bound_transcript(t, output_cap=8000)
        content = out[0]["data"]["content"]
        assert len(content) < 20_000
        assert content.startswith("x" * 100)
        assert "truncated" in content

    def test_leaves_short_outputs_verbatim(self):
        t = [_tool_msg("short")]
        assert _bound_transcript(t) == t

    def test_leaves_non_tool_and_malformed_entries(self):
        t = [
            {"type": "human", "data": {"content": "x" * 20_000}},
            "not-a-dict",
            {"type": "tool", "data": "weird"},
            None,
        ]
        assert _bound_transcript(t) == t

    def test_does_not_mutate_input(self):
        t = [_tool_msg("x" * 20_000)]
        _bound_transcript(t)
        assert len(t[0]["data"]["content"]) == 20_000


class TestStreamStateWithTail:
    def _state(self, n: int, base: int = 100):
        from temporalio.contrib.workflow_streams import WorkflowStreamState

        return WorkflowStreamState(
            log=[_wire({"i": i}) for i in range(n)],
            base_offset=base,
            publishers={},
        )

    def test_keeps_newest_and_advances_base_offset(self):
        s = self._state(10, base=100)
        out = _stream_state_with_tail(s, keep=3)
        assert len(out.log) == 3
        assert out.base_offset == 107
        # newest entries survive
        assert json.loads(base64.b64decode(out.log[-1].data))["i"] == 9

    def test_offsets_stay_monotonic(self):
        s = self._state(10, base=100)
        out = _stream_state_with_tail(s, keep=3)
        # old log[7] had offset 107; new first entry must still be 107
        assert out.base_offset + 0 == 107

    def test_keep_larger_than_log_is_noop(self):
        s = self._state(5, base=0)
        assert _stream_state_with_tail(s, keep=10) is s


class TestCapJson:
    def test_small_value_untouched(self):
        assert _cap_json({"a": 1}, 2000) == {"a": 1}

    def test_big_value_replaced_with_marker(self):
        out = _cap_json({"a": "x" * 5000}, 1000)
        assert out["truncated_to"] == 1000
        assert len(out["head"]) == 1000

    def test_unserializable_value_handled(self):
        value = {1, 2}  # sets don't JSON-serialize; str() fallback is small
        assert _cap_json(value, 10) is value

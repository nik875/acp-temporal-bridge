"""Approval-gate types + interrupt parsing for the session workflow.

Hosts that want human-in-the-loop gates supply an ``interrupt_compiler`` via
:func:`acp_temporal_bridge.configure`; when a gated tool call pauses the
graph, the parsed gate surfaces through the workflow's ``pending_approval``
query and the host resolves it with ``resolve_approval``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ._config import get_interrupt_compiler


@dataclass(frozen=True)
class ApprovalDecision:
    """A human's resolution of a pending approval, delivered via workflow update."""

    action: str  # "approve" | "respond"
    response: str | None = None  # deny reason; fed back to the model as tool error

    def to_resume_payload(self) -> dict[str, Any]:
        """LangGraph-native resume protocol: Command(resume={"decisions": [...]}).

        Decision schemas per langchain's HITL middleware: approve = {"type":
        "approve"}; respond = {"type": "respond", "message": <tool answer>} —
        the tool does not run and the message returns to the model as a
        synthetic successful ToolMessage it can route around.
        """
        decision: dict[str, Any] = {"type": self.action}
        if self.response is not None:
            decision["message"] = self.response
        return {"decisions": [decision]}


@dataclass(frozen=True)
class PendingApproval:
    """What the workflow exposes to approvers via the pending_approval query."""

    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    description: str = ""


def build_interrupt_on(tool_names: list[str]) -> dict[str, bool]:
    """Compile the host's policy into DeepAgents' ``interrupt_on`` map."""
    return get_interrupt_compiler()(tool_names)


def pending_from_interrupt(value: Any) -> PendingApproval:
    """Tolerant extractor for LangGraph HITL interrupt values.

    Observed payload: Interrupt(value={'action_requests': [{'name': ...,
    'args': {...}, 'description': ...}], 'review_configs': [...]}) — parsed
    here; unknown shapes degrade to a stringified fallback so the approver
    always sees something.
    """
    inner = getattr(value, "value", value)
    if isinstance(inner, dict) and inner.get("action_requests"):
        first = inner["action_requests"][0]
        args = first.get("args") if isinstance(first, dict) else None
        return PendingApproval(
            tool=first.get("name", "unknown") if isinstance(first, dict) else "unknown",
            args=args if isinstance(args, dict) else {},
            description=first.get("description", "") or str(inner)[:2000],
        )
    request = getattr(inner, "action_request", None)
    if request is not None:
        args = getattr(request, "args", None)
        return PendingApproval(
            tool=getattr(request, "action", None)
            or getattr(request, "name", "unknown"),
            args=dict(args) if isinstance(args, dict) else {},
            description=str(inner)[:2000],
        )
    if isinstance(inner, dict):
        return PendingApproval(
            tool=str(inner.get("name", inner.get("action", "unknown"))),
            args=inner.get("args") if isinstance(inner.get("args"), dict) else {},
            description=str(inner)[:2000],
        )
    return PendingApproval(tool="unknown", args={}, description=str(inner)[:2000])

"""acp-temporal-bridge: Agent Client Protocol bridge for durable agent
sessions on Temporal.

One Temporal workflow per client session (DeepAgents loop embedded in the
workflow), a small TCP bridge speaking ACP, and a merge-watcher that wakes
sessions when their PRs land. See README.md for the setup contract
(:func:`configure` must be called by the host at import time).
"""

from ._config import SessionSpec, configure
from .bridge import BridgeAgent, main, serve, translate_stream_item
from .policy import ApprovalDecision, PendingApproval, build_interrupt_on
from .session import (
    AgentSession,
    PrEvent,
    PromptIn,
    SessionStart,
    session_workflow_id,
)
from .watcher import PrWatcher, fetch_merged_prs

__all__ = [
    "AgentSession",
    "ApprovalDecision",
    "BridgeAgent",
    "PendingApproval",
    "PrEvent",
    "PrWatcher",
    "PromptIn",
    "SessionSpec",
    "SessionStart",
    "build_interrupt_on",
    "configure",
    "fetch_merged_prs",
    "main",
    "serve",
    "session_workflow_id",
    "translate_stream_item",
]

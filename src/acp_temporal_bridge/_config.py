"""Host configuration seam.

The bridge package is host-agnostic: the host application (the thing that
embeds Temporal + DeepAgents) calls :func:`configure` once at import time to
plug in its tool builder, approval-policy compiler, and default agent spec.
Everything configured here must be deterministic — workflow replay re-imports
this package, so the same values must be configured in the replayed process.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

TOOL_BUILDER = Callable[[list[str]], list[Any]]
"""names -> LangChain tools (workflow-side wrappers; bodies run as activities)."""

INTERRUPT_COMPILER = Callable[[list[str]], dict[str, bool]]
"""names -> {tool_name: should_interrupt} map for DeepAgents interrupt_on."""

BACKEND_BUILDER = Callable[[], Any]
"""() -> BackendProtocol | None — the filesystem backend for the DeepAgents
built-in tools (ls/read_file/write_file/edit_file/glob/grep).

Called inside the session workflow. In-workflow construction is correct for a
``TemporalBackend``-wrapped backend: the wrapper registers its inner backend
worker-side at build time and every real-I/O op dispatches out as a durable
``deepagents.backend_op`` activity. Return None (or omit the seam) to keep
deepagents' default ``StateBackend`` — files live in workflow state, not on
any disk. Must be deterministic under replay."""


@dataclass
class SessionSpec:
    """The full agent spec for a session — carried in workflow input.

    No registry lookups at run time: the workflow sandbox re-imports the
    package, so a by-name lookup inside the run would be a second, divergent
    world. Carrying the spec in the input makes history self-contained as an
    audit trail.
    """

    model: str
    system_prompt: str
    tools: list[str] = field(default_factory=list)
    task: str = ""


_tool_builder: TOOL_BUILDER | None = None
_interrupt_compiler: INTERRUPT_COMPILER | None = None
_backend_builder: BACKEND_BUILDER | None = None
_default_spec: SessionSpec | None = None


def configure(
    *,
    tool_builder: TOOL_BUILDER | None = None,
    interrupt_compiler: INTERRUPT_COMPILER | None = None,
    backend_builder: BACKEND_BUILDER | None = None,
    default_spec: SessionSpec | None = None,
) -> None:
    """Set the host seams. Call once at import time (see module docstring)."""
    from . import _delta

    _delta.install()  # idempotent: model-call input delta protocol (Phase 1b)
    global _tool_builder, _interrupt_compiler, _backend_builder, _default_spec
    if tool_builder is not None:
        _tool_builder = tool_builder
    if interrupt_compiler is not None:
        _interrupt_compiler = interrupt_compiler
    if backend_builder is not None:
        _backend_builder = backend_builder
    if default_spec is not None:
        _default_spec = default_spec


def get_backend_builder() -> BACKEND_BUILDER | None:
    """Optional seam: None means "use deepagents' default (StateBackend)"."""
    return _backend_builder


def get_tool_builder() -> TOOL_BUILDER:
    if _tool_builder is None:
        raise RuntimeError(
            "acp_temporal_bridge.configure(tool_builder=...) was never called — "
            "sessions cannot resolve their tools."
        )
    return _tool_builder


def get_interrupt_compiler() -> INTERRUPT_COMPILER:
    if _interrupt_compiler is None:
        return _default_interrupt_compiler
    return _interrupt_compiler


def get_default_spec() -> SessionSpec:
    if _default_spec is None:
        raise RuntimeError(
            "acp_temporal_bridge.configure(default_spec=...) was never called — "
            "the bridge has no agent spec for new sessions."
        )
    return _default_spec


def _default_interrupt_compiler(tool_names: list[str]) -> dict[str, bool]:
    # Permissive default: the package ships no policy; the host supplies one
    # when it wants HITL gates.
    return {name: False for name in tool_names}

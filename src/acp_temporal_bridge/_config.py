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
    # Hard token threshold for compaction. None keeps deepagents' default
    # auto-compaction. Registry-sourced (host: AGENTS[name].compact_at_tokens);
    # a default so pre-existing histories (no field) replay unchanged.
    compact_at_tokens: int | None = None


_tool_builder: TOOL_BUILDER | None = None
_interrupt_compiler: INTERRUPT_COMPILER | None = None
_backend_builder: BACKEND_BUILDER | None = None
_default_spec: SessionSpec | None = None
# Memory files injected into the system prompt every turn (host seam). Empty
# means "no memory middleware" — deepagents' default. Configured at host import
# time, so replay re-creates the same value.
_memory_sources: list[str] = []


def configure(
    *,
    tool_builder: TOOL_BUILDER | None = None,
    interrupt_compiler: INTERRUPT_COMPILER | None = None,
    backend_builder: BACKEND_BUILDER | None = None,
    default_spec: SessionSpec | None = None,
    memory_sources: list[str] | None = None,
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
    if memory_sources is not None:
        set_memory_sources(memory_sources)


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


def set_memory_sources(sources: list[str] | None) -> None:
    """Set the memory files injected each turn (host seam, import-time)."""
    global _memory_sources
    _memory_sources = list(sources or [])


def get_memory_sources() -> list[str] | None:
    """Memory files for ``create_deep_agent(memory=...)``.

    ``None`` (no sources configured) keeps deepagents' default — no memory
    middleware — rather than installing an empty one.
    """
    return list(_memory_sources) or None


def _default_interrupt_compiler(tool_names: list[str]) -> dict[str, bool]:
    # Permissive default: the package ships no policy; the host supplies one
    # when it wants HITL gates.
    return {name: False for name in tool_names}

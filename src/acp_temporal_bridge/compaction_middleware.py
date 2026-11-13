"""Shared compaction-middleware factory — ONE behavior on BOTH paths.

Background (2026-10-26): a compaction fix shipped claiming to solve the
sysadmin agent's per-turn context evaporation, but it only touched the
headless DurableAgent path. The interactive AgentSession path (what Paseo
actually drives) never saw it and the bug shipped anyway. This module is the
single construction site so a future fix cannot land on one path only; the
``tests/test_path_parity.py`` guard inspects both call sites' source.

Why the module is itself CALLABLE
---------------------------------
The parity guard does ``pytest.importorskip("acp_temporal_bridge."
"compaction_middleware")`` and then CALLS the returned object. A plain module
is not callable, so the module object's class is swapped for one with
``__call__`` forwarding to the function below. That keeps the dotted import
form working while callers can also just ``from .compaction_middleware import
compaction_middleware`` (the function). Both resolve to the same behavior.

Replay safety
-------------
deepagents is imported lazily inside the function: the workflow sandbox must
reuse the main-process module objects (the plugin's ``SummarizationMiddleware``
patch wraps a *string* model in ``TemporalModel``), so the import must happen
at call time, in-workflow, not at sandbox module-import time.
"""

from __future__ import annotations

import sys
import types
from typing import Any

__all__ = ["compaction_middleware"]


def compaction_middleware(
    model: Any, backend: Any, trigger: int | None
) -> list[Any]:
    """Build the compaction middleware list for one agent build.

    Args:
        model: the model the summarizer call should use. Already wrapped by
            the caller (e.g. the bridge's guard model), so the plugin's
            string-model patch is a no-op passthrough here.
        backend: the DeepAgents backend whose store offloaded history lands
            in — MUST be the same store the file tools see, so a compacted
            turn's history is still reachable with ``read_file``. ``None``
            falls back to ``StateBackend`` (deepagents' own default when no
            backend is configured).
        trigger: hard token threshold, or ``None`` to keep deepagents'
            default auto-compaction behavior.

    Returns:
        ``[]`` when ``trigger is None`` (deepagents installs its own default
        when no middleware replaces the slot), otherwise a one-element list
        carrying a ``SummarizationMiddleware``. A one-element list with the
        default name ``"SummarizationMiddleware"`` makes deepagents swap it
        into the default slot (``graph.py::_apply_custom_middleware``).
    """
    if trigger is None:
        return []

    # Lazy: see module docstring (in-workflow import for the plugin patch).
    from deepagents.backends import StateBackend
    from deepagents.middleware.summarization import SummarizationMiddleware

    return [
        SummarizationMiddleware(
            model=model,
            backend=backend if backend is not None else StateBackend(),
            trigger=("tokens", trigger),
        )
    ]


class _CallableModule(types.ModuleType):
    """Module that also forwards ``module(...)`` to ``compaction_middleware``."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return compaction_middleware(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule

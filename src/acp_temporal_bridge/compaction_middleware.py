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

Event reporting
---------------
Compaction is otherwise invisible: the summarizer model call happens inside
``awrap_model_call`` (not on the client's token stream), so a turn silently
grows a "where did the context go" hole. Passing ``on_event`` makes the
middleware report the two edges of a compaction — ``{"phase": "start", ...}``
before the summarizer runs and ``{"phase": "end", ...}`` when it returns (or
raises) — so a host can render it as a first-class action, exactly like a
tool call or a thought. Both events carry a stable ``id`` for the pair, plus
the number of messages summarized and the trigger threshold; the ``end``
event carries the generated ``summary`` (or the error class name).

The sink may be sync or async (``None`` disables reporting and keeps the
stock ``SummarizationMiddleware``). Reporting is asynchronous-path only:
both hosts (the local chat agent and the Temporal bridge) drive the agent
through ``astream``.

Replay safety
-------------
deepagents is imported lazily inside the function: the workflow sandbox must
reuse the main-process module objects (the plugin's ``SummarizationMiddleware``
patch wraps a *string* model in ``TemporalModel``), so the import must happen
at call time, in-workflow, not at sandbox module-import time.
"""

from __future__ import annotations

import inspect
import sys
import types
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    EventSink = Callable[[dict[str, Any]], Awaitable[None] | None]

__all__ = ["compaction_middleware"]


async def _report(sink: EventSink, event: dict[str, Any]) -> None:
    """Hand one event to the sink, awaiting it only when it is async."""
    result = sink(event)
    if inspect.isawaitable(result):
        await result


def compaction_middleware(
    model: Any, backend: Any, trigger: int | None, on_event: EventSink | None = None
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
        on_event: optional sink notified at the two edges of every compaction
            (see the module docstring). ``None`` keeps the stock middleware.

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
    from langchain_core.messages.utils import count_tokens_approximately

    class _EventedSummarizationMiddleware(SummarizationMiddleware):
        """SummarizationMiddleware that reports compaction edges to a sink.

        Defined inside the factory for the same lazy-import reason as the
        classes above: it subclasses the deepagents middleware, so it must be
        built from the main-process module object the plugin patched.
        """

        @property
        def name(self) -> str:
            # Keep the public alias: the default-slot swap keys on the name,
            # and a subclass would otherwise be appended as a SECOND
            # summarizer next to deepagents' own default.
            return "SummarizationMiddleware"

        async def _acreate_summary(self, messages: list[Any]) -> str:
            event_id = f"compact-{uuid.uuid4().hex[:12]}"
            base = {
                "id": event_id,
                "messages": len(messages),
                "tokens": count_tokens_approximately(messages),
                "trigger": trigger,
            }
            if on_event is not None:
                await _report(on_event, {"phase": "start", **base})
            try:
                summary = await super()._acreate_summary(messages)
            except BaseException as exc:
                if on_event is not None:
                    await _report(
                        on_event,
                        {"phase": "end", **base, "summary": "", "error": type(exc).__name__},
                    )
                raise
            if on_event is not None:
                await _report(on_event, {"phase": "end", **base, "summary": summary, "error": None})
            return summary

    return [
        _EventedSummarizationMiddleware(
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

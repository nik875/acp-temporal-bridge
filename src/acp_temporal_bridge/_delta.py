"""Bridge-local delta protocol for model-call activity inputs (Phase 1b).

The 2026-10-26 postmortem (session sysadmin-61be2651056d, terminated at
71.19 MB / 1277 events) showed 96% of history was 145 growing
``invoke_model_streaming`` activity inputs — every model call re-embeds the
full serialized conversation, quadratic per turn. Phase 1a (mid-turn CAN)
bounds the blast radius; this module shrinks the inputs themselves so CANs
become rare.

Design — two seams, both bridge-local, no library fork:

* Workflow side: ``temporalio.deepagents.workflow.call_model`` is imported
  at call time by ``TemporalModel._agenerate/_astream``, so a module-level
  wrapper intercepts every model-activity dispatch. The wrapper replaces the
  full serialized message list with a single content-addressed marker:
  ``[{DELTA_KEY: {prefix_sha256, prefix_len, delta, full_sha256}}]``. The
  delta base is the *previous* call's full list when it is a strict prefix
  of the current one (the conversation grows append-only within a turn).
* Activity side: ``DeepAgentsPlugin`` constructs its activity object from
  the name ``temporalio.deepagents._plugin.DeepAgentActivities``; a subclass
  installed there expands the marker back to the full message list before
  delegating to the library implementation (heartbeats, provider call,
  streaming — all unchanged).

Durability without a cache store: the cache is a small worker-process LRU
keyed by content hash. A miss (worker restart, pod reschedule, cold replay)
is always recoverable — the activity raises a NON-RETRYABLE
``bridge:delta_cache_miss`` ApplicationError, the workflow-side wrapper
catches it and re-dispatches the full input. Worst case is exactly today's
behavior plus one tiny failed attempt; the common case records one marker
(~1 KB) instead of ~800 KB per call.

Determinism: the shrink decision reads the process cache (nondeterministic
across task-fit boundaries), but replay matches recorded activity events by
sequence, not input bytes; the recorded input is authoritative on
re-execution. Old histories recorded before this feature are protected by
the ``workflow.patched("bridge.model-input-delta")`` gate: executions that
started before the patch marker exists take the unshrunk path, byte-for-byte
identical to pre-1b behavior.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from collections import OrderedDict
from typing import Any

from temporalio import workflow as _workflow
from temporalio.deepagents._activity import (
    INVOKE_MODEL,
    INVOKE_MODEL_STREAMING,
    DeepAgentActivities,
    ModelActivityInput,
    ModelActivityOutput,
)
from temporalio.exceptions import ActivityError, ApplicationError

logger = logging.getLogger(__name__)

DELTA_PATCH_ID = "bridge.model-input-delta"
"""Patch-gate id: executions started before this marker exist replay the
unshrunk path (see module docstring — replay safety for old histories)."""

DELTA_MISS_ERR = "bridge:delta_cache_miss"
"""ApplicationError type raised worker-side when a marker references a base
not in the process cache. Non-retryable: the workflow wrapper re-dispatches
the full input instead."""

DELTA_KEY = "__bridge_delta__"
"""Marker key inside a one-element ``messages`` list."""

DELTA_MIN_BYTES = 32_000
"""Shrink only model-call inputs whose serialized payload is at least this
large — small calls gain nothing from the marker round-trip."""

_CACHE_MAX_ENTRIES = 8
"""Content-addressed full-payload LRU (per entry ≈ one model-call input)."""

_CHAIN_MAX_SESSIONS = 64
"""Per-workflow prefix-chain LRU (one entry per live session)."""


class _DeltaCache:
    """Worker-process content cache: digest → serialized full message list.

    Shared by both sides of the protocol (workflow shrink and activity
    expand run in the same process). Bounded: _CACHE_MAX_ENTRIES payloads
    plus per-session chain pointers. Everything is content-addressed, so
    cross-session sharing is safe by construction.
    """

    def __init__(self) -> None:
        self._payloads: OrderedDict[str, list[Any]] = OrderedDict()
        self._chains: OrderedDict[str, tuple[str, list[Any]]] = OrderedDict()

    def put(self, digest: str, payload: list[Any]) -> None:
        self._payloads[digest] = payload
        self._payloads.move_to_end(digest)
        while len(self._payloads) > _CACHE_MAX_ENTRIES:
            self._payloads.popitem(last=False)

    def get(self, digest: str) -> list[Any] | None:
        payload = self._payloads.get(digest)
        if payload is not None:
            self._payloads.move_to_end(digest)
        return payload

    def set_chain(self, workflow_id: str, digest: str, payload: list[Any]) -> None:
        self._chains[workflow_id] = (digest, payload)
        self._chains.move_to_end(workflow_id)
        while len(self._chains) > _CHAIN_MAX_SESSIONS:
            self._chains.popitem(last=False)

    def get_chain(self, workflow_id: str) -> tuple[str, list[Any]] | None:
        entry = self._chains.get(workflow_id)
        if entry is not None:
            self._chains.move_to_end(workflow_id)
        return entry


_CACHE = _DeltaCache()


def _digest(payload: Any) -> tuple[str, int]:
    """Canonical digest of a serialized payload, with its byte length."""
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest(), len(blob)


def shrink(workflow_id: str, messages: list[Any]) -> list[Any] | None:
    """Return a one-element marker list, or None to send ``messages`` as-is.

    A marker is emitted only when the session's previous call payload is a
    strict prefix of the current one and the current payload clears
    DELTA_MIN_BYTES. Either way the current payload is cached for the next
    call.
    """
    digest, nbytes = _digest(messages)
    _CACHE.put(digest, messages)
    if nbytes < DELTA_MIN_BYTES:
        return None
    chain = _CACHE.get_chain(workflow_id)
    if chain is None:
        _CACHE.set_chain(workflow_id, digest, messages)
        return None
    base_digest, base = chain
    if len(messages) > len(base) and messages[: len(base)] == base:
        marker = {
            DELTA_KEY: {
                "v": 1,
                "prefix_sha256": base_digest,
                "prefix_len": len(base),
                "delta": messages[len(base) :],
                "full_sha256": digest,
            }
        }
        _CACHE.set_chain(workflow_id, digest, messages)
        return [marker]
    _CACHE.set_chain(workflow_id, digest, messages)
    return None


def expand(messages: list[Any]) -> list[Any]:
    """Worker-side inverse of :func:`shrink`. Raises ApplicationError on a
    cache miss (non-retryable — the workflow re-dispatches the full input).
    """
    if not (len(messages) == 1 and isinstance(messages[0], dict) and DELTA_KEY in messages[0]):
        return messages
    marker = messages[0][DELTA_KEY]
    base = _CACHE.get(marker["prefix_sha256"])
    if base is None or len(base) != marker["prefix_len"]:
        raise ApplicationError(
            "delta base not in worker cache; workflow will re-dispatch full input",
            non_retryable=True,
            type=DELTA_MISS_ERR,
        )
    full = [*base, *marker["delta"]]
    digest, _ = _digest(full)
    if digest != marker["full_sha256"]:
        raise ApplicationError(
            "reconstructed payload digest mismatch",
            non_retryable=True,
            type=DELTA_MISS_ERR,
        )
    _CACHE.put(digest, full)
    return full


def install() -> None:
    """Patch both seams. Idempotent; called from configure() at import time.

    Both seams are main-process module objects (temporalio modules pass
    through the workflow sandbox), so a single install in the host process
    covers workflow dispatch, activity execution, and replay.
    """
    import temporalio.deepagents._activity as _activity_mod
    import temporalio.deepagents._plugin as _plugin_mod
    import temporalio.deepagents.workflow as _wf_mod

    model_activity_names = {INVOKE_MODEL, INVOKE_MODEL_STREAMING}

    if not getattr(_wf_mod.call_model, "_bridge_delta_patch", False):
        orig_call_model = _wf_mod.call_model

        async def call_model(activity_name, activity_input, *, summary, **opts):
            if activity_name in model_activity_names and _workflow.patched(DELTA_PATCH_ID):
                marker = shrink(_workflow.info().workflow_id, activity_input.messages)
                if marker is not None:
                    shrunk = dataclasses.replace(activity_input, messages=marker)
                    try:
                        return await orig_call_model(activity_name, shrunk, summary=summary, **opts)
                    except (ApplicationError, ActivityError) as exc:
                        err = exc
                        if isinstance(err, ActivityError):
                            err = err.cause
                        if not (isinstance(err, ApplicationError) and err.type == DELTA_MISS_ERR):
                            raise
                        logger.warning(
                            "delta cache miss on %s — re-dispatching full input",
                            activity_name,
                        )
            return await orig_call_model(activity_name, activity_input, summary=summary, **opts)

        call_model._bridge_delta_patch = True  # type: ignore[attr-defined]
        _wf_mod.call_model = call_model

    if getattr(_plugin_mod.DeepAgentActivities, "_bridge_delta_patch", False):
        return

    class _PatchedActivities(DeepAgentActivities):
        """Expands delta markers before delegating to the library activities.

        The @activity.defn names are identical to the library's — the plugin
        registers ``self._activities.<method>`` bound methods, so these
        shadows take over registration while the originals keep running
        underneath via super() (heartbeats, provider call, streaming all
        unchanged).
        """

        @_activity_mod.activity.defn(name=INVOKE_MODEL)
        async def invoke_model(self, input: ModelActivityInput) -> ModelActivityOutput:
            input.messages = expand(input.messages)
            return await super().invoke_model(input)

        @_activity_mod.activity.defn(name=INVOKE_MODEL_STREAMING)
        async def invoke_model_streaming(self, input: ModelActivityInput) -> ModelActivityOutput:
            input.messages = expand(input.messages)
            return await super().invoke_model_streaming(input)

    _PatchedActivities._bridge_delta_patch = True  # type: ignore[attr-defined]
    _plugin_mod.DeepAgentActivities = _PatchedActivities

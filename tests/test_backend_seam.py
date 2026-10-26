"""Unit: the backend_builder seam reaches create_deep_agent's backend=.

Verifies (1) configure(backend_builder=...) is stored and returned, (2) with
no seam configured the lookup returns None (StateBackend default preserved),
and (3) a seam returning a real TemporalBackend(FilesystemBackend) builds and
presents the BackendProtocol surface. Per-op activity dispatch is covered by
the plugin's own backend tests; here we pin the seam contract.
"""

from __future__ import annotations

import acp_temporal_bridge._config as cfg
from acp_temporal_bridge import configure
from acp_temporal_bridge._config import get_backend_builder


def _reset() -> None:
    # configure() only overwrites non-None args, so reset by hand.
    cfg._backend_builder = None


def test_no_seam_returns_none():
    _reset()
    assert get_backend_builder() is None


def test_seam_roundtrip():
    _reset()

    def _build():
        return None  # explicit None == keep the default

    configure(backend_builder=_build)
    assert get_backend_builder() is _build


def test_seam_builds_temporal_backend():
    _reset()
    from deepagents.backends.filesystem import FilesystemBackend
    from temporalio.deepagents import TemporalBackend

    def _build():
        return TemporalBackend(FilesystemBackend(root_dir="/tmp"))

    configure(backend_builder=_build)
    built = get_backend_builder()()
    assert isinstance(built, TemporalBackend)
    assert hasattr(built, "read") and hasattr(built, "write")
    assert hasattr(built, "grep") and hasattr(built, "glob")

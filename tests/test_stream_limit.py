"""Regression: the TCP reader must accept an attachment-sized frame.

``asyncio.start_server`` defaults to a 64 KiB StreamReader limit. An ACP prompt
carrying one base64 image is a *single* NDJSON line far larger than that, so
``readline()`` died with "Separator is not found, and chunk exceed the limit"
and the client saw the agent as unresponsive ("chat stopped answering when I
attach an image"). The server now passes ``limit=ACP_STREAM_LIMIT`` (ACP's own
50 MiB stdio bound); these guards keep it wired.
"""

from __future__ import annotations

import asyncio

import pytest

from acp_temporal_bridge.bridge import ACP_STREAM_LIMIT


def test_limit_is_far_above_the_asyncio_default() -> None:
    # The asyncio default (64 KiB) is the bug; anything >= a few MiB is fine.
    assert ACP_STREAM_LIMIT > 8 * 1024 * 1024


@pytest.mark.asyncio
async def test_oversized_frame_survives_the_reader() -> None:
    seen: dict[str, int] = {}

    async def server(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        line = await reader.readline()
        seen["n"] = len(line)
        writer.write(b"ok\n")
        await writer.drain()
        writer.close()

    srv = await asyncio.start_server(server, "127.0.0.1", 0, limit=ACP_STREAM_LIMIT)
    port = srv.sockets[0].getsockname()[1]
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            writer.write(b"x" * 300_000 + b"\n")  # 300 KiB > the 64 KiB default
            await writer.drain()
            assert await asyncio.wait_for(reader.readline(), 5) == b"ok\n"
            assert seen["n"] == 300_001
        finally:
            writer.close()
    finally:
        srv.close()
        await srv.wait_closed()

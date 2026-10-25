"""Regression: the TCP serve loop must not double-read the socket.

The v0.1.0 bug: AgentSideConnection's constructor starts the receive loop
when listening defaults True; _handle_connection then called listen()
anyway -> two coroutines in readuntil() on the same reader. The losing
loop dies as a background task ("Background task failed" /
"readuntil() called while another coroutine is already waiting") and the
connection degrades — the first prompt through the deployed bridge
returned RequestError: Internal error.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from acp.client.connection import ClientSideConnection
from acp.schema import ClientCapabilities, Implementation, InitializeResponse

from acp_temporal_bridge.bridge import _handle_connection


class TinyClient:
    async def session_update(self, params):
        pass

    async def request_permission(self, params):
        raise AssertionError("not expected")

    async def write_text_file(self, params):
        raise AssertionError("not expected")

    async def read_text_file(self, params):
        raise AssertionError("not expected")

    async def session_notification(self, params):
        pass


@pytest.mark.asyncio
async def test_initialize_over_real_socket(caplog):
    temporal = MagicMock()  # initialize never touches Temporal

    async def server(client_reader, client_writer):
        await _handle_connection(temporal, client_reader, client_writer)

    srv = await asyncio.start_server(server, "127.0.0.1", 0)
    port = srv.sockets[0].getsockname()[1]

    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    client = ClientSideConnection(TinyClient(), writer, reader)

    resp: InitializeResponse = await asyncio.wait_for(
        client.initialize(
            protocol_version=1,
            client_info=Implementation(name="regression", version="0"),
            client_capabilities=ClientCapabilities(),
        ),
        timeout=10,
    )
    assert resp.protocol_version == 1

    # The double-read bug does NOT fail the first request: the losing
    # receive loop dies as a background task. Give it a beat to collide,
    # then assert it never did.
    await asyncio.sleep(0.5)
    collisions = [
        r
        for r in caplog.records
        if "already waiting for incoming data"
        in (r.getMessage() + str(r.exc_info if r.exc_info else ""))
        or (r.exc_info and "already waiting for incoming data" in str(r.exc_info[1]))
    ]
    assert not collisions, (
        "a second receive loop is reading the same socket "
        "(AgentSideConnection constructed with listening=True + explicit listen())"
    )

    writer.close()
    srv.close()
    await srv.wait_closed()

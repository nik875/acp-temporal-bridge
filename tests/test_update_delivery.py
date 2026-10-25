"""Regression: agent-side session_update must reach the client over TCP.

Companion to test_tcp_serve: initialize proved the receive path; this
proves the SEND path (bridge -> client notifications) actually delivers
AgentMessageChunk updates. The deployed v0.1.1 bridge crashed its send
loop with "cannot reuse already awaited coroutine" while emitting model
deltas, so the client saw only the prompt response and zero chunks.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from acp.client.connection import ClientSideConnection
from acp.schema import (
    AgentMessageChunk,
    ClientCapabilities,
    Implementation,
    InitializeResponse,
    TextContentBlock,
)

import acp_temporal_bridge.bridge as bridge_mod
from acp_temporal_bridge.bridge import _handle_connection


class RecordingClient:
    def __init__(self):
        self.updates: list = []

    async def session_update(self, session_id, update, **kwargs):
        # Signature must match acp's Client protocol: the router calls
        # handlers keyword-style (session_id=, update=) for underscore
        # attributes, and _run_notification suppresses any TypeError —
        # a (params) signature makes updates vanish silently.
        self.updates.append(update)

    async def request_permission(self, params):
        raise AssertionError("not expected")

    async def write_text_file(self, params):
        raise AssertionError("not expected")

    async def read_text_file(self, params):
        raise AssertionError("not expected")

    async def session_notification(self, params):
        pass


@pytest.mark.asyncio
async def test_agent_message_chunk_reaches_client(monkeypatch):
    temporal = MagicMock()
    captured: dict = {}

    real_agent = bridge_mod.BridgeAgent

    class SpyAgent(real_agent):
        def __init__(self, client, temporal):
            captured["client"] = client
            super().__init__(client, temporal)

    monkeypatch.setattr(bridge_mod, "BridgeAgent", SpyAgent)

    async def server(client_reader, client_writer):
        await _handle_connection(temporal, client_reader, client_writer)

    srv = await asyncio.start_server(server, "127.0.0.1", 0)
    port = srv.sockets[0].getsockname()[1]

    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    client = RecordingClient()
    conn = ClientSideConnection(client, writer, reader)

    resp: InitializeResponse = await asyncio.wait_for(
        conn.initialize(
            protocol_version=1,
            client_info=Implementation(name="regression", version="0"),
            client_capabilities=ClientCapabilities(),
        ),
        timeout=10,
    )
    assert resp.protocol_version == 1

    # Send 3 updates the way prompt() does: agent side -> client.
    agent_conn = captured.get("client")
    assert agent_conn is not None, "BridgeAgent never received the client interface"
    for text in ("a", "b", "c"):
        await asyncio.wait_for(
            agent_conn.session_update(
                "sid123",
                AgentMessageChunk(
                    session_update="agent_message_chunk",
                    content=TextContentBlock(type="text", text=text),
                ),
            ),
            timeout=10,
        )

    await asyncio.sleep(0.3)
    got = [u.content.text for u in client.updates]
    assert got == ["a", "b", "c"], f"client got: {got}"

    writer.close()
    srv.close()
    await srv.wait_closed()

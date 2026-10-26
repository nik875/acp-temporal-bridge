"""load_session history render + live tail — reconnect after crash/pod-roll.

Contract (v0.1.6):
* load_session renders the durable transcript as user/agent message chunks
  immediately (a reattached client must never sit on an empty session).
* Tool/system transcript messages are not rendered.
* The stream head is tracked, and a background tail task streams in-flight
  turn items until turn_end.
* prompt() drains the tail task before subscribing (no double-stream).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest
from acp.schema import AgentMessageChunk, UserMessageChunk

from acp_temporal_bridge import bridge as bridge_mod
from acp_temporal_bridge.bridge import BridgeAgent


@dataclass
class _Item:
    data: Any
    offset: int


class _FakeHandle:
    """Handle answering transcript_tail/stream_head queries; records describe."""

    def __init__(
        self,
        transcript: list[dict[str, Any]] | None = None,
        head: int = 7,
    ) -> None:
        self.transcript = transcript or []
        self.head = head

    async def describe(self) -> None:
        return None

    async def query(self, q: Any, *args: Any) -> Any:
        name = getattr(q, "name", "") or str(q)
        if "transcript_tail" in name:
            return {"transcript": self.transcript, "queued": []}
        if "stream_head" in name:
            return self.head
        raise AssertionError(f"unexpected query {name}")

    async def execute_update(self, *args: Any, **kwargs: Any) -> None:
        return None


class _FakeTemporal:
    def __init__(self, handle: _FakeHandle) -> None:
        self.handle = handle

    def get_workflow_handle(self, workflow_id: str) -> _FakeHandle:
        return self.handle


class _FakeConn:
    def __init__(self) -> None:
        self.updates: list[Any] = []

    async def session_update(self, session_id: str, update: Any) -> None:
        self.updates.append(update)


def _msg(role: str, text: str) -> dict[str, Any]:
    """Durable-transcript form: langchain message_to_dict output."""
    return {"type": role, "data": {"content": text, "type": role}}


async def test_load_renders_transcript_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Human + AI transcript messages render as chunks at load time."""
    conn = _FakeConn()
    handle = _FakeHandle(
        transcript=[
            _msg("human", "hello"),
            _msg("ai", "hi there"),
            _msg("tool", "raw tool output — not renderable"),
            _msg("system", "system — not renderable"),
            # unknown shapes are skipped, never fatal
            "garbage",
            {"type": "future_role", "data": {"content": "x"}},
        ]
    )
    agent = BridgeAgent(conn, _FakeTemporal(handle))  # type: ignore[arg-type]

    class _NoStream:
        @staticmethod
        def create(client: Any, workflow_id: str) -> _NoStream:
            return _NoStream()

        async def subscribe(self, topic: str, from_offset: int = 0) -> Any:
            return
            yield  # pragma: no cover

    monkeypatch.setattr(bridge_mod, "WorkflowStreamClient", _NoStream)
    await agent.load_session("/tmp", "s1")
    await asyncio.wait_for(agent._tail_tasks["s1"], timeout=5)
    kinds = [(type(u).__name__, u.content.text) for u in conn.updates]
    assert kinds == [
        ("UserMessageChunk", "hello"),
        ("AgentMessageChunk", "hi there"),
    ]
    assert all(
        isinstance(u, (UserMessageChunk, AgentMessageChunk)) for u in conn.updates
    )
    assert agent._offsets["s1"] == 7


async def test_load_tails_inflight_turn_until_turn_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Items after the head stream to the client; tail ends at turn_end."""
    conn = _FakeConn()
    handle = _FakeHandle(transcript=[], head=10)
    agent = BridgeAgent(conn, _FakeTemporal(handle))  # type: ignore[arg-type]

    items = [
        _Item({"generations": [{"text": "partial"}]}, 10),
        _Item({"kind": "turn_end", "stop_reason": "end_turn"}, 11),
        _Item({"kind": "message", "text": "never sent"}, 12),
    ]
    got_from_offset: list[int] = []

    class _Stream:
        @staticmethod
        def create(client: Any, workflow_id: str) -> _Stream:
            return _Stream()

        async def subscribe(self, topic: str, from_offset: int = 0) -> Any:
            got_from_offset.append(from_offset)
            for item in items:
                if item.offset >= from_offset:
                    yield item

    monkeypatch.setattr(bridge_mod, "WorkflowStreamClient", _Stream)
    await agent.load_session("/tmp", "s1")
    tail = agent._tail_tasks["s1"]
    await asyncio.wait_for(tail, timeout=5)
    kinds = [(type(u).__name__, u.content.text) for u in conn.updates]
    assert kinds == [("AgentMessageChunk", "partial")]
    assert got_from_offset == [10]
    assert agent._offsets["s1"] == 12


async def test_prompt_drains_pending_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """prompt() waits for the load tail to finish before subscribing."""
    conn = _FakeConn()
    handle = _FakeHandle(transcript=[], head=3)
    agent = BridgeAgent(conn, _FakeTemporal(handle))  # type: ignore[arg-type]

    released = asyncio.Event()
    drained_order: list[str] = []

    async def slow_tail_item() -> Any:
        await released.wait()
        drained_order.append("tail_done")
        yield _Item({"kind": "turn_end", "stop_reason": "end_turn"}, 3)

    class _TailStream:
        @staticmethod
        def create(client: Any, workflow_id: str) -> _TailStream:
            return _TailStream()

        async def subscribe(self, topic: str, from_offset: int = 0) -> Any:
            async for item in slow_tail_item():
                yield item

    class _PromptStream:
        @staticmethod
        def create(client: Any, workflow_id: str) -> _PromptStream:
            return _PromptStream()

        async def subscribe(self, topic: str, from_offset: int = 0) -> Any:
            drained_order.append(f"prompt_from_{from_offset}")
            yield _Item({"kind": "turn_end", "stop_reason": "end_turn"}, 99)

    monkeypatch.setattr(bridge_mod, "WorkflowStreamClient", _TailStream)
    await agent.load_session("/tmp", "s1")
    tail = agent._tail_tasks["s1"]
    # Let the tail task bind its stream class and block on the gate.
    await asyncio.sleep(0.05)
    assert drained_order == []

    monkeypatch.setattr(bridge_mod, "WorkflowStreamClient", _PromptStream)
    prompt_task = asyncio.create_task(agent.prompt("s1", []))
    await asyncio.sleep(0.05)
    assert drained_order == []  # prompt blocked while tail is pending
    released.set()
    resp = await asyncio.wait_for(prompt_task, timeout=5)
    await asyncio.wait_for(tail, timeout=5)
    assert resp.stop_reason == "end_turn"
    assert drained_order == ["tail_done", "prompt_from_4"]  # offset advanced

"""AgentSession._on_message_chunk: tool results must not echo as text.

``stream_mode="messages"`` yields the graph's messages, not only model tokens.
The old guard (``not isinstance(chunk, AIMessage) and not hasattr(chunk,
"content")``) never fired — a ``ToolMessage`` has a ``content`` attribute — so
a tool result was published as a ``message`` event AND rendered as the tool
card. The guard is now ``isinstance`` only.

Model thinking (``reasoning_content`` on the chunk) must go out as a ``thought``
event, never into the answer text.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

import acp_temporal_bridge.session as session_mod
from acp_temporal_bridge.session import STREAM_TOPIC


class _StubTopic:
    def __init__(self, sink: list[dict[str, Any]]) -> None:
        self._sink = sink

    def publish(self, entry: dict[str, Any]) -> None:
        self._sink.append(entry)


class _StubStream:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def topic(self, name: str) -> _StubTopic:
        assert name == STREAM_TOPIC
        return _StubTopic(self.events)


def _session() -> Any:
    sess = object.__new__(session_mod.AgentSession)
    sess._stream = _StubStream()
    return sess


def test_tool_message_chunk_is_not_echoed_as_text() -> None:
    sess = _session()
    sess._on_message_chunk((ToolMessage(content="TOOLBLOB", tool_call_id="t1"), {}))
    assert sess._stream.events == []


def test_model_chunk_still_publishes_text() -> None:
    sess = _session()
    sess._on_message_chunk((AIMessageChunk(content="hi"), {}))
    assert sess._stream.events == [{"kind": "message", "text": "hi"}]


def test_reasoning_chunk_publishes_a_thought_not_text() -> None:
    sess = _session()
    sess._on_message_chunk(
        (AIMessageChunk(content="", additional_kwargs={"reasoning_content": "hmm"}), {})
    )
    assert sess._stream.events == [{"kind": "thought", "text": "hmm"}]


def test_reasoning_and_text_are_published_in_order() -> None:
    sess = _session()
    chunk = AIMessageChunk(content="answer", additional_kwargs={"reasoning_content": "why"})
    sess._on_message_chunk((chunk, {}))
    assert [e["kind"] for e in sess._stream.events] == ["thought", "message"]


def test_a_plain_non_message_object_is_ignored() -> None:
    sess = _session()
    sess._on_message_chunk(("not a message", {}))
    assert sess._stream.events == []


def test_ai_message_type_check_is_not_a_hasattr_probe() -> None:
    # The regression: a class with `content` but not an AIMessage must be
    # dropped (a ToolMessage is the real-world case).
    class Contentish:
        content = "x"

    sess = _session()
    sess._on_message_chunk((Contentish(), {}))
    assert sess._stream.events == []


def test_ai_message_publishes_text() -> None:
    sess = _session()
    sess._on_message_chunk((AIMessage(content="done"), {}))
    assert sess._stream.events == [{"kind": "message", "text": "done"}]

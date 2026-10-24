"""Pure translation tests: stream-event dict -> ACP updates (no Temporal).

Covers the event vocabulary session.py publishes + the plugin's dumped
ChatGenerationChunk shape. Unknown shapes must yield [] (never raise).
"""

from __future__ import annotations

from acp.schema import (
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    ToolCallProgress,
    ToolCallStart,
)

from acp_temporal_bridge.bridge import _chunk_text, translate_stream_item


def test_message_chunk() -> None:
    outs = translate_stream_item({"kind": "message", "text": "hello "})
    assert len(outs) == 1
    assert isinstance(outs[0], AgentMessageChunk)
    assert outs[0].content.text == "hello "


def test_thought_chunk() -> None:
    outs = translate_stream_item({"kind": "thought", "text": "thinking"})
    assert len(outs) == 1
    assert isinstance(outs[0], AgentThoughtChunk)


def test_empty_text_is_skipped() -> None:
    assert translate_stream_item({"kind": "message", "text": ""}) == []


def test_tool_start_and_end() -> None:
    start = translate_stream_item(
        {"kind": "tool_start", "id": "t1", "name": "shell", "args": {"cmd": "ls"}}
    )
    assert len(start) == 1
    assert isinstance(start[0], ToolCallStart)
    assert start[0].tool_call_id == "t1"
    assert start[0].title == "shell"
    assert start[0].status == "in_progress"

    end = translate_stream_item({"kind": "tool_end", "id": "t1", "output": "ok"})
    assert len(end) == 1
    assert isinstance(end[0], ToolCallProgress)
    assert end[0].tool_call_id == "t1"
    assert end[0].status == "completed"
    assert end[0].content and end[0].content[0].content.text == "ok"


def test_plan_update() -> None:
    outs = translate_stream_item(
        {
            "kind": "plan",
            "todos": [
                {"content": "clone infra", "status": "completed", "priority": "high"},
                {"content": "open PR", "status": "pending", "priority": "medium"},
            ],
        }
    )
    assert len(outs) == 1
    assert isinstance(outs[0], AgentPlanUpdate)
    assert [e.content for e in outs[0].entries] == ["clone infra", "open PR"]


def test_plugin_chat_generation_chunk() -> None:
    # The streaming activity publishes dump_object(ChatGenerationChunk) —
    # recognize it by the "generations" key and pull the delta text out.
    dump = {
        "generations": [
            {"text": "de", "message": {"content": "de", "type": "AIMessageChunk"}}
        ],
        "type": "ChatGenerationChunk",
    }
    outs = translate_stream_item(dump)
    assert len(outs) == 1
    assert outs[0].content.text == "de"
    assert _chunk_text(dump) == "de"


def test_turn_end_is_not_an_acp_update() -> None:
    # turn_end terminates prompt() in the bridge; translation yields nothing.
    assert translate_stream_item({"kind": "turn_end", "stop_reason": "end_turn"}) == []


def test_unknown_shapes_yield_empty() -> None:
    assert translate_stream_item({"kind": "mystery", "x": 1}) == []
    assert translate_stream_item("not a dict") == []
    assert translate_stream_item(None) == []
    assert translate_stream_item({"generations": "bogus"}) == []

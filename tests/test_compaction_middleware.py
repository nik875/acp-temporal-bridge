"""Compaction-factory event reporting: start/end edges around a summary.

The sink is how a host renders compaction as a visible action (chat/tool-card
parity). These tests pin the contract: one event pair per compaction, a stable
pair id, the host-facing counters, and the public alias that keeps deepagents
from appending a second summarizer.
"""

from __future__ import annotations

import pytest
from deepagents.middleware.summarization import SummarizationMiddleware
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage

from acp_temporal_bridge.compaction_middleware import compaction_middleware


def _middleware(sink=None, trigger=100_000):
    mw_list = compaction_middleware(
        FakeListChatModel(responses=["the summary"]), None, trigger, on_event=sink
    )
    assert len(mw_list) == 1
    return mw_list[0]


def test_none_trigger_stays_empty_even_with_a_sink():
    empty = compaction_middleware(
        FakeListChatModel(responses=["s"]), None, None, on_event=lambda e: None
    )
    assert empty == []


def test_no_sink_still_returns_a_working_middleware():
    assert isinstance(_middleware(sink=None), SummarizationMiddleware)


def test_evented_subclass_keeps_the_public_default_slot_name():
    """deepagents swaps the default slot BY NAME — a subclass that reports
    type(self).__name__ would land as a second summarizer instead."""
    mw = _middleware(sink=lambda event: None)
    assert isinstance(mw, SummarizationMiddleware)
    assert mw.name == "SummarizationMiddleware"


async def test_start_and_end_events_bracket_the_summary():
    seen: list[dict] = []

    async def sink(event):
        seen.append(event)

    mw = _middleware(sink=sink)
    summary = await mw._acreate_summary([HumanMessage("a long history")])

    assert summary == "the summary"
    assert [e["phase"] for e in seen] == ["start", "end"]
    start, end = seen
    # One stable id for the pair, host-facing counters on both edges.
    assert start["id"] == end["id"] and start["id"].startswith("compact-")
    assert start["messages"] == end["messages"] == 1
    assert start["trigger"] == 100_000 and start["tokens"] > 0
    assert end["summary"] == "the summary" and end["error"] is None


async def test_sync_sink_is_supported():
    seen: list[str] = []
    mw = _middleware(sink=lambda event: seen.append(event["phase"]))

    await mw._acreate_summary([HumanMessage("x")])

    assert seen == ["start", "end"]


async def test_end_event_fires_before_reraising_a_summary_failure():
    seen: list[dict] = []

    class Boom(FakeListChatModel):
        def _generate(self, *args, **kwargs):
            raise RuntimeError("summarizer down")

        async def _agenerate(self, *args, **kwargs):
            raise RuntimeError("summarizer down")

    mw = compaction_middleware(Boom(responses=["s"]), None, 100_000, on_event=seen.append)
    with pytest.raises(RuntimeError):
        await mw[0]._acreate_summary([HumanMessage("x")])

    assert [e["phase"] for e in seen] == ["start", "end"]
    assert seen[-1]["error"] == "RuntimeError"


class _FakeStream:
    """Minimal WorkflowStream stand-in: topic(name).publish(entry)."""

    def __init__(self) -> None:
        self.entries: list[dict] = []

    def topic(self, name: str) -> _FakeStream:
        return self

    def publish(self, entry: dict) -> None:
        self.entries.append(entry)


def test_durable_path_sink_renders_a_tool_card_pair():
    """session._report_compaction is the Temporal-path wiring: it must publish
    a tool_start/tool_end pair (what the bridge maps to the client card)."""
    from types import SimpleNamespace

    from acp_temporal_bridge.session import AgentSession

    fake = _FakeStream()
    session = SimpleNamespace(_stream=fake)
    AgentSession._report_compaction(
        session, {"phase": "start", "id": "compact-abc", "messages": 12, "trigger": 225_000}
    )
    AgentSession._report_compaction(
        session,
        {"phase": "end", "id": "compact-abc", "messages": 12, "summary": "S", "error": None},
    )

    assert [e["kind"] for e in fake.entries] == ["tool_start", "tool_end"]
    assert {e["id"] for e in fake.entries} == {"compact-abc"}
    assert "12 messages" in fake.entries[0]["name"]
    assert fake.entries[1]["output"] == "S"


def test_durable_path_sink_reports_a_failed_compaction():
    from types import SimpleNamespace

    from acp_temporal_bridge.session import AgentSession

    fake = _FakeStream()
    AgentSession._report_compaction(
        SimpleNamespace(_stream=fake),
        {"phase": "end", "id": "c", "summary": "", "error": "RuntimeError"},
    )
    assert fake.entries[0]["output"] == "compaction failed: RuntimeError"

"""PrWatcher + AgentSession integration test — Temporal test env, Forgejo stubbed.

`_http_get` is the only impure seam (Forgejo API) inside the fetch activity;
tests monkeypatch the module global, which both the activity and the worker
registration resolve at call time.
"""

from __future__ import annotations

import asyncio
import os
from datetime import timedelta

import pytest
from temporalio import workflow
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

with workflow.unsafe.imports_passed_through():
    from temporalio.deepagents import DeepAgentsPlugin
    from temporalio.deepagents.testing import mock_model_provider

from acp_temporal_bridge import (
    AgentSession,
    PrWatcher,
    SessionSpec,
    SessionStart,
    configure,
    session_workflow_id,
    watcher,
)


@pytest.fixture(autouse=True)
def _config() -> None:
    configure(
        tool_builder=lambda names: [],
        default_spec=SessionSpec(
            model="mock", system_prompt="test", tools=["noop_tool"]
        ),
    )


@pytest.fixture(autouse=True)
def _prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(watcher, "POLL_INTERVAL", timedelta(milliseconds=50))
    monkeypatch.setenv("ACP_BRIDGE_WATCHED_REPOS", "infra,agents")
    monkeypatch.setenv("FORGEJO_URL", "https://forgejo.example")
    monkeypatch.setenv("FORGEJO_OWNER", "someone")


def _session_start(session_id: str) -> SessionStart:
    return SessionStart(
        run=SessionSpec(model="mock", system_prompt="test", tools=["noop_tool"]),
        session_id=session_id,
    )


async def _workflow_env():
    cli = os.environ.get("TEMPORAL_CLI_PATH")
    if cli is not None:
        return await WorkflowEnvironment.start_local(dev_server_existing_path=cli)
    return await WorkflowEnvironment.start_time_skipping()


async def test_watcher_signals_session_once_per_pr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A merged PR signals its session exactly once; dedup holds across polls."""

    # Canned Forgejo response, keyed by repo:
    http_calls = {"n": 0}

    def fake_http_get(url: str) -> tuple[int, str]:
        http_calls["n"] += 1
        if "/infra/pulls" in url:
            return 200, (
                '[{"number": 7, "title": "add thing", "state": "closed",'
                ' "merged": true, "html_url": "https://x/7",'
                ' "body": "<!-- agent-session: agent-sess-1 -->",'
                ' "merged_at": "2026-10-24T00:00:00Z"}]'
            )
        return 200, "[]"

    monkeypatch.setattr(watcher, "_http_get", fake_http_get)

    async with await _workflow_env() as env, Worker(
        env.client,
        task_queue="t",
        workflows=[AgentSession, PrWatcher],
        activities=[watcher.fetch_merged_prs],
        plugins=[
            # The package's seam state (configure()) lives in module globals —
            # the sandbox must reuse the main-process modules, not re-import.
            DeepAgentsPlugin(
                model_provider=mock_model_provider(["ok."]),
                passthrough_modules=["acp_temporal_bridge"],
            )
        ],
    ):
        session_handle = await env.client.start_workflow(
            AgentSession.run,
            args=[
                _session_start("sess-1"),
                None,  # stream_state
                None,  # transcript — arg count must match run signature (see bridge.py)
            ],
            id=session_workflow_id("sess-1"),
            task_queue="t",
        )
        watcher_handle = await env.client.start_workflow(
            PrWatcher.run, id="pr-watcher-test", task_queue="t"
        )
        info: dict = {}

        def _pr_system_messages(d: dict) -> list[str]:
            return [
                msg.get("data", {}).get("content", "")
                for msg in d.get("transcript", [])
                if isinstance(msg, dict)
                and str(msg.get("data", {}).get("content", "")).startswith(
                    "[system] PR #"
                )
            ]

        for _ in range(100):
            info = await session_handle.query(AgentSession.transcript_tail)  # type: ignore[arg-type]
            if _pr_system_messages(info):
                break
            await asyncio.sleep(0.05)
        landed = _pr_system_messages(info)
        assert landed, (
            f"session never received the pr_merged signal (http_calls={http_calls['n']})"
        )
        assert "PR #7 (infra)" in landed[0]
        # Give the watcher another poll or two; the signal must not re-land.
        await asyncio.sleep(0.3)
        info2 = await session_handle.query(AgentSession.transcript_tail)  # type: ignore[arg-type]
        assert len(_pr_system_messages(info2)) == 1, "duplicate signal — dedup broken"
        await watcher_handle.cancel()

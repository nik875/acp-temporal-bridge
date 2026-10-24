"""PR merge watcher: wake sessions when their PRs merge.

Forgejo (self-hosted git) has no egress into most isolated networks by
design, so webhooks are often impossible — this workflow polls the Forgejo
API from INSIDE the cluster/network instead, on a cadence. (A GitHub backend
can slot in behind the same poll->diff->signal shape later.)

Dedup across restarts: the merged-PR set (keyed by `repo#number`) is carried
across continue-as-new. New merges -> signal the originating session workflow
(parsed from the ``<!-- agent-session: ... -->`` marker the bridge host
embeds in PR bodies).

Session workflows may be gone (session over) — signaling a missing workflow
is tolerated: the PR stays marked seen either way.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import re
import urllib.error
import urllib.request
from datetime import timedelta

from temporalio import activity, workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from .session import AgentSession, PrEvent

POLL_INTERVAL = timedelta(
    seconds=int(os.environ.get("ACP_BRIDGE_WATCHER_POLL_SECS", "30"))
)
# History-protection: CAN well before history limits are a concern.
CAN_AFTER_POLLS = 50

_SESSION_MARKER = re.compile(r"agent-session:\s*([A-Za-z0-9._-]+)")


def _watched_repos() -> list[str]:
    raw = os.environ.get("ACP_BRIDGE_WATCHED_REPOS", "")
    return [repo.strip() for repo in raw.split(",") if repo.strip()]


@dataclasses.dataclass
class MergedPR:
    repo: str
    number: int
    title: str
    url: str
    body: str = ""
    merged_at: str = ""

    @property
    def key(self) -> str:
        return f"{self.repo}#{self.number}"

    @property
    def session_workflow_id(self) -> str | None:
        match = _SESSION_MARKER.search(self.body or "")
        return match.group(1) if match else None


def _forgejo_url() -> str:
    return os.environ["FORGEJO_URL"]


def _owner() -> str:
    return os.environ["FORGEJO_OWNER"]


def _http_get(url: str) -> tuple[int, str]:
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"token {os.environ.get('FORGEJO_TOKEN', '')}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status, resp.read().decode(errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode(errors="replace")
    except OSError as exc:
        return 0, str(exc)


@activity.defn
async def fetch_merged_prs(since_marker: str) -> list[MergedPR]:
    """List merged PRs across the watched repos.

    `since_marker` is a human-readable cutoff for logging only; dedup is the
    workflow's job (seen-set). HTTP goes through the module-level `_http_get`
    seam (tests monkeypatch it; the real one hits Forgejo with FORGEJO_TOKEN
    from the environment).
    """
    del since_marker  # logging breadcrumb only
    out: list[MergedPR] = []
    for repo in _watched_repos():
        url = (
            f"{_forgejo_url()}/api/v1/repos/{_owner()}/{repo}/pulls"
            "?state=closed&limit=20&sort=recentupdated"
        )
        status, body = await asyncio.to_thread(_http_get, url)
        if status != 200:
            activity.logger.warning("fetch %s: HTTP %s", repo, status)
            continue
        try:
            pulls = json.loads(body)
        except json.JSONDecodeError:
            continue
        for pull in pulls:
            if not pull.get("merged"):
                continue
            out.append(
                MergedPR(
                    repo=repo,
                    number=pull.get("number", 0),
                    title=pull.get("title", ""),
                    url=pull.get("html_url", ""),
                    body=pull.get("body") or "",
                    merged_at=pull.get("merged_at") or "",
                )
            )
    return out


@workflow.defn
class PrWatcher:
    """Singleton workflow (id ``pr-watcher``): poll -> diff -> signal sessions."""

    @workflow.init
    def __init__(self, seen: list[str] | None = None) -> None:
        self._seen: set[str] = set(seen or [])

    @workflow.run
    async def run(self, seen: list[str] | None = None) -> None:
        self._seen = set(seen or [])
        polls = 0
        while True:
            fresh = await workflow.execute_activity(
                fetch_merged_prs,
                args=[", ".join(sorted(self._seen))],
                start_to_close_timeout=timedelta(seconds=60),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            for pr in fresh:
                if pr.key in self._seen or not pr.session_workflow_id:
                    continue
                self._seen.add(pr.key)
                handle = workflow.get_external_workflow_handle(pr.session_workflow_id)
                try:
                    await handle.signal(
                        AgentSession.pr_merged,
                        PrEvent(
                            repo=pr.repo,
                            number=pr.number,
                            title=pr.title,
                            url=pr.url,
                            workflow_id=pr.session_workflow_id,
                            merged_at=pr.merged_at,
                        ),
                    )
                    workflow.logger.info(
                        "signaled %s: %s#%s",
                        pr.session_workflow_id,
                        pr.repo,
                        pr.number,
                    )
                except Exception as exc:  # noqa: BLE001 — session may be closed, not fatal
                    workflow.logger.info(
                        "signal to %s failed (session gone?): %s",
                        pr.session_workflow_id,
                        exc,
                    )
            polls += 1
            if polls >= CAN_AFTER_POLLS or workflow.info().is_continue_as_new_suggested():
                workflow.continue_as_new(args=[sorted(self._seen)])
            await workflow.sleep(POLL_INTERVAL.total_seconds())

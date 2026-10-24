# acp-temporal-bridge

[Agent Client Protocol](https://agentclientprotocol.com) bridge for **durable
agent sessions on [Temporal](https://temporal.io)**.

One Temporal workflow per editor/agent session. The agent's LangGraph/
DeepAgents loop runs *inside* the workflow — every model call and tool call
is a Temporal activity — so sessions survive worker restarts, machine
reboots, and network drops. An editor speaking ACP (e.g. Zed, or any ACP
client) just sees a normal agent: `new_session` / `prompt` / `cancel` /
streaming updates.

```
ACP client ──tcp/NDJSON-RPC──▶ BridgeAgent ──Temporal──▶ AgentSession (durable)
                                                          │
                                                 merge-watcher ──▶ "your PR
                                                     poller          merged" signal
```

## What's in the box

| Piece | What it is |
|---|---|
| `AgentSession` workflow | One durable workflow per session: prompt queue, turn loop with streaming, cancel, transcript carried across continue-as-new (CAN) |
| `BridgeAgent` + `serve()` | TCP server speaking ACP (newline-delimited JSON-RPC); maps ACP verbs onto workflow updates/signals |
| `PrWatcher` workflow | Polls a Forgejo instance for merged PRs and wakes the session that authored them (parsed from an `<!-- agent-session: ... -->` marker) |
| `translate_stream_item` | Pure function: durable stream events → ACP `session/update` notifications |

## Design

- **No durable checkpointer, ever.** Temporal history is the store; replay
  rehydrates. Multi-turn memory is an in-memory checkpointer seeded from a
  transcript that rides continue-as-new at turn boundaries.
- **The stream log is the replay source.** Model deltas (via the
  `temporalio-deepagents` plugin's `streaming_topic`) plus tool/plan/turn
  events publish to one `WorkflowStream` topic; the bridge replays from a
  per-session offset. `load_session` rewinds to 0 — the client sees the full
  transcript re-rendered.
- **Host-agnostic core, thin host seam.** The package never decides policy or
  builds tools itself; the embedding application wires those in.

## Setup contract

```python
import acp_temporal_bridge as atb

atb.configure(
    # names -> LangChain tool objects (workflow-side wrappers around
    # Temporal activities; see temporalio-deepagents' activity_as_tool)
    tool_builder=my_app.build_tools,
    # optional: names -> {tool: bool} for DeepAgents interrupt_on (HITL
    # gates). Default: nothing gated.
    interrupt_compiler=my_app.build_interrupt_on,
    # the spec every new session starts with
    default_spec=atb.SessionSpec(
        model="your-model-id",
        system_prompt="You are ...",
        tools=["my_tool", "other_tool"],
    ),
)
```

Call `configure()` **at import time** — workflow replay re-imports the
package, so the same seams must be configured identically in the replayed
process.

Run the worker and the bridge (same process or two):

```python
# worker: register the workflows + your activities + the plugin
DeepAgentsPlugin(
    model_provider=...,
    streaming_topic="acp",  # model deltas → the bridge-facing stream
    # REQUIRED if your host modules carry state the workflow needs (e.g.
    # configure() seams): the sandbox must reuse the main-process modules
    # instead of re-importing fresh copies.
    passthrough_modules=["acp_temporal_bridge", "my_app"],
)
Worker(..., workflows=[AgentSession, PrWatcher], activities=[...])

# bridge: one TCP port
await atb.serve()   # ACP_BRIDGE_PORT (default 8080)
```

Point the ACP client at the port and go.

### Starting a session workflow directly

When starting `AgentSession` yourself (not through the bridge), pass **all
three run arguments** — temporalio drops run-parameter type hints entirely
when the payload count doesn't match the signature length, which silently
decodes the input dataclass as a plain dict:

```python
await client.start_workflow(
    AgentSession.run,
    args=[SessionStart(run=spec, session_id=sid), None, None],  # stream_state, transcript
    id=atb.session_workflow_id(sid),
)
```

### Environment

| Variable | Default | Meaning |
|---|---|---|
| `TEMPORAL_HOST` / `TEMPORAL_NAMESPACE` | `localhost:7233` / `default` | Temporal frontend |
| `ACP_BRIDGE_PORT` / `ACP_BRIDGE_BIND` | `8080` / `0.0.0.0` | Bridge listener |
| `ACP_BRIDGE_TASK_QUEUE` | `durable-agents` | Task queue for workflows |
| `ACP_BRIDGE_WORKFLOW_PREFIX` | `agent-` | Session workflow-id prefix |
| `ACP_BRIDGE_WATCHED_REPOS` | *(empty)* | Comma-separated repos for PrWatcher |
| `FORGEJO_URL` / `FORGEJO_OWNER` / `FORGEJO_TOKEN` | — | Forgejo API for the watcher |

Secrets stay in the worker's environment (never in workflow inputs, history,
or tool args). The watcher tolerates a missing session workflow; the bridge
tolerates unknown stream events — both fail toward "keep running."

## Development

```sh
uv sync
uv run pytest        # unit tests; e2e needs a Temporal dev server
uv run ruff check .
```

MIT — see [LICENSE](LICENSE).

# agent-runtime

[![CI](https://github.com/Mattbusel/agent-runtime/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattbusel/agent-runtime/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/agent-runtime.svg)](https://crates.io/crates/agent-runtime)
[![docs.rs](https://docs.rs/agent-runtime/badge.svg)](https://docs.rs/agent-runtime)

A unified Tokio async agent runtime that brings together orchestration, episodic and semantic
memory, an in-memory knowledge graph, and a ReAct (Thought-Action-Observation) agent loop in a
single Rust crate. The library consolidates the public APIs of `tokio-prompt-orchestrator`,
`tokio-agent-memory`, `mem-graph`, and `wasm-agent`, adds pluggable LLM providers, optional
file-based persistence with per-step checkpointing, and lock-free runtime metrics, all governed
by a compile-time typestate builder that prevents misconfiguration at zero runtime cost.

---

## Architecture

The diagram below shows the primary components and the direction of data flow when
`AgentRuntime::run_agent` is called.

```
  User Code
     |
     v
+--------------------+      compile-time typestate
|  AgentRuntime      |<---- AgentRuntimeBuilder<NeedsConfig>
|  runtime.rs        |          .with_agent_config() ->
+----+----+----+-----+      AgentRuntimeBuilder<HasConfig>
     |    |    |                  .build()
     |    |    |
     |    |    +-------------------------------------------+
     |    |                                                |
     |    +-------------------+                           |
     |                        |                           |
     v                        v                           v
+--------------------+  +---------------------+  +--------------------+
|  memory.rs         |  |  graph.rs           |  |  orchestrator.rs   |
|                    |  |                     |  |                    |
|  EpisodicStore     |  |  GraphStore         |  |  CircuitBreaker    |
|    DecayPolicy     |  |    BFS / DFS        |  |    pluggable       |
|    RecallPolicy    |  |    Dijkstra         |  |    backends        |
|    per-agent cap   |  |    transitive close |  |  RetryPolicy       |
|  SemanticStore     |  |    centrality       |  |    exponential     |
|    cosine search   |  |    community detect |  |    backoff         |
|  WorkingMemory     |  |    cycle detection  |  |  Deduplicator      |
|    LRU eviction    |  |    subgraph extract |  |    TTL cache       |
+--------------------+  +---------------------+  |  BackpressureGuard |
                                                  |    hard+soft limit |
                                                  |  Pipeline          |
                                                  |    stage chain     |
                                                  +--------------------+
     |
     v
+--------------------+
|  agent.rs          |
|                    |
|  ReActLoop         |<--- ToolRegistry
|    parse_react     |       ToolSpec (sync/async)
|    step            |       required-field validation
|    max_iterations  |       per-tool CircuitBreaker
|  AgentConfig       |
|  AgentSession      |
+--------------------+
     |
     +----------------------------+
     |                            |
     v                            v
+--------------------+  +--------------------+
|  providers.rs      |  |  persistence.rs    |
|                    |  |                    |
|  LlmProvider trait |  |  PersistenceBackend|
|  AnthropicProvider |  |  FilePersistence   |
|  OpenAiProvider    |  |  session checkpoint|
|  (streaming SSE)   |  |  per-step snapshot |
+--------------------+  +--------------------+
                                   |
                         +---------+
                         |
                         v
               +--------------------+
               |  metrics.rs        |
               |                    |
               |  RuntimeMetrics    |
               |  (atomic counters) |
               +--------------------+
```

### Data flow inside `run_agent`

1. `BackpressureGuard` is checked; the session is rejected immediately if capacity is exhausted.
2. `EpisodicStore` is recalled for the agent; matching items are injected into the prompt, subject to `max_memory_recalls` and `max_memory_tokens` budgets.
3. `WorkingMemory` key-value pairs are appended to the enriched prompt.
4. `GraphStore` entity count is recorded for session metadata.
5. `ReActLoop` runs Thought-Action-Observation cycles, dispatching tool calls through `ToolRegistry`.
6. Per-tool `CircuitBreaker` (optional) fast-fails unhealthy tools and records structured error observations.
7. On completion an `AgentSession` is returned; if a `PersistenceBackend` is attached, final and per-step checkpoints are saved.
8. `RuntimeMetrics` counters are updated atomically at each stage.

---

## Quickstart

### 1. Add to Cargo.toml

```toml
[dependencies]
agent-runtime = "1.0"
tokio = { version = "1", features = ["full"] }
```

For built-in LLM providers:

```toml
agent-runtime = { version = "0.1", features = ["anthropic", "openai"] }
```

### 2. Set environment variables

```sh
export ANTHROPIC_API_KEY="sk-ant-..."   # required for AnthropicProvider
export OPENAI_API_KEY="sk-..."          # required for OpenAiProvider
export RUST_LOG="agent_runtime=debug"   # optional structured logging
```

### 3. Minimal code example

```rust
use agent_runtime::prelude::*;

#[tokio::main]
async fn main() {
    // The typestate builder enforces that with_agent_config() is called before
    // build(). Calling build() on AgentRuntimeBuilder<NeedsConfig> is a
    // compile-time error, not a runtime panic.
    let memory = EpisodicStore::new();
    let agent_id = AgentId::new("assistant");

    memory
        .add_episode(agent_id.clone(), "Paris is the capital of France.", 0.9)
        .expect("episodic store failed");

    let runtime = AgentRuntime::builder()
        .with_memory(memory)
        .with_agent_config(
            AgentConfig::new(10, "claude-sonnet-4-6")
                .with_system_prompt("You are a helpful assistant.")
                .with_max_memory_recalls(5),
        )
        .build();

    let session = runtime
        .run_agent(
            agent_id,
            "What is the capital of France?",
            |ctx: String| async move {
                // Send `ctx` to your LLM here and return the raw text.
                // The runtime parses lines beginning with "Thought:" and "Action:".
                let _ = ctx;
                "Thought: The answer is in memory.\nAction: FINAL_ANSWER Paris".to_string()
            },
        )
        .await
        .expect("agent run failed");

    println!(
        "Completed in {} step(s), {} memory hit(s), {}ms",
        session.step_count(),
        session.memory_hits,
        session.duration_ms,
    );
}
```

### Built-in provider example

```rust,no_run
use agent_runtime::providers::{AnthropicProvider, LlmProvider};

#[tokio::main]
async fn main() {
    let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set");
    let provider = AnthropicProvider::new(api_key);

    let reply = provider
        .complete("Say hello in one sentence.", "claude-sonnet-4-6")
        .await
        .expect("provider call failed");

    println!("{reply}");
}
```

---

## Feature Flags

| Feature | Default | Description |
|---|---|---|
| `orchestrator` | yes | `CircuitBreaker` with pluggable backends, `RetryPolicy` with exponential backoff, `Deduplicator` with TTL cache, `BackpressureGuard` with hard and soft limits, `Pipeline` |
| `memory` | yes | `EpisodicStore` with `DecayPolicy`, hybrid `RecallPolicy`, and per-agent capacity limits; `SemanticStore` with cosine-similarity search; `WorkingMemory` with LRU eviction |
| `graph` | yes | `GraphStore` with BFS, DFS, Dijkstra shortest-path, weighted shortest-path, transitive closure, degree and betweenness centrality, label-propagation community detection, subgraph extraction, and cycle detection |
| `wasm` | yes | `ReActLoop`, `ToolRegistry`, `ToolSpec` (sync and async handlers), `parse_react_step`, `AgentConfig` |
| `persistence` | no | `PersistenceBackend` async trait and `FilePersistenceBackend` disk implementation; session and per-step checkpointing on `AgentSession` |
| `providers` | no | `LlmProvider` async object-safe trait |
| `anthropic` | no | Built-in Anthropic Messages API provider with SSE streaming (implies `providers` + `reqwest`) |
| `openai` | no | Built-in OpenAI Chat Completions API provider with SSE streaming; Azure and custom base-URL support (implies `providers` + `reqwest`) |
| `redis-circuit-breaker` | no | Distributed `CircuitBreakerBackend` via Redis |
| `full` | no | All features above enabled simultaneously |

Selecting only what you need:

```toml
agent-runtime = { version = "0.1", default-features = false, features = ["memory", "orchestrator"] }
```

---

## Configuration Reference

### Environment Variables

| Variable | Required by | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | `AnthropicProvider` | API key for the Anthropic Messages API |
| `OPENAI_API_KEY` | `OpenAiProvider` | API key for the OpenAI Chat Completions API |
| `RUST_LOG` | tracing | Log filter string (e.g. `agent_runtime=debug,warn`) |

### Builder Methods

All builder methods return `Self` for chaining. `with_agent_config` is the only required call; it
transitions the builder from `AgentRuntimeBuilder<NeedsConfig>` to `AgentRuntimeBuilder<HasConfig>`,
which is the only state from which `build()` is callable.

| Method | Argument | Description |
|---|---|---|
| `.with_agent_config(cfg)` | `AgentConfig` | Required. Sets the ReAct loop configuration. |
| `.with_memory(store)` | `EpisodicStore` | Attaches episodic memory; recalled items are injected into the prompt. |
| `.with_working_memory(wm)` | `WorkingMemory` | Attaches bounded key-value working memory. |
| `.with_graph(graph)` | `GraphStore` | Attaches the knowledge graph. |
| `.with_backpressure(guard)` | `BackpressureGuard` | Rejects new sessions when in-flight count exceeds capacity. |
| `.register_tool(spec)` | `ToolSpec` | Registers a callable tool available to the ReAct loop. |
| `.with_metrics(m)` | `Arc<RuntimeMetrics>` | Shares a custom metrics instance across runtimes. |
| `.with_checkpoint_backend(b)` | `Arc<dyn PersistenceBackend>` | Enables per-step and final-session persistence (`persistence` feature). |

### AgentConfig Options

| Field | Type | Default | Description |
|---|---|---|---|
| `max_iterations` | `usize` | required | Maximum Thought-Action-Observation cycles before the loop errors |
| `model` | `String` | required | Model identifier forwarded to the `infer` closure |
| `system_prompt` | `String` | `"You are a helpful AI agent."` | Injected at the head of every context string |
| `max_memory_recalls` | `usize` | `3` | Maximum episodic items injected per run |
| `max_memory_tokens` | `Option<usize>` | `None` | Approximate token budget for memory injection (~4 chars per token) |

Builder methods: `.with_system_prompt(s)`, `.with_max_memory_recalls(n)`, `.with_max_memory_tokens(n)`.

### EpisodicStore Constructors

| Constructor | Description |
|---|---|
| `EpisodicStore::new()` | Unbounded, no decay, importance-ranked recall |
| `EpisodicStore::with_decay(policy)` | Applies `DecayPolicy::exponential(half_life_hours)` before each recall |
| `EpisodicStore::with_recall_policy(policy)` | `RecallPolicy::Hybrid { recency_weight, frequency_weight }` for blended scoring |
| `EpisodicStore::with_per_agent_capacity(n)` | Evicts the lowest-importance item when an agent accumulates more than `n` memories |

### BackpressureGuard

```rust
let guard = BackpressureGuard::new(100)   // hard limit: reject above 100 in-flight
    .unwrap()
    .with_soft_limit(75)                  // log warning when depth reaches 75
    .unwrap();
```

---

## Quick Start (no external services)

The entire default feature set (`orchestrator`, `memory`, `graph`, `wasm`) works with zero
external dependencies — no LLM API key, no Redis, no database.  The snippet below compiles and
runs locally with only `cargo run`:

```rust
use agent_runtime::prelude::*;

#[tokio::main]
async fn main() -> Result<(), AgentRuntimeError> {
    // Build an EpisodicStore with two pre-seeded memories.
    let memory = EpisodicStore::new();
    let agent_id = AgentId::new("demo");
    memory.add_episode(agent_id.clone(), "Rust is fast and memory-safe.", 0.9)?;
    memory.add_episode(agent_id.clone(), "Tokio is an async runtime for Rust.", 0.8)?;

    // Register a calculator tool that works entirely in-process.
    let runtime = AgentRuntime::builder()
        .with_memory(memory)
        .with_agent_config(
            AgentConfig::new(5, "stub-model")
                .with_system_prompt("You are a demo agent.")
                .with_max_memory_recalls(3),
        )
        .register_tool(ToolSpec::new("double", "Doubles a number", |args| {
            let n = args.get("n").and_then(|v| v.as_i64()).unwrap_or(0);
            serde_json::json!(n * 2)
        }))
        .build();

    // The `infer` closure acts as the model — replace with a real provider in production.
    let mut step = 0usize;
    let session = runtime
        .run_agent(agent_id, "Double the number 21.", move |_ctx: String| {
            step += 1;
            let s = step;
            async move {
                if s == 1 {
                    r#"Thought: I will double 21 using the tool.
Action: double {"n":21}"#
                        .to_string()
                } else {
                    "Thought: The answer is 42.\nAction: FINAL_ANSWER 42".to_string()
                }
            }
        })
        .await?;

    println!(
        "Done in {} step(s), {} memory hit(s), {}ms",
        session.step_count(),
        session.memory_hits,
        session.duration_ms,
    );
    Ok(())
}
```

---

## Error Handling

All public APIs return `Result<T, AgentRuntimeError>`.  The error enum is non-exhaustive by
design — match only the variants you need and use a catch-all for future additions:

```rust
use agent_runtime::prelude::*;

fn handle(err: AgentRuntimeError) {
    match err {
        AgentRuntimeError::CircuitOpen { service } => {
            eprintln!("Circuit open for {service} — backing off");
        }
        AgentRuntimeError::BackpressureShed { depth, capacity } => {
            eprintln!("Shed: {depth}/{capacity} in-flight — try again later");
        }
        AgentRuntimeError::AgentLoop(msg) => {
            eprintln!("Agent loop failed: {msg}");
        }
        AgentRuntimeError::Memory(msg) => {
            eprintln!("Memory subsystem error: {msg}");
        }
        AgentRuntimeError::Graph(msg) => {
            eprintln!("Graph subsystem error: {msg}");
        }
        AgentRuntimeError::Persistence(msg) => {
            eprintln!("Persistence error: {msg}");
        }
        AgentRuntimeError::Provider(msg) => {
            eprintln!("LLM provider error: {msg}");
        }
        other => eprintln!("Other error: {other}"),
    }
}
```

**Error propagation in `run_agent`:**

1. `BackpressureShed` — returned immediately if `BackpressureGuard` is at capacity.
2. `Memory` — returned if `EpisodicStore::recall` or `WorkingMemory::entries` fails.
3. `Graph` — returned if `GraphStore::entity_count` fails.
4. `AgentLoop` — returned when `max_iterations` is exhausted or a response cannot be parsed.
5. `Persistence` — returned when checkpointing fails (only if `persistence` feature is enabled).

All paths are non-panicking.  The Clippy deny rules `unwrap_used`, `expect_used`, `panic`, and
`todo` are enforced for all production code in `src/`.

---

## Contributing

See the full contributing guide in [CONTRIBUTING](#contributing-1) below, or open a pull request
with a description of motivation and design decisions.

**Checklist before opening a PR:**

- [ ] `cargo test --all-features` passes with 0 failures
- [ ] `cargo clippy --all-features -- -D warnings` passes with 0 warnings
- [ ] `cargo fmt --all -- --check` passes
- [ ] `./scripts/ratio_check.sh` shows a test-to-production ratio ≥ 1:1
- [ ] New public items have `///` doc comments

---

## Running Tests

```sh
# Default feature set (orchestrator, memory, graph, wasm)
cargo test

# All features, including persistence and providers
cargo test --all-features

# A specific module
cargo test --lib memory
cargo test --lib graph

# With structured log output
RUST_LOG=agent_runtime=debug cargo test -- --nocapture
```

Check the test-to-production line ratio before every commit:

```sh
./scripts/ratio_check.sh
```

Run Clippy with the project's deny rules:

```sh
cargo clippy --all-features -- -D warnings
```

---

## Contributing

1. Fork the repository and create a descriptive feature branch.
2. Add tests for every new public function, struct, and trait. The project maintains a minimum 1:1
   test-to-production line ratio; `./scripts/ratio_check.sh` enforces this.
3. All production paths must be panic-free. Use `Result` for every fallible operation. The project
   denies `unwrap_used`, `expect_used`, `panic`, and `todo` in Clippy.
4. Run `cargo test --all-features` and `cargo clippy --all-features -- -D warnings` with zero
   failures before opening a pull request.
5. Describe the motivation and design decisions in the PR body. Link any related issues.

---

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.

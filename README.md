# agent-runtime

[![CI](https://github.com/Mattbusel/agent-runtime/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattbusel/agent-runtime/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/agent-runtime.svg)](https://crates.io/crates/agent-runtime)
[![docs.rs](https://docs.rs/agent-runtime/badge.svg)](https://docs.rs/agent-runtime)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust 1.75+](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)

**agent-runtime** is a unified Tokio async agent runtime for Rust. It combines orchestration
primitives, episodic and semantic memory, an in-memory knowledge graph, and a ReAct
(Thought-Action-Observation) agent loop in a single crate.

The library consolidates the public APIs of `tokio-prompt-orchestrator`, `tokio-agent-memory`,
`mem-graph`, and `wasm-agent`, and extends them with pluggable LLM providers, optional
file-based persistence with per-step checkpointing, lock-free runtime metrics, and a
compile-time typestate builder that prevents misconfiguration at zero runtime cost.

---

## What it does

- **ReAct agent loop** — runs Thought → Action → Observation cycles with a pluggable tool
  registry. Terminates on `FINAL_ANSWER` or `max_iterations`, whichever comes first.
- **Episodic memory** — per-agent event store with configurable decay, hybrid recall scoring,
  per-agent capacity limits, and recall-count tracking.
- **Semantic memory** — key-value store with tag-based retrieval and cosine-similarity vector
  search.
- **Working memory** — bounded LRU key-value store injected into the agent prompt.
- **Knowledge graph** — directed in-memory graph with BFS, DFS, Dijkstra shortest-path,
  transitive closure, degree and betweenness centrality, community detection, cycle detection,
  and subgraph extraction.
- **Circuit breaker** — configurable failure threshold and recovery window with a pluggable
  backend trait for distributed state (e.g., Redis).
- **Retry policy** — exponential backoff capped at 60 s.
- **Deduplicator** — TTL-based request deduplication with in-flight tracking.
- **Backpressure guard** — hard and soft capacity limits with tracing warnings.
- **Pipeline** — composable string-transform stage chain.
- **LLM providers** — built-in `AnthropicProvider` and `OpenAiProvider` with SSE streaming
  (behind feature flags).
- **Persistence** — async `PersistenceBackend` trait and `FilePersistenceBackend` for
  session and per-step checkpointing.
- **Metrics** — atomic counters for active/total sessions, steps, tool calls, backpressure
  sheds, memory recalls, and checkpoint errors.
- **`types` module** — `AgentId` and `MemoryId` are always available without enabling the
  `memory` feature, allowing the runtime to compile in minimal configurations.

---

## How it works

```
  User Code
     |
     v
+--------------------+      compile-time typestate
|  AgentRuntime      |<---- AgentRuntimeBuilder<NeedsConfig>
|  runtime.rs        |          .with_agent_config()  -->
+----+----+----+-----+      AgentRuntimeBuilder<HasConfig>
     |    |    |                  .build()             (infallible)
     |    |    |
     |    |    +--------------------------------------------+
     |    |                                                 |
     |    +-------------------+                            |
     |                        |                            |
     v                        v                            v
+--------------------+  +---------------------+  +--------------------+
|  memory.rs         |  |  graph.rs           |  |  orchestrator.rs   |
|                    |  |                     |  |                    |
|  EpisodicStore     |  |  GraphStore         |  |  CircuitBreaker    |
|    DecayPolicy     |  |    BFS / DFS        |  |  RetryPolicy       |
|    RecallPolicy    |  |    Dijkstra         |  |  Deduplicator      |
|    per-agent cap   |  |    transitive close |  |  BackpressureGuard |
|  SemanticStore     |  |    centrality       |  |  Pipeline          |
|    cosine search   |  |    community detect |  +--------------------+
|  WorkingMemory     |  |    cycle detection  |
|    LRU eviction    |  +---------------------+
+--------------------+
     |
     v
+--------------------+
|  agent.rs          |
|                    |
|  ReActLoop         |<--- ToolRegistry (ToolSpec, per-tool CircuitBreaker)
|  AgentConfig       |
|  AgentSession      |
+--------------------+
     |
     +---------------------------+
     |                           |
     v                           v
+--------------------+  +--------------------+
|  providers.rs      |  |  persistence.rs    |
|  LlmProvider trait |  |  PersistenceBackend|
|  AnthropicProvider |  |  FilePersistence   |
|  OpenAiProvider    |  |  session checkpoint|
+--------------------+  |  per-step snapshot |
                         +--------------------+
                                  |
                        +---------+
                        v
              +--------------------+
              |  metrics.rs        |
              |  RuntimeMetrics    |
              |  (atomic counters) |
              +--------------------+
```

### Data flow inside `run_agent`

1. `BackpressureGuard` is checked; sessions exceeding capacity are rejected immediately with
   `AgentRuntimeError::BackpressureShed`.
2. `EpisodicStore` is recalled for the agent; matching items are injected into the prompt,
   subject to `max_memory_recalls` and the optional `max_memory_tokens` token budget.
3. `WorkingMemory` key-value pairs are appended to the enriched prompt.
4. `GraphStore` entity count is captured for session metadata.
5. `ReActLoop` runs Thought-Action-Observation cycles, dispatching tool calls through
   `ToolRegistry`.
6. Per-tool `CircuitBreaker` (optional) fast-fails unhealthy tools and records structured
   error observations with `kind` classification (`not_found`, `transient`, `permanent`).
7. On completion an `AgentSession` is returned; if a `PersistenceBackend` is configured,
   the final session and every per-step snapshot are saved atomically.
8. `RuntimeMetrics` counters are updated atomically throughout.

---

## Quickstart

### 1. Add to `Cargo.toml`

```toml
[dependencies]
agent-runtime = "1.0"
tokio = { version = "1", features = ["full"] }
```

To enable built-in LLM providers:

```toml
agent-runtime = { version = "1.0", features = ["anthropic", "openai"] }
```

To opt in to only the subsystems you need:

```toml
agent-runtime = { version = "1.0", default-features = false, features = ["memory", "orchestrator"] }
```

### 2. Environment variables

```sh
export ANTHROPIC_API_KEY="sk-ant-..."   # required for AnthropicProvider
export OPENAI_API_KEY="sk-..."          # required for OpenAiProvider
export RUST_LOG="agent_runtime=debug"   # optional structured logging
```

### 3. Minimal example (no external services)

The default feature set (`orchestrator`, `memory`, `graph`, `wasm`) runs entirely in-process
with no API keys, no Redis, and no database.

```rust
use agent_runtime::prelude::*;

#[tokio::main]
async fn main() -> Result<(), AgentRuntimeError> {
    // Seed episodic memory.
    let memory = EpisodicStore::new();
    let agent_id = AgentId::new("demo");
    memory.add_episode(agent_id.clone(), "Rust is fast and memory-safe.", 0.9)?;
    memory.add_episode(agent_id.clone(), "Tokio is an async runtime for Rust.", 0.8)?;

    // Build the runtime.  The typestate builder enforces that
    // with_agent_config() is called before build() — at compile time.
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

    // The `infer` closure acts as the model.
    // Replace with a real provider call in production.
    let mut step = 0usize;
    let session = runtime
        .run_agent(agent_id, "Double the number 21.", move |_ctx: String| {
            step += 1;
            let s = step;
            async move {
                if s == 1 {
                    r#"Thought: I will use the double tool.
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

### 4. Using a built-in provider

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

### 5. Wiring a provider into the runtime

`run_agent_with_provider` removes the boilerplate closure when using a built-in
provider:

```rust,no_run
use agent_runtime::prelude::*;
use agent_runtime::providers::AnthropicProvider;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), AgentRuntimeError> {
    let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set");
    let provider = Arc::new(AnthropicProvider::new(api_key));

    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(10, "claude-sonnet-4-6"))
        .build();

    let session = runtime
        .run_agent_with_provider(AgentId::new("agent-1"), "What is 6 * 7?", provider)
        .await?;

    println!("Answer: {}", session.final_answer().unwrap_or("no answer"));
    Ok(())
}
```

---

## Feature Flags

| Feature | Default | Description |
|---|---|---|
| `orchestrator` | yes | `CircuitBreaker` with pluggable backends, `RetryPolicy`, `Deduplicator`, `BackpressureGuard` with soft limit, `Pipeline` |
| `memory` | yes | `EpisodicStore` with `DecayPolicy`, `RecallPolicy::Hybrid`, per-agent capacity; `SemanticStore` with cosine search; `WorkingMemory` LRU |
| `graph` | yes | `GraphStore` — BFS, DFS, Dijkstra, transitive closure, degree/betweenness centrality, community detection, subgraph, cycle detection |
| `wasm` | yes | `ReActLoop` with sync and streaming inference, `ToolRegistry`, `ToolSpec`, `parse_react_step`, `AgentConfig`, observer callbacks, and step-level metrics |
| `persistence` | no | `PersistenceBackend` trait + `FilePersistenceBackend`; session and per-step checkpointing |
| `providers` | no | `LlmProvider` async trait |
| `anthropic` | no | Built-in Anthropic Messages API provider with SSE streaming (implies `providers` + `reqwest`) |
| `openai` | no | Built-in OpenAI Chat Completions API provider with SSE streaming and custom base-URL support (implies `providers` + `reqwest`) |
| `redis-circuit-breaker` | no | Distributed `CircuitBreakerBackend` via Redis |
| `full` | no | All features simultaneously |

---

## API Reference

### `AgentRuntime` builder

```rust
let runtime = AgentRuntime::builder()       // AgentRuntimeBuilder<NeedsConfig>
    .with_memory(EpisodicStore::new())
    .with_working_memory(WorkingMemory::new(64)?)
    .with_graph(GraphStore::new())
    .with_backpressure(BackpressureGuard::new(100)?)
    .register_tool(my_tool)
    .with_metrics(metrics_arc)
    .with_checkpoint_backend(backend_arc)   // persistence feature
    .with_agent_config(config)              // --> AgentRuntimeBuilder<HasConfig>
    .build();                               // infallible
```

| Method | Argument | Description |
|---|---|---|
| `.with_agent_config(cfg)` | `AgentConfig` | **Required.** Transitions builder to `HasConfig`. |
| `.with_memory(store)` | `EpisodicStore` | Episodic memory recalled and injected into the prompt. |
| `.with_working_memory(wm)` | `WorkingMemory` | Bounded key-value working memory appended to the prompt. |
| `.with_graph(graph)` | `GraphStore` | Knowledge graph; entity count captured in session metadata. |
| `.with_backpressure(guard)` | `BackpressureGuard` | Rejects sessions when in-flight count exceeds capacity. |
| `.register_tool(spec)` | `ToolSpec` | Adds a callable tool to the ReAct loop. |
| `.with_metrics(m)` | `Arc<RuntimeMetrics>` | Shares a custom metrics instance. |
| `.with_checkpoint_backend(b)` | `Arc<dyn PersistenceBackend>` | Enables checkpointing (`persistence` feature). |

### `AgentConfig`

| Field / Builder | Type | Default | Description |
|---|---|---|---|
| `max_iterations` | `usize` | required | Maximum Thought-Action-Observation cycles |
| `model` | `String` | required | Model identifier forwarded to the `infer` closure |
| `.with_system_prompt(s)` | `String` | `"You are a helpful AI agent."` | Injected at the head of every context string |
| `.with_max_memory_recalls(n)` | `usize` | `3` | Maximum episodic items injected per run |
| `.with_max_memory_tokens(n)` | `usize` | `None` | Approximate token budget (~4 chars/token) |
| `.with_stop_sequences(v)` | `Vec<String>` | `[]` | Stop sequences forwarded to the provider on every call |
| `.with_loop_timeout_ms(n)` | `u64` | `None` | Wall-clock deadline for the entire ReAct loop in milliseconds |

### `EpisodicStore` constructors

| Constructor | Description |
|---|---|
| `EpisodicStore::new()` | Unbounded, no decay, importance-ranked |
| `EpisodicStore::with_decay(policy)` | `DecayPolicy::exponential(half_life_hours)` |
| `EpisodicStore::with_recall_policy(p)` | `RecallPolicy::Hybrid { recency_weight, frequency_weight }` |
| `EpisodicStore::with_per_agent_capacity(n)` | Evicts lowest-importance item when agent exceeds `n` memories |

### `BackpressureGuard`

```rust
let guard = BackpressureGuard::new(100)?   // hard limit
    .with_soft_limit(75)?;                 // warn when depth reaches 75
```

### `CircuitBreaker`

```rust
let cb = CircuitBreaker::new("my-service", 5, Duration::from_secs(30))?;
let result = cb.call(|| my_fallible_operation())?;
```

### `ToolSpec`

```rust
// Synchronous handler
let spec = ToolSpec::new("greet", "Greets someone", |args| {
    serde_json::json!({ "message": "hello" })
});

// Async handler
let spec = ToolSpec::new_async("fetch", "Fetches a URL", |args| {
    Box::pin(async move { serde_json::json!({ "status": "ok" }) })
});

// With validation and circuit breaker
let spec = ToolSpec::new("search", "Searches the web", |args| {
    serde_json::json!({ "results": [] })
})
.with_required_fields(vec!["q".to_string()])
.with_circuit_breaker(cb_arc);
```

### `ToolRegistry`

```rust
// Check and remove tools at runtime
if registry.contains("old-tool") {
    let removed: Option<ToolSpec> = registry.remove("old-tool");
}
```

### `FilePersistenceBackend`

```rust
// Conditional save — only checkpoint if one doesn't exist yet
if !backend.exists("agent-1-session").await? {
    backend.save("agent-1-session", &data).await?;
}
```

---

## Error Handling

All public APIs return `Result<T, AgentRuntimeError>`. Match only the variants you care about:

```rust
use agent_runtime::prelude::*;

fn handle(err: AgentRuntimeError) {
    match err {
        AgentRuntimeError::CircuitOpen { service } =>
            eprintln!("Circuit open for {service} — backing off"),
        AgentRuntimeError::BackpressureShed { depth, capacity } =>
            eprintln!("Shed: {depth}/{capacity} in-flight — try again later"),
        AgentRuntimeError::AgentLoop(msg) =>
            eprintln!("Agent loop failed: {msg}"),
        AgentRuntimeError::Memory(msg) =>
            eprintln!("Memory subsystem error: {msg}"),
        AgentRuntimeError::Graph(msg) =>
            eprintln!("Graph subsystem error: {msg}"),
        AgentRuntimeError::Persistence(msg) =>
            eprintln!("Persistence error: {msg}"),
        AgentRuntimeError::Provider(msg) =>
            eprintln!("LLM provider error: {msg}"),
        other => eprintln!("Other error: {other}"),
    }
}
```

All production code paths are panic-free. Clippy denies `unwrap_used`, `expect_used`,
`panic`, and `todo` in `src/`.

---

## Running Tests

```sh
# Default feature set
cargo test

# All features including persistence and providers
cargo test --all-features

# A specific module
cargo test --lib memory
cargo test --lib graph

# With structured log output
RUST_LOG=agent_runtime=debug cargo test -- --nocapture
```

Run the full CI suite locally:

```sh
cargo fmt --all -- --check
cargo clippy --all-features -- -D warnings
cargo test --all-features
cargo doc --no-deps --all-features
cargo build --release --all-features
```

---

## Contributing

1. Fork the repository and create a descriptive feature branch.
2. Add tests for every new public function, struct, and trait. The project targets a minimum
   1:1 test-to-production line ratio.
3. All production paths must be panic-free. Use `Result` for every fallible operation.
   Clippy denies `unwrap_used`, `expect_used`, `panic`, and `todo` in `src/`.
4. Run `cargo test --all-features` and `cargo clippy --all-features -- -D warnings` with
   zero failures before opening a pull request.
5. New public items require `///` doc comments. The crate enforces `#![deny(missing_docs)]`.
6. Describe the motivation and design decisions in the PR body.

**Checklist before opening a PR:**

- [ ] `cargo test --all-features` passes
- [ ] `cargo clippy --all-features -- -D warnings` passes
- [ ] `cargo fmt --all -- --check` passes
- [ ] `cargo doc --no-deps --all-features` passes with `RUSTDOCFLAGS="-D warnings"`
- [ ] New public items have `///` doc comments

---

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.

# agent-runtime

[![CI](https://github.com/Mattbusel/agent-runtime/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattbusel/agent-runtime/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/llm-agent-runtime.svg)](https://crates.io/crates/llm-agent-runtime)
[![docs.rs](https://docs.rs/llm-agent-runtime/badge.svg)](https://docs.rs/llm-agent-runtime)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust 1.85+](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)

`agent-runtime` is a batteries-included, async-first Rust crate for building production LLM agents. It unifies a ReAct (Thought-Action-Observation) loop, episodic and semantic memory with decay and cosine-similarity recall, a directed knowledge graph with centrality and community detection, an orchestration layer with circuit breakers and retry/backpressure, pluggable LLM providers with SSE streaming, optional file-based session checkpointing, intelligent memory compression for long-running agents, and a peer-discovery registry for multi-agent systems — all in a single crate, driven by a compile-time typestate builder that makes misconfiguration a compiler error rather than a runtime panic.

---

## Feature Matrix

| Feature | Default | What you get |
|---|:---:|---|
| `orchestrator` | yes | `CircuitBreaker` (pluggable backends), `RetryPolicy` (exp. backoff), `Deduplicator` (TTL), `BackpressureGuard` (hard + soft limits), `Pipeline` |
| `memory` | yes | `EpisodicStore` (`DecayPolicy`, `RecallPolicy::Hybrid`, per-agent capacity), `SemanticStore` (cosine-similarity vector search, tag recall), `WorkingMemory` (bounded LRU) |
| `graph` | yes | `GraphStore` — BFS, DFS, Dijkstra shortest-path, transitive closure, degree/betweenness centrality, community detection, cycle detection, subgraph extraction |
| `wasm` | yes | `ReActLoop` with sync + streaming inference, `ToolRegistry`, `ToolSpec`, `parse_react_step`, `AgentConfig`, observer callbacks, step-level metrics |
| `persistence` | no | `PersistenceBackend` async trait + `FilePersistenceBackend`; per-session and per-step checkpointing to disk |
| `providers` | no | `LlmProvider` async trait |
| `anthropic` | no | Built-in Anthropic Messages API provider with SSE streaming (implies `providers` + `reqwest`) |
| `openai` | no | Built-in OpenAI Chat Completions provider with SSE streaming and custom base-URL support (implies `providers` + `reqwest`) |
| `redis-circuit-breaker` | no | Distributed `CircuitBreakerBackend` state via Redis |
| `distributed` | no | Distributed agent coordination via Redis: work queue and leader election |
| `otel` | no | OpenTelemetry tracing spans for tool calls (implies `opentelemetry` + `opentelemetry_sdk` + `opentelemetry-otlp`) |
| `compression` | no | `MemoryCompressor`, `ImportanceStrategy`, `MemorySummary` — token-budget-aware compression of episodic memory |
| `discovery` | no | `AgentRegistry`, `CapabilityQuery`, `CapabilityMatch` — TTL-based peer capability advertisement and tag-overlap matching |
| `full` | no | All of the above simultaneously |

---

## 5-Minute Quickstart

### 1. Add to `Cargo.toml`

```toml
[dependencies]
llm-agent-runtime = "1.75"
tokio = { version = "1", features = ["full"] }
```

To opt in to specific subsystems only:

```toml
llm-agent-runtime = { version = "1.75", default-features = false, features = ["memory", "orchestrator"] }
```

To enable built-in LLM providers:

```toml
llm-agent-runtime = { version = "1.75", features = ["anthropic", "openai"] }
```

### 2. Set environment variables (if using a provider)

```sh
export ANTHROPIC_API_KEY="sk-ant-..."   # required for AnthropicProvider
export OPENAI_API_KEY="sk-..."          # required for OpenAiProvider
export RUST_LOG="agent_runtime=debug"   # optional structured tracing output
```

### 3. Run an agent (no external services required)

The default feature set runs entirely in-process:

```rust
use llm_agent_runtime::prelude::*;

#[tokio::main]
async fn main() -> Result<(), AgentRuntimeError> {
    // Seed episodic memory.
    let memory = EpisodicStore::new();
    let agent_id = AgentId::new("demo");
    memory.add_episode(agent_id.clone(), "Rust is fast and memory-safe.", 0.9)?;
    memory.add_episode(agent_id.clone(), "Tokio is an async runtime for Rust.", 0.8)?;

    // Build the runtime.  The typestate builder enforces that
    // with_agent_config() is called before build() at compile time.
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

    // The `infer` closure acts as the model — replace with a provider call in production.
    let mut step = 0usize;
    let session = runtime
        .run_agent(agent_id, "Double the number 21.", move |_ctx: String| {
            step += 1;
            let s = step;
            async move {
                if s == 1 {
                    "Thought: I will use the double tool.\nAction: double {\"n\":21}".to_string()
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

### 4. Use a built-in provider

```rust,no_run
use llm_agent_runtime::prelude::*;
use llm_agent_runtime::providers::AnthropicProvider;
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

## Architecture

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
     |    |    +----------------------------------------------+
     |    |                                                   |
     |    +-------------------+                              |
     |                         |                              |
     v                         v                              v
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
+---------------------------+  +---------------------------+
|  memory_compression.rs    |  |  discovery.rs             |
|                           |  |                           |
|  MemoryCompressor         |  |  AgentRegistry            |
|    ImportanceStrategy     |  |    register / deregister  |
|    recency protection     |  |    heartbeat / evict      |
|    MemorySummary          |  |    CapabilityQuery        |
|    token-budget aware     |  |    tag-overlap scoring    |
+---------------------------+  +---------------------------+
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

1. `BackpressureGuard` is checked; sessions exceeding capacity are rejected immediately with `AgentRuntimeError::BackpressureShed`.
2. `EpisodicStore` is recalled for the agent; matching items are injected into the prompt, subject to `max_memory_recalls` and the optional `max_memory_tokens` token budget.
3. `WorkingMemory` key-value pairs are appended to the enriched prompt.
4. `GraphStore` entity count is captured for session metadata.
5. `ReActLoop` runs Thought-Action-Observation cycles, dispatching tool calls through `ToolRegistry`.
6. Per-tool `CircuitBreaker` (optional) fast-fails unhealthy tools and records structured error observations with `kind` classification (`not_found`, `transient`, `permanent`).
7. On completion an `AgentSession` is returned; if a `PersistenceBackend` is configured, the final session and every per-step snapshot are saved atomically.
8. `RuntimeMetrics` counters are updated atomically throughout.

---

## Memory Compression

When episodic history grows large, `MemoryCompressor` prunes it to fit a token budget while retaining the most useful turns.

```rust
use llm_agent_runtime::memory_compression::{
    ImportanceStrategy, MemoryCompressor, MemoryTurn, Role,
};
use std::time::SystemTime;

// Build a compressor targeting a 4 096-token budget.
let compressor = MemoryCompressor::new(4096)
    // Always keep the 8 most-recent turns verbatim.
    .with_recency_keep(8)
    // Turns below this score are candidates for compression.
    .with_importance_threshold(0.35)
    // Score by keyword density; high-frequency signal words score higher.
    .with_strategy(ImportanceStrategy::KeywordDensity(vec![
        "error".to_string(),
        "critical".to_string(),
        "user".to_string(),
        "result".to_string(),
    ]));

// Prepare turns (e.g. loaded from EpisodicStore).
let turns: Vec<MemoryTurn> = vec![
    MemoryTurn {
        id: "t1".to_string(),
        role: Role::User,
        content: "Please summarise the quarterly results.".to_string(),
        timestamp: SystemTime::now(),
        importance_score: 0.0,  // will be re-scored by compress()
        token_count: 9,
        tags: vec!["finance".to_string()],
    },
    // ... more turns
];

let (kept_turns, summaries, tokens_saved) = compressor.compress(turns);

println!(
    "Kept {} turns, created {} summaries, saved {} tokens",
    kept_turns.len(),
    summaries.len(),
    tokens_saved,
);

// Summaries carry structured metadata.
for s in &summaries {
    println!("Summary covers {} turns: {}", s.covers_turns.len(), s.summary_text);
    println!("  Key facts: {:?}", s.key_facts);
    println!("  Entities:  {:?}", s.entities);
}
```

### Available `ImportanceStrategy` variants

| Variant | Score formula |
|---|---|
| `KeywordDensity(keywords)` | `min(1, keyword_hits / word_count)` |
| `EntityDensity` | `min(1, entity_count / word_count)` — capitalised-word heuristic |
| `RecencyDecay { decay_per_hour }` | `exp(-decay_per_hour * age_hours)` |
| `Composite([(strategy, weight), ...])` | Weighted average of sub-strategies |

---

## Agent Discovery

`AgentRegistry` is a shared, `Arc`-backed, async capability store. Agents register their capabilities and other agents query for the best match.

```rust
use llm_agent_runtime::discovery::{
    AgentAdvertisement, AgentHealth, AgentRegistry,
    Capability, CapabilityQuery,
};
use std::time::SystemTime;

#[tokio::main]
async fn main() {
    // Shared registry with a 60-second heartbeat TTL.
    let registry = AgentRegistry::new(60);

    // Agent 1 advertises an NLP summarisation capability.
    registry.register(AgentAdvertisement {
        agent_id: "summariser-1".to_string(),
        agent_name: "Summariser".to_string(),
        capabilities: vec![Capability {
            name: "summarise-text".to_string(),
            description: "Summarises long documents into bullet points".to_string(),
            tags: vec!["nlp".to_string(), "summarise".to_string(), "text".to_string()],
            input_schema: None,
            output_schema: None,
            avg_latency_ms: Some(250),
            cost_per_call_usd: Some(0.002),
        }],
        endpoint: Some("http://summariser-1:8080".to_string()),
        registered_at: SystemTime::now(),
        last_heartbeat: SystemTime::now(),
        health: AgentHealth::Healthy,
    }).await;

    // Another agent queries for an NLP capability.
    let matches = registry.query(&CapabilityQuery {
        required_tags: vec!["nlp".to_string()],
        preferred_tags: vec!["summarise".to_string()],
        max_latency_ms: Some(500),
        max_cost_usd: Some(0.01),
    }).await;

    for m in &matches {
        println!(
            "Agent {} — capability '{}' — score {:.2}",
            m.agent_id, m.capability.name, m.score
        );
    }

    // Send heartbeats periodically to prevent TTL eviction.
    registry.heartbeat("summariser-1", AgentHealth::Healthy).await;

    // Evict agents that missed their TTL window.
    registry.evict_stale().await;
}
```

### Scoring details

Capability matching is a two-pass algorithm:

1. **Required-tag filter** — capabilities missing any `required_tags` entry are discarded.
2. **Latency / cost filters** — capabilities exceeding `max_latency_ms` or `max_cost_usd` are discarded.
3. **Score** — surviving capabilities are scored with Jaccard similarity between the capability's tags and `required_tags ∪ preferred_tags`, then multiplied by a health factor (`Healthy=1.0`, `Degraded=0.6`, `Unhealthy/Unknown=0.0`).
4. Results are returned sorted by score descending.

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
| `.with_stop_sequences(v)` | `Vec<String>` | `[]` | Stop sequences forwarded to the provider |
| `.with_loop_timeout_ms(n)` | `u64` | `None` | Wall-clock deadline for the entire ReAct loop |

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
let spec = ToolSpec::new("greet", "Greets someone", |_args| {
    serde_json::json!({ "message": "hello" })
});

// Async handler
let spec = ToolSpec::new_async("fetch", "Fetches a URL", |_args| {
    Box::pin(async move { serde_json::json!({ "status": "ok" }) })
});

// With validation and circuit breaker
let spec = ToolSpec::new("search", "Searches the web", |_args| {
    serde_json::json!({ "results": [] })
})
.with_required_fields(vec!["q".to_string()])
.with_circuit_breaker(cb_arc);
```

### `AgentSession` introspection

| Method | Return | Description |
|---|---|---|
| `step_count()` | `usize` | Total Thought-Action-Observation cycles |
| `has_final_answer()` | `bool` | Whether the session ended with `FINAL_ANSWER` |
| `final_answer()` | `Option<&str>` | The final answer text, if any |
| `duration_secs()` | `f64` | Wall-clock duration in seconds |
| `failed_tool_call_count()` | `usize` | Steps with error-bearing observations |
| `all_thoughts()` | `Vec<&str>` | All thought strings in step order |
| `all_actions()` | `Vec<&str>` | All action strings in step order |
| `all_observations()` | `Vec<&str>` | All observation strings in step order |
| `most_common_action()` | `Option<String>` | Most frequently used action string |
| `step_at_index(i)` | `Option<&ReActStep>` | The step at the given zero-based index |

### `RuntimeMetrics` live methods

| Method | Return | Description |
|---|---|---|
| `total_steps()` | `u64` | Total ReAct steps recorded |
| `total_sessions()` | `u64` | Total completed sessions |
| `top_called_tool()` | `Option<String>` | Tool with the highest total call count |
| `avg_step_latency_ms()` | `f64` | Mean step latency across all recorded latencies |
| `failure_rate_for(tool)` | `f64` | Per-tool failure rate (0.0–1.0) |
| `tool_calls_per_session()` | `f64` | Mean tool calls per completed session |

### `MetricsSnapshot`

Obtained via `runtime.metrics().snapshot()`. All fields are plain integers, safe to log or serialize.

| Method | Return | Description |
|---|---|---|
| `tool_call_count(name)` | `u64` | Total calls for a named tool |
| `tool_failure_count(name)` | `u64` | Total failures for a named tool |
| `failure_rate()` | `f64` | Overall failure rate (0.0–1.0) |
| `most_called_tool()` | `Option<String>` | Tool name with the highest call count |
| `to_json()` | `serde_json::Value` | Serialize for logging or export |

---

## Examples

Run any example with:

```sh
cargo run --example <name> --features <required-features>
```

| Example | Features | Description |
|---|---|---|
| `multi_turn_chat` | `memory` | Multi-turn dialogue with episodic memory |
| `multi_agent` | `memory` | Multiple agents sharing a memory store |
| `resilient_tool` | `orchestrator` | Tool calls protected by circuit breaker |
| `orchestrator_composition` | `orchestrator` | Composed pipeline + retry + dedup |
| `custom_persistence` | `persistence,memory` | Custom checkpointing backend |
| `working_memory_evolution` | `memory` | Working memory LRU eviction |
| `streaming_inference` | _(default)_ | SSE streaming with a stub provider |
| `graph_query_agent` | `graph` | Agent that queries a knowledge graph |
| `anthropic_provider` | `anthropic,memory` | Live Anthropic API call |

---

## Contributing

Contributions are welcome. Please follow these guidelines:

1. **Fork and branch** — create a feature branch from `main` (`git checkout -b feat/my-feature`).
2. **Stay zero-panic** — the project enforces `clippy::unwrap_used = "deny"` and `clippy::panic = "deny"` in all non-test code. Use `?`, `if let`, or `match` instead of `.unwrap()` / `.expect()` in `src/`.
3. **Document public items** — `#![deny(missing_docs)]` is set at the crate root. Every new `pub` item must have a doc comment.
4. **Write tests** — unit tests live in an inline `#[cfg(test)] mod tests` block at the bottom of each module. Use `#[allow(clippy::unwrap_used)]` only inside test modules.
5. **Run the full check suite locally** before opening a PR:
   ```sh
   cargo fmt --check
   cargo clippy --all-features -- -D warnings
   cargo test --all-features
   ```
6. **Open a PR** against `main` with a clear description of what the change does and why.
7. **Changelog** — add a line to `CHANGELOG.md` under the `Unreleased` section (create the file if it does not exist).

For bug reports, please include the `cargo --version`, `rustc --version`, your `Cargo.toml` feature flags, and a minimal reproducible example.

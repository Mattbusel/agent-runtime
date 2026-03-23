# agent-runtime

[![CI](https://github.com/Mattbusel/agent-runtime/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattbusel/agent-runtime/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/llm-agent-runtime.svg)](https://crates.io/crates/llm-agent-runtime)
[![docs.rs](https://docs.rs/llm-agent-runtime/badge.svg)](https://docs.rs/llm-agent-runtime)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust 1.85+](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)
[![Multi-Agent](https://img.shields.io/badge/multi--agent-bus%20%7C%20teams-blueviolet)](#multi-agent-message-bus)
[![Streaming](https://img.shields.io/badge/inference-streaming-brightgreen)](#streaming-inference)

`agent-runtime` is a batteries-included, async-first Rust crate for building production LLM agents. It unifies a ReAct (Thought-Action-Observation) loop, a **Plan-Execute-Verify** structured agent loop, episodic and semantic memory with decay and cosine-similarity recall, **automatic background memory consolidation** via TF-IDF k-means clustering, a directed knowledge graph with centrality and community detection, an orchestration layer with circuit breakers and retry/backpressure, pluggable LLM providers with SSE streaming, optional file-based session checkpointing, intelligent memory compression for long-running agents, a peer-discovery registry, a **multi-agent message bus** with role-based routing, **agent teams** with Star/Mesh/Ring topologies and Majority/Pipeline/Parallel consensus, and **token-by-token streaming inference** with real-time callbacks — all in a single crate, driven by a compile-time typestate builder that makes misconfiguration a compiler error rather than a runtime panic.

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

## Architecture

```
  User Code
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  AgentRuntime  (typestate builder: NeedsConfig → HasConfig → build())   │
│                                                                         │
│  .run_agent(id, prompt, infer_fn)        Single agent, ReAct loop       │
│  .run_team(team_config, infer_fn)        Multi-agent team               │
│  .bus(capacity)                          AgentBus factory               │
└──┬──────────────┬──────────────┬─────────────────────────┬─────────────┘
   │              │              │                         │
   ▼              ▼              ▼                         ▼
┌────────┐  ┌─────────┐  ┌────────────┐  ┌───────────────────────────────┐
│bus.rs  │  │team.rs  │  │streaming.rs│  │  agent.rs  /  runtime.rs      │
│        │  │         │  │            │  │                               │
│AgentBus│  │AgentTeam│  │Streaming   │  │  ReActLoop   ToolRegistry     │
│Broadcast│ │TeamConfig│ │InferenceI  │  │  AgentConfig CircuitBreaker   │
│Role    │  │Consensus│  │Callbacks   │  │  EpisodicStore  GraphStore    │
│Target  │  │Topology │  │TokenStream │  │  WorkingMemory  BackpressureG │
└────────┘  └─────────┘  └────────────┘  └───────────────────────────────┘
```

### New in this release

| Module | Key Types | Purpose |
|--------|-----------|---------|
| `bus` | `AgentBus`, `AgentMessage`, `AgentTarget`, `BusSubscription` | Async broadcast bus for peer-to-peer and role-based agent messaging |
| `team` | `AgentTeam`, `TeamConfig`, `TeamOrchestrator`, `ConsensusStrategy` | Spawn teams of agents with Star/Mesh/Ring topologies and Majority/Pipeline/Parallel consensus |
| `streaming` | `InferenceToken`, `StreamingInference`, `StreamingReActLoop`, `StreamingCallbacks` | Token-by-token streaming ReAct loop with per-token, per-thought, and per-action callbacks |

---

## Multi-Agent Message Bus

The `AgentBus` is an async broadcast channel where agents can publish and subscribe to
messages.  Every subscriber receives every published message; each `BusSubscription`
filters silently for messages addressed to its agent ID or role.

### Targets

| `AgentTarget` variant | Who receives the message |
|---|---|
| `Broadcast` | All current subscribers |
| `Specific(id)` | The single agent with the matching `AgentId` |
| `Role(name)` | Every agent registered with that role name |

### Example

```rust
use llm_agent_runtime::bus::{AgentBus, AgentMessage, AgentTarget};
use llm_agent_runtime::types::AgentId;

#[tokio::main]
async fn main() {
    let bus = AgentBus::new(256);

    let planner = AgentId::new("planner");
    let executor = AgentId::new("executor");

    // Subscribe both agents.  executor registers the "executor" role.
    let mut planner_sub = bus.subscribe(planner.clone(), Some("planner".to_string()));
    let mut executor_sub = bus.subscribe(executor.clone(), Some("executor".to_string()));

    // Planner sends a task directly to the executor.
    bus.send(AgentMessage::new(
        planner.clone(),
        AgentTarget::Specific(executor.clone()),
        "Summarise Q3 results",
    )).unwrap();

    // Executor receives it.
    let msg = executor_sub.recv().await.unwrap();
    println!("executor received: {}", msg.content);

    // Broadcast a status update to all agents.
    bus.send(AgentMessage::new(
        executor.clone(),
        AgentTarget::Broadcast,
        "Task complete",
    )).unwrap();

    let reply = planner_sub.recv().await.unwrap();
    println!("planner received: {}", reply.content);
}
```

You can also create a bus from the runtime:

```rust
let bus = runtime.bus(256);
```

---

## Agent Teams

`AgentTeam` declares a group of collaborating agents.  `TeamOrchestrator` drives the
run using the `AgentBus` for message routing and one of three consensus strategies.

### Communication Topologies

| `CommunicationTopology` | Message routing |
|---|---|
| `Star` (default) | Members report to the leader only |
| `Mesh` | Every agent broadcasts to all others |
| `Ring` | Directed ring: agent i → agent (i+1) % n |

### Consensus Strategies

| `ConsensusStrategy` | How the final answer is chosen |
|---|---|
| `Parallel` (default) | All members work concurrently; leader synthesises their answers |
| `Majority` | Each agent votes; the most frequent answer wins |
| `Pipeline` | Agents run in sequence; each refines the previous agent's output |

### Example

```rust
use llm_agent_runtime::prelude::*;
use llm_agent_runtime::team::{AgentTeam, TeamConfig, ConsensusStrategy, CommunicationTopology};

#[tokio::main]
async fn main() -> Result<(), AgentRuntimeError> {
    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(5, "my-model"))
        .build();

    let team = AgentTeam::new(
        AgentId::new("leader"),
        vec![AgentId::new("analyst-1"), AgentId::new("analyst-2")],
        "Classify these support tickets by severity",
    );

    let config = TeamConfig::new(team)
        .with_topology(CommunicationTopology::Star)
        .with_consensus(ConsensusStrategy::Parallel)
        .with_max_rounds(3);

    // infer is called once per agent — swap for a real provider call.
    let result = runtime
        .run_team(config, |agent_id, prompt| async move {
            format!("{agent_id}: severity=high (stubbed)")
        })
        .await?;

    println!("Team answer: {}", result.final_answer);
    println!("Rounds completed: {}", result.rounds_completed);
    println!("Messages exchanged: {}", result.messages_exchanged);
    Ok(())
}
```

---

## Streaming Inference

`StreamingReActLoop` runs the same Thought-Action-Observation protocol as `ReActLoop`
but consumes tokens one at a time, firing callbacks as they arrive.

### Callbacks

| Callback | Fires when |
|---|---|
| `on_token(token)` | Every `InferenceToken` arrives from the provider |
| `on_thought(text)` | A complete `Thought:` line has been assembled |
| `on_action(action)` | A complete `Action:` line has been parsed |

### Implementing `StreamingInference`

```rust
use llm_agent_runtime::streaming::{InferenceToken, StreamingInference, TokenStream};
use futures::stream;

struct MyProvider;

impl StreamingInference for MyProvider {
    fn infer_stream<'a>(&'a self, prompt: &'a str) -> TokenStream<'a> {
        // In practice, call your LLM API and stream tokens.
        let tokens = vec![
            InferenceToken::thought("I should answer directly"),
            InferenceToken::final_answer("The answer is 42"),
        ];
        Box::pin(stream::iter(tokens))
    }
}
```

### Full streaming example

```rust
use llm_agent_runtime::streaming::{
    InferenceToken, StreamingCallbacks, StreamingInference, StreamingReActLoop, TokenStream,
};
use futures::stream;

struct Stub;
impl StreamingInference for Stub {
    fn infer_stream<'a>(&'a self, _prompt: &'a str) -> TokenStream<'a> {
        Box::pin(stream::iter(vec![
            InferenceToken::thought("I should answer"),
            InferenceToken::final_answer("42"),
        ]))
    }
}

#[tokio::main]
async fn main() {
    let callbacks = StreamingCallbacks::new()
        .on_token(|tok| print!("{}", tok.text))
        .on_thought(|thought| println!("\n[THOUGHT] {thought}"))
        .on_action(|action| println!("[ACTION]  {action:?}"));

    let mut loop_ = StreamingReActLoop::new(Stub, 10)
        .with_callbacks(callbacks)
        .with_tool_handler(|name, args| {
            format!("tool {name} returned: {args}")
        });

    let session = loop_.run("What is 6 * 7?").await.unwrap();

    println!("\nCompleted: {}", session.is_completed());
    println!("Steps: {}", session.step_count());
    println!("Tokens: {}", session.total_token_count());
    println!("Answer: {:?}", session.final_answer());
}
```

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

## Plan-Execute-Verify Loop

The `plan_execute` module provides a **structured three-phase agent loop** as a production-grade alternative to open-ended ReAct for well-defined multi-step workflows:

```
  Goal
   |
   v
+------------------+
|  Plan Phase      |  LLM produces a numbered step list:
|  (1 LLM call)    |  "1. Search web | tool:web_search | expected:results"
+--------+---------+  "2. Summarise  | tool:none       | expected:summary"
         |
         v
+------------------+
|  Execute Phase   |  Steps run in sequence.
|  (1 call/step)   |  Named tools are dispatched directly.
|                  |  Unknown tools fall back to inference.
|  StepStatus:     |
|    Pending       |
|    Running       |
|    Completed(o)  |
|    Failed(e)     |
|    Skipped       |
+--------+---------+
         |
         v
+------------------+
|  Verify Phase    |  LLM receives the full plan + all outputs.
|  (1 LLM call)    |  Returns VerificationResult { achieved, confidence,
|                  |  issues, raw_response }.
+------------------+
```

### Usage

```rust
use llm_agent_runtime::prelude::*;

#[tokio::main]
async fn main() -> Result<(), AgentRuntimeError> {
    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(5, "claude-sonnet-4-6"))
        .build();

    let mut n = 0usize;
    let (plan, verification) = runtime
        .run_plan_execute(
            AgentId::new("researcher"),
            "Research the latest Rust async developments",
            move |ctx: String| {
                n += 1;
                let step = n;
                async move {
                    if step == 1 {
                        // Planning response — numbered steps.
                        "1. Search Rust blog | tool:web_search | expected:recent posts\n\
                         2. Summarise findings | tool:none | expected:summary\n"
                            .to_string()
                    } else if step == 2 {
                        // Step 1 output (tool not registered → inference fallback).
                        "Found: tokio 1.37, async-std 2.0, new stabilisations.".to_string()
                    } else if step == 3 {
                        // Step 2 output.
                        "Summary: Rust async has matured significantly in 2025.".to_string()
                    } else {
                        // Verification.
                        "ACHIEVED 0.92\nBoth steps completed with good output.".to_string()
                    }
                }
            },
        )
        .await?;

    println!("Plan: {} steps, {} succeeded", plan.step_count(), plan.completed_count());
    println!("Goal achieved: {} (confidence: {:.0}%)", verification.achieved,
             verification.confidence * 100.0);
    Ok(())
}
```

### `PlanExecuteConfig` options

| Field | Default | Description |
|---|---|---|
| `max_steps` | `20` | Maximum steps to execute; extras are `Skipped` |
| `stop_on_failure` | `true` | Halt on first failed step |
| `system_prompt` | _see source_ | Injected into the planning prompt |

---

## Memory Consolidation

Long-running agents accumulate hundreds of episodic memories that gradually become redundant.  The `consolidation` module provides a background Tokio task that automatically merges similar memories using **TF-IDF k-means clustering**.

### How it works

1. Every `run_interval_secs` (default: 5 minutes) the consolidator wakes up.
2. For each tracked agent it fetches all episodic memories and computes a bag-of-words **TF-IDF embedding** for each one.
3. It runs **k-means clustering** (deterministic farthest-point initialisation) to group similar memories.
4. Clusters that meet the `min_cluster_size` threshold and whose older members exceed `max_age_secs` have their redundant entries merged into a single `ConsolidatedMemory` summary.
5. The most-recent `keep_recent_per_cluster` items in each cluster are always preserved verbatim.

### Metrics

| Counter | Description |
|---|---|
| `consolidation_runs_total` | Total number of consolidation passes completed |
| `memories_consolidated_total` | Total memories merged or archived |

### Usage

```rust
use llm_agent_runtime::consolidation::{ConsolidationPolicy, MemoryConsolidator};
use llm_agent_runtime::memory::EpisodicStore;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let store = Arc::new(EpisodicStore::new());
    // … populate the store …

    let policy = ConsolidationPolicy {
        min_cluster_size: 3,
        max_age_secs: 3600,       // archive memories older than 1 hour
        similarity_threshold: 0.65,
        run_interval_secs: 300,   // run every 5 minutes
        k_clusters: 8,
        kmeans_iterations: 10,
        keep_recent_per_cluster: 2,
    };

    let consolidator = MemoryConsolidator::new(Arc::clone(&store), policy);
    let metrics = consolidator.metrics();

    // Spawn as a background task — runs indefinitely.
    tokio::spawn(consolidator.run());

    // Periodically inspect consolidated summaries.
    // let summaries = consolidator.take_consolidated();
    println!("Runs: {}", metrics.runs());
}
```

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

## Advanced: Tool Capability Sandbox

The `sandbox` module enforces a capability-based security model — every tool
must declare what permissions it requires, and the runtime only allows calls
when the current grant set satisfies all requirements.

This prevents prompt-injection attacks from escalating to file writes, network
calls, or shell execution without explicit operator approval.

```rust
use llm_agent_runtime::{Sandbox, SandboxConfig, ToolManifest, ToolPermission};

// Build a sandbox granting only file reads for this agent session.
let sandbox = Sandbox::new(SandboxConfig::default())
    .grant(ToolPermission::FileRead);

// Tools declare their required permissions up-front.
sandbox.register_tool(
    ToolManifest::new("read_file")
        .require(ToolPermission::FileRead)
        .with_description("Read a file from disk"),
);
sandbox.register_tool(
    ToolManifest::new("delete_file")
        .require(ToolPermission::FileRead)
        .require(ToolPermission::FileWrite),
);

// Before invoking any tool from the ReAct loop:
match sandbox.check("delete_file") {
    Ok(()) => { /* call the tool */ }
    Err(reason) => {
        // Return a denial observation to the agent so it can adapt.
        println!("TOOL_DENIED: {reason}");
    }
}

// Inspect the audit trail.
for entry in sandbox.audit_log() {
    println!("{}: {} — {}", entry.tool_name,
        if entry.allowed { "ALLOW" } else { "DENY" },
        entry.denied_capabilities.join(", "));
}
```

**Built-in permissions:** `FileRead`, `FileWrite`, `Network`, `DatabaseRead`,
`DatabaseWrite`, `ShellExec`, `SecretAccess`, `Custom(name)`.

---

## Advanced: HTN Goal Planner

The `planner` module adds a **Hierarchical Task Network** planner as a
structured alternative to the open-ended ReAct loop.  For well-defined
multi-step workflows you register decomposition methods once, and the planner
expands any compound goal into a fully ordered sequence of primitive tool calls
before execution begins.

```rust
use llm_agent_runtime::{Planner, PlannerConfig, Task, Method, Precondition};

let mut planner = Planner::new(PlannerConfig { max_depth: 16, max_steps: 256 });

// Register a method: "ResearchCompany" decomposes into 3 primitives.
planner.register_method(Method {
    task: "ResearchCompany".into(),
    subtasks: vec![
        Task::primitive("web_search", r#"{"query":"<company>"}"#),
        Task::primitive("scrape_page", r#"{"url":"<top_result>"}"#),
        Task::primitive("summarise",   r#"{"text":"<content>"}"#),
    ],
    preconditions: vec![],
    description: "Standard research workflow".into(),
});

// Register a conditional variant (only applies when the company is public).
planner.register_method(Method {
    task: "ResearchCompany".into(),
    subtasks: vec![
        Task::primitive("web_search",    r#"{"query":"<company>"}"#),
        Task::primitive("fetch_sec_filings", r#"{"ticker":"<ticker>"}"#),
        Task::primitive("summarise",    r#"{"text":"<content>"}"#),
    ],
    preconditions: vec![Precondition::new("company_type", "public")],
    description: "Research including SEC filings for public companies".into(),
});

// Plan against a world state.
let mut world = std::collections::HashMap::new();
world.insert("company_type".into(), "public".into());
let plan = planner.plan_with_world(Task::compound("ResearchCompany"), &world)?;

println!("Plan has {} steps:", plan.len());
for step in plan.steps() {
    println!("  [depth={}] {} {}", step.depth, step.tool, step.args);
}
```

HTN planning is most useful when:
- You have well-known multi-step workflows that always follow the same structure.
- You want to enforce step ordering without relying on the LLM to rediscover it.
- Preconditions (world state) should select between alternative strategies.

Combine with the ReAct loop: use the planner to build the top-level task
structure, then let `ReActLoop` handle each primitive step with its own
Thought-Action-Observation cycle.

---

## Agent Swarm Coordination

`SwarmOrchestrator` decomposes a complex task into parallelisable subtasks,
dispatches each to a registered worker agent, and synthesises the results into
one coherent answer.  A configurable convergence threshold lets the orchestrator
short-circuit as soon as enough results arrive, avoiding unnecessary latency
when a partial answer is sufficient.

```rust,no_run
use llm_agent_runtime::swarm_v2::{SwarmConfig, SwarmOrchestrator, TaskSplitStrategy};
use llm_agent_runtime::types::AgentId;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = SwarmConfig {
        max_agents: 4,
        task_split_strategy: TaskSplitStrategy::Breadth,
        // Stop once 80 % of subtasks have completed successfully.
        convergence_threshold: 0.8,
    };

    let mut orchestrator = SwarmOrchestrator::new(config);
    orchestrator.add_agent(AgentId::new("worker-1"));
    orchestrator.add_agent(AgentId::new("worker-2"));
    orchestrator.add_agent(AgentId::new("worker-3"));

    // The `inference_fn` is called concurrently for every subtask.
    let result = orchestrator
        .run_swarm(
            "Research Rust async runtimes, compare with C++ coroutines, and summarise tradeoffs",
            |agent_id, subtask_prompt| async move {
                // Replace with a real provider call in production.
                format!("[{agent_id}] completed: {subtask_prompt}")
            },
        )
        .await?;

    println!("Threshold met: {}", result.threshold_met);
    println!("Failures:      {}", result.failure_count);
    println!("\n{}", result.converged_answer);
    Ok(())
}
```

### `SwarmConfig`

| Field | Type | Default | Description |
|---|---|---|---|
| `max_agents` | `usize` | `0` | Maximum concurrently active agents (`0` = unlimited) |
| `task_split_strategy` | `TaskSplitStrategy` | `Breadth` | How the task is decomposed into subtasks |
| `convergence_threshold` | `f64` | `1.0` | Fraction of successful subtasks required before early exit |

### `TaskSplitStrategy`

| Variant | Behaviour |
|---|---|
| `Breadth` | Split on commas, semicolons, and ` and ` — produces a flat parallel list |
| `Depth` | Recursively split on sequencing connectors (` then `, ` next `, …) for DFS ordering |

---

## High-Level Memory Compression

`memory_compress` builds on the low-level `memory_compression` module with
three composable strategies that can be chained in sequence.

| Strategy | What it does |
|---|---|
| `TemporalCompression` | Keeps the N most-recent turns verbatim; merges older turns into `CompressedEpisode` windows |
| `SemanticDedup` | Removes near-duplicate turns using Jaccard similarity on word sets |
| `ImportanceFilter` | Re-inserts turns tagged `"important"` or with a high importance score that were removed by other strategies |

The async `MemoryCompressor::run_if_needed` method is designed to be called
from a `tokio::spawn` background task so compression never blocks the agent loop.

```rust
use llm_agent_runtime::memory_compress::{CompressorConfig, MemoryCompressor};
use llm_agent_runtime::memory_compression::{MemoryTurn, Role};
use std::time::SystemTime;

# tokio_test::block_on(async {
let config = CompressorConfig {
    max_turns: 50,          // trigger when buffer exceeds 50 turns
    recency_keep: 10,       // always keep the 10 most-recent turns verbatim
    jaccard_dedup_threshold: 0.85, // remove near-duplicates with ≥ 85 % word overlap
    importance_tag: "important".to_string(),
};

let compressor = MemoryCompressor::new(config);

let turns: Vec<MemoryTurn> = (0..60).map(|i| MemoryTurn {
    id: format!("t{i}"),
    role: Role::User,
    content: format!("message {i}"),
    timestamp: SystemTime::now(),
    importance_score: if i == 5 { 0.95 } else { 0.1 },
    token_count: 8,
    tags: if i == 5 { vec!["important".into()] } else { vec![] },
}).collect();

let (kept, episodes, triggered) = compressor.run_if_needed(turns).await;
println!("Triggered: {triggered}");
println!("Kept {} turns, {} compressed episodes", kept.len(), episodes.len());

for ep in &episodes {
    println!(
        "Episode '{}': {} → {} turns over {:?}",
        ep.id, ep.original_count, ep.compressed_to, ep.time_range
    );
}
# });
```

### `CompressedEpisode` fields

| Field | Type | Description |
|---|---|---|
| `id` | `String` | Unique identifier for this compressed block |
| `original_count` | `usize` | Number of turns before compression |
| `compressed_to` | `usize` | Number of representative turns kept |
| `summary` | `String` | Prose description of what was in the window |
| `time_range` | `(u64, u64)` | Unix-epoch seconds of earliest/latest compressed turn |
| `common_tags` | `Vec<String>` | Tags present in ≥ 50 % of the compressed turns |

---

## Skill Marketplace

The `marketplace` module provides a runtime-discoverable registry of named
skills, a filesystem loader, a task-description-based matcher, and a pipeline
composer.

```rust,no_run
use llm_agent_runtime::marketplace::{Skill, SkillRegistry, SkillMatcher, SkillComposer};
use std::sync::Arc;

fn main() {
    // Build a shared registry — wrap in Arc for multi-task sharing.
    let registry = Arc::new(SkillRegistry::new());

    // Register skills.
    registry.register(Skill::new(
        "web_search",
        "Search the web for up-to-date information",
        "1.0.0",
        "core-team",
        vec!["search".into(), "web".into(), "retrieval".into()],
    ));
    registry.register(Skill::new(
        "summarise",
        "Condense long text into a concise summary",
        "1.2.0",
        "core-team",
        vec!["nlp".into(), "summarise".into(), "text".into()],
    ));
    registry.register(Skill::new(
        "translate",
        "Translate text between languages",
        "1.0.0",
        "i18n-team",
        vec!["nlp".into(), "translation".into(), "language".into()],
    ));

    // --- SkillLoader: discover skills from ~/.agent-runtime/skills/*.toml ---
    // use llm_agent_runtime::marketplace::SkillLoader;
    // let loader = SkillLoader::default();
    // let n = loader.load_into(&registry);
    // println!("Loaded {n} skills from disk");

    // --- SkillMatcher: find relevant skills for a task ---
    let matcher = SkillMatcher::new(Arc::clone(&registry));
    let matches = matcher.match_skills(
        "I need to search the web for recent news and summarise it",
        3,
    );
    for m in &matches {
        println!("  {:<20} score={:.2f}", m.skill.name, m.score);
    }

    // --- SkillComposer: build a multi-skill pipeline ---
    let composer = SkillComposer::new(Arc::clone(&registry));
    let pipeline = composer.compose(vec!["web_search".into(), "summarise".into()]);
    println!("Pipeline: {}", pipeline.description);
    println!("Stages:   {:?}", pipeline.stage_names());

    // Auto-compose from task description.
    let auto_pipeline = composer.compose_for_task("translate and summarise a document", 2);
    println!("Auto pipeline: {}", auto_pipeline.description);
}
```

### Skill manifest TOML (`~/.agent-runtime/skills/my_skill.toml`)

```toml
name         = "sentiment_analysis"
description  = "Classify the sentiment of a text passage"
version      = "0.3.1"
author       = "ml-team"
capabilities = ["nlp", "sentiment", "classification", "text"]
```

`SkillLoader::default()` scans `~/.agent-runtime/skills/` for `*.toml` files
and registers each as a `Skill` in the registry.  Malformed files are logged
and skipped without aborting the load.

### Scoring details (SkillMatcher)

The score for each skill against a task description is:

```
score = 0.5 × (capability_overlap / capability_count)
      + 0.5 × (description_word_overlap / description_word_count)
```

Common stop-words are stripped before comparison.  Results are returned sorted
by score descending; a `min_score` filter (default `0.1`) suppresses skills
with no meaningful overlap.

---

---

## Tool Validation

The `tool_validator` module provides JSON-schema-like validation of tool call
arguments before execution, helping catch malformed inputs at the boundary
rather than inside handler logic.

### Key Types

| Type | Role |
|---|---|
| `ToolSchema` | Describes one tool: name, description, and a list of `ParameterSpec`s |
| `ParameterSpec` | One parameter: name, `ParameterTypeHint`, required flag, description |
| `ParameterTypeHint` | `String`, `Number`, `Boolean`, `Array`, `Object`, or `Any` |
| `ToolCall` | A concrete invocation: `tool_name` + JSON `arguments` |
| `SchemaValidator` | Registry of schemas; validates a `ToolCall` against the matching schema |
| `ValidationError` | `MissingRequired(String)`, `TypeMismatch { param, expected, got }`, `UnknownTool(String)` |

### Validation Rules

1. `tool_name` must match a registered schema (`UnknownTool` otherwise).
2. Every required parameter must be present (`MissingRequired` otherwise).
3. Parameters that are present are type-checked against their `ParameterTypeHint` (`TypeMismatch` otherwise).
4. Extra keys not declared in the schema are silently ignored.

### Example

```rust
use llm_agent_runtime::tool_validator::{
    ParameterSpec, ParameterTypeHint, SchemaValidator, ToolCall, ToolSchema,
};
use serde_json::json;

// Declare the schema once.
let schema = ToolSchema::new("search", "Web search")
    .with_param(ParameterSpec::required("query", ParameterTypeHint::String, "Search query"))
    .with_param(ParameterSpec::optional("limit", ParameterTypeHint::Number, "Max results"));

let mut validator = SchemaValidator::new();
validator.register(schema);

// Validate before dispatching.
let call = ToolCall::new("search", json!({ "query": "rust async", "limit": 10 }));
validator.validate(&call).expect("valid call");

// Collect all errors at once (does not short-circuit).
let errors = validator.validate_all(&call);
assert!(errors.is_empty());
```

---

## Agent Metrics

The `agent_metrics` module provides structured, per-agent metrics tracking for
multi-agent systems.  Unlike the crate-level `RuntimeMetrics` (which aggregates
across all sessions), `AgentMetrics` tracks each agent independently.

### Key Types

| Type | Role |
|---|---|
| `AgentMetrics` | Mutable live counters for one agent: steps, tokens, tool calls, latency, memory |
| `AgentMetricsSnapshot` | Serialisable (`serde`) point-in-time capture of `AgentMetrics` |
| `AgentMetricsRegistry` | `Arc<Mutex<HashMap<String, AgentMetrics>>>` — thread-safe multi-agent registry |

### Tracked Fields

| Field | Description |
|---|---|
| `total_steps` | ReAct / plan-execute steps completed |
| `total_tokens_in` | Input tokens consumed across all LLM calls |
| `total_tokens_out` | Output tokens produced across all LLM calls |
| `tool_calls_made` | Tool calls dispatched (successful + failed) |
| `tool_call_failures` | Tool calls that returned an error |
| `avg_step_latency_ms` | Running incremental mean of per-step latency |
| `peak_memory_kb` | Highest memory sample observed, in KB |
| `session_start` | `Instant` at which the metrics object was created |

### Example

```rust
use llm_agent_runtime::agent_metrics::AgentMetricsRegistry;

// Registry is Arc-backed: cheap to clone, safe to share across tasks.
let registry = AgentMetricsRegistry::new();
let r2 = registry.clone(); // shares the same data

// Record activity from any task.
registry.record_step("agent-1", 45 /* ms */);
registry.record_step_with_memory("agent-1", 30, 2048 /* KB */);
registry.record_tool_call("agent-1", false /* success */);
registry.record_tool_call("agent-1", true  /* failure */);
registry.record_tokens("agent-1", 512, 128);

// Snapshot is Serialize/Deserialize — log it, store it, or send it over the wire.
let snap = registry.snapshot("agent-1").expect("agent was registered");
println!("steps={} failure_rate={:.2}", snap.total_steps, snap.failure_rate);

// Aggregate view across all agents.
let all = registry.snapshot_all();
println!("{} agents tracked", all.len());
```

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

---

## Tool Registry

The built-in `ToolRegistry` lets you register named async tools and call them by name.
Three tools ship out of the box: `echo`, `calculator`, and `timestamp`.

```
  ┌──────────────────────────────────────────────────────────┐
  │                     ToolRegistry                         │
  │                                                          │
  │  register(Arc<dyn Tool>)                                 │
  │  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐   │
  │  │   echo   │  │  calculator  │  │   timestamp      │   │
  │  └──────────┘  └──────────────┘  └──────────────────┘   │
  │                                                          │
  │  call(name, input) ──► Result<String, ToolError>         │
  │  list()            ──► Vec<ToolInfo>                     │
  │  with_timeout(dur) ──► wraps every call                  │
  └──────────────────────────────────────────────────────────┘
```

### Usage

```rust
use std::sync::Arc;
use llm_agent_runtime::tools::{ToolRegistry, EchoTool, CalculatorTool, TimestampTool};

let mut registry = ToolRegistry::new();
registry.register(Arc::new(EchoTool));
registry.register(Arc::new(CalculatorTool));
registry.register(Arc::new(TimestampTool));

// Add a timeout
let registry = registry.with_timeout(std::time::Duration::from_secs(5));

// Call a tool
tokio::runtime::Runtime::new().unwrap().block_on(async {
    let result = registry.call("echo", "hello").await.unwrap();
    assert_eq!(result, "hello");

    let sum = registry.call("calculator", "(2 + 3) * 4").await.unwrap();
    assert_eq!(sum, "20");

    let ts = registry.call("timestamp", "").await.unwrap();
    println!("Current UTC: {ts}"); // e.g. "2026-03-22T10:00:00Z"

    // Inspect per-tool stats
    for info in registry.list().await {
        println!("{}: {} calls, {} errors, {:.2}ms avg",
            info.name, info.call_count, info.error_count, info.avg_latency_ms);
    }
});
```

### Custom tools

```rust
use llm_agent_runtime::tools::{Tool, ToolError};
use async_trait::async_trait;

struct UppercaseTool;

#[async_trait]
impl Tool for UppercaseTool {
    fn name(&self) -> &str { "uppercase" }
    fn description(&self) -> &str { "Converts input to uppercase." }
    async fn call(&self, input: &str) -> Result<String, ToolError> {
        Ok(input.to_uppercase())
    }
}
```

### Accessing the registry from AgentRuntime

```rust
use llm_agent_runtime::{AgentRuntime, AgentConfig};
use std::sync::Arc;

let runtime = AgentRuntime::builder()
    .with_agent_config(AgentConfig::new(5, "my-model"))
    .build();

// Access the registry
let registry = runtime.tool_registry();
```

---

## Agent Checkpointing

`CheckpointManager` wraps a pluggable `CheckpointStore` with auto-checkpointing
and per-agent rotation of old checkpoints.

```
  ┌─────────────────────────────────────────────────────────────┐
  │                   CheckpointManager                         │
  │                                                             │
  │  maybe_checkpoint(agent_id, step, memory, ctx, force)       │
  │       │                                                     │
  │       ▼  (every N steps, or force=true)                     │
  │  ┌──────────────────────────┐                               │
  │  │     CheckpointStore      │◄──── InMemoryCheckpointStore  │
  │  │  save / load / list /    │◄──── FileCheckpointStore      │
  │  │  delete                  │◄──── (custom implementation)  │
  │  └──────────────────────────┘                               │
  │                                                             │
  │  Rotation: keeps ≤ max_checkpoints per agent                │
  │  (oldest by created_at deleted automatically)               │
  └─────────────────────────────────────────────────────────────┘
```

### In-memory store (testing)

```rust
use std::sync::Arc;
use llm_agent_runtime::checkpoint::{
    CheckpointManager, InMemoryCheckpointStore, MemoryEntry,
};

let store = Arc::new(InMemoryCheckpointStore::new());
let mgr = CheckpointManager::new(store, /* interval_steps */ 5, /* max_checkpoints */ 10);

tokio::runtime::Runtime::new().unwrap().block_on(async {
    // Auto-checkpoint at steps 5, 10, 15, ...
    mgr.maybe_checkpoint(
        "agent-1", 5,
        vec![MemoryEntry { key: "task".into(), value: "summarise doc".into() }],
        "step 5 context",
        false, // not forced
    ).await.unwrap();

    // Load the latest checkpoint
    if let Some(cp) = mgr.load_latest("agent-1").await.unwrap() {
        println!("Restored step {} for {}", cp.step, cp.agent_id);
    }
});
```

### File store (production)

```rust
use std::sync::Arc;
use llm_agent_runtime::checkpoint::{CheckpointManager, FileCheckpointStore};

let store = Arc::new(FileCheckpointStore::new("/var/lib/my-agent/checkpoints"));
let mgr = CheckpointManager::new(store, 10, 5);
```

### Accessing the checkpoint manager from AgentRuntime

```rust
use llm_agent_runtime::{AgentRuntime, AgentConfig};

let runtime = AgentRuntime::builder()
    .with_agent_config(AgentConfig::new(5, "my-model"))
    .build();

// Access the checkpoint manager
let mgr = runtime.checkpoint();
```

# Performance Tuning Guide

This document covers the knobs available in `llm-agent-runtime` for squeezing
throughput and reducing latency in production deployments.

---

## 1. Agent configuration

`AgentConfig` exposes three runtime parameters that have the largest impact on
per-request performance:

| Field | Default | Effect |
|-------|---------|--------|
| `max_iterations` | caller-set | Caps the ReAct loop; lower values mean faster worst-case latency at the cost of answer quality. |
| `temperature` | `None` (provider default) | Lower temperatures (0.0–0.3) reduce token variance and can speed up sampling on some backends. |
| `max_tokens` | `None` (provider default) | Setting an explicit cap prevents runaway generation costs. |
| `request_timeout` | `None` (no timeout) | Always set a timeout in production to bound worst-case latency. |

```rust
let config = AgentConfig::new(10, "gpt-4o")
    .with_temperature(0.2)
    .with_max_tokens(512)
    .with_request_timeout(std::time::Duration::from_secs(30));
```

---

## 2. Tool caching

For tools that are deterministic or change infrequently (e.g. knowledge-base
lookups, static data retrieval) implement the `ToolCache` trait and attach it
to `ToolRegistry`:

```rust
registry.with_cache(Arc::new(MyInMemoryCache::new()));
```

The cache is checked before invoking the tool handler, and results are stored
after a successful call.  This eliminates redundant LLM tool invocations when
the same arguments are seen multiple times within a session.

---

## 3. Pipeline parallelism

`Pipeline::execute` runs stages sequentially.  If your stages are independent,
split them into separate pipelines and run them with `tokio::join!`:

```rust
let (result_a, result_b) = tokio::join!(
    pipeline_a.execute(input.clone()),
    pipeline_b.execute(input.clone()),
);
```

Use `Pipeline::execute_timed` during development to measure per-stage
wall-clock time and identify bottlenecks before committing to a topology.

---

## 4. Memory recall

`SemanticStore::retrieve_similar` performs a linear scan over all stored
embeddings.  For stores with more than ~10 000 entries consider:

- Sharding entries across multiple `SemanticStore` instances by topic.
- Reducing `expected_dim` — smaller embedding dimensions lower both memory
  footprint and dot-product cost.
- Running retrieval in a `tokio::task::spawn_blocking` block to avoid blocking
  the async executor during the scan.

`EpisodicStore` bounds memory per agent via `per_agent_capacity`.  Use the
`EvictionPolicy::Oldest` variant to prefer keeping recent episodes when
capacity is tight; use the default `LowestImportance` to retain high-importance
episodes regardless of age.

---

## 5. Graph traversal

`GraphStore::bfs_bounded` and `dfs_bounded` accept `max_depth` and `max_nodes`
limits.  In production always set both to prevent unbounded traversal of large
graphs:

```rust
let nodes = graph.bfs_bounded("start-entity", /*max_depth=*/ 3, /*max_nodes=*/ 200);
```

Prefer BFS for "find nodes near the root" queries; prefer DFS for "explore a
specific branch" queries.

---

## 6. Metrics-driven tuning

`RuntimeMetrics` tracks tool call counts, error rates, and — via
`LatencyHistogram` — step latency distribution.  After a load test call
`metrics.snapshot()` and inspect:

- `step_latency.mean_ms()` — average ReAct step time.
- `step_latency.buckets()` — histogram counts per latency tier
  (`<1 ms`, `<5 ms`, `<25 ms`, `<100 ms`, `<250 ms`, `<1 000 ms`, `≥1 000 ms`).
- `tool_calls` / `tool_errors` — tool error rate.

Emit these values to your observability platform on every `AgentSession` close.

---

## 7. Tokio runtime sizing

`llm-agent-runtime` is built on Tokio.  For CPU-bound workloads (embedding
generation, graph traversal) increase the worker thread count:

```rust
#[tokio::main(flavor = "multi_thread", worker_threads = 8)]
async fn main() { /* … */ }
```

For I/O-bound workloads (LLM API calls, file persistence) the default
`multi_thread` flavor with `available_parallelism` threads is usually optimal.

---

## 8. Release profile

The project ships a `[profile.release]` in `Cargo.toml` with:

- `lto = true` — link-time optimisation across crate boundaries.
- `codegen-units = 1` — single codegen unit for maximum inlining.
- `opt-level = 3` — full optimisation.
- `panic = "abort"` — removes unwinding machinery (~10 % binary size reduction).

Always benchmark with `--release`; debug builds can be 10–50× slower due to
the absence of inlining and bounds-check elimination.

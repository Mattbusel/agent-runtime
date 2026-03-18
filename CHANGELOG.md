# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-03-17

### Added

- `#![deny(missing_docs)]` in `lib.rs` — enforces `///` documentation on every
  public item at compile time; CI will fail on any undocumented addition.
- Three new integration test files covering:
  - **Agent lifecycle** (`tests/test_agent_lifecycle.rs`): create, run, two-step
    session, unique session IDs, FINAL_ANSWER termination, max-iterations error,
    and metrics consistency.
  - **Memory persistence** (`tests/test_memory_persistence.rs`): episodic write/
    read-back, importance ordering, agent isolation, recall limit, shared-Arc clone
    semantics, working-memory set/overwrite/entries/clear, semantic tag filtering,
    cosine-similarity read-back, and memory injection into the agent prompt context.
  - **Error paths** (`tests/test_error_paths.rs`): invalid config (zero capacity,
    negative half-life, zero retry attempts), backpressure shedding, shed-count
    metric increment, max-iterations error, missing-tool observation, required-field
    validation, provider-failure simulation, and circuit-breaker fast-fail (behind
    `#[cfg(feature = "orchestrator")]`).
- `.github/workflows/release.yml` — tag-triggered release workflow that verifies
  formatting, runs Clippy and all tests, builds docs with private items, and
  publishes to crates.io.
- `cargo doc --no-deps --document-private-items --all-features` step added to the
  existing CI workflow so private-item docs are verified on every PR.

### Changed

- Version bumped from `0.1.0` to `0.2.0`.

---

## [Unreleased] (previous passes)

### Added (production-ready pass — 2026-03-17)

- `RuntimeMetrics::reset()` — zeroes all atomic counters (useful in tests and for gauge resets).
- `RuntimeMetrics::to_snapshot()` — returns all seven counters as a plain tuple for easy assertion.
- Inline `#[cfg(test)]` module in `metrics.rs` with 12 unit tests covering: zero-initialisation,
  per-counter increment, `reset()`, `to_snapshot()` value accuracy, `Send + Sync` assertion,
  cumulative increments, and `Arc` clone state sharing.
- Inline `#[cfg(test)]` modules were already present in `error.rs`, `agent.rs`, `memory.rs`,
  `orchestrator.rs`, `persistence.rs`, `providers.rs`, and `runtime.rs`; this pass completes
  the full-module coverage requirement by filling the gap in `metrics.rs`.
- Error handling: confirmed all `.unwrap()` calls in non-test `src/` code are either
  inside `#[cfg(test)]` blocks or carry an `#[allow(clippy::unwrap_used)]` annotation with a
  safety comment (typestate guarantee in `runtime.rs:build()`). No new panicking paths added.

---

## [Unreleased] (previous)

### Added

- Comprehensive README.md with architecture ASCII diagram, quickstart guide, feature-flag table,
  configuration reference, testing instructions, and contributing guidelines
- CHANGELOG.md (this file) with full capability listing for the initial release
- `.gitignore` covering `target/`, `.env`, `*.pdb`, `.DS_Store`, and `Thumbs.db`
- Doc comments (`///`) added to all previously undocumented public items:
  `Message::new`, `Role` enum variants (`System`, `User`, `Assistant`, `Tool`),
  `Message` struct fields, `AgentError` enum variants, and `RuntimeMetrics` accessor methods

---

## [0.1.0] - 2026-03-17

### Added

- `AgentRuntime` with typestate builder pattern enforcing compile-time configuration.
  `AgentRuntimeBuilder<NeedsConfig>` has no `build()` method; calling `with_agent_config()`
  transitions to `AgentRuntimeBuilder<HasConfig>` which exposes `build()`. This is a
  compile-time guarantee, not a runtime check.

- `AgentSession` representing the result of a single `run_agent` call, containing step list,
  memory hit count, graph lookup count, wall-clock duration, and a UUID v4 session identifier.
  Session and per-step checkpointing via `save_checkpoint` / `load_checkpoint` /
  `load_step_checkpoint` when the `persistence` feature is enabled.

- `ReActLoop` implementing the Thought-Action-Observation agent loop. Parses model responses
  for `Thought:` and `Action:` prefixes (case-insensitive, tolerates spaces around the colon).
  Terminates on `FINAL_ANSWER` or when `max_iterations` is exhausted.

- `ToolRegistry` with support for both synchronous and asynchronous tool handlers, per-tool
  required-field validation returning structured error observations, and optional per-tool
  `CircuitBreaker` integration (`orchestrator` feature).

- `ToolSpec` with `new` (sync closure) and `new_async` (async closure) constructors,
  `with_required_fields`, and `with_circuit_breaker` builder methods.

- `AgentConfig` with `max_iterations`, `model`, `system_prompt`, `max_memory_recalls`, and
  `max_memory_tokens` (approximate 4-chars-per-token budget for injected episodic memories).

- `EpisodicStore` for event-based memories with:
  - `DecayPolicy::exponential(half_life_hours)` for time-based importance decay
  - `RecallPolicy::Importance` (default) and `RecallPolicy::Hybrid { recency_weight, frequency_weight }`
  - Per-agent capacity limits with lowest-importance eviction
  - `recall_count` tracking incremented on every retrieval
  - `add_episode_at` for inserting memories with explicit timestamps

- `SemanticStore` for fact-based knowledge:
  - Tag-based retrieval with intersection semantics (all tags must match)
  - Cosine-similarity vector search via `store_with_embedding` / `retrieve_similar`

- `WorkingMemory` as a bounded LRU key-value store:
  - Deterministic eviction of the oldest entry when capacity is exceeded
  - `set`, `get`, `clear`, `len`, `is_empty`, `entries` operations

- `GraphStore` in-memory directed knowledge graph:
  - Entity CRUD: `add_entity`, `get_entity`, `remove_entity` (cascades relationships)
  - Relationship management with duplicate-edge rejection `(from, to, kind)` triple
  - `bfs` and `dfs` traversal (non-recursive, stack-safe)
  - `shortest_path` by hop count (BFS)
  - `shortest_path_weighted` using Dijkstra's algorithm (rejects negative weights)
  - `transitive_closure` returning all reachable entities
  - `degree_centrality` (normalized in+out degree)
  - `betweenness_centrality` using Brandes' algorithm (O(V*E), normalized for directed graphs)
  - `label_propagation_communities` with configurable iteration limit
  - `subgraph` extraction from a list of entity IDs
  - `entity_count` and `relationship_count` accessors

- `CircuitBreaker` with:
  - Configurable `threshold` (consecutive failures before opening) and `recovery_window`
  - State machine: `Closed` -> `Open` -> `HalfOpen` -> `Closed`
  - Pluggable `CircuitBreakerBackend` trait for distributed state sharing
  - `InMemoryCircuitBreakerBackend` as the default single-process backend
  - `with_backend` builder method for injecting custom backends
  - `state()` and `failure_count()` introspection

- `RetryPolicy::exponential(max_attempts, base_ms)` with delays capped at `MAX_RETRY_DELAY`
  (60 seconds). `delay_for(attempt)` computes `base_delay * 2^(attempt-1)`.

- `Deduplicator` with TTL-based cache and in-flight request tracking:
  - `check_and_register` returns `New`, `Cached(result)`, or `InProgress`
  - `complete` moves a key from in-flight to cached

- `BackpressureGuard` with hard capacity limit and optional soft limit:
  - `try_acquire` / `release` for slot management
  - `with_soft_limit` emits a tracing warning when depth reaches the threshold
  - `depth` and `soft_depth_ratio` observability methods

- `Pipeline` of composable string-transform stages:
  - `add_stage(name, handler)` builder method
  - `run(input)` executes stages in insertion order, short-circuiting on first error

- `PersistenceBackend` async trait (`async-trait` powered) with `save`, `load`, and `delete`
  methods. Object-safe; usable behind `Arc<dyn PersistenceBackend>`.

- `FilePersistenceBackend` storing each key as `<sanitized_key>.bin` in a base directory.
  Key sanitization prevents path-traversal escapes. `load` returns `None` for missing keys;
  `delete` is a no-op for missing keys.

- `LlmProvider` async trait with `complete` and `stream_complete` methods. The default
  `stream_complete` wraps `complete` as a single-chunk channel, allowing providers to
  override with true streaming.

- `AnthropicProvider` (feature `anthropic`): Anthropic Messages API with SSE streaming support.
  Hard-coded to `max_tokens = 1024`; configurable model via `complete` / `stream_complete`.

- `OpenAiProvider` (feature `openai`): OpenAI Chat Completions API with SSE streaming support.
  `with_base_url` constructor enables Azure OpenAI and other compatible endpoints.

- `RuntimeMetrics` with atomic counters (lock-free, `Send + Sync`):
  `active_sessions`, `total_sessions`, `total_steps`, `total_tool_calls`, `failed_tool_calls`,
  `backpressure_shed_count`, `memory_recall_count`.

- `prelude` module re-exporting all commonly used types for `use agent_runtime::prelude::*`.

- Feature flags: `orchestrator` (default), `memory` (default), `graph` (default), `wasm` (default),
  `persistence`, `providers`, `anthropic`, `openai`, `redis-circuit-breaker`, `full`.

- Clippy deny rules: `unwrap_used`, `expect_used`, `panic`, `todo` enforced in production code.

- Comprehensive test suite:
  - Unit tests for every public function and error path
  - Concurrency stress test for `BackpressureGuard`
  - Integration tests for `AgentRuntime` wiring: memory injection, graph lookups,
    backpressure shedding, working memory injection, token budget limiting,
    per-step checkpointing, and metrics accuracy
  - Custom backend tests for `CircuitBreaker` shared state
  - Vector similarity tests for `SemanticStore`
  - Hybrid recall policy tests for `EpisodicStore`

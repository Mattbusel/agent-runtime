# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [1.0.0] - 2026-03-18

### Added

- **`[profile.release]`** in `Cargo.toml` with `opt-level = 3`, `lto = true`,
  `codegen-units = 1`, `strip = true`, and `panic = "abort"` for production-optimised
  release binaries.
- **`[lints.clippy]`** extended with `dbg_macro`, `print_stdout`, and `print_stderr`
  warnings in addition to the existing `unwrap_used`, `expect_used`, `panic`, and `todo`
  denies.
- **`[lints.rust]`** — `missing_docs = "warn"` added as belt-and-suspenders alongside the
  existing `#![deny(missing_docs)]` in `lib.rs`.
- **`authors`** field added to `[package]`.
- **`exclude`** field added to `[package]` to omit `.github/` and `CHANGELOG.md` from
  published crates.io tarball.
- **CI release build steps** — `cargo build --release` (default features) and
  `cargo build --release --all-features` added to `.github/workflows/ci.yml` to verify
  optimised compilation on every PR.
- **`RUSTDOCFLAGS="-D warnings"`** added to all `cargo doc` steps in CI so that broken
  intra-doc links and missing doc comments are caught as CI failures.
- **Clippy feature matrix in CI** expanded: added `persistence` and `providers` feature
  combinations to the Clippy step list.
- **README.md** fully rewritten with architecture diagram, quickstart (with and without
  external services), complete API reference tables, error-handling guide, running tests
  section, and a PR checklist.

### Changed

- `thiserror` dependency updated from `"1"` to `"2"` for the latest zero-overhead error
  derive improvements.
- Version bumped from `0.2.0` to `1.0.0` to reflect API stability.

---

## [0.2.0] - 2026-03-17

### Added

- `#![deny(missing_docs)]` in `lib.rs` — enforces `///` documentation on every public item
  at compile time; CI fails on any undocumented addition.
- Three integration test files:
  - `tests/test_agent_lifecycle.rs` — create, run, two-step session, unique session IDs,
    `FINAL_ANSWER` termination, max-iterations error, metrics consistency.
  - `tests/test_memory_persistence.rs` — episodic write/read-back, importance ordering,
    agent isolation, recall limit, shared-`Arc` clone semantics, working-memory
    set/overwrite/entries/clear, semantic tag filtering, cosine-similarity read-back, and
    memory injection into the agent prompt context.
  - `tests/test_error_paths.rs` — invalid config (zero capacity, negative half-life, zero
    retry attempts), backpressure shedding, shed-count metric increment, max-iterations
    error, missing-tool observation, required-field validation, provider-failure simulation,
    and circuit-breaker fast-fail (behind `#[cfg(feature = "orchestrator")]`).
- `.github/workflows/release.yml` — tag-triggered release workflow that verifies
  formatting, runs Clippy and all tests, builds docs with private items, and publishes to
  crates.io.
- `cargo doc --no-deps --document-private-items --all-features` step added to CI.

### Changed

- Version bumped from `0.1.0` to `0.2.0`.

### Fixed

- Integration tests that accessed the private `EpisodicStore::inner` field now use the
  new `bump_recall_count_by_content` helper and the public `add_episode_at` API.
- `graph_entity_with_properties_stored_correctly` test updated to use
  `Entity::with_properties(id, label, props)` correctly.
- `link_w` test helper corrected from `f64` to `f32` to match `Relationship::new`
  signature.

---

## [0.1.0] - 2026-03-17

### Added

- `AgentRuntime` with typestate builder pattern enforcing compile-time configuration.
  `AgentRuntimeBuilder<NeedsConfig>` has no `build()` method; calling
  `with_agent_config()` transitions to `AgentRuntimeBuilder<HasConfig>` which exposes
  `build()`.

- `AgentSession` representing the result of a single `run_agent` call, containing step
  list, memory hit count, graph lookup count, wall-clock duration, and a UUID v4 session
  identifier.  Session and per-step checkpointing via `save_checkpoint` /
  `load_checkpoint` / `load_step_checkpoint` when the `persistence` feature is enabled.

- `ReActLoop` implementing the Thought-Action-Observation agent loop.  Parses model
  responses for `Thought:` and `Action:` prefixes (case-insensitive, tolerates spaces
  around the colon).  Terminates on `FINAL_ANSWER` or when `max_iterations` is exhausted.

- `ToolRegistry` with support for synchronous and asynchronous tool handlers, per-tool
  required-field validation returning structured error observations (`ok`, `error`, `kind`
  fields), and optional per-tool `CircuitBreaker` integration.

- `ToolSpec` with `new` (sync closure) and `new_async` (async closure) constructors,
  `with_required_fields`, and `with_circuit_breaker` builder methods.

- `AgentConfig` with `max_iterations`, `model`, `system_prompt`, `max_memory_recalls`, and
  `max_memory_tokens` (approximate 4-chars-per-token budget for injected episodic
  memories).

- `EpisodicStore` for event-based memories with `DecayPolicy::exponential`,
  `RecallPolicy::Importance` and `RecallPolicy::Hybrid { recency_weight,
  frequency_weight }`, per-agent capacity limits with lowest-importance eviction, and
  `recall_count` tracking incremented on every retrieval.

- `SemanticStore` for fact-based knowledge with tag-based retrieval (intersection
  semantics) and cosine-similarity vector search.

- `WorkingMemory` — bounded LRU key-value store with deterministic oldest-entry eviction.

- `GraphStore` — directed in-memory knowledge graph with entity CRUD, relationship
  management, BFS, DFS, Dijkstra shortest-path, weighted shortest-path,
  transitive closure, degree centrality, betweenness centrality (Brandes algorithm),
  label-propagation community detection, subgraph extraction, and cycle detection.

- `CircuitBreaker` — configurable failure threshold and recovery window; states:
  `Closed` → `Open` → `HalfOpen` → `Closed`; pluggable `CircuitBreakerBackend` trait;
  `InMemoryCircuitBreakerBackend` default.

- `RetryPolicy::exponential(max_attempts, base_ms)` with delays capped at
  `MAX_RETRY_DELAY` (60 s).

- `Deduplicator` with TTL-based cache and in-flight request tracking.

- `BackpressureGuard` with hard capacity limit and optional soft limit with tracing
  warnings.

- `Pipeline` of composable named string-transform stages.

- `PersistenceBackend` async trait and `FilePersistenceBackend` with key sanitization
  preventing path-traversal escapes.

- `LlmProvider` async object-safe trait with `complete` and `stream_complete`.
  `AnthropicProvider` (Anthropic Messages API, SSE streaming) and `OpenAiProvider`
  (OpenAI Chat Completions, SSE streaming, custom base-URL support).

- `RuntimeMetrics` with atomic lock-free counters: `active_sessions`, `total_sessions`,
  `total_steps`, `total_tool_calls`, `failed_tool_calls`, `backpressure_shed_count`,
  `memory_recall_count`.

- `prelude` module re-exporting all commonly used types.

- Clippy deny rules: `unwrap_used`, `expect_used`, `panic`, `todo` in all production code.

- Comprehensive unit and integration test suite with concurrency stress tests.

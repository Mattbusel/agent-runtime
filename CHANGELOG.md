# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [1.8.0] - 2026-03-20

### Added

- **`EpisodicStore::recall_all`** (`memory.rs`) — retrieve every episode for an
  agent without a limit, useful for export and migration.
- **`SemanticStore::update_if_exists`** (`memory.rs`) — update a fact value only
  when the key is present; returns `Ok(false)` rather than inserting a new entry.
- **`MetricsSnapshot::to_json`** (`metrics.rs`) — serialize the snapshot to a
  `serde_json::Value` for structured log sinks and dashboards.
- **`RuntimeMetrics::checkpoint_errors`** (`metrics.rs`) — atomic counter for
  persistence checkpoint failures, exposed alongside the existing metric accessors.
- **`MetricsSnapshot::checkpoint_errors`** field and delta calculation
  (`metrics.rs`) — included in `delta()` comparisons and `Display` output.
- **`AgentConfig::stop_sequences`** field + **`AgentConfig::with_stop_sequences`**
  (`agent.rs`) — pass stop sequences through to the provider on every call.
- **`AnthropicProvider::with_stop_sequences`** /
  **`OpenAiProvider::with_stop_sequences`** (`providers.rs`) — configure provider-
  level stop sequences forwarded in the API request body.
- **`ToolRegistry::remove`** / **`ToolRegistry::contains`** (`agent.rs`) —
  unregister a tool by name and check registration status.
- **`ToolRegistry::unregister`** (`agent.rs`) — alias for `remove` that returns
  `bool` instead of `Option<ToolSpec>`.
- **`ReActLoop::unregister_tool`** (`agent.rs`) — convenience wrapper that
  delegates to the inner registry's `unregister`.

---

## [1.7.0] - 2026-03-20

### Added

- **`GraphStore::topological_sort`** (`graph.rs`) — iterative DFS post-order
  topological ordering; returns `Err(AgentRuntimeError::Graph)` on cycles.
- **`GraphStore::update_entity_property`** (`graph.rs`) — in-place mutation of a
  single property on an existing entity without a full replace.
- **`EpisodicStore::recall_by_id`** (`memory.rs`) — fetch a single episode by its
  `MemoryId` without a full recall scan.
- **`EpisodicStore::merge_from`** (`memory.rs`) — bulk-import episodes from
  another `EpisodicStore` (useful for merging agent memories).
- **`SemanticStore::update`** (`memory.rs`) — overwrite the value of an existing
  entry by key; returns `Ok(false)` when the key is not found.
- **`WorkingMemory::get_or_default`** (`memory.rs`) — return the stored value or
  a caller-supplied default without requiring an explicit `None` check.
- **`Message::user`** / **`Message::assistant`** (`agent.rs`) — named constructors
  that replace the verbose `Message::new(Role::User, …)` pattern.
- **`AgentConfig::with_model`** (`agent.rs`) — builder method to override the
  model after initial construction.
- **`AgentSession::elapsed`** (`runtime.rs`) — typed `std::time::Duration`
  accessor instead of raw `duration_ms` arithmetic.
- **`AgentSession::tool_calls_made`** (`runtime.rs`) — counts the number of
  tool-dispatching steps (excludes the final-answer step).
- **`LatencyHistogram::min_ms`** / **`LatencyHistogram::max_ms`** (`metrics.rs`) —
  basic histogram statistics without a full snapshot.
- **`MetricsSnapshot`** now implements `Display` (`metrics.rs`) — human-readable
  one-liner suitable for logs and dashboards.
- **`WorkingMemory::contains`** (`memory.rs`) — O(1) key existence check.
- **`SemanticStore::list_tags`** (`memory.rs`) — enumerate all unique tags across
  all stored facts.
- **`EpisodicStore::update_importance`** (`memory.rs`) — in-place importance
  mutation avoids a remove + re-insert cycle.
- **`GraphStore::neighbor_entities`** (`graph.rs`) — return full `Entity` objects
  for all out-neighbours of a node.
- **`GraphStore::remove_all_relationships_for`** (`graph.rs`) — bulk-remove all
  edges incident on a node before deletion.
- **`CircuitBreaker::reset`** (`orchestrator.rs`) — force the breaker back to
  `Closed` state; useful for test teardown and manual recovery.
- **`ToolRegistry::get`** (`agent.rs`) — look up a registered tool by name.
- **`InMemoryToolCache::contains`** / **`InMemoryToolCache::remove`** (`agent.rs`)
  — cache introspection and manual invalidation.
- **`WorkingMemory::remove`** / **`WorkingMemory::keys`** (`memory.rs`) — full
  CRUD: delete by key and enumerate all keys.
- **`SemanticStore::remove`** / **`SemanticStore::clear`** /
  **`SemanticStore::retrieve_by_key`** (`memory.rs`) — complete remove and
  key-based lookup.
- **`EpisodicStore::recall_tagged`** (`memory.rs`) — retrieve episodes filtered
  by a tag set without a full recall pass.
- **`GraphStore::get_relationships_for`** (`graph.rs`) — all outgoing edges from
  a node.
- **`GraphStore::relationships_between`** (`graph.rs`) — all edges from `A` to `B`.
- **`GraphStore::find_entities_by_property`** (`graph.rs`) — property-value scan
  across all entities.
- **`Deduplicator::purge_expired`** (`orchestrator.rs`) — explicit TTL eviction
  sweep; handy for long-lived deduplicators with infrequent traffic.
- **`LatencyHistogram::p50`** / **`p95`** / **`p99`** (`metrics.rs`) — named
  percentile shortcuts.
- **`MemoryItem::age_hours`** (`memory.rs`) — compute memory age in hours from its
  `timestamp_ms` field.
- **`EpisodicStore::recall_recent`** (`memory.rs`) — insertion-order recall,
  returning the most recently added episodes first.
- **`GraphStore::entity_count_by_label`** (`graph.rs`) — count entities matching
  a given label without a full scan.
- **`GraphStore::update_entity_label`** (`graph.rs`) — rename an entity's label
  in-place.
- **`Pipeline::stage_count`** / **`Pipeline::stage_names`** (`orchestrator.rs`)
  — introspect the number and names of registered pipeline stages.
- **`RetryPolicy::with_max_attempts`** (`orchestrator.rs`) — builder-style
  mutation of the attempt budget after construction.
- **`Deduplicator::clear`** (`orchestrator.rs`) — reset all in-flight and cached
  state, useful between test runs.
- **`BackpressureGuard::utilization_ratio`** (`orchestrator.rs`) — current depth
  divided by capacity as `f32` (0.0–1.0).
- **`ReActLoop::registry`** (`agent.rs`) — immutable read-only access to the
  registered tool registry.
- **`AgentSession::is_successful`** (`runtime.rs`) — returns `true` when the
  session ended with a `FINAL_ANSWER` step.

---

## [1.6.0] - 2026-03-20

### Changed

- **`Deduplicator::complete`** (`orchestrator.rs`) — eviction of the oldest cache
  entry under `max_entries` now uses the insertion-ordered `cache_order:
  VecDeque<String>` added to `DeduplicatorInner`.  The previous implementation
  scanned the entire cache with `Iterator::min_by_key` (O(N)) on every insert;
  `pop_front` is O(1) amortised.  Ghost entries (keys already expired by a prior
  `retain` call) are skipped with a loop so that a real eviction always occurs.
- **`GraphStore::subgraph`** (`graph.rs`) — entity-copy phase acquires
  `new_store`'s mutex **once** for the full batch instead of once per entity,
  mirroring the same optimisation previously applied to the relationship-copy phase.
- **`GraphStore::relationship_exists`** (`graph.rs`) — replaced
  `inner.relationships.iter().any(|r| r.from == *from && r.to == *to)` (O(|E|))
  with an `adjacency.get(from)` lookup (O(out-degree)), consistent with how all
  other traversal methods now use the adjacency index.
- **`AgentRuntime` memory-context builder** (`runtime.rs`) — eliminated the
  intermediate `Vec<String>` that collected formatted episode strings before
  joining; replaced with a single `write!` loop directly into a pre-allocated
  `String`, removing one allocation and the join pass per context-build.

---

## [1.5.0] - 2026-03-20

### Changed

- **`GraphStore::detect_cycles`** (`graph.rs`) — DFS inner loop no longer rebuilds
  the neighbour `Vec` via an O(|E|) relationship scan on every while-loop step.
  Now uses `inner.adjacency.get(node)` for O(1) lookup, reducing the per-cycle
  detection cost from O(V·E) to O(V + E).
- **`GraphStore::label_propagation_communities`** (`graph.rs`) — the `freq:
  HashMap<usize, usize>` used to find the most-frequent neighbour label is now
  allocated once before the outer iteration loop and cleared each step, removing
  `V * max_iterations` HashMap allocations.
- **`GraphStore::subgraph`** (`graph.rs`) — the relationship-copy phase now
  acquires `new_store`'s mutex **once** for all relationships rather than once per
  relationship, eliminating O(|E_subgraph|) redundant lock round-trips.
- **`AnthropicProvider::complete_with_options`** and
  **`OpenAiProvider::complete_with_options`** (`providers.rs`) — `options.timeout`
  is now applied to the reqwest request builder when set, replacing the previous
  behaviour where per-request timeouts were silently ignored.

---

## [1.4.0] - 2026-03-20

### Added

- **`Deduplicator::dedup_many`** (`orchestrator.rs`) — rewritten to acquire the
  internal mutex **once** for the entire batch instead of once per key.  The previous
  implementation called `self.check(key, ttl)` in a loop, taking and releasing the
  lock on every iteration.

### Changed

- **`GraphStore::betweenness_centrality`** (`graph.rs`) — two improvements in one:
  1. The inner BFS now uses `inner.adjacency.get(&v)` (O(degree)) instead of
     `inner.relationships.iter().filter(...)` (O(|E|)) per visited node.
  2. Work buffers (`stack`, `predecessors`, `sigma`, `dist`, `delta`, `queue`)
     are pre-allocated once and cleared at the top of each source-node iteration
     rather than being re-allocated from scratch, reducing total allocation cost
     from O(V²) to O(V).
- **`GraphStore::label_propagation_communities`** (`graph.rs`) — replaced the
  O(V·|E|) neighbour scan (`relationships.iter().filter(r.from==node||r.to==node)`)
  with a reverse-adjacency index built once before the iteration loop.  Each
  node now queries its out-neighbours via the existing adjacency index and its
  in-neighbours via the new reverse index, making each iteration O(V + E)
  instead of O(V · E).

---

## [1.3.0] - 2026-03-20

### Added

- **`OpenAiProvider::complete_with_options`** (`providers.rs`) — proper override that
  sends `max_tokens` and `temperature` in the request body when set, instead of
  silently falling back to the API default.  `complete` now delegates to
  `complete_with_options` for consistency with `AnthropicProvider`.

### Changed

- **`EntityId::new`** (`graph.rs`) — apply the same `if id.is_empty() { debug_assert!;
  tracing::warn! }` pattern already used by `AgentId::new` and `MemoryId::new`.
- **`GraphStore::weighted_shortest_path`** Dijkstra loop (`graph.rs`) — replaced
  `inner.relationships.iter().filter(|r| r.from == current)` with
  `inner.adjacency.get(&current)`.  The adjacency index was added in Round 1
  specifically to make BFS/DFS/shortest-path O(degree) per step; Dijkstra was the
  remaining method that still did an O(|E|) scan per visited node.
- **`GraphStore::degree_centrality`** (`graph.rs`) — eliminated the separate
  `out_degree: HashMap` by reading out-degree directly from the adjacency index
  (`adjacency.get(id).map_or(0, |v| v.len())`).  Saves one HashMap allocation and
  an O(V) initialization loop.
- **`SemanticStore::retrieve_similar`** (`memory.rs`) — same partial-sort
  optimization applied to `EpisodicStore::recall` in 1.2.0: uses
  `select_nth_unstable_by` (O(n)) + sort of only the top-`top_k` candidates
  (O(top_k log top_k)) rather than a full O(n log n) sort.

---

## [1.2.0] - 2026-03-20

### Added

- **`EpisodicStoreBuilder::try_per_agent_capacity`** (`memory.rs`) — non-panicking
  alternative to `per_agent_capacity`; returns `Err(AgentRuntimeError::Memory)` when
  `capacity == 0` instead of calling `assert!`.
- **`EpisodicStore::try_with_per_agent_capacity`** (`memory.rs`) — non-panicking
  alternative to `with_per_agent_capacity` for use in user-facing / library code.
- **`AnthropicProvider::with_stream_max_tokens`** (`providers.rs`) — builder method to
  override the hard-coded 1 024-token cap used by `stream_complete`.

### Changed

- **`MemoryId::new`** (`memory.rs`) — bare `debug_assert!` replaced with the same
  `if id.is_empty() { debug_assert!(false, ...); tracing::warn!(...) }` pattern already
  used by `AgentId::new`, so release builds emit a log warning instead of silently
  proceeding.
- **`EpisodicStore::recall`** (`memory.rs`) — replaced full O(n log n) sort with a
  two-phase partial sort: `select_nth_unstable_by` (O(n)) followed by a sort of only
  the top-`limit` elements (O(limit log limit)).  Reduces CPU work significantly when
  `limit << n`.
- **`ReActLoop` context formatting** (`agent.rs`) — replaced `context.push_str(&format!())`
  with `write!(context, ...)` to avoid allocating a temporary `String` on every step.
- **`AnthropicProvider::stream_complete`** (`providers.rs`) — `max_tokens` in the SSE
  request body now uses `self.stream_max_tokens.unwrap_or(Self::MAX_TOKENS)` instead of
  the hard-coded constant, honouring any value set via `with_stream_max_tokens`.

---

## [1.1.0] - 2026-03-20

### Added

- **`#[non_exhaustive]`** on `AgentRuntimeError` — new variants can be added in
  future minor releases without breaking external `match` arms.
- **`DebugBuilderState` sealed trait** (`runtime.rs`) — internal trait that drives
  a single generic `Debug` impl for `AgentRuntimeBuilder<S>`, replacing two
  near-identical manual implementations.

### Changed

- **`FilePersistenceBackend::save`** now writes data to a UUID-named temporary
  file in the same directory and atomically renames it into place.  Readers no
  longer observe half-written files if the process crashes mid-write.
- **`ToolSpec::with_required_fields`** now accepts
  `impl IntoIterator<Item = impl Into<String>>` instead of `Vec<String>`.
  Callers can now pass `&["field1", "field2"]` or any other iterable directly.
- **`ReActLoop::error_observation`** — silently-discarded `tool_name` parameter
  renamed to `_tool_name` so the intent is clear at the call site.
- **`AgentRuntimeBuilder::with_agent_config`** — removed duplicate doc-comment
  block that appeared twice verbatim.

### Deprecated

- **`AgentSession::load_step_checkpoint`** — use `load_checkpoint_at_step`
  instead (same semantics; the alias will be removed in 2.0).

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

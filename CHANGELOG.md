# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [1.51.0] - 2026-03-20

### Added

- **`AgentSession::has_final_answer`** — returns true if any step's action begins with `FINAL_ANSWER`.
- **`AgentSession::avg_action_length`** — mean byte length of all step action strings.
- **`EpisodicStore::episode_count_for`** — number of episodes stored for a specific agent.
- **`SemanticStore::unique_tags`** — sorted list of distinct tags across all entries.
- **`WorkingMemory::count_matching_prefix`** — count of keys starting with a given prefix.
- **`MetricsSnapshot::has_failures`** — true if any tool-call failures have been recorded.
- **`LatencyHistogram::max_occupied_ms`** — bound of the largest bucket with at least one sample.
- **`Pipeline::count_stages_matching`** — count of pipeline stages whose name contains a keyword.
- **`GraphStore::outgoing_count_for`** — number of outbound edges from a given entity.
- **`ToolRegistry::tool_names_with_keyword`** — names of tools whose description contains a keyword.

---

## [1.50.0] - 2026-03-20

### Added

- **`AgentSession::duration_secs`** — session duration in whole seconds.
- **`AgentSession::steps_above_thought_length`** — count of steps whose thought exceeds a byte threshold.
- **`EpisodicStore::max_importance_overall`** — highest importance value across all agents.
- **`SemanticStore::rename_tag`** — rename a tag across all entries, returns count updated.
- **`WorkingMemory::entry_count`** — number of key-value pairs currently stored.
- **`MetricsSnapshot::step_to_tool_ratio`** — ratio of tool calls to total steps.
- **`LatencyHistogram::min_occupied_ms`** — bound of the smallest bucket with at least one recorded value.
- **`RetryPolicy::backoff_factor`** — multiplier per retry (2.0 for exponential, 1.0 for constant).
- **`GraphStore::incoming_count_for`** — number of inbound edges for a given entity.
- **`ToolRegistry::unregister_all`** — remove all registered tools at once.

---

## [1.49.0] - 2026-03-20

### Added

- **`AgentSession::observation_lengths`** — byte lengths of all observations in step order.
- **`AgentSession::avg_observation_length`** — mean observation byte length.
- **`EpisodicStore::importance_variance_for`** — variance of importance scores for one agent.
- **`SemanticStore::count_matching_value`** — entries whose value contains a substring.
- **`WorkingMemory::keys_with_value_longer_than`** — keys with values exceeding a threshold.
- **`MetricsSnapshot::active_session_ratio`** — fraction of sessions currently active.
- **`LatencyHistogram::bucket_counts`** — raw count per bucket as an array.
- **`RetryPolicy::avg_delay_ms`** — average delay per attempt.
- **`GraphStore::labels`** — sorted list of distinct entity labels.
- **`ToolRegistry::count_with_description_containing`** — count of tools matching a keyword.

### Tests

- Full test coverage for all ten Round 30 methods.

---

## [1.48.0] - 2026-03-20

### Added

- **`AgentSession::steps_with_empty_observations`** — count of steps with no observation.
- **`AgentSession::min_thought_length`** — shortest non-empty thought byte length.
- **`EpisodicStore::agent_with_most_episodes`** — agent ID with the most stored episodes.
- **`SemanticStore::most_tagged_key`** — key with the most tags.
- **`WorkingMemory::value_lengths`** — list of (key, byte-length) pairs.
- **`MetricsSnapshot::memory_efficiency`** — memory recalls per step.
- **`LatencyHistogram::is_uniform`** — `true` when all samples are in one bucket.
- **`RetryPolicy::delay_sum_ms`** — cumulative delay for first N attempts.
- **`GraphStore::orphan_count`** — entities with no outgoing edges.
- **`ToolRegistry::description_for`** — description of a named tool.

### Tests

- Full test coverage for all ten Round 29 methods.

---

## [1.47.0] - 2026-03-20

### Added

- **`AgentSession::total_thought_length`** — total bytes across all thought strings.
- **`AgentSession::longest_observation`** — reference to the longest observation string.
- **`EpisodicStore::agents_with_min_episodes`** — agents with at least N episodes.
- **`SemanticStore::entries_with_no_tags`** — keys of untagged entries.
- **`WorkingMemory::longest_value_key`** — key whose value has the most bytes.
- **`MetricsSnapshot::backpressure_rate`** — average backpressure shed events per session.
- **`LatencyHistogram::percentile_spread`** — difference between p99 and p50.
- **`RetryPolicy::is_last_attempt`** — `true` when attempt number >= max_attempts.
- **`GraphStore::entity_type_count`** — number of distinct entity labels.
- **`ToolRegistry::tool_names_starting_with`** — sorted tool names matching a prefix.

### Tests

- Full test coverage for all ten Round 28 methods.

---

## [1.46.0] - 2026-03-20

### Added

- **`AgentSession::failure_rate`** — fraction of steps with tool-call failures.
- **`AgentSession::unique_action_count`** — number of distinct action names used
  across all steps.
- **`EpisodicStore::total_episode_count`** — total episodes across all agents.
- **`MetricsSnapshot::tool_call_rate`** — average tool calls per session.
- **`LatencyHistogram::sample_count`** — total samples recorded.
- **`RetryPolicy::max_total_delay_ms`** — sum of per-attempt delays across all
  `max_attempts`.
- **`GraphStore::has_any_entities`** — `true` when the graph is non-empty.
- **`ToolRegistry::names`** — sorted list of all registered tool names.

### Tests

- Full test coverage for all eight Round 27 methods.

---

## [1.45.0] - 2026-03-20

### Tests

- **`Entity::with_property`** — stores value and supports chaining.
- **`GraphStore::remove_relationship`** — removes edge successfully; errors
  when edge not found.
- **`GraphStore::update_entity_property`** — returns `true` and updates for
  existing entity; returns `false` for unknown entity.

6 new tests in `graph.rs`.

---

## [1.44.0] - 2026-03-20

### Added

- **`AgentSession::has_graph_lookups`** — returns `true` when the session
  performed at least one graph lookup.
- **`AgentSession::consecutive_same_action_at_end`** — counts how many trailing
  steps repeat the last action (useful for loop-detection).
- **`EpisodicStore::has_episodes`** — returns `true` if an agent has any stored
  episodes.
- **`SemanticStore::value_for`** — retrieves the stored value string for a key,
  returning `None` when the key is absent.
- **`WorkingMemory::count_above_value_length`** — counts entries whose value
  exceeds a given byte length threshold.
- **`MetricsSnapshot::has_errors`** — quick predicate: `true` when
  `failed_tool_calls > 0` or `checkpoint_errors > 0`.
- **`LatencyHistogram::is_above_p99`** — returns `true` when a given latency
  exceeds the p99 bucket.
- **`RetryPolicy::first_delay_ms`** — convenience alias for `base_delay_ms()`.
- **`GraphStore::has_entity`** — returns `true` when an entity ID exists in the
  graph.
- **`ToolRegistry::tool_count_with_required_fields`** — counts tools that
  declare at least one required field.

### Tests

- Full test coverage for all ten Round 26 methods across `runtime.rs`,
  `memory.rs`, `metrics.rs`, `orchestrator.rs`, `graph.rs`, and `agent.rs`.

---

## [1.43.0] - 2026-03-20

### Tests

- **`AgentSession::observation_at`** — correct value at index; `None` when
  out of bounds.
- **`AgentSession::action_at`** — correct value at index; `None` when out of
  bounds.
- **`Message::is_user` / `is_assistant`** — role-based predicate checks.
- **`AgentConfig::stop_sequence_count`** — zero by default; reflects
  configured count after `with_stop_sequences`.
- **`AgentConfig::has_request_timeout`** — false by default; true after
  `with_request_timeout`.
- **`Deduplicator::fail`** — removes in-flight key; no-op on unknown key;
  allows re-registration after failure.
- **`LatencyHistogram::percentile`** — returns 0 when empty; matches `p50`
  and `p99` at corresponding percentile values.
- **`RuntimeMetrics::record_agent_tool_failure`** — failure count appears in
  per-agent snapshot.
- **`RuntimeMetrics::per_agent_tool_calls_snapshot`** — empty initially.
- **`RuntimeMetrics::record_step_latency`** — operation succeeds without panic.

27 new tests across `runtime.rs`, `agent.rs`, `orchestrator.rs`, and
`metrics.rs`.

---

## [1.42.0] - 2026-03-20

### Added

- **`AgentSession::last_n_observations`** (`runtime.rs`) — last `n` non-empty
  observation strings (oldest to newest); skips empty observations.
- **`AgentSession::actions_in_window`** (`runtime.rs`) — action names from the
  last `n` steps; returns all steps when fewer than `n` exist.
- **`AgentConfig::stop_sequence_count`** (`agent.rs`) — number of configured
  stop sequences.
- **`GraphStore::avg_out_degree`** (`graph.rs`) — mean outbound edges per
  entity; `0.0` for empty graphs.
- **`GraphStore::avg_in_degree`** (`graph.rs`) — mean inbound edges per entity;
  `0.0` for empty graphs.
- **`RetryPolicy::will_retry_at_all`** (`orchestrator.rs`) — `true` when
  `max_attempts > 1`; complement of `is_no_retry`.
- **`Deduplicator::total_count`** (`orchestrator.rs`) — total items tracked
  (in-flight + cached), regardless of TTL expiry.
- **`BackpressureGuard::acquired_count`** (`orchestrator.rs`) — number of
  currently held slots (`capacity - available_capacity()`).
- **`Pipeline::swap_stages`** (`orchestrator.rs`) — swap positions of two named
  stages by name; returns `false` if either is not found.
- **`ToolRegistry::find_by_description_keyword`** (`agent.rs`) — return all
  tool specs whose description contains a keyword (case-insensitive).

### Tests

- 33 new unit tests covering all ten methods listed above.

---

## [1.41.0] - 2026-03-20

### Added

- **`AgentSession::action_repetition_rate`** (`runtime.rs`) — fraction of step
  transitions that repeat the immediately preceding action; useful for loop
  detection.  Returns `0.0` for sessions with fewer than two steps.
- **`AgentSession::max_consecutive_failures`** (`runtime.rs`) — length of the
  longest unbroken run of failed steps (observation contains `"error"`).
- **`AgentSession::avg_thought_length`** (`runtime.rs`) — mean character length
  of non-empty thought strings; `0.0` when no thoughts exist.
- **`EpisodicStore::avg_importance`** (`memory.rs`) — mean importance score
  across all episodes for an agent; `0.0` when the agent has no episodes.
- **`EpisodicStore::importance_range`** (`memory.rs`) — `(min, max)` importance
  tuple; `None` when the agent has no stored episodes.
- **`SemanticStore::entries_without_tags`** (`memory.rs`) — count of entries
  that have an empty tag list.
- **`SemanticStore::avg_tag_count_per_entry`** (`memory.rs`) — mean number of
  tags per entry; `0.0` when the store is empty.
- **`WorkingMemory::longest_key`** (`memory.rs`) — the key with the most bytes;
  `None` when the store is empty.
- **`WorkingMemory::longest_value`** (`memory.rs`) — the value string with the
  most bytes; `None` when the store is empty.
- **`LatencyHistogram::coefficient_of_variation`** (`metrics.rs`) —
  `std_dev_ms / mean_ms`; `0.0` when `mean_ms` is zero.

### Tests

- 35 new unit tests covering all ten methods listed above.

---

## [1.40.0] - 2026-03-20

### Tests

- **`RetryPolicy::is_constant`** — verify constant vs exponential discrimination.
- **`RetryPolicy::total_max_delay_ms`** — verify sum of per-attempt delays.
- **`CircuitBreaker::is_healthy`** — healthy when Closed, unhealthy when Open.
- **`CircuitBreaker::is_half_open`** — enters HalfOpen after zero-duration
  recovery window elapses.
- **`Deduplicator::is_idle`** — idle when empty; not idle with in-flight
  request; idle again after complete.
- **`EpisodicStore::has_agent`** — false for unknown agent, true after episode.
- **`EpisodicStore::export_agent_memory`** — empty for unknown agent, correct
  count after episodes.
- **`SemanticStore::store_with_embedding`** — rejects empty vector; stores
  entry; rejects dimension mismatch on second call.
- **`GraphStore::reverse`** — flips edge direction; empty graph gives empty
  reverse.
- **`GraphStore::max_in_degree_entity`** — None for empty graph; returns node
  with highest in-degree.
- **`GraphStore::shortest_path_length`** — None when unreachable; 1 for direct
  edge; 2 for two-hop path.

28 new tests across `orchestrator.rs`, `memory.rs`, and `graph.rs`.

---

## [1.39.0] - 2026-03-20

### Added

- **`AgentSession::last_observation`** (`runtime.rs`) — the observation string
  from the most recent step that has a non-empty observation; `None` when no
  step has produced one.
- **`AgentSession::thought_count`** (`runtime.rs`) — number of steps with a
  non-empty thought string.
- **`AgentSession::observation_rate`** (`runtime.rs`) — fraction of steps that
  contain a non-empty observation; `0.0` for empty sessions.
- **`LatencyHistogram::median_ms`** (`metrics.rs`) — convenience alias for
  `p50()` that communicates intent more clearly at call sites.
- **`MetricsSnapshot::steps_per_session`** (`metrics.rs`) — average ReAct
  steps per session; `0.0` when no sessions have been recorded.
- **`RuntimeMetrics::p50_latency_ms`** (`metrics.rs`) — shorthand for
  `step_latency.p50()`; returns `0` when no latencies have been recorded.
- **`EpisodicStore::latest_episode`** (`memory.rs`) — the most recently
  inserted episode for an agent (by timestamp); `None` when no episodes exist.
- **`SemanticStore::oldest_key`** (`memory.rs`) — key of the earliest inserted
  semantic entry; `None` when the store is empty.
- **`WorkingMemory::key_count_matching`** (`memory.rs`) — count of keys whose
  text contains a given case-sensitive substring.
- **`WorkingMemory::avg_value_length`** (`memory.rs`) — mean byte length of all
  stored values; `0.0` when the store is empty.
- **`GraphStore::isolated_nodes`** (`graph.rs`) — set of entity IDs that have
  no inbound and no outbound edges.

### Tests

- 37 new unit tests covering all eleven methods listed above.

---

## [1.38.0] - 2026-03-20

### Added

- **`LatencyHistogram::is_below_p99`** (`metrics.rs`) — SLO helper; returns
  `true` if the p99 latency is strictly below the given threshold.  Returns
  `true` when no samples have been recorded.
- **`MetricsSnapshot::is_healthy`** (`metrics.rs`) — returns `true` when all
  of `failed_tool_calls`, `backpressure_shed_count`, and `checkpoint_errors`
  are zero.
- **`Deduplicator::cached_keys`** (`orchestrator.rs`) — snapshot of all keys
  whose results have been stored in the cache (includes expired entries until
  purged).
- **`WorkingMemory::set_if_absent`** (`memory.rs`) — conditional insert;
  returns `Ok(true)` when the key was new, `Ok(false)` when the key already
  existed (existing value unchanged).
- **`AgentRuntime::has_memory`** (`runtime.rs`, `#[cfg(feature = "memory")]`)
  — `true` when episodic memory was configured on this runtime.
- **`AgentRuntime::has_graph`** (`runtime.rs`, `#[cfg(feature = "graph")]`)
  — `true` when a graph store was configured on this runtime.
- **`AgentRuntime::has_working_memory`** (`runtime.rs`,
  `#[cfg(feature = "memory")]`) — `true` when working memory was configured
  on this runtime.

### Tests

- 20 new tests across `metrics.rs`, `orchestrator.rs`, `memory.rs`, and
  `runtime.rs` covering all seven methods above.

---

## [1.37.0] - 2026-03-20

### Added

- **`AgentSession::most_used_action`** (`runtime.rs`) — returns the action name
  that appears most frequently across all ReAct steps; `None` when no steps
  have been recorded.
- **`AgentSession::graph_lookup_rate`** (`runtime.rs`) — graph lookups per
  ReAct step; `0.0` when the session has no steps.
- **`AgentConfig::has_max_context_chars`** (`agent.rs`) — `true` when a maximum
  context character limit has been configured.
- **`AgentConfig::max_context_chars`** (`agent.rs`) — getter for the optional
  maximum context character limit.
- **`GraphStore::weight_stats`** (`graph.rs`) — returns `(min, max, mean)`
  relationship weights; `None` when the graph has no edges.
- **`EpisodicStore::max_recall_count_for`** (`memory.rs`) — highest
  `recall_count` across all episodes for an agent; `None` when the agent has
  no episodes.
- **`SemanticStore::most_recent_key`** (`memory.rs`) — the insertion key of the
  last stored semantic entry; `None` when the store is empty.
- **`WorkingMemory::max_key_length`** (`memory.rs`) — byte length of the
  longest key currently stored; `0` when the store is empty.
- **`LatencyHistogram::p10`** (`metrics.rs`) — 10th-percentile latency in
  milliseconds; useful for assessing the fast tail of the distribution.
- **`CircuitBreaker::failures_until_open`** (`orchestrator.rs`) — number of
  additional failures needed before the circuit opens; `0` when already at or
  beyond the threshold.

### Tests

- 32 new unit tests covering all ten methods listed above.

---

## [1.36.0] - 2026-03-20

### Added

- **`AgentSession::throughput_steps_per_sec`** (`runtime.rs`) — ReAct steps per
  second; `0.0` when `duration_ms` is zero.
- **`AgentConfig::temperature`** (`agent.rs`) — getter for the optional sampling
  temperature field.
- **`AgentConfig::max_tokens`** (`agent.rs`) — getter for the optional
  max-output-tokens field.
- **`AgentConfig::request_timeout`** (`agent.rs`) — getter for the optional
  per-inference request timeout.
- **`MetricsSnapshot::error_rate`** (`metrics.rs`) — `failed_tool_calls /
  total_tool_calls`; `0.0` when no calls recorded.
- **`MetricsSnapshot::memory_recall_rate`** (`metrics.rs`) — memory recalls per
  completed session; `0.0` when no sessions recorded.
- **`CircuitBreaker::is_at_threshold`** (`orchestrator.rs`) — `true` when the
  current failure count has reached the configured threshold.
- **`BackpressureGuard::headroom_ratio`** (`orchestrator.rs`) — fraction of
  capacity still available; `1.0` when empty, `0.0` when full.
- **`GraphStore::sum_edge_weights`** (`graph.rs`) — sum of all relationship
  weights; `0.0` for graphs with no relationships.
- **`EpisodicStore::sum_recall_counts`** (`memory.rs`) — total `recall_count`
  across all episodes for an agent; `0` when the agent has no episodes.

---

## [1.35.0] - 2026-03-20

### Added

- **`Entity::property_value(key)`** (`graph.rs`) — returns `Option<&Value>` for a
  property key; wraps `HashMap::get` for ergonomic property access.
- **`GraphStore::find_relationships_by_kind(kind)`** (`graph.rs`) — filter all
  stored relationships by `kind` (case-sensitive).
- **`GraphStore::count_relationships_by_kind(kind)`** (`graph.rs`) — count of
  relationships matching `kind` without allocating a full `Vec`.
- **`LatencyHistogram::interquartile_range_ms()`** (`metrics.rs`) — p75 − p25
  spread; less sensitive to outliers than `range_ms`.
- **`MetricsSnapshot::avg_steps_per_session()`** (`metrics.rs`) — mean ReAct steps
  per completed session; `0.0` when no sessions have been recorded.
- **`WorkingMemory::peek_oldest()`** (`memory.rs`) — inspect the oldest entry
  without removing it; read-only counterpart to `pop_oldest`.
- **`SemanticStore::values()`** (`memory.rs`) — all stored values in insertion
  order.
- **`SemanticStore::get_tags(key)`** (`memory.rs`) — tags for the first entry
  matching `key`, or `None` if absent.
- **`Deduplicator::get_result(key)`** (`orchestrator.rs`) — look up the cached
  result for `key` without side effects.
- **`Pipeline::rename_stage(old, new)`** (`orchestrator.rs`) — rename an existing
  pipeline stage in place.
- **`CircuitBreaker::failure_rate()`** (`orchestrator.rs`) — `failure_count /
  threshold` ratio for alerting and dashboards.
- **`RunResult::failed_tool_call_count()`** (`runtime.rs`) — count of steps
  with error observations; avoids a `Vec` allocation vs `failed_steps().len()`.
- **`AgentSession::throughput_steps_per_sec()`** (`runtime.rs`) — steps per
  second computed as `step_count / (duration_ms / 1000.0)`; `0.0` for zero
  duration sessions.
- **`AgentConfig::temperature()`** / **`max_tokens()`** / **`request_timeout()`**
  (`agent.rs`) — direct getters for the optional config fields that previously
  required field access.
- **`GraphStore::sum_edge_weights()`** (`graph.rs`) — sum of all relationship
  weights as `f64` for cost/flow computations.
- **`EpisodicStore::sum_recall_counts(agent_id)`** (`memory.rs`) — sum of
  `recall_count` across all episodes for an agent.
- **`MetricsSnapshot::error_rate()`** (`metrics.rs`) — `failed_tool_calls /
  total_tool_calls`; `0.0` when no calls recorded.
- **`MetricsSnapshot::memory_recall_rate()`** (`metrics.rs`) — `memory_recall_count
  / total_steps`; `0.0` when no steps recorded.
- **`BackpressureGuard::is_at_threshold()`** (`orchestrator.rs`) — `true` when
  depth is at or above the soft limit.
- **`BackpressureGuard::headroom_ratio()`** (`orchestrator.rs`) — remaining
  capacity as a fraction of total capacity in `[0.0, 1.0]`.

---

## [1.34.0] - 2026-03-20

### Added

- **`AgentSession::observation_count`** (`runtime.rs`) — count of steps with a
  non-empty observation string.
- **`AgentSession::steps_without_observation`** (`runtime.rs`) — count of steps
  whose observation is empty; useful for detecting silent tool failures.
- **`ReActStep::observation_is_empty`** (`agent.rs`) — `true` if the observation
  field is an empty string.
- **`GraphStore::max_out_degree`** (`graph.rs`) — maximum out-degree value across
  all entities; `0` for empty graphs.
- **`GraphStore::max_in_degree`** (`graph.rs`) — maximum in-degree value across
  all entities; `0` for empty graphs.
- **`EpisodicStore::oldest_episode`** (`memory.rs`) — episode with the earliest
  timestamp for an agent; `None` if the agent has no episodes.
- **`SemanticStore::remove_by_key`** (`memory.rs`) — remove all entries whose
  key matches exactly; returns the number of entries removed.
- **`WorkingMemory::total_value_bytes`** (`memory.rs`) — total byte count of all
  stored values; useful for memory-pressure estimation.
- **`Deduplicator::evict_oldest`** (`orchestrator.rs`) — remove the oldest cached
  result entry (FIFO); returns `false` if the cache was empty.
- **`LatencyHistogram::mode_bucket_ms`** (`metrics.rs`) — upper bound of the
  bucket with the highest sample count; `None` if no samples recorded.

---

## [1.32.0] - 2026-03-20

### Added

- **`AgentSession::total_memory_hits`** (`runtime.rs`) — raw count of episodic
  memory hits for the session; direct getter for the `memory_hits` field.
- **`AgentSession::action_diversity`** (`runtime.rs`) — ratio of unique actions
  to total steps; `0.0` for empty sessions, `1.0` when every step uses a
  distinct action.
- **`GraphStore::is_acyclic`** (`graph.rs`) — shorthand for `!contains_cycle()`;
  returns `true` for empty graphs.
- **`EpisodicStore::agent_count`** (`memory.rs`) — number of distinct agents
  that have at least one stored episode.
- **`WorkingMemory::is_at_capacity`** (`memory.rs`) — `true` when the number of
  stored entries equals the configured capacity.
- **`WorkingMemory::remove_keys_starting_with`** (`memory.rs`) — batch-remove all
  entries whose key begins with a given prefix; returns the removed count.
- **`SemanticStore::has_key`** (`memory.rs`) — `true` if any entry with the given
  key exists.
- **`SemanticStore::entry_count_with_tag`** (`memory.rs`) — count of entries that
  include a specific tag in their tag list.
- **`LatencyHistogram::is_empty`** (`metrics.rs`) — `true` when no samples have
  been recorded yet.
- **`RuntimeMetrics::checkpoint_error_rate`** (`metrics.rs`) — ratio of checkpoint
  errors to total completed sessions; `0.0` when no sessions recorded.

---

## [1.31.0] - 2026-03-20

### Added

- **`Entity::property_value(key)`** (`graph.rs`) — returns `Option<&Value>` for a
  property key; wraps `HashMap::get` for ergonomic typed property access.
- **`GraphStore::find_relationships_by_kind(kind)`** (`graph.rs`) — filter all
  stored relationships by `kind` (case-sensitive), returning a `Vec<Relationship>`.
- **`GraphStore::count_relationships_by_kind(kind)`** (`graph.rs`) — count of
  relationships matching `kind` without allocating a full `Vec`.
- **`LatencyHistogram::interquartile_range_ms()`** (`metrics.rs`) — p75 − p25 as a
  measure of spread less sensitive to outliers than `range_ms`.
- **`MetricsSnapshot::avg_steps_per_session()`** (`metrics.rs`) — `total_steps /
  total_sessions` as `f64`; returns `0.0` when no sessions recorded.
- **`WorkingMemory::peek_oldest()`** (`memory.rs`) — read the oldest entry without
  removing it; read-only counterpart to `pop_oldest`.
- **`SemanticStore::values()`** (`memory.rs`) — all stored values in insertion
  order as `Vec<String>`.
- **`SemanticStore::get_tags(key)`** (`memory.rs`) — tags for the first entry
  matching `key`, or `None` if absent.
- **`Deduplicator::get_result(key)`** (`orchestrator.rs`) — look up the cached
  result for `key` if present and not expired; read-only, no state changes.
- **`Pipeline::rename_stage(old, new)`** (`orchestrator.rs`) — rename an existing
  stage in place; returns `false` if no stage with `old` name exists.
- **`CircuitBreaker::failure_rate()`** (`orchestrator.rs`) — `failure_count /
  threshold` as `f64`; useful for dashboards and alerting thresholds.
- **`RunResult::failed_tool_call_count()`** (`runtime.rs`) — count of steps with
  error observations; equivalent to `failed_steps().len()` without the allocation.

---

## [1.30.0] - 2026-03-20

### Added

- **`AgentSession::last_n_steps`** (`runtime.rs`) — slice of the last `n` steps;
  returns all steps if `n` exceeds step count, empty slice for empty sessions.
- **`GraphStore::reachable_from`** (`graph.rs`) — BFS traversal returning a
  `HashSet` of all entity IDs reachable from a start node (start node excluded).
- **`GraphStore::contains_cycle`** (`graph.rs`) — iterative DFS with three-colour
  scheme to detect directed cycles; returns `false` for empty graphs.
- **`EpisodicStore::count_above_importance`** (`memory.rs`) — count of episodes
  whose importance is strictly greater than a threshold; `0` for unknown agents.
- **`WorkingMemory::value_length`** (`memory.rs`) — byte length of the stored
  value for a key, or `None` when the key is absent.
- **`SemanticStore::tags_for`** (`memory.rs`) — tags list for the first entry
  matching a key; `None` when no entry with that key exists.
- **`Pipeline::prepend_stage`** (`orchestrator.rs`) — insert a stage at the
  front of the pipeline; all existing stages shift to higher indices.
- **`RuntimeMetrics::avg_steps_per_session`** (`metrics.rs`) — mean ReAct steps
  per completed session; `0.0` when no sessions have been recorded.

### Fixed

- Pre-existing compilation errors in linter-generated tests:
  - Added missing `make_step` helper in `runtime.rs` test module.
  - Replaced `LatencyHistogram::new()` with `LatencyHistogram::default()` in
    `metrics.rs` tests.

---

## [1.29.0] - 2026-03-20

### Added

- **`AgentSession::first_action`** (`runtime.rs`) — action name from the first
  step; `None` for empty sessions.
- **`AgentSession::last_action`** (`runtime.rs`) — action name from the last
  step; useful for detecting `FINAL_ANSWER` without scanning all steps.
- **`GraphStore::successors`** (`graph.rs`) — entities directly reachable via
  outgoing edges (immediate out-neighbors).
- **`GraphStore::is_sink`** (`graph.rs`) — `true` when an entity has no
  outgoing edges; identifies DAG leaf nodes.
- **`EpisodicStore::max_importance`** (`memory.rs`) — highest importance score
  stored for an agent; `None` if no episodes exist.
- **`EpisodicStore::min_importance`** (`memory.rs`) — lowest importance score;
  symmetric counterpart to `max_importance`.
- **`WorkingMemory::values_matching`** (`memory.rs`) — return all `(key, value)`
  pairs whose value contains a given substring.
- **`RuntimeMetrics::avg_tool_calls_per_session`** (`metrics.rs`) — mean tool
  calls per completed session; `0.0` when no sessions recorded.

---

## [1.28.0] - 2026-03-20

### Added

- **`ToolSpec::required_field_count`** (`agent.rs`) — number of required fields
  configured; complements `has_required_fields`.
- **`ToolSpec::has_required_fields`** (`agent.rs`) — `true` when at least one
  required field is configured.
- **`ToolSpec::has_validators`** (`agent.rs`) — `true` when at least one custom
  argument validator is attached.
- **`Relationship::is_self_loop`** (`graph.rs`) — `true` when source and target
  entity IDs are equal.
- **`Relationship::reversed`** (`graph.rs`) — return a new `Relationship` with
  `from` and `to` swapped, preserving `kind` and `weight`.
- **`GraphStore::find_entities_by_labels`** (`graph.rs`) — return entities whose
  label is any of the given strings; multi-label generalisation of
  `find_entities_by_label`.
- **`GraphStore::remove_isolated`** (`graph.rs`) — remove all entities with no
  incoming or outgoing edges and return the count; useful for graph compaction.
- **`Pipeline::first_stage_name`** / **`last_stage_name`** (`orchestrator.rs`) —
  return the name of the first/last stage without constructing a full `stage_names()`
  list.
- **`Deduplicator::contains`** (`orchestrator.rs`) — read-only check whether a
  key is currently in-flight or cached; does not register the key.
- **`EpisodicStore::most_recalled`** (`memory.rs`) — return the episode with the
  highest `recall_count` for an agent.
- **`AgentConfig::remaining_iterations_after`** (`agent.rs`) — iterations
  remaining after `n` have been completed; uses saturating subtraction.

### Fixed

- Test assertion in `test_remove_isolated_removes_nodes_without_edges` — changed
  incorrect `.unwrap().is_none()` / `.unwrap().is_some()` calls on `Entity` to
  `.is_err()` / `.is_ok()` on the `Result` returned by `get_entity`.

---

## [1.27.0] - 2026-03-20

### Added

- **`AgentSession::first_thought`** (`runtime.rs`) — thought string from the
  first step; `None` for empty sessions.
- **`AgentSession::last_thought`** (`runtime.rs`) — thought string from the
  last step; complements `first_thought`.
- **`GraphStore::out_degree_for`** (`graph.rs`) — outgoing edge count for a
  single entity; symmetric counterpart to `in_degree_for`.
- **`GraphStore::predecessors`** (`graph.rs`) — return all entities with a
  directed edge pointing to the given node (incoming neighbors).
- **`GraphStore::is_source`** (`graph.rs`) — `true` when the entity has no
  incoming edges; identifies DAG root nodes.
- **`BackpressureGuard::percent_full`** (`orchestrator.rs`) — current depth as
  a percentage of hard capacity; clamped to `[0.0, 100.0]`.
- **`RetryPolicy::base_delay_ms`** (`orchestrator.rs`) — getter for the
  configured base delay in milliseconds; complements `with_base_delay_ms`.
- **`LatencyHistogram::range_ms`** (`metrics.rs`) — max − min latency spread;
  `None` when no samples recorded.
- **`ToolRegistry::all_tool_names`** (`agent.rs`) — sorted alphabetical list of
  all registered tool names; deterministic for diagnostics and help text.

---

## [1.26.0] - 2026-03-20

### Added

- **`EntityId::is_empty`** / **`EntityId::starts_with`** (`graph.rs`) — predicate
  helpers consistent with `AgentId` and `MemoryId`; `starts_with` is useful for
  namespace-based entity lookup.
- **`Entity::has_property`** (`graph.rs`) — `true` when the entity's property map
  contains the given key; avoids accessing the raw `HashMap` at call sites.
- **`GraphStore::entity_labels`** (`graph.rs`) — return all distinct entity label
  strings present in the graph, sorted; complements `relationship_kinds`.
- **`MemoryItem::has_tag`** (`memory.rs`) — `true` when the memory item's tag
  list contains the given tag string.
- **`MemoryItem::word_count`** (`memory.rs`) — approximate whitespace-separated
  word count of the memory content.
- **`EpisodicStore::has_agent`** (`memory.rs`) — `true` when the store contains
  at least one episode for the given agent; cheaper than `count_for(id)? > 0`.
- **`AgentConfig::has_temperature`** (`agent.rs`) — `true` when a sampling
  temperature has been configured.
- **`AgentConfig::has_request_timeout`** (`agent.rs`) — `true` when a
  per-inference timeout has been configured.
- **`RuntimeMetrics::step_latency_p50`** / **`step_latency_p99`** (`metrics.rs`) —
  expose median and 99th-percentile step latency directly without accessing the
  histogram field.
- **`AgentSession::checkpoint_error_count`** (`runtime.rs`) — count of
  checkpoint errors recorded during the session; companion to
  `has_checkpoint_errors`.

---

## [1.25.0] - 2026-03-20

### Added

- **`AgentSession::avg_step_duration_ms`** (`runtime.rs`) — arithmetic mean of
  all step durations; `0.0` for sessions with no steps.
- **`AgentSession::longest_step`** (`runtime.rs`) — step reference with the
  largest `step_duration_ms`; useful for hotspot analysis.
- **`AgentSession::shortest_step`** (`runtime.rs`) — step reference with the
  smallest `step_duration_ms`; complements `longest_step`.
- **`EpisodicStore::most_recent`** (`memory.rs`) — return the most recently
  stored episode for an agent (last-inserted); `None` if none exist.
- **`WorkingMemory::contains_all`** (`memory.rs`) — `true` when every key in
  the iterator is present; vacuously `true` for an empty iterator.
- **`WorkingMemory::has_any_key`** (`memory.rs`) — `true` when at least one key
  in the iterator is present; `false` for an empty iterator.
- **`BackpressureGuard::is_soft_limited`** (`orchestrator.rs`) — `true` when a
  soft capacity has been configured; more readable than `soft_limit().is_some()`.
- **`RuntimeMetrics::tool_success_rate`** (`metrics.rs`) — fraction of tool
  calls that did not fail; `1.0` when no calls have been recorded.

---

## [1.24.0] - 2026-03-20

### Added

- **`FilePersistenceBackend::key_count`** (`persistence.rs`) — count stored keys
  without building a string list; more efficient than `list_keys().await?.len()`.
- **`AgentId::starts_with`** / **`MemoryId::starts_with`** (`types.rs`) — check
  whether an ID string begins with a given prefix; useful for namespace-based
  routing and ID classification.

---

## [1.23.0] - 2026-03-20

### Added

- **`AgentSession::step_count_for_action`** (`runtime.rs`) — count how many
  steps used a specific action name; complements `has_action` with a frequency.
- **`AgentSession::observations`** (`runtime.rs`) — collect all observation
  strings in step order; avoids manual iteration over `steps`.
- **`EpisodicStore::retain_top_n`** (`memory.rs`) — keep only the `n`
  most-important episodes per agent, evicting the rest; returns removed count.
- **`WorkingMemory::keys_starting_with`** (`memory.rs`) — return keys matching
  a given prefix, enabling namespace-based key grouping.
- **`GraphStore::average_out_degree`** (`graph.rs`) — mean out-degree across
  all entities; useful for detecting dense vs. sparse graph regions.
- **`GraphStore::in_degree_for`** (`graph.rs`) — incoming edge count for a
  single entity; complements `out_degree_for` for hub-detection.
- **`RetryPolicy::is_exponential`** (`orchestrator.rs`) — `true` when the
  policy uses exponential back-off; companion to `is_no_retry`.
- **`AgentConfig::max_iterations`** (`agent.rs`) — getter for the configured
  maximum iteration count; avoids accessing the public field directly.
- **`LatencyHistogram::std_dev_ms`** (`metrics.rs`) — bucket-midpoint
  approximation of the standard deviation; completes the descriptive stats suite.

---

## [1.22.0] - 2026-03-20

### Added

- **`AgentConfig::has_loop_timeout`** (`agent.rs`) — `true` when a loop timeout
  has been set; avoids inspecting `loop_timeout.is_some()` at call sites.
- **`AgentConfig::has_stop_sequences`** (`agent.rs`) — `true` when at least one
  stop sequence is configured.
- **`AgentConfig::is_single_shot`** (`agent.rs`) — `true` when `max_iterations == 1`;
  flags one-shot agents without inspecting the raw field.
- **`ReActStep::with_duration`** (`agent.rs`) — builder method to set
  `step_duration_ms`; useful for constructing test sessions with realistic
  timings without running the full loop.
- **`ReActStep::is_empty`** (`agent.rs`) — `true` when all three text fields
  (thought, action, observation) are empty strings.
- **`Message::is_empty`** (`agent.rs`) — `true` when the message content is
  an empty string.
- **`Message::word_count`** (`agent.rs`) — approximate whitespace-separated
  word count of the message content.
- **`ToolRegistry::descriptions`** (`agent.rs`) — return `(name, description)`
  pairs for all registered tools, sorted by name; useful for generating help
  text at startup.
- **`InMemoryToolCache::capacity`** (`agent.rs`) — return the configured maximum
  entry limit, or `None` if the cache is unbounded.
- **`RetryPolicy::is_constant`** (`orchestrator.rs`) — `true` when the policy
  uses a fixed-interval delay; companion to `is_exponential`.
- **`AgentId::starts_with`** / **`MemoryId::starts_with`** (`types.rs`) — check
  whether an ID string begins with a given prefix; useful for namespace-based
  routing.
- **`FilePersistenceBackend::key_count`** (`persistence.rs`) — count stored keys
  without building a string list; more efficient than `list_keys().len()`.
- **`MetricsSnapshot::total_successful_tool_calls`** (`metrics.rs`) — total tool
  calls minus failed calls using saturating arithmetic.

---

## [1.21.0] - 2026-03-20

### Added

- **`AgentSession::has_action`** (`runtime.rs`) — `true` if any step used the
  named action; replaces manual iteration over `action_sequence()`.
- **`AgentSession::thought_at`** (`runtime.rs`) — return the thought string at a
  given step index, or `None` if out of bounds.
- **`EpisodicStore::recall_top_n`** (`memory.rs`) — retrieve the `n`
  highest-importance episodes for an agent in descending order.
- **`EpisodicStore::filter_by_importance`** (`memory.rs`) — return all episodes
  whose importance falls within an inclusive `[min, max]` range.
- **`WorkingMemory::update_many`** (`memory.rs`) — batch-set multiple key/value
  pairs in one call; returns the count of updated entries.
- **`SemanticStore::entry_count_with_embedding`** (`memory.rs`) — count entries
  that have a non-empty embedding vector stored.
- **`GraphStore::weakly_connected`** (`graph.rs`) — `true` when the graph has at
  most one weakly-connected component (treats edges as undirected).
- **`RetryPolicy::is_no_retry`** (`orchestrator.rs`) — `true` when
  `max_attempts ≤ 1`; allows hot-path short-circuiting without inspecting the
  full policy.
- **`ToolRegistry::filter_tools`** (`agent.rs`) — return references to all
  registered `ToolSpec`s matching a caller-supplied predicate.

---

## [1.20.0] - 2026-03-20

### Added

- **`LatencyHistogram::p25`** (`metrics.rs`) — 25th-percentile latency; completes
  the p25/p50/p75/p90/p95/p99 percentile suite.
- **`EpisodicStore::agents`** (`memory.rs`) — return all agent IDs that have at
  least one stored episode, sorted; useful for per-agent bulk operations.
- **`SemanticStore::to_map`** (`memory.rs`) — return all entries as
  `HashMap<key, value>`; convenient for serialization or bulk inspection.
- **`Pipeline::has_stage`** (`orchestrator.rs`) — `true` if any stage has the
  given name; cheaper than iterating `stage_names()`.
- **`Pipeline::stage_index`** (`orchestrator.rs`) — zero-based index of the
  first stage with a given name, or `None` if absent.
- **`BackpressureGuard::is_empty`** (`orchestrator.rs`) — `true` when no slots
  are currently in use (depth == 0).
- **`BackpressureGuard::available_capacity`** (`orchestrator.rs`) — number of
  additional request slots available before the hard cap.

---

## [1.19.0] - 2026-03-20

### Added

- **`LatencyHistogram::has_data`** (`metrics.rs`) — `true` when at least one
  sample has been recorded; guards percentile calls after a reset.
- **`MetricsSnapshot::success_rate`** (`metrics.rs`) — `1.0 - failure_rate()`
  convenience complement (also present in `RuntimeMetrics`).
- **`MetricsSnapshot::tool_success_count`** (`metrics.rs`) — successful calls
  for a named tool (`calls - failures`) with `saturating_sub` guard.
- **`MetricsSnapshot::tool_failure_rate`** (`metrics.rs`) — per-tool failure
  fraction; returns `0.0` if the tool has no recorded calls.
- **`RuntimeMetrics::is_active`** (`metrics.rs`) — `true` when
  `active_sessions > 0`; useful for health-check endpoints.
- **`Deduplicator::is_idle`** (`orchestrator.rs`) — `true` when there are no
  in-flight requests; convenient for test teardown and drain-wait loops.
- **`CircuitBreaker::is_healthy`** (`orchestrator.rs`) — `true` in both
  `Closed` and `HalfOpen` states (calls are allowed); `false` only when `Open`.

---

## [1.18.0] - 2026-03-20

### Added

- **`AgentSession::total_latency_ms`** (`runtime.rs`) — sum of all step
  durations in milliseconds; cheaper than allocating a Vec via `step_durations_ms`.
- **`AgentSession::action_sequence`** (`runtime.rs`) — ordered list of action
  names as owned `String`s; unlike `all_actions()` the result outlives the borrow.
- **`EpisodicStore::importance_avg`** (`memory.rs`) — arithmetic mean importance
  for an agent; returns `0.0` for empty agents.
- **`EpisodicStore::deduplicate_content`** (`memory.rs`) — remove exact-content
  duplicates for an agent, keeping the episode with the highest importance.
- **`WorkingMemory::iter_sorted`** (`memory.rs`) — all entries as `(key, value)`
  pairs sorted alphabetically by key; deterministic alternative to `iter()`.
- **`SemanticStore::get_value`** (`memory.rs`) — return just the stored value
  for a key; simpler than `retrieve_by_key` when tags are not needed.
- **`GraphStore::hub_nodes`** (`graph.rs`) — return all entities with
  out-degree ≥ threshold; useful for identifying gateway nodes.
- **`GraphStore::incident_relationships`** (`graph.rs`) — return all
  relationships where an entity is the source or target.
- **`ToolRegistry::rename_tool`** (`agent.rs`) — rename a registered tool,
  updating both the registry key and the `ToolSpec::name` field.

---

## [1.17.0] - 2026-03-20

### Added

- **`LatencyHistogram::p75`** (`metrics.rs`) — 75th-percentile latency; fills
  the gap between `p50` and `p95`.
- **`LatencyHistogram::p90`** (`metrics.rs`) — 90th-percentile latency.
- **`CompletionOptions::with_timeout_secs`** (`providers.rs`) — convenience
  builder that accepts seconds; mirrors `AgentConfig::with_loop_timeout_secs`.
- **`CompletionOptions::with_timeout_ms`** (`providers.rs`) — convenience
  builder that accepts milliseconds.
- **`CompletionOptions::has_stop_sequences`** (`providers.rs`) — returns `true`
  when at least one stop sequence is configured.
- **`CompletionOptions::stop_sequence_count`** (`providers.rs`) — number of
  configured stop sequences.

---

## [1.16.0] - 2026-03-20

### Added

- **`AgentSession::action_counts`** (`runtime.rs`) — `HashMap<action, count>`
  of how many times each action was taken in a session.
- **`AgentSession::unique_actions`** (`runtime.rs`) — sorted, deduplicated list
  of action names used in a session.
- **`EpisodicStore::agent_ids`** (`memory.rs`) — return all `AgentId`s that have
  at least one stored episode.
- **`EpisodicStore::find_by_content`** (`memory.rs`) — find episodes whose
  content contains a given substring, sorted by descending importance.
- **`GraphStore::top_n_by_out_degree`** (`graph.rs`) — return the top-N entities
  sorted by outgoing-edge count; useful for finding hub nodes.
- **`GraphStore::remove_entity_and_edges`** (`graph.rs`) — remove an entity and
  all incident edges in a single lock acquisition.
- **`ToolRegistry::tool_specs`** (`agent.rs`) — return references to all
  registered `ToolSpec`s for inspection or serialization.
- **`RetryPolicy::delay_ms_for`** (`orchestrator.rs`) — return the delay for a
  given attempt as a `u64` millisecond value; easier for logging than `Duration`.
- **`BackpressureGuard::soft_limit`** (`orchestrator.rs`) — read accessor for
  the configured soft capacity, or `None` if no soft limit was set.
- **`Pipeline::has_stage`** (`orchestrator.rs`) — return `true` if a stage with
  the given name is registered.

---

## [1.15.0] - 2026-03-20

### Added

- **`EpisodicStore::clear_for`** (`memory.rs`) — remove all episodes for a
  specific agent in one call; returns the count of removed items.
- **`EpisodicStore::importance_sum`** (`memory.rs`) — sum of all importance
  scores for an agent; useful for computing average importance.
- **`WorkingMemory::merge_from`** (`memory.rs`) — copy all entries from another
  `WorkingMemory`; capacity limits and eviction policies apply.
- **`WorkingMemory::entry_count_satisfying`** (`memory.rs`) — count entries
  matching a `(key, value) -> bool` predicate without allocating a result vec.
- **`SemanticStore::update_value`** (`memory.rs`) — update the stored value of
  an entry by key; mirrors `update_tags` for the value field.
- **`GraphStore::max_out_degree_entity`** (`graph.rs`) — return the entity with
  the most outgoing edges, or `None` for an empty graph.
- **`GraphStore::leaf_nodes`** (`graph.rs`) — return all entities with
  out-degree = 0; efficiently found via the adjacency index.
- **`AgentConfig::with_max_iterations`** (`agent.rs`) — fluent builder to set
  `max_iterations` after construction; completes the builder surface.
- **`ToolRegistry::tool_names_owned`** (`agent.rs`) — return registered tool
  names as `Vec<String>`; unlike `tool_names()` this does not borrow `self`.

---

## [1.14.0] - 2026-03-20

### Added

- **`AgentRuntimeError::is_agent_loop`** (`error.rs`) — predicate for `AgentLoop` variant.
- **`AgentRuntimeError::is_orchestration`** (`error.rs`) — predicate for `Orchestration` variant.
- **`AgentRuntimeError::is_persistence`** (`error.rs`) — predicate for `Persistence` variant.
- **`AgentRuntimeError::is_not_configured`** (`error.rs`) — predicate for `NotConfigured` variant.
- **`AgentRuntimeError::is_deduplication_conflict`** (`error.rs`) — predicate for `DeduplicationConflict` variant.
- **`AgentRuntimeError::is_retryable`** (`error.rs`) — returns `true` for `Provider` and
  `Persistence` errors (transient/IO); returns `false` for all logic/configuration errors.
- **`AgentSession::thoughts_containing`** (`runtime.rs`) — filter steps by case-insensitive
  substring match on the thought field; mirrors `observations_matching`.
- **`AgentSession::step_durations_ms`** (`runtime.rs`) — return all per-step durations as
  `Vec<u64>`; useful for computing custom latency percentiles over a session.
- **`MetricsSnapshot::success_rate`** (`metrics.rs`) — `1.0 - failure_rate()`; complement
  to the existing `failure_rate()`.
- **`MetricsSnapshot::tool_success_count`** (`metrics.rs`) — successful calls for a named
  tool (`calls - failures`); uses `saturating_sub` to guard against counter skew.
- **`MetricsSnapshot::tool_failure_rate`** (`metrics.rs`) — per-tool failure rate; returns
  `0.0` if the tool has no recorded calls.
- **`RuntimeMetrics::success_rate`** (`metrics.rs`) — live-read counterpart to `failure_rate()`.
- **`RuntimeMetrics::is_active`** (`metrics.rs`) — `true` when `active_sessions > 0`; useful
  for health-check endpoints.

---

## [1.12.0] - 2026-03-20

### Added

- **`EpisodicStore::add_episode_with_tags`** (`memory.rs`) — create an episode
  with explicit tags in a single call; combines `add_episode` + `update_tags_by_id`.
- **`EpisodicStore::remove_by_id`** (`memory.rs`) — delete a specific episode by
  `MemoryId`; returns `Ok(true)` if found and removed, `Ok(false)` if absent.
- **`EpisodicStore::update_tags_by_id`** (`memory.rs`) — replace the tag list of
  an existing episode; returns `Ok(true/false)` like other by-id mutators.
- **`EpisodicStore::max_importance_for`** (`memory.rs`) — return the highest
  importance score for a given agent, or `None` if no episodes exist.
- **`WorkingMemory::snapshot`** (`memory.rs`) — return a `HashMap<String,String>`
  clone of all current entries; useful for serialization and debugging.
- **`WorkingMemory::pop_oldest`** (`memory.rs`) — remove and return the
  oldest inserted `(key, value)` pair, or `None` if empty.
- **`SemanticStore::keys_with_tag`** (`memory.rs`) — return all entry keys that
  carry a specific tag; complement to `retrieve(&[tag])` when only keys are needed.
- **`RuntimeMetrics::top_tools_by_failures`** (`metrics.rs`) — return the top-N
  tools sorted by descending failure count; useful for SLO dashboards.
- **`LatencyHistogram::sum_ms`** (`metrics.rs`) — return the sum of all recorded
  latency samples in milliseconds; enables computing mean without floating-point.
- **`CircuitBreaker::threshold`** (`orchestrator.rs`) — read accessor for the
  failure threshold configured at construction.
- **`CircuitBreaker::recovery_window`** (`orchestrator.rs`) — read accessor for
  the recovery window `Duration` configured at construction.
- **`Pipeline::get_stage_name_at`** (`orchestrator.rs`) — return the name of a
  stage by zero-based index, or `None` if out of bounds.
- **`AgentConfig::clone_with_model`** (`agent.rs`) — clone the config and override
  only the model name; handy for A/B testing different models on the same config.
- **`ToolSpec::with_name`** (`agent.rs`) — override the tool name after
  construction; mirrors the existing `with_description` builder method.
---

## [1.13.0] - 2026-03-20

### Added

- **`EpisodicStore::count_for`** (`memory.rs`) — return the number of episodes
  stored for a given agent without triggering recall sorting or recall-count
  increments; faster than `recall(agent, usize::MAX)?.len()`.
- **`EpisodicStore::recall_by_tag`** (`memory.rs`) — recall episodes for an agent
  that carry a specific tag, sorted by descending importance.
- **`EpisodicStore::merge_from`** (`memory.rs`) — copy all episodes from another
  store for a given agent, respecting per-agent capacity and eviction policies.
- **`WorkingMemory::get_all_keys`** (`memory.rs`) — return all keys in insertion
  order without their values; cheaper than a full `snapshot()` when only keys
  are needed.
- **`WorkingMemory::replace_all`** (`memory.rs`) — atomically replace all entries
  with a new `HashMap`; capacity limits are enforced as entries are inserted.
- **`SemanticStore::count`** (`memory.rs`) — alias for `len()` using conventional
  collection naming; returns the total number of stored entries.
- **`SemanticStore::remove`** (`memory.rs`) — delete the first entry with the given
  key; returns `Ok(true)` if found and removed.
- **`GraphStore::neighbor_ids`** (`graph.rs`) — return the IDs of all directly
  reachable neighbours (outgoing-edge targets) without allocating full `Entity`
  objects; cheaper than `neighbor_entities`.
- **`RetryPolicy::max_attempts`** (`orchestrator.rs`) — read accessor for the
  configured maximum attempt count; previously only available via `can_retry`.
- **`Pipeline::stage_names_owned`** (`orchestrator.rs`) — return stage names as
  owned `String`s; unlike `stage_names()` the result can be used after `self` is
  moved or mutated.

---

## [1.11.0] - 2026-03-20

### Added

- **`SemanticStore::list_keys`** (`memory.rs`) — return all stored entry keys in
  insertion order; complements the existing `list_tags()`.
- **`SemanticStore::update_tags`** (`memory.rs`) — update the tag list for an
  existing entry in-place; returns `Ok(true/false)` like `update()`.
- **`EpisodicStore::recall_since`** (`memory.rs`) — recall episodes whose
  `timestamp` is at or after a `chrono::DateTime<Utc>` cutoff, ordered by
  descending importance; `limit = 0` returns all matching items.
- **`EpisodicStore::update_content`** (`memory.rs`) — update the `content` field
  of an existing episode by `MemoryId`; returns `Ok(true/false)`.
- **`DecayPolicy::half_life_hours`** (`memory.rs`) — read accessor for the
  previously private `half_life_hours` field.
- **`MemoryItem` `Display` impl** (`memory.rs`) — one-liner
  `[id] importance=X.XX recalls=N content="..."` for debug output and logging.
- **`WorkingMemory::capacity`** (`memory.rs`) — returns the maximum number of
  entries set at construction time; completes the capacity-introspection surface.
- **`RetryPolicy::with_base_delay_ms`** (`orchestrator.rs`) — fluent builder to
  override the base retry delay after construction; mirrors `with_max_attempts`.
- **`BackpressureGuard::reset_depth`** (`orchestrator.rs`) — force the in-flight
  counter to zero; useful for test teardown when slots were never released.
- **`Pipeline::remove_stage`** (`orchestrator.rs`) — remove the first stage with
  a matching name; returns `true` if found.
- **`Pipeline::clear`** (`orchestrator.rs`) — remove all stages while preserving
  the error handler.
- **`GraphStore::entity_ids`** (`graph.rs`) — return all entity IDs without
  allocating full `Entity` objects; cheaper than `all_entities()` when only IDs
  are needed.
- **`GraphStore::is_empty`** (`graph.rs`) — `true` when the graph contains no
  entities.
- **`GraphStore::clear`** (`graph.rs`) — remove all entities, relationships, and
  the adjacency index in one call; resets the cycle-detection cache.
- **`AgentSession::failed_steps`** (`runtime.rs`) — return references to steps
  whose observation starts with `{"error"` or contains `"error"` (case-insensitive);
  useful for post-run diagnostics.
- **`AgentConfig::validate`** (`agent.rs`) — validate configuration and return
  `Err(AgentRuntimeError::AgentLoop)` with a descriptive message; a structured
  alternative to the existing `is_valid() -> bool`.
- **`LatencyHistogram::clear`** (`metrics.rs`) — alias for `reset()`; resets all
  histogram counters to zero using conventional naming.
- **`RuntimeMetrics::tool_names`** (`metrics.rs`) — return a sorted list of tool
  names that have at least one recorded call; useful for iterating over per-tool
  metrics without knowing registered names in advance.
- **`ToolRegistry::is_empty`** (`agent.rs`) — `true` when no tools are registered;
  symmetric counterpart to the existing `tool_count() == 0` check.
- **Graph reverse adjacency index** (`graph.rs`) — `GraphInner` now maintains a
  `reverse_adjacency: HashMap<EntityId, Vec<EntityId>>` in sync with every
  `add_relationship`, `remove_relationship`, and `remove_entity` call; reduces
  `in_degree` from O(|E|) to O(in-degree), `source_nodes` from O(V+E) to O(V),
  and `isolates` from O(V+E) to O(V).

---

## [1.10.0] - 2026-03-20

### Added

- **`src/types.rs`** — new unconditional module housing `AgentId` and `MemoryId`.
  Both types are now available without enabling the `memory` feature; `memory.rs`
  re-exports them for backward compatibility.
- **`AgentRuntime` / `AgentRuntimeBuilder` no longer require the `memory` feature**
  (`runtime.rs`) — memory-specific fields (`EpisodicStore`, `WorkingMemory`) and
  their accessor/builder methods are now individually gated with per-field
  `#[cfg(feature = "memory")]`, so the runtime compiles and operates without the
  `memory` feature enabled.
- **`AgentRuntime::run_agent_with_provider`** (`runtime.rs`, `providers` feature) —
  convenience method that wires an `Arc<dyn LlmProvider>` into `run_agent` directly,
  removing the need to write a manual closure when using a built-in provider.
- **`FilePersistenceBackend::exists`** (`persistence.rs`) — async probe that returns
  `Ok(true)` when a checkpoint key already exists on disk; useful for conditional
  save/load patterns without a full load round-trip.
- **`WorkingMemory::clear`** (`memory.rs`) — remove all entries from a working
  memory instance without recreating it; completes the CRUD surface alongside the
  existing `remove` and `keys` methods.

### Changed

- **`SemanticStore::store_with_embedding`** (`memory.rs`) — embeddings are now
  L2-normalised to unit vectors at insert time.  `retrieve_similar` normalises the
  query vector at query time and computes a dot product instead of full cosine
  similarity, reducing per-query work from two norm computations + dot product to
  one norm computation + dot product.  The removed `cosine_similarity` helper was
  unused after this change and has been deleted.
- **`EpisodicStore::recall` / `recall_recent` / `recall_all` / `recall_tagged` /
  `SemanticStore::retrieve_similar` / `WorkingMemory` sort paths** (`memory.rs`) —
  all `sort_by` calls replaced with `sort_unstable_by`.  For `f32` importance scores
  the ordering among equal elements is irrelevant, making the unstable variant
  strictly faster with no observable behavioural difference.
- **`Deduplicator::check_and_register` / `check`** (`orchestrator.rs`) — full-scan
  TTL expiry (`retain` over the entire cache and in-flight maps) now runs only every
  64 calls instead of on every invocation.  Correctness is maintained by an inline
  per-key expiry check that treats a found-but-expired entry as a cache miss.  The
  new `call_count: u64` field on `DeduplicatorInner` wraps silently on overflow
  (all `u64::MAX` traffic is handled correctly).

### Tests

- **`test_react_iteration_span_is_emitted`** / **`test_tool_dispatch_span_is_emitted`**
  (`tests/test_tracing_logging.rs`) — new integration tests using a custom
  `tracing_subscriber::Layer` (`SpanCollector`) to assert that the
  `"react_iteration"` and `"tool_dispatch"` span names are emitted during a
  real `ReActLoop` run.
- **`error_loop_timeout_fires_when_inference_is_slow`**
  (`tests/test_error_paths.rs`) — verifies that `AgentConfig::with_loop_timeout_ms`
  actually terminates the loop when the cumulative iteration time exceeds the
  deadline, using a 20 ms timeout and a 15 ms-per-iteration sleep to reliably
  trigger it.

---

## [1.9.0] - 2026-03-20

### Added

- **`Message::is_user`** / **`is_assistant`** / **`is_system`** / **`is_tool`**
  (`agent.rs`) — boolean role predicates replacing verbose `==` comparisons.
- **`ToolSpec::with_description`** (`agent.rs`) — override the tool description
  after initial construction.
- **`AgentSession::average_step_duration_ms`** (`runtime.rs`) — arithmetic mean
  of per-step durations; returns `0` for empty sessions.
- **`AgentSession::slowest_step`** / **`fastest_step`** (`runtime.rs`) — return
  a reference to the step with the highest/lowest `step_duration_ms`; useful for
  profiling long-running agents.
- **`AgentSession::all_thoughts`** / **`all_observations`** (`runtime.rs`) —
  collect all thought or observation strings across every step.
- **`WorkingMemory::values`** (`memory.rs`) — return all stored values in
  insertion order, parallel to the existing `keys()` method.
- **`Pipeline::is_empty`** (`orchestrator.rs`) — return `true` when no stages
  have been added.
- **`BackpressureGuard::remaining_capacity`** (`orchestrator.rs`) — number of
  additional slots that can be acquired before hitting hard capacity.

### Fixed

- **`Deduplicator::purge_expired`** (`orchestrator.rs`) — after removing expired
  entries from `cache`, the `cache_order` VecDeque was not updated, allowing
  ghost keys to accumulate and cause spurious no-op evictions on the next
  `complete` call when `max_entries` is set.  `purge_expired` now calls
  `cache_order.retain` to drop any keys that are no longer in `cache`.

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

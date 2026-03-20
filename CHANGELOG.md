# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

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

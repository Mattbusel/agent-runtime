# Troubleshooting Guide

Common failure modes and how to diagnose them.

---

## Backpressure Shedding

**Symptom:** `run_agent` returns `Err(AgentRuntimeError::BackpressureShed { depth, capacity })`.

**Cause:** More concurrent agent sessions are running than `BackpressureGuard::new(capacity)` permits.

**Diagnosis:**
```rust
// Check depth in your metrics
let depth = runtime.metrics().active_sessions();
println!("active={depth}");
```

**Fix options:**
1. Increase `BackpressureGuard` capacity to match your expected concurrency.
2. Add a soft limit and back off before hitting the hard cap:
   ```rust
   BackpressureGuard::new(100)?.with_soft_limit(80)?
   ```
3. Queue requests rather than calling `run_agent` directly when `depth()` is near capacity.

---

## Circuit Breaker Opens

**Symptom:** Tool calls return `Err(AgentRuntimeError::CircuitOpen { service })`.

**Cause:** The tool's circuit breaker has tripped after `threshold` consecutive failures.

**Diagnosis:**
```rust
let cb = /* your CircuitBreaker */;
println!("state: {:?}", cb.state()?);
println!("failures: {}", cb.failure_count()?);
```

**Fix options:**
1. Wait for `recovery_window` to elapse; the circuit will transition to `HalfOpen` automatically.
2. Investigate the underlying failure by examining the observations in your `AgentSession::steps`.
3. Reduce `threshold` only if you want to open faster; increase `recovery_window` to give the downstream service more time.

---

## Memory Decay Tuning

`DecayPolicy::exponential(half_life_hours)` controls how quickly memory importance decays.

| half_life_hours | Effect |
|---|---|
| 1 | Memories halve in importance every hour — very aggressive |
| 24 | Memories halve every day — moderate |
| 168 | Memories halve every week — gentle |

**Symptom:** Important memories are being dropped from injected context.

**Fix:** Increase `half_life_hours`, or set `with_max_memory_recalls(n)` to a larger value so lower-importance items can still be included.

**Symptom:** The context is full of stale low-importance memories.

**Fix:** Decrease `half_life_hours`, or set an absolute `max_age_hours` eviction:
```rust
EpisodicStore::with_max_age(72.0)? // evict anything older than 3 days
```

---

## Token Budget Overruns

`AgentConfig::with_max_memory_tokens(budget)` limits injected memories by approximate token count (heuristic: 4 chars ≈ 1 token).

**Symptom:** `session.memory_hits` is lower than expected.

**Cause:** Items are being trimmed by the token budget.

**Fix:** Increase the budget, or reduce individual memory content length when calling `add_episode`.

---

## Step Duration Tracking

`ReActStep::step_duration_ms` measures wall-clock time from inference call to observation completion.

**Symptom:** Sessions are slow but you cannot identify which step.

**Fix:**
```rust
for (i, step) in session.steps.iter().enumerate() {
    println!("step {i}: {}ms — action={}", step.step_duration_ms, step.action);
}
```

---

## Per-Tool Reliability

Use `RuntimeMetrics::per_tool_calls_snapshot()` and `per_tool_failures_snapshot()` to identify unreliable tools:

```rust
let calls = runtime.metrics().per_tool_calls_snapshot();
let failures = runtime.metrics().per_tool_failures_snapshot();
for (tool, count) in &calls {
    let fail = failures.get(tool).copied().unwrap_or(0);
    let rate = fail as f64 / *count as f64 * 100.0;
    println!("{tool}: {count} calls, {fail} failures ({rate:.1}%)");
}
```

---

## Lock Contention

The `timed_lock` wrapper in the orchestrator logs a warning when mutex acquisition takes > 5 ms:

```
WARN slow mutex acquisition duration_ms=12 ctx=BackpressureGuard::try_acquire
```

**Fix:** Reduce the time locks are held (avoid I/O inside a lock), or switch to `parking_lot::Mutex` for lower overhead.

---

## Cycle Detection in Graph

`GraphStore::detect_cycles()` returns whether the directed graph has any cycles. Results are **cached** and invalidated on any mutation (`add_entity`, `add_relationship`, `remove_entity`).

```rust
if graph.detect_cycles()? {
    eprintln!("WARNING: graph contains cycles — algorithms may not terminate");
}
```

---

## Persistence Failures

**Symptom:** `Err(AgentRuntimeError::Persistence(...))` during `run_agent`.

**Cause:** The checkpoint backend's `save` call failed (e.g. disk full, directory missing).

**Diagnosis:** The error message includes the key and the underlying OS error. Check that `base_dir` exists and is writable before creating `FilePersistenceBackend`.

**Note:** Per-step checkpoint failures are logged as warnings and do not abort the agent run. Only the final session checkpoint propagates as an error.

---

## Concurrency and Thread Safety

All stores (`EpisodicStore`, `SemanticStore`, `WorkingMemory`, `GraphStore`) use `Arc<Mutex<_>>` and are `Send + Sync`. They can be shared across Tokio tasks via `Arc::clone`.

If a task panics while holding a lock, the mutex is poisoned. The `recover_lock` utility logs a warning and continues — the data may be in an inconsistent state. Monitor for `"mutex poisoned"` log lines in production.

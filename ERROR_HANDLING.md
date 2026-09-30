# Error Handling Guide

This document describes the error model used by `llm-agent-runtime`, explains
each variant of `AgentRuntimeError`, and provides recipes for robust error
handling in production code.

---

## The `AgentRuntimeError` enum

All fallible public APIs return `Result<T, AgentRuntimeError>`.  The enum is
defined in `src/error.rs` and derives `thiserror::Error` so every variant
carries a human-readable `Display` message.

| Variant | When raised | Key fields |
|---------|-------------|------------|
| `ToolNotFound(name)` | `ToolRegistry::call` is given an unknown tool name | `name: String` |
| `ToolExecution(msg)` | A registered tool handler returns `Err` | `msg: String` |
| `MaxIterationsReached(n)` | `ReActLoop::run` exhausts its iteration budget | `n: usize` |
| `ParseError(msg)` | `parse_react_step` cannot interpret the LLM response | `msg: String` |
| `LockPoisoned(ctx)` | A `Mutex` was poisoned; lock recovered automatically | `ctx: String` |
| `Persistence(msg)` | `FilePersistenceBackend` encounters an I/O error | `msg: String` |
| `Internal(msg)` | Unexpected internal invariant violation | `msg: String` |
| `Validation { field, code, message }` | Input fails a precondition check | structured fields |

### `Validation` variant

Use the `Validation` variant when user-supplied data fails a precondition:

```rust
return Err(AgentRuntimeError::Validation {
    field: "embedding".into(),
    code: "DIM_MISMATCH".into(),
    message: format!("expected {expected}, got {actual}"),
});
```

The structured fields let callers pattern-match on `code` without parsing the
display string.

---

## "Did you mean?" for unknown tools

When `ToolRegistry::call` receives an unknown tool name it automatically
computes a Levenshtein distance against every registered name.  If the closest
match is within edit-distance 3 the error message includes a suggestion:

```
ToolNotFound("calclulate") — did you mean "calculate"?
```

This makes debugging typos in tool names significantly easier.

---

## Handling lock poisoning

All internal `Mutex` locks are acquired through `util::timed_lock`, which:

1. Logs a warning with the lock context name and elapsed acquisition time.
2. Calls `.unwrap_or_else(|p| p.into_inner())` to recover the guard from a
   poisoned mutex rather than propagating a panic.

If you hold a guard across an `await` point in your own code use a `tokio::sync::Mutex`
instead of `std::sync::Mutex` — the latter is not `Send` across `.await`.

---

## Recipes

### Retry on `ToolExecution`

```rust
use llm_agent_runtime::error::AgentRuntimeError;

async fn run_with_retry(runtime: &AgentRuntime, /* … */) -> Result<String, AgentRuntimeError> {
    for attempt in 0..3 {
        match runtime.run(/* … */).await {
            Ok(reply) => return Ok(reply),
            Err(AgentRuntimeError::ToolExecution(msg)) if attempt < 2 => {
                tracing::warn!("tool error (attempt {attempt}): {msg}");
                tokio::time::sleep(std::time::Duration::from_millis(200 << attempt)).await;
            }
            Err(e) => return Err(e),
        }
    }
    unreachable!()
}
```

### Distinguish validation errors by code

```rust
match store.retrieve_similar(&query_embedding, 5) {
    Err(AgentRuntimeError::Validation { code, .. }) if code == "DIM_MISMATCH" => {
        // Re-embed with the correct dimension
    }
    Err(e) => return Err(e),
    Ok(results) => { /* … */ }
}
```

### Surface `MaxIterationsReached` to the end user

```rust
match react_loop.run(messages, &registry).await {
    Err(AgentRuntimeError::MaxIterationsReached(n)) => {
        eprintln!("Agent hit the {n}-step limit; try increasing max_iterations.");
    }
    Err(e) => return Err(e),
    Ok(answer) => println!("{answer}"),
}
```

---

## Feature-gated errors

`Persistence` errors are only reachable when the `persistence` feature is
enabled.  If you compile without that feature, match arms for `Persistence`
will produce a dead-code warning — wrap them in `#[cfg(feature = "persistence")]`.

---

## Logging recommendations

Enable `tracing` at the `DEBUG` level during development to see per-step
observations emitted by `ReActLoop`.  In production `INFO` is sufficient; set
`RUST_LOG=llm_agent_runtime=warn` to suppress routine chatter.

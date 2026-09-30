# QUICKSTART — Your First Agent in 5 Minutes

This guide shows you how to build a working agent using **echo mode** — no API
key required.  The agent runs a full Thought-Action-Observation loop using a
local inference function you supply.

---

## 1. Add the dependency

```toml
# Cargo.toml
[dependencies]
llm-agent-runtime = { version = "*", features = ["memory"] }
tokio = { version = "1", features = ["full"] }
```

---

## 2. Write your first agent

Create `src/main.rs`:

```rust
use llm_agent_runtime::prelude::*;

#[tokio::main]
async fn main() -> Result<(), AgentRuntimeError> {
    // 1. Build the runtime (no API key needed).
    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(5, "echo-model"))
        .build();

    // 2. Define an "echo" inference function.
    //    A real implementation would call an LLM API here.
    //    This one always produces a FINAL_ANSWER immediately.
    let infer = |ctx: String| async move {
        // The ReAct loop passes the full context string.
        // We reply with a minimal valid ReAct response.
        let _ = ctx; // echo mode ignores the context
        "Thought: The user greeted me.\nAction: FINAL_ANSWER Hello from agent-runtime!".to_string()
    };

    // 3. Run the agent.
    let session = runtime
        .run_agent(AgentId::new("my-agent"), "Hello!", infer)
        .await?;

    println!("Completed in {} step(s).", session.step_count());
    println!(
        "Final answer: {}",
        session.final_answer().unwrap_or("(none)")
    );

    Ok(())
}
```

Run it:

```bash
cargo run
```

Expected output:

```
Completed in 1 step(s).
Final answer: FINAL_ANSWER Hello from agent-runtime!
```

---

## 3. Add a tool

Tools let the agent take actions — searching, calculating, writing files, etc.

```rust
use llm_agent_runtime::prelude::*;
use serde_json::{json, Value};

#[tokio::main]
async fn main() -> Result<(), AgentRuntimeError> {
    // Register a simple calculator tool.
    let mut loop_ = ReActLoop::new(AgentConfig::new(5, "echo-model"));
    loop_.register_tool(ToolSpec::new("add", "Add two numbers", |args: Value| {
        let a = args["a"].as_f64().unwrap_or(0.0);
        let b = args["b"].as_f64().unwrap_or(0.0);
        json!({ "result": a + b })
    }));

    // Inference function that calls the `add` tool once, then answers.
    let mut call_count = 0usize;
    let steps = loop_
        .run("What is 3 + 4?", |_ctx| {
            call_count += 1;
            async move {
                if call_count == 1 {
                    // First iteration: invoke the add tool.
                    r#"Thought: I need to add 3 and 4.
Action: add {"a": 3, "b": 4}"#.to_string()
                } else {
                    // Second iteration: produce the final answer.
                    "Thought: The result is 7.\nAction: FINAL_ANSWER 7".to_string()
                }
            }
        })
        .await?;

    for step in &steps {
        println!("{}", step.summary());
    }
    Ok(())
}
```

---

## 4. Track conversation history

Keep a bounded log of every message in the session:

```rust
use llm_agent_runtime::conversation::ConversationHistory;
use llm_agent_runtime::agent::Message;

let mut history = ConversationHistory::new(20); // keep last 20 turns
history.push(Message::user("Hello!"));
history.push(Message::assistant("Hi there!"));

// When the history exceeds max_turns, old turns are auto-summarised.
println!("Live turns: {}", history.len());
println!("Context:\n{}", history.to_context_string());
```

---

## 5. Add budget tracking

Guard against runaway token usage:

```rust
use llm_agent_runtime::budget::{BudgetTracker, BudgetConfig};

let tracker = BudgetTracker::new(
    BudgetConfig::new()
        .with_max_cost_usd(1.00)          // halt after $1.00
        .with_max_total_tokens(100_000),  // halt after 100k tokens
);

// Call after each inference.
tracker.record_usage("my-agent", "gpt-4o", 512, 128)?;

// Export a JSON cost report.
println!("{}", tracker.export_json()?);
```

---

## 6. Next steps

| What you want | Where to look |
|---|---|
| Real LLM (Anthropic / OpenAI) | `features = ["anthropic"]` or `["openai"]`, see `examples/anthropic_provider.rs` |
| Persist checkpoints to disk | `features = ["persistence"]`, see `examples/custom_persistence.rs` |
| Multi-agent orchestration | `examples/multi_agent.rs` |
| Knowledge graph | `features = ["graph"]`, see `examples/graph_query_agent.rs` |
| Distributed coordination (Redis) | `features = ["distributed"]`, see `src/distributed.rs` |
| OTel tracing | `features = ["otel"]`, see `src/telemetry.rs` |
| Full API docs | `cargo doc --open` |

---

## Troubleshooting

**`parse_react_step` error** — your inference function must return a string
containing both `Thought:` and `Action:` on separate lines.  Check the
`TROUBLESHOOTING.md` for a full list of valid formats.

**Compilation error with `#[deny(missing_docs)]`** — all public items require
doc comments.  Add `//!` module docs and `///` item docs before building.

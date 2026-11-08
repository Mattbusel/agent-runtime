# agent-runtime

[![Crates.io](https://img.shields.io/crates/v/llm-agent-runtime.svg)](https://crates.io/crates/llm-agent-runtime) [![docs.rs](https://docs.rs/llm-agent-runtime/badge.svg)](https://docs.rs/llm-agent-runtime) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Rust crate for building LLM agents on Tokio.

- Runs ReAct and plan-execute loops
- Keeps memory and a knowledge graph
- Calls Anthropic and OpenAI with streaming
- Connects agents on a message bus
- Tests agents with no API key

## Install

```toml
llm-agent-runtime = { version = "1.76", features = ["anthropic", "openai"] }
```

## Use

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

    println!("Answer: {}", session.final_answer().unwrap_or_default());
    Ok(())
}
```

[Docs](https://docs.rs/llm-agent-runtime) · [crates.io](https://crates.io/crates/llm-agent-runtime)

More: [docs/DETAILS.md](docs/DETAILS.md)

License: MIT

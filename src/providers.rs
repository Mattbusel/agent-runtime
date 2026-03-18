//! # Module: Providers
//!
//! ## Responsibility
//! Provides built-in LLM inference integrations behind the `LlmProvider` trait.
//! Optional feature flags gate each provider:
//! - `anthropic` — Anthropic Messages API
//! - `openai`    — OpenAI Chat Completions API (and compatible endpoints)
//!
//! ## Guarantees
//! - `LlmProvider` is an async, object-safe trait
//! - Both providers are `Send + Sync` and usable behind `Arc<dyn LlmProvider>`
//! - Non-panicking: all operations return `Result`
//!
//! ## Feature Gate
//! This module is only compiled when the `providers` feature (or a sub-feature) is enabled.

use crate::error::AgentRuntimeError;
use async_trait::async_trait;

// ── LlmProvider ───────────────────────────────────────────────────────────────

/// Abstraction over an LLM inference endpoint.
///
/// Implement this trait to integrate any model API with `AgentRuntime`.
/// Built-in implementations are provided for Anthropic and OpenAI
/// when the corresponding feature flags are enabled.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Send a prompt to the model and return the completion text.
    ///
    /// # Arguments
    /// * `prompt` — the full prompt / context string
    /// * `model`  — model identifier (e.g. `"claude-sonnet-4-6"`, `"gpt-4o"`)
    async fn complete(&self, prompt: &str, model: &str) -> Result<String, AgentRuntimeError>;
}

// ── AnthropicProvider ─────────────────────────────────────────────────────────

#[cfg(feature = "anthropic")]
/// Built-in provider for the Anthropic Messages API.
///
/// Requires the `anthropic` feature flag.
///
/// # Example
/// ```no_run
/// use agent_runtime::providers::AnthropicProvider;
/// let provider = AnthropicProvider::new("sk-ant-...");
/// ```
pub struct AnthropicProvider {
    api_key: String,
    client: reqwest::Client,
}

#[cfg(feature = "anthropic")]
impl AnthropicProvider {
    const API_URL: &'static str = "https://api.anthropic.com/v1/messages";
    const API_VERSION: &'static str = "2023-06-01";
    const MAX_TOKENS: u32 = 1024;

    /// Create a new Anthropic provider with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            client: reqwest::Client::new(),
        }
    }
}

#[cfg(feature = "anthropic")]
#[async_trait]
impl LlmProvider for AnthropicProvider {
    async fn complete(&self, prompt: &str, model: &str) -> Result<String, AgentRuntimeError> {
        let body = serde_json::json!({
            "model": model,
            "max_tokens": Self::MAX_TOKENS,
            "messages": [{ "role": "user", "content": prompt }]
        });

        let response = self
            .client
            .post(Self::API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", Self::API_VERSION)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| AgentRuntimeError::Provider(format!("Anthropic request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(AgentRuntimeError::Provider(format!(
                "Anthropic API error {status}: {text}"
            )));
        }

        let json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| AgentRuntimeError::Provider(format!("Anthropic parse failed: {e}")))?;

        let text = json["content"]
            .as_array()
            .and_then(|arr| arr.first())
            .and_then(|block| block["text"].as_str())
            .ok_or_else(|| {
                AgentRuntimeError::Provider(
                    "Anthropic response missing content[0].text".into(),
                )
            })?;

        Ok(text.to_owned())
    }
}

// ── OpenAiProvider ────────────────────────────────────────────────────────────

#[cfg(feature = "openai")]
/// Built-in provider for the OpenAI Chat Completions API.
///
/// Also compatible with Azure OpenAI and any OpenAI-compatible endpoint.
/// Requires the `openai` feature flag.
///
/// # Example
/// ```no_run
/// use agent_runtime::providers::OpenAiProvider;
/// let provider = OpenAiProvider::new("sk-...");
/// // For Azure or custom endpoints:
/// let custom = OpenAiProvider::with_base_url("sk-...", "https://my-endpoint/v1");
/// ```
pub struct OpenAiProvider {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
}

#[cfg(feature = "openai")]
impl OpenAiProvider {
    const DEFAULT_BASE_URL: &'static str = "https://api.openai.com/v1";

    /// Create a new OpenAI provider with the default base URL.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: Self::DEFAULT_BASE_URL.to_owned(),
            client: reqwest::Client::new(),
        }
    }

    /// Create a provider pointing to a custom base URL (e.g. Azure, local models).
    pub fn with_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: base_url.into(),
            client: reqwest::Client::new(),
        }
    }
}

#[cfg(feature = "openai")]
#[async_trait]
impl LlmProvider for OpenAiProvider {
    async fn complete(&self, prompt: &str, model: &str) -> Result<String, AgentRuntimeError> {
        let url = format!("{}/chat/completions", self.base_url);
        let body = serde_json::json!({
            "model": model,
            "messages": [{ "role": "user", "content": prompt }]
        });

        let response = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| AgentRuntimeError::Provider(format!("OpenAI request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(AgentRuntimeError::Provider(format!(
                "OpenAI API error {status}: {text}"
            )));
        }

        let json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| AgentRuntimeError::Provider(format!("OpenAI parse failed: {e}")))?;

        let text = json["choices"]
            .as_array()
            .and_then(|arr| arr.first())
            .and_then(|choice| choice["message"]["content"].as_str())
            .ok_or_else(|| {
                AgentRuntimeError::Provider(
                    "OpenAI response missing choices[0].message.content".into(),
                )
            })?;

        Ok(text.to_owned())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// A stub provider for testing.
    struct StubProvider {
        response: String,
    }

    #[async_trait]
    impl LlmProvider for StubProvider {
        async fn complete(&self, _prompt: &str, _model: &str) -> Result<String, AgentRuntimeError> {
            Ok(self.response.clone())
        }
    }

    #[tokio::test]
    async fn test_stub_provider_returns_configured_response() {
        let p = StubProvider {
            response: "hello".into(),
        };
        let result = p.complete("prompt", "stub-model").await.unwrap();
        assert_eq!(result, "hello");
    }

    #[tokio::test]
    async fn test_llm_provider_is_object_safe() {
        let p: Arc<dyn LlmProvider> = Arc::new(StubProvider {
            response: "ok".into(),
        });
        let result = p.complete("test", "model").await.unwrap();
        assert_eq!(result, "ok");
    }

    #[tokio::test]
    async fn test_stub_provider_ignores_model_parameter() {
        let p = StubProvider {
            response: "42".into(),
        };
        let r1 = p.complete("q", "model-a").await.unwrap();
        let r2 = p.complete("q", "model-b").await.unwrap();
        assert_eq!(r1, r2);
    }
}

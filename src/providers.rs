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

// ── CompletionOptions ─────────────────────────────────────────────────────────

/// Options for a completion request, passed to [`LlmProvider::complete_with_options`].
///
/// Allows callers to supply per-request parameters (max tokens, temperature,
/// timeout) in addition to the model name, without changing the base
/// [`LlmProvider::complete`] signature.
#[derive(Debug)]
pub struct CompletionOptions<'a> {
    /// Model identifier (e.g. `"claude-sonnet-4-6"`, `"gpt-4o"`).
    pub model: &'a str,
    /// Maximum number of output tokens. Overrides provider defaults when set.
    pub max_tokens: Option<usize>,
    /// Sampling temperature in `[0.0, 2.0]`. Higher = more random.
    pub temperature: Option<f32>,
    /// Per-request wall-clock timeout.
    pub timeout: Option<std::time::Duration>,
}

impl<'a> CompletionOptions<'a> {
    /// Create options with just a model name and all other fields unset.
    pub fn new(model: &'a str) -> Self {
        Self {
            model,
            max_tokens: None,
            temperature: None,
            timeout: None,
        }
    }

    /// Set the maximum output tokens.
    pub fn with_max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = Some(n);
        self
    }

    /// Set the sampling temperature.
    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }

    /// Set the per-request timeout.
    pub fn with_timeout(mut self, d: std::time::Duration) -> Self {
        self.timeout = Some(d);
        self
    }
}

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

    /// Send a prompt with additional per-request options.
    ///
    /// The default implementation ignores `options.max_tokens`,
    /// `options.temperature`, and `options.timeout` and delegates to
    /// `complete(prompt, options.model)`.  Override this method to honour those
    /// fields in your provider implementation.
    async fn complete_with_options(
        &self,
        prompt: &str,
        options: CompletionOptions<'_>,
    ) -> Result<String, AgentRuntimeError> {
        self.complete(prompt, options.model).await
    }

    /// Stream the completion token-by-token.
    ///
    /// Returns a `Receiver` that yields string chunks as they arrive.
    /// The channel closes when the stream is complete or an error occurs.
    ///
    /// # Default implementation
    ///
    /// The default wraps `complete` into a single-chunk stream using a channel
    /// with a buffer of 64 slots.  Custom providers that support true token
    /// streaming should override this method; the 64-slot buffer is sized so
    /// that fast producers do not block waiting for a slow consumer to drain
    /// the first chunk.
    ///
    /// # Note for implementors
    ///
    /// If you override this method, choose a channel capacity that balances
    /// memory use against throughput for your expected token rate.  A capacity
    /// of 1 will cause the producer to block after each token; a capacity of
    /// 0 is unbounded and may exhaust memory on a slow consumer.
    async fn stream_complete(
        &self,
        prompt: &str,
        model: &str,
    ) -> Result<tokio::sync::mpsc::Receiver<Result<String, AgentRuntimeError>>, AgentRuntimeError>
    {
        let result = self.complete(prompt, model).await;
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        // Ignore send error — receiver may already be dropped
        let _ = tx.send(result).await;
        Ok(rx)
    }
}

// ── AnthropicProvider ─────────────────────────────────────────────────────────

#[cfg(feature = "anthropic")]
/// Built-in provider for the Anthropic Messages API.
///
/// Requires the `anthropic` feature flag.
///
/// # Example
/// ```no_run
/// use llm_agent_runtime::providers::AnthropicProvider;
/// let provider = AnthropicProvider::new("sk-ant-...");
/// ```
pub struct AnthropicProvider {
    api_key: String,
    /// Messages API endpoint. Overridable via [`with_base_url`](AnthropicProvider::with_base_url)
    /// to point at a mock server or proxy during testing.
    api_url: String,
    client: reqwest::Client,
    /// Semaphore bounding concurrent SSE background tasks.
    stream_semaphore: std::sync::Arc<tokio::sync::Semaphore>,
    /// Maximum output tokens for `stream_complete`. Falls back to `MAX_TOKENS`
    /// when `None`.
    stream_max_tokens: Option<u32>,
}

#[cfg(feature = "anthropic")]
impl AnthropicProvider {
    const DEFAULT_API_URL: &'static str = "https://api.anthropic.com/v1/messages";
    const API_VERSION: &'static str = "2023-06-01";
    const MAX_TOKENS: u32 = 1024;
    /// Default maximum number of concurrent streaming tasks.
    const DEFAULT_STREAM_CONCURRENCY: usize = 32;

    /// Create a new Anthropic provider with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            api_url: Self::DEFAULT_API_URL.to_owned(),
            client: reqwest::Client::new(),
            stream_semaphore: std::sync::Arc::new(tokio::sync::Semaphore::new(
                Self::DEFAULT_STREAM_CONCURRENCY,
            )),
            stream_max_tokens: None,
        }
    }

    /// Create a provider pointing to a custom API endpoint.
    ///
    /// Useful for testing against a mock server or routing through a proxy.
    pub fn with_base_url(api_key: impl Into<String>, api_url: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            api_url: api_url.into(),
            client: reqwest::Client::new(),
            stream_semaphore: std::sync::Arc::new(tokio::sync::Semaphore::new(
                Self::DEFAULT_STREAM_CONCURRENCY,
            )),
            stream_max_tokens: None,
        }
    }

    /// Create a provider with a custom limit on concurrent streaming tasks.
    ///
    /// Useful when many agents share a single provider and you want to cap
    /// the number of simultaneous SSE parse tasks.
    pub fn with_max_concurrent_streams(api_key: impl Into<String>, max: usize) -> Self {
        Self {
            api_key: api_key.into(),
            api_url: Self::DEFAULT_API_URL.to_owned(),
            client: reqwest::Client::new(),
            stream_semaphore: std::sync::Arc::new(tokio::sync::Semaphore::new(max)),
            stream_max_tokens: None,
        }
    }

    /// Set the maximum output tokens used by [`stream_complete`].
    ///
    /// By default `stream_complete` uses the hard-coded `MAX_TOKENS` constant
    /// (1024).  Call this method to override that value, e.g. when a task
    /// requires longer streamed responses.
    ///
    /// [`stream_complete`]: AnthropicProvider::stream_complete
    pub fn with_stream_max_tokens(mut self, max_tokens: u32) -> Self {
        self.stream_max_tokens = Some(max_tokens);
        self
    }
}

#[cfg(feature = "anthropic")]
#[async_trait]
impl LlmProvider for AnthropicProvider {
    async fn complete(&self, prompt: &str, model: &str) -> Result<String, AgentRuntimeError> {
        self.complete_with_options(prompt, CompletionOptions::new(model))
            .await
    }

    /// Complete with per-request options (max_tokens, temperature).
    ///
    /// Falls back to `Self::MAX_TOKENS` when `options.max_tokens` is not set.
    #[tracing::instrument(skip(self, prompt, options), fields(model = options.model, provider = "anthropic"))]
    async fn complete_with_options(
        &self,
        prompt: &str,
        options: CompletionOptions<'_>,
    ) -> Result<String, AgentRuntimeError> {
        let max_tokens = options
            .max_tokens
            .unwrap_or(Self::MAX_TOKENS as usize) as u32;

        let mut body = serde_json::json!({
            "model": options.model,
            "max_tokens": max_tokens,
            "messages": [{ "role": "user", "content": prompt }]
        });
        if let Some(t) = options.temperature {
            body["temperature"] = serde_json::json!(t);
        }

        let mut req = self
            .client
            .post(&self.api_url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", Self::API_VERSION)
            .header("content-type", "application/json")
            .json(&body);
        if let Some(timeout) = options.timeout {
            req = req.timeout(timeout);
        }
        let response = req
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
                AgentRuntimeError::Provider("Anthropic response missing content[0].text".into())
            })?;

        Ok(text.to_owned())
    }

    /// Stream the completion token-by-token using `chunk()` for true streaming.
    ///
    /// Uses `response.chunk()` to consume the SSE body incrementally so tokens
    /// are emitted as they arrive rather than after the entire response has been
    /// buffered, fixing the original `bytes().await` implementation.
    async fn stream_complete(
        &self,
        prompt: &str,
        model: &str,
    ) -> Result<tokio::sync::mpsc::Receiver<Result<String, AgentRuntimeError>>, AgentRuntimeError>
    {
        let max_tokens = self.stream_max_tokens.unwrap_or(Self::MAX_TOKENS);
        let body = serde_json::json!({
            "model": model,
            "max_tokens": max_tokens,
            "stream": true,
            "messages": [{ "role": "user", "content": prompt }]
        });

        let response = self
            .client
            .post(&self.api_url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", Self::API_VERSION)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                AgentRuntimeError::Provider(format!("Anthropic stream request failed: {e}"))
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(AgentRuntimeError::Provider(format!(
                "Anthropic stream API error {status}: {text}"
            )));
        }

        let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, AgentRuntimeError>>(32);

        let permit = std::sync::Arc::clone(&self.stream_semaphore)
            .acquire_owned()
            .await
            .map_err(|_| {
                AgentRuntimeError::Provider("Anthropic stream semaphore closed".into())
            })?;

        // Parse SSE incrementally using chunk() so tokens are emitted as they
        // arrive rather than after the full response body has been buffered.
        tokio::spawn(async move {
            let _permit = permit;
            let mut response = response;
            let mut buffer = String::new();
            loop {
                match response.chunk().await {
                    Ok(Some(chunk)) => {
                        match String::from_utf8(chunk.to_vec()) {
                            Ok(s) => buffer.push_str(&s),
                            Err(e) => {
                                let _ = tx
                                    .send(Err(AgentRuntimeError::Provider(format!(
                                        "Anthropic stream: invalid UTF-8 in chunk: {e}"
                                    ))))
                                    .await;
                                return;
                            }
                        }
                        // Drain complete SSE lines from the buffer.
                        while let Some(newline) = buffer.find('\n') {
                            let line = buffer[..newline].trim().to_owned();
                            buffer = buffer[newline + 1..].to_owned();
                            if let Some(data) = line.strip_prefix("data: ") {
                                if data == "[DONE]" {
                                    return;
                                }
                                if let Ok(json) =
                                    serde_json::from_str::<serde_json::Value>(data)
                                {
                                    if let Some(delta) = json["delta"]["text"].as_str() {
                                        if tx.send(Ok(delta.to_owned())).await.is_err() {
                                            return;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Ok(None) => break,
                    Err(e) => {
                        let _ = tx
                            .send(Err(AgentRuntimeError::Provider(format!(
                                "Anthropic stream chunk error: {e}"
                            ))))
                            .await;
                        return;
                    }
                }
            }
        });

        Ok(rx)
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
/// use llm_agent_runtime::providers::OpenAiProvider;
/// let provider = OpenAiProvider::new("sk-...");
/// // For Azure or custom endpoints:
/// let custom = OpenAiProvider::with_base_url("sk-...", "https://my-endpoint/v1");
/// ```
pub struct OpenAiProvider {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
    /// Semaphore bounding concurrent SSE background tasks (item 10).
    stream_semaphore: std::sync::Arc<tokio::sync::Semaphore>,
}

#[cfg(feature = "openai")]
impl OpenAiProvider {
    const DEFAULT_BASE_URL: &'static str = "https://api.openai.com/v1";
    /// Default maximum number of concurrent streaming tasks.
    const DEFAULT_STREAM_CONCURRENCY: usize = 32;

    /// Create a new OpenAI provider with the default base URL.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: Self::DEFAULT_BASE_URL.to_owned(),
            client: reqwest::Client::new(),
            stream_semaphore: std::sync::Arc::new(tokio::sync::Semaphore::new(
                Self::DEFAULT_STREAM_CONCURRENCY,
            )),
        }
    }

    /// Create a provider pointing to a custom base URL (e.g. Azure, local models).
    pub fn with_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: base_url.into(),
            client: reqwest::Client::new(),
            stream_semaphore: std::sync::Arc::new(tokio::sync::Semaphore::new(
                Self::DEFAULT_STREAM_CONCURRENCY,
            )),
        }
    }

    /// Create a provider with a custom limit on concurrent streaming tasks.
    pub fn with_max_concurrent_streams(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        max: usize,
    ) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: base_url.into(),
            client: reqwest::Client::new(),
            stream_semaphore: std::sync::Arc::new(tokio::sync::Semaphore::new(max)),
        }
    }
}

#[cfg(feature = "openai")]
#[async_trait]
impl LlmProvider for OpenAiProvider {
    #[tracing::instrument(skip(self, prompt), fields(model, provider = "openai"))]
    async fn complete(&self, prompt: &str, model: &str) -> Result<String, AgentRuntimeError> {
        self.complete_with_options(prompt, CompletionOptions::new(model))
            .await
    }

    /// Complete with per-request options (max_tokens, temperature, timeout).
    ///
    /// Honours `options.max_tokens` and `options.temperature` when set, falling
    /// back to the API default when unset.
    #[tracing::instrument(skip(self, prompt, options), fields(model = options.model, provider = "openai"))]
    async fn complete_with_options(
        &self,
        prompt: &str,
        options: CompletionOptions<'_>,
    ) -> Result<String, AgentRuntimeError> {
        let url = format!("{}/chat/completions", self.base_url);
        let mut body = serde_json::json!({
            "model": options.model,
            "messages": [{ "role": "user", "content": prompt }]
        });
        if let Some(max_tokens) = options.max_tokens {
            body["max_tokens"] = serde_json::json!(max_tokens);
        }
        if let Some(temp) = options.temperature {
            body["temperature"] = serde_json::json!(temp);
        }

        let mut req = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .header("content-type", "application/json")
            .json(&body);
        if let Some(timeout) = options.timeout {
            req = req.timeout(timeout);
        }
        let response = req
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

    async fn stream_complete(
        &self,
        prompt: &str,
        model: &str,
    ) -> Result<tokio::sync::mpsc::Receiver<Result<String, AgentRuntimeError>>, AgentRuntimeError>
    {
        let url = format!("{}/chat/completions", self.base_url);
        let body = serde_json::json!({
            "model": model,
            "stream": true,
            "messages": [{ "role": "user", "content": prompt }]
        });

        let mut response = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                AgentRuntimeError::Provider(format!("OpenAI stream request failed: {e}"))
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(AgentRuntimeError::Provider(format!(
                "OpenAI stream API error {status}: {text}"
            )));
        }

        let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, AgentRuntimeError>>(32);

        // Item 10 — bound concurrent tasks via semaphore.
        let permit = std::sync::Arc::clone(&self.stream_semaphore)
            .acquire_owned()
            .await
            .map_err(|_| {
                AgentRuntimeError::Provider("OpenAI stream semaphore closed".into())
            })?;

        tokio::spawn(async move {
            let _permit = permit;
            // Incremental chunk-by-chunk reading — avoids buffering the full
            // response body before emitting any tokens.
            let mut buffer = String::new();
            loop {
                match response.chunk().await {
                    Ok(Some(chunk)) => {
                        let text = match std::str::from_utf8(&chunk) {
                            Ok(t) => t,
                            Err(e) => {
                                let _ = tx
                                    .send(Err(AgentRuntimeError::Provider(format!(
                                        "OpenAI stream chunk is not valid UTF-8: {e}"
                                    ))))
                                    .await;
                                return;
                            }
                        };
                        buffer.push_str(text);
                        // Drain complete SSE lines from the buffer.
                        while let Some(newline) = buffer.find('\n') {
                            let line: String = buffer.drain(..=newline).collect();
                            let line = line.trim_end_matches(['\r', '\n']);
                            if let Some(data) = line.strip_prefix("data: ") {
                                if data == "[DONE]" {
                                    return;
                                }
                                if let Ok(json) =
                                    serde_json::from_str::<serde_json::Value>(data)
                                {
                                    if let Some(content) = json["choices"]
                                        .as_array()
                                        .and_then(|c| c.first())
                                        .and_then(|c| c["delta"]["content"].as_str())
                                    {
                                        if tx.send(Ok(content.to_owned())).await.is_err() {
                                            return;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Ok(None) => break,
                    Err(e) => {
                        let _ = tx
                            .send(Err(AgentRuntimeError::Provider(format!(
                                "OpenAI stream read failed: {e}"
                            ))))
                            .await;
                        return;
                    }
                }
            }
        });

        Ok(rx)
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
        // Uses default stream_complete implementation which wraps complete().
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

    #[tokio::test]
    async fn test_stub_provider_stream_returns_single_chunk() {
        let p = StubProvider {
            response: "hello world".into(),
        };
        let mut rx = p.stream_complete("prompt", "model").await.unwrap();
        let mut collected = String::new();
        while let Some(chunk) = rx.recv().await {
            collected.push_str(&chunk.unwrap());
        }
        assert_eq!(collected, "hello world");
    }

    #[tokio::test]
    async fn test_stream_receiver_closes_after_completion() {
        let p = StubProvider {
            response: "done".into(),
        };
        let mut rx = p.stream_complete("prompt", "model").await.unwrap();
        // Drain all chunks
        while let Some(_chunk) = rx.recv().await {}
        // Channel should now be closed — next recv returns None
        assert!(rx.recv().await.is_none());
    }
}

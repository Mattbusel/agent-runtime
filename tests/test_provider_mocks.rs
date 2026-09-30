//! Integration tests for built-in LLM providers using a local mock HTTP server.
//!
//! These tests verify request serialization and response deserialization for
//! `AnthropicProvider` and `OpenAiProvider` without making live API calls.
//! The wiremock crate is used to intercept HTTP traffic.

#[cfg(all(feature = "anthropic", feature = "openai"))]
mod provider_mock_tests {
    use llm_agent_runtime::providers::{AnthropicProvider, LlmProvider, OpenAiProvider};
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    // ── AnthropicProvider ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn anthropic_complete_sends_correct_request_and_parses_response() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("x-api-key", "test-key"))
            .and(header("anthropic-version", "2023-06-01"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "content": [{ "type": "text", "text": "Hello from mock Anthropic!" }],
                "model": "claude-sonnet-4-6",
                "role": "assistant"
            })))
            .mount(&server)
            .await;

        let provider = AnthropicProvider::with_base_url(
            "test-key",
            format!("{}/v1/messages", server.uri()),
        );

        let result = provider
            .complete("Say hello", "claude-sonnet-4-6")
            .await
            .expect("complete() should succeed");

        assert_eq!(result, "Hello from mock Anthropic!");
    }

    #[tokio::test]
    async fn anthropic_complete_returns_error_on_non_200() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(401).set_body_string("Unauthorized"))
            .mount(&server)
            .await;

        let provider = AnthropicProvider::with_base_url(
            "bad-key",
            format!("{}/v1/messages", server.uri()),
        );

        let result = provider.complete("test", "claude-sonnet-4-6").await;
        assert!(result.is_err(), "should fail on 401");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("401"), "error should mention status code: {msg}");
    }

    #[tokio::test]
    async fn anthropic_complete_returns_error_when_content_missing() {
        let server = MockServer::start().await;

        // Response has no `content` array
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "model": "claude-sonnet-4-6"
            })))
            .mount(&server)
            .await;

        let provider = AnthropicProvider::with_base_url(
            "test-key",
            format!("{}/v1/messages", server.uri()),
        );

        let result = provider.complete("test", "claude-sonnet-4-6").await;
        assert!(result.is_err(), "should fail when content array is absent");
    }

    // ── OpenAiProvider ────────────────────────────────────────────────────────

    #[tokio::test]
    async fn openai_complete_sends_correct_request_and_parses_response() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", "Bearer test-openai-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{
                    "message": { "role": "assistant", "content": "Hello from mock OpenAI!" },
                    "finish_reason": "stop"
                }]
            })))
            .mount(&server)
            .await;

        let provider = OpenAiProvider::with_base_url("test-openai-key", server.uri());

        let result = provider
            .complete("Say hello", "gpt-4o")
            .await
            .expect("complete() should succeed");

        assert_eq!(result, "Hello from mock OpenAI!");
    }

    #[tokio::test]
    async fn openai_complete_returns_error_on_non_200() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
            .mount(&server)
            .await;

        let provider = OpenAiProvider::with_base_url("test-key", server.uri());

        let result = provider.complete("test", "gpt-4o").await;
        assert!(result.is_err(), "should fail on 429");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("429"), "error should mention status code: {msg}");
    }

    #[tokio::test]
    async fn openai_complete_returns_error_when_choices_empty() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": []
            })))
            .mount(&server)
            .await;

        let provider = OpenAiProvider::with_base_url("test-key", server.uri());

        let result = provider.complete("test", "gpt-4o").await;
        assert!(result.is_err(), "should fail when choices array is empty");
    }
}

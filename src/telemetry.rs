//! # Module: Telemetry
//!
//! ## Responsibility
//! Provides OpenTelemetry integration for ReAct loop tool call tracing.
//!
//! Every tool call that goes through [`ToolRegistry::call`](crate::agent::ToolRegistry::call)
//! (the text ReAct loop, [`run_native`](crate::agent::ReActLoop::run_native)
//! and direct registry calls) creates an OTel span named
//! `agent.tool_call.{tool_name}` with attributes covering:
//! - `agent.tool.name` — the tool identifier
//! - `agent.tool.input_bytes` — byte length of the serialized input
//! - `agent.tool.output_bytes` — byte length of the serialized output
//! - `agent.tool.success` — `true` / `false`
//! - `agent.tool.error` — error message when `success = false`
//!
//! Spans go to whatever tracer provider is installed globally; call
//! [`init_otlp_tracer`] to send them to an OTLP collector (Jaeger, Tempo,
//! Honeycomb, Grafana, ...).
//!
//! ## Feature Gate
//! This module is only compiled when the `otel` feature is enabled.

use opentelemetry::{
    global::{self, BoxedSpan},
    trace::{Span, SpanKind, Status, Tracer},
    KeyValue,
};
use serde_json::Value;

/// A guard that holds an active OTel span for a single tool call.
///
/// Drop the guard to end the span.  Use [`ToolCallSpan::set_result`] before
/// dropping to record success/failure attributes.
pub struct ToolCallSpan {
    span: BoxedSpan,
}

impl std::fmt::Debug for ToolCallSpan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolCallSpan").finish_non_exhaustive()
    }
}

impl ToolCallSpan {
    /// Start a new span for a tool call.
    ///
    /// # Arguments
    /// * `tool_name` — the name of the tool being invoked
    /// * `input` — the JSON arguments passed to the tool
    ///
    /// The span is named `agent.tool_call.{tool_name}` and tagged with
    /// `agent.tool.input_bytes`.
    pub fn start(tool_name: &str, input: &Value) -> Self {
        let tracer = global::tracer("agent-runtime");
        let span_name = format!("agent.tool_call.{tool_name}");
        let input_bytes = input.to_string().len() as i64;
        let span = tracer
            .span_builder(span_name)
            .with_kind(SpanKind::Internal)
            .with_attributes(vec![
                KeyValue::new("agent.tool.name", tool_name.to_owned()),
                KeyValue::new("agent.tool.input_bytes", input_bytes),
            ])
            .start(&tracer);
        Self { span }
    }

    /// Record the tool result on the span and mark it as succeeded or failed.
    ///
    /// Call this before dropping the guard.  If not called, the span ends
    /// without explicit success/failure attributes.
    ///
    /// # Arguments
    /// * `output` — the JSON value returned by the tool
    /// * `success` — `true` if the tool succeeded, `false` if it failed
    /// * `error_msg` — optional error description when `success = false`
    pub fn set_result(&mut self, output: &Value, success: bool, error_msg: Option<&str>) {
        let output_bytes = output.to_string().len() as i64;
        self.span
            .set_attribute(KeyValue::new("agent.tool.output_bytes", output_bytes));
        self.span
            .set_attribute(KeyValue::new("agent.tool.success", success));
        if let Some(msg) = error_msg {
            self.span
                .set_attribute(KeyValue::new("agent.tool.error", msg.to_owned()));
            self.span.set_status(Status::error(msg.to_owned()));
        } else {
            self.span.set_status(Status::Ok);
        }
    }

    /// End the span explicitly (also called automatically on `Drop`).
    pub fn end(mut self) {
        self.span.end();
    }
}

impl Drop for ToolCallSpan {
    fn drop(&mut self) {
        self.span.end();
    }
}

// ── OtelTracer ────────────────────────────────────────────────────────────────

/// A lightweight facade for creating tool-call spans in the ReAct loop.
///
/// Construct one per agent session or share across sessions — the underlying
/// OTel global tracer is a process-wide singleton.
///
/// # Example
/// ```rust,no_run
/// # #[cfg(feature = "otel")]
/// # {
/// use llm_agent_runtime::telemetry::OtelTracer;
/// use serde_json::json;
///
/// let tracer = OtelTracer::new();
/// let mut span = tracer.start_tool_call("search", &json!({"query": "Rust"}));
/// // ... invoke the tool ...
/// span.set_result(&json!({"results": []}), true, None);
/// span.end();
/// # }
/// ```
#[derive(Debug, Clone, Default)]
pub struct OtelTracer;

impl OtelTracer {
    /// Create a new `OtelTracer` facade.
    pub fn new() -> Self {
        Self
    }

    /// Start a [`ToolCallSpan`] for the named tool with the given input args.
    pub fn start_tool_call(&self, tool_name: &str, input: &Value) -> ToolCallSpan {
        ToolCallSpan::start(tool_name, input)
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Send spans to an OTLP collector over gRPC, e.g.
/// `init_otlp_tracer("http://localhost:4317")` for a local Jaeger or
/// OpenTelemetry Collector. Spans are exported in batches on the Tokio
/// runtime; call `opentelemetry::global::shutdown_tracer_provider()` before
/// exit to flush the last batch.
///
/// Must be called inside a Tokio runtime.
///
/// # Errors
/// Returns the exporter's error if the endpoint is invalid.
pub fn init_otlp_tracer(endpoint: &str) -> Result<(), String> {
    use opentelemetry_otlp::WithExportConfig;

    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .build()
        .map_err(|e| e.to_string())?;
    let provider = opentelemetry_sdk::trace::TracerProvider::builder()
        .with_batch_exporter(exporter, opentelemetry_sdk::runtime::Tokio)
        .build();
    global::set_tracer_provider(provider);
    Ok(())
}

/// Install an SDK tracer provider with no exporter: spans are recorded and
/// then dropped. Despite the name, nothing is printed.
///
/// # Errors
/// Never fails; the `Result` is kept for compatibility.
#[deprecated(
    since = "1.76.0",
    note = "installs no exporter and prints nothing; use init_otlp_tracer, or install your own provider"
)]
pub fn init_stdout_tracer() -> Result<(), String> {
    use opentelemetry_sdk::trace::TracerProvider;

    global::set_tracer_provider(TracerProvider::builder().build());
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::agent::{ToolRegistry, ToolSpec};
    use opentelemetry::trace::Status;
    use opentelemetry_sdk::testing::trace::InMemorySpanExporter;
    use opentelemetry_sdk::trace::TracerProvider;

    #[tokio::test]
    async fn every_registry_call_emits_a_tool_span() {
        let exporter = InMemorySpanExporter::default();
        let provider = TracerProvider::builder()
            .with_simple_exporter(exporter.clone())
            .build();
        opentelemetry::global::set_tracer_provider(provider);

        let registry = ToolRegistry::new()
            .with_tool(ToolSpec::new("echo", "Echo", |args| args))
            .with_tool(ToolSpec::new("needs_q", "Needs q", |args| args).with_required_fields(["q"]));
        registry.call("echo", serde_json::json!({ "x": 1 })).await.unwrap();
        assert!(registry.call("needs_q", serde_json::json!({})).await.is_err());

        let spans = exporter.get_finished_spans().unwrap();
        let ok = spans.iter().find(|s| s.name == "agent.tool_call.echo").expect("echo span");
        assert_eq!(ok.status, Status::Ok);
        let failed = spans.iter().find(|s| s.name == "agent.tool_call.needs_q").expect("needs_q span");
        assert!(matches!(failed.status, Status::Error { .. }));
        assert!(failed.attributes.iter().any(|kv| kv.key.as_str() == "agent.tool.success"
            && kv.value == opentelemetry::Value::Bool(false)));
    }
}

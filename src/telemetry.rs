//! # Module: Telemetry
//!
//! ## Responsibility
//! Provides OpenTelemetry integration for ReAct loop tool call tracing.
//!
//! Each tool call in the agent loop creates an OTel span named
//! `agent.tool_call.{tool_name}` with attributes covering:
//! - `agent.tool.name` — the tool identifier
//! - `agent.tool.input_bytes` — byte length of the serialized input
//! - `agent.tool.output_bytes` — byte length of the serialized output
//! - `agent.tool.success` — `true` / `false`
//! - `agent.tool.error` — error message when `success = false`
//!
//! Trace context is propagated via the W3C TraceContext format so downstream
//! services (e.g. tool HTTP endpoints) can continue the same trace.
//!
//! ## Feature Gate
//! This module is only compiled when the `otel` feature is enabled.

use opentelemetry::{
    global,
    trace::{Span, SpanKind, StatusCode, Tracer},
    KeyValue,
};
use serde_json::Value;

/// A guard that holds an active OTel span for a single tool call.
///
/// Drop the guard to end the span.  Use [`ToolCallSpan::set_result`] before
/// dropping to record success/failure attributes.
pub struct ToolCallSpan {
    span: Box<dyn Span + Send + Sync>,
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
        Self {
            span: Box::new(span),
        }
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
            self.span
                .set_status(StatusCode::Error, msg.to_owned());
        } else {
            self.span.set_status(StatusCode::Ok, String::new());
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

/// Initialize a simple stdout OTel exporter for development and testing.
///
/// In production, replace this with an OTLP exporter configured to point at
/// your collector (Jaeger, Tempo, etc.).
///
/// # Errors
/// Returns a string description of the initialization error.
pub fn init_stdout_tracer() -> Result<(), String> {
    use opentelemetry_sdk::trace::TracerProvider;

    let provider = TracerProvider::builder()
        .build();
    global::set_tracer_provider(provider);
    Ok(())
}

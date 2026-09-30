//! # Distributed Tracing Context Propagation
//!
//! ## Responsibility
//! Generate, propagate, and inject/extract W3C Trace Context headers for
//! distributed tracing across async tasks.
//!
//! ## W3C Trace Context
//! Headers follow the [W3C Trace Context Level 1](https://www.w3.org/TR/trace-context/) spec:
//! - `traceparent: 00-<trace-id>-<span-id>-<flags>`
//! - `tracestate:  <baggage pairs>`
//!
//! ## Task-local storage
//! [`ContextPropagator`] uses a `tokio::task_local!` to store the current
//! context for a task without explicit threading through every call frame.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// ── Atomic ID sources ─────────────────────────────────────────────────────────

static TRACE_HI: AtomicU64 = AtomicU64::new(1);
static TRACE_LO: AtomicU64 = AtomicU64::new(1);
static SPAN_CTR: AtomicU64 = AtomicU64::new(1);

// ── TraceId ───────────────────────────────────────────────────────────────────

/// A 128-bit distributed trace identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TraceId([u8; 16]);

impl TraceId {
    /// Generate a new unique trace ID from a pair of monotonically-incrementing
    /// [`AtomicU64`] counters.
    pub fn new() -> Self {
        let hi = TRACE_HI.fetch_add(1, Ordering::Relaxed);
        let lo = TRACE_LO.fetch_add(7919, Ordering::Relaxed); // coprime step for spread
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&hi.to_le_bytes());
        bytes[8..].copy_from_slice(&lo.to_le_bytes());
        Self(bytes)
    }

    /// Parse from a 32-character lowercase hex string.
    pub fn from_hex(s: &str) -> Option<Self> {
        if s.len() != 32 {
            return None;
        }
        let mut bytes = [0u8; 16];
        for (i, chunk) in s.as_bytes().chunks(2).enumerate() {
            let hi = hex_digit(chunk[0])?;
            let lo = hex_digit(chunk[1])?;
            bytes[i] = (hi << 4) | lo;
        }
        Some(Self(bytes))
    }

    /// Raw bytes.
    pub fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }
}

impl Default for TraceId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for TraceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for b in &self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

// ── SpanId ────────────────────────────────────────────────────────────────────

/// A 64-bit span identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpanId([u8; 8]);

impl SpanId {
    /// Generate a new unique span ID.
    pub fn new() -> Self {
        let v = SPAN_CTR.fetch_add(1, Ordering::Relaxed);
        Self(v.to_le_bytes())
    }

    /// Parse from a 16-character lowercase hex string.
    pub fn from_hex(s: &str) -> Option<Self> {
        if s.len() != 16 {
            return None;
        }
        let mut bytes = [0u8; 8];
        for (i, chunk) in s.as_bytes().chunks(2).enumerate() {
            let hi = hex_digit(chunk[0])?;
            let lo = hex_digit(chunk[1])?;
            bytes[i] = (hi << 4) | lo;
        }
        Some(Self(bytes))
    }

    /// Raw bytes.
    pub fn as_bytes(&self) -> &[u8; 8] {
        &self.0
    }
}

impl Default for SpanId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SpanId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for b in &self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn hex_digit(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

// ── TracingContext ────────────────────────────────────────────────────────────

/// A propagated tracing context for one logical unit of work.
#[derive(Debug, Clone)]
pub struct TracingContext {
    /// Unique identifier for the distributed trace this span belongs to.
    pub trace_id: TraceId,
    /// Unique identifier for this span.
    pub span_id: SpanId,
    /// The span that caused this span to be created, if any.
    pub parent_span_id: Option<SpanId>,
    /// Key-value metadata propagated across service boundaries.
    pub baggage: HashMap<String, String>,
    /// Whether this trace should be recorded/exported.
    pub sampled: bool,
}

impl TracingContext {
    /// Create a new root context (no parent, new trace ID).
    pub fn new_root() -> Self {
        Self {
            trace_id: TraceId::new(),
            span_id: SpanId::new(),
            parent_span_id: None,
            baggage: HashMap::new(),
            sampled: true,
        }
    }

    /// Derive a child context: inherits `trace_id` and `baggage`, gets a new
    /// `span_id`, and sets `parent_span_id` to this context's `span_id`.
    pub fn child_context(&self) -> Self {
        Self {
            trace_id: self.trace_id,
            span_id: SpanId::new(),
            parent_span_id: Some(self.span_id),
            baggage: self.baggage.clone(),
            sampled: self.sampled,
        }
    }

    /// Serialise to W3C Trace Context HTTP headers.
    ///
    /// | Header | Value |
    /// |--------|-------|
    /// | `traceparent` | `00-{trace_id}-{span_id}-{flags}` |
    /// | `tracestate`  | comma-separated `k=v` baggage pairs |
    pub fn inject_headers(&self) -> HashMap<String, String> {
        let flags: u8 = if self.sampled { 0x01 } else { 0x00 };
        let traceparent = format!("00-{}-{}-{flags:02x}", self.trace_id, self.span_id);
        let mut headers = HashMap::new();
        headers.insert("traceparent".to_owned(), traceparent);
        if !self.baggage.is_empty() {
            let tracestate = self
                .baggage
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(",");
            headers.insert("tracestate".to_owned(), tracestate);
        }
        headers
    }

    /// Parse a [`TracingContext`] from W3C Trace Context HTTP headers.
    ///
    /// Returns `None` if the `traceparent` header is absent or malformed.
    pub fn extract_headers(headers: &HashMap<String, String>) -> Option<Self> {
        let tp = headers.get("traceparent")?;
        let parts: Vec<&str> = tp.splitn(4, '-').collect();
        if parts.len() != 4 {
            return None;
        }
        // parts[0] = version, parts[1] = trace-id, parts[2] = parent-id, parts[3] = flags
        let trace_id = TraceId::from_hex(parts[1])?;
        let span_id = SpanId::from_hex(parts[2])?;
        let flags = u8::from_str_radix(parts[3], 16).ok()?;
        let sampled = (flags & 0x01) != 0;

        let mut baggage = HashMap::new();
        if let Some(ts) = headers.get("tracestate") {
            for pair in ts.split(',') {
                if let Some((k, v)) = pair.split_once('=') {
                    baggage.insert(k.trim().to_owned(), v.trim().to_owned());
                }
            }
        }

        Some(Self {
            trace_id,
            span_id, // The extracted span-id becomes our parent; we keep it as span_id for the root of local processing.
            parent_span_id: None,
            baggage,
            sampled,
        })
    }

    /// Set a baggage key-value pair.
    pub fn set_baggage(&mut self, key: &str, value: &str) {
        self.baggage.insert(key.to_owned(), value.to_owned());
    }

    /// Retrieve a baggage value by key.
    pub fn get_baggage(&self, key: &str) -> Option<&str> {
        self.baggage.get(key).map(|s| s.as_str())
    }
}

// ── ContextPropagator ─────────────────────────────────────────────────────────

tokio::task_local! {
    static CURRENT_CTX: TracingContext;
}

/// Utilities for propagating tracing context across async task boundaries.
pub struct ContextPropagator;

impl ContextPropagator {
    /// Create a child [`TracingContext`] for a spawned task.
    ///
    /// The child inherits the same `trace_id` and `baggage` but gets a fresh
    /// `span_id`, with `parent_span_id` pointing to the parent span.
    pub fn propagate_to_task(ctx: &TracingContext) -> TracingContext {
        ctx.child_context()
    }

    /// Retrieve the current task's [`TracingContext`], if one has been set via
    /// [`CURRENT_CTX.scope`](tokio::task::LocalKey::scope).
    pub fn current_context() -> Option<TracingContext> {
        CURRENT_CTX.try_with(|ctx| ctx.clone()).ok()
    }

    /// Run `fut` with `ctx` as the task-local tracing context.
    pub async fn with_context<F, T>(ctx: TracingContext, fut: F) -> T
    where
        F: std::future::Future<Output = T>,
    {
        CURRENT_CTX.scope(ctx, fut).await
    }
}

// ── TimedSpan ────────────────────────────────────────────────────────────────

/// A span in progress: captures a child context and records its start time.
pub struct TimedSpan {
    /// The tracing context for this span.
    pub ctx: TracingContext,
    /// Human-readable span name.
    pub name: String,
    start: Instant,
}

impl TimedSpan {
    /// Begin a new span as a child of `ctx`.
    pub fn start(ctx: &TracingContext, name: &str) -> Self {
        Self {
            ctx: ctx.child_context(),
            name: name.to_owned(),
            start: Instant::now(),
        }
    }

    /// End the span and produce a [`SpanRecord`].
    pub fn finish(self) -> SpanRecord {
        let duration_ns = self.start.elapsed().as_nanos() as u64;
        SpanRecord {
            name: self.name,
            trace_id: self.ctx.trace_id,
            span_id: self.ctx.span_id,
            parent_span_id: self.ctx.parent_span_id,
            duration_ns,
            baggage: self.ctx.baggage,
        }
    }
}

// ── SpanRecord ────────────────────────────────────────────────────────────────

/// An immutable record of a completed span.
#[derive(Debug, Clone)]
pub struct SpanRecord {
    /// Human-readable name for this span.
    pub name: String,
    /// The trace this span belongs to.
    pub trace_id: TraceId,
    /// This span's unique ID.
    pub span_id: SpanId,
    /// The parent span, if any.
    pub parent_span_id: Option<SpanId>,
    /// Elapsed time in nanoseconds.
    pub duration_ns: u64,
    /// Baggage carried by this span.
    pub baggage: HashMap<String, String>,
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn trace_id_display_is_hex() {
        let id = TraceId::new();
        let s = id.to_string();
        assert_eq!(s.len(), 32);
        assert!(s.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn span_id_display_is_hex() {
        let id = SpanId::new();
        let s = id.to_string();
        assert_eq!(s.len(), 16);
        assert!(s.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn parent_child_propagation() {
        let root = TracingContext::new_root();
        let child = root.child_context();

        assert_eq!(root.trace_id, child.trace_id, "trace_id must be inherited");
        assert_ne!(root.span_id, child.span_id, "child gets a new span_id");
        assert_eq!(child.parent_span_id, Some(root.span_id));
    }

    #[test]
    fn baggage_propagation() {
        let mut root = TracingContext::new_root();
        root.set_baggage("user", "alice");
        root.set_baggage("region", "us-east-1");

        let child = root.child_context();
        assert_eq!(child.get_baggage("user"), Some("alice"));
        assert_eq!(child.get_baggage("region"), Some("us-east-1"));
    }

    #[test]
    fn w3c_header_injection_and_extraction() {
        let mut ctx = TracingContext::new_root();
        ctx.set_baggage("env", "prod");

        let headers = ctx.inject_headers();
        assert!(headers.contains_key("traceparent"));
        assert!(headers.contains_key("tracestate"));

        let tp = headers.get("traceparent").unwrap();
        assert!(tp.starts_with("00-"), "must start with version '00-'");

        let recovered = TracingContext::extract_headers(&headers).unwrap();
        assert_eq!(recovered.trace_id, ctx.trace_id);
        assert_eq!(recovered.sampled, ctx.sampled);
        assert_eq!(recovered.get_baggage("env"), Some("prod"));
    }

    #[test]
    fn extract_headers_missing_returns_none() {
        let empty: HashMap<String, String> = HashMap::new();
        assert!(TracingContext::extract_headers(&empty).is_none());
    }

    #[test]
    fn extract_headers_malformed_returns_none() {
        let mut bad = HashMap::new();
        bad.insert("traceparent".to_owned(), "not-a-valid-header".to_owned());
        assert!(TracingContext::extract_headers(&bad).is_none());
    }

    #[test]
    fn timed_span_produces_record() {
        let root = TracingContext::new_root();
        let span = TimedSpan::start(&root, "my-operation");
        let record = span.finish();
        assert_eq!(record.name, "my-operation");
        assert_eq!(record.trace_id, root.trace_id);
        assert_eq!(record.parent_span_id, Some(root.span_id));
    }

    #[tokio::test]
    async fn propagator_task_local() {
        let ctx = TracingContext::new_root();
        let trace_id = ctx.trace_id;
        ContextPropagator::with_context(ctx, async move {
            let current = ContextPropagator::current_context().unwrap();
            assert_eq!(current.trace_id, trace_id);
        })
        .await;
    }

    #[tokio::test]
    async fn propagate_to_task_gives_child() {
        let parent = TracingContext::new_root();
        let child = ContextPropagator::propagate_to_task(&parent);
        assert_eq!(parent.trace_id, child.trace_id);
        assert_eq!(child.parent_span_id, Some(parent.span_id));
    }
}

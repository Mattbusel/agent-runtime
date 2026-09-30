//! Distributed tracing spans and metrics aggregation.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::Instant;

/// W3C-style trace/span context passed between services.
#[derive(Clone)]
pub struct SpanContext {
    /// Globally unique trace identifier.
    pub trace_id: u64,
    /// Unique span identifier within the trace.
    pub span_id: u64,
    /// Parent span, if any.
    pub parent_span_id: Option<u64>,
    /// Arbitrary key-value metadata propagated with the trace.
    pub baggage: HashMap<String, String>,
}

/// Status of a finished span.
#[derive(Clone, Debug)]
pub enum SpanStatus {
    /// Completed successfully.
    Ok,
    /// Completed with an error.
    Error {
        /// Error description.
        message: String,
    },
    /// Timed out before completion.
    Timeout,
}

/// A single timing record within a trace.
pub struct Span {
    /// Span context (IDs + baggage).
    pub context: SpanContext,
    /// Operation name.
    pub name: String,
    /// Wall-clock start.
    pub start_time: Instant,
    /// Wall-clock end (None if still open).
    pub end_time: Option<Instant>,
    /// Arbitrary key-value tags.
    pub tags: HashMap<String, String>,
    /// Timestamped log messages emitted during the span.
    pub logs: Vec<(Instant, String)>,
    /// Final status.
    pub status: SpanStatus,
}

impl Span {
    /// Start a new span with the given name and context.
    pub fn start(name: impl Into<String>, context: SpanContext) -> Self {
        Self {
            context,
            name: name.into(),
            start_time: Instant::now(),
            end_time: None,
            tags: HashMap::new(),
            logs: Vec::new(),
            status: SpanStatus::Ok,
        }
    }

    /// Mark the span as finished successfully.
    pub fn finish(&mut self) {
        self.end_time = Some(Instant::now());
        self.status = SpanStatus::Ok;
    }

    /// Mark the span as finished with an error.
    pub fn finish_with_error(&mut self, msg: impl Into<String>) {
        self.end_time = Some(Instant::now());
        self.status = SpanStatus::Error { message: msg.into() };
    }

    /// Duration in milliseconds. Returns 0.0 if not yet finished.
    pub fn duration_ms(&self) -> f64 {
        match self.end_time {
            Some(end) => end.duration_since(self.start_time).as_secs_f64() * 1000.0,
            None => 0.0,
        }
    }

    /// Add a key-value tag.
    pub fn add_tag(&mut self, k: impl Into<String>, v: impl Into<String>) {
        self.tags.insert(k.into(), v.into());
    }

    /// Append a timestamped log entry.
    pub fn add_log(&mut self, msg: impl Into<String>) {
        self.logs.push((Instant::now(), msg.into()));
    }
}

/// A complete distributed trace containing one or more spans.
pub struct Trace {
    /// Unique trace identifier.
    pub trace_id: u64,
    /// All spans in this trace.
    pub spans: Vec<Span>,
    /// Span ID of the root span.
    pub root_span_id: u64,
}

impl Trace {
    /// End-to-end duration of the root span in milliseconds.
    pub fn total_duration_ms(&self) -> f64 {
        self.spans
            .iter()
            .find(|s| s.context.span_id == self.root_span_id)
            .map(|s| s.duration_ms())
            .unwrap_or(0.0)
    }

    /// All spans whose status is Error.
    pub fn error_spans(&self) -> Vec<&Span> {
        self.spans
            .iter()
            .filter(|s| matches!(s.status, SpanStatus::Error { .. }))
            .collect()
    }

    /// Critical path: root span followed by the deepest child chain.
    pub fn critical_path(&self) -> Vec<&Span> {
        let mut path = Vec::new();
        let root = self
            .spans
            .iter()
            .find(|s| s.context.span_id == self.root_span_id);
        let Some(root) = root else { return path };
        path.push(root);

        let mut current_id = self.root_span_id;
        loop {
            // Find the longest-duration direct child of current_id
            let best_child = self
                .spans
                .iter()
                .filter(|s| s.context.parent_span_id == Some(current_id))
                .max_by(|a, b| {
                    a.duration_ms()
                        .partial_cmp(&b.duration_ms())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            match best_child {
                Some(child) => {
                    current_id = child.context.span_id;
                    path.push(child);
                }
                None => break,
            }
        }
        path
    }
}

/// Collects and aggregates distributed traces.
pub struct TelemetryCollector {
    traces: RwLock<HashMap<u64, Trace>>,
    next_id: AtomicU64,
    /// Fraction of traces to retain (0.0–1.0).
    pub sampling_rate: f64,
    /// Maximum number of traces to keep in memory.
    pub max_traces: usize,
}

impl TelemetryCollector {
    /// Create a new collector with the given sampling rate and capacity.
    pub fn new(sampling_rate: f64, max_traces: usize) -> Self {
        Self {
            traces: RwLock::new(HashMap::new()),
            next_id: AtomicU64::new(1),
            sampling_rate,
            max_traces,
        }
    }

    fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Start a new trace with a root span.
    pub fn start_trace(&self, root_span_name: &str) -> SpanContext {
        let trace_id = self.next_id();
        let span_id = self.next_id();
        let ctx = SpanContext {
            trace_id,
            span_id,
            parent_span_id: None,
            baggage: HashMap::new(),
        };

        // Sampling: only record if within sampling_rate
        let sampled = {
            let seed = trace_id % 1000;
            (seed as f64 / 1000.0) < self.sampling_rate
        };

        if sampled {
            let span = Span::start(root_span_name, ctx.clone());
            let trace = Trace {
                trace_id,
                spans: vec![span],
                root_span_id: span_id,
            };
            let mut traces = self.traces.write().unwrap();
            traces.insert(trace_id, trace);
        }
        ctx
    }

    /// Start a child span within an existing trace.
    pub fn start_child_span(&self, parent_ctx: &SpanContext, name: &str) -> SpanContext {
        let span_id = self.next_id();
        let ctx = SpanContext {
            trace_id: parent_ctx.trace_id,
            span_id,
            parent_span_id: Some(parent_ctx.span_id),
            baggage: parent_ctx.baggage.clone(),
        };
        let span = Span::start(name, ctx.clone());
        let mut traces = self.traces.write().unwrap();
        if let Some(trace) = traces.get_mut(&parent_ctx.trace_id) {
            trace.spans.push(span);
        }
        ctx
    }

    /// Finish a span identified by its context.
    pub fn finish_span(&self, ctx: &SpanContext, status: SpanStatus) {
        let mut traces = self.traces.write().unwrap();
        if let Some(trace) = traces.get_mut(&ctx.trace_id) {
            if let Some(span) = trace
                .spans
                .iter_mut()
                .find(|s| s.context.span_id == ctx.span_id)
            {
                span.end_time = Some(Instant::now());
                span.status = status;
            }
        }
    }

    /// Retrieve a clone of a trace by ID.
    pub fn get_trace(&self, trace_id: u64) -> Option<Trace> {
        let traces = self.traces.read().unwrap();
        traces.get(&trace_id).map(|t| {
            // We need to reconstruct since Trace is not Clone (Span contains Instant)
            // Build a new Trace with cloned data
            Trace {
                trace_id: t.trace_id,
                spans: t
                    .spans
                    .iter()
                    .map(|s| Span {
                        context: s.context.clone(),
                        name: s.name.clone(),
                        start_time: s.start_time,
                        end_time: s.end_time,
                        tags: s.tags.clone(),
                        logs: s.logs.clone(),
                        status: s.status.clone(),
                    })
                    .collect(),
                root_span_id: t.root_span_id,
            }
        })
    }

    /// Return the `n` most recently started traces (cloned).
    pub fn recent_traces(&self, n: usize) -> Vec<Trace> {
        let traces = self.traces.read().unwrap();
        let mut ids: Vec<u64> = traces.keys().copied().collect();
        ids.sort_unstable_by(|a, b| b.cmp(a)); // highest ID = most recent
        ids.truncate(n);
        ids.into_iter()
            .filter_map(|id| {
                traces.get(&id).map(|t| Trace {
                    trace_id: t.trace_id,
                    spans: t
                        .spans
                        .iter()
                        .map(|s| Span {
                            context: s.context.clone(),
                            name: s.name.clone(),
                            start_time: s.start_time,
                            end_time: s.end_time,
                            tags: s.tags.clone(),
                            logs: s.logs.clone(),
                            status: s.status.clone(),
                        })
                        .collect(),
                    root_span_id: t.root_span_id,
                })
            })
            .collect()
    }

    /// Compute the p95 latency in ms across all finished root spans.
    pub fn p95_latency_ms(&self) -> f64 {
        let traces = self.traces.read().unwrap();
        let mut durations: Vec<f64> = traces
            .values()
            .filter_map(|t| {
                t.spans
                    .iter()
                    .find(|s| s.context.span_id == t.root_span_id && s.end_time.is_some())
                    .map(|s| s.duration_ms())
            })
            .collect();
        if durations.is_empty() {
            return 0.0;
        }
        durations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((durations.len() as f64 * 0.95) as usize).min(durations.len() - 1);
        durations[idx]
    }

    /// Fraction of finished root spans that have at least one error span.
    pub fn error_rate(&self) -> f64 {
        let traces = self.traces.read().unwrap();
        let finished: Vec<&Trace> = traces
            .values()
            .filter(|t| {
                t.spans
                    .iter()
                    .any(|s| s.context.span_id == t.root_span_id && s.end_time.is_some())
            })
            .collect();
        if finished.is_empty() {
            return 0.0;
        }
        let errors = finished
            .iter()
            .filter(|t| !t.error_spans().is_empty())
            .count();
        errors as f64 / finished.len() as f64
    }

    /// Drop old traces, keeping only the most recent `keep_n`.
    pub fn prune_old_traces(&self, keep_n: usize) {
        let mut traces = self.traces.write().unwrap();
        if traces.len() <= keep_n {
            return;
        }
        let mut ids: Vec<u64> = traces.keys().copied().collect();
        ids.sort_unstable(); // ascending — oldest first
        let to_remove = ids.len() - keep_n;
        for id in ids.into_iter().take(to_remove) {
            traces.remove(&id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_trace_lifecycle() {
        let collector = TelemetryCollector::new(1.0, 100);
        let root_ctx = collector.start_trace("root-op");
        let child_ctx = collector.start_child_span(&root_ctx, "child-op");
        collector.finish_span(&child_ctx, SpanStatus::Ok);
        collector.finish_span(&root_ctx, SpanStatus::Ok);

        let trace = collector.get_trace(root_ctx.trace_id).unwrap();
        assert_eq!(trace.spans.len(), 2);
        assert!(trace.total_duration_ms() >= 0.0);
        assert!(trace.error_spans().is_empty());
    }

    #[test]
    fn error_rate_and_p95() {
        let collector = TelemetryCollector::new(1.0, 100);
        for _ in 0..10 {
            let ctx = collector.start_trace("op");
            collector.finish_span(&ctx, SpanStatus::Ok);
        }
        let ctx = collector.start_trace("op");
        collector.finish_span(&ctx, SpanStatus::Error { message: "boom".into() });

        assert!(collector.error_rate() > 0.0);
        assert!(collector.p95_latency_ms() >= 0.0);
    }
}

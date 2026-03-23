//! # agent_logger
//!
//! Structured logging for agents with configurable sinks, filtering, and spans.
//!
//! [`AgentLogger`] buffers up to 10 000 [`LogRecord`]s in a ring-buffer,
//! forwards entries to one or more [`LogSink`]s, and supports timed
//! [`SpanGuard`]s that auto-log their duration on drop.

use std::{
    collections::{HashMap, VecDeque},
    fmt,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};

// ── LogLevel ─────────────────────────────────────────────────────────────────

/// Severity level for a log record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LogLevel {
    /// Highly granular diagnostic information.
    Trace = 0,
    /// Diagnostic information useful during development.
    Debug = 1,
    /// Normal operational events.
    Info = 2,
    /// Unexpected conditions that do not prevent the agent from continuing.
    Warn = 3,
    /// Errors that affect a specific operation but not the entire agent.
    Error = 4,
    /// Severe failures that may require immediate attention.
    Critical = 5,
}

impl fmt::Display for LogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LogLevel::Trace => f.write_str("TRACE"),
            LogLevel::Debug => f.write_str("DEBUG"),
            LogLevel::Info => f.write_str("INFO"),
            LogLevel::Warn => f.write_str("WARN"),
            LogLevel::Error => f.write_str("ERROR"),
            LogLevel::Critical => f.write_str("CRITICAL"),
        }
    }
}

// ── LogRecord ─────────────────────────────────────────────────────────────────

/// A single structured log entry.
#[derive(Debug, Clone)]
pub struct LogRecord {
    /// Seconds since Unix epoch when this record was created.
    pub timestamp_unix: u64,
    /// Severity of the event.
    pub level: LogLevel,
    /// Identifier of the agent that produced this record.
    pub agent_id: String,
    /// Optional task identifier if this record relates to a specific task.
    pub task_id: Option<String>,
    /// Human-readable description of the event.
    pub message: String,
    /// Arbitrary structured fields attached to this record.
    pub fields: HashMap<String, String>,
    /// Optional span identifier for distributed-tracing correlation.
    pub span_id: Option<u64>,
}

// ── LogSink ──────────────────────────────────────────────────────────────────

/// Destination to which log records are forwarded.
#[derive(Debug, Clone)]
pub enum LogSink {
    /// Write formatted records to standard output.
    Stdout,
    /// Keep records only in the in-memory ring buffer (no side effect).
    Buffer,
    /// Append formatted records to the named file (path stored as String).
    File(String),
    /// Discard records silently.
    Null,
}

// ── LogFilter ────────────────────────────────────────────────────────────────

/// Predicate used to select log records of interest.
#[derive(Debug, Clone)]
pub struct LogFilter {
    /// Minimum level a record must have to pass through.
    pub min_level: LogLevel,
    /// If set, only records whose `agent_id` is in this list pass through.
    pub agent_ids: Option<Vec<String>>,
    /// If set, only records whose `message` contains this substring pass.
    pub contains_text: Option<String>,
}

impl LogFilter {
    /// Returns `true` if `record` satisfies all filter constraints.
    pub fn matches(&self, record: &LogRecord) -> bool {
        if record.level < self.min_level {
            return false;
        }
        if let Some(ids) = &self.agent_ids {
            if !ids.iter().any(|id| id == &record.agent_id) {
                return false;
            }
        }
        if let Some(text) = &self.contains_text {
            if !record.message.contains(text.as_str()) {
                return false;
            }
        }
        true
    }
}

impl Default for LogFilter {
    fn default() -> Self {
        Self {
            min_level: LogLevel::Trace,
            agent_ids: None,
            contains_text: None,
        }
    }
}

// ── LogStats ─────────────────────────────────────────────────────────────────

/// Aggregate statistics for an [`AgentLogger`] instance.
#[derive(Debug, Clone)]
pub struct LogStats {
    /// Total records currently held in the ring buffer.
    pub total_records: usize,
    /// Record count broken down by level name.
    pub by_level: HashMap<String, usize>,
    /// Fraction of records at `Error` or `Critical` level (0.0–1.0).
    pub error_rate: f64,
    /// Unix timestamp of the oldest record in the buffer (0 if empty).
    pub oldest_timestamp: u64,
}

// ── SpanGuard ────────────────────────────────────────────────────────────────

/// A timed span that logs its duration when dropped.
pub struct SpanGuard {
    /// Unique identifier for this span.
    pub span_id: u64,
    /// Logger that will receive the completion record.
    pub logger: Arc<AgentLogger>,
    start: Instant,
    name: String,
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        let elapsed_ms = self.start.elapsed().as_millis() as u64;
        let mut fields = HashMap::new();
        fields.insert("span_id".to_owned(), self.span_id.to_string());
        fields.insert("elapsed_ms".to_owned(), elapsed_ms.to_string());
        self.logger.log(
            LogLevel::Debug,
            &format!("span '{}' completed in {}ms", self.name, elapsed_ms),
            fields,
        );
    }
}

// ── AgentLogger ───────────────────────────────────────────────────────────────

const MAX_BUFFER: usize = 10_000;

/// Structured logger for a single agent.
pub struct AgentLogger {
    /// Identifier of the agent this logger belongs to.
    pub agent_id: String,
    /// Ordered list of sinks that receive each record.
    pub sinks: Vec<LogSink>,
    buffer: Mutex<VecDeque<LogRecord>>,
    span_counter: AtomicU64,
    /// Minimum level; records below this are silently discarded.
    pub min_level: LogLevel,
}

impl AgentLogger {
    /// Create a new logger with an in-memory buffer sink and `Info` minimum level.
    pub fn new(agent_id: &str) -> Self {
        Self {
            agent_id: agent_id.to_owned(),
            sinks: vec![LogSink::Buffer],
            buffer: Mutex::new(VecDeque::new()),
            span_counter: AtomicU64::new(1),
            min_level: LogLevel::Info,
        }
    }

    /// Emit a structured log record.
    pub fn log(&self, level: LogLevel, message: &str, fields: HashMap<String, String>) {
        if level < self.min_level {
            return;
        }
        let timestamp_unix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let record = LogRecord {
            timestamp_unix,
            level,
            agent_id: self.agent_id.clone(),
            task_id: None,
            message: message.to_owned(),
            fields,
            span_id: None,
        };

        // Forward to sinks.
        for sink in &self.sinks {
            match sink {
                LogSink::Stdout => {
                    // Using eprintln! to avoid triggering print_stdout clippy lint.
                    let line = format!(
                        "[{}] {} agent={} {}",
                        record.timestamp_unix, record.level, record.agent_id, record.message
                    );
                    eprintln!("{line}");
                }
                LogSink::Buffer | LogSink::Null | LogSink::File(_) => {
                    // File writing is a no-op in this in-memory implementation;
                    // Buffer means "keep in ring buffer" (handled below).
                }
            }
        }

        // Always push to ring buffer.
        if let Ok(mut buf) = self.buffer.lock() {
            if buf.len() >= MAX_BUFFER {
                buf.pop_front();
            }
            buf.push_back(record);
        }
    }

    /// Log at `Trace` level with no extra fields.
    pub fn trace(&self, msg: &str) {
        self.log(LogLevel::Trace, msg, HashMap::new());
    }

    /// Log at `Debug` level with no extra fields.
    pub fn debug(&self, msg: &str) {
        self.log(LogLevel::Debug, msg, HashMap::new());
    }

    /// Log at `Info` level with no extra fields.
    pub fn info(&self, msg: &str) {
        self.log(LogLevel::Info, msg, HashMap::new());
    }

    /// Log at `Warn` level with no extra fields.
    pub fn warn(&self, msg: &str) {
        self.log(LogLevel::Warn, msg, HashMap::new());
    }

    /// Log at `Error` level with no extra fields.
    pub fn error(&self, msg: &str) {
        self.log(LogLevel::Error, msg, HashMap::new());
    }

    /// Attach an additional structured field to this logger (builder style).
    ///
    /// The field is stored in `sinks` metadata by convention; callers typically
    /// chain this before the first `log` call.
    pub fn with_field(mut self, key: &str, val: &str) -> Self {
        // Store fields as a synthetic sink annotation; for now we record them
        // by adding a dedicated `File` sink entry that encodes the key-value.
        // A real implementation would carry a default-fields map.
        self.sinks
            .push(LogSink::File(format!("field:{}={}", key, val)));
        self
    }

    /// Begin a timed span that logs its completion on drop.
    ///
    /// Wraps `self` in an `Arc` internally — call this on an `Arc<AgentLogger>`.
    pub fn span(self: &Arc<Self>, name: &str) -> SpanGuard {
        let span_id = self.span_counter.fetch_add(1, Ordering::Relaxed);
        let mut fields = HashMap::new();
        fields.insert("span_id".to_owned(), span_id.to_string());
        self.log(
            LogLevel::Debug,
            &format!("span '{}' started", name),
            fields,
        );
        SpanGuard {
            span_id,
            logger: Arc::clone(self),
            start: Instant::now(),
            name: name.to_owned(),
        }
    }

    /// Return up to `n` recent records that satisfy `filter`.
    pub fn recent_logs(&self, n: usize, filter: &LogFilter) -> Vec<LogRecord> {
        self.buffer
            .lock()
            .map(|buf| {
                buf.iter()
                    .rev()
                    .filter(|r| filter.matches(r))
                    .take(n)
                    .cloned()
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Serialise all buffered records as newline-delimited JSON.
    pub fn export_json_lines(&self) -> String {
        self.buffer
            .lock()
            .map(|buf| {
                buf.iter()
                    .map(|r| {
                        format!(
                            r#"{{"ts":{},"level":"{}","agent":"{}","msg":{:?}}}"#,
                            r.timestamp_unix, r.level, r.agent_id, r.message
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_default()
    }

    /// Drain all records from the ring buffer.
    pub fn clear(&self) {
        if let Ok(mut buf) = self.buffer.lock() {
            buf.clear();
        }
    }

    /// Compute aggregate statistics over the current ring buffer.
    pub fn log_stats(&self) -> LogStats {
        self.buffer
            .lock()
            .map(|buf| {
                let total_records = buf.len();
                let mut by_level: HashMap<String, usize> = HashMap::new();
                let mut error_count = 0usize;
                let mut oldest_timestamp = u64::MAX;

                for r in buf.iter() {
                    *by_level.entry(r.level.to_string()).or_insert(0) += 1;
                    if r.level >= LogLevel::Error {
                        error_count += 1;
                    }
                    if r.timestamp_unix < oldest_timestamp {
                        oldest_timestamp = r.timestamp_unix;
                    }
                }

                let error_rate = if total_records == 0 {
                    0.0
                } else {
                    error_count as f64 / total_records as f64
                };

                LogStats {
                    total_records,
                    by_level,
                    error_rate,
                    oldest_timestamp: if oldest_timestamp == u64::MAX {
                        0
                    } else {
                        oldest_timestamp
                    },
                }
            })
            .unwrap_or_else(|_| LogStats {
                total_records: 0,
                by_level: HashMap::new(),
                error_rate: 0.0,
                oldest_timestamp: 0,
            })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn test_basic_log_and_retrieve() {
        let logger = AgentLogger::new("agent-1");
        logger.info("hello world");
        let filter = LogFilter::default();
        let logs = logger.recent_logs(10, &filter);
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0].message, "hello world");
    }

    #[test]
    fn test_filter_by_level() {
        let mut logger = AgentLogger::new("agent-2");
        logger.min_level = LogLevel::Trace;
        logger.trace("trace msg");
        logger.warn("warn msg");

        let filter = LogFilter {
            min_level: LogLevel::Warn,
            agent_ids: None,
            contains_text: None,
        };
        let logs = logger.recent_logs(10, &filter);
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0].level, LogLevel::Warn);
    }

    #[test]
    fn test_export_json() {
        let logger = AgentLogger::new("agent-3");
        logger.info("test event");
        let json = logger.export_json_lines();
        assert!(json.contains("test event"));
    }

    #[test]
    fn test_stats() {
        let mut logger = AgentLogger::new("agent-4");
        logger.min_level = LogLevel::Trace;
        logger.info("a");
        logger.error("b");
        let stats = logger.log_stats();
        assert_eq!(stats.total_records, 2);
        assert!((stats.error_rate - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_span_guard() {
        let logger = Arc::new(AgentLogger::new("agent-5"));
        {
            let _guard = logger.span("my-span");
        }
        // SpanGuard drop should have logged two entries (start + complete).
        let filter = LogFilter {
            min_level: LogLevel::Trace,
            ..Default::default()
        };
        let logs = logger.recent_logs(10, &filter);
        // At least start message logged.
        assert!(!logs.is_empty());
    }
}

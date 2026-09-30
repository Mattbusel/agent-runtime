//! Cron expression parser and job scheduler.
//!
//! Supports standard 5-field cron syntax (minute hour day month weekday)
//! with field types: `*`, specific values, ranges, steps, and `L` (last).

use std::collections::HashMap;
use std::sync::Mutex;

/// Errors produced by cron expression parsing or scheduling operations.
#[derive(Debug, Clone, PartialEq)]
pub enum CronError {
    /// The overall expression is structurally invalid.
    InvalidExpression(String),
    /// A specific field could not be parsed.
    InvalidField(String),
    /// A range was inverted (start > end) or out of bounds.
    InvalidRange,
}

impl std::fmt::Display for CronError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CronError::InvalidExpression(s) => write!(f, "invalid expression: {s}"),
            CronError::InvalidField(s) => write!(f, "invalid field: {s}"),
            CronError::InvalidRange => write!(f, "invalid range"),
        }
    }
}

impl std::error::Error for CronError {}

/// A single parsed field from a cron expression.
#[derive(Debug, Clone, PartialEq)]
pub enum CronField {
    /// `*` — matches every value.
    All,
    /// `1,2,5` — matches any of the listed values.
    Specific(Vec<u32>),
    /// `1-5` — matches every value in the closed range.
    Range(u32, u32),
    /// `*/2` or `1/3` — matches start and every `step` values thereafter.
    Step {
        /// First value in the step sequence.
        start: u32,
        /// Interval between matched values.
        step: u32,
    },
    /// `L` — matches the last valid value for the field (e.g. last day of month).
    Last,
}

impl CronField {
    fn parse(s: &str) -> Result<Self, CronError> {
        if s == "*" {
            return Ok(CronField::All);
        }
        if s == "L" {
            return Ok(CronField::Last);
        }
        // Step: */n or start/n
        if let Some(slash) = s.find('/') {
            let left = &s[..slash];
            let right = &s[slash + 1..];
            let step: u32 = right
                .parse()
                .map_err(|_| CronError::InvalidField(s.to_string()))?;
            if step == 0 {
                return Err(CronError::InvalidField("step cannot be zero".to_string()));
            }
            let start: u32 = if left == "*" {
                0
            } else {
                left.parse()
                    .map_err(|_| CronError::InvalidField(s.to_string()))?
            };
            return Ok(CronField::Step { start, step });
        }
        // Range: a-b
        if let Some(dash) = s.find('-') {
            let left: u32 = s[..dash]
                .parse()
                .map_err(|_| CronError::InvalidField(s.to_string()))?;
            let right: u32 = s[dash + 1..]
                .parse()
                .map_err(|_| CronError::InvalidField(s.to_string()))?;
            if left > right {
                return Err(CronError::InvalidRange);
            }
            return Ok(CronField::Range(left, right));
        }
        // List: a,b,c
        if s.contains(',') {
            let values: Result<Vec<u32>, _> = s.split(',').map(|p| p.parse::<u32>()).collect();
            let values = values.map_err(|_| CronError::InvalidField(s.to_string()))?;
            return Ok(CronField::Specific(values));
        }
        // Single value
        let v: u32 = s
            .parse()
            .map_err(|_| CronError::InvalidField(s.to_string()))?;
        Ok(CronField::Specific(vec![v]))
    }
}

/// A fully parsed 5-field cron expression.
#[derive(Debug, Clone)]
pub struct CronExpression {
    /// Minute field (0-59).
    pub minute: CronField,
    /// Hour field (0-23).
    pub hour: CronField,
    /// Day-of-month field (1-31).
    pub day_of_month: CronField,
    /// Month field (1-12).
    pub month: CronField,
    /// Day-of-week field (0-6, Sunday = 0).
    pub day_of_week: CronField,
}

impl CronExpression {
    /// Parse a standard 5-field cron expression string.
    pub fn parse(expr: &str) -> Result<Self, CronError> {
        let fields: Vec<&str> = expr.split_whitespace().collect();
        if fields.len() != 5 {
            return Err(CronError::InvalidExpression(format!(
                "expected 5 fields, got {}",
                fields.len()
            )));
        }
        Ok(CronExpression {
            minute: CronField::parse(fields[0])?,
            hour: CronField::parse(fields[1])?,
            day_of_month: CronField::parse(fields[2])?,
            month: CronField::parse(fields[3])?,
            day_of_week: CronField::parse(fields[4])?,
        })
    }

    /// Check whether a field matches a value, given the field's maximum valid value.
    pub fn field_matches(field: &CronField, value: u32, max: u32) -> bool {
        match field {
            CronField::All => true,
            CronField::Specific(vals) => vals.contains(&value),
            CronField::Range(lo, hi) => value >= *lo && value <= *hi,
            CronField::Step { start, step } => {
                if value < *start {
                    return false;
                }
                (value - start) % step == 0
            }
            CronField::Last => value == max,
        }
    }

    /// Return true if this expression matches the given Unix timestamp (seconds).
    pub fn matches(&self, unix_ts: u64) -> bool {
        let (min, hour, dom, month, dow) = decompose(unix_ts);
        let days_in_month = days_in(year_of(unix_ts), month);
        Self::field_matches(&self.minute, min, 59)
            && Self::field_matches(&self.hour, hour, 23)
            && Self::field_matches(&self.day_of_month, dom, days_in_month)
            && Self::field_matches(&self.month, month, 12)
            && Self::field_matches(&self.day_of_week, dow, 6)
    }

    /// Return the next Unix timestamp (seconds) at or after `unix_ts` that matches.
    /// Searches forward minute-by-minute up to 4 years to avoid infinite loops.
    pub fn next_after(&self, unix_ts: u64) -> u64 {
        // Start at the next whole minute after unix_ts
        let mut candidate = (unix_ts / 60 + 1) * 60;
        let limit = unix_ts + 4 * 366 * 24 * 3600;
        while candidate < limit {
            if self.matches(candidate) {
                return candidate;
            }
            candidate += 60;
        }
        candidate
    }
}

// ── Calendar helpers ─────────────────────────────────────────────────────────

/// Decompose a Unix timestamp into (minute, hour, day-of-month, month, day-of-week).
fn decompose(ts: u64) -> (u32, u32, u32, u32, u32) {
    let secs = ts % 86400;
    let days = ts / 86400;
    let minute = ((secs % 3600) / 60) as u32;
    let hour = (secs / 3600) as u32;
    // Day of week: Unix epoch (Jan 1 1970) was a Thursday = 4
    let dow = ((days + 4) % 7) as u32;
    // Compute year, month, day from days since epoch
    let (year, month, day) = days_to_ymd(days);
    (minute, hour, day, month, year as u32)
}

fn year_of(ts: u64) -> u32 {
    let (y, _, _) = days_to_ymd(ts / 86400);
    y as u32
}

fn is_leap(year: u64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

fn days_in(year: u32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap(year as u64) {
                29
            } else {
                28
            }
        }
        _ => 30,
    }
}

fn days_to_ymd(days: u64) -> (u64, u32, u32) {
    // Rata Die algorithm
    let mut remaining = days as i64;
    let mut year: i64 = 1970;
    loop {
        let dy = if is_leap(year as u64) { 366 } else { 365 };
        if remaining < dy {
            break;
        }
        remaining -= dy;
        year += 1;
    }
    let months = [31i64, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut month: u32 = 1;
    for &dm in &months {
        let dm = if month == 2 && is_leap(year as u64) {
            29
        } else {
            dm as u32
        };
        if remaining < dm as i64 {
            break;
        }
        remaining -= dm as i64;
        month += 1;
    }
    (year as u64, month, remaining as u32 + 1)
}

// ── CronJob ───────────────────────────────────────────────────────────────────

/// A scheduled job with an associated cron expression.
#[derive(Debug, Clone)]
pub struct CronJob {
    /// Unique identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Parsed schedule expression.
    pub expression: CronExpression,
    /// Unix timestamp of the last run, if any.
    pub last_run: Option<u64>,
    /// Unix timestamp of the next scheduled run.
    pub next_run: u64,
    /// Whether the job is currently active.
    pub enabled: bool,
    /// Total number of times this job has been executed.
    pub run_count: u64,
    /// Arbitrary string key-value metadata.
    pub payload: HashMap<String, String>,
}

/// The outcome of a single job execution.
#[derive(Debug, Clone)]
pub struct JobResult {
    /// ID of the job that produced this result.
    pub job_id: String,
    /// Unix timestamp when execution started.
    pub started_at: u64,
    /// Wall-clock execution duration in milliseconds.
    pub duration_ms: u64,
    /// Whether the execution succeeded.
    pub success: bool,
    /// Captured output text.
    pub output: String,
}

// ── SchedulerStats ────────────────────────────────────────────────────────────

/// Aggregate statistics for the scheduler.
#[derive(Debug, Clone)]
pub struct SchedulerStats {
    /// Total registered jobs.
    pub total_jobs: usize,
    /// Number of enabled jobs.
    pub enabled_jobs: usize,
    /// Total executions across all jobs.
    pub total_runs: u64,
    /// Fraction of successful executions (0.0–1.0).
    pub success_rate: f64,
    /// Seconds until the next due job, if any.
    pub next_due_in_secs: Option<u64>,
}

// ── CronScheduler ─────────────────────────────────────────────────────────────

/// Thread-safe cron job scheduler.
pub struct CronScheduler {
    jobs: Mutex<Vec<CronJob>>,
    history: Mutex<Vec<JobResult>>,
}

impl CronScheduler {
    /// Create an empty scheduler.
    pub fn new() -> Self {
        CronScheduler {
            jobs: Mutex::new(Vec::new()),
            history: Mutex::new(Vec::new()),
        }
    }

    /// Register a new job with the given name, cron expression, and payload.
    ///
    /// Returns the newly generated job ID.
    pub fn add_job(
        &self,
        name: &str,
        expr: &str,
        payload: HashMap<String, String>,
    ) -> Result<String, CronError> {
        let expression = CronExpression::parse(expr)?;
        let id = format!("job-{}", uuid_v4_simple());
        let now = 0u64; // caller should supply current time; we default to epoch
        let next_run = expression.next_after(now);
        let job = CronJob {
            id: id.clone(),
            name: name.to_string(),
            expression,
            last_run: None,
            next_run,
            enabled: true,
            run_count: 0,
            payload,
        };
        let mut jobs = lock_or_default(&self.jobs);
        jobs.push(job);
        Ok(id)
    }

    /// Remove a job by ID. Returns `true` if a job was removed.
    pub fn remove_job(&self, id: &str) -> bool {
        let mut jobs = lock_or_default(&self.jobs);
        let before = jobs.len();
        jobs.retain(|j| j.id != id);
        jobs.len() < before
    }

    /// Enable or disable a job by ID.
    pub fn enable_job(&self, id: &str, enabled: bool) {
        let mut jobs = lock_or_default(&self.jobs);
        for job in jobs.iter_mut() {
            if job.id == id {
                job.enabled = enabled;
            }
        }
    }

    /// Return all enabled jobs whose `next_run` is at or before `now`.
    pub fn due_jobs(&self, now: u64) -> Vec<CronJob> {
        let jobs = lock_or_default(&self.jobs);
        jobs.iter()
            .filter(|j| j.enabled && j.next_run <= now)
            .cloned()
            .collect()
    }

    /// Append an execution result to the history log.
    pub fn record_result(&self, result: JobResult) {
        let mut history = lock_or_default(&self.history);
        history.push(result);
    }

    /// Advance a job's `next_run` and update its `last_run` and `run_count`.
    pub fn update_next_run(&self, id: &str, now: u64) {
        let mut jobs = lock_or_default(&self.jobs);
        for job in jobs.iter_mut() {
            if job.id == id {
                job.last_run = Some(now);
                job.run_count += 1;
                job.next_run = job.expression.next_after(now);
            }
        }
    }

    /// Return up to `limit` most-recent results for a job.
    pub fn job_history(&self, id: &str, limit: usize) -> Vec<JobResult> {
        let history = lock_or_default(&self.history);
        let results: Vec<JobResult> = history
            .iter()
            .filter(|r| r.job_id == id)
            .cloned()
            .collect();
        let start = results.len().saturating_sub(limit);
        results[start..].to_vec()
    }

    /// Return aggregate scheduler statistics.
    pub fn scheduler_stats(&self) -> SchedulerStats {
        let jobs = lock_or_default(&self.jobs);
        let history = lock_or_default(&self.history);

        let total_jobs = jobs.len();
        let enabled_jobs = jobs.iter().filter(|j| j.enabled).count();
        let total_runs: u64 = jobs.iter().map(|j| j.run_count).sum();
        let success_count = history.iter().filter(|r| r.success).count();
        let success_rate = if history.is_empty() {
            1.0
        } else {
            success_count as f64 / history.len() as f64
        };
        let now_proxy: u64 = 0; // scheduler is time-agnostic; caller drives clock
        let next_due_in_secs = jobs
            .iter()
            .filter(|j| j.enabled)
            .map(|j| j.next_run)
            .min()
            .map(|t| t.saturating_sub(now_proxy));

        SchedulerStats {
            total_jobs,
            enabled_jobs,
            total_runs,
            success_rate,
            next_due_in_secs,
        }
    }
}

impl Default for CronScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

fn lock_or_default<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    // In a production setting a poisoned mutex is a programming error; we
    // cannot panic (clippy::panic) so we re-initialise via into_inner.
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Generate a simple pseudo-unique string without depending on the `uuid` crate
/// at the module level (the crate already has it as a dependency, but we keep
/// this module self-contained so it can compile even without the feature flags).
fn uuid_v4_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    format!("{:08x}-{:04x}", nanos, nanos.wrapping_mul(2891336453_u32))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_all_star() {
        let e = CronExpression::parse("* * * * *").expect("parse failed");
        assert!(matches!(e.minute, CronField::All));
    }

    #[test]
    fn parse_step() {
        let e = CronExpression::parse("*/5 * * * *").expect("parse failed");
        assert!(matches!(e.minute, CronField::Step { start: 0, step: 5 }));
    }

    #[test]
    fn matches_every_minute() {
        let e = CronExpression::parse("* * * * *").expect("parse failed");
        // 2024-01-01 00:00:00 UTC = 1704067200
        assert!(e.matches(1_704_067_200));
    }

    #[test]
    fn next_after_advances() {
        let e = CronExpression::parse("0 * * * *").expect("parse failed");
        let ts = 1_704_067_200u64; // exactly on the hour
        let next = e.next_after(ts);
        assert!(next > ts);
    }

    #[test]
    fn scheduler_add_remove() {
        let sched = CronScheduler::new();
        let id = sched
            .add_job("test", "* * * * *", HashMap::new())
            .expect("add failed");
        assert!(sched.remove_job(&id));
        assert!(!sched.remove_job(&id));
    }
}

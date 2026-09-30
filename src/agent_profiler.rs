//! # Module: agent_profiler
//!
//! Profile agent execution for bottleneck detection.
//!
//! Provides:
//! - [`ProfileSample`]: a single measurement snapshot for one agent.
//! - [`ProfileWindow`]: a sliding window of recent samples.
//! - [`AgentProfile`]: per-agent accumulated profile.
//! - [`AgentProfiler`]: global profiler collecting samples across agents.
//! - [`AgentProfileReport`]: summary report for a single agent.
//! - [`GlobalProfileSummary`]: aggregated summary across all agents.

use std::collections::VecDeque;
use std::sync::Arc;
use std::collections::HashMap;

// ── ProfileSample ─────────────────────────────────────────────────────────────

/// A single profiling snapshot for one agent at a point in time.
#[derive(Debug, Clone)]
pub struct ProfileSample {
    /// ID of the agent that produced this sample.
    pub agent_id: String,
    /// Wall-clock timestamp in milliseconds since epoch.
    pub timestamp_ms: u64,
    /// CPU time consumed by this measurement interval, in microseconds.
    pub cpu_time_us: u64,
    /// Memory currently allocated by the agent, in bytes.
    pub memory_bytes: usize,
    /// Number of items currently waiting in the agent's task queue.
    pub queue_depth: usize,
    /// Name of the task being executed at sample time.
    pub task_name: String,
}

// ── ProfileWindow ─────────────────────────────────────────────────────────────

/// A fixed-capacity sliding window of [`ProfileSample`]s.
#[derive(Debug, Clone)]
pub struct ProfileWindow {
    /// Ordered collection of recent samples.
    pub samples: VecDeque<ProfileSample>,
    /// Maximum number of samples retained.
    pub window_size: usize,
}

impl ProfileWindow {
    /// Create a new window with the given capacity.
    pub fn new(window_size: usize) -> Self {
        ProfileWindow {
            samples: VecDeque::with_capacity(window_size),
            window_size,
        }
    }

    /// Add a sample, evicting the oldest if the window is full.
    pub fn add_sample(&mut self, sample: ProfileSample) {
        if self.samples.len() >= self.window_size {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
    }

    /// Return the average CPU time in microseconds across all samples in the window.
    pub fn avg_cpu_us(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.samples.iter().map(|s| s.cpu_time_us).sum();
        sum as f64 / self.samples.len() as f64
    }

    /// Return the average memory usage in bytes across all samples in the window.
    pub fn avg_memory(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let sum: usize = self.samples.iter().map(|s| s.memory_bytes).sum();
        sum as f64 / self.samples.len() as f64
    }

    /// Return the maximum queue depth seen across all samples in the window.
    pub fn max_queue_depth(&self) -> usize {
        self.samples.iter().map(|s| s.queue_depth).max().unwrap_or(0)
    }

    /// Return the given percentile (0–100) of CPU time across samples in the window.
    ///
    /// Uses nearest-rank method. Returns `0.0` if the window is empty.
    pub fn cpu_percentile(&self, pct: f64) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let mut values: Vec<u64> = self.samples.iter().map(|s| s.cpu_time_us).collect();
        values.sort_unstable();
        let idx = ((pct / 100.0) * (values.len() as f64 - 1.0))
            .round()
            .clamp(0.0, (values.len() - 1) as f64) as usize;
        values[idx] as f64
    }
}

// ── AgentProfile ──────────────────────────────────────────────────────────────

/// Accumulated profiling state for a single agent.
#[derive(Debug, Clone)]
pub struct AgentProfile {
    /// ID of the profiled agent.
    pub agent_id: String,
    /// Sliding sample window.
    pub window: ProfileWindow,
    /// Total number of tasks processed since profiling started.
    pub total_tasks: u64,
    /// Total CPU time consumed since profiling started, in microseconds.
    pub total_cpu_us: u64,
    /// Timestamp (ms) when profiling for this agent began.
    pub start_time_ms: u64,
}

// ── AgentProfileReport ────────────────────────────────────────────────────────

/// Summary report for a single agent.
#[derive(Debug, Clone)]
pub struct AgentProfileReport {
    /// ID of the profiled agent.
    pub agent_id: String,
    /// Average CPU time per sample in the current window, in microseconds.
    pub avg_cpu_us: f64,
    /// 95th-percentile CPU time in the current window, in microseconds.
    pub p95_cpu_us: f64,
    /// Average memory usage in the current window, in bytes.
    pub avg_memory: f64,
    /// Maximum queue depth observed in the current window.
    pub max_queue: usize,
    /// Total number of tasks processed.
    pub total_tasks: u64,
    /// Tasks processed per second since profiling began.
    pub throughput_per_sec: f64,
}

// ── GlobalProfileSummary ──────────────────────────────────────────────────────

/// Aggregated summary across all profiled agents.
#[derive(Debug, Clone)]
pub struct GlobalProfileSummary {
    /// Total number of agents currently tracked.
    pub total_agents: usize,
    /// Agent ID with the highest average CPU usage, if any.
    pub busiest_agent: Option<String>,
    /// Combined total tasks processed across all agents.
    pub total_tasks_across_agents: u64,
    /// Average of per-agent average CPU times, in microseconds.
    pub avg_cpu_across_agents: f64,
}

// ── AgentProfiler ─────────────────────────────────────────────────────────────

/// Global profiler that collects [`ProfileSample`]s across multiple agents.
///
/// Uses an `Arc<std::sync::Mutex<HashMap<...>>>` internally so `record` can be
/// called from a shared reference without requiring `dashmap` in this crate.
#[derive(Debug, Clone)]
pub struct AgentProfiler {
    profiles: Arc<std::sync::Mutex<HashMap<String, AgentProfile>>>,
    window_size: usize,
}

impl AgentProfiler {
    /// Create a new profiler with the default sample window size of 100.
    pub fn new() -> Self {
        AgentProfiler {
            profiles: Arc::new(std::sync::Mutex::new(HashMap::new())),
            window_size: 100,
        }
    }

    /// Record a sample, upserting the [`AgentProfile`] as needed.
    ///
    /// If the lock is poisoned the sample is silently dropped.
    pub fn record(&self, sample: ProfileSample) {
        let Ok(mut map) = self.profiles.lock() else {
            return;
        };
        let agent_id = sample.agent_id.clone();
        let start_ms = sample.timestamp_ms;
        let profile = map.entry(agent_id.clone()).or_insert_with(|| AgentProfile {
            agent_id: agent_id.clone(),
            window: ProfileWindow::new(self.window_size),
            total_tasks: 0,
            total_cpu_us: 0,
            start_time_ms: start_ms,
        });
        profile.total_cpu_us += sample.cpu_time_us;
        profile.total_tasks += 1;
        profile.window.add_sample(sample);
    }

    /// Return IDs of agents whose average CPU or max queue depth exceeds the given thresholds.
    pub fn bottleneck_agents(
        &self,
        cpu_threshold_us: u64,
        queue_threshold: usize,
    ) -> Vec<String> {
        let Ok(map) = self.profiles.lock() else {
            return Vec::new();
        };
        map.values()
            .filter(|p| {
                p.window.avg_cpu_us() as u64 > cpu_threshold_us
                    || p.window.max_queue_depth() > queue_threshold
            })
            .map(|p| p.agent_id.clone())
            .collect()
    }

    /// Return a [`AgentProfileReport`] for the given agent, or `None` if unknown.
    pub fn agent_report(&self, agent_id: &str) -> Option<AgentProfileReport> {
        let Ok(map) = self.profiles.lock() else {
            return None;
        };
        let profile = map.get(agent_id)?;
        let avg_cpu_us = profile.window.avg_cpu_us();
        let p95_cpu_us = profile.window.cpu_percentile(95.0);
        let avg_memory = profile.window.avg_memory();
        let max_queue = profile.window.max_queue_depth();
        let total_tasks = profile.total_tasks;

        // Compute throughput using last sample timestamp if available.
        let last_ts = profile
            .window
            .samples
            .back()
            .map(|s| s.timestamp_ms)
            .unwrap_or(profile.start_time_ms);
        let elapsed_ms = last_ts.saturating_sub(profile.start_time_ms);
        let throughput_per_sec = if elapsed_ms == 0 {
            0.0
        } else {
            total_tasks as f64 / (elapsed_ms as f64 / 1000.0)
        };

        Some(AgentProfileReport {
            agent_id: agent_id.to_string(),
            avg_cpu_us,
            p95_cpu_us,
            avg_memory,
            max_queue,
            total_tasks,
            throughput_per_sec,
        })
    }

    /// Return a [`GlobalProfileSummary`] across all tracked agents.
    pub fn global_summary(&self) -> GlobalProfileSummary {
        let Ok(map) = self.profiles.lock() else {
            return GlobalProfileSummary {
                total_agents: 0,
                busiest_agent: None,
                total_tasks_across_agents: 0,
                avg_cpu_across_agents: 0.0,
            };
        };
        let total_agents = map.len();
        let total_tasks_across_agents: u64 = map.values().map(|p| p.total_tasks).sum();

        let cpu_values: Vec<(String, f64)> = map
            .values()
            .map(|p| (p.agent_id.clone(), p.window.avg_cpu_us()))
            .collect();

        let avg_cpu_across_agents = if cpu_values.is_empty() {
            0.0
        } else {
            cpu_values.iter().map(|(_, c)| c).sum::<f64>() / cpu_values.len() as f64
        };

        let busiest_agent = cpu_values
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(id, _)| id.clone());

        GlobalProfileSummary {
            total_agents,
            busiest_agent,
            total_tasks_across_agents,
            avg_cpu_across_agents,
        }
    }
}

impl Default for AgentProfiler {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn make_sample(agent_id: &str, cpu: u64, mem: usize, queue: usize, ts: u64) -> ProfileSample {
        ProfileSample {
            agent_id: agent_id.to_string(),
            timestamp_ms: ts,
            cpu_time_us: cpu,
            memory_bytes: mem,
            queue_depth: queue,
            task_name: "task".to_string(),
        }
    }

    #[test]
    fn add_samples_and_averages() {
        let mut window = ProfileWindow::new(10);
        window.add_sample(make_sample("a", 100, 1024, 2, 0));
        window.add_sample(make_sample("a", 200, 2048, 4, 1));
        assert!((window.avg_cpu_us() - 150.0).abs() < 1e-6);
        assert!((window.avg_memory() - 1536.0).abs() < 1e-6);
        assert_eq!(window.max_queue_depth(), 4);
    }

    #[test]
    fn percentile_calculation() {
        let mut window = ProfileWindow::new(10);
        for i in 1u64..=10 {
            window.add_sample(make_sample("a", i * 100, 0, 0, i));
        }
        // p50 of [100,200,...,1000]: idx = round(0.5 * 9) = 4 → 500
        let p50 = window.cpu_percentile(50.0);
        assert!((p50 - 500.0).abs() < 1e-6, "p50={p50}");
        // p100 → last element = 1000
        let p100 = window.cpu_percentile(100.0);
        assert!((p100 - 1000.0).abs() < 1e-6, "p100={p100}");
    }

    #[test]
    fn bottleneck_detection() {
        let profiler = AgentProfiler::new();
        profiler.record(make_sample("busy", 5000, 0, 0, 0));
        profiler.record(make_sample("idle", 10, 0, 0, 0));
        let bottlenecks = profiler.bottleneck_agents(1000, 100);
        assert!(bottlenecks.contains(&"busy".to_string()));
        assert!(!bottlenecks.contains(&"idle".to_string()));
    }

    #[test]
    fn empty_profiler_global_summary() {
        let profiler = AgentProfiler::new();
        let summary = profiler.global_summary();
        assert_eq!(summary.total_agents, 0);
        assert!(summary.busiest_agent.is_none());
        assert_eq!(summary.total_tasks_across_agents, 0);
        assert!((summary.avg_cpu_across_agents - 0.0).abs() < 1e-6);
    }
}

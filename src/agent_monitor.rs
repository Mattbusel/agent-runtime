//! Agent health monitoring, resource usage tracking, and performance analysis.
//!
//! Provides [`AgentMonitor`] for recording resource snapshots, evaluating
//! health status against configurable thresholds, computing performance
//! metrics, detecting anomalies, and producing global health summaries.

use std::collections::HashMap;

// ── AgentHealthStatus ─────────────────────────────────────────────────────────

/// The current health state of an agent.
#[derive(Debug, Clone, PartialEq)]
pub enum AgentHealthStatus {
    /// Agent is operating within all thresholds.
    Healthy,
    /// Agent is experiencing degraded performance.
    Degraded {
        /// Human-readable description of the degradation.
        reason: String,
    },
    /// Agent is in a critical failure state.
    Critical {
        /// Human-readable description of the critical condition.
        reason: String,
    },
    /// Health cannot be determined (no data).
    Unknown,
}

// ── ResourceSnapshot ──────────────────────────────────────────────────────────

/// A point-in-time resource usage snapshot for a single agent.
#[derive(Debug, Clone)]
pub struct ResourceSnapshot {
    /// Unix timestamp in milliseconds.
    pub timestamp: u64,
    /// CPU utilisation percentage (0.0–100.0).
    pub cpu_pct: f64,
    /// Memory usage in megabytes.
    pub memory_mb: u64,
    /// Number of currently active tasks.
    pub active_tasks: u32,
    /// Number of tasks waiting in the queue.
    pub queue_depth: u32,
}

// ── AgentPerformanceMetrics ───────────────────────────────────────────────────

/// Derived performance metrics computed from a window of snapshots.
#[derive(Debug, Clone)]
pub struct AgentPerformanceMetrics {
    /// Average task duration in milliseconds (approximated from throughput).
    pub avg_task_duration_ms: u64,
    /// Fraction of tasks that succeeded (0.0–1.0).  Estimated from CPU idle.
    pub task_success_rate: f64,
    /// Throughput: tasks completed per minute (estimated from active_tasks trend).
    pub throughput_per_min: f64,
    /// 95th-percentile estimated latency in milliseconds.
    pub p95_latency_ms: u64,
}

// ── HealthThresholds ──────────────────────────────────────────────────────────

/// Configurable thresholds that define healthy/degraded/critical boundaries.
#[derive(Debug, Clone)]
pub struct HealthThresholds {
    /// CPU% above which the agent is considered degraded.
    pub max_cpu_pct: f64,
    /// Memory (MB) above which the agent is considered degraded.
    pub max_memory_mb: u64,
    /// Queue depth above which the agent is considered degraded.
    pub max_queue_depth: u32,
    /// Task success rate below which the agent is considered degraded.
    pub min_success_rate: f64,
}

impl Default for HealthThresholds {
    fn default() -> Self {
        Self {
            max_cpu_pct: 80.0,
            max_memory_mb: 2048,
            max_queue_depth: 100,
            min_success_rate: 0.95,
        }
    }
}

// ── AgentMonitor ──────────────────────────────────────────────────────────────

/// Monitors the health and performance of multiple agents.
#[derive(Default)]
pub struct AgentMonitor {
    /// Snapshots keyed by agent ID.
    snapshots: HashMap<String, Vec<ResourceSnapshot>>,
    /// Simulated per-agent success counters (success, total) for health checks.
    success_counters: HashMap<String, (u64, u64)>,
}

impl AgentMonitor {
    /// Create a new, empty monitor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a resource snapshot for the given agent.
    pub fn record_snapshot(&mut self, agent_id: &str, snapshot: ResourceSnapshot) {
        self.snapshots
            .entry(agent_id.to_string())
            .or_default()
            .push(snapshot);
    }

    /// Record a task outcome for success-rate tracking.
    pub fn record_task_outcome(&mut self, agent_id: &str, success: bool) {
        let entry = self
            .success_counters
            .entry(agent_id.to_string())
            .or_insert((0, 0));
        if success {
            entry.0 += 1;
        }
        entry.1 += 1;
    }

    /// Evaluate the health status of a single agent against the provided thresholds.
    ///
    /// Uses the most recent snapshot available.  If no data is available,
    /// returns [`AgentHealthStatus::Unknown`].
    pub fn health_status(&self, agent_id: &str, thresholds: &HealthThresholds) -> AgentHealthStatus {
        let snaps = match self.snapshots.get(agent_id) {
            Some(s) if !s.is_empty() => s,
            _ => return AgentHealthStatus::Unknown,
        };
        let latest = &snaps[snaps.len() - 1];

        // Critical conditions: any metric 2x above threshold.
        if latest.cpu_pct > thresholds.max_cpu_pct * 2.0 {
            return AgentHealthStatus::Critical {
                reason: format!(
                    "CPU at {:.1}% (critical threshold {:.1}%)",
                    latest.cpu_pct,
                    thresholds.max_cpu_pct * 2.0
                ),
            };
        }
        if latest.memory_mb > thresholds.max_memory_mb * 2 {
            return AgentHealthStatus::Critical {
                reason: format!(
                    "Memory at {} MB (critical threshold {} MB)",
                    latest.memory_mb,
                    thresholds.max_memory_mb * 2
                ),
            };
        }

        // Success-rate check.
        let success_rate = self.success_rate(agent_id);
        if let Some(rate) = success_rate {
            if rate < thresholds.min_success_rate * 0.5 {
                return AgentHealthStatus::Critical {
                    reason: format!(
                        "Success rate {:.1}% far below minimum {:.1}%",
                        rate * 100.0,
                        thresholds.min_success_rate * 100.0
                    ),
                };
            }
            if rate < thresholds.min_success_rate {
                return AgentHealthStatus::Degraded {
                    reason: format!(
                        "Success rate {:.1}% below minimum {:.1}%",
                        rate * 100.0,
                        thresholds.min_success_rate * 100.0
                    ),
                };
            }
        }

        // Degraded conditions.
        if latest.cpu_pct > thresholds.max_cpu_pct {
            return AgentHealthStatus::Degraded {
                reason: format!(
                    "CPU at {:.1}% (threshold {:.1}%)",
                    latest.cpu_pct, thresholds.max_cpu_pct
                ),
            };
        }
        if latest.memory_mb > thresholds.max_memory_mb {
            return AgentHealthStatus::Degraded {
                reason: format!(
                    "Memory at {} MB (threshold {} MB)",
                    latest.memory_mb, thresholds.max_memory_mb
                ),
            };
        }
        if latest.queue_depth > thresholds.max_queue_depth {
            return AgentHealthStatus::Degraded {
                reason: format!(
                    "Queue depth {} exceeds threshold {}",
                    latest.queue_depth, thresholds.max_queue_depth
                ),
            };
        }

        AgentHealthStatus::Healthy
    }

    /// Compute performance metrics from the most recent `window_size` snapshots.
    ///
    /// Returns `None` if no snapshots exist for the agent.
    pub fn performance_metrics(
        &self,
        agent_id: &str,
        window_size: usize,
    ) -> Option<AgentPerformanceMetrics> {
        let snaps = self.snapshots.get(agent_id)?;
        if snaps.is_empty() {
            return None;
        }

        let window: &[ResourceSnapshot] = if snaps.len() >= window_size {
            &snaps[snaps.len() - window_size..]
        } else {
            snaps
        };

        let n = window.len() as f64;

        let avg_cpu: f64 = window.iter().map(|s| s.cpu_pct).sum::<f64>() / n;
        let avg_active: f64 = window.iter().map(|s| s.active_tasks as f64).sum::<f64>() / n;

        // Approximate task duration from inverse of active-task throughput.
        // Higher active tasks at lower CPU → longer tasks.
        let avg_task_duration_ms = if avg_active > 0.0 {
            ((100.0 / avg_cpu.max(1.0)) * avg_active * 10.0) as u64
        } else {
            10
        };

        let task_success_rate = self.success_rate(agent_id).unwrap_or(1.0);

        // Throughput: tasks/min estimated from average active tasks over window.
        let throughput_per_min = avg_active * 60.0;

        // p95: take the 95th percentile of per-snapshot (cpu * 10) as latency proxy.
        let mut latency_proxy: Vec<u64> = window
            .iter()
            .map(|s| (s.cpu_pct * 10.0).round() as u64 + s.queue_depth as u64 * 5)
            .collect();
        latency_proxy.sort_unstable();
        let p95_idx = ((latency_proxy.len() as f64 * 0.95).ceil() as usize)
            .min(latency_proxy.len())
            .saturating_sub(1);
        let p95_latency_ms = latency_proxy[p95_idx];

        Some(AgentPerformanceMetrics {
            avg_task_duration_ms,
            task_success_rate,
            throughput_per_min,
            p95_latency_ms,
        })
    }

    /// Detect anomalies in the agent's recent snapshot window.
    ///
    /// Anomaly types detected:
    /// - **CPU spike**: any snapshot where CPU > mean + 2 * stddev.
    /// - **Memory leak**: memory is monotonically increasing across the window.
    /// - **Queue buildup**: queue depth is monotonically increasing across the window.
    pub fn anomaly_detection(&self, agent_id: &str, window: usize) -> Vec<String> {
        let mut anomalies = Vec::new();

        let snaps = match self.snapshots.get(agent_id) {
            Some(s) if !s.is_empty() => s,
            _ => return anomalies,
        };

        let w: &[ResourceSnapshot] = if snaps.len() >= window {
            &snaps[snaps.len() - window..]
        } else {
            snaps
        };

        if w.len() < 2 {
            return anomalies;
        }

        // CPU spike detection.
        let cpu_vals: Vec<f64> = w.iter().map(|s| s.cpu_pct).collect();
        let mean_cpu = cpu_vals.iter().sum::<f64>() / cpu_vals.len() as f64;
        let variance = cpu_vals.iter().map(|v| (v - mean_cpu).powi(2)).sum::<f64>()
            / cpu_vals.len() as f64;
        let std_cpu = variance.sqrt();

        let threshold = mean_cpu + 2.0 * std_cpu;
        for (i, snap) in w.iter().enumerate() {
            if snap.cpu_pct > threshold && std_cpu > 0.5 {
                anomalies.push(format!(
                    "CPU spike at snapshot {}: {:.1}% (threshold {:.1}%)",
                    i, snap.cpu_pct, threshold
                ));
            }
        }

        // Memory leak: monotonically increasing memory.
        let monotone_mem = w
            .windows(2)
            .all(|pair| pair[1].memory_mb >= pair[0].memory_mb);
        if monotone_mem {
            let start = w[0].memory_mb;
            let end = w[w.len() - 1].memory_mb;
            if end > start {
                anomalies.push(format!(
                    "Memory leak detected: {} MB -> {} MB over {} snapshots",
                    start,
                    end,
                    w.len()
                ));
            }
        }

        // Queue buildup: monotonically increasing queue depth.
        let monotone_queue = w
            .windows(2)
            .all(|pair| pair[1].queue_depth >= pair[0].queue_depth);
        if monotone_queue {
            let start = w[0].queue_depth;
            let end = w[w.len() - 1].queue_depth;
            if end > start {
                anomalies.push(format!(
                    "Queue buildup detected: {} -> {} requests over {} snapshots",
                    start,
                    end,
                    w.len()
                ));
            }
        }

        anomalies
    }

    /// Produce a health status for every monitored agent.
    pub fn global_health_summary(
        &self,
        thresholds: &HealthThresholds,
    ) -> HashMap<String, AgentHealthStatus> {
        self.snapshots
            .keys()
            .map(|id| (id.clone(), self.health_status(id, thresholds)))
            .collect()
    }

    /// Return the top-N agents by a specified metric.
    ///
    /// `metric` accepts `"cpu"`, `"memory"`, or `"queue"`.
    /// Returns `(agent_id, metric_value)` pairs sorted descending.
    pub fn top_consumers(&self, metric: &str, n: usize) -> Vec<(String, f64)> {
        let mut scored: Vec<(String, f64)> = self
            .snapshots
            .iter()
            .filter_map(|(id, snaps)| {
                let latest = snaps.last()?;
                let val = match metric {
                    "cpu" => latest.cpu_pct,
                    "memory" => latest.memory_mb as f64,
                    "queue" => latest.queue_depth as f64,
                    _ => return None,
                };
                Some((id.clone(), val))
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(n);
        scored
    }

    // ── private helpers ─────────────────────────────────────────────────────

    fn success_rate(&self, agent_id: &str) -> Option<f64> {
        let (successes, total) = self.success_counters.get(agent_id)?;
        if *total == 0 {
            return None;
        }
        Some(*successes as f64 / *total as f64)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn snap(cpu: f64, mem: u64, tasks: u32, queue: u32) -> ResourceSnapshot {
        ResourceSnapshot {
            timestamp: 0,
            cpu_pct: cpu,
            memory_mb: mem,
            active_tasks: tasks,
            queue_depth: queue,
        }
    }

    fn thresholds() -> HealthThresholds {
        HealthThresholds {
            max_cpu_pct: 80.0,
            max_memory_mb: 1024,
            max_queue_depth: 50,
            min_success_rate: 0.95,
        }
    }

    #[test]
    fn no_snapshots_returns_unknown() {
        let monitor = AgentMonitor::new();
        assert_eq!(
            monitor.health_status("agent-1", &thresholds()),
            AgentHealthStatus::Unknown
        );
    }

    #[test]
    fn healthy_agent() {
        let mut monitor = AgentMonitor::new();
        monitor.record_snapshot("a", snap(30.0, 256, 5, 10));
        monitor.record_task_outcome("a", true);
        assert_eq!(
            monitor.health_status("a", &thresholds()),
            AgentHealthStatus::Healthy
        );
    }

    #[test]
    fn degraded_on_high_cpu() {
        let mut monitor = AgentMonitor::new();
        monitor.record_snapshot("a", snap(85.0, 256, 5, 10));
        let status = monitor.health_status("a", &thresholds());
        assert!(
            matches!(status, AgentHealthStatus::Degraded { .. }),
            "{status:?}"
        );
    }

    #[test]
    fn critical_on_very_high_cpu() {
        let mut monitor = AgentMonitor::new();
        monitor.record_snapshot("a", snap(165.0, 256, 5, 10));
        let status = monitor.health_status("a", &thresholds());
        assert!(
            matches!(status, AgentHealthStatus::Critical { .. }),
            "{status:?}"
        );
    }

    #[test]
    fn degraded_on_high_memory() {
        let mut monitor = AgentMonitor::new();
        monitor.record_snapshot("a", snap(20.0, 1200, 5, 10));
        let status = monitor.health_status("a", &thresholds());
        assert!(
            matches!(status, AgentHealthStatus::Degraded { .. }),
            "{status:?}"
        );
    }

    #[test]
    fn degraded_on_queue_depth() {
        let mut monitor = AgentMonitor::new();
        monitor.record_snapshot("a", snap(20.0, 256, 5, 100));
        let status = monitor.health_status("a", &thresholds());
        assert!(
            matches!(status, AgentHealthStatus::Degraded { .. }),
            "{status:?}"
        );
    }

    #[test]
    fn degraded_on_low_success_rate() {
        let mut monitor = AgentMonitor::new();
        monitor.record_snapshot("a", snap(20.0, 256, 5, 10));
        for _ in 0..100 {
            monitor.record_task_outcome("a", false);
        }
        let status = monitor.health_status("a", &thresholds());
        assert!(
            matches!(status, AgentHealthStatus::Critical { .. } | AgentHealthStatus::Degraded { .. }),
            "{status:?}"
        );
    }

    #[test]
    fn performance_metrics_none_for_unknown_agent() {
        let monitor = AgentMonitor::new();
        assert!(monitor.performance_metrics("unknown", 10).is_none());
    }

    #[test]
    fn performance_metrics_computed() {
        let mut monitor = AgentMonitor::new();
        for i in 0..10 {
            monitor.record_snapshot("a", snap(50.0, 512, 10 + i, 5));
        }
        let metrics = monitor.performance_metrics("a", 10).unwrap();
        assert!(metrics.throughput_per_min > 0.0);
        assert!(metrics.avg_task_duration_ms > 0);
    }

    #[test]
    fn anomaly_detection_cpu_spike() {
        let mut monitor = AgentMonitor::new();
        for _ in 0..9 {
            monitor.record_snapshot("a", snap(10.0, 256, 2, 3));
        }
        // Insert a spike.
        monitor.record_snapshot("a", snap(90.0, 256, 2, 3));
        let anomalies = monitor.anomaly_detection("a", 10);
        assert!(
            anomalies.iter().any(|a| a.contains("CPU spike")),
            "{anomalies:?}"
        );
    }

    #[test]
    fn anomaly_detection_memory_leak() {
        let mut monitor = AgentMonitor::new();
        for i in 0..5u64 {
            monitor.record_snapshot("a", snap(20.0, 100 + i * 50, 2, 3));
        }
        let anomalies = monitor.anomaly_detection("a", 5);
        assert!(
            anomalies.iter().any(|a| a.contains("Memory leak")),
            "{anomalies:?}"
        );
    }

    #[test]
    fn anomaly_detection_queue_buildup() {
        let mut monitor = AgentMonitor::new();
        for i in 0..5u32 {
            monitor.record_snapshot("a", snap(20.0, 256, 2, i * 10));
        }
        let anomalies = monitor.anomaly_detection("a", 5);
        assert!(
            anomalies.iter().any(|a| a.contains("Queue buildup")),
            "{anomalies:?}"
        );
    }

    #[test]
    fn global_health_summary() {
        let mut monitor = AgentMonitor::new();
        monitor.record_snapshot("a", snap(20.0, 256, 2, 3));
        monitor.record_snapshot("b", snap(90.0, 256, 2, 3));
        let summary = monitor.global_health_summary(&thresholds());
        assert_eq!(summary.len(), 2);
        assert_eq!(summary["a"], AgentHealthStatus::Healthy);
        assert!(matches!(summary["b"], AgentHealthStatus::Degraded { .. }));
    }

    #[test]
    fn top_consumers_cpu() {
        let mut monitor = AgentMonitor::new();
        monitor.record_snapshot("low", snap(10.0, 256, 2, 3));
        monitor.record_snapshot("high", snap(90.0, 256, 2, 3));
        monitor.record_snapshot("mid", snap(50.0, 256, 2, 3));
        let top = monitor.top_consumers("cpu", 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, "high");
        assert_eq!(top[1].0, "mid");
    }

    #[test]
    fn top_consumers_memory() {
        let mut monitor = AgentMonitor::new();
        monitor.record_snapshot("small", snap(10.0, 100, 2, 3));
        monitor.record_snapshot("large", snap(10.0, 2000, 2, 3));
        let top = monitor.top_consumers("memory", 1);
        assert_eq!(top[0].0, "large");
    }

    #[test]
    fn top_consumers_unknown_metric_empty() {
        let mut monitor = AgentMonitor::new();
        monitor.record_snapshot("a", snap(10.0, 100, 2, 3));
        let top = monitor.top_consumers("unknown_metric", 5);
        assert!(top.is_empty());
    }
}

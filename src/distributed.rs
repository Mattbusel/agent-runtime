//! # Module: Distributed
//!
//! ## Responsibility
//! Provides distributed agent coordination primitives backed by Redis.
//!
//! - [`DistributedCoordinator`] — shared work queue, task claiming, and
//!   leader election with TTL keys.
//!
//! ## Feature Gate
//! This module is only compiled when the `distributed` feature is enabled.

use crate::error::AgentRuntimeError;
use redis::AsyncCommands;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

// ── WorkItem ──────────────────────────────────────────────────────────────────

/// A unit of work that can be placed on the distributed queue.
///
/// Tasks are JSON-serialized when pushed to Redis and deserialized when
/// a worker claims them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkItem {
    /// Stable identifier for this task (caller-assigned).
    pub id: String,
    /// Logical type/kind of the task (e.g. `"summarize"`, `"classify"`).
    pub kind: String,
    /// Arbitrary JSON payload passed to the worker.
    pub payload: serde_json::Value,
    /// UTC timestamp when the item was enqueued (ISO 8601).
    pub enqueued_at: String,
}

impl WorkItem {
    /// Construct a new `WorkItem` with the current timestamp.
    pub fn new(id: impl Into<String>, kind: impl Into<String>, payload: serde_json::Value) -> Self {
        Self {
            id: id.into(),
            kind: kind.into(),
            payload,
            enqueued_at: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Return the byte length of the serialized payload.
    pub fn payload_byte_len(&self) -> usize {
        self.payload.to_string().len()
    }
}

impl std::fmt::Display for WorkItem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WorkItem({}, kind={})", self.id, self.kind)
    }
}

// ── CoordinatorConfig ─────────────────────────────────────────────────────────

/// Configuration for [`DistributedCoordinator`].
#[derive(Debug, Clone)]
pub struct CoordinatorConfig {
    /// Redis key used as the shared work queue (LPUSH / BRPOP).
    pub queue_key: String,
    /// Redis key prefix for leader election (`{prefix}:leader`).
    pub leader_key_prefix: String,
    /// TTL in seconds for the leader election key.  The leader must renew
    /// its lease before this expires or another instance may claim leadership.
    pub leader_ttl_secs: u64,
    /// Unique identifier for this agent instance (used in leader election).
    pub instance_id: String,
}

impl CoordinatorConfig {
    /// Create a new config with sensible defaults.
    ///
    /// `instance_id` should be unique across all replicas (e.g. hostname + PID).
    pub fn new(instance_id: impl Into<String>) -> Self {
        Self {
            queue_key: "agent:work_queue".into(),
            leader_key_prefix: "agent:coord".into(),
            leader_ttl_secs: 30,
            instance_id: instance_id.into(),
        }
    }

    /// Override the Redis work-queue key.
    pub fn with_queue_key(mut self, key: impl Into<String>) -> Self {
        self.queue_key = key.into();
        self
    }

    /// Override the leader election key prefix.
    pub fn with_leader_key_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.leader_key_prefix = prefix.into();
        self
    }

    /// Override the leader TTL in seconds.
    pub fn with_leader_ttl_secs(mut self, secs: u64) -> Self {
        self.leader_ttl_secs = secs;
        self
    }
}

// ── DistributedCoordinator ────────────────────────────────────────────────────

/// Distributed agent coordinator backed by Redis.
///
/// Multiple instances of `DistributedCoordinator` (across processes or hosts)
/// sharing the same Redis database can cooperate on a single shared work queue
/// and elect a leader using a TTL key.
///
/// # Work Queue
///
/// Tasks are pushed with [`enqueue`] and atomically claimed with [`claim`].
/// Claiming is atomic: only one worker receives each task regardless of how
/// many are racing.
///
/// # Leader Election
///
/// [`try_become_leader`] attempts a `SET NX PX` on the leader key.  The first
/// caller wins; subsequent callers return `false` until the key expires (or is
/// released with [`resign_leader`]).  The current leader should call
/// [`renew_leader`] periodically to reset the TTL and prevent accidental
/// failover.
///
/// [`enqueue`]: DistributedCoordinator::enqueue
/// [`claim`]: DistributedCoordinator::claim
/// [`try_become_leader`]: DistributedCoordinator::try_become_leader
/// [`resign_leader`]: DistributedCoordinator::resign_leader
/// [`renew_leader`]: DistributedCoordinator::renew_leader
pub struct DistributedCoordinator {
    config: CoordinatorConfig,
    /// Async Redis connection (Mutex for interior mutability across tasks).
    conn: Arc<Mutex<redis::aio::MultiplexedConnection>>,
}

impl std::fmt::Debug for DistributedCoordinator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DistributedCoordinator")
            .field("instance_id", &self.config.instance_id)
            .field("queue_key", &self.config.queue_key)
            .field("leader_ttl_secs", &self.config.leader_ttl_secs)
            .finish()
    }
}

impl DistributedCoordinator {
    /// Connect to Redis at `redis_url` (e.g. `"redis://127.0.0.1/"`) and
    /// return a ready coordinator.
    ///
    /// # Errors
    /// Returns `AgentRuntimeError::Persistence` if the connection fails.
    pub async fn connect(
        redis_url: &str,
        config: CoordinatorConfig,
    ) -> Result<Self, AgentRuntimeError> {
        let client = redis::Client::open(redis_url)
            .map_err(|e| AgentRuntimeError::Persistence(format!("redis client: {e}")))?;
        let conn = client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| AgentRuntimeError::Persistence(format!("redis connect: {e}")))?;
        Ok(Self {
            config,
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    // ── Work Queue ─────────────────────────────────────────────────────────

    /// Push a [`WorkItem`] onto the tail of the shared work queue.
    ///
    /// The item is JSON-serialized before being stored in Redis.
    ///
    /// # Errors
    /// Returns `AgentRuntimeError::Persistence` if serialization or the Redis
    /// LPUSH command fails.
    pub async fn enqueue(&self, item: &WorkItem) -> Result<(), AgentRuntimeError> {
        let json = serde_json::to_string(item)
            .map_err(|e| AgentRuntimeError::Persistence(format!("serialize WorkItem: {e}")))?;
        let mut conn = self.conn.lock().await;
        conn.lpush::<_, _, ()>(&self.config.queue_key, json)
            .await
            .map_err(|e| AgentRuntimeError::Persistence(format!("redis LPUSH: {e}")))?;
        Ok(())
    }

    /// Attempt to claim the next available work item from the queue.
    ///
    /// Returns `Ok(Some(item))` if an item was available, `Ok(None)` if the
    /// queue was empty.  The pop is atomic — only one caller receives each item.
    ///
    /// # Errors
    /// Returns `AgentRuntimeError::Persistence` if the Redis RPOP command fails
    /// or the stored JSON cannot be deserialized.
    pub async fn claim(&self) -> Result<Option<WorkItem>, AgentRuntimeError> {
        let mut conn = self.conn.lock().await;
        let raw: Option<String> = conn
            .rpop(&self.config.queue_key, None)
            .await
            .map_err(|e| AgentRuntimeError::Persistence(format!("redis RPOP: {e}")))?;
        match raw {
            None => Ok(None),
            Some(json) => {
                let item: WorkItem = serde_json::from_str(&json).map_err(|e| {
                    AgentRuntimeError::Persistence(format!("deserialize WorkItem: {e}"))
                })?;
                Ok(Some(item))
            }
        }
    }

    /// Return the number of items currently waiting in the work queue.
    ///
    /// # Errors
    /// Returns `AgentRuntimeError::Persistence` if the Redis LLEN command fails.
    pub async fn queue_depth(&self) -> Result<usize, AgentRuntimeError> {
        let mut conn = self.conn.lock().await;
        let len: usize = conn
            .llen(&self.config.queue_key)
            .await
            .map_err(|e| AgentRuntimeError::Persistence(format!("redis LLEN: {e}")))?;
        Ok(len)
    }

    // ── Leader Election ────────────────────────────────────────────────────

    fn leader_key(&self) -> String {
        format!("{}:leader", self.config.leader_key_prefix)
    }

    /// Attempt to become the leader using a Redis `SET NX PX` command.
    ///
    /// Returns `Ok(true)` if this instance successfully acquired leadership,
    /// `Ok(false)` if another instance already holds the leader key.
    ///
    /// # Errors
    /// Returns `AgentRuntimeError::Persistence` if the Redis command fails.
    pub async fn try_become_leader(&self) -> Result<bool, AgentRuntimeError> {
        let key = self.leader_key();
        let ttl_ms = self.config.leader_ttl_secs * 1000;
        let mut conn = self.conn.lock().await;
        // SET key value NX PX ms — returns "OK" on success, nil on failure.
        let result: Option<String> = redis::cmd("SET")
            .arg(&key)
            .arg(&self.config.instance_id)
            .arg("NX")
            .arg("PX")
            .arg(ttl_ms)
            .query_async(&mut *conn)
            .await
            .map_err(|e| AgentRuntimeError::Persistence(format!("redis SET NX: {e}")))?;
        Ok(result.as_deref() == Some("OK"))
    }

    /// Renew the leader TTL without releasing leadership.
    ///
    /// Only renews if this instance is the current leader (value matches
    /// `instance_id`).  Returns `Ok(true)` if the TTL was successfully renewed.
    ///
    /// # Errors
    /// Returns `AgentRuntimeError::Persistence` if the Redis commands fail.
    pub async fn renew_leader(&self) -> Result<bool, AgentRuntimeError> {
        let key = self.leader_key();
        let mut conn = self.conn.lock().await;
        let current: Option<String> = conn
            .get(&key)
            .await
            .map_err(|e| AgentRuntimeError::Persistence(format!("redis GET leader: {e}")))?;
        if current.as_deref() != Some(self.config.instance_id.as_str()) {
            return Ok(false);
        }
        let ttl_ms = self.config.leader_ttl_secs * 1000;
        conn.pexpire::<_, ()>(&key, ttl_ms as i64)
            .await
            .map_err(|e| AgentRuntimeError::Persistence(format!("redis PEXPIRE: {e}")))?;
        Ok(true)
    }

    /// Release leadership if this instance currently holds it.
    ///
    /// Uses a Lua script to atomically check the value and delete only if it
    /// matches this instance's `instance_id`, preventing accidental release of
    /// another instance's lock.  Returns `Ok(true)` if leadership was released.
    ///
    /// # Errors
    /// Returns `AgentRuntimeError::Persistence` if the Redis command fails.
    pub async fn resign_leader(&self) -> Result<bool, AgentRuntimeError> {
        let key = self.leader_key();
        let script = redis::Script::new(
            r#"
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
            "#,
        );
        let mut conn = self.conn.lock().await;
        let deleted: i64 = script
            .key(&key)
            .arg(&self.config.instance_id)
            .invoke_async(&mut *conn)
            .await
            .map_err(|e| AgentRuntimeError::Persistence(format!("redis resign script: {e}")))?;
        Ok(deleted > 0)
    }

    /// Return `Ok(true)` if this instance is the current leader.
    ///
    /// # Errors
    /// Returns `AgentRuntimeError::Persistence` if the Redis GET fails.
    pub async fn is_leader(&self) -> Result<bool, AgentRuntimeError> {
        let key = self.leader_key();
        let mut conn = self.conn.lock().await;
        let current: Option<String> = conn
            .get(&key)
            .await
            .map_err(|e| AgentRuntimeError::Persistence(format!("redis GET leader: {e}")))?;
        Ok(current.as_deref() == Some(self.config.instance_id.as_str()))
    }

    /// Return the `instance_id` of the current leader, or `None` if no leader
    /// has been elected or the key has expired.
    ///
    /// # Errors
    /// Returns `AgentRuntimeError::Persistence` if the Redis GET fails.
    pub async fn current_leader(&self) -> Result<Option<String>, AgentRuntimeError> {
        let key = self.leader_key();
        let mut conn = self.conn.lock().await;
        let value: Option<String> = conn
            .get(&key)
            .await
            .map_err(|e| AgentRuntimeError::Persistence(format!("redis GET leader: {e}")))?;
        Ok(value)
    }

    /// Return this instance's configured `instance_id`.
    pub fn instance_id(&self) -> &str {
        &self.config.instance_id
    }

    /// Return a reference to the coordinator configuration.
    pub fn config(&self) -> &CoordinatorConfig {
        &self.config
    }
}

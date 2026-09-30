//! # Distributed Lock
//!
//! Optimistic locking with versioning and lease management.
//!
//! Provides a fully in-process distributed lock implementation with:
//! - Version-stamped lock entries for optimistic concurrency control
//! - TTL-based lease expiry with automatic eviction
//! - Compare-and-swap for optimistic acquisition
//! - Lease renewal to extend TTL without releasing

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

// ── Error ────────────────────────────────────────────────────────────────────

/// Errors that can occur during lock operations.
#[derive(Debug, PartialEq)]
pub enum LockError {
    /// The resource is already held by another holder.
    AlreadyLocked {
        /// The current lock holder.
        holder: String,
    },
    /// A compare-and-swap found an unexpected version.
    VersionConflict {
        /// The version the caller expected.
        expected: u64,
        /// The version actually present.
        actual: u64,
    },
    /// The lock lease has expired.
    Expired,
    /// The caller is not the owner of the lock.
    NotOwner,
}

impl fmt::Display for LockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LockError::AlreadyLocked { holder } => {
                write!(f, "resource is already locked by '{holder}'")
            }
            LockError::VersionConflict { expected, actual } => {
                write!(
                    f,
                    "version conflict: expected {expected}, found {actual}"
                )
            }
            LockError::Expired => write!(f, "lock lease has expired"),
            LockError::NotOwner => write!(f, "caller is not the lock owner"),
        }
    }
}

impl std::error::Error for LockError {}

// ── LockVersion ──────────────────────────────────────────────────────────────

/// Metadata describing a particular version of a lock.
#[derive(Debug, Clone)]
pub struct LockVersion {
    /// Identifier of the resource being locked.
    pub resource_id: String,
    /// Monotonically increasing version number.
    pub version: u64,
    /// Identity of the holder that acquired this version.
    pub holder: String,
    /// Wall-clock milliseconds when the lock was acquired.
    pub acquired_at_ms: u64,
    /// Time-to-live in milliseconds from `acquired_at_ms`.
    pub ttl_ms: u64,
}

impl LockVersion {
    /// Returns `true` if this version has expired relative to `now_ms`.
    fn is_expired(&self, now_ms: u64) -> bool {
        now_ms.saturating_sub(self.acquired_at_ms) >= self.ttl_ms
    }
}

// ── LockEntry ────────────────────────────────────────────────────────────────

/// Internal storage for a lock slot.
#[derive(Debug, Clone)]
pub struct LockEntry {
    /// The current version metadata.
    pub version: LockVersion,
    /// Whether the lock is currently held (not expired or released).
    pub is_locked: bool,
}

// ── LockStats ────────────────────────────────────────────────────────────────

/// Aggregate statistics for a [`DistributedLock`] instance.
#[derive(Debug, Clone, Default)]
pub struct LockStats {
    /// Total number of lock slots tracked (including expired ones not yet evicted).
    pub total_locks: usize,
    /// Number of currently active (non-expired, held) locks.
    pub active_locks: usize,
    /// Cumulative number of expired locks evicted via [`DistributedLock::evict_expired`].
    pub expired_evicted: usize,
}

// ── DistributedLock ──────────────────────────────────────────────────────────

/// Thread-safe optimistic lock registry with versioned lease management.
pub struct DistributedLock {
    locks: Mutex<HashMap<String, LockEntry>>,
    next_version: AtomicU64,
    /// Running total of evictions, updated by `evict_expired`.
    evicted_total: AtomicU64,
}

impl DistributedLock {
    /// Create a new, empty lock registry.
    pub fn new() -> Self {
        Self {
            locks: Mutex::new(HashMap::new()),
            next_version: AtomicU64::new(1),
            evicted_total: AtomicU64::new(0),
        }
    }

    /// Allocate and return the next version number.
    fn next_ver(&self) -> u64 {
        self.next_version.fetch_add(1, Ordering::Relaxed)
    }

    /// Acquire a lock on `resource_id` for `holder` with the given TTL.
    ///
    /// Returns the version number on success.  Fails with
    /// [`LockError::AlreadyLocked`] if the resource is currently held by a
    /// different (non-expired) holder.
    pub fn acquire(
        &self,
        resource_id: &str,
        holder: &str,
        ttl_ms: u64,
        now_ms: u64,
    ) -> Result<u64, LockError> {
        let mut locks = self
            .locks
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        if let Some(entry) = locks.get(resource_id) {
            if entry.is_locked && !entry.version.is_expired(now_ms) {
                return Err(LockError::AlreadyLocked {
                    holder: entry.version.holder.clone(),
                });
            }
        }

        let version = self.next_ver();
        locks.insert(
            resource_id.to_string(),
            LockEntry {
                version: LockVersion {
                    resource_id: resource_id.to_string(),
                    version,
                    holder: holder.to_string(),
                    acquired_at_ms: now_ms,
                    ttl_ms,
                },
                is_locked: true,
            },
        );
        Ok(version)
    }

    /// Release the lock on `resource_id`.
    ///
    /// Fails if the caller is not the holder or the version does not match.
    pub fn release(
        &self,
        resource_id: &str,
        holder: &str,
        version: u64,
    ) -> Result<(), LockError> {
        let mut locks = self
            .locks
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        match locks.get_mut(resource_id) {
            None => Err(LockError::NotOwner),
            Some(entry) => {
                if entry.version.holder != holder {
                    return Err(LockError::NotOwner);
                }
                if entry.version.version != version {
                    return Err(LockError::VersionConflict {
                        expected: version,
                        actual: entry.version.version,
                    });
                }
                entry.is_locked = false;
                Ok(())
            }
        }
    }

    /// Renew the lease on `resource_id`, extending its TTL by `additional_ttl_ms`.
    ///
    /// Returns the new version number.  Fails if the caller is not the holder,
    /// version does not match, or the lease has already expired.
    pub fn renew(
        &self,
        resource_id: &str,
        holder: &str,
        version: u64,
        additional_ttl_ms: u64,
        now_ms: u64,
    ) -> Result<u64, LockError> {
        let mut locks = self
            .locks
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        match locks.get_mut(resource_id) {
            None => Err(LockError::NotOwner),
            Some(entry) => {
                if entry.version.holder != holder {
                    return Err(LockError::NotOwner);
                }
                if entry.version.version != version {
                    return Err(LockError::VersionConflict {
                        expected: version,
                        actual: entry.version.version,
                    });
                }
                if entry.version.is_expired(now_ms) {
                    return Err(LockError::Expired);
                }
                let new_version = self.next_version.fetch_add(1, Ordering::Relaxed);
                entry.version.version = new_version;
                entry.version.acquired_at_ms = now_ms;
                entry.version.ttl_ms = additional_ttl_ms;
                entry.is_locked = true;
                Ok(new_version)
            }
        }
    }

    /// Optimistically acquire `resource_id` only if its current version equals
    /// `expected_version`.
    ///
    /// Returns the new version number on success.
    pub fn compare_and_swap(
        &self,
        resource_id: &str,
        expected_version: u64,
        holder: &str,
        ttl_ms: u64,
        now_ms: u64,
    ) -> Result<u64, LockError> {
        let mut locks = self
            .locks
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let actual_version = locks
            .get(resource_id)
            .map(|e| e.version.version)
            .unwrap_or(0);

        if actual_version != expected_version {
            return Err(LockError::VersionConflict {
                expected: expected_version,
                actual: actual_version,
            });
        }

        // Check if currently locked by someone else (and not expired).
        if let Some(entry) = locks.get(resource_id) {
            if entry.is_locked && !entry.version.is_expired(now_ms) {
                return Err(LockError::AlreadyLocked {
                    holder: entry.version.holder.clone(),
                });
            }
        }

        let new_version = self.next_version.fetch_add(1, Ordering::Relaxed);
        locks.insert(
            resource_id.to_string(),
            LockEntry {
                version: LockVersion {
                    resource_id: resource_id.to_string(),
                    version: new_version,
                    holder: holder.to_string(),
                    acquired_at_ms: now_ms,
                    ttl_ms,
                },
                is_locked: true,
            },
        );
        Ok(new_version)
    }

    /// Returns `true` if `resource_id` is currently locked and not expired.
    pub fn is_locked(&self, resource_id: &str, now_ms: u64) -> bool {
        let locks = self
            .locks
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        locks
            .get(resource_id)
            .map(|e| e.is_locked && !e.version.is_expired(now_ms))
            .unwrap_or(false)
    }

    /// Returns the holder of `resource_id` if it is currently locked and not expired.
    pub fn lock_holder(&self, resource_id: &str, now_ms: u64) -> Option<String> {
        let locks = self
            .locks
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        locks.get(resource_id).and_then(|e| {
            if e.is_locked && !e.version.is_expired(now_ms) {
                Some(e.version.holder.clone())
            } else {
                None
            }
        })
    }

    /// Remove all entries whose leases have expired.  Returns the number evicted.
    pub fn evict_expired(&self, now_ms: u64) -> usize {
        let mut locks = self
            .locks
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let before = locks.len();
        locks.retain(|_, entry| {
            // Keep entries that are either unlocked-but-still-fresh or not expired
            !(entry.is_locked && entry.version.is_expired(now_ms))
        });
        let evicted = before - locks.len();
        self.evicted_total.fetch_add(evicted as u64, Ordering::Relaxed);
        evicted
    }

    /// Return aggregate statistics.
    pub fn stats(&self, now_ms: u64) -> LockStats {
        let locks = self
            .locks
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let total_locks = locks.len();
        let active_locks = locks
            .values()
            .filter(|e| e.is_locked && !e.version.is_expired(now_ms))
            .count();
        let expired_evicted = self.evicted_total.load(Ordering::Relaxed) as usize;
        LockStats {
            total_locks,
            active_locks,
            expired_evicted,
        }
    }
}

impl Default for DistributedLock {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn acquire_returns_version() {
        let dl = DistributedLock::new();
        let v = dl.acquire("res1", "alice", 5000, 1000).unwrap();
        assert!(v > 0);
        assert!(dl.is_locked("res1", 1000));
        assert_eq!(dl.lock_holder("res1", 1000).as_deref(), Some("alice"));
    }

    #[test]
    fn double_acquire_fails() {
        let dl = DistributedLock::new();
        dl.acquire("res1", "alice", 5000, 1000).unwrap();
        let err = dl.acquire("res1", "bob", 5000, 1001).unwrap_err();
        assert!(matches!(err, LockError::AlreadyLocked { .. }));
    }

    #[test]
    fn release_by_wrong_holder_fails() {
        let dl = DistributedLock::new();
        let v = dl.acquire("res1", "alice", 5000, 1000).unwrap();
        let err = dl.release("res1", "bob", v).unwrap_err();
        assert_eq!(err, LockError::NotOwner);
    }

    #[test]
    fn expired_lock_can_be_reacquired() {
        let dl = DistributedLock::new();
        // Acquire with ttl=100ms at t=0; re-acquire at t=200 (expired).
        dl.acquire("res1", "alice", 100, 0).unwrap();
        let v2 = dl.acquire("res1", "bob", 5000, 200).unwrap();
        assert!(v2 > 0);
        assert_eq!(dl.lock_holder("res1", 200).as_deref(), Some("bob"));
    }

    #[test]
    fn cas_version_conflict() {
        let dl = DistributedLock::new();
        let v = dl.acquire("res1", "alice", 5000, 0).unwrap();
        // Release so CAS can try to acquire, but pass wrong expected_version.
        dl.release("res1", "alice", v).unwrap();
        let err = dl
            .compare_and_swap("res1", 999, "bob", 5000, 100)
            .unwrap_err();
        assert!(matches!(err, LockError::VersionConflict { .. }));
    }

    #[test]
    fn renew_extends_ttl() {
        let dl = DistributedLock::new();
        let v = dl.acquire("res1", "alice", 100, 0).unwrap();
        // At t=50 (not expired), renew with 10000ms TTL.
        let v2 = dl.renew("res1", "alice", v, 10_000, 50).unwrap();
        assert_ne!(v, v2);
        // Now at t=200 the lock should still be valid (renewed at t=50 with 10000ms).
        assert!(dl.is_locked("res1", 200));
    }

    #[test]
    fn evict_expired_removes_and_counts() {
        let dl = DistributedLock::new();
        dl.acquire("r1", "a", 100, 0).unwrap();
        dl.acquire("r2", "b", 100, 0).unwrap();
        dl.acquire("r3", "c", 5000, 0).unwrap();
        // At t=200, r1 and r2 are expired.
        let count = dl.evict_expired(200);
        assert_eq!(count, 2);
        assert!(!dl.is_locked("r1", 200));
        assert!(!dl.is_locked("r2", 200));
        assert!(dl.is_locked("r3", 200));
    }
}

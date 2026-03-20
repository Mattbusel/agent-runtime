//! Core identifier types, available without feature gates.
//!
//! [`AgentId`] and [`MemoryId`] are defined here so they can be used by the
//! agent runtime regardless of whether the `memory` feature is enabled.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Stable identifier for an agent instance.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId(pub String);

impl AgentId {
    /// Create a new `AgentId` from any string-like value.
    ///
    /// # Panics (debug only)
    ///
    /// Triggers a `debug_assert!` if `id` is empty.  In release builds a
    /// `tracing::warn!` is emitted instead so that the misconfiguration is
    /// surfaced in production logs without aborting the process.
    pub fn new(id: impl Into<String>) -> Self {
        let id = id.into();
        if id.is_empty() {
            debug_assert!(false, "AgentId must not be empty");
            tracing::warn!("AgentId::new called with an empty string — agent IDs should be non-empty to avoid lookup ambiguity");
        }
        Self(id)
    }

    /// Create a validated `AgentId`, returning an error if `id` is empty.
    ///
    /// Prefer this constructor in user-facing code where empty IDs must be
    /// rejected explicitly rather than silently warned about.
    pub fn try_new(id: impl Into<String>) -> Result<Self, crate::error::AgentRuntimeError> {
        let id = id.into();
        if id.is_empty() {
            return Err(crate::error::AgentRuntimeError::Memory(
                "AgentId must not be empty".into(),
            ));
        }
        Ok(Self(id))
    }

    /// Generate a random `AgentId` backed by a UUID v4.
    pub fn random() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Return the inner ID string as a `&str`.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Return the byte length of the inner ID string.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Return `true` if the inner ID string is empty.
    ///
    /// Note: `AgentId::new` warns (debug: asserts) against empty IDs.
    /// This predicate is provided for defensive checks.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Return `true` if the inner ID string starts with `prefix`.
    pub fn starts_with(&self, prefix: &str) -> bool {
        self.0.starts_with(prefix)
    }
}

impl AsRef<str> for AgentId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for AgentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Stable identifier for a memory item.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemoryId(pub String);

impl MemoryId {
    /// Create a new `MemoryId` from any string-like value.
    ///
    /// # Panics (debug only)
    ///
    /// Triggers a `debug_assert!` if `id` is empty.  In release builds a
    /// `tracing::warn!` is emitted instead so that the misconfiguration is
    /// surfaced in production logs without aborting the process.
    pub fn new(id: impl Into<String>) -> Self {
        let id = id.into();
        if id.is_empty() {
            debug_assert!(false, "MemoryId must not be empty");
            tracing::warn!("MemoryId::new called with an empty string — memory IDs should be non-empty to avoid lookup ambiguity");
        }
        Self(id)
    }

    /// Create a validated `MemoryId`, returning an error if `id` is empty.
    pub fn try_new(id: impl Into<String>) -> Result<Self, crate::error::AgentRuntimeError> {
        let id = id.into();
        if id.is_empty() {
            return Err(crate::error::AgentRuntimeError::Memory(
                "MemoryId must not be empty".into(),
            ));
        }
        Ok(Self(id))
    }

    /// Generate a random `MemoryId` backed by a UUID v4.
    pub fn random() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Return the inner ID string as a `&str`.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Return the byte length of the inner ID string.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Return `true` if the inner ID string is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Return `true` if the inner ID string starts with `prefix`.
    pub fn starts_with(&self, prefix: &str) -> bool {
        self.0.starts_with(prefix)
    }
}

impl AsRef<str> for MemoryId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for MemoryId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_id_new_stores_value() {
        let id = AgentId::new("agent-1");
        assert_eq!(id.as_str(), "agent-1");
    }

    #[test]
    fn test_agent_id_try_new_rejects_empty() {
        assert!(AgentId::try_new("").is_err());
    }

    #[test]
    fn test_agent_id_try_new_accepts_nonempty() {
        let id = AgentId::try_new("ok").unwrap();
        assert_eq!(id.as_str(), "ok");
    }

    #[test]
    fn test_agent_id_random_generates_unique_ids() {
        let a = AgentId::random();
        let b = AgentId::random();
        assert_ne!(a, b);
    }

    #[test]
    fn test_agent_id_len_and_is_empty() {
        let id = AgentId::new("abc");
        assert_eq!(id.len(), 3);
        assert!(!id.is_empty());
    }

    #[test]
    fn test_agent_id_display() {
        let id = AgentId::new("my-agent");
        assert_eq!(id.to_string(), "my-agent");
    }

    #[test]
    fn test_memory_id_new_stores_value() {
        let id = MemoryId::new("mem-42");
        assert_eq!(id.as_str(), "mem-42");
    }

    #[test]
    fn test_memory_id_try_new_rejects_empty() {
        assert!(MemoryId::try_new("").is_err());
    }

    #[test]
    fn test_memory_id_len_and_is_empty() {
        let id = MemoryId::new("hello");
        assert_eq!(id.len(), 5);
        assert!(!id.is_empty());
    }

    #[test]
    fn test_memory_id_display() {
        let id = MemoryId::new("mem-id");
        assert_eq!(id.to_string(), "mem-id");
    }

    #[test]
    fn test_memory_id_random_generates_unique_ids() {
        let a = MemoryId::random();
        let b = MemoryId::random();
        assert_ne!(a, b);
    }

    // ── Round 11: starts_with ─────────────────────────────────────────────────

    #[test]
    fn test_agent_id_starts_with_matching_prefix() {
        let id = AgentId::new("agent-001");
        assert!(id.starts_with("agent-"));
        assert!(!id.starts_with("user-"));
    }

    #[test]
    fn test_agent_id_starts_with_empty_prefix_always_true() {
        let id = AgentId::new("anything");
        assert!(id.starts_with(""));
    }

    #[test]
    fn test_memory_id_starts_with_matching_prefix() {
        let id = MemoryId::new("mem-42");
        assert!(id.starts_with("mem-"));
        assert!(!id.starts_with("ep-"));
    }
}

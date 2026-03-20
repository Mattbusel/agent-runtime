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

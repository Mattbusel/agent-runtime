//! # Module: capability
//!
//! ## Responsibility
//! Agent capability registry and capability negotiation.  Agents declare a set
//! of [`Capability`] values that describe what they can do; the registry lets
//! other agents discover and negotiate compatible peers at runtime.
//!
//! ## Design
//! - [`Capability`] carries a SemVer-style `(major, minor, patch)` version.
//! - [`CapabilityRegistry`] is a plain `HashMap`-backed store.
//! - [`negotiate`] compares a set of *offered* capabilities against a list of
//!   *required* capability names and returns a [`NegotiationResult`]:
//!   - **Compatible** — all required capabilities were found with a compatible
//!     version (same major, offered minor ≥ required minor).
//!   - **Incompatible** — reports which names were missing or version-
//!     mismatched.
//! - [`CapabilitySet`] wraps a collection of capabilities and supports set
//!   intersection.
//!
//! ## Version compatibility rule
//! An offered capability satisfies a requirement when:
//! ```text
//! offered.major == required.major  AND  offered.minor >= required.minor
//! ```

use std::collections::HashMap;

// ── ParamType ─────────────────────────────────────────────────────────────────

/// The primitive type of a capability parameter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParamType {
    /// UTF-8 text value.
    String,
    /// 64-bit signed integer.
    Integer,
    /// 64-bit floating-point number.
    Float,
    /// Boolean flag.
    Boolean,
    /// Ordered list of strings.
    List,
}

// ── CapabilityParam ───────────────────────────────────────────────────────────

/// A single named parameter accepted by a capability.
#[derive(Debug, Clone)]
pub struct CapabilityParam {
    /// Parameter name (unique within its capability).
    pub name: std::string::String,
    /// Expected value type.
    pub param_type: ParamType,
    /// Whether the caller must supply this parameter.
    pub required: bool,
    /// Serialised default value, if any.
    pub default: Option<std::string::String>,
}

// ── Capability ────────────────────────────────────────────────────────────────

/// A self-describing capability that an agent declares it can perform.
#[derive(Debug, Clone)]
pub struct Capability {
    /// Short, stable machine-readable name (e.g. `"text-completion"`).
    pub name: std::string::String,
    /// SemVer-style version: `(major, minor, patch)`.
    pub version: (u32, u32, u32),
    /// Human-readable description.
    pub description: std::string::String,
    /// Ordered list of parameters accepted by this capability.
    pub params: Vec<CapabilityParam>,
}

impl Capability {
    /// Built-in: text completion capability at version 1.0.0.
    pub fn text_completion() -> Self {
        Self {
            name: "text-completion".into(),
            version: (1, 0, 0),
            description: "Generate text completions given a prompt.".into(),
            params: vec![
                CapabilityParam {
                    name: "prompt".into(),
                    param_type: ParamType::String,
                    required: true,
                    default: None,
                },
                CapabilityParam {
                    name: "max_tokens".into(),
                    param_type: ParamType::Integer,
                    required: false,
                    default: Some("256".into()),
                },
            ],
        }
    }

    /// Built-in: tool-use capability at version 1.0.0.
    pub fn tool_use() -> Self {
        Self {
            name: "tool-use".into(),
            version: (1, 0, 0),
            description: "Invoke registered tools by name with structured arguments.".into(),
            params: vec![CapabilityParam {
                name: "tool_name".into(),
                param_type: ParamType::String,
                required: true,
                default: None,
            }],
        }
    }

    /// Built-in: memory access capability at version 1.0.0.
    pub fn memory_access() -> Self {
        Self {
            name: "memory-access".into(),
            version: (1, 0, 0),
            description: "Read from and write to the agent's episodic and semantic memory.".into(),
            params: vec![CapabilityParam {
                name: "query".into(),
                param_type: ParamType::String,
                required: true,
                default: None,
            }],
        }
    }

    /// Built-in: streaming output capability at version 1.0.0.
    pub fn streaming() -> Self {
        Self {
            name: "streaming".into(),
            version: (1, 0, 0),
            description: "Emit incremental token-level output over a streaming channel.".into(),
            params: vec![],
        }
    }

    /// Returns `true` when `self` satisfies the version requirement of
    /// `required`: same major version and `self.minor >= required.minor`.
    pub fn satisfies_version(&self, required: &Self) -> bool {
        self.version.0 == required.version.0 && self.version.1 >= required.version.1
    }
}

// ── CapabilityRegistry ────────────────────────────────────────────────────────

/// A simple name-keyed registry of capabilities.
#[derive(Debug, Default)]
pub struct CapabilityRegistry {
    capabilities: HashMap<std::string::String, Capability>,
}

impl CapabilityRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers `cap`, replacing any previous capability with the same name.
    pub fn register(&mut self, cap: Capability) {
        self.capabilities.insert(cap.name.clone(), cap);
    }

    /// Looks up a capability by name.
    pub fn lookup(&self, name: &str) -> Option<&Capability> {
        self.capabilities.get(name)
    }

    /// Returns an iterator over all registered capabilities.
    pub fn iter(&self) -> impl Iterator<Item = &Capability> {
        self.capabilities.values()
    }

    /// Returns the number of registered capabilities.
    pub fn len(&self) -> usize {
        self.capabilities.len()
    }

    /// Returns `true` when no capabilities are registered.
    pub fn is_empty(&self) -> bool {
        self.capabilities.is_empty()
    }
}

// ── NegotiationResult ─────────────────────────────────────────────────────────

/// Outcome of a capability negotiation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NegotiationResult {
    /// All required capabilities are covered.
    Compatible {
        /// Names of the capabilities that were successfully matched.
        matched: Vec<std::string::String>,
    },
    /// One or more required capabilities could not be satisfied.
    Incompatible {
        /// Required capability names that were not present at all.
        missing: Vec<std::string::String>,
        /// Required capability names that were present but at an incompatible version.
        version_mismatched: Vec<std::string::String>,
    },
}

impl NegotiationResult {
    /// Returns `true` when the result is [`Compatible`][NegotiationResult::Compatible].
    pub fn is_compatible(&self) -> bool {
        matches!(self, NegotiationResult::Compatible { .. })
    }
}

// ── negotiate ─────────────────────────────────────────────────────────────────

/// Negotiate whether `offered` capabilities satisfy every name in `required`.
///
/// `required` is a slice of capability names.  Version requirements are
/// expressed by passing a slice of [`Capability`] values that carry the
/// minimum acceptable version.  This function compares by name first, then
/// checks version compatibility.
///
/// # Compatibility rule
/// ```text
/// offered.major == required.major  AND  offered.minor >= required.minor
/// ```
pub fn negotiate(offered: &[Capability], required: &[Capability]) -> NegotiationResult {
    let offered_map: HashMap<&str, &Capability> =
        offered.iter().map(|c| (c.name.as_str(), c)).collect();

    let mut matched = Vec::new();
    let mut missing = Vec::new();
    let mut version_mismatched = Vec::new();

    for req in required {
        match offered_map.get(req.name.as_str()) {
            None => missing.push(req.name.clone()),
            Some(offered_cap) => {
                if offered_cap.satisfies_version(req) {
                    matched.push(req.name.clone());
                } else {
                    version_mismatched.push(req.name.clone());
                }
            }
        }
    }

    if missing.is_empty() && version_mismatched.is_empty() {
        NegotiationResult::Compatible { matched }
    } else {
        NegotiationResult::Incompatible {
            missing,
            version_mismatched,
        }
    }
}

/// Negotiate by name only (no version requirement).
///
/// Returns `Compatible` when every string in `required_names` appears in
/// `offered`, or `Incompatible` listing the absent names.
pub fn negotiate_by_name(offered: &[Capability], required_names: &[std::string::String]) -> NegotiationResult {
    let offered_names: std::collections::HashSet<&str> =
        offered.iter().map(|c| c.name.as_str()).collect();

    let mut matched = Vec::new();
    let mut missing = Vec::new();

    for name in required_names {
        if offered_names.contains(name.as_str()) {
            matched.push(name.clone());
        } else {
            missing.push(name.clone());
        }
    }

    if missing.is_empty() {
        NegotiationResult::Compatible { matched }
    } else {
        NegotiationResult::Incompatible {
            missing,
            version_mismatched: vec![],
        }
    }
}

// ── CapabilitySet ─────────────────────────────────────────────────────────────

/// A named set of capabilities that an agent declares.
///
/// Use [`CapabilitySet::intersect`] to find common capabilities between two
/// agents.
#[derive(Debug, Clone, Default)]
pub struct CapabilitySet {
    capabilities: HashMap<std::string::String, Capability>,
}

impl CapabilitySet {
    /// Creates an empty set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts `cap` into the set (replaces any with the same name).
    pub fn insert(&mut self, cap: Capability) {
        self.capabilities.insert(cap.name.clone(), cap);
    }

    /// Returns `true` when the set contains a capability with `name`.
    pub fn contains(&self, name: &str) -> bool {
        self.capabilities.contains_key(name)
    }

    /// Returns the capability with `name`, if present.
    pub fn get(&self, name: &str) -> Option<&Capability> {
        self.capabilities.get(name)
    }

    /// Returns the number of capabilities in the set.
    pub fn len(&self) -> usize {
        self.capabilities.len()
    }

    /// Returns `true` when the set is empty.
    pub fn is_empty(&self) -> bool {
        self.capabilities.is_empty()
    }

    /// Returns an iterator over all capabilities in the set.
    pub fn iter(&self) -> impl Iterator<Item = &Capability> {
        self.capabilities.values()
    }

    /// Returns a new [`CapabilitySet`] containing only capabilities whose
    /// name appears in **both** `self` and `other`.  The version taken is from
    /// `self`.
    pub fn intersect(&self, other: &CapabilitySet) -> CapabilitySet {
        let mut result = CapabilitySet::new();
        for (name, cap) in &self.capabilities {
            if other.capabilities.contains_key(name) {
                result.insert(cap.clone());
            }
        }
        result
    }
}

impl FromIterator<Capability> for CapabilitySet {
    fn from_iter<I: IntoIterator<Item = Capability>>(iter: I) -> Self {
        let mut set = Self::new();
        for cap in iter {
            set.insert(cap);
        }
        set
    }
}

// ── Unit tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    fn make_cap(name: &str, major: u32, minor: u32) -> Capability {
        Capability {
            name: name.into(),
            version: (major, minor, 0),
            description: format!("{name} capability"),
            params: vec![],
        }
    }

    // ── Registry ──────────────────────────────────────────────────────────────

    #[test]
    fn test_register_and_lookup() {
        let mut reg = CapabilityRegistry::new();
        reg.register(Capability::text_completion());
        let found = reg.lookup("text-completion");
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, "text-completion");
    }

    #[test]
    fn test_lookup_missing_returns_none() {
        let reg = CapabilityRegistry::new();
        assert!(reg.lookup("nonexistent").is_none());
    }

    #[test]
    fn test_register_replaces_existing() {
        let mut reg = CapabilityRegistry::new();
        reg.register(make_cap("foo", 1, 0));
        reg.register(make_cap("foo", 2, 0));
        assert_eq!(reg.lookup("foo").unwrap().version.0, 2);
    }

    // ── negotiate ─────────────────────────────────────────────────────────────

    #[test]
    fn test_negotiate_all_compatible() {
        let offered = vec![
            make_cap("text-completion", 1, 2),
            make_cap("tool-use", 1, 0),
        ];
        let required = vec![
            make_cap("text-completion", 1, 1), // minor 1 <= 2 ✓
            make_cap("tool-use", 1, 0),
        ];
        let result = negotiate(&offered, &required);
        assert!(result.is_compatible());
        if let NegotiationResult::Compatible { matched } = result {
            assert_eq!(matched.len(), 2);
        }
    }

    #[test]
    fn test_negotiate_missing_capability() {
        let offered = vec![make_cap("text-completion", 1, 0)];
        let required = vec![
            make_cap("text-completion", 1, 0),
            make_cap("streaming", 1, 0),
        ];
        let result = negotiate(&offered, &required);
        assert!(!result.is_compatible());
        if let NegotiationResult::Incompatible { missing, .. } = result {
            assert!(missing.contains(&"streaming".to_string()));
        }
    }

    #[test]
    fn test_negotiate_version_mismatch_major() {
        let offered = vec![make_cap("text-completion", 2, 0)];
        let required = vec![make_cap("text-completion", 1, 0)];
        let result = negotiate(&offered, &required);
        assert!(!result.is_compatible());
        if let NegotiationResult::Incompatible {
            version_mismatched, ..
        } = result
        {
            assert!(version_mismatched.contains(&"text-completion".to_string()));
        }
    }

    #[test]
    fn test_negotiate_version_mismatch_minor() {
        // Offered minor (0) < required minor (5) — incompatible.
        let offered = vec![make_cap("text-completion", 1, 0)];
        let required = vec![make_cap("text-completion", 1, 5)];
        let result = negotiate(&offered, &required);
        assert!(!result.is_compatible());
    }

    #[test]
    fn test_negotiate_higher_minor_is_compatible() {
        let offered = vec![make_cap("text-completion", 1, 9)];
        let required = vec![make_cap("text-completion", 1, 3)];
        assert!(negotiate(&offered, &required).is_compatible());
    }

    // ── CapabilitySet ─────────────────────────────────────────────────────────

    #[test]
    fn test_capability_set_intersect() {
        let mut a = CapabilitySet::new();
        a.insert(Capability::text_completion());
        a.insert(Capability::tool_use());
        a.insert(Capability::streaming());

        let mut b = CapabilitySet::new();
        b.insert(Capability::text_completion());
        b.insert(Capability::memory_access());

        let common = a.intersect(&b);
        assert_eq!(common.len(), 1);
        assert!(common.contains("text-completion"));
        assert!(!common.contains("tool-use"));
        assert!(!common.contains("memory-access"));
    }

    #[test]
    fn test_capability_set_empty_intersection() {
        let mut a = CapabilitySet::new();
        a.insert(Capability::tool_use());

        let b = CapabilitySet::new();

        let common = a.intersect(&b);
        assert!(common.is_empty());
    }

    #[test]
    fn test_builtin_capabilities_are_valid() {
        let caps = vec![
            Capability::text_completion(),
            Capability::tool_use(),
            Capability::memory_access(),
            Capability::streaming(),
        ];
        for cap in &caps {
            assert!(!cap.name.is_empty());
            assert!(!cap.description.is_empty());
        }
    }

    #[test]
    fn test_negotiate_by_name_compatible() {
        let offered = vec![Capability::text_completion(), Capability::tool_use()];
        let required = vec!["text-completion".to_string()];
        let result = negotiate_by_name(&offered, &required);
        assert!(result.is_compatible());
    }

    #[test]
    fn test_negotiate_by_name_missing() {
        let offered = vec![Capability::text_completion()];
        let required = vec!["streaming".to_string()];
        let result = negotiate_by_name(&offered, &required);
        assert!(!result.is_compatible());
    }
}

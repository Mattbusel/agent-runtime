//! # Policy Enforcement — RBAC-lite with Allow/Deny Rules
//!
//! ## Responsibility
//! Evaluate whether a [`Principal`] is permitted to perform an `action` on a
//! `resource`, using a store of [`Role`]s and [`PolicyRule`]s.
//!
//! ## Evaluation order
//! 1. Collect all rules whose `principals`, `resources`, and `actions` match
//!    the request (glob matching supported).
//! 2. If **any** matching rule has effect [`PolicyEffect::Deny`], return
//!    [`AuthResult::Denied`] immediately — explicit Deny always wins.
//! 3. If **any** matching rule has effect [`PolicyEffect::Allow`], return
//!    [`AuthResult::Allowed`].
//! 4. If no rules match, return [`AuthResult::Denied`] (default-deny).
//!
//! ## Glob matching
//! - `*`  — matches any single path segment (no `/`).
//! - `**` — matches any sequence of segments (including none).

use std::collections::{HashMap, HashSet};
use std::time::Instant;

// ── Permission ────────────────────────────────────────────────────────────────

/// A discrete permission: a `(resource, action)` pair.
///
/// # Example
/// ```
/// use llm_agent_runtime::policy::Permission;
/// let p = Permission { resource: "tool:calculator".into(), action: "execute".into() };
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Permission {
    /// The resource being accessed, e.g. `"tool:calculator"` or `"db:users"`.
    pub resource: String,
    /// The action being performed, e.g. `"execute"`, `"read"`, `"write"`.
    pub action: String,
}

// ── Role ──────────────────────────────────────────────────────────────────────

/// A named role that carries a set of permissions and may inherit from parent roles.
#[derive(Debug, Clone)]
pub struct Role {
    /// Unique name for this role, e.g. `"admin"` or `"reader"`.
    pub name: String,
    /// Permissions granted directly by this role.
    pub permissions: Vec<Permission>,
    /// Names of roles whose permissions this role inherits.
    pub parent_roles: Vec<String>,
}

// ── Principal ────────────────────────────────────────────────────────────────

/// An authenticated entity making requests.
#[derive(Debug, Clone)]
pub struct Principal {
    /// Unique identifier (e.g. user ID or agent ID).
    pub id: String,
    /// Role names assigned to this principal.
    pub roles: Vec<String>,
    /// Arbitrary key-value attributes (e.g. `department`, `clearance_level`).
    pub attributes: HashMap<String, String>,
}

// ── PolicyEffect ─────────────────────────────────────────────────────────────

/// Whether a rule grants or denies access.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyEffect {
    /// Grant access.
    Allow,
    /// Deny access. Deny rules always take priority over Allow rules.
    Deny,
}

// ── PolicyCondition ──────────────────────────────────────────────────────────

/// An additional condition that must hold for a rule to match.
#[derive(Debug, Clone)]
pub enum PolicyCondition {
    /// The principal must have the given attribute set to the given value.
    AttributeEquals {
        /// Attribute key.
        key: String,
        /// Required value.
        value: String,
    },
    /// The current UTC hour must be in `[hour_start, hour_end)`.
    TimeOfDay {
        /// Inclusive start hour (0–23).
        hour_start: u8,
        /// Exclusive end hour (0–23).
        hour_end: u8,
    },
    /// The principal+resource pair must not exceed `max_per_minute` calls in
    /// the last 60 seconds. Counters are maintained by [`PolicyEngine`].
    RateLimit {
        /// Maximum number of requests per minute.
        max_per_minute: u32,
    },
}

// ── PolicyRule ───────────────────────────────────────────────────────────────

/// A single access-control rule.
#[derive(Debug, Clone)]
pub struct PolicyRule {
    /// Unique rule identifier.
    pub id: String,
    /// Whether this rule grants or denies access.
    pub effect: PolicyEffect,
    /// Principal IDs this rule applies to. Use `"*"` to match all principals.
    pub principals: Vec<String>,
    /// Resource glob patterns this rule applies to.
    pub resources: Vec<String>,
    /// Action patterns this rule applies to. Use `"*"` to match all actions.
    pub actions: Vec<String>,
    /// Additional conditions that must all hold for this rule to match.
    pub conditions: Vec<PolicyCondition>,
}

// ── AuthResult ────────────────────────────────────────────────────────────────

/// The outcome of an authorization check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthResult {
    /// The request is permitted.
    Allowed,
    /// The request is denied, with a human-readable reason.
    Denied {
        /// Human-readable explanation.
        reason: String,
    },
}

// ── PolicyStore ──────────────────────────────────────────────────────────────

/// Holds the role hierarchy and the ordered set of policy rules.
#[derive(Debug, Default)]
pub struct PolicyStore {
    /// All known roles, keyed by name.
    pub roles: HashMap<String, Role>,
    /// Policy rules evaluated in insertion order.
    pub rules: Vec<PolicyRule>,
}

impl PolicyStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a role. Overwrites any existing role with the same name.
    pub fn add_role(&mut self, role: Role) {
        self.roles.insert(role.name.clone(), role);
    }

    /// Append a policy rule.
    pub fn add_rule(&mut self, rule: PolicyRule) {
        self.rules.push(rule);
    }

    /// Collect all permissions reachable by `principal` through the role
    /// hierarchy, de-duplicated. Cycles in the role graph are handled via a
    /// visited set.
    pub fn get_effective_permissions(&self, principal: &Principal) -> Vec<Permission> {
        let mut visited: HashSet<String> = HashSet::new();
        let mut permissions: Vec<Permission> = Vec::new();
        for role_name in &principal.roles {
            self.collect_permissions(role_name, &mut visited, &mut permissions);
        }
        permissions
    }

    // Recursive DFS through the role hierarchy.
    fn collect_permissions(
        &self,
        role_name: &str,
        visited: &mut HashSet<String>,
        out: &mut Vec<Permission>,
    ) {
        if !visited.insert(role_name.to_owned()) {
            return; // already visited — break the cycle
        }
        if let Some(role) = self.roles.get(role_name) {
            for perm in &role.permissions {
                if !out.contains(perm) {
                    out.push(perm.clone());
                }
            }
            for parent in &role.parent_roles {
                self.collect_permissions(parent, visited, out);
            }
        }
    }
}

// ── Glob matching ─────────────────────────────────────────────────────────────

/// Return `true` if `text` matches the glob `pattern`.
///
/// Rules:
/// - `**` matches any sequence of characters (including `/`).
/// - `*`  matches any sequence of characters except `/`.
/// - All other characters match literally.
fn glob_match(pattern: &str, text: &str) -> bool {
    glob_match_inner(pattern.as_bytes(), text.as_bytes())
}

fn glob_match_inner(pat: &[u8], text: &[u8]) -> bool {
    // We use an iterative approach to avoid stack overflows on long inputs.
    // `pi` = pattern index, `ti` = text index.
    // `star_pi` / `star_ti` = saved position of the last `*` wildcard.
    let mut pi = 0usize;
    let mut ti = 0usize;
    // For backtracking on single `*`.
    let mut star_pi: Option<usize> = None;
    let mut star_ti: usize = 0;

    loop {
        // Handle `**` — matches any sequence including `/`.
        while pi + 1 < pat.len() && pat[pi] == b'*' && pat[pi + 1] == b'*' {
            pi += 2;
            // Skip optional `/` separator after `**`.
            if pi < pat.len() && pat[pi] == b'/' {
                pi += 1;
            }
            // `**` saves a backtrack point that CAN cross `/`.
            // We handle it by trying all text positions from current ti onward.
            // Store state: try rest of pattern at each remaining text position.
            for advance in 0..=text.len().saturating_sub(ti) {
                if glob_match_inner(&pat[pi..], &text[ti + advance..]) {
                    return true;
                }
            }
            return false;
        }

        if pi < pat.len() && pat[pi] == b'*' {
            // Single `*` — matches any sequence of non-`/` characters.
            star_pi = Some(pi + 1); // pattern position after `*`
            star_ti = ti;           // text position where `*` started consuming
            pi += 1;
            continue;
        }

        if pi < pat.len() && ti < text.len() && pat[pi] == text[ti] {
            pi += 1;
            ti += 1;
            continue;
        }

        // Mismatch or pattern exhausted before text.
        if let Some(spi) = star_pi {
            // Can we advance the text by one (non-`/`) character and retry?
            if ti < text.len() && text[ti] != b'/' {
                // Backtrack: let `*` consume one more character.
                star_ti += 1;
                ti = star_ti;
                pi = spi;
                continue;
            }
        }

        // No backtrack available.
        break;
    }

    // Success if both pattern and text are fully consumed.
    pi == pat.len() && ti == text.len()
}

// ── PolicyEngine ─────────────────────────────────────────────────────────────

/// Evaluates authorization requests against a [`PolicyStore`].
///
/// # Rate limiting
/// Rate-limit counters are stored in-process. Each `(principal_id, resource)`
/// pair has an independent counter that resets every 60 seconds.
pub struct PolicyEngine {
    /// The backing store of roles and rules.
    pub store: PolicyStore,
    /// Rate-limit counters: key = `(principal_id, resource)`, value = `(count, window_start)`.
    rate_counters: HashMap<(String, String), (u32, Instant)>,
}

impl PolicyEngine {
    /// Create a new engine wrapping `store`.
    pub fn new(store: PolicyStore) -> Self {
        Self {
            store,
            rate_counters: HashMap::new(),
        }
    }

    /// Evaluate whether `principal` may perform `action` on `resource`.
    ///
    /// Deny rules take absolute priority over Allow rules.
    pub fn authorize(
        &mut self,
        principal: &Principal,
        resource: &str,
        action: &str,
    ) -> AuthResult {
        let mut any_allow = false;

        for rule in &self.store.rules.clone() {
            if !self.rule_matches_principal(rule, principal) {
                continue;
            }
            if !rule.resources.iter().any(|r| glob_match(r, resource)) {
                continue;
            }
            if !rule.actions.iter().any(|a| a == "*" || glob_match(a, action)) {
                continue;
            }
            if !self.conditions_hold(rule, principal, resource) {
                continue;
            }

            match rule.effect {
                PolicyEffect::Deny => {
                    return AuthResult::Denied {
                        reason: format!("denied by rule '{}'", rule.id),
                    };
                }
                PolicyEffect::Allow => {
                    any_allow = true;
                }
            }
        }

        if any_allow {
            // Bump rate counters for all matching rate-limit conditions.
            self.bump_rate_counters(principal, resource);
            AuthResult::Allowed
        } else {
            AuthResult::Denied {
                reason: "no allow rule matched".to_owned(),
            }
        }
    }

    // ── helpers ───────────────────────────────────────────────────────────

    fn rule_matches_principal(&self, rule: &PolicyRule, principal: &Principal) -> bool {
        rule.principals.iter().any(|p| p == "*" || p == &principal.id)
    }

    fn conditions_hold(
        &mut self,
        rule: &PolicyRule,
        principal: &Principal,
        resource: &str,
    ) -> bool {
        for cond in &rule.conditions.clone() {
            match cond {
                PolicyCondition::AttributeEquals { key, value } => {
                    if principal.attributes.get(key).map(|s| s.as_str()) != Some(value) {
                        return false;
                    }
                }
                PolicyCondition::TimeOfDay {
                    hour_start,
                    hour_end,
                } => {
                    let hour = current_utc_hour();
                    if !(hour >= *hour_start && hour < *hour_end) {
                        return false;
                    }
                }
                PolicyCondition::RateLimit { max_per_minute } => {
                    let key = (principal.id.clone(), resource.to_owned());
                    let now = Instant::now();
                    let entry = self.rate_counters.entry(key).or_insert((0, now));
                    // Reset window if 60 s have elapsed.
                    if now.duration_since(entry.1).as_secs() >= 60 {
                        *entry = (0, now);
                    }
                    if entry.0 >= *max_per_minute {
                        return false;
                    }
                }
            }
        }
        true
    }

    fn bump_rate_counters(&mut self, principal: &Principal, resource: &str) {
        let key = (principal.id.clone(), resource.to_owned());
        let now = Instant::now();
        let entry = self.rate_counters.entry(key).or_insert((0, now));
        if now.duration_since(entry.1).as_secs() >= 60 {
            *entry = (1, now);
        } else {
            entry.0 += 1;
        }
    }
}

// Return the current UTC hour (0–23).
fn current_utc_hour() -> u8 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    ((secs / 3600) % 24) as u8
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn make_principal(id: &str, roles: &[&str]) -> Principal {
        Principal {
            id: id.to_owned(),
            roles: roles.iter().map(|r| r.to_string()).collect(),
            attributes: HashMap::new(),
        }
    }

    fn allow_rule(id: &str, principals: &[&str], resources: &[&str], actions: &[&str]) -> PolicyRule {
        PolicyRule {
            id: id.to_owned(),
            effect: PolicyEffect::Allow,
            principals: principals.iter().map(|s| s.to_string()).collect(),
            resources: resources.iter().map(|s| s.to_string()).collect(),
            actions: actions.iter().map(|s| s.to_string()).collect(),
            conditions: vec![],
        }
    }

    fn deny_rule(id: &str, principals: &[&str], resources: &[&str], actions: &[&str]) -> PolicyRule {
        PolicyRule {
            id: id.to_owned(),
            effect: PolicyEffect::Deny,
            principals: principals.iter().map(|s| s.to_string()).collect(),
            resources: resources.iter().map(|s| s.to_string()).collect(),
            actions: actions.iter().map(|s| s.to_string()).collect(),
            conditions: vec![],
        }
    }

    #[test]
    fn allow_rule_grants_access() {
        let mut store = PolicyStore::new();
        store.add_rule(allow_rule("r1", &["alice"], &["tool:calculator"], &["execute"]));
        let mut engine = PolicyEngine::new(store);
        let alice = make_principal("alice", &[]);
        assert_eq!(engine.authorize(&alice, "tool:calculator", "execute"), AuthResult::Allowed);
    }

    #[test]
    fn no_rule_denies() {
        let store = PolicyStore::new();
        let mut engine = PolicyEngine::new(store);
        let bob = make_principal("bob", &[]);
        assert!(matches!(engine.authorize(&bob, "tool:calc", "execute"), AuthResult::Denied { .. }));
    }

    #[test]
    fn deny_overrides_allow() {
        let mut store = PolicyStore::new();
        store.add_rule(allow_rule("allow-all", &["*"], &["tool:*"], &["*"]));
        store.add_rule(deny_rule("deny-dave", &["dave"], &["tool:*"], &["*"]));
        let mut engine = PolicyEngine::new(store);
        let alice = make_principal("alice", &[]);
        let dave = make_principal("dave", &[]);
        assert_eq!(engine.authorize(&alice, "tool:x", "run"), AuthResult::Allowed);
        assert!(matches!(engine.authorize(&dave, "tool:x", "run"), AuthResult::Denied { .. }));
    }

    #[test]
    fn role_hierarchy() {
        let mut store = PolicyStore::new();
        // reader < editor < admin
        store.add_role(Role {
            name: "reader".to_owned(),
            permissions: vec![Permission { resource: "db:users".into(), action: "read".into() }],
            parent_roles: vec![],
        });
        store.add_role(Role {
            name: "editor".to_owned(),
            permissions: vec![Permission { resource: "db:users".into(), action: "write".into() }],
            parent_roles: vec!["reader".to_owned()],
        });

        let principal = Principal {
            id: "carol".to_owned(),
            roles: vec!["editor".to_owned()],
            attributes: HashMap::new(),
        };

        let perms = store.get_effective_permissions(&principal);
        assert!(perms.contains(&Permission { resource: "db:users".into(), action: "write".into() }));
        assert!(perms.contains(&Permission { resource: "db:users".into(), action: "read".into() }));
    }

    #[test]
    fn role_hierarchy_cycle_safe() {
        let mut store = PolicyStore::new();
        store.add_role(Role {
            name: "a".to_owned(),
            permissions: vec![],
            parent_roles: vec!["b".to_owned()],
        });
        store.add_role(Role {
            name: "b".to_owned(),
            permissions: vec![],
            parent_roles: vec!["a".to_owned()], // cycle
        });
        let p = make_principal("x", &["a"]);
        // Should not stack-overflow.
        let _ = store.get_effective_permissions(&p);
    }

    #[test]
    fn glob_matching() {
        assert!(glob_match("tool:*", "tool:calculator"));
        assert!(glob_match("tool:*", "tool:anything"));
        assert!(!glob_match("tool:*", "other:calculator"));
        assert!(glob_match("**", "a/b/c"));
        assert!(glob_match("a/**/z", "a/b/c/z"));
        assert!(glob_match("*", "hello"));
        assert!(!glob_match("*", "hello/world")); // `*` doesn't cross `/`
    }

    #[test]
    fn glob_allow_wildcard_principal() {
        let mut store = PolicyStore::new();
        store.add_rule(allow_rule("open", &["*"], &["public:*"], &["read"]));
        let mut engine = PolicyEngine::new(store);
        let anon = make_principal("anon-user", &[]);
        assert_eq!(engine.authorize(&anon, "public:docs", "read"), AuthResult::Allowed);
    }

    #[test]
    fn rate_limit_condition() {
        let mut store = PolicyStore::new();
        store.add_rule(PolicyRule {
            id: "rate-limited".to_owned(),
            effect: PolicyEffect::Allow,
            principals: vec!["*".to_owned()],
            resources: vec!["api:search".to_owned()],
            actions: vec!["query".to_owned()],
            conditions: vec![PolicyCondition::RateLimit { max_per_minute: 2 }],
        });
        let mut engine = PolicyEngine::new(store);
        let user = make_principal("user1", &[]);

        // First two calls should succeed.
        assert_eq!(engine.authorize(&user, "api:search", "query"), AuthResult::Allowed);
        assert_eq!(engine.authorize(&user, "api:search", "query"), AuthResult::Allowed);
        // Third call exceeds the limit.
        assert!(matches!(
            engine.authorize(&user, "api:search", "query"),
            AuthResult::Denied { .. }
        ));
    }

    #[test]
    fn attribute_condition() {
        let mut store = PolicyStore::new();
        store.add_rule(PolicyRule {
            id: "dept-check".to_owned(),
            effect: PolicyEffect::Allow,
            principals: vec!["*".to_owned()],
            resources: vec!["hr:records".to_owned()],
            actions: vec!["read".to_owned()],
            conditions: vec![PolicyCondition::AttributeEquals {
                key: "department".to_owned(),
                value: "hr".to_owned(),
            }],
        });
        let mut engine = PolicyEngine::new(store);

        let mut hr_user = make_principal("alice", &[]);
        hr_user.attributes.insert("department".to_owned(), "hr".to_owned());

        let mut eng_user = make_principal("bob", &[]);
        eng_user.attributes.insert("department".to_owned(), "engineering".to_owned());

        assert_eq!(engine.authorize(&hr_user, "hr:records", "read"), AuthResult::Allowed);
        assert!(matches!(engine.authorize(&eng_user, "hr:records", "read"), AuthResult::Denied { .. }));
    }
}

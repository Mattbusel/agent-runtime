//! # Built-in Skills
//!
//! Ready-to-use [`Skill`] implementations that cover common agent patterns.
//!
//! Register them with [`SkillLibrary::register_skill`]:
//!
//! ```rust
//! use llm_agent_runtime::skills::{SkillLibrary, builtins::EchoSkill};
//!
//! let mut library = SkillLibrary::new();
//! library.register_skill(Box::new(EchoSkill));
//! ```

use super::{Skill, SkillFuture};
use serde_json::{json, Value};

// ── EchoSkill ─────────────────────────────────────────────────────────────────

/// A trivial skill that echoes its input back as `{"echo": <input>}`.
///
/// Useful for testing and pipeline debugging.
pub struct EchoSkill;

impl Skill for EchoSkill {
    fn name(&self) -> &str {
        "echo"
    }

    fn description(&self) -> &str {
        "Echoes the input parameters verbatim as {\"echo\": <params>}"
    }

    fn execute(&self, params: Value) -> SkillFuture {
        Box::pin(async move { Ok(json!({"echo": params})) })
    }
}

// ── NoOpSkill ─────────────────────────────────────────────────────────────────

/// A skill that always returns `{}`.  Useful as a placeholder.
pub struct NoOpSkill {
    skill_name: String,
}

impl NoOpSkill {
    /// Create a no-op skill with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            skill_name: name.into(),
        }
    }
}

impl Skill for NoOpSkill {
    fn name(&self) -> &str {
        &self.skill_name
    }

    fn description(&self) -> &str {
        "No-op placeholder skill — always returns an empty JSON object"
    }

    fn execute(&self, _params: Value) -> SkillFuture {
        Box::pin(async move { Ok(json!({})) })
    }
}

// ── ConstantSkill ─────────────────────────────────────────────────────────────

/// A skill that always returns the same pre-configured JSON value.
pub struct ConstantSkill {
    skill_name: String,
    value: Value,
}

impl ConstantSkill {
    /// Create a constant skill.
    pub fn new(name: impl Into<String>, value: Value) -> Self {
        Self {
            skill_name: name.into(),
            value,
        }
    }
}

impl Skill for ConstantSkill {
    fn name(&self) -> &str {
        &self.skill_name
    }

    fn description(&self) -> &str {
        "Returns a fixed, pre-configured JSON value regardless of input"
    }

    fn execute(&self, _params: Value) -> SkillFuture {
        let v = self.value.clone();
        Box::pin(async move { Ok(v) })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn echo_skill() {
        let skill = EchoSkill;
        let result = skill.execute(json!({"x": 1})).await.unwrap();
        assert_eq!(result["echo"]["x"], 1);
    }

    #[tokio::test]
    async fn noop_skill() {
        let skill = NoOpSkill::new("placeholder");
        assert_eq!(skill.name(), "placeholder");
        let result = skill.execute(json!({"any": "data"})).await.unwrap();
        assert_eq!(result, json!({}));
    }

    #[tokio::test]
    async fn constant_skill() {
        let skill = ConstantSkill::new("always_42", json!({"answer": 42}));
        let result = skill.execute(json!({})).await.unwrap();
        assert_eq!(result["answer"], 42);
    }
}

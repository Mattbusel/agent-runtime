//! # Module: Skill Library
//!
//! ## Responsibility
//! Skills are named, parameterised sequences of tool calls that can be
//! defined in TOML and reused across agents.  A [`SkillLibrary`] manages the
//! registry; individual skills implement the [`Skill`] trait.
//!
//! ## Example TOML
//!
//! ```toml
//! [[skill]]
//! name        = "summarize_url"
//! description = "Fetch a URL, extract text, then summarize"
//!
//! [[skill.step]]
//! tool   = "fetch_url"
//! params = { url = "${url}" }
//!
//! [[skill.step]]
//! tool   = "extract_text"
//! params = { html = "${previous}" }
//!
//! [[skill.step]]
//! tool   = "summarize"
//! params = { text = "${previous}" }
//! ```
//!
//! ## Usage
//!
//! ```rust,no_run
//! use llm_agent_runtime::skills::{SkillLibrary, SkillConfig};
//! use serde_json::json;
//!
//! # tokio_test::block_on(async {
//! let mut library = SkillLibrary::from_toml(r#"
//! [[skill]]
//! name = "greet"
//! description = "Greet a user"
//!
//! [[skill.step]]
//! tool   = "say_hello"
//! params = { name = "${name}" }
//! "#).unwrap();
//!
//! library.register_tool("say_hello", |args| {
//!     Box::pin(async move { Ok(serde_json::json!({"message": format!("Hello, {}!", args["name"].as_str().unwrap_or("world"))})) })
//! });
//!
//! let result = library.run("greet", json!({"name": "Alice"})).await.unwrap();
//! println!("{result}");
//! # });
//! ```

pub mod builtins;

use crate::error::AgentRuntimeError;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

// ── Trait ─────────────────────────────────────────────────────────────────────

/// A pinned, boxed future returning a fallible JSON value.
pub type SkillFuture = Pin<Box<dyn Future<Output = Result<Value, String>> + Send>>;

/// Core abstraction for a named, parameterised agent capability.
pub trait Skill: Send + Sync {
    /// Human-readable name of the skill.
    fn name(&self) -> &str;

    /// Optional description for documentation.
    fn description(&self) -> &str {
        ""
    }

    /// Execute the skill with the supplied parameter map.
    fn execute(&self, params: Value) -> SkillFuture;
}

// ── Config types ──────────────────────────────────────────────────────────────

/// A single step inside a skill definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillStep {
    /// Name of the tool to invoke.
    pub tool: String,
    /// JSON parameter map; `${previous}` is substituted with the previous
    /// step's output, and `${key}` is substituted from the call-site params.
    #[serde(default)]
    pub params: serde_json::Map<String, Value>,
}

/// Full definition of a TOML-declared skill.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillConfig {
    /// Unique skill name.
    pub name: String,
    /// Human-readable description.
    #[serde(default)]
    pub description: String,
    /// Ordered list of steps.
    #[serde(rename = "step", default)]
    pub steps: Vec<SkillStep>,
}

/// Top-level TOML wrapper.
#[derive(Debug, Deserialize)]
struct RawSkillFile {
    #[serde(rename = "skill")]
    skills: Vec<SkillConfig>,
}

// ── Tool handler registry ─────────────────────────────────────────────────────

/// A handler for a tool used inside a skill step.
pub type ToolHandler = Box<dyn Fn(Value) -> SkillFuture + Send + Sync>;

// ── Parameter interpolation ───────────────────────────────────────────────────

/// Substitute `${key}` placeholders in a string with values from `params`.
///
/// `${previous}` is replaced with `previous_output` serialised as a JSON
/// string.
fn interpolate(template: &str, params: &Value, previous_output: &Value) -> String {
    let mut result = template.to_string();

    // Replace ${previous} first.
    let prev_str = match previous_output {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    result = result.replace("${previous}", &prev_str);

    // Replace ${key} for each key in params.
    if let Value::Object(map) = params {
        for (key, val) in map {
            let placeholder = format!("${{{key}}}");
            let replacement = match val {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            result = result.replace(&placeholder, &replacement);
        }
    }

    result
}

/// Apply interpolation to every value inside a step's params map, producing a
/// new JSON object.
fn interpolate_params(
    step_params: &serde_json::Map<String, Value>,
    call_params: &Value,
    previous: &Value,
) -> Value {
    let mut out = serde_json::Map::new();
    for (k, v) in step_params {
        let interpolated = match v {
            Value::String(s) => Value::String(interpolate(s, call_params, previous)),
            other => other.clone(),
        };
        out.insert(k.clone(), interpolated);
    }
    Value::Object(out)
}

// ── Library ───────────────────────────────────────────────────────────────────

/// A registry of [`SkillConfig`]s and their associated tool handlers.
///
/// Load skills from TOML with [`SkillLibrary::from_toml`], register tool
/// implementations with [`SkillLibrary::register_tool`], then execute skills
/// by name with [`SkillLibrary::run`].
#[derive(Default)]
pub struct SkillLibrary {
    skills: HashMap<String, SkillConfig>,
    tools: HashMap<String, ToolHandler>,
    /// User-supplied [`Skill`] trait objects registered programmatically.
    trait_skills: HashMap<String, Box<dyn Skill>>,
}

impl SkillLibrary {
    /// Create an empty library.
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse skills from a TOML string and return a new library.
    ///
    /// # Errors
    /// Returns [`AgentRuntimeError::Internal`] on parse failure.
    pub fn from_toml(toml_str: &str) -> Result<Self, AgentRuntimeError> {
        let raw: RawSkillFile = toml::from_str(toml_str)
            .map_err(|e| AgentRuntimeError::Internal(format!("Skill TOML parse error: {e}")))?;
        let skills = raw
            .skills
            .into_iter()
            .map(|s| (s.name.clone(), s))
            .collect();
        Ok(Self {
            skills,
            tools: HashMap::new(),
            trait_skills: HashMap::new(),
        })
    }

    /// Register a tool handler used by skill steps.
    pub fn register_tool<F, Fut>(&mut self, tool_name: &str, handler: F)
    where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value, String>> + Send + 'static,
    {
        self.tools.insert(
            tool_name.to_string(),
            Box::new(move |args| Box::pin(handler(args))),
        );
    }

    /// Register a programmatic [`Skill`] implementation.
    pub fn register_skill(&mut self, skill: Box<dyn Skill>) {
        self.trait_skills.insert(skill.name().to_string(), skill);
    }

    /// Load additional skill configs from a TOML string, merging into this library.
    pub fn load_toml(&mut self, toml_str: &str) -> Result<(), AgentRuntimeError> {
        let raw: RawSkillFile = toml::from_str(toml_str)
            .map_err(|e| AgentRuntimeError::Internal(format!("Skill TOML parse error: {e}")))?;
        for skill in raw.skills {
            self.skills.insert(skill.name.clone(), skill);
        }
        Ok(())
    }

    /// Execute a skill by name with the supplied call-site parameters.
    ///
    /// Skill steps run sequentially.  The output of each step is passed to the
    /// next via `${previous}` interpolation.
    ///
    /// # Errors
    /// - [`AgentRuntimeError::Internal`] if the skill is not found.
    /// - [`AgentRuntimeError::Internal`] if a referenced tool is not registered.
    pub async fn run(&self, skill_name: &str, params: Value) -> Result<Value, AgentRuntimeError> {
        // Prefer programmatic trait skills.
        if let Some(skill) = self.trait_skills.get(skill_name) {
            return skill
                .execute(params)
                .await
                .map_err(AgentRuntimeError::Internal);
        }

        let config = self.skills.get(skill_name).ok_or_else(|| {
            AgentRuntimeError::Internal(format!("Skill '{skill_name}' not found in library"))
        })?;

        let mut previous: Value = Value::Null;

        for step in &config.steps {
            let tool = self.tools.get(&step.tool).ok_or_else(|| {
                AgentRuntimeError::Internal(format!(
                    "Tool '{}' not registered (required by skill '{skill_name}')",
                    step.tool
                ))
            })?;

            let step_params = interpolate_params(&step.params, &params, &previous);
            tracing::debug!(
                skill = skill_name,
                tool = %step.tool,
                "skills: executing step"
            );
            previous = tool(step_params)
                .await
                .map_err(AgentRuntimeError::Internal)?;
        }

        Ok(previous)
    }

    /// List all skill names (TOML-defined + trait-registered).
    pub fn skill_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.skills.keys().map(|s| s.as_str()).collect();
        names.extend(self.trait_skills.keys().map(|s| s.as_str()));
        names.sort();
        names
    }

    /// Look up a TOML-defined skill config.
    pub fn get_config(&self, name: &str) -> Option<&SkillConfig> {
        self.skills.get(name)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const SAMPLE_TOML: &str = r#"
[[skill]]
name = "summarize_url"
description = "Fetch a URL, extract text, then summarize"

[[skill.step]]
tool   = "fetch_url"
params = { url = "${url}" }

[[skill.step]]
tool   = "extract_text"
params = { html = "${previous}" }

[[skill.step]]
tool   = "summarize"
params = { text = "${previous}" }

[[skill]]
name = "echo"
description = "Echo input"

[[skill.step]]
tool   = "echo_tool"
params = { value = "${value}" }
"#;

    #[test]
    fn parse_skills_from_toml() {
        let lib = SkillLibrary::from_toml(SAMPLE_TOML).unwrap();
        let mut names = lib.skill_names();
        names.sort();
        assert!(names.contains(&"summarize_url"));
        assert!(names.contains(&"echo"));
    }

    #[tokio::test]
    async fn run_echo_skill() {
        let mut lib = SkillLibrary::from_toml(SAMPLE_TOML).unwrap();
        lib.register_tool("echo_tool", |args| {
            Box::pin(async move { Ok(json!({"echoed": args["value"]})) })
        });

        let result = lib.run("echo", json!({"value": "hello"})).await.unwrap();
        assert_eq!(result["echoed"], "hello");
    }

    #[tokio::test]
    async fn run_multi_step_skill_with_previous() {
        let mut lib = SkillLibrary::from_toml(SAMPLE_TOML).unwrap();
        lib.register_tool("fetch_url", |args| {
            Box::pin(async move {
                Ok(Value::String(format!(
                    "<html>content for {}</html>",
                    args["url"].as_str().unwrap_or("")
                )))
            })
        });
        lib.register_tool("extract_text", |args| {
            Box::pin(async move {
                Ok(Value::String(
                    args["html"]
                        .as_str()
                        .unwrap_or("")
                        .replace("<html>", "")
                        .replace("</html>", ""),
                ))
            })
        });
        lib.register_tool("summarize", |args| {
            Box::pin(async move {
                Ok(json!({"summary": format!("Summary: {}", args["text"].as_str().unwrap_or(""))}))
            })
        });

        let result = lib
            .run("summarize_url", json!({"url": "https://example.com"}))
            .await
            .unwrap();
        assert!(result["summary"]
            .as_str()
            .unwrap_or("")
            .contains("example.com"));
    }

    #[tokio::test]
    async fn missing_skill_returns_error() {
        let lib = SkillLibrary::new();
        let err = lib.run("nonexistent", json!({})).await.unwrap_err();
        assert!(matches!(err, AgentRuntimeError::Internal(_)));
    }

    #[tokio::test]
    async fn programmatic_skill_registration() {
        struct GreetSkill;
        impl Skill for GreetSkill {
            fn name(&self) -> &str {
                "greet"
            }
            fn execute(&self, params: Value) -> SkillFuture {
                Box::pin(async move {
                    let name = params["name"].as_str().unwrap_or("world");
                    Ok(json!({"greeting": format!("Hello, {name}!")}))
                })
            }
        }

        let mut lib = SkillLibrary::new();
        lib.register_skill(Box::new(GreetSkill));

        let result = lib.run("greet", json!({"name": "Bob"})).await.unwrap();
        assert_eq!(result["greeting"], "Hello, Bob!");
    }
}

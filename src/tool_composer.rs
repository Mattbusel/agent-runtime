//! # Module: Tool Composer
//!
//! Compose simple tools into higher-order compound tools using configurable
//! composition strategies: Sequential, Parallel, Conditional, and Pipeline.
//!
//! ## Design
//!
//! ```text
//! ToolRegistry ──► ToolComposer::build(specs, strategy) ──► ComposedTool
//!                                                                 │
//!                                                          execute(inputs)
//!                                                                 │
//!                                                         Vec<ToolOutput>
//! ```

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

// ── Primitive types ───────────────────────────────────────────────────────────

/// A single named input to a tool invocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInput {
    /// Name of the input parameter.
    pub name: String,
    /// Value of the input parameter.
    pub value: Value,
}

/// The result of executing one tool step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutput {
    /// Whether the tool step completed without error.
    pub success: bool,
    /// Structured result value.
    pub result: Value,
    /// Error message, if any.
    pub error: Option<String>,
    /// Simulated wall-clock latency in milliseconds.
    pub latency_ms: u64,
}

impl ToolOutput {
    /// Construct a successful output.
    pub fn ok(result: Value, latency_ms: u64) -> Self {
        Self { success: true, result, error: None, latency_ms }
    }

    /// Construct a failed output.
    pub fn err(message: impl Into<String>, latency_ms: u64) -> Self {
        Self {
            success: false,
            result: Value::Null,
            error: Some(message.into()),
            latency_ms,
        }
    }
}

// ── Tool specification ────────────────────────────────────────────────────────

/// Declarative description of a tool that can be composed.
///
/// Note: named `ComposerToolSpec` to avoid clashing with the `ToolSpec` already
/// exported from the `agent` module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComposerToolSpec {
    /// Unique tool name used as the registry key.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Required input parameter names.
    pub input_schema: Vec<String>,
    /// Declared output field names.
    pub output_schema: Vec<String>,
}

// ── Composition strategies ────────────────────────────────────────────────────

/// How a [`ComposedTool`] chains its constituent steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompositionStrategy {
    /// Run all steps one after another; pass the previous output forward.
    Sequential,
    /// Run all steps on the same input set; collect independent outputs.
    Parallel,
    /// Run only the step whose name matches `condition_key` in the inputs.
    Conditional {
        /// The input key whose *value* (as a string) selects the step to run.
        condition_key: String,
    },
    /// Each step receives its predecessor's `result` field as an extra input
    /// named `"pipeline_input"`.
    Pipeline,
}

// ── ComposedTool ─────────────────────────────────────────────────────────────

/// A higher-order tool built by composing multiple [`ComposerToolSpec`]s.
#[derive(Debug, Clone)]
pub struct ComposedTool {
    /// Ordered list of steps in this composition.
    pub steps: Vec<ComposerToolSpec>,
    /// How the steps are wired together.
    pub strategy: CompositionStrategy,
}

impl ComposedTool {
    /// Return the list of required input names that are absent from `inputs`.
    pub fn validate_inputs(
        &self,
        inputs: &HashMap<String, Value>,
    ) -> Vec<String> {
        let mut missing = Vec::new();
        for step in &self.steps {
            for key in &step.input_schema {
                if !inputs.contains_key(key.as_str()) && !missing.contains(key) {
                    missing.push(key.clone());
                }
            }
        }
        missing
    }

    /// Simulate execution of all steps according to the composition strategy.
    ///
    /// Execution is entirely in-memory with mock latencies derived from the
    /// step name length — no real I/O is performed.
    pub fn execute(&self, inputs: HashMap<String, Value>) -> Vec<ToolOutput> {
        match &self.strategy {
            CompositionStrategy::Sequential => self.execute_sequential(&inputs),
            CompositionStrategy::Parallel => self.execute_parallel(&inputs),
            CompositionStrategy::Conditional { condition_key } => {
                self.execute_conditional(&inputs, condition_key)
            }
            CompositionStrategy::Pipeline => self.execute_pipeline(&inputs),
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn mock_latency(step: &ComposerToolSpec) -> u64 {
        // Deterministic fake latency: 10 ms base + 2 ms per character in name.
        10 + (step.name.len() as u64) * 2
    }

    fn mock_output(step: &ComposerToolSpec, extra: Option<Value>) -> ToolOutput {
        let result = serde_json::json!({
            "tool": step.name,
            "output_fields": step.output_schema,
            "echo": extra,
        });
        ToolOutput::ok(result, Self::mock_latency(step))
    }

    fn execute_sequential(&self, inputs: &HashMap<String, Value>) -> Vec<ToolOutput> {
        let mut outputs = Vec::with_capacity(self.steps.len());
        let mut carry: Option<Value> = None;
        for step in &self.steps {
            let extra = carry.clone().or_else(|| {
                inputs.get(&step.name).cloned()
            });
            let out = Self::mock_output(step, extra);
            carry = Some(out.result.clone());
            outputs.push(out);
        }
        outputs
    }

    fn execute_parallel(&self, inputs: &HashMap<String, Value>) -> Vec<ToolOutput> {
        self.steps
            .iter()
            .map(|step| {
                let extra = inputs.get(&step.name).cloned();
                Self::mock_output(step, extra)
            })
            .collect()
    }

    fn execute_conditional(
        &self,
        inputs: &HashMap<String, Value>,
        condition_key: &str,
    ) -> Vec<ToolOutput> {
        let selected = inputs
            .get(condition_key)
            .and_then(|v| v.as_str())
            .unwrap_or("");

        match self.steps.iter().find(|s| s.name == selected) {
            Some(step) => vec![Self::mock_output(step, inputs.get(&step.name).cloned())],
            None => vec![ToolOutput::err(
                format!("no step matching condition key '{selected}'"),
                0,
            )],
        }
    }

    fn execute_pipeline(&self, inputs: &HashMap<String, Value>) -> Vec<ToolOutput> {
        let mut outputs = Vec::with_capacity(self.steps.len());
        let mut pipeline_input: Option<Value> = None;
        for step in &self.steps {
            let mut merged = inputs.clone();
            if let Some(prev) = pipeline_input.take() {
                merged.insert("pipeline_input".to_string(), prev);
            }
            let extra = merged.get(&step.name).cloned();
            let out = Self::mock_output(step, extra);
            pipeline_input = Some(out.result.clone());
            outputs.push(out);
        }
        outputs
    }
}

// ── ToolRegistry ──────────────────────────────────────────────────────────────

/// Registry that maps tool names to their specifications.
#[derive(Debug, Default)]
pub struct ToolRegistry {
    tools: HashMap<String, ComposerToolSpec>,
}

impl ToolRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tool specification.  Overwrites any previous entry with the
    /// same name.
    pub fn register(&mut self, spec: ComposerToolSpec) {
        self.tools.insert(spec.name.clone(), spec);
    }

    /// Look up a tool by name.
    pub fn get(&self, name: &str) -> Option<&ComposerToolSpec> {
        self.tools.get(name)
    }

    /// List all registered tool names in alphabetical order.
    pub fn list(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.tools.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Returns `true` if no tools have been registered.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

// ── ToolComposer ──────────────────────────────────────────────────────────────

/// Factory for constructing [`ComposedTool`]s from a set of specs and a strategy.
pub struct ToolComposer;

impl ToolComposer {
    /// Build a [`ComposedTool`] from the given specifications and strategy.
    pub fn build(specs: Vec<ComposerToolSpec>, strategy: CompositionStrategy) -> ComposedTool {
        ComposedTool { steps: specs, strategy }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #[allow(clippy::wildcard_imports)]
    use super::*;

    fn make_spec(name: &str, inputs: &[&str], outputs: &[&str]) -> ComposerToolSpec {
        ComposerToolSpec {
            name: name.to_string(),
            description: format!("Mock tool: {name}"),
            input_schema: inputs.iter().map(|s| s.to_string()).collect(),
            output_schema: outputs.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn sample_specs() -> Vec<ComposerToolSpec> {
        vec![
            make_spec("search", &["query"], &["results"]),
            make_spec("summarize", &["query", "results"], &["summary"]),
        ]
    }

    // ── ToolOutput helpers ────────────────────────────────────────────────────

    #[test]
    fn tool_output_ok() {
        let out = ToolOutput::ok(serde_json::json!({"x": 1}), 42);
        assert!(out.success);
        assert!(out.error.is_none());
        assert_eq!(out.latency_ms, 42);
    }

    #[test]
    fn tool_output_err() {
        let out = ToolOutput::err("boom", 5);
        assert!(!out.success);
        assert_eq!(out.error.as_deref(), Some("boom"));
    }

    // ── validate_inputs ───────────────────────────────────────────────────────

    #[test]
    fn validate_missing_inputs() {
        let tool = ToolComposer::build(sample_specs(), CompositionStrategy::Sequential);
        let inputs = HashMap::from([("query".to_string(), serde_json::json!("hello"))]);
        let missing = tool.validate_inputs(&inputs);
        assert!(missing.contains(&"results".to_string()));
        assert!(!missing.contains(&"query".to_string()));
    }

    #[test]
    fn validate_all_present() {
        let tool = ToolComposer::build(sample_specs(), CompositionStrategy::Sequential);
        let inputs = HashMap::from([
            ("query".to_string(), serde_json::json!("hello")),
            ("results".to_string(), serde_json::json!(["r1"])),
        ]);
        let missing = tool.validate_inputs(&inputs);
        assert!(missing.is_empty());
    }

    // ── Sequential ────────────────────────────────────────────────────────────

    #[test]
    fn sequential_produces_output_per_step() {
        let tool = ToolComposer::build(sample_specs(), CompositionStrategy::Sequential);
        let inputs = HashMap::from([("query".to_string(), serde_json::json!("test"))]);
        let outputs = tool.execute(inputs);
        assert_eq!(outputs.len(), 2);
        assert!(outputs.iter().all(|o| o.success));
    }

    #[test]
    fn sequential_carries_result_forward() {
        let specs = vec![
            make_spec("step_a", &[], &["a_out"]),
            make_spec("step_b", &[], &["b_out"]),
        ];
        let tool = ToolComposer::build(specs, CompositionStrategy::Sequential);
        let outputs = tool.execute(HashMap::new());
        // step_b's echo should contain step_a's result
        let b_echo = outputs[1].result["echo"].clone();
        assert!(!b_echo.is_null());
    }

    // ── Parallel ──────────────────────────────────────────────────────────────

    #[test]
    fn parallel_runs_all_steps() {
        let tool = ToolComposer::build(sample_specs(), CompositionStrategy::Parallel);
        let outputs = tool.execute(HashMap::new());
        assert_eq!(outputs.len(), 2);
        assert!(outputs.iter().all(|o| o.success));
    }

    // ── Conditional ───────────────────────────────────────────────────────────

    #[test]
    fn conditional_selects_matching_step() {
        let tool = ToolComposer::build(
            sample_specs(),
            CompositionStrategy::Conditional { condition_key: "route".to_string() },
        );
        let inputs =
            HashMap::from([("route".to_string(), serde_json::json!("search"))]);
        let outputs = tool.execute(inputs);
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].success);
        assert_eq!(outputs[0].result["tool"].as_str(), Some("search"));
    }

    #[test]
    fn conditional_returns_error_when_no_match() {
        let tool = ToolComposer::build(
            sample_specs(),
            CompositionStrategy::Conditional { condition_key: "route".to_string() },
        );
        let inputs =
            HashMap::from([("route".to_string(), serde_json::json!("nonexistent"))]);
        let outputs = tool.execute(inputs);
        assert_eq!(outputs.len(), 1);
        assert!(!outputs[0].success);
    }

    // ── Pipeline ──────────────────────────────────────────────────────────────

    #[test]
    fn pipeline_threads_results() {
        let tool = ToolComposer::build(sample_specs(), CompositionStrategy::Pipeline);
        let outputs = tool.execute(HashMap::new());
        assert_eq!(outputs.len(), 2);
        assert!(outputs.iter().all(|o| o.success));
    }

    // ── ToolRegistry ──────────────────────────────────────────────────────────

    #[test]
    fn registry_register_and_lookup() {
        let mut reg = ToolRegistry::new();
        assert!(reg.is_empty());
        reg.register(make_spec("alpha", &["x"], &["y"]));
        assert_eq!(reg.len(), 1);
        assert!(reg.get("alpha").is_some());
        assert!(reg.get("beta").is_none());
    }

    #[test]
    fn registry_list_sorted() {
        let mut reg = ToolRegistry::new();
        reg.register(make_spec("zebra", &[], &[]));
        reg.register(make_spec("apple", &[], &[]));
        let names = reg.list();
        assert_eq!(names, vec!["apple", "zebra"]);
    }

    #[test]
    fn registry_overwrite() {
        let mut reg = ToolRegistry::new();
        reg.register(make_spec("tool", &["a"], &["b"]));
        reg.register(make_spec("tool", &["x"], &["y"]));
        assert_eq!(reg.len(), 1);
        let spec = reg.get("tool").expect("should exist");
        assert_eq!(spec.input_schema, vec!["x"]);
    }

    // ── ToolComposer::build ───────────────────────────────────────────────────

    #[test]
    fn composer_build_returns_composed_tool() {
        let specs = sample_specs();
        let tool = ToolComposer::build(specs, CompositionStrategy::Parallel);
        assert_eq!(tool.steps.len(), 2);
    }

    #[test]
    fn mock_latency_is_deterministic() {
        let spec = make_spec("abc", &[], &[]);
        let latency = ComposedTool::mock_latency(&spec);
        // 10 + 3*2 = 16
        assert_eq!(latency, 16);
    }
}

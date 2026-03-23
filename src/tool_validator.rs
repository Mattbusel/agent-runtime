//! # Module: Tool Validator
//!
//! JSON-schema-like validation of tool call arguments before execution.
//!
//! ## Overview
//!
//! [`ToolSchema`] describes the expected parameters of a single tool.
//! [`SchemaValidator`] holds a registry of schemas and validates a
//! [`ToolCall`] against the matching schema before the tool is dispatched.
//!
//! ## Usage
//!
//! ```rust
//! use llm_agent_runtime::tool_validator::{
//!     ParameterSpec, ParameterTypeHint, SchemaValidator, ToolCall, ToolSchema,
//! };
//! use serde_json::json;
//!
//! let schema = ToolSchema::new("search", "Web search tool")
//!     .with_param(ParameterSpec::required("query", ParameterTypeHint::String, "Search query"))
//!     .with_param(ParameterSpec::optional("limit", ParameterTypeHint::Number, "Max results"));
//!
//! let mut validator = SchemaValidator::new();
//! validator.register(schema);
//!
//! let call = ToolCall {
//!     tool_name: "search".to_string(),
//!     arguments: json!({ "query": "rust async" }),
//! };
//!
//! assert!(validator.validate(&call).is_ok());
//! ```

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

// ── ParameterTypeHint ─────────────────────────────────────────────────────────

/// The expected JSON type of a tool parameter.
///
/// Used for lightweight type-hint validation: the validator checks that the
/// actual JSON value's type matches the declared hint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParameterTypeHint {
    /// A JSON string value.
    String,
    /// A JSON number value (integer or float).
    Number,
    /// A JSON boolean value.
    Boolean,
    /// A JSON array value.
    Array,
    /// A JSON object value.
    Object,
    /// Any JSON value is accepted; no type check is performed.
    Any,
}

impl ParameterTypeHint {
    /// Return a human-readable name for use in error messages.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::String => "string",
            Self::Number => "number",
            Self::Boolean => "boolean",
            Self::Array => "array",
            Self::Object => "object",
            Self::Any => "any",
        }
    }

    /// Return `true` if the given JSON `value` satisfies this type hint.
    pub fn matches(&self, value: &Value) -> bool {
        match self {
            Self::String => value.is_string(),
            Self::Number => value.is_number(),
            Self::Boolean => value.is_boolean(),
            Self::Array => value.is_array(),
            Self::Object => value.is_object(),
            Self::Any => true,
        }
    }
}

impl std::fmt::Display for ParameterTypeHint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── ParameterSpec ─────────────────────────────────────────────────────────────

/// Specification for a single tool parameter.
///
/// Describes the parameter's name, expected type, whether it is required, and
/// an optional human-readable description.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSpec {
    /// The parameter name as it appears in the JSON arguments object.
    pub name: String,
    /// The expected JSON type of this parameter.
    pub type_hint: ParameterTypeHint,
    /// Whether the parameter must be present in every tool call.
    pub required: bool,
    /// Human-readable description of the parameter's purpose.
    pub description: String,
}

impl ParameterSpec {
    /// Create a new required parameter spec.
    pub fn required(
        name: impl Into<String>,
        type_hint: ParameterTypeHint,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            type_hint,
            required: true,
            description: description.into(),
        }
    }

    /// Create a new optional parameter spec.
    pub fn optional(
        name: impl Into<String>,
        type_hint: ParameterTypeHint,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            type_hint,
            required: false,
            description: description.into(),
        }
    }
}

// ── ToolSchema ────────────────────────────────────────────────────────────────

/// Schema describing the expected arguments of a single tool.
///
/// Register schemas with a [`SchemaValidator`] to enable pre-dispatch
/// validation of [`ToolCall`]s.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSchema {
    /// The tool name that this schema describes.
    pub name: String,
    /// Human-readable description of what the tool does.
    pub description: String,
    /// Ordered list of parameter specifications.
    pub parameters: Vec<ParameterSpec>,
}

impl ToolSchema {
    /// Create a new `ToolSchema` with no parameters.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: Vec::new(),
        }
    }

    /// Append a parameter spec and return `self` for builder-style chaining.
    pub fn with_param(mut self, param: ParameterSpec) -> Self {
        self.parameters.push(param);
        self
    }

    /// Return an iterator over required parameter specs.
    pub fn required_params(&self) -> impl Iterator<Item = &ParameterSpec> {
        self.parameters.iter().filter(|p| p.required)
    }

    /// Return an iterator over optional parameter specs.
    pub fn optional_params(&self) -> impl Iterator<Item = &ParameterSpec> {
        self.parameters.iter().filter(|p| !p.required)
    }

    /// Return the number of declared parameters (required + optional).
    pub fn param_count(&self) -> usize {
        self.parameters.len()
    }
}

// ── ToolCall ──────────────────────────────────────────────────────────────────

/// A concrete tool invocation: name plus JSON arguments.
///
/// Pass this to [`SchemaValidator::validate`] before dispatching to the handler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// The name of the tool to invoke.
    pub tool_name: String,
    /// JSON arguments to pass to the tool handler.
    pub arguments: Value,
}

impl ToolCall {
    /// Construct a new `ToolCall`.
    pub fn new(tool_name: impl Into<String>, arguments: Value) -> Self {
        Self {
            tool_name: tool_name.into(),
            arguments,
        }
    }
}

// ── ValidationError ───────────────────────────────────────────────────────────

/// Errors that may arise during tool-call argument validation.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ValidationError {
    /// A required parameter was absent from the arguments object.
    #[error("Missing required parameter '{0}'")]
    MissingRequired(String),

    /// A parameter value's JSON type did not match the declared type hint.
    #[error("Type mismatch for parameter '{param}': expected {expected}, got {got}")]
    TypeMismatch {
        /// The name of the parameter whose type was wrong.
        param: String,
        /// The declared expected type.
        expected: String,
        /// The actual JSON type found.
        got: String,
    },

    /// No schema was registered for the named tool.
    #[error("Unknown tool '{0}': no schema registered")]
    UnknownTool(String),
}

impl ValidationError {
    /// Return `true` if this is a [`ValidationError::MissingRequired`] error.
    pub fn is_missing_required(&self) -> bool {
        matches!(self, Self::MissingRequired(_))
    }

    /// Return `true` if this is a [`ValidationError::TypeMismatch`] error.
    pub fn is_type_mismatch(&self) -> bool {
        matches!(self, Self::TypeMismatch { .. })
    }

    /// Return `true` if this is a [`ValidationError::UnknownTool`] error.
    pub fn is_unknown_tool(&self) -> bool {
        matches!(self, Self::UnknownTool(_))
    }
}

// ── json_type_name ─────────────────────────────────────────────────────────────

/// Return the JSON type name of a `Value` for use in error messages.
fn json_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

// ── SchemaValidator ────────────────────────────────────────────────────────────

/// Validates [`ToolCall`]s against registered [`ToolSchema`]s.
///
/// Register one schema per tool with [`SchemaValidator::register`], then call
/// [`SchemaValidator::validate`] before dispatching a tool call.
///
/// # Validation Rules
///
/// 1. The `tool_name` must match a registered schema; otherwise
///    [`ValidationError::UnknownTool`] is returned.
/// 2. Every required parameter must be present in `arguments`; missing ones
///    produce [`ValidationError::MissingRequired`].
/// 3. Every parameter present in `arguments` that has a declared spec (required
///    or optional) is checked against its `type_hint`; mismatches produce
///    [`ValidationError::TypeMismatch`].
/// 4. Extra keys not declared in the schema are silently ignored.
#[derive(Debug, Default)]
pub struct SchemaValidator {
    schemas: HashMap<String, ToolSchema>,
}

impl SchemaValidator {
    /// Create an empty `SchemaValidator` with no registered schemas.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a [`ToolSchema`].  If a schema with the same name already
    /// exists it is replaced.
    pub fn register(&mut self, schema: ToolSchema) {
        self.schemas.insert(schema.name.clone(), schema);
    }

    /// Register multiple schemas at once.
    pub fn register_all(&mut self, schemas: impl IntoIterator<Item = ToolSchema>) {
        for s in schemas {
            self.register(s);
        }
    }

    /// Remove the schema for `tool_name` and return it, if present.
    pub fn deregister(&mut self, tool_name: &str) -> Option<ToolSchema> {
        self.schemas.remove(tool_name)
    }

    /// Return the number of registered schemas.
    pub fn len(&self) -> usize {
        self.schemas.len()
    }

    /// Return `true` if no schemas are registered.
    pub fn is_empty(&self) -> bool {
        self.schemas.is_empty()
    }

    /// Return a reference to the schema for `tool_name`, if registered.
    pub fn get(&self, tool_name: &str) -> Option<&ToolSchema> {
        self.schemas.get(tool_name)
    }

    /// Validate a [`ToolCall`] against its registered schema.
    ///
    /// Returns `Ok(())` when the call is valid.  The first validation failure
    /// encountered is returned as `Err(ValidationError)`.
    ///
    /// # Errors
    ///
    /// - [`ValidationError::UnknownTool`] — no schema registered for the tool.
    /// - [`ValidationError::MissingRequired`] — a required parameter is absent.
    /// - [`ValidationError::TypeMismatch`] — a parameter's value has the wrong type.
    pub fn validate(&self, call: &ToolCall) -> Result<(), ValidationError> {
        let schema = self
            .schemas
            .get(&call.tool_name)
            .ok_or_else(|| ValidationError::UnknownTool(call.tool_name.clone()))?;

        // arguments must be an object; treat non-objects as an empty object
        // (required-field checks will then fire for every required param).
        let args_obj = call.arguments.as_object();

        // 1. Check required parameters are present.
        for param in schema.required_params() {
            let present = args_obj.map_or(false, |m| m.contains_key(&param.name));
            if !present {
                return Err(ValidationError::MissingRequired(param.name.clone()));
            }
        }

        // 2. Type-check each declared parameter that is actually present.
        for param in &schema.parameters {
            if let Some(value) = args_obj.and_then(|m| m.get(&param.name)) {
                if !param.type_hint.matches(value) {
                    return Err(ValidationError::TypeMismatch {
                        param: param.name.clone(),
                        expected: param.type_hint.to_string(),
                        got: json_type_name(value).to_string(),
                    });
                }
            }
        }

        Ok(())
    }

    /// Validate a [`ToolCall`] and collect *all* validation errors rather than
    /// stopping at the first failure.
    ///
    /// Returns an empty `Vec` when the call is valid.
    pub fn validate_all(&self, call: &ToolCall) -> Vec<ValidationError> {
        let schema = match self.schemas.get(&call.tool_name) {
            Some(s) => s,
            None => return vec![ValidationError::UnknownTool(call.tool_name.clone())],
        };

        let mut errors = Vec::new();
        let args_obj = call.arguments.as_object();

        for param in schema.required_params() {
            let present = args_obj.map_or(false, |m| m.contains_key(&param.name));
            if !present {
                errors.push(ValidationError::MissingRequired(param.name.clone()));
            }
        }

        for param in &schema.parameters {
            if let Some(value) = args_obj.and_then(|m| m.get(&param.name)) {
                if !param.type_hint.matches(value) {
                    errors.push(ValidationError::TypeMismatch {
                        param: param.name.clone(),
                        expected: param.type_hint.to_string(),
                        got: json_type_name(value).to_string(),
                    });
                }
            }
        }

        errors
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #[allow(clippy::wildcard_imports)]
    use super::*;
    use serde_json::json;

    fn search_schema() -> ToolSchema {
        ToolSchema::new("search", "Web search")
            .with_param(ParameterSpec::required(
                "query",
                ParameterTypeHint::String,
                "Search query",
            ))
            .with_param(ParameterSpec::optional(
                "limit",
                ParameterTypeHint::Number,
                "Max result count",
            ))
    }

    fn calculator_schema() -> ToolSchema {
        ToolSchema::new("calculator", "Arithmetic calculator")
            .with_param(ParameterSpec::required(
                "expression",
                ParameterTypeHint::String,
                "Math expression",
            ))
            .with_param(ParameterSpec::required(
                "precision",
                ParameterTypeHint::Number,
                "Decimal places",
            ))
    }

    // ── basic happy-path ──────────────────────────────────────────────────────

    #[test]
    fn valid_call_with_required_only() {
        let mut v = SchemaValidator::new();
        v.register(search_schema());
        let call = ToolCall::new("search", json!({ "query": "rust" }));
        assert!(v.validate(&call).is_ok());
    }

    #[test]
    fn valid_call_with_required_and_optional() {
        let mut v = SchemaValidator::new();
        v.register(search_schema());
        let call = ToolCall::new("search", json!({ "query": "rust", "limit": 10 }));
        assert!(v.validate(&call).is_ok());
    }

    #[test]
    fn extra_keys_are_silently_ignored() {
        let mut v = SchemaValidator::new();
        v.register(search_schema());
        let call = ToolCall::new(
            "search",
            json!({ "query": "hello", "unknown_field": true }),
        );
        assert!(v.validate(&call).is_ok());
    }

    // ── UnknownTool ───────────────────────────────────────────────────────────

    #[test]
    fn unknown_tool_returns_error() {
        let v = SchemaValidator::new();
        let call = ToolCall::new("nonexistent", json!({}));
        let err = v.validate(&call).unwrap_err();
        assert!(err.is_unknown_tool());
        assert!(err.to_string().contains("nonexistent"));
    }

    // ── MissingRequired ───────────────────────────────────────────────────────

    #[test]
    fn missing_required_param_returns_error() {
        let mut v = SchemaValidator::new();
        v.register(search_schema());
        let call = ToolCall::new("search", json!({ "limit": 5 }));
        let err = v.validate(&call).unwrap_err();
        assert!(err.is_missing_required());
        assert!(err.to_string().contains("query"));
    }

    #[test]
    fn missing_one_of_two_required_params() {
        let mut v = SchemaValidator::new();
        v.register(calculator_schema());
        // "expression" present but "precision" absent
        let call = ToolCall::new("calculator", json!({ "expression": "1+1" }));
        let err = v.validate(&call).unwrap_err();
        assert!(err.is_missing_required());
    }

    #[test]
    fn non_object_args_treated_as_empty() {
        let mut v = SchemaValidator::new();
        v.register(search_schema());
        let call = ToolCall::new("search", json!(null));
        let err = v.validate(&call).unwrap_err();
        assert!(err.is_missing_required());
    }

    // ── TypeMismatch ──────────────────────────────────────────────────────────

    #[test]
    fn type_mismatch_string_vs_number() {
        let mut v = SchemaValidator::new();
        v.register(search_schema());
        // "query" should be String but we pass a number
        let call = ToolCall::new("search", json!({ "query": 42 }));
        let err = v.validate(&call).unwrap_err();
        assert!(err.is_type_mismatch());
        assert!(err.to_string().contains("query"));
        assert!(err.to_string().contains("string"));
        assert!(err.to_string().contains("number"));
    }

    #[test]
    fn type_mismatch_number_vs_boolean() {
        let mut v = SchemaValidator::new();
        v.register(calculator_schema());
        let call = ToolCall::new(
            "calculator",
            json!({ "expression": "2*3", "precision": true }),
        );
        let err = v.validate(&call).unwrap_err();
        assert!(err.is_type_mismatch());
        assert!(err.to_string().contains("precision"));
    }

    #[test]
    fn optional_param_type_checked_when_present() {
        let mut v = SchemaValidator::new();
        v.register(search_schema());
        // "limit" is optional but declared as Number; pass a string
        let call = ToolCall::new("search", json!({ "query": "x", "limit": "ten" }));
        let err = v.validate(&call).unwrap_err();
        assert!(err.is_type_mismatch());
        assert!(err.to_string().contains("limit"));
    }

    // ── ParameterTypeHint::Any ────────────────────────────────────────────────

    #[test]
    fn any_type_hint_accepts_all_json_types() {
        let schema = ToolSchema::new("flexible", "Accepts anything")
            .with_param(ParameterSpec::required(
                "data",
                ParameterTypeHint::Any,
                "Any value",
            ));
        let mut v = SchemaValidator::new();
        v.register(schema);

        for value in [
            json!("string"),
            json!(42),
            json!(true),
            json!([1, 2]),
            json!({"k": "v"}),
            json!(null),
        ] {
            let call = ToolCall::new("flexible", json!({ "data": value }));
            assert!(v.validate(&call).is_ok(), "should accept {value}");
        }
    }

    // ── Array / Object type hints ─────────────────────────────────────────────

    #[test]
    fn array_type_hint_matches_json_array() {
        let schema = ToolSchema::new("batch", "Batch processor").with_param(
            ParameterSpec::required("items", ParameterTypeHint::Array, "List of items"),
        );
        let mut v = SchemaValidator::new();
        v.register(schema);
        let call = ToolCall::new("batch", json!({ "items": [1, 2, 3] }));
        assert!(v.validate(&call).is_ok());
    }

    #[test]
    fn object_type_hint_matches_json_object() {
        let schema = ToolSchema::new("config", "Apply config").with_param(
            ParameterSpec::required("settings", ParameterTypeHint::Object, "Config map"),
        );
        let mut v = SchemaValidator::new();
        v.register(schema);
        let call = ToolCall::new("config", json!({ "settings": { "key": "val" } }));
        assert!(v.validate(&call).is_ok());
    }

    // ── validate_all ──────────────────────────────────────────────────────────

    #[test]
    fn validate_all_returns_empty_on_valid_call() {
        let mut v = SchemaValidator::new();
        v.register(search_schema());
        let call = ToolCall::new("search", json!({ "query": "hello" }));
        assert!(v.validate_all(&call).is_empty());
    }

    #[test]
    fn validate_all_collects_multiple_errors() {
        let mut v = SchemaValidator::new();
        v.register(calculator_schema());
        // both required params missing
        let call = ToolCall::new("calculator", json!({}));
        let errors = v.validate_all(&call);
        assert_eq!(errors.len(), 2, "expected two MissingRequired errors");
        assert!(errors.iter().all(|e| e.is_missing_required()));
    }

    #[test]
    fn validate_all_unknown_tool_returns_single_error() {
        let v = SchemaValidator::new();
        let call = ToolCall::new("ghost", json!({}));
        let errors = v.validate_all(&call);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].is_unknown_tool());
    }

    // ── registry management ───────────────────────────────────────────────────

    #[test]
    fn register_and_deregister() {
        let mut v = SchemaValidator::new();
        v.register(search_schema());
        assert_eq!(v.len(), 1);
        let removed = v.deregister("search");
        assert!(removed.is_some());
        assert!(v.is_empty());
    }

    #[test]
    fn register_all_adds_multiple_schemas() {
        let mut v = SchemaValidator::new();
        v.register_all([search_schema(), calculator_schema()]);
        assert_eq!(v.len(), 2);
        assert!(v.get("search").is_some());
        assert!(v.get("calculator").is_some());
    }

    #[test]
    fn overwrite_schema_on_re_register() {
        let mut v = SchemaValidator::new();
        v.register(search_schema());
        // Replace with a schema that has no parameters
        v.register(ToolSchema::new("search", "Updated, no params"));
        // Now no required params — call with empty args should succeed
        let call = ToolCall::new("search", json!({}));
        assert!(v.validate(&call).is_ok());
    }

    // ── ToolSchema helpers ────────────────────────────────────────────────────

    #[test]
    fn tool_schema_param_count() {
        let s = search_schema();
        assert_eq!(s.param_count(), 2);
        assert_eq!(s.required_params().count(), 1);
        assert_eq!(s.optional_params().count(), 1);
    }

    // ── ParameterTypeHint display / matches ───────────────────────────────────

    #[test]
    fn boolean_type_hint_matches_bool_value() {
        assert!(ParameterTypeHint::Boolean.matches(&json!(true)));
        assert!(!ParameterTypeHint::Boolean.matches(&json!(1)));
    }

    #[test]
    fn type_hint_display_matches_as_str() {
        for hint in [
            ParameterTypeHint::String,
            ParameterTypeHint::Number,
            ParameterTypeHint::Boolean,
            ParameterTypeHint::Array,
            ParameterTypeHint::Object,
            ParameterTypeHint::Any,
        ] {
            assert_eq!(hint.to_string(), hint.as_str());
        }
    }
}

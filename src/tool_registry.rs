//! # Tool Registry
//!
//! A tool registration and execution tracking system for agent tools.
//! Provides schema validation, invocation recording, statistics, and
//! keyword-based tool discovery.

use std::{
    collections::HashMap,
    sync::{
        Arc, RwLock,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

// ── ToolParamType ─────────────────────────────────────────────────────────────

/// The type of a tool parameter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolParamType {
    /// UTF-8 string value.
    String,
    /// 64-bit signed integer.
    Integer,
    /// 64-bit floating-point number.
    Float,
    /// Boolean true/false.
    Boolean,
    /// JSON-style array.
    Array,
    /// JSON-style object / map.
    Object,
    /// Null / absent value.
    Null,
}

impl std::fmt::Display for ToolParamType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            ToolParamType::String => "string",
            ToolParamType::Integer => "integer",
            ToolParamType::Float => "float",
            ToolParamType::Boolean => "boolean",
            ToolParamType::Array => "array",
            ToolParamType::Object => "object",
            ToolParamType::Null => "null",
        };
        f.write_str(s)
    }
}

// ── ToolParam ─────────────────────────────────────────────────────────────────

/// Describes a single parameter of a tool.
#[derive(Debug, Clone)]
pub struct ToolParam {
    /// Parameter name (used as map key in invocations).
    pub name: String,
    /// Expected type of the parameter value.
    pub param_type: ToolParamType,
    /// Human-readable description of the parameter's purpose.
    pub description: String,
    /// Whether this parameter must be present in every invocation.
    pub required: bool,
    /// Optional default value (as a string representation).
    pub default_value: Option<String>,
}

// ── ToolSchema ────────────────────────────────────────────────────────────────

/// Full schema definition for a registered tool.
#[derive(Debug, Clone)]
pub struct ToolSchema {
    /// Unique tool name used as the registry key.
    pub name: String,
    /// Human-readable description of what the tool does.
    pub description: String,
    /// List of accepted parameters.
    pub parameters: Vec<ToolParam>,
    /// The return type produced by this tool.
    pub returns: ToolParamType,
    /// Example invocation strings for documentation/discovery.
    pub examples: Vec<String>,
    /// Semantic version string (e.g. `"1.0.0"`).
    pub version: String,
    /// Category label for grouping (e.g. `"io"`, `"compute"`, `"network"`).
    pub category: String,
}

// ── ToolInvocation ────────────────────────────────────────────────────────────

/// Global counter used to assign unique invocation IDs.
static INVOCATION_COUNTER: AtomicU64 = AtomicU64::new(1);

/// A record of a single tool call.
#[derive(Debug)]
pub struct ToolInvocation {
    /// Name of the tool that was invoked.
    pub tool_name: String,
    /// Globally unique invocation identifier.
    pub invocation_id: u64,
    /// Parameters passed to the tool.
    pub parameters: HashMap<String, String>,
    /// Wall-clock time when the invocation started.
    pub started_at: Instant,
    /// Wall-clock time when the invocation finished (if it has).
    pub completed_at: Option<Instant>,
    /// Whether the invocation completed successfully.
    pub success: bool,
    /// Optional error message when `success` is `false`.
    pub error: Option<String>,
    /// Size of the result payload in bytes.
    pub result_size_bytes: usize,
}

impl ToolInvocation {
    /// Create a new `ToolInvocation` with an auto-assigned ID, recording the
    /// start time as `Instant::now()`.
    pub fn new(tool_name: impl Into<String>, parameters: HashMap<String, String>) -> Self {
        Self {
            tool_name: tool_name.into(),
            invocation_id: INVOCATION_COUNTER.fetch_add(1, Ordering::Relaxed),
            parameters,
            started_at: Instant::now(),
            completed_at: None,
            success: false,
            error: None,
            result_size_bytes: 0,
        }
    }

    /// Returns the elapsed duration in milliseconds, or `None` if the
    /// invocation has not yet completed.
    pub fn duration_ms(&self) -> Option<u64> {
        self.completed_at
            .map(|end| end.duration_since(self.started_at).as_millis() as u64)
    }
}

// ── ToolStats ─────────────────────────────────────────────────────────────────

/// Aggregate statistics for a single registered tool.
#[derive(Debug, Clone)]
pub struct ToolStats {
    /// Total number of invocations recorded.
    pub invocation_count: u64,
    /// Number of invocations that completed successfully.
    pub success_count: u64,
    /// Number of invocations that failed.
    pub failure_count: u64,
    /// Mean duration in milliseconds across completed invocations.
    pub avg_duration_ms: f64,
    /// 95th-percentile duration in milliseconds across completed invocations.
    pub p95_duration_ms: f64,
    /// Total bytes returned by the tool across all invocations.
    pub total_result_bytes: u64,
}

// ── ToolRegistry ──────────────────────────────────────────────────────────────

/// Registry of tool schemas with invocation history and statistics.
///
/// All operations are thread-safe via internal `RwLock`s.
pub struct ToolRegistry {
    schemas: Arc<RwLock<HashMap<String, ToolSchema>>>,
    invocations: Arc<RwLock<HashMap<String, Vec<ToolInvocation>>>>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    /// Create an empty `ToolRegistry`.
    pub fn new() -> Self {
        Self {
            schemas: Arc::new(RwLock::new(HashMap::new())),
            invocations: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a new tool schema.
    ///
    /// Returns `Err` if a tool with the same name is already registered.
    pub fn register(&self, schema: ToolSchema) -> Result<(), String> {
        let mut schemas = self
            .schemas
            .write()
            .map_err(|e| format!("lock poisoned: {e}"))?;
        if schemas.contains_key(&schema.name) {
            return Err(format!("tool '{}' is already registered", schema.name));
        }
        let name = schema.name.clone();
        schemas.insert(name.clone(), schema);
        drop(schemas);

        let mut invocations = self
            .invocations
            .write()
            .map_err(|e| format!("lock poisoned: {e}"))?;
        invocations.entry(name).or_insert_with(Vec::new);
        Ok(())
    }

    /// Remove a tool from the registry (no-op if the tool is not registered).
    pub fn deregister(&self, name: &str) {
        if let Ok(mut schemas) = self.schemas.write() {
            schemas.remove(name);
        }
        if let Ok(mut invocations) = self.invocations.write() {
            invocations.remove(name);
        }
    }

    /// Look up the schema for a tool by name.
    pub fn get_schema(&self, name: &str) -> Option<ToolSchema> {
        self.schemas
            .read()
            .ok()
            .and_then(|g| g.get(name).cloned())
    }

    /// Validate a proposed invocation against the tool's schema.
    ///
    /// Returns `Ok(())` if all required parameters are present and their
    /// string values are consistent with the declared type.  Returns `Err`
    /// with a list of validation messages otherwise.
    pub fn validate_invocation(
        &self,
        name: &str,
        params: &HashMap<String, String>,
    ) -> Result<(), Vec<String>> {
        let schemas = self.schemas.read().map_err(|_| vec!["lock poisoned".into()])?;
        let schema = schemas
            .get(name)
            .ok_or_else(|| vec![format!("tool '{name}' is not registered")])?;

        let mut errors = Vec::new();

        for param in &schema.parameters {
            match params.get(&param.name) {
                None => {
                    if param.required && param.default_value.is_none() {
                        errors.push(format!(
                            "required parameter '{}' is missing",
                            param.name
                        ));
                    }
                }
                Some(value) => {
                    // Type-hint validation on the string representation.
                    let type_ok = match param.param_type {
                        ToolParamType::Integer => value.parse::<i64>().is_ok(),
                        ToolParamType::Float => value.parse::<f64>().is_ok(),
                        ToolParamType::Boolean => {
                            matches!(value.to_lowercase().as_str(), "true" | "false" | "1" | "0")
                        }
                        // String, Array, Object, Null — any non-empty value is
                        // accepted at this layer; deeper validation is up to the tool.
                        _ => true,
                    };
                    if !type_ok {
                        errors.push(format!(
                            "parameter '{}' expects type {} but got '{}'",
                            param.name, param.param_type, value
                        ));
                    }
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Append an invocation record for the named tool.
    ///
    /// If no history bucket exists yet (tool was never registered or was
    /// deregistered) the record is silently dropped.
    pub fn record_invocation(&self, invocation: ToolInvocation) {
        if let Ok(mut inv) = self.invocations.write() {
            inv.entry(invocation.tool_name.clone())
                .or_insert_with(Vec::new)
                .push(invocation);
        }
    }

    /// Compute aggregate statistics for a tool from its invocation history.
    ///
    /// Returns `None` if the tool is not registered.
    pub fn tool_stats(&self, name: &str) -> Option<ToolStats> {
        let inv = self.invocations.read().ok()?;
        let records = inv.get(name)?;

        let invocation_count = records.len() as u64;
        let success_count = records.iter().filter(|r| r.success).count() as u64;
        let failure_count = invocation_count - success_count;
        let total_result_bytes: u64 = records.iter().map(|r| r.result_size_bytes as u64).sum();

        let mut durations: Vec<f64> = records
            .iter()
            .filter_map(|r| r.duration_ms().map(|d| d as f64))
            .collect();
        durations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let avg_duration_ms = if durations.is_empty() {
            0.0
        } else {
            durations.iter().sum::<f64>() / durations.len() as f64
        };

        let p95_duration_ms = if durations.is_empty() {
            0.0
        } else {
            let idx = ((durations.len() as f64) * 0.95) as usize;
            let idx = idx.min(durations.len() - 1);
            durations[idx]
        };

        Some(ToolStats {
            invocation_count,
            success_count,
            failure_count,
            avg_duration_ms,
            p95_duration_ms,
            total_result_bytes,
        })
    }

    /// Return a snapshot of all registered tool schemas.
    pub fn all_tools(&self) -> Vec<ToolSchema> {
        self.schemas
            .read()
            .map(|g| g.values().cloned().collect())
            .unwrap_or_default()
    }

    /// Return all tools that belong to the given category.
    pub fn tools_by_category(&self, category: &str) -> Vec<ToolSchema> {
        self.schemas
            .read()
            .map(|g| {
                g.values()
                    .filter(|s| s.category == category)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Rank `schemas` by keyword overlap with `description`.
    ///
    /// Returns a list of `(tool_name, score)` pairs sorted by descending score,
    /// omitting tools with a score of zero.
    pub fn suggest_tool(description: &str, schemas: &[ToolSchema]) -> Vec<(String, f64)> {
        let query_words: Vec<&str> = description.split_whitespace().collect();
        if query_words.is_empty() {
            return Vec::new();
        }

        let mut ranked: Vec<(String, f64)> = schemas
            .iter()
            .filter_map(|schema| {
                let corpus = format!(
                    "{} {} {}",
                    schema.name,
                    schema.description,
                    schema.category
                )
                .to_lowercase();
                let hits = query_words
                    .iter()
                    .filter(|w| corpus.contains(&w.to_lowercase() as &str))
                    .count();
                if hits == 0 {
                    None
                } else {
                    let score = hits as f64 / query_words.len() as f64;
                    Some((schema.name.clone(), score))
                }
            })
            .collect();

        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }

    /// Produce a Markdown table listing all registered tools.
    pub fn registry_report(&self) -> String {
        let schemas = match self.schemas.read() {
            Ok(g) => g,
            Err(_) => return "Error: registry lock poisoned.\n".into(),
        };

        let mut lines = Vec::new();
        lines.push("| Name | Category | Version | Parameters | Description |".into());
        lines.push("|------|----------|---------|------------|-------------|".into());

        let mut sorted: Vec<&ToolSchema> = schemas.values().collect();
        sorted.sort_by_key(|s| &s.name);

        for schema in sorted {
            let param_names: Vec<&str> = schema.parameters.iter().map(|p| p.name.as_str()).collect();
            lines.push(format!(
                "| {} | {} | {} | {} | {} |",
                schema.name,
                schema.category,
                schema.version,
                param_names.join(", "),
                schema.description,
            ));
        }

        lines.join("\n")
    }
}

//! Hot-loadable plugin interface with manifest parsing and lifecycle hooks.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::Instant;

/// Manifest describing a plugin's identity and capabilities.
#[derive(Clone, Debug)]
pub struct PluginManifest {
    /// Unique plugin identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Semantic version string.
    pub version: String,
    /// Short description.
    pub description: String,
    /// Author name or contact.
    pub author: String,
    /// List of capability tokens this plugin provides.
    pub capabilities: Vec<String>,
    /// Plugin IDs this plugin depends on.
    pub dependencies: Vec<String>,
    /// Entry point symbol or path.
    pub entry_point: String,
}

/// Parse a simple TOML-like key=value manifest string.
///
/// Supports:
/// - `key = "quoted value"` (strings)
/// - `key = [item1, item2]` (arrays)
/// - Comments starting with `#`
pub fn parse_manifest(toml_str: &str) -> Result<PluginManifest, String> {
    let mut map: HashMap<String, String> = HashMap::new();
    let mut array_map: HashMap<String, Vec<String>> = HashMap::new();

    for line in toml_str.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let eq = line.find('=').ok_or_else(|| format!("no '=' in line: {line}"))?;
        let key = line[..eq].trim().to_string();
        let val = line[eq + 1..].trim();

        if val.starts_with('[') {
            // Array value
            let inner = val
                .trim_start_matches('[')
                .trim_end_matches(']')
                .trim();
            let items: Vec<String> = if inner.is_empty() {
                vec![]
            } else {
                inner
                    .split(',')
                    .map(|s| s.trim().trim_matches('"').to_string())
                    .collect()
            };
            array_map.insert(key, items);
        } else {
            // Scalar value — strip optional surrounding quotes
            let scalar = val.trim_matches('"').to_string();
            map.insert(key, scalar);
        }
    }

    let get = |k: &str| -> Result<String, String> {
        map.get(k)
            .cloned()
            .ok_or_else(|| format!("missing field: {k}"))
    };
    let get_arr = |k: &str| -> Vec<String> {
        array_map.get(k).cloned().unwrap_or_default()
    };

    Ok(PluginManifest {
        id: get("id")?,
        name: get("name")?,
        version: get("version")?,
        description: map.get("description").cloned().unwrap_or_default(),
        author: map.get("author").cloned().unwrap_or_default(),
        capabilities: get_arr("capabilities"),
        dependencies: get_arr("dependencies"),
        entry_point: get("entry_point")?,
    })
}

/// Lifecycle state of a plugin.
#[derive(Debug, PartialEq)]
pub enum PluginState {
    /// Manifest registered, not yet loaded.
    Discovered,
    /// Currently loading.
    Loading,
    /// Loaded and ready to receive commands.
    Active,
    /// Temporarily paused.
    Suspended,
    /// Failed with a reason.
    Failed {
        /// Human-readable failure reason.
        reason: String,
    },
    /// Unloaded and released.
    Unloaded,
}

/// Lifecycle hooks implemented by each plugin.
pub trait PluginHooks: Send + Sync {
    /// Called once when the plugin is activated.
    fn on_load(&self) -> Result<(), String>;
    /// Called when the plugin transitions to Active.
    fn on_activate(&self) -> Result<(), String>;
    /// Called when the plugin is suspended.
    fn on_suspend(&self);
    /// Called when the plugin is unloaded.
    fn on_unload(&self);
    /// Execute a command with arguments.
    fn execute(&self, command: &str, args: &[String]) -> Result<String, String>;
}

/// Registry entry for one plugin.
pub struct PluginEntry {
    /// Plugin manifest.
    pub manifest: PluginManifest,
    /// Current lifecycle state.
    pub state: PluginState,
    /// Optional hooks implementation.
    pub hooks: Option<Box<dyn PluginHooks>>,
    /// When this entry was created / loaded.
    pub load_time: Instant,
    /// Total successful `dispatch` calls.
    pub execution_count: AtomicU64,
    /// Last error message, if any.
    pub last_error: Option<String>,
}

/// Registry holding all plugins indexed by ID plus a capability index.
pub struct PluginRegistry {
    plugins: RwLock<HashMap<String, PluginEntry>>,
    capability_index: RwLock<HashMap<String, Vec<String>>>,
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            plugins: RwLock::new(HashMap::new()),
            capability_index: RwLock::new(HashMap::new()),
        }
    }

    /// Register a manifest. Validates that all declared dependency IDs are already registered.
    pub fn register_manifest(&self, manifest: PluginManifest) -> Result<(), String> {
        {
            let plugins = self.plugins.read().unwrap();
            for dep in &manifest.dependencies {
                if !plugins.contains_key(dep.as_str()) {
                    return Err(format!(
                        "dependency '{}' not registered before '{}'",
                        dep, manifest.id
                    ));
                }
            }
        }

        // Update capability index
        {
            let mut cap_idx = self.capability_index.write().unwrap();
            for cap in &manifest.capabilities {
                cap_idx
                    .entry(cap.clone())
                    .or_default()
                    .push(manifest.id.clone());
            }
        }

        let entry = PluginEntry {
            manifest,
            state: PluginState::Discovered,
            hooks: None,
            load_time: Instant::now(),
            execution_count: AtomicU64::new(0),
            last_error: None,
        };

        let mut plugins = self.plugins.write().unwrap();
        plugins.insert(entry.manifest.id.clone(), entry);
        Ok(())
    }

    /// Activate a plugin by providing its hooks implementation.
    pub fn activate(&self, plugin_id: &str, hooks: Box<dyn PluginHooks>) -> Result<(), String> {
        let mut plugins = self.plugins.write().unwrap();
        let entry = plugins
            .get_mut(plugin_id)
            .ok_or_else(|| format!("plugin not found: {plugin_id}"))?;

        entry.state = PluginState::Loading;
        hooks.on_load().map_err(|e| {
            entry.state = PluginState::Failed { reason: e.clone() };
            entry.last_error = Some(e.clone());
            e
        })?;
        hooks.on_activate().map_err(|e| {
            entry.state = PluginState::Failed { reason: e.clone() };
            entry.last_error = Some(e.clone());
            e
        })?;
        entry.hooks = Some(hooks);
        entry.state = PluginState::Active;
        Ok(())
    }

    /// Suspend an active plugin.
    pub fn suspend(&self, plugin_id: &str) {
        let mut plugins = self.plugins.write().unwrap();
        if let Some(entry) = plugins.get_mut(plugin_id) {
            if let Some(hooks) = &entry.hooks {
                hooks.on_suspend();
            }
            entry.state = PluginState::Suspended;
        }
    }

    /// Unload a plugin and release its hooks.
    pub fn unload(&self, plugin_id: &str) {
        let mut plugins = self.plugins.write().unwrap();
        if let Some(entry) = plugins.get_mut(plugin_id) {
            if let Some(hooks) = &entry.hooks {
                hooks.on_unload();
            }
            entry.hooks = None;
            entry.state = PluginState::Unloaded;
        }
    }

    /// Dispatch a command to a plugin. Plugin must be Active.
    pub fn dispatch(
        &self,
        plugin_id: &str,
        command: &str,
        args: &[String],
    ) -> Result<String, String> {
        let plugins = self.plugins.read().unwrap();
        let entry = plugins
            .get(plugin_id)
            .ok_or_else(|| format!("plugin not found: {plugin_id}"))?;
        if entry.state != PluginState::Active {
            return Err(format!("plugin '{plugin_id}' is not active"));
        }
        let hooks = entry
            .hooks
            .as_ref()
            .ok_or_else(|| format!("no hooks for plugin: {plugin_id}"))?;
        let result = hooks.execute(command, args)?;
        entry.execution_count.fetch_add(1, Ordering::Relaxed);
        Ok(result)
    }

    /// Return all plugin IDs that advertise the given capability.
    pub fn plugins_with_capability(&self, cap: &str) -> Vec<String> {
        let cap_idx = self.capability_index.read().unwrap();
        cap_idx.get(cap).cloned().unwrap_or_default()
    }

    /// Topological sort of plugins by their dependency relationships.
    /// Returns plugin IDs in load order (dependencies before dependents).
    pub fn dependency_order(&self) -> Vec<String> {
        let plugins = self.plugins.read().unwrap();
        let mut visited: HashMap<String, bool> = HashMap::new();
        let mut order: Vec<String> = Vec::new();

        fn visit(
            id: &str,
            plugins: &HashMap<String, PluginEntry>,
            visited: &mut HashMap<String, bool>,
            order: &mut Vec<String>,
        ) {
            if visited.get(id).copied().unwrap_or(false) {
                return;
            }
            visited.insert(id.to_string(), true);
            if let Some(entry) = plugins.get(id) {
                for dep in &entry.manifest.dependencies {
                    visit(dep, plugins, visited, order);
                }
            }
            order.push(id.to_string());
        }

        let ids: Vec<String> = plugins.keys().cloned().collect();
        for id in &ids {
            visit(id, &plugins, &mut visited, &mut order);
        }
        order
    }

    /// Return `(total, active, failed)` counts.
    pub fn registry_stats(&self) -> (usize, usize, usize) {
        let plugins = self.plugins.read().unwrap();
        let total = plugins.len();
        let active = plugins
            .values()
            .filter(|e| e.state == PluginState::Active)
            .count();
        let failed = plugins
            .values()
            .filter(|e| matches!(e.state, PluginState::Failed { .. }))
            .count();
        (total, active, failed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NoopHooks;
    impl PluginHooks for NoopHooks {
        fn on_load(&self) -> Result<(), String> { Ok(()) }
        fn on_activate(&self) -> Result<(), String> { Ok(()) }
        fn on_suspend(&self) {}
        fn on_unload(&self) {}
        fn execute(&self, _cmd: &str, _args: &[String]) -> Result<String, String> {
            Ok("ok".to_string())
        }
    }

    #[test]
    fn parse_and_register() {
        let toml = r#"
id = "test-plugin"
name = "Test Plugin"
version = "1.0.0"
description = "A test plugin"
author = "Tester"
capabilities = [compute, search]
dependencies = []
entry_point = "test_entry"
"#;
        let manifest = parse_manifest(toml).unwrap();
        assert_eq!(manifest.id, "test-plugin");
        assert_eq!(manifest.capabilities, vec!["compute", "search"]);

        let registry = PluginRegistry::new();
        registry.register_manifest(manifest).unwrap();
        registry
            .activate("test-plugin", Box::new(NoopHooks))
            .unwrap();

        let result = registry
            .dispatch("test-plugin", "run", &[])
            .unwrap();
        assert_eq!(result, "ok");

        let (total, active, failed) = registry.registry_stats();
        assert_eq!((total, active, failed), (1, 1, 0));
    }
}

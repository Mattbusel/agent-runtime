//! # Module: Agent Skill Marketplace
//!
//! ## Responsibility
//! The marketplace provides a runtime-discoverable registry of named
//! [`Skill`]s.  It extends the lower-level [`crate::skills`] TOML-based
//! library with:
//!
//! - A richer [`Skill`] *struct* (distinct from the `skills::Skill` *trait*)
//!   carrying versioning, authorship, and capability metadata.
//! - [`SkillRegistry`] — a global, `Arc`-backed, thread-safe store of all
//!   known skills.
//! - [`SkillLoader`] — discovers and loads `*.toml` skill manifests from a
//!   configurable directory (defaults to `~/.agent-runtime/skills/`).
//! - [`SkillMatcher`] — given a task description, returns the best-matching
//!   skills by tag/keyword overlap.
//! - [`SkillComposer`] — chains multiple skills into an ordered pipeline for
//!   complex tasks.
//!
//! ## Example
//!
//! ```rust,no_run
//! use llm_agent_runtime::marketplace::{
//!     Skill, SkillRegistry, SkillMatcher, SkillComposer, SkillPipeline,
//! };
//! use std::sync::Arc;
//!
//! # tokio_test::block_on(async {
//! // Build a shared registry.
//! let registry = Arc::new(SkillRegistry::new());
//!
//! // Register skills manually.
//! registry.register(Skill {
//!     name: "web_search".into(),
//!     description: "Search the web for up-to-date information".into(),
//!     version: "1.0.0".into(),
//!     author: "core".into(),
//!     capabilities: vec!["search".into(), "web".into(), "retrieval".into()],
//! });
//!
//! registry.register(Skill {
//!     name: "summarise".into(),
//!     description: "Condense text into a concise summary".into(),
//!     version: "1.0.0".into(),
//!     author: "core".into(),
//!     capabilities: vec!["nlp".into(), "summarise".into(), "text".into()],
//! });
//!
//! // Match skills for a task.
//! let matcher = SkillMatcher::new(Arc::clone(&registry));
//! let matches = matcher.match_skills("I need to search the web and summarise results", 3);
//! for m in &matches {
//!     println!("Skill '{}' — score {:.2}", m.skill.name, m.score);
//! }
//!
//! // Compose a pipeline from matched skills.
//! let composer = SkillComposer::new(Arc::clone(&registry));
//! let pipeline = composer.compose(vec!["web_search".into(), "summarise".into()]);
//! println!("Pipeline: {:?}", pipeline.stage_names());
//! # });
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

// ── Skill ─────────────────────────────────────────────────────────────────────

/// A named, versioned, discoverable skill in the marketplace.
///
/// Unlike `skills::Skill` (a trait for runtime execution), this struct
/// describes a skill's *identity and capabilities* for discovery and matching.
/// Execution is delegated to [`crate::skills::SkillLibrary`] once the right
/// skill is selected.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Skill {
    /// Unique machine-readable name (e.g. `"web_search"`).
    pub name: String,
    /// Human-readable description of what the skill does.
    pub description: String,
    /// Semantic version string (e.g. `"1.2.0"`).
    pub version: String,
    /// Author or team that owns this skill.
    pub author: String,
    /// Capability tags used for matching (e.g. `["search", "web", "retrieval"]`).
    pub capabilities: Vec<String>,
}

impl Skill {
    /// Create a new skill with the given metadata.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        version: impl Into<String>,
        author: impl Into<String>,
        capabilities: Vec<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            version: version.into(),
            author: author.into(),
            capabilities,
        }
    }

    /// Return the capability set as a `HashSet<String>` for set operations.
    pub fn capability_set(&self) -> HashSet<String> {
        self.capabilities.iter().cloned().collect()
    }
}

// ── Skill manifest (TOML) ─────────────────────────────────────────────────────

/// TOML-deserialisable manifest for a single skill file.
#[derive(Debug, Deserialize)]
struct SkillManifest {
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default = "default_version")]
    version: String,
    #[serde(default)]
    author: String,
    #[serde(default)]
    capabilities: Vec<String>,
}

fn default_version() -> String {
    "0.1.0".to_string()
}

impl From<SkillManifest> for Skill {
    fn from(m: SkillManifest) -> Self {
        Self {
            name: m.name,
            description: m.description,
            version: m.version,
            author: m.author,
            capabilities: m.capabilities,
        }
    }
}

// ── SkillRegistry ─────────────────────────────────────────────────────────────

/// Global registry of available [`Skill`]s.
///
/// The registry is `Arc<SkillRegistry>` so it can be shared across threads
/// and tasks without cloning the skill data.
#[derive(Default)]
pub struct SkillRegistry {
    skills: RwLock<HashMap<String, Skill>>,
}

impl SkillRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a skill.  If a skill with the same name already exists it is
    /// replaced.
    pub fn register(&self, skill: Skill) {
        let mut guard = self.skills.write().unwrap_or_else(|e| e.into_inner());
        guard.insert(skill.name.clone(), skill);
    }

    /// Remove a skill by name.  Returns the removed skill if it existed.
    pub fn deregister(&self, name: &str) -> Option<Skill> {
        let mut guard = self.skills.write().unwrap_or_else(|e| e.into_inner());
        guard.remove(name)
    }

    /// Look up a skill by name.
    pub fn get(&self, name: &str) -> Option<Skill> {
        let guard = self.skills.read().unwrap_or_else(|e| e.into_inner());
        guard.get(name).cloned()
    }

    /// Return all registered skills sorted by name.
    pub fn all(&self) -> Vec<Skill> {
        let guard = self.skills.read().unwrap_or_else(|e| e.into_inner());
        let mut skills: Vec<Skill> = guard.values().cloned().collect();
        skills.sort_by(|a, b| a.name.cmp(&b.name));
        skills
    }

    /// Return the number of registered skills.
    pub fn len(&self) -> usize {
        let guard = self.skills.read().unwrap_or_else(|e| e.into_inner());
        guard.len()
    }

    /// Returns `true` when the registry contains no skills.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ── SkillLoader ───────────────────────────────────────────────────────────────

/// Discovers and loads `*.toml` skill manifests from a directory.
///
/// Default search path: `~/.agent-runtime/skills/` (created on demand by
/// callers; the loader does not create directories itself).
pub struct SkillLoader {
    /// Directory to scan for `*.toml` skill manifest files.
    pub skills_dir: PathBuf,
}

impl Default for SkillLoader {
    fn default() -> Self {
        // Prefer `~/.agent-runtime/skills/`; fall back to CWD if home is not
        // available.
        let dir = home_dir()
            .map(|h| h.join(".agent-runtime").join("skills"))
            .unwrap_or_else(|| PathBuf::from("skills"));
        Self { skills_dir: dir }
    }
}

impl SkillLoader {
    /// Create a loader that reads from `skills_dir`.
    pub fn new(skills_dir: impl Into<PathBuf>) -> Self {
        Self { skills_dir: skills_dir.into() }
    }

    /// Scan [`Self::skills_dir`] for `*.toml` files, parse each as a
    /// [`SkillManifest`], and register the resulting [`Skill`]s into
    /// `registry`.
    ///
    /// Files that fail to parse are logged via `tracing::warn` and skipped;
    /// the method never returns an error so a single malformed file does not
    /// block loading the rest.
    ///
    /// Returns the number of skills successfully loaded.
    pub fn load_into(&self, registry: &SkillRegistry) -> usize {
        let dir = &self.skills_dir;
        if !dir.exists() {
            tracing::debug!(
                path = %dir.display(),
                "SkillLoader: skills directory does not exist, skipping"
            );
            return 0;
        }

        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(err) => {
                tracing::warn!(path = %dir.display(), error = %err, "SkillLoader: cannot read skills directory");
                return 0;
            }
        };

        let mut loaded = 0usize;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("toml") {
                continue;
            }
            match self.load_file(&path) {
                Ok(skill) => {
                    tracing::debug!(name = %skill.name, path = %path.display(), "SkillLoader: loaded skill");
                    registry.register(skill);
                    loaded += 1;
                }
                Err(err) => {
                    tracing::warn!(path = %path.display(), error = %err, "SkillLoader: failed to parse skill manifest");
                }
            }
        }
        loaded
    }

    /// Parse a single TOML file into a [`Skill`].
    fn load_file(&self, path: &Path) -> Result<Skill, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("I/O error reading {}: {e}", path.display()))?;
        let manifest: SkillManifest = toml::from_str(&content)
            .map_err(|e| format!("TOML parse error in {}: {e}", path.display()))?;
        Ok(Skill::from(manifest))
    }
}

/// Attempt to determine the current user's home directory.
fn home_dir() -> Option<PathBuf> {
    // `std::env::home_dir` is deprecated but is the only stdlib option on all
    // platforms.  We use it here as a best-effort fallback only.
    #[allow(deprecated)]
    std::env::home_dir()
}

// ── SkillMatch ────────────────────────────────────────────────────────────────

/// A skill returned by [`SkillMatcher`] with an associated relevance score.
#[derive(Debug, Clone)]
pub struct SkillMatch {
    /// The matched skill.
    pub skill: Skill,
    /// Relevance score in `[0.0, 1.0]` — higher is more relevant.
    pub score: f32,
}

// ── SkillMatcher ──────────────────────────────────────────────────────────────

/// Given a task description, returns the best-matching skills from a
/// [`SkillRegistry`].
///
/// ## Scoring
/// For each skill the score is computed as:
///
/// ```text
/// score = capability_overlap(task_words ∩ skill_capabilities) / |skill_capabilities|
///       + description_overlap(task_words ∩ description_words) / |description_words|
///       (each term contributes up to 0.5)
/// ```
///
/// Skills are returned sorted by score descending.  A `min_score` filter
/// (default `0.1`) eliminates skills with no real overlap.
pub struct SkillMatcher {
    registry: Arc<SkillRegistry>,
    /// Minimum score a skill must achieve to appear in results.
    pub min_score: f32,
}

impl SkillMatcher {
    /// Create a matcher backed by `registry` with the default `min_score = 0.1`.
    pub fn new(registry: Arc<SkillRegistry>) -> Self {
        Self { registry, min_score: 0.1 }
    }

    /// Set the minimum relevance score threshold.
    pub fn with_min_score(mut self, min_score: f32) -> Self {
        self.min_score = min_score.clamp(0.0, 1.0);
        self
    }

    /// Return up to `limit` best-matching skills for `task_description`.
    ///
    /// Pass `limit = 0` to return all matches above the threshold.
    pub fn match_skills(&self, task_description: &str, limit: usize) -> Vec<SkillMatch> {
        let task_words = Self::normalise(task_description);
        let skills = self.registry.all();

        let mut scored: Vec<SkillMatch> = skills
            .into_iter()
            .filter_map(|skill| {
                let score = self.score(&skill, &task_words);
                if score >= self.min_score {
                    Some(SkillMatch { skill, score })
                } else {
                    None
                }
            })
            .collect();

        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        if limit > 0 {
            scored.truncate(limit);
        }
        scored
    }

    /// Score a skill against a normalised task word set.
    fn score(&self, skill: &Skill, task_words: &HashSet<String>) -> f32 {
        // Capability overlap component (0.0–0.5).
        let cap_set: HashSet<String> = skill
            .capabilities
            .iter()
            .map(|c| c.to_lowercase())
            .collect();
        let cap_intersection = task_words.intersection(&cap_set).count();
        let cap_score = if cap_set.is_empty() {
            0.0_f32
        } else {
            0.5 * (cap_intersection as f32 / cap_set.len() as f32)
        };

        // Description overlap component (0.0–0.5).
        let desc_words = Self::normalise(&skill.description);
        let desc_intersection = task_words.intersection(&desc_words).count();
        let desc_score = if desc_words.is_empty() {
            0.0_f32
        } else {
            0.5 * (desc_intersection as f32 / desc_words.len() as f32)
        };

        (cap_score + desc_score).clamp(0.0, 1.0)
    }

    /// Lowercase, strip punctuation, and split into a word set, filtering
    /// common stop-words.
    fn normalise(text: &str) -> HashSet<String> {
        const STOP_WORDS: &[&str] = &[
            "a", "an", "the", "is", "are", "was", "be", "been", "being",
            "to", "of", "and", "or", "for", "in", "on", "at", "with",
            "by", "from", "that", "this", "it", "i", "need", "want",
            "please", "can", "would", "my", "me", "you",
        ];
        let stop: HashSet<&str> = STOP_WORDS.iter().copied().collect();
        text.split_whitespace()
            .map(|w| {
                w.chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
                    .to_lowercase()
            })
            .filter(|w| !w.is_empty() && !stop.contains(w.as_str()))
            .collect()
    }
}

// ── SkillPipeline ─────────────────────────────────────────────────────────────

/// An ordered sequence of skills to be executed as a pipeline for a complex
/// task.
///
/// Produced by [`SkillComposer::compose`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillPipeline {
    /// Ordered stages (skill names).
    stages: Vec<String>,
    /// Optional human-readable description of the pipeline's purpose.
    pub description: String,
}

impl SkillPipeline {
    /// Return the ordered list of stage skill names.
    pub fn stage_names(&self) -> &[String] {
        &self.stages
    }

    /// Return the number of stages.
    pub fn len(&self) -> usize {
        self.stages.len()
    }

    /// Returns `true` when the pipeline has no stages.
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }
}

// ── SkillComposer ─────────────────────────────────────────────────────────────

/// Chains multiple skills from the registry into an ordered [`SkillPipeline`].
///
/// Skills that are not found in the registry are silently skipped with a
/// `tracing::warn` log so that pipelines degrade gracefully when optional
/// skills are absent.
pub struct SkillComposer {
    registry: Arc<SkillRegistry>,
}

impl SkillComposer {
    /// Create a new composer backed by `registry`.
    pub fn new(registry: Arc<SkillRegistry>) -> Self {
        Self { registry }
    }

    /// Build a [`SkillPipeline`] from an ordered list of skill names.
    ///
    /// Only skill names that exist in the registry are included.
    pub fn compose(&self, skill_names: Vec<String>) -> SkillPipeline {
        let stages: Vec<String> = skill_names
            .into_iter()
            .filter(|name| {
                if self.registry.get(name).is_some() {
                    true
                } else {
                    tracing::warn!(skill = %name, "SkillComposer: skill not found in registry, skipping");
                    false
                }
            })
            .collect();

        let description = if stages.is_empty() {
            "Empty pipeline (no matching skills found)".to_string()
        } else {
            format!("Pipeline: {}", stages.join(" → "))
        };

        SkillPipeline { stages, description }
    }

    /// Automatically compose a pipeline for `task_description` by matching
    /// the top `limit` skills and ordering them by relevance score descending.
    ///
    /// This combines [`SkillMatcher`] with [`SkillComposer::compose`].
    pub fn compose_for_task(&self, task_description: &str, limit: usize) -> SkillPipeline {
        let matcher = SkillMatcher::new(Arc::clone(&self.registry));
        let matches = matcher.match_skills(task_description, limit);
        let names: Vec<String> = matches.into_iter().map(|m| m.skill.name).collect();
        self.compose(names)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn sample_skill(name: &str, caps: &[&str]) -> Skill {
        Skill::new(
            name,
            format!("A skill for {name}"),
            "1.0.0",
            "test",
            caps.iter().map(|s| s.to_string()).collect(),
        )
    }

    fn populated_registry() -> Arc<SkillRegistry> {
        let reg = Arc::new(SkillRegistry::new());
        reg.register(sample_skill("web_search", &["search", "web", "retrieval"]));
        reg.register(sample_skill("summarise", &["nlp", "summarise", "text"]));
        reg.register(sample_skill("code_review", &["code", "review", "analysis"]));
        reg.register(sample_skill("translate", &["nlp", "translation", "language"]));
        reg
    }

    // ── SkillRegistry ─────────────────────────────────────────────────────────

    #[test]
    fn registry_register_and_get() {
        let reg = SkillRegistry::new();
        let skill = sample_skill("test_skill", &["test"]);
        reg.register(skill.clone());
        let fetched = reg.get("test_skill").unwrap();
        assert_eq!(fetched.name, "test_skill");
    }

    #[test]
    fn registry_deregister_removes_skill() {
        let reg = SkillRegistry::new();
        reg.register(sample_skill("to_remove", &[]));
        assert!(!reg.is_empty());
        let removed = reg.deregister("to_remove");
        assert!(removed.is_some());
        assert!(reg.is_empty());
    }

    #[test]
    fn registry_all_returns_sorted() {
        let reg = populated_registry();
        let names: Vec<&str> = reg.all().iter().map(|s| s.name.as_str()).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted);
    }

    #[test]
    fn registry_get_missing_returns_none() {
        let reg = SkillRegistry::new();
        assert!(reg.get("nonexistent").is_none());
    }

    #[test]
    fn registry_len_and_is_empty() {
        let reg = SkillRegistry::new();
        assert!(reg.is_empty());
        reg.register(sample_skill("s1", &[]));
        assert_eq!(reg.len(), 1);
    }

    // ── SkillMatcher ──────────────────────────────────────────────────────────

    #[test]
    fn matcher_returns_relevant_skills() {
        let reg = populated_registry();
        let matcher = SkillMatcher::new(reg);
        let matches = matcher.match_skills("search the web for information", 5);
        assert!(!matches.is_empty());
        assert_eq!(matches[0].skill.name, "web_search");
    }

    #[test]
    fn matcher_limits_results() {
        let reg = populated_registry();
        let matcher = SkillMatcher::new(reg);
        let matches = matcher.match_skills("nlp text analysis code", 2);
        assert!(matches.len() <= 2);
    }

    #[test]
    fn matcher_returns_sorted_by_score_descending() {
        let reg = populated_registry();
        let matcher = SkillMatcher::new(reg);
        let matches = matcher.match_skills("search web retrieval", 5);
        if matches.len() >= 2 {
            assert!(matches[0].score >= matches[1].score);
        }
    }

    #[test]
    fn matcher_respects_min_score_filter() {
        let reg = populated_registry();
        let matcher = SkillMatcher::new(reg).with_min_score(0.9);
        // A vague task that doesn't strongly match any skill.
        let matches = matcher.match_skills("do something", 10);
        // All returned matches should be above the threshold.
        for m in &matches {
            assert!(m.score >= 0.9);
        }
    }

    // ── SkillComposer ─────────────────────────────────────────────────────────

    #[test]
    fn composer_builds_pipeline_from_names() {
        let reg = populated_registry();
        let composer = SkillComposer::new(reg);
        let pipeline = composer.compose(vec!["web_search".into(), "summarise".into()]);
        assert_eq!(pipeline.len(), 2);
        assert_eq!(pipeline.stage_names(), &["web_search", "summarise"]);
    }

    #[test]
    fn composer_skips_missing_skills() {
        let reg = populated_registry();
        let composer = SkillComposer::new(reg);
        let pipeline = composer.compose(vec!["web_search".into(), "nonexistent".into()]);
        assert_eq!(pipeline.len(), 1);
    }

    #[test]
    fn composer_builds_pipeline_for_task() {
        let reg = populated_registry();
        let composer = SkillComposer::new(reg);
        let pipeline = composer.compose_for_task("search the web for recent news", 3);
        assert!(!pipeline.is_empty());
        assert!(pipeline.description.contains("→") || pipeline.description.contains("Empty"));
    }

    #[test]
    fn composer_empty_pipeline_when_no_matches() {
        let reg = Arc::new(SkillRegistry::new()); // empty
        let composer = SkillComposer::new(reg);
        let pipeline = composer.compose(vec!["missing".into()]);
        assert!(pipeline.is_empty());
    }

    // ── SkillPipeline ─────────────────────────────────────────────────────────

    #[test]
    fn pipeline_is_empty_for_no_stages() {
        let p = SkillPipeline { stages: vec![], description: "empty".into() };
        assert!(p.is_empty());
    }

    #[test]
    fn pipeline_stage_names_returns_ordered_slice() {
        let p = SkillPipeline {
            stages: vec!["a".into(), "b".into(), "c".into()],
            description: "test".into(),
        };
        assert_eq!(p.stage_names(), &["a", "b", "c"]);
    }

    // ── Skill struct ──────────────────────────────────────────────────────────

    #[test]
    fn skill_capability_set() {
        let skill = sample_skill("s", &["nlp", "search", "search"]); // dupe
        let caps = skill.capability_set();
        assert!(caps.contains("nlp"));
        assert!(caps.contains("search"));
        // HashSet deduplicates.
        assert_eq!(caps.len(), 2);
    }

    // ── SkillLoader (file-free path) ──────────────────────────────────────────

    #[test]
    fn skill_loader_nonexistent_dir_loads_zero() {
        let loader = SkillLoader::new("/nonexistent/path/that/does/not/exist");
        let reg = SkillRegistry::new();
        let count = loader.load_into(&reg);
        assert_eq!(count, 0);
    }
}

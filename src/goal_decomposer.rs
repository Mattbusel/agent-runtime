//! # Module: goal_decomposer
//!
//! ## Responsibility
//! Decompose high-level goals into subtasks using rule-based heuristics.
//! Supports Sequential, Parallel, Hierarchical, and Iterative strategies,
//! conjunction-based splitting, iterative pattern detection, and topological
//! ordering of subtasks by their dependency graph.

use std::collections::{HashMap, HashSet, VecDeque};

// ── Data structures ───────────────────────────────────────────────────────────

/// A high-level goal to be decomposed.
#[derive(Debug, Clone)]
pub struct Goal {
    /// Unique identifier.
    pub id: String,
    /// Natural-language description of the goal.
    pub description: String,
    /// Scheduling priority 1 (lowest) – 10 (highest).
    pub priority: u8,
    /// Optional wall-clock deadline in milliseconds since epoch.
    pub deadline_ms: Option<u64>,
}

/// A concrete subtask derived from a [`Goal`].
#[derive(Debug, Clone)]
pub struct Subtask {
    /// Unique identifier.
    pub id: String,
    /// Parent goal this subtask belongs to.
    pub parent_goal_id: String,
    /// Human-readable description of the subtask.
    pub description: String,
    /// Estimated token cost for this subtask.
    pub estimated_tokens: u64,
    /// IDs of subtasks that must complete before this one can start.
    pub dependencies: Vec<String>,
}

/// Strategy used when decomposing a [`Goal`] into [`Subtask`]s.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecompositionStrategy {
    /// Subtasks are ordered sequentially; each depends on the previous.
    Sequential,
    /// All subtasks are independent and can run concurrently.
    Parallel,
    /// Subtasks form a hierarchy: the first is a parent, rest are children.
    Hierarchical,
    /// Iterative pattern: repeated steps detected in the goal description.
    Iterative,
}

// ── GoalDecomposer ────────────────────────────────────────────────────────────

/// Decomposes goals into subtasks using rule-based heuristics.
#[derive(Debug, Default)]
pub struct GoalDecomposer;

/// Conjunction keywords used to split goal descriptions into parts.
const CONJUNCTIONS: &[&str] = &[" and ", " then ", " also ", " while "];

/// Iterative keywords that trigger the [`DecompositionStrategy::Iterative`] path.
const ITERATIVE_KEYWORDS: &[&str] = &["for each", "repeatedly", "until", "iterate", "loop"];

impl GoalDecomposer {
    /// Create a new `GoalDecomposer`.
    pub fn new() -> Self {
        Self
    }

    /// Decompose `goal` into [`Subtask`]s using the given `strategy`.
    ///
    /// - **Sequential** — split on conjunctions; each subtask depends on the previous.
    /// - **Parallel** — split on conjunctions; all subtasks are independent.
    /// - **Hierarchical** — first part becomes the parent subtask; subsequent parts
    ///   each depend only on the parent.
    /// - **Iterative** — detect iterative phrases; produce a numbered sequence of
    ///   repeated steps (falls back to 3 iterations when no count can be inferred).
    pub fn decompose(&self, goal: &Goal, strategy: DecompositionStrategy) -> Vec<Subtask> {
        let parts = self.split_on_conjunctions(&goal.description);

        match strategy {
            DecompositionStrategy::Sequential => self.build_sequential(goal, parts),
            DecompositionStrategy::Parallel => self.build_parallel(goal, parts),
            DecompositionStrategy::Hierarchical => self.build_hierarchical(goal, parts),
            DecompositionStrategy::Iterative => self.build_iterative(goal, parts),
        }
    }

    /// Estimate the complexity of a goal on a scale of 1–5.
    ///
    /// Scoring heuristic:
    /// - Base score 1.
    /// - +1 if description length > 60 characters.
    /// - +1 if description length > 120 characters.
    /// - +1 if there are 3+ distinct keyword indicators (iterative keywords or
    ///   domain-specific verbs: "analyse", "generate", "optimize", "deploy").
    /// - +1 if conjunction count ≥ 3.
    pub fn estimate_complexity(&self, goal: &Goal) -> u8 {
        let desc = goal.description.to_lowercase();
        let mut score: u8 = 1;

        if desc.len() > 60 {
            score = score.saturating_add(1);
        }
        if desc.len() > 120 {
            score = score.saturating_add(1);
        }

        let complexity_keywords: &[&str] = &[
            "analyse", "analyze", "generate", "optimize", "optimise",
            "deploy", "migrate", "integrate", "evaluate", "transform",
            "for each", "repeatedly", "until", "iterate", "loop",
        ];
        let kw_count = complexity_keywords
            .iter()
            .filter(|&&kw| desc.contains(kw))
            .count();
        if kw_count >= 3 {
            score = score.saturating_add(1);
        }

        let conjunction_count = CONJUNCTIONS
            .iter()
            .map(|&c| desc.matches(c).count())
            .sum::<usize>();
        if conjunction_count >= 3 {
            score = score.saturating_add(1);
        }

        score.min(5)
    }

    /// Sort `subtasks` in-place by topological dependency order (Kahn's algorithm).
    ///
    /// Subtasks with no dependencies come first; dependents follow once all their
    /// prerequisites have been ordered.  Cycles are broken by preserving original
    /// order for the remainder.
    pub fn prioritize_subtasks(&self, subtasks: &mut Vec<Subtask>) {
        let n = subtasks.len();
        if n == 0 {
            return;
        }

        // Build index map: id -> position.
        let index: HashMap<String, usize> = subtasks
            .iter()
            .enumerate()
            .map(|(i, s)| (s.id.clone(), i))
            .collect();

        // in-degree count and adjacency list.
        let mut in_degree = vec![0usize; n];
        let mut adj: Vec<Vec<usize>> = vec![vec![]; n];

        for (i, subtask) in subtasks.iter().enumerate() {
            for dep_id in &subtask.dependencies {
                if let Some(&j) = index.get(dep_id) {
                    adj[j].push(i);
                    in_degree[i] += 1;
                }
            }
        }

        let mut queue: VecDeque<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut sorted_indices: Vec<usize> = Vec::with_capacity(n);

        while let Some(node) = queue.pop_front() {
            sorted_indices.push(node);
            for &neighbour in &adj[node] {
                in_degree[neighbour] -= 1;
                if in_degree[neighbour] == 0 {
                    queue.push_back(neighbour);
                }
            }
        }

        // Append any nodes involved in cycles, preserving original order.
        let seen: HashSet<usize> = sorted_indices.iter().copied().collect();
        for i in 0..n {
            if !seen.contains(&i) {
                sorted_indices.push(i);
            }
        }

        // Re-order the Vec according to `sorted_indices`.
        let original: Vec<Subtask> = std::mem::take(subtasks);
        for idx in sorted_indices {
            subtasks.push(original[idx].clone());
        }
    }

    /// Group subtasks into execution *waves*.
    ///
    /// - Wave 0: subtasks with no dependencies.
    /// - Wave k: subtasks whose dependencies are all satisfied by waves 0..k-1.
    pub fn completion_plan(&self, subtasks: &[Subtask]) -> Vec<Vec<String>> {
        let id_set: HashSet<&str> = subtasks.iter().map(|s| s.id.as_str()).collect();

        // Map id -> deps (filtered to only those present in the slice).
        let mut remaining_deps: HashMap<&str, HashSet<&str>> = subtasks
            .iter()
            .map(|s| {
                let deps: HashSet<&str> = s
                    .dependencies
                    .iter()
                    .map(|d| d.as_str())
                    .filter(|d| id_set.contains(d))
                    .collect();
                (s.id.as_str(), deps)
            })
            .collect();

        let mut waves: Vec<Vec<String>> = Vec::new();
        let mut completed: HashSet<&str> = HashSet::new();

        loop {
            let wave: Vec<&str> = remaining_deps
                .iter()
                .filter(|(_, deps)| deps.is_empty())
                .map(|(&id, _)| id)
                .collect();

            if wave.is_empty() {
                break;
            }

            let mut wave_sorted: Vec<&str> = wave;
            wave_sorted.sort_unstable();
            waves.push(wave_sorted.iter().map(|s| s.to_string()).collect());

            for id in &wave_sorted {
                completed.insert(id);
                remaining_deps.remove(id);
            }

            // Remove completed ids from remaining deps.
            for deps in remaining_deps.values_mut() {
                for id in &wave_sorted {
                    deps.remove(id);
                }
            }
        }

        // If anything remains (cycle), emit as a final wave.
        if !remaining_deps.is_empty() {
            let mut leftover: Vec<String> =
                remaining_deps.keys().map(|s| s.to_string()).collect();
            leftover.sort_unstable();
            waves.push(leftover);
        }

        let _ = completed; // suppress unused warning
        waves
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Split `description` on conjunction keywords, trimming each part.
    fn split_on_conjunctions(&self, description: &str) -> Vec<String> {
        let lower = description.to_lowercase();
        let mut splits: Vec<usize> = Vec::new();

        for &conj in CONJUNCTIONS {
            let mut start = 0;
            while let Some(pos) = lower[start..].find(conj) {
                splits.push(start + pos + conj.len());
                start += pos + conj.len();
            }
        }

        if splits.is_empty() {
            return vec![description.trim().to_string()];
        }

        splits.sort_unstable();
        splits.dedup();

        let mut parts: Vec<String> = Vec::new();
        let mut prev = 0usize;

        for split_pos in splits {
            // Find the corresponding original-case slice boundary.
            // Because we searched in `lower` but apply cuts to `description`:
            let part = description[prev..split_pos]
                .trim_end_matches(|c: char| {
                    // strip trailing conjunction words
                    CONJUNCTIONS.iter().any(|&k| {
                        description[prev..split_pos]
                            .to_lowercase()
                            .ends_with(k.trim())
                    }) && !c.is_alphanumeric()
                })
                .trim()
                .to_string();
            if !part.is_empty() {
                parts.push(part);
            }
            prev = split_pos;
        }
        let tail = description[prev..].trim().to_string();
        if !tail.is_empty() {
            parts.push(tail);
        }

        if parts.is_empty() {
            vec![description.trim().to_string()]
        } else {
            parts
        }
    }

    fn make_id(goal_id: &str, index: usize) -> String {
        format!("{}-subtask-{}", goal_id, index)
    }

    fn token_estimate(description: &str) -> u64 {
        // Rough heuristic: ~1.3 tokens per character / 4 chars per token ≈ word count * 1.5.
        let words = description.split_whitespace().count();
        (words as u64).saturating_mul(3).max(10)
    }

    fn build_sequential(&self, goal: &Goal, parts: Vec<String>) -> Vec<Subtask> {
        let mut subtasks = Vec::with_capacity(parts.len());
        for (i, part) in parts.iter().enumerate() {
            let id = Self::make_id(&goal.id, i);
            let dependencies = if i == 0 {
                vec![]
            } else {
                vec![Self::make_id(&goal.id, i - 1)]
            };
            subtasks.push(Subtask {
                id,
                parent_goal_id: goal.id.clone(),
                description: part.clone(),
                estimated_tokens: Self::token_estimate(part),
                dependencies,
            });
        }
        subtasks
    }

    fn build_parallel(&self, goal: &Goal, parts: Vec<String>) -> Vec<Subtask> {
        parts
            .iter()
            .enumerate()
            .map(|(i, part)| Subtask {
                id: Self::make_id(&goal.id, i),
                parent_goal_id: goal.id.clone(),
                description: part.clone(),
                estimated_tokens: Self::token_estimate(part),
                dependencies: vec![],
            })
            .collect()
    }

    fn build_hierarchical(&self, goal: &Goal, parts: Vec<String>) -> Vec<Subtask> {
        if parts.is_empty() {
            return vec![];
        }
        let parent_id = Self::make_id(&goal.id, 0);
        let mut subtasks = vec![Subtask {
            id: parent_id.clone(),
            parent_goal_id: goal.id.clone(),
            description: parts[0].clone(),
            estimated_tokens: Self::token_estimate(&parts[0]),
            dependencies: vec![],
        }];
        for (i, part) in parts.iter().enumerate().skip(1) {
            subtasks.push(Subtask {
                id: Self::make_id(&goal.id, i),
                parent_goal_id: goal.id.clone(),
                description: part.clone(),
                estimated_tokens: Self::token_estimate(part),
                dependencies: vec![parent_id.clone()],
            });
        }
        subtasks
    }

    fn build_iterative(&self, goal: &Goal, parts: Vec<String>) -> Vec<Subtask> {
        let desc = goal.description.to_lowercase();
        let has_iterative = ITERATIVE_KEYWORDS.iter().any(|&kw| desc.contains(kw));

        if has_iterative {
            // Produce 3 repeated iterations of the whole goal description.
            let iterations = 3usize;
            let mut subtasks = Vec::with_capacity(iterations);
            for i in 0..iterations {
                let id = Self::make_id(&goal.id, i);
                let dependencies = if i == 0 {
                    vec![]
                } else {
                    vec![Self::make_id(&goal.id, i - 1)]
                };
                subtasks.push(Subtask {
                    id,
                    parent_goal_id: goal.id.clone(),
                    description: format!("Iteration {}: {}", i + 1, goal.description),
                    estimated_tokens: Self::token_estimate(&goal.description),
                    dependencies,
                });
            }
            subtasks
        } else {
            // Fall back to sequential decomposition of the conjunction-split parts.
            self.build_sequential(goal, parts)
        }
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn make_goal(desc: &str) -> Goal {
        Goal {
            id: "g1".to_string(),
            description: desc.to_string(),
            priority: 5,
            deadline_ms: None,
        }
    }

    #[test]
    fn sequential_creates_chain() {
        let d = GoalDecomposer::new();
        let goal = make_goal("fetch data and process it and store it");
        let subtasks = d.decompose(&goal, DecompositionStrategy::Sequential);
        assert!(subtasks.len() >= 2);
        // Each subtask (except the first) depends on the previous.
        for i in 1..subtasks.len() {
            assert_eq!(subtasks[i].dependencies, vec![subtasks[i - 1].id.clone()]);
        }
    }

    #[test]
    fn parallel_no_dependencies() {
        let d = GoalDecomposer::new();
        let goal = make_goal("fetch data and process it and store it");
        let subtasks = d.decompose(&goal, DecompositionStrategy::Parallel);
        assert!(subtasks.len() >= 2);
        for s in &subtasks {
            assert!(s.dependencies.is_empty());
        }
    }

    #[test]
    fn hierarchical_first_is_parent() {
        let d = GoalDecomposer::new();
        let goal = make_goal("initialise system and configure settings and launch service");
        let subtasks = d.decompose(&goal, DecompositionStrategy::Hierarchical);
        assert!(subtasks.len() >= 2);
        assert!(subtasks[0].dependencies.is_empty());
        for s in subtasks.iter().skip(1) {
            assert_eq!(s.dependencies, vec![subtasks[0].id.clone()]);
        }
    }

    #[test]
    fn iterative_produces_three_iterations() {
        let d = GoalDecomposer::new();
        let goal = make_goal("for each item process and store");
        let subtasks = d.decompose(&goal, DecompositionStrategy::Iterative);
        assert_eq!(subtasks.len(), 3);
        assert!(subtasks[0].description.contains("Iteration 1"));
        assert!(subtasks[1].description.contains("Iteration 2"));
        assert_eq!(subtasks[1].dependencies, vec![subtasks[0].id.clone()]);
    }

    #[test]
    fn estimate_complexity_scales_with_length() {
        let d = GoalDecomposer::new();
        let short = make_goal("do thing");
        let long = make_goal(
            "analyse and generate and optimize and deploy and migrate and integrate the full system repeatedly",
        );
        assert!(d.estimate_complexity(&short) < d.estimate_complexity(&long));
    }

    #[test]
    fn estimate_complexity_bounded_1_to_5() {
        let d = GoalDecomposer::new();
        let goal = make_goal("x");
        let c = d.estimate_complexity(&goal);
        assert!((1..=5).contains(&c));
    }

    #[test]
    fn prioritize_subtasks_topological_order() {
        let d = GoalDecomposer::new();
        let mut subtasks = vec![
            Subtask {
                id: "b".into(),
                parent_goal_id: "g".into(),
                description: "B".into(),
                estimated_tokens: 10,
                dependencies: vec!["a".into()],
            },
            Subtask {
                id: "a".into(),
                parent_goal_id: "g".into(),
                description: "A".into(),
                estimated_tokens: 10,
                dependencies: vec![],
            },
            Subtask {
                id: "c".into(),
                parent_goal_id: "g".into(),
                description: "C".into(),
                estimated_tokens: 10,
                dependencies: vec!["b".into()],
            },
        ];
        d.prioritize_subtasks(&mut subtasks);
        let ids: Vec<&str> = subtasks.iter().map(|s| s.id.as_str()).collect();
        let pos_a = ids.iter().position(|&x| x == "a").unwrap();
        let pos_b = ids.iter().position(|&x| x == "b").unwrap();
        let pos_c = ids.iter().position(|&x| x == "c").unwrap();
        assert!(pos_a < pos_b);
        assert!(pos_b < pos_c);
    }

    #[test]
    fn completion_plan_waves() {
        let subtasks = vec![
            Subtask {
                id: "a".into(),
                parent_goal_id: "g".into(),
                description: "A".into(),
                estimated_tokens: 10,
                dependencies: vec![],
            },
            Subtask {
                id: "b".into(),
                parent_goal_id: "g".into(),
                description: "B".into(),
                estimated_tokens: 10,
                dependencies: vec![],
            },
            Subtask {
                id: "c".into(),
                parent_goal_id: "g".into(),
                description: "C".into(),
                estimated_tokens: 10,
                dependencies: vec!["a".into(), "b".into()],
            },
        ];
        let d = GoalDecomposer::new();
        let plan = d.completion_plan(&subtasks);
        assert_eq!(plan.len(), 2);
        assert!(plan[0].contains(&"a".to_string()));
        assert!(plan[0].contains(&"b".to_string()));
        assert_eq!(plan[1], vec!["c".to_string()]);
    }

    #[test]
    fn single_part_goal_still_works() {
        let d = GoalDecomposer::new();
        let goal = make_goal("do one thing");
        let subtasks = d.decompose(&goal, DecompositionStrategy::Sequential);
        assert_eq!(subtasks.len(), 1);
        assert!(subtasks[0].dependencies.is_empty());
    }
}

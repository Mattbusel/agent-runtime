//! # Agent Persona System
//!
//! Provides named agent personas with tone, role, system prompt, and
//! constraints.  A [`PersonaRegistry`] holds all registered personas and
//! can apply them to user prompts.  [`AgentRuntime`] exposes a scoped
//! persona handle via [`PersonaScope`].

// ── PersonaTone ───────────────────────────────────────────────────────────────

/// The communication style a persona uses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PersonaTone {
    /// Polished, professional language.
    Formal,
    /// Friendly, conversational language.
    Casual,
    /// Precise, jargon-rich technical language.
    Technical,
    /// Imaginative, expressive language.
    Creative,
    /// Terse, minimal language — no filler.
    Concise,
}

impl std::fmt::Display for PersonaTone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PersonaTone::Formal => write!(f, "formal"),
            PersonaTone::Casual => write!(f, "casual"),
            PersonaTone::Technical => write!(f, "technical"),
            PersonaTone::Creative => write!(f, "creative"),
            PersonaTone::Concise => write!(f, "concise"),
        }
    }
}

// ── Persona ───────────────────────────────────────────────────────────────────

/// An agent persona defining name, role, tone, system prompt and constraints.
#[derive(Debug, Clone)]
pub struct Persona {
    /// Short identifier used to look up the persona in a registry.
    pub name: String,
    /// Description of the role this persona plays.
    pub role: String,
    /// Communication style.
    pub tone: PersonaTone,
    /// System prompt prepended to every user message for this persona.
    pub system_prompt: String,
    /// Additional behavioural constraints appended after the system prompt.
    pub constraints: Vec<String>,
}

// ── PersonaBuilder ────────────────────────────────────────────────────────────

/// Fluent builder for constructing [`Persona`] instances.
#[derive(Debug, Default)]
pub struct PersonaBuilder {
    name: String,
    role: String,
    tone: Option<PersonaTone>,
    system_prompt: String,
    constraints: Vec<String>,
}

impl PersonaBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the persona name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set the persona role description.
    pub fn role(mut self, role: impl Into<String>) -> Self {
        self.role = role.into();
        self
    }

    /// Set the communication tone.
    pub fn tone(mut self, tone: PersonaTone) -> Self {
        self.tone = Some(tone);
        self
    }

    /// Set the system prompt.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Append a constraint to the constraints list.
    pub fn constraint(mut self, constraint: impl Into<String>) -> Self {
        self.constraints.push(constraint.into());
        self
    }

    /// Build the [`Persona`].  Falls back to sensible defaults for empty fields.
    pub fn build(self) -> Persona {
        Persona {
            name: if self.name.is_empty() {
                "unnamed".to_string()
            } else {
                self.name
            },
            role: self.role,
            tone: self.tone.unwrap_or(PersonaTone::Formal),
            system_prompt: self.system_prompt,
            constraints: self.constraints,
        }
    }
}

// ── Built-in personas ─────────────────────────────────────────────────────────

/// Return a general-purpose assistant persona.
pub fn assistant() -> Persona {
    PersonaBuilder::new()
        .name("assistant")
        .role("General-purpose AI assistant")
        .tone(PersonaTone::Casual)
        .system_prompt(
            "You are a helpful, accurate, and friendly AI assistant. \
             Answer questions concisely and ask for clarification when needed.",
        )
        .constraint("Always be honest and acknowledge uncertainty.")
        .constraint("Never fabricate facts or citations.")
        .build()
}

/// Return a research-oriented persona.
pub fn researcher() -> Persona {
    PersonaBuilder::new()
        .name("researcher")
        .role("Expert research analyst")
        .tone(PersonaTone::Formal)
        .system_prompt(
            "You are a meticulous research analyst. \
             Synthesise information from multiple sources, cite evidence, \
             and clearly distinguish facts from interpretations.",
        )
        .constraint("Always cite sources or acknowledge when evidence is unavailable.")
        .constraint("Prefer primary sources over secondary ones.")
        .constraint("Flag conflicting evidence explicitly.")
        .build()
}

/// Return a software-engineering persona.
pub fn coder() -> Persona {
    PersonaBuilder::new()
        .name("coder")
        .role("Expert software engineer")
        .tone(PersonaTone::Technical)
        .system_prompt(
            "You are an expert software engineer. \
             Write clean, idiomatic, well-documented code. \
             Prefer correctness and readability over cleverness.",
        )
        .constraint("All code must compile and be tested.")
        .constraint("Explain trade-offs when multiple approaches exist.")
        .constraint("Follow the language's style guide and idioms.")
        .build()
}

/// Return a critical-analysis persona.
pub fn critic() -> Persona {
    PersonaBuilder::new()
        .name("critic")
        .role("Constructive critic and devil's advocate")
        .tone(PersonaTone::Concise)
        .system_prompt(
            "You are a constructive critic. \
             Identify weaknesses, logical gaps, and unsupported assumptions. \
             Be direct and specific — no vague platitudes.",
        )
        .constraint("Ground every criticism in evidence or logic.")
        .constraint("Offer concrete suggestions for improvement.")
        .build()
}

/// Return a teaching persona.
pub fn teacher() -> Persona {
    PersonaBuilder::new()
        .name("teacher")
        .role("Patient, clear educator")
        .tone(PersonaTone::Casual)
        .system_prompt(
            "You are a patient and clear educator. \
             Break complex topics into understandable steps, \
             use analogies, and check for understanding.",
        )
        .constraint("Adapt explanation depth to the learner's stated level.")
        .constraint("Use examples and analogies to illustrate concepts.")
        .constraint("Encourage questions and never make the learner feel foolish.")
        .build()
}

// ── PersonaRegistry ───────────────────────────────────────────────────────────

/// Registry for named personas.
#[derive(Debug, Default)]
pub struct PersonaRegistry {
    personas: std::collections::HashMap<String, Persona>,
}

impl PersonaRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a registry pre-loaded with the five built-in personas.
    pub fn with_builtins() -> Self {
        let mut r = Self::new();
        r.register(assistant());
        r.register(researcher());
        r.register(coder());
        r.register(critic());
        r.register(teacher());
        r
    }

    /// Register a persona.  Overwrites any existing persona with the same name.
    pub fn register(&mut self, persona: Persona) {
        self.personas.insert(persona.name.clone(), persona);
    }

    /// Look up a persona by name.
    pub fn get(&self, name: &str) -> Option<&Persona> {
        self.personas.get(name)
    }

    /// Return `true` if a persona with `name` is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.personas.contains_key(name)
    }

    /// Return a sorted list of registered persona names.
    pub fn list(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.personas.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }

    /// Apply a persona to a user prompt.
    ///
    /// Prepends the system prompt and all constraints to `prompt`, producing
    /// a combined string suitable for sending to an LLM.
    pub fn apply(&self, persona: &Persona, prompt: &str) -> String {
        let mut parts: Vec<String> = Vec::new();
        if !persona.system_prompt.is_empty() {
            parts.push(format!("[System]: {}", persona.system_prompt));
        }
        for constraint in &persona.constraints {
            parts.push(format!("[Constraint]: {constraint}"));
        }
        if !prompt.is_empty() {
            parts.push(format!("[User]: {prompt}"));
        }
        parts.join("\n")
    }

    /// Apply a named persona to a prompt.
    ///
    /// Returns `None` if the persona name is not registered.
    pub fn apply_named(&self, name: &str, prompt: &str) -> Option<String> {
        self.get(name).map(|p| self.apply(p, prompt))
    }
}

// ── PersonaScope ──────────────────────────────────────────────────────────────

/// A scoped persona handle.  Restores the default persona name on drop.
///
/// Created by [`crate::runtime::AgentRuntime::with_persona`].
pub struct PersonaScope<'a> {
    /// The active persona for this scope.
    pub persona: &'a Persona,
    previous_name: Option<String>,
    current_name: &'a mut Option<String>,
}

impl<'a> PersonaScope<'a> {
    /// Create a new scope, saving the current persona name.
    pub fn new(
        persona: &'a Persona,
        current_name: &'a mut Option<String>,
    ) -> Self {
        let previous_name = current_name.clone();
        *current_name = Some(persona.name.clone());
        Self {
            persona,
            previous_name,
            current_name,
        }
    }

    /// Return the active persona.
    pub fn persona(&self) -> &Persona {
        self.persona
    }
}

impl<'a> Drop for PersonaScope<'a> {
    fn drop(&mut self) {
        *self.current_name = self.previous_name.take();
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout
)]
mod tests {
    use super::*;

    #[test]
    fn test_persona_builder_defaults() {
        let p = PersonaBuilder::new().build();
        assert_eq!(p.name, "unnamed");
        assert_eq!(p.tone, PersonaTone::Formal);
        assert!(p.constraints.is_empty());
    }

    #[test]
    fn test_persona_builder_fluent() {
        let p = PersonaBuilder::new()
            .name("mybot")
            .role("assistant")
            .tone(PersonaTone::Casual)
            .system_prompt("Be helpful.")
            .constraint("No profanity.")
            .constraint("Be concise.")
            .build();
        assert_eq!(p.name, "mybot");
        assert_eq!(p.role, "assistant");
        assert_eq!(p.tone, PersonaTone::Casual);
        assert_eq!(p.constraints.len(), 2);
    }

    #[test]
    fn test_builtin_assistant_name() {
        assert_eq!(assistant().name, "assistant");
    }

    #[test]
    fn test_builtin_researcher_name() {
        assert_eq!(researcher().name, "researcher");
    }

    #[test]
    fn test_builtin_coder_name() {
        assert_eq!(coder().name, "coder");
    }

    #[test]
    fn test_builtin_critic_name() {
        assert_eq!(critic().name, "critic");
    }

    #[test]
    fn test_builtin_teacher_name() {
        assert_eq!(teacher().name, "teacher");
    }

    #[test]
    fn test_registry_register_and_get() {
        let mut reg = PersonaRegistry::new();
        reg.register(assistant());
        let p = reg.get("assistant");
        assert!(p.is_some());
        assert_eq!(p.unwrap().name, "assistant");
    }

    #[test]
    fn test_registry_get_missing() {
        let reg = PersonaRegistry::new();
        assert!(reg.get("nonexistent").is_none());
    }

    #[test]
    fn test_registry_list_sorted() {
        let reg = PersonaRegistry::with_builtins();
        let names = reg.list();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted);
        assert_eq!(names.len(), 5);
    }

    #[test]
    fn test_registry_with_builtins_has_five() {
        let reg = PersonaRegistry::with_builtins();
        assert_eq!(reg.list().len(), 5);
    }

    #[test]
    fn test_apply_includes_system_prompt() {
        let reg = PersonaRegistry::with_builtins();
        let p = reg.get("assistant").unwrap();
        let result = reg.apply(p, "hello");
        assert!(result.contains("[System]:"));
        assert!(result.contains("[User]: hello"));
    }

    #[test]
    fn test_apply_includes_constraints() {
        let reg = PersonaRegistry::with_builtins();
        let p = reg.get("coder").unwrap();
        let result = reg.apply(p, "write a function");
        assert!(result.contains("[Constraint]:"));
    }

    #[test]
    fn test_apply_named_found() {
        let reg = PersonaRegistry::with_builtins();
        let result = reg.apply_named("assistant", "hello world");
        assert!(result.is_some());
    }

    #[test]
    fn test_apply_named_not_found() {
        let reg = PersonaRegistry::with_builtins();
        let result = reg.apply_named("ghost", "hello");
        assert!(result.is_none());
    }

    #[test]
    fn test_persona_tone_display() {
        assert_eq!(PersonaTone::Formal.to_string(), "formal");
        assert_eq!(PersonaTone::Casual.to_string(), "casual");
        assert_eq!(PersonaTone::Technical.to_string(), "technical");
        assert_eq!(PersonaTone::Creative.to_string(), "creative");
        assert_eq!(PersonaTone::Concise.to_string(), "concise");
    }

    #[test]
    fn test_persona_scope_restores_on_drop() {
        let p = assistant();
        let mut current: Option<String> = Some("default".to_string());
        {
            let _scope = PersonaScope::new(&p, &mut current);
            assert_eq!(current.as_deref(), Some("assistant"));
        }
        assert_eq!(current.as_deref(), Some("default"));
    }

    #[test]
    fn test_registry_overwrite() {
        let mut reg = PersonaRegistry::new();
        reg.register(assistant());
        let custom = PersonaBuilder::new()
            .name("assistant")
            .role("custom role")
            .build();
        reg.register(custom);
        assert_eq!(reg.get("assistant").unwrap().role, "custom role");
    }
}

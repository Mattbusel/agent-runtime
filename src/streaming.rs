//! # Module: Streaming
//!
//! ## Responsibility
//! Emits agent reasoning events (thought, action, observation, result, error)
//! as Server-Sent Events (SSE) over a `tokio::sync::broadcast` channel so that
//! multiple HTTP clients can subscribe to a single agent run in real time.
//!
//! ## Design
//! - [`AgentEvent`] — the typed event enum; each variant serialises to JSON and
//!   is wrapped in the SSE `data: …\n\n` envelope.
//! - [`StreamBroadcaster`] — thin wrapper around a `broadcast::Sender<String>`.
//!   Call [`StreamBroadcaster::send`] to publish an event; call
//!   [`StreamBroadcaster::subscribe`] to obtain a `broadcast::Receiver<String>`
//!   from which a subscriber reads raw SSE lines.
//! - [`AgentEventStream`] — convenience helper that owns a `StreamBroadcaster`
//!   and exposes typed `emit_*` methods.
//!
//! ## Guarantees
//! - Non-panicking: all operations return `Result`
//! - No `unwrap` / `expect` / `panic` in production paths
//! - Thread-safe: `StreamBroadcaster` and `AgentEventStream` are `Clone`,
//!   `Send`, and `Sync`
//!
//! ## NOT Responsible For
//! - HTTP transport (callers wire the `Receiver` output into their own HTTP
//!   layer — axum, actix-web, hyper, etc.)
//! - Persistence of streamed events

use crate::error::AgentRuntimeError;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

// ── AgentEvent ────────────────────────────────────────────────────────────────

/// A single event emitted by a running agent.
///
/// Each variant maps to one stage of the ReAct loop or to an error condition.
/// Serialises to a JSON object with a `"type"` discriminant field, e.g.:
///
/// ```json
/// {"type":"thought","content":"I should search the web."}
/// {"type":"action","tool":"search","input":"{\"q\":\"Rust\"}"}
/// {"type":"observation","content":"Rust is a systems language."}
/// {"type":"result","content":"The answer is 42."}
/// {"type":"error","content":"Tool not found: foobar"}
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum AgentEvent {
    /// A reasoning thought produced before taking an action.
    Thought(
        /// The thought text.
        #[serde(rename = "content")]
        String,
    ),
    /// An action dispatched to a tool.
    Action {
        /// The name of the tool being called.
        tool: String,
        /// The serialised input arguments for the tool.
        input: String,
    },
    /// An observation returned by a tool after an action.
    Observation(
        /// The observation text.
        #[serde(rename = "content")]
        String,
    ),
    /// The final result produced by the agent.
    Result(
        /// The result text.
        #[serde(rename = "content")]
        String,
    ),
    /// An error that occurred during agent execution.
    Error(
        /// A human-readable error description.
        #[serde(rename = "content")]
        String,
    ),
}

impl AgentEvent {
    /// Serialise the event to an SSE-formatted string: `data: {json}\n\n`.
    ///
    /// Returns an error only if `serde_json` fails to serialise the value,
    /// which cannot happen for this well-formed enum in practice.
    pub fn to_sse(&self) -> Result<String, AgentRuntimeError> {
        let json =
            serde_json::to_string(self).map_err(|e| AgentRuntimeError::AgentLoop(e.to_string()))?;
        Ok(format!("data: {json}\n\n"))
    }

    /// Return a short machine-readable event type label.
    pub fn event_type(&self) -> &'static str {
        match self {
            AgentEvent::Thought(_) => "thought",
            AgentEvent::Action { .. } => "action",
            AgentEvent::Observation(_) => "observation",
            AgentEvent::Result(_) => "result",
            AgentEvent::Error(_) => "error",
        }
    }

    /// Return `true` if this is a terminal event (result or error).
    pub fn is_terminal(&self) -> bool {
        matches!(self, AgentEvent::Result(_) | AgentEvent::Error(_))
    }
}

// ── StreamBroadcaster ─────────────────────────────────────────────────────────

/// A broadcast publisher for raw SSE strings.
///
/// `StreamBroadcaster` is cheap to clone — all clones share the same underlying
/// `broadcast::Sender`.  Each subscriber gets its own `broadcast::Receiver`
/// from [`subscribe`][StreamBroadcaster::subscribe].
///
/// The `capacity` parameter controls the internal ring-buffer size.  Slow
/// subscribers that fall more than `capacity` messages behind will receive a
/// [`broadcast::error::RecvError::Lagged`] error on the next `recv()` call;
/// the broadcaster itself is never blocked.
#[derive(Debug, Clone)]
pub struct StreamBroadcaster {
    sender: broadcast::Sender<String>,
}

impl StreamBroadcaster {
    /// Create a new broadcaster with the given channel capacity.
    ///
    /// # Errors
    /// Returns `Err` if `capacity` is zero (the broadcast channel requires a
    /// capacity of at least 1).
    pub fn new(capacity: usize) -> Result<Self, AgentRuntimeError> {
        if capacity == 0 {
            return Err(AgentRuntimeError::AgentLoop(
                "broadcast channel capacity must be greater than zero".into(),
            ));
        }
        let (sender, _) = broadcast::channel(capacity);
        Ok(Self { sender })
    }

    /// Publish a raw SSE-formatted string to all active subscribers.
    ///
    /// Returns the number of active receivers that received the message.
    /// A return value of `0` simply means there are no subscribers at this
    /// moment; it is not an error.
    pub fn send(&self, sse_line: String) -> Result<usize, AgentRuntimeError> {
        match self.sender.send(sse_line) {
            Ok(n) => Ok(n),
            // `SendError` only occurs when there are no receivers, which is not
            // an error condition for a broadcaster.
            Err(_) => Ok(0),
        }
    }

    /// Publish an [`AgentEvent`] by converting it to SSE format first.
    pub fn send_event(&self, event: &AgentEvent) -> Result<usize, AgentRuntimeError> {
        let sse = event.to_sse()?;
        self.send(sse)
    }

    /// Create a new subscriber receiver.
    ///
    /// The receiver will see all messages published *after* this call.
    pub fn subscribe(&self) -> broadcast::Receiver<String> {
        self.sender.subscribe()
    }

    /// Return the number of active subscribers.
    pub fn receiver_count(&self) -> usize {
        self.sender.receiver_count()
    }
}

// ── AgentEventStream ──────────────────────────────────────────────────────────

/// High-level typed event stream for a single agent run.
///
/// `AgentEventStream` owns a [`StreamBroadcaster`] and exposes named `emit_*`
/// helpers so that agent loop code does not need to construct [`AgentEvent`]
/// variants directly.
///
/// ```rust,no_run
/// use llm_agent_runtime::streaming::{AgentEventStream, StreamBroadcaster};
///
/// # fn main() -> Result<(), llm_agent_runtime::AgentRuntimeError> {
/// let stream = AgentEventStream::new(64)?;
/// let mut rx = stream.broadcaster().subscribe();
///
/// stream.emit_thought("I should search the web.")?;
/// stream.emit_action("search", r#"{"q":"Rust"}"#)?;
/// stream.emit_observation("Rust is a systems language.")?;
/// stream.emit_result("The answer is: Rust.")?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct AgentEventStream {
    broadcaster: StreamBroadcaster,
}

impl AgentEventStream {
    /// Create a new stream with a broadcast channel of the given `capacity`.
    ///
    /// # Errors
    /// Returns `Err` if `capacity` is zero.
    pub fn new(capacity: usize) -> Result<Self, AgentRuntimeError> {
        Ok(Self {
            broadcaster: StreamBroadcaster::new(capacity)?,
        })
    }

    /// Borrow the underlying [`StreamBroadcaster`] (e.g. to create additional
    /// subscribers).
    pub fn broadcaster(&self) -> &StreamBroadcaster {
        &self.broadcaster
    }

    /// Emit a `Thought` event.
    pub fn emit_thought(&self, thought: impl Into<String>) -> Result<usize, AgentRuntimeError> {
        self.broadcaster
            .send_event(&AgentEvent::Thought(thought.into()))
    }

    /// Emit an `Action` event.
    pub fn emit_action(
        &self,
        tool: impl Into<String>,
        input: impl Into<String>,
    ) -> Result<usize, AgentRuntimeError> {
        self.broadcaster.send_event(&AgentEvent::Action {
            tool: tool.into(),
            input: input.into(),
        })
    }

    /// Emit an `Observation` event.
    pub fn emit_observation(
        &self,
        observation: impl Into<String>,
    ) -> Result<usize, AgentRuntimeError> {
        self.broadcaster
            .send_event(&AgentEvent::Observation(observation.into()))
    }

    /// Emit a `Result` event.
    pub fn emit_result(&self, result: impl Into<String>) -> Result<usize, AgentRuntimeError> {
        self.broadcaster
            .send_event(&AgentEvent::Result(result.into()))
    }

    /// Emit an `Error` event.
    pub fn emit_error(&self, error: impl Into<String>) -> Result<usize, AgentRuntimeError> {
        self.broadcaster
            .send_event(&AgentEvent::Error(error.into()))
    }

    /// Return the number of active subscribers.
    pub fn receiver_count(&self) -> usize {
        self.broadcaster.receiver_count()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── AgentEvent serialisation ──────────────────────────────────────────────

    #[test]
    fn test_thought_event_serializes_with_type_tag() {
        let e = AgentEvent::Thought("think".into());
        let json = serde_json::to_string(&e).unwrap();
        assert!(json.contains(r#""type":"thought""#));
        assert!(json.contains(r#""content":"think""#));
    }

    #[test]
    fn test_action_event_serializes_tool_and_input() {
        let e = AgentEvent::Action {
            tool: "search".into(),
            input: r#"{"q":"rust"}"#.into(),
        };
        let json = serde_json::to_string(&e).unwrap();
        assert!(json.contains(r#""type":"action""#));
        assert!(json.contains(r#""tool":"search""#));
    }

    #[test]
    fn test_observation_event_serializes() {
        let e = AgentEvent::Observation("some observation".into());
        let json = serde_json::to_string(&e).unwrap();
        assert!(json.contains(r#""type":"observation""#));
    }

    #[test]
    fn test_result_event_serializes() {
        let e = AgentEvent::Result("42".into());
        let json = serde_json::to_string(&e).unwrap();
        assert!(json.contains(r#""type":"result""#));
        assert!(json.contains(r#""content":"42""#));
    }

    #[test]
    fn test_error_event_serializes() {
        let e = AgentEvent::Error("oops".into());
        let json = serde_json::to_string(&e).unwrap();
        assert!(json.contains(r#""type":"error""#));
    }

    #[test]
    fn test_event_deserializes_roundtrip() {
        let orig = AgentEvent::Action {
            tool: "double".into(),
            input: "21".into(),
        };
        let json = serde_json::to_string(&orig).unwrap();
        let back: AgentEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back, orig);
    }

    // ── to_sse ────────────────────────────────────────────────────────────────

    #[test]
    fn test_to_sse_format() {
        let e = AgentEvent::Thought("hi".into());
        let sse = e.to_sse().unwrap();
        assert!(sse.starts_with("data: "));
        assert!(sse.ends_with("\n\n"));
    }

    // ── event_type ────────────────────────────────────────────────────────────

    #[test]
    fn test_event_type_labels() {
        assert_eq!(AgentEvent::Thought("x".into()).event_type(), "thought");
        assert_eq!(
            AgentEvent::Action { tool: "t".into(), input: "i".into() }.event_type(),
            "action"
        );
        assert_eq!(AgentEvent::Observation("o".into()).event_type(), "observation");
        assert_eq!(AgentEvent::Result("r".into()).event_type(), "result");
        assert_eq!(AgentEvent::Error("e".into()).event_type(), "error");
    }

    // ── is_terminal ───────────────────────────────────────────────────────────

    #[test]
    fn test_result_is_terminal() {
        assert!(AgentEvent::Result("done".into()).is_terminal());
    }

    #[test]
    fn test_error_is_terminal() {
        assert!(AgentEvent::Error("bad".into()).is_terminal());
    }

    #[test]
    fn test_thought_is_not_terminal() {
        assert!(!AgentEvent::Thought("thinking".into()).is_terminal());
    }

    #[test]
    fn test_action_is_not_terminal() {
        assert!(!AgentEvent::Action { tool: "t".into(), input: "i".into() }.is_terminal());
    }

    #[test]
    fn test_observation_is_not_terminal() {
        assert!(!AgentEvent::Observation("obs".into()).is_terminal());
    }

    // ── StreamBroadcaster ─────────────────────────────────────────────────────

    #[test]
    fn test_broadcaster_zero_capacity_returns_err() {
        assert!(StreamBroadcaster::new(0).is_err());
    }

    #[test]
    fn test_broadcaster_new_succeeds_with_nonzero_capacity() {
        assert!(StreamBroadcaster::new(16).is_ok());
    }

    #[test]
    fn test_broadcaster_receiver_count_after_subscribe() {
        let b = StreamBroadcaster::new(8).unwrap();
        assert_eq!(b.receiver_count(), 0);
        let _rx = b.subscribe();
        assert_eq!(b.receiver_count(), 1);
    }

    #[test]
    fn test_broadcaster_send_with_no_receivers_returns_ok() {
        let b = StreamBroadcaster::new(8).unwrap();
        let result = b.send("data: test\n\n".to_string());
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_broadcaster_subscriber_receives_message() {
        let b = StreamBroadcaster::new(8).unwrap();
        let mut rx = b.subscribe();
        b.send("data: hello\n\n".to_string()).unwrap();
        let msg = rx.recv().await.unwrap();
        assert_eq!(msg, "data: hello\n\n");
    }

    #[tokio::test]
    async fn test_broadcaster_send_event_delivers_sse() {
        let b = StreamBroadcaster::new(8).unwrap();
        let mut rx = b.subscribe();
        let event = AgentEvent::Thought("test thought".into());
        b.send_event(&event).unwrap();
        let msg = rx.recv().await.unwrap();
        assert!(msg.starts_with("data: "));
        assert!(msg.contains("thought"));
    }

    #[tokio::test]
    async fn test_broadcaster_multiple_subscribers_all_receive() {
        let b = StreamBroadcaster::new(8).unwrap();
        let mut rx1 = b.subscribe();
        let mut rx2 = b.subscribe();
        b.send("data: ping\n\n".to_string()).unwrap();
        assert_eq!(rx1.recv().await.unwrap(), "data: ping\n\n");
        assert_eq!(rx2.recv().await.unwrap(), "data: ping\n\n");
    }

    // ── AgentEventStream ──────────────────────────────────────────────────────

    #[test]
    fn test_stream_zero_capacity_fails() {
        assert!(AgentEventStream::new(0).is_err());
    }

    #[test]
    fn test_stream_nonzero_capacity_succeeds() {
        assert!(AgentEventStream::new(32).is_ok());
    }

    #[tokio::test]
    async fn test_stream_emit_thought_received_by_subscriber() {
        let stream = AgentEventStream::new(16).unwrap();
        let mut rx = stream.broadcaster().subscribe();
        stream.emit_thought("some thought").unwrap();
        let msg = rx.recv().await.unwrap();
        assert!(msg.contains("thought"));
        assert!(msg.contains("some thought"));
    }

    #[tokio::test]
    async fn test_stream_emit_action_received() {
        let stream = AgentEventStream::new(16).unwrap();
        let mut rx = stream.broadcaster().subscribe();
        stream.emit_action("mytool", r#"{"x":1}"#).unwrap();
        let msg = rx.recv().await.unwrap();
        assert!(msg.contains("action"));
        assert!(msg.contains("mytool"));
    }

    #[tokio::test]
    async fn test_stream_emit_observation_received() {
        let stream = AgentEventStream::new(16).unwrap();
        let mut rx = stream.broadcaster().subscribe();
        stream.emit_observation("found result").unwrap();
        let msg = rx.recv().await.unwrap();
        assert!(msg.contains("observation"));
    }

    #[tokio::test]
    async fn test_stream_emit_result_received() {
        let stream = AgentEventStream::new(16).unwrap();
        let mut rx = stream.broadcaster().subscribe();
        stream.emit_result("final answer").unwrap();
        let msg = rx.recv().await.unwrap();
        assert!(msg.contains("result"));
        assert!(msg.contains("final answer"));
    }

    #[tokio::test]
    async fn test_stream_emit_error_received() {
        let stream = AgentEventStream::new(16).unwrap();
        let mut rx = stream.broadcaster().subscribe();
        stream.emit_error("something went wrong").unwrap();
        let msg = rx.recv().await.unwrap();
        assert!(msg.contains("error"));
        assert!(msg.contains("something went wrong"));
    }

    #[test]
    fn test_stream_receiver_count() {
        let stream = AgentEventStream::new(8).unwrap();
        assert_eq!(stream.receiver_count(), 0);
        let _rx = stream.broadcaster().subscribe();
        assert_eq!(stream.receiver_count(), 1);
    }

    #[test]
    fn test_stream_clone_shares_broadcaster() {
        let stream = AgentEventStream::new(8).unwrap();
        let clone = stream.clone();
        let _rx = stream.broadcaster().subscribe();
        assert_eq!(clone.receiver_count(), 1);
    }

    #[test]
    fn test_event_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<AgentEvent>();
        assert_send_sync::<StreamBroadcaster>();
        assert_send_sync::<AgentEventStream>();
    }
}

// ── Token-by-token streaming inference ────────────────────────────────────────
//
// The types below extend the SSE broadcaster with a lower-level token stream
// API that drives the [`StreamingReActLoop`].  They are deliberately in this
// module so all streaming concerns live in one place.

use crate::agent::Action;
use futures::Stream;
use std::pin::Pin;

// ── InferenceToken ─────────────────────────────────────────────────────────────

/// A single token emitted by a streaming inference provider.
///
/// `text` is the raw token string.  `is_final` is `true` for the last token
/// in a response.  `logprob` carries an optional log-probability useful for
/// calibration and perplexity estimation.
///
/// # Example
///
/// ```rust
/// use llm_agent_runtime::streaming::InferenceToken;
///
/// let tok = InferenceToken::new("Hello", None, false);
/// assert_eq!(tok.text, "Hello");
/// assert!(!tok.is_final);
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InferenceToken {
    /// The raw text of this token.
    pub text: String,
    /// Optional log-probability of this token under the model's distribution.
    pub logprob: Option<f64>,
    /// `true` if this is the last token in the current inference response.
    pub is_final: bool,
}

impl InferenceToken {
    /// Create a new `InferenceToken`.
    pub fn new(text: impl Into<String>, logprob: Option<f64>, is_final: bool) -> Self {
        Self { text: text.into(), logprob, is_final }
    }

    /// Create a non-final token with no log-probability.
    pub fn text(text: impl Into<String>) -> Self {
        Self::new(text, None, false)
    }

    /// Create a final token with no log-probability.
    pub fn final_token(text: impl Into<String>) -> Self {
        Self::new(text, None, true)
    }

    /// Create a thought-prefix token (`"Thought: <text>"`), non-final.
    pub fn thought(text: impl Into<String>) -> Self {
        Self::text(format!("Thought: {}", text.into()))
    }

    /// Create a final-answer token (`"\nAction: FINAL_ANSWER <text>"`), final.
    pub fn final_answer(text: impl Into<String>) -> Self {
        Self::final_token(format!("\nAction: FINAL_ANSWER {}", text.into()))
    }

    /// Return `true` if this token contains only whitespace.
    pub fn is_whitespace(&self) -> bool {
        self.text.trim().is_empty()
    }

    /// Return the log-probability, or `0.0` if not present.
    pub fn logprob_or_zero(&self) -> f64 {
        self.logprob.unwrap_or(0.0)
    }
}

impl std::fmt::Display for InferenceToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.text)
    }
}

// ── TokenStream ───────────────────────────────────────────────────────────────

/// An async stream that yields [`InferenceToken`]s one at a time.
///
/// Returned by [`StreamingInference::infer_stream`].
pub type TokenStream<'a> = Pin<Box<dyn Stream<Item = InferenceToken> + Send + 'a>>;

// ── StreamingInference ────────────────────────────────────────────────────────

/// A provider capable of streaming inference tokens one by one.
///
/// Implement this trait to connect any token-streaming LLM API to the
/// [`StreamingReActLoop`].
///
/// # Example
///
/// ```rust
/// use llm_agent_runtime::streaming::{InferenceToken, StreamingInference, TokenStream};
/// use futures::stream;
///
/// struct ConstantStreamer(&'static str);
///
/// impl StreamingInference for ConstantStreamer {
///     fn infer_stream<'a>(&'a self, _prompt: &'a str) -> TokenStream<'a> {
///         let tokens = vec![InferenceToken::final_token(self.0)];
///         Box::pin(stream::iter(tokens))
///     }
/// }
/// ```
pub trait StreamingInference {
    /// Begin a streaming inference request for `prompt`.
    ///
    /// The returned [`TokenStream`] must eventually yield a token with
    /// `is_final = true`.
    fn infer_stream<'a>(&'a self, prompt: &'a str) -> TokenStream<'a>;
}

// ── StreamingCallbacks ────────────────────────────────────────────────────────

/// Per-event callbacks invoked during a [`StreamingReActLoop`] run.
///
/// All callbacks are optional.  They are called synchronously inside the
/// streaming loop, so they must not block the Tokio runtime.
///
/// # Example
///
/// ```rust
/// use llm_agent_runtime::streaming::StreamingCallbacks;
///
/// let callbacks = StreamingCallbacks::new()
///     .on_token(|tok| { let _ = tok.text.len(); })
///     .on_thought(|t| { let _ = t.len(); })
///     .on_action(|a| { let _ = a; });
/// ```
pub struct StreamingCallbacks {
    /// Called for every token received from the stream.
    pub on_token: Option<Box<dyn Fn(&InferenceToken) + Send + Sync>>,
    /// Called when a complete thought line has been assembled.
    pub on_thought: Option<Box<dyn Fn(&str) + Send + Sync>>,
    /// Called when a complete action has been parsed.
    pub on_action: Option<Box<dyn Fn(&Action) + Send + Sync>>,
}

impl StreamingCallbacks {
    /// Create a new `StreamingCallbacks` with all callbacks unset.
    pub fn new() -> Self {
        Self { on_token: None, on_thought: None, on_action: None }
    }

    /// Attach a per-token callback.
    pub fn on_token(mut self, f: impl Fn(&InferenceToken) + Send + Sync + 'static) -> Self {
        self.on_token = Some(Box::new(f));
        self
    }

    /// Attach a per-thought callback.
    pub fn on_thought(mut self, f: impl Fn(&str) + Send + Sync + 'static) -> Self {
        self.on_thought = Some(Box::new(f));
        self
    }

    /// Attach a per-action callback.
    pub fn on_action(mut self, f: impl Fn(&Action) + Send + Sync + 'static) -> Self {
        self.on_action = Some(Box::new(f));
        self
    }

    pub(crate) fn fire_token(&self, token: &InferenceToken) {
        if let Some(ref cb) = self.on_token { cb(token); }
    }
    pub(crate) fn fire_thought(&self, thought: &str) {
        if let Some(ref cb) = self.on_thought { cb(thought); }
    }
    pub(crate) fn fire_action(&self, action: &Action) {
        if let Some(ref cb) = self.on_action { cb(action); }
    }
}

impl Default for StreamingCallbacks {
    fn default() -> Self { Self::new() }
}

impl std::fmt::Debug for StreamingCallbacks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingCallbacks")
            .field("on_token", &self.on_token.is_some())
            .field("on_thought", &self.on_thought.is_some())
            .field("on_action", &self.on_action.is_some())
            .finish()
    }
}

// ── StreamingStep ─────────────────────────────────────────────────────────────

/// A single step recorded by the [`StreamingReActLoop`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StreamingStep {
    /// The assembled thought string for this step.
    pub thought: String,
    /// The assembled action string for this step.
    pub action: String,
    /// The observation produced by dispatching the action.
    pub observation: String,
    /// Total tokens received during this step.
    pub token_count: usize,
    /// Sum of log-probabilities for tokens in this step.
    pub logprob_sum: f64,
}

impl StreamingStep {
    /// Return `true` if the action is a `FINAL_ANSWER`.
    pub fn is_final_answer(&self) -> bool {
        self.action.trim().to_ascii_uppercase().starts_with("FINAL_ANSWER")
    }

    /// Return the mean log-probability per token. Returns `0.0` with no tokens.
    pub fn mean_logprob(&self) -> f64 {
        if self.token_count == 0 { return 0.0; }
        self.logprob_sum / self.token_count as f64
    }
}

// ── StreamingSession ──────────────────────────────────────────────────────────

/// The result of a completed [`StreamingReActLoop`] run.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StreamingSession {
    /// All steps executed during the session.
    pub steps: Vec<StreamingStep>,
    /// Total tokens received across all steps.
    pub total_tokens: usize,
    /// Wall-clock duration in milliseconds.
    pub duration_ms: u64,
    /// Whether the loop ended with a `FINAL_ANSWER`.
    pub completed: bool,
}

impl StreamingSession {
    /// Return the number of steps in this session.
    pub fn step_count(&self) -> usize { self.steps.len() }

    /// Return `true` if the session ended with a `FINAL_ANSWER`.
    pub fn is_completed(&self) -> bool { self.completed }

    /// Return the final answer text, if the session completed successfully.
    pub fn final_answer(&self) -> Option<String> {
        let last = self.steps.last()?;
        if last.is_final_answer() {
            let raw = last.action.trim();
            let stripped = raw.get("FINAL_ANSWER".len()..).unwrap_or("").trim();
            Some(stripped.to_owned())
        } else {
            None
        }
    }

    /// Return the mean tokens per step. Returns `0.0` for empty sessions.
    pub fn avg_tokens_per_step(&self) -> f64 {
        if self.steps.is_empty() { return 0.0; }
        self.total_tokens as f64 / self.steps.len() as f64
    }

    /// Return all thought strings in step order.
    pub fn all_thoughts(&self) -> Vec<&str> {
        self.steps.iter().map(|s| s.thought.as_str()).collect()
    }

    /// Return all action strings in step order.
    pub fn all_actions(&self) -> Vec<&str> {
        self.steps.iter().map(|s| s.action.as_str()).collect()
    }

    /// Return the mean log-probability across all tokens in the session.
    pub fn overall_mean_logprob(&self) -> f64 {
        if self.total_tokens == 0 { return 0.0; }
        let total_lp: f64 = self.steps.iter().map(|s| s.logprob_sum).sum();
        total_lp / self.total_tokens as f64
    }
}

// ── StreamingReActLoop ─────────────────────────────────────────────────────────

/// A ReAct loop that processes inference tokens as they stream in.
///
/// Fires [`StreamingCallbacks`] for each token, assembled thought, and parsed
/// action.  The loop terminates on `FINAL_ANSWER` or `max_steps`.
///
/// # Example
///
/// ```rust,no_run
/// use llm_agent_runtime::streaming::{
///     InferenceToken, StreamingInference, StreamingReActLoop, TokenStream,
/// };
/// use futures::stream;
///
/// struct Stub;
/// impl StreamingInference for Stub {
///     fn infer_stream<'a>(&'a self, _p: &'a str) -> TokenStream<'a> {
///         Box::pin(stream::iter(vec![
///             InferenceToken::thought("I should answer"),
///             InferenceToken::final_answer("42"),
///         ]))
///     }
/// }
///
/// # tokio_test::block_on(async {
/// let mut loop_ = StreamingReActLoop::new(Stub, 5);
/// let session = loop_.run("What is 6*7?").await.unwrap();
/// assert_eq!(session.final_answer().as_deref(), Some("42"));
/// # });
/// ```
pub struct StreamingReActLoop<P: StreamingInference> {
    provider: P,
    max_steps: usize,
    callbacks: StreamingCallbacks,
    tool_handler: Option<Box<dyn Fn(&str, &str) -> String + Send + Sync>>,
}

impl<P: StreamingInference> StreamingReActLoop<P> {
    /// Create a new `StreamingReActLoop`.
    ///
    /// * `provider` — implements [`StreamingInference`].
    /// * `max_steps` — maximum Thought-Action-Observation cycles.
    pub fn new(provider: P, max_steps: usize) -> Self {
        Self {
            provider,
            max_steps: max_steps.max(1),
            callbacks: StreamingCallbacks::new(),
            tool_handler: None,
        }
    }

    /// Attach streaming callbacks.
    pub fn with_callbacks(mut self, callbacks: StreamingCallbacks) -> Self {
        self.callbacks = callbacks;
        self
    }

    /// Attach a tool handler `(tool_name, json_args) -> observation`.
    ///
    /// Without a handler, tool calls return `"(no tool handler registered)"`.
    pub fn with_tool_handler(
        mut self,
        handler: impl Fn(&str, &str) -> String + Send + Sync + 'static,
    ) -> Self {
        self.tool_handler = Some(Box::new(handler));
        self
    }

    /// Run the streaming ReAct loop for `prompt`.
    ///
    /// Returns a [`StreamingSession`] regardless of whether `FINAL_ANSWER`
    /// was reached.  Check [`StreamingSession::completed`] to distinguish.
    pub async fn run(
        &mut self,
        prompt: &str,
    ) -> Result<StreamingSession, crate::error::AgentRuntimeError> {
        use futures::StreamExt;

        let start = std::time::Instant::now();
        let mut steps: Vec<StreamingStep> = Vec::new();
        let mut total_tokens = 0usize;
        let mut context = prompt.to_owned();

        for _step_idx in 0..self.max_steps {
            let mut response_buf = String::new();
            let mut step_tokens = 0usize;
            let mut step_logprob_sum = 0.0f64;

            let mut stream = self.provider.infer_stream(&context);
            while let Some(token) = stream.next().await {
                step_tokens += 1;
                total_tokens += 1;
                step_logprob_sum += token.logprob.unwrap_or(0.0);
                response_buf.push_str(&token.text);
                self.callbacks.fire_token(&token);
                let done = token.is_final;
                if done { break; }
            }

            let (thought, action) = parse_streaming_response(&response_buf);

            if !thought.is_empty() {
                self.callbacks.fire_thought(&thought);
            }

            let parsed_action = crate::agent::parse_react_step(&response_buf)
                .map(|s| s.action.clone())
                .unwrap_or_else(|| Action::FinalAnswer(action.clone()));

            self.callbacks.fire_action(&parsed_action);

            let observation = match &parsed_action {
                Action::FinalAnswer(_) => String::new(),
                Action::ToolCall { tool, args } => {
                    if let Some(ref handler) = self.tool_handler {
                        let args_str = serde_json::to_string(args).unwrap_or_default();
                        handler(tool.as_str(), &args_str)
                    } else {
                        "(no tool handler registered)".to_string()
                    }
                }
            };

            let is_final = matches!(parsed_action, Action::FinalAnswer(_));

            steps.push(StreamingStep {
                thought: thought.clone(),
                action: action.clone(),
                observation: observation.clone(),
                token_count: step_tokens,
                logprob_sum: step_logprob_sum,
            });

            if is_final {
                return Ok(StreamingSession {
                    steps,
                    total_tokens,
                    duration_ms: start.elapsed().as_millis() as u64,
                    completed: true,
                });
            }

            context = format!(
                "{context}\nThought: {thought}\nAction: {action}\nObservation: {observation}"
            );
        }

        Ok(StreamingSession {
            steps,
            total_tokens,
            duration_ms: start.elapsed().as_millis() as u64,
            completed: false,
        })
    }

    /// Return a reference to the underlying provider.
    pub fn provider(&self) -> &P { &self.provider }

    /// Return the maximum step count.
    pub fn max_steps(&self) -> usize { self.max_steps }
}

/// Extract `(thought, action)` from a raw LLM response string.
fn parse_streaming_response(response: &str) -> (String, String) {
    let mut thought = String::new();
    let mut action = String::new();
    for line in response.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("Thought:") {
            thought = rest.trim().to_owned();
        } else if let Some(rest) = trimmed.strip_prefix("Action:") {
            action = rest.trim().to_owned();
        }
    }
    (thought, action)
}

#[cfg(test)]
mod streaming_inference_tests {
    use super::*;
    use futures::stream;
    use std::sync::{Arc, Mutex};

    struct MockStreamer {
        responses: Vec<Vec<InferenceToken>>,
        call_count: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl MockStreamer {
        fn new(responses: Vec<Vec<InferenceToken>>) -> Self {
            Self { responses, call_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)) }
        }
    }

    impl StreamingInference for MockStreamer {
        fn infer_stream<'a>(&'a self, _prompt: &'a str) -> TokenStream<'a> {
            let idx = self.call_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let tokens = self.responses.get(idx).cloned().unwrap_or_else(|| {
                vec![InferenceToken::final_token("Thought: done\nAction: FINAL_ANSWER done")]
            });
            Box::pin(stream::iter(tokens))
        }
    }

    #[test]
    fn test_inference_token_constructors() {
        let tok = InferenceToken::new("hi", Some(-0.5), false);
        assert_eq!(tok.text, "hi");
        assert_eq!(tok.logprob, Some(-0.5));
        assert!(!tok.is_final);

        let final_tok = InferenceToken::final_token("bye");
        assert!(final_tok.is_final);

        let thought = InferenceToken::thought("plan");
        assert!(thought.text.starts_with("Thought:"));

        let fa = InferenceToken::final_answer("42");
        assert!(fa.text.contains("FINAL_ANSWER"));
        assert!(fa.is_final);
    }

    #[test]
    fn test_inference_token_is_whitespace() {
        assert!(InferenceToken::text("  ").is_whitespace());
        assert!(!InferenceToken::text("a").is_whitespace());
    }

    #[test]
    fn test_inference_token_logprob_or_zero() {
        assert_eq!(InferenceToken::text("x").logprob_or_zero(), 0.0);
        assert_eq!(InferenceToken::new("y", Some(-2.0), false).logprob_or_zero(), -2.0);
    }

    #[test]
    fn test_streaming_callbacks_fire_token() {
        let log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let log2 = Arc::clone(&log);
        let cb = StreamingCallbacks::new().on_token(move |tok| {
            log2.lock().unwrap().push(tok.text.clone());
        });
        cb.fire_token(&InferenceToken::text("a"));
        cb.fire_token(&InferenceToken::text("b"));
        assert_eq!(*log.lock().unwrap(), vec!["a", "b"]);
    }

    #[test]
    fn test_streaming_step_mean_logprob() {
        let step = StreamingStep {
            thought: String::new(), action: String::new(), observation: String::new(),
            token_count: 4, logprob_sum: -8.0,
        };
        assert_eq!(step.mean_logprob(), -2.0);
    }

    #[test]
    fn test_streaming_step_is_final_answer() {
        let step = StreamingStep {
            thought: String::new(), action: "FINAL_ANSWER 42".into(),
            observation: String::new(), token_count: 1, logprob_sum: 0.0,
        };
        assert!(step.is_final_answer());
    }

    #[tokio::test]
    async fn test_streaming_loop_completes_with_final_answer() {
        let streamer = MockStreamer::new(vec![vec![
            InferenceToken::text("Thought: done\nAction: FINAL_ANSWER 42"),
            InferenceToken::final_token(""),
        ]]);
        let mut loop_ = StreamingReActLoop::new(streamer, 5);
        let session = loop_.run("What is 6*7?").await.unwrap();
        assert!(session.is_completed());
        assert_eq!(session.final_answer().as_deref(), Some("42"));
    }

    #[tokio::test]
    async fn test_streaming_loop_respects_max_steps() {
        let streamer = MockStreamer::new(
            (0..10).map(|_| vec![
                InferenceToken::text("Thought: t\nAction: search foo"),
                InferenceToken::final_token(""),
            ]).collect(),
        );
        let mut loop_ = StreamingReActLoop::new(streamer, 2)
            .with_tool_handler(|_, _| "obs".to_string());
        let session = loop_.run("p").await.unwrap();
        assert!(!session.is_completed());
        assert_eq!(session.step_count(), 2);
    }

    #[tokio::test]
    async fn test_streaming_loop_fires_callbacks() {
        let token_log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let tl = Arc::clone(&token_log);
        let thought_log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let thl = Arc::clone(&thought_log);

        let callbacks = StreamingCallbacks::new()
            .on_token(move |tok| { tl.lock().unwrap().push(tok.text.clone()); })
            .on_thought(move |t| { thl.lock().unwrap().push(t.to_string()); });

        let streamer = MockStreamer::new(vec![vec![
            InferenceToken::text("Thought: I should answer\nAction: FINAL_ANSWER 42"),
            InferenceToken::final_token(""),
        ]]);
        let mut loop_ = StreamingReActLoop::new(streamer, 5).with_callbacks(callbacks);
        loop_.run("p").await.unwrap();

        assert!(!token_log.lock().unwrap().is_empty());
        assert!(thought_log.lock().unwrap().iter().any(|t| t.contains("I should answer")));
    }

    #[test]
    fn test_streaming_session_avg_tokens_per_step() {
        let session = StreamingSession {
            steps: vec![
                StreamingStep { thought: String::new(), action: String::new(),
                    observation: String::new(), token_count: 6, logprob_sum: 0.0 },
                StreamingStep { thought: String::new(), action: String::new(),
                    observation: String::new(), token_count: 4, logprob_sum: 0.0 },
            ],
            total_tokens: 10, duration_ms: 0, completed: false,
        };
        assert_eq!(session.avg_tokens_per_step(), 5.0);
    }

    #[test]
    fn test_parse_streaming_response() {
        let (thought, action) = parse_streaming_response(
            "Thought: plan A\nAction: search {\"q\":\"rust\"}"
        );
        assert_eq!(thought, "plan A");
        assert_eq!(action, "search {\"q\":\"rust\"}");
    }
}

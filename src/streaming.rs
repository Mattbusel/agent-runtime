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

//! Engine events → OpenAI SSE chunks, with disconnect-triggered
//! cancellation.
//!
//! The guard is moved into the response stream: when the client disconnects,
//! axum drops the body, the guard's `Drop` runs and the engine is told to
//! cancel the request — no inference continues for a dead connection. For
//! non-streaming responses the guard can be [`DisconnectGuard::defuse`]d
//! once generation completed normally, so no pointless cancel is sent.

use std::convert::Infallible;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use axum::response::sse::{Event, KeepAlive, Sse};
use mini_vllm_core::{FinishReason, GenerationEvent};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use crate::openai;
use crate::SharedState;

/// Sends `Cancel` to the engine when the value is dropped. The guard is
/// moved into the response stream (SSE) or held by the handler future
/// (non-streaming), so a client disconnect cancels the generation.
struct CancelOnDrop {
    state: SharedState,
    request_id: String,
    armed: AtomicBool,
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if self.armed.load(Ordering::Relaxed) {
            tracing::debug!(request_id = %self.request_id, "response dropped; cancelling generation");
            self.state.engine.cancel(&self.request_id);
        }
    }
}

/// Public handle kept alive for the duration of a non-streaming response.
/// Dropping it cancels the request; call [`DisconnectGuard::defuse`] after
/// normal completion to disarm the cancellation.
pub struct DisconnectGuard(Option<Arc<CancelOnDrop>>);

impl DisconnectGuard {
    /// Disarm the guard: the generation finished normally, so dropping the
    /// guard must not send a (no-op but noisy) cancel to the engine.
    pub fn defuse(mut self) {
        if let Some(guard) = self.0.take() {
            guard.armed.store(false, Ordering::Relaxed);
        }
    }
}

/// JSON payload of one `chat.completion.chunk`.
#[derive(serde::Serialize)]
struct ChatChunkChoice<'a> {
    index: usize,
    delta: ChatChunkDelta<'a>,
    finish_reason: Option<&'static str>,
}

#[derive(serde::Serialize)]
struct ChatChunkDelta<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<&'a str>,
}

#[derive(serde::Serialize)]
struct ChatChunk<'a> {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatChunkChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<openai::UsageOut>,
}

fn chat_chunk(
    id: &str,
    model: &str,
    role: Option<&'static str>,
    content: Option<&str>,
    finish: Option<&'static str>,
    usage: Option<mini_vllm_core::Usage>,
) -> String {
    let chunk = ChatChunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created: openai::unix_now(),
        model: model.to_string(),
        usage: usage.map(openai::UsageOut::from_usage),
        choices: vec![ChatChunkChoice {
            index: 0,
            delta: ChatChunkDelta { role, content },
            finish_reason: finish,
        }],
    };
    serde_json::to_string(&chunk).unwrap_or_else(|_| "{}".to_string())
}

#[derive(serde::Serialize)]
struct TextChunkChoice<'a> {
    index: usize,
    text: &'a str,
    finish_reason: Option<&'static str>,
}

#[derive(serde::Serialize)]
struct TextChunk<'a> {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<TextChunkChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<openai::UsageOut>,
}

fn text_chunk(
    id: &str,
    model: &str,
    text: &str,
    finish: Option<&'static str>,
    usage: Option<mini_vllm_core::Usage>,
) -> String {
    let chunk = TextChunk {
        id: id.to_string(),
        object: "text_completion",
        created: openai::unix_now(),
        model: model.to_string(),
        usage: usage.map(openai::UsageOut::from_usage),
        choices: vec![TextChunkChoice {
            index: 0,
            text,
            finish_reason: finish,
        }],
    };
    serde_json::to_string(&chunk).unwrap_or_else(|_| "{}".to_string())
}

/// Engine-side errors are reported as a dedicated error frame, never as
/// assistant content (clients would otherwise render the message as text).
fn error_payload(message: &str) -> String {
    serde_json::json!({
        "error": {
            "message": message,
            "type": "server_error",
        }
    })
    .to_string()
}

fn finish_str(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop | FinishReason::Cancelled => "stop",
        FinishReason::Length => "length",
        FinishReason::Error => "stop",
    }
}

fn sse_event(payload: String) -> Result<Event, Infallible> {
    Ok(Event::default().data(payload))
}

/// Build a streaming chat-completion response.
pub fn chat_sse(
    state: SharedState,
    request_id: String,
    completion_id: String,
    events: tokio::sync::mpsc::Receiver<GenerationEvent>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let model = state.model_id.clone();
    let guard = Arc::new(CancelOnDrop {
        state,
        request_id,
        armed: AtomicBool::new(true),
    });
    let guard2 = Arc::clone(&guard);
    let terminal = Arc::new(AtomicBool::new(false));
    let seen_terminal = Arc::clone(&terminal);
    let stream = ReceiverStream::new(events).map(move |ev| {
        if matches!(
            &ev,
            GenerationEvent::Finished { .. } | GenerationEvent::Error { .. }
        ) {
            seen_terminal.store(true, Ordering::Relaxed);
            guard2.armed.store(false, Ordering::Relaxed);
        }
        let _ = &guard2;
        match ev {
            GenerationEvent::Finished {
                reason: FinishReason::Cancelled | FinishReason::Error,
                ..
            } => sse_event(error_payload("generation cancelled or failed")),
            GenerationEvent::Token { text, .. } => sse_event(chat_chunk(
                &completion_id,
                &model,
                None,
                Some(&text),
                None,
                None,
            )),
            GenerationEvent::Finished { reason, usage } => sse_event(chat_chunk(
                &completion_id,
                &model,
                None,
                Some(""),
                Some(finish_str(reason)),
                Some(usage),
            )),
            GenerationEvent::Error { kind, message } => sse_event(
                serde_json::json!({"error":{"message":message,"type":kind.as_str()}}).to_string(),
            ),
        }
    });
    let done = tokio_stream::iter(vec![Ok(Event::default().data("[DONE]"))]);
    let incomplete = tokio_stream::iter([()]).filter_map(move |_| {
        if terminal.load(Ordering::Relaxed) {
            None
        } else {
            Some(sse_event(error_payload(
                "generation ended without a terminal event",
            )))
        }
    });
    Sse::new(stream.chain(incomplete).chain(done)).keep_alive(KeepAlive::default())
}

/// Build a streaming text-completion response.
pub fn completion_sse(
    state: SharedState,
    request_id: String,
    completion_id: String,
    events: tokio::sync::mpsc::Receiver<GenerationEvent>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let model = state.model_id.clone();
    let guard = Arc::new(CancelOnDrop {
        state,
        request_id,
        armed: AtomicBool::new(true),
    });
    let guard2 = Arc::clone(&guard);
    let terminal = Arc::new(AtomicBool::new(false));
    let seen_terminal = Arc::clone(&terminal);
    let stream = ReceiverStream::new(events).map(move |ev| {
        if matches!(
            &ev,
            GenerationEvent::Finished { .. } | GenerationEvent::Error { .. }
        ) {
            seen_terminal.store(true, Ordering::Relaxed);
            guard2.armed.store(false, Ordering::Relaxed);
        }
        let _ = &guard2;
        match ev {
            GenerationEvent::Finished {
                reason: FinishReason::Cancelled | FinishReason::Error,
                ..
            } => sse_event(error_payload("generation cancelled or failed")),
            GenerationEvent::Token { text, .. } => {
                sse_event(text_chunk(&completion_id, &model, &text, None, None))
            }
            GenerationEvent::Finished { reason, usage } => sse_event(text_chunk(
                &completion_id,
                &model,
                "",
                Some(finish_str(reason)),
                Some(usage),
            )),
            GenerationEvent::Error { kind, message } => sse_event(
                serde_json::json!({"error":{"message":message,"type":kind.as_str()}}).to_string(),
            ),
        }
    });
    let done = tokio_stream::iter(vec![Ok(Event::default().data("[DONE]"))]);
    let incomplete = tokio_stream::iter([()]).filter_map(move |_| {
        if terminal.load(Ordering::Relaxed) {
            None
        } else {
            Some(sse_event(error_payload(
                "generation ended without a terminal event",
            )))
        }
    });
    Sse::new(stream.chain(incomplete).chain(done)).keep_alive(KeepAlive::default())
}

/// Guard for the non-streaming path: cancels the request when the handler
/// future is dropped (e.g. client disconnect).
pub fn disconnect_guard(state: SharedState, request_id: String) -> DisconnectGuard {
    DisconnectGuard(Some(Arc::new(CancelOnDrop {
        state,
        request_id,
        armed: AtomicBool::new(true),
    })))
}

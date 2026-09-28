//! HTTP handlers: OpenAI-compatible endpoints plus health/metrics.

// Returning `axum::Response` from helper fallible functions trips this lint;
// the size is an axum implementation detail and handlers must return the
// concrete type.
#![allow(clippy::result_large_err)]

use crate::preprocessing::Input;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json as JsonResponse, Response};
use mini_vllm_core::{FinishReason, GenerationEvent, GenerationRequest, SamplingParams, Usage};
use mini_vllm_tokenizer::ChatMessage;

use crate::openai::{
    self, error_body, ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessageIn,
    ChoiceMessage, CompletionRequest, CompletionResponse, ModelEntry, ModelList, StopField,
    TextChoice, UsageOut,
};
use crate::streaming::{self, DisconnectGuard};
use crate::SharedState;

pub(crate) fn api_error(
    status: StatusCode,
    message: impl Into<String>,
    kind: &'static str,
) -> Response {
    (status, JsonResponse(error_body(message, kind))).into_response()
}

fn bad_request(message: impl Into<String>) -> Response {
    api_error(StatusCode::BAD_REQUEST, message, "invalid_request_error")
}

fn unavailable(message: impl Into<String>) -> Response {
    api_error(
        StatusCode::SERVICE_UNAVAILABLE,
        message,
        "engine_unavailable",
    )
}

fn internal(message: impl Into<String>) -> Response {
    api_error(StatusCode::INTERNAL_SERVER_ERROR, message, "internal_error")
}

pub async fn health(State(state): State<SharedState>) -> Response {
    if state.engine.is_accepting() {
        JsonResponse(serde_json::json!({"ok":true})).into_response()
    } else {
        unavailable("engine is shutting down")
    }
}

pub async fn models(State(state): State<SharedState>) -> Response {
    let list = ModelList {
        object: "list",
        data: vec![ModelEntry {
            id: state.model_id.clone(),
            object: "model",
            created: openai::unix_now(),
            owned_by: "mini-vllm",
        }],
    };
    JsonResponse(list).into_response()
}

pub async fn metrics(
    State(state): State<SharedState>,
    axum::Extension(processor): axum::Extension<crate::preprocessing::Preprocessor>,
) -> Response {
    let mut snapshot = serde_json::json!(state.engine.metrics());
    snapshot["preprocessing"] = serde_json::json!(processor.snapshot());
    JsonResponse(snapshot).into_response()
}

struct ResolvedSampling {
    params: SamplingParams,
    max_new_tokens: usize,
    stop_strings: Vec<String>,
}

/// Validate the client-provided generation parameters.
fn resolve_sampling(
    max_tokens: usize,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    repetition_penalty: Option<f32>,
    seed: Option<u64>,
    stop: Option<StopField>,
) -> Result<ResolvedSampling, Response> {
    let params = SamplingParams {
        temperature: temperature.unwrap_or(1.0),
        top_p,
        top_k,
        repetition_penalty,
        seed,
    };
    let probe = GenerationRequest {
        id: String::new(),
        prompt_token_ids: vec![1],
        sampling: params.clone(),
        max_new_tokens: 1,
        stop_token_ids: vec![],
        stop_strings: vec![],
    };
    // `validate` also checks sampling-parameter sanity; max_model_len is
    // enforced later against the real prompt.
    probe
        .validate(usize::MAX)
        .map_err(|e| bad_request(e.to_string()))?;
    if max_tokens == 0 {
        return Err(bad_request("max_tokens must be > 0"));
    }
    Ok(ResolvedSampling {
        params,
        max_new_tokens: max_tokens,
        stop_strings: stop.map(StopField::into_vec).unwrap_or_default(),
    })
}

/// Tokenize + validate + submit to the engine.
fn submit(
    state: &SharedState,
    prompt_token_ids: Vec<u32>,
    sampling: ResolvedSampling,
) -> Result<(String, tokio::sync::mpsc::Receiver<GenerationEvent>), Response> {
    let request_id = format!("req-{}", openai::new_completion_id(false));
    let request = GenerationRequest {
        id: request_id.clone(),
        prompt_token_ids,
        sampling: sampling.params,
        max_new_tokens: sampling.max_new_tokens,
        stop_token_ids: vec![],
        stop_strings: sampling.stop_strings,
    };
    request
        .validate(state.max_model_len)
        .map_err(|e| bad_request(e.to_string()))?;
    let events = state.engine.generate(request).map_err(|e| match e {
        mini_vllm_engine::EngineApiError::InvalidRequest(message) => bad_request(message),
        other => unavailable(other.to_string()),
    })?;
    Ok((request_id, events))
}

/// Drain the event stream into the final (text, reason, usage).
///
/// `guard` keeps disconnect-cancellation alive while this future runs; if
/// the client disappears, the future is dropped and the guard cancels the
/// generation. On normal completion the guard is disarmed so no pointless
/// cancel reaches the engine.
async fn collect(
    mut events: tokio::sync::mpsc::Receiver<GenerationEvent>,
    guard: DisconnectGuard,
) -> Result<(String, FinishReason, Usage), Response> {
    let mut text = String::new();
    loop {
        match events.recv().await {
            Some(GenerationEvent::Token { text: delta, .. }) => text.push_str(&delta),
            Some(GenerationEvent::Finished { reason, usage }) => {
                guard.defuse();
                if matches!(reason, FinishReason::Cancelled | FinishReason::Error) {
                    return Err(internal(format!("generation {}", reason.as_str())));
                }
                return Ok((text, reason, usage));
            }
            Some(GenerationEvent::Error { kind, message }) => {
                guard.defuse();
                let status = match kind {
                    mini_vllm_core::GenerationErrorKind::InvalidRequest => StatusCode::BAD_REQUEST,
                    mini_vllm_core::GenerationErrorKind::Overloaded => {
                        StatusCode::SERVICE_UNAVAILABLE
                    }
                    mini_vllm_core::GenerationErrorKind::Timeout => StatusCode::GATEWAY_TIMEOUT,
                    mini_vllm_core::GenerationErrorKind::Execution => {
                        StatusCode::INTERNAL_SERVER_ERROR
                    }
                };
                return Err(api_error(status, message, kind.as_str()));
            }
            None => {
                return Err(internal(
                    "request cancelled by the engine (consumer too slow, or engine shutting down)",
                ));
            }
        }
    }
}

pub async fn completions(
    State(state): State<SharedState>,
    Input(req, ticket): Input<CompletionRequest>,
) -> Response {
    if req
        .model
        .as_ref()
        .is_some_and(|model| model != &state.model_id)
    {
        return api_error(
            StatusCode::NOT_FOUND,
            "requested model is not served",
            "model_not_found",
        );
    }
    let sampling = resolve_sampling(
        req.max_tokens
            .unwrap_or_else(|| state.engine.default_max_new_tokens()),
        req.temperature,
        req.top_p,
        req.top_k,
        req.repetition_penalty,
        req.seed,
        req.stop,
    );
    let sampling = match sampling {
        Ok(s) => s,
        Err(resp) => return resp,
    };
    let tokenizer = state.tokenizer.clone();
    let prompt_token_ids = match ticket
        .run(move || {
            tokenizer
                .encode(&req.prompt, true)
                .map_err(|e| format!("tokenization failed: {e}"))
        })
        .await
    {
        Ok(ids) => ids,
        Err(response) => return response,
    };
    let (request_id, events) = match submit(&state, prompt_token_ids, sampling) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let model_name = req.model.unwrap_or_else(|| state.model_id.clone());
    let completion_id = openai::new_completion_id(false);

    if req.stream {
        return streaming::completion_sse(state, request_id, completion_id, events).into_response();
    }

    let guard = streaming::disconnect_guard(state, request_id);
    let (text, reason, usage) = match collect(events, guard).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let response = CompletionResponse {
        id: completion_id,
        object: "text_completion",
        created: openai::unix_now(),
        model: model_name,
        choices: vec![TextChoice {
            index: 0,
            text,
            finish_reason: finish_reason_str(reason),
        }],
        usage: UsageOut::from_usage(usage),
    };
    JsonResponse(response).into_response()
}

pub async fn chat_completions(
    State(state): State<SharedState>,
    Input(req, ticket): Input<ChatCompletionRequest>,
) -> Response {
    if req
        .model
        .as_ref()
        .is_some_and(|model| model != &state.model_id)
    {
        return api_error(
            StatusCode::NOT_FOUND,
            "requested model is not served",
            "model_not_found",
        );
    }
    let sampling = resolve_sampling(
        req.max_tokens
            .unwrap_or_else(|| state.engine.default_max_new_tokens()),
        req.temperature,
        req.top_p,
        req.top_k,
        req.repetition_penalty,
        req.seed,
        req.stop,
    );
    let sampling = match sampling {
        Ok(s) => s,
        Err(resp) => return resp,
    };
    let tokenizer = state.tokenizer.clone();
    let template = state.template.clone();
    let prompt_token_ids = match ticket
        .run(move || {
            let messages: Vec<ChatMessage> = req
                .messages
                .into_iter()
                .map(|m: ChatMessageIn| ChatMessage::new(m.role, m.content))
                .collect();
            let prompt = template.render(&messages).map_err(|e| e.to_string())?;
            tokenizer
                .encode(&prompt, false)
                .map_err(|e| format!("tokenization failed: {e}"))
        })
        .await
    {
        Ok(ids) => ids,
        Err(response) => return response,
    };
    let (request_id, events) = match submit(&state, prompt_token_ids, sampling) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let model_name = req.model.unwrap_or_else(|| state.model_id.clone());
    let completion_id = openai::new_completion_id(true);

    if req.stream {
        return streaming::chat_sse(state, request_id, completion_id, events).into_response();
    }

    let guard = streaming::disconnect_guard(state, request_id);
    let (text, reason, usage) = match collect(events, guard).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let response = ChatCompletionResponse {
        id: completion_id,
        object: "chat.completion",
        created: openai::unix_now(),
        model: model_name,
        choices: vec![ChatChoice {
            index: 0,
            message: ChoiceMessage {
                role: "assistant",
                content: text,
            },
            finish_reason: finish_reason_str(reason),
        }],
        usage: UsageOut::from_usage(usage),
    };
    JsonResponse(response).into_response()
}

fn finish_reason_str(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop | FinishReason::Cancelled => "stop",
        FinishReason::Length => "length",
        FinishReason::Error => "stop",
    }
}

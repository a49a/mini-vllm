//! HTTP integration tests (design doc §52.6).
//!
//! The engine is mocked so HTTP behavior is tested in isolation; the real
//! engine (including batching and cancellation) is covered by the engine
//! crate's tests.

use std::sync::Mutex;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use mini_vllm_core::{FinishReason, GenerationEvent, GenerationRequest, Usage};
use mini_vllm_server::{routes, AppState, EngineApi, SharedState};
use mini_vllm_tokenizer::{QwenChatTemplate, TokenizerWrapper};
use tokio::sync::mpsc;
use tower::ServiceExt;

struct MockEngine {
    events: Mutex<Vec<GenerationEvent>>,
    cancelled: Mutex<Vec<String>>,
}

impl MockEngine {
    fn events() -> Vec<GenerationEvent> {
        vec![
            GenerationEvent::Token {
                token_id: 4,
                text: "a".into(),
            },
            GenerationEvent::Token {
                token_id: 9,
                text: " b".into(),
            },
            GenerationEvent::Finished {
                reason: FinishReason::Stop,
                usage: Usage {
                    prompt_tokens: 5,
                    completion_tokens: 2,
                    total_tokens: 7,
                },
            },
        ]
    }
}

impl EngineApi for MockEngine {
    fn generate(
        &self,
        _request: GenerationRequest,
    ) -> Result<mpsc::Receiver<GenerationEvent>, mini_vllm_engine::EngineApiError> {
        let (tx, rx) = mpsc::channel(8);
        let events: Vec<_> = self.events.lock().unwrap().clone();
        tokio::spawn(async move {
            for ev in events {
                if tx.send(ev).await.is_err() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        });
        Ok(rx)
    }

    fn cancel(&self, request_id: &str) {
        self.cancelled.lock().unwrap().push(request_id.to_string());
    }

    fn metrics(&self) -> mini_vllm_engine::MetricsSnapshot {
        mini_vllm_engine::Metrics::new().snapshot()
    }
}

fn state() -> (SharedState, std::sync::Arc<MockEngine>) {
    let mock = std::sync::Arc::new(MockEngine {
        events: Mutex::new(MockEngine::events()),
        cancelled: Mutex::new(Vec::new()),
    });
    let engine: std::sync::Arc<dyn EngineApi> = mock.clone();
    let state: SharedState = std::sync::Arc::new(AppState {
        engine,
        tokenizer: std::sync::Arc::new(TokenizerWrapper::from_inner(
            mini_vllm_tokenizer::testutil::char_tokenizer(),
        )),
        template: std::sync::Arc::new(QwenChatTemplate),
        model_id: "test-model".into(),
        max_model_len: 64,
        vocab_size: 12,
    });
    (state, mock)
}

async fn post_json(uri: &str, body: String) -> axum::http::Response<Body> {
    let (state, _mock) = state();
    let app = routes::router(state);
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();
    app.oneshot(req).await.unwrap()
}

#[tokio::test]
async fn health_and_models() {
    let (state, _) = state();
    let app = routes::router(state);

    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let res = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(res.into_body(), 1 << 20)
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["data"][0]["id"], "test-model");
}

#[tokio::test]
async fn completions_non_streaming() {
    let res = post_json(
        "/v1/completions",
        serde_json::json!({
            "model": "test-model",
            "prompt": "a b c",
            "max_tokens": 16,
            "temperature": 0.7
        })
        .to_string(),
    )
    .await;
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(res.into_body(), 1 << 20)
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["object"], "text_completion");
    assert_eq!(v["choices"][0]["text"], "a b");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert_eq!(v["usage"]["total_tokens"], 7);
}

#[tokio::test]
async fn completions_streaming_sse() {
    let res = post_json(
        "/v1/completions",
        serde_json::json!({
            "prompt": "a b c",
            "max_tokens": 16,
            "stream": true
        })
        .to_string(),
    )
    .await;
    assert_eq!(res.status(), StatusCode::OK);
    assert!(res
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .starts_with("text/event-stream"));
    let bytes = axum::body::to_bytes(res.into_body(), 1 << 20)
        .await
        .unwrap();
    let body = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(body.contains("text_completion"));
    assert!(body.contains("\"finish_reason\":\"stop\""));
    assert!(body.trim_end().ends_with("data: [DONE]"));
}

#[tokio::test]
async fn chat_completions_non_streaming() {
    let res = post_json(
        "/v1/chat/completions",
        serde_json::json!({
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "a b"}
            ],
            "temperature": 0.7,
            "max_tokens": 32
        })
        .to_string(),
    )
    .await;
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(res.into_body(), 1 << 20)
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["choices"][0]["message"]["role"], "assistant");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn unknown_fields_are_rejected() {
    let res = post_json(
        "/v1/completions",
        serde_json::json!({
            "prompt": "a",
            "logprobs": true
        })
        .to_string(),
    )
    .await;
    assert_eq!(res.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn oversized_prompt_is_rejected() {
    // 32 tokens > max_model_len(64)? Use a long prompt of valid tokens.
    let prompt = "a ".repeat(100);
    let res = post_json(
        "/v1/completions",
        serde_json::json!({ "prompt": prompt.trim(), "max_tokens": 64 }).to_string(),
    )
    .await;
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    let bytes = axum::body::to_bytes(res.into_body(), 1 << 20)
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(v["error"]["message"]
        .as_str()
        .unwrap()
        .contains("max_model_len"));
}

#[tokio::test]
async fn bad_role_is_rejected() {
    let res = post_json(
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "wizard", "content": "hi"}]
        })
        .to_string(),
    )
    .await;
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn interrupted_generations_are_errors_in_json_and_sse() {
    for endpoint in ["/v1/completions", "/v1/chat/completions"] {
        for stream in [false, true] {
            for reason in [
                None,
                Some(FinishReason::Cancelled),
                Some(FinishReason::Error),
            ] {
                let (state, mock) = state();
                let mut events = vec![GenerationEvent::Token {
                    token_id: 4,
                    text: "partial".into(),
                }];
                if let Some(reason) = reason {
                    events.push(GenerationEvent::Finished {
                        reason,
                        usage: Usage::default(),
                    });
                }
                *mock.events.lock().unwrap() = events;
                let mut body = serde_json::json!({"max_tokens": 4, "stream": stream});
                if endpoint.contains("chat") {
                    body["messages"] = serde_json::json!([{"role":"user", "content":"a"}]);
                } else {
                    body["prompt"] = "a".into();
                }
                let request = Request::builder()
                    .method("POST")
                    .uri(endpoint)
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap();
                let response = routes::router(state).oneshot(request).await.unwrap();
                assert_eq!(
                    response.status(),
                    if stream {
                        StatusCode::OK
                    } else {
                        StatusCode::INTERNAL_SERVER_ERROR
                    }
                );
                let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
                    .await
                    .unwrap();
                let text = String::from_utf8(bytes.to_vec()).unwrap();
                assert!(text.contains("\"error\""), "{text}");
                assert!(!text.contains("\"finish_reason\":\"stop\""), "{text}");
                if stream {
                    assert!(text.trim_end().ends_with("data: [DONE]"));
                }
            }
        }
    }
}

#[tokio::test]
async fn successful_json_and_sse_do_not_send_cancel_commands() {
    for streaming in [false, true] {
        let (state, mock) = state();
        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::json!({"prompt":"a","max_tokens":4,"stream":streaming}).to_string(),
            ))
            .unwrap();
        let response = routes::router(state).oneshot(request).await.unwrap();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .unwrap();
        if streaming {
            assert!(String::from_utf8_lossy(&bytes).contains("\"completion_tokens\":2"));
        }
        assert!(mock.cancelled.lock().unwrap().is_empty());
    }
}

#[tokio::test]
async fn real_engine_http_pipeline_matches_json_and_sse() {
    let mut cfg = mini_vllm_model::testutil::tiny_config();
    cfg.vocab_size = 12;
    let weights = mini_vllm_model::testutil::random_tensors(&cfg, false, &candle_core::Device::Cpu);
    let model = std::sync::Arc::new(
        mini_vllm_model::Qwen2::load(
            cfg,
            &weights,
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .unwrap(),
    );
    let tokenizer = std::sync::Arc::new(mini_vllm_tokenizer::testutil::wrapper());
    let engine = mini_vllm_engine::spawn_engine(
        model,
        Some(tokenizer.clone()),
        mini_vllm_core::EngineConfig {
            max_model_len: 64,
            max_batch_tokens: 2,
            kv_block_size: 2,
            prefix_cache_tokens: 16,
            ..Default::default()
        },
        1,
    )
    .unwrap();
    let state = std::sync::Arc::new(AppState {
        engine: std::sync::Arc::new(engine),
        tokenizer,
        template: std::sync::Arc::new(QwenChatTemplate),
        model_id: "test-model".into(),
        max_model_len: 64,
        vocab_size: 12,
    });
    for chat in [false, true] {
        let mut outputs = Vec::new();
        for streaming in [false, true] {
            let mut body = serde_json::json!({"max_tokens":4,"temperature":0,"stream":streaming});
            let endpoint = if chat {
                body["messages"] = serde_json::json!([{"role":"user","content":"a b c d"}]);
                "/v1/chat/completions"
            } else {
                body["prompt"] = "a b c d".into();
                "/v1/completions"
            };
            let request = Request::builder()
                .method("POST")
                .uri(endpoint)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap();
            let response = routes::router(state.clone())
                .oneshot(request)
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let bytes = tokio::time::timeout(
                Duration::from_secs(5),
                axum::body::to_bytes(response.into_body(), 1 << 20),
            )
            .await
            .unwrap()
            .unwrap();
            if streaming {
                let mut text = String::new();
                let mut usage = None;
                let mut done = false;
                for line in String::from_utf8_lossy(&bytes)
                    .lines()
                    .filter_map(|l| l.strip_prefix("data: "))
                {
                    if line == "[DONE]" {
                        done = true;
                        continue;
                    }
                    let v: serde_json::Value = serde_json::from_str(line).unwrap();
                    assert!(v.get("error").is_none());
                    let delta = if chat {
                        &v["choices"][0]["delta"]["content"]
                    } else {
                        &v["choices"][0]["text"]
                    };
                    text.push_str(delta.as_str().unwrap_or(""));
                    if let Some(u) = v.get("usage") {
                        usage = Some(u.clone());
                    }
                }
                assert!(done);
                outputs.push((text, usage.unwrap()));
            } else {
                let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
                let text = if chat {
                    &v["choices"][0]["message"]["content"]
                } else {
                    &v["choices"][0]["text"]
                };
                outputs.push((text.as_str().unwrap().to_string(), v["usage"].clone()));
            }
        }
        assert_eq!(outputs[0], outputs[1]);
    }
}

#[tokio::test]
async fn dropping_an_unfinished_response_cancels_once() {
    let (state, mock) = state();
    {
        let _guard = mini_vllm_server::streaming::disconnect_guard(state, "unfinished".into());
    }
    assert_eq!(*mock.cancelled.lock().unwrap(), vec!["unfinished"]);
}

#[tokio::test]
async fn unknown_model_is_not_found_for_both_endpoints() {
    for uri in ["/v1/completions", "/v1/chat/completions"] {
        let body = if uri.ends_with("/chat/completions") {
            serde_json::json!({"model":"unknown","messages":[{"role":"user","content":"a"}],"max_tokens":2})
        } else {
            serde_json::json!({"model":"unknown","prompt":"a","max_tokens":2})
        };
        assert_eq!(
            post_json(uri, body.to_string()).await.status(),
            StatusCode::NOT_FOUND
        );
    }
}

#[tokio::test]
async fn async_overload_is_503_or_typed_sse_error() {
    for stream in [false, true] {
        let (state, mock) = state();
        *mock.events.lock().unwrap() = vec![GenerationEvent::Error {
            kind: mini_vllm_core::GenerationErrorKind::Overloaded,
            message: "waiting queue is full".into(),
        }];
        let res = routes::router(state)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({"prompt":"a","max_tokens":2,"stream":stream})
                            .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            res.status(),
            if stream {
                StatusCode::OK
            } else {
                StatusCode::SERVICE_UNAVAILABLE
            }
        );
        let body = axum::body::to_bytes(res.into_body(), 1 << 20)
            .await
            .unwrap();
        assert!(String::from_utf8_lossy(&body).contains("engine_overloaded"));
        assert!(mock.cancelled.lock().unwrap().is_empty());
    }
}

#[tokio::test]
async fn tcp_disconnect_cancels_an_unfinished_request() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let (state, mock) = state();
    // The producer sends slowly enough for the client to close mid-stream.
    *mock.events.lock().unwrap() = (0..500)
        .map(|_| GenerationEvent::Token {
            token_id: 4,
            text: "a".into(),
        })
        .collect();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, routes::router(state)).await.unwrap();
    });
    let mut socket = tokio::net::TcpStream::connect(addr).await.unwrap();
    let body = r#"{"prompt":"a","max_tokens":2,"stream":true}"#;
    socket.write_all(format!("POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",body.len(),body).as_bytes()).await.unwrap();
    let mut buf = [0; 4096];
    let n = tokio::time::timeout(Duration::from_secs(2), socket.read(&mut buf))
        .await
        .unwrap()
        .unwrap();
    assert!(n > 0);
    drop(socket);
    tokio::time::timeout(Duration::from_secs(2), async {
        while mock.cancelled.lock().unwrap().is_empty() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    server.abort();
}

#[tokio::test]
async fn empty_messages_and_nested_unknown_fields_are_rejected() {
    for body in [
        serde_json::json!({"messages":[],"max_tokens":1}),
        serde_json::json!({"messages":[{"role":"user","content":"a","tool_calls":[]}],"max_tokens":1}),
    ] {
        let response = post_json("/v1/chat/completions", body.to_string()).await;
        assert!(response.status().is_client_error());
    }
}

struct DefaultsEngine;
impl EngineApi for DefaultsEngine {
    fn default_max_new_tokens(&self) -> usize {
        3
    }
    fn generate(
        &self,
        request: GenerationRequest,
    ) -> Result<mpsc::Receiver<GenerationEvent>, mini_vllm_engine::EngineApiError> {
        let (tx, rx) = mpsc::channel(1);
        tx.try_send(GenerationEvent::Finished {
            reason: FinishReason::Length,
            usage: Usage {
                prompt_tokens: request.prompt_token_ids.len(),
                completion_tokens: request.max_new_tokens,
                total_tokens: request.prompt_token_ids.len() + request.max_new_tokens,
            },
        })
        .unwrap();
        Ok(rx)
    }
    fn cancel(&self, _: &str) {}
    fn metrics(&self) -> mini_vllm_engine::MetricsSnapshot {
        mini_vllm_engine::Metrics::new().snapshot()
    }
}
#[tokio::test]
async fn omitted_limits_use_engine_default_and_explicit_limits_win() {
    for chat in [false, true] {
        for limit in [None, Some(2)] {
            let (state, _) = state();
            let app = routes::router(std::sync::Arc::new(AppState {
                engine: std::sync::Arc::new(DefaultsEngine),
                tokenizer: state.tokenizer.clone(),
                template: state.template.clone(),
                model_id: state.model_id.clone(),
                max_model_len: 64,
                vocab_size: 12,
            }));
            let mut body = if chat {
                serde_json::json!({"messages":[{"role":"user","content":"a"}]})
            } else {
                serde_json::json!({"prompt":"a"})
            };
            if let Some(limit) = limit {
                body["max_tokens"] = limit.into();
            }
            let response = app
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(if chat {
                            "/v1/chat/completions"
                        } else {
                            "/v1/completions"
                        })
                        .header("content-type", "application/json")
                        .body(Body::from(body.to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
                .await
                .unwrap();
            let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(json["usage"]["completion_tokens"], limit.unwrap_or(3));
        }
    }
}

#[tokio::test]
async fn engine_deadline_maps_to_gateway_timeout() {
    let (state, mock) = state();
    *mock.events.lock().unwrap() = vec![GenerationEvent::Error {
        kind: mini_vllm_core::GenerationErrorKind::Timeout,
        message: "deadline exceeded".into(),
    }];
    let response = routes::router(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"prompt":"a","max_tokens":1}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
}

//! OpenAI-compatible request/response types (useful subset, design doc §22).
//!
//! Unknown fields are rejected explicitly — the server never silently
//! claims behavior it does not implement.

use serde::{Deserialize, Serialize};

/// `stop` may be a single string or a list.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StopField {
    One(String),
    Many(Vec<String>),
}

impl StopField {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            StopField::One(s) => vec![s],
            StopField::Many(v) => v,
        }
    }
}

fn default_max_tokens() -> usize {
    512
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Not part of the standard OpenAI API; accepted like vLLM does.
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<StopField>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessageIn>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<StopField>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessageIn {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct UsageOut {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

impl UsageOut {
    pub fn from_usage(u: mini_vllm_core::Usage) -> Self {
        Self {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ChoiceMessage {
    pub role: &'static str,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChoiceMessage,
    pub finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: UsageOut,
}

#[derive(Debug, Serialize)]
pub struct TextChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<TextChoice>,
    pub usage: UsageOut,
}

#[derive(Debug, Serialize)]
pub struct ModelEntry {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelEntry>,
}

#[derive(Debug, Serialize)]
pub struct ErrorBody {
    pub error: ErrorDetail,
}

#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
}

pub fn error_body(message: impl Into<String>, kind: &'static str) -> ErrorBody {
    ErrorBody {
        error: ErrorDetail {
            message: message.into(),
            kind,
        },
    }
}

/// Random `chatcmpl-…` / `cmpl-…` style ids.
pub fn new_completion_id(chat: bool) -> String {
    const HEX: &[u8] = b"0123456789abcdef";
    let mut buf = [0u8; 12];
    // `rand::rng()` thread RNG; ids are not security sensitive here.
    use rand::Rng;
    rand::rng().fill(&mut buf);
    let s: String = buf
        .iter()
        .map(|b| HEX[(b & 0xf) as usize] as char)
        .collect();
    format!("{}{}", if chat { "chatcmpl-" } else { "cmpl-" }, s)
}

pub fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

//! Chat templates.
//!
//! Building a universal Jinja interpreter is out of scope for the first
//! version (see the design doc). Instead the behavior is isolated behind a
//! narrow trait; the first supported format is Qwen2/2.5 ChatML.

use mini_vllm_core::Result;

/// One message in a chat conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

/// Render chat messages into a raw prompt string for the model.
pub trait ChatTemplate: Send + Sync {
    /// Render with `add_generation_prompt = true` (ready for the assistant).
    fn render(&self, messages: &[ChatMessage]) -> Result<String>;

    /// Marker of the template implementation, useful for logs/inspect.
    fn name(&self) -> &'static str;
}

/// Qwen2 / Qwen2.5 ChatML format:
///
/// ```text
/// <|im_start|>system
/// ...<|im_end|>
/// <|im_start|>user
/// ...<|im_end|>
/// <|im_start|>assistant
/// ```
#[derive(Debug, Clone, Default)]
pub struct QwenChatTemplate;

const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";

impl ChatTemplate for QwenChatTemplate {
    fn render(&self, messages: &[ChatMessage]) -> Result<String> {
        if messages.is_empty() {
            return Err(mini_vllm_core::Error::InvalidRequest(
                "messages must not be empty".into(),
            ));
        }
        let mut out = String::new();
        for m in messages {
            match m.role.as_str() {
                "system" | "user" | "assistant" => {}
                other => {
                    return Err(mini_vllm_core::Error::InvalidRequest(format!(
                        "unsupported chat role `{other}`"
                    )))
                }
            }
            out.push_str(IM_START);
            out.push_str(&m.role);
            out.push('\n');
            out.push_str(&m.content);
            out.push_str(IM_END);
            out.push('\n');
        }
        // add_generation_prompt
        out.push_str(IM_START);
        out.push_str("assistant\n");
        Ok(out)
    }

    fn name(&self) -> &'static str {
        "qwen-chatml"
    }
}

/// A model-validated template. Only the exact bundled Qwen2.5 Instruct template
/// is supported; unknown Jinja must never silently fall back to another prompt.
/// Tools and multimodal messages remain outside this server's API.
#[derive(Debug, Clone)]
pub struct ModelChatTemplate;
impl ModelChatTemplate {
    pub fn from_model_dir(dir: impl AsRef<std::path::Path>) -> Result<Self> {
        let path = dir.as_ref().join("tokenizer_config.json");
        let bytes = std::fs::read(&path).map_err(|e| {
            mini_vllm_core::Error::InvalidRequest(format!("cannot read {}: {e}", path.display()))
        })?;
        let config: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| {
            mini_vllm_core::Error::InvalidRequest(format!("invalid tokenizer_config.json: {e}"))
        })?;
        Self::from_config(&config)
    }
    fn from_config(config: &serde_json::Value) -> Result<Self> {
        if config.get("chat_template").and_then(|v| v.as_str())
            != Some(include_str!("templates/qwen2.5-instruct.jinja"))
        {
            return Err(mini_vllm_core::Error::InvalidRequest(
                "unsupported or missing chat_template: serve requires the bundled Qwen2.5 Instruct template; see docs/chat-templates.md".into()));
        }
        Ok(Self)
    }
}
impl ChatTemplate for ModelChatTemplate {
    fn render(&self, messages: &[ChatMessage]) -> Result<String> {
        // Validate even empty input before deciding whether to add the default.
        let rendered = QwenChatTemplate.render(messages)?;
        if messages[0].role == "system" {
            return Ok(rendered);
        }
        Ok(format!("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n{rendered}"))
    }
    fn name(&self) -> &'static str {
        "qwen2.5-instruct-validated"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_chatml_with_generation_prompt() {
        let msgs = vec![
            ChatMessage::new("system", "You are a concise assistant."),
            ChatMessage::new("user", "What is ownership in Rust?"),
        ];
        let out = QwenChatTemplate.render(&msgs).unwrap();
        assert_eq!(
            out,
            "<|im_start|>system\nYou are a concise assistant.<|im_end|>\n\
             <|im_start|>user\nWhat is ownership in Rust?<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn renders_without_system_message() {
        let msgs = vec![ChatMessage::new("user", "hi")];
        let out = QwenChatTemplate.render(&msgs).unwrap();
        assert!(out.starts_with("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"));
    }

    #[test]
    fn rejects_unknown_roles() {
        let msgs = vec![ChatMessage::new("tool", "x")];
        assert!(QwenChatTemplate.render(&msgs).is_err());
    }
    #[test]
    fn rejects_missing_modified_and_named_templates() {
        for config in [
            serde_json::json!({}),
            serde_json::json!({"chat_template":"custom"}),
            serde_json::json!({"chat_template": {"default": "custom"}}),
        ] {
            assert!(ModelChatTemplate::from_config(&config).is_err());
        }
        assert!(ModelChatTemplate::from_config(&serde_json::json!({
            "chat_template": include_str!("templates/qwen2.5-instruct.jinja")
        }))
        .is_ok());
    }

    #[test]
    fn matches_transformers_rendered_fixtures() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/chat-reference.json")).unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let messages: Vec<_> = case["messages"]
                .as_array()
                .unwrap()
                .iter()
                .map(|m| {
                    ChatMessage::new(m["role"].as_str().unwrap(), m["content"].as_str().unwrap())
                })
                .collect();
            assert_eq!(
                ModelChatTemplate.render(&messages).unwrap(),
                case["rendered"].as_str().unwrap(),
                "{}",
                case["name"]
            );
        }
    }
}

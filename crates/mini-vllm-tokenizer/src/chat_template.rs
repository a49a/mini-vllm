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
}

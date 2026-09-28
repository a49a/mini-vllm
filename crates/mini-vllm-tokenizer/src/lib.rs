//! Tokenizer subsystem: Hugging Face tokenizer loading, incremental
//! detokenization for streaming, and chat-template rendering (isolated
//! behind the [`chat_template::ChatTemplate`] abstraction).

pub mod chat_template;
pub mod tokenizer;

pub use chat_template::{ChatMessage, ChatTemplate, ModelChatTemplate, QwenChatTemplate};
#[cfg(feature = "test-util")]
pub use tokenizer::testutil;
pub use tokenizer::{IncrementalDetokenizer, TokenizerError, TokenizerWrapper};

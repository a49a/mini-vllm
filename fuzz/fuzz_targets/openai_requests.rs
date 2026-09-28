#![no_main]

use libfuzzer_sys::fuzz_target;
use mini_vllm_server::openai::{ChatCompletionRequest, CompletionRequest};

fuzz_target!(|data: &[u8]| {
    if data.len() > 1024 * 1024 {
        return;
    }
    let _ = serde_json::from_slice::<CompletionRequest>(data);
    let _ = serde_json::from_slice::<ChatCompletionRequest>(data);
});

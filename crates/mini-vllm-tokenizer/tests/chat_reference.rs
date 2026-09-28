use mini_vllm_tokenizer::{ChatMessage, ChatTemplate, ModelChatTemplate, TokenizerWrapper};

#[test]
#[ignore = "requires local Qwen2.5 tokenizer; set MINI_VLLM_CHAT_MODEL"]
fn rendered_token_ids_match_transformers() {
    let path = std::env::var("MINI_VLLM_CHAT_MODEL").expect("set MINI_VLLM_CHAT_MODEL");
    let template = ModelChatTemplate::from_model_dir(&path).unwrap();
    let tokenizer = TokenizerWrapper::from_model_dir(&path).unwrap();
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/chat-reference.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let messages: Vec<_> = case["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| ChatMessage::new(m["role"].as_str().unwrap(), m["content"].as_str().unwrap()))
            .collect();
        let expected: Vec<u32> = serde_json::from_value(case["token_ids"].clone()).unwrap();
        let actual = tokenizer
            .encode(&template.render(&messages).unwrap(), false)
            .unwrap();
        assert_eq!(actual, expected, "{}", case["name"]);
    }
}

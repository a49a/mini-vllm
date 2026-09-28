# Chat template compatibility / 对话模板兼容性

`serve` reads `tokenizer_config.json` before loading weights and accepts only the exact Qwen2.5 Instruct template bundled in [the tokenizer crate](../crates/mini-vllm-tokenizer/src/templates/qwen2.5-instruct.jinja). This template comes from Qwen/Qwen2.5-0.5B-Instruct ([Apache-2.0 license and attribution](../crates/mini-vllm-tokenizer/src/templates/LICENSE-Qwen)). Missing, modified or named/multiple templates fail startup with an explicit error. Even whitespace changes are rejected intentionally: supporting a new variant requires reviewing its semantics and adding reference fixtures. `run` and `inspect` retain their existing model support.

`serve` 在加载权重前检查模型目录中的模板。仅接受已核对的 Qwen2.5 Instruct 模板；缺失、自定义或多模板配置都会明确报错。即使只修改模板空白也需要重新验证，避免静默套用错误提示格式。普通文本推理不受此限制。

For supported text messages (`system`, `user`, `assistant`), missing initial system messages get the official default: “You are Qwen, created by Alibaba Cloud. You are a helpful assistant.” An explicit system message, including an empty one, is preserved. Multi-turn content, late system messages, Unicode and literal special-token strings follow the reference template. Tools and multimodal messages remain unsupported. The low-level `QwenChatTemplate` is still a plain ChatML formatter for library fixtures; production serving uses `ModelChatTemplate`.

没有开头 system 消息时会补上官方默认提示；显式 system 消息（包括空字符串）会原样保留。多轮对话、Unicode 和正文中的特殊 token 与参考实现对齐。本次没有扩展工具调用或多模态支持。

The committed [oracle](../crates/mini-vllm-tokenizer/tests/fixtures/chat-reference.json) was generated independently with Transformers 4.46.3. Regular CI compares rendered text for five conversations. To regenerate and verify token IDs against local tokenizer assets (absolute model path required for Cargo tests):

```bash
python scripts/generate_chat_reference.py --model models/qwen2.5-0.5b-instruct --output crates/mini-vllm-tokenizer/tests/fixtures/chat-reference.json
MINI_VLLM_CHAT_MODEL="$(pwd)/models/qwen2.5-0.5b-instruct" cargo test -p mini-vllm-tokenizer --test chat_reference -- --ignored
```

Use the isolated reference environment described by [requirements-reference.txt](../scripts/requirements-reference.txt). No remote code or downloads are performed. Normal CI does not download the full tokenizer; the token-ID check is explicitly opt-in.

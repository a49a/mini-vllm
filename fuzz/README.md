# Parser fuzzing / 解析器模糊测试

The targets cover local Safetensors files and shard indexes (`weight_files`) and the two supported OpenAI request JSON types (`openai_requests`). Inputs are capped at 64 KiB for weights and 1 MiB for requests. Expected parser errors are accepted; panics, memory errors, and sanitizer findings are failures.

目标分别覆盖 Safetensors 文件及分片索引、两种 OpenAI 请求 JSON。正常解析错误不算失败；panic、内存错误和 sanitizer 报告算失败。

Install nightly Rust and [`cargo-fuzz`](https://rust-fuzz.github.io/book/cargo-fuzz/setup.html), then run from the repository root:

```bash
rustup toolchain install nightly --profile minimal
cargo install cargo-fuzz --locked
cargo +nightly fuzz run weight_files -- -max_total_time=60 -max_len=4096
cargo +nightly fuzz run openai_requests -- -max_total_time=60 -max_len=4096
```

Seed inputs live under `fuzz/corpus/`. Minimize and check in any crash input as a regression test before fixing it. The `fuzz/` crate has its own workspace so the normal Rust 1.87 build and CI suite do not require nightly or libFuzzer.

种子输入位于 `fuzz/corpus/`。修复发现的问题前，先将最小化输入加入回归测试。`fuzz/` 是独立工作区，不影响普通构建的 Rust 1.87 要求。

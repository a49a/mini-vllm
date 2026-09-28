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

## Continuous runs / 持续运行

[The fuzz workflow](../.github/workflows/fuzz.yml) runs both targets every Monday at 03:23 UTC and supports manual dispatch from GitHub Actions. Each target gets five minutes with ASan, a 10-second per-input timeout, and a 4 GiB RSS limit. The workflow always uploads logs, toolchain versions, evolved corpus and any crash inputs for 30 days. A failing target fails its job; the other target still runs. Artifact corpora are retained for investigation, but only reviewed, committed seeds are automatically reused next run.

工作流每周一北京时间 11:23 运行，也可手动触发。每个目标运行五分钟；崩溃、超时和内存限制触发均使任务失败。日志、语料和故障输入保存 30 天。新增语料需要审核并提交后，才会在后续运行中自动复用。

To turn a finding into a regression (replace `CRASH` with the downloaded input path):

```bash
cargo +nightly fuzz run weight_files CRASH
cargo +nightly fuzz tmin weight_files CRASH
# Copy the minimized input into the relevant crate's tests/fixtures directory.
# Add a normal stable-Rust test asserting the expected parser result.
cargo test --locked --workspace
```

Use `openai_requests` for request-parser findings. Commit the minimized input and failing regression first, then the fix; also add the reviewed seed to `fuzz/corpus/<target>/` with `git add -f` (generated corpus files are ignored). Never publish a security-sensitive crash before following [SECURITY.md](../SECURITY.md).

复现并最小化输入后，将样本和结果断言加入普通 Rust 测试，使后续每次 PR 都检查该回归；修复后再把已审核样本加入 fuzz 种子。不要把敏感漏洞输入直接公开。

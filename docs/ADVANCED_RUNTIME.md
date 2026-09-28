# Advanced runtime experiments / 进阶运行实验

[README](../README.md) · [Implementation / 实现说明](IMPLEMENTATION.md) · [中文教程](TUTORIAL.zh-CN.md) · [English tutorial](TUTORIAL.en.md)

## Defaults and validation / 默认值与校验

`serve --default-max-new-tokens 128` configures the single engine-owned default. Both completion endpoints query it when `max_tokens` is omitted or null. An explicit positive `max_tokens` takes precedence; zero is rejected. The default is still subject to prompt/context limits and is not silently shortened.

HTTP 不再另写一个 512 默认值。`EngineApi::default_max_new_tokens` 连接传输层与引擎配置。空 `messages`、不支持的 role、消息对象中的未知字段都会被拒绝。schema 错误由 Axum 返回客户端错误；参数语义错误返回 400，未知模型返回 404。

## Per-request deadlines / 单请求截止时间

```bash
cargo run --release -p mini-vllm-cli -- serve \
  --model models/qwen2.5-0.5b-instruct --device cpu \
  --queue-timeout-ms 5000 --request-timeout-ms 30000 \
  --default-max-new-tokens 128
```

Both limits default to zero (disabled). Queue timeout runs from successful engine submission until admission, including time inside the command channel. Request timeout runs from submission through prefill/decode completion; it includes queue time. They are server-wide policies applied independently to each request, not HTTP fields.

`queue_timeout_ms` 覆盖命令通道和等待队列；`request_timeout_ms` 还覆盖 prefill 与 decode。调度前和 forward 返回后都会检查。超时后不再发出成功 token/终态，释放 KV、前缀借用和请求许可。非流式 HTTP 返回 504；SSE 已发响应头时使用 `request_timeout` 类型的 error 帧。输出交付仍使用独立的 `output_drain_timeout_ms`。

A synchronous kernel cannot be interrupted. If a forward call crosses the deadline, its result is discarded at the next boundary, so this is not a hard wall-clock latency guarantee. Timeout tests cover commands, waiting sequences, active sequences and results arriving after the deadline.

## Block prefix trie / 按块索引的前缀树

[Source / 代码](../crates/mini-vllm-engine/src/prefix.rs)

Keys are `(parent node, exact token block)`. Hash lookup still compares the complete block, so hash collisions cannot cause incorrect reuse. Lookup walks prompt blocks until the first miss, always leaving at least one token for logits. This avoids scanning every cached prefix.

每个节点对应一个新 token 块，容量只记一次；例如 `[A,B]` 与 `[A,B,C]` 共用 A、B，不再按 2+3 个块收费。插入重复历史时，将请求中的相同历史替换为规范化的共享页。节点只保留自己的单块页引用，借用时沿祖先链重建活动请求的页表；保留页表的引用数量与唯一块数成正比，而不再随链深度平方增长。它仍不等于全局物理 arena。

Only unpinned leaves are evicted. An active borrower pins its terminal node; ancestors remain retained because they have descendants. If every eligible leaf is pinned, insertion stops rather than exceeding the budget. The active reservation subtracts only the prefix borrowed at admission; newly cached suffix pages remain conservatively charged to that request until retirement.

**Exercise / 练习：** Set a three-block retention pool. Insert `[A,B]`, then `[A,B,C]`: usage is three blocks. Pin C and attempt `[A,D]`: D cannot enter while C is pinned. Release C and retry: C can be evicted, leaving `[A,B]` and `[A,D]` in three unique blocks. The unit test `overlapping_prefixes_charge_each_block_once_and_preserve_borrowers` checks this sequence.

## Online softmax / 分页在线 softmax

[Source / 代码](../crates/mini-vllm-model/src/attention.rs)

The paged path keeps a running maximum `m`, denominator `l`, and weighted value numerator `o`, independently for each head/query. For scores `s` and values `v` from the next page:

```text
m_new = max(m, max(s))
a     = exp(m - m_new)
p     = exp(s - m_new)
l_new = a * l + sum(p)
o_new = a * o + p @ v
output = o / l
```

第一块初始化状态，之后每块对旧结果重新缩放，再累加新贡献。mask 使用全局 token 位置，因此后面的页即使对某个 query 完全不可见，也只贡献零。第一个页含 position 0，对所有合法 query 至少有一个可见位置，避免 `-inf - -inf`。

Reductions and accumulation use F32, even with F16/BF16 projections. Score workspace scales with `heads × query_length × page_size`, rather than the full cached context; there is also the running output accumulator. This portable implementation still launches multiple Candle operations and is not a fused GPU kernel. Lower temporary memory does not guarantee lower latency.

测试覆盖不满尾页、完整 prefill、单 token decode、未来页全部被 mask、较大 scores、独立 Transformers logits，以及真实 Qwen CPU/F32。CPU/F16 数值对齐也通过。无缓存 dense attention 保留为参考。

## JSONL teaching replay / JSONL 教学回放

```bash
mkdir -p artifacts
# Choose a new filename each run; an existing trace is never overwritten.
target/release/mini-vllm serve \
  --model models/qwen2.5-0.5b-instruct --device cpu \
  --trace-jsonl artifacts/session.jsonl --prefix-cache-tokens 512
# Send requests, then stop the server with Ctrl-C.
python3 scripts/trace_replay.py artifacts/session.jsonl --output artifacts/session.html
```

The JSONL schema is versioned (`schema_version=1`) and includes elapsed microseconds, request IDs, admission/shared-prefix counts, scheduled batch step numbers, phase, positions/page counts, completed chunks and retirement reasons. It contains no prompt text or generated text. Caller-supplied request IDs remain visible. Scheduling page counts describe the planned step; a failed/expired forward may not commit that step.

离线 HTML 可拖动时间滑块或自动播放，逐请求观察状态、同批 step、位置和共享前缀。页面无外部依赖，用文本节点显示 ID；JSON 中的 `<` 被转义，避免请求 ID 注入脚本。追踪最多尝试保留 100,000 条事件，然后尝试加入一条 `truncated` 标记；队列满时标记也可能被丢弃。引擎通过容量为 1,024 的有界队列交给独立线程写盘；队列满时直接丢弃事件，`/metrics` 的 `trace_events_dropped` 记录累计数量，关闭时在 JSONL 末尾写入 `trace_dropped` 计数。文件写入失败会告警并关闭追踪；推理继续。关闭追踪时不会构造 JSON payload。

Tracing queues events without blocking on disk I/O, but JSON construction and queue operations still add overhead: disable it for performance baselines. The writer flushes when the engine exits, so read the final JSONL after shutdown. The generated [sample replay](benchmarks/request-trace.html) and [source JSONL](benchmarks/request-trace.jsonl) come from real HTTP requests. File generation, schema/content and injection checks passed. Browser visual inspection was blocked by the browser tool's local-file URL policy and is not claimed as completed.

## Weight validation and minimum Rust / 权重校验与最低 Rust 版本

Before allocating device tensors, the loader bounds the index and Safetensors header sizes, checks shard filenames and manifest coverage, rejects duplicate names and invalid shapes/offsets, and validates the required Qwen2 tensor dimensions against `config.json`. `mini-vllm inspect` uses the same header checks without loading weight bytes.

加载器在分配设备张量前检查索引与头部大小、分片文件名和映射完整性，并拒绝重复名称、越界偏移与维度错误。工作区声明的最低版本是 Rust 1.87；CI 会用该版本对锁定依赖执行 `cargo check --locked --workspace --all-targets`。Rust 1.81 无法解析当前依赖清单，1.85 和 1.86 则无法编译锁定的 `yoke-derive`。

## Load and device matrices / 负载与设备矩阵

```bash
# Run against a fresh server per storage mode, with enough context/KV capacity.
python3 scripts/workload_matrix.py --requests 64 --concurrency 1 2 4 8 \
  --hit-rates 0 0.5 1 --long-repeat 16 --short-output 4 --long-output 32
python3 scripts/compare_kv.py --model models/qwen2.5-0.5b-instruct \
  --requests 8 --concurrency 2 --max-tokens 16 --trials 3
python3 scripts/validate_backends.py --devices cpu --dtypes f32 f16
# Explicit GPU test; failures remain failures in the report.
python3 scripts/validate_backends.py --devices metal --dtypes f32 f16 bf16
```

The load matrix alternates short/long prompts and output limits, varies concurrency and warmed-request fraction, and reports latency/TTFT/ITL P50/P95/P99 using linear interpolation over successful samples. Prompt nonces are deterministic from `--seed`; start a fresh server for repeatable cold-cache conditions. Requested reuse fraction is a request-level target; measured prefix-token reuse is reported separately. Failure counts are never mixed into successful-token throughput, and any failed request produces a nonzero exit code.

本轮 [混合负载原始数据](benchmarks/mixed-workload.json) 为并发 1/2 × 复用请求比例 0/0.5/1，共 6 组、24 个测量请求，全部成功，另有 12 个预热请求。这轮开启追踪用于回放，小样本 P95/P99 只演示统计管线，不构成生产尾延迟结论。无追踪的 [三模式对照](benchmarks/online-kv-comparison.md) 另行记录。

[Device report / 设备报告](benchmarks/backend-validation.json) records commands, exit codes and logs. CPU/F32 and CPU/F16 numerical/lifecycle tests pass, including a 24-request cancellation/disconnect/prefix stress sequence per supported combination. CPU/BF16 is explicitly unsupported by the current Candle CPU backend. Metal is built with its feature but no device is visible in this process; all Metal entries fail initialization. CUDA has not been executed here. The script returns nonzero for unsupported/unavailable combinations and does not relabel them as passed or skipped.

## Validation / 验收记录

137 regular Rust tests and 5 Python tests pass; fmt, clippy and documentation checks pass. Additional real Qwen CPU/F32 alignment, CPU/F32 and CPU/F16 backend numerical/lifecycle/stress runs pass. Real HTTP smoke confirms configured defaults, deadline 504 with KV reclamation, JSONL generation and clean shutdown. Browser visual QA and successful GPU execution remain unverified for the environmental reasons above.

The follow-up [long-prefix/trace report](benchmarks/prefix-trace-2026-09-28.md) records 16 successful measured requests, 97.7% prompt-token reuse, and zero trace drops. It also records the latest CPU passes and Metal initialization failures. Parser fuzz targets are documented in [fuzz/README.md](../fuzz/README.md). On 2026-09-28, both targets completed 2,000 libFuzzer runs with AddressSanitizer and no crash (cargo-fuzz 0.13.2, Rust 1.101.0-nightly; seeds 1366339630 for weight files and 1397009862 for requests, max input length 4096). These are smoke runs, not exhaustive validation.

# Inference efficiency and fair-load experiments / 推理效率与公平负载实验

## 中文

这轮实验把三个问题分开：模型是否在做无用计算、调度是否造成队头阻塞、负载工具是否掩盖过载。所有性能数字都需要同时读设备、样本数、失败数和测量边界。

### 选择性 logits

`CausalLm::forward_cached_selected` 仍对整个 batch 更新所有层的 KV，但只对指定序列的最后一个位置计算最终 RMSNorm 和 LM head。返回行顺序与指定索引顺序相同。空列表返回 `[0, vocab]`，Qwen2 跳过最终归一化与词表投影，executor 也不做 logits 的主机拷贝。中间 prefill 不采样，不推进随机数状态。

旧的 `forward_cached` 仍返回每个序列的 logits，可作为对照；第三方模型的默认 `forward_cached_selected` 实现会先调用旧接口，因此只有覆写接口的模型才能省去投影。非法索引在写入 KV 前拒绝。混合批次失败仍回滚缓存。

CPU 真实模型实验使用 256 个固定 token、16-token chunk、独立缓存、每种模式一次预热，三轮交替顺序测量；验证最终 logits，而非只比较生成文本：

```bash
cargo run --release --locked -p mini-vllm-model --example selected_logits -- models/qwen2.5-0.5b-instruct
```

### GQA 分页 attention

生产路径把 Q 从 `[kv_heads × group, query_tokens, head_dim]` 变为 `[kv_heads, group × query_tokens, head_dim]`，与原始 KV heads 相乘。这样无需将 K/V 复制 `group` 份；softmax 仍逐 query、跨页面归一化，累加仍使用 F32。完全位于第一个 query 之前的页面不再构造 causal mask；混合可见/不可见页面保留 mask。

原始扩展 KV heads 的实现保留在同一个算法的测试实例中，和 dense attention 一起做数值对照。微基准只测 CPU attention（含结果读取），不包含线性层、采样或 HTTP，不能代表全服务加速：

```bash
cargo test --release --locked -p mini-vllm-model paged_attention_timing -- --ignored --nocapture
```

### FIFO 与有限插队

默认 `--admission-lookahead 1` 保持 FIFO。设为 8 时，一次接纳最多查看队首起的 8 个候选，从中选择第一个能容纳的请求。容量检查只有在接纳时才扣减预算，仍按完整输出上限预留 KV。

`--admission-max-wait-ms 3000` 表示：扫描遇到等待至少 3 秒、但无法容纳的请求时立即停止，禁止其后请求继续插队。它是防饥饿屏障，不是保证 3 秒内完成接纳的 SLA。启用 lookahead 时该值必须大于零；扫描窗口之外的请求不会被检查。取消、队列超时和请求超时继续生效。

在两个新启动的服务上分别设置 lookahead 为 1 和 8，保持其余参数一致：

```bash
cargo build --release --locked -p mini-vllm-cli
./target/release/mini-vllm serve --model models/qwen2.5-0.5b-instruct \
  --device cpu --dtype f32 --port 8000 --max-model-len 512 \
  --max-num-seqs 4 --max-batch-tokens 64 --max-prefill-chunk-tokens 16 \
  --max-kv-tokens 128 --kv-block-size 8 --max-waiting-requests 32 \
  --admission-lookahead 8 --admission-max-wait-ms 3000 \
  --request-timeout-ms 30000 --shutdown-timeout-secs 3
```

在另一终端发送持续负载；本地运行时可添加 `--server-pid PID` 采样服务进程 RSS：

```bash
python3 scripts/sustained_load.py --arrival-rate 1 --duration 120 \
  --concurrency 16 --long-repeat 4 --short-output 4 --long-output 12 \
  --timeout 35 --output artifacts/sustained-lookahead.json
```

每个到达时刻都计入结果：客户端 16 个并发槽位全部占用时记为 `client_dropped`，不会无限堆积到客户端线程池。非成功 HTTP、SSE 错误或缺失成功终帧计为错误。`scheduled_latency_ms` 和 `scheduled_ttft_ms` 从计划到达时刻计时，包含发起延误；普通 `latency_ms` 从实际请求开始计时。P50/P95/P99 使用 nearest-rank，只统计成功请求，并显示样本数。失败率分母包含客户端丢弃。

吞吐分母包含完整到达窗口和最后的排空时间，token 数来自成功终帧 usage。JSON 同时保留逐请求结果和定期 `/metrics`、RSS 样本，可查看队列/KV/内存趋势。RSS 是主机进程内存，不是 VRAM。监控失败会记录错误并让命令非零退出；HTTP timeout 是 socket 操作超时，不是绝对请求 deadline。

持续实验应逐档提高到达率，并至少重复、交替模式运行；一次 120 秒测试不能证明没有内存泄漏，也不能证明一种调度策略在所有负载下更快。

### 后端验证

固定参考测试现在也覆盖选择性 logits：先不返回中间 chunk logits，再验证后续结果，同时覆盖 contiguous、paged 和共享 prefix fork。

```bash
python3 scripts/validate_backends.py --devices cpu metal --dtypes f32 f16 \
  --output artifacts/backend-validation.json
# 在可访问 CUDA 的机器上运行：
python3 scripts/validate_backends.py --devices cuda --dtypes f32 f16 \
  --output artifacts/backend-cuda.json
```

报告包括实现源文件与测试源文件的 SHA-256，避免未提交工作区只有 Git revision 而无法识别实际测试代码。GPU 工作流需要已注册、可信的对应 self-hosted runner；设备不可用仍是失败，不能写成通过。

## English

The engine now selects logits rows for sampling while still updating every KV layer. `forward_cached_selected` accepts sequence indices in output order; Qwen2 skips the final norm and vocabulary projection for an empty selection, and the executor avoids the host copy. Intermediate chunks do not advance the sampler RNG. The original interface and a default compatibility implementation remain available. Tests cover reordered rows, empty selections followed by decode, tied/untied heads, paged/contiguous caches, and invalid indices before mutation.

Paged GQA folds query groups into the query dimension instead of duplicating KV heads. Fully visible pages need no causal mask. The online softmax still accumulates in F32. The opt-in timing test above compares the expanded-head reference with the grouped implementation; its numbers cover only CPU attention, not end-to-end serving.

Admission defaults to FIFO (`--admission-lookahead 1`). A larger window allows the first fitting request within that window to proceed. An unfitting request older than `--admission-max-wait-ms` stops the scan, preventing further bypasses behind it. This age barrier is not a latency guarantee; active requests still need to finish or cancel to release capacity. KV horizon reservations, cancellation and deadlines remain in force.

Use the two-terminal commands above with fresh servers for FIFO and lookahead. The sustained workload alternates short/long prompts and outputs at a fixed arrival rate. Client concurrency is bounded; excess arrivals become explicit client drops, not hidden executor backlog. Scheduled latency includes dispatch lag. Percentiles use nearest rank over successful requests and include sample counts; failures and drops remain visible. Throughput uses successful terminal-usage tokens and wall time including drain. Periodic JSON metrics and optional local process RSS expose queue and memory trends; RSS is not device VRAM. Socket timeouts are per operation, not absolute deadlines.

Run several durations, arrival rates and alternating trials before drawing capacity or fairness conclusions. The backend validator now checks selective chunk logits against the fixed reference and fingerprints implementation sources as well as test files. GPU validation requires accessible hardware; a missing device is reported as a failure.

## Recorded results / 实测记录

See [the CPU experiments and hardware limitations](benchmarks/inference-efficiency-2026-09-28.md).

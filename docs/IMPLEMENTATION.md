# Implementation and validation / 实现与验证

[README](../README.md) · [中文教程](TUTORIAL.zh-CN.md) · [English tutorial](TUTORIAL.en.md)

This document records the implemented behavior after the runtime expansion.
本文记录调度、容量、输出交付与分页缓存完善后的实际行为。

See [advanced experiments / 进阶实验](ADVANCED_RUNTIME.md) for current defaults, deadlines, block-trie accounting, online softmax, JSONL replay and load/device matrices.

## Runtime controls / 运行参数

| CLI option | Default | Meaning / 含义 |
|---|---:|---|
| `--default-max-new-tokens` | 512 | HTTP default from engine configuration / 引擎配置中的 HTTP 默认生成长度 |
| `--queue-timeout-ms` | 0 | Submission to admission; 0 disables / 准入前超时，0 禁用 |
| `--request-timeout-ms` | 0 | Submission through generation; 0 disables / 请求总生成期限，0 禁用 |
| `--trace-jsonl` | none | New JSONL teaching trace file / 新建 JSONL 教学追踪文件 |
| `--max-batch-tokens` | 2048 | Combined input-token budget for one mixed model call / 每次混合调用的总输入 token 预算 |
| `--max-prefill-chunk-tokens` | 256 | Round-robin prefill slice cap / 轮转 prefill 分块上限 |
| `--trace-request` | false | Request scheduling/state traces / 请求调度与状态追踪 |
| `--shutdown-timeout-secs` | 30 | Drain deadline before cancellation / 排空超时后取消 |
| `--max-num-seqs` | 32 | Maximum admitted sequences / 活动序列上限 |
| `--max-waiting-requests` | 256 | Waiting queue limit / 等待队列上限 |
| `--output-drain-timeout-ms` | 1000 | Grace after releasing KV / 释放 KV 后的输出交付宽限期 |
| `--max-kv-tokens` | 32768 | Active suffix horizon reservations / 活动请求扣除命中前缀后的跨度预算 |
| `--kv-block-size` | 16 | Physical page and logical accounting block size / 物理页和记账块大小 |
| `--prefix-cache-tokens` | 0 | Additional bounded unique-prefix-block budget / 额外前缀保留预算，0 禁用 |
| `--contiguous-kv` | false | Select contiguous reference storage / 使用连续存储参考路径 |

`spawn_engine` is now fallible: Rust callers handle `Result<EngineHandle>`.
Zero capacities, invalid rotary dimensions, invalid token IDs, non-finite sampling
parameters, and arithmetic overflow are rejected. A request ID must be unique
while its request is pending, running, or draining.

`spawn_engine` 现在返回 Result。等待队列之外，还有一个以
`max_num_seqs + max_waiting_requests` 为上限的请求许可注册表，覆盖命令中、运行中、
等待中与交付中的请求。许可由 RAII 回收。取消设置原子标记，不需要命令队列空位；
引擎每轮检查运行和等待请求的断连与取消。正在执行的同步 forward 不会被强行中断。

## Scheduling and memory / 调度与内存

A model call can contain one-token decodes and multiple prefill chunks. The sum
of their input lengths cannot exceed the budget. Partial prefill does not sample;
the last prompt chunk produces the first generated token. When decodes exceed
the budget, selection rotates. If the budget is greater than one, pending
prefill receives at least one token of budget. With budget one, decode and prefill alternate. Prefill selection rotates and each chunk is capped by `max_prefill_chunk_tokens`.

物理分页由每层的页列表实现；attention 分页读取 K/V，跨页统一归一化 scores，
再累加输出。它没有融合 GPU kernel，也不宣称具有生产 vLLM 的性能。scores 现在逐页生成，用在线 softmax 累加，不再保留完整上下文 scores；这个实现主要用于学习与数值验证。

Pages are immutable once shared. An exclusive partial page is appended in place; a shared partial page is copied before writing. Pages allocate a full block, and readers see only the valid prefix;
full prefix pages are shared by reference count. The block trie stores only complete
block-aligned prefixes and leaves at least one prompt token to regenerate logits.
Evicting a snapshot or cancelling a sequence cannot invalidate another owner's pages.
The cache belongs to one model instance, so prefixes cannot cross model identities.

`max_kv_tokens` and `prefix_cache_tokens` are **separate** budgets. Active requests
reserve `ceil((prompt + max_new - shared_prefix) / block_size)` blocks. A reference-counted pin keeps the reused prefix charged to the retention pool until the active cache is released. Pinned nodes and ancestors with descendants cannot be evicted; insertion stops if no unpinned leaf can make room. Thus hits improve admission capacity without losing accounting when another request attempts eviction. Cold requests still must fit the active pool on their own. The block trie charges each retained canonical block once, including ancestors shared by overlapping prefixes. Allocator overhead,
weights, temporary scores, and short-lived COW buffers are outside token budgets.
The logical block manager is not a global physical-page arena.

## Output and metrics / 输出与指标

Model KV is released before output draining. A completed request keeps only its
sequence/output state and request permit during the configured grace period.
Timeout, receiver closure, or outbox overflow yields cancellation, never a
successful truncated answer. Request metadata remains until delivery completes
or expires, so total outstanding requests are bounded.

终态 SSE 帧携带 `usage.completion_tokens`；benchmark 用它统计成功请求实际生成
的 token，包括没有文字增量的 token。`[DONE]` 不能独立证明成功，脚本还要求成功
finish_reason、usage 和无 error 帧。失败请求单列，不计入成功吞吐；脚本有任意
失败时返回非零退出码。预热请求不进入正式计时。

- `tokens_per_second`: recent ten-second generated-token rate, including work
  later cancelled; `lifetime_tokens_per_second`: generated tokens / uptime.
- `requests_finishing`: output draining requests; `cached_prefix_tokens`: unique retained block
  accounting; `prefix_cache_hit_tokens`: reused prompt positions.
- `scheduled_tokens_total / model_steps_total`: combined scheduled input work;
  failed batch retries are additional execution work, not additional scheduling.
- Engine TTFT starts when sequence state is created. Client TTFT starts before
  the HTTP request; benchmark ITL times token events, even empty text increments.

## Independent reference / 独立参考数值

Normal `cargo test --workspace` uses a small Transformers-generated Qwen2 fixture.
It is independent of the Rust random-weight fixture and checks all vocabulary
logits, argmax, chunked prefill, contiguous KV, and physical pages with prefix sharing.
No Python environment or model download is needed to run that test.

真实 Qwen2.5-0.5B 的测试另行启用。仓库包含 8 个固定位置的参考数据：固定词表
采样点与每行 top-32 logits、argmax、配置和权重 SHA-256。这不是对所有词表
logits 的穷举比较；默认绝对误差阈值为 0.002。测试先校验权重摘要，避免把另一份
模型当成相同参考。它验证 CPU/F32，不证明其他 dtype/GPU 的数值误差。

From the repository root:

```bash
cargo test -p mini-vllm-model --test reference hf_tiny
MINI_VLLM_REFERENCE_MODEL=models/qwen2.5-0.5b-instruct \
  cargo test -p mini-vllm-model --test reference hf_real -- --ignored
```

To regenerate fixtures deliberately in an isolated environment with PyTorch:

```bash
python3 -m venv --system-site-packages .venv-reference
.venv-reference/bin/pip install -r scripts/requirements-reference.txt
.venv-reference/bin/python scripts/generate_reference.py --tiny \
  --output crates/mini-vllm-model/tests/fixtures/tiny-reference.json
.venv-reference/bin/python scripts/generate_reference.py \
  --model models/qwen2.5-0.5b-instruct \
  --output crates/mini-vllm-model/tests/fixtures/qwen2.5-0.5b-reference.json
```

The exporter uses seed 1234, CPU/F32, eager attention, no remote code, and records
Torch/Transformers versions. Regeneration is explicit: CI never overwrites the
reference with the implementation under test. Review fixture changes separately.
PyTorch must already be installed in the selected environment; version metadata
in each JSON fixture records the environment that produced it.

## Checks / 验收

```bash
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked --workspace
python3 -m unittest discover -s scripts/tests -v
python3 scripts/check_docs.py
```

GitHub Actions runs these checks on Linux and macOS. Real-weight alignment is
opt-in because weights are not committed; the tiny independent reference runs
in ordinary CI. HTTP integration tests exercise actual tokenizer → engine →
Qwen2 → JSON/SSE, in addition to protocol mocks and cancellation tests.

Remaining performance work includes fused GPU page attention and quantization. GPU verification requires hardware visible to the process; passing CPU tests is not GPU verification.

## Compatibility, failures and shutdown / 兼容性、失败与关闭

The loader rejects non-Qwen2 model types/architectures, RoPE scaling, enabled sliding windows, non-SiLU activation, custom head dimensions, and quantized configurations. A missing optional architecture field remains compatible with minimal Qwen2 fixtures. Inactive `sliding_window` metadata is allowed when `use_sliding_window=false`.

加载器不再“尽力尝试”其他架构。HTTP 请求中的 model 名称必须与 `/v1/models` 一致，否则返回 404。非法请求返回 400，异步等待队列过载返回 503，执行/解码错误返回 500。SSE 响应头发出后不能更改 HTTP 状态，改用带类型的 error 帧；不会伪装成成功终态。tokenizer 缺少模型生成的 ID 也会明确失败。

Rust callers use `request_shutdown(ShutdownMode::Drain, grace)` or `ShutdownMode::Cancel`, then blocking `join(timeout)` (use `spawn_blocking` in async code). Submission closes immediately. Drain processes already accepted commands; the deadline switches to cancellation. KV, prefix snapshots and request permits are released before the thread exits. `/health` returns 503 once the engine stops accepting. CLI SIGINT/SIGTERM starts drain, bounds HTTP drain as well, and joins the engine. A running synchronous device kernel cannot be preempted; join timeout is reported and the handle remains joinable.

关闭是显式生命周期：停止接新 → 排空或取消 → 回收资源 → join。不是通过“所有 handle 恰好被 drop”来猜测退出时机。`--shutdown-timeout-secs` 控制 CLI 排空宽限期；之后最多等待引擎 join 5 秒，正在执行的设备计算可能使 join 超时。

## Teaching trace / 教学追踪

```bash
RUST_LOG=mini_vllm_trace=info target/release/mini-vllm serve \
  --model models/qwen2.5-0.5b-instruct --device cpu --trace-request \
  --max-batch-tokens 8 --max-prefill-chunk-tokens 4
```

Each step logs request ID, shared batch step number, phase, input position/count, logical page count before/after, prefix hits and KV retirement. It does not log prompt text or generated text. Page counts describe each sequence's page table, not globally unique allocations. `kv_storage_allocations_total` separately counts persistent K/V storage tensor allocations, including copy-on-write; temporary attention tensors and index buffers are excluded.

同一个 step 编号的记录属于同一批。观察长 prompt 的位置逐块推进，以及短请求如何插入；命中前缀时，首个计算位置会从缓存边界开始。用 `prefix_admission_saves_capacity_and_pins_eviction`、`prefill_rotation_prevents_long_prompt_monopoly` 和 `partial_layer_failures_restore_all_cursors_on_both_storage_paths` 对照阅读测试。

## Device matrix and benchmark / 设备矩阵与性能实验

```bash
MINI_VLLM_TEST_DEVICE=cpu MINI_VLLM_TEST_DTYPE=f16 \
  cargo test -p mini-vllm-model --test reference backend_dtype_reference -- --ignored
MINI_VLLM_TEST_DEVICE=cpu MINI_VLLM_TEST_DTYPE=f16 \
  cargo test -p mini-vllm-engine --test engine backend_dtype_lifecycle -- --ignored
# GPU: replace cpu with metal/cuda and enable the corresponding feature:
MINI_VLLM_TEST_DEVICE=metal MINI_VLLM_TEST_DTYPE=f16 \
  cargo test -p mini-vllm-model --features metal --test reference backend_dtype_reference -- --ignored
# For engine GPU lifecycle tests, enable --features mini-vllm-model/metal (or /cuda).
python3 scripts/compare_kv.py --model models/qwen2.5-0.5b-instruct \
  --requests 8 --concurrency 2 --max-tokens 16 --trials 3
```

The opt-in numerical test checks all tiny-fixture logits, argmax, contiguous/paged chunked caches and prefix forks. Absolute tolerances are F32=0.002, F16=0.005, BF16=0.04. The lifecycle test also checks prefix reuse, cancellation, shutdown and reclamation. Missing devices are explicit failures, not silent skips.

本次环境实测：真实 Qwen CPU/F32 对齐、CPU/F16 小模型数值和生命周期测试通过。CPU/BF16 的 Candle matmul 不支持，加载阶段已改为明确拒绝；启用 Metal 编译特性后，本进程仍无法发现 Metal 设备，GPU 测试未通过设备初始化，不能宣称 GPU 已验证。CUDA 未执行。

[CPU smoke report / CPU 实测报告](benchmarks/cpu-smoke.md) and [raw trials / 原始数据](benchmarks/cpu-smoke.json) record three rotated trials per mode, two requests per trial and four generated tokens per request. All 18 measured requests succeeded (plus nine warmups). RSS is sampled process memory including load/warmup, not GPU VRAM. Small workloads and shared-host noise limit performance conclusions. The script starts a fresh server per trial, separates warmup, retains exact commands and errors, and exits nonzero on any failed request. Each mode uses the same prompt; the prefix mode intentionally measures a warm reusable prefix.

本轮验收：125 项常规 Rust 测试、5 项 Python 测试通过；fmt、clippy 和文档链接检查通过。额外执行的真实 Qwen CPU/F32 对齐、CPU/F16 数值与生命周期测试，以及真实模型 HTTP/trace/SIGTERM 排空 smoke 均通过。远端 CI 和 GPU 成功路径尚未执行。

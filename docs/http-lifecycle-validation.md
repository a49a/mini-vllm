# HTTP admission and lifecycle testing / HTTP 准入与生命周期验证

## Bounded preprocessing / 有界预处理

The two generation endpoints now take admission **before reading JSON bodies**. Defaults are two CPU workers, sixteen extra admitted requests and a ten-second total preprocessing deadline. The total eighteen slots cover requests reading bodies, waiting for a worker and executing CPU work. `/health`, `/metrics` and `/v1/models` do not take these slots.

两个生成端点在读取 JSON 前占用准入名额，默认最多两个 CPU 任务、额外十六个请求，总计十八个名额。总超时覆盖请求体读取、排队和预处理结果等待。健康检查和指标查询不受这些名额限制。

```bash
cargo run -p mini-vllm-cli -- serve --model models/qwen2.5-0.5b-instruct --device cpu --dtype f32 --preprocessing-workers 2 --preprocessing-waiting 16 --preprocessing-timeout-ms 10000
```

Chat rendering and tokenization run on Tokio's blocking pool. A full admission pool returns 503 immediately. A preprocessing deadline returns 408. A running CPU job cannot be forcibly cancelled: it retains both its worker and admission permits until the closure exits, even when its HTTP caller times out or disconnects. Work that has not started is discarded on expiry. The engine's generation queue and generation deadlines remain separate.

模板渲染与分词转移到阻塞线程池。满载立即返回 503；超过预处理预算返回 408。已经执行的 CPU 任务不会因客户端离开而释放名额，避免反复断开连接制造超额后台任务。尚未开始的过期任务不再执行。该限制与引擎推理队列分开计数。

This bounds admitted request bodies (each still limited to 1 MiB) and CPU jobs, not TCP connections or all web-server memory. JSON deserialization itself remains on the async task and is bounded by the body size and admission count. The deadline is cooperative for that synchronous parse; it cannot interrupt an individual serde call.

这里限制的是已准入请求体和 CPU 任务，不是整个服务器的连接数或内存总量。JSON 解析仍在异步任务中执行，受请求体大小和准入数量约束；单次同步 serde 解析不能被计时器强行中断。

See [service validation](service-validation.md) for preprocessing metrics, bounded runtime teardown and real CLI process tests.

## Consistent errors / 一致的错误响应

Extractor and business errors use `{"error":{"message":"...","type":"..."}}` with `application/json`. HTTP statuses are preserved:

| Condition / 条件 | Status | Type |
|---|---:|---|
| Malformed JSON / JSON 语法错误 | 400 | invalid_request_error |
| Missing/unknown fields / 缺失或未知字段 | 422 | invalid_request_error |
| Body over 1 MiB / 请求体过大 | 413 | invalid_request_error |
| Missing/wrong Content-Type / 类型错误 | 415 | invalid_request_error |
| Preprocessing admission full / 预处理容量已满 | 503 | preprocessing_unavailable |
| Preprocessing deadline / 预处理超时 | 408 | request_timeout |
| Blocking worker panic / 工作线程 panic | 500 | internal_error |

Both generation endpoints test these extractor errors. Tests also hold a body open while checking that health remains responsive, and block CPU jobs while cancelling or timing out callers to verify permits are not released early.

## Generated lifecycle tests / 生成式生命周期测试

[The state-machine test](../crates/mini-vllm-engine/tests/state_machine.rs) generates 24 sequences from a reproducible seed. Operations include submitting repeated/disjoint prefixes, polling slowly, cancelling, disconnecting, pausing and shutting down repeatedly. It uses the real engine with committed tiny-model weights. Invariants include nonnegative gauges, KV use within capacity, unique/last observed terminal events, exactly-once retirement accounting and zero retained resources after shutdown. Slow consumers may legally close without receiving a terminal event; the oracle preserves that contract.

生成测试使用真实引擎和仓库内的小模型。它检查资源预算、终止事件顺序、请求只结算一次，以及关闭后的资源归还。慢消费者可能按现有协议直接关闭，因此测试不会错误地要求每个已断开的客户端都收到终止事件。

Normal tests use a fixed seed; CI adds a seed derived from its run ID. On an invariant failure, bounded deletion shrinking retains a smaller sequence only when the same failure reproduces three times. It saves `original.json` and `minimized.json` under `target/lifecycle-failures/`; CI uploads them on failure. This is best-effort reduction, not proof of a globally minimal sequence, and OS scheduling is not deterministic.

```bash
MINI_VLLM_LIFECYCLE_SEED=12345 cargo test -p mini-vllm-engine --test state_machine
MINI_VLLM_LIFECYCLE_CASE="$(pwd)/target/lifecycle-failures/minimized.json" cargo test -p mini-vllm-engine --test state_machine generated_lifecycle_sequences_preserve_invariants
```

发现失败后会自动尝试删除操作并反复复现，保留原始和缩减样本供回归测试。线程调度仍受操作系统影响；缩减不保证全局最小。修复时应将稳定复现的样本转为永久回归测试。

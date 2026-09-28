# Service shutdown and reference validation / 服务关闭与参考验证

## Shutdown ownership / 关闭的资源归属

The CLI uses `BoundedRuntime` so success, error returns and unwinding all call `Runtime::shutdown_timeout(250 ms)`. Engine cancellation and its five-second join happen outside Tokio, even after a bind/transport error, so a saturated blocking pool cannot delay the join task. Runtime creation precedes starting the engine. Running CPU closures are not forcibly stopped; if still blocked after the runtime budget, they are detached and the CLI can exit.

CLI 的 runtime 在正常返回、错误返回和 panic 展开时都最多等待 250 毫秒。引擎取消与五秒 join 在 Tokio 外执行；端口绑定或传输失败也走清理路径。已经运行的 CPU 任务不能强行终止，超过 runtime 预算后不再等待它们。

For a running service, the configured HTTP drain grace is followed by at most five seconds of engine join and 250 ms of runtime teardown, plus scheduling overhead. This bounds application-level waits; it is not a hard real-time guarantee or a model-loading deadline. A blocking-task regression is tested in subprocesses on success, error and panic paths, with a parent watchdog that kills hung children.

启动完成后的关闭预算为 HTTP grace、五秒引擎 join 和 250 毫秒 runtime 清理，另有线程调度开销；这不是实时系统保证，也不限制模型加载时间。故障测试放在子进程中，父进程监控退出期限，避免测试本身永久挂起。

## Preprocessing metrics / 预处理指标

`GET /metrics` keeps existing engine fields and adds a `preprocessing` object:

| Field | Meaning / 含义 |
|---|---|
| `admitted_total`, `rejected_total` | Accepted preprocessing tickets / capacity rejections；准入与容量拒绝次数 |
| `timeouts_total` | Request preprocessing deadlines exceeded；预处理超时次数 |
| `callers_cancelled_total` | Dropped caller futures before returning a preprocessing result；预处理调用方中途取消次数 |
| `job_errors_total`, `job_panics_total` | Work returned an error / panicked；任务失败或 panic 次数 |
| `detached_jobs`, `detached_jobs_total` | Current / cumulative jobs retained after the caller left；调用方离开后仍占用资源的任务数 |
| `body_read`, `queue_wait`, `execution` | Per-stage gauges and timing aggregates；各阶段在途量与耗时统计 |

Each stage contains `in_flight`, `completed`, `total_us`, and `max_us`. Mean elapsed time is `total_us / completed` when the count is nonzero. Completed stages include errors and cancellations. `execution` measures worker-permit occupancy, including any wait inside Tokio's blocking pool; it is not CPU profiling time. Gauges return to zero when resources actually finish, not merely when an HTTP deadline expires. Snapshot fields for preprocessing are read under one lock; engine and preprocessing snapshots are not one atomic transaction.

各阶段完成数包括失败或取消的阶段。`execution` 统计持有 worker 名额的墙钟时间，包含阻塞线程池排队，不是纯 CPU 时间。超时不会提前清零仍运行任务的指标。预处理快照内部一致，但它和引擎快照不是同一事务。

## Process tests / 真实进程测试

```bash
cargo test -p mini-vllm-cli
```

On Linux/macOS the suite launches the actual CLI binary with real tiny-model weights and a complete 64-token fixture tokenizer. It tests slow body uploads, responsive health/metrics, mid-SSE TCP disconnect and resource reclamation, a successful request after cancellation, SIGTERM with a nonreading peer, and bind errors. All child processes have watchdogs and cleanup guards. No network model download is needed. It does not claim to exhaust every OS socket-buffer saturation pattern.

这套测试运行真实 CLI 和真实小模型，不使用 mock 引擎。它验证跨 HTTP、预处理、引擎、runtime 的完整路径；普通 CI 在 Linux/macOS 执行，Windows 的信号测试暂不运行。

## Pinned references / 固定版本参考验证

[The manifest](../scripts/reference-model.json) fixes Qwen/Qwen2.5-0.5B-Instruct at revision `7ae557604adf67be50417f59c2c2f167def9a775`, with byte sizes and SHA-256 for all five required files. The weight digest agrees with that revision's LFS object, and the small files were checked against its Git blob identities. The [workflow](../.github/workflows/reference.yml) runs manually or Tuesday at 04:41 UTC (12:41 Beijing), caches model files by manifest hash, and preserves reports for 30 days.

```bash
# Existing local model: no downloads allowed.
python3 scripts/validate_reference.py --model models/qwen2.5-0.5b-instruct --output /tmp/reference.json
# Explicitly permit downloads into a separate cache.
python3 scripts/validate_reference.py --download --output /tmp/reference.json
```

Every run checks asset hashes, even after a cache hit. Downloads go to temporary files and replace cache entries only after validation; mismatched bytes fail the run. Then it executes the ignored real-tokenizer token-ID test and the real-model logits/cache-path test against independently generated, committed Transformers oracles. It requires each command to report one passing test; zero matched tests is a failure. Downloads execute no remote model code. This workflow does not require Python Transformers or PyTorch because the oracle outputs are committed; regenerating the oracles remains a separate, reviewed operation.

每次运行都验 hash，缓存命中也不例外。下载文件通过校验后才替换缓存；损坏文件、执行失败、超时或匹配到零个测试都会失败。参考输出沿用独立生成并提交的 Transformers fixture，不在验证时自动重新生成，避免同时修改被测实现和答案。

The report records revision/dirty state, Rust version, expected/actual hashes, oracle hashes, commands, elapsed time and full output. [Local reference results](benchmarks/pinned-reference-2026-09-28.json) passed both tests against pre-commit working-tree code. GPU validation remains separate in [the GPU guide](gpu-validation.md).

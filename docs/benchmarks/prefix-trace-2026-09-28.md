# Long-prefix and trace validation / 长前缀与追踪验证

2026-09-28, Apple M1 Pro host, CPU/F32, Qwen2.5-0.5B-Instruct. Each row starts a fresh server, warms up once, then measures four requests at concurrency two with four output tokens. The repeated prompt contains 221 tokens; 216 tokens (27 blocks of eight) are reused per measured request. Raw commands and binary/model hashes are in [the JSON report](prefix-trace-2026-09-28.json).

| Trace | Trial | Success | tok/s | TTFT median ms | Peak process RSS MiB | Prefix-hit tokens | Computed prompt tokens | Trace drops |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| off | 0 | 4/4 | 12.68 | 270.25 | 1977.20 | 864 | 20 | 0 |
| off | 1 | 4/4 | 13.12 | 261.19 | 1990.64 | 864 | 20 | 0 |
| on | 0 | 4/4 | 14.88 | 186.65 | 1980.17 | 864 | 20 | 0 |
| on | 1 | 4/4 | 15.64 | 180.26 | 1998.73 | 864 | 20 | 0 |

All 16 measured requests and four warmups succeeded. The measured prompt-token reuse ratio is 216/221 = 97.7%. RSS includes model weights, warmup and allocator overhead; it does not isolate page-table metadata and is not device VRAM.

The trace-off trials ran before the trace-on trials. This is a small smoke comparison on a shared host: the lower observed latency with tracing enabled is not evidence that tracing improves performance, nor a reliable estimate of tracing overhead. A performance claim needs interleaved repeated trials and larger workloads.

全部 16 个测量请求及 4 个预热请求成功。开启追踪的两次运行均无丢弃事件，并在关闭后成功解析全部 JSONL。小样本与执行顺序会影响结果，不能据此宣称追踪更快。

## Reproduction / 复现

```bash
python3 scripts/compare_kv.py --model models/qwen2.5-0.5b-instruct \
  --modes prefix --prompt-repeat 10 --requests 4 --concurrency 2 \
  --max-tokens 4 --trials 2 --output /tmp/prefix-trace-off.json
python3 scripts/compare_kv.py --model models/qwen2.5-0.5b-instruct \
  --modes prefix --prompt-repeat 10 --requests 4 --concurrency 2 \
  --max-tokens 4 --trials 2 --trace --output /tmp/prefix-trace-on.json
```

## Backend verification / 后端验证

[The backend report](backend-validation-2026-09-28.json) records all commands and failures. CPU/F32 and CPU/F16 pass the numerical and lifecycle tests. Metal/F32 and Metal/F16 fail initialization because no Metal device is visible to this process, despite the host reporting an M1 Pro GPU. CUDA was not executed because this host has no CUDA device/toolchain. Successful GPU execution remains unverified.

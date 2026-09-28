# KV comparison / KV 对照实验

Platform: `macOS-14.8.9-arm64-arm-64bit`. Device: cpu; dtype: f32.

Warmup excluded from timing/allocations. RSS includes model loading and warmup; it is not VRAM. Allocation counts include persistent KV tensors only. Small samples are smoke measurements, not capacity claims.

| Mode | Trace | Trial | Success | tok/s | TTFT median ms | Peak RSS MiB | KV allocations | Computed prompt tokens | Prefix hits | Trace drops |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prefix | off | 0 | 4/4 | 14.80 | 187.0 | 1982.17 | 192 | 20 | 864 | 0 |
| prefix | on | 0 | 4/4 | 13.59 | 189.14 | 2040.44 | 192 | 20 | 864 | 0 |
| prefix | on | 1 | 4/4 | 12.73 | 275.23 | 2096.86 | 192 | 20 | 864 | 0 |
| prefix | off | 1 | 4/4 | 14.73 | 196.69 | 1997.47 | 192 | 20 | 864 | 0 |
| prefix | off | 2 | 4/4 | 13.88 | 189.32 | 2003.98 | 192 | 20 | 864 | 0 |
| prefix | on | 2 | 4/4 | 12.48 | 274.4 | 2069.88 | 192 | 20 | 864 | 0 |

Percentiles use nearest rank; request counts are small. Raw request latency, TTFT and token-event intervals are preserved in JSON.

| Mode | Trace | Trial | N | Latency P50 ms | P95 ms | P99 ms |
|---|---|---:|---:|---:|---:|---:|
| prefix | False | 0 | 4 | 530.37 | 545.49 | 545.49 |
| prefix | True | 0 | 4 | 534.22 | 642.79 | 642.79 |
| prefix | True | 1 | 4 | 623.28 | 634.66 | 634.66 |
| prefix | False | 1 | 4 | 526.28 | 559.86 | 559.86 |
| prefix | False | 2 | 4 | 542.13 | 610.19 | 610.19 |
| prefix | True | 2 | 4 | 619.47 | 662.85 | 662.85 |

## Interpretation / 结果解读

Three paired trials, each with a fresh process and one warmup, produced 24 successful measured requests and zero failures. Each prompt had 221 tokens; 216 were reused per request (864 cache-hit tokens per four-request trial). Trace runs reported zero queue drops and zero writer errors before shutdown. All six server processes shut down successfully.

本次三轮交错实验共完成 24 个测量请求。三个配对中开启 trace 的吞吐都较低，但样本量较小，仍不足以给出稳定开销比例或统计显著性结论。P95/P99 在每轮仅四个样本时都是最大值，不能视作线上尾延迟估计。

This is CPU/F32 on one M1 Pro host, with concurrency 2 and four generated tokens per request. It does not validate Metal/CUDA or production capacity. RSS includes model loading and warmup; do not infer prefix metadata size from the RSS difference. See [raw per-request results](paired-trace-2026-09-28.json) for hashes, settings and nearest-rank latency distributions. Trace paths in JSON refer to the original local run and are not portable artifacts.

## Independent metadata experiment / 独立元数据实验

The [standalone example](../../crates/mini-vllm-engine/examples/prefix_metadata.rs) uses one CPU layer with head dimension 1 and block size 8. No model weights are loaded. It asserts zero allocations during 1,000 longest-prefix lookups at each size, linear retained key/page-reference counts, and successful replacement by a disjoint prefix.

| Blocks | Key tokens | Page references | Lookup allocations / 1,000 | Insert µs | Replace µs | 1,000 lookups µs |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 128 | 16 | 0 | 17 | 21 | 1157 |
| 64 | 512 | 64 | 0 | 26 | 38 | 4602 |
| 256 | 2048 | 256 | 0 | 121 | 162 | 18769 |
| 1024 | 8192 | 1024 | 0 | 405 | 808 | 87115 |

Counts are logical retained metadata; they are not measured heap bytes. Durations are single-run observations, not a before/after speedup claim. Lookup still traverses every matching block. [Raw metadata results](prefix-metadata-2026-09-28.json).

可确认的是查找没有临时键分配，保留的键与页引用随块数线性增长；不能据此宣称查找为常数时间，或给出堆内存节省比例。

Reproduction commands and limitations: [runtime guide](../runtime-hardening.md).

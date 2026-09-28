# CPU efficiency measurements / CPU 效率实测

Host: macOS 14.8.9 arm64, CPU/F32. Base revision: `19c78aa`, with uncommitted implementation changes identified by the source hashes in the raw reports. These are local experiments, not GPU or production capacity claims.

## Selective prefill logits

Qwen2.5-0.5B-Instruct, 256 fixed tokens, chunk size 16, paged KV, one warmup per mode, three alternating-order trials. Every trial starts with a fresh cache; both paths use the same grouped attention implementation. The baseline requests logits for every chunk; the selected path requests only the final chunk. Timings include the host reads that the executor would perform.

| Mode | LM head rows | Median prefill time |
|---|---:|---:|
| All chunk logits | 16 | 2539.68 ms |
| Final chunk only | 1 | 2148.76 ms |

The selected path reduced median elapsed time by **15.4%** in this experiment. Final logits matched exactly in all three trials (maximum absolute error 0). This is fixed-token prefill, not generated-text throughput.

## Paged GQA attention

14 query heads, 2 KV heads, head dimension 64, page size 16, CPU/F32. Five alternating-order trials per shape, ten calls per trial, warmups excluded. Timing includes reading the output; neither LM head nor HTTP is included. Both implementations use the same online softmax, and the old expanded-head path remains available to the test.

| Context tokens | Query tokens | Expanded KV median | Grouped KV median | Reduction |
|---:|---:|---:|---:|---:|
| 256 | 1 | 0.383 ms | 0.301 ms | 21.5% |
| 256 | 16 | 1.296 ms | 1.056 ms | 18.5% |
| 1024 | 1 | 1.429 ms | 1.173 ms | 17.9% |
| 1024 | 16 | 5.366 ms | 4.136 ms | 22.9% |

All four shapes passed the numeric comparison (absolute tolerance 1e-4). These small CPU kernel measurements do not establish an end-to-end speedup or GPU performance.

[Raw microbenchmark measurements, commands and hashes](efficiency-2026-09-28.json).

## Sustained FIFO / lookahead workload

Each mode used a fresh CPU/F32 Qwen2.5-0.5B server, 120 seconds at one arrival per second, alternating short/long prompts and 4/12 output tokens. The client allowed at most 16 in-flight requests; the server allowed 4 active sequences, a 128-token KV budget and 16-token prefill chunks. Lookahead used an 8-candidate window and a 3000 ms age barrier. FIFO was run first, lookahead second, once each. Commands, binary/source hashes, per-request records and telemetry are retained in the JSON reports.

| Mode | Success / scheduled | Errors / client drops | Scheduled latency P95 / P99 | Scheduled TTFT P95 / P99 | tok/s including drain |
|---|---:|---:|---:|---:|---:|
| FIFO | 120 / 120 | 0 / 0 | 2040.94 / 3969.75 ms | 1143.73 / 2583.84 ms | 7.967 |
| Lookahead 8 | 120 / 120 | 0 / 0 | 1593.72 / 1652.78 ms | 519.96 / 593.91 ms | 7.966 |

Percentiles use nearest rank over successful requests, measured from scheduled arrival. The equal throughput is limited by the offered rate, so it is not a capacity measurement. The lower latency in the second run cannot be attributed solely to scheduling: there was only one run per mode and order/host-load effects were not controlled by repetitions. FIFO remains the default. The deterministic scheduler tests separately demonstrate bypass, the age barrier, slot limits and KV reservation/reclamation.

| Mode | Telemetry samples | Max sampled waiting | RSS first / last / peak MiB | Final running / waiting / used KV blocks |
|---|---:|---:|---:|---:|
| FIFO | 120 | 2 | 1994.97 / 1916.56 / 1997.22 | 0 / 0 / 0 |
| Lookahead 8 | 121 | 0 | 1970.56 / 1943.69 / 1970.56 | 0 / 0 / 0 |

Telemetry had no sampling failures. RSS excludes dedicated device memory; sampled queue maxima can miss events between samples. Both processes shut down cleanly. These short runs show reclaimed active KV and do not establish the absence of long-term memory leaks.

[Raw FIFO requests and telemetry](sustained-fifo-2026-09-28.json) · [Raw lookahead requests and telemetry](sustained-lookahead-2026-09-28.json).

## GPU evidence

CPU/F32 and CPU/F16 numerical-reference and lifecycle checks passed, including selective prefill logits, paged/contiguous KV and prefix sharing. Metal/F32 and Metal/F16 were attempted with the Metal feature enabled; all four checks failed because the process could not create a Metal device. CUDA was not run: no accessible CUDA device/toolchain was available. The repository runner inventory returned **zero self-hosted runners**, so there was no registered remote GPU runner to use either.

[Raw backend report](backend-improvements-2026-09-28.json). GPU numerical and lifecycle correctness remains unverified. Reproduction instructions are in [the bilingual guide](../inference-efficiency.md).

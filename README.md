# mini-vllm-rs

A small, educational **LLM inference engine in Rust**, inspired by vLLM's
architecture and serving patterns. It is not a production vLLM replacement —
the goal is to implement the core ideas of an LLM serving stack deeply enough
to understand how modern inference systems work.

> 中文设计文档见 [`DESIGN.md`](./DESIGN.md) ·
> 原始规格见 [`MINI_VLLM_RS_DESIGN.md`](./MINI_VLLM_RS_DESIGN.md)

## Learn the implementation

Follow the request lifecycle, work through tensor shapes and KV memory accounting,
and run exercises based on the project's correctness tests:

- [中文教学文档：从零读懂 mini-vllm-rs](docs/TUTORIAL.zh-CN.md)
- [English tutorial: Understanding mini-vllm-rs](docs/TUTORIAL.en.md)

Both editions cover scheduling, sampling, streaming, the five correctness fixes,
and extension exercises. They describe current code behavior and distinguish it
from planned features in the design documents.

## What it is / what it is not

**Is:** a working end-to-end runtime — model loading (Safetensors), HF
tokenizer, Qwen2-style decoder-only forward pass (RMSNorm / RoPE / GQA /
SwiGLU), physical KV pages with bounded prefix reuse, seeded sampling pipeline,
continuous batching with chunked prefill and mixed batches, streaming engine, OpenAI-compatible HTTP API
(SSE), metrics, and a CLI.

**Is not:** distributed, quantized, or performance-competitive. No tensor/pipeline
parallelism, no speculative decoding, no custom GPU
kernels (see the roadmap in [`DESIGN.md`](./DESIGN.md#13-已知限制与路线图)).

## Supported models & devices

- **Architecture:** Qwen2 / Qwen2.5 decoder-only (`Qwen2ForCausalLM`),
  tied or untied LM head, GQA. Test model:
  [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
- **Weights:** `model.safetensors` or sharded + `model.safetensors.index.json`.
- **Devices:** CPU (default compute dtype F32), Metal (default on macOS
  builds; gracefully degrades when no GPU is visible), CUDA (`--features cuda`).

## Installation

Rust 1.87 or newer is required. CI checks the declared minimum version against the locked dependencies, then runs the full suite on stable Rust.

```bash
cargo build --release -p mini-vllm-cli        # Metal on macOS
cargo build --release -p mini-vllm-cli --features cuda   # CUDA builds
```

Model preparation (Qwen2/2.5 weights; `serve` validates the bundled Qwen2.5 Instruct chat template — see [template compatibility](docs/chat-templates.md)):

```bash
mkdir -p models/qwen2.5-0.5b-instruct && cd models/qwen2.5-0.5b-instruct
for f in config.json tokenizer.json tokenizer_config.json model.safetensors; do
  curl -LO "https://hf-mirror.com/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/$f"
done
```

## CLI

```bash
# Inspect a model directory (headers only — no weights are loaded)
mini-vllm inspect --model ./models/qwen2.5-0.5b-instruct --device auto

# Generate (streams to stdout; goes through the full engine)
mini-vllm generate --model ./models/qwen2.5-0.5b-instruct \
  --prompt "The capital of France is" \
  --max-new-tokens 32 --temperature 0

# Serve an OpenAI-compatible API
mini-vllm serve --model ./models/qwen2.5-0.5b-instruct \
  --host 127.0.0.1 --port 8000 \
  --device auto --dtype auto \
  --max-num-seqs 32 --max-batch-tokens 2048 --max-kv-tokens 32768
```

## OpenAI API examples

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-0.5b-instruct",
    "messages": [
      {"role": "system", "content": "You are a concise assistant."},
      {"role": "user", "content": "What is ownership in Rust?"}
    ],
    "temperature": 0.7, "max_tokens": 128, "stream": false
  }'
```

Streaming (`"stream": true`) emits `chat.completion.chunk` /
`text_completion` SSE events and terminates with `data: [DONE]`. Disconnecting
a stream cancels the underlying generation and releases its KV memory.

Also available: `GET /v1/models`, `GET /health`, `GET /metrics` (JSON
snapshot of TTFT / inter-token latency / batch sizes / KV usage).

Supported request fields: `model, prompt, messages, max_tokens, temperature,
top_p, top_k, repetition_penalty, seed, stream, stop`. Unknown fields are
rejected explicitly.

## Architecture

```text
HTTP (axum) ──commands──▶ engine thread ──▶ scheduler (FIFO + KV admission)
                              │                    │
                              │          continuous batching:
                              │          retire → admit → prefill → decode
                              ▼
                 mixed prefill+decode batch (linear layers batched,
                 attention per sequence over per-sequence KV caches)
                              ▼
                 bounded per-request event streams → JSON / SSE
```

Key correctness gates (all in `cargo test --workspace`):

- cached decode ≡ full-sequence recomputation (KV equivalence, §52.4)
- batched greedy decode ≡ independent decoding (batching equivalence at the
  model level; through the engine, per-mode determinism — batched GEMM float
  ordering can flip argmax on near-ties)
- block manager alloc/release/reuse with no leaks; cancellation (client
  disconnect *and* slow consumer) releases KV; engine E2E tests cover
  length/stop conditions, mid-flight cancel, and slow-consumer backpressure
- seeded, deterministic sampling (top-k keeps exactly k, ties broken by id)

## Scheduling, memory, and delivery

- `--max-batch-tokens` is a **combined** prefill/decode token budget. Long
  prompts advance in chunks; prefill and decode share one model call.
- `--max-waiting-requests` bounds the waiting queue. A separate in-flight
  registry bounds pending commands, running requests, and output draining;
  duplicate request IDs are rejected. Cancellation bypasses command capacity.
- Physical KV pages are the default. Attention reads pages directly, normalizes
  scores across pages, and accumulates value contributions. This portable
  implementation is not a fused GPU PagedAttention kernel.
- `--contiguous-kv` selects the contiguous correctness reference.
- `--prefix-cache-tokens 1024` enables a bounded block trie of shared prefix pages.
  It is an **additional memory budget** beyond `--max-kv-tokens`; active requests
  reserve only the remaining suffix on hits. Active borrowers pin retained entries.
  Exclusive tail pages append in place; shared tails use copy-on-write.
- KV is released before completed output is drained. `--output-drain-timeout-ms`
  (default 1000) gives readers a bounded grace period. Timeout/overflow never
  produces a successful truncated response.
- Terminal SSE frames include exact `usage`. An error frame or missing successful
  finish makes a stream unsuccessful even if it subsequently contains `[DONE]`.

- `--max-prefill-chunk-tokens 256` caps each round-robin prefill slice.
- `--trace-request` logs positions, shared batch step IDs, pages and prefix hits.
- `--shutdown-timeout-secs 30` bounds signal-triggered drain before cancellation.
- Unknown model names return 404; overload returns 503 (typed errors after SSE headers).
  Unsupported model semantics and CPU/BF16 are rejected explicitly.

- `--default-max-new-tokens` supplies omitted HTTP limits from engine configuration.
- `--queue-timeout-ms` and `--request-timeout-ms` expire requests at scheduling boundaries (0 disables).
- `--trace-jsonl session.jsonl` records a replayable trace; `scripts/trace_replay.py` creates an offline HTML timeline.
- The block prefix trie deduplicates retained blocks; paged attention uses online softmax instead of concatenating scores.

See [HTTP admission and lifecycle testing](docs/http-lifecycle-validation.md) for bounded preprocessing and reproducible generated tests; [GPU validation](docs/gpu-validation.md) describes hardware requirements and the manual workflow.

See [advanced runtime experiments](docs/ADVANCED_RUNTIME.md) for teaching examples, load matrices and device validation.

See [service shutdown, preprocessing metrics and pinned reference validation](docs/service-validation.md) for process tests and the scheduled real-model checks.

## Performance measurement

See the [three-mode CPU report](docs/benchmarks/online-kv-comparison.md) and [reproduction/device commands](docs/IMPLEMENTATION.md).
See [bounded trace shutdown and metadata experiments](docs/runtime-hardening.md) for the paired trace matrix and standalone prefix measurements.
Run `python3 scripts/compare_kv.py --model models/qwen2.5-0.5b-instruct` for fresh-server trials, RSS and KV allocation counts.

Use `scripts/benchmark.py --requests 8 --concurrency 1 4 --max-tokens 32`.
It warms up first, counts model tokens from terminal usage, and reports failures
separately. `/metrics` provides a ten-second generated-token throughput window,
a separate lifetime average, finishing requests, prefix occupancy/hit tokens,
and combined scheduled-token counts. Client event latency and engine sampling
latency have different boundaries. Previous contiguous-path benchmark numbers
are not a performance claim for the new paged implementation.

See [implementation and validation guide](docs/IMPLEMENTATION.md) for all new
controls, memory semantics, and independent reference-model tests.

## Repository layout

```text
crates/mini-vllm-core       domain types (requests, events, engine config)
crates/mini-vllm-sampling   penalty → temperature → top-k → top-p → sample
crates/mini-vllm-kv         KV tensor cache + block manager
crates/mini-vllm-tokenizer  HF tokenizer, incremental detokenizer, ChatML
crates/mini-vllm-model      config/device/loader + Qwen2 components
crates/mini-vllm-engine     engine thread, scheduler, batches, metrics
crates/mini-vllm-server     axum routes, OpenAI types, SSE
crates/mini-vllm-cli        mini-vllm binary
scripts/benchmark.py        concurrency benchmark (TTFT / TPOT / tok/s)
```

## Development

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md) for the full checks and [SECURITY.md](SECURITY.md) for private vulnerability reporting. Parser fuzz targets and reproducible commands are in [fuzz/README.md](fuzz/README.md).

## Limitations

Educational/experimental quality: single-replica, no auth, no quantization,
attention not fused across sequences, physical pages implemented through portable
Candle operations rather than a custom GPU kernel. See
[`DESIGN.md §13`](./DESIGN.md#13-已知限制与路线图) for the full list and roadmap.

## License

MIT — see [LICENSE](./LICENSE). The bundled Qwen chat template retains its [Apache-2.0 license and attribution](crates/mini-vllm-tokenizer/src/templates/LICENSE-Qwen).

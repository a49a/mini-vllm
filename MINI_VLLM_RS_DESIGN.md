# mini-vLLM.rs — Rust LLM Inference Engine Design Specification

> This document is intended to be consumed directly by an AI coding agent.
> Build the system incrementally.
> Prioritize correctness, observability, and a working vertical slice before optimization.

---

# 1. Project Goal

Build a small but real **LLM inference engine in Rust**, inspired by vLLM architecture and serving patterns.

The project should demonstrate:

- decoder-only transformer inference
- model loading
- tokenization
- KV cache
- autoregressive generation
- sampling
- request scheduling
- continuous batching
- streaming generation
- OpenAI-compatible HTTP serving
- CPU / Metal / CUDA-capable execution where practical
- performance metrics and profiling

The project is NOT intended to compete with production vLLM.

The objective is:

> Implement the core ideas of an inference server deeply enough to understand how modern LLM serving systems work.

Working name:

```text
mini-vllm-rs
```

Binary name:

```text
mini-vllm
```

---

# 2. Product Positioning

Think of the project as:

```text
Candle
  +
custom inference runtime
  +
scheduler
  +
KV cache manager
  +
OpenAI-compatible server
```

The system should eventually support:

```text
POST /v1/completions
POST /v1/chat/completions
GET  /v1/models
```

with streaming:

```text
stream=true
```

The project should be usable as a local inference server.

Example:

```bash
mini-vllm serve \
  --model ./models/qwen2.5-0.5b-instruct \
  --host 127.0.0.1 \
  --port 8000
```

Then:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-model",
    "messages": [
      {"role": "user", "content": "Explain Rust ownership briefly."}
    ],
    "stream": false
  }'
```

---

# 3. Core Design Principle

Do NOT start by implementing batching, paged attention, quantization, or CUDA kernels.

Build the project in layers:

```text
model load
   ↓
tokenizer
   ↓
single forward pass
   ↓
single-request generation
   ↓
KV cache
   ↓
streaming
   ↓
HTTP API
   ↓
multiple requests
   ↓
scheduler
   ↓
continuous batching
   ↓
KV memory management
   ↓
optimization
```

Correctness comes before throughput.

---

# 4. Target Scope

The first complete version should support:

- one decoder-only transformer architecture
- local model loading
- Safetensors weights
- Hugging Face tokenizer files
- text completion
- chat completion
- greedy decoding
- temperature sampling
- top-k
- top-p
- repetition penalty
- stop token handling
- KV cache
- streaming
- concurrent HTTP clients
- continuous batching
- per-request cancellation
- basic metrics

Recommended first supported architecture:

```text
Qwen2 / Qwen2.5-style decoder-only model
```

Recommended initial test model:

```text
Qwen2.5-0.5B-Instruct
```

The implementation should keep model-specific code isolated so another decoder-only architecture can be added later.

Possible future models:

```text
Llama
Mistral
Gemma
Phi
```

---

# 5. Explicit Non-Goals for the First Version

Do NOT initially implement:

- training
- fine-tuning
- distributed inference
- tensor parallelism
- pipeline parallelism
- speculative decoding
- multimodal inference
- LoRA hot-swapping
- prefix caching
- disaggregated serving
- Mixture-of-Experts routing
- custom CUDA kernels
- FlashAttention kernels written from scratch
- multi-node serving
- production authentication
- Kubernetes deployment logic
- embeddings API
- vision models
- speech models

Do not add these before the core runtime is stable.

---

# 6. Suggested Technology Stack

## Language

```text
Rust
```

## Async Runtime

```text
Tokio
```

## Tensor / Device Runtime

Preferred:

```text
Candle
```

Use Candle for:

- tensors
- devices
- common tensor operations
- Safetensors loading
- CPU execution
- Metal where supported
- CUDA where supported

Do NOT hide the entire inference runtime behind a high-level generation helper.

The point of the project is to own:

- generation loop
- KV cache lifecycle
- scheduling
- batching
- sampling
- request state

---

## Tokenizer

Recommended:

```text
tokenizers
```

Use Hugging Face tokenizer files when possible.

---

## HTTP Server

Recommended:

```text
axum
```

Supporting libraries:

```text
tower
tower-http
```

---

## Serialization

```text
serde
serde_json
```

---

## Errors

```text
thiserror
anyhow
```

Use:

```text
thiserror
```

for typed subsystem errors.

Use:

```text
anyhow
```

at application boundaries where appropriate.

---

## Logging / Tracing

```text
tracing
tracing-subscriber
```

---

## CLI

```text
clap
```

---

## Metrics

Initially:

```text
internal counters + tracing
```

Later optionally expose Prometheus-compatible metrics.

---

# 7. High-Level Architecture

Target architecture:

```text
                           ┌──────────────────────┐
                           │     HTTP Clients     │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │   OpenAI API Layer   │
                           │       Axum           │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │    Request Engine    │
                           │ validation / IDs     │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │      Scheduler       │
                           │ admission / batching │
                           └──────────┬───────────┘
                                      │
                         ┌────────────┴─────────────┐
                         ▼                          ▼
              ┌────────────────────┐     ┌────────────────────┐
              │   Prefill Batch    │     │   Decode Batch     │
              └──────────┬─────────┘     └──────────┬─────────┘
                         │                          │
                         └────────────┬─────────────┘
                                      ▼
                           ┌──────────────────────┐
                           │   Model Executor     │
                           │ Candle / Device      │
                           └──────────┬───────────┘
                                      │
                    ┌─────────────────┼──────────────────┐
                    ▼                 ▼                  ▼
             ┌────────────┐   ┌──────────────┐   ┌────────────┐
             │ KV Manager │   │   Sampler    │   │ Tokenizer  │
             └────────────┘   └──────────────┘   └────────────┘
```

---

# 8. Workspace Layout

Use a Cargo workspace.

Suggested layout:

```text
mini-vllm-rs/
├── Cargo.toml
├── README.md
├── LICENSE
├── DESIGN.md
│
├── crates/
│   ├── mini-vllm-core/
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── config.rs
│   │       ├── error.rs
│   │       └── types.rs
│   │
│   ├── mini-vllm-model/
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── loader.rs
│   │       ├── config.rs
│   │       ├── qwen2.rs
│   │       ├── attention.rs
│   │       ├── rope.rs
│   │       ├── rms_norm.rs
│   │       └── mlp.rs
│   │
│   ├── mini-vllm-tokenizer/
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── tokenizer.rs
│   │       └── chat_template.rs
│   │
│   ├── mini-vllm-kv/
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── cache.rs
│   │       ├── block.rs
│   │       └── manager.rs
│   │
│   ├── mini-vllm-sampling/
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── params.rs
│   │       └── sampler.rs
│   │
│   ├── mini-vllm-engine/
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── engine.rs
│   │       ├── request.rs
│   │       ├── sequence.rs
│   │       ├── scheduler.rs
│   │       ├── batch.rs
│   │       └── executor.rs
│   │
│   ├── mini-vllm-server/
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── api.rs
│   │       ├── openai.rs
│   │       ├── streaming.rs
│   │       └── routes.rs
│   │
│   └── mini-vllm-cli/
│       └── src/
│           └── main.rs
│
├── tests/
│   ├── integration/
│   └── fixtures/
│
└── scripts/
```

Do not create all crates immediately if it slows initial implementation.

It is acceptable to begin with:

```text
core
model
engine
server
cli
```

and split further once boundaries are clear.

---

# 9. Fundamental Domain Objects

The runtime should model requests explicitly.

Suggested types:

```rust
pub type RequestId = String;

pub struct GenerationRequest {
    pub id: RequestId,
    pub prompt_token_ids: Vec<u32>,
    pub sampling: SamplingParams,
    pub max_new_tokens: usize,
    pub stop_token_ids: Vec<u32>,
}

pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub seed: Option<u64>,
}
```

Per-request runtime state:

```rust
pub enum SequenceStatus {
    Waiting,
    Prefill,
    Running,
    Finished,
    Cancelled,
    Failed,
}
```

```rust
pub struct SequenceState {
    pub request_id: RequestId,
    pub prompt_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
    pub status: SequenceStatus,
    pub max_new_tokens: usize,
}
```

Do not mix HTTP-specific types into the core engine.

---

# 10. Model Abstraction

Create a narrow model abstraction.

Example:

```rust
pub trait CausalLm: Send + Sync {
    fn device(&self) -> &candle_core::Device;

    fn vocab_size(&self) -> usize;

    fn forward_prefill(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_cache: &mut KvCache,
    ) -> Result<Tensor>;

    fn forward_decode(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_cache: &mut KvCache,
    ) -> Result<Tensor>;
}
```

The exact API may change based on Candle constraints.

The important architectural rule is:

> The engine must not depend directly on Qwen-specific internals.

---

# 11. Initial Model Architecture

Implement a Qwen2-style decoder-only transformer.

Core layer structure:

```text
Token Embedding
      ↓
N × Transformer Decoder Layer
      ↓
RMSNorm
      ↓
LM Head
      ↓
Logits
```

Each decoder layer:

```text
input
  ↓
RMSNorm
  ↓
Self Attention
  ↓
Residual
  ↓
RMSNorm
  ↓
SwiGLU MLP
  ↓
Residual
```

Required concepts:

- RMSNorm
- RoPE
- Grouped Query Attention if required by model config
- causal masking
- SwiGLU
- KV cache
- tied or untied LM head according to config

Do not hard-code tensor dimensions.

Read them from:

```text
config.json
```

---

# 12. Model Configuration

Parse model configuration from Hugging Face-compatible `config.json`.

Possible internal representation:

```rust
pub struct ModelConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
}
```

Validate invariants during startup.

Examples:

```text
hidden_size % num_attention_heads == 0
num_attention_heads % num_key_value_heads == 0
```

Fail early on unsupported configuration.

---

# 13. Weight Loading

Initial supported format:

```text
Safetensors
```

Support:

```text
model.safetensors
```

and optionally sharded models:

```text
model-00001-of-00002.safetensors
model-00002-of-00002.safetensors
model.safetensors.index.json
```

Do not require GGUF for MVP.

GGUF can be added later.

Loading flow:

```text
model directory
   ↓
config.json
   ↓
tokenizer files
   ↓
Safetensors index
   ↓
tensor mapping
   ↓
device tensors
```

Log:

```text
model architecture
parameter count if practical
dtype
device
load duration
```

Do not print large tensors.

---

# 14. Device Selection

CLI should support:

```text
--device cpu
--device metal
--device cuda
--device auto
```

`auto` behavior:

```text
prefer CUDA if available
else Metal if available
else CPU
```

If automatic detection is difficult or unreliable in a given build, document the limitation clearly.

Do not silently fall back if the user explicitly requested a specific unavailable device.

---

# 15. Tokenizer

Load a Hugging Face tokenizer.

Required operations:

```rust
encode(text) -> Vec<u32>
decode(token_ids) -> String
```

Chat requests additionally require:

```text
messages
   ↓
chat template
   ↓
prompt string
   ↓
tokenizer
```

Do not initially build a universal Jinja interpreter unless necessary.

For the first supported model:

- support the known model chat format
- isolate chat-template behavior behind an abstraction

Example:

```rust
pub trait ChatTemplate {
    fn render(&self, messages: &[ChatMessage]) -> Result<String>;
}
```

---

# 16. Basic Inference — No KV Cache

Before implementing KV caching, create a correctness-only path.

Algorithm:

```text
prompt tokens
    ↓
run entire sequence through model
    ↓
take last-token logits
    ↓
sample token
    ↓
append token
    ↓
run entire expanded sequence again
    ↓
repeat
```

This is intentionally inefficient.

Purpose:

- validate model loading
- validate attention
- validate RoPE
- validate tokenizer
- validate sampling
- validate logits

Acceptance test:

Given a fixed model and seed, generation should produce plausible text.

This phase must work before KV caching.

---

# 17. KV Cache

After correctness is established, implement KV caching.

For each transformer layer store:

```text
K
V
```

Conceptually:

```text
[layer][sequence][kv_head][token][head_dim]
```

Exact tensor layout should be selected for efficient Candle operations.

Initial implementation may use a contiguous per-sequence cache.

Do NOT implement paged KV in the first KV-cache version.

Possible type:

```rust
pub struct LayerKvCache {
    pub key: Tensor,
    pub value: Tensor,
}

pub struct KvCache {
    pub layers: Vec<LayerKvCache>,
    pub seq_len: usize,
    pub capacity: usize,
}
```

Design it so the implementation can later evolve toward block-based storage.

---

# 18. Prefill vs Decode

The runtime must distinguish:

## Prefill

Input:

```text
entire prompt
```

Work:

```text
compute attention for prompt tokens
populate KV cache
produce logits for final prompt token
```

## Decode

Input:

```text
one newly generated token per active sequence
```

Work:

```text
compute Q for the new token
append K/V
attend over existing cache
produce next-token logits
```

This distinction becomes fundamental once batching is implemented.

---

# 19. Generation Loop

Single-request generation loop:

```text
tokenize prompt
     ↓
prefill
     ↓
sample next token
     ↓
emit token
     ↓
decode using KV cache
     ↓
sample next token
     ↓
...
```

Stop conditions:

```text
EOS token
stop token
max_new_tokens reached
request cancelled
runtime error
```

The generation engine should return a finish reason.

Suggested enum:

```rust
pub enum FinishReason {
    Stop,
    Length,
    Cancelled,
    Error,
}
```

---

# 20. Sampling

Implement in this order:

## Phase A

```text
greedy decoding
```

## Phase B

```text
temperature
top-k
```

## Phase C

```text
top-p
repetition penalty
```

Sampling pipeline should be explicit.

Conceptually:

```text
logits
  ↓
repetition penalty
  ↓
temperature
  ↓
top-k filtering
  ↓
top-p filtering
  ↓
softmax
  ↓
sample
```

Use stable numerical operations.

Add deterministic seeded tests.

---

# 21. Streaming

The engine must support token streaming.

Internally, use a channel.

Example:

```rust
pub enum GenerationEvent {
    Token {
        token_id: u32,
        text: String,
    },
    Finished {
        reason: FinishReason,
    },
    Error {
        message: String,
    },
}
```

Possible transport:

```text
tokio::sync::mpsc
```

The HTTP layer should transform engine events into Server-Sent Events.

Do not make the engine depend on SSE.

---

# 22. OpenAI-Compatible API

Implement:

```text
GET /v1/models
POST /v1/completions
POST /v1/chat/completions
```

Start with a useful subset of request fields.

Supported generation parameters:

```text
model
prompt
messages
max_tokens
temperature
top_p
stream
stop
seed
```

Optional:

```text
top_k
```

even though it is not standard OpenAI API.

Ignore or reject unsupported fields explicitly.

Do not silently claim unsupported behavior.

---

# 23. Chat Completion Request

Example request:

```json
{
  "model": "local-model",
  "messages": [
    {
      "role": "system",
      "content": "You are a concise assistant."
    },
    {
      "role": "user",
      "content": "What is ownership in Rust?"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 128,
  "stream": false
}
```

Non-streaming response should resemble:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "local-model",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 42,
    "total_tokens": 62
  }
}
```

Exact compatibility can improve over time.

---

# 24. Streaming API

Use:

```text
Content-Type: text/event-stream
```

Emit chunks similar to OpenAI streaming responses.

Final event:

```text
data: [DONE]
```

Important:

- flush incrementally
- handle client disconnect
- cancel the underlying generation request when practical

Do not continue expensive inference indefinitely after a client disconnect.

---

# 25. Engine API

The server should communicate with the inference runtime through a stable interface.

Example:

```rust
pub struct EngineHandle {
    command_tx: mpsc::Sender<EngineCommand>,
}
```

Commands:

```rust
pub enum EngineCommand {
    Generate {
        request: GenerationRequest,
        events: mpsc::Sender<GenerationEvent>,
    },
    Cancel {
        request_id: RequestId,
    },
}
```

The model executor should preferably live in one controlled runtime context rather than allowing every HTTP request to invoke the model independently.

---

# 26. Why Use a Dedicated Engine Loop

Do NOT structure the server like this:

```text
HTTP request
   ↓
lock model
   ↓
run full generation
   ↓
unlock
```

That architecture makes continuous batching difficult.

Instead:

```text
many HTTP requests
        ↓
 engine command queue
        ↓
    scheduler
        ↓
 batched model execution
```

The scheduler decides which requests run at each step.

---

# 27. Scheduler — First Version

Initial scheduler may be simple FIFO.

Maintain:

```text
waiting queue
running sequences
```

Possible structure:

```rust
pub struct Scheduler {
    waiting: VecDeque<SequenceGroup>,
    running: Vec<SequenceGroup>,
    max_num_seqs: usize,
    max_batch_tokens: usize,
}
```

Admission criteria may include:

```text
maximum concurrent sequences
KV cache capacity
maximum batch tokens
```

Do not optimize policy prematurely.

---

# 28. Continuous Batching

Once multiple requests work independently, implement continuous batching.

Key idea:

Traditional static batching:

```text
request A ───────────────┐
request B ───────────────┼─ batch waits until everyone finishes
request C ───────────────┘
```

Continuous batching:

```text
step 1: A B C
step 2: A B C
C finishes
step 3: A B D
B finishes
step 4: A D E
```

The scheduler should be able to:

- remove finished sequences
- admit waiting requests
- form a decode batch each iteration

---

# 29. Prefill Scheduling

Initially, keep prefill scheduling simple.

Possible policy:

```text
if decode requests exist:
    prioritize decode
else:
    run prompt prefill
```

Later improve to mixed batching.

The first version may process one prompt prefill at a time.

Then evolve toward:

```text
batched prompt prefill
```

Do not begin with chunked prefill.

---

# 30. Decode Batch

For active sequences:

```text
sequence A -> token 128
sequence B -> token 731
sequence C -> token 42
```

Create a batch:

```text
input_ids = [128, 731, 42]
positions = [pos_A, pos_B, pos_C]
```

The model should produce logits:

```text
[batch_size, vocab_size]
```

Sampler then chooses one token per sequence.

Important:

Each sequence has an independent:

```text
KV cache state
generation length
sampling configuration
stop condition
```

---

# 31. Batching Abstraction

Suggested representation:

```rust
pub struct DecodeBatch {
    pub request_ids: Vec<RequestId>,
    pub input_token_ids: Vec<u32>,
    pub positions: Vec<usize>,
}
```

Later this may need:

```text
cache block tables
sequence lengths
attention metadata
```

Do not over-generalize before needed.

---

# 32. KV Cache Manager — Stage 1

First multi-request implementation may allocate a contiguous KV cache per request.

This is acceptable initially.

Example:

```text
Request A -> own cache
Request B -> own cache
Request C -> own cache
```

Problems:

- fragmentation
- reallocation
- inefficient large memory reservation

These issues motivate the next stage.

---

# 33. Block-Based KV Cache — Stage 2

After continuous batching is stable, introduce fixed-size KV blocks.

Example:

```text
BLOCK_SIZE = 16 tokens
```

Physical blocks:

```text
block 0
block 1
block 2
...
```

Logical sequence:

```text
Sequence A
logical blocks: [0, 1, 2]
physical blocks: [7, 13, 4]
```

Possible types:

```rust
pub type BlockId = usize;

pub struct BlockTable {
    pub blocks: Vec<BlockId>,
}

pub struct KvBlockManager {
    pub free_blocks: Vec<BlockId>,
    pub sequence_blocks: HashMap<RequestId, BlockTable>,
}
```

This stage is inspired by paged KV-cache concepts.

Do not call it fully equivalent to vLLM PagedAttention unless the attention implementation actually consumes paged storage efficiently.

---

# 34. KV Memory Accounting

Expose:

```text
total KV memory
used KV memory
free KV memory
active sequence count
allocated blocks
free blocks
```

The engine should be able to reject or queue requests when cache capacity is insufficient.

Never rely on out-of-memory crashes as normal flow control.

---

# 35. Attention

Correctness-first implementation:

```text
Q = input × Wq
K = input × Wk
V = input × Wv

apply RoPE(Q, K)

scores = Q × K^T / sqrt(head_dim)

apply causal mask

probs = softmax(scores)

output = probs × V
```

Support GQA where the model configuration requires it.

The attention implementation should clearly separate:

```text
prefill attention
decode attention
```

even if some code is shared.

---

# 36. RoPE

Implement rotary positional embeddings correctly.

Inputs:

```text
Q
K
position ids
rope_theta
head_dim
```

Add tests for:

- shape correctness
- deterministic outputs
- known small tensor cases if practical

Do not bury RoPE inside a giant forward method.

---

# 37. Causal Mask

During prefill:

```text
token i must not attend to future token j > i
```

During single-token decode:

```text
new token can attend to all previous cached tokens and itself
```

KV caching should avoid rebuilding a full large triangular mask during every decode step when possible.

---

# 38. Numerical Types

Initial default:

```text
F32
```

Then add:

```text
F16
BF16
```

where device support permits.

The loader should respect the model's available weight dtype where practical.

Do not introduce quantization until unquantized inference is correct.

---

# 39. Quantization — Later Phase

Possible later formats:

```text
GGUF
Q4
Q5
Q8
```

Quantization is explicitly post-MVP.

If added, keep quantized model execution behind the model/executor abstraction.

Do not contaminate scheduler logic with quantization-specific details.

---

# 40. Engine Concurrency Model

Recommended design:

```text
Tokio HTTP tasks
       ↓
mpsc
       ↓
single logical engine task
       ↓
scheduler
       ↓
model executor
```

This makes mutation of:

```text
scheduler state
request state
KV state
```

easier to reason about.

The tensor runtime may have its own device synchronization behavior.

Avoid fine-grained locking around every tensor operation.

---

# 41. Request Lifecycle

Lifecycle:

```text
HTTP request received
      ↓
validation
      ↓
tokenization
      ↓
request accepted
      ↓
Waiting
      ↓
Prefill
      ↓
Running
      ↓
Finished
```

Exceptional states:

```text
Cancelled
Failed
Rejected
```

Log state transitions at debug or trace level.

---

# 42. Cancellation

Support:

```text
request cancelled by API client
server shutdown
generation deadline
```

Cancellation should:

- mark request cancelled
- stop scheduling it
- release KV cache
- close streaming channel

Cancellation must not leak cache memory.

---

# 43. Backpressure

Streaming channels must have bounded capacity.

Do not use unbounded queues for token streams.

Example:

```text
channel capacity: 32 or 64 events
```

If a client stops consuming data:

- do not allow unbounded memory growth
- eventually cancel or stall the request safely

---

# 44. Timeouts

Configurable limits:

```text
request timeout
maximum prompt tokens
maximum generated tokens
maximum context length
```

Example defaults:

```text
max_model_len = model config maximum
max_new_tokens = 512
```

Reject obviously invalid inputs before model execution.

---

# 45. CLI

Initial commands:

```bash
mini-vllm inspect --model <path>
```

```bash
mini-vllm generate \
  --model <path> \
  --prompt "Hello"
```

```bash
mini-vllm serve \
  --model <path> \
  --host 127.0.0.1 \
  --port 8000
```

Possible serve flags:

```text
--device auto
--dtype auto
--max-model-len
--max-num-seqs
--max-batch-tokens
--seed
--log-level
```

---

# 46. Inspect Command

Example:

```bash
mini-vllm inspect --model ./model
```

Output:

```text
Architecture: Qwen2
Layers: 24
Hidden size: 896
Attention heads: 14
KV heads: 2
Vocabulary size: ...
Max context: ...
Device: Metal
Weight dtype: BF16
```

This command helps validate the loader independently of generation.

---

# 47. Generate Command

Before HTTP serving exists:

```bash
mini-vllm generate \
  --model ./model \
  --prompt "The capital of France is" \
  --max-new-tokens 20 \
  --temperature 0
```

This command is a critical vertical slice.

It should stream generated text to stdout.

---

# 48. Configuration

Support command-line configuration first.

Optional config file later.

Possible runtime config:

```rust
pub struct EngineConfig {
    pub max_model_len: usize,
    pub max_num_seqs: usize,
    pub max_batch_tokens: usize,
    pub kv_block_size: usize,
}
```

Possible server config:

```rust
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}
```

Keep model config separate from runtime config.

---

# 49. Metrics

Collect basic counters and latency measurements.

Important metrics:

```text
requests_total
requests_running
requests_waiting
requests_finished
requests_failed

prompt_tokens_total
generated_tokens_total

time_to_first_token_ms
inter_token_latency_ms
request_latency_ms

tokens_per_second
batch_size

kv_blocks_total
kv_blocks_used
```

Do not optimize based only on tokens/sec.

Track:

```text
TTFT
TPOT / inter-token latency
throughput
```

---

# 50. Logging

Useful logs:

```text
model loading started
model loading completed
server listening

request admitted
request queued
prefill started
prefill completed
decode step
request finished
request cancelled

KV block allocated
KV block released
out of KV capacity

batch formed
batch size
batch token count
```

Never log full user prompts by default.

Allow prompt logging only through an explicit development setting.

---

# 51. Profiling

Add optional timing spans around:

```text
tokenization
prefill
attention
MLP
sampling
decode step
HTTP serialization
```

Use:

```text
tracing spans
```

The implementation should make it possible to identify whether time is spent in:

```text
model compute
scheduler overhead
tokenizer
HTTP
```

---

# 52. Testing Strategy

Testing is essential.

---

## 52.1 Unit Tests

Add unit tests for:

```text
sampling
top-k
top-p
repetition penalty
stop conditions
scheduler admission
scheduler removal
KV block allocation
KV block release
request lifecycle
config validation
```

---

## 52.2 Model Component Tests

Test:

```text
RMSNorm
RoPE
causal mask
GQA shape transformations
MLP shapes
attention output shapes
```

Use tiny deterministic tensors.

---

## 52.3 Golden Logit Tests

For a tiny model fixture or small public model:

```text
prompt tokens
   ↓
forward
   ↓
logits
```

Compare selected logits against a known-good reference implementation within tolerance.

This is more valuable than simply checking that generated text "looks reasonable."

---

## 52.4 KV Cache Equivalence Test

Critical test:

```text
full-sequence decoding
```

must closely match:

```text
prefill + cached incremental decoding
```

for the same sequence.

Example:

```text
logits_without_cache ≈ logits_with_cache
```

within numerical tolerance.

Do not proceed to batching if this test fails.

---

## 52.5 Batched vs Unbatched Equivalence

For two prompts:

```text
run independently
```

and:

```text
run as decode batch
```

Results under greedy decoding should match.

---

## 52.6 API Integration Tests

Test:

```text
/v1/models
/v1/completions
/v1/chat/completions
stream=false
stream=true
```

Mock the engine when HTTP behavior alone is being tested.

---

# 53. Benchmarks

Use Criterion or a lightweight benchmark harness later.

Benchmark separately:

```text
tokenization
prefill
single-token decode
sampling
scheduler step
KV allocation
```

End-to-end scenarios:

```text
1 request
4 concurrent requests
16 concurrent requests
```

Measure:

```text
TTFT
tokens/sec
average decode batch size
peak KV memory
```

---

# 54. Correctness Before Optimization

Every optimization should have:

```text
baseline implementation
correctness test
optimized implementation
equivalence test
benchmark
```

Examples:

```text
full sequence
    ↓
KV cache
```

```text
single request
    ↓
batched decode
```

```text
contiguous cache
    ↓
block cache
```

Never delete the ability to verify optimized behavior against a simpler baseline until the optimization is well tested.

---

# 55. Implementation Phases

Implement strictly in dependency order.

---

# Phase 0 — Workspace Bootstrap

Create:

```text
Cargo workspace
CLI crate
core crate
model crate
engine crate
server crate
```

Set up:

```text
clap
tokio
serde
tracing
```

Acceptance criteria:

```bash
cargo build --workspace
cargo test --workspace
```

passes.

CLI:

```bash
mini-vllm --help
```

works.

---

# Phase 1 — Model Inspection

Implement:

```text
load config.json
load tokenizer
discover Safetensors files
inspect metadata
select device
```

CLI:

```bash
mini-vllm inspect --model ./model
```

Acceptance criteria:

- model configuration parses
- tokenizer loads
- weights are discoverable
- supported architecture is validated
- useful model metadata is printed

Do NOT implement generation yet.

---

# Phase 2 — Model Forward Pass

Implement Qwen2 components:

```text
embedding
RMSNorm
RoPE
self-attention
SwiGLU MLP
decoder layer
LM head
```

Load model weights.

Implement:

```text
forward(input_ids)
```

without KV caching.

Acceptance criteria:

Given a small prompt:

```text
forward returns logits
shape = [batch, sequence, vocab]
```

Add component tests.

---

# Phase 3 — Golden Logit Validation

Compare model output against a known-good reference for a fixed prompt.

Acceptance criteria:

Selected logits are numerically close to the reference.

Do not begin generation until this passes.

---

# Phase 4 — Simple Generation

Implement:

```text
greedy sampling
full-sequence autoregressive generation
```

CLI:

```bash
mini-vllm generate \
  --model ./model \
  --prompt "Hello" \
  --temperature 0
```

Acceptance criteria:

- generated text is valid
- EOS works
- max_new_tokens works
- deterministic greedy output works

Performance is not important yet.

---

# Phase 5 — Sampling

Implement:

```text
temperature
top-k
top-p
repetition penalty
seed
```

Acceptance criteria:

- deterministic seeded tests
- temperature=0 maps to greedy behavior
- top-k filtering works
- top-p filtering works

---

# Phase 6 — KV Cache

Implement:

```text
prefill
incremental decode
per-layer KV cache
```

Acceptance criteria:

For the same sequence:

```text
cached logits ≈ uncached logits
```

Generation output under greedy decoding must match the Phase 4 implementation.

---

# Phase 7 — Streaming Engine

Create engine request types.

Implement:

```text
Generate command
GenerationEvent
bounded event channel
cancellation
```

CLI generation should now use the engine API instead of calling the model directly.

Acceptance criteria:

Tokens are emitted incrementally.

---

# Phase 8 — HTTP Server

Implement Axum server.

Endpoints:

```text
GET /health
GET /v1/models
POST /v1/completions
POST /v1/chat/completions
```

Support:

```text
stream=false
stream=true
```

Acceptance criteria:

An OpenAI-style client can receive local model output.

---

# Phase 9 — Multi-Request Runtime

Support multiple simultaneous submitted requests.

Initially requests may still execute mostly sequentially.

Acceptance criteria:

- concurrent HTTP clients are accepted
- request state is independent
- cancellation is independent
- no request corrupts another request's KV cache

---

# Phase 10 — Scheduler

Implement:

```text
waiting queue
running set
admission rules
FIFO policy
```

Acceptance criteria:

```text
max_num_seqs
```

is respected.

Finished requests release resources.

---

# Phase 11 — Batched Decode

Combine active decode steps.

Acceptance criteria:

For greedy decoding:

```text
batched output == independently generated output
```

for the same requests.

Record:

```text
decode batch size
```

in tracing.

---

# Phase 12 — Continuous Batching

Allow finished requests to leave and waiting requests to join between decode iterations.

Acceptance criteria:

Given concurrent requests of different lengths:

```text
short requests finish early
new requests can enter
long requests continue
```

without waiting for a static batch to finish.

---

# Phase 13 — Block KV Manager

Replace or augment per-request cache allocation with fixed-size blocks.

Implement:

```text
free block pool
block tables
allocation
release
capacity accounting
```

Acceptance criteria:

- blocks are reused after request completion
- cancellation releases blocks
- no double allocation
- no cache leaks in tests

The model may still copy/cache data into contiguous temporary tensors if Candle operations require it.

Correctness first.

---

# Phase 14 — Performance Instrumentation

Expose:

```text
TTFT
request latency
generated token throughput
average batch size
KV usage
```

Add benchmark script.

Establish baseline performance numbers.

Do not optimize before measurements exist.

---

# 56. First Vertical Slice

The first major milestone should be:

```text
local Qwen model
      ↓
Rust loader
      ↓
Candle forward
      ↓
tokenizer
      ↓
greedy decoding
      ↓
terminal output
```

No server.

No scheduler.

No batching.

No paged KV.

This must be proven correct first.

---

# 57. Second Vertical Slice

Then:

```text
prompt
  ↓
prefill
  ↓
KV cache
  ↓
incremental decode
  ↓
streaming token events
```

This establishes the core inference engine.

---

# 58. Third Vertical Slice

Then:

```text
OpenAI request
     ↓
Axum
     ↓
Engine queue
     ↓
inference
     ↓
SSE chunks
```

At this point the project becomes practically usable.

---

# 59. Fourth Vertical Slice

Then:

```text
many requests
      ↓
scheduler
      ↓
continuous batching
      ↓
batched decode
      ↓
independent token streams
```

This is where the project starts to resemble a real LLM serving runtime.

---

# 60. Code Quality Rules

Run before completing each phase:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
```

Avoid:

```text
unwrap() in runtime code
expect() without a strong invariant
large god structs
global mutable state
blocking operations in async HTTP handlers
holding async locks across expensive model execution
unbounded channels
silent error swallowing
```

Prefer:

```text
small typed modules
explicit request states
bounded queues
structured errors
tracing spans
tests around scheduling and memory ownership
```

---

# 61. Safety Against Resource Exhaustion

The server should validate:

```text
prompt length
max_new_tokens
context length
active sequence count
KV capacity
request body size
```

Do not allow arbitrary user input to cause unlimited memory growth.

Return clear errors when limits are exceeded.

---

# 62. Graceful Shutdown

On shutdown:

```text
stop accepting new requests
mark server draining
finish or cancel active requests
release KV resources
stop engine loop
exit cleanly
```

Do not terminate while model state is being mutated without coordination.

---

# 63. README Requirements

README should eventually include:

```text
what mini-vLLM.rs is
what it is not
supported models
supported devices
installation
model preparation
CLI examples
OpenAI API examples
architecture diagram
performance notes
development roadmap
limitations
license
```

Clearly state that it is an educational / experimental inference runtime.

---

# 64. Recommended Repository Documentation

Keep:

```text
README.md
DESIGN.md
docs/
```

Possible future docs:

```text
docs/model-loader.md
docs/kv-cache.md
docs/scheduler.md
docs/continuous-batching.md
docs/openai-api.md
docs/benchmarking.md
```

Document design changes that materially differ from this specification.

---

# 65. Important Architectural Boundaries

Keep these concerns separated:

```text
HTTP API
≠
scheduler

scheduler
≠
model implementation

model implementation
≠
sampling

sampling
≠
tokenization

KV ownership
≠
HTTP connection lifetime
```

A cancelled HTTP connection should communicate cancellation to the engine.

It should not directly mutate model internals.

---

# 66. What the AI Coding Agent Must NOT Do

Do not:

1. replace the custom engine with a shell call to another inference server
2. wrap llama.cpp and claim the inference runtime has been implemented
3. delegate generation to Python
4. call an external model API
5. skip correctness testing and jump directly to continuous batching
6. add distributed systems before local inference works
7. implement custom unsafe GPU kernels without necessity
8. hide all generation logic inside a third-party high-level pipeline
9. silently change the first supported model architecture
10. implement every phase in one giant commit

External libraries are allowed for:

```text
tensor operations
tokenization
HTTP serving
Safetensors parsing
async runtime
```

But the project must own:

```text
generation loop
request lifecycle
sampling orchestration
KV cache lifecycle
scheduler
batch construction
continuous batching
streaming engine
```

---

# 67. Definition of MVP Complete

MVP is complete when:

1. a supported local model loads from disk
2. tokenizer loads correctly
3. CLI generation works
4. KV-cached decoding works
5. greedy and stochastic sampling work
6. OpenAI-style completion API works
7. chat completion API works
8. streaming works
9. multiple concurrent requests work
10. continuous batching works
11. finished/cancelled requests release KV resources
12. basic inference metrics are available
13. core correctness tests pass

Block-based KV caching may be considered either late-MVP or immediately post-MVP depending on implementation complexity.

---

# 68. Post-MVP Roadmap

Possible order:

```text
block-based KV cache
      ↓
better batched prefill
      ↓
mixed prefill/decode scheduling
      ↓
F16/BF16 tuning
      ↓
GGUF / quantization
      ↓
prefix caching
      ↓
chunked prefill
      ↓
speculative decoding
      ↓
additional architectures
```

Possible deeper systems work:

```text
custom Metal kernels
custom CUDA kernels
FlashAttention integration
paged attention kernels
tensor parallelism
```

Only pursue these after profiling demonstrates why they matter.

---

# 69. Learning Objectives

The project should make the implementation owner understand:

```text
why KV cache matters

why prefill and decode have different performance characteristics

why static batching wastes accelerator capacity

how continuous batching improves utilization

how request scheduling affects latency and throughput

why KV cache becomes a memory-management problem

why paged KV storage exists

how OpenAI-compatible streaming maps onto token generation

how model architecture affects runtime implementation

where TTFT comes from

where per-token latency comes from
```

If a design choice obscures these concepts, prefer a more explicit implementation.

---

# 70. Initial Task for the Coding Agent

Start with:

```text
Phase 0
Phase 1
```

Then implement the minimum of Phase 2 required to load real tensors.

Immediate objective:

> Create a Rust workspace that can inspect a local Qwen2/Qwen2.5 model directory, parse its configuration, load its tokenizer, discover/load Safetensors weights with Candle, select a device, and print validated model metadata.

Do NOT implement:

```text
HTTP server
scheduler
continuous batching
block KV cache
quantization
```

yet.

Before proceeding to text generation, demonstrate that:

```bash
mini-vllm inspect --model <path>
```

works against a real supported model directory.

Required validation:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
```

After Phase 1, continue into model-forward implementation incrementally.

For every completed phase, report:

```text
files changed
architecture decisions
tests added
commands run
remaining limitations
next phase
```

Do not ask for permission between ordinary implementation steps unless blocked by a genuinely external requirement.

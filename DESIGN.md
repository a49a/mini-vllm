# mini-vllm-rs 设计文档（中文）

> 本文描述 `mini-vllm-rs` 的实际实现架构，与代码同步维护。
> 原始英文规格见 [`MINI_VLLM_RS_DESIGN.md`](./MINI_VLLM_RS_DESIGN.md)。

---

## 1. 项目定位

`mini-vllm-rs` 是一个用 Rust 从零实现的**小型 LLM 推理引擎**，借鉴 vLLM 的架构与服务模式，用于深入理解现代 LLM 服务系统的核心机制：

- decoder-only Transformer 推理（首个支持架构：**Qwen2 / Qwen2.5**）
- 模型加载（Safetensors 单文件 / 分片）
- Hugging Face tokenizer 与 ChatML 对话模板
- KV Cache 与显存（内存）块管理
- 采样管线（repetition penalty / temperature / top-k / top-p / 贪心）
- 连续批处理（continuous batching）调度器
- 流式生成（token 事件流 + SSE）
- OpenAI 兼容 HTTP API
- CPU / Metal / CUDA（按编译特性）执行
- TTFT / TPOT / 吞吐等性能指标

**它不是**生产级 vLLM 替代品，而是教学/实验性质的推理运行时：
正确性优先于吞吐，每一项优化都必须有等价性测试与基线对照。

### 快速上手

```bash
cargo build --release -p mini-vllm-cli

# 检查模型目录
./target/release/mini-vllm inspect --model ./models/qwen2.5-0.5b-instruct

# 命令行生成（流式输出到 stdout）
./target/release/mini-vllm generate \
  --model ./models/qwen2.5-0.5b-instruct \
  --prompt "The capital of France is" \
  --max-new-tokens 32 --temperature 0

# 启动 OpenAI 兼容服务
./target/release/mini-vllm serve \
  --model ./models/qwen2.5-0.5b-instruct --host 127.0.0.1 --port 8000
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-0.5b-instruct",
    "messages": [{"role": "user", "content": "用一句话解释 Rust 所有权"}],
    "temperature": 0.7, "max_tokens": 128, "stream": false
  }'
```

---

## 2. 总体架构

```text
                           ┌──────────────────────┐
                           │     HTTP 客户端       │
                           └──────────┬───────────┘
                                      ▼
                           ┌──────────────────────┐
                           │  OpenAI API 层 (axum) │  /v1/completions
                           │  校验 / 分词 / SSE     │  /v1/chat/completions
                           └──────────┬───────────┘  /v1/models  /metrics
                                      │  EngineCommand (有界 mpsc)
                                      ▼
                           ┌──────────────────────┐
                           │     引擎专用线程       │  唯一可变状态拥有者
                           │   ┌────────────────┐ │
                           │   │    调度器        │ │  waiting → running
                           │   │  准入 / FIFO     │ │  max_num_seqs / KV 预算
                           │   └───────┬────────┘ │
                           │           ▼          │
                           │   ┌────────────────┐ │
                           │   │  模型执行器      │ │  混合 prefill+decode 批
                           │   │ Candle / 设备    │ │  线性层全批 + 注意力分序
                           │   └───────┬────────┘ │
                           └───────────┼──────────┘
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                  ▼
             ┌────────────┐   ┌──────────────┐   ┌────────────┐
             │ KV 块管理器  │   │    采样器     │   │  分词器      │
             │ 块表/容量核算 │   │ 温度/top-k/p  │   │ 增量反解码   │
             └────────────┘   └──────────────┘   └────────────┘
                    │
                    ▼  GenerationEvent (有界通道, 容量 64)
              每请求事件流 → JSON 响应 / SSE 流
```

**并发模型（关键决策）**：所有可变运行时状态（调度队列、序列状态、KV 缓存、块表）
只属于一个专用 OS 线程。HTTP 层通过有界命令队列与之通信，永不直接触碰模型内部。
这样做的理由：

- 避免在 async HTTP 处理器中执行阻塞的模型计算；
- 调度器/KV 状态不需要细粒度锁；
- 取消、背压、优雅关闭都有单一仲裁点。

模型在专用线程上逐步执行；空闲时 `blocking_recv()` 等待新命令；
被 KV 容量卡住的队列头用 10ms 休眠避免忙等。

---

## 3. Workspace 结构

```text
crates/
├── mini-vllm-core/       领域类型：GenerationRequest、SamplingParams、
│                         GenerationEvent、FinishReason、EngineConfig
├── mini-vllm-sampling/   采样管线（纯函数，输入 &[f32]，与张量库解耦）
├── mini-vllm-kv/         KV 张量缓存（cache.rs）+ 块管理器（manager.rs）
├── mini-vllm-tokenizer/  HF tokenizer 封装 + 增量反解码（HF DecodeStream 滑动窗口，O(窗口) 每 token）+ ChatML 模板
├── mini-vllm-model/      ModelConfig 解析、设备选择、Qwen2 组件、权重加载
├── mini-vllm-engine/     引擎线程、调度器、批构造、执行器、指标
├── mini-vllm-server/     axum 路由、OpenAI 类型、SSE 流式、断连取消
└── mini-vllm-cli/        mini-vllm 二进制：inspect / generate / serve
```

依赖方向单向：`cli → server/engine → model/kv/sampling/tokenizer → core`。
HTTP 类型只存在于 server 层， Candle 张量只出现在 model/kv 层。

---

## 4. 核心数据模型

```rust
// core（引擎语言，无 HTTP / 张量依赖）
pub struct GenerationRequest {
    pub id: RequestId,
    pub prompt_token_ids: Vec<u32>,      // 进入引擎前已分词
    pub sampling: SamplingParams,        // temperature/top_k/top_p/penalty/seed
    pub max_new_tokens: usize,
    pub stop_token_ids: Vec<String>...   // ∪ 模型 EOS ids
    pub stop_strings: Vec<String>,       // 命中即截断并停止
}

pub enum GenerationEvent {
    Token   { token_id: u32, text: String },   // text 为增量片段
    Finished{ reason: FinishReason, usage: Usage },
    Error   { message: String },
}

pub enum FinishReason { Stop, Length, Cancelled, Error }
```

序列运行时状态 `SequenceGroup`（engine 层）持有：请求、状态机
（Waiting/Prefill/Running/Finished/Cancelled/Failed）、按请求隔离的 `KvCache`、
按请求播种的 `Sampler`、增量反解码器、有界事件通道发送端、TTFT/ITL 时间戳。

**背压契约（§43）**：事件通道容量 64，且引擎线程**从不阻塞在消费者上**。
发送一律走 `try_send`：通道满时事件进入同等容量的有界 outbox，outbox 溢出
（即消费者慢了约 2×通道容量的事件量）时该请求被取消并停止计算——
拖累的只是它自己，不是引擎和其他请求。消费者还应把"流在没有终端事件的
情况下关闭"视为取消信号（慢消费者被取消时终端事件是尽力投递的）。

---

## 5. 模型层（mini-vllm-model）

### 5.1 配置解析与校验

`config.json`（HF snake_case 字段）→ `ModelConfig`，启动时校验：

```text
hidden_size % num_attention_heads == 0
num_attention_heads % num_key_value_heads == 0
num_key_value_heads <= num_attention_heads
维度字段 > 0；非 Qwen2 架构给出警告
eos_token_id 兼容单值与列表两种 HF 写法
```

张量维度一律读自配置，不硬编码。

### 5.2 组件与权重名（Qwen2 布局，与 HF 权重一一对应）

| 组件 | 权重 | 说明 |
|---|---|---|
| Embedding | `model.embed_tokens.weight` | `index_select` 查表 |
| RMSNorm | `*.input_layernorm / post_attention_layernorm / model.norm` | F32 计算 `x/√(mean(x²)+eps)·w` |
| QKV 投影 | `self_attn.{q,k,v}_proj.{weight,bias}` | Qwen2 带 bias；线性层全批执行 |
| RoPE | 无权重 | rotate_half 约定；cos/sin 表按 max_position 预计算并按位置索引 |
| GQA 注意力 | `self_attn.o_proj` | KV 头重复 `group` 次对齐 Q 头序 |
| SwiGLU MLP | `mlp.{gate,up,down}_proj` | `down(silu(gate(x)) ⊙ up(x))`，sigmoid 用 `0.5·tanh(x/2)+0.5` 稳定实现 |
| LM Head | `lm_head.weight` 或 tied | `tie_word_embeddings=true` 时复用 embedding 转置 |

### 5.3 前向接口：混合批（prefill + decode 同批）

模型抽象收敛在 `CausalLm` trait，引擎不接触 Qwen2 内部：

```rust
pub trait CausalLm: Send + Sync {
    fn forward_cached(&self, input: &BatchTokens, caches: &mut [KvCache])
        -> Result<Tensor /* [num_seqs, vocab] 每序列末 token logits */>;
    fn forward_nocache(&self, token_ids: &[u32], positions: &[u32])
        -> Result<Tensor /* [seq, vocab] 全位置 logits */>;
}

pub struct BatchTokens {
    pub token_ids: Vec<u32>,   // 拼接的输入 token
    pub positions: Vec<u32>,   // 每个 token 的绝对位置
    pub seq_lens: Vec<usize>,  // 按序列切分
}
```

`forward_cached` 的批处理策略：**线性层（QKV/o/MLP/LM head）在整批上执行
GEMM，注意力核按序列循环**（每序列独享 KV 缓存）。LM head 只对每序列末行
计算，prefill 大幅节省 logits GEMM。同一个批里可以同时包含
「一个 prompt 的全部 token」与「若干序列的各一个 decode token」，
这正是连续批处理需要的混合批形态。

`forward_nocache` 是设计文档 §16 要求的正确性基线路径（无缓存全量重算），
也是 §52.4 KV 等价性测试的参照。

### 5.4 设备与数值类型

- `--device auto|cpu|metal|cuda`：auto 依 CUDA → Metal → CPU 顺序探测；
  **显式指定的设备不可用时直接报错，绝不静默回退**。
- Metal 通过 macOS 目标默认启用；CUDA 需 `--features cuda` 构建。
- `--dtype auto|f32|f16|bf16`：auto=F32（正确性优先），加载时统一转换。

---

## 6. KV Cache 与内存块管理（mini-vllm-kv）

### 6.1 物理分页与连续参考路径

默认使用 `paged.rs` 的物理 KV 页，各层按页持有 K/V 张量；`cache.rs` 保留连续张量路径，供 `--contiguous-kv` 与数值对照使用。页大小由 `--kv-block-size` 控制。完整页可通过 Arc 共享；独占部分页原地追加，共享部分页写入前复制。attention 直接读取页列表，逐页执行在线 softmax（运行最大值、归一化分母与 V 加权分子），不拼接完整 scores，不拼接完整 K/V。当前使用通用 Candle 算子，未实现融合 GPU kernel。

### 6.2 容量与前缀缓存

活动请求按 horizon 扣除命中且被 pin 的完整前缀后预留逻辑块，准入时逐请求扣减预算；实际页按需分配。`--prefix-cache-tokens` 是活动 KV 预算之外的独立前缀树保留上限（按唯一块记账，淘汰未被 pin 的叶节点），默认 0。缓存仅保留完整块边界的 prompt 前缀；至少留下一个 prompt token 重新计算 logits。模型实例内共享，取消一个序列不会修改其他序列的页。

回滚通过恢复长度并裁剪页表完成；历史页不可变。逻辑块管理器负责保守准入和块号复用，物理页生命周期由引用计数管理，二者不是同一个全局设备内存分配器。临时 attention scores、模型权重不包含在 token 容量预算内。

---

## 7. 采样（mini-vllm-sampling）

显式管线（顺序即实现顺序，全部可单测、可播种复现）：

```text
logits → repetition penalty → temperature(0 ⇒ 贪心 argmax)
       → top-k 过滤 → top-p 过滤 → softmax → 按分布采样
```

- 与张量库解耦：输入 `&[f32]`；执行器对每步的 `[batch, vocab]` logits
  **只做一次** 设备→主机 F32 转换，然后逐行采样（转换失败会让该步失败，
  绝不静默用零 logits 继续）；
- 每请求 `StdRng::seed_from_u64(seed)`：请求带 seed 用之，否则
  `engine_seed ^ counter·φ`，保证可复现性；
- 重复惩罚上下文是引擎侧增量维护的 seen-set（O(1) 插入），不再每
  token 重建+排序整个上下文；
- 数值稳定：softmax 先减最大值；top-p 永远保留最高概率 token；
  top-k 恰好保留 k 个（并列时按 token id 低者优先，确定性截断）。

---

## 8. 引擎与调度（mini-vllm-engine）

### 8.1 命令协议

```rust
pub enum EngineCommand {
    Generate { request: GenerationRequest,
               events:  mpsc::Sender<GenerationEvent> },  // 有界
    Cancel   { request_id: RequestId },
}
```

`EngineHandle`（可 Clone）暴露 `generate() -> Receiver`、`cancel()`、
`metrics()`；generate 使用 `try_send`，cancel 设置独立原子标记，不占命令队列容量。`spawn_engine` 返回 Result 并先验证配置。

### 8.2 调度迭代（每步）

```text
检查取消/断连 → retire → admit → mixed forward → retire → metrics
```

`max_batch_tokens` 是 prefill + decode 的合计 token 预算。decode 优先，预算不足时轮转选择运行序列；预算大于 1 且有 prefill 时给 prefill 至少预留一个 token。长 prompt 分块推进，未完成 prefill 不采样、不推进 RNG。同一模型调用可以包含 decode token 与不同长度的 prefill 分块。

批量失败时先回滚所有 cache 再逐序列重试。`max_waiting_requests` 限制等待队列；独立请求注册表进一步限制命令、运行与输出交付中的请求总数，拒绝重复 ID，并保证取消不依赖命令通道空位。

### 8.3 停止条件与终止

支持 EOS、stop token、跨 token 的 stop string、长度限制、取消与失败。潜在 stop 前缀暂存，正常 EOS/长度结束时释放未匹配尾部。

退休先释放活动 KV，再进入有界的输出交付阶段。默认给 1000ms 排空 outbox，期间不会占用模型槽或 KV；超时或溢出按取消处理。JSON 和 SSE 都不会把失败或截断输出包装成成功。输出交付仍占请求总量许可，因此慢读者不会无限累积。

### 8.4 指标（/metrics，JSON）

```text
requests_total/running/waiting/finished/cancelled/failed
prompt_tokens_total / generated_tokens_total
prefill_steps / decode_steps / 平均 decode 批大小
TTFT 均值 · inter-token latency 均值 · 请求时延均值 · tokens/sec
kv_blocks_total/used/free · 活跃序列数
```

主要计数使用原子量；最近 10 秒吞吐使用最多约 101 个时间桶和短锁。新增 requests_finishing、cached_prefix_tokens、prefix_cache_hit_tokens、scheduled_tokens_total、model_steps_total。tokens_per_second 是近期窗口值，lifetime_tokens_per_second 才是累计生成数/运行时间。tracing span 覆盖
模型加载、准入、prefill、decode step、retire 等关键路径；
默认不打印用户 prompt 内容。

**历史实测，仅代表分页改动前的连续路径，不作为当前版本性能承诺**（Apple M1 Pro，CPU F32，Qwen2.5-0.5B-Instruct，
`scripts/benchmark.py`，max_tokens=48）：

| 并发 | 吞吐 | TPOT 均值 | TTFT 均值 |
|---|---|---|---|
| 1 | 11.1 tok/s | 89 ms | 138 ms |
| 4 | 34.5 tok/s | 109 ms | 393 ms |
| 16 | 58.0 tok/s | 213 ms | 2250 ms |

并发提升总吞吐约 5 倍，正是连续批处理对算力利用率的改善；
总吞吐随批增大而提升的同时，TTFT 上升（prompt 需与在途 decode 竞争
调度）——这就是 §69 学习目标中"调度如何影响延迟与吞吐"的直接体现。
（修复"逐序列重复转换全量 logits"后，4 并发吞吐从 26.7 → 34.5 tok/s。）

---

## 9. HTTP 服务（mini-vllm-server）

### 9.1 端点

| 端点 | 说明 |
|---|---|
| `GET /health` | 存活检查 |
| `GET /v1/models` | 模型列表（id = 模型目录名） |
| `POST /v1/completions` | 文本补全，`stream` 可选 |
| `POST /v1/chat/completions` | 对话补全（ChatML 模板渲染后分词） |
| `GET /metrics` | 引擎指标 JSON 快照 |

支持参数：`model / prompt / messages / max_tokens / temperature / top_p /
stream / stop(单值或数组) / seed`，外加非标准的 `top_k / repetition_penalty`。
**未知字段显式拒绝**（`deny_unknown_fields`，422），不静默假装支持。

### 9.2 流式（SSE）

`text/event-stream`，逐 token 增量 flush，chunk 结构对齐
`chat.completion.chunk` / `text_completion`，终止 chunk 带
`finish_reason`，最后发送 `data: [DONE]`。引擎侧错误以独立的
`{"error":{...}}` 帧上报，绝不伪装成 assistant 内容。

**断连取消**：SSE 场景把 `CancelOnDrop` 守卫移入响应流；非流式场景由
handler future 持有守卫（正常完成后解除，避免无谓的取消命令）。
客户端断开 ⇒ axum 丢弃 body/future ⇒ 守卫 Drop ⇒ 设置取消标记 ⇒
引擎停止调度并释放 KV。推理不会为死连接继续。

### 9.3 输入防护

请求体上限 1MB；prompt + max_tokens > `max_model_len` 返回 400；
长 prompt 按 `max_batch_tokens` 分块；KV 水平超出预算直接拒绝；所有校验发生在模型执行之前。

---

## 10. CLI（mini-vllm）

| 命令 | 作用 |
|---|---|
| `inspect --model <dir>` | 打印架构、层数、hidden、GQA、词表、上下文、权重文件数、张量数、参数量、权重 dtype、计算 dtype、设备——只读 header，不加载权重 |
| `generate --model --prompt` | 引擎 API 驱动（非直连模型），流式打印 token；`--temperature 0` 贪心、`--seed` 复现 |
| `serve --model [--host --port --device --dtype --max-model-len --max-num-seqs --max-batch-tokens --max-kv-tokens --seed --log-level]` | 启动 HTTP 服务；SIGINT/SIGTERM 优雅关闭（停止接新 → 限时排空 → 超时取消 → join 引擎） |

`generate` 走完整引擎路径（队列 → 调度 → 批 → 事件流），是垂直切片
的验收入口，也让 CLI 与服务端共享同一条生产代码路径。

---

## 11. 测试策略与正确性门禁

共 125 个测试（`cargo test --workspace`），关键门禁按设计文档 §52：

| 层 | 测试 |
|---|---|
| core | 请求校验（上下文溢出/空 prompt/非法采样参数） |
| sampling | 贪心确定性、seed 复现、top-k 支撑集**恰好 k 个（含并列）**、top-p 支撑集、penalty 双向缩放、softmax 归一 |
| kv | 原地写入/前缀保持/跨层独立/溢出报错；块分配-释放-复用、失败分配零副作用、双重释放检测、水平门控、用量核算 |
| model | RMSNorm 数值、RoPE（形状/旋转不变性/位置敏感/确定性）、GQA 形状、causal mask、批解码==逐序解码、**KV 等价性**（§52.4：缓存 logits ≈ 无缓存 logits，1e-4 容差）、repeat_kv 头序 |
| engine 单元 | 调度准入 FIFO/max_num_seqs/门控阻塞、批构造位置推算、emit 背压（满→排队→溢出→取消）、通道关闭检测 |
| engine E2E | 真实引擎线程 + 随机小模型：生成直到 Length（跨引擎确定性）、stop token 立即停止、超 max_model_len 拒绝（超预算 prompt 则分块推进）、首 token 后取消、**慢消费者被取消而引擎不卡死**、并发独立完成且各模式内确定、KV 块归零、prefix 复用输出一致且保留有界、显式关闭（Drain/Cancel + join）、输出宽限期与预留全回收 |
| tokenizer | 增量解码拼接==全文、滑动窗口不随输出增长、stop string 截断增量与最终文本 |
| server | mock 引擎下的 /health、/models、completions/chat（流式 SSE 帧结构、`[DONE]`）、未知字段拒绝、超长 prompt 400、非法 role 400 |

**等价性优先**：随机小模型（`testutil::random_model`，2 层 GQA tied）上
验证「缓存==无缓存」「批==单序」（模型层，容差可控）；引擎 E2E 层验证
调度独立性与各模式内确定性——批 GEMM 浮点顺序可使近并列 argmax 翻转，
solo 与 batched 的贪心链因此允许合法分叉（vLLM 同样如此），精确等价由
模型层测试把关。任何影响数值的重构都必须先过这两关。

---

## 12. 与 vLLM 概念对照

| vLLM 概念 | 本项目实现 | 状态 |
|---|---|---|
| 分页 attention | 物理页、页读取与跨页 softmax | ✅ 通用算子，未融合 GPU kernel |
| Continuous batching | 每迭代 retire/admit/prefill/decode | ✅ |
| 混合 prefill+decode 批 | 单个 BatchTokens 混合批 | ✅ |
| Chunked prefill | 合计 token 预算下分块推进 | ✅ |
| Prefix caching | 独立预算、按块前缀树去重、叶节点 LRU、不可变页共享 | ✅ 可选 |
| OpenAI server | axum + SSE | ✅ 子集 |
| Scheduler / admission | FIFO + KV 门控 + max_num_seqs | ✅ |
| Metrics（TTFT/TPOT 等） | 原子计数 + /metrics JSON | ✅（Prometheus 后续） |

---

## 13. 已知限制与路线图

**当前限制**

- decode 批的线性层已批处理，但注意力核按序列循环（无跨序列融合 kernel）；
- KV 默认物理分页，仍使用保守 horizon 准入；前缀预算与活动预算分离；
- 权重默认转 F32 计算（`--dtype bf16/f16` 可选）；未做量化；
- 仅 Qwen2/2.5 架构（Llama/Mistral 等 decoder-only 可按 `CausalLm` 扩展）；
- 指标为 JSON 快照，未导出 Prometheus 文本格式。

**路线图（按依赖序）**

```text
跨序列批注意力 / 融合分页 GPU kernel
→ F16/BF16 全链路调优 → GGUF/量化
→ 动态 KV 准入/抢占 → speculative decoding
→ 更多架构（Llama / Mistral / Gemma / Phi）
```

---

## 14. 代码质量约定

每次交付前必须通过：

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings   # cuda 特性需 CUDA 工具链，本地验证用默认特性
cargo test --workspace
```

禁止：运行时 `unwrap()`、无强不变量的 `expect()`、无界通道、
async 处理器中的阻塞调用、跨模型执行的锁持有、静默吞错。
优先：小类型模块、显式状态机、有界队列、结构化错误、tracing span、
围绕调度与内存所有权的测试。

实现与独立参考模型验证见 [IMPLEMENTATION.md](docs/IMPLEMENTATION.md)。原始规格文件保留为历史设计输入。

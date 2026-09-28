# Trace shutdown and prefix metadata / 追踪关闭与前缀元数据

Trace events still use a bounded nonblocking queue. Dropping the writer closes that queue and waits up to 250 ms for the worker to flush. Normal shutdown drains queued events. A blocked filesystem write cannot be cancelled safely; on timeout the worker is detached, shutdown continues, and the trace may be incomplete. Library callers may use `TraceWriter::shutdown(Duration)` for an explicit budget; its boolean reports completion within that budget, not write success.

追踪关闭最多等待 250 毫秒。正常情况下会写完队列并 flush；文件系统阻塞时会记录超时并让关闭流程继续，文件可能不完整。底层阻塞 I/O 不会被强行取消。

`/metrics` exposes `trace_writer_errors` (failed/panicked workers), `trace_shutdown_timeouts` and `trace_events_dropped`. The error counter updates even if no further event is emitted. Timeout counters remain readable through a retained engine handle after shutdown; the HTTP endpoint itself may already have stopped. These are separate signals: dropped-event counts do not include every event lost to an I/O failure or timeout.

这些指标分别反映写入失败、关闭超时和队列丢弃，不应把队列丢弃数当作所有缺失事件的总数。HTTP 服务关闭后，可通过保留的引擎句柄检查最终指标。

Prefix keys now use shared token slices, with borrowed lookup and no temporary token-vector allocation. A separate ordered leaf index tracks LRU age. Eviction visits leaves in age order and skips pinned/protected leaves; it no longer scans interior nodes. Worst-case work still scales with the number of pinned leaves. When eviction makes a parent a leaf, the parent keeps its original access age.

前缀查找不再复制 token 块；淘汰只扫描按访问顺序排列的叶子。被请求占用的叶子仍受保护，所以最坏情况下仍需跳过所有被占用叶子，不能宣称淘汰永远是常数时间。

Run the independent experiment without model files:

```bash
cargo run --release -p mini-vllm-engine --example prefix_metadata
```

It checks 16–1024 blocks, linear key/page-reference counts, zero allocations across 1,000 lookups, and replacement of a deep trie. Logical metadata counts are not allocator bytes, process RSS, or device VRAM. Timings depend on host load. Regular correctness CI runs the allocation/scaling assertions as well.

For paired end-to-end trials:

```bash
python3 scripts/compare_kv.py --model models/qwen2.5-0.5b-instruct --modes prefix --trials 3 --trace-matrix --prompt-repeat 10 --requests 4 --max-tokens 4 --output /tmp/paired.json
```

Trace off/on order alternates between trials, and mode order rotates. JSON preserves per-request samples and nearest-rank P50/P95/P99 for latency, TTFT and token-event intervals. Sparse samples cannot support tail-latency or statistical-significance claims; model-loading RSS remains a separate measure.

# GPU validation / GPU 验证

The [backend script](../scripts/validate_backends.py) records platform and GPU inventory, UTC time, Git revision and dirty state, Rust version, fixture/test-source hashes, commands, durations and complete test output. A build failure, unsupported operation or invisible device is a failure, never a passing skip. Numerical tests compare full, contiguous-cache and paged-cache inference against the committed Transformers oracle. Lifecycle tests exercise prefix reuse, forced eviction and cold/warm recomputation, cancellation, disconnects and final resource reclamation.

验证报告记录设备清单、版本、源码与模型摘要及完整命令。设备不可见、构建失败和不支持的运算都算失败。生命周期验证还会强制淘汰前缀，再对照冷启动和缓存复用的输出及资源归还。

```bash
python3 scripts/validate_backends.py --devices cpu --dtypes f32 f16 --output /tmp/backend-cpu.json
python3 scripts/validate_backends.py --devices metal --dtypes f32 f16 --output /tmp/backend-metal.json
python3 scripts/validate_backends.py --devices cuda --dtypes f32 f16 --output /tmp/backend-cuda.json
```

The [manual GPU workflow](../.github/workflows/backends.yml) requires an existing trusted self-hosted runner labelled `metal` or `cuda` (in addition to `self-hosted`), with the corresponding GPU, drivers/toolkit, Python 3 and Rust toolchain prerequisites installed. It only runs `main`, has no pull-request trigger and saves the report for 30 days. Adding the workflow does not provision GPU hardware; do not dispatch it without a matching runner. On macOS, use a login session in which Metal devices are actually visible, not merely a host reporting Metal support.

手动工作流只运行 main，需要已有且可信的 GPU runner；配置文件不会自动创建硬件。没有对应 runner 时不要触发并等待空队列。macOS 的系统硬件清单显示支持 Metal，不代表当前进程实际能使用该设备。

This round's local verification and remaining hardware gap are recorded in [the support matrix](benchmarks/backend-matrix-2026-09-28.md).

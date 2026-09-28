# Backend support evidence / 后端验证证据

Date: 2026-09-28. Host: macOS 14.8.9, Apple M1 Pro (14 GPU cores, Metal 3 reported by the system).

| Backend / dtype | Numerical reference | Prefix reuse + eviction + cancellation + reclamation | Evidence |
|---|---|---|---|
| CPU / F32 | PASS | PASS | Executed locally |
| CPU / F16 | PASS | PASS | Executed locally |
| Metal / F32 | FAIL: device unavailable | FAIL: device unavailable | Metal feature explicitly enabled |
| Metal / F16 | FAIL: device unavailable | FAIL: device unavailable | Metal feature explicitly enabled |
| CUDA / F32, F16 | NOT RUN | NOT RUN | No accessible CUDA host/toolchain supplied |

[Raw report](backend-validation-2026-09-28.json) includes commands, full failure output, hardware inventory, test-source hashes and fixture hash. It records the base Git revision plus a dirty working tree because validation ran before these changes were committed. The explicit test-source hashes identify the tested test code.

本轮确认了 CPU 路径。系统能列出 M1 Pro，并不代表当前进程能创建 Metal 设备；四项 Metal 检查均明确失败，没有当作跳过成功。没有提供 CUDA 环境，所以 CUDA 标为未执行。新增工作流只是复用验证的入口，不能代替真实 GPU 通过记录。

Actual GPU correctness remains unverified. To complete that part, run the [documented commands or manual workflow](../gpu-validation.md) on an accessible GPU host and attach the resulting report. Do not claim GPU support as validated solely because the tests compile or CPU runs pass.

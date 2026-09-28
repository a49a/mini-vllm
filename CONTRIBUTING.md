# Contributing / 参与贡献

Thank you for helping improve this educational inference engine. Please keep changes focused on one behavior or learning objective. The [design roadmap](DESIGN.md#13-已知限制与路线图) describes larger extensions; an issue is a good place to discuss their scope before a large implementation.

欢迎改进这个教学用推理引擎。请让每次修改聚焦一个行为或教学目标。大型功能可以先在 issue 中讨论范围。

## Local setup / 本地环境

Install Rust 1.87 or newer and Python 3.10 or newer. The default test suite uses tiny checked-in fixtures; the full Qwen2.5 model is optional and belongs in the ignored `models/` directory. Do not commit model weights, API keys, request traces containing private IDs, or generated benchmark logs.

安装 Rust 1.87 及以上版本和 Python 3.10 及以上版本。默认测试使用仓库内的小型样例；完整模型不是必需品，也不应提交到仓库。

```bash
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked --workspace
python3 -m unittest discover -s scripts/tests -v
python3 scripts/check_docs.py
```

For parser robustness work, install nightly Rust and `cargo-fuzz`, then run the targets in [`fuzz/`](fuzz/README.md). For backend changes, run `scripts/validate_backends.py` on the actual device and include the device, dtype, command, and result in the pull request. Unavailable hardware should be reported as untested rather than passed.

解析器健壮性修改请运行 `fuzz/` 中的目标；设备相关修改请在真实设备上记录验证结果。无法使用的设备应标注“未验证”。

## Pull requests / 合并请求

Describe the trigger, observed behavior, resulting behavior, and validation. Add tests for correctness or resource ownership changes. For performance claims, include the exact model, device, dtype, workload, warmup, commands, and raw data; report failed requests separately. Keep documentation aligned with current code and distinguish a portable reference path from an optimized kernel.

请说明触发条件、修改前后的行为及验证方式。涉及正确性或资源所有权时添加测试；性能结论需附可复现命令和原始数据，并单列失败请求。

Report suspected security issues using [SECURITY.md](SECURITY.md), rather than including exploit details in a public issue.

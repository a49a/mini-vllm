# Security policy / 安全策略

This is an educational, single-replica inference server. It has no authentication or transport security and defaults to `127.0.0.1`; do not expose it directly to an untrusted network. Support is limited to the current `main` branch because there are no maintained release branches.

这是教学项目，不提供身份认证或传输加密，默认只监听 `127.0.0.1`。请勿直接暴露到不可信网络。目前仅维护 `main` 分支。

For a suspected vulnerability, use GitHub's private **Report a vulnerability** flow on this repository if available. If that option is unavailable, open a public issue asking the maintainer to provide a private reporting channel **without including exploit details, private data, or a proof of concept**. Include affected commit or version, impact, and reproduction steps in the subsequent private report. Please allow time for a fix before public disclosure.

如发现疑似漏洞，优先使用仓库的 GitHub 私密漏洞报告入口。如入口不可用，可先开一个不含漏洞细节的 issue，请维护者提供私密沟通方式；随后私下提供受影响版本、影响范围与复现步骤。

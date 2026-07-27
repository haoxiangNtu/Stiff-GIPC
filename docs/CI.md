# 验证策略：推送时门禁（pre-push hook）

**拥有者定版（2026-07-27）**：不做后台轮询/定时——验证只在**真正推送**时发生。

- `git push`（分支）→ pre-push 钩子自动跑全武装 12 段门禁（~25 分钟），
  **任何一门红 = 推送被拦**。
- `git push` 携带 **tag**（= 发版时刻）→ 额外跑重炮包：FD 门 + 19 demo 全扫
  + 三模式矩阵 + sanitizer，全绿才放行。
- 紧急逃生口：`SKIP_GATES=1 git push`（仅限救火，会打日志）。

钩子本体版本化在 `scripts/hooks/pre-push`，经 `git config core.hooksPath
scripts/hooks` 生效（新 clone 需执行一次这条 config，写进上手文档即可）。

附录：`scripts/ci_watch.sh` / `nightly.sh`（轮询式 runner）保留在仓库但
**不启用**——将来若想要持续后台验证，装回 crontab 即可（见脚本头注释）。

# CI（本地 runner 版）

**机制**：cron 每 10 分钟跑 `scripts/ci_watch.sh`——`origin` 上被监视分支
（默认 `internal/v0.8.6-rc1`、`refactor/v086-energy-modular`）一有新提交，
就在**专用 CI worktree**（`~/Downloads/Stiff-GIPC-ci`，与开发树完全隔离）
构建并跑全武装 12 段门禁；结果写 `~/stiffgipc-ci/ci-history.log` 与
`ci-status.md`，并 `notify-send` 弹桌面通知。夜里 02:30 `nightly.sh` 跑重炮：
FD 门 + 19 demo 全扫 + 三模式矩阵 + sanitizer。

- 状态一眼看：`cat ~/stiffgipc-ci/ci-status.md`
- 手动触发：`scripts/ci_watch.sh --once internal/v0.8.6-rc1`
- 改监视分支：cron 行加 `CI_BRANCHES="a b"`

**升级为 GitHub Actions（可选）**：仓库 Settings → Actions → Runners →
New self-hosted runner，把注册命令在本机跑一遍（需要你登录 GitHub 拿
token），然后加 `.github/workflows/gates.yml`（runs-on: self-hosted，
steps 调 verify_gates.sh）。收益=提交页原生绿勾/红叉；本地 poller 可共存。

# Stiff-GIPC v0.8.6-rc1（内部测试版）

**性质**：内部测试用 release candidate，非对外发布。分支 `internal/v0.8.6-rc1`，
tag `v0.8.6-rc1-internal`，仅推私有仓（对外镜像不动）。

## 主线：v0.8.6 模块化重构（27.6k 行单体 → 语义模块 + 物理 TU）

- **Phase 1/1b**：四单体 #include 复合拆分（字节等价断言），gipc/mlbvh/mas/engine
  全部语义模块化
- **Phase 2a–2d**：归约模板统一（22/25 kernel）、pair-buffer 增长机制单属主
  （顺手抓获 per-env CCD 镜像缩水潜伏 OOB）、隔离铁律机制单属主+契约首次成文、
  帧编排单属主 + `core/frame_pipeline.h` 帧序唯一权威
- **物理 TU 分离**：core/ipc_solver、contact/pair_buffers、multienv/isolation +
  **能量层 E1–E3 全竟**：九项能量各自独立 .cu（kinetic/fem_elastic/membrane/
  bending/soft/barrier/friction/ground/delta），"加本构 = 加文件 + 注册表一行"
  （`GIPC_ENERGY_TERMS` X-macro 单源）。融合装配（energy/03）留复合 TU=成文例外
- **锚不动纪录**：全部语义/物理阶段 strict 锚 `f7fb5a786c2d7935` 逐位不变，
  含跨架构（4090 sm_89 ≡ A800 sm_80）

## 结构性疫苗（P3 四针 + 扩面）

- `contact/encoding.h` int4 编码契约真源；`mode_contract.h` 三模式承诺表；
  `frame_pipeline.h`；隔离契约
- `HostMirror`（镜像保鲜审计）+ `DeviceBuffer`（RAII）+ 槽位契约审计 +
  discard 合法性单发窗口——**全套武装常驻门禁**
- 门禁体系 12 段：G0 真实构建 + G0.5 冻结绊线 + 八门 + G9 三模式门 +
  G10 坏网格 ABD 门

## 关键修复

- **towel-strict 崩溃根因**（pre-solve discard 毁活跃矩阵）→ preserve/discard
  契约 + 帧首合法窗口
- **case-26 布不掉/能量 NaN**（本 rc 新修）：坏网格 ABD 体经 PSD 钳制后质量
  矩阵奇异 → `M⁻¹` 产 NaN q̃；修复=特征值正 ε 底板。根子早于重构、由 NaN
  拒绝修复（fa0cd63）暴露。永久防线：per-body 哨兵 + 15 槽能量 dump + G10
- 审计批修复（NaN 线搜索判定、ExclusiveSum 时序等六项）+ 五镜头七缺陷批

## 三模式验证（本 rc 的验收依据）

双平台（4090+A800）33 格矩阵全绿：strict 位级（r2r+跨架构+双配置路径互证）、
isolated 铁律隔离（漂移 0）、merged 吞吐（契约成文 fail-fast）。结构发现：
峰值迭代跨架构逐一相同；strict 峰值系统性最低；isolated 布料场景常最快且不触
cap。详表：`docs/MODE_MATRIX_REPORT_2026-07-27.md`。
自动化：`scripts/demo_verify.py` 10 headless demo 全绿；UI demo 拥有者 13 连逐个目验通过（软爪六件套、整机/全尺度/统一场景、finray UI 三连；含 case-26 修复复验）。

## 已知事项 / 冻结区

- smooth/mollifier 分支与 close-set 链按指令冻结（G0.5 绊线在钉）
- 融合 barrier 装配仍复合 TU（后续战役）
- foldshirt 类场景峰值=newton_iter_cap(50) 系脚本设定（盘子 228f 无触顶）
- UI 软爪 demo 的 headless 移植未做（由 finray replay 家族承担自动化）

## 部署

wheels：cp310（本地）+ cp311（A800，sm_80/89/120）。A800 bundle 配方与坑
（npz 双件、盘子 episode 走 /data + STIFF_REPLAY_PATH、cwd=/data/stiff-physics）
见记忆/报告。

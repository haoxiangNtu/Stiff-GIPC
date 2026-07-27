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
自动化：`scripts/demo_verify.py` **19 个 headless demo**（含 replay 轨迹家族九项：case39 双形态、UMI 四变体、finray 单/多 env、diag）全绿；UI demo 拥有者 13 连逐个目验通过（软爪六件套、整机/全尺度/统一场景、finray UI 三连；含 case-26 修复复验）。

## 已知事项 / 冻结区

- smooth/mollifier 分支与 close-set 链按指令冻结（G0.5 绊线在钉）
- 融合 barrier 装配仍复合 TU（后续战役）
- foldshirt 类场景峰值=newton_iter_cap(50) 系脚本设定（盘子 228f 无触顶）
- UI 软爪 demo 的 headless 移植未做（由 finray replay 家族承担自动化）

## 部署

wheel：cp311（sm_80/89/120；包声明 requires-python>=3.11，cp310 本地开发直接用 build/ 树不经 wheel）。A800 bundle 配方与坑
（npz 双件、盘子 episode 走 /data + STIFF_REPLAY_PATH、cwd=/data/stiff-physics）
见记忆/报告。

---

# 附录：rc1 后加固合并（6c562bc + 补充，未随 rc1 tag 发布）

外部审核分支 audit/codex-fable-review-20260727 经三路对抗审计后合入。要点与
**调用方可见的行为变更**：

## 行为变更（升级注意）
- 性能计数 getter（total_newton_iters/pcg_iters/collision_pairs/max_collision_pairs/
  frames_done）从进程级累计改为 **per-Engine 实例**；依赖跨 reset 累计值的调用方
  读数会变。
- `get_fem_von_mises_stress` 按编译期本构（SNK1/SNK2/ARAP）计算，不再硬编码
  Neo-Hookean；两个接触力/导出 getter 改为**输入序**返回（与
  `get_vertex_positions()` 对齐）。
- `STIFF_*` 旗标解析统一为**值感知**（`=0`/空串=关）；标准 resolver 只写 "1"
  或删除变量，不受影响；手写 `=0` 当"开"用的环境会翻转。
- 单进程单 Engine 强制（RuntimeOwnerLease）：先前静默共享可变缓冲的多 Engine
  用法现在显式抛 LifecycleError；reset() 后可重建。
- 侵入式 FD API 移出生产 ABI（diagnostics 构建才有）。checkpoint v1（STKP）移除，
  由带 CRC64/事务校验的 v2（STIFFCP2）替代。

## 修复定性更正（相对该分支自述）
- g_vloc：基线树中为 **泄漏 + 陈旧全局设备指针**（仅 STIFF_EE_CANON 路径可触发，
  非默认公开 API 路径）；字面 use-after-free 是该提交自身补加的 `release(m_d_vloc)`
  才会引入的，修复（buildCP 顶部无条件重发布）与该 free 正确同船。合入后闭环。
- “三模型 FD 验证”原实为三模型不变量验证 + SNK1 FD；本合并补 `FD_MATRIX=1`
  重炮腿后，三模型解析梯度均经 FD 对照（见 PHYSICS_VALIDATION.md）。

## 工程裁决存档
- 融合 barrier 核 TU 抽取被性能带否决（双盲复现：+9.5% SASS、栈帧 +8KB、
  460B/线程溢出、--maxrregcount=200 恶化至 1072B）；裁决注释入
  energy/03_barrier_fused_assembly.inl 头部。
- 性能基准协议：median-of-3，首跑受时钟爬坡/冷缓存污染约 10%，禁止单跑定带；
  GPU 非空闲时 mode_bench rc=2 fail-closed（tag 门禁 INFRA-BLOCK）。

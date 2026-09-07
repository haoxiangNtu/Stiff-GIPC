# 待核实项登记表(OPEN_POINTS)

> **用途**:全手册待核实项的**唯一登记表**。各分册的"待核实"专节与行内标注
> (README §7、API_CORE 行内+附录、API_EXECUTION 附录 A、PRINCIPLES_CONTACT §9、
> PRINCIPLES_DYNAMICS 附 C、PRINCIPLES_EXECUTION 行内、CHANGELOG_TIMELINE §8、
> KNOWN_ISSUES §6)共约 45 处标记在此收拢为 **37 条编号条目**(4 组跨文档重复各并为一条)。
> **关闭一条时引用其编号**(如"OP-018 已关闭:Release 页逐字节核对,见 commit xxx"),
> 把状态改为 `closed` 并附证据;各分册原位标注可保留,以本表状态为准。
> 汇总口径出自 [`../manual_review_gaps.md`](../manual_review_gaps.md) §4(2026-09-07)。

状态取值:`open`(未决)/ `closed`(已关闭,附证据)。
**当前统计:37 条 = 34 open / 3 closed**(最近关闭:OP-001、OP-018,2026-09-08;OP-003,2026-09-07)。

---

## A. 跨文档重复项(已合并,每条一处登记)

| 编号 | 描述 | 出处 | 验证方法建议 | 状态 |
|---|---|---|---|---|
| OP-001 | 稳定仓 tag `v0.8.5.4`(absolute_epsv / friction_anchor 真静摩擦)是否已对外发布 wheel / 是否已撤回 | README §7#2、API_CORE 版本口径、API_EXECUTION 附 A#2、KNOWN_ISSUES §6#2、CHANGELOG_TIMELINE §8#1、PRINCIPLES_EXECUTION §0 | **已关闭(2026-09-08):v0.8.5.4 已正式发布,未撤回。**`gh release view v0.8.5.4 --repo haoxiangNtu/stiff-physics` 亲验:`published: 2026-08-11T17:07:37Z`、`draft: false`、`prerelease: false`,资产 `stiff_physics-0.8.5.4-cp311/cp312-linux_x86_64.whl` **两个 wheel 均已挂出**;公开仓 README 安装 URL 已由提交 `a38ede4`("Bump install URLs to v0.8.5.4 (true static friction default-on)")指向 v0.8.5.4 | closed |
| OP-002 | phase-cd 摩擦读数恒零(friction_lagged 分量)的**运行时**实测确认——代码结构已证(无 `snapshotFrictionForce`、accessor 提交后现算) | KNOWN_ISSUES §6#3、PRINCIPLES_CONTACT §9#5、CHANGELOG_TIMELINE §8#5、PRINCIPLES_DYNAMICS 附 C#11、API_EXECUTION 头部 | phase-cd 构建上跑一次剪切台实验,确认 `get_vertex_contact_forces(components=1)` 恒零;移植后以稳定线滑比 0.600±0.008 @ μ=0.6 为验收基准 | open |
| OP-003 | `fix_obj_winding.py` 的实际位置(`case_26_perf_tuned.py` docstring 写 `examples/` 前缀,该处不存在) | README §7#3、API_EXECUTION 附 A#7 | 已亲验:文件在**两树** `tools/fix_obj_winding.py`,docstring 路径前缀过时(manual_review_gaps §2.2#2) | closed |
| OP-004 | `replay_case39_UMI_obb_cup_shirt_forcegrip.py` 代码默认 `CASE39_GRIP_MODE="trackgrip"` 的语义,与 docstring 五模式列表不一致(旧线对照示例,低优先) | README §7#4、README §6 | 读脚本 trackgrip 分支实现,与 docstring 对照后二选一修正 | open |

## B. API/语义类

| 编号 | 描述 | 出处 | 验证方法建议 | 状态 |
|---|---|---|---|---|
| OP-005 | `velocity_damping` 是否作用于 FEM 顶点(勘探只见 ABD 消费点) | API_CORE §2.1、PRINCIPLES_DYNAMICS §6.7/附 C#10 | grep FEM 积分路径的消费点;或纯 FEM 场景设大阻尼观察速度衰减 | open |
| OP-006 | `load_mesh_from_data` dim=3+vpf=3+FEM 的 "internally tetrahedralize" 声明疑似不可用(未找到实现) | API_CORE §3.3 | 用三角表面网格 + `body_type="FEM"` 实际调用一次,观察报错/行为;grep 四面体化实现 | open |
| OP-007 | `/tmp/stiffgipc_mesh_data/` 固定目录的多进程同名竞态 | API_CORE §3.3 | 审阅写入路径是否含 PID/随机后缀;两进程并行加载同名网格复现互踩 | open |
| OP-008 | `set_body_animated_target` 的能量门控经公开 loader 疑似永不触发(Animated boundary_type 不可达) | API_CORE §3.2/§10.4、PRINCIPLES_DYNAMICS 附 C#8 | grep 公开 loader 可产生的 boundary_type 集合;最小脚本调用并观察 animated 能量项是否激活 | open |
| OP-009 | fixed joint 非正交 n/b 输入的退化行为 | API_CORE §4.2 | 构造非正交 n/b 最小场景,观察约束行为(静默扭曲/报错) | open |
| OP-010 | `set_per_tet_young_for_body` 在 MAS 预条件下的 tet 序对应关系 | API_CORE §5.2 | MAS 下逐 tet 设杨氏并读回/染色可视化,与用户 tet 序对照 | open |
| OP-011 | stitch 能量核 vs G/H 核目标不一致的实际影响(有 pin+大旋转+非零 offset 时) | API_CORE §4.8、PRINCIPLES_DYNAMICS §8.1/附 C#5 | 构造 pin+大旋转+非零 offset 场景,比对能量残差与 Newton 收敛行为 | open |
| OP-012 | phase-cd 无独立 Kappa 清理入口的残留影响量级(~0.9 µm 系推定) | API_CORE §9.5、KNOWN_ISSUES §1.2 | phase-cd 上 episode 就地重置 vs 新进程对照跑,量化位移发散 | open |
| OP-013 | `get_body_contact_force` 的 ground 口径(代码含 ground,docstring 说无) | API_EXECUTION 附 A#3、PRINCIPLES_CONTACT §9#7 | 单体落地、无自碰场景读 body contact force:非零即含 ground;据结果修 docstring | open |
| OP-014 | 稳定线 getter 是否带 metis 解扰(两线输出顶点序可能不同) | API_EXECUTION 附 A#4、PRINCIPLES_CONTACT §9#8 | 两线同场景同序加载,逐字节比对 `get_vertices()` 输出序 | open |
| OP-015 | per-env 遥测在纯设备快路径下的返回值(推理未运行验证) | API_EXECUTION §1.8/附 A#1 | `prepare_gpu_rl` 快路径跑一段,读 per-env 遥测字段与宿主通道对照 | open |
| OP-016 | `m_avg_env_bbox2` 回退分支的触达条件 | API_EXECUTION 附 A#5 | 审阅该成员全部写点/读点;构造空 env 或极端 bbox 场景触发回退分支 | open |
| OP-017 | gpu_rl `joint_observations` 的字节级打包布局 | API_EXECUTION 附 A#6 | torch 零拷贝视图读出与宿主 getter 逐字段比对 | open |
| OP-018 | v0.8.5.3 wheel Release 资产文件名按 v0.8.4 发布页模板推断,下载 URL 未逐字节验证 | README §7#1 | **已关闭(2026-09-08)**:`gh release view v0.8.5.3 --repo haoxiangNtu/stiff-physics` 亲验资产 = `stiff_physics-0.8.5.3-cp311-cp311-linux_x86_64.whl` / `-cp312-cp312-`(v0.8.5.4 同模板),README §3.1 的下载 URL 与之逐字节一致 | closed |
| OP-019 | `newton_velocity_tol` 的 uipc 参考默认 0.05 只来自注释 | API_CORE §2.1 | 查 uipc 上游源码/文档确认默认值 | open |

## C. 原理/机制类

| 编号 | 描述 | 出处 | 验证方法建议 | 状态 |
|---|---|---|---|---|
| OP-020 | 地面 barrier/λ 用 RANK-1 而自碰 RANK-2 的设计意图(附带:地面摩擦 C0 vs 自碰 C1) | PRINCIPLES_CONTACT §9#1、#2、PRINCIPLES_DYNAMICS 附 C#6 | 查上游 GIPC 论文/提交历史;或向 owner/上游确认是否有意设计 | open |
| OP-021 | `_cpNum[1]` 槽位无独立消费 | PRINCIPLES_CONTACT §9#3 | grep 两树全部读点确认;若确系死槽,在册标注或清理 | open |
| OP-022 | 稳定线 mlbvh 发射 smooth=false 未逐行复核(与 phase-cd 同谱系推定) | PRINCIPLES_CONTACT §9#4 | 稳定树 grep mlbvh 发射点,逐行核对 smooth 实参 | open |
| OP-023 | "folded 布料 44 万 mollify 请求 / 0 执行"数字未复测(机制已证:请求被计数、smooth=false 使执行为零) | KNOWN_ISSUES §6#1 | folded 布料场景加计数器复跑一次,核对量级 | open |
| OP-024 | 0.792 暂态峰值原始日志未存档;剪切台界面摩擦构成(几何平均假设 vs 0.600 吻合)存疑 | PRINCIPLES_CONTACT §9#6/§5.1 | 复跑剪切台存档逐帧日志;核对界面摩擦系数构成假设 | open |
| OP-025 | 418 N / 10.5 µm stitch 滞后实测未归档 | PRINCIPLES_DYNAMICS §8.2/附 C#13 | 复跑对应 stitch 滞后实验,把日志归档入库 | open |
| OP-026 | binned K/W/E0 覆盖窗外的溢出/下溢行为 | PRINCIPLES_EXECUTION §4.1 | 构造窗外参数值跑 binned 路径,观察 clamp/溢出行为 | open |
| OP-027 | `Cub_PCG_DotReduction` 的确定性依赖 CUB 实现惯例(非成文契约) | PRINCIPLES_EXECUTION §2.2 | 查 CUB 文档/源码确认 reduction 顺序保证;或换 CUB 版本对照金锚 | open |
| OP-028 | `PCGSolverConfig.use_bsr` 疑为死字段 | PRINCIPLES_EXECUTION §2.1 | grep 全部读点;确认后删除或标注为保留字段 | open |
| OP-029 | 非图 strict 宿主槽序可重复的机制保证(当前为经验性质) | PRINCIPLES_EXECUTION §3.4(f) | 审阅槽位分配代码路径给出机制论证;或多轮 run-to-run 位级对照加固经验证据 | open |
| OP-030 | 稳定线金锚是否 = `f7fb5a786c2d7935`(世系推断,稳定树内未跑门禁验证) | PRINCIPLES_EXECUTION §4.2 | 稳定树跑一次 strict 门禁,比对输出锚值 | open |

## D. 时间线/实测口径类

| 编号 | 描述 | 出处 | 验证方法建议 | 状态 |
|---|---|---|---|---|
| OP-031 | beaker +7% 破案的证据锚:提交史无对应破案提交,"每帧持平"的原始逐帧数据未定位 | CHANGELOG_TIMELINE §8#2 | 定位原始逐帧数据文件;或复跑 beaker 交错对照重采启动段/每帧数据 | open |
| OP-032 | 成对审计 "B' 9342±230 vs 宿主 8808±60(15σ)" 与 "+6%/+4%" 细分出自提交正文与工作记忆合读,未逐字核对 | CHANGELOG_TIMELINE §8#3 | `git show 4219f37` 核对完整 body 后再引用 ±/σ 数字 | open |
| OP-033 | tag 打在 release 提交后一条的惯例(三例)是否有意 | CHANGELOG_TIMELINE §8#4 | 向 owner 确认;`git log` 核对各 tag 落点是否成惯例 | open |
| OP-034 | BVH 战役分支(`codex/bvh-full-campaign`)~20 条 validation 提交的归属未逐条核对 | CHANGELOG_TIMELINE §8#6 | `git log` 该分支与主线逐条比对归属 | open |
| OP-035 | ~30 条纯 docs 提交只核对了标题/摘要,内部数字未回查 | CHANGELOG_TIMELINE §8#7 | 引用其内部数字前按 hash `git show` 回查全文(如 `79f88f4` episode 经济学数值表) | open |
| OP-036 | RL 微步 19.3/9.7/3.85 ms/步的平台标签在 A800_ALLEXAMPLES 与 SIMULATOR_EXECUTION_DESIGN 之间冲突(手册已取 A800,待 owner 确认) | CHANGELOG_TIMELINE §8#8、README §1 | 向 owner 确认 A800_ALLEX 该节 "(4090, clean GPU)" 标题是否笔误;或定位原始 bench 日志的平台记录 | open |
| OP-037 | 稳定线单体行号(stable `GIPC.cu:15221` line-search WARN、`:10318` grow-redo 等)出自当日勘探,复引前建议复核 | KNOWN_ISSUES §6#5 | 稳定树 grep 对应符号/字符串,刷新行号后再引用 | open |

---

*维护约定:新增待核实项在此表追加编号(不复用已关闭编号);关闭时改状态为 `closed`
并在"验证方法建议"列改写为关闭证据(命令/commit/文档行号),分册原位标注随后清理。*

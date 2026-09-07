# 手册完整性审查报告(manual_review_gaps)

> 审查对象:`docs/manual/` 全部 8 份分册(README / API_CORE / API_EXECUTION / PRINCIPLES_DYNAMICS /
> PRINCIPLES_CONTACT / PRINCIPLES_EXECUTION / CHANGELOG_TIMELINE / KNOWN_ISSUES,共 6494 行)。
> 审查视角:一名要独立使用/维护 StiffGIPC 的资深工程师。
> 审查日期:2026-09-07。除标注"推断"外,本文所有断言均经文件/代码/git 亲验
> (亲验命令与路径随条目给出)。
>
> 总评:这套手册在**已覆盖面上的质量异常高**——行号出处、两线标注、"待核实"纪律、
> 对既有证据文档过时项的主动纠偏(beaker +7% 破案、GRAPH_DEFAULT_ON 过时倍率、
> handbook 三处过时)都做到了。缺口集中在**手册边界之外**:Python 便利层
> (Robot/Pipeline/USD)零覆盖、7 个幻影分册名造成 14 处断链、一处自相矛盾的运行指令、
> 维护者发布/门禁旅程缺章、40+ 条待核实项散落无汇总。

---

## 1. 图件(docmap)/仓库里有、而手册全集漏掉的重要事实

### 1.1 【最大缺口】`Robot` 类与 `Pipeline` 类零文档化

- `stiff_physics/__init__.py` 的 `__all__` 导出 `Robot / JointInfo / Pipeline`(两线同);
  手册 README §1 的定位句自己写着"`stiff_physics` 包(`Config` / `Engine` / `Robot`)提供
  Isaac 风格的 `step()` 接口",API_CORE §1.2 的导出表也列了 `Robot`、`Pipeline`——
  **但 8 份分册没有任何一节、任何一个条目讲这两个类的 API**。
- `Robot`(`stiff_physics/robot.py:32-213`,亲验)是完整的公开便利层:
  `Robot(engine)` 构造、按名/序号取关节 `get_joint(name_or_index)`、
  `revolute_joints/prismatic_joints/all_joints`、**度数单位**的
  `set_revolute_position(...deg)` / `set_prismatic_position(...)` /
  `set_joint_position(name, value)`、`set_gripper_strength(...)`、`reset_all()`、
  `JointInfo.lower_limit_deg/upper_limit_deg` 属性等。
- 后果是手册内部出现**悬空引用**:API_EXECUTION §9.1 的 GRIP_MODE=pos 配方原文是
  "`set_prismatic_position(pi, cl + s*(op-cl))`"——该方法在 Engine 上不存在
  (`grep engine.py` 0 命中,亲验),只在 Robot 上;README §6 的 duck 示例行写
  "PD `set_joint_target` 驱动"——两树都没有叫 `set_joint_target` 的 API
  (那是 duck_grasp docstring 里对 set_revolute_target/set_prismatic_target 的口头简称)。
  读者按手册索引这两个名字会一无所获。
- `Pipeline`(`stiff_physics/pipeline.py:21-160`)是 polyscope 可视化运行循环
  (`run()` / `user_gui()` 钩子)——GUI 示例的地基,同样零文档。
- 溯源:上游图件 `docmap/api-python-engine.md` 自己也只在 §1(`__all__`)提到这两个类
  而未勘探其方法——手册**继承了图件的盲区**,不是删减了图件内容;但对最终用户
  这仍是公开 API 面的实缺。

### 1.2 `stiff_physics` 其余辅助模块零覆盖(部分是重要用户入口)

亲验 `stiff_physics/` 目录,以下模块在 8 份手册中 0 命中:

| 模块 | 内容(取自模块 docstring) | 影响 |
|---|---|---|
| `usd_scene_parser.py` | "Parse an Omniverse USD stage and translate it to StiffGIPC engine load calls";单引擎多实例模式(只解析 env_0 模板再按 env 偏移复载) | **Isaac/Omniverse 用户的 USD 迁移入口**,手册完全没提这条路存在 |
| `urdf2usd.py` | URDF → USD stage 转换(UsdGeom.Mesh + UsdPhysics API) | 同上工具链 |
| `urdf_loader.py` | 纯 Python URDF 解析器(engine.load_urdf 之外的宿主侧 FK/网格收集) | 维护者需要知道两条 URDF 路径并存 |
| `trajectory.py` | `Assets/trajectories/*.txt` 文本轨迹格式的读取/插值器 | 回放示例的数据格式无文档 |
| `mesh_utils.py` | 网格简化与凸分解工具 | — |
| `utils.py` | 路径解析/数学转换 | — |

### 1.3 URDF primitive 碰撞代理机制只剩一句话

- README 能力表有 "primitive 碰撞代理(box/sphere/cylinder)" 七个字;
  API_EXECUTION §7.7 讲了旋钮 `STIFF_URDF_PRIM_PROXY` 的**typo 陷阱**(不在注册表,误报 WARN)。
- 但机制本身——box 精确 / icosphere subdiv-2 膨胀 ~2.4% / 24 段圆柱 ~0.9%、
  `=0` 恢复 ≤0.8.4 的"URDF primitive 静默跳过"行为(即老版本 primitive **根本不参与碰撞**,
  这是行为差异不是优化)——在任何分册正文都没有。出处:CHANGELOG.md 0.8.4.2 条目
  (:109-121,由 docmap/existing-docs-harvest.md §A.2 收割,手册未吸收)。
  对"URDF 机器人为什么和 0.8.4 行为不同"的排障者,这是关键事实。

### 1.4 pre-v0.8.5 版本史无指路

- CHANGELOG_TIMELINE 标题明示覆盖窗是 v0.8.5 → 2026-09-07(这个裁剪本身合理),
  但全手册没有告诉读者**更早历史去哪查**:0.8.0 三模式首发、0.8.2 SymGH −37.5% triplets 与
  `pcg_tol` 回 1e-4 的缘由、0.8.4.1 接触力符号修复、0.8.4.2 MAS 批次不变性三根因……
  这些散落在个别注脚里。完整早期 CHANGELOG(0.1.0→0.8.5.1,1097 行)在 phase-cd 树根
  `CHANGELOG.md`,0.8.5.2/0.8.5.3 条目在稳定线树——CHANGELOG_TIMELINE §7 提了两树分叉,
  但导读(README §5)没给"早期版本史入口"。一行链接即可修复。

### 1.5 其余小项

- `Assets/trajectories/` 下的三个默认 episode HDF5
  (`episode_fold_shirt_umi.hdf5` 等,亲验存在)与其 `(T,16) actions` 格式:
  API_EXECUTION §9.2 的命令行写了 `[episode.hdf5]` 可选参数,但没有说默认 episode
  在哪、格式是什么(replay_foldshirt_multienv.py docstring 有,手册未收)。
- polyscope 版本锁 `>=2.4,<2.6` 只出现在 README §3.1 安装表一处;
  PRINCIPLES/示例章节谈 GUI 时未再提示(可接受,列此备查)。

---

## 2. 文档间断链(引用了不存在的文件/章节)

### 2.1 七个幻影分册文件名,14 处引用(全部亲验 `ls` 不存在)

手册显然经历过分册改名/合并,三份分册的"姊妹分册"引用没有随之更新:

| 断链文件名 | 引用位置 | 实际内容现在在哪 |
|---|---|---|
| `PRINCIPLES_SOLVER.md` | PRINCIPLES_DYNAMICS.md:4;PRINCIPLES_CONTACT.md:5、86(§5)、285(§4)、306(§4.6)、312(§7) | 主循环/line search → PRINCIPLES_DYNAMICS §2–4;PCG/MAS → PRINCIPLES_EXECUTION §2;LS 图形态/starved-LS → PRINCIPLES_EXECUTION §5 |
| `PRINCIPLES_ENERGY.md` | PRINCIPLES_CONTACT.md:5、59(§2)、505(§2.2) | PRINCIPLES_DYNAMICS §1(组合公式)/§5(逐槽) |
| `PRINCIPLES_MULTIENV.md` | PRINCIPLES_DYNAMICS.md:4 | API_EXECUTION §1 + PRINCIPLES_EXECUTION §3 |
| `RUNTIME_GRAPH_RL.md` | PRINCIPLES_DYNAMICS.md:4 | PRINCIPLES_EXECUTION §5–6 + API_EXECUTION §3–4 |
| `MULTIENV_MODES.md` | KNOWN_ISSUES.md:3 | API_EXECUTION §1 |
| `FRAME_GRAPH_RL.md` | KNOWN_ISSUES.md:3、347(§4.1 正文"机制链详见") | PRINCIPLES_EXECUTION §5 |
| `CONTACT_CCD.md` | KNOWN_ISSUES.md:3 | PRINCIPLES_CONTACT |

注:PRINCIPLES_DYNAMICS 头部自带免责("若分册文件名与最终目录不一致,以 docs/manual/ 索引为准"),
但 README §5 导读并没有提供幻影名→实际文件的映射,免责并不能让读者落地;
且断链引用带着**章节号**(如"PRINCIPLES_SOLVER.md §5 close 集机制"),即便猜到目标文件也要再找一次。

### 2.2 工具脚本断链

1. **`tools/build_hybrid_mesh.py` 两树均不存在**(find 亲验)。
   API_CORE §3.9 说 `add_hybrid_fem_body` 加载"`tools/build_hybrid_mesh.py` 产出的
   ABD-FEM 混合四面体网格(.npz)"——照抄了 engine.py:750 的 docstring,但该工具已不在树内。
   现成资产(`Assets/sim_data/umi_hybrid_sf_v800/v1340/v1690`)可用,
   **新混合爪资产不可再生**。手册未把它标为断链/待核实。
2. **`fix_obj_winding.py` 其实存在,两条待核实项可以关闭**:
   README §7 待核实项 3 与 API_EXECUTION 附录 A#7 都写"两树 examples/ 均不存在(可能在
   tools/ 或已删除)"。亲验:**它在两棵树的 `tools/fix_obj_winding.py`**
   (phase-cd 与 stable-08 均有)。`case_26_perf_tuned.py` docstring 的 `examples/` 前缀错了,
   文件没丢。手册应把两处 open point 改写为"位于 tools/,docstring 路径前缀过时"。

### 2.3 其余引用健康度

- 对 `../` 既有证据文档的全部 24 个引用(SIMULATOR_EXECUTION_DESIGN / OPTIMIZATION_ROADMAP /
  GPU_NATIVE_RL_PLAN / CI / PHASE_C / A800_ALLEXAMPLES / RELEASE_NOTES / BVH 两份 / 其他)
  逐一亲验**全部存在**,无断链。
- 分册互链(API_CORE ↔ API_EXECUTION ↔ KNOWN_ISSUES ↔ PRINCIPLES_*)存在的文件间无断链。

---

## 3. 用户旅程断点(装好 wheel / 拉起源码后想做 X,手册无入口)

按人物分列:

### 3.1 wheel 用户(稳定线)

| 想做 | 现状 | 断点等级 |
|---|---|---|
| 按关节名控制机器人("Isaac 式") | `Robot` 类无文档(§1.1);README 首页承诺了它 | **高** |
| 从 USD/Isaac 场景迁移 | `usd_scene_parser`/`urdf2usd` 无文档(§1.2) | **高**(如果 USD 是支持面)/中(如果仅内部工具——手册至少应声明其状态) |
| 可视化仿真结果 | `Pipeline` 无文档;GUI 示例可抄但无原理说明;polyscope 版本锁只在安装表 | 中 |
| 跑通带 GUI 的抓取示例 | ✅ 覆盖良好(README §4/§6 + API_EXECUTION §9) | — |
| 触觉/摩擦力读数 | ✅ 指引明确(用稳定线) | — |

### 3.2 phase-cd 用户(RL/引擎开发)

| 想做 | 现状 | 断点等级 |
|---|---|---|
| 照 API_EXECUTION §9 跑示例 | **§9 通用前缀教你设 `STIFF_SKIP_CCD_SANITY=1`,§9.5 又教一次**;而 README §6 明令"勿再设置"并给了理由(引擎已不读它;phase-cd 上触发 unknown-knob WARN,`STIFF_KNOB_STRICT=1` 下直接 `ConfigurationError`)。亲验代码:`gipc_modules/14_energy_linesearch_solver.inl:556-574` 只读 `GIPC_FORCE_CCD_SANITY`——**README 对、API_EXECUTION 错**。这是全手册唯一一处自相矛盾的操作指令,且照错的那份做会在推荐的 `STIFF_KNOB_STRICT=1` 姿势下直接抛错 | **高** |
| 在 phase-cd 拿到摩擦读数 | 5 处(README §2.1、API_CORE §8.1、API_EXECUTION 头部、PRINCIPLES_CONTACT §7.2、PRINCIPLES_DYNAMICS §5.8)一律只说"未移植,用稳定线";**只有 KNOWN_ISSUES §1.1 披露移植成品已在分支 `port/friction-anchor-086`**(亲验分支存在,含 friction-anchor checkpoint 序列化提交)。跨文档口径不一致:准备做移植的维护者若只读 API 分册会白做一遍 | 中 |
| 自造混合软爪资产 | `build_hybrid_mesh.py` 缺失(§2.2) | 中 |

### 3.3 维护者(发布/门禁)

| 想做 | 现状 | 断点等级 |
|---|---|---|
| **构建并发布一个新 wheel** | 手册没有发布章。§3.2 只有开发构建(`cmake --build … pystiffgipc`),wheel 打包(scikit-build-core / 从 tag worktree 构建 / cp311+cp312 双 ABI / Release 资产命名)完全缺失;唯一指针是 KNOWN_ISSUES §5.5 一句"见 `STIFF_PHYSICS_RELEASE_HANDBOOK.md`"并同时警告该手册三处过时。对"独立维护"目标,这是一整章的缺失 | **高** |
| 改代码后跑 22 段门禁 | 手册多处引用 `../CI.md` 与 `scripts/verify_gates.sh`,但没有 how-to(怎么装 hook `scripts/install_hooks.sh`、怎么手动全跑、单门重跑、`SKIP_GATES=1` 的后果);CI.md 存在可补位,属"有引用无正文" | 中 |
| 换锚流程 | KNOWN_ISSUES §4.4 + PRINCIPLES_EXECUTION §4.2 讲了原则(换锚=战役,跨架构重验),无 step-by-step;可接受 | 低 |
| 合并 port/friction-anchor-086 | KNOWN_ISSUES §1.1 给了 5 步移植要点+前移警告,**这一条做得很好** | — |

---

## 4. 各文档 open_points / 待核实标记汇总

现状:**约 45 条待核实散落在 8 份文档的 6 个专节 + 正文行内标注,无任何汇总页,
跨文档重复项互不链接**。下面按主题归并(括号内为出现位置);
建议在 docs/manual/ 增加一页 OPEN_POINTS.md 作为唯一登记表。

### 4.1 跨文档重复的待核实(应合并为单条)

| 主题 | 出现处 | 状态 |
|---|---|---|
| v0.8.5.4 对外发布状态(wheel 是否已挂/撤回) | README §7#2、API_CORE 版本口径、API_EXECUTION 附 A#2、KNOWN_ISSUES §6#2、CHANGELOG_TIMELINE §8#1、PRINCIPLES_EXECUTION §0 | 未决,需问 owner |
| phase-cd 摩擦读数恒零的运行时实测确认 | KNOWN_ISSUES §6#3、PRINCIPLES_CONTACT §9#5、CHANGELOG_TIMELINE §8#5、API_EXECUTION 头部 | 代码结构已证,跑一次剪切台即可关闭 |
| `fix_obj_winding.py` 位置 | README §7#3、API_EXECUTION 附 A#7 | **本审查已关闭:在两树 `tools/fix_obj_winding.py`**(§2.2) |
| `replay_case39_UMI_obb_cup_shirt_forcegrip.py` 默认 `trackgrip` 语义 | README §7#4、README §6 | 未决(旧线对照示例,低优先) |

### 4.2 API/语义类(API_CORE 行内 + 附录)

1. `velocity_damping` 是否作用于 FEM 顶点(只见 ABD 消费点)——API_CORE §2.1、PRINCIPLES_DYNAMICS §6.7/附 C#10。
2. `load_mesh_from_data` dim=3+vpf=3+FEM 的 "internally tetrahedralize" 声明疑似不可用(未找到实现)——API_CORE §3.3。
3. `/tmp/stiffgipc_mesh_data/` 固定目录的多进程同名竞态——API_CORE §3.3。
4. `set_body_animated_target` 的能量门控经公开 loader 疑似永不触发(Animated 不可达)——API_CORE §3.2/§10.4、PRINCIPLES_DYNAMICS 附 C#8。
5. fixed joint 非正交 n/b 输入的退化行为——API_CORE §4.2。
6. `set_per_tet_young_for_body` 在 MAS 下的 tet 序对应关系——API_CORE §5.2。
7. stitch 能量核 vs G/H 核目标不一致的实际影响(有 pin+大旋转+非零 offset)——API_CORE §4.8、PRINCIPLES_DYNAMICS §8.1/附 C#5。
8. phase-cd 无独立 Kappa 清理入口的残留影响量级(~0.9 µm 推定)——API_CORE §9.5、KNOWN_ISSUES §1.2。
9. `get_body_contact_force` 的 ground 口径(代码含 ground,docstring 说无)——API_EXECUTION 附 A#3、PRINCIPLES_CONTACT §9#7。
10. 稳定线 getter 是否带 metis 解扰(输出序两线可能不同)——API_EXECUTION 附 A#4、PRINCIPLES_CONTACT §9#8。
11. per-env 遥测在纯设备快路径的返回值(推理未运行验证)——API_EXECUTION §1.8/附 A#1。
12. `m_avg_env_bbox2` 回退分支触达条件——API_EXECUTION 附 A#5。
13. gpu_rl joint_observations 字节级打包——API_EXECUTION 附 A#6。
14. wheel 资产文件名按 v0.8.4 模板推断,URL 未逐字节验证——README §7#1。
15. `newton_velocity_tol` 的 uipc 参考默认 0.05 只来自注释——API_CORE §2.1。

### 4.3 原理/机制类

16. 地面 barrier/λ 用 RANK-1 而自碰 RANK-2 的设计意图——PRINCIPLES_CONTACT §9#1(+#2 地面摩擦 C0 vs 自碰 C1)。
17. `_cpNum[1]` 槽位无独立消费——PRINCIPLES_CONTACT §9#3。
18. 稳定线 mlbvh 发射 smooth=false 未逐行复核——PRINCIPLES_CONTACT §9#4。
19. "44 万 mollify 请求/0 执行"数字未复测——KNOWN_ISSUES §6#1。
20. 0.792 暂态峰值原始日志未存档;剪切台界面构成(几何平均 vs 0.600 吻合)疑点——PRINCIPLES_CONTACT §9#6/§5.1。
21. 418 N / 10.5 µm stitch 滞后实测未归档——PRINCIPLES_DYNAMICS §8.2/附 C#13。
22. binned K/W/E0 覆盖窗外的溢出/下溢行为——PRINCIPLES_EXECUTION §4.1。
23. `Cub_PCG_DotReduction` 的确定性依赖 CUB 实现惯例——PRINCIPLES_EXECUTION §2.2。
24. `PCGSolverConfig.use_bsr` 疑为死字段——PRINCIPLES_EXECUTION §2.1。
25. 非图 strict 宿主槽序可重复的机制保证(经验性质)——PRINCIPLES_EXECUTION §3.4(f)。
26. 稳定线金锚是否 = `f7fb5a786c2d7935`(推断,稳定树内未跑门禁验证)——PRINCIPLES_EXECUTION §4.2。

### 4.4 时间线/实测口径类(CHANGELOG_TIMELINE §8)

27. beaker +7% 破案的证据锚(无对应提交,原始逐帧数据未定位)。
28. 成对审计 ±/σ 细分需 `git show 4219f37` 核对全文。
29. tag 打在 release 后一条的惯例是否有意。
30. BVH 分支内提交归属未逐条核对。
31. ~30 条纯 docs 提交只核对了标题。
32. RL 微步 19.3/9.7/3.85 的平台标签在 A800_ALLEX 与 SIM_EXEC 之间冲突(已取 A800,待 owner 确认)。
33. KNOWN_ISSUES §6#5:稳定线单体行号(GIPC.cu:15221 等)复引前建议 grep 复核。

---

## 5. 与既有 `docs/` 文档的冲突核查

结论:**手册对既有文档的已知过时项处理得当**,大部分冲突被主动标注并给出更正
(beaker "+7% 回归"→启动段破案;GRAPH_DEFAULT_ON 倍率表"已过时,结论方向存活";
RELEASE_HANDBOOK 三处过时;A800_ALLEX vs SIM_EXEC 的 RL 平台标签冲突入待核实;
UMI_FINRAY_NOTES force 段旧实现警告;README(公开镜像)sm 支持列表过时)。未被手册标注的残余:

1. **`docs/FORCE_CONTROL_DESIGN.md` 头部"prototype 分支未入 release"已过时**
   (力控 API v0.6.4 起已入线;由 docmap/existing-docs-harvest 冲突表 #7 指出)。
   手册的力控条目(API_CORE §4.5)没有引用该文档,因此无直接矛盾,
   但也没有在 KNOWN_ISSUES §5.5 那样的"过时文档警示"里点名它——
   读者独立打开 docs/FORCE_CONTROL_DESIGN.md 会被头部误导。
2. `CLAUDE.md` 分支表不含 codex/* 战役分支(手册 README §1 与 KNOWN_ISSUES §5.1 引它作分支拓扑依据,未附此提醒;工程线分支未推送这一事实 CHANGELOG_TIMELINE §7 已另行覆盖,矛盾风险低)。
3. `docs/GPU_SIMULATION_EXECUTION_OPTIONS.md` 的 frame_transaction.cu 行号钉在 2026-08-03(harvest #9);手册未引用它,无冲突,列此备查。
4. 手册内部数字口径自查:examples 计数(91/89、test 30/29)亲验与磁盘一致;
   两线 Config 逐字节相同、`STIFF_SKIP_CCD_SANITY` 死旋钮、`STIFF_LOG_LEVEL` 死旋钮、
   knob 注册表 170 条、22/22 门禁等抽查均与代码/脚本一致。
   **唯一实质性内部矛盾即 §3.2 所列 STIFF_SKIP_CCD_SANITY 一处**(README 对、API_EXECUTION 错)。

---

## 6. 修复建议(按优先级)

1. **改 API_EXECUTION §9 开头与 §9.5**:删除两处 `STIFF_SKIP_CCD_SANITY=1` 建议,
   与 README §6 对齐(该变量引擎已不读;phase-cd 触发 WARN/抛错)。一行改动,消除唯一自相矛盾。
2. **修 14 处幻影分册引用**:按 §2.1 的映射表逐处替换为现有文件+章节;
   或最低成本在 README §5 导读加一张"旧分册名→现文件"对照表。
3. **补 API_CORE 一节"便利层:Robot / Pipeline"**(约 1 页):Robot 全方法表
   (注明度数单位与 Engine 弧度 API 的换算)、Pipeline 的 run/user_gui;
   并顺手把 §9.1 配方的 `set_prismatic_position` 与 README §6 的 `set_joint_target`
   改成真实 API 名或标注所属类。
4. **补"资产与场景工具链"小节**:usd_scene_parser/urdf2usd/trajectory 的存在与状态声明
   (支持面 or 内部工具);标注 `tools/build_hybrid_mesh.py` 缺失(hybrid 资产现状=只用现成 npz);
   关闭 fix_obj_winding 两条待核实(改指 `tools/`)。
5. **补"维护者:发布与门禁"一章**(或在 README §5 明确挂出既有文档并列出其过时项):
   wheel 打包流程、发布 handbook 的三处过时更正、`install_hooks.sh`/`verify_gates.sh` 用法、
   换锚流程 checklist。
6. **建 OPEN_POINTS.md 汇总页**:收拢 §4 的 45 条,合并 4 组跨文档重复项,
   给每条一个编号供关闭时引用。
7. 小项:URDF primitive proxy 机制补 3 行(质量数字+`=0` 的行为差异);
   README §5 加 pre-v0.8.5 历史指路(两树 CHANGELOG 位置);
   摩擦修复的 5 处"未移植"表述统一加一句"移植成品见 port/friction-anchor-086(需前移)"。

---

## 附:审查方法

- 通读 8 份分册全文(6494 行);
- 抽读 scratchpad docmap 图件:appendix_table(旋钮全表)、issue-scan、existing-docs-harvest
  (含冲突清单)、examples-recipes、api-python-engine、env-knobs 各主要章节;
- 亲验:全部 markdown 链接目标存在性(`ls`/grep)、`stiff_physics/` 模块清单与
  Robot/Pipeline 方法、engine.py 116 个公开方法在手册的命中率(115/116,唯一漏网是
  正则误切的 "framed")、tools/ 目录、Assets/trajectories、两树 examples 计数、
  `STIFF_SKIP_CCD_SANITY`/`GIPC_FORCE_CCD_SANITY` 的实际消费点、
  `port/friction-anchor-086` 分支存在性、`build_hybrid_mesh.py` 双树缺失。

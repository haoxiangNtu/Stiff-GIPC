# Per-engine 设备描述符战役蓝图（v0.9）

目标：消灭"单进程单引擎"约束的根源——进程级可变设备全局。清点基线（06a6710，
全库静态扫描）：**31 个可变 `__device__` 全局、34 个 `cudaMemcpyToSymbol` 写点**
（无 `__constant__`/`__managed__`；无 Async 写；无 GetSymbolAddress 别名——写点
集合即完整咽喉）。

## 分组

- **(a) per-engine 指针/容量（16）**：`g_gbin`、MAS `g_mRbin/g_mZbin/g_matbin`、
  ABD `g_massbin/g_abd_sysbin/g_abd_hessbin/g_abd_wrenchbin`、
  `g_vloc`、`g_self_p2g`、`g_vertex_env_id`、`g_dcd_cp_cap/g_ccd_cp_cap` 等
  → 描述符结构体成员。
- **(b) 模式旗标（11）**：`g_binned_on/g_det_reduce/g_seg_binned/g_seg_warp/
  g_ee_canon/g_ee_detgate/g_bvh_envmajor/g_bvh_envpart/g_ee_nodedup/
  g_ee_nomollify/g_ground_hess_legacy` → 引擎切换时经单一咽喉重发布。
- **(c) 调试探针（8）**：trace/tgt 族 + 设备写的 `g_max_stack`（atomicMax）、
  `g_xskip`（atomicAdd）——**设备写探针不能进只读描述符**，保持全局或经描述符
  携带的可写 scratch 指针。
- **(d) 真进程级**：`g_gipc_log_level`、`g_dec_frame/k`（调试计数）——保持全局。

## 方案裁决

**单个 `__constant__ EngineDeviceDescriptor`**（~13 指针 + ~15 int）+ 引擎切换时
drain-then-publish。**零热核签名改动**——参数穿线会踩融合核 200reg/零溢出悬崖
（见 energy/03 头部裁决），被否。读点集中于 ~8 个 inline helper（`_gfxAdd`、
`binned_deposit`、`_emit_slot`、`_vless/_edge_lkey`、`_cross_env_skip/_same_env`
族），改 helper 即覆盖 ~50-70 个内核。此方案买到**串行**多引擎；真并发需参数化
描述符 + (c) 类重定向，另立项。

## 顺序要害（切换协议必须全集发布）

按写频分层：init 一次（g_gbin、ground_hess_legacy）/ 首解一次（四个模式闩锁——
见 phase-0）/ 每 buildCP 两次（EE 门族 + g_vloc + **g_self_p2g 每次重置**）/
每 solve（ABD/MAS bin 族）/ 每 grow（emit caps、mRbin/mZbin）/ 每 API 调用
（**g_vertex_env_id 黏性**——与 g_self_p2g 的重置/黏性不对称是成文契约
（04_lbvh_build.inl:423-428），统一化会复引入 env 叠加 bug）。

## Phase-0（前置，已并入本分支）：跨引擎串态闩锁清除

清点发现三处**现行 bug**（同进程引擎 B 静默继承引擎 A 状态）：
1. `s_binned_set`（13_kappa:470）——`g_binned_on/g_det_reduce` 一进程只发布一次；
   strict→merged 序列永不放松、反向永不收紧。修复：按**解析值**追踪，值变即重
   发布（热路径零成本）。
2. `s_seg_binned_set`/`s_seg_binned_host`（pcg_solver ~884/859）——同缺陷，且
   host 镜像还选内核路径。同修法。
3. 函数级静态设备 scratch 跨引擎共享：`s_ebin`（cal_abd_energy）、`pe_all/g_sink`
   （energy/01）+ ipc_solver/11/13 的十余个 `d_*` 缓存（泄漏+尺寸按首引擎）。
   归宿：所有权入 GIPC/ABDSystem 成员 + FREE 路径（Impl 已为导出缓冲做过同样
   迁移，00_impl_api_surface.inl:168-181 是范式）。

## 首迁三族（phase-1 顺序）

1. EE/BVH 门族（唯一发布点已存在：10_ccd_buildcp:106-184）。
2. emit caps（pair_buffers 契约点，三写者一 setter）。
3. determinism 标量（与 phase-0 闩锁清除同 commit）。

## 风险清单

热核参数表=锚敏感（禁穿线）；`__constant__` 写与在飞内核无序（切换必须 drain）；
RDC 跨 TU 的 extern/单定义纪律（错则各 TU 各持未初始化拷贝，极难查）；(c) 类
设备写探针不可入只读结构；host 静态 scratch 是平行战线（§Phase-0.3），只扫
memcpy 咽喉会漏。

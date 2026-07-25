# engine_modules — sim_engine.cu 复合编译单元的语义模块（v0.8.6 Phase 1b）

按序 #include 进同一 TU；拼接与拆分前文件字节级相等（sha256 见
ORIGINAL_SHA256.txt）。include 顺序禁止重排。

| 模块 | 内容 |
|---|---|
| 00_impl_api_surface | Impl 结构、构造/析构、全部 finalize 前 API（load/set/get/joint/groups） |
| 01_config_upload | apply_config_to_ipc、do_initFEM、MAS 分区、do_upload_to_gpu、BVH/solver 初始化 |
| 02_finalize_nandiag | finalize()、NaN 哨兵/诊断结构 |
| 03_step_getters_export | step()、getter 族、接触力/应力导出、录制 |
| 04_teleport_checkpoint | teleport_fem_vertices（METIS 置换语义）、checkpoint 存取 |

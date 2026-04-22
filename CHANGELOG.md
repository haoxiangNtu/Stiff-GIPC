# Changelog

All notable changes to **stiff-physics** are documented here. This project
follows the spirit of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed
- **`metis_partition` write path no longer hardcoded to maintainer's source
  tree.** Previously, the wheel binary embedded a compile-time path
  (`<source>/MeshProcess/metis_partition/../../Assets/sorted_mesh/`) that
  the metis library used to write `*_sorted.16.obj` and `*_sorted.16.part`
  intermediates. On any user machine where that path didn't exist, loading
  a FEM cloth (the default `preconditioner_type=1` MAS path) raised
  `RuntimeError: filesystem error: cannot create directory`. Fix: drop the
  `OUTPUT_DIR` macro to an empty default and plumb a runtime
  `metis_output_folder` parameter through `metis_sort()` →
  `SimpleSceneImporter::load_geometry()` → `SimEngine::load_mesh()`. The
  runtime folder is now derived from `Config.assets_dir` (or the
  `GIPC_ASSETS_DIR` macro fallback). Verified: the wheel binary no longer
  contains any source-tree paths under `strings(1)`, and the simulator
  loads correctly with the build-time path absent on disk.

## [0.1.0] — 2026-04-14

Initial public release of `stiff-physics` Python wheel.

### Added
- StiffGIPC IPC physics engine with Python bindings (`pystiffgipc`).
- Pre-compiled wheel for Linux x86_64, Python 3.11, CUDA 12.x:
  - sm_89 (RTX 4090)
  - sm_120 (RTX 5090)
- Examples: cloth + rigid + URDF arm interaction (`case_0` … `case_26`).
- Headless joint-control example (`headless_joint_control.py`).
- URDF and USD scene loading APIs.
- `Config.gravity` / `Config.ground_normal` / `Config.ground_offset` for
  arbitrary up-axes.

### Source mapping
The v0.1.0 wheel was built from a working-tree state corresponding to
private-repo commit `87f90be` (tag `v0.1.0-source`), reconstructed
post-hoc. See the release handbook for the audit recipe.

### Known limitations
- Z-up coordinate system is ~10× slower than Y-up due to inherent cloth
  folding geometry under different gravity orientations. Workaround: use
  Y-up internally and transform externally.
- FEM cloth vertex order is reordered by the MAS preconditioner (default).
  External rendering pipelines that need vertex order matching the source
  `.obj` should set `Config(preconditioner_type=0)` (at the cost of slower
  PCG preconditioning).

[Unreleased]: https://github.com/haoxiangNtu/stiff-physics/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/haoxiangNtu/stiff-physics/releases/tag/v0.1.0

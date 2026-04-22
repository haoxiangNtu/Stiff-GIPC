# Changelog

All notable changes to **stiff-physics** are documented here. This project
follows the spirit of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/).

## [Unreleased]

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

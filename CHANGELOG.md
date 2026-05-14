# Changelog

All notable changes to **stiff-physics** are documented here. This project
follows the spirit of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/).

## [0.5.0] — 2026-05-15

### Added

#### Hybrid ABD-FEM mesh API
- **`add_fem_pins_with_local_pos(fem_ids, body_ids, local_pos)`**: bulk pin
  API for hybrid mesh scenarios where the rigid region of a continuous tet
  mesh is kinematically driven by an ABD body.  Avoids the per-pin
  Python↔C++ round-trip of `add_fem_pin_to_abd` at hybrid-mesh scales
  (1k+ pins).  Caller provides each pinned vertex's coordinate in the
  ABD body's REST frame directly.
- **`Engine.add_hybrid_fem_body(npz, transform=None)`**: Python wrapper
  that loads a hybrid mesh `.npz` (from `tools/build_hybrid_mesh.py`)
  and bulk-pins its rigid verts in one call.
- **Phase 4 hybrid kernel skip**: rigid-internal tets (all 4 verts pinned
  to the same ABD body) now skipped in `_calculate_fem_gradient_hessian`
  — their Green strain is structurally redundant w.r.t. the ABD body's
  E_orth penalty.  Saves a co-rotational SVD per such tet per Newton iter.
- **Chain-rule buffer 2× → 32×** in `GIPC::init`: the M3.5 chain-rule
  kernel reserves an extension range past the FEM triplets for diff-body
  pin-pin expansion (worst-case 16×).  Original 2× margin overflowed when
  the rigid region was large → CUDA illegal memory access.  32× gives
  comfortable margin.

#### URDF helpers
- **`get_urdf_link_transform(link_name)` → 4×4 world transform**: returns
  the importer's FK output for any URDF link.  Available right after
  `load_urdf` (no need to finalize).  Use to attach extra bodies in the
  correct world frame.
- **`set_urdf_mesh_override(link_name, msh_path, young_modulus=1e7)`**:
  override the mesh used for a URDF link in the next `load_urdf` call.
  Required when URDF references a mesh file that doesn't exist on disk
  (without override, the importer silently skips the link).

#### Per-body / per-joint controls
- **`set_body_animated_target(body_id, x, y, z, strength=1e6)`**: per-step
  soft-target driver for ABD bodies loaded with `boundary_type='Animated'`
  (=3).  Pulls `q.t` toward `(x,y,z)` and `q.A` toward identity via a
  quadratic penalty.  Body's 12 DOFs stay in PCG so joint constraints +
  M3.5 chain-rule pins still propagate.
- **`set_body_apply_gravity(body_id, enabled)`**: toggle gravity per body
  at runtime (ABD path zeros/restores the body's 12-DOF gravity vector
  via cached pre-toggle state; FEM path flips the per-vertex
  `apply_gravity[]` flags).  Use for gripper ABD bodies hanging off URDF
  arms via revolute joint to avoid joint-vs-gravity drift accumulation.
- **`set_fixed_joint_strength(idx, kappa)`**: override per-fixed-joint
  kappa post-finalize.  Default `joint_strength_ratio·(m_p+m_c)` ≈ 8e-3
  is too weak for a hybrid gripper welded to a heavy arm hand.  Set ~1e6
  for tight tracking.
- **`engine.py` `_BOUNDARY_MAP`** adds `"Motor"` (=2) and `"Animated"`
  (=3) so users can pass them as strings to `load_mesh`.

#### Developer experience
- **`STIFFGIPC_NATIVE_DIR` env var**: override the engine `.so` load path.
  Useful when developing across multiple worktrees — the venv's installed
  `stiff_physics._native` points to whichever worktree was last
  pip-installed; `STIFFGIPC_NATIVE_DIR=/path/to/another/worktree/build_312`
  uses a different build without re-installing.

### Fixed

- **`_apply_fem_pins` kernel: A·lp not A^T·lp** (commit `3d1d54b`).
  The hard-pin projection kernel computed `world.x = q[0] + q[3]*lp.x +
  q[6]*lp.y + q[9]*lp.z` which is `A.col(0)·lp = (A^T·lp)[0]`.  Per the
  canonical `q` layout in `abd_jacobi_matrix.inl`, `q[3..5]` is `A.row(0)`
  — so the correct expression is `q[3]*lp.x + q[4]*lp.y + q[5]*lp.z`.
  Bug only surfaced when ABD bodies rotated non-trivially (URDF arms
  with revolute joints driving a hybrid gripper); pinned FEM verts
  visually drifted off the rigid sub-mesh by `~|sin(θ)·(n × lp)|`.
  Fully backward compatible: A symmetric → old/new numerically identical.

- **`_computeSoftConstraintGradient` kernels: A·lo not A^T·lo** (commit
  `27922c1`).  The `add_stitch_spring` local-frame target computation
  had the same row/col bug as `_apply_fem_pins`.  Stitch springs that
  pull FEM verts toward an anchor on a rotating ABD body now track the
  rigid frame correctly instead of lagging by `~|sin(θ)·(n × lo)|`.
  Also corrects the misleading `R = [axis_x | axis_y | axis_z] columns`
  comment to reflect the canonical row layout.

### Known issues

- ⚠️ **`case_27_mobile_s1_softgripper_cup.py` regression**: the URDF
  importer joint-angle clamp (this release) puts the arm at limit pose
  at frame 0 instead of the legacy 0-pose.  case_27's softpad placement
  is computed via a Python URDF parser that assumes joint angles = 0,
  causing a frame-0 mismatch between FEM softpad rest position and
  pinned-vertex world target → ~1M self-intersection contact pairs +
  step 3/8 multi-second-stuck (vs ~30ms baseline).  No NaN, no crash —
  simulation completes but is very slow and visually wrong.
  **Workaround**: pass explicit `initial_joint_angles={...}` matching
  the new clamp values, OR retune softpad placement to use
  `eng.native.get_urdf_link_transform(soft_material_link_name)` after
  `load_urdf`.  Other demos using `add_fem_pins_with_local_pos`
  (case_29..41 hybrid family) are unaffected because the hybrid pin API
  computes pin positions in the ABD body's REST frame (independent of
  joint pose).  Tracking issue: maintainer-only
  `docs/internal/BUG_y5_case27_softpad_intersect.md`.

## [0.2.0] — 2026-04-24

### Performance (case_26 scene, validated)

- **case_26 step speedup: ~18%** vs v0.1.x baseline (23.8 → 19.5 ms/step
  median, n=30 paired t-test, p<0.0001, bootstrap 95% CI [0.791, 0.823]).
  Measured on RTX 4090D + `case_26_arm_cloth_semi_implicit.py`.
  Physics cross-drift within 1.04× GPU non-determinism floor across
  9 checkpoints (1 → 300 steps): no observable physics divergence.

  Composed of three orthogonal engine-level optimizations:

  - **PCG D2H elimination**: alpha/beta/convergence moved to device-side
    scalars; stride-K=8 convergence check. Saves 504 D2H sync stalls
    per frame (contribution: -5.7% on free-fall scenario).
  - **Multi-stream BVH self-collision**: `bvh_e` on side stream,
    concurrent with `bvh_f` on default. case_26's 8593 surface verts
    underfill the 114-SM GPU; concurrency ~14% more SM occupancy
    (contribution: -10.5%).
  - **Fused PCG inner kernels**: `update_vector_dx_r_fused` re-derives
    alpha per-thread (eliminates `compute_alpha_kernel` launch);
    `cub::DeviceReduce::Sum` + `TransformInputIterator` replaces manual
    tree reduction (saves 2 launches per dot); combined swap+convergence
    kernel (contribution: -1.6% on top of the above, n=90 aggregate).

  See private `docs/internal/RELEASE_LOG.md` §v0.2.0 prep for the full
  audit trail including 9 null/infeasible experiments that were rejected.

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

### Added
- New example `examples/case_26_render_obj_indices.py`: per-body coloured
  rendering of the case_26 scene (XArm7 + falling shirt). Each arm link
  gets its own polyscope mesh with an HSV hue ramp; the shirt is shaded
  by a smooth XYZ-as-RGB gradient locked to material points. Uses the
  default MAS preconditioner so cloth deformation matches the basic
  `case_26_arm_cloth_semi_implicit.py` (no chaotic divergence from a
  different inner solver path).

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

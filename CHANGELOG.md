# Changelog

All notable changes to **stiff-physics** are documented here. This project
follows the spirit of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/).

## [0.8.4.2] — 2026-07-22

Versioning note: this train uses a 4th "hotfix/consolidation" digit on top of
SemVer's three (0.MINOR.PATCH.HOTFIX); 0.8.4.2 consolidates the M1 quick-win
plan onto the 0.8.4 line. Scope is larger than a narrow hotfix — it adds
public API surface and changes two defaults — the highlights below are the
migration-relevant items.

### ⚠ Behavior changes
- **URDF primitive collision (box/sphere/cylinder) now COLLIDES.** Previously
  such links were silently skipped (no collision body at all — e.g. the
  ridgeback base wheels). The importer now generates conservative,
  circumscribed proxy meshes (box exact; icosphere subdiv-2, faces pushed to
  >= r, ~2.4% inflation; 24-seg circumscribed cylinder, ~0.9%), merges all
  collision elements of a link into one body (per-element origins baked), and
  loads them through the normal mesh path. A scene that relied on primitive
  links being non-colliding (e.g. wheels resting through the ground plane)
  will now throw at finalize(). Escape hatch: `STIFF_URDF_PRIM_PROXY=0`
  (exact literal `0`) disables proxy generation and restores the old skip for
  primitive-only links. Mixed [primitive, mesh] links now use the first MESH
  element in BOTH modes (<=0.8.4 dropped the whole link's collision when the
  first element was a primitive).
- **MAS preconditioner numerics moved (within pcg_tol).** The batch-
  determinism fix below changes floating-point summation paths for every
  scene using the default `preconditioner_type=1`; solutions move ~1e-9-level
  relative to 0.8.4.1. Bit-exact comparisons against pre-0.8.4.2 goldens will
  differ; physics is equivalent.

### Fixed
- **MAS batch-invariance (strict determinism contract).** The same env
  produced ~1e-9-different results in an N=2 vs an N=8 batch. Three root
  causes, all fixed: (1) hierarchy depth was computed from the GLOBAL node
  count, so the level count itself changed with batch size — now derived from
  the per-env node count when env segmentation is active; (2) the multilevel-
  residual fast path issued `__shfl_down_sync` reads from padding lanes
  OUTSIDE its `__activemask()` — CUDA-undefined register garbage (~1e-21)
  that varied with launch shape — strict mode now uses the deterministic
  ascending-lane reduction; non-strict keeps a CORRECTED tree (no early
  return, all 32 lanes participate, padding contributes exact zeros, static
  bank-segment geometry replaces the former `mark << 32` shift UB); (3) the coarse-matrix fast path
  pre-summed segment blocks with plain double adds (layout-dependent
  rounding) and ran its warp collectives divergently under a full mask (UB
  on partially filled banks, present since <=v0.7) — replaced by exact
  order-independent binned deposits on the strict path and a corrected
  all-lanes tree on the non-strict fast path. examples/test_strict_quadgate.py
  now asserts run-to-run, cross-env AND batch-size bit-identity under the
  DEFAULT MAS preconditioner (plus a diagonal-PC cross-check gate).
- **MAS cluster-space arrays could overflow on small multi-env scenes**: the
  per-env padding can push a level's cluster count above vertNum; the scratch
  arrays (nextConnectMask/nextPrefix/goingNext) are now sized for the padded
  capacity and fully zeroed (found by compute-sanitizer memcheck; 0 errors
  after the fix, synccheck also clean).
- **Heterogeneous multi-env guard**: MAS env segmentation now verifies the
  FEM vertices are env-major contiguous with equal per-env counts before
  engaging (total divisibility alone was not sufficient); otherwise it falls
  back to the global hierarchy with a printed notice.
- **Unbounded intersection backtracking** (line-search type 0/1, boundary
  move): the three `while(isIntersected())` loops now share the line-search
  budget (`line_search_max_iter`, default 64) and throw a diagnosed
  initial-infeasibility error instead of halving alpha forever when the step
  already started intersecting.
- **von Mises export now shares the solver's constitutive model** (USE_SNK1:
  P = mu*F + r*(J-1-mu/r)*cofF): exported stress matches what the solver
  actually computes; degenerate elements (J~0) export NaN instead of a
  silent 0 that polluted statistics.
- **semi-implicit beta timing** (`semi_implicit_enabled=True` only; default
  stays off): the per-env beta decay now reads the PREVIOUS iteration's
  ACCEPTED line-search alpha (not the current iteration's unvalidated CCD
  candidate), and freeze flags take effect the NEXT iteration.

### Added
- **`get_vertex_contact_forces(components=...)`**: `"normal"` (barrier),
  `"friction_lagged"` (the frozen lastH friction force the solver actually
  used this frame; read-only, never rebuilds friction sets), `"total"`.
  Newtons; additivity normal+friction==total asserted at 0 error.
- **Joint tuning guide** `docs/JOINT_TUNING_v0.8.4.2_zh.md` (audit items
  A3/B3): why prismatic needs 20-40x the revolute ratio, symptoms of
  joint_strength_ratio=100 under load, three verified profiles.
- **Regression suite**: four-bar closed-loop (coupler-stays-translating
  assertion), gd_friction_rate direct assertion, strict quad-gate (5 gates),
  four M0 sentinels (D2 press / D7 ghost-drag / D10 dhat=5mm / A1 Z-up)
  promoted to exit-code regressions, URDF primitive containment + escape-
  hatch checks.
- **A1 (Z-up 10x slowdown) closed permanently**: strictly-equivalent rotated
  scenes (minimal + real fr3 URDF grasp contact) show a 1.000x iteration
  ratio (121 vs 121); the historical claim was a non-equivalent A/B
  measurement artifact. Sentinel guards against future axis bias.

### Fixed (continued)
- **MAS reorder kernels: intra-warp shared-memory races closed.** The six
  reorder kernels relied on pre-Volta warp lockstep for their shared-memory
  handshakes (zero -> vote -> publish -> cross-lane read with no
  synchronization; <=v0.7 legacy). All are now no-early-return with explicit
  `__syncwarp()` phase barriers. compute-sanitizer: racecheck 40 hazards ->
  **0**, memcheck/synccheck 0 errors (strict + non-strict, N=2/N=3 odd-bank,
  and a 19k-vertex 5-level hierarchy).
- **MAS env-segmentation hardening**: bank-range homogeneity guard (every
  METIS bank single-env, contiguous equal ranges — equal vertex counts alone
  are not sufficient); `STIFF_MAS_SEG` forced values that contradict the
  verified env count are ignored with a notice; 4096-env scratch cap
  enforced; env scratch freed in FreeMAS.

## [0.8.4.1] — 2026-07-22

Post-release audit hotfixes (two-agent problem audit, 4 review rounds).

### Fixed
- **`get_vertex_contact_forces` sign**: the export scaled the incremental-
  potential gradient by +1/dt^2; physical force is **-gradient/dt^2**, so every
  vector pointed the wrong way. Regression now asserts the SIGNED net vertical
  force on a resting cube is +mg (verified +627.2 N == weight). NOTE: behavior
  change for consumers of the v0.8.4 wheel.
- **Legacy force-API documentation told the wrong units** at seven sites
  across Python/pybind/C++ (claimed Newtons / "REAL grip force"): all now
  state the truth — raw dE/dx = -force*dt^2, body-body barrier only, no
  ground, no friction — and point at `get_vertex_contact_forces`.
- **merged DCD grow did not grow the CCD mirror**: the DCD detect kernels
  mirror every pair into `_ccd_collisonPairs` at the same slot; a dynamically
  grown DCD cap beyond the CCD cap made mirror writes (and the narrow-self
  snapshot) run out of bounds. Caps now grow in lockstep (per-env already did).
- **grow-redo stream races**: both merged re-detection passes (DCD and swept
  CCD) reset the pair counter on the default stream without an event/wait
  before the aux-stream detect — a race under `--default-stream=per-thread`.
  Mirrored the first-pass event/wait in both redo paths.
- `set_vertex_velocities_gpu` docstring now warns it does not rebuild xTilta
  (use `teleport_fem_vertices` for kinematic state changes).

### Audit
- Full problem-status matrix vs the requirement lists (what is fixed / partial
  / open, with per-item evidence): see the release audit document.

## [0.8.4] — 2026-07-22

Stability, contact-solver consistency, and public API hardening release.

### Added
- Passive revolute and prismatic joints through `passive=True`, plus active
  revolute-limit energy, gradient, Hessian, and device-side limit data.
- Per-soft-body density, per-body friction, and explicit ABD mass/inertia
  overrides through `set_abd_body_mass` and `set_abd_body_inertia`.
- Per-vertex contact forces in Newtons, FEM von Mises stress, per-environment
  Newton iteration/status telemetry, timeout freezing, and NaN/Inf quarantine.
- `Config` controls for per-environment exit, environment iteration caps,
  friction descent rate, collision buffer scaling, line-search budget, and
  optional absolute/relative energy comparison tolerances.
- Regression examples for passive joints, reversed parent/child body order,
  density, friction, contact force/stress, d-floor handling, and telemetry.

### Changed
- CCD step selection now carries direct `alpha` candidates and performs `MIN`
  reductions in merged, isolated, and strict modes. Reciprocal-alpha recovery
  and its divide-by-zero guard are removed; non-finite or non-positive active
  distances now fail loudly instead of becoming an unconstrained candidate.
- Isolated/strict contact stiffness uses the actual 12-dimensional ABD
  generalized degrees of freedom rather than counting collision vertices as
  independent 3D degrees of freedom.
- Segmented execution separates the fixed 256-slot capacity from the active
  group count, so a single active environment no longer processes 255 empty
  slots while retaining the isolated numerical path.
- Merged and isolated line searches share the same energy rule. The default is
  the exact comparison (`energy_abs_tol=energy_rel_tol=0`); configured
  tolerances use `abs_tol + rel_tol*abs(E0)` and are reported by a counter.
- The merged line-search budget is configurable and defaults to 64 halvings;
  budget exhaustion is always reported.
- Semi-implicit decoupled execution maintains and freezes per-environment beta
  values instead of re-coupling the batch through one global exit state.

### Fixed
- ABD contact Hessian assembly now writes `A + A^T` when two contact vertices
  map to the same rigid body, and uses a canonical transpose when body IDs are
  reversed. This fixes the hidden same-body folded-block asymmetry.
- Programmatic joint Hessian assembly no longer assumes
  `parent_id < child_id`; reversed IDs are canonicalized, validated, and
  surfaced to users instead of silently producing zero cross-body blocks.
- Partial-block CUDA reductions now keep all participating threads at barriers
  and use valid shuffle masks, removing undefined behavior in direct alpha,
  segmented PCG, and related reductions.
- Merged Newton convergence checks use the direction produced by the current
  PCG solve instead of a stale previous-iteration direction.
- Newton convergence is evaluated from the current unscaled solve direction,
  never from the displacement after CCD or line-search scaling. A contact-
  limited small `alpha` therefore cannot masquerade as convergence and make
  cloth motion under-resolved after impact.
- Per-group contact energy, stiffness, alpha, and frozen-environment handling
  now consistently use the owning environment's parameters.
- `teleport_fem_vertices` now applies the input-to-engine METIS permutation to
  positions and velocities; sorted cloth meshes no longer scramble on a
  get/teleport round trip.
- ABD preconditioner accumulation, ground-contact Hessian PSD projection,
  surface-mesh inertia validation, semi-implicit exit scoping, and persistent
  ground-distance invariant handling were hardened. Closed surface
  meshes with globally reversed winding now normalize all signed mass moments
  together instead of being rejected as negative mass.
- Ground distance validation now rejects non-finite or non-positive distances
  immediately while preserving every finite positive distance as feasible.
- Ground CCD now follows libuipc's relative fraction-to-boundary semantics:
  direct `0.9*d/c` alpha candidates reduced by `MIN`, with no absolute distance
  floor. After trial coordinates are written, merged and per-environment line
  searches verify strict positive distance and backtrack any alpha whose
  floating-point update lands on the barrier boundary. Accepted-displacement
  convergence covers CCD-, CFL-, and energy-limited steps.
- Zero-contact Hessian partitioning now returns without launching radix-sort or
  zero-length CUDA copies, and contact partitioning now reserves its exact
  out-of-place `[n,2n)` reorder range before use. Contact-free and large-contact
  ABD scenes therefore avoid invalid CUDA copies and scratch overflows.
- URDF import now warns about unsupported primitive collision geometry instead
  of silently skipping it.

### Fixed (post-candidate, final release)
- **strict cross-env bit-identity restored: DCD-time CCD snapshot.**
  `_ccd_collisonPairs` has two producers — the DCD detect kernels write a
  mirror copy of every DCD pair at the same atomic slot, and the swept-CCD
  build overwrites the whole buffer every Newton iteration. The S1 narrow-self
  feasibility sweep was designed to read "the first h_cpNum[0] entries = the
  DCD mirror" but actually swept a race-ordered, env-unbalanced prefix of the
  PREVIOUS swept emission. Measured consequences: strict cross-env broken
  (two bit-identical envs forked ~1e-2 by frame 200 via an asymmetric
  narrow-self alpha), strict N=8 run-to-run unstable (concurrent-emission
  race), and a self-sustaining hs -> ta_e feedback loop. The latent defect
  predates v0.8.3, which passed its gates on emission-timing luck. buildCP now
  snapshots the mirror into a dedicated immutable buffer consumed in full by
  both narrow-self paths. Release gates after the fix: run-to-run N=2 and N=8,
  cross-env, and batch-invariance ALL BIT-IDENTICAL (N=8 run-to-run stable for
  the first time). Per-env direct-alpha scratch is also neutrally initialized
  (independent defect: the first swept build read cudaMalloc garbage).
- **One-sided revolute limits: lagged active-set.** The limit term re-tested
  theta every Newton iteration, so at an active bound the active set CHATTERED
  (inactive-side Hessian has no limit curvature -> huge free-fall direction ->
  line search cuts alpha to ~1e-7 -> next iterate lands active and is pushed
  back out). Every frame after first bound contact ran to the 1000-iteration
  cap (~19 s/frame). The activation flag is now frozen once per frame from the
  previous converged angle and released by the multiplier-sign test (a
  one-sided limit may push, never pull). Limited hinge: cap-outs -> 1
  iteration/frame, momentum overshoot then exact settle at the bound. Known
  trade-offs: bound contact engages one frame late (overshoot ~ v*dt) and an
  engaged frame is two-sided until release.

### Validation (final build: alpha-exit removal + lagged limits included)
- ModelScope plate teleop replay (228 frames, the only teleop-format
  trajectory) completed in every mode with no CCD guard, line-search warning,
  NaN, or cap-out. Peak Newton iterations: 25 (merged), 36 (isolated),
  45 (strict); slowest frame 1.34 s (strict). Merged peak is run-to-run
  variable by design (atomic reductions); strict is reproducible.
- ModelScope plate collect task (config_plate_new): grasp completed at frame
  82 in all three modes; peak Newton 3-4, physical frames < 0.1 s.
- ModelScope beaker task (config_beaker_soft_new): completed 93 frames in
  1.1 s worst-frame, peak Newton 11 — the historical 13-19 s/frame stall
  (frames 80-92) is gone. The pre-July-18 config_beaker_soft no longer loads
  under the current env stack (robot joint renames), unrelated to the engine.
- Passive-limit (free + limited), reversed joint-order, density, friction,
  force/stress, kick/ABD-preconditioner, per-environment telemetry, and
  ground-domain regressions pass on the final build.
- Historical solver baseline for the same plate replay: 0.8.2 peaked at 621
  Newton iterations (16.7 s single frame) at the grasp; the final build peaks
  at 76 or lower depending on mode.

### Known limitations
- Merged, isolated, and strict modes share physical parameters and acceptance
  rules, but do not promise identical trajectories: segmented reductions,
  per-environment line search, and nonlinear contact active-set changes can
  legitimately select different floating-point paths.
- Line-search exhaustion remains a loud warning followed by acceptance of the
  final candidate for compatibility; production callers should treat it as a
  solver-health failure and adjust time step, drive stiffness, or the budget.
- Friction anchors remain lagged within a Newton solve. A 0.01 s time step is
  the supported mitigation until the close-set/kappa pipeline is rebuilt per
  Newton iteration.

## [0.8.3] — 2026-07-07

Multi-env MAS preconditioner determinism — resolves the v0.8.2.1 known limitation.

### Fixed
- **strict + MAS preconditioner (`CASE39_PRECOND=1`) cross-env / batch / run-to-run
  bit-identity** (the v0.8.2.1 "MAS × multi-env is WIP" limitation). Root cause: the MAS
  builds ONE global hierarchy over all envs; at coarse aggregation levels the (physically
  independent) envs' clusters land in the same `BANKSIZE` bank, so the block-diagonal
  Schwarz smoother's per-bank solve couples envs — violating the block-diagonal multi-env
  PCG assumption that each env is an independent solve. That made the MAS preconditioner
  env-asymmetric AND, under strict, pathologically slow (block-diagonal solver vs a
  coupling preconditioner → poor convergence; the full trajectory did not finish in 60 min
  at N=8). Fix: **per-env-segmented aggregation** — each env's clusters are padded to a
  `BANKSIZE`-aligned block at EVERY level, so envs never share a bank and the smoother
  stays intra-env. All-device (3 small integer kernels, no host round-trip; exact → zero
  determinism impact). Verified (foldshirt, RTX 4090): strict+MAS env0==env1 (cross-env
  0.0), env0@N=2==env0@N=8 (batch invariance 0.0), run-to-run identical, for N=2/4/8.

### Changed
- **per-env MAS is now the default for multi-env** (bodies in >1 group) in every mode
  when the MAS preconditioner is active. The old global MAS aggregates non-interacting
  envs into shared coarse banks — meaningless (the system is block-diagonal per env) AND
  slower. Single-env → no segmentation. `STIFF_MAS_SEG` overrides: `0` = force off
  (measure the old global MAS), `1` = force on (body-group count), `N` = force N envs.
  The diagonal preconditioner (`CASE39_PRECOND=0`) is unaffected (it has no hierarchy);
  strict on the diagonal path was already bit-identical since v0.8.2.1.

### Performance (foldshirt full trajectory ~1618 frames, RTX 4090, ms/frame)
- **strict N=8: MAS (per-env) 608.9 vs diagonal 711.2 → 14 % faster.** per-env MAS is
  the only usable MAS at strict scale (global MAS is pathological) and beats the diagonal
  preconditioner (fewer PCG iterations → less of strict's per-iteration binned work).
- merged/isolated: the diagonal preconditioner remains fastest (merged N=8 = 432.9);
  with MAS, per-env is the *correct* hierarchy but a few % slower than the old global MAS
  in these non-bit-identical fast paths (the extra per-env padding nodes are not offset by
  convergence gains, as MAS does not pay off there). Use `CASE39_PRECOND=0` for speed.

### Hardening
- The multilevel-R restrict reduction (partial-cluster warp `else` branch) used a
  non-deterministic float `atomicAdd`; it is now a deterministic per-cluster ordered sum
  (own-slot deposit + `__syncwarp` + ascending-lane sum). Defensive — no measured effect,
  but removes a latent non-deterministic reduction on the strict path.

## [0.8.2.1] — 2026-07-07

Determinism correctness patch.

### Fixed
- **strict cross-env / batch bit-identity regression** introduced by the v0.8.2
  seg-dot warp pre-reduction. The per-warp `__shfl` tree pre-sum is a plain
  (non-binned) float add AND its warp grouping is keyed on the global DOF index
  (env *k* starts at offset Σⱼ Nⱼ, not 32-aligned), so env0 and env1 summed
  different lane groups → env0 ≠ env1 by ~7 ULP at Newton k=1, amplified by
  stiff contact to ~8 mm by frame 0. It preserved run-to-run (fixed layout
  within a run — all v0.8.2 verified) but broke cross-env AND batch invariance.
  The pre-reduce is now **positively gated OFF for strict** (`STIFF_SPMV_DET`);
  merged/isolated keep it. Verified: foldshirt strict (non-MAS diagonal precond)
  env0==env1 and env0@N=2==env0@N=4 bit-identical (0.0) across all frames.

### Corrected
- The v0.8.2 changelog's **strict −24% / seg-dot −26%** figure is **retracted**
  — it timed a determinism-broken strict. Correct strict keeps the per-lane
  binned seg-dot (~111 µs). Recovering a warp pre-sum for strict deterministically
  needs per-env DOF padding to 32 (future work).

### Known limitation
- The **MAS preconditioner** (`CASE39_PRECOND=1`) is separately cross-env
  asymmetric (its warp-bank clustering is keyed on the global vertex layout);
  multi-env determinism is only guaranteed on the diagonal preconditioner path.
  MAS × multi-env is WIP.

## [0.8.2] — 2026-07-04

Memory- and speed-focused release. Foldshirt multi-env N=8 (200 frames,
RTX 4090, ms/frame): merged 1312.7 → **984.4** (-25 %), strict 2514.4 →
**1921.8** (-24 %). Red-cloth full-episode max envs on 24 GB: **16 → 30**.

### Added
- **Block-upper-triangular Hessian storage (SymGH)**: only `row ≤ col` blocks
  are stored/assembled; the SpMV mirrors the transpose. −37.5 % triplets.
  Combined with the allocator fixes below: red-cloth max N **16 → 30** (+87 %),
  plus ~6-14 % speed from the smaller assembly/SpMV stream.
- **Pre-assembly discard-growth for triplet buffers**: capacity grows via
  `cudaFree`-then-`cudaMalloc` (no copy, no old+new double-residency) at the
  point where the exact extent is known; growth margin capped at an absolute
  512 MB (was multiplicative 2×1.3).
- **seg-dot warp pre-reduction** (`STIFF_SEG_WARP=0` opts out, `=2` prints a
  d2g warp-layout audit): per-warp fixed-tree `__shfl_down_sync` pre-sum emits
  ONE shared-bin deposit per warp instead of 32. The per-env dot kernel drops
  111.5 µs → 14.2 µs (7.9×); a strict frame drops 26 %. Bit-identity and
  batch-invariance preserved (fixed lane→DOF map, fixed tree order,
  N-independent env DOF ranges); env-boundary warps fall back to per-lane.
- **Targeted `__launch_bounds__` variant for the EE broad-phase**
  (`STIFF_EE_LB=2`, defaulted for **merged**): 168 → 128 reg/thread doubles
  resident warps on the latency-bound `_selfQuery_ee` (occupancy 11.8 %,
  DRAM 0.3 %); spills are only 112 B/thread into an idle L1. Merged −5.6 %.
  `STIFF_EE_LB=3` (80 reg) exists for experiments but measured **+52 %**
  (L1 thrash against the 8 KB/thread traversal stack) — do not use.
- **BVH stack-depth probe gated** (`STIFF_STACK_DIAG`): the per-pop
  `atomicMax` on a single global address in all four traversal kernels was
  always-on; it is now diagnostics-only.

### Fixed
- **Strict-mode run-to-run bit-identity regression** (from the det-gating
  refactor): the central `g_det_reduce` gate defaulted to the fast path, so
  `binned_deposit` users firing before the first `computeGradientAndHessian`
  latch (frame-0 init energy reductions) accumulated in non-deterministic
  order and seeded divergence at the first line search. Default is now
  conservative (binned); the latch only relaxes merged/isolated. Verified:
  rz0 probe 17-digit identical across runs and bit-equal to the pre-det-gate
  baseline.
- **Determinism machinery positively gated**: strict opts *in* via
  `STIFF_SPMV_DET`; merged/isolated never pay the binned/ybin/seg-binned tax
  and may use CUDA-graph PCG. A stray `STIFF_FAST_GRAD=1` can no longer
  silently break strict's bit-identity contract (the flag is not read at all).

### Changed
- **`pcg_tol` default back to `1e-4`** (the 0.8.0/0.8.1 shell silently
  defaulted to `1e-6`, costing ~22 % at N=1 with no accuracy requirement
  behind it).
- The shell no longer sets `STIFF_FAST_GRAD` for merged/isolated (dead flag);
  merged defaults `STIFF_EE_LB=2`.

## [0.8.1] — 2026-07-04

### Fixed
- **Per-env machinery vs undeclared groups**: `d_point_to_group` is always
  allocated (all `-1` wildcard when `set_body_groups` was never called), and
  pointer-nullness gates half-engaged the multi-env machinery: per-group κ
  kernels read `kappa_grp[-1]` (out-of-bounds barrier stiffness), the per-env
  line search validated itself with zero env coverage (Newton pegged at the
  iteration cap), and the per-env BVH excluded every primitive (**silent loss
  of all self-collision**). All activation sites now require a real
  `groups_present` flag; per-group kernels guard `-1` (mixed grouped+wildcard
  scenes are now correct too). isolated/strict without groups run
  merged-equivalent and print a one-line warning.

### Changed
- **Convergence threshold sources** (design: env-scale relative):
  with declared groups, the merged-mode Newton exit uses the AVERAGE per-env
  rest bbox and per-env freeze checks use each env's OWN bbox — env-count
  invariant without reading `absolute_dhat`/`relative_dhat` (the
  `abs²/rel²` reverse-solve is removed from the exit path; `relative_dhat`
  is now fully inert for convergence). Ungrouped scenes keep the legacy
  whole-scene-bbox exit bit-for-bit.

### Added
- **`newton_velocity_tol`** (m/s, default 0 = off): opt-in uipc-style physical
  Newton exit (`max step displacement ≤ v·dt`), uniform across
  merged/isolated/strict; scene-size and env-count independent.

## [0.8.0] — 2026-07-03

> The "multi-env engine" release: v0.6.7 unified with the entire per-env
> solver campaign. One tree now carries both the IsaacLab/Newton integration
> APIs (v0.6.5–0.6.7) and the three-tier multi-env execution engine.
> (0.7.x was an experimental perf series, later audited: its beneficial
> changes are included here; its unstable ones are not.)

### Added
- **Multi-env execution modes** — `Config(multienv_mode=...)` or
  `STIFF_MULTIENV_MODE`, three tiers:
  - `"merged"` (default): all envs in one solve, fastest. κ/dHat now use the
    absolute-dhat scale (see *Changed*), so contact physics no longer softens
    as envs are added.
  - `"isolated"`: per-env decoupled physics — env-id collision isolation,
    per-env broad-phase BVH (K-stream concurrent build), per-env κ, per-env
    line-search α, and a segmented block-diagonal PCG with per-env
    α/β/convergence. Each env is an independent, physically-correct sim.
  - `"strict"`: isolated + full determinism machinery (canonical contact
    ordering, order-independent binned reductions, deterministic SpMV).
    env_0 is BIT-IDENTICAL run-to-run, across mate content, and across env
    COUNT (N=2/4/8 verified 0.000 at release).
- **Per-group world offsets** — `Engine.set_env_offsets` (render/broad-phase
  separation while narrow-phase stays in local frames; the substrate for
  strict-mode local-frame layouts).
- **`Engine.get_point_groups`** — per-vertex env id aligned to
  `get_vertices()` order (extract any single env's vertices).
- **Full-state checkpoint** — `Engine.save_checkpoint` / `load_checkpoint`
  (FEM+ABD+κ state; deterministic mid-trajectory restart).
- **Per-contact force export** — the exact I5/NEWF barrier-gradient kernel
  now optionally attributes per-pair forces (`GIPC::exportContacts` hook),
  wired through the batched contact-force readback APIs.

### Changed
- **Default `pcg_tol` 1e-4 → 1e-6.** The 1e-4 default was inherited from
  upstream and never tuned; with correct (absolute-dhat) contact stiffness it
  explodes Newton counts (measured 1407 vs 507 total Newton over 30 frames on
  a multi-env grasp; net time strictly worse). Stiff-contact scenes may
  benefit from 1e-8 (`pcg_tol` arg or `STIFF_PCG_TOL`).
- **κ scale consistency**: when `absolute_dhat > 0`, suggest/upper-bound κ and
  the Newton convergence threshold all use the effective (absolute) scale in
  ALL modes — merged multi-env scenes previously diluted κ through the
  whole-scene bbox (softer, batch-dependent physics).

### Performance
- Segmented PCG: CUDA-graph replay of the inner iteration block (K=8),
  spmv-fused and preconditioner-fused per-env dot products, cub-based Morton
  sort on preallocated scratch (no per-sort device sync), K-stream parallel
  per-env BVH pool. Isolated-mode cost per env reduced ~46% over the campaign
  (942.7 → ~505 ms/env on the N=4 grasp reference).
- Batched host↔device control-scalar traffic (v0.7-audit ports): contact
  Hessian partition ids 4→1 D2H, energy reductions 9→1, line-search feasible
  steps 2→1, cpNum+gpNum 2→1. Friction/close-constraint buffers are now
  persistent grow-only allocations (no per-frame cudaMalloc/cudaFree device
  drains); reduction workspaces sized by pair capacity (removes a latent OOB
  class).
- Release-gate perf reference (N=4 grasp, 30f): merged 397 ms/env,
  isolated 475 ms/env, both faster than either parent tree.

### Fixed
- **Batch-size invariance**: per-env line-search S3 backtrack decisions moved
  fully on-device — a stale host mirror previously destroyed per-env α state
  when a backtrack fired, making env_0 depend on the env COUNT.
- **meanMass batch invariance** (strict): κ's scale is seeded from env_0's
  intensive per-vertex mean instead of a serial FP sum over all envs.
- Ground d=0 NaN guard, `set_vertex_boundary` metis-order remap,
  `teleport_fem_vertices` FEM-block offset, OBJ `f v//vn` parsing (inherited
  from the 0.6.x line and preserved through the unification).
- Python module loader: ABI-tagged build dirs resolve before generic
  `build/` (mixed py3.11/3.12 worktrees no longer load a stale module).

### Known limitations
- `isolated` mode decouples physics but does not promise bit-identity; use
  `strict` for reproducibility experiments (~2× the isolated PCG cost).
- MAS preconditioner (`preconditioner_type=1`) is unvalidated with the
  multi-env machinery (known Newton-cap regression on one hybrid scene);
  multi-env modes currently assume `preconditioner_type=0`.

## [0.6.7] — 2026-07-02

> The "unification" release: v0.6.6 + exactly the changes proven necessary by
> the IsaacLab/Newton multi-env RL validation campaign (each item gated by a
> dedicated test; see `docs/STIFFGIPC_UNIFY_PLAN.md` in IsaacLab-3.0).

### Added
- **Per-vertex environment isolation** for contact: `Engine.set_vertex_env_ids`
  (+ `SimEngine::set_vertex_env_ids` / pybind). The broad-phase skips any contact
  pair whose two vertices carry different (>=0) env ids — uniform FEM+ABD
  cross-env isolation WITHOUT spatial separation, decoupled from the solve
  (contact filtering only). env id < 0 = shared geometry. Unlike the per-body
  skip matrix this isolates a combined FEM body whose particles span many envs.
- **Engine-native joint limits** (revolute + prismatic): mass-scaled one-sided
  penalty spring toward the violated bound (`joint_limit_strength_ratio`,
  default 20000; implicit energy with gradient + SPD Gauss-Newton Hessian —
  a MuJoCo-style penalty, NOT a log-barrier). 0 disables.
- **`absolute_dhat` passthrough** in the Python `Config`/`Engine` shell — the
  engine's fixed-dHat mode existed but was unreachable from Python (the shell
  never wrote it into `SimEngineConfig`, so it silently stayed 0/relative).

### Fixed
- **Newton convergence threshold is env-count-independent**: `gradVanish` now
  uses the effective bbox (`eff_bboxDiagSize2` = absolute_dhat²/relative_dhat²
  when absolute_dhat>0) instead of the raw whole-scene bbox, so convergence no
  longer loosens as envs are tiled (was: N-env scenes converged to a different
  quality than 1-env).
- **Ground-barrier d=0 guard no longer freezes grounded bodies**: the
  `fmax(dist2, 1e-12)` clamp capped the barrier force near d=0 (force should
  → ∞), collapsing the line-search alpha and freezing any vertex resting
  exactly on the ground. Replaced with the conditional
  `dist2 == 0.0 ? 1e-12 : dist2` (NaN guard only at exact zero) in all 4 sites
  (gradient/Hessian, energy, friction lambda). Strict non-penetration preserved.
- **`teleport_fem_vertices` respects the FEM block offset** (`fem_offset`) in
  mixed ABD+FEM scenes — per-env FEM resets previously wrote from vertex 0
  (ABD vertices) instead of the FEM block start.
- **OBJ loader parses `f v//vn` faces** (double-slash, no texcoord): the missing
  branch pushed an uninitialized face index → CUDA out-of-bounds / frozen soft
  bodies when loading normal-split OBJ exports.

## [0.6.6] — 2026-06-24

> Released without a changelog entry at the time; recorded retroactively.
> Highlights (relative to 0.6.5): engine-side absolute dHat mode
> (`absolute_dhat` → fixed contact thickness via effective bbox), gravity
> body-force rotation term for trimesh ABD bodies (`compute_trimesh_body_force`
> full integral — frozen-articulation gravity torque fix), revolute slew cap
> (`max_revolute_step_per_frame`), collision exclusion/groups/ground-skip APIs,
> per-body density & inertia setters, and the corresponding pybind bindings.

## [0.6.5] — 2026-06-22

> v0.6.5 = v0.6.4 + the full multi-env backport (engine + examples) and the
> UMI finray soft-gripper suite, plus a fixed-joint correctness fix. Stays on
> the conservative 0.6.x line (no v0.7.x perf rewrite). The wheel delta over
> 0.6.4 is the engine multi-env layer + the fixed-joint fix; everything else is
> example code.

### Added
- **Multi-env (block-structured)**: per-env group/collision isolation
  (`set_body_groups`), block-structured DOF layout + segmented per-env
  reduction (P2a/P3a), per-env line-search (S1–S4, validated EXACT vs the
  global solve), dynamic global-triplet buffer (~3× smaller per-env memory,
  never-overflow), configurable `triplet_internal_margin`.
- **UMI finray soft-gripper example suite**: beaker / cupshirt / foldshirt ×
  pos / stitch / force × single-replay / multi-env / interactive-UI
  (`examples/umi_finray_lib.py` + thin entry scripts).
- **Duck multi-env scaling example** (rigid grasp → 512 envs).
- **Batched contact-force readback** `get_body_contact_force_batched(offsets, counts)`
  — rebuilds contacts ONCE and sums every finger in one kernel + one D2H, vs the
  per-body `get_body_contact_force` which rebuilt contacts on every call. Used by
  the finray pinch latch; ~2.1× faster multi-env force replay (foldshirt N=10:
  795s → 374s, approaching the pos/stitch ~322s solve floor).

### Changed
- **`force` gripper mode** rewritten to *force-control-through-position-drive*:
  soft-K impedance close, with an optional contact-stop **pinch** (default on,
  `GRIP_PINCH=1`) that latches on the finray IPC contact force so the jaw stops
  at the object surface instead of closing dead. Cloth (contact≈0) falls through
  to a full close — no rigid/cloth classification. The IPC limit barrier is no
  longer used (pure position drive cannot overshoot a limit), which also removes
  the multi-env merged-solve barrier ill-conditioning.

### Fixed
- **Fixed joint** now penalizes all 3 affine basis axes (t, n, b), matching
  libuipc `affine_body_fixed_joint` — removes single-anchor rotational slack
  (rank-6 → rank-9 constraint Hessian).
- **`absolute_dhat`**: contact thickness no longer inflates with scene size
  (was making contact super-linear in multi-env).
- **`load_triMesh`**: multi-cloth vertex offset + per-body bending edges.
- Expose `get_prismatic_drive_force` / `get_prismatic_current_distance` on the
  Python `Engine` (C++ bindings shipped in 0.6.4; Engine wrappers were missing).
- finray examples resolve the assets dir **case-robustly** (`assets/` → `Assets/`
  fallback) so a fresh clone of the tag — which only tracks capital `Assets/` —
  finds them instead of 404-ing on the gitignored lowercase symlink.

## [0.6.4] — 2026-06-18

> v0.6.4 = v0.6.3 + an **ABD joint force-control** layer (all new APIs are
> default-OFF, so existing scenes are byte-identical). Stays on the conservative
> v0.6.x line (no v0.7.x perf rewrite).

### Added
- **External force / torque / velocity on ABD joints** (libuipc-aligned, via the
  `q_tilde` path — no Newton Hessian term): `set_revolute_torque`,
  `set_prismatic_force`, `set_body_external_wrench`, articulation velocity control,
  plus `get_prismatic_drive_force` / `get_prismatic_current_distance` readouts.
- **`set_prismatic_limit_barrier(idx, cl, dir, dhat, kappa)`** — a one-sided IPC
  log-barrier on a prismatic joint's closed limit, so the solver guarantees the
  opening never crosses the limit under pure-force control (no overshoot). Off by
  default (`kappa=0`).
- **On-GPU stitch-stretch reduction** — `get_stitch_max_stretch` and
  `get_stitch_max_stretch_batched` (one CUDA block per finger/env; scalar result,
  no full-vertex device→host copy) for cheap soft-gripper grip sensing.
- **Examples**: force/torque/velocity demos (`examples/case_force_*.py`,
  `test_force_*.py`) and finray soft-gripper grip-control demos
  (`replay_case39_UMI_obb_cup_shirt_forcegrip.py`, `case_umi_finray_force_ui.py`)
  showing feedback-driven position vs pure-force + barrier control.
- **Docs**: `docs/FORCE_CONTROL_DESIGN.md` (force-path design + soft-gripper
  grip-control study).

## [0.6.3] — 2026-06-16

> v0.6.3 = v0.6.2 + two GPU collision-buffer overflow fixes + a new UMI
> finray gripper example suite (replay + interactive UI, detailed and
> OBB-collision arm variants).  Stays on the conservative v0.6.x line — it
> does **not** pull in the v0.7.x perf rewrite (CUDA-Graph PCG,
> sparsity-cache, pair-type sort).  Recommended build for RL data collection.

### Fixed

- **CCD reduction-scratch overflow (the OBB-arm + soft-gripper blow-up).**
  `pcg_data.squeue` was sized for `max(vertexNum, tetrahedraNum)` doubles but
  reused as the scratch buffer for collision-pair-count reductions.  When the
  number of CCD pairs exceeded that capacity (e.g. a fin-ray soft gripper
  reaching deep into cloth behind an OBB-collision arm), the reduction wrote
  out of bounds and corrupted adjacent GPU memory — `_moveDir` blew up to
  ~1e29 and the sim exploded.  A dedicated, dynamically-grown reduction
  scratch buffer (`ensure_reduce_scratch`) now sizes to the actual pair count
  every frame.  A detailed-collision arm masked the bug only because it
  produced fewer CCD pairs; it was never arm-specific.

- **Collision pair-emission buffer overflow.**  The narrow-phase /
  swept-CCD kernels `atomicAdd`-emit pairs into fixed-capacity buffers
  (`_collisonPairs`, `_ccd_collisonPairs`, `_MatIndex`).  Emits past the cap
  are now redirected to a guard ("trash") slot so they can never write out of
  bounds, and the host grows the buffers and re-runs detection until every
  pair fits — no pairs are silently dropped and overflow is impossible (it
  degrades to higher VRAM use, never corruption).

- **Stale metis sort-cache corruption (mixed rigid-object + hybrid scenes).**
  The metis sort/partition disk cache is keyed only on the input mesh's
  basename-stem.  `load_mesh_from_data` names its temp mesh
  `tmp_mesh_<load_count>.msh`, so the same finray mesh maps to different stems
  depending on load order, and one stem can be reused across runs for meshes
  of different vertex counts.  The cache-hit path trusted any matching-stem
  file, so a stale permutation sized for a different mesh got applied to a
  freshly written one — corrupting the hybrid finray FEM offsets (gap=0 stitch
  wired to wrong vertices → large gaps + ~100× slowdown).  Surfaced by the new
  beaker-grasp replay with the v800 LOD finrays (L=816, R=820 verts).  The
  cache hit is now validated against the input mesh's actual vertex count (via
  the `.idx` sidecar) and regenerated on any mismatch.

### Added

- **UMI finray gripper example suite** (`examples/`):
  - `replay_case39_UMI_sf.py` — replay a recorded trajectory on the
    fold-shirt scene with a STRATEGY_F hybrid fin-ray gripper (rigid ABD
    mounting root + FEM truss, zero-gap stitch).
  - `replay_case39_UMI_sf_obb.py` — same, with the arm links replaced by
    oriented-bounding-box collision geometry (faster broad phase).
  - `case_umi_finray_ui.py` / `case_umi_finray_ui_obb.py` — interactive
    slider control of the arm joints and grippers (detailed / OBB variants).
  - `replay_case39_UMI_beaker.py` — beaker-grasp replay (the gripper picks up
    a rigid 100 ml beaker) + the bundled beaker trajectory and collision mesh.
- **Level-of-detail fin-ray FEM assets** (`Assets/sim_data/`):
  `umi_hybrid_sf_v800` (default, ~816 verts/finger), `_v1340`, `_v1690`.
- **OBB-collision arm URDF variants** plus their `meshes/obb_umi/` boxes
  under `Assets/sim_data/urdf/ridgeback_dual_panda_UMI/`.
- **Mesh-prep tools** (`examples/build_umi_obb_urdf.py`,
  `examples/build_umi_finray_strategyF.py`).

## [0.6.2] — 2026-06-05

> **For RL data collection / long replays, this is the recommended build.**
>
> v0.6.2 = v0.6.1 + 3 backported fixes from the v0.7.x line + 1 small CCD
> tweak.  It deliberately does NOT pull in the v0.7.0 perf rewrite
> (CUDA-Graph PCG, sparsity-cache, pair-type sort) — that batch contained
> commit `98ac3dc` (CCD-pair sort) which caused random Newton-cap stutter
> on long contact-dense trajectories.  See [0.7.1] for the in-place fix on
> the v0.7 line; users who can't tolerate any tail-event flakiness on long
> trajectories should pin v0.6.2 instead.

### Fixed

- **Statistics overhead (long-replay perf + RAM)** — backport of v0.7.1
  fix.  `Statistics::write_to_file` was called every step and serialized
  the entire accumulated `m_json["frames"]` array (4-space indent) to disk
  → O(N²) cumulative writes (~800 MB per 1551-step run).  Plus
  `m_json["frames"][m_frame]` grew ~50 KB/step in RAM (~8.5 GB after
  100 episodes → eventual OOM on long RL data collection).

  Both behaviors now gated behind `GIPC_STATS_ENABLED` env var (default off).
  Disabled mode uses a per-frame scratch `Json` for writes so caller code
  is unchanged.

  To re-enable full stats.json dumping:
  ```bash
  GIPC_STATS_ENABLED=1 python your_script.py
  ```

- **`cudaEventDestroy` leak in `IPC_Solver`** — backport of v0.7.1 fix.
  Per-step timing events were created but never destroyed.  Long-running
  RL collection processes could eventually exhaust the driver's
  event-handle pool.  Resource-hygiene fix.

### Performance

- **`isIntersected()` post-line-search check now skipped by default.**
  `_edgeTriIntersectionQuery` (the BVH self-intersection re-test after
  every line-search α bisection) was 42 % of GPU time on case_39.  CCD
  line-search already guarantees a non-intersecting step on smooth
  contact; the recheck never fires on those scenes.  Default flipped to
  SKIP (was opt-in via `STIFF_SKIP_CCD_SANITY=1`).

  Effect on `replay_case_39` (cup grasp, smooth contact, n=3 paired):
  ~38 ms/step → ~32 ms/step (~+15 %).

  To restore the v0.6.1 behavior (e.g. on scenes with very thin tets or
  near-coincident features):

  ```bash
  GIPC_FORCE_CCD_SANITY=1 python your_script.py
  # or, back-compat with the v0.7.0 opt-out form:
  STIFF_SKIP_CCD_SANITY=0 python your_script.py
  ```

### Why not just use v0.7.x?

v0.7.0 introduced a ~14 % per-step speedup via a bundle of optimizations
(CUDA-Graph PCG, sparsity-cache, pair-type sorts), but the CCD-pair sort
in that bundle (`98ac3dc`) made Newton occasionally fail to converge
on long contact-dense trajectories — symptom was random 1-2 s "frozen
frames" in cloth-fold replays.  v0.7.1 reverts that single commit, but
the other v0.7.x rewrites still change kernel timing in ways that some
users prefer to avoid for long RL data collection.

v0.6.2 is the conservative path: same numerical core as v0.6.1, only
backports the proven-safe fixes.

## [0.6.1] — 2026-05-27

### Fixed

- **MAS preconditioner now works with stitch springs** (`sim_engine.cu`):
  when `Config(preconditioner_type=1)` (MAS) was used together with FEM bodies
  that had `add_stitch_spring` constraints, Newton would stall at the
  iteration cap because stitch indices were passed in input-mesh order while
  the FEM solver remapped vertices to metis-sorted order.  Fixed by
  translating stitch indices through `vertex_metis_to_input` at upload time;
  `preconditioner_type=0` and pure-FEM scenes are unaffected.  Effect on the
  bundled `replay_case39` example: ~1.9x speed-up (9.4 → 18.0 fps) versus the
  previous mandatory MAS-off fallback.

### Added

- **`SimEngine.set_per_tet_young_for_body(body_offset, per_tet_young)`**: per-tet
  Young's modulus override for one FEM body, callable before `finalize()`.
  Lets one mesh have different stiffness regions (e.g. stiff contact face +
  soft interior).  Array length must equal the body's tet count (tets all of
  whose 4 vertices lie in the body's vertex range).
- **`examples/replay_case39.py`**: pre-recorded qpos trajectory replay on the
  case_39 dual-panda + soft-gripper + cup + shirt scene, with defaults tuned
  for successful cup pickup (`FRICTION=0.8`, `CLOSE_RATIO=0.5`, `PRECOND=1`).
  Includes an opt-in `CASE39_YOUNG_PART2 / PART3` knob that demonstrates
  per-region Young's modulus.  Bundled trajectory:
  `assets/trajectories/qpos_case39.h5` (29 KB, 1018 frames).

### Changed

- **`case_39_full_scale.py` and `case_40_unified.py`**: default
  `preconditioner_type` is now MAS-on (`CASE39_PRECOND=1` / `CASE40_PRECOND=1`)
  now that the stitch fix above makes MAS correct on hybrid grippers.  Set
  the env var to `0` to fall back to the prior diagonal-preconditioner
  behavior.

### Assets

- **Added missing assets** that the `case_39_full_scale.py` example shipped in
  v0.6.0 referenced but that were never tracked: `ridgeback_dual_panda2_mobile_s1_softgripper.urdf`,
  `ridgeback_dual_panda2_mobile_s1_full.urdf`, hybrid_d STRATEGY_F (`rigid.msh`,
  `rigid_remap.npz`, `unified.npz`), `softgriper_part2.msh`, `softgriper_part3.msh`.
  v0.6.0 users who tried to run `case_39_full_scale.py` would hit
  "URDF file does not exist" — v0.6.1 fixes that.

## [0.6.0] — 2026-05-15

### Added

- **`SimEngine.set_log_level(level)`**: control per-frame solver log verbosity
  (0 = silent via `std::cout` redirect + `printf` gating, >= 1 = verbose
  default).  For co-simulation and piped tooling.
- **`SimEngine.reset()`**: tear down the whole world (bodies, FEM/ABD,
  constraints, GPU buffers) and return to an empty state with the current
  `Config` preserved.  Re-run `load_*() + finalize()` afterwards without
  recreating the `Engine`.
- **Hybrid-gripper example improvements** (`case_39_full_scale.py`,
  `case_40_unified.py`):
  - default to `CASE40_BRIDGE_B` hybrid gripper mesh (bridge-filled
    finger↔soft gap, fTetWild 0% bad tet, `rigid:FEM` 0.18); FEM young `1e8`.
  - `FJ_ANCHORS=3` multi-anchor green→finger fixed joint to fix the
    anchor/finger split observed in the single-anchor variant.

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

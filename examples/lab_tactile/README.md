# Tactile lab-bench demo (FR3 + VBTS finger pads, StiffGIPC)

Five-step chemistry protocol run by a Franka FR3 whose fingertips carry two
vision-based tactile pads (FEM gels, `stiff_physics.sensors`):

1. take the rubber-bulb **dropper** out of the Erlenmeyer flask, move it over the
   mixing beaker, squeeze the bulb ("drip"), put it back;
2. grasp the **test tube**, pour into the mixing beaker (115° tilt), put it back in the rack;
3. grasp the **glass rod** from its stand on the drip tray, stir in the beaker, put it back;
4. grasp the **mixing beaker**, shake it;
5. pour it into the **waste beaker** and return everything (arm goes home).

Liquids are not simulated (per the brief).  During every grasp the pads'
depth images + normal forces feed a **Young's-modulus estimator**
(`YoungEstimator` in `lab_scene.py`): travel after contact is split into gel
indentation (depth image) and object deformation, and the object part is
fitted with the Hertz law — the rubber bulb reads a finite modulus
(ground truth `E_BULB` in `lab_assets.py`), the glassware reads "rigid".

Files
-----
| file | role |
|---|---|
| `lab_assets.py` | props: beaker (from `glass_beaker.glb`), dropper (profile from `straw.glb`, one tet mesh with per-tet Young), flask / test tube / rack / tray / rod (procedural). Cached in `Assets/lab_tactile/lab_assets.npz`. |
| `lab_pad.py` | fabricates the 15.9 × 14.0 × 2.5 mm fingertip pad (tetgen) under `Assets/lab_tactile/pad/`. |
| `fr3_kin.py` | URDF FK + damped-least-squares IK (joint limits, posture bias). |
| `lab_scene.py` | engine scene: arm, props, FEM dropper, pinned gel pads, exclusions; `FrankaTactileGripper` (POSITION / SAFE force servo on the two finger joints); `YoungEstimator`. |
| `lab_task.py` | the five stages as timelines of segments (lazy geometry from the live scene). |
| `run_lab_demo.py` | main loop (`--dry` = kinematics only, no GPU) → `lab_demo.npz`. |
| `render_lab_video.py` | npz → mp4 (pyrender/EGL + tactile dashboard). |
| `export_lab_usd.py` | npz → USD (`pxr` python; rigid = transform anim, FEM = points anim, telemetry attrs). |
| `smoke_test.py` | build + 40 physics steps + placement checks. |
| `run_all.sh` | physics → video → USD in one go. |

Run
---
```bash
cd Stiff-GIPC-tactile-0854
# kinematics-only check (no GPU)
PYTHONPATH=. python examples/lab_tactile/run_lab_demo.py --dry --dt 0.02 --rec-every 2 --out out/lab_tactile_dry
# physics (needs ~3 GB free GPU memory), then video + USD
examples/lab_tactile/run_all.sh
```
Solver python: `python` (engine from `build_311/`);
USD python: `python` (has `pxr`).

Conventions / gotchas
---------------------
* World is Y-up, table = ground plane y=0, arm base at the origin (URDF Z-up
  mapped with a −90° X rotation).  Engine joint targets are absolute URDF angles.
* Load order: ABD (arm, props) first, FEM (dropper, pads) last.  FEM load
  records carry a FEM-local `body_offset`; the setters that take a body index
  (`set_per_tet_young_for_body`, `set_soft_body_density`, `set_body_friction`,
  `add_collision_exclusion`) take the **load-record index** (global id).
* Pads are hard-pinned (`GelBody.add_pinned`) to the finger links because the
  fingers rotate (pouring); stitch springs do not co-rotate.  Pads are
  collision-excluded from the hand, both fingers and each other (they overlap
  at the closed load pose).
* Finger joint distance is measured from the load pose (closed): pad gap =
  2·d − 2·pad_thickness.
* The stir uses a vertical rod in a stand: a rod lying flat would need a
  horizontal-hand pose over the beaker that the FR3 cannot reach there.

Engine findings from the A800 bring-up (2026-10-02)
----------------------------------------------------
* **`get_abd_body_transforms()` returns the affine block transposed**
  (`sim_engine.cu`: `mat[:3,:3] = A^T`).  Composing `C @ inv(C0)` directly gives
  a body "pose" whose rotation is R(−θ): every single-joint move showed a hand
  orientation error of exactly 2θ and link positions off by up to 100 mm while
  `get_revolute_current_angles_abs()` matched the command to 1e-4.  Use
  `abd_world_tf()` (`lab_scene.py`) which transposes the block first; after that
  every link matches the URDF FK to 0.0 mm / 0.0°.  Translation-only carriers
  (the shipped `ParallelGripper.rigid_motion`) never exposed this.
* `load_mesh_from_data` writes meshes to the shared, hard-coded
  `/tmp/stiffgipc_mesh_data/` keyed by content hash.  Two engine processes
  started at the same moment on one machine race on those files
  (`[load_surfaceMesh_ABD] Failed to load ...`).  Stagger launches by ~30 s.
* The arm's per-step cost is dominated by Newton iterations while it moves
  (1 → 7–25 per step, ~20 ms each on an A800); joint drive ratio 100 vs 1000
  makes no difference; dt=0.01 exploded the collision-pair count (not used).
* Two Fixed bodies with coincident faces (rod stand on the tray floor) make the
  barrier infinite (NaN energy from frame 0) — exclude such pairs explicitly.
* **`newton_tol` matters for carried objects.**  With the engine default
  `1e-2` (exit when the Newton step < 1e-2·scene_diag·dt ≈ 19 mm/s here) a
  dropper carried by its soft bulb trailed the gripper and swung up to 40°
  during a 0.3 m/s² transport, then relaxed slowly — it looked like a
  pendulum but was under-converged weakly-coupled DOFs.  Measured peak tilt:
  `newton_tol=1e-2` 40°, `pcg_tol=1e-6` 5°, `newton_tol=1e-3` 3.3° (0.3° at
  the end of the move).  Cost while moving: 526 → 957 ms/frame (1e-3), but
  holds get cheaper (19–35 ms/frame).  MAS (`preconditioner_type=1`) was
  slower in this scene.  `LabScene` now defaults to `newton_tol=1e-3`.
* Pour paths must schedule the pivot height with the tilt: an upright tube
  whose mouth is 45 mm above the rim hangs *inside* the beaker and levers it
  over when tilted (run 1).  See `p_mouth(u)` / `p_rim(u)` in `lab_task.py`.
* Insertions measure the held object's actual tip and centre it over the
  receptacle first (`centre_over` / `descend_until`): objects lean a few
  degrees in their holders and in the soft pads, which is enough to miss a
  12 mm pocket by 8–25 mm with open-loop poses.

Result (2026-10-04)
-------------------
One continuous physics run of all five steps completed on the A800
(134.2 s of simulation, 26 840 frames at dt = 5 ms, 12.3 h wall, 0 IK
failures): dropper taken out, squeezed over the beaker and seated back on the
flask neck; test tube poured (side grasp, wrist roll to ~98° tilt) and
returned to its rack slot; rod stirred and returned to its stand; mixing
beaker shaken, poured into the waste beaker (~100°) and set back down.  Every
prop ends within a few mm of where it started.  Tactile modulus estimate on
the rubber bulb: ~550 kPa vs 300 kPa ground truth (Hertz-model overestimate);
glassware reads "rigid".  Deliverables: `out/review/final/lab_demo.{mp4,usdc}`.
Grasp design per the user's review: vertical top-down grasps for the dropper
and rod, SIDE grasps + wrist-roll pours for the tube and beaker (a top-down
grasp tilted sideways swept the hand's 200 mm width into the glassware).

Dropper lag study (2026-10-04, `probe_lag.py`, results in `out/lag/`)
--------------------------------------------------------------------
Question: why does the glass tube trail / jitter behind the gripper while the
dropper is carried?  Facts from the final recording: no pad slip (<0.1 mm),
tube does not bend (<0.6 mm bow), the whole dropper rotates about the pad
closing axis, and the tip trails the gripper velocity even while
decelerating (a pendulum would swing ahead) -> solver lag, not physics.
4-way A/B on the A800, stage-1 lift + 0.25 m transport, transport segment:

| run                        | tilt peak | jitter rms (40 ms) | Newton/frame | ms/frame |
|----------------------------|-----------|--------------------|--------------|----------|
| A newton_tol 1e-3, dt 5 ms, 6 N  | 5.9 deg | 1.43 deg | 41 | 904  |
| B newton_tol 1e-4, dt 5 ms, 6 N  | 2.3 deg | 0.19 deg | 41 | 870  |
| C newton_tol 1e-3, dt 2.5 ms, 6 N| 3.1 deg | 0.63 deg | 38 | 1032 (x2 frames) |
| D newton_tol 1e-3, dt 5 ms, 12 N | 5.4 deg | 1.30 deg | 45 | 1004 |

The 'jitter' is a sustained ~5 Hz rocking of the dropper in the pinch (its
natural mode), whose amplitude scales with the Newton tolerance (5-95 % band
A 0.3-5.0 deg, B 1.1-1.8 deg, C 0.7-2.7 deg, D 0.4-4.7 deg): the unconverged
per-frame residual keeps kicking a lightly damped mode.
Conclusion: the jitter is Newton-convergence precision.  tol 1e-4 removes it
at no extra cost; halving dt is weaker and 2x the cost; doubling the grip
force (grasp rotational stiffness) changes nothing.  The ~1.2 deg present in
every run at rest is the as-grasped orientation (the dropper rests slightly
tilted in the flask neck), not lag.  The final video was produced at 1e-3;
switch `LAB_NEWTON_TOL=1e-4` (or the engine's scene-size-independent
`newton_velocity_tol`) for the next full run.

Modelling notes for the record: the dropper is ONE solid tet mesh (tube and
bulb share nodes, materials assigned per tet; real bulb is a hollow shell,
real tube is hollow); Fixed ABD props drop their 12 affine DOFs as a whole
(static collision geometry); gel pads are pinned one-way (stick vertices
BoundaryType=Fixed, positions overwritten from the finger's ABD q after every
Newton step; the finger feels no gel reaction, the grip force loop is the
Python SAFE servo on the measured gel contact force).

Round 2 (same probe, 2026-10-04 evening): root cause confirmed and fixed.

| run                                   | tilt peak | 5-95 % band | jitter rms | Newton/frame | ms/frame |
|---------------------------------------|-----------|-------------|------------|--------------|----------|
| A  tol 1e-3                           | 5.9 deg | 0.3-5.0 deg | 1.43 deg | 41 | 904  |
| E3 tol 1e-3, finger targets frozen    | 6.1 deg | 0.3-5.2 deg | 1.34 deg | 42 | 904  |
| B  tol 1e-4                           | 2.3 deg | 1.1-1.8 deg | 0.19 deg | 41 | 870  |
| E4 tol 1e-4, finger targets frozen    | 2.1 deg | 1.0-1.8 deg | 0.23 deg | 38 | 799  |
| E1 tol 1e-4 + pcg_tol 1e-6            | 1.3 deg | 1.1-1.3 deg | 0.03 deg | 50 | 1682 |
| E2 tol 1e-5 (pcg 1e-4)                | 1.3 deg | 1.1-1.3 deg | 0.03 deg | 59 | 1355 (1 line-search energy-floor warning) |

* Freezing the finger targets (no SAFE servo) changes nothing -> the grip
  controller is not pumping the mode.
* A converged solve (E1 / E2) has NO oscillation at all (no spectral peak,
  0.03 deg rms), without any damping.  So the rocking is purely the
  unconverged per-frame residual (inexact PCG direction at 1e-4 relative +
  loose step exit) turning into spurious velocity.
* The 1.1-1.3 deg that remains in every run is the as-grasped orientation
  held by stiction (the dropper rested 1.2 deg tilted in the flask neck; the
  neck straightens it to 0.2 deg during extraction and the friction anchors
  then return it to the anchored orientation).  Grasp it vertical to remove.
* Fix applied: `LabScene` now defaults to newton_tol 1e-4 + pcg_tol 1e-6
  (E1).  E2 is cheaper but touches the double-precision energy floor in the
  line search; E1 is the principled choice.  Moving frames cost ~1.9x;
  holding frames (1 Newton) are unaffected.  The final video in
  out/review/final was rendered BEFORE this change (tol 1e-3).

Liquid (MPM Lite water, 2026-10-05, branch feat/mpm-liquid)
------------------------------------------------------------
`stiff_physics/liquid/` = vendored official MPM Lite engine (Warp, Apache-2.0)
+ `LiquidMPM` coupling layer (engine surface triangles -> grid-node boundary
conditions, one-way).  Needs `pip install warp-lang==1.10.1`.

    # water in the mixing beaker, stage 4 pour; particles land in the npz
    python examples/lab_tactile/run_lab_demo.py --liquid beaker_mix --stages 4 \
        --liquid-dx 0.005 --liquid-fill 0.5 --out out/liq4
    python examples/lab_tactile/render_lab_video.py out/liq4/lab_demo.npz   # particles drawn as points
    python examples/lab_tactile/export_lab_usd.py   out/liq4/lab_demo.npz   # /World/Liquid/water (Points)
    python examples/lab_tactile/test_liquid_cup.py cuda:0 0.005             # standalone containment test

Rules learnt: walls thinner than dx leak with a nearest-face-normal BC; the
layer therefore makes nodes inside a solid sticky and projects particles out of
solids every sub-step (test: 16 % leak -> 0 at dx 10 mm / 4 mm wall).  Cost on
the A800: ~0.66 s per 5 ms solid step for 15 k particles, 50 explicit
sub-steps (1e-4 s, kappa 1e5).  Details: MPM_LIQUID_PLAN.md
(main repo, local-only).

Liquid settings that survived the 2026-10-05/06 diagnostics (6 scene runs)
--------------------------------------------------------------------------
* APIC transfers (flip 0) + particle push-apart floor 0.9 h (wp.HashGrid): APIC
  alone jitters ~0.04 m/s at rest and random-walks the particles into denser
  packing while per-particle J stays 0.998 (the EOS cannot see sub-cell
  compaction) -> clumps -> spray at the pour; pure PIC is quiet but behaves
  like honey (onset 69 deg).  With the floor: clean jet, 0 % clumping.
* Boundary mesh = all props + dropper, thickened OUTWARD only by 2 dx (cavities
  and rim tops exact; upper 3 dx of each body not thickened or the pour runs
  down the outside); grid nodes inside a solid: two-sided slip; exterior band
  1 dx: separable; normals = closest-point direction; projection net on.
* Water: kappa 1e5, no tension, J >= 0.1 clamp, NaN guard, dt 1e-4 explicit.
* Fill 0.9 of the cavity (spill angle ~54 deg), seeding margin 0.6 dx.
* Diagnostics: test_liquid_static_tilt.py (BC/clumping isolation),
  test_liquid_beaker_tilt.py (real mesh; snapshots + jet plots),
  probe_liquid_scene.py (engine scene at rest: J, NN spacing, surface height).
Known limits: thin glass (3.5 mm) vs dx 5 mm needs the thickening + projection;
~2 %/s slow settling of the column at rest; liquid cost ~0.7 s per 5 ms step.

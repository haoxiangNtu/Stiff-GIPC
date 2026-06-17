# Force / Torque / Velocity Control for ABD Joints — Design

Status: prototype on branch `proto/force-control` (not on any release line).
Goal: give StiffGIPC's affine-body (ABD) joints the same external force/torque
and articulation-velocity control that the public IPC library (libuipc) exposes,
matching its formulation as closely as possible, and ship a GUI demo for every
force-control test in the public suite.

---

## 1. Formulation — the `q_tilde` (incremental-potential) path

StiffGIPC, like libuipc, advances each ABD body by minimising the incremental
potential

```
Phi(q) = 1/2 (q - q_tilde)^T M (q - q_tilde) + dt^2 * sum E_pot(q)
```

where `q` is the 12-DOF ABD state `[p (3); vec(A) row-major (9)]` and `q_tilde`
is the predicted (inertial) state. External forces enter through `q_tilde`, NOT
through a gradient/Hessian term, exactly as libuipc does:

```
q_tilde = q_prev + dt * v + dt^2 * (a_gravity + a_ext),   a_ext = M^-1 * F
```

`F` is the per-body 12-DOF generalized force. Because `F` is constant within a
Newton solve, the kinetic term `1/2 (q-q_tilde)^T M (q-q_tilde)` already provides
the correct response — no extra Hessian is needed. This is implemented in
`cal_q_tilde.cu`.

Two accumulators feed `a_ext` (both `DeviceBuffer<Vector12>`, in `abd_sim_data.h`):

- `body_id_to_abd_ext_force`   — user per-body wrench (set from Python);
- `body_id_to_abd_joint_wrench` — per-step joint force/torque, recomputed each
  step from the active driving joints.

### 1.1 Revolute external torque (libuipc `affine_body_revolute_joint.cu`)

A scalar torque `tau` about the joint axis becomes, for the parent (−) and child
(+) bodies, an affine wrench:

```
e_i = normalize(A_p * (p_bar x pN_bar))      # per-body world axis
e_j = normalize(A_c * (q_bar x qN_bar))
e_j = 1/2 (e_i + e_j);  e_i = -e_j           # symmetrize (libuipc, byte-exact)
F_k.segment<9>(3) = vec( tau/2 * skew(e_k) * A_k^-T )   # child +, parent -
```

`A^-T` is computed with a device-safe cofactor inverse (`inv_transpose_3x3`),
matching libuipc's full affine `A^-T` (not just the rigid limit). See the
`cal_joint_torque_wrench` kernel in `cal_q_tilde.cu`.

### 1.2 Prismatic external force (libuipc `affine_body_prismatic_joint.cu`, UID 667)

A scalar force `f` along the joint axis is a LINEAR force on the translation DOF
(no skew, no `A^-T` — it is translational, not rotational):

```
t_i = A_p * t_bar_parent;  t_j = A_c * t_bar_child   # per-body world tangent
t_j = 1/2 (t_i + t_j);  t_i = -t_j                   # symmetrize (libuipc)
F_i.segment<3>(0) = f * t_i;  F_j.segment<3>(0) = f * t_j
```

To carry a per-body parent tangent, `PrismaticDrivingGPUData` gained `tp_bar`
(parent-material axis) alongside the existing `tq_bar`. See the
`cal_joint_prismatic_force_wrench` kernel in `cal_q_tilde.cu`. Verified: a
prismatic axis force equals a direct body force of the same magnitude along that
axis to machine precision (ratio 1.0000).

### 1.3 Articulation / velocity control (libuipc `external_articulation_constraint`)

libuipc's articulation constraint writes `delta_theta_tilde = velocity * dt` each
step — a desired per-step increment, i.e. joint VELOCITY control, enforced by the
same position PD-penalty StiffGIPC already uses for driving. It is therefore
reproduced purely in Python with the existing target API (no new C++):

```
revolute:  set_revolute_target(j, current_angle + omega * dt)   # closed-loop
prismatic: target += v * dt;  set_prismatic_target(j, target)
```

Tracking is exact: 60 frames at pi/6 rad/s gives 0.314 rad vs pi/6*0.6 = 0.314.

---

## 2. New / changed engine API

| API | Purpose |
|---|---|
| `set_body_external_force(body_id, fx, fy, fz)` | per-body LINEAR external force (N) |
| `set_body_external_wrench(body_id, wrench12)`  | per-body FULL 12-DOF wrench (linear + affine; e.g. `w[5]=+omega, w[9]=-omega` is a spin torque about Y) |
| `set_revolute_torque(idx, torque)`             | external torque (N·m) on a revolute driving joint |
| `set_prismatic_force(idx, force)`              | external force (N) along a prismatic driving joint axis |
| `set_max_prismatic_step_per_frame(m)`          | raise the prismatic position-driving rate clamp (default 0.002 m/frame) |

All persistent until changed; pass 0 to clear. For PURE force/torque control on a
joint, also call `set_revolute_strength(idx, 0)` / `set_prismatic_strength(idx, 0)`
so the PD position term is off. Position control and force control are independent
and can coexist on different joints in one sim (validated dual-arm: left position,
right torque).

Files touched: `abd_system/abd_sim_data.h`, `abd_system/abd_driving_joint.h`,
`abd_system/abd_system.cu`, `abd_system/abd_system_function/cal_q_tilde.cu`,
`abd_system/abd_system_function/setup_abd_system_gradient_and_hessian.cu`,
`joint_angle_control.h`, `sim_engine.{h,cu}`, `bindings/pystiffgipc.cu`.

---

## 3. GUI demos — one per public force-control test

All in `examples/`, run with
`STIFF_SKIP_CCD_SANITY=1 python examples/<name>.py`. The common scene mirrors the
public tests: two/three cubes scaled 0.4, placed at +-0.6 (or along Z for the
multi-joint), young 1e8, density 1e3, ground pushed far below (`ground_offset=-100`,
the public tests have no ground), with the public magnitudes/phase timing.

| demo | public test | what it shows |
|---|---|---|
| `case_force_body_force_ui.py`            | 44 | free cube, full 12-DOF wrench: orbiting linear force + Y spin |
| `case_force_revolute_torque_ui.py`       | 74 | pure torque ±1000 @50 → pendulum swings down then to the other side |
| `case_force_revolute_drive_then_torque_ui.py` | 80 | phase1 velocity-ramp drive (∓5→±5 rad/s), phase2 torque (∓1000 @150) |
| `case_force_prismatic_force_ui.py`       | 73 | pure axis force +1000/−5000 @50 → slides along Z rail then back |
| `case_force_prismatic_drive_then_force_ui.py` | 79 | phase1 velocity-ramp drive, phase2 force (∓1000 @150) |
| `case_force_joint_velocity_ui.py`        | 46 | revolute velocity control ω=pi/6, with optional auto-reverse |
| `case_force_multijoint_velocity_ui.py`   | 47 | 1:1 layout: 3 cubes along Z, revolute(X)+prismatic(Z), ω=pi/6 & v=0.1, `skip_all_collision=True` (= public `contact.enable=false`) |

Extra prototypes kept: `case_force_cube_ui.py`, `case_force_dualarm_ui.py`
(mixed position+torque dual-arm). Headless regression: `test_force_control.py`,
`test_mixed_pos_torque.py`, `test_dualarm_mixed.py`.

### Deviations from the public tests
- `case_force_body_force_ui.py` keeps `scale 0.4` + larger default magnitudes for
  visibility; the public test uses a full-size cube with |F|=10 and spin=0.01,
  which is nearly imperceptible. Sliders expose the full range. (Intentional, per
  request.)
- All other demos match the public setup parameter-for-parameter (gravity, scale,
  ground, young, density, joint axes/centers, magnitudes, phase frames, contact).

---

## 4. Verification

- revolute torque: torque→angle scales linearly with magnitude when collision-free
  (300/3000/30000 → 0.010/0.0998/0.993 rad), confirming the wrench magnitude is
  applied correctly; pendulum swing reproduced visually (0° → −127° → +127°).
- prismatic force: equals a same-magnitude body force to machine precision (1.0000).
- velocity control: tracks the commanded rate to <0.1% (pi/6 → measured pi/6).
- multi-joint (test 47, strict 1:1): rev 1.258 vs pi/6*2.4 = 1.257, pris 0.240 vs
  0.1*2.4 = 0.240, with collisions globally disabled.
- All demos screenshot-verified under polyscope (y-up; gravity −Y, ground normal +Y).

---

## 5. Notes for a future audit before any release
- This is the `q_tilde` external-force path; it adds no Newton Hessian term, so it
  does not affect solver SPD-ness or convergence of existing scenes.
- New buffers are zero by default; with no force/torque set, behaviour is identical
  to before (the joint-wrench kernels early-out when `ext_torque/ext_force == 0`).
- `set_max_prismatic_step_per_frame` only relaxes a kinematic safety clamp; leave
  it at the 0.002 default for scenes with FEM softpads pinned to ABD bodies.

---

## 6. Soft-gripper grip control study (finray STRATEGY_F)

Built on the §1–5 force path, this section adds the engine pieces + examples for
controlling a *compliant* finray gripper (rigid mount seat → fixed-joint → rigid
finray root → gap-0 stitch → soft FEM truss → object).

### 6.1 New engine APIs (this work)
- `set_prismatic_limit_barrier(idx, cl, dir, dhat, kappa)` — a one-sided IPC
  log-barrier on the prismatic coordinate `d` at the closed end `cl`. The solver
  then **guarantees `d` never crosses `cl`** regardless of the (force) drive — a
  hard no-overshoot guarantee while the joint stays pure force-controlled. Added
  to `prismatic_driving_{energy,gradient_hessian}` (reuses the existing `dd/dq`
  chain; Gauss-Newton, SPD-safe). Default `kappa=0` → OFF → zero effect on existing
  scenes (verified: `host_data(n)` value-initialises the new fields).
- `get_stitch_max_stretch(start, count)` and
  `get_stitch_max_stretch_batched(starts, counts)` — on-GPU max stitch-spring
  stretch over a spring range / many ranges. The batched form is ONE kernel,
  block-per-segment (= one finger, or one finger of one env), returns per-segment
  scalars with NO full-vertex D2H. Block independence ⇒ multi-env isolated.

### 6.2 The grip signal (which "force" to regulate on)
On the compliant finray the rigid-prismatic drive force/lag does NOT register the
grasp (the rigid root keeps moving as the soft truss deforms). Measured signals,
most-direct → cheapest:
- soft-fingertip IPC contact force (`get_body_contact_force`): most direct, but
  EXPENSIVE (rebuilds BVH+pairs each call, ~200 ms/frame).
- stitch stretch at the truss base (max per-pair `‖fem−rigid‖`): one elastic layer
  removed but clean+monotone, CHEAP (~70 ms/frame). The chosen default.
- prismatic drive force/lag: two layers removed, noisy/non-monotone — unusable.

### 6.3 Control laws (examples expose all of them)
- `stitchgrip` / `trackgrip` — **feedback-driven POSITION** (default). Position
  drive (anchored ⇒ stable, no slide under arm motion, provably no overshoot) with
  a target that MARCHES toward closed but PAUSES when the grip signal (stitch
  stretch / contact force) hits a threshold ⇒ adaptive half-close, and RESUMES if
  the object is removed (signal drops). Gets adaptive + no-slide + resume together.
- `forcebarrier` — **literal pure force** (`strength=0`) + the §6.1 cl-barrier. No
  overshoot (barrier), adaptive, resumes. BUT pure force has no positional anchor,
  so under arm motion the soft seat↔root link stretches → the finray visibly
  *slides* off the mount. Kept as a teaching example of the trade-off.
- `forcelock` / `truegrip` — pure-force grasp then position-LOCK (no drift, no
  resume). `position` — plain direct-set baseline.

### 6.4 The grip-control trilemma (key finding)
For a compliant multi-body gripper, simple controllers get only TWO of
{adaptive half-close, no-slide-under-motion, resume-on-vanish}:
- pure force (`strength=0`): adaptive ✓, resume ✓, **slide ✗** (no anchor).
- position locked at grip width: adaptive ✓, no-slide ✓, **resume ✗**.
- position to `cl`: no-slide ✓, resume ✓, **crushes ✗** (no half-close).
All three at once requires a **feedback-driven** target (§6.3 `stitchgrip`): it is
position-anchored (no slide) yet its target is advanced by grip feedback (adaptive
+ resume). The slide is intrinsic to `strength=0`; libuipc composes joint
constraint + driving + external force + limit as separate constitutions, but its
grippers also avoid the extra soft seat↔root link that stretches here.

### 6.5 Verification
- cl-barrier: pure force F=20 AND F=40 both held at ~dhat from `cl`, never crossing
  (vs +30 m runaway without the barrier); standoff ≈ `dhat`; default off = no
  regression on stitch/position modes.
- GPU stitch reduction numerically identical to the CPU path (half-close at
  L_open=0.0146 m either way). Single-env perf: CPU 50.9 ms vs batched-GPU 50.4 ms
  (within run-to-run noise — the physics step dominates; the GPU path is for the
  multi-env case where the per-finger full-vertex D2H would serialise ×N).
- Required pure-force magnitude scales linearly with finray Young's modulus
  (F≈60 @1e7, F≈10 @1e6) and is ~0 in free space (F=2 fully closes an unobstructed
  gripper at any Young) — the large force is elastic deformation against the
  object/scene, not a units bug.

### 6.6 Examples (2 files × 2 modes)
- `examples/replay_case39_UMI_obb_cup_shirt_forcegrip.py` — OBB arm + finray +
  cup + shirt, replays `qpos_case39.h5`. `CASE39_GRIP_MODE=stitchgrip`
  (+`CASE39_STITCH_GPU=1`) | `forcebarrier` | trackgrip/forcelock/truegrip/position.
- `examples/case_umi_finray_force_ui.py` — same scene, arm = imgui sliders, gripper
  = per-arm CLOSE/OPEN buttons. `CASE_UMI_GATE=stitch` (+`CASE_UMI_STITCH_GPU=1`) |
  `CASE_UMI_MODE=forcebarrier`.

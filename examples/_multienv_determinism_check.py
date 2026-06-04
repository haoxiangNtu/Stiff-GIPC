#!/usr/bin/env python3
"""Multi-env determinism / bit-equality harness (P2/P3 golden test).

Two uses:
  1. DETERMINISM GAP: how much do supposedly-identical tiled envs diverge under
     the current global merged solve? (Baseline that P3 block-diagonal solve must
     drive to ~0.) Compares each env's cloth/cup vertices RELATIVE to its env
     offset; reports max inter-env divergence over time.
  2. GOLDEN TEST for the P2b DOF re-layout: a physics-changing refactor must NOT
     change results vs the pre-refactor build. Run this on both builds with the
     SAME config and diff the per-env trajectories (should match to ~1e-10).

Usage:
    PYTHONPATH=. python examples/_multienv_determinism_check.py [case39|foldshirt] [N] [frames]
"""
import sys, os, json, math, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.spatial.transform import Rotation
from stiff_physics import Engine, Config
from stiff_physics.robot import Robot

scene = sys.argv[1] if len(sys.argv) > 1 else "case39"
N     = int(sys.argv[2]) if len(sys.argv) > 2 else 2
FRAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 300
SP = float(os.environ.get("CASE39ME_SPACING", "4.0"))
ISO = int(os.environ.get("CASE39ME_ISOLATE", "1"))

if scene == "case39":
    import examples.replay_case39_multienv as M
    _A = M._ASSETS_DIR
    import h5py
    with h5py.File(_A + "trajectories/qpos_case39.h5", "r") as f:
        actions = f["qpos"][:]
    arm_tf0 = M.make_arm_tf(M.ARM_SCALE)
    cloth_obj = _A + "triMesh/shirt_6436v.obj"
    cup_scale, cup_xyz = 0.8, np.array([0.67, -0.2, -0.4])
    cup_T0 = np.eye(4); cup_T0[:3, :3] *= cup_scale; cup_T0[:3, 3] = cup_xyz
    shirt_T0 = np.eye(4); shirt_T0[:3, 3] = [0.67, 0, 0]
    abs_dhat = 0.00239
    def apply(robot, ej, raw): M.apply_frame(robot, ej, raw, 0.5)
else:
    import examples.replay_foldshirt_multienv as M
    _A = M._ASSETS_DIR
    import h5py
    ep = "/tmp/replay_0528/episode_00000.hdf5"
    with h5py.File(ep, "r") as f:
        actions = f["actions"][:]; rip = np.asarray(f.attrs["robot_init_pose"])
        oi = json.loads(f.attrs["object_init_info"])
    ck = next(k for k, v in oi.items() if v.get("body_type") != "ABD")
    cloth_T0 = np.asarray(oi[ck]["initial_pose"]).reshape(4, 4)
    cloth_obj = _A + "objects/m-panda_single/scaled.obj"
    arm_tf0 = M.make_arm_tf(rip[:3], 1.0)
    abs_dhat = 0.0019
    def apply(robot, ej, raw): M.apply_frame(robot, ej, raw, 0.0)

cfg = Config(dt=0.02, cloth_thickness=1e-3, cloth_young_modulus=1e4, bend_young_modulus=1e3,
             cloth_density=200, strain_rate=100, soft_motion_rate=1e4, poisson_rate=0.49,
             friction_rate=0.4, relative_dhat=1e-3, joint_strength_ratio=100.,
             revolute_driving_strength_ratio=100., prismatic_strength_ratio=2000.,
             semi_implicit_enabled=False, semi_implicit_beta_tol=5e-2, semi_implicit_min_iter=1,
             newton_tol=5e-2, newton_iter_cap=50, preconditioner_type=1,
             ground_offset=(-1.67 if scene == "case39" else 0.75), assets_dir=_A)
cfg._cfg.collision_detection_buff_scale = 8.0
cfg._cfg.linear_system_buff_scale = 2.5
cfg._cfg.triplet_internal_margin = 6.0
cfg._cfg.absolute_dhat = abs_dhat
eng = Engine(cfg); eng.set_log_level(0)

_co = np.eye(4)
_co[:3, :3] = Rotation.from_euler('xyz', [-1.57079632679, 0.20245819348, -1.57079632679]).as_matrix()
_co[:3, 3] = [-0.0165, 0.0165, 0.12773331296]
rr = np.load(M.RIGID_REMAP, allow_pickle=True); hd = np.load(M.UNIFIED_NPZ)
hybrid = dict(collision_origin=_co, rigid_v_idx=rr['rigid_v_idx'], n_rigid_v=len(rr['rigid_v_idx']),
              verts=np.ascontiguousarray(hd['vertices'], np.float64),
              tets=np.ascontiguousarray(hd['tets'], np.int32))
offs = M.make_env_offsets(N, SP)

if scene == "case39":
    ge = dict(fem_young=1e7, shirt_young=1e2, abd_cursor=0)
    envs = [M.build_env_abd(eng, o, hybrid, cup_T0, ge) for o in offs]
    for env, o in zip(envs, offs): M.build_env_fem(eng, env, o, hybrid, shirt_T0, ge)
    cl_rng = [(env['shirt_rec'].vertex_offset, env['shirt_rec'].vertex_count) for env in envs]
else:
    ge = dict(fem_young=1e7, cloth_young=1e2, abd_cursor=0)
    envs = [M.build_env_abd(eng, o, hybrid, arm_tf0, ge) for o in offs]
    for env, o in zip(envs, offs): M.build_env_fem(eng, env, o, hybrid, cloth_obj, cloth_T0, ge)
    cl_rng = [(env['cloth_rec'].vertex_offset, env['cloth_rec'].vertex_count) for env in envs]

n_abd = sum(1 for r in eng.get_load_records() if r.body_type == 0)
for env in envs: M.exclusions_for_env(eng, env, n_abd)
if ISO:
    n_fem = sum(1 for r in eng.get_load_records() if r.body_type == 1)
    ma, mf = n_abd // N, n_fem // N
    eng.native.set_body_groups([c // ma for c in range(n_abd)] + [f // mf for f in range(n_fem)])
eng.finalize()
for env in envs:
    for a in env['arm_ids']: eng.native.set_body_apply_gravity(a, False)
    for g in env['grippers']:
        eng.native.set_body_apply_gravity(g['abd_id'], False)
        eng.native.set_fixed_joint_strength(g['fj_idx'], 1e3)
eng.native.set_max_revolute_step_per_frame(0.04)
robot = Robot(eng)
for i in range(len(robot.prismatic_joints)): eng.native.set_prismatic_strength(i, 15.0)
ejs = M.slice_env_joints(robot, N)
off_xz = [(o[0, 3], o[2, 3]) for o in offs]

print(f"# scene={scene} N={N} spacing={SP} isolate={ISO}  cloth verts/env={cl_rng[0][1]}")
print(f"# frame   max|env_i - env_0| (cloth verts, relative to env offset)")
for fr in range(min(FRAMES, len(actions))):
    raw = actions[fr]
    for ej in ejs: apply(robot, ej, raw)
    eng.step()
    if fr % 20 == 0 or fr == FRAMES - 1:
        v = eng.get_vertices()
        base = None; maxd = 0.0
        for e, (o, c) in enumerate(cl_rng):
            pts = v[o:o + c].copy()
            pts[:, 0] -= off_xz[e][0]; pts[:, 2] -= off_xz[e][1]  # relative to env offset
            if base is None: base = pts
            else: maxd = max(maxd, float(np.abs(pts - base).max()))
        print(f"{fr:5d}   {maxd:.3e}")
import os as _o; sys.stdout.flush(); _o._exit(0)

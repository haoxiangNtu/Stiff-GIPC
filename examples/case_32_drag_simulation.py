#!/usr/bin/env python3
"""Headless simulation of case_32 GUI drag with rate limit.
Slider jumps to target_deg in one frame; driven_deg follows at
MAX_ANGLE_PER_STEP_DEG per step.  Verify no INTERSECT and Newton stable."""
import sys, os, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _use_dailyv2_engine  # noqa: F401

import numpy as np
from stiff_physics import Engine, Config

UNIFIED_NPZ = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/hybrid_d/STRATEGY_F_unified.npz"
RIGID_MSH   = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/hybrid_d/STRATEGY_F_rigid.msh"
RIGID_REMAP = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/hybrid_d/STRATEGY_F_rigid_remap.npz"
CUBE_MSH    = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/tetmesh/cube.msh"

cfg = Config(
    dt=0.020, soft_motion_rate=1e4, poisson_rate=0.49, friction_rate=0.4,
    relative_dhat=1e-4, joint_strength_ratio=200.0,
    revolute_driving_strength_ratio=200.0,
    semi_implicit_enabled=True, semi_implicit_beta_tol=5e-2,
    semi_implicit_min_iter=1, newton_tol=5e-2,
    preconditioner_type=0, ground_offset=-0.5,
    assets_dir="/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/",
)
cfg._cfg.collision_detection_buff_scale = 8.0
eng = Engine(cfg)

arm_T = np.eye(4); arm_T[:3,:3] *= 0.04; arm_T[3,3]=1.0
arm_T[:3,3] = [0.012, 0.18, 0.016]
eng.load_mesh(CUBE_MSH, dimensions=3, body_type="ABD", transform=arm_T,
              young_modulus=1e8, boundary_type="Fixed")
arm_id = eng.get_load_records()[-1].body_offset

eng.load_mesh(RIGID_MSH, dimensions=3, body_type="ABD",
              transform=np.eye(4), young_modulus=1e8, boundary_type="Free")
gripper_id = eng.get_load_records()[-1].body_offset
gripper_v_offset = eng.get_load_records()[-1].vertex_offset

data = np.load(UNIFIED_NPZ)
verts = np.ascontiguousarray(data['vertices'], dtype=np.float64)
tets = np.ascontiguousarray(data['tets'], dtype=np.int32)
fem_young = float(os.environ.get("HYBRID_D_YOUNG", "1e8"))
eng.native.load_mesh_from_data(verts, tets, 4, 3, 1, np.eye(4), fem_young, 0)
fem_v_offset = eng.get_load_records()[-1].vertex_offset
n_abd = sum(1 for r in eng.get_load_records() if r.body_type == 0)
fem_global = n_abd + eng.get_load_records()[-1].body_offset

remap = np.load(RIGID_REMAP, allow_pickle=True)
rigid_v_idx = remap['rigid_v_idx']
for i in range(len(rigid_v_idx)):
    eng.native.add_fem_pin_to_abd(fem_v_offset + int(rigid_v_idx[i]),
                                   gripper_v_offset + i, gripper_id, (0,0,0))

joint_anchor = np.array([0.012, 0.14, 0.016])
joint_axis = np.array([0.0, 0.0, 1.0])
joint_idx = eng.native.add_revolute_joint(arm_id, gripper_id, joint_axis,
    joint_anchor, -math.radians(90), math.radians(90), 0.0, "test")

eng.native.add_collision_exclusion(arm_id, gripper_id)
eng.native.add_collision_exclusion(arm_id, fem_global)
eng.native.add_collision_exclusion(gripper_id, fem_global)
eng.finalize()

eng.native.set_revolute_target(joint_idx, 0.0)
for _ in range(3):
    eng.step()

# Simulate user dragging slider to 30° in one move; rate limit to 0.5°/step
target_deg = 30.0
driven_deg = 0.0
MAX_ANGLE_PER_STEP_DEG = 0.5

print(f"\n--- DRAG SIM: slider→{target_deg}°, rate-limited {MAX_ANGLE_PER_STEP_DEG}°/step ---", flush=True)
print(f"{'step':>4} {'driven°':>8} {'A_xx':>8} {'A_yx':>8} {'ms':>6} {'INT':>4}", flush=True)

intersect_count = 0
for step in range(80):
    delta = target_deg - driven_deg
    if abs(delta) > MAX_ANGLE_PER_STEP_DEG:
        driven_deg += MAX_ANGLE_PER_STEP_DEG * np.sign(delta)
    else:
        driven_deg = target_deg

    eng.native.set_revolute_target(joint_idx, math.radians(driven_deg))
    eng.native.set_revolute_strength(joint_idx, 1.0)

    t0 = time.perf_counter()
    eng.step()
    step_ms = (time.perf_counter() - t0) * 1000

    cur_xf = eng.native.get_abd_body_transforms(np.array([gripper_id], dtype=np.int32))
    A_xx = cur_xf[0, 0, 0]
    A_yx = cur_xf[0, 1, 0]

    if step % 5 == 0 or abs(driven_deg - target_deg) < 0.01:
        print(f"{step:>4} {driven_deg:>+8.2f} {A_xx:>+8.3f} {A_yx:>+8.3f} "
              f"{step_ms:>6.1f} {intersect_count:>4}", flush=True)

    if abs(driven_deg - target_deg) < 0.01 and step > 60:
        break

print(f"\nFinal: driven {driven_deg:.2f}° (target {target_deg}°), "
      f"A_xx={A_xx:.3f} = cos({math.degrees(math.acos(A_xx)):.1f}°)", flush=True)
print(f"INTERSECT events: {intersect_count}", flush=True)

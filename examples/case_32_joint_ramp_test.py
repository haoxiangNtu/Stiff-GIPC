#!/usr/bin/env python3
"""Headless joint angle ramp test for case_32: ramp revolute target,
verify (a) joint actually rotates gripper, (b) FEM softpad follows
rotation via chain-rule."""
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
    dt=0.020, soft_motion_rate=1e4, poisson_rate=0.49,
    friction_rate=0.4, relative_dhat=1e-4,
    joint_strength_ratio=200.0, revolute_driving_strength_ratio=500.0,
    semi_implicit_enabled=False, newton_tol=5e-2,
    preconditioner_type=0, ground_offset=-0.5,
    assets_dir="/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/",
)
cfg._cfg.collision_detection_buff_scale = 8.0

eng = Engine(cfg)

# arm_link Fixed
arm_T = np.eye(4); arm_T[:3,:3] *= 0.04; arm_T[3,3] = 1.0
arm_T[:3, 3] = [0.012, 0.18, 0.016]
eng.load_mesh(CUBE_MSH, dimensions=3, body_type="ABD",
              transform=arm_T, young_modulus=1e8, boundary_type="Fixed")
arm_id = eng.get_load_records()[-1].body_offset

# gripper Free
eng.load_mesh(RIGID_MSH, dimensions=3, body_type="ABD",
              transform=np.eye(4), young_modulus=1e8, boundary_type="Free")
gripper_id = eng.get_load_records()[-1].body_offset
gripper_v_offset = eng.get_load_records()[-1].vertex_offset

# FEM unified
data = np.load(UNIFIED_NPZ)
verts = np.ascontiguousarray(data['vertices'], dtype=np.float64)
tets = np.ascontiguousarray(data['tets'], dtype=np.int32)
vertex_region = data['vertex_region']
fem_young = float(os.environ.get("HYBRID_D_YOUNG", "1e8"))
eng.native.load_mesh_from_data(verts, tets, 4, 3, 1, np.eye(4), fem_young, 0)
fem_v_offset = eng.get_load_records()[-1].vertex_offset
n_abd = sum(1 for r in eng.get_load_records() if r.body_type == 0)
fem_global = n_abd + eng.get_load_records()[-1].body_offset

# chain-rule pins
remap = np.load(RIGID_REMAP, allow_pickle=True)
rigid_v_idx = remap['rigid_v_idx']
n_rigid = len(rigid_v_idx)
for i in range(n_rigid):
    eng.native.add_fem_pin_to_abd(fem_v_offset + int(rigid_v_idx[i]),
                                   gripper_v_offset + i, gripper_id, (0,0,0))

# joint
joint_anchor = np.array([0.012, 0.14, 0.016])
joint_axis   = np.array([0.0, 0.0, 1.0])
joint_idx = eng.native.add_revolute_joint(
    arm_id, gripper_id, joint_axis, joint_anchor,
    -math.radians(90), math.radians(90), 0.0, "test")

eng.native.add_collision_exclusion(arm_id, gripper_id)
eng.native.add_collision_exclusion(arm_id, fem_global)
eng.native.add_collision_exclusion(gripper_id, fem_global)

eng.finalize()
print(f"finalized; pins={n_rigid}, joint={joint_idx}", flush=True)

# Settle 3 steps with target=0
eng.native.set_revolute_target(joint_idx, 0.0)
eng.native.set_revolute_strength(joint_idx, 1.0)
for _ in range(3):
    eng.step()

verts_at_rest = eng.get_vertices().copy()
abd_xform0 = eng.native.get_abd_body_transforms(np.array([gripper_id], dtype=np.int32))
gripper_initial = np.array(abd_xform0[0], dtype=np.float64).copy()

rigid_global = fem_v_offset + rigid_v_idx
fem_only_local = np.nonzero(vertex_region == 0)[0]
fem_only_global = fem_v_offset + fem_only_local

# Reference: anchor & rest gripper centroid
gripper_centroid = verts_at_rest[gripper_v_offset:gripper_v_offset + 150].mean(0)
print(f"\ngripper centroid={gripper_centroid}, anchor={joint_anchor}", flush=True)

print("\n--- JOINT RAMP TEST: revolute target 0° → 30° over 10 steps ---", flush=True)
print(f"{'step':>4} {'tgt°':>6} {'A_xx':>8} {'A_yx':>8} {'gx':>9} "
      f"{'gy':>9} {'rigid_FEM_dxy':>14} {'FEM near_dxy':>13} {'ms':>6}",
      flush=True)

for step in range(10):
    target_deg = 3.0 * (step + 1)  # 3° per step → 30° max
    eng.native.set_revolute_target(joint_idx, math.radians(target_deg))
    t0 = time.perf_counter()
    eng.step()
    step_ms = (time.perf_counter() - t0) * 1000

    cur = eng.get_vertices()
    cur_xf = eng.native.get_abd_body_transforms(np.array([gripper_id], dtype=np.int32))
    A_xx = cur_xf[0, 0, 0]
    A_yx = cur_xf[0, 1, 0]
    gx = cur_xf[0, 0, 3] - gripper_initial[0, 3]
    gy = cur_xf[0, 1, 3] - gripper_initial[1, 3]

    rigid_fem_disp = (cur[rigid_global] - verts_at_rest[rigid_global])[:, :2]
    rigid_fem_dxy = np.linalg.norm(rigid_fem_disp.mean(0))

    # Pure FEM disp magnitude (XY plane motion = rotation effect)
    fem_disp = (cur[fem_only_global] - verts_at_rest[fem_only_global])[:, :2]
    fem_near_dxy = np.linalg.norm(fem_disp.mean(0))

    print(f"{step:>4} {target_deg:>+6.1f} {A_xx:>+8.3f} {A_yx:>+8.3f} "
          f"{gx*1000:>+9.2f} {gy*1000:>+9.2f} "
          f"{rigid_fem_dxy*1000:>+14.2f} {fem_near_dxy*1000:>+13.2f} {step_ms:>6.1f}",
          flush=True)

print("\nUnits: A_xx/A_yx = gripper rotation matrix col 0 (cos/sin of angle)", flush=True)
print("       gx/gy = gripper centroid translation (mm) — non-zero because", flush=True)
print("              joint anchor is offset from gripper center → swing arm", flush=True)
print("       rigid_FEM_dxy = mean displacement of pinned FEM verts (chain-rule)", flush=True)
print("       FEM near_dxy = mean displacement of pure-FEM verts (elastic follow)", flush=True)
print("\nIf joint works: A_xx → cos(θ), A_yx → sin(θ) as θ ramps 0→30°", flush=True)
print("                rigid_FEM_dxy ≈ |gripper rotation effect on chain-pinned verts|", flush=True)
print("                FEM near_dxy follows rigid_FEM_dxy via elasticity", flush=True)

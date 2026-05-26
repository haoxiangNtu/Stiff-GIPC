#!/usr/bin/env python3
"""Diagnostic: check ABD vs FEM rigid-region vertex alignment in case_35.
Prints initial position diff + after-step position diff."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _use_dailyv2_engine
import numpy as np
from scipy.spatial.transform import Rotation
from stiff_physics import Engine, Config

URDF_PATH    = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/urdf/ridgeback_dual_panda_soft/ridgeback_softgripper_no_finger.urdf"
RIGID_MSH    = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/hybrid_d/STRATEGY_F_rigid.msh"
RIGID_REMAP  = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/hybrid_d/STRATEGY_F_rigid_remap.npz"
UNIFIED_NPZ  = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/hybrid_d/STRATEGY_F_unified.npz"
ARM_SCALE = 0.3

cfg = Config(
    dt=0.020, joint_strength_ratio=1000.0, revolute_driving_strength_ratio=300.0,
    semi_implicit_enabled=True, semi_implicit_beta_tol=5e-2, newton_tol=5e-2,
    preconditioner_type=0, ground_offset=-0.5,
    assets_dir="/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/",
)
cfg._cfg.collision_detection_buff_scale = 64.0
eng = Engine(cfg)

arm_tf = np.eye(4)
arm_tf[:3, :3] = ARM_SCALE * Rotation.from_rotvec([-math.pi/2, 0, 0]).as_matrix()
arm_tf[1, 3] = -0.9
eng.native.load_urdf(URDF_PATH, arm_tf, True, False, 1e7, {})
recs = eng.get_load_records()
hand_rec = next(r for r in recs if r.label == "left_arm_hand")
all_v = eng.native.get_vertices_host()
hand_center = all_v[hand_rec.vertex_offset:hand_rec.vertex_offset + hand_rec.vertex_count].mean(0)

gripper_rest_centroid = np.array([0.01166745, 0.08269585, 0.01558503])
translation = hand_center - gripper_rest_centroid * ARM_SCALE
gripper_T = np.eye(4)
gripper_T[:3, :3] *= ARM_SCALE
gripper_T[:3, 3] = translation
eng.load_mesh(RIGID_MSH, dimensions=3, body_type="ABD",
              transform=gripper_T, young_modulus=1e8, boundary_type="Free")
gripper_rec = eng.get_load_records()[-1]
gripper_id = gripper_rec.body_offset
gripper_v_offset = gripper_rec.vertex_offset

data = np.load(UNIFIED_NPZ)
verts = np.ascontiguousarray(data['vertices'], dtype=np.float64)
tets = np.ascontiguousarray(data['tets'], dtype=np.int32)
eng.native.load_mesh_from_data(verts, tets, 4, 3, 1, gripper_T, 1e8, 0)
gripper_fem_rec = eng.get_load_records()[-1]
gripper_fem_v_offset = gripper_fem_rec.vertex_offset
n_abd = sum(1 for r in eng.get_load_records() if r.body_type == 0)
gripper_fem_global = n_abd + gripper_fem_rec.body_offset

remap = np.load(RIGID_REMAP, allow_pickle=True)
rigid_v_idx = remap['rigid_v_idx']
n_rigid = len(rigid_v_idx)
for i in range(n_rigid):
    eng.native.add_fem_pin_to_abd(
        gripper_fem_v_offset + int(rigid_v_idx[i]),
        gripper_v_offset + i,
        gripper_id, (0,0,0))

eng.native.add_collision_exclusion(gripper_id, gripper_fem_global)
eng.native.add_collision_exclusion(hand_rec.body_offset, gripper_id)
eng.native.add_collision_exclusion(hand_rec.body_offset, gripper_fem_global)
for r in recs:
    if r.body_type == 0 and r.body_offset != hand_rec.body_offset:
        eng.native.add_collision_exclusion(r.body_offset, gripper_id)
        eng.native.add_collision_exclusion(r.body_offset, gripper_fem_global)

# Add fixed_joint
fj_idx = eng.native.add_fixed_joint(
    parent_body=hand_rec.body_offset, child_body=gripper_id,
    world_anchor=hand_center,
    world_normal=np.array([1.0, 0.0, 0.0]),
    world_bitangent=np.array([0.0, 0.0, 1.0]),
)

eng.finalize()
eng.native.set_body_apply_gravity(gripper_id, False)
FJ_K = float(os.environ.get("FJ_K", "1e3"))
eng.native.set_fixed_joint_strength(fj_idx, FJ_K)
print(f"[align] fixed_joint kappa = {FJ_K}")

print(f"\n[align] gripper transform scale = {ARM_SCALE}")
print(f"[align] hand_center = {hand_center}")

# At t=0, get verts
v0 = eng.get_vertices()
abd_verts = v0[gripper_v_offset:gripper_v_offset + 150]
fem_verts = v0[gripper_fem_v_offset:gripper_fem_v_offset + 450]
fem_rigid_verts = fem_verts[rigid_v_idx]   # 150 FEM verts pinned to ABD

# ABD vert i should equal FEM rigid vert i (same world position)
diff_init = abd_verts - fem_rigid_verts
print(f"\n[align] === t=0 init alignment ===")
print(f"  ABD verts:        first 3 = {abd_verts[:3].mean(0)}")
print(f"  FEM rigid verts:  first 3 = {fem_rigid_verts[:3].mean(0)}")
print(f"  diff per-vert:    mean={np.linalg.norm(diff_init, axis=1).mean()*1000:.4f} mm  "
      f"max={np.linalg.norm(diff_init, axis=1).max()*1000:.4f} mm")

# Step a few times (no joint motion, just settle)
for s in range(5):
    eng.step()

v1 = eng.get_vertices()
abd1 = v1[gripper_v_offset:gripper_v_offset + 150]
fem1 = v1[gripper_fem_v_offset:gripper_fem_v_offset + 450]
fem_rigid1 = fem1[rigid_v_idx]
diff_after = abd1 - fem_rigid1
print(f"\n[align] === after 5 settle steps (joint targets at 0) ===")
print(f"  ABD center:       {abd1.mean(0)}")
print(f"  FEM rigid center: {fem_rigid1.mean(0)}")
print(f"  diff per-vert:    mean={np.linalg.norm(diff_after, axis=1).mean()*1000:.4f} mm  "
      f"max={np.linalg.norm(diff_after, axis=1).max()*1000:.4f} mm")

# Check ABD bbox vs FEM rigid bbox (size check — if ABD scaled differently)
abd_bbox = abd1.max(0) - abd1.min(0)
fem_rigid_bbox = fem_rigid1.max(0) - fem_rigid1.min(0)
print(f"\n[align] === bbox size comparison ===")
print(f"  ABD rigid bbox:   {abd_bbox * 1000} mm")
print(f"  FEM rigid bbox:   {fem_rigid_bbox * 1000} mm")
print(f"  size ratio:       {abd_bbox / fem_rigid_bbox}")

# === Now ramp left_arm_joint5 (wrist) and re-check alignment ===
panda_j5 = None
for i in range(eng.native.get_num_revolute_joints()):
    info = eng.native.get_revolute_joint_info(i)
    if info.name == "left_arm_joint5":
        panda_j5 = i
        break
print(f"\n[align] left_arm_joint5 idx={panda_j5}")

# Ramp joint5 to 30 deg, take 60 steps
for s in range(60):
    target = math.radians(30.0 * min((s+1)/30, 1.0))   # ramp over 30 steps, hold for 30
    eng.native.set_revolute_target(panda_j5, target)
    eng.native.set_revolute_strength(panda_j5, 1.0)
    eng.step()

v2 = eng.get_vertices()
abd2 = v2[gripper_v_offset:gripper_v_offset + 150]
fem2 = v2[gripper_fem_v_offset:gripper_fem_v_offset + 450]
fem_rigid2 = fem2[rigid_v_idx]
diff_motion = abd2 - fem_rigid2

hand_now = v2[hand_rec.vertex_offset:hand_rec.vertex_offset + hand_rec.vertex_count].mean(0)

print(f"\n[align] === after joint5 ramp to 30°, 60 steps ===")
print(f"  hand center:      {hand_now}  (moved by {(hand_now - hand_center)*1000} mm)")
print(f"  ABD center:       {abd2.mean(0)}  (moved {(abd2.mean(0) - abd1.mean(0))*1000} mm)")
print(f"  FEM rigid center: {fem_rigid2.mean(0)}  (moved {(fem_rigid2.mean(0) - fem_rigid1.mean(0))*1000} mm)")
print(f"  ABD vs FEM diff:  mean={np.linalg.norm(diff_motion, axis=1).mean()*1000:.4f} mm  "
      f"max={np.linalg.norm(diff_motion, axis=1).max()*1000:.4f} mm")
print(f"  hand vs ABD diff: {np.linalg.norm(hand_now - abd2.mean(0))*1000:.3f} mm  (fixed_joint check)")

#!/usr/bin/env python3
"""Headless drift test for case_32: rotate gripper 0 → +60 → 0 → -60 → 0
and measure final position vs initial.  With gravity disabled on gripper,
drift should be sub-mm (joint penalty residual only)."""
import sys, os, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _use_dailyv2_engine  # noqa: F401

import numpy as np
from stiff_physics import Engine, Config

UNIFIED_NPZ = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/hybrid_d/STRATEGY_F_unified.npz"
RIGID_MSH   = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/hybrid_d/STRATEGY_F_rigid.msh"
RIGID_REMAP = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/hybrid_d/STRATEGY_F_rigid_remap.npz"
CUBE_MSH    = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/tetmesh/cube.msh"


def setup_engine(disable_gripper_gravity: bool):
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

    arm_T = np.eye(4); arm_T[:3,:3] *= 0.04; arm_T[3,3] = 1.0
    arm_T[:3, 3] = [0.012, 0.18, 0.016]
    eng.load_mesh(CUBE_MSH, dimensions=3, body_type="ABD",
                  transform=arm_T, young_modulus=1e8, boundary_type="Fixed")
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

    joint_idx = eng.native.add_revolute_joint(
        arm_id, gripper_id, np.array([0.0, 0.0, 1.0]),
        np.array([0.012, 0.14, 0.016]),
        -math.radians(90), math.radians(90), 0.0, "test")
    eng.native.add_collision_exclusion(arm_id, gripper_id)
    eng.native.add_collision_exclusion(arm_id, fem_global)
    eng.native.add_collision_exclusion(gripper_id, fem_global)
    eng.finalize()

    if disable_gripper_gravity:
        eng.native.set_body_apply_gravity(gripper_id, False)

    eng.native.set_revolute_target(joint_idx, 0.0)
    return eng, gripper_id, joint_idx


def run_cycle(eng, gripper_id, joint_idx, target_deg_sequence,
              steps_per_segment=120, rate_deg_per_step=0.5):
    """Drive joint through a sequence of target angles, rate-limited."""
    driven_deg = 0.0
    for target_deg in target_deg_sequence:
        for _ in range(steps_per_segment):
            delta = target_deg - driven_deg
            if abs(delta) > rate_deg_per_step:
                driven_deg += rate_deg_per_step * np.sign(delta)
            else:
                driven_deg = target_deg
            eng.native.set_revolute_target(joint_idx, math.radians(driven_deg))
            eng.step()


def measure_state(eng, gripper_id):
    xf = eng.native.get_abd_body_transforms(np.array([gripper_id], dtype=np.int32))
    return np.array(xf[0], dtype=np.float64)


for label, gravity_on in [("NO gravity (fix)",       False)]:
    print(f"\n===== {label} =====", flush=True)
    eng, gripper_id, joint_idx = setup_engine(disable_gripper_gravity=not gravity_on)

    # Settle 5 steps at 0°
    for _ in range(5):
        eng.step()
    initial_xf = measure_state(eng, gripper_id)
    print(f"initial q.t = ({initial_xf[0,3]:+.5f}, {initial_xf[1,3]:+.5f}, "
          f"{initial_xf[2,3]:+.5f})", flush=True)

    # Drive: 0 → +5° → 0 → -5° → 0
    print("driving cycle: 0→+5→0→-5→0 ...", flush=True)
    run_cycle(eng, gripper_id, joint_idx,
              [5.0, 0.0, -5.0, 0.0],
              steps_per_segment=20, rate_deg_per_step=0.3)

    final_xf = measure_state(eng, gripper_id)
    drift_t = final_xf[:3, 3] - initial_xf[:3, 3]
    drift_t_mm = drift_t * 1000

    # Rotation drift: ||A_final - A_initial||_F
    drift_R = np.linalg.norm(final_xf[:3, :3] - initial_xf[:3, :3])
    drift_R_deg = math.degrees(drift_R)

    print(f"final   q.t = ({final_xf[0,3]:+.5f}, {final_xf[1,3]:+.5f}, "
          f"{final_xf[2,3]:+.5f})", flush=True)
    print(f"DRIFT translation: ({drift_t_mm[0]:+.3f}, {drift_t_mm[1]:+.3f}, "
          f"{drift_t_mm[2]:+.3f}) mm  ‖drift‖={np.linalg.norm(drift_t_mm):.3f} mm",
          flush=True)
    print(f"DRIFT rotation: ‖ΔA‖_F = {drift_R:.5f}  (~{drift_R_deg:.3f}° equiv)",
          flush=True)

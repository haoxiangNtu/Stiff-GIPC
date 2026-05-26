#!/usr/bin/env python3
"""case_35_step1: bare-bones smoke test — load stripped ridgeback URDF
only (no hybrid gripper, no other bodies) and step a few times.
Validates that engine can ingest the URDF's .stl collision meshes via
load_surfaceMesh_ABD path."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _use_dailyv2_engine  # noqa: F401

import numpy as np
from stiff_physics import Engine, Config

URDF_PATH = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/urdf/ridgeback_dual_panda_soft/ridgeback_left_arm_only.urdf"
SCALE = 0.3


def main():
    cfg = Config(
        dt=0.020,
        cloth_thickness=1e-3, cloth_young_modulus=1e4, bend_young_modulus=1e3,
        cloth_density=200, strain_rate=100, soft_motion_rate=1e4,
        poisson_rate=0.49, friction_rate=0.4, relative_dhat=1e-4,
        joint_strength_ratio=200.0, revolute_driving_strength_ratio=300.0,
        semi_implicit_enabled=True, semi_implicit_beta_tol=5e-2,
        semi_implicit_min_iter=1, newton_tol=5e-2,
        preconditioner_type=0, ground_offset=-1.0,
        assets_dir="/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/",
    )
    cfg._cfg.collision_detection_buff_scale = 32.0
    eng = Engine(cfg)

    # Position arm in world; root_fixed=True keeps base frozen
    arm_tf = np.eye(4)
    arm_tf[:3, :3] *= SCALE
    arm_tf[3, 3] = 1.0
    arm_tf[1, 3] = 0.5  # raise base slightly above ground

    # Stripped URDF only contains the 18 links needed for left_arm_hand
    # kinematic chain. plate_link is still 28k verts so we override it +
    # other heavy non-arm links.  panda link0..7 (~150-300 verts each) and
    # left_arm_hand (~1k verts) keep their original .stl meshes.
    CUBE_MSH = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/tetmesh/cube.msh"
    HEAVY_LINKS = ["chassis_link", "plate_link", "arm_mount_link",
                   "left_arm_mount_link", "left_arm_hand"]
    for link_name in HEAVY_LINKS:
        eng.native.set_urdf_mesh_override(link_name, CUBE_MSH, 1e7)

    print(f"\n[case35-step1] loading URDF: {URDF_PATH}", flush=True)
    print(f"[case35-step1] {len(HEAVY_LINKS)} non-arm heavy links → cube.msh",
          flush=True)
    eng.native.load_urdf(URDF_PATH, arm_tf,
                         True,    # root_fixed
                         False,   # revolute_as_motor (False = simple constraint)
                         1e7,     # default_young
                         {})      # initial_joint_angles
    print(f"[case35-step1] loaded; ABD bodies = {eng.abd_body_count}", flush=True)
    print(f"[case35-step1] revolute joints = {eng.native.get_num_revolute_joints()}",
          flush=True)
    print(f"[case35-step1] prismatic joints = {eng.native.get_num_prismatic_joints()}",
          flush=True)

    # Print loaded body labels
    recs = eng.get_load_records()
    print(f"\n[case35-step1] {len(recs)} body records (first 15):", flush=True)
    for r in recs[:15]:
        print(f"  body {r.body_offset:>2}: type={r.body_type} verts={r.vertex_count:>5}  "
              f"label={r.label[:60] if r.label else '?'}",
              flush=True)
    if len(recs) > 15:
        print(f"  ... and {len(recs) - 15} more", flush=True)

    # Print joint names
    print(f"\n[case35-step1] revolute joints:", flush=True)
    for i in range(min(eng.native.get_num_revolute_joints(), 20)):
        info = eng.native.get_revolute_joint_info(i)
        print(f"  rev #{i}: name='{info.name}' limits=[{info.lower_limit:.2f}, "
              f"{info.upper_limit:.2f}]", flush=True)

    eng.finalize()
    print(f"\n[case35-step1] finalized, running 5 steps...", flush=True)

    for i in range(5):
        t0 = time.perf_counter()
        eng.step()
        print(f"  step {i}: {(time.perf_counter()-t0)*1000:.1f} ms", flush=True)


if __name__ == "__main__":
    main()

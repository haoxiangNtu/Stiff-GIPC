#!/usr/bin/env python3
"""Ridgeback Dual Panda + Shirt free-fall (semi-implicit) — Case 27-S1 (mobile, OBB except gripper, all ABD).

Direct port of Case 26 with the arm swapped from XArm7+Gripper to the
Ridgeback Dual Panda (nomobile, OBB collision URDF). All physics config,
shirt placement, scene up-axis (y_up), and arm orientation transform
(URDF z-up -> scene y-up) match Case 26 exactly so timings can be
compared head-to-head.

Usage:
    PYTHONPATH=. python examples/case_27_ridgeback_panda_cloth.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from pathlib import Path
from stiff_physics.engine import Engine, Config
from stiff_physics.robot import Robot

ASSETS_DIR = str(Path(__file__).resolve().parent.parent / "assets") + "/"

URDF_RELPATH = "sim_data/urdf/ridgeback_dual_panda_soft/ridgeback_dual_panda2_mobile_s1.urdf"


def _make_arm_transform(scale: float = 0.3) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    tf = np.eye(4)
    tf[:3, :3] = scale * Rotation.from_rotvec([-math.pi / 2, 0, 0]).as_matrix()
    tf[0, 3] = 0.0
    tf[1, 3] = -0.9
    tf[2, 3] = 0.0
    return tf


def main():
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")

    config = Config(
        dt=0.020,                          # perf_tuned tier (was 0.010)
        cloth_thickness=1e-3,
        cloth_young_modulus=1e4,
        bend_young_modulus=1e3,
        cloth_density=200,
        strain_rate=100,
        soft_motion_rate=1.0,
        poisson_rate=0.49,
        friction_rate=0.4,
        relative_dhat=1e-3,
        joint_strength_ratio=100.0,                   # perf_tuned tier (was 1000)
        revolute_driving_strength_ratio=100.0,        # perf_tuned tier (was 1000)
        semi_implicit_enabled=True,
        semi_implicit_beta_tol=5e-2,                  # perf_tuned tier (was 1e-3)
        semi_implicit_min_iter=1,
        newton_tol=5e-2,                              # perf_tuned tier
        preconditioner_type=0,
        ground_offset=-0.5,  # half of default y=-1.0; lowered from previous -0.35
        assets_dir=ASSETS_DIR,
    )

    engine = Engine(config)
    assets_dir = engine.native.get_assets_dir()

    # --- Robot arm (Ridgeback dual-panda, nomobile, OBB collision) ---
    arm_tf = _make_arm_transform(0.3)
    engine.native.load_urdf(
        assets_dir + URDF_RELPATH,
        arm_tf, True, False, 1e7,
    )

    arm_body_count = engine.abd_body_count
    print(f"[case27] Loaded {arm_body_count} ABD bodies from URDF")
    for bid in range(arm_body_count):
        engine.add_ground_collision_skip(bid)

    # --- Shirt (free-fall, no pinning) ---
    shirt_scale = 0.5
    shirt_tf = np.eye(4)
    shirt_tf[:3, :3] *= shirt_scale
    shirt_tf[0, 3] = 0.34   # moved right (+x) by 0.28 so shirt clears the arm during free-fall
    shirt_tf[1, 3] = 0.3
    shirt_tf[2, 3] = 0.0
    engine.load_mesh("triMesh/shirt_6436v.obj", dimensions=2, body_type="FEM",
                     transform=shirt_tf, young_modulus=1e2)

    # Auto-exclude non-gripper arm bodies vs cloth -- triggers BVH-skip #2/#3
    # for ~8.5x speedup. Only finger / soft_material bodies stay collidable
    # with cloth.
    #
    # Physical consequence: non-finger bodies pass THROUGH cloth without
    # contact (no IPC barrier records the pair). Sanity-check honors the
    # same exclusion matrix, so no INTERSECT spam.
    #
    # BVHSKIP_NO_EXCL=1 disables (back to baseline ~370ms with full contact).
    if os.environ.get("BVHSKIP_NO_EXCL") != "1":
        cloth_global_id = arm_body_count
        records = engine.get_load_records()
        finger_kw = ("finger", "soft_material")
        excl_n = 0
        for r in records:
            if r.body_type != 0:
                continue
            if any(k in r.label.lower() for k in finger_kw):
                continue
            engine.add_collision_exclusion(r.body_offset, cloth_global_id)
            excl_n += 1
        print(f"[case27] arm-cloth exclusions added for {excl_n} non-gripper "
              f"bodies (set BVHSKIP_NO_EXCL=1 to disable). Non-gripper bodies "
              f"will pass THROUGH cloth -- finger bodies still interact normally.")

    engine.finalize()
    robot = Robot(engine)

    verts = engine.get_vertices()
    faces = engine.get_surface_faces()
    mesh = ps.register_surface_mesh("scene", verts, faces, smooth_shade=True)
    mesh.set_color((0.6, 0.7, 0.8))

    running = [False]
    step_count = [0]

    def _begin_window(title):
        result = psim.Begin(title, True)
        return result[0] if isinstance(result, tuple) else result

    def callback():
        psim.SetNextWindowPos((330, 10), psim.ImGuiCond_Once)
        psim.SetNextWindowSize((380, 0), psim.ImGuiCond_Once)

        if _begin_window("Ridgeback Panda + Shirt Free-Fall (Case 27-S1 (mobile, OBB except gripper, all ABD))"):
            if running[0]:
                if psim.Button("Pause"):
                    running[0] = False
            else:
                if psim.Button("Run"):
                    running[0] = True

            psim.SameLine()
            psim.Text(f"Step: {step_count[0]}")
            psim.Separator()

            if robot.revolute_joints:
                psim.Text(f"Revolute Joints ({len(robot.revolute_joints)})")
                psim.Separator()
                for i, ji in enumerate(robot.revolute_joints):
                    lo = math.degrees(ji.lower_limit)
                    hi = math.degrees(ji.upper_limit)
                    cur = robot.get_revolute_target_deg(i)
                    changed, new_val = psim.SliderFloat(ji.name, cur, lo, hi)
                    if changed:
                        robot.set_revolute_position(i, new_val, degree=True)

            if robot.prismatic_joints:
                psim.Spacing()
                psim.Text(f"Prismatic Joints ({len(robot.prismatic_joints)})")
                psim.Separator()
                for i, ji in enumerate(robot.prismatic_joints):
                    lo_mm = ji.lower_limit * 1000.0
                    hi_mm = ji.upper_limit * 1000.0
                    cur_mm = robot.get_prismatic_target_mm(i)
                    changed, new_val = psim.SliderFloat(
                        f"{ji.name} (mm)", cur_mm, lo_mm, hi_mm)
                    if changed:
                        robot.set_prismatic_position(i, new_val, millimeters=True)

            psim.Spacing()
            if psim.Button("Reset All Joints"):
                robot.reset_all()

        psim.End()

        if running[0]:
            engine.step()
            step_count[0] += 1
            mesh.update_vertex_positions(engine.get_vertices())

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()

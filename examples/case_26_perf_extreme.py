#!/usr/bin/env python3
"""Case 26 — extreme-performance variant (builds on case_26_perf_tuned.py).

Adds one extra layer on top of perf_tuned: per-joint strength override via
the `Robot.set_gripper_strength()` API (available in v0.3.0+).

  Layer 3 — per-joint compliance on gripper (~1.8× extra on stall):
    arm joints      : global joint_strength_ratio = 200 (stable feel when
                      dragging sliders; no visible shake)
    gripper joints  : per-joint multiplier = 0.1 → effective K = 20 (soft;
                      yields freely under cloth contact)

Why this split:
    The arm's link-to-link joints (joint1..joint7) carry the arm's own
    weight and are what the user visually grabs via sliders. Keeping them
    at 200 avoids the "whole arm shaking" symptom observed at global
    strength≤50.

    The gripper's finger/knuckle joints are at the end of the kinematic
    chain, carry tiny mass, and are what actually contacts cloth in
    grasping. Dropping them to effective K=20 relieves the arm-sweep-
    cloth pinch without affecting the main arm's interactive feel.

    The stall is driven by joint-penalty Hessian stiffness clashing with
    IPC barrier near pinch contact (see docs/internal/case26_gripper_
    strength_ab_n30.log). Since case_26's arm-sweep contact is mediated
    partially through the gripper (when gripper trails the arm into
    cloth), softening the gripper link gives a smaller but measurable
    extra speedup.

Measured on perf_tuned baseline (arm=200, grip=200) with joint2 slider
descent from 0° to -113° over 60 frames:

    grip_mul=1.0  (grip effective 200): descend median ~86 ms/step
    grip_mul=0.5  (grip effective 100): descend median ~63 ms/step  (1.4×)
    grip_mul=0.25 (grip effective 50):  descend median ~47 ms/step  (1.8×)
    grip_mul=0.1  (grip effective 20):  descend median ~47 ms/step  (1.8×)  ← this script
    grip_mul=0.05 (grip effective 10):  descend median ~48 ms/step  (plateau)
    grip_mul=0.01 (grip effective 2):   descend median ~63 ms/step  (gripper flops)

Sweet spot: grip_mul in [0.1, 0.25]. Below that, gripper starts flopping
under its own weight. Above that, less stall relief.

Trade-off to know:
    * Gripper fingers may visibly droop / swing slightly under gravity
      when the arm is held static — this is spring-mass physics with a
      softer spring, not a bug.
    * If you need the gripper to firmly grasp a heavier object, raise
      grip_mul back toward 1.0 dynamically via
      `robot.set_gripper_strength(1.0)` before the grasp phase.

When to prefer this script vs `case_26_perf_tuned.py`:
    * Use `perf_extreme` when you want maximum speed for interactive
      prototyping with a cloth scene where the arm will sweep around.
    * Use `perf_tuned` when you want uniform joint behaviour and don't
      mind the extra stall in sweep scenarios.

Usage:
    python examples/case_26_perf_extreme.py
"""

import math
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from pathlib import Path
from stiff_physics.engine import Engine, Config
from stiff_physics.robot import Robot

ASSETS_DIR = str(Path(__file__).resolve().parent.parent / "assets") + "/"


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

    # Same perf_tuned config as case_26_perf_tuned.py.
    config = Config(
        dt=0.020,
        cloth_thickness=1e-3,
        cloth_young_modulus=1e4,
        bend_young_modulus=1e3,
        cloth_density=200,
        strain_rate=100,
        soft_motion_rate=1.0,
        poisson_rate=0.49,
        friction_rate=0.4,
        relative_dhat=1e-3,
        joint_strength_ratio=200.0,             # arm (global) — stable feel
        revolute_driving_strength_ratio=200.0,
        semi_implicit_enabled=True,
        semi_implicit_beta_tol=5e-2,
        semi_implicit_min_iter=1,
        newton_tol=5e-2,
        pcg_tol=1e-4,
        assets_dir=ASSETS_DIR,
    )

    engine = Engine(config)
    assets_dir = engine.native.get_assets_dir()

    arm_tf = _make_arm_transform(0.3)
    engine.native.load_urdf(
        assets_dir + "sim_data/urdf/xarm/xarm7_with_gripper.urdf",
        arm_tf, True, False, 1e7,
    )

    arm_body_count = engine.abd_body_count
    for bid in range(arm_body_count):
        engine.add_ground_collision_skip(bid)

    shirt_scale = 0.5
    shirt_tf = np.eye(4)
    shirt_tf[:3, :3] *= shirt_scale
    shirt_tf[0, 3] = 0.25
    shirt_tf[1, 3] = 0.3
    shirt_tf[2, 3] = 0.0
    engine.load_mesh("triMesh/shirt_6436v.obj", dimensions=2, body_type="FEM",
                     transform=shirt_tf, young_modulus=1e2)

    engine.finalize()
    robot = Robot(engine)

    # ★ Layer 3 — per-joint gripper softening. The key addition over perf_tuned.
    #
    # Default matches 7 arm joints (joint1..joint7) + 6 gripper joints
    # (finger/knuckle/drive_joint patterns). Returns number set so you can
    # sanity-check pattern matching against your URDF.
    GRIP_MUL = 0.1  # edit here to re-tune: 0.25 is the conservative option
    n_set = robot.set_gripper_strength(GRIP_MUL)
    print(f"[perf_extreme] {n_set} gripper joints at multiplier {GRIP_MUL} "
          f"(effective K = {200*GRIP_MUL:.0f});  "
          f"{engine.native.get_num_revolute_joints() - n_set} arm joints "
          f"at 1.0 (K = 200).")

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

        if _begin_window("Arm + Shirt — perf_extreme (per-joint gripper)"):
            if running[0]:
                if psim.Button("Pause"):
                    running[0] = False
            else:
                if psim.Button("Run"):
                    running[0] = True
            psim.SameLine()
            psim.Text(f"Step: {step_count[0]}")
            psim.Separator()
            psim.TextColored((0.5, 1.0, 0.5, 1.0),
                             f"gripper mul = {GRIP_MUL} (K = {200*GRIP_MUL:.0f})")
            psim.TextColored((0.7, 0.7, 0.7, 1.0), "arm joints K = 200 (stable)")
            psim.Separator()

            if robot.revolute_joints:
                psim.Text("Revolute Joints")
                psim.Separator()
                for i, ji in enumerate(robot.revolute_joints):
                    lo = math.degrees(ji.lower_limit)
                    hi = math.degrees(ji.upper_limit)
                    cur = robot.get_revolute_target_deg(i)
                    changed, new_val = psim.SliderFloat(ji.name, cur, lo, hi)
                    if changed:
                        robot.set_revolute_position(i, new_val, degree=True)

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

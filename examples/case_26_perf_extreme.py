#!/usr/bin/env python3
"""Case 26 — extreme-performance variant (Option B — builds on perf_tuned).

Adds one extra layer on top of `case_26_perf_tuned.py`: per-joint
strength override via `Robot.set_gripper_strength()` (v0.3.0+).

PREREQUISITE — collision mesh fix (same as perf_tuned)
------------------------------------------------------
Both this script and perf_tuned assume xarm7 collision meshes have
been pre-fixed:

    python examples/fix_obj_winding.py path/to/your/robot.urdf \\
        --auto-fix --collision-only --in-place

Without that step, multiple bodies (notably gripper_base_link, several
fingers / knuckles, link2, link6, link_base) are non-manifold, which
makes the mass / centroid / inertia integrals ill-defined and the
strength tuning here suboptimal.

Configuration on top of perf_tuned (Option A)
---------------------------------------------
  Layer 3 — per-joint compliance on gripper:
    arm joints      : global joint_strength_ratio = 100 (same as perf_tuned)
    gripper joints  : per-joint multiplier = 0.1 → effective K = 10 (soft;
                      yields freely under cloth contact)

Why this split:
    The arm's joint1..joint7 carry the arm's own weight and are what the
    user grabs via sliders. Keeping them at K=100 avoids visible shake.

    The gripper's finger/knuckle joints are at the end of the kinematic
    chain, carry tiny mass, and are what actually contacts cloth. Dropping
    their effective K to 10 absorbs pinch-spike stalls (the contact-
    induced p95 long tail in the Newton solver).

Measured impact at strength=100 (post mesh-fix), n=200 user-qpos frames:

    grip_mul=1.0  (no per-joint softening): median 19.93 ms  p95 93.95 ms
    grip_mul=0.1  (this script's value)   : median 18.33 ms  p95 50.69 ms
                                            ─────────────  ─────────────
                                            -8% median      -46% p95 ⭐

Median improvement is small (-8%); the value of this layer is the
**p95 stall reduction**: cuts long-tail spikes by ~50%. Visible as
"smoother slider drag" (no occasional 'kerchunk' moments when the
gripper enters cloth contact).

Sweet spot from sweep at perf_tuned baseline: grip_mul in [0.1, 0.25].
Below 0.05 the gripper starts flopping under its own weight; above 0.5
diminishing returns on stall reduction.

Trade-off:
    * Gripper fingers may visibly droop / swing slightly under gravity
      when the arm is held static — this is spring-mass physics with a
      softer spring, not a bug.
    * If you need the gripper to firmly grasp a heavier object, raise
      grip_mul back toward 1.0 dynamically:
        `robot.set_gripper_strength(1.0)` before the grasp phase

When to prefer perf_extreme (Option B) vs perf_tuned (Option A)
---------------------------------------------------------------
    * perf_extreme: smooth interactive feel (50 ms p95) — recommended
      for slider-driven prototyping, demos, manipulation control loops.
    * perf_tuned:   simpler API surface, single Config knob, no
      per-joint API call. p95 ~94 ms (occasional perceptible stall).
      Recommended when you want minimal Python plumbing or you're
      wrapping the engine for high-throughput batch sims where p95
      doesn't matter as much as throughput.

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
        joint_strength_ratio=100.0,             # arm (global) — same as perf_tuned post-mesh-fix
        revolute_driving_strength_ratio=100.0,
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
    GLOBAL_K = 100.0  # mirror Config.joint_strength_ratio above; used in display only.
    GRIP_MUL = 0.1    # edit here to re-tune: 0.25 is the conservative option
    n_set = robot.set_gripper_strength(GRIP_MUL)
    print(f"[perf_extreme] {n_set} gripper joints at multiplier {GRIP_MUL} "
          f"(effective K = {GLOBAL_K * GRIP_MUL:.0f});  "
          f"{engine.native.get_num_revolute_joints() - n_set} arm joints "
          f"at 1.0 (K = {GLOBAL_K:.0f}).")

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
                             f"gripper mul = {GRIP_MUL} (K = {GLOBAL_K * GRIP_MUL:.0f})")
            psim.TextColored((0.7, 0.7, 0.7, 1.0), f"arm joints K = {GLOBAL_K:.0f} (stable)")
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

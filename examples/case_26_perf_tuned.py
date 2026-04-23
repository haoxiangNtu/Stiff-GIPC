#!/usr/bin/env python3
"""Case 26 — performance-tuned variant (~1.26× faster than the default).

Same scene as `case_26_arm_cloth_semi_implicit.py` (XArm7 + Gripper +
shirt_6436v free-fall), but with two solver-tolerance knobs loosened to
trade a small amount of per-step convergence accuracy for speed.

Tuned vs default Config (everything else identical):

    semi_implicit_beta_tol = 5e-2    # default 1e-3 (50× looser)
    newton_tol             = 5e-2    # default 1e-2 (5× looser)
    relative_dhat          = 1e-3    # ★ default unchanged
    pcg_tol                = 1e-4    # ★ default unchanged

Measured on case_26 free-fall (100 steps, RTX 4090, v0.1.1 wheel):

    default:   42.7 ms/step
    tuned:     33.9 ms/step              (1.26× speedup)
    cloth Y-fall: identical to within 0.3 mm
    cloth shape drift (sorted L∞): 3.1 mm (~1% of cloth dimension)

Knobs deliberately kept at default:

    relative_dhat:  Tightening to 5e-4 was reported in early tuning notes
                    to halve collision pair count. In practice (with the
                    v0.1.x engine binary), it gives 0% to -4% net speedup
                    — the cp-count win is offset elsewhere — and risks
                    missing fast contacts. Leave alone.
    pcg_tol:        Loosening hurts PCG search-direction accuracy, which
                    can require more outer Newton iters to compensate.
                    Net win is unclear and the safety risk is real. Leave
                    alone.

Trade-offs to be aware of:

    * `semi_implicit_beta_tol = 5e-2` makes Newton exit ~5× sooner. In
      simple free-fall + light contact this is fine; in complex multi-body
      contact scenes the early exit may accumulate physics error.
    * `newton_tol = 5e-2` relaxes the per-step convergence threshold.
      Same trade-off — fine for visual sims, worth re-evaluating if you
      need force-accurate contact responses.
    * Numbers above are from case_26 specifically. Re-measure on your own
      scene before assuming the same speedup or accuracy holds.

When to prefer this script:
    * Visual sim / demo / interactive prototyping
    * Cloth free-fall and light cloth-rigid contact

When to prefer `case_26_arm_cloth_semi_implicit.py`:
    * Force-accurate contact studies
    * Convergence ablation experiments
    * As a baseline reference for further tuning

Usage:
    python examples/case_26_perf_tuned.py
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

    # ---- Performance-tuned Config ----
    # Only two knobs differ from case_26_arm_cloth_semi_implicit.py:
    #   semi_implicit_beta_tol: 1e-3 -> 5e-2  (Newton exits ~5x sooner)
    #   newton_tol:             1e-2 -> 5e-2  (5x looser convergence)
    # See module docstring for the rationale and trade-offs.
    config = Config(
        dt=0.01,
        cloth_thickness=1e-3,
        cloth_young_modulus=1e4,
        bend_young_modulus=1e3,
        cloth_density=200,
        strain_rate=100,
        soft_motion_rate=1.0,
        poisson_rate=0.49,
        friction_rate=0.4,
        relative_dhat=1e-3,                  # default
        joint_strength_ratio=1000.0,
        revolute_driving_strength_ratio=1000.0,
        semi_implicit_enabled=True,
        semi_implicit_beta_tol=5e-2,         # ★ tuned (default 1e-3)
        semi_implicit_min_iter=1,
        newton_tol=5e-2,                     # ★ tuned (default 1e-2)
        pcg_tol=1e-4,                        # default
        assets_dir=ASSETS_DIR,
    )

    engine = Engine(config)
    assets_dir = engine.native.get_assets_dir()

    # --- Robot arm ---
    arm_tf = _make_arm_transform(0.3)
    engine.native.load_urdf(
        assets_dir + "sim_data/urdf/xarm/xarm7_with_gripper.urdf",
        arm_tf, True, False, 1e7,
    )

    arm_body_count = engine.abd_body_count
    for bid in range(arm_body_count):
        engine.add_ground_collision_skip(bid)

    # --- Shirt (free-fall, no pinning) ---
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

    verts = engine.get_vertices()
    faces = engine.get_surface_faces()
    mesh = ps.register_surface_mesh("scene", verts, faces, smooth_shade=True)
    mesh.set_color((0.6, 0.7, 0.8))

    running = [False]
    step_count = [0]

    def _begin_window(title):
        """Compatible with both old polyscope (returns bool) and new (returns tuple)."""
        result = psim.Begin(title, True)
        return result[0] if isinstance(result, tuple) else result

    def callback():
        psim.SetNextWindowPos((330, 10), psim.ImGuiCond_Once)
        psim.SetNextWindowSize((380, 0), psim.ImGuiCond_Once)

        if _begin_window("Arm + Shirt Free-Fall (Case 26 — Perf Tuned)"):
            if running[0]:
                if psim.Button("Pause"):
                    running[0] = False
            else:
                if psim.Button("Run"):
                    running[0] = True

            psim.SameLine()
            psim.Text(f"Step: {step_count[0]}")
            psim.Separator()
            psim.Text("Tuned: beta_tol=5e-2, newton_tol=5e-2")
            psim.Text("(Expect ~1.26x speedup vs basic case_26)")
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

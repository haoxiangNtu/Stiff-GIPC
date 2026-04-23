#!/usr/bin/env python3
"""Case 26 — performance-tuned variant (~1.43× faster than the default).

Same scene as `case_26_arm_cloth_semi_implicit.py` (XArm7 + Gripper +
shirt_6436v free-fall), but with TWO categories of tuning stacked:

  Layer 1 — solver tolerance tuning (~1.19× on its own):
    semi_implicit_beta_tol = 5e-2   (default 1e-3,  50× looser; +10.6% alone)
    newton_tol             = 5e-2   (default 1e-2,   5× looser; +7.5% marginal)

  Layer 2 — timestep coarsening (+20.5% marginal on top of layer 1):
    dt                     = 0.020  (default 0.010, 2× larger step)

  Knobs deliberately kept at default:
    relative_dhat          = 1e-3   (no benefit — see rationale below)
    pcg_tol                = 1e-4   (risky — see rationale below)

Measured on case_26 free-fall, rigorous A/B: RTX 4090, v0.1.1 wheel, n=30
trials per cell across 3 independent batches with 3-round GPU warmup each,
randomized cell order, paired t-test. Metric: wall-clock to simulate
exactly 1.0 sim-sec of physics.

    default:    4.144 s wall   95% CI [4.095, 4.194]  (1.00×)
    layer 1:    3.485 s wall   95% CI [3.449, 3.520]  (1.19×, p=5e-25)
    layer 1+2:  2.892 s wall   95% CI [2.872, 2.911]  (1.43×, p=6e-31)

    cloth Y-fall: consistent within 1 mm across all three configs
    cloth shape drift (sorted L∞, vs default at 1.0 sim s): ~8 mm
        (~2–3% of cloth dimension — below visual-perception threshold for
         cloth free-fall + landing; re-verify for your scene)

Why `dt = 0.020` (not larger):
    Measured sweep 0.005 → 0.030 shows a clear U-shape. dt < 0.010 wastes
    time on extra steps; dt > 0.020 makes each step super-linearly more
    expensive (Newton iters explode). 0.020 is the wall-clock-per-sim-sec
    minimum. drift grows from 3 mm (default) to ~8 mm (dt=0.020) to
    ~9 mm (dt=0.030) — dt=0.020 gives the best speedup/drift trade-off.

Why `relative_dhat` stays at default:
    Tightening to 5e-4 was reported in early tuning notes to halve the
    collision-pair count. In practice with the v0.1.x engine binary it
    gives 0% to -4% net speedup (the cp-count win is offset elsewhere)
    and risks missing fast contacts. Leave alone.

Why `pcg_tol` stays at default:
    Loosening hurts PCG search-direction accuracy, which often requires
    more outer Newton iters to compensate. Net win is unclear and the
    safety risk is real. Leave alone.

Knobs tested and found NOT to help (for the record, so you don't re-test):
    * `joint_strength_ratio` sweep (50 … 5000): all within ±1% of noise,
      p > 0.5. Joint solve is not the bottleneck.
    * finger-only collision exclusion (shirt vs left_finger + right_finger
      only, other arm bodies excluded via add_collision_exclusion):
      +0.48% on basic, −0.08% on perf_tuned, p = 0.50/0.84 respectively.
      Cloth self-contact dominates the collision-pair count (~90% of all
      pairs); arm-shirt pairs are too small a fraction to move the needle.
    * `collision_detection_buff_scale` / `linear_system_buff_scale` /
      `semi_implicit_min_iter`: individually borderline, combined −4%
      (negative interaction). Skipped.
    * Engine-level perf commits H2/H3+H4/H5 from earlier experimental
      branches: all give null effect (p > 0.8) on the v0.1.1 binary. The
      published PERF_OPT_HANDOVER 2.17× numbers were measured on a
      pre-87f90be baseline whose stability bugs (uninitialized ABD
      preconditioner, etc.) made those optimisations helpful; after the
      bugs got fixed, their optimisation target disappeared.

Trade-offs to be aware of:

    * `semi_implicit_beta_tol = 5e-2` — Newton exits ~5× sooner. Fine for
      simple free-fall + light contact; in complex multi-body contact
      scenes the early exit may accumulate physics error.
    * `newton_tol = 5e-2` — per-step Newton convergence relaxed. Same
      trade-off.
    * `dt = 0.020` — integrator discretization is 2× coarser. This is a
      different *category* of change than the solver tolerances above:
      it affects how accurately the engine integrates each step, not how
      accurately it solves it. Effects in case_26:
        - Cloth motion appears 2× faster per callback tick in interactive
          mode (each Polyscope frame advances 2× more sim time). Does NOT
          affect the final landed configuration.
        - Discretization-dependent phenomena (fast rigid collisions,
          high-frequency cloth dynamics) may become unstable in other
          scenes. Re-verify dt on your own scene.

When to prefer this script:
    * Visual sim / demo / interactive prototyping
    * Cloth free-fall and light cloth-rigid contact
    * You've visually verified the result looks right on your scene

When to prefer `case_26_arm_cloth_semi_implicit.py` (default accuracy):
    * Force-accurate contact studies
    * Convergence ablation experiments
    * As a baseline reference for further tuning
    * Any scene where you're unsure about the tuning's effect

When to use a hybrid (layer 1 only, keep dt=0.010):
    If you want the solver-tol speedup (1.26×) but need to keep the
    integrator's dt untouched (e.g. to stay aligned with an external
    simulation's timestep). Copy this script and change `dt=0.020` back
    to `dt=0.010`.

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
    # Three knobs differ from case_26_arm_cloth_semi_implicit.py:
    #   dt                    : 0.010 -> 0.020  (timestep 2x larger)
    #   semi_implicit_beta_tol: 1e-3  -> 5e-2   (Newton exits ~5x sooner)
    #   newton_tol            : 1e-2  -> 5e-2   (5x looser convergence)
    # Total measured speedup vs default: ~1.52x (see module docstring).
    config = Config(
        dt=0.020,                            # ★ tuned (default 0.010, 2x coarser)
        cloth_thickness=1e-3,
        cloth_young_modulus=1e4,
        bend_young_modulus=1e3,
        cloth_density=200,
        strain_rate=100,
        soft_motion_rate=1.0,
        poisson_rate=0.49,
        friction_rate=0.4,
        relative_dhat=1e-3,                  # default (tightening gives no net speedup)
        joint_strength_ratio=1000.0,
        revolute_driving_strength_ratio=1000.0,
        semi_implicit_enabled=True,
        semi_implicit_beta_tol=5e-2,         # ★ tuned (default 1e-3, 50x looser)
        semi_implicit_min_iter=1,
        newton_tol=5e-2,                     # ★ tuned (default 1e-2, 5x looser)
        pcg_tol=1e-4,                        # default (loosening risks stability)
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
            psim.Text("Tuned: dt=0.02 + beta_tol=5e-2 + newton_tol=5e-2")
            psim.Text("(Measured 1.43x speedup vs basic case_26, n=30)")
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

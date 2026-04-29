"""Sweep joint_strength_ratio while keeping other perf_tuned params fixed.
Measures both (a) descend-phase wall-time median and (b) max tracking error
|actual - target| in degrees, so we can find the breakpoint where the arm
starts lagging too much.
"""
import sys, os, math, time
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stiff_physics.engine import Engine, Config
from stiff_physics.robot import Robot


def _resolve_assets():
    here = Path(__file__).resolve().parent.parent
    for n in ("assets", "Assets"):
        c = here / n
        if (c / "sim_data" / "urdf" / "xarm" / "xarm7_with_gripper.urdf").exists():
            return str(c) + "/"


def main():
    strength = float(sys.argv[1]) if len(sys.argv) > 1 else 200.0

    # === perf_tuned config (all other knobs locked at their chosen values) ===
    cfg = Config(
        dt=0.020,                          # perf_tuned
        cloth_thickness=1e-3, cloth_young_modulus=1e4, bend_young_modulus=1e3,
        cloth_density=200, strain_rate=100, soft_motion_rate=1.0,
        poisson_rate=0.49, friction_rate=0.4, relative_dhat=1e-3,
        joint_strength_ratio=strength,                  # ← swept
        revolute_driving_strength_ratio=strength,        # ← swept
        semi_implicit_enabled=True,
        semi_implicit_beta_tol=5e-2,       # perf_tuned
        semi_implicit_min_iter=1,
        newton_tol=5e-2,                   # perf_tuned
        pcg_tol=1e-4,
        assets_dir=_resolve_assets(),
    )
    engine = Engine(cfg)
    assets = engine.native.get_assets_dir()

    arm_tf = np.eye(4)
    arm_tf[:3, :3] = 0.3 * Rotation.from_rotvec([-math.pi / 2, 0, 0]).as_matrix()
    arm_tf[1, 3] = -0.9
    engine.native.load_urdf(assets + "sim_data/urdf/xarm/xarm7_with_gripper.urdf",
                            arm_tf, True, False, 1e7)
    for bid in range(engine.abd_body_count):
        engine.add_ground_collision_skip(bid)
    shirt_tf = np.eye(4); shirt_tf[:3, :3] *= 0.5
    shirt_tf[0, 3] = 0.25; shirt_tf[1, 3] = 0.3
    engine.load_mesh("triMesh/shirt_6436v.obj", dimensions=2, body_type="FEM",
                     transform=shirt_tf, young_modulus=1e2)
    engine.finalize()
    robot = Robot(engine)

    shoulder_idx = next(i for i, ji in enumerate(robot.revolute_joints) if "joint2" in ji.name)
    init_target = robot.get_revolute_target_deg(shoulder_idx)
    end_target = math.degrees(robot.revolute_joints[shoulder_idx].lower_limit) + 5.0

    # warmup
    for _ in range(30):
        engine.step()

    descend_ms = []
    tracking_errs_deg = []
    for frame in range(60):
        alpha = frame / 59.0
        target = init_target + (end_target - init_target) * alpha
        robot.set_revolute_position(shoulder_idx, target, degree=True)
        t0 = time.perf_counter()
        engine.step()
        descend_ms.append((time.perf_counter() - t0) * 1000.0)
        actual_rad = engine.native.get_revolute_current_angles()[shoulder_idx]
        actual_deg = math.degrees(actual_rad)
        tracking_errs_deg.append(abs(target - actual_deg))

    descend_ms.sort()
    median = descend_ms[len(descend_ms)//2]
    max_err = max(tracking_errs_deg)
    mean_err = sum(tracking_errs_deg) / len(tracking_errs_deg)

    v = engine.get_vertices()
    has_nan = bool(np.isnan(v).any())

    print(f"STRENGTH={strength:<7} "
          f"WALL_MEDIAN_MS={median:<7.2f} "
          f"TRACK_MAX_DEG={max_err:<8.4f} "
          f"TRACK_MEAN_DEG={mean_err:<8.4f} "
          f"NAN={has_nan}")


if __name__ == "__main__":
    main()

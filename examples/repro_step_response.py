"""Step-response test: apply a sudden target change (like user dragging
slider abruptly), then record how much the arm oscillates / overshoots
before settling.

Metrics:
  - overshoot_deg: max |actual - target_new| right after step change
  - settle_frames: frames until |actual - target_new| stays < 0.5 deg
  - ringing_amplitude: max swing between frame 3 and frame 20
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

    cfg = Config(
        dt=0.020, cloth_thickness=1e-3, cloth_young_modulus=1e4,
        bend_young_modulus=1e3, cloth_density=200, strain_rate=100,
        soft_motion_rate=1.0, poisson_rate=0.49, friction_rate=0.4,
        relative_dhat=1e-3,
        joint_strength_ratio=strength, revolute_driving_strength_ratio=strength,
        semi_implicit_enabled=True, semi_implicit_beta_tol=5e-2,
        semi_implicit_min_iter=1, newton_tol=5e-2, pcg_tol=1e-4,
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

    # Warmup: hold target at 0 deg, let shirt settle
    init_target_deg = 0.0
    robot.set_revolute_position(shoulder_idx, init_target_deg, degree=True)
    for _ in range(40):
        engine.step()

    # === STEP INPUT: abruptly change target from 0° to -30° ===
    # (simulates user dragging slider in one quick motion)
    step_target_deg = -30.0
    robot.set_revolute_position(shoulder_idx, step_target_deg, degree=True)

    # Record actual angle for 40 frames after step
    actuals = []
    wall_ms = []
    for frame in range(40):
        t0 = time.perf_counter()
        engine.step()
        wall_ms.append((time.perf_counter() - t0) * 1000)
        actuals.append(math.degrees(engine.native.get_revolute_current_angles()[shoulder_idx]))

    # Metrics
    errs = [abs(a - step_target_deg) for a in actuals]
    max_overshoot = max(actuals[i] - step_target_deg for i in range(len(actuals)))  # signed, >0 if overshooting past target
    # "Overshoot" here: how far did the angle go PAST the target
    # (e.g. if target is -30 and actual hits -35, overshoot = 5)
    overshoot = 0.0
    for a in actuals:
        if step_target_deg < init_target_deg:  # target is BELOW initial
            ot = step_target_deg - a  # positive if a < target (past it)
            overshoot = max(overshoot, ot)

    # Settle time: first frame where all subsequent |err| < 0.5 deg
    settle = None
    for i in range(len(errs)):
        if all(e < 0.5 for e in errs[i:]):
            settle = i
            break
    if settle is None:
        settle = len(errs)

    # Ringing: peak-to-peak angular swing in frames 3-20 (after initial overshoot)
    if len(actuals) > 20:
        window = actuals[3:20]
        ringing = max(window) - min(window)
    else:
        ringing = 0.0

    # Steady-state error (last 10 frames mean err)
    ss_err = sum(errs[-10:]) / 10

    v = engine.get_vertices()
    has_nan = bool(np.isnan(v).any())

    wall_ms.sort(); wall_med = wall_ms[len(wall_ms)//2]

    print(f"STRENGTH={strength:<6} "
          f"OVERSHOOT_DEG={overshoot:5.2f} "
          f"RINGING_DEG={ringing:5.2f} "
          f"SETTLE_FRAMES={settle:3d} "
          f"SS_ERR_DEG={ss_err:5.3f} "
          f"WALL_MED_MS={wall_med:6.2f} "
          f"NAN={has_nan}")


if __name__ == "__main__":
    main()

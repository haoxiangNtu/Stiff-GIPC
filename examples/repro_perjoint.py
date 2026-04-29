"""Test per-joint strength API.

Mode ('arm_mode' + 'grip_mode' factor on top of global=1000):
  arm=1.0, grip=1.0    : all joints at 1000 effective (same as S1000 baseline)
  arm=1.0, grip=0.1    : arm 1000 / gripper 100 (per-joint, precise arm + soft gripper)
  arm=0.2, grip=0.2    : all 200 (same as S200 global)
  arm=0.1, grip=0.1    : all 100 (same as S100 global)
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
    arm_mul  = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    grip_mul = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    cfg = Config(
        dt=0.01, cloth_thickness=1e-3, cloth_young_modulus=1e4, bend_young_modulus=1e3,
        cloth_density=200, strain_rate=100, soft_motion_rate=1.0,
        poisson_rate=0.49, friction_rate=0.4, relative_dhat=1e-3,
        joint_strength_ratio=1000.0,   # stays at 1000 globally
        revolute_driving_strength_ratio=1000.0,
        semi_implicit_enabled=True, semi_implicit_beta_tol=1e-3,
        semi_implicit_min_iter=1, assets_dir=_resolve_assets(),
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

    # --- Apply per-joint strengths ---
    # Arm joints (joint1-7) get arm_mul.
    # Gripper joints get grip_mul.
    GRIPPER_PATTERNS = ("finger", "knuckle", "drive_joint")
    arm_count, grip_count = 0, 0
    for i, ji in enumerate(robot.revolute_joints):
        is_grip = any(p in ji.name for p in GRIPPER_PATTERNS)
        mul = grip_mul if is_grip else arm_mul
        robot.set_revolute_strength(i, mul)
        if is_grip: grip_count += 1
        else:       arm_count  += 1
    print(f"# arm joints (mul={arm_mul}): {arm_count}")
    print(f"# gripper joints (mul={grip_mul}): {grip_count}")

    shoulder_idx = next(i for i, ji in enumerate(robot.revolute_joints) if "joint2" in ji.name)
    init_target = robot.get_revolute_target_deg(shoulder_idx)
    end_target = math.degrees(robot.revolute_joints[shoulder_idx].lower_limit) + 5.0

    for _ in range(30):
        engine.step()

    descend_ms = []
    for frame in range(60):
        alpha = frame / 59.0
        target = init_target + (end_target - init_target) * alpha
        robot.set_revolute_position(shoulder_idx, target, degree=True)
        t0 = time.perf_counter()
        engine.step()
        descend_ms.append((time.perf_counter() - t0) * 1000.0)

    descend_ms.sort()
    median = descend_ms[len(descend_ms)//2]
    v = engine.get_vertices()
    has_nan = bool(np.isnan(v).any())
    print(f"ARM_MUL={arm_mul} GRIP_MUL={grip_mul} DESCEND_MEDIAN_MS={median:.4f} NAN={has_nan}")


if __name__ == "__main__":
    main()

"""Minimal wall-time benchmark for case_26 (no diagnostic API needed)."""
import sys, os, math, time
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stiff_physics.engine import Engine, Config


def _resolve_assets() -> str:
    here = Path(__file__).resolve().parent.parent
    for n in ("assets", "Assets"):
        c = here / n
        if (c / "sim_data" / "urdf" / "xarm" / "xarm7_with_gripper.urdf").exists():
            return str(c) + "/"
    raise FileNotFoundError(f"No assets dir under {here}")


def run(steps=100, warmup=20, quiet=False):
    cfg = Config(
        dt=0.01,
        cloth_thickness=1e-3,
        cloth_young_modulus=1e4,
        bend_young_modulus=1e3,
        cloth_density=200,
        strain_rate=100,
        soft_motion_rate=1.0,
        poisson_rate=0.49,
        friction_rate=0.4,
        relative_dhat=1e-3,
        joint_strength_ratio=1000.0,
        revolute_driving_strength_ratio=1000.0,
        semi_implicit_enabled=True,
        semi_implicit_beta_tol=1e-3,
        semi_implicit_min_iter=1,
        assets_dir=_resolve_assets(),
    )
    engine = Engine(cfg)
    assets = engine.native.get_assets_dir()

    arm_tf = np.eye(4)
    arm_tf[:3, :3] = 0.3 * Rotation.from_rotvec([-math.pi / 2, 0, 0]).as_matrix()
    arm_tf[1, 3] = -0.9
    engine.native.load_urdf(
        assets + "sim_data/urdf/xarm/xarm7_with_gripper.urdf",
        arm_tf, True, False, 1e7,
    )
    for bid in range(engine.abd_body_count):
        engine.add_ground_collision_skip(bid)

    shirt_tf = np.eye(4)
    shirt_tf[:3, :3] *= 0.5
    shirt_tf[0, 3] = 0.25
    shirt_tf[1, 3] = 0.3
    engine.load_mesh("triMesh/shirt_6436v.obj", dimensions=2, body_type="FEM",
                     transform=shirt_tf, young_modulus=1e2)
    engine.finalize()

    for _ in range(warmup):
        engine.step()

    times = []
    for i in range(steps):
        t0 = time.perf_counter()
        engine.step()
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    n = len(times)
    median = times[n // 2]
    total = sum(times)
    print(f"WALL_MS_PER_STEP_MEDIAN = {median:.4f}")
    print(f"WALL_MS_PER_STEP_MEAN   = {total/n:.4f}")
    print(f"TOTAL_MS_FOR_{n}_STEPS   = {total:.2f}")

    # Sanity: vertices should not be NaN
    v = engine.get_vertices()
    if np.isnan(v).any():
        print("FAIL: NaN in vertex positions", file=sys.stderr)
        sys.exit(2)
    return median


if __name__ == "__main__":
    steps  = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    warmup = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    run(steps, warmup)

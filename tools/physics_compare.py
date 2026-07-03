"""Run case_26 N steps with a given binary, dump final vertex positions.

Usage: physics_compare.py <steps> <out.npy>
"""
import sys, os, math
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


def main():
    steps = int(sys.argv[1])
    out_path = sys.argv[2]

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

    for _ in range(steps):
        engine.step()

    v = engine.get_vertices()
    np.save(out_path, v)
    print(f"SAVED {out_path}: {v.shape}, mean={np.mean(v):.6f}, "
          f"min={np.min(v):.6f}, max={np.max(v):.6f}, "
          f"any_nan={bool(np.isnan(v).any())}")


if __name__ == "__main__":
    main()

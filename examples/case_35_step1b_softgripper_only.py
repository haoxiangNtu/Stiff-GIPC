#!/usr/bin/env python3
"""Smoke test: load case_27's _softgripper URDF as-is (no mesh override,
no strip) to confirm full ridgeback+panda renders correctly.  Same setup
as case_27_mobile_s1_softgripper_cup.py but without cup/table/cloth."""
import sys, os, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _use_dailyv2_engine  # noqa: F401

import numpy as np
from scipy.spatial.transform import Rotation
from stiff_physics import Engine, Config

URDF_PATH = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/urdf/ridgeback_dual_panda_soft/ridgeback_dual_panda2_mobile_s1_softgripper.urdf"
SCALE = 0.3


def make_arm_tf(scale):
    """Match case_27's transform: scale + rotate URDF z-up to scene y-up."""
    tf = np.eye(4)
    tf[:3, :3] = scale * Rotation.from_rotvec([-math.pi/2, 0, 0]).as_matrix()
    tf[0, 3] = 0.0
    tf[1, 3] = -0.9
    tf[2, 3] = 0.0
    return tf


def main():
    cfg = Config(
        dt=0.020,
        cloth_thickness=1e-3, cloth_young_modulus=1e4, bend_young_modulus=1e3,
        cloth_density=200, strain_rate=100, soft_motion_rate=1.0,
        poisson_rate=0.49, friction_rate=0.4, relative_dhat=1e-3,
        joint_strength_ratio=100.0, revolute_driving_strength_ratio=100.0,
        semi_implicit_enabled=True, semi_implicit_beta_tol=5e-2,
        semi_implicit_min_iter=1, newton_tol=5e-2,
        preconditioner_type=0, ground_offset=-0.5,
        assets_dir="/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/",
    )
    cfg._cfg.collision_detection_buff_scale = 64.0  # case_27 needed this

    eng = Engine(cfg)
    arm_tf = make_arm_tf(SCALE)
    print(f"\n[smoke] loading {URDF_PATH}", flush=True)
    eng.native.load_urdf(URDF_PATH, arm_tf, True, False, 1e7, {})
    n = eng.abd_body_count
    print(f"[smoke] {n} ABD bodies", flush=True)
    for b in range(n):
        eng.add_ground_collision_skip(b)

    # Print body verts (find the heavy ones)
    recs = eng.get_load_records()
    print(f"\n[smoke] body verts (sorted by count, top 15):")
    by_verts = sorted(recs, key=lambda r: -r.vertex_count)
    for r in by_verts[:15]:
        print(f"  body {r.body_offset:>2}: verts={r.vertex_count:>6}  {r.label}",
              flush=True)
    total_verts = sum(r.vertex_count for r in recs)
    print(f"  TOTAL: {total_verts} verts across {len(recs)} bodies", flush=True)

    eng.finalize()
    print(f"\n[smoke] finalized, running 3 steps...", flush=True)
    for i in range(3):
        t0 = time.perf_counter()
        eng.step()
        print(f"  step {i}: {(time.perf_counter()-t0)*1000:.1f} ms", flush=True)


if __name__ == "__main__":
    main()

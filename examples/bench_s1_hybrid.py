#!/usr/bin/env python3
"""Headless bench for case_27 s1_hybrid with auto arm-cloth exclusion.

Adds add_collision_exclusion(arm_b, cloth_b) for every arm body whose URDF
link name does NOT contain 'finger' or 'soft_material'. Non-gripper arm parts
become BVH-skip-isolated, exercising #1 (collision matrix), #2 (diag iso +
empty-bbox / early-return), and #3 (indirect BVH input filter).

Toggle individual patches via env vars to isolate effects:
    SKIP_EXCLUSIONS=1  no add_collision_exclusion calls (true baseline)
    BVHSKIP2=0         disables #2 (also auto-disables #3)
    BVHSKIP3=0         disables #3 only
"""
import sys, os, time, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from stiff_physics.engine import Engine, Config

ASSETS_DIR = str(Path(__file__).resolve().parent.parent / "assets") + "/"
URDF_RELPATH = "sim_data/urdf/ridgeback_dual_panda_soft/ridgeback_dual_panda2_mobile_s1_hybrid.urdf"


def make_arm_tf(scale=0.3):
    tf = np.eye(4)
    tf[:3, :3] = scale * Rotation.from_rotvec([-math.pi / 2, 0, 0]).as_matrix()
    tf[1, 3] = -0.9
    return tf


def main():
    cfg = Config(
        dt=0.020, cloth_thickness=1e-3, cloth_young_modulus=1e4, bend_young_modulus=1e3,
        cloth_density=200, strain_rate=100, soft_motion_rate=1.0,
        poisson_rate=0.49, friction_rate=0.4, relative_dhat=1e-3,
        joint_strength_ratio=100.0, revolute_driving_strength_ratio=100.0,
        semi_implicit_enabled=True, semi_implicit_beta_tol=5e-2, semi_implicit_min_iter=1,
        newton_tol=5e-2, preconditioner_type=0, ground_offset=-0.5,
        assets_dir=ASSETS_DIR,
    )
    eng = Engine(cfg)
    eng.native.load_urdf(eng.native.get_assets_dir() + URDF_RELPATH,
                         make_arm_tf(0.3), True, False, 1e7)
    n_arm = eng.abd_body_count
    for b in range(n_arm):
        eng.add_ground_collision_skip(b)

    shirt_tf = np.eye(4); shirt_tf[:3, :3] *= 0.5
    shirt_tf[0, 3] = 0.34; shirt_tf[1, 3] = 0.3
    eng.load_mesh("triMesh/shirt_6436v.obj", dimensions=2, body_type="FEM",
                  transform=shirt_tf, young_modulus=1e2)

    # Auto-detect non-gripper arm bodies and exclude them from cloth.
    # Cloth is the (0-based) FEM body; its global body id = n_arm.
    # SKIP_EXCLUSIONS=1 makes this a true no-#1 baseline (matrix stays all-zero).
    skip_excl = os.environ.get("SKIP_EXCLUSIONS") == "1"
    if skip_excl:
        print("[bench] SKIP_EXCLUSIONS=1 -- no add_collision_exclusion calls (true baseline)")
    else:
        cloth_global_id = n_arm
        records = eng.get_load_records()
        finger_keywords = ("finger", "soft_material")
        excluded = []
        kept = []
        for r in records:
            if r.body_type != 0:  # skip FEM/cloth
                continue
            label_lc = r.label.lower()
            is_finger = any(k in label_lc for k in finger_keywords)
            if is_finger:
                kept.append((r.body_offset, r.label))
            else:
                eng.add_collision_exclusion(r.body_offset, cloth_global_id)
                excluded.append((r.body_offset, r.label))
        print(f"[bench] arm-cloth exclusions added for {len(excluded)} non-gripper bodies; "
              f"kept {len(kept)} gripper bodies collidable with cloth")
        if kept:
            print(f"[bench] gripper bodies: {[lbl for _,lbl in kept]}")

    eng.finalize()
    print(f"[bench] verts={len(eng.get_vertices())}", flush=True)

    N_WARM, N_MEAS = 5, 15
    for i in range(N_WARM):
        eng.step()
    times = []
    for i in range(N_MEAS):
        t0 = time.perf_counter()
        eng.step()
        times.append(time.perf_counter() - t0)
    times.sort()
    print(f"[bench] N={N_MEAS}  min={1e3*min(times):.1f}ms  "
          f"median={1e3*statistics.median(times):.1f}ms  "
          f"mean={1e3*statistics.mean(times):.1f}ms  "
          f"max={1e3*max(times):.1f}ms")
    print(f"[bench] all: {[round(1e3*t,1) for t in times]}")


if __name__ == "__main__":
    main()

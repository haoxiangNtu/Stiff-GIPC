#!/usr/bin/env python3
"""Detailed perf bench for case_27 mobile s1 obb_except_gripper, comparing
original (rough non-watertight finger) vs clean (case 11 watertight finger).

Captures per-step:
  - wall time (perf_counter)
  - Newton iterations consumed
  - PCG iterations consumed
  - Collision pair count

Set CLEAN=1 to use clean-finger URDF, CLEAN=0 for original.
"""
import sys, os, time, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from stiff_physics.engine import Engine, Config

ASSETS_DIR = str(Path(__file__).resolve().parent.parent / "assets") + "/"
USE_CLEAN = os.environ.get("CLEAN") == "1"
URDF_NAME = ("ridgeback_dual_panda2_mobile_s1_clean.urdf" if USE_CLEAN
             else "ridgeback_dual_panda2_mobile_s1.urdf")
URDF_RELPATH = f"sim_data/urdf/ridgeback_dual_panda_soft/{URDF_NAME}"


def make_arm_tf(scale=0.3):
    tf = np.eye(4)
    tf[:3, :3] = scale * Rotation.from_rotvec([-math.pi / 2, 0, 0]).as_matrix()
    tf[1, 3] = -0.9
    return tf


def main():
    cfg = Config(
        dt=0.020, cloth_thickness=1e-3, cloth_young_modulus=1e4,
        bend_young_modulus=1e3, cloth_density=200, strain_rate=100,
        soft_motion_rate=1.0, poisson_rate=0.49, friction_rate=0.4,
        relative_dhat=1e-3, joint_strength_ratio=100.0,
        revolute_driving_strength_ratio=100.0,
        semi_implicit_enabled=True, semi_implicit_beta_tol=5e-2,
        semi_implicit_min_iter=1, newton_tol=5e-2,
        preconditioner_type=0, ground_offset=-0.5,
        assets_dir=ASSETS_DIR,
    )
    eng = Engine(cfg)
    print(f"[bench] URDF = {URDF_NAME}", flush=True)
    eng.native.load_urdf(eng.native.get_assets_dir() + URDF_RELPATH,
                         make_arm_tf(0.3), True, False, 1e7)
    n_arm = eng.abd_body_count
    for b in range(n_arm):
        eng.add_ground_collision_skip(b)

    shirt_tf = np.eye(4); shirt_tf[:3, :3] *= 0.5
    shirt_tf[0, 3] = 0.34; shirt_tf[1, 3] = 0.3
    eng.load_mesh("triMesh/shirt_6436v.obj", dimensions=2, body_type="FEM",
                  transform=shirt_tf, young_modulus=1e2)

    # Auto-exclude non-finger arm bodies vs cloth (triggers BVH-skip #1+#2+#3)
    cloth_global_id = n_arm
    records = eng.get_load_records()
    finger_kw = ("finger", "soft_material")
    n_excl = 0
    for r in records:
        if r.body_type != 0:
            continue
        if any(k in r.label.lower() for k in finger_kw):
            continue
        eng.add_collision_exclusion(r.body_offset, cloth_global_id)
        n_excl += 1
    print(f"[bench] arm-cloth exclusions: {n_excl}", flush=True)

    eng.finalize()
    n_verts = len(eng.get_vertices())
    n_faces = len(eng.get_surface_faces())
    print(f"[bench] verts={n_verts} faces={n_faces}", flush=True)

    # Warmup
    N_WARM, N_MEAS = 3, 12
    for _ in range(N_WARM):
        eng.step()

    # Measurement: capture per-step deltas
    e = eng.native
    t_list, nt_list, pcg_list, cp_list = [], [], [], []
    nt_prev = e.get_total_newton_iters()
    pcg_prev = e.get_total_pcg_iters()
    cp_prev = e.get_total_collision_pairs()
    for i in range(N_MEAS):
        t0 = time.perf_counter()
        eng.step()
        dt = time.perf_counter() - t0
        nt_now = e.get_total_newton_iters()
        pcg_now = e.get_total_pcg_iters()
        cp_now = e.get_total_collision_pairs()
        t_list.append(dt * 1000)
        nt_list.append(nt_now - nt_prev)
        pcg_list.append(pcg_now - pcg_prev)
        cp_list.append(cp_now - cp_prev)
        nt_prev, pcg_prev, cp_prev = nt_now, pcg_now, cp_now

    def stats(name, arr):
        sa = sorted(arr)
        med = statistics.median(sa)
        mn, mx, avg = min(sa), max(sa), statistics.mean(sa)
        print(f"  {name:>14}: min={mn:7.2f} median={med:7.2f} mean={avg:7.2f} max={mx:7.2f}")

    print(f"\n[bench] === {URDF_NAME} === (CLEAN={int(USE_CLEAN)})")
    print(f"[bench] verts={n_verts} faces={n_faces} N_MEAS={N_MEAS}")
    stats("step_ms", t_list)
    stats("newton_iter", nt_list)
    stats("pcg_iter", pcg_list)
    stats("contact_pair", cp_list)

    print(f"\n[bench] all_step_ms = {[round(x, 2) for x in t_list]}")
    print(f"[bench] all_newton  = {nt_list}")
    print(f"[bench] all_pcg     = {pcg_list}")
    print(f"[bench] all_pairs   = {[int(x) for x in cp_list]}")


if __name__ == "__main__":
    main()

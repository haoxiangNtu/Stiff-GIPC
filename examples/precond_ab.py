#!/usr/bin/env python3
"""Headless A/B comparison: preconditioner_type=1 (MAS) vs =0 (no MAS).

Same scene as case_26_arm_cloth_semi_implicit.py, run twice with different
preconditioner_type, dump shirt vertex positions per step, report per-step
L-inf distance and a final summary.

Run:
    PYTHONPATH=. python examples/precond_ab.py [N_STEPS]
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

from stiff_physics.engine import Engine, Config


ASSETS_DIR = str(Path(__file__).resolve().parent.parent / "assets") + "/"


def _arm_tf(scale: float = 0.3) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    tf = np.eye(4)
    tf[:3, :3] = scale * Rotation.from_rotvec([-math.pi / 2, 0, 0]).as_matrix()
    tf[1, 3] = -0.9
    return tf


def _shirt_tf() -> np.ndarray:
    tf = np.eye(4) * 0.5
    tf[3, 3] = 1.0
    tf[0, 3] = 0.25
    tf[1, 3] = 0.3
    tf[2, 3] = 0.0
    return tf


def run_simulation(precond_type: int, n_steps: int) -> tuple[np.ndarray, int, int]:
    """Run case_26 scene with the given preconditioner_type. Return:
        positions[step, vertex, xyz]   shape (n_steps+1, n_shirt_verts, 3)
        v_offset                       shirt's first vertex index in global array
        v_count                        shirt vertex count
    """
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
        preconditioner_type=precond_type,
        assets_dir=ASSETS_DIR,
    )

    engine = Engine(cfg)
    assets_dir = engine.native.get_assets_dir()

    engine.native.load_urdf(
        assets_dir + "sim_data/urdf/xarm/xarm7_with_gripper.urdf",
        _arm_tf(0.3), True, False, 1e7,
    )
    for bid in range(engine.abd_body_count):
        engine.add_ground_collision_skip(bid)

    engine.load_mesh(
        "triMesh/shirt_6436v.obj",
        dimensions=2, body_type="FEM",
        transform=_shirt_tf(), young_modulus=1e2,
    )

    engine.finalize()

    # Locate shirt vertices in the global array via load records
    records = engine.native.get_all_load_records()
    shirt = next(r for r in records
                 if r.body_type == 1 and "shirt" in r.label)
    v_off, v_cnt = shirt.vertex_offset, shirt.vertex_count

    positions = np.zeros((n_steps + 1, v_cnt, 3), dtype=np.float64)
    positions[0] = engine.get_vertices()[v_off:v_off + v_cnt]

    t0 = time.perf_counter()
    for k in range(n_steps):
        engine.step()
        positions[k + 1] = engine.get_vertices()[v_off:v_off + v_cnt]
    elapsed = time.perf_counter() - t0
    print(f"  precond={precond_type}: {n_steps} steps, {elapsed:.2f}s "
          f"({elapsed * 1000 / n_steps:.1f} ms/step), shirt verts={v_cnt}",
          flush=True)
    return positions, v_off, v_cnt


def summarise(label: str, A: np.ndarray, B: np.ndarray) -> None:
    """Per-step + cumulative L-inf distance between two trajectories."""
    assert A.shape == B.shape, f"{A.shape} vs {B.shape}"
    n_steps = A.shape[0]
    print(f"\n=== {label} (shape {A.shape}) ===")
    print(f"{'step':>5} | {'L_inf (m)':>14} | {'L_2 mean (m)':>14}")
    print("-" * 45)
    diffs = []
    for k in range(n_steps):
        d = A[k] - B[k]
        linf = float(np.max(np.abs(d)))
        l2_mean = float(np.linalg.norm(d, axis=1).mean())
        diffs.append((k, linf, l2_mean))
    # Print every 5th step + first/last
    keep = set([0, 1, 2, 3, 4] + list(range(0, n_steps, 5)) + [n_steps - 1])
    for k, linf, l2_mean in diffs:
        if k in keep:
            print(f"{k:>5d} | {linf:>14.4e} | {l2_mean:>14.4e}")
    print("-" * 45)
    final_linf = diffs[-1][1]
    max_linf_over_traj = max(d[1] for d in diffs)
    print(f"final L_inf      : {final_linf:.4e} m")
    print(f"max L_inf in run : {max_linf_over_traj:.4e} m")


def main() -> None:
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    precond_a = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    precond_b = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    label = {0: "no MAS (preconditioner_type=0)",
             1: "MAS / metis (preconditioner_type=1)"}

    print(f"=== Run A: {label.get(precond_a, '?')} ===", flush=True)
    posA, off_A, cnt_A = run_simulation(precond_type=precond_a, n_steps=n_steps)

    print(f"\n=== Run B: {label.get(precond_b, '?')} ===", flush=True)
    posB, off_B, cnt_B = run_simulation(precond_type=precond_b, n_steps=n_steps)

    if cnt_A != cnt_B:
        print(f"\n!!! shirt vertex count mismatch: A={cnt_A} vs B={cnt_B}")
        print("    can't do per-vertex compare; preconditioner_type may have")
        print("    changed mesh topology somehow. Aborting comparison.")
        return

    if off_A != off_B:
        print(f"\n  note: shirt vertex_offset differs (A={off_A}, B={off_B})")
        print("        this is fine — comparison is on the per-shirt slice")

    # MAS reorders FEM vertices relative to .obj file. So Run A's vertex k and
    # Run B's vertex k may correspond to DIFFERENT material points. The most
    # honest comparison is to look at distributions, not per-index distances.
    print("\n=== Per-step direct L_inf (NOT meaningful if MAS reorder differs) ===")
    summarise("vertex-index aligned (A vs B)", posA, posB)

    # Sort each step by per-vertex coordinates so reordering doesn't fool us.
    print("\n=== Per-step bag-of-points distance (sort-by-XYZ both sides) ===")
    posA_sorted = np.sort(posA.reshape(posA.shape[0], -1), axis=1)
    posB_sorted = np.sort(posB.reshape(posB.shape[0], -1), axis=1)
    diffs_sorted = posA_sorted - posB_sorted
    print(f"{'step':>5} | {'L_inf sorted (m)':>20}")
    print("-" * 32)
    for k in range(posA.shape[0]):
        if k in set([0, 1, 2, 3, 4] + list(range(0, posA.shape[0], 5)) + [posA.shape[0] - 1]):
            linf = float(np.max(np.abs(diffs_sorted[k])))
            print(f"{k:>5d} | {linf:>20.4e}")

    print("\n=== Bbox comparison per step ===")
    print(f"{'step':>5} | {'bbox-min A':>20} | {'bbox-min B':>20} | "
          f"{'bbox-max A':>20} | {'bbox-max B':>20}")
    for k in (0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps):
        if 0 <= k < posA.shape[0]:
            mnA = posA[k].min(axis=0)
            mnB = posB[k].min(axis=0)
            mxA = posA[k].max(axis=0)
            mxB = posB[k].max(axis=0)
            fmt = lambda v: f"({v[0]:+.3f},{v[1]:+.3f},{v[2]:+.3f})"
            print(f"{k:>5d} | {fmt(mnA):>20} | {fmt(mnB):>20} | "
                  f"{fmt(mxA):>20} | {fmt(mxB):>20}")


if __name__ == "__main__":
    main()

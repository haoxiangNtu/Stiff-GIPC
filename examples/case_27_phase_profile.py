#!/usr/bin/env python3
"""Headless per-phase profiler for case_27 (ridgeback dual-panda + shirt).

Same physics setup as case_27_ridgeback_panda_cloth.py without polyscope.
Reports per-phase wall-clock breakdown (hessian / pcg / ccd / linesearch /
postls) so we can see what dominates step time.

Usage:
    python examples/case_27_phase_profile.py [warmup=20] [measure=200] [precond=0]

  precond: 0 = no MAS (matches the demo default), 1 = MAS preconditioner

To compare both, run twice:
    python examples/case_27_phase_profile.py 20 200 0
    python examples/case_27_phase_profile.py 20 200 1
"""

import math
import sys
import time
import statistics
import numpy as np
from pathlib import Path

from stiff_physics.engine import Engine, Config


URDF_RELPATH = (
    "sim_data/urdf/ridgeback_dual_panda_soft/franka/"
    "ridgeback_dual_panda2_nomobile_obb.urdf"
)


def _resolve_assets_dir() -> str:
    here = Path(__file__).resolve().parent.parent
    for name in ("assets", "Assets"):
        cand = here / name
        if (cand / URDF_RELPATH).exists():
            return str(cand) + "/"
    raise FileNotFoundError(f"No assets dir with ridgeback OBB urdf found under {here}")


ASSETS_DIR = _resolve_assets_dir()


def _make_arm_transform(scale: float = 0.3) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    tf = np.eye(4)
    tf[:3, :3] = scale * Rotation.from_rotvec([-math.pi / 2, 0, 0]).as_matrix()
    tf[1, 3] = -0.9
    return tf


def build_engine(precond: int) -> Engine:
    config = Config(
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
        preconditioner_type=precond,
        assets_dir=ASSETS_DIR,
    )

    engine = Engine(config)
    assets_dir = engine.native.get_assets_dir()

    arm_tf = _make_arm_transform(0.3)
    engine.native.load_urdf(
        assets_dir + URDF_RELPATH,
        arm_tf, True, False, 1e7,
    )

    arm_body_count = engine.abd_body_count
    for bid in range(arm_body_count):
        engine.add_ground_collision_skip(bid)

    shirt_scale = 0.5
    shirt_tf = np.eye(4)
    shirt_tf[:3, :3] *= shirt_scale
    shirt_tf[0, 3] = 0.25
    shirt_tf[1, 3] = 0.3
    engine.load_mesh("triMesh/shirt_6436v.obj", dimensions=2, body_type="FEM",
                     transform=shirt_tf, young_modulus=1e2)

    engine.finalize()
    return engine


def main():
    warmup = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    measure = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    precond = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    engine = build_engine(precond)

    print(f"# warmup={warmup} measure={measure} preconditioner_type={precond}")
    print(f"# engine vertices={engine.vertex_count}, "
          f"abd_bodies={engine.abd_body_count}, fem_bodies={engine.fem_body_count}")

    if not hasattr(engine, "get_last_diag"):
        print("ERROR: Engine has no get_last_diag(); rebuild required.", file=sys.stderr)
        sys.exit(1)

    for _ in range(warmup):
        engine.step()

    keys = ["hessian_ms", "pcg_ms", "ccd_ms", "linesearch_ms", "postls_ms",
            "ls_cd_ms", "ls_energy_ms", "ls_other_ms",
            "cp_count_mean", "ccd_count_mean", "ccd_cp_ratio"]
    samples = {k: [] for k in keys}
    totals_engine = []
    totals_wall = []

    print("frame,wall_ms,hessian_ms,pcg_ms,ccd_ms,linesearch_ms,"
          "ls_cd_ms,ls_energy_ms,ls_other_ms,postls_ms,total_ms,"
          "cp_mean,ccd_mean,ratio,cp_calls,ccd_calls")
    for f in range(measure):
        t0 = time.perf_counter()
        engine.step()
        wall_ms = (time.perf_counter() - t0) * 1000.0
        d = engine.get_last_diag()
        for k in keys:
            samples[k].append(d[k])
        totals_engine.append(d["total_ms"])
        totals_wall.append(wall_ms)
        print(f"{f},{wall_ms:.3f},{d['hessian_ms']:.3f},{d['pcg_ms']:.3f},"
              f"{d['ccd_ms']:.3f},{d['linesearch_ms']:.3f},"
              f"{d['ls_cd_ms']:.3f},{d['ls_energy_ms']:.3f},{d['ls_other_ms']:.3f},"
              f"{d['postls_ms']:.3f},{d['total_ms']:.3f},"
              f"{d['cp_count_mean']:.1f},{d['ccd_count_mean']:.1f},{d['ccd_cp_ratio']:.2f},"
              f"{d['cp_call_count']},{d['ccd_call_count']}")

    print()
    print(f"=== Aggregate (precond={precond}, median, mean, fraction of engine total) ===")
    eng_med = statistics.median(totals_engine)
    eng_mean = statistics.mean(totals_engine)
    wall_med = statistics.median(totals_wall)
    wall_mean = statistics.mean(totals_wall)
    print(f"  wall_ms     : median={wall_med:7.3f}  mean={wall_mean:7.3f}")
    print(f"  engine_total: median={eng_med:7.3f}  mean={eng_mean:7.3f}")
    print()
    print(f"  {'phase':16s} {'median':>12s} {'mean':>12s} {'frac_med':>9s}")
    time_keys = ["hessian_ms", "pcg_ms", "ccd_ms", "linesearch_ms", "postls_ms",
                 "ls_cd_ms", "ls_energy_ms", "ls_other_ms"]
    for k in time_keys:
        med = statistics.median(samples[k])
        mn = statistics.mean(samples[k])
        frac = (med / eng_med) if eng_med > 0 else 0.0
        print(f"  {k:16s} {med:12.3f} {mn:12.3f} {frac:9.2%}")

    print()
    print("  --- pair-count diagnostics ---")
    for k in ["cp_count_mean", "ccd_count_mean", "ccd_cp_ratio"]:
        vals = [v for v in samples[k] if v > 0]
        if not vals:
            print(f"  {k:16s} (no samples)")
            continue
        med = statistics.median(vals)
        mn = statistics.mean(vals)
        mx = max(vals)
        mn_val = min(vals)
        print(f"  {k:16s} median={med:10.2f}  mean={mn:10.2f}  "
              f"min={mn_val:10.2f}  max={mx:10.2f}")

    print()
    overhead = wall_med - eng_med
    print(f"  python/wrap overhead (wall - engine_total) median: {overhead:7.3f} ms "
          f"({overhead/wall_med*100:.1f}% of wall)")


if __name__ == "__main__":
    main()

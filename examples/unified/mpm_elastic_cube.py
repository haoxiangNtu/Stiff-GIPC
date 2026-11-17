#!/usr/bin/env python3
"""Bring-up test of the elastic MPM material (model 1, fixed corotated): a particle cube dropped on the ground.

    PYTHONPATH=. STIFFGIPC_NATIVE_DIR=build_4090 python examples/unified/mpm_elastic_cube.py --out out/unified/elastic_cube --steps 150
"""
from __future__ import annotations
import argparse, json, os, os.path as osp, sys, time
import numpy as np
HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(HERE, "..", "robotgym"))
from common import Recorder, Scene  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--edge", type=float, default=0.2)
    ap.add_argument("--spacing", type=float, default=0.0125)
    ap.add_argument("--drop", type=float, default=0.25)
    ap.add_argument("--young", type=float, default=1e5)
    ap.add_argument("--poisson", type=float, default=0.3)
    ap.add_argument("--rho", type=float, default=1000.0)
    ap.add_argument("--dx", type=float, default=0.0, help="0 = 2 x spacing")
    ap.add_argument("--with-box", action="store_true", help="drop a rigid ABD box on the cube afterwards")
    ap.add_argument("--log-level", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sc = Scene(dict(dt=args.dt, preconditioner_type=0, newton_iter_cap=60, newton_tol=1e-3, pcg_tol=1e-5, absolute_dhat=2e-3,
                    friction_rate=0.3, gd_friction_rate=0.3, gravity=(0.0, -9.81, 0.0)), log_level=args.log_level)
    # the engine needs at least one mesh body for its BVH: a fixed thin plate under the ground plane
    sc.add_box("plate", (0.0, -0.03, 0.0), (1.2, 0.06, 1.2), fixed=True, color=(0.8, 0.8, 0.78), mu=0.3, cell=0.3)
    if args.with_box:
        sc.add_box("box", (0.0, args.drop + args.edge + 0.25, 0.0), (0.12, 0.12, 0.12), fixed=False, density=2000.0, mu=0.3, cell=0.06)
    n1 = int(round(args.edge / args.spacing))
    g = (np.arange(n1) + 0.5) * args.spacing - 0.5 * args.edge
    pts = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    pts[:, 1] += args.drop + 0.5 * args.edge
    pvol = args.spacing ** 3
    dx = args.dx or 2.0 * args.spacing
    lo = np.array([-0.6, -0.05, -0.6]); hi = np.array([0.6, args.drop + 2 * args.edge + 0.6, 0.6])
    sc.eng.add_liquid(pts, dx, lo, hi, rho=args.rho, kappa=1e5, particle_volume=pvol, node_pool=60000, push_apart=0.0, j_mode=0,
                      ground_mu=0.3, model=1, young=args.young, poisson=args.poisson)
    sc.finalize()
    rec = Recorder(sc)
    print(f"[scene] elastic cube {len(pts)} particles, edge {args.edge} m, E {args.young:.3g} nu {args.poisson}, dx {dx}", flush=True)
    t0 = time.time()
    for k in range(args.steps + 1):
        P = sc.eng.get_liquid_particles()
        rec.record(k * args.dt, 0, "drop", liquid_pts=P)
        if not np.isfinite(P).all():
            raise RuntimeError("non-finite particles")
        if k == args.steps:
            break
        ts = time.time(); sc.step()
        P = sc.eng.get_liquid_particles()
        ext = P.max(0) - P.min(0)
        print(f"step {k+1}/{args.steps} t={(k+1)*args.dt:.3f} wall {time.time()-ts:.2f}s | y [{P[:,1].min():.4f},{P[:,1].max():.4f}] extent {ext[0]:.3f} {ext[1]:.3f} {ext[2]:.3f}", flush=True)
    meta = dict(example="elastic_cube", dt=args.dt, fps=1.0 / args.dt, n_liquid=len(pts), particle_spacing=args.spacing, liquid_dx=dx,
                young=args.young, poisson=args.poisson, wall_min=(time.time() - t0) / 60)
    rec.save(osp.join(args.out, "recording.npz"), meta)
    print(f"[done] -> {osp.join(args.out, 'recording.npz')}")


if __name__ == "__main__":
    main()

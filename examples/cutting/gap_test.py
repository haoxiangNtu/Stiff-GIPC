#!/usr/bin/env python3
"""Two blocks of the cucumber material, one dropped onto the other, in ONE MPM particle body (like the pieces after a cut):
at what distance does the top block come to rest?  (2026-11-13, user: the cut slices seem to collide while still visibly apart.)

Within one MPM body two pieces touch only through the background grid: their particles share grid nodes (quadratic B-spline
transfer, support 1.5 dx around a node) as soon as they are less than ~3 dx apart, so the grid should hold the top block up
at a distance of the order of dx, not at one particle spacing.  Run at two spacings: if the rest distance scales with dx, that
is the cause.  The bottom block's distance to the board (IPC barrier, dhat 0.5 mm) is printed for comparison.

    PYTHONPATH=. STIFFGIPC_NATIVE_DIR=build_a800_cut12 LIQ_EL_LIN=1 python examples/cutting/gap_test.py --spacing 0.00075 --out out/cutting/gap_075
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
import time

import numpy as np

os.environ.setdefault("LIQ_EL_LIN", "1")
HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(HERE, "..", "robotgym"))
from common import Scene  # noqa: E402

BOARD_TOP = 0.03


def block(lo, hi, s, rng):
    ax = [np.arange(lo[k] + 0.5 * s, hi[k] - 0.49 * s, s) for k in range(3)]
    g = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    return g + rng.uniform(-0.15 * s, 0.15 * s, g.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--spacing", type=float, default=0.00075)
    ap.add_argument("--drop", type=float, default=0.004, help="initial gap between the blocks (m)")
    ap.add_argument("--size", type=float, nargs=3, default=(0.02, 0.008, 0.02), help="block size x y z (m)")
    ap.add_argument("--time", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=0.0025)
    ap.add_argument("--young", type=float, default=2e6)
    ap.add_argument("--pieces", action="store_true", help="2026-11-13: give the two blocks piece ids (Engine.set_liquid_pieces): a grid node "
                    "both reference is split unless their particles are within --contact seed spacings (needs LIQ_CUT_FIELDS=2)")
    ap.add_argument("--contact", type=float, default=1.0, help="piece contact distance in seed spacings")
    ap.add_argument("--vx", type=float, default=0.0, help="initial sideways speed of the top block (m/s): does it slide off / separate again?")
    args = ap.parse_args()
    if args.pieces:
        os.environ["LIQ_CUT_FIELDS"] = "2"
    s = args.spacing
    rng = np.random.default_rng(0)
    sx, sy, sz = args.size
    c = np.array([0.50, 0.0, 0.0])
    A = block((c[0] - sx / 2, BOARD_TOP, -sz / 2), (c[0] + sx / 2, BOARD_TOP + sy, sz / 2), s, rng)
    yb = BOARD_TOP + sy + args.drop
    B = block((c[0] - sx / 2, yb, -sz / 2), (c[0] + sx / 2, yb + sy, sz / 2), s, rng)
    pts = np.vstack([A, B]); lab = np.r_[np.zeros(len(A), int), np.ones(len(B), int)]
    print(f"[gap] spacing {s*1e3:.2f} mm, dx {2*s*1e3:.2f} mm, blocks {len(A)} + {len(B)} particles, drop {args.drop*1e3:.1f} mm", flush=True)
    os.makedirs(args.out, exist_ok=True)
    sc = Scene(dict(dt=args.dt, preconditioner_type=0, newton_iter_cap=80, newton_tol=1e-3, pcg_tol=1e-5, absolute_dhat=5e-4,
                    friction_rate=0.4, gd_friction_rate=0.4, gravity=(0.0, -9.81, 0.0)), log_level=0)
    sc.add_box("board", (c[0], BOARD_TOP - 0.015, 0.0), (0.12, 0.03, 0.12), fixed=True, mu=0.4, cell=0.01)
    lo = np.array([c[0] - 0.04, 0.0, -0.04]); hi = np.array([c[0] + 0.04, BOARD_TOP + 0.05, 0.04])
    sc.eng.add_liquid(pts, 2.0 * s, lo, hi, rho=960.0, kappa=1e5, particle_volume=s ** 3, node_pool=int(8 * len(pts)), push_apart=0.0,
                      j_mode=0, ground_mu=0.4, model=1, young=args.young, poisson=0.4, plastic_stretch=0.0, plastic_compress=0.0,
                      plastic_yield=0.0, plastic_damage=0.0, plastic_damage_residual=0.1, mu=0.4, plastic_damage_strain=0.0)
    sc.finalize()
    if args.pieces:
        sc.eng.set_liquid_pieces(lab, args.contact)
    n = int(round(args.time / args.dt)); every = max(1, int(round(0.02 / args.dt)))
    rows = []
    t0 = time.time()
    for k in range(n + 1):
        if k % every == 0:
            P = np.asarray(sc.eng.get_liquid_particles(), np.float64)
            Pa, Pb = P[lab == 0], P[lab == 1]
            topA = np.percentile(Pa[:, 1], 99.5); botB = np.percentile(Pb[:, 1], 0.5); botA = np.percentile(Pa[:, 1], 0.5)
            from scipy.spatial import cKDTree
            d, _ = cKDTree(Pa).query(Pb, k=1)
            rows.append(dict(t=k * args.dt, gap_layers=float(botB - topA), dmin=float(d.min()), d_p1=float(np.percentile(d, 1)),
                             vB=float(np.mean(Pb[:, 1])), board=float(botA - BOARD_TOP)))
            r = rows[-1]
            print(f"  t {r['t']:5.2f} s  B-bottom minus A-top layer {r['gap_layers']*1e3:6.3f} mm  closest pair {r['dmin']*1e3:6.3f} mm (p1 {r['d_p1']*1e3:6.3f})"
                  f"  A-bottom layer above the board {r['board']*1e3:6.3f} mm  B centroid y {r['vB']*1e3:7.3f} mm", flush=True)
        if k < n:
            sc.step()
    json.dump(dict(spacing=s, dx=2 * s, pieces=bool(args.pieces), contact=args.contact, rows=rows, wall_s=time.time() - t0), open(osp.join(args.out, "gap.json"), "w"), indent=0)
    last = rows[-5:]
    print(f"[gap] REST: layer distance {np.mean([r['gap_layers'] for r in last])*1e3:.3f} mm, closest pair {np.mean([r['dmin'] for r in last])*1e3:.3f} mm "
          f"(touching = one spacing {s*1e3:.2f} mm); A above the board {np.mean([r['board'] for r in last])*1e3:.3f} mm", flush=True)
    print(f"[gap] done in {time.time()-t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()

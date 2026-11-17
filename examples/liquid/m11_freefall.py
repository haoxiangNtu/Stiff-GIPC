"""M11: free fall of small drops released at rest in the air -> every drop should accelerate at g.

Cubes of n^3 particles (n = 1, 2, 4, 8; spacing dx/2, like every liquid scene) are released at rest, far from any wall,
and fall for 0.2 s. The centroid acceleration is fitted from y(t) over t >= 0.05 s. A grid node whose P2G mass is below
the conditioning floor (mass_floor * rho * dx^3) carries extra inertia; whether that floor is also weighed is
LIQ_FLOOR_GRAVITY (1: weighed, 0: inertia only), so the smallest drops are where the two settings differ.
    python examples/liquid/m11_freefall.py --out out/liquid_m11
"""
import argparse, json, os, sys, time
if __name__ == "__main__": import os as _os; _os.environ.setdefault("LIQ_UG", "0")   # main-path example, when run as a script: pinned to the main liquid path (the engine default is the UG mode since 2026-11-18)
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from stiff_physics.engine import Engine, Config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=0.005)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--duration", type=float, default=0.2)
    ap.add_argument("--sizes", default="1,2,4,8", help="drop edge lengths in particles")
    ap.add_argument("--kappa", type=float, default=1e5)
    ap.add_argument("--push", type=float, default=0.9)
    ap.add_argument("--jmode", type=int, default=0)
    ap.add_argument("--mass-floor", type=float, default=0.05)
    ap.add_argument("--out", default="out/liquid_m11")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dx = args.dx; hs = dx / 2.0
    sizes = [int(s) for s in args.sizes.split(",")]
    drops, ids, start = [], [], 0
    for i, n in enumerate(sizes):
        g = (np.arange(n) - (n - 1) / 2.0) * hs
        X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
        p = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1) + np.array([0.10 + 0.08 * i, 0.25, 0.20])
        drops.append(p); ids.append(np.arange(start, start + len(p))); start += len(p)
    pts = np.vstack(drops)
    cfg = Config(dt=args.dt, preconditioner_type=0, absolute_dhat=5e-4, newton_tol=1e-4, pcg_tol=1e-6, friction_rate=0.0,
                 gd_friction_rate=0.0, ground_offset=0.0, energy_rel_tol=0.0, newton_iter_cap=200, pcg_max_iter_ratio=0.3,
                 gravity=(0.0, -9.81, 0.0))
    eng = Engine(cfg); eng.set_log_level(0)
    # a small fixed box outside the liquid domain: the collision BVH needs at least one surface
    bv = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float) * 0.02 + np.array([0.9, 0.9, 0.9])
    bf = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    eng.load_mesh_from_data(bv, bf, verts_per_face=3, dimensions=3, body_type="ABD", transform=np.eye(4), young_modulus=1e8, boundary_type="Fixed")
    lo = np.array([0.0, -0.02, 0.10]); hi = np.array([0.10 + 0.08 * len(sizes), 0.32, 0.30])
    eng.add_liquid(pts, dx, lo, hi, rho=1000.0, kappa=args.kappa, particle_volume=hs ** 3, push_apart=args.push, j_mode=args.jmode,
                   mass_floor=args.mass_floor)
    eng.finalize()
    fg = os.environ.get("LIQ_FLOOR_GRAVITY", "1 (default)")
    print(f"[m11] {len(pts)} particles in drops of {[n ** 3 for n in sizes]}, dx {dx*1e3:.1f} mm, dt {args.dt*1e3:.1f} ms, "
          f"mass floor {args.mass_floor}, LIQ_FLOOR_GRAVITY={fg}", flush=True)
    ts, ys = [], []
    t = 0.0; wall0 = time.time()
    while t < args.duration + 1e-12:
        x = eng.get_liquid_particles()
        ts.append(t); ys.append([float(x[i, 1].mean()) for i in ids])
        if t >= args.duration - 1e-12: break
        eng.step(); t += args.dt
    ts, ys = np.array(ts), np.array(ys)
    sel = ts >= 0.05 - 1e-12
    res = {}
    for k, n in enumerate(sizes):
        c = np.polyfit(ts[sel], ys[sel, k], 2); a = 2 * c[0]
        v_end = (ys[-1, k] - ys[-2, k]) / (ts[-1] - ts[-2])
        res[n ** 3] = dict(a=a, fall_mm=(ys[0, k] - ys[-1, k]) * 1e3, v_end=v_end)
        print(f"[m11] drop of {n**3:4d} particles: fitted a_y {a:7.2f} m/s^2 ({a/-9.81*100:5.1f} % of g), fell {res[n**3]['fall_mm']:6.1f} mm "
              f"in {ts[-1]:.2f} s (free fall {0.5*9.81*ts[-1]**2*1e3:.1f} mm), v_end {v_end:6.3f} m/s", flush=True)
    print(f"[m11] wall {time.time()-wall0:.0f} s", flush=True)
    json.dump(dict(floor_gravity=fg, dt=args.dt, dx=dx, mass_floor=args.mass_floor, drops=res), open(os.path.join(args.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

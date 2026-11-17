"""m10: isolated spill test for the bottle demos. A 30 ml water column is released against a vertical fixed plastic
plate above the ground plane; it drains down the plate (film) and spreads on the table (puddle). Prints, every 0.1 s,
the film's count and median vertical speed and the puddle's count, spread radius and speed, so wall/ground treatments
(rest drag, ghost penalty mode, surface tension, ground friction) can be compared with one number each.

    LIQ_WFC=1 LIQ_GP=1 LIQ_GP_FULL=6 LIQ_REST_K=20 LIQ_REST_KV=0.003 PYTHONPATH=. python examples/liquid/m10_spill.py --tag base
"""
import argparse, os, os.path as osp, sys, time
if __name__ == "__main__": import os as _os; _os.environ.setdefault("LIQ_UG", "0")   # main-path example, when run as a script: pinned to the main liquid path (the engine default is the UG mode since 2026-11-18)
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "..", "robotgym"))
from common import Scene  # noqa: E402

PLATE_X = 0.30   # plate face at x = PLATE_X (the water sits on the +x side)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base"); ap.add_argument("--out", default="out/audit_local/m10")
    ap.add_argument("--dt", type=float, default=0.005); ap.add_argument("--duration", type=float, default=2.5)
    ap.add_argument("--dx", type=float, default=0.005); ap.add_argument("--ground-mu", type=float, default=0.3)
    ap.add_argument("--column", type=float, nargs=3, default=(0.02, 0.075, 0.02), help="water column size x y z (m)")
    ap.add_argument("--gap", type=float, default=0.002, help="initial gap between the column and the plate")
    ap.add_argument("--y0", type=float, default=0.10, help="column bottom height")
    ap.add_argument("--log-level", type=int, default=0)
    ap.add_argument("--mesh-floor", type=float, default=0.0, help="> 0: a fixed box table whose top is this high above the ground plane (the water rests on a mesh, which gets the grid wall constraint, instead of the analytic plane)")
    ap.add_argument("--back-pool", type=float, default=0.0, help="depth (m) of a resting pool on the far side of the plate (0 = none); reproduces the thin-wall case of the bottle")
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    sc = Scene(dict(dt=args.dt, preconditioner_type=0, newton_iter_cap=200), log_level=args.log_level)
    # vertical "plastic" plate: 3 mm thick fixed box, 20 cm wide, 25 cm tall, standing on the table
    sc.add_box("plate", (PLATE_X - 0.0015, 0.125, 0.0), (0.003, 0.25, 0.20), fixed=True, mu=0.5, cell=0.02)
    if args.back_pool > 0:   # a basin behind the plate: back wall + two side walls (overlapping the plate; exclusions keep the overlaps out of the barrier)
        sc.add_box("back", (PLATE_X - 0.0815, 0.075, 0.0), (0.003, 0.15, 0.20), fixed=True, mu=0.5, cell=0.02)
        sc.add_box("side0", (PLATE_X - 0.0415, 0.075, 0.1015), (0.09, 0.15, 0.003), fixed=True, mu=0.5, cell=0.02)
        sc.add_box("side1", (PLATE_X - 0.0415, 0.075, -0.1015), (0.09, 0.15, 0.003), fixed=True, mu=0.5, cell=0.02)
        ids = {n: sc.bodies[n].body_id for n in ("plate", "back", "side0", "side1")}
        for a in ("side0", "side1"):
            for b in ("plate", "back"):
                sc.eng.add_collision_exclusion(ids[a], ids[b])
    if args.mesh_floor > 0:   # the walls stand 0..0.15 m, the floor box overlaps their feet: excluded pairs
        sc.add_box("floor", (PLATE_X + 0.165, args.mesh_floor - 0.0055, 0.0), (0.57, 0.011, 0.50), fixed=True, mu=0.5, cell=0.02)
        for b in ("plate", "back", "side0", "side1"):
            if b in sc.bodies: sc.eng.add_collision_exclusion(sc.bodies["floor"].body_id, sc.bodies[b].body_id)
    hs = args.dx / 2; m = 0.3 * args.dx; n_col = 0
    sx, sy, sz = args.column
    ax = np.arange(PLATE_X + args.gap + m, PLATE_X + args.gap + sx, hs); ay = np.arange(args.y0, args.y0 + sy, hs); az = np.arange(-sz / 2, sz / 2, hs)
    gx, gy, gz = np.meshgrid(ax, ay, az, indexing="ij"); pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)
    pts += np.random.default_rng(0).uniform(-0.3 * hs, 0.3 * hs, pts.shape)
    if args.back_pool > 0:
        bx = np.arange(PLATE_X - 0.078, PLATE_X - 0.003 - m, hs); by = np.arange(0.006, args.back_pool, hs); bz = np.arange(-0.05, 0.05, hs)
        gx, gy, gz = np.meshgrid(bx, by, bz, indexing="ij"); pool = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)
        pool += np.random.default_rng(1).uniform(-0.3 * hs, 0.3 * hs, pool.shape)
        n_col = len(pts); pts = np.vstack([pts, pool]); print(f"[m10] back pool {len(pool)} particles = {len(pool)*hs**3*1e6:.0f} ml", flush=True)
    lo = np.array([PLATE_X - 0.12, -0.02, -0.25]); hi = np.array([PLATE_X + 0.45, 0.30, 0.25])
    sc.eng.add_liquid(pts, args.dx, lo, hi, rho=1000.0, kappa=1e5, particle_volume=hs ** 3, node_pool=60000,
                      push_apart=0.9, j_mode=0, stick_h=0.0, ground_mu=args.ground_mu)
    print(f"[m10] {len(pts)} particles = {len(pts)*hs**3*1e6:.1f} ml, ground_mu {args.ground_mu}", flush=True)
    sc.finalize()
    n_steps = int(round(args.duration / args.dt)); every = int(round(0.1 / args.dt))
    prev = sc.eng.get_liquid_particles().copy(); t0 = time.time(); rows = []; snaps = []
    print("  t    | film: n  v_y_med(mm/s) y_med | table: n  r90(mm) v_med v_p90 (mm/s) 1nn p50 (mm) | wall(min)")
    for k in range(1, n_steps + 1):
        sc.step()
        if k % every == 0:
            P = sc.eng.get_liquid_particles(); V = (P - prev) / (every * args.dt); prev = P.copy()
            front = P[:, 0] > PLATE_X - 0.0015
            film = front & (P[:, 0] < PLATE_X + 0.008) & (P[:, 1] > 0.015)
            table = front & (P[:, 1] < 0.010)
            crossed = int((~front).sum()) - (len(P) - n_col if args.back_pool > 0 else 0)
            r = np.hypot(P[:, 0] - PLATE_X, P[:, 2]); sp = np.linalg.norm(V, axis=1) * 1e3
            row = dict(t=k * args.dt, film_n=int(film.sum()), film_vy=float(np.median(V[film, 1]) * 1e3) if film.any() else 0.0,
                       film_y=float(np.median(P[film, 1])) if film.any() else 0.0, table_n=int(table.sum()),
                       r90=float(np.percentile(r[table], 90) * 1e3) if table.any() else 0.0,
                       v_med=float(np.median(sp[table])) if table.any() else 0.0, v_p90=float(np.percentile(sp[table], 90)) if table.any() else 0.0, crossed=crossed,
                       nn1=float(np.median(cKDTree(P[table]).query(P[table], k=2)[0][:, 1]) * 1e3) if table.sum() > 2 else 0.0)   # puddle packing (seed spacing dx/2)
            rows.append(row)
            if k % (5 * every) == 0: snaps.append(P.copy())
            print(f"  {row['t']:4.1f} | {row['film_n']:5d} {row['film_vy']:9.1f} {row['film_y']:6.3f} | {row['table_n']:5d} {row['r90']:7.1f} {row['v_med']:6.1f} {row['v_p90']:6.1f} {row['nn1']:5.2f} | {(time.time()-t0)/60:4.1f} | crossed {row['crossed']}", flush=True)
    if args.back_pool > 0:   # bulk compression check: 6-nn spacing in the pool at the end (seeded 2.5 mm; cohesion must not shrink it)
        Pf = sc.eng.get_liquid_particles(); pool = Pf[n_col:]; core = pool[(pool[:, 1] > 0.015) & (pool[:, 1] < 0.045) & (np.abs(pool[:, 2]) < 0.03)]
        dd, _ = cKDTree(core).query(core, k=7); print(f"[m10 {args.tag}] pool bulk 6-nn spacing at the end: {np.median(dd[:, 1:].mean(1))*1e3:.2f} mm (seed {hs*1e3:.2f}); pool level p99 {np.percentile(pool[:, 1], 99):.3f} m", flush=True)
    np.savez(osp.join(args.out, f"{args.tag}.npz"), rows=np.array([[r[k] for k in ("t", "film_n", "film_vy", "film_y", "table_n", "r90", "v_med", "v_p90")] for r in rows]),
             final=sc.eng.get_liquid_particles(), snaps=np.array(snaps))
    print(f"[m10 {args.tag}] done {(time.time()-t0)/60:.1f} min", flush=True)

if __name__ == "__main__":
    main()

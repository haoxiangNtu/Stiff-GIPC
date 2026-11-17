"""M1 bring-up test for the in-engine MPM Lite liquid: a 10 cm water column
collapses on the ground plane. No solid contact yet (nodes see the ground only).

    python examples/liquid/m1_dambreak.py --steps 200 --out out/liquid_m1

Checks: no ground penetration (min y >= 0), the column spreads, J stays near 1,
nearest-neighbour spacing does not collapse (no clumping), Newton converges.
"""
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from stiff_physics.engine import Engine, Config  # noqa: E402


def cube_mesh(c, s):
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float) * s + (np.asarray(c) - s / 2)
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], np.int32)
    return v, f


def nn_spacing(x, k=1):
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return float("nan"), float("nan")
    d, _ = cKDTree(x).query(x, k=2)
    d = d[:, 1]
    return float(np.median(d)), float(np.percentile(d, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--dx", type=float, default=0.01)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--kappa", type=float, default=1e5)
    ap.add_argument("--ppc", type=int, default=8)
    ap.add_argument("--size", type=float, default=0.10, help="water column edge (m)")
    ap.add_argument("--newton-tol", type=float, default=1e-3)
    ap.add_argument("--mass-floor", type=float, default=0.05)
    ap.add_argument("--apic", type=int, default=1)
    ap.add_argument("--node-pool", type=int, default=0)
    ap.add_argument("--push", type=float, default=0.9)
    ap.add_argument("--jmode", type=int, default=3)
    ap.add_argument("--newton-cap", type=int, default=1000)
    ap.add_argument("--out", default="out/liquid_m1")
    ap.add_argument("--snap", type=str, default="0,40,100,200")
    ap.add_argument("--log-level", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dx = args.dx
    lo = np.array([0.0, 0.001, 0.0])
    hi = np.array([0.40, 0.20, 0.20])
    cfg = Config(dt=args.dt, preconditioner_type=0, absolute_dhat=5e-4, newton_tol=args.newton_tol, newton_iter_cap=args.newton_cap,
                 friction_rate=0.0, gd_friction_rate=0.0, ground_offset=0.0)
    eng = Engine(cfg)
    if hasattr(eng, "set_log_level"):
        eng.set_log_level(args.log_level)

    # one far-away fixed box so the scene has an ordinary solid body
    v, f = cube_mesh([0.60, 0.06, 0.10], 0.06)
    eng.load_mesh_from_data(v, f, 3, 3, "ABD", np.eye(4), 1e7, "Fixed")

    # water column in the middle of the domain, ppc particles per cell on a sub-lattice
    n_sub = int(round(args.ppc ** (1.0 / 3.0)))
    h = dx / n_sub
    s = args.size
    c0 = np.array([0.20, lo[1] + 0.0 * dx, 0.10])
    ax = np.arange(0, s, h) + h / 2
    gx, gy, gz = np.meshgrid(ax, ax, ax, indexing="ij")
    pts = np.stack([gx.ravel() + c0[0] - s / 2, gy.ravel() + lo[1] + 1.0 * dx, gz.ravel() + c0[2] - s / 2], 1)
    pv = dx ** 3 / (n_sub ** 3)
    eng.add_liquid(pts, dx, lo, hi, rho=1000.0, kappa=args.kappa, particle_volume=pv,
                   node_pool=args.node_pool, mass_floor=args.mass_floor, apic=bool(args.apic),
                   push_apart=args.push, j_mode=args.jmode)
    eng.finalize()
    st = eng.get_liquid_stats()
    print(f"[m1] particles {len(pts)} mass {1000*pv*len(pts):.3f} kg  stats {st}", flush=True)

    snaps = set(int(x) for x in args.snap.split(",") if x)
    rec = []
    t0 = time.time()
    nt_prev = 0
    x = eng.get_liquid_particles()
    med0, p1_0 = nn_spacing(x)
    for step in range(args.steps + 1):
        if step in snaps:
            np.save(os.path.join(args.out, f"pts_{step:04d}.npy"), eng.get_liquid_particles())
        if step == args.steps:
            break
        ts = time.time()
        eng.step()
        wall = time.time() - ts
        x = eng.get_liquid_particles()
        J = eng.get_liquid_J()
        vel = eng.get_liquid_velocities()
        st = eng.get_liquid_stats()
        nt = eng._engine.get_total_newton_iters() if hasattr(eng._engine, "get_total_newton_iters") else 0
        if not np.all(np.isfinite(x)):
            print(f"[m1] NaN in particle positions at step {step}", flush=True)
            sys.exit(2)
        if step % 10 == 0 or step < 5:
            med, p1 = nn_spacing(x)
            spread = x[:, 0].max() - x[:, 0].min()
            print(f"t={(step+1)*args.dt:6.3f}s  miny {x[:,1].min()*1000:7.2f} mm  maxy {x[:,1].max()*1000:6.1f} mm  spread_x {spread*1000:6.1f} mm"
                  f"  J mean {J.mean():.4f} min {J.min():.3f}  NN med {med*1000:.2f} p1 {p1*1000:.2f} mm (seed {med0*1000:.2f})"
                  f"  |v|max {np.linalg.norm(vel,axis=1).max():.3f}  cells {st['active_cells']} nodes {st['active_nodes']}/{st['node_pool']}"
                  f"  newton {nt-nt_prev:4d}  {wall*1000:7.1f} ms", flush=True)
        rec.append([step, x[:, 1].min(), x[:, 1].max(), J.mean(), J.min(), nt - nt_prev, wall])
        nt_prev = nt
    np.save(os.path.join(args.out, "record.npy"), np.array(rec))
    print(f"[m1] done {args.steps} steps in {time.time()-t0:.1f} s", flush=True)

    # side-view snapshots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        files = sorted(fn for fn in os.listdir(args.out) if fn.startswith("pts_"))
        fig, axs = plt.subplots(1, len(files), figsize=(4 * len(files), 3.2))
        axs = np.atleast_1d(axs)
        for a, fn in zip(axs, files):
            p = np.load(os.path.join(args.out, fn))
            a.scatter(p[:, 0] * 1000, p[:, 1] * 1000, s=1, c=p[:, 2])
            a.set_xlim(0, 400); a.set_ylim(-5, 150); a.set_aspect("equal"); a.set_title(fn)
            a.axhline(0, color="k", lw=0.5)
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "snapshots.png"), dpi=110)
        print("[m1] snapshots.png written", flush=True)
    except Exception as e:  # pragma: no cover
        print("[m1] plot skipped:", e)


if __name__ == "__main__":
    main()

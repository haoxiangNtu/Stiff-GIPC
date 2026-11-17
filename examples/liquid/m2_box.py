"""M2 test: water inside a static open box (ABD, Fixed) — particle-triangle IPC
barrier chain-ruled onto the grid nodes. Checks containment (no particle
outside the cavity), settling height, velocity decay, Newton cost.

    python examples/liquid/m2_box.py --steps 200 --out out/liquid_m2
"""
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from stiff_physics.engine import Engine, Config  # noqa: E402


def nn_spacing(x):
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return float('nan'), float('nan')
    d, _ = cKDTree(x).query(x, k=2)
    d = d[:, 1]
    return float(np.median(d)), float(np.percentile(d, 1))


def cuboid(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    v = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
                  [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]])
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], np.int32)
    return v, f


def open_box(center_xz, inner, wall, floor_y, height, floor_th):
    """5 cuboids: floor + 4 walls, inner cavity `inner` x `inner` (m)."""
    cx, cz = center_xz
    h = inner / 2
    parts = []
    parts.append(cuboid([cx - h - wall, floor_y - floor_th, cz - h - wall], [cx + h + wall, floor_y, cz + h + wall]))
    parts.append(cuboid([cx - h - wall, floor_y, cz - h - wall], [cx - h, floor_y + height, cz + h + wall]))
    parts.append(cuboid([cx + h, floor_y, cz - h - wall], [cx + h + wall, floor_y + height, cz + h + wall]))
    parts.append(cuboid([cx - h, floor_y, cz - h - wall], [cx + h, floor_y + height, cz - h]))
    parts.append(cuboid([cx - h, floor_y, cz + h], [cx + h, floor_y + height, cz + h + wall]))
    V, F, off = [], [], 0
    for v, f in parts:
        V.append(v); F.append(f + off); off += len(v)
    return np.vstack(V), np.vstack(F)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--dx", type=float, default=0.01)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--kappa", type=float, default=1e5)
    ap.add_argument("--ppc", type=int, default=8)
    ap.add_argument("--inner", type=float, default=0.12)
    ap.add_argument("--height", type=float, default=0.16, help="wall height above the floor (m); the --lid sits on top of it")
    ap.add_argument("--water", type=float, nargs=3, default=[0.10, 0.08, 0.10])
    ap.add_argument("--gap", type=float, default=0.002, help="initial water-wall gap (m)")
    ap.add_argument("--newton-tol", type=float, default=1e-3)
    ap.add_argument("--push", type=float, default=0.9)
    ap.add_argument("--jmode", type=int, default=3)
    ap.add_argument("--dhat", type=float, default=5e-4)
    ap.add_argument("--out", default="out/liquid_m2")
    ap.add_argument("--snap", type=str, default="0,20,60,120,200")
    ap.add_argument("--log-level", type=int, default=0)
    ap.add_argument("--jitter", type=float, default=0.0, help="seed jitter as a fraction of the spacing")
    ap.add_argument("--mass-floor", type=float, default=0.05, help="node mass floor as a fraction of rho*dx^3")
    ap.add_argument("--far-box", type=float, default=0.0, help="place an extra fixed box this far away (enlarges the scene bbox / Newton threshold)")
    ap.add_argument("--pcg-tol", type=float, default=1e-4)
    ap.add_argument("--newton-cap", type=int, default=1000)
    ap.add_argument("--apic", type=float, default=1.0, help="APIC ratio of the transfer (0 = PIC)")
    ap.add_argument("--dump-every", type=int, default=0, help="save frames/f_<step>.npz (x, v) every N steps (video)")
    ap.add_argument("--lid", action="store_true", help="close the box with a fixed lid 1 cm thick at the top of the walls (no free surface when --water fills the cavity)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dx = args.dx
    floor_y = 0.02
    cxz = (0.20, 0.20)
    wall = 0.005
    V, F = open_box(cxz, args.inner, wall, floor_y, args.height, 0.01)
    lo = np.array([0.0, 0.001, 0.0])
    hi = np.array([0.40, 0.25, 0.40])
    cfg = Config(dt=args.dt, preconditioner_type=0, absolute_dhat=args.dhat, newton_tol=args.newton_tol,
                 friction_rate=0.0, gd_friction_rate=0.0, ground_offset=0.0, pcg_tol=args.pcg_tol,
                 newton_iter_cap=args.newton_cap)
    eng = Engine(cfg)
    eng.set_log_level(args.log_level)
    eng.load_mesh_from_data(V, F, 3, 3, "ABD", np.eye(4), 1e7, "Fixed")
    if args.lid:
        hh = args.inner / 2 + wall
        vl, fl = cuboid([cxz[0] - hh, floor_y + args.height, cxz[1] - hh], [cxz[0] + hh, floor_y + args.height + 0.01, cxz[1] + hh])
        eng.load_mesh_from_data(vl, fl, 3, 3, "ABD", np.eye(4), 1e7, "Fixed")
    if args.far_box > 0:
        vf, ff = cuboid([args.far_box, 0.05, args.far_box], [args.far_box + 0.05, 0.10, args.far_box + 0.05])
        eng.load_mesh_from_data(vf, ff, 3, 3, "ABD", np.eye(4), 1e7, "Fixed")

    n_sub = int(round(args.ppc ** (1.0 / 3.0)))
    h = dx / n_sub
    wx, wy, wz = args.water
    ax = np.arange(0, wx, h) + h / 2
    ay = np.arange(0, wy, h) + h / 2
    az = np.arange(0, wz, h) + h / 2
    gx, gy, gz = np.meshgrid(ax, ay, az, indexing="ij")
    pts = np.stack([gx.ravel() + cxz[0] - wx / 2, gy.ravel() + floor_y + args.gap, gz.ravel() + cxz[1] - wz / 2], 1)
    if args.jitter > 0:
        pts = pts + np.random.default_rng(0).uniform(-args.jitter * h, args.jitter * h, pts.shape)
    pv = dx ** 3 / (n_sub ** 3)
    eng.add_liquid(pts, dx, lo, hi, rho=1000.0, kappa=args.kappa, particle_volume=pv, push_apart=args.push, j_mode=args.jmode,
                   mass_floor=args.mass_floor, apic_ratio=args.apic)
    eng.finalize()
    st = eng.get_liquid_stats()
    vol = wx * wy * wz
    print(f"[m2] particles {len(pts)} water {vol*1e3:.2f} L, cavity {args.inner*100:.0f} cm -> rest depth {vol/args.inner**2*1000:.1f} mm; stats {st}", flush=True)

    hin = args.inner / 2
    snaps = set(int(x) for x in args.snap.split(",") if x)
    t0 = time.time()
    nt_prev = 0
    for step in range(args.steps + 1):
        if step in snaps:
            np.save(os.path.join(args.out, f"pts_{step:04d}.npy"), eng.get_liquid_particles())
            np.save(os.path.join(args.out, f"vel_{step:04d}.npy"), eng.get_liquid_velocities())
        if args.dump_every and step % args.dump_every == 0:
            os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)
            np.savez(os.path.join(args.out, "frames", f"f_{step:05d}.npz"), x=eng.get_liquid_particles().astype(np.float32), v=eng.get_liquid_velocities().astype(np.float32), t=step * args.dt)
        if step == args.steps:
            break
        ts = time.time()
        eng.step()
        wall_t = time.time() - ts
        x = eng.get_liquid_particles()
        vel = eng.get_liquid_velocities()
        J = eng.get_liquid_J()
        nt = eng._engine.get_total_newton_iters()
        if not np.all(np.isfinite(x)):
            print(f"[m2] NaN at step {step}"); sys.exit(2)
        if step % 10 == 0 or step < 3:
            dxw = np.maximum(np.abs(x[:, 0] - cxz[0]), np.abs(x[:, 2] - cxz[1]))
            outside = np.sum((dxw > hin) | (x[:, 1] < floor_y))
            pen = max(0.0, (dxw.max() - hin) * 1000, (floor_y - x[:, 1].min()) * 1000)
            y = x[:, 1] - floor_y
            nnm, nn1 = nn_spacing(x)
            print(f"t={(step+1)*args.dt:6.3f}s outside {outside:5d} pen {pen:5.2f} mm  depth p50 {np.percentile(y,50)*1000:5.1f} p95 {np.percentile(y,95)*1000:5.1f} max {y.max()*1000:5.1f} mm  NN med {nnm*1000:.2f} p1 {nn1*1000:.2f} mm"
                  f"  gap-wall min {(hin - dxw.max())*1000:5.2f} mm  J {J.mean():.3f}/{J.min():.3f}  |v|max {np.linalg.norm(vel,axis=1).max():.3f} rms {np.sqrt((vel**2).sum(1).mean()):.3f}"
                  f"  cells {st['active_cells']} newton {nt-nt_prev:3d} {wall_t*1000:6.1f} ms", flush=True)
            st = eng.get_liquid_stats()
        nt_prev = nt
    print(f"[m2] done {args.steps} steps in {time.time()-t0:.1f} s", flush=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        files = sorted(fn for fn in os.listdir(args.out) if fn.startswith("pts_"))
        fig, axs = plt.subplots(1, len(files), figsize=(3.6 * len(files), 3.4))
        for a, fn in zip(np.atleast_1d(axs), files):
            p = np.load(os.path.join(args.out, fn))
            a.scatter(p[:, 0] * 1000, p[:, 1] * 1000, s=1, c=p[:, 2])
            for xw in (cxz[0] - hin, cxz[0] + hin):
                a.axvline(xw * 1000, color="k", lw=0.6)
            a.axhline(floor_y * 1000, color="k", lw=0.6)
            a.set_xlim(100, 300); a.set_ylim(0, 180); a.set_aspect("equal"); a.set_title(fn)
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "snapshots.png"), dpi=110)
        print("[m2] snapshots.png written", flush=True)
    except Exception as e:
        print("[m2] plot skipped:", e)


if __name__ == "__main__":
    main()

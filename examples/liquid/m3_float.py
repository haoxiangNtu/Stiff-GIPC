"""M3 two-way test: a light free ABD cube (density 500 kg/m^3) dropped into the
water box of m2_box.py must float with ~half its volume submerged; the water
surface must rise by the displaced volume.

    python examples/liquid/m3_float.py --steps 400 --out out/liquid_m3
"""
import argparse, os, sys, time
if __name__ == "__main__": import os as _os; _os.environ.setdefault("LIQ_UG", "0")   # main-path example, when run as a script: pinned to the main liquid path (the engine default is the UG mode since 2026-11-18)
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from stiff_physics.engine import Engine, Config  # noqa: E402
from m2_box import cuboid, open_box, nn_spacing  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--dx", type=float, default=0.01)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--kappa", type=float, default=1e5)
    ap.add_argument("--ppc", type=int, default=8)
    ap.add_argument("--inner", type=float, default=0.12)
    ap.add_argument("--water", type=float, nargs=3, default=[0.10, 0.08, 0.10])
    ap.add_argument("--cube", type=float, default=0.04)
    ap.add_argument("--cube-density", type=float, default=500.0)
    ap.add_argument("--drop-height", type=float, default=0.02, help="cube bottom above the (rest) water surface")
    ap.add_argument("--newton-tol", type=float, default=1e-3)
    ap.add_argument("--dhat", type=float, default=5e-4)
    ap.add_argument("--push", type=float, default=0.9)
    ap.add_argument("--jmode", type=int, default=3)
    ap.add_argument("--cube-fixed", action="store_true", help="cube is a Fixed body placed half-submerged (coupling isolation)")
    ap.add_argument("--pcg-tol", type=float, default=1e-4)
    ap.add_argument("--newton-cap", type=int, default=1000)
    ap.add_argument("--out", default="out/liquid_m3")
    ap.add_argument("--snap", type=str, default="0,50,150,300,400")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dx = args.dx
    floor_y, cxz, wall = 0.02, (0.20, 0.20), 0.005
    V, F = open_box(cxz, args.inner, wall, floor_y, 0.20, 0.01)
    lo = np.array([0.0, 0.001, 0.0]); hi = np.array([0.40, 0.30, 0.40])
    cfg = Config(dt=args.dt, preconditioner_type=0, absolute_dhat=args.dhat, newton_tol=args.newton_tol,
                 friction_rate=0.0, gd_friction_rate=0.0, ground_offset=0.0, pcg_tol=args.pcg_tol,
                 newton_iter_cap=args.newton_cap)
    eng = Engine(cfg)
    eng.set_log_level(0)
    eng.load_mesh_from_data(V, F, 3, 3, "ABD", np.eye(4), 1e7, "Fixed")
    wx, wy, wz = args.water
    depth = wx * wy * wz / args.inner ** 2
    surf = floor_y + depth
    c = args.cube
    y_bot = surf - c / 2 if args.cube_fixed else surf + args.drop_height
    cv, cf = cuboid([cxz[0] - c / 2, y_bot, cxz[1] - c / 2], [cxz[0] + c / 2, y_bot + c, cxz[1] + c / 2])
    eng.load_mesh_from_data(cv, cf, 3, 3, "ABD", np.eye(4), 1e7, "Fixed" if args.cube_fixed else "Free")
    recs = eng._engine.get_all_load_records() if hasattr(eng._engine, "get_all_load_records") else eng._engine.get_load_records()
    cube_rec = recs[-1]
    eng._engine.set_abd_body_density(int(cube_rec.body_offset), float(args.cube_density))

    n_sub = int(round(args.ppc ** (1.0 / 3.0))); h = dx / n_sub
    ax = np.arange(0, wx, h) + h / 2; ay = np.arange(0, wy, h) + h / 2; az = np.arange(0, wz, h) + h / 2
    gx, gy, gz = np.meshgrid(ax, ay, az, indexing="ij")
    pts = np.stack([gx.ravel() + cxz[0] - wx / 2, gy.ravel() + floor_y + 0.002, gz.ravel() + cxz[1] - wz / 2], 1)
    if args.cube_fixed:
        inside = (np.abs(pts[:, 0] - cxz[0]) < c / 2 + 0.002) & (np.abs(pts[:, 2] - cxz[1]) < c / 2 + 0.002) & (pts[:, 1] > y_bot - 0.002)
        pts = pts[~inside]
    eng.add_liquid(pts, dx, lo, hi, rho=1000.0, kappa=args.kappa, particle_volume=h ** 3, push_apart=args.push, j_mode=args.jmode)
    eng.finalize()
    v0, n0 = int(cube_rec.vertex_offset), int(cube_rec.vertex_count)
    print(f"[m3] water {wx*wy*wz*1e3:.2f} L rest depth {depth*1000:.1f} mm (surface y {surf*1000:.1f} mm); cube {c*100:.0f} cm rho {args.cube_density}"
          f" -> expected submerged fraction {args.cube_density/1000:.2f}, cube body {cube_rec.body_offset} verts [{v0},{v0+n0})", flush=True)
    snaps = set(int(x) for x in args.snap.split(",") if x)
    t0 = time.time(); nt_prev = 0; pc_prev = 0
    for step in range(args.steps + 1):
        if step in snaps:
            np.save(os.path.join(args.out, f"pts_{step:04d}.npy"), eng.get_liquid_particles())
            np.save(os.path.join(args.out, f"cube_{step:04d}.npy"), eng.get_vertices()[v0:v0 + n0])
        if step == args.steps:
            break
        ts = time.time(); eng.step(); wall_t = time.time() - ts
        if step % 20 == 0 or step < 3:
            x = eng.get_liquid_particles(); cube = eng.get_vertices()[v0:v0 + n0]
            nt = eng._engine.get_total_newton_iters()
            pc = eng._engine.get_total_pcg_iters()
            ymin, ymax = cube[:, 1].min(), cube[:, 1].max()
            level = np.percentile(x[:, 1], 97)   # free-surface proxy
            sub = np.clip((level - ymin) / (ymax - ymin), 0, 1)
            nnm, nn1 = nn_spacing(x)
            print(f"t={(step+1)*args.dt:6.3f}s cube y [{ymin*1000:6.1f},{ymax*1000:6.1f}] mm tilt {np.degrees(np.arctan2(cube[:,1].max()-cube[:,1].min()-c, c)):5.1f}"
                  f"  water p97 {level*1000:6.1f} mm (rest {surf*1000:.1f})  submerged {sub:.2f}  |v|max {np.linalg.norm(eng.get_liquid_velocities(),axis=1).max():.3f}"
                  f"  NN {nnm*1000:.2f}/{nn1*1000:.2f} mm  newton {nt-nt_prev:3d} pcg {pc-pc_prev:6.0f} {wall_t*1000:6.1f} ms", flush=True)
            nt_prev = nt; pc_prev = pc
    print(f"[m3] done {args.steps} steps in {time.time()-t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()

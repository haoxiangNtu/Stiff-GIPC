"""Nuka-Physics 'jelly ball drop' with StiffGIPC: a soft FEM ball (E 30 kPa, nu 0.3, r 100 mm) dropped 50 mm onto the floor.

Reference (Nuka mpm_jelly_demo): MLS-MPM elastic ball, radius 0.10 m, E 3e4, nu 0.3, density 1000, bottom 50 mm above the
floor, 5 s at dt 1/240. Here: structured tet ball (cube-to-sphere map of a Kuhn-subdivided grid), stable neo-Hookean FEM,
IPC contact with the ground plane.

    PYTHONPATH=. python examples/nuka/jelly_ball.py --out out/nuka/jelly1
"""
import argparse, os, os.path as osp, sys, time
import numpy as np

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "..", "robotgym"))
from common import Recorder, Scene  # noqa: E402
from sponge_transfer import box_tets  # noqa: E402


def ball_tets(radius, n_cells):
    """Tet ball: Kuhn-subdivided cube [-1,1]^3 with n_cells per side, vertices mapped to the sphere (cube-to-sphere)."""
    v, t = box_tets(np.array([2.0, 2.0, 2.0]), 2.0 / n_cells)
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    s = np.stack([x * np.sqrt(1 - y * y / 2 - z * z / 2 + y * y * z * z / 3),
                  y * np.sqrt(1 - z * z / 2 - x * x / 2 + z * z * x * x / 3),
                  z * np.sqrt(1 - x * x / 2 - y * y / 2 + x * x * y * y / 3)], 1) * radius
    a, b, c, d = (s[t[:, i]] for i in range(4))
    vol = np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)
    t = t.copy(); t[vol < 0] = t[vol < 0][:, [0, 2, 1, 3]]
    return s, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--radius", type=float, default=0.10)
    ap.add_argument("--cells", type=int, default=10, help="cube cells per side before the sphere map (10 -> 6000 tets)")
    ap.add_argument("--young", type=float, default=3e4)
    ap.add_argument("--poisson", type=float, default=0.3)
    ap.add_argument("--density", type=float, default=1000.0)
    ap.add_argument("--drop-gap", type=float, default=0.05, help="ball bottom above the floor at t = 0")
    ap.add_argument("--mu", type=float, default=0.4)
    ap.add_argument("--newton-cap", type=int, default=40)
    ap.add_argument("--log-level", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sc = Scene(dict(dt=args.dt, preconditioner_type=1, newton_iter_cap=args.newton_cap, poisson_rate=args.poisson,
                    absolute_dhat=1e-3), log_level=args.log_level)
    v, t = ball_tets(args.radius, args.cells)
    v = v + np.array([0.0, args.radius + args.drop_gap, 0.0])
    ball = sc.add_fem("jelly", v, t, young=args.young, density=args.density, color=(0.95, 0.35, 0.30))
    sc.eng.set_body_friction(len(sc.eng.get_load_records()) - 1, args.mu)
    sc.finalize()
    rec = Recorder(sc)
    print(f"[scene] jelly ball: r {args.radius*1e3:.0f} mm, {len(v)} verts, {len(t)} tets, E {args.young:.0f} Pa, nu {args.poisson}, "
          f"mass {4/3*np.pi*args.radius**3*args.density*1e3:.0f} g, bottom {args.drop_gap*1e3:.0f} mm above the floor", flush=True)
    rec_every = max(1, int(round(1.0 / (args.fps * args.dt))))
    n_steps = int(round(args.duration / args.dt))
    meta = dict(example="jelly_ball", dt=args.dt, fps=args.fps, radius=args.radius, young=args.young, poisson=args.poisson,
                density=args.density, drop_gap=args.drop_gap, n_tets=int(len(t)), fem_materials={"jelly": "jelly"}, fem_colors=[[0.95, 0.35, 0.30]])
    t0 = time.time(); t_last = t0
    for k in range(n_steps + 1):
        tt = k * args.dt
        if k % max(1, int(round(0.1 / args.dt))) == 0:
            V = sc.fem_verts("jelly"); now = time.time()
            print(f"[{tt:5.2f}s] ball y [{V[:,1].min()*1e3:6.1f}, {V[:,1].max()*1e3:6.1f}] mm, width {(V[:,0].max()-V[:,0].min())*1e3:6.1f} mm | "
                  f"{(now-t0)/60:4.1f} min, {(now-t_last)/max(1,int(round(0.1/args.dt))):.3f} s/step", flush=True)
            t_last = now
        if k % rec_every == 0:
            rec.record(tt, 0, "drop", liquid_pts=None)
        sc.step()
    meta["wall_min"] = (time.time() - t0) / 60
    rec.save(osp.join(args.out, "recording.npz"), meta)
    print(f"[done] {n_steps} steps in {meta['wall_min']:.1f} min -> {osp.join(args.out, 'recording.npz')}")


if __name__ == "__main__":
    main()

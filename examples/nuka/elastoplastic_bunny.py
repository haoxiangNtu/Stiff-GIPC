"""Nuka-Physics 'bunny elastoplastic impact' with StiffGIPC: a heavy rigid bunny falls onto an elastoplastic pad and leaves a dent.

Reference (Nuka elastoplastic_bunny_demo): 2.5 kg bunny (120 mm) dropped 180 mm onto a 240 x 200 x 60 mm pad on a plate;
MLS-MPM Hencky J2, E 50 kPa, nu 0.30, rho 1000, yield 12 kPa, hardening 3 kPa, 80 Pa.s Kelvin-Voigt viscosity, friction 0.2;
3.6 s in real time. Here: FEM pad with the engine's volumetric plasticity (set_fem_plasticity), ABD bunny, IPC contact.

    PYTHONPATH=. python examples/nuka/elastoplastic_bunny.py --out out/nuka/pad1
"""
import argparse, os, os.path as osp, sys, time
import numpy as np

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "..", "robotgym"))
from common import ASSETS, Recorder, Scene, T_of, abd_world_tf  # noqa: E402
from sponge_transfer import box_tets  # noqa: E402
from bunny_water import add_rigid_mesh, load_obj  # noqa: E402

PAD = np.array([0.24, 0.06, 0.20])          # pad x, height, z
PLATE = np.array([0.36, 0.024, 0.30])       # base plate x, thickness, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--duration", type=float, default=3.6)
    ap.add_argument("--cell", type=float, default=0.012, help="pad tet cell (0.012 -> 20 x 5 x 17 cubes x 6 = 10200 tets)")
    ap.add_argument("--young", type=float, default=5e4)
    ap.add_argument("--poisson", type=float, default=0.30)
    ap.add_argument("--density", type=float, default=1000.0)
    ap.add_argument("--yield-stress", type=float, default=12e3)
    ap.add_argument("--hardening", type=float, default=3e3)
    ap.add_argument("--bunny-extent", type=float, default=0.13)
    ap.add_argument("--bunny-mass", type=float, default=2.5)
    ap.add_argument("--drop-height", type=float, default=0.18, help="bunny bottom above the pad top at t = 0")
    ap.add_argument("--bunny-yaw", type=float, default=35.0)
    ap.add_argument("--mu", type=float, default=0.2)
    ap.add_argument("--newton-cap", type=int, default=50)
    ap.add_argument("--ptype", type=int, default=0)
    ap.add_argument("--log-level", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sc = Scene(dict(dt=args.dt, preconditioner_type=args.ptype, newton_iter_cap=args.newton_cap, poisson_rate=args.poisson,
                    absolute_dhat=1e-3), log_level=args.log_level)
    sc.add_box("plate", (0.0, PLATE[1] / 2, 0.0), PLATE, fixed=True, color=(0.55, 0.57, 0.60), mu=args.mu, cell=0.04)
    V, F = load_obj(osp.join(ASSETS, "triMesh", "bunny_surface.obj"))
    V = V - 0.5 * (V.min(0) + V.max(0)); V *= args.bunny_extent / (V.max(0) - V.min(0)).max()
    a = np.deg2rad(args.bunny_yaw); Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]); V = V @ Ry.T
    pad_top = PLATE[1] + PAD[1]
    V[:, 1] += pad_top + 0.0015 + args.drop_height - V[:, 1].min()
    vol = np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6.0
    rho = args.bunny_mass / vol
    bunny = add_rigid_mesh(sc, "bunny", V, F, rho, args.mu, color=(0.93, 0.93, 0.90))
    # (ABD bodies must be loaded before FEM bodies)
    v, t = box_tets(PAD, args.cell)
    pad = sc.add_fem("pad", v, t, transform=T_of(None, (0.0, PLATE[1] + PAD[1] / 2 + 0.0015, 0.0)), young=args.young,
                     density=args.density, color=(0.93, 0.62, 0.25))
    sc.eng.set_body_friction(len(sc.eng.get_load_records()) - 1, args.mu)
    sc.finalize()
    if args.yield_stress > 0:
        sc.eng.set_fem_plasticity(0, args.yield_stress, args.hardening)
    rec = Recorder(sc)
    print(f"[scene] pad {PAD[0]*1e3:.0f} x {PAD[2]*1e3:.0f} x {PAD[1]*1e3:.0f} mm, {len(v)} verts, {len(t)} tets, E {args.young:.0f}, nu {args.poisson}, yield {args.yield_stress:.0f}, H {args.hardening:.0f}; "
          f"bunny {args.bunny_extent*1e3:.0f} mm, {vol*1e6:.0f} ml, {args.bunny_mass:.2f} kg (rho {rho:.0f}), dropped {args.drop_height*1e3:.0f} mm", flush=True)
    n_steps = int(round(args.duration / args.dt)); rec_every = max(1, int(round(1.0 / (args.fps * args.dt))))
    ids = np.array([bunny.body_id], np.int32)
    top_v = np.nonzero(np.abs(v[:, 1] - PAD[1] / 2) < 1e-6)[0]     # pad top-surface vertices (local rest coords)
    meta = dict(example="elastoplastic_bunny", dt=args.dt, fps=args.fps, young=args.young, poisson=args.poisson, density=args.density,
                yield_stress=args.yield_stress, hardening=args.hardening, bunny_mass=args.bunny_mass, drop_height=args.drop_height,
                n_tets=int(len(t)), fem_materials={"pad": "plastic"}, fem_colors=[[0.93, 0.62, 0.25]])
    series = []; t0 = time.time()
    for k in range(n_steps + 1):
        tt = k * args.dt
        if k % max(1, int(round(0.1 / args.dt))) == 0:
            P = sc.fem_verts("pad"); dent = pad_top + 0.0015 - P[top_v, 1].min()
            y = float(abd_world_tf(sc.eng.get_abd_body_transforms(ids)[0])[1, 3])
            ep = sc.eng.mean_plastic_strain(0) if args.yield_stress > 0 else 0.0
            series.append((tt, dent, y, ep))
            print(f"[{tt:5.2f}s] bunny centre y {y*1e3:6.1f} mm | pad dent {dent*1e3:5.1f} mm, mean plastic strain {ep:.4f} | {(time.time()-t0)/60:.1f} min", flush=True)
        if k % rec_every == 0:
            rec.record(tt, 0, "free fall" if tt < np.sqrt(2 * args.drop_height / 9.81) else "impact and rest")
        sc.step()
    P = sc.fem_verts("pad"); dent = pad_top + 0.0015 - P[top_v, 1].min()
    meta["wall_min"] = (time.time() - t0) / 60; meta["series"] = series; meta["final_dent"] = float(dent)
    rec.save(osp.join(args.out, "recording.npz"), meta)
    print(f"[result] final dent {dent*1e3:.1f} mm, mean plastic strain {sc.eng.mean_plastic_strain(0) if args.yield_stress > 0 else 0:.4f}")
    print(f"[done] {n_steps} steps in {meta['wall_min']:.1f} min -> {osp.join(args.out, 'recording.npz')}")


if __name__ == "__main__":
    main()

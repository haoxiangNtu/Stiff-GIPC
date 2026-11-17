"""Nuka-Physics 'elastoplastic compression' with StiffGIPC: an 80 x 80 x 100 mm specimen between two platens, light compression
(gap 96 mm, recovers), unload, strong compression (gap 45 mm, leaves a permanent set), unload.

Reference (Nuka elastoplastic_compression_demo): MLS-MPM Hencky J2, E 50 kPa, nu 0.30, rho 1000, yield 12 kPa, hardening 3 kPa;
light-load recovery error 0.065 %, strong load leaves 33.1 % height compression. Here: FEM tets (stable neo-Hookean) with the
engine's volumetric plasticity (set_fem_plasticity: Hencky J2 return map between Newton steps), IPC contact with the platens.

    PYTHONPATH=. python examples/nuka/elastoplastic_compression.py --out out/nuka/comp1
"""
import argparse, os, os.path as osp, sys, time
if __name__ == "__main__": import os as _os; _os.environ.setdefault("LIQ_UG", "0")   # main-path example, when run as a script: pinned to the main liquid path (the engine default is the UG mode since 2026-11-18)
import numpy as np

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "..", "robotgym"))
from common import ASSETS, Body, Recorder, Scene, T_of, _load_urdf_capture  # noqa: E402
from sponge_transfer import box_tets  # noqa: E402

SPEC = np.array([0.08, 0.10, 0.08])        # specimen x, height (y), z (Nuka: 80 x 80 x 100 mm, z up)
PLATE = np.array([0.17, 0.018, 0.14])      # platen x, thickness, z
BASE_TOP = 0.018                           # bottom platen top face (its bottom sits on the ground plane)


def schedule(t, open_gap, mild_gap, strong_gap, v=0.02, hold=1.5, dwell=1.0):
    """Top-platen gap (bottom face above the base top) vs time: open -> mild -> open -> strong -> open, speed v m/s."""
    segs = []
    g = open_gap; tt = 0.0
    plan = [("settle", open_gap, dwell), ("light load", mild_gap, None), ("hold light", mild_gap, hold), ("unload light", open_gap, None),
            ("dwell", open_gap, dwell), ("strong load", strong_gap, None), ("hold strong", strong_gap, hold), ("unload strong", open_gap, None),
            ("rest", open_gap, 2.0)]
    for label, target, dur in plan:
        d = abs(target - g) / v if dur is None else dur
        segs.append((label, tt, tt + d, g, target)); g = target; tt += d
    for label, t0, t1, g0, g1 in segs:
        if t <= t1:
            s = 0.0 if t1 == t0 else np.clip((t - t0) / (t1 - t0), 0, 1)
            return label, g0 + (g1 - g0) * s, tt
    return segs[-1][0], segs[-1][4], tt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--cell", type=float, default=0.01, help="specimen tet cell (0.01 -> 3840 tets)")
    ap.add_argument("--young", type=float, default=5e4)
    ap.add_argument("--poisson", type=float, default=0.30)
    ap.add_argument("--density", type=float, default=1000.0)
    ap.add_argument("--yield-stress", type=float, default=12e3)
    ap.add_argument("--hardening", type=float, default=3e3)
    ap.add_argument("--open-gap", type=float, default=0.13)
    ap.add_argument("--mild-gap", type=float, default=0.096)
    ap.add_argument("--strong-gap", type=float, default=0.045)
    ap.add_argument("--speed", type=float, default=0.02, help="platen speed (m/s)")
    ap.add_argument("--mu", type=float, default=0.0, help="specimen<->platen friction (Nuka: 0)")
    ap.add_argument("--newton-cap", type=int, default=50)
    ap.add_argument("--press-stiff", type=float, default=500.0, help="prismatic drive strength of the press joint (--drive press)")
    ap.add_argument("--drive", default="press", choices=["press", "teleport", "set", "drop"], help="how the top platen moves: teleport q each step, set_abd_body_transforms each step, or free fall (debug)")
    ap.add_argument("--platen-density", type=float, default=2e6, help="top platen density: heavy enough that the contact push-back per step (F dt^2 / m) is << the teleport advance")
    ap.add_argument("--ptype", type=int, default=0, help="preconditioner_type (0 = plain PCG; 1 = MAS, metis-cache trap with load_mesh_from_data)")
    ap.add_argument("--log-level", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sc = Scene(dict(dt=args.dt, preconditioner_type=args.ptype, newton_iter_cap=args.newton_cap, poisson_rate=args.poisson,
                    absolute_dhat=1e-3), log_level=args.log_level)
    base = sc.add_box("platen_base", (0.0, BASE_TOP / 2, 0.0), PLATE, fixed=True, color=(0.55, 0.57, 0.60), mu=args.mu, cell=0.03)
    # the top platen is a FREE but very heavy ABD box teleported to the prescribed pose every step (a Fixed body ignores
    # teleports in the contact geometry): within one step it moves < 1 um under gravity / contact, i.e. it is kinematic
    press_j = None
    if args.drive == "press":
        # prismatic-joint press (URDF: fixed bar + platen on a prismatic joint along -y): the driver lives inside the Newton
        # solve like the FR3 finger joints, so the IPC line search / CCD see the platen motion (a teleported body does not)
        T = np.eye(4); T[1, 3] = BASE_TOP + args.open_gap + PLATE[1] / 2 + 0.06
        n_rec0 = len(sc.eng.get_load_records())
        _load_urdf_capture(sc.eng, osp.join(ASSETS, "nuka", "press", "press.urdf"), T, True, False, 1e7, {})
        sc.eng.set_log_level(args.log_level)
        for r in sc.eng.get_load_records()[n_rec0:]:
            if r.body_type == 0:
                b = Body(r, f"press/{r.label}", "fixed" if "bar" in str(r.label) else "prop", color=(0.55, 0.57, 0.60))
                sc.bodies[b.name] = b; sc.eng.add_ground_collision_skip(b.body_id); sc.eng.set_body_friction(b.body_id, args.mu)
                if "platen" in str(r.label): top = b
        press_j = sc.eng.native.get_num_prismatic_joints() - 1
    else:
        top = sc.add_box("platen_top", (0.0, BASE_TOP + args.open_gap + PLATE[1] / 2, 0.0), PLATE, fixed=False, density=args.platen_density, color=(0.55, 0.57, 0.60), mu=args.mu, cell=0.03)
    v, t = box_tets(SPEC, args.cell)
    spec = sc.add_fem("specimen", v, t, transform=T_of(None, (0.0, BASE_TOP + SPEC[1] / 2 + 0.0015, 0.0)), young=args.young,
                      density=args.density, color=(0.93, 0.62, 0.25))
    sc.eng.set_body_friction(len(sc.eng.get_load_records()) - 1, args.mu)
    sc.finalize()
    if args.yield_stress > 0:
        sc.eng.set_fem_plasticity(0, args.yield_stress, args.hardening)
    rec = Recorder(sc)
    h0 = SPEC[1]
    print(f"[scene] specimen {SPEC[0]*1e3:.0f} x {SPEC[2]*1e3:.0f} x {h0*1e3:.0f} mm, {len(v)} verts, {len(t)} tets, E {args.young:.0f}, nu {args.poisson}, "
          f"yield {args.yield_stress:.0f} Pa, H {args.hardening:.0f} Pa; gaps open/mild/strong {args.open_gap*1e3:.0f}/{args.mild_gap*1e3:.0f}/{args.strong_gap*1e3:.0f} mm", flush=True)
    _, _, T_total = schedule(1e9, args.open_gap, args.mild_gap, args.strong_gap, args.speed)
    n_steps = int(round(T_total / args.dt)); rec_every = max(1, int(round(1.0 / (args.fps * args.dt))))
    top_ids = np.array([top.body_id], np.int32)
    C_top0 = np.array(sc.eng.get_abd_body_transforms(top_ids)[0], np.float64)   # engine layout; translation in [:3, 3]
    meta = dict(example="elastoplastic_compression", dt=args.dt, fps=args.fps, young=args.young, poisson=args.poisson, density=args.density,
                yield_stress=args.yield_stress, hardening=args.hardening, gaps=[args.open_gap, args.mild_gap, args.strong_gap], speed=args.speed,
                n_tets=int(len(t)), fem_materials={"specimen": "plastic"}, fem_colors=[[0.93, 0.62, 0.25]])
    series = []
    t0 = time.time(); last = None; heights = {}
    for k in range(n_steps + 1):
        tt = k * args.dt
        label, gap, _ = schedule(tt, args.open_gap, args.mild_gap, args.strong_gap, args.speed)
        pt_post = np.asarray(sc.eng.get_abd_body(top.body_id).get_vertices(), np.float64)[:, 1].min() - BASE_TOP
        C = C_top0.copy(); C[1, 3] += gap - args.open_gap          # move relative to the loaded pose
        if args.drive == "press":
            sc.eng.native.set_prismatic_force(press_j, 0.0); sc.eng.native.set_prismatic_strength(press_j, args.press_stiff)
            sc.eng.native.set_prismatic_target(press_j, float(args.open_gap - gap))
        elif args.drive == "teleport":
            sc.eng.teleport_abd_bodies(top_ids, C[None])
        elif args.drive == "set":
            sc.eng.set_abd_body_transforms(top_ids, C[None])
        if k % max(1, int(round(0.1 / args.dt))) == 0 or label != last:
            V = sc.fem_verts("specimen"); h = V[:, 1].max() - BASE_TOP; w = V[:, 0].max() - V[:, 0].min()
            pt = np.asarray(sc.eng.get_abd_body(top.body_id).get_vertices(), np.float64)[:, 1].min() - BASE_TOP   # platen bottom gap AFTER the last step (before this teleport)
            ep = sc.eng.mean_plastic_strain(0) if args.yield_stress > 0 else 0.0
            series.append((tt, gap, h, w, ep))
            if label != last:
                print(f"[{tt:6.2f}s] {label:14s} gap {gap*1e3:6.1f} (platen after solve {pt_post*1e3:6.1f}) mm | specimen height {h*1e3:6.2f} mm ({(1-h/h0)*100:5.2f} % compressed), width {w*1e3:6.1f} mm, "
                      f"mean plastic strain {ep:.4f} | {(time.time()-t0)/60:.1f} min", flush=True)
                heights[label] = h; last = label
        if k % rec_every == 0:
            rec.record(tt, 0, label)
        sc.step()
    V = sc.fem_verts("specimen"); h = V[:, 1].max() - BASE_TOP
    meta["wall_min"] = (time.time() - t0) / 60; meta["series"] = series; meta["final_height"] = float(h)
    rec.save(osp.join(args.out, "recording.npz"), meta)
    print(f"[result] after light load+unload: height {heights.get('dwell', float('nan'))*1e3:.2f} mm (recovery error {abs(heights.get('dwell', h0)/h0-1)*100:.3f} %); "
          f"after strong load+unload: {h*1e3:.2f} mm = {(1-h/h0)*100:.1f} % residual compression (Nuka: 0.065 % / 33.1 %)")
    print(f"[done] {n_steps} steps in {meta['wall_min']:.1f} min -> {osp.join(args.out, 'recording.npz')}")


if __name__ == "__main__":
    main()

"""Nuka-Physics 'dynamic elastoplastic gripper' with StiffGIPC: an FR3 pinches a 24 x 36 x 28 mm elastoplastic specimen
lightly (gap 16.5 mm), opens, pinches strongly (gap 4 mm) and releases; the specimen keeps a permanent set.

Reference (Nuka robot_elastoplastic_demo): Panda, open 40 / mild 16.5 / strong 4 mm, MLS-MPM Hencky J2 with E 50 kPa, nu 0.25,
rho 1000, yield 12 kPa, hardening 7.5 kPa; their penetration 4 mm against a 2 mm budget ("visual preview"). Here: FEM tets with
the engine's volumetric plasticity, IPC contact (penetration-free), FR3 driven by IK with prismatic finger joints.

    PYTHONPATH=. python examples/nuka/elastoplastic_gripper.py --out out/nuka/grip1
"""
import argparse, os, os.path as osp, sys, time
if __name__ == "__main__": import os as _os; _os.environ.setdefault("LIQ_UG", "0")   # main-path example, when run as a script: pinned to the main liquid path (the engine default is the UG mode since 2026-11-18)
import numpy as np

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "..", "robotgym"))
from common import Arm, Recorder, Scene, T_of, Timeline, arm_base_T  # noqa: E402
from sponge_transfer import box_tets, hand_R_topdown  # noqa: E402

ARM_BASE = ((0.0, 0.0, 0.0), 0.0)
PED_C, PED = np.array([0.45, 0.05, 0.0]), np.array([0.12, 0.10, 0.12])     # pedestal centre / size (top at y = 0.10)
SPEC = np.array([0.024, 0.036, 0.028])                                       # specimen x (pinch axis), height, z
LIFT = 0.15


def timeline(open_d, mild_d, strong_d, y_grip, pinch_s=2.0):
    R = hand_R_topdown(); P = lambda x, y, z: T_of(R, (x, y, z))
    gx, gz = PED_C[0], PED_C[2]
    tl = Timeline(P(gx, y_grip + LIFT, gz))
    tl.add("settle (open fingers)", 1.0, finger=open_d)
    tl.add("descend", 2.5, P(gx, y_grip, gz))
    tl.add("light pinch", pinch_s, finger=mild_d)
    tl.add("hold light", 1.5)
    tl.add("open", 1.5, finger=open_d)
    tl.add("dwell", 0.5)
    tl.add("strong pinch", pinch_s + 0.5, finger=strong_d)
    tl.add("hold strong", 1.5)
    tl.add("release", 1.5, finger=open_d)
    tl.add("retreat", 1.5, P(gx, y_grip + LIFT, gz))
    tl.add("rest", 1.0)
    return tl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--cell", type=float, default=0.004, help="specimen tet cell (0.004 -> 6 x 9 x 7 cubes x 6 = 2268 tets)")
    ap.add_argument("--young", type=float, default=5e4)
    ap.add_argument("--poisson", type=float, default=0.25)
    ap.add_argument("--density", type=float, default=1000.0)
    ap.add_argument("--yield-stress", type=float, default=12e3)
    ap.add_argument("--hardening", type=float, default=7.5e3)
    ap.add_argument("--open-gap", type=float, default=0.040)
    ap.add_argument("--mild-gap", type=float, default=0.0165)
    ap.add_argument("--strong-gap", type=float, default=0.004)
    ap.add_argument("--pinch-speed", type=float, default=0.015, help="pad gap speed during pinches (m/s)")
    ap.add_argument("--finger-speed", type=float, default=0.06)
    ap.add_argument("--finger-stiff", type=float, default=25.0)
    ap.add_argument("--mu", type=float, default=0.6)
    ap.add_argument("--newton-cap", type=int, default=40)
    ap.add_argument("--ptype", type=int, default=0)
    ap.add_argument("--log-level", type=int, default=0)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--max-time", type=float, default=0.0)
    args = ap.parse_args()

    y_spec = PED_C[1] + PED[1] / 2 + SPEC[1] / 2 + 0.0015          # specimen centre (1.5 mm above the pedestal top)
    tl = timeline(args.open_gap / 2, args.mild_gap / 2, args.strong_gap / 2, y_spec)
    T_total = tl.total() if callable(tl.total) else float(tl.total)
    if args.dry:
        arm = Arm("arm", arm_base_T(*ARM_BASE)); q = arm.kin.HOME.copy(); worst = 0.0
        q, ok = arm.kin.ik(tl.segs[0].T_to, q, iters=300)
        for t in np.arange(0.0, T_total, 0.05):
            seg, s = tl.at(t); T = seg.pose(s); qn, ok = arm.kin.ik(T, q, iters=80)
            worst = max(worst, np.linalg.norm(arm.kin.fk(qn)[:3, 3] - T[:3, 3])); q = qn
        print(f"[dry] worst position error {worst*1e3:.2f} mm over {T_total:.1f} s"); return

    os.makedirs(args.out, exist_ok=True)
    sc = Scene(dict(dt=args.dt, preconditioner_type=args.ptype, newton_iter_cap=args.newton_cap, poisson_rate=args.poisson,
                    absolute_dhat=1e-3), log_level=args.log_level)
    probe = Arm("arm", arm_base_T(*ARM_BASE)); q0, ok = probe.kin.ik(tl.segs[0].T_to, probe.kin.HOME, iters=300)
    arm = sc.add_arm("arm", ARM_BASE[0], ARM_BASE[1], q0=q0)
    arm.stiff = args.finger_stiff; arm.finger_v = args.finger_speed
    sc.add_box("pedestal", PED_C, PED, fixed=True, color=(0.55, 0.57, 0.60), mu=args.mu, cell=0.04)
    v, t = box_tets(SPEC, args.cell)
    spec = sc.add_fem("specimen", v, t, transform=T_of(None, (PED_C[0], y_spec, PED_C[2])), young=args.young, density=args.density,
                      color=(0.93, 0.62, 0.25))
    recs = sc.eng.get_load_records()
    sc.eng.set_body_friction(len(recs) - 1, args.mu)
    for i, r in enumerate(recs):
        if "finger" in str(r.label):
            sc.eng.set_body_friction(i, args.mu)
    sc.finalize()
    if args.yield_stress > 0:
        sc.eng.set_fem_plasticity(0, args.yield_stress, args.hardening)
    rec = Recorder(sc)
    print(f"[scene] specimen {SPEC[0]*1e3:.0f} x {SPEC[2]*1e3:.0f} x {SPEC[1]*1e3:.0f} mm, {len(v)} verts, {len(t)} tets, E {args.young:.0f}, nu {args.poisson}, "
          f"yield {args.yield_stress:.0f}, H {args.hardening:.0f}; gaps open/mild/strong {args.open_gap*1e3:.1f}/{args.mild_gap*1e3:.1f}/{args.strong_gap*1e3:.1f} mm; {T_total:.1f} s at dt {args.dt}", flush=True)
    arm.drive(sc.eng, arm.q); arm.finger_goal = args.open_gap / 2
    n_steps = int(round(T_total / args.dt)); rec_every = max(1, int(round(1.0 / (args.fps * args.dt))))
    if args.max_time > 0:
        n_steps = min(n_steps, int(round(args.max_time / args.dt)))
    fingers = [(str(r.label), i) for i, r in enumerate(recs) if "finger" in str(r.label)]
    meta = dict(example="elastoplastic_gripper", dt=args.dt, fps=args.fps, young=args.young, poisson=args.poisson, density=args.density,
                yield_stress=args.yield_stress, hardening=args.hardening, gaps=[args.open_gap, args.mild_gap, args.strong_gap],
                n_tets=int(len(t)), fem_materials={"specimen": "plastic"}, fem_colors=[[0.93, 0.62, 0.25]])
    series = []; widths = {}; t0 = time.time(); last = None
    for k in range(n_steps + 1):
        tt = k * args.dt
        seg, s = tl.at(tt); T = seg.pose(s)
        q, ok = arm.kin.ik(T, arm.q, iters=40); arm.drive(sc.eng, q)
        if seg.finger is not None:
            arm.finger_goal = seg.finger
        arm.finger_v = args.pinch_speed if "pinch" in seg.label else args.finger_speed
        if k % max(1, int(round(0.1 / args.dt))) == 0 or seg.label != last:
            V = sc.fem_verts("specimen"); h = V[:, 1].max() - V[:, 1].min()
            mid = np.abs(V[:, 1] - y_spec) < 0.006; w = V[mid, 0].max() - V[mid, 0].min()      # waist width (pad height band), not the bulging ends
            ep = sc.eng.mean_plastic_strain(0) if args.yield_stress > 0 else 0.0; gap = arm.gap_m(sc.eng)
            series.append((tt, gap, w, h, ep))
            if seg.label != last:
                print(f"[{tt:6.2f}s] {seg.label:22s} gap {gap*1e3:5.1f} mm | specimen waist {w*1e3:5.2f} mm ({(1-w/SPEC[0])*100:5.1f} % squeezed), height {h*1e3:5.2f} mm, "
                      f"mean plastic strain {ep:.4f} | {(time.time()-t0)/60:.1f} min", flush=True)
                widths[seg.label] = w; last = seg.label
                bb = lambda P: f"x[{P[:,0].min()*1e3:6.1f},{P[:,0].max()*1e3:6.1f}] y[{P[:,1].min()*1e3:6.1f},{P[:,1].max()*1e3:6.1f}] z[{P[:,2].min()*1e3:6.1f},{P[:,2].max()*1e3:6.1f}]"
                print(f"          specimen bbox {bb(V)} (mm)")
                for nm, bi in fingers:
                    print(f"          {nm:24s} bbox {bb(np.asarray(sc.eng.get_abd_body(bi).get_vertices(), np.float64))}")
        if k % rec_every == 0:
            rec.record(tt, 0, seg.label)
        sc.step()
    V = sc.fem_verts("specimen"); mid = np.abs(V[:, 1] - y_spec) < 0.006; w = V[mid, 0].max() - V[mid, 0].min()
    meta["wall_min"] = (time.time() - t0) / 60; meta["series"] = series; meta["final_width"] = float(w)
    rec.save(osp.join(args.out, "recording.npz"), meta)
    print(f"[result] waist after the light pinch+open: {widths.get('dwell', float('nan'))*1e3:.2f} mm; final waist {w*1e3:.2f} mm = "
          f"{(1-w/SPEC[0])*100:.1f} % permanent set (rest 24.0 mm); mean plastic strain {sc.eng.mean_plastic_strain(0) if args.yield_stress > 0 else 0:.4f}")
    print(f"[done] {n_steps} steps in {meta['wall_min']:.1f} min -> {osp.join(args.out, 'recording.npz')}")


if __name__ == "__main__":
    main()

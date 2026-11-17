#!/usr/bin/env python3
"""RobotGym bed task on the user's FBX assets: a throw blanket (毛毯.fbx, 1.9 x 1.2 m) on a double bed (床.fbx x1.4,
1.56 x 1.98 m frame, bedding top 0.50 m), folded by two FR3 arms. IPC cloth on the bed's own triangles (one fixed body:
duvet + top sheet + frame + legs + headboard; make_bed_fbx.py), contact-only grasps.

    PYTHONPATH=. python examples/robotgym/bed_blanket_fold.py --no-arms --max-time 3 --out out/rg_fbx/bed_drop   # drape only
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
import time

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from common import (ASSETS, R_cols, Recorder, Scene, Segment, T_of, Timeline, load_obj, minjerk)  # noqa: E402

BED_DIR = osp.join(ASSETS, "robotgym", "fbx", "bed")
TIP = 0.0089                      # fingertip beyond the TCP along the approach (finger_collision.stl: tip at hand z 0.1123, TCP 0.1034)


def hand_R_topdown():
    """hand z (approach) down, fingers (hand y) along world z: the pads close across the foot edge (along the fold direction),
    so the cloth gathered between them is a ridge parallel to the edge"""
    zh = np.array([0.0, -1.0, 0.0]); yh = np.array([0.0, 0.0, 1.0]); xh = np.cross(yh, zh)
    return R_cols(xh, yh, zh)


def bedding_y(Vb, Fb, x, z, y_max=0.6):
    """height of the bedding top at (x, z): highest hit of a vertical ray on the bed collision mesh below y_max"""
    A, B, C = Vb[Fb[:, 0]], Vb[Fb[:, 1]], Vb[Fb[:, 2]]
    d = (B[:, 2] - C[:, 2]) * (A[:, 0] - C[:, 0]) + (C[:, 0] - B[:, 0]) * (A[:, 2] - C[:, 2])
    ok = np.abs(d) > 1e-12
    d = np.where(ok, d, 1.0)
    l1 = ((B[:, 2] - C[:, 2]) * (x - C[:, 0]) + (C[:, 0] - B[:, 0]) * (z - C[:, 2])) / d
    l2 = ((C[:, 2] - A[:, 2]) * (x - C[:, 0]) + (A[:, 0] - C[:, 0]) * (z - C[:, 2])) / d
    l3 = 1.0 - l1 - l2
    y = l1 * A[:, 1] + l2 * B[:, 1] + l3 * C[:, 1]
    y = y[ok & (l1 >= 0) & (l2 >= 0) & (l3 >= 0) & (y < y_max)]
    return float(y.max()) if y.size else float("nan")


def pinch_tilt(a, Vb, Fb):
    """--pinch-tilt auto: roll each hand about world x so its two open fingertips at the landing point sit at the same height
    above the bedding (under the right pinch the bedding top falls 4 mm toward the foot across the 60 mm pad span, so only the
    head-side pad pressed); returns degrees per side, + = foot-side fingertip lower"""
    if a.pinch_tilt != "auto":
        return {s: float(a.pinch_tilt) for s in ("right", "left")}
    z_in = a.pinch_z + (0.5 * a.open_gap - a.finger_close if a.pinch_mode == "head" else 0.0)
    out = {}
    for side, sx in (("right", 1.0), ("left", -1.0)):
        h_head = bedding_y(Vb, Fb, sx * a.pinch_x, z_in - 0.5 * a.open_gap)
        h_foot = bedding_y(Vb, Fb, sx * a.pinch_x, z_in + 0.5 * a.open_gap)
        out[side] = float(np.degrees(np.arctan2(h_head - h_foot, a.open_gap)))
    return out


class ArcSegment(Segment):
    """carry the pinched line over on a circle about the fold line (c = (y, z) on the bedding surface) instead of a straight
    lift + carry, which stretches the strip from the pinch to the fold line and drags the head half toward the foot; the
    radius goes from |start - c| to |end - c|, minus `sag` at the top (slack in the strip); the hand keeps its orientation"""

    def __init__(self, label, dur, T_from, T_to, c, sag):
        super().__init__(label, dur, T_from, T_to)
        self.c, self.sag = np.asarray(c, float), float(sag)

    def pose(self, s):
        u = minjerk(s)
        p0, p1 = self.T_from[:3, 3], self.T_to[:3, 3]        # T_from is replaced at run time after the force-controlled pinch
        d0, d1 = np.array([p0[1], p0[2]]) - self.c, np.array([p1[1], p1[2]]) - self.c
        th = (1 - u) * np.arctan2(d0[0], d0[1]) + u * np.arctan2(d1[0], d1[1])      # angle from +z toward +y
        r = (1 - u) * np.linalg.norm(d0) + u * np.linalg.norm(d1) - self.sag * np.sin(np.pi * u)
        return T_of(self.T_from[:3, :3], ((1 - u) * p0[0] + u * p1[0], self.c[0] + r * np.sin(th), self.c[1] + r * np.cos(th)))


def timeline(side, a, y_surf):
    """top-down pinch on the blanket lying on the bed (both fingertips on the cloth, the pads gather a fold between them),
    lift, carry the pinched line over toward the head (fold in half), lay down, release"""
    sx = 1.0 if side == "right" else -1.0
    phi = np.radians(getattr(a, "tilt", {}).get(side, 0.0))          # roll about world x (--pinch-tilt)
    R = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(phi), -np.sin(phi)], [0.0, np.sin(phi), np.cos(phi)]]) @ hand_R_topdown()
    P = lambda x, y, z: T_of(R, (sx * x, y, z))
    y_tcp = y_surf + TIP + a.press                         # fingertips `press` above the bedding surface
    z_in = a.pinch_z + (0.5 * a.open_gap - a.finger_close if a.pinch_mode == "head" else 0.0)
    tl = Timeline(P(a.pinch_x, y_surf + 0.15, z_in))
    tl.add("settle (blanket drapes, fingers open)", a.settle, finger=a.open_gap / 2)
    tl.add("descend onto the blanket", 2.0, P(a.pinch_x, y_tcp, z_in))
    if a.press_force > 0:
        tl.add("press (find the cloth by force)", a.press_time)
    tl.add("pinch (gather a fold)", a.pinch_time, P(a.pinch_x, y_tcp, a.pinch_z) if a.pinch_mode == "head" else None, finger=a.finger_close)
    if a.fold_path == "arc":                               # one segment as long as lift + fold over
        tl.segs.append(ArcSegment("fold over (arc)", 2.0 + a.fold_time, tl.segs[-1].T_to, P(a.pinch_x, y_surf + a.lay_y + 0.03, a.fold_z),
                                  (y_surf, 0.5 * (a.pinch_z + a.fold_z)), a.arc_sag))
    else:
        tl.add("lift", 2.0, P(a.pinch_x, y_surf + a.lift, a.pinch_z))
        tl.add("fold over", a.fold_time, P(a.pinch_x, y_surf + a.lift - 0.02, a.fold_z))
    tl.add("lay down", 2.0, P(a.pinch_x, y_surf + a.lay_y, a.fold_z))
    tl.add("release", 1.0, finger=a.open_gap / 2)
    tl.add("lift off", 1.5, P(a.pinch_x, y_surf + a.lift, a.fold_z))
    tl.add("retreat", 1.5, P(a.pinch_x + 0.10, y_surf + a.lift, a.fold_z + 0.10))
    tl.add("rest", 1.5)
    return tl


def arm_base(side, a):
    sx = 1.0 if side == "right" else -1.0
    return (sx * a.base_x, a.pedestal, a.base_z), (180.0 if side == "right" else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/rg_fbx/bed")
    ap.add_argument("--no-arms", action="store_true", help="bed + blanket only (drape test)")
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--blanket", default="robotgym/fbx/bed/blanket_20mm.obj")
    ap.add_argument("--blanket-y", type=float, default=0.52, help="start height of the flat blanket (bedding top 0.496)")
    ap.add_argument("--blanket-z0", type=float, default=0.10, help="foot edge of the blanket (bedding foot face at z = 0): overhang")
    ap.add_argument("--blanket-x", type=float, default=0.0)
    ap.add_argument("--cloth-young", type=float, default=1e5)
    ap.add_argument("--bend-young", type=float, default=1e5, help="1e5 at t = 1 mm -> 5.6e-6 N m (fleece/wool throw; cotton sheet 3e4)")
    ap.add_argument("--areal-density", type=float, default=0.40, help="kg/m^2 (wool/fleece throw ~0.3-0.5)")
    ap.add_argument("--precond", type=int, default=1)
    ap.add_argument("--newton-cap", type=int, default=100)
    ap.add_argument("--dhat", type=float, default=2e-3)
    ap.add_argument("--friction", type=float, default=0.5)
    ap.add_argument("--bed-mu", type=float, default=0.5)
    ap.add_argument("--newton-tol", type=float, default=1e-3)
    ap.add_argument("--max-time", type=float, default=0.0)
    ap.add_argument("--save-every", type=float, default=1.0)
    ap.add_argument("--log-level", type=int, default=0)
    ap.add_argument("--dry", action="store_true", help="IK reachability check only")
    ap.add_argument("--clearance", action="store_true", help="with --dry: link-to-bed clearance along the path")
    ap.add_argument("--base-x", type=float, default=1.12, help="|x| of the arm bases (bedding side at 0.795)")
    ap.add_argument("--base-z", type=float, default=-0.47, help="z of the arm bases (middle of the fold path)")
    ap.add_argument("--pedestal", type=float, default=0.30, help="height of the arm mounts (mobile-base platform) [m]")
    ap.add_argument("--pinch-x", type=float, default=0.62)
    ap.add_argument("--pinch-z", type=float, default=-0.10, help="pinch line (bedding foot face at z = 0; the foot edge rounds off over 6 cm)")
    ap.add_argument("--surface-y", type=float, default=0.456, help="bedding top under the pinch (duvet top near x 0.6, z -0.1)")
    ap.add_argument("--press", type=float, default=0.0015, help="fingertip height above the bedding surface when gathering (m)")
    ap.add_argument("--open-gap", type=float, default=0.060)
    ap.add_argument("--finger-close", type=float, default=5e-4)
    ap.add_argument("--pinch-tilt", default="0", help="roll of the hand about world x in degrees (+ = foot-side fingertip lower), "
                    "or 'auto': per hand, so both open fingertips sit at the same height above the bedding at the landing point")
    ap.add_argument("--pinch-mode", choices=("sym", "head"), default="sym",
                    help="head: the head-side pad stays put while the pads close (the hand follows half the gap change toward the head), "
                         "so gathering the fold does not pull the head half toward the foot; the pinch still ends on --pinch-z")
    ap.add_argument("--finger-speed", type=float, default=0.03, help="pad gap speed while gathering (m/s)")
    ap.add_argument("--finger-mu", type=float, default=1.0, help="finger friction (rubber-like pads; pair mu = geometric mean)")
    ap.add_argument("--settle", type=float, default=2.0)
    ap.add_argument("--pinch-time", type=float, default=2.5)
    ap.add_argument("--lift", type=float, default=0.22, help="TCP height above the surface while carrying")
    ap.add_argument("--fold-z", type=float, default=-0.85, help="where the pinched line is laid (mirror of -0.10 about the fold line -0.475)")
    ap.add_argument("--fold-time", type=float, default=4.0)
    ap.add_argument("--fold-path", choices=("line", "arc"), default="line",
                    help="arc: carry the pinched line over on a circle about the fold line (keeps the strip to the fold line slack)")
    ap.add_argument("--arc-sag", type=float, default=0.06, help="with --fold-path arc: radius reduction at the top of the arc (m)")
    ap.add_argument("--lay-y", type=float, default=0.05)
    ap.add_argument("--press-force", type=float, default=0.0, help="> 0: force-controlled press, N of upward contact force on the two fingers")
    ap.add_argument("--descend-speed", type=float, default=0.005, help="force-controlled descent speed (m/s)")
    ap.add_argument("--press-time", type=float, default=1.5, help="force-controlled press segment (fingers open) before the pinch")
    args = ap.parse_args()
    Vb, Fb = load_obj(osp.join(BED_DIR, "bed_collision.obj"))
    args.tilt = pinch_tilt(args, Vb, Fb)
    if any(args.tilt.values()):
        print("[scene] pinch tilt " + ", ".join(f"{s} {v:+.2f} deg" for s, v in args.tilt.items()))
    tls = {s: timeline(s, args, args.surface_y) for s in ("right", "left")}
    if args.dry:
        from common import Arm, arm_base_T
        for side in ("right", "left"):
            pos, yaw = arm_base(side, args)
            arm = Arm(side, arm_base_T(pos, yaw)); q = arm.kin.HOME.copy(); worst = 0.0; wdq = 0.0
            q, ok = arm.kin.ik(tls[side].segs[0].T_to, q, iters=300)
            for t in np.arange(0.0, tls[side].total, 0.05):
                seg, sg = tls[side].at(t); T = seg.pose(sg); qn, ok = arm.kin.ik(T, q, iters=80)
                err = np.linalg.norm(arm.kin.fk(qn)[:3, 3] - T[:3, 3]); worst = max(worst, err); wdq = max(wdq, np.abs(qn - q).max()); q = qn
                if err > 2e-3:
                    print(f"[dry] {side} t {t:5.2f} {seg.label:36s} IK err {err*1e3:.1f} mm")
            print(f"[dry] {side}: worst IK error {worst*1e3:.2f} mm, max dq/50ms {wdq:.3f} rad over {tls[side].total:.1f} s")
            if args.clearance:
                from common import arm_clearance
                arm_clearance(arm, tls[side], Vb, Fb, sample=0.006)
        return

    os.makedirs(args.out, exist_ok=True)
    meta_bed = json.load(open(osp.join(BED_DIR, "bed_meta.json")))
    t_cloth = 1e-3
    sc = Scene(dict(dt=args.dt, cloth_thickness=t_cloth, cloth_young_modulus=args.cloth_young, bend_young_modulus=args.bend_young,
                    cloth_density=args.areal_density / t_cloth, strain_rate=100.0, preconditioner_type=args.precond,
                    collision_detection_buff_scale=12.0, newton_iter_cap=args.newton_cap, absolute_dhat=args.dhat,
                    friction_rate=args.friction, newton_tol=args.newton_tol), log_level=args.log_level)
    arms = {}
    if not args.no_arms:
        from common import Arm, arm_base_T
        for side in ("right", "left"):
            pos, yaw = arm_base(side, args)
            probe = Arm(side, arm_base_T(pos, yaw))
            q0, ok = probe.kin.ik(tls[side].segs[0].T_to, probe.kin.HOME, iters=300)
            print(f"[scene] {side} arm base {pos} yaw {yaw}: start IK ok={ok}")
            arms[side] = sc.add_arm(side, pos, yaw, q0=q0)
            arms[side].finger_v = args.finger_speed
    sc.add_static_mesh("bed", Vb, Fb, color=(0.93, 0.93, 0.95), mu=args.bed_mu)
    if arms:
        for side, arm in arms.items():          # mobile-base platform under each arm (2 mm below the base; never touches the arm)
            pos, _ = arm_base(side, args)
            ped = sc.add_box(f"pedestal_{side}", (pos[0], (args.pedestal - 0.002) / 2, pos[2]), (0.36, args.pedestal - 0.002, 0.36),
                             fixed=True, color=(0.20, 0.20, 0.22), cell=0.06)
            for b in arm.bodies.values():
                sc.eng.add_collision_exclusion(ped.body_id, b.body_id)
            for nm in ("fr3_leftfinger", "fr3_rightfinger"):
                sc.eng.set_body_friction(arm.bodies[nm].body_id, args.finger_mu)
    lz = meta_bed["blanket_size"][1]
    blanket_T = T_of(None, (args.blanket_x, args.blanket_y, args.blanket_z0 - lz / 2))
    sc.add_cloth("blanket", osp.join(ASSETS, args.blanket), blanket_T, young=args.cloth_young, color=(0.80, 0.55, 0.42))
    sc.finalize()
    rec = Recorder(sc)
    print(f"[scene] bed {len(Fb)} faces; blanket {sc.fem['blanket']['n']} verts at y {args.blanket_y}, z {args.blanket_z0 - lz:.3f}..{args.blanket_z0:.3f}")
    T_total = tls["right"].total if arms else 3.0
    if args.max_time > 0:
        T_total = min(T_total, args.max_time) if arms else args.max_time
    n_steps = int(round(T_total / args.dt))
    for a in sc.arms:
        a.drive(sc.eng, a.q); a.finger_goal = 0.0
    last_seg = None
    rec_every = max(1, int(round(1.0 / (args.fps * args.dt))))
    save_every = max(1, int(round(args.save_every / args.dt)))
    meta = dict(example="bed_blanket_fold", dt=args.dt, fps=args.fps, blanket=args.blanket, cloth_young=args.cloth_young,
                bend_young=args.bend_young, areal_density=args.areal_density, bed_scale=meta_bed["scale"], pinch_tilt_deg=args.tilt,
                args={k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool))})
    rec_path = osp.join(args.out, "recording.npz")
    t0 = time.time(); newton_prev = int(sc.eng.native.get_total_newton_iters())
    # force-controlled press (--press-force > 0): during the descent the hand goes down at --descend-speed until the cloth pushes
    # back on its two fingers with press_force (N, vertical), then holds that force while the pads close; a fixed press depth
    # failed on the sloping, rigid bedding (one fingertip 7 mm above the cloth, the other pressing; 1.0 mm stalled Newton)
    fc = {a.name: dict(y=None, F=0.0, last=None, T=None) for a in sc.arms}
    finger_slices = {a.name: [(a.bodies[nm].v_off, a.bodies[nm].v_off + len(a.bodies[nm].rest_verts)) for nm in ("fr3_leftfinger", "fr3_rightfinger")]
                     for a in sc.arms}
    def finger_up_force():
        F = np.asarray(sc.eng.get_vertex_contact_forces(False, "total"))
        return {n: [float(F[i0:i1, 1].sum()) for i0, i1 in sl] for n, sl in finger_slices.items()}
    for k in range(n_steps + 1):
        t = k * args.dt
        label = "drape"
        Fup = finger_up_force() if (args.press_force > 0 and any(tls[a.name].at(t)[0].label.startswith(("press", "pinch")) for a in sc.arms)) else None
        for a in sc.arms:
            seg, sg = tls[a.name].at(t)
            T = seg.pose(sg)
            st = fc[a.name]
            if args.press_force > 0 and seg.label.startswith(("press", "pinch")):
                f_now = sum(Fup[a.name]); st["F"] = f_now
                if st["y"] is None:
                    st["y"] = float(T[1, 3])
                y_floor = args.surface_y + TIP - 0.004
                if seg.label.startswith("press"):          # go down until the cloth pushes back with press_force
                    if f_now < args.press_force:
                        st["y"] = max(y_floor, st["y"] - args.descend_speed * args.dt)
                else:                                       # pinch: hold the force in [0.5, 2] x target (5 mm/s corrections)
                    if f_now < 0.5 * args.press_force:
                        st["y"] = max(y_floor, st["y"] - 2.5e-5)
                    elif f_now > 2.0 * args.press_force:
                        st["y"] += 2.5e-5
                T = T.copy(); T[1, 3] = st["y"]
            elif st["y"] is not None and st["last"] is not None and st["last"].startswith("pinch") and seg.label in ("lift", "fold over (arc)"):
                seg.T_from = st["T"].copy()      # lift starts where the force-controlled hand actually was
                T = seg.pose(sg)
            if args.pinch_mode == "head" and seg.label.startswith("pinch"):     # head-side pad face held at pinch_z - finger_close
                T = T.copy(); T[2, 3] = args.pinch_z - args.finger_close + 0.5 * a.gap_m(sc.eng)
            st["last"] = seg.label; st["T"] = T
            q, ok = a.kin.ik(T, a.q, iters=40)
            a.drive(sc.eng, q)
            if seg.finger is not None:
                a.finger_goal = seg.finger
            label = seg.label
        if label != last_seg:
            print(f"[{t:6.2f}s] {label}  ({(time.time()-t0)/60:.1f} min)", flush=True); last_seg = label
        if k % rec_every == 0:
            rec.record(t, 0, label)
        if k and k % int(round(0.25 / args.dt)) == 0:
            X = sc.fem_verts("blanket"); nt = int(sc.eng.native.get_total_newton_iters())
            extra = "".join(f" | {a.name}: gap {a.gap_m(sc.eng)*1e3:5.1f} mm, cloth near pads {np.sum(np.linalg.norm(X - a.tcp()[:3, 3], axis=1) < 0.03)}"
                            + (f", F {fc[a.name]['F']:5.2f} N y {fc[a.name]['y']:.4f}" if fc[a.name]['y'] is not None else "")
                            for a in sc.arms)
            print(f"[{t:5.2f}s] newton {(nt-newton_prev)/int(round(0.25/args.dt)):5.1f}/step  blanket y {X[:,1].min():.3f}..{X[:,1].max():.3f}"
                  f"{extra}  ({(time.time()-t0)/60:.1f} min)", flush=True); newton_prev = nt
        if k and k % save_every == 0:
            rec.save(rec_path, dict(meta, partial=True))
        sc.step()
    meta["wall_min"] = (time.time() - t0) / 60
    rec.save(rec_path, meta)
    print(f"[done] {n_steps} steps in {meta['wall_min']:.1f} min -> {rec_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""RobotGym showcase 3 — bed sheet fold with two FR3 arms, StiffGIPC IPC cloth.

A 1.0 x 0.6 m sheet rests on a thin fixed slab (1.0 x 0.4 m, top at y = 0.30) so that its near and
far edges overhang by 10 cm and hang down.  Each arm pinches the hanging near flap 4 cm in from
its corner (the open fingers straddle the flap, approaching along the edge), lifts it up and over to
the far half and lets go: the sheet is folded in half.  Everything is contact (IPC + friction), no
attachment constraints.

    PYTHONPATH=. python examples/robotgym/bedsheet_fold.py --out out/robotgym/bedsheet          # pod
    PYTHONPATH=. python examples/robotgym/bedsheet_fold.py --dry                               # IK check only (CPU)
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
from common import (ASSETS, R_cols, Recorder, Scene, T_of, Timeline, minjerk)  # noqa: E402

# ------------------------------------------------------------------ geometry (metres, y up)
Y_TOP = 0.30                      # slab top
SLAB = dict(centre=(0.0, Y_TOP - 0.01, 0.35), size=(1.0, 0.02, 0.40))   # z in [0.15, 0.55]
SHEET_LX, SHEET_LZ = 1.0, 0.6     # sheet centred at z = 0.35 -> z in [0.05, 0.65]: 10 cm overhang near/far
SHEET_C = np.array([0.0, Y_TOP + 1.5e-3, 0.35])
FLAP_Z = 0.115                    # pinch 3.5 cm outside the slab edge (z = 0.15): both pads (+-2.6 cm) stay clear of the slab
CLEAR_Z = 0.03                    # TCP z at which the 20 cm wide hand is fully outside the slab footprint (lift straight up from here)
PINCH_Y = Y_TOP - 0.07            # pinch 3 cm above the hem (flap hangs 10 cm)
PINCH_X = 0.46                    # 4 cm in from each corner. 0.40 (10 cm, sheet15_*/sheetF_* runs): the fingers point 25 deg up, so the
                                  # wrist (link7) hangs ~9 cm below the TCP and sits right above the slab's side edge (x = 0.5); laying
                                  # the hem down to Y_TOP + 0.06 drove it ~1 cm into the slab -> arm jammed, Newton at the cap 17-20 s
                                  # and the released cloth frozen in the air. At 0.46 the wrist passes outside the slab (>= 15 mm clear)
# base placement, hand orientation: IK sweep 2026-10-18 (worst position error 0.2 mm over the whole motion)
ARM_BASE = {"right": ((1.05, 0.0, 0.15), 180.0),    # at +x, facing -x
            "left": ((-1.05, 0.0, 0.15), -20.0)}    # at -x, facing +x (yawed 20 deg)
HAND_PITCH = -25.0                # approach tilted downward (deg) about the finger axis
FINGER_AXIS_SIGN = {"right": 1.0, "left": -1.0}


def hand_R(side):
    """Hand z (approach) along the edge toward the sheet centre, tilted HAND_PITCH down; fingers
    (hand y) along world z, so the open fingers straddle the hanging flap."""
    from common import axis_angle_R
    sx = 1.0 if side == "right" else -1.0
    zh = np.array([-sx, 0.0, 0.0])
    yh = np.array([0.0, 0.0, FINGER_AXIS_SIGN[side]])
    xh = np.cross(yh, zh)
    R0 = R_cols(xh, yh, zh)
    return axis_angle_R([0, 0, 1], np.radians(HAND_PITCH * sx)) @ R0


def timeline(side, finger_close=0.0, end="liftoff", lay_y=0.06, lay_z=0.50, pinch_x=PINCH_X):
    sx = 1.0 if side == "right" else -1.0
    R = hand_R(side)
    P = lambda x, y, z: T_of(R, (sx * x, y, z))
    tl = Timeline(P(0.64, PINCH_Y + 0.06, FLAP_Z))
    tl.add("settle (open fingers)", 1.5, finger=0.030)
    tl.add("approach flap", 2.0, P(0.64, PINCH_Y, FLAP_Z))
    tl.add("slide in along the edge", 2.0, P(pinch_x, PINCH_Y, FLAP_Z))
    tl.add("pinch", 1.5, finger=finger_close)
    # the fingers sit 5 cm below the slab underside: rising here jams the arm against the fixed slab (full2/full3/x1 froze at
    # 8.0 s with dX = dB = 0). Pull the flap outward until the hand is clear of the slab footprint, then go up and over.
    tl.add("pull out", 2.0, P(pinch_x, PINCH_Y, CLEAR_Z))
    tl.add("lift", 3.0, P(pinch_x, Y_TOP + 0.15, CLEAR_Z))         # 0.22 put the hands 0.81 m from the bases (elbow straight, IK 24 mm)
    tl.add("fold over", 4.0, P(pinch_x, Y_TOP + 0.10, 0.50))   # laying the edge at z 0.64 (past the slab end 0.55) dragged the whole sheet off: 13% fold
    if end == "legacy":   # bedsheet_full5 / sheet15_* ending: the hand ends ~1 cm above the slab's far end edge -> on the 15 mm
        # sheets the right hand sagged onto that edge at release (jam: arm lag 12.7 deg, Newton at the cap, 0.6 s frozen, then a
        # 1.9 m/s pop) and the cloth near the hands looked suspended in the air for ~2 s (user review 2026-10-21)
        tl.add("lay down", 2.0, P(pinch_x, Y_TOP + 0.035, 0.56))
        tl.add("release", 1.2, finger=0.030)
        tl.add("retreat", 2.0, P(pinch_x, Y_TOP + 0.15, 0.45))
        tl.add("rest", 1.5)
        return tl
    # liftoff ending: lay the hem down from lay_y above the slab, inside its footprint (z < 0.55), open, lift straight up so the
    # hand never sweeps over the slab edge or the folded cloth, then back off and let the cloth settle
    tl.add("lay down", 2.0, P(pinch_x, Y_TOP + lay_y, lay_z))
    tl.add("release", 1.0, finger=0.030)
    tl.add("lift off", 1.5, P(pinch_x, Y_TOP + 0.15, lay_z))
    tl.add("retreat", 1.5, P(pinch_x + 0.08, Y_TOP + 0.15, 0.45))
    tl.add("rest", 2.0)
    return tl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/robotgym/bedsheet")
    ap.add_argument("--tactile", action="store_true", help="VBTS gel pads on the fingertips (examples/lab_tactile): "
                    "indentation images + normal forces recorded; gaps are pad-face gaps")
    ap.add_argument("--dry", action="store_true", help="IK reachability check only, no engine")
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--sheet", default="robotgym/sheet_100x60_15mm.obj", help="15 mm (2788 verts): 25 mm folds into element-scale creases")
    ap.add_argument("--cloth-young", type=float, default=1e5, help="membrane E (Pa); 1e6 (engine default) makes the lift hit the Newton cap")
    ap.add_argument("--bend-young", type=float, default=3e4, help="bending E (Pa)")
    ap.add_argument("--precond", type=int, default=1, help="1 = MAS (needed once the pinch loads the cloth)")
    ap.add_argument("--save-every", type=float, default=2.0, help="checkpoint the recording every N sim seconds")
    ap.add_argument("--diag", action="store_true", help="print per-record-frame cloth/arm diagnostics")
    ap.add_argument("--max-time", type=float, default=0.0)
    ap.add_argument("--log-level", type=int, default=0)
    ap.add_argument("--newton-cap", type=int, default=100)
    ap.add_argument("--slab-cell", type=float, default=0.05)
    ap.add_argument("--dhat", type=float, default=2e-3, help="absolute IPC barrier distance (m); 2 mm gives the zero-thickness cloth a graspable thickness")
    ap.add_argument("--finger-close", type=float, default=5e-4, help="per-finger joint distance when pinching: 0.5 mm -> 1 mm gap, each cloth side at dhat/4 (0 = pads crush the cloth to 0.1 mm and the Newton stalls)")
    ap.add_argument("--no-sheet", action="store_true", help="arms + slab only (isolation test)")
    ap.add_argument("--slab-mu", type=float, default=0.5, help="slab friction coefficient")
    ap.add_argument("--friction", type=float, default=0.5, help="global friction_rate (cloth, fingers)")
    ap.add_argument("--end", default="liftoff", choices=["liftoff", "legacy"], help="ending after the fold (legacy = full5/sheet15_* runs)")
    ap.add_argument("--lay-y", type=float, default=0.06, help="liftoff ending: TCP height above the slab top when laying the hem down")
    ap.add_argument("--lay-z", type=float, default=0.50, help="liftoff ending: TCP z when laying the hem down vertically (slab ends at 0.55); 0.54 let the hem overhang the far end (the lift drags the sheet ~10 cm toward the robots): 67 %% fold vs 100 %%")
    ap.add_argument("--newton-tol", type=float, default=1e-3)
    ap.add_argument("--pinch-x", type=float, default=PINCH_X, help="|x| of the pinch points (sheet corners at 0.5); 0.40 = sheet15_*/sheetF_* runs")
    args = ap.parse_args()

    tls = {s: timeline(s, args.finger_close, args.end, args.lay_y, args.lay_z, args.pinch_x) for s in ("right", "left")}
    T_total = tls["right"].total

    if args.dry:
        from common import Arm, arm_base_T
        for side, (pos, yaw) in ARM_BASE.items():
            arm = Arm(side, arm_base_T(pos, yaw))
            q = arm.kin.HOME.copy()
            worst = 0.0
            for t in np.arange(0.0, T_total, 0.1):
                seg, s = tls[side].at(t)
                T = seg.pose(s)
                q, ok = arm.kin.ik(T, q, iters=120)
                err = np.linalg.norm(arm.kin.fk(q)[:3, 3] - T[:3, 3])
                worst = max(worst, err)
                if not ok:
                    print(f"[dry] {side} t={t:5.2f}s {seg.label:26s} IK not converged, pos err {err*1e3:.2f} mm")
            print(f"[dry] {side}: worst position error {worst*1e3:.2f} mm over {T_total:.1f} s; q range "
                  f"{np.round(arm.kin.lo, 2).tolist()} .. {np.round(arm.kin.hi, 2).tolist()}")
        return

    os.makedirs(args.out, exist_ok=True)
    sc = Scene(dict(dt=args.dt, cloth_thickness=1e-3, cloth_young_modulus=args.cloth_young,
                    bend_young_modulus=args.bend_young, cloth_density=200.0, strain_rate=100.0,
                    preconditioner_type=args.precond, collision_detection_buff_scale=12.0,
                    newton_iter_cap=args.newton_cap, absolute_dhat=args.dhat, friction_rate=args.friction, newton_tol=args.newton_tol),
               log_level=args.log_level, tactile=args.tactile)
    # arms spawn at the IK solution of their first pose (no sweep through the slab)
    arms = {}
    for side, (pos, yaw) in ARM_BASE.items():
        from common import Arm, arm_base_T
        probe = Arm(side, arm_base_T(pos, yaw))
        q0, ok = probe.kin.ik(tls[side].segs[0].T_to, probe.kin.HOME, iters=200)
        print(f"[scene] {side} start pose IK ok={ok}")
        arms[side] = sc.add_arm(side, pos, yaw, q0=q0)
    sc.add_box("slab", SLAB["centre"], SLAB["size"], fixed=True, color=(0.55, 0.42, 0.30), mu=args.slab_mu, cell=args.slab_cell)
    sc.add_tactile_pads()          # FEM gels after the last ABD body
    sheet_T = T_of(None, SHEET_C)
    if not args.no_sheet:
        sc.add_cloth("sheet", osp.join(ASSETS, args.sheet), sheet_T, young=args.cloth_young)
    # the sheet starts flat in the air over the overhangs; nothing touches the arms at t = 0
    sc.finalize()
    rec = Recorder(sc)
    print(f"[scene] bodies {len(sc.bodies)} (arms {sum(len(a.bodies) for a in sc.arms)}), cloth verts "
          f"{sc.fem['sheet']['n'] if not args.no_sheet else 0}; total {T_total:.1f} s at dt {args.dt}")

    for a in sc.arms:
        a.drive(sc.eng, a.q)
        a.finger_goal = 0.0
    rec_every = max(1, int(round(1.0 / (args.fps * args.dt))))
    n_steps = int(round(T_total / args.dt))
    if args.max_time > 0:
        n_steps = min(n_steps, int(round(args.max_time / args.dt)))
    meta = dict(example="bedsheet_fold", dt=args.dt, fps=args.fps, sheet=args.sheet,
                cloth_young=args.cloth_young, bend_young=args.bend_young, pinch_x=args.pinch_x, end=args.end, lay_y=args.lay_y)
    rec_path = osp.join(args.out, "recording.npz")
    save_every = max(1, int(round(args.save_every / args.dt))) if args.save_every > 0 else 0
    cloth = lambda: sc.fem_verts("sheet") if not args.no_sheet else np.full((1, 3), 9.9)
    X_prev = cloth(); pcg_prev = 0.0
    body_ids = np.array([b.body_id for b in sc.bodies.values()], np.int32)
    B_prev = np.asarray(sc.eng.get_abd_body_transforms(body_ids), float)
    t0 = time.time()
    last_seg = None
    newton_every = max(1, int(round(0.5 / args.dt))); newton_prev = int(sc.eng.native.get_total_newton_iters())
    for k in range(n_steps + 1):
        t = k * args.dt
        for a in sc.arms:
            seg, s = tls[a.name].at(t)
            T = seg.pose(s)
            q, ok = a.kin.ik(T, a.q, iters=40)
            a.drive(sc.eng, q)
            if seg.finger is not None:
                a.finger_goal = seg.finger
        seg_r, _ = tls["right"].at(t)
        if seg_r.label != last_seg:
            print(f"[{t:6.2f}s] {seg_r.label}  ({(time.time()-t0)/60:.1f} min)", flush=True)
            last_seg = seg_r.label
        if k and k % newton_every == 0:   # solver health: Newton iterations per step over the last 0.5 s (cap = stalled / truncated steps)
            nt = int(sc.eng.native.get_total_newton_iters()); per = (nt - newton_prev) / newton_every; newton_prev = nt
            print(f"  [newton {t:6.2f}s] {per:5.1f} iters/step (cap {args.newton_cap})  {(time.time()-t0)/60:.1f} min", flush=True)
        if k % rec_every == 0:
            rec.record(t, 0, seg_r.label)
            if args.diag:
                X = cloth(); dX = np.linalg.norm(X - X_prev, axis=1).max() * 1e3; X_prev = X
                Bn = np.asarray(sc.eng.get_abd_body_transforms(body_ids), float)
                dB = np.abs(Bn - B_prev).max() * 1e3; B_prev = Bn
                pcg = float(getattr(sc.eng.native, "get_total_pcg_iters", lambda: 0.0)()); parts = []
                ang = np.asarray(sc.eng.get_revolute_current_angles_abs(), float)
                for a in sc.arms:
                    p = a.tcp()[:3, 3]; d = np.linalg.norm(X - p, axis=1).min() * 1e3
                    dq = np.abs(np.array([ang[j] for j in a.rev_ids]) - a.q).max()
                    parts.append(f"{a.name}: d={d:5.1f}mm gap={a.gap_m(sc.eng)*1e3:5.1f}mm dq={dq:.3f}")
                print(f"[diag {t:6.2f}s] dX={dX:6.2f}mm dB={dB:6.2f} ymin={X[:,1].min():.3f} ymax={X[:,1].max():.3f} | "
                      + " | ".join(parts) + f" | pcg={pcg-pcg_prev:.0f}", flush=True); pcg_prev = pcg
        if save_every and k and k % save_every == 0:
            rec.save(rec_path, dict(meta, partial=True, wall_min=(time.time() - t0) / 60))
        sc.step()
    meta["wall_min"] = (time.time() - t0) / 60
    rec.save(rec_path, meta)
    json.dump(meta, open(osp.join(args.out, "meta.json"), "w"), indent=1)
    print(f"[done] {n_steps} steps in {(time.time()-t0)/60:.1f} min -> {args.out}/recording.npz")


if __name__ == "__main__":
    main()

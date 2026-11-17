#!/usr/bin/env python3
"""RobotGym showcase 1 — sponge water transfer: a soft open-cell sponge (FEM tets, porous coupling to the engine
MPM liquid) soaks in a basin of water; a FR3 picks it up, carries it over a cup and wrings it three times.

    PYTHONPATH=. python examples/robotgym/sponge_transfer.py --out out/robotgym/sponge      # pod
    PYTHONPATH=. python examples/robotgym/sponge_transfer.py --dry                          # IK check only

--no-porous runs the same scene with plain contact (water cannot enter the sponge): mechanics baseline.
"""
from __future__ import annotations
if __name__ == "__main__": import os as _os; _os.environ.setdefault("LIQ_UG", "0")   # main-path example, when run as a script: pinned to the main liquid path (the engine default is the UG mode since 2026-11-18)

import argparse
import json
import os
import os.path as osp
import sys
import time

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from common import (ASSETS, R_cols, Recorder, Scene, T_of, Timeline, Arm, arm_base_T)  # noqa: E402

# ------------------------------------------------------------------ geometry (metres, y up, ground y = 0)
WALL = 0.015
BOWL_C = np.array([0.45, 0.0, -0.13])          # basin centre (floor level)
BOWL_IN = np.array([0.28, 0.07, 0.14])         # interior x, wall height, interior z (the 20 cm hand lies along x inside it)
WATER_H = 0.040                                # water depth in the basin
CUP_C = np.array([0.45, 0.0, 0.17])
CUP_IN = np.array([0.09, 0.11, 0.09])
FLOOR_T = 0.012                                # basin / cup floor thickness
SPONGE = np.array([0.06, 0.035, 0.10])         # kitchen sponge, long side along z; the pads close on the 60 mm width along x
SPONGE_C0 = BOWL_C + np.array([0.0, FLOOR_T + 0.5 * SPONGE[1] + 0.004, 0.0])
ARM_BASE = ((0.0, 0.0, 0.0), 0.0)
LIFT_H = 0.16


def hand_R_topdown():
    """Hand z down, fingers (hand y) along world x: the pads close on the sponge's 60 mm width. The hand body is
    204 mm long along the finger axis and dips 10 mm below the basin rim at the grasp, so the basin interior along
    x (280 mm) has to clear it: with the fingers along z in a 140 mm basin the hand sat on both walls 18 mm above
    the planned grasp height, the pads caught only the top 14 mm of the block and the arm fought the fixed walls
    (Newton at the cap through the whole lift)."""
    zh = np.array([0.0, -1.0, 0.0]); yh = np.array([1.0, 0.0, 0.0]); xh = np.cross(yh, zh)
    return R_cols(xh, yh, zh)


def timeline(n_wring=3, open_d=0.039, hold_d=0.022, squeeze_d=0.010, wring_s=2.5, grip_dy=0.0, hold_s=1.0, regrasp=None):
    """finger values are per-finger joint distances (gap = 2 d): open 78 mm, hold 44 mm (sponge 60 -> 27 % squeeze;
    at 50 mm the 2.3 N soaked block slipped 10 mm down the pads and tilted 30 deg during the lift), wring 20 mm.
    The pads move at Arm.finger_v (set per segment in main: the wring is slow, 12 mm/s of gap, so the soft block is
    squeezed, not shot out of the 2 cm pads)."""
    R = hand_R_topdown()
    P = lambda x, y, z: T_of(R, (x, y, z))
    gx, gz = SPONGE_C0[0], SPONGE_C0[2]
    y_grip = SPONGE_C0[1] + grip_dy               # pad centre at the sponge mid-height (fingertips ~9 mm above the basin floor)
    cx, cz = CUP_C[0], CUP_C[2]
    y_cup = CUP_C[1] + CUP_IN[1] + 0.05           # sponge bottom 5 cm above the cup rim
    dz0 = regrasp[0] if regrasp else 0.0          # regrasp: the first grasp is already at its offset (regrasp4_v3 grasped the centre,
    tl = Timeline(P(gx, y_grip + LIFT_H, gz + dz0))   # then put the sponge back 36 mm off and every later grasp landed off-plan)
    tl.add("settle (open fingers)", 1.5, finger=open_d)
    tl.add("descend into the basin", 3.0, P(gx, y_grip, gz + dz0))
    tl.add("grasp", 2.0, finger=hold_d)
    tl.add("soak", 1.0)
    if regrasp:
        # regrasp cycles: the 20 mm pads only compress the column they hold, so each wring squeezes a DIFFERENT part of
        # the block: wring, carry back, release, regrasp shifted by regrasp[i] along the sponge (z), soak, lift, wring...
        for i, dz in enumerate(regrasp):
            sq = squeeze_d[min(i, len(squeeze_d) - 1)] if isinstance(squeeze_d, (list, tuple)) else squeeze_d
            if i > 0:
                tl.add(f"regrasp {i+1}: move along the sponge", 1.5, P(gx, y_grip + 0.5 * LIFT_H, gz + dz))
                tl.add(f"regrasp {i+1}: descend", 1.5, P(gx, y_grip, gz + dz))
                tl.add(f"regrasp {i+1}: grasp", 2.0, finger=hold_d)
                tl.add(f"regrasp {i+1}: soak", 1.0)
            tl.add(f"lift {i+1} out of the water", 2.0, P(gx, y_grip + LIFT_H, gz + dz))
            tl.add(f"carry {i+1} over the cup", 2.5, P(cx, y_cup + 0.5 * SPONGE[1], cz + dz))   # sponge centred over the cup
            tl.add(f"wring {i+1}", wring_s, finger=sq)
            tl.add(f"hold {i+1}", hold_s)
            tl.add(f"relax {i+1}", 1.5, finger=hold_d)
            tl.add(f"carry back {i+1}", 2.5, P(gx, y_grip + LIFT_H, gz + dz))
            tl.add(f"lower {i+1} into the basin", 2.0, P(gx, y_grip, gz + dz))
            tl.add(f"release {i+1}", 1.5, finger=open_d)
        tl.add("retreat", 2.0, P(gx, y_grip + LIFT_H, gz + regrasp[-1]))
        tl.add("rest", 1.0)
        return tl
    tl.add("lift out of the water", 2.0, P(gx, y_grip + LIFT_H, gz))
    tl.add("carry over the cup", 2.5, P(cx, y_cup + 0.5 * SPONGE[1], cz))
    for i in range(n_wring):
        sq = squeeze_d[min(i, len(squeeze_d) - 1)] if isinstance(squeeze_d, (list, tuple)) else squeeze_d
        tl.add(f"wring {i+1}", wring_s, finger=sq)
        tl.add(f"hold {i+1}", hold_s)
        tl.add(f"relax {i+1}", 1.5, finger=hold_d)
    tl.add("carry back", 2.5, P(gx, y_grip + LIFT_H, gz))
    tl.add("lower into the basin", 2.0, P(gx, y_grip, gz))
    tl.add("release", 1.5, finger=open_d)
    tl.add("retreat", 2.0, P(gx, y_grip + LIFT_H, gz))
    tl.add("rest", 1.0)
    return tl


def box_tets(size, cell):
    """Structured tet grid of an axis-aligned box centred at the origin (6 tets per cube)."""
    n = np.maximum(1, np.round(np.asarray(size) / cell).astype(int))
    xs = [np.linspace(-size[k] / 2, size[k] / 2, n[k] + 1) for k in range(3)]
    gx, gy, gz = np.meshgrid(*xs, indexing="ij")
    verts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)
    vid = lambda i, j, k: (i * (n[1] + 1) + j) * (n[2] + 1) + k
    tets = []
    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                c = [vid(i + a, j + b, k + d) for a in (0, 1) for b in (0, 1) for d in (0, 1)]  # index a*4+b*2+d
                v = lambda a, b, d: c[a * 4 + b * 2 + d]
                # Kuhn subdivision along the main diagonal v(0,0,0)-v(1,1,1)
                p0, p7 = v(0, 0, 0), v(1, 1, 1)
                for path in [(v(1,0,0), v(1,1,0)), (v(1,0,0), v(1,0,1)), (v(0,1,0), v(1,1,0)),
                             (v(0,1,0), v(0,1,1)), (v(0,0,1), v(1,0,1)), (v(0,0,1), v(0,1,1))]:
                    tets.append((p0, path[0], path[1], p7))
    tets = np.array(tets, np.int32)
    # consistent positive orientation
    a, b, cc, d = (verts[tets[:, i]] for i in range(4))
    vol = np.einsum("ij,ij->i", np.cross(b - a, cc - a), d - a)
    tets[vol < 0] = tets[vol < 0][:, [0, 2, 1, 3]]
    return verts, tets


def add_basin(sc, name, centre, inner, color):
    """Open-top rectangular basin: floor + 4 walls (fixed tessellated boxes). The walls are sunk 4 mm into the
    floor and the five parts exclude each other from collision: coincident fixed faces (distance 0) would give a
    NaN barrier energy (first sponge run), and a gap would let particles through."""
    cx, cy, cz = centre; ix, h, iz = inner
    emb = 0.004
    parts = [sc.add_box(f"{name}_floor", (cx, cy + FLOOR_T / 2, cz), (ix + 2 * WALL, FLOOR_T, iz + 2 * WALL), color=color, cell=0.03)]
    yc = cy + FLOOR_T - emb + (h + emb) / 2
    parts.append(sc.add_box(f"{name}_wall_xp", (cx + ix / 2 + WALL / 2, yc, cz), (WALL, h + emb, iz + 2 * WALL), color=color, cell=0.03))
    parts.append(sc.add_box(f"{name}_wall_xm", (cx - ix / 2 - WALL / 2, yc, cz), (WALL, h + emb, iz + 2 * WALL), color=color, cell=0.03))
    parts.append(sc.add_box(f"{name}_wall_zp", (cx, yc, cz + iz / 2 + WALL / 2), (ix, h + emb, WALL), color=color, cell=0.03))
    parts.append(sc.add_box(f"{name}_wall_zm", (cx, yc, cz - iz / 2 - WALL / 2), (ix, h + emb, WALL), color=color, cell=0.03))
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            sc.eng.add_collision_exclusion(parts[i].body_id, parts[j].body_id)


def seed_water(dx, ppc, rng):
    n_sub = max(1, int(round(ppc ** (1.0 / 3.0)))); hs = dx / n_sub; m = 0.6 * dx
    lo = BOWL_C + np.array([-BOWL_IN[0] / 2 + m, FLOOR_T + m, -BOWL_IN[2] / 2 + m])
    hi = BOWL_C + np.array([BOWL_IN[0] / 2 - m, FLOOR_T + WATER_H, BOWL_IN[2] / 2 - m])
    ax = [np.arange(lo[k], hi[k], hs) for k in range(3)]
    g = np.meshgrid(*ax, indexing="ij")
    pts = np.stack([a.ravel() for a in g], 1) + rng.uniform(-0.3 * hs, 0.3 * hs, (g[0].size, 3))
    # keep the sponge footprint free of water at t=0 (the sponge sits on the floor; water fills around it)
    sp_lo, sp_hi = SPONGE_C0 - SPONGE / 2 - m, SPONGE_C0 + SPONGE / 2 + m
    inside = np.all((pts > sp_lo) & (pts < sp_hi), 1)
    return pts[~inside], hs


def main():
    global WATER_H
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/robotgym/sponge")
    ap.add_argument("--tactile", action="store_true", help="VBTS gel pads on the fingertips (examples/lab_tactile): "
                    "indentation images + normal forces recorded; gaps are pad-face gaps")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--sponge-young", type=float, default=2.5e4)
    ap.add_argument("--squeeze-gap", default="0.024", help="pad gap at full wring (m); comma-separated per wring, e.g. "
                    "0.028,0.024,0.020 (a wrung sponge only gives more water when squeezed harder than before)")
    ap.add_argument("--hold-gap", type=float, default=0.044, help="pad gap while carrying (m); 50 mm slipped (p6)")
    ap.add_argument("--wring-hold", type=float, default=2.0, help="seconds the pads stay closed after each wring (the pore water "
                    "leaves at ~g(1-1/s)/drag, 10 mm/s at drag 200: a 1 s hold only emptied the bottom centimetre)")
    ap.add_argument("--grip-dy", type=float, default=0.0, help="pad centre height relative to the sponge centre (m)")
    ap.add_argument("--finger-stiff", type=float, default=25.0, help="prismatic drive strength of the finger joints")
    ap.add_argument("--finger-speed", type=float, default=0.03, help="pad gap speed (m/s) outside the wring")
    ap.add_argument("--wring-speed", type=float, default=0.012, help="pad gap speed (m/s) during the wring")
    ap.add_argument("--grip-mu", type=float, default=1.0, help="friction of the finger pads and the sponge (pair mu = geometric mean)")
    ap.add_argument("--sponge-density", type=float, default=120.0)
    ap.add_argument("--sponge-len", type=float, default=SPONGE[2],
                    help="sponge length along z (m). The 20 mm pads only compress ~1/5 of a 100 mm sponge (8 ml per wring); "
                         "60 mm puts ~1/3 of the block between the pads")
    ap.add_argument("--water-h", type=float, default=WATER_H,
                    help="water depth in the basin (m). 0.04 leaves the top third of the pinched sponge above the surface once "
                         "the sponge has drawn ~180 ml out of the basin; 0.055 keeps it under water during the soak")
    ap.add_argument("--sponge-poisson", type=float, default=0.35,
                    help="tet energy is the stable neo-Hookean WITHOUT the log term (USE_SNK1): below nu ~0.17 total collapse "
                         "(J=0) costs less than rest and at 50 %% pad squeeze no stable state exists (full10-12 collapsed to a "
                         "1 mm sheet); 0.35 loses volume with ~no lateral bulge, 0.45 bulges 30-50 %% and slips out of the pads")
    ap.add_argument("--sponge-cell", type=float, default=0.0125)
    ap.add_argument("--liquid-dx", type=float, default=0.006)
    ap.add_argument("--liquid-ppc", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0, help="water seeding jitter seed (a different seed = a different trial of the same command)")
    ap.add_argument("--no-water", action="store_true")
    ap.add_argument("--no-porous", action="store_true", help="plain contact between sponge and water (baseline)")
    ap.add_argument("--porous-drag", type=float, default=200.0, help="1/s relaxation of pore water toward the sponge velocity")
    ap.add_argument("--porous-capacity", type=float, default=0.85, help="pore volume fraction at rest")
    ap.add_argument("--porous-suction", type=float, default=2.0, help="m/s^2 inward pull on free water touching an unsaturated sponge")
    ap.add_argument("--wring", type=int, default=3)
    ap.add_argument("--regrasp", default="", help="regrasp cycles: comma-separated grasp offsets along the sponge (m) per wring, e.g. -0.03,0,0.03; "
                    "each wring then squeezes a different column (release, regrasp, soak between wrings)")
    ap.add_argument("--no-pad-water-contact", action="store_true", help="exclude finger pads <-> water particle contacts (diagnostic)")
    ap.add_argument("--newton-cap", type=int, default=200)
    ap.add_argument("--max-time", type=float, default=0.0)
    ap.add_argument("--save-every", type=float, default=2.0)
    ap.add_argument("--log-level", type=int, default=0)
    args = ap.parse_args()

    squeeze_gaps = [float(v) for v in str(args.squeeze_gap).split(",")]
    WATER_H = args.water_h
    SPONGE[2] = args.sponge_len
    regrasp = [float(v) for v in args.regrasp.split(",")] if args.regrasp else None
    tl = timeline(len(regrasp) if regrasp else args.wring, hold_d=args.hold_gap / 2, squeeze_d=[g / 2 for g in squeeze_gaps], grip_dy=args.grip_dy,
                  hold_s=args.wring_hold, regrasp=regrasp)
    T_total = tl.total
    if args.dry:
        arm = Arm("arm", arm_base_T(*ARM_BASE)); q = arm.kin.HOME.copy(); worst = 0.0; wdq = 0.0
        q, ok = arm.kin.ik(tl.segs[0].T_to, q, iters=300)
        for t in np.arange(0.0, T_total, 0.05):
            seg, s = tl.at(t); T = seg.pose(s); qn, ok = arm.kin.ik(T, q, iters=80)
            worst = max(worst, np.linalg.norm(arm.kin.fk(qn)[:3, 3] - T[:3, 3])); wdq = max(wdq, np.abs(qn - q).max()); q = qn
        print(f"[dry] worst position error {worst*1e3:.2f} mm, max dq/50ms {wdq:.3f} rad over {T_total:.1f} s")
        v, tt = box_tets(SPONGE, args.sponge_cell); print(f"[dry] sponge tets {len(tt)}, verts {len(v)}")
        return

    os.makedirs(args.out, exist_ok=True)
    sc = Scene(dict(dt=args.dt, preconditioner_type=(0 if not args.no_water else 1), newton_iter_cap=args.newton_cap, poisson_rate=args.sponge_poisson,
                    collision_detection_buff_scale=12.0, absolute_dhat=1e-3), log_level=args.log_level, tactile=args.tactile)
    probe = Arm("arm", arm_base_T(*ARM_BASE))
    q0, ok = probe.kin.ik(tl.segs[0].T_to, probe.kin.HOME, iters=300)
    arm = sc.add_arm("arm", ARM_BASE[0], ARM_BASE[1], q0=q0)
    arm.stiff = args.finger_stiff; arm.finger_v = args.finger_speed
    add_basin(sc, "bowl", BOWL_C, BOWL_IN, color=(0.80, 0.82, 0.85))
    add_basin(sc, "cup", CUP_C, CUP_IN, color=(0.95, 0.95, 0.97))
    sc.add_tactile_pads()          # FEM gels after the last ABD body
    v, tets = box_tets(SPONGE, args.sponge_cell)
    sp = sc.add_fem("sponge", v, tets, transform=T_of(None, SPONGE_C0), young=args.sponge_young,
                    density=args.sponge_density, color=(0.98, 0.80, 0.25))
    recs = sc.eng.get_load_records()
    sc.eng.set_body_friction(len(recs) - 1, args.grip_mu)                       # wet foam
    for i, r in enumerate(recs):
        if "finger" in str(r.label):
            sc.eng.set_body_friction(i, args.grip_mu)                          # rubber pads
    n_liq = 0
    if not args.no_water:
        pts, hs = seed_water(args.liquid_dx, args.liquid_ppc, np.random.default_rng(args.seed))
        lo = np.minimum(BOWL_C, CUP_C) - np.array([0.16, 0.02, 0.12]); hi = np.maximum(BOWL_C, CUP_C) + np.array([0.16, 0.40, 0.12])
        sc.eng.add_liquid(pts, args.liquid_dx, lo, hi, rho=1000.0, kappa=1e5, particle_volume=hs ** 3,
                          node_pool=120000, push_apart=0.9, j_mode=0, stick_h=0.0)
        n_liq = len(pts)
        print(f"[scene] water: {n_liq} particles ({n_liq*hs**3*1e6:.0f} ml), spacing {hs*1e3:.1f} mm")
        if args.no_pad_water_contact:
            st = sc.eng.get_liquid_stats(); pb = int(st["particle_body_id"])
            from common import FINGER_L, FINGER_R
            for nm in (FINGER_L, FINGER_R):
                sc.eng.add_collision_exclusion(int(arm.bodies[nm].body_id), pb)
            print(f"[scene] pad<->water contact excluded (particle body {pb}, fingers {[int(arm.bodies[n].body_id) for n in (FINGER_L, FINGER_R)]})")
    sc.finalize()
    if n_liq and not args.no_porous:
        sc.eng.set_liquid_porous(int(sp["body_id"]), args.porous_drag, args.porous_capacity, args.porous_suction)
        print(f"[scene] porous coupling on sponge FEM body {sp['body_id']} (liquid stats {sc.eng.get_liquid_stats()})")
    rec = Recorder(sc)
    print(f"[scene] sponge verts {sp['n']} tets {len(tets)}; bodies {len(sc.bodies)}; total {T_total:.1f} s at dt {args.dt}")
    arm.drive(sc.eng, arm.q); arm.finger_goal = 0.0
    rec_every = max(1, int(round(1.0 / (args.fps * args.dt))))
    n_steps = int(round(T_total / args.dt))
    if args.max_time > 0:
        n_steps = min(n_steps, int(round(args.max_time / args.dt)))
    meta = dict(example="sponge_transfer", dt=args.dt, fps=args.fps, sponge_young=args.sponge_young, n_liquid=n_liq, seed=args.seed,
                hold_gap=args.hold_gap, squeeze_gap=squeeze_gaps, wring_hold=args.wring_hold, water_h=args.water_h, sponge_len=args.sponge_len, finger_stiff=args.finger_stiff, wring_speed=args.wring_speed,
                porous=not args.no_porous, liquid_dx=args.liquid_dx, fem_materials={"sponge": "foam"})
    rec_path = osp.join(args.out, "recording.npz")
    save_every = max(1, int(round(args.save_every / args.dt))) if args.save_every > 0 else 0
    t0 = time.time(); last = None
    # closed-loop regrasp: the sponge drifts a few mm per put-back (and the pads push it while descending), so each regrasp
    # is re-aimed at the sponge's MEASURED centre + its offset; open-loop, regrasp4_v3's 4th grasp landed on the sponge's
    # end (+51 mm instead of +36) and the block slipped out of the pads during the wring
    reaim = {}
    if regrasp:
        for i in range(1, len(regrasp)):
            reaim[f"regrasp {i+1}: move along the sponge"] = [f"regrasp {i+1}: move along the sponge", f"regrasp {i+1}: descend",
                                                              f"regrasp {i+1}: grasp", f"regrasp {i+1}: soak", f"lift {i+1} out of the water"]
    meta["regrasp"] = regrasp; meta["reaim_mm"] = []
    for k in range(n_steps + 1):
        t = k * args.dt
        seg, s = tl.at(t); T = seg.pose(s)
        q, ok = arm.kin.ik(T, arm.q, iters=40); arm.drive(sc.eng, q)
        if seg.finger is not None:
            arm.finger_goal = seg.finger
        arm.finger_v = args.wring_speed if seg.label.startswith("wring") else args.finger_speed
        if seg.label != last and seg.label in reaim:
            c = sc.fem_verts("sponge").mean(0)
            dd = np.array([c[0] - SPONGE_C0[0], 0.0, c[2] - SPONGE_C0[2]])
            tl.shift(reaim[seg.label], dd); meta["reaim_mm"].append([round(float(dd[0]) * 1e3, 1), round(float(dd[2]) * 1e3, 1)])
            print(f"[regrasp] sponge centre x {c[0]:.3f} z {c[2]:+.3f} -> grasp re-aimed by ({dd[0]*1e3:+.1f}, {dd[2]*1e3:+.1f}) mm", flush=True)
        if seg.label != last:
            extra = ""
            if n_liq:
                P = sc.eng.get_liquid_particles(); in_cup = np.sum((np.abs(P[:, 0] - CUP_C[0]) < CUP_IN[0] / 2) & (np.abs(P[:, 2] - CUP_C[2]) < CUP_IN[2] / 2) & (P[:, 1] < CUP_C[1] + CUP_IN[1]))
                extra = f" water in cup {in_cup*hs**3*1e6:.1f} ml, in sponge {sc.eng.count_liquid_in_porous()*hs**3*1e6:.1f} ml (sat {sc.eng.get_liquid_porous_saturation():.2f})"
            print(f"[{t:6.2f}s] {seg.label}  ({(time.time()-t0)/60:.1f} min){extra}", flush=True); last = seg.label
        if k % rec_every == 0:
            rec.record(t, 0, seg.label, liquid_pts=sc.eng.get_liquid_particles() if n_liq else None)
        if save_every and k and k % save_every == 0:
            rec.save(rec_path, dict(meta, partial=True, wall_min=(time.time() - t0) / 60))
        sc.step()
    meta["wall_min"] = (time.time() - t0) / 60
    rec.save(rec_path, meta)
    json.dump(meta, open(osp.join(args.out, "meta.json"), "w"), indent=1)
    print(f"[done] {n_steps} steps in {(time.time()-t0)/60:.1f} min -> {rec_path}")


if __name__ == "__main__":
    main()

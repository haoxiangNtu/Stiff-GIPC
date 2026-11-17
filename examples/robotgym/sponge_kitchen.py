#!/usr/bin/env python3
"""RobotGym sponge task in the user's kitchen (厨房.fbx, real scale; make_kitchen_fbx.py): the left bowl of the double sink
holds water (drain stopped), a FR3 on a platform in front of the island picks the soaked sponge out of it, carries it over
the right bowl and wrings it there (or regrasps and wrings several columns). Same sponge, porous coupling and liquid
settings as the sponge_transfer.py deliverables; the scene is the FBX kitchen's own triangles (one fixed body).

    PYTHONPATH=. python examples/robotgym/sponge_kitchen.py --dry
    PYTHONPATH=. python examples/robotgym/sponge_kitchen.py --out out/robotgym/kitchen_sponge     # pod, deliverable env
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
from common import (ASSETS, R_cols, Recorder, Scene, T_of, Timeline, Arm, arm_base_T, load_obj)  # noqa: E402
from sponge_transfer import box_tets  # noqa: E402

KDIR = osp.join(ASSETS, "robotgym", "fbx", "kitchen")
KM = json.load(open(osp.join(KDIR, "kitchen_meta.json")))
BA, BB = KM["bowls"]["a"], KM["bowls"]["b"]
SPONGE = np.array([0.10, 0.035, 0.06])     # kitchen sponge, long side along x; the pads close on the 60 mm width (z)


def hand_R_topdown():
    """hand z down, fingers (hand y) along world z: the pads close on the sponge's 60 mm width; the 204 mm hand lies along z,
    inside the bowl's 318 mm z-span when it dips below the rim (centred grasp leaves 57 mm each side)"""
    zh = np.array([0.0, -1.0, 0.0]); yh = np.array([0.0, 0.0, 1.0]); xh = np.cross(yh, zh)
    return R_cols(xh, yh, zh)


def sponge_c0(a):
    return np.array([a.sponge_x, BA["bottom_y"] + 0.5 * SPONGE[1] + 0.004, 0.5 * (BA["interior_lo"][1] + BA["interior_hi"][1])])


def timeline(a, n_wring, open_d, hold_d, squeeze_d, regrasp=None):
    R = hand_R_topdown()
    P = lambda x, y, z: T_of(R, (x, y, z))
    c0 = sponge_c0(a)
    gx, gz = c0[0], c0[2]
    y_grip = c0[1] + a.grip_dy
    y_lift = BA["rim_y"] + a.clear + 0.5 * SPONGE[1] + 0.0                      # sponge bottom `clear` above the rim
    wz = 0.5 * (BB["interior_lo"][1] + BB["interior_hi"][1])
    wx = a.wring_x
    y_wring = BB["rim_y"] + a.wring_clear + 0.5 * SPONGE[1]
    dx0 = regrasp[0] if regrasp else 0.0
    tl = Timeline(P(gx + dx0, y_lift, gz))
    tl.add("settle (open fingers)", 1.5, finger=open_d)
    tl.add("descend into the left bowl", 3.0, P(gx + dx0, y_grip, gz))
    tl.add("grasp", 2.0, finger=hold_d)
    tl.add("soak", 1.0)
    if regrasp:
        for i, dx in enumerate(regrasp):
            sq = squeeze_d[min(i, len(squeeze_d) - 1)]
            if i > 0:
                tl.add(f"regrasp {i+1}: move along the sponge", 1.5, P(gx + dx, y_grip + 0.5 * (y_lift - y_grip), gz))
                tl.add(f"regrasp {i+1}: descend", 1.5, P(gx + dx, y_grip, gz))
                tl.add(f"regrasp {i+1}: grasp", 2.0, finger=hold_d)
                tl.add(f"regrasp {i+1}: soak", 1.0)
            tl.add(f"lift {i+1} out of the water", 2.0, P(gx + dx, y_lift, gz))
            tl.add(f"carry {i+1} over the right bowl", 2.5, P(wx + dx, y_wring, wz))
            tl.add(f"wring {i+1}", a.wring_s, finger=sq)
            tl.add(f"hold {i+1}", a.wring_hold)
            tl.add(f"relax {i+1}", 1.5, finger=hold_d)
            tl.add(f"carry back {i+1}", 2.5, P(gx + dx, y_lift, gz))
            tl.add(f"lower {i+1} into the left bowl", 2.0, P(gx + dx, y_grip, gz))
            tl.add(f"release {i+1}", 1.5, finger=open_d)
        tl.add("retreat", 2.0, P(gx + regrasp[-1], y_lift + 0.05, gz))
        tl.add("rest", 1.0)
        return tl
    tl.add("lift out of the water", 2.0, P(gx, y_lift, gz))
    tl.add("carry over the right bowl", 2.5, P(wx, y_wring, wz))
    for i in range(n_wring):
        tl.add(f"wring {i+1}", a.wring_s, finger=squeeze_d[min(i, len(squeeze_d) - 1)])
        tl.add(f"hold {i+1}", a.wring_hold)
        tl.add(f"relax {i+1}", 1.5, finger=hold_d)
    tl.add("carry back", 2.5, P(gx, y_lift, gz))
    tl.add("lower into the left bowl", 2.0, P(gx, y_grip, gz))
    tl.add("release", 1.5, finger=open_d)
    tl.add("retreat", 2.0, P(gx, y_lift + 0.05, gz))
    tl.add("rest", 1.0)
    return tl


def inside_bowl(P, bowl, margin):
    """points (N,3) inside a bowl's water space: in the mid-depth outline (inset by margin), above the bottom by margin, outside
    the bottom fillet (radius r_f) and the strainer basket"""
    poly = np.asarray(bowl["outline_xz"]); x, z = P[:, 0], P[:, 2]
    inside = np.zeros(len(P), bool)
    j = len(poly) - 1
    for i in range(len(poly)):                                    # even-odd ray casting
        xi, zi = poly[i]; xj, zj = poly[j]
        cross = ((zi > z) != (zj > z)) & (x < (xj - xi) * (z - zi) / (zj - zi + 1e-300) + xi)
        inside ^= cross; j = i
    # distance to the outline
    a_ = poly; b_ = np.roll(poly, -1, 0); ab = b_ - a_
    d = np.full(len(P), np.inf)
    for k in range(len(poly)):
        t = np.clip(((x - a_[k, 0]) * ab[k, 0] + (z - a_[k, 1]) * ab[k, 1]) / (ab[k] @ ab[k] + 1e-300), 0, 1)
        d = np.minimum(d, np.hypot(x - (a_[k, 0] + t * ab[k, 0]), z - (a_[k, 1] + t * ab[k, 1])))
    h = P[:, 1] - bowl["bottom_y"]
    rf = bowl["fillet_r"]
    fillet_ok = ~((d < rf) & (h < rf) & ((rf - d) ** 2 + (rf - h) ** 2 > (rf - margin) ** 2))
    dr = np.hypot(x - bowl["drain_xz"][0], z - bowl["drain_xz"][1])
    strainer = (dr < bowl["drain_r"] + margin) & (P[:, 1] < bowl["strainer_top_y"] + margin)
    return inside & (d >= margin) & (h >= margin) & fillet_ok & ~strainer


def seed_water(a, rng):
    n_sub = max(1, int(round(a.liquid_ppc ** (1.0 / 3.0)))); hs = a.liquid_dx / n_sub; m = 0.6 * a.liquid_dx
    lo = np.array([BA["interior_lo"][0], BA["bottom_y"], BA["interior_lo"][1]]); hi = np.array([BA["interior_hi"][0], BA["bottom_y"] + a.water_h, BA["interior_hi"][1]])
    ax = [np.arange(lo[k], hi[k], hs) for k in range(3)]
    g = np.meshgrid(*ax, indexing="ij")
    pts = np.stack([v.ravel() for v in g], 1) + rng.uniform(-0.3 * hs, 0.3 * hs, (g[0].size, 3))
    pts = pts[inside_bowl(pts, BA, m)]
    c0 = sponge_c0(a)
    sp_lo, sp_hi = c0 - SPONGE / 2 - m, c0 + SPONGE / 2 + m
    return pts[~np.all((pts > sp_lo) & (pts < sp_hi), 1)], hs


def main():
    global SPONGE
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/robotgym/kitchen_sponge")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--clearance", action="store_true", help="with --dry: link-to-kitchen clearance along the path")
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--base", type=float, nargs=3, default=[0.53, 0.78, 0.0], help="arm base (x y z): on a platform in front of the island")
    ap.add_argument("--sponge-x", type=float, default=0.03, help="sponge centre x in the left bowl (drain at x -0.131)")
    ap.add_argument("--wring-x", type=float, default=0.03, help="x over the right bowl where the sponge is wrung")
    ap.add_argument("--clear", type=float, default=0.04, help="sponge bottom above the rim when lifted (m)")
    ap.add_argument("--wring-clear", type=float, default=0.05, help="sponge bottom above the right bowl's rim while wringing (m)")
    ap.add_argument("--grip-dy", type=float, default=0.0)
    ap.add_argument("--sponge-size", type=float, nargs=3, default=[0.10, 0.035, 0.06], help="sponge x (long side), y (thickness), z (pad direction) (m)")
    ap.add_argument("--sponge-young", type=float, default=5e4)
    ap.add_argument("--sponge-poisson", type=float, default=0.35)
    ap.add_argument("--sponge-density", type=float, default=120.0)
    ap.add_argument("--sponge-cell", type=float, default=0.00625)
    ap.add_argument("--squeeze-gap", default="0.024,0.018,0.012")
    ap.add_argument("--hold-gap", type=float, default=0.044)
    ap.add_argument("--wring", type=int, default=3)
    ap.add_argument("--wring-s", type=float, default=2.5)
    ap.add_argument("--wring-hold", type=float, default=2.0)
    ap.add_argument("--regrasp", default="", help="regrasp offsets along the sponge (x, m), e.g. -0.03,-0.01,0.01,0.03")
    ap.add_argument("--finger-stiff", type=float, default=25.0)
    ap.add_argument("--finger-speed", type=float, default=0.03)
    ap.add_argument("--wring-speed", type=float, default=0.012)
    ap.add_argument("--grip-mu", type=float, default=1.0)
    ap.add_argument("--water-h", type=float, default=0.040, help="water depth in the left bowl (m); the sponge is 35 mm thick")
    ap.add_argument("--liquid-dx", type=float, default=0.008)
    ap.add_argument("--liquid-ppc", type=float, default=8.0)
    ap.add_argument("--node-pool", type=int, default=300000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-water", action="store_true")
    ap.add_argument("--porous-drag", type=float, default=200.0)
    ap.add_argument("--porous-capacity", type=float, default=0.85)
    ap.add_argument("--porous-suction", type=float, default=2.0)
    ap.add_argument("--pedestal", type=float, nargs=3, default=[0.30, 0.0, 0.36], help="platform footprint x, (unused), z (m)")
    ap.add_argument("--newton-cap", type=int, default=200)
    ap.add_argument("--max-time", type=float, default=0.0)
    ap.add_argument("--save-every", type=float, default=2.0)
    ap.add_argument("--log-level", type=int, default=0)
    args = ap.parse_args()
    SPONGE = np.array(args.sponge_size)

    sq = [float(v) / 2 for v in str(args.squeeze_gap).split(",")]
    regrasp = [float(v) for v in args.regrasp.split(",")] if args.regrasp else None
    tl = timeline(args, len(regrasp) if regrasp else args.wring, 0.039, args.hold_gap / 2, sq, regrasp)
    base = tuple(args.base)
    if args.dry:
        arm = Arm("arm", arm_base_T(base, 180.0)); q = arm.kin.HOME.copy(); worst = 0.0; wdq = 0.0
        q, ok = arm.kin.ik(tl.segs[0].T_to, q, iters=300)
        for t in np.arange(0.0, tl.total, 0.05):
            seg, s = tl.at(t); T = seg.pose(s); qn, ok = arm.kin.ik(T, q, iters=80)
            err = np.linalg.norm(arm.kin.fk(qn)[:3, 3] - T[:3, 3]); worst = max(worst, err); wdq = max(wdq, np.abs(qn - q).max()); q = qn
            if err > 2e-3:
                print(f"[dry] t {t:5.2f} {seg.label:34s} IK err {err*1e3:.1f} mm")
        if args.clearance:
            from common import arm_clearance
            Vk, Fk = load_obj(osp.join(KDIR, "kitchen_collision.obj"))
            arm_clearance(arm, tl, Vk, Fk)
        v, tt = box_tets(SPONGE, args.sponge_cell)
        pts, hs = seed_water(args, np.random.default_rng(args.seed))
        print(f"[dry] worst IK error {worst*1e3:.2f} mm, max dq/50ms {wdq:.3f} rad over {tl.total:.1f} s; sponge {len(v)} verts {len(tt)} tets; "
              f"water {len(pts)} particles ({len(pts)*hs**3*1e6:.0f} ml), spacing {hs*1e3:.1f} mm")
        return

    os.makedirs(args.out, exist_ok=True)
    sc = Scene(dict(dt=args.dt, preconditioner_type=(0 if not args.no_water else 1), newton_iter_cap=args.newton_cap, poisson_rate=args.sponge_poisson,
                    collision_detection_buff_scale=12.0, absolute_dhat=1e-3), log_level=args.log_level)
    probe = Arm("arm", arm_base_T(base, 180.0))
    q0, ok = probe.kin.ik(tl.segs[0].T_to, probe.kin.HOME, iters=300)
    arm = sc.add_arm("arm", base, 180.0, q0=q0)
    arm.stiff = args.finger_stiff; arm.finger_v = args.finger_speed
    Vk, Fk = load_obj(osp.join(KDIR, "kitchen_collision.obj"))
    kit = sc.add_static_mesh("kitchen", Vk, Fk, color=(0.78, 0.79, 0.80), mu=0.5)
    ped = sc.add_box("pedestal", (base[0], (base[1] - 0.002) / 2, base[2]), (args.pedestal[0], base[1] - 0.002, args.pedestal[2]),
                     fixed=True, color=(0.20, 0.20, 0.22), cell=0.06)
    for b in arm.bodies.values():
        sc.eng.add_collision_exclusion(ped.body_id, b.body_id)
    sc.eng.add_collision_exclusion(ped.body_id, kit.body_id)
    c0 = sponge_c0(args)
    v, tets = box_tets(SPONGE, args.sponge_cell)
    sp = sc.add_fem("sponge", v, tets, transform=T_of(None, c0), young=args.sponge_young, density=args.sponge_density, color=(0.98, 0.80, 0.25))
    recs = sc.eng.get_load_records()
    sc.eng.set_body_friction(len(recs) - 1, args.grip_mu)
    for i, r in enumerate(recs):
        if "finger" in str(r.label):
            sc.eng.set_body_friction(i, args.grip_mu)
    n_liq = 0; hs = 0.0
    if not args.no_water:
        pts, hs = seed_water(args, np.random.default_rng(args.seed))
        lo = np.array([-0.30, BA["bottom_y"] - 0.02, -0.44]); hi = np.array([0.36, BA["rim_y"] + 0.30, 0.44])
        sc.eng.add_liquid(pts, args.liquid_dx, lo, hi, rho=1000.0, kappa=1e5, particle_volume=hs ** 3,
                          node_pool=args.node_pool, push_apart=0.9, j_mode=0, stick_h=0.0)
        n_liq = len(pts)
        print(f"[scene] water: {n_liq} particles ({n_liq*hs**3*1e6:.0f} ml, {args.water_h*1e3:.0f} mm deep in the left bowl), spacing {hs*1e3:.1f} mm")
    sc.finalize()
    if n_liq:
        sc.eng.set_liquid_porous(int(sp["body_id"]), args.porous_drag, args.porous_capacity, args.porous_suction)
        print(f"[scene] porous coupling on sponge FEM body {sp['body_id']} (liquid stats {sc.eng.get_liquid_stats()})")
    rec = Recorder(sc)
    print(f"[scene] kitchen {len(Fk)} faces; sponge {sp['n']} verts {len(tets)} tets at {np.round(c0, 4).tolist()}; total {tl.total:.1f} s at dt {args.dt}")
    arm.drive(sc.eng, arm.q); arm.finger_goal = 0.0
    rec_every = max(1, int(round(1.0 / (args.fps * args.dt))))
    n_steps = int(round(tl.total / args.dt))
    if args.max_time > 0:
        n_steps = min(n_steps, int(round(args.max_time / args.dt)))
    meta = dict(example="sponge_kitchen", dt=args.dt, fps=args.fps, n_liquid=n_liq, liquid_dx=args.liquid_dx,
                args={k: val for k, val in vars(args).items() if isinstance(val, (int, float, str, bool))},
                bowl_b=dict(lo=BB["interior_lo"], hi=BB["interior_hi"], rim_y=BB["rim_y"]), fem_materials={"sponge": "foam"}, sponge_size=SPONGE.tolist())
    rec_path = osp.join(args.out, "recording.npz")
    save_every = max(1, int(round(args.save_every / args.dt)))
    t0 = time.time(); last = None
    def in_bowl_b(P):
        return np.sum((P[:, 0] > BB["interior_lo"][0]) & (P[:, 0] < BB["interior_hi"][0]) & (P[:, 2] > BB["interior_lo"][1]) &
                      (P[:, 2] < BB["interior_hi"][1]) & (P[:, 1] < BB["rim_y"]))
    for k in range(n_steps + 1):
        t = k * args.dt
        seg, s = tl.at(t); T = seg.pose(s)
        q, ok = arm.kin.ik(T, arm.q, iters=40); arm.drive(sc.eng, q)
        if seg.finger is not None:
            arm.finger_goal = seg.finger
        arm.finger_v = args.wring_speed if seg.label.startswith("wring") else args.finger_speed
        if seg.label != last:
            extra = ""
            if n_liq:
                P_ = sc.eng.get_liquid_particles()
                extra = (f" water in the right bowl {in_bowl_b(P_)*hs**3*1e6:.1f} ml, in sponge {sc.eng.count_liquid_in_porous()*hs**3*1e6:.1f} ml "
                         f"(sat {sc.eng.get_liquid_porous_saturation():.2f})")
            print(f"[{t:6.2f}s] {seg.label}  ({(time.time()-t0)/60:.1f} min){extra}", flush=True); last = seg.label
        if k % rec_every == 0:
            rec.record(t, 0, seg.label, liquid_pts=sc.eng.get_liquid_particles() if n_liq else None)
        if save_every and k and k % save_every == 0:
            rec.save(rec_path, dict(meta, partial=True, wall_min=(time.time() - t0) / 60))
        sc.step()
    meta["wall_min"] = (time.time() - t0) / 60
    if n_liq:
        meta["water_right_bowl_ml"] = float(in_bowl_b(sc.eng.get_liquid_particles()) * hs ** 3 * 1e6)
    rec.save(rec_path, meta)
    json.dump(meta, open(osp.join(args.out, "meta.json"), "w"), indent=1)
    print(f"[done] {n_steps} steps in {meta['wall_min']:.1f} min -> {rec_path}")


if __name__ == "__main__":
    main()

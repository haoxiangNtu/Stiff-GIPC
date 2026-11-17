#!/usr/bin/env python3
"""RobotGym showcase 2 — PET bottle squeeze: a 0.3 mm plastic shell (IPC triangle shell, PET stiffness, plastic
hinges) part-filled with water (engine MPM liquid, two-way coupled through the shell), squeezed by a FR3 gripper.
The shell buckles under the fingers and the dents stay after release (bend_yield_angle > 0, needs the engine
built with -DGIPC_QUADRATIC_BENDING=OFF, i.e. dihedral bending).

    PYTHONPATH=. python examples/robotgym/bottle_squeeze.py --out out/robotgym/bottle          # pod (build_shell)
    PYTHONPATH=. python examples/robotgym/bottle_squeeze.py --dry                              # IK check only
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

BOTTLE_POS = np.array([0.45, 0.0, 0.0])      # base centre on the table (ground plane y = 0)
BOTTLE_R, BOTTLE_H = 0.032, 0.20              # must match make_bottle_shell.py defaults
SQUEEZE_Y = 0.095                             # finger centre height on the body (body spans y 0.006..0.165)
GRIP_X = BOTTLE_POS[0] - 0.010                # TCP 1 cm short of the bottle axis (palm stays 2 cm off the wall)
BACK_X = 0.30                                 # retracted hand x
ARM_BASE = ((-0.30, 0.0, 0.0), 0.0)           # FR3 30 cm behind the origin, facing +x
HAND_PITCH = -40.0                            # approach tilted down (deg) about the finger axis; IK sweep 2026-10-18: 0.2 mm
START_Y = 0.18


def hand_R_side():
    """Hand z (approach) along +x tilted HAND_PITCH down, fingers (hand y) along world z: the pads close on the
    bottle from both sides. (A top-down grasp is impossible: the fingers are 58 mm long, the bottle 225 mm tall.)"""
    from common import axis_angle_R
    zh = np.array([1.0, 0.0, 0.0])
    yh = np.array([0.0, 0.0, 1.0])
    xh = np.cross(yh, zh)
    return axis_angle_R([0, 0, 1], np.radians(HAND_PITCH)) @ R_cols(xh, yh, zh)


def timeline(open_gap=0.078, squeeze_gap=0.040, squeeze_time=2.5):
    R = hand_R_side()
    P = lambda x, y, z: T_of(R, (x, y, z))
    bz = BOTTLE_POS[2]
    tl = Timeline(P(BACK_X, START_Y, bz))
    tl.add("settle (open fingers)", 1.5, finger=open_gap / 2)
    tl.add("lower to grip height", 2.0, P(BACK_X, SQUEEZE_Y, bz))
    tl.add("approach the bottle", 2.5, P(GRIP_X, SQUEEZE_Y, bz))
    tl.add("squeeze", squeeze_time, finger=squeeze_gap / 2)
    tl.add("hold", 1.5)
    tl.add("release", 1.5, finger=open_gap / 2)
    tl.add("retreat", 2.0, P(BACK_X, SQUEEZE_Y, bz))
    tl.add("rest", 1.5)
    return tl


def seed_water(fill_h, dx, ppc, rng, margin=0.6, shape=None):
    n_sub = max(1, int(round(ppc ** (1.0 / 3.0))))
    hs = dx / n_sub
    m = margin * dx                                             # clearance to the shell (fraction of dx)
    cx, cz = BOTTLE_POS[0], BOTTLE_POS[2]
    if shape is not None:                                       # FBX bottle: height maps R(phi, y) / y_base(u, phi) (make_bottle_fbx.py)
        rmax = float(shape.R.max())
        ax = np.arange(cx - rmax, cx + rmax, hs); ay = np.arange(m, fill_h, hs); az = np.arange(cz - rmax, cz + rmax, hs)
        gx, gy, gz = np.meshgrid(ax, ay, az, indexing="ij")
        pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)
        pts = pts + rng.uniform(-0.3 * hs, 0.3 * hs, pts.shape)
        loc = pts - np.array([cx, 0.0, cz])
        return pts[shape.inside(loc, m)], hs
    ax = np.arange(cx - BOTTLE_R + m, cx + BOTTLE_R - m, hs)
    ay = np.arange(0.006 + m, fill_h, hs)                      # bottle floor dome rises 6 mm at the rim
    az = np.arange(cz - BOTTLE_R + m, cz + BOTTLE_R - m, hs)
    gx, gy, gz = np.meshgrid(ax, ay, az, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)
    pts = pts + rng.uniform(-0.3 * hs, 0.3 * hs, pts.shape)
    pts = pts[np.hypot(pts[:, 0] - cx, pts[:, 2] - cz) < bottle_inner_r(pts[:, 1]) - m]   # inside the shell everywhere (shoulder, neck)
    return pts, hs


def bottle_inner_r(y, R=BOTTLE_R, H=BOTTLE_H, rn=0.0125, hn=0.025, hs_=0.035):
    """Radius profile of make_bottle_shell.py: body to H-hn-hs_, cosine shoulder to the neck radius, neck."""
    y = np.asarray(y, float); y1 = H - hn - hs_
    r = np.full_like(y, R)
    sh = (y > y1) & (y <= y1 + hs_); u = (y[sh] - y1) / hs_
    r[sh] = rn + (R - rn) * 0.5 * (1 + np.cos(np.pi * u))
    r[y > y1 + hs_] = rn
    r[y < 0.006 * (1 - np.minimum(1.0, r / R)) ** 2] = 0.0   # under the base dome (centre raised 6 mm)
    return r


def main():
    global SQUEEZE_Y
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/robotgym/bottle")
    ap.add_argument("--tactile", action="store_true", help="VBTS gel pads on the fingertips (examples/lab_tactile): "
                    "indentation images + normal forces recorded; gaps are pad-face gaps")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--bottle", default="robotgym/pet_bottle_5mm_v2.obj")
    ap.add_argument("--shape", default="", help="height-map npz of an FBX bottle (make_bottle_fbx.py; default <bottle>.npz if it exists): "
                    "water is seeded inside its real cross-section (rounded square, ribs, base push-up)")
    ap.add_argument("--squeeze-y", type=float, default=SQUEEZE_Y, help="finger centre height on the body (m)")
    ap.add_argument("--yaw", type=float, default=0.0, help="bottle rotation about its axis (deg); 0 = FBX flats facing the pads")
    ap.add_argument("--young", type=float, default=2.5e9, help="PET Young's modulus (Pa)")
    ap.add_argument("--thickness", type=float, default=3e-4)
    ap.add_argument("--shell-density", type=float, default=1380.0 * 3e-4, help="surface density kg/m^2")
    ap.add_argument("--yield-angle", type=float, default=0.20, help="plastic hinge yield (rad); 0 = elastic")
    ap.add_argument("--plastic-rate", type=float, default=1.0)
    ap.add_argument("--fill", type=float, default=0.16, help="water height (m); the shoulder starts at 0.14, the seeding follows the shell profile")
    ap.add_argument("--top-up", type=float, default=0.0, help="extra water (ml) seeded as a column over the mouth inside the neck radius; it drains in "
                    "during the first ~0.3 s while the seeded water settles ~5 %% lower (dx 5 mm cut cells in the 55 mm bottle), so a factory-full "
                    "level (water in the neck) needs it")
    ap.add_argument("--liquid-dx", type=float, default=0.005)
    ap.add_argument("--liquid-ppc", type=float, default=8.0)
    ap.add_argument("--seed-margin", type=float, default=0.6, help="seeding clearance to the shell, fraction of dx (0.6 = 3 mm; 0.3 fills the bottle fuller)")
    ap.add_argument("--domain-half", type=float, default=0.08, help="liquid domain half-width in x/z around the bottle (m); enlarge when water is squeezed out")
    ap.add_argument("--domain-height", type=float, default=0.30, help="liquid domain height above y = -0.02 (m)")
    ap.add_argument("--node-pool", type=int, default=60000)
    ap.add_argument("--liquid-ground-mu", type=float, default=0.3, help="friction of the water on the table (1.0 stops the spilled monolayer from creeping)")
    ap.add_argument("--no-water", action="store_true")
    ap.add_argument("--squeeze-gap", type=float, default=0.040)
    ap.add_argument("--squeeze-time", type=float, default=2.5, help="seconds for the pads to close (0.8 gives a jet out of the neck instead of an overflow)")
    ap.add_argument("--squeeze-speed", type=float, default=0.0, help="pad gap speed during the squeeze (m/s); 0 = the hand default 0.08 "
                    "(78 -> 18 mm in 0.75 s whatever --squeeze-time is; --squeeze-time only sets the segment length)")
    ap.add_argument("--newton-cap", type=int, default=200)
    ap.add_argument("--max-time", type=float, default=0.0)
    ap.add_argument("--log-level", type=int, default=0)
    ap.add_argument("--save-every", type=float, default=2.0, help="checkpoint the recording every N sim seconds")
    args = ap.parse_args()

    SQUEEZE_Y = args.squeeze_y
    shape = None
    shape_path = args.shape or osp.splitext(osp.join(ASSETS, args.bottle))[0] + ".npz"
    if osp.exists(shape_path):
        from make_bottle_fbx import Shape
        shape = Shape.load(shape_path)
        print(f"[scene] bottle shape {shape_path}: height {shape.H*1e3:.0f} mm, R {shape.R.min()*1e3:.1f}..{shape.R.max()*1e3:.1f} mm")
    tl = timeline(squeeze_gap=args.squeeze_gap, squeeze_time=args.squeeze_time)
    T_total = tl.total
    if args.dry:
        arm = Arm("arm", arm_base_T(*ARM_BASE))
        q = arm.kin.HOME.copy(); worst = 0.0
        for t in np.arange(0.0, T_total, 0.1):
            seg, s = tl.at(t); T = seg.pose(s); q, ok = arm.kin.ik(T, q, iters=120)
            worst = max(worst, np.linalg.norm(arm.kin.fk(q)[:3, 3] - T[:3, 3]))
        print(f"[dry] worst position error {worst*1e3:.2f} mm over {T_total:.1f} s")
        return

    os.makedirs(args.out, exist_ok=True)
    sc = Scene(dict(dt=args.dt, cloth_thickness=args.thickness, cloth_young_modulus=args.young,
                    bend_young_modulus=args.young, cloth_density=args.shell_density / args.thickness,
                    bend_yield_angle=args.yield_angle, bend_plastic_rate=args.plastic_rate,
                    preconditioner_type=(0 if not args.no_water else 1),   # engine liquid requires type 0 (no MAS reorder)
                    newton_iter_cap=args.newton_cap, collision_detection_buff_scale=12.0),
               log_level=args.log_level, tactile=args.tactile)
    probe = Arm("arm", arm_base_T(*ARM_BASE))
    q0, ok = probe.kin.ik(tl.segs[0].T_to, probe.kin.HOME, iters=200)
    arm = sc.add_arm("arm", ARM_BASE[0], ARM_BASE[1], q0=q0)
    sc.add_tactile_pads()          # FEM gels after the last ABD body
    from common import Ry
    bottle_T = T_of(Ry(args.yaw), BOTTLE_POS + np.array([0.0, 1.0e-3, 0.0]))
    if shape is not None and args.yaw != 0.0:
        raise SystemExit("--yaw with an FBX shape: the water seeding uses the unrotated height map")
    sc.add_cloth("bottle", osp.join(ASSETS, args.bottle), bottle_T, young=args.young, color=(0.85, 0.92, 1.0))
    n_liq = 0
    if not args.no_water:
        pts, hs = seed_water(args.fill, args.liquid_dx, args.liquid_ppc, np.random.default_rng(0), margin=args.seed_margin, shape=shape)
        n_top = 0
        if args.top_up > 0.0:
            if shape is None:
                raise SystemExit("--top-up needs the FBX height map (--shape)")
            m = args.seed_margin * args.liquid_dx
            top = float(shape.y[-1]) + 1.0e-3 + hs                     # one lattice step over the mouth (the shell sits 1 mm up)
            r_col = float(shape.R[-8:].min()) - m
            h_col = args.top_up * 1e-6 / (np.pi * r_col ** 2)
            g = np.arange(-r_col, r_col + 1e-9, hs)
            gx, gy, gz = np.meshgrid(g, np.arange(top, top + h_col, hs), g, indexing="ij")
            col = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)
            col = col + np.random.default_rng(1).uniform(-0.3 * hs, 0.3 * hs, col.shape)
            col = col[np.hypot(col[:, 0], col[:, 2]) < r_col] + np.array([BOTTLE_POS[0], 0.0, BOTTLE_POS[2]])
            n_top = len(col); pts = np.vstack([pts, col])
            print(f"[scene] top-up column: {n_top} particles ({n_top*hs**3*1e6:.0f} ml), r {r_col*1e3:.1f} mm, y {top*1e3:.0f}..{(top+h_col)*1e3:.0f} mm")
        o = BOTTLE_POS + np.array([-args.domain_half, -0.02, -args.domain_half]); ext = np.array([2 * args.domain_half, args.domain_height, 2 * args.domain_half])
        sc.eng.add_liquid(pts, args.liquid_dx, o, o + ext, rho=1000.0, kappa=1e5, particle_volume=hs ** 3,
                          node_pool=args.node_pool, push_apart=0.9, j_mode=0, stick_h=0.0, ground_mu=args.liquid_ground_mu)
        n_liq = len(pts)
        print(f"[scene] water: {n_liq} particles to {args.fill*1e3:.0f} mm ({n_liq*hs**3*1e6:.0f} ml)")
    sc.finalize()
    rec = Recorder(sc)
    print(f"[scene] shell verts {sc.fem['bottle']['n']}, faces {len(sc.fem['bottle']['faces'])}; E {args.young:.2g}, "
          f"t {args.thickness*1e3:.2f} mm, yield {args.yield_angle} rad; total {T_total:.1f} s at dt {args.dt}")
    arm.drive(sc.eng, arm.q); arm.finger_goal = 0.0
    rec_every = max(1, int(round(1.0 / (args.fps * args.dt))))
    n_steps = int(round(T_total / args.dt))
    if args.max_time > 0:
        n_steps = min(n_steps, int(round(args.max_time / args.dt)))
    meta = dict(example="bottle_squeeze", dt=args.dt, fps=args.fps, bottle=args.bottle, young=args.young, squeeze_y=args.squeeze_y,
                squeeze_gap=args.squeeze_gap, squeeze_time=args.squeeze_time, squeeze_speed=args.squeeze_speed, plastic_rate=args.plastic_rate,
                thickness=args.thickness, yield_angle=args.yield_angle, fill=args.fill, liquid_dx=args.liquid_dx,
                n_liquid=n_liq, fem_materials={"bottle": "pet"})
    rec_path = osp.join(args.out, "recording.npz")
    save_every = max(1, int(round(args.save_every / args.dt))) if args.save_every > 0 else 0
    t0 = time.time(); last = None
    stat_every = max(1, int(round(0.25 / args.dt))); nt_prev = int(sc.eng.native.get_total_newton_iters())
    pc_prev = float(getattr(sc.eng.native, "get_total_pcg_iters", lambda: 0.0)())
    for k in range(n_steps + 1):
        t = k * args.dt
        seg, s = tl.at(t); T = seg.pose(s)
        q, ok = arm.kin.ik(T, arm.q, iters=40); arm.drive(sc.eng, q)
        if seg.finger is not None:
            arm.finger_goal = seg.finger
        arm.finger_v = args.squeeze_speed if (args.squeeze_speed > 0 and seg.label == "squeeze") else 0.08
        if seg.label != last:
            print(f"[{t:6.2f}s] {seg.label}  ({(time.time()-t0)/60:.1f} min)", flush=True); last = seg.label
        if k and k % stat_every == 0:     # solver health: Newton / PCG per step over the last 0.25 s
            nt = int(sc.eng.native.get_total_newton_iters()); pc = float(getattr(sc.eng.native, "get_total_pcg_iters", lambda: 0.0)())
            print(f"  [solver {t:6.2f}s] newton {(nt-nt_prev)/stat_every:5.1f}/step, pcg {(pc-pc_prev)/max(nt-nt_prev,1):6.0f}/newton  "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True); nt_prev, pc_prev = nt, pc
        if k % rec_every == 0:
            rec.record(t, 0, seg.label, liquid_pts=sc.eng.get_liquid_particles() if n_liq else None)
        if save_every and k and k % save_every == 0:
            rec.save(rec_path, dict(meta, partial=True, wall_min=(time.time() - t0) / 60))
        sc.step()
    meta["wall_min"] = (time.time() - t0) / 60
    rec.save(rec_path, meta)
    json.dump(meta, open(osp.join(args.out, "meta.json"), "w"), indent=1)
    print(f"[done] {n_steps} steps in {(time.time()-t0)/60:.1f} min -> {args.out}/recording.npz")


if __name__ == "__main__":
    main()

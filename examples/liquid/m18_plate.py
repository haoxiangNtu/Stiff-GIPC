"""M18: a kinematic plate pushed into still water (moving-solid contact check for the liquid particle proxies).

The m2 box holds 0.6 L of water at rest. A 40 x 40 x 5 mm plate (ABD Free, huge density, gravity off, teleported every
step like the m9 piston / m4 beaker) starts 5 mm above the surface, moves straight down at --speed until its bottom is
--depth below the initial level, holds, then rises back out. Reported every step: particles inside the plate volume,
the closest particle-to-bottom-face distance, the water level. A correct IPC coupling keeps the inside count at 0.

    python examples/liquid/m18_plate.py --dx 0.005 --ppc 8 --out out/liquid_m18
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from stiff_physics.engine import Engine, Config  # noqa: E402
from m2_box import cuboid, open_box  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=0.005)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--ppc", type=int, default=8)
    ap.add_argument("--h", type=float, default=0.0)
    ap.add_argument("--speed", type=float, default=0.02, help="plate speed (m/s)")
    ap.add_argument("--depth", type=float, default=0.015, help="plate bottom below the initial level at the end of the stroke (m)")
    ap.add_argument("--settle", type=float, default=0.3, help="still water before the plate moves (s)")
    ap.add_argument("--hold", type=float, default=0.5)
    ap.add_argument("--kappa", type=float, default=1e5)
    ap.add_argument("--newton-tol", type=float, default=1e-4)
    ap.add_argument("--pcg-tol", type=float, default=1e-6)
    ap.add_argument("--push", type=float, default=0.9)
    ap.add_argument("--jmode", type=int, default=3)
    ap.add_argument("--mass-floor", type=float, default=0.05)
    ap.add_argument("--dhat", type=float, default=5e-4)
    ap.add_argument("--out", default="out/liquid_m18")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dx = args.dx; n_sub = max(1, int(round(args.ppc ** (1 / 3)))); hs = args.h if args.h > 0 else dx / n_sub
    cx = cz = 0.20; floor_y = 0.02; inner = 0.12; gap = 0.002
    V, F = open_box((cx, cz), inner, 0.005, floor_y, 0.16, 0.01)
    side = inner - 2 * gap; vol = 0.6e-3; depth = vol / side ** 2
    ax = np.arange(cx - side / 2 + hs / 2, cx + side / 2 - hs / 4, hs); ay = np.arange(floor_y + gap + hs / 2, floor_y + gap + depth, hs); az = ax - cx + cz
    gx, gy, gz = np.meshgrid(ax, ay, az, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)
    pts = pts + np.random.default_rng(0).uniform(-0.3 * hs, 0.3 * hs, pts.shape)
    level0 = floor_y + gap + len(ay) * hs
    pw, pt = 0.04, 0.005
    y0 = level0 + 0.005
    Vp, Fp = cuboid([cx - pw / 2, y0, cz - pw / 2], [cx + pw / 2, y0 + pt, cz + pw / 2])

    cfg = Config(dt=args.dt, preconditioner_type=0, absolute_dhat=args.dhat, newton_tol=args.newton_tol, pcg_tol=args.pcg_tol,
                 friction_rate=0.0, gd_friction_rate=0.0, ground_offset=0.0, energy_rel_tol=0.0, newton_iter_cap=200, pcg_max_iter_ratio=0.3,
                 gravity=(0.0, -9.81, 0.0))
    eng = Engine(cfg); eng.set_log_level(0)
    eng.load_mesh_from_data(V, F, 3, 3, "ABD", np.eye(4), 1e7, "Fixed")
    eng.load_mesh_from_data(Vp, Fp, 3, 3, "ABD", np.eye(4), 1e8, "Free")
    plate = int(eng.get_load_records()[-1].body_offset)
    eng.set_abd_body_density(plate, 1.0e7)
    lo = np.array([0.0, 0.001, 0.0]); hi = np.array([0.40, 0.25, 0.40])
    eng.add_liquid(pts, dx, lo, hi, rho=1000.0, kappa=args.kappa, particle_volume=hs ** 3, push_apart=args.push, j_mode=args.jmode, mass_floor=args.mass_floor)
    eng.finalize()
    eng._engine.set_body_apply_gravity(plate, False)
    C0 = np.asarray(eng.get_abd_body_transforms(np.array([plate], np.int32)))[0]
    W0 = np.array(C0, np.float64); W0[:3, :3] = C0[:3, :3].T

    def pose(dy):
        T = np.eye(4); T[1, 3] = dy
        W = T @ W0; C = np.array(W); C[:3, :3] = W[:3, :3].T
        return C
    stroke = 0.005 + args.depth; t_in = stroke / args.speed
    t1 = args.settle; t2 = t1 + t_in; t3 = t2 + args.hold; t4 = t3 + t_in

    def plate_dy(tt):
        if tt < t1: return 0.0
        if tt < t2: return -args.speed * (tt - t1)
        if tt < t3: return -stroke
        if tt < t4: return -stroke + args.speed * (tt - t3)
        return 0.0
    print(f"[plate] water {len(pts)} particles, level {(level0-floor_y)*1e3:.1f} mm; plate {pw*1e3:.0f}x{pw*1e3:.0f}x{pt*1e3:.0f} mm starts {5:.0f} mm above the water, "
          f"down {stroke*1e3:.0f} mm at {args.speed*1e3:.0f} mm/s from t {t1:.2f} s, hold {t2:.2f}-{t3:.2f} s, up until {t4:.2f} s; dx {dx*1e3:.1f} mm h {hs*1e3:.2f} mm", flush=True)

    S = []; t = 0.0; step = 0; wall0 = time.time(); seen = np.zeros(len(pts), bool); reported = 0
    while t < t4 + 0.1 - 1e-12:
        dy = plate_dy(t + args.dt)
        eng.teleport_abd_bodies(np.array([plate], np.int32), pose(dy)[None])
        x_prev = eng.get_liquid_particles().copy()
        eng.step(); t += args.dt; step += 1
        x = eng.get_liquid_particles()
        b0, b1 = y0 + dy, y0 + pt + dy
        foot = (np.abs(x[:, 0] - cx) < pw / 2) & (np.abs(x[:, 2] - cz) < pw / 2)
        inside = foot & (x[:, 1] > b0) & (x[:, 1] < b1)
        below = foot & (x[:, 1] <= b0)
        gapmin = float((b0 - x[below, 1]).min() * 1e3) if below.any() else float("nan")
        far = ~((np.abs(x[:, 0] - cx) < pw / 2 + 0.015) & (np.abs(x[:, 2] - cz) < pw / 2 + 0.015))
        level = float(np.percentile(x[far, 1], 95) + hs / 2)
        s = dict(t=t, plate_bottom_mm=b0 * 1e3, inside=int(inside.sum()), gap_min_mm=gapmin, level_mm=level * 1e3, vmax=float(np.linalg.norm(eng.get_liquid_velocities(), axis=1).max()))
        S.append(s)
        new = np.flatnonzero(inside & ~seen)
        for i in new[: max(0, 8 - reported)]:
            print(f"[track] t {t:.3f} particle {i} entered: before y {x_prev[i,1]*1e3:.2f} after y {x[i,1]*1e3:.2f} mm (|dx| {np.linalg.norm(x[i]-x_prev[i])*1e3:.2f} mm); plate bottom before {(y0+plate_dy(t-args.dt))*1e3:.2f} after {b0*1e3:.2f} mm; "
                  f"side margin {(pw/2 - max(abs(x_prev[i,0]-cx), abs(x_prev[i,2]-cz)))*1e3:+.2f} mm", flush=True)
            reported += 1
        seen |= inside
        if step % 10 == 0 or step < 3:
            print(f"t={t:6.3f}s plate bottom {b0*1e3:6.2f} mm  inside {s['inside']:4d}  gap-min {gapmin:6.2f} mm  level {level*1e3:6.2f} mm  |v|max {s['vmax']:.3f}", flush=True)
    ins = np.array([s["inside"] for s in S]); gp = np.array([s["gap_min_mm"] for s in S])
    print(f"[plate] inside-plate particles: max {ins.max()} (first at t {S[int(np.argmax(ins > 0))]['t']:.3f} s), final {ins[-1]}; min gap to the bottom face {np.nanmin(gp):.3f} mm; "
          f"wall {time.time()-wall0:.0f} s", flush=True)
    json.dump(dict(inside_max=int(ins.max()), inside_final=int(ins[-1]), gap_min_mm=float(np.nanmin(gp)), samples=S), open(os.path.join(args.out, "summary.json"), "w"))


if __name__ == "__main__":
    main()

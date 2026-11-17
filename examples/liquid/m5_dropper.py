"""M5: hollow rubber-bulb dropper (lab asset, kinematic) aspirates water from a cup through its
6 mm bore (engine liquid pump: +3 g up the tube), holds the column (+g) while it is lifted and
moved aside, then drips it onto the table with -2 g pulses. Checks that particles pass the
bore, stay in during transport, and leave one by one on the pulses.

    python examples/liquid/m5_dropper.py --out out/liquid_m5
"""
import argparse, os, sys, time
if __name__ == "__main__": import os as _os; _os.environ.setdefault("LIQ_UG", "0")   # main-path example, when run as a script: pinned to the main liquid path (the engine default is the UG mode since 2026-11-18)
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "lab_tactile"))
sys.path.insert(0, HERE)
from stiff_physics.engine import Engine, Config  # noqa: E402
from lab_assets import build_dropper  # noqa: E402
from m2_box import open_box  # noqa: E402

G = 9.81


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=0.005)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--kappa", type=float, default=1e5)
    ap.add_argument("--suck", type=float, default=0.04, help="aspiration conveyor speed up the bore (m/s)")
    ap.add_argument("--squeeze", type=float, default=0.06, help="squeeze conveyor speed out of the tip during a pulse (m/s)")
    ap.add_argument("--pulse", type=float, default=0.10, help="squeeze pulse length (s)")
    ap.add_argument("--period", type=float, default=0.5, help="pulse period (s)")
    ap.add_argument("--radius", type=float, default=3.5e-3, help="pump region radius (m)")
    ap.add_argument("--dip", type=float, default=0.015, help="tip depth below the water surface (m)")
    ap.add_argument("--lift", type=float, default=0.08, help="lift height (m)")
    ap.add_argument("--move", type=float, default=0.10, help="sideways move (m, +x)")
    ap.add_argument("--newton-tol", type=float, default=1e-3)
    ap.add_argument("--newton-cap", type=int, default=80)
    ap.add_argument("--jmode", type=int, default=3)
    ap.add_argument("--out", default="out/liquid_m5")
    ap.add_argument("--log-level", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dx, dt = args.dx, args.dt

    # cup: 6 cm cavity, floor 2 cm above the table; water 5 cm deep
    floor_y, cxz, inner, wall = 0.02, (0.20, 0.20), 0.06, 0.005
    V, F = open_box(cxz, inner, wall, floor_y, 0.09, 0.01)
    cfg = Config(dt=dt, preconditioner_type=0, absolute_dhat=5e-4, newton_tol=args.newton_tol, friction_rate=0.0,
                 gd_friction_rate=0.0, ground_offset=0.0, pcg_tol=1e-6, newton_iter_cap=args.newton_cap)
    eng = Engine(cfg); eng.set_log_level(args.log_level)
    eng.load_mesh_from_data(V, F, 3, 3, "ABD", np.eye(4), 1e7, "Fixed")

    # hollow dropper (lab asset), kinematic: ABD Free + huge density + gravity off + teleport each step
    d = build_dropper(hollow=True)
    meta = d.meta
    water_top = floor_y + 0.05
    T0 = np.eye(4); T0[:3, 3] = [cxz[0], water_top - args.dip, cxz[1]]       # tip (local origin) dipped into the water
    wv0 = d.verts @ T0[:3, :3].T + T0[:3, 3]
    eng.load_mesh_from_data(wv0, d.faces, verts_per_face=3, dimensions=3, body_type="ABD", transform=np.eye(4),
                            young_modulus=1e8, boundary_type="Free")
    rec = eng.get_load_records()[-1]; body = int(rec.body_offset)
    eng.set_abd_body_density(body, 1.0e7)

    # water: cylinder-ish block filling the cavity to water_top, leaving the tube free (seed excludes the bore)
    hs = dx / 2; m = 0.6 * dx
    h2 = inner / 2
    ax = np.arange(cxz[0] - h2 + m, cxz[0] + h2 - m, hs); ay = np.arange(floor_y + m, water_top, hs); az = np.arange(cxz[1] - h2 + m, cxz[1] + h2 - m, hs)
    gx, gy, gz = np.meshgrid(ax, ay, az, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)
    pts = pts + np.random.default_rng(0).uniform(-0.3 * hs, 0.3 * hs, pts.shape)
    rr = np.hypot(pts[:, 0] - cxz[0], pts[:, 2] - cxz[1])
    pts = pts[~((rr < meta["tube_r"] + 1.5e-3) & (pts[:, 1] > water_top - args.dip - 2e-3))]   # not inside / around the dipped tube
    lo = np.array([0.0, 0.001, 0.0]); hi = np.array([0.45, 0.45, 0.45])
    eng.add_liquid(pts, dx, lo, hi, rho=1000.0, kappa=args.kappa, particle_volume=hs ** 3, push_apart=0.9, j_mode=args.jmode, node_pool=30000)
    eng.finalize()
    eng._engine.set_body_apply_gravity(body, False)
    print(f"[m5] dropper bore r {meta['tube_r_in']*1e3:.1f} mm, cavity top {meta['cavity_top']*1e3:.0f} mm; water {len(pts)} particles; {eng.get_liquid_stats()}", flush=True)

    C0 = np.asarray(eng.get_abd_body_transforms(np.array([body], np.int32)))[0]
    W0 = np.array(C0, np.float64); W0[:3, :3] = C0[:3, :3].T

    def pose(offset):
        T = np.eye(4); T[:3, 3] = offset
        W = T @ W0
        C = np.array(W); C[:3, :3] = W[:3, :3].T
        return T, C

    # schedule: settle, aspirate, hold, lift 8 cm, move 10 cm in +x, hold, drip (pulses), release
    phases = [("settle", 0.4), ("aspirate", 1.5), ("hold", 0.5), ("lift", 1.0), ("move", 1.0), ("hold2", 0.5), ("drip", 2.5), ("release", 1.0)]
    t = 0.0; step = 0; nt_prev = 0; off = np.zeros(3); wall0 = time.time()
    out_tip_prev = 0
    for name, dur in phases:
        t_ph = 0.0
        while t_ph < dur:
            # kinematic motion
            if name == "lift": off = np.array([0.0, args.lift * t_ph / dur, 0.0])
            elif name == "move": off = np.array([args.move * t_ph / dur, args.lift, 0.0])
            elif name in ("hold2", "drip", "release"): off = np.array([args.move, args.lift, 0.0])
            T, C = pose(off)
            eng.teleport_abd_bodies(np.array([body], np.int32), C[None])
            tip = T @ T0 @ np.array([0, 0, 0, 1.0]); top = T @ T0 @ np.array([0, meta["cavity_top"], 0, 1.0])
            # follow = 0: the velocity tracking alone carries the column with the tube (a position
            # follow on top doubles the motion: the column rams the leading wall / ceiling and is spat out)
            if name == "aspirate": acc, conv, fol, trk = G, args.suck, 0.0, 0.5
            elif name in ("hold", "lift", "move", "hold2"): acc, conv, fol, trk = G, 0.0, 0.0, 0.5
            elif name == "drip": acc, conv, fol, trk = (G, -args.squeeze, 0.0, 0.5) if (t_ph % args.period) < args.pulse else (G, 0.0, 0.0, 0.5)
            else: acc, conv, fol, trk = 0.0, 0.0, 0.0, 0.0
            eng.set_liquid_pump(tip[:3], top[:3], args.radius, acc, conv, fol, trk)
            ts = time.time(); eng.step(); wall = time.time() - ts
            t += dt; t_ph += dt; step += 1
            if step % 20 == 0:
                x = eng.get_liquid_particles(); nt = eng._engine.get_total_newton_iters(); vel = eng.get_liquid_velocities()
                # particles in the bore: within tube_r_in of the axis and above the tip
                axis = (top[:3] - tip[:3]); L = np.linalg.norm(axis); axis /= L
                dvec = x - tip[:3]; s = dvec @ axis; rad = np.linalg.norm(dvec - np.outer(s, axis), axis=1)
                in_bore = int(np.sum((s > 0) & (s < L) & (rad < meta["tube_r_in"])))
                n_pump = eng.count_liquid_in_pump()
                on_table = int(np.sum((x[:, 1] < 0.012) & (np.maximum(np.abs(x[:, 0] - cxz[0]), np.abs(x[:, 2] - cxz[1])) > inner / 2)))
                outside_cup = int(np.sum((np.maximum(np.abs(x[:, 0] - cxz[0]), np.abs(x[:, 2] - cxz[1])) > inner / 2 + wall) & ~((s > -0.01) & (s < L) & (rad < meta["tube_r"] + 2e-3))))
                mb = (s > 0) & (s < L) & (rad < meta["tube_r_in"])
                vb = vel[mb].mean(0) if mb.any() else np.zeros(3)
                print(f"t={t:5.2f}s {name:9s} acc {acc/G:+.1f} g conv {conv*100:+.0f} cm/s | in bore {in_bore:3d}  in pump region {n_pump:3d}  on table {on_table:3d}  outside cup {outside_cup:3d} | newton {nt-nt_prev:3d} {wall*1000:6.0f} ms | tip x {tip[0]:.3f} y {tip[1]:.3f}  bore v mean ({vb[0]*100:+.1f},{vb[1]*100:+.1f},{vb[2]*100:+.1f}) cm/s  bore y min {x[mb][:,1].min()*1e3 if mb.any() else 0:.1f} mm", flush=True)
                nt_prev = nt
    np.save(os.path.join(args.out, "pts_final.npy"), eng.get_liquid_particles())
    print(f"[m5] done {step} steps in {time.time()-wall0:.0f} s", flush=True)


if __name__ == "__main__":
    main()

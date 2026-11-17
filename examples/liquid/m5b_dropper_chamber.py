"""M5b: dropper with a CHAMBER-PRESSURE pump (re-audit step 4, 2026-10-20). Same cup / kinematic hollow dropper / water as
m5_dropper.py, but the liquid column is driven only by the air pressure of a chamber whose volume V_cav(t) is prescribed
(the external drive: a squeezed bulb released in the water, later squeezed in pulses over the table). No conveyor, no
position follow, no velocity tracking, no gravity cancellation: suction, column weight, pressure and contact do the rest.

    PYTHONPATH=. python examples/liquid/m5b_dropper_chamber.py --out out/liquid_m5b
"""
import argparse, json, os, sys, time
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
    ap.add_argument("--vcav", type=float, default=8.3e-6, help="chamber volume at rest (m^3): shell bulb cavity + bore")
    ap.add_argument("--squeeze0", type=float, default=0.08, help="initial squeeze fraction (bulb squeezed in air before dipping)")
    ap.add_argument("--release", type=float, default=1.5, help="release duration (s)")
    ap.add_argument("--pulse-dv", type=float, default=0.012, help="squeeze fraction per drip pulse")
    ap.add_argument("--pulse", type=float, default=0.10)
    ap.add_argument("--period", type=float, default=0.5)
    ap.add_argument("--radius", type=float, default=3.5e-3, help="column region radius (m)")
    ap.add_argument("--dip", type=float, default=0.015)
    ap.add_argument("--lift", type=float, default=0.08)
    ap.add_argument("--move", type=float, default=0.10)
    ap.add_argument("--newton-tol", type=float, default=1e-3)
    ap.add_argument("--newton-cap", type=int, default=80)
    ap.add_argument("--jmode", type=int, default=3)
    ap.add_argument("--out", default="out/liquid_m5b")
    ap.add_argument("--log-level", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dx, dt = args.dx, args.dt
    floor_y, cxz, inner, wall = 0.02, (0.20, 0.20), 0.06, 0.005
    V, F = open_box(cxz, inner, wall, floor_y, 0.09, 0.01)
    cfg = Config(dt=dt, preconditioner_type=0, absolute_dhat=5e-4, newton_tol=args.newton_tol, friction_rate=0.0,
                 gd_friction_rate=0.0, ground_offset=0.0, pcg_tol=1e-6, newton_iter_cap=args.newton_cap)
    eng = Engine(cfg); eng.set_log_level(args.log_level)
    eng.load_mesh_from_data(V, F, 3, 3, "ABD", np.eye(4), 1e7, "Fixed")
    d = build_dropper(hollow=True); meta = d.meta
    water_top = floor_y + 0.05
    T0 = np.eye(4); T0[:3, 3] = [cxz[0], water_top - args.dip, cxz[1]]
    wv0 = d.verts @ T0[:3, :3].T + T0[:3, 3]
    eng.load_mesh_from_data(wv0, d.faces, verts_per_face=3, dimensions=3, body_type="ABD", transform=np.eye(4), young_modulus=1e8, boundary_type="Free")
    rec = eng.get_load_records()[-1]; body = int(rec.body_offset)
    eng.set_abd_body_density(body, 1.0e7)
    hs = dx / 2; m = 0.6 * dx; h2 = inner / 2
    ax = np.arange(cxz[0] - h2 + m, cxz[0] + h2 - m, hs); ay = np.arange(floor_y + m, water_top, hs); az = np.arange(cxz[1] - h2 + m, cxz[1] + h2 - m, hs)
    gx, gy, gz = np.meshgrid(ax, ay, az, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1) + np.random.default_rng(0).uniform(-0.3 * hs, 0.3 * hs, (ax.size * ay.size * az.size, 3))
    rr = np.hypot(pts[:, 0] - cxz[0], pts[:, 2] - cxz[1])
    pts = pts[~((rr < meta["tube_r"] + 1.5e-3) & (pts[:, 1] > water_top - args.dip - 2e-3))]
    lo = np.array([0.0, 0.001, 0.0]); hi = np.array([0.45, 0.45, 0.45])
    eng.add_liquid(pts, dx, lo, hi, rho=1000.0, kappa=args.kappa, particle_volume=hs ** 3, push_apart=0.9, j_mode=args.jmode, node_pool=30000)
    eng.finalize()
    eng._engine.set_body_apply_gravity(body, False)
    r_in = float(meta["tube_r_in"])
    print(f"[m5b] bore r {r_in*1e3:.1f} mm (A {np.pi*r_in**2*1e6:.1f} mm^2), chamber {args.vcav*1e6:.2f} ml, squeeze0 {args.squeeze0:.3f}; water {len(pts)} particles", flush=True)
    C0 = np.asarray(eng.get_abd_body_transforms(np.array([body], np.int32)))[0]
    W0 = np.array(C0, np.float64); W0[:3, :3] = C0[:3, :3].T
    def pose(offset):
        T = np.eye(4); T[:3, 3] = offset; W = T @ W0; C = np.array(W); C[:3, :3] = W[:3, :3].T
        return T, C
    # schedule: release (aspirate) -> hold -> lift -> move -> hold2 -> drip pulses -> release the bulb in the air
    phases = [("release", args.release), ("hold", 0.5), ("lift", 1.0), ("move", 1.0), ("hold2", 0.5), ("drip", 2.5), ("open", 1.0)]
    t = 0.0; step = 0; off = np.zeros(3); wall0 = time.time(); s_drip_base = 0.0; log = []
    for name, dur in phases:
        t_ph = 0.0
        while t_ph < dur:
            if name == "lift": off = np.array([0.0, args.lift * t_ph / dur, 0.0])
            elif name == "move": off = np.array([args.move * t_ph / dur, args.lift, 0.0])
            elif name in ("hold2", "drip", "open"): off = np.array([args.move, args.lift, 0.0])
            T, C = pose(off)
            eng.teleport_abd_bodies(np.array([body], np.int32), C[None])
            tip = (T @ T0 @ np.array([0, 0, 0, 1.0]))[:3]; top = (T @ T0 @ np.array([0, meta["cavity_top"], 0, 1.0]))[:3]
            # squeeze fraction s(t): the external drive
            if name == "release": s = args.squeeze0 * (1.0 - t_ph / dur)
            elif name in ("hold", "lift", "move", "hold2"): s = 0.0
            elif name == "drip":
                k_pulse = int(t_ph // args.period); u = t_ph - k_pulse * args.period
                s = args.pulse_dv * (k_pulse + min(1.0, u / args.pulse))
            else: s = 0.0
            eng.set_liquid_pump_chamber(tip, top, args.radius, args.vcav * (1.0 - s), 101325.0, r_in)
            ts = time.time(); eng.step(); wall = time.time() - ts
            t += dt; t_ph += dt; step += 1
            if step % 20 == 0:
                x = eng.get_liquid_particles(); st = eng.get_liquid_pump_state(); nt = eng._engine.get_total_newton_iters()
                axis = top - tip; L = np.linalg.norm(axis); axis /= L
                dvec = x - tip; sa = dvec @ axis; rad = np.linalg.norm(dvec - np.outer(sa, axis), axis=1)
                in_bore = int(np.sum((sa > 0) & (sa < L) & (rad < r_in)))
                on_table = int(np.sum((x[:, 1] < 0.012) & (np.maximum(np.abs(x[:, 0] - cxz[0]), np.abs(x[:, 2] - cxz[1])) > inner / 2)))
                col_h = float(sa[(sa > 0) & (sa < L) & (rad < r_in)].max()) if in_bore else 0.0
                rec_ = dict(t=t, phase=name, s=s, in_bore=in_bore, on_table=on_table, col_h=col_h, newton=nt, **{k: (float(v) if not isinstance(v, bool) else v) for k, v in st.items()})
                log.append(rec_)
                print(f"t={t:5.2f}s {name:8s} s {s:.3f} | V_air {st['V_air']*1e6:6.3f} ml p_gauge {st['p_gauge']:+8.1f} Pa sealed {int(st['sealed'])} col {int(st['n_column']):3d} | in bore {in_bore:3d} col h {col_h*1e3:5.1f} mm  on table {on_table:3d} | newton {nt} {wall*1e3:6.0f} ms", flush=True)
    np.save(os.path.join(args.out, "pts_final.npy"), eng.get_liquid_particles())
    json.dump(log, open(os.path.join(args.out, "log.json"), "w"))
    print(f"[m5b] done {step} steps in {time.time()-wall0:.0f} s", flush=True)


if __name__ == "__main__":
    main()

"""M16: still water in a tank that slides on a frictionless floor under a slightly tilted gravity (frame-invariance check).

Gravity has a small horizontal component, so tank and water accelerate together along x; in the tank's frame the effective
gravity is vertical and the water should come to rest. Reports the particle speed RELATIVE to the tank over the last 3 s
(tank velocity from its vertex displacement). A rest treatment that looks at world velocities (e.g. a speed-gated ghost
penalty) stops working once the vessel moves; one measured relative to the wall keeps working. No sleep / freezing.
    python examples/liquid/m16_sliding_tank.py --tilt-deg 0.2 --out out/liquid_m16
"""
import argparse, json, os, sys, time
if __name__ == "__main__": import os as _os; _os.environ.setdefault("LIQ_UG", "0")   # main-path example, when run as a script: pinned to the main liquid path (the engine default is the UG mode since 2026-11-18)
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from stiff_physics.engine import Engine, Config  # noqa: E402
from m2_box import cuboid  # noqa: E402
from m13_efflux import merge  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inner", type=float, default=0.06)
    ap.add_argument("--h0", type=float, default=0.04)
    ap.add_argument("--tilt-deg", type=float, default=0.2, help="horizontal gravity component, deg")
    ap.add_argument("--dx", type=float, default=0.005)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--duration", type=float, default=4.5)
    ap.add_argument("--push", type=float, default=0.9)
    ap.add_argument("--out", default="out/liquid_m16")
    args = ap.parse_args()
    for key in ("LIQ_ISL_V", "LIQ_SLEEP_V", "LIQ_REST_DAMP", "LIQ_STICK_H"):
        if float(os.environ.get(key, "0")) != 0:
            raise ValueError(f"{key} must be 0 (no freezing) in this test")
    os.makedirs(args.out, exist_ok=True)
    dx, hs = args.dx, args.dx / 2.0
    W, h0, fy, th = args.inner, args.h0, 0.0075, 0.006
    top = fy + h0 + 0.02
    hw = W / 2
    tank = [cuboid([-hw - th, fy - th, -hw - th], [hw + th, fy, hw + th]),
            cuboid([-hw - th, fy, -hw - th], [-hw, top, hw + th]), cuboid([hw, fy, -hw - th], [hw + th, top, hw + th]),
            cuboid([-hw, fy, -hw - th], [hw, top, -hw]), cuboid([-hw, fy, hw], [hw, top, hw + th])]
    V, F = merge(tank)
    off = np.array([0.37 * dx, 0.0, 0.37 * dx]); V = V + off
    m = 0.6 * hs
    ax = [np.arange(-hw + m, hw - m + 1e-9, hs), np.arange(fy + m, fy + h0 + 1e-9, hs), np.arange(-hw + m, hw - m + 1e-9, hs)]
    X, Y, Z = np.meshgrid(*ax, indexing="ij"); pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1) + off
    pts = pts + np.random.default_rng(0).uniform(-0.3 * hs, 0.3 * hs, pts.shape)
    g = 9.81; a = np.radians(args.tilt_deg)
    cfg = Config(dt=args.dt, preconditioner_type=0, absolute_dhat=5e-4, newton_tol=1e-4, pcg_tol=1e-6, friction_rate=0.0,
                 gd_friction_rate=0.0, ground_offset=0.0, energy_rel_tol=0.0, newton_iter_cap=200, pcg_max_iter_ratio=0.3,
                 gravity=(g * np.sin(a), -g * np.cos(a), 0.0))
    eng = Engine(cfg); eng.set_log_level(0)
    eng.load_mesh_from_data(V, F, 3, 3, "ABD", np.eye(4), 1e7, "Free")
    nv = len(V)
    lo = np.array([-0.06, -0.003, -0.06]); hi = np.array([0.48, top + 0.02, 0.06])
    eng.add_liquid(pts, dx, lo, hi, rho=1000.0, kappa=1e5, particle_volume=hs ** 3, push_apart=args.push, j_mode=0)
    eng.finalize()
    print(f"[m16] tank {W*1e3:.0f} mm, water {h0*1e3:.0f} mm ({len(pts)} particles), gravity tilt {args.tilt_deg} deg "
          f"(tank+water accelerate {g*np.sin(a)*1e3:.1f} mm/s^2 if frictionless), dx {dx*1e3:.1f} mm, dt {args.dt*1e3:.1f} ms", flush=True)
    xs, cs, ts = [], [], []
    nsteps = round(args.duration / args.dt); wall0 = time.time()
    for step in range(nsteps + 1):
        if step % 8 == 0 or step == nsteps:
            x = eng.get_liquid_particles()
            if not np.isfinite(x).all(): raise RuntimeError(f"nonfinite liquid at step {step}")
            vv = eng.get_vertices()
            xs.append(x.copy()); ts.append(step * args.dt)
            cs.append(np.asarray(vv)[:nv].mean(0) if vv is not None else np.full(3, np.nan))
        if step != nsteps: eng.step()
        if step == nsteps // 2:
            print(f"[m16]   t {step*args.dt:.1f} s: tank centre x {cs[-1][0]*1e3:.1f} mm, water mean x {xs[-1][:,0].mean()*1e3:.1f} mm", flush=True)
    x, t, c = np.asarray(xs), np.asarray(ts), np.asarray(cs)
    first = np.searchsorted(t, t[-1] - 3.0)
    dtr = np.diff(t[first:])[:, None]
    vt = np.diff(c[first:], axis=0) / dtr                               # tank velocity per record interval
    vrel = np.diff(x[first:], axis=0) / dtr[:, :, None] - vt[:, None, :]
    sp = np.linalg.norm(vrel, axis=2).mean(0) * 1e3
    res = dict(particles=len(pts), tilt_deg=args.tilt_deg, tank_speed_end_mm_s=float(np.linalg.norm(vt[-1]) * 1e3),
               water_minus_tank_x_mm=float((x[-1][:, 0].mean() - c[-1][0]) * 1e3),
               rel_median_speed_mm_s=float(np.median(sp)), rel_p90_speed_mm_s=float(np.percentile(sp, 90)), rel_max_speed_mm_s=float(sp.max()),
               wall_s=time.time() - wall0, newton_iters=int(eng._engine.get_total_newton_iters()),
               environment={k: v for k, v in os.environ.items() if k.startswith("LIQ_")}, parameters=vars(args))
    print("[m16] " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k not in ("environment", "parameters")}), flush=True)
    json.dump(res, open(os.path.join(args.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

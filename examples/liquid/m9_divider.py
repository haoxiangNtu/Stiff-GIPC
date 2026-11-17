"""M9: thin fixed divider between two water chambers (acceptance test 2 of the 2026-10-18 re-audit).

A fixed box is split by a thin plate (default 1 mm, thinner than the 5 mm grid cell) into chambers A and B,
both filled with water. A kinematic piston plunges into chamber A only. The plate is part of the fixed body,
so there is NO physical path for momentum or volume from A to B: anything B does beyond the quiet-water
residual is numerical cross-talk through the grid nodes shared across the plate (12 shared nodes of the 27-node
stencil for particles 0.1 mm off a 1 mm plate; CPU counterexample in liquid_reaudit_2026-10-18).

    PYTHONPATH=. python examples/liquid/m9_divider.py --out out/liquid_m9            # piston in A
    PYTHONPATH=. python examples/liquid/m9_divider.py --out out/liquid_m9_ctl --no-piston   # control (quiet water)
"""
import argparse, json, os, sys, time
if __name__ == "__main__": import os as _os; _os.environ.setdefault("LIQ_UG", "0")   # main-path example, when run as a script: pinned to the main liquid path (the engine default is the UG mode since 2026-11-18)
from pathlib import Path
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from stiff_physics.engine import Engine, Config  # noqa: E402
from m2_box import cuboid, open_box  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dx", type=float, default=0.005)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--plate", type=float, default=1e-3, help="divider thickness (m)")
    ap.add_argument("--plate-x-offset", type=float, default=0.0, help="shift the plate off the grid plane (m); 0 = plate centred on a node plane")
    ap.add_argument("--inner", type=float, default=0.08, help="box cavity (m), split in two along x")
    ap.add_argument("--depth", type=float, default=0.04, help="water depth in both chambers (m)")
    ap.add_argument("--margin-plate", type=float, default=1.5e-3, help="seed gap to the plate faces (m)")
    ap.add_argument("--duration", type=float, default=3.0)
    ap.add_argument("--piston-t0", type=float, default=1.0)
    ap.add_argument("--piston-plunge", type=float, default=0.03, help="plunge depth below the water surface (m)")
    ap.add_argument("--piston-speed", type=float, default=0.1)
    ap.add_argument("--no-piston", action="store_true", help="control run: identical scene, piston parked above the water")
    ap.add_argument("--push", type=float, default=0.9)
    ap.add_argument("--mass-floor", type=float, default=0.05)
    ap.add_argument("--epsv", type=float, default=1e-3)
    ap.add_argument("--jmode", type=int, default=0)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--newton-tol", type=float, default=1e-4)
    ap.add_argument("--pcg-tol", type=float, default=1e-6)
    args = ap.parse_args()
    for key in ("LIQ_ISL_V", "LIQ_SLEEP_V", "LIQ_REST_K", "LIQ_REST_DAMP", "LIQ_STICK_H"):
        if float(os.environ.get(key, "0")) != 0:
            raise ValueError(f"{key} must be 0 for this physical test")
        os.environ[key] = "0"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dx, dt = args.dx, args.dt
    cx, cz, inner, wall, floor_y, height, floor_th = 0.20, 0.20, args.inner, 0.005, 0.02, 0.10, 0.01
    h, t = inner / 2, args.plate
    px = cx + args.plate_x_offset                              # plate centre (x)
    V, F = open_box((cx, cz), inner, wall, floor_y, height, floor_th)
    # the plate is part of the same fixed body (overlaps the side walls and the floor: sealed, no inter-body contact)
    pv, pf = cuboid([px - t / 2, floor_y - 2e-3, cz - h - 2e-3], [px + t / 2, floor_y + 0.09, cz + h + 2e-3])
    V, F = np.vstack([V, pv]), np.vstack([F, pf + len(V)])
    cfg = Config(dt=dt, preconditioner_type=0, absolute_dhat=0.0005, newton_tol=args.newton_tol, absolute_epsv=args.epsv, pcg_tol=args.pcg_tol,
                 newton_iter_cap=120, pcg_max_iter_ratio=0.3, energy_rel_tol=0.0, friction_rate=0.0, gd_friction_rate=0.0, ground_offset=0.0)
    eng = Engine(cfg); eng.set_log_level(0)
    eng.load_mesh_from_data(V, F, 3, 3, "ABD", np.eye(4), 1e8, "Fixed")
    # kinematic piston (ABD Free, huge density, gravity off, teleported each step) above chamber A
    water_top = floor_y + args.depth
    xa = 0.5 * (cx - h + px - t / 2)                            # chamber A centre line
    pw, ph = 0.03, 0.04
    y_park = water_top + 0.01                                   # bottom face 1 cm above the water
    bv, bf = cuboid([xa - pw / 2, y_park, cz - pw / 2], [xa + pw / 2, y_park + ph, cz + pw / 2])
    eng.load_mesh_from_data(bv, bf, 3, 3, "ABD", np.eye(4), 1e8, "Free")
    rec = eng.get_load_records()[-1]; piston = int(rec.body_offset)
    eng.set_abd_body_density(piston, 1.0e7)
    # water: two blocks on a dx/2 lattice with jitter, 0.6 dx from the outer walls, margin_plate from the plate
    hs = dx / 2; m = 0.6 * dx; mp = args.margin_plate
    rng = np.random.default_rng(0)
    def block(x0, x1):
        ax = np.arange(x0, x1, hs); ay = np.arange(floor_y + m, water_top, hs); az = np.arange(cz - h + m, cz + h - m, hs)
        gx, gy, gz = np.meshgrid(ax, ay, az, indexing="ij")
        pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)
        return pts + rng.uniform(-0.3 * hs, 0.3 * hs, pts.shape)
    pa = block(cx - h + m, px - t / 2 - mp); pb = block(px + t / 2 + mp, cx + h - m)
    pts = np.vstack([pa, pb]); side0 = np.r_[np.zeros(len(pa), int), np.ones(len(pb), int)]
    lo = np.array([0.0, 0.001, 0.0]); hi = np.array([0.45, 0.45, 0.45])
    eng.add_liquid(pts, dx, lo, hi, rho=1000.0, kappa=1e5, particle_volume=hs ** 3, push_apart=args.push, j_mode=args.jmode,
                   mass_floor=args.mass_floor, ground_mu=0.0, stick_h=0.0, node_pool=40000)
    eng.finalize()
    eng._engine.set_body_apply_gravity(piston, False)
    print(f"[m9] plate {t*1e3:.1f} mm, chambers {len(pa)} + {len(pb)} particles, dx {dx*1e3:.0f} mm, piston {'off' if args.no_piston else 'on'}; "
          f"{eng.get_liquid_stats()}", flush=True)
    C0 = np.asarray(eng.get_abd_body_transforms(np.array([piston], np.int32)))[0]
    W0 = np.array(C0, np.float64); W0[:3, :3] = C0[:3, :3].T
    def pose(dy):
        T = np.eye(4); T[1, 3] = dy
        W = T @ W0; C = np.array(W); C[:3, :3] = W[:3, :3].T
        return C
    t0 = args.piston_t0; plunge = 0.01 + args.piston_plunge; t_in = plunge / args.piston_speed
    t_hold = t0 + t_in + 0.7; t_out_end = t_hold + t_in
    def piston_dy(tt):
        if args.no_piston or tt < t0: return 0.0
        if tt < t0 + t_in: return -args.piston_speed * (tt - t0)
        if tt < t_hold: return -plunge
        if tt < t_out_end: return -plunge + args.piston_speed * (tt - t_hold)
        return 0.0
    nsteps = round(args.duration / dt)
    series = []; fallbacks = {}; wall0 = time.time(); newton0 = eng._engine.get_total_newton_iters()
    def sample(step):
        x = eng.get_liquid_particles(); v = eng.get_liquid_velocities(); jj = eng.get_liquid_J()
        if not (np.isfinite(x).all() and np.isfinite(v).all()):
            raise RuntimeError(f"nonfinite liquid at step {step}")
        sp = np.linalg.norm(v, axis=1)
        inA, inB = x[:, 0] < px - t / 2, x[:, 0] > px + t / 2
        crossed = int(np.sum((side0 == 0) & (x[:, 0] > px)) + np.sum((side0 == 1) & (x[:, 0] < px)))
        in_plate = int(np.sum(~inA & ~inB))
        rec = dict(t=step * dt,
                   A=dict(median=float(np.median(sp[inA])), p90=float(np.percentile(sp[inA], 90)), max=float(sp[inA].max()),
                          level=float(np.percentile(x[inA, 1], 95)), vmean=v[inA].mean(0).tolist()),
                   B=dict(median=float(np.median(sp[inB])), p90=float(np.percentile(sp[inB], 90)), max=float(sp[inB].max()),
                          level=float(np.percentile(x[inB, 1], 95)), vmean=v[inB].mean(0).tolist()),
                   crossed=crossed, in_plate=in_plate, piston_dy=piston_dy(step * dt),
                   J=dict(mean=float(jj.mean()), min=float(jj.min()), max=float(jj.max()), p05=float(np.percentile(jj, 5)), p95=float(np.percentile(jj, 95))),
                   n_high=int(np.sum(x[:, 1] > water_top + 0.005)),
                   newton=int(eng._engine.get_total_newton_iters() - newton0))
        series.append(rec)
        return rec
    for step in range(nsteps + 1):
        if step % args.every == 0 or step == nsteps:
            r = sample(step)
            if step % (40 * args.every) == 0:
                print(f"[m9] t={r['t']:.3f} piston {r['piston_dy']*1e3:+.1f} mm | A med {r['A']['median']*1e3:.3f} p90 {r['A']['p90']*1e3:.3f} max {r['A']['max']*1e3:.2f} mm/s "
                      f"lvl {r['A']['level']*1e3:.1f} | B med {r['B']['median']*1e3:.3f} p90 {r['B']['p90']*1e3:.3f} max {r['B']['max']*1e3:.2f} mm/s lvl {r['B']['level']*1e3:.1f} "
                      f"| crossed {r['crossed']} in-plate {r['in_plate']} | newton {r['newton']} | {time.time()-wall0:.0f} s", flush=True)
        if step == nsteps: break
        eng.teleport_abd_bodies(np.array([piston], np.int32), pose(piston_dy(step * dt))[None])
        eng.step()
        try:
            for k, v in eng.get_liquid_stats().items():
                if k.startswith("fb_"): fallbacks[k] = fallbacks.get(k, 0) + int(v)
        except Exception:
            pass
    ts = np.array([r["t"] for r in series])
    def window(lo_t, hi_t, side, key):
        sel = (ts >= lo_t) & (ts < hi_t)
        return float(np.mean([r[side][key] for r, s in zip(series, sel) if s])) if sel.any() else float("nan")
    pre = (max(0.0, t0 - 0.5), t0); act = (t0, t_out_end); post = (t_out_end, args.duration + 1e-9)
    summary = dict(particles=dict(A=int(len(pa)), B=int(len(pb))), plate_mm=t * 1e3, piston=not args.no_piston,
                   windows=dict(pre=pre, active=act, post=post),
                   B_median_mm_s=dict(pre=window(*pre, "B", "median") * 1e3, active=window(*act, "B", "median") * 1e3, post=window(*post, "B", "median") * 1e3),
                   B_p90_mm_s=dict(pre=window(*pre, "B", "p90") * 1e3, active=window(*act, "B", "p90") * 1e3, post=window(*post, "B", "p90") * 1e3),
                   B_max_mm_s=dict(pre=window(*pre, "B", "max") * 1e3, active=window(*act, "B", "max") * 1e3, post=window(*post, "B", "max") * 1e3),
                   A_median_mm_s=dict(pre=window(*pre, "A", "median") * 1e3, active=window(*act, "A", "median") * 1e3, post=window(*post, "A", "median") * 1e3),
                   A_p90_mm_s=dict(pre=window(*pre, "A", "p90") * 1e3, active=window(*act, "A", "p90") * 1e3, post=window(*post, "A", "p90") * 1e3),
                   B_level_mm=dict(pre=window(*pre, "B", "level") * 1e3, active=window(*act, "B", "level") * 1e3, post=window(*post, "B", "level") * 1e3),
                   A_level_mm=dict(pre=window(*pre, "A", "level") * 1e3, active=window(*act, "A", "level") * 1e3, post=window(*post, "A", "level") * 1e3),
                   crossed_max=int(max(r["crossed"] for r in series)), in_plate_max=int(max(r["in_plate"] for r in series)),
                   J_final=series[-1]["J"], n_high_max=int(max(r["n_high"] for r in series)), n_high_final=int(series[-1]["n_high"]),
                   newton_iters=int(eng._engine.get_total_newton_iters() - newton0), fallbacks=fallbacks, wall_s=time.time() - wall0,
                   parameters=vars(args), environment={k: v for k, v in os.environ.items() if k.startswith("LIQ_")})
    (out / "series.json").write_text(json.dumps(series))
    np.savez_compressed(out / "final.npz", x=eng.get_liquid_particles(), v=eng.get_liquid_velocities(), J=eng.get_liquid_J(), side0=side0)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("parameters", "environment")}, indent=2), flush=True)


if __name__ == "__main__":
    main()

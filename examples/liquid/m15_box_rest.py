"""M15: still water in a square tank whose walls are aligned with the grid or rotated about the vertical axis.

Separates the still-water creep that comes from walls cutting the Cartesian grid at an angle (cut cells whose pressure acts
along the grid axes, not along the wall normal) from creep that does not depend on the wall orientation. Measures as m14:
particle speeds and drift over the last 3 s. Nothing freezes particles (island / particle sleep, velocity reset forbidden).
    python examples/liquid/m15_box_rest.py --yaw 30 --shift 0.37 --out out/liquid_m15
"""
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from stiff_physics.engine import Engine, Config  # noqa: E402
from m2_box import cuboid  # noqa: E402
from m13_efflux import merge  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inner", type=float, default=0.08, help="tank inner width (square, m)")
    ap.add_argument("--h0", type=float, default=0.06, help="water depth (m)")
    ap.add_argument("--yaw", type=float, default=0.0, help="tank rotation about the vertical axis (deg)")
    ap.add_argument("--shift", type=float, default=0.0, help="tank offset along x and z, fraction of dx")
    ap.add_argument("--dx", type=float, default=0.005)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--kappa", type=float, default=1e5)
    ap.add_argument("--push", type=float, default=0.9)
    ap.add_argument("--jmode", type=int, default=0)
    ap.add_argument("--mass-floor", type=float, default=0.05)
    ap.add_argument("--out", default="out/liquid_m15")
    args = ap.parse_args()
    for key in ("LIQ_ISL_V", "LIQ_SLEEP_V", "LIQ_REST_DAMP", "LIQ_STICK_H"):
        if float(os.environ.get(key, "0")) != 0:
            raise ValueError(f"{key} must be 0 (no freezing) in this test")
    os.makedirs(args.out, exist_ok=True)
    dx, hs = args.dx, args.dx / 2.0
    W, h0, fy, th = args.inner, args.h0, 0.02, 0.01
    top = fy + h0 + 0.03
    hw = W / 2
    tank = [cuboid([-hw - th, fy - th, -hw - th], [hw + th, fy, hw + th]),
            cuboid([-hw - th, fy, -hw - th], [-hw, top, hw + th]), cuboid([hw, fy, -hw - th], [hw + th, top, hw + th]),
            cuboid([-hw, fy, -hw - th], [hw, top, -hw]), cuboid([-hw, fy, hw], [hw, top, hw + th])]
    V, F = merge(tank)
    m = 0.6 * hs
    ax = [np.arange(-hw + m, hw - m + 1e-9, hs), np.arange(fy + m, fy + h0 + 1e-9, hs), np.arange(-hw + m, hw - m + 1e-9, hs)]
    X, Y, Z = np.meshgrid(*ax, indexing="ij"); pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    pts = pts + np.random.default_rng(0).uniform(-0.3 * hs, 0.3 * hs, pts.shape)
    c, s_ = np.cos(np.radians(args.yaw)), np.sin(np.radians(args.yaw))
    R = np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]]); off = np.array([args.shift * dx, 0.0, args.shift * dx])
    V = V @ R.T + off; pts = pts @ R.T + off
    cfg = Config(dt=args.dt, preconditioner_type=0, absolute_dhat=5e-4, newton_tol=1e-4, pcg_tol=1e-6, friction_rate=0.0,
                 gd_friction_rate=0.0, ground_offset=0.0, energy_rel_tol=0.0, newton_iter_cap=200, pcg_max_iter_ratio=0.3,
                 gravity=(0.0, -9.81, 0.0))
    eng = Engine(cfg); eng.set_log_level(0)
    eng.load_mesh_from_data(V, F, 3, 3, "ABD", np.eye(4), 1e7, "Fixed")
    lo = np.array([-0.1, -0.005, -0.1]); hi = np.array([0.1, top + 0.03, 0.1])
    eng.add_liquid(pts, dx, lo, hi, rho=1000.0, kappa=args.kappa, particle_volume=hs ** 3, push_apart=args.push, j_mode=args.jmode,
                   mass_floor=args.mass_floor)
    eng.finalize()
    print(f"[m15] tank {W*1e3:.0f} mm square, water {h0*1e3:.0f} mm ({len(pts)} particles), yaw {args.yaw} deg, shift {args.shift} dx, "
          f"dx {dx*1e3:.1f} mm, dt {args.dt*1e3:.1f} ms", flush=True)
    xs, ts, t, wall0, step = [], [], 0.0, time.time(), 0
    nsteps = round(args.duration / args.dt)
    for step in range(nsteps + 1):
        if step % 8 == 0 or step == nsteps:
            x = eng.get_liquid_particles()
            if not np.isfinite(x).all(): raise RuntimeError(f"nonfinite liquid at step {step}")
            xs.append(x.copy()); ts.append(step * args.dt)
            if len(xs) >= 2 and (step % 100 == 0 or step == nsteps):   # running report: speed over the last record interval
                v = np.linalg.norm(xs[-1] - xs[-2], axis=1) / (ts[-1] - ts[-2]) * 1e3
                print(f"[m15]   t {step*args.dt:.2f} s: speed median {np.median(v):.4f} p90 {np.percentile(v, 90):.4f} max {v.max():.3f} mm/s; wall {time.time()-wall0:.0f} s", flush=True)
                if step % 200 == 0 or step == nsteps:   # where is the motion: wall layer, floor layer, surface layer, interior (tank frame)
                    lc = (xs[-1] - off) @ R
                    dw_ = np.minimum(hw - np.abs(lc[:, 0]), hw - np.abs(lc[:, 2])) * 1e3
                    df_ = (lc[:, 1] - fy) * 1e3
                    dt_ = (lc[:, 1].max() - lc[:, 1]) * 1e3
                    parts = []
                    for nm, m_ in (("wall<5", (dw_ < 5) & (df_ >= 5) & (dt_ >= 5)), ("floor<5", (df_ < 5) & (dw_ >= 5)), ("top<5", (dt_ < 5) & (dw_ >= 5)),
                                   ("interior", (dw_ >= 10) & (df_ >= 10) & (dt_ >= 10))):
                        if m_.any(): parts.append(f"{nm} med {np.median(v[m_]):.4f} p90 {np.percentile(v[m_], 90):.4f} (n {int(m_.sum())})")
                    print("[m15]     regions: " + "; ".join(parts), flush=True)
        if step != nsteps: eng.step()
    x, t = np.asarray(xs), np.asarray(ts)
    first = np.searchsorted(t, t[-1] - 3.0)
    sp = (np.linalg.norm(np.diff(x[first:], axis=0), axis=2) / np.diff(t[first:])[:, None]).mean(0) * 1e3   # mm/s per particle, window mean
    drift = np.linalg.norm(x[-1] - x[first], axis=1) * 1e3
    loc = (x[-1] - off) @ R                                               # tank frame
    dwall = np.minimum(hw - np.abs(loc[:, 0]), hw - np.abs(loc[:, 2])) * 1e3
    dfloor = (loc[:, 1] - fy) * 1e3
    res = dict(particles=len(pts), yaw=args.yaw, shift=args.shift, median_speed_mm_s=float(np.median(sp)), p90_speed_mm_s=float(np.percentile(sp, 90)),
               max_speed_mm_s=float(sp.max()), drift_median_mm=float(np.median(drift)), drift_p90_mm=float(np.percentile(drift, 90)),
               wall_s=time.time() - wall0, newton_iters=int(eng._engine.get_total_newton_iters()),
               environment={k: v for k, v in os.environ.items() if k.startswith("LIQ_")}, parameters=vars(args))
    for name, msk in (("near_wall_lt5mm", dwall < 5), ("near_floor_lt5mm", dfloor < 5), ("interior", (dwall >= 10) & (dfloor >= 10))):
        res[name] = dict(n=int(msk.sum()), median=float(np.median(sp[msk])) if msk.any() else None,
                         p90=float(np.percentile(sp[msk], 90)) if msk.any() else None)
    # depth profile (distance below the top particle, particles >= 10 mm from the walls): is the residual motion at the surface?
    dtop = (loc[:, 1].max() - loc[:, 1]) * 1e3
    prof = {}
    for lo_, hi_ in ((0, 5), (5, 10), (10, 20), (20, 40), (40, 1e9)):
        msk = (dtop >= lo_) & (dtop < hi_) & (dwall >= 10)
        if msk.any(): prof[f"{lo_}-{min(hi_, 999):.0f}mm"] = dict(n=int(msk.sum()), median=round(float(np.median(sp[msk])), 4), p90=round(float(np.percentile(sp[msk], 90)), 4))
    res["depth_profile"] = prof
    # coarse flow field over the last 1 s (tank frame, z-averaged): is the residual a coherent circulation?
    try:
        f1 = np.searchsorted(t, t[-1] - 1.0)
        vel = (x[-1] - x[f1]) / (t[-1] - t[f1])                      # m/s per particle, 1 s mean
        vloc = vel @ R; lc = (x[-1] - off) @ R
        nx, ny = 8, 6
        ex = np.linspace(-hw, hw, nx + 1); ey = np.linspace(fy, lc[:, 1].max() + 1e-9, ny + 1)
        ix = np.clip(np.searchsorted(ex, lc[:, 0]) - 1, 0, nx - 1); iy = np.clip(np.searchsorted(ey, lc[:, 1]) - 1, 0, ny - 1)
        cnt = np.zeros((ny, nx)); vx = np.zeros((ny, nx)); vy = np.zeros((ny, nx)); vz = np.zeros((ny, nx))
        np.add.at(cnt, (iy, ix), 1); np.add.at(vx, (iy, ix), vloc[:, 0]); np.add.at(vy, (iy, ix), vloc[:, 1]); np.add.at(vz, (iy, ix), vloc[:, 2])
        cnt[cnt == 0] = 1
        print("[m15] mean flow field (last 1 s, mm/s, rows top->bottom, cols -x..+x; each cell 'vx,vy'):", flush=True)
        for j in range(ny - 1, -1, -1):
            print("[m15]   " + " | ".join(f"{vx[j, i] / cnt[j, i] * 1e3:+6.2f},{vy[j, i] / cnt[j, i] * 1e3:+6.2f}" for i in range(nx)), flush=True)
        print("[m15]   z-component rms over cells: %.3f mm/s; |mean field| rms: %.3f mm/s" % (np.sqrt(np.mean((vz / cnt) ** 2)) * 1e3, np.sqrt(np.mean((vx / cnt) ** 2 + (vy / cnt) ** 2)) * 1e3), flush=True)
    except Exception as e:
        print(f"[m15] flow field failed: {e}", flush=True)
    # spectrum of the bulk motion over the last 4 s: coherent sloshing (a peak near the tank's first mode) or broadband noise?
    try:
        f0 = np.searchsorted(t, t[-1] - 4.0)
        tt = t[f0:]; cm = x[f0:].mean(1); dtop_all = (x[f0:, :, 1].max(1, keepdims=True) - x[f0:, :, 1])
        topm = np.array([x[f0 + i][dtop_all[i] * 1e3 < 5].mean(0) for i in range(len(tt))])
        spec = {}
        for name, sig in (("cm_x", cm[:, 0]), ("cm_z", cm[:, 2]), ("top_y", topm[:, 1]), ("top_x", topm[:, 0])):
            sg = sig - np.polyval(np.polyfit(tt, sig, 1), tt)
            F = np.fft.rfft(sg * np.hanning(len(sg))); fr = np.fft.rfftfreq(len(sg), d=float(np.median(np.diff(tt))))
            k = int(np.argmax(np.abs(F[1:])) + 1)
            spec[name] = dict(peak_hz=round(float(fr[k]), 2), amp_um=round(float(2 * np.abs(F[k]) / np.sum(np.hanning(len(sg))) * 1e6), 3), rms_um=round(float(sg.std() * 1e6), 3))
        res["spectrum_last4s"] = spec
        print("[m15] spectrum (last 4 s, detrended): " + "; ".join(f"{k}: peak {v['peak_hz']} Hz amp {v['amp_um']} um rms {v['rms_um']} um" for k, v in spec.items()), flush=True)
        np.savez(os.path.join(args.out, "traj.npz"), t=t, cm=x.mean(1), top=np.array([x[i][((x[i, :, 1].max() - x[i, :, 1]) * 1e3) < 5].mean(0) for i in range(len(t))]))
    except Exception as e:
        print(f"[m15] spectrum failed: {e}", flush=True)
    print("[m15] depth profile (mm below the top, >= 10 mm from walls): " + "; ".join(f"{k}: med {v['median']} p90 {v['p90']} (n {v['n']})" for k, v in prof.items()), flush=True)
    print("[m15] " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k not in ("environment", "parameters")}), flush=True)
    json.dump(res, open(os.path.join(args.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

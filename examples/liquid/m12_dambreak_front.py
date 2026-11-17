"""M12: dam break in a channel -> surge-front speed against the classic experiments.

A water column of width a and height H (Martin & Moyce 1952: a = 57 mm, H = 57 or 114 mm) stands against the back
wall of a frictionless 60 mm channel on the ground plane and is released at t = 0 (no gate). The front position is
the 99.5th percentile of the particle x; its mean speed over t* = t sqrt(g/H) in [1, 2.5] is reported as v/sqrt(gH),
the quantity tabulated by Lobovsky et al. 2014 (Table 1): Martin & Moyce H 57 mm 1.48, H 114 mm 1.69; Dressler 1954
H 55 / 110 / 220 mm 1.54 / 1.70 / 1.74; Lobovsky H 300 / 600 mm 1.56 / 1.34; Ritter's frictionless shallow-water
solution 2.0. The liquid's own ground friction (add_liquid ground_mu) is a scene choice the experiments do not have.
    python examples/liquid/m12_dambreak_front.py --H 0.114 --out out/liquid_m12
"""
import argparse, json, os, sys, time
if __name__ == "__main__": import os as _os; _os.environ.setdefault("LIQ_UG", "0")   # main-path example, when run as a script: pinned to the main liquid path (the engine default is the UG mode since 2026-11-18)
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from stiff_physics.engine import Engine, Config  # noqa: E402
from m2_box import cuboid  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=float, default=0.057, help="column width along the channel (m)")
    ap.add_argument("--H", type=float, default=0.114, help="column height (m)")
    ap.add_argument("--w", type=float, default=0.06, help="channel width (m)")
    ap.add_argument("--L", type=float, default=0.65, help="channel length (m)")
    ap.add_argument("--dx", type=float, default=0.005)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--h", type=float, default=0.0, help="particle spacing (m); default dx/2 (8 per cell)")
    ap.add_argument("--tstar", type=float, default=2.6, help="run until t sqrt(g/H) reaches this")
    ap.add_argument("--kappa", type=float, default=1e5)
    ap.add_argument("--push", type=float, default=0.9)
    ap.add_argument("--jmode", type=int, default=0)
    ap.add_argument("--ground-mu", type=float, default=0.3, help="water-table friction (lab default 0.3)")
    ap.add_argument("--jitter", type=float, default=0.3)
    ap.add_argument("--energy-rel-tol", type=float, default=0.0, help="line-search acceptance tolerance relative to |E| (0 = strict decrease)")
    ap.add_argument("--log-level", type=int, default=0)
    ap.add_argument("--dump-every", type=int, default=0, help="save frames/f_<step>.npz (x, v) every N steps (video)")
    ap.add_argument("--out", default="out/liquid_m12")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dx, hs, g, H, a, w, L = args.dx, (args.h if args.h > 0 else args.dx / 2.0), 9.81, args.H, args.a, args.w, args.L
    m = 0.6 * hs
    ax = [np.arange(m, a - m + 1e-9, hs), np.arange(m, H + 1e-9, hs), np.arange(-w / 2 + m, w / 2 - m + 1e-9, hs)]
    X, Y, Z = np.meshgrid(*ax, indexing="ij"); pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    pts = pts + np.random.default_rng(0).uniform(-args.jitter * hs, args.jitter * hs, pts.shape)
    cfg = Config(dt=args.dt, preconditioner_type=0, absolute_dhat=5e-4, newton_tol=1e-4, pcg_tol=1e-6, friction_rate=0.0,
                 gd_friction_rate=0.0, ground_offset=0.0, energy_rel_tol=args.energy_rel_tol, newton_iter_cap=200, pcg_max_iter_ratio=0.3,
                 gravity=(0.0, -9.81, 0.0))
    eng = Engine(cfg); eng.set_log_level(args.log_level)
    t_w, hw, y0 = 0.01, H + 0.03, 0.0006       # walls stand 0.6 mm above the ground plane (> dhat 0.5 mm; touching it is an IPC invariant violation)
    parts = [cuboid([-t_w, y0, -w / 2 - t_w], [0.0, hw, w / 2 + t_w]),            # back wall
             cuboid([0.001, y0, w / 2], [L, hw, w / 2 + t_w]),                  # side walls (1 mm from the back / end wall:
             cuboid([0.001, y0, -w / 2 - t_w], [L, hw, -w / 2]),               #  coincident fixed faces are a zero-distance contact)
             cuboid([L + 0.001, y0, -w / 2 - t_w], [L + 0.001 + t_w, hw, w / 2 + t_w])]   # end wall
    for v, f in parts:
        eng.load_mesh_from_data(v, f, 3, 3, "ABD", np.eye(4), 1e7, "Fixed")
    lo = np.array([-0.02, -0.02, -w / 2 - 0.015]); hi = np.array([L + 0.02, hw + 0.02, w / 2 + 0.015])
    eng.add_liquid(pts, dx, lo, hi, rho=1000.0, kappa=args.kappa, particle_volume=hs ** 3, push_apart=args.push, j_mode=args.jmode,
                   ground_mu=args.ground_mu)
    eng.finalize()
    print(f"[m12] column a {a*1e3:.0f} x H {H*1e3:.0f} mm, channel {w*1e3:.0f} mm, {len(pts)} particles, dx {dx*1e3:.1f} mm, "
          f"dt {args.dt*1e3:.1f} ms, water-table friction {args.ground_mu}", flush=True)
    ts, xf = [], []
    t, wall0 = 0.0, time.time()
    t_end = args.tstar * np.sqrt(H / g)
    step = 0
    while True:
        x = eng.get_liquid_particles()
        if args.dump_every and step % args.dump_every == 0:
            os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)
            np.savez(os.path.join(args.out, "frames", f"f_{step:05d}.npz"), x=x.astype(np.float32), v=eng.get_liquid_velocities().astype(np.float32), t=t)
        step += 1
        ts.append(t); xf.append(float(np.percentile(x[:, 0], 99.5)))
        if t >= t_end - 1e-12: break
        eng.step(); t += args.dt
    ts, xf = np.array(ts), np.array(xf)
    tst = ts * np.sqrt(g / H)
    sel = (tst >= 1.0) & (tst <= 2.5)
    v = np.polyfit(ts[sel], xf[sel], 1)[0] if sel.sum() >= 3 else float("nan")
    vn = v / np.sqrt(g * H)
    for q in (0.5, 1.0, 1.5, 2.0, 2.5):
        k = int(np.argmin(np.abs(tst - q)))
        print(f"[m12]   t* {tst[k]:4.2f}: front x/H {xf[k]/H:5.2f} (x {xf[k]*1e3:6.1f} mm)", flush=True)
    print(f"[m12] front speed over t* 1-2.5: {v:.3f} m/s = {vn:.2f} sqrt(gH)  (experiments 1.48-1.74 for H 55-220 mm, "
          f"Ritter frictionless 2.0); wall {time.time()-wall0:.0f} s", flush=True)
    json.dump(dict(H=H, a=a, dx=dx, dt=args.dt, ground_mu=args.ground_mu, v=v, v_norm=vn, t=ts.tolist(), x_front=xf.tolist()),
              open(os.path.join(args.out, "front.json"), "w"))


if __name__ == "__main__":
    main()

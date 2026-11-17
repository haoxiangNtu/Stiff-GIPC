"""M6: sloshing decay in the lab beaker (numerical damping vs time step / resolution / liquid model).

The beaker is a Fixed ABD body. The water starts at rest with its free surface tilted by --tilt degrees about
the beaker axis (first antisymmetric sloshing mode) and is released. The horizontal centre of mass along the
tilt direction is sampled every --sample seconds; a damped cosine fit gives the frequency and damping ratio.
Analytic first mode for this beaker: f = (1/2pi) sqrt(g k tanh(k h)), k = 1.8412 / r_in.

    python examples/liquid/m6_slosh.py --dt 0.001 --out out/liquid_m6_dt1
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "lab_tactile"))
from stiff_physics.engine import Engine, Config  # noqa: E402
from lab_assets import build_all  # noqa: E402
from m4_beaker_tilt import cavity  # noqa: E402


def fit_damped(t, y):
    """least-squares fit y = C + A exp(-zeta w t) cos(w sqrt(1-zeta^2) t + phi); coarse grid over (f, zeta) then linear in the rest"""
    best = None
    for f in np.linspace(1.0, 10.0, 181):
        for z in np.concatenate([np.linspace(0.0, 0.2, 81), np.linspace(0.21, 1.0, 40)]):
            w = 2 * np.pi * f; wd = w * np.sqrt(max(1e-9, 1 - z * z)); e = np.exp(-z * w * t)
            M = np.c_[np.ones_like(t), e * np.cos(wd * t), e * np.sin(wd * t)]
            c, res, *_ = np.linalg.lstsq(M, y, rcond=None)
            r = float(((M @ c - y) ** 2).sum())
            if best is None or r < best[0]: best = (r, f, z, c)
    r, f, z, c = best
    return f, z, float(np.hypot(c[1], c[2])), 1.0 - r / max(1e-30, float(((y - y.mean()) ** 2).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=0.005)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--fill", type=float, default=0.6)
    ap.add_argument("--ppc", type=int, default=8)
    ap.add_argument("--kappa", type=float, default=1e5)
    ap.add_argument("--tilt", type=float, default=10.0, help="initial free-surface tilt (deg)")
    ap.add_argument("--duration", type=float, default=3.0)
    ap.add_argument("--sample", type=float, default=0.01)
    ap.add_argument("--newton-tol", type=float, default=1e-4)
    ap.add_argument("--pcg-tol", type=float, default=1e-6)
    ap.add_argument("--push", type=float, default=0.9)
    ap.add_argument("--jmode", type=int, default=3)
    ap.add_argument("--apic", type=float, default=1.0)
    ap.add_argument("--mass-floor", type=float, default=0.05)
    ap.add_argument("--dhat", type=float, default=5e-4)
    ap.add_argument("--out", default="out/liquid_m6")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    props = build_all(verbose=False)
    p = props["beaker_mix"]
    T0 = np.eye(4); T0[:3, 3] = [0.20, 0.0005, 0.20]
    wv0 = p.verts @ T0[:3, :3].T + T0[:3, 3]
    c_xz, r_in, floor_y, rim_y = cavity(wv0)
    dx = args.dx
    cfg = Config(dt=args.dt, preconditioner_type=0, absolute_dhat=args.dhat, newton_tol=args.newton_tol, pcg_tol=args.pcg_tol,
                 friction_rate=0.0, gd_friction_rate=0.0, ground_offset=0.0, energy_rel_tol=0.0,
                 newton_iter_cap=200, pcg_max_iter_ratio=0.3)
    eng = Engine(cfg); eng.set_log_level(0)
    eng.load_mesh_from_data(wv0, p.faces, verts_per_face=3, dimensions=3, body_type="ABD", transform=np.eye(4),
                            young_modulus=1e8, boundary_type="Fixed")
    # water: cylinder up to the mean level h, free surface tilted about the axis along +z
    h = floor_y + args.fill * (rim_y - floor_y)
    n_sub = max(1, int(round(args.ppc ** (1 / 3)))); hs = dx / n_sub; m = 0.6 * dx
    ax = np.arange(c_xz[0] - r_in + m, c_xz[0] + r_in - m, hs); ay = np.arange(floor_y + m, rim_y, hs); az = np.arange(c_xz[1] - r_in + m, c_xz[1] + r_in - m, hs)
    gx, gy, gz = np.meshgrid(ax, ay, az, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)
    pts = pts + np.random.default_rng(0).uniform(-0.3 * hs, 0.3 * hs, pts.shape)
    s = np.tan(np.radians(args.tilt))
    keep = (np.hypot(pts[:, 0] - c_xz[0], pts[:, 2] - c_xz[1]) < r_in - m) & (pts[:, 1] < h + s * (pts[:, 2] - c_xz[1]))
    pts = pts[keep]
    lo = np.array([0.10, -0.02, 0.10]); hi = np.array([0.30, 0.15, 0.30])
    eng.add_liquid(pts, dx, lo, hi, rho=1000.0, kappa=args.kappa, particle_volume=hs ** 3, push_apart=args.push, j_mode=args.jmode,
                   apic_ratio=args.apic, mass_floor=args.mass_floor)
    eng.finalize()
    depth = h - floor_y
    k = 1.8412 / r_in; f_th = np.sqrt(9.81 * k * np.tanh(k * depth)) / (2 * np.pi)
    print(f"[m6] r_in {r_in*1e3:.1f} mm depth {depth*1e3:.1f} mm tilt {args.tilt} deg -> {len(pts)} particles; analytic first mode {f_th:.2f} Hz; dt {args.dt*1e3:.2f} ms dx {dx*1e3:.2f} mm", flush=True)

    t, step, t_next = 0.0, 0, 0.0
    ts, zs, wall0, newton0 = [], [], time.time(), eng._engine.get_total_newton_iters()
    while t < args.duration - 1e-12:
        if t >= t_next - 1e-12:
            x = eng.get_liquid_particles(); ts.append(t); zs.append(float(x[:, 2].mean() - c_xz[1])); t_next += args.sample
        eng.step(); t += args.dt; step += 1
    ts, zs = np.array(ts), np.array(zs) * 1e3
    wall = time.time() - wall0; newton = (eng._engine.get_total_newton_iters() - newton0) / max(1, step)
    sel = ts >= 0.05            # skip the first instant (release transient)
    f, z, A, r2 = fit_damped(ts[sel], zs[sel])
    print(f"[m6] fit: f {f:.2f} Hz (analytic {f_th:.2f}), damping ratio {z:.3f}, amplitude {A:.2f} mm, R^2 {r2:.3f}; "
          f"per-cycle amplitude loss {100*(1-np.exp(-2*np.pi*z/np.sqrt(max(1e-9,1-z*z)))):.1f}%; wall {wall:.0f} s, {newton:.1f} Newton/step", flush=True)
    np.savez(os.path.join(args.out, "slosh.npz"), t=ts, z_mm=zs, f=f, zeta=z, A=A, r2=r2, f_th=f_th, dt=args.dt, dx=dx, wall=wall)
    json.dump(dict(f=f, zeta=z, A=A, r2=r2, f_th=f_th, dt=args.dt, dx=dx, wall=wall, newton_per_step=newton, n=len(pts)),
              open(os.path.join(args.out, "summary.json"), "w"))


if __name__ == "__main__":
    main()

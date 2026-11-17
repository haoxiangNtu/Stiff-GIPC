"""M6: sloshing decay in the lab beaker (numerical damping vs time step / resolution / liquid model).

The beaker is a Fixed ABD body. The water starts at rest with its free surface tilted by --tilt degrees about
the beaker axis (first antisymmetric sloshing mode) and is released. The horizontal centre of mass along the
tilt direction is sampled every --sample seconds; a damped cosine fit gives the frequency and damping ratio.
Analytic first mode for this beaker: f = (1/2pi) sqrt(g k tanh(k h)), k = 1.8412 / r_in.

    python examples/liquid/m6_slosh.py --dt 0.001 --out out/liquid_m6_dt1
"""
import argparse, json, os, sys, time
if __name__ == "__main__": import os as _os; _os.environ.setdefault("LIQ_UG", "0")   # main-path example, when run as a script: pinned to the main liquid path (the engine default is the UG mode since 2026-11-18)
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "lab_tactile"))
from stiff_physics.engine import Engine, Config  # noqa: E402
from lab_assets import build_all  # noqa: E402
from m4_beaker_tilt import cavity, beaker_profile  # noqa: E402


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
    ap.add_argument("--margin", type=float, default=0.6, help="seeding gap to the beaker wall and floor, in dx (0.25 = h/2: the natural 2-per-cell lattice)")
    ap.add_argument("--duration", type=float, default=3.0)
    ap.add_argument("--sample", type=float, default=0.01)
    ap.add_argument("--newton-tol", type=float, default=1e-4)
    ap.add_argument("--pcg-tol", type=float, default=1e-6)
    ap.add_argument("--newton-cap", type=int, default=0, help="cap the Newton iterations per step (0 = engine default 200)")
    ap.add_argument("--push", type=float, default=0.9)
    ap.add_argument("--jmode", type=int, default=3)
    ap.add_argument("--apic", type=float, default=1.0)
    ap.add_argument("--mass-floor", type=float, default=0.05)
    ap.add_argument("--dhat", type=float, default=5e-4)
    ap.add_argument("--legacy-cavity", action="store_true", help="the old geometry estimate (vertex-mean centre, minimum upper vertex radius; before 10-26)")
    ap.add_argument("--kappa-mean-mass", type=float, default=0.0, help="pin the IPC barrier mass scale (kg); 0 = engine default (mesh mean mass)")
    ap.add_argument("--pcg-ratio", type=float, default=0.3, help="PCG max-iteration ratio (engine default 0.3)")
    ap.add_argument("--dump-every", type=int, default=0, help="save frames/f_<step>.npz (x, v) every N steps (video)")
    ap.add_argument("--out", default="out/liquid_m6")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    props = build_all(verbose=False)
    p = props["beaker_mix"]
    T0 = np.eye(4); T0[:3, 3] = [0.20, 0.0005, 0.20]
    wv0 = p.verts @ T0[:3, :3].T + T0[:3, 3]
    c_xz, r_in, floor_y, rim_y = cavity(wv0)
    r_slope = 0.0
    if not args.legacy_cavity:
        c_xz = T0[[0, 2], 3].copy()
        floor_y, r_in, r_slope, rfit = beaker_profile(wv0, p.faces, c_xz, rim_y)
        print(f"[m6] beaker (ray cast): axis ({c_xz[0]*1e3:.2f}, {c_xz[1]*1e3:.2f}) mm, inner floor {floor_y*1e3:.3f} mm, wall radius {r_in*1e3:.3f} mm + "
              f"{r_slope:.4f} x height (fit residual {rfit*1e6:.1f} um)", flush=True)
    r_wall = lambda y: r_in + r_slope * (y - floor_y)
    dx = args.dx
    cfg = Config(dt=args.dt, preconditioner_type=0, absolute_dhat=args.dhat, newton_tol=args.newton_tol, pcg_tol=args.pcg_tol,
                 friction_rate=0.0, gd_friction_rate=0.0, ground_offset=0.0, energy_rel_tol=0.0,
                 newton_iter_cap=(args.newton_cap if args.newton_cap > 0 else 200), pcg_max_iter_ratio=args.pcg_ratio, kappa_mean_mass=args.kappa_mean_mass)
    eng = Engine(cfg); eng.set_log_level(0)
    eng.load_mesh_from_data(wv0, p.faces, verts_per_face=3, dimensions=3, body_type="ABD", transform=np.eye(4),
                            young_modulus=1e8, boundary_type="Fixed")
    # water: cylinder up to the mean level h, free surface tilted about the axis along +z
    h = floor_y + args.fill * (rim_y - floor_y)
    n_sub = max(1, int(round(args.ppc ** (1 / 3)))); hs = dx / n_sub; m = args.margin * dx
    rmax = r_wall(rim_y)
    ax = np.arange(c_xz[0] - rmax + m, c_xz[0] + rmax - m, hs); ay = np.arange(floor_y + m, rim_y, hs); az = np.arange(c_xz[1] - rmax + m, c_xz[1] + rmax - m, hs)
    gx, gy, gz = np.meshgrid(ax, ay, az, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)
    pts = pts + np.random.default_rng(0).uniform(-0.3 * hs, 0.3 * hs, pts.shape)
    s = np.tan(np.radians(args.tilt))
    keep = (np.hypot(pts[:, 0] - c_xz[0], pts[:, 2] - c_xz[1]) < r_wall(pts[:, 1]) - m) & (pts[:, 1] < h + s * (pts[:, 2] - c_xz[1]))
    pts = pts[keep]
    lo = np.array([0.10, -0.02, 0.10]); hi = np.array([0.30, 0.15, 0.30])
    eng.add_liquid(pts, dx, lo, hi, rho=1000.0, kappa=args.kappa, particle_volume=hs ** 3, push_apart=args.push, j_mode=args.jmode,
                   apic_ratio=args.apic, mass_floor=args.mass_floor)
    eng.finalize()
    depth = h - floor_y
    r_s = r_wall(h)   # a 2 deg draft: the first mode follows the free-surface radius
    k = 1.8412 / r_s; f_th = np.sqrt(9.81 * k * np.tanh(k * depth)) / (2 * np.pi)
    print(f"[m6] r_in {r_in*1e3:.1f} mm (surface {r_s*1e3:.1f}) depth {depth*1e3:.1f} mm tilt {args.tilt} deg -> {len(pts)} particles; analytic first mode {f_th:.2f} Hz; "
          f"dt {args.dt*1e3:.2f} ms dx {dx*1e3:.2f} mm", flush=True)

    t, step, t_next = 0.0, 0, 0.0
    ts, zs, wall0, newton0 = [], [], time.time(), eng._engine.get_total_newton_iters()
    x_init, J_init = eng.get_liquid_particles().copy(), eng.get_liquid_J().copy()
    while t < args.duration - 1e-12:
        if t >= t_next - 1e-12:
            x = eng.get_liquid_particles(); ts.append(t); zs.append(float(x[:, 2].mean() - c_xz[1])); t_next += args.sample
        if args.dump_every and step % args.dump_every == 0:
            os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)
            np.savez(os.path.join(args.out, "frames", f"f_{step:05d}.npz"), x=eng.get_liquid_particles().astype(np.float32), v=eng.get_liquid_velocities().astype(np.float32), t=t)
        eng.step(); t += args.dt; step += 1
    ts, zs = np.array(ts), np.array(zs) * 1e3
    wall = time.time() - wall0; newton = (eng._engine.get_total_newton_iters() - newton0) / max(1, step)
    sel = ts >= 0.05            # skip the first instant (release transient)
    f, z, A, r2 = fit_damped(ts[sel], zs[sel])
    print(f"[m6] fit: f {f:.2f} Hz (analytic {f_th:.2f}), damping ratio {z:.3f}, amplitude {A:.2f} mm, R^2 {r2:.3f}; "
          f"per-cycle amplitude loss {100*(1-np.exp(-2*np.pi*z/np.sqrt(max(1e-9,1-z*z)))):.1f}%; wall {wall:.0f} s, {newton:.1f} Newton/step", flush=True)
    np.savez(os.path.join(args.out, "slosh.npz"), t=ts, z_mm=zs, f=f, zeta=z, A=A, r2=r2, f_th=f_th, dt=args.dt, dx=dx, wall=wall)
    np.savez(os.path.join(args.out, "particles.npz"), x0=x_init, J0=J_init, x=eng.get_liquid_particles(), v=eng.get_liquid_velocities(), J=eng.get_liquid_J(),
             c_xz=np.array(c_xz), r_in=r_in, r_slope=r_slope, floor_y=floor_y, h=h)
    json.dump(dict(f=f, zeta=z, A=A, r2=r2, f_th=f_th, dt=args.dt, dx=dx, wall=wall, newton_per_step=newton, n=len(pts)),
              open(os.path.join(args.out, "summary.json"), "w"))


if __name__ == "__main__":
    main()

"""M7: free drop in zero gravity -> calibrate the surface tension (Rayleigh n=2 mode).

A water ellipsoid (semi-axes R(1+e), R, R(1+e)^-1/2 ... volume-preserving) floats without gravity or solids.
Surface tension makes it oscillate between prolate and oblate; the n = 2 frequency gives the effective sigma:
    omega^2 = 8 sigma / (rho R^3)   ->   sigma_eff = (2 pi f)^2 rho R^3 / 8      (water: 0.072 N/m)
    python examples/liquid/m7_drop.py --gamma 1.0 --out out/liquid_m7
"""
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from stiff_physics.engine import Engine, Config  # noqa: E402
from m6_slosh import fit_damped  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=0.005)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--R", type=float, default=0.015)
    ap.add_argument("--e", type=float, default=0.25, help="initial elongation along x")
    ap.add_argument("--duration", type=float, default=2.0)
    ap.add_argument("--sample", type=float, default=0.01)
    ap.add_argument("--kappa", type=float, default=1e5)
    ap.add_argument("--push", type=float, default=0.9)
    ap.add_argument("--jmode", type=int, default=3)
    ap.add_argument("--out", default="out/liquid_m7")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dx = args.dx; hs = dx / 2.0
    c = np.array([0.2, 0.2, 0.2])
    ax = np.array([args.R * (1 + args.e), args.R / np.sqrt(1 + args.e), args.R / np.sqrt(1 + args.e)])   # volume-preserving
    g = np.arange(-ax.max(), ax.max() + hs, hs)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij"); pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    pts = pts + np.random.default_rng(0).uniform(-0.3 * hs, 0.3 * hs, pts.shape)
    pts = pts[((pts / ax) ** 2).sum(1) <= 1.0] + c
    cfg = Config(dt=args.dt, preconditioner_type=0, absolute_dhat=5e-4, newton_tol=1e-4, pcg_tol=1e-6, friction_rate=0.0,
                 gd_friction_rate=0.0, ground_offset=0.0, energy_rel_tol=0.0, newton_iter_cap=200, pcg_max_iter_ratio=0.3, gravity=(0.0, -1e-3, 0.0))
    eng = Engine(cfg); eng.set_log_level(0)
    # a small fixed box far from the drop: the collision BVH needs at least one surface
    bv = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float) * 0.02 + np.array([0.6, 0.6, 0.6])
    bf = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    eng.load_mesh_from_data(bv, bf, verts_per_face=3, dimensions=3, body_type="ABD", transform=np.eye(4), young_modulus=1e8, boundary_type="Fixed")
    lo = c - 0.06; hi = c + 0.06; lo[1] = -0.02     # the liquid grid must reach below the ground plane (as in every lab/test scene)
    eng.add_liquid(pts, dx, lo, hi, rho=1000.0, kappa=args.kappa, particle_volume=hs ** 3, push_apart=args.push, j_mode=args.jmode)
    eng.finalize()
    R0 = np.cbrt(len(pts) * hs ** 3 * 3 / (4 * np.pi))
    print(f"[m7] {len(pts)} particles, equivalent radius {R0*1e3:.1f} mm, e {args.e}; water sigma 0.072 -> f2 {np.sqrt(8*0.072/(1000*R0**3))/(2*np.pi):.2f} Hz", flush=True)
    t, t_next, ts, ys = 0.0, 0.0, [], []
    wall0 = time.time()
    while t < args.duration - 1e-12:
        if t >= t_next - 1e-12:
            x = eng.get_liquid_particles(); d = x - x.mean(0)
            ts.append(t); ys.append(float(np.sqrt((d[:, 0] ** 2).mean()) - np.sqrt((d[:, 2] ** 2).mean())) * 1e3); t_next += args.sample
        eng.step(); t += args.dt
    ts, ys = np.array(ts), np.array(ys)
    sel = ts >= 0.02
    f, z, A, r2 = fit_damped(ts[sel], ys[sel])
    sig = (2 * np.pi * f) ** 2 * 1000.0 * R0 ** 3 / 8.0
    x = eng.get_liquid_particles(); d = x - x.mean(0)
    print(f"[m7] fit: f2 {f:.2f} Hz, damping {z:.3f}, amplitude {A:.2f} mm, R^2 {r2:.3f} -> sigma_eff {sig:.4f} N/m (water 0.072); "
          f"final RMS extents x/y/z {np.sqrt((d**2).mean(0))*1e3} mm; wall {time.time()-wall0:.0f} s", flush=True)
    np.savez(os.path.join(args.out, "drop.npz"), t=ts, y=ys, f=f, zeta=z, sigma=sig, R0=R0)
    json.dump(dict(f=f, zeta=z, A=A, r2=r2, sigma=sig, R0=R0), open(os.path.join(args.out, "summary.json"), "w"))


if __name__ == "__main__":
    main()

"""M13: a tank drains through a square hole in its floor -> Torricelli outflow, and (with --lid) the missing air.

Open top: the outflow should follow Torricelli, Q = C_d A sqrt(2 g h), with C_d ~ 0.6 for a sharp-edged orifice; the effective
C_d = Q / (A sqrt(2 g h)) is reported over time. With --lid the tank is a closed bottle standing on its neck: real water can only
leave as fast as air bubbles back in through the same hole (glugging); Clanet & Searby (J. Fluid Mech. 510, 2004) measured
T_e ~ 3.0 L / sqrt(g D0) * (D0/d)^(5/2) for a cylinder of diameter D0 and liquid length L emptying through a hole of diameter d.
The engine's liquid has no air phase (empty space is at zero pressure), so the lidded tank is expected to drain like the open one.
    python examples/liquid/m13_efflux.py [--lid] --out out/liquid_m13
"""
import argparse, json, os, sys, time
if __name__ == "__main__": import os as _os; _os.environ.setdefault("LIQ_UG", "0")   # main-path example, when run as a script: pinned to the main liquid path (the engine default is the UG mode since 2026-11-18)
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from stiff_physics.engine import Engine, Config  # noqa: E402
from m2_box import cuboid, open_box  # noqa: E402


def merge(parts):
    V, F, off = [], [], 0
    for v, f in parts:
        V.append(v); F.append(f + off); off += len(v)
    return np.vstack(V), np.vstack(F)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inner", type=float, default=0.08, help="tank inner width (square, m)")
    ap.add_argument("--hole", type=float, default=0.02, help="hole width (square, m)")
    ap.add_argument("--h0", type=float, default=0.10, help="initial water height (m)")
    ap.add_argument("--floor-y", type=float, default=0.15, help="tank floor height above the ground (m)")
    ap.add_argument("--lid", action="store_true", help="close the top (a bottle standing on its neck)")
    ap.add_argument("--dx", type=float, default=0.005)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--kappa", type=float, default=1e5)
    ap.add_argument("--push", type=float, default=0.9)
    ap.add_argument("--jmode", type=int, default=0)
    ap.add_argument("--out", default="out/liquid_m13")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dx, hs, g = args.dx, args.dx / 2.0, 9.81
    W, s, h0, fy, th = args.inner, args.hole, args.h0, args.floor_y, 0.01
    top = fy + h0 + 0.03
    hw, hh = W / 2, s / 2
    tank = [cuboid([-hw - th, fy - th, -hw - th], [-hh, fy, hw + th]), cuboid([hh, fy - th, -hw - th], [hw + th, fy, hw + th]),   # floor around the hole
            cuboid([-hh, fy - th, -hw - th], [hh, fy, -hh]), cuboid([-hh, fy - th, hh], [hh, fy, hw + th]),
            cuboid([-hw - th, fy, -hw - th], [-hw, top, hw + th]), cuboid([hw, fy, -hw - th], [hw + th, top, hw + th]),       # walls
            cuboid([-hw, fy, -hw - th], [hw, top, -hw]), cuboid([-hw, fy, hw], [hw, top, hw + th])]
    if args.lid:
        tank.append(cuboid([-hw - th, top, -hw - th], [hw + th, top + th, hw + th]))
    m = 0.6 * hs
    ax = [np.arange(-hw + m, hw - m + 1e-9, hs), np.arange(fy + m, fy + h0 + 1e-9, hs), np.arange(-hw + m, hw - m + 1e-9, hs)]
    X, Y, Z = np.meshgrid(*ax, indexing="ij"); pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    pts = pts + np.random.default_rng(0).uniform(-0.3 * hs, 0.3 * hs, pts.shape)
    cfg = Config(dt=args.dt, preconditioner_type=0, absolute_dhat=5e-4, newton_tol=1e-4, pcg_tol=1e-6, friction_rate=0.0,
                 gd_friction_rate=0.0, ground_offset=0.0, energy_rel_tol=0.0, newton_iter_cap=200, pcg_max_iter_ratio=0.3,
                 gravity=(0.0, -9.81, 0.0))
    eng = Engine(cfg); eng.set_log_level(0)
    V, F = merge(tank)
    eng.load_mesh_from_data(V, F, 3, 3, "ABD", np.eye(4), 1e7, "Fixed")                      # one rigid body: no contacts between its parts
    SV, SF = open_box((0.0, 0.0), 0.22, 0.01, 0.0021, 0.06, 0.001)                          # catch basin, its floor 1.1 mm above the ground (> dhat)
    eng.load_mesh_from_data(SV, SF, 3, 3, "ABD", np.eye(4), 1e7, "Fixed")
    lo = np.array([-0.125, -0.02, -0.125]); hi = np.array([0.125, top + 0.03, 0.125])
    eng.add_liquid(pts, dx, lo, hi, rho=1000.0, kappa=args.kappa, particle_volume=hs ** 3, push_apart=args.push, j_mode=args.jmode)
    eng.finalize()
    pv = hs ** 3; A = s * s; At = W * W
    print(f"[m13] tank {W*1e3:.0f} mm square, hole {s*1e3:.0f} mm, water {h0*1e3:.0f} mm ({len(pts)*pv*1e6:.0f} ml, {len(pts)} particles), "
          f"{'LID (closed bottle)' if args.lid else 'open top'}, dx {dx*1e3:.1f} mm, dt {args.dt*1e3:.1f} ms", flush=True)
    D0, d = np.sqrt(4 * At / np.pi), np.sqrt(4 * A / np.pi)
    Te_open = (np.sqrt(h0) / (0.61 * (A / At) * np.sqrt(g / 2)))
    Te_glug = 3.0 * h0 / np.sqrt(g * D0) * (D0 / d) ** 2.5
    print(f"[m13] references: open-top Torricelli (C_d 0.61) empties in {Te_open:.2f} s; closed bottle (Clanet & Searby, equivalent "
          f"D0 {D0*1e3:.0f} mm, d {d*1e3:.0f} mm) ~{Te_glug:.1f} s", flush=True)
    ts, ns = [], []
    t, wall0 = 0.0, time.time()
    while True:
        x = eng.get_liquid_particles()
        ts.append(t); ns.append(int(((x[:, 1] > fy - th) & (np.abs(x[:, 0]) < hw) & (np.abs(x[:, 2]) < hw)).sum()))
        if t >= args.duration - 1e-12 or ns[-1] == 0: break
        eng.step(); t += args.dt
    ts, ns = np.array(ts), np.array(ns)
    h = ns * pv / At
    for frac in (0.75, 0.5, 0.25, 0.1):
        k = np.nonzero(ns <= frac * ns[0])[0]
        print(f"[m13]   {int((1-frac)*100):3d} % drained at t {ts[k[0]]:.2f} s" if len(k) else f"[m13]   {int((1-frac)*100):3d} % drained: not within {ts[-1]:.1f} s", flush=True)
    # effective discharge coefficient over the window where 10-60 % has drained (past the start-up transient)
    sel = (ns <= 0.9 * ns[0]) & (ns >= 0.4 * ns[0])
    if sel.sum() > 5:
        Q = -np.gradient(ns * pv, ts)[sel]; Cd = Q / (A * np.sqrt(2 * g * h[sel]))
        print(f"[m13] effective C_d (10-60 % drained): median {np.median(Cd):.2f} (sharp-edged orifice ~0.61; ideal 1.0)", flush=True)
    print(f"[m13] wall {time.time()-wall0:.0f} s", flush=True)
    json.dump(dict(lid=args.lid, t=ts.tolist(), n_in_tank=ns.tolist(), pv=pv, A=A, At=At, Te_open=Te_open, Te_glug=Te_glug),
              open(os.path.join(args.out, "drain.json"), "w"))


if __name__ == "__main__":
    main()

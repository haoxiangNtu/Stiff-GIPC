"""M17: two-way coupling check — a light block dropped on still water (the same scene as run_ugref.py float in the
recovered Unified_GIPC engine, for a side-by-side comparison).

The m2 box (Fixed ABD) holds 0.6 L of water seeded over the whole cavity footprint (2 mm from the walls); a free
40 x 40 x 20 mm ABD block of density 500 kg/m^3 is released with its bottom 5 mm above the surface. Expected: it
bobs at f = sqrt(rho g A / m) / 2 pi = 5.0 Hz and settles with a 10 mm draft (half its thickness), level (the wide
footprint makes the upright orientation stable: GM = +8 mm). Reported: draft, tilt, bob frequency / damping,
particles inside the block.

    python examples/liquid/m17_float.py --dx 0.005 --ppc 8 --out out/liquid_m17
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from stiff_physics.engine import Engine, Config  # noqa: E402
from m2_box import cuboid, open_box  # noqa: E402
from m6_slosh import fit_damped  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=0.005)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--ppc", type=int, default=8)
    ap.add_argument("--h", type=float, default=0.0, help="particle spacing (m); overrides --ppc")
    ap.add_argument("--duration", type=float, default=2.0)
    ap.add_argument("--kappa", type=float, default=1e5)
    ap.add_argument("--newton-tol", type=float, default=1e-4)
    ap.add_argument("--pcg-tol", type=float, default=1e-6)
    ap.add_argument("--push", type=float, default=0.9)
    ap.add_argument("--jmode", type=int, default=3)
    ap.add_argument("--mass-floor", type=float, default=0.05)
    ap.add_argument("--dhat", type=float, default=5e-4)
    ap.add_argument("--rho-block", type=float, default=500.0)
    ap.add_argument("--no-block", action="store_true", help="water only (gentle still-water start in the m2 box; isolates the two-way coupling)")
    ap.add_argument("--lo-y", type=float, default=0.001, help="liquid grid origin y (m): sets where the node planes sit relative to the box floor at y = 0.020")
    ap.add_argument("--friction", type=float, default=0.0, help="IPC friction coefficient (particle proxies vs the box / block)")
    ap.add_argument("--apic", type=float, default=1.0, help="APIC ratio of the transfer (0 = PIC)")
    ap.add_argument("--dump-nodes", action="store_true", help="also dump the active grid-node positions (slots) in the frames (node field diagnostic)")
    ap.add_argument("--newton-cap", type=int, default=0, help="cap the Newton iterations per step (0 = engine default)")
    ap.add_argument("--jitter", type=float, default=0.3, help="seed jitter as a fraction of the spacing (uniform +-)")
    ap.add_argument("--kappa-mean-mass", type=float, default=0.0, help="pin the IPC barrier mass scale (kg); 0 = engine default (mesh mean mass)")
    ap.add_argument("--pcg-ratio", type=float, default=0.3, help="PCG max-iteration ratio (engine default 0.3)")
    ap.add_argument("--track", type=int, default=0, help="report the first N particles that enter the block (position before / after the step)")
    ap.add_argument("--dump-every", type=int, default=0, help="save frames/f_<step>.npz (x, v, block verts) every N steps (video)")
    ap.add_argument("--out", default="out/liquid_m17")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dx = args.dx; n_sub = max(1, int(round(args.ppc ** (1 / 3)))); hs = args.h if args.h > 0 else dx / n_sub
    cx = cz = 0.20; floor_y = 0.02; inner = 0.12; gap = 0.002
    V, F = open_box((cx, cz), inner, 0.005, floor_y, 0.16, 0.01)
    side = inner - 2 * gap; vol = 0.6e-3; depth = vol / side ** 2
    ax = np.arange(cx - side / 2 + hs / 2, cx + side / 2 - hs / 4, hs); ay = np.arange(floor_y + gap + hs / 2, floor_y + gap + depth, hs); az = ax - cx + cz
    gx, gy, gz = np.meshgrid(ax, ay, az, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)
    if args.jitter > 0:
        pts = pts + np.random.default_rng(0).uniform(-args.jitter * hs, args.jitter * hs, pts.shape)
    level0 = floor_y + gap + len(ay) * hs
    bw, bt, rho_b = 0.04, 0.02, args.rho_block
    yb = level0 + 0.005
    Vb, Fb = cuboid([cx - bw / 2, yb, cz - bw / 2], [cx + bw / 2, yb + bt, cz + bw / 2])

    cfg = Config(dt=args.dt, preconditioner_type=0, absolute_dhat=args.dhat, newton_tol=args.newton_tol, pcg_tol=args.pcg_tol,
                 friction_rate=args.friction, gd_friction_rate=0.0, ground_offset=0.0, energy_rel_tol=0.0, newton_iter_cap=(args.newton_cap if args.newton_cap > 0 else 200), pcg_max_iter_ratio=args.pcg_ratio, kappa_mean_mass=args.kappa_mean_mass,
                 gravity=(0.0, -9.81, 0.0))
    eng = Engine(cfg); eng.set_log_level(0)
    eng.load_mesh_from_data(V, F, 3, 3, "ABD", np.eye(4), 1e7, "Fixed")
    if args.no_block:
        # a far-away parked copy keeps the body bookkeeping identical; it never touches the water
        Vb, Fb = cuboid([0.35, 0.20, 0.35], [0.39, 0.22, 0.39])
    eng.load_mesh_from_data(Vb, Fb, 3, 3, "ABD", np.eye(4), 1e8, "Free")
    block = int(eng.get_load_records()[-1].body_offset)
    eng.set_abd_body_density(block, rho_b)
    lo = np.array([0.0, args.lo_y, 0.0]); hi = np.array([0.40, 0.25, 0.40])
    eng.add_liquid(pts, dx, lo, hi, rho=1000.0, kappa=args.kappa, particle_volume=hs ** 3, push_apart=args.push, j_mode=args.jmode, mass_floor=args.mass_floor, apic_ratio=args.apic)
    eng.finalize()
    m_b = rho_b * bw * bw * bt; f_bob = np.sqrt(1000 * 9.81 * bw * bw / m_b) / (2 * np.pi)
    print(f"[float] water {len(pts)} particles, seeded level {(level0-floor_y)*1e3:.1f} mm above the floor (volume {vol*1e3:.2f} L); block {bw*1e3:.0f}x{bw*1e3:.0f}x{bt*1e3:.0f} mm "
          f"rho {rho_b:.0f} -> mass {m_b*1e3:.1f} g, expected draft {bt*rho_b/1000*1e3:.1f} mm, bob {f_bob:.2f} Hz; dx {dx*1e3:.1f} mm h {hs*1e3:.2f} mm; {eng.get_liquid_stats()}", flush=True)

    # current block vertices from the ABD transform (get_abd_surface_body_vertices is the pre-finalize rest copy)
    def W(C):
        T = np.array(C, np.float64); T[:3, :3] = C[:3, :3].T; return T
    W0inv = np.linalg.inv(W(np.asarray(eng.get_abd_body_transforms(np.array([block], np.int32)))[0]))

    def block_verts():
        M = W(np.asarray(eng.get_abd_body_transforms(np.array([block], np.int32)))[0]) @ W0inv
        return Vb @ M[:3, :3].T + M[:3, 3]
    assert np.abs(block_verts() - Vb).max() < 1e-9, "ABD transform convention: block vertices do not reproduce the load positions at t=0"

    c0 = Vb.mean(0); half = np.array([bw / 2, bt / 2, bw / 2])

    def block_frame():
        M = W(np.asarray(eng.get_abd_body_transforms(np.array([block], np.int32)))[0]) @ W0inv
        return M

    def inside_block(x, M):
        # exact test in the block's own frame (an axis-aligned test on a tilted block flags particles under the raised side)
        loc = (x - M[:3, 3]) @ np.linalg.inv(M[:3, :3]).T - c0
        return np.all(np.abs(loc) < half, axis=1)

    def probe(t):
        x = eng.get_liquid_particles(); M = block_frame(); xb = Vb @ M[:3, :3].T + M[:3, 3]
        far = (np.abs(x[:, 0] - cx) > bw / 2 + 0.015) | (np.abs(x[:, 2] - cz) > bw / 2 + 0.015)
        level = float(np.percentile(x[far, 1], 95)) + hs / 2 if far.sum() > 50 else float("nan")
        c = M[:3, :3] @ c0 + M[:3, 3]; bottom = M[:3, :3] @ (c0 - [0, bt / 2, 0]) + M[:3, 3]   # centre of the bottom face
        up = M[:3, :3] @ np.array([0.0, 1.0, 0.0]); tilt = float(np.degrees(np.arccos(min(1.0, abs(up[1])))))
        under = int(inside_block(x, M).sum())
        return dict(t=t, bottom_mm=float(bottom[1] * 1e3), centre_y_mm=float(c[1] * 1e3), level_mm=float(level * 1e3), draft_mm=float((level - bottom[1]) * 1e3),
                    tilt_deg=tilt, xz_mm=[float((c[0] - cx) * 1e3), float((c[2] - cz) * 1e3)], vmax=float(np.linalg.norm(eng.get_liquid_velocities(), axis=1).max()), under=under)

    S = [probe(0.0)]; t = 0.0; step = 0; wall0 = time.time(); n0 = eng._engine.get_total_newton_iters()
    seen = np.zeros(len(pts), bool); reported = 0; x_prev = eng.get_liquid_particles().copy()
    while t < args.duration - 1e-12:
        if args.dump_every and step % args.dump_every == 0:
            os.makedirs(os.path.join(args.out, "frames"), exist_ok=True); Mf = block_frame()
            extra = {}
            if args.dump_nodes:
                st_ = eng.get_liquid_stats(); nb_ = int(st_["node_slot_base"]); na_ = int(st_["active_nodes"])
                extra["nodes"] = np.asarray(eng.get_vertices(), np.float64)[nb_:nb_ + na_]
            np.savez(os.path.join(args.out, "frames", f"f_{step:05d}.npz"), x=eng.get_liquid_particles().astype(np.float32), v=eng.get_liquid_velocities().astype(np.float32), t=t, block=(Vb @ Mf[:3, :3].T + Mf[:3, 3]).astype(np.float32),
                     J=np.asarray(eng.get_liquid_J(), np.float64), **extra)
        eng.step(); t += args.dt; step += 1
        s = probe(t); S.append(s)
        if args.track and reported < args.track:
            x = eng.get_liquid_particles(); M = block_frame(); inside = inside_block(x, M)
            new = np.flatnonzero(inside & ~seen)
            for i in new[: max(0, args.track - reported)]:
                loc = np.linalg.inv(M[:3, :3]) @ (x[i] - M[:3, 3]) - c0; locp = np.linalg.inv(M[:3, :3]) @ (x_prev[i] - M[:3, 3]) - c0
                print(f"[track] t {t:.3f} particle {i} entered: local before ({locp[0]*1e3:6.2f},{locp[1]*1e3:6.2f},{locp[2]*1e3:6.2f}) after ({loc[0]*1e3:6.2f},{loc[1]*1e3:6.2f},{loc[2]*1e3:6.2f}) mm"
                      f" (half extents {half[0]*1e3:.0f},{half[1]*1e3:.0f},{half[2]*1e3:.0f}; frame of the step end); |dx| {np.linalg.norm(x[i]-x_prev[i])*1e3:.2f} mm", flush=True)
                reported += 1
            seen |= inside
            x_prev = x.copy()
        if step % 10 == 0 or step < 3:
            print(f"t={t:6.3f}s " + " ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in s.items() if k != "t" and not isinstance(v, list)), flush=True)
    ts = np.array([s["t"] for s in S]); cy = np.array([s["centre_y_mm"] for s in S]); dr = np.array([s["draft_mm"] for s in S]); tl = np.array([s["tilt_deg"] for s in S])
    sel = ts >= ts[-1] - 0.5
    f, z, A, r2 = fit_damped(ts[ts >= 0.15], cy[ts >= 0.15])
    wall = time.time() - wall0; newton = (eng._engine.get_total_newton_iters() - n0) / max(1, step)
    print(f"[float] final 0.5 s: draft {dr[sel].mean():.2f} mm (expected {bt*rho_b/1000*1e3:.1f}), tilt {tl[sel].mean():.2f} deg, inside-block particles {S[-1]['under']}; "
          f"bob fit f {f:.2f} Hz (expected {f_bob:.2f}) damping {z:.3f} R^2 {r2:.2f}; min draft {dr.min():.1f} max {dr.max():.1f} mm; wall {wall:.0f} s, {newton:.1f} Newton/step", flush=True)
    json.dump(dict(draft_mm=float(dr[sel].mean()), draft_expected_mm=bt * rho_b, tilt_deg=float(tl[sel].mean()), f=f, zeta=z, r2=r2, f_bob=f_bob, dx=dx, h=hs, dt=args.dt,
                   wall=wall, newton_per_step=newton, t=ts.tolist(), centre_y_mm=cy.tolist(), draft=dr.tolist(), tilt=tl.tolist()), open(os.path.join(args.out, "summary.json"), "w"))
    json.dump(S, open(os.path.join(args.out, "samples.json"), "w"))
    np.savez(os.path.join(args.out, "final.npz"), x=eng.get_liquid_particles(), v=eng.get_liquid_velocities(), block=Vb @ block_frame()[:3, :3].T + block_frame()[:3, 3])


if __name__ == "__main__":
    main()

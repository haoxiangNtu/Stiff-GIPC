"""M4: the lab beaker (surface ABD mesh from lab_assets) filled with engine
water, tilted kinematically about its lip (teleport each step). Measures the
spill-onset angle, how much stays inside, the jet, and the solver cost.

    python examples/liquid/m4_beaker_tilt.py --fill 0.6 --out out/liquid_m4
"""
import argparse, os, sys, time
if __name__ == "__main__": import os as _os; _os.environ.setdefault("LIQ_UG", "0")   # main-path example, when run as a script: pinned to the main liquid path (the engine default is the UG mode since 2026-11-18)
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "lab_tactile"))
from stiff_physics.engine import Engine, Config  # noqa: E402
from lab_assets import build_all  # noqa: E402
from m2_box import nn_spacing  # noqa: E402


def cavity(v):
    cx, cz = v[:, 0].mean(), v[:, 2].mean()
    y0, y1 = v[:, 1].min(), v[:, 1].max()
    r = np.hypot(v[:, 0] - cx, v[:, 2] - cz)
    upper = v[:, 1] > y0 + 0.2 * (y1 - y0)
    r_in = r[upper].min()
    low = (r < 0.5 * r_in) & (v[:, 1] < y0 + 0.5 * (y1 - y0))
    floor_y = v[low, 1].max() if low.any() else y0 + 2e-3
    return np.array([cx, cz]), r_in, floor_y, y1


def ray_hit(tri, o, d):
    """nearest positive hit of the ray o + t d with the triangles tri (T,3,3) (Moller-Trumbore), nan if none"""
    e1 = tri[:, 1] - tri[:, 0]; e2 = tri[:, 2] - tri[:, 0]
    pv = np.cross(d, e2); det = np.einsum("ij,ij->i", e1, pv)
    ok = np.abs(det) > 1e-18; inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tv = o - tri[:, 0]; u = np.einsum("ij,ij->i", tv, pv) * inv
    qv = np.cross(tv, e1); w = (qv @ d) * inv; t = np.einsum("ij,ij->i", e2, qv) * inv
    m = ok & (u >= 0) & (w >= 0) & (u + w <= 1) & (t > 1e-12)
    return float(t[m].min()) if m.any() else float("nan")


def beaker_profile(v, faces, cxz, rim_y):
    """the beaker's real inner wall around its axis cxz (the load translation: the asset is axis-centred) by ray casting: inner floor
    (ray down the axis) and wall radius r(y) = r0 + s (y - floor) (72 horizontal rays per height, median, linear fit below the spout).
    The old estimate (vertex-mean centre, minimum upper vertex radius) is off by 2.7 mm in x (the spout pulls the mean) and reads
    23.0 mm for a wall at 23.8-25.1 mm (2 deg draft): the water started 0-5 mm from the wall, asymmetrically, and slumped outward."""
    tri = v[np.asarray(faces).reshape(-1, 3)]
    o = np.array([cxz[0], 0.5 * (v[:, 1].min() + rim_y), cxz[1]])
    floor_y = o[1] - ray_hit(tri, o, np.array([0.0, -1.0, 0.0]))
    A = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
    ys = np.linspace(floor_y + 0.004, floor_y + 0.75 * (rim_y - floor_y), 8)
    rs = np.array([np.nanmedian([ray_hit(tri, np.array([cxz[0], y, cxz[1]]), np.array([np.cos(a), 0.0, np.sin(a)])) for a in A]) for y in ys])
    s, r0 = np.polyfit(ys - floor_y, rs, 1)
    return floor_y, r0, s, float(np.abs(np.polyval([s, r0], ys - floor_y) - rs).max())


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=0.005)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--fill", type=float, default=0.6)
    ap.add_argument("--ppc", type=int, default=8)
    ap.add_argument("--kappa", type=float, default=1e5)
    ap.add_argument("--settle", type=float, default=1.0, help="seconds before tilting")
    ap.add_argument("--tilt-time", type=float, default=5.0)
    ap.add_argument("--tilt-max", type=float, default=100.0)
    ap.add_argument("--tilt-start", type=float, default=0.0, help="initial tilt (deg): start near the spill onset to reproduce the rim-crossing regime quickly")
    ap.add_argument("--hold", type=float, default=2.0)
    ap.add_argument("--newton-tol", type=float, default=1e-4)
    ap.add_argument("--pcg-tol", type=float, default=1e-6)
    ap.add_argument("--newton-cap", type=int, default=1000)
    ap.add_argument("--pcg-ratio", type=float, default=0.3)
    ap.add_argument("--energy-rel-tol", type=float, default=1e-9, help="line-search acceptance E1 <= E0 (1 + tol): avoids roundoff ties at convergence")
    ap.add_argument("--push", type=float, default=0.9)
    ap.add_argument("--jmode", type=int, default=3)
    ap.add_argument("--legacy-cavity", action="store_true", help="the old geometry estimate (vertex-mean centre, minimum upper vertex radius; before 10-26) for seeding and the inside test")
    ap.add_argument("--out", default="out/liquid_m4")
    ap.add_argument("--snap-every", type=float, default=0.5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    props = build_all(verbose=False)
    p = props["beaker_mix"]
    T0 = np.eye(4); T0[:3, 3] = [0.20, 0.0005 + 0.12, 0.20]   # lifted 12 cm so the pour falls onto the ground
    wv0 = p.verts @ T0[:3, :3].T + T0[:3, 3]
    c_xz, r_in, floor_y, rim_y = cavity(wv0)
    lip = np.array([c_xz[0], rim_y, c_xz[1] + r_in])   # tilt pivot: the legacy estimate's +z rim point (kept: the beaker's motion is unchanged)
    r_slope = 0.0
    if not args.legacy_cavity:   # the real beaker (ray cast) -- the old estimate sat 2.7 mm off axis with a 23.0 mm radius for a 23.7-25.8 mm cone
        c_xz = T0[[0, 2], 3].copy()
        floor_y, r_in, r_slope, _ = beaker_profile(wv0, p.faces, c_xz, rim_y)
    r_wall = lambda y: r_in + r_slope * (y - floor_y)
    dx = args.dx
    cfg = Config(dt=args.dt, preconditioner_type=0, absolute_dhat=5e-4, newton_tol=args.newton_tol, pcg_tol=args.pcg_tol,
                 friction_rate=0.0, gd_friction_rate=0.0, ground_offset=0.0, energy_rel_tol=args.energy_rel_tol,
                 newton_iter_cap=args.newton_cap, pcg_max_iter_ratio=args.pcg_ratio)
    eng = Engine(cfg); eng.set_log_level(0)
    eng.load_mesh_from_data(wv0, p.faces, verts_per_face=3, dimensions=3, body_type="ABD", transform=np.eye(4),
                            young_modulus=1e8, boundary_type="Free")
    rec = eng.get_load_records()[-1]
    body = int(rec.body_offset)
    eng.set_abd_body_density(body, 1.0e7)   # kinematic: so heavy that the water cannot move it within a step (teleport resets it anyway)
    # water cylinder
    h = floor_y + args.fill * (rim_y - floor_y)
    n_sub = max(1, int(round(args.ppc ** (1 / 3)))); hs = dx / n_sub; m = 0.6 * dx
    rmax = r_wall(h)
    ax = np.arange(c_xz[0] - rmax + m, c_xz[0] + rmax - m, hs); ay = np.arange(floor_y + m, h, hs); az = np.arange(c_xz[1] - rmax + m, c_xz[1] + rmax - m, hs)
    gx, gy, gz = np.meshgrid(ax, ay, az, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)
    pts = pts + np.random.default_rng(0).uniform(-0.3 * hs, 0.3 * hs, pts.shape)
    pts = pts[np.hypot(pts[:, 0] - c_xz[0], pts[:, 2] - c_xz[1]) < r_wall(pts[:, 1]) - m]
    lo = np.array([0.0, -0.02, -0.10]); hi = np.array([0.40, 0.40, 0.45])
    if args.tilt_start != 0.0:
        # the beaker is teleported to tilt_start at the first step; rotate the seed the same way
        # (about the lip point, see transform_at) so the water starts inside the tilted cavity
        Rs = rot_x(np.radians(args.tilt_start)); lip0 = np.array([c_xz[0], rim_y, c_xz[1] + r_in])
        pts = (pts - lip0) @ Rs.T + lip0
    eng.add_liquid(pts, dx, lo, hi, rho=1000.0, kappa=args.kappa, particle_volume=hs ** 3, push_apart=args.push, j_mode=args.jmode,
                   node_pool=40000)   # the pour spreads over the ground: ~12k active nodes at 65 % on the floor, more at 100 deg
    eng.finalize()
    eng._engine.set_body_apply_gravity(body, False)   # kinematic: no in-step free fall between teleports
    print(f"[m4] beaker r_in {r_in*1e3:.1f} mm floor {floor_y*1e3:.1f} rim {rim_y*1e3:.1f} mm, fill to {h*1e3:.1f} mm -> {len(pts)} particles; {eng.get_liquid_stats()}", flush=True)

    # tilt about the lip point (the rim point on the +z side), axis = +x
    v0, n0 = int(rec.vertex_offset), int(rec.vertex_count)
    # engine convention: get_abd_body_transforms stores the affine block transposed and
    # maps the body's reference coordinates (not the loaded world positions) to world.
    C0 = np.asarray(eng.get_abd_body_transforms(np.array([body], np.int32)))[0]
    W0 = np.array(C0, np.float64); W0[:3, :3] = C0[:3, :3].T          # reference -> world at load

    def transform_at(ang_deg):
        R = rot_x(np.radians(ang_deg))
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = lip - R @ lip           # world -> world rotation about the lip
        W = T @ W0
        C = np.array(W); C[:3, :3] = W[:3, :3].T                          # back to the engine convention
        return T, C

    t, step = 0.0, 0
    nt_prev = 0; pc_prev = 0; t_snap = 0.0; onset = None; wall0 = time.time(); rows = []
    total = args.settle + args.tilt_time + args.hold
    while t < total:
        ang = args.tilt_start if t < args.settle else min(args.tilt_max, args.tilt_start + (args.tilt_max - args.tilt_start) * (t - args.settle) / args.tilt_time)
        eng.teleport_abd_bodies(np.array([body], np.int32), transform_at(ang)[1][None])   # kinematic beaker (also during settle)
        ts = time.time(); eng.step(); wall = time.time() - ts
        t += args.dt; step += 1
        if step % 20 == 0:
            x = eng.get_liquid_particles(); bv = eng.get_vertices()[v0:v0 + n0]
            # measured tilt from the beaker vertices (compare with commanded ang)
            T = transform_at(ang)[0]; R = T[:3, :3]
            pl = (x - T[:3, 3]) @ R    # back to the untilted frame
            inside = (np.hypot(pl[:, 0] - c_xz[0], pl[:, 2] - c_xz[1]) < r_wall(pl[:, 1]) + 0.002) & (pl[:, 1] > floor_y - 0.005) & (pl[:, 1] < rim_y + 0.002)
            pose_err = np.abs(bv - (wv0 @ R.T + T[:3, 3])).max()   # teleport convention check (should be ~0)
            nt = eng._engine.get_total_newton_iters(); pc = eng._engine.get_total_pcg_iters()
            fin = inside.mean(); ground = (x[:, 1] < 0.006).mean()
            if onset is None and t > args.settle and fin < 0.98: onset = ang
            nnm, nn1 = nn_spacing(x)
            print(f"t={t:5.2f}s cmd {ang:6.1f} deg (pose err {pose_err*1e3:.2f} mm)  inside {fin*100:5.1f}%  ground {ground*100:5.1f}%  NN {nnm*1e3:.2f}/{nn1*1e3:.2f} mm  |v|max {np.linalg.norm(eng.get_liquid_velocities(),axis=1).max():.2f}"
                  f"  newton {(nt-nt_prev)/20:5.1f}/step pcg {(pc-pc_prev)/20:7.0f}/step {wall*1000:6.0f} ms", flush=True)
            nt_prev = nt; pc_prev = pc
            rows.append([t, ang, fin, ground])
        if t >= t_snap:
            np.save(os.path.join(args.out, f"pts_{int(round(t*100)):05d}.npy"), eng.get_liquid_particles()); t_snap += args.snap_every
    np.save(os.path.join(args.out, "rows.npy"), np.array(rows))
    print(f"[m4] onset (first <98% inside): {onset if onset is not None else 'none'} deg; final inside {rows[-1][2]*100:.1f}% ground {rows[-1][3]*100:.1f}%; wall {time.time()-wall0:.0f} s", flush=True)


if __name__ == "__main__":
    main()

"""Nuka-Physics 'Bunny x Water' with StiffGIPC: a rigid (ABD) Stanford bunny falls into a glass tank of engine MPM water.

Reference setup (Nuka bunny_water_demo): 120 mm bunny, density 1200 kg/m^3 (0.41 kg), dropped 150 mm into a 360 x 280 mm
tank holding 180 mm of water; weakly compressible MLS-MPM water, 670 k particles; 1.5 s recorded, shown at 2x slow motion.
Here: same tank and drop, ABD bunny two-way coupled with the liquid inside the IPC Newton; particle budget set by --liquid-dx.

    PYTHONPATH=. python examples/nuka/bunny_water.py --out out/nuka/bw1 --liquid-dx 0.016
"""
import argparse, os, os.path as osp, sys, time
import numpy as np

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "..", "robotgym"))
from common import ASSETS, Body, Recorder, Scene, abd_world_tf  # noqa: E402

# ------------------------------------------------------------------ geometry (metres, y up, ground y = 0)
TANK_IN = np.array([0.36, 0.30, 0.28])     # inner x, wall height, inner z (Nuka: 360 x 280 mm, rim 340 mm)
WALL, FLOOR_T = 0.010, 0.012
TANK_C = np.array([0.0, 0.0, 0.0])          # floor level at y = 0
WATER_H = 0.18


def load_obj(path):
    V, F = [], []
    for line in open(path):
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            F.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    return np.asarray(V, float), np.asarray(F, np.int32)


def add_tank(sc, name, centre, inner, color, mu=0.0):
    """Open-top glass tank: floor + 4 walls, fixed tessellated boxes sunk 4 mm into each other and mutually excluded
    (coincident fixed faces give a NaN barrier; a gap lets particles through)."""
    cx, cy, cz = centre; ix, h, iz = inner; emb = 0.004
    parts = [sc.add_box(f"{name}_floor", (cx, cy + FLOOR_T / 2, cz), (ix + 2 * WALL, FLOOR_T, iz + 2 * WALL), color=color, cell=0.04, mu=mu)]
    yc = cy + FLOOR_T - emb + (h + emb) / 2
    parts.append(sc.add_box(f"{name}_wall_xp", (cx + ix / 2 + WALL / 2, yc, cz), (WALL, h + emb, iz + 2 * WALL), color=color, cell=0.04, mu=mu))
    parts.append(sc.add_box(f"{name}_wall_xm", (cx - ix / 2 - WALL / 2, yc, cz), (WALL, h + emb, iz + 2 * WALL), color=color, cell=0.04, mu=mu))
    parts.append(sc.add_box(f"{name}_wall_zp", (cx, yc, cz + iz / 2 + WALL / 2), (ix, h + emb, WALL), color=color, cell=0.04, mu=mu))
    parts.append(sc.add_box(f"{name}_wall_zm", (cx, yc, cz - iz / 2 - WALL / 2), (ix, h + emb, WALL), color=color, cell=0.04, mu=mu))
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            sc.eng.add_collision_exclusion(parts[i].body_id, parts[j].body_id)
    return parts


def add_rigid_mesh(sc, name, verts, faces, density, mu, color):
    """Free ABD body from a closed triangle mesh (same loading path as Scene.add_box, dynamic branch)."""
    sc.eng.load_mesh_from_data(np.asarray(verts, float), np.asarray(faces, np.int32), verts_per_face=3, dimensions=3,
                               body_type="ABD", transform=np.eye(4), young_modulus=1e8, boundary_type="Free")
    rec = sc.eng.get_load_records()[-1]
    b = Body(rec, name, "prop", color=color)
    sc.eng.set_abd_body_density(b.body_id, float(density))
    sc.eng.set_body_friction(b.body_id, mu)
    sc.bodies[name] = b
    return b


def seed_water(dx, ppc, rng, depth):
    n_sub = max(1, int(round(ppc ** (1.0 / 3.0)))); hs = dx / n_sub; m = 0.6 * dx
    lo = TANK_C + np.array([-TANK_IN[0] / 2 + m, FLOOR_T + m, -TANK_IN[2] / 2 + m])
    hi = TANK_C + np.array([TANK_IN[0] / 2 - m, FLOOR_T + depth, TANK_IN[2] / 2 - m])
    ax = [np.arange(lo[k], hi[k], hs) for k in range(3)]
    g = np.meshgrid(*ax, indexing="ij")
    pts = np.stack([a.ravel() for a in g], 1) + rng.uniform(-0.3 * hs, 0.3 * hs, (g[0].size, 3))
    return pts, hs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dt", type=float, default=0.0025)
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--duration", type=float, default=2.0)
    ap.add_argument("--max-time", type=float, default=0.0, help="stop after this many simulated seconds (0 = full duration)")
    ap.add_argument("--newton-cap", type=int, default=30)
    ap.add_argument("--liquid-dx", type=float, default=0.016, help="grid spacing; particles at dx/2 (8 per cell)")
    ap.add_argument("--liquid-ppc", type=int, default=8)
    ap.add_argument("--node-pool", type=int, default=200000)
    ap.add_argument("--water-h", type=float, default=WATER_H)
    ap.add_argument("--bunny-extent", type=float, default=0.14, help="longest AABB extent of the bunny (Nuka: 0.12)")
    ap.add_argument("--bunny-density", type=float, default=1200.0, help="1200 sinks (Nuka); 500 floats (buoyancy check)")
    ap.add_argument("--drop-height", type=float, default=0.15, help="bunny bottom above the water surface at t = 0")
    ap.add_argument("--bunny-yaw", type=float, default=35.0, help="yaw about y (deg) so the splash is not axis-aligned")
    ap.add_argument("--soft", action="store_true", help="FEM (tet) bunny instead of the rigid ABD one (Assets/nuka/bunny_coarse_tets.npz)")
    ap.add_argument("--soft-young", type=float, default=5e4)
    ap.add_argument("--soft-poisson", type=float, default=0.35)
    ap.add_argument("--mu", type=float, default=0.0, help="bunny<->everything friction (water particles included; 0 = wet)")
    ap.add_argument("--tank-mu", type=float, default=0.0, help="tank<->water friction: a frictional floor jams the bottom water layer")
    ap.add_argument("--push-apart", type=float, default=0.9, help="liquid particle regularisation strength")
    ap.add_argument("--kappa", type=float, default=1e5, help="liquid bulk modulus (Pa)")
    ap.add_argument("--newton-tol", type=float, default=1e-3)
    ap.add_argument("--pcg-tol", type=float, default=1e-5)
    ap.add_argument("--dhat", type=float, default=1e-3, help="absolute IPC barrier distance (m)")
    ap.add_argument("--save-every", type=float, default=0.25, help="write a partial recording every N simulated seconds")
    ap.add_argument("--log-level", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sc = Scene(dict(dt=args.dt, preconditioner_type=0, newton_iter_cap=args.newton_cap, newton_tol=args.newton_tol, pcg_tol=args.pcg_tol,
                    collision_detection_buff_scale=12.0, absolute_dhat=args.dhat, poisson_rate=args.soft_poisson), log_level=args.log_level)
    add_tank(sc, "tank_glass", TANK_C, TANK_IN, color=(0.85, 0.90, 0.95), mu=args.tank_mu)
    # bunny: centre the AABB, scale the longest extent, keep y up (the surface mesh stands on its y-min plane), yaw, lift
    if args.soft:
        d = np.load(osp.join(ASSETS, "nuka", "bunny_coarse_tets.npz")); V, TT = d["verts"].astype(float), d["tets"].astype(np.int32); F = None
    else:
        V, F = load_obj(osp.join(ASSETS, "triMesh", "bunny_surface.obj")); TT = None
    V = V - 0.5 * (V.min(0) + V.max(0)); V *= args.bunny_extent / (V.max(0) - V.min(0)).max()
    a = np.deg2rad(args.bunny_yaw); Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    V = V @ Ry.T
    y_water = TANK_C[1] + FLOOR_T + args.water_h
    V[:, 1] += y_water + args.drop_height - V[:, 1].min()
    V[:, 0] += TANK_C[0]; V[:, 2] += TANK_C[2]
    if args.soft:
        a_, b_, c_, d_ = (V[TT[:, i]] for i in range(4)); vol = np.abs(np.einsum("ij,ij->i", np.cross(b_ - a_, c_ - a_), d_ - a_)).sum() / 6.0
        bunny = sc.add_fem("bunny", V, TT, young=args.soft_young, density=args.bunny_density, color=(0.93, 0.93, 0.90))
        sc.eng.set_body_friction(len(sc.eng.get_load_records()) - 1, args.mu)
        print(f"[scene] SOFT bunny: {len(V)} verts, {len(TT)} tets, E {args.soft_young:.0f} Pa, nu {args.soft_poisson}, extent {np.round((V.max(0)-V.min(0))*1e3,1)} mm, "
              f"volume {vol*1e6:.0f} ml, mass {vol*args.bunny_density*1e3:.0f} g, bottom {args.drop_height*1e3:.0f} mm above the water")
    else:
        vol = np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6.0
        bunny = add_rigid_mesh(sc, "bunny", V, F, args.bunny_density, args.mu, color=(0.93, 0.93, 0.90))
        print(f"[scene] bunny: {len(V)} verts, {len(F)} faces, extent {np.round((V.max(0)-V.min(0))*1e3,1)} mm, volume {vol*1e6:.0f} ml, "
              f"mass {vol*args.bunny_density*1e3:.0f} g, bottom {args.drop_height*1e3:.0f} mm above the water")
    pts, hs = seed_water(args.liquid_dx, args.liquid_ppc, np.random.default_rng(0), args.water_h)
    lo = TANK_C - np.array([TANK_IN[0] / 2 + 0.06, 0.02, TANK_IN[2] / 2 + 0.06])
    hi = TANK_C + np.array([TANK_IN[0] / 2 + 0.06, TANK_IN[1] + 0.35, TANK_IN[2] / 2 + 0.06])
    sc.eng.add_liquid(pts, args.liquid_dx, lo, hi, rho=1000.0, kappa=args.kappa, particle_volume=hs ** 3,
                      node_pool=args.node_pool, push_apart=args.push_apart, j_mode=0, stick_h=0.0)
    n_liq = len(pts)
    print(f"[scene] water: {n_liq} particles ({n_liq*hs**3*1e6:.0f} ml), spacing {hs*1e3:.1f} mm, depth {args.water_h*1e3:.0f} mm, "
          f"tank {TANK_IN[0]*1e3:.0f} x {TANK_IN[2]*1e3:.0f} mm", flush=True)
    sc.finalize()
    rec = Recorder(sc)
    rec_every = max(1, int(round(1.0 / (args.fps * args.dt))))
    n_steps = int(round(args.duration / args.dt))
    if args.max_time > 0:
        n_steps = min(n_steps, int(round(args.max_time / args.dt)))
    save_every = max(1, int(round(args.save_every / args.dt))) if args.save_every > 0 else 0
    meta = dict(example="bunny_water", dt=args.dt, fps=args.fps, n_liquid=n_liq, liquid_dx=args.liquid_dx, particle_spacing=hs,
                water_h=args.water_h, tank_in=TANK_IN.tolist(), bunny_extent=args.bunny_extent, bunny_density=args.bunny_density,
                bunny_volume_ml=vol * 1e6, drop_height=args.drop_height, mu=args.mu, tank_mu=args.tank_mu, soft=bool(args.soft),
                soft_young=args.soft_young, fem_colors=[[0.93, 0.93, 0.90]] if args.soft else [],
                push_apart=args.push_apart, kappa=args.kappa, newton_tol=args.newton_tol, pcg_tol=args.pcg_tol, dhat=args.dhat,
                fem_materials=({"bunny": "soft"} if args.soft else {}))
    rec_path = osp.join(args.out, "recording.npz")
    ids = None if args.soft else np.array([bunny.body_id], np.int32)
    t_hit = np.sqrt(2 * args.drop_height / 9.81)
    rim_y = TANK_C[1] + FLOOR_T + TANK_IN[1]
    t0 = time.time(); t_last = t0; y_prev = None
    for k in range(n_steps + 1):
        t = k * args.dt
        seg = "free fall" if t < t_hit else "splash and settle"
        if k % max(1, int(round(0.1 / args.dt))) == 0:
            y = float(sc.fem_verts("bunny")[:, 1].mean()) if args.soft else float(abd_world_tf(sc.eng.get_abd_body_transforms(ids)[0])[1, 3])
            vy = 0.0 if y_prev is None else (y - y_prev) / 0.1; y_prev = y
            P = sc.eng.get_liquid_particles()
            inside = (np.abs(P[:, 0] - TANK_C[0]) < TANK_IN[0] / 2 + WALL) & (np.abs(P[:, 2] - TANK_C[2]) < TANK_IN[2] / 2 + WALL)
            escaped = int(np.sum(~inside | (P[:, 1] < TANK_C[1] + FLOOR_T - 0.01)))
            now = time.time()
            print(f"[{t:5.2f}s] {seg:18s} bunny y {y*1e3:7.1f} mm vy {vy:+6.2f} m/s | water top {P[:,1].max()*1e3:6.1f} mm, above rim {int(np.sum(P[:,1] > rim_y))}, "
                  f"escaped {escaped} | {(now-t0)/60:5.1f} min, {(now-t_last)/max(1,int(round(0.1/args.dt))):.2f} s/step", flush=True)
            t_last = now
        if k % rec_every == 0:
            rec.record(t, 0, seg, liquid_pts=sc.eng.get_liquid_particles())
        if save_every and k and k % save_every == 0:
            rec.save(rec_path, dict(meta, partial=True, wall_min=(time.time() - t0) / 60))
        sc.step()
    meta["wall_min"] = (time.time() - t0) / 60
    rec.save(rec_path, meta)
    print(f"[done] {n_steps} steps in {meta['wall_min']:.1f} min -> {rec_path}")


if __name__ == "__main__":
    main()

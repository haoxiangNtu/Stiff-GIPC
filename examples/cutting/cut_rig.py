#!/usr/bin/env python3
"""Cutting bring-up rig (model-1 runs set LIQ_EL_LIN=1: linear particle kinematics, see liquid_system.cu k_g2p): a 4 mm rigid blade driven straight down (prismatic joint) through an MPM cylinder lying on a
fixed board, to see which model-1 material the blade can cut (elastic / tension cut-off / von Mises).

The cylinder has the cucumber's cross-section (50 mm) and lies along z; the blade is thin in z and cuts at z = 0.
At the end the particles are split into connected groups (neighbours within 1.5 seed spacings): a cut cylinder is two
groups, one each side of the blade, with no particle left in the slit.

    PYTHONPATH=. STIFFGIPC_NATIVE_DIR=build_4090 python examples/cutting/cut_rig.py --out out/cutting/rig_elastic
    ... --stretch 0.02 --compress 0.1                     # tension cut-off (fracture stand-in)
    ... --yield 0.03 --stretch 0.05                      # von Mises + cut-off
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
import time

import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(HERE, "..", "robotgym"))
from common import Body, Recorder, Scene, minjerk  # noqa: E402

BOARD_TOP = 0.02


def bevel_blade(centre, length, height, thick, bevel, edge_t, cell):
    """closed prism along x: a blade of `thick` (z) and `height` (y) whose bottom `bevel` tapers to an edge `edge_t` thick
    (the knife model's blade: 4 mm, a ~5 mm bevel to a rounded ~1 mm edge); faces ~cell, normals outward"""
    hy, ht, he = 0.5 * height, 0.5 * thick, 0.5 * edge_t
    poly = np.array([(-ht, hy), (ht, hy), (ht, -hy + bevel), (he, -hy), (-he, -hy), (-ht, -hy + bevel)])   # (z, y)
    ring = []
    for k in range(len(poly)):
        a, b = poly[k], poly[(k + 1) % len(poly)]
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / cell)))
        ring += [a + (b - a) * i / n for i in range(n)]
    ring = np.array(ring); m = len(ring)
    nx = max(1, int(np.ceil(length / cell)))
    xs = np.linspace(-0.5 * length, 0.5 * length, nx + 1)
    V = [np.array([x, r[1], r[0]]) for x in xs for r in ring]
    F = []
    for i in range(nx):
        for k in range(m):
            a, b, c, d = i * m + k, i * m + (k + 1) % m, (i + 1) * m + (k + 1) % m, (i + 1) * m + k
            F += [(a, b, c), (a, c, d)]
    for end, base in ((0, 0), (1, nx * m)):          # end caps: fan around an added centre vertex
        ci = len(V); V.append(np.array([xs[0] if end == 0 else xs[-1], 0.0, 0.0]))
        F += [(ci, base + k, base + (k + 1) % m) for k in range(m)]
    V = np.array(V) + np.asarray(centre, float); F = np.array(F, np.int64)
    c0 = V.mean(0)                                   # convex: orient every face away from the centroid
    nrm = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    flip = np.einsum("ij,ij->i", nrm, V[F].mean(1) - c0) < 0
    F[flip] = F[flip][:, ::-1]
    return V, F.astype(np.int32)


def groups(P, h):
    """connected groups of particles (neighbours within h): sizes, largest first, and the label of every particle"""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    pr = cKDTree(P).query_pairs(h, output_type="ndarray")
    n, lab = connected_components(coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(P), len(P))), directed=False)
    return np.sort(np.bincount(lab))[::-1], lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dt", type=float, default=0.0025)
    ap.add_argument("--spacing", type=float, default=0.0015, help="particle seed spacing (m); grid dx = 2 x spacing")
    ap.add_argument("--radius", type=float, default=0.025)
    ap.add_argument("--length", type=float, default=0.060)
    ap.add_argument("--young", type=float, default=2e6)
    ap.add_argument("--skin", type=float, default=0.0015, help="skin depth (m) for the zone factors below")
    ap.add_argument("--skin-young-x", type=float, default=1.0, help="skin: Young's modulus = this x --young")
    ap.add_argument("--skin-damage-x", type=float, default=1.0, help="skin: damage stretch threshold = this x --damage")
    ap.add_argument("--core-young-x", type=float, default=1.0, help="core: Young's modulus = this x --young")
    ap.add_argument("--core-damage-x", type=float, default=1.0, help="core: damage stretch threshold = this x --damage")
    ap.add_argument("--core-frac", type=float, default=0.55, help="core radius / cylinder radius")
    ap.add_argument("--core-blend", type=float, default=0.12, help="half-width of the core / flesh transition (fraction of the radius)")
    ap.add_argument("--material-api", action="store_true", help="call set_liquid_particle_material even with all factors 1 (regression of the code path)")
    ap.add_argument("--poisson", type=float, default=0.4)
    ap.add_argument("--rho", type=float, default=960.0)
    ap.add_argument("--stretch", type=float, default=0.0, help="plastic_stretch: principal stretch cap 1 + s (0 = off)")
    ap.add_argument("--compress", type=float, default=0.0, help="plastic_compress: principal stretch floor 1 - c (0 = off)")
    ap.add_argument("--yield", dest="yield_", type=float, default=0.0, help="plastic_yield: von Mises bound on |dev ln S| (0 = off)")
    ap.add_argument("--damage", type=float, default=0.0, help="plastic_damage: a particle breaks for good above principal stretch 1 + d (0 = off)")
    ap.add_argument("--dmg-res", type=float, default=0.1, help="plastic_damage_residual: share of the elastic energy a broken particle keeps")
    ap.add_argument("--dmg-eps", type=float, default=0.0, help="plastic_damage_strain: ductile damage above this accumulated plastic strain (0 = off)")
    ap.add_argument("--cutter-gate", action="store_true", help="set_liquid_cutter(blade): only particles the blade has touched may break")
    ap.add_argument("--particle-mu", type=float, default=0.0, help="add_liquid mu: particle friction (pair = sqrt(mu x body mu)); 0 = frictionless")
    ap.add_argument("--blade-mu", type=float, default=None, help="blade body friction (default --mu)")
    ap.add_argument("--edge-gap", type=float, default=-0.0005, help="blade bottom height above the board at full travel (m); < 0 drives it into the board's barrier band")
    ap.add_argument("--no-blade-board", action="store_true", help="no blade-board contact: the edge can reach the board top, only the particles resist")
    ap.add_argument("--blade-t", type=float, default=0.004, help="blade thickness (m), the knife model's 4 mm spine")
    ap.add_argument("--bevel", type=float, default=0.0, help="ground-edge height (m): the blade tapers to --edge-t over its bottom (0 = flat bottom); the knife model: 0.005 to 0.001")
    ap.add_argument("--edge-t", type=float, default=0.0002, help="edge thickness of a bevelled blade (m)")
    ap.add_argument("--speed", type=float, default=0.05, help="blade speed (m/s)")
    ap.add_argument("--settle", type=float, default=0.3, help="s before the blade moves")
    ap.add_argument("--hold", type=float, default=0.3)
    ap.add_argument("--lift", type=float, default=0.6, help="s to lift the blade back out (0 = stay down)")
    ap.add_argument("--mu", type=float, default=0.3)
    ap.add_argument("--dhat", type=float, default=5e-4)
    ap.add_argument("--newton-cap", type=int, default=60)
    ap.add_argument("--newton-tol", type=float, default=1e-3)
    ap.add_argument("--pcg-tol", type=float, default=1e-5)
    ap.add_argument("--max-time", type=float, default=0.0, help="stop after this many simulated seconds (0 = whole timeline)")
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--log-level", type=int, default=0)
    ap.add_argument("--cpic", action="store_true", help="cutter CPIC (set_liquid_cut_plane): the blade's mid-plane, from the edge up plus --cpic-lead below it, is a crack surface the grid does not couple across")
    ap.add_argument("--cpic-lead", type=float, default=0.001, help="crack front ahead of the edge (m)")
    ap.add_argument("--cpic-band", type=float, default=0.0, help="mask band half-width around the mid-plane (m); 0 = dx")
    ap.add_argument("--grid-shift", type=float, default=0.0, help="shift the MPM grid origin along z (m): with 0 the blade plane z = 0 lies on a node plane")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sc = Scene(dict(dt=args.dt, preconditioner_type=0, newton_iter_cap=args.newton_cap, newton_tol=args.newton_tol, pcg_tol=args.pcg_tol,
                    absolute_dhat=args.dhat, friction_rate=args.mu, gd_friction_rate=args.mu, gravity=(0.0, -9.81, 0.0)),
               log_level=args.log_level)
    sc.add_box("board", (0.0, BOARD_TOP - 0.01, 0.0), (0.20, 0.02, 0.16), fixed=True, color=(0.72, 0.55, 0.36), mu=args.mu, cell=0.02)
    top = BOARD_TOP + 2 * args.radius
    y0 = top + 0.003 + 0.5 * 0.06                       # blade centre: its bottom edge 3 mm above the cylinder
    if args.bevel > 0:
        Vb, Fb = bevel_blade((0.0, y0, 0.0), 0.10, 0.06, args.blade_t, args.bevel, args.edge_t, 0.005)
        sc.eng.load_mesh_from_data(Vb, Fb, verts_per_face=3, dimensions=3, body_type="ABD", transform=np.eye(4), young_modulus=1e8,
                                   boundary_type="Free")
        blade = Body(sc.eng.get_load_records()[-1], "blade", "prop", color=(0.8, 0.82, 0.85))
        sc.eng.set_abd_body_density(blade.body_id, 7800.0)
        sc.eng.set_body_friction(blade.body_id, args.mu if args.blade_mu is None else args.blade_mu)
        sc.bodies["blade"] = blade
    else:
        blade = sc.add_box("blade", (0.0, y0, 0.0), (0.10, 0.06, args.blade_t), fixed=False, color=(0.8, 0.82, 0.85),
                           density=7800.0, mu=args.mu if args.blade_mu is None else args.blade_mu, cell=0.01)
    anchor = sc.add_box("anchor", (0.0, y0 + 0.12, 0.0), (0.04, 0.02, 0.04), fixed=True, color=(0.3, 0.3, 0.3), cell=0.02)
    sc.eng.add_collision_exclusion(anchor.body_id, blade.body_id)
    board = sc.bodies["board"]
    if args.no_blade_board:
        sc.eng.add_collision_exclusion(board.body_id, blade.body_id)
    travel = (y0 - 0.03) - (BOARD_TOP + args.edge_gap)  # bottom edge down to edge_gap above the board
    jid = sc.eng.add_prismatic_joint(anchor.body_id, blade.body_id, (0.0, y0, 0.0), (0.0, -1.0, 0.0), -0.01, travel + 0.01, "blade")

    # cylinder along z, resting on the board (1 mm drop)
    s = args.spacing
    g = np.arange(-args.radius + 0.5 * s, args.radius, s)
    gz = np.arange(-0.5 * args.length + 0.5 * s, 0.5 * args.length, s)
    X, Y, Z = np.meshgrid(g, g, gz, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    pts = pts[np.hypot(pts[:, 0], pts[:, 1]) < args.radius]
    pts += np.random.default_rng(0).uniform(-0.2 * s, 0.2 * s, pts.shape)
    r_ax = np.hypot(pts[:, 0], pts[:, 1])                      # distance from the cylinder axis (before it is lifted onto the board)
    rfrac = np.clip(r_ax / args.radius, 0.0, 1.0)
    depth = args.radius - r_ax
    lo_c, hi_c = args.core_frac - args.core_blend, args.core_frac + args.core_blend
    wz = np.clip((hi_c - rfrac) / max(hi_c - lo_c, 1e-9), 0.0, 1.0); wz = wz * wz * (3.0 - 2.0 * wz)
    young_x = np.exp(wz * np.log(args.core_young_x)); damage_x = np.exp(wz * np.log(args.core_damage_x))
    zone = np.where(rfrac < args.core_frac, 2, 0).astype(np.int8)
    sk = depth < args.skin
    young_x[sk], damage_x[sk] = args.skin_young_x, args.skin_damage_x
    zone[sk] = 1
    zoned = bool(np.any(young_x != 1.0) or np.any(damage_x != 1.0))
    pts[:, 1] += BOARD_TOP + args.radius + 0.001
    dx = 2.0 * s
    lo = np.array([-0.12, 0.0, -0.10 + args.grid_shift]); hi = np.array([0.12, top + 0.08, 0.10])
    sc.eng.add_liquid(pts, dx, lo, hi, rho=args.rho, kappa=1e5, particle_volume=s ** 3, node_pool=max(200000, 4 * len(pts)), push_apart=0.0, j_mode=0,
                      ground_mu=args.mu, model=1, young=args.young, poisson=args.poisson,
                      plastic_stretch=args.stretch, plastic_compress=args.compress, plastic_yield=args.yield_,
                      plastic_damage=args.damage, plastic_damage_residual=args.dmg_res, mu=args.particle_mu,
                      plastic_damage_strain=args.dmg_eps)
    if args.cutter_gate:
        sc.eng.set_liquid_cutter(blade.body_id)
    sc.finalize()
    if zoned or args.material_api:
        sc.eng.set_liquid_particle_material(young_x, damage_x)
    print(f"[rig] zones: skin {int((zone == 1).sum())} (E x {args.skin_young_x:g}, damage x {args.skin_damage_x:g}), core {int((zone == 2).sum())} "
          f"(r < {args.core_frac:g} R, E x {args.core_young_x:g}, damage x {args.core_damage_x:g}), flesh {int((zone == 0).sum())}; zoned {zoned}", flush=True)
    near = (np.abs(pts[:, 2]) < 0.006) & (np.abs(pts[:, 0]) < 0.006)       # column under the blade: its top = indentation
    mid = (np.abs(pts[:, 1] - (BOARD_TOP + args.radius + 0.001)) < 0.003) & (np.abs(pts[:, 2]) > 0.008) & (np.abs(pts[:, 2]) < 0.02)   # mid-height band beside the cut: width
    first_break = None
    # the strip of skin right under the edge (its mean height = the dent) and the per-record series (1 mm of travel at 50 mm/s):
    # the first break used to be looked for only at the log lines, i.e. every 5 mm of travel
    under = (np.abs(pts[:, 2]) < 0.0015) & (np.abs(pts[:, 0]) < 0.006) & (pts[:, 1] > pts[:, 1].max() - 0.002)
    series = []
    sc.eng.native.set_prismatic_strength(jid, 1.0)
    rec = Recorder(sc)
    T_move = travel / args.speed
    T = args.settle + T_move + args.hold + args.lift
    T = min(T, args.max_time) if args.max_time > 0 else T
    n_steps = int(round(T / args.dt))
    rec_every = max(1, int(round(1.0 / (args.fps * args.dt))))
    print(f"[rig] {len(pts)} particles, spacing {s*1e3:.2f} mm, dx {dx*1e3:.1f} mm, E {args.young:.3g} nu {args.poisson}, "
          f"stretch {args.stretch} compress {args.compress} yield {args.yield_} damage {args.damage} / plastic strain {args.dmg_eps} (residual {args.dmg_res}); particle mu {args.particle_mu}, blade mu {args.mu if args.blade_mu is None else args.blade_mu}, board mu {args.mu}; bevel {args.bevel*1e3:.1f} mm to a {args.edge_t*1e3:.2f} mm edge; blade {args.blade_t*1e3:.1f} mm travel {travel*1e3:.1f} mm "
          f"at {args.speed*1e3:.0f} mm/s; {n_steps} steps", flush=True)
    # blade frame for the cutter CPIC: native x = -world y (spine -> edge), y = world x (along the blade), z = world z (across);
    # origin at the blade's initial centre, so the edge is native x = +0.03 over native y in [-0.05, 0.05]
    Rn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]); c0 = np.array([0.0, y0, 0.0])
    band = args.cpic_band if args.cpic_band > 0 else dx
    def cut_plane():
        Tb = blade.pose_from(sc.eng.get_abd_body_transforms(np.array([blade.body_id], np.int32))[0])
        M = Rn @ np.linalg.inv(Tb[:3, :3])
        sc.eng.set_liquid_cut_plane(np.c_[M, -M @ Tb[:3, 3] - Rn @ c0], 0.0, -0.05, 0.10, np.array([0.03, 0.03]), args.cpic_lead, band)
    t0 = time.time()
    status = "completed"
    fmax = {"cut": 0.0, "hold": 0.0}                  # peak drive force while cutting / holding at the bottom
    for k in range(n_steps + 1):
        t = k * args.dt
        if t < args.settle:
            d, stage = 0.0, "settle"
        elif t < args.settle + T_move:
            d, stage = args.speed * (t - args.settle), "cut"
        elif t < args.settle + T_move + args.hold:
            d, stage = travel, "hold"
        else:
            d, stage = travel * (1.0 - minjerk((t - args.settle - T_move - args.hold) / max(args.lift, 1e-9))), "lift"
        sc.eng.native.set_prismatic_target(jid, float(d))
        if k % rec_every == 0:
            P = sc.eng.get_liquid_particles()
            rec.record(t, 0, stage, liquid_pts=P)
            if not np.isfinite(P).all():
                status = f"non-finite particles at t = {t:.3f}"
                break
            Tb = blade.pose_from(sc.eng.get_abd_body_transforms(np.array([blade.body_id], np.int32))[0])
            yb = (Tb[:3, :3] @ np.array([0.0, y0, 0.0]) + Tb[:3, 3])[1] - 0.03
            f = sc.eng.native.get_prismatic_drive_force(jid)
            nb = int(np.asarray(sc.eng.get_liquid_damage()).sum()) if (args.damage > 0 or args.dmg_eps > 0) else 0
            yu = float(P[under, 1].mean()) if under.any() else float("nan")
            series.append((float(t), float(yb * 1e3), float(f), nb, yu * 1e3))
            if nb > 0 and first_break is None:
                pre = np.array(series)
                first_break = dict(t=float(t), blade_bottom_mm=float(yb * 1e3), top0_mm=float(top * 1e3), force_N=float(f),
                                   force_before_max_N=float(pre[:, 2].max()), skin_under_edge_mm=yu * 1e3,
                                   skin_under_edge_lowest_mm=float(np.nanmin(pre[:, 4])), skin_under_edge_rest_mm=float(pre[0, 4]))
            if k % (rec_every * 5) == 0:
                wid = float(P[mid, 0].max() - P[mid, 0].min()) if mid.any() else 0.0
                far = (np.abs(P[:, 2]) > 0.012) & (np.abs(P[:, 0]) < 0.006)
                print(f"t {t:5.2f} {stage:6s} target {d*1e3:6.1f} mm, blade bottom y {yb*1e3:6.1f} mm (board {BOARD_TOP*1e3:.0f}), "
                      f"drive force {f:7.1f} N, broken {nb}, top away from the cut {P[far, 1].max()*1e3:.2f} mm, width at mid-height {wid*1e3:.2f} mm, "
                      f"particles y {P[:,1].min()*1e3:.1f}..{P[:,1].max()*1e3:.1f} mm, "
                      f"cpic masked {sc.eng.get_liquid_cut_masked() if args.cpic else 0}, wall {time.time()-t0:6.1f} s", flush=True)
        if k == n_steps:
            break
        if args.cpic:
            cut_plane()
        sc.step()
        if stage in fmax:
            fmax[stage] = max(fmax[stage], float(sc.eng.native.get_prismatic_drive_force(jid)))
    P = sc.eng.get_liquid_particles()
    broken = np.asarray(sc.eng.get_liquid_damage()).astype(bool) if (args.damage > 0 or args.dmg_eps > 0) else np.zeros(len(P), bool)
    epl = np.asarray(sc.eng.get_liquid_plastic_strain())
    ok_sizes = groups(P[~broken], 1.5 * s)[0] if (~broken).sum() > 1 else np.zeros(0, int)
    sizes, lab = groups(P, 1.5 * s)
    big = sizes[sizes > 0.02 * len(P)]
    slit = int((np.abs(P[:, 2]) < 0.5 * args.blade_t).sum())
    side = [int(((P[:, 2] < 0) & (lab == l)).sum()) for l in np.argsort(np.bincount(lab))[::-1][:2]]
    # geometry of the cut: axial shift of the two far ends, and how far the intact cut faces sit from the blade plane near
    # the board (bottom) and near the top (kerf = the two summed)
    neg0 = pts[:, 2] < 0
    far_n, far_p = pts[:, 2] < pts[:, 2].min() + 0.003, pts[:, 2] > pts[:, 2].max() - 0.003
    lo_b, hi_b = P[:, 1] < BOARD_TOP + 0.015, P[:, 1] > BOARD_TOP + 0.035
    face = lambda m: round(float(np.percentile(np.abs(P[m, 2]), 2)) * 1e3, 2) if m.sum() > 20 else None
    def face_rough(sign):
        """roughness of one cut face: per (x, y) bin of 2 spacings, the particle closest to the blade plane (broken or not: crumbs
        on a face are roughness too); residual of a plane fit over the central part of the section (|x| < 0.6 R, 4 mm clear of
        the board and of the top), as rms and p95 in mm"""
        h = 2 * s
        Q = P[(sign * P[:, 2] > 0) & (np.abs(P[:, 0]) < 0.6 * args.radius) & (P[:, 1] > BOARD_TOP + 0.004)
              & (P[:, 1] < BOARD_TOP + 2 * np.sqrt(args.radius ** 2 - (0.6 * args.radius) ** 2) - 0.004)]
        if len(Q) < 50:
            return None
        key = np.floor(Q[:, 0] / h).astype(np.int64) * 100003 + np.floor(Q[:, 1] / h).astype(np.int64)
        za = np.abs(Q[:, 2]); o = np.lexsort((za, key)); ks = key[o]
        idx = o[np.r_[True, ks[1:] != ks[:-1]]]
        A = np.c_[np.ones(len(idx)), Q[idx, 0], Q[idx, 1]]
        coef = np.linalg.lstsq(A, za[idx], rcond=None)[0]
        r = za[idx] - A @ coef
        return [round(float(np.sqrt(np.mean(r ** 2))) * 1e3, 3), round(float(np.percentile(np.abs(r), 95)) * 1e3, 3), int(len(idx))]
    geo = dict(face_rough_mm=[face_rough(-1), face_rough(1)], far_ends_mm=[round(float(P[far_n, 2].mean() - pts[far_n, 2].mean()) * 1e3, 2),
                            round(float(P[far_p, 2].mean() - pts[far_p, 2].mean()) * 1e3, 2)],
               face_bottom_mm=[face(neg0 & ~broken & lo_b), face(~neg0 & ~broken & lo_b)],
               face_top_mm=[face(neg0 & ~broken & hi_b), face(~neg0 & ~broken & hi_b)],
               f_cut_max_N=round(fmax["cut"], 1), f_hold_max_N=round(fmax["hold"], 1))
    zb = {name: int((broken & (zone == k)).sum()) for k, name in ((0, "flesh"), (1, "skin"), (2, "core"))}
    res = dict(status=status, n=len(P), broken=int(broken.sum()), broken_by_zone=zb, first_break=first_break, series=series, zoned=zoned, geometry=geo, intact_groups_over_2pct=ok_sizes[ok_sizes > 0.02 * len(P)].tolist(),
               groups_over_2pct=big.tolist(), n_groups=int(len(sizes)), particles_in_slit=slit,
               largest_two_negz=side, wall_min=(time.time() - t0) / 60,
               plastic_strain=dict(p50=float(np.percentile(epl, 50)), p90=float(np.percentile(epl, 90)), p99=float(np.percentile(epl, 99)),
                                   max=float(epl.max()), over=[int((epl > e).sum()) for e in (0.1, 0.2, 0.4, 0.8, 1.6)]))
    print("[rig] result", json.dumps(res), flush=True)
    meta = dict(example="cut_rig", dt=args.dt, fps=1.0 / (rec_every * args.dt), n_liquid=len(pts), particle_spacing=s, liquid_dx=dx,
                args=vars(args), result=res, wall_min=res["wall_min"])
    rec.save(osp.join(args.out, "recording.npz"), meta)
    json.dump(meta, open(osp.join(args.out, "meta.json"), "w"), indent=1)
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""An FR3 cuts a cucumber with a chef's knife: three slices from the thin end of an MPM cucumber lying on a cutting board.

Assets (Assets/cutting/, licensed, not in git): cucumber.obj and knife.obj, converted from the user's FBX files at x1.0
(metres, Y up). The cucumber is 252 mm long and 48-50 mm thick (thin end 23-38 mm); the knife is a 327 mm chef's knife
whose blade is 4 mm thick with a ~5 mm bevel to a rounded ~1 mm edge, 13-42 mm from spine to edge.

  * World: metres, Y up, floor at y = 0. A wooden board (fixed ABD box) lies on the floor, top at y = 0.03.
  * Cucumber: model-1 MPM solid (fixed corotated) with the return-mapping plasticity of this branch: the tension cut-off
    (--stretch) lets it tear open under the blade, the stand-in for fracture. Particles on a lattice inside the closed mesh;
    the ones within --skin of the surface are the skin (colour only, same material).
  * Knife: ABD body welded to fr3_hand by three fixed joints spread along the handle (one anchor leaves a rotational slack,
    case_40). No collision between the knife and the hand / fingers; the fingers close on the handle for the picture.
  * Arm: FR3 driven by joint targets from IK; the hand stays top-down with the fingers across the handle.
  * Motion: the knife cuts straight down at the section of the blade that is ~41 mm high (native y = 0.19), lifts, moves one
    slice along the cucumber and cuts again; --push adds a forward push during each cut.

    PYTHONPATH=. STIFFGIPC_NATIVE_DIR=build_4090 python examples/cutting/cucumber_cut.py --dry
    PYTHONPATH=. STIFFGIPC_NATIVE_DIR=build_4090 python examples/cutting/cucumber_cut.py --out out/cutting/cuc --spacing 0.002
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
import time

import numpy as np

# model-1 solids need the linear particle kinematics (liquid_system.cu k_g2p / g_liq_lin): with the default two-hop transfer an
# elastic body compacts under its own weight on fine grids (CUCUMBER_CUT_PLAN.md, 10-29)
os.environ.setdefault("LIQ_EL_LIN", "1")

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(HERE, "..", "robotgym"))
from common import REPO, Recorder, Rz, Scene, Timeline, R_cols, T_of  # noqa: E402

ASSET = osp.join(REPO, "Assets", "cutting")
BOARD_TOP = 0.03
CUC_X = 0.50                                       # cucumber axis (world z) at x = 0.50, its centre at z = 0
R_CUC = R_cols((0, 0, 1), (0, 1, 0), (-1, 0, 0))    # native x (long axis) -> world +z, thin end (native -x) towards -z
R_KNIFE = R_cols((0, -1, 0), (-1, 0, 0), (0, 0, -1))  # native +x (spine -> edge) -> world -y, native +y (tip -> handle) -> world -x
R_HAND = R_cols((1, 0, 0), (0, 0, 1), (0, -1, 0))   # TCP approach (z) down, finger axis (y) along world z = across the handle
KNIFE_CUT_Y = 0.19                                  # native y of the blade section over the cucumber (41 mm spine to edge)
GRIP_Y = 0.31                                       # native y of the grip (handle centre)
ANCHOR_Y = (0.27, 0.31, 0.35)                       # fixed-joint anchors along the handle axis
ARM_BASE = (0.0, 0.0, -0.09)


def load_obj(path):
    V, F = [], []
    for line in open(path, encoding="utf-8", errors="replace"):
        s = line.split()
        if not s:
            continue
        if s[0] == "v":
            V.append([float(x) for x in s[1:4]])
        elif s[0] == "f":
            idx = [int(t.split("/")[0]) - 1 for t in s[1:]]
            F += [[idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1)]
    return np.array(V, np.float64), np.array(F, np.int32)


def lattice_inside(V, F, s, rng, jitter=0.2):
    """lattice points (spacing s) inside a closed triangle mesh, by ray parity along +x per (y, z) row; jittered by +-jitter s"""
    lo, hi = V.min(0), V.max(0)
    xs = np.arange(lo[0] + 0.5 * s, hi[0], s)
    ys = np.arange(lo[1] + 0.5 * s, hi[1], s) + 1.37e-7
    zs = np.arange(lo[2] + 0.5 * s, hi[2], s) + 2.11e-7
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    out = []
    for y0 in ys:
        m = (np.minimum(np.minimum(A[:, 1], B[:, 1]), C[:, 1]) <= y0) & (np.maximum(np.maximum(A[:, 1], B[:, 1]), C[:, 1]) >= y0)
        a, b, c = A[m], B[m], C[m]
        if not len(a):
            continue
        d = (b[:, 1] - a[:, 1]) * (c[:, 2] - a[:, 2]) - (c[:, 1] - a[:, 1]) * (b[:, 2] - a[:, 2])
        ok = np.abs(d) > 1e-14
        a, b, c, d = a[ok], b[ok], c[ok], d[ok]
        Z = zs[None, :]
        u = ((y0 - a[:, 1:2]) * (c[:, 2:3] - a[:, 2:3]) - (c[:, 1:2] - a[:, 1:2]) * (Z - a[:, 2:3])) / d[:, None]
        v = ((b[:, 1:2] - a[:, 1:2]) * (Z - a[:, 2:3]) - (y0 - a[:, 1:2]) * (b[:, 2:3] - a[:, 2:3])) / d[:, None]
        hit = (u >= 0) & (v >= 0) & (u + v <= 1)
        xh = a[:, 0:1] + u * (b[:, 0:1] - a[:, 0:1]) + v * (c[:, 0:1] - a[:, 0:1])
        for j, z0 in enumerate(zs):
            xc = np.sort(xh[hit[:, j], j])
            if len(xc) < 2:
                continue
            inside = (np.searchsorted(xc, xs) % 2) == 1
            if inside.any():
                p = np.empty((inside.sum(), 3))
                p[:, 0], p[:, 1], p[:, 2] = xs[inside], y0, z0
                out.append(p)
    P = np.concatenate(out)
    return P + rng.uniform(-jitter * s, jitter * s, P.shape)


def surface_distance(P, V, F, h):
    """distance (approx.) from each point to the mesh surface: nearest of dense surface samples (spacing ~h/3)"""
    from scipy.spatial import cKDTree
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
    n = np.maximum(1, np.ceil(area / (h / 3.0) ** 2).astype(int))
    rng = np.random.default_rng(1)
    idx = np.repeat(np.arange(len(F)), n)
    r1, r2 = rng.random(len(idx)), rng.random(len(idx))
    sq = np.sqrt(r1)
    S = (1 - sq)[:, None] * A[idx] + (sq * (1 - r2))[:, None] * B[idx] + (sq * r2)[:, None] * C[idx]
    return cKDTree(S).query(P)[0]


def sharpened_knife(V, depth, edge_t, y_range=(0.10, 0.235)):
    """collision-proxy knife with a sharper edge: in every section ring of the blade (native y in y_range; the cut section is
    y = 0.19 and the pull cut slides it to ~0.16), the vertices within `depth` of the edge (native +x) are pulled towards the
    blade's mid-plane so that the thickness falls linearly from what the ring has at `depth` to `edge_t` at the apex. The model's
    own apex is rounded, ~1 mm thick; outside that band nothing moves, the outline is unchanged (only thinned, never thickened)."""
    V = V.copy()
    ys = np.round(V[:, 1], 5)
    for y in np.unique(ys[(V[:, 1] > y_range[0]) & (V[:, 1] < y_range[1])]):
        r = np.nonzero(ys == y)[0]
        if len(r) < 8:
            continue
        x_edge = V[r, 0].max(); zm = 0.5 * (V[r, 2].min() + V[r, 2].max())
        d = x_edge - V[r, 0]; h = np.abs(V[r, 2] - zm)
        o = np.argsort(d); h_at = float(np.interp(depth, d[o], h[o]))     # half-thickness of this ring at `depth`
        band = d < depth
        target = 0.5 * edge_t + (h_at - 0.5 * edge_t) * d[band] / depth
        sc = np.minimum(1.0, target / np.maximum(h[band], 1e-9))
        V[r[band], 2] = zm + (V[r[band], 2] - zm) * sc
    return V


class Knife:
    """knife geometry in native coordinates and its placement: world = R (x - cut_native) + P_cut + (0, lift, 0) with
    R = Rz(-tilt) R_KNIFE. The edge is curved (2.4 mm lower at the heel than at the cut section, rising towards the tip), so the
    knife is tilted nose-down by the angle that keeps its lowest point highest (~3 deg: the lowest point is then within a few mm of
    the cut section) and lifted so that this lowest point, not the heel, meets the board at the bottom of each cut."""

    def __init__(self, tilt_deg=None, sharpen=0.0, sharpen_edge=0.0002):
        self.V, self.F = load_obj(osp.join(ASSET, "knife.obj"))
        if sharpen > 0:   # collision proxy only differs within `sharpen` of the edge (sharpen_edge)
            self.V = sharpened_knife(self.V, sharpen, sharpen_edge)
        blade = self.V[np.abs(self.V[:, 1] - KNIFE_CUT_Y) < 0.006]
        self.cut_native = np.array([blade[:, 0].max(), KNIFE_CUT_Y, 0.5 * (blade[:, 2].min() + blade[:, 2].max())])
        hd = self.V[np.abs(self.V[:, 1] - GRIP_Y) < 0.012]
        self.handle_c = lambda y: np.array([0.5 * (hd[:, 0].min() + hd[:, 0].max()), y, 0.5 * (hd[:, 2].min() + hd[:, 2].max())])
        self.grip_native = self.handle_c(GRIP_Y)
        self.handle_half_t = 0.5 * (hd[:, 2].max() - hd[:, 2].min())
        self.spine_over_edge = self.cut_native[0] - blade[:, 0].min()
        low = lambda th: ((self.V - self.cut_native) @ (Rz(-th) @ R_KNIFE).T)[:, 1].min()
        if tilt_deg is None:
            ths = np.arange(0.0, 8.001, 0.1)
            tilt_deg = float(ths[int(np.argmax([low(th) for th in ths]))])
        self.tilt = float(tilt_deg)
        self.R = Rz(-self.tilt) @ R_KNIFE
        self.lift = -float(low(self.tilt))          # the cut section sits this much above the knife's lowest point

    def world(self, P_cut, X=None):
        X = self.V if X is None else X
        return (X - self.cut_native) @ self.R.T + P_cut + np.array([0.0, self.lift, 0.0])

    def tcp(self, P_cut):
        return T_of(Rz(-self.tilt) @ R_HAND, self.world(P_cut, self.grip_native[None])[0])

    def edge_profile(self, K=160):
        """the blade's edge in native coordinates for the cutter CPIC: per section ring of the blade (rings thinner than 5 mm,
        i.e. not the handle), the edge is the ring's largest x; resampled at K points over the blade's span. Mid-plane z = the
        median ring centre."""
        ys = np.round(self.V[:, 1], 5)
        ring_y, ring_x, ring_z = [], [], []
        for y in np.unique(ys):
            r = self.V[ys == y]
            if len(r) < 8 or np.ptp(r[:, 2]) > 0.005:
                continue
            ring_y.append(y); ring_x.append(r[:, 0].max()); ring_z.append(0.5 * (r[:, 2].min() + r[:, 2].max()))
        ring_y = np.array(ring_y); o = np.argsort(ring_y)
        s = np.linspace(ring_y[o][0], ring_y[o][-1], K)
        return float(np.median(ring_z)), float(s[0]), float(s[1] - s[0]), np.interp(s, ring_y[o], np.array(ring_x)[o]), np.ptp(ring_z)


def cucumber(spacing, skin, rng, roll=0.0):
    V, F = load_obj(osp.join(ASSET, "cucumber.obj"))
    Vw = V @ R_CUC.T
    if roll:   # rest pose: rolled about its long axis (world z) to where it lies still, so it does not rock under the first cut
        c = Vw.mean(0)
        Vw = (Vw - c) @ Rz(roll).T + c
    lo, hi = Vw.min(0), Vw.max(0)
    shift = np.array([CUC_X - 0.5 * (lo[0] + hi[0]), BOARD_TOP + 0.0015 - lo[1], -0.5 * (lo[2] + hi[2])])
    Vw = Vw + shift
    P = lattice_inside(Vw, F, spacing, rng)
    d = surface_distance(P, Vw, F, spacing)
    return Vw, F, P, d < skin, d


def radial_fraction(P, d, slab=0.004):
    """0 on the cucumber's axis .. 1 at its surface, per particle: 1 - depth / local radius, the local radius being the
    largest depth found in a slab of +-`slab` along the body axis (world z), smoothed along z. Needs no centre line, so it
    follows the bend and the taper of the model."""
    z = P[:, 2]
    zb = np.arange(z.min(), z.max() + 1e-9, 0.001)
    order = np.argsort(z)
    zs, ds = z[order], d[order]
    R = np.empty(len(zb))
    for i, zc in enumerate(zb):
        a, b = np.searchsorted(zs, zc - slab), np.searchsorted(zs, zc + slab)
        R[i] = np.percentile(ds[a:b], 99.5) if b - a > 20 else np.nan
    ok = np.isfinite(R)
    R = np.interp(zb, zb[ok], R[ok])
    k = np.ones(5) / 5.0
    R = np.convolve(np.pad(R, 2, mode="edge"), k, mode="valid")
    Rp = np.interp(z, zb, R)
    return np.clip(1.0 - d / np.maximum(Rp, 1e-6), 0.0, 1.0), Rp


def zone_material(depth, rfrac, args):
    """per-particle factors on Young's modulus and on the damage stretch threshold: skin (depth < --skin), flesh, core
    (radial fraction < --core-frac, blended into the flesh over +- --core-blend). zone: 0 flesh, 1 skin, 2 core."""
    n = len(depth)
    ys, ds, zone = np.ones(n), np.ones(n), np.zeros(n, np.int8)
    if args.core_young_x != 1.0 or args.core_damage_x != 1.0:
        lo, hi = args.core_frac - args.core_blend, args.core_frac + args.core_blend
        w = np.clip((hi - rfrac) / max(hi - lo, 1e-9), 0.0, 1.0)            # 1 in the core, 0 in the flesh
        w = w * w * (3.0 - 2.0 * w)
        ys = np.exp(w * np.log(args.core_young_x))                           # geometric blend: equal ratios per unit radius
        ds = np.exp(w * np.log(args.core_damage_x))
        zone[rfrac < args.core_frac] = 2
    sk = depth < args.skin
    if args.skin_young_x != 1.0 or args.skin_damage_x != 1.0:
        ys[sk], ds[sk] = args.skin_young_x, args.skin_damage_x
    zone[sk] = 1
    return ys, ds, zone


def profile(Vw, z):
    """cucumber top (y) near z"""
    S = Vw[np.abs(Vw[:, 2] - z) < 0.004]
    return float(S[:, 1].max())


def timeline(knife, Vw, cuts, args):
    tops = [profile(Vw, z) for z in cuts]
    P = lambda z, y, x=CUC_X: np.array([x, y, z])
    edge_low = BOARD_TOP + args.edge_gap
    tl = Timeline(knife.tcp(P(cuts[0], tops[0] + args.clear + 0.02)))
    tl.add("settle", args.settle)
    for i, (z, top) in enumerate(zip(cuts, tops)):
        tl.add(f"approach {i+1}", args.approach, knife.tcp(P(z, top + args.clear)))
        dur = (top + args.clear - edge_low) / args.speed
        tl.add(f"cut {i+1}", dur, knife.tcp(P(z, edge_low, CUC_X + args.push)))
        if args.slice > 0:     # pull cut along the blade at the bottom: the edge drawn through the last layer on the board
            tl.add(f"slice {i+1}", args.slice_time, knife.tcp(P(z, edge_low, CUC_X + args.push - args.slice)))
        tl.add(f"hold {i+1}", args.hold)
        x_end = CUC_X + args.push - max(args.slice, 0.0)
        tl.add(f"lift {i+1}", args.lift, knife.tcp(P(z, top + args.clear + 0.01, x_end)))
    tl.add("retreat", 1.2, knife.tcp(P(cuts[-1] + 0.06, tops[-1] + 0.08, CUC_X - 0.05)))
    tl.add("rest", 0.5)
    return tl, tops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/cutting/cucumber")
    ap.add_argument("--dry", action="store_true", help="IK, clearances and particle seeding only (no engine)")
    ap.add_argument("--dt", type=float, default=0.0025)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--spacing", type=float, default=0.002, help="particle seed spacing (m); grid dx = 2 x spacing")
    ap.add_argument("--skin", type=float, default=0.002, help="particles within this depth are skin (colour only)")
    ap.add_argument("--young", type=float, default=2e6, help="Young's modulus of the flesh (Pa)")
    ap.add_argument("--skin-young-x", type=float, default=1.0, help="skin (within --skin of the surface): Young's modulus = this x --young")
    ap.add_argument("--skin-damage-x", type=float, default=1.0, help="skin: damage stretch threshold = this x --damage (tougher skin > 1)")
    ap.add_argument("--core-young-x", type=float, default=1.0, help="seed core: Young's modulus = this x --young (soft core < 1)")
    ap.add_argument("--core-damage-x", type=float, default=1.0, help="seed core: damage stretch threshold = this x --damage")
    ap.add_argument("--core-frac", type=float, default=0.55, help="core radius as a fraction of the local radius of the cucumber")
    ap.add_argument("--core-blend", type=float, default=0.12, help="half-width of the core / flesh transition (fraction of the radius)")
    ap.add_argument("--poisson", type=float, default=0.4)
    ap.add_argument("--rho", type=float, default=960.0)
    ap.add_argument("--stretch", type=float, default=0.02, help="plastic_stretch (tension cut-off)")
    ap.add_argument("--compress", type=float, default=0.1, help="plastic_compress")
    ap.add_argument("--yield", dest="yield_", type=float, default=0.0, help="plastic_yield (von Mises)")
    ap.add_argument("--damage", type=float, default=0.0, help="plastic_damage: a particle breaks for good above principal stretch 1 + d")
    ap.add_argument("--cuts", default="-0.102,-0.087,-0.072", help="world z of the cuts (slices of 15 mm from the thin end)")
    ap.add_argument("--speed", type=float, default=0.03, help="mean blade speed while cutting (m/s)")
    ap.add_argument("--push", type=float, default=0.0, help="forward push of the knife during each cut (m)")
    ap.add_argument("--slice", type=float, default=0.03, help="pull cut at the bottom of each cut: knife moves this far back along its length (m)")
    ap.add_argument("--slice-time", type=float, default=0.6)
    ap.add_argument("--edge-gap", type=float, default=0.0008, help="edge height above the board at the bottom of each cut (m)")
    ap.add_argument("--roll", type=float, default=-27.0, help="cucumber roll about its long axis (deg): -27 is where the model comes to rest (measured: it rolled there and rocked for ~3 s when placed at 0; the rigid-mesh minimum of the centroid height is -19.5 / -30)")
    ap.add_argument("--aim", default="far", choices=["far", "face"], help="re-aim each cut by the axial shift of the far part "
                    "(far), or, for the cuts after the first, at the planned slice thickness from the body's actual cut face (face)")
    ap.add_argument("--aim-extra", type=float, default=0.0, help="--aim face: put the knife this much (m) further from the face "
                    "than the planned thickness, for the body sliding away while the knife goes down (no holder)")
    ap.add_argument("--no-track", action="store_true", help="cut at the fixed planned z instead of re-aiming each cut at the cucumber's current position (every cut pushes the body ~4-6 mm along its axis)")
    ap.add_argument("--sharpen", type=float, default=0.0, help="collision proxy: thin the knife within this distance of its edge (m) down to --sharpen-edge at the apex (the model's apex is rounded, ~1 mm thick); 0 = the model as is")
    ap.add_argument("--sharpen-edge", type=float, default=0.0002, help="apex thickness of the sharpened proxy (m)")
    ap.add_argument("--knife-board", action="store_true", help="keep knife-board contact (default: excluded, so the edge can reach the board top and only the cucumber resists; with contact the arm presses the knife into the board's barrier)")
    ap.add_argument("--clear", type=float, default=0.012, help="edge above the cucumber top before each cut (m)")
    ap.add_argument("--zmax", type=float, default=1.0, help="keep only the cucumber up to this world z (m): the thin-end "
                    "section at a finer spacing; the cut-off end is a flat face (pair with --stop to hold it there)")
    ap.add_argument("--stop", type=float, default=-1.0, help="fixed stop block this far (m) beyond the cucumber's far end, the other "
                    "hand holding it: the cuts no longer push the body along its axis (< 0: none)")
    ap.add_argument("--settle", type=float, default=0.6)
    ap.add_argument("--approach", type=float, default=1.0)
    ap.add_argument("--hold", type=float, default=0.3)
    ap.add_argument("--lift", type=float, default=0.8)
    ap.add_argument("--tilt", type=float, default=None, help="knife nose-down tilt (deg); default: the smallest that makes the cut section lowest")
    ap.add_argument("--knife-density", type=float, default=2900.0, help="kg/m3: the 77 ml knife weighs ~0.22 kg")
    ap.add_argument("--mu", type=float, default=0.4, help="board / global friction")
    ap.add_argument("--particle-mu", type=float, default=0.4, help="cucumber particle friction (pair = sqrt(this x the body's)): 0.4 on the board")
    ap.add_argument("--knife-mu", type=float, default=0.05, help="knife body friction: pair with the cucumber sqrt(0.4 x 0.05) = 0.14 (wet steel)")
    ap.add_argument("--dmg-res", type=float, default=0.1, help="plastic_damage_residual: share of the elastic energy a broken particle keeps")
    ap.add_argument("--dmg-eps", type=float, default=0.0, help="plastic_damage_strain: ductile damage above this accumulated plastic strain (0 = off)")
    ap.add_argument("--cutter-gate", action="store_true", help="set_liquid_cutter(knife): only particles the knife has touched may break")
    ap.add_argument("--dhat", type=float, default=5e-4)
    ap.add_argument("--newton-cap", type=int, default=80)
    ap.add_argument("--max-time", type=float, default=0.0)
    ap.add_argument("--log-level", type=int, default=0)
    ap.add_argument("--cpic", action="store_true", help="cutter CPIC (set_liquid_cut_plane): the knife's mid-plane, from its edge up plus --cpic-lead beyond it, is a crack surface the MPM grid does not couple across (the knife model itself is unchanged)")
    ap.add_argument("--cpic-lead", type=float, default=0.001, help="crack front ahead of the edge (m)")
    ap.add_argument("--cpic-band", type=float, default=0.0, help="mask band half-width around the mid-plane (m); 0 = dx")
    ap.add_argument("--pieces", action="store_true", help="2026-11-13: persistent pieces (Engine.set_liquid_pieces, needs --cpic and LIQ_CUT_FIELDS=2). "
                    "When the knife starts to lift after cut i, the body's particles on the cut-off side of the knife's mid-plane become piece i; "
                    "from then on a grid node that two pieces reference is split unless their particles are within --piece-contact seed spacings, "
                    "so slices only push on each other where they touch (before: from up to two cells away)")
    ap.add_argument("--piece-contact", type=float, default=1.0, help="piece contact distance in seed spacings")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    knife = Knife(args.tilt, args.sharpen, args.sharpen_edge)
    Vc, Fc, pts, skin, depth = cucumber(args.spacing, args.skin, rng, args.roll)
    keep = pts[:, 2] < args.zmax
    pts, skin, depth = pts[keep], skin[keep], depth[keep]
    rfrac, r_local = radial_fraction(pts, depth)
    young_x, damage_x, zone = zone_material(depth, rfrac, args)
    zoned = bool(np.any(young_x != 1.0) or np.any(damage_x != 1.0))
    zend = float(min(Vc[:, 2].max(), pts[:, 2].max() + 0.5 * args.spacing))
    cuts = [float(z) for z in args.cuts.split(",")]
    tl, tops = timeline(knife, Vc, cuts, args)
    T_total = tl.total if args.max_time <= 0 else min(tl.total, args.max_time)
    print(f"[scene] cucumber {len(pts)} particles (skin {skin.sum()}), spacing {args.spacing*1e3:.1f} mm, dx {2*args.spacing*1e3:.1f} mm, "
          f"{len(pts)*args.spacing**3*1e6:.0f} ml; cuts at z {cuts} where the top is {[round(1e3*(t-BOARD_TOP), 1) for t in tops]} mm "
          f"above the board; knife spine {knife.spine_over_edge*1e3:.1f} mm over the edge there; handle half-thickness "
          f"{knife.handle_half_t*1e3:.1f} mm; knife tilted {knife.tilt:.2f} deg nose-down (cut section {knife.lift*1e3:.2f} mm above its lowest point); timeline {tl.total:.1f} s", flush=True)

    from common import Arm, arm_base_T  # noqa: E402
    arm_kin = Arm("arm", arm_base_T(ARM_BASE, 0.0)).kin
    q0, ok = arm_kin.ik(tl.segs[0].T_to, arm_kin.HOME, iters=400)
    print(f"[scene] start pose IK ok {ok}, q0 {np.round(q0, 3)}", flush=True)
    print(f"[scene] zones: local radius {r_local.min()*1e3:.1f}..{r_local.max()*1e3:.1f} mm (median {np.median(r_local)*1e3:.1f}); skin {int((zone == 1).sum())} particles "
          f"(E x {args.skin_young_x:g}, damage x {args.skin_damage_x:g}), core {int((zone == 2).sum())} (radial fraction < {args.core_frac:g}, E x {args.core_young_x:g}, "
          f"damage x {args.core_damage_x:g}), flesh {int((zone == 0).sum())}; E factor {young_x.min():.3g}..{young_x.max():.3g}"
          + ("" if zoned else "  -> uniform material"), flush=True)
    finger = min(0.04, knife.handle_half_t + 0.0005)

    if args.dry:
        from scipy.spatial import cKDTree
        tree_c = cKDTree(Vc)
        q = q0.copy(); worst = 0.0
        dmin = {}
        for t in np.arange(0.0, tl.total, 0.02):
            seg, s = tl.at(t); T = seg.pose(s)
            q, ok = arm_kin.ik(T, q, iters=80)
            worst = max(worst, float(np.linalg.norm(arm_kin.fk(q)[:3, 3] - T[:3, 3])))
            lp = arm_kin.link_poses(q, finger)
            for l, Tl in lp.items():
                if l == "fr3_hand_tcp":
                    continue
                y = Tl[1, 3]
                dmin[l] = min(dmin.get(l, (9, 9))[0], y - BOARD_TOP), min(dmin.get(l, (9, 9))[1], float(tree_c.query(Tl[:3, 3])[0]))
        print(f"[dry] worst IK position error {worst*1e3:.2f} mm")
        for l, (dy, dc) in dmin.items():
            print(f"[dry] {l:16s} frame origin: min height over the board {dy*1e3:7.1f} mm, min distance to the cucumber {dc*1e3:7.1f} mm")
        for i, z in enumerate(cuts):
            Kw = knife.world(np.array([CUC_X + args.push, BOARD_TOP + args.edge_gap, z]))
            print(f"[dry] knife at the bottom of cut {i+1}: lowest point {1e3*(Kw[:,1].min()-BOARD_TOP):.2f} mm above the board "
                  f"(cut section {args.edge_gap*1e3:.2f}), top {1e3*(Kw[:,1].max()-BOARD_TOP):.1f} mm, x {Kw[:,0].min():.3f}..{Kw[:,0].max():.3f}, "
                  f"z {Kw[:,2].min():.4f}..{Kw[:,2].max():.4f}")
        np.savez_compressed(osp.join(HERE, "dry_seed.npz"), pts=pts.astype(np.float32), skin=skin, Vc=Vc, Fc=Fc, zone=zone, rfrac=rfrac.astype(np.float32),
                            young_x=young_x.astype(np.float32), damage_x=damage_x.astype(np.float32), r_local=r_local.astype(np.float32))
        return

    os.makedirs(args.out, exist_ok=True)
    sc = Scene(dict(dt=args.dt, preconditioner_type=0, newton_iter_cap=args.newton_cap, newton_tol=1e-3, pcg_tol=1e-5,
                    absolute_dhat=args.dhat, friction_rate=args.mu, gd_friction_rate=args.mu, gravity=(0.0, -9.81, 0.0)),
               log_level=args.log_level)
    arm = sc.add_arm("arm", ARM_BASE, 0.0, q0=q0)
    arm.finger_d = arm.finger_goal = finger
    sc.add_box("board", (CUC_X - 0.04, BOARD_TOP - 0.015, -0.05), (0.34, 0.03, 0.42), fixed=True, color=(0.70, 0.52, 0.33),
               mu=args.mu, cell=0.02)
    if args.stop >= 0:   # 80 x 60 x 20 mm block, 2 mm above the board (no fixed-fixed contact), facing the far tip
        sc.add_box("stop", (CUC_X, BOARD_TOP + 0.032, zend + args.stop + 0.01), (0.08, 0.06, 0.02), fixed=True,
                   color=(0.55, 0.40, 0.26), mu=args.mu, cell=0.01)
    P0 = np.array([CUC_X, tops[0] + args.clear + 0.02, cuts[0]])
    Kw = knife.world(P0)
    sc.eng.load_mesh_from_data(np.ascontiguousarray(Kw), np.ascontiguousarray(knife.F), verts_per_face=3, dimensions=3,
                               body_type="ABD", transform=np.eye(4), young_modulus=1e8, boundary_type="Free")
    from common import Body  # noqa: E402
    rec_k = sc.eng.get_load_records()[-1]
    kb = Body(rec_k, "knife", "prop", color=(0.82, 0.84, 0.86))
    sc.bodies["knife"] = kb
    sc.eng.set_abd_body_density(kb.body_id, args.knife_density)
    sc.eng.set_body_friction(kb.body_id, args.knife_mu)
    hand = arm.bodies["fr3_hand"].body_id
    for l in ("fr3_hand", "fr3_leftfinger", "fr3_rightfinger", "fr3_link7"):
        sc.eng.add_collision_exclusion(arm.bodies[l].body_id, kb.body_id)
    if not args.knife_board:
        sc.eng.add_collision_exclusion(sc.bodies["board"].body_id, kb.body_id)
    nrm = knife.R @ np.array([1.0, 0.0, 0.0]); bit = knife.R @ np.array([0.0, 0.0, 1.0])
    for y in ANCHOR_Y:
        a = knife.world(P0, knife.handle_c(y)[None])[0]
        sc.fixed_joint_ids.append(sc.eng.native.add_fixed_joint(parent_body=hand, child_body=kb.body_id, world_anchor=a,
                                                                world_normal=nrm, world_bitangent=bit))
    lo = np.array([CUC_X - 0.09, 0.0, -0.17]); hi = np.array([CUC_X + 0.09, BOARD_TOP + 0.12, 0.17])
    sc.eng.add_liquid(pts, 2.0 * args.spacing, lo, hi, rho=args.rho, kappa=1e5, particle_volume=args.spacing ** 3,
                      node_pool=int(4 * len(pts)), push_apart=0.0, j_mode=0, ground_mu=args.mu, model=1, young=args.young,
                      poisson=args.poisson, plastic_stretch=args.stretch, plastic_compress=args.compress, plastic_yield=args.yield_,
                      plastic_damage=args.damage, plastic_damage_residual=args.dmg_res, mu=args.particle_mu,
                      plastic_damage_strain=args.dmg_eps)
    if args.cutter_gate:
        sc.eng.set_liquid_cutter(kb.body_id)
    sc.finalize()
    if zoned:
        sc.eng.set_liquid_particle_material(young_x, damage_x)
    rec = Recorder(sc)
    np.savez_compressed(osp.join(args.out, "particles.npz"), skin=skin, x0=pts.astype(np.float32), cucumber_V=Vc.astype(np.float32),
                        cucumber_F=Fc, zone=zone, rfrac=rfrac.astype(np.float32), young_x=young_x.astype(np.float32),
                        damage_x=damage_x.astype(np.float32), r_local=r_local.astype(np.float32))
    meta = dict(example="cucumber_cut", dt=args.dt, n_liquid=len(pts), particle_spacing=args.spacing, liquid_dx=2 * args.spacing,
                args=vars(args), cuts=cuts, tops=tops, board_top=BOARD_TOP, timeline=[(s.label, s.dur) for s in tl.segs[1:]],
                zones=dict(skin_depth=args.skin, skin_young_x=args.skin_young_x, skin_damage_x=args.skin_damage_x, core_frac=args.core_frac,
                           core_blend=args.core_blend, core_young_x=args.core_young_x, core_damage_x=args.core_damage_x, zoned=zoned),
                material=dict(model="fixed corotated + return mapping", young=args.young, poisson=args.poisson,
                              stretch_cap=args.stretch, compress_floor=args.compress, von_mises=args.yield_, damage=args.damage,
                              linear_kinematics=os.environ.get("LIQ_EL_LIN")))
    if args.cpic:
        # (names prefixed cp_: the loop below reuses `edge` for the logged edge point, and this closure must not see that)
        cp_zm, cp_s0, cp_ds, cp_edge, zspread = knife.edge_profile()
        cp_band = args.cpic_band if args.cpic_band > 0 else 2.0 * args.spacing
        cp_Lw = P0 + np.array([0.0, knife.lift, 0.0])
        print(f"[cpic] blade span native y {cp_s0:.4f}..{cp_s0 + cp_ds * (len(cp_edge) - 1):.4f} m, edge x {cp_edge.min():.4f}..{cp_edge.max():.4f}, "
              f"mid-plane z {cp_zm:.5f} (ring centres spread {zspread*1e3:.2f} mm), lead {args.cpic_lead*1e3:.2f} mm, band {cp_band*1e3:.2f} mm", flush=True)
        meta.update(cpic=dict(lead=args.cpic_lead, band=cp_band, zm=cp_zm, s0=cp_s0, ds=cp_ds, K=len(cp_edge)))

        def cut_plane():
            Tk = kb.pose_from(sc.eng.get_abd_body_transforms(np.array([kb.body_id], np.int32))[0])
            M = knife.R.T @ np.linalg.inv(Tk[:3, :3])
            sc.eng.set_liquid_cut_plane(np.c_[M, -M @ Tk[:3, 3] - knife.R.T @ cp_Lw + knife.cut_native], cp_zm, cp_s0, cp_ds, cp_edge,
                                        args.cpic_lead, cp_band)
    piece_id = np.zeros(len(pts), np.int32)          # --pieces: 0 = the body, i = cut off by cut i
    piece_log = []
    if args.pieces and not args.cpic:
        raise SystemExit("--pieces needs --cpic (the knife's mid-plane defines the pieces)")

    def assign_piece(i):
        """Cut i is done (the knife starts to lift): everything of the body on the cut-off side of the knife's mid-plane is piece i."""
        Tk = kb.pose_from(sc.eng.get_abd_body_transforms(np.array([kb.body_id], np.int32))[0])
        M = knife.R.T @ np.linalg.inv(Tk[:3, :3])
        tt = -M @ Tk[:3, 3] - knife.R.T @ cp_Lw + knife.cut_native
        Pn = np.asarray(sc.eng.get_liquid_particles(), np.float64)
        dz = Pn @ M[2] + tt[2] - cp_zm                 # signed distance to the mid-plane in the knife's frame
        body_sign = np.sign(np.median(dz[far]))        # the far part of the cucumber is the body
        cutoff = (dz * body_sign < 0.0) & (piece_id == 0)
        piece_id[cutoff] = i
        sc.eng.set_liquid_pieces(piece_id, args.piece_contact)
        piece_log.append(dict(cut=i, t=float(t), particles=int(cutoff.sum())))
        print(f"[pieces] cut {i}: {int(cutoff.sum())} particles on the cut-off side become piece {i} "
              f"(pieces now: {dict(zip(*[x.tolist() for x in np.unique(piece_id, return_counts=True)]))})", flush=True)

    rec_every = max(1, int(round(1.0 / (args.fps * args.dt))))
    n_steps = int(round(T_total / args.dt))
    q = q0.copy(); last = None; status = "completed"; t0 = time.time()
    far = pts[:, 2] > min(-0.03, zend - 0.03)         # the far part of the cucumber: its axial position tracks the body
    zfar0 = float(pts[far, 2].mean()); aimed = []
    dmg_frames = []                                  # bit-packed damage flags per recorded frame (cut surfaces in the render)
    knife_f = []                                     # contact force on the knife per recorded frame (N): it only touches the cucumber
    kv0, kv1 = kb.v_off, kb.v_off + int(kb.rec.vertex_count)
    try:
        for k in range(n_steps + 1):
            t = k * args.dt
            seg, s = tl.at(t)
            if seg.label != last and seg.label.startswith("approach ") and not args.no_track:
                # closed-loop re-aiming (what a camera would do): the earlier cuts pushed the body along its axis, so move
                # this cut by the body's displacement to keep the planned slice thickness in the material
                i = int(seg.label.split()[1])
                Pn = np.asarray(sc.eng.get_liquid_particles())
                dz = float(Pn[far, 2].mean() - zfar0)
                how = f"the body has moved {dz*1e3:+.1f} mm along its axis"
                if args.aim == "face" and i > 1:
                    # the far part's shift misses how much the near end was pushed or compressed (thin slices came out 0.4-6 mm
                    # for 3-5 mm planned): measure the body's cut face instead, in the core of the section next to it, and put
                    # the knife the planned slice thickness beyond it
                    body = Pn[:, 2] > aimed[-1]
                    zb = Pn[body, 2]
                    near = np.zeros(len(Pn), bool); near[body] = zb < np.percentile(zb, 0.5) + 0.004
                    c = Pn[near, :2].mean(0)
                    core = near & (np.hypot(Pn[:, 0] - c[0], Pn[:, 1] - c[1]) < 0.006)
                    zface = float(np.percentile(Pn[core, 2], 2.0))
                    dz = zface + (cuts[i - 1] - cuts[i - 2]) + args.aim_extra - cuts[i - 1]
                    how = f"its cut face is at z = {zface*1e3:.1f} mm ({int(core.sum())} core particles), slice + {args.aim_extra*1e3:.1f} mm"
                tl.shift([f"{lab} {i}" for lab in ("approach", "cut", "slice", "hold", "lift")], np.array([0.0, 0.0, dz]))
                aimed.append(cuts[i - 1] + dz)
                print(f"[track] cut {i}: {how} -> cut at z = {aimed[-1]*1e3:.1f} mm (planned {cuts[i-1]*1e3:.1f})", flush=True)
            q, _ = arm.kin.ik(seg.pose(s), q, iters=40)
            arm.drive(sc.eng, q)
            if seg.label != last:
                print(f"[{t:6.2f}s] {seg.label}  (wall {(time.time()-t0)/60:.1f} min)", flush=True)
                if args.pieces and seg.label.startswith("lift "):
                    assign_piece(int(seg.label.split()[1]))
                last = seg.label
            if k % rec_every == 0:
                P = sc.eng.get_liquid_particles()
                rec.record(t, 0, seg.label, liquid_pts=P)
                D = np.asarray(sc.eng.get_liquid_damage()).astype(bool)
                dmg_frames.append(np.packbits(D))
                knife_f.append(np.asarray(sc.eng.get_vertex_contact_forces(include_ground=False, components="total"))[kv0:kv1].sum(0))
                if not np.isfinite(P).all():
                    raise FloatingPointError(f"non-finite particles at t = {t:.3f}")
                if k % (rec_every * 5) == 0:
                    Tk = kb.pose_from(sc.eng.get_abd_body_transforms(np.array([kb.body_id], np.int32))[0])
                    edge = (Tk[:3, :3] @ knife.world(P0, knife.cut_native[None])[0] + Tk[:3, 3])
                    print(f"  t {t:6.2f} {seg.label:10s} broken {int(D.sum())} edge at y {edge[1]*1e3:6.1f} mm (board {BOARD_TOP*1e3:.0f}, TCP target y {seg.pose(s)[1,3]*1e3:6.1f}), "
                          f"particles y {P[:,1].min()*1e3:.1f}..{P[:,1].max()*1e3:.1f} mm, knife force {np.linalg.norm(knife_f[-1]):6.1f} N "
                          f"(y {knife_f[-1][1]:+6.1f}), cpic masked {sc.eng.get_liquid_cut_masked() if args.cpic else 0}, wall {time.time()-t0:7.1f} s", flush=True)
            if k == n_steps:
                break
            if args.cpic:
                cut_plane()
            sc.step()
    except BaseException as e:
        import traceback
        status = f"failed: {type(e).__name__}: {e}"
        traceback.print_exc()
    meta.update(status=status, sim_s=k * args.dt, wall_min=(time.time() - t0) / 60, aimed_cuts=aimed)
    if args.pieces:
        meta.update(pieces=dict(contact_spacings=args.piece_contact, log=piece_log))
        np.save(osp.join(args.out, "piece_id.npy"), piece_id)
    if dmg_frames:
        np.savez_compressed(osp.join(args.out, "damage.npz"), packed=np.stack(dmg_frames), n=len(pts))
    if knife_f:
        np.savez_compressed(osp.join(args.out, "knife_force.npz"), F=np.stack(knife_f), fps=args.fps)
        Fn = np.linalg.norm(np.stack(knife_f), axis=1)
        meta.update(knife_force_peak_N=float(Fn.max()), knife_force_p95_N=float(np.percentile(Fn, 95)))
    rec.save(osp.join(args.out, "recording.npz"), meta)
    json.dump(meta, open(osp.join(args.out, "meta.json"), "w"), indent=1, default=float)
    print(f"[done] {status} sim {k*args.dt:.2f} s wall {(time.time()-t0)/60:.1f} min -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

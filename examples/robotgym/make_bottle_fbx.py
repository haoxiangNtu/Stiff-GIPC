#!/usr/bin/env python3
"""PET bottle shell for the squeeze task from the user's FBX asset (塑料水瓶.fbx): rounded-square body, 5 wavy ribs, two
straight grooves, pyramidal shoulder, concave base (rim lowest), open mouth.

The FBX body surface is sampled as a cylindrical height map R(phi, y) (horizontal slices, 2D ray casts from the axis) and the
base as y(u, phi) (vertical rays; u = r / rim radius), rescaled, and re-meshed as one single-layer triangle shell (the IPC
shell model is a mid-surface; the render mesh's thickness/inner wall and the label layer are dropped).

Only a uniform scale is applied (the shape, ribs, base and the neck support ring are kept).
Scale: the FBX bottle is 44.9 mm across the flats, 168 mm tall, with a 22.1 mm neck. No PET water bottle has a neck under
26 mm; at x1.22 the neck becomes the standard 28 mm finish (PCO 1881: thread 27.4 mm, support ring 33 mm -> FBX 26.6 x 1.22 =
32.5 mm) and the bottle a standard 500 ml one (54.8 mm flats, 205 mm, 496 ml to the brim).

    python examples/robotgym/make_bottle_fbx.py ~/Downloads/塑料水瓶.fbx --out Assets/robotgym/fbx/pet_bottle_fbx_2p5mm.obj
writes the .obj (metres, axis +y through the origin, base rim at y = 0) and <out>.npz (scaled height maps for seeding water).
"""
from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from fbx_read import Scene, clean, local_tf, props70  # noqa: E402


def body_triangles(fbx_path, material_hint="059"):
    """World-space (m) triangles of the bottle body material (the other material is the label band)."""
    sc = Scene(fbx_path)
    s = 0.01 * sc.unit_cm
    for mid, m in sc.models().items():
        geo = sc.mesh_of(mid)
        if geo is None:
            continue
        W = sc.world(mid); _, G = local_tf(props70(m)); M = W @ G
        V, idx, starts, lens, mat, _ = sc.geometry_arrays(geo)
        Vw = (V @ M[:3, :3].T + M[:3, 3]) * s
        tris, poly_of = sc.triangulate(idx, starts, lens)
        mats = sc.materials_of(mid)
        keep = [k for k, n in enumerate(mats) if material_hint in n]
        sel = np.isin(mat[poly_of], keep) if keep else np.ones(len(tris), bool)
        return Vw, idx[tris[sel]]
    raise SystemExit("no mesh in the FBX")


def ray_seg_max(segs, D):
    """segs (M,2,2) in axis-centred (x,z); farthest positive hit along each ray direction D (K,2) (nan if none)."""
    a, b = segs[:, 0, :], segs[:, 1, :]
    e = b - a
    out = np.full(len(D), np.nan)
    for i0 in range(0, len(D), 180):
        d = D[i0:i0 + 180]
        det = d[:, None, 0] * (-e[None, :, 1]) - d[:, None, 1] * (-e[None, :, 0])
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (a[None, :, 0] * (-e[None, :, 1]) - a[None, :, 1] * (-e[None, :, 0])) / det
            s = (d[:, None, 0] * a[None, :, 1] - d[:, None, 1] * a[None, :, 0]) / det
        ok = (np.abs(det) > 1e-14) & (t > 0) & (s >= -1e-9) & (s <= 1 + 1e-9)
        t = np.where(ok, t, -np.inf).max(1)
        out[i0:i0 + 180] = np.where(np.isfinite(t), t, np.nan)
    return out


def sample(V, F, dphi_deg=0.5, dy=0.0005, fine=(0.1545, 0.1605), dy_fine=0.0001):
    import trimesh
    m = trimesh.Trimesh(V, F, process=True)
    lo, hi = m.vertices.min(0), m.vertices.max(0)
    cx, cz = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[2] + hi[2])
    y0, y1 = lo[1], hi[1]
    PHI = np.radians(np.arange(0.0, 360.0, dphi_deg))
    D = np.stack([np.cos(PHI), np.sin(PHI)], 1)
    # 0.1 mm steps through the neck support ring (flat-bottomed flange: 157.0 mm, rim 14.4 mm, 0.9 mm tall, FBX scale)
    YS = np.unique(np.round(np.concatenate([np.arange(y0 + 0.0002, y0 + fine[0], dy), np.arange(y0 + fine[0], y0 + fine[1], dy_fine),
                                            np.arange(y0 + fine[1], y1 - 0.0001, dy)]), 7))
    R = np.full((len(YS), len(PHI)), np.nan)
    for k, y in enumerate(YS):
        segs = trimesh.intersections.mesh_plane(m, [0, 1, 0], [cx, y, cz])
        if len(segs):
            R[k] = ray_seg_max(np.stack([segs[:, :, 0] - cx, segs[:, :, 2] - cz], 2), D)
    assert np.isfinite(R).all(), "height map has holes"
    U = np.linspace(0.0, 1.0, 41)
    PHB = np.radians(np.arange(0.0, 360.0, 2.0))
    Rrim = np.interp(np.degrees(PHB), np.degrees(PHI), R[0], period=360.0)
    orig = np.array([[cx + u * Rrim[j] * 0.999 * np.cos(ph), y0 - 0.01, cz + u * Rrim[j] * 0.999 * np.sin(ph)]
                     for u in U for j, ph in enumerate(PHB)])
    loc, ri, _ = m.ray.intersects_location(orig, np.tile([0.0, 1.0, 0.0], (len(orig), 1)), multiple_hits=True)
    YB = np.full(len(orig), np.inf)
    np.minimum.at(YB, ri, loc[:, 1])
    YB = YB.reshape(len(U), len(PHB))
    assert np.isfinite(YB).all(), "base height map has holes"
    return dict(phi=PHI, y=YS - y0, R=R, u=U, phib=PHB, ybase=YB - y0, height=y1 - y0)


class Shape:
    """Scaled bottle: R(phi, y) (side) and y_base(u, phi) (bottom), periodic in phi."""

    def __init__(self, hm, scale, neck_y0=None, neck_r=None):
        self.s = scale
        self.phi = hm["phi"]; self.y = hm["y"] * scale; self.R = hm["R"] * scale
        self.u = hm["u"]; self.phib = hm["phib"]; self.yb = hm["ybase"] * scale
        self.H = hm["height"] * scale
        if neck_y0 is not None and neck_r is not None:      # optional: drop the support ring (shape change, off by default)
            nk = self.y >= neck_y0 * scale
            self.R[nk] = np.minimum(self.R[nk], neck_r * scale)

    @classmethod
    def load(cls, npz_path):
        """the scaled, neck-clamped height maps written next to the .obj"""
        d = np.load(npz_path)
        self = cls.__new__(cls)
        self.s = float(d["scale"]); self.phi = d["phi"]; self.y = d["y"]; self.R = d["R"]
        self.u = d["u"]; self.phib = d["phib"]; self.yb = d["ybase"]; self.H = float(d["height"])
        return self

    def R_at(self, phi, y):
        """bilinear in (y, phi), phi periodic"""
        phi = np.mod(phi, 2 * np.pi)
        dphi = self.phi[1] - self.phi[0]
        fj = phi / dphi; j0 = np.floor(fj).astype(int) % len(self.phi); j1 = (j0 + 1) % len(self.phi); tj = fj - np.floor(fj)
        y = np.clip(y, self.y[0], self.y[-1])
        k0 = np.clip(np.searchsorted(self.y, y, side="right") - 1, 0, len(self.y) - 2); k1 = k0 + 1
        ty = (y - self.y[k0]) / (self.y[k1] - self.y[k0])
        r0 = self.R[k0, j0] * (1 - tj) + self.R[k0, j1] * tj
        r1 = self.R[k1, j0] * (1 - tj) + self.R[k1, j1] * tj
        return r0 * (1 - ty) + r1 * ty

    def ybase_at(self, u, phi):
        phi = np.mod(phi, 2 * np.pi)
        dphi = self.phib[1] - self.phib[0]
        fj = phi / dphi; j0 = np.floor(fj).astype(int) % len(self.phib); j1 = (j0 + 1) % len(self.phib); tj = fj - np.floor(fj)
        fu = np.clip(u, 0, 1) * (len(self.u) - 1); k0 = np.minimum(np.floor(fu).astype(int), len(self.u) - 2); tu = fu - k0
        a = self.yb[k0, j0] * (1 - tj) + self.yb[k0, j1] * tj
        b = self.yb[k0 + 1, j0] * (1 - tj) + self.yb[k0 + 1, j1] * tj
        return a * (1 - tu) + b * tu

    def inside(self, P, margin, fin=0.002):
        """points (N,3) in the bottle frame (axis through the origin) at least `margin` inside the shell. The radius is the
        minimum over y-fin..y+fin: the neck's support ring is a solid flange, but the single-layer shell turns it into a
        1.1 mm-tall annular pocket open to the inside -- water seeded there (12 particles at fill 0.195) is shot out through
        the shell in the first 0.3 s. Features thinner than 2 fin are not water space."""
        r = np.hypot(P[:, 0], P[:, 2]); ph = np.arctan2(P[:, 2], P[:, 0]); y = P[:, 1]
        yc = np.clip(y, self.y[0], self.y[-1])
        Rr = np.minimum.reduce([self.R_at(ph, np.clip(yc + d, self.y[0], self.y[-1])) for d in (-fin, -0.5 * fin, 0.0, 0.5 * fin, fin)])
        u = r / np.maximum(self.R_at(ph, self.y[0]), 1e-9)
        return (r < Rr - margin) & (y > self.ybase_at(u, ph) + margin) & (y < self.H)


def ring_angles(shape, y, n, phase=0.0):
    """n angles with equal arc length along the cross-section at height y, starting at phi = phase (flat centre)"""
    ph = np.radians(np.arange(0.0, 360.0, 0.25)) + phase
    r = shape.R_at(ph, np.full_like(ph, y))
    p = np.stack([r * np.cos(ph), r * np.sin(ph)], 1)
    seg = np.linalg.norm(np.diff(np.vstack([p, p[:1]]), axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = s[-1]
    t = np.arange(n) * L / n
    ph_ext = np.concatenate([ph, [ph[0] + 2 * np.pi]])
    return np.interp(t, s, ph_ext), L


def build(shape, edge, dy_ribs, rib_y=(0.015, 0.095), rib_wave=0.0055, wave_band=(0.016, 0.028, 0.066, 0.080),
          ring_band=(0.1900, 0.1960), dy_ring=0.0011, edge_ring=0.0015):
    verts, rings = [], []
    def rib_taper(y):
        a, b, c, d = wave_band
        return float(np.clip(min((y - a) / (b - a), (d - y) / (d - c)), 0.0, 1.0))
    # ---- base: centre vertex + rings of constant u
    R_rim_mean = np.mean(shape.R_at(shape.phi, np.full_like(shape.phi, shape.y[0])))
    n_b = max(4, int(round(R_rim_mean / edge)))
    verts.append((0.0, float(shape.ybase_at(np.array([0.0]), np.array([0.0]))[0]), 0.0))
    for k in range(1, n_b + 1):
        u = k / n_b
        # perimeter of this base ring ~ u * rim perimeter
        _, Lrim = ring_angles(shape, shape.y[0], 360)
        n_k = max(8, int(round(u * Lrim / edge)))
        ph, _ = ring_angles(shape, shape.y[0], n_k)
        rr = u * shape.R_at(ph, np.full_like(ph, shape.y[0]))
        yy = shape.ybase_at(np.full_like(ph, u), ph)
        ring = list(range(len(verts), len(verts) + n_k))
        verts += [(rr[j] * np.cos(ph[j]), yy[j], rr[j] * np.sin(ph[j])) for j in range(n_k)]
        rings.append(ring)
    # ---- side wall: ring heights by arc length along the mean meridian (finer in the rib band)
    ys = shape.y
    rmean = shape.R.mean(1)
    seg = np.hypot(np.diff(ys), np.diff(rmean))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    # target spacing per arc-length position
    h = np.where((ys >= rib_y[0]) & (ys <= rib_y[1]), dy_ribs, edge)
    h = np.where((ys >= ring_band[0]) & (ys <= ring_band[1]), dy_ring, h)     # neck support ring: flat bottom, 1 mm rim
    w = np.concatenate([[0.0], np.cumsum(seg / h[:-1])])   # "number of rings" coordinate
    n_side = int(round(w[-1]))
    y_rings = np.interp(np.linspace(0.0, w[-1], n_side + 1), w, ys)[1:]   # y[0] = base rim ring (already placed)
    y_rings[-1] = ys[-1]
    for y in y_rings:
        _, L = ring_angles(shape, y, 360)
        e_k = edge_ring if ring_band[0] <= y <= ring_band[1] else edge
        n_k = max(8, int(round(L / e_k)))
        n_k = int(4 * round(n_k / 4))          # quarter-symmetric (flats / corners on the same vertices)
        ph, _ = ring_angles(shape, y, n_k)
        # the wavy ribs rise 5.5 mm from the flat centres (phi = 0, 90, ..) to the corners as ~sin^2(2 phi) (measured on the
        # height map: 0.6 / 1.8 / 3.7 / 4.9 / 5.5 mm at 10 / 20 / 30 / 40 / 45 deg): warp the rings the same way so the
        # grooves run along ring lines instead of stair-stepping across them
        yy = y + rib_wave * np.sin(2 * ph) ** 2 * rib_taper(y)
        rr = shape.R_at(ph, yy)
        ring = list(range(len(verts), len(verts) + n_k))
        verts += [(rr[j] * np.cos(ph[j]), yy[j], rr[j] * np.sin(ph[j])) for j in range(n_k)]
        rings.append(ring)
    V = np.array(verts)
    faces = []
    for j in range(len(rings[0])):                      # base fan: outward = -y
        a, b = rings[0][j], rings[0][(j + 1) % len(rings[0])]
        faces.append((0, a, b))
    def ang(i):
        return np.mod(np.arctan2(V[i, 2], V[i, 0]), 2 * np.pi)
    for k in range(len(rings) - 1):
        lo, hi = rings[k], rings[k + 1]
        nl, nh = len(lo), len(hi)
        if nl == nh:                                    # equal counts: alternating diagonals (no helical bias)
            for j in range(nl):
                a, b, c, d = lo[j], lo[(j + 1) % nl], hi[(j + 1) % nh], hi[j]
                if (j + k) % 2 == 0:
                    faces += [(a, d, b), (b, d, c)]
                else:
                    faces += [(a, d, c), (a, c, b)]
            continue
        # different counts: walk both rings by angle (both start near phi = 0)
        al = np.array([ang(i) for i in lo]); ah = np.array([ang(i) for i in hi])
        ol = np.argsort(al); oh = np.argsort(ah)
        lo = [lo[i] for i in ol]; hi = [hi[i] for i in oh]
        al = al[ol]; ah = ah[oh]
        i = j = 0
        while i < nl or j < nh:
            ai = al[(i + 1) % nl] + (2 * np.pi if i + 1 >= nl else 0.0) if i < nl else 9e9
            aj = ah[(j + 1) % nh] + (2 * np.pi if j + 1 >= nh else 0.0) if j < nh else 9e9
            if ai <= aj and i < nl:     # (hi - lo) x (tangent) = outward, same as the equal-count strips
                faces.append((lo[i % nl], hi[j % nh], lo[(i + 1) % nl])); i += 1
            else:
                faces.append((lo[i % nl], hi[j % nh], hi[(j + 1) % nh])); j += 1
    F = np.array(faces, np.int64)
    # orientation: outward normals -> signed volume of the shell closed at the mouth by a fan must be > 0
    top = rings[-1]
    cap = [(len(V), top[(j + 1) % len(top)], top[j]) for j in range(len(top))]
    Vc = np.vstack([V, [[0.0, V[top, 1].mean(), 0.0]]])
    Fc = np.vstack([F, np.array(cap)])
    vol = np.einsum("ij,ij->i", Vc[Fc[:, 0]], np.cross(Vc[Fc[:, 1]], Vc[Fc[:, 2]])).sum() / 6.0
    if vol < 0:
        F = F[:, [0, 2, 1]]; vol = -vol
    return V, F, rings, abs(vol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fbx")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=1.22, help="1.22: neck -> standard 28 mm PCO 1881 finish, 500 ml bottle")
    ap.add_argument("--edge", type=float, default=0.0025)
    ap.add_argument("--edge-ribs", type=float, default=0.0020, help="ring spacing in the rib band")
    ap.add_argument("--rib-wave", type=float, default=0.0055, help="height rise of the wavy ribs from flat centre to corner (scaled, m); 0 = flat rings")
    ap.add_argument("--drop-support-ring", action="store_true", help="clamp the neck to 11.1 mm (FBX scale) above 150 mm: removes the "
                    "support ring from the SIMULATION shell (off by default). A single-layer shell turns the solid ring into a 1.1 mm "
                    "annular pocket open to the inside; water pushed up the neck (fast / full squeezes) gets into it and stalls the "
                    "solver. The renderer draws the ring back, moving rigidly with the neck (render_mitsuba BOTTLE_RING).")
    a = ap.parse_args()
    V0, F0 = body_triangles(a.fbx)
    hm = sample(V0, F0)
    shape = Shape(hm, a.scale, *((0.150, 0.01110) if a.drop_support_ring else (None, None)))
    ring_kw = dict(ring_band=(1.0, 1.0)) if a.drop_support_ring else {}     # no ring: no local refinement either
    V, F, rings, vol = build(shape, a.edge, a.edge_ribs, rib_y=(0.015, 0.115), rib_wave=a.rib_wave, **ring_kw)
    os.makedirs(osp.dirname(osp.abspath(a.out)), exist_ok=True)
    e = np.concatenate([np.linalg.norm(V[F[:, i]] - V[F[:, (i + 1) % 3]], axis=1) for i in range(3)])
    with open(a.out, "w") as f:
        f.write(f"# PET bottle shell from {osp.basename(a.fbx)} x{a.scale}: {len(V)} verts, {len(F)} faces, edge {e.min()*1e3:.2f}-{e.max()*1e3:.2f} mm, "
                f"{vol*1e6:.0f} ml to the brim, height {V[:,1].max()*1e3:.1f} mm\n")
        f.write("".join(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n" for p in V))
        f.write("".join(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n" for t in F))
    np.savez(osp.splitext(a.out)[0] + ".npz", phi=shape.phi, y=shape.y, R=shape.R, u=shape.u, phib=shape.phib, ybase=shape.yb,
             height=shape.H, scale=a.scale, volume=vol)
    flats = shape.R_at(np.array([0.0, np.pi / 2]), np.array([0.08, 0.08]))
    diag = shape.R_at(np.array([np.pi / 4]), np.array([0.08]))
    print(f"{a.out}: {len(V)} verts, {len(F)} faces, {len(rings)} rings ({min(map(len, rings))}..{max(map(len, rings))}), "
          f"edge {e.min()*1e3:.2f}/{np.median(e)*1e3:.2f}/{e.max()*1e3:.2f} mm (min/med/max)")
    print(f"  height {V[:,1].max()*1e3:.1f} mm, flats {2*flats.mean()*1e3:.1f} mm, diagonal {2*diag[0]*1e3:.1f} mm, "
          f"neck {2*shape.R[-1].mean()*1e3:.1f} mm, base push-up {V[0,1]*1e3:.1f} mm, volume to the brim {vol*1e6:.0f} ml")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Deform-pieces render data for a cucumber_cut run (the way Houdini's "MPM Deform Pieces" renders MPM fracture): the
cucumber's own surface mesh, cut along the crack surfaces the simulation actually produced, and moved with the particles.

Input: the small npz written on the pod by pieces_extract.py (rest particles, final piece of each particle = connected
unbroken group at the end, knife timeline, the rest cucumber mesh).  Output: an npz for render_cucumber.py --pieces:
  - the rest mesh, Loop-subdivided and cut into the pieces along planes fitted (rest frame) between neighbouring pieces;
    each cut face (cap) is a separate, finely triangulated mesh with UVs onto a procedural cucumber cross-section texture
    (skin rim, green rind, pale flesh, three-lobed seed core with seeds), the same texture on both faces of one cut;
  - for every render vertex, the moving-least-squares (affine) interpolation weights from the nearby unbroken particles of
    its own piece (and, near a cut, of the union of both pieces, used until the crack front has passed that vertex), so a
    frame is one sparse product: x = S @ P.  Affine fields are reproduced exactly.
Needs trimesh (+ manifold3d or mapbox_earcut for the cap triangulation), scipy, PIL; runs locally, not on the pod.

    python prep_pieces.py pieces_<run>.npz render_pieces_<run>.npz [--subdiv 3] [--cap-edge 0.0005] [--tex 1024]
"""
import argparse

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy import sparse
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.spatial import Delaunay, cKDTree
import shapely

SKIN = (0.10, 0.24, 0.07)        # sRGB, as render_cucumber.py
SKIN_LIGHT = (0.20, 0.36, 0.11)  # paler stripes
RIND = (0.40, 0.60, 0.22)        # bright green layer under the skin
FLESH = (0.80, 0.87, 0.62)       # pale flesh
CORE = (0.89, 0.91, 0.76)        # seed core (jelly)
SEED = (0.95, 0.94, 0.84)
SEED_RIM = (0.74, 0.76, 0.55)
WALL = (0.74, 0.82, 0.55)        # locule walls


def lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def smooth(e0, e1, x):
    u = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return u * u * (3 - 2 * u)


def fit_planes(x0, piece, cuts, s):
    """rest-frame plane between piece k and k+1: least-squares z = a + b x + c y through the midpoints of close pairs"""
    planes = []
    for k in range(len(cuts)):
        A = x0[piece == k]; B = x0[piece == k + 1]
        # centre the 5 s window on where the two pieces really meet, not on the planned cut: with --aim face the knife
        # follows the body, and a softer cucumber (2026-11-12 zone runs) ends up > 5 s from the plan (plane 3: 0 pairs)
        zc = cuts[k]
        if len(A) and len(B):
            d0, j0 = cKDTree(B).query(A, distance_upper_bound=3 * s)
            if np.isfinite(d0).sum() >= 50:
                zc = float(np.median(0.5 * (A[np.isfinite(d0), 2] + B[j0[np.isfinite(d0)], 2])))
        A = A[np.abs(A[:, 2] - zc) < 5 * s]; B = B[np.abs(B[:, 2] - zc) < 5 * s]
        dist, j = cKDTree(B).query(A, distance_upper_bound=3 * s)
        ok = np.isfinite(dist)
        M = 0.5 * (A[ok] + B[j[ok]])
        G = np.c_[np.ones(len(M)), M[:, 0], M[:, 1]]
        coef = np.linalg.lstsq(G, M[:, 2], rcond=None)[0]
        n = np.array([-coef[1], -coef[2], 1.0]); n /= np.linalg.norm(n)
        c = M.mean(0); c[2] = coef[0] + coef[1] * c[0] + coef[2] * c[1]
        rms = float(np.sqrt(np.mean((M[:, 2] - G @ coef) ** 2)))
        planes.append((n, c))
        print(f"  plane {k}: through z {c[2]*1e3:.2f} mm (planned {cuts[k]*1e3:.1f}), tilt {np.degrees(np.arccos(n[2])):.1f} deg, "
              f"{ok.sum()} pairs, rms {rms*1e3:.2f} mm")
    return planes


def boundary_loop(V, F, n, c):
    """the ordered boundary loop of an open (sliced) mesh that lies on the plane (n, c)"""
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    u, cnt = np.unique(E, axis=0, return_counts=True)
    B = u[cnt == 1]
    on = np.abs((V - c) @ n) < 1e-7
    B = B[on[B].all(1)]
    nxt = {}
    for a_, b_ in B:
        nxt.setdefault(a_, []).append(b_); nxt.setdefault(b_, []).append(a_)
    loops, seen = [], set()
    for start in nxt:
        if start in seen:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nb = [x for x in nxt[cur] if x != prev]
            if not nb or nb[0] == start:
                break
            prev, cur = cur, nb[0]
            if cur in seen:
                break
            loop.append(cur); seen.add(cur)
        loops.append(np.array(loop))
    return max(loops, key=len)


def cap_triangulation(B2, h):
    """conforming Delaunay of the polygon B2 (ordered, closed) with a hex grid of spacing h inside: (points, triangles);
    the first len(B2) points are B2, the next ones are boundary midpoints inserted where a segment was not a Delaunay edge"""
    poly = shapely.Polygon(B2)
    lo, hi = B2.min(0), B2.max(0)
    ys = np.arange(lo[1], hi[1], h * np.sqrt(3) / 2)
    G = np.concatenate([np.c_[np.arange(lo[0] + (i % 2) * h / 2, hi[0], h), np.full(len(np.arange(lo[0] + (i % 2) * h / 2, hi[0], h)), y)]
                        for i, y in enumerate(ys)])
    inside = shapely.contains_xy(poly, G[:, 0], G[:, 1])
    G = G[inside]
    G = G[shapely.distance(shapely.points(G), poly.exterior) > 0.6 * h]
    nb = len(B2)
    pts = [*B2]
    segs = [(i, (i + 1) % nb) for i in range(nb)]
    pts = np.concatenate([B2, G])
    extra = []
    for _ in range(12):
        tri = Delaunay(pts)
        S = tri.simplices
        E = set(map(tuple, np.sort(np.concatenate([S[:, [0, 1]], S[:, [1, 2]], S[:, [2, 0]]]), axis=1).tolist()))
        miss = [sg for sg in segs if tuple(sorted(sg)) not in E]
        if not miss:
            break
        new = []
        for a_, b_ in miss:
            q = len(pts) + len(new)
            new.append(0.5 * (pts[a_] + pts[b_]))
            segs.remove((a_, b_)); segs += [(a_, q), (q, b_)]
        pts = np.concatenate([pts, np.array(new)])
    cen = pts[S].mean(1)
    keep = shapely.contains_xy(poly, cen[:, 0], cen[:, 1])
    return pts, S[keep], len(miss)


def plane_frame(n):
    e1 = np.array([1.0, 0.0, 0.0]) - n[0] * n; e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def section_texture(B2, half, res, theta0, rng):
    """procedural cucumber cross-section (linear RGB, res x res) of the polygon B2 (plane coordinates, m, centred) over the
    square [-half, half]^2; image row i is v = i / res and column j is u = j / res, as Mitsuba's bitmap texture reads it
    (checked with an AOV render: row 0, column 0 lands at uv (0, 0))"""
    px = 2 * half / res
    img = Image.new("L", (res, res), 0)
    uv = (B2 + half) / px
    ImageDraw.Draw(img).polygon([(float(u), float(v)) for u, v in uv], fill=255)
    mask = np.asarray(img) > 0
    d = distance_transform_edt(mask) * px * 1e3                     # mm to the skin
    yy, xx = np.mgrid[0:res, 0:res]
    cx, cy = xx[mask].mean(), yy[mask].mean()
    X = (xx - cx) * px * 1e3; Y = (yy - cy) * px * 1e3              # mm from the centre
    rho = np.hypot(X, Y); th = np.arctan2(Y, X)
    R = d.max()
    col = np.zeros((res, res, 3))
    # skin rim -> rind -> flesh
    a = smooth(0.45, 0.9, d)
    b = smooth(1.0, 4.5, d)
    col[:] = lin(SKIN)
    col = col * (1 - a[..., None]) + a[..., None] * ((1 - b[..., None]) * lin(RIND) + b[..., None] * lin(FLESH))
    # vascular ring: faint greener dots about 4 mm in
    ring = np.zeros_like(d)
    for j in range(14):
        ang = theta0 + 2 * np.pi * j / 14 + rng.normal(0, 0.06)
        r0 = R - 4.2
        ring = np.maximum(ring, np.exp(-((X - r0 * np.cos(ang)) ** 2 + (Y - r0 * np.sin(ang)) ** 2) / (2 * 0.45 ** 2)))
    col = col * (1 - 0.35 * ring[..., None]) + 0.35 * ring[..., None] * lin(RIND)
    # three-lobed seed core
    rc = 0.50 * R * (1 + 0.10 * np.cos(3 * (th - theta0)))
    core = 1 - smooth(rc - 0.8, rc + 0.8, rho)
    col = col * (1 - core[..., None]) + core[..., None] * lin(CORE)
    # locule walls: three thin lines from the centre to the lobes
    for j in range(3):
        ang = theta0 + 2 * np.pi * j / 3
        along = X * np.cos(ang) + Y * np.sin(ang); across = -X * np.sin(ang) + Y * np.cos(ang)
        w = np.exp(-(across / 0.35) ** 2) * smooth(-0.5, 0.5, along) * (1 - smooth(0.8 * rc, rc, rho))
        col = col * (1 - 0.5 * w[..., None]) + 0.5 * w[..., None] * lin(WALL)
    # seeds: two per locule, oval, radial
    for j in range(3):
        for sgn in (-1, 1):
            ang = theta0 + np.pi / 3 + 2 * np.pi * j / 3 + sgn * 0.33
            rr = 0.62 * 0.50 * R
            sx, sy = rr * np.cos(ang), rr * np.sin(ang)
            u = (X - sx) * np.cos(ang) + (Y - sy) * np.sin(ang); v = -(X - sx) * np.sin(ang) + (Y - sy) * np.cos(ang)
            e = np.sqrt((u / 2.0) ** 2 + (v / 0.85) ** 2)
            seed = 1 - smooth(0.85, 1.0, e)
            rim = smooth(0.7, 0.9, e) * (1 - smooth(0.95, 1.1, e))
            col = col * (1 - seed[..., None]) + seed[..., None] * lin(SEED)
            col = col * (1 - 0.5 * rim[..., None]) + 0.5 * rim[..., None] * lin(SEED_RIM)
    # gentle low-frequency variation
    noise = gaussian_filter(rng.normal(0, 1, (res, res)), res / 40)
    noise /= max(np.abs(noise).max(), 1e-9)
    col *= (1 + 0.04 * noise)[..., None]
    col[~mask] = lin(FLESH)                                           # outside: never sampled
    return np.clip(col, 0, 1).astype(np.float32)


def mls_weights(Xv, Xp, tree, ids, s, k=64, hmin=5.0):
    """rows of S: x(Xv) = sum_i c_i P[ids[i]] (affine MLS from the k nearest particles of the given set, kernel radius at
    least hmin particle spacings = hmin / 2 grid cells: the MPM field is resolved on the grid, so the particle-scale jitter
    below it is quadrature noise, not shape)"""
    dist, j = tree.query(Xv, k=k)
    h = np.maximum(hmin * s, 1.1 * dist[:, -1])
    w = np.clip(1 - (dist / h[:, None]) ** 2, 0, None) ** 3 + 1e-12
    w /= w.sum(1, keepdims=True)
    Xr = Xp[j] - Xv[:, None, :]                                       # rest offsets from the vertex
    mX = np.einsum("vk,vki->vi", w, Xr)
    dX = Xr - mX[:, None, :]
    C = np.einsum("vk,vki,vkj->vij", w, dX, dX)
    C += (1e-4 * np.trace(C, axis1=1, axis2=2)[:, None, None] + (0.01 * s) ** 2) * np.eye(3)
    Ci = np.linalg.inv(C)
    c = w * (1 - np.einsum("vki,vij,vj->vk", dX, Ci, mX))
    return ids[j], c


def cucumber_axis(V):
    """centre and unit long axis of the rest cucumber mesh (principal axis), and the perpendicular frame for the angle"""
    c = V.mean(0)
    _, _, W = np.linalg.svd(V - c, full_matrices=False)
    ax = W[0] * np.sign(W[0][2])
    e1 = np.array([0.0, 1.0, 0.0]) - ax[1] * ax; e1 /= np.linalg.norm(e1)
    return c, ax, e1, np.cross(ax, e1)


def skin_uv(V, frame, span):
    """cylindrical uv: u = angle about the long axis / 2 pi, v = position along it over the cucumber's length"""
    c, ax, e1, e2 = frame
    r = V - c
    u = (np.arctan2(r @ e2, r @ e1) + np.pi) / (2 * np.pi)
    v = (r @ ax - span[0]) / (span[1] - span[0])
    return np.c_[u, v]


def seam_split(V, F, uv):
    """duplicate the low-u vertices of faces that straddle the u = 0 / 1 seam (u + 1 with a repeating texture)"""
    du = uv[F][:, :, 0]
    bad = np.nonzero(du.max(1) - du.min(1) > 0.5)[0]
    if not len(bad):
        return V, F, uv, np.arange(len(V))
    F = F.copy()
    src = list(range(len(V)))
    extra = {}
    for f in bad:
        for a_ in range(3):
            vtx = F[f, a_]
            if uv[vtx, 0] < 0.5:
                if vtx not in extra:
                    extra[vtx] = len(src); src.append(vtx)
                F[f, a_] = extra[vtx]
    src = np.array(src)
    uv2 = uv[src].copy()
    uv2[len(V):, 0] += 1.0
    return V[src], F, uv2, src


def skin_texture(circ, length, rng, W=1024, H=2048):
    """procedural cucumber skin over u (around, circ m) x v (along, length m): base colour (linear RGB) and a bump height map
    (0..1): glossy dark green, faint lighter stripes, pale speckled warts (raised), a paler blossom end"""
    mu, mv = circ * 1e3 / W, length * 1e3 / H                       # mm per pixel
    vv, uu = np.mgrid[0:H, 0:W]
    U = uu * mu; Vv = vv * mv                                         # mm
    col = np.zeros((H, W, 3)); col[:] = lin(SKIN)
    low = gaussian_filter(rng.normal(0, 1, (H // 8, W // 8)), 3, mode="wrap")
    low = np.kron(low / max(np.abs(low).max(), 1e-9), np.ones((8, 8)))[:H, :W]
    stripe = 0.5 + 0.5 * np.sin(2 * np.pi * (10 * uu / W) + 1.5 * low)
    stripe = smooth(0.62, 0.95, stripe)
    col = col * (1 - 0.30 * stripe[..., None]) + 0.30 * stripe[..., None] * lin(SKIN_LIGHT)
    col *= (1 + 0.10 * low)[..., None]
    end = smooth(0.80, 1.0, 1 - vv / H)                               # paler towards one end
    col = col * (1 - 0.25 * end[..., None]) + 0.25 * end[..., None] * lin((0.32, 0.45, 0.16))
    height = np.zeros((H, W))
    n_w = int(circ * length * 1e6 / 30.0)                             # one wart per ~30 mm^2
    cu, cv = rng.uniform(0, circ * 1e3, n_w), rng.uniform(0, length * 1e3, n_w)
    rad = rng.uniform(0.5, 1.0, n_w)
    for x_, y_, r_ in zip(cu, cv, rad):
        i0, i1 = int((y_ - 3 * r_) / mv), int((y_ + 3 * r_) / mv) + 1
        j0, j1 = int((x_ - 3 * r_) / mu), int((x_ + 3 * r_) / mu) + 1
        ii = np.arange(max(i0, 0), min(i1, H))
        jj = np.arange(j0, j1) % W
        if not len(ii):
            continue
        dy = (ii[:, None] * mv - y_); dx = ((jj[None, :] * mu - x_ + circ * 5e2) % (circ * 1e3)) - circ * 5e2
        g = np.exp(-(dx ** 2 + dy ** 2) / (2 * (0.55 * r_) ** 2))
        height[np.ix_(ii, jj)] = np.maximum(height[np.ix_(ii, jj)], g)
        pale = (np.exp(-(dx ** 2 + dy ** 2) / (2 * (0.35 * r_) ** 2)))[..., None]
        sub = col[np.ix_(ii, jj)]
        col[np.ix_(ii, jj)] = sub * (1 - 0.55 * pale) + 0.55 * pale * lin((0.62, 0.70, 0.36))
    return np.clip(col, 0, 1).astype(np.float32), height.astype(np.float32)


def skin_colour(V, axis_c):
    """dark green with faint longitudinal stripes and a low-frequency variation (linear RGB per vertex)"""
    th = np.arctan2(V[:, 1] - axis_c[1], V[:, 0] - axis_c[0])
    stripe = smooth(0.55, 0.95, 0.5 + 0.5 * np.sin(9 * th + 18 * V[:, 2]))
    var = 0.5 + 0.5 * np.sin(31 * V[:, 2] + 3 * th)
    rgb = lin(SKIN) * (1 - 0.35 * stripe[:, None]) + 0.35 * stripe[:, None] * lin(SKIN_LIGHT)
    return rgb * (0.94 + 0.12 * var)[:, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--subdiv", type=int, default=3)
    ap.add_argument("--cap-h", type=float, default=0.0006, help="cap triangulation spacing (m)")
    ap.add_argument("--tex", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    d = np.load(a.inp)
    x0, piece, s = d["x0"].astype(np.float64), d["piece"].astype(int), float(d["spacing"])
    cuts = np.asarray(d["cuts"], float); K = len(cuts) + 1
    m = trimesh.Trimesh(d["cucumber_V"].astype(np.float64), d["cucumber_F"].astype(np.int64), process=True)
    v0 = m.volume
    m = m.subdivide_loop(iterations=a.subdiv)
    print(f"mesh: subdivided x{a.subdiv} -> {len(m.vertices)} v / {len(m.faces)} f, volume {m.volume/v0:.4f} of the original")
    planes = fit_planes(x0, piece, cuts, s)
    zmax = float(d["zmax"]) if "zmax" in d.files else 1.0
    if zmax < np.asarray(m.vertices)[:, 2].max():   # thin-end section (cucumber_cut.py --zmax): a flat end face that never opens
        planes.append((np.array([0.0, 0.0, 1.0]), np.array([x0[:, 0].mean(), x0[:, 1].mean(), x0[:, 2].max() + 0.5 * s])))
        print(f"  end face at z {planes[-1][1][2]*1e3:.1f} mm (section up to --zmax {zmax*1e3:.1f} mm)")
    NPL = len(planes)
    sec_c = x0[np.abs(x0[:, 2] - cuts.mean()) < 0.02].mean(0)       # cucumber axis near the cuts (x, y)
    theta0 = rng.uniform(0, 2 * np.pi)
    origin = x0.mean(0)
    frame = cucumber_axis(np.asarray(m.vertices))
    along = (np.asarray(m.vertices) - frame[0]) @ frame[1]
    span = (along.min(), along.max())
    rr = np.linalg.norm((np.asarray(m.vertices) - frame[0]) - np.outer(along, frame[1]), axis=1)
    skin_tex, skin_bump = skin_texture(2 * np.pi * np.percentile(rr, 90), span[1] - span[0], rng)
    print(f"  skin texture: {skin_tex.shape[1]} x {skin_tex.shape[0]} over {2*np.pi*np.percentile(rr, 90)*1e3:.0f} x {(span[1]-span[0])*1e3:.0f} mm")
    V_all, vert_piece, skin_rgb, skins, caps, capUV, loops2d, skin_uvs = [], [], [], [], [], [], {}, []
    nv = 0
    for k in range(K):
        mk = m
        if k > 0:
            mk = trimesh.intersections.slice_mesh_plane(mk, plane_normal=planes[k - 1][0], plane_origin=planes[k - 1][1], cap=False)
        if k < NPL:
            mk = trimesh.intersections.slice_mesh_plane(mk, plane_normal=-planes[k][0], plane_origin=planes[k][1], cap=False)
        mk = mk.copy(); mk.merge_vertices()                         # the slice leaves duplicated vertices along the cut
        V = np.asarray(mk.vertices, float); F = np.asarray(mk.faces, np.int64)
        loops = {j: boundary_loop(V, F, planes[j][0], planes[j][1]) for j in (k - 1, k) if 0 <= j < NPL}
        Vl = V
        V, F, uvs, _ = seam_split(V, F, skin_uv(V, frame, span))
        V_all.append(V); skins.append((k, F + nv)); vert_piece.append(np.full(len(V), k)); skin_rgb.append((nv, skin_colour(V, sec_c)))
        skin_uvs.append((nv, uvs))
        nv += len(V)
        msg = f"  piece {k}: skin {len(V)} v / {len(F)} f"
        for j, sign in ((k - 1, -1.0), (k, 1.0)):                    # plane below (outward -n) / above (outward +n)
            if not 0 <= j < NPL:
                continue
            n, c = planes[j]; e1, e2 = plane_frame(n)
            loop = loops[j]
            B3 = Vl[loop]
            B2 = np.c_[(B3 - c) @ e1, (B3 - c) @ e2]
            if j not in loops2d:
                cen = B2.mean(0)
                loops2d[j] = (cen, np.abs(B2 - cen).max() + 0.0015, B2 - cen)
            cen, half, _ = loops2d[j]
            P2, T, miss = cap_triangulation(B2, a.cap_h)
            P3 = c + P2[:, :1] * e1 + P2[:, 1:] * e2
            P3[:len(loop)] = B3
            nrm = np.cross(P3[T[:, 1]] - P3[T[:, 0]], P3[T[:, 2]] - P3[T[:, 0]])
            flip = (nrm @ (sign * n)) < 0
            T[flip] = T[flip][:, ::-1]
            UV = ((P2 - cen) + half) / (2 * half)
            V_all.append(P3); caps.append((k, j, T + nv, nv)); capUV.append(UV.astype(np.float32)); vert_piece.append(np.full(len(P3), k))
            nv += len(P3)
            msg += f" | cap on plane {j}: loop {len(loop)}, {len(P3)} v / {len(T)} f (unresolved segments {miss})"
        print(msg)
    tex = []
    for j in range(NPL):
        cen, half, B2c = loops2d[j]
        tex.append(section_texture(B2c, half, a.tex, theta0, rng))
        print(f"  texture {j}: {a.tex}^2 over {2*half*1e3:.1f} mm")
    V_all = np.concatenate(V_all); vert_piece = np.concatenate(vert_piece)
    is_cap = np.zeros(len(V_all), bool)
    for (k, j, T, v_0) in caps:
        is_cap[np.unique(T)] = True
    # which cut plane a vertex sits next to (the union weights are used there until the crack front has passed it)
    vplane = -np.ones(len(V_all), int); vdist = np.full(len(V_all), np.inf)
    for j, (n, c) in enumerate(planes[:K - 1]):       # not the end face: it never opens
        dd = np.abs((V_all - c) @ n)
        near = (dd < 3 * s) & ((vert_piece == j) | (vert_piece == j + 1)) & (dd < vdist)
        vplane[near] = j; vdist[near] = dd[near]
    KN = 64
    own_i, own_c = np.zeros((len(V_all), KN), np.int64), np.zeros((len(V_all), KN))
    for k in range(K):
        ids = np.nonzero(piece == k)[0]
        sel = np.nonzero(vert_piece == k)[0]
        own_i[sel], own_c[sel] = mls_weights(V_all[sel], x0[ids], cKDTree(x0[ids]), ids, s, KN)
    uni_i, uni_c = own_i.copy(), own_c.copy()
    for j in range(K - 1):
        ids = np.nonzero((piece == j) | (piece == j + 1))[0]
        sel = np.nonzero(vplane == j)[0]
        if len(sel):
            uni_i[sel], uni_c[sel] = mls_weights(V_all[sel], x0[ids], cKDTree(x0[ids]), ids, s, KN)
    N = len(x0)

    def csr(I, C):
        r = np.repeat(np.arange(len(I)), I.shape[1])
        return sparse.csr_matrix((C.ravel().astype(np.float32), (r, I.ravel())), shape=(len(I), N))
    So, Su = csr(own_i, own_c), csr(uni_i, uni_c)
    for name, S_ in (("own", So), ("union", Su)):
        e = np.linalg.norm(S_ @ (x0 - origin) + origin - V_all, axis=1)
        print(f"  rest reproduction ({name}): skin p99 {np.percentile(e[~is_cap], 99)*1e6:.1f} max {e[~is_cap].max()*1e6:.1f} um; "
              f"caps p99 {np.percentile(e[is_cap], 99)*1e6:.1f} max {e[is_cap].max()*1e6:.1f} um")
    print(f"vertices {len(V_all)} (caps {int(is_cap.sum())}), near a cut {int((vplane >= 0).sum())}")
    rgb_all = np.zeros((len(V_all), 3))                               # per global vertex (caps are textured instead)
    for v_0, rgb in skin_rgb:
        rgb_all[v_0:v_0 + len(rgb)] = rgb
    uv_all = np.zeros((len(V_all), 2), np.float32)                    # skin uv per global vertex
    for v_0, uvs in skin_uvs:
        uv_all[v_0:v_0 + len(uvs)] = uvs
    out = dict(V=V_all.astype(np.float64), origin=origin, vert_piece=vert_piece.astype(np.int8), vplane=vplane.astype(np.int8),
               skin_rgb=rgb_all.astype(np.float32), skin_uv=uv_all,
               skin_tex=(np.clip(skin_tex, 0, 1) * 65535).astype(np.uint16), skin_bump=(np.clip(skin_bump, 0, 1) * 65535).astype(np.uint16),
               So_data=So.data, So_indices=So.indices.astype(np.int32), So_indptr=So.indptr.astype(np.int64),
               Su_data=Su.data, Su_indices=Su.indices.astype(np.int32), Su_indptr=Su.indptr.astype(np.int64), N=N,
               n_skin=len(skins), n_cap=len(caps), n_plane=NPL, planes=np.array([np.r_[n, c] for n, c in planes]),
               lead=float(d["lead"]), segment=d["segment"], t=d["t"], edge=d["edge"], board_top=float(d["board_top"]))
    for i, (k, F) in enumerate(skins):
        out[f"skin{i}_F"] = F.astype(np.int32); out[f"skin{i}_piece"] = k
    for i, ((k, j, T, v_0), UV) in enumerate(zip(caps, capUV)):
        out[f"cap{i}_F"] = T.astype(np.int32); out[f"cap{i}_piece"] = k; out[f"cap{i}_plane"] = j
        out[f"cap{i}_UV"] = UV; out[f"cap{i}_v0"] = v_0
    for j, T in enumerate(tex):
        out[f"tex{j}"] = (np.clip(T, 0, 1) * 65535).astype(np.uint16)
    np.savez_compressed(a.out, **out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

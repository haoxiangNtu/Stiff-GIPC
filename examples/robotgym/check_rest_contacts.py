#!/usr/bin/env python3
"""Rest-state IPC proximity check of a shell / cloth mesh: the smallest vertex-triangle and edge-edge distances between
NON-incident primitives (the pairs the barrier would see at rest). Anything below the scene's dhat starts with an active
barrier (spurious forces + a stiff first Newton solve).

    python examples/robotgym/check_rest_contacts.py mesh.obj [--dhat 5e-4] [--scale 1.0]
"""
import argparse

import numpy as np
from scipy.spatial import cKDTree


def load_obj(p):
    V, F = [], []
    for line in open(p):
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            F.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    return np.array(V), np.array(F, np.int64)


def point_tri_dist(P, A, B, C):
    """vectorised point-triangle distance (Ericson, Real-Time Collision Detection 5.1.5)"""
    ab, ac, ap = B - A, C - A, P - A
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    bp = P - B; d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    cp = P - C; d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    va = d3 * d6 - d5 * d4; vb = d5 * d2 - d1 * d6; vc = d1 * d4 - d3 * d2
    Q = np.empty_like(P)
    denom = va + vb + vc
    with np.errstate(divide="ignore", invalid="ignore"):
        v = vb / denom; w = vc / denom
    Q[:] = A + ab * v[:, None] + ac * w[:, None]
    m = (d1 <= 0) & (d2 <= 0); Q[m] = A[m]
    m = (d3 >= 0) & (d4 <= d3); Q[m] = B[m]
    m = (d6 >= 0) & (d5 <= d6); Q[m] = C[m]
    with np.errstate(divide="ignore", invalid="ignore"):
        m = (vc <= 0) & (d1 >= 0) & (d3 <= 0); t = d1 / (d1 - d3); Q[m] = (A + ab * t[:, None])[m]
        m = (vb <= 0) & (d2 >= 0) & (d6 <= 0); t = d2 / (d2 - d6); Q[m] = (A + ac * t[:, None])[m]
        m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0); t = (d4 - d3) / ((d4 - d3) + (d5 - d6)); Q[m] = (B + (C - B) * t[:, None])[m]
    return np.linalg.norm(P - Q, axis=1)


def seg_seg_dist(P1, Q1, P2, Q2):
    d1, d2, r = Q1 - P1, Q2 - P2, P1 - P2
    a = np.einsum("ij,ij->i", d1, d1); e = np.einsum("ij,ij->i", d2, d2); f = np.einsum("ij,ij->i", d2, r)
    c = np.einsum("ij,ij->i", d1, r); b = np.einsum("ij,ij->i", d1, d2); den = a * e - b * b
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(den > 1e-30, np.clip((b * f - c * e) / den, 0, 1), 0.0)
        t = (b * s + f) / e
        s = np.where(t < 0, np.clip(-c / a, 0, 1), np.where(t > 1, np.clip((b - c) / a, 0, 1), s))
        t = np.clip(t, 0, 1)
    return np.linalg.norm((P1 + d1 * s[:, None]) - (P2 + d2 * t[:, None]), axis=1)


def check(V, F, dhat, radius=None):
    radius = radius or max(4 * dhat, 0.003)
    # vertex - triangle
    cen = V[F].mean(1); trad = np.linalg.norm(V[F] - cen[:, None], axis=2).max(1)
    tree = cKDTree(cen)
    vi, ti = [], []
    for i, nb in enumerate(tree.query_ball_point(V, radius + trad.max())):
        nb = np.array(nb, np.int64)
        nb = nb[~np.any(F[nb] == i, axis=1)]
        vi.append(np.full(len(nb), i)); ti.append(nb)
    vi, ti = np.concatenate(vi), np.concatenate(ti)
    dvt = point_tri_dist(V[vi], V[F[ti, 0]], V[F[ti, 1]], V[F[ti, 2]])
    # edge - edge
    E = np.unique(np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1), axis=0)
    ec = V[E].mean(1); erad = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1).max() / 2
    et = cKDTree(ec)
    pairs = et.query_pairs(radius + 2 * erad, output_type="ndarray")
    shared = (E[pairs[:, 0], 0] == E[pairs[:, 1], 0]) | (E[pairs[:, 0], 0] == E[pairs[:, 1], 1]) | \
             (E[pairs[:, 0], 1] == E[pairs[:, 1], 0]) | (E[pairs[:, 0], 1] == E[pairs[:, 1], 1])
    pairs = pairs[~shared]
    dee = seg_seg_dist(V[E[pairs[:, 0], 0]], V[E[pairs[:, 0], 1]], V[E[pairs[:, 1], 0]], V[E[pairs[:, 1], 1]])
    return dvt, vi, ti, dee, E, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("obj")
    ap.add_argument("--dhat", type=float, default=5e-4)
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    V, F = load_obj(a.obj)
    V = V * a.scale
    dvt, vi, ti, dee, E, pairs = check(V, F, a.dhat)
    nvt, nee = int((dvt < a.dhat).sum()), int((dee < a.dhat).sum())
    print(f"{a.obj}: {len(V)} verts {len(F)} faces; non-incident vertex-triangle min {dvt.min()*1e3:.3f} mm ({nvt} pairs < dhat {a.dhat*1e3:.2f} mm), "
          f"edge-edge min {dee.min()*1e3:.3f} mm ({nee} pairs < dhat)")
    if nvt:
        bad = np.argsort(dvt)[:5]
        for b in bad:
            print(f"   vertex {vi[b]} at {np.round(V[vi[b]]*1e3, 2).tolist()} mm  -> triangle {ti[b]}  d {dvt[b]*1e3:.3f} mm")
    return nvt + nee


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)

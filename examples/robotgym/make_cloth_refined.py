#!/usr/bin/env python3
"""Locally refined cloth: red-green refinement of a grid .obj (make_cloth_grid.py) around given points, two levels
(h -> h/2 within --r1, -> h/4 within --r2). Red splits keep the right-isosceles grid triangles similar; green
bisections close the hanging midpoints at the outer rim. For the tactile bed sheet: the 15 mm grid puts only two cloth
vertices under a 16 x 14 mm gel pad, so its indentation image shows the mesh, not the cloth.
    python examples/robotgym/make_cloth_refined.py Assets/robotgym/sheet_100x60_15mm.obj --at=0.462,-0.278 --at=-0.462,-0.278 \\
        --out Assets/robotgym/sheet_100x60_15mm_pinch4.obj"""
import argparse
import numpy as np


def read_obj(p):
    V, F = [], []
    for ln in open(p):
        if ln.startswith("v "):
            V.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith("f "):
            F.append([int(x.split("/")[0]) - 1 for x in ln.split()[1:4]])
    return np.array(V), np.array(F, int)


def refine(V, F, red, keep_level):
    """One red-green pass. red: faces to split 1->4 (closure: any face with >= 2 split edges is split too).
    keep_level: per-face refinement level (green children inherit -1 so they are never refined again)."""
    red = red.copy()
    E = lambda f: [tuple(sorted((F[f, i], F[f, (i + 1) % 3]))) for i in range(3)]
    while True:
        split = {e for f in np.where(red)[0] for e in E(f)}
        grow = [f for f in np.where(~red)[0] if sum(e in split for e in E(f)) >= 2]
        if not grow:
            break
        red[grow] = True
    V = list(map(tuple, V)); mid = {}
    for e in sorted(split):
        mid[e] = len(V); V.append(tuple((np.array(V[e[0]]) + np.array(V[e[1]])) / 2))
    NF, NL = [], []
    for f in range(len(F)):
        a, b, c = F[f]
        if red[f]:
            ab, bc, ca = mid[tuple(sorted((a, b)))], mid[tuple(sorted((b, c)))], mid[tuple(sorted((c, a)))]
            NF += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]; NL += [keep_level[f] + 1] * 4
            continue
        es = [(a, b, c), (b, c, a), (c, a, b)]
        hit = [(p, q, r) for p, q, r in es if tuple(sorted((p, q))) in split]
        if hit:                                    # exactly one hanging midpoint (closure guarantees it): bisect
            p, q, r = hit[0]; m = mid[tuple(sorted((p, q)))]
            NF += [(p, m, r), (m, q, r)]; NL += [-1, -1]
        else:
            NF.append((a, b, c)); NL.append(keep_level[f])
    return np.array(V), np.array(NF, int), np.array(NL, int)


def min_angles(V, F):
    P = V[F][:, :, [0, 2]]
    ang = []
    for i in range(3):
        u = P[:, (i + 1) % 3] - P[:, i]; w = P[:, (i + 2) % 3] - P[:, i]
        ang.append(np.degrees(np.arccos(np.clip((u * w).sum(1) / np.linalg.norm(u, axis=1) / np.linalg.norm(w, axis=1), -1, 1))))
    return np.min(ang, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("--out", required=True)
    ap.add_argument("--at", action="append", required=True, help="x,z refinement centre (cloth-local metres)")
    ap.add_argument("--r1", type=float, default=0.050); ap.add_argument("--r2", type=float, default=0.022)
    a = ap.parse_args()
    C = np.array([[float(v) for v in s.split(",")] for s in a.at])
    V, F = read_obj(a.src); n0 = len(V)
    lev = np.zeros(len(F), int)
    dist = lambda V, F: np.min(np.linalg.norm(V[F].mean(1)[:, None, [0, 2]] - C[None], axis=2), axis=1)
    V, F, lev = refine(V, F, dist(V, F) < a.r1, lev)
    V, F, lev = refine(V, F, (dist(V, F) < a.r2) & (lev == 1), lev)
    # orientation check against the source (all normals along the same side of the XZ plane)
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])[:, 1]
    assert (n > 0).all() or (n < 0).all(), "mixed orientation"
    e = np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]; L = np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1)
    with open(a.out, "w") as f:
        f.write(f"# {a.src.split('/')[-1]} refined around {C.tolist()} (h/2 within {a.r1} m, h/4 within {a.r2} m): {len(V)} vertices\n")
        for v in V:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in F:
            f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")
    print(f"{a.out}: {n0} -> {len(V)} vertices, {len(F)} faces; edge length {L.min()*1e3:.2f}..{L.max()*1e3:.2f} mm; "
          f"min angle {min_angles(V, F).min():.1f} deg; levels {np.bincount(lev + 1).tolist()} (green, 0, 1, 2)")


if __name__ == "__main__":
    main()

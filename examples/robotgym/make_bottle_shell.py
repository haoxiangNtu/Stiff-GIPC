#!/usr/bin/env python3
"""Thin-shell PET bottle as a triangle surface of revolution (.obj, metres, axis = +y, base at y = 0).
Closed base with a slight dome, cylindrical body, shoulder, neck (open mouth). Quad strips triangulated with
alternating diagonals; ring spacing ~= target edge length so the shell buckles on its own without a preferred fold.
    python examples/robotgym/make_bottle_shell.py --out Assets/robotgym/pet_bottle_5mm.obj --edge 0.005
"""
import argparse, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--edge", type=float, default=0.005, help="target edge length (m)")
ap.add_argument("--radius", type=float, default=0.032)       # body radius
ap.add_argument("--height", type=float, default=0.20)        # overall height to the mouth
ap.add_argument("--neck-r", type=float, default=0.0125)
ap.add_argument("--neck-h", type=float, default=0.025)
ap.add_argument("--shoulder-h", type=float, default=0.035)
a = ap.parse_args()
R, H, rn, hn, hs = a.radius, a.height, a.neck_r, a.neck_h, a.shoulder_h
# profile r(y): base dome (y 0 .. 0.008), body, shoulder (smooth cosine), neck
ys, rs = [], []
def add(y, r): ys.append(y); rs.append(r)
# base: a shallow dome from the centre (r=0) to the rim
n_base = max(4, int(round(R / a.edge)))
for i in range(n_base + 1):
    u = i / n_base
    add(0.006 * (1 - u) ** 2, R * u)   # concave base: the rim (u=1) is the lowest point y=0, the centre is pushed up 6 mm
y_body0 = ys[-1]
y_body1 = H - hn - hs
n_body = max(2, int(round((y_body1 - y_body0) / a.edge)))
for i in range(1, n_body + 1):
    add(y_body0 + (y_body1 - y_body0) * i / n_body, R)
n_sh = max(3, int(round(hs / a.edge)))
for i in range(1, n_sh + 1):
    u = i / n_sh
    add(y_body1 + hs * u, rn + (R - rn) * 0.5 * (1 + np.cos(np.pi * u)))
n_neck = max(2, int(round(hn / a.edge)))
for i in range(1, n_neck + 1):
    add(y_body1 + hs + hn * i / n_neck, rn)
ys, rs = np.array(ys), np.array(rs)
# rings with segment counts proportional to their radius (no slivers near the base centre); strips between rings
# of different counts are triangulated by walking both rings in angle order
verts, rings = [(0.0, ys[0], 0.0)], []
for k in range(1, len(ys)):
    n_k = max(8, int(round(2 * np.pi * rs[k] / a.edge)))
    ring = []
    for j in range(n_k):
        ang = 2 * np.pi * j / n_k
        ring.append(len(verts)); verts.append((rs[k] * np.cos(ang), ys[k], rs[k] * np.sin(ang)))
    rings.append(ring)
faces = []
for j, vid in enumerate(rings[0]):            # base fan (outward = -y: wind clockwise seen from +y)
    faces.append((0, vid, rings[0][(j + 1) % len(rings[0])]))
def strip(lo, hi):
    """triangles between two rings (lower, upper), outward normals; both ordered by angle."""
    nl, nh = len(lo), len(hi)
    i = j = 0
    out = []
    while i < nl or j < nh:
        ai = (i + 1) / nl if i < nl else 2.0     # angle (in turns) of the NEXT vertex on each ring
        aj = (j + 1) / nh if j < nh else 2.0
        if ai <= aj and i < nl:
            out.append((lo[i % nl], hi[j % nh], lo[(i + 1) % nl])); i += 1
        else:
            out.append((lo[i % nl], hi[j % nh], hi[(j + 1) % nh])); j += 1
    return out
for k in range(len(rings) - 1):
    faces += strip(rings[k], rings[k + 1])
with open(a.out, "w") as f:
    f.write(f"# PET bottle shell: R {R} m, H {H} m, neck r {rn}, {len(verts)} verts, {len(faces)} faces, edge ~{a.edge} m\n")
    for v in verts: f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
    for t in faces: f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")
V = np.array(verts); vol = np.pi * R * R * (y_body1 - y_body0)
print(f"{a.out}: {len(verts)} verts, {len(faces)} faces, rings {len(rings)} ({len(rings[0])}..{max(len(r) for r in rings)} segs); body volume ~{vol*1e6:.0f} ml, mouth r {rn*1e3:.1f} mm at y {ys[-1]*1e3:.0f} mm")

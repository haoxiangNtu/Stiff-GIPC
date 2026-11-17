#!/usr/bin/env python3
"""FBX bottle squeeze metrics from a recording: width across the flats at the finger height (dent), the same after release
(residual = plastic set), enclosed shell volume (mouth closed by a fan; what is displaced pushes water out), water inside /
above the mouth / on the table.
    python examples/robotgym/check_bottle_fbx.py out/robotgym_pod/fbx_bottle_slow/recording.npz"""
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "..", "lab_tactile"))
from lab_rec import load_recording  # noqa: E402

p = sys.argv[1]
z = load_recording(p)
meta = z["meta"]
fem = [str(n) for n in z["fem_names"]]
X = z["fem_verts"][fem.index("bottle")]; F = z["fem_faces"][fem.index("bottle")]
t, seg = z["t"], z["segment"]
gap = z["gap_arms"][:, 0] if z["gap_arms"].ndim == 2 and z["gap_arms"].shape[1] else z["gap_mm"]
L = z.get("liquid_pts")
cx, cz = 0.45, 0.0
y_sq = float(meta.get("squeeze_y", 0.098))
X0 = X[0]
band = (np.abs(X0[:, 1] - y_sq) < 0.004) & (np.abs(X0[:, 0] - cx) < 0.008)       # the flat-panel centre line under the pads
w0 = np.ptp(X0[band, 2])
# mouth ring: the top boundary loop
E = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
u, c = np.unique(E, axis=0, return_counts=True)
mouth = np.unique(u[c == 1])
dv = (0.005 / 2) ** 3 * 1e6          # ml per particle (dx 5 mm, 8 per cell)


def enclosed(V):
    top = V[mouth].mean(0)
    Vc = np.vstack([V, top]); ci = len(V)
    o = np.argsort(np.arctan2(V[mouth, 2] - top[2], V[mouth, 0] - top[0]))
    ring = mouth[o]
    cap = np.array([(ci, ring[(j + 1) % len(ring)], ring[j]) for j in range(len(ring))])
    Fc = np.vstack([F, cap])
    vol = np.einsum("ij,ij->i", Vc[Fc[:, 0]], np.cross(Vc[Fc[:, 1]], Vc[Fc[:, 2]])).sum() / 6.0
    return abs(vol)


v0 = enclosed(X0)
print(f"{p}: frames {len(t)}, t_end {t[-1]:.2f} s; rest width across the flats at y {y_sq*1e3:.0f} mm = {w0*1e3:.1f} mm; "
      f"enclosed {v0*1e6:.0f} ml; {meta.get('n_liquid', 0)} particles ({meta.get('n_liquid', 0)*dv:.0f} ml)")
last = None
for i in range(len(t)):
    if seg[i] != last or i % 25 == 0 or i == len(t) - 1:
        w = np.ptp(X[i][band, 2]); vol = enclosed(X[i])
        s = (f"t {t[i]:6.2f} {str(seg[i])[:22]:22s} gap {gap[i]:5.1f} mm  width {w*1e3:5.1f} mm (dent {(w0-w)*1e3:5.1f})  "
             f"enclosed {vol*1e6:5.0f} ml (displaced {(v0-vol)*1e6:5.1f})  top {X[i][:,1].max()*1e3:5.1f}")
        if L is not None:
            P = np.asarray(L[i]); r = np.hypot(P[:, 0] - cx, P[:, 2] - cz)
            out = r > 0.040; above = (P[:, 1] > X[i][:, 1].max() + 0.002) & ~out
            s += f" | outside {out.sum()*dv:5.1f} ml (table {np.sum(out & (P[:,1] < 0.02))*dv:5.1f}), above the mouth {above.sum()*dv:4.1f} ml, level p99 {np.percentile(P[~out,1],99)*1e3:5.1f} mm"
        print(s); last = seg[i]

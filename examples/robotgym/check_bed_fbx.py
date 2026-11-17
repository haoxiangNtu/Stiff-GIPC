#!/usr/bin/env python3
"""Bed-blanket fold metrics from a recording: per record the pinch state (cloth vertices between each hand's pads, pad gap),
the blanket hem height, and at the end the fold: share of the foot half (rest z > 0 in the blanket's grid) lying on the head
half (z < fold line, above the head half), how far the head half was dragged.
    python examples/robotgym/check_bed_fbx.py out/robotgym_pod/fbx_bed_p15_b03/recording.npz"""
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "..", "lab_tactile"))
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from lab_rec import load_recording  # noqa: E402

p = sys.argv[1]
z = load_recording(p)
fem = [str(n) for n in z["fem_names"]]
X = z["fem_verts"][fem.index("blanket")]
t, seg = z["t"], z["segment"]
gaps = z["gap_arms"]
names = [str(n) for n in z["body_names"]]
X0 = X[0]
zc0 = 0.5 * (X0[:, 2].min() + X0[:, 2].max())
foot = X0[:, 2] > zc0                      # foot half of the blanket (rest)
fold_line = zc0
print(f"{p}: frames {len(t)}, t_end {t[-1]:.2f} s; blanket {len(X0)} verts, rest z {X0[:,2].min():.3f}..{X0[:,2].max():.3f}, fold line z {fold_line:.3f}")
fingers = {}
for side in ("right", "left"):
    for f in ("fr3_leftfinger", "fr3_rightfinger"):
        i = names.index(f"{side}/{f}")
        fingers.setdefault(side, []).append(i)
last = None
for k in range(len(t)):
    if seg[k] != last or k % 25 == 0 or k == len(t) - 1:
        s = f"t {t[k]:6.2f} {str(seg[k])[:34]:34s}"
        for ai, side in enumerate(("right", "left")):
            P = []
            for i in fingers[side]:
                M = z["motion"][k][i]; P.append(z["body_rest"][i] @ M[:3, :3].T + M[:3, 3])
            c = 0.5 * (P[0].mean(0) + P[1].mean(0))
            between = np.sum(np.linalg.norm(X[k] - c, axis=1) < 0.02)
            s += f" | {side[0]}: gap {gaps[k][ai]:5.1f} mm, cloth<2cm {between:3d}, pads y {c[1]:.3f}"
        s += f" | foot-half y max {X[k][foot,1].max():.3f}, hem y {X[k][np.argmax(X0[:,2]),1]:.3f}"
        print(s); last = seg[k]
Xe = X[-1]
on_head = foot & (Xe[:, 2] < fold_line)
print(f"END: foot-half vertices over the head half (z < {fold_line:.3f}): {100*on_head.sum()/foot.sum():.0f} %; "
      f"head half dragged (median dz) {np.median(Xe[~foot,2]-X0[~foot,2])*1e3:+.0f} mm; blanket y {Xe[:,1].min():.3f}..{Xe[:,1].max():.3f}")

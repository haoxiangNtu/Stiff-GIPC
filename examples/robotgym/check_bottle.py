#!/usr/bin/env python3
"""Bottle-squeeze metrics from a recording: shell cross-section at the squeeze height (min/max radius -> dent
depth), residual dent after release (plasticity), water level / spill."""
import sys
import numpy as np

p = sys.argv[1] if len(sys.argv) > 1 else "out/robotgym/bottle/recording.npz"
d = np.load(p, allow_pickle=True)
fem = [str(x) for x in d["fem_names"]] if "fem_names" in d else ["bottle"]
t, seg = d["t"], d["segment"]; X = d[f"fem_verts_{fem.index('bottle') if 'bottle' in fem else 0}"];   # tactile runs: pads come first meta = d["meta"].item() if d["meta"].ndim == 0 else {}
gap = d["gap_arms"][:, 0] if d["gap_arms"].ndim == 2 else d["gap_mm"]
R0 = 0.032; cx, cz = 0.45, 0.0; y_sq = 0.095
liq = [d[f"liquid_pts_{i}"] for i in range(int(d["liquid_pts__n"]))] if "liquid_pts__n" in d else None
print(f"frames {len(t)}  t_end {t[-1]:.2f}  shell verts {X.shape[1]}  liquid {'yes' if liq else 'no'}")
band = np.abs(X[0][:, 1] - y_sq) < 0.012          # ring of vertices around the squeeze height (rest state)
body = (X[0][:, 1] > 0.02) & (X[0][:, 1] < 0.15)
last = None
for i in range(len(t)):
    if seg[i] != last or i == len(t) - 1 or i % 25 == 0:
        ring = X[i][band]; cen = ring.mean(0); r = np.hypot(ring[:, 0] - cen[0], ring[:, 2] - cen[2])   # radius about the ring's own centre
        rb = np.hypot(X[i][body][:, 0] - cx, X[i][body][:, 2] - cz)
        top = X[i][:, 1].max(); base = X[i][:, 1].min()
        s = f"t={t[i]:6.2f} {seg[i][:22]:22s} gap={gap[i]:5.1f}mm ring r[{r.min()*1e3:5.1f},{r.max()*1e3:5.1f}]mm (dent {(R0-r.min())*1e3:5.1f}mm) tilt {np.hypot(cen[0]-cx, cen[2]-cz)*1e3:4.1f}mm top {top:.3f} base {base:.3f}"
        if liq:
            P = liq[i]; inside = np.hypot(P[:, 0] - cx, P[:, 2] - cz) < 0.06
            s += f" | water lvl {np.percentile(P[inside, 1], 99) if inside.any() else 0:.3f} spilled {int((~inside).sum())} ymin {P[:,1].min():.3f}"
        print(s); last = seg[i]

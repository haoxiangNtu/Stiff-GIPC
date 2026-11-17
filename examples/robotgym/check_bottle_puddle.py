#!/usr/bin/env python3
"""Spilled-water metrics for bottle recordings: water on the table outside the bottle (count, ml), its radius about the
bottle axis (p50 / p90), its in-plane speed over 0.2 s (p50 / p90), and the bulk spacing inside the bottle (median 6-NN
distance, compression check).   python examples/robotgym/check_bottle_puddle.py a=rec.npz b=rec.npz ..."""
import sys
import numpy as np
from scipy.spatial import cKDTree

CX, CZ, ML = 0.45, 0.0, 0.0156          # bottle axis, ml per particle (dx 2.5 mm)
for arg in sys.argv[1:]:
    label, path = arg.split("=", 1) if "=" in arg else (arg, arg)
    d = np.load(path, allow_pickle=True); n = int(d["liquid_pts__n"]); T = d["t"]
    print(f"=== {label}: {n} records, t_end {T[-1]:.1f} s")
    step = max(1, int(round(0.2 / (T[1] - T[0]))))
    for i in range(step, n, max(1, int(round(1.0 / (T[1] - T[0]))))):
        x = d[f"liquid_pts_{i}"]; xp = d[f"liquid_pts_{i - step}"]
        r = np.hypot(x[:, 0] - CX, x[:, 2] - CZ)
        tab = (x[:, 1] < 0.015) & (r > 0.05)
        inb = (r < 0.045) & (x[:, 1] > 0.005)
        msg = f"  t={T[i]:5.1f}s table {tab.sum():5d} ({tab.sum()*ML:5.1f} ml)"
        if tab.sum() > 10:
            v = np.hypot(*(x[tab] - xp[tab])[:, [0, 2]].T) / (T[i] - T[i - step])
            msg += f" r p50 {np.percentile(r[tab], 50)*1e3:5.0f} p90 {np.percentile(r[tab], 90)*1e3:5.0f} mm | speed p50 {np.percentile(v, 50)*1e3:5.1f} p90 {np.percentile(v, 90)*1e3:5.1f} mm/s"
        if inb.sum() > 100:
            dd, _ = cKDTree(x[inb]).query(x[inb], k=7)
            msg += f" | in-bottle {inb.sum()} 6nn {np.median(dd[:, 6])*1e3:.2f} mm lvl {np.percentile(x[inb, 1], 99)*1e3:.0f} mm"
        print(msg)

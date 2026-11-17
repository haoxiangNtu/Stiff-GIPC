#!/usr/bin/env python
"""Slice thickness of a cucumber_cut run in the rest frame, from the cut planes prep_pieces.py fitted between the pieces.

Slice k (k = 1..K-1) lies between rest plane k-1 and plane k.  Its thickness is measured along z (the cucumber axis
direction of the rest pose) at three points of the section: on the axis, at the top and at the bottom.  Section = rest
particles within 20 mm of the mean cut; axis = the centre of its x/y bounding box, top / bottom = its highest / lowest y.
Also prints each plane's z on the axis and its tilt.

    python slice_thick.py pieces_<tag>.npz render_pieces_<tag>.npz

(Rewritten 2026-11-02 after the scratchpad original was lost in a reboot; reproduces the 05:40 table within 0.1 mm:
9c 3.1/6.0/4.6 | 4.8/4.9/5.0 | 1.4/7.2/4.1 (table 4.7, 7.1), 10b 6.2/6.3/6.2 | 7.7/6.1/6.3 | 4.6/6.6/6.1 (table 6.1, 7.6).)
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pieces"); ap.add_argument("render")
    a = ap.parse_args()
    d = np.load(a.pieces); r = np.load(a.render)
    x0 = d["x0"].astype(np.float64); cuts = np.asarray(d["cuts"], float)
    P = np.asarray(r["planes"], float)                      # rows: n (3), c (3)
    near = np.abs(x0[:, 2] - cuts.mean()) < 0.02
    lo, hi = x0[near, :2].min(0), x0[near, :2].max(0)
    cxy = 0.5 * (lo + hi)

    def zat(j, x, y):
        n, c = P[j, :3], P[j, 3:]
        return c[2] - (n[0] * (x - c[0]) + n[1] * (y - c[1])) / n[2]

    pts = {"axis": (cxy[0], cxy[1]), "top": (cxy[0], hi[1]), "bottom": (cxy[0], lo[1])}
    print(f"centre ({cxy[0]*1e3:.1f}, {cxy[1]*1e3:.1f}) mm, y {lo[1]*1e3:.1f}..{hi[1]*1e3:.1f} mm, {len(P)} planes")
    for j in range(len(P)):
        tilt = np.degrees(np.arccos(abs(P[j, 2])))
        print(f"  plane {j}: z on axis {zat(j, *pts['axis'])*1e3:7.1f} mm, tilt {tilt:4.1f} deg")
    nc = min(len(P), len(cuts))
    res = {}
    for name, (x, y) in pts.items():
        res[name] = [(zat(j, x, y) - zat(j - 1, x, y)) * 1e3 for j in range(1, nc)]
    for name, v in res.items():
        print(f"{name:7s} " + " / ".join(f"{t:.1f}" for t in v))
    print("table   " + " | ".join(" / ".join(f"{t:.1f}" for t in res[k]) for k in ("axis", "top", "bottom")))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Pre-bend the sleeves of a shirt mesh down over the table's side edges: flat vertices (y = 0) with |z| > ys go to
(x, -(|z| - ys), +-ys), i.e. the sleeves start hanging along a table of half-width ~ys (contact-only grasp of a hanging flap,
as for the hem). Write under a NEW file name: the engine's metis sort cache is keyed by the mesh stem.
    python3 examples/nuka/tools/make_shirt_sleeves.py Assets/nuka/shirt_bent17.obj --ys 0.141 --out Assets/nuka/shirt_bent17_sl141.obj"""
import argparse
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("--ys", type=float, required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
lines = open(a.src).read().splitlines()
out, nb = [], 0
for ln in lines:
    if ln.startswith("v "):
        x, y, z = (float(v) for v in ln.split()[1:4])
        if abs(y) < 1e-9 and abs(z) > a.ys:
            y, z = -(abs(z) - a.ys), np.sign(z) * a.ys; nb += 1
        out.append(f"v {x:.6f} {y:.6f} {z:.6f}")
    else:
        out.append(ln)
open(a.out, "w").write("\n".join(out) + "\n")
print(f"{a.out}: {nb} sleeve vertices bent down at |z| = {a.ys}")

#!/usr/bin/env python3
"""Rectangular cloth grid (.obj) in the XZ plane at y=0, centred at the origin, metres.
    python examples/robotgym/make_cloth_grid.py --lx 1.2 --lz 0.8 --h 0.025 --out Assets/robotgym/sheet_120x80_25mm.obj
Triangles alternate the diagonal (no preferred fold direction)."""
import argparse, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--lx", type=float, required=True); ap.add_argument("--lz", type=float, required=True)
ap.add_argument("--h", type=float, required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
nx, nz = int(round(a.lx / a.h)) + 1, int(round(a.lz / a.h)) + 1
xs, zs = np.linspace(-a.lx / 2, a.lx / 2, nx), np.linspace(-a.lz / 2, a.lz / 2, nz)
with open(a.out, "w") as f:
    f.write(f"# cloth grid {a.lx} x {a.lz} m, spacing {a.h} m, {nx}x{nz} vertices\n")
    for k in range(nz):
        for i in range(nx):
            f.write(f"v {xs[i]:.6f} 0 {zs[k]:.6f}\n")
    vid = lambda i, k: k * nx + i + 1
    for k in range(nz - 1):
        for i in range(nx - 1):
            a0, b0, c0, d0 = vid(i, k), vid(i + 1, k), vid(i + 1, k + 1), vid(i, k + 1)
            if (i + k) % 2 == 0:
                f.write(f"f {a0} {c0} {b0}\nf {a0} {d0} {c0}\n")
            else:
                f.write(f"f {a0} {d0} {b0}\nf {b0} {d0} {c0}\n")
print(f"{a.out}: {nx*nz} vertices, {2*(nx-1)*(nz-1)} faces")

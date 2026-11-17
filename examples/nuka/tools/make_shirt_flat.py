"""Flat single-layer T-shirt silhouette as a triangle cloth mesh (OBJ): body rectangle + two sleeves + collar notch, lying in the
x-z plane (y = 0), hem at x = 0, collar at x = length. Grid cells inside the outline are split into two triangles.

    python3 examples/nuka/tools/make_shirt_flat.py --out Assets/nuka/shirt_flat.obj --length 0.40 --width 0.32 --sleeve 0.05 --cell 0.008
"""
import argparse
import numpy as np


def inside(x, z, L, W, sleeve_w, sleeve_len, collar_r):
    """x along the shirt length (0 = hem), z across (0 = centre line)."""
    half = W / 2
    if 0 <= x <= L and abs(z) <= half:
        if x > L - collar_r and abs(z) < collar_r and (x - (L - collar_r)) ** 2 + z ** 2 > 0 and np.hypot(x - L, z) < collar_r:
            return False                                            # collar notch (half disc at the collar end)
        return True
    if L - sleeve_len - 0.0 <= x <= L - 0.02 and half < abs(z) <= half + sleeve_w:   # sleeves hanging at the collar end
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--length", type=float, default=0.40)
    ap.add_argument("--width", type=float, default=0.32)
    ap.add_argument("--sleeve", type=float, default=0.05, help="sleeve extension beyond the body on each side")
    ap.add_argument("--sleeve-len", type=float, default=0.14, help="sleeve extent along the length, from the collar end")
    ap.add_argument("--collar", type=float, default=0.05)
    ap.add_argument("--cell", type=float, default=0.008)
    ap.add_argument("--prebend", type=float, default=0.0, help="hem flap length folded DOWN 90 deg (x < prebend -> x = prebend, y = -(prebend - x)): starts hanging")
    args = ap.parse_args()
    L, W = args.length, args.width
    nx = int(round(L / args.cell)); nz = int(round((W + 2 * args.sleeve) / args.cell))
    xs = np.linspace(0.0, L, nx + 1); zs = np.linspace(-(W / 2 + args.sleeve), W / 2 + args.sleeve, nz + 1)
    idx = -np.ones((nx + 1, nz + 1), int); verts = []
    cells = []
    for i in range(nx):
        for k in range(nz):
            cx, cz = 0.5 * (xs[i] + xs[i + 1]), 0.5 * (zs[k] + zs[k + 1])
            if inside(cx, cz, L, W, args.sleeve, args.sleeve_len, args.collar):
                cells.append((i, k))
    for i, k in cells:
        for di, dk in ((0, 0), (1, 0), (0, 1), (1, 1)):
            if idx[i + di, k + dk] < 0:
                idx[i + di, k + dk] = len(verts); verts.append((xs[i + di], 0.0, zs[k + dk]))
    faces = []
    for i, k in cells:
        a, b, c, d = idx[i, k], idx[i + 1, k], idx[i + 1, k + 1], idx[i, k + 1]
        if (i + k) % 2 == 0:
            faces += [(a, b, c), (a, c, d)]
        else:
            faces += [(a, b, d), (b, c, d)]
    if args.prebend > 0:
        verts = [(args.prebend, -(args.prebend - x), z) if x < args.prebend - 1e-9 else (x, y, z) for (x, y, z) in verts]
    with open(args.out, "w") as f:
        f.write(f"# flat T-shirt silhouette {L} x {W} m, sleeves {args.sleeve}, cell {args.cell}\n")
        for v in verts:
            f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
        for t in faces:
            f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")
    V = np.array(verts)
    print(f"[shirt] {len(verts)} verts, {len(faces)} faces, bbox x [{V[:,0].min():.3f}, {V[:,0].max():.3f}] y [{V[:,1].min():.3f}, {V[:,1].max():.3f}] z [{V[:,2].min():.3f}, {V[:,2].max():.3f}] -> {args.out}")


if __name__ == "__main__":
    main()

"""Geometry-preserving refinement of coarse collision meshes.

Used to build Assets/fr3/meshes_sub50 (FR3 arm links + hand, every edge <= 50 mm) from Assets/fr3/meshes:
    python tools/subdivide_long_edges.py 50 Assets/fr3/meshes/link0.stl Assets/fr3/meshes_sub50/link0.stl ...
(finger_collision.stl is left as is). Why: a single 186 mm edge of hand.stl made the edge-edge BVH query
serial once the gripper closed (2102 narrow-phase checks on one thread per query); see
AUDIT_VERIFICATION_2026-10-09.md section 2.4.

Split every triangle-mesh edge longer than L_max at its midpoint (conforming red/green refinement),
repeat until no edge exceeds L_max. New vertices lie on the original edges and every new triangle lies in
its parent's plane, so the surface (and its volume integrals) is unchanged; only the discretisation is finer.
usage: subdivide_long_edges.py L_max_mm in.stl out.stl [...pairs]"""
import struct, sys
import numpy as np

def load_stl(p):
    data = open(p, 'rb').read()
    if data[:5] == b'solid' and b'facet' in data[:300]:
        V = [[float(x) for x in l.split()[1:4]] for l in data.decode(errors='ignore').splitlines() if l.split()[:1] == ['vertex']]
        tri = np.array(V, dtype=np.float64).reshape(-1, 3, 3)
    else:
        n = struct.unpack('<I', data[80:84])[0]
        arr = np.frombuffer(data[84:84 + n * 50], dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]))
        tri = arr['v'].astype(np.float64)
    P = tri.reshape(-1, 3)
    U, inv = np.unique(P, axis=0, return_inverse=True)      # exact float32-derived coordinates, no rounding merge
    return U, inv.reshape(-1, 3).astype(np.int64)

def save_stl(p, V, F):
    with open(p, 'wb') as f:
        f.write(b'subdivided (geometry-preserving edge bisection)'.ljust(80, b' '))
        f.write(struct.pack('<I', len(F)))
        for a, b, c in F:
            va, vb, vc = V[a], V[b], V[c]
            nrm = np.cross(vb - va, vc - va); l = np.linalg.norm(nrm); nrm = nrm / l if l > 0 else nrm
            f.write(struct.pack('<12fH', *nrm, *va, *vb, *vc, 0))

def subdivide(V, F, lmax):
    V = [tuple(v) for v in V]
    F = [tuple(f) for f in F]
    while True:
        Va = np.array(V)
        mid = {}
        for (a, b, c) in F:
            for u, w in ((a, b), (b, c), (c, a)):
                k = (min(u, w), max(u, w))
                if k not in mid and np.linalg.norm(Va[u] - Va[w]) > lmax:
                    mid[k] = None
        if not mid:
            return np.array(V), np.array(F)
        for k in mid:
            # midpoint in the stored precision: the STL is float32, so round the midpoint to float32 as well,
            # which keeps it on the segment to float32 accuracy (same as any vertex of the input file)
            m = tuple(np.float32((np.float64(V[k[0]][i]) + np.float64(V[k[1]][i])) * 0.5).item() for i in range(3))
            mid[k] = len(V); V.append(m)
        G = []
        for (a, b, c) in F:
            m = [mid.get((min(u, w), max(u, w))) for u, w in ((a, b), (b, c), (c, a))]
            n = sum(x is not None for x in m)
            if n == 0:
                G.append((a, b, c)); continue
            if n == 3:
                mab, mbc, mca = m
                G += [(a, mab, mca), (mab, b, mbc), (mca, mbc, c), (mab, mbc, mca)]; continue
            # rotate so the split edges come first: (a,b) split always
            verts = [a, b, c]
            r = 0
            while m[r] is None or (n == 2 and m[(r + 1) % 3] is None):
                r += 1
            a, b, c = verts[r], verts[(r + 1) % 3], verts[(r + 2) % 3]
            mab, mbc, mca = m[r], m[(r + 1) % 3], m[(r + 2) % 3]
            if n == 1:
                G += [(a, mab, c), (mab, b, c)]
            else:  # (a,b) and (b,c) split, (c,a) not
                G += [(mab, b, mbc)]
                # quad a-mab-mbc-c: cut along the shorter diagonal
                Va2 = np.array(V)
                if np.linalg.norm(Va2[a] - Va2[mbc]) <= np.linalg.norm(Va2[mab] - Va2[c]):
                    G += [(a, mab, mbc), (a, mbc, c)]
                else:
                    G += [(a, mab, c), (mab, mbc, c)]
        F = G

def edge_stats(V, F):
    E = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)
    L = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    return len(E), L.max()

def volume_area(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    vol = np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6.0
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()
    com = (np.einsum('ij,ij->i', a, np.cross(b, c))[:, None] * (a + b + c)).sum(0) / (24.0 * vol)
    return vol, area, com

if __name__ == '__main__':
    lmax_mm = float(sys.argv[1]); pairs = sys.argv[2:]
    for src, dst in zip(pairs[0::2], pairs[1::2]):
        V, F = load_stl(src)
        unit = 1e-3 if np.abs(V).max() < 5 else 1.0          # metres or millimetres
        V2, F2 = subdivide(V, F, lmax_mm * 1e-3 / unit * unit if unit == 1.0 else lmax_mm * 1e-3)
        save_stl(dst, V2, F2)
        V3, F3 = load_stl(dst)
        e0, l0 = edge_stats(V, F); e1, l1 = edge_stats(V3, F3)
        v0, a0, c0 = volume_area(V, F); v1, a1, c1 = volume_area(V3, F3)
        s = 1e3 if unit == 1e-3 else 1.0
        print(f'{src.split("/")[-1]:22s} V {len(V):5d} -> {len(V3):5d}  E {e0:5d} -> {e1:5d}  max edge {l0*s:6.1f} -> {l1*s:5.1f} mm  '
              f'volume rel diff {abs(v1-v0)/abs(v0):.1e}  area rel diff {abs(a1-a0)/a0:.1e}  com shift {np.linalg.norm(c1-c0)*s:.2e} mm')

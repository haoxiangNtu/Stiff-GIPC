#!/usr/bin/env python3
"""Bed + blanket for the bed task from the user's FBX assets (床.fbx, 毛毯.fbx). Only a uniform scale is applied; no
part is re-shaped. The bed is one 243k-vertex mesh with four materials; its parts are split into connected components.

Scale: the FBX bed is 1.13 x 1.41 m (frame), the bedding top 0.35 m and the headboard 0.61 m above its floor, pillows
0.48 m wide -- a double bed (two pillows, two bolsters) at ~0.7x real size. x1.4 gives a 1.56 x 1.98 m frame, bedding
top 0.50 m, headboard 0.85 m, pillows 0.67 m, bolsters 12.6 cm: a standard 1.5 m double bed. The blanket (1.9 x 1.2 m,
a flat 1 mm box in the FBX) is a real throw-blanket size and is NOT scaled; at the bed's x1.4 it is a throw across the
foot half of the bed. (Scaled with the bed it would be 2.66 x 1.68 m, out of reach of two fixed FR3s for any fold.)

Frame: metres, y up, floor y = 0, bed centred on x = 0, the bedding's foot face at z = 0, the head toward -z.

Outputs (--out-dir):
  bed_collision.obj   duvet + top sheet + frame + legs + headboard (all original triangles, merged into one fixed body;
                      triangles with an edge > --max-edge are split in their own plane: shape unchanged, but no single
                      0.3-1.8 m edge to serialise the BVH edge queries). Pillows and bolsters are render-only (> 0.5 m
                      from the blanket at any time).
  render/<part>.obj   every part (incl. pillows, bolsters) for the renderer, with render/parts.json (names, colours)
  blanket_<h>mm.obj   the blanket as a cloth grid (h spacing, alternating diagonals), centred, y = 0
  bed_meta.json       key dimensions (bedding top, faces, blanket placement of the FBX scene)
    python examples/robotgym/make_bed_fbx.py ~/Downloads/床.fbx ~/Downloads/毛毯.fbx --out-dir Assets/robotgym/fbx/bed
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from fbx_read import Scene, clean, local_tf, props70  # noqa: E402


def mesh_parts(fbx_path):
    """[(material_name, V (m, world), F)] per material of every mesh model"""
    sc = Scene(fbx_path)
    s = 0.01 * sc.unit_cm
    out = []
    for mid, m in sc.models().items():
        geo = sc.mesh_of(mid)
        if geo is None:
            continue
        W = sc.world(mid); _, G = local_tf(props70(m)); M = W @ G
        V, idx, starts, lens, mat, _ = sc.geometry_arrays(geo)
        Vw = (V @ M[:3, :3].T + M[:3, 3]) * s
        tris, poly_of = sc.triangulate(idx, starts, lens)
        mats = sc.materials_of(mid)
        tm = mat[poly_of]
        for k in np.unique(tm):
            T = idx[tris[tm == k]]
            used = np.unique(T); remap = -np.ones(len(Vw), np.int64); remap[used] = np.arange(len(used))
            out.append((mats[k] if k < len(mats) else f"mat{k}", Vw[used], remap[T], clean(m.props[1])))
    return out


def components(V, F):
    n = len(V)
    A = coo_matrix((np.ones(len(F) * 2), (np.r_[F[:, 0], F[:, 1]], np.r_[F[:, 1], F[:, 2]])), shape=(n, n))
    _, lab = connected_components(A, directed=False)
    flab = lab[F[:, 0]]
    out = []
    for c in np.unique(flab):
        sel = flab == c
        used = np.unique(F[sel]); remap = -np.ones(n, np.int64); remap[used] = np.arange(len(used))
        out.append((V[used], remap[F[sel]]))
    out.sort(key=lambda vf: -len(vf[1]))
    return out


def split_long_edges(V, F, max_edge, max_iter=40):
    """conforming bisection of every edge longer than max_edge (midpoint, in the triangle's own plane: shape unchanged).
    A triangle with 1 / 2 / 3 marked edges splits into 2 / 3 / 4, so slivers only gain triangles along their long edge."""
    Vs = [np.asarray(V, float)]; F = np.asarray(F, np.int64); nv = len(Vs[0])
    for _ in range(max_iter):
        Vall = np.concatenate(Vs)
        E = np.sort(np.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1), 2)          # edge i = (v_i, v_i+1)
        L = np.linalg.norm(Vall[E[..., 0]] - Vall[E[..., 1]], axis=2)
        mark = L > max_edge
        if not mark.any():
            break
        ue, inv = np.unique(E[mark].reshape(-1, 2), axis=0, return_inverse=True)
        Vs.append(0.5 * (Vall[ue[:, 0]] + Vall[ue[:, 1]]))
        M = -np.ones(F.shape, np.int64); M[mark] = nv + inv.ravel(); nv += len(ue)
        code = 4 * mark[:, 0] + 2 * mark[:, 1] + mark[:, 2]
        a, b, c = F[:, 0], F[:, 1], F[:, 2]; ab, bc, ca = M[:, 0], M[:, 1], M[:, 2]
        T = lambda *cols: np.stack(cols, 1)
        pieces = {0: [T(a, b, c)],
                  4: [T(a, ab, c), T(ab, b, c)], 2: [T(a, b, bc), T(a, bc, c)], 1: [T(a, b, ca), T(ca, b, c)],
                  6: [T(ab, b, bc), T(a, ab, bc), T(a, bc, c)], 3: [T(bc, c, ca), T(a, b, bc), T(a, bc, ca)],
                  5: [T(a, ab, ca), T(ab, b, c), T(ab, c, ca)],
                  7: [T(a, ab, ca), T(ab, b, bc), T(ca, bc, c), T(ab, bc, ca)]}
        F = np.concatenate([pc[code == k] for k, lst in pieces.items() for pc in lst])
    return np.concatenate(Vs), F


def write_obj(path, V, F, header=""):
    with open(path, "w") as f:
        if header:
            f.write(f"# {header}\n")
        f.write("".join(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n" for p in V))
        f.write("".join(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n" for t in F))


def cloth_grid(lx, lz, h):
    nx, nz = int(round(lx / h)) + 1, int(round(lz / h)) + 1
    xs, zs = np.linspace(-lx / 2, lx / 2, nx), np.linspace(-lz / 2, lz / 2, nz)
    V = np.array([[xs[i], 0.0, zs[k]] for k in range(nz) for i in range(nx)])
    vid = lambda i, k: k * nx + i
    F = []
    for k in range(nz - 1):
        for i in range(nx - 1):
            a, b, c, d = vid(i, k), vid(i + 1, k), vid(i + 1, k + 1), vid(i, k + 1)
            F += [(a, c, b), (a, d, c)] if (i + k) % 2 == 0 else [(a, d, b), (b, d, c)]
    return V, np.array(F, np.int64), (nx, nz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bed_fbx")
    ap.add_argument("blanket_fbx")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scale", type=float, default=1.4, help="bed scale (1.4: standard 1.5 x 2.0 m double bed)")
    ap.add_argument("--blanket-scale", type=float, default=1.0)
    ap.add_argument("--blanket-h", type=float, nargs="+", default=[0.020, 0.015], help="cloth grid spacings to write (m)")
    ap.add_argument("--max-edge", type=float, default=0.04, help="duvet (the surface the blanket lies on): split edges longer than this")
    ap.add_argument("--max-edge-other", type=float, default=0.30, help="frame / sheet / headboard: only the 0.3-2.5 m edges are split")
    a = ap.parse_args()
    os.makedirs(osp.join(a.out_dir, "render"), exist_ok=True)

    parts = mesh_parts(a.bed_fbx)
    # bed frame of reference (FBX metres): frame centre x, floor y, bedding foot face z
    allV = np.concatenate([p[1] for p in parts])
    wood = [p for p in parts if p[0] == "木床"][0]
    x_c = 0.5 * (wood[1][:, 0].min() + wood[1][:, 0].max())
    y_floor = allV[:, 1].min()
    bedding = [p for p in parts if p[0] == "枕头2"][0]          # duvet + top sheet + pillows (material names in the FBX are swapped)
    comps_b = components(bedding[1], bedding[2])
    duvet = comps_b[0]
    z_foot = duvet[0][:, 2].max()
    s = a.scale
    def tf(V):
        return np.stack([(V[:, 0] - x_c) * s, (V[:, 1] - y_floor) * s, (V[:, 2] - z_foot) * s], 1)
    # name the components by their place in the bed (the FBX material names do not match the parts: "床垫" is the
    # headboard cushion, "枕头2" the duvet / sheet / pillows, "枕头1" the bolsters)
    named = {}
    for mname, V, F, model in parts:
        comps = components(V, F)
        for ci, (cv, cf) in enumerate(comps):
            W = tf(cv)
            lo, hi = W.min(0), W.max(0)
            size = hi - lo
            if mname == "木床":
                key = "frame" if size[2] > 1.0 else ("headboard_frame" if size[0] > 1.0 else f"leg{ci}")
            elif mname == "床垫":
                key = "headboard_cushion"
            elif mname == "枕头2":
                key = "duvet" if ci == 0 else ("sheet" if size[0] > 1.0 else f"pillow{ci}")
            else:
                key = f"bolster{ci}" if size[0] > 0.2 else f"bolster_cap{ci}"
            named[key] = (W, cf)
    colors = {"frame": (0.55, 0.38, 0.24), "headboard_frame": (0.55, 0.38, 0.24), "headboard_cushion": (0.78, 0.74, 0.68),
              "duvet": (0.93, 0.93, 0.95), "sheet": (0.90, 0.91, 0.93)}
    meta_parts = {}
    for k, (W, F) in named.items():
        write_obj(osp.join(a.out_dir, "render", f"{k}.obj"), W, F, f"bed part {k}, FBX x{s}")
        c = colors.get(k, (0.55, 0.38, 0.24) if k.startswith("leg") else ((0.95, 0.95, 0.96) if k.startswith("pillow") else (0.80, 0.76, 0.66)))
        meta_parts[k] = dict(file=f"render/{k}.obj", color=c, verts=len(W), faces=len(F),
                             lo=np.round(W.min(0), 4).tolist(), hi=np.round(W.max(0), 4).tolist())
    json.dump(meta_parts, open(osp.join(a.out_dir, "render", "parts.json"), "w"), indent=1)
    # collision body: everything except pillows / bolsters
    coll = [k for k in named if not (k.startswith("pillow") or k.startswith("bolster"))]
    Vs, Fs, off = [], [], 0
    for k in coll:
        W, F = named[k]
        W2, F2 = split_long_edges(W, F, a.max_edge if k == "duvet" else a.max_edge_other)
        Vs.append(W2); Fs.append(F2 + off); off += len(W2)
    Vc, Fc = np.concatenate(Vs), np.concatenate(Fs)
    write_obj(osp.join(a.out_dir, "bed_collision.obj"), Vc, Fc, f"bed collision ({', '.join(coll)}), FBX x{s}, duvet edges <= {a.max_edge} m, others <= {a.max_edge_other} m")
    # blanket: FBX placement (flat box) mapped into the bed frame with the BLANKET's own scale about its centre
    bl = mesh_parts(a.blanket_fbx)[0][1]
    blo, bhi = bl.min(0), bl.max(0)
    bc = 0.5 * (blo + bhi)
    size = (bhi - blo) * a.blanket_scale
    for h in a.blanket_h:
        Vg, Fg, (nx, nz) = cloth_grid(size[0], size[2], h)
        write_obj(osp.join(a.out_dir, f"blanket_{h*1e3:.0f}mm.obj"), Vg, Fg,
                  f"blanket {size[0]:.3f} x {size[2]:.3f} m (FBX 毛毯 x{a.blanket_scale}), grid {h} m, {nx}x{nz} verts")
    duv = named["duvet"][0]
    meta = dict(scale=s, blanket_scale=a.blanket_scale, blanket_size=[float(size[0]), float(size[2])],
                fbx_frame=dict(x_c=float(x_c), y_floor=float(y_floor), z_foot=float(z_foot)),
                fbx_blanket_centre_in_bed_frame=np.round(tf(bc[None])[0], 4).tolist(),
                fbx_blanket_extent_in_bed_frame=dict(x=[float((blo[0] - x_c) * s), float((bhi[0] - x_c) * s)],
                                                     z=[float((blo[2] - z_foot) * s), float((bhi[2] - z_foot) * s)]),
                duvet_lo=np.round(duv.min(0), 4).tolist(), duvet_hi=np.round(duv.max(0), 4).tolist(),
                collision=dict(parts=coll, verts=int(len(Vc)), faces=int(len(Fc)), lo=np.round(Vc.min(0), 4).tolist(), hi=np.round(Vc.max(0), 4).tolist()),
                parts=meta_parts)
    json.dump(meta, open(osp.join(a.out_dir, "bed_meta.json"), "w"), indent=1)
    print(f"bed x{s}: collision {len(Vc)} verts {len(Fc)} faces ({', '.join(coll)}); bbox {np.round(Vc.min(0),3).tolist()} .. {np.round(Vc.max(0),3).tolist()}")
    for k, v in meta_parts.items():
        print(f"  {k:18s} {v['faces']:7d} faces  lo {v['lo']} hi {v['hi']}")
    print(f"  blanket {size[0]:.3f} x {size[2]:.3f} m; duvet top y {duv[:,1].max():.4f} m, duvet x {duv[:,0].min():.3f}..{duv[:,0].max():.3f}, "
          f"z {duv[:,2].min():.3f}..{duv[:,2].max():.3f}")


if __name__ == "__main__":
    main()

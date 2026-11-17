#!/usr/bin/env python3
"""Kitchen island for the sponge task from the user's FBX asset (厨房.fbx): double-bowl sink, gooseneck faucet, 0.90 m
countertop, cabinets. The FBX is already at real scale (countertop 0.90 m, cabinets 0.60 m deep, bowls 439 x 318 x 112 mm,
faucet 0.34 m above the counter), so it is only translated: origin on the floor below the midpoint of the two bowl
centres (x right of the faucet toward the open side of the island, y up, z along the counter).

Collision (one fixed body, the original triangles; countertop edges > --max-edge split in their own plane): countertop,
counter edge strips, both bowls (single-layer stainless shells), both strainer baskets, faucet. Plus one thing that is not
in the FBX: a drain stopper (8 mm disc, r 36 mm) under each strainer -- the bowls have a 56 mm drain hole and a sink only
holds water with the drain stopped. Cabinets / doors / shelves / plinth are render-only (below the counter, out of reach).

Outputs (--out-dir): kitchen_collision.obj, render/<part>.obj + render/parts.json, kitchen_meta.json (bowl interiors:
outline polygon at mid-depth, bottom height, fillet radius, rim height, drain centres; counter edge; faucet bbox).
    python examples/robotgym/make_kitchen_fbx.py ~/Downloads/厨房.fbx --out-dir Assets/robotgym/fbx/kitchen
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from fbx_read import Scene, clean, local_tf, props70  # noqa: E402
from make_bed_fbx import split_long_edges, write_obj  # noqa: E402

ROLE = {   # FBX model name -> (part, collision?)
    "cgaxis_models_86_20_02_1": ("countertop", True), "cgaxis_models_86_20_01_2": ("countertop_edges", True),
    "cgaxis_models_86_20_05_2": ("bowl_a", True), "cgaxis_models_86_20_05_3": ("bowl_b", True),
    "cgaxis_models_86_20_05_1": ("strainer_a", True), "cgaxis_models_86_20_05_2_001": ("strainer_b", True),
    "cgaxis_models_86_20_06_1": ("faucet_base", True), "cgaxis_models_86_20_06_2": ("faucet_handle", True),
    "cgaxis_models_86_20_06_3": ("faucet_spout", True),
}
COLORS = {"countertop": (0.30, 0.30, 0.32), "countertop_edges": (0.45, 0.33, 0.22), "bowl": (0.78, 0.79, 0.80),
          "strainer": (0.70, 0.71, 0.72), "faucet": (0.08, 0.08, 0.09), "cabinet": (0.86, 0.86, 0.86), "plinth": (0.25, 0.25, 0.26),
          "shelf": (0.50, 0.37, 0.25), "stopper": (0.10, 0.10, 0.10)}


def fbx_models(path):
    sc = Scene(path)
    s = 0.01 * sc.unit_cm
    out = []
    for mid, m in sc.models().items():
        geo = sc.mesh_of(mid)
        if geo is None:
            continue
        W = sc.world(mid); _, G = local_tf(props70(m)); M = W @ G
        V, idx, starts, lens, mat, _ = sc.geometry_arrays(geo)
        tris, _ = sc.triangulate(idx, starts, lens)
        Vw = (V @ M[:3, :3].T + M[:3, 3]) * s
        out.append((clean(m.props[1]), Vw, idx[tris], sc.materials_of(mid)))
    return out


def disc(centre, r, y0, y1, n=48):
    """closed cylinder (outward normals): bottom y0, top y1"""
    cx, cz = centre
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([cx + r * np.cos(ang), np.zeros(n), cz + r * np.sin(ang)], 1)
    V = np.vstack([ring + [0, y0, 0], ring + [0, y1, 0], [[cx, y0, cz], [cx, y1, cz]]])
    cb, ct = 2 * n, 2 * n + 1
    F = []
    for j in range(n):
        a, b = j, (j + 1) % n
        F += [(a, n + b, b), (a, n + a, n + b)]          # side
        F += [(cb, a, b), (ct, n + b, n + a)]            # bottom (normal -y), top (+y)
    return V, np.array(F, np.int64)


def outline(V, F, y):
    """largest closed loop of the horizontal section at height y (xz polygon, CCW)"""
    import trimesh
    from shapely.geometry import MultiLineString
    from shapely.ops import polygonize, unary_union
    segs = trimesh.intersections.mesh_plane(trimesh.Trimesh(V, F, process=True), [0, 1, 0], [0, y, 0])
    segs = np.round(segs, 6)                       # endpoints of neighbouring triangles' segments must coincide to be noded
    polys = sorted(polygonize(unary_union(MultiLineString([((a[0], a[2]), (b[0], b[2])) for a, b in segs]))), key=lambda p: -p.area)
    p = polys[0].simplify(0.0005)
    xy = np.asarray(p.exterior.coords)[:-1]
    if np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1]) < 0:
        xy = xy[::-1]
    return xy, p.area


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fbx")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-edge", type=float, default=0.15, help="countertop: split edges longer than this (m)")
    a = ap.parse_args()
    os.makedirs(osp.join(a.out_dir, "render"), exist_ok=True)
    models = fbx_models(a.fbx)
    byname = {n: (V, F) for n, V, F, _ in models}
    ca = byname["cgaxis_models_86_20_05_2"][0]; cb = byname["cgaxis_models_86_20_05_3"][0]
    cA = 0.5 * (ca.min(0) + ca.max(0)); cB = 0.5 * (cb.min(0) + cb.max(0))
    origin = np.array([0.5 * (cA[0] + cB[0]), 0.0, 0.5 * (cA[2] + cB[2])])
    tf = lambda V: V - origin
    parts, coll = {}, []
    ncab = 0
    for name, V, F, mats in models:
        role = ROLE.get(name)
        if role is None:
            size = V.max(0) - V.min(0)
            if size[1] < 0.08 and V.min(0)[1] < 0.01:
                key = "plinth"
            elif size[1] < 0.06:
                key = f"shelf_{name[-3:].strip('_')}"
            else:
                ncab += 1; key = f"cabinet_{ncab:02d}"
            role = (key, False)
        key, is_coll = role
        W = tf(V)
        parts[key] = (W, F)
        if is_coll:
            coll.append(key)
    # drain stoppers under each strainer
    for s_ in ("a", "b"):
        Vs = parts[f"strainer_{s_}"][0]
        c = 0.5 * (Vs.min(0) + Vs.max(0)); r = 0.5 * (Vs.max(0)[0] - Vs.min(0)[0])
        y_bot = Vs.min(0)[1]
        Vd, Fd = disc((c[0], c[2]), r + 0.002, y_bot - 0.0092, y_bot - 0.0012)
        parts[f"stopper_{s_}"] = (Vd, Fd); coll.append(f"stopper_{s_}")
    meta_parts = {}
    for k, (W, F) in parts.items():
        write_obj(osp.join(a.out_dir, "render", f"{k}.obj"), W, F, f"kitchen part {k} (FBX, translated by {np.round(-origin, 4).tolist()})")
        base = k.split("_")[0]
        meta_parts[k] = dict(file=f"render/{k}.obj", collision=k in coll, color=COLORS.get(base, (0.8, 0.8, 0.8)),
                             verts=len(W), faces=len(F), lo=np.round(W.min(0), 4).tolist(), hi=np.round(W.max(0), 4).tolist())
    json.dump(meta_parts, open(osp.join(a.out_dir, "render", "parts.json"), "w"), indent=1)
    Vs, Fs, off = [], [], 0
    for k in coll:
        W, F = parts[k]
        if k.startswith("countertop"):
            W, F = split_long_edges(W, F, a.max_edge)
        Vs.append(W); Fs.append(F + off); off += len(W)
    Vc, Fc = np.concatenate(Vs), np.concatenate(Fs)
    write_obj(osp.join(a.out_dir, "kitchen_collision.obj"), Vc, Fc, f"kitchen collision ({', '.join(coll)})")
    bowls = {}
    for s_ in ("a", "b"):
        W, F = parts[f"bowl_{s_}"]
        poly, area = outline(W, F, 0.84)
        bottom = float(np.percentile(W[:, 1], 0.5))
        st = parts[f"strainer_{s_}"][0]
        bowls[s_] = dict(outline_xz=np.round(poly, 5).tolist(), area_m2=float(area), bottom_y=bottom, rim_y=float(W[:, 1].max()),
                         fillet_r=0.012, interior_lo=np.round(poly.min(0), 4).tolist(), interior_hi=np.round(poly.max(0), 4).tolist(),
                         drain_xz=np.round(0.5 * (st.min(0) + st.max(0))[[0, 2]], 4).tolist(), drain_r=float(0.5 * (st.max(0)[0] - st.min(0)[0])),
                         strainer_top_y=float(st.max(0)[1]))
    fct = np.vstack([parts[k][0] for k in ("faucet_base", "faucet_handle", "faucet_spout")])
    ct = parts["countertop"][0]
    meta = dict(origin_in_fbx=origin.tolist(), bowls=bowls, counter_top_y=float(ct[:, 1].max()),
                counter_x=[float(ct[:, 0].min()), float(ct[:, 0].max())], counter_z=[float(ct[:, 2].min()), float(ct[:, 2].max())],
                faucet_lo=np.round(fct.min(0), 4).tolist(), faucet_hi=np.round(fct.max(0), 4).tolist(),
                collision=dict(parts=coll, verts=int(len(Vc)), faces=int(len(Fc))), parts=meta_parts)
    json.dump(meta, open(osp.join(a.out_dir, "kitchen_meta.json"), "w"), indent=1)
    print(f"kitchen: origin (FBX) {np.round(origin, 4).tolist()}; collision {len(Vc)} verts {len(Fc)} faces ({', '.join(coll)})")
    for s_, b in bowls.items():
        print(f"  bowl {s_}: interior x {b['interior_lo'][0]:.4f}..{b['interior_hi'][0]:.4f} z {b['interior_lo'][1]:.4f}..{b['interior_hi'][1]:.4f} "
              f"({b['area_m2']*1e4:.0f} cm2), bottom {b['bottom_y']:.4f}, rim {b['rim_y']:.4f}, drain {b['drain_xz']} r {b['drain_r']*1e3:.1f} mm")
    print(f"  counter top {meta['counter_top_y']:.4f}, x {meta['counter_x']}, z {meta['counter_z']}; faucet {meta['faucet_lo']} .. {meta['faucet_hi']}")
    print(f"  render parts: {sorted(parts)}")


if __name__ == "__main__":
    main()

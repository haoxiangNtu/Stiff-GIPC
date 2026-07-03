#!/usr/bin/env python3
"""build_umi_finray_strategyF.py — STRATEGY_F-style hybrid mesh prep for the
UMI finray gripper.

Unlike the direct-stitch route (build_umi_finray_hybrid.py), this mirrors
case_39's STRATEGY_F: the finray's SOLID MOUNTING ROOT (the bottom band, the
part that plugs into the mount seat) is split off as a RIGID sub-mesh and
loaded as a separate ABD body.  The hollow fin-ray truss above stays FEM.  The
rigid sub-mesh is extracted FROM the same unified tet mesh, so its vertices are
COINCIDENT with the unified mesh's rigid verts — stitching them with gap=0 +
the engine's default high soft_motion_rate gives a clean, wrinkle-free join.

Z-solidity profile of the rotated TRI_gripper mesh (mesh/mount frame):
  Z 0.0245-0.031 : 87% solid  -> SOLID MOUNTING ROOT (rigid)
  Z 0.031-0.125  : ~4% solid  -> hollow fin-ray truss (FEM)
  Z 0.125-0.150  : 87% solid  -> solid contact tip (FEM, carried by the truss)

Rigid/FEM split: a TET is rigid iff its centroid Z < 0.032.

A VERTEX is rigid (vertex_region=1) iff ANY incident tet is rigid (so interface
verts belong to the rigid side — same convention as case_39's STRATEGY_F).

Output dir (per side L/R):
  UMI_finray_{side}_unified.npz   — vertices f64, tets i32, vertex_region i32
  UMI_finray_{side}_rigid.msh     — GMSH v2.2 ASCII rigid sub-mesh (ABD)
  UMI_finray_{side}_rigid_remap.npz — rigid_v_idx i32 (unified indices, in .msh
                                      vertex order so stitch pairs line up)

Usage:
  conda run -n test_v062 python build_umi_finray_strategyF.py \
      --umi-dir /home/ps/Downloads/assets/sim_data/urdf/ridgeback_dual_panda_UMI \
      --out-dir /home/ps/Downloads/assets/sim_data/umi_hybrid_sf
"""
import argparse
from pathlib import Path

import numpy as np
import meshio


RIGID_Z_THRESH = 0.032   # tet rigid iff centroid Z < this (mesh/mount frame)


def load_tet(vtk_path):
    m = meshio.read(vtk_path)
    verts = np.ascontiguousarray(m.points, dtype=np.float64)
    tets = None
    for c in m.cells:
        if c.type == "tetra":
            tets = np.ascontiguousarray(c.data, dtype=np.int32)
            break
    if tets is None:
        raise RuntimeError(f"no tetra cells in {vtk_path}")
    return verts, tets


def _Rz(deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def write_gmsh_v22_ascii(path, verts, tets):
    """GMSH ASCII v2.2 tet mesh compatible with Stiff-GIPC's .msh parser.
    Element type 4 = linear tetrahedron, 1-based indices."""
    n_verts = verts.shape[0]
    n_tets = tets.shape[0]
    with open(path, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write(f"$Nodes\n{n_verts}\n")
        for i in range(n_verts):
            v = verts[i]
            f.write(f"{i + 1} {v[0]:.17g} {v[1]:.17g} {v[2]:.17g}\n")
        f.write("$EndNodes\n")
        f.write(f"$Elements\n{n_tets}\n")
        for i in range(n_tets):
            t = tets[i]
            f.write(f"{i + 1} 4 2 0 0 "
                    f"{int(t[0]) + 1} {int(t[1]) + 1} "
                    f"{int(t[2]) + 1} {int(t[3]) + 1}\n")
        f.write("$EndElements\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--umi-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--z-thresh", type=float, default=RIGID_Z_THRESH)
    args = ap.parse_args()

    mesh_dir = Path(args.umi_dir) / "meshes" / "UMI" / "franka_fr3"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for side in ["L", "R"]:
        src = mesh_dir / f"TRI_gripper_{side}.vtk"
        assert src.exists(), src
        verts, tets = load_tet(src)
        # Rotate +90deg-Z into the body/mount frame (matches the existing prep).
        verts = (verts @ _Rz(90).T).copy()

        # --- Label each tet: rigid if centroid Z < thresh, else FEM ---
        centroids = verts[tets].mean(axis=1)         # (T,3)
        tet_rigid = centroids[:, 2] < args.z_thresh  # (T,) bool
        n_rigid_tets = int(tet_rigid.sum())

        # --- vertex_region: a vertex is rigid (1) if ANY incident tet is rigid ---
        vertex_region = np.zeros(len(verts), dtype=np.int32)
        rigid_tet_verts = np.unique(tets[tet_rigid].flatten())
        vertex_region[rigid_tet_verts] = 1
        n_rigid_v = int((vertex_region == 1).sum())

        # --- Extract the rigid sub-mesh (only the rigid tets) ---
        rigid_tets_global = tets[tet_rigid]                       # (Tr,4) unified idx
        rigid_v_idx = np.unique(rigid_tets_global.flatten()).astype(np.int32)
        rigid_verts = verts[rigid_v_idx]                         # (.msh vertex order)
        remap = -np.ones(len(verts), dtype=np.int32)
        remap[rigid_v_idx] = np.arange(len(rigid_v_idx), dtype=np.int32)
        rigid_tets_local = remap[rigid_tets_global].astype(np.int32)

        # Sanity: every rigid sub-mesh vert is region==1.
        assert np.all(vertex_region[rigid_v_idx] == 1)

        # --- Write outputs ---
        msh_path = out / f"UMI_finray_{side}_rigid.msh"
        write_gmsh_v22_ascii(str(msh_path), rigid_verts, rigid_tets_local)

        np.savez(out / f"UMI_finray_{side}_unified.npz",
                 vertices=verts.astype(np.float64),
                 tets=tets.astype(np.int32),
                 vertex_region=vertex_region.astype(np.int32))
        np.savez(out / f"UMI_finray_{side}_rigid_remap.npz",
                 rigid_v_idx=rigid_v_idx.astype(np.int32))

        zr = verts[rigid_v_idx, 2]
        print(f"=== {side} ===")
        print(f"  unified : {len(verts)} verts, {len(tets)} tets")
        print(f"  rigid   : {n_rigid_tets} tets, {n_rigid_v} verts "
              f"(rigid_v_idx in .msh order: {len(rigid_v_idx)})")
        print(f"  FEM     : {len(tets) - n_rigid_tets} tets, "
              f"{len(verts) - n_rigid_v} pure-FEM verts")
        print(f"  rigid Z-band: {zr.min():.4f} .. {zr.max():.4f} "
              f"(centroid thresh {args.z_thresh})")
        print(f"  wrote {msh_path.name}, "
              f"UMI_finray_{side}_unified.npz, "
              f"UMI_finray_{side}_rigid_remap.npz")

    print("\nDone. Assets in", out)


if __name__ == "__main__":
    main()

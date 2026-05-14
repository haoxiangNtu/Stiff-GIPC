#!/usr/bin/env python3
"""tools/build_case41_unified.py — fine variant of CASE40_UNIFIED.

Same source meshes (finger_clean.obj + soft_clean.obj) and same
boolean-union + post-tet labeling pipeline as build_case40_unified.py,
but tetgen is run with quality refinement (fixedvolume + maxvolume)
to produce a DENSER tet mesh.  This gives a higher-resolution FEM
soft pad while keeping rigid finger geometry coincident with the URDF
finger ABD body (so vertex_region label test stays valid).
"""
from pathlib import Path
import numpy as np

import meshio
import tetgen
import pyvista as pv
import trimesh
import pymeshfix


REPO   = Path("/home/ps/Downloads/Stiff-GIPC-hybrid-mesh")
SHARP_DIR = REPO / "Assets/sim_data/urdf/ridgeback_dual_panda_soft/meshes/plate/visual/soft_hard_segmenation"
FINGER = SHARP_DIR / "finger_clean.obj"
SOFT   = SHARP_DIR / "soft_clean.obj"
OUT_UNIFIED = REPO / "Assets/sim_data/hybrid_d/CASE41_UNIFIED_unified.npz"
OUT_RIGID   = REPO / "Assets/sim_data/hybrid_d/CASE41_UNIFIED_rigid.msh"
OUT_REMAP   = REPO / "Assets/sim_data/hybrid_d/CASE41_UNIFIED_rigid_remap.npz"

# Tet density target — case_40 (no refinement) yielded ~2500 tets.
# Aim ~6000-9000 tets here for "fine" variant.  3.0e-7 ≈ 0.67mm-edge tets;
# at scale 1.0 (≈14cm finger length) that's ~2x finer than case_40.
MAXVOL = 3.0e-7
MIN_VOL_KEEP = 5e-11   # drop near-degenerate tets (matches case_41 fine recipe)


def load_and_repair(path):
    m = trimesh.load(str(path), process=False)
    fix = pymeshfix.MeshFix(np.asarray(m.vertices), np.asarray(m.faces))
    fix.repair()
    pd = fix.mesh
    verts = np.asarray(pd.points, dtype=np.float64)
    faces = pd.faces.reshape(-1, 4)[:, 1:].astype(np.int32)
    print(f"  {path.name}: {len(m.vertices)}v → {len(verts)}v {len(faces)}f")
    return trimesh.Trimesh(verts, faces, process=False)


def main():
    print("[case41] pymeshfix repair...")
    finger = load_and_repair(FINGER)
    soft   = load_and_repair(SOFT)

    print("[case41] boolean union ...")
    union = trimesh.boolean.union([finger, soft])
    uv = np.asarray(union.vertices); uf = np.asarray(union.faces)
    print(f"  union: {len(uv)}v {len(uf)}f watertight={union.is_watertight} "
          f"bbox {uv.min(0)} → {uv.max(0)}")

    pd = pv.PolyData(uv, np.hstack([np.full((len(uf), 1), 3), uf]).flatten())
    tgen = tetgen.TetGen(pd)
    out = tgen.tetrahedralize(
        plc=True, quality=True, minratio=2.0,
        fixedvolume=True, maxvolume=MAXVOL,
    )
    verts = np.asarray(out[0], dtype=np.float64)
    tets  = np.asarray(out[1], dtype=np.int32)
    print(f"[case41] tetgen (refined): {len(verts)}v {len(tets)}t "
          f"(maxvolume={MAXVOL})")

    # Drop near-degenerate tets at PLC corners (avoids IPC NaN in barrier)
    def _vol(p0, p1, p2, p3):
        return np.abs(np.einsum('ij,ij->i', np.cross(p1-p0, p2-p0), p3-p0)) / 6.0
    vols = _vol(verts[tets[:,0]], verts[tets[:,1]],
                verts[tets[:,2]], verts[tets[:,3]])
    keep = vols >= MIN_VOL_KEEP
    n_drop = int((~keep).sum())
    if n_drop:
        tets = tets[keep]
        used = np.zeros(len(verts), dtype=bool)
        used[tets.ravel()] = True
        o2n = -np.ones(len(verts), dtype=np.int64)
        o2n[used] = np.arange(int(used.sum()))
        verts = verts[used]
        tets = o2n[tets].astype(np.int32)
        print(f"[case41] dropped {n_drop} degenerate tets → "
              f"{len(verts)}v {len(tets)}t")

    # Label tets by centroid-in-finger test
    centroids = verts[tets].mean(axis=1)
    inside_finger = finger.contains(centroids)
    print(f"[case41] tet labels: {inside_finger.sum()} rigid (inside finger), "
          f"{(~inside_finger).sum()} FEM")

    vertex_region = np.zeros(len(verts), dtype=np.int32)
    rigid_tet_idx = np.where(inside_finger)[0]
    vertex_region[tets[rigid_tet_idx].ravel()] = 1
    n_rigid_v = int(vertex_region.sum())
    print(f"[case41] vertex_region: {n_rigid_v} rigid, {len(verts)-n_rigid_v} FEM "
          f"({100*n_rigid_v/len(verts):.1f}% rigid)")

    tet_all_rigid = vertex_region[tets].min(axis=1) == 1
    rigid_tets_in_unified = tets[tet_all_rigid]
    rigid_v_idx = np.where(vertex_region == 1)[0]
    o2n = -np.ones(len(verts), dtype=np.int64)
    o2n[rigid_v_idx] = np.arange(len(rigid_v_idx))
    rigid_tets_local = o2n[rigid_tets_in_unified].astype(np.int32)
    rigid_verts_local = verts[rigid_v_idx]
    print(f"[case41] rigid sub-mesh: {len(rigid_verts_local)}v "
          f"{len(rigid_tets_local)}t")

    tet_min = vertex_region[tets].min(axis=1)
    tet_max = vertex_region[tets].max(axis=1)
    interface = int(((tet_min == 0) & (tet_max == 1)).sum())
    print(f"[case41] interface tets (mixed rigid+FEM): {interface} "
          f"— these bridge the two regions")

    OUT_UNIFIED.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_UNIFIED, vertices=verts, tets=tets,
             vertex_region=vertex_region,
             density=np.float64(1000.0), young_modulus=np.float64(1e6))
    print(f"unified → {OUT_UNIFIED}")
    meshio.Mesh(points=rigid_verts_local,
                cells=[("tetra", rigid_tets_local)]
               ).write(str(OUT_RIGID), file_format="gmsh22", binary=False)
    print(f"rigid .msh → {OUT_RIGID}")
    np.savez(OUT_REMAP,
             rigid_v_idx=rigid_v_idx.astype(np.int32),
             rigid_local=rigid_verts_local,
             rigid_tets_local=rigid_tets_local)
    print(f"remap → {OUT_REMAP}")
    print("DONE.")


if __name__ == "__main__":
    main()

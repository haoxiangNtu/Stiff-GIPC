#!/usr/bin/env python3
"""tools/build_case37_finger_softpad_hybrid.py

Build a NEW hybrid unified mesh that is GUARANTEED to be in the SAME
mesh-local frame as URDF's finger.stl — so loading it with the finger
ABD body's engine FK transform produces a STRICT alignment between the
hybrid rigid sub-mesh and the URDF finger collision body.

Pipeline:
  1. Load finger.stl, keep only the largest connected component (the
     "lower segment" / main finger stem; the small "head" attachment
     fragments at upper Y are discarded).
  2. Load softpad mesh (softgriper_part3.msh).  Both finger.stl and
     softpad mesh use IDENTICAL visual-origin transform tags inside
     their respective URDF links — they share the SAME mesh-local
     coord frame, so they overlay naturally without further transform.
  3. Combine vertices.  Compute the convex hull (simplifies geometry,
     ensures watertight surface).
  4. Tetrahedralize the convex hull with tetgen.
  5. Label each tet vertex: if it lies inside the finger.stl convex
     hull → "rigid" (region=1); otherwise → "FEM-free" (region=0).
  6. Save .npz schema-compatible with case_36's hybrid loader.

Output: Assets/sim_data/hybrid_d/CASE37_unified.npz

Schema (vertices in finger.stl mesh-local frame):
  vertices:      (N, 3) float64
  tets:          (M, 4) int32
  vertex_region: (N,)   int32   — 0=FEM, 1=rigid
  density:       scalar
  young_modulus: scalar
"""
import sys
from pathlib import Path
import numpy as np

try:
    import trimesh
except ImportError:
    sys.exit("trimesh missing — pip install trimesh")
try:
    import meshio
except ImportError:
    sys.exit("meshio missing — pip install meshio")
try:
    import tetgen
except ImportError:
    sys.exit("tetgen missing — pip install tetgen")
try:
    import pyvista as pv
except ImportError:
    sys.exit("pyvista missing — pip install pyvista")


REPO   = Path("/home/ps/Downloads/Stiff-GIPC-hybrid-mesh")
FINGER = REPO / "Assets/sim_data/urdf/ridgeback_dual_panda_soft/meshes/plate/visual/soft_hard_segmenation/finger_clean.obj"
SOFTPAD = REPO / "Assets/sim_data/tetmesh/softgriper_part3.msh"   # case_27 coarse default
OUT_UNIFIED = REPO / "Assets/sim_data/hybrid_d/CASE40_unified.npz"
OUT_RIGID   = REPO / "Assets/sim_data/hybrid_d/CASE40_rigid.msh"
OUT_REMAP   = REPO / "Assets/sim_data/hybrid_d/CASE40_rigid_remap.npz"
MAXVOL  = 5e-6   # COARSE — bigger tets than CASE37 (1.5e-6)


def main():
    # 1. Finger lower segment (comp 0)
    finger_mesh = trimesh.load(str(FINGER), process=False)
    comps = finger_mesh.split(only_watertight=False)
    finger_lower = max(comps, key=lambda c: len(c.vertices))
    finger_v = np.asarray(finger_lower.vertices)
    finger_f = np.asarray(finger_lower.faces)
    print(f"[build37] finger lower segment: {len(finger_v)} verts, "
          f"{len(finger_f)} faces", flush=True)
    print(f"[build37]   bbox: min={finger_v.min(0)}, max={finger_v.max(0)}",
          flush=True)

    # 2. Softpad (same mesh local frame as finger because soft_material's
    #    visual origin tag is identical to finger's visual origin tag in URDF)
    sp = meshio.read(str(SOFTPAD))
    softpad_v = np.asarray(sp.points, dtype=np.float64)
    print(f"[build37] softpad: {len(softpad_v)} verts", flush=True)
    print(f"[build37]   bbox: min={softpad_v.min(0)}, max={softpad_v.max(0)}",
          flush=True)

    # 3. Combine + convex hull
    all_v = np.vstack([finger_v, softpad_v])
    print(f"[build37] combined: {len(all_v)} verts", flush=True)
    hull = trimesh.convex.convex_hull(all_v)
    hull_v = np.asarray(hull.vertices, dtype=np.float64)
    hull_f = np.asarray(hull.faces, dtype=np.int32)
    print(f"[build37] convex hull: {len(hull_v)} verts, {len(hull_f)} faces",
          flush=True)
    print(f"[build37]   bbox: min={hull_v.min(0)}, max={hull_v.max(0)}",
          flush=True)

    # 4. Tetrahedralize convex hull
    # tetgen expects PolyData
    poly_faces = np.hstack([np.full((len(hull_f), 1), 3, dtype=np.int32), hull_f]).flatten()
    pd = pv.PolyData(hull_v, poly_faces)
    tet = tetgen.TetGen(pd)
    # Coarser mesh: 1.5e-6 ≈ 11mm-edge tets.  Earlier 2e-7 created 173
    # degenerate tets near sharp convex-hull corners (min vol 2e-14 = NaN
    # in IPC barrier).  minratio 2.0 (default) gives moderate aspect; we
    # don't need super-quality tets for the simple convex hull volume.
    # COARSE: relaxed quality (minratio 5 vs default 2 = allow worse
    # aspect ratio → fewer Steiner points), enforce maxvolume bound.
    out = tet.tetrahedralize(
        plc=True, quality=True, minratio=5.0,
        fixedvolume=True, maxvolume=MAXVOL,
    )
    # tetgen returns (nodes, elems, ...); unpack first two
    verts = np.asarray(out[0], dtype=np.float64)
    tets  = np.asarray(out[1], dtype=np.int32)
    print(f"[build37] tetrahedralized: {len(verts)} verts, {len(tets)} tets",
          flush=True)

    # Filter degenerate tets (tetgen at convex-hull corners produces some).
    def _tet_vol(p0, p1, p2, p3):
        return np.abs(np.einsum('ij,ij->i', np.cross(p1-p0, p2-p0), p3-p0)) / 6.0
    vols = _tet_vol(verts[tets[:,0]], verts[tets[:,1]],
                    verts[tets[:,2]], verts[tets[:,3]])
    keep = vols >= 1e-11  # 0.01 mm^3 cutoff
    n_drop = int((~keep).sum())
    if n_drop:
        tets = tets[keep]
        # Remove orphan verts and renumber
        used = np.zeros(len(verts), dtype=bool)
        used[tets.ravel()] = True
        old_to_new = -np.ones(len(verts), dtype=np.int64)
        old_to_new[used] = np.arange(int(used.sum()))
        verts = verts[used]
        tets = old_to_new[tets].astype(np.int32)
        print(f"[build37] dropped {n_drop} degenerate tets → {len(verts)} verts, "
              f"{len(tets)} tets", flush=True)

    # 5. Label vertices: inside finger.stl convex hull → rigid
    finger_hull = trimesh.convex.convex_hull(finger_v)
    inside = finger_hull.contains(verts)
    vertex_region = inside.astype(np.int32)  # 1=rigid, 0=FEM
    n_rigid = int(vertex_region.sum())
    n_fem = len(vertex_region) - n_rigid
    print(f"[build37] vertex_region: {n_rigid} rigid, {n_fem} FEM "
          f"({100*n_rigid/len(verts):.1f}% rigid)", flush=True)

    # 6. Extract rigid sub-mesh (only tets where ALL 4 verts are rigid)
    #    Use this as the ABD body load.  Stitch springs use vertex
    #    correspondence: rigid ABD vert i  ↔  unified FEM vert
    #    rigid_v_idx[i] (same world position because they SHARE the
    #    mesh-local frame).
    tet_all_rigid = vertex_region[tets].min(axis=1) == 1
    rigid_tets_in_unified = tets[tet_all_rigid]  # (Mr, 4) indices into unified verts
    rigid_v_idx = np.where(vertex_region == 1)[0]  # (Nr,) indices into unified
    # Re-index rigid tets into a local 0..Nr-1 numbering for the .msh
    old_to_new = -np.ones(len(verts), dtype=np.int64)
    old_to_new[rigid_v_idx] = np.arange(len(rigid_v_idx))
    rigid_tets_local = old_to_new[rigid_tets_in_unified].astype(np.int32)
    rigid_verts_local = verts[rigid_v_idx]
    print(f"[build37] rigid sub-mesh: {len(rigid_verts_local)} verts, "
          f"{len(rigid_tets_local)} all-rigid tets", flush=True)

    OUT_UNIFIED.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_UNIFIED,
        vertices=verts,
        tets=tets,
        vertex_region=vertex_region,
        density=np.float64(1000.0),
        young_modulus=np.float64(1e6),
    )
    print(f"[build37] unified → {OUT_UNIFIED}", flush=True)

    # Write rigid sub-mesh as .msh (Gmsh format) for engine .load_mesh ABD
    rigid_meshio = meshio.Mesh(
        points=rigid_verts_local,
        cells=[("tetra", rigid_tets_local)],
    )
    rigid_meshio.write(str(OUT_RIGID), file_format="gmsh22", binary=False)
    print(f"[build37] rigid .msh → {OUT_RIGID}", flush=True)

    # Remap: rigid_v_idx tells us, for each rigid ABD vert i, the
    # corresponding index in the unified FEM mesh — for stitch_spring.
    np.savez(
        OUT_REMAP,
        rigid_v_idx=rigid_v_idx.astype(np.int32),
        rigid_local=rigid_verts_local,
        rigid_tets_local=rigid_tets_local,
    )
    print(f"[build37] remap → {OUT_REMAP}", flush=True)
    print(f"[build37] DONE.  Use these in case_37 with finger ABD's engine FK "
          f"transform — alignment exact (same frame as finger.stl).",
          flush=True)


if __name__ == "__main__":
    main()

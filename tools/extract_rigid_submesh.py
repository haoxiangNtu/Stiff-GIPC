#!/usr/bin/env python3
"""extract_rigid_submesh.py — extract rigid-region tets from a unified
hybrid mesh and write them as a standalone GMSH ASCII .msh file suitable
for loading as a Stiff-GIPC ABD body.

Input:  STRATEGY_F_unified.npz with keys:
            vertices       (V,3) float64
            tets           (T,4) int32
            vertex_region  (V,)  int32  (0=FEM, 1=rigid)

Output: rigid_only.msh   — GMSH v2.2 ASCII
        Plus an .npz alongside containing remap arrays for case_31:
            unified_npz    — original path (for reload in case_31)
            rigid_v_idx    — (N_rigid,) int32 — indices into unified vertices
            rigid_local    — (N_rigid,3) float64 — rigid verts in input coords
            rigid_tets_local (T_rigid,4) int32 — tet indices into rigid_local

Usage:
    python tools/extract_rigid_submesh.py <unified.npz> <output_prefix>
    # writes <output_prefix>.msh and <output_prefix>_remap.npz
"""
import sys
import os
import numpy as np


def write_gmsh_v22_ascii(path: str, verts: np.ndarray, tets: np.ndarray) -> None:
    """Write a tet mesh in GMSH ASCII v2.2 format compatible with Stiff-GIPC's
    .msh parser (load_mesh.cpp:482-635).  Element type 4 = linear tetrahedron.

    Note: parser expects 1-based vertex/element indices and the format
        $Nodes
        N
        i x y z
        ...
        $EndNodes
        $Elements
        M
        i 4 2 0 0 v0 v1 v2 v3
        ...
        $EndElements
    """
    n_verts = verts.shape[0]
    n_tets = tets.shape[0]
    with open(path, 'w') as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write(f"$Nodes\n{n_verts}\n")
        for i in range(n_verts):
            v = verts[i]
            f.write(f"{i + 1} {v[0]:.17g} {v[1]:.17g} {v[2]:.17g}\n")
        f.write("$EndNodes\n")
        f.write(f"$Elements\n{n_tets}\n")
        for i in range(n_tets):
            t = tets[i]
            # element_id, type=4 (tet), n_tags=2, tag1=0, tag2=0, v0..v3 (1-based)
            f.write(f"{i + 1} 4 2 0 0 "
                    f"{int(t[0]) + 1} {int(t[1]) + 1} "
                    f"{int(t[2]) + 1} {int(t[3]) + 1}\n")
        f.write("$EndElements\n")


def extract_rigid_submesh(unified_npz_path: str, output_prefix: str) -> dict:
    """Extract rigid sub-mesh, write .msh and remap .npz.  Returns a dict
    of metadata (counts, paths)."""
    data = np.load(unified_npz_path)
    verts = np.ascontiguousarray(data['vertices'], dtype=np.float64)
    tets = np.ascontiguousarray(data['tets'], dtype=np.int32)
    vertex_region = np.ascontiguousarray(data['vertex_region'], dtype=np.int32)

    # Identify rigid-internal tets (all 4 verts in rigid region).
    # Mixed (interface) tets are NOT included in ABD body — they stay in
    # the FEM body with chain-rule pinning interface verts to ABD.
    rigid_tet_mask = np.all(vertex_region[tets] == 1, axis=1)
    n_rigid_tets = int(rigid_tet_mask.sum())
    n_mixed_tets = int(np.any(vertex_region[tets] == 1, axis=1).sum()) - n_rigid_tets
    n_fem_tets = int(np.all(vertex_region[tets] == 0, axis=1).sum())

    rigid_tets_global = tets[rigid_tet_mask]            # (T_r, 4) into unified
    rigid_v_idx = np.unique(rigid_tets_global.flatten()).astype(np.int32)
    rigid_verts = verts[rigid_v_idx]                    # (N_r, 3)

    # Remap rigid_tets_global → local [0..N_r) indices.
    remap = -np.ones(len(verts), dtype=np.int32)
    remap[rigid_v_idx] = np.arange(len(rigid_v_idx), dtype=np.int32)
    rigid_tets_local = remap[rigid_tets_global].astype(np.int32)

    # Sanity: every rigid_tet vertex must be in rigid region.
    for r in rigid_v_idx:
        assert vertex_region[r] == 1, f"vertex {r} mapped as rigid but region={vertex_region[r]}"

    out_msh = output_prefix + ".msh"
    out_npz = output_prefix + "_remap.npz"

    write_gmsh_v22_ascii(out_msh, rigid_verts, rigid_tets_local)
    np.savez(out_npz,
             unified_npz=np.array(unified_npz_path, dtype=object),
             rigid_v_idx=rigid_v_idx,
             rigid_local=rigid_verts,
             rigid_tets_local=rigid_tets_local)

    info = dict(
        n_unified_verts=int(len(verts)),
        n_unified_tets=int(len(tets)),
        n_rigid_verts=int(len(rigid_v_idx)),
        n_rigid_tets=n_rigid_tets,
        n_mixed_tets=n_mixed_tets,        # kept in FEM (interface tets)
        n_fem_tets=n_fem_tets,
        out_msh=out_msh,
        out_npz=out_npz,
    )
    return info


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <unified.npz> <output_prefix>")
        sys.exit(1)

    unified_npz = sys.argv[1]
    output_prefix = sys.argv[2]

    if not os.path.exists(unified_npz):
        print(f"Error: {unified_npz} does not exist", flush=True)
        sys.exit(1)

    os.makedirs(os.path.dirname(output_prefix) or ".", exist_ok=True)

    info = extract_rigid_submesh(unified_npz, output_prefix)
    print("[extract_rigid_submesh] done", flush=True)
    for k, v in info.items():
        print(f"  {k} = {v}", flush=True)


if __name__ == "__main__":
    main()

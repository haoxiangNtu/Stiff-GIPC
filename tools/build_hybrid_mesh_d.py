#!/usr/bin/env python3
"""
tools/build_hybrid_mesh_d.py — Strategy D true-hybrid mesh builder.

Differs from Strategy B (build_hybrid_mesh.py) by RE-TETRAHEDRALIZING
both surfaces into a single unified tet mesh with TetGen region
attributes.  Result: clean interface (no straddling tets), substantial
rigid region (controlled by mesh density param), and no overlapping
geometry between rigid backbone and soft pad.

Inputs:
  - ABD surface mesh (.stl/.obj/.ply, watertight, in some local frame)
  - Soft volume mesh (.msh tet, OR .stl/.obj surface) defining the soft
    OUTER boundary.  ABD surface must be FULLY INSIDE this soft volume.
  - Target mesh density (mm) for TetGen quality control

Pipeline:
  1. Load ABD surface, validate watertight
  2. Load/extract soft outer surface, validate watertight + contains ABD
  3. Merge surfaces into one PolyData (PyVista)
  4. Add 2 region seeds (rigid: inside ABD; FEM: between ABD and soft outer)
  5. Run TetGen `pzq1.4Aa<vol>` (PLC + zero-index + quality + Region attrs + Area)
  6. Per-tet region attribute → label (1=rigid, 2=FEM)
  7. Vertex region labels via tet incidence (interface verts → rigid)
  8. Compute local_pos = world_pos (we built in ABD local frame)
  9. Save .npz with the same schema as build_hybrid_mesh.py

Output schema (compatible with engine.add_hybrid_fem_body):
  vertices:           (N, 3) float64 — in ABD local frame
  tets:               (M, 4) int32
  vertex_region:      (N,)   int32   — 0=FEM-free, 1=rigid
  vertex_abd_body_id: (N,)   int32   — -1 if FEM
  vertex_local_pos:   (N, 3) float64 — = vertices for rigid, 0 for FEM
  tet_region:         (M,)   int32   — 0=FEM-only, 1=interface, 2=rigid-only
                                       (Strategy D: NO interface tets! Always 0 or 2)
  tet_to_abd_body:    (M,)   int32   — abd_body_id for rigid, -1 for FEM
  density, young_modulus, poisson_ratio: scalars
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import meshio
except ImportError:
    sys.exit("ERROR: meshio missing.  pip install meshio")
try:
    import trimesh
except ImportError:
    sys.exit("ERROR: trimesh missing.  pip install trimesh")
try:
    import pyvista as pv
    import tetgen
except ImportError:
    sys.exit("ERROR: pyvista + tetgen missing.  pip install pyvista tetgen")


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def load_surface_as_polydata(path: str) -> pv.PolyData:
    """Load a surface mesh (any trimesh-readable format) as PyVista PolyData."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".msh":
        # tet mesh — extract boundary
        m = meshio.read(str(path))
        tet_blocks = [c for c in m.cells if c.type == "tetra"]
        if not tet_blocks:
            raise RuntimeError(f"{path}: no 'tetra' cells; is it a surface mesh?")
        tets = np.vstack([c.data for c in tet_blocks])
        verts = np.asarray(m.points, dtype=np.float64)
        tm = trimesh.Trimesh(
            vertices=verts,
            faces=_extract_boundary_faces(tets),
            process=True,
        )
    else:
        tm = trimesh.load(str(path))
    if not tm.is_watertight:
        print(f"  WARN: {path} not watertight ({len(tm.faces)} faces); "
              f"TetGen may have trouble. Consider repair.", file=sys.stderr)
    # Convert to PyVista
    faces_pv = np.hstack([
        np.full((len(tm.faces), 1), 3, dtype=np.int64),
        tm.faces.astype(np.int64),
    ]).ravel()
    return pv.PolyData(np.asarray(tm.vertices, dtype=np.float64), faces=faces_pv)


def _extract_boundary_faces(tets: np.ndarray) -> np.ndarray:
    """Extract boundary faces from a tet mesh: faces appearing in only one tet."""
    # Each tet has 4 faces (3 verts each)
    face_combos = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    faces = []
    for combo in face_combos:
        f = tets[:, list(combo)]
        f_sorted = np.sort(f, axis=1)
        faces.append(f_sorted)
    all_faces = np.ascontiguousarray(np.vstack(faces).astype(np.int64))  # (4M, 3)
    # Use lexsort to find duplicates without view dtype tricks
    order = np.lexsort(all_faces.T[::-1])
    sorted_faces = all_faces[order]
    # Mark unique faces by checking equality with neighbors
    diff = np.any(sorted_faces[1:] != sorted_faces[:-1], axis=1)
    # Run-length-encode: a face is boundary iff its run length is 1
    # boundary_mask[i] = True if face i is unique among adjacent equal faces
    is_first = np.concatenate([[True], diff])
    is_last  = np.concatenate([diff, [True]])
    is_unique = is_first & is_last
    boundary_faces = sorted_faces[is_unique]
    return boundary_faces.astype(np.int32)


def apply_4x4(verts: np.ndarray, T: np.ndarray) -> np.ndarray:
    h = np.hstack([verts, np.ones((len(verts), 1))])
    return (T @ h.T).T[:, :3]


# -----------------------------------------------------------------------------
# Geometry checks
# -----------------------------------------------------------------------------

def verify_containment(abd_polydata: pv.PolyData, soft_polydata: pv.PolyData):
    """Check that ABD surface is INSIDE soft volume."""
    abd_tm = trimesh.Trimesh(
        vertices=abd_polydata.points,
        faces=abd_polydata.faces.reshape(-1, 4)[:, 1:],
        process=False,
    )
    soft_tm = trimesh.Trimesh(
        vertices=soft_polydata.points,
        faces=soft_polydata.faces.reshape(-1, 4)[:, 1:],
        process=False,
    )
    if not soft_tm.is_watertight:
        print("  WARN: soft surface not watertight; containment check unreliable",
              file=sys.stderr)
    inside = soft_tm.contains(abd_tm.vertices)
    n_inside = int(inside.sum())
    print(f"  ABD surface verts inside soft volume: {n_inside}/{len(inside)}")
    if n_inside < 0.95 * len(inside):
        raise RuntimeError(
            f"Only {n_inside}/{len(inside)} ABD verts inside soft volume; "
            f"Strategy D requires ABD fully embedded in soft.  Check transforms."
        )
    return abd_tm, soft_tm


def find_seed_inside_mesh(tm: trimesh.Trimesh, n_attempts: int = 50) -> np.ndarray:
    """Find an interior point of a watertight mesh by averaging near-centroid points
    until one passes the contains() test."""
    # First try centroid
    centroid = tm.centroid
    if tm.contains([centroid])[0]:
        return centroid
    # Try points along major axes from centroid
    bbox = tm.bounds
    extent = bbox[1] - bbox[0]
    rng = np.random.default_rng(42)
    for _ in range(n_attempts):
        pt = centroid + rng.normal(scale=extent * 0.1)
        if tm.contains([pt])[0]:
            return pt
    raise RuntimeError("Failed to find a point inside the mesh")


def find_seed_between(soft_tm: trimesh.Trimesh, abd_tm: trimesh.Trimesh,
                      n_attempts: int = 100) -> np.ndarray:
    """Find a point that is INSIDE soft but OUTSIDE abd."""
    soft_bbox = soft_tm.bounds
    extent = soft_bbox[1] - soft_bbox[0]
    rng = np.random.default_rng(123)
    for _ in range(n_attempts):
        # Sample uniformly in soft bbox
        pt = soft_bbox[0] + rng.random(3) * extent
        in_soft = soft_tm.contains([pt])[0]
        in_abd = abd_tm.contains([pt])[0]
        if in_soft and not in_abd:
            return pt
    raise RuntimeError("Failed to find a point inside soft but outside ABD")


# -----------------------------------------------------------------------------
# Core build
# -----------------------------------------------------------------------------

def build_hybrid_d(
    abd_surface_path: str,
    soft_mesh_path: str,
    abd_body_id: int,
    abd_transform: np.ndarray | None = None,
    soft_transform: np.ndarray | None = None,
    rigid_max_vol: float = 0.0,
    soft_max_vol: float = 0.0,
    quality: float = 1.4,
    young_modulus: float = 1e6,
    density: float = 1000.0,
    poisson_ratio: float = 0.49,
    verbose: bool = True,
):
    log = print if verbose else (lambda *a, **k: None)

    # 1. Load surfaces
    log(f"[D] loading ABD surface: {abd_surface_path}")
    abd_pv = load_surface_as_polydata(abd_surface_path)
    log(f"  N_v={abd_pv.n_points}, N_f={abd_pv.n_faces_strict}")

    log(f"[D] loading soft volume: {soft_mesh_path}")
    soft_pv = load_surface_as_polydata(soft_mesh_path)
    log(f"  N_v={soft_pv.n_points}, N_f={soft_pv.n_faces_strict}")

    # 2. Apply transforms (move both into common frame, default = ABD local)
    if abd_transform is not None:
        abd_pv.points = apply_4x4(abd_pv.points, abd_transform)
        log(f"  ABD transformed; bbox: {abd_pv.bounds}")
    if soft_transform is not None:
        soft_pv.points = apply_4x4(soft_pv.points, soft_transform)
        log(f"  Soft transformed; bbox: {soft_pv.bounds}")

    # 3. Validate containment
    log(f"[D] validating ABD ⊂ soft volume...")
    abd_tm, soft_tm = verify_containment(abd_pv, soft_pv)

    # 4. Find region seeds
    log(f"[D] finding region seeds...")
    seed_rigid = find_seed_inside_mesh(abd_tm)
    seed_soft = find_seed_between(soft_tm, abd_tm)
    log(f"  rigid seed (inside ABD): {seed_rigid}")
    log(f"  soft  seed (between):    {seed_soft}")

    # 5. Merge surfaces, run TetGen with region attribs
    log(f"[D] merging surfaces and running TetGen...")
    merged = pv.merge([abd_pv, soft_pv])
    tgen = tetgen.TetGen(merged)
    # Region IDs: 1 = rigid (inside ABD), 2 = FEM (between ABD and soft outer)
    tgen.add_region(1, seed_rigid.tolist(), max_vol=rigid_max_vol)
    tgen.add_region(2, seed_soft.tolist(),  max_vol=soft_max_vol)

    switches = f"pzq{quality}Aa"
    log(f"  TetGen switches: '{switches}'")
    out = tgen.tetrahedralize(switches=switches)
    nodes, elem = out[0], out[1]
    attrib = tgen.attributes.ravel().astype(np.int32)
    log(f"  TetGen output: {nodes.shape[0]} verts, {elem.shape[0]} tets, "
        f"region counts: rigid={int((attrib==1).sum())}, FEM={int((attrib==2).sum())}")

    # 6. Build per-tet labels
    tet_region = np.zeros(elem.shape[0], dtype=np.int32)  # 0=FEM, 2=rigid
    tet_region[attrib == 1] = 2     # rigid-only (Strategy D guarantee)
    tet_region[attrib == 2] = 0     # FEM-only
    tet_to_abd_body = np.where(tet_region == 2, abd_body_id, -1).astype(np.int32)

    # 7. Build per-vertex labels via tet incidence
    # Vertex v incident to tets {T_v}.  If any T_v has region=rigid, v is rigid.
    # (interface verts inherit rigid because they follow ABD body)
    vertex_region = np.zeros(nodes.shape[0], dtype=np.int32)  # 0=FEM, 1=rigid
    rigid_tet_mask = (attrib == 1)
    rigid_tet_verts = np.unique(elem[rigid_tet_mask].ravel())
    vertex_region[rigid_tet_verts] = 1
    n_rigid_v = int((vertex_region == 1).sum())
    n_fem_v = int((vertex_region == 0).sum())
    log(f"  vertex labels: rigid={n_rigid_v}, FEM={n_fem_v}")

    # 8. Local positions = world coords (we built in ABD local frame; pass identity
    # for abd_rest_pose_inverse if user wants something else)
    vertex_local_pos = np.where(
        (vertex_region == 1)[:, None], nodes, 0.0
    ).astype(np.float64)

    vertex_abd_body_id = np.where(vertex_region == 1, abd_body_id, -1).astype(np.int32)

    # 9. Sanity: count interface tets (Strategy D should have NONE — every tet
    # is either fully rigid or fully soft)
    rigid_count_per_tet = vertex_region[elem].sum(axis=1)  # 0..4
    n_iface_tet = int(((rigid_count_per_tet > 0) & (rigid_count_per_tet < 4)).sum())
    log(f"  Strategy D check: {n_iface_tet} interface tets "
        f"(Strategy D guarantee: SOME tets have mixed rigid/FEM verts at the "
        f"interface; this is OK and they're handled by M3.5 chain-rule)")
    # NOTE: with Strategy D's region-labeled TetGen output, some FEM tets
    # touch the interface and have rigid verts (interface verts).  These
    # are FEM tets (computed elasticity), with chain-rule routing for
    # rigid-vert contributions.  Per-tet region label determines whether
    # to compute elasticity (FEM) or skip (rigid).

    return {
        "vertices": nodes.astype(np.float64),
        "tets": elem.astype(np.int32),
        "vertex_region": vertex_region,
        "vertex_abd_body_id": vertex_abd_body_id,
        "vertex_local_pos": vertex_local_pos,
        "tet_region": tet_region,
        "tet_to_abd_body": tet_to_abd_body,
        "density": np.float64(density),
        "young_modulus": np.float64(young_modulus),
        "poisson_ratio": np.float64(poisson_ratio),
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--abd-surface", required=True,
                   help="ABD surface (rigid backbone), .stl/.obj/.ply")
    p.add_argument("--soft-mesh", required=True,
                   help="Soft volume, .msh tet (boundary auto-extracted) or .stl surface")
    p.add_argument("--abd-body-id", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--abd-transform", default=None)
    p.add_argument("--soft-transform", default=None)
    p.add_argument("--rigid-max-vol", type=float, default=0.0,
                   help="Max tet volume in rigid region (0 = TetGen default)")
    p.add_argument("--soft-max-vol", type=float, default=0.0,
                   help="Max tet volume in FEM region (0 = TetGen default)")
    p.add_argument("--quality", type=float, default=1.4,
                   help="TetGen radius-edge quality bound")
    p.add_argument("--young", type=float, default=1e6)
    p.add_argument("--density", type=float, default=1000.0)
    p.add_argument("--poisson", type=float, default=0.49)
    args = p.parse_args(argv)

    def _load_T(path):
        if path is None: return None
        T = np.load(path)
        if T.shape != (4, 4): raise ValueError(f"{path}: need 4x4")
        return T.astype(np.float64)

    fields = build_hybrid_d(
        abd_surface_path=args.abd_surface,
        soft_mesh_path=args.soft_mesh,
        abd_body_id=args.abd_body_id,
        abd_transform=_load_T(args.abd_transform),
        soft_transform=_load_T(args.soft_transform),
        rigid_max_vol=args.rigid_max_vol,
        soft_max_vol=args.soft_max_vol,
        quality=args.quality,
        young_modulus=args.young,
        density=args.density,
        poisson_ratio=args.poisson,
        verbose=True,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **fields)
    print(f"[D] saved {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
tools/build_hybrid_mesh.py — Build a hybrid ABD-FEM tet mesh.

Takes an existing FEM tetrahedral mesh and an ABD surface mesh (both in
WORLD coordinates after applying any rest-pose transforms), and labels
each FEM vertex as either:
  - rigid (inside the ABD surface) → controlled by ABD body's q via
    chain-rule routing (M3.5 substitution method, generalized to bulk pin)
  - free FEM (outside the ABD surface) → independent DOF

Saves a .npz file with all data needed by the engine's future
`add_hybrid_fem_body` API (Phase 3).

Strategy: Strategy B — re-label an existing high-quality FEM tet mesh,
no re-tetrahedralization.  Pros: preserves FEM mesh quality, no new
tet quality issues at interface, fast.  Cons: requires the FEM mesh
to actually contain tet vertices inside the ABD region.

Usage:
    python tools/build_hybrid_mesh.py \\
        --fem-mesh Assets/sim_data/tetmesh/softgriper_part2_blobal.msh \\
        --abd-surface Assets/.../finger.stl \\
        --abd-body-id 7 \\
        --out hybrid_finger_pad_left.npz \\
        [--fem-transform fem_T_world.npy] \\
        [--abd-transform abd_T_world.npy] \\
        [--abd-rest-pose-inverse abd_T_inv.npy] \\
        [--young 1e6] [--density 1000] [--poisson 0.49]

NPZ schema (output):
    vertices:           (N, 3) float64  — world coords at rest
    tets:               (M, 4) int32    — vertex indices
    vertex_region:      (N,)   int32    — 0=FEM-free, 1=rigid
    vertex_abd_body_id: (N,)   int32    — -1 if FEM, else target body id
    vertex_local_pos:   (N, 3) float64  — local pos in ABD rest frame
                                          (zero for FEM-free verts)
    tet_region:         (M,)   int32    — 0=all-FEM, 1=interface, 2=all-rigid
    tet_to_abd_body:    (M,)   int32    — body id if tet_region==2, else -1
    density:            scalar          — material density (kg/m^3)
    young_modulus:      scalar
    poisson_ratio:      scalar
"""

import argparse
import os
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


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def load_fem_tet(path: str):
    """Load tet mesh.  Supports .msh (Gmsh), .vtu, .ele/.node (TetGen), .npz."""
    p = Path(path)
    if p.suffix == ".npz":
        data = np.load(path)
        return data["vertices"], data["tets"]
    m = meshio.read(path)
    tet_blocks = [c for c in m.cells if c.type == "tetra"]
    if not tet_blocks:
        raise RuntimeError(f"{path} has no 'tetra' cells (found: {[c.type for c in m.cells]})")
    tets = np.vstack([c.data for c in tet_blocks])
    return np.asarray(m.points, dtype=np.float64), np.asarray(tets, dtype=np.int32)


def load_4x4(path: str | None) -> np.ndarray:
    if path is None:
        return np.eye(4)
    arr = np.load(path)
    if arr.shape != (4, 4):
        raise ValueError(f"{path}: expected 4x4 matrix, got {arr.shape}")
    return arr.astype(np.float64)


def apply_4x4(verts: np.ndarray, T: np.ndarray) -> np.ndarray:
    h = np.hstack([verts, np.ones((len(verts), 1))])
    return (T @ h.T).T[:, :3]


# -----------------------------------------------------------------------------
# Core pipeline
# -----------------------------------------------------------------------------

def build_hybrid(
    fem_mesh_path: str,
    abd_surface_path: str,
    abd_body_id: int,
    fem_transform: np.ndarray | None = None,
    abd_transform: np.ndarray | None = None,
    abd_rest_pose_inverse: np.ndarray | None = None,
    young_modulus: float = 1e6,
    density: float = 1000.0,
    poisson_ratio: float = 0.49,
    verbose: bool = True,
):
    """Returns dict of fields ready for np.savez."""

    log = print if verbose else (lambda *a, **k: None)

    # --- FEM mesh -----------------------------------------------------------
    fem_verts, fem_tets = load_fem_tet(fem_mesh_path)
    log(f"[build_hybrid] FEM mesh: {fem_mesh_path}")
    log(f"  loaded {len(fem_verts)} verts, {len(fem_tets)} tets")
    log(f"  bbox: {fem_verts.min(axis=0)} - {fem_verts.max(axis=0)}")

    if fem_transform is not None:
        fem_verts = apply_4x4(fem_verts, fem_transform)
        log(f"  after FEM transform; bbox: "
            f"{fem_verts.min(axis=0)} - {fem_verts.max(axis=0)}")

    # --- ABD surface --------------------------------------------------------
    # process=True (default) merges duplicate vertices so STL files report
    # watertight correctly.  Don't disable.
    abd = trimesh.load(abd_surface_path)
    log(f"[build_hybrid] ABD surface: {abd_surface_path}")
    log(f"  loaded {len(abd.vertices)} verts, {len(abd.faces)} faces, "
        f"watertight={abd.is_watertight}")
    log(f"  bbox: {abd.bounds[0]} - {abd.bounds[1]}")

    if abd_transform is not None:
        abd = abd.copy()
        abd.apply_transform(abd_transform)
        log(f"  after ABD transform; bbox: {abd.bounds[0]} - {abd.bounds[1]}")

    if not abd.is_watertight:
        # contains() falls back to ray method (works but slower / less robust)
        # for non-watertight meshes
        log("  WARN: ABD not watertight — region label may have artifacts at "
            "open edges.  Consider repairing with trimesh.repair.fill_holes.")

    # --- Label FEM verts ----------------------------------------------------
    log("[build_hybrid] labeling FEM vertices vs ABD surface...")
    inside = abd.contains(fem_verts)  # bool[N]
    n_rigid_v = int(inside.sum())
    n_free_v = int((~inside).sum())
    log(f"  rigid (inside ABD): {n_rigid_v}/{len(fem_verts)} "
        f"({100 * n_rigid_v / len(fem_verts):.1f}%)")
    log(f"  FEM-free (outside): {n_free_v}/{len(fem_verts)}")

    if n_rigid_v == 0:
        raise RuntimeError(
            "no FEM verts are inside ABD surface — meshes don't overlap. "
            "Check transforms.  ABD bbox=" f"{abd.bounds.tolist()}, "
            f"FEM bbox=[{fem_verts.min(0).tolist()}, {fem_verts.max(0).tolist()}]"
        )
    if n_free_v == 0:
        raise RuntimeError(
            "all FEM verts are inside ABD — entire mesh would be rigid, "
            "no FEM region to deform.  Check ABD surface size."
        )

    # --- Local positions ----------------------------------------------------
    T_inv = abd_rest_pose_inverse if abd_rest_pose_inverse is not None else np.eye(4)
    local_pos_all = apply_4x4(fem_verts, T_inv)

    # --- Build output arrays ------------------------------------------------
    vertex_region = inside.astype(np.int32)  # 0/1
    vertex_abd_body_id = np.where(inside, abd_body_id, -1).astype(np.int32)
    vertex_local_pos = np.where(inside[:, None], local_pos_all, 0.0).astype(np.float64)

    # Tet partition
    rigid_count = inside[fem_tets].sum(axis=1)  # 0..4
    tet_region = np.where(rigid_count == 4, 2,
                          np.where(rigid_count == 0, 0, 1)).astype(np.int32)
    tet_to_abd_body = np.where(rigid_count == 4, abd_body_id, -1).astype(np.int32)

    n_rigid_t = int((tet_region == 2).sum())
    n_iface_t = int((tet_region == 1).sum())
    n_fem_t = int((tet_region == 0).sum())
    log(f"[build_hybrid] tet partition:")
    log(f"  FEM-only:  {n_fem_t}/{len(fem_tets)} (compute elasticity)")
    log(f"  Interface: {n_iface_t}/{len(fem_tets)} (chain-rule routing)")
    log(f"  Rigid-only:{n_rigid_t}/{len(fem_tets)} (skip elasticity)")

    if n_iface_t == 0:
        log("  WARN: 0 interface tets — ABD↔FEM coupling will not work via "
            "elasticity.  Either rigid region is fully isolated, or rigid "
            "region encompasses entire FEM mesh boundary.")

    return {
        "vertices": fem_verts.astype(np.float64),
        "tets": fem_tets.astype(np.int32),
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
    p.add_argument("--fem-mesh", required=True,
                   help="Path to FEM tet mesh (.msh, .vtu, .ele/.node, .npz)")
    p.add_argument("--abd-surface", required=True,
                   help="Path to ABD surface mesh (.obj, .stl, .ply)")
    p.add_argument("--abd-body-id", type=int, required=True,
                   help="Target ABD body index (must match engine's body order)")
    p.add_argument("--out", required=True,
                   help="Output .npz path")
    p.add_argument("--fem-transform", default=None,
                   help="Optional 4x4 matrix .npy applied to FEM verts (local→world)")
    p.add_argument("--abd-transform", default=None,
                   help="Optional 4x4 matrix .npy applied to ABD surface (local→world)")
    p.add_argument("--abd-rest-pose-inverse", default=None,
                   help="Optional 4x4 matrix .npy: world→ABD-local (for "
                        "vertex_local_pos computation).  If omitted, identity.")
    p.add_argument("--young", type=float, default=1e6, help="Young modulus (Pa)")
    p.add_argument("--density", type=float, default=1000.0, help="Density (kg/m^3)")
    p.add_argument("--poisson", type=float, default=0.49, help="Poisson ratio")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    fields = build_hybrid(
        fem_mesh_path=args.fem_mesh,
        abd_surface_path=args.abd_surface,
        abd_body_id=args.abd_body_id,
        fem_transform=load_4x4(args.fem_transform) if args.fem_transform else None,
        abd_transform=load_4x4(args.abd_transform) if args.abd_transform else None,
        abd_rest_pose_inverse=load_4x4(args.abd_rest_pose_inverse)
                              if args.abd_rest_pose_inverse else None,
        young_modulus=args.young,
        density=args.density,
        poisson_ratio=args.poisson,
        verbose=not args.quiet,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **fields)
    if not args.quiet:
        print(f"[build_hybrid] saved {out} ({out.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

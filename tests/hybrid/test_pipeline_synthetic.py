"""
Phase 1 sanity test: run build_hybrid_mesh.py on a fully-known synthetic
fixture and verify region labels make geometric sense.

Fixture: a 1m^3 sphere (FEM) with a cylinder (ABD rod) of radius 0.3m,
height 0.5m embedded along the z-axis at the center.

Expected:
  - some FEM verts inside the cylinder → rigid
  - most FEM verts outside → free
  - non-zero interface tets
  - tet partition counts add up to total tet count
"""

import sys
from pathlib import Path

import numpy as np
import trimesh

# Repo root on path so we can import the tool
THIS = Path(__file__).resolve()
REPO = THIS.parents[2]
sys.path.insert(0, str(REPO))

from tools.build_hybrid_mesh import build_hybrid


def make_synthetic_fixture(tmpdir: Path):
    """Generate sphere FEM tet + cylinder ABD surface."""
    # Use trimesh primitives, then tet via tetgen
    sphere_surf = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    cyl_surf = trimesh.creation.cylinder(radius=0.3, height=0.5, sections=32)

    # Tetrahedralize the sphere using PyVista's tetgen wrapper
    import pyvista as pv
    import tetgen

    pv_sphere = pv.wrap(sphere_surf)
    tet = tetgen.TetGen(pv_sphere)
    tet.tetrahedralize(quality=True, mindihedral=20, minratio=1.5)
    fem_grid = tet.grid
    fem_verts = np.asarray(fem_grid.points)
    # PyVista stores tets via 'cells' (each preceded by 4); use 'cells_dict'.
    cells_dict = fem_grid.cells_dict
    # tet cell type = 10 in VTK
    fem_tets = cells_dict[10].astype(np.int32)
    print(f"  synthetic FEM sphere: {len(fem_verts)} verts, {len(fem_tets)} tets")

    # Save FEM as .msh-compatible (or just write a .npz fixture)
    fem_path = tmpdir / "fem_sphere.npz"
    np.savez(fem_path, vertices=fem_verts, tets=fem_tets)

    # Save ABD cylinder as .obj
    abd_path = tmpdir / "abd_cyl.obj"
    cyl_surf.export(str(abd_path))

    return str(fem_path), str(abd_path), fem_verts, fem_tets


def main():
    import tempfile

    tmpdir = Path(tempfile.mkdtemp(prefix="hybrid_test_"))
    print(f"[test] tmpdir: {tmpdir}")

    fem_path, abd_path, fem_verts, fem_tets = make_synthetic_fixture(tmpdir)

    print(f"[test] running build_hybrid pipeline...")
    fields = build_hybrid(
        fem_mesh_path=fem_path,
        abd_surface_path=abd_path,
        abd_body_id=42,
        verbose=True,
    )

    # --- Sanity checks ------------------------------------------------------
    print("\n[test] verifying output...")
    N = len(fem_verts)
    M = len(fem_tets)

    # Schema
    assert fields["vertices"].shape == (N, 3), "vertices shape"
    assert fields["tets"].shape == (M, 4), "tets shape"
    assert fields["vertex_region"].shape == (N,), "vertex_region shape"
    assert fields["vertex_abd_body_id"].shape == (N,), "vertex_abd_body_id shape"
    assert fields["vertex_local_pos"].shape == (N, 3), "vertex_local_pos shape"
    assert fields["tet_region"].shape == (M,), "tet_region shape"
    assert fields["tet_to_abd_body"].shape == (M,), "tet_to_abd_body shape"

    # Geometric sanity
    cyl_radius = 0.3
    cyl_half_h = 0.25
    rigid = fields["vertex_region"] == 1
    n_rigid = int(rigid.sum())
    print(f"  rigid count: {n_rigid}/{N}")
    assert n_rigid > 0, "expected some rigid verts (sphere overlaps cylinder)"
    assert n_rigid < N, "expected some FEM verts outside cylinder"

    # All rigid verts must be inside cylinder geometry
    rigid_pos = fields["vertices"][rigid]
    r_cyl = np.sqrt(rigid_pos[:, 0] ** 2 + rigid_pos[:, 1] ** 2)
    z_cyl = np.abs(rigid_pos[:, 2])
    # Allow small slack at boundary due to winding number near surface
    assert np.all(r_cyl < cyl_radius + 0.05), \
        f"rigid verts outside cyl radius: max r={r_cyl.max():.3f}"
    assert np.all(z_cyl < cyl_half_h + 0.05), \
        f"rigid verts outside cyl height: max |z|={z_cyl.max():.3f}"

    # vertex_abd_body_id consistency
    assert np.all(fields["vertex_abd_body_id"][rigid] == 42)
    assert np.all(fields["vertex_abd_body_id"][~rigid] == -1)

    # Local pos: identity transform → local_pos == world_pos for rigid verts
    np.testing.assert_allclose(
        fields["vertex_local_pos"][rigid], fields["vertices"][rigid],
        err_msg="local_pos should equal world_pos under identity transform")
    # For FEM verts, local_pos is zero
    assert np.all(fields["vertex_local_pos"][~rigid] == 0.0)

    # Tet partition consistency
    n_fem_t = int((fields["tet_region"] == 0).sum())
    n_iface_t = int((fields["tet_region"] == 1).sum())
    n_rigid_t = int((fields["tet_region"] == 2).sum())
    print(f"  tet partition: FEM={n_fem_t}, interface={n_iface_t}, rigid={n_rigid_t}")
    assert n_fem_t + n_iface_t + n_rigid_t == M, "tet counts must sum to total"
    assert n_iface_t > 0, "expected non-zero interface tets (sphere/cyl overlap)"

    # Cross-check: tet_region == 2 iff all 4 verts are rigid
    rigid_count = rigid[fields["tets"]].sum(axis=1)
    np.testing.assert_array_equal(
        fields["tet_region"] == 2, rigid_count == 4,
        err_msg="tet_region 2 must mean all 4 verts rigid")
    np.testing.assert_array_equal(
        fields["tet_region"] == 0, rigid_count == 0,
        err_msg="tet_region 0 must mean all 4 verts FEM")

    # tet_to_abd_body consistency
    assert np.all(fields["tet_to_abd_body"][fields["tet_region"] == 2] == 42)
    assert np.all(fields["tet_to_abd_body"][fields["tet_region"] != 2] == -1)

    # Material params
    assert fields["young_modulus"] == 1e6
    assert fields["density"] == 1000.0
    assert fields["poisson_ratio"] == 0.49

    print("\n✅ ALL CHECKS PASSED — Phase 1 build_hybrid_mesh.py is correct.")
    print(f"   Sphere FEM ({N}v/{M}t) + cylinder ABD → "
          f"{n_rigid}v rigid, {n_iface_t}t interface, {n_rigid_t}t internal-rigid")
    return 0


if __name__ == "__main__":
    sys.exit(main())

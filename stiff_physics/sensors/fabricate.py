"""Fabricate a gel pad: STL surface -> tet mesh + masks + markers.

Re-implementation of Taccel's ``examples/fabricate_sensor.py``, parameterised and
writing the same on-disk format (``pad_maxv=<v>.vtk`` + ``pad_maxv=<v>.pkl``), so
anything produced here loads through ``SensorAsset.from_taccel_fabrication``
unchanged and sits next to Taccel's own pads.

Differences from upstream, all deliberate:
  * tetgen instead of meshpy (wheels exist; meshpy needs a source build),
  * the masks and the marker grid are arguments rather than module-level globals,
  * the metadata pickle holds plain numpy arrays, so it loads without PyVista
    (upstream's pickles carry ``pyvista_ndarray`` - see pkl_compat.py),
  * marker barycentric coordinates are solved vectorised over all markers rather
    than by a Python double loop over every surface triangle.

Why refabricate at all: the finest pad Taccel ships has ~0.8 mm coat triangles,
which is ~10 px at 400x400 / 0.079375 mm, so a rendered contact shows the
triangulation. Image-fidelity work needs a finer coat than that.
"""

from __future__ import annotations

import os
import os.path as osp
import pickle
from dataclasses import dataclass

import numpy as np


@dataclass
class FabricationSpec:
    """Everything that defines a gel pad, in metres."""

    stl_path: str
    max_volume: float = 1e-10  # tetgen -a, m^3
    coat_z_max: float = 1e-4  # verts below this local z are the sensing surface
    stick_z_min: float | None = None  # verts above this are glued to the carrier
    marker_extent_mm: float = 13.3  # half-extent of the marker grid
    marker_grid: int = 8  # markers per side (8 -> 8x8 = 64)
    out_dir: str | None = None  # default: next to the STL
    name: str = "pad"


def _tetrahedralise(verts: np.ndarray, faces: np.ndarray, max_volume: float,
                    quality: float = 1.414, verbose: bool = False):
    import tetgen

    tet = tetgen.TetGen(verts, faces)
    # `maxvolume` alone is silently ignored -- both it and `fixedvolume` map to
    # tetgen's -a switch and the volume constraint is only armed when
    # fixedvolume=True. Likewise `minratio` needs quality=True to arm -q.
    # (Symptom without them: every max_volume gives the same ~400-tet mesh,
    # i.e. the bare surface triangulation.)
    tet.tetrahedralize(order=1, quality=True, mindihedral=10.0, minratio=quality,
                       fixedvolume=True, maxvolume=max_volume, verbose=int(verbose))
    return np.asarray(tet.node, dtype=np.float64), np.asarray(tet.elem, dtype=np.int32)


def _boundary_faces(tets: np.ndarray, points: np.ndarray) -> np.ndarray:
    from .vtk_io import boundary_faces

    return boundary_faces(tets, points)


def _marker_barycentric(points: np.ndarray, coat_tris: np.ndarray,
                        targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Snap each target to the nearest coat triangle; return (tri verts, weights).

    Vectorised point-in-triangle projection over (n_markers x n_tris); the coat is
    a few thousand triangles, so the dense pass is cheap and exact.
    """
    a, b, c = (points[coat_tris[:, 0]], points[coat_tris[:, 1]], points[coat_tris[:, 2]])
    ab, ac = b - a, c - a
    n = np.cross(ab, ac)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)

    d = targets[:, None, :] - a[None]  # (M, T, 3)
    proj = d - (d * n[None]).sum(-1, keepdims=True) * n[None]  # onto each triangle plane

    d00 = (ab * ab).sum(-1)
    d01 = (ab * ac).sum(-1)
    d11 = (ac * ac).sum(-1)
    denom = np.maximum(d00 * d11 - d01 * d01, 1e-30)
    d20 = (proj * ab[None]).sum(-1)
    d21 = (proj * ac[None]).sum(-1)
    v = (d11[None] * d20 - d01[None] * d21) / denom[None]
    w = (d00[None] * d21 - d01[None] * d20) / denom[None]
    u = 1.0 - v - w

    bc = np.stack([u, v, w], axis=-1)
    bc_clamped = np.clip(bc, 0.0, 1.0)
    bc_clamped /= bc_clamped.sum(-1, keepdims=True)
    closest = (a[None] * bc_clamped[..., 0:1] + b[None] * bc_clamped[..., 1:2]
               + c[None] * bc_clamped[..., 2:3])
    best = np.linalg.norm(closest - targets[:, None, :], axis=-1).argmin(axis=1)

    rows = np.arange(len(targets))
    return coat_tris[best], bc_clamped[rows, best]


def fabricate(spec: FabricationSpec, verbose: bool = True) -> dict:
    """Build the pad and write ``<out_dir>/<name>_maxv=<v>.{vtk,pkl}``."""
    from .stl_io import mesh_info, read_stl
    from .vtk_io import select_faces_by_vertex_mask

    verts, faces = read_stl(spec.stl_path)
    info = mesh_info(verts, faces)
    if not info["closed"]:
        raise ValueError(f"{spec.stl_path}: surface is not closed "
                         f"({info['boundary_edges']} boundary edges)")

    points, tets = _tetrahedralise(verts, faces, spec.max_volume, verbose=verbose)

    zmax = points[:, 2].max()
    stick_z = spec.stick_z_min if spec.stick_z_min is not None else zmax - 1e-4
    coat_mask = points[:, 2] < spec.coat_z_max
    stick_mask = points[:, 2] > stick_z
    if not coat_mask.any() or not stick_mask.any():
        raise ValueError(f"empty mask: coat={int(coat_mask.sum())} stick={int(stick_mask.sum())} "
                         f"(pad local z spans {points[:,2].min():.4g}..{zmax:.4g})")

    surf = _boundary_faces(tets, points)
    coat_tris = select_faces_by_vertex_mask(surf, coat_mask)

    g = np.linspace(-spec.marker_extent_mm, spec.marker_extent_mm, spec.marker_grid) * 1e-3
    gx, gy = np.meshgrid(g, g, indexing="ij")
    targets = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=-1)
    marker_tris, marker_bc = _marker_barycentric(points, coat_tris, targets)
    recovered = (points[marker_tris] * marker_bc[..., None]).sum(axis=-2)
    marker_err = np.linalg.norm(recovered - targets, axis=1)

    out_dir = spec.out_dir or osp.dirname(osp.abspath(spec.stl_path))
    os.makedirs(out_dir, exist_ok=True)
    stem = osp.join(out_dir, f"{spec.name}_maxv={spec.max_volume:g}")

    import pyvista as pv

    cells = np.hstack([np.full((len(tets), 1), 4, np.int64), tets.astype(np.int64)]).ravel()
    grid = pv.UnstructuredGrid(cells, np.full(len(tets), pv.CellType.TETRA), points)
    grid.save(stem + ".vtk")

    meta = {
        "max_v": float(grid.compute_cell_sizes()["Volume"].max()),
        "target_v": float(spec.max_volume),
        "stick_mask": stick_mask,
        "coat_mask": coat_mask,
        "coat_mask_surf": np.zeros(0, dtype=bool),  # unused by stiff_tactile
        "surface_index": np.zeros(0, dtype=np.int64),
        "marker_vert_idx": marker_tris.astype(np.int64).tolist(),
        "marker_vert_idx_surf": [],
        "marker_bc_coords": marker_bc.tolist(),
    }
    with open(stem + ".pkl", "wb") as f:
        pickle.dump(meta, f)

    edge = np.linalg.norm(points[coat_tris[:, 1]] - points[coat_tris[:, 0]], axis=1)
    stats = {
        "vtk": stem + ".vtk", "pkl": stem + ".pkl",
        "n_verts": len(points), "n_tets": len(tets),
        "n_stick": int(stick_mask.sum()), "n_coat": int(coat_mask.sum()),
        "n_coat_tris": len(coat_tris), "n_markers": len(targets),
        "coat_edge_mean_mm": float(edge.mean() * 1e3),
        "coat_edge_max_mm": float(edge.max() * 1e3),
        "marker_snap_max_um": float(marker_err.max() * 1e6),
        "max_volume_actual": meta["max_v"],
    }
    if verbose:
        print(f"[fabricate] {osp.basename(spec.stl_path)} @ maxv={spec.max_volume:g}: "
              f"{stats['n_verts']}v / {stats['n_tets']}t, "
              f"{stats['n_stick']} stick, {stats['n_coat']} coat, "
              f"{stats['n_coat_tris']} coat tris "
              f"(edge mean {stats['coat_edge_mean_mm']:.3f} mm, max {stats['coat_edge_max_mm']:.3f} mm), "
              f"{stats['n_markers']} markers snapped within "
              f"{stats['marker_snap_max_um']:.1f} um")
        print(f"[fabricate] wrote {stats['vtk']}")
    return stats


def write_fabrication_json(json_path: str, link_name: str, mesh_path: str,
                           reso: float, pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)) -> None:
    """Write the ``tac_fabr_*.json`` that SensorAsset.from_taccel_fabrication reads."""
    import json

    entry = [{"link_name": link_name, "mesh_path": mesh_path,
              "reso": f"{reso:g}", "pos": list(pos), "rot": list(rot)}]
    with open(json_path, "w") as f:
        json.dump(entry, f, indent=4)

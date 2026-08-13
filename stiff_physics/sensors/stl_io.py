"""Minimal STL reader (binary + ASCII), numpy only.

Taccel ships its indenter geometry as STL: ``assets/objects/patterns/*.stl``
(the printed shapes used for the tactile-image-fidelity experiment) and
``assets/objects/mech/C1..C10.stl`` (the classification parts).  StiffGIPC can
take a closed triangle surface straight in as an ABD body::

    v, f = read_stl("patterns/sphere.stl")
    engine.load_mesh_from_data(v, f, verts_per_face=3, dimensions=3,
                              body_type="ABD", transform=T)

so no tetrahedralisation and no trimesh dependency is needed for rigid objects.
"""

from __future__ import annotations

import numpy as np


def read_stl(path: str, weld_tol: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    """Read an STL into welded (verts (N,3) float64, faces (F,3) int32).

    STL is a soup of independent triangles; ABD needs a closed mesh with shared
    vertices, so identical positions are merged (quantised at `weld_tol`).
    """
    buf = open(path, "rb").read()
    is_ascii = buf[:5].lower() == b"solid" and b"facet" in buf[:2048].lower()

    if is_ascii:
        tokens = buf.decode("ascii", "ignore").split()
        pts = [
            (float(tokens[i + 1]), float(tokens[i + 2]), float(tokens[i + 3]))
            for i, t in enumerate(tokens)
            if t == "vertex"
        ]
        tris = np.asarray(pts, dtype=np.float64).reshape(-1, 3, 3)
    else:
        n_tri = int(np.frombuffer(buf, dtype="<u4", count=1, offset=80)[0])
        rec = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
        data = np.frombuffer(buf, dtype=rec, count=n_tri, offset=84)
        tris = data["v"].astype(np.float64)

    flat = tris.reshape(-1, 3)
    scale = 1.0 / max(weld_tol, 1e-15)
    _, first, inverse = np.unique(
        np.round(flat * scale).astype(np.int64), axis=0, return_index=True, return_inverse=True
    )
    verts = flat[first]
    faces = inverse.reshape(-1, 3).astype(np.int32)

    # drop degenerate triangles created by welding
    ok = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    return np.ascontiguousarray(verts), np.ascontiguousarray(faces[ok])


def mesh_info(verts: np.ndarray, faces: np.ndarray) -> dict:
    """bbox / volume / closedness — worth checking before handing to ABD."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    vol = np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return {
        "n_verts": len(verts),
        "n_faces": len(faces),
        "bbox_min": verts.min(axis=0),
        "bbox_max": verts.max(axis=0),
        "extent": verts.max(axis=0) - verts.min(axis=0),
        "volume": float(vol),
        "closed": bool(np.all(counts == 2)),
        "boundary_edges": int(np.sum(counts != 2)),
    }


def transform_to_place(
    verts: np.ndarray, scale: float = 1.0, center_xz: bool = True, min_y: float | None = None
) -> np.ndarray:
    """Build a 4x4 that scales, centres in X/Z and puts the mesh's lowest point
    at world y = `min_y` (Y-up world, as StiffGIPC uses)."""
    T = np.eye(4)
    T[:3, :3] *= scale
    v = verts * scale
    if center_xz:
        c = (v.max(axis=0) + v.min(axis=0)) / 2.0
        T[0, 3] = -c[0]
        T[2, 3] = -c[2]
    if min_y is not None:
        T[1, 3] = min_y - v[:, 1].min()
    return T

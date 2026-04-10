"""Mesh simplification and convex decomposition utilities.

Shared by ``engine.py``, ``usd_scene_parser.py``, and CLI tools.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def simplify_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_verts: int = 500,
    method: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a triangle mesh to at most *max_verts* vertices.

    Parameters
    ----------
    vertices : (V, 3) float64
    faces : (F, 3) int32
    max_verts : target vertex budget
    method : ``"auto"`` tries quadric decimation first, then convex hull.
             ``"convex_hull"`` always uses convex hull.
             ``"decimate"`` only tries decimation (returns original on failure).

    Returns
    -------
    (new_vertices, new_faces) – simplified mesh.  Returned unchanged when
    the mesh is already within budget or simplification fails.
    """
    if len(vertices) <= max_verts:
        return vertices, faces

    try:
        import trimesh
    except ImportError:
        print("[mesh_utils] trimesh not installed – skipping simplification", flush=True)
        return vertices, faces

    tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    if method in ("auto", "decimate"):
        target_faces = max(max_verts, 2 * max_verts)
        if len(tm.faces) > target_faces:
            ratio = max(0.01, 1.0 - target_faces / len(tm.faces))
            try:
                dec = tm.simplify_quadric_decimation(ratio)
                if len(dec.vertices) > 0 and len(dec.faces) > 0:
                    print(
                        f"[mesh_utils] Decimated: {len(vertices)}→{len(dec.vertices)} verts, "
                        f"{len(faces)}→{len(dec.faces)} faces",
                        flush=True,
                    )
                    return (
                        np.asarray(dec.vertices, dtype=np.float64),
                        np.asarray(dec.faces, dtype=np.int32),
                    )
            except Exception:
                pass

    if method in ("auto", "convex_hull"):
        try:
            hull = tm.convex_hull
            print(
                f"[mesh_utils] Convex hull: {len(vertices)}→{len(hull.vertices)} verts",
                flush=True,
            )
            return (
                np.asarray(hull.vertices, dtype=np.float64),
                np.asarray(hull.faces, dtype=np.int32),
            )
        except Exception:
            pass

    print("[mesh_utils] Simplification failed – using original mesh", flush=True)
    return vertices, faces


def simplify_mesh_data(
    mesh_data: dict,
    max_verts: int = 500,
    method: str = "auto",
) -> dict:
    """Dict-based wrapper around :func:`simplify_mesh`.

    Accepts and returns ``{"vertices": ..., "faces": ..., "verts_per_face": 3}``.
    Used by :class:`UsdSceneParser`.
    """
    verts = mesh_data["vertices"]
    if len(verts) <= max_verts:
        return mesh_data
    new_v, new_f = simplify_mesh(verts, mesh_data["faces"], max_verts, method)
    return {
        "vertices": new_v,
        "faces": new_f,
        "verts_per_face": 3,
    }


# ---------------------------------------------------------------------------
# Convex decomposition via CoACD
# ---------------------------------------------------------------------------

def convex_decompose(
    vertices: np.ndarray,
    faces: np.ndarray,
    threshold: float = 0.05,
    max_convex_hull: int = -1,
    max_ch_vertex: int = 256,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Approximate convex decomposition using CoACD.

    Parameters
    ----------
    vertices : (V, 3) float64
    faces : (F, 3) int32
    threshold : concavity threshold – lower means finer decomposition
    max_convex_hull : max number of output parts (-1 = auto)
    max_ch_vertex : max vertices per convex hull

    Returns
    -------
    List of ``(verts, faces)`` tuples, one per convex part.
    Falls back to a single-element list containing the original mesh
    when CoACD is unavailable.
    """
    try:
        import coacd
    except ImportError:
        print("[mesh_utils] coacd not installed – returning original mesh as single part", flush=True)
        return [(vertices, faces)]

    coacd.set_log_level("warning")
    mesh = coacd.Mesh(
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int32),
    )
    raw_parts = coacd.run_coacd(
        mesh,
        threshold=threshold,
        max_convex_hull=max_convex_hull,
        max_ch_vertex=max_ch_vertex,
    )
    parts = []
    for v, f in raw_parts:
        parts.append((
            np.asarray(v, dtype=np.float64),
            np.asarray(f, dtype=np.int32),
        ))
    print(f"[mesh_utils] CoACD: {len(vertices)} verts → {len(parts)} convex parts", flush=True)
    return parts


def write_vhacd_obj(
    path: str,
    parts: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    """Write convex decomposition parts as a multi-component OBJ file.

    Each part is written with an ``o convex_N`` header, compatible with
    the C++ ``ObjMeshLoader`` which parses ``o``/``g`` groups into
    ``ObjConvexComponent`` entries.
    """
    with open(path, "w") as fp:
        fp.write(f"# CoACD convex decomposition: {len(parts)} components\n")
        vert_offset = 0
        for i, (verts, faces) in enumerate(parts):
            fp.write(f"o convex_{i}\n")
            for v in verts:
                fp.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
            for f in faces:
                fp.write(f"f {f[0]+1+vert_offset} {f[1]+1+vert_offset} {f[2]+1+vert_offset}\n")
            vert_offset += len(verts)
    print(f"[mesh_utils] Wrote {len(parts)} components to {path}", flush=True)

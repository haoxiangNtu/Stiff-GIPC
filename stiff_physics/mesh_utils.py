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


# ---------------------------------------------------------------------------
# Per-face orientation labeling (libuipc-style)
# ---------------------------------------------------------------------------

def compute_face_orient_flood_fill(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """Compute per-face orient labels via flood-fill BFS.

    For an ABD body whose source mesh has inconsistent triangle winding,
    this returns an ``orient`` array where ``orient[i] = -1`` flags
    triangles that should be sign-flipped at integration time, and
    ``+1`` (or implicit) for already-correct ones. Pass the result to
    ``Engine.set_abd_body_face_orient(body_id, orient)`` BEFORE
    finalize() to fix mass/centroid/inertia without mutating face
    vertex order.

    Algorithm:
      1. Build edge → list of face ids.
      2. BFS from face 0 (restart for each disconnected component): a
         neighbor sharing edge (a,b) must traverse it as (b,a) for
         consistent winding; if it traverses (a,b) instead, mark its
         orient as flipped relative to its current direction.
      3. After flood-fill, all faces in each component agree on a local
         "outward" direction. Compute signed volume; if < 0, the
         component's outward direction is actually inward, so flip ALL
         labels in it so outward convention is restored.

    Limitations: requires a manifold mesh (each edge in ≤2 faces).
    Cannot fix non-closed meshes (where the volume integral is
    mathematically meaningless) — for those, fix the asset upstream
    (e.g. ``examples/fix_obj_winding.py --convex-hull``).

    Returns
    -------
    orient : (n_faces,) int32 array, values in {-1, +1}.
    """
    from collections import defaultdict, deque

    verts = np.asarray(vertices, dtype=np.float64)
    tris  = np.asarray(faces, dtype=np.int32)
    n     = len(tris)
    if n == 0:
        return np.empty((0,), dtype=np.int32)

    # Edge → face list
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, t in enumerate(tris):
        for k in range(3):
            a, b = int(t[k]), int(t[(k + 1) % 3])
            edge_to_faces[(min(a, b), max(a, b))].append(fi)

    # current_swap[fi] tracks whether face fi has been (logically) flipped
    # during the BFS. We don't mutate `tris` to keep the algorithm
    # idempotent / non-destructive.
    swap = np.zeros(n, dtype=np.int8)  # 0 = original, 1 = flipped

    def edge_dir_after_swap(fi: int, a: int, b: int) -> int:
        """Returns 0 if face fi (with its current swap state) traverses
        edge (a,b) in the order a→b; 1 if b→a."""
        t = tris[fi]
        if swap[fi]:
            t0, t1, t2 = int(t[0]), int(t[2]), int(t[1])
        else:
            t0, t1, t2 = int(t[0]), int(t[1]), int(t[2])
        for k, (x, y) in enumerate([(t0, t1), (t1, t2), (t2, t0)]):
            if x == a and y == b:
                return 0
            if x == b and y == a:
                return 1
        raise ValueError(f"face {fi} does not contain edge ({a},{b})")

    visited = np.zeros(n, dtype=bool)

    # BFS each component independently
    for seed in range(n):
        if visited[seed]:
            continue
        visited[seed] = True
        component = [seed]
        q = deque([seed])
        while q:
            f = q.popleft()
            t = tris[f]
            t_eff = (int(t[0]), int(t[2]), int(t[1])) if swap[f] else \
                    (int(t[0]), int(t[1]), int(t[2]))
            for k in range(3):
                a, b = t_eff[k], t_eff[(k + 1) % 3]
                my_dir = 0  # by construction we just read t_eff
                for nbr in edge_to_faces[(min(a, b), max(a, b))]:
                    if nbr == f or visited[nbr]:
                        continue
                    visited[nbr] = True
                    nd = edge_dir_after_swap(nbr, a, b)
                    if nd == my_dir:
                        # Same direction → inconsistent; mark neighbor as
                        # needing a swap to bring it in line.
                        swap[nbr] = 1
                    component.append(nbr)
                    q.append(nbr)

        # Component-level orientation check via signed volume
        V = 0.0
        for fi in component:
            t = tris[fi]
            if swap[fi]:
                p0, p1, p2 = verts[t[0]], verts[t[2]], verts[t[1]]
            else:
                p0, p1, p2 = verts[t[0]], verts[t[1]], verts[t[2]]
            V += float(np.dot(p0, np.cross(p1, p2)))
        if V < 0:
            # Whole component is consistently inward → flip every label
            for fi in component:
                swap[fi] ^= 1

    # Convert swap[] (0/1) to orient (+1/-1)
    orient = np.where(swap == 1, np.int32(-1), np.int32(1))
    return orient


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

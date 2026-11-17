"""Legacy-VTK tet-mesh reader + boundary extraction, numpy only.

Taccel stores gel pads as legacy .vtk UnstructuredGrid written by PyVista, and
reads them back with ``pv.read()``.  We do not want a PyVista/VTK dependency in
the simulation loop, so this module parses the two legacy layouts PyVista emits
(VTK 4.2 "CELLS n size" and VTK 5.1 "OFFSETS/CONNECTIVITY") directly.
"""

from __future__ import annotations

import re

import numpy as np

_DTYPES = {
    "float": ">f4",
    "double": ">f8",
    "int": ">i4",
    "unsigned_int": ">u4",
    "long": ">i8",
    "vtktypeint32": ">i4",
    "vtktypeint64": ">i8",
    "vtkidtype": ">i4",
}


def _read_block(buf: bytes, pos: int, count: int, dtype: str, binary: bool):
    """Read `count` scalars of `dtype` starting at `pos`; returns (array, new_pos)."""
    np_dt = _DTYPES[dtype.lower()]
    if binary:
        nbytes = count * np.dtype(np_dt).itemsize
        arr = np.frombuffer(buf, dtype=np_dt, count=count, offset=pos)
        return np.asarray(arr), pos + nbytes
    # ASCII: consume `count` whitespace-separated tokens
    tokens, n = [], 0
    while n < count:
        m = re.compile(rb"\S+").search(buf, pos)
        tokens.append(m.group())
        pos = m.end()
        n += 1
    kind = np.dtype(np_dt).kind
    return np.array([float(t) if kind == "f" else int(t) for t in tokens]), pos


def read_tet_mesh(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read a legacy .vtk UnstructuredGrid of tetrahedra.

    Returns:
        points: (N, 3) float64
        tets:   (M, 4) int32
    """
    buf = open(path, "rb").read()
    binary = b"\nBINARY" in buf[:256]

    m = re.search(rb"POINTS\s+(\d+)\s+(\w+)\s*\n", buf)
    if m is None:
        raise ValueError(f"{path}: no POINTS section")
    n_pts, pt_dtype = int(m.group(1)), m.group(2).decode()
    flat, pos = _read_block(buf, m.end(), n_pts * 3, pt_dtype, binary)
    points = flat.reshape(n_pts, 3).astype(np.float64)

    m = re.compile(rb"CELLS\s+(\d+)\s+(\d+)\s*\n").search(buf, pos)
    if m is None:
        raise ValueError(f"{path}: no CELLS section")
    a, b, pos = int(m.group(1)), int(m.group(2)), m.end()

    m5 = re.compile(rb"\s*OFFSETS\s+(\w+)\s*\n").match(buf, pos)
    if m5 is not None:  # VTK 5.1: CELLS n_offsets n_conn / OFFSETS / CONNECTIVITY
        offsets, pos = _read_block(buf, m5.end(), a, m5.group(1).decode(), binary)
        mc = re.compile(rb"\s*CONNECTIVITY\s+(\w+)\s*\n").search(buf, pos)
        conn, pos = _read_block(buf, mc.end(), b, mc.group(1).decode(), binary)
        sizes = np.diff(offsets)
        if not np.all(sizes == 4):
            raise ValueError(f"{path}: non-tet cells present (sizes {set(sizes.tolist())})")
        tets = conn.reshape(-1, 4)
    else:  # VTK 4.2: CELLS n_cells total_ints, connectivity as [4, i0, i1, i2, i3] ...
        raw, pos = _read_block(buf, pos, b, "int", binary)
        raw = raw.reshape(a, -1)
        if raw.shape[1] != 5 or not np.all(raw[:, 0] == 4):
            raise ValueError(f"{path}: non-tet cells present")
        tets = raw[:, 1:]

    return points, np.ascontiguousarray(tets, dtype=np.int32)


def boundary_faces(tets: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Outward-oriented boundary triangles of a tet mesh. Returns (F, 3) int32.

    Orientation is fixed geometrically (normal pointing away from the owning
    tet's centroid), so no assumption is made about the input tet winding.
    """
    t = np.asarray(tets, dtype=np.int64)
    faces = np.concatenate(
        [t[:, [1, 2, 3]], t[:, [0, 3, 2]], t[:, [0, 1, 3]], t[:, [0, 2, 1]]], axis=0
    )
    owner = np.tile(np.arange(t.shape[0]), 4)

    keys = np.sort(faces, axis=1)
    _, first, counts = np.unique(keys, axis=0, return_index=True, return_counts=True)
    keep = first[counts == 1]
    tri, own = faces[keep], owner[keep]

    v0, v1, v2 = points[tri[:, 0]], points[tri[:, 1]], points[tri[:, 2]]
    nrm = np.cross(v1 - v0, v2 - v0)
    centroid = points[t[own]].mean(axis=1)
    flip = np.einsum("ij,ij->i", nrm, (v0 + v1 + v2) / 3.0 - centroid) < 0
    tri[flip] = tri[flip][:, ::-1]
    return np.ascontiguousarray(tri, dtype=np.int32)


def select_faces_by_vertex_mask(faces: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Sub-set of `faces` whose three vertices are all inside `mask` (body indices)."""
    m = np.asarray(mask, dtype=bool)
    return faces[m[faces].all(axis=1)]

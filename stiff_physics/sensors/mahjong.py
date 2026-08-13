"""Procedural mahjong tile with an engraved character.

Taccel's Figure 1 grasps a mahjong tile whose engraved face the fingertip gels
read back. That asset is not in the released repo, so this builds one: a solid
tile whose front face carries a recessed glyph, as a closed manifold triangle
mesh that StiffGIPC can load straight in as an ABD body.

Construction: rasterise the glyph to a binary mask, turn it into a height field
on a regular grid (engraved where the glyph is), then close the mesh with side
walls and a flat back. Grid topology means the manifold is correct by
construction -- no boolean ops, no repair.
"""

from __future__ import annotations

import numpy as np

# Sans-Black FIRST, deliberately. A real mahjong tile's character is cut with
# even stroke weight; a serif face makes the horizontal strokes of a glyph like
# the mouth radical roughly half the width of the verticals, and a sub-millimetre
# groove is below what a 3 mm gel can dip into -- the readout then recovers the
# verticals and misses the horizontals entirely (IoU 0.76, recall on the
# horizontals near zero). That is a genuine sensor bandwidth limit, but it should
# not be imposed by the typeface.
CJK_FONTS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]


def glyph_mask(char: str = "中", n: int = 160, coverage: float = 0.78,
               font_path: str = "") -> np.ndarray:
    """(n, n) bool mask of the glyph, True inside the strokes."""
    from PIL import Image, ImageDraw, ImageFont

    paths = [font_path] if font_path else CJK_FONTS
    font = None
    for p in paths:
        try:
            font = ImageFont.truetype(p, int(n * coverage))
            break
        except Exception:
            continue
    if font is None:
        raise RuntimeError(f"no usable CJK font among {paths}")

    img = Image.new("L", (n, n), 0)
    dr = ImageDraw.Draw(img)
    l, t, r, b = dr.textbbox((0, 0), char, font=font)
    dr.text(((n - (r - l)) / 2 - l, (n - (b - t)) / 2 - t), char, fill=255, font=font)
    return np.array(img) > 127


def tile_mesh(width_mm: float = 20.0, height_mm: float = 26.0,
              thick_mm: float = 12.0, char: str = "中",
              engrave_mm: float = 0.9, glyph_mm: float = 13.0,
              grid: int = 121, font_path: str = ""):
    """Closed triangle mesh of the tile, in METRES, centred at the origin.

    The engraved face is +Z; the glyph is recessed by `engrave_mm`.
    Returns (verts (N,3) float64, faces (F,3) int32).
    """
    m = glyph_mask(char, n=max(64, grid), font_path=font_path)

    # height field on a grid over the front face: z = +thick/2, minus the engraving
    gx = np.linspace(-width_mm / 2, width_mm / 2, grid)
    gy = np.linspace(-height_mm / 2, height_mm / 2, grid)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")

    # sample the glyph mask over the glyph's own square footprint
    u = (GX / glyph_mm + 0.5) * (m.shape[0] - 1)
    v = (1.0 - (GY / glyph_mm + 0.5)) * (m.shape[1] - 1)     # image y is flipped
    inside = np.zeros_like(GX, bool)
    ok = (u >= 0) & (u <= m.shape[0] - 1) & (v >= 0) & (v <= m.shape[1] - 1)
    ui = np.clip(np.round(u).astype(int), 0, m.shape[0] - 1)
    vi = np.clip(np.round(v).astype(int), 0, m.shape[1] - 1)
    inside[ok] = m[vi[ok], ui[ok]]

    zf = np.full(GX.shape, thick_mm / 2)
    zf[inside] -= engrave_mm

    n = grid
    front = np.stack([GX, GY, zf], -1).reshape(-1, 3)
    back = np.stack([GX, GY, np.full(GX.shape, -thick_mm / 2)], -1).reshape(-1, 3)
    verts = np.vstack([front, back]) * 1e-3                  # mm -> m

    def quad(a, b, c, d):                                    # ccw when seen from outside
        return [[a, b, c], [a, c, d]]

    F = []
    idx = lambda i, j: i * n + j                             # noqa: E731
    bidx = lambda i, j: n * n + i * n + j                    # noqa: E731
    for i in range(n - 1):
        for j in range(n - 1):
            F += quad(idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1))
            F += quad(bidx(i, j), bidx(i, j + 1), bidx(i + 1, j + 1), bidx(i + 1, j))
    for j in range(n - 1):                                   # x- and x+ walls
        F += quad(idx(0, j), idx(0, j + 1), bidx(0, j + 1), bidx(0, j))
        F += quad(idx(n - 1, j + 1), idx(n - 1, j), bidx(n - 1, j), bidx(n - 1, j + 1))
    for i in range(n - 1):                                   # y- and y+ walls
        F += quad(idx(i + 1, 0), idx(i, 0), bidx(i, 0), bidx(i + 1, 0))
        F += quad(idx(i, n - 1), idx(i + 1, n - 1), bidx(i + 1, n - 1), bidx(i, n - 1))

    faces = np.asarray(F, np.int32)

    # NO centroid-based winding fix here. The height-field grid is already wound
    # consistently by construction, and the centroid test is WRONG for concave
    # features: the engraving's side walls have horizontal outward normals while
    # their offset from the body centroid is nearly vertical, so the test flips a
    # random subset of them. That produced a non-manifold-looking surface whose
    # second moment came out indefinite ("[ABD][WARN] ... INDEFINITE centered
    # second moment") and then an illegal memory access inside the ABD setup.
    # Verify with the signed volume instead (see tile_mesh_check).
    return verts, faces


def tile_mesh_check(verts, faces, width_mm, height_mm, thick_mm, engrave_mm) -> dict:
    """Signed volume vs the analytic expectation -- catches winding mistakes."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    vol = np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0
    box = width_mm * height_mm * thick_mm * 1e-9
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]],
                                    faces[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(edges, axis=0, return_counts=True)
    return dict(volume_mm3=vol * 1e9, box_mm3=box * 1e9,
                engraved_mm3=(box - vol) * 1e9,
                closed=bool(np.all(cnt == 2)), bad_edges=int((cnt != 2).sum()))


def tile_info(verts, faces) -> dict:
    from .stl_io import mesh_info

    return mesh_info(verts, faces)

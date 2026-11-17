"""Tactile observations: depth map, normal map, marker flow.

Taccel casts one warp ray per pixel from local z=+ray_start_level along -z at the
coat surface and sets ``depth = |start-rest| - t``.  With a -z ray the first hit
is the largest-z intersection, so that depth is exactly the coat surface's local
height under each pixel -- i.e. an orthographic z-buffer.  This module implements
that z-buffer directly (numpy reference path, no warp/CUDA needed), which makes
the observation identical in meaning and easy to unit-test.

For production throughput, replace `depth_from_coat_surface` with a warp kernel
(see `Taccel taccel/vbts.py: raycast_depth_kernel`) or a CUDA rasteriser; the
rest of the pipeline is unchanged.
"""

from __future__ import annotations

import numpy as np

from .sensor_asset import CameraConfig, SensorAsset


def pixel_grid_local(cam: CameraConfig) -> tuple[np.ndarray, np.ndarray]:
    """Sensor-local (x, y) of every pixel centre. Returns (xs (H,), ys (W,))."""
    p = cam.pixel_size_m
    xs = np.arange(cam.height) * p
    ys = np.arange(cam.width) * p
    # Taccel centres the whole grid on the gel origin.
    xs = xs - (np.arange(cam.height) * p).mean()
    ys = ys - (np.arange(cam.width) * p).mean()
    return xs, ys


def depth_from_coat_surface(
    verts_local: np.ndarray,
    coat_faces: np.ndarray,
    cam: CameraConfig,
    return_normals: bool = False,
):
    """Orthographic z-buffer of the deformed coat surface.

    Args:
        verts_local: (N, 3) gel vertex positions in the SENSOR-LOCAL frame.
        coat_faces:  (F, 3) coat triangles (body vertex indices).
        cam:         camera config.
        return_normals: also return the true geometric surface normal of the
            winning triangle per pixel (see `normal_mode="geometric"`).
    Returns:
        (H, W) float32 depth in metres, 0 = untouched, clamped to cam.max_depth,
        plus (H, W, 3) float32 normals if `return_normals`.
    """
    xs, ys = pixel_grid_local(cam)
    H, W = cam.height, cam.width
    p = cam.pixel_size_m
    x0, y0 = xs[0], ys[0]

    zbuf = np.full((H, W), -np.inf, dtype=np.float64)
    nbuf = np.zeros((H, W, 3), dtype=np.float64) if return_normals else None
    tri = verts_local[coat_faces]  # (F, 3, 3)

    # Per-triangle bbox rasterisation with barycentric z interpolation.
    for a, b, c in tri:
        i_lo = int(np.ceil((min(a[0], b[0], c[0]) - x0) / p))
        i_hi = int(np.floor((max(a[0], b[0], c[0]) - x0) / p))
        j_lo = int(np.ceil((min(a[1], b[1], c[1]) - y0) / p))
        j_hi = int(np.floor((max(a[1], b[1], c[1]) - y0) / p))
        i_lo, i_hi = max(i_lo, 0), min(i_hi, H - 1)
        j_lo, j_hi = max(j_lo, 0), min(j_hi, W - 1)
        if i_lo > i_hi or j_lo > j_hi:
            continue

        px = xs[i_lo : i_hi + 1][:, None]
        py = ys[j_lo : j_hi + 1][None, :]

        d = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
        if abs(d) < 1e-18:  # degenerate in projection
            continue
        w1 = ((px - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (py - a[1])) / d
        w2 = ((b[0] - a[0]) * (py - a[1]) - (px - a[0]) * (b[1] - a[1])) / d
        w0 = 1.0 - w1 - w2
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if not inside.any():
            continue
        z = w0 * a[2] + w1 * b[2] + w2 * c[2]
        block = zbuf[i_lo : i_hi + 1, j_lo : j_hi + 1]
        win = inside & (z > block)
        np.copyto(block, z, where=win)
        if nbuf is not None:
            fn = np.cross(b - a, c - a)
            fn = fn / max(np.linalg.norm(fn), 1e-30)
            if fn[2] < 0:
                fn = -fn  # face the camera (+z)
            np.copyto(nbuf[i_lo : i_hi + 1, j_lo : j_hi + 1], fn, where=win[..., None])

    depth = np.where(np.isfinite(zbuf), zbuf - cam.ray_rest_level, 0.0)
    depth = np.clip(depth, 0.0, cam.max_depth).astype(np.float32)
    if nbuf is None:
        return depth
    flat = np.all(nbuf == 0.0, axis=-1)
    nbuf[flat] = (0.0, 0.0, 1.0)
    return depth, nbuf.astype(np.float32)


def smooth_depth(depth: np.ndarray, sigma_px: float) -> np.ndarray:
    """Separable Gaussian blur of the depth map (numpy only, edge-clamped).

    The gel tet mesh is much coarser than the camera: at gel_resolution=1e-09 the
    coat triangles are ~0.5 mm while a pixel is 0.079 mm, so a raw z-buffer shows
    the triangulation as facets and the normals inherit them.  Taccel's
    ``render.yml`` smooths the coat for the same reason (``coat: smoothing:
    laplacian``).  sigma_px ~2-3 removes the faceting without visibly softening a
    real contact edge.
    """
    if sigma_px <= 0:
        return depth
    r = max(1, int(round(3 * sigma_px)))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma_px) ** 2)
    k /= k.sum()
    out = depth.astype(np.float64)
    for axis in (-2, -1):
        pad = [(0, 0)] * out.ndim
        pad[axis] = (r, r)
        p = np.pad(out, pad, mode="edge")
        p = np.moveaxis(p, axis, -1)
        acc = np.zeros(np.moveaxis(out, axis, -1).shape, dtype=np.float64)
        for i, w in enumerate(k):
            acc += w * p[..., i : i + acc.shape[-1]]
        out = np.moveaxis(acc, -1, axis)
    return out.astype(np.float32)


def depth_to_normal(depth: np.ndarray, pixel_size_m: float | None = None) -> np.ndarray:
    """(H, W) depth in metres -> (H, W, 3) unit normals.

    ``pixel_size_m=None`` reproduces Taccel's ``depth_to_normal`` exactly: the
    gradient is taken per pixel INDEX, so with metre-scale depth the tangential
    components come out ~1e-5 and the normal is (0, 0, 1) to five decimals.
    That makes ``n2rgb.pth`` output independent of the contact -- its first-layer
    weights on n_x/n_y are O(1..9), so it needs |n_xy| >~ 1e-2 to respond.
    Taccel's own ``render_tactile`` has this line commented out right above:
    ``# all_normals = wp.to_torch(self.vbts_normals)`` (the true raycast normals).

    Pass ``pixel_size_m=cam.pixel_size_m`` for the physically correct slope,
    which is what the checkpoint expects.
    """
    s = 1.0 if pixel_size_m is None else float(pixel_size_m)
    dz_dx = np.gradient(depth, axis=-2) / s
    dz_dy = np.gradient(depth, axis=-1) / s
    n = np.stack([-dz_dx, -dz_dy, np.ones_like(depth)], axis=-1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def markers_to_pixels(markers_local: np.ndarray, cam: CameraConfig) -> np.ndarray:
    """Sensor-local marker positions (K, 3) -> (K, 2) pixel coordinates."""
    shift = np.array(cam.marker_shift_px) + np.array([cam.height, cam.width]) / 2.0
    return markers_local[..., :2] * 1e3 / cam.pixel_size_mm + shift


def marker_flow(
    markers_local: np.ndarray, rest_markers_local: np.ndarray, cam: CameraConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (markers_px, rest_px, flow_px)."""
    m = markers_to_pixels(markers_local, cam)
    r = markers_to_pixels(rest_markers_local, cam)
    return m, r, m - r


def observe(
    asset: SensorAsset,
    verts_local: np.ndarray,
    with_markers: bool = True,
    normal_mode: str = "metric",
    smooth_sigma_px: float = 2.0,
) -> dict:
    """Full observation for one sensor from its deformed vertices (local frame).

    normal_mode:
        "metric"    -- normals from the depth gradient scaled by the pixel pitch
                       (physically correct slope; what n2rgb.pth expects). Default.
        "geometric" -- true per-pixel triangle normal from the rasteriser
                       (sharper at the contact rim, slightly noisier on a coarse gel).
        "taccel"    -- bit-compatible with Taccel's released render_tactile
                       (index-space gradient; renders a contact-independent image).
    smooth_sigma_px: Gaussian blur applied to the depth map before the normals
        are taken, to suppress the gel triangulation (see `smooth_depth`).
        0 disables it; "geometric" normals are unaffected (they come from the
        triangles themselves), so use "metric" when smoothing matters.
    """
    cam = asset.camera
    if normal_mode == "geometric":
        depth, normal = depth_from_coat_surface(verts_local, asset.coat_faces, cam,
                                                return_normals=True)
        depth = smooth_depth(depth, smooth_sigma_px)
    else:
        depth = smooth_depth(
            depth_from_coat_surface(verts_local, asset.coat_faces, cam), smooth_sigma_px)
        normal = depth_to_normal(depth, None if normal_mode == "taccel" else cam.pixel_size_m)
    out = {"depth": depth, "normal": normal}
    if with_markers:
        cur = asset.markers_from_verts(verts_local)
        m, r, f = marker_flow(cur, asset.rest_markers, asset.camera)
        out.update(markers_3d=cur, markers_px=m, markers_rest_px=r, marker_flow_px=f)
        out["marker_shear_mm"] = np.linalg.norm(
            (cur - asset.rest_markers)[..., :2] * 1e3, axis=-1
        )
    return out

"""Zero-copy GPU tactile observation.

The CPU reference path (`observation.py`) rasterises the coat surface in numpy:
26 ms per frame for the fine gel, on one core, per sensor — which is the wall for
any many-environment run long before the solver is. This does the same z-buffer
in CUDA straight off ``Engine.get_vertices_device_ptr()``, so no vertex data
crosses PCIe, and keeps depth / normals / markers as torch CUDA tensors.

Semantics are identical to `observation.observe(..., normal_mode="metric")`; the
CPU path stays the readable reference and the test asserts they agree.

Requires: torch with CUDA, nvcc on PATH, and an engine exposing
``get_vertex_input_to_metis`` (added on the feat/tactile-vbts branch — the raw
device buffer is MAS-permuted, so input-order indices must be remapped).
"""

from __future__ import annotations

import os
import os.path as osp

import numpy as np

from .sensor_asset import SensorAsset

_EXT = None


def _load_ext(verbose: bool = False):
    """JIT-compile csrc/tactile_raster.cu (cached under ~/.cache/torch_extensions)."""
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        # csrc lives INSIDE the package when shipped with stiff-physics, and one
        # level up in the standalone dev workspace -- accept both layouts
        _here = osp.dirname(osp.abspath(__file__))
        _cand = osp.join(_here, "csrc")
        if not osp.isdir(_cand):
            _cand = osp.join(osp.dirname(_here), "csrc")
        src = osp.join(_cand,
                       "tactile_raster.cu")
        _EXT = load(
            name="stiff_tactile_raster",
            sources=[src],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_89,code=sm_89"],
            verbose=verbose,
        )
    return _EXT


class _CudaArrayView:
    """Expose a raw device pointer as a (n, 3) float64 CUDA array for torch."""

    def __init__(self, ptr: int, n: int):
        self.__cuda_array_interface__ = {
            "shape": (n, 3), "typestr": "<f8", "data": (int(ptr), False), "version": 3}


class GpuTactileRenderer:
    """GPU depth / normal / marker readout for one gel pad."""

    def __init__(self, asset: SensorAsset, engine, gel, device: str = "cuda:0",
                 verbose_build: bool = False):
        import torch

        self.asset = asset
        self.engine = engine
        self.gel = gel
        self.device = device
        self.ext = _load_ext(verbose_build)
        self.torch = torch

        if not hasattr(engine, "get_vertex_input_to_metis"):
            raise RuntimeError(
                "this engine build does not expose get_vertex_input_to_metis(); the raw "
                "device vertex buffer is MAS-permuted, so GPU indexing would be wrong. "
                "Use the feat/tactile-vbts engine build (STIFFGIPC_NATIVE_DIR).")
        i2m = engine.get_vertex_input_to_metis()

        # coat triangles: gel-local input ids -> global input ids -> device ids
        tri_global = asset.coat_faces.astype(np.int64) + gel.vertex_offset
        self.tri = torch.as_tensor(
            i2m[tri_global].astype(np.int32), device=device).contiguous()

        mk = asset.marker_vert_idx.astype(np.int64) + gel.vertex_offset
        self.marker_dev_idx = torch.as_tensor(i2m[mk].astype(np.int64), device=device)
        self.marker_bc = torch.as_tensor(asset.marker_bc_coords, device=device).to(torch.float64)
        self.rest_markers = torch.as_tensor(asset.rest_markers, device=device).to(torch.float64)
        self.n_verts_total = int(engine.vertex_count)

    # ------------------------------------------------------------------ core

    def verts_device(self):
        """(N, 3) float64 CUDA view of ALL engine vertices, in DEVICE order."""
        return self.torch.as_tensor(
            _CudaArrayView(self.engine.get_vertices_device_ptr(), self.n_verts_total))

    def depth(self, sensor_tf: np.ndarray | None = None):
        """(H, W) float32 CUDA depth map in metres."""
        cam = self.asset.camera
        tf = np.asarray(self.gel.world_tf if sensor_tf is None else sensor_tf, dtype=np.float64)
        rt = np.ascontiguousarray(tf[:3, :3].T)  # world -> local rotation
        return self.ext.raster_depth(
            int(self.engine.get_vertices_device_ptr()), self.tri,
            self.torch.from_numpy(rt), self.torch.from_numpy(np.ascontiguousarray(tf[:3, 3])),
            int(cam.height), int(cam.width), float(cam.pixel_size_m),
            float(cam.ray_rest_level), float(cam.max_depth))

    def depth_normal_phong(self, sensor_tf: np.ndarray | None = None):
        """(depth, normal) with Phong-interpolated normals from the deformed coat.

        Refining the gel is not a viable route to a facet-free normal field: coat
        edge length scales as V^(1/3), so the shipped 15.5 px triangles would need
        ~175x more tets to reach sub-pixel. Interpolating area-weighted vertex
        normals removes the triangulation from the normal field -- which is what
        n2rgb.pth actually consumes -- at a coarse mesh's cost.
        """
        cam = self.asset.camera
        tf = np.asarray(self.gel.world_tf if sensor_tf is None else sensor_tf, dtype=np.float64)
        rt = np.ascontiguousarray(tf[:3, :3].T)
        d, n = self.ext.raster_depth_normal(
            int(self.engine.get_vertices_device_ptr()), self.tri,
            self.torch.from_numpy(rt), self.torch.from_numpy(np.ascontiguousarray(tf[:3, 3])),
            int(cam.height), int(cam.width), float(cam.pixel_size_m),
            float(cam.ray_rest_level), float(cam.max_depth), int(self.n_verts_total))
        # untouched pixels: flat surface facing the camera
        flat = n.abs().sum(-1) == 0
        n[flat] = self.torch.tensor([0.0, 0.0, 1.0], device=n.device)
        return d, n

    def smooth(self, depth, sigma_px: float):
        """Separable Gaussian blur on GPU (matches observation.smooth_depth)."""
        if sigma_px <= 0:
            return depth
        torch = self.torch
        r = max(1, int(round(3 * sigma_px)))
        x = torch.arange(-r, r + 1, device=depth.device, dtype=torch.float32)
        k = torch.exp(-0.5 * (x / sigma_px) ** 2)
        k = k / k.sum()
        d = depth[None, None]
        d = torch.nn.functional.conv2d(
            torch.nn.functional.pad(d, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
        d = torch.nn.functional.conv2d(
            torch.nn.functional.pad(d, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
        return d[0, 0]

    def normal(self, depth):
        """(H, W, 3) unit normals from the metric depth gradient (see depth_to_normal)."""
        torch = self.torch
        p = self.asset.camera.pixel_size_m
        gx, gy = torch.gradient(depth, dim=(0, 1))
        n = torch.stack([-gx / p, -gy / p, torch.ones_like(depth)], dim=-1)
        return n / n.norm(dim=-1, keepdim=True)

    def markers(self, sensor_tf: np.ndarray | None = None):
        """(K, 3) marker positions in the sensor-local frame, from device vertices."""
        torch = self.torch
        tf = np.asarray(self.gel.world_tf if sensor_tf is None else sensor_tf, dtype=np.float64)
        v = self.verts_device()
        tri = v[self.marker_dev_idx]                       # (K, 3, 3) world
        m = (tri * self.marker_bc[..., None]).sum(dim=-2)  # (K, 3) world
        R = torch.as_tensor(np.ascontiguousarray(tf[:3, :3]), device=m.device)
        t = torch.as_tensor(np.ascontiguousarray(tf[:3, 3]), device=m.device)
        return (m - t) @ R

    # --------------------------------------------------------------- bundle

    def observe(self, sensor_tf: np.ndarray | None = None, with_markers: bool = True,
                smooth_sigma_px: float = 2.0, normal_mode: str = "metric") -> dict:
        """Same keys as `observation.observe`, but torch CUDA tensors.

        normal_mode "metric" reproduces the CPU path (depth gradient / pixel pitch);
        "phong" interpolates deformed vertex normals instead, which is the only
        practical way to get a facet-free normal field out of a solver-sized gel.
        """
        if normal_mode == "phong":
            d_raw, n = self.depth_normal_phong(sensor_tf)
            return_d = self.smooth(d_raw, smooth_sigma_px)
            out = {"depth": return_d, "normal": n}
            d = return_d
        else:
            d = self.smooth(self.depth(sensor_tf), smooth_sigma_px)
            out = {"depth": d, "normal": self.normal(d)}
        if with_markers:
            cam = self.asset.camera
            cur = self.markers(sensor_tf)
            shift = self.torch.as_tensor(
                np.array(cam.marker_shift_px) + np.array([cam.height, cam.width]) / 2.0,
                device=cur.device)
            px = cur[:, :2] * 1e3 / cam.pixel_size_mm + shift
            rpx = self.rest_markers[:, :2] * 1e3 / cam.pixel_size_mm + shift
            out.update(markers_3d=cur, markers_px=px, markers_rest_px=rpx,
                       marker_flow_px=px - rpx,
                       marker_shear_mm=((cur - self.rest_markers)[:, :2] * 1e3).norm(dim=-1))
        return out

"""SensorAsset: everything about one vision-based tactile sensor that is
independent of the physics backend.

This is the port of Taccel's ``VBTSConfig`` (taccel/vbts.py) +
``TactileRobot._load_tac_fabrications`` (taccel/tactile_robot.py), with the
PyVista and warp_ipc dependencies removed.  Semantics preserved:

* gel = tet mesh; contact happens on the coat (reflective) surface at local z=0,
  the sensor is bolted to its carrier through the stick vertices at local z=+t.
* markers live on coat-surface triangles, stored as (3 body vertex ids,
  3 barycentric weights) so their position is a linear function of the FEM state.
* the "camera" is an orthographic z-buffer: pixel pitch `cam_pixel_size` mm,
  rays start at local z=`ray_start_level` and travel along -z.
"""

from __future__ import annotations

import json
import os.path as osp
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from .pkl_compat import load_fabrication_metadata
from .vtk_io import boundary_faces, read_tet_mesh, select_faces_by_vertex_mask


@dataclass
class CameraConfig:
    """Orthographic tactile camera. Defaults are Taccel's GelSight-like sensor."""

    height: int = 400
    width: int = 400
    pixel_size_mm: float = 0.079375
    ray_start_level: float = 0.01  # m, local z where rays start
    ray_rest_level: float = 0.0  # m, local z of the undeformed coat surface
    max_depth: float = 3e-3  # m, depth clamp (= gel thickness)
    marker_shift_px: tuple[float, float] = (-5.0, 0.0)

    @property
    def depth_offset(self) -> float:
        return abs(self.ray_start_level - self.ray_rest_level)

    @property
    def pixel_size_m(self) -> float:
        return self.pixel_size_mm * 1e-3


@dataclass
class MaterialConfig:
    """Gel material. Taccel's defaults for the GelSight-like pad."""

    density: float = 1e3  # kg/m^3
    E: float = 5e5  # Pa
    nu: float = 0.4
    mu: float = 1.0  # friction coefficient of the gel surface


@dataclass
class SensorAsset:
    """One gel pad + its masks + its markers + its camera."""

    name: str
    link_name: str
    points: np.ndarray  # (N, 3) float64, rest positions in sensor-local frame
    tets: np.ndarray  # (M, 4) int32
    stick_mask: np.ndarray  # (N,) bool -- attach to carrier
    coat_mask: np.ndarray  # (N,) bool -- sensing surface
    marker_vert_idx: np.ndarray  # (K, 3) int -- body vertex ids
    marker_bc_coords: np.ndarray  # (K, 3) float
    attach_rel_pose: np.ndarray = field(default_factory=lambda: np.zeros(6))  # xyz + rpy
    camera: CameraConfig = field(default_factory=CameraConfig)
    material: MaterialConfig = field(default_factory=MaterialConfig)
    source_files: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- geometry

    @cached_property
    def surface_faces(self) -> np.ndarray:
        """All outward boundary triangles of the tet mesh, body indices."""
        return boundary_faces(self.tets, self.points)

    @cached_property
    def coat_faces(self) -> np.ndarray:
        """Boundary triangles fully inside the coat mask -- the raycast target."""
        return select_faces_by_vertex_mask(self.surface_faces, self.coat_mask)

    @cached_property
    def attach_rel_tf(self) -> np.ndarray:
        """4x4 sensor->carrier-link transform built from `attach_rel_pose`."""
        from scipy.spatial.transform import Rotation as R  # optional dep

        tf = np.eye(4)
        tf[:3, 3] = self.attach_rel_pose[:3]
        tf[:3, :3] = R.from_euler("xyz", self.attach_rel_pose[3:]).as_matrix()
        return tf

    @property
    def n_verts(self) -> int:
        return self.points.shape[0]

    @property
    def n_markers(self) -> int:
        return self.marker_vert_idx.shape[0]

    # ---------------------------------------------------------------- markers

    def markers_from_verts(self, verts: np.ndarray) -> np.ndarray:
        """Marker positions for a given FEM state.

        Args:
            verts: (N, 3) current positions of THIS gel's vertices (any frame).
        Returns:
            (K, 3) marker positions in the same frame.
        """
        tri = verts[self.marker_vert_idx]  # (K, 3, 3)
        return (tri * self.marker_bc_coords[..., None]).sum(axis=-2)

    @cached_property
    def rest_markers(self) -> np.ndarray:
        """Marker positions in the undeformed sensor-local frame."""
        return self.markers_from_verts(self.points)

    def reframed(self, R: np.ndarray, t: np.ndarray, name: str = "") -> "SensorAsset":
        """Same gel in a rotated/translated LOCAL frame.

        The observation code assumes the sensing convention of Taccel's flat pad:
        coat at local z=0, body extending to +z, object arriving from -z. Curved
        fingertip pads do not ship that way -- the Allegro digit's coat is the
        bulge around local x=+16.71 mm with the glued face flat at x=+13.71 --
        so they must be reframed once at load:

            R = [[0,0,1],[0,1,0],[-1,0,0]]; t = [0, 0, 16.71e-3]
            -> new = (old_z, old_y, 16.71mm - old_x)   (det +1, a real rotation)

        Marker indices/weights are index-based so they carry over untouched.
        """
        R = np.asarray(R, dtype=np.float64)
        if abs(np.linalg.det(R) - 1.0) > 1e-9:
            raise ValueError(f"R must be a rotation (det={np.linalg.det(R):.6f}); "
                             "a mirror would flip every surface normal")
        return SensorAsset(
            name=name or f"{self.name}_reframed",
            link_name=self.link_name,
            points=self.points @ R.T + np.asarray(t, dtype=np.float64),
            tets=self.tets.copy(),
            stick_mask=self.stick_mask.copy(),
            coat_mask=self.coat_mask.copy(),
            marker_vert_idx=self.marker_vert_idx.copy(),
            marker_bc_coords=self.marker_bc_coords.copy(),
            attach_rel_pose=self.attach_rel_pose.copy(),
            camera=self.camera,
            material=self.material,
            source_files=dict(self.source_files, reframed="yes"),
        )

    # ---------------------------------------------------------------- loading

    @classmethod
    def from_taccel_fabrication(
        cls,
        fab_json: str,
        index: int = 0,
        camera: CameraConfig | None = None,
        material: MaterialConfig | None = None,
    ) -> "SensorAsset":
        """Load sensor `index` from a Taccel ``tac_fabr_*.json`` / ``tac_fab_*.json``.

        Example:
            SensorAsset.from_taccel_fabrication(
                "<taccel>/assets/robots/single_sensor/tac_fabr_1e-07.json")
        """
        entries = json.load(open(fab_json, "r"))
        fab = entries[index]
        base = osp.dirname(fab_json)

        mesh_dir = osp.dirname(fab["mesh_path"])
        vtk = osp.join(base, fab.get("mesh_vtk_file", osp.join(mesh_dir, f"pad_maxv={fab['reso']}.vtk")))
        pkl = osp.join(base, fab.get("mesh_metadata_file", osp.join(mesh_dir, f"pad_maxv={fab['reso']}.pkl")))

        points, tets = read_tet_mesh(vtk)
        meta = load_fabrication_metadata(pkl)

        stick = np.asarray(meta["stick_mask"], dtype=bool)
        coat = np.asarray(meta["coat_mask"], dtype=bool)
        if stick.shape != (points.shape[0],) or coat.shape != (points.shape[0],):
            raise ValueError(
                f"mask/mesh mismatch: {points.shape[0]} verts vs stick {stick.shape} coat {coat.shape}"
            )

        return cls(
            name=fab.get("sensor_name", f"{fab['link_name']}_vbts"),
            link_name=fab["link_name"],
            points=points,
            tets=tets,
            stick_mask=stick,
            coat_mask=coat,
            marker_vert_idx=np.asarray(meta["marker_vert_idx"], dtype=np.int64).reshape(-1, 3),
            marker_bc_coords=np.asarray(meta["marker_bc_coords"], dtype=np.float64).reshape(-1, 3),
            attach_rel_pose=np.asarray(list(fab["pos"]) + list(fab["rot"]), dtype=np.float64),
            camera=camera or CameraConfig(),
            material=material or MaterialConfig(),
            source_files={"fab_json": fab_json, "vtk": vtk, "pkl": pkl},
        )

    @classmethod
    def all_from_taccel_fabrication(cls, fab_json: str, **kw) -> list["SensorAsset"]:
        """Load every sensor in a fabrication file (e.g. 15 pads of the F-TAC hand)."""
        n = len(json.load(open(fab_json, "r")))
        return [cls.from_taccel_fabrication(fab_json, i, **kw) for i in range(n)]

    def summary(self) -> str:
        return (
            f"SensorAsset({self.name}, link={self.link_name}): "
            f"{self.n_verts} verts / {self.tets.shape[0]} tets, "
            f"{int(self.stick_mask.sum())} stick, {int(self.coat_mask.sum())} coat, "
            f"{len(self.coat_faces)} coat tris, {self.n_markers} markers, "
            f"E={self.material.E:.3g} nu={self.material.nu} mu={self.material.mu}, "
            f"cam={self.camera.height}x{self.camera.width}@{self.camera.pixel_size_mm}mm"
        )

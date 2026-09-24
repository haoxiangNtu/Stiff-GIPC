"""SDK-style output layer over the simulated VBTS (vendor-neutral naming).

Commercial VBTS pads converge on one de-facto output vocabulary (an OutputType
enum served by a same-frame multi-output query). This module exposes that
vocabulary over a simulated GelBody so code written against a commercial
sensor SDK maps 1:1 onto the simulation — and the coverage claim is checkable
by running it rather than by a table in a chat message.

Mapping, with the honest caveats inline:

  Rectify        n2rgb MLP tactile image. CAVEAT: colour style follows the
                 Taccel GelSight calibration — geometry is faithful, the
                 palette is one specific sensor family's.
  Difference     Rectify minus the reference frame (calibrateSensor()).
  Depth          indentation depth map in mm (depth - reference), the
                 validated core output of this stack.
  Marker2D       (K,2) marker pixel positions.
  Marker3D       (K,3) marker positions in the sensor frame, mm.
  Marker3DFlow   (K,3) marker displacement vs reference, mm.
  Force          distributed 3D contact force at the coat vertices, sensor
                 frame, N — returned with the vertex positions (the exact
                 per-vertex field the IPC solver computed; shear slip ratio
                 validated 0.600).
  ForceNorm      normal (z) component of that distribution, N.
  ForceResultant 6-D [Fx,Fy,Fz,Tx,Ty,Tz]: force in N, torque in N*m about
                 the sensor-frame origin (validated on the bolt lift:
                 tangential sum = -m*g).
  Mesh3D         coat surface mesh vertices in the sensor frame, mm.
  Mesh3DInit     the reference (rest) coat mesh, mm.
  Mesh3DFlow     Mesh3D - Mesh3DInit, mm.
  TimeStamp      simulation time in seconds (frames x dt).

Lifecycle (mirrors the commercial SDKs' local mode):

  TactileSensor.scanSerialNumber()   -> {serial: index} of live sim sensors
  TactileSensor.create("SIM000001")  -> the registered sensor (or by index)
  sensor.release()                   -> deregister and drop engine references

Post-processing mode (their record-and-replay workflow, sim semantics):

  live loop:   snap = sensor.snapshot()        # raw state: verts + forces + t
  once:        sensor.exportRuntimeConfig(dir) # config + reference frames
  offline:     solver = TactileSensor.createSolver(runtime_path)
               solver.selectSensorInfo(*outs, verts_local=..., forces=..., t=...)

The offline solver RECOMPUTES the imaging outputs (depth raster, MLP RGB,
markers, meshes) from the recorded raw vertex state on CPU — no engine, no
GPU renderer — and replays the recorded force field for the force outputs
(forces cannot be recomputed without the solver). Fixed input order + the
recorded reference = reproducible regression runs.

Not mapped, deliberately: serial-number *hardware* discovery and licensing
(no simulation meaning); the vendors' image->depth/force inverse solvers
(their IP; the simulation has ground truth and does not need them).
"""

from __future__ import annotations

import json
import os
import os.path as osp
from enum import Enum, auto

import numpy as np


class OutputType(Enum):
    Rectify = auto()
    Difference = auto()
    Depth = auto()
    Marker2D = auto()
    Marker3D = auto()
    Marker3DFlow = auto()
    Force = auto()
    ForceNorm = auto()
    ForceResultant = auto()
    Mesh3D = auto()
    Mesh3DInit = auto()
    Mesh3DFlow = auto()
    TimeStamp = auto()


class TactileSensor:
    """One simulated pad speaking the commercial-SDK output vocabulary.

    Args:
        gel: backend GelBody (already in a finalized engine).
        gpu: GpuTactileRenderer for this gel (depth/markers).
        rgb: optional TactileRenderer (n2rgb MLP) — needed for
             Rectify/Difference only.
        dt:  engine step time, for TimeStamp.
        serial: sensor id; auto-assigned "SIM000001", ... when omitted.
    """

    OutputType = OutputType          # Sensor.OutputType.* like the real SDKs
    _registry: dict[str, "TactileSensor"] = {}
    _serial_seq = 0

    def __init__(self, gel, gpu, rgb=None, dt: float = 0.005,
                 serial: str | None = None):
        self.gel = gel
        self.gpu = gpu
        self.rgb = rgb
        self.dt = dt
        self.frame = 0
        self._coat = np.nonzero(gel.asset.coat_mask)[0]
        if serial is None:
            TactileSensor._serial_seq += 1
            serial = f"SIM{TactileSensor._serial_seq:06d}"
        self.serial = serial
        TactileSensor._registry[serial] = self
        self.calibrateSensor()

    # ----------------------------------------------------------- lifecycle
    @classmethod
    def scanSerialNumber(cls) -> dict[str, int]:
        """{serial: index} of every live sim sensor (their scan, sim scope)."""
        return {s: i for i, s in enumerate(cls._registry)}

    @classmethod
    def create(cls, serial_or_index) -> "TactileSensor":
        """Look up a registered sensor by serial string or scan index."""
        if isinstance(serial_or_index, str):
            return cls._registry[serial_or_index]
        return list(cls._registry.values())[int(serial_or_index)]

    def release(self) -> None:
        """Deregister and drop engine references (their resource release)."""
        TactileSensor._registry.pop(self.serial, None)
        self.gel = self.gpu = self.rgb = None

    # ------------------------------------------------------------ reference
    def calibrateSensor(self, sensor_tf=None):
        """Refresh the no-contact reference (their unloaded-sensor refresh)."""
        obs = self.gpu.observe(sensor_tf=sensor_tf, with_markers=True,
                               smooth_sigma_px=1.0, normal_mode="metric")
        self._ref_depth = obs["depth"].cpu().numpy()
        self._ref_mesh = self.gel.verts_local(sensor_tf)[self._coat] * 1e3
        self._ref_mk3 = self.gel.asset.markers_from_verts(
            self.gel.verts_local(sensor_tf)) * 1e3
        self._ref_rect = self._rectify(obs) if self.rgb is not None else None

    def tick(self, n: int = 1):
        """Advance the frame counter alongside engine.step() calls."""
        self.frame += n

    # ------------------------------------------------------------- internals
    def _rectify(self, obs):
        from .observation import depth_to_normal

        d = obs["depth"].cpu().numpy()
        n = depth_to_normal(d, self.gel.asset.camera.pixel_size_m)
        return self.rgb.render(d[None], n[None])[0]

    # ---------------------------------------------------------------- query
    def selectSensorInfo(self, *outputs: OutputType, sensor_tf=None):
        """Serve every requested output from ONE engine snapshot (same-frame
        guarantee). Returns a single value for one output, else a tuple in
        parameter order."""
        obs = self.gpu.observe(sensor_tf=sensor_tf, with_markers=True,
                               smooth_sigma_px=1.0, normal_mode="metric")
        depth_mm = np.clip(obs["depth"].cpu().numpy() - self._ref_depth,
                           0, None) * 1e3
        loc = self.gel.verts_local(sensor_tf)
        coat_mm = loc[self._coat] * 1e3
        need_force = any(o in (OutputType.Force, OutputType.ForceNorm,
                               OutputType.ForceResultant) for o in outputs)
        if need_force:
            tf = self.gel.world_tf if sensor_tf is None else np.asarray(sensor_tf)
            F_all = self.gel.contact_force_map("total") @ tf[:3, :3]
            F_coat = F_all[self._coat]
            # Resultant over the SENSING SURFACE only — a real pad reports
            # what its membrane feels, not what the solver knows about the
            # pad's side walls (an oversized object overhanging the pad edge
            # loads non-coat vertices; the verification assert caught the
            # all-vertex version disagreeing with SumForceNorm).
            pos_c = loc[self._coat]  # metres, sensor frame
            resultant = np.concatenate(
                [F_coat.sum(0), np.cross(pos_c, F_coat).sum(0)])

        out = []
        for o in outputs:
            if o is OutputType.Rectify:
                out.append(self._rectify(obs))
            elif o is OutputType.Difference:
                out.append(self._rectify(obs).astype(np.float32)
                           - self._ref_rect.astype(np.float32))
            elif o is OutputType.Depth:
                out.append(depth_mm)
            elif o is OutputType.Marker2D:
                out.append(obs["markers_px"].cpu().numpy())
            elif o is OutputType.Marker3D:
                out.append(self.gel.asset.markers_from_verts(loc) * 1e3)
            elif o is OutputType.Marker3DFlow:
                out.append(self.gel.asset.markers_from_verts(loc) * 1e3
                           - self._ref_mk3)
            elif o is OutputType.Force:
                out.append((coat_mm.copy(), F_coat))
            elif o is OutputType.ForceNorm:
                out.append(F_coat[:, 2].copy())
            elif o is OutputType.ForceResultant:
                out.append(resultant)
            elif o is OutputType.Mesh3D:
                out.append(coat_mm.copy())
            elif o is OutputType.Mesh3DInit:
                out.append(self._ref_mesh.copy())
            elif o is OutputType.Mesh3DFlow:
                out.append(coat_mm - self._ref_mesh)
            elif o is OutputType.TimeStamp:
                out.append(self.frame * self.dt)
            else:
                raise ValueError(f"unknown output {o}")
        return out[0] if len(out) == 1 else tuple(out)

    # -------------------------------------------------- post-processing mode
    def snapshot(self, sensor_tf=None) -> dict:
        """Raw per-frame state for offline replay: sensor-local vertices, the
        coat contact-force field (sensor frame, N) and the timestamp. This is
        the sim analogue of persisting the raw camera frames."""
        tf = self.gel.world_tf if sensor_tf is None else np.asarray(sensor_tf)
        loc = self.gel.verts_local(sensor_tf)
        F_coat = (self.gel.contact_force_map("total") @ tf[:3, :3])[self._coat]
        return dict(verts_local=loc.astype(np.float32),
                    forces=F_coat.astype(np.float32),
                    t=self.frame * self.dt)

    def exportRuntimeConfig(self, save_dir: str) -> str:
        """Write runtime_<serial>.npz: sensor config + reference frames + the
        gel asset arrays the offline solver needs. Unencrypted — the sim has
        no licensing to protect."""
        os.makedirs(save_dir, exist_ok=True)
        a = self.gel.asset
        cam = a.camera
        cfg = dict(serial=self.serial, dt=self.dt,
                   cam_height=cam.height, cam_width=cam.width,
                   cam_pixel_size_mm=cam.pixel_size_mm,
                   cam_ray_start_level=cam.ray_start_level,
                   cam_ray_rest_level=cam.ray_rest_level,
                   cam_max_depth=cam.max_depth,
                   cam_marker_shift_px=list(cam.marker_shift_px),
                   rgb_paths=getattr(self.rgb, "source_paths", None))
        path = osp.join(save_dir, f"runtime_{self.serial}.npz")
        arrays = dict(points=a.points, tets=a.tets,
                      stick_mask=a.stick_mask, coat_mask=a.coat_mask,
                      marker_vert_idx=a.marker_vert_idx,
                      marker_bc_coords=a.marker_bc_coords,
                      ref_depth=self._ref_depth, ref_mesh=self._ref_mesh,
                      ref_mk3=self._ref_mk3,
                      config=np.frombuffer(json.dumps(cfg).encode(), np.uint8))
        if self._ref_rect is not None:
            arrays["ref_rect"] = self._ref_rect
        np.savez_compressed(path, **arrays)
        return path

    @staticmethod
    def createSolver(runtime_path: str) -> "OfflineSolver":
        return OfflineSolver(runtime_path)


class OfflineSolver:
    """Offline replay of the sensor pipeline from recorded raw state.

    Recomputes Depth (CPU raster), Rectify/Difference (MLP), markers and
    meshes from ``verts_local``; force outputs replay the recorded field.
    No engine, no GPU. Same ``selectSensorInfo`` shape as the live sensor,
    plus the per-frame raw inputs:

        solver.selectSensorInfo(OT.Depth, OT.Rectify,
                                verts_local=snap["verts_local"],
                                forces=snap["forces"], t=snap["t"])
    """

    OutputType = OutputType

    def __init__(self, runtime_path: str):
        from .sensor_asset import CameraConfig, SensorAsset

        z = np.load(runtime_path, allow_pickle=False)
        cfg = json.loads(bytes(z["config"]).decode())
        self.cfg = cfg
        cam = CameraConfig(height=int(cfg["cam_height"]),
                           width=int(cfg["cam_width"]),
                           pixel_size_mm=float(cfg["cam_pixel_size_mm"]),
                           ray_start_level=float(cfg["cam_ray_start_level"]),
                           ray_rest_level=float(cfg["cam_ray_rest_level"]),
                           max_depth=float(cfg["cam_max_depth"]),
                           marker_shift_px=tuple(cfg["cam_marker_shift_px"]))
        self.asset = SensorAsset(
            name=f"offline_{cfg['serial']}", link_name="offline",
            points=z["points"], tets=z["tets"],
            stick_mask=z["stick_mask"], coat_mask=z["coat_mask"],
            marker_vert_idx=z["marker_vert_idx"],
            marker_bc_coords=z["marker_bc_coords"], camera=cam)
        self._coat = np.nonzero(self.asset.coat_mask)[0]
        self._ref_depth = z["ref_depth"]
        self._ref_mesh = z["ref_mesh"]
        self._ref_mk3 = z["ref_mk3"]
        self._ref_rect = z["ref_rect"] if "ref_rect" in z.files else None
        self.rgb = None
        if cfg.get("rgb_paths"):
            try:
                from .render import TactileRenderer
                self.rgb = TactileRenderer(*cfg["rgb_paths"],
                                           resolution=(cam.height, cam.width))
            except Exception as exc:
                print(f"[offline] RGB renderer unavailable ({exc}); "
                      "Rectify/Difference disabled")

    def release(self):
        self.rgb = None

    def _rectify(self, depth_m):
        from .observation import depth_to_normal

        n = depth_to_normal(depth_m, self.asset.camera.pixel_size_m)
        return self.rgb.render(depth_m[None], n[None])[0]

    def selectSensorInfo(self, *outputs: OutputType, verts_local=None,
                         forces=None, t=0.0):
        from .observation import observe

        OT = OutputType
        v = np.asarray(verts_local, np.float64)
        need_img = any(o in (OT.Rectify, OT.Difference, OT.Depth, OT.Marker2D)
                       for o in outputs)
        if need_img:
            obs = observe(self.asset, v, with_markers=True,
                          normal_mode="metric", smooth_sigma_px=1.0)
            depth_mm = np.clip(obs["depth"] - self._ref_depth, 0, None) * 1e3
        coat_mm = v[self._coat] * 1e3
        if forces is not None:
            F = np.asarray(forces, np.float64)
            resultant = np.concatenate(
                [F.sum(0), np.cross(v[self._coat], F).sum(0)])

        out = []
        for o in outputs:
            if o is OT.Rectify:
                out.append(self._rectify(obs["depth"]))
            elif o is OT.Difference:
                out.append(self._rectify(obs["depth"]).astype(np.float32)
                           - self._ref_rect.astype(np.float32))
            elif o is OT.Depth:
                out.append(depth_mm)
            elif o is OT.Marker2D:
                out.append(obs["markers_px"])
            elif o is OT.Marker3D:
                out.append(self.asset.markers_from_verts(v) * 1e3)
            elif o is OT.Marker3DFlow:
                out.append(self.asset.markers_from_verts(v) * 1e3 - self._ref_mk3)
            elif o is OT.Force:
                out.append((coat_mm.copy(), F))
            elif o is OT.ForceNorm:
                out.append(F[:, 2].copy())
            elif o is OT.ForceResultant:
                out.append(resultant)
            elif o is OT.Mesh3D:
                out.append(coat_mm.copy())
            elif o is OT.Mesh3DInit:
                out.append(self._ref_mesh.copy())
            elif o is OT.Mesh3DFlow:
                out.append(coat_mm - self._ref_mesh)
            elif o is OT.TimeStamp:
                out.append(float(t))
            else:
                raise ValueError(f"unknown output {o}")
        return out[0] if len(out) == 1 else tuple(out)

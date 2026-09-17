"""XenseSDK-style output alignment layer over the simulated VBTS.

docs.xenserobotics.com defines the de-facto output vocabulary for commercial
VBTS pads (XenseSDK 2.1.2, OutputType enum + `selectSensorInfo`). This module
exposes the SAME vocabulary over a simulated GelBody, so code written against
their naming maps 1:1 onto the simulation and the coverage claim is checkable
by running it rather than by a table in a chat message.

Mapping, with the honest caveats inline:

  Rectify        n2rgb MLP tactile image. CAVEAT: colour style follows the
                 Taccel GelSight calibration, not Xense's optics -- geometry
                 is faithful, the palette is another sensor's.
  Difference     Rectify minus the reference frame (calibrateSensor()).
  Depth          indentation depth map in mm (depth - reference), the
                 validated core output of this stack.
  Marker2D       (K,2) marker pixel positions.
  Marker3D       (K,3) marker positions in the sensor frame, mm.
  Marker3DFlow   (K,3) marker displacement vs reference, mm.
  Force          distributed 3D contact force at the coat vertices, sensor
                 frame, N -- returned with the vertex positions (theirs is a
                 field over their mesh; ours is the exact per-vertex field
                 the IPC solver computed, shear slip ratio validated 0.600).
  ForceNorm      normal (z) component of that distribution, N.
  ForceResultant 6-D [Fx,Fy,Fz,Tx,Ty,Tz]: force in N, torque in N*m about
                 the sensor-frame origin (validated on the bolt lift:
                 tangential sum = -m*g).
  Mesh3D         coat surface mesh vertices in the sensor frame, mm.
  Mesh3DInit     the reference (rest) coat mesh, mm.
  Mesh3DFlow     Mesh3D - Mesh3DInit, mm.
  TimeStamp      simulation time in seconds (frames x dt).

`selectSensorInfo(*outputs)` reads ONE engine snapshot and serves every
requested output from it -- the same same-frame guarantee their SDK makes.
`calibrateSensor()` re-captures the reference exactly like their no-contact
reference refresh.

Not mapped, deliberately: exportRuntimeConfig / scanSerialNumber (hardware
licensing and discovery -- no simulation meaning).
"""

from __future__ import annotations

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


class XenseStyleSensor:
    """One simulated pad speaking the XenseSDK output vocabulary.

    Args:
        gel: backend GelBody (already in a finalized engine).
        gpu: GpuTactileRenderer for this gel (depth/markers).
        rgb: optional TactileRenderer (n2rgb MLP) -- needed for
             Rectify/Difference only.
        dt:  engine step time, for TimeStamp.
    """

    def __init__(self, gel, gpu, rgb=None, dt: float = 0.005):
        self.gel = gel
        self.gpu = gpu
        self.rgb = rgb
        self.dt = dt
        self.frame = 0
        self._coat = np.nonzero(gel.asset.coat_mask)[0]
        self.calibrateSensor()

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
            # Resultant over the SENSING SURFACE only -- a real pad reports
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

"""Finger-face VBTS pad sized for the Franka hand fingertip.

Pad-local frame = flat-pad convention (coat at z=0 facing the object, body to
+z).  Camera pitch is Taccel's 0.079375 mm; the pad is cut to an integer
number of pixels so the orthographic depth image covers it exactly.
"""

from __future__ import annotations

import os
import os.path as osp

PAD_CAM = (200, 176)                       # px: height (pad x), width (pad y)
PIXEL_MM = 0.079375
PAD_SIZE = (PAD_CAM[0] * PIXEL_MM * 1e-3, PAD_CAM[1] * PIXEL_MM * 1e-3, 2.5e-3)
PAD_MAX_VOLUME = 4.0e-10

HERE = osp.dirname(osp.abspath(__file__))
REPO = osp.abspath(osp.join(HERE, "..", ".."))
PAD_DIR = osp.join(REPO, "Assets", "lab_tactile", "pad")


def fabricate_franka_pad(force: bool = False) -> str:
    from stiff_physics.sensors.fabricate import FabricationSpec, fabricate, write_fabrication_json
    from stiff_physics.sensors.linkerhand import pad_box_stl

    j = osp.join(PAD_DIR, f"tac_fabr_{PAD_MAX_VOLUME:g}.json")
    if osp.exists(j) and not force:
        return j
    os.makedirs(osp.join(PAD_DIR, "meshes"), exist_ok=True)
    stl = pad_box_stl(osp.join(PAD_DIR, "meshes", "franka_pad.stl"), size=PAD_SIZE, seg_mm=1.4)
    spec = FabricationSpec(stl_path=stl, max_volume=PAD_MAX_VOLUME, coat_z_max=1e-4,
                           stick_z_min=PAD_SIZE[2] - 1e-4, marker_extent_mm=5.0,
                           marker_grid=6, out_dir=osp.join(PAD_DIR, "meshes"), name="pad")
    fabricate(spec, verbose=True)
    write_fabrication_json(j, "finger", f"meshes/pad_maxv={PAD_MAX_VOLUME:g}.vtk", PAD_MAX_VOLUME)
    return j


def load_pad_asset():
    from stiff_physics.sensors import CameraConfig, SensorAsset
    j = fabricate_franka_pad()
    cam = CameraConfig(height=PAD_CAM[0], width=PAD_CAM[1], max_depth=PAD_SIZE[2])
    return SensorAsset.from_taccel_fabrication(j, 0, camera=cam)


if __name__ == "__main__":
    a = load_pad_asset()
    print(a.summary())
    print("pad size mm", [round(s * 1e3, 3) for s in PAD_SIZE])

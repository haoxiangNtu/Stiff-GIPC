"""Generic parallel-jaw gripper with finger-face VBTS pads (digital-twin rig).

A two-jaw parallel gripper is the LOWEST-risk tactile configuration in this
stack: the jaws close by pure translation, so the two pads are parallel by
construction -- the pad-opposition tilt that limits the Allegro (hard 15 deg)
and the L20 (6.3 deg after co-design) does not exist here, and the drive is
the translation-only stitch pattern every validated scene uses.

This module supplies only what the hand modules supply for their hands:
  * procedural VISUAL geometry (base + two jaws) -- the physics scene, as
    everywhere else in this stack, contains carriers + FEM gels + object;
  * the pad MOUNT frames on the jaw inner faces;
  * a fabrication spec for the finger-face pad (40 x 20 x 3 mm by default,
    camera 504x252 @ 0.079375 mm = the pad exactly).

Frames: the gripper closes along world Z (jaws at z = +-opening/2), fingers
stand along +Y, the grasped object's long axis runs along X. Pad-local frame
is the flat-pad convention (coat z=0 facing the object, body +z into the jaw);
pad-local x = world X (object axis), pad-local y = world Y (vertical).
"""

from __future__ import annotations

import os.path as osp
from dataclasses import dataclass

import numpy as np

PAD_SIZE = (40e-3, 20e-3, 3e-3)          # pad-local x, y, thickness
PAD_CAM = (504, 252)                     # height (x), width (y) @ 0.079375 mm
JAW = dict(w=16e-3, h=46e-3, d=26e-3)    # jaw block: thickness(z), height(y), depth(x)... d along x
BASE = dict(w=110e-3, h=26e-3, d=46e-3)  # crossbar above the jaws


def _box(size, centre):
    sx, sy, sz = size
    v = np.array([[x, y, z] for x in (-sx / 2, sx / 2) for y in (-sy / 2, sy / 2)
                  for z in (-sz / 2, sz / 2)], float) + np.asarray(centre, float)
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]],
                 np.int32)
    c = v.mean(0)
    for i, t in enumerate(f):
        p = v[t]
        if np.dot(np.cross(p[1] - p[0], p[2] - p[0]), p.mean(0) - c) < 0:
            f[i] = t[::-1]
    return v, f


@dataclass
class GripperRig:
    """Visual geometry in GRIPPER-LOCAL frames + pad mounts.

    Jaw local frame: jaw inner face at z=0 (pad glues onto it), body extends
    to +z; the LEFT jaw (at world -z) is the same geometry mirrored by the
    rig's placement transform, not by a special mesh.
    """

    grasp_centre: np.ndarray             # world point between the pads
    pad_size: tuple = PAD_SIZE

    def jaw_mesh(self):
        w, h, d = JAW["w"], JAW["h"], JAW["d"]
        return _box((d, h, w), (0.0, h / 2 - 14e-3, w / 2))

    def base_mesh(self):
        w, h, d = BASE["w"], BASE["h"], BASE["d"]
        return _box((d, h, w), (0.0, JAW["h"] - 14e-3 + h / 2, 0.0))

    def jaw_tf(self, side: int, opening: float) -> np.ndarray:
        """side=+1 world +z jaw, -1 world -z jaw; opening = pad-face gap."""
        T = np.eye(4)
        if side < 0:                     # mirror by 180 deg about Y
            T[0, 0] = T[2, 2] = -1.0
        T[:3, 3] = self.grasp_centre + np.array(
            [0.0, 0.0, side * (opening / 2 + self.pad_size[2])])
        return T

    def base_tf(self, lift: float = 0.0) -> np.ndarray:
        T = np.eye(4)
        T[:3, 3] = self.grasp_centre + np.array([0.0, lift, 0.0])
        return T

    def pad_world_tf(self, side: int, opening: float) -> np.ndarray:
        """Flat-pad convention: coat z=0 faces the object (toward -side)."""
        T = np.eye(4)
        if side > 0:                     # coat normal -z must point at object
            pass                         # pad local z == world z: object at -z ok
        else:
            T[0, 0] = T[2, 2] = -1.0     # rotate 180 about Y: local z -> world -z
        T[:3, 3] = self.grasp_centre + np.array([0.0, 0.0, side * opening / 2])
        return T


def fabricate_pad(out_root: str, max_volume: float = 2e-10, force: bool = False):
    """Build (once) the gripper finger pad; returns the tac_fabr json path."""
    from .fabricate import FabricationSpec, fabricate, write_fabrication_json
    from .linkerhand import pad_box_stl

    d = osp.join(out_root, "assets", "gripper_pad")
    j = osp.join(d, f"tac_fabr_{max_volume:g}.json")
    if osp.exists(j) and not force:
        return j
    import os
    os.makedirs(osp.join(d, "meshes"), exist_ok=True)
    stl = pad_box_stl(osp.join(d, "meshes", "gripper_pad.stl"),
                      size=PAD_SIZE, seg_mm=1.5)
    spec = FabricationSpec(stl_path=stl, max_volume=max_volume, coat_z_max=1e-4,
                           stick_z_min=PAD_SIZE[2] - 1e-4, marker_extent_mm=8.0,
                           marker_grid=7, out_dir=osp.join(d, "meshes"),
                           name="pad")
    fabricate(spec, verbose=False)
    write_fabrication_json(j, "jaw", f"meshes/pad_maxv={max_volume:g}.vtk",
                           max_volume)
    return j

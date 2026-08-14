"""Offscreen 3D rendering with real shading (pyrender + EGL).

The matplotlib Poly3DCollection "renders" used in the first demo videos have no
depth-correct shading, no smooth normals and no lighting model -- next to a
Houdini/Omniverse viewport they read as triangle soup, and the user said so.
This is the replacement: a headless PBR-ish viewport look (smooth vertex
normals, two directional lights + ambient, neutral studio background) matching
what a DCC shows when the same stage is opened interactively.

Meshes coming from SolidWorks STLs are vertex soup; `merge=True` welds
duplicates so smooth vertex normals exist -- that single flag is most of the
difference between "CAD soup" and "viewport" appearance.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np


class SceneRenderer:
    """Reusable offscreen renderer: static topology, per-frame poses."""

    def __init__(self, width: int = 1100, height: int = 860,
                 bg=(0.90, 0.92, 0.95)):
        import pyrender

        self._pr = pyrender
        self.r = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene(bg_color=[*bg, 1.0],
                                    ambient_light=[0.35, 0.35, 0.38])
        self._nodes = {}

    # ------------------------------------------------------------------ build
    def add_mesh(self, name: str, verts: np.ndarray, faces: np.ndarray,
                 color=(0.55, 0.55, 0.58), merge: bool = True,
                 roughness: float = 0.6, pose: np.ndarray | None = None):
        import trimesh

        m = trimesh.Trimesh(np.asarray(verts, np.float64),
                            np.asarray(faces, np.int64), process=False)
        if merge:
            m.merge_vertices()                     # weld soup -> smooth normals
        mat = self._pr.MetallicRoughnessMaterial(
            baseColorFactor=[*color, 1.0], metallicFactor=0.05,
            roughnessFactor=roughness)
        pm = self._pr.Mesh.from_trimesh(m, material=mat, smooth=merge)
        node = self.scene.add(pm, pose=np.eye(4) if pose is None else pose,
                              name=name)
        self._nodes[name] = node
        return node

    def set_pose(self, name: str, T: np.ndarray):
        self.scene.set_pose(self._nodes[name], np.asarray(T, np.float64))

    def look_at(self, eye, target, up=(0, 0, 1), yfov: float = 0.75):
        eye = np.asarray(eye, float)
        target = np.asarray(target, float)
        z = eye - target
        z = z / np.linalg.norm(z)                   # camera looks along -z
        x = np.cross(np.asarray(up, float), z)
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, eye
        if "cam" not in self._nodes:
            cam = self._pr.PerspectiveCamera(yfov=yfov)
            self._nodes["cam"] = self.scene.add(cam, pose=T, name="cam")
            key = self._pr.DirectionalLight(intensity=3.2)
            fill = self._pr.DirectionalLight(intensity=1.2)
            self._nodes["key"] = self.scene.add(key, pose=T, name="key")
            Tf = T.copy()
            Tf[:3, :3] = Tf[:3, :3] @ self._rot_y(2.2)
            self._nodes["fill"] = self.scene.add(fill, pose=Tf, name="fill")
        else:
            self.scene.set_pose(self._nodes["cam"], T)
            self.scene.set_pose(self._nodes["key"], T)

    @staticmethod
    def _rot_y(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    # ------------------------------------------------------------------ frame
    def render(self) -> np.ndarray:
        img, _ = self.r.render(self.scene)
        return img

    def close(self):
        self.r.delete()

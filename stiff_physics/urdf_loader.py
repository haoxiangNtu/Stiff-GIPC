"""Python URDF parser for StiffGIPC.

Ported from ``rbs_physics.urdf_loader.UrdfLoader`` with simplifications.
Parses link/joint trees, computes global transforms, and collects collision
mesh paths (supports multiple collision entries per link).
"""

from __future__ import annotations

import pathlib as pl
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass
class MeshLinkInfo:
    """Geometry and transform info for one URDF link."""
    name: str = ""
    transform: np.ndarray = field(default_factory=lambda: np.eye(4))
    visual_mesh_directories: list[str] = field(default_factory=list)
    visual_mesh_transforms: list[np.ndarray] = field(default_factory=list)
    collision_mesh_directories: list[str] = field(default_factory=list)
    collision_mesh_transforms: list[np.ndarray] = field(default_factory=list)


@dataclass
class LinkInfo:
    name: str = ""
    has_mesh: bool = False
    transform: np.ndarray = field(default_factory=lambda: np.eye(4))


@dataclass
class RevoluteJointInfo:
    name: str = ""
    parent_link_name: str = ""
    child_link_name: str = ""
    parent_mesh_link_name: str | None = None
    child_mesh_link_name: str | None = None

    local_point_0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    local_point_1: np.ndarray | None = None
    local_axis: np.ndarray | None = None
    global_point_0: np.ndarray | None = None
    global_point_1: np.ndarray | None = None

    lower_limit: float = 0.0
    upper_limit: float = 0.0


@dataclass
class FixedJointInfo:
    name: str = ""
    parent_link_name: str = ""
    child_link_name: str = ""
    parent_mesh_link_name: str | None = None
    child_mesh_link_name: str | None = None


@dataclass
class PrismaticJointInfo:
    name: str = ""
    parent_link_name: str = ""
    child_link_name: str = ""
    parent_mesh_link_name: str | None = None
    child_mesh_link_name: str | None = None

    local_axis: np.ndarray | None = None
    global_point_0: np.ndarray | None = None
    global_axis: np.ndarray | None = None

    lower_limit: float = 0.0
    upper_limit: float = 0.0


def _xyz_rpy_to_transform(xyz, rpy) -> np.ndarray:
    rot = R.from_euler("xyz", rpy).as_matrix()
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = xyz
    return t


class UrdfLoader:
    """Parse a URDF file and extract link geometry + joint info.

    Mirrors ``rbs_physics.urdf_loader.UrdfLoader``.
    """

    def __init__(self, urdf_path: str):
        from urdf_parser_py.urdf import URDF

        self._robot = URDF.from_xml_file(urdf_path)
        self._link_map = self._robot.link_map
        self._joint_map = self._robot.joint_map
        self.urdf_folder = pl.Path(urdf_path).parent.resolve()
        self.package_path: str | None = None

        self._parent_link_map: dict[str, tuple[str, str]] = {}
        self._child_link_map: dict[str, list[str]] = {}
        self._link_info_map: dict[str, LinkInfo] = {}
        self._mesh_link_info_map: dict[str, MeshLinkInfo] = {}
        self._revolute_joint_info_map: dict[str, RevoluteJointInfo] = {}
        self._fixed_joint_info_map: dict[str, FixedJointInfo] = {}
        self._prismatic_joint_info_map: dict[str, PrismaticJointInfo] = {}
        self._root_mesh_link_name: str | None = None
        self._mesh_link_2_parent: dict[str, MeshLinkInfo | None] = {}

        self._process()

    # ------------------------------------------------------------------
    # Internal processing
    # ------------------------------------------------------------------

    def _compute_link_transform(self, link, current: np.ndarray | None = None) -> np.ndarray:
        if current is None:
            current = np.eye(4)
        parent_info = self._parent_link_map.get(link.name)
        if parent_info is None:
            return current
        parent_link_name, joint_name = parent_info
        joint = self._joint_map.get(joint_name)
        if joint is None or joint.origin is None:
            parent_link = self._link_map.get(parent_link_name)
            if parent_link is not None:
                return self._compute_link_transform(parent_link, current)
            return current
        rpy = joint.origin.rpy if joint.origin.rpy else [0, 0, 0]
        xyz = joint.origin.xyz if joint.origin.xyz else [0, 0, 0]
        trans = _xyz_rpy_to_transform(xyz, rpy)
        parent_link = self._link_map.get(parent_link_name)
        assert parent_link is not None
        return self._compute_link_transform(parent_link, trans) @ current

    def _process(self):
        self._collect_info()
        self._set_link_basic_info()
        self._set_joint_info()
        self._process_mesh_link_info()
        self._root_mesh_link_name = self._find_root_mesh_link()

    def _collect_info(self):
        for joint_name, joint in self._joint_map.items():
            self._parent_link_map[joint.child] = (joint.parent, joint_name)
            self._child_link_map.setdefault(joint.parent, []).append(joint.child)

            if joint.type == "revolute" or joint.type == "continuous":
                info = RevoluteJointInfo(
                    name=joint_name,
                    parent_link_name=joint.parent,
                    child_link_name=joint.child,
                )
                if joint.limit is not None:
                    info.lower_limit = joint.limit.lower
                    info.upper_limit = joint.limit.upper
                self._revolute_joint_info_map[joint_name] = info
            elif joint.type == "fixed":
                self._fixed_joint_info_map[joint_name] = FixedJointInfo(
                    name=joint_name,
                    parent_link_name=joint.parent,
                    child_link_name=joint.child,
                )
            elif joint.type == "prismatic":
                info = PrismaticJointInfo(
                    name=joint_name,
                    parent_link_name=joint.parent,
                    child_link_name=joint.child,
                )
                if joint.limit is not None:
                    info.lower_limit = joint.limit.lower
                    info.upper_limit = joint.limit.upper
                self._prismatic_joint_info_map[joint_name] = info

    def _set_link_basic_info(self):
        for link_name, link in self._link_map.items():
            has_mesh = bool(link.collisions)
            li = LinkInfo(name=link_name, has_mesh=has_mesh)
            li.transform = self._compute_link_transform(link)
            self._link_info_map[link_name] = li
            if has_mesh:
                mli = MeshLinkInfo(name=link_name)
                mli.transform = li.transform.copy()
                self._mesh_link_info_map[link_name] = mli

    def _try_get_mesh_parent(self, link_name: str) -> str | None:
        current = link_name
        while current is not None:
            if current in self._mesh_link_info_map:
                return current
            info = self._parent_link_map.get(current)
            if info is None:
                return None
            current = info[0]
        return None

    def _try_get_mesh_child(self, link_name: str) -> str | None:
        current = link_name
        while current is not None:
            if current in self._mesh_link_info_map:
                return current
            children = self._child_link_map.get(current)
            if not children:
                return None
            current = children[0]
        return None

    def _set_joint_info(self):
        for jname, jinfo in self._revolute_joint_info_map.items():
            child_li = self._link_info_map.get(jinfo.child_link_name)
            if child_li is None:
                continue

            jinfo.parent_mesh_link_name = self._try_get_mesh_parent(jinfo.parent_link_name)
            jinfo.child_mesh_link_name = self._try_get_mesh_child(jinfo.child_link_name)

            joint = self._joint_map.get(jname)
            jinfo.local_axis = np.array(joint.axis if joint.axis else [0, 0, 1], dtype=np.float64)
            jinfo.local_axis /= np.linalg.norm(jinfo.local_axis)
            jinfo.local_point_1 = jinfo.local_point_0 + jinfo.local_axis

            t = child_li.transform
            gp1 = t @ np.append(jinfo.local_point_1, 1.0)
            gp0 = t @ np.append(jinfo.local_point_0, 1.0)
            jinfo.global_point_1 = gp1[:3]
            jinfo.global_point_0 = gp0[:3]

        for jname, jinfo in self._fixed_joint_info_map.items():
            jinfo.parent_mesh_link_name = self._try_get_mesh_parent(jinfo.parent_link_name)
            jinfo.child_mesh_link_name = self._try_get_mesh_child(jinfo.child_link_name)

        for jname, jinfo in self._prismatic_joint_info_map.items():
            child_li = self._link_info_map.get(jinfo.child_link_name)
            if child_li is None:
                continue
            jinfo.parent_mesh_link_name = self._try_get_mesh_parent(jinfo.parent_link_name)
            jinfo.child_mesh_link_name = self._try_get_mesh_child(jinfo.child_link_name)

            joint = self._joint_map.get(jname)
            jinfo.local_axis = np.array(joint.axis if joint.axis else [0, 0, 1], dtype=np.float64)
            jinfo.local_axis /= np.linalg.norm(jinfo.local_axis)

            t = child_li.transform
            gp0 = t @ np.array([0, 0, 0, 1.0])
            jinfo.global_point_0 = gp0[:3]
            jinfo.global_axis = (t[:3, :3] @ jinfo.local_axis)
            jinfo.global_axis /= np.linalg.norm(jinfo.global_axis)

    def _process_mesh_link_info(self):
        for link_name, mli in self._mesh_link_info_map.items():
            link = self._link_map.get(link_name)
            if link is None:
                continue

            if link.visuals:
                for visual in link.visuals:
                    if visual.geometry and hasattr(visual.geometry, 'filename') and visual.geometry.filename:
                        mli.visual_mesh_directories.append(visual.geometry.filename)
                        origin = visual.origin
                        t = _xyz_rpy_to_transform(
                            origin.xyz if origin and origin.xyz else [0, 0, 0],
                            origin.rpy if origin and origin.rpy else [0, 0, 0],
                        )
                        mli.visual_mesh_transforms.append(t)

            if link.collisions:
                for collision in link.collisions:
                    if collision.geometry and hasattr(collision.geometry, 'filename') and collision.geometry.filename:
                        mli.collision_mesh_directories.append(collision.geometry.filename)
                        origin = collision.origin
                        t = _xyz_rpy_to_transform(
                            origin.xyz if origin and origin.xyz else [0, 0, 0],
                            origin.rpy if origin and origin.rpy else [0, 0, 0],
                        )
                        mli.collision_mesh_transforms.append(t)

        for jinfo in self._revolute_joint_info_map.values():
            if jinfo.child_mesh_link_name:
                parent_mli = self._mesh_link_info_map.get(jinfo.parent_mesh_link_name)
                self._mesh_link_2_parent[jinfo.child_mesh_link_name] = parent_mli

    def _find_root_mesh_link(self) -> str | None:
        for link_name in self._mesh_link_info_map:
            if link_name not in self._mesh_link_2_parent:
                return link_name
        return None

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def resolve_mesh_path(self, mesh_path: str) -> str:
        """Resolve a mesh path from URDF (handles package://, relative, absolute).

        Mirrors the C++ UrdfSceneImporter fallback: when the resolved path
        does not exist, progressively strip leading directory components and
        search relative to the URDF folder and its ancestors.
        """
        if mesh_path.startswith("package://"):
            rel = mesh_path.replace("package://", "")
            if self.package_path is not None:
                candidate = pl.Path(self.package_path) / rel
                if candidate.exists():
                    return str(candidate.resolve())
            candidate = self.urdf_folder / rel
            if candidate.exists():
                return str(candidate.resolve())
            return self._suffix_search_fallback(rel)

        # Strip protocol prefix (e.g. "file://")
        proto_pos = mesh_path.find("://")
        if proto_pos != -1:
            mesh_path = mesh_path[proto_pos + 3:]

        p = pl.Path(mesh_path)

        # Try relative to URDF folder
        candidate = self.urdf_folder / mesh_path
        if candidate.exists():
            return str(candidate.resolve())

        # Try as absolute path
        if p.is_absolute() and p.exists():
            return str(p.resolve())

        # Fallback: progressive suffix stripping (matches C++ behaviour)
        return self._suffix_search_fallback(mesh_path)

    def _suffix_search_fallback(self, path_str: str) -> str:
        """Strip leading path components and search URDF folder ancestors."""
        suffix = path_str.replace("\\", "/")
        search_base = self.urdf_folder
        for _ in range(6):
            s = suffix
            while s:
                candidate = search_base / s
                if candidate.exists():
                    return str(candidate.resolve())
                slash = s.find("/")
                if slash == -1:
                    break
                s = s[slash + 1:]
            parent = search_base.parent
            if parent == search_base:
                break
            search_base = parent
        return str(self.urdf_folder / path_str)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def mesh_link_infos(self) -> dict[str, MeshLinkInfo]:
        return self._mesh_link_info_map

    @property
    def revolute_joint_infos(self) -> dict[str, RevoluteJointInfo]:
        return self._revolute_joint_info_map

    @property
    def fixed_joint_infos(self) -> dict[str, FixedJointInfo]:
        return self._fixed_joint_info_map

    @property
    def prismatic_joint_infos(self) -> dict[str, PrismaticJointInfo]:
        return self._prismatic_joint_info_map

    @property
    def root_mesh_link_name(self) -> str | None:
        return self._root_mesh_link_name

    @property
    def link_infos(self) -> dict[str, LinkInfo]:
        return self._link_info_map

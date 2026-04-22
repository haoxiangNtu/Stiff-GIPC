"""Convert a URDF robot to a USD stage consumable by StiffGipcUsdParser.

Ported from ``rbs_physics.urdf2usd.Urdf2Usd``.  Builds:
  - ``UsdGeom.Mesh`` prims for each link collision mesh
  - ``UsdPhysics.RigidBodyAPI`` + ``CollisionAPI`` on each link
  - ``UsdPhysics.RevoluteJoint`` / ``FixedJoint`` / ``PrismaticJoint``
  - ``UsdPhysics.ArticulationRootAPI`` on the robot root

After conversion the stage can be fed directly to
:class:`~stiff_physics.usd_scene_parser.StiffGipcUsdParser` so that the
unified USD pipeline (including ``physics:approximation``) is used.
"""

from __future__ import annotations

import pathlib as pl
import re

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from stiff_physics.urdf_loader import UrdfLoader


def _sanitize_prim_name(name: str) -> str:
    """Replace characters invalid in USD prim names with underscores."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def _quat_from_veca_to_vecb(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute quaternion (w, x, y, z) rotating *a* to *b*."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    dot = float(np.dot(a, b))
    if dot > 1.0 - 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -1.0 + 1e-8:
        axis = np.cross(a, [1, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0, 1, 0])
        axis /= np.linalg.norm(axis)
        return np.array([0.0, *axis])
    axis = np.cross(a, b)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(np.clip(dot, -1, 1))
    w = np.cos(angle / 2)
    xyz = axis * np.sin(angle / 2)
    return np.array([w, *xyz])


class Urdf2Usd:
    """Convert a URDF file to a USD stage.

    Usage::

        from pxr import Usd
        stage = Usd.Stage.CreateInMemory()
        converter = Urdf2Usd(stage)
        converter.from_urdf_file("/path/to/robot.urdf")
        # stage is now ready for StiffGipcUsdParser
    """

    def __init__(
        self,
        usd_stage: Usd.Stage,
        root_prim: Usd.Prim | None = None,
        package_path: str | None = None,
        with_visual_mesh: bool = False,
    ):
        self.usd_stage = usd_stage
        if root_prim is None:
            self.root_prim = self.usd_stage.DefinePrim("/Robot", "Xform")
        else:
            self.root_prim = root_prim
        self.urdf_loader: UrdfLoader | None = None
        self.package_path: str | None = package_path
        self.urdf_folder: str = ""
        self.with_visual_mesh = with_visual_mesh

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def from_urdf_file(self, urdf_path: str, mesh_map: dict[str, str] | None = None):
        """Parse a URDF and populate the USD stage.

        Parameters
        ----------
        urdf_path : str
            Path to the URDF file.
        mesh_map : dict, optional
            Mapping from original mesh paths to replacement mesh paths.
        """
        self.urdf_loader = UrdfLoader(urdf_path)
        self.urdf_folder = str(pl.Path(urdf_path).parent.resolve())
        if self.package_path is None:
            self.package_path = self.urdf_folder
        self.urdf_loader.package_path = self.package_path

        mesh_link_infos = self.urdf_loader.mesh_link_infos

        root_path = self.root_prim.GetPath()
        UsdPhysics.ArticulationRootAPI.Apply(self.root_prim.GetPrim())

        # --- Create links ---
        for link_name, info in mesh_link_infos.items():
            link_prim = UsdGeom.Xform.Define(
                self.usd_stage, f"{root_path}/{link_name}")
            UsdPhysics.RigidBodyAPI.Apply(link_prim.GetPrim())
            UsdPhysics.CollisionAPI.Apply(link_prim.GetPrim())
            link_prim.AddTransformOp().Set(Gf.Matrix4d(info.transform.T))

            # Visual meshes (optional)
            if self.with_visual_mesh and info.visual_mesh_directories:
                visuals = UsdGeom.Xform.Define(
                    self.usd_stage, f"{link_prim.GetPath()}/visuals")
                for path, trans in zip(info.visual_mesh_directories,
                                       info.visual_mesh_transforms):
                    resolved = self._resolve_path(path, mesh_map)
                    name = _sanitize_prim_name(pl.Path(resolved).stem)
                    mesh_prim = UsdGeom.Mesh.Define(
                        self.usd_stage, f"{visuals.GetPath()}/{name}")
                    self._write_mesh_to_prim(mesh_prim, trans, resolved)

            # Collision meshes
            collisions = UsdGeom.Xform.Define(
                self.usd_stage, f"{link_prim.GetPath()}/collisions")
            for path, trans in zip(info.collision_mesh_directories,
                                   info.collision_mesh_transforms):
                resolved = self._resolve_path(path, mesh_map)
                name = _sanitize_prim_name(pl.Path(resolved).stem)
                mesh_prim = UsdGeom.Mesh.Define(
                    self.usd_stage, f"{collisions.GetPath()}/{name}")
                UsdPhysics.CollisionAPI.Apply(mesh_prim.GetPrim())
                self._write_mesh_to_prim(mesh_prim, trans, resolved)

        # --- Create joints ---
        joint_scope = UsdGeom.Scope.Define(
            self.usd_stage, f"{root_path}/joints")

        # Revolute joints
        for jname, jinfo in self.urdf_loader.revolute_joint_infos.items():
            self._create_revolute_joint(
                joint_scope, jname, jinfo, mesh_link_infos, root_path)

        # Fixed joints
        for jname, jinfo in self.urdf_loader.fixed_joint_infos.items():
            self._create_fixed_joint(
                joint_scope, jname, jinfo, mesh_link_infos, root_path)

        # Prismatic joints
        for jname, jinfo in self.urdf_loader.prismatic_joint_infos.items():
            self._create_prismatic_joint(
                joint_scope, jname, jinfo, mesh_link_infos, root_path)

        # Add root-fixed joint if root mesh link exists
        root_mesh_link = self.urdf_loader.root_mesh_link_name
        if root_mesh_link:
            self._add_fixed_joint_to_root(root_mesh_link, root_path)

    @staticmethod
    def setup_stage(stage: Usd.Stage, up_axis: str = "Z",
                    meters_per_unit: float = 1.0):
        """Configure stage metadata."""
        stage.SetStartTimeCode(-1)
        stage.SetEndTimeCode(0)
        UsdGeom.SetStageUpAxis(stage, up_axis)
        UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
        world = UsdGeom.Xform.Define(stage, "/World")
        world.AddTransformOp().Set(Gf.Matrix4d().SetIdentity())
        stage.SetDefaultPrim(world.GetPrim())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, path: str,
                      mesh_map: dict[str, str] | None) -> str:
        resolved = self.urdf_loader.resolve_mesh_path(path)
        if mesh_map and resolved in mesh_map:
            return mesh_map[resolved]
        return resolved

    def _write_mesh_to_prim(self, usd_mesh: UsdGeom.Mesh,
                            transform: np.ndarray,
                            mesh_path: str) -> UsdGeom.Mesh:
        path = pl.Path(mesh_path)
        if not path.exists():
            print(f"[Urdf2Usd] WARNING: mesh file not found: {mesh_path}")
            return usd_mesh

        mesh = trimesh.load(str(path), force="mesh")
        usd_mesh.ClearXformOpOrder()
        usd_mesh.AddTransformOp().Set(Gf.Matrix4d(transform.T))
        usd_mesh.CreatePointsAttr(
            [Gf.Vec3f(*v) for v in mesh.vertices])
        usd_mesh.CreateFaceVertexCountsAttr(
            [len(f) for f in mesh.faces])
        usd_mesh.CreateFaceVertexIndicesAttr(
            [int(idx) for face in mesh.faces for idx in face])
        return usd_mesh

    def _create_revolute_joint(self, scope, jname, jinfo,
                               mesh_link_infos, root_path):
        if jinfo.global_point_0 is None or jinfo.global_point_1 is None:
            return

        joint_prim = UsdPhysics.RevoluteJoint.Define(
            self.usd_stage, f"{scope.GetPath()}/{jname}")

        axis = jinfo.global_point_1 - jinfo.global_point_0
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            return
        axis = axis / norm

        body0_name = jinfo.parent_mesh_link_name
        body1_name = jinfo.child_mesh_link_name
        if body0_name is None or body1_name is None:
            return

        body0_info = mesh_link_infos.get(body0_name)
        body1_info = mesh_link_infos.get(body1_name)
        if body0_info is None or body1_info is None:
            return

        trans0 = body0_info.transform
        trans1 = body1_info.transform
        inv0 = np.linalg.inv(trans0)
        inv1 = np.linalg.inv(trans1)

        point = jinfo.global_point_0
        local_pos0 = (inv0 @ np.append(point, 1.0))[:3]
        local_pos1 = (inv1 @ np.append(point, 1.0))[:3]

        joint_prim.CreateAxisAttr().Set("Z")
        z_axis = np.array([0, 0, 1], dtype=np.float64)
        axis_in_z0 = inv0[:3, :3] @ axis
        axis_in_z1 = inv1[:3, :3] @ axis
        local_rot0 = _quat_from_veca_to_vecb(z_axis, axis_in_z0)
        local_rot1 = _quat_from_veca_to_vecb(z_axis, axis_in_z1)

        joint_prim.CreateLocalPos0Attr().Set(Gf.Vec3f(*local_pos0))
        joint_prim.CreateLocalPos1Attr().Set(Gf.Vec3f(*local_pos1))
        joint_prim.CreateLocalRot0Attr().Set(Gf.Quatf(*local_rot0))
        joint_prim.CreateLocalRot1Attr().Set(Gf.Quatf(*local_rot1))
        joint_prim.CreateBody0Rel().SetTargets(
            [Sdf.Path(f"{root_path}/{body0_name}")])
        joint_prim.CreateBody1Rel().SetTargets(
            [Sdf.Path(f"{root_path}/{body1_name}")])

        if jinfo.lower_limit != 0.0 or jinfo.upper_limit != 0.0:
            joint_prim.CreateLowerLimitAttr().Set(np.rad2deg(jinfo.lower_limit))
            joint_prim.CreateUpperLimitAttr().Set(np.rad2deg(jinfo.upper_limit))

    def _create_fixed_joint(self, scope, jname, jinfo,
                            mesh_link_infos, root_path):
        body0_name = jinfo.parent_mesh_link_name
        body1_name = jinfo.child_mesh_link_name
        if body0_name is None or body1_name is None:
            return

        joint_prim = UsdPhysics.FixedJoint.Define(
            self.usd_stage, f"{scope.GetPath()}/{jname}")
        joint_prim.CreateBody0Rel().SetTargets(
            [Sdf.Path(f"{root_path}/{body0_name}")])
        joint_prim.CreateBody1Rel().SetTargets(
            [Sdf.Path(f"{root_path}/{body1_name}")])

    def _create_prismatic_joint(self, scope, jname, jinfo,
                                mesh_link_infos, root_path):
        if jinfo.global_axis is None:
            return

        body0_name = jinfo.parent_mesh_link_name
        body1_name = jinfo.child_mesh_link_name
        if body0_name is None or body1_name is None:
            return

        joint_prim = UsdPhysics.PrismaticJoint.Define(
            self.usd_stage, f"{scope.GetPath()}/{jname}")
        joint_prim.CreateAxisAttr().Set("Z")

        body0_info = mesh_link_infos.get(body0_name)
        body1_info = mesh_link_infos.get(body1_name)
        if body0_info is None or body1_info is None:
            return

        inv0 = np.linalg.inv(body0_info.transform)
        inv1 = np.linalg.inv(body1_info.transform)

        z_axis = np.array([0, 0, 1], dtype=np.float64)
        axis = jinfo.global_axis
        axis_in_z0 = inv0[:3, :3] @ axis
        axis_in_z1 = inv1[:3, :3] @ axis
        local_rot0 = _quat_from_veca_to_vecb(z_axis, axis_in_z0)
        local_rot1 = _quat_from_veca_to_vecb(z_axis, axis_in_z1)

        point = jinfo.global_point_0
        local_pos0 = (inv0 @ np.append(point, 1.0))[:3]
        local_pos1 = (inv1 @ np.append(point, 1.0))[:3]

        joint_prim.CreateLocalPos0Attr().Set(Gf.Vec3f(*local_pos0))
        joint_prim.CreateLocalPos1Attr().Set(Gf.Vec3f(*local_pos1))
        joint_prim.CreateLocalRot0Attr().Set(Gf.Quatf(*local_rot0))
        joint_prim.CreateLocalRot1Attr().Set(Gf.Quatf(*local_rot1))
        joint_prim.CreateBody0Rel().SetTargets(
            [Sdf.Path(f"{root_path}/{body0_name}")])
        joint_prim.CreateBody1Rel().SetTargets(
            [Sdf.Path(f"{root_path}/{body1_name}")])

        if jinfo.lower_limit != 0.0 or jinfo.upper_limit != 0.0:
            joint_prim.CreateLowerLimitAttr().Set(jinfo.lower_limit)
            joint_prim.CreateUpperLimitAttr().Set(jinfo.upper_limit)

    def _add_fixed_joint_to_root(self, mesh_link_name: str, root_path):
        joint_prim = UsdPhysics.FixedJoint.Define(
            self.usd_stage, f"{root_path}/joints/root_fixed_joint")
        joint_prim.CreateBody1Rel().SetTargets(
            [Sdf.Path(f"{root_path}/{mesh_link_name}")])

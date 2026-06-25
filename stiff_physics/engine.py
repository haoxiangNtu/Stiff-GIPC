"""Thin Python wrapper around the pystiffgipc C++ module."""

from __future__ import annotations

import os
import sys
import sysconfig
import numpy as np
from pathlib import Path
from typing import Optional


_INSTALLED_MODE = False

def _import_native():
    """Import pystiffgipc from the installed _native/ sub-package first,
    falling back to the development build/ directory.

    Override path via env var STIFFGIPC_NATIVE_DIR — useful when working
    in a worktree whose build/ should be used instead of the venv's
    installed _native/ (which may point at a different worktree).
    """
    global _INSTALLED_MODE

    # 0. Env-var override (worktree dev workflow) — always wins
    override = os.environ.get("STIFFGIPC_NATIVE_DIR")
    if override:
        if os.path.isdir(override) and override not in sys.path:
            sys.path.insert(0, override)
        try:
            import pystiffgipc
            return pystiffgipc
        except ImportError as exc:
            raise ImportError(
                f"STIFFGIPC_NATIVE_DIR={override} but pystiffgipc not "
                f"importable from there ({exc})") from exc

    # 1. Try installed location (wheel / pip install -e .)
    try:
        from stiff_physics._native import pystiffgipc
        _INSTALLED_MODE = True
        return pystiffgipc
    except ImportError:
        pass

    # 2. Fallback: dev build directory next to the project root.
    #
    # Keep ABI-specific build dirs ahead of the generic "build/" dir.  This
    # worktree may contain both py3.11 and py3.12 extension modules; loading a
    # stale py3.12 module from build/ against a freshly rebuilt core library can
    # corrupt SimEngineConfig layout (e.g. cuda_device is read from the wrong
    # offset).  Newton/IsaacLab py3.12 should therefore resolve build_312 first.
    _project_root = Path(__file__).resolve().parent.parent
    py_tag = f"{sys.version_info.major}{sys.version_info.minor}"
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ""
    build_dirs = [
        _project_root / f"build_{py_tag}",
        _project_root / "build",
    ]
    for _build_dir in build_dirs:
        if not _build_dir.is_dir():
            continue
        if ext_suffix and not any(_build_dir.glob(f"pystiffgipc*{ext_suffix}")):
            continue
        bd = str(_build_dir)
        if bd not in sys.path:
            sys.path.insert(0, bd)
        try:
            import pystiffgipc
            return pystiffgipc
        except ImportError:
            try:
                sys.path.remove(bd)
            except ValueError:
                pass

    raise ImportError(
        "pystiffgipc C++ module not found. "
        "Install the stiff-physics wheel, or build with: "
        "cmake -DBUILD_PYTHON_BINDINGS=ON .. && make pystiffgipc"
    )


_C = _import_native()

_PACKAGE_DATA_DIR = Path(__file__).resolve().parent / "data"


class BodyView:
    """A read-only view of one body's slice of the global vertex/face arrays.

    Lightweight — holds no data of its own; every accessor re-slices the
    engine's current state. Cheap because slicing is a numpy view, but the
    `get_vertices()` slice DOES reflect the latest engine.step() (you don't
    need to fetch a fresh BodyView each frame).

    Use `Engine.get_bodies()` / `get_abd_body(id)` / `get_fem_body(id)` to
    obtain BodyViews — don't construct directly.
    """

    __slots__ = ("_engine", "_record")

    def __init__(self, engine: "Engine", record):
        self._engine = engine
        self._record = record

    @property
    def kind(self) -> str:
        """'ABD' (rigid affine body) or 'FEM' (deformable)."""
        return "ABD" if self._record.body_type == 0 else "FEM"

    @property
    def body_id(self) -> int:
        """Index within this body's kind (e.g. 0..N-1 for ABD bodies)."""
        return self._record.body_offset

    @property
    def label(self) -> str:
        """URDF link name (e.g. 'link_base'), .obj filename, or other source label."""
        return self._record.label

    @property
    def asset_id(self) -> int:
        """Shared MeshAsset id (-1 if not from a shared asset)."""
        return self._record.asset_id

    @property
    def instance_id(self) -> int:
        """Instance index for instanced loads (0 if not instanced)."""
        return self._record.instance_id

    @property
    def vertex_offset(self) -> int:
        """Start index in engine.get_vertices()."""
        return self._record.vertex_offset

    @property
    def vertex_count(self) -> int:
        return self._record.vertex_count

    def get_vertices(self) -> np.ndarray:
        """Current deformed vertices for this body, shape (vertex_count, 3)."""
        v = self._engine.get_vertices()
        s = self._record.vertex_offset
        return v[s : s + self._record.vertex_count]

    def get_vertex_velocities(self) -> np.ndarray:
        v = self._engine.get_vertex_velocities()
        s = self._record.vertex_offset
        return v[s : s + self._record.vertex_count]

    def get_surface_faces(self, local_indices: bool = True) -> np.ndarray:
        """Surface triangles belonging to this body.

        Args:
            local_indices: when True (default), vertex refs are 0-based for
                this body so faces directly index into self.get_vertices().
                Set False to keep global indices that match
                engine.get_vertices().
        """
        all_faces = self._engine.get_surface_faces()
        s = self._record.vertex_offset
        e = s + self._record.vertex_count
        if self._record.vertex_count == 0:
            return np.empty((0, all_faces.shape[1]), dtype=all_faces.dtype)
        in_range = ((all_faces >= s) & (all_faces < e)).all(axis=1)
        body_faces = all_faces[in_range]
        if local_indices:
            body_faces = (body_faces - s).astype(all_faces.dtype, copy=False)
        return body_faces

    def __repr__(self) -> str:
        return (f"BodyView(kind={self.kind} body_id={self.body_id} "
                f"label='{self.label}' verts={self.vertex_count})")


class Config:
    """Simulation configuration mirroring gipc::SimEngineConfig."""

    def __init__(
        self,
        dt: float = 0.01,
        density: float = 1e3,
        young_modulus: float = 1e7,
        poisson_rate: float = 0.49,
        friction_rate: float = 0.4,
        newton_tol: float = 1e-2,
        pcg_tol: float = 1e-4,
        relative_dhat: float = 1e-3,
        joint_strength_ratio: float = 100.0,
        revolute_driving_strength_ratio: float = 100.0,
        semi_implicit_enabled: bool = False,
        semi_implicit_beta_tol: float = 1e-3,
        semi_implicit_min_iter: int = 1,
        newton_iter_cap: int = 1000,
        skip_all_collision: bool = False,
        preconditioner_type: int = 1,
        cuda_device: int = 0,
        assets_dir: str = "",
        prismatic_strength_ratio: float = 100.0,
        prismatic_driving_strength_ratio: float = 100.0,
        max_revolute_step_per_frame: float = 0.1,
        max_prismatic_step_per_frame: float = 0.002,
        gravity: tuple[float, float, float] = (0.0, -9.8, 0.0),
        ground_normal: tuple[float, float, float] = (0.0, 1.0, 0.0),
        ground_offset: float = -1.0,
        velocity_damping: float = 0.0,
        **kwargs,
    ):
        self._cfg = _C.Config()
        self._cfg.dt = dt
        self._cfg.density = density
        self._cfg.young_modulus = young_modulus
        self._cfg.poisson_rate = poisson_rate
        self._cfg.friction_rate = friction_rate
        self._cfg.gd_friction_rate = friction_rate
        self._cfg.newton_tol = newton_tol
        self._cfg.pcg_tol = pcg_tol
        self._cfg.relative_dhat = relative_dhat
        self._cfg.joint_strength_ratio = joint_strength_ratio
        self._cfg.revolute_driving_strength_ratio = revolute_driving_strength_ratio
        self._cfg.prismatic_strength_ratio = prismatic_strength_ratio
        self._cfg.prismatic_driving_strength_ratio = prismatic_driving_strength_ratio
        if hasattr(self._cfg, "max_revolute_step_per_frame"):
            self._cfg.max_revolute_step_per_frame = max_revolute_step_per_frame
        if hasattr(self._cfg, "max_prismatic_step_per_frame"):
            self._cfg.max_prismatic_step_per_frame = max_prismatic_step_per_frame
        self._cfg.semi_implicit_enabled = semi_implicit_enabled
        self._cfg.semi_implicit_beta_tol = semi_implicit_beta_tol
        self._cfg.semi_implicit_min_iter = semi_implicit_min_iter
        self._cfg.newton_iter_cap = newton_iter_cap
        self._cfg.skip_all_collision = skip_all_collision
        self._cfg.preconditioner_type = preconditioner_type
        self._cfg.cuda_device = cuda_device
        self._cfg.collision_detection_buff_scale = 6.0
        self._cfg.velocity_damping = velocity_damping
        if not assets_dir and _INSTALLED_MODE and _PACKAGE_DATA_DIR.is_dir():
            self._cfg.assets_dir = str(_PACKAGE_DATA_DIR) + "/"
        else:
            self._cfg.assets_dir = assets_dir
        import numpy as np
        self._cfg.gravity = np.array(gravity, dtype=np.float64)
        self._cfg.ground_normal = np.array(ground_normal, dtype=np.float64)
        self._cfg.ground_offset = ground_offset

        for k, v in kwargs.items():
            if hasattr(self._cfg, k):
                setattr(self._cfg, k, v)

    @property
    def native(self) -> _C.Config:
        return self._cfg

    def __repr__(self) -> str:
        return f"Config(dt={self._cfg.dt}, density={self._cfg.density})"


class Engine:
    """High-level Python interface to the StiffGIPC simulation engine.

    Lifecycle::

        engine = Engine(config)
        engine.load_urdf("path/to/robot.urdf", scale=0.3)
        engine.finalize()
        for _ in range(1000):
            engine.step()
            verts = engine.get_vertices()
    """

    def __init__(self, config: Optional[Config] = None):
        self._engine = _C.SimEngine()
        self._config = config or Config()
        self._engine.set_config(self._config.native)
        self._engine.init_cuda()
        self._finalized = False

    @property
    def native(self) -> _C.SimEngine:
        return self._engine

    def load_urdf(
        self,
        urdf_path: str,
        scale: float = 1.0,
        translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        root_fixed: bool = True,
        revolute_as_motor: bool = False,
        default_young: float = 1e7,
        initial_joint_angles: dict[str, float] | None = None,
    ) -> None:
        """Load a URDF robot model into the scene.

        Args:
            initial_joint_angles: Optional dict mapping joint names to angles
                (radians). When provided, the arm is loaded at the FK target
                pose instead of the zero pose, avoiding pass-through collisions.
        """
        transform = np.eye(4)
        transform[:3, :3] *= scale
        transform[0, 3] = translation[0]
        transform[1, 3] = translation[1]
        transform[2, 3] = translation[2]

        resolved = urdf_path
        if not os.path.isabs(resolved):
            assets = self._engine.get_assets_dir()
            candidate = os.path.join(assets, resolved)
            if os.path.exists(candidate):
                resolved = candidate

        self._engine.load_urdf(resolved, transform, root_fixed,
                               revolute_as_motor, default_young,
                               initial_joint_angles or {})

    _BODY_TYPE_MAP = {"ABD": 0, "abd": 0, "FEM": 1, "fem": 1}
    _BOUNDARY_MAP  = {"Free": 0, "free": 0, "Fixed": 1, "fixed": 1,
                      "Motor": 2, "motor": 2, "Animated": 3, "animated": 3}

    def load_mesh(
        self,
        mesh_path: str,
        dimensions: int = 3,
        body_type: str | int = "FEM",
        transform: Optional[np.ndarray] = None,
        young_modulus: float = 1e7,
        boundary_type: str | int = "Free",
    ) -> None:
        """Load a raw mesh (.msh for 3D tet, .obj for 2D cloth/shell).

        Args:
            mesh_path: Path to mesh file, resolved against assets_dir if relative.
            dimensions: 2 for triangle shell/cloth, 3 for tet volume.
            body_type: "ABD" (0) for rigid or "FEM" (1) for deformable.
            transform: 4x4 transformation matrix (identity if None).
            young_modulus: Young's modulus for this body.
            boundary_type: "Free" (0) or "Fixed" (1).
        """
        if transform is None:
            transform = np.eye(4)

        bt = self._BODY_TYPE_MAP.get(body_type, body_type) if isinstance(body_type, str) else body_type
        bb = self._BOUNDARY_MAP.get(boundary_type, boundary_type) if isinstance(boundary_type, str) else boundary_type

        resolved = mesh_path
        if not os.path.isabs(resolved):
            assets = self._engine.get_assets_dir()
            candidate = os.path.join(assets, resolved)
            if os.path.exists(candidate):
                resolved = candidate

        self._engine.load_mesh(resolved, dimensions, bt, transform, young_modulus, bb)

    def load_mesh_from_data(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        verts_per_face: int = 3,
        dimensions: int = 3,
        body_type: str | int = "FEM",
        transform: Optional[np.ndarray] = None,
        young_modulus: float = 1e7,
        boundary_type: str | int = "Free",
    ) -> None:
        """Load mesh from in-memory vertex/face arrays (no file I/O)."""
        if transform is None:
            transform = np.eye(4)
        bt = self._BODY_TYPE_MAP.get(body_type, body_type) if isinstance(body_type, str) else body_type
        bb = self._BOUNDARY_MAP.get(boundary_type, boundary_type) if isinstance(boundary_type, str) else boundary_type
        verts = np.ascontiguousarray(vertices, dtype=np.float64)
        fcs = np.ascontiguousarray(faces, dtype=np.int32)
        self._engine.load_mesh_from_data(verts, fcs, verts_per_face,
                                         dimensions, bt, transform,
                                         young_modulus, bb)

    def load_mesh_instanced(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        transforms_list: list[np.ndarray],
        verts_per_face: int = 3,
        dimensions: int = 3,
        body_type: str | int = "ABD",
        young_modulus: float = 1e7,
        boundary_type: str | int = "Free",
    ) -> dict:
        """Load one mesh as N instances with different transforms.

        Args:
            vertices: (V, 3) float64 rest-pose vertex positions.
            faces: (F, vpf) int32 face index array.
            transforms_list: List of N (4, 4) float64 transform matrices.
            verts_per_face: vertices per face (3 for tri, 4 for tet).
            dimensions: 2 for shell, 3 for volume.
            body_type: "ABD" / "FEM" (or 0 / 1).
            young_modulus: Young's modulus.
            boundary_type: "Free" / "Fixed" (or 0 / 1).

        Returns:
            dict with keys: body_offsets, vertex_offsets, vertex_counts, asset_id
        """
        bt = self._BODY_TYPE_MAP.get(body_type, body_type) if isinstance(body_type, str) else body_type
        bb = self._BOUNDARY_MAP.get(boundary_type, boundary_type) if isinstance(boundary_type, str) else boundary_type
        verts = np.ascontiguousarray(vertices, dtype=np.float64)
        fcs = np.ascontiguousarray(faces, dtype=np.int32)
        tf_list = [np.ascontiguousarray(t, dtype=np.float64) for t in transforms_list]

        result = self._engine.load_mesh_instanced(
            verts, fcs, verts_per_face, dimensions, bt,
            tf_list, young_modulus, bb
        )
        return {
            "body_offsets":   list(result.body_offsets),
            "vertex_offsets": list(result.vertex_offsets),
            "vertex_counts":  list(result.vertex_counts),
            "asset_id":       result.asset_id,
        }

    def get_mesh_asset(self, asset_id: int):
        """Return the MeshAsset object for a given asset_id."""
        return self._engine.get_mesh_asset(asset_id)

    @property
    def mesh_asset_count(self) -> int:
        return self._engine.get_mesh_asset_count()

    def add_collision_exclusion(self, body_a: int, body_b: int) -> None:
        self._engine.add_collision_exclusion(body_a, body_b)

    def set_body_groups(self, groups) -> None:
        """Set per-collision-body group (environment) id, length collision_body_num
        (ABD bodies first [0,n_abd), then FEM [n_abd, n_abd+n_fem)).

        Bodies in different groups (both >= 0) never collide — folded into the
        collision-skip matrix at finalize(), so spatially-tiled environments are
        guaranteed isolated regardless of spacing. group < 0 = wildcard (collides
        with everything). Call before finalize().
        """
        self._engine.set_body_groups([int(g) for g in groups])

    def add_ground_collision_skip(self, body_id: int) -> None:
        self._engine.add_ground_collision_skip(body_id)

    def add_hybrid_fem_body(self, hybrid_data, transform=None,
                            target_abd_body_offset=None):
        """Load a hybrid ABD-FEM tet mesh produced by tools/build_hybrid_mesh.py.

        Parameters
        ----------
        hybrid_data : str | dict | numpy.lib.npyio.NpzFile
            Either a path to a .npz file produced by build_hybrid_mesh.py,
            or an already-loaded mapping with the same field names
            (vertices, tets, vertex_abd_body_id, vertex_local_pos, density,
            young_modulus, poisson_ratio, ...).
        transform : np.ndarray of shape (4,4), optional
            World-frame transform applied to the FEM mesh on load.  Pass
            this when the .npz vertices are in a per-body local frame and
            you need to place the hybrid mesh somewhere specific in the
            world.  Default: identity (.npz vertices used as-is).
        target_abd_body_offset : int, optional
            If given, ALL pin body_ids in the .npz are shifted by this
            offset before being passed to the engine.  Useful when the
            .npz was authored against a different ABD body indexing.
            Default: 0 (use .npz body_ids verbatim).

        Returns
        -------
        fem_body_offset : int
            The vertex_offset of the loaded FEM body (use this to convert
            local FEM vertex indices to global if you need them later).
        """
        if isinstance(hybrid_data, (str, os.PathLike)):
            data = np.load(str(hybrid_data))
        else:
            data = hybrid_data

        verts = np.ascontiguousarray(data["vertices"], dtype=np.float64)
        tets = np.ascontiguousarray(data["tets"], dtype=np.int32)
        v_body = np.ascontiguousarray(data["vertex_abd_body_id"], dtype=np.int32)
        v_lo = np.ascontiguousarray(data["vertex_local_pos"], dtype=np.float64)
        young = float(data["young_modulus"])
        # density/poisson currently unused by the FEM loader (engine derives
        # mass from volume & a fixed density); kept here for completeness.

        T = np.eye(4) if transform is None else np.asarray(transform, dtype=np.float64)

        # Add FEM body via load_mesh_from_data.  Tet mesh: verts (N,3),
        # tets (M,4), pass as-is (engine binding expects faces shape[0]==M
        # and a separate verts_per_face int).  body_type='FEM', initial
        # boundary_type='Free' — pinned verts are upgraded to 'Fixed' in
        # finalize() via the add_fem_pins_with_local_pos → existing M1 path.
        n_v = verts.shape[0]
        n_t = tets.shape[0]
        self.load_mesh_from_data(
            verts, tets, verts_per_face=4, dimensions=3,
            body_type="FEM", transform=T, young_modulus=young,
            boundary_type="Free",
        )
        rec = self.get_load_records()[-1]
        body_offset_v = rec.vertex_offset

        # Build pin arrays — only rigid verts (vertex_abd_body_id >= 0).
        rigid_mask = v_body >= 0
        if not np.any(rigid_mask):
            print(f"[add_hybrid_fem_body] no rigid verts in npz — pure FEM body")
            return body_offset_v
        rigid_local_idx = np.nonzero(rigid_mask)[0].astype(np.int32)
        fem_global_ids = (rigid_local_idx + body_offset_v).astype(np.int32)
        body_ids_arr = v_body[rigid_mask].astype(np.int32)
        if target_abd_body_offset is not None:
            body_ids_arr = body_ids_arr + int(target_abd_body_offset)
        local_pos_arr = np.ascontiguousarray(v_lo[rigid_mask], dtype=np.float64)

        self._engine.add_fem_pins_with_local_pos(
            fem_global_ids, body_ids_arr, local_pos_arr)

        n_rigid = int(rigid_mask.sum())
        n_iface_t = int((data["tet_region"] == 1).sum())
        n_rigid_t = int((data["tet_region"] == 2).sum())
        print(f"[add_hybrid_fem_body] {n_v}v/{n_t}t loaded; "
              f"{n_rigid} rigid pins to bodies {sorted(set(body_ids_arr.tolist()))}; "
              f"{n_iface_t} interface tets, {n_rigid_t} rigid-internal tets")
        return body_offset_v

    def add_stitch_spring(self, fem_vertex_id: int, abd_anchor_vertex_id: int,
                          abd_body_id: int,
                          rest_offset_world=(0.0, 0.0, 0.0)) -> None:
        """Stitch a FEM vertex to an ABD body via a soft-spring constraint.

        At each step the FEM vertex is pulled toward
            target_world = abd_anchor_vertex_world_pos + rest_offset_world

        For best behavior under ABD body rotation, place the FEM vertex
        coincident with the ABD anchor vertex at finalize time and pass
        rest_offset_world = (0, 0, 0). Then the spring naturally tracks
        both translation and rotation of the ABD body.

        Spring stiffness is governed by Config.soft_motion_rate.

        Must be called BEFORE finalize().
        """
        self._engine.add_stitch_spring(
            int(fem_vertex_id), int(abd_anchor_vertex_id), int(abd_body_id),
            tuple(float(x) for x in rest_offset_world))

    # ---- libuipc-style per-face orient labels (pre-finalize) ----

    def set_abd_body_face_orient(self, body_id: int, orient) -> bool:
        """Override per-face orientation for an ABD surface body.

        `orient` is one int per triangle in {-1, 0, +1}: -1 flips that
        face's normal at integration time (mass / centroid / inertia),
        0 or +1 leaves it untouched. Topology (face vertex order) is
        NOT modified — collision / BVH / render code see the original.

        Must be called BEFORE finalize() (after that, surface mesh data
        has already been copied to ABDSystem and changes are ignored).
        Returns True on success, False if body_id has no surface body.
        """
        arr = np.ascontiguousarray(np.asarray(orient, dtype=np.int32))
        return self._engine.set_abd_body_face_orient(body_id, arr.tolist())

    def get_abd_body_face_orient(self, body_id: int) -> np.ndarray:
        """Read current per-face orient labels (empty if none set)."""
        return np.asarray(self._engine.get_abd_body_face_orient(body_id),
                          dtype=np.int32)

    def get_abd_surface_body_vertices(self, body_id: int) -> np.ndarray:
        """Pre-finalize: per-body local surface vertices. Shape (N, 3) float64."""
        return self._engine.get_abd_surface_body_vertices(body_id)

    def get_abd_surface_body_triangles(self, body_id: int) -> np.ndarray:
        """Pre-finalize: per-body local triangle indices. Shape (M, 3) int32."""
        return self._engine.get_abd_surface_body_triangles(body_id)

    def label_face_orient_for_abd_body(self, body_id: int, method: str = "flood_fill") -> int:
        """Compute per-face orient labels for an ABD body and write them
        into the engine. Must be called BEFORE finalize().

        method: only 'flood_fill' supported currently. Builds edge
            adjacency, BFS-propagates winding consistency from face 0
            (handles multi-component meshes by restarting from each
            unvisited face), then checks signed volume sign — if
            negative, flips the orient label of every face so that
            outward normal convention is restored.

        Returns: number of faces marked as inverted (orient = -1).
        Returns 0 if no fix was needed (mesh already correctly wound)
        or if body_id has no surface body.
        """
        from stiff_physics.mesh_utils import compute_face_orient_flood_fill
        verts = self.get_abd_surface_body_vertices(body_id)
        faces = self.get_abd_surface_body_triangles(body_id)
        if verts.size == 0 or faces.size == 0:
            return 0
        orient = compute_face_orient_flood_fill(verts, faces)
        self.set_abd_body_face_orient(body_id, orient)
        return int((orient == -1).sum())

    def add_fixed_joint(self, parent_body: int, child_body: int,
                        world_anchor, world_normal, world_bitangent) -> int:
        """Create a fixed joint between two ABD bodies. Must call before finalize()."""
        import numpy as np
        a = np.asarray(world_anchor, dtype=np.float64).ravel()
        n = np.asarray(world_normal, dtype=np.float64).ravel()
        b = np.asarray(world_bitangent, dtype=np.float64).ravel()
        return self._engine.add_fixed_joint(parent_body, child_body, a, n, b)

    def add_revolute_joint(self, parent_body: int, child_body: int,
                           world_axis, joint_pos,
                           lower_limit: float, upper_limit: float,
                           initial_angle: float = 0.0,
                           name: str = "") -> int:
        """Create a revolute joint between two ABD bodies. Must call before finalize()."""
        import numpy as np
        ax = np.asarray(world_axis, dtype=np.float64).ravel()
        p = np.asarray(joint_pos, dtype=np.float64).ravel()
        return self._engine.add_revolute_joint(parent_body, child_body,
                                               ax, p, lower_limit, upper_limit,
                                               initial_angle, name)

    def add_prismatic_joint(self, parent_body: int, child_body: int,
                            world_center, world_axis,
                            lower_limit: float, upper_limit: float,
                            name: str = "") -> int:
        """Create a prismatic joint between two ABD bodies. Must call before finalize()."""
        import numpy as np
        c = np.asarray(world_center, dtype=np.float64).ravel()
        ax = np.asarray(world_axis, dtype=np.float64).ravel()
        return self._engine.add_prismatic_joint(parent_body, child_body,
                                                c, ax, lower_limit, upper_limit,
                                                name)

    def set_vertex_boundary(self, vertex_index: int, boundary_type: int) -> None:
        """Set per-vertex boundary type (0=Free, 1=Fixed). Must call before finalize()."""
        self._engine.set_vertex_boundary(vertex_index, boundary_type)

    def set_vertex_boundaries(self, indices, boundary_type: int) -> None:
        """Batch version of set_vertex_boundary."""
        for idx in indices:
            self._engine.set_vertex_boundary(int(idx), boundary_type)

    @property
    def abd_body_count(self) -> int:
        return self._engine.get_abd_body_count()

    @property
    def fem_body_count(self) -> int:
        return self._engine.get_fem_body_count()

    @property
    def vertex_count_host(self) -> int:
        """Vertex count on host (available before finalize)."""
        return self._engine.get_vertex_count_host()

    def get_vertex_position_host(self, idx: int) -> tuple[float, float, float]:
        """Read a single vertex position from host memory (before finalize)."""
        return self._engine.get_vertex_position_host(idx)

    def finalize(self) -> None:
        """Finalize the scene: compute FEM data, upload to GPU, build BVH."""
        self._engine.finalize()
        self._finalized = True

    def step(self) -> None:
        """Advance simulation by one timestep (dt)."""
        self._engine.step()

    def set_log_level(self, level: int) -> None:
        """Control per-frame solver log verbosity.

        0 = silent (suppress solver banner, Newton-iteration, Kappa,
        ``average time cost``, timer breakdown and one-time setup prints);
        >=1 = verbose (default).  Use 0 for co-simulation / piped tooling that
        needs a clean stdout.
        """
        self._engine.set_log_level(int(level))

    def reset(self) -> None:
        """Tear down the entire world and return to a fresh empty state.

        Frees all loaded bodies (FEM/ABD), constraints and GPU buffers while
        keeping the current :class:`Config`.  After ``reset()`` re-run the usual
        ``load_mesh()`` / ``load_urdf()`` / ``add_*`` + :meth:`finalize` sequence
        in the same process — no need to recreate the Engine.
        """
        self._engine.reset()
        self._finalized = False

    # ---- State queries ----

    def get_vertices(self) -> np.ndarray:
        """Return vertex positions as (N, 3) float64 array."""
        return self._engine.get_vertices()

    def get_vertices_device_ptr(self) -> int:
        """[gpu-direct] Raw CUDA device pointer (int) to the (N,3) float64 vertex
        buffer, N = :meth:`get_vertex_count`. Wrap with a zero-copy GPU array
        (e.g. ``warp.array(ptr=..., dtype=wp.vec3d, length=N)``) to read FEM
        vertex positions without a host round-trip. Valid after finalize()."""
        return self._engine.get_vertices_device_ptr()

    def get_vertex_velocities(self) -> np.ndarray:
        """Return vertex velocities as (N, 3) float64 array."""
        return self._engine.get_vertex_velocities()

    def set_vertex_positions_gpu(self, positions: np.ndarray) -> None:
        """Write vertex positions to GPU from (N, 3) float64."""
        self._engine.set_vertex_positions_gpu(
            np.ascontiguousarray(positions, dtype=np.float64))

    def set_vertex_velocities_gpu(self, velocities: np.ndarray) -> None:
        """Write vertex velocities to GPU from (N, 3) float64."""
        self._engine.set_vertex_velocities_gpu(
            np.ascontiguousarray(velocities, dtype=np.float64))

    def teleport_fem_vertices(self, positions: np.ndarray,
                              velocities: Optional[np.ndarray] = None) -> None:
        """Teleport FEM vertices: writes new positions to _vertexes (current),
        o_vertexes (previous-step committed), and xTilta (predictor). Use this
        instead of set_vertex_positions_gpu for handoff/reset, otherwise the
        next engine.step() reverts to the stale rest pose via xTilta.

        If ``velocities`` is provided, also writes velocities and extends
        xTilta = x + v*dt + g*dt^2 so inertia is preserved across handoff.
        Default (None) zeros velocities per teleport_abd_bodies semantics.
        """
        pos = np.ascontiguousarray(positions, dtype=np.float64)
        if velocities is None:
            self._engine.teleport_fem_vertices(pos)
        else:
            self._engine.teleport_fem_vertices(
                pos,
                np.ascontiguousarray(velocities, dtype=np.float64))

    def get_surface_faces(self) -> np.ndarray:
        """Return surface triangle indices as (F, 3) uint32 array."""
        return self._engine.get_surface_faces()

    def get_surface_vertex_indices(self) -> np.ndarray:
        """Return indices of surface vertices as (S,) uint32 array."""
        return self._engine.get_surface_vertex_indices()

    @property
    def vertex_count(self) -> int:
        return self._engine.get_vertex_count()

    @property
    def surface_face_count(self) -> int:
        return self._engine.get_surface_face_count()

    # ---- ABD body state ----

    def get_abd_body_transforms(self, body_offsets: np.ndarray) -> np.ndarray:
        """Return (N, 4, 4) float64 transforms for ABD bodies at given offsets."""
        offsets = np.ascontiguousarray(body_offsets, dtype=np.int32)
        return self._engine.get_abd_body_transforms(offsets)

    def set_abd_body_transforms(self, body_offsets: np.ndarray,
                                transforms: np.ndarray) -> None:
        """Set ABD body transforms from (N, 4, 4) float64."""
        offsets = np.ascontiguousarray(body_offsets, dtype=np.int32)
        tfs = np.ascontiguousarray(transforms, dtype=np.float64)
        self._engine.set_abd_body_transforms(offsets, tfs)

    def teleport_abd_bodies(self, body_offsets: np.ndarray,
                            transforms: np.ndarray) -> None:
        """Teleport ABD bodies: sets q/q_prev/q_tilde/q_temp and zeros velocity.

        Use this instead of set_abd_body_transforms for init/reset to avoid
        phantom velocities from stale q_prev.
        """
        offsets = np.ascontiguousarray(body_offsets, dtype=np.int32)
        tfs = np.ascontiguousarray(transforms, dtype=np.float64)
        self._engine.teleport_abd_bodies(offsets, tfs)

    def get_abd_body_velocities(self, body_offsets: np.ndarray) -> np.ndarray:
        """Return (N, 4, 4) float64 velocity matrices for ABD bodies."""
        offsets = np.ascontiguousarray(body_offsets, dtype=np.int32)
        return self._engine.get_abd_body_velocities(offsets)

    def set_abd_body_velocities(self, body_offsets: np.ndarray,
                                velocities: np.ndarray) -> None:
        """Set ABD body velocities from (N, 4, 4) float64."""
        offsets = np.ascontiguousarray(body_offsets, dtype=np.int32)
        vels = np.ascontiguousarray(velocities, dtype=np.float64)
        self._engine.set_abd_body_velocities(offsets, vels)

    # ---- FEM body state ----

    def get_fem_body_vertex_range(self, fem_body_idx: int) -> tuple[int, int]:
        """Return (vertex_start, vertex_count) for a FEM body."""
        return self._engine.get_fem_body_vertex_range(fem_body_idx)

    # ---- Load record tracking ----

    def get_load_records(self) -> list:
        """Return all BodyLoadRecord objects."""
        return self._engine.get_all_load_records()

    # ---- Per-body views ----

    def get_bodies(self) -> list[BodyView]:
        """All bodies (ABD + FEM mixed) in load-record order."""
        return [BodyView(self, r) for r in self._engine.get_all_load_records()]

    def get_abd_body(self, body_id: int) -> BodyView:
        """Get the ABD body with the given index (0..abd_body_count-1)."""
        for r in self._engine.get_all_load_records():
            if r.body_type == 0 and r.body_offset == body_id:
                return BodyView(self, r)
        raise IndexError(f"ABD body {body_id} not found")

    def get_fem_body(self, body_id: int) -> BodyView:
        """Get the FEM body with the given index (0..fem_body_count-1)."""
        for r in self._engine.get_all_load_records():
            if r.body_type == 1 and r.body_offset == body_id:
                return BodyView(self, r)
        raise IndexError(f"FEM body {body_id} not found")

    def get_vertex_body_ids(self) -> np.ndarray:
        """Per-vertex body identification, shape (N, 2) int32.

        Column 0: body_type (0=ABD, 1=FEM)
        Column 1: body_offset (index within its kind)

        Useful for mask-based filtering across the global vertex array::

            ids = engine.get_vertex_body_ids()
            verts = engine.get_vertices()
            fem_verts = verts[ids[:, 0] == 1]      # all FEM vertices
            link3_verts = verts[(ids[:, 0] == 0) & (ids[:, 1] == 3)]
        """
        n = len(self.get_vertices())
        ids = np.zeros((n, 2), dtype=np.int32)
        for r in self._engine.get_all_load_records():
            s = r.vertex_offset
            e = s + r.vertex_count
            ids[s:e, 0] = r.body_type
            ids[s:e, 1] = r.body_offset
        return ids

    def get_body_contact_force(self, vertex_offset: int, vertex_count: int) -> np.ndarray:
        """Net IPC contact force (world frame, N) on a body.

        Sums the per-vertex contact forces over the contiguous global vertex
        range ``[vertex_offset, vertex_offset + vertex_count)``.  Returns a
        ``(3,)`` float64 array.  Must be called AFTER :meth:`step` so the
        contact solver state is current.

        The vertex range for a given body can be looked up from
        :meth:`get_all_load_records` (``vertex_offset`` / ``vertex_count`` per
        record) or, for FEM bodies, from the USD-parser ``geometry_dict``.
        """
        return self._engine.get_body_contact_force(int(vertex_offset), int(vertex_count))

    def get_pair_contact_force(self, a_off: int, a_cnt: int, b_off: int, b_cnt: int) -> np.ndarray:
        """Net IPC contact force (world frame, raw IP scaling) on body A FROM body B.

        Barrier gradient summed over A's vertex range ``[a_off, a_off+a_cnt)``,
        restricted to collision pairs that connect A and B's vertex range
        ``[b_off, b_off+b_cnt)``.  Apply the same ``-1/dt**2`` + sign convention
        as :meth:`get_body_contact_force` to get Newtons.  Call AFTER :meth:`step`.
        """
        return self._engine.get_pair_contact_force(int(a_off), int(a_cnt), int(b_off), int(b_cnt))

    def get_collision_pairs_clean(self) -> np.ndarray:
        """Current body-body collision pairs as an ``(N, 4)`` int array of plain
        vertex indices (``-1`` padded for PP/PE), UIPC-style — the "clean export
        layer".  The solver keeps its MMCVID packing internally; this is a
        read-only decoded view for sensors / inspection.  Call AFTER :meth:`step`.
        """
        return self._engine.get_collision_pairs_clean()

    def get_contacts_device(self):
        """[Step B] GPU-resident per-contact export. Returns
        ``(count, pair_ptr, force_ptr)`` — device pointers (uintptr) to an int2
        ``(bodyA, bodyB)`` (bodyB=-1 for ground) and a double3 world contact force
        (N) on bodyA. Wrap with ``warp.array(ptr=..., copy=False)``. Read-only,
        valid until the next call. Call AFTER :meth:`step`.
        """
        return self._engine.get_contacts_device()

    def get_contacts(self):
        """[Step B] Host readback of the per-contact export: ``(pair (N,2) int,
        force (N,3) double world N on bodyA)``. For validation; the GPU-direct
        path uses :meth:`get_contacts_device`. Call AFTER :meth:`step`.
        """
        return self._engine.get_contacts()

    def set_abd_body_density(self, body_id: int, density: float) -> None:
        """Override one ABD body's density (mass = density × volume).

        Lets a scene mix per-body densities (the global ``Config.density`` is
        the default for bodies without an override).  Must be called AFTER the
        body is loaded and BEFORE :meth:`finalize`.
        """
        self._engine.set_abd_body_density(int(body_id), float(density))

    def set_abd_body_inertia(self, body_id: int, mass: float,
                             com, inertia) -> None:
        """Override an ABD body's mass / COM / inertia (load frame).

        ``com`` is a length-3 array (world/load frame), ``inertia`` a 3x3 (about
        the COM). Uses authored values (e.g. URDF inertial tags via Newton
        body_mass/body_com/body_inertia) instead of the welded collision-mesh
        geometry. Must be called AFTER loading the body and BEFORE finalize().
        """
        import numpy as _np
        c = _np.ascontiguousarray(com, dtype=_np.float64).reshape(3)
        I = _np.ascontiguousarray(inertia, dtype=_np.float64).reshape(9)
        self._engine.set_abd_body_inertia(int(body_id), float(mass), c, I)

    # ---- Joint control ----

    @property
    def num_revolute_joints(self) -> int:
        return self._engine.get_num_revolute_joints()

    @property
    def num_prismatic_joints(self) -> int:
        return self._engine.get_num_prismatic_joints()

    def get_revolute_joint_info(self, idx: int):
        return self._engine.get_revolute_joint_info(idx)

    def get_prismatic_joint_info(self, idx: int):
        return self._engine.get_prismatic_joint_info(idx)

    def set_revolute_target(self, idx: int, angle_rad: float) -> None:
        self._engine.set_revolute_target(idx, angle_rad)

    def set_revolute_initial_offset(self, idx: int, offset_rad: float) -> None:
        self._engine.set_revolute_initial_offset(idx, offset_rad)

    def set_prismatic_target(self, idx: int, distance_m: float) -> None:
        self._engine.set_prismatic_target(idx, distance_m)

    def set_revolute_strength(self, idx: int, strength: float) -> None:
        """Set per-joint driving strength multiplier.

        Effective K = Config.revolute_driving_strength_ratio * strength *
        (m_parent + m_child). Default 1.0 (no change vs global ratio).
        Lower = joint yields under contact.
        """
        self._engine.set_revolute_strength(idx, strength)

    def set_prismatic_strength(self, idx: int, strength: float) -> None:
        self._engine.set_prismatic_strength(idx, strength)

    def get_prismatic_drive_force(self, idx: int) -> float:
        return self._engine.get_prismatic_drive_force(idx)

    def get_prismatic_current_distance(self, idx: int) -> float:
        return self._engine.get_prismatic_current_distance(idx)

    def get_body_contact_force_batched(self, offsets, counts) -> np.ndarray:
        """BATCHED net IPC contact force: one contact rebuild + one D2H for all
        segments. `offsets`/`counts` are per-finger vertex ranges; returns an
        (n_seg, 3) array of per-finger grip forces. Call AFTER step()."""
        return self._engine.get_body_contact_force_batched(
            np.ascontiguousarray(offsets, dtype=np.int32),
            np.ascontiguousarray(counts, dtype=np.int32))

    def set_max_revolute_step_per_frame(self, rad: float) -> None:
        """Set the per-step revolute target slew limit in radians."""
        self._engine.set_max_revolute_step_per_frame(float(rad))

    def set_max_prismatic_step_per_frame(self, meters: float) -> None:
        """Set the per-step prismatic target slew limit in meters."""
        self._engine.set_max_prismatic_step_per_frame(float(meters))

    def get_revolute_current_angles(self) -> np.ndarray:
        """Read actual joint angles from GPU state.

        Returns (N,) float64 in radians.  IMPORTANT: this is the angle
        RELATIVE TO THE LOAD-TIME POSE.  If the URDF was loaded via
        `load_urdf(initial_joint_angles=...)`, this returns 0 immediately
        after load (not the offset values).  For ABSOLUTE URDF angles
        use `get_revolute_current_angles_abs()`.
        """
        return self._engine.get_revolute_current_angles()

    def get_revolute_initial_offsets(self) -> np.ndarray:
        """Per-joint URDF angle at URDF load time.

        Returns (N,) float64 in radians.  Comes from the `initial_joint_angles`
        dict passed to `load_urdf`; all zeros if it wasn't passed.
        """
        return self._engine.get_revolute_initial_offsets()

    def get_revolute_current_angles_abs(self) -> np.ndarray:
        """Absolute URDF joint angles = relative-from-load + initial-offset.

        Returns (N,) float64 in radians.  Use this when you need 'what
        angle does the URDF think the joint is at right now' — e.g. for
        RL state observation, saving to disk, comparison with joint limits.
        """
        return (self._engine.get_revolute_current_angles()
                + self._engine.get_revolute_initial_offsets())

    def get_all_joint_infos(self):
        return self._engine.get_all_joint_infos()

"""StiffGIPC backend: put a SensorAsset into a stiff_physics.Engine.

This replaces Taccel's ``warp_ipc`` calls one-for-one:

    Taccel (warp_ipc)                        StiffGIPC (stiff_physics.Engine)
    -----------------------------------------------------------------------------
    add_soft_vol_body(mesh, rho, E, nu, mu)  load_mesh_from_data(..., body_type="FEM")
                                             + set_soft_body_density + set_body_friction
    set_soft_kinematic_constraint(h, mask)    mode="fixed":  set_vertex_boundaries(stick, 1)
                                             mode="pinned": add_hybrid_fem_body(... pins ...)
    set_body_collision_layer / filter         set_body_groups / add_collision_exclusion
    get_element_by_handle(h)                  get_vertices()[off : off+n]
    (no equivalent)                           get_vertex_contact_forces() -> Newtons

Two attachment modes:

* ``fixed``  - the gel's stick vertices are world-fixed (boundary_type=1).  Use
  for the indentation rig where the sensor does not move: the indenter is the
  moving body.  Simplest, no ABD carrier needed.
* ``pinned`` - the stick vertices are pinned to an ABD carrier body (robot link)
  through ``Engine.add_hybrid_fem_body``, so the gel follows the link exactly.
  Use for grippers/hands.
"""

from __future__ import annotations

import numpy as np

from .observation import observe
from .sensor_asset import SensorAsset


def clear_mesh_cache(assets_dir: str) -> int:
    """Delete StiffGIPC's stale METIS cache for in-memory meshes. Returns files removed.

    ``load_mesh_from_data`` writes its METIS-sorted mesh to
    ``<assets_dir>/sorted_mesh/tmp_mesh_<body_index>_sorted.16.{msh,part,idx}``
    and reuses that file on the next run — **keyed by body index, not by mesh
    content**. Any two scenes that load different geometry at the same body index
    therefore silently swap meshes, with no warning beyond a
    ``metis files exist (validated: N verts)`` line. This cost hours of debugging:
    a gel appeared unrotated (engine y span 40.3..80.3 mm instead of 60.3..66.2)
    because an earlier run of a different scene had cached its own gel at the same
    body index.

    Call this at the start of every scene that uses ``load_mesh_from_data``.
    """
    import glob
    import os

    removed = 0
    for f in glob.glob(os.path.join(assets_dir, "sorted_mesh", "tmp_mesh_*")):
        os.remove(f)
        removed += 1
    return removed


def _tf(transform: np.ndarray | None) -> np.ndarray:
    return np.eye(4) if transform is None else np.asarray(transform, dtype=np.float64)


class GelBody:
    """One gel pad living inside a StiffGIPC Engine."""

    def __init__(self, asset: SensorAsset, engine, world_tf: np.ndarray | None = None):
        self.asset = asset
        self.engine = engine
        self.world_tf = _tf(world_tf)
        self.vertex_offset: int | None = None
        self.body_offset: int | None = None  # index into engine.get_load_records()

    # ------------------------------------------------------------- build phase

    def add_fixed(self) -> "GelBody":
        """Add the gel as a FEM body with world-fixed stick vertices.

        Must be called BEFORE engine.finalize().
        """
        e, a = self.engine, self.asset
        e.load_mesh_from_data(
            a.points, a.tets, verts_per_face=4, dimensions=3,
            body_type="FEM", transform=self.world_tf,
            young_modulus=a.material.E, boundary_type="Free",
        )
        self._record_offsets()
        e.set_vertex_boundaries(self.global_ids(np.nonzero(a.stick_mask)[0]), 1)  # 1 = Fixed
        self._apply_material()
        return self

    def add_pinned(self, carrier_abd_body_id: int, carrier_anchor_vertex: int) -> "GelBody":
        """Add the gel with its stick vertices hard-pinned to an ABD carrier body.

        Uses the engine's anchor-based pin (``add_fem_pin_to_abd``), which derives
        each pin's coordinate in the ABD rest frame inside ``finalize()`` from the
        vertex's actual world position (``StiffGIPC/sim_engine.cu``)::

            local_pos = A(q)^T · (world − q.t)     # once, in finalize()
            world     = q.t + A(q) · local_pos     # every step, _apply_fem_pins

        Do NOT hand-compute ``local_pos`` for the bulk
        ``add_fem_pins_with_local_pos`` path unless you know this engine's ``q``
        convention exactly: ``q.t`` is the body's own translation DOF, not the mesh
        centroid. Getting it wrong silently yanks the gel — measured 51 mm of pin
        displacement on frame 1, 242 inverted tets, and the first line search
        failing with E=inf.

        This pin is **one-way by construction**: pinned vertices get
        ``BoundaryType = Fixed``, their rows/cols are dropped from the IPC Hessian,
        and their positions are overwritten from the ABD ``q`` after every
        line-search step, so the carrier feels no reaction from the gel — the same
        one-way behaviour as Taccel's kinematic penalty. Use
        ``Engine.add_stitch_spring`` if you need bilateral coupling.

        Args:
            carrier_abd_body_id: ABD body id of the link (``BodyView.body_id``).
            carrier_anchor_vertex: any global vertex index of that ABD body.
                Bookkeeping only on this path (the per-step kernel reads just the
                body's ``q``), but it must be a valid index.
        Must be called BEFORE engine.finalize().
        """
        e, a = self.engine, self.asset
        # Place the gel with the engine's `transform` argument and the pristine
        # rest points -- do NOT pre-rotate the vertices and pass identity. With
        # another ABD body already loaded, the pre-transformed path silently comes
        # back unrotated (measured: engine y span 40.3..80.3 mm instead of
        # 60.3..66.2), while the transform argument is order-independent.
        e.load_mesh_from_data(
            a.points @ self.world_tf[:3, :3].T + self.world_tf[:3, 3],
            a.tets, verts_per_face=4, dimensions=3,
            body_type="FEM", transform=np.eye(4),
            young_modulus=a.material.E, boundary_type="Free",
        )
        self._record_offsets()
        for vid in self.global_ids(np.nonzero(a.stick_mask)[0]):
            e.native.add_fem_pin_to_abd(
                int(vid), int(carrier_anchor_vertex), int(carrier_abd_body_id),
                (0.0, 0.0, 0.0),
            )
        self.n_pins = int(a.stick_mask.sum())
        self._apply_material()
        return self

    def add_stitched(self, carrier_abd_body_id: int, carrier_anchor_vertex: int,
                     anchor_world: "np.ndarray") -> "GelBody":
        """Add the gel with its stick vertices SOFT-stitched to an ABD carrier.

        This is the faithful equivalent of Taccel's kinematic-target penalty on
        the stick vertices (their kinematic stiffness 1e5): each stick vertex is
        pulled toward ``anchor_vertex_world + const_offset`` by a spring of
        stiffness ``Config.soft_motion_rate``. The boundary force is therefore
        BOUNDED -- under contact resistance the gel lags the carrier instead of
        having the displacement hard-injected. Measured on the peg scene: the
        hard-pin drive (add_pinned) ground the solver into 175 line-search
        failures at 4.5 s/step with zero peg motion; the stitch drive is the
        one that works.

        The constant world offset is exact for translation-only carrier motion
        (our kinematic drives); under carrier ROTATION the offsets do not
        co-rotate -- use add_pinned for rotating carriers.

        Args:
            carrier_abd_body_id: ABD body id of the carrier.
            carrier_anchor_vertex: a global vertex index of the carrier whose
                position the springs track.
            anchor_world: (3,) world position of that anchor vertex at build
                time (pass it from the geometry you authored; the engine's
                vertex buffer is not readable before finalize()).
        Must be called BEFORE engine.finalize().
        """
        e, a = self.engine, self.asset
        e.load_mesh_from_data(
            a.points, a.tets, verts_per_face=4, dimensions=3,
            body_type="FEM", transform=self.world_tf,
            young_modulus=a.material.E, boundary_type="Free",
        )
        self._record_offsets()
        verts_world = a.points @ self.world_tf[:3, :3].T + self.world_tf[:3, 3]
        anchor_world = np.asarray(anchor_world, dtype=np.float64)
        stick_local = np.nonzero(a.stick_mask)[0]
        for li, vid in zip(stick_local, self.global_ids(stick_local)):
            off = verts_world[li] - anchor_world
            e.add_stitch_spring(int(vid), int(carrier_anchor_vertex),
                                int(carrier_abd_body_id), tuple(off))
        self.n_stitches = len(stick_local)
        self._apply_material()
        return self

    def _record_offsets(self) -> None:
        records = self.engine.get_load_records()
        self.body_offset = len(records) - 1
        self.vertex_offset = int(records[-1].vertex_offset)

    def _apply_material(self) -> None:
        m = self.asset.material
        try:
            self.engine.set_soft_body_density(self.body_offset, m.density)
        except Exception as exc:  # engine may derive mass differently per version
            print(f"[stiff_tactile] set_soft_body_density skipped: {exc}")
        self.engine.set_body_friction(self.body_offset, m.mu)

    def global_ids(self, local_ids: np.ndarray) -> np.ndarray:
        """Gel-local vertex indices -> engine global vertex indices."""
        if self.vertex_offset is None:
            raise RuntimeError("call add_fixed()/add_pinned() first")
        return np.asarray(local_ids, dtype=np.int64) + self.vertex_offset

    # -------------------------------------------------------------- run phase

    @property
    def slice(self) -> slice:
        return slice(self.vertex_offset, self.vertex_offset + self.asset.n_verts)

    def verts_world(self) -> np.ndarray:
        """(N, 3) current gel vertex positions in world frame."""
        return np.asarray(self.engine.get_vertices())[self.slice]

    def verts_local(self, sensor_tf: np.ndarray | None = None) -> np.ndarray:
        """(N, 3) current gel vertices in the SENSOR-LOCAL frame.

        Pass the sensor's current world pose (link pose @ asset.attach_rel_tf).
        With ``mode="fixed"`` the sensor never moves, so the build-time
        ``world_tf`` is correct and can be omitted.
        """
        t = _tf(sensor_tf) if sensor_tf is not None else self.world_tf
        return (self.verts_world() - t[:3, 3]) @ t[:3, :3]

    def markers_world(self) -> np.ndarray:
        return self.asset.markers_from_verts(self.verts_world())

    def contact_force(self, components: str = "total", include_ground: bool = False) -> np.ndarray:
        """(3,) net contact force on the gel in NEWTONS.

        Uses ``get_vertex_contact_forces`` (physical units) rather than the
        legacy ``get_body_contact_force`` (which returns -force*dt^2 and omits
        friction).  ``components``: "normal" | "friction_lagged" | "total".
        """
        f = self.engine.get_vertex_contact_forces(include_ground, components)
        return np.asarray(f)[self.slice].sum(axis=0)

    def contact_force_map(self, components: str = "total") -> np.ndarray:
        """(N, 3) per-vertex contact force in Newtons for this gel."""
        return np.asarray(self.engine.get_vertex_contact_forces(False, components))[self.slice]

    def von_mises(self) -> np.ndarray:
        """(N,) per-vertex von Mises stress (Pa) for this gel."""
        return np.asarray(self.engine.get_fem_von_mises_stress())[self.slice]

    def observe(self, sensor_tf: np.ndarray | None = None, with_markers: bool = True) -> dict:
        """depth / normal / marker flow for the current engine state."""
        return observe(self.asset, self.verts_local(sensor_tf), with_markers=with_markers)

    # ------------------------------------------------------------ diagnostics

    def health_check(self) -> dict:
        """Cheap per-step sanity numbers: penetration proxy + element inversion."""
        v = self.verts_local()
        t = self.asset.tets
        d = np.einsum(
            "ij,ij->i",
            np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]]),
            v[t[:, 3]] - v[t[:, 0]],
        )
        rest = self.asset.points
        d0 = np.einsum(
            "ij,ij->i",
            np.cross(rest[t[:, 1]] - rest[t[:, 0]], rest[t[:, 2]] - rest[t[:, 0]]),
            rest[t[:, 3]] - rest[t[:, 0]],
        )
        return {
            "inverted_tets": int(np.sum(np.sign(d) != np.sign(d0))),
            "min_volume_ratio": float(np.min(np.abs(d) / np.maximum(np.abs(d0), 1e-30))),
            "max_coat_z": float(v[self.asset.coat_mask, 2].max()),
        }

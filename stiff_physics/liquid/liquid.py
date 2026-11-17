"""MPM Lite water for StiffGIPC scenes.

`LiquidMPM` wraps the vendored MPM Lite solver (``mpmlite/``, Feng et al.
2026) and adds what the paper leaves out for our use:

* a world-space domain (the solver grid starts at ``origin``),
* particle seeding from boxes and closed surface meshes,
* static half-space boundaries (ground plane, domain walls),
* **moving triangle-surface boundaries**: the engine's rigid (ABD) and soft
  (FEM) surface triangles are rasterised every sub-step into grid-node
  boundary conditions (separable, with the surface velocity) through a Warp
  BVH mesh query, which is the one-way solid -> liquid coupling,
* sub-stepping of a solid step (``dt_solid`` = 5 ms) into liquid steps.

Conventions: positions are metres, world frame, Y-up or Z-up is decided by
the gravity vector.  Surface triangles must be wound outward (the engine's
tet-boundary faces are; check GLB-derived meshes with
``LiquidMPM.check_orientation``).
"""
from __future__ import annotations

import math
import numpy as np
import warp as wp

from .mpmlite.solver3d import MPMSolver
from .mpmlite.types import Material, real, vec3
from .mpmlite.sp_grid import B, activate_blocks_by_ijk, get_active_block_num
from .mpmlite.boundary_utils import paint_bc_kernel

BC_FREE, BC_STICKY, BC_SLIP, BC_SEPARABLE = 0, 1, 2, 3


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------
@wp.kernel
def _surface_bc_kernel(
    mesh_id: wp.uint64,
    dx: float,
    reach: float,
    band_ext: float,
    nx: int, ny: int, nz: int,
    bc_type_in: int,
    interior_type: int,
    max_bc: int,
    bc_count: wp.array(dtype=int),
    bc_ijk: wp.array(dtype=wp.vec3i),
    bc_type: wp.array(dtype=int),
    bc_n: wp.array(dtype=vec3),
    bc_v: wp.array(dtype=vec3),
):
    """One thread per grid node.  Nodes INSIDE the (thickened) solid always get
    a boundary condition (`interior_type`); nodes outside get one only within
    `band_ext` of the surface (`bc_type_in`; default 1 dx).  Normal = nearest face normal,
    velocity = surface velocity at the closest point.

    History (2026-10-05): an exterior band of 1 dx with any interior rule kept
    the water inside a tilted beaker until 70-85 deg: the thin sheet leaving
    the lip interpolates from inner-wall nodes whose normal blocks outward
    motion.  Interior-only nodes (band_ext = 0) on a mesh thickened to >= 2
    cells seal walls without touching the flow outside."""
    i, j, k = wp.tid()
    if i >= nx or j >= ny or k >= nz:
        return
    p = wp.vec3(float(i) * dx, float(j) * dx, float(k) * dx)
    q = wp.mesh_query_point_sign_normal(mesh_id, p, reach)
    if not q.result:
        return
    cp = wp.mesh_eval_position(mesh_id, q.face, q.u, q.v)
    d = wp.length(p - cp)
    inside = q.sign < 0.0 or d < 1.0e-4 * dx
    if (not inside) and d > band_ext:
        return
    n = wp.mesh_eval_face_normal(mesh_id, q.face)
    if d > 1.0e-3 * dx:
        # closest-point direction = normal of the offset surface: correct at
        # edges and corners where the nearest face normal would leave one
        # velocity component unconstrained (water leaked along box edges)
        nd = (p - cp) / d
        if inside:
            nd = -nd
        n = nd
    v = wp.mesh_eval_velocity(mesh_id, q.face, q.u, q.v)
    slot = wp.atomic_add(bc_count, 0, 1)
    if slot >= max_bc:
        return
    bc_ijk[slot] = wp.vec3i(i, j, k)
    if inside:
        bc_type[slot] = interior_type
    else:
        bc_type[slot] = bc_type_in
    bc_n[slot] = vec3(real(n[0]), real(n[1]), real(n[2]))
    bc_v[slot] = vec3(real(v[0]), real(v[1]), real(v[2]))


@wp.kernel
def _particle_project_kernel(
    mesh_id: wp.uint64,
    max_dist: float,
    push: float,
    ptc_x: wp.array(dtype=vec3),
    ptc_v: wp.array(dtype=vec3),
    n_hit: wp.array(dtype=int),
):
    """Safety net for thin walls: a particle found inside the solid is moved to
    the closest surface point (+push along the normal) and its approaching
    normal velocity relative to the surface is removed."""
    p = wp.tid()
    x = ptc_x[p]
    xf = wp.vec3(float(x[0]), float(x[1]), float(x[2]))
    q = wp.mesh_query_point_sign_normal(mesh_id, xf, max_dist)
    if not q.result:
        return
    if q.sign >= 0.0:
        return
    cp = wp.mesh_eval_position(mesh_id, q.face, q.u, q.v)
    n = wp.mesh_eval_face_normal(mesh_id, q.face)
    vb = wp.mesh_eval_velocity(mesh_id, q.face, q.u, q.v)
    xn = cp + push * n
    ptc_x[p] = vec3(real(xn[0]), real(xn[1]), real(xn[2]))
    nd = vec3(real(n[0]), real(n[1]), real(n[2]))
    vbd = vec3(real(vb[0]), real(vb[1]), real(vb[2]))
    vrel = wp.dot(ptc_v[p] - vbd, nd)
    if vrel < real(0.0):
        ptc_v[p] = ptc_v[p] - vrel * nd
    wp.atomic_add(n_hit, 0, 1)


@wp.kernel
def _nan_guard_kernel(
    ptc_x: wp.array(dtype=vec3),
    ptc_v: wp.array(dtype=vec3),
    ptc_x_prev: wp.array(dtype=vec3),
    n_bad: wp.array(dtype=int),
):
    """A particle whose position went non-finite (seen once in 3000 after a
    projection) would poison every grid node it touches: restore its last
    finite position and zero its velocity."""
    p = wp.tid()
    x = ptc_x[p]
    if wp.isfinite(x[0]) and wp.isfinite(x[1]) and wp.isfinite(x[2]):
        ptc_x_prev[p] = x
        return
    ptc_x[p] = ptc_x_prev[p]
    ptc_v[p] = vec3(real(0.0), real(0.0), real(0.0))
    wp.atomic_add(n_bad, 0, 1)


@wp.kernel
def _to_f32_kernel(x: wp.array(dtype=vec3), out: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    out[i] = wp.vec3(float(x[i][0]), float(x[i][1]), float(x[i][2]))


@wp.kernel
def _push_apart_kernel(
    grid_id: wp.uint64,
    xf: wp.array(dtype=wp.vec3),
    r_min: float,
    max_move: float,
    disp: wp.array(dtype=wp.vec3),
):
    """Anti-compaction: particles closer than r_min are pushed apart by half
    the overlap each (symmetric), capped at max_move per call."""
    i = wp.tid()
    xi = xf[i]
    d = wp.vec3(0.0, 0.0, 0.0)
    for j in wp.hash_grid_query(grid_id, xi, r_min):
        if j == i:
            continue
        dx = xi - xf[j]
        L = wp.length(dx)
        if L < r_min and L > 1.0e-9:
            d = d + (0.5 * (r_min - L) / L) * dx
    Ld = wp.length(d)
    if Ld > max_move:
        d = d * (max_move / Ld)
    disp[i] = d


@wp.kernel
def _apply_disp_kernel(x: wp.array(dtype=vec3), disp: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    d = disp[i]
    x[i] = x[i] + vec3(real(d[0]), real(d[1]), real(d[2]))


@wp.kernel
def _lerp_points_kernel(
    a: wp.array(dtype=wp.vec3),
    b: wp.array(dtype=wp.vec3),
    t: float,
    out: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    out[i] = a[i] + t * (b[i] - a[i])


# --------------------------------------------------------------------------
class LiquidMPM:
    def __init__(self, origin, extent, dx, device="cuda:0", gravity=(0.0, -9.81, 0.0),
                 kappa=1e5, density=1000.0, ppc=8, solver="lite_explicit", flip_ratio=0.0,
                 substep_dt=1e-4, band=None, domain_walls=True, verbose=False, step_kwargs=None, apic=True):
        self.origin = np.asarray(origin, np.float64).reshape(3)
        self.extent = np.asarray(extent, np.float64).reshape(3)
        self.dx = float(dx)
        self.device = device
        self.density = float(density)
        self.kappa = float(kappa)
        self.ppc = float(ppc)
        self.substep_dt = float(substep_dt)
        # band: mesh-query reach (must cover the thickened wall interior);
        # band_ext: exterior nodes within this distance also get a separable BC.
        # Validated 2026-10-05 (static tilted box, 23 k particles, dx 5 mm, 4 mm walls
        # thickened by 4 mm): flip_ratio 0 (pure APIC) + interior & 1 dx exterior
        # separable nodes + projection net -> 97 % contained, ~10 % clumped at 2 s.
        # FLIP 0.95 leaked through the walls (particles keep their own approaching
        # velocity) and the projection net then stacked them into clumps.
        self.band = float(band) if band is not None else 3.0 * self.dx    # mesh-query reach: must cover the thickened wall interior
        self.band_ext = 1.0 * self.dx
        self.verbose = verbose
        self.grid_size = tuple(int(math.ceil(e / self.dx)) + 1 for e in self.extent)
        g = np.asarray(gravity, np.float64); self._up = -g / (np.linalg.norm(g) + 1e-12)
        self.solver = MPMSolver(grid_size=self.grid_size, dx=self.dx, device=device,
                                gravity=tuple(float(g) for g in gravity), n_psi=1, ppc=ppc,
                                solver_type=solver, enable_apic=bool(apic), flip_ratio=flip_ratio)
        self.solver.add_material(0, Material.water, kappa=self.kappa)
        self.solver.ensure_bc_storage()
        self.solver.set_dt(self.substep_dt)
        self._halfspaces = []
        if domain_walls:
            self.add_domain_walls()
        self._commit_halfspaces()
        # surface-boundary storage
        self.mesh = None
        self._mesh_pts_prev = None
        self._mesh_pts_now = None
        self._mesh_vel = None
        self.max_bc = 1 << 16
        self._alloc_bc_lists()
        self.n_bc = 0
        self.sim_time = 0.0
        self.particle_projection = True
        self.push_apart = 0.9       # minimum particle spacing in units of the seeding spacing h (0 = off); guards against sub-cell compaction/clumping
        self.interior_type = BC_SLIP          # grid nodes inside a solid: two-sided (one-sided far-half nodes leak); exterior band: separable
        # implicit mode options (max_iters, v_tol, damping, print_every) passed to MPMSolver.step
        self.step_kwargs = dict(step_kwargs or {})
        if solver == "lite_implicit":
            self.step_kwargs.setdefault("max_iters", 20); self.step_kwargs.setdefault("v_tol", 1e-4)
            self.step_kwargs.setdefault("print_every", 10 ** 9)
        self._n_hit = wp.zeros(1, dtype=int, device=device)

    # ------------------------------------------------------------ boundaries
    def add_halfspace(self, point, normal, btype=BC_SEPARABLE, velocity=(0, 0, 0)):
        """Static half-space: liquid is kept on the side `normal` points to."""
        n = np.asarray(normal, np.float64); n = n / np.linalg.norm(n)
        self._halfspaces.append((np.asarray(point, np.float64) - self.origin, n,
                                 np.asarray(velocity, np.float64), int(btype)))
        self._commit_halfspaces()

    def add_domain_walls(self, inset=None, btype=BC_SEPARABLE):
        m = 1.5 * self.dx if inset is None else float(inset)
        L = (np.array(self.grid_size) - 1) * self.dx
        for ax in range(3):
            n = np.zeros(3); n[ax] = 1.0; p = np.zeros(3); p[ax] = m
            self._halfspaces.append((p, n.copy(), np.zeros(3), btype))
            p2 = np.zeros(3); p2[ax] = L[ax] - m
            self._halfspaces.append((p2, -n, np.zeros(3), btype))

    def _commit_halfspaces(self):
        if not self._halfspaces:
            return
        P = np.array([h[0] for h in self._halfspaces]); N = np.array([h[1] for h in self._halfspaces])
        V = np.array([h[2] for h in self._halfspaces]); T = np.array([h[3] for h in self._halfspaces], np.int32)
        self.solver.paint_hf_boundary(P, N, V, T)

    def _alloc_bc_lists(self):
        d = self.device
        self._bc_count = wp.zeros(1, dtype=int, device=d)
        self._bc_ijk = wp.zeros(self.max_bc, dtype=wp.vec3i, device=d)
        self._bc_type = wp.zeros(self.max_bc, dtype=int, device=d)
        self._bc_n = wp.zeros(self.max_bc, dtype=vec3, device=d)
        self._bc_v = wp.zeros(self.max_bc, dtype=vec3, device=d)

    # ------------------------------------------------------------ surfaces
    @staticmethod
    def check_orientation(verts, faces):
        """Signed volume of a closed mesh: positive = outward winding."""
        v = np.asarray(verts, np.float64); f = np.asarray(faces, np.int64)
        a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)

    @staticmethod
    def _vertex_normals(V, F):
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        fn = np.cross(b - a, c - a)                      # area-weighted face normals
        vn = np.zeros_like(V)
        for k in range(3):
            np.add.at(vn, F[:, k], fn)
        nrm = np.linalg.norm(vn, axis=1); nrm[nrm == 0] = 1.0
        return vn / nrm[:, None]

    def _thickened(self, V):
        if self.thicken <= 0.0:
            return V
        n = self._vertex_normals(V, self._faces3)
        if self._thicken_outward_mask is None:
            return V + self.thicken * n
        return V + (self.thicken * self._thicken_outward_mask)[:, None] * n

    def _outward_mask(self, V):
        """Per-vertex 1/0: vertex normal points away from the centroid of its own
        connected component (outer skin, rim top, floor bottom) -> offset; inner
        cavity surfaces are left exact so the liquid volume is not reduced."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        F = self._faces3; nv = len(V)
        rows = np.r_[F[:, 0], F[:, 1], F[:, 2]]; cols = np.r_[F[:, 1], F[:, 2], F[:, 0]]
        g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(nv, nv))
        _, lab = connected_components(g, directed=False)
        cent = np.zeros((lab.max() + 1, 3))
        for c in range(lab.max() + 1):
            cent[c] = V[lab == c].mean(0)
        n = self._vertex_normals(V, F)
        outward = np.einsum("ij,ij->i", n, V - cent[lab]) > 0.0
        not_up = n @ self._up < 0.7              # rim tops / lips must not rise (they would dam the opening)
        # keep the lip zone (upper 3 dx of each body) at its true thickness: a
        # thickened outer wall right under the lip turns the pour into a dribble
        # that runs down the outside (v5: the stream landed 30 mm behind the lip)
        h = V @ self._up
        top = np.zeros(lab.max() + 1)
        for c in range(lab.max() + 1):
            top[c] = h[lab == c].max()
        below_lip = h < top[lab] - 3.0 * self.dx
        return (outward & not_up & below_lip).astype(np.float64)

    def set_surface_mesh(self, verts, faces, thicken=None, outward_only=True):
        """(Re)build the boundary mesh from world-space vertices and triangles.

        thicken: offset (m) applied along vertex normals so that thin walls
        span >= 2 grid cells (default 2 dx); with outward_only (default) only
        vertices whose normal points away from their body's centroid move
        (outer skin, rim top, underside), so cavities keep their exact shape
        and the liquid volume.  The offset is recomputed from the current
        vertex positions on every update so it follows rigid motion."""
        self._faces3 = np.asarray(faces, np.int64).reshape(-1, 3)
        self.thicken = float(thicken) if thicken is not None else 2.0 * self.dx
        V0 = np.asarray(verts, np.float64) - self.origin
        self._thicken_outward_mask = self._outward_mask(V0) if (outward_only and self.thicken > 0) else None
        V = self._thickened(V0).astype(np.float32)
        F = np.asarray(faces, np.int32).reshape(-1)
        d = self.device
        self._mesh_pts_prev = wp.array(V, dtype=wp.vec3, device=d)
        self._mesh_pts_now = wp.array(V, dtype=wp.vec3, device=d)
        pts = wp.array(V, dtype=wp.vec3, device=d)
        self._mesh_vel = wp.zeros(len(V), dtype=wp.vec3, device=d)
        self.mesh = wp.Mesh(points=pts, indices=wp.array(F, dtype=wp.int32, device=d), velocities=self._mesh_vel)
        self._faces = F

    def update_surface(self, verts_now, dt_solid):
        """Call once per solid step with the committed vertex positions; the
        previous call's positions give the surface velocity."""
        V = self._thickened(np.asarray(verts_now, np.float64) - self.origin).astype(np.float32)
        self._mesh_pts_prev.assign(self._mesh_pts_now)
        self._mesh_pts_now.assign(wp.array(V, dtype=wp.vec3, device=self.device))
        prev = self._mesh_pts_prev.numpy(); now = self._mesh_pts_now.numpy()
        self._mesh_vel.assign(wp.array(((now - prev) / float(dt_solid)).astype(np.float32), dtype=wp.vec3, device=self.device))

    def _paint_surface(self, theta, btype=BC_SEPARABLE):
        """Rasterise the surface at interpolation parameter theta in [0,1]
        between the previous and current solid positions into node BCs."""
        if self.mesh is None:
            return 0
        wp.launch(_lerp_points_kernel, dim=len(self._faces) and self._mesh_pts_now.shape[0],
                  inputs=[self._mesh_pts_prev, self._mesh_pts_now, float(theta), self.mesh.points], device=self.device)
        self.mesh.refit()
        while True:
            self._bc_count.zero_()
            nx, ny, nz = self.grid_size
            wp.launch(_surface_bc_kernel, dim=(nx, ny, nz),
                      inputs=[self.mesh.id, float(self.dx), float(max(self.band, self.band_ext)), float(self.band_ext), nx, ny, nz, int(btype), int(self.interior_type), self.max_bc,
                              self._bc_count, self._bc_ijk, self._bc_type, self._bc_n, self._bc_v], device=self.device)
            n = int(self._bc_count.numpy()[0])
            if n <= self.max_bc:
                break
            self.max_bc = max(2 * self.max_bc, n)
            self._alloc_bc_lists()
        self.n_bc = n
        self._paint_nodes_device(n)
        return n

    def _paint_nodes_device(self, n):
        """Device-side version of MPMSolver.paint_boundary that reuses storage."""
        s = self.solver
        s.bc_block_count.zero_(); s.bc_block_activated.zero_()
        if n > 0:
            wp.launch(activate_blocks_by_ijk, dim=n, inputs=[self._bc_ijk, n, s.bc_block_activated], device=self.device)
        wp.launch(get_active_block_num, dim=s.block_size, inputs=[s.bc_block_activated, s.bc_block_count], device=self.device)
        nb = int(s.bc_block_count.numpy()[0])
        while nb >= s.BC_MAX_BLOCKS:
            s.BC_MAX_BLOCKS *= 2
            s.bc_block_xyz_by_id = wp.zeros((s.BC_MAX_BLOCKS,), dtype=wp.vec3i, device=self.device)
            s.bc_type = wp.zeros((s.BC_MAX_BLOCKS, B, B, B), dtype=int, device=self.device)
            s.bc_norm = wp.zeros((s.BC_MAX_BLOCKS, B, B, B), dtype=s.vecn, device=self.device)
            s.bc_velo = wp.zeros((s.BC_MAX_BLOCKS, B, B, B), dtype=s.vecn, device=self.device)
        s.bc_block2bid.fill_(-1)
        s.bc_type.zero_(); s.bc_norm.zero_(); s.bc_velo.zero_()
        if n == 0:
            return
        s.ps3d.compute(s.bc_block_activated, s.bc_block_xyz_by_id, s.bc_block2bid)
        wp.launch(paint_bc_kernel, dim=n,
                  inputs=[self._bc_ijk, self._bc_type, self._bc_n, self._bc_v,
                          s.bc_block2bid, s.bc_type, s.bc_norm, s.bc_velo], device=self.device)
        s.need_record_again = True

    # ------------------------------------------------------------ seeding
    def _lattice(self, lo, hi):
        h = self.dx / self.ppc ** (1.0 / 3.0)
        axes = [np.arange(lo[a] + 0.5 * h, hi[a], h) for a in range(3)]
        g = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        return g, h ** 3

    def seed_box(self, lo, hi, jitter=0.0, velocity=(0, 0, 0)):
        lo = np.asarray(lo, np.float64); hi = np.asarray(hi, np.float64)
        pts, vol0 = self._lattice(lo, hi)
        if jitter > 0:
            pts = pts + (np.random.rand(*pts.shape) - 0.5) * jitter * self.dx / self.ppc ** (1 / 3)
        return self._seed(pts, vol0, velocity)

    def seed_mesh(self, verts, faces, jitter=0.0, velocity=(0, 0, 0), max_points=None):
        """Fill a closed surface mesh (world coords) with particles."""
        import trimesh
        m = trimesh.Trimesh(np.asarray(verts, np.float64), np.asarray(faces, np.int64), process=False)
        lo, hi = m.bounds
        pts, vol0 = self._lattice(lo, hi)
        if jitter > 0:
            pts = pts + (np.random.rand(*pts.shape) - 0.5) * jitter * self.dx / self.ppc ** (1 / 3)
        inside = m.contains(pts)
        pts = pts[inside]
        if max_points is not None and len(pts) > max_points:
            pts = pts[np.random.choice(len(pts), max_points, replace=False)]
        return self._seed(pts, vol0, velocity)

    def _seed(self, pts_world, vol0, velocity):
        pts = pts_world - self.origin
        L = (np.array(self.grid_size) - 1) * self.dx
        ok = np.all((pts > 1.0 * self.dx) & (pts < L - 1.0 * self.dx), axis=1)
        if not ok.all() and self.verbose:
            print(f"[liquid] dropping {int((~ok).sum())} seed points outside the domain")
        pts = pts[ok]
        self.solver.seed_particles(pts, 0, self.density, vol0, velocity=np.asarray(velocity, np.float64))
        return len(pts)

    # ------------------------------------------------------------ stepping
    @property
    def n_particles(self):
        return self.solver.n_ptc

    def positions(self):
        return self.solver.get_points() + self.origin

    def velocities(self):
        return self.solver.ptc_v.numpy().copy()

    def step(self, dt_solid, surface_substeps=True):
        """Advance the liquid by one solid step (sub-stepped)."""
        n_sub = max(1, int(round(dt_solid / self.substep_dt)))
        dt = dt_solid / n_sub
        self.solver.set_dt(dt)
        for s in range(n_sub):
            if self.mesh is not None and (surface_substeps or s == 0):
                self._paint_surface((s + 1) / n_sub)
            self.solver.step(**self.step_kwargs)
            if self.mesh is not None and self.particle_projection:
                self._project_particles()
            if self.push_apart > 0.0:
                self._push_apart()
            self._nan_guard()
        self.sim_time += dt_solid

    def _push_apart(self):
        """Keep particles at least push_apart * h apart (h = seeding spacing):
        the J-tracked water does not react to sub-cell compaction, so without
        this the column slowly sinks (15 % in 3 s at rest) and then clumps."""
        n = self.solver.n_ptc
        if n == 0:
            return
        h = self.dx / self.ppc ** (1.0 / 3.0)
        r_min = float(self.push_apart * h)
        if getattr(self, "_pa_n", 0) != n:
            self._pa_xf = wp.zeros(n, dtype=wp.vec3, device=self.device)
            self._pa_disp = wp.zeros(n, dtype=wp.vec3, device=self.device)
            self._pa_grid = wp.HashGrid(128, 128, 128, device=self.device)
            self._pa_n = n
        wp.launch(_to_f32_kernel, dim=n, inputs=[self.solver.ptc_x, self._pa_xf], device=self.device)
        self._pa_grid.build(self._pa_xf, r_min)
        wp.launch(_push_apart_kernel, dim=n, inputs=[self._pa_grid.id, self._pa_xf, r_min, float(0.25 * h), self._pa_disp], device=self.device)
        wp.launch(_apply_disp_kernel, dim=n, inputs=[self.solver.ptc_x, self._pa_disp], device=self.device)

    def _nan_guard(self):
        n = self.solver.n_ptc
        if n == 0:
            return
        if getattr(self, "_x_prev", None) is None or self._x_prev.shape[0] != n:
            self._x_prev = wp.clone(self.solver.ptc_x)
            self._n_bad = wp.zeros(1, dtype=int, device=self.device)
        wp.launch(_nan_guard_kernel, dim=n, inputs=[self.solver.ptc_x, self.solver.ptc_v, self._x_prev, self._n_bad], device=self.device)

    def nan_restored(self):
        return int(self._n_bad.numpy()[0]) if getattr(self, "_n_bad", None) is not None else 0

    def _project_particles(self):
        n = self.solver.n_ptc
        if n == 0:
            return
        self._n_hit.zero_()
        wp.launch(_particle_project_kernel, dim=n,
                  inputs=[self.mesh.id, float(self.dx), float(0.05 * self.dx), self.solver.ptc_x, self.solver.ptc_v, self._n_hit],
                  device=self.device)

    def projected_last(self):
        return int(self._n_hit.numpy()[0])

    def mass(self):
        return float(self.solver.ptc_m.numpy().sum())


# --------------------------------------------------------------------------
class EngineLiquidCoupler:
    """One-way coupling: engine surfaces -> liquid boundary conditions.

    faces: (T,3) int array of GLOBAL engine vertex indices (input order, as
    returned by `Engine.get_vertices()`), covering the bodies the liquid
    should see.  Call `step()` right after `engine.step()`.
    """

    def __init__(self, engine, liquid: LiquidMPM, faces, dt):
        self.engine, self.liquid, self.dt = engine, liquid, float(dt)
        self.faces = np.asarray(faces, np.int64)
        self.used = np.unique(self.faces)
        self.remap = -np.ones(int(self.used.max()) + 1, np.int64)
        self.remap[self.used] = np.arange(len(self.used))
        X = np.asarray(engine.get_vertices(), np.float64)
        liquid.set_surface_mesh(X[self.used], self.remap[self.faces])

    def step(self):
        X = np.asarray(self.engine.get_vertices(), np.float64)
        self.liquid.update_surface(X[self.used], self.dt)
        self.liquid.step(self.dt)

import numpy as np
import warp as wp
def ic(*a, **k):  # icecream removed (vendored copy)
    return None
from .sampling_utils import *
from .types import *
from .math import VecTiledDot, VecTiledSum, PrefixSum3D, matrix_free_cg
try:
    import pyvista as pv  # optional: only the pv-based add_* helpers need it
except ImportError:  # pragma: no cover
    pv = None
import time

from .kernel.d3.helper_system import (
    linesearch_update_velocity,
    node_psi, center_psi, particle_psi,
)
from .kernel.d3.helper_misc import (
    max_particle_speed_kernel,
    update_vnew_with_iterate_kernel,
    commit_dv_to_iterate_kernel,
)
from .kernel.d3.kernel_lite import (
    lite_p2c,
    lite_c2g,
    lite_g2c_kernel,
    lite_c2p_kernel,
)
from .kernel.d3.kernel_lite_misc import (
    lite_initialize_v_iterate_kernel,
    lite_reduce_active_nodes_and_centers_kernel,
    lite_center_grad_from_grid_velocity_kernel,
    lite_update_quadrature_scratch_kernel,
)
from .kernel.d3.kernel_lite_implicit import (
    lite_implicit_grad_v_dv_kernel,
    lite_implicit_precond_kernel_Hii,
    lite_implicit_residual_from_scratch_kernel_fast,
    lite_implicit_update_stress_scratch_kernel,
    lite_implicit_df_kernel_fast,
)
from .sp_grid import (
    activate_blocks, B,
    activate_blocks_by_ijk,
    get_active_block_num,
)
from .boundary_utils import (
    paint_bc_kernel,
    boundary_projection_kernel,
)


class MPMSolver:
    def __init__(
        self,
        grid_size: GridSize,
        dx: float = None,
        device: Optional[str] = 'cpu',
        gravity=(0.0, 0.0, -9.81),   # 3-vector (StiffGIPC patch; scalar = legacy z-up)
        n_psi: int = 1,
        ppc: float = 4,
        solver_type: str = 'lite_explicit',
        enable_apic: bool = True,
        flip_ratio: float = 0.9,
    ):
        if not solver_type in ['lite_explicit', 'lite_implicit']:
            raise ValueError(f"Only 'lite_explicit' and 'lite_implicit' are supported in current repo.")

        self.dim = len(grid_size)
        assert self.dim == 3, "Only 3D is supported."
        self.vecn = vec3
        self.matnn = mat33
        self.matn2 = mat99

        self.use_vbd_pbng = True

        self.solver_type = solver_type
        self.grid_size = wp.vec3i(*grid_size) if self.dim == 3 else wp.vec2i(*grid_size)
        self.dx = dx if dx else 1.0 / (grid_size[0] - 1) # first dimension to 1.0
        self.ppc = ppc # average particles per cell
        g = np.asarray(gravity, dtype=np.float64).reshape(-1)
        if g.size == 1:
            g = np.array([0.0, 0.0, float(g[0])])
        self.gravity = vec3(real(g[0]), real(g[1]), real(g[2]))
        self.flip_ratio = flip_ratio
        self.enable_apic = enable_apic
        self.device = device
        self.n_ptc = 0 # number of particles
        self.n_ptc_wp = wp.zeros(1, dtype=int, device=device)
        self.center_size = wp.vec3i(*tuple(s - 1 for s in grid_size))
        self.n_psi = n_psi # number of different psi functions
        self.mat_initialized = np.zeros((self.n_psi,), dtype=bool)
        self.psi_params = wp.zeros(self.n_psi, dtype=MaterialParams, device=device) # material type + constitutive type

        self.ptc_x    = wp.zeros(0, dtype=self.vecn, device=device)
        self.ptc_v    = wp.zeros(0, dtype=self.vecn, device=device)
        self.ptc_m    = wp.zeros(0, dtype=real, device=device)
        self.ptc_k    = wp.zeros(0, dtype=int, device=device) # index of psi functions / materials
        self.ptc_vol0 = wp.zeros(0, dtype=real, device=device)
        self.ptc_F    = wp.zeros(0, dtype=self.matnn, device=device)
        self.ptc_dlogJ = wp.zeros(0, dtype=real, device=device)
        self.ptc_G = wp.zeros(0, dtype=self.matnn, device=device)
        # data for visualization
        self.ptc_color = np.empty((0, 3), dtype=np.float32)
        self.sim_steps = 0
        self.sim_time = 0.0

        self.pd_damping_arr = wp.zeros(1, dtype=real, device=device)
        self.pd_damping = 1e-2
        self.max_update = wp.zeros(1, dtype=real, device=device)

        self.ps3d = PrefixSum3D(device=device)
        self.block_size = wp.vec3i((self.grid_size.x + B - 1) // B, (self.grid_size.y + B - 1) // B, (self.grid_size.z + B - 1) // B)
        self.block_activated = wp.zeros(self.block_size, dtype=int, device=device)
        self.block2bid = wp.zeros(self.block_size, dtype=int, device=device)
        self.block_activated_sum = VecTiledSum(
            max_length=self.block_size.x * self.block_size.y * self.block_size.z,
            dtype=wp.int32, device=self.device, tile_size=512, max_column_count=1)

        self.MAX_PTS = 1 << 18  # 262K particles to start with
        self.MAX_DOF = 1 << 10  # 1K DOFs to start with
        self.MAX_BLOCKS = 1 << 6 # 64 blocks to start with

        self.init_blocks()
        self.init_dofs()
        self.init_pts_storage()

        self.hf_bc_p = wp.empty(dtype=vec3, device=self.device)
        self.hf_bc_n = wp.empty(dtype=vec3, device=self.device)
        self.hf_bc_v = wp.empty(dtype=vec3, device=self.device)
        self.hf_bc_type = wp.empty(dtype=int, device=self.device)
        self.num_hf = 0
        
    def get_points(self) -> np.array:
        return self.ptc_x.numpy().copy()
    
    def set_dt(self, dt):
        self.dt = dt

    def max_particle_speed(self) -> float:
        vmax = wp.zeros(1, dtype=real, device=self.device)
        wp.launch(
            kernel=max_particle_speed_kernel,
            dim=self.n_ptc,
            inputs=[self.ptc_v, vmax],
            device=self.device,
        )
        # TODO: synchronize?
        return float(vmax.numpy().reshape(-1)[0])

    def init_blocks(self):
        self.block_count = wp.empty((1,), dtype=int, device=self.device)
        self.block_xyz_by_id = wp.zeros(self.MAX_BLOCKS, dtype=wp.vec3i, device=self.device)

        # Grid data
        VOL4D = (self.MAX_BLOCKS, B, B, B)
        VOL4D2 = (self.n_psi, self.MAX_BLOCKS, B, B*B)
        self.grid_m = wp.zeros(VOL4D, dtype=real, device=self.device)
        self.grid_v = wp.zeros(VOL4D, dtype=self.vecn, device=self.device)
        self.grid_v_new = wp.zeros(VOL4D, dtype=self.vecn, device=self.device)

        if not ('explicit' in self.solver_type):        
            self.node2dof = wp.zeros(VOL4D, dtype=int, device=self.device)
            if self.solver_type in ['lite_vbd', 'lite_implicit']:
                self.center2dof = wp.zeros(VOL4D, dtype=int, device=self.device)
            self.grid_v_it = wp.zeros(VOL4D, dtype=self.vecn, device=self.device)   # current iterate v
            if self.solver_type in ['lite_vbd']:
                self.grid_v_it_prev = wp.zeros(VOL4D, dtype=self.vecn, device=self.device)
            if self.solver_type in ['lite_implicit']:
                self.grid_v_ls = wp.zeros(VOL4D, dtype=self.vecn, device=self.device)   # linesearch v
                self.center_G_ls = wp.zeros(VOL4D, dtype=self.matnn, device=self.device)  # NEW: ∇v at centers
            if self.solver_type in ['implicit']:
                self.grid_v_ls = wp.zeros(VOL4D, dtype=self.vecn, device=self.device)   # linesearch v

        if 'lite' in self.solver_type:
            # grid center data
            self.center_m = wp.zeros(VOL4D, dtype=real, device=self.device)
            self.center_v = wp.zeros(VOL4D, dtype=self.vecn, device=self.device)
            self.center_dv = wp.zeros(VOL4D, dtype=self.vecn, device=self.device)  # Δv at centers (from G2C)
            self.center_G = wp.zeros(VOL4D, dtype=self.matnn, device=self.device)  # NEW: ∇v at centers
            self.center_vol = wp.zeros(VOL4D2, dtype=real, device=self.device)
            self.center_tau = wp.zeros(VOL4D2, dtype=self.matnn, device=self.device)  # Σ V0 * τ at centers
            self.center_F = wp.zeros(VOL4D2, dtype=self.matnn, device=self.device)
            self.center_dlogJ = wp.zeros(VOL4D2, dtype=real, device=self.device)

        self.need_record_again = True

    def init_dofs(self):
        device = self.device
        DOF = (self.MAX_DOF,)
        DOF2 = (self.n_psi, self.MAX_DOF)

        if self.solver_type == 'implicit':
            self.ndof2bijk = wp.zeros(DOF, dtype=wp.vec2i, device=device) # x,y,z,block_id
            self.n_active_nodes = wp.zeros(1, dtype=int, device=device)
            self.node_residual = wp.zeros(DOF, dtype=self.vecn, device=device)
            self.node_residual_ls = wp.zeros(DOF, dtype=self.vecn, device=device)
            self.dv_guess = wp.zeros(DOF, dtype=self.vecn, device=device)
            self.A_matvec_out = wp.zeros(DOF, dtype=self.vecn, device=device)
            self.node_psi = wp.zeros(DOF, dtype=real, device=self.device)
            self.tiled_sum0 = VecTiledSum(self.MAX_DOF, dtype=real, device=self.device, tile_size=512)

        if self.solver_type == 'lite_vbd':
            self.ndof2bijk = wp.zeros(DOF, dtype=wp.vec2i, device=device) # x,y,z,block_id
            self.n_active_nodes = wp.zeros(1, dtype=int, device=device)
            self.cdof2bijk = wp.zeros(DOF, dtype=wp.vec2i, device=device) # x,y,z,block_id
            self.n_active_centers = wp.zeros(1, dtype=int, device=device)

            # --- VBD storage ---
            # nodal residual & 2x2 diagonal block for Jacobi step
            self.vbd_res = wp.zeros(DOF, dtype=self.vecn, device=self.device)      # g_i = M(v - v̂) - Δt f_int - Δt f_ext
            self.vbd_H = wp.zeros(DOF, dtype=self.matnn, device=self.device)  # 2x2 block (diagonal of global Hessian)
            self.chebyshev_omega_wp = wp.zeros(1, dtype=real, device=self.device)

            self.center_stress_scratch = wp.zeros(DOF2, dtype=self.matnn, device=device)
            if self.use_vbd_pbng:
                self.center_quadrature_scratch = wp.zeros(DOF2, dtype=QuadratureScratch3D_PBNG, device=self.device)
            else:
                pass
            
        if self.solver_type == 'lite_implicit':
            self.ndof2bijk = wp.zeros(DOF, dtype=wp.vec2i, device=device) # x,y,z,block_id
            self.n_active_nodes = wp.zeros(1, dtype=int, device=device)
            self.cdof2bijk = wp.zeros(DOF, dtype=wp.vec2i, device=device) # x,y,z,block_id
            self.n_active_centers = wp.zeros(1, dtype=int, device=device)
            self.node_residual = wp.zeros(DOF, dtype=self.vecn, device=device)
            self.node_residual_ls = wp.zeros(DOF, dtype=self.vecn, device=device)
            self.node_Hii_inv = wp.zeros(DOF, dtype=self.matnn, device=device)  # diagonal preconditioner
            self.node_tikhonov = wp.zeros(DOF, dtype=real, device=device)  # diagonal preconditioner
            self.dv_guess = wp.zeros(DOF, dtype=self.vecn, device=device)
            self.A_matvec_out = wp.zeros(DOF, dtype=self.vecn, device=device)
            self.center_grad_dv_tmp = wp.zeros(DOF, dtype=self.matnn, device=device)
            self.center_stress_scratch = wp.zeros(DOF2, dtype=self.matnn, device=device)
            self.center_quadrature_scratch = wp.zeros(DOF2, dtype=QuadratureScratch3D, device=self.device)
        
            self.center_quadrature_scratch_ls = wp.zeros(DOF2, dtype=QuadratureScratch3D, device=self.device)

            self.center_psi = wp.zeros(DOF2, dtype=real, device=self.device)
            self.node_psi = wp.zeros(DOF, dtype=real, device=self.device)
            self.tiled_sum = VecTiledSum(self.MAX_DOF, dtype=real, device=self.device, tile_size=512)

        self.need_record_again = True

    def init_pts_storage(self):
        device = self.device
        if self.solver_type == 'implicit':
            PTS = (self.MAX_PTS,)
            self.ptc_quadrature_scratch = wp.zeros(PTS, dtype=QuadratureScratch3D, device=device)
            self.ptc_grad_dv_tmp = wp.zeros(PTS, dtype=self.matnn, device=device)
            self.ptc_stress_scratch = wp.zeros(PTS, dtype=self.matnn, device=device)
            self.ptc_G_ls = wp.zeros(PTS, dtype=self.matnn, device=device)
            self.ptc_quadrature_scratch_ls = wp.zeros(PTS, dtype=QuadratureScratch3D, device=device)
            self.tiled_sum1 = VecTiledSum(self.MAX_PTS, dtype=real, device=self.device, tile_size=512)
            self.ptc_psi = wp.zeros(PTS, dtype=real, device=self.device)

    def reset_grid(self):
        # after each transfer, the grid is reset
        if not ('lite' in self.solver_type):
            # lite c2g will initialize grid so we don't need to zero here
            self.grid_m.zero_()
            self.grid_v.zero_()
            self.grid_v_new.zero_()

            if not 'explicit' in self.solver_type:
                self.grid_v_it.zero_()

        if 'lite' in self.solver_type:
            self.center_m.zero_()
            self.center_v.zero_()
            self.center_F.zero_() # we don't need to reset deformation gradient, as we will reconstruct it
            self.center_dv.zero_()
            self.center_G.zero_()
            self.center_vol.zero_()
            self.center_tau.zero_()
            self.center_dlogJ.zero_()
    
    def paint_hf_boundary(self, hf_bc_p:np.array, hf_bc_n:np.array, hf_bc_v:np.array, hf_bc_type:np.array):
        self.hf_bc_p = wp.array(hf_bc_p, dtype=wp.vec3d, device=self.device)
        self.hf_bc_n = wp.array(hf_bc_n, dtype=wp.vec3d, device=self.device)
        self.hf_bc_v = wp.array(hf_bc_v, dtype=wp.vec3d, device=self.device)
        self.hf_bc_type = wp.array(hf_bc_type, dtype=int, device=self.device)
        self.num_hf = len(hf_bc_p)
        assert self.num_hf == len(hf_bc_v)
        assert self.num_hf == len(hf_bc_n)
        assert self.num_hf == len(hf_bc_type)
        self.need_record_again = True

    def paint_boundary(self, boundary_ijk:np.array, boundary: np.array, boundary_n: np.array=None, boundary_v: np.array=None):
        # boundary_ijk: (N, 3) node indices, for which we want to paint boundary conditions
        # boundary: (N, 3) boundary type, 0 for free (should not be passed in), 1 for fixed, 2 for slippery
        # boundary_n: (N, 3) normal of boundary
        n_paint_nodes = boundary_ijk.shape[0]
        # allocate boundary blocks
        boundary_ijk_wp = wp.array(boundary_ijk.astype(np.int32), dtype=wp.vec3i, device=self.device)
        boundary_wp = wp.array(boundary.astype(np.int32), dtype=int, device=self.device)
        boundary_n_wp = wp.array(boundary_n.astype(np.float32), dtype=self.vecn, device=self.device) if boundary_n is not None else (
            wp.zeros(boundary_ijk_wp.shape[0], dtype=self.vecn, device=self.device)
        )
        boundary_v_wp = wp.array(boundary_v.astype(np.float32), dtype=self.vecn, device=self.device) if boundary_v is not None else (
            wp.zeros(boundary_ijk_wp.shape[0], dtype=self.vecn, device=self.device)
        )

        self.BC_MAX_BLOCKS = 1 << 6
        self.bc_block_count = wp.empty((1,), dtype=int, device=self.device)
        self.bc_block_activated = wp.zeros(self.block_size, dtype=int, device=self.device)
        self.bc_block2bid = wp.zeros(self.block_size, dtype=int, device=self.device)
        while True:
            self.bc_block_xyz_by_id = wp.empty((self.BC_MAX_BLOCKS,), dtype=wp.vec3i, device=self.device)
            self.bc_type = wp.zeros((self.BC_MAX_BLOCKS, B, B, B), dtype=int, device=self.device)
            self.bc_norm = wp.zeros((self.BC_MAX_BLOCKS, B, B, B), dtype=self.vecn, device=self.device)
            self.bc_velo = wp.zeros((self.BC_MAX_BLOCKS, B, B, B), dtype=self.vecn, device=self.device)
            self.bc_block_count.fill_(0)
            self.bc_block_activated.fill_(0)
            
            wp.launch(
                kernel=activate_blocks_by_ijk,
                dim=n_paint_nodes,
                inputs=[boundary_ijk_wp, n_paint_nodes, self.bc_block_activated],
                device=self.device,
            )
            wp.launch(
                kernel=get_active_block_num,
                dim=self.block_size,
                inputs=[self.bc_block_activated, self.bc_block_count],
                device=self.device,
            )
            if self.bc_block_count.numpy().reshape(-1)[0] >= self.BC_MAX_BLOCKS:
                self.BC_MAX_BLOCKS *= 2
                continue

            self.block2bid.fill_(-1)

            self.ps3d.compute(
                self.bc_block_activated,
                self.bc_block_xyz_by_id,
                self.bc_block2bid,
            )

            wp.launch(
                kernel=paint_bc_kernel,
                dim=n_paint_nodes,
                inputs=[
                    boundary_ijk_wp,
                    boundary_wp,
                    boundary_n_wp,
                    boundary_v_wp,
                    self.bc_block2bid,
                    self.bc_type,
                    self.bc_norm,
                    self.bc_velo,
                ],
            device=self.device,
        )
            break
        self.need_record_again = True

    def activate_sparse_grid(self):
        while True:
            self.block_activated.fill_(0)
            # 1) activate blocks from particle stencils
            wp.launch(
                kernel=activate_blocks,
                dim=self.n_ptc,
                inputs=[
                    self.ptc_x, self.dx, self.grid_size,
                    self.block_activated,
                ],
                device=self.device
            )
            self.bcn = int(self.block_activated_sum.compute(self.block_activated.reshape(-1)).numpy().reshape(-1)[0])
            self.block_count.fill_(self.bcn)
            
            self.block2bid.fill_(-1)
            self.ps3d.compute(
                self.block_activated,
                self.block_xyz_by_id,
                self.block2bid,
            )

            if self.bcn < self.MAX_BLOCKS: break

            ic(self.bcn, self.MAX_BLOCKS)
            self.MAX_BLOCKS *= 2
            self.init_blocks()
        
        # # (optional) you can read back block_count occasionally
        # if self.sim_steps % 100 == 0:
        #     bc = self.block_count.numpy().reshape(-1)[0]
        #     print(f"step {self.sim_steps}: active blocks = {bc} (cap {MAX_BLOCKS})")
        #     if bc >= MAX_BLOCKS:
        #         print(f"\033[31m [OOM] MAX_BLOCKS cap reached. Increase MAX_BLOCKS or reduce particles.\033[0m")
        #         exit(0)

    def reduce_sparse_grid(self):
        while True:
            overflow = wp.zeros(1, dtype=int, device=self.device)
            self.n_active_nodes.zero_()
            self.node2dof.fill_(-1)
            self.ndof2bijk.fill_(wp.vec2i(-1, -1))

            if 'lite' in self.solver_type:
                self.n_active_centers.zero_()
                self.center2dof.fill_(-1)
                self.cdof2bijk.fill_(wp.vec2i(-1, -1))
                wp.launch(
                    kernel=lite_reduce_active_nodes_and_centers_kernel,
                    dim=(self.bcn, B, B, B),
                    inputs=[
                        self.block_count,
                        self.block_xyz_by_id,
                        self.grid_m,
                        self.n_active_nodes,
                        self.ndof2bijk,
                        self.node2dof,
                        self.center_m,
                        self.n_active_centers,
                        self.cdof2bijk,
                        self.center2dof,
                        self.grid_size,
                        overflow,
                        self.MAX_DOF,
                        True,
                    ],
                    device=self.device,
                )

            else:
                # only reduce nodes
                wp.launch(
                    kernel=lite_reduce_active_nodes_and_centers_kernel,
                    dim=(self.bcn, B, B, B),
                    inputs=[
                        self.block_count,
                        self.block_xyz_by_id,
                        self.grid_m,
                        self.n_active_nodes,
                        self.ndof2bijk,
                        self.node2dof,
                        None,
                        None,
                        None,
                        None,
                        self.grid_size,
                        overflow,
                        self.MAX_DOF,
                        False,
                    ],
                    device=self.device,
                )
            
            if overflow.numpy().reshape(-1)[0] == 0: break
            self.MAX_DOF *= 2
            self.init_dofs()

    def ensure_bc_storage(self):
        """Allocate empty node-boundary storage (StiffGIPC patch): the kernels
        always query bc_* arrays, so a solver without painted boundaries needs
        them to exist with no active block."""
        if hasattr(self, "bc_block2bid"):
            return
        self.BC_MAX_BLOCKS = 1 << 2
        self.bc_block_count = wp.zeros((1,), dtype=int, device=self.device)
        self.bc_block_activated = wp.zeros(self.block_size, dtype=int, device=self.device)
        self.bc_block2bid = wp.full(self.block_size, -1, dtype=int, device=self.device)
        self.bc_block_xyz_by_id = wp.zeros((self.BC_MAX_BLOCKS,), dtype=wp.vec3i, device=self.device)
        self.bc_type = wp.zeros((self.BC_MAX_BLOCKS, B, B, B), dtype=int, device=self.device)
        self.bc_norm = wp.zeros((self.BC_MAX_BLOCKS, B, B, B), dtype=self.vecn, device=self.device)
        self.bc_velo = wp.zeros((self.BC_MAX_BLOCKS, B, B, B), dtype=self.vecn, device=self.device)

    def step(
        self,
        **kwargs,
    ):
        self.ensure_bc_storage()
        self.activate_sparse_grid()

        self.reset_grid()

        if self.solver_type == 'lite_explicit':
            self.lite_explicit_step(**kwargs)

        elif self.solver_type == 'lite_implicit':
            self.lite_implicit_step(**kwargs)

        self.sim_steps += 1
        self.sim_time += self.dt

    # ============================================================================
    # lite MPM explicit solver
    # ============================================================================
    def lite_explicit_step(self, **kwargs):
        lite_p2c(
            self.block_count,
            self.block2bid,
            self.block_xyz_by_id,
            self.ptc_x,
            self.ptc_v,
            self.ptc_k,
            self.ptc_F,
            self.ptc_vol0,
            self.ptc_m,
            self.ptc_G,
            self.ptc_dlogJ,
            self.psi_params,
            self.center_m,
            self.center_v,
            self.center_F,
            self.center_G,
            self.center_vol,
            self.center_tau,
            self.center_dlogJ,
            self.center_size,
            self.dx,
            self.n_ptc,
            self.device,
            reconstruct_center_F=False,
            enable_apic=self.enable_apic,
        )
        lite_c2g(
            self.block_count,
            self.block2bid,
            self.block_xyz_by_id,
            self.bc_block2bid,
            self.bc_type,
            self.bc_norm,
            self.bc_velo,
            self.hf_bc_p,
            self.hf_bc_n,
            self.hf_bc_v,
            self.hf_bc_type,
            self.num_hf,
            self.center_m,
            self.center_v,
            self.center_G,
            self.center_vol,
            self.center_tau,
            self.grid_m,
            self.grid_v,
            self.grid_v_new,
            self.grid_size,
            self.center_size,
            self.gravity,
            self.dx,
            self.dt,
            self.n_psi,
            self.device,
            explicit_force=True,
            enable_apic=self.enable_apic,
        )
        wp.launch(
            kernel=lite_g2c_kernel,
            dim=(self.bcn, B, B, B),
            inputs=[
                self.block_count,
                self.block2bid,
                self.block_xyz_by_id,
                self.grid_v,
                self.grid_v_new,
                self.center_v,
                self.center_dv,
                self.center_G,
                self.center_size,
                self.dx,
            ],
            device=self.device
        )
        wp.launch(
            kernel=lite_c2p_kernel,
            dim=self.n_ptc,
            inputs=[
                self.block2bid,
                self.ptc_x,
                self.ptc_v,
                self.ptc_k,
                self.ptc_F,
                self.ptc_G,
                self.ptc_dlogJ,
                self.center_m,
                self.center_v,
                self.center_dv,
                self.center_G,
                self.psi_params,
                self.center_size,
                self.dx,
                self.dt,
                self.flip_ratio,
            ],
            device=self.device
        )


    # ============================================================================
    # lite MPM implicit solver
    # ============================================================================
    def lite_implicit_iterations(self, **kwargs):
        max_iters = kwargs.get('max_iters', 1)
        print_every = kwargs.get('print_every', 1)
        v_tol = kwargs.get('v_tol', 1e-4) 
        lagged_interval = kwargs.get('lagged_interval', 1)

        n_active_nodes = int(self.n_active_nodes.numpy().reshape(-1)[0])
        n_active_centers = int(self.n_active_centers.numpy().reshape(-1)[0])
        ic(n_active_nodes, n_active_centers)

        for it in range(max_iters):
            self.it = it
            # just like particles, we define quadrature only on active centers
            # in each CG iteration, df is computed by:
            #  dv -> ∇dv -> dF -> dP -> df
            # we do not compute full dPdF, but just use their linear relationship

            # ∇v_c from current iterate
            wp.launch(
                kernel=lite_center_grad_from_grid_velocity_kernel,
                dim=n_active_centers,
                inputs=[
                    self.n_active_centers,
                    self.cdof2bijk,
                    self.block2bid,
                    self.block_xyz_by_id,
                    self.grid_m,
                    self.grid_v_it,
                    self.center_G,
                    self.grid_size,
                    self.dx,
                ],
                device=self.device,
            )
            wp.launch(
                kernel=lite_update_quadrature_scratch_kernel,
                dim=(self.n_psi, n_active_centers),
                inputs=[
                    self.n_active_centers,
                    self.cdof2bijk,
                    self.block_xyz_by_id,
                    self.it % lagged_interval == 0,
                    self.center_quadrature_scratch,
                    self.center_vol,
                    self.center_F,
                    self.center_G,
                    self.center_dlogJ,
                    self.psi_params,
                    self.center_size,
                    self.dt,
                ],
                device=self.device,
            )
            wp.launch(
                kernel=lite_implicit_residual_from_scratch_kernel_fast,
                dim=self.MAX_DOF, # IMPORTANT: make sure zero all intialized
                inputs=[
                    self.n_active_nodes,
                    self.ndof2bijk,
                    self.center2dof,
                    self.block2bid,
                    self.block_xyz_by_id,
                    self.node_residual,
                    self.node_Hii_inv,
                    self.node_tikhonov,
                    self.bc_block2bid,
                    self.bc_type,
                    self.bc_norm,
                    self.bc_velo,
                    self.hf_bc_p,
                    self.hf_bc_n,
                    self.hf_bc_v,
                    self.hf_bc_type,
                    self.num_hf,
                    self.center_vol,
                    self.center_F,
                    self.center_quadrature_scratch,
                    self.grid_m,
                    self.grid_v,
                    self.grid_v_it,
                    self.psi_params,
                    self.grid_size,
                    self.gravity,
                    self.dx,
                    self.dt,
                    self.n_psi,
                    0,
                ],
            device=self.device,
        )

            self.pd_damping = 0.0
            self.pd_damping_arr.fill_(self.pd_damping)

            if (not hasattr(self, "_A_matvec") or self.need_record_again):
                self.update_grad_dv_launch = wp.launch(
                    kernel=lite_implicit_grad_v_dv_kernel,
                    dim=self.MAX_DOF,
                    inputs=[
                        self.n_active_centers,
                        self.cdof2bijk,
                        self.node2dof,
                        self.block2bid,
                        self.block_xyz_by_id,
                        self.dv_guess, # placeholder for dv
                        self.center_grad_dv_tmp,
                        self.grid_size,
                        self.dx,
                    ],
                    device=self.device,
                    record_cmd=True,
                )
                self.update_stress_scratch_launch = wp.launch(
                    kernel=lite_implicit_update_stress_scratch_kernel,
                    dim=(self.n_psi, self.MAX_DOF),
                    inputs=[
                        self.n_active_centers,
                        self.cdof2bijk,
                        self.center_stress_scratch,
                        self.center_quadrature_scratch,
                        self.center_grad_dv_tmp,
                        self.center_vol,
                        self.center_F,
                        self.psi_params,
                        self.dt,
                    ],
                    device=self.device,
                    record_cmd=True,
                )
                self.compute_df_launch = wp.launch(
                    kernel=lite_implicit_df_kernel_fast,
                    dim=self.MAX_DOF,
                    inputs=[
                        self.n_active_nodes,
                        self.ndof2bijk,
                        self.center2dof,
                        self.block2bid,
                        self.block_xyz_by_id,
                        self.dv_guess,
                        self.A_matvec_out,
                        self.pd_damping_arr,
                        self.node_tikhonov,
                        self.bc_block2bid,
                        self.bc_type,
                        self.bc_norm,
                        self.bc_velo,
                        self.hf_bc_p,
                        self.hf_bc_n,
                        self.hf_bc_v,
                        self.hf_bc_type,
                        self.num_hf,
                        self.center_stress_scratch,
                        self.center_vol,
                        self.grid_m,
                        self.grid_size,
                        self.dx,
                        self.dt,
                        self.n_psi,
                    ],
                    device=self.device,
                    record_cmd=True,
                )
                self.matrix_free_M_inv_matvec_launch = wp.launch(
                    kernel=lite_implicit_precond_kernel_Hii,
                    dim=self.MAX_DOF,
                    inputs=[
                        self.n_active_nodes,
                        self.ndof2bijk,
                        self.block_xyz_by_id,
                        self.node_Hii_inv,
                        self.dv_guess,
                        self.A_matvec_out,
                        self.bc_block2bid,
                        self.bc_type,
                        self.bc_norm,
                        self.bc_velo,
                        self.hf_bc_p,
                        self.hf_bc_n,
                        self.hf_bc_v,
                        self.hf_bc_type,
                        self.num_hf,
                        self.dx,
                    ],
                    device=self.device,
                    record_cmd=True,
                )

                def _A_matvec(p, Ap):
                    self.update_grad_dv_launch.set_param_at_index(0, self.n_active_centers)
                    self.update_grad_dv_launch.set_param_at_index(1, self.cdof2bijk)
                    self.update_grad_dv_launch.set_param_at_index(2, self.node2dof)
                    self.update_grad_dv_launch.set_param_at_index(3, self.block2bid)
                    self.update_grad_dv_launch.set_param_at_index(4, self.block_xyz_by_id)
                    self.update_grad_dv_launch.set_param_at_index(5, p)
                    self.update_grad_dv_launch.launch()

                    self.update_stress_scratch_launch.set_param_at_index(0, self.n_active_centers)
                    self.update_stress_scratch_launch.set_param_at_index(1, self.cdof2bijk)
                    self.update_stress_scratch_launch.launch()

                    self.compute_df_launch.set_param_at_index(0, self.n_active_nodes)
                    self.compute_df_launch.set_param_at_index(1, self.ndof2bijk)
                    self.compute_df_launch.set_param_at_index(2, self.center2dof)
                    self.compute_df_launch.set_param_at_index(3, self.block2bid)
                    self.compute_df_launch.set_param_at_index(4, self.block_xyz_by_id)
                    self.compute_df_launch.set_param_at_index(5, p)
                    self.compute_df_launch.set_param_at_index(6, Ap)
                    self.compute_df_launch.set_param_at_index(7, self.pd_damping_arr)
                    self.compute_df_launch.launch()

                def _M_inv_matvec(p, Mp):
                    self.matrix_free_M_inv_matvec_launch.set_param_at_index(0, self.n_active_nodes)
                    self.matrix_free_M_inv_matvec_launch.set_param_at_index(1, self.ndof2bijk)
                    self.matrix_free_M_inv_matvec_launch.set_param_at_index(2, self.block_xyz_by_id)
                    self.matrix_free_M_inv_matvec_launch.set_param_at_index(3, self.node_Hii_inv)
                    self.matrix_free_M_inv_matvec_launch.set_param_at_index(4, p)
                    self.matrix_free_M_inv_matvec_launch.set_param_at_index(5, Mp)
                    self.matrix_free_M_inv_matvec_launch.launch()

                self._A_matvec = _A_matvec
                self._M_inv_matvec = _M_inv_matvec

                self.need_record_again = False

            r0 = self.compute_residual_norm(self.node_residual)
            self.dv_guess.zero_()
            n_cg_iters, error, tol = matrix_free_cg(
                A_matvec=self._A_matvec,
                b=-self.node_residual,
                x=self.dv_guess,
                maxiter=5000,
                tol=0.5,
                atol=1e-9,
                M_inv_matvec=self._M_inv_matvec,
                use_cuda_graph=True,
            )

            alpha = 1.0 # line-search is ommitted for current repo
            if it == 0: self.R0 = r0

            self.max_update.fill_(0.0)
            wp.launch(
                kernel=commit_dv_to_iterate_kernel,
                dim=n_active_nodes,
                inputs=[
                    self.n_active_nodes,
                    self.ndof2bijk,
                    self.block_xyz_by_id,
                    self.bc_block2bid,
                    self.bc_type,
                    self.bc_norm,
                    self.bc_velo,
                    self.hf_bc_p,
                    self.hf_bc_n,
                    self.hf_bc_v,
                    self.hf_bc_type,
                    self.num_hf,
                    self.dv_guess,
                    self.grid_v_it,
                    self.max_update,
                    alpha,
                    self.dx,
                ],
                device=self.device,
            )

            # check convergence
            dv_inf = float(self.max_update.numpy().reshape(-1)[0])  # ||Δv_search||_∞ (pre-clip)
            if alpha * dv_inf < v_tol:
                if print_every > 0 and (self.sim_steps % print_every == 0):   # StiffGIPC patch: quiet by default
                    print(f"\033[32m[Solver] sim_step={self.sim_steps:04d} newton_iters={it:04d} cg_iters={n_cg_iters:04d} R0={self.R0:.3e} r0={r0:.3e} ||Δv_search||_∞={alpha * dv_inf:.3e}\033[0m")
                self.last_newton_iters = it + 1
                break
            elif print_every > 0 and (it % print_every == 0) and (self.sim_steps % print_every == 0):
                print(f"[Solver] sim_step={self.sim_steps:04d} newton_iters={it:04d} cg_iters={n_cg_iters:04d} R0={self.R0:.3e} r0={r0:.3e} ||Δv_search||_∞={alpha * dv_inf:9.3e}")

            self.r_prev = r0

    def lite_implicit_step(self, **kwargs):
        lite_p2c(
            self.block_count,
            self.block2bid,
            self.block_xyz_by_id,
            self.ptc_x,
            self.ptc_v,
            self.ptc_k,
            self.ptc_F,
            self.ptc_vol0,
            self.ptc_m,
            self.ptc_G,
            self.ptc_dlogJ,
            self.psi_params,
            self.center_m,
            self.center_v,
            self.center_F,
            self.center_G,
            self.center_vol,
            self.center_tau,
            self.center_dlogJ,
            self.center_size,
            self.dx,
            self.n_ptc,
            self.device,
            enable_apic=self.enable_apic,
        )

        lite_c2g(
            self.block_count,
            self.block2bid,
            self.block_xyz_by_id,
            self.bc_block2bid,
            self.bc_type,
            self.bc_norm,
            self.bc_velo,
            self.hf_bc_p,
            self.hf_bc_n,
            self.hf_bc_v,
            self.hf_bc_type,
            self.num_hf,
            self.center_m,
            self.center_v,
            self.center_G,
            self.center_vol,
            self.center_tau,
            self.grid_m,
            self.grid_v,
            self.grid_v_new,
            self.grid_size,
            self.center_size,
            self.gravity,
            self.dx,
            self.dt,
            self.n_psi,
            self.device,
            explicit_force=False,
            enable_apic=self.enable_apic,
        )

        self.reduce_sparse_grid()

        n_active_nodes = int(self.n_active_nodes.numpy().reshape(-1)[0])
        n_active_centers = int(self.n_active_centers.numpy().reshape(-1)[0])

        wp.launch(
            kernel=lite_initialize_v_iterate_kernel,
            dim=(n_active_nodes,),
            inputs=[
                self.n_active_nodes,
                self.ndof2bijk,
                self.block_xyz_by_id,
                self.bc_block2bid,
                self.bc_type,
                self.bc_norm,
                self.bc_velo,
                self.hf_bc_p,
                self.hf_bc_n,
                self.hf_bc_v,
                self.hf_bc_type,
                self.num_hf,
                self.grid_m,
                self.grid_v,
                self.grid_v_it,
                self.grid_size,
                self.gravity,
                self.dt,
                self.dx,
            ],
            device=self.device
        )

        self.lite_implicit_iterations(**kwargs)

        damping = kwargs.get('damping', 1.0)
        # copy converged iterate → v_new
        wp.launch(
            kernel=update_vnew_with_iterate_kernel,
            dim=(self.bcn, B, B, B),
            inputs=[
                self.block_count,
                self.block_xyz_by_id,
                self.bc_block2bid,
                self.bc_type,
                self.bc_norm,
                self.bc_velo,
                self.hf_bc_p,
                self.hf_bc_n,
                self.hf_bc_v,
                self.hf_bc_type,
                self.num_hf,
                self.grid_m,
                self.grid_v_it,
                self.grid_v_new,
                self.grid_size,
                self.dx,
                damping,
            ],
            device=self.device,
        )
        wp.launch(
            kernel=lite_g2c_kernel,
            dim=(self.bcn, B, B, B),
            inputs=[
                self.block_count,
                self.block2bid,
                self.block_xyz_by_id,
                self.grid_v,
                self.grid_v_new,
                self.center_v,
                self.center_dv,
                self.center_G,
                self.center_size,
                self.dx,
            ],
            device=self.device
        )
        wp.launch(
            kernel=lite_c2p_kernel,
            dim=self.n_ptc,
            inputs=[
                self.block2bid,
                self.ptc_x,
                self.ptc_v,
                self.ptc_k,
                self.ptc_F,
                self.ptc_G,
                self.ptc_dlogJ,
                self.center_m,
                self.center_v,
                self.center_dv,
                self.center_G,
                self.psi_params,
                self.center_size,
                self.dx,
                self.dt,
                self.flip_ratio,
            ],
            device=self.device
        )

    def compute_residual_norm(self, residual):
        if not hasattr(self, 'residual_tiled_dot') or self.residual_tiled_dot is None or (
            self.residual_tiled_dot.max_length != self.MAX_DOF
        ):
            self.residual_tiled_dot = VecTiledDot(self.MAX_DOF, vec_dtype=self.vecn, scalar_type=real, tile_size=512, device=self.device, max_column_count=2)
        self.residual_tiled_dot.compute(residual, residual)
        return self.residual_tiled_dot.col(0).numpy().reshape(-1)[0]**0.5

    def add_material(
        self,
        material_k,
        material,
        **kwargs,
    ):
        assert self.mat_initialized[material_k] == False
        self.mat_initialized[material_k] = True

        material_params = MaterialParams()
        material_params.mate_type = material

        mu_lam_kappa = vec3(real(0.0))
        if material == Material.water:
            material_params.cons_type = kwargs.get('constitutive', Constitutive.pressure)
            kappa = kwargs.get("kappa", 1e5)
            mu_lam_kappa[2] = kappa
        else:
            material_params.cons_type = kwargs.get('constitutive', Constitutive.stvk)
            E = kwargs.get("E", 1e6)
            nu = kwargs.get("nu", 0.3)
            mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu)) # Lame parameters
            mu_lam_kappa[0] = mu
            mu_lam_kappa[1] = lam
            if material == Material.sand:
                friction_angle = kwargs.get('friction_angle', 35.0)
                cohesion = kwargs.get('cohesion', 0.0)
                material_params.friction_angle = friction_angle
                mu_lam_kappa[2] = cohesion
            elif material == Material.vonmises:
                yield_stress = kwargs.get('yield_stress', 5000.0)
                material_params.yield_stress = yield_stress
            elif material == Material.nacc:
                M = kwargs.get('M', 1.85)
                beta = kwargs.get('beta', 0.05)
                xi = kwargs.get('xi', 30)
                mu_lam_kappa[2] = M
                material_params.friction_angle = beta
                material_params.yield_stress = xi
            elif material == Material.snow13:
                hardening = kwargs.get('hardening', 10.0)
                theta_c = kwargs.get('theta_c', 2.5e-2)
                theta_s = kwargs.get('theta_s', 4.5e-3)
                mu_lam_kappa[2] = hardening
                material_params.friction_angle = theta_c
                material_params.yield_stress = theta_s
            elif material == Material.foam:
                yield_stress = kwargs.get('yield_stress', 1e5)
                plastic_pow = kwargs.get('plastic_pow', 1.0)
                plastic_viscosity = kwargs.get('plastic_viscosity', 1.0)
                mu_lam_kappa[2] = plastic_pow
                material_params.yield_stress = yield_stress
                material_params.friction_angle = plastic_viscosity
        material_params.mu_lam_kappa = mu_lam_kappa
        
        wp.launch(
            kernel=set_material_params,
            dim=1,
            inputs=[self.psi_params, material_k, material_params],
            device=self.device,
        )

    def seed_particles(
        self,
        x,
        use_material_k,
        density,
        vol0,
        velocity=None,
        color=(1,1,1),
        **kwargs,
    ):
        print('seeding particles...')
        n_particles = len(x)
        v = np.array([[0.0] * self.dim]) if velocity is None else np.array(velocity).reshape(1,self.dim)
        v = np.tile(v, (n_particles, 1))
        vol0 = np.ones((n_particles,), dtype=np.float64) * vol0
        m = vol0.copy() * density
        k = np.ones((n_particles,), dtype=np.int32) * use_material_k
        color = np.tile(np.array(color), (n_particles, 1))
        if color.max() > 1: color = color/255.0

        self.ptc_color = np.concatenate([self.ptc_color, color], axis=0)
        self.ptc_x = wp.from_numpy(np.concatenate([self.ptc_x.numpy(), x], axis=0), dtype=self.vecn, device=self.device)
        self.ptc_v = wp.from_numpy(np.concatenate([self.ptc_v.numpy(), v], axis=0), dtype=self.vecn, device=self.device)
        self.ptc_m = wp.from_numpy(np.concatenate([self.ptc_m.numpy(), m], axis=0), dtype=real, device=self.device)
        self.ptc_k = wp.from_numpy(np.concatenate([self.ptc_k.numpy(), k], axis=0), dtype=wp.int32, device=self.device)
        self.ptc_vol0 = wp.from_numpy(np.concatenate([self.ptc_vol0.numpy(), vol0], axis=0), dtype=real, device=self.device)
        self.ptc_F = wp.from_numpy(np.concatenate([self.ptc_F.numpy(),
                                                   np.tile(np.eye(self.dim)*1.0, (n_particles, 1, 1))],
                                                   axis=0), dtype=self.matnn, device=self.device)
        self.ptc_G = wp.from_numpy(np.concatenate([self.ptc_G.numpy(),
                                                        np.zeros((n_particles, self.dim, self.dim))],
                                                        axis=0), dtype=self.matnn, device=self.device)
        self.ptc_dlogJ = wp.from_numpy(np.concatenate([self.ptc_dlogJ.numpy(),
                                                       np.zeros((n_particles,))],
                                                       axis=0), dtype=real, device=self.device)

        self.n_ptc += len(x)
        self.n_ptc_wp.fill_(self.n_ptc)
        ic(self.n_ptc, "Total particles after seeding.")

        if self.solver_type in ['implicit']:
            while self.n_ptc >= self.MAX_PTS:
                self.MAX_PTS *= 2
                ic(self.MAX_PTS, "Reallocating particle storage...")
                self.init_pts_storage()
            self.need_record_again = True

    def add_cube(
        self,
        lower_corner,
        cube_size,
        use_material_k,
        density,
        velocity=None,
        color=(1,1,1),
        **kwargs,
    ):
        cell_vol = self.dx ** self.dim
        sample_vol = np.prod(np.array(cube_size))
        should_sample_n_particles = int(self.ppc * sample_vol / cell_vol)

        if not hasattr(self, 'geom_cube'):
            self.geom_cube = pv.read("assets/geom_cube.vtk")
        
        pts = self.geom_cube.points * (np.array(cube_size) / 2.0)[None, :] + (np.array(lower_corner) + np.array(cube_size) / 2.0)[None, :]
        tets = self.geom_cube.cells.reshape(-1, 5)[:, 1:5]
        x = sample_vtk(pts, tets, should_sample_n_particles)
        vol0 = sample_vol / should_sample_n_particles

        self.seed_particles(x, use_material_k, density, vol0, velocity, color, **kwargs)

    def add_cylinder(
        self,
        center,
        radius,
        axis,
        axis_length,
        use_material_k,
        density,
        velocity=None,
        color=(1,1,1),
        **kwargs,
    ):
        cell_vol = self.dx ** self.dim
        sample_vol = np.pi * (radius ** 2) * axis_length
        should_sample_n_particles = int(self.ppc * sample_vol / cell_vol)

        if not hasattr(self, 'geom_cylinder'):
            self.geom_cylinder = pv.read("assets/geom_cylinder.vtk")

        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        
        z = np.array([0.0, 0.0, 1.0])
        if np.dot(z, axis) < 0: axis = -axis
        v, c = np.cross(z, axis), np.dot(z, axis)
        R = np.eye(3)
        s = np.linalg.norm(v)
        if s >= 1e-12:
            vx = np.array([
                [ 0,   -v[2],  v[1]],
                [ v[2], 0,   -v[0]],
                [-v[1], v[0],  0  ],
            ])
            R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))

        S = np.diag([radius, radius, axis_length / 2.0])
        pts = self.geom_cylinder.points @ S.T @ R.T + np.asarray(center)
        tets = self.geom_cylinder.cells.reshape(-1, 5)[:, 1:5]
        x = sample_vtk(pts, tets, should_sample_n_particles)
        vol0 = sample_vol / should_sample_n_particles

        self.seed_particles(x, use_material_k, density, vol0, velocity, color, **kwargs)

    def add_sphere(
        self,
        center,
        radius,
        use_material_k,
        density,
        velocity=None,
        color=(1,1,1),
        **kwargs,
    ):
        cell_vol = self.dx ** self.dim
        sample_vol = 4/3 * np.pi * (radius ** 3)
        should_sample_n_particles = int(self.ppc * sample_vol / cell_vol)

        if not hasattr(self, 'geom_sphere'):
            self.geom_sphere = pv.read("assets/geom_sphere.vtk")
        
        S = np.diag([radius, radius, radius])
        pts = self.geom_sphere.points @ S.T + np.asarray(center)
        tets = self.geom_sphere.cells.reshape(-1, 5)[:, 1:5]
        x = sample_vtk(pts, tets, should_sample_n_particles)
        vol0 = sample_vol / should_sample_n_particles

        self.seed_particles(x, use_material_k, density, vol0, velocity, color, **kwargs)
    
    def add_mesh(
        self,
        surface_mesh,
        offset,
        use_material_k,
        density,
        velocity=None,
        color=(1, 1, 1),
        scale=1.0,
        ppc_scale=1.0,
        tetra_method="auto",
        delaunay_alpha=None,
        enforce_closed=True,
        **kwargs,
    ):
        cell_vol = self.dx ** self.dim

        if isinstance(surface_mesh, str):
            mesh = pv.read(surface_mesh)
        else:
            mesh = pv.wrap(surface_mesh)

        tet = None

        if isinstance(mesh, pv.UnstructuredGrid) and mesh.n_cells > 0:
            try:
                if hasattr(pv, "CellType"):
                    is_tet = np.all(mesh.celltypes == pv.CellType.TETRA)
                else:
                    is_tet = np.all(np.asarray(mesh.celltypes) == 10)
                if is_tet:
                    tet = mesh
            except Exception:
                tet = None

        if tet is None:
            if not isinstance(mesh, pv.PolyData):
                mesh = mesh.extract_surface()
            surf = mesh.triangulate().clean()

            if enforce_closed and getattr(surf, "n_open_edges", 0) > 0:
                raise ValueError("add_mesh requires a closed (watertight) surface mesh (n_open_edges == 0).")

            # prefer tetgen-based tetrahedralize when available
            if tetra_method in ("auto", "tetgen") and hasattr(surf, "tetrahedralize"):
                try:
                    tet = surf.tetrahedralize()
                except Exception:
                    if tetra_method == "tetgen":
                        raise
                    tet = None

            # fallback: delaunay_3d + (optional) keep only cells inside surface
            if tet is None:
                alpha = 0.0 if delaunay_alpha is None else float(delaunay_alpha)
                tet_raw = surf.delaunay_3d(alpha=alpha)

                # keep only tetra cells whose centers are inside the surface (best-effort)
                try:
                    centers = tet_raw.cell_centers()
                    sel = centers.select_enclosed_points(surf)
                    inside = np.asarray(sel.point_data["SelectedPoints"]).astype(bool)
                    tet = tet_raw.extract_cells(inside)
                except Exception:
                    tet = tet_raw

        try:
            if hasattr(pv, "CellType"):
                tet = tet.extract_cells(tet.celltypes == pv.CellType.TETRA)
            else:
                tet = tet.extract_cells(np.asarray(tet.celltypes) == 10)
        except Exception:
            pass

        if tet.n_cells == 0:
            raise ValueError("Failed to build a tetrahedral volume mesh from the given surface.")

        try:
            sample_vol = float(tet.volume)
        except Exception:
            sizes = tet.compute_cell_sizes(length=False, area=False, volume=True)
            sample_vol = float(np.asarray(sizes.cell_data["Volume"]).sum())

        should_sample_n_particles = max(1, int(self.ppc * ppc_scale * sample_vol / cell_vol))

        pts = np.asarray(tet.points) * scale + np.asarray(offset)
        cells = np.asarray(tet.cells)

        if cells.size % 5 == 0:
            tets = cells.reshape(-1, 5)[:, 1:5]
        else:
            # fallback: use cells_dict if available
            if hasattr(tet, "cells_dict"):
                tet_key = pv.CellType.TETRA if hasattr(pv, "CellType") else 10
                tets = np.asarray(tet.cells_dict[tet_key], dtype=np.int64)
            else:
                raise ValueError("Unexpected cell layout; mesh must be tetrahedral.")

        x = sample_vtk(pts, tets, should_sample_n_particles)
        vol0 = sample_vol / should_sample_n_particles

        self.seed_particles(x, use_material_k, density, vol0, velocity, color, **kwargs)
        
    def add_point_cloud(
        self,
        points,
        tot_vol,
        use_material_k,
        density,
        velocity=None,
        color=(1,1,1),
        **kwargs,
    ):
        cell_vol = self.dx ** self.dim
        sample_vol = tot_vol
        should_sample_n_particles = int(self.ppc * sample_vol / cell_vol)
        if len(points) > 1.05 * should_sample_n_particles or len(points) < 0.95 * should_sample_n_particles:
            print(
                f"\033[33m[CFL WARNING] The provided point cloud has {len(points)} points, "
                f"which is different from the expected {should_sample_n_particles} points "
                f"based on ppc and volume.\033[0m"
            )
        
        vol0 = tot_vol / len(points)
        x = points

        self.seed_particles(x, use_material_k, density, vol0, velocity, color, **kwargs)

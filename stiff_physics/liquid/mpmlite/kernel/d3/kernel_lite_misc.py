from typing import Any
import warp as wp
from ...types import *
from ..constant import (
    _0, _1, _05, _n1, _2, _14, _13,
    s_min, dPdF_use_project, dPdF_eps_small, dPdF_eps_pd,
)
from .helper_misc import (
    in_region,
)
from .helper_constitutive_model import (
    Drucker_Prager_return_mapping,
    NonAssoc_CamClay_return_mapping,
    von_mises_apply,
    dPdF_StVK_Hencky_3D_analytic_v2,
    water_PK1_3D,
    StVK_Hencky_PK1_3D_svdfree,
    snow13_return_mapping, snow13_hardening,
    foam_return_mapping,
)
from ...sp_grid import (
    B, BLOCK_VOL,
    lin_JK, lin_I_JK, lin_IJK, unlin_JK, unlin_I_JK, unlin_IJK,
    block_coords_from_node, local_coords_in_block, local_linear, local2global,
)
from ...boundary_utils import (
    proj_boundary_vel,
    query_bc_sp,
)

@wp.kernel
def lite_initialize_v_iterate_kernel(
    n_active_nodes: wp.array(dtype=int, ndim=1),
    ndof2bijk: wp.array(dtype=wp.vec2i, ndim=1),
    block_xyz_by_id: wp.array(dtype=wp.vec3i, ndim=1),

    bc_block2bid: wp.array(dtype=int, ndim=3),
    bc_type: wp.array(dtype=int, ndim=4),
    bc_norm: wp.array(dtype=vec3, ndim=4),
    bc_velo: wp.array(dtype=vec3, ndim=4),

    hf_bc_p: wp.array(dtype=vec3),
    hf_bc_n: wp.array(dtype=vec3),
    hf_bc_v: wp.array(dtype=vec3),
    hf_bc_type: wp.array(dtype=int),
    num_hf: wp.int32,

    grid_m: wp.array(dtype=real, ndim=4),
    grid_v: wp.array(dtype=vec3, ndim=4),
    grid_v_it: wp.array(dtype=vec3, ndim=4),

    grid_size: wp.vec3i,
    gravity: vec3,
    dt: real,
    dx: real,
):
    dof = wp.tid()
    if dof >= n_active_nodes[0]: return
    bid, local1d = ndof2bijk[dof].x, ndof2bijk[dof].y
    node = local2global(block_xyz_by_id[bid], local1d)
    if not in_region(node, grid_size): return
    lijk = local_coords_in_block(node.x, node.y, node.z)
    li, lj, lk = lijk.x, lijk.y, lijk.z

    if grid_m[bid, li, lj, lk] > _0:
        bct, bcn, bcv = query_bc_sp(node, bc_block2bid, bc_type, bc_norm, bc_velo, hf_bc_p, hf_bc_n, hf_bc_v, hf_bc_type, num_hf, dx)
        v = grid_v[bid, li, lj, lk] + dt * gravity
        if bct > 0:
            v = proj_boundary_vel(v, bct, bcn, bcv)
        grid_v_it[bid, li, lj, lk] = v
    else:
        grid_v_it[bid, li, lj, lk] = vec3(_0)


@wp.kernel
def lite_reduce_active_nodes_and_centers_kernel(
    block_count: wp.array(dtype=int, ndim=1),
    block_xyz_by_id: wp.array(dtype=wp.vec3i, ndim=1),

    grid_m: wp.array(dtype=real, ndim=4),
    n_active_nodes: wp.array(dtype=int),
    ndof2bijk: wp.array(dtype=wp.vec2i),
    node2dof: wp.array(dtype=int, ndim=4),

    center_m: wp.array(dtype=real, ndim=4),
    n_active_centers: wp.array(dtype=int),
    cdof2bijk: wp.array(dtype=wp.vec2i),
    center2dof: wp.array(dtype=int, ndim=4),

    grid_size: wp.vec3i,
    overflow: wp.array(dtype=int, ndim=1),
    n_max_dofs: int,

    check_centers: bool = True,
):
    bid, li, lj, lk = wp.tid()
    if bid >= block_count[0]: return
    ijk = local2global(block_xyz_by_id[bid], lin_IJK(li, lj, lk))

    if in_region(ijk, grid_size):
        if grid_m[bid, li, lj, lk] > _0:
            cur_ndof = wp.atomic_add(n_active_nodes, 0, 1)
            if cur_ndof >= n_max_dofs:
                overflow[0] = 1
                return
            else:
                ndof2bijk[cur_ndof] = wp.vec2i(bid, lin_IJK(li, lj, lk))
                node2dof[bid, li, lj, lk] = wp.int32(cur_ndof)
        else:
            node2dof[bid, li, lj, lk] = -1

    center_size = wp.vec3i(grid_size[0]-1, grid_size[1]-1, grid_size[2]-1)
    if check_centers and in_region(ijk, center_size):
        if center_m[bid, li, lj, lk] > _0:
            cur_cdof = wp.atomic_add(n_active_centers, 0, 1)
            if cur_cdof >= n_max_dofs:
                overflow[0] = 1
                return
            else:
                cdof2bijk[cur_cdof] = wp.vec2i(bid, lin_IJK(li, lj, lk))
                center2dof[bid, li, lj, lk] = cur_cdof
        else:
            center2dof[bid, li, lj, lk] = -1


@wp.kernel
def lite_center_grad_from_grid_velocity_kernel(
    n_active_centers: wp.array(dtype=int, ndim=1),
    cdof2bijk: wp.array(dtype=wp.vec2i, ndim=1),
    block2bid: wp.array(dtype=int, ndim=3),
    block_xyz_by_id: wp.array(dtype=wp.vec3i, ndim=1),
    grid_m: wp.array(dtype=real, ndim=4),
    grid_v_it: wp.array(dtype=vec3, ndim=4),
    center_G: wp.array(dtype=mat33, ndim=4),
    grid_size: wp.vec3i,
    dx: real,
):
    cdof = wp.tid()
    if cdof >= n_active_centers[0]: return
    cbid, local_c1d = cdof2bijk[cdof].x, cdof2bijk[cdof].y
    center = local2global(block_xyz_by_id[cbid], local_c1d)
    center_size = wp.vec3i(grid_size[0]-1, grid_size[1]-1, grid_size[2]-1)
    if not in_region(center, center_size): return

    ci, cj, ck = center.x, center.y, center.z
    lci, lcj, lck = unlin_IJK(local_c1d) 

    G = mat33(_0)
    inv_4dx = _14 / dx
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                ni, nj, nk = ci + di, cj + dj, ck + dk
                node = wp.vec3i(ni, nj, nk)
                if not in_region(node, grid_size): continue

                bcoords = block_coords_from_node(ni, nj, nk)
                bid = block2bid[bcoords.x, bcoords.y, bcoords.z]
                if bid < 0: continue
                local_n1d = local_linear(local_coords_in_block(ni, nj, nk))
                li, lj, lk = unlin_IJK(local_n1d)
                if grid_m[bid, li, lj, lk] <= _0: continue
                sx = _n1 if di == 0 else _1
                sy = _n1 if dj == 0 else _1
                sz = _n1 if dk == 0 else _1
                grad_w = inv_4dx * vec3(sx, sy, sz)
                G += wp.outer(grid_v_it[bid, li, lj, lk], grad_w)
    center_G[cbid, lci, lcj, lck] = G


@wp.kernel
def lite_update_quadrature_scratch_kernel(
    n_active_centers: wp.array(dtype=int, ndim=1),
    cdof2bijk: wp.array(dtype=wp.vec2i, ndim=1),
    block_xyz_by_id: wp.array(dtype=wp.vec3i, ndim=1),
    do_return_mapping: bool,
    quadrature_scratch: wp.array(dtype=QuadratureScratch3D, ndim=2),
    center_vol: wp.array(dtype=real, ndim=4),
    center_F: wp.array(dtype=mat33, ndim=4),
    center_G: wp.array(dtype=mat33, ndim=4),
    center_dlogJ: wp.array(dtype=real, ndim=4),
    psi_params: wp.array(dtype=MaterialParams),
    center_size: wp.vec3i,
    dt: real,
):
    psi_k, cdof = wp.tid()
    if cdof >= n_active_centers[0]: return
    cbid, local_c1d = cdof2bijk[cdof].x, cdof2bijk[cdof].y
    center = local2global(block_xyz_by_id[cbid], local_c1d)
    if not in_region(center, center_size): return

    lci, lcjk = unlin_I_JK(local_c1d)
    lcj, lck = unlin_JK(lcjk)

    if center_vol[psi_k, cbid, lci, lcjk] <= _0:
        return

    Gv = center_G[cbid, lci, lcj, lck]
    Fref = center_F[psi_k, cbid, lci, lcjk]

    pp = psi_params[psi_k]
    mu_lam_kappa = pp.mu_lam_kappa
    mu, lam = mu_lam_kappa[0], mu_lam_kappa[1]
    qs = QuadratureScratch3D()
    if pp.cons_type == Constitutive.stvk:
        F_tr = (wp.identity(3, dtype=real) + dt * Gv) @ Fref  # trial F

        if do_return_mapping:
            if pp.mate_type == Material.sand:
                # kappa for cohesion
                U, Sigma, V = Drucker_Prager_return_mapping(F_tr, mu_lam_kappa[2], mu, lam, pp.friction_angle)
                qs.U = U; qs.V = V; qs.Sigma = Sigma
            elif pp.mate_type == Material.vonmises:
                U, Sigma, V = von_mises_apply(F_tr, mu, pp.yield_stress, s_min)
                qs.U = U; qs.V = V; qs.Sigma = Sigma
            elif pp.mate_type == Material.elastic:
                U, Sigma, V = wp.svd3(F_tr)
                Sigma = vec3(
                    wp.max(Sigma.x, s_min),
                    wp.max(Sigma.y, s_min),
                    wp.max(Sigma.z, s_min),
                )
                qs.U = U; qs.V = V; qs.Sigma = Sigma
            elif pp.mate_type == Material.nacc:
                dlogJ = center_dlogJ[psi_k, cbid, lci, lcjk]
                # kappa for M, friction_angle for beta, yield_stress for xi
                U, Sigma, V, _ = NonAssoc_CamClay_return_mapping(F_tr, mu, lam, dlogJ, mu_lam_kappa[2], pp.friction_angle, pp.yield_stress)
                qs.U = U; qs.V = V; qs.Sigma = Sigma
            elif pp.mate_type == Material.snow13:
                theta_c = pp.friction_angle
                theta_s = pp.yield_stress
                hardening = mu_lam_kappa[2]
                h = snow13_hardening(dlogJ=center_dlogJ[psi_k, cbid, lci, lcjk], hardening=hardening)
                mu *= h; lam *= h
                U, Sigma, V = snow13_return_mapping(F_tr, theta_c, theta_s)
                qs.U = U; qs.V = V; qs.Sigma = Sigma
            elif pp.mate_type == Material.foam:
                yield_stress = pp.yield_stress
                plastic_pow = mu_lam_kappa[2]
                plastic_viscosity = pp.friction_angle
                U, Sigma, V = foam_return_mapping(F_tr, mu, yield_stress, plastic_pow, plastic_viscosity, dt)
                qs.U = U; qs.V = V; qs.Sigma = Sigma
        else:
            U, Sigma, V = wp.svd3(F_tr)
            qs.U = U; qs.V = V; qs.Sigma = Sigma
        
        Aij, B012 = dPdF_StVK_Hencky_3D_analytic_v2(
            qs.Sigma,
            mu,
            lam,
            dPdF_use_project,
            s_min,
            dPdF_eps_small,
            dPdF_eps_pd
        )
        qs.Aij = Aij
        qs.B012 = B012

        qs.tau = StVK_Hencky_PK1_3D_svdfree(qs.U, qs.Sigma, qs.V, mu, lam, s_min) @ wp.transpose(Fref)

    elif pp.cons_type == Constitutive.pressure:
        Jbase = Fref[0,0]*Fref[1,1]*Fref[2,2]
        Jt = (_1 + dt * wp.trace(Gv)) * Jbase
        axis_stretch = wp.pow(Jt, _13)

        # F_tr = (wp.identity(3, dtype=real) + dt * Gv) @ Fref  # trial F
        qs.U = wp.identity(3, dtype=real)
        qs.V = wp.identity(3, dtype=real)
        qs.Sigma = vec3(axis_stretch)

        qs.tau = water_PK1_3D(Jt, mu_lam_kappa[2]) * Jbase * wp.identity(3, dtype=real)
        
    quadrature_scratch[psi_k, cdof] = qs
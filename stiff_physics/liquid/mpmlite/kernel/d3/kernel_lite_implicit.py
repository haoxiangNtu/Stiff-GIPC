import warp as wp
from ...types import *

from ..constant import (
    _0, _1, _05, _2, _3, _4, _n1, _13, _14,
    s_min, dPdF_use_project, dPdF_eps_small, dPdF_eps_pd,
)
from .helper_misc import (
    in_region,
    is_bc,
)
from .helper_constitutive_model import (
    StVK_Hencky_PK1_3D,
    StVK_Hencky_dPK1_3D,
    water_PK1_3D,
)
from ...sp_grid import (
    B, BLOCK_VOL,
    lin_JK, lin_I_JK, lin_IJK, unlin_JK, unlin_IJK, unlin_I_JK,
    local_coords_in_block, local_linear, local2global, block_coords_from_node,
)
from ...boundary_utils import (
    query_bc_sp,
    proj_boundary_vel,
    proj_boundary_delta_vel,
)

@wp.kernel
def lite_implicit_residual_from_scratch_kernel_fast(
    n_active_nodes: wp.array(dtype=int, ndim=1),
    ndof2bijk: wp.array(dtype=wp.vec2i, ndim=1),
    center2dof: wp.array(dtype=int, ndim=4),
    block2bid: wp.array(dtype=int, ndim=3),
    block_xyz_by_id: wp.array(dtype=wp.vec3i, ndim=1),
    node_residual: wp.array(dtype=vec3, ndim=1),
    node_Hii_inv: wp.array(dtype=mat33, ndim=1),
    node_tikhonov: wp.array(dtype=real, ndim=1),

    bc_block2bid: wp.array(dtype=int, ndim=3),
    bc_type: wp.array(dtype=int, ndim=4),
    bc_norm: wp.array(dtype=vec3, ndim=4),
    bc_velo: wp.array(dtype=vec3, ndim=4),

    hf_bc_p: wp.array(dtype=vec3),
    hf_bc_n: wp.array(dtype=vec3),
    hf_bc_v: wp.array(dtype=vec3),
    hf_bc_type: wp.array(dtype=int),
    num_hf: wp.int32,

    center_vol   : wp.array(dtype=real,  ndim=4),
    center_F     : wp.array(dtype=mat33, ndim=4),
    quadrature_scratch: wp.array(dtype=QuadratureScratch3D, ndim=2),

    grid_m       : wp.array(dtype=real,  ndim=4),
    grid_v       : wp.array(dtype=vec3,  ndim=4),
    grid_v_it    : wp.array(dtype=vec3,  ndim=4),

    psi_params   : wp.array(dtype=MaterialParams),
    grid_size    : wp.vec3i,
    gravity      : vec3,
    dx           : real,
    dt           : real,
    n_psi        : int,
    tikhonov     : real,
):
    dof = wp.tid()
    node_residual[dof] = vec3(_0)
    if dof >= n_active_nodes[0]:
        node_residual[dof] = vec3(_0)
        node_Hii_inv[dof] = mat33(_0)
        node_tikhonov[dof] = _0
        return
    bid, local_n1d = ndof2bijk[dof].x, ndof2bijk[dof].y
    node = local2global(block_xyz_by_id[bid], local_n1d)
    if not in_region(node, grid_size):
        node_residual[dof] = vec3(_0)
        node_Hii_inv[dof] = mat33(_0)
        node_tikhonov[dof] = _0
        return
    ni, nj, nk = node.x, node.y, node.z
    li, lj, lk = unlin_IJK(local_n1d)

    bct, bcn, bcv = query_bc_sp(node, bc_block2bid, bc_type, bc_norm, bc_velo, hf_bc_p, hf_bc_n, hf_bc_v, hf_bc_type, num_hf, dx)

    # sticky bc
    if bct == 1:
        node_residual[dof] = vec3(_0)
        node_Hii_inv[dof] = mat33(_0)
        node_tikhonov[dof] = _0
        return

    mi = grid_m[bid, li, lj, lk]

    # inertia term
    node_residual[dof] = mi * (grid_v_it[bid, li, lj, lk] - grid_v[bid, li, lj, lk])
    node_residual[dof] += (-dt * mi) * gravity   # gravity already negative

    center_size = wp.vec3i(grid_size[0]-1, grid_size[1]-1, grid_size[2]-1)
    inv_4dx = _14 / dx

    Hii = mi * wp.identity(3, real)

    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                ci = ni - di; cj = nj - dj; ck = nk - dk
                center = wp.vec3i(ci, cj, ck)
                if not in_region(center, center_size): continue

                bcoords = block_coords_from_node(ci, cj, ck)
                cbid = block2bid[bcoords.x, bcoords.y, bcoords.z]
                if cbid < 0: continue

                local_c1d = local_linear(local_coords_in_block(ci, cj, ck))
                lci, lcj, lck = unlin_IJK(local_c1d)
                lcjk = lin_JK(lcj, lck)
                cdof = center2dof[cbid, lci, lcj, lck]
                if cdof < 0: continue

                sx_ic = _n1 if di == 0 else _1
                sy_ic = _n1 if dj == 0 else _1
                sz_ic = _n1 if dk == 0 else _1
                dw_ic = inv_4dx * vec3(sx_ic, sy_ic, sz_ic) # ∇w_ic

                for psi_k in range(n_psi):
                    V0 = center_vol[psi_k, cbid, lci, lcjk]
                    if V0 <= _0: continue

                    qs = quadrature_scratch[psi_k, cdof]
                    node_residual[dof] += dt * V0 * (qs.tau @ dw_ic)

                    Fref = center_F[psi_k, cbid, lci, lcjk]
                    pp = psi_params[psi_k]

                    col0 = vec3(_0); col1 = vec3(_0); col2 = vec3(_0)
                    for k in range(3):
                        ek = vec3(_1 if k == 0 else _0,
                                  _1 if k == 1 else _0,
                                  _1 if k == 2 else _0)

                        if pp.cons_type == Constitutive.stvk:
                            dF = dt * (wp.outer(ek, dw_ic) @ Fref)
                            dP = StVK_Hencky_dPK1_3D(
                                dF,
                                qs.U,
                                qs.V,
                                qs.Aij,
                                qs.B012,
                            )
                            dTau = dP @ wp.transpose(Fref)
                            col  = V0 * (dTau @ dw_ic)
                        elif pp.cons_type == Constitutive.pressure:
                            Jbase = Fref[0,0]*Fref[1,1]*Fref[2,2]
                            dpdJ = pp.mu_lam_kappa[2] # kappa
                            dJ = dt * Jbase * wp.dot(ek, dw_ic)
                            dp = dpdJ * dJ # scalar pressure
                            dscalar = Jbase * dp
                            col = V0 * dscalar * dw_ic

                        if k == 0:   col0 = col
                        elif k == 1: col1 = col
                        else:        col2 = col
                    # accumulate H_ii (scaled by dt again → dt²)
                    Hii += wp.matrix_from_cols(dt * col0, dt * col1, dt * col2)
    
    # tiny Tikhonov
    eps = tikhonov * (Hii[0,0] + Hii[1,1] + Hii[2,2] + _1)
    Hii += eps * wp.identity(3, real)
    node_Hii_inv[dof] = wp.inverse(Hii)
    node_tikhonov[dof] = eps

    if bct > 0:  # boundary projection
        node_residual[dof] = proj_boundary_delta_vel(node_residual[dof], bct, bcn)


@wp.kernel
def lite_implicit_grad_v_dv_kernel(
    n_active_centers: wp.array(dtype=int, ndim=1),
    cdof2bijk: wp.array(dtype=wp.vec2i, ndim=1),
    node2dof: wp.array(dtype=int, ndim=4),
    block2bid: wp.array(dtype=int, ndim=3),
    block_xyz_by_id: wp.array(dtype=wp.vec3i, ndim=1),
    dv: wp.array(dtype=vec3, ndim=1),
    center_grad_dv: wp.array(dtype=mat33, ndim=1),
    grid_size: wp.vec3i,
    dx: real,
):
    # ∇v_c = Σ_a v_a ⊗ ∇w_a|_c (3X3 corner nodes)
    cdof = wp.tid()
    if cdof >= n_active_centers[0]:
        return
    cdof_bid, local_c1d = cdof2bijk[cdof].x, cdof2bijk[cdof].y
    cijk = local2global(block_xyz_by_id[cdof_bid], local_c1d)
    ci, cj, ck = cijk.x, cijk.y, cijk.z

    inv_4dx = _14 / dx

    G = mat33(_0)
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                node = wp.vec3i(ci + di, cj + dj, ck + dk)
                if not in_region(node, grid_size): continue

                bcoords = block_coords_from_node(node.x, node.y, node.z)
                bid = block2bid[bcoords.x, bcoords.y, bcoords.z]
                if bid < 0: continue
                local_n1d = local_linear(local_coords_in_block(node.x, node.y, node.z))
                li, lj, lk = unlin_IJK(local_n1d)

                dof = node2dof[bid, li, lj, lk]
                if dof < 0: continue

                sx_ic = _n1 if di == 0 else _1
                sy_ic = _n1 if dj == 0 else _1
                sz_ic = _n1 if dk == 0 else _1
                dw_ic = inv_4dx * vec3(sx_ic, sy_ic, sz_ic) # ∇w_ic

                G += wp.outer(dv[dof], dw_ic)
    center_grad_dv[cdof] = G


@wp.kernel
def lite_implicit_update_stress_scratch_kernel(
    n_active_centers: wp.array(dtype=int, ndim=1),
    cdof2bijk: wp.array(dtype=wp.vec2i, ndim=1),

    stress_scratch: wp.array(dtype=mat33, ndim=2),
    quadrature_scratch: wp.array(dtype=QuadratureScratch3D, ndim=2),
    center_grad_dv: wp.array(dtype=mat33, ndim=1),
    center_vol: wp.array(dtype=real, ndim=4),
    center_F: wp.array(dtype=mat33, ndim=4),
    psi_params: wp.array(dtype=MaterialParams),
    dt: real,
):
    psi_k, cdof = wp.tid()
    if cdof >= n_active_centers[0]:
        return
    cbid, local_c1d = cdof2bijk[cdof].x, cdof2bijk[cdof].y
    lci, lcjk = unlin_I_JK(local_c1d)

    V0 = center_vol[psi_k, cbid, lci, lcjk]
    if V0 <= _0: return

    Fref = center_F[psi_k, cbid, lci, lcjk]
    pp = psi_params[psi_k]

    if pp.cons_type == Constitutive.stvk:
        qs = quadrature_scratch[psi_k, cdof]
        dF = dt * (center_grad_dv[cdof] @ Fref)

        dP = StVK_Hencky_dPK1_3D(
            dF,
            qs.U,
            qs.V,
            qs.Aij,
            qs.B012,
        )
        stress_scratch[psi_k, cdof] = V0 * (dP @ wp.transpose(Fref))

    elif pp.cons_type == Constitutive.pressure:
        Jbase = Fref[0,0]*Fref[1,1]*Fref[2,2]
        dJ = dt * Jbase * wp.trace(center_grad_dv[cdof])
        dpdJ = pp.mu_lam_kappa[2] # kappa
        dp = dpdJ * dJ # scalar pressure
        stress_scratch[psi_k, cdof] = V0 * (dp * Jbase) * wp.identity(3, dtype=real)


@wp.kernel
def lite_implicit_df_kernel_fast(
    n_active_nodes: wp.array(dtype=int, ndim=1),
    ndof2bijk: wp.array(dtype=wp.vec2i, ndim=1),
    center2dof: wp.array(dtype=int, ndim=4),
    block2bid: wp.array(dtype=int, ndim=3),
    block_xyz_by_id: wp.array(dtype=wp.vec3i, ndim=1),
    dv: wp.array(dtype=vec3),
    df: wp.array(dtype=vec3),
    pd_damping_arr: wp.array(dtype=real),
    node_tikhonov: wp.array(dtype=real, ndim=1),

    bc_block2bid: wp.array(dtype=int, ndim=3),
    bc_type: wp.array(dtype=int, ndim=4),
    bc_norm: wp.array(dtype=vec3, ndim=4),
    bc_velo: wp.array(dtype=vec3, ndim=4),

    hf_bc_p: wp.array(dtype=vec3),
    hf_bc_n: wp.array(dtype=vec3),
    hf_bc_v: wp.array(dtype=vec3),
    hf_bc_type: wp.array(dtype=int),
    num_hf: wp.int32,

    stress_scratch: wp.array(dtype=mat33, ndim=2),
    center_vol: wp.array(dtype=real, ndim=4),
    grid_m: wp.array(dtype=real, ndim=4),
    grid_size: wp.vec3i,
    dx: real,
    dt: real,
    n_psi: int,
):
    dof = wp.tid()
    if dof >= n_active_nodes[0]:
        df[dof] = vec3(_0)
        return

    bid, local_n1d = ndof2bijk[dof].x, ndof2bijk[dof].y
    node = local2global(block_xyz_by_id[bid], local_n1d)
    ni, nj, nk = node.x, node.y, node.z

    bct, bcn, bcv = query_bc_sp(node, bc_block2bid, bc_type, bc_norm, bc_velo, hf_bc_p, hf_bc_n, hf_bc_v, hf_bc_type, num_hf, dx)
    if bct == 1:  # sticky
        df[dof] = vec3(_0)
        return

    li, lj, lk = unlin_IJK(local_n1d)
    mi = grid_m[bid, li, lj, lk]

    df[dof] = (pd_damping_arr[0] + _1) * mi * dv[dof] + node_tikhonov[dof] * dv[dof]
    inv_4dx = _14 / dx
    center_size = wp.vec3i(grid_size[0]-1, grid_size[1]-1, grid_size[2]-1)

    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                ci, cj, ck = ni - di, nj - dj, nk - dk
                center = wp.vec3i(ci, cj, ck)
                if not in_region(center, center_size): continue

                bcoords = block_coords_from_node(ci, cj, ck)
                cbid = block2bid[bcoords.x, bcoords.y, bcoords.z]
                if cbid < 0: continue

                local_c1d = local_linear(local_coords_in_block(ci, cj, ck))
                lci, lcj, lck = unlin_IJK(local_c1d)
                lcjk = lin_JK(lcj, lck)
                cdof = center2dof[cbid, lci, lcj, lck]
                if cdof < 0: continue

                sx_ic = _n1 if di == 0 else _1
                sy_ic = _n1 if dj == 0 else _1
                sz_ic = _n1 if dk == 0 else _1
                dw_ic = inv_4dx * vec3(sx_ic, sy_ic, sz_ic) # ∇w_ic

                for psi_k in range(n_psi):
                    if center_vol[psi_k, cbid, lci, lcjk] <= _0: continue
                    df[dof] += dt * stress_scratch[psi_k, cdof] @ dw_ic
    
    if bct > 0:  # boundary projection
        df[dof] = proj_boundary_delta_vel(df[dof], bct, bcn)


@wp.kernel
def lite_implicit_precond_kernel_mass(
    n_active_nodes: wp.array(dtype=int, ndim=1),
    ndof2bijk: wp.array(dtype=wp.vec2i, ndim=1),
    block_xyz_by_id: wp.array(dtype=wp.vec3i, ndim=1),
    r: wp.array(dtype=vec3, ndim=1),
    z: wp.array(dtype=vec3, ndim=1),

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
    dx: real,
):
    dof = wp.tid()
    if dof >= n_active_nodes[0]:
        z[dof] = vec3(_0)
        return
    bid, local_n1d = ndof2bijk[dof].x, ndof2bijk[dof].y
    node = local2global(block_xyz_by_id[bid], local_n1d)

    bct, bcn, bcv = query_bc_sp(node, bc_block2bid, bc_type, bc_norm, bc_velo, hf_bc_p, hf_bc_n, hf_bc_v, hf_bc_type, num_hf, dx)
    if bct == 1: # sticky
        z[dof] = vec3(_0)
        return

    li, lj, lk = unlin_IJK(local_n1d)
    mi = grid_m[bid, li, lj, lk]

    z[dof] = (_1 / (mi)) * r[dof]
    if bct > 0:  # boundary projection
        z[dof] = proj_boundary_delta_vel(z[dof], bct, bcn)

@wp.kernel
def lite_implicit_precond_kernel_Hii(
    n_active_nodes: wp.array(dtype=int, ndim=1),
    ndof2bijk: wp.array(dtype=wp.vec2i, ndim=1),
    block_xyz_by_id: wp.array(dtype=wp.vec3i, ndim=1),
    node_Hii_inv: wp.array(dtype=mat33, ndim=1),
    r: wp.array(dtype=vec3, ndim=1),
    z: wp.array(dtype=vec3, ndim=1),

    bc_block2bid: wp.array(dtype=int, ndim=3),
    bc_type: wp.array(dtype=int, ndim=4),
    bc_norm: wp.array(dtype=vec3, ndim=4),
    bc_velo: wp.array(dtype=vec3, ndim=4),

    hf_bc_p: wp.array(dtype=vec3),
    hf_bc_n: wp.array(dtype=vec3),
    hf_bc_v: wp.array(dtype=vec3),
    hf_bc_type: wp.array(dtype=int),
    num_hf: wp.int32,

    dx: real,
):
    dof = wp.tid()
    if dof >= n_active_nodes[0]:
        z[dof] = vec3(_0)
        return
    bid, local_n1d = ndof2bijk[dof].x, ndof2bijk[dof].y
    node = local2global(block_xyz_by_id[bid], local_n1d)

    bct, bcn, bcv = query_bc_sp(node, bc_block2bid, bc_type, bc_norm, bc_velo, hf_bc_p, hf_bc_n, hf_bc_v, hf_bc_type, num_hf, dx)
    if bct == 1: # sticky
        z[dof] = vec3(_0)
        return

    z[dof] = node_Hii_inv[dof] @ r[dof]
    if bct > 0:  # boundary projection
        z[dof] = proj_boundary_delta_vel(z[dof], bct, bcn)
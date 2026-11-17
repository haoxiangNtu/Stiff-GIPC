from ...types import *
from ..constant import (
    _05, _0, _1, _13,
)
from ...sp_grid import (
    lin_IJK, unlin_IJK,
    local2global,
)
from ...boundary_utils import (
    proj_boundary_vel,
    proj_boundary_delta_vel,
    query_bc_sp,
)

@wp.func
def in_region(ijk: wp.vec3i, region: wp.vec3i) -> wp.int32:
    return wp.int32((ijk.x >= 0) and (ijk.x < region.x) and
                    (ijk.y >= 0) and (ijk.y < region.y) and
                    (ijk.z >= 0) and (ijk.z < region.z))

@wp.func
def is_bc(ijk: wp.vec3i, region: wp.vec3i) -> wp.int32:
    # returns 1 if on boundary band, else 0
    return wp.int32((ijk.x <= 5) or (ijk.x > region.x - 5) or (ijk.y <= 5) or (ijk.y > region.y - 5) or (ijk.z <= 5) or (ijk.z > region.z - 5))

@wp.func
def is_boundary(
    node1d: int,
    boundary: wp.array(dtype=int, ndim=1)
) -> wp.int32:
    return wp.int32(boundary[node1d] != 0)

@wp.func
def rowmajor_index(i: wp.int32, j: wp.int32) -> wp.int32:
    return i * 3 + j

@wp.func
def vec9_rowmajor(A: mat33) -> vec9:
    # 3x3 -> [a00, a01, a02, a10, a11, a12, a20, a21, a22]
    return vec9(A[0, 0], A[0, 1], A[0, 2], A[1, 0], A[1, 1], A[1, 2], A[2, 0], A[2, 1], A[2, 2])

@wp.func
def mat3_from_vec9_rowmajor(v: vec9) -> mat33:
    # [a00, a01, a02, a10, a11, a12, a20, a21, a22] -> 3x3
    return mat33(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8])

@wp.func
def dev3(A: mat33) -> mat33:
    # deviatoric part in 3D
    return A - _13 * (A[0, 0] + A[1, 1] + A[2, 2]) * wp.identity(3, real)

@wp.kernel
def max_particle_speed_kernel(
    v: wp.array(dtype=vec3),       # particle velocities
    vmax: wp.array(dtype=real), # length-1 array for result
):
    p = wp.tid()
    s = wp.length(v[p])
    wp.atomic_max(vmax, 0, s)


@wp.kernel
def update_vnew_with_iterate_kernel(
    block_count: wp.array(dtype=int, ndim=1),
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

    grid_m     : wp.array(dtype=real, ndim=4),
    grid_v_it  : wp.array(dtype=vec3, ndim=4),
    grid_v_new : wp.array(dtype=vec3, ndim=4),
    grid_size  : wp.vec3i,

    dx: real,
    damping: real = 1.0,
):
    bid, li, lj, lk = wp.tid()
    if bid >= block_count[0]: return
    node = local2global(block_xyz_by_id[bid], lin_IJK(li, lj, lk))
    if not in_region(node, grid_size): return

    bct, bcn, bcv = query_bc_sp(node, bc_block2bid, bc_type, bc_norm, bc_velo, hf_bc_p, hf_bc_n, hf_bc_v, hf_bc_type, num_hf, dx)

    if grid_m[bid, li, lj, lk] <= _0:
        # inactive or sticky bc
        grid_v_new[bid, li, lj, lk] = vec3(_0)
    else:
        v = grid_v_it[bid, li, lj, lk]
        if bct > 0:
            v = proj_boundary_vel(v, bct, bcn, bcv)
        grid_v_new[bid, li, lj, lk] = damping * v


@wp.kernel
def commit_dv_to_iterate_kernel(
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

    search_dv: wp.array(dtype=vec3, ndim=1),
    grid_v_it: wp.array(dtype=vec3, ndim=4),
    max_update: wp.array(dtype=real, ndim=1),
    alpha: real,
    dx: real,
):
    dof = wp.tid()
    if dof >= n_active_nodes[0]: return

    bid, local_n1d = ndof2bijk[dof].x, ndof2bijk[dof].y
    node = local2global(block_xyz_by_id[bid], local_n1d)
    li, lj, lk = unlin_IJK(local_n1d)

    bct, bcn, bcv = query_bc_sp(node, bc_block2bid, bc_type, bc_norm, bc_velo, hf_bc_p, hf_bc_n, hf_bc_v, hf_bc_type, num_hf, dx)
    dv_search = search_dv[dof]
    if bct > 0:
        dv_search = proj_boundary_delta_vel(dv_search, bct, bcn)
    upd = wp.max(wp.max(wp.abs(dv_search[0]), wp.abs(dv_search[1])), wp.abs(dv_search[2]))
    wp.atomic_max(max_update, 0, upd)
    grid_v_it[bid, li, lj, lk] += alpha * dv_search
    if bct > 0:
        grid_v_it[bid, li, lj, lk] = proj_boundary_vel(grid_v_it[bid, li, lj, lk], bct, bcn, bcv)

# ---- helpers ----

@wp.func
def jacobi_rot_01(c: real, s: real, A: mat33, V: mat33):
    # rotate in plane (0,1)
    a00 = A[0,0]; a01 = A[0,1]; a02 = A[0,2]
    a11 = A[1,1]; a12 = A[1,2]
    a22 = A[2,2]

    a00n = c*c*a00 - real(2.0)*c*s*a01 + s*s*a11
    a11n = s*s*a00 + real(2.0)*c*s*a01 + c*c*a11
    a02n = c*a02 - s*a12
    a12n = s*a02 + c*a12

    A = mat33(
        a00n, real(0.0), a02n,
        real(0.0), a11n, a12n,
        a02n, a12n, a22
    )

    # V <- V R01
    v0 = vec3(V[0,0], V[1,0], V[2,0])
    v1 = vec3(V[0,1], V[1,1], V[2,1])
    v2 = vec3(V[0,2], V[1,2], V[2,2])

    v0n = c*v0 - s*v1
    v1n = s*v0 + c*v1
    V = wp.matrix_from_cols(v0n, v1n, v2)
    return A, V


@wp.func
def jacobi_rot_02(c: real, s: real, A: mat33, V: mat33):
    # rotate in plane (0,2)
    a00 = A[0,0]; a01 = A[0,1]; a02 = A[0,2]
    a11 = A[1,1]; a12 = A[1,2]
    a22 = A[2,2]

    a00n = c*c*a00 - real(2.0)*c*s*a02 + s*s*a22
    a22n = s*s*a00 + real(2.0)*c*s*a02 + c*c*a22
    a01n = c*a01 - s*a12
    a12n = s*a01 + c*a12

    A = mat33(
        a00n, a01n, real(0.0),
        a01n, a11,  a12n,
        real(0.0), a12n, a22n
    )

    # V <- V R02
    v0 = vec3(V[0,0], V[1,0], V[2,0])
    v1 = vec3(V[0,1], V[1,1], V[2,1])
    v2 = vec3(V[0,2], V[1,2], V[2,2])

    v0n = c*v0 - s*v2
    v2n = s*v0 + c*v2
    V = wp.matrix_from_cols(v0n, v1, v2n)
    return A, V


@wp.func
def jacobi_rot_12(c: real, s: real, A: mat33, V: mat33):
    # rotate in plane (1,2)
    a00 = A[0,0]; a01 = A[0,1]; a02 = A[0,2]
    a11 = A[1,1]; a12 = A[1,2]
    a22 = A[2,2]

    a11n = c*c*a11 - real(2.0)*c*s*a12 + s*s*a22
    a22n = s*s*a11 + real(2.0)*c*s*a12 + c*c*a22
    a01n = c*a01 - s*a02
    a02n = s*a01 + c*a02

    A = mat33(
        a00,  a01n, a02n,
        a01n, a11n, real(0.0),
        a02n, real(0.0), a22n
    )

    # V <- V R12
    v0 = vec3(V[0,0], V[1,0], V[2,0])
    v1 = vec3(V[0,1], V[1,1], V[2,1])
    v2 = vec3(V[0,2], V[1,2], V[2,2])

    v1n = c*v1 - s*v2
    v2n = s*v1 + c*v2
    V = wp.matrix_from_cols(v0, v1n, v2n)
    return A, V

@wp.func
def sym_eig3_jacobi(E_in: mat33, sweeps: int = 12, rel_tol: real = real(1.0e-10)):
    # Symmetric eigen decomposition via fixed Jacobi sweeps:
    # returns (Q, w) such that E = Q diag(w) Q^T
    A = real(0.5) * (E_in + wp.transpose(E_in))
    V = wp.identity(3, dtype=real)

    # Fixed number of sweeps: each sweep zeroes 3 off-diagonal elements
    # (0,1), (0,2), (1,2)
    for _ in range(sweeps):
        # relative threshold
        thr = rel_tol * (wp.abs(A[0,0]) + wp.abs(A[1,1]) + wp.abs(A[2,2]) + real(1.0))

        # convergence check
        off = wp.abs(A[0,1]) + wp.abs(A[0,2]) + wp.abs(A[1,2])
        if off <= thr:
            break

        # --- (0,1) ---
        a00 = A[0, 0]; a11 = A[1, 1]; a01 = A[0, 1]
        if wp.abs(a01) > thr:
            tau = (a11 - a00) / (real(2.0) * a01)
            t = wp.sign(tau) / (wp.abs(tau) + wp.sqrt(real(1.0) + tau*tau))
            c = real(1.0) / wp.sqrt(real(1.0) + t*t)
            s = t * c
            A, V = jacobi_rot_01(c, s, A, V)

        # --- (0,2) ---
        a00 = A[0, 0]; a22 = A[2, 2]; a02 = A[0, 2]
        if wp.abs(a02) > thr:
            tau = (a22 - a00) / (real(2.0) * a02)
            t = wp.sign(tau) / (wp.abs(tau) + wp.sqrt(real(1.0) + tau*tau))
            c = real(1.0) / wp.sqrt(real(1.0) + t*t)
            s = t * c
            A, V = jacobi_rot_02(c, s, A, V)

        # --- (1,2) ---
        a11 = A[1, 1]; a22 = A[2, 2]; a12 = A[1, 2]
        if wp.abs(a12) > thr:
            tau = (a22 - a11) / (real(2.0) * a12)
            t = wp.sign(tau) / (wp.abs(tau) + wp.sqrt(real(1.0) + tau*tau))
            c = real(1.0) / wp.sqrt(real(1.0) + t*t)
            s = t * c
            A, V = jacobi_rot_12(c, s, A, V)

    # Eigenvalues on diagonal, eigenvectors in V
    w = vec3(A[0, 0], A[1, 1], A[2, 2])
    Q = V
    return Q, w


@wp.func
def sort_eig_desc(Q: mat33, w: vec3):
    # Sort eigenvalues descending, permute columns of Q accordingly.
    # Simple 3-element sorting network.
    # swap 0,1
    if w[0] < w[1]:
        w = vec3(w[1], w[0], w[2])
        Q = mat33(
            Q[0,1], Q[0,0], Q[0,2],
            Q[1,1], Q[1,0], Q[1,2],
            Q[2,1], Q[2,0], Q[2,2],
        )
    # swap 0,2
    if w[0] < w[2]:
        w = vec3(w[2], w[1], w[0])
        Q = mat33(
            Q[0,2], Q[0,1], Q[0,0],
            Q[1,2], Q[1,1], Q[1,0],
            Q[2,2], Q[2,1], Q[2,0],
        )
    # swap 1,2
    if w[1] < w[2]:
        w = vec3(w[0], w[2], w[1])
        Q = mat33(
            Q[0,0], Q[0,2], Q[0,1],
            Q[1,0], Q[1,2], Q[1,1],
            Q[2,0], Q[2,2], Q[2,1],
        )
    return Q, w


@wp.func
# def sym_exp3(E: mat33) -> mat33:
def sym_exp3(E: mat33) -> Tuple[mat33, vec3]:
    # Robust symmetric matrix exponential via eigen-decomposition (Jacobi),
    # avoids SVD U/V mismatch and sign/ordering instabilities.
    Q, w = sym_eig3_jacobi(E, sweeps=6, rel_tol=real(1.0e-17))
    Q, w = sort_eig_desc(Q, w)
    return Q, w

    # exp_w = vec3(wp.exp(w[0]), wp.exp(w[1]), wp.exp(w[2]))

    # # exp(E) = Q diag(exp(w)) Q^T
    # return Q @ wp.diag(exp_w) @ wp.transpose(Q)
import warp as wp
from ...types import *
from ..constant import (
    _0, _1, _05, _2, _3, _4, _n1, _13, _14,
    s_min,
)
from .helper_misc import (
    dev3,
    sym_exp3,
    in_region,
)
from .helper_constitutive_model import (
    StVK_Hencky_tau_3D,
    water_tau_3D,
    Drucker_Prager_return_mapping,
    von_mises_apply,
    NonAssoc_CamClay_return_mapping,
    snow13_return_mapping, snow13_hardening,
    foam_return_mapping,
)
from ...sp_grid import (
    B, BLOCK_VOL,
    lin_JK, lin_I_JK, lin_IJK, unlin_JK,
    block_coords_from_node, local_coords_in_block, local_linear, local2global,
)
from ...boundary_utils import (
    proj_boundary_vel,
    query_bc_sp,
)

@wp.kernel
def lite_p2c_kernel_1(
    block2bid: wp.array(dtype=int, ndim=3),

    ptc_x: wp.array(dtype=vec3),
    ptc_v: wp.array(dtype=vec3),
    ptc_k: wp.array(dtype=int),
    ptc_F: wp.array(dtype=mat33), 
    ptc_vol0: wp.array(dtype=real),
    ptc_m: wp.array(dtype=real),
    ptc_G: wp.array(dtype=mat33),
    ptc_dlogJ: wp.array(dtype=real),
    psi_params: wp.array(dtype=MaterialParams),

    center_m: wp.array(dtype=real, ndim=4),
    center_v: wp.array(dtype=vec3, ndim=4),
    center_G: wp.array(dtype=mat33, ndim=4),
    center_vol: wp.array(dtype=real, ndim=4),
    center_tau: wp.array(dtype=mat33, ndim=4),
    center_dlogJ: wp.array(dtype=real, ndim=4),

    center_size: wp.vec3i,
    dx: real,
    enable_apic: bool = True,
):
    p = wp.tid()
    psi_k = ptc_k[p]
    pp = psi_params[psi_k]
    Fp = ptc_F[p]
    xp = ptc_x[p]

    pos_c = xp / dx - vec3(_05)
    base = wp.vec3i(int(wp.floor(pos_c[0])), int(wp.floor(pos_c[1])), int(wp.floor(pos_c[2])))
    fx = pos_c - vec3(real(base.x), real(base.y), real(base.z))
    wx0 = _1 - fx[0]; wx1 = fx[0]
    wy0 = _1 - fx[1]; wy1 = fx[1]
    wz0 = _1 - fx[2]; wz1 = fx[2]
    Vp = ptc_vol0[p]
    Gp = ptc_G[p]
    dlogJ = ptc_dlogJ[p]
    mu_lam_kappa = pp.mu_lam_kappa
    if pp.cons_type == Constitutive.stvk:
        mu = mu_lam_kappa[0]; lam = mu_lam_kappa[1]
        if pp.mate_type == Material.snow13:
            hardening = mu_lam_kappa[2]
            h = snow13_hardening(dlogJ=dlogJ, hardening=hardening)
            mu *= h; lam *= h
        tau_p = StVK_Hencky_tau_3D(Fp, mu, lam, s_min)
    elif pp.cons_type == Constitutive.pressure:
        Jp = Fp[0,0] * Fp[1,1] * Fp[2,2]
        tau_p = wp.identity(3, real) * water_tau_3D(Jp, mu_lam_kappa[2])
    else:
        tau_p = mat33(_0)

    V0_tau = Vp * tau_p

    for i in range(2):
        for j in range(2):
            for k in range(2):
                center = base + wp.vec3i(i, j, k)
                ci, cj, ck = center.x, center.y, center.z
                if not in_region(center, center_size): continue
                w = (wx0 if i == 0 else wx1) * (wy0 if j == 0 else wy1) * (wz0 if k == 0 else wz1)
                dm = w * ptc_m[p]
                vel = ptc_v[p]

                # map to sparse node
                bc = block_coords_from_node(ci, cj, ck)
                bid = block2bid[bc[0], bc[1], bc[2]]
                if bid < 0: # not active? this should not happen
                    continue

                local_cijk = local_coords_in_block(ci, cj, ck)
                lci, lcj, lck = local_cijk.x, local_cijk.y, local_cijk.z
                lcjk = lin_JK(lcj, lck) # = lcj * 32 + lck

                wp.atomic_add(center_m, bid, lci, lcj, lck, dm)
                wp.atomic_add(center_G, bid, lci, lcj, lck, dm * Gp)
                if enable_apic:
                    xc = dx * vec3(real(center[0]) + _05, real(center[1]) + _05, real(center[2]) + _05)
                    vel = vel + Gp @ (xc - xp)
                wp.atomic_add(center_v, bid, lci, lcj, lck, dm * vel)
                wp.atomic_add(center_tau, psi_k, bid, lci, lcjk, w * V0_tau)
                wp.atomic_add(center_vol, psi_k, bid, lci, lcjk, w * Vp)
                wp.atomic_add(center_dlogJ, psi_k, bid, lci, lcjk, w * Vp * dlogJ)


@wp.kernel
def lite_p2c_kernel_2( # (do NOT normalize tau0)
    block_count: wp.array(dtype=int),
    block_xyz_by_id: wp.array(dtype=wp.vec3i),

    center_m: wp.array(dtype=real, ndim=4),
    center_vol: wp.array(dtype=real, ndim=4),
    center_tau: wp.array(dtype=mat33, ndim=4),
    center_F: wp.array(dtype=mat33, ndim=4),
    center_v: wp.array(dtype=vec3, ndim=4),
    center_G: wp.array(dtype=mat33, ndim=4),
    center_dlogJ: wp.array(dtype=real, ndim=4),

    center_size: wp.vec3i,
    psi_params: wp.array(dtype=MaterialParams),
    reconstruct_center_F: bool,
):
    psi_k, bid, lci, lcjk = wp.tid()
    if reconstruct_center_F:
        center_F[psi_k, bid, lci, lcjk] = mat33(_0)
    if bid >= block_count[0]: return
    center = local2global(block_xyz_by_id[bid], lin_I_JK(lci, lcjk))  # to suppress unused variable warning

    if not in_region(center, center_size): return
    lcj, lck = unlin_JK(lcjk)

    if center_m[bid, lci, lcj, lck] <= _0: return
    if psi_k == 0: # only normalize once
        invm = _1 / center_m[bid, lci, lcj, lck]
        center_v[bid, lci, lcj, lck] *= invm
        center_G[bid, lci, lcj, lck] *= invm

    if center_vol[psi_k, bid, lci, lcjk] > _0:
        inv_vol = _1 / center_vol[psi_k, bid, lci, lcjk]
        center_tau[psi_k, bid, lci, lcjk] *= inv_vol
        center_dlogJ[psi_k, bid, lci, lcjk] *= inv_vol
        if not reconstruct_center_F:
            return
        pp = psi_params[psi_k]
        tau_c = center_tau[psi_k, bid, lci, lcjk]
        mu_lam_kappa = pp.mu_lam_kappa
        if pp.cons_type == Constitutive.stvk: # center Kirchhoff stress (intensive)
            mu = mu_lam_kappa[0]; lam = mu_lam_kappa[1]
            if pp.mate_type == Material.snow13:
                dlogJ = center_dlogJ[psi_k, bid, lci, lcjk]
                hardening = mu_lam_kappa[2]
                h = snow13_hardening(dlogJ=dlogJ, hardening=hardening)
                mu *= h; lam *= h
            tr_tau = tau_c[0, 0] + tau_c[1, 1] + tau_c[2, 2]
            E = (_1 / (_2 * mu)) * dev3(tau_c) \
                + (tr_tau / (_3 * (_2 * mu + _3 * lam))) * wp.identity(3, real)
            Q, e = sym_exp3(E) # stretch
            e_exp = vec3(wp.exp(e[0]), wp.exp(e[1]), wp.exp(e[2]))
            Uc = Q @ wp.diag(e_exp) @ wp.transpose(Q)
            center_F[psi_k, bid, lci, lcjk] = Uc # choose R = I so that P(Fc) Fc^T = tau_c
        elif pp.cons_type == Constitutive.pressure:
            kappa = mu_lam_kappa[2]
            Jbase = _05 * (_1 + wp.sqrt(_1 + _4 * tau_c[0,0] / kappa))
            axis_stretch = wp.pow(Jbase, _13)
            center_F[psi_k, bid, lci, lcjk] = wp.diag(vec3(axis_stretch))


def lite_p2c(
    block_count: wp.array(dtype=int),
    block2bid: wp.array(dtype=int, ndim=3),
    block_xyz_by_id: wp.array(dtype=wp.vec3i),

    ptc_x: wp.array(dtype=vec3),
    ptc_v: wp.array(dtype=vec3),
    ptc_k: wp.array(dtype=int),
    ptc_F: wp.array(dtype=mat33), 
    ptc_vol0: wp.array(dtype=real),
    ptc_m: wp.array(dtype=real),
    ptc_G: wp.array(dtype=mat33),
    ptc_dlogJ: wp.array(dtype=real),
    psi_params: wp.array(dtype=MaterialParams),

    center_m: wp.array(dtype=real, ndim=4),
    center_v: wp.array(dtype=vec3, ndim=4),
    center_F: wp.array(dtype=mat33, ndim=4),
    center_G: wp.array(dtype=mat33, ndim=4),
    center_vol: wp.array(dtype=real, ndim=4),
    center_tau: wp.array(dtype=mat33, ndim=4),
    center_dlogJ: wp.array(dtype=real, ndim=4),

    center_size: wp.vec3i,
    dx: real,
    n_ptc: int,
    device: str,
    enable_apic: bool = True,
    reconstruct_center_F: bool = True, # not needed for explicit setting
):
    wp.launch(
        kernel=lite_p2c_kernel_1,
        dim=n_ptc,
        inputs=[
            block2bid,
            ptc_x, ptc_v, ptc_k, ptc_F, ptc_vol0, ptc_m, ptc_G, ptc_dlogJ, psi_params,
            center_m, center_v, center_G, center_vol, center_tau, center_dlogJ,
            center_size, dx, enable_apic,
        ],
        device=device,
    )
    wp.launch(
        kernel=lite_p2c_kernel_2,
        dim=center_vol.shape,
        inputs=[
            block_count, block_xyz_by_id,
            center_m, center_vol, center_tau, center_F, center_v, center_G, center_dlogJ,
            center_size, psi_params, reconstruct_center_F,
        ],
        device=device,
    )

# 1) scatter from centers to grid (PIC + optional GRAD_APIC)
@wp.kernel
def lite_c2g_kernel_1(
    block_count: wp.array(dtype=int),
    block2bid: wp.array(dtype=int, ndim=3),
    block_xyz_by_id: wp.array(dtype=wp.vec3i),

    bc_block2bid: wp.array(dtype=int, ndim=3),
    bc_type: wp.array(dtype=int, ndim=4),
    bc_norm: wp.array(dtype=vec3, ndim=4),
    bc_velo: wp.array(dtype=vec3, ndim=4),

    hf_bc_p: wp.array(dtype=vec3),
    hf_bc_n: wp.array(dtype=vec3),
    hf_bc_v: wp.array(dtype=vec3),
    hf_bc_type: wp.array(dtype=int),
    num_hf: wp.int32,

    center_m: wp.array(dtype=real, ndim=4),       # [center_size, center_size]
    center_v: wp.array(dtype=vec3, ndim=4),
    center_G: wp.array(dtype=mat33, ndim=4),
    center_vol: wp.array(dtype=real, ndim=4),
    center_tau: wp.array(dtype=mat33, ndim=4),
    grid_m: wp.array(dtype=real, ndim=4),         # [grid_size, grid_size]
    grid_v: wp.array(dtype=vec3, ndim=4),
    grid_v_new: wp.array(dtype=vec3, ndim=4),
    grid_size: wp.vec3i,  
    center_size: wp.vec3i,  
    gravity: vec3,
    dx: real,
    dt: real,
    n_psi: int,
    explicit_force: bool,
    enable_apic: bool = True,
):
    bid, li, lj, lk = wp.tid()
    if bid >= block_count[0]: return
    ljk = lin_JK(lj, lk)
    node = local2global(block_xyz_by_id[bid], lin_I_JK(li, ljk))
    if not in_region(node, grid_size): return
    i, j, k = node.x, node.y, node.z

    w = real(0.125)
    xi = vec3(real(i), real(j), real(k)) * dx
    grid_m[bid, li, lj, lk] = _0
    grid_v[bid, li, lj, lk] = vec3(_0)
    impulse = vec3(_0)
    inv_4dx =  _14 / dx
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                ci = i - di; cj = j - dj; ck = k - dk
                center = wp.vec3i(ci, cj, ck)
                if not in_region(center, center_size): continue

                cbcoords = block_coords_from_node(ci, cj, ck)
                cbid = block2bid[cbcoords[0], cbcoords[1], cbcoords[2]]
                if cbid < 0: # not active? continue
                    continue
                lcijk = local_coords_in_block(ci, cj, ck)
                lci, lcj, lck = lcijk.x, lcijk.y, lcijk.z
                lcjk = lin_JK(lcj, lck)

                cm = center_m[cbid, lci, lcj, lck]
                if cm <= _0: continue
                mom = cm * center_v[cbid, lci, lcj, lck]
                grid_m[bid, li, lj, lk] += w * cm
                grid_v[bid, li, lj, lk] += w * mom
                if enable_apic:
                    xc = vec3(real(ci) + _05, real(cj) + _05, real(ck) + _05) * dx
                    grid_v[bid, li, lj, lk] += w * cm * center_G[cbid, lci, lcj, lck] @ (xi - xc)
                if explicit_force:
                    for psi_k in range(n_psi):
                        vol_c = center_vol[psi_k, cbid, lci, lcjk]
                        if vol_c <= _0: continue
                        sx = _n1 if di == 0 else _1
                        sy = _n1 if dj == 0 else _1
                        sz = _n1 if dk == 0 else _1
                        grad_w = inv_4dx * vec3(sx, sy, sz) 
                        fi = -vol_c * (center_tau[psi_k, cbid, lci, lcjk] @ grad_w) # (equals -V0 * tau_c ∇w)
                        impulse += dt * fi

    mi = grid_m[bid, li, lj, lk]
    if mi > _0:
        grid_v[bid, li, lj, lk] = grid_v[bid, li, lj, lk] / mi
        # apply boundary condition
        bct, bcn, bcv = query_bc_sp(node, bc_block2bid, bc_type, bc_norm, bc_velo, hf_bc_p, hf_bc_n, hf_bc_v, hf_bc_type, num_hf, dx)
        if bct > 0:
            grid_v[bid, li, lj, lk] = proj_boundary_vel(grid_v[bid, li, lj, lk], bct, bcn, bcv)

        if explicit_force:
            grid_v_new[bid, li, lj, lk] = grid_v[bid, li, lj, lk] + impulse / mi + dt * gravity
            if bct > 0:
                grid_v_new[bid, li, lj, lk] = proj_boundary_vel(grid_v_new[bid, li, lj, lk], bct, bcn, bcv)
    else:
        grid_v[bid, li, lj, lk] = vec3(_0)
        grid_v_new[bid, li, lj, lk] = vec3(_0)


def lite_c2g(
    block_count: wp.array(dtype=int),
    block2bid: wp.array(dtype=int, ndim=3),
    block_xyz_by_id: wp.array(dtype=wp.vec3i),
    bc_block2bid: wp.array(dtype=int, ndim=3),
    bc_type: wp.array(dtype=int, ndim=4),
    bc_norm: wp.array(dtype=vec3, ndim=4),
    bc_velo: wp.array(dtype=vec3, ndim=4),
    hf_bc_p: wp.array(dtype=vec3),
    hf_bc_n: wp.array(dtype=vec3),
    hf_bc_v: wp.array(dtype=vec3),
    hf_bc_type: wp.array(dtype=int),
    num_hf: wp.int32,
    center_m: wp.array(dtype=real, ndim=4),       # [center_size, center_size]
    center_v: wp.array(dtype=vec3, ndim=4),
    center_G: wp.array(dtype=mat33, ndim=4),
    center_vol: wp.array(dtype=real, ndim=4),
    center_tau: wp.array(dtype=mat33, ndim=4),
    grid_m: wp.array(dtype=real, ndim=4),         # [grid_size, grid_size]
    grid_v: wp.array(dtype=vec3, ndim=4),
    grid_v_new: wp.array(dtype=vec3, ndim=4),
    grid_size: wp.vec3i,
    center_size: wp.vec3i,
    gravity: vec3,
    dx: real,
    dt: real,
    n_psi: int,
    device: str,
    explicit_force: bool = False,
    enable_apic: bool = True,
):
    wp.launch(
        kernel=lite_c2g_kernel_1,
        dim=(block_count.numpy()[0], B, B, B),
        inputs=[
            block_count, block2bid, block_xyz_by_id,
            bc_block2bid, bc_type, bc_norm, bc_velo,
            hf_bc_p, hf_bc_n, hf_bc_v, hf_bc_type, num_hf,
            center_m, center_v, center_G, center_vol, center_tau,
            grid_m, grid_v, grid_v_new, grid_size, center_size,
            gravity, dx, dt, n_psi, explicit_force, enable_apic,
        ],
        device=device,
    )


@wp.kernel
def lite_g2c_kernel(
    block_count: wp.array(dtype=int),
    block2bid: wp.array(dtype=int, ndim=3),
    block_xyz_by_id: wp.array(dtype=wp.vec3i),

    grid_v      : wp.array(dtype=vec3, ndim=4),
    grid_v_new  : wp.array(dtype=vec3, ndim=4),
    center_v    : wp.array(dtype=vec3, ndim=4),
    center_dv   : wp.array(dtype=vec3, ndim=4),
    center_G    : wp.array(dtype=mat33, ndim=4),
    center_size : wp.vec3i,
    dx          : real,
):
    cbid, lci, lcj, lck = wp.tid()
    if cbid >= block_count[0]: return
    center = local2global(block_xyz_by_id[cbid], lin_IJK(lci, lcj, lck))
    if not in_region(center, center_size): return
    ci, cj, ck = center.x, center.y, center.z

    v_c  = vec3(_0)
    dv_c = vec3(_0)
    grad_v = mat33(_0)

    inv_4dx = _14 / dx

    w = real(0.125)
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                node = wp.vec3i(ci + di, cj + dj, ck + dk)
                bcoords = block_coords_from_node(node.x, node.y, node.z)
                bid = block2bid[bcoords[0], bcoords[1], bcoords[2]]
                if bid < 0: # not active? this should not happen
                    continue
                lijk = local_coords_in_block(node.x, node.y, node.z)
                li, lj, lk = lijk.x, lijk.y, lijk.z

                v_node = grid_v_new[bid, li, lj, lk]
                v_c  = v_c  + w * v_node
                dv_c = dv_c + w * (v_node - grid_v[bid, li, lj, lk])
                sx = _n1 if di == 0 else _1
                sy = _n1 if dj == 0 else _1
                sz = _n1 if dk == 0 else _1
                grad_w = inv_4dx * vec3(sx, sy, sz) # ∇w at cell center
                grad_v += wp.outer(v_node, grad_w) # Σ v ⊗ ∇w  == ∇v at center

    center_v[cbid, lci, lcj, lck] = v_c
    center_dv[cbid, lci, lcj, lck] = dv_c
    center_G[cbid, lci, lcj, lck] = grad_v


@wp.kernel
def lite_c2p_kernel(
    block2bid: wp.array(dtype=int, ndim=3),

    ptc_x      : wp.array(dtype=vec3),
    ptc_v      : wp.array(dtype=vec3),
    ptc_k      : wp.array(dtype=int),
    ptc_F      : wp.array(dtype=mat33),
    ptc_G      : wp.array(dtype=mat33),
    ptc_dlogJ  : wp.array(dtype=real),

    center_m   : wp.array(dtype=real,  ndim=4),
    center_v   : wp.array(dtype=vec3,  ndim=4),
    center_dv  : wp.array(dtype=vec3,  ndim=4),
    center_G   : wp.array(dtype=mat33, ndim=4),
    psi_params : wp.array(dtype=MaterialParams),

    center_size  : wp.vec3i,
    dx           : real,
    dt           : real,
    flip_ratio   : real,
):
    p = wp.tid()
    psi_k = ptc_k[p]
    pp = psi_params[psi_k]
    pos_c = ptc_x[p] / dx - vec3(_05)
    base = wp.vec3i(int(wp.floor(pos_c[0])), int(wp.floor(pos_c[1])), int(wp.floor(pos_c[2])))
    fx = pos_c - vec3(real(base.x), real(base.y), real(base.z))

    wx0 = _1 - fx[0]; wx1 = fx[0]
    wy0 = _1 - fx[1]; wy1 = fx[1]
    wz0 = _1 - fx[2]; wz1 = fx[2]

    v_pic = vec3(_0)
    dv_acc = vec3(_0)
    Gc = mat33(_0)

    for i in range(2):
        for j in range(2):
            for k in range(2):
                center = base + wp.vec3i(i, j, k)
                if not in_region(center, center_size): continue
                ci, cj, ck = center.x, center.y, center.z

                w = (wx0 if i == 0 else wx1) * (wy0 if j == 0 else wy1) * (wz0 if k == 0 else wz1)

                # map to sparse node
                bc = block_coords_from_node(ci, cj, ck)
                cbid = block2bid[bc[0], bc[1], bc[2]]
                if cbid < 0: # not active? this should not happen
                    continue
                lcijk = local_coords_in_block(ci, cj, ck)
                lci, lcj, lck = lcijk.x, lcijk.y, lcijk.z

                if center_m[cbid, lci, lcj, lck] <= _0: continue

                v_pic += w * center_v[cbid, lci, lcj, lck]
                dv_acc += w * center_dv[cbid, lci, lcj, lck]
                Gc += w * center_G[cbid, lci, lcj, lck]
    
    # FLIP + PIC blend
    v_flip = ptc_v[p] + dv_acc
    ptc_v[p] = flip_ratio * v_flip + (_1 - flip_ratio) * v_pic
    ptc_G[p] = Gc

    Fp = ptc_F[p]
    if pp.mate_type == Material.water:
        Jp = Fp[0,0] * Fp[1,1] * Fp[2,2]
        Jt = (_1 + dt * wp.trace(Gc)) * Jp
        Jt = wp.max(Jt, real(0.1))        # StiffGIPC patch: a negative J (extreme compression rate) would give NaN via pow(J, 1/3)
        ptc_F[p] = wp.identity(3, real) * wp.pow(Jt, _13)
    else:
        F_tr = (wp.identity(3, real) + dt * Gc) @ ptc_F[p]
        mu_lam_kappa = pp.mu_lam_kappa
        if pp.mate_type == Material.sand:
            # Drucker Prager
            # kappa for cohesion
            U, sig, V = Drucker_Prager_return_mapping(F_tr, mu_lam_kappa[2], mu_lam_kappa[0], mu_lam_kappa[1], pp.friction_angle)
            new_F = U @ wp.diag(sig) @ wp.transpose(V)
            ptc_dlogJ[p] += -wp.log(wp.determinant(new_F)) + wp.log(wp.determinant(F_tr)) 
            ptc_F[p] = new_F
        elif pp.mate_type == Material.vonmises:
            # plasticity (Von Mises)
            U, sig, V = von_mises_apply(F_tr, mu_lam_kappa[0], pp.yield_stress, s_min)
            ptc_F[p] = U @ wp.diag(sig) @ wp.transpose(V)
        elif pp.mate_type == Material.nacc:
            # kappa for M, friction_angle for beta, yield_stress for xi
            U, sig, V, new_dlogJ = NonAssoc_CamClay_return_mapping(
                F_tr,
                mu_lam_kappa[0],
                mu_lam_kappa[1],
                ptc_dlogJ[p],
                mu_lam_kappa[2],
                pp.friction_angle,
                pp.yield_stress
            )
            new_F = U @ wp.diag(sig) @ wp.transpose(V)
            # ptc_dlogJ[p] += -wp.log(wp.determinant(new_F)) + wp.log(wp.determinant(F_tr)) 
            ptc_dlogJ[p] = new_dlogJ
            ptc_F[p] = new_F
        elif pp.mate_type == Material.snow13:
            theta_c = pp.friction_angle
            theta_s = pp.yield_stress
            U, sig, V = snow13_return_mapping(F_tr, theta_c, theta_s)
            new_F = U @ wp.diag(sig) @ wp.transpose(V)
            ptc_dlogJ[p] += -wp.log(wp.determinant(new_F)) + wp.log(wp.determinant(F_tr)) 
            ptc_F[p] = new_F
        elif pp.mate_type == Material.foam:
            U, sig, V = foam_return_mapping(F_tr, mu_lam_kappa[0], pp.yield_stress, mu_lam_kappa[2], pp.friction_angle, dt)
            new_F = U @ wp.diag(sig) @ wp.transpose(V)
            ptc_dlogJ[p] += -wp.log(wp.determinant(new_F)) + wp.log(wp.determinant(F_tr)) 
            ptc_F[p] = new_F
        elif pp.mate_type == Material.elastic:
            # Elasticity Only
            # U, sig, V = wp.svd3(F_tr)
            # sig = vec3(wp.max(sig[0], s_min), wp.max(sig[1], s_min), wp.max(sig[2], s_min))
            ptc_F[p] = F_tr #U @ wp.diag(sig) @ wp.transpose(V)

    # advect particle using PIC velocity
    ptc_x[p] += dt * v_pic
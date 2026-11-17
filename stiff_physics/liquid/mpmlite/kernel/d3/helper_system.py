import warp as wp

from .helper_constitutive_model import snow13_hardening
from ...types import *
from ..constant import (
    _0, _05, _1, _2
)

@wp.kernel
def record_system_energy_kernel(
    ptc_x: wp.array(dtype=vec3),
    ptc_v: wp.array(dtype=vec3),
    ptc_m: wp.array(dtype=real),
    ptc_vol0: wp.array(dtype=real),
    ptc_F: wp.array(dtype=mat33),
    ptc_dlogJ: wp.array(dtype=real),
    ptc_k: wp.array(dtype=int),
    psi_params: wp.array(dtype=MaterialParams),
    dx: real,
    
    kinetic_energy: wp.array(dtype=real, ndim=1),
    elastic_energy: wp.array(dtype=real, ndim=1),
    linear_momentum: wp.array(dtype=vec3, ndim=1),
    angular_momentum: wp.array(dtype=vec3, ndim=1),
):
    p = wp.tid()
    psi_k = ptc_k[p]
    pp = psi_params[psi_k]
    mu_lam_kappa = pp.mu_lam_kappa
    mu, lam, kappa = mu_lam_kappa[0], mu_lam_kappa[1], mu_lam_kappa[2]
    xp = ptc_x[p]
    vp = ptc_v[p]
    Fp = ptc_F[p]
    mp = ptc_m[p]

    Ml = mp * vp  # linear momentum
    Ma = mp * wp.cross(xp, vp) # angular momentum

    Ee = _0
    Ek = _05 * mp * wp.dot(vp, vp)
    if pp.cons_type == Constitutive.stvk:
        # compute strain energy density
        if pp.mate_type == Material.snow13:
            hardening = mu_lam_kappa[2]
            dlogJ = ptc_dlogJ[p]
            h = snow13_hardening(dlogJ=dlogJ, hardening=hardening)
            mu *= h; lam *= h
        U, sig, V = wp.svd3(Fp)
        log_sig = vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
        psi = mu * wp.dot(log_sig, log_sig) + _05 * lam * (log_sig[0] + log_sig[1] + log_sig[2])**_2
        Ee = ptc_vol0[p] * psi
    else:
        Jp = Fp[0,0] * Fp[1,1] * Fp[2,2]
        psi = _05 * kappa * (Jp - _1)**_2
        Ee = ptc_vol0[p] * psi

    wp.atomic_add(kinetic_energy, 0, Ek)
    wp.atomic_add(elastic_energy, 0, Ee)
    wp.atomic_add(linear_momentum, 0, Ml)
    wp.atomic_add(angular_momentum, 0, Ma)

from ..d3.helper_misc import in_region
from ...sp_grid import unlin_IJK, local2global, local_coords_in_block, lin_JK

@wp.kernel
def linesearch_update_velocity(
    n_active_nodes: wp.array(dtype=int, ndim=1),
    ndof2bijk: wp.array(dtype=wp.vec2i, ndim=1),
    search_dv: wp.array(dtype=vec3, ndim=1),
    grid_v_ls: wp.array(dtype=vec3, ndim=4),
    grid_v_it: wp.array(dtype=vec3, ndim=4),
    alpha: real,
):
    dof = wp.tid()
    if dof >= n_active_nodes[0]: return
    bid, local_n1d = ndof2bijk[dof].x, ndof2bijk[dof].y
    li, lj, lk = unlin_IJK(local_n1d)
    grid_v_ls[bid, li, lj, lk] = grid_v_it[bid, li, lj, lk] + alpha * search_dv[dof]


@wp.kernel
def node_psi(
    n_active_nodes: wp.array(dtype=int),
    ndof2bijk: wp.array(dtype=wp.vec2i, ndim=1),
    block_xyz_by_id: wp.array(dtype=wp.vec3i, ndim=1),
    grid_m: wp.array(dtype=real, ndim=4),
    grid_v: wp.array(dtype=vec3, ndim=4),
    grid_v_it: wp.array(dtype=vec3, ndim=4),
    grid_size: wp.vec3i,
    gravity: vec3,
    dt: real,
    psi: wp.array(dtype=real, ndim=1),
):
    dof = wp.tid()
    if dof >= n_active_nodes[0]:
        psi[dof] = _0
        return
    bid, local_n1d = ndof2bijk[dof].x, ndof2bijk[dof].y
    node = local2global(block_xyz_by_id[bid], local_n1d)
    if not in_region(node, grid_size):
        psi[dof] = _0
        return
    lijk = local_coords_in_block(node.x, node.y, node.z)
    li, lj, lk = lijk.x, lijk.y, lijk.z
    mi = grid_m[bid, li, lj, lk]
    v_it = grid_v_it[bid, li, lj, lk]
    v = grid_v[bid, li, lj, lk]
    inertia = v_it - v
    psi[dof] = _05 * mi * wp.dot(inertia, inertia) - dt * mi * wp.dot(gravity, v_it)


@wp.kernel
def center_psi(
    n_active_centers: wp.array(dtype=int),
    cdof2bijk: wp.array(dtype=wp.vec2i, ndim=1),
    block_xyz_by_id: wp.array(dtype=wp.vec3i, ndim=1),
    center_vol: wp.array(dtype=real, ndim=4),
    center_quadrature_scratch: wp.array(dtype=QuadratureScratch3D, ndim=2),
    psi_params: wp.array(dtype=MaterialParams),
    center_size: wp.vec3i,
    dt: real,
    psi: wp.array(dtype=real, ndim=2),
):
    psi_k, cdof = wp.tid()
    if cdof >= n_active_centers[0]:
        psi[psi_k, cdof] = _0
        return
    cbid, local_c1d = cdof2bijk[cdof].x, cdof2bijk[cdof].y
    center = local2global(block_xyz_by_id[cbid], local_c1d)
    if not in_region(center, center_size):
        psi[psi_k, cdof] = _0
        return
    lcijk = local_coords_in_block(center.x, center.y, center.z)
    lci, lcj, lck = lcijk.x, lcijk.y, lcijk.z
    lcjk = lin_JK(lcj, lck)

    if center_vol[psi_k, cbid, lci, lcjk] <= real(0): 
        psi[psi_k, cdof] = _0
        return

    qs = center_quadrature_scratch[psi_k, cdof]
    vol = center_vol[psi_k, cbid, lci, lcjk]
    sig = qs.Sigma
    pp = psi_params[psi_k]
    mu_lam_kappa = pp.mu_lam_kappa
    mu, lam, kappa = mu_lam_kappa[0], mu_lam_kappa[1], mu_lam_kappa[2]
    
    if pp.cons_type == Constitutive.stvk:
        log_sig = vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
        psi_0 = mu * wp.dot(log_sig, log_sig) + _05 * lam * (log_sig[0] + log_sig[1] + log_sig[2])**_2
        psi[psi_k, cdof] = vol * psi_0

    elif pp.cons_type == Constitutive.pressure:
        J = sig[0] * sig[1] * sig[2]
        psi_0 = _05 * (J - _1)**_2 * kappa
        psi[psi_k, cdof] = vol * psi_0

@wp.kernel
def particle_psi(
    n_ptc_wp: wp.array(dtype=int),
    ptc_k: wp.array(dtype=int, ndim=1),
    ptc_vol0: wp.array(dtype=real, ndim=1),
    particle_quadrature_scratch: wp.array(dtype=QuadratureScratch3D, ndim=1),
    psi_params: wp.array(dtype=MaterialParams),
    dt: real,
    psi: wp.array(dtype=real, ndim=1),
):
    p = wp.tid()
    if p >= n_ptc_wp[0]:
        psi[p] = _0
        return

    psi_k = ptc_k[p]

    qs = particle_quadrature_scratch[p]
    vol = ptc_vol0[p]
    sig = qs.Sigma
    pp = psi_params[psi_k]
    mu_lam_kappa = pp.mu_lam_kappa
    mu, lam, kappa = mu_lam_kappa[0], mu_lam_kappa[1], mu_lam_kappa[2]
    
    if pp.cons_type == Constitutive.stvk:
        log_sig = vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
        psi_0 = mu * wp.dot(log_sig, log_sig) + _05 * lam * (log_sig[0] + log_sig[1] + log_sig[2])**_2
        psi[p] = vol * psi_0

    elif pp.cons_type == Constitutive.pressure:
        J = sig[0] * sig[1] * sig[2]
        psi_0 = _05 * (J - _1)**_2 * kappa
        psi[p] = vol * psi_0
import warp as wp
import os as _os
WATER_NO_TENSION = wp.constant(int(_os.environ.get("LIQ_NO_TENSION", "1")))
from ...types import *
from ..constant import (
    _05, _0, _1, _2, _3, _4, _n1,
    s_min, dPdF_use_project, dPdF_eps_small, dPdF_eps_pd,
)
from .helper_misc import rowmajor_index

@wp.func
def make_pd_shift_2x2(M: mat22, eps_pd: real) -> mat22:
    # Symmetrize, then shift spectrum so λ_min >= eps_pd
    a = M[0, 0]
    c = M[1, 1]
    b = _05 * (M[0, 1] + M[1, 0])
    t = a + c
    delta = wp.sqrt((a - c) * (a - c) + _4 * b * b)
    lam_min = _05 * (t - delta)
    s = wp.max(_0, eps_pd - lam_min)
    N = mat22(a + s, b,
              b,     c + s)
    return N


@wp.func
def make_pd_shift_3x3(M: mat33, eps_pd: real) -> mat33:
    # symmetrize
    a00 = M[0, 0]
    a11 = M[1, 1]
    a22 = M[2, 2]
    a01 = _05 * (M[0, 1] + M[1, 0])
    a02 = _05 * (M[0, 2] + M[2, 0])
    a12 = _05 * (M[1, 2] + M[2, 1])

    # trace / 3
    tr = a00 + a11 + a22
    q = tr / _3

    # B = A - qI
    b00 = a00 - q
    b11 = a11 - q
    b22 = a22 - q

    # p = sqrt(tr(B^2)/6)
    p2 = (b00*b00 + b11*b11 + b22*b22 + _2*(a01*a01 + a02*a02 + a12*a12)) / real(6)
    p  = wp.sqrt(wp.max(p2, _0))

    lam_min = q
    if p >= real(1e-12):
        inv_p = _1 / p
        c00 = b00 * inv_p
        c11 = b11 * inv_p
        c22 = b22 * inv_p
        c01 = a01 * inv_p
        c02 = a02 * inv_p
        c12 = a12 * inv_p

        detC = (
            c00 * (c11 * c22 - c12 * c12)
          - c01 * (c01 * c22 - c12 * c02)
          + c02 * (c01 * c12 - c11 * c02)
        )

        r = wp.clamp(detC * _05, _n1, _1)
        phi = wp.acos(r) / _3

        # smallest eigen: q + 2p cos(phi + 2pi/3)
        lam_min = q + _2 * p * wp.cos(phi + _2 * real(wp.pi) / _3)

    s = wp.max(_0, eps_pd - lam_min)

    return mat33(
        a00 + s, a01,     a02,
        a01,     a11 + s, a12,
        a02,     a12,     a22 + s
    )


@wp.func
def diff_interlock_log_over_diff(x: real, y: real, logy: real, eps: real) -> real:
    # diff_interlock_log_over_diff(const T x, const T y, const T logy, const T eps)
    #   return logy - y * diff_log_over_diff(x, y, eps);
    # diff_log_over_diff(const T x, const T y, const T eps)
    #   T p = x / y - 1;
    #   return log_1px_over_x(p, eps) / y;
    # log_1px_over_x(const T x, const T eps)
    #   if (std::fabs(x) < eps) return (T)1;
    #   else return std::log1p(x) / x;
    p = x / y - _1
    log_1px_over_x = _1 if wp.abs(p) < eps else wp.log(_1 + p) / p
    return logy - log_1px_over_x

@wp.func # from Ziran2019 StvkWithHenckyIsotropic
def dPdF_StVK_Hencky_3D_analytic(
    F: mat33,
    mu: real,
    lam: real,
    use_project: wp.int32,
    s_min: real,
    eps_small: real,
    eps_pd: real,
) -> mat99:
    # SVD with spectral clamp
    U, sigma, V = wp.svd3(F)  # sigma: vec3
    s0 = wp.max(s_min, sigma[0])
    s1 = wp.max(s_min, sigma[1])
    s2 = wp.max(s_min, sigma[2])
    log0 = wp.log(s0)
    log1 = wp.log(s1)
    log2 = wp.log(s2)

    g = _2 * mu + lam
    sum_log = log0 + log1 + log2
    prod01 = s0 * s1
    prod02 = s0 * s2
    prod12 = s1 * s2

    psi0 = (_2 * mu * log0 + lam * sum_log) / s0
    psi1 = (_2 * mu * log1 + lam * sum_log) / s1
    psi2 = (_2 * mu * log2 + lam * sum_log) / s2
    psi00 = (g * (_1 - log0) - lam * (log1 + log2)) / (s0 * s0)
    psi11 = (g * (_1 - log1) - lam * (log0 + log2)) / (s1 * s1)
    psi22 = (g * (_1 - log2) - lam * (log0 + log1)) / (s2 * s2)
    psi01 = lam / prod01
    psi02 = lam / prod02
    psi12 = lam / prod12

    # (psiA-psiB)/(sigmaA-sigmaB)
    m01 = -(lam * sum_log + _2 * mu * diff_interlock_log_over_diff(s0, s1, log1, eps_small)) / prod01
    m02 = -(lam * sum_log + _2 * mu * diff_interlock_log_over_diff(s0, s2, log2, eps_small)) / prod02
    m12 = -(lam * sum_log + _2 * mu * diff_interlock_log_over_diff(s1, s2, log2, eps_small)) / prod12

    # (psiA+psiB)/(sigmaA+sigmaB)
    p01 = (psi0 + psi1) / wp.max(s0 + s1, eps_small)
    p02 = (psi0 + psi2) / wp.max(s0 + s2, eps_small)
    p12 = (psi1 + psi2) / wp.max(s1 + s2, eps_small)

    Aij = mat33(
        psi00, psi01, psi02,
        psi01, psi11, psi12,
        psi02, psi12, psi22)
    
    B01_0 = _05 * (m01 + p01)
    B01_1 = _05 * (m01 - p01)
    B02_0 = _05 * (m02 + p02)
    B02_1 = _05 * (m02 - p02)
    B12_0 = _05 * (m12 + p12)
    B12_1 = _05 * (m12 - p12)

    B01 = mat22(B01_0, B01_1,
                B01_1, B01_0)
    B02 = mat22(B02_0, B02_1,
                B02_1, B02_0)
    B12 = mat22(B12_0, B12_1,
                B12_1, B12_0)
    
    if use_project == 1:
        Aij = make_pd_shift_3x3(Aij, eps_pd)
        B01 = make_pd_shift_2x2(B01, eps_pd)
        B02 = make_pd_shift_2x2(B02, eps_pd)
        B12 = make_pd_shift_2x2(B12, eps_pd)

    # Fill 4x4 dPdF in our ROW-MAJOR flattening (00,01,10,11)
    H = mat99(_0)  # zeros

    A00 = Aij[0, 0]; A01 = Aij[0, 1]; A02 = Aij[0, 2]
    A10 = Aij[1, 0]; A11 = Aij[1, 1]; A12 = Aij[1, 2]
    A20 = Aij[2, 0]; A21 = Aij[2, 1]; A22 = Aij[2, 2]

    # Bij cache
    B01_00 = B01[0, 0]; B01_01 = B01[0, 1]; B01_10 = B01[1, 0]; B01_11 = B01[1, 1]
    B02_00 = B02[0, 0]; B02_01 = B02[0, 1]; B02_10 = B02[1, 0]; B02_11 = B02[1, 1]
    B12_00 = B12[0, 0]; B12_01 = B12[0, 1]; B12_10 = B12[1, 0]; B12_11 = B12[1, 1]

    for i in range(3):
        Ui0 = U[i, 0]; Ui1 = U[i, 1]; Ui2 = U[i, 2]
        for j in range(3):
            Vj0 = V[j, 0]; Vj1 = V[j, 1]; Vj2 = V[j, 2]
            ij = rowmajor_index(i, j)

            # only calculate T value once for each (i,j)
            T00 = Ui0 * Vj0; T01 = Ui0 * Vj1; T02 = Ui0 * Vj2
            T10 = Ui1 * Vj0; T11 = Ui1 * Vj1; T12 = Ui1 * Vj2
            T20 = Ui2 * Vj0; T21 = Ui2 * Vj1; T22 = Ui2 * Vj2

            for r in range(3):
                Ur0 = U[r, 0]; Ur1 = U[r, 1]; Ur2 = U[r, 2]
                for s in range(3):
                    Vs0 = V[s, 0]; Vs1 = V[s, 1]; Vs2 = V[s, 2]
                    rs = rowmajor_index(r, s)

                    # calculate S (r,s), 9 elements
                    S00 = Ur0 * Vs0; S01 = Ur0 * Vs1; S02 = Ur0 * Vs2
                    S10 = Ur1 * Vs0; S11 = Ur1 * Vs1; S12 = Ur1 * Vs2
                    S20 = Ur2 * Vs0; S21 = Ur2 * Vs1; S22 = Ur2 * Vs2

                    # ------------------------------
                    # 1) diagonal contribution: only T00,T11,T22 and S00,S11,S22
                    #    term_diag = sum_{a,b} A_ab * (U_i a V_j a) * (U_r b V_s b)
                    #            = sum_{a,b} A_ab * T_{a,a} * S_{b,b}
                    # ------------------------------
                    term = _0
                    term += T00 * (A00 * S00 + A01 * S11 + A02 * S22)
                    term += T11 * (A10 * S00 + A11 * S11 + A12 * S22)
                    term += T22 * (A20 * S00 + A21 * S11 + A22 * S22)

                    # ------------------------------
                    # 2) pair (0,1): basis {U0 V1, U1 V0} -> {T01, T10}
                    # ------------------------------
                    term += B01_00 * T01 * S01 + B01_01 * T01 * S10 + B01_10 * T10 * S01 + B01_11 * T10 * S10

                    # 3) pair (0,2): basis {U0 V2, U2 V0} -> {T02, T20}
                    term += B02_00 * T02 * S02 + B02_01 * T02 * S20 + B02_10 * T20 * S02 + B02_11 * T20 * S20

                    # 4) pair (1,2): basis {U1 V2, U2 V1} -> {T12, T21}
                    term += B12_00 * T12 * S12 + B12_01 * T12 * S21 + B12_10 * T21 * S12 + B12_11 * T21 * S21

                    H[ij, rs] = term

    return H


###############################################################################
# Von Mises plasticity without hardening
###############################################################################
@wp.func
def von_mises_apply(F: mat33, mu: real, yield_stress: real, s_min: real) -> Tuple[mat33, vec3, mat33]:
    dim = _3
    U, sigma, V = wp.svd3(F)  # sigma: vec3
    sigma = vec3(
        wp.max(sigma[0], s_min),
        wp.max(sigma[1], s_min),
        wp.max(sigma[2], s_min)
    )
    sig = sigma               # vec3(s0, s1, s2)
    epsilon = vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    tr_eps = epsilon[0] + epsilon[1] + epsilon[2]
    epsilon_hat = epsilon - vec3(tr_eps / dim)
    s_trial = epsilon_hat * (_2 * mu)
    s_trial_norm = wp.sqrt(wp.dot(s_trial, s_trial) + real(1.0e-8))
    y = s_trial_norm - wp.sqrt(_2 / _3) * yield_stress
    sig_out = sig
    if y > _0:
        s_new_norm = s_trial_norm - y
        scale = s_new_norm / s_trial_norm
        s_new = vec3(s_trial[0] * scale, s_trial[1] * scale, s_trial[2] * scale)
        sig_out = vec3(
            wp.exp(s_new[0] / (_2 * mu) + tr_eps / dim),
            wp.exp(s_new[1] / (_2 * mu) + tr_eps / dim),
            wp.exp(s_new[2] / (_2 * mu) + tr_eps / dim),
        )
    # sig_diag = wp.diag(sig_out)
    return U, sig_out, V


###############################################################################
# Snow Plasticity from the 2013 Snow Paper
###############################################################################
@wp.func
def snow13_return_mapping(
    F: mat33,
    theta_c: real,
    theta_s: real,
) -> Tuple[mat33, vec3, mat33]:
    # returns: U, sig_clamped, V, logJp_new, mu_eff, lam_eff, dlogJ_step
    U, sig, V = wp.svd3(F)
    sig = vec3(wp.max(sig[0], s_min), wp.max(sig[1], s_min), wp.max(sig[2], s_min))

    smin = _1 - theta_c; smax = _1 + theta_s

    s0 = sig[0]; s1 = sig[1]; s2 = sig[2]
    s0c = wp.min(wp.max(s0, smin), smax)
    s1c = wp.min(wp.max(s1, smin), smax)
    s2c = wp.min(wp.max(s2, smin), smax)
    sig_out = vec3(s0c, s1c, s2c)

    return U, sig_out, V
@wp.func
def snow13_hardening(
    dlogJ: real,
    hardening: real,
) -> real:
    h = wp.exp(hardening * (_1 - wp.exp(dlogJ)))

    # Clamp to the common range used by Snow 2013.
    h_min = wp.exp(real(-10.0))
    h_max = wp.exp(real( 10.0))
    h = wp.min(wp.max(h, h_min), h_max)

    return h


###########################################################################
# Non-associative Cam Clay plasticity from the CD-MPM paper
###########################################################################
@wp.func
def sinh(x: real) -> real:
    return (wp.exp(x)-wp.exp(-x))/_2

@wp.func
def NonAssoc_CamClay_return_mapping(
    F: mat33, mu: real, la: real, dlogJ: real, M: real, beta: real, xi: real
) -> Tuple[mat33, vec3, mat33, real]:
    kappa = real(.6666666666) * mu + la
    U, sigma, V = wp.svd3(F)
    Se = vec3(
        wp.max(sigma[0], real(0.05)),
        wp.max(sigma[1], real(0.05)),
        wp.max(sigma[2], real(0.05))
    )
    p0 = kappa * (real(1e-5) + sinh(xi * wp.max(-dlogJ, _0))) # hardening modifies p0 using the current logJp
    a = (_1 + _2 * beta) * _3 / _2; b = beta * p0; c = p0
    M2 = M*M
    Je = Se[0] * Se[1] * Se[2]
    Se2 = vec3(Se[0]*Se[0], Se[1]*Se[1], Se[2]*Se[2])
    Se2_sum = Se2[0] + Se2[1] + Se2[2]
    deviatoric_Se2 = Se2 - vec3(Se2_sum / _3)
    s_hat_trial = mu * wp.pow(Je, real(-2.0) / _3) * deviatoric_Se2
    Uprime = kappa / _2 * (Je - _1 / Je)
    p_trial = -Uprime * Je
    if p_trial > c:
        Je_new = wp.sqrt(real(-2) * c / kappa + _1)
        Se = vec3(wp.pow(Je_new, _1/_3))
        dlogJ += -wp.log(Je_new) + wp.log(Je)
    elif p_trial < -b:
        Je_new = wp.sqrt(_2 * b / kappa + _1)
        Se = vec3(wp.pow(Je_new, _1/_3))
        dlogJ += -wp.log(Je_new) + wp.log(Je)
    else:
        k = wp.sqrt(-M2 * (p_trial + b) * (p_trial - c) / a)
        s_hat_trial_norm = wp.sqrt(wp.dot(s_hat_trial, s_hat_trial) + real(1e-12))
        y = a * s_hat_trial_norm * s_hat_trial_norm  + M2 * (p_trial + b) * (p_trial - c)
        if y > real(1e-4):
            pc = (_1 - beta) * p0 / _2
            if p0 > real(1e-4) and p_trial < p0 - (real(1e-4)) and p_trial > -beta * p0 + (real(1e-4)):
                aa = M2 * (p_trial - pc)**_2 / (a * (s_hat_trial_norm**_2))
                dd = _1 + aa
                ff = aa * beta * p0 - aa * p0 - _2 * pc
                gg = (pc)**_2 - aa * beta * (p0)**_2
                zz = wp.sqrt(wp.abs((ff)**_2 - _4 * dd * gg))
                p1 = (-ff + zz) / (_2 * dd)
                p2 = (-ff - zz) / (_2 * dd)
                p_fake = p1 if (p_trial - pc) * (p1 - pc) > _0 else p2
                Je_new_fake = wp.sqrt(wp.abs(real(-2) * p_fake / kappa + _1))
                if Je_new_fake > real(1e-4):
                    dlogJ += -wp.log(Je_new_fake) + wp.log(Je)
            be_new = k / mu * wp.pow(Je, _2 / _3) * s_hat_trial / s_hat_trial_norm + _1 / _3 * vec3(Se2_sum)
            Se = vec3(wp.sqrt(be_new[0]), wp.sqrt(be_new[1]), wp.sqrt(be_new[2]))
    return U, Se, V, dlogJ


################################
# Stvk Hencky Elasticity
################################
# ANCHOR: stvk
@wp.func
def StVK_Hencky_PK1_3D(F: mat33, mu: real, lam: real, s_min: real) -> mat33:
    # F = U diag(sigma) V^T
    U, sigma, V = wp.svd3(F)         # sigma: vec3
    sigma = vec3(
        wp.max(sigma[0], s_min),
        wp.max(sigma[1], s_min),
        wp.max(sigma[2], s_min)
    )
    inv_sig = vec3(_1 / sigma[0], _1 / sigma[1], _1 / sigma[2])
    e = vec3(wp.log(sigma[0]), wp.log(sigma[1]), wp.log(sigma[2]))
    m = _2 * mu * vec3(inv_sig[0] * e[0], inv_sig[1] * e[1], inv_sig[2] * e[2]) + lam * (e[0] + e[1] + e[2]) * inv_sig
    M = wp.diag(m)
    Pk1 = U @ M @ wp.transpose(V)
    return Pk1


@wp.func
def StVK_Hencky_PK1_3D_svdfree(U: mat33, sigma: vec3, V:mat33, mu: real, lam: real, s_min: real) -> mat33:
    sigma = vec3(
        wp.max(sigma[0], s_min),
        wp.max(sigma[1], s_min),
        wp.max(sigma[2], s_min)
    )
    inv_sig = vec3(_1 / sigma[0], _1 / sigma[1], _1 / sigma[2])
    e = vec3(wp.log(sigma[0]), wp.log(sigma[1]), wp.log(sigma[2]))
    m = _2 * mu * vec3(inv_sig[0] * e[0], inv_sig[1] * e[1], inv_sig[2] * e[2]) + lam * (e[0] + e[1] + e[2]) * inv_sig
    M = wp.diag(m)
    Pk1 = U @ M @ wp.transpose(V)
    return Pk1

@wp.func
def StVK_Hencky_tau_3D(F: mat33, mu: real, lam: real, s_min: real) -> mat33:
    Pk1 = StVK_Hencky_PK1_3D(F, mu, lam, s_min)
    tau = Pk1 @ wp.transpose(F)   # Kirchhoff stress
    return tau

@wp.func
def dPdF_StVK_Hencky_3D_analytic_v2(
    sigma: vec3,
    mu: real,
    lam: real,
    use_project: wp.int32,
    s_min: real,
    eps_small: real,
    eps_pd: real,
) -> Tuple[vec6, vec6]:
    s0 = wp.max(s_min, sigma[0])
    s1 = wp.max(s_min, sigma[1])
    s2 = wp.max(s_min, sigma[2])
    log0 = wp.log(s0)
    log1 = wp.log(s1)
    log2 = wp.log(s2)

    g = _2 * mu + lam
    sum_log = log0 + log1 + log2
    prod01 = s0 * s1
    prod02 = s0 * s2
    prod12 = s1 * s2

    psi0 = (_2 * mu * log0 + lam * sum_log) / s0
    psi1 = (_2 * mu * log1 + lam * sum_log) / s1
    psi2 = (_2 * mu * log2 + lam * sum_log) / s2
    psi00 = (g * (_1 - log0) - lam * (log1 + log2)) / (s0 * s0)
    psi11 = (g * (_1 - log1) - lam * (log0 + log2)) / (s1 * s1)
    psi22 = (g * (_1 - log2) - lam * (log0 + log1)) / (s2 * s2)
    psi01 = lam / prod01
    psi02 = lam / prod02
    psi12 = lam / prod12

    # (psiA-psiB)/(sigmaA-sigmaB)
    m01 = -(lam * sum_log + _2 * mu * diff_interlock_log_over_diff(s0, s1, log1, eps_small)) / prod01
    m02 = -(lam * sum_log + _2 * mu * diff_interlock_log_over_diff(s0, s2, log2, eps_small)) / prod02
    m12 = -(lam * sum_log + _2 * mu * diff_interlock_log_over_diff(s1, s2, log2, eps_small)) / prod12

    # (psiA+psiB)/(sigmaA+sigmaB)
    p01 = (psi0 + psi1) / wp.max(s0 + s1, eps_small)
    p02 = (psi0 + psi2) / wp.max(s0 + s2, eps_small)
    p12 = (psi1 + psi2) / wp.max(s1 + s2, eps_small)

    Aij = mat33(
        psi00, psi01, psi02,
        psi01, psi11, psi12,
        psi02, psi12, psi22)
    
    B01_0 = _05 * (m01 + p01)
    B01_1 = _05 * (m01 - p01)
    B02_0 = _05 * (m02 + p02)
    B02_1 = _05 * (m02 - p02)
    B12_0 = _05 * (m12 + p12)
    B12_1 = _05 * (m12 - p12)

    B01 = mat22(B01_0, B01_1,
                B01_1, B01_0)
    B02 = mat22(B02_0, B02_1,
                B02_1, B02_0)
    B12 = mat22(B12_0, B12_1,
                B12_1, B12_0)
    
    if use_project == 1:
        Aij = make_pd_shift_3x3(Aij, eps_pd)
        B01 = make_pd_shift_2x2(B01, eps_pd)
        B02 = make_pd_shift_2x2(B02, eps_pd)
        B12 = make_pd_shift_2x2(B12, eps_pd)

    return (
        vec6(
            Aij[0, 0],    # psi00: real
            Aij[0, 1],    # psi01: real
            Aij[0, 2],    # psi02: real
            Aij[1, 1],    # psi11: real
            Aij[1, 2],    # psi12: real
            Aij[2, 2],    # psi22: real
        ),
        vec6(
            B01[0, 0],
            B01[0, 1],
            B02[0, 0],
            B02[0, 1],
            B12[0, 0],
            B12[0, 1],
        )
    )

@wp.func # from Ziran2019
def StVK_Hencky_dPK1_3D(
    dF: mat33,
    s_U: mat33,
    s_V: mat33,
    Aij: vec6,
    B012: vec6,
) -> mat33:
    D = wp.transpose(s_U) @ dF @ s_V
    K = mat33(_0)
    psi00, psi01, psi02 = Aij[0], Aij[1], Aij[2]
    psi11, psi12, psi22 = Aij[3], Aij[4], Aij[5]
    _05_m01_p_p01, _05_m01_s_p01 = B012[0], B012[1]
    _05_m02_p_p02, _05_m02_s_p02 = B012[2], B012[3]
    _05_m12_p_p12, _05_m12_s_p12 = B012[4], B012[5]
    K[0, 0] = psi00 * D[0, 0] + psi01 * D[1, 1] + psi02 * D[2, 2]
    K[1, 1] = psi01 * D[0, 0] + psi11 * D[1, 1] + psi12 * D[2, 2]
    K[2, 2] = psi02 * D[0, 0] + psi12 * D[1, 1] + psi22 * D[2, 2]
    K[0, 1] = _05_m01_p_p01 * D[0, 1] + _05_m01_s_p01 * D[1, 0]
    K[1, 0] = _05_m01_s_p01 * D[0, 1] + _05_m01_p_p01 * D[1, 0]
    K[0, 2] = _05_m02_p_p02 * D[0, 2] + _05_m02_s_p02 * D[2, 0]
    K[2, 0] = _05_m02_s_p02 * D[0, 2] + _05_m02_p_p02 * D[2, 0]
    K[1, 2] = _05_m12_p_p12 * D[1, 2] + _05_m12_s_p12 * D[2, 1]
    K[2, 1] = _05_m12_s_p12 * D[1, 2] + _05_m12_p_p12 * D[2, 1]
    dP = s_U @ K @ wp.transpose(s_V)
    return dP

@wp.func
def water_PK1_3D(J: real, kappa: real) -> real:
    return kappa * (J - _1)

@wp.func
def water_tau_3D(J: real, kappa: real) -> real:
    # StiffGIPC patch: optional no-tension (tau = 0 for J > 1); env LIQ_NO_TENSION=0
    # restores the paper's kappa J (J-1) (tension allowed)
    if WATER_NO_TENSION == 1:
        return kappa * J * wp.min(J - _1, _0)
    return kappa * J * (J - _1)
    
@wp.func
def Drucker_Prager_return_mapping(
    F: mat33,
    cohesion: real,
    # dlogJ: real,
    mu: real,
    lam: real,
    friction_angle: real,
) -> Tuple[mat33, vec3, mat33]:
    dim = _3
    sin_phi = wp.sin((friction_angle / real(180.0)) * real(wp.pi))
    friction_alpha = wp.sqrt(_2 / _3) * _2 * sin_phi / (_3 - sin_phi)
    U, sigma, V = wp.svd3(F)   # sigma: vec3
    # clamp stretches
    epsilon = vec3(
        wp.log(wp.max(sigma.x, real(0.05))),
        wp.log(wp.max(sigma.y, real(0.05))),
        wp.log(wp.max(sigma.z, real(0.05))))
    # volume correction treatment
    # epsilon += vec3(dlogJ / dim)
    trace_epsilon = epsilon[0] + epsilon[1] + epsilon[2]
    shifted_trace = trace_epsilon - cohesion * dim
    if shifted_trace >= _0:
        epsilon = vec3(cohesion)
    else:
        epsilon_hat = epsilon - vec3(trace_epsilon / dim)
        epsilon_hat_norm = wp.sqrt(wp.dot(epsilon_hat, epsilon_hat) + real(1.0e-8))
        delta_gamma = epsilon_hat_norm + (dim * lam + _2 * mu) / (_2 * mu) * shifted_trace * friction_alpha
        epsilon -= wp.max(delta_gamma, _0) / epsilon_hat_norm * epsilon_hat
    sig_out = vec3(wp.exp(epsilon[0]), wp.exp(epsilon[1]), wp.exp(epsilon[2]))
    return U, sig_out, V

@wp.func
def foam_return_mapping(
    F: mat33, mu: real, yield_stress: real,
    plastic_pow: real, plastic_viscosity: real,
    dt: real,
) -> Tuple[mat33, vec3, mat33]:
    dim = _3
    U, sigma, V = wp.svd3(F)   # sigma: vec3
    # clamp stretches
    epsilon = vec3(
        wp.log(wp.max(sigma.x, real(0.05))),
        wp.log(wp.max(sigma.y, real(0.05))),
        wp.log(wp.max(sigma.z, real(0.05))))
    trace_epsilon = epsilon[0] + epsilon[1] + epsilon[2]
    epsilon_hat = epsilon - vec3(trace_epsilon / dim)
    s_trial =  epsilon_hat * (_2 * mu)
    s_trial_norm = wp.sqrt(wp.dot(s_trial, s_trial) + real(1.0e-8))
    y = s_trial_norm - wp.sqrt(_2/_3) * yield_stress
    sig_out = sigma
    if y > _0:
        b_trial_sum = wp.dot(sigma, sigma)
        mu_hat = mu * b_trial_sum / dim * y ** (_1 / plastic_pow - _1)
        s_new_norm = s_trial_norm - y / (_1 + plastic_viscosity / (_2 * mu_hat * dt))
        s_new = (s_new_norm / s_trial_norm) * s_trial
        H = s_new / (_2 * mu) + vec3(trace_epsilon / dim)
        sig_out = vec3(wp.exp(H[0]), wp.exp(H[1]), wp.exp(H[2]))
    return U, sig_out, V

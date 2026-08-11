#pragma once
#include "contact/barrier_rank.h"
#include "FrictionUtils.cuh"
// ============================================================================
// energy/02_contact_energy_device.inl — contact-term ENERGY device functions
// (v0.8.6 energy separation E1d): __cal_Barrier_energy (frozen smooth
// branches travel VERBATIM, still dead, still frozen), __cal_Friction_energy
// (+_gd) and the _pair_mu helper. Included at the old module-01 position —
// consumers are the term reductions (energy/15/16) and the gradient kernels;
// definition must precede module 05's gradient kernels.
// ============================================================================
__device__ inline double __cal_Barrier_energy(const double3* _vertexes,
                                       const double3* _rest_vertexes,
                                       int4           MMCVIDI,
                                       double         _Kappa,
                                       double         _dHat)
{
    double dHat_sqrt = sqrt(_dHat);
    double dHat      = _dHat;
    double Kappa     = _Kappa;
    if(MMCVIDI.x >= 0)
    {
        if(MMCVIDI.w >= 0)
        {
            double dis;
            _d_EE(_vertexes[MMCVIDI.x],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            double I5 = dis / dHat;

            double lenE = (dis - dHat);
#if (RANK == 1)
            return -Kappa * lenE * lenE * log(I5);
#elif (RANK == 2)
            return Kappa * lenE * lenE * log(I5) * log(I5);
#elif (RANK == 3)
            return -Kappa * lenE * lenE * log(I5) * log(I5) * log(I5);
#elif (RANK == 4)
            return Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5);
#elif (RANK == 5)
            return -Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5) * log(I5);
#elif (RANK == 6)
            return Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5)
                   * log(I5) * log(I5);
#endif
        }
        else
        {
            //return 0;
            MMCVIDI.w = -MMCVIDI.w - 1;
            double3 v0 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.x]);
            double3 v1 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.z]);
            double c = __GEIGEN__::__norm(__GEIGEN__::__v_vec_cross(v0, v1)) /*/ __GEIGEN__::__norm(v0)*/;
            double I1 = c * c;
            if(I1 == 0)
                return 0;
            double dis;
            _d_EE(_vertexes[MMCVIDI.x],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            double I2    = dis / dHat;
            double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                        _rest_vertexes[MMCVIDI.y],
                                        _rest_vertexes[MMCVIDI.z],
                                        _rest_vertexes[MMCVIDI.w]);
#if (RANK == 1)
            double Energy = Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                            * -(dHat - dHat * I2) * (dHat - dHat * I2) * log(I2);
#elif (RANK == 2)
            double Energy =
                Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2) * log(I2);
#elif (RANK == 4)
            double Energy = Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                            * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2)
                            * log(I2) * log(I2) * log(I2);
#elif (RANK == 6)
            double Energy = Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                            * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2)
                            * log(I2) * log(I2) * log(I2) * log(I2) * log(I2);
#endif
            if(Energy < 0)
                printf("I am pee\n");
            return Energy;
        }
    }
    else
    {
        int v0I = -MMCVIDI.x - 1;
        if(MMCVIDI.z < 0)
        {
            if(MMCVIDI.y < 0)
            {
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.z = -MMCVIDI.z - 1;
                MMCVIDI.w = -MMCVIDI.w - 1;
                MMCVIDI.x = v0I;

                double3 v0 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.x]);
                double3 v1 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.y]);
                double c = __GEIGEN__::__norm(__GEIGEN__::__v_vec_cross(v0, v1)) /*/ __GEIGEN__::__norm(v0)*/;
                double I1 = c * c;
                if(I1 == 0)
                    return 0;
                double dis;
                _d_PP(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], dis);
                double I2    = dis / dHat;
                double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                            _rest_vertexes[MMCVIDI.z],
                                            _rest_vertexes[MMCVIDI.y],
                                            _rest_vertexes[MMCVIDI.w]);
#if (RANK == 1)
                double Energy =
                    Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                    * -(dHat - dHat * I2) * (dHat - dHat * I2) * log(I2);
#elif (RANK == 2)
                double Energy =
                    Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                    * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2) * log(I2);
#elif (RANK == 4)
                double Energy =
                    Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                    * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2)
                    * log(I2) * log(I2) * log(I2);
#elif (RANK == 6)
                double Energy =
                    Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                    * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2)
                    * log(I2) * log(I2) * log(I2) * log(I2) * log(I2);
#endif
                if(Energy < 0)
                    printf("I am pp\n");
                return Energy;
            }
            else
            {
                double dis;
                _d_PP(_vertexes[v0I], _vertexes[MMCVIDI.y], dis);
                double I5 = dis / dHat;

                double lenE = (dis - dHat);
#if (RANK == 1)
                return -Kappa * lenE * lenE * log(I5);
#elif (RANK == 2)
                return Kappa * lenE * lenE * log(I5) * log(I5);
#elif (RANK == 3)
                return -Kappa * lenE * lenE * log(I5) * log(I5) * log(I5);
#elif (RANK == 4)
                return Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5);
#elif (RANK == 5)
                return -Kappa * lenE * lenE * log(I5) * log(I5) * log(I5)
                       * log(I5) * log(I5);
#elif (RANK == 6)
                return Kappa * lenE * lenE * log(I5) * log(I5) * log(I5)
                       * log(I5) * log(I5) * log(I5);
#endif
            }
        }
        else if(MMCVIDI.w < 0)
        {
            if(MMCVIDI.y < 0)
            {
                MMCVIDI.y = -MMCVIDI.y - 1;
                //MMCVIDI.z = -MMCVIDI.z - 1;
                MMCVIDI.w = -MMCVIDI.w - 1;
                MMCVIDI.x = v0I;

                double3 v0 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.x]);
                double3 v1 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.y]);
                double c = __GEIGEN__::__norm(__GEIGEN__::__v_vec_cross(v0, v1)) /*/ __GEIGEN__::__norm(v0)*/;
                double I1 = c * c;
                if(I1 == 0)
                    return 0;
                double dis;
                _d_PE(_vertexes[MMCVIDI.x],
                      _vertexes[MMCVIDI.y],
                      _vertexes[MMCVIDI.z],
                      dis);
                double I2    = dis / dHat;
                double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                            _rest_vertexes[MMCVIDI.w],
                                            _rest_vertexes[MMCVIDI.y],
                                            _rest_vertexes[MMCVIDI.z]);
#if (RANK == 1)
                double Energy =
                    Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                    * -(dHat - dHat * I2) * (dHat - dHat * I2) * log(I2);
#elif (RANK == 2)
                double Energy =
                    Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                    * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2) * log(I2);
#elif (RANK == 4)
                double Energy =
                    Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                    * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2)
                    * log(I2) * log(I2) * log(I2);
#elif (RANK == 6)
                double Energy =
                    Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                    * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2)
                    * log(I2) * log(I2) * log(I2) * log(I2) * log(I2);
#endif
                if(Energy < 0)
                    printf("I am ppe\n");
                return Energy;
            }
            else
            {
                double dis;
                _d_PE(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dis);
                double I5 = dis / dHat;

                double lenE = (dis - dHat);
#if (RANK == 1)
                return -Kappa * lenE * lenE * log(I5);
#elif (RANK == 2)
                return Kappa * lenE * lenE * log(I5) * log(I5);
#elif (RANK == 3)
                return -Kappa * lenE * lenE * log(I5) * log(I5) * log(I5);
#elif (RANK == 4)
                return Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5);
#elif (RANK == 5)
                return -Kappa * lenE * lenE * log(I5) * log(I5) * log(I5)
                       * log(I5) * log(I5);
#elif (RANK == 6)
                return Kappa * lenE * lenE * log(I5) * log(I5) * log(I5)
                       * log(I5) * log(I5) * log(I5);
#endif
            }
        }
        else
        {
            double dis;
            _d_PT(_vertexes[v0I],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            double I5 = dis / dHat;

            double lenE = (dis - dHat);
#if (RANK == 1)
            return -Kappa * lenE * lenE * log(I5);
#elif (RANK == 2)
            return Kappa * lenE * lenE * log(I5) * log(I5);
#elif (RANK == 3)
            return -Kappa * lenE * lenE * log(I5) * log(I5) * log(I5);
#elif (RANK == 4)
            return Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5);
#elif (RANK == 5)
            return -Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5) * log(I5);
#elif (RANK == 6)
            return Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5)
                   * log(I5) * log(I5);
#endif
        }
    }
}


// ── friction energy device functions (pre-E1d gipc_modules/01 lines 552..677) ──
__device__ inline double __cal_Friction_gd_energy(const double3* _vertexes,
                                           const double3* _o_vertexes,
                                           const double3* _normal,
                                           uint32_t       gidx,
                                           double         dt,
                                           double         lastH,
                                           double         eps,
                                           double3 anchor = make_double3(0., 0., 0.))
{

    double3 normal = *_normal;
    double3 Vdiff  = __GEIGEN__::__minus(_vertexes[gidx], _o_vertexes[gidx]);
    Vdiff          = __GEIGEN__::__add(Vdiff, anchor);   // [fric-anchor]
    double3 VProj  = __GEIGEN__::__minus(
        Vdiff, __GEIGEN__::__s_vec_multiply(normal, __GEIGEN__::__v_vec_dot(Vdiff, normal)));
    double VProjMag2 = __GEIGEN__::__squaredNorm(VProj);
    if(VProjMag2 > eps * eps)
    {
        return lastH * (sqrt(VProjMag2) - eps * 0.5);
    }
    else
    {
        return lastH * VProjMag2 / eps * 0.5;
    }
}


__device__ inline double __cal_Friction_energy(const double3*         _vertexes,
                                        const double3*         _o_vertexes,
                                        int4                   MMCVIDI,
                                        double                 dt,
                                        double2                distCoord,
                                        __GEIGEN__::Matrix3x2d tanBasis,
                                        double                 lastH,
                                        double                 fricDHat,
                                        double                 eps,
                                        double3 anchor = make_double3(0., 0., 0.))
{
    double3 relDX3D;
    if(MMCVIDI.x >= 0)
    {
        if(MMCVIDI.w >= 0)
        {
            Friction::computeRelDX_EE(
                __GEIGEN__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]),
                distCoord.x,
                distCoord.y,
                relDX3D);
        }
    }
    else
    {
        int v0I = -MMCVIDI.x - 1;
        if(MMCVIDI.z < 0)
        {
            if(MMCVIDI.y >= 0)
            {
                Friction::computeRelDX_PP(
                    __GEIGEN__::__minus(_vertexes[v0I], _o_vertexes[v0I]),
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.y],
                                        _o_vertexes[MMCVIDI.y]),
                    relDX3D);
            }
        }
        else if(MMCVIDI.w < 0)
        {
            if(MMCVIDI.y >= 0)
            {
                Friction::computeRelDX_PE(
                    __GEIGEN__::__minus(_vertexes[v0I], _o_vertexes[v0I]),
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.y],
                                        _o_vertexes[MMCVIDI.y]),
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.z],
                                        _o_vertexes[MMCVIDI.z]),
                    distCoord.x,
                    relDX3D);
            }
        }
        else
        {
            Friction::computeRelDX_PT(
                __GEIGEN__::__minus(_vertexes[v0I], _o_vertexes[v0I]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]),
                distCoord.x,
                distCoord.y,
                relDX3D);
        }
    }
    relDX3D = __GEIGEN__::__add(relDX3D, anchor);   // [fric-anchor]
    __GEIGEN__::Matrix2x3d tB_T = __GEIGEN__::__Transpose3x2(tanBasis);
    double                 relDXSqNorm =
        __GEIGEN__::__squaredNorm(__GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D));
    if(relDXSqNorm > fricDHat)
    {
        return lastH * sqrt(relDXSqNorm);
    }
    else
    {
        double f0;
        Friction::f0_SF(relDXSqNorm, eps, f0);
        return lastH * f0;
    }
}

// [per-body friction] Per-pair mu from the per-vertex table. A contact pair
// couples two primitives; each primitive lives on ONE body, so one
// representative vertex per side carries the side's mu exactly:
//   EE  (x>=0):            edges (x,y)-(z,w)   -> reps x and z
//   PT/PE/PP (x<0, enc.):  point -x-1 vs prim  -> reps -x-1 and y (decoded)
// Combine = geometric mean (PhysX-style multiplicative feel; symmetric).
// vmu == nullptr -> feature off, return the scalar fallback (legacy path).
__device__ __forceinline__ double _pair_mu(const int4 v, const double* vmu, double fallback)
{
    if(!vmu)
        return fallback;
    int a, b;
    if(v.x >= 0) { a = v.x; b = v.z; }
    else
    {
        a = -v.x - 1;
        b = (v.y >= 0) ? v.y : -v.y - 1;
    }
    return sqrt(vmu[a] * vmu[b]);
}


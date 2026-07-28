// ============================================================================
// energy/composite_externs.cuh — extern __global__ declarations the COMPOSITE
// needs for kernels whose definitions moved into per-term TUs (E3). Kernel
// externs self-check at device link (signatures mangle); grow this file one
// rung at a time, verbatim signatures only.
// ============================================================================
#pragma once

// ── rung 1: kinetic (energy/kinetic.cu) ──
// reduction kernel: still launched by the composite's BLOCKING dispatcher
// (single live caller, type 3) — extern carries the default args (legal:
// defaults may differ per TU declaration; the defining TU keeps its own).
extern __global__ void _getKineticEnergy_Reduction_3D(
    double3* _vertexes, double3* _xTilta, double* _energy, double* _masses, int number,
    double* penv = nullptr, const int* p2g = nullptr, int ng = 0);
extern __global__ void _calKineticGradient(
    double3* vertexes, double3* xTilta, double3* gradient, double* masses, int numbers);

// ── rung 2: ground (energy/ground.cu) ──
extern __global__ void _computeGroundGradientAndHessian(const double3* vertexes,
                                                 const double*  g_offset,
                                                 const double3* g_normal,
                                                 const uint32_t* _environment_collisionPair,
                                                 double3*  gradient,
                                                 uint32_t* _gpNum,
                                                 Eigen::Matrix3d* triplet_values,
                                                 int*   row_ids,
                                                 int*   col_ids,
                                                 double dHat,
                                                 double Kappa_scalar,
                                                 int    global_offset,
                                                 int    number,
                                                 const double* kappa_grp = nullptr,
                                                 const int*    p2g       = nullptr);
extern __global__ void _computeGroundGradient(const double3* vertexes,
                                       const double*  g_offset,
                                       const double3* g_normal,
                                       const uint32_t* _environment_collisionPair,
                                       double3*  gradient,
                                       uint32_t* _gpNum,
                                       double    dHat,
                                       double    Kappa_scalar,
                                       int       number,
                                       const double* kappa_grp = nullptr,
                                       const int*    p2g       = nullptr);
extern __global__ void _computeGroundEnergy_Reduction(double*        squeue,
                                               const double3* vertexes,
                                               const double*  g_offset,
                                               const double3* g_normal,
                                               const uint32_t* _environment_collisionPair,
                                               double dHat,
                                               double Kappa,
                                               int    number,
                                               double* penv = nullptr, const int* p2g = nullptr, int ng = 0,
                                               const uint32_t* d_live = nullptr);  // reduction: blocking-dispatcher launch, defaults carried

// ── rung 2: soft_constraints (energy/soft_constraints.cu) ──
extern __global__ void _computeSoftConstraintGradientAndHessian(const double3* vertexes,
                                                         const double3* targetVert,
                                                         const uint32_t* targetInd,
                                                         double3*  gradient,
                                                         uint32_t* _gpNum,
                                                         Eigen::Matrix3d* triplet_values,
                                                         int*   row_ids,
                                                         int*   col_ids,
                                                         double motionRate,
                                                         double rate,
                                                         int    global_offset,
                                                         int global_hessian_fem_offset,
                                                         const int*     stitch_paired_vertex,
                                                         const double3* stitch_rest_offset,
                                                         const int*     stitch_abd_body_id,
                                                         const __GEIGEN__::Vector12* abd_body_q,
                                                         int number);
extern __global__ void _computeSoftConstraintGradient(const double3*  vertexes,
                                               const double3*  targetVert,
                                               const uint32_t* targetInd,
                                               double3*        gradient,
                                               double          motionRate,
                                               double          rate,
                                               const int*      stitch_paired_vertex,
                                               const double3*  stitch_rest_offset,
                                               const int*      stitch_abd_body_id,
                                               const __GEIGEN__::Vector12* abd_body_q,
                                               int             number);
extern __global__ void _computeSoftConstraintEnergy_Reduction(double*        squeue,
                                                       const double3* vertexes,
                                                       const double3* targetVert,
                                                       const uint32_t* targetInd,
                                                       double motionRate,
                                                       double rate,
                                                       const int*     stitch_paired_vertex,
                                                       const double3* stitch_rest_offset,
                                                       int    number,
                                                       double* penv = nullptr, const int* p2g = nullptr, int ng = 0);  // reduction: blocking-dispatcher launch, defaults carried

// ── rung 2: delta (energy/delta.cu) ──
extern __global__ void _getDeltaEnergy_Reduction(double* squeue, const double3* b, const double3* dx, int vertexNum);  // reduction: blocking-dispatcher launch, defaults carried

// ── rung 3: bending (energy/bending.cu) ──
extern __global__ void _getQuadBendingEnergy_Reduction(double*        squeue,
                                                const double3* vertexes,
                                                const double3* rest_vertexex,
                                                const uint2*   edges,
                                                const uint2*   edge_adj_vertex,
                                                const Eigen::Matrix4d* quad_bending_Q,
                                                int    edgesNum,
                                                double bendStiff,
                                                double* penv = nullptr, const int* p2g = nullptr, int ng = 0);
extern __global__ void _getBendingEnergy_Reduction(double*        squeue,
                                            const double3* vertexes,
                                            const double3* rest_vertexex,
                                            const uint2*   edges,
                                            const uint2*   edge_adj_vertex,
                                            int            edgesNum,
                                            double         bendStiff,
                                            double* penv = nullptr, const int* p2g = nullptr, int ng = 0);

// ── rung 3: triangle_membrane (energy/triangle_membrane.cu) ──
extern __global__ void _get_triangleFEMEnergy_Reduction_3D(double*        squeue,
                                                    const double3* vertexes,
                                                    const uint3*   triangles,
                                                    const __GEIGEN__::Matrix2x2d* triDmInverses,
                                                    const double* area,
                                                    int           trianglesNum,
                                                    double        stretchStiff,
                                                    double        shearStiff,
                                                    double        strainRate,
                                                    double* penv = nullptr, const int* p2g = nullptr, int ng = 0);

// ── rung 4: fem_elastic (energy/fem_elastic.cu) ──
extern __global__ void _getFEMEnergy_Reduction_3D(double*        squeue,
                                           const double3* vertexes,
                                           const uint4*   tetrahedras,
                                           const __GEIGEN__::Matrix3x3d* DmInverses,
                                           const double* volume,
                                           int           tetrahedraNum,
                                           double*       lenRate,
                                           double*       volRate,
                                           double* penv = nullptr, const int* p2g = nullptr, int ng = 0);
extern __global__ void _getRestStableNHKEnergy_Reduction_3D(double*       squeue,
                                                     const double* volume,
                                                     int    tetrahedraNum,
                                                     double lenRate,
                                                     double volRate);

// ── rung 5: friction (energy/friction.cu) ──
extern __global__ void _calFrictionHessian(const double3*          _vertexes,
                                    const double3*          _o_vertexes,
                                    const int4*             _last_collisionPair,
                                    Eigen::Matrix3d*        triplet_values,
                                    int*                    row_ids,
                                    int*                    col_ids,
                                    uint32_t*               _cpNum,
                                    int                     number,
                                    double                  dt,
                                    double2*                distCoord,
                                    __GEIGEN__::Matrix3x2d* tanBasis,
                                    double                  eps2,
                                    double*                 lastH,
                                    double                  coef,
                                    const double*           vert_mu,
                                    int                     cd_offset4,
                                    int                     cd_offset3,
                                    int                     cd_offset2,
                                    int                     f_offset4,
                                    int                     f_offset3,
                                    int                     f_offset2);
extern __global__ void _calFrictionHessian_gd(const double3*   _vertexes,
                                       const double3*   _o_vertexes,
                                       const double3*   _normal,
                                       const uint32_t*  _last_collisionPair_gd,
                                       Eigen::Matrix3d* triplet_values,
                                       int*             row_ids,
                                       int*             col_ids,
                                       int              number,
                                       double           dt,
                                       double           eps2,
                                       double*          lastH,
                                       int              global_offset,
                                       double           coef,
                                       const double*    vert_mu_gd);
extern __global__ void _calFrictionGradient(const double3*    _vertexes,
                                     const double3*    _o_vertexes,
                                     const const int4* _last_collisionPair,
                                     double3*          _gradient,
                                     int               number,
                                     double            dt,
                                     double2*          distCoord,
                                     __GEIGEN__::Matrix3x2d* tanBasis,
                                     double                  eps2,
                                     double*                 lastH,
                                     double                  coef,
                                     const double*           vert_mu);
extern __global__ void _calFrictionGradient_gd(const double3* _vertexes,
                                        const double3* _o_vertexes,
                                        const double3* _normal,
                                        const const uint32_t* _last_collisionPair_gd,
                                        double3* _gradient,
                                        int      number,
                                        double   dt,
                                        double   eps2,
                                        double*  lastH,
                                        double   coef,
                                        const double* vert_mu_gd);
extern __global__ void _getFrictionEnergy_Reduction_3D(double*        squeue,
                                                const double3* vertexes,
                                                const double3* o_vertexes,
                                                const int4*    _collisionPair,
                                                int            cpNum,
                                                double         dt,
                                                const double2* distCoord,
                                                const __GEIGEN__::Matrix3x2d* tanBasis,
                                                const double* lastH,
                                                double        fricDHat,
                                                double        eps,
                                                double* penv = nullptr, const int* p2g = nullptr, int ng = 0,
                                                const double* vert_mu = nullptr, double mu_global = 1.0

);
extern __global__ void _getFrictionEnergy_gd_Reduction_3D(double*        squeue,
                                                   const double3* vertexes,
                                                   const double3* o_vertexes,
                                                   const double3* _normal,
                                                   const uint32_t* _collisionPair_gd,
                                                   int           gpNum,
                                                   double        dt,
                                                   const double* lastH,
                                                   double        eps,
                                                   double* penv = nullptr, const int* p2g = nullptr, int ng = 0,
                                                   const double* vert_mu_gd = nullptr, double mu_global = 1.0

);

// ── rung 6: barrier (energy/barrier.cu) ──
extern __global__ void _calBarrierGradient(const double3*    _vertexes,
                                    const double3*    _rest_vertexes,
                                    const const int4* _collisionPair,
                                    double3*          _gradient,
                                    double            dHat,
                                    double            Kappa_scalar,
                                    int               number,
                                    const double*     kappa_grp = nullptr,
                                    const int*        p2g       = nullptr,
                                    int2*             _ec_out_pair  = nullptr,
                                    double3*          _ec_out_force = nullptr,
                                    const int*        _ec_pbid      = nullptr,
                                    double            _ec_inv_dt2   = 0.0);
extern __global__ void _getBarrierEnergy_Reduction_3D(double*        squeue,
                                               const double3* vertexes,
                                               const double3* rest_vertexes,
                                               int4*          _collisionPair,
                                               double         _Kappa,
                                               double         _dHat,
                                               int            cpNum,
                                               double* penv = nullptr, const int* p2g = nullptr, int ng = 0,
                                               const uint32_t* d_live = nullptr);

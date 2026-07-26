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
                                               double* penv = nullptr, const int* p2g = nullptr, int ng = 0);  // reduction: blocking-dispatcher launch, defaults carried

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

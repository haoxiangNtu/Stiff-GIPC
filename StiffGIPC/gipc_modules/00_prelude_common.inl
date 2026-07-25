//
// GIPC.cu
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#include "GIPC.cuh"
#include "eigen_data.h"  // Vector12 for stitch local-frame fix
#include <stdexcept>     // [d-floor fail-fast] ground-distance collapse -> throw
#include <string>
#include <gipc/gipc.h>
#include "cuda_tools/cuda_tools.h"
#include "GIPC_PDerivative.cuh"
#include "fem_parameters.h"
#include "device_common/reductions.cuh"   // [v0.8.6 2a] unified block reductions
#include "contact/pair_buffers.cuh"       // [v0.8.6 2b] pair-buffer growth mechanics owner
#include "multienv/isolation.cuh"        // [v0.8.6 2c] env-isolation machinery owner
#include "ACCD.cuh"
#include "femEnergy.cuh"
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include "FrictionUtils.cuh"
#include <cfloat>
#include <cstring>
#include <fstream>
#include <cstdlib>   // std::getenv for STIFF_SKIP_CCD_SANITY
#include <limits>
#include "Eigen/Eigen"
#include <gipc/statistics.h>
#include <gipc_path.h>
#include <gipc/utils/timer.h>

#include <muda/cub/device/device_radix_sort.h>
#include <cub/device/device_radix_sort.cuh>   // [perenv-parallel #2] pool sort-scratch sizing
using namespace Eigen;

// Global log verbosity for the per-frame + one-time solver prints.  0 = silent
// (engine plays nice in co-simulation / piped tooling); >=1 = default verbose.
// Defined here (top of TU) so every use below — incl. the buffer-alloc print
// at ~line 8843, which precedes solve_subIP — can see it.  Set from Python via
// SimEngine::set_log_level (extern'd in sim_engine.cu).
int g_gipc_log_level = 1;
#define RANK 2
#define NEWF

#include "contact/ccd_invalid_bits.h"   // CCD invalid-mask contract (hoisted, 2d step 2)

template <typename Scalar, int size>
__device__ __host__ void makePDGeneral(Eigen::Matrix<Scalar, size, size>& symMtr)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, size, size>> eigen_solver;

    if constexpr(size <= 3)
        eigen_solver.computeDirect(symMtr);
    else
        eigen_solver.compute(symMtr);
    Eigen::Vector<Scalar, size> eigen_values = eigen_solver.eigenvalues();
    Eigen::Matrix<Scalar, size, size> eigen_vectors = eigen_solver.eigenvectors();


    if(eigen_values[0] >= 0.0)
    {
        return;
    }

    for(int i = 0; i < size; ++i)
    {
        if(eigen_values(i) < 0)
        {
            eigen_values(i) = 0;
        }
    }
    symMtr = eigen_vectors * eigen_values.asDiagonal() * eigen_vectors.transpose();
}

template <typename Scalar, int size>
__device__ __host__ void makePD(Eigen::Matrix<Scalar, size, size>& symMtr)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, size, size>> eigenSolver(symMtr);
    if(eigenSolver.eigenvalues()[0] >= 0.0)
    {
        return;
    }
    Eigen::Matrix<Scalar, size, size> D;  //(eigenSolver.eigenvalues());
    D.setZero();
    int rows = size;  //((size == Eigen::Dynamic) ? symMtr.rows() : size);
    for(int i = 0; i < rows; i++)
    {
        if(eigenSolver.eigenvalues()[i] > 0.0)
        {
            D(i, i) = eigenSolver.eigenvalues()[i];
        }
    }
    symMtr = eigenSolver.eigenvectors() * D * eigenSolver.eigenvectors().transpose();
}

template <int ROWS, int COLS>
__device__ inline void write_triplet(Eigen::Matrix3d*    triplet_value,
                                     int*                row_ids,
                                     int*                col_ids,
                                     const unsigned int* index,
                                     const double        input[ROWS][COLS],
                                     const int&          offset)
{
    int rown = ROWS / 3;
    int coln = COLS / 3;
    for(int ii = 0; ii < rown; ii++)
    {
#ifdef SymGH
        int start = ii;
#else
        int start = 0;
#endif
        for(int jj = start; jj < coln; jj++)
        {

#ifdef SymGH
            int kk = ii * rown + jj - ii * (ii + 1) / 2;
#else
            int kk = ii * rown + jj;  // - ii * (ii + 1) / 2;
#endif
            int row = index[ii];
            int col = index[jj];
#ifdef SymGH
            if(row > col)
            {
                row_ids[offset + kk] = col;
                col_ids[offset + kk] = row;
                for(int iii = 0; iii < 3; iii++)
                {
                    for(int jjj = 0; jjj < 3; jjj++)
                    {
                        triplet_value[offset + kk](iii, jjj) =
                            input[jj * 3 + iii][ii * 3 + jjj];
                    }
                }
            }
            else
#endif
            {
                row_ids[offset + kk] = row;
                col_ids[offset + kk] = col;
                for(int iii = 0; iii < 3; iii++)
                {
                    for(int jjj = 0; jjj < 3; jjj++)
                    {
                        triplet_value[offset + kk](iii, jjj) =
                            input[ii * 3 + iii][jj * 3 + jjj];
                    }
                }
            }
        }
    }
}

// [contact-slot hygiene] Contact triplet slots are RESERVED at emission time
// (mlbvh MatIndex = atomicAdd on the per-type counter), so an assembly-side
// early-out must still deposit its blocks. The whole-buffer memset in
// computeGradientAndHessian already zeros unwritten slots to (0,0,0), which
// keeps the SpMV in-bounds — but (0,0) MISCLASSIFIES as abd_abd in mixed
// ABD+FEM scenes (both indices < abd_vert_num) where the real pair is
// fem_*/fem_fem. Exactly-parallel edge pairs (I1==0) carry zero mollified
// barrier energy, so depositing a zero 12x12 block at the pair's DECODED
// (valid) vertex ids is the correct contribution and classifies correctly.
// (Not the towel-strict OOB root cause — that was the pre-solve grow, see
// GlobalLinearSystem::build; this is the reserved-slot correctness contract.)
__device__ inline void write_zero_triplet12(Eigen::Matrix3d* triplet_value,
                                            int*             row_ids,
                                            int*             col_ids,
                                            const uint4&     gidx,
                                            int              offset)
{
    double Z[12][12] = {};
    write_triplet<12, 12>(triplet_value, row_ids, col_ids, &(gidx.x), Z, offset);
}


__device__ __host__ inline uint32_t expand_bits(std::uint32_t v) noexcept
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ __host__ inline uint32_t hash_code(
    int type, double x, double y, double z, double resolution = 1024) noexcept
{
    x = std::min(std::max(x * resolution, 0.0), resolution - 1.0);
    y = std::min(std::max(y * resolution, 0.0), resolution - 1.0);
    z = std::min(std::max(z * resolution, 0.0), resolution - 1.0);


    //
    if(type == -1)
    {
        const uint32_t xx     = expand_bits(static_cast<uint32_t>(x));
        const uint32_t yy     = expand_bits(static_cast<uint32_t>(y));
        const uint32_t zz     = expand_bits(static_cast<uint32_t>(z));
        std::uint32_t  mchash = ((xx << 2) + (yy << 1) + zz);

        return mchash;
    }
    else if(type == 0)
    {
        return (((static_cast<uint32_t>(z) * 1024) + static_cast<uint32_t>(y)) * 1024)
               + static_cast<uint32_t>(x);
    }
    else if(type == 1)
    {
        return (((static_cast<uint32_t>(y) * 1024) + static_cast<uint32_t>(z)) * 1024)
               + static_cast<uint32_t>(x);
    }
    else if(type == 2)
    {
        return (((static_cast<uint32_t>(x) * 1024) + static_cast<uint32_t>(z)) * 1024)
               + static_cast<uint32_t>(y);
    }
    else if(type == 3)
    {
        return (((static_cast<uint32_t>(z) * 1024) + static_cast<uint32_t>(x)) * 1024)
               + static_cast<uint32_t>(y);
    }
    else if(type == 4)
    {
        return (((static_cast<uint32_t>(y) * 1024) + static_cast<uint32_t>(x)) * 1024)
               + static_cast<uint32_t>(z);
    }
    else
    {
        return (((static_cast<uint32_t>(x) * 1024) + static_cast<uint32_t>(y)) * 1024)
               + static_cast<uint32_t>(z);
    }
    //std::uint32_t mchash = (((static_cast<std::uint32_t>(z) * 1024) + static_cast<std::uint32_t>(y)) * 1024) + static_cast<std::uint32_t>(x);//((xx << 2) + (yy << 1) + zz);
    //return mchash;
}

__global__ void _partition_collision_triplets(const uint64_t* sort_hash,
                                              int*            abd_abd_offset,
                                              int*            abd_fem_offset,
                                              int*            fem_abd_offset,
                                              int*            fem_fem_offset,
                                              int             number)
{
    extern __shared__ int shared_hash[];
    unsigned int          idx = threadIdx.x + (blockDim.x * blockIdx.x);
    //if(idx == 0)
    //{
    //    *abd_abd_offset = -1;
    //    *abd_fem_offset = -1;
    //    *fem_abd_offset = -1;
    //    *fem_fem_offset = -1;
    //}
    int self_hash;
    if(idx < number)
    {
        self_hash                    = sort_hash[idx];
        shared_hash[threadIdx.x + 1] = self_hash;
        if(idx > 0 && threadIdx.x == 0)
        {
            shared_hash[0] = sort_hash[idx - 1];
        }
    }
    __syncthreads();
    if(idx < number)
    {
        int prior_hash = idx == 0 ? -1 : shared_hash[threadIdx.x];
        if(self_hash != prior_hash)
        {
            if(self_hash == 3)
            {
                *abd_abd_offset = idx;
            }
            else if(self_hash == 1)
            {
                *abd_fem_offset = idx;
            }
            else if(self_hash == 2)
            {
                *fem_abd_offset = idx;
            }
            else if(self_hash == 0)
            {
                *fem_fem_offset = idx;
            }
        }
    }
}

__global__ void _reorder_triplets(int*             row_ids_input,
                                  int*             col_ids_input,
                                  Eigen::Matrix3d* triplet_value_inpuit,
                                  int*             row_ids,
                                  int*             col_ids,
                                  Eigen::Matrix3d* triplet_value,
                                  const uint32_t*  sort_index,
                                  int              number)
{
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= number)
        return;
    row_ids[idx]       = row_ids_input[sort_index[idx]];
    col_ids[idx]       = col_ids_input[sort_index[idx]];
    triplet_value[idx] = triplet_value_inpuit[sort_index[idx]];
}


uint64_t GIPC::getHashCode(double3 p, uint32_t i)
{
    uint64_t code = hash_code(-1, p.x, p.y, p.z);
    return (code << 32) | i;
}

__global__ void _calcTetMChash(uint64_t*       _MChash,
                               const double3*  _vertexes,
                               uint4*          tets,
                               const AABB*     _MaxBv,
                               const uint32_t* sortMapVertIndex,
                               int             number)
{
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= number)
        return;

    tets[idx].x = sortMapVertIndex[tets[idx].x];
    tets[idx].y = sortMapVertIndex[tets[idx].y];
    tets[idx].z = sortMapVertIndex[tets[idx].z];
    tets[idx].w = sortMapVertIndex[tets[idx].w];

    double3 SceneSize = make_double3((*_MaxBv).upper.x - (*_MaxBv).lower.x,
                                     (*_MaxBv).upper.y - (*_MaxBv).lower.y,
                                     (*_MaxBv).upper.z - (*_MaxBv).lower.z);
    double3 centerP   = __GEIGEN__::__s_vec_multiply(
        __GEIGEN__::__add(
            __GEIGEN__::__add(_vertexes[tets[idx].x], _vertexes[tets[idx].y]),
            __GEIGEN__::__add(_vertexes[tets[idx].z], _vertexes[tets[idx].w])),
        0.25);
    double3 offset = make_double3(centerP.x - (*_MaxBv).lower.x,
                                  centerP.y - (*_MaxBv).lower.y,
                                  centerP.z - (*_MaxBv).lower.z);

    int type = 0;
    if(SceneSize.x > SceneSize.y && SceneSize.y > SceneSize.z)
    {
        type = 0;
    }
    else if(SceneSize.x > SceneSize.z && SceneSize.z > SceneSize.y)
    {
        type = 1;
    }
    else if(SceneSize.y > SceneSize.z && SceneSize.z > SceneSize.x)
    {
        type = 2;
    }
    else if(SceneSize.y > SceneSize.x && SceneSize.x > SceneSize.z)
    {
        type = 3;
    }
    else if(SceneSize.z > SceneSize.x && SceneSize.x > SceneSize.y)
    {
        type = 4;
    }
    else
    {
        type = 5;
    }

    //printf("%d   %f     %f     %f\n", offset.x, offset.y, offset.z);
    uint64_t mc32 = hash_code(type,
                              offset.x / SceneSize.x,
                              offset.y / SceneSize.y,
                              offset.z / SceneSize.z);
    uint64_t mc64 = ((mc32 << 32) | idx);
    //printf("morton code %d\n", mc64);
    _MChash[idx] = mc64;
}

__global__ void _updateTopology(uint4*          tets,
                                uint3*          tris,
                                const uint32_t* sortMapVertIndex,
                                int             traNumber,
                                int             triNumber)
{
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < traNumber)
    {

        tets[idx].x = sortMapVertIndex[tets[idx].x];
        tets[idx].y = sortMapVertIndex[tets[idx].y];
        tets[idx].z = sortMapVertIndex[tets[idx].z];
        tets[idx].w = sortMapVertIndex[tets[idx].w];
    }
    if(idx < triNumber)
    {
        tris[idx].x = sortMapVertIndex[tris[idx].x];
        tris[idx].y = sortMapVertIndex[tris[idx].y];
        tris[idx].z = sortMapVertIndex[tris[idx].z];
    }
}


__global__ void _updateVertexes(double3*                      o_vertexes,
                                const double3*                _vertexes,
                                double*                       tempM,
                                const double*                 mass,
                                __GEIGEN__::Matrix3x3d*       tempCons,
                                int*                          tempBtype,
                                const __GEIGEN__::Matrix3x3d* cons,
                                const int*                    bType,
                                const uint32_t*               sortIndex,
                                uint32_t*                     sortMapIndex,
                                int                           number)
{
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= number)
        return;
    o_vertexes[idx]              = _vertexes[sortIndex[idx]];
    tempM[idx]                   = mass[sortIndex[idx]];
    tempCons[idx]                = cons[sortIndex[idx]];
    sortMapIndex[sortIndex[idx]] = idx;
    tempBtype[idx]               = bType[sortIndex[idx]];
    //printf("original idx: %d        new idx: %d\n", sortIndex[idx], idx);
}

__global__ void _updateTetrahedras(uint4*                        o_tetrahedras,
                                   uint4*                        tetrahedras,
                                   double*                       tempV,
                                   const double*                 volum,
                                   __GEIGEN__::Matrix3x3d*       tempDmInverse,
                                   const __GEIGEN__::Matrix3x3d* dmInverse,
                                   const uint32_t*               sortTetIndex,
                                   const uint32_t* sortMapVertIndex,
                                   int             number)
{
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= number)
        return;
    //tetrahedras[idx].x = sortMapVertIndex[tetrahedras[idx].x];
    //tetrahedras[idx].y = sortMapVertIndex[tetrahedras[idx].y];
    //tetrahedras[idx].z = sortMapVertIndex[tetrahedras[idx].z];
    //tetrahedras[idx].w = sortMapVertIndex[tetrahedras[idx].w];
    o_tetrahedras[idx] = tetrahedras[sortTetIndex[idx]];
    tempV[idx]         = volum[sortTetIndex[idx]];
    tempDmInverse[idx] = dmInverse[sortTetIndex[idx]];
}

__global__ void _calcVertMChash(uint64_t* _MChash, const double3* _vertexes, const AABB* _MaxBv, int number)
{
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= number)
        return;
    double3 SceneSize = make_double3((*_MaxBv).upper.x - (*_MaxBv).lower.x,
                                     (*_MaxBv).upper.y - (*_MaxBv).lower.y,
                                     (*_MaxBv).upper.z - (*_MaxBv).lower.z);
    double3 centerP   = _vertexes[idx];
    double3 offset    = make_double3(centerP.x - (*_MaxBv).lower.x,
                                  centerP.y - (*_MaxBv).lower.y,
                                  centerP.z - (*_MaxBv).lower.z);
    int     type      = -1;
    if(type >= 0)
    {
        if(SceneSize.x > SceneSize.y && SceneSize.y > SceneSize.z)
        {
            type = 0;
        }
        else if(SceneSize.x > SceneSize.z && SceneSize.z > SceneSize.y)
        {
            type = 1;
        }
        else if(SceneSize.y > SceneSize.z && SceneSize.z > SceneSize.x)
        {
            type = 2;
        }
        else if(SceneSize.y > SceneSize.x && SceneSize.x > SceneSize.z)
        {
            type = 3;
        }
        else if(SceneSize.z > SceneSize.x && SceneSize.x > SceneSize.y)
        {
            type = 4;
        }
        else
        {
            type = 5;
        }
    }

    //printf("minSize %f     %f     %f\n", SceneSize.x, SceneSize.y, SceneSize.z);
    uint64_t mc32 = hash_code(type,
                              offset.x / SceneSize.x,
                              offset.y / SceneSize.y,
                              offset.z / SceneSize.z);
    uint64_t mc64 = ((mc32 << 32) | idx);
    //printf("morton code %lld\n", mc64);
    _MChash[idx] = mc64;
}

__global__ void _reduct_max_double3_to_double(const double3* _double3Dim, double* _double1Dim, int number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    // Inactive lanes must still participate in both barriers and full-warp
    // shuffles. The former early return made every non-full final block
    // undefined; calcMinMovement hits this twice for almost every mesh size.
    double temp = 0.0;
    if(idx < number)
    {
        double3 tempMove = _double3Dim[idx];
        temp = std::max(std::max(abs(tempMove.x), abs(tempMove.y)), abs(tempMove.z));
    }

    // [v0.8.6 2a] unified tail — see device_common/reductions.cuh
    gipc_block_max_full_to(temp, tep, number, idof, 0.0, _double1Dim + blockIdx.x);
}

__global__ void _reduct_min_double(double* _double1Dim, int number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    // Every thread must reach the block barriers. DBL_MAX is the neutral value
    // for inactive lanes in the final partial block.
    double temp = (idx < number) ? _double1Dim[idx] : DBL_MAX;

    __threadfence();


    // [v0.8.6 2a] unified tail — see device_common/reductions.cuh
    gipc_block_min_full_to(temp, tep, number, idof, DBL_MAX, _double1Dim + blockIdx.x);
}

__global__ void _reduct_M_double2(double2* _double2Dim, int number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double2 sdata[];

    double2 temp = idx < number ? _double2Dim[idx] : make_double2(0.0, 0.0);

    __threadfence();


    int    warpTid = threadIdx.x % 32;
    int    warpId  = (threadIdx.x >> 5);
    double nextTp;
    int    warpNum;
    //int tidNum = 32;
    if(blockIdx.x == gridDim.x - 1)
    {
        //tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        double tempMin = __shfl_down_sync(0xffffffff, temp.x, i);
        double tempMax = __shfl_down_sync(0xffffffff, temp.y, i);
        temp.x         = std::max(temp.x, tempMin);
        temp.y         = std::max(temp.y, tempMax);
    }
    if(warpTid == 0)
    {
        sdata[warpId] = temp;
    }
    __syncthreads();
    if(warpId != 0)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = (warpTid < warpNum) ? sdata[warpTid] : make_double2(0.0, 0.0);

        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            double tempMin = __shfl_down_sync(0xffffffff, temp.x, i);
            double tempMax = __shfl_down_sync(0xffffffff, temp.y, i);
            temp.x         = std::max(temp.x, tempMin);
            temp.y         = std::max(temp.y, tempMax);
        }
    }
    if(threadIdx.x == 0)
    {
        _double2Dim[blockIdx.x] = temp;
    }
}

__global__ void _reduct_max_double(double* _double1Dim, int number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    double temp = (idx < number) ? _double1Dim[idx] : -DBL_MAX;

    __threadfence();


    // [v0.8.6 2a] unified tail — see device_common/reductions.cuh
    gipc_block_max_full_to(temp, tep, number, idof, -DBL_MAX, _double1Dim + blockIdx.x);
}


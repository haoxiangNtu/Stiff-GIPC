//
// GIPC.cu
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#include "GIPC.cuh"
#include "eigen_data.h"  // Vector12 for stitch local-frame fix
#include <gipc/gipc.h>
#include "cuda_tools/cuda_tools.h"
#include "GIPC_PDerivative.cuh"
#include "fem_parameters.h"
#include "ACCD.cuh"
#include "femEnergy.cuh"
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include "FrictionUtils.cuh"
#include <fstream>
#include <cstdlib>   // std::getenv for STIFF_SKIP_CCD_SANITY
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

    if(idx >= number)
        return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    double3 tempMove = _double3Dim[idx];

    double temp =
        std::max(std::max(abs(tempMove.x), abs(tempMove.y)), abs(tempMove.z));

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
        double tempMin = __shfl_down_sync(0xffffffff, temp, i);
        temp           = std::max(temp, tempMin);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            double tempMin = __shfl_down_sync(0xffffffff, temp, i);
            temp           = std::max(temp, tempMin);
        }
    }
    if(threadIdx.x == 0)
    {
        _double1Dim[blockIdx.x] = temp;
    }
}

__global__ void _reduct_min_double(double* _double1Dim, int number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= number)
        return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    double temp = _double1Dim[idx];

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
        double tempMin = __shfl_down_sync(0xffffffff, temp, i);
        temp           = std::min(temp, tempMin);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            double tempMin = __shfl_down_sync(0xffffffff, temp, i);
            temp           = std::min(temp, tempMin);
        }
    }
    if(threadIdx.x == 0)
    {
        _double1Dim[blockIdx.x] = temp;
    }
}

__global__ void _reduct_M_double2(double2* _double2Dim, int number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double2 sdata[];

    if(idx >= number)
        return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    double2 temp = _double2Dim[idx];

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
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = sdata[threadIdx.x];

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

    if(idx >= number)
        return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    double temp = _double1Dim[idx];

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
        double tempMax = __shfl_down_sync(0xffffffff, temp, i);
        temp           = std::max(temp, tempMax);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            double tempMax = __shfl_down_sync(0xffffffff, temp, i);
            temp           = std::max(temp, tempMax);
        }
    }
    if(threadIdx.x == 0)
    {
        _double1Dim[blockIdx.x] = temp;
    }
}

__device__ double __cal_Barrier_energy(const double3* _vertexes,
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

__device__ bool segTriIntersect(const double3& ve0,
                                const double3& ve1,
                                const double3& vt0,
                                const double3& vt1,
                                const double3& vt2)
{

    //printf("check for tri and lines\n");

    __GEIGEN__::Matrix3x3d coefMtr;
    double3                col0 = __GEIGEN__::__minus(vt1, vt0);
    double3                col1 = __GEIGEN__::__minus(vt2, vt0);
    double3                col2 = __GEIGEN__::__minus(ve0, ve1);

    __GEIGEN__::__set_Mat_val_column(coefMtr, col0, col1, col2);

    double3 n = __GEIGEN__::__v_vec_cross(col0, col1);
    if(__GEIGEN__::__v_vec_dot(n, __GEIGEN__::__minus(ve0, vt0))
           * __GEIGEN__::__v_vec_dot(n, __GEIGEN__::__minus(ve1, vt0))
       > 0)
    {
        return false;
    }

    double det = __GEIGEN__::__Determiant(coefMtr);

    if(abs(det) < 1e-20)
    {
        return false;
    }

    __GEIGEN__::Matrix3x3d D1, D2, D3;
    double3                b = __GEIGEN__::__minus(ve0, vt0);

    __GEIGEN__::__set_Mat_val_column(D1, b, col1, col2);
    __GEIGEN__::__set_Mat_val_column(D2, col0, b, col2);
    __GEIGEN__::__set_Mat_val_column(D3, col0, col1, b);

    double uvt[3];
    uvt[0] = __GEIGEN__::__Determiant(D1) / det;
    uvt[1] = __GEIGEN__::__Determiant(D2) / det;
    uvt[2] = __GEIGEN__::__Determiant(D3) / det;

    if(uvt[0] >= 0.0 && uvt[1] >= 0.0 && uvt[0] + uvt[1] <= 1.0 && uvt[2] >= 0.0
       && uvt[2] <= 1.0)
    {
        return true;
    }
    else
    {
        return false;
    }
}

__device__ __host__ inline bool _overlap(const AABB& lhs, const AABB& rhs, const double& gapL) noexcept
{
    if((rhs.lower.x - lhs.upper.x) >= gapL || (lhs.lower.x - rhs.upper.x) >= gapL)
        return false;
    if((rhs.lower.y - lhs.upper.y) >= gapL || (lhs.lower.y - rhs.upper.y) >= gapL)
        return false;
    if((rhs.lower.z - lhs.upper.z) >= gapL || (lhs.lower.z - rhs.upper.z) >= gapL)
        return false;
    return true;
}

__device__ double _selfConstraintVal(const double3* vertexes, const int4& active)
{
    double val;
    if(active.x >= 0)
    {
        if(active.w >= 0)
        {
            _d_EE(vertexes[active.x],
                  vertexes[active.y],
                  vertexes[active.z],
                  vertexes[active.w],
                  val);
        }
        else
        {
            _d_EE(vertexes[active.x],
                  vertexes[active.y],
                  vertexes[active.z],
                  vertexes[-active.w - 1],
                  val);
        }
    }
    else
    {
        if(active.z < 0)
        {
            if(active.y < 0)
            {
                _d_PP(vertexes[-active.x - 1], vertexes[-active.y - 1], val);
            }
            else
            {
                _d_PP(vertexes[-active.x - 1], vertexes[active.y], val);
            }
        }
        else if(active.w < 0)
        {
            if(active.y < 0)
            {
                _d_PE(vertexes[-active.x - 1],
                      vertexes[-active.y - 1],
                      vertexes[active.z],
                      val);
            }
            else
            {
                _d_PE(
                    vertexes[-active.x - 1], vertexes[active.y], vertexes[active.z], val);
            }
        }
        else
        {
            _d_PT(vertexes[-active.x - 1],
                  vertexes[active.y],
                  vertexes[active.z],
                  vertexes[active.w],
                  val);
        }
    }
    return val;
}

__device__ double _computeInjectiveStepSize_3d(const double3*  verts,
                                               const double3*  mv,
                                               const uint32_t& v0,
                                               const uint32_t& v1,
                                               const uint32_t& v2,
                                               const uint32_t& v3,
                                               double          ratio,
                                               double          errorRate)
{

    double x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4;
    double p1, p2, p3, p4, q1, q2, q3, q4, r1, r2, r3, r4;
    double a, b, c, d, t;


    x1 = verts[v0].x;
    x2 = verts[v1].x;
    x3 = verts[v2].x;
    x4 = verts[v3].x;

    y1 = verts[v0].y;
    y2 = verts[v1].y;
    y3 = verts[v2].y;
    y4 = verts[v3].y;

    z1 = verts[v0].z;
    z2 = verts[v1].z;
    z3 = verts[v2].z;
    z4 = verts[v3].z;

    int _3Fii0 = v0 * 3;
    int _3Fii1 = v1 * 3;
    int _3Fii2 = v2 * 3;
    int _3Fii3 = v3 * 3;

    p1 = -mv[v0].x;
    p2 = -mv[v1].x;
    p3 = -mv[v2].x;
    p4 = -mv[v3].x;

    q1 = -mv[v0].y;
    q2 = -mv[v1].y;
    q3 = -mv[v2].y;
    q4 = -mv[v3].y;

    r1 = -mv[v0].z;
    r2 = -mv[v1].z;
    r3 = -mv[v2].z;
    r4 = -mv[v3].z;

    a = -p1 * q2 * r3 + p1 * r2 * q3 + q1 * p2 * r3 - q1 * r2 * p3 - r1 * p2 * q3
        + r1 * q2 * p3 + p1 * q2 * r4 - p1 * r2 * q4 - q1 * p2 * r4 + q1 * r2 * p4
        + r1 * p2 * q4 - r1 * q2 * p4 - p1 * q3 * r4 + p1 * r3 * q4 + q1 * p3 * r4
        - q1 * r3 * p4 - r1 * p3 * q4 + r1 * q3 * p4 + p2 * q3 * r4 - p2 * r3 * q4
        - q2 * p3 * r4 + q2 * r3 * p4 + r2 * p3 * q4 - r2 * q3 * p4;
    b = -x1 * q2 * r3 + x1 * r2 * q3 + y1 * p2 * r3 - y1 * r2 * p3 - z1 * p2 * q3
        + z1 * q2 * p3 + x2 * q1 * r3 - x2 * r1 * q3 - y2 * p1 * r3
        + y2 * r1 * p3 + z2 * p1 * q3 - z2 * q1 * p3 - x3 * q1 * r2
        + x3 * r1 * q2 + y3 * p1 * r2 - y3 * r1 * p2 - z3 * p1 * q2 + z3 * q1 * p2
        + x1 * q2 * r4 - x1 * r2 * q4 - y1 * p2 * r4 + y1 * r2 * p4 + z1 * p2 * q4
        - z1 * q2 * p4 - x2 * q1 * r4 + x2 * r1 * q4 + y2 * p1 * r4 - y2 * r1 * p4
        - z2 * p1 * q4 + z2 * q1 * p4 + x4 * q1 * r2 - x4 * r1 * q2 - y4 * p1 * r2
        + y4 * r1 * p2 + z4 * p1 * q2 - z4 * q1 * p2 - x1 * q3 * r4 + x1 * r3 * q4
        + y1 * p3 * r4 - y1 * r3 * p4 - z1 * p3 * q4 + z1 * q3 * p4 + x3 * q1 * r4
        - x3 * r1 * q4 - y3 * p1 * r4 + y3 * r1 * p4 + z3 * p1 * q4 - z3 * q1 * p4
        - x4 * q1 * r3 + x4 * r1 * q3 + y4 * p1 * r3 - y4 * r1 * p3 - z4 * p1 * q3
        + z4 * q1 * p3 + x2 * q3 * r4 - x2 * r3 * q4 - y2 * p3 * r4 + y2 * r3 * p4
        + z2 * p3 * q4 - z2 * q3 * p4 - x3 * q2 * r4 + x3 * r2 * q4 + y3 * p2 * r4
        - y3 * r2 * p4 - z3 * p2 * q4 + z3 * q2 * p4 + x4 * q2 * r3 - x4 * r2 * q3
        - y4 * p2 * r3 + y4 * r2 * p3 + z4 * p2 * q3 - z4 * q2 * p3;
    c = -x1 * y2 * r3 + x1 * z2 * q3 + x1 * y3 * r2 - x1 * z3 * q2 + y1 * x2 * r3
        - y1 * z2 * p3 - y1 * x3 * r2 + y1 * z3 * p2 - z1 * x2 * q3
        + z1 * y2 * p3 + z1 * x3 * q2 - z1 * y3 * p2 - x2 * y3 * r1
        + x2 * z3 * q1 + y2 * x3 * r1 - y2 * z3 * p1 - z2 * x3 * q1 + z2 * y3 * p1
        + x1 * y2 * r4 - x1 * z2 * q4 - x1 * y4 * r2 + x1 * z4 * q2 - y1 * x2 * r4
        + y1 * z2 * p4 + y1 * x4 * r2 - y1 * z4 * p2 + z1 * x2 * q4 - z1 * y2 * p4
        - z1 * x4 * q2 + z1 * y4 * p2 + x2 * y4 * r1 - x2 * z4 * q1 - y2 * x4 * r1
        + y2 * z4 * p1 + z2 * x4 * q1 - z2 * y4 * p1 - x1 * y3 * r4 + x1 * z3 * q4
        + x1 * y4 * r3 - x1 * z4 * q3 + y1 * x3 * r4 - y1 * z3 * p4 - y1 * x4 * r3
        + y1 * z4 * p3 - z1 * x3 * q4 + z1 * y3 * p4 + z1 * x4 * q3 - z1 * y4 * p3
        - x3 * y4 * r1 + x3 * z4 * q1 + y3 * x4 * r1 - y3 * z4 * p1 - z3 * x4 * q1
        + z3 * y4 * p1 + x2 * y3 * r4 - x2 * z3 * q4 - x2 * y4 * r3 + x2 * z4 * q3
        - y2 * x3 * r4 + y2 * z3 * p4 + y2 * x4 * r3 - y2 * z4 * p3 + z2 * x3 * q4
        - z2 * y3 * p4 - z2 * x4 * q3 + z2 * y4 * p3 + x3 * y4 * r2 - x3 * z4 * q2
        - y3 * x4 * r2 + y3 * z4 * p2 + z3 * x4 * q2 - z3 * y4 * p2;
    d = (ratio)
        * (x1 * z2 * y3 - x1 * y2 * z3 + y1 * x2 * z3 - y1 * z2 * x3 - z1 * x2 * y3
           + z1 * y2 * x3 + x1 * y2 * z4 - x1 * z2 * y4 - y1 * x2 * z4 + y1 * z2 * x4
           + z1 * x2 * y4 - z1 * y2 * x4 - x1 * y3 * z4 + x1 * z3 * y4 + y1 * x3 * z4
           - y1 * z3 * x4 - z1 * x3 * y4 + z1 * y3 * x4 + x2 * y3 * z4 - x2 * z3 * y4
           - y2 * x3 * z4 + y2 * z3 * x4 + z2 * x3 * y4 - z2 * y3 * x4);


    //printf("a b c d:   %f  %f  %f  %f     %f     %f,    id0, id1, id2, id3:  %d  %d  %d  %d\n", a, b, c, d, ratio, errorRate, v0, v1, v2, v3);
    if(abs(a) <= errorRate /** errorRate*/)
    {
        if(abs(b) <= errorRate /** errorRate*/)
        {
            if(false && abs(c) <= errorRate)
            {
                t = 1;
            }
            else
            {
                t = -d / c;
            }
        }
        else
        {
            double desc = c * c - 4 * b * d;
            if(desc > 0)
            {
                t = (-c - sqrt(desc)) / (2 * b);
                if(t < 0)
                    t = (-c + sqrt(desc)) / (2 * b);
            }
            else
                t = 1;
        }
    }
    else
    {
        //double results[3];
        //int number = 0;
        //__GEIGEN__::__NewtonSolverForCubicEquation(a, b, c, d, results, number, errorRate);

        //t = 1;
        //for (int index = 0;index < number;index++) {
        //    if (results[index] > 0 && results[index] < t) {
        //        t = results[index];
        //    }
        //}
        //zs::complex<double> i(0, 1);
        //zs::complex<double> delta0(b * b - 3 * a * c, 0);
        //zs::complex<double> delta1(2 * b * b * b - 9 * a * b * c + 27 * a * a * d, 0);
        //zs::complex<double> C =
        //    pow((delta1 + sqrt(delta1 * delta1 - 4.0 * delta0 * delta0 * delta0)) / 2.0,
        //        1.0 / 3.0);
        //if(abs(C) == 0.0)
        //{
        //    // a corner case listed by wikipedia found by our collaborate from another project
        //    C = pow((delta1 - sqrt(delta1 * delta1 - 4.0 * delta0 * delta0 * delta0)) / 2.0,
        //            1.0 / 3.0);
        //}

        //zs::complex<double> u2 = (-1.0 + sqrt(3.0) * i) / 2.0;
        //zs::complex<double> u3 = (-1.0 - sqrt(3.0) * i) / 2.0;

        //zs::complex<double> t1 = (b + C + delta0 / C) / (-3.0 * a);
        //zs::complex<double> t2 = (b + u2 * C + delta0 / (u2 * C)) / (-3.0 * a);
        //zs::complex<double> t3 = (b + u3 * C + delta0 / (u3 * C)) / (-3.0 * a);
        //t                      = -1;
        //if((abs(imag(t1)) < errorRate /** errorRate*/) && (real(t1) > 0))
        //    t = real(t1);
        //if((abs(imag(t2)) < errorRate /** errorRate*/) && (real(t2) > 0)
        //   && ((real(t2) < t) || (t < 0)))
        //    t = real(t2);
        //if((abs(imag(t3)) < errorRate /** errorRate*/) && (real(t3) > 0)
        //   && ((real(t3) < t) || (t < 0)))
        //    t = real(t3);
    }
    if(t <= 0)
        t = 1;
    return t;
}

__device__ double __cal_Friction_gd_energy(const double3* _vertexes,
                                           const double3* _o_vertexes,
                                           const double3* _normal,
                                           uint32_t       gidx,
                                           double         dt,
                                           double         lastH,
                                           double         eps)
{

    double3 normal = *_normal;
    double3 Vdiff  = __GEIGEN__::__minus(_vertexes[gidx], _o_vertexes[gidx]);
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


__device__ double __cal_Friction_energy(const double3*         _vertexes,
                                        const double3*         _o_vertexes,
                                        int4                   MMCVIDI,
                                        double                 dt,
                                        double2                distCoord,
                                        __GEIGEN__::Matrix3x2d tanBasis,
                                        double                 lastH,
                                        double                 fricDHat,
                                        double                 eps)
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

__global__ void _calFrictionHessian_gd(const double3*   _vertexes,
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
                                       double           coef)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    double                 eps           = sqrt(eps2);
    unsigned int           gidx          = _last_collisionPair_gd[idx];
    double                 multiplier_vI = coef * lastH[idx];
    __GEIGEN__::Matrix3x3d H_vI;

    double3 Vdiff  = __GEIGEN__::__minus(_vertexes[gidx], _o_vertexes[gidx]);
    double3 normal = *_normal;
    double3 VProj  = __GEIGEN__::__minus(
        Vdiff, __GEIGEN__::__s_vec_multiply(normal, __GEIGEN__::__v_vec_dot(Vdiff, normal)));
    double VProjMag2 = __GEIGEN__::__squaredNorm(VProj);

    // Build tangent basis from ground normal (works for any orientation).
    double3 t1, t2;
    {
        double3 ref = make_double3(1.0, 0.0, 0.0);
        if(fabs(normal.x) > 0.9)
            ref = make_double3(0.0, 1.0, 0.0);
        t1 = __GEIGEN__::__v_vec_cross(normal, ref);
        double t1_len = sqrt(__GEIGEN__::__squaredNorm(t1));
        t1 = __GEIGEN__::__s_vec_multiply(t1, 1.0 / t1_len);
        t2 = __GEIGEN__::__v_vec_cross(normal, t1);
    }

    double2 relDX = make_double2(__GEIGEN__::__v_vec_dot(t1, VProj),
                                 __GEIGEN__::__v_vec_dot(t2, VProj));
    double  relDXSqNorm = relDX.x * relDX.x + relDX.y * relDX.y;

    __GEIGEN__::Matrix2x2d projH;

    if(relDXSqNorm > eps2)
    {
        double relDXNorm = sqrt(relDXSqNorm);

        __GEIGEN__::__set_Mat2x2_val_column(projH, make_double2(0, 0), make_double2(0, 0));

        double  eigenValues[2];
        int     eigenNum = 0;
        double2 eigenVecs[2];
        __GEIGEN__::__makePD2x2(
            relDX.x * relDX.x * -multiplier_vI / relDXSqNorm / relDXNorm
                + (multiplier_vI / relDXNorm),
            relDX.x * relDX.y * -multiplier_vI / relDXSqNorm / relDXNorm,
            relDX.x * relDX.y * -multiplier_vI / relDXSqNorm / relDXNorm,
            relDX.y * relDX.y * -multiplier_vI / relDXSqNorm / relDXNorm
                + (multiplier_vI / relDXNorm),
            eigenValues,
            eigenNum,
            eigenVecs);
        for(int i = 0; i < eigenNum; i++)
        {
            if(eigenValues[i] > 0)
            {
                __GEIGEN__::Matrix2x2d eigenMatrix =
                    __GEIGEN__::__v2_vec2_toMat2x2(eigenVecs[i], eigenVecs[i]);
                eigenMatrix =
                    __GEIGEN__::__s_Mat2x2_multiply(eigenMatrix, eigenValues[i]);
                projH = __GEIGEN__::__Mat2x2_add(projH, eigenMatrix);
            }
        }
    }
    else
    {
        __GEIGEN__::__set_Mat2x2_val_column(projH,
            make_double2(multiplier_vI / eps, 0),
            make_double2(0, multiplier_vI / eps));
    }

    // Map 2x2 tangent-space Hessian back to 3x3: H = T * projH * T^T
    // where T = [t1 | t2] is a 3x2 matrix.
    // H_ij = sum_ab t_i^a * projH_ab * t_j^b
    double t1a[3] = {t1.x, t1.y, t1.z};
    double t2a[3] = {t2.x, t2.y, t2.z};
    double h[3][3];
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            h[i][j] = t1a[i] * projH.m[0][0] * t1a[j]
                     + t1a[i] * projH.m[0][1] * t2a[j]
                     + t2a[i] * projH.m[1][0] * t1a[j]
                     + t2a[i] * projH.m[1][1] * t2a[j];
        }
    }
    __GEIGEN__::__set_Mat_val(H_vI,
                              h[0][0], h[0][1], h[0][2],
                              h[1][0], h[1][1], h[1][2],
                              h[2][0], h[2][1], h[2][2]);

    write_triplet<3, 3>(triplet_values, row_ids, col_ids, &gidx, H_vI.m, global_offset + idx);
}

__global__ void _calFrictionHessian(const double3*          _vertexes,
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
                                    int                     cd_offset4,
                                    int                     cd_offset3,
                                    int                     cd_offset2,
                                    int                     f_offset4,
                                    int                     f_offset3,
                                    int                     f_offset2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int4    MMCVIDI = _last_collisionPair[idx];
    double  eps     = sqrt(eps2);
    double3 relDX3D;
    int global_offset = cd_offset4 * M12_Off + cd_offset3 * M9_Off + cd_offset2 * M6_Off;
    if(MMCVIDI.x >= 0)
    {
        Friction::computeRelDX_EE(
            __GEIGEN__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
            __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
            __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
            __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]),
            distCoord[idx].x,
            distCoord[idx].y,
            relDX3D);


        __GEIGEN__::Matrix2x3d tB_T = __GEIGEN__::__Transpose3x2(tanBasis[idx]);
        double2 relDX       = __GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D);
        double  relDXSqNorm = __GEIGEN__::__squaredNorm(relDX);
        double  relDXNorm   = sqrt(relDXSqNorm);
        __GEIGEN__::Matrix12x2d T;
        Friction::computeT_EE(tanBasis[idx], distCoord[idx].x, distCoord[idx].y, T);
        __GEIGEN__::Matrix2x2d M2;
        if(relDXSqNorm > eps2)
        {
            __GEIGEN__::__set_Mat_identity(M2);
            M2.m[0][0] /= relDXNorm;
            M2.m[1][1] /= relDXNorm;
            M2 = __GEIGEN__::__Mat2x2_minus(
                M2,
                __GEIGEN__::__s_Mat2x2_multiply(__GEIGEN__::__v2_vec2_toMat2x2(relDX, relDX),
                                                1 / (relDXSqNorm * relDXNorm)));
        }
        else
        {
            double f1_div_relDXNorm;
            Friction::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
            double f2;
            Friction::f2_SF(relDXSqNorm, eps, f2);
            if(f2 != f1_div_relDXNorm && relDXSqNorm)
            {

                __GEIGEN__::__set_Mat_identity(M2);
                M2.m[0][0] *= f1_div_relDXNorm;
                M2.m[1][1] *= f1_div_relDXNorm;
                M2 = __GEIGEN__::__Mat2x2_minus(
                    M2,
                    __GEIGEN__::__s_Mat2x2_multiply(__GEIGEN__::__v2_vec2_toMat2x2(relDX, relDX),
                                                    (f1_div_relDXNorm - f2) / relDXSqNorm));
            }
            else
            {
                __GEIGEN__::__set_Mat_identity(M2);
                M2.m[0][0] *= f1_div_relDXNorm;
                M2.m[1][1] *= f1_div_relDXNorm;
            }
        }

        __GEIGEN__::Matrix2x2d projH;

        Matrix2d F_mat2;
        F_mat2 << M2.m[0][0], M2.m[0][1], M2.m[1][0], M2.m[1][1];
        makePDGeneral<double, 2>(F_mat2);
        projH.m[0][0] = F_mat2(0, 0);
        projH.m[0][1] = F_mat2(0, 1);
        projH.m[1][0] = F_mat2(1, 0);
        projH.m[1][1] = F_mat2(1, 1);


        __GEIGEN__::Matrix12x2d TM2 = __GEIGEN__::__M12x2_M2x2_Multiply(T, projH);

        __GEIGEN__::Matrix12x12d HessianBlock =
            __GEIGEN__::__s_M12x12_Multiply(__M12x2_M12x2T_Multiply(TM2, T),
                                            coef * lastH[idx]);
        int Hidx   = atomicAdd(_cpNum + 4, 1);
        int offset = global_offset + Hidx * M12_Off;
        //Hidx += cd_offset4;
        //H12x12[Hidx]  = HessianBlock;
        uint4 global_index = make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
        //D4Index[Hidx] = global_index;

        write_triplet<12, 12>(
            triplet_values, row_ids, col_ids, &(global_index.x), HessianBlock.m, offset);
    }
    else
    {
        int v0I = -MMCVIDI.x - 1;
        if(MMCVIDI.z < 0)
        {

            MMCVIDI.x = v0I;
            Friction::computeRelDX_PP(
                __GEIGEN__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                relDX3D);

            __GEIGEN__::Matrix2x3d tB_T = __GEIGEN__::__Transpose3x2(tanBasis[idx]);
            double2 relDX       = __GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D);
            double  relDXSqNorm = __GEIGEN__::__squaredNorm(relDX);
            double  relDXNorm   = sqrt(relDXSqNorm);
            __GEIGEN__::Matrix6x2d T;
            Friction::computeT_PP(tanBasis[idx], T);
            __GEIGEN__::Matrix2x2d M2;
            if(relDXSqNorm > eps2)
            {
                __GEIGEN__::__set_Mat_identity(M2);
                M2.m[0][0] /= relDXNorm;
                M2.m[1][1] /= relDXNorm;
                M2 = __GEIGEN__::__Mat2x2_minus(
                    M2,
                    __GEIGEN__::__s_Mat2x2_multiply(__GEIGEN__::__v2_vec2_toMat2x2(relDX, relDX),
                                                    1 / (relDXSqNorm * relDXNorm)));
            }
            else
            {
                double f1_div_relDXNorm;
                Friction::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                double f2;
                Friction::f2_SF(relDXSqNorm, eps, f2);
                if(f2 != f1_div_relDXNorm && relDXSqNorm)
                {

                    __GEIGEN__::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                    M2 = __GEIGEN__::__Mat2x2_minus(
                        M2,
                        __GEIGEN__::__s_Mat2x2_multiply(
                            __GEIGEN__::__v2_vec2_toMat2x2(relDX, relDX),
                            (f1_div_relDXNorm - f2) / relDXSqNorm));
                }
                else
                {
                    __GEIGEN__::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                }
            }
            __GEIGEN__::Matrix2x2d projH;
            Matrix2d               F_mat2;
            F_mat2 << M2.m[0][0], M2.m[0][1], M2.m[1][0], M2.m[1][1];
            makePDGeneral<double, 2>(F_mat2);
            projH.m[0][0] = F_mat2(0, 0);
            projH.m[0][1] = F_mat2(0, 1);
            projH.m[1][0] = F_mat2(1, 0);
            projH.m[1][1] = F_mat2(1, 1);

            __GEIGEN__::Matrix6x2d TM2 = __GEIGEN__::__M6x2_M2x2_Multiply(T, projH);

            __GEIGEN__::Matrix6x6d HessianBlock =
                __GEIGEN__::__s_M6x6_Multiply(__M6x2_M6x2T_Multiply(TM2, T),
                                              coef * lastH[idx]);

            int Hidx   = atomicAdd(_cpNum + 2, 1);
            int offset = global_offset + f_offset4 * M12_Off
                         + f_offset3 * M9_Off + Hidx * M6_Off;
            //Hidx += cd_offset2;
            //H6x6[Hidx]    = HessianBlock;
            uint2 global_index = make_uint2(MMCVIDI.x, MMCVIDI.y);
            //D2Index[Hidx]      = global_index;


            write_triplet<6, 6>(triplet_values,
                                row_ids,
                                col_ids,
                                &(global_index.x),
                                HessianBlock.m,
                                offset);
        }
        else if(MMCVIDI.w < 0)
        {

            MMCVIDI.x = v0I;
            Friction::computeRelDX_PE(
                __GEIGEN__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
                distCoord[idx].x,
                relDX3D);

            __GEIGEN__::Matrix2x3d tB_T = __GEIGEN__::__Transpose3x2(tanBasis[idx]);
            double2 relDX       = __GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D);
            double  relDXSqNorm = __GEIGEN__::__squaredNorm(relDX);
            double  relDXNorm   = sqrt(relDXSqNorm);
            __GEIGEN__::Matrix9x2d T;
            Friction::computeT_PE(tanBasis[idx], distCoord[idx].x, T);
            __GEIGEN__::Matrix2x2d M2;
            if(relDXSqNorm > eps2)
            {
                __GEIGEN__::__set_Mat_identity(M2);
                M2.m[0][0] /= relDXNorm;
                M2.m[1][1] /= relDXNorm;
                M2 = __GEIGEN__::__Mat2x2_minus(
                    M2,
                    __GEIGEN__::__s_Mat2x2_multiply(__GEIGEN__::__v2_vec2_toMat2x2(relDX, relDX),
                                                    1 / (relDXSqNorm * relDXNorm)));
            }
            else
            {
                double f1_div_relDXNorm;
                Friction::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                double f2;
                Friction::f2_SF(relDXSqNorm, eps, f2);
                if(f2 != f1_div_relDXNorm && relDXSqNorm)
                {

                    __GEIGEN__::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                    M2 = __GEIGEN__::__Mat2x2_minus(
                        M2,
                        __GEIGEN__::__s_Mat2x2_multiply(
                            __GEIGEN__::__v2_vec2_toMat2x2(relDX, relDX),
                            (f1_div_relDXNorm - f2) / relDXSqNorm));
                }
                else
                {
                    __GEIGEN__::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                }
            }
            __GEIGEN__::Matrix2x2d projH;
            Matrix2d               F_mat2;
            F_mat2 << M2.m[0][0], M2.m[0][1], M2.m[1][0], M2.m[1][1];
            makePDGeneral<double, 2>(F_mat2);
            projH.m[0][0] = F_mat2(0, 0);
            projH.m[0][1] = F_mat2(0, 1);
            projH.m[1][0] = F_mat2(1, 0);
            projH.m[1][1] = F_mat2(1, 1);

            __GEIGEN__::Matrix9x2d TM2 = __GEIGEN__::__M9x2_M2x2_Multiply(T, projH);

            __GEIGEN__::Matrix9x9d HessianBlock =
                __GEIGEN__::__s_M9x9_Multiply(__M9x2_M9x2T_Multiply(TM2, T),
                                              coef * lastH[idx]);
            int Hidx   = atomicAdd(_cpNum + 3, 1);
            int offset = global_offset + f_offset4 * M12_Off + Hidx * M9_Off;
            //Hidx += cd_offset3;
            //H9x9[Hidx]    = HessianBlock;
            uint3 global_index = make_uint3(v0I, MMCVIDI.y, MMCVIDI.z);
            //D3Index[Hidx]      = global_index;


            write_triplet<9, 9>(triplet_values,
                                row_ids,
                                col_ids,
                                &(global_index.x),
                                HessianBlock.m,
                                offset);
        }
        else
        {
            MMCVIDI.x = v0I;
            Friction::computeRelDX_PT(
                __GEIGEN__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]),
                distCoord[idx].x,
                distCoord[idx].y,
                relDX3D);


            __GEIGEN__::Matrix2x3d tB_T = __GEIGEN__::__Transpose3x2(tanBasis[idx]);
            double2 relDX       = __GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D);
            double  relDXSqNorm = __GEIGEN__::__squaredNorm(relDX);
            double  relDXNorm   = sqrt(relDXSqNorm);
            __GEIGEN__::Matrix12x2d T;
            Friction::computeT_PT(
                tanBasis[idx], distCoord[idx].x, distCoord[idx].y, T);
            __GEIGEN__::Matrix2x2d M2;
            if(relDXSqNorm > eps2)
            {
                __GEIGEN__::__set_Mat_identity(M2);
                M2.m[0][0] /= relDXNorm;
                M2.m[1][1] /= relDXNorm;
                M2 = __GEIGEN__::__Mat2x2_minus(
                    M2,
                    __GEIGEN__::__s_Mat2x2_multiply(__GEIGEN__::__v2_vec2_toMat2x2(relDX, relDX),
                                                    1 / (relDXSqNorm * relDXNorm)));
            }
            else
            {
                double f1_div_relDXNorm;
                Friction::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                double f2;
                Friction::f2_SF(relDXSqNorm, eps, f2);
                if(f2 != f1_div_relDXNorm && relDXSqNorm)
                {

                    __GEIGEN__::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                    M2 = __GEIGEN__::__Mat2x2_minus(
                        M2,
                        __GEIGEN__::__s_Mat2x2_multiply(
                            __GEIGEN__::__v2_vec2_toMat2x2(relDX, relDX),
                            (f1_div_relDXNorm - f2) / relDXSqNorm));
                }
                else
                {
                    __GEIGEN__::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                }
            }
            __GEIGEN__::Matrix2x2d projH;
            Matrix2d               F_mat2;
            F_mat2 << M2.m[0][0], M2.m[0][1], M2.m[1][0], M2.m[1][1];
            makePDGeneral<double, 2>(F_mat2);
            projH.m[0][0] = F_mat2(0, 0);
            projH.m[0][1] = F_mat2(0, 1);
            projH.m[1][0] = F_mat2(1, 0);
            projH.m[1][1] = F_mat2(1, 1);

            __GEIGEN__::Matrix12x2d TM2 = __GEIGEN__::__M12x2_M2x2_Multiply(T, projH);

            __GEIGEN__::Matrix12x12d HessianBlock =
                __GEIGEN__::__s_M12x12_Multiply(__M12x2_M12x2T_Multiply(TM2, T),
                                                coef * lastH[idx]);
            int Hidx   = atomicAdd(_cpNum + 4, 1);
            int offset = global_offset + Hidx * M12_Off;
            //Hidx += cd_offset4;
            //H12x12[Hidx]  = HessianBlock;
            uint4 global_index = make_uint4(v0I, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
            //D4Index[Hidx] = global_index;


            write_triplet<12, 12>(triplet_values,
                                  row_ids,
                                  col_ids,
                                  &(global_index.x),
                                  HessianBlock.m,
                                  offset);
        }
    }
}

template <typename T>
__global__ inline void moveMemory_1(T* data, int output_start, int input_start, int length)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= length)
        return;
    data[output_start + idx] = data[input_start + idx];
}

__global__ void _calBarrierHessian(const double3*   _vertexes,
                                   const double3*   _rest_vertexes,
                                   const int4*      _collisionPair,
                                   Eigen::Matrix3d* triplet_values,
                                   int*             row_ids,
                                   int*             col_ids,
                                   uint32_t*        _cpNum,
                                   int*             matIndex,
                                   double           dHat,
                                   double           Kappa,
                                   int              offset4,
                                   int              offset3,
                                   int              offset2,
                                   int              number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int4   MMCVIDI   = _collisionPair[idx];
    double dHat_sqrt = sqrt(dHat);

    double gassThreshold = 1e-6;
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
            dis = sqrt(dis);
            __GEIGEN__::Matrix12x9d PFPxT;
            pFpx_ee2(_vertexes[MMCVIDI.x],
                     _vertexes[MMCVIDI.y],
                     _vertexes[MMCVIDI.z],
                     _vertexes[MMCVIDI.w],
                     dHat_sqrt,
                     PFPxT);
            double              I5 = pow(dis / dHat_sqrt, 2);
            __GEIGEN__::Vector9 q0;
            q0.v[0] = q0.v[1] = q0.v[2] = q0.v[3] = q0.v[4] = q0.v[5] =
                q0.v[6] = q0.v[7] = 0;
            q0.v[8]               = 1;
            __GEIGEN__::Matrix9x9d H;
            __GEIGEN__::__init_Mat9x9(H, 0);

#if (RANK == 1)
            double lambda0 =
                Kappa
                * (2 * dHat * dHat
                   * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5 - 6 * I5 * I5 * log(I5) + 1))
                / I5;
            if(dis * dis < gassThreshold * dHat)
            {
                double lambda1 =
                    Kappa
                    * (2 * dHat * dHat
                       * (6 * gassThreshold + 2 * gassThreshold * log(gassThreshold)
                          - 7 * gassThreshold * gassThreshold
                          - 6 * gassThreshold * gassThreshold * log(gassThreshold) + 1))
                    / gassThreshold;
                lambda0 = lambda1;
            }
#elif (RANK == 2)
            double lambda0 =
                -(4 * Kappa * dHat * dHat
                  * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 6 * I5 * log(I5)
                     - 2 * I5 * I5 + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2))
                / I5;
            if(dis * dis < gassThreshold * dHat)
            {
                double lambda1 =
                    -(4 * Kappa * dHat * dHat
                      * (4 * gassThreshold + log(gassThreshold)
                         - 3 * gassThreshold * gassThreshold * log(gassThreshold) * log(gassThreshold)
                         + 6 * gassThreshold * log(gassThreshold) - 2 * gassThreshold * gassThreshold
                         + gassThreshold * log(gassThreshold) * log(gassThreshold)
                         - 7 * gassThreshold * gassThreshold * log(gassThreshold) - 2))
                    / gassThreshold;
                lambda0 = lambda1;
            }
#elif (RANK == 3)
            double lambda0 =
                (2 * Kappa * dHat * dHat * log(I5)
                 * (24 * I5 + 3 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5)
                    + 18 * I5 * log(I5) - 12 * I5 * I5
                    + 2 * I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 12))
                / I5;
#elif (RANK == 4)
            double lambda0 =
                -(4 * Kappa * dHat * dHat * log(I5) * log(I5)
                  * (24 * I5 + 2 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                     + 12 * I5 * log(I5) - 12 * I5 * I5 + I5 * log(I5) * log(I5)
                     - 14 * I5 * I5 * log(I5) - 12))
                / I5;
#elif (RANK == 5)
            double lambda0 =
                (2 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                 * (80 * I5 + 5 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5)
                    + 30 * I5 * log(I5) - 40 * I5 * I5
                    + 2 * I5 * log(I5) * log(I5) - 35 * I5 * I5 * log(I5) - 40))
                / I5;
#elif (RANK == 6)
            double lambda0 =
                -(4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5)
                  * (60 * I5 + 3 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                     + 18 * I5 * log(I5) - 30 * I5 * I5 + I5 * log(I5) * log(I5)
                     - 21 * I5 * I5 * log(I5) - 30))
                / I5;
#endif

            H = __GEIGEN__::__S_Mat9x9_multiply(__GEIGEN__::__v9_vec9_toMat9x9(q0, q0), lambda0);

            __GEIGEN__::Matrix12x12d Hessian;  // = __GEIGEN__::__M12x9_M9x12_Multiply(__GEIGEN__::__M12x9_M9x9_Multiply(PFPxT, H), __GEIGEN__::__Transpose12x9(PFPxT));

            __GEIGEN__::__M12x9_S9x9_MT9x12_Multiply(PFPxT, H, Hessian);

            int Hidx = matIndex[idx];

            uint4 global_index =
                make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);

            int triplet_id_offset = Hidx * 16;
            write_triplet<12, 12>(
                triplet_values, row_ids, col_ids, &(global_index.x), Hessian.m, triplet_id_offset);
        }
        else
        {
            MMCVIDI.w = -MMCVIDI.w - 1;
            double3 v0 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.x]);
            double3 v1 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.z]);
            double c  = __GEIGEN__::__norm(__GEIGEN__::__v_vec_cross(v0, v1));
            double I1 = c * c;
            if(I1 == 0)
                return;
            __GEIGEN__::Matrix12x9d PFPx;
            pFpx_pee(_vertexes[MMCVIDI.x],
                     _vertexes[MMCVIDI.y],
                     _vertexes[MMCVIDI.z],
                     _vertexes[MMCVIDI.w],
                     dHat_sqrt,
                     PFPx);

            double dis;
            _d_EE(_vertexes[MMCVIDI.x],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            double I2 = dis / dHat;
            dis       = sqrt(dis);

            __GEIGEN__::Matrix3x3d F;
            __GEIGEN__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
            double3 n1 = make_double3(0, 1, 0);
            double3 n2 = make_double3(0, 0, 1);

            double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                        _rest_vertexes[MMCVIDI.y],
                                        _rest_vertexes[MMCVIDI.z],
                                        _rest_vertexes[MMCVIDI.w]);

#if (RANK == 1)
            double lambda10 =
                Kappa * (4 * dHat * dHat * log(I2) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                / (eps_x * eps_x);
            double lambda11 =
                Kappa * 2
                * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                / (eps_x * eps_x);
            double lambda12 =
                Kappa * 2
                * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                / (eps_x * eps_x);
#elif (RANK == 2)
            double lambda10 = -Kappa
                              * (4 * dHat * dHat * log(I2) * log(I2) * (I2 - 1)
                                 * (I2 - 1) * (3 * I1 - eps_x))
                              / (eps_x * eps_x);
            double lambda11 = -Kappa
                              * (4 * dHat * dHat * log(I2) * log(I2)
                                 * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                              / (eps_x * eps_x);
            double lambda12 = -Kappa
                              * (4 * dHat * dHat * log(I2) * log(I2)
                                 * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                              / (eps_x * eps_x);
#elif (RANK == 4)
            double lambda10 = -Kappa
                              * (4 * dHat * dHat * pow(log(I2), 4) * (I2 - 1)
                                 * (I2 - 1) * (3 * I1 - eps_x))
                              / (eps_x * eps_x);
            double lambda11 = -Kappa
                              * (4 * dHat * dHat * pow(log(I2), 4)
                                 * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                              / (eps_x * eps_x);
            double lambda12 = -Kappa
                              * (4 * dHat * dHat * pow(log(I2), 4)
                                 * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                              / (eps_x * eps_x);
#elif (RANK == 6)
            double lambda10 = -Kappa
                              * (4 * dHat * dHat * pow(log(I2), 6) * (I2 - 1)
                                 * (I2 - 1) * (3 * I1 - eps_x))
                              / (eps_x * eps_x);
            double lambda11 = -Kappa
                              * (4 * dHat * dHat * pow(log(I2), 6)
                                 * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                              / (eps_x * eps_x);
            double lambda12 = -Kappa
                              * (4 * dHat * dHat * pow(log(I2), 6)
                                 * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                              / (eps_x * eps_x);
#endif
            __GEIGEN__::Matrix3x3d fnn;
            __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(n1, n1);
            __GEIGEN__::__M_Mat_multiply(F, nn, fnn);
            __GEIGEN__::Vector9 q10 = __GEIGEN__::__Mat3x3_to_vec9_double(fnn);
            q10 = __GEIGEN__::__s_vec9_multiply(q10, 1.0 / sqrt(I1));

            __GEIGEN__::Matrix3x3d Tx, Ty, Tz;
            __GEIGEN__::__set_Mat_val(Tx, 0, 0, 0, 0, 0, 1, 0, -1, 0);
            __GEIGEN__::__set_Mat_val(Ty, 0, 0, -1, 0, 0, 0, 1, 0, 0);
            __GEIGEN__::__set_Mat_val(Tz, 0, 1, 0, -1, 0, 0, 0, 0, 0);

            double ratio = 1.f / sqrt(2.f);
            Tx           = __S_Mat_multiply(Tx, ratio);
            Ty           = __S_Mat_multiply(Ty, ratio);
            Tz           = __S_Mat_multiply(Tz, ratio);

            __GEIGEN__::Vector9 q11 = __GEIGEN__::__Mat3x3_to_vec9_double(
                __GEIGEN__::__M_Mat_multiply(Tx, fnn));
            __GEIGEN__::__normalized_vec9_double(q11);
            __GEIGEN__::Vector9 q12 = __GEIGEN__::__Mat3x3_to_vec9_double(
                __GEIGEN__::__M_Mat_multiply(Tz, fnn));
            //__GEIGEN__::__s_vec9_multiply(q12, c);
            __GEIGEN__::__normalized_vec9_double(q12);

            __GEIGEN__::Matrix9x9d projectedH;
            __GEIGEN__::__init_Mat9x9(projectedH, 0);

            __GEIGEN__::Matrix9x9d M9_temp = __GEIGEN__::__v9_vec9_toMat9x9(q11, q11);
            M9_temp    = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, lambda11);
            projectedH = __GEIGEN__::__Mat9x9_add(projectedH, M9_temp);

            M9_temp    = __GEIGEN__::__v9_vec9_toMat9x9(q12, q12);
            M9_temp    = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, lambda12);
            projectedH = __GEIGEN__::__Mat9x9_add(projectedH, M9_temp);

#if (RANK == 1)
            double lambda20 =
                -Kappa
                * (2 * I1 * dHat * dHat * (I1 - 2 * eps_x)
                   * (6 * I2 + 2 * I2 * log(I2) - 7 * I2 * I2 - 6 * I2 * I2 * log(I2) + 1))
                / (I2 * eps_x * eps_x);
#elif (RANK == 2)
            double lambda20 =
                Kappa
                * (4 * I1 * dHat * dHat * (I1 - 2 * eps_x)
                   * (4 * I2 + log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                      + 6 * I2 * log(I2) - 2 * I2 * I2 + I2 * log(I2) * log(I2)
                      - 7 * I2 * I2 * log(I2) - 2))
                / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
            double lambda20 =
                Kappa
                * (4 * I1 * dHat * dHat * log(I2) * log(I2) * (I1 - 2 * eps_x)
                   * (24 * I2 + 2 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                      + 12 * I2 * log(I2) - 12 * I2 * I2
                      + I2 * log(I2) * log(I2) - 14 * I2 * I2 * log(I2) - 12))
                / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
            double lambda20 =
                Kappa
                * (4 * I1 * dHat * dHat * pow(log(I2), 4) * (I1 - 2 * eps_x)
                   * (60 * I2 + 3 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                      + 18 * I2 * log(I2) - 30 * I2 * I2
                      + I2 * log(I2) * log(I2) - 21 * I2 * I2 * log(I2) - 30))
                / (I2 * (eps_x * eps_x));
#endif
            nn = __GEIGEN__::__v_vec_toMat(n2, n2);
            __GEIGEN__::__M_Mat_multiply(F, nn, fnn);
            __GEIGEN__::Vector9 q20 = __GEIGEN__::__Mat3x3_to_vec9_double(fnn);
            q20 = __GEIGEN__::__s_vec9_multiply(q20, 1.0 / sqrt(I2));


#if (RANK == 1)
            double lambdag1g = Kappa * 4 * c * F.m[2][2]
                               * ((2 * dHat * dHat * (I1 - eps_x) * (I2 - 1)
                                   * (I2 + 2 * I2 * log(I2) - 1))
                                  / (I2 * eps_x * eps_x));
#elif (RANK == 2)
            double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                               * (4 * dHat * dHat * log(I2) * (I1 - eps_x)
                                  * (I2 - 1) * (I2 + I2 * log(I2) - 1))
                               / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
            double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                               * (4 * dHat * dHat * pow(log(I2), 3) * (I1 - eps_x)
                                  * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2))
                               / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
            double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                               * (4 * dHat * dHat * pow(log(I2), 5) * (I1 - eps_x)
                                  * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3))
                               / (I2 * (eps_x * eps_x));
#endif

            Eigen::Matrix2d FMat2;
            FMat2 << lambda10, lambdag1g, lambdag1g, lambda20;
            makePDGeneral<double, 2>(FMat2);
            projectedH.m[4][4] += FMat2(0, 0);
            projectedH.m[4][8] += FMat2(0, 1);
            projectedH.m[8][4] += FMat2(1, 0);
            projectedH.m[8][8] += FMat2(1, 1);

            __GEIGEN__::Matrix12x12d Hessian;
            __GEIGEN__::__M12x9_S9x9_MT9x12_Multiply(PFPx, projectedH, Hessian);
            int Hidx = matIndex[idx];  //int Hidx = atomicAdd(_cpNum + 4, 1);

            uint4 global_index =
                make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
            int triplet_id_offset = Hidx * 16;
            write_triplet<12, 12>(
                triplet_values, row_ids, col_ids, &(global_index.x), Hessian.m, triplet_id_offset);
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
                //printf("ppp condition  ***************************************\n: %d  %d  %d  %d\n***************************************\n", MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
                double3 v0 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.x]);
                double3 v1 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.y]);
                double c = __GEIGEN__::__norm(__GEIGEN__::__v_vec_cross(v0, v1)) /*/ __GEIGEN__::__norm(v0)*/;
                double I1 = c * c;
                if(I1 == 0)
                    return;
                __GEIGEN__::Matrix12x9d PFPx;
                pFpx_ppp(_vertexes[MMCVIDI.x],
                         _vertexes[MMCVIDI.y],
                         _vertexes[MMCVIDI.z],
                         _vertexes[MMCVIDI.w],
                         dHat_sqrt,
                         PFPx);

                double dis;
                _d_PP(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], dis);
                double I2 = dis / dHat;
                dis       = sqrt(dis);

                __GEIGEN__::Matrix3x3d F;
                __GEIGEN__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
                double3 n1 = make_double3(0, 1, 0);
                double3 n2 = make_double3(0, 0, 1);

                double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                            _rest_vertexes[MMCVIDI.z],
                                            _rest_vertexes[MMCVIDI.y],
                                            _rest_vertexes[MMCVIDI.w]);

#if (RANK == 1)
                double lambda10 =
                    Kappa * (4 * dHat * dHat * log(I2) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                    / (eps_x * eps_x);
                double lambda11 =
                    Kappa * 2
                    * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                    / (eps_x * eps_x);
                double lambda12 =
                    Kappa * 2
                    * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                    / (eps_x * eps_x);
#elif (RANK == 2)
                double lambda10 = -Kappa
                                  * (4 * dHat * dHat * log(I2) * log(I2)
                                     * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                                  / (eps_x * eps_x);
                double lambda11 = -Kappa
                                  * (4 * dHat * dHat * log(I2) * log(I2)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
                double lambda12 = -Kappa
                                  * (4 * dHat * dHat * log(I2) * log(I2)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
#elif (RANK == 4)
                double lambda10 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 4)
                                     * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                                  / (eps_x * eps_x);
                double lambda11 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 4)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
                double lambda12 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 4)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
#elif (RANK == 6)
                double lambda10 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 6)
                                     * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                                  / (eps_x * eps_x);
                double lambda11 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 6)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
                double lambda12 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 6)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
#endif
                __GEIGEN__::Matrix3x3d fnn;
                __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(n1, n1);
                __GEIGEN__::__M_Mat_multiply(F, nn, fnn);
                __GEIGEN__::Vector9 q10 = __GEIGEN__::__Mat3x3_to_vec9_double(fnn);
                q10 = __GEIGEN__::__s_vec9_multiply(q10, 1.0 / sqrt(I1));

                __GEIGEN__::Matrix3x3d Tx, Ty, Tz;
                __GEIGEN__::__set_Mat_val(Tx, 0, 0, 0, 0, 0, 1, 0, -1, 0);
                __GEIGEN__::__set_Mat_val(Ty, 0, 0, -1, 0, 0, 0, 1, 0, 0);
                __GEIGEN__::__set_Mat_val(Tz, 0, 1, 0, -1, 0, 0, 0, 0, 0);

                double ratio = 1.f / sqrt(2.f);
                Tx           = __S_Mat_multiply(Tx, ratio);
                Ty           = __S_Mat_multiply(Ty, ratio);
                Tz           = __S_Mat_multiply(Tz, ratio);

                __GEIGEN__::Vector9 q11 = __GEIGEN__::__Mat3x3_to_vec9_double(
                    __GEIGEN__::__M_Mat_multiply(Tx, fnn));
                __GEIGEN__::__normalized_vec9_double(q11);
                __GEIGEN__::Vector9 q12 = __GEIGEN__::__Mat3x3_to_vec9_double(
                    __GEIGEN__::__M_Mat_multiply(Tz, fnn));
                //__GEIGEN__::__s_vec9_multiply(q12, c);
                __GEIGEN__::__normalized_vec9_double(q12);

                __GEIGEN__::Matrix9x9d projectedH;
                __GEIGEN__::__init_Mat9x9(projectedH, 0);

                __GEIGEN__::Matrix9x9d M9_temp = __GEIGEN__::__v9_vec9_toMat9x9(q11, q11);
                M9_temp    = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, lambda11);
                projectedH = __GEIGEN__::__Mat9x9_add(projectedH, M9_temp);

                M9_temp    = __GEIGEN__::__v9_vec9_toMat9x9(q12, q12);
                M9_temp    = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, lambda12);
                projectedH = __GEIGEN__::__Mat9x9_add(projectedH, M9_temp);

#if (RANK == 1)
                double lambda20 = -Kappa
                                  * (2 * I1 * dHat * dHat * (I1 - 2 * eps_x)
                                     * (6 * I2 + 2 * I2 * log(I2) - 7 * I2 * I2
                                        - 6 * I2 * I2 * log(I2) + 1))
                                  / (I2 * eps_x * eps_x);
#elif (RANK == 2)
                double lambda20 =
                    Kappa
                    * (4 * I1 * dHat * dHat * (I1 - 2 * eps_x)
                       * (4 * I2 + log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                          + 6 * I2 * log(I2) - 2 * I2 * I2
                          + I2 * log(I2) * log(I2) - 7 * I2 * I2 * log(I2) - 2))
                    / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                double lambda20 =
                    Kappa
                    * (4 * I1 * dHat * dHat * log(I2) * log(I2) * (I1 - 2 * eps_x)
                       * (24 * I2 + 2 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                          + 12 * I2 * log(I2) - 12 * I2 * I2
                          + I2 * log(I2) * log(I2) - 14 * I2 * I2 * log(I2) - 12))
                    / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                double lambda20 =
                    Kappa
                    * (4 * I1 * dHat * dHat * pow(log(I2), 4) * (I1 - 2 * eps_x)
                       * (60 * I2 + 3 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                          + 18 * I2 * log(I2) - 30 * I2 * I2
                          + I2 * log(I2) * log(I2) - 21 * I2 * I2 * log(I2) - 30))
                    / (I2 * (eps_x * eps_x));
#endif
                nn = __GEIGEN__::__v_vec_toMat(n2, n2);
                __GEIGEN__::__M_Mat_multiply(F, nn, fnn);
                __GEIGEN__::Vector9 q20 = __GEIGEN__::__Mat3x3_to_vec9_double(fnn);
                q20 = __GEIGEN__::__s_vec9_multiply(q20, 1.0 / sqrt(I2));


#if (RANK == 1)
                double lambdag1g = Kappa * 4 * c * F.m[2][2]
                                   * ((2 * dHat * dHat * (I1 - eps_x) * (I2 - 1)
                                       * (I2 + 2 * I2 * log(I2) - 1))
                                      / (I2 * eps_x * eps_x));
#elif (RANK == 2)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                                   * (4 * dHat * dHat * log(I2) * (I1 - eps_x)
                                      * (I2 - 1) * (I2 + I2 * log(I2) - 1))
                                   / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                                   * (4 * dHat * dHat * pow(log(I2), 3) * (I1 - eps_x)
                                      * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2))
                                   / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                                   * (4 * dHat * dHat * pow(log(I2), 5) * (I1 - eps_x)
                                      * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3))
                                   / (I2 * (eps_x * eps_x));
#endif
                Eigen::Matrix2d FMat2;
                FMat2 << lambda10, lambdag1g, lambdag1g, lambda20;
                makePDGeneral<double, 2>(FMat2);
                projectedH.m[4][4] += FMat2(0, 0);
                projectedH.m[4][8] += FMat2(0, 1);
                projectedH.m[8][4] += FMat2(1, 0);
                projectedH.m[8][8] += FMat2(1, 1);

                //__GEIGEN__::Matrix9x12d PFPxTransPos = __GEIGEN__::__Transpose12x9(PFPx);
                __GEIGEN__::Matrix12x12d Hessian;  // = __GEIGEN__::__M12x9_M9x12_Multiply(__GEIGEN__::__M12x9_M9x9_Multiply(PFPx, projectedH), PFPxTransPos);
                __GEIGEN__::__M12x9_S9x9_MT9x12_Multiply(PFPx, projectedH, Hessian);
                int Hidx = matIndex[idx];  //atomicAdd(_cpNum + 4, 1);

                uint4 global_index =
                    make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
                //D4Index[Hidx] = global_index;


                int triplet_id_offset = Hidx * 16;
                write_triplet<12, 12>(
                    triplet_values, row_ids, col_ids, &(global_index.x), Hessian.m, triplet_id_offset);
            }
            else
            {
#ifdef NEWF
                double dis;
                _d_PP(_vertexes[v0I], _vertexes[MMCVIDI.y], dis);
                dis                            = sqrt(dis);
                double              d_hat_sqrt = sqrt(dHat);
                __GEIGEN__::Vector6 PFPxT;
                pFpx_pp2(_vertexes[v0I], _vertexes[MMCVIDI.y], d_hat_sqrt, PFPxT);
                double I5 = pow(dis / d_hat_sqrt, 2);
                //double q0 = 1;
#else
                double3 v0 = __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[v0I]);
                double3 Ds  = v0;
                double  dis = __GEIGEN__::__norm(v0);
                //if (dis > dHat_sqrt) return;
                double3 vec_normal =
                    __GEIGEN__::__normalized(make_double3(-v0.x, -v0.y, -v0.z));
                double3 target = make_double3(0, 1, 0);
                double3 vec    = __GEIGEN__::__v_vec_cross(vec_normal, target);
                double  cos    = __GEIGEN__::__v_vec_dot(vec_normal, target);
                __GEIGEN__::Matrix3x3d rotation;
                __GEIGEN__::__set_Mat_val(rotation, 1, 0, 0, 0, 1, 0, 0, 0, 1);
                if(cos + 1 == 0)
                {
                    rotation.m[0][0] = -1;
                    rotation.m[1][1] = -1;
                }
                else
                {
                    //pDmpx_pp(_vertexes[v0I], _vertexes[MMCVIDI.y], dHat_sqrt, PDmPx);
                    __GEIGEN__::Matrix3x3d cross_vec;
                    __GEIGEN__::__set_Mat_val(
                        cross_vec, 0, -vec.z, vec.y, vec.z, 0, -vec.x, -vec.y, vec.x, 0);

                    rotation = __GEIGEN__::__Mat_add(
                        rotation,
                        __GEIGEN__::__Mat_add(cross_vec,
                                              __GEIGEN__::__S_Mat_multiply(
                                                  __GEIGEN__::__M_Mat_multiply(cross_vec, cross_vec),
                                                  1.0 / (1 + cos))));
                }

                double3 pos0 = __GEIGEN__::__add(
                    _vertexes[v0I],
                    __GEIGEN__::__s_vec_multiply(vec_normal, dHat_sqrt - dis));
                double3 rotate_uv0 = __GEIGEN__::__M_v_multiply(rotation, pos0);
                double3 rotate_uv1 =
                    __GEIGEN__::__M_v_multiply(rotation, _vertexes[MMCVIDI.y]);

                double uv0 = rotate_uv0.y;
                double uv1 = rotate_uv1.y;

                double u0    = uv1 - uv0;
                double Dm    = u0;
                double DmInv = 1 / u0;

                double3 F  = __GEIGEN__::__s_vec_multiply(Ds, DmInv);
                double  I5 = __GEIGEN__::__squaredNorm(F);

                double3 fnn = F;

                __GEIGEN__::Matrix3x6d PFPx = __computePFDsPX3D_3x6_double(DmInv);
#endif


#if (RANK == 1)
                double lambda0 = Kappa
                                 * (2 * dHat * dHat
                                    * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5
                                       - 6 * I5 * I5 * log(I5) + 1))
                                 / I5;
                if(dis * dis < gassThreshold * dHat)
                {
                    double lambda1 =
                        Kappa
                        * (2 * dHat * dHat
                           * (6 * gassThreshold + 2 * gassThreshold * log(gassThreshold)
                              - 7 * gassThreshold * gassThreshold
                              - 6 * gassThreshold * gassThreshold * log(gassThreshold) + 1))
                        / gassThreshold;
                    lambda0 = lambda1;
                }
#elif (RANK == 2)
                double lambda0 =
                    -(4 * Kappa * dHat * dHat
                      * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                         + 6 * I5 * log(I5) - 2 * I5 * I5
                         + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2))
                    / I5;
                if(dis * dis < gassThreshold * dHat)
                {
                    double lambda1 =
                        -(4 * Kappa * dHat * dHat
                          * (4 * gassThreshold + log(gassThreshold)
                             - 3 * gassThreshold * gassThreshold
                                   * log(gassThreshold) * log(gassThreshold)
                             + 6 * gassThreshold * log(gassThreshold) - 2 * gassThreshold * gassThreshold
                             + gassThreshold * log(gassThreshold) * log(gassThreshold)
                             - 7 * gassThreshold * gassThreshold * log(gassThreshold) - 2))
                        / gassThreshold;
                    lambda0 = lambda1;
                }
#elif (RANK == 3)
                double lambda0 =
                    (2 * Kappa * dHat * dHat * log(I5)
                     * (24 * I5 + 3 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5)
                        + 18 * I5 * log(I5) - 12 * I5 * I5
                        + 2 * I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 12))
                    / I5;
#elif (RANK == 4)
                double lambda0 =
                    -(4 * Kappa * dHat * dHat * log(I5) * log(I5)
                      * (24 * I5 + 2 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                         + 12 * I5 * log(I5) - 12 * I5 * I5
                         + I5 * log(I5) * log(I5) - 14 * I5 * I5 * log(I5) - 12))
                    / I5;
#elif (RANK == 5)
                double lambda0 =
                    (2 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * (80 * I5 + 5 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5)
                        + 30 * I5 * log(I5) - 40 * I5 * I5
                        + 2 * I5 * log(I5) * log(I5) - 35 * I5 * I5 * log(I5) - 40))
                    / I5;
#elif (RANK == 6)
                double lambda0 =
                    -(4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5)
                      * (60 * I5 + 3 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                         + 18 * I5 * log(I5) - 30 * I5 * I5
                         + I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 30))
                    / I5;
#endif

#ifdef NEWF
                double                 H       = lambda0;
                __GEIGEN__::Matrix6x6d Hessian = __GEIGEN__::__s_M6x6_Multiply(
                    __GEIGEN__::__v6_vec6_toMat6x6(PFPxT, PFPxT), H);
#else
                double3 q0 = __GEIGEN__::__s_vec_multiply(F, 1 / sqrt(I5));

                __GEIGEN__::Matrix3x3d H =
                    __GEIGEN__::__S_Mat_multiply(__GEIGEN__::__v_vec_toMat(q0, q0),
                                                 lambda0);  //lambda0 * q0 * q0.transpose();

                __GEIGEN__::Matrix6x3d PFPxTransPos = __GEIGEN__::__Transpose3x6(PFPx);
                __GEIGEN__::Matrix6x6d Hessian = __GEIGEN__::__M6x3_M3x6_Multiply(
                    __GEIGEN__::__M6x3_M3x3_Multiply(PFPxTransPos, H), PFPx);
#endif
                int Hidx = matIndex[idx];  //atomicAdd(_cpNum + 4, 1);

                uint2 global_index = make_uint2(v0I, MMCVIDI.y);
                //D2Index[Hidx]      = global_index;

                int triplet_id_offset = Hidx * 4 + offset3 * 9 + offset4 * 16;
                write_triplet<6, 6>(
                    triplet_values, row_ids, col_ids, &(global_index.x), Hessian.m, triplet_id_offset);
            }
        }
        else if(MMCVIDI.w < 0)
        {
            if(MMCVIDI.y < 0)
            {
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.w = -MMCVIDI.w - 1;
                MMCVIDI.x = v0I;
                //printf("ppe condition  ***************************************\n: %d  %d  %d  %d\n***************************************\n", MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
                double3 v0 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.x]);
                double3 v1 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.y]);
                double c = __GEIGEN__::__norm(__GEIGEN__::__v_vec_cross(v0, v1)) /*/ __GEIGEN__::__norm(v0)*/;
                double I1 = c * c;
                if(I1 == 0)
                    return;
                __GEIGEN__::Matrix12x9d PFPx;
                pFpx_ppe(_vertexes[MMCVIDI.x],
                         _vertexes[MMCVIDI.y],
                         _vertexes[MMCVIDI.z],
                         _vertexes[MMCVIDI.w],
                         dHat_sqrt,
                         PFPx);

                double dis;
                _d_PE(_vertexes[MMCVIDI.x],
                      _vertexes[MMCVIDI.y],
                      _vertexes[MMCVIDI.z],
                      dis);
                double I2 = dis / dHat;
                dis       = sqrt(dis);

                __GEIGEN__::Matrix3x3d F;
                __GEIGEN__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
                double3 n1 = make_double3(0, 1, 0);
                double3 n2 = make_double3(0, 0, 1);

                double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                            _rest_vertexes[MMCVIDI.w],
                                            _rest_vertexes[MMCVIDI.y],
                                            _rest_vertexes[MMCVIDI.z]);

#if (RANK == 1)
                double lambda10 =
                    Kappa * (4 * dHat * dHat * log(I2) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                    / (eps_x * eps_x);
                double lambda11 =
                    Kappa * 2
                    * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                    / (eps_x * eps_x);
                double lambda12 =
                    Kappa * 2
                    * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                    / (eps_x * eps_x);
#elif (RANK == 2)
                double lambda10 = -Kappa
                                  * (4 * dHat * dHat * log(I2) * log(I2)
                                     * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                                  / (eps_x * eps_x);
                double lambda11 = -Kappa
                                  * (4 * dHat * dHat * log(I2) * log(I2)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
                double lambda12 = -Kappa
                                  * (4 * dHat * dHat * log(I2) * log(I2)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
#elif (RANK == 4)
                double lambda10 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 4)
                                     * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                                  / (eps_x * eps_x);
                double lambda11 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 4)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
                double lambda12 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 4)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
#elif (RANK == 6)
                double lambda10 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 6)
                                     * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                                  / (eps_x * eps_x);
                double lambda11 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 6)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
                double lambda12 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 6)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
#endif
                __GEIGEN__::Matrix3x3d fnn;
                __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(n1, n1);
                __GEIGEN__::__M_Mat_multiply(F, nn, fnn);
                __GEIGEN__::Vector9 q10 = __GEIGEN__::__Mat3x3_to_vec9_double(fnn);
                q10 = __GEIGEN__::__s_vec9_multiply(q10, 1.0 / sqrt(I1));

                __GEIGEN__::Matrix3x3d Tx, Ty, Tz;
                __GEIGEN__::__set_Mat_val(Tx, 0, 0, 0, 0, 0, 1, 0, -1, 0);
                __GEIGEN__::__set_Mat_val(Ty, 0, 0, -1, 0, 0, 0, 1, 0, 0);
                __GEIGEN__::__set_Mat_val(Tz, 0, 1, 0, -1, 0, 0, 0, 0, 0);

                double ratio = 1.f / sqrt(2.f);
                Tx           = __S_Mat_multiply(Tx, ratio);
                Ty           = __S_Mat_multiply(Ty, ratio);
                Tz           = __S_Mat_multiply(Tz, ratio);

                __GEIGEN__::Vector9 q11 = __GEIGEN__::__Mat3x3_to_vec9_double(
                    __GEIGEN__::__M_Mat_multiply(Tx, fnn));
                __GEIGEN__::__normalized_vec9_double(q11);
                __GEIGEN__::Vector9 q12 = __GEIGEN__::__Mat3x3_to_vec9_double(
                    __GEIGEN__::__M_Mat_multiply(Tz, fnn));
                //__GEIGEN__::__s_vec9_multiply(q12, c);
                __GEIGEN__::__normalized_vec9_double(q12);

                __GEIGEN__::Matrix9x9d projectedH;
                __GEIGEN__::__init_Mat9x9(projectedH, 0);

                __GEIGEN__::Matrix9x9d M9_temp = __GEIGEN__::__v9_vec9_toMat9x9(q11, q11);
                M9_temp    = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, lambda11);
                projectedH = __GEIGEN__::__Mat9x9_add(projectedH, M9_temp);

                M9_temp    = __GEIGEN__::__v9_vec9_toMat9x9(q12, q12);
                M9_temp    = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, lambda12);
                projectedH = __GEIGEN__::__Mat9x9_add(projectedH, M9_temp);

#if (RANK == 1)
                double lambda20 = -Kappa
                                  * (2 * I1 * dHat * dHat * (I1 - 2 * eps_x)
                                     * (6 * I2 + 2 * I2 * log(I2) - 7 * I2 * I2
                                        - 6 * I2 * I2 * log(I2) + 1))
                                  / (I2 * eps_x * eps_x);
#elif (RANK == 2)
                double lambda20 =
                    Kappa
                    * (4 * I1 * dHat * dHat * (I1 - 2 * eps_x)
                       * (4 * I2 + log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                          + 6 * I2 * log(I2) - 2 * I2 * I2
                          + I2 * log(I2) * log(I2) - 7 * I2 * I2 * log(I2) - 2))
                    / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                double lambda20 =
                    Kappa
                    * (4 * I1 * dHat * dHat * log(I2) * log(I2) * (I1 - 2 * eps_x)
                       * (24 * I2 + 2 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                          + 12 * I2 * log(I2) - 12 * I2 * I2
                          + I2 * log(I2) * log(I2) - 14 * I2 * I2 * log(I2) - 12))
                    / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                double lambda20 =
                    Kappa
                    * (4 * I1 * dHat * dHat * pow(log(I2), 4) * (I1 - 2 * eps_x)
                       * (60 * I2 + 3 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                          + 18 * I2 * log(I2) - 30 * I2 * I2
                          + I2 * log(I2) * log(I2) - 21 * I2 * I2 * log(I2) - 30))
                    / (I2 * (eps_x * eps_x));
#endif
                nn = __GEIGEN__::__v_vec_toMat(n2, n2);
                __GEIGEN__::__M_Mat_multiply(F, nn, fnn);
                __GEIGEN__::Vector9 q20 = __GEIGEN__::__Mat3x3_to_vec9_double(fnn);
                q20 = __GEIGEN__::__s_vec9_multiply(q20, 1.0 / sqrt(I2));


#if (RANK == 1)
                double lambdag1g = Kappa * 4 * c * F.m[2][2]
                                   * ((2 * dHat * dHat * (I1 - eps_x) * (I2 - 1)
                                       * (I2 + 2 * I2 * log(I2) - 1))
                                      / (I2 * eps_x * eps_x));
#elif (RANK == 2)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                                   * (4 * dHat * dHat * log(I2) * (I1 - eps_x)
                                      * (I2 - 1) * (I2 + I2 * log(I2) - 1))
                                   / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                                   * (4 * dHat * dHat * pow(log(I2), 3) * (I1 - eps_x)
                                      * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2))
                                   / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                                   * (4 * dHat * dHat * pow(log(I2), 5) * (I1 - eps_x)
                                      * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3))
                                   / (I2 * (eps_x * eps_x));
#endif
                Eigen::Matrix2d FMat2;
                FMat2 << lambda10, lambdag1g, lambdag1g, lambda20;
                makePDGeneral<double, 2>(FMat2);
                projectedH.m[4][4] += FMat2(0, 0);
                projectedH.m[4][8] += FMat2(0, 1);
                projectedH.m[8][4] += FMat2(1, 0);
                projectedH.m[8][8] += FMat2(1, 1);
                //__GEIGEN__::Matrix9x12d PFPxTransPos = __GEIGEN__::__Transpose12x9(PFPx);
                __GEIGEN__::Matrix12x12d Hessian;  // = __GEIGEN__::__M12x9_M9x12_Multiply(__GEIGEN__::__M12x9_M9x9_Multiply(PFPx, projectedH), PFPxTransPos);
                __GEIGEN__::__M12x9_S9x9_MT9x12_Multiply(PFPx, projectedH, Hessian);
                int Hidx = matIndex[idx];  //atomicAdd(_cpNum + 4, 1);

                uint4 global_index =
                    make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);

                //D4Index[Hidx] = global_index;


                int triplet_id_offset = Hidx * 16;
                write_triplet<12, 12>(
                    triplet_values, row_ids, col_ids, &(global_index.x), Hessian.m, triplet_id_offset);
            }
            else
            {
#ifdef NEWF
                double dis;
                _d_PE(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dis);
                dis                               = sqrt(dis);
                double                 d_hat_sqrt = sqrt(dHat);
                __GEIGEN__::Matrix9x4d PFPxT;
                pFpx_pe2(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], d_hat_sqrt, PFPxT);
                double              I5 = pow(dis / d_hat_sqrt, 2);
                __GEIGEN__::Vector4 q0;
                q0.v[0] = q0.v[1] = q0.v[2] = 0;
                q0.v[3]                     = 1;

                __GEIGEN__::Matrix4x4d H;
                //__GEIGEN__::__init_Mat4x4_val(H, 0);
#else
                double3 v0 = __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[v0I]);
                double3 v1 = __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[v0I]);


                __GEIGEN__::Matrix3x2d Ds;
                __GEIGEN__::__set_Mat3x2_val_column(Ds, v0, v1);

                double3 triangle_normal =
                    __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(v0, v1));
                double3 target = make_double3(0, 1, 0);

                double3 vec = __GEIGEN__::__v_vec_cross(triangle_normal, target);
                double cos = __GEIGEN__::__v_vec_dot(triangle_normal, target);

                double3 edge_normal = __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z]),
                    triangle_normal));
                double dis = __GEIGEN__::__v_vec_dot(
                    __GEIGEN__::__minus(_vertexes[v0I], _vertexes[MMCVIDI.y]), edge_normal);

                //if (dis > dHat_sqrt) return;

                __GEIGEN__::Matrix3x3d rotation;
                __GEIGEN__::__set_Mat_val(rotation, 1, 0, 0, 0, 1, 0, 0, 0, 1);

                __GEIGEN__::Matrix9x4d PDmPx;

                if(cos + 1 == 0)
                {
                    rotation.m[0][0] = -1;
                    rotation.m[1][1] = -1;
                }
                else
                {
                    //pDmpx_pe(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dHat_sqrt, PDmPx);
                    __GEIGEN__::Matrix3x3d cross_vec;
                    __GEIGEN__::__set_Mat_val(
                        cross_vec, 0, -vec.z, vec.y, vec.z, 0, -vec.x, -vec.y, vec.x, 0);

                    rotation = __GEIGEN__::__Mat_add(
                        rotation,
                        __GEIGEN__::__Mat_add(cross_vec,
                                              __GEIGEN__::__S_Mat_multiply(
                                                  __GEIGEN__::__M_Mat_multiply(cross_vec, cross_vec),
                                                  1.0 / (1 + cos))));
                }

                double3 pos0 = __GEIGEN__::__add(
                    _vertexes[v0I],
                    __GEIGEN__::__s_vec_multiply(edge_normal, dHat_sqrt - dis));

                double3 rotate_uv0 = __GEIGEN__::__M_v_multiply(rotation, pos0);
                double3 rotate_uv1 =
                    __GEIGEN__::__M_v_multiply(rotation, _vertexes[MMCVIDI.y]);
                double3 rotate_uv2 =
                    __GEIGEN__::__M_v_multiply(rotation, _vertexes[MMCVIDI.z]);
                double3 rotate_normal = __GEIGEN__::__M_v_multiply(rotation, edge_normal);

                double2 uv0    = make_double2(rotate_uv0.x, rotate_uv0.z);
                double2 uv1    = make_double2(rotate_uv1.x, rotate_uv1.z);
                double2 uv2    = make_double2(rotate_uv2.x, rotate_uv2.z);
                double2 normal = make_double2(rotate_normal.x, rotate_normal.z);

                double2 u0 = __GEIGEN__::__minus_v2(uv1, uv0);
                double2 u1 = __GEIGEN__::__minus_v2(uv2, uv0);

                __GEIGEN__::Matrix2x2d Dm;

                __GEIGEN__::__set_Mat2x2_val_column(Dm, u0, u1);

                __GEIGEN__::Matrix2x2d DmInv;
                __GEIGEN__::__Inverse2x2(Dm, DmInv);

                __GEIGEN__::Matrix3x2d F = __GEIGEN__::__M3x2_M2x2_Multiply(Ds, DmInv);

                double3 FxN = __GEIGEN__::__M3x2_v2_multiply(F, normal);
                double  I5  = __GEIGEN__::__squaredNorm(FxN);

                __GEIGEN__::Matrix3x2d fnn;

                __GEIGEN__::Matrix2x2d nn = __GEIGEN__::__v2_vec2_toMat2x2(normal, normal);

                fnn = __GEIGEN__::__M3x2_M2x2_Multiply(F, nn);

                __GEIGEN__::Matrix6x9d PFPx = __computePFDsPX3D_6x9_double(DmInv);
#endif

#if (RANK == 1)
                double lambda0 = Kappa
                                 * (2 * dHat * dHat
                                    * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5
                                       - 6 * I5 * I5 * log(I5) + 1))
                                 / I5;
                if(dis * dis < gassThreshold * dHat)
                {
                    double lambda1 =
                        Kappa
                        * (2 * dHat * dHat
                           * (6 * gassThreshold + 2 * gassThreshold * log(gassThreshold)
                              - 7 * gassThreshold * gassThreshold
                              - 6 * gassThreshold * gassThreshold * log(gassThreshold) + 1))
                        / gassThreshold;
                    lambda0 = lambda1;
                }
#elif (RANK == 2)
                double lambda0 =
                    -(4 * Kappa * dHat * dHat
                      * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                         + 6 * I5 * log(I5) - 2 * I5 * I5
                         + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2))
                    / I5;
                if(dis * dis < gassThreshold * dHat)
                {
                    double lambda1 =
                        -(4 * Kappa * dHat * dHat
                          * (4 * gassThreshold + log(gassThreshold)
                             - 3 * gassThreshold * gassThreshold
                                   * log(gassThreshold) * log(gassThreshold)
                             + 6 * gassThreshold * log(gassThreshold) - 2 * gassThreshold * gassThreshold
                             + gassThreshold * log(gassThreshold) * log(gassThreshold)
                             - 7 * gassThreshold * gassThreshold * log(gassThreshold) - 2))
                        / gassThreshold;
                    lambda0 = lambda1;
                }
#elif (RANK == 3)
                double lambda0 =
                    (2 * Kappa * dHat * dHat * log(I5)
                     * (24 * I5 + 3 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5)
                        + 18 * I5 * log(I5) - 12 * I5 * I5
                        + 2 * I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 12))
                    / I5;
#elif (RANK == 4)
                double lambda0 =
                    -(4 * Kappa * dHat * dHat * log(I5) * log(I5)
                      * (24 * I5 + 2 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                         + 12 * I5 * log(I5) - 12 * I5 * I5
                         + I5 * log(I5) * log(I5) - 14 * I5 * I5 * log(I5) - 12))
                    / I5;
#elif (RANK == 5)
                double lambda0 =
                    (2 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * (80 * I5 + 5 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5)
                        + 30 * I5 * log(I5) - 40 * I5 * I5
                        + 2 * I5 * log(I5) * log(I5) - 35 * I5 * I5 * log(I5) - 40))
                    / I5;
#elif (RANK == 6)
                double lambda0 =
                    -(4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5)
                      * (60 * I5 + 3 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                         + 18 * I5 * log(I5) - 30 * I5 * I5
                         + I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 30))
                    / I5;
#endif

#ifdef NEWF
                H = __GEIGEN__::__S_Mat4x4_multiply(
                    __GEIGEN__::__v4_vec4_toMat4x4(q0, q0), lambda0);

                __GEIGEN__::Matrix9x9d Hessian;  // = __GEIGEN__::__M9x4_M4x9_Multiply(__GEIGEN__::__M9x4_M4x4_Multiply(PFPxT, H), __GEIGEN__::__Transpose9x4(PFPxT));
                __M9x4_S4x4_MT4x9_Multiply(PFPxT, H, Hessian);
#else

                __GEIGEN__::Vector6 q0 = __GEIGEN__::__Mat3x2_to_vec6_double(fnn);

                q0 = __GEIGEN__::__s_vec6_multiply(q0, 1.0 / sqrt(I5));

                __GEIGEN__::Matrix6x6d H;
                __GEIGEN__::__init_Mat6x6(H, 0);

                H = __GEIGEN__::__S_Mat6x6_multiply(
                    __GEIGEN__::__v6_vec6_toMat6x6(q0, q0), lambda0);

                __GEIGEN__::Matrix9x6d PFPxTransPos = __GEIGEN__::__Transpose6x9(PFPx);
                __GEIGEN__::Matrix9x9d Hessian = __GEIGEN__::__M9x6_M6x9_Multiply(
                    __GEIGEN__::__M9x6_M6x6_Multiply(PFPxTransPos, H), PFPx);
#endif
                int Hidx = matIndex[idx];  //atomicAdd(_cpNum + 4, 1);

                uint3 global_index = make_uint3(v0I, MMCVIDI.y, MMCVIDI.z);

                //D3Index[Hidx] = global_index;

                int triplet_id_offset = Hidx * 9 + offset4 * 16;
                write_triplet<9, 9>(
                    triplet_values, row_ids, col_ids, &(global_index.x), Hessian.m, triplet_id_offset);
            }
        }
        else
        {
#ifdef NEWF
            double dis;
            //printf("PT: %d %d %d %d\n", v0I, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
            _d_PT(_vertexes[v0I],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            double I5                          = dis / dHat;
            dis                                = sqrt(dis);
            double                  d_hat_sqrt = sqrt(dHat);
            __GEIGEN__::Matrix12x9d PFPxT;
            pFpx_pt2(_vertexes[v0I],
                     _vertexes[MMCVIDI.y],
                     _vertexes[MMCVIDI.z],
                     _vertexes[MMCVIDI.w],
                     d_hat_sqrt,
                     PFPxT);

            __GEIGEN__::Vector9 q0;
            q0.v[0] = q0.v[1] = q0.v[2] = q0.v[3] = q0.v[4] = q0.v[5] =
                q0.v[6] = q0.v[7] = 0;
            q0.v[8]               = 1;


#else
            double3 v0 = __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[v0I]);
            double3 v1 = __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[v0I]);
            double3 v2 = __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[v0I]);

            __GEIGEN__::Matrix3x3d Ds;
            __GEIGEN__::__set_Mat_val_column(Ds, v0, v1, v2);

            double3 normal = __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.y]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.y])));
            double  dis    = __GEIGEN__::__v_vec_dot(v0, normal);

            if(dis > 0)
            {
                normal = make_double3(-normal.x, -normal.y, -normal.z);
            }
            else
            {
                dis = -dis;
            }

            double3 pos0 = __GEIGEN__::__add(
                _vertexes[v0I], __GEIGEN__::__s_vec_multiply(normal, dHat_sqrt - dis));


            double3 u0 = __GEIGEN__::__minus(_vertexes[MMCVIDI.y], pos0);
            double3 u1 = __GEIGEN__::__minus(_vertexes[MMCVIDI.z], pos0);
            double3 u2 = __GEIGEN__::__minus(_vertexes[MMCVIDI.w], pos0);

            __GEIGEN__::Matrix3x3d Dm, DmInv;
            __GEIGEN__::__set_Mat_val_column(Dm, u0, u1, u2);

            __GEIGEN__::__Inverse(Dm, DmInv);

            __GEIGEN__::Matrix3x3d F;
            __GEIGEN__::__M_Mat_multiply(Ds, DmInv, F);
            __GEIGEN__::Matrix3x3d uu, vv, ss;
            __GEIGEN__::SVD(F, uu, vv, ss);
            double values = ss.m[0][0] + ss.m[1][1] + ss.m[2][2];
            values        = (values - 2) * (values - 2);
            double3 FxN   = __GEIGEN__::__M_v_multiply(F, normal);
            double  I5    = __GEIGEN__::__squaredNorm(FxN);

            __GEIGEN__::Matrix9x12d PFPx = __computePFDsPX3D_double(DmInv);
#endif

#if (RANK == 1)
            double lambda0 =
                Kappa
                * (2 * dHat * dHat
                   * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5 - 6 * I5 * I5 * log(I5) + 1))
                / I5;
            if(dis * dis < gassThreshold * dHat)
            {
                double lambda1 =
                    Kappa
                    * (2 * dHat * dHat
                       * (6 * gassThreshold + 2 * gassThreshold * log(gassThreshold)
                          - 7 * gassThreshold * gassThreshold
                          - 6 * gassThreshold * gassThreshold * log(gassThreshold) + 1))
                    / gassThreshold;
                lambda0 = lambda1;
            }
#elif (RANK == 2)
            double lambda0 =
                -(4 * Kappa * dHat * dHat
                  * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 6 * I5 * log(I5)
                     - 2 * I5 * I5 + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2))
                / I5;
            if(dis * dis < gassThreshold * dHat)
            {
                double lambda1 =
                    -(4 * Kappa * dHat * dHat
                      * (4 * gassThreshold + log(gassThreshold)
                         - 3 * gassThreshold * gassThreshold * log(gassThreshold) * log(gassThreshold)
                         + 6 * gassThreshold * log(gassThreshold) - 2 * gassThreshold * gassThreshold
                         + gassThreshold * log(gassThreshold) * log(gassThreshold)
                         - 7 * gassThreshold * gassThreshold * log(gassThreshold) - 2))
                    / gassThreshold;
                lambda0 = lambda1;
            }
#elif (RANK == 3)
            double lambda0 =
                (2 * Kappa * dHat * dHat * log(I5)
                 * (24 * I5 + 3 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5)
                    + 18 * I5 * log(I5) - 12 * I5 * I5
                    + 2 * I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 12))
                / I5;
#elif (RANK == 4)
            double lambda0 =
                -(4 * Kappa * dHat * dHat * log(I5) * log(I5)
                  * (24 * I5 + 2 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                     + 12 * I5 * log(I5) - 12 * I5 * I5 + I5 * log(I5) * log(I5)
                     - 14 * I5 * I5 * log(I5) - 12))
                / I5;
#elif (RANK == 5)
            double lambda0 =
                (2 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                 * (80 * I5 + 5 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5)
                    + 30 * I5 * log(I5) - 40 * I5 * I5
                    + 2 * I5 * log(I5) * log(I5) - 35 * I5 * I5 * log(I5) - 40))
                / I5;
#elif (RANK == 6)
            double lambda0 =
                -(4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5)
                  * (60 * I5 + 3 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                     + 18 * I5 * log(I5) - 30 * I5 * I5 + I5 * log(I5) * log(I5)
                     - 21 * I5 * I5 * log(I5) - 30))
                / I5;
#endif

#ifdef NEWF
            //printf("lamdba0:    %f\n", lambda0*1e6);
            //__GEIGEN__::__v9_vec9_toMat9x9(H,q0, q0, lambda0); //__GEIGEN__::__S_Mat9x9_multiply(__GEIGEN__::__v9_vec9_toMat9x9(q0, q0), lambda0);
            //__GEIGEN__::Matrix9x9d H;
            //__GEIGEN__::__init_Mat9x9(H, 0);
            //H.m[8][8] = lambda0;
            __GEIGEN__::Matrix9x9d H = __GEIGEN__::__S_Mat9x9_multiply(
                __GEIGEN__::__v9_vec9_toMat9x9(q0, q0), lambda0);  //__GEIGEN__::__v9_vec9_toMat9x9(q0, q0, lambda0);
            __GEIGEN__::Matrix12x12d Hessian;  // = __GEIGEN__::__M12x9_M9x12_Multiply(__GEIGEN__::__M12x9_M9x9_Multiply(PFPxT, H), __GEIGEN__::__Transpose12x9(PFPxT));
            __GEIGEN__::__M12x9_S9x9_MT9x12_Multiply(PFPxT, H, Hessian);

#else

            __GEIGEN__::Matrix3x3d Q0;

            __GEIGEN__::Matrix3x3d fnn;

            __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(normal, normal);

            __GEIGEN__::__M_Mat_multiply(F, nn, fnn);

            __GEIGEN__::Vector9 q0 = __GEIGEN__::__Mat3x3_to_vec9_double(fnn);

            q0 = __GEIGEN__::__s_vec9_multiply(q0, 1.0 / sqrt(I5));

            __GEIGEN__::Matrix9x9d H = __GEIGEN__::__S_Mat9x9_multiply(
                __GEIGEN__::__v9_vec9_toMat9x9(q0, q0), lambda0);

            __GEIGEN__::Matrix12x9d PFPxTransPos = __GEIGEN__::__Transpose9x12(PFPx);
            __GEIGEN__::Matrix12x12d H2 = __GEIGEN__::__M12x9_M9x12_Multiply(
                __GEIGEN__::__M12x9_M9x9_Multiply(PFPxTransPos, H), PFPx);
#endif

            int Hidx = matIndex[idx];  //atomicAdd(_cpNum + 4, 1);

            uint4 global_index = make_uint4(v0I, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
            //D4Index[Hidx]         = global_index;
            int triplet_id_offset = Hidx * 16;
            write_triplet<12, 12>(
                triplet_values, row_ids, col_ids, &(global_index.x), Hessian.m, triplet_id_offset);
        }
    }
}

// [multi-env determinism 4.3] BINNED (reproducible) FP gradient scatter — see GIPC.cuh.
// K exponent bins, width W, top anchor 2^E0. Each deposit splits x across bins; each bin's
// adds are EXACT (anchor-aligned) so atomic accumulation is order-independent ⇒ bit-identical.
// __dadd_rn/__dsub_rn block compiler reassociation (survives --use_fast_math). Verified with a
// standalone unit test (order-independent across permutations + atomic + vs reference).
#define BINNED_K 4
#define BINNED_W 30
#define BINNED_E0 60
__device__ double* g_gbin = nullptr;
// [multienv-mode] binned (Demmel-Nguyen order-free) gradient is a DETERMINISM feature (strict mode).
// merged/isolated don't need bit-identical gradients → fast plain-atomic path (bin 0 as a raw
// accumulator, full precision, non-deterministic order). g_binned_on=1 default (back-compat / strict).
__device__ int g_binned_on = 1;
static void set_binned_on(int v){ CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_binned_on, &v, sizeof(int))); }
// [xenv crack] target verts for the reliable (low-volume) deposit trace — set via env.
__device__ int g_bar_trace = 0;
__device__ int g_tgt0 = -1;
__device__ int g_tgt1 = -1;
static void set_bar_targets(int t, int a, int b){
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_bar_trace, &t, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_tgt0, &a, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_tgt1, &b, sizeof(int))); }
__device__ inline void _gfxAdd(int v, int comp, double val)
{
    if(g_bar_trace && (v == g_tgt0 || v == g_tgt1))
        printf("DEP %d %d %.17e\n", v, comp, val);
    double x    = val;
    size_t base = ((size_t)v * 3 + comp) * BINNED_K;
    if(!g_binned_on)
    {
        atomicAdd(&g_gbin[base], val);   // fast path: raw plain-atomic into bin 0 (correct, non-det order)
        return;
    }
#pragma unroll
    for(int k = 0; k < BINNED_K; ++k)
    {
        double M  = ldexp(1.5, BINNED_E0 - k * BINNED_W);   // constant args → folded
        double q  = __dadd_rn(M, x);
        double hi = __dsub_rn(q, M);
        atomicAdd(&g_gbin[base + k], hi);                   // same-grid exact ⇒ order-free
        x         = __dsub_rn(x, hi);
    }
}
// [4.3] base-pointer binned deposit (for the ABD FEM-pin coupling into g_abd_sysbin).
extern __device__ double* g_abd_sysbin;
__device__ inline void _binDepBase(double* bins, double val)
{
    double x = val;
#pragma unroll
    for(int k = 0; k < BINNED_K; ++k)
    { double M = ldexp(1.5, BINNED_E0 - k * BINNED_W); double q = __dadd_rn(M, x);
      double hi = __dsub_rn(q, M); atomicAdd(&bins[k], hi); x = __dsub_rn(x, hi); }
}
// combine the K bins per vertex back into the (double3) contact gradient (+= onto ground grad).
__global__ void _gfxToGrad(double3* _grad, const double* gbin, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    double gx = 0, gy = 0, gz = 0;
    // [fast-grad root fix] NO bin-0-only fast read: depositors are MIXED — _gfxAdd
    // goes raw into bin 0 in fast mode, but header binned_deposit users (femEnergy
    // elastic gradient: initKappa / getTotalForce / semi paths) ALWAYS 4-bin split.
    // Reading only bin 0 dropped their bins 1..K-1 (bin 0 alone is the value
    // rounded to ulp(2^60·1.5)=256!) -> garbage κ suggestion at contact onset ->
    // line-search death spiral on bbox-dHat scenes (case_26 family). Summing all
    // K bins is correct for BOTH deposit forms (fast leaves bins 1..K-1 = 0).
#pragma unroll
    for(int k = BINNED_K - 1; k >= 0; --k)   // finest bin first, fixed order
    {
        gx += gbin[((size_t)i * 3 + 0) * BINNED_K + k];
        gy += gbin[((size_t)i * 3 + 1) * BINNED_K + k];
        gz += gbin[((size_t)i * 3 + 2) * BINNED_K + k];
    }
    _grad[i].x += gx;
    _grad[i].y += gy;
    _grad[i].z += gz;
}
// [4.3] zero / combine the binned accumulator. ANY path that calls a barrier/friction
// gradient kernel must zero before and combine after (the kernels scatter to g_grad_binned,
// not to their _gradient arg). Used by computeGradientAndHessian, the kappa path, and the
// contact-force getters (get_*_contact_force_*).
void GIPC::zeroBinnedGrad()
{
    CUDA_SAFE_CALL(cudaMemset(g_grad_binned, 0,
                              3 * (size_t)vertexNum * BINNED_K * sizeof(double)));
}
void GIPC::combineBinnedGrad(double3* out)
{
    int bs = 256, gs = (vertexNum + bs - 1) / bs;
    _gfxToGrad<<<gs, bs>>>(out, g_grad_binned, vertexNum);
}

__global__ void _calBarrierGradientAndHessian(const double3*   _vertexes,
                                              const double3*   _rest_vertexes,
                                              const int4*      _collisionPair,
                                              double3*         _gradient,
                                              Eigen::Matrix3d* triplet_values,
                                              int*             row_ids,
                                              int*             col_ids,
                                              uint32_t*        _cpNum,
                                              int*             matIndex,
                                              double           dHat,
                                              double           Kappa_scalar,
                                              int              offset4,
                                              int              offset3,
                                              int              offset2,
                                              int              number,
                                              const double*    kappa_grp = nullptr,
                                              const int*       p2g       = nullptr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int4   MMCVIDI   = _collisionPair[idx];
    if(g_bar_trace) {
        int dx = MMCVIDI.x>=0?MMCVIDI.x:(-MMCVIDI.x-1);
        bool hit = (dx==g_tgt0||dx==g_tgt1)
                 || (MMCVIDI.y>=0 && (MMCVIDI.y==g_tgt0||MMCVIDI.y==g_tgt1))
                 || (MMCVIDI.z>=0 && (MMCVIDI.z==g_tgt0||MMCVIDI.z==g_tgt1))
                 || (MMCVIDI.w>=0 && (MMCVIDI.w==g_tgt0||MMCVIDI.w==g_tgt1));
        if(hit) printf("CT %d %d %d %d\n", MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
    }
    // [multi-env per-group κ] pair's env via its first (decoded) vertex; intra-env after P1.
    // nullptr → scalar (baseline, bit-identical). Computed BEFORE MMCVIDI is mutated below.
    double Kappa = Kappa_scalar;
    if(kappa_grp && p2g)
    { int _gv = (MMCVIDI.x >= 0) ? MMCVIDI.x : (-MMCVIDI.x - 1); int _gg = p2g[_gv]; if(_gg >= 0) Kappa = kappa_grp[_gg]; }  /* [-1 guard] wildcard -> scalar */
    double dHat_sqrt = sqrt(dHat);
    //double dHat = dHat_sqrt * dHat_sqrt;
    //double Kappa = 1;
    double gassThreshold = 1e-6;
    if(MMCVIDI.x >= 0)
    {
        if(MMCVIDI.w >= 0)
        {
#ifdef NEWF
            double dis;
            _d_EE(_vertexes[MMCVIDI.x],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            dis                                = sqrt(dis);
            double                  d_hat_sqrt = sqrt(dHat);
            __GEIGEN__::Matrix12x9d PFPxT;
            pFpx_ee2(_vertexes[MMCVIDI.x],
                     _vertexes[MMCVIDI.y],
                     _vertexes[MMCVIDI.z],
                     _vertexes[MMCVIDI.w],
                     d_hat_sqrt,
                     PFPxT);
            double              I5 = pow(dis / d_hat_sqrt, 2);
            __GEIGEN__::Vector9 tmp;
            tmp.v[0] = tmp.v[1] = tmp.v[2] = tmp.v[3] = tmp.v[4] = tmp.v[5] =
                tmp.v[6] = tmp.v[7] = 0;
            tmp.v[8]                = dis / d_hat_sqrt;

            __GEIGEN__::Vector9 q0;
            q0.v[0] = q0.v[1] = q0.v[2] = q0.v[3] = q0.v[4] = q0.v[5] =
                q0.v[6] = q0.v[7] = 0;
            q0.v[8]               = 1;
            //q0 = __GEIGEN__::__s_vec9_multiply(q0, 1.0 / sqrt(I5));

            __GEIGEN__::Matrix9x9d H;
            //__GEIGEN__::__init_Mat9x9(H, 0);
#else

            double3 v0 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.x]);
            double3 v1 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.x]);
            double3 v2 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.x]);
            __GEIGEN__::Matrix3x3d Ds;
            __GEIGEN__::__set_Mat_val_column(Ds, v0, v1, v2);
            double3 normal = __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(
                v0, __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.z])));
            double  dis    = __GEIGEN__::__v_vec_dot(v1, normal);
            if(dis < 0)
            {
                normal = make_double3(-normal.x, -normal.y, -normal.z);
                dis    = -dis;
            }

            double3 pos2 =
                __GEIGEN__::__add(_vertexes[MMCVIDI.z],
                                  __GEIGEN__::__s_vec_multiply(normal, dHat_sqrt - dis));
            double3 pos3 =
                __GEIGEN__::__add(_vertexes[MMCVIDI.w],
                                  __GEIGEN__::__s_vec_multiply(normal, dHat_sqrt - dis));

            double3 u0 = v0;
            double3 u1 = __GEIGEN__::__minus(pos2, _vertexes[MMCVIDI.x]);
            double3 u2 = __GEIGEN__::__minus(pos3, _vertexes[MMCVIDI.x]);

            __GEIGEN__::Matrix3x3d Dm, DmInv;
            __GEIGEN__::__set_Mat_val_column(Dm, u0, u1, u2);

            __GEIGEN__::__Inverse(Dm, DmInv);

            __GEIGEN__::Matrix3x3d F;
            __GEIGEN__::__M_Mat_multiply(Ds, DmInv, F);

            double3 FxN = __GEIGEN__::__M_v_multiply(F, normal);
            double  I5  = __GEIGEN__::__squaredNorm(FxN);

            __GEIGEN__::Matrix9x12d PFPx = __computePFDsPX3D_double(DmInv);

            __GEIGEN__::Matrix3x3d fnn;

            __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(normal, normal);

            __GEIGEN__::__M_Mat_multiply(F, nn, fnn);

            __GEIGEN__::Vector9 tmp = __GEIGEN__::__Mat3x3_to_vec9_double(fnn);

#endif

#if (RANK == 1)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
#elif (RANK == 2)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);
#elif (RANK == 3)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                -2
                    * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1)
                       * (3 * I5 + 2 * I5 * log(I5) - 3))
                    / I5);
#elif (RANK == 4)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                 * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                    / I5);
#elif (RANK == 5)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                -2
                    * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                       * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                    / I5);
#elif (RANK == 6)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5)
                 * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                    / I5);
#endif


#if (RANK == 1)
            double lambda0 =
                Kappa
                * (2 * dHat * dHat
                   * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5 - 6 * I5 * I5 * log(I5) + 1))
                / I5;
            if(dis * dis < gassThreshold * dHat)
            {
                double lambda1 =
                    Kappa
                    * (2 * dHat * dHat
                       * (6 * gassThreshold + 2 * gassThreshold * log(gassThreshold)
                          - 7 * gassThreshold * gassThreshold
                          - 6 * gassThreshold * gassThreshold * log(gassThreshold) + 1))
                    / gassThreshold;
                lambda0 = lambda1;
            }
#elif (RANK == 2)
            double lambda0 =
                -(4 * Kappa * dHat * dHat
                  * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 6 * I5 * log(I5)
                     - 2 * I5 * I5 + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2))
                / I5;
            if(dis * dis < gassThreshold * dHat)
            {
                double lambda1 =
                    -(4 * Kappa * dHat * dHat
                      * (4 * gassThreshold + log(gassThreshold)
                         - 3 * gassThreshold * gassThreshold * log(gassThreshold) * log(gassThreshold)
                         + 6 * gassThreshold * log(gassThreshold) - 2 * gassThreshold * gassThreshold
                         + gassThreshold * log(gassThreshold) * log(gassThreshold)
                         - 7 * gassThreshold * gassThreshold * log(gassThreshold) - 2))
                    / gassThreshold;
                lambda0 = lambda1;
            }
#elif (RANK == 3)
            double lambda0 =
                (2 * Kappa * dHat * dHat * log(I5)
                 * (24 * I5 + 3 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5)
                    + 18 * I5 * log(I5) - 12 * I5 * I5
                    + 2 * I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 12))
                / I5;
#elif (RANK == 4)
            double lambda0 =
                -(4 * Kappa * dHat * dHat * log(I5) * log(I5)
                  * (24 * I5 + 2 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                     + 12 * I5 * log(I5) - 12 * I5 * I5 + I5 * log(I5) * log(I5)
                     - 14 * I5 * I5 * log(I5) - 12))
                / I5;
#elif (RANK == 5)
            double lambda0 =
                (2 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                 * (80 * I5 + 5 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5)
                    + 30 * I5 * log(I5) - 40 * I5 * I5
                    + 2 * I5 * log(I5) * log(I5) - 35 * I5 * I5 * log(I5) - 40))
                / I5;
#elif (RANK == 6)
            double lambda0 =
                -(4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5)
                  * (60 * I5 + 3 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                     + 18 * I5 * log(I5) - 30 * I5 * I5 + I5 * log(I5) * log(I5)
                     - 21 * I5 * I5 * log(I5) - 30))
                / I5;
#endif


#ifdef NEWF
            __GEIGEN__::Vector12 gradient_vec =
                __GEIGEN__::__M12x9_v9_multiply((PFPxT), flatten_pk1);
            H = __GEIGEN__::__S_Mat9x9_multiply(__GEIGEN__::__v9_vec9_toMat9x9(q0, q0), lambda0);

            __GEIGEN__::Matrix12x12d Hessian;  // = __GEIGEN__::__M12x9_M9x12_Multiply(__GEIGEN__::__M12x9_M9x9_Multiply(PFPxT, H), __GEIGEN__::__Transpose12x9(PFPxT));
            __GEIGEN__::__M12x9_S9x9_MT9x12_Multiply(PFPxT, H, Hessian);
#else

            __GEIGEN__::Vector12 gradient_vec =
                __GEIGEN__::__M12x9_v9_multiply(__GEIGEN__::__Transpose9x12(PFPx), flatten_pk1);
            //__GEIGEN__::Matrix3x3d Q0;

            //            __GEIGEN__::Matrix3x3d fnn;

            //           __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(normal, normal);

            //            __GEIGEN__::__M_Mat_multiply(F, nn, fnn);

            __GEIGEN__::Vector9 q0 = __GEIGEN__::__Mat3x3_to_vec9_double(fnn);

            q0 = __GEIGEN__::__s_vec9_multiply(q0, 1.0 / sqrt(I5));

            __GEIGEN__::Matrix9x9d H;
            __GEIGEN__::__init_Mat9x9(H, 0);

            H = __GEIGEN__::__S_Mat9x9_multiply(__GEIGEN__::__v9_vec9_toMat9x9(q0, q0), lambda0);

            __GEIGEN__::Matrix12x9d PFPxTransPos = __GEIGEN__::__Transpose9x12(PFPx);
            __GEIGEN__::Matrix12x12d Hessian = __GEIGEN__::__M12x9_M9x12_Multiply(
                __GEIGEN__::__M12x9_M9x9_Multiply(PFPxTransPos, H), PFPx);
#endif

            {
                _gfxAdd(MMCVIDI.x, 0, gradient_vec.v[0]);
                _gfxAdd(MMCVIDI.x, 1, gradient_vec.v[1]);
                _gfxAdd(MMCVIDI.x, 2, gradient_vec.v[2]);
                _gfxAdd(MMCVIDI.y, 0, gradient_vec.v[3]);
                _gfxAdd(MMCVIDI.y, 1, gradient_vec.v[4]);
                _gfxAdd(MMCVIDI.y, 2, gradient_vec.v[5]);
                _gfxAdd(MMCVIDI.z, 0, gradient_vec.v[6]);
                _gfxAdd(MMCVIDI.z, 1, gradient_vec.v[7]);
                _gfxAdd(MMCVIDI.z, 2, gradient_vec.v[8]);
                _gfxAdd(MMCVIDI.w, 0, gradient_vec.v[9]);
                _gfxAdd(MMCVIDI.w, 1, gradient_vec.v[10]);
                _gfxAdd(MMCVIDI.w, 2, gradient_vec.v[11]);
            }
            int Hidx = matIndex[idx];  //atomicAdd(_cpNum + 4, 1);

            uint4 global_index =
                make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);

            int triplet_id_offset = Hidx * M12_Off;
            write_triplet<12, 12>(
                triplet_values, row_ids, col_ids, &(global_index.x), Hessian.m, triplet_id_offset);
        }
        else
        {
            //return;
            MMCVIDI.w = -MMCVIDI.w - 1;
            double3 v0 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.x]);
            double3 v1 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.z]);
            double c = __GEIGEN__::__norm(__GEIGEN__::__v_vec_cross(v0, v1)) /*/ __GEIGEN__::__norm(v0)*/;
            double I1 = c * c;
            if(I1 == 0)
                return;
            double dis;
            _d_EE(_vertexes[MMCVIDI.x],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            double I2 = dis / dHat;
            dis       = sqrt(dis);

            __GEIGEN__::Matrix3x3d F;
            __GEIGEN__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
            double3 n1 = make_double3(0, 1, 0);
            double3 n2 = make_double3(0, 0, 1);

            double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                        _rest_vertexes[MMCVIDI.y],
                                        _rest_vertexes[MMCVIDI.z],
                                        _rest_vertexes[MMCVIDI.w]);

            __GEIGEN__::Matrix3x3d g1, g2;

            __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(n1, n1);
            __GEIGEN__::__M_Mat_multiply(F, nn, g1);
            nn = __GEIGEN__::__v_vec_toMat(n2, n2);
            __GEIGEN__::__M_Mat_multiply(F, nn, g2);

            __GEIGEN__::Vector9 flatten_g1 = __GEIGEN__::__Mat3x3_to_vec9_double(g1);
            __GEIGEN__::Vector9 flatten_g2 = __GEIGEN__::__Mat3x3_to_vec9_double(g2);

            __GEIGEN__::Matrix12x9d PFPx;
            pFpx_pee(_vertexes[MMCVIDI.x],
                     _vertexes[MMCVIDI.y],
                     _vertexes[MMCVIDI.z],
                     _vertexes[MMCVIDI.w],
                     dHat_sqrt,
                     PFPx);

#if (RANK == 1)
            double p1 = Kappa * 2
                        * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                        / (eps_x * eps_x);
            double p2 = Kappa * 2
                        * (I1 * dHat * dHat * (I1 - 2 * eps_x) * (I2 - 1)
                           * (I2 + 2 * I2 * log(I2) - 1))
                        / (I2 * eps_x * eps_x);
#elif (RANK == 2)
            double p1 = -Kappa * 2
                        * (2 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x)
                           * (I2 - 1) * (I2 - 1))
                        / (eps_x * eps_x);
            double p2 = -Kappa * 2
                        * (2 * I1 * dHat * dHat * log(I2) * (I1 - 2 * eps_x)
                           * (I2 - 1) * (I2 + I2 * log(I2) - 1))
                        / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
            double p1 = -Kappa * 2
                        * (2 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x)
                           * (I2 - 1) * (I2 - 1))
                        / (eps_x * eps_x);
            double p2 = -Kappa * 2
                        * (2 * I1 * dHat * dHat * pow(log(I2), 3) * (I1 - 2 * eps_x)
                           * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2))
                        / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
            double p1 = -Kappa * 2
                        * (2 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x)
                           * (I2 - 1) * (I2 - 1))
                        / (eps_x * eps_x);
            double p2 = -Kappa * 2
                        * (2 * I1 * dHat * dHat * pow(log(I2), 5) * (I1 - 2 * eps_x)
                           * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3))
                        / (I2 * (eps_x * eps_x));
#endif
            __GEIGEN__::Vector9 flatten_pk1 =
                __GEIGEN__::__add9(__GEIGEN__::__s_vec9_multiply(flatten_g1, p1),
                                   __GEIGEN__::__s_vec9_multiply(flatten_g2, p2));
            __GEIGEN__::Vector12 gradient_vec =
                __GEIGEN__::__M12x9_v9_multiply(PFPx, flatten_pk1);

            {
                _gfxAdd(MMCVIDI.x, 0, gradient_vec.v[0]);
                _gfxAdd(MMCVIDI.x, 1, gradient_vec.v[1]);
                _gfxAdd(MMCVIDI.x, 2, gradient_vec.v[2]);
                _gfxAdd(MMCVIDI.y, 0, gradient_vec.v[3]);
                _gfxAdd(MMCVIDI.y, 1, gradient_vec.v[4]);
                _gfxAdd(MMCVIDI.y, 2, gradient_vec.v[5]);
                _gfxAdd(MMCVIDI.z, 0, gradient_vec.v[6]);
                _gfxAdd(MMCVIDI.z, 1, gradient_vec.v[7]);
                _gfxAdd(MMCVIDI.z, 2, gradient_vec.v[8]);
                _gfxAdd(MMCVIDI.w, 0, gradient_vec.v[9]);
                _gfxAdd(MMCVIDI.w, 1, gradient_vec.v[10]);
                _gfxAdd(MMCVIDI.w, 2, gradient_vec.v[11]);
            }

#if (RANK == 1)
            double lambda10 =
                Kappa * (4 * dHat * dHat * log(I2) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                / (eps_x * eps_x);
            double lambda11 =
                Kappa * 2
                * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                / (eps_x * eps_x);
            double lambda12 =
                Kappa * 2
                * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                / (eps_x * eps_x);
#elif (RANK == 2)
            double lambda10 = -Kappa
                              * (4 * dHat * dHat * log(I2) * log(I2) * (I2 - 1)
                                 * (I2 - 1) * (3 * I1 - eps_x))
                              / (eps_x * eps_x);
            double lambda11 = -Kappa
                              * (4 * dHat * dHat * log(I2) * log(I2)
                                 * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                              / (eps_x * eps_x);
            double lambda12 = -Kappa
                              * (4 * dHat * dHat * log(I2) * log(I2)
                                 * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                              / (eps_x * eps_x);
#elif (RANK == 4)
            double lambda10 = -Kappa
                              * (4 * dHat * dHat * pow(log(I2), 4) * (I2 - 1)
                                 * (I2 - 1) * (3 * I1 - eps_x))
                              / (eps_x * eps_x);
            double lambda11 = -Kappa
                              * (4 * dHat * dHat * pow(log(I2), 4)
                                 * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                              / (eps_x * eps_x);
            double lambda12 = -Kappa
                              * (4 * dHat * dHat * pow(log(I2), 4)
                                 * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                              / (eps_x * eps_x);
#elif (RANK == 6)
            double lambda10 = -Kappa
                              * (4 * dHat * dHat * pow(log(I2), 6) * (I2 - 1)
                                 * (I2 - 1) * (3 * I1 - eps_x))
                              / (eps_x * eps_x);
            double lambda11 = -Kappa
                              * (4 * dHat * dHat * pow(log(I2), 6)
                                 * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                              / (eps_x * eps_x);
            double lambda12 = -Kappa
                              * (4 * dHat * dHat * pow(log(I2), 6)
                                 * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                              / (eps_x * eps_x);
#endif
            __GEIGEN__::Matrix3x3d Tx, Ty, Tz;
            __GEIGEN__::__set_Mat_val(Tx, 0, 0, 0, 0, 0, 1, 0, -1, 0);
            __GEIGEN__::__set_Mat_val(Ty, 0, 0, -1, 0, 0, 0, 1, 0, 0);
            __GEIGEN__::__set_Mat_val(Tz, 0, 1, 0, -1, 0, 0, 0, 0, 0);

            __GEIGEN__::Vector9 q11 = __GEIGEN__::__Mat3x3_to_vec9_double(
                __GEIGEN__::__M_Mat_multiply(Tx, g1));
            __GEIGEN__::__normalized_vec9_double(q11);
            __GEIGEN__::Vector9 q12 = __GEIGEN__::__Mat3x3_to_vec9_double(
                __GEIGEN__::__M_Mat_multiply(Tz, g1));
            __GEIGEN__::__normalized_vec9_double(q12);

            __GEIGEN__::Matrix9x9d projectedH;
            __GEIGEN__::__init_Mat9x9(projectedH, 0);

            __GEIGEN__::Matrix9x9d M9_temp = __GEIGEN__::__v9_vec9_toMat9x9(q11, q11);
            M9_temp    = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, lambda11);
            projectedH = __GEIGEN__::__Mat9x9_add(projectedH, M9_temp);

            M9_temp    = __GEIGEN__::__v9_vec9_toMat9x9(q12, q12);
            M9_temp    = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, lambda12);
            projectedH = __GEIGEN__::__Mat9x9_add(projectedH, M9_temp);

#if (RANK == 1)
            double lambda20 =
                -Kappa
                * (2 * I1 * dHat * dHat * (I1 - 2 * eps_x)
                   * (6 * I2 + 2 * I2 * log(I2) - 7 * I2 * I2 - 6 * I2 * I2 * log(I2) + 1))
                / (I2 * eps_x * eps_x);
#elif (RANK == 2)
            double lambda20 =
                Kappa
                * (4 * I1 * dHat * dHat * (I1 - 2 * eps_x)
                   * (4 * I2 + log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                      + 6 * I2 * log(I2) - 2 * I2 * I2 + I2 * log(I2) * log(I2)
                      - 7 * I2 * I2 * log(I2) - 2))
                / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
            double lambda20 =
                Kappa
                * (4 * I1 * dHat * dHat * log(I2) * log(I2) * (I1 - 2 * eps_x)
                   * (24 * I2 + 2 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                      + 12 * I2 * log(I2) - 12 * I2 * I2
                      + I2 * log(I2) * log(I2) - 14 * I2 * I2 * log(I2) - 12))
                / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
            double lambda20 =
                Kappa
                * (4 * I1 * dHat * dHat * pow(log(I2), 4) * (I1 - 2 * eps_x)
                   * (60 * I2 + 3 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                      + 18 * I2 * log(I2) - 30 * I2 * I2
                      + I2 * log(I2) * log(I2) - 21 * I2 * I2 * log(I2) - 30))
                / (I2 * (eps_x * eps_x));
#endif

#if (RANK == 1)
            double lambdag1g = Kappa * 4 * c * F.m[2][2]
                               * ((2 * dHat * dHat * (I1 - eps_x) * (I2 - 1)
                                   * (I2 + 2 * I2 * log(I2) - 1))
                                  / (I2 * eps_x * eps_x));
#elif (RANK == 2)
            double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                               * (4 * dHat * dHat * log(I2) * (I1 - eps_x)
                                  * (I2 - 1) * (I2 + I2 * log(I2) - 1))
                               / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
            double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                               * (4 * dHat * dHat * pow(log(I2), 3) * (I1 - eps_x)
                                  * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2))
                               / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
            double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                               * (4 * dHat * dHat * pow(log(I2), 5) * (I1 - eps_x)
                                  * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3))
                               / (I2 * (eps_x * eps_x));
#endif
            Eigen::Matrix2d FMat2;
            FMat2 << lambda10, lambdag1g, lambdag1g, lambda20;
            makePDGeneral<double, 2>(FMat2);
            projectedH.m[4][4] += FMat2(0, 0);
            projectedH.m[4][8] += FMat2(0, 1);
            projectedH.m[8][4] += FMat2(1, 0);
            projectedH.m[8][8] += FMat2(1, 1);

            //__GEIGEN__::Matrix9x12d PFPxTransPos = __GEIGEN__::__Transpose12x9(PFPx);
            __GEIGEN__::Matrix12x12d Hessian;  // = __GEIGEN__::__M12x9_M9x12_Multiply(__GEIGEN__::__M12x9_M9x9_Multiply(PFPx, projectedH), PFPxTransPos);
            __GEIGEN__::__M12x9_S9x9_MT9x12_Multiply(PFPx, projectedH, Hessian);
            int Hidx = matIndex[idx];  //int Hidx = atomicAdd(_cpNum + 4, 1);

            uint4 global_index =
                make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);

            int triplet_id_offset = Hidx * M12_Off;
            write_triplet<12, 12>(
                triplet_values, row_ids, col_ids, &(global_index.x), Hessian.m, triplet_id_offset);
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
                    return;
                double dis;
                _d_PP(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], dis);
                double I2 = dis / dHat;
                dis       = sqrt(dis);

                __GEIGEN__::Matrix3x3d F;
                __GEIGEN__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
                double3 n1 = make_double3(0, 1, 0);
                double3 n2 = make_double3(0, 0, 1);

                double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                            _rest_vertexes[MMCVIDI.z],
                                            _rest_vertexes[MMCVIDI.y],
                                            _rest_vertexes[MMCVIDI.w]);

                __GEIGEN__::Matrix3x3d g1, g2;

                __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(n1, n1);
                __GEIGEN__::__M_Mat_multiply(F, nn, g1);
                nn = __GEIGEN__::__v_vec_toMat(n2, n2);
                __GEIGEN__::__M_Mat_multiply(F, nn, g2);

                __GEIGEN__::Vector9 flatten_g1 = __GEIGEN__::__Mat3x3_to_vec9_double(g1);
                __GEIGEN__::Vector9 flatten_g2 = __GEIGEN__::__Mat3x3_to_vec9_double(g2);

                __GEIGEN__::Matrix12x9d PFPx;
                pFpx_ppp(_vertexes[MMCVIDI.x],
                         _vertexes[MMCVIDI.y],
                         _vertexes[MMCVIDI.z],
                         _vertexes[MMCVIDI.w],
                         dHat_sqrt,
                         PFPx);

#if (RANK == 1)
                double p1 =
                    Kappa * 2
                    * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                    / (eps_x * eps_x);
                double p2 = Kappa * 2
                            * (I1 * dHat * dHat * (I1 - 2 * eps_x) * (I2 - 1)
                               * (I2 + 2 * I2 * log(I2) - 1))
                            / (I2 * eps_x * eps_x);
#elif (RANK == 2)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * log(I2) * log(I2)
                               * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * log(I2) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (I2 + I2 * log(I2) - 1))
                            / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x)
                               * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * pow(log(I2), 3) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2))
                            / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x)
                               * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * pow(log(I2), 5) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3))
                            / (I2 * (eps_x * eps_x));
#endif
                __GEIGEN__::Vector9 flatten_pk1 =
                    __GEIGEN__::__add9(__GEIGEN__::__s_vec9_multiply(flatten_g1, p1),
                                       __GEIGEN__::__s_vec9_multiply(flatten_g2, p2));
                __GEIGEN__::Vector12 gradient_vec =
                    __GEIGEN__::__M12x9_v9_multiply(PFPx, flatten_pk1);

                {
                    _gfxAdd(MMCVIDI.x, 0, gradient_vec.v[0]);
                    _gfxAdd(MMCVIDI.x, 1, gradient_vec.v[1]);
                    _gfxAdd(MMCVIDI.x, 2, gradient_vec.v[2]);
                    _gfxAdd(MMCVIDI.y, 0, gradient_vec.v[3]);
                    _gfxAdd(MMCVIDI.y, 1, gradient_vec.v[4]);
                    _gfxAdd(MMCVIDI.y, 2, gradient_vec.v[5]);
                    _gfxAdd(MMCVIDI.z, 0, gradient_vec.v[6]);
                    _gfxAdd(MMCVIDI.z, 1, gradient_vec.v[7]);
                    _gfxAdd(MMCVIDI.z, 2, gradient_vec.v[8]);
                    _gfxAdd(MMCVIDI.w, 0, gradient_vec.v[9]);
                    _gfxAdd(MMCVIDI.w, 1, gradient_vec.v[10]);
                    _gfxAdd(MMCVIDI.w, 2, gradient_vec.v[11]);
                }

#if (RANK == 1)
                double lambda10 =
                    Kappa * (4 * dHat * dHat * log(I2) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                    / (eps_x * eps_x);
                double lambda11 =
                    Kappa * 2
                    * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                    / (eps_x * eps_x);
                double lambda12 =
                    Kappa * 2
                    * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                    / (eps_x * eps_x);
#elif (RANK == 2)
                double lambda10 = -Kappa
                                  * (4 * dHat * dHat * log(I2) * log(I2)
                                     * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                                  / (eps_x * eps_x);
                double lambda11 = -Kappa
                                  * (4 * dHat * dHat * log(I2) * log(I2)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
                double lambda12 = -Kappa
                                  * (4 * dHat * dHat * log(I2) * log(I2)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
#elif (RANK == 4)
                double lambda10 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 4)
                                     * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                                  / (eps_x * eps_x);
                double lambda11 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 4)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
                double lambda12 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 4)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
#elif (RANK == 6)
                double lambda10 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 6)
                                     * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                                  / (eps_x * eps_x);
                double lambda11 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 6)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
                double lambda12 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 6)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
#endif
                __GEIGEN__::Matrix3x3d Tx, Ty, Tz;
                __GEIGEN__::__set_Mat_val(Tx, 0, 0, 0, 0, 0, 1, 0, -1, 0);
                __GEIGEN__::__set_Mat_val(Ty, 0, 0, -1, 0, 0, 0, 1, 0, 0);
                __GEIGEN__::__set_Mat_val(Tz, 0, 1, 0, -1, 0, 0, 0, 0, 0);

                __GEIGEN__::Vector9 q11 = __GEIGEN__::__Mat3x3_to_vec9_double(
                    __GEIGEN__::__M_Mat_multiply(Tx, g1));
                __GEIGEN__::__normalized_vec9_double(q11);
                __GEIGEN__::Vector9 q12 = __GEIGEN__::__Mat3x3_to_vec9_double(
                    __GEIGEN__::__M_Mat_multiply(Tz, g1));
                __GEIGEN__::__normalized_vec9_double(q12);

                __GEIGEN__::Matrix9x9d projectedH;
                __GEIGEN__::__init_Mat9x9(projectedH, 0);

                __GEIGEN__::Matrix9x9d M9_temp = __GEIGEN__::__v9_vec9_toMat9x9(q11, q11);
                M9_temp    = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, lambda11);
                projectedH = __GEIGEN__::__Mat9x9_add(projectedH, M9_temp);

                M9_temp    = __GEIGEN__::__v9_vec9_toMat9x9(q12, q12);
                M9_temp    = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, lambda12);
                projectedH = __GEIGEN__::__Mat9x9_add(projectedH, M9_temp);

#if (RANK == 1)
                double lambda20 = -Kappa
                                  * (2 * I1 * dHat * dHat * (I1 - 2 * eps_x)
                                     * (6 * I2 + 2 * I2 * log(I2) - 7 * I2 * I2
                                        - 6 * I2 * I2 * log(I2) + 1))
                                  / (I2 * eps_x * eps_x);
#elif (RANK == 2)
                double lambda20 =
                    Kappa
                    * (4 * I1 * dHat * dHat * (I1 - 2 * eps_x)
                       * (4 * I2 + log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                          + 6 * I2 * log(I2) - 2 * I2 * I2
                          + I2 * log(I2) * log(I2) - 7 * I2 * I2 * log(I2) - 2))
                    / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                double lambda20 =
                    Kappa
                    * (4 * I1 * dHat * dHat * log(I2) * log(I2) * (I1 - 2 * eps_x)
                       * (24 * I2 + 2 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                          + 12 * I2 * log(I2) - 12 * I2 * I2
                          + I2 * log(I2) * log(I2) - 14 * I2 * I2 * log(I2) - 12))
                    / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                double lambda20 =
                    Kappa
                    * (4 * I1 * dHat * dHat * pow(log(I2), 4) * (I1 - 2 * eps_x)
                       * (60 * I2 + 3 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                          + 18 * I2 * log(I2) - 30 * I2 * I2
                          + I2 * log(I2) * log(I2) - 21 * I2 * I2 * log(I2) - 30))
                    / (I2 * (eps_x * eps_x));
#endif

#if (RANK == 1)
                double lambdag1g = Kappa * 4 * c * F.m[2][2]
                                   * ((2 * dHat * dHat * (I1 - eps_x) * (I2 - 1)
                                       * (I2 + 2 * I2 * log(I2) - 1))
                                      / (I2 * eps_x * eps_x));
#elif (RANK == 2)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                                   * (4 * dHat * dHat * log(I2) * (I1 - eps_x)
                                      * (I2 - 1) * (I2 + I2 * log(I2) - 1))
                                   / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                                   * (4 * dHat * dHat * pow(log(I2), 3) * (I1 - eps_x)
                                      * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2))
                                   / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                                   * (4 * dHat * dHat * pow(log(I2), 5) * (I1 - eps_x)
                                      * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3))
                                   / (I2 * (eps_x * eps_x));
#endif
                Eigen::Matrix2d FMat2;
                FMat2 << lambda10, lambdag1g, lambdag1g, lambda20;
                makePDGeneral<double, 2>(FMat2);
                projectedH.m[4][4] += FMat2(0, 0);
                projectedH.m[4][8] += FMat2(0, 1);
                projectedH.m[8][4] += FMat2(1, 0);
                projectedH.m[8][8] += FMat2(1, 1);

                //__GEIGEN__::Matrix9x12d PFPxTransPos = __GEIGEN__::__Transpose12x9(PFPx);
                __GEIGEN__::Matrix12x12d Hessian;  // = __GEIGEN__::__M12x9_M9x12_Multiply(__GEIGEN__::__M12x9_M9x9_Multiply(PFPx, projectedH), PFPxTransPos);
                __GEIGEN__::__M12x9_S9x9_MT9x12_Multiply(PFPx, projectedH, Hessian);
                int Hidx = matIndex[idx];  //int Hidx = atomicAdd(_cpNum + 4, 1);

                uint4 global_index =
                    make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);

                int triplet_id_offset = Hidx * M12_Off;
                write_triplet<12, 12>(
                    triplet_values, row_ids, col_ids, &(global_index.x), Hessian.m, triplet_id_offset);
            }
            else
            {
#ifdef NEWF
                double dis;
                _d_PP(_vertexes[v0I], _vertexes[MMCVIDI.y], dis);
                dis                            = sqrt(dis);
                double              d_hat_sqrt = sqrt(dHat);
                __GEIGEN__::Vector6 PFPxT;
                pFpx_pp2(_vertexes[v0I], _vertexes[MMCVIDI.y], d_hat_sqrt, PFPxT);
                double I5  = pow(dis / d_hat_sqrt, 2);
                double fnn = dis / d_hat_sqrt;

#if (RANK == 1)
                double flatten_pk1 =
                    fnn * 2 * Kappa
                    * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5;
#elif (RANK == 2)
                double flatten_pk1 = fnn * 2
                                     * (2 * Kappa * dHat * dHat * log(I5)
                                        * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                                     / I5;
#elif (RANK == 3)
                double flatten_pk1 = fnn * -2
                                     * (Kappa * dHat * dHat * log(I5) * log(I5)
                                        * (I5 - 1) * (3 * I5 + 2 * I5 * log(I5) - 3))
                                     / I5;
#elif (RANK == 4)
                double flatten_pk1 =
                    fnn
                    * (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                       * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                    / I5;
#elif (RANK == 5)
                double flatten_pk1 =
                    fnn * -2
                    * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                       * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                    / I5;
#elif (RANK == 6)
                double flatten_pk1 =
                    fnn
                    * (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                       * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                    / I5;
#endif

                __GEIGEN__::Vector6 gradient_vec =
                    __GEIGEN__::__s_vec6_multiply(PFPxT, flatten_pk1);

#else
                double3 v0 = __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[v0I]);
                double3 Ds  = v0;
                double  dis = __GEIGEN__::__norm(v0);
                //if (dis > dHat_sqrt) return;
                double3 vec_normal =
                    __GEIGEN__::__normalized(make_double3(-v0.x, -v0.y, -v0.z));
                double3 target = make_double3(0, 1, 0);
                double3 vec    = __GEIGEN__::__v_vec_cross(vec_normal, target);
                double  cos    = __GEIGEN__::__v_vec_dot(vec_normal, target);
                __GEIGEN__::Matrix3x3d rotation;
                __GEIGEN__::__set_Mat_val(rotation, 1, 0, 0, 0, 1, 0, 0, 0, 1);
                __GEIGEN__::Vector6 PDmPx;
                if(cos + 1 == 0)
                {
                    rotation.m[0][0] = -1;
                    rotation.m[1][1] = -1;
                }
                else
                {
                    __GEIGEN__::Matrix3x3d cross_vec;
                    __GEIGEN__::__set_Mat_val(
                        cross_vec, 0, -vec.z, vec.y, vec.z, 0, -vec.x, -vec.y, vec.x, 0);

                    rotation = __GEIGEN__::__Mat_add(
                        rotation,
                        __GEIGEN__::__Mat_add(cross_vec,
                                              __GEIGEN__::__S_Mat_multiply(
                                                  __GEIGEN__::__M_Mat_multiply(cross_vec, cross_vec),
                                                  1.0 / (1 + cos))));
                }

                double3 pos0 = __GEIGEN__::__add(
                    _vertexes[v0I],
                    __GEIGEN__::__s_vec_multiply(vec_normal, dHat_sqrt - dis));
                double3 rotate_uv0 = __GEIGEN__::__M_v_multiply(rotation, pos0);
                double3 rotate_uv1 =
                    __GEIGEN__::__M_v_multiply(rotation, _vertexes[MMCVIDI.y]);

                double uv0 = rotate_uv0.y;
                double uv1 = rotate_uv1.y;

                double u0    = uv1 - uv0;
                double Dm    = u0;  //PFPx
                double DmInv = 1 / u0;

                double3 F  = __GEIGEN__::__s_vec_multiply(Ds, DmInv);
                double  I5 = __GEIGEN__::__squaredNorm(F);

                double3 tmp = F;

#if (RANK == 1)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
#elif (RANK == 2)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp,
                    2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                        / I5);

#elif (RANK == 3)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1)
                           * (3 * I5 + 2 * I5 * log(I5) - 3))
                        / I5);
#elif (RANK == 4)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                        / I5);
#elif (RANK == 5)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                           * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                        / I5);
#elif (RANK == 6)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                        / I5);
#endif
                __GEIGEN__::Matrix3x6d PFPx = __computePFDsPX3D_3x6_double(DmInv);

                __GEIGEN__::Vector6 gradient_vec =
                    __GEIGEN__::__M6x3_v3_multiply(__GEIGEN__::__Transpose3x6(PFPx), flatten_pk1);
#endif


                {
                    _gfxAdd(v0I, 0, gradient_vec.v[0]);
                    _gfxAdd(v0I, 1, gradient_vec.v[1]);
                    _gfxAdd(v0I, 2, gradient_vec.v[2]);
                    _gfxAdd(MMCVIDI.y, 0, gradient_vec.v[3]);
                    _gfxAdd(MMCVIDI.y, 1, gradient_vec.v[4]);
                    _gfxAdd(MMCVIDI.y, 2, gradient_vec.v[5]);
                }

#if (RANK == 1)
                double lambda0 = Kappa
                                 * (2 * dHat * dHat
                                    * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5
                                       - 6 * I5 * I5 * log(I5) + 1))
                                 / I5;
                if(dis * dis < gassThreshold * dHat)
                {
                    double lambda1 =
                        Kappa
                        * (2 * dHat * dHat
                           * (6 * gassThreshold + 2 * gassThreshold * log(gassThreshold)
                              - 7 * gassThreshold * gassThreshold
                              - 6 * gassThreshold * gassThreshold * log(gassThreshold) + 1))
                        / gassThreshold;
                    lambda0 = lambda1;
                }
#elif (RANK == 2)
                double lambda0 =
                    -(4 * Kappa * dHat * dHat
                      * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                         + 6 * I5 * log(I5) - 2 * I5 * I5
                         + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2))
                    / I5;
                if(dis * dis < gassThreshold * dHat)
                {
                    double lambda1 =
                        -(4 * Kappa * dHat * dHat
                          * (4 * gassThreshold + log(gassThreshold)
                             - 3 * gassThreshold * gassThreshold
                                   * log(gassThreshold) * log(gassThreshold)
                             + 6 * gassThreshold * log(gassThreshold) - 2 * gassThreshold * gassThreshold
                             + gassThreshold * log(gassThreshold) * log(gassThreshold)
                             - 7 * gassThreshold * gassThreshold * log(gassThreshold) - 2))
                        / gassThreshold;
                    lambda0 = lambda1;
                }
#elif (RANK == 3)
                double lambda0 =
                    (2 * Kappa * dHat * dHat * log(I5)
                     * (24 * I5 + 3 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5)
                        + 18 * I5 * log(I5) - 12 * I5 * I5
                        + 2 * I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 12))
                    / I5;
#elif (RANK == 4)
                double lambda0 =
                    -(4 * Kappa * dHat * dHat * log(I5) * log(I5)
                      * (24 * I5 + 2 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                         + 12 * I5 * log(I5) - 12 * I5 * I5
                         + I5 * log(I5) * log(I5) - 14 * I5 * I5 * log(I5) - 12))
                    / I5;
#elif (RANK == 5)
                double lambda0 =
                    (2 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * (80 * I5 + 5 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5)
                        + 30 * I5 * log(I5) - 40 * I5 * I5
                        + 2 * I5 * log(I5) * log(I5) - 35 * I5 * I5 * log(I5) - 40))
                    / I5;
#elif (RANK == 6)
                double lambda0 =
                    -(4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5)
                      * (60 * I5 + 3 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                         + 18 * I5 * log(I5) - 30 * I5 * I5
                         + I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 30))
                    / I5;
#endif


#ifdef NEWF
                double                 H       = lambda0;
                __GEIGEN__::Matrix6x6d Hessian = __GEIGEN__::__s_M6x6_Multiply(
                    __GEIGEN__::__v6_vec6_toMat6x6(PFPxT, PFPxT), H);
#else
                double3 q0 = __GEIGEN__::__s_vec_multiply(F, 1 / sqrt(I5));

                __GEIGEN__::Matrix3x3d H =
                    __GEIGEN__::__S_Mat_multiply(__GEIGEN__::__v_vec_toMat(q0, q0),
                                                 lambda0);  //lambda0 * q0 * q0.transpose();

                __GEIGEN__::Matrix6x3d PFPxTransPos = __GEIGEN__::__Transpose3x6(PFPx);
                __GEIGEN__::Matrix6x6d Hessian = __GEIGEN__::__M6x3_M3x6_Multiply(
                    __GEIGEN__::__M6x3_M3x3_Multiply(PFPxTransPos, H), PFPx);
#endif
                int Hidx = matIndex[idx];  //int Hidx = atomicAdd(_cpNum + 2, 1);

                //H6x6[Hidx]    = Hessian;
                uint2 global_index = make_uint2(v0I, MMCVIDI.y);
                //D2Index[Hidx]      = global_index;

                int triplet_id_offset = Hidx * M6_Off + offset3 * M9_Off + offset4 * M12_Off;
                write_triplet<6, 6>(
                    triplet_values, row_ids, col_ids, &(global_index.x), Hessian.m, triplet_id_offset);
            }
        }
        else if(MMCVIDI.w < 0)
        {
            if(MMCVIDI.y < 0)
            {
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.x = v0I;
                MMCVIDI.w = -MMCVIDI.w - 1;
                double3 v0 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.x]);
                double3 v1 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.y]);
                double c = __GEIGEN__::__norm(__GEIGEN__::__v_vec_cross(v0, v1)) /*/ __GEIGEN__::__norm(v0)*/;
                double I1 = c * c;
                if(I1 == 0)
                    return;
                double dis;
                _d_PE(_vertexes[MMCVIDI.x],
                      _vertexes[MMCVIDI.y],
                      _vertexes[MMCVIDI.z],
                      dis);
                double I2 = dis / dHat;
                dis       = sqrt(dis);

                __GEIGEN__::Matrix3x3d F;
                __GEIGEN__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
                double3 n1 = make_double3(0, 1, 0);
                double3 n2 = make_double3(0, 0, 1);

                double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                            _rest_vertexes[MMCVIDI.w],
                                            _rest_vertexes[MMCVIDI.y],
                                            _rest_vertexes[MMCVIDI.z]);

                __GEIGEN__::Matrix3x3d g1, g2;

                __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(n1, n1);
                __GEIGEN__::__M_Mat_multiply(F, nn, g1);
                nn = __GEIGEN__::__v_vec_toMat(n2, n2);
                __GEIGEN__::__M_Mat_multiply(F, nn, g2);

                __GEIGEN__::Vector9 flatten_g1 = __GEIGEN__::__Mat3x3_to_vec9_double(g1);
                __GEIGEN__::Vector9 flatten_g2 = __GEIGEN__::__Mat3x3_to_vec9_double(g2);

                __GEIGEN__::Matrix12x9d PFPx;
                pFpx_ppe(_vertexes[MMCVIDI.x],
                         _vertexes[MMCVIDI.y],
                         _vertexes[MMCVIDI.z],
                         _vertexes[MMCVIDI.w],
                         dHat_sqrt,
                         PFPx);

#if (RANK == 1)
                double p1 =
                    Kappa * 2
                    * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                    / (eps_x * eps_x);
                double p2 = Kappa * 2
                            * (I1 * dHat * dHat * (I1 - 2 * eps_x) * (I2 - 1)
                               * (I2 + 2 * I2 * log(I2) - 1))
                            / (I2 * eps_x * eps_x);
#elif (RANK == 2)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * log(I2) * log(I2)
                               * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * log(I2) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (I2 + I2 * log(I2) - 1))
                            / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x)
                               * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * pow(log(I2), 3) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2))
                            / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x)
                               * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * pow(log(I2), 5) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3))
                            / (I2 * (eps_x * eps_x));
#endif
                __GEIGEN__::Vector9 flatten_pk1 =
                    __GEIGEN__::__add9(__GEIGEN__::__s_vec9_multiply(flatten_g1, p1),
                                       __GEIGEN__::__s_vec9_multiply(flatten_g2, p2));
                __GEIGEN__::Vector12 gradient_vec =
                    __GEIGEN__::__M12x9_v9_multiply(PFPx, flatten_pk1);

                {
                    _gfxAdd(MMCVIDI.x, 0, gradient_vec.v[0]);
                    _gfxAdd(MMCVIDI.x, 1, gradient_vec.v[1]);
                    _gfxAdd(MMCVIDI.x, 2, gradient_vec.v[2]);
                    _gfxAdd(MMCVIDI.y, 0, gradient_vec.v[3]);
                    _gfxAdd(MMCVIDI.y, 1, gradient_vec.v[4]);
                    _gfxAdd(MMCVIDI.y, 2, gradient_vec.v[5]);
                    _gfxAdd(MMCVIDI.z, 0, gradient_vec.v[6]);
                    _gfxAdd(MMCVIDI.z, 1, gradient_vec.v[7]);
                    _gfxAdd(MMCVIDI.z, 2, gradient_vec.v[8]);
                    _gfxAdd(MMCVIDI.w, 0, gradient_vec.v[9]);
                    _gfxAdd(MMCVIDI.w, 1, gradient_vec.v[10]);
                    _gfxAdd(MMCVIDI.w, 2, gradient_vec.v[11]);
                }

#if (RANK == 1)
                double lambda10 =
                    Kappa * (4 * dHat * dHat * log(I2) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                    / (eps_x * eps_x);
                double lambda11 =
                    Kappa * 2
                    * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                    / (eps_x * eps_x);
                double lambda12 =
                    Kappa * 2
                    * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                    / (eps_x * eps_x);
#elif (RANK == 2)
                double lambda10 = -Kappa
                                  * (4 * dHat * dHat * log(I2) * log(I2)
                                     * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                                  / (eps_x * eps_x);
                double lambda11 = -Kappa
                                  * (4 * dHat * dHat * log(I2) * log(I2)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
                double lambda12 = -Kappa
                                  * (4 * dHat * dHat * log(I2) * log(I2)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
#elif (RANK == 4)
                double lambda10 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 4)
                                     * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                                  / (eps_x * eps_x);
                double lambda11 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 4)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
                double lambda12 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 4)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
#elif (RANK == 6)
                double lambda10 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 6)
                                     * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                                  / (eps_x * eps_x);
                double lambda11 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 6)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
                double lambda12 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 6)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
#endif
                __GEIGEN__::Matrix3x3d Tx, Ty, Tz;
                __GEIGEN__::__set_Mat_val(Tx, 0, 0, 0, 0, 0, 1, 0, -1, 0);
                __GEIGEN__::__set_Mat_val(Ty, 0, 0, -1, 0, 0, 0, 1, 0, 0);
                __GEIGEN__::__set_Mat_val(Tz, 0, 1, 0, -1, 0, 0, 0, 0, 0);

                __GEIGEN__::Vector9 q11 = __GEIGEN__::__Mat3x3_to_vec9_double(
                    __GEIGEN__::__M_Mat_multiply(Tx, g1));
                __GEIGEN__::__normalized_vec9_double(q11);
                __GEIGEN__::Vector9 q12 = __GEIGEN__::__Mat3x3_to_vec9_double(
                    __GEIGEN__::__M_Mat_multiply(Tz, g1));
                __GEIGEN__::__normalized_vec9_double(q12);

                __GEIGEN__::Matrix9x9d projectedH;
                __GEIGEN__::__init_Mat9x9(projectedH, 0);

                __GEIGEN__::Matrix9x9d M9_temp = __GEIGEN__::__v9_vec9_toMat9x9(q11, q11);
                M9_temp    = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, lambda11);
                projectedH = __GEIGEN__::__Mat9x9_add(projectedH, M9_temp);

                M9_temp    = __GEIGEN__::__v9_vec9_toMat9x9(q12, q12);
                M9_temp    = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, lambda12);
                projectedH = __GEIGEN__::__Mat9x9_add(projectedH, M9_temp);

#if (RANK == 1)
                double lambda20 = -Kappa
                                  * (2 * I1 * dHat * dHat * (I1 - 2 * eps_x)
                                     * (6 * I2 + 2 * I2 * log(I2) - 7 * I2 * I2
                                        - 6 * I2 * I2 * log(I2) + 1))
                                  / (I2 * eps_x * eps_x);
#elif (RANK == 2)
                double lambda20 =
                    Kappa
                    * (4 * I1 * dHat * dHat * (I1 - 2 * eps_x)
                       * (4 * I2 + log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                          + 6 * I2 * log(I2) - 2 * I2 * I2
                          + I2 * log(I2) * log(I2) - 7 * I2 * I2 * log(I2) - 2))
                    / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                double lambda20 =
                    Kappa
                    * (4 * I1 * dHat * dHat * log(I2) * log(I2) * (I1 - 2 * eps_x)
                       * (24 * I2 + 2 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                          + 12 * I2 * log(I2) - 12 * I2 * I2
                          + I2 * log(I2) * log(I2) - 14 * I2 * I2 * log(I2) - 12))
                    / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                double lambda20 =
                    Kappa
                    * (4 * I1 * dHat * dHat * pow(log(I2), 4) * (I1 - 2 * eps_x)
                       * (60 * I2 + 3 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                          + 18 * I2 * log(I2) - 30 * I2 * I2
                          + I2 * log(I2) * log(I2) - 21 * I2 * I2 * log(I2) - 30))
                    / (I2 * (eps_x * eps_x));
#endif

#if (RANK == 1)
                double lambdag1g = Kappa * 4 * c * F.m[2][2]
                                   * ((2 * dHat * dHat * (I1 - eps_x) * (I2 - 1)
                                       * (I2 + 2 * I2 * log(I2) - 1))
                                      / (I2 * eps_x * eps_x));
#elif (RANK == 2)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                                   * (4 * dHat * dHat * log(I2) * (I1 - eps_x)
                                      * (I2 - 1) * (I2 + I2 * log(I2) - 1))
                                   / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                                   * (4 * dHat * dHat * pow(log(I2), 3) * (I1 - eps_x)
                                      * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2))
                                   / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                                   * (4 * dHat * dHat * pow(log(I2), 5) * (I1 - eps_x)
                                      * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3))
                                   / (I2 * (eps_x * eps_x));
#endif
                Eigen::Matrix2d FMat2;
                FMat2 << lambda10, lambdag1g, lambdag1g, lambda20;
                makePDGeneral<double, 2>(FMat2);
                projectedH.m[4][4] += FMat2(0, 0);
                projectedH.m[4][8] += FMat2(0, 1);
                projectedH.m[8][4] += FMat2(1, 0);
                projectedH.m[8][8] += FMat2(1, 1);

                //__GEIGEN__::Matrix9x12d PFPxTransPos = __GEIGEN__::__Transpose12x9(PFPx);
                __GEIGEN__::Matrix12x12d Hessian;  // = __GEIGEN__::__M12x9_M9x12_Multiply(__GEIGEN__::__M12x9_M9x9_Multiply(PFPx, projectedH), PFPxTransPos);
                __GEIGEN__::__M12x9_S9x9_MT9x12_Multiply(PFPx, projectedH, Hessian);
                int Hidx = matIndex[idx];  //int Hidx = atomicAdd(_cpNum + 4, 1);

                uint4 global_index =
                    make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);

                int triplet_id_offset = Hidx * M12_Off;
                write_triplet<12, 12>(
                    triplet_values, row_ids, col_ids, &(global_index.x), Hessian.m, triplet_id_offset);
            }
            else
            {
#ifdef NEWF
                double dis;
                _d_PE(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dis);
                dis                               = sqrt(dis);
                double                 d_hat_sqrt = sqrt(dHat);
                __GEIGEN__::Matrix9x4d PFPxT;
                pFpx_pe2(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], d_hat_sqrt, PFPxT);
                double              I5 = pow(dis / d_hat_sqrt, 2);
                __GEIGEN__::Vector4 fnn;
                fnn.v[0] = fnn.v[1] = fnn.v[2] = 0;  // = fnn.v[3] = fnn.v[4] = 1;
                fnn.v[3] = dis / d_hat_sqrt;
                //__GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(fnn, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
                __GEIGEN__::Vector4 q0;
                q0.v[0] = q0.v[1] = q0.v[2] = 0;
                q0.v[3]                     = 1;
                __GEIGEN__::Matrix4x4d H;
                //__GEIGEN__::__init_Mat4x4_val(H, 0);
#if (RANK == 1)
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
#elif (RANK == 2)
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn,
                    2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                        / I5);
#elif (RANK == 3)
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1)
                           * (3 * I5 + 2 * I5 * log(I5) - 3))
                        / I5);
#elif (RANK == 4)
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                        / I5);
#elif (RANK == 5)
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                           * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                        / I5);
#elif (RANK == 6)
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                        / I5);
#endif

                __GEIGEN__::Vector9 gradient_vec =
                    __GEIGEN__::__M9x4_v4_multiply(PFPxT, flatten_pk1);
#else

                double3 v0 = __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[v0I]);
                double3 v1 = __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[v0I]);


                __GEIGEN__::Matrix3x2d Ds;
                __GEIGEN__::__set_Mat3x2_val_column(Ds, v0, v1);

                double3 triangle_normal =
                    __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(v0, v1));
                double3 target = make_double3(0, 1, 0);

                double3 vec = __GEIGEN__::__v_vec_cross(triangle_normal, target);
                double cos = __GEIGEN__::__v_vec_dot(triangle_normal, target);

                double3 edge_normal = __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z]),
                    triangle_normal));
                double dis = __GEIGEN__::__v_vec_dot(
                    __GEIGEN__::__minus(_vertexes[v0I], _vertexes[MMCVIDI.y]), edge_normal);

                __GEIGEN__::Matrix3x3d rotation;
                __GEIGEN__::__set_Mat_val(rotation, 1, 0, 0, 0, 1, 0, 0, 0, 1);

                __GEIGEN__::Matrix9x4d PDmPx;

                if(cos + 1 == 0)
                {
                    rotation.m[0][0] = -1;
                    rotation.m[1][1] = -1;
                }
                else
                {
                    __GEIGEN__::Matrix3x3d cross_vec;
                    __GEIGEN__::__set_Mat_val(
                        cross_vec, 0, -vec.z, vec.y, vec.z, 0, -vec.x, -vec.y, vec.x, 0);

                    rotation = __GEIGEN__::__Mat_add(
                        rotation,
                        __GEIGEN__::__Mat_add(cross_vec,
                                              __GEIGEN__::__S_Mat_multiply(
                                                  __GEIGEN__::__M_Mat_multiply(cross_vec, cross_vec),
                                                  1.0 / (1 + cos))));
                }

                double3 pos0 = __GEIGEN__::__add(
                    _vertexes[v0I],
                    __GEIGEN__::__s_vec_multiply(edge_normal, dHat_sqrt - dis));

                double3 rotate_uv0 = __GEIGEN__::__M_v_multiply(rotation, pos0);
                double3 rotate_uv1 =
                    __GEIGEN__::__M_v_multiply(rotation, _vertexes[MMCVIDI.y]);
                double3 rotate_uv2 =
                    __GEIGEN__::__M_v_multiply(rotation, _vertexes[MMCVIDI.z]);
                double3 rotate_normal = __GEIGEN__::__M_v_multiply(rotation, edge_normal);

                double2 uv0    = make_double2(rotate_uv0.x, rotate_uv0.z);
                double2 uv1    = make_double2(rotate_uv1.x, rotate_uv1.z);
                double2 uv2    = make_double2(rotate_uv2.x, rotate_uv2.z);
                double2 normal = make_double2(rotate_normal.x, rotate_normal.z);

                double2 u0 = __GEIGEN__::__minus_v2(uv1, uv0);
                double2 u1 = __GEIGEN__::__minus_v2(uv2, uv0);

                __GEIGEN__::Matrix2x2d Dm;

                __GEIGEN__::__set_Mat2x2_val_column(Dm, u0, u1);

                __GEIGEN__::Matrix2x2d DmInv;
                __GEIGEN__::__Inverse2x2(Dm, DmInv);

                __GEIGEN__::Matrix3x2d F = __GEIGEN__::__M3x2_M2x2_Multiply(Ds, DmInv);

                double3 FxN = __GEIGEN__::__M3x2_v2_multiply(F, normal);
                double  I5  = __GEIGEN__::__squaredNorm(FxN);

                __GEIGEN__::Matrix3x2d fnn;

                __GEIGEN__::Matrix2x2d nn = __GEIGEN__::__v2_vec2_toMat2x2(normal, normal);

                fnn = __GEIGEN__::__M3x2_M2x2_Multiply(F, nn);

                __GEIGEN__::Vector6 tmp = __GEIGEN__::__Mat3x2_to_vec6_double(fnn);

#if (RANK == 1)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
#elif (RANK == 2)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp,
                    2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                        / I5);
#elif (RANK == 3)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1)
                           * (3 * I5 + 2 * I5 * log(I5) - 3))
                        / I5);
#elif (RANK == 4)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                        / I5);
#elif (RANK == 5)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                           * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                        / I5);
#elif (RANK == 6)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                        / I5);
#endif

                __GEIGEN__::Matrix6x9d PFPx = __computePFDsPX3D_6x9_double(DmInv);

                __GEIGEN__::Vector9 gradient_vec =
                    __GEIGEN__::__M9x6_v6_multiply(__GEIGEN__::__Transpose6x9(PFPx), flatten_pk1);
#endif

                {
                    _gfxAdd(v0I, 0, gradient_vec.v[0]);
                    _gfxAdd(v0I, 1, gradient_vec.v[1]);
                    _gfxAdd(v0I, 2, gradient_vec.v[2]);
                    _gfxAdd(MMCVIDI.y, 0, gradient_vec.v[3]);
                    _gfxAdd(MMCVIDI.y, 1, gradient_vec.v[4]);
                    _gfxAdd(MMCVIDI.y, 2, gradient_vec.v[5]);
                    _gfxAdd(MMCVIDI.z, 0, gradient_vec.v[6]);
                    _gfxAdd(MMCVIDI.z, 1, gradient_vec.v[7]);
                    _gfxAdd(MMCVIDI.z, 2, gradient_vec.v[8]);
                }

#if (RANK == 1)
                double lambda0 = Kappa
                                 * (2 * dHat * dHat
                                    * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5
                                       - 6 * I5 * I5 * log(I5) + 1))
                                 / I5;
                if(dis * dis < gassThreshold * dHat)
                {
                    double lambda1 =
                        Kappa
                        * (2 * dHat * dHat
                           * (6 * gassThreshold + 2 * gassThreshold * log(gassThreshold)
                              - 7 * gassThreshold * gassThreshold
                              - 6 * gassThreshold * gassThreshold * log(gassThreshold) + 1))
                        / gassThreshold;
                    lambda0 = lambda1;
                }
#elif (RANK == 2)
                double lambda0 =
                    -(4 * Kappa * dHat * dHat
                      * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                         + 6 * I5 * log(I5) - 2 * I5 * I5
                         + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2))
                    / I5;
                if(dis * dis < gassThreshold * dHat)
                {
                    double lambda1 =
                        -(4 * Kappa * dHat * dHat
                          * (4 * gassThreshold + log(gassThreshold)
                             - 3 * gassThreshold * gassThreshold
                                   * log(gassThreshold) * log(gassThreshold)
                             + 6 * gassThreshold * log(gassThreshold) - 2 * gassThreshold * gassThreshold
                             + gassThreshold * log(gassThreshold) * log(gassThreshold)
                             - 7 * gassThreshold * gassThreshold * log(gassThreshold) - 2))
                        / gassThreshold;
                    lambda0 = lambda1;
                }
#elif (RANK == 3)
                double lambda0 =
                    (2 * Kappa * dHat * dHat * log(I5)
                     * (24 * I5 + 3 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5)
                        + 18 * I5 * log(I5) - 12 * I5 * I5
                        + 2 * I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 12))
                    / I5;
#elif (RANK == 4)
                double lambda0 =
                    -(4 * Kappa * dHat * dHat * log(I5) * log(I5)
                      * (24 * I5 + 2 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                         + 12 * I5 * log(I5) - 12 * I5 * I5
                         + I5 * log(I5) * log(I5) - 14 * I5 * I5 * log(I5) - 12))
                    / I5;
#elif (RANK == 5)
                double lambda0 =
                    (2 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * (80 * I5 + 5 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5)
                        + 30 * I5 * log(I5) - 40 * I5 * I5
                        + 2 * I5 * log(I5) * log(I5) - 35 * I5 * I5 * log(I5) - 40))
                    / I5;
#elif (RANK == 6)
                double lambda0 =
                    -(4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5)
                      * (60 * I5 + 3 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                         + 18 * I5 * log(I5) - 30 * I5 * I5
                         + I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 30))
                    / I5;
#endif


#ifdef NEWF
                H = __GEIGEN__::__S_Mat4x4_multiply(
                    __GEIGEN__::__v4_vec4_toMat4x4(q0, q0), lambda0);

                __GEIGEN__::Matrix9x9d Hessian;  // = __GEIGEN__::__M9x4_M4x9_Multiply(__GEIGEN__::__M9x4_M4x4_Multiply(PFPxT, H), __GEIGEN__::__Transpose9x4(PFPxT));
                __GEIGEN__::__M9x4_S4x4_MT4x9_Multiply(PFPxT, H, Hessian);
#else

                __GEIGEN__::Vector6 q0 = __GEIGEN__::__Mat3x2_to_vec6_double(fnn);

                q0 = __GEIGEN__::__s_vec6_multiply(q0, 1.0 / sqrt(I5));

                __GEIGEN__::Matrix6x6d H;
                __GEIGEN__::__init_Mat6x6(H, 0);

                H = __GEIGEN__::__S_Mat6x6_multiply(
                    __GEIGEN__::__v6_vec6_toMat6x6(q0, q0), lambda0);

                __GEIGEN__::Matrix9x6d PFPxTransPos = __GEIGEN__::__Transpose6x9(PFPx);
                __GEIGEN__::Matrix9x9d Hessian = __GEIGEN__::__M9x6_M6x9_Multiply(
                    __GEIGEN__::__M9x6_M6x6_Multiply(PFPxTransPos, H), PFPx);
#endif
                int Hidx = matIndex[idx];  //int Hidx = atomicAdd(_cpNum + 3, 1);

                //H9x9[Hidx]    = Hessian;

                uint3 global_index = make_uint3(v0I, MMCVIDI.y, MMCVIDI.z);

                //D3Index[Hidx] = global_index;

                int triplet_id_offset = Hidx * M9_Off + offset4 * M12_Off;
                write_triplet<9, 9>(
                    triplet_values, row_ids, col_ids, &(global_index.x), Hessian.m, triplet_id_offset);
            }
        }
        else
        {
#ifdef NEWF
            double dis;
            _d_PT(_vertexes[v0I],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            dis                                = sqrt(dis);
            double                  d_hat_sqrt = sqrt(dHat);
            __GEIGEN__::Matrix12x9d PFPxT;
            pFpx_pt2(_vertexes[v0I],
                     _vertexes[MMCVIDI.y],
                     _vertexes[MMCVIDI.z],
                     _vertexes[MMCVIDI.w],
                     d_hat_sqrt,
                     PFPxT);
            double              I5 = pow(dis / d_hat_sqrt, 2);
            __GEIGEN__::Vector9 tmp;
            tmp.v[0] = tmp.v[1] = tmp.v[2] = tmp.v[3] = tmp.v[4] = tmp.v[5] =
                tmp.v[6] = tmp.v[7] = 0;
            tmp.v[8]                = dis / d_hat_sqrt;

            __GEIGEN__::Vector9 q0;
            q0.v[0] = q0.v[1] = q0.v[2] = q0.v[3] = q0.v[4] = q0.v[5] =
                q0.v[6] = q0.v[7] = 0;
            q0.v[8]               = 1;

            __GEIGEN__::Matrix9x9d H;
            //__GEIGEN__::__init_Mat9x9(H, 0);
#else
            double3 v0 = __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[v0I]);
            double3 v1 = __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[v0I]);
            double3 v2 = __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[v0I]);

            __GEIGEN__::Matrix3x3d Ds;
            __GEIGEN__::__set_Mat_val_column(Ds, v0, v1, v2);

            double3 normal = __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.y]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.y])));
            double  dis    = __GEIGEN__::__v_vec_dot(v0, normal);
            //if (abs(dis) > dHat_sqrt) return;
            __GEIGEN__::Matrix12x9d PDmPx;
            //bool is_flip = false;

            if(dis > 0)
            {
                //is_flip = true;
                normal = make_double3(-normal.x, -normal.y, -normal.z);
                //pDmpx_pt_flip(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dHat_sqrt, PDmPx);
                //printf("dHat_sqrt = %f,   dis = %f\n", dHat_sqrt, dis);
            }
            else
            {
                dis = -dis;
                //pDmpx_pt(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dHat_sqrt, PDmPx);
                //printf("dHat_sqrt = %f,   dis = %f\n", dHat_sqrt, dis);
            }

            double3 pos0 = __GEIGEN__::__add(
                _vertexes[v0I], __GEIGEN__::__s_vec_multiply(normal, dHat_sqrt - dis));


            double3 u0 = __GEIGEN__::__minus(_vertexes[MMCVIDI.y], pos0);
            double3 u1 = __GEIGEN__::__minus(_vertexes[MMCVIDI.z], pos0);
            double3 u2 = __GEIGEN__::__minus(_vertexes[MMCVIDI.w], pos0);

            __GEIGEN__::Matrix3x3d Dm, DmInv;
            __GEIGEN__::__set_Mat_val_column(Dm, u0, u1, u2);

            __GEIGEN__::__Inverse(Dm, DmInv);

            __GEIGEN__::Matrix3x3d F;  //, Ftest;
            __GEIGEN__::__M_Mat_multiply(Ds, DmInv, F);
            //__GEIGEN__::__M_Mat_multiply(Dm, DmInv, Ftest);

            double3 FxN = __GEIGEN__::__M_v_multiply(F, normal);
            double  I5  = __GEIGEN__::__squaredNorm(FxN);

            //printf("I5 = %f,   dist/dHat_sqrt = %f\n", I5, (dis / dHat_sqrt)* (dis / dHat_sqrt));


            __GEIGEN__::Matrix9x12d PFPx = __computePFDsPX3D_double(DmInv);

            __GEIGEN__::Matrix3x3d fnn;

            __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(normal, normal);

            __GEIGEN__::__M_Mat_multiply(F, nn, fnn);

            __GEIGEN__::Vector9 tmp = __GEIGEN__::__Mat3x3_to_vec9_double(fnn);
#endif
#if (RANK == 1)
            double lambda0 =
                Kappa
                * (2 * dHat * dHat
                   * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5 - 6 * I5 * I5 * log(I5) + 1))
                / I5;
            if(dis * dis < gassThreshold * dHat)
            {
                double lambda1 =
                    Kappa
                    * (2 * dHat * dHat
                       * (6 * gassThreshold + 2 * gassThreshold * log(gassThreshold)
                          - 7 * gassThreshold * gassThreshold
                          - 6 * gassThreshold * gassThreshold * log(gassThreshold) + 1))
                    / gassThreshold;
                lambda0 = lambda1;
            }
#elif (RANK == 2)
            double lambda0 =
                -(4 * Kappa * dHat * dHat
                  * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 6 * I5 * log(I5)
                     - 2 * I5 * I5 + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2))
                / I5;
            if(dis * dis < gassThreshold * dHat)
            {
                double lambda1 =
                    -(4 * Kappa * dHat * dHat
                      * (4 * gassThreshold + log(gassThreshold)
                         - 3 * gassThreshold * gassThreshold * log(gassThreshold) * log(gassThreshold)
                         + 6 * gassThreshold * log(gassThreshold) - 2 * gassThreshold * gassThreshold
                         + gassThreshold * log(gassThreshold) * log(gassThreshold)
                         - 7 * gassThreshold * gassThreshold * log(gassThreshold) - 2))
                    / gassThreshold;
                lambda0 = lambda1;
            }
#elif (RANK == 3)
            double lambda0 =
                (2 * Kappa * dHat * dHat * log(I5)
                 * (24 * I5 + 3 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5)
                    + 18 * I5 * log(I5) - 12 * I5 * I5
                    + 2 * I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 12))
                / I5;
#elif (RANK == 4)
            double lambda0 =
                -(4 * Kappa * dHat * dHat * log(I5) * log(I5)
                  * (24 * I5 + 2 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                     + 12 * I5 * log(I5) - 12 * I5 * I5 + I5 * log(I5) * log(I5)
                     - 14 * I5 * I5 * log(I5) - 12))
                / I5;
#elif (RANK == 5)
            double lambda0 =
                (2 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                 * (80 * I5 + 5 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5)
                    + 30 * I5 * log(I5) - 40 * I5 * I5
                    + 2 * I5 * log(I5) * log(I5) - 35 * I5 * I5 * log(I5) - 40))
                / I5;
#elif (RANK == 6)
            double lambda0 =
                -(4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5)
                  * (60 * I5 + 3 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                     + 18 * I5 * log(I5) - 30 * I5 * I5 + I5 * log(I5) * log(I5)
                     - 21 * I5 * I5 * log(I5) - 30))
                / I5;
#endif

#if (RANK == 1)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
#elif (RANK == 2)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);
#elif (RANK == 3)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                -2
                    * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1)
                       * (3 * I5 + 2 * I5 * log(I5) - 3))
                    / I5);
#elif (RANK == 4)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                 * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                    / I5);
#elif (RANK == 5)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                -2
                    * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                       * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                    / I5);
#elif (RANK == 6)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5)
                 * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                    / I5);
#endif

#ifdef NEWF
            __GEIGEN__::Vector12 gradient_vec =
                __GEIGEN__::__M12x9_v9_multiply(PFPxT, flatten_pk1);
#else
            __GEIGEN__::Vector12 gradient_vec =
                __GEIGEN__::__M12x9_v9_multiply(__GEIGEN__::__Transpose9x12(PFPx), flatten_pk1);
#endif

            _gfxAdd(v0I, 0, gradient_vec.v[0]);
            _gfxAdd(v0I, 1, gradient_vec.v[1]);
            _gfxAdd(v0I, 2, gradient_vec.v[2]);
            _gfxAdd(MMCVIDI.y, 0, gradient_vec.v[3]);
            _gfxAdd(MMCVIDI.y, 1, gradient_vec.v[4]);
            _gfxAdd(MMCVIDI.y, 2, gradient_vec.v[5]);
            _gfxAdd(MMCVIDI.z, 0, gradient_vec.v[6]);
            _gfxAdd(MMCVIDI.z, 1, gradient_vec.v[7]);
            _gfxAdd(MMCVIDI.z, 2, gradient_vec.v[8]);
            _gfxAdd(MMCVIDI.w, 0, gradient_vec.v[9]);
            _gfxAdd(MMCVIDI.w, 1, gradient_vec.v[10]);
            _gfxAdd(MMCVIDI.w, 2, gradient_vec.v[11]);

#ifdef NEWF

            H = __GEIGEN__::__S_Mat9x9_multiply(__GEIGEN__::__v9_vec9_toMat9x9(q0, q0), lambda0);

            __GEIGEN__::Matrix12x12d Hessian;  // = __GEIGEN__::__M12x9_M9x12_Multiply(__GEIGEN__::__M12x9_M9x9_Multiply(PFPxT, H), __GEIGEN__::__Transpose12x9(PFPxT));
            __GEIGEN__::__M12x9_S9x9_MT9x12_Multiply(PFPxT, H, Hessian);
#else

            //__GEIGEN__::Matrix3x3d Q0;

            //__GEIGEN__::Matrix3x3d fnn;

            //__GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(normal, normal);

            //__GEIGEN__::__M_Mat_multiply(F, nn, fnn);

            __GEIGEN__::Vector9 q0 = __GEIGEN__::__Mat3x3_to_vec9_double(fnn);

            q0 = __GEIGEN__::__s_vec9_multiply(q0, 1.0 / sqrt(I5));

            __GEIGEN__::Matrix9x9d H = __GEIGEN__::__S_Mat9x9_multiply(
                __GEIGEN__::__v9_vec9_toMat9x9(q0, q0), lambda0);

            __GEIGEN__::Matrix12x9d PFPxTransPos = __GEIGEN__::__Transpose9x12(PFPx);
            __GEIGEN__::Matrix12x12d Hessian = __GEIGEN__::__M12x9_M9x12_Multiply(
                __GEIGEN__::__M12x9_M9x9_Multiply(PFPxTransPos, H), PFPx);
#endif

            int Hidx = matIndex[idx];  //int Hidx = atomicAdd(_cpNum + 4, 1);

            //H12x12[Hidx]  = Hessian;
            uint4 global_index = make_uint4(v0I, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
            //D4Index[Hidx]         = global_index;
            int triplet_id_offset = Hidx * M12_Off;
            write_triplet<12, 12>(
                triplet_values, row_ids, col_ids, &(global_index.x), Hessian.m, triplet_id_offset);
        }
    }
}


__global__ void _calSelfCloseVal(const double3* _vertexes,
                                 const int4*    _collisionPair,
                                 int4*          _close_collisionPair,
                                 double*        _close_collisionVal,
                                 uint32_t*      _close_cpNum,
                                 double         dTol,
                                 int            number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int4   MMCVIDI = _collisionPair[idx];
    double dist2   = _selfConstraintVal(_vertexes, MMCVIDI);
    if(dist2 < dTol)
    {
        int tidx                   = atomicAdd(_close_cpNum, 1);
        _close_collisionPair[tidx] = MMCVIDI;
        _close_collisionVal[tidx]  = dist2;
    }
}

__global__ void _checkSelfCloseVal(const double3* _vertexes,
                                   int*           _isChange,
                                   int4*          _close_collisionPair,
                                   double*        _close_collisionVal,
                                   int            number,
                                   int*           _isChange_grp = nullptr,
                                   const int*     p2g           = nullptr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int4   MMCVIDI = _close_collisionPair[idx];
    double dist2   = _selfConstraintVal(_vertexes, MMCVIDI);
    if(dist2 < _close_collisionVal[idx])
    {
        *_isChange = 1;
        // [multi-env per-group κ] flag only THIS pair's env (intra-env after P1).
        if(_isChange_grp && p2g)
        { int _gv = (MMCVIDI.x >= 0) ? MMCVIDI.x : (-MMCVIDI.x - 1); int _gg = p2g[_gv]; if(_gg >= 0) _isChange_grp[_gg] = 1; }  /* [-1 guard] */
    }
}


__global__ void _reduct_MSelfDist(const double3* _vertexes,
                                  int4*          _collisionPairs,
                                  double2*       _queue,
                                  int            number)
{
    int                       idof = blockIdx.x * blockDim.x;
    int                       idx  = threadIdx.x + idof;
    extern __shared__ double2 sdata[];

    if(idx >= number)
        return;
    int4    MMCVIDI = _collisionPairs[idx];
    double  tempv   = _selfConstraintVal(_vertexes, MMCVIDI);
    double2 temp    = make_double2(1.0 / tempv, tempv);
    int     warpTid = threadIdx.x % 32;
    int     warpId  = (threadIdx.x >> 5);
    double  nextTp;
    int     warpNum;
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
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = sdata[threadIdx.x];

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
        _queue[blockIdx.x] = temp;
    }
}

__global__ void _calFrictionGradient_gd(const double3* _vertexes,
                                        const double3* _o_vertexes,
                                        const double3* _normal,
                                        const const uint32_t* _last_collisionPair_gd,
                                        double3* _gradient,
                                        int      number,
                                        double   dt,
                                        double   eps2,
                                        double*  lastH,
                                        double   coef)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    double   eps    = sqrt(eps2);
    double3  normal = *_normal;
    uint32_t gidx   = _last_collisionPair_gd[idx];
    double3  Vdiff  = __GEIGEN__::__minus(_vertexes[gidx], _o_vertexes[gidx]);
    double3  VProj  = __GEIGEN__::__minus(
        Vdiff, __GEIGEN__::__s_vec_multiply(normal, __GEIGEN__::__v_vec_dot(Vdiff, normal)));
    double VProjMag2 = __GEIGEN__::__squaredNorm(VProj);
    if(VProjMag2 > eps2)
    {
        double3 gdf =
            __GEIGEN__::__s_vec_multiply(VProj, coef * lastH[idx] / sqrt(VProjMag2));
        /*_gfxAdd(gidx, 0, gdf.x);
        _gfxAdd(gidx, 1, gdf.y);
        _gfxAdd(gidx, 2, gdf.z);*/
        _gradient[gidx] = __GEIGEN__::__add(_gradient[gidx], gdf);
    }
    else
    {
        double3 gdf = __GEIGEN__::__s_vec_multiply(VProj, coef * lastH[idx] / eps);
        /*_gfxAdd(gidx, 0, gdf.x);
        _gfxAdd(gidx, 1, gdf.y);
        _gfxAdd(gidx, 2, gdf.z);*/
        _gradient[gidx] = __GEIGEN__::__add(_gradient[gidx], gdf);
    }
}

__global__ void _calFrictionGradient(const double3*    _vertexes,
                                     const double3*    _o_vertexes,
                                     const const int4* _last_collisionPair,
                                     double3*          _gradient,
                                     int               number,
                                     double            dt,
                                     double2*          distCoord,
                                     __GEIGEN__::Matrix3x2d* tanBasis,
                                     double                  eps2,
                                     double*                 lastH,
                                     double                  coef)
{
    double eps = std::sqrt(eps2);
    int    idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int4    MMCVIDI = _last_collisionPair[idx];
    double3 relDX3D;
    if(MMCVIDI.x >= 0)
    {
        Friction::computeRelDX_EE(
            __GEIGEN__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
            __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
            __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
            __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]),
            distCoord[idx].x,
            distCoord[idx].y,
            relDX3D);

        __GEIGEN__::Matrix2x3d tB_T = __GEIGEN__::__Transpose3x2(tanBasis[idx]);
        double2 relDX       = __GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D);
        double  relDXSqNorm = __GEIGEN__::__squaredNorm(relDX);
        if(relDXSqNorm > eps2)
        {
            relDX = __GEIGEN__::__s_vec_multiply(relDX, 1.0 / sqrt(relDXSqNorm));
        }
        else
        {
            double f1_div_relDXNorm;
            Friction::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
            relDX = __GEIGEN__::__s_vec_multiply(relDX, f1_div_relDXNorm);
        }
        __GEIGEN__::Vector12 TTTDX;
        Friction::liftRelDXTanToMesh_EE(
            relDX, tanBasis[idx], distCoord[idx].x, distCoord[idx].y, TTTDX);
        TTTDX = __GEIGEN__::__s_vec12_multiply(TTTDX, lastH[idx] * coef);
        {
            _gfxAdd(MMCVIDI.x, 0, TTTDX.v[0]);
            _gfxAdd(MMCVIDI.x, 1, TTTDX.v[1]);
            _gfxAdd(MMCVIDI.x, 2, TTTDX.v[2]);
            _gfxAdd(MMCVIDI.y, 0, TTTDX.v[3]);
            _gfxAdd(MMCVIDI.y, 1, TTTDX.v[4]);
            _gfxAdd(MMCVIDI.y, 2, TTTDX.v[5]);
            _gfxAdd(MMCVIDI.z, 0, TTTDX.v[6]);
            _gfxAdd(MMCVIDI.z, 1, TTTDX.v[7]);
            _gfxAdd(MMCVIDI.z, 2, TTTDX.v[8]);
            _gfxAdd(MMCVIDI.w, 0, TTTDX.v[9]);
            _gfxAdd(MMCVIDI.w, 1, TTTDX.v[10]);
            _gfxAdd(MMCVIDI.w, 2, TTTDX.v[11]);
        }
    }
    else
    {
        int v0I = -MMCVIDI.x - 1;
        if(MMCVIDI.z < 0)
        {
            MMCVIDI.x = v0I;

            Friction::computeRelDX_PP(
                __GEIGEN__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                relDX3D);

            __GEIGEN__::Matrix2x3d tB_T = __GEIGEN__::__Transpose3x2(tanBasis[idx]);
            double2 relDX       = __GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D);
            double  relDXSqNorm = __GEIGEN__::__squaredNorm(relDX);
            if(relDXSqNorm > eps2)
            {
                relDX = __GEIGEN__::__s_vec_multiply(relDX, 1.0 / sqrt(relDXSqNorm));
            }
            else
            {
                double f1_div_relDXNorm;
                Friction::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                relDX = __GEIGEN__::__s_vec_multiply(relDX, f1_div_relDXNorm);
            }

            __GEIGEN__::Vector6 TTTDX;
            Friction::liftRelDXTanToMesh_PP(relDX, tanBasis[idx], TTTDX);
            TTTDX = __GEIGEN__::__s_vec6_multiply(TTTDX, lastH[idx] * coef);
            {
                _gfxAdd(MMCVIDI.x, 0, TTTDX.v[0]);
                _gfxAdd(MMCVIDI.x, 1, TTTDX.v[1]);
                _gfxAdd(MMCVIDI.x, 2, TTTDX.v[2]);
                _gfxAdd(MMCVIDI.y, 0, TTTDX.v[3]);
                _gfxAdd(MMCVIDI.y, 1, TTTDX.v[4]);
                _gfxAdd(MMCVIDI.y, 2, TTTDX.v[5]);
            }
        }
        else if(MMCVIDI.w < 0)
        {
            MMCVIDI.x = v0I;
            Friction::computeRelDX_PE(
                __GEIGEN__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
                distCoord[idx].x,
                relDX3D);

            __GEIGEN__::Matrix2x3d tB_T = __GEIGEN__::__Transpose3x2(tanBasis[idx]);
            double2 relDX       = __GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D);
            double  relDXSqNorm = __GEIGEN__::__squaredNorm(relDX);
            if(relDXSqNorm > eps2)
            {
                relDX = __GEIGEN__::__s_vec_multiply(relDX, 1.0 / sqrt(relDXSqNorm));
            }
            else
            {
                double f1_div_relDXNorm;
                Friction::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                relDX = __GEIGEN__::__s_vec_multiply(relDX, f1_div_relDXNorm);
            }
            __GEIGEN__::Vector9 TTTDX;
            Friction::liftRelDXTanToMesh_PE(relDX, tanBasis[idx], distCoord[idx].x, TTTDX);
            TTTDX = __GEIGEN__::__s_vec9_multiply(TTTDX, lastH[idx] * coef);
            {
                _gfxAdd(MMCVIDI.x, 0, TTTDX.v[0]);
                _gfxAdd(MMCVIDI.x, 1, TTTDX.v[1]);
                _gfxAdd(MMCVIDI.x, 2, TTTDX.v[2]);
                _gfxAdd(MMCVIDI.y, 0, TTTDX.v[3]);
                _gfxAdd(MMCVIDI.y, 1, TTTDX.v[4]);
                _gfxAdd(MMCVIDI.y, 2, TTTDX.v[5]);
                _gfxAdd(MMCVIDI.z, 0, TTTDX.v[6]);
                _gfxAdd(MMCVIDI.z, 1, TTTDX.v[7]);
                _gfxAdd(MMCVIDI.z, 2, TTTDX.v[8]);
            }
        }
        else
        {
            MMCVIDI.x = v0I;
            Friction::computeRelDX_PT(
                __GEIGEN__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]),
                distCoord[idx].x,
                distCoord[idx].y,
                relDX3D);

            __GEIGEN__::Matrix2x3d tB_T = __GEIGEN__::__Transpose3x2(tanBasis[idx]);
            double2 relDX = __GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D);

            double relDXSqNorm = __GEIGEN__::__squaredNorm(relDX);
            if(relDXSqNorm > eps2)
            {
                relDX = __GEIGEN__::__s_vec_multiply(relDX, 1.0 / sqrt(relDXSqNorm));
            }
            else
            {
                double f1_div_relDXNorm;
                Friction::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                relDX = __GEIGEN__::__s_vec_multiply(relDX, f1_div_relDXNorm);
            }
            __GEIGEN__::Vector12 TTTDX;
            Friction::liftRelDXTanToMesh_PT(
                relDX, tanBasis[idx], distCoord[idx].x, distCoord[idx].y, TTTDX);
            TTTDX = __GEIGEN__::__s_vec12_multiply(TTTDX, lastH[idx] * coef);

            _gfxAdd(MMCVIDI.x, 0, TTTDX.v[0]);
            _gfxAdd(MMCVIDI.x, 1, TTTDX.v[1]);
            _gfxAdd(MMCVIDI.x, 2, TTTDX.v[2]);
            _gfxAdd(MMCVIDI.y, 0, TTTDX.v[3]);
            _gfxAdd(MMCVIDI.y, 1, TTTDX.v[4]);
            _gfxAdd(MMCVIDI.y, 2, TTTDX.v[5]);
            _gfxAdd(MMCVIDI.z, 0, TTTDX.v[6]);
            _gfxAdd(MMCVIDI.z, 1, TTTDX.v[7]);
            _gfxAdd(MMCVIDI.z, 2, TTTDX.v[8]);
            _gfxAdd(MMCVIDI.w, 0, TTTDX.v[9]);
            _gfxAdd(MMCVIDI.w, 1, TTTDX.v[10]);
            _gfxAdd(MMCVIDI.w, 2, TTTDX.v[11]);
        }
    }
}


// [Step B] optional per-contact export hook. When _ec_out_pair != nullptr, each
// contact also writes (bodyA,bodyB) and the physical contact force on bodyA
// (N) = -(sum of this contact's gradient over bodyA's verts)/dt^2. bodyA = body
// of the first vertex; same-body (self) contacts get bodyB==bodyA (consumer
// skips). nullptr -> no-op (solve path unchanged).
__device__ inline void _ec_emit(int idx, int2* op, double3* of, const int* pbid,
                                double inv_dt2, int va, int vb, int vc, int vd,
                                int nv, const double* g)
{
    if(!op)
        return;
    int    vs[4] = {va, vb, vc, vd};
    int    bA = pbid[va], bB = pbid[va];
    double fx = 0, fy = 0, fz = 0;
    for(int k = 0; k < nv; k++)
    {
        int b = pbid[vs[k]];
        if(b == bA)
        {
            fx += g[3 * k]; fy += g[3 * k + 1]; fz += g[3 * k + 2];
        }
        else
            bB = b;
    }
    op[idx] = make_int2(bA, bB);
    of[idx] = make_double3(-fx * inv_dt2, -fy * inv_dt2, -fz * inv_dt2);
}

__global__ void _calBarrierGradient(const double3*    _vertexes,
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
                                    double            _ec_inv_dt2   = 0.0)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int4   MMCVIDI   = _collisionPair[idx];
    // [multi-env per-group κ] nullptr → scalar (baseline). Before MMCVIDI mutation.
    double Kappa = Kappa_scalar;
    if(kappa_grp && p2g)
    { int _gv = (MMCVIDI.x >= 0) ? MMCVIDI.x : (-MMCVIDI.x - 1); int _gg = p2g[_gv]; if(_gg >= 0) Kappa = kappa_grp[_gg]; }  /* [-1 guard] wildcard -> scalar */
    double dHat_sqrt = sqrt(dHat);
    //double dHat = dHat_sqrt * dHat_sqrt;
    //double Kappa = 1;
    if(MMCVIDI.x >= 0)
    {
        if(MMCVIDI.w >= 0)
        {
#ifdef NEWF
            double dis;
            _d_EE(_vertexes[MMCVIDI.x],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            dis                                = sqrt(dis);
            double                  d_hat_sqrt = sqrt(dHat);
            __GEIGEN__::Matrix12x9d PFPxT;
            pFpx_ee2(_vertexes[MMCVIDI.x],
                     _vertexes[MMCVIDI.y],
                     _vertexes[MMCVIDI.z],
                     _vertexes[MMCVIDI.w],
                     d_hat_sqrt,
                     PFPxT);
            double              I5 = pow(dis / d_hat_sqrt, 2);
            __GEIGEN__::Vector9 tmp;
            tmp.v[0] = tmp.v[1] = tmp.v[2] = tmp.v[3] = tmp.v[4] = tmp.v[5] =
                tmp.v[6] = tmp.v[7] = 0;
            tmp.v[8]                = dis / d_hat_sqrt;
#else

            double3 v0 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.x]);
            double3 v1 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.x]);
            double3 v2 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.x]);
            __GEIGEN__::Matrix3x3d Ds;
            __GEIGEN__::__set_Mat_val_column(Ds, v0, v1, v2);
            double3 normal = __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(
                v0, __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.z])));
            double  dis    = __GEIGEN__::__v_vec_dot(v1, normal);
            if(dis < 0)
            {
                normal = make_double3(-normal.x, -normal.y, -normal.z);
                dis    = -dis;
            }

            double3 pos2 =
                __GEIGEN__::__add(_vertexes[MMCVIDI.z],
                                  __GEIGEN__::__s_vec_multiply(normal, dHat_sqrt - dis));
            double3 pos3 =
                __GEIGEN__::__add(_vertexes[MMCVIDI.w],
                                  __GEIGEN__::__s_vec_multiply(normal, dHat_sqrt - dis));

            double3 u0 = v0;
            double3 u1 = __GEIGEN__::__minus(pos2, _vertexes[MMCVIDI.x]);
            double3 u2 = __GEIGEN__::__minus(pos3, _vertexes[MMCVIDI.x]);

            __GEIGEN__::Matrix3x3d Dm, DmInv;
            __GEIGEN__::__set_Mat_val_column(Dm, u0, u1, u2);

            __GEIGEN__::__Inverse(Dm, DmInv);

            __GEIGEN__::Matrix3x3d F;
            __GEIGEN__::__M_Mat_multiply(Ds, DmInv, F);

            double3 FxN = __GEIGEN__::__M_v_multiply(F, normal);
            double  I5  = __GEIGEN__::__squaredNorm(FxN);

            __GEIGEN__::Matrix9x12d PFPx = __computePFDsPX3D_double(DmInv);

            __GEIGEN__::Matrix3x3d fnn;

            __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(normal, normal);

            __GEIGEN__::__M_Mat_multiply(F, nn, fnn);

            __GEIGEN__::Vector9 tmp = __GEIGEN__::__Mat3x3_to_vec9_double(fnn);

#endif

#if (RANK == 1)
            double judge =
                (2 * dHat * dHat
                 * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5 - 6 * I5 * I5 * log(I5) + 1))
                / I5;
            double judge2 = 2 * (dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1))
                            / I5 * dis / d_hat_sqrt;
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
            //if (dis*dis<1e-2*dHat)
            //flatten_pk1 = __GEIGEN__::__s_vec9_multiply(tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5 / (I5) /*/ (I5) / (I5)*/);
#elif (RANK == 2)
            //__GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(tmp, 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);

            double judge = -(4 * dHat * dHat
                             * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                                + 6 * I5 * log(I5) - 2 * I5 * I5
                                + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2))
                           / I5;
            double judge2 =
                2 * (2 * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                / I5 * dis / dHat_sqrt;
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);
            //if (dis*dis<1e-2*dHat)
            //flatten_pk1 = __GEIGEN__::__s_vec9_multiply(tmp, 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5/I5);

#elif (RANK == 3)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                -2
                    * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1)
                       * (3 * I5 + 2 * I5 * log(I5) - 3))
                    / I5);
#elif (RANK == 4)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                 * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                    / I5);
#elif (RANK == 5)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                -2
                    * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                       * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                    / I5);
#elif (RANK == 6)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5)
                 * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                    / I5);
#endif

#ifdef NEWF
            __GEIGEN__::Vector12 gradient_vec =
                __GEIGEN__::__M12x9_v9_multiply((PFPxT), flatten_pk1);
#else

            __GEIGEN__::Vector12 gradient_vec =
                __GEIGEN__::__M12x9_v9_multiply(__GEIGEN__::__Transpose9x12(PFPx), flatten_pk1);
#endif

            {
                _gfxAdd(MMCVIDI.x, 0, gradient_vec.v[0]);
                _gfxAdd(MMCVIDI.x, 1, gradient_vec.v[1]);
                _gfxAdd(MMCVIDI.x, 2, gradient_vec.v[2]);
                _gfxAdd(MMCVIDI.y, 0, gradient_vec.v[3]);
                _gfxAdd(MMCVIDI.y, 1, gradient_vec.v[4]);
                _gfxAdd(MMCVIDI.y, 2, gradient_vec.v[5]);
                _gfxAdd(MMCVIDI.z, 0, gradient_vec.v[6]);
                _gfxAdd(MMCVIDI.z, 1, gradient_vec.v[7]);
                _gfxAdd(MMCVIDI.z, 2, gradient_vec.v[8]);
                _gfxAdd(MMCVIDI.w, 0, gradient_vec.v[9]);
                _gfxAdd(MMCVIDI.w, 1, gradient_vec.v[10]);
                _gfxAdd(MMCVIDI.w, 2, gradient_vec.v[11]);
                _ec_emit(idx, _ec_out_pair, _ec_out_force, _ec_pbid, _ec_inv_dt2, MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w, 4, gradient_vec.v);
            }
        }
        else
        {
            //return;
            MMCVIDI.w = -MMCVIDI.w - 1;
            double3 v0 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.x]);
            double3 v1 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.z]);
            double c = __GEIGEN__::__norm(__GEIGEN__::__v_vec_cross(v0, v1)) /*/ __GEIGEN__::__norm(v0)*/;
            double I1 = c * c;
            if(I1 == 0)
                return;
            double dis;
            _d_EE(_vertexes[MMCVIDI.x],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            double I2 = dis / dHat;
            dis       = sqrt(dis);

            __GEIGEN__::Matrix3x3d F;
            __GEIGEN__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
            double3 n1 = make_double3(0, 1, 0);
            double3 n2 = make_double3(0, 0, 1);

            double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                        _rest_vertexes[MMCVIDI.y],
                                        _rest_vertexes[MMCVIDI.z],
                                        _rest_vertexes[MMCVIDI.w]);

            __GEIGEN__::Matrix3x3d g1, g2;

            __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(n1, n1);
            __GEIGEN__::__M_Mat_multiply(F, nn, g1);
            nn = __GEIGEN__::__v_vec_toMat(n2, n2);
            __GEIGEN__::__M_Mat_multiply(F, nn, g2);

            __GEIGEN__::Vector9 flatten_g1 = __GEIGEN__::__Mat3x3_to_vec9_double(g1);
            __GEIGEN__::Vector9 flatten_g2 = __GEIGEN__::__Mat3x3_to_vec9_double(g2);

            __GEIGEN__::Matrix12x9d PFPx;
            pFpx_pee(_vertexes[MMCVIDI.x],
                     _vertexes[MMCVIDI.y],
                     _vertexes[MMCVIDI.z],
                     _vertexes[MMCVIDI.w],
                     dHat_sqrt,
                     PFPx);


#if (RANK == 1)
            double p1 = Kappa * 2
                        * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                        / (eps_x * eps_x);
            double p2 = Kappa * 2
                        * (I1 * dHat * dHat * (I1 - 2 * eps_x) * (I2 - 1)
                           * (I2 + 2 * I2 * log(I2) - 1))
                        / (I2 * eps_x * eps_x);
#elif (RANK == 2)
            double p1 = -Kappa * 2
                        * (2 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x)
                           * (I2 - 1) * (I2 - 1))
                        / (eps_x * eps_x);
            double p2 = -Kappa * 2
                        * (2 * I1 * dHat * dHat * log(I2) * (I1 - 2 * eps_x)
                           * (I2 - 1) * (I2 + I2 * log(I2) - 1))
                        / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
            double p1 = -Kappa * 2
                        * (2 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x)
                           * (I2 - 1) * (I2 - 1))
                        / (eps_x * eps_x);
            double p2 = -Kappa * 2
                        * (2 * I1 * dHat * dHat * pow(log(I2), 3) * (I1 - 2 * eps_x)
                           * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2))
                        / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
            double p1 = -Kappa * 2
                        * (2 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x)
                           * (I2 - 1) * (I2 - 1))
                        / (eps_x * eps_x);
            double p2 = -Kappa * 2
                        * (2 * I1 * dHat * dHat * pow(log(I2), 5) * (I1 - 2 * eps_x)
                           * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3))
                        / (I2 * (eps_x * eps_x));
#endif


            __GEIGEN__::Vector9 flatten_pk1 =
                __GEIGEN__::__add9(__GEIGEN__::__s_vec9_multiply(flatten_g1, p1),
                                   __GEIGEN__::__s_vec9_multiply(flatten_g2, p2));
            __GEIGEN__::Vector12 gradient_vec =
                __GEIGEN__::__M12x9_v9_multiply(PFPx, flatten_pk1);

            {
                _gfxAdd(MMCVIDI.x, 0, gradient_vec.v[0]);
                _gfxAdd(MMCVIDI.x, 1, gradient_vec.v[1]);
                _gfxAdd(MMCVIDI.x, 2, gradient_vec.v[2]);
                _gfxAdd(MMCVIDI.y, 0, gradient_vec.v[3]);
                _gfxAdd(MMCVIDI.y, 1, gradient_vec.v[4]);
                _gfxAdd(MMCVIDI.y, 2, gradient_vec.v[5]);
                _gfxAdd(MMCVIDI.z, 0, gradient_vec.v[6]);
                _gfxAdd(MMCVIDI.z, 1, gradient_vec.v[7]);
                _gfxAdd(MMCVIDI.z, 2, gradient_vec.v[8]);
                _gfxAdd(MMCVIDI.w, 0, gradient_vec.v[9]);
                _gfxAdd(MMCVIDI.w, 1, gradient_vec.v[10]);
                _gfxAdd(MMCVIDI.w, 2, gradient_vec.v[11]);
                _ec_emit(idx, _ec_out_pair, _ec_out_force, _ec_pbid, _ec_inv_dt2, MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w, 4, gradient_vec.v);
            }
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
                    return;
                double dis;
                _d_PP(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], dis);
                double I2 = dis / dHat;
                dis       = sqrt(dis);

                __GEIGEN__::Matrix3x3d F;
                __GEIGEN__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
                double3 n1 = make_double3(0, 1, 0);
                double3 n2 = make_double3(0, 0, 1);

                double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                            _rest_vertexes[MMCVIDI.z],
                                            _rest_vertexes[MMCVIDI.y],
                                            _rest_vertexes[MMCVIDI.w]);

                __GEIGEN__::Matrix3x3d g1, g2;

                __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(n1, n1);
                __GEIGEN__::__M_Mat_multiply(F, nn, g1);
                nn = __GEIGEN__::__v_vec_toMat(n2, n2);
                __GEIGEN__::__M_Mat_multiply(F, nn, g2);

                __GEIGEN__::Vector9 flatten_g1 = __GEIGEN__::__Mat3x3_to_vec9_double(g1);
                __GEIGEN__::Vector9 flatten_g2 = __GEIGEN__::__Mat3x3_to_vec9_double(g2);

                __GEIGEN__::Matrix12x9d PFPx;
                pFpx_ppp(_vertexes[MMCVIDI.x],
                         _vertexes[MMCVIDI.y],
                         _vertexes[MMCVIDI.z],
                         _vertexes[MMCVIDI.w],
                         dHat_sqrt,
                         PFPx);
#if (RANK == 1)
                double p1 =
                    Kappa * 2
                    * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                    / (eps_x * eps_x);
                double p2 = Kappa * 2
                            * (I1 * dHat * dHat * (I1 - 2 * eps_x) * (I2 - 1)
                               * (I2 + 2 * I2 * log(I2) - 1))
                            / (I2 * eps_x * eps_x);
#elif (RANK == 2)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * log(I2) * log(I2)
                               * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * log(I2) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (I2 + I2 * log(I2) - 1))
                            / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x)
                               * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * pow(log(I2), 3) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2))
                            / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x)
                               * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * pow(log(I2), 5) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3))
                            / (I2 * (eps_x * eps_x));
#endif
                __GEIGEN__::Vector9 flatten_pk1 =
                    __GEIGEN__::__add9(__GEIGEN__::__s_vec9_multiply(flatten_g1, p1),
                                       __GEIGEN__::__s_vec9_multiply(flatten_g2, p2));
                __GEIGEN__::Vector12 gradient_vec =
                    __GEIGEN__::__M12x9_v9_multiply(PFPx, flatten_pk1);

                {
                    _gfxAdd(MMCVIDI.x, 0, gradient_vec.v[0]);
                    _gfxAdd(MMCVIDI.x, 1, gradient_vec.v[1]);
                    _gfxAdd(MMCVIDI.x, 2, gradient_vec.v[2]);
                    _gfxAdd(MMCVIDI.y, 0, gradient_vec.v[3]);
                    _gfxAdd(MMCVIDI.y, 1, gradient_vec.v[4]);
                    _gfxAdd(MMCVIDI.y, 2, gradient_vec.v[5]);
                    _gfxAdd(MMCVIDI.z, 0, gradient_vec.v[6]);
                    _gfxAdd(MMCVIDI.z, 1, gradient_vec.v[7]);
                    _gfxAdd(MMCVIDI.z, 2, gradient_vec.v[8]);
                    _gfxAdd(MMCVIDI.w, 0, gradient_vec.v[9]);
                    _gfxAdd(MMCVIDI.w, 1, gradient_vec.v[10]);
                    _gfxAdd(MMCVIDI.w, 2, gradient_vec.v[11]);
                    _ec_emit(idx, _ec_out_pair, _ec_out_force, _ec_pbid, _ec_inv_dt2, MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w, 4, gradient_vec.v);
                }
            }
            else
            {
#ifdef NEWF
                double dis;
                _d_PP(_vertexes[v0I], _vertexes[MMCVIDI.y], dis);
                dis                            = sqrt(dis);
                double              d_hat_sqrt = sqrt(dHat);
                __GEIGEN__::Vector6 PFPxT;
                pFpx_pp2(_vertexes[v0I], _vertexes[MMCVIDI.y], d_hat_sqrt, PFPxT);
                double I5  = pow(dis / d_hat_sqrt, 2);
                double fnn = dis / d_hat_sqrt;

#if (RANK == 1)


                double judge = (2 * dHat * dHat
                                * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5
                                   - 6 * I5 * I5 * log(I5) + 1))
                               / I5;
                double judge2 =
                    2 * (dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1))
                    / I5 * dis / d_hat_sqrt;
                double flatten_pk1 =
                    fnn * 2 * Kappa
                    * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5;
                //if (dis*dis<1e-2*dHat)
                //flatten_pk1 = fnn * 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5 / (I5) /*/ (I5) / (I5)*/;
#elif (RANK == 2)
                //double flatten_pk1 = fnn * 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5;

                double judge =
                    -(4 * dHat * dHat
                      * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                         + 6 * I5 * log(I5) - 2 * I5 * I5
                         + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2))
                    / I5;
                double judge2 =
                    2 * (2 * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                    / I5 * dis / dHat_sqrt;
                double flatten_pk1 = fnn * 2
                                     * (2 * Kappa * dHat * dHat * log(I5)
                                        * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                                     / I5;
                //if (dis*dis<1e-2*dHat)
                //flatten_pk1 = fnn * 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5/I5;

#elif (RANK == 3)
                double flatten_pk1 = fnn * -2
                                     * (Kappa * dHat * dHat * log(I5) * log(I5)
                                        * (I5 - 1) * (3 * I5 + 2 * I5 * log(I5) - 3))
                                     / I5;
#elif (RANK == 4)
                double flatten_pk1 =
                    fnn
                    * (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                       * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                    / I5;
#elif (RANK == 5)
                double flatten_pk1 =
                    fnn * -2
                    * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                       * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                    / I5;
#elif (RANK == 6)
                double flatten_pk1 =
                    fnn
                    * (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                       * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                    / I5;
#endif

                __GEIGEN__::Vector6 gradient_vec =
                    __GEIGEN__::__s_vec6_multiply(PFPxT, flatten_pk1);

#else
                double3 v0 = __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[v0I]);
                double3 Ds  = v0;
                double  dis = __GEIGEN__::__norm(v0);
                //if (dis > dHat_sqrt) return;
                double3 vec_normal =
                    __GEIGEN__::__normalized(make_double3(-v0.x, -v0.y, -v0.z));
                double3 target = make_double3(0, 1, 0);
                double3 vec    = __GEIGEN__::__v_vec_cross(vec_normal, target);
                double  cos    = __GEIGEN__::__v_vec_dot(vec_normal, target);
                __GEIGEN__::Matrix3x3d rotation;
                __GEIGEN__::__set_Mat_val(rotation, 1, 0, 0, 0, 1, 0, 0, 0, 1);
                __GEIGEN__::Vector6 PDmPx;
                if(cos + 1 == 0)
                {
                    rotation.m[0][0] = -1;
                    rotation.m[1][1] = -1;
                }
                else
                {
                    __GEIGEN__::Matrix3x3d cross_vec;
                    __GEIGEN__::__set_Mat_val(
                        cross_vec, 0, -vec.z, vec.y, vec.z, 0, -vec.x, -vec.y, vec.x, 0);

                    rotation = __GEIGEN__::__Mat_add(
                        rotation,
                        __GEIGEN__::__Mat_add(cross_vec,
                                              __GEIGEN__::__S_Mat_multiply(
                                                  __GEIGEN__::__M_Mat_multiply(cross_vec, cross_vec),
                                                  1.0 / (1 + cos))));
                }

                double3 pos0 = __GEIGEN__::__add(
                    _vertexes[v0I],
                    __GEIGEN__::__s_vec_multiply(vec_normal, dHat_sqrt - dis));
                double3 rotate_uv0 = __GEIGEN__::__M_v_multiply(rotation, pos0);
                double3 rotate_uv1 =
                    __GEIGEN__::__M_v_multiply(rotation, _vertexes[MMCVIDI.y]);

                double uv0 = rotate_uv0.y;
                double uv1 = rotate_uv1.y;

                double u0    = uv1 - uv0;
                double Dm    = u0;  //PFPx
                double DmInv = 1 / u0;

                double3 F  = __GEIGEN__::__s_vec_multiply(Ds, DmInv);
                double  I5 = __GEIGEN__::__squaredNorm(F);

                double3 tmp = F;

#if (RANK == 1)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
#elif (RANK == 2)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp,
                    2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                        / I5);
#elif (RANK == 3)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1)
                           * (3 * I5 + 2 * I5 * log(I5) - 3))
                        / I5);
#elif (RANK == 4)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                        / I5);
#elif (RANK == 5)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                           * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                        / I5);
#elif (RANK == 6)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                        / I5);
#endif
                __GEIGEN__::Matrix3x6d PFPx = __computePFDsPX3D_3x6_double(DmInv);

                __GEIGEN__::Vector6 gradient_vec =
                    __GEIGEN__::__M6x3_v3_multiply(__GEIGEN__::__Transpose3x6(PFPx), flatten_pk1);
#endif


                {
                    _gfxAdd(v0I, 0, gradient_vec.v[0]);
                    _gfxAdd(v0I, 1, gradient_vec.v[1]);
                    _gfxAdd(v0I, 2, gradient_vec.v[2]);
                    _gfxAdd(MMCVIDI.y, 0, gradient_vec.v[3]);
                    _gfxAdd(MMCVIDI.y, 1, gradient_vec.v[4]);
                    _gfxAdd(MMCVIDI.y, 2, gradient_vec.v[5]);
                    _ec_emit(idx, _ec_out_pair, _ec_out_force, _ec_pbid, _ec_inv_dt2, v0I, MMCVIDI.y, 0, 0, 2, gradient_vec.v);
                }
            }
        }
        else if(MMCVIDI.w < 0)
        {
            if(MMCVIDI.y < 0)
            {
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.x = v0I;
                MMCVIDI.w = -MMCVIDI.w - 1;
                double3 v0 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.x]);
                double3 v1 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.y]);
                double c = __GEIGEN__::__norm(__GEIGEN__::__v_vec_cross(v0, v1)) /*/ __GEIGEN__::__norm(v0)*/;
                double I1 = c * c;
                if(I1 == 0)
                    return;
                double dis;
                _d_PE(_vertexes[MMCVIDI.x],
                      _vertexes[MMCVIDI.y],
                      _vertexes[MMCVIDI.z],
                      dis);
                double I2 = dis / dHat;
                dis       = sqrt(dis);

                __GEIGEN__::Matrix3x3d F;
                __GEIGEN__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
                double3 n1 = make_double3(0, 1, 0);
                double3 n2 = make_double3(0, 0, 1);

                double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                            _rest_vertexes[MMCVIDI.w],
                                            _rest_vertexes[MMCVIDI.y],
                                            _rest_vertexes[MMCVIDI.z]);

                __GEIGEN__::Matrix3x3d g1, g2;

                __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(n1, n1);
                __GEIGEN__::__M_Mat_multiply(F, nn, g1);
                nn = __GEIGEN__::__v_vec_toMat(n2, n2);
                __GEIGEN__::__M_Mat_multiply(F, nn, g2);

                __GEIGEN__::Vector9 flatten_g1 = __GEIGEN__::__Mat3x3_to_vec9_double(g1);
                __GEIGEN__::Vector9 flatten_g2 = __GEIGEN__::__Mat3x3_to_vec9_double(g2);

                __GEIGEN__::Matrix12x9d PFPx;
                pFpx_ppe(_vertexes[MMCVIDI.x],
                         _vertexes[MMCVIDI.y],
                         _vertexes[MMCVIDI.z],
                         _vertexes[MMCVIDI.w],
                         dHat_sqrt,
                         PFPx);

#if (RANK == 1)
                double p1 =
                    Kappa * 2
                    * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                    / (eps_x * eps_x);
                double p2 = Kappa * 2
                            * (I1 * dHat * dHat * (I1 - 2 * eps_x) * (I2 - 1)
                               * (I2 + 2 * I2 * log(I2) - 1))
                            / (I2 * eps_x * eps_x);
#elif (RANK == 2)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * log(I2) * log(I2)
                               * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * log(I2) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (I2 + I2 * log(I2) - 1))
                            / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x)
                               * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * pow(log(I2), 3) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2))
                            / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x)
                               * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * pow(log(I2), 5) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3))
                            / (I2 * (eps_x * eps_x));
#endif
                __GEIGEN__::Vector9 flatten_pk1 =
                    __GEIGEN__::__add9(__GEIGEN__::__s_vec9_multiply(flatten_g1, p1),
                                       __GEIGEN__::__s_vec9_multiply(flatten_g2, p2));
                __GEIGEN__::Vector12 gradient_vec =
                    __GEIGEN__::__M12x9_v9_multiply(PFPx, flatten_pk1);

                {
                    _gfxAdd(MMCVIDI.x, 0, gradient_vec.v[0]);
                    _gfxAdd(MMCVIDI.x, 1, gradient_vec.v[1]);
                    _gfxAdd(MMCVIDI.x, 2, gradient_vec.v[2]);
                    _gfxAdd(MMCVIDI.y, 0, gradient_vec.v[3]);
                    _gfxAdd(MMCVIDI.y, 1, gradient_vec.v[4]);
                    _gfxAdd(MMCVIDI.y, 2, gradient_vec.v[5]);
                    _gfxAdd(MMCVIDI.z, 0, gradient_vec.v[6]);
                    _gfxAdd(MMCVIDI.z, 1, gradient_vec.v[7]);
                    _gfxAdd(MMCVIDI.z, 2, gradient_vec.v[8]);
                    _gfxAdd(MMCVIDI.w, 0, gradient_vec.v[9]);
                    _gfxAdd(MMCVIDI.w, 1, gradient_vec.v[10]);
                    _gfxAdd(MMCVIDI.w, 2, gradient_vec.v[11]);
                    _ec_emit(idx, _ec_out_pair, _ec_out_force, _ec_pbid, _ec_inv_dt2, MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w, 4, gradient_vec.v);
                }
            }
            else
            {
#ifdef NEWF
                double dis;
                _d_PE(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dis);
                dis                               = sqrt(dis);
                double                 d_hat_sqrt = sqrt(dHat);
                __GEIGEN__::Matrix9x4d PFPxT;
                pFpx_pe2(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], d_hat_sqrt, PFPxT);
                double              I5 = pow(dis / d_hat_sqrt, 2);
                __GEIGEN__::Vector4 fnn;
                fnn.v[0] = fnn.v[1] = fnn.v[2] = 0;  // = fnn.v[3] = fnn.v[4] = 1;
                fnn.v[3] = dis / d_hat_sqrt;
                //__GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(fnn, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);

#if (RANK == 1)


                double judge = (2 * dHat * dHat
                                * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5
                                   - 6 * I5 * I5 * log(I5) + 1))
                               / I5;
                double judge2 =
                    2 * (dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1))
                    / I5 * dis / d_hat_sqrt;
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
                //if (dis*dis<1e-2*dHat)
                //flatten_pk1 = __GEIGEN__::__s_vec4_multiply(fnn, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5 / (I5) /*/ (I5) / (I5)*/);

#elif (RANK == 2)
                //__GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(fnn, 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);

                double judge =
                    -(4 * dHat * dHat
                      * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                         + 6 * I5 * log(I5) - 2 * I5 * I5
                         + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2))
                    / I5;
                double judge2 =
                    2 * (2 * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                    / I5 * dis / dHat_sqrt;
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn,
                    2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                        / I5);
                //if (dis*dis<1e-2*dHat)
                //flatten_pk1 = __GEIGEN__::__s_vec4_multiply(fnn, 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5/I5);
#elif (RANK == 3)
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1)
                           * (3 * I5 + 2 * I5 * log(I5) - 3))
                        / I5);
#elif (RANK == 4)
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                        / I5);
#elif (RANK == 5)
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                           * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                        / I5);
#elif (RANK == 6)
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                        / I5);
#endif

                __GEIGEN__::Vector9 gradient_vec =
                    __GEIGEN__::__M9x4_v4_multiply(PFPxT, flatten_pk1);
#else

                double3 v0 = __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[v0I]);
                double3 v1 = __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[v0I]);


                __GEIGEN__::Matrix3x2d Ds;
                __GEIGEN__::__set_Mat3x2_val_column(Ds, v0, v1);

                double3 triangle_normal =
                    __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(v0, v1));
                double3 target = make_double3(0, 1, 0);

                double3 vec = __GEIGEN__::__v_vec_cross(triangle_normal, target);
                double cos = __GEIGEN__::__v_vec_dot(triangle_normal, target);

                double3 edge_normal = __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z]),
                    triangle_normal));
                double dis = __GEIGEN__::__v_vec_dot(
                    __GEIGEN__::__minus(_vertexes[v0I], _vertexes[MMCVIDI.y]), edge_normal);

                __GEIGEN__::Matrix3x3d rotation;
                __GEIGEN__::__set_Mat_val(rotation, 1, 0, 0, 0, 1, 0, 0, 0, 1);

                __GEIGEN__::Matrix9x4d PDmPx;

                if(cos + 1 == 0)
                {
                    rotation.m[0][0] = -1;
                    rotation.m[1][1] = -1;
                }
                else
                {
                    __GEIGEN__::Matrix3x3d cross_vec;
                    __GEIGEN__::__set_Mat_val(
                        cross_vec, 0, -vec.z, vec.y, vec.z, 0, -vec.x, -vec.y, vec.x, 0);

                    rotation = __GEIGEN__::__Mat_add(
                        rotation,
                        __GEIGEN__::__Mat_add(cross_vec,
                                              __GEIGEN__::__S_Mat_multiply(
                                                  __GEIGEN__::__M_Mat_multiply(cross_vec, cross_vec),
                                                  1.0 / (1 + cos))));
                }

                double3 pos0 = __GEIGEN__::__add(
                    _vertexes[v0I],
                    __GEIGEN__::__s_vec_multiply(edge_normal, dHat_sqrt - dis));

                double3 rotate_uv0 = __GEIGEN__::__M_v_multiply(rotation, pos0);
                double3 rotate_uv1 =
                    __GEIGEN__::__M_v_multiply(rotation, _vertexes[MMCVIDI.y]);
                double3 rotate_uv2 =
                    __GEIGEN__::__M_v_multiply(rotation, _vertexes[MMCVIDI.z]);
                double3 rotate_normal = __GEIGEN__::__M_v_multiply(rotation, edge_normal);

                double2 uv0    = make_double2(rotate_uv0.x, rotate_uv0.z);
                double2 uv1    = make_double2(rotate_uv1.x, rotate_uv1.z);
                double2 uv2    = make_double2(rotate_uv2.x, rotate_uv2.z);
                double2 normal = make_double2(rotate_normal.x, rotate_normal.z);

                double2 u0 = __GEIGEN__::__minus_v2(uv1, uv0);
                double2 u1 = __GEIGEN__::__minus_v2(uv2, uv0);

                __GEIGEN__::Matrix2x2d Dm;

                __GEIGEN__::__set_Mat2x2_val_column(Dm, u0, u1);

                __GEIGEN__::Matrix2x2d DmInv;
                __GEIGEN__::__Inverse2x2(Dm, DmInv);

                __GEIGEN__::Matrix3x2d F = __GEIGEN__::__M3x2_M2x2_Multiply(Ds, DmInv);

                double3 FxN = __GEIGEN__::__M3x2_v2_multiply(F, normal);
                double  I5  = __GEIGEN__::__squaredNorm(FxN);

                __GEIGEN__::Matrix3x2d fnn;

                __GEIGEN__::Matrix2x2d nn = __GEIGEN__::__v2_vec2_toMat2x2(normal, normal);

                fnn = __GEIGEN__::__M3x2_M2x2_Multiply(F, nn);

                __GEIGEN__::Vector6 tmp = __GEIGEN__::__Mat3x2_to_vec6_double(fnn);


#if (RANK == 1)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
#elif (RANK == 2)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp,
                    2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                        / I5);
#elif (RANK == 3)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1)
                           * (3 * I5 + 2 * I5 * log(I5) - 3))
                        / I5);
#elif (RANK == 4)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                        / I5);
#elif (RANK == 5)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                           * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                        / I5);
#elif (RANK == 6)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                        / I5);
#endif

                __GEIGEN__::Matrix6x9d PFPx = __computePFDsPX3D_6x9_double(DmInv);

                __GEIGEN__::Vector9 gradient_vec =
                    __GEIGEN__::__M9x6_v6_multiply(__GEIGEN__::__Transpose6x9(PFPx), flatten_pk1);
#endif

                {
                    _gfxAdd(v0I, 0, gradient_vec.v[0]);
                    _gfxAdd(v0I, 1, gradient_vec.v[1]);
                    _gfxAdd(v0I, 2, gradient_vec.v[2]);
                    _gfxAdd(MMCVIDI.y, 0, gradient_vec.v[3]);
                    _gfxAdd(MMCVIDI.y, 1, gradient_vec.v[4]);
                    _gfxAdd(MMCVIDI.y, 2, gradient_vec.v[5]);
                    _gfxAdd(MMCVIDI.z, 0, gradient_vec.v[6]);
                    _gfxAdd(MMCVIDI.z, 1, gradient_vec.v[7]);
                    _gfxAdd(MMCVIDI.z, 2, gradient_vec.v[8]);
                    _ec_emit(idx, _ec_out_pair, _ec_out_force, _ec_pbid, _ec_inv_dt2, v0I, MMCVIDI.y, MMCVIDI.z, 0, 3, gradient_vec.v);
                }
            }
        }
        else
        {
#ifdef NEWF
            double dis;
            _d_PT(_vertexes[v0I],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            dis                                = sqrt(dis);
            double                  d_hat_sqrt = sqrt(dHat);
            __GEIGEN__::Matrix12x9d PFPxT;
            pFpx_pt2(_vertexes[v0I],
                     _vertexes[MMCVIDI.y],
                     _vertexes[MMCVIDI.z],
                     _vertexes[MMCVIDI.w],
                     d_hat_sqrt,
                     PFPxT);
            double              I5 = pow(dis / d_hat_sqrt, 2);
            __GEIGEN__::Vector9 tmp;
            tmp.v[0] = tmp.v[1] = tmp.v[2] = tmp.v[3] = tmp.v[4] = tmp.v[5] =
                tmp.v[6] = tmp.v[7] = 0;
            tmp.v[8]                = dis / d_hat_sqrt;
#else
            double3 v0 = __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[v0I]);
            double3 v1 = __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[v0I]);
            double3 v2 = __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[v0I]);

            __GEIGEN__::Matrix3x3d Ds;
            __GEIGEN__::__set_Mat_val_column(Ds, v0, v1, v2);

            double3 normal = __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.y]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.y])));
            double  dis    = __GEIGEN__::__v_vec_dot(v0, normal);
            //if (abs(dis) > dHat_sqrt) return;
            __GEIGEN__::Matrix12x9d PDmPx;
            //bool is_flip = false;

            if(dis > 0)
            {
                //is_flip = true;
                normal = make_double3(-normal.x, -normal.y, -normal.z);
                //pDmpx_pt_flip(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dHat_sqrt, PDmPx);
                //printf("dHat_sqrt = %f,   dis = %f\n", dHat_sqrt, dis);
            }
            else
            {
                dis = -dis;
                //pDmpx_pt(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dHat_sqrt, PDmPx);
                //printf("dHat_sqrt = %f,   dis = %f\n", dHat_sqrt, dis);
            }

            double3 pos0 = __GEIGEN__::__add(
                _vertexes[v0I], __GEIGEN__::__s_vec_multiply(normal, dHat_sqrt - dis));


            double3 u0 = __GEIGEN__::__minus(_vertexes[MMCVIDI.y], pos0);
            double3 u1 = __GEIGEN__::__minus(_vertexes[MMCVIDI.z], pos0);
            double3 u2 = __GEIGEN__::__minus(_vertexes[MMCVIDI.w], pos0);

            __GEIGEN__::Matrix3x3d Dm, DmInv;
            __GEIGEN__::__set_Mat_val_column(Dm, u0, u1, u2);

            __GEIGEN__::__Inverse(Dm, DmInv);

            __GEIGEN__::Matrix3x3d F;  //, Ftest;
            __GEIGEN__::__M_Mat_multiply(Ds, DmInv, F);
            //__GEIGEN__::__M_Mat_multiply(Dm, DmInv, Ftest);

            double3 FxN = __GEIGEN__::__M_v_multiply(F, normal);
            double  I5  = __GEIGEN__::__squaredNorm(FxN);

            //printf("I5 = %f,   dist/dHat_sqrt = %f\n", I5, (dis / dHat_sqrt)* (dis / dHat_sqrt));


            __GEIGEN__::Matrix9x12d PFPx = __computePFDsPX3D_double(DmInv);

            __GEIGEN__::Matrix3x3d fnn;

            __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(normal, normal);

            __GEIGEN__::__M_Mat_multiply(F, nn, fnn);

            __GEIGEN__::Vector9 tmp = __GEIGEN__::__Mat3x3_to_vec9_double(fnn);
#endif


#if (RANK == 1)


            double judge =
                (2 * dHat * dHat
                 * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5 - 6 * I5 * I5 * log(I5) + 1))
                / I5;
            double judge2 = 2 * (dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1))
                            / I5 * dis / d_hat_sqrt;
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
            //if (dis*dis<1e-2*dHat)
            //flatten_pk1 = __GEIGEN__::__s_vec9_multiply(tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5 / (I5) /*/ (I5) / (I5)*/);

#elif (RANK == 2)
            //__GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(tmp, 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);

            double judge = -(4 * dHat * dHat
                             * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                                + 6 * I5 * log(I5) - 2 * I5 * I5
                                + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2))
                           / I5;
            double judge2 =
                2 * (2 * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                / I5 * dis / dHat_sqrt;
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);
            //if (dis*dis<1e-2*dHat)
            //flatten_pk1 = __GEIGEN__::__s_vec9_multiply(tmp, 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5/I5);
#elif (RANK == 3)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                -2
                    * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1)
                       * (3 * I5 + 2 * I5 * log(I5) - 3))
                    / I5);
#elif (RANK == 4)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                 * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                    / I5);
#elif (RANK == 5)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                -2
                    * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                       * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                    / I5);
#elif (RANK == 6)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5)
                 * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                    / I5);
#endif

#ifdef NEWF
            __GEIGEN__::Vector12 gradient_vec =
                __GEIGEN__::__M12x9_v9_multiply(PFPxT, flatten_pk1);
#else
            __GEIGEN__::Vector12 gradient_vec =
                __GEIGEN__::__M12x9_v9_multiply(__GEIGEN__::__Transpose9x12(PFPx), flatten_pk1);
#endif

            _gfxAdd(v0I, 0, gradient_vec.v[0]);
            _gfxAdd(v0I, 1, gradient_vec.v[1]);
            _gfxAdd(v0I, 2, gradient_vec.v[2]);
            _gfxAdd(MMCVIDI.y, 0, gradient_vec.v[3]);
            _gfxAdd(MMCVIDI.y, 1, gradient_vec.v[4]);
            _gfxAdd(MMCVIDI.y, 2, gradient_vec.v[5]);
            _gfxAdd(MMCVIDI.z, 0, gradient_vec.v[6]);
            _gfxAdd(MMCVIDI.z, 1, gradient_vec.v[7]);
            _gfxAdd(MMCVIDI.z, 2, gradient_vec.v[8]);
            _gfxAdd(MMCVIDI.w, 0, gradient_vec.v[9]);
            _gfxAdd(MMCVIDI.w, 1, gradient_vec.v[10]);
            _gfxAdd(MMCVIDI.w, 2, gradient_vec.v[11]);
            _ec_emit(idx, _ec_out_pair, _ec_out_force, _ec_pbid, _ec_inv_dt2, MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w, 4, gradient_vec.v);
        }
    }
}

__global__ void _calKineticGradient(
    double3* vertexes, double3* xTilta, double3* gradient, double* masses, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    double3 deltaX = __GEIGEN__::__minus(vertexes[idx], xTilta[idx]);
    //masses[idx] = 1;
    gradient[idx] = make_double3(
        deltaX.x * masses[idx], deltaX.y * masses[idx], deltaX.z * masses[idx]);
    //printf("%f  %f  %f\n", gradient[idx].x, gradient[idx].y, gradient[idx].z);
}

__global__ void _calKineticEnergy(
    double3* vertexes, double3* xTilta, double3* gradient, double* masses, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    double3 deltaX = __GEIGEN__::__minus(vertexes[idx], xTilta[idx]);
    gradient[idx]  = make_double3(
        deltaX.x * masses[idx], deltaX.y * masses[idx], deltaX.z * masses[idx]);
}

__global__ void _computeSoftConstraintGradientAndHessian(const double3* vertexes,
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
                                                         int number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    uint32_t vInd = targetInd[idx];
    double   x = vertexes[vInd].x, y = vertexes[vInd].y, z = vertexes[vInd].z;
    double   a, b, c;
    // For bilateral stitch springs, compute target dynamically from current ABD vertex.
    // [stitch local-frame fix] target = anchor_world + A_now * local_offset where
    // local_offset is in the ABD body's rest frame.  Per the canonical q layout in
    // abd_jacobi_matrix.inl operator*(ABDJacobi, Vector12):
    //   q[3..5]  = A.row(0),  q[6..8]  = A.row(1),  q[9..11] = A.row(2)
    // So (A * lo).x = q[3]*lo.x + q[4]*lo.y + q[5]*lo.z, etc.
    // Without this, the stitch target only follows ABD translation, not rotation,
    // so FEM mesh visibly fails to track ABD rotation.
    if(stitch_paired_vertex && stitch_paired_vertex[idx] >= 0)
    {
        int abd_idx = stitch_paired_vertex[idx];
        double3 lo = stitch_rest_offset[idx];
        if(abd_body_q != nullptr && stitch_abd_body_id != nullptr)
        {
            int bid = stitch_abd_body_id[idx];
            const __GEIGEN__::Vector12& q = abd_body_q[bid];
            // Previous code used q[3]/q[6]/q[9] for the x-component, which is
            // A.col(0)·lo = (A^T·lo)[0] — wrong for non-symmetric A.  Only
            // worked when A ≈ scale·I (Animated mode where ABD barely rotates).
            a = vertexes[abd_idx].x + q.v[3] * lo.x + q.v[4] * lo.y + q.v[5]  * lo.z;
            b = vertexes[abd_idx].y + q.v[6] * lo.x + q.v[7] * lo.y + q.v[8]  * lo.z;
            c = vertexes[abd_idx].z + q.v[9] * lo.x + q.v[10] * lo.y + q.v[11] * lo.z;
        }
        else
        {
            // Fallback: legacy world-frame offset (no rotation tracking).
            a = vertexes[abd_idx].x + lo.x;
            b = vertexes[abd_idx].y + lo.y;
            c = vertexes[abd_idx].z + lo.z;
        }
    }
    else
    {
        a = targetVert[idx].x;
        b = targetVert[idx].y;
        c = targetVert[idx].z;
    }
    //double dis = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(vertexes[vInd], targetVert[idx]));
    //printf("%f\n", dis);
    double d = motionRate;
    {
        _gfxAdd(vInd, 0, d * rate * rate * (x - a));
        _gfxAdd(vInd, 1, d * rate * rate * (y - b));
        _gfxAdd(vInd, 2, d * rate * rate * (z - c));
    }
    __GEIGEN__::Matrix3x3d Hpg;
    Hpg.m[0][0] = rate * rate * d;
    Hpg.m[0][1] = 0;
    Hpg.m[0][2] = 0;
    Hpg.m[1][0] = 0;
    Hpg.m[1][1] = rate * rate * d;
    Hpg.m[1][2] = 0;
    Hpg.m[2][0] = 0;
    Hpg.m[2][1] = 0;
    Hpg.m[2][2] = rate * rate * d;
    int pidx    = atomicAdd(_gpNum, 1);
    //H3x3[pidx]    = Hpg;
    //D1Index[pidx] = vInd;
    vInd += global_hessian_fem_offset;
    write_triplet<3, 3>(triplet_values, row_ids, col_ids, &vInd, Hpg.m, global_offset + idx);
    //_environment_collisionPair[atomicAdd(_gpNum, 1)] = surfVertIds[idx];
}

__global__ void _computeSoftConstraintGradient(const double3*  vertexes,
                                               const double3*  targetVert,
                                               const uint32_t* targetInd,
                                               double3*        gradient,
                                               double          motionRate,
                                               double          rate,
                                               const int*      stitch_paired_vertex,
                                               const double3*  stitch_rest_offset,
                                               const int*      stitch_abd_body_id,
                                               const __GEIGEN__::Vector12* abd_body_q,
                                               int             number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    uint32_t vInd = targetInd[idx];
    double   x = vertexes[vInd].x, y = vertexes[vInd].y, z = vertexes[vInd].z;
    double   a, b, c;
    // [stitch local-frame fix] see _computeSoftConstraintGradientAndHessian
    if(stitch_paired_vertex && stitch_paired_vertex[idx] >= 0)
    {
        int abd_idx = stitch_paired_vertex[idx];
        double3 lo = stitch_rest_offset[idx];
        if(abd_body_q != nullptr && stitch_abd_body_id != nullptr)
        {
            int bid = stitch_abd_body_id[idx];
            const __GEIGEN__::Vector12& q = abd_body_q[bid];
            // q[3..5]/q[6..8]/q[9..11] = A.row(0/1/2); see _computeSoftConstraintGradientAndHessian.
            a = vertexes[abd_idx].x + q.v[3] * lo.x + q.v[4] * lo.y + q.v[5]  * lo.z;
            b = vertexes[abd_idx].y + q.v[6] * lo.x + q.v[7] * lo.y + q.v[8]  * lo.z;
            c = vertexes[abd_idx].z + q.v[9] * lo.x + q.v[10] * lo.y + q.v[11] * lo.z;
        }
        else
        {
            a = vertexes[abd_idx].x + lo.x;
            b = vertexes[abd_idx].y + lo.y;
            c = vertexes[abd_idx].z + lo.z;
        }
    }
    else
    {
        a = targetVert[idx].x;
        b = targetVert[idx].y;
        c = targetVert[idx].z;
    }
    //double dis = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(vertexes[vInd], targetVert[idx]));
    //printf("%f\n", dis);
    double d = motionRate;
    {
        _gfxAdd(vInd, 0, d * rate * rate * (x - a));
        _gfxAdd(vInd, 1, d * rate * rate * (y - b));
        _gfxAdd(vInd, 2, d * rate * rate * (z - c));
    }
}

__global__ void _GroundCollisionDetect(const double3*  vertexes,
                                       const uint32_t* surfVertIds,
                                       const double*   g_offset,
                                       const double3*  g_normal,
                                       uint32_t* _environment_collisionPair,
                                       uint32_t* _gpNum,
                                       double    dHat,
                                       int       number,
                                       const int* _point_body_id,
                                       const int* _ground_skip_body,
                                       int        _ground_body_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int svI = surfVertIds[idx];
    if(_point_body_id && _ground_skip_body && _ground_body_count > 0)
    {
        int bid = _point_body_id[svI];
        if(bid >= 0 && bid < _ground_body_count && _ground_skip_body[bid])
            return;
    }
    double dist = __GEIGEN__::__v_vec_dot(*g_normal, vertexes[svI]) - *g_offset;
    if(dist * dist > dHat)
        return;

    _environment_collisionPair[atomicAdd(_gpNum, 1)] = svI;
}

__global__ void _getTotalForce(const double3* _force0, double3* _force, int number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    _force[idx].x += _force0[idx].x;
    _force[idx].y += _force0[idx].y;
    _force[idx].z += _force0[idx].z;
}


__global__ void _computeGroundGradientAndHessian(const double3* vertexes,
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
                                                 const int*    p2g       = nullptr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    double3      normal = *g_normal;
    unsigned int gidx   = _environment_collisionPair[idx];
    // [multi-env per-group κ] ground pair is a single vertex; nullptr → scalar (baseline).
    double Kappa = (kappa_grp && p2g && p2g[gidx] >= 0) ? kappa_grp[p2g[gidx]] : Kappa_scalar /* [-1 guard] */;
    double dist  = __GEIGEN__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    double dist2 = dist * dist;
    // [d=0 guard] a vertex sitting EXACTLY on the ground (dist2==0) makes
    // log(dist2/dHat) and 1/dist2 below blow up to +-inf -> NaN in the gradient,
    // which poisons the whole global RHS and zeros the Newton search direction
    // (everything freezes). Clamp to a tiny positive value so d=0 yields a
    // large-but-finite push-out instead of NaN.
    dist2 = (dist2 == 0.0 ? 1e-12 : dist2);

    double t   = dist2 - dHat;
    double g_b = t * log(dist2 / dHat) * -2.0 - (t * t) / dist2;

    double H_b = (log(dist2 / dHat) * -2.0 - t * 4.0 / dist2)
                 + 1.0 / (dist2 * dist2) * (t * t);

    //printf("H_b   dist   g_b    is  %lf  %lf  %lf\n", H_b, dist2, g_b);

    double3 grad = __GEIGEN__::__s_vec_multiply(normal, Kappa * g_b * 2 * dist);

    {
        _gfxAdd(gidx, 0, grad.x);
        _gfxAdd(gidx, 1, grad.y);
        _gfxAdd(gidx, 2, grad.z);
    }

    double param = 4.0 * H_b * dist2 + 2.0 * g_b;
    //if(param > 0)
    {
        __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(normal, normal);
        __GEIGEN__::Matrix3x3d Hpg = __GEIGEN__::__S_Mat_multiply(nn, Kappa * param);

        int pidx = atomicAdd(_gpNum, 1);
        //H3x3[pidx]    = Hpg;
        //D1Index[pidx] = gidx;

        write_triplet<3, 3>(triplet_values, row_ids, col_ids, &gidx, Hpg.m, global_offset + idx);
    }
    //_environment_collisionPair[atomicAdd(_gpNum, 1)] = surfVertIds[idx];
}

__global__ void _computeGroundGradient(const double3* vertexes,
                                       const double*  g_offset,
                                       const double3* g_normal,
                                       const uint32_t* _environment_collisionPair,
                                       double3*  gradient,
                                       uint32_t* _gpNum,
                                       double    dHat,
                                       double    Kappa_scalar,
                                       int       number,
                                       const double* kappa_grp = nullptr,
                                       const int*    p2g       = nullptr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    double3 normal = *g_normal;
    int     gidx   = _environment_collisionPair[idx];
    // [multi-env per-group κ] nullptr → scalar (baseline).
    double  Kappa = (kappa_grp && p2g && p2g[gidx] >= 0) ? kappa_grp[p2g[gidx]] : Kappa_scalar /* [-1 guard] */;
    double  dist  = __GEIGEN__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    double  dist2 = dist * dist;
    dist2 = (dist2 == 0.0 ? 1e-12 : dist2);  // [d=0 guard] avoid ground-barrier NaN (see _computeGroundGradientAndHessian)

    double t   = dist2 - dHat;
    double g_b = t * std::log(dist2 / dHat) * -2.0 - (t * t) / dist2;

    //double H_b = (std::log(dist2 / dHat) * -2.0 - t * 4.0 / dist2) + 1.0 / (dist2 * dist2) * (t * t);
    double3 grad = __GEIGEN__::__s_vec_multiply(normal, Kappa * g_b * 2 * dist);

    {
        _gfxAdd(gidx, 0, grad.x);
        _gfxAdd(gidx, 1, grad.y);
        _gfxAdd(gidx, 2, grad.z);
    }
}

__global__ void _computeGroundCloseVal(const double3* vertexes,
                                       const double*  g_offset,
                                       const double3* g_normal,
                                       const uint32_t* _environment_collisionPair,
                                       double    dTol,
                                       uint32_t* _closeConstraintID,
                                       double*   _closeConstraintVal,
                                       uint32_t* _close_gpNum,
                                       int       number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    double3 normal = *g_normal;
    int     gidx   = _environment_collisionPair[idx];
    double  dist  = __GEIGEN__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    double  dist2 = dist * dist;

    if(dist2 < dTol)
    {
        int tidx                  = atomicAdd(_close_gpNum, 1);
        _closeConstraintID[tidx]  = gidx;
        _closeConstraintVal[tidx] = dist2;
    }
}

__global__ void _checkGroundCloseVal(const double3* vertexes,
                                     const double*  g_offset,
                                     const double3* g_normal,
                                     int*           _isChange,
                                     uint32_t*      _closeConstraintID,
                                     double*        _closeConstraintVal,
                                     int            number,
                                     int*           _isChange_grp = nullptr,
                                     const int*     p2g           = nullptr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    double3 normal = *g_normal;
    int     gidx   = _closeConstraintID[idx];
    double  dist  = __GEIGEN__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    double  dist2 = dist * dist;

    if(dist2 < _closeConstraintVal[idx])
    {
        *_isChange = 1;
        if(_isChange_grp && p2g && p2g[gidx] >= 0) _isChange_grp[p2g[gidx]] = 1;  /* [-1 guard] */   // [per-group κ]
    }
}

__global__ void _reduct_MGroundDist(const double3* vertexes,
                                    const double*  g_offset,
                                    const double3* g_normal,
                                    uint32_t*      _environment_collisionPair,
                                    double2*       _queue,
                                    int            number)
{
    int                       idof = blockIdx.x * blockDim.x;
    int                       idx  = threadIdx.x + idof;
    extern __shared__ double2 sdata[];

    if(idx >= number)
        return;
    double3 normal = *g_normal;
    int     gidx   = _environment_collisionPair[idx];
    double  dist  = __GEIGEN__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    double  tempv = dist * dist;
    double2 temp  = make_double2(1.0 / tempv, tempv);

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
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = sdata[threadIdx.x];

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
        _queue[blockIdx.x] = temp;
    }
}

__global__ void _computeSelfCloseVal(const double3*  vertexes,
                                     const double*   g_offset,
                                     const double3*  g_normal,
                                     const uint32_t* _environment_collisionPair,
                                     double          dTol,
                                     uint32_t*       _closeConstraintID,
                                     double*         _closeConstraintVal,
                                     uint32_t*       _close_gpNum,
                                     int             number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    double3 normal = *g_normal;
    int     gidx   = _environment_collisionPair[idx];
    double  dist  = __GEIGEN__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    double  dist2 = dist * dist;

    if(dist2 < dTol)
    {
        int tidx                  = atomicAdd(_close_gpNum, 1);
        _closeConstraintID[tidx]  = gidx;
        _closeConstraintVal[tidx] = dist2;
    }
}


__global__ void _checkGroundIntersection(const double3* vertexes,
                                         const double*  g_offset,
                                         const double3* g_normal,
                                         const uint32_t* _environment_collisionPair,
                                         int* _isIntersect,
                                         int  number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    double3 normal = *g_normal;
    int     gidx   = _environment_collisionPair[idx];
    double  dist = __GEIGEN__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    //printf("%f  %f\n", *g_offset, dist);
    if(dist < 0)
        *_isIntersect = -1;
}

// [multi-env S3] per-env energy accumulation helper. Element's env = group of one
// of its vertices (intra-env after P1). atomicAdd the per-element energy `e` into
// the env bucket. penv==nullptr -> no-op (global-only callers stay byte-identical).
// vid is a GLOBAL vertex id (p2g is the full point_to_group), except the kinetic
// caller passes p2g already offset to the FEM region and vid local.
__device__ inline void _penv_energy_accum(double* penv, const int* p2g, int vid, int ng, double e)
{
    if(!penv || !p2g) return;
    int g = p2g[vid];
    if(g >= 0 && g < ng) atomicAdd(&penv[g], e);
}

__global__ void _getFrictionEnergy_Reduction_3D(double*        squeue,
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
                                                double* penv = nullptr, const int* p2g = nullptr, int ng = 0

)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    int                      numbers = cpNum;
    if(idx >= numbers)
        return;

    double temp = __cal_Friction_energy(
        vertexes, o_vertexes, _collisionPair[idx], dt, distCoord[idx], tanBasis[idx], lastH[idx], fricDHat, eps);

    { int v0 = _collisionPair[idx].x; if(v0 < 0) v0 = -v0 - 1;  // [S3] friction pair env
      _penv_energy_accum(penv, p2g, v0, ng, temp); }

    int    warpTid = threadIdx.x % 32;
    int    warpId  = (threadIdx.x >> 5);
    double nextTp;
    int    warpNum;
    //int tidNum = 32;
    if(blockIdx.x == gridDim.x - 1)
    {
        //tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        temp += __shfl_down_sync(0xffffffff, temp, i);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];
        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        squeue[blockIdx.x] = temp;
    }
}

__global__ void _getFrictionEnergy_gd_Reduction_3D(double*        squeue,
                                                   const double3* vertexes,
                                                   const double3* o_vertexes,
                                                   const double3* _normal,
                                                   const uint32_t* _collisionPair_gd,
                                                   int           gpNum,
                                                   double        dt,
                                                   const double* lastH,
                                                   double        eps,
                                                   double* penv = nullptr, const int* p2g = nullptr, int ng = 0

)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    int                      numbers = gpNum;
    if(idx >= numbers)
        return;

    double temp = __cal_Friction_gd_energy(
        vertexes, o_vertexes, _normal, _collisionPair_gd[idx], dt, lastH[idx], eps);

    _penv_energy_accum(penv, p2g, _collisionPair_gd[idx], ng, temp);  // [S3] gd friction env

    int    warpTid = threadIdx.x % 32;
    int    warpId  = (threadIdx.x >> 5);
    double nextTp;
    int    warpNum;
    //int tidNum = 32;
    if(blockIdx.x == gridDim.x - 1)
    {
        //tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        temp += __shfl_down_sync(0xffffffff, temp, i);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];
        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        squeue[blockIdx.x] = temp;
    }
}

__global__ void _computeGroundEnergy_Reduction(double*        squeue,
                                               const double3* vertexes,
                                               const double*  g_offset,
                                               const double3* g_normal,
                                               const uint32_t* _environment_collisionPair,
                                               double dHat,
                                               double Kappa,
                                               int    number,
                                               double* penv = nullptr, const int* p2g = nullptr, int ng = 0)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= number)
        return;

    double3 normal = *g_normal;
    int     gidx   = _environment_collisionPair[idx];
    double  dist  = __GEIGEN__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    double  dist2 = dist * dist;
    dist2 = (dist2 == 0.0 ? 1e-12 : dist2);  // [d=0 guard] avoid ground-barrier energy NaN at d=0
    double  temp  = -(dist2 - dHat) * (dist2 - dHat) * log(dist2 / dHat);

    _penv_energy_accum(penv, p2g, gidx, ng, temp);  // [S3] ground pair's vertex env

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
        temp += __shfl_down_sync(0xffffffff, temp, i);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        squeue[blockIdx.x] = temp;
    }
}

__global__ void _reduct_min_groundTimeStep_to_double(const double3* vertexes,
                                                     const uint32_t* surfVertIds,
                                                     const double*  g_offset,
                                                     const double3* g_normal,
                                                     const double3* moveDir,
                                                     double* minStepSizes,
                                                     double  slackness,
                                                     int     number,
                                                     const int* _point_body_id,
                                                     const int* _ground_skip_body,
                                                     int        _ground_body_count)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= number)
        return;
    int     svI    = surfVertIds[idx];
    double  temp   = 1.0;
    bool skip = false;
    if(_point_body_id && _ground_skip_body && _ground_body_count > 0)
    {
        int bid = _point_body_id[svI];
        if(bid >= 0 && bid < _ground_body_count && _ground_skip_body[bid])
            skip = true;
    }
    if(!skip)
    {
        double3 normal = *g_normal;
        double  coef   = __GEIGEN__::__v_vec_dot(normal, moveDir[svI]);
        if(coef > 0.0)
        {
            double dist = __GEIGEN__::__v_vec_dot(normal, vertexes[svI]) - *g_offset;
            temp = coef / (dist * slackness);
        }
    }
    /*if (blockIdx.x == 4) {
        printf("%f\n", temp);
    }
    __syncthreads();*/
    //printf("%f\n", temp);
    int    warpTid = threadIdx.x % 32;
    int    warpId  = (threadIdx.x >> 5);
    double nextTp;
    int    warpNum;
    //int tidNum = 32;
    if(blockIdx.x == gridDim.x - 1)
    {
        //tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
        //printf("warpNum %d\n", warpNum);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        double tempMin = __shfl_down_sync(0xffffffff, temp, i);
        temp           = std::max(temp, tempMin);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            double tempMin = __shfl_down_sync(0xffffffff, temp, i);
            temp           = std::max(temp, tempMin);
        }
    }
    if(threadIdx.x == 0)
    {
        minStepSizes[blockIdx.x] = temp;
        //printf("%f   %d\n", temp, blockIdx.x);
    }
}

__global__ void _reduct_min_InjectiveTimeStep_to_double(const double3* vertexes,
                                                        const uint4* tetrahedra,
                                                        const double3* moveDir,
                                                        double* minStepSizes,
                                                        double  slackness,
                                                        double  errorRate,
                                                        int     number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= number)
        return;
    double ratio = 1 - slackness;

    double temp = 1.0
                  / _computeInjectiveStepSize_3d(vertexes,
                                                 moveDir,
                                                 tetrahedra[idx].x,
                                                 tetrahedra[idx].y,
                                                 tetrahedra[idx].z,
                                                 tetrahedra[idx].w,
                                                 ratio,
                                                 errorRate);

    int    warpTid = threadIdx.x % 32;
    int    warpId  = (threadIdx.x >> 5);
    double nextTp;
    int    warpNum;
    //int tidNum = 32;
    if(blockIdx.x == gridDim.x - 1)
    {
        //tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
        //printf("warpNum %d\n", warpNum);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        double tempMin = __shfl_down_sync(0xffffffff, temp, i);
        temp           = std::max(temp, tempMin);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            double tempMin = __shfl_down_sync(0xffffffff, temp, i);
            temp           = std::max(temp, tempMin);
        }
    }
    if(threadIdx.x == 0)
    {
        minStepSizes[blockIdx.x] = temp;
        //printf("%f   %d\n", temp, blockIdx.x);
    }
}

__global__ void _reduct_min_selfTimeStep_to_double(const double3* vertexes,
                                                   const int4* _ccd_collitionPairs,
                                                   const double3* moveDir,
                                                   double*        minStepSizes,
                                                   double         slackness,
                                                   int            number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= number)
        return;
    double temp         = 1.0;
    double CCDDistRatio = 1.0 - slackness;

    int4 MMCVIDI = _ccd_collitionPairs[idx];

    if(MMCVIDI.x < 0)
    {
        MMCVIDI.x = -MMCVIDI.x - 1;

        double temp1 =
            point_triangle_ccd(vertexes[MMCVIDI.x],
                               vertexes[MMCVIDI.y],
                               vertexes[MMCVIDI.z],
                               vertexes[MMCVIDI.w],
                               __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.x], -1),
                               __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.y], -1),
                               __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.z], -1),
                               __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.w], -1),
                               CCDDistRatio,
                               0);

        //double temp2 = doCCDVF(vertexes[MMCVIDI.x],
        //    vertexes[MMCVIDI.y],
        //    vertexes[MMCVIDI.z],
        //    vertexes[MMCVIDI.w],
        //    __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.x], -1),
        //    __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.y], -1),
        //    __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.z], -1),
        //    __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.w], -1), 1e-9, 0.2);

        temp = 1.0 / temp1;
    }
    else
    {
        temp = 1.0
               / edge_edge_ccd(vertexes[MMCVIDI.x],
                               vertexes[MMCVIDI.y],
                               vertexes[MMCVIDI.z],
                               vertexes[MMCVIDI.w],
                               __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.x], -1),
                               __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.y], -1),
                               __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.z], -1),
                               __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.w], -1),
                               CCDDistRatio,
                               0);
    }

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
        double tempMin = __shfl_down_sync(0xffffffff, temp, i);
        temp           = std::max(temp, tempMin);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            double tempMin = __shfl_down_sync(0xffffffff, temp, i);
            temp           = std::max(temp, tempMin);
        }
    }
    if(threadIdx.x == 0)
    {
        minStepSizes[blockIdx.x] = temp;
    }
}

__global__ void _reduct_max_cfl_to_double(const double3* moveDir,
                                          double*        max_double_val,
                                          uint32_t*      mSVI,
                                          int            number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= number)
        return;

    double temp = __GEIGEN__::__norm(moveDir[mSVI[idx]]);


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
        double tempMax = __shfl_down_sync(0xffffffff, temp, i);
        temp           = std::max(temp, tempMax);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            double tempMax = __shfl_down_sync(0xffffffff, temp, i);
            temp           = std::max(temp, tempMax);
        }
    }
    if(threadIdx.x == 0)
    {
        max_double_val[blockIdx.x] = temp;
    }
}

__global__ void _reduct_double3Sqn_to_double(const double3* A, double* D, int number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= number)
        return;

    double temp = __GEIGEN__::__squaredNorm(A[idx]);


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
        //double tempMax = __shfl_down_sync(0xffffffff, temp, i);
        temp += __shfl_down_sync(0xffffffff, temp, i);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        D[blockIdx.x] = temp;
    }
}

__global__ void _reduct_double3Dot_to_double(const double3* A, const double3* B, double* D, int number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= number)
        return;

    double temp = __GEIGEN__::__v_vec_dot(A[idx], B[idx]);


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
        //double tempMax = __shfl_down_sync(0xffffffff, temp, i);
        temp += __shfl_down_sync(0xffffffff, temp, i);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        D[blockIdx.x] = temp;
    }
}


__global__ void _getKineticEnergy_Reduction_3D(
    double3* _vertexes, double3* _xTilta, double* _energy, double* _masses, int number,
    double* penv = nullptr, const int* p2g = nullptr, int ng = 0)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= number)
        return;

    double temp =
        __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(_vertexes[idx], _xTilta[idx]))
        * _masses[idx] * 0.5;

    _penv_energy_accum(penv, p2g, idx, ng, temp);  // [S3] caller offsets p2g to FEM region

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
        temp += __shfl_down_sync(0xffffffff, temp, i);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        _energy[blockIdx.x] = temp;
    }
}

#ifdef USE_QUADRATIC_BENDING
__global__ void _getQuadBendingEnergy_Reduction(double*        squeue,
                                                const double3* vertexes,
                                                const double3* rest_vertexex,
                                                const uint2*   edges,
                                                const uint2*   edge_adj_vertex,
                                                const Eigen::Matrix4d* quad_bending_Q,
                                                int    edgesNum,
                                                double bendStiff,
                                                double* penv = nullptr, const int* p2g = nullptr, int ng = 0)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    int                      numbers = edgesNum;
    if(idx >= numbers)
        return;

    uint2  adj  = edge_adj_vertex[idx];
    double temp = __cal_quad_bending_energy(
        vertexes, rest_vertexex, edges[idx], adj, quad_bending_Q[idx], bendStiff);

    _penv_energy_accum(penv, p2g, edges[idx].x, ng, temp);  // [S3] quad bending edge env

    int    warpTid = threadIdx.x % 32;
    int    warpId  = (threadIdx.x >> 5);
    double nextTp;
    int    warpNum;
    if(blockIdx.x == gridDim.x - 1)
    {
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        temp += __shfl_down_sync(0xffffffff, temp, i);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        temp = tep[threadIdx.x];
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        squeue[blockIdx.x] = temp;
    }
}
#endif

__global__ void _getBendingEnergy_Reduction(double*        squeue,
                                            const double3* vertexes,
                                            const double3* rest_vertexex,
                                            const uint2*   edges,
                                            const uint2*   edge_adj_vertex,
                                            int            edgesNum,
                                            double         bendStiff,
                                            double* penv = nullptr, const int* p2g = nullptr, int ng = 0)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    int                      numbers = edgesNum;
    if(idx >= numbers)
        return;

    //double temp = __cal_BaraffWitkinStretch_energy(vertexes, triangles[idx], triDmInverses[idx], area[idx], stretchStiff, shearStiff);
    // double temp = __cal_hc_cloth_energy(vertexes, triangles[idx], triDmInverses[idx], area[idx], stretchStiff, shearStiff);
    uint2   adj     = edge_adj_vertex[idx];
    double3 rest_x0 = rest_vertexex[edges[idx].x];
    double3 rest_x1 = rest_vertexex[edges[idx].y];
    double  length  = __GEIGEN__::__norm(__GEIGEN__::__minus(rest_x0, rest_x1));
    double  temp =
        __cal_bending_energy(vertexes, rest_vertexex, edges[idx], adj, length, bendStiff);
    _penv_energy_accum(penv, p2g, edges[idx].x, ng, temp);  // [S3] bending edge's env
    //double temp = 0;
    //printf("%f    %f\n\n\n", lenRate, volRate);
    int    warpTid = threadIdx.x % 32;
    int    warpId  = (threadIdx.x >> 5);
    double nextTp;
    int    warpNum;
    //int tidNum = 32;
    if(blockIdx.x == gridDim.x - 1)
    {
        //tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        temp += __shfl_down_sync(0xffffffff, temp, i);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];
        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        squeue[blockIdx.x] = temp;
    }
}


__global__ void _getFEMEnergy_Reduction_3D(double*        squeue,
                                           const double3* vertexes,
                                           const uint4*   tetrahedras,
                                           const __GEIGEN__::Matrix3x3d* DmInverses,
                                           const double* volume,
                                           int           tetrahedraNum,
                                           double*       lenRate,
                                           double*       volRate,
                                           double* penv = nullptr, const int* p2g = nullptr, int ng = 0)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    int                      numbers = tetrahedraNum;
    if(idx >= numbers)
        return;

#ifdef USE_SNK1
    double temp = __cal_StabbleNHK_energy1_3D(
        vertexes, tetrahedras[idx], DmInverses[idx], volume[idx], lenRate[idx], volRate[idx]);
#elif USE_SNK2
    double temp = __cal_StabbleNHK_energy2_3D(
        vertexes, tetrahedras[idx], DmInverses[idx], volume[idx], lenRate[idx], volRate[idx]);
#else
    double temp = __cal_ARAP_energy_3D(
        vertexes, tetrahedras[idx], DmInverses[idx], volume[idx], lenRate[idx]);
#endif

    _penv_energy_accum(penv, p2g, tetrahedras[idx].x, ng, temp);  // [S3] tet's env

    //printf("%f    %f\n\n\n", lenRate, volRate);
    int    warpTid = threadIdx.x % 32;
    int    warpId  = (threadIdx.x >> 5);
    double nextTp;
    int    warpNum;
    //int tidNum = 32;
    if(blockIdx.x == gridDim.x - 1)
    {
        //tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        temp += __shfl_down_sync(0xffffffff, temp, i);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];
        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        squeue[blockIdx.x] = temp;
    }
}
__global__ void _computeSoftConstraintEnergy_Reduction(double*        squeue,
                                                       const double3* vertexes,
                                                       const double3* targetVert,
                                                       const uint32_t* targetInd,
                                                       double motionRate,
                                                       double rate,
                                                       const int*     stitch_paired_vertex,
                                                       const double3* stitch_rest_offset,
                                                       int    number,
                                                       double* penv = nullptr, const int* p2g = nullptr, int ng = 0)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= number)
        return;
    uint32_t vInd = targetInd[idx];
    double3 target;
    if(stitch_paired_vertex && stitch_paired_vertex[idx] >= 0)
    {
        int abd_idx = stitch_paired_vertex[idx];
        target = make_double3(
            vertexes[abd_idx].x + stitch_rest_offset[idx].x,
            vertexes[abd_idx].y + stitch_rest_offset[idx].y,
            vertexes[abd_idx].z + stitch_rest_offset[idx].z);
    }
    else
    {
        target = targetVert[idx];
    }
    double   dis  = __GEIGEN__::__squaredNorm(__GEIGEN__::__s_vec_multiply(
        __GEIGEN__::__minus(vertexes[vInd], target), rate));
    double   d    = motionRate;
    double   temp = d * dis * 0.5;

    _penv_energy_accum(penv, p2g, vInd, ng, temp);  // [S3] soft-constraint vertex env

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
        temp += __shfl_down_sync(0xffffffff, temp, i);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        squeue[blockIdx.x] = temp;
    }
}
__global__ void _get_triangleFEMEnergy_Reduction_3D(double*        squeue,
                                                    const double3* vertexes,
                                                    const uint3*   triangles,
                                                    const __GEIGEN__::Matrix2x2d* triDmInverses,
                                                    const double* area,
                                                    int           trianglesNum,
                                                    double        stretchStiff,
                                                    double        shearStiff,
                                                    double        strainRate,
                                                    double* penv = nullptr, const int* p2g = nullptr, int ng = 0)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    int                      numbers = trianglesNum;
    if(idx >= numbers)
        return;

    double temp = __cal_BaraffWitkinStretch_energy(
        vertexes, triangles[idx], triDmInverses[idx], area[idx], stretchStiff, shearStiff, strainRate);

    _penv_energy_accum(penv, p2g, triangles[idx].x, ng, temp);  // [S3] triangle's env

    //printf("%f    %f\n\n\n", lenRate, volRate);
    int    warpTid = threadIdx.x % 32;
    int    warpId  = (threadIdx.x >> 5);
    double nextTp;
    int    warpNum;
    //int tidNum = 32;
    if(blockIdx.x == gridDim.x - 1)
    {
        //tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        temp += __shfl_down_sync(0xffffffff, temp, i);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];
        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        squeue[blockIdx.x] = temp;
    }
}
__global__ void _getRestStableNHKEnergy_Reduction_3D(double*       squeue,
                                                     const double* volume,
                                                     int    tetrahedraNum,
                                                     double lenRate,
                                                     double volRate)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    int                      numbers = tetrahedraNum;
    if(idx >= numbers)
        return;

    double temp = ((0.5 * volRate * (3 * lenRate / 4 / volRate) * (3 * lenRate / 4 / volRate)
                    - 0.5 * lenRate * log(4.0)))
                  * volume[idx];

    int    warpTid = threadIdx.x % 32;
    int    warpId  = (threadIdx.x >> 5);
    double nextTp;
    int    warpNum;
    //int tidNum = 32;
    if(blockIdx.x == gridDim.x - 1)
    {
        //tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        temp += __shfl_down_sync(0xffffffff, temp, i);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];
        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        squeue[blockIdx.x] = temp;
    }
}

__global__ void _getBarrierEnergy_Reduction_3D(double*        squeue,
                                               const double3* vertexes,
                                               const double3* rest_vertexes,
                                               int4*          _collisionPair,
                                               double         _Kappa,
                                               double         _dHat,
                                               int            cpNum,
                                               double* penv = nullptr, const int* p2g = nullptr, int ng = 0)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    int                      numbers = cpNum;
    if(idx >= numbers)
        return;

    double temp =
        __cal_Barrier_energy(vertexes, rest_vertexes, _collisionPair[idx], _Kappa, _dHat);

    { int v0 = _collisionPair[idx].x; if(v0 < 0) v0 = -v0 - 1;     // [S3] pair's env
      _penv_energy_accum(penv, p2g, v0, ng, temp); }

    int    warpTid = threadIdx.x % 32;
    int    warpId  = (threadIdx.x >> 5);
    double nextTp;
    int    warpNum;
    //int tidNum = 32;
    if(blockIdx.x == gridDim.x - 1)
    {
        //tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        temp += __shfl_down_sync(0xffffffff, temp, i);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];
        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        squeue[blockIdx.x] = temp;
    }
}

__global__ void _getDeltaEnergy_Reduction(double* squeue, const double3* b, const double3* dx, int vertexNum)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    int                      numbers = vertexNum;
    if(idx >= numbers)
        return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);

    double temp = __GEIGEN__::__v_vec_dot(b[idx], dx[idx]);

    int    warpTid = threadIdx.x % 32;
    int    warpId  = (threadIdx.x >> 5);
    double nextTp;
    int    warpNum;
    //int tidNum = 32;
    if(blockIdx.x == gridDim.x - 1)
    {
        //tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        temp += __shfl_down_sync(0xffffffff, temp, i);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];
        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        squeue[blockIdx.x] = temp;
    }
}

__global__ void __add_reduction(double* mem, int numbers)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= numbers)
        return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    double temp = mem[idx];

    __threadfence();

    int    warpTid = threadIdx.x % 32;
    int    warpId  = (threadIdx.x >> 5);
    double nextTp;
    int    warpNum;
    //int tidNum = 32;
    if(blockIdx.x == gridDim.x - 1)
    {
        //tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        temp += __shfl_down_sync(0xffffffff, temp, i);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];
        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        mem[blockIdx.x] = temp;
    }
}

__global__ void _stepForward(double3* _vertexes,
                             double3* _vertexesTemp,
                             double3* _moveDir,
                             int*     bType,
                             double   alpha,
                             bool     moveBoundary,
                             int      numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    if(abs(bType[idx]) == 0 || moveBoundary)
    {
        _vertexes[idx] =
            __GEIGEN__::__minus(_vertexesTemp[idx],
                                __GEIGEN__::__s_vec_multiply(_moveDir[idx], alpha));
    }
}

// [multi-env S2] per-env FEM step: vertex idx moves by env_alpha[p2g[idx]] instead
// of a global scalar (env g's verts step uniformly with env g's ABD bodies).
// p2g and env_alpha are indexed in the SAME local frame as _vertexes here (caller
// passes p2g already offset to the FEM region). Falls back to `alpha` if group<0.
__global__ void _stepForward_perenv(double3*       _vertexes,
                                    const double3* _vertexesTemp,
                                    const double3* _moveDir,
                                    const int*     bType,
                                    const int*     p2g,
                                    const double*  env_alpha,
                                    double         alpha,
                                    bool           moveBoundary,
                                    int            numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers) return;
    if(abs(bType[idx]) == 0 || moveBoundary)
    {
        int    g = p2g[idx];
        double a = (g >= 0 && env_alpha[g] >= 0.0) ? env_alpha[g] : alpha;
        _vertexes[idx] =
            __GEIGEN__::__minus(_vertexesTemp[idx],
                                __GEIGEN__::__s_vec_multiply(_moveDir[idx], a));
    }
}

// [multi-env S2] gather per-ABD-body alpha: body b (0..abd_body_num) belongs to
// collision body b -> group body_to_group[b] -> env_alpha[group]. -1 if ungrouped.
__global__ void _gather_abd_body_alpha(const int* body_to_group, const double* env_alpha,
                                       double* abd_body_alpha, int abd_body_num, int ng)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if(b >= abd_body_num) return;
    int g = body_to_group[b];
    abd_body_alpha[b] = (g >= 0 && g < ng) ? env_alpha[g] : -1.0;
}

__global__ void _updateVelocities(double3* _vertexes,
                                  double3* _o_vertexes,
                                  double3* _velocities,
                                  int*     btype,
                                  double   ipc_dt,
                                  int      numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    if(btype[idx] == 0)
    {
        _velocities[idx] = __GEIGEN__::__s_vec_multiply(
            __GEIGEN__::__minus(_vertexes[idx], _o_vertexes[idx]), 1 / ipc_dt);
        //_velocities[idx] = make_double3(0, 0, 0);
        _o_vertexes[idx] = _vertexes[idx];
    }
    else
    {
        _velocities[idx] = make_double3(0, 0, 0);
        _o_vertexes[idx] = _vertexes[idx];
    }
}

__global__ void _updateBoundary(double3* _vertexes, int* _btype, double3* _moveDir, double ipc_dt, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    if((_btype[idx]) == -1 || (_btype[idx]) == 1)
    {
        _vertexes[idx] = __GEIGEN__::__add(_vertexes[idx], _moveDir[idx]);
    }
}

__global__ void _updateBoundary2(int* _btype, __GEIGEN__::Matrix3x3d* _constraints, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    if((_btype[idx]) == 1)
    {
        _btype[idx] = 0;
        __GEIGEN__::__set_Mat_val(_constraints[idx], 1, 0, 0, 0, 1, 0, 0, 0, 1);
    }
}


__global__ void _updateBoundaryMoveDir(double3* _vertexes,
                                       int*     _btype,
                                       double3* _moveDir,
                                       double   ipc_dt,
                                       double   PI,
                                       double   alpha,
                                       int      numbers,
                                       int      frameid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    double                 massSum = 0;
    double                 angleX  = PI / 2.5 * ipc_dt * alpha;
    __GEIGEN__::Matrix3x3d rotationL, rotationR;
    __GEIGEN__::__set_Mat_val(
        rotationL, 1, 0, 0, 0, cos(angleX), sin(angleX), 0, -sin(angleX), cos(angleX));
    __GEIGEN__::__set_Mat_val(
        rotationR, 1, 0, 0, 0, cos(angleX), -sin(angleX), 0, sin(angleX), cos(angleX));

    //_moveDir[idx] = make_double3(0, 0, 0);
    double mvl = -0.3 * ipc_dt * alpha;
    //if((_btype[idx]) == 1)
    //{
    //    _moveDir[idx] = make_double3(mvl, 0, 0);  //__GEIGEN__::__minus(__GEIGEN__::__M_v_multiply(rotationL, _vertexes[idx]), _vertexes[idx]);
    //}
    if((_btype[idx]) > 0)
    {
        if(frameid < 32)
        {
            if(_vertexes[idx].y > 0.01)
            {
                _moveDir[idx] = make_double3(0, -mvl, 0);
            }
            else if(_vertexes[idx].y < -0.01)
            {
                _moveDir[idx] = make_double3(0, mvl, 0);
            }
        }
        else
        {
            _moveDir[idx] = __GEIGEN__::__minus(
                __GEIGEN__::__M_v_multiply(rotationL, _vertexes[idx]), _vertexes[idx]);
        }
    }
    if((_btype[idx]) < 0)
    {
        if(frameid < 32)
        {
            if(_vertexes[idx].y > 0.01)
            {
                _moveDir[idx] = make_double3(0, -mvl, 0);
            }
            else if(_vertexes[idx].y < -0.01)
            {
                _moveDir[idx] = make_double3(0, mvl, 0);
            }
        }
        else
        {
            _moveDir[idx] = __GEIGEN__::__minus(
                __GEIGEN__::__M_v_multiply(rotationR, _vertexes[idx]), _vertexes[idx]);
        }
    }
}

__global__ void _computeXTilta(int*     _btype,
                               double3* _velocities,
                               double3* _o_vertexes,
                               double3* _xTilta,
                               int*     _apply_gravity,
                               double   ipc_dt,
                               double   rate,
                               double3  gravity_vec,
                               int      numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    double3 gravityDtSq = make_double3(0, 0, 0);
    if(_btype[idx] == 0 && _apply_gravity[idx])
    {
        gravityDtSq = __GEIGEN__::__s_vec_multiply(gravity_vec, ipc_dt * ipc_dt);
    }
    _xTilta[idx] = __GEIGEN__::__add(
        _o_vertexes[idx],
        __GEIGEN__::__add(__GEIGEN__::__s_vec_multiply(_velocities[idx], ipc_dt),
                          gravityDtSq));
}

__global__ void _updateSurfaces(uint32_t* sortIndex, uint3* _faces, int _offset_num, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    if(_faces[idx].x < _offset_num)
    {
        _faces[idx].x = sortIndex[_faces[idx].x];
    }
    else
    {
        _faces[idx].x = _faces[idx].x;
    }
    if(_faces[idx].y < _offset_num)
    {
        _faces[idx].y = sortIndex[_faces[idx].y];
    }
    else
    {
        _faces[idx].y = _faces[idx].y;
    }
    if(_faces[idx].z < _offset_num)
    {
        _faces[idx].z = sortIndex[_faces[idx].z];
    }
    else
    {
        _faces[idx].z = _faces[idx].z;
    }
    //printf("sorted face: %d  %d  %d\n", _faces[idx].x, _faces[idx].y, _faces[idx].z);
}

__global__ void _updateNeighborNum(unsigned int*   _neighborNumInit,
                                   unsigned int*   _neighborNum,
                                   const uint32_t* sortMapVertIndex,
                                   int             numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    _neighborNum[idx] = _neighborNumInit[sortMapVertIndex[idx]];
}

__global__ void _updateNeighborList(unsigned int*   _neighborListInit,
                                    unsigned int*   _neighborList,
                                    unsigned int*   _neighborNum,
                                    unsigned int*   _neighborStart,
                                    unsigned int*   _neighborStartTemp,
                                    const uint32_t* sortIndex,
                                    const uint32_t* sortMapVertIndex,
                                    int             numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    int startId   = _neighborStartTemp[idx];
    int o_startId = _neighborStart[sortIndex[idx]];
    int neiNum    = _neighborNum[idx];
    for(int i = 0; i < neiNum; i++)
    {
        _neighborList[startId + i] = sortMapVertIndex[_neighborListInit[o_startId + i]];
    }
    //_neighborStart[sortMapVertIndex[idx]] = startId;
    //_neighborNum[idx] = _neighborNum[sortMapVertIndex[idx]];
}

__global__ void _updateEdges(uint32_t* sortIndex, uint2* _edges, int _offset_num, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    if(_edges[idx].x < _offset_num)
    {
        _edges[idx].x = sortIndex[_edges[idx].x];
    }
    else
    {
        _edges[idx].x = _edges[idx].x;
    }
    if(_edges[idx].y < _offset_num)
    {
        _edges[idx].y = sortIndex[_edges[idx].y];
    }
    else
    {
        _edges[idx].y = _edges[idx].y;
    }
}

__global__ void _updateTriEdges_adjVerts(
    uint32_t* sortIndex, uint2* _edges, uint2* _adj_verts, int _offset_num, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    if(_edges[idx].x < _offset_num)
    {
        _edges[idx].x = sortIndex[_edges[idx].x];
    }
    else
    {
        _edges[idx].x = _edges[idx].x;
    }
    if(_edges[idx].y < _offset_num)
    {
        _edges[idx].y = sortIndex[_edges[idx].y];
    }
    else
    {
        _edges[idx].y = _edges[idx].y;
    }


    if(_adj_verts[idx].x < _offset_num)
    {
        _adj_verts[idx].x = sortIndex[_adj_verts[idx].x];
    }
    else
    {
        _adj_verts[idx].x = _adj_verts[idx].x;
    }
    if(_adj_verts[idx].y < _offset_num)
    {
        _adj_verts[idx].y = sortIndex[_adj_verts[idx].y];
    }
    else
    {
        _adj_verts[idx].y = _adj_verts[idx].y;
    }
}

__global__ void _updateSurfVerts(uint32_t* sortIndex, uint32_t* _sVerts, int _offset_num, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    if(_sVerts[idx] < _offset_num)
    {
        _sVerts[idx] = sortIndex[_sVerts[idx]];
    }
    else
    {
        _sVerts[idx] = _sVerts[idx];
    }
}

// Check if collision between bodyA and bodyB should be skipped
// according to the collision exclusion matrix.
//
// [multi-FEM-bodyid] Previously mapped body_id == -1 (legacy FEM sentinel)
// to the last matrix slot. Now every body has its real body_id and indexes
// the matrix directly.
__device__ inline bool _is_collision_excluded_gipc(int bodyA, int bodyB,
                                                   const int* _collision_skip_matrix,
                                                   int _collision_body_count)
{
    if(_collision_skip_matrix == nullptr || _collision_body_count <= 0)
        return false;
    if(bodyA < 0 || bodyB < 0 || bodyA >= _collision_body_count || bodyB >= _collision_body_count)
        return false;
    return _collision_skip_matrix[bodyA * _collision_body_count + bodyB] != 0;
}

// [multi-FEM-bodyid] Skip same-body filter for sanity-check kernels
// (segment-triangle intersection). Mirrors mlbvh.cu's _should_check_pair
// but inverted: returns TRUE if the pair should be SKIPPED (same ABD body,
// no point checking).
__device__ inline bool _skip_same_abd_body(int bodyA, int bodyB,
                                           const int* _body_id_to_is_fem)
{
    if(bodyA != bodyB) return false;        // different bodies -> not skipped here
    if(bodyA < 0) return false;             // unassigned -> defensive (don't skip)
    if(_body_id_to_is_fem == nullptr) return true;  // legacy fallback: skip same body
    return _body_id_to_is_fem[bodyA] == 0;  // skip if same ABD body
}

__global__ void _edgeTriIntersectionQuery(const int*     _bodyId,
                                          const int*     _btype,
                                          const double3* _vertexes,
                                          const uint2*   _edges,
                                          const uint3*   _faces,
                                          const AABB*    _edge_bvs,
                                          const Node*    _edge_nodes,
                                          int*           _isIntesect,
                                          double         dHat,
                                          int            number,
                                          const int*     _collision_skip_matrix,
                                          int            _collision_body_count,
                                          const int*     _body_id_to_is_fem)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    uint32_t  stack[64];
    uint32_t* stack_ptr = stack;
    *stack_ptr++        = 0;

    uint3 face = _faces[idx];
    //idx = idx + number - 1;


    AABB _bv;

    double3 _v = _vertexes[face.x];
    _bv.combines(_v.x, _v.y, _v.z);
    _v = _vertexes[face.y];
    _bv.combines(_v.x, _v.y, _v.z);
    _v = _vertexes[face.z];
    _bv.combines(_v.x, _v.y, _v.z);

    //uint32_t self_eid = _edge_nodes[idx].element_idx;
    //double bboxDiagSize2 = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(_edge_bvs[0].upper, _edge_bvs[0].lower));
    //printf("%f\n", bboxDiagSize2);
    double gapl = 0;  //sqrt(dHat);
    //double dHat = gapl * gapl;// *bboxDiagSize2;
    unsigned int num_found = 0;
    do
    {
        const uint32_t node_id = *--stack_ptr;
        const uint32_t L_idx   = _edge_nodes[node_id].left_idx;
        const uint32_t R_idx   = _edge_nodes[node_id].right_idx;

        if(_overlap(_bv, _edge_bvs[L_idx], gapl))
        {
            const auto obj_idx = _edge_nodes[L_idx].element_idx;
            if(obj_idx != 0xFFFFFFFF)
            {
                if(!(face.x == _edges[obj_idx].x || face.x == _edges[obj_idx].y
                     || face.y == _edges[obj_idx].x || face.y == _edges[obj_idx].y
                     || face.z == _edges[obj_idx].x || face.z == _edges[obj_idx].y))
                {
                    // [multi-FEM-bodyid] Skip if face and edge belong to the same
                    // ABD body. FEM body self-intersection sanity-check is still
                    // run (allowed) since the FEM mesh might fold onto itself.
                    if(_skip_same_abd_body(_bodyId[face.x], _bodyId[_edges[obj_idx].x],
                                           _body_id_to_is_fem))
                    {
                        // same ABD body, skip
                    }
                    // Skip if bodies are in the collision exclusion list
                    else if(_is_collision_excluded_gipc(_bodyId[face.x], _bodyId[_edges[obj_idx].x],
                                                       _collision_skip_matrix, _collision_body_count))
                    {
                        // excluded body pair, skip
                    }
                    else if(!(_btype[face.x] >= 2 && _btype[face.y] >= 2
                         && _btype[face.z] >= 2 && _btype[_edges[obj_idx].x] >= 2
                         && _btype[_edges[obj_idx].y] >= 2))
                        if(segTriIntersect(_vertexes[_edges[obj_idx].x],
                                           _vertexes[_edges[obj_idx].y],
                                           _vertexes[face.x],
                                           _vertexes[face.y],
                                           _vertexes[face.z]))
                        {
                            *_isIntesect = -1;
                            printf("[INTERSECT-L] tri(%d,%d,%d) body=%d  edge(%d,%d) body=%d\n",
                                   face.x, face.y, face.z, _bodyId[face.x],
                                   _edges[obj_idx].x, _edges[obj_idx].y, _bodyId[_edges[obj_idx].x]);
                            return;
                        }
                }
            }
            else  // the node is not a leaf.
            {
                *stack_ptr++ = L_idx;
            }
        }
        if(_overlap(_bv, _edge_bvs[R_idx], gapl))
        {
            const auto obj_idx = _edge_nodes[R_idx].element_idx;
            if(obj_idx != 0xFFFFFFFF)
            {
                if(!(face.x == _edges[obj_idx].x || face.x == _edges[obj_idx].y
                     || face.y == _edges[obj_idx].x || face.y == _edges[obj_idx].y
                     || face.z == _edges[obj_idx].x || face.z == _edges[obj_idx].y))
                {
                    // [multi-FEM-bodyid] Skip if face and edge belong to the same
                    // ABD body. FEM body self-intersection sanity-check is still
                    // run (allowed) since the FEM mesh might fold onto itself.
                    if(_skip_same_abd_body(_bodyId[face.x], _bodyId[_edges[obj_idx].x],
                                           _body_id_to_is_fem))
                    {
                        // same ABD body, skip
                    }
                    // Skip if bodies are in the collision exclusion list
                    else if(_is_collision_excluded_gipc(_bodyId[face.x], _bodyId[_edges[obj_idx].x],
                                                       _collision_skip_matrix, _collision_body_count))
                    {
                        // excluded body pair, skip
                    }
                    else if(!(_btype[face.x] >= 2 && _btype[face.y] >= 2
                         && _btype[face.z] >= 2 && _btype[_edges[obj_idx].x] >= 2
                         && _btype[_edges[obj_idx].y] >= 2))
                        if(segTriIntersect(_vertexes[_edges[obj_idx].x],
                                           _vertexes[_edges[obj_idx].y],
                                           _vertexes[face.x],
                                           _vertexes[face.y],
                                           _vertexes[face.z]))
                        {
                            *_isIntesect = -1;
                            printf("[INTERSECT-R] tri(%d,%d,%d) body=%d  edge(%d,%d) body=%d\n",
                                   face.x, face.y, face.z, _bodyId[face.x],
                                   _edges[obj_idx].x, _edges[obj_idx].y, _bodyId[_edges[obj_idx].x]);
                            return;
                        }
                }
            }
            else  // the node is not a leaf.
            {
                *stack_ptr++ = R_idx;
            }
        }
    } while(stack < stack_ptr);
}

__global__ void _calFrictionLastH_gd(const double3* _vertexes,
                                     const double*  g_offset,
                                     const double3* g_normal,
                                     const const uint32_t* _collisionPair_environment,
                                     double*   lambda_lastH_gd,
                                     uint32_t* _collisionPair_last_gd,
                                     double    dHat,
                                     double    Kappa,
                                     int       number,
                                     const double* kappa_grp = nullptr,
                                     const int*    p2g = nullptr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    double3 normal = *g_normal;
    int     gidx   = _collisionPair_environment[idx];
    double  dist = __GEIGEN__::__v_vec_dot(normal, _vertexes[gidx]) - *g_offset;
    double  dist2 = dist * dist;
    dist2 = (dist2 == 0.0 ? 1e-12 : dist2);  // [d=0 guard] avoid ground-friction lambda NaN at d=0

    double t   = dist2 - dHat;
    double g_b = t * log(dist2 / dHat) * -2.0 - (t * t) / dist2;

    // [decouple] per-group κ so env0's friction normal-force is batch-invariant (global Kappa is a
    // reduction over ALL envs ⇒ batch-dependent; friction Hessian ∝ λ exposes it even at zero sliding).
    double Kp = (kappa_grp && p2g && p2g[gidx] >= 0) ? kappa_grp[p2g[gidx]] : Kappa;  /* [-1 guard] */
    lambda_lastH_gd[idx]        = -Kp * 2.0 * sqrt(dist2) * g_b;
    _collisionPair_last_gd[idx] = gidx;
}

__global__ void _calFrictionLastH_DistAndTan(const double3*    _vertexes,
                                             const const int4* _collisionPair,
                                             double*           lambda_lastH,
                                             double2*          distCoord,
                                             __GEIGEN__::Matrix3x2d* tanBasis,
                                             int4*     _collisionPair_last,
                                             double    dHat,
                                             double    Kappa,
                                             uint32_t* _cpNum_last,
                                             int       number,
                                             const double* kappa_grp = nullptr,
                                             const int*    p2g = nullptr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int4   MMCVIDI = _collisionPair[idx];
    double dis;
    int    last_index = -1;
    // [decouple] per-group κ for the lagged friction normal-force λ (batch-invariant). gv = pair's
    // representative vertex (same convention as the barrier, GIPC.cu:3250).
    double Kappa_eff = Kappa;
    if(kappa_grp && p2g)
    { int gv = (MMCVIDI.x >= 0) ? MMCVIDI.x : (-MMCVIDI.x - 1); if(gv >= 0) { int _gg = p2g[gv]; if(_gg >= 0) Kappa_eff = kappa_grp[_gg]; } }  /* [-1 guard] */
    if(MMCVIDI.x >= 0)
    {
        if(MMCVIDI.w >= 0)
        {
            last_index = atomicAdd(_cpNum_last, 1);
            atomicAdd(_cpNum_last + 4, 1);
            _d_EE(_vertexes[MMCVIDI.x],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            Friction::computeClosestPoint_EE(_vertexes[MMCVIDI.x],
                                             _vertexes[MMCVIDI.y],
                                             _vertexes[MMCVIDI.z],
                                             _vertexes[MMCVIDI.w],
                                             distCoord[last_index]);
            Friction::computeTangentBasis_EE(_vertexes[MMCVIDI.x],
                                             _vertexes[MMCVIDI.y],
                                             _vertexes[MMCVIDI.z],
                                             _vertexes[MMCVIDI.w],
                                             tanBasis[last_index]);
        }
    }
    else
    {
        int v0I = -MMCVIDI.x - 1;
        if(MMCVIDI.z < 0)
        {
            if(MMCVIDI.y >= 0)
            {
                last_index = atomicAdd(_cpNum_last, 1);
                atomicAdd(_cpNum_last + 2, 1);
                _d_PP(_vertexes[v0I], _vertexes[MMCVIDI.y], dis);
                distCoord[last_index].x = 0;
                distCoord[last_index].y = 0;
                Friction::computeTangentBasis_PP(
                    _vertexes[v0I], _vertexes[MMCVIDI.y], tanBasis[last_index]);
            }
        }
        else if(MMCVIDI.w < 0)
        {
            if(MMCVIDI.y >= 0)
            {
                last_index = atomicAdd(_cpNum_last, 1);
                atomicAdd(_cpNum_last + 3, 1);
                _d_PE(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dis);
                Friction::computeClosestPoint_PE(_vertexes[v0I],
                                                 _vertexes[MMCVIDI.y],
                                                 _vertexes[MMCVIDI.z],
                                                 distCoord[last_index].x);
                distCoord[last_index].y = 0;
                Friction::computeTangentBasis_PE(_vertexes[v0I],
                                                 _vertexes[MMCVIDI.y],
                                                 _vertexes[MMCVIDI.z],
                                                 tanBasis[last_index]);
            }
        }
        else
        {
            last_index = atomicAdd(_cpNum_last, 1);
            atomicAdd(_cpNum_last + 4, 1);
            _d_PT(_vertexes[v0I],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            Friction::computeClosestPoint_PT(_vertexes[v0I],
                                             _vertexes[MMCVIDI.y],
                                             _vertexes[MMCVIDI.z],
                                             _vertexes[MMCVIDI.w],
                                             distCoord[last_index]);
            Friction::computeTangentBasis_PT(_vertexes[v0I],
                                             _vertexes[MMCVIDI.y],
                                             _vertexes[MMCVIDI.z],
                                             _vertexes[MMCVIDI.w],
                                             tanBasis[last_index]);
        }
    }
    if(last_index >= 0)
    {
//        double t = dis - dHat;
//        lambda_lastH[last_index] = -Kappa * 2.0 * std::sqrt(dis) * (t * std::log(dis / dHat) * -2.0 - (t * t) / dis);
#if (RANK == 1)
        double t = dis - dHat;
        lambda_lastH[last_index] =
            -Kappa_eff * 2.0 * sqrt(dis) * (t * log(dis / dHat) * -2.0 - (t * t) / dis);
#elif (RANK == 2)
        lambda_lastH[last_index] =
            -Kappa_eff * 2.0 * sqrt(dis)
            * (log(dis / dHat) * log(dis / dHat) * (2 * dis - 2 * dHat)
               + (2 * log(dis / dHat) * (dis - dHat) * (dis - dHat)) / dis);
#endif
        _collisionPair_last[last_index] = _collisionPair[idx];
    }
}

/// <summary>
///  host code
/// </summary>
void GIPC::FREE_DEVICE_MEM()
{
    CUDA_SAFE_CALL(cudaFree(_MatIndex));
    if(m_reduce_scratch) { CUDA_SAFE_CALL(cudaFree(m_reduce_scratch)); m_reduce_scratch=nullptr; m_reduce_cap=0; }
    CUDA_SAFE_CALL(cudaFree(_collisonPairs));
    CUDA_SAFE_CALL(cudaFree(_ccd_collisonPairs));
    CUDA_SAFE_CALL(cudaFree(_cpNum));
    CUDA_SAFE_CALL(cudaFree(_close_cpNum));
    CUDA_SAFE_CALL(cudaFree(_close_gpNum));
    CUDA_SAFE_CALL(cudaFree(_environment_collisionPair));
    // [9d28824-port] _gpNum aliases (_cpNum + 5) — freed above with _cpNum.
    _gpNum = nullptr;
    CUDA_SAFE_CALL(cudaFree(_groundNormal));
    CUDA_SAFE_CALL(cudaFree(_groundOffset));

    CUDA_SAFE_CALL(cudaFree(_faces));
    CUDA_SAFE_CALL(cudaFree(_edges));
    CUDA_SAFE_CALL(cudaFree(_surfVerts));

    // [multi-env S1] free per-env line-search substrate
    if(m_env_alpha)    { CUDA_SAFE_CALL(cudaFree(m_env_alpha));    m_env_alpha    = nullptr; }
    if(m_env_scratch)  { CUDA_SAFE_CALL(cudaFree(m_env_scratch));  m_env_scratch  = nullptr; }
    if(m_abd_body_alpha) { CUDA_SAFE_CALL(cudaFree(m_abd_body_alpha)); m_abd_body_alpha = nullptr; }
    if(m_env_active)   { CUDA_SAFE_CALL(cudaFree(m_env_active));   m_env_active   = nullptr; }

    // [0be8da3-port] free the persistent (grow-only) friction/close buffers and
    // reset capacities so engine.reset() starts clean.
    if(lambda_lastH_scalar)
    {
        CUDA_SAFE_CALL(cudaFree(lambda_lastH_scalar));
        CUDA_SAFE_CALL(cudaFree(distCoord));
        CUDA_SAFE_CALL(cudaFree(tanBasis));
        CUDA_SAFE_CALL(cudaFree(_collisonPairs_lastH));
        CUDA_SAFE_CALL(cudaFree(_MatIndex_last));
        lambda_lastH_scalar = nullptr; distCoord = nullptr; tanBasis = nullptr;
        _collisonPairs_lastH = nullptr; _MatIndex_last = nullptr;
    }
    if(lambda_lastH_scalar_gd)
    {
        CUDA_SAFE_CALL(cudaFree(lambda_lastH_scalar_gd));
        CUDA_SAFE_CALL(cudaFree(_collisonPairs_lastH_gd));
        lambda_lastH_scalar_gd = nullptr; _collisonPairs_lastH_gd = nullptr;
    }
    if(_closeConstraintID)
    {
        CUDA_SAFE_CALL(cudaFree(_closeConstraintID));
        CUDA_SAFE_CALL(cudaFree(_closeConstraintVal));
        _closeConstraintID = nullptr; _closeConstraintVal = nullptr;
    }
    if(_closeMConstraintID)
    {
        CUDA_SAFE_CALL(cudaFree(_closeMConstraintID));
        CUDA_SAFE_CALL(cudaFree(_closeMConstraintVal));
        _closeMConstraintID = nullptr; _closeMConstraintVal = nullptr;
    }
    m_fric_cp_cap = 0; m_fric_gd_cap = 0; m_close_gp_cap = 0; m_close_cp_cap = 0;

    // ②-D2H: free energy slots
    if(m_energy_slots) { CUDA_SAFE_CALL(cudaFree(m_energy_slots)); m_energy_slots = nullptr; }
    if(m_alpha_slots)  { CUDA_SAFE_CALL(cudaFree(m_alpha_slots));  m_alpha_slots  = nullptr; }

    pcg_data.FREE_DEVICE_MEM();

    bvh_e.FREE_DEVICE_MEM();
    bvh_f.FREE_DEVICE_MEM();
}

void GIPC::MALLOC_DEVICE_MEM()
{
    // +1 trash slot: pair-emit overflow is redirected to index==cap (see _emit_slot
    // in mlbvh.cu) so detection never writes out of bounds; the host then grows.
    CUDA_SAFE_CALL(cudaMalloc((void**)&_MatIndex, ((size_t)MAX_COLLITION_PAIRS_NUM + 1) * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_collisonPairs,
                              ((size_t)MAX_COLLITION_PAIRS_NUM + 1) * sizeof(int4)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_ccd_collisonPairs,
                              ((size_t)MAX_CCD_COLLITION_PAIRS_NUM + 1) * sizeof(int4)));
    set_emit_caps(MAX_COLLITION_PAIRS_NUM, MAX_CCD_COLLITION_PAIRS_NUM);
    CUDA_SAFE_CALL(cudaMalloc((void**)&_environment_collisionPair,
                              surf_vertexNum * sizeof(int)));
    //CUDA_SAFE_CALL(cudaMalloc((void**)&_moveDir, vertexNum * sizeof(double3)));
    // [9d28824-port] one contiguous [6]-uint32 block: _cpNum aliases [0:5],
    // _gpNum aliases [5]. Kernel-side code unchanged (takes uint32_t*); the
    // paired cpNum+gpNum reads become ONE 6-int D2H.
    CUDA_SAFE_CALL(cudaMalloc((void**)&_cpNum, 6 * sizeof(uint32_t)));
    _gpNum = _cpNum + 5;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_groundNormal, 5 * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_groundOffset, 5 * sizeof(double)));
    double  h_offset[5] = {ground_offset_cfg, -1, 1, -1, 1};
    double3 H_normal[5];
    H_normal[0] = ground_normal_cfg;
    H_normal[1] = make_double3(1, 0, 0);
    H_normal[2] = make_double3(-1, 0, 0);
    H_normal[3] = make_double3(0, 0, 1);
    H_normal[4] = make_double3(0, 0, -1);
    CUDA_SAFE_CALL(cudaMemcpy(_groundOffset, &h_offset, 5 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(_groundNormal, &H_normal, 5 * sizeof(double3), cudaMemcpyHostToDevice));


    CUDA_SAFE_CALL(cudaMalloc((void**)&_faces, surface_Num * sizeof(uint3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_edges, edge_Num * sizeof(uint2)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_surfVerts, surf_vertexNum * sizeof(uint32_t)));

    CUDA_SAFE_CALL(cudaMalloc((void**)&_close_cpNum, sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_close_gpNum, sizeof(uint32_t)));

    // [multi-env S1] per-env feasible-alpha substrate (physics-neutral until S2).
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_env_alpha, kEnvAlphaSlots * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_env_scratch, 4 * kEnvAlphaSlots * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_env_active, kEnvAlphaSlots * sizeof(int)));
    h_env_alpha.assign(kEnvAlphaSlots, 1.0);
    h_env_active.assign(kEnvAlphaSlots, 1);
    // [batch-size hygiene] m_env_alpha starts at 1.0 like the host mirror: the device fast path
    // (2262b33) writes only PRESENT envs' slots — absent slots must not hold cudaMalloc garbage.
    CUDA_SAFE_CALL(cudaMemcpy(m_env_alpha, h_env_alpha.data(),
                              kEnvAlphaSlots * sizeof(double), cudaMemcpyHostToDevice));
    { std::vector<int> ones(kEnvAlphaSlots, 1);
      CUDA_SAFE_CALL(cudaMemcpy(m_env_active, ones.data(), kEnvAlphaSlots * sizeof(int), cudaMemcpyHostToDevice)); }

    // ②-D2H: 9-slot device buffer for batched energy reductions in computeEnergy.
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_energy_slots, kEnergySlotCount * sizeof(double)));
    // ②-D2H: 2-slot buffer for ground+self largestFeasibleStepSize batching.
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_alpha_slots, 2 * sizeof(double)));

    CUDA_SAFE_CALL(cudaMemset(_close_cpNum, 0, sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemset(_close_gpNum, 0, sizeof(uint32_t)));

    // [multi-env determinism] per-vertex env offset (0 by default = no-op) + BVH vertex buffer.
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_env_offset, vertexNum * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_bvh_vertexes, vertexNum * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMemset(d_env_offset, 0, vertexNum * sizeof(double3)));

    // [multi-env determinism 4.3] binned gradient accumulator (BINNED_K bins per vert*comp).
    CUDA_SAFE_CALL(cudaMalloc((void**)&g_grad_binned,
                              3 * (size_t)vertexNum * BINNED_K * sizeof(double)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_gbin, &g_grad_binned, sizeof(double*)));

    pcg_data.Malloc_DEVICE_MEM(vertexNum, tetrahedraNum);
}


void GIPC::initBVH(int* _btype, int* _bodyId, int* _collision_skip_matrix, int _collision_body_count)
{

    bvh_e.init(_bodyId,
               _btype,
               d_bvh_vertexes,   // [multi-env] BVH on offset-separated verts; narrow-phase uses local _vertexes
               _rest_vertexes,
               _edges,
               _collisonPairs,
               _ccd_collisonPairs,
               _cpNum,
               _MatIndex,
               edge_Num,
               surf_vertexNum,
               _collision_skip_matrix,
               _collision_body_count);
    bvh_f.init(_bodyId,
               _btype,
               d_bvh_vertexes,   // [multi-env] BVH on offset-separated verts; narrow-phase uses local _vertexes
               _faces,
               _surfVerts,
               _collisonPairs,
               _ccd_collisonPairs,
               _cpNum,
               _MatIndex,
               surface_Num,
               surf_vertexNum,
               _collision_skip_matrix,
               _collision_body_count);
    // [multi-FEM-bodyid] forward the per-body FEM flag table directly
    // (set by sim_engine.cu after this call returns; see ipc._body_id_to_is_fem
    // assignment in do_init_bvh_and_solver).
    bvh_e._body_id_to_is_fem = _body_id_to_is_fem;
    bvh_f._body_id_to_is_fem = _body_id_to_is_fem;
}

void GIPC::init(double m_meanMass, double m_meanVolumn, double3 minConer, double3 maxConer, double buffScale)
{
    if(m_skip_all_collision)
    {
        SceneSize.upper = make_double3(maxConer.x, maxConer.y, maxConer.z);
        SceneSize.lower = make_double3(minConer.x, minConer.y, minConer.z);
    }
    else
    {
        SceneSize = bvh_f.scene;
    }
    bboxDiagSize2 = __GEIGEN__::__squaredNorm(
        __GEIGEN__::__minus(SceneSize.upper, SceneSize.lower));
    // [absolute-dhat fix] The scene-bbox diagonal grows with env count / spacing,
    // which inflates the bbox-derived dHat (contact thickness) — a physics bug
    // and the root cause of super-linear contact growth in multi-env. When
    // absolute_dhat>0, derive an EFFECTIVE bbox so dHat == absolute_dhat^2 and
    // dTol/fDhat stay consistent with a single-env scene of that contact scale.
    double eff_bboxDiagSize2 = bboxDiagSize2;
    if(absolute_dhat > 0.0 && relative_dhat > 0.0)
        eff_bboxDiagSize2 = (absolute_dhat * absolute_dhat)
                            / (relative_dhat * relative_dhat);
    dTol         = 1e-18 * eff_bboxDiagSize2;
    minKappaCoef = 1e11;
    meanMass     = m_meanMass;
    meanVolumn   = m_meanVolumn;
    dHat = relative_dhat * relative_dhat * eff_bboxDiagSize2;  // = absolute_dhat^2 when set
    fDhat = 1e-4 * eff_bboxDiagSize2;
    if(::g_gipc_log_level >= 1)
        printf("[dhat] bboxDiagSize2=%.6g (eff=%.6g)  relative_dhat=%.3g  abs_dhat=%.3g  dHat_sqrt=%.6g%s\n",
               bboxDiagSize2, eff_bboxDiagSize2, relative_dhat, absolute_dhat,
               sqrt(dHat), absolute_dhat > 0.0 ? " (ABSOLUTE)" : " (scene-bbox)");
    if(getenv("STIFF_SEED_DIAG"))
        printf("[seed-diag] bboxDiagSize2=%.17g eff=%.17g meanMass=%.17g meanVolumn=%.17g dHat=%.17g fDhat=%.17g dTol=%.17g scene=[%.17g,%.17g,%.17g]-[%.17g,%.17g,%.17g]\n",
               bboxDiagSize2, eff_bboxDiagSize2, meanMass, meanVolumn, dHat, fDhat, dTol,
               SceneSize.lower.x, SceneSize.lower.y, SceneSize.lower.z,
               SceneSize.upper.x, SceneSize.upper.y, SceneSize.upper.z);


    int global_matrix_block3_size =
        abd_fem_count_info.abd_body_num * 4 + abd_fem_count_info.fem_point_num;


    uint32_t Minimum = 100000 * buffScale;
    int minCollisionBuffer4 = std::max(2 * (surf_vertexNum + edge_Num), Minimum);
    int minCollisionBuffer3 = std::max(2 * (surf_vertexNum + edge_Num), Minimum);
    int minCollisionBuffer2 = std::max(2 * (surf_vertexNum + edge_Num), Minimum);
    int minCollisionBuffer1 = 2 * surf_vertexNum;

    long long unsigned total_internal_triplet_num =
        ((abd_fem_count_info.fem_tet_num + tri_edge_num) * 10 + triangleNum * 6)
        + softNum
        + abd_fem_count_info.abd_body_num * 10
        + num_joint_constraints * 16
        + static_cast<long long>(m_abd_system->m_num_revolute_driving) * 16
        + static_cast<long long>(m_abd_system->m_num_prismatic) * 16
        + static_cast<long long>(m_abd_system->m_num_prismatic_driving) * 16
        + static_cast<long long>(softNum) * 4;
    long long unsigned total_max_collision_triplet_num =
        minCollisionBuffer4 * 16 + minCollisionBuffer3 * 9
        + minCollisionBuffer2 * 4 + minCollisionBuffer1;
    // [Strategy D] M3.5 chain-rule kernel reserves an extension range past
    // the FEM triplets, with capacity = fem_triplet_num * 16 (worst-case
    // diff-body pin-pin expansion).  When rigid region is large (Strategy D
    // hybrid mesh), this 16× factor easily exceeds the previous 2×
    // allocation → CUDA illegal memory access.  Use 32× to give margin
    // (the actual ext_count is usually < 16× but allocation math conservative).
    long long unsigned total_max_global_triplet_num =
        total_internal_triplet_num
            * static_cast<long long unsigned>(m_triplet_internal_margin)
        + total_max_collision_triplet_num;
    // [P1-dyn] Non-hybrid scenes (margin forced to 1 by sim_engine when n_fem_pins==0)
    // size the triplet buffer per-step from ACTUAL contact counts instead of the
    // worst-case 2*(surf+edge)*29 (cp-stats: <1% used). Allocate a small initial buffer;
    // computeGradientAndHessian() grows it to 2*length each step (2x = converter's
    // documented [length:2*length) scratch region). Hybrid keeps the worst-case alloc.
    m_fixed_triplet_base = static_cast<long long>(total_internal_triplet_num);
    m_dynamic_triplet    = (m_triplet_internal_margin <= 1.0);
    long long unsigned init_total = total_internal_triplet_num
        + static_cast<long long unsigned>(abd_fem_count_info.fem_point_num)
        + 2u * static_cast<long long unsigned>(surf_vertexNum) + 100000u;
    long long unsigned triplet_alloc = m_dynamic_triplet
        ? (2u * init_total)            // 2x for the converter scratch/output region
        : (total_max_global_triplet_num * (long long unsigned)buffScale);
    long long unsigned hash_alloc = m_dynamic_triplet
        ? init_total
        : (long long unsigned)((total_internal_triplet_num + total_max_collision_triplet_num) * buffScale);
    if(::g_gipc_log_level >= 1) printf("[buffer] internal=%llu worst=%llu init_alloc=%llu (dynamic=%d, ~%llu MB)\n",
           total_internal_triplet_num, total_max_global_triplet_num, triplet_alloc,
           (int)m_dynamic_triplet, triplet_alloc * 80 / 1024 / 1024);

    gipc_global_triplet.init_var();

    gipc_global_triplet.resize(global_matrix_block3_size,
                               global_matrix_block3_size,
                               triplet_alloc);

    gipc_global_triplet.global_external_max_capcity = hash_alloc;
    gipc_global_triplet.resize_collision_hash_size(hash_alloc);


    m_global_linear_system->gipc_global_triplet = &(gipc_global_triplet);
    m_abd_system->global_triplet                = &(gipc_global_triplet);
    init_abd_system();
}

GIPC::~GIPC()
{
    if(m_aux_stream)
    {
        cudaStreamDestroy(m_aux_stream);
        m_aux_stream = nullptr;
    }
    FREE_DEVICE_MEM();
}

GIPC::GIPC()
{
    IPC_dt            = 0.01;
    animation_subRate = 1.0;
    animation         = false;

    h_cpNum_last[0] = 0;
    h_cpNum_last[1] = 0;
    h_cpNum_last[2] = 0;
    h_cpNum_last[3] = 0;
    h_cpNum_last[4] = 0;
}

static void _dbg_ksum(const char*, const void*, size_t);  // [4.3 fwd]
void GIPC::buildFrictionSets()
{
    CUDA_SAFE_CALL(cudaMemset(_cpNum, 0, 5 * sizeof(uint32_t)));
    int                numbers   = h_cpNum[0];
    if(getenv("STIFF_KSUM"))
    {
        cudaDeviceSynchronize();
        _dbg_ksum("prep_verts", _vertexes, (size_t)vertexNum * sizeof(double3));
    }
    const unsigned int threadNum = 256;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    if(numbers > 0)
    {
        // [multi-env determinism 4.3] zero distCoord first: PP (point-point) lagged pairs write
        // tanBasis but NOT distCoord (no barycentric coords), leaving GARBAGE in their slots.
        // Those slots sit at non-deterministic positions (atomicAdd last_index) → the lagged
        // friction data is non-deterministic run-to-run → the friction Hessian (frame 0) → the
        // whole solve. Zeroing makes PP distCoord deterministically 0 (the friction Hessian for
        // PP doesn't use it; EE/PE/PT overwrite it). THIS is the residual non-atomic source.
        CUDA_SAFE_CALL(cudaMemset(distCoord, 0, (size_t)h_cpNum[0] * sizeof(double2)));
        _calFrictionLastH_DistAndTan<<<blockNum, threadNum>>>(_vertexes,
                                                              _collisonPairs,
                                                              lambda_lastH_scalar,
                                                              distCoord,
                                                              tanBasis,
                                                              _collisonPairs_lastH,
                                                              dHat,
                                                              Kappa,
                                                              _cpNum,
                                                              h_cpNum[0],
                                                              m_pergroup_kappa ? m_kappa_group : nullptr,
                                                              m_pergroup_kappa ? m_d_p2g : nullptr);
    }
    CUDA_SAFE_CALL(cudaMemcpy(h_cpNum_last, _cpNum, 5 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    numbers = h_gpNum;
    if(numbers > 0)
    {

        blockNum = (numbers + threadNum - 1) / threadNum;
        _calFrictionLastH_gd<<<blockNum, threadNum>>>(_vertexes,
                                                      _groundOffset,
                                                      _groundNormal,
                                                      _environment_collisionPair,
                                                      lambda_lastH_scalar_gd,
                                                      _collisonPairs_lastH_gd,
                                                      dHat,
                                                      Kappa,
                                                      h_gpNum,
                                                      m_pergroup_kappa ? m_kappa_group : nullptr,
                                                      m_pergroup_kappa ? m_d_p2g : nullptr);
    }
    h_gpNum_last = h_gpNum;
}


void GIPC::GroundCollisionDetect()
{
    int numbers = surf_vertexNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _GroundCollisionDetect<<<blockNum, threadNum>>>(
        _vertexes, _surfVerts, _groundOffset, _groundNormal, _environment_collisionPair, _gpNum, dHat, numbers,
        _point_body_id, _ground_skip_body, _ground_body_count);
}

void GIPC::computeSoftConstraintGradientAndHessian(double3* _gradient, int global_hessian_fem_offset)
{
    int numbers = softNum;
    if(numbers < 1)
    {
        return;
    }
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    // offset
    _computeSoftConstraintGradientAndHessian<<<blockNum, threadNum>>>(
        _vertexes,
        targetVert,
        targetInd,
        _gradient,
        _gpNum,
        gipc_global_triplet.block_values(),
        gipc_global_triplet.block_row_indices(),
        gipc_global_triplet.block_col_indices(),
        softMotionRate,
        animation_fullRate,
        gipc_global_triplet.global_triplet_offset,
        global_hessian_fem_offset,
        m_d_stitch_paired_vertex,
        m_d_stitch_rest_offset,
        m_d_stitch_abd_body_id,
        reinterpret_cast<const __GEIGEN__::Vector12*>(m_d_abd_body_q),
        softNum);
}

void GIPC::getTotalForce(double3* _gradient0, double3* _gradient1)
{

    int numbers = vertexNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _getTotalForce<<<blockNum, threadNum>>>(_gradient0, _gradient1, numbers);
}


void GIPC::computeGroundGradientAndHessian(double3* _gradient)
{
#ifndef USE_FRICTION
    CUDA_SAFE_CALL(cudaMemset(_gpNum, 0, sizeof(uint32_t)));
#endif
    int numbers = h_gpNum;
    if(numbers < 1)
    {
        return;
    }
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _computeGroundGradientAndHessian<<<blockNum, threadNum>>>(
        _vertexes,
        _groundOffset,
        _groundNormal,
        _environment_collisionPair,
        _gradient,
        _gpNum,
        gipc_global_triplet.block_values(),
        gipc_global_triplet.block_row_indices(),
        gipc_global_triplet.block_col_indices(),
        dHat,
        Kappa,
        gipc_global_triplet.global_triplet_offset,
        numbers,
        m_pergroup_kappa ? m_kappa_group : nullptr,
        m_pergroup_kappa ? m_d_p2g : nullptr);
}

void GIPC::computeCloseGroundVal()
{
    int numbers = h_gpNum;
    if(h_gpNum <= 0)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _computeGroundCloseVal<<<blockNum, threadNum>>>(_vertexes,
                                                    _groundOffset,
                                                    _groundNormal,
                                                    _environment_collisionPair,
                                                    dTol,
                                                    _closeConstraintID,
                                                    _closeConstraintVal,
                                                    _close_gpNum,
                                                    numbers);
    // NOTE: h_close_gpNum is intentionally NOT synced from device here.
    // The adaptive Kappa doubling path (checkCloseGroundVal) that depends on it
    // was never functional in the original code and enabling it causes instability.
    // Fixing this properly requires reworking the adaptive Kappa mechanism.
}

bool GIPC::checkCloseGroundVal()
{
    int numbers = h_close_gpNum;
    if(numbers < 1)
        return false;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    int*               _isChange;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_isChange, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(_isChange, 0, sizeof(int)));
    _checkGroundCloseVal<<<blockNum, threadNum>>>(
        _vertexes, _groundOffset, _groundNormal, _isChange, _closeConstraintID, _closeConstraintVal, numbers,
        m_pergroup_kappa ? m_d_close_grp : nullptr, m_pergroup_kappa ? m_d_p2g : nullptr);
    int isChange;
    CUDA_SAFE_CALL(cudaMemcpy(&isChange, _isChange, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(_isChange));

    return (isChange == 1);
}

double2 GIPC::minMaxGroundDist()
{
    //_reduct_minGroundDist << <blockNum, threadNum >> > (_vertexes, _groundOffset, _groundNormal, _isChange, _closeConstraintID, _closeConstraintVal, numbers);

    int numbers = h_gpNum;
    if(numbers < 1)
        return make_double2(1e32, 0);
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double2) * (threadNum >> 5);

    double2* _queue;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_queue, numbers * sizeof(double2)));
    //CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number * sizeof(AABB), cudaMemcpyDeviceToDevice));
    _reduct_MGroundDist<<<blockNum, threadNum, sharedMsize>>>(
        _vertexes, _groundOffset, _groundNormal, _environment_collisionPair, _queue, numbers);
    //_reduct_min_double3_to_double << <blockNum, threadNum, sharedMsize >> > (_moveDir, _tempMinMovement, numbers);

    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while(numbers > 1)
    {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        _reduct_M_double2<<<blockNum, threadNum, sharedMsize>>>(_queue, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double2 minMaxValue;
    cudaMemcpy(&minMaxValue, _queue, sizeof(double2), cudaMemcpyDeviceToHost);
    CUDA_SAFE_CALL(cudaFree(_queue));
    minMaxValue.x = 1.0 / minMaxValue.x;
    return minMaxValue;
}

void GIPC::computeGroundGradient(double3* _gradient, double mKappa)
{
    int numbers = h_gpNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _computeGroundGradient<<<blockNum, threadNum>>>(_vertexes,
                                                    _groundOffset,
                                                    _groundNormal,
                                                    _environment_collisionPair,
                                                    _gradient,
                                                    _gpNum,
                                                    dHat,
                                                    mKappa,
                                                    numbers,
                                                    m_pergroup_kappa ? m_kappa_group : nullptr,
                                                    m_pergroup_kappa ? m_d_p2g : nullptr);
}

void GIPC::computeSoftConstraintGradient(double3* _gradient)
{
    int numbers = softNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    // offset
    _computeSoftConstraintGradient<<<blockNum, threadNum>>>(
        _vertexes, targetVert, targetInd, _gradient, softMotionRate, animation_fullRate,
        m_d_stitch_paired_vertex, m_d_stitch_rest_offset,
        m_d_stitch_abd_body_id,
        reinterpret_cast<const __GEIGEN__::Vector12*>(m_d_abd_body_q),
        softNum);
}

double* GIPC::ensure_reduce_scratch(int count)
{
    // ceil(count/default_threads) doubles are written by the first reduction pass.
    size_t need = (size_t)((count + default_threads - 1) / default_threads) + 1;
    if(need > m_reduce_cap)
    {
        if(m_reduce_scratch)
            CUDA_SAFE_CALL(cudaFree(m_reduce_scratch));
        m_reduce_cap = need + need / 2;  // 1.5x slack → no realloc churn after warmup
        CUDA_SAFE_CALL(cudaMalloc((void**)&m_reduce_scratch, m_reduce_cap * sizeof(double)));
    }
    return m_reduce_scratch;
}

double GIPC::self_largestFeasibleStepSize(double slackness, double* mqueue, int numbers)
{
    if(m_skip_all_collision)
        return 1.0;
    //slackness = 0.9;
    //int numbers = h_cpNum[0];
    if(numbers < 1)
        return 1;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    //double* _minSteps;
    //CUDA_SAFE_CALL(cudaMalloc((void**)&_minSteps, numbers * sizeof(double)));
    //CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number * sizeof(AABB), cudaMemcpyDeviceToDevice));
    _reduct_min_selfTimeStep_to_double<<<blockNum, threadNum, sharedMsize>>>(
        _vertexes, _ccd_collisonPairs, _moveDir, mqueue, slackness, numbers);
    //_reduct_min_double3_to_double << <blockNum, threadNum, sharedMsize >> > (_moveDir, _tempMinMovement, numbers);

    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while(numbers > 1)
    {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        _reduct_max_double<<<blockNum, threadNum, sharedMsize>>>(mqueue, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double minValue;
    cudaMemcpy(&minValue, mqueue, sizeof(double), cudaMemcpyDeviceToHost);
    //printf("                 full ccd time step:  %f\n", 1.0 / minValue);
    //CUDA_SAFE_CALL(cudaFree(_minSteps));
    return 1.0 / minValue;
}

double GIPC::cfl_largestSpeed(double* mqueue)
{
    int                numbers   = surf_vertexNum;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    /*double* _maxV;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_maxV, numbers * sizeof(double)));*/
    //CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number * sizeof(AABB), cudaMemcpyDeviceToDevice));
    _reduct_max_cfl_to_double<<<blockNum, threadNum, sharedMsize>>>(
        _moveDir, mqueue, _surfVerts, numbers);
    //_reduct_min_double3_to_double << <blockNum, threadNum, sharedMsize >> > (_moveDir, _tempMinMovement, numbers);

    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while(numbers > 1)
    {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        _reduct_max_double<<<blockNum, threadNum, sharedMsize>>>(mqueue, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double minValue;
    cudaMemcpy(&minValue, mqueue, sizeof(double), cudaMemcpyDeviceToHost);
    //CUDA_SAFE_CALL(cudaFree(_maxV));
    return minValue;
}

double reduction2Kappa(int type, const double3* A, const double3* B, double* _queue, int vertexNum)
{
    int                numbers   = vertexNum;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    /*double* _queue;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_queue, numbers * sizeof(double)));*/
    if(type == 0)
    {
        //CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number * sizeof(AABB), cudaMemcpyDeviceToDevice));
        _reduct_double3Dot_to_double<<<blockNum, threadNum, sharedMsize>>>(A, B, _queue, numbers);
    }
    else if(type == 1)
    {
        _reduct_double3Sqn_to_double<<<blockNum, threadNum, sharedMsize>>>(A, _queue, numbers);
    }
    //_reduct_min_double3_to_double << <blockNum, threadNum, sharedMsize >> > (_moveDir, _tempMinMovement, numbers);

    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while(numbers > 1)
    {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        __add_reduction<<<blockNum, threadNum, sharedMsize>>>(_queue, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double dotValue;
    cudaMemcpy(&dotValue, _queue, sizeof(double), cudaMemcpyDeviceToHost);
    //CUDA_SAFE_CALL(cudaFree(_queue));
    return dotValue;
}

double GIPC::ground_largestFeasibleStepSize(double slackness, double* mqueue)
{
    if(m_skip_all_collision)
        return 1.0;

    int numbers = surf_vertexNum;
    if(numbers < 1)
        return 1;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    //double* _minSteps;
    //CUDA_SAFE_CALL(cudaMalloc((void**)&_minSteps, numbers * sizeof(double)));

    //if (h_cpNum[0] > 0) {
    //    double3* mvd = new double3[vertexNum];
    //    cudaMemcpy(mvd, _moveDir, sizeof(double3) * vertexNum, cudaMemcpyDeviceToHost);
    //    for (int i = 0;i < vertexNum;i++) {
    //        printf("%f  %f  %f\n", mvd[i].x, mvd[i].y, mvd[i].z);
    //    }
    //    delete[] mvd;
    //}
    _reduct_min_groundTimeStep_to_double<<<blockNum, threadNum, sharedMsize>>>(
        _vertexes, _surfVerts, _groundOffset, _groundNormal, _moveDir, mqueue, slackness, numbers,
        _point_body_id, _ground_skip_body, _ground_body_count);


    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while(numbers > 1)
    {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        _reduct_max_double<<<blockNum, threadNum, sharedMsize>>>(mqueue, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double minValue;
    cudaMemcpy(&minValue, mqueue, sizeof(double), cudaMemcpyDeviceToHost);
    //CUDA_SAFE_CALL(cudaFree(_minSteps));
    return 1.0 / minValue;
}

// ②-D2H batched variants for the two CCD step-size reductions that fire
// back-to-back at the top of each line search. Each writes minValue to
// out_slot via D2D (no blocking sync). Caller does host-side 1.0/x and the
// m_skip_all_collision / numbers<1 guards (we don't queue any kernels when
// the early-return condition holds).

void GIPC::ground_largestFeasibleStepSize_DeviceOut(double slackness, double* mqueue, double* out_slot)
{
    int numbers = surf_vertexNum;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    _reduct_min_groundTimeStep_to_double<<<blockNum, threadNum, sharedMsize>>>(
        _vertexes, _surfVerts, _groundOffset, _groundNormal, _moveDir, mqueue, slackness, numbers,
        _point_body_id, _ground_skip_body, _ground_body_count);

    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;
    while(numbers > 1)
    {
        _reduct_max_double<<<blockNum, threadNum, sharedMsize>>>(mqueue, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    CUDA_SAFE_CALL(cudaMemcpyAsync(out_slot, mqueue, sizeof(double), cudaMemcpyDeviceToDevice));
}

void GIPC::self_largestFeasibleStepSize_DeviceOut(double slackness, double* mqueue, int numbers, double* out_slot)
{
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    _reduct_min_selfTimeStep_to_double<<<blockNum, threadNum, sharedMsize>>>(
        _vertexes, _ccd_collisonPairs, _moveDir, mqueue, slackness, numbers);

    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;
    while(numbers > 1)
    {
        _reduct_max_double<<<blockNum, threadNum, sharedMsize>>>(mqueue, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    CUDA_SAFE_CALL(cudaMemcpyAsync(out_slot, mqueue, sizeof(double), cudaMemcpyDeviceToDevice));
}


double GIPC::InjectiveStepSize(double slackness, double errorRate, double* mqueue, uint4* tets)
{

    int numbers = tetrahedraNum;
    if(numbers < 1)
        return 1;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    _reduct_min_InjectiveTimeStep_to_double<<<blockNum, threadNum, sharedMsize>>>(
        _vertexes, tets, _moveDir, mqueue, slackness, errorRate, numbers);


    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while(numbers > 1)
    {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        _reduct_max_double<<<blockNum, threadNum, sharedMsize>>>(mqueue, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double minValue;
    cudaMemcpy(&minValue, mqueue, sizeof(double), cudaMemcpyDeviceToHost);
    //printf("Injective Time step:   %f\n", 1.0 / minValue);
    //if (1.0 / minValue < 1) {
    //    system("pause");
    //}
    //CUDA_SAFE_CALL(cudaFree(_minSteps));
    return 1.0 / minValue;
}

void GIPC::buildCP()
{
    if(m_skip_all_collision)
    {
        memset(h_cpNum, 0, sizeof(h_cpNum));
        h_gpNum = 0;
        return;
    }

    // [env-det] EE detection settings + env-local vertex map MUST be set BEFORE the per-env branch,
    // else the per-env path (which returns early) runs the EE dedup with GLOBAL edge indices (not
    // env-local) → cross-env asymmetric. Idempotent; the merged path below re-runs harmlessly.
    set_ee_nodedup(getenv("STIFF_EE_NODEDUP") ? 1 : 0);
    set_ee_detgate(getenv("STIFF_EE_DETGATE") ? 1 : 0);
    set_bvh_envpart(getenv("STIFF_BVH_ENVPART") ? 1 : 0);
    // [perenv-par] per-vertex cross-env skip at self-collision emission (robust where BVH env-part is
    // bypassed by env-MIXED co-located nodes). Gated STIFF_DECOUPLE_THRESH; null = off (legacy path).
    set_self_p2g((getenv("STIFF_DECOUPLE_THRESH") && m_d_p2g) ? m_d_p2g : nullptr);
    set_ee_canon(getenv("STIFF_EE_CANON") ? 1 : 0);
    set_ee_nomollify(getenv("STIFF_EE_NOMOLLIFY") ? 1 : 0);
    if(getenv("STIFF_EE_CANON") && m_d_p2g && !m_vloc_built)
    {
        std::vector<int> hp(vertexNum);
        CUDA_SAFE_CALL(cudaMemcpy(hp.data(), m_d_p2g, (size_t)vertexNum*sizeof(int), cudaMemcpyDeviceToHost));
        std::vector<int> vloc(vertexNum, 0); std::vector<int> ec;
        for(int v=0; v<vertexNum; v++){ int g=hp[v]; if(g<0) continue; if(g>=(int)ec.size()) ec.resize(g+1,0); vloc[v]=ec[g]++; }
        CUDA_SAFE_CALL(cudaMalloc((void**)&m_d_vloc, (size_t)vertexNum*sizeof(int)));
        CUDA_SAFE_CALL(cudaMemcpy(m_d_vloc, vloc.data(), (size_t)vertexNum*sizeof(int), cudaMemcpyHostToDevice));
        set_ee_vloc(m_d_vloc); m_vloc_built = true;
    }

    // [multi-env P2] per-env BVH path: build each env's tree on LOCAL verts + detect, looped.
    if(m_perenv_bvh && m_d_p2g)
    {
        if(m_perenv_bvh_groups == 0)
            buildPerEnvBVHIndex(kEnvAlphaSlots, m_d_p2g);
        buildBVH_and_CP_perenv(dHat);
        return;
    }

    if(!m_aux_stream)
        cudaStreamCreate(&m_aux_stream);

    // Memsets on default stream. Use an event so aux stream observes them
    // before its kernel reads/atomicAdds _cpNum.
    CUDA_SAFE_CALL(cudaMemsetAsync(_cpNum, 0, 5 * sizeof(uint32_t), 0));
    CUDA_SAFE_CALL(cudaMemsetAsync(_gpNum, 0, sizeof(uint32_t), 0));
    cudaEvent_t reset_evt;
    cudaEventCreateWithFlags(&reset_evt, cudaEventDisableTiming);
    cudaEventRecord(reset_evt, 0);
    cudaStreamWaitEvent(m_aux_stream, reset_evt, 0);

    // bvh_f on default stream, bvh_e on aux stream -> overlap.
    // Both atomicAdd into _cpNum & _collisionPair; CUDA atomics handle
    // cross-stream contention correctly. Pair-set order doesn't matter
    // to consumers (they iterate 0..h_cpNum[0]).
    // [xenv pin] isolate the two detection passes to localize the asymmetry:
    //   STIFF_SKIP_F=1 → only edge-edge (tests EE ownership obj_idx<self_eid)
    //   STIFF_SKIP_E=1 → only point-triangle (tests Morton/candidate; PT has no index dedup)
    set_ee_nodedup(getenv("STIFF_EE_NODEDUP") ? 1 : 0);
    set_ee_detgate(getenv("STIFF_EE_DETGATE") ? 1 : 0);
    set_bvh_envpart(getenv("STIFF_BVH_ENVPART") ? 1 : 0);  // [env-part B]
    // [perenv-par] per-vertex cross-env skip at self-collision emission (see note above). null = off.
    set_self_p2g((getenv("STIFF_DECOUPLE_THRESH") && m_d_p2g) ? m_d_p2g : nullptr);
    set_ee_canon(getenv("STIFF_EE_CANON") ? 1 : 0);
    set_ee_nomollify(getenv("STIFF_EE_NOMOLLIFY") ? 1 : 0);
    { static int _tc = 0; set_ee_trace((getenv("STIFF_EE_TRACE") && _tc++ == 0) ? 1 : 0); }  // first buildCP (iter0) only
    set_ee_tgt(getenv("STIFF_BAR_TGT0")?atoi(getenv("STIFF_BAR_TGT0")):-1, getenv("STIFF_BAR_TGT1")?atoi(getenv("STIFF_BAR_TGT1")):-1);
    // [env-det] build the global→env-local vertex id map once (canon total-order tie-break).
    if(getenv("STIFF_EE_CANON") && m_d_p2g && !m_vloc_built)
    {
        std::vector<int> hp(vertexNum);
        CUDA_SAFE_CALL(cudaMemcpy(hp.data(), m_d_p2g, (size_t)vertexNum*sizeof(int), cudaMemcpyDeviceToHost));
        std::vector<int> vloc(vertexNum, 0); std::vector<int> ec;
        for(int v=0; v<vertexNum; v++){ int g=hp[v]; if(g<0) continue; if(g>=(int)ec.size()) ec.resize(g+1,0); vloc[v]=ec[g]++; }
        CUDA_SAFE_CALL(cudaMalloc((void**)&m_d_vloc, (size_t)vertexNum*sizeof(int)));
        CUDA_SAFE_CALL(cudaMemcpy(m_d_vloc, vloc.data(), (size_t)vertexNum*sizeof(int), cudaMemcpyHostToDevice));
        set_ee_vloc(m_d_vloc); m_vloc_built = true;
    }
    if(getenv("STIFF_STACK_DIAG")) reset_max_stack();
    // [env-det dump] one-shot dump of the edge BVH Morton hashes + edges to verify env0/env1 trees
    // are byte-identical modulo the env bit. STIFF_MCDUMP.
    { static int _mdc=0;
      if(getenv("STIFF_MCDUMP") && _mdc++==1){   // fire on 2nd buildCP (frame-0 pre-solve, still mirror)
        int nE=(int)bvh_e.edge_number;
        std::vector<uint64_t> hm(nE); std::vector<uint2> he(nE);
        CUDA_SAFE_CALL(cudaMemcpy(hm.data(), bvh_e._MChash, (size_t)nE*sizeof(uint64_t), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(he.data(), bvh_e._edges, (size_t)nE*sizeof(uint2), cudaMemcpyDeviceToHost));
        std::vector<uint32_t> hi(nE);
        CUDA_SAFE_CALL(cudaMemcpy(hi.data(), bvh_e._indices, (size_t)nE*sizeof(uint32_t), cudaMemcpyDeviceToHost));
        FILE*f=fopen("/tmp/xd_mch.bin","wb"); fwrite(hm.data(),sizeof(uint64_t),nE,f); fclose(f);
        f=fopen("/tmp/xd_edges2.bin","wb"); fwrite(he.data(),sizeof(uint2),nE,f); fclose(f);
        f=fopen("/tmp/xd_idx.bin","wb"); fwrite(hi.data(),sizeof(uint32_t),nE,f); fclose(f);
        printf("[mcdump] dumped %d edge MChash + edges + indices\n", nE); } }
    if(!getenv("STIFF_SKIP_F")) bvh_f.SelfCollitionDetect(dHat);
    if(!getenv("STIFF_SKIP_E")) bvh_e.SelfCollitionDetect(dHat, m_aux_stream);
    if(getenv("STIFF_STACK_DIAG")) { CUDA_SAFE_CALL(cudaDeviceSynchronize());
        static int _sd=0; if(_sd++<3) printf("[stack] max traversal depth = %d (cap 2048)\n", get_max_stack()); }
    GroundCollisionDetect();
    CUDA_SAFE_CALL(cudaStreamSynchronize(m_aux_stream));
    cudaEventDestroy(reset_evt);

    {   // [9d28824-port] contiguous _cpNum[0:5]+_gpNum[5]: one 6-int D2H.
        uint32_t cp_gp_buf[6];
        CUDA_SAFE_CALL(cudaMemcpy(cp_gp_buf, _cpNum, 6 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        memcpy(h_cpNum, cp_gp_buf, 5 * sizeof(uint32_t));
        h_gpNum = cp_gp_buf[5];
    }

    // Overflow → grow DCD pair buffers + redo detection (BVH unchanged, no pairs
    // lost; emits were redirected to the trash slot so nothing was corrupted).
    while((int)h_cpNum[0] > MAX_COLLITION_PAIRS_NUM)
    {
        int newcap = (int)(h_cpNum[0] + h_cpNum[0] / 2) + 1;
        printf("[DCD-grow] h_cpNum=%u > cap=%d -> grow to %d, redo detection\n",
               h_cpNum[0], MAX_COLLITION_PAIRS_NUM, newcap);
        CUDA_SAFE_CALL(cudaFree(_collisonPairs));
        CUDA_SAFE_CALL(cudaFree(_MatIndex));
        CUDA_SAFE_CALL(cudaMalloc((void**)&_collisonPairs, ((size_t)newcap + 1) * sizeof(int4)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&_MatIndex,      ((size_t)newcap + 1) * sizeof(int)));
        MAX_COLLITION_PAIRS_NUM = newcap;
        bvh_f._collisionPair = bvh_e._collisionPair = _collisonPairs;
        bvh_f._MatIndex      = bvh_e._MatIndex      = _MatIndex;
        set_emit_caps(MAX_COLLITION_PAIRS_NUM, MAX_CCD_COLLITION_PAIRS_NUM);
        CUDA_SAFE_CALL(cudaMemsetAsync(_cpNum, 0, 5 * sizeof(uint32_t), 0));
        CUDA_SAFE_CALL(cudaMemsetAsync(_gpNum, 0, sizeof(uint32_t), 0));
        bvh_f.SelfCollitionDetect(dHat);
        bvh_e.SelfCollitionDetect(dHat, m_aux_stream);
        GroundCollisionDetect();
        CUDA_SAFE_CALL(cudaStreamSynchronize(m_aux_stream));
        {   // [9d28824-port] one 6-int D2H
            uint32_t cp_gp_buf[6];
            CUDA_SAFE_CALL(cudaMemcpy(cp_gp_buf, _cpNum, 6 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            memcpy(h_cpNum, cp_gp_buf, 5 * sizeof(uint32_t));
            h_gpNum = cp_gp_buf[5];
        }
    }
}

// [multi-env P3a] segmented per-env reduction PRIMITIVE — the core machinery the
// block-diagonal solve needs (per-env residual / dot-products). For each entry i,
// route its |vec[i]|^2 (and a unit count) into bucket d_point_to_group[i]. Atomic
// bucketing here is fine for the read-only diagnostic; the in-solver version (P3a
// step2) must use fixed-order segmented reduce (cub::DeviceSegmentedReduce) so the
// per-env sums are order-deterministic.
__global__ void _per_env_sqnorm_accum(const int* p2g, const double3* vec,
                                      double* per_env_sq, int* per_env_cnt,
                                      int n, int ng)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    int g = p2g[i];
    if(g < 0 || g >= ng) return;
    if(per_env_cnt) atomicAdd(&per_env_cnt[g], 1);
    if(vec)
    {
        double3 v = vec[i];
        atomicAdd(&per_env_sq[g], v.x * v.x + v.y * v.y + v.z * v.z);
    }
}

// [decouple] per-env binned reduction for kappa: gsum_g = Σ_{v∈g} gc·GE, gsnorm_g = Σ_{v∈g} |gc|².
// Binned (exact, order-independent) ⇒ env_g's value depends ONLY on env_g's verts → batch-invariant
// (env_0's kappa no longer depends on its batch-mates' contact state). bins: [ng*BINNED_K] each.
__global__ void _per_env_kappa_deposit(const int* p2g, const double3* gc, const double3* GE,
                                       double* gsum_bin, double* gsnorm_bin, int n, int ng)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    int g = p2g[i];
    if(g < 0 || g >= ng) return;
    double3 a = gc[i], b = GE[i];
    _binDepBase(gsum_bin   + (size_t)g * BINNED_K, a.x * b.x + a.y * b.y + a.z * b.z);
    _binDepBase(gsnorm_bin + (size_t)g * BINNED_K, a.x * a.x + a.y * a.y + a.z * a.z);
}
__global__ void _per_env_kappa_combine(double* gsum_g, double* gsnorm_g,
                                       const double* gsum_bin, const double* gsnorm_bin, int ng)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if(g >= ng) return;
    double s = 0.0, q = 0.0;
    for(int k = BINNED_K - 1; k >= 0; --k)   // finest bin first, fixed order
    { s += gsum_bin[(size_t)g * BINNED_K + k]; q += gsnorm_bin[(size_t)g * BINNED_K + k]; }
    gsum_g[g] = s; gsnorm_g[g] = q;
}

// [multi-env P3a] per-env max CFL speed = max over an env's SURFACE verts of
// |moveDir[v]| (same metric as _reduct_max_cfl_to_double). Used by a read-only
// diagnostic to see whether the global alpha_CFL = sqrt(dHat)/maxSpeed*0.5 is
// being dragged down by ONE env (=> per-env line-search would decouple them).
// atomicMax on double via bit-twiddling (positive doubles only -> monotone bits).
__device__ inline void _atomicMaxPosDouble(double* addr, double val)
{
    unsigned long long* a = (unsigned long long*)addr;
    unsigned long long  old = *a, assumed;
    do { assumed = old;
         double cur = __longlong_as_double((long long)assumed);
         if(cur >= val) break;
         old = atomicCAS(a, assumed, (unsigned long long)__double_as_longlong(val));
    } while(assumed != old);
}
__global__ void _per_env_max_cfl(const int* p2g, const double3* moveDir,
                                 const uint32_t* mSVI, double* per_env_max,
                                 int n_surf, int ng)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n_surf) return;
    int v = mSVI[i];
    int g = p2g[v];
    if(g < 0 || g >= ng) return;
    _atomicMaxPosDouble(&per_env_max[g], __GEIGEN__::__norm(moveDir[v]));
}

// [perf] DEVICE-SIDE per-env feasible alpha + freeze — replaces the per-Newton-iter host round-trip
// (cudaDeviceSynchronize + 4x256-double D2H + 256-env host loop + H2D) that made the per-env path
// host-bound. scratch layout: [0*ng)=ground, [1*ng)=narrow-self, [2*ng)=refined-self, [3*ng)=cfl-max.
// Writes env_alpha[g] directly (device), atomics n_env/n_frozen into cnt[2]. Math is bit-identical to
// the host loop (same per-env formulas, no reduction) → preserves strict cross-env bit-identity.
__global__ void _per_env_alpha_compute(double* env_alpha, const double* scratch, int ng,
                                       double sq, double ccd_size, int have_ccd,
                                       double temp_alpha, double alpha_CFL, int decouple,
                                       int no_refine, double thr_cv, int* cnt)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if(g >= ng) return;
    double hmx = scratch[3 * ng + g];
    if(hmx <= 0.0) return;   // absent env → leave env_alpha[g] unchanged (matches host `continue`)
    double hg = scratch[0 * ng + g], hs = scratch[1 * ng + g], hr = scratch[2 * ng + g];
    double ta = 1.0;
    if(hg > 0.0) ta = fmin(ta, 1.0 / hg);
    if(hs > 0.0) ta = fmin(ta, 1.0 / hs);
    double a = ta;
    if(have_ccd)
    {
        double acfl     = sq / hmx * 0.5;
        a               = fmin(ta, acfl);
        double gate_lhs = decouple ? ta : temp_alpha;
        double gate_rhs = decouple ? acfl : alpha_CFL;
        if(!no_refine && gate_lhs > 2.0 * gate_rhs)
        {
            double refined = (hr > 0.0) ? 1.0 / hr : 1.0;
            a              = fmin(ta, refined * ccd_size);
            a              = fmax(a, acfl);
        }
    }
    if(decouple && hmx < thr_cv) a = 0.0;   // freeze converged env
    env_alpha[g] = a;
    atomicAdd(&cnt[0], 1);                   // n_env (present)
    if(a == 0.0) atomicAdd(&cnt[1], 1);      // n_frozen
}

// [de-CPU] per-env CCD search-inflation alpha ta_e = min(1, 1/ground_e, 1/narrowSelf_e), computed
// ON DEVICE from m_env_scratch (filled by S1 Phase A) — replaces the per-Newton D2H(2*NG doubles) +
// host ta[] loop in buildFullCP. Math identical to the removed host loop (IEEE div/min → bit-exact).
__global__ void _compute_perenv_ta(const double* scratch, double* ta, int ng)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if(g >= ng) return;
    double t  = 1.0;
    double gs = scratch[0 * ng + g];
    double ns = scratch[1 * ng + g];
    if(gs > 0.0) t = fmin(t, 1.0 / gs);
    if(ns > 0.0) t = fmin(t, 1.0 / ns);
    ta[g] = t;
}

// [S4-dev] device-derived per-env active mask — replaces the S4 host detection (own max-move
// kernel + D2H + host loop + H2D per iter) with ZERO added D2H: the freeze decision is already on
// device in m_env_alpha (set by _per_env_alpha_compute from a REAL solve). active = (alpha != 0).
// A masked env's next moveDir is 0 (RHS zeroed) -> hmx=0 -> _per_env_alpha_compute treats it as
// absent and leaves env_alpha unchanged (stays 0) -> stays masked until the periodic all-active
// recheck (top of loop) re-solves it for bounce-back detection. Deterministic (fixed cadence,
// per-env decision) -> strict/batch-invariance safe.
__global__ void _mask_fill(int* m, int v, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) m[i] = v;
}
__global__ void _mask_from_env_alpha(int* env_active, const double* env_alpha, int ng)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if(g >= ng) return;
    env_active[g] = (env_alpha[g] == 0.0) ? 0 : 1;
}

// [perf] DEVICE-SIDE per-group κ doubling (postLineSearch) — replaces the per-Newton host round-trip
// (D2H close flags + 256-env host loop + H2D). Doubles m_kappa_group[g] in place for groups that hit a
// close contact (capped at kappaMax, host scalar), and atomicMax's the envelope into maxK_out (init =
// current Kappa). Bit-identical to the host loop (same double+cap; max is order-free) → strict OK.
__global__ void _per_group_kappa_double(double* kappa_group, const int* close_grp, int ng,
                                        double kappaMax, double* maxK_out)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if(g >= ng) return;
    if(close_grp[g])
    {
        double k = kappa_group[g] * 2.0;
        if(k > kappaMax) k = kappaMax;
        kappa_group[g] = k;
        _atomicMaxPosDouble(maxK_out, k);
    }
}

// [perf] DEVICE-SIDE per-env initKappa finalize — replaces the per-frame D2H(gsum_g/gsnorm_g) +
// NG-env host loop + H2D(kappa_group). Kg = clamp(max(-gsum/gsnorm, suggested), 0, kmax), where
// suggested/kmax are env-independent host scalars. Bit-identical to the host loop → strict OK.
__global__ void _per_env_kappa_finalize(const double* gsum_g, const double* gsnorm_g,
                                        double* kappa_group, int ng, double suggested, double kmax)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if(g >= ng) return;
    double mk = (gsnorm_g[g] > 0.0) ? (-gsum_g[g] / gsnorm_g[g]) : 0.0;
    if(mk < 0.0) mk = 0.0;
    double Kg = (mk > suggested) ? mk : suggested;
    if(Kg > kmax) Kg = kmax;
    kappa_group[g] = Kg;
}

// [multi-env S4 probe / aa17212] per-env MAX move (the real Newton-exit metric):
// max over an env's verts of |moveDir[i]|. Used by the S4 active-mask detection
// and the P3a-step2 per-env Newton convergence diagnostic. per_env_max pre-zeroed.
__global__ void _per_env_max_move(const int* p2g, const double3* moveDir,
                                  double* per_env_max, int n, int ng)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    int g = p2g[i];
    if(g < 0 || g >= ng) return;
    _atomicMaxPosDouble(&per_env_max[g], __GEIGEN__::__norm(moveDir[i]));
}

// [multi-env S1] per-env CCD feasible-step reductions. Same per-element 1/timestep
// as _reduct_min_groundTimeStep_to_double / _reduct_min_selfTimeStep_to_double, but
// accumulated into per-env MAX(1/step) buckets (env = group of the element's vertex;
// intra-env after P1) instead of one global max. per-env feasible step = 1/bucket.
// per_env_inv must be pre-zeroed by caller.
__global__ void _per_env_groundTimeStep_max(const double3* vertexes,
                                            const uint32_t* surfVertIds,
                                            const double* g_offset, const double3* g_normal,
                                            const double3* moveDir, const int* p2g,
                                            double* per_env_inv, double slackness, int number,
                                            const int* _point_body_id,
                                            const int* _ground_skip_body, int _ground_body_count,
                                            int ng)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number) return;
    int svI = surfVertIds[idx];
    int g   = p2g[svI];
    if(g < 0 || g >= ng) return;
    double temp = 0.0;
    bool   skip = false;
    if(_point_body_id && _ground_skip_body && _ground_body_count > 0)
    {
        int bid = _point_body_id[svI];
        if(bid >= 0 && bid < _ground_body_count && _ground_skip_body[bid]) skip = true;
    }
    if(!skip)
    {
        double3 normal = *g_normal;
        double  coef   = __GEIGEN__::__v_vec_dot(normal, moveDir[svI]);
        if(coef > 0.0)
        {
            double dist = __GEIGEN__::__v_vec_dot(normal, vertexes[svI]) - *g_offset;
            temp        = coef / (dist * slackness);
        }
    }
    if(temp > 0.0) _atomicMaxPosDouble(&per_env_inv[g], temp);
}

__global__ void _per_env_selfTimeStep_max(const double3* vertexes, const int4* pairs,
                                          const double3* moveDir, const int* p2g,
                                          double* per_env_inv, double slackness, int number, int ng,
                                          const int* vloc)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number) return;
    double CCDDistRatio = 1.0 - slackness;
    int4   MMCVIDI      = pairs[idx];
    double temp;
    int    v0;
    if(MMCVIDI.x < 0)
    {
        MMCVIDI.x = -MMCVIDI.x - 1;
        v0        = MMCVIDI.x;
        int p = MMCVIDI.x, t0 = MMCVIDI.y, t1 = MMCVIDI.z, t2 = MMCVIDI.w;
        // [env-det] canonicalize the triangle vertex order by env-local id so mirror PT pairs feed
        // point_triangle_ccd in an identical order ⇒ bit-identical TOI cross-env.
        if(vloc) { int a=t0,b=t1,c=t2;
            if(vloc[b]<vloc[a]){int t=a;a=b;b=t;} if(vloc[c]<vloc[b]){int t=b;b=c;c=t;}
            if(vloc[b]<vloc[a]){int t=a;a=b;b=t;} t0=a; t1=b; t2=c; }
        temp = 1.0 / point_triangle_ccd(vertexes[p], vertexes[t0],
                                        vertexes[t1], vertexes[t2],
                                        __GEIGEN__::__s_vec_multiply(moveDir[p], -1),
                                        __GEIGEN__::__s_vec_multiply(moveDir[t0], -1),
                                        __GEIGEN__::__s_vec_multiply(moveDir[t1], -1),
                                        __GEIGEN__::__s_vec_multiply(moveDir[t2], -1),
                                        CCDDistRatio, 0);
    }
    else
    {
        v0   = MMCVIDI.x;
        int e0a=MMCVIDI.x, e0b=MMCVIDI.y, e1a=MMCVIDI.z, e1b=MMCVIDI.w;
        // [env-det] canonicalize edge endpoints + edge order by env-local id ⇒ mirror EE pairs feed
        // edge_edge_ccd identically ⇒ bit-identical TOI cross-env (the refined-CCD `hr` 1-ULP seed).
        if(vloc) {
            if(vloc[e0b]<vloc[e0a]){int t=e0a;e0a=e0b;e0b=t;}
            if(vloc[e1b]<vloc[e1a]){int t=e1a;e1a=e1b;e1b=t;}
            if(vloc[e1a]<vloc[e0a]){int t=e0a;e0a=e1a;e1a=t; t=e0b;e0b=e1b;e1b=t;}
        }
        temp = 1.0 / edge_edge_ccd(vertexes[e0a], vertexes[e0b],
                                   vertexes[e1a], vertexes[e1b],
                                   __GEIGEN__::__s_vec_multiply(moveDir[e0a], -1),
                                   __GEIGEN__::__s_vec_multiply(moveDir[e0b], -1),
                                   __GEIGEN__::__s_vec_multiply(moveDir[e1a], -1),
                                   __GEIGEN__::__s_vec_multiply(moveDir[e1b], -1),
                                   CCDDistRatio, 0);
    }
    int g = p2g[v0];
    if(g < 0 || g >= ng) return;
    if(temp > 0.0) _atomicMaxPosDouble(&per_env_inv[g], temp);
}

// [multi-env P2] per-env CCD: build each env's swept tree on LOCAL verts + full-detect, looped.
void GIPC::buildBVH_and_CP_perenv_CCD(double alpha)
{
    if(m_skip_all_collision) { h_ccd_cpNum = 0; return; }
    int NG = m_perenv_bvh_groups;
    double3* sf = bvh_f._vertexes;
    double3* se = bvh_e._vertexes;
    bvh_f._vertexes = _vertexes;
    bvh_e._vertexes = _vertexes;
    (void)NG;
    // [decouple] PER-ENV CCD search inflation. The global `alpha` (=temp_alpha=min over ALL envs)
    // makes env e's swept-BVH search — and thus its refined-self CCD pair set + hr (refined-self
    // timestep) — depend on the MATES → batch-coupling (the confirmed root: frame0 k=3 hr diverged
    // 5.48 vs 4.93 when global temp_alpha diverged 0.272 vs 0.033). Here each env e searches with its
    // OWN feasible alpha ta_e = min(1, 1/ground_e, 1/narrowSelf_e) (from m_env_scratch, filled by S1
    // Phase A; depends only on env e → batch-invariant). ta_e ≥ global alpha → conservative superset
    // of pairs → hr is the true most-constraining value → refinement KEPT (unlike STIFF_NO_REFINE,
    // which skipped it and caused excessive backtracking). Gated STIFF_DECOUPLE_THRESH.
    const int KNG = kEnvAlphaSlots;
    bool perenv_ta = (getenv("STIFF_DECOUPLE_THRESH") && m_env_scratch && getenv("STIFF_PERENV_ALPHA"));
    // [de-CPU] ta computed on device (see _compute_perenv_ta) — the per-env launches below read
    // their env's slot via the kernels' alpha_dev param; NO D2H / host loop.
    static double* d_perenv_ta = nullptr;
    if(perenv_ta)
    {
        if(!d_perenv_ta) CUDA_SAFE_CALL(cudaMalloc((void**)&d_perenv_ta, KNG * sizeof(double)));
        _compute_perenv_ta<<<(KNG + 255) / 256, 256>>>(m_env_scratch, d_perenv_ta, KNG);
    }
    // scalar fallback (kernels use it when alpha_dev == nullptr)
    auto env_alpha_dev = [&](int e) -> const double* { return perenv_ta ? d_perenv_ta + e : nullptr; };
    // [perenv-parallel #2] STIFF_PERENV_PAR: run the per-env SWEPT (CCD) builds+queries concurrently
    // on the K-stream scratch pool — mirrors the DCD loop. The 6-7ms _selfQuery_*_ccd kernels are
    // occupancy-starved at 1-env size (~25 blocks); overlapping K envs fills the GPU.
    const char* _ccd_pe_par = getenv("STIFF_PERENV_PAR");   // value-aware (=0 disables)
    bool ccd_par = _ccd_pe_par && atoi(_ccd_pe_par) != 0;
    int  ccd_K   = 1;
    if(ccd_par) { int cap = getenv("STIFF_PERENV_K") ? atoi(getenv("STIFF_PERENV_K")) : 8;
                  ccd_K = (int)h_perenv_active.size(); if(ccd_K > cap) ccd_K = cap; if(ccd_K < 1) ccd_K = 1;
                  allocPerEnvPool(ccd_K); }
    BvhScratch cof{bvh_f._nodes,bvh_f._bvs,bvh_f._MChash,bvh_f._indices,bvh_f._tempLeafBox,bvh_f._flags,bvh_f.m_node_env,
                   bvh_f._sort_tmp,bvh_f._sort_tmp_bytes,bvh_f._mch_alt,bvh_f._idx_alt,bvh_f._sort_cap};
    BvhScratch coe{bvh_e._nodes,bvh_e._bvs,bvh_e._MChash,bvh_e._indices,bvh_e._tempLeafBox,bvh_e._flags,bvh_e.m_node_env,
                   bvh_e._sort_tmp,bvh_e._sort_tmp_bytes,bvh_e._mch_alt,bvh_e._idx_alt,bvh_e._sort_cap};
    auto cswapIn = [](lbvh& b, BvhScratch& s){ b._nodes=s.nodes; b._bvs=s.bvs; b._MChash=s.mch;
        b._indices=s.idx; b._tempLeafBox=s.tmp; b._flags=s.flags; b.m_node_env=s.node_env;
        b._sort_tmp=s.sort_tmp; b._sort_tmp_bytes=s.sort_bytes;
        b._mch_alt=s.mch_alt; b._idx_alt=s.idx_alt; b._sort_cap=s.sort_cap; };
  ccd_redo:
    CUDA_SAFE_CALL(cudaMemset(_cpNum, 0, sizeof(uint32_t)));
    // memset is on the DEFAULT stream; pool-stream detects atomicAdd _cpNum → make the zero globally
    // visible before any pool-stream work (same fix as the DCD loop).
    if(ccd_par) CUDA_SAFE_CALL(cudaDeviceSynchronize());
    {
        int ci = 0;
        for(int e : h_perenv_active)
        {
            const double* aedev = env_alpha_dev(e);   // [de-CPU] device slot (nullptr -> scalar alpha)
            cudaStream_t st = ccd_par ? m_pool_streams[ci % ccd_K] : (cudaStream_t)0;
            if(h_perenv_face_cnt[e] > 0)
            {
                if(ccd_par) cswapIn(bvh_f, m_pool_f[ci % ccd_K]);
                bvh_f._active_idx        = d_perenv_face_idx + h_perenv_face_off[e];
                bvh_f.face_number_active = h_perenv_face_cnt[e];
                bvh_f.ConstructFullCCD(_moveDir, alpha, st, aedev);
                bvh_f.SelfCollitionFullDetect(dHat, _moveDir, alpha, st, aedev);
            }
            if(h_perenv_edge_cnt[e] > 0)
            {
                if(ccd_par) cswapIn(bvh_e, m_pool_e[ci % ccd_K]);
                bvh_e._active_idx        = d_perenv_edge_idx + h_perenv_edge_off[e];
                bvh_e.face_number_active = h_perenv_edge_cnt[e];
                bvh_e.ConstructFullCCD(_moveDir, alpha, st, aedev);
                bvh_e.SelfCollitionFullDetect(dHat, _moveDir, alpha, st, aedev);
            }
            ++ci;
        }
    }
    if(ccd_par) { for(int k2 = 0; k2 < ccd_K; ++k2) CUDA_SAFE_CALL(cudaStreamSynchronize(m_pool_streams[k2]));
                  cswapIn(bvh_f, cof); cswapIn(bvh_e, coe); }  // restore original scratch
    CUDA_SAFE_CALL(cudaMemcpy(&h_ccd_cpNum, _cpNum, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    // [perenv-parallel #1 FIX] the per-env CCD path (like the merged buildFullCP) MUST grow + redo on
    // overflow — else at the grasp h_ccd_cpNum exceeds the cap and the line-search per-env alpha reads
    // _ccd_collisonPairs OOB → illegal access (the N>4 crash). Emits past cap went to the trash slot.
    if((int)h_ccd_cpNum > MAX_CCD_COLLITION_PAIRS_NUM)
    {
        int newcap = (int)(h_ccd_cpNum + h_ccd_cpNum / 2) + 1;
        printf("[perenv CCD-grow] h_ccd_cpNum=%u > cap=%d -> grow to %d, redo\n",
               h_ccd_cpNum, MAX_CCD_COLLITION_PAIRS_NUM, newcap);
        CUDA_SAFE_CALL(cudaFree(_ccd_collisonPairs));
        CUDA_SAFE_CALL(cudaMalloc((void**)&_ccd_collisonPairs, ((size_t)newcap + 1) * sizeof(int4)));
        MAX_CCD_COLLITION_PAIRS_NUM = newcap;
        bvh_f._ccd_collisionPair = bvh_e._ccd_collisionPair = _ccd_collisonPairs;
        set_emit_caps(MAX_COLLITION_PAIRS_NUM, MAX_CCD_COLLITION_PAIRS_NUM);
        goto ccd_redo;
    }
    bvh_f._active_idx = nullptr; bvh_f.face_number_active = 0;
    bvh_e._active_idx = nullptr; bvh_e.face_number_active = 0;
    bvh_f._vertexes = sf;
    bvh_e._vertexes = se;
}

void GIPC::buildFullCP(const double& alpha)
{
    if(m_skip_all_collision)
    {
        h_ccd_cpNum = 0;
        return;
    }

    // [multi-env P2] per-env CCD path (the swept-BVH equivalent of the per-env DCD path).
    if(m_perenv_bvh && m_d_p2g && m_perenv_bvh_groups > 0)
    {
        buildBVH_and_CP_perenv_CCD(alpha);
        return;
    }

    if(!m_aux_stream)
        cudaStreamCreate(&m_aux_stream);

    CUDA_SAFE_CALL(cudaMemsetAsync(_cpNum, 0, sizeof(uint32_t), 0));
    cudaEvent_t reset_evt;
    cudaEventCreateWithFlags(&reset_evt, cudaEventDisableTiming);
    cudaEventRecord(reset_evt, 0);
    cudaStreamWaitEvent(m_aux_stream, reset_evt, 0);

    // Same overlap pattern as buildCP.
    bvh_f.SelfCollitionFullDetect(dHat, _moveDir, alpha);
    bvh_e.SelfCollitionFullDetect(dHat, _moveDir, alpha, m_aux_stream);
    CUDA_SAFE_CALL(cudaStreamSynchronize(m_aux_stream));
    cudaEventDestroy(reset_evt);

    CUDA_SAFE_CALL(cudaMemcpy(&h_ccd_cpNum, _cpNum, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Overflow → grow CCD pair buffer + redo detection. The swept BVH
    // (ConstructFullCCD) is unchanged, so we only re-run the query into the
    // larger buffer. Emits past the old cap went to the trash slot (no OOB), and
    // consumers (self_largestFeasibleStepSize) run only after this returns, so
    // they always see a fully-populated, in-bounds buffer.
    while((int)h_ccd_cpNum > MAX_CCD_COLLITION_PAIRS_NUM)
    {
        int newcap = (int)(h_ccd_cpNum + h_ccd_cpNum / 2) + 1;
        printf("[CCD-grow] h_ccd_cpNum=%u > cap=%d -> grow to %d, redo detection\n",
               h_ccd_cpNum, MAX_CCD_COLLITION_PAIRS_NUM, newcap);
        CUDA_SAFE_CALL(cudaFree(_ccd_collisonPairs));
        CUDA_SAFE_CALL(cudaMalloc((void**)&_ccd_collisonPairs, ((size_t)newcap + 1) * sizeof(int4)));
        MAX_CCD_COLLITION_PAIRS_NUM = newcap;
        bvh_f._ccd_collisionPair = _ccd_collisonPairs;
        bvh_e._ccd_collisionPair = _ccd_collisonPairs;
        set_emit_caps(MAX_COLLITION_PAIRS_NUM, MAX_CCD_COLLITION_PAIRS_NUM);
        CUDA_SAFE_CALL(cudaMemsetAsync(_cpNum, 0, sizeof(uint32_t), 0));
        bvh_f.SelfCollitionFullDetect(dHat, _moveDir, alpha);
        bvh_e.SelfCollitionFullDetect(dHat, _moveDir, alpha, m_aux_stream);
        CUDA_SAFE_CALL(cudaStreamSynchronize(m_aux_stream));
        CUDA_SAFE_CALL(cudaMemcpy(&h_ccd_cpNum, _cpNum, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    }
}


// [multi-env determinism] d_bvh_vertexes = _vertexes + d_env_offset (no-op when offset=0).
__global__ void _addEnvOffset(double3* out, const double3* v, const double3* off, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    out[i].x = v[i].x + off[i].x;
    out[i].y = v[i].y + off[i].y;
    out[i].z = v[i].z + off[i].z;
}

void GIPC::buildBVH()
{
    if(m_skip_all_collision)
        return;
    // [multi-env P2] per-env mode builds trees inside buildCP (per-env loop); skip the merged build.
    if(m_perenv_bvh && m_perenv_bvh_groups > 0)
        return;
    { int bs = 256, gs = (vertexNum + bs - 1) / bs;
      _addEnvOffset<<<gs, bs>>>(d_bvh_vertexes, _vertexes, d_env_offset, vertexNum); }
    bvh_f.Construct();
    bvh_e.Construct();
}

// [multi-env cross-env DIAGNOSTIC] scatter a per-vertex buffer into [env*maxL + localid] for
// env0/env1, so the host can compare corresponding entries (identical envs ⇒ should be equal).
__global__ void _xenv_scatter(const double3* buf, const int* p2g, const int* lid,
                              double* out, int maxL, int n)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if(v >= n) return;
    int g = p2g[v];
    if(g < 0 || g > 1) return;
    int L = lid[v];
    if(L < 0 || L >= maxL) return;
    out[((size_t)g * maxL + L) * 3 + 0] = buf[v].x;
    out[((size_t)g * maxL + L) * 3 + 1] = buf[v].y;
    out[((size_t)g * maxL + L) * 3 + 2] = buf[v].z;
}
// [xenv] classify each contact pair: intra-env0 / intra-env1 / cross-env / other. Decodes the
// first two vertices of the int4 pair (gv = c>=0 ? c : -c-1) and compares their groups.
__global__ void _xenv_paircount(const int4* pairs, const int* p2g, int n,
                                int* cnt_g0, int* cnt_g1, int* cross, int* other)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    int4 p = pairs[i];
    int gv0 = (p.x >= 0) ? p.x : (-p.x - 1);
    int gv1 = (p.y >= 0) ? p.y : (-p.y - 1);
    int g0 = p2g[gv0];
    int g1 = p2g[gv1];
    if(g0 != g1)        atomicAdd(cross, 1);
    else if(g0 == 0)    atomicAdd(cnt_g0, 1);
    else if(g0 == 1)    atomicAdd(cnt_g1, 1);
    else                atomicAdd(other, 1);
}
void GIPC::xenvPairClassify(const int4* pairs, int n, const char* label)
{
    if(!getenv("STIFF_XENV") || !m_d_p2g || n <= 0) return;
    static int* d4 = nullptr;
    if(!d4) CUDA_SAFE_CALL(cudaMalloc((void**)&d4, 4 * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(d4, 0, 4 * sizeof(int)));
    int bs = 256, gs = (n + bs - 1) / bs;
    _xenv_paircount<<<gs, bs>>>(pairs, m_d_p2g, n, d4 + 0, d4 + 1, d4 + 2, d4 + 3);
    int h[4]; CUDA_SAFE_CALL(cudaMemcpy(h, d4, 4 * sizeof(int), cudaMemcpyDeviceToHost));
    printf("[xenv]   %s: intra-env0=%d intra-env1=%d CROSS-env=%d other=%d (total=%d)\n",
           label, h[0], h[1], h[2], h[3], n);
}
// max |env0[k]-env1[k]| over corresponding (local-id) vertices. Builds the local-id map once.
double GIPC::xenvDiff(const double3* buf, const char* label)
{
    if(!getenv("STIFF_XENV") || !m_d_p2g) return -1.0;
    if(!m_xenv_ready)
    {
        std::vector<int> p2g(vertexNum);
        CUDA_SAFE_CALL(cudaMemcpy(p2g.data(), m_d_p2g, vertexNum * sizeof(int), cudaMemcpyDeviceToHost));
        std::vector<int> lid(vertexNum, -1), cnt(2, 0);
        for(int v = 0; v < vertexNum; v++)
        { int g = p2g[v]; if(g == 0 || g == 1) lid[v] = cnt[g]++; }
        m_xenv_maxlocal = (cnt[0] > cnt[1]) ? cnt[0] : cnt[1];
        printf("[xenv] env0=%d env1=%d verts (maxlocal=%d)\n", cnt[0], cnt[1], m_xenv_maxlocal);
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_xenv_lid, vertexNum * sizeof(int)));
        CUDA_SAFE_CALL(cudaMemcpy(d_xenv_lid, lid.data(), vertexNum * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_xenv_buf, (size_t)2 * m_xenv_maxlocal * 3 * sizeof(double)));
        m_xenv_ready = true;
    }
    CUDA_SAFE_CALL(cudaMemset(d_xenv_buf, 0, (size_t)2 * m_xenv_maxlocal * 3 * sizeof(double)));
    { int bs = 256, gs = (vertexNum + bs - 1) / bs;
      _xenv_scatter<<<gs, bs>>>(buf, m_d_p2g, d_xenv_lid, d_xenv_buf, m_xenv_maxlocal, vertexNum); }
    std::vector<double> h((size_t)2 * m_xenv_maxlocal * 3);
    CUDA_SAFE_CALL(cudaMemcpy(h.data(), d_xenv_buf, h.size() * sizeof(double), cudaMemcpyDeviceToHost));
    double mx = 0.0; int worst = -1;
    for(int L = 0; L < m_xenv_maxlocal; L++)
        for(int c = 0; c < 3; c++)
        { double d = fabs(h[((size_t)0 * m_xenv_maxlocal + L) * 3 + c] - h[((size_t)1 * m_xenv_maxlocal + L) * 3 + c]);
          if(d > mx) { mx = d; worst = L; } }
    // [env-det dbg] identify the worst lid: env0 global vert + btype (boundary/driven vs free).
    // STIFF_XENV_ID: build env0 lid→vert inverse once, report on first nonzero diff.
    if(getenv("STIFF_XENV_ID") && worst >= 0 && mx > 0.0)
    {
        static std::vector<int> inv;
        if(inv.empty())
        {
            inv.assign(m_xenv_maxlocal, -1);
            std::vector<int> hp(vertexNum), hl(vertexNum);
            CUDA_SAFE_CALL(cudaMemcpy(hp.data(), m_d_p2g, (size_t)vertexNum*sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(hl.data(), d_xenv_lid, (size_t)vertexNum*sizeof(int), cudaMemcpyDeviceToHost));
            for(int v=0; v<vertexNum; v++) if(hp[v]==0 && hl[v]>=0 && hl[v]<m_xenv_maxlocal) inv[hl[v]]=v;
        }
        printf("[xenv-id] %-20s worst lid=%d -> env0 global vert=%d (vertexNum=%d)\n",
               label, worst, inv[worst], vertexNum);
    }
    printf("[xenv] %-22s maxdiff %.6e  (worst lid=%d)\n", label, mx, worst);
    return mx;
}

// [multi-env P2 / per-env BVH] primitive→env via the first vertex's group.
__global__ void _prim_env_f(const uint3* faces, const int* p2g, int* env, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    env[i] = p2g[faces[i].x];
}
__global__ void _prim_env_e(const uint2* edges, const int* p2g, int* env, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    env[i] = p2g[edges[i].x];
}
// P1: group faces/edges by env (host, one-time; topology static) → env-contiguous _active_idx.
static void _build_perenv_list(const int* d_env, int n, int NG,
                               int*& d_idx, std::vector<int>& off, std::vector<int>& cnt,
                               const std::vector<uint64_t>* key = nullptr)
{
    std::vector<int> h_env(n);
    CUDA_SAFE_CALL(cudaMemcpy(h_env.data(), d_env, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost));
    off.assign(NG, 0);
    cnt.assign(NG, 0);
    int ungrouped = 0;
    for(int i = 0; i < n; i++) { int e = h_env[i]; if(e >= 0 && e < NG) cnt[e]++; else ungrouped++; }
    int acc = 0;
    for(int e = 0; e < NG; e++) { off[e] = acc; acc += cnt[e]; }
    std::vector<int> idx(acc), cur(off);
    for(int i = 0; i < n; i++) { int e = h_env[i]; if(e >= 0 && e < NG) idx[cur[e]++] = i; }
    // [env-det BVH] order each env's active list by an ENV-LOCAL key so ALL envs share an identical
    // local prim ordering (mirror) ⇒ the per-env Construct sees identical inputs ⇒ identical trees.
    // (default builds in ascending global-prim order, which is NOT env-mirror for co-located envs.)
    if(key && getenv("STIFF_BVH_ENVDET"))
        for(int e = 0; e < NG; e++)
        { int s = off[e], c = cnt[e];
          std::sort(idx.begin() + s, idx.begin() + s + c,
                    [&](int a, int b){ return (*key)[a] < (*key)[b]; }); }
    if(ungrouped) printf("[perenv-bvh] WARNING %d/%d prims ungrouped (env<0) — excluded from per-env BVH\n", ungrouped, n);
    if(d_idx) CUDA_SAFE_CALL(cudaFree(d_idx));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_idx, (size_t)(acc > 0 ? acc : 1) * sizeof(int)));
    if(acc) CUDA_SAFE_CALL(cudaMemcpy(d_idx, idx.data(), (size_t)acc * sizeof(int), cudaMemcpyHostToDevice));
}
// [env-det] enable env-major Morton on the MERGED BVH: compute per-prim env id (p2g of the prim's
// first vertex) once, point the BVHs at it, and turn on the env-major sort key. Co-located identical
// envs then build env-blocked (mirror) trees ⇒ env-symmetric broad-phase enumeration.
void GIPC::enableEnvMajorBVH(const int* p2g)
{
    if(!p2g || d_face_env) return;   // once
    m_d_p2g = p2g;
    int nF = (int)bvh_f.face_number, nE = (int)bvh_e.edge_number;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_face_env, (size_t)nF * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_edge_env, (size_t)nE * sizeof(int)));
    { int bs=256, gs=(nF+bs-1)/bs; _prim_env_f<<<gs,bs>>>(bvh_f._faces, p2g, d_face_env, nF); }
    { int bs=256, gs=(nE+bs-1)/bs; _prim_env_e<<<gs,bs>>>(bvh_e._edges, p2g, d_edge_env, nE); }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    bvh_f.m_prim_env = d_face_env;
    bvh_e.m_prim_env = d_edge_env;
    // [env-det] Morton low-bits tie-break = a MIRROR key derived from the prim's env-LOCAL VERTEX ids
    // (the env-local vertex id IS mirror across identical envs; the global prim/edge numbering is NOT).
    // env-local vert id = rank of a vertex among its env's verts by ascending global index.
    std::vector<int> hp2(vertexNum);
    CUDA_SAFE_CALL(cudaMemcpy(hp2.data(), p2g, (size_t)vertexNum*sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<int> vloc(vertexNum, 0); { std::vector<int> ec;
        for(int v=0;v<vertexNum;v++){ int g=hp2[v]; if(g<0) continue; if(g>=(int)ec.size()) ec.resize(g+1,0); vloc[v]=ec[g]++; } }
    std::vector<uint3> hf(nF); std::vector<uint2> he(nE);
    CUDA_SAFE_CALL(cudaMemcpy(hf.data(), bvh_f._faces, (size_t)nF*sizeof(uint3), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(he.data(), bvh_e._edges, (size_t)nE*sizeof(uint2), cudaMemcpyDeviceToHost));
    std::vector<int> floc(nF), eloc(nE);
    for(int i=0;i<nE;i++){ int a=vloc[he[i].x], b=vloc[he[i].y]; int lo=a<b?a:b, hi=a<b?b:a;
        eloc[i] = (lo<<13)|hi; }                              // 2×13-bit env-local vert ids (verts/env<8192)
    for(int i=0;i<nF;i++){ int a=vloc[hf[i].x],b=vloc[hf[i].y],c=vloc[hf[i].z];
        if(a>b){int t=a;a=b;b=t;} if(b>c){int t=b;b=c;c=t;} if(a>b){int t=a;a=b;b=t;}
        floc[i] = (int)((((uint64_t)a*9973u + b)*9973u + c) & 0x3FFFFFFu); }  // mirror hash (26-bit)
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_face_localid,(size_t)nF*sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_edge_localid,(size_t)nE*sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpy(d_face_localid,floc.data(),(size_t)nF*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_edge_localid,eloc.data(),(size_t)nE*sizeof(int),cudaMemcpyHostToDevice));
    bvh_f.m_prim_localid = d_face_localid;
    bvh_e.m_prim_localid = d_edge_localid;
    // [env-det] static per-prim first-vertex index so _calcMChash can subtract the LIVE per-env world
    // offset (d_env_offset is populated per-frame, AFTER this once-call) ⇒ Morton computed in the
    // local frame ⇒ mirror trees, while spacing>0 is kept for broad-phase efficiency.
    std::vector<uint32_t> fv0(nF), ev0(nE);
    for(int i=0;i<nF;i++) fv0[i]=hf[i].x;
    for(int i=0;i<nE;i++) ev0[i]=he[i].x;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_face_v0,(size_t)nF*sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_edge_v0,(size_t)nE*sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemcpy(d_face_v0,fv0.data(),(size_t)nF*sizeof(uint32_t),cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_edge_v0,ev0.data(),(size_t)nE*sizeof(uint32_t),cudaMemcpyHostToDevice));
    bvh_f.m_env_offset = d_env_offset; bvh_f.m_prim_v0 = d_face_v0;
    bvh_e.m_env_offset = d_env_offset; bvh_e.m_prim_v0 = d_edge_v0;
    set_bvh_envmajor(1);
    printf("[env-major-bvh] enabled: %d faces, %d edges\n", nF, nE);
}
void GIPC::buildPerEnvBVHIndex(int NG, const int* p2g)
{
    m_perenv_bvh_groups = NG;
    int nF = (int)bvh_f.face_number, nE = (int)bvh_e.edge_number;
    int *d_fenv = nullptr, *d_eenv = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_fenv, (size_t)nF * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_eenv, (size_t)nE * sizeof(int)));
    { int bs = 256, gs = (nF + bs - 1) / bs; _prim_env_f<<<gs, bs>>>(bvh_f._faces, p2g, d_fenv, nF); }
    { int bs = 256, gs = (nE + bs - 1) / bs; _prim_env_e<<<gs, bs>>>(bvh_e._edges, p2g, d_eenv, nE); }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    // [env-det BVH] per-prim ENV-LOCAL sort key = packed sorted env-local vertex ids. Env-local id =
    // rank of a vertex among its env's vertices by ascending GLOBAL index (mirror across identical
    // envs, proven by xenvDiff call#0==0). Used to canonicalize the per-env active-list ordering.
    std::vector<uint64_t> fkey, ekey;
    if(getenv("STIFF_BVH_ENVDET"))
    {
        std::vector<int> hp2g(vertexNum);
        CUDA_SAFE_CALL(cudaMemcpy(hp2g.data(), p2g, (size_t)vertexNum * sizeof(int), cudaMemcpyDeviceToHost));
        std::vector<int> vloc(vertexNum, 0); std::vector<int> ec(NG, 0);
        for(int v = 0; v < vertexNum; v++){ int g = hp2g[v]; if(g >= 0 && g < NG) vloc[v] = ec[g]++; }
        auto pack3 = [](uint64_t a, uint64_t b, uint64_t c){
            uint64_t lo=a<b?a:b, hi=a<b?b:a; if(c<lo){uint64_t t=lo;lo=c;c=t;} if(c<hi){uint64_t t=hi;hi=c;c=t;}
            return (lo<<40)|(hi<<20)|c; };
        std::vector<uint3> hf(nF); std::vector<uint2> he(nE);
        CUDA_SAFE_CALL(cudaMemcpy(hf.data(), bvh_f._faces, (size_t)nF*sizeof(uint3), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(he.data(), bvh_e._edges, (size_t)nE*sizeof(uint2), cudaMemcpyDeviceToHost));
        fkey.resize(nF); ekey.resize(nE);
        for(int i=0;i<nF;i++) fkey[i]=pack3(vloc[hf[i].x],vloc[hf[i].y],vloc[hf[i].z]);
        for(int i=0;i<nE;i++){ uint64_t a=vloc[he[i].x],b=vloc[he[i].y]; ekey[i]=(a<b)?((a<<20)|b):((b<<20)|a); }
    }
    _build_perenv_list(d_fenv, nF, NG, d_perenv_face_idx, h_perenv_face_off, h_perenv_face_cnt, fkey.empty()?nullptr:&fkey);
    _build_perenv_list(d_eenv, nE, NG, d_perenv_edge_idx, h_perenv_edge_off, h_perenv_edge_cnt, ekey.empty()?nullptr:&ekey);
    CUDA_SAFE_CALL(cudaFree(d_fenv));
    CUDA_SAFE_CALL(cudaFree(d_eenv));
    int tf = 0, te = 0;
    h_perenv_active.clear();
    for(int e = 0; e < NG; e++)
    {
        tf += h_perenv_face_cnt[e]; te += h_perenv_edge_cnt[e];
        if(h_perenv_face_cnt[e] > 0 || h_perenv_edge_cnt[e] > 0) h_perenv_active.push_back(e);
    }
    printf("[perenv-bvh] built per-env index: NG=%d active=%zu faces %d/%d edges %d/%d\n",
           NG, h_perenv_active.size(), tf, nF, te, nE);
}

// P2: per-env Construct+Detect on LOCAL _vertexes (full precision, per-env identical, no cross-
// env candidates) — replaces buildBVH()+buildCP() when m_perenv_bvh. Pairs append to the shared
// _collisonPairs via the atomic _cpNum (env-order-independent, consumers iterate 0..h_cpNum[0]).
// [perenv-parallel #1] allocate K scratch sets (face + edge sized) + K streams (once).
void GIPC::allocPerEnvPool(int K)
{
    if(m_pool_K >= K) return;
    int nF = (int)bvh_f.face_number, nE = (int)bvh_e.edge_number;
    m_pool_f.resize(K); m_pool_e.resize(K);
    // [perenv-parallel #2] per-slot cub sort scratch, pre-sized to the FULL prim count so the
    // in-loop ensure_sort_scratch never reallocates (pointer stability across swapIn/restore).
    auto allocSort = [](GIPC::BvhScratch& s, int cap)
    {
        if(s.sort_tmp) return;
        size_t bytes = 0;
        cub::DeviceRadixSort::SortPairs((void*)nullptr, bytes, (const uint64_t*)nullptr,
                                        (uint64_t*)nullptr, (const uint32_t*)nullptr,
                                        (uint32_t*)nullptr, cap, 0, 64);
        CUDA_SAFE_CALL(cudaMalloc(&s.sort_tmp, bytes));
        CUDA_SAFE_CALL(cudaMalloc((void**)&s.mch_alt, (size_t)cap * sizeof(uint64_t)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&s.idx_alt, (size_t)cap * sizeof(uint32_t)));
        s.sort_bytes = bytes; s.sort_cap = cap;
    };
    for(int k = 0; k < K; ++k)
    {
        if(!m_pool_f[k].nodes) { BvhScratch& s = m_pool_f[k];
            CUDA_SAFE_CALL(cudaMalloc((void**)&s.nodes,(2*nF-1)*sizeof(Node)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&s.bvs,(2*nF-1)*sizeof(AABB)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&s.mch,nF*sizeof(uint64_t)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&s.idx,nF*sizeof(uint32_t)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&s.tmp,nF*sizeof(AABB)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&s.flags,(nF-1)*sizeof(uint32_t)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&s.node_env,(2*nF-1)*sizeof(int)));
            allocSort(s, nF); }
        if(!m_pool_e[k].nodes) { BvhScratch& s = m_pool_e[k];
            CUDA_SAFE_CALL(cudaMalloc((void**)&s.nodes,(2*nE-1)*sizeof(Node)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&s.bvs,(2*nE-1)*sizeof(AABB)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&s.mch,nE*sizeof(uint64_t)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&s.idx,nE*sizeof(uint32_t)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&s.tmp,nE*sizeof(AABB)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&s.flags,(nE-1)*sizeof(uint32_t)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&s.node_env,(2*nE-1)*sizeof(int)));
            allocSort(s, nE); }
    }
    m_pool_streams.resize(K);
    for(int k = 0; k < K; ++k) CUDA_SAFE_CALL(cudaStreamCreate(&m_pool_streams[k]));
    m_pool_K = K;
}

void GIPC::buildBVH_and_CP_perenv(double dHat)
{
    if(m_skip_all_collision) return;
    int NG = m_perenv_bvh_groups;
    // point the BVH at LOCAL verts (the whole reason this kills cross-env divergence)
    double3* saved_f = bvh_f._vertexes;
    double3* saved_e = bvh_e._vertexes;
    bvh_f._vertexes = _vertexes;
    bvh_e._vertexes = _vertexes;
    (void)NG;
    bool skipF = getenv("STIFF_SKIP_F"), skipE = getenv("STIFF_SKIP_E");  // [decomp] per-env skips
    // [perenv-parallel #1] STIFF_PERENV_PAR: run envs concurrently on a K-stream scratch pool.
    // Value-aware (=0 disables) — isolated mode defaults it ON via the Python resolver.
    const char* _pe_par = getenv("STIFF_PERENV_PAR");
    bool par = _pe_par && atoi(_pe_par) != 0;
    int  K   = 1;
    if(par) { int cap = getenv("STIFF_PERENV_K") ? atoi(getenv("STIFF_PERENV_K")) : 8;  // concurrency cap
              K = (int)h_perenv_active.size(); if(K > cap) K = cap; if(K < 1) K = 1; allocPerEnvPool(K); }
    // snapshot bvh scratch so we can point-swap per env + restore at the end.
    BvhScratch of{bvh_f._nodes,bvh_f._bvs,bvh_f._MChash,bvh_f._indices,bvh_f._tempLeafBox,bvh_f._flags,bvh_f.m_node_env,
                  bvh_f._sort_tmp,bvh_f._sort_tmp_bytes,bvh_f._mch_alt,bvh_f._idx_alt,bvh_f._sort_cap};
    BvhScratch oe{bvh_e._nodes,bvh_e._bvs,bvh_e._MChash,bvh_e._indices,bvh_e._tempLeafBox,bvh_e._flags,bvh_e.m_node_env,
                  bvh_e._sort_tmp,bvh_e._sort_tmp_bytes,bvh_e._mch_alt,bvh_e._idx_alt,bvh_e._sort_cap};
    auto swapIn = [](lbvh& b, BvhScratch& s){ b._nodes=s.nodes; b._bvs=s.bvs; b._MChash=s.mch;
        b._indices=s.idx; b._tempLeafBox=s.tmp; b._flags=s.flags; b.m_node_env=s.node_env;
        b._sort_tmp=s.sort_tmp; b._sort_tmp_bytes=s.sort_bytes;   // [perenv-parallel #2]
        b._mch_alt=s.mch_alt; b._idx_alt=s.idx_alt; b._sort_cap=s.sort_cap; };
  perenv_redo:
    CUDA_SAFE_CALL(cudaMemset(_cpNum, 0, 5 * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemset(_gpNum, 0, sizeof(uint32_t)));
    // [FIX] memsets are on DEFAULT stream; per-env detects on POOL streams. Sync once so the zero is
    // globally visible before any pool-stream detect atomicAdds to _cpNum (else garbage slot -> OOB).
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    int i = 0;
    for(int e : h_perenv_active)
    {
        cudaStream_t st = par ? m_pool_streams[i % K] : (cudaStream_t)0;
        if(h_perenv_face_cnt[e] > 0 && !skipF)
        {
            if(par) swapIn(bvh_f, m_pool_f[i % K]);
            bvh_f._active_idx        = d_perenv_face_idx + h_perenv_face_off[e];
            bvh_f.face_number_active = h_perenv_face_cnt[e];
            bvh_f.Construct(st);
            bvh_f.SelfCollitionDetect(dHat, st);
        }
        if(h_perenv_edge_cnt[e] > 0 && !skipE)
        {
            if(par) swapIn(bvh_e, m_pool_e[i % K]);
            bvh_e._active_idx        = d_perenv_edge_idx + h_perenv_edge_off[e];
            bvh_e.face_number_active = h_perenv_edge_cnt[e];  // base member reused as edge active count
            bvh_e.Construct(st);
            bvh_e.SelfCollitionDetect(dHat, st);
        }
        ++i;
    }
    if(par) { for(int k = 0; k < K; ++k) CUDA_SAFE_CALL(cudaStreamSynchronize(m_pool_streams[k]));
              swapIn(bvh_f, of); swapIn(bvh_e, oe); }  // restore original scratch
    CUDA_SAFE_CALL(cudaMemcpy(&h_cpNum, _cpNum, 5 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    // [perenv-parallel #1 FIX] the per-env path (like the merged path) MUST grow the pair buffers on
    // overflow + redo — else at large N the pair count exceeds the cap and consumers (line-search,
    // gradient) read OOB → illegal access. Per-env DCD fills BOTH _collisonPairs and _ccd_collisonPairs
    // (1:1), so grow both. Emits past cap went to the trash slot, so nothing was corrupted.
    if((int)h_cpNum[0] > MAX_COLLITION_PAIRS_NUM || (int)h_cpNum[0] > MAX_CCD_COLLITION_PAIRS_NUM)
    {
        int newcap = (int)(h_cpNum[0] + h_cpNum[0] / 2) + 1;
        printf("[perenv DCD-grow] h_cpNum=%u > cap(dcd=%d,ccd=%d) -> grow to %d, redo\n",
               h_cpNum[0], MAX_COLLITION_PAIRS_NUM, MAX_CCD_COLLITION_PAIRS_NUM, newcap);
        CUDA_SAFE_CALL(cudaFree(_collisonPairs));
        CUDA_SAFE_CALL(cudaFree(_MatIndex));
        CUDA_SAFE_CALL(cudaFree(_ccd_collisonPairs));
        CUDA_SAFE_CALL(cudaMalloc((void**)&_collisonPairs,     ((size_t)newcap + 1) * sizeof(int4)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&_MatIndex,          ((size_t)newcap + 1) * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&_ccd_collisonPairs, ((size_t)newcap + 1) * sizeof(int4)));
        MAX_COLLITION_PAIRS_NUM     = newcap;
        if(newcap > MAX_CCD_COLLITION_PAIRS_NUM) MAX_CCD_COLLITION_PAIRS_NUM = newcap;
        bvh_f._collisionPair     = bvh_e._collisionPair     = _collisonPairs;
        bvh_f._ccd_collisionPair = bvh_e._ccd_collisionPair = _ccd_collisonPairs;
        bvh_f._MatIndex          = bvh_e._MatIndex          = _MatIndex;
        set_emit_caps(MAX_COLLITION_PAIRS_NUM, MAX_CCD_COLLITION_PAIRS_NUM);
        goto perenv_redo;
    }
    bvh_f._active_idx = nullptr; bvh_f.face_number_active = 0;
    bvh_e._active_idx = nullptr; bvh_e.face_number_active = 0;
    bvh_f._vertexes = saved_f;
    bvh_e._vertexes = saved_e;
    if(!getenv("STIFF_SKIP_GRND")) GroundCollisionDetect();
    {   // [9d28824-port] one 6-int D2H
        uint32_t cp_gp_buf[6];
        CUDA_SAFE_CALL(cudaMemcpy(cp_gp_buf, _cpNum, 6 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        memcpy(h_cpNum, cp_gp_buf, 5 * sizeof(uint32_t));
        h_gpNum = cp_gp_buf[5];
    }
}

AABB* GIPC::calcuMaxSceneSize()
{
    return bvh_f.getSceneSize();
}

void GIPC::buildBVH_FULLCCD(const double& alpha)
{
    if(m_skip_all_collision)
        return;
    // [multi-env P2] per-env mode builds swept trees inside buildFullCP; skip merged build.
    if(m_perenv_bvh && m_perenv_bvh_groups > 0)
        return;
    { int bs = 256, gs = (vertexNum + bs - 1) / bs;
      _addEnvOffset<<<gs, bs>>>(d_bvh_vertexes, _vertexes, d_env_offset, vertexNum); }
    bvh_f.ConstructFullCCD(_moveDir, alpha);
    bvh_e.ConstructFullCCD(_moveDir, alpha);
}

void GIPC::calBarrierGradientAndHessian(double3* _gradient, double mKappa)
{
    int numbers = h_cpNum[0];
    if(numbers < 1)
        return;
    const unsigned int threadNum = 256;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    _calBarrierGradientAndHessian<<<blockNum, threadNum>>>(
        _vertexes,
        _rest_vertexes,
        _collisonPairs,
        _gradient,
        gipc_global_triplet.block_values(),
        gipc_global_triplet.block_row_indices(),
        gipc_global_triplet.block_col_indices(),
        _cpNum,
        _MatIndex,
        dHat,
        mKappa,
        h_cpNum[4],
        h_cpNum[3],
        h_cpNum[2],
        numbers,
        m_pergroup_kappa ? m_kappa_group : nullptr,   // [per-group κ] nullptr → scalar
        m_pergroup_kappa ? m_d_p2g : nullptr);
}


void GIPC::calBarrierHessian()
{

    int numbers = h_cpNum[0];
    if(numbers < 1)
        return;
    const unsigned int threadNum = 32;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //

    _calBarrierHessian<<<blockNum, threadNum>>>(_vertexes,
                                                _rest_vertexes,
                                                _collisonPairs,
                                                gipc_global_triplet.block_values(),
                                                gipc_global_triplet.block_row_indices(),
                                                gipc_global_triplet.block_col_indices(),
                                                _cpNum,
                                                _MatIndex,
                                                dHat,
                                                Kappa,
                                                h_cpNum[4],
                                                h_cpNum[3],
                                                h_cpNum[2],
                                                numbers);
}

static void _dbg_ksum(const char*, const void*, size_t);            // [4.3 fwd]
static void _dbg_ksum_comm(const char*, const void*, size_t);       // [4.3 fwd]

void GIPC::calFrictionHessian(device_TetraData& TetMesh)
{
    int numbers = h_cpNum_last[0];
    //if (numbers < 1) return;
    const unsigned int threadNum = 256;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    if(numbers > 0)
    {
        if(getenv("STIFF_KSUM"))
        {
            cudaDeviceSynchronize();
            _dbg_ksum("fric_o_verts", TetMesh.o_vertexes, (size_t)vertexNum * sizeof(double3));
            _dbg_ksum("fric_cur_verts", _vertexes, (size_t)vertexNum * sizeof(double3));
            _dbg_ksum_comm("fric_distCoord", distCoord, (size_t)numbers * sizeof(double2));
            _dbg_ksum_comm("fric_tanBasis", tanBasis, (size_t)numbers * sizeof(__GEIGEN__::Matrix3x2d));
            _dbg_ksum_comm("fric_lambdaH", lambda_lastH_scalar, (size_t)numbers * sizeof(double));
            _dbg_ksum_comm("fric_pairs", _collisonPairs_lastH, (size_t)numbers * sizeof(int4));
        }
        _calFrictionHessian<<<blockNum, threadNum>>>(
            _vertexes,
            TetMesh.o_vertexes,
            _collisonPairs_lastH,
            gipc_global_triplet.block_values(),
            gipc_global_triplet.block_row_indices(),
            gipc_global_triplet.block_col_indices(),
            _cpNum,
            numbers,
            IPC_dt,
            distCoord,
            tanBasis,
            fDhat * IPC_dt * IPC_dt,
            lambda_lastH_scalar,
            frictionRate,
            h_cpNum[4],
            h_cpNum[3],
            h_cpNum[2],
            h_cpNum_last[4],
            h_cpNum_last[3],
            h_cpNum_last[2]);
    }

    numbers = h_gpNum_last;
    CUDA_SAFE_CALL(cudaMemcpy(_gpNum, &h_gpNum_last, sizeof(uint32_t), cudaMemcpyHostToDevice));
    if(numbers < 1)
        return;

    blockNum = (numbers + threadNum - 1) / threadNum;
    int global_offset = gipc_global_triplet.global_triplet_offset + h_cpNum_last[4] * M12_Off
                        + h_cpNum_last[3] * M9_Off + h_cpNum_last[2] * M6_Off;
    _calFrictionHessian_gd<<<blockNum, threadNum>>>(
        _vertexes,
        TetMesh.o_vertexes,
        _groundNormal,
        _collisonPairs_lastH_gd,
        gipc_global_triplet.block_values(),
        gipc_global_triplet.block_row_indices(),
        gipc_global_triplet.block_col_indices(),
        numbers,
        IPC_dt,
        fDhat * IPC_dt * IPC_dt,
        lambda_lastH_scalar_gd,
        global_offset,
        gd_frictionRate);
}

void GIPC::computeSelfCloseVal()
{
    int numbers = h_cpNum[0];
    if(numbers <= 0)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calSelfCloseVal<<<blockNum, threadNum>>>(
        _vertexes, _collisonPairs, _closeMConstraintID, _closeMConstraintVal, _close_cpNum, dTol, numbers);
    // NOTE: h_close_cpNum intentionally not synced (same as h_close_gpNum above).
}

bool GIPC::checkSelfCloseVal()
{
    int numbers = h_close_cpNum;
    if(numbers < 1)
        return false;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    int*               _isChange;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_isChange, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(_isChange, 0, sizeof(int)));
    _checkSelfCloseVal<<<blockNum, threadNum>>>(
        _vertexes, _isChange, _closeMConstraintID, _closeMConstraintVal, numbers,
        m_pergroup_kappa ? m_d_close_grp : nullptr, m_pergroup_kappa ? m_d_p2g : nullptr);
    int isChange;
    CUDA_SAFE_CALL(cudaMemcpy(&isChange, _isChange, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(_isChange));

    return (isChange == 1);
}

double2 GIPC::minMaxSelfDist()
{
    int numbers = h_cpNum[0];
    if(numbers < 1)
        return make_double2(1e32, 0);
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double2) * (threadNum >> 5);

    double2* _queue;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_queue, numbers * sizeof(double2)));
    //CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number * sizeof(AABB), cudaMemcpyDeviceToDevice));
    _reduct_MSelfDist<<<blockNum, threadNum, sharedMsize>>>(
        _vertexes, _collisonPairs, _queue, numbers);
    //_reduct_min_double3_to_double << <blockNum, threadNum, sharedMsize >> > (_moveDir, _tempMinMovement, numbers);

    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while(numbers > 1)
    {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        _reduct_M_double2<<<blockNum, threadNum, sharedMsize>>>(_queue, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double2 minValue;
    cudaMemcpy(&minValue, _queue, sizeof(double2), cudaMemcpyDeviceToHost);
    CUDA_SAFE_CALL(cudaFree(_queue));
    minValue.x = 1.0 / minValue.x;
    return minValue;
}

// void GIPC::calBarrierGradient(double3* _gradient, double mKappa) {
//     int numbers = h_cpNum[0];
//     if (numbers < 1)return;
//     const unsigned int threadNum = 256;
//     int blockNum = (numbers + threadNum - 1) / threadNum;
//     _calBarrierGradient << <blockNum, threadNum >> > (_vertexes, _rest_vertexes, _collisonPairs, _gradient, dHat, mKappa, numbers);
// }

// ===================== per-contact force export (Step B) =====================
// Ground contacts use the simple distance barrier (lambda * ground_normal).
// Body-body & FEM-coupled contacts reuse the exact I5/NEWF barrier gradient
// via the _calBarrierGradient per-contact hook (see _ec_emit).

__global__ void _exportGroundContactForces(const double3*   _vertexes,
                                           const uint32_t*  envPair,
                                           const double3*   g_normal,
                                           const double*    g_offset,
                                           const int*       _point_body_id,
                                           double Kappa, double dHat, double dt,
                                           int number, int base,
                                           int2* out_pair, double3* out_force)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int    gidx = (int)envPair[idx];
    double3 nrm = g_normal[0];
    double  dist = __GEIGEN__::__v_vec_dot(nrm, _vertexes[gidx]) - g_offset[0];
    double  dis  = dist * dist;
    if(dis < 1e-12)
        dis = 1e-12;  // [d=0 guard]
    double t      = dis - dHat;
    double g_b    = t * log(dis / dHat) * -2.0 - (t * t) / dis;
    double lambda = -Kappa * 2.0 * sqrt(dis) * g_b;
    double c      = lambda / (dt * dt);
    out_force[base + idx] = make_double3(c * nrm.x, c * nrm.y, c * nrm.z);
    out_pair[base + idx]  = make_int2(_point_body_id[gidx], -1);
}

int GIPC::exportContacts(int2* out_pair, double3* out_force)
{
    int ncp = (int)h_cpNum[0];
    int ngp = (int)h_gpNum;
    const unsigned int threads = 256;
    double inv_dt2 = (IPC_dt > 0.0) ? 1.0 / (IPC_dt * IPC_dt) : 0.0;
    // body-body: reuse the EXACT I5/NEWF barrier gradient (calBarrierGradient)
    // with the per-contact export hook. Needs a per-vertex scratch gradient.
    if(ncp > 0)
    {
        if(vertexNum > _ec_scratch_cap)
        {
            if(_ec_grad_scratch) CUDA_SAFE_CALL(cudaFree(_ec_grad_scratch));
            _ec_scratch_cap = vertexNum;
            CUDA_SAFE_CALL(cudaMalloc((void**)&_ec_grad_scratch, _ec_scratch_cap * sizeof(double3)));
        }
        CUDA_SAFE_CALL(cudaMemset(_ec_grad_scratch, 0, vertexNum * sizeof(double3)));
        // default body-body entries to "skip" (bodyA<0); _ec_emit overwrites
        // the ones it attributes.
        CUDA_SAFE_CALL(cudaMemset(out_pair, 0xFF, ncp * sizeof(int2)));   // -1,-1
        CUDA_SAFE_CALL(cudaMemset(out_force, 0, ncp * sizeof(double3)));
        calBarrierGradient(_ec_grad_scratch, Kappa, out_pair, out_force, _point_body_id, inv_dt2);
    }
    // ground: simple distance barrier (lambda * ground_normal)
    if(ngp > 0)
    {
        int blocks = (ngp + threads - 1) / threads;
        _exportGroundContactForces<<<blocks, threads>>>(
            _vertexes, _environment_collisionPair, _groundNormal, _groundOffset,
            _point_body_id, Kappa, dHat, IPC_dt, ngp, ncp, out_pair, out_force);
    }
    return ncp + ngp;
}

void GIPC::calBarrierGradient(double3* _gradient, double mKappa,
                              int2* ec_pair, double3* ec_force,
                              const int* ec_pbid, double ec_inv_dt2)
{
    int numbers = h_cpNum[0];
    if(numbers < 1)
        return;
    const unsigned int threadNum = 256;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;


    _calBarrierGradient<<<blockNum, threadNum>>>(
        _vertexes, _rest_vertexes, _collisonPairs, _gradient, dHat, mKappa, numbers,
        m_pergroup_kappa ? m_kappa_group : nullptr, m_pergroup_kappa ? m_d_p2g : nullptr,
        ec_pair, ec_force, ec_pbid, ec_inv_dt2);
}

void GIPC::calFrictionGradient(double3* _gradient, device_TetraData& TetMesh)
{
    int                numbers   = h_cpNum_last[0];
    const unsigned int threadNum = 256;
    int                blockNum  = 0;
    if(numbers > 0)
    {
        blockNum = (numbers + threadNum - 1) / threadNum;
        _calFrictionGradient<<<blockNum, threadNum>>>(_vertexes,
                                                      TetMesh.o_vertexes,
                                                      _collisonPairs_lastH,
                                                      _gradient,
                                                      numbers,
                                                      IPC_dt,
                                                      distCoord,
                                                      tanBasis,
                                                      fDhat * IPC_dt * IPC_dt,
                                                      lambda_lastH_scalar,
                                                      frictionRate);
    }
    numbers = h_gpNum_last;
    if(numbers < 1)
        return;
    blockNum = (numbers + threadNum - 1) / threadNum;

    _calFrictionGradient_gd<<<blockNum, threadNum>>>(_vertexes,
                                                     TetMesh.o_vertexes,
                                                     _groundNormal,
                                                     _collisonPairs_lastH_gd,
                                                     _gradient,
                                                     numbers,
                                                     IPC_dt,
                                                     fDhat * IPC_dt * IPC_dt,
                                                     lambda_lastH_scalar_gd,
                                                     gd_frictionRate);
}


void calKineticGradient(double3* _vertexes, double3* _xTilta, double3* _gradient, double* _masses, int numbers)
{
    const unsigned int threadNum = default_threads;
    if(numbers < 1)
        return;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _calKineticGradient<<<blockNum, threadNum>>>(_vertexes, _xTilta, _gradient, _masses, numbers);
}


void calculate_fem_gradient_hessian(__GEIGEN__::Matrix3x3d* DmInverses,
                                    const double3*          vertexes,
                                    const uint4*            tetrahedras,
                                    const double*           volume,
                                    double3*                gradient,
                                    int                     tetrahedraNum_FEM,
                                    int                     tetrahedraNum_ABD,
                                    const double*           lenRate,
                                    const double*           volRate,
                                    int                     global_offset,
                                    Eigen::Matrix3d*        triplet_values,
                                    int*                    row_ids,
                                    int*                    col_ids,
                                    double                  IPC_dt,
                                    int global_hessian_fem_offset,
                                    const int*              tet_to_abd_body /* nullable, indexed in GLOBAL tet space */)
{
    int numbers = tetrahedraNum_FEM;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calculate_fem_gradient_hessian<<<blockNum, threadNum>>>(
        DmInverses + tetrahedraNum_ABD,
        vertexes,
        tetrahedras + tetrahedraNum_ABD,
        volume + tetrahedraNum_ABD,
        gradient,
        numbers,
        lenRate + tetrahedraNum_ABD,
        volRate + tetrahedraNum_ABD,
        //tet_ids,
        global_offset,
        triplet_values,
        row_ids,
        col_ids,
        IPC_dt,
        global_hessian_fem_offset,
        tet_to_abd_body ? tet_to_abd_body + tetrahedraNum_ABD : nullptr);
}

void calculate_triangle_fem_gradient_hessian(__GEIGEN__::Matrix2x2d* triDmInverses,
                                             const double3*   vertexes,
                                             const uint3*     triangles,
                                             const double*    area,
                                             double3*         gradient,
                                             int              triangleNum,
                                             double           stretchStiff,
                                             double           shearStiff,
                                             double           strainRate,
                                             int              global_offset,
                                             Eigen::Matrix3d* triplet_values,
                                             int*             row_ids,
                                             int*             col_ids,
                                             double           IPC_dt,
                                             int global_hessian_fem_offset)
{
    int numbers = triangleNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calculate_triangle_fem_gradient_hessian<<<blockNum, threadNum>>>(triDmInverses,
                                                                      vertexes,
                                                                      triangles,
                                                                      area,
                                                                      gradient,
                                                                      triangleNum,
                                                                      stretchStiff,
                                                                      shearStiff,
                                                                      IPC_dt,
                                                                      global_offset,
                                                                      triplet_values,
                                                                      row_ids,
                                                                      col_ids,
                                                                      strainRate,
                                                                      global_hessian_fem_offset);
}


void calculate_triangle_fem_strain_limiting_gradient_hessian(__GEIGEN__::Matrix2x2d* triDmInverses,
                                                             const double3* vertexes,
                                                             const uint3* triangles,
                                                             __GEIGEN__::Matrix9x9d* Hessians,
                                                             const uint32_t& offset,
                                                             const double* area,
                                                             double3* gradient,
                                                             int triangleNum,
                                                             Eigen::Matrix3d* U3x2,
                                                             Eigen::Matrix2d* V3x2,
                                                             Eigen::Vector2d* S3x2,
                                                             double IPC_dt)
{
    int numbers = triangleNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calculate_triangle_fem_strain_limiting_gradient_hessian<<<blockNum, threadNum>>>(
        triDmInverses, vertexes, triangles, Hessians, offset, area, gradient, triangleNum, U3x2, V3x2, S3x2, IPC_dt);
}


void calculate_triangle_fem_deformationF(__GEIGEN__::Matrix2x2d* triDmInverses,
                                         const double3*          vertexes,
                                         const uint3*            triangles,
                                         int                     triangleNum,
                                         Eigen::Matrix<double, 3, 2>* F3x2)
{
    int numbers = triangleNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calculate_triangle_fem_deformationF<<<blockNum, threadNum>>>(triDmInverses,
                                                                  vertexes,
                                                                  triangles,

                                                                  triangleNum,
                                                                  F3x2);
}

void calculate_bending_gradient_hessian(const double3*   vertexes,
                                        const double3*   rest_vertexes,
                                        const uint2*     edges,
                                        const uint2*     edges_adj_vertex,
                                        double3*         gradient,
                                        int              edgeNum,
                                        double           bendStiff,
                                        int              global_offset,
                                        Eigen::Matrix3d* triplet_values,
                                        int*             row_ids,
                                        int*             col_ids,
                                        double           IPC_dt,
                                        int global_hessian_fem_offset)
{
    int numbers = edgeNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calculate_bending_gradient_hessian<<<blockNum, threadNum>>>(vertexes,
                                                                 rest_vertexes,
                                                                 edges,
                                                                 edges_adj_vertex,
                                                                 gradient,
                                                                 edgeNum,
                                                                 bendStiff,
                                                                 global_offset,
                                                                 triplet_values,
                                                                 row_ids,
                                                                 col_ids,
                                                                 IPC_dt,
                                                                 global_hessian_fem_offset);
}


#ifdef USE_QUADRATIC_BENDING
void calculate_quad_bending_gradient_hessian(const double3* vertexes,
                                             const double3* rest_vertexes,
                                             const uint2*   edges,
                                             const uint2*   edges_adj_vertex,
                                             const Eigen::Matrix4d* quad_bending_Q,
                                             double3*         gradient,
                                             int              edgeNum,
                                             double           bendStiff,
                                             int              global_offset,
                                             Eigen::Matrix3d* triplet_values,
                                             int*             row_ids,
                                             int*             col_ids,
                                             double           IPC_dt,
                                             int global_hessian_fem_offset)
{
    int numbers = edgeNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calculate_quad_bending_gradient_hessian<<<blockNum, threadNum>>>(vertexes,
                                                                      rest_vertexes,
                                                                      edges,
                                                                      edges_adj_vertex,
                                                                      quad_bending_Q,
                                                                      gradient,
                                                                      edgeNum,
                                                                      bendStiff,
                                                                      global_offset,
                                                                      triplet_values,
                                                                      row_ids,
                                                                      col_ids,
                                                                      IPC_dt,
                                                                      global_hessian_fem_offset);
}
#endif


void calculate_fem_gradient(__GEIGEN__::Matrix3x3d* DmInverses,
                            const double3*          vertexes,
                            const uint4*            tetrahedras,
                            const double*           volume,
                            double3*                gradient,
                            int                     tetrahedraNum,
                            double*                 lenRate,
                            double*                 volRate,
                            double                  dt)
{
    int numbers = tetrahedraNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calculate_fem_gradient<<<blockNum, threadNum>>>(
        DmInverses, vertexes, tetrahedras, volume, gradient, tetrahedraNum, lenRate, volRate, dt);
}

void calculate_triangle_fem_gradient(__GEIGEN__::Matrix2x2d* triDmInverses,
                                     const double3*          vertexes,
                                     const uint3*            triangles,
                                     const double*           area,
                                     double3*                gradient,
                                     int                     triangleNum,
                                     double                  stretchStiff,
                                     double                  shearStiff,
                                     double                  IPC_dt,
                                     double                  strainRate)
{
    int numbers = triangleNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calculate_triangle_fem_gradient<<<blockNum, threadNum>>>(
        triDmInverses, vertexes, triangles, area, gradient, triangleNum, stretchStiff, shearStiff, IPC_dt, strainRate);
}

double calcMinMovement(const double3* _moveDir, double* _queue, const int& number)
{

    int numbers = number;
    if(numbers < 1)
        return 0;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    /*double* _tempMinMovement;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_tempMinMovement, numbers * sizeof(double)));*/
    //CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number * sizeof(AABB), cudaMemcpyDeviceToDevice));

    _reduct_max_double3_to_double<<<blockNum, threadNum, sharedMsize>>>(_moveDir, _queue, numbers);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());

    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while(numbers > 1)
    {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        _reduct_max_double<<<blockNum, threadNum, sharedMsize>>>(_queue, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double minValue;
    cudaMemcpy(&minValue, _queue, sizeof(double), cudaMemcpyDeviceToHost);
    //CUDA_SAFE_CALL(cudaFree(_tempMinMovement));
    return minValue;
}

void stepForward(double3* _vertexes,
                 double3* _vertexesTemp,
                 double3* _moveDir,
                 int*     bType,
                 double   alpha,
                 bool     moveBoundary,
                 int      numbers)
{
    const unsigned int threadNum = default_threads;
    if(numbers < 1)
        return;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _stepForward<<<blockNum, threadNum>>>(
        _vertexes, _vertexesTemp, _moveDir, bType, alpha, moveBoundary, numbers);
}

// [M1 substitution method] Hard-constraint projection kernel.
// world_pos = q.t + A(q) * local_pos
// Canonical q layout per abd_jacobi_matrix.inl operator*(ABDJacobi, Vector12):
//   q.v[0..2]  : translation t
//   q.v[3..5]  : A.row(0)
//   q.v[6..8]  : A.row(1)
//   q.v[9..11] : A.row(2)
// So (A * lp).x = q[3]*lp.x + q[4]*lp.y + q[5]*lp.z, etc.
// Called after ABD step_forward (each line-search alpha try). pinned FEM
// vertices follow ABD's q exactly, so their motion is consistent with the
// ABD body's affine transform, and IPC line search sees a smooth energy.
__global__ void _apply_fem_pins(double3* _vertexes,
                                const int* _pin_fem_vertex,
                                const int* _pin_abd_body_id,
                                const double3* _pin_abd_local_pos,
                                const __GEIGEN__::Vector12* _abd_q,
                                int n_pins)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n_pins) return;
    int fem_v = _pin_fem_vertex[idx];
    int bid   = _pin_abd_body_id[idx];
    const __GEIGEN__::Vector12& q = _abd_q[bid];
    double3 lp = _pin_abd_local_pos[idx];
    // Previous code used q[3]/q[6]/q[9] for the x-component, which is
    // A.col(0)·lp = (A^T·lp)[0] — wrong for non-symmetric A.  Only
    // worked when A ≈ scale·I (e.g. Animated mode where the ABD
    // barely rotates).  Showed up under URDF arms with revolute
    // joints (case_35+): pinned FEM verts visually drift off the
    // rigid sub-mesh as the ABD body rotates.
    _vertexes[fem_v].x = q.v[0] + q.v[3] * lp.x + q.v[4]  * lp.y + q.v[5]  * lp.z;
    _vertexes[fem_v].y = q.v[1] + q.v[6] * lp.x + q.v[7]  * lp.y + q.v[8]  * lp.z;
    _vertexes[fem_v].z = q.v[2] + q.v[9] * lp.x + q.v[10] * lp.y + q.v[11] * lp.z;
}

void apply_fem_pins(double3* _vertexes,
                    const int* _pin_fem_vertex,
                    const int* _pin_abd_body_id,
                    const double3* _pin_abd_local_pos,
                    const void* _abd_q,
                    int n_pins)
{
    if(n_pins <= 0 || _abd_q == nullptr) return;
    const unsigned int threadNum = default_threads;
    int blockNum = (n_pins + threadNum - 1) / threadNum;
    _apply_fem_pins<<<blockNum, threadNum>>>(
        _vertexes, _pin_fem_vertex, _pin_abd_body_id, _pin_abd_local_pos,
        reinterpret_cast<const __GEIGEN__::Vector12*>(_abd_q),
        n_pins);
}

void GIPC::step_forward(device_TetraData& TetMesh, double alpha, bool move_boundary)
{
    auto vertexes = muda::BufferView<double3>{TetMesh.vertexes, vertexNum};
    auto vertexes_temp = muda::BufferView<double3>{TetMesh.temp_double3Mem, vertexNum};
    auto move_dir = muda::BufferView<double3>{_moveDir, vertexNum};
    if(abd_fem_count_info.fem_point_num > 0)
    {
        auto fem_vertexes = vertexes.subview(abd_fem_count_info.fem_point_offset,
                                             abd_fem_count_info.fem_point_num);
        auto fem_vertexes_temp =
            vertexes_temp.subview(abd_fem_count_info.fem_point_offset,
                                  abd_fem_count_info.fem_point_num);

        auto fem_move_dir = move_dir.subview(abd_fem_count_info.fem_point_offset,
                                             abd_fem_count_info.fem_point_num);

        auto btype = muda::BufferView<int>{TetMesh.BoundaryType, vertexNum}.subview(
            abd_fem_count_info.fem_point_offset, abd_fem_count_info.fem_point_num);


        // [multi-env S2] per-env FEM step when active (env g's verts step by
        // m_env_alpha[g]); else the standard uniform step.
        if(m_perenv_apply && m_env_alpha && TetMesh.d_point_to_group
           && abd_fem_count_info.fem_point_num > 0)
        {
            const unsigned int tn = default_threads;
            int n = (int)fem_vertexes.size();
            int bn = (n + tn - 1) / tn;
            _stepForward_perenv<<<bn, tn>>>(
                fem_vertexes.data(), fem_vertexes_temp.data(), fem_move_dir.data(),
                btype.data(), TetMesh.d_point_to_group + abd_fem_count_info.fem_point_offset,
                m_env_alpha, alpha, move_boundary, n);
        }
        else
        {
            stepForward(fem_vertexes.data(),
                        fem_vertexes_temp.data(),
                        fem_move_dir.data(),
                        btype.data(),
                        alpha,
                        move_boundary,
                        fem_vertexes.size());
        }
    }
    if(abd_fem_count_info.abd_point_num <= 0)
        return;

    auto abd_vertexes = muda::BufferView<double3>{TetMesh.vertexes, vertexNum}.subview(
        abd_fem_count_info.abd_point_offset, abd_fem_count_info.abd_point_num);

    // [multi-env S2] per-body ABD alpha when active (gathered from m_env_alpha via
    // body_to_group); nullptr -> uniform scalar alpha (baseline).
    const double* abd_alpha_ptr = nullptr;
    if(m_perenv_apply && m_abd_body_alpha && TetMesh.d_body_to_group)
        abd_alpha_ptr = m_abd_body_alpha;
    m_abd_system->step_forward(*m_abd_sim_data, abd_vertexes, alpha, abd_alpha_ptr);

    // [M1 substitution method] After ABD step_forward updates q, project
    // pinned FEM vertices to ABD-derived positions: world = q.t + R(q)*lp.
    // Combined with mass=∞ and BoundaryType=Fixed (set in finalize), the
    // pinned vertices' Δx from PCG is ~0 and step_forward leaves them
    // alone; this kernel writes the correct ABD-derived position. IPC
    // line search now sees a smooth energy E(alpha) along the ABD's q
    // direction.
    if(TetMesh.n_fem_pins > 0 && m_d_abd_body_q != nullptr)
    {
        apply_fem_pins(TetMesh.vertexes,
                       TetMesh.d_fem_pin_fem_vertex,
                       TetMesh.d_fem_pin_abd_body_id,
                       TetMesh.d_fem_pin_abd_local_pos,
                       m_d_abd_body_q,
                       TetMesh.n_fem_pins);
    }
}

void updateSurfaces(uint32_t* sortIndex, uint3* _faces, const int& offset_num, const int& numbers)
{
    const unsigned int threadNum = default_threads;
    if(numbers < 1)
        return;
    int blockNum = (numbers + threadNum - 1) / threadNum;  //
    _updateSurfaces<<<blockNum, threadNum>>>(sortIndex, _faces, offset_num, numbers);
}

void updateSurfaceEdges(uint32_t* sortIndex, uint2* _edges, const int& offset_num, const int& numbers)
{
    const unsigned int threadNum = default_threads;
    if(numbers < 1)
        return;
    int blockNum = (numbers + threadNum - 1) / threadNum;  //
    _updateEdges<<<blockNum, threadNum>>>(sortIndex, _edges, offset_num, numbers);
}

void updateTriEdges_adjVerts(uint32_t*  sortIndex,
                             uint2*     _tri_edges,
                             uint2*     _adj_verts,
                             const int& offset_num,
                             const int& numbers)
{
    const unsigned int threadNum = default_threads;
    if(numbers < 1)
        return;
    int blockNum = (numbers + threadNum - 1) / threadNum;  //
    _updateTriEdges_adjVerts<<<blockNum, threadNum>>>(
        sortIndex, _tri_edges, _adj_verts, offset_num, numbers);
}


void updateSurfaceVerts(uint32_t* sortIndex, uint32_t* _sVerts, const int& offset_num, const int& numbers)
{
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _updateSurfVerts<<<blockNum, threadNum>>>(sortIndex, _sVerts, offset_num, numbers);
}

void updateNeighborInfo(unsigned int*   _neighborList,
                        unsigned int*   d_neighborListInit,
                        unsigned int*   _neighborNum,
                        unsigned int*   _neighborNumInit,
                        unsigned int*   _neighborStart,
                        unsigned int*   _neighborStartTemp,
                        const uint32_t* sortIndex,
                        const uint32_t* sortMapVertIndex,
                        const int&      numbers,
                        const int&      neighborListSize)
{
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _updateNeighborNum<<<blockNum, threadNum>>>(_neighborNumInit, _neighborNum, sortIndex, numbers);
    thrust::exclusive_scan(thrust::device_ptr<unsigned int>(_neighborNum),
                           thrust::device_ptr<unsigned int>(_neighborNum) + numbers,
                           thrust::device_ptr<unsigned int>(_neighborStartTemp));
    _updateNeighborList<<<blockNum, threadNum>>>(d_neighborListInit,
                                                 _neighborList,
                                                 _neighborNum,
                                                 _neighborStart,
                                                 _neighborStartTemp,
                                                 sortIndex,
                                                 sortMapVertIndex,
                                                 numbers);
    CUDA_SAFE_CALL(cudaMemcpy(d_neighborListInit,
                              _neighborList,
                              neighborListSize * sizeof(unsigned int),
                              cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(_neighborStart,
                              _neighborStartTemp,
                              numbers * sizeof(unsigned int),
                              cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(
        _neighborNumInit, _neighborNum, numbers * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
}

void calcTetMChash(uint64_t*         _MChash,
                   const double3*    _vertexes,
                   uint4*            tets,
                   const const AABB* _MaxBv,
                   const uint32_t*   sortMapVertIndex,
                   int               number)
{
    int numbers = number;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calcTetMChash<<<blockNum, threadNum>>>(
        _MChash, _vertexes, tets, _MaxBv, sortMapVertIndex, number);
}

void updateTopology(uint4* tets, uint3* tris, const uint32_t* sortMapVertIndex, int traNumber, int triNumber)
{
    int numbers = std::max(traNumber, triNumber);
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _updateTopology<<<blockNum, threadNum>>>(tets, tris, sortMapVertIndex, traNumber, triNumber);
}

void updateVertexes(double3*                      o_vertexes,
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
    int numbers = number;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _updateVertexes<<<blockNum, threadNum>>>(
        o_vertexes, _vertexes, tempM, mass, tempCons, tempBtype, cons, bType, sortIndex, sortMapIndex, numbers);
}

void updateTetrahedras(uint4*                        o_tetrahedras,
                       uint4*                        tetrahedras,
                       double*                       tempV,
                       const double*                 volum,
                       __GEIGEN__::Matrix3x3d*       tempDmInverse,
                       const __GEIGEN__::Matrix3x3d* dmInverse,
                       const uint32_t*               sortTetIndex,
                       const uint32_t*               sortMapVertIndex,
                       int                           number)
{
    int numbers = number;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _updateTetrahedras<<<blockNum, threadNum>>>(
        o_tetrahedras, tetrahedras, tempV, volum, tempDmInverse, dmInverse, sortTetIndex, sortMapVertIndex, number);
}

void calcVertMChash(uint64_t* _MChash, const double3* _vertexes, const AABB* _MaxBv, int number)
{
    int numbers = number;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calcVertMChash<<<blockNum, threadNum>>>(_MChash, _vertexes, _MaxBv, number);
}

void sortGeometry(device_TetraData& TetMesh,
                  const AABB*       _MaxBv,
                  const int&        vertex_num,
                  const int&        tetradedra_num,
                  const int&        triangle_num)
{


}

////////////////////////TO DO LATER/////////////////////////////////////////


void compute_H_b(double d, double dHat, double& H)
{
    double t = d - dHat;
    H = (std::log(d / dHat) * -2.0 - t * 4.0 / d) + 1.0 / (d * d) * (t * t);
}

void GIPC::suggestKappa(double& kappa)
{
    double H_b;
    // [decouple] kappa's only batch-dependent input is the MERGED bboxDiagSize2 (dHat is already
    // abs_dhat-fixed). STIFF_DECOUPLE_THRESH uses the abs_dhat-fixed eff bbox → kappa batch-invariant
    // → env_0's barrier stiffness no longer depends on its batch-mates' contact state.
    double bb = bboxDiagSize2;
    // [abs-kappa consistency] when the user declares an ABSOLUTE contact scale (absolute_dhat>0),
    // κ's scale MUST follow it in ALL modes — deriving κ from the merged scene bbox dilutes the
    // barrier super-linearly with env count/spacing (softer, batch-dependent physics; ablation C2:
    // 258 vs 489 Newton was ENTIRELY this). Same consistency rule as the Newton-exit fix (c1d4d78)
    // and dHat/dTol/fDhat (init). uipc-style: stiffness from a physical contact scale, no bbox.
    // Scenes without absolute_dhat keep the classic bbox derivation.
    // [ablation diag] STIFF_DIAG_KAPPA_MERGEDBB forces the old merged-bbox κ (DIAGNOSTIC ONLY).
    if(!getenv("STIFF_DIAG_KAPPA_MERGEDBB")
       && absolute_dhat > 0.0 && relative_dhat > 0.0)
        bb = (absolute_dhat * absolute_dhat) / (relative_dhat * relative_dhat);
    compute_H_b(1.0e-16 * bb, dHat, H_b);
    if(meanMass == 0.0)
    {
        kappa = minKappaCoef / (4.0e-16 * bb * H_b);
    }
    else
    {
        kappa = minKappaCoef * meanMass / (4.0e-16 * bb * H_b);
    }
    //    printf("bboxDiagSize2: %f\n", bboxDiagSize2);
    //    printf("H_b: %f\n", H_b);
    //    printf("sug Kappa: %f\n", kappa);
}

void GIPC::upperBoundKappa(double& kappa)
{
    double H_b;
    double bb = bboxDiagSize2;   // [abs-kappa consistency] absolute_dhat ⇒ absolute κ scale
    if(!getenv("STIFF_DIAG_KAPPA_MERGEDBB")   // (see suggestKappa; diag = old-bbox escape)
       && absolute_dhat > 0.0 && relative_dhat > 0.0)
        bb = (absolute_dhat * absolute_dhat) / (relative_dhat * relative_dhat);
    compute_H_b(1.0e-16 * bb, dHat, H_b);
    double kappaMax = 100 * minKappaCoef * meanMass / (4.0e-16 * bb * H_b);
    //printf("max Kappa: %f\n", kappaMax);
    if(meanMass == 0.0)
    {
        kappaMax = 100 * minKappaCoef / (4.0e-16 * bb * H_b);
    }

    if(kappa > kappaMax)
    {
        kappa = kappaMax;
    }
}


void GIPC::initKappa(device_TetraData& TetMesh)
{
    // [batch-size fix] IPC_Solver calls initKappa() BEFORE the first computeGradientAndHessian(),
    // where per-group κ is normally enabled. On frame 0 that left m_pergroup_kappa=false, so env_0
    // fell back to the GLOBAL Kappa (= -gsum/gsnorm, a reduction over ALL envs' verts → N-dependent)
    // → the batch-SIZE divergence seed. Enable per-group κ here too so env_0 uses its OWN per-env κ
    // (binned over d_point_to_group) from the very first step → N-independent.
    if(getenv("STIFF_PERGROUP_KAPPA") && TetMesh.d_point_to_group
       && TetMesh.h_groups_present   /* [N=1 guard] wildcard p2g -> kappa_grp[-1] OOB */
       && !m_pergroup_kappa)
    {
        m_pergroup_kappa = true;
        m_d_p2g          = TetMesh.d_point_to_group;
        int NG           = kEnvAlphaSlots;
        CUDA_SAFE_CALL(cudaMalloc((void**)&m_kappa_group, NG * sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&m_d_close_grp, NG * sizeof(int)));
        h_kappa_group.assign(NG, 0.0);
        printf("[pergroup-kappa] enabled (early, in initKappa) NG=%d\n", NG);
    }
    bool perenv_kappa_filled = false;   // [decouple] set when per-env κ replaces the stub broadcast
    if(h_cpNum[0] > 0)
    {
        double3* _GE = TetMesh.fb;
        double3* _gc = TetMesh.temp_double3Mem;
        //CUDA_SAFE_CALL(cudaMalloc((void**)&_gc, vertexNum * sizeof(double3)));
        //CUDA_SAFE_CALL(cudaMalloc((void**)&_GE, vertexNum * sizeof(double3)));
        CUDA_SAFE_CALL(cudaMemset(_gc, 0, vertexNum * sizeof(double3)));
        CUDA_SAFE_CALL(cudaMemset(_GE, 0, vertexNum * sizeof(double3)));
        calKineticGradient(TetMesh.vertexes, TetMesh.xTilta, _GE, TetMesh.masses, vertexNum);
        // [multi-env determinism 4.3] elastic-side binned bracket: FEM + soft → g_gbin → _GE
        // (kinetic already written directly to _GE above).
        zeroBinnedGrad();
        calculate_fem_gradient(TetMesh.DmInverses,
                               TetMesh.vertexes,
                               TetMesh.tetrahedras,
                               TetMesh.volum,
                               _GE,
                               tetrahedraNum,
                               TetMesh.lengthRate,
                               TetMesh.volumeRate,
                               IPC_dt);
        //calculate_triangle_fem_gradient(TetMesh.triDmInverses, TetMesh.vertexes, TetMesh.triangles, TetMesh.area, _GE, triangleNum, stretchStiff, shearStiff, IPC_dt);
        // soft constraint is the last elastic-side gradient → g_gbin; close the bracket → _GE:
        computeSoftConstraintGradient(_GE);
        combineBinnedGrad(_GE);
        // ground + barrier → _gc:
        zeroBinnedGrad();
        computeGroundGradient(_gc, 1);
        calBarrierGradient(_gc, 1);
        combineBinnedGrad(_gc);
        double gsum = reduction2Kappa(0, _gc, _GE, pcg_data.squeue, vertexNum);
        double gsnorm = reduction2Kappa(1, _gc, _GE, pcg_data.squeue, vertexNum);
        //CUDA_SAFE_CALL(cudaFree(_gc));
        //CUDA_SAFE_CALL(cudaFree(_GE));
        double minKappa = -gsum / gsnorm;
        if(minKappa > 0.0)
        {
            Kappa = minKappa;
        }
        suggestKappa(minKappa);
        if(Kappa < minKappa)
        {
            Kappa = minKappa;
        }
        upperBoundKappa(Kappa);

        // [decouple] PER-ENV initKappa: the global minKappa = -gsum/gsnorm above is a GLOBAL
        // reduction over ALL envs' verts → batch-dependent, and it overrides the batch-invariant
        // suggestKappa. Here we instead set each env's κ from ITS OWN gradient ratio (binned per-env
        // reduction over d_point_to_group) → env_0's κ depends only on env_0 → batch-invariant.
        // suggested (=minKappa after suggestKappa, eff-bbox) is the batch-invariant floor; kmax the cap.
        if(getenv("STIFF_DECOUPLE_THRESH") && m_pergroup_kappa && m_kappa_group
           && TetMesh.d_point_to_group)
        {
            const int NG = kEnvAlphaSlots;
            static double* d_gsum_bin = nullptr; static double* d_gsnorm_bin = nullptr;
            static double* d_gsum_g = nullptr;   static double* d_gsnorm_g = nullptr;
            if(!d_gsum_bin) {
                cudaMalloc((void**)&d_gsum_bin,   (size_t)NG * BINNED_K * sizeof(double));
                cudaMalloc((void**)&d_gsnorm_bin, (size_t)NG * BINNED_K * sizeof(double));
                cudaMalloc((void**)&d_gsum_g,     NG * sizeof(double));
                cudaMalloc((void**)&d_gsnorm_g,   NG * sizeof(double));
            }
            cudaMemset(d_gsum_bin,   0, (size_t)NG * BINNED_K * sizeof(double));
            cudaMemset(d_gsnorm_bin, 0, (size_t)NG * BINNED_K * sizeof(double));
            int bs = 256, gs = (vertexNum + bs - 1) / bs;
            _per_env_kappa_deposit<<<gs, bs>>>(TetMesh.d_point_to_group, _gc, _GE,
                                               d_gsum_bin, d_gsnorm_bin, vertexNum, NG);
            _per_env_kappa_combine<<<(NG + bs - 1) / bs, bs>>>(d_gsum_g, d_gsnorm_g,
                                                               d_gsum_bin, d_gsnorm_bin, NG);
            double suggested = minKappa;   // batch-invariant (suggestKappa wrote it, eff bbox)
            double H_b, bb = (absolute_dhat > 0.0 && relative_dhat > 0.0)
                             ? (absolute_dhat * absolute_dhat) / (relative_dhat * relative_dhat)
                             : bboxDiagSize2;
            compute_H_b(1.0e-16 * bb, dHat, H_b);
            double kmax = 100.0 * minKappaCoef * (meanMass == 0.0 ? 1.0 : meanMass)
                          / (4.0e-16 * bb * H_b);
            // [perf/device-residence] finalize κ per env ON DEVICE (in m_kappa_group) — no D2H(gsum/gsnorm)
            // + host loop + H2D. suggested/kmax are env-independent scalars → bit-identical → strict OK.
            _per_env_kappa_finalize<<<(NG + bs - 1) / bs, bs>>>(d_gsum_g, d_gsnorm_g,
                                                               m_kappa_group, NG, suggested, kmax);
            if(getenv("STIFF_SEED_DIAG"))   // diag only: mirror first entries back for the print below
            {
                if((int)h_kappa_group.size() < NG) h_kappa_group.resize(NG, Kappa);
                CUDA_SAFE_CALL(cudaMemcpy(h_kappa_group.data(), m_kappa_group,
                                          NG * sizeof(double), cudaMemcpyDeviceToHost));
            }
            perenv_kappa_filled = true;
        }
    }

    // [multi-env per-group κ] broadcast the init κ to all groups (STUB: all groups = global κ).
    // SKIPPED when the per-env initKappa above filled m_kappa_group with true per-env values.
    if(m_pergroup_kappa && m_kappa_group && !perenv_kappa_filled)
    {
        int NG = kEnvAlphaSlots;
        if((int)h_kappa_group.size() < NG) h_kappa_group.resize(NG, Kappa);
        for(int g = 0; g < NG; g++) h_kappa_group[g] = Kappa;
        CUDA_SAFE_CALL(cudaMemcpy(m_kappa_group, h_kappa_group.data(), NG * sizeof(double), cudaMemcpyHostToDevice));
    }
    //printf("Kappa ====== %f\n", Kappa);
    if(getenv("STIFF_SEED_DIAG"))
        printf("[seed-kappa] Kappa=%.17g kappa_group[0]=%.17g kappa_group[1]=%.17g h_cpNum0=%u perenv_filled=%d\n",
               Kappa, (h_kappa_group.size() > 0 ? h_kappa_group[0] : -1.0),
               (h_kappa_group.size() > 1 ? h_kappa_group[1] : -1.0), h_cpNum[0], (int)perenv_kappa_filled);
}


void GIPC::partitionContactHessian()
{

    muda::DeviceRadixSort().SortPairs(gipc_global_triplet.block_hash_value(),
                                      gipc_global_triplet.block_sort_hash_value(),
                                      gipc_global_triplet.block_index(),
                                      gipc_global_triplet.block_sort_index(),
                                      gipc_global_triplet.global_collision_triplet_offset);

    int threadNum = 256;

    LaunchCudaKernal_default(
        gipc_global_triplet.global_collision_triplet_offset,
        threadNum,
        0,
        _reorder_triplets,
        gipc_global_triplet.block_row_indices(),
        gipc_global_triplet.block_col_indices(),
        gipc_global_triplet.block_values(),
        gipc_global_triplet.block_row_indices(gipc_global_triplet.global_collision_triplet_offset),
        gipc_global_triplet.block_col_indices(gipc_global_triplet.global_collision_triplet_offset),
        gipc_global_triplet.block_values(gipc_global_triplet.global_collision_triplet_offset),
        (const uint32_t*)gipc_global_triplet.block_sort_index(),
        gipc_global_triplet.global_collision_triplet_offset);

    //gipc_global_triplet.d_abd_abd_contact_start_id = -1;
    //gipc_global_triplet.d_abd_fem_contact_start_id = -1;
    //gipc_global_triplet.d_fem_abd_contact_start_id = -1;
    //gipc_global_triplet.d_fem_fem_contact_start_id = -1;

    CUDA_SAFE_CALL(cudaMemset(gipc_global_triplet.d_abd_abd_contact_start_id, -1, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(gipc_global_triplet.d_abd_fem_contact_start_id, -1, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(gipc_global_triplet.d_fem_abd_contact_start_id, -1, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(gipc_global_triplet.d_fem_fem_contact_start_id, -1, sizeof(int)));

    size_t shareMem = (threadNum + 1) * sizeof(int);
    LaunchCudaKernal_default(gipc_global_triplet.global_collision_triplet_offset,
                             threadNum,
                             shareMem,
                             _partition_collision_triplets,
                             (const uint64_t*)gipc_global_triplet.block_sort_hash_value(),
                             gipc_global_triplet.d_abd_abd_contact_start_id,
                             gipc_global_triplet.d_abd_fem_contact_start_id,
                             gipc_global_triplet.d_fem_abd_contact_start_id,
                             gipc_global_triplet.d_fem_fem_contact_start_id,
                             //abd_fem_count_info.abd_point_num,
                             gipc_global_triplet.global_collision_triplet_offset);


    //gipc_global_triplet.h_abd_abd_contact_start_id =
    //    gipc_global_triplet.d_abd_abd_contact_start_id;
    //gipc_global_triplet.h_abd_fem_contact_start_id =
    //    gipc_global_triplet.d_abd_fem_contact_start_id;
    //gipc_global_triplet.h_fem_abd_contact_start_id =
    //    gipc_global_triplet.d_fem_abd_contact_start_id;
    //gipc_global_triplet.h_fem_fem_contact_start_id =
    //    gipc_global_triplet.d_fem_fem_contact_start_id;

    // ②-D2H: single batched copy of the 4 contiguous start-ids (block[0..3])
    // replaces 4 separate blocking D2H (each of which drains the GPU).
    int h_csb[4];
    CUDA_SAFE_CALL(cudaMemcpy(h_csb,
                              gipc_global_triplet.d_abd_abd_contact_start_id,
                              4 * sizeof(int),
                              cudaMemcpyDeviceToHost));
    gipc_global_triplet.h_abd_abd_contact_start_id = h_csb[0];
    gipc_global_triplet.h_abd_fem_contact_start_id = h_csb[1];
    gipc_global_triplet.h_fem_abd_contact_start_id = h_csb[2];
    gipc_global_triplet.h_fem_fem_contact_start_id = h_csb[3];


    if(gipc_global_triplet.h_fem_fem_contact_start_id >= 0)
    {
        if(gipc_global_triplet.h_abd_fem_contact_start_id > 0)
        {
            gipc_global_triplet.fem_fem_contact_num =
                gipc_global_triplet.h_abd_fem_contact_start_id
                - gipc_global_triplet.h_fem_fem_contact_start_id;
            if(gipc_global_triplet.h_fem_abd_contact_start_id > 0)
            {
                gipc_global_triplet.abd_fem_contact_num =
                    gipc_global_triplet.h_fem_abd_contact_start_id
                    - gipc_global_triplet.h_abd_fem_contact_start_id;

                gipc_global_triplet.fem_abd_contact_num =
                    gipc_global_triplet.h_abd_abd_contact_start_id
                    - gipc_global_triplet.h_fem_abd_contact_start_id;
            }
            else
            {
                gipc_global_triplet.abd_fem_contact_num =
                    gipc_global_triplet.h_abd_abd_contact_start_id
                    - gipc_global_triplet.h_abd_fem_contact_start_id;

                gipc_global_triplet.fem_abd_contact_num = 0;
            }
            gipc_global_triplet.abd_abd_contact_num =
                gipc_global_triplet.global_collision_triplet_offset
                - gipc_global_triplet.h_abd_abd_contact_start_id;
        }
        else if(gipc_global_triplet.h_abd_abd_contact_start_id > 0)
        {
            gipc_global_triplet.fem_fem_contact_num =
                gipc_global_triplet.h_abd_abd_contact_start_id
                - gipc_global_triplet.h_fem_fem_contact_start_id;
            gipc_global_triplet.abd_abd_contact_num =
                gipc_global_triplet.global_collision_triplet_offset
                - gipc_global_triplet.h_abd_abd_contact_start_id;

            gipc_global_triplet.abd_fem_contact_num = 0;
            gipc_global_triplet.fem_abd_contact_num = 0;
        }
        else
        {
            gipc_global_triplet.fem_fem_contact_num =
                gipc_global_triplet.global_collision_triplet_offset;

            gipc_global_triplet.abd_abd_contact_num = 0;

            gipc_global_triplet.abd_fem_contact_num = 0;
            gipc_global_triplet.fem_abd_contact_num = 0;
        }
    }
    else if(gipc_global_triplet.h_abd_abd_contact_start_id >= 0)
    {
        gipc_global_triplet.abd_abd_contact_num =
            gipc_global_triplet.global_collision_triplet_offset;

        gipc_global_triplet.fem_fem_contact_num = 0;
        gipc_global_triplet.abd_fem_contact_num = 0;
        gipc_global_triplet.fem_abd_contact_num = 0;
    }
    else
    {
        gipc_global_triplet.abd_abd_contact_num = 0;
        gipc_global_triplet.fem_fem_contact_num = 0;
        gipc_global_triplet.abd_fem_contact_num = 0;
        gipc_global_triplet.fem_abd_contact_num = 0;
    }

    gipc_global_triplet.h_fem_fem_contact_start_id = 0;
    gipc_global_triplet.h_abd_fem_contact_start_id =
        gipc_global_triplet.h_fem_fem_contact_start_id + gipc_global_triplet.fem_fem_contact_num;
    gipc_global_triplet.h_fem_abd_contact_start_id =
        gipc_global_triplet.h_abd_fem_contact_start_id + gipc_global_triplet.abd_fem_contact_num;
    gipc_global_triplet.h_abd_abd_contact_start_id =
        gipc_global_triplet.h_fem_abd_contact_start_id + gipc_global_triplet.fem_abd_contact_num;


    int number = gipc_global_triplet.global_collision_triplet_offset;

    CUDA_SAFE_CALL(
        cudaMemcpy(gipc_global_triplet.block_row_indices(),
                   gipc_global_triplet.block_row_indices() + gipc_global_triplet.global_collision_triplet_offset,
                   gipc_global_triplet.global_collision_triplet_offset * sizeof(int),
                   cudaMemcpyDeviceToDevice));

    CUDA_SAFE_CALL(
        cudaMemcpy(gipc_global_triplet.block_col_indices(),
                   gipc_global_triplet.block_col_indices() + gipc_global_triplet.global_collision_triplet_offset,
                   gipc_global_triplet.global_collision_triplet_offset * sizeof(int),
                   cudaMemcpyDeviceToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(
        gipc_global_triplet.block_values(),
        gipc_global_triplet.block_values() + gipc_global_triplet.global_collision_triplet_offset,
        gipc_global_triplet.global_collision_triplet_offset * sizeof(Eigen::Matrix3d),
        cudaMemcpyDeviceToDevice));
}

static void _dbg_ksum_comm(const char* name, const void* dptr, size_t nbytes);  // [4.3 fwd]
#define KSEG(nm) if(getenv("STIFF_KSUM")) { cudaDeviceSynchronize(); \
    _dbg_ksum_comm(nm, gipc_global_triplet.block_values(), \
        (size_t)gipc_global_triplet.global_triplet_offset * 9 * sizeof(double)); }

// [decouple probe] file-scope frame/k counters so computeGradientAndHessian can gate per-stage
// shape-gradient dumps (set by solve_subIP). STIFF_SHAPE_STAGE dumps shape_grads after kinetic
// (.s1) and after the elastic bracket (.s2) → cross-batch compare splits kinetic vs elastic.
int g_dec_frame = -1;   // [decouple probe] non-static so other TUs (pcg_solver) can gate dumps by frame/k
int g_dec_k     = -1;

float GIPC::computeGradientAndHessian(device_TetraData& TetMesh)
{
    gipc::Timer timer{"cal_gradient_hessian"};

    // [multienv-mode] one-time: fast plain-atomic gradient for merged/isolated (STIFF_FAST_GRAD),
    // binned order-free gradient for strict/default. Set once (device symbol).
    static bool s_binned_set = false;
    if(!s_binned_set) { set_binned_on((getenv("STIFF_FAST_GRAD") && !getenv("STIFF_DIAG_BINNED_GRAD")) ? 0 : 1); s_binned_set = true; }  // [DIAG] STIFF_DIAG_BINNED_GRAD=1 forces binned gradient under FAST

    // [multi-env P2] capture d_point_to_group + enable per-env BVH (once). buildCP uses these
    // lazily (it has no TetMesh). STIFF_PERENV_BVH gates; needs grouped envs (d_point_to_group).
    if(getenv("STIFF_PERENV_BVH") && TetMesh.d_point_to_group
       && TetMesh.h_groups_present)   // [N=1 guard] all -1 p2g -> per-env index excludes
                                      // EVERY prim (active=0) -> ZERO self-collision
    {
        m_perenv_bvh = true;
        m_d_p2g      = TetMesh.d_point_to_group;
    }
    // [multi-env cross-env diagnostic] capture p2g + report env0-vs-env1 vertex divergence at the
    // START of each gradient/Hessian (= verts from the previous step's line search). STIFF_XENV.
    if(getenv("STIFF_XENV") && TetMesh.d_point_to_group)
    {
        m_d_p2g = TetMesh.d_point_to_group;
        static int _xc = 0;
        char lbl[48]; snprintf(lbl, sizeof(lbl), "verts call#%d", _xc++);
        xenvDiff(_vertexes, lbl);
    }
    // [multi-env per-group κ] enable + allocate (once). STIFF_PERGROUP_KAPPA gates; needs groups.
    if(getenv("STIFF_PERGROUP_KAPPA") && TetMesh.d_point_to_group
       && TetMesh.h_groups_present   /* [N=1 guard] wildcard p2g -> kappa_grp[-1] OOB */
       && !m_pergroup_kappa)
    {
        m_pergroup_kappa = true;
        m_d_p2g          = TetMesh.d_point_to_group;
        int NG           = kEnvAlphaSlots;
        CUDA_SAFE_CALL(cudaMalloc((void**)&m_kappa_group, NG * sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&m_d_close_grp, NG * sizeof(int)));
        h_kappa_group.assign(NG, 0.0);
        printf("[pergroup-kappa] enabled, NG=%d\n", NG);
    }

    CUDA_SAFE_CALL(cudaMemset(TetMesh.fb, 0, vertexNum * sizeof(double3)));
    // [multi-env determinism 4.3] zero the binned contact/friction gradient accumulator
    // (bins start at 0; deposits add exactly). Combined back into contact_grads after ground.
    CUDA_SAFE_CALL(cudaMemset(g_grad_binned, 0,
                              3 * (size_t)vertexNum * BINNED_K * sizeof(double)));
    CUDA_SAFE_CALL(cudaMemset(TetMesh.shape_grads, 0, vertexNum * sizeof(double3)));

    // [multi-env determinism 4.3] zero the WHOLE triplet buffer (block values + row/col) to the
    // reserved capacity. The triplet count is a provable UPPER BOUND (16 slots/pair, but PP/PE/PT
    // write fewer) → reserved-but-unwritten slots otherwise hold GARBAGE that the converter
    // processes → non-deterministic matrix. Zeroing makes them (0,0)=0 (benign + deterministic).
    {
        size_t cap = gipc_global_triplet.triplet_capacity();
        if(cap > 0)
        {
            CUDA_SAFE_CALL(cudaMemset(gipc_global_triplet.block_values(), 0, cap * 9 * sizeof(double)));
            CUDA_SAFE_CALL(cudaMemset(gipc_global_triplet.block_row_indices(), 0, cap * sizeof(int)));
            CUDA_SAFE_CALL(cudaMemset(gipc_global_triplet.block_col_indices(), 0, cap * sizeof(int)));
        }
    }

    //muda::BufferView<double3>{TetMesh.shape_grads, vertexNum}.fill(double3{0, 0, 0});


    auto shape_grads   = TetMesh.shape_grads;
    auto contact_grads = TetMesh.fb;
    {
        gipc::Timer timer{"cal_kinetic_gradient"};
        calKineticGradient(
            TetMesh.vertexes, TetMesh.xTilta, shape_grads, TetMesh.masses, vertexNum);
    }
    if(getenv("STIFF_XENV") && m_d_p2g) xenvDiff(shape_grads, "  a.kinetic");

    // [decouple probe] sub-stage 1: shape_grads = kinetic only (per-vertex-local, expect clean).
    if(getenv("STIFF_SHAPE_STAGE") && getenv("STIFF_GRAD_PRE") && TetMesh.d_point_to_group
       && g_dec_frame == atoi(getenv("STIFF_DUMP_FRAME"))
       && g_dec_k == (getenv("STIFF_PROBE_K") ? atoi(getenv("STIFF_PROBE_K")) : 0))
    {
        std::vector<double3> h(vertexNum);
        CUDA_SAFE_CALL(cudaMemcpy(h.data(), shape_grads, vertexNum*sizeof(double3), cudaMemcpyDeviceToHost));
        FILE* a=fopen((std::string(getenv("STIFF_GRAD_PRE"))+".s1").c_str(),"wb");
        if(a){fwrite(h.data(),sizeof(double3),vertexNum,a);fclose(a);}
        printf("[shape-stage] s1 (kinetic) dumped @frame %d k=%d\n", g_dec_frame, g_dec_k);
    }

    gipc_global_triplet.global_triplet_offset = 0;

    // [P1-dyn] Grow the global triplet buffer BEFORE any assembly writes, to a PROVABLE
    // UPPER BOUND on this step's triplet count (so the unchecked assembly kernels can
    // NEVER overflow — not merely "usually fit"). The bound:
    //   - fixed (topology) internal triplets: m_fixed_triplet_base + fem_point (exact)
    //   - collision: h_cpNum[0] is the EXACT number of contact pairs calBarrier iterates;
    //     any pair writes at most M12_Off(=16) 3x3 blocks, so 16*h_cpNum[0] >= the real
    //     collision-triplet count for ANY type mix (PP/PE/PT/EE). Same for lagged
    //     friction (h_cpNum_last[0]) and ground (h_gpNum, generous *M6_Off).
    // block_values needs 2*bound (converter writes scratch/output to [length:2*length),
    // global_linear_system.cu); hash scratch needs bound. Non-hybrid only; hybrid keeps
    // the worst-case finalize allocation. Provable: bound >= actual length always.
    if(m_dynamic_triplet)
    {
        long long bound = m_fixed_triplet_base
            + static_cast<long long>(abd_fem_count_info.fem_point_num)
            + static_cast<long long>(h_cpNum[0]) * M12_Off     // all contact pairs x max blocks
            + static_cast<long long>(h_gpNum) * M6_Off;        // ground (generous)
#ifdef USE_FRICTION
        bound += static_cast<long long>(h_cpNum_last[0]) * M12_Off
               + static_cast<long long>(h_gpNum_last) * M6_Off;
#endif
        bound += 4096;                                          // fixed slack
        // Pre-assembly only needs capacity >= this step's length (assembly writes [0:length));
        // bound is a provable upper bound on length, so 1*bound is assembly-safe. The
        // converter's 2*length region is grown exactly at the convert site
        // (global_linear_system.cu), so the peak capacity = 2*length (the irreducible
        // out-of-place-converter floor), not 2*bound. Saves ~20% of the grasp-peak buffer.
        long long bv_need = bound;
        if(gipc_global_triplet.triplet_capacity() < static_cast<size_t>(bv_need))
            gipc_global_triplet.reserve_triplets(static_cast<size_t>(bv_need * 1.1));
        if(gipc_global_triplet.global_external_max_capcity < bound)
        {
            gipc_global_triplet.resize_collision_hash_size(static_cast<size_t>(bound * 1.1));
            gipc_global_triplet.global_external_max_capcity = static_cast<int>(bound * 1.1);
        }
    }

    {
        gipc::Timer timer{"cal_barrier_gradient_hessian"};
        CUDA_SAFE_CALL(cudaMemset(_cpNum, 0, 5 * sizeof(uint32_t)));
        //calBarrierHessian();
        //calBarrierGradient(contact_grads, Kappa);

        { static int _bc = 0;
          int on = (_bc++ == 0 && getenv("STIFF_BAR_TRACE")) ? 1 : 0;
          int t0 = getenv("STIFF_BAR_TGT0") ? atoi(getenv("STIFF_BAR_TGT0")) : -1;
          int t1 = getenv("STIFF_BAR_TGT1") ? atoi(getenv("STIFF_BAR_TGT1")) : -1;
          set_bar_targets(on, t0, t1); }
        calBarrierGradientAndHessian(contact_grads, Kappa);
        set_bar_targets(0, -1, -1);
        gipc_global_triplet.global_triplet_offset +=
            h_cpNum[4] * M12_Off + h_cpNum[3] * M9_Off + h_cpNum[2] * M6_Off;
    }
    KSEG("seg_contact")

    float time00 = 0;

#ifdef USE_FRICTION
    {

        gipc::Timer timer{"cal_friction_gradient_hessian"};
        if(!getenv("STIFF_SKIP_FRIC")) {   // [xenv pin] isolate friction's contribution to b
        calFrictionGradient(contact_grads, TetMesh);
        //CUDA_SAFE_CALL(cudaDeviceSynchronize());
        calFrictionHessian(TetMesh);
        }
        gipc_global_triplet.global_triplet_offset +=
            h_cpNum_last[4] * M12_Off + h_cpNum_last[3] * M9_Off
            + h_cpNum_last[2] * M6_Off + h_gpNum_last;
        //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }
#endif
    KSEG("seg_thru_friction")
    if(getenv("STIFF_KSUM")) printf("[ksum] cpNumLast=%u gpNumLast=%u\n", h_cpNum_last[0], h_gpNum_last);

    computeGroundGradientAndHessian(contact_grads);
    // [multi-env determinism 4.3] combine the binned contact+friction gradient into
    // contact_grads (holds the ground gradient). Order-independent ⇒ bit-identical.
    { int bs = 256, gs = (vertexNum + bs - 1) / bs;
      _gfxToGrad<<<gs, bs>>>(contact_grads, g_grad_binned, vertexNum); }
    if(getenv("STIFF_XENV") && m_d_p2g) {
        printf("[xenv]   (counts: cpNum=%u cpNumLast=%u gpNum=%u gpNumLast=%u)\n",
               h_cpNum[0], h_cpNum_last[0], (unsigned)h_gpNum, (unsigned)h_gpNum_last);
        xenvPairClassify(_collisonPairs, h_cpNum[0], "pairs");
        // [xenv dump] one-shot raw dump of pairs + p2g + verts to localize the differing pair.
        static bool _xdumped = false;
        if(getenv("STIFF_XENV_DUMP") && !_xdumped) {
            _xdumped = true;
            int n = h_cpNum[0];
            std::vector<int4> hp(n); std::vector<int> hg(vertexNum); std::vector<double3> hv(vertexNum);
            cudaMemcpy(hp.data(), _collisonPairs, (size_t)n*sizeof(int4), cudaMemcpyDeviceToHost);
            // [xenv] full-4-vert CCD pairs (no encoding loss) for clean env-local membership compare
            { std::vector<int4> hc(n);
              cudaMemcpy(hc.data(), _ccd_collisonPairs, (size_t)n*sizeof(int4), cudaMemcpyDeviceToHost);
              FILE* fc=fopen("/tmp/xd_ccd.bin","wb"); fwrite(hc.data(),sizeof(int4),n,fc); fclose(fc); }
            cudaMemcpy(hg.data(), m_d_p2g, (size_t)vertexNum*sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(hv.data(), _vertexes, (size_t)vertexNum*sizeof(double3), cudaMemcpyDeviceToHost);
            FILE* f;
            f=fopen("/tmp/xd_pairs.bin","wb"); fwrite(hp.data(),sizeof(int4),n,f); fclose(f);
            f=fopen("/tmp/xd_p2g.bin","wb");   fwrite(hg.data(),sizeof(int),vertexNum,f); fclose(f);
            f=fopen("/tmp/xd_verts.bin","wb"); fwrite(hv.data(),sizeof(double3),vertexNum,f); fclose(f);
            f=fopen("/tmp/xd_meta.txt","w");   fprintf(f,"%d %d %.17g\n",n,vertexNum,dHat); fclose(f);
            // [xenv] dump the EDGE list too — to check if env0/env1 edge sets are mirror-identical
            int nE = (int)bvh_e.edge_number;
            std::vector<uint2> he(nE);
            cudaMemcpy(he.data(), bvh_e._edges, (size_t)nE*sizeof(uint2), cudaMemcpyDeviceToHost);
            f=fopen("/tmp/xd_edges.bin","wb"); fwrite(he.data(),sizeof(uint2),nE,f); fclose(f);
            std::vector<double3> hfb(vertexNum);
            cudaMemcpy(hfb.data(), contact_grads, (size_t)vertexNum*sizeof(double3), cudaMemcpyDeviceToHost);
            f=fopen("/tmp/xd_fb.bin","wb"); fwrite(hfb.data(),sizeof(double3),vertexNum,f); fclose(f);
            // per-env edge count (by group of edge.x)
            int e0=0,e1=0; for(auto&e:he){ int g=hg[e.x]; if(g==0)e0++; else if(g==1)e1++; }
            printf("[xenv]   DUMPED %d pairs, %d verts, %d edges (env0=%d env1=%d), dHat=%.17g\n",
                   n, vertexNum, nE, e0, e1, dHat);
        }
        xenvDiff(contact_grads, "  b.barrier+fric+grnd");
    }
    gipc_global_triplet.global_triplet_offset += h_gpNum;
    KSEG("seg_thru_ground")
    gipc_global_triplet.global_collision_triplet_offset =
        gipc_global_triplet.global_triplet_offset;

    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    gipc_global_triplet.update_hash_value(abd_fem_count_info.abd_point_num);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    partitionContactHessian();
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());


    {
        gipc::Timer timer{"setup_abd_system_gradient_hessian"};

        // Set stitch spring parameters for ABD-side bilateral coupling
        // Only report stitch count when data pointers are valid; otherwise
        // the ABD system reserves triplet slots that are never written,
        // leaving garbage in the preconditioner and causing illegal access.
        m_abd_system->m_stitch_count              = m_d_stitch_paired_vertex ? softNum : 0;
        m_abd_system->m_d_stitch_paired_vertex    = m_d_stitch_paired_vertex;
        m_abd_system->m_d_stitch_rest_offset      = m_d_stitch_rest_offset;
        m_abd_system->m_d_stitch_abd_body_id      = m_d_stitch_abd_body_id;
        m_abd_system->m_d_stitch_fem_vertex_id    = targetInd;
        m_abd_system->m_d_all_vertexes            = _vertexes;
        m_abd_system->m_stitch_motion_rate        = softMotionRate;
        m_abd_system->m_stitch_rate               = animation_fullRate;

        m_abd_system->setup_abd_system_gradient_hessian(
            *m_abd_sim_data,
            TetMesh.BoundaryType,
            muda::BufferView<double3>{TetMesh.fb, vertexNum}.subview(
                abd_fem_count_info.abd_point_offset, abd_fem_count_info.abd_point_num),
            gipc_global_triplet);
    }
    if(getenv("STIFF_XENV") && m_d_p2g) xenvDiff(TetMesh.fb, "  c.+abd");

    int abd_dofs = abd_fem_count_info.abd_body_num * 4;
    int fem_global_hessian_index_offset = -abd_fem_count_info.abd_point_num + abd_dofs;
    {
        muda::ParallelFor(256)
            .kernel_name(__FUNCTION__)
            .apply(gipc_global_triplet.fem_fem_contact_num,
                   [cfem_rows = gipc_global_triplet.block_row_indices(
                        gipc_global_triplet.h_fem_fem_contact_start_id),
                    cfem_cols = gipc_global_triplet.block_col_indices(
                        gipc_global_triplet.h_fem_fem_contact_start_id),
                    cfem_vals = gipc_global_triplet.block_values(
                        gipc_global_triplet.h_fem_fem_contact_start_id),
                    BDType = TetMesh.BoundaryType,
                    fem_global_hessian_index_offset] __device__(int i) mutable
                   {
                       int row = cfem_rows[i];
                       int col = cfem_cols[i];
                       int btypeA = BDType[row];
                       int btypeB = BDType[col];
                       if(row <= col)
                       {
                           cfem_rows[i] = row + fem_global_hessian_index_offset;
                           cfem_cols[i] = col + fem_global_hessian_index_offset;
                           if(btypeA != 0 || btypeB != 0)
                           {
                               cfem_vals[i].setZero();
                           }
                       }
                       else
                       {
                           cfem_rows[i] = col + fem_global_hessian_index_offset;
                           cfem_cols[i] = row + fem_global_hessian_index_offset;
                           cfem_vals[i].setZero();
                       }
                   });
    }

    {
        gipc::Timer timer{"cal_fem_gradient_hessian"};
        int fem_triplet_start = gipc_global_triplet.global_triplet_offset;
        //CUDA_SAFE_CALL(cudaDeviceSynchronize());
        // [multi-env determinism 4.3] open the elastic-gradient binned bracket: FEM, bending,
        // triangle-FEM, strain-limiting AND soft-constraint all scatter to g_gbin; combined
        // into shape_grads after soft (kinetic is already in shape_grads, written directly).
        zeroBinnedGrad();
        calculate_fem_gradient_hessian(TetMesh.DmInverses,
                                       TetMesh.vertexes,
                                       TetMesh.tetrahedras,
                                       TetMesh.volum,
                                       shape_grads,
                                       abd_fem_count_info.fem_tet_num,
                                       abd_fem_count_info.abd_tet_num,
                                       TetMesh.lengthRate,
                                       TetMesh.volumeRate,
                                       gipc_global_triplet.global_triplet_offset,
                                       gipc_global_triplet.block_values(),
                                       gipc_global_triplet.block_row_indices(),
                                       gipc_global_triplet.block_col_indices(),
                                       IPC_dt,
                                       fem_global_hessian_index_offset,
                                       TetMesh.d_tet_to_abd_body);
        gipc_global_triplet.global_triplet_offset += abd_fem_count_info.fem_tet_num * 10;


#ifdef USE_QUADRATIC_BENDING
        calculate_quad_bending_gradient_hessian(TetMesh.vertexes,
                                                TetMesh.rest_vertexes,
                                                TetMesh.tri_edges,
                                                TetMesh.tri_edge_adj_vertex,
                                                TetMesh.quad_bending_Q,
                                                shape_grads,
                                                tri_edge_num,
                                                bendStiff,
                                                gipc_global_triplet.global_triplet_offset,
                                                gipc_global_triplet.block_values(),
                                                gipc_global_triplet.block_row_indices(),
                                                gipc_global_triplet.block_col_indices(),
                                                IPC_dt,
                                                fem_global_hessian_index_offset);
#else
        calculate_bending_gradient_hessian(TetMesh.vertexes,
                                           TetMesh.rest_vertexes,
                                           TetMesh.tri_edges,
                                           TetMesh.tri_edge_adj_vertex,
                                           shape_grads,
                                           tri_edge_num,
                                           bendStiff,
                                           gipc_global_triplet.global_triplet_offset,
                                           gipc_global_triplet.block_values(),
                                           gipc_global_triplet.block_row_indices(),
                                           gipc_global_triplet.block_col_indices(),
                                           IPC_dt,
                                           fem_global_hessian_index_offset);
#endif
        gipc_global_triplet.global_triplet_offset += tri_edge_num * 10;
        //CUDA_SAFE_CALL(cudaDeviceSynchronize());

        calculate_triangle_fem_gradient_hessian(TetMesh.triDmInverses,
                                                TetMesh.vertexes,
                                                TetMesh.triangles,
                                                TetMesh.area,
                                                shape_grads,
                                                triangleNum,
                                                stretchStiff,
                                                shearStiff,
                                                strainRate,
                                                gipc_global_triplet.global_triplet_offset,
                                                gipc_global_triplet.block_values(),
                                                gipc_global_triplet.block_row_indices(),
                                                gipc_global_triplet.block_col_indices(),
                                                IPC_dt,
                                                fem_global_hessian_index_offset);

        gipc_global_triplet.global_triplet_offset += triangleNum * 6;


        // [decouple probe] sub-stage e_presoft: kinetic+FEM+bending+triangle (NO soft yet).
        // Combine current bins into a scratch copy (does NOT disturb shape_grads or the bins).
        if(getenv("STIFF_SHAPE_STAGE") && getenv("STIFF_GRAD_PRE") && TetMesh.d_point_to_group
           && g_dec_frame == atoi(getenv("STIFF_DUMP_FRAME"))
           && g_dec_k == (getenv("STIFF_PROBE_K") ? atoi(getenv("STIFF_PROBE_K")) : 0))
        {
            double3* scratch = TetMesh.temp_double3Mem;
            CUDA_SAFE_CALL(cudaMemcpy(scratch, shape_grads, vertexNum*sizeof(double3), cudaMemcpyDeviceToDevice));
            combineBinnedGrad(scratch);   // scratch = kinetic + FEM+bending+triangle (bins so far)
            std::vector<double3> h(vertexNum);
            CUDA_SAFE_CALL(cudaMemcpy(h.data(), scratch, vertexNum*sizeof(double3), cudaMemcpyDeviceToHost));
            FILE* a=fopen((std::string(getenv("STIFF_GRAD_PRE"))+".epresoft").c_str(),"wb");
            if(a){fwrite(h.data(),sizeof(double3),vertexNum,a);fclose(a);}
            printf("[shape-stage] e_presoft (kin+fem+bend+tri, NO soft) dumped @frame %d k=%d\n", g_dec_frame, g_dec_k);
        }

        // [multi-env determinism 4.3] soft constraint is the LAST elastic-side gradient; it
        // also scatters to g_gbin. Close the bracket: combine all of FEM+bending+triangle+
        // strain+soft into shape_grads (deterministic).
        computeSoftConstraintGradientAndHessian(shape_grads, fem_global_hessian_index_offset);
        combineBinnedGrad(shape_grads);
        if(getenv("STIFF_XENV") && m_d_p2g) xenvDiff(shape_grads, "  d.elastic(fem+bend+tri+soft)");
        // [decouple probe] sub-stage 2: shape_grads = kinetic + elastic-bracket (FEM+bend+tri+strain+soft).
        if(getenv("STIFF_SHAPE_STAGE") && getenv("STIFF_GRAD_PRE") && TetMesh.d_point_to_group
           && g_dec_frame == atoi(getenv("STIFF_DUMP_FRAME"))
           && g_dec_k == (getenv("STIFF_PROBE_K") ? atoi(getenv("STIFF_PROBE_K")) : 0))
        {
            std::vector<double3> h(vertexNum);
            CUDA_SAFE_CALL(cudaMemcpy(h.data(), shape_grads, vertexNum*sizeof(double3), cudaMemcpyDeviceToHost));
            FILE* a=fopen((std::string(getenv("STIFF_GRAD_PRE"))+".s2").c_str(),"wb");
            if(a){fwrite(h.data(),sizeof(double3),vertexNum,a);fclose(a);}
            printf("[shape-stage] s2 (kinetic+elastic) dumped @frame %d k=%d\n", g_dec_frame, g_dec_k);
        }
        gipc_global_triplet.global_triplet_offset += softNum;
        KSEG("seg_end_soft")

        int fem_triplet_num = gipc_global_triplet.global_triplet_offset - fem_triplet_start;

        // [M3.5 substitution method] Chain-rule pinned-vertex FEM Hessian
        // rows/cols onto the ABD body's q-DOFs.  After this:
        //   pinned-pinned block H_pp  -> J_p^T * H_pp * J_p added at (body*4+r, body*4+c)
        //   pinned-free  block H_pf   -> J_p^T * H_pf added at (body*4+r, free_v_col)
        //   free-pinned  block H_fp   -> H_fp * J_p added at (free_v_row, body*4+c)
        //   in all cases, original (pinned-touching) triplet is zeroed.
        //
        // Without this routing, PCG sees pinned vertex as a free DOF whose
        // dx is later overridden by apply_fem_pins; the free-vertex dx is
        // computed against an incorrect dx_p estimate (elasticity-driven
        // instead of J*dq), and Newton fails to converge under joint
        // motion (k=1000 cap, verified by m5_drive_joint2_test).  After
        // routing, the substituted system has the pinned DOF's contribution
        // baked into the ABD body row, which Newton can solve consistently.
        //
        // Output goes to an "extension range" past the current
        // global_triplet_offset; we use an atomic counter for slot
        // allocation, then bump global_triplet_offset by the final count.
        if(TetMesh.n_fem_pins > 0 && TetMesh.vertex_to_pin_idx != nullptr)
        {
            int ext_start    = gipc_global_triplet.global_triplet_offset;
            // Reserve space (worst case: 16 expansions per pinned-touching triplet).
            // Actual count tracked via atomic counter.
            int ext_capacity = fem_triplet_num * 16;

            // Reset counter device-side.  Reuse the existing
            // d_unique_key_number scratch int* on GIPCTripletMatrix.
            CUDA_SAFE_CALL(cudaMemsetAsync(gipc_global_triplet.d_unique_key_number,
                                           0, sizeof(int)));

            muda::ParallelFor(256)
                .file_line(__FILE__, __LINE__)
                .apply(fem_triplet_num,
                       [cfem_rows = gipc_global_triplet.block_row_indices(fem_triplet_start),
                        cfem_cols = gipc_global_triplet.block_col_indices(fem_triplet_start),
                        triplet_fem = gipc_global_triplet.block_values(fem_triplet_start),
                        ext_rows = gipc_global_triplet.block_row_indices(ext_start),
                        ext_cols = gipc_global_triplet.block_col_indices(ext_start),
                        ext_vals = gipc_global_triplet.block_values(ext_start),
                        ext_count = gipc_global_triplet.d_unique_key_number,
                        BDType   = TetMesh.BoundaryType,
                        v2pin    = TetMesh.vertex_to_pin_idx,
                        pin_body = TetMesh.d_fem_pin_abd_body_id,
                        pin_lo   = TetMesh.d_fem_pin_abd_local_pos,
                        hess_index2fem_index = fem_global_hessian_index_offset,
                        ext_capacity] __device__(int i) mutable
                       {
                           int row = cfem_rows[i];
                           int col = cfem_cols[i];
                           int row_v = row - hess_index2fem_index;
                           int col_v = col - hess_index2fem_index;
                           int btypeA = BDType[row_v];
                           int btypeB = BDType[col_v];
                           if(btypeA == 0 && btypeB == 0)
                               return;  // both free: keep original

                           // Read original block, then zero it (will be replaced
                           // with chain-ruled triplets in extension range).
                           gipc::Matrix3x3 H = triplet_fem[i];
                           triplet_fem[i].setZero();

                           int pin_a = (btypeA != 0) ? v2pin[row_v] : -1;
                           int pin_b = (btypeB != 0) ? v2pin[col_v] : -1;

                           // Helper: append a triplet to extension range.
                           auto append = [&](int r, int c, const gipc::Matrix3x3& V) {
                               int slot = atomicAdd(ext_count, 1);
                               if(slot < ext_capacity) {
                                   ext_rows[slot] = r;
                                   ext_cols[slot] = c;
                                   ext_vals[slot] = V;
                               }
                           };

                           if(pin_a >= 0 && pin_b < 0)
                           {
                               // Row pinned, col free: write 4 triplets at
                               // (body*4+r, col) for r=0..3.
                               // J^T * H (12x3) split into 4 (3x3) sub-blocks.
                               int     body = pin_body[pin_a];
                               double3 lo3  = pin_lo[pin_a];
                               // Sub-block 0 = H itself
                               append(body * 4 + 0, col, H);
                               // Sub-block r (r=1,2,3) = lo * H[r-1, :]^T (outer product)
                               // == column-vec lo times row r-1 of H
                               #pragma unroll
                               for(int r = 1; r < 4; ++r)
                               {
                                   gipc::Matrix3x3 B;
                                   double Hr0 = H(r - 1, 0), Hr1 = H(r - 1, 1), Hr2 = H(r - 1, 2);
                                   B(0, 0) = lo3.x * Hr0; B(0, 1) = lo3.x * Hr1; B(0, 2) = lo3.x * Hr2;
                                   B(1, 0) = lo3.y * Hr0; B(1, 1) = lo3.y * Hr1; B(1, 2) = lo3.y * Hr2;
                                   B(2, 0) = lo3.z * Hr0; B(2, 1) = lo3.z * Hr1; B(2, 2) = lo3.z * Hr2;
                                   int new_row = body * 4 + r;
                                   if(new_row <= col) append(new_row, col, B);
                                   else               append(col, new_row, B.transpose());
                               }
                           }
                           else if(pin_a < 0 && pin_b >= 0)
                           {
                               // Col pinned, row free: write 4 triplets routed to
                               // (row, body*4+c) for c=0..3.  But row > body*4+c
                               // typically (FEM row is in [N_abd*4, ...) range and
                               // body*4+c is in [0, N_abd*4)).  So store at
                               // (body*4+c, row) with TRANSPOSED block to keep
                               // upper-triangle convention.
                               // H * J (3x12) split into 4 (3x3) sub-blocks per col.
                               int     body = pin_body[pin_b];
                               double3 lo3  = pin_lo[pin_b];
                               // Sub-block 0 (cols 0-2) = H itself
                               // Stored at (body*4+0, row) transposed = H.transpose()
                               int new_col = body * 4 + 0;
                               if(new_col <= row) append(new_col, row, H.transpose());
                               else               append(row, new_col, H);
                               // Sub-block c (c=1,2,3) = H[:, c-1] * lo^T (outer)
                               // Stored at (body*4+c, row) transposed = lo * H[:, c-1]^T
                               #pragma unroll
                               for(int c = 1; c < 4; ++c)
                               {
                                   gipc::Matrix3x3 B;  // = H[:, c-1] outer lo
                                   double H0 = H(0, c - 1), H1 = H(1, c - 1), H2 = H(2, c - 1);
                                   B(0, 0) = H0 * lo3.x; B(0, 1) = H0 * lo3.y; B(0, 2) = H0 * lo3.z;
                                   B(1, 0) = H1 * lo3.x; B(1, 1) = H1 * lo3.y; B(1, 2) = H1 * lo3.z;
                                   B(2, 0) = H2 * lo3.x; B(2, 1) = H2 * lo3.y; B(2, 2) = H2 * lo3.z;
                                   // B is original (row, body*4+c). Transposed = (body*4+c, row)
                                   int nc = body * 4 + c;
                                   if(nc <= row) append(nc, row, B.transpose());
                                   else          append(row, nc, B);
                               }
                           }
                           else  // both pinned
                           {
                               // The original triplet (p1, p2, H) with p1<p2
                               // represents H_{p1,p2}=H AND H_{p2,p1}=H^T (sym).
                               // After substitution x_p1=J_a*q_a, x_p2=J_b*q_b:
                               //   y_a += J_a^T * H * J_b * q_b   (path 1)
                               //   y_b += J_b^T * H^T * J_a * q_a (path 2, = transpose of path 1)
                               //
                               // For SAME body (body_a==body_b==body), both paths
                               // target body's diagonal:
                               //   total = J_a^T*H*J_b + (J_a^T*H*J_b)^T (symmetric)
                               // Sym storage upper-tri at body's (r,c) sub-block:
                               //   For r<=c: store M12.block(r,c) + M12.block(c,r)^T
                               //
                               // For DIFFERENT bodies (body_a < body_b), the paths
                               // target distinct off-diagonal block (body_a, body_b)
                               // with M12 stored once; sym SpMV via M^T handles the
                               // implicit (body_b, body_a) direction.
                               int     body_a = pin_body[pin_a];
                               int     body_b = pin_body[pin_b];
                               double3 lo_a   = pin_lo[pin_a];
                               double3 lo_b   = pin_lo[pin_b];
                               gipc::Vector3   xa{lo_a.x, lo_a.y, lo_a.z};
                               gipc::Vector3   xb{lo_b.x, lo_b.y, lo_b.z};
                               gipc::ABDJacobi   Ja(xa), Jb(xb);
                               gipc::Matrix12x12 M12 =
                                   gipc::ABDJacobi::JT_H_J(Ja.T(), H, Jb);
                               if(body_a == body_b)
                               {
                                   #pragma unroll
                                   for(int r = 0; r < 4; ++r)
                                   {
                                       #pragma unroll
                                       for(int c = r; c < 4; ++c)
                                       {
                                           gipc::Matrix3x3 B =
                                               M12.block<3, 3>(r * 3, c * 3)
                                               + M12.block<3, 3>(c * 3, r * 3).transpose();
                                           append(body_a * 4 + r, body_a * 4 + c, B);
                                       }
                                   }
                               }
                               else
                               {
                                   // body_a != body_b: store M12 (no symmetrization)
                                   // at (body_a*4+r, body_b*4+c) for body_a < body_b
                                   // (so always upper-tri); transpose if reversed.
                                   bool a_lt_b = (body_a < body_b);
                                   #pragma unroll
                                   for(int r = 0; r < 4; ++r)
                                   {
                                       #pragma unroll
                                       for(int c = 0; c < 4; ++c)
                                       {
                                           gipc::Matrix3x3 B =
                                               M12.block<3, 3>(r * 3, c * 3);
                                           if(a_lt_b)
                                               append(body_a * 4 + r, body_b * 4 + c, B);
                                           else
                                               append(body_b * 4 + c, body_a * 4 + r, B.transpose());
                                       }
                                   }
                               }
                           }
                       });

            // Read final extension count and bump triplet offset.
            int h_ext_count = 0;
            CUDA_SAFE_CALL(cudaMemcpy(&h_ext_count,
                                      gipc_global_triplet.d_unique_key_number,
                                      sizeof(int),
                                      cudaMemcpyDeviceToHost));
            if(h_ext_count > ext_capacity)
            {
                if(g_gipc_log_level >= 1) printf("[M3.5] WARN ext_count=%d > capacity=%d (truncated; expect "
                       "Newton instability)\n", h_ext_count, ext_capacity);
                h_ext_count = ext_capacity;
            }
            gipc_global_triplet.global_triplet_offset += h_ext_count;
        }
        else
        {
            // No pins: use the original simple zeroing logic for
            // non-zero BoundaryType (e.g. user-defined boundary conditions).
            muda::ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(fem_triplet_num,
                       [cfem_rows = gipc_global_triplet.block_row_indices(fem_triplet_start),
                        cfem_cols = gipc_global_triplet.block_col_indices(fem_triplet_start),
                        triplet_fem = gipc_global_triplet.block_values(fem_triplet_start),
                        BDType   = TetMesh.BoundaryType,
                        hess_index2fem_index = fem_global_hessian_index_offset] __device__(int i) mutable
                       {
                           int row    = cfem_rows[i];
                           int col    = cfem_cols[i];
                           int btypeA = BDType[row - hess_index2fem_index];
                           int btypeB = BDType[col - hess_index2fem_index];
                           if(btypeA != 0 || btypeB != 0)
                           {
                               triplet_fem[i].setZero();
                           }
                       });
        }


        //int massNum =
        muda::ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(abd_fem_count_info.fem_point_num,
                   [mass      = TetMesh.masses,
                    cfem_rows = gipc_global_triplet.block_row_indices(
                        gipc_global_triplet.global_triplet_offset),
                    cfem_cols = gipc_global_triplet.block_col_indices(
                        gipc_global_triplet.global_triplet_offset),
                    triplet_fem = gipc_global_triplet.block_values(
                        gipc_global_triplet.global_triplet_offset),
                    fem_global_hessian_index_offset,
                    fem_pint_start = abd_fem_count_info.abd_point_num,
                    abd_num = abd_fem_count_info.abd_body_num] __device__(int i) mutable
                   {
                       triplet_fem[i] =
                           mass[i + fem_pint_start] * gipc::Matrix3x3::Identity();
                       cfem_rows[i] = i + abd_num * 4;
                       cfem_cols[i] = i + abd_num * 4;
                   });
        gipc_global_triplet.global_triplet_offset += abd_fem_count_info.fem_point_num;

        //cudaMemcpy(TetMesh.totalForce, contact_grads, vertexNum * sizeof(double3), cudaMemcpyDeviceToDevice);
        //getTotalForce(shape_grads, TetMesh.totalForce);
    }

    // [M2 substitution method] Chain-rule pinned FEM vertex gradient to ABD body q-DOFs.
    // For each pinned vertex p (body b, rest-frame local_pos lo):
    //   ABD gradient += J_p^T * (shape_grads[p] + fb[p])
    // where J_p^T * g = [g; lo.x*g; lo.y*g; lo.z*g]  (from ABDJacobiT operator*).
    // FEMLinearSubsystem::assemble() already zeros pinned DOFs in the PCG RHS via
    // BoundaryType check, so no double-counting occurs.
    if(TetMesh.n_fem_pins > 0 && m_abd_system && m_d_abd_body_q != nullptr)
    {
        m_abd_system->couple_bin_open((int)m_abd_system->system_gradient.size());  // [4.3] bin the coupling
        muda::ParallelFor(256)
            .file_line(__FILE__, __LINE__)
            .apply(TetMesh.n_fem_pins,
                   [sys_grad    = m_abd_system->system_gradient.viewer(),
                    shape_grads = TetMesh.shape_grads,
                    fb          = TetMesh.fb,
                    pin_fem_v   = TetMesh.d_fem_pin_fem_vertex,
                    pin_body_id = TetMesh.d_fem_pin_abd_body_id,
                    pin_lo      = TetMesh.d_fem_pin_abd_local_pos] __device__(int i) mutable
                   {
                       int     fem_v   = pin_fem_v[i];
                       int     body_id = pin_body_id[i];
                       double3 lo      = pin_lo[i];

                       double gx = shape_grads[fem_v].x + fb[fem_v].x;
                       double gy = shape_grads[fem_v].y + fb[fem_v].y;
                       double gz = shape_grads[fem_v].z + fb[fem_v].z;

                       // J_p^T * [gx, gy, gz]:
                       // segment [0:3]  = g
                       // segment [3:6]  = lo * gx
                       // segment [6:9]  = lo * gy
                       // segment [9:12] = lo * gz
                       gipc::Vector12 g12;
                       g12(0)  = gx;        g12(1)  = gy;        g12(2)  = gz;
                       g12(3)  = lo.x * gx; g12(4)  = lo.y * gx; g12(5)  = lo.z * gx;
                       g12(6)  = lo.x * gy; g12(7)  = lo.y * gy; g12(8)  = lo.z * gy;
                       g12(9)  = lo.x * gz; g12(10) = lo.y * gz; g12(11) = lo.z * gz;

                       for(int c = 0; c < 12; ++c)
                           _binDepBase(g_abd_sysbin + ((size_t)body_id * 12 + c) * BINNED_K, g12(c));
                   });
        m_abd_system->couple_bin_close((int)m_abd_system->system_gradient.size());  // [4.3] combine
    }

    // [M3 substitution method] Add J^T * (m * I) * J to the global Hessian at
    // the pinned ABD body's diagonal block.  Writes the UPPER TRIANGLE only
    // (10 triplets per pin), matching write_abd_body_hessian's storage
    // convention.  The CSR converter sums duplicates with the ABD body's
    // own 10 triplets at the same (i,j) positions.
    //
    // KNOWN LIMITATION: only the inertia term (mass*I) is chain-ruled.
    // The FEM elasticity Hessian's cross-terms H_fp * J_p (free-free row,
    // ABD col) and their transposes are NOT chain-ruled — the BoundaryType
    // zeroing at line 10644-10655 drops them.  Without these cross-terms,
    // PCG decouples FEM and ABD: ABD moves q ignoring elasticity pull-back
    // from the free FEM vertices, and Newton fails to converge under
    // joint-driven motion (k=1000 cap hit, verified via m5_drive_joint2_test).
    //
    // For static gripper-close scenarios M2 + M3 inertia is sufficient
    // (Newton k=1-2).  For dynamic joint motion the user should set
    // USE_HARD_PIN=0 (stitch spring) until full elasticity chain-rule is
    // implemented (TODO M3.5).
    if(TetMesh.n_fem_pins > 0)
    {
        int triplet_offset_start = gipc_global_triplet.global_triplet_offset;
        muda::ParallelFor(256)
            .file_line(__FILE__, __LINE__)
            .apply(TetMesh.n_fem_pins,
                   [tri_rows = gipc_global_triplet.block_row_indices(triplet_offset_start),
                    tri_cols = gipc_global_triplet.block_col_indices(triplet_offset_start),
                    tri_vals = gipc_global_triplet.block_values(triplet_offset_start),
                    pin_fem_v   = TetMesh.d_fem_pin_fem_vertex,
                    pin_body_id = TetMesh.d_fem_pin_abd_body_id,
                    pin_lo      = TetMesh.d_fem_pin_abd_local_pos,
                    masses      = TetMesh.masses] __device__(int i) mutable
                   {
                       int     fem_v   = pin_fem_v[i];
                       int     body_id = pin_body_id[i];
                       double3 lo3     = pin_lo[i];
                       double  m       = masses[fem_v];

                       gipc::Vector3   lo{lo3.x, lo3.y, lo3.z};
                       gipc::Matrix3x3 mI = m * gipc::Matrix3x3::Identity();
                       gipc::ABDJacobi   J(lo);
                       gipc::Matrix12x12 H =
                           gipc::ABDJacobi::JT_H_J(J.T(), mI, J);

                       int slot_base = i * 10;
                       int kk = 0;
                       #pragma unroll
                       for(int r = 0; r < 4; ++r)
                       {
                           #pragma unroll
                           for(int c = r; c < 4; ++c)
                           {
                               int slot           = slot_base + kk;
                               tri_rows[slot]     = body_id * 4 + r;
                               tri_cols[slot]     = body_id * 4 + c;
                               tri_vals[slot]     = H.block<3, 3>(r * 3, c * 3);
                               kk++;
                           }
                       }
                   });
        gipc_global_triplet.global_triplet_offset += TetMesh.n_fem_pins * 10;
    }

    return time00;
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
}


double GIPC::Energy_Add_Reduction_Algorithm(int type, device_TetraData& TetMesh)
{
    int tet_offset   = abd_fem_count_info.fem_tet_offset;
    int tet_count    = abd_fem_count_info.fem_tet_num;
    int point_offset = abd_fem_count_info.fem_point_offset;
    int point_count  = abd_fem_count_info.fem_point_num;

    int numbers = tet_count;

    if(type == 0 || type == 3)
    {
        numbers = point_count;
    }
    else if(type == 2)
    {
        numbers = h_cpNum[0];
    }
    else if(type == 4)
    {
        numbers = h_gpNum;
    }
    else if(type == 5)
    {
        numbers = h_cpNum_last[0];
    }
    else if(type == 6)
    {
        numbers = h_gpNum_last;
    }
    else if(type == 7 || type == 1)
    {
        numbers = tet_count;
    }
    else if(type == 8 || type == 11)
    {
        numbers = triangleNum;
    }
    else if(type == 9)
    {
        numbers = softNum;
    }
    else if(type == 10)
    {
        numbers = tri_edge_num;
    }
    if(numbers == 0)
        return 0;
    // pair-count energy reductions (barrier/friction) need a pair-sized buffer,
    // not the mesh-sized squeue (V2/V3 overflow fix).
    double* queue = ensure_reduce_scratch(numbers);
    //CUDA_SAFE_CALL(cudaMalloc((void**)&queue, numbers * sizeof(double)));*/

    const unsigned int threadNum = 256;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);
    switch(type)
    {
        case 0:
            _getKineticEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                TetMesh.vertexes + point_offset,
                TetMesh.xTilta + point_offset,
                queue,
                TetMesh.masses + point_offset,
                numbers);
            break;
        case 1:
            _getFEMEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue,
                TetMesh.vertexes,
                TetMesh.tetrahedras + tet_offset,
                TetMesh.DmInverses + tet_offset,
                TetMesh.volum + tet_offset,
                numbers,
                TetMesh.lengthRate + tet_offset,
                TetMesh.volumeRate + tet_offset);
            break;
        case 2:
            _getBarrierEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.vertexes, TetMesh.rest_vertexes, _collisonPairs, Kappa, dHat, numbers);
            break;
        case 3:
            _getDeltaEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.fb + point_offset, _moveDir + point_offset, numbers);
            break;
        case 4:
            _computeGroundEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.vertexes, _groundOffset, _groundNormal, _environment_collisionPair, dHat, Kappa, numbers);
            break;
        case 5:
            _getFrictionEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue,
                TetMesh.vertexes,
                TetMesh.o_vertexes,
                _collisonPairs_lastH,
                numbers,
                IPC_dt,
                distCoord,
                tanBasis,
                lambda_lastH_scalar,
                fDhat * IPC_dt * IPC_dt,
                sqrt(fDhat) * IPC_dt);
            break;
        case 6:
            _getFrictionEnergy_gd_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue,
                TetMesh.vertexes,
                TetMesh.o_vertexes,
                _groundNormal,
                _collisonPairs_lastH_gd,
                numbers,
                IPC_dt,
                lambda_lastH_scalar_gd,
                sqrt(fDhat) * IPC_dt);
            break;
        case 7:
            _getRestStableNHKEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.volum + tet_offset, numbers, lengthRate, volumeRate);
            break;
        case 8:
            _get_triangleFEMEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue,
                TetMesh.vertexes,
                TetMesh.triangles,
                TetMesh.triDmInverses,
                TetMesh.area,
                numbers,
                stretchStiff,
                shearStiff,
                strainRate);
            break;
        case 9:
            _computeSoftConstraintEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.vertexes, TetMesh.targetVert, TetMesh.targetIndex, softMotionRate, animation_fullRate,
                TetMesh.d_stitch_paired_vertex, TetMesh.d_stitch_rest_offset, numbers);
            break;
        case 10:
#ifdef USE_QUADRATIC_BENDING
            _getQuadBendingEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(
                queue,
                TetMesh.vertexes,
                TetMesh.rest_vertexes,
                TetMesh.tri_edges,
                TetMesh.tri_edge_adj_vertex,
                TetMesh.quad_bending_Q,
                numbers,
                bendStiff);
#else
            _getBendingEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(
                queue,
                TetMesh.vertexes,
                TetMesh.rest_vertexes,
                TetMesh.tri_edges,
                TetMesh.tri_edge_adj_vertex,
                numbers,
                bendStiff);
#endif
            break;
    }
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while(numbers > 1)
    {
        __add_reduction<<<blockNum, threadNum, sharedMsize>>>(queue, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    double result;
    cudaMemcpy(&result, queue, sizeof(double), cudaMemcpyDeviceToHost);
    //CUDA_SAFE_CALL(cudaFree(queue));
    return result;
}


// [backport] standalone per-env energy dispatcher (from S3/4a370e5). Modeled on
// Energy_Add_Reduction_Algorithm but writes the global scalar to a caller-provided
// device slot (out_slot, D2D) and, when out_penv != nullptr, buckets each element's
// energy into per-env slots via the kernels' (penv,p2g,ng) params. Used ONLY by
// computeEnergy_perenv (the per-env line-search path); v0.6.4's computeEnergy is
// untouched. NOTE: not the full ②-D2H batched computeEnergy rewrite — isolated here.
void GIPC::Energy_Add_Reduction_Algorithm_DeviceOut(int               type,
                                                     device_TetraData& TetMesh,
                                                     double*           out_slot,
                                                     double*           out_penv)
{
    // [multi-env S3] per-env energy bucket for this term (size kEnvAlphaSlots) and
    // the global point_to_group; passed to the kernels when out_penv != nullptr.
    double*    pe  = out_penv;
    const int* p2g = TetMesh.d_point_to_group;
    const int  ng  = kEnvAlphaSlots;
    if(pe) CUDA_SAFE_CALL(cudaMemsetAsync(pe, 0, ng * sizeof(double)));
    int tet_offset   = abd_fem_count_info.fem_tet_offset;
    int tet_count    = abd_fem_count_info.fem_tet_num;
    int point_offset = abd_fem_count_info.fem_point_offset;
    int point_count  = abd_fem_count_info.fem_point_num;

    int numbers = tet_count;
    if(type == 0 || type == 3)      numbers = point_count;
    else if(type == 2)              numbers = h_cpNum[0];
    else if(type == 4)              numbers = h_gpNum;
    else if(type == 5)              numbers = h_cpNum_last[0];
    else if(type == 6)              numbers = h_gpNum_last;
    else if(type == 7 || type == 1) numbers = tet_count;
    else if(type == 8 || type == 11)numbers = triangleNum;
    else if(type == 9)              numbers = softNum;
    else if(type == 10)             numbers = tri_edge_num;

    if(numbers == 0)
    {
        // Match original `return 0;` behavior — pre-zero the slot.
        CUDA_SAFE_CALL(cudaMemsetAsync(out_slot, 0, sizeof(double)));
        return;
    }

    // Pair-count types (2/5) can exceed the mesh-sized squeue block capacity —
    // use the growable pair-capacity scratch (V2/V3 overflow fix), like the
    // blocking variant.
    double*            queue       = ensure_reduce_scratch(numbers);
    const unsigned int threadNum   = 256;
    int                blockNum    = (numbers + threadNum - 1) / threadNum;
    unsigned int       sharedMsize = sizeof(double) * (threadNum >> 5);

    switch(type)
    {
        case 0:
            _getKineticEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                TetMesh.vertexes + point_offset, TetMesh.xTilta + point_offset,
                queue, TetMesh.masses + point_offset, numbers,
                pe, pe ? p2g + point_offset : nullptr, ng);
            break;
        case 1:
            _getFEMEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.vertexes, TetMesh.tetrahedras + tet_offset,
                TetMesh.DmInverses + tet_offset, TetMesh.volum + tet_offset,
                numbers, TetMesh.lengthRate + tet_offset, TetMesh.volumeRate + tet_offset,
                pe, pe ? p2g : nullptr, ng);
            break;
        case 2:
            _getBarrierEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.vertexes, TetMesh.rest_vertexes, _collisonPairs, Kappa, dHat, numbers,
                pe, pe ? p2g : nullptr, ng);
            break;
        case 3:
            _getDeltaEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.fb + point_offset, _moveDir + point_offset, numbers);
            break;
        case 4:
            _computeGroundEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.vertexes, _groundOffset, _groundNormal,
                _environment_collisionPair, dHat, Kappa, numbers,
                pe, pe ? p2g : nullptr, ng);
            break;
        case 5:
            _getFrictionEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.vertexes, TetMesh.o_vertexes, _collisonPairs_lastH,
                numbers, IPC_dt, distCoord, tanBasis, lambda_lastH_scalar,
                fDhat * IPC_dt * IPC_dt, sqrt(fDhat) * IPC_dt,
                pe, pe ? p2g : nullptr, ng);
            break;
        case 6:
            _getFrictionEnergy_gd_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.vertexes, TetMesh.o_vertexes, _groundNormal,
                _collisonPairs_lastH_gd, numbers, IPC_dt, lambda_lastH_scalar_gd,
                sqrt(fDhat) * IPC_dt,
                pe, pe ? p2g : nullptr, ng);
            break;
        case 7:
            _getRestStableNHKEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.volum + tet_offset, numbers, lengthRate, volumeRate);
            break;
        case 8:
            _get_triangleFEMEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.vertexes, TetMesh.triangles, TetMesh.triDmInverses,
                TetMesh.area, numbers, stretchStiff, shearStiff, strainRate,
                pe, pe ? p2g : nullptr, ng);
            break;
        case 9:
            _computeSoftConstraintEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.vertexes, TetMesh.targetVert, TetMesh.targetIndex,
                softMotionRate, animation_fullRate, TetMesh.d_stitch_paired_vertex,
                TetMesh.d_stitch_rest_offset, numbers,
                pe, pe ? p2g : nullptr, ng);
            break;
        case 10:
#ifdef USE_QUADRATIC_BENDING
            _getQuadBendingEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.vertexes, TetMesh.rest_vertexes, TetMesh.tri_edges,
                TetMesh.tri_edge_adj_vertex, TetMesh.quad_bending_Q, numbers, bendStiff,
                pe, pe ? p2g : nullptr, ng);
#else
            _getBendingEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.vertexes, TetMesh.rest_vertexes, TetMesh.tri_edges,
                TetMesh.tri_edge_adj_vertex, numbers, bendStiff,
                pe, pe ? p2g : nullptr, ng);
#endif
            break;
    }

    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;
    while(numbers > 1)
    {
        __add_reduction<<<blockNum, threadNum, sharedMsize>>>(queue, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }

    // D2D copy queue[0] into the caller's slot — queued on PTDS, async.
    // Next call's reduction kernel will not start until this D2D completes
    // (stream ordering), so reusing `queue` for the next call is safe.
    CUDA_SAFE_CALL(cudaMemcpyAsync(out_slot, queue, sizeof(double),
                                   cudaMemcpyDeviceToDevice));
}


double GIPC::computeEnergy(device_TetraData& TetMesh)
{
    // ②-D2H: batch the 9 Energy_Add_Reduction_Algorithm calls (types
    // 0,1,2,4,5,6,8,9,10) into a single D2H. Each reduction writes its scalar
    // to a device slot via D2D (queued, async); ONE blocking D2H grabs all 9
    // at the end. Saves 8 blocking syncs per energy evaluation (called every
    // line-search trial).
    //
    // ABD energies (m_abd_system->cal_abd_*) are NOT batched here — they have
    // their own internal scratch + D2H. Future refactor target. We KEEP the
    // ORIGINAL host-side summation ORDER below so vertex checksum stays
    // bit-identical (FP add is non-associative).
    //
    // slot indices: 0=fem_kinetic 1=fem 2=tri_fem 3=bend 4=constraint
    //               5=barrier   6=ground 7=fric  8=fric_ground

    Energy_Add_Reduction_Algorithm_DeviceOut(0,  TetMesh, m_energy_slots + 0);
    Energy_Add_Reduction_Algorithm_DeviceOut(1,  TetMesh, m_energy_slots + 1);
    Energy_Add_Reduction_Algorithm_DeviceOut(8,  TetMesh, m_energy_slots + 2);
    Energy_Add_Reduction_Algorithm_DeviceOut(10, TetMesh, m_energy_slots + 3);
    Energy_Add_Reduction_Algorithm_DeviceOut(9,  TetMesh, m_energy_slots + 4);
    Energy_Add_Reduction_Algorithm_DeviceOut(2,  TetMesh, m_energy_slots + 5);
    Energy_Add_Reduction_Algorithm_DeviceOut(4,  TetMesh, m_energy_slots + 6);
#ifdef USE_FRICTION
    Energy_Add_Reduction_Algorithm_DeviceOut(5,  TetMesh, m_energy_slots + 7);
    Energy_Add_Reduction_Algorithm_DeviceOut(6,  TetMesh, m_energy_slots + 8);
#endif

    double h_slots[9] = {0,0,0,0,0,0,0,0,0};
#ifdef USE_FRICTION
    CUDA_SAFE_CALL(cudaMemcpy(h_slots, m_energy_slots, 9 * sizeof(double),
                              cudaMemcpyDeviceToHost));
#else
    CUDA_SAFE_CALL(cudaMemcpy(h_slots, m_energy_slots, 7 * sizeof(double),
                              cudaMemcpyDeviceToHost));
#endif

    double Energy      = 0.0;
    auto   fem_kinetic = h_slots[0];
    Energy += fem_kinetic;

    auto abd_kinetic = m_abd_system->cal_abd_kinetic_energy(*m_abd_sim_data);
    Energy += abd_kinetic;

    auto abd_shape = m_abd_system->cal_abd_shape_energy(*m_abd_sim_data);
    Energy += abd_shape;

    auto abd_joint = m_abd_system->cal_abd_joint_energy(*m_abd_sim_data);
    Energy += abd_joint;

    auto abd_revolute_driving = m_abd_system->cal_abd_revolute_driving_energy(*m_abd_sim_data);
    Energy += abd_revolute_driving;

    auto abd_prismatic = m_abd_system->cal_abd_prismatic_energy(*m_abd_sim_data);
    Energy += abd_prismatic;

    auto abd_prismatic_driving = m_abd_system->cal_abd_prismatic_driving_energy(*m_abd_sim_data);
    Energy += abd_prismatic_driving;

    auto fem = IPC_dt * IPC_dt * h_slots[1];
    Energy += fem;

    auto tri_fem = IPC_dt * IPC_dt * h_slots[2];
    Energy += tri_fem;

    auto bend = IPC_dt * IPC_dt * h_slots[3];
    Energy += bend;

    auto constraint = h_slots[4];
    Energy += constraint;

    auto barrier = h_slots[5];
    Energy += barrier;

    auto ground = Kappa * h_slots[6];
    Energy += ground;

#ifdef USE_FRICTION
    auto fric = frictionRate * h_slots[7];
    Energy += fric;
    auto fric_ground = gd_frictionRate * h_slots[8];
    Energy += fric_ground;
#endif

    return Energy;
}

// [multi-env S3] per-env total energy E_g into env_out[kEnvAlphaSlots].
// FEM terms via the per-env-instrumented reductions (each element's energy added
// to its env bucket); ABD terms added per-env (task #2 — currently lumped into a
// validation-only global until per-body ABD energy lands). Returns the GLOBAL
// energy. Built-in correctness gate (gated STIFF_PENV_STATS): Sum_g E_g(FEM) must
// equal the global FEM energy (independent computeEnergy minus ABD).
// [de-CPU S3] device combine of the pe_all slices -> per-env energy E_g. One thread per env;
// accumulation ORDER AND FACTORS replicate the host combine exactly (kinetic, dt2*fem, dt2*tri,
// dt2*bend, constraint, barrier(kappa-rescaled), ground(kappa-scaled), friction, ABD) -> E_g is
// bit-identical to the host env_out. Slices: 0=kin 1=fem 2=tri 3=bend 4=cons 5=barrier 6=fricS
// 7=fricGd 8=ground 9=kappa_group 10=abd.
__global__ void _perenv_energy_combine(const double* pe, double* out, double Kappa, double dt2,
                                       double fr, double gfr, int perenv_k, int ng)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if(g >= ng) return;
    double kg = perenv_k ? pe[(size_t)9 * ng + g] : Kappa;
    double e  = 0.0;
    e += pe[(size_t)0 * ng + g];
    e += dt2 * pe[(size_t)1 * ng + g];
    e += dt2 * pe[(size_t)2 * ng + g];
    e += dt2 * pe[(size_t)3 * ng + g];
    e += pe[(size_t)4 * ng + g];
    e += (perenv_k ? kg / Kappa : 1.0) * pe[(size_t)5 * ng + g];
    e += (perenv_k ? kg : Kappa) * pe[(size_t)8 * ng + g];
#ifdef USE_FRICTION
    e += fr * pe[(size_t)6 * ng + g];
    e += gfr * pe[(size_t)7 * ng + g];
#endif
    e += pe[(size_t)10 * ng + g];
    out[g] = e;
}
// [de-CPU S3] per-env backtrack decision ON DEVICE, operating on the TRUE in-frame alpha state
// m_env_alpha (the old host loop read the h_env_alpha MIRROR, which is STALE in the fast S1 path —
// its failure branch would have H2D'd stale alphas over the device state; dormant only because S3
// virtually always accepts at bt=0). Same tolerance band and halving as the host loop.
__global__ void _s3_decide(const double* Eg0, const double* Eg1, double* env_alpha, int* nfail, int ng)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if(g >= ng) return;
    if(env_alpha[g] <= 0.0) return;               // absent (or frozen) env
    double tol = 1e-12 * (fabs(Eg0[g]) + 1.0);
    if(Eg1[g] > Eg0[g] + tol)
    {
        env_alpha[g] *= 0.5;
        atomicAdd(nfail, 1);
    }
}
// [de-CPU S3] intersect-safety halving (was: host loop over the stale mirror + H2D).
__global__ void _s3_halve_all(double* env_alpha, int ng)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if(g >= ng) return;
    if(env_alpha[g] > 0.0) env_alpha[g] *= 0.5;
}

// [de-CPU S3] shared term-launcher for the per-env energy: fills the pe_all slice block on device
// (see slice layout below) and reports whether per-env kappa rescale applies. Used by BOTH the
// host-combine (validation) and the device-combine (S3 decision) variants.
double* GIPC::_launch_perenv_energy_terms(device_TetraData& TetMesh, bool& perenv_k_out)
{
    const int NG = kEnvAlphaSlots;
    // [perf/de-CPU] SINGLE-D2H layout: every term's per-env bucket goes to its own NG-slice of one
    // device block; the κ-group + ABD arrays are staged into slices too; then ONE blocking D2H of
    // the whole block replaces the previous 12 per-term blocking D2H round-trips (the dominant
    // lineSearch host-mixing). The term kernels/launch order/host combine order+factors are
    // UNCHANGED → every env_out[g] is bit-identical to the per-term-D2H version.
    // slices: 0=kinetic 1=fem 2=tri_fem 3=bend 4=constraint 5=barrier 6=fricS 7=fricGd
    //         8=ground 9=kappa_group 10=abd
    constexpr int PE_SLOTS = 11;
    static double* pe_all = nullptr;
    if(!pe_all) CUDA_SAFE_CALL(cudaMalloc((void**)&pe_all, (size_t)PE_SLOTS * NG * sizeof(double)));
    // [backport] throwaway sink for DeviceOut's global scalar (ignored here; the
    // per-env path uses the pe slices, and full_sum is assembled on host below).
    static double* g_sink = nullptr;
    if(!g_sink) CUDA_SAFE_CALL(cudaMalloc((void**)&g_sink, sizeof(double)));

    auto slice = [&](int s) { return pe_all + (size_t)s * NG; };
    Energy_Add_Reduction_Algorithm_DeviceOut(0,  TetMesh, g_sink, slice(0));   // kinetic
    Energy_Add_Reduction_Algorithm_DeviceOut(1,  TetMesh, g_sink, slice(1));   // fem elastic
    Energy_Add_Reduction_Algorithm_DeviceOut(8,  TetMesh, g_sink, slice(2));   // tri_fem
    Energy_Add_Reduction_Algorithm_DeviceOut(10, TetMesh, g_sink, slice(3));   // bend
    Energy_Add_Reduction_Algorithm_DeviceOut(9,  TetMesh, g_sink, slice(4));   // constraint
    // [decouple] barrier + ground energy must use PER-ENV kappa (m_kappa_group), matching the
    // gradient. The global `Kappa` is a batch-dependent global reduction (initKappa), so scaling
    // env_0's barrier/ground energy by it makes env_0's per-env energy — and thus the per-env
    // line-search backtrack decision (h_env_alpha*=0.5 if Eg1>Eg0) — batch-dependent → env_0
    // drifts → chaos amplifies. Both energies are LINEAR in kappa (barrier: kernel applies global
    // Kappa → h=Kappa*raw; ground: kernel emits raw, host applies Kappa). Rescale to kappa_g.
    bool perenv_k = (getenv("STIFF_DECOUPLE_THRESH") && m_pergroup_kappa && m_kappa_group && Kappa > 0.0);
    Energy_Add_Reduction_Algorithm_DeviceOut(2,  TetMesh, g_sink, slice(5));   // barrier (Kappa-scaled)
    Energy_Add_Reduction_Algorithm_DeviceOut(4,  TetMesh, g_sink, slice(8));   // ground (raw)
#ifdef USE_FRICTION
    Energy_Add_Reduction_Algorithm_DeviceOut(5,  TetMesh, g_sink, slice(6));   // friction (self)
    Energy_Add_Reduction_Algorithm_DeviceOut(6,  TetMesh, g_sink, slice(7));   // friction (ground)
#endif
    if(perenv_k)   // κ-group staged into the block (D2D) — read back with the same single D2H
        CUDA_SAFE_CALL(cudaMemcpyAsync(slice(9), m_kappa_group, NG * sizeof(double),
                                       cudaMemcpyDeviceToDevice));
    // [S3] per-env ABD energy (segment-summed by body_to_group in the subsystem) → slice 10.
    CUDA_SAFE_CALL(cudaMemsetAsync(slice(10), 0, NG * sizeof(double)));
    double abd_total = m_abd_system->cal_abd_energy_perenv(
        *m_abd_sim_data, TetMesh.d_body_to_group, NG, slice(10));

    (void)abd_total;
    perenv_k_out = perenv_k;
    return pe_all;
}

double GIPC::computeEnergy_perenv(device_TetraData& TetMesh, std::vector<double>& env_out)
{
    const int NG = kEnvAlphaSlots;
    env_out.assign(NG, 0.0);
    if(!TetMesh.d_point_to_group)
        return computeEnergy(TetMesh);  // no groups -> nothing to decompose
    constexpr int PE_SLOTS = 11;
    bool    perenv_k = false;
    double* pe_all   = _launch_perenv_energy_terms(TetMesh, perenv_k);

    // THE one blocking D2H of everything.
    static std::vector<double> hb;
    hb.resize((size_t)PE_SLOTS * NG);
    CUDA_SAFE_CALL(cudaMemcpy(hb.data(), pe_all, (size_t)PE_SLOTS * NG * sizeof(double),
                              cudaMemcpyDeviceToHost));

    // Host combine — SAME order and factors as the per-term version (bit-identical env_out).
    const double dt2 = IPC_dt * IPC_dt;
    auto hs = [&](int s) { return hb.data() + (size_t)s * NG; };
    for(int g = 0; g < NG; ++g) env_out[g] += 1.0 * hs(0)[g];   // kinetic
    for(int g = 0; g < NG; ++g) env_out[g] += dt2 * hs(1)[g];   // fem elastic
    for(int g = 0; g < NG; ++g) env_out[g] += dt2 * hs(2)[g];   // tri_fem
    for(int g = 0; g < NG; ++g) env_out[g] += dt2 * hs(3)[g];   // bend
    for(int g = 0; g < NG; ++g) env_out[g] += 1.0 * hs(4)[g];   // constraint
    // barrier (h[g] = Kappa_global * raw_g): per-env → (kappa_g/Kappa_global)*h = kappa_g*raw_g
    for(int g = 0; g < NG; ++g)
        env_out[g] += (perenv_k ? hs(9)[g] / Kappa : 1.0) * hs(5)[g];
    // ground (h[g] = raw_g, kernel does not apply Kappa): per-env → kappa_g*raw_g
    for(int g = 0; g < NG; ++g)
        env_out[g] += (perenv_k ? hs(9)[g] : Kappa) * hs(8)[g];
#ifdef USE_FRICTION
    for(int g = 0; g < NG; ++g) env_out[g] += frictionRate * hs(6)[g];      // friction (self)
    for(int g = 0; g < NG; ++g) env_out[g] += gd_frictionRate * hs(7)[g];   // friction (ground)
#endif
    double abd_sum = 0.0;
    for(int g = 0; g < NG; ++g) { env_out[g] += hs(10)[g]; abd_sum += hs(10)[g]; }

    double full_sum = 0.0;
    for(int g = 0; g < NG; ++g) full_sum += env_out[g];

    // full_sum == global computeEnergy (validated machine-precision). Only run the
    // independent computeEnergy for the gated validation print (it ~doubles cost,
    // and the per-env backtracking loop calls this repeatedly).
    if(getenv("STIFF_S3_VALIDATE"))
    {
        double E_global = computeEnergy(TetMesh);
        printf("[S3-energy] sum_g E_g=%.9e  global=%.9e  rel=%.2e  (ABD sum_g=%.6e)\n",
               full_sum, E_global,
               fabs(full_sum - E_global) / std::max(fabs(E_global), 1e-30), abd_sum);
        return E_global;
    }
    return full_sum;
}

void GIPC::computeEnergy_perenv_dev(device_TetraData& TetMesh, double* d_Eg)
{
    const int NG = kEnvAlphaSlots;
    bool    perenv_k = false;
    double* pe_all   = _launch_perenv_energy_terms(TetMesh, perenv_k);
    _perenv_energy_combine<<<(NG + 255) / 256, 256>>>(
        pe_all, d_Eg, Kappa, IPC_dt * IPC_dt, frictionRate, gd_frictionRate, perenv_k ? 1 : 0, NG);
}

// [4.3 debug] FNV-1a hash of a device buffer (host-copy, deterministic) to bisect the residual
// non-atomic non-determinism. STIFF_KSUM=1 prints checksums; run twice + diff to localize.
static void _dbg_ksum(const char* name, const void* dptr, size_t nbytes)
{
    if(!getenv("STIFF_KSUM") || !dptr || nbytes == 0) return;
    std::vector<uint64_t> h((nbytes + 7) / 8, 0);
    cudaMemcpy(h.data(), dptr, nbytes, cudaMemcpyDeviceToHost);
    uint64_t acc = 1469598103934665603ULL;
    for(uint64_t v : h) { acc ^= v; acc *= 1099511628211ULL; }
    printf("[ksum] %-14s %016llx\n", name, (unsigned long long)acc);
}
// COMMUTATIVE checksum (order-independent): sum+xor of bit-patterns. Tells value-non-det from
// order-non-det on the raw triplets (which sit at non-deterministic slots).
static void _dbg_ksum_comm(const char* name, const void* dptr, size_t nbytes)
{
    if(!getenv("STIFF_KSUM") || !dptr || nbytes == 0) return;
    std::vector<uint64_t> h((nbytes + 7) / 8, 0);
    cudaMemcpy(h.data(), dptr, nbytes, cudaMemcpyDeviceToHost);
    uint64_t s = 0, x = 0;
    for(uint64_t v : h) { s += v; x ^= v; }
    printf("[ksum] %-14s sum=%016llx xor=%016llx\n", name, (unsigned long long)s, (unsigned long long)x);
}

// [decouple probe] env0-masked COMMUTATIVE (order-free) hash of the RAW Hessian triplet VALUES.
// Masks triplets whose row block is a FEM vertex in env0 (p2g[row]==0). Order-free integer
// bit-sum ⇒ batch-dependent triplet ORDER is irrelevant; only the env0 VALUE multiset + COUNT
// matter. Compare A-vs-B: ntrip differs⇒contact-pair SET differs; count same+hash differs⇒a
// per-pair value differs; both same⇒merge/preconditioner is the seed (values are bit-exact).
static void _dbg_hess_env0(const void* vals9, const int* rows, const int* cols, long ntrip, const int* p2g_dev, int vN,
                           long b_end = -1, long f_end = -1, long g_end = -1)
{
    std::vector<double> hv((size_t)ntrip * 9);
    std::vector<int>    hr(ntrip), hc(ntrip), hp(vN);
    cudaMemcpy(hv.data(), vals9, (size_t)ntrip * 9 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hr.data(), rows, (size_t)ntrip * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hc.data(), cols, (size_t)ntrip * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hp.data(), p2g_dev, (size_t)vN * sizeof(int), cudaMemcpyDeviceToHost);
    uint64_t src_sum[4] = {0,0,0,0}; long src_cnt[4] = {0,0,0,0};  // 0=BARRIER 1=FRICTION 2=GROUND 3=FEM
    // split: INTRA = row∈env0 AND col∈env0 ; CROSS = row∈env0 AND col∉env0 (cross-env leak)
    uint64_t si = 0, xi = 0, sc = 0, xc = 0; long ci = 0, cc = 0;
    long col_neg1 = 0, col_oob = 0, col_mate = 0;  // CROSS col classification
    long mate_min = (long)ntrip + 1, mate_max = -1;  // triplet-index range of cross-env-mate triplets
    for(long t = 0; t < ntrip; ++t)
    {
        int r = hr[t], c = hc[t];
        bool re = (r >= 0 && r < vN && hp[r] == 0);
        if(!re) continue;
        bool ce = (c >= 0 && c < vN && hp[c] == 0);
        uint64_t hh = 0, hx = 0;
        for(int e = 0; e < 9; ++e) { uint64_t b; memcpy(&b, &hv[(size_t)t * 9 + e], 8); hh += b; hx ^= b; }
        if(ce) { ++ci; si += hh; xi ^= hx;
                 if(b_end >= 0) { int s = (t < b_end) ? 0 : (t < f_end) ? 1 : (t < g_end) ? 2 : 3;
                                  src_sum[s] += hh; ++src_cnt[s]; } }
        else
        {
            ++cc; sc += hh; xc ^= hx;
            if(c < 0 || c >= vN) ++col_oob;        // col out-of-range (ABD body dof? ground?)
            else if(hp[c] < 0) ++col_neg1;          // col ungrouped (-1): env0 gripper/static
            else { ++col_mate;                       // col in a DIFFERENT env (1/2/3): TRUE cross-env
                   if(t < mate_min) mate_min = t; if(t > mate_max) mate_max = t; }
        }
    }
    printf("[ksum-env0] INTRA: ntrip=%ld sum=%016llx | CROSS: ntrip=%ld | by-src BARRIER(n=%ld,s=%016llx) FRICTION(n=%ld,s=%016llx) GROUND(n=%ld,s=%016llx) FEM(n=%ld,s=%016llx)\n",
           ci, (unsigned long long)si, cc,
           src_cnt[0], (unsigned long long)src_sum[0], src_cnt[1], (unsigned long long)src_sum[1],
           src_cnt[2], (unsigned long long)src_sum[2], src_cnt[3], (unsigned long long)src_sum[3]);
}

int GIPC::calculateMovingDirection(device_TetraData& TetMesh, int cpNum, int preconditioner_type)
{
    gipc::Timer timer{"solve_linear_system"};
    auto        iter = 0;

    if(getenv("STIFF_HESS_ENV0") && TetMesh.d_point_to_group
       && g_dec_frame == (getenv("STIFF_DUMP_FRAME") ? atoi(getenv("STIFF_DUMP_FRAME")) : -1)
       && g_dec_k == (getenv("STIFF_PROBE_K") ? atoi(getenv("STIFF_PROBE_K")) : 0))
    {
        cudaDeviceSynchronize();
        { extern unsigned long long get_xskip(); printf("[xskip] cross-env pairs skipped so far = %llu\n", get_xskip());
          long Bb = (long)h_cpNum[4]*M12_Off + (long)h_cpNum[3]*M9_Off + (long)h_cpNum[2]*M6_Off;
          long Ff = (long)h_cpNum_last[4]*M12_Off + (long)h_cpNum_last[3]*M9_Off + (long)h_cpNum_last[2]*M6_Off + (long)h_gpNum_last;
          printf("[tri-bounds] BARRIER=[0,%ld) FRICTION=[%ld,%ld) GROUND=[%ld,%ld) FEM=[%ld,end) total=%lld\n",
                 Bb, Bb, Bb+Ff, Bb+Ff, Bb+Ff+(long)h_gpNum, Bb+Ff+(long)h_gpNum, (long long)gipc_global_triplet.global_triplet_offset); }
        // [corrected] mask by d_dof_to_group (the BLOCK index space the triplet row/col live in:
        // ABD blocks first, then FEM vertex blocks), NOT d_point_to_group (per-vertex, no ABD prefix).
        { long Bb = (long)h_cpNum[4]*M12_Off + (long)h_cpNum[3]*M9_Off + (long)h_cpNum[2]*M6_Off;
          long Ff = (long)h_cpNum_last[4]*M12_Off + (long)h_cpNum_last[3]*M9_Off + (long)h_cpNum_last[2]*M6_Off + (long)h_gpNum_last;
          _dbg_hess_env0(gipc_global_triplet.block_values(), gipc_global_triplet.block_row_indices(),
                       gipc_global_triplet.block_col_indices(),
                       (long)gipc_global_triplet.global_triplet_offset, TetMesh.d_dof_to_group, TetMesh.dof_block_count,
                       Bb, Bb+Ff, Bb+Ff+(long)h_gpNum); }
    }

    if(getenv("STIFF_KSUM"))
    {
        cudaDeviceSynchronize();
        _dbg_ksum("fb_in",      TetMesh.fb,          (size_t)vertexNum * sizeof(double3));
        _dbg_ksum("shapegrad_in", TetMesh.shape_grads, (size_t)vertexNum * sizeof(double3));
        // COMMUTATIVE hash of the RAW triplets (block values + row/col), order-independent →
        // tells if the Hessian VALUE-multiset is deterministic (vs just non-det slot order).
        _dbg_ksum_comm("rawHess_comm", gipc_global_triplet.block_values(),
                       (size_t)gipc_global_triplet.global_triplet_offset * 9 * sizeof(double));
        _dbg_ksum_comm("rawRow_comm", gipc_global_triplet.block_row_indices(),
                       (size_t)gipc_global_triplet.global_triplet_offset * sizeof(int));
        printf("[ksum] cpNum=%d Kappa=%.17g toff=%lld\n",
               h_cpNum[0], Kappa, (long long)gipc_global_triplet.global_triplet_offset);
    }

    // [xenv] gradient RHS asymmetry — is fb already env-asymmetric BEFORE the solve?
    // (fb is assembled by computeGradientAndHessian from identical co-located verts.)
    if(getenv("STIFF_XENV") && m_d_p2g) xenvDiff(TetMesh.fb, "fb(grad) pre-solve");

    iter = m_global_linear_system->solve_linear_system();

    // [xenv] search-direction asymmetry — did the SOLVE (SpMV/dot/precond) introduce it?
    if(getenv("STIFF_XENV") && m_d_p2g) xenvDiff(_moveDir, "moveDir post-solve");

    if(getenv("STIFF_KSUM"))
    {
        cudaDeviceSynchronize();
        // merged matrix (converter output, what the spmv used) + the solve result
        _dbg_ksum("matrix_merged", gipc_global_triplet.block_values(),
                  (size_t)gipc_global_triplet.h_unique_key_number * 9 * sizeof(double));
        printf("[ksum] nuniq_merged=%d\n", gipc_global_triplet.h_unique_key_number);
        _dbg_ksum("moveDir_out", _moveDir, (size_t)vertexNum * sizeof(double3));
    }


    auto& json = gipc::Statistics::instance().at_current_frame();
    json["newton"].back()["pcg"]["iterations"] = iter;
    return iter;
}


bool edgeTriIntersectionQuery(const int*     _bodyId,
                              const int*     _btype,
                              const double3* _vertexes,
                              const uint2*   _edges,
                              const uint3*   _faces,
                              const AABB*    _edge_bvs,
                              const Node*    _edge_nodes,
                              double         dHat,
                              int            number,
                              const int*     _collision_skip_matrix,
                              int            _collision_body_count,
                              const int*     _body_id_to_is_fem)
{
    int numbers = number;
    if(numbers <= 0)
        return false;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    int*               _isIntersect;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_isIntersect, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(_isIntersect, 0, sizeof(int)));

    _edgeTriIntersectionQuery<<<blockNum, threadNum>>>(
        _bodyId, _btype, _vertexes, _edges, _faces, _edge_bvs, _edge_nodes, _isIntersect, dHat, numbers,
        _collision_skip_matrix, _collision_body_count, _body_id_to_is_fem);

    int h_isITST;
    cudaMemcpy(&h_isITST, _isIntersect, sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_SAFE_CALL(cudaFree(_isIntersect));
    if(h_isITST < 0)
    {
        return true;
    }
    return false;
}

bool GIPC::checkEdgeTriIntersectionIfAny(device_TetraData& TetMesh)
{
    return edgeTriIntersectionQuery(bvh_e._bodyId,
                                    bvh_e._btype,
                                    TetMesh.vertexes,
                                    bvh_e._edges,
                                    bvh_f._faces,
                                    bvh_e._bvs,
                                    bvh_e._nodes,
                                    dHat,
                                    bvh_f.face_number,
                                    bvh_e._collision_skip_matrix,
                                    bvh_e._collision_body_count,
                                    _body_id_to_is_fem);
}

bool GIPC::checkGroundIntersection()
{
    int numbers = h_gpNum;
    if(numbers <= 0)
        return false;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //

    int* _isIntersect;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_isIntersect, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(_isIntersect, 0, sizeof(int)));
    _checkGroundIntersection<<<blockNum, threadNum>>>(
        _vertexes, _groundOffset, _groundNormal, _environment_collisionPair, _isIntersect, numbers);

    int h_isITST;
    cudaMemcpy(&h_isITST, _isIntersect, sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_SAFE_CALL(cudaFree(_isIntersect));
    if(h_isITST < 0)
    {
        return true;
    }
    return false;
}

bool GIPC::isIntersected(device_TetraData& TetMesh)
{
    if(m_skip_all_collision)
        return false;

    // CCD line-search already constrains alpha to a non-intersecting step.
    // The line-search-tail isIntersected() check is a paranoid second pass
    // that re-runs _edgeTriIntersectionQuery (42% of GPU time in case39)
    // for every line-search alpha bisection.  On smooth-contact scenes it
    // never fires (verified across 9/9 paired runs on case39).
    //
    // Default: SKIP the recheck (was opt-in via STIFF_SKIP_CCD_SANITY=1 in
    // dc11e10).  Set GIPC_FORCE_CCD_SANITY=1 to restore the v0.6-and-earlier
    // behavior of running the check.  STIFF_SKIP_CCD_SANITY=0 also restored
    // (back-compat); any other value or unset = skip.
    static const bool keep_sanity = []{
        const char* v_force = std::getenv("GIPC_FORCE_CCD_SANITY");
        if(v_force && v_force[0] && v_force[0] != '0') return true;
        const char* v_skip = std::getenv("STIFF_SKIP_CCD_SANITY");
        if(v_skip && v_skip[0] == '0') return true;  // explicit opt-out of skip
        return false;
    }();
    if(!keep_sanity) return false;

    if(checkGroundIntersection())
    {
        return true;
    }

    if(checkEdgeTriIntersectionIfAny(TetMesh))
    {
        std::cout << "is edge triangle\n";
        return true;
    }
    return false;
}



// [phase-time] lineSearch inner split (per frame): energy evals vs buildBVH+intersect vs buildCP vs step.
static double g_ls_e_ms = 0.0, g_ls_bvh_ms = 0.0, g_ls_cp_ms = 0.0, g_ls_step_ms = 0.0;
// gated event-pair stopwatch (STIFF_PHASE_TIME only; events are stream-ordered, no extra syncs —
// callers place it around ops that already end host-synchronous).
struct _LsTimer {
    bool on; cudaEvent_t a, b; double* acc;
    _LsTimer(double* accum) : on(getenv("STIFF_PHASE_TIME") != nullptr), acc(accum)
    { if(on){ cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a); } }
    void stop()
    { if(on){ cudaEventRecord(b); cudaEventSynchronize(b); float m=0; cudaEventElapsedTime(&m,a,b);
              *acc += m; cudaEventDestroy(a); cudaEventDestroy(b); on=false; } }
};

bool GIPC::lineSearch(device_TetraData& TetMesh, double& alpha, const double& cfl_alpha)
{
    muda::wait_device();
    bool   stopped       = false;
    // NOTE(perf, rejected): a "lazy" variant skipped this global entry energy on the per-env (S3)
    // path (it is only read by the rare uniform FALLBACK) and used full_sum(Eg0) there instead —
    // mathematically the same quantity (per-env decomposition is a partition; S3-validated equal to
    // machine precision) but its atomicAdd summation is bit-wobbly run-to-run and the fallback
    // compare has no tolerance band (unlike S3's 1e-12 band) → a theoretical knife-edge risk to
    // strict bit-identity for ~1ms/iter (ls-inner: energy is ~6% of lineSearch; buildCP is 94%).
    // Not worth it: keep the eager entry energy = bit-exact original semantics on ALL paths.
    double lastEnergyVal = computeEnergy(TetMesh);
    bool perenv_try = (m_env_alpha_valid && m_env_alpha && TetMesh.d_point_to_group
                       && abd_fem_count_info.fem_point_num > 0
                       && getenv("STIFF_PERENV_ALPHA"));

    // [multi-env S3] validate per-env energy decomposition (Sum_g E_g(FEM) ==
    // global FEM). Read-only; gated. Run a couple times then it's confirmed.
    if(getenv("STIFF_S3_VALIDATE") && TetMesh.d_point_to_group)
    {
        std::vector<double> eg;
        computeEnergy_perenv(TetMesh, eg);  // prints [S3-energy] when STIFF_PENV_STATS
    }

    double c1m         = 0.0;
    double armijoParam = 0;
    if(armijoParam > 0.0)
    {
        c1m += armijoParam * Energy_Add_Reduction_Algorithm(3, TetMesh);
    }

    CUDA_SAFE_CALL(cudaMemcpy(TetMesh.temp_double3Mem,
                              TetMesh.vertexes,
                              vertexNum * sizeof(double3),
                              cudaMemcpyDeviceToDevice));

    m_abd_system->copy_q_to_q_temp(*m_abd_sim_data);


    double alpha_SL = alpha;

    // [multi-env S2/S3] RIGOROUS per-env line search. Step each env by its own
    // CCD-feasible alpha (m_env_alpha, S1-validated safe), then enforce PER-ENV
    // energy DESCENT: any env whose own energy E_g rose halves its alpha_g and
    // re-steps (independent per-env backtracking) — NOT the global-energy heuristic
    // (which can mask one env's increase). Per-env energy is validated exact
    // (Sum_g E_g == global, machine precision). Accept when every env satisfies
    // E_g(alpha_g) <= E_g(0) with no intersection. Falls back to the standard
    // uniform search only if an env can't descend within maxBT halvings. Gated.
    if(perenv_try)   // (perenv_try hoisted to the top — see the lazy lastEnergyVal comment)
    {
        const int NG = kEnvAlphaSlots;
        const int abdN = (int)abd_fem_count_info.abd_body_num;
        const int maxBT = 8;
        // [de-CPU S3] energies + decision fully DEVICE-resident: computeEnergy_perenv_dev fills
        // d_Eg0/d_Eg1 (no 22KB D2H), _s3_decide halves m_env_alpha IN PLACE (the true state; the
        // old host loop operated on the stale h_env_alpha mirror). Host reads ONE int per round —
        // required: it decides whether to re-run the step/rebuild/energy round (host loop control).
        static double* d_Eg0 = nullptr; static double* d_Eg1 = nullptr; static int* d_nfail = nullptr;
        if(!d_Eg0)
        {
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_Eg0, NG * sizeof(double)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_Eg1, NG * sizeof(double)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_nfail, sizeof(int)));
        }
        _LsTimer _t0(&g_ls_e_ms);
        computeEnergy_perenv_dev(TetMesh, d_Eg0);   // per-env energy at START (temp) config
        _t0.stop();
        bool accepted = false;
        for(int bt = 0; bt <= maxBT; ++bt)
        {
            if(abdN > 0 && TetMesh.d_body_to_group)   // refresh per-body ABD alpha
            {
                if(!m_abd_body_alpha)
                    CUDA_SAFE_CALL(cudaMalloc((void**)&m_abd_body_alpha, abdN * sizeof(double)));
                int tn = 256, bn = (abdN + tn - 1) / tn;
                _gather_abd_body_alpha<<<bn, tn>>>(TetMesh.d_body_to_group, m_env_alpha,
                                                   m_abd_body_alpha, abdN, NG);
            }
            _LsTimer _ts(&g_ls_step_ms);
            m_perenv_apply = true;
            step_forward(TetMesh, alpha, false);   // per-env step from temp
            m_perenv_apply = false;
            _ts.stop();
            _LsTimer _tb(&g_ls_bvh_ms);
            buildBVH();
            bool _isect = isIntersected(TetMesh);
            _tb.stop();
            if(_isect)   // CCD-safe alpha should prevent this; safety net
            {
                _s3_halve_all<<<(NG + 255) / 256, 256>>>(m_env_alpha, NG);
                continue;
            }
            _LsTimer _tc(&g_ls_cp_ms);
            buildCP();
            _tc.stop();
            // [backport] buildCP() already D2H-syncs h_cpNum/h_gpNum and the energy
            // kernels consume unsorted pairs directly, so the perf branch's
            // sync_cpNum()/sort_collision_pairs_by_type() are unneeded in this path.
            _LsTimer _t1(&g_ls_e_ms);
            computeEnergy_perenv_dev(TetMesh, d_Eg1);   // per-env energy AFTER step
            _t1.stop();
            CUDA_SAFE_CALL(cudaMemsetAsync(d_nfail, 0, sizeof(int)));
            _s3_decide<<<(NG + 255) / 256, 256>>>(d_Eg0, d_Eg1, m_env_alpha, d_nfail, NG);
            int nfail = 0;
            CUDA_SAFE_CALL(cudaMemcpy(&nfail, d_nfail, sizeof(int), cudaMemcpyDeviceToHost));
            if(nfail == 0)
            {
                accepted = true;
                if(getenv("STIFF_PENV_STATS")) printf("[S3-accept] per-env descent ok (bt=%d)\n", bt);
                break;
            }
            // (m_env_alpha already halved in place on device — no H2D)
            if(getenv("STIFF_PENV_STATS")) printf("[S3-bt] bt=%d halved %d envs\n", bt, nfail);
        }
        if(accepted) return false;   // accepted; collision state already rebuilt
        if(getenv("STIFF_PENV_STATS")) printf("[S3-fallback] per-env descent not reached -> global\n");
        // fall through to standard uniform-alpha line search (re-steps from temp); the Armijo
        // reference lastEnergyVal is the eager entry computeEnergy (bit-exact original semantics).
    }

    step_forward(TetMesh, alpha, false);

    bool rehash = true;

    buildBVH();

    int numOfIntersect = 0;
    int insectNum      = 0;

    bool checkInterset = true;

    while(checkInterset && isIntersected(TetMesh))
    {
        printf("type 0 intersection happened 0:  %d\n", insectNum);
        insectNum++;
        alpha /= 2.0;
        numOfIntersect++;
        alpha = std::min(cfl_alpha, alpha);
        step_forward(TetMesh, alpha, false);
        buildBVH();
        //break;
    }

    buildCP();

    double testingE = computeEnergy(TetMesh);

    int    numOfLineSearch = 0;
    double LFStepSize      = alpha;

    std::cout.precision(18);
    constexpr int report_line_search_threshold = 8;

    while((testingE > lastEnergyVal + c1m * alpha) && numOfLineSearch <= report_line_search_threshold)
    {
        //std::cout << "[" << numOfLineSearch << "]   testE:    " << testingE
        //          << "      lastEnergyVal:        " << lastEnergyVal << std::endl;
        alpha /= 2.0;
        ++numOfLineSearch;

        step_forward(TetMesh, alpha, false);
        buildBVH();
        buildCP();
        testingE = computeEnergy(TetMesh);
    }
    if(numOfLineSearch > report_line_search_threshold)
        if(g_gipc_log_level >= 1) printf("!!!!!!!!!!!!!!!!!!!linesearch number is a bit high, lineSearchCount=%d !!!!!!!!!!!!!!!!!!!!!!\n",
               numOfLineSearch);


    if(alpha < LFStepSize)
    {
        bool needRecomputeCS = false;
        while(checkInterset && isIntersected(TetMesh))
        {
            printf("type 1 intersection happened 1:  %d\n", insectNum);
            insectNum++;
            alpha /= 2.0;
            numOfIntersect++;
            alpha = std::min(cfl_alpha, alpha);

            step_forward(TetMesh, alpha, false);
            buildBVH();
            needRecomputeCS = true;
        }
        if(needRecomputeCS)
        {
            buildCP();
        }
    }

    return stopped;
}


void GIPC::postLineSearch(device_TetraData& TetMesh, double alpha)
{
    if(Kappa == 0.0)
    {
        initKappa(TetMesh);
    }
    else
    {
        if(m_pergroup_kappa && m_kappa_group && m_d_close_grp)
        {
            // [multi-env per-group κ] run BOTH close-val checks (no short-circuit) to populate the
            // per-group flags, then double ONLY the groups that hit a close contact. This decouples
            // the cross-env bifurcation (a global κ doubling was the dominant cross-env coupling).
            int NG = m_perenv_bvh_groups > 0 ? m_perenv_bvh_groups : kEnvAlphaSlots;
            CUDA_SAFE_CALL(cudaMemset(m_d_close_grp, 0, NG * sizeof(int)));
            (void)checkCloseGroundVal();   // populates m_d_close_grp (global bool ignored)
            (void)checkSelfCloseVal();
            // [perf/device-residence] double the close groups' κ ON DEVICE (in m_kappa_group), taking the
            // envelope via atomicMax — replaces the per-Newton D2H(close flags) + 256-env host loop +
            // H2D(κ). kappaMax is a host scalar (env-independent), the doubling+cap+max are all order-free
            // → strict bit-identical. h_kappa_group is NOT touched here (it is re-seeded by initKappa each
            // frame and read nowhere else during the frame; device m_kappa_group is the in-frame truth).
            double kappaMax = 1e300;
            upperBoundKappa(kappaMax);     // kappaMax = env-independent cap (was recomputed per group)
            static double* d_maxK = nullptr;
            if(!d_maxK) CUDA_SAFE_CALL(cudaMalloc(&d_maxK, sizeof(double)));
            CUDA_SAFE_CALL(cudaMemcpy(d_maxK, &Kappa, sizeof(double), cudaMemcpyHostToDevice));  // envelope init
            {
                int bs = 256;
                _per_group_kappa_double<<<(NG + bs - 1) / bs, bs>>>(m_kappa_group, m_d_close_grp, NG, kappaMax, d_maxK);
            }
            CUDA_SAFE_CALL(cudaMemcpy(&Kappa, d_maxK, sizeof(double), cudaMemcpyDeviceToHost));  // scalar envelope
            tempFree_closeConstraint();
            tempMalloc_closeConstraint();
            CUDA_SAFE_CALL(cudaMemset(_close_cpNum, 0, sizeof(uint32_t)));
            CUDA_SAFE_CALL(cudaMemset(_close_gpNum, 0, sizeof(uint32_t)));
            computeCloseGroundVal();
            computeSelfCloseVal();
            return;
        }

        bool updateKappa = checkCloseGroundVal();
        if(!updateKappa)
        {
            updateKappa = checkSelfCloseVal();
        }
        if(updateKappa)
        {
            Kappa *= 2.0;
            upperBoundKappa(Kappa);
        }
        tempFree_closeConstraint();
        tempMalloc_closeConstraint();
        CUDA_SAFE_CALL(cudaMemset(_close_cpNum, 0, sizeof(uint32_t)));
        CUDA_SAFE_CALL(cudaMemset(_close_gpNum, 0, sizeof(uint32_t)));

        computeCloseGroundVal();

        computeSelfCloseVal();
    }
}

void GIPC::tempMalloc_closeConstraint()
{
    // [0be8da3-port, grow-only] buffers persist across sub-iterations/frames;
    // (re)allocate only on growth. Kernels touch [0, h_gpNum)/[0, h_cpNum[0]).
    if((size_t)h_gpNum > m_close_gp_cap)
    {
        if(_closeConstraintID)
        {
            CUDA_SAFE_CALL(cudaFree(_closeConstraintID));
            CUDA_SAFE_CALL(cudaFree(_closeConstraintVal));
        }
        size_t n = (size_t)h_gpNum + h_gpNum / 4;
        CUDA_SAFE_CALL(cudaMalloc((void**)&_closeConstraintID, n * sizeof(uint32_t)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&_closeConstraintVal, n * sizeof(double)));
        m_close_gp_cap = n;
    }
    if((size_t)h_cpNum[0] > m_close_cp_cap)
    {
        if(_closeMConstraintID)
        {
            CUDA_SAFE_CALL(cudaFree(_closeMConstraintID));
            CUDA_SAFE_CALL(cudaFree(_closeMConstraintVal));
        }
        size_t n = (size_t)h_cpNum[0] + h_cpNum[0] / 4;
        CUDA_SAFE_CALL(cudaMalloc((void**)&_closeMConstraintID, n * sizeof(int4)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&_closeMConstraintVal, n * sizeof(double)));
        m_close_cp_cap = n;
    }
}

void GIPC::tempFree_closeConstraint()
{
    // [0be8da3-port] no-op — buffers persist; freed in FREE_DEVICE_MEM.
}

void GIPC::ensure_frictionBuffers()
{
    // [0be8da3-port, grow-only] replaces the per-frame/per-sub-iter free+malloc
    // of the 7 friction lastH buffers. distCoord's live range [0, h_cpNum[0])
    // is re-zeroed on EVERY call (a fresh cudaMalloc'd buffer was memset the
    // same way), keeping the [4.3] frame-0 lag fix value-identical.
    if((size_t)h_cpNum[0] > m_fric_cp_cap)
    {
        if(lambda_lastH_scalar)
        {
            CUDA_SAFE_CALL(cudaFree(lambda_lastH_scalar));
            CUDA_SAFE_CALL(cudaFree(distCoord));
            CUDA_SAFE_CALL(cudaFree(tanBasis));
            CUDA_SAFE_CALL(cudaFree(_collisonPairs_lastH));
            CUDA_SAFE_CALL(cudaFree(_MatIndex_last));
        }
        size_t n = (size_t)h_cpNum[0] + h_cpNum[0] / 4;
        CUDA_SAFE_CALL(cudaMalloc((void**)&lambda_lastH_scalar, n * sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&distCoord, n * sizeof(double2)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&tanBasis, n * sizeof(__GEIGEN__::Matrix3x2d)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&_collisonPairs_lastH, n * sizeof(int4)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&_MatIndex_last, n * sizeof(int)));
        m_fric_cp_cap = n;
    }
    if((size_t)h_gpNum > m_fric_gd_cap)
    {
        if(lambda_lastH_scalar_gd)
        {
            CUDA_SAFE_CALL(cudaFree(lambda_lastH_scalar_gd));
            CUDA_SAFE_CALL(cudaFree(_collisonPairs_lastH_gd));
        }
        size_t n = (size_t)h_gpNum + h_gpNum / 4;
        CUDA_SAFE_CALL(cudaMalloc((void**)&lambda_lastH_scalar_gd, n * sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&_collisonPairs_lastH_gd, n * sizeof(uint32_t)));
        m_fric_gd_cap = n;
    }
    if(h_cpNum[0])
        CUDA_SAFE_CALL(cudaMemset(distCoord, 0, h_cpNum[0] * sizeof(double2)));  // [4.3] frame-0 lag uninit
}
double maxCOllisionPairNum = 0;
double totalCollisionPairs = 0;
double total_Cg_count      = 0;
double timemakePd          = 0;
#include <vector>
#include <fstream>
std::vector<int> iterV;

// [phase-time] time3 sub-split (per frame): S1 per-env-alpha block vs lineSearch proper.
static double g_t3_s1_ms = 0.0, g_t3_ls_ms = 0.0;

int              GIPC::solve_subIP(device_TetraData& TetMesh,
                      double&           time0,
                      double&           time1,
                      double&           time2,
                      double&           time3,
                      double&           time4)
{
    auto& stats_at_current_frame = gipc::Statistics::instance().at_current_frame();
    if(g_gipc_log_level >= 1)
        std::cout << "solve_subIP >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
                  << std::endl;

    stats_at_current_frame["newton"] = gipc::Json::array();
    g_t3_s1_ms = 0.0; g_t3_ls_ms = 0.0;   // [phase-time] time3 sub-split, per frame
    g_ls_e_ms = 0.0; g_ls_bvh_ms = 0.0; g_ls_cp_ms = 0.0; g_ls_step_ms = 0.0;
    // [env-det] capture p2g at entry so the canon env-local-id tiebreak (g_vloc) is built BEFORE the
    // first buildCP of this frame (buildCP runs before computeGradientAndHessian sets m_d_p2g).
    if(getenv("STIFF_EE_CANON") && TetMesh.d_point_to_group) m_d_p2g = TetMesh.d_point_to_group;

    int iterCap = newton_iter_cap, k = 0;
    double semi_beta = 1.0;

    CUDA_SAFE_CALL(cudaMemset(_moveDir, 0, vertexNum * sizeof(double3)));
    double totalTimeStep = 0;

    // [multi-env S4] inject per-env mask into the linear system: solve_linear_system
    // zeros masked envs' RHS, spmv skips their triplets. Reset all-active at frame
    // start (detection updates m_env_active each iter); cleared after the loop.
    const bool s4_mask_on = (m_env_active && TetMesh.d_dof_to_group
                             && TetMesh.d_point_to_group
                             && (getenv("STIFF_PERENV_MASK") || getenv("STIFF_PERENV_MASK_DEV")));
    // [S4-dev] device-derived mask (from m_env_alpha, zero D2H). Requires the per-env alpha
    // machinery (m_env_alpha filled by the S1 line-search block each iter).
    const bool s4_dev_mask = s4_mask_on && m_env_alpha && getenv("STIFF_PERENV_MASK_DEV")
                             && getenv("STIFF_PERENV_ALPHA");
    if(s4_mask_on)
    {
        std::fill(h_env_active.begin(), h_env_active.end(), 1);
        CUDA_SAFE_CALL(cudaMemcpy(m_env_active, h_env_active.data(),
                                  kEnvAlphaSlots * sizeof(int), cudaMemcpyHostToDevice));
        m_global_linear_system->set_env_mask(m_env_active, TetMesh.d_dof_to_group, kEnvAlphaSlots);
    }
    // [multi-env P3] register the DOF→group map for the SEGMENTED block-diagonal PCG even when
    // masking is off (the PCG reads m_s4_dof_to_group/m_s4_ng). active=nullptr ⇒ no RHS masking.
    else if(getenv("STIFF_SEGMENTED_PCG") && TetMesh.d_dof_to_group && TetMesh.d_point_to_group)
        m_global_linear_system->set_env_mask(nullptr, TetMesh.d_dof_to_group, kEnvAlphaSlots);
    else
        m_global_linear_system->set_env_mask(nullptr, nullptr, 0);

    int& s_dec_frame = g_dec_frame;   // [decouple probe] per-solve_subIP-call counter (~frame)
    s_dec_frame++;

    // [decouple] per-env convergence latch for the loop-exit override. The global gradVanish can
    // fire while a per-env env is still UNDER-converged (its mates converged fast → loop ends → that
    // env is cut off at a batch-dependent iter → drift). When DECOUPLE_THRESH, the loop exits only
    // once ALL present envs are per-env frozen (each reached ITS OWN convergence), so an env's final
    // state is independent of the mates / loop length. Updated at the end of S1 Phase B each iter.
    bool all_env_frozen = false;

    for(; k < iterCap; ++k)
    {
        if(g_gipc_log_level >= 1 && k > 0 && k % 10 == 0)
            printf("  Newton iter %d ...\n", k);
        stats_at_current_frame["newton"].push_back(gipc::Json::object());

        // [S4-dev] periodic all-active recheck (bounce-back detection): every RECHECK iters, unmask
        // ALL envs so masked (frozen) envs get one REAL solve — if a κ doubling / friction update
        // moved a frozen env off its optimum, its hmx exceeds thr and _per_env_alpha_compute
        // un-freezes it (mask follows at end of this iter). Placed BEFORE the solve so the recheck
        // iter solves the full system. Fixed cadence → deterministic.
        if(s4_dev_mask && (k % 4 == 0))
            _mask_fill<<<(kEnvAlphaSlots + 255) / 256, 256>>>(m_env_active, 1, kEnvAlphaSlots);

        totalCollisionPairs += h_cpNum[0];
        maxCOllisionPairNum =
            (maxCOllisionPairNum > h_cpNum[0]) ? maxCOllisionPairNum : h_cpNum[0];
        cudaEvent_t start, end0, end1, end2, end3, end4, e2b;
        cudaEventCreate(&start);
        cudaEventCreate(&end0);
        cudaEventCreate(&end1);
        cudaEventCreate(&end2);
        cudaEventCreate(&end3);
        cudaEventCreate(&end4);
        cudaEventCreate(&e2b);

        //printf("\n\n\ncollision num  %d\n\n\n", h_cpNum[0]+h_gpNum);

        cudaEventRecord(start);
        g_dec_k = (int)k;   // [decouple probe] expose k to computeGradientAndHessian's stage dumps
        timemakePd += computeGradientAndHessian(TetMesh);

        // [decouple probe] PRE-SOLVE gradient dump (shape_grads + fb hold the CLEAN gradient here,
        // before calculateMovingDirection clobbers shape_grads as scratch). frame STIFF_DUMP_FRAME,
        // any k if STIFF_PROBE_K unset → use k==0. Python compares env0 across batches.
        if(getenv("STIFF_GRAD_PRE") && TetMesh.d_point_to_group
           && s_dec_frame == atoi(getenv("STIFF_DUMP_FRAME"))
           && (int)k == (getenv("STIFF_PROBE_K") ? atoi(getenv("STIFF_PROBE_K")) : 0))
        {
            std::vector<double3> hsh(vertexNum), hfb(vertexNum);
            std::vector<int>     hp(vertexNum);
            CUDA_SAFE_CALL(cudaMemcpy(hsh.data(), TetMesh.shape_grads, vertexNum*sizeof(double3), cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(hfb.data(), TetMesh.fb, vertexNum*sizeof(double3), cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(hp.data(), TetMesh.d_point_to_group, vertexNum*sizeof(int), cudaMemcpyDeviceToHost));
            const char* fn = getenv("STIFF_GRAD_PRE");
            FILE* a=fopen((std::string(fn)+".shape").c_str(),"wb"); if(a){fwrite(hsh.data(),sizeof(double3),vertexNum,a);fclose(a);}
            FILE* b=fopen((std::string(fn)+".fb").c_str(),"wb");    if(b){fwrite(hfb.data(),sizeof(double3),vertexNum,b);fclose(b);}
            FILE* c=fopen((std::string(fn)+".grp").c_str(),"wb");   if(c){fwrite(hp.data(),sizeof(int),vertexNum,c);fclose(c);}
            printf("[grad-pre] dumped shape+fb+grp @frame %d k=%d\n", s_dec_frame, (int)k);
        }

        double distToOpt_PN = calcMinMovement(_moveDir, pcg_data.squeue, vertexNum);

        // [decouple] the Newton convergence threshold uses the MERGED-scene bboxDiagSize2, which
        // varies with the batch (mates' extents) → env_0's stop point depends on its batch-mates.
        // STIFF_DECOUPLE_THRESH replaces it with the abs_dhat-fixed eff bbox (batch-INVARIANT, same
        // physical contact scale a single-env run would use), removing this coupling.
        double thr_bbox2 = bboxDiagSize2;
        // [convergence consistency] when an ABSOLUTE contact scale is set (absolute_dhat>0), the Newton
        // convergence tolerance MUST follow that physical scale, NOT the merged-scene bboxDiagSize2
        // (which grows with env count / spacing → the tolerance becomes batch-dependent and looser for
        // merged, so merged 'converges' at a coarser residual than a single-env run). Gated on
        // absolute_dhat>0 ALONE (not STIFF_DECOUPLE_THRESH) so ALL modes converge to the SAME physical
        // tolerance. dHat/dTol/fDhat already use this eff bbox; this makes the Newton exit consistent.
        if(absolute_dhat > 0.0 && relative_dhat > 0.0)
            thr_bbox2 = (absolute_dhat * absolute_dhat) / (relative_dhat * relative_dhat);

        // [uipc-style opt-in] newton_velocity_tol>0: physical exit (max step displacement
        // <= v_tol*dt), scene-size/env-count independent, relative_dhat fully inert.
        double _newton_thr = (newton_velocity_tol > 0.0)
                                 ? (newton_velocity_tol * IPC_dt)
                                 : sqrt(Newton_solver_threshold * Newton_solver_threshold
                                        * thr_bbox2 * IPC_dt * IPC_dt);
        bool gradVanish = (distToOpt_PN < _newton_thr);

        // [multi-env P3a step2] per-env Newton convergence tracking (precursor to
        // mask early-exit). The merged Newton loop currently breaks on the GLOBAL
        // move norm = the HARDEST env. Here we measure per-env move RMS each Newton
        // iter so we can see envs converge at DIFFERENT k (the early-exit premise,
        // on real merged-run data). Read-only diagnostic, gated STIFF_PENV_STATS.
        if(TetMesh.d_point_to_group && getenv("STIFF_PENV_STATS"))
        {
            const int NG = 256;
            static double* d_sq = nullptr; static int* d_cnt = nullptr;
            if(!d_sq) { cudaMalloc((void**)&d_sq, NG*sizeof(double));
                        cudaMalloc((void**)&d_cnt, NG*sizeof(int)); }
            cudaMemset(d_sq, 0, NG*sizeof(double)); cudaMemset(d_cnt, 0, NG*sizeof(int));
            int bs = 256, gs = (vertexNum + bs - 1) / bs;
            _per_env_sqnorm_accum<<<gs, bs>>>(TetMesh.d_point_to_group, _moveDir,
                                              d_sq, d_cnt, vertexNum, NG);
            cudaDeviceSynchronize();
            std::vector<double> hs(NG); std::vector<int> hc(NG);
            cudaMemcpy(hs.data(), d_sq, NG*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(hc.data(), d_cnt, NG*sizeof(int), cudaMemcpyDeviceToHost);
            double thr = sqrt(Newton_solver_threshold * Newton_solver_threshold
                              * bboxDiagSize2 * IPC_dt * IPC_dt);
            // [S4 probe] per-env MAX move (the real Newton-exit metric), not RMS.
            static double* d_mxm = nullptr;
            if(!d_mxm) cudaMalloc((void**)&d_mxm, NG*sizeof(double));
            cudaMemset(d_mxm, 0, NG*sizeof(double));
            _per_env_max_move<<<gs, bs>>>(TetMesh.d_point_to_group, _moveDir, d_mxm, vertexNum, NG);
            cudaDeviceSynchronize();
            std::vector<double> hmm(NG);
            cudaMemcpy(hmm.data(), d_mxm, NG*sizeof(double), cudaMemcpyDeviceToHost);
            printf("[P3a-newton] k=%d thr=%.3e per-env maxMove:", k, thr);
            int n_conv = 0, n_present = 0;
            for(int g = 0; g < NG; ++g) if(hc[g] > 0) {
                ++n_present;
                bool conv = (k && hmm[g] < thr);   // matches gradVanish (max move)
                if(conv) ++n_conv;
                printf(" g%d=%.2e%s", g, hmm[g], conv ? "*" : "");
            }
            printf("  (%d/%d done-by-maxmove)\n", n_conv, n_present);
        }

        // [multi-env S4] per-env active-mask DETECTION (foundation; the assembly/
        // PCG/SpMV skips read m_env_active). Mask env once its Newton max-move <
        // thr*margin; every RECHECK iters unmask ALL present envs + re-check
        // (catches non-monotonic bounce-back). Self-contained; gated STIFF_PERENV_MASK.
        // Skip detection on early Newton iters: nothing converges before ~k=MINK
        // (measured), so the per-iter D2H+sync overhead there is pure waste.
        if(m_env_active && TetMesh.d_point_to_group && getenv("STIFF_PERENV_MASK")
           && !s4_dev_mask && k >= 4)   // [S4-dev] device-derived mask supersedes host detection
        {
            const int NG = kEnvAlphaSlots;
            const int RECHECK = 4;
            const double margin = 0.5;
            double thr = ((newton_velocity_tol > 0.0) ? (newton_velocity_tol * IPC_dt) : sqrt(Newton_solver_threshold * Newton_solver_threshold * thr_bbox2 * IPC_dt * IPC_dt));   // [decouple] batch-invariant; velocity_tol opt-in
            static double* d_mm = nullptr; static int* d_ct = nullptr;
            if(!d_mm) { cudaMalloc((void**)&d_mm, NG*sizeof(double));
                        cudaMalloc((void**)&d_ct, NG*sizeof(int)); }
            cudaMemset(d_mm, 0, NG*sizeof(double)); cudaMemset(d_ct, 0, NG*sizeof(int));
            int bs = 256, gs = (vertexNum + bs - 1) / bs;
            _per_env_max_move<<<gs, bs>>>(TetMesh.d_point_to_group, _moveDir, d_mm, vertexNum, NG);
            _per_env_sqnorm_accum<<<gs, bs>>>(TetMesh.d_point_to_group, nullptr, nullptr, d_ct, vertexNum, NG);
            cudaDeviceSynchronize();
            std::vector<double> hmm(NG); std::vector<int> hct(NG);
            cudaMemcpy(hmm.data(), d_mm, NG*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(hct.data(), d_ct, NG*sizeof(int), cudaMemcpyDeviceToHost);
            const bool recheck = (k % RECHECK == 0);
            int n_active = 0, n_present = 0;
            for(int g = 0; g < NG; ++g)
            {
                if(hct[g] <= 0) { h_env_active[g] = 0; continue; }  // absent env
                ++n_present;
                if(recheck || k == 0) h_env_active[g] = 1;          // periodic full re-check
                else if(h_env_active[g] && hmm[g] < thr * margin)   // deeply converged -> mask
                    h_env_active[g] = 0;
                if(h_env_active[g]) ++n_active;
            }
            CUDA_SAFE_CALL(cudaMemcpy(m_env_active, h_env_active.data(),
                                      NG * sizeof(int), cudaMemcpyHostToDevice));
            if(getenv("STIFF_PENV_STATS"))
                printf("[S4-mask] k=%d active=%d/%d%s\n", k, n_active, n_present,
                       recheck ? " (recheck)" : "");
        }

        //double distToOpt_PN = calcMinMovement(TetMesh.totalForce, pcg_data.squeue, vertexNum);
        //printf("disToopt:  %f        %f\n",
        //       distToOpt_PN,
        //       2 * sqrt(Newton_solver_threshold * Newton_solver_threshold * bboxDiagSize2)
        //           * IPC_dt * IPC_dt);

        //bool gradVanish =
        //    (distToOpt_PN < 1
        //                        * sqrt(Newton_solver_threshold * Newton_solver_threshold * bboxDiagSize2)
        //                        * IPC_dt * IPC_dt);

        // [decouple] DECOUPLE_THRESH: exit only when ALL envs are per-env frozen (each converged),
        // NOT on the global gradVanish (which can cut off an under-converged env when its mates
        // finish first → batch-dependent final state). all_env_frozen is from the prev iter's S1
        // Phase B. Baseline (off) keeps the global gradVanish exit.
        // [robustness] the frozen-exit needs the S1 per-env-alpha machinery to actually run
        // (m_env_alpha_valid, set by the prev iter's S1). With DECOUPLE_THRESH but WITHOUT
        // STIFF_PERENV_ALPHA, all_env_frozen stays false forever → the loop ran to iterCap every
        // frame (pathological, found by the flag ablation). Fall back to gradVanish in that case.
        bool do_break = (getenv("STIFF_DECOUPLE_THRESH") && m_env_alpha_valid)
                            ? (k && all_env_frozen)
                            : (k && gradVanish);
        if(do_break)
        {
            break;
        }
        cudaEventRecord(end0);

        auto cg_count = calculateMovingDirection(TetMesh, h_cpNum[0], pcg_data.P_type);
        //std::cout << "[" << k << "]"
        //          << "cg_count = " << cg_count << std::endl;
        total_Cg_count += cg_count;
        // [decouple probe] full-precision moveDir dump (engine order) + d_point_to_group (engine
        // order, aligned). At frame STIFF_DUMP_FRAME, Newton iter STIFF_PROBE_K. Python masks
        // verts[grp==g] and compares matesA vs matesB to find the fine per-step coupling seed.
        if(getenv("STIFF_GRAD_PROBE") && TetMesh.d_point_to_group
           && s_dec_frame == atoi(getenv("STIFF_DUMP_FRAME"))
           && (int)k == (getenv("STIFF_PROBE_K") ? atoi(getenv("STIFF_PROBE_K")) : 1))
        {
            std::vector<double3> hmd(vertexNum), hgr(vertexNum), hsh(vertexNum);
            std::vector<int>     hp(vertexNum);
            CUDA_SAFE_CALL(cudaMemcpy(hmd.data(), _moveDir, vertexNum * sizeof(double3), cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(hgr.data(), TetMesh.fb, vertexNum * sizeof(double3), cudaMemcpyDeviceToHost));  // contact+ground grad
            CUDA_SAFE_CALL(cudaMemcpy(hsh.data(), TetMesh.shape_grads, vertexNum * sizeof(double3), cudaMemcpyDeviceToHost));  // kinetic+elastic
            CUDA_SAFE_CALL(cudaMemcpy(hp.data(), TetMesh.d_point_to_group, vertexNum * sizeof(int), cudaMemcpyDeviceToHost));
            const char* fn = getenv("STIFF_GRAD_PROBE");
            FILE* f = fopen(fn, "wb"); if(f){ fwrite(hmd.data(), sizeof(double3), vertexNum, f); fclose(f); }
            std::string gradn = std::string(fn) + ".grad";
            FILE* fg = fopen(gradn.c_str(), "wb"); if(fg){ fwrite(hgr.data(), sizeof(double3), vertexNum, fg); fclose(fg); }
            std::string shn = std::string(fn) + ".shape";
            FILE* fs = fopen(shn.c_str(), "wb"); if(fs){ fwrite(hsh.data(), sizeof(double3), vertexNum, fs); fclose(fs); }
            std::string gn = std::string(fn) + ".grp";
            FILE* g = fopen(gn.c_str(), "wb"); if(g){ fwrite(hp.data(), sizeof(int), vertexNum, g); fclose(g); }
            printf("[grad-probe] dumped moveDir+grad+grp @frame %d k=%d (%d verts) -> %s\n",
                   s_dec_frame, (int)k, vertexNum, fn);
            // [decouple] ABD body pose dump (verify the gripper/arm pose drifts across batches → the
            // FEM-pin seed). m_d_abd_body_q = Vector12 (12 doubles) per body; d_body_to_group = env id.
            if(getenv("STIFF_ABD_DUMP") && m_abd_sim_data && TetMesh.d_body_to_group)
            {
                int nb = (int)abd_fem_count_info.abd_body_num;
                const double* qptr = reinterpret_cast<const double*>(m_abd_sim_data->device.body_id_to_q.data());
                const double* dqptr = reinterpret_cast<const double*>(m_abd_sim_data->device.body_id_to_dq.data());
                std::vector<double> hq(12 * nb), hdq(12 * nb); std::vector<int> hbg(nb);
                CUDA_SAFE_CALL(cudaMemcpy(hq.data(), qptr, 12 * nb * sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_SAFE_CALL(cudaMemcpy(hdq.data(), dqptr, 12 * nb * sizeof(double), cudaMemcpyDeviceToHost));
                FILE* dq=fopen((std::string(fn)+".abddq").c_str(),"wb"); if(dq){fwrite(hdq.data(),sizeof(double),12*nb,dq);fclose(dq);}
                CUDA_SAFE_CALL(cudaMemcpy(hbg.data(), TetMesh.d_body_to_group, nb * sizeof(int), cudaMemcpyDeviceToHost));
                FILE* q=fopen((std::string(fn)+".abdq").c_str(),"wb"); if(q){fwrite(hq.data(),sizeof(double),12*nb,q);fclose(q);}
                FILE* b=fopen((std::string(fn)+".abdg").c_str(),"wb"); if(b){fwrite(hbg.data(),sizeof(int),nb,b);fclose(b);}
                printf("[abd-dump] %d bodies @frame %d k=%d\n", nb, s_dec_frame, (int)k);
            }
        }
        cudaEventRecord(end1);
        double alpha = 1.0, slackness_a = 0.8, slackness_m = 0.8;

        // ②-D2H: batch the two back-to-back CCD step-size reductions (ground +
        // self) into one D2H of 2 doubles. Preserves original early-return
        // semantics: m_skip_all_collision skips ALL reductions; surf_vertexNum<1
        // skips ground; h_cpNum[0]<1 skips self. Each "did" branch only queues
        // kernels when its preconditions are met.
        if(m_skip_all_collision)
        {
            // both functions short-circuit to 1.0 -> no change to alpha
        }
        else
        {
            bool g_did = (surf_vertexNum >= 1);
            bool s_did = (h_cpNum[0]     >= 1);
            if(g_did) ground_largestFeasibleStepSize_DeviceOut(slackness_a, pcg_data.squeue, m_alpha_slots + 0);
            // self reduces over PAIR count — squeue is mesh-sized (v0.6.3 OOB fix):
            // must use the pair-capacity scratch, NOT pcg_data.squeue.
            if(s_did) self_largestFeasibleStepSize_DeviceOut  (slackness_m, ensure_reduce_scratch(h_cpNum[0]), h_cpNum[0], m_alpha_slots + 1);
            if(g_did || s_did)
            {
                double h_alpha[2] = {1.0, 1.0};
                CUDA_SAFE_CALL(cudaMemcpy(h_alpha, m_alpha_slots, 2 * sizeof(double), cudaMemcpyDeviceToHost));
                if(g_did) alpha = std::min(alpha, 1.0 / h_alpha[0]);
                if(s_did) alpha = std::min(alpha, 1.0 / h_alpha[1]);
            }
        }
        //alpha = std::min(alpha, InjectiveStepSize(0.2, 1e-6, pcg_data.squeue, TetMesh.tetrahedras));
        double temp_alpha = alpha;
        double alpha_CFL  = alpha;

        double ccd_size = 1.0;
        //#ifdef USE_FRICTION
        //        ccd_size = 0.6;
        //#endif

        // [multi-env S1 Phase A] per-env temp_alpha terms (ground + narrow-self),
        // computed HERE because buildFullCP below overwrites _ccd_collisonPairs.
        // Mirrors the engine's temp_alpha reductions (lines above) but per-env.
        // Regions: m_env_scratch[0*NG]=ground invstep, [1*NG]=narrow-self invstep.
        m_env_alpha_valid = false;  // reset each Newton iter; S1 sets true below
        const bool s1_on = (m_env_scratch && TetMesh.d_point_to_group
                            // [N=1 guard] all -1 p2g = zero env coverage: S1 would flag itself
                            // valid, all_env_frozen unreachable -> Newton pegs at iterCap.
                            && TetMesh.h_groups_present
                            && surf_vertexNum >= 1 && !m_skip_all_collision
                            && getenv("STIFF_PERENV_ALPHA"));
        if(s1_on)
        {
            const int NG = kEnvAlphaSlots, bs = 256;
            cudaMemset(m_env_scratch, 0, 2 * NG * sizeof(double));  // ground+self regions
            _per_env_groundTimeStep_max<<<(surf_vertexNum+bs-1)/bs, bs>>>(
                _vertexes, _surfVerts, _groundOffset, _groundNormal, _moveDir,
                TetMesh.d_point_to_group, m_env_scratch + 0*NG, slackness_a,
                surf_vertexNum, _point_body_id, _ground_skip_body, _ground_body_count, NG);
            if(h_cpNum[0] >= 1)  // narrow-self over OLD _ccd_collisonPairs[0..h_cpNum[0])
                _per_env_selfTimeStep_max<<<(h_cpNum[0]+bs-1)/bs, bs>>>(
                    _vertexes, _ccd_collisonPairs, _moveDir, TetMesh.d_point_to_group,
                    m_env_scratch + 1*NG, slackness_m, h_cpNum[0], NG,
                    getenv("STIFF_CCD_CANON") ? m_d_vloc : nullptr);
        }

        buildBVH_FULLCCD(temp_alpha);
        buildFullCP(temp_alpha);
        if(h_ccd_cpNum > 0)
        {
            double maxSpeed = cfl_largestSpeed(pcg_data.squeue);
            alpha_CFL       = sqrt(dHat) / maxSpeed * 0.5;
            alpha           = std::min(alpha, alpha_CFL);
            if(temp_alpha > 2 * alpha_CFL)
            {
                /*buildBVH_FULLCCD(temp_alpha);
                buildFullCP(temp_alpha);*/
                alpha =
                    std::min(temp_alpha,
                             self_largestFeasibleStepSize(slackness_m, ensure_reduce_scratch(h_ccd_cpNum), h_ccd_cpNum)
                                 * ccd_size);
                alpha = std::max(alpha, alpha_CFL);
            }
        }

        cudaEventRecord(end2);
        //printf("alpha:  %f\n", alpha);

        // [multi-env P3a] read-only: per-env alpha_CFL spread. The global alpha
        // above is ONE scalar (= min over ALL envs of CCD/CFL feasible step). If
        // per-env maxSpeed differs a lot, the global alpha is dragged by the
        // fastest env -> slower envs forced to over-small steps -> per-env
        // line-search would decouple them. Decides whether per-env alpha is worth
        // building. Gated STIFF_PENV_STATS.
        if(TetMesh.d_point_to_group && surf_vertexNum >= 1 && getenv("STIFF_PENV_STATS"))
        {
            const int NG = 256;
            static double* d_mx = nullptr;
            if(!d_mx) cudaMalloc((void**)&d_mx, NG * sizeof(double));
            cudaMemset(d_mx, 0, NG * sizeof(double));
            int bs = 256, gs = (surf_vertexNum + bs - 1) / bs;
            _per_env_max_cfl<<<gs, bs>>>(TetMesh.d_point_to_group, _moveDir,
                                         _surfVerts, d_mx, surf_vertexNum, NG);
            cudaDeviceSynchronize();
            std::vector<double> hmx(NG);
            cudaMemcpy(hmx.data(), d_mx, NG * sizeof(double), cudaMemcpyDeviceToHost);
            double sq = sqrt(dHat);
            printf("[P3a-cfl] k=%d global_alpha=%.3e per-env alpha_CFL=", k, alpha);
            for(int g = 0; g < NG; ++g) if(hmx[g] > 0.0)
                printf(" g%d=%.3e", g, sq / hmx[g] * 0.5);
            printf("\n");
        }

        // [multi-env S1 Phase B] per-env feasible-alpha substrate (PHYSICS-NEUTRAL).
        // Phase A filled ground+narrow-self; here add refined-self (over the NEW
        // _ccd_collisonPairs) + CFL, then combine per the engine's exact logic:
        //   temp_alpha_env = min(1, ground_env, narrowSelf_env)
        //   if ccd pairs: alpha_env = min(temp_alpha_env, alpha_CFL_env);
        //                 if temp_alpha_env > 2*alpha_CFL_env:
        //                     alpha_env = max(min(temp_alpha_env, refinedSelf_env*ccd_size), alpha_CFL_env)
        // Each CCD term = per-env segmented MAX of (1/timestep) -> step = 1/max.
        // The global scalar `alpha` applied below is UNCHANGED -> identical to
        // baseline; this only fills the substrate S2 will consume. Invariant
        // (validated): min_g(m_env_alpha) == global feasibility alpha; N=1 -> one
        // group -> m_env_alpha[g0] == global alpha.
        if(s1_on)
        {
            const int NG = kEnvAlphaSlots, bs = 256;
            cudaMemset(m_env_scratch + 2*NG, 0, 2 * NG * sizeof(double));  // ref+cfl regions
            if(h_ccd_cpNum > 0)  // refined-self over NEW _ccd_collisonPairs[0..h_ccd_cpNum)
                _per_env_selfTimeStep_max<<<(h_ccd_cpNum+bs-1)/bs, bs>>>(
                    _vertexes, _ccd_collisonPairs, _moveDir, TetMesh.d_point_to_group,
                    m_env_scratch + 2*NG, slackness_m, h_ccd_cpNum, NG,
                    getenv("STIFF_CCD_CANON") ? m_d_vloc : nullptr);
            _per_env_max_cfl<<<(surf_vertexNum+bs-1)/bs, bs>>>(
                TetMesh.d_point_to_group, _moveDir, _surfVerts, m_env_scratch + 3*NG,
                surf_vertexNum, NG);
            // [perf] DEVICE-SIDE per-env alpha + freeze (no cudaDeviceSynchronize, no 4x256 D2H, no
            // host loop, no H2D) — writes m_env_alpha directly + a 2-int counter D2H for all_env_frozen.
            // Bit-identical to the host loop (same per-env formulas). Host path kept only under a
            // diagnostic flag.
            const double _sq_    = sqrt(dHat);
            const double _thrcv_ = getenv("STIFF_DECOUPLE_THRESH") ? ((newton_velocity_tol > 0.0) ? (newton_velocity_tol * IPC_dt) : sqrt(Newton_solver_threshold * Newton_solver_threshold * thr_bbox2 * IPC_dt * IPC_dt)) : 0.0;
            const bool _s1diag_ = getenv("STIFF_PENV_STATS") || getenv("STIFF_A0_DUMP")
                               || getenv("STIFF_S1_DEBUG") || getenv("STIFF_ALPHA_DBG");
            if(!_s1diag_)
            {
                static int* d_env_cnt = nullptr;
                if(!d_env_cnt) CUDA_SAFE_CALL(cudaMalloc((void**)&d_env_cnt, 2 * sizeof(int)));
                CUDA_SAFE_CALL(cudaMemsetAsync(d_env_cnt, 0, 2 * sizeof(int)));
                _per_env_alpha_compute<<<(NG + bs - 1) / bs, bs>>>(
                    m_env_alpha, m_env_scratch, NG, _sq_, 1.0, (h_ccd_cpNum > 0) ? 1 : 0,
                    temp_alpha, alpha_CFL, getenv("STIFF_DECOUPLE_THRESH") ? 1 : 0,
                    getenv("STIFF_NO_REFINE") ? 1 : 0, _thrcv_, d_env_cnt);
                int _hc_[2];
                CUDA_SAFE_CALL(cudaMemcpy(_hc_, d_env_cnt, 2 * sizeof(int), cudaMemcpyDeviceToHost));
                m_env_alpha_valid = true;
                all_env_frozen    = (_hc_[0] > 0 && _hc_[1] == _hc_[0]);
            }
            else
            {
            cudaDeviceSynchronize();
            std::vector<double> hg(NG), hs(NG), hr(NG), hmx(NG);
            cudaMemcpy(hg.data(),  m_env_scratch + 0*NG, NG*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(hs.data(),  m_env_scratch + 1*NG, NG*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(hr.data(),  m_env_scratch + 2*NG, NG*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(hmx.data(), m_env_scratch + 3*NG, NG*sizeof(double), cudaMemcpyDeviceToHost);
            const double sq       = sqrt(dHat);
            const double ccd_size = 1.0;
            const bool   have_ccd = (h_ccd_cpNum > 0);
            double       min_env  = 1e30, max_env = 0.0;
            int          n_env    = 0, n_frozen = 0;
            for(int g = 0; g < NG; ++g)
            {
                if(hmx[g] <= 0.0) continue;  // env g has no surface verts -> absent
                double ta = 1.0;             // temp_alpha_env (1/max(invstep))
                if(hg[g] > 0.0) ta = std::min(ta, 1.0 / hg[g]);
                if(hs[g] > 0.0) ta = std::min(ta, 1.0 / hs[g]);
                double a = ta;
                if(have_ccd)
                {
                    double acfl = sq / hmx[g] * 0.5;
                    a = std::min(ta, acfl);
                    // The refinement GATE: the engine gates on the GLOBAL temp_alpha/alpha_CFL,
                    // which makes env_0's branch decision (enter refined CCD or not) depend on the
                    // MATES (global temp_alpha = min over all envs) → BREAKS batch-invariance: at a
                    // frame where global temp_alpha diverges across batches, env_0 enters refinement
                    // in one batch but not the other → env_0's feasible alpha differs → drift →
                    // chaos amplifies. STIFF_DECOUPLE_THRESH gates per-env (ta_g vs acfl_g) so env_0's
                    // branch depends only on env_0 (batch-invariant). Off → exact engine behavior.
                    // [decouple TEST] STIFF_NO_REFINE: skip refinement entirely (env0's a=min(ta,acfl)
                    // → fully per-env/batch-invariant; hr — built from global-temp_alpha CCD pairs — is
                    // the confirmed last leak). isIntersected safety net in lineSearch catches any
                    // resulting penetration. Used to verify hr is the only remaining batch-coupling.
                    double gate_lhs = getenv("STIFF_DECOUPLE_THRESH") ? ta   : temp_alpha;
                    double gate_rhs = getenv("STIFF_DECOUPLE_THRESH") ? acfl : alpha_CFL;
                    if(!getenv("STIFF_NO_REFINE") && gate_lhs > 2.0 * gate_rhs)
                    {
                        double refined = (hr[g] > 0.0) ? 1.0 / hr[g] : 1.0;
                        a = std::min(ta, refined * ccd_size);
                        a = std::max(a, acfl);
                    }
                }
                h_env_alpha[g] = a;
                // [decouple] FREEZE env g once IT has converged (per-env max-move < the per-env Newton
                // threshold), so env g stops stepping at ITS OWN convergence iter — NOT the global loop
                // count. The merged Newton loop runs a BATCH-DEPENDENT number of iters (harder mates →
                // more iters: measured A=13 vs B=16 at frame0); during the extra iters an already-
                // converged env's ABD (dq small but nonzero) keeps stepping → its gripper/arm pose
                // drifts batch-dependently → FEM-pin seed. Freezing at the env's own convergence makes
                // env g's total steps batch-invariant. Block-diagonal per-env solve ⇒ monotonic ⇒ no
                // re-activation needed. Gated STIFF_DECOUPLE_THRESH.
                if(getenv("STIFF_DECOUPLE_THRESH"))
                {
                    double thr_cv = ((newton_velocity_tol > 0.0) ? (newton_velocity_tol * IPC_dt) : sqrt(Newton_solver_threshold * Newton_solver_threshold * thr_bbox2 * IPC_dt * IPC_dt));
                    if(hmx[g] < thr_cv) h_env_alpha[g] = 0.0;
                }
                if(h_env_alpha[g] == 0.0) ++n_frozen;   // [decouple] per-env converged (frozen)
                min_env = std::min(min_env, a);
                max_env = std::max(max_env, a);
                ++n_env;
                if(getenv("STIFF_A0_DUMP") && g == 0
                   && (!getenv("STIFF_DUMP_FRAME") || g_dec_frame == atoi(getenv("STIFF_DUMP_FRAME"))))
                {   // [decouple] env0 per-iter: applied alpha (h_env_alpha[g], post-freeze) + hmx (max-move) + frozen?
                    double thr_cv = ((newton_velocity_tol > 0.0) ? (newton_velocity_tol * IPC_dt) : sqrt(Newton_solver_threshold * Newton_solver_threshold * thr_bbox2 * IPC_dt * IPC_dt));
                    printf("[a0] frame=%d k=%d a_applied=%.17e a_feasible=%.17e hmx=%.17e thr_cv=%.6e frozen=%d\n",
                           g_dec_frame, (int)k, h_env_alpha[g], a, hmx[g], thr_cv, (int)(h_env_alpha[g] == 0.0));
                }
                if(getenv("STIFF_S1_DEBUG") && alpha > 0.99 && a < 0.9)
                    printf("  [S1-dbg] g%d a=%.4e ta=%.4e ground=%.4e narrow=%.4e "
                           "refined=%.4e acfl=%.4e ccdN=%d alphaCFLglob=%.4e (global=%.4e)\n",
                           g, a, ta, hg[g]>0?1.0/hg[g]:9.99, hs[g]>0?1.0/hs[g]:9.99,
                           hr[g]>0?1.0/hr[g]:9.99, have_ccd?sq/hmx[g]*0.5:9.99,
                           (int)h_ccd_cpNum, alpha_CFL, alpha);
            }
            // [env-det dbg] cross-env alpha mismatch (STIFF_ALPHA_DBG): pin the component that differs.
            if(getenv("STIFF_ALPHA_DBG") && NG >= 2 && hmx[0] > 0.0 && hmx[1] > 0.0
               && (h_env_alpha[0] != h_env_alpha[1] || hg[0] != hg[1] || hs[0] != hs[1]
                   || hr[0] != hr[1] || hmx[0] != hmx[1]))
            {
                static int _ad = 0;
                if(_ad++ < 12)
                    printf("[alpha-dbg] a0=%.17e a1=%.17e | hg %.17e/%.17e hs %.17e/%.17e "
                           "hr %.17e/%.17e hmx %.17e/%.17e ccdN=%d\n",
                           h_env_alpha[0], h_env_alpha[1], hg[0], hg[1], hs[0], hs[1],
                           hr[0], hr[1], hmx[0], hmx[1], (int)h_ccd_cpNum);
            }
            CUDA_SAFE_CALL(cudaMemcpy(m_env_alpha, h_env_alpha.data(),
                                      NG * sizeof(double), cudaMemcpyHostToDevice));
            m_env_alpha_valid = true;  // m_env_alpha fresh -> lineSearch may use it
            // [decouple] all present envs per-env frozen (converged)? → drives the loop-exit override
            // so the loop runs until env_0 (and every env) reaches ITS OWN convergence, batch-independent.
            all_env_frozen = (n_env > 0 && n_frozen == n_env);
            if(getenv("STIFF_PENV_STATS"))
                printf("[S1-envalpha] k=%d global_alpha=%.6e min_env=%.6e max_env=%.6e "
                       "n_env=%d rel=%.2e (neutral-check; headroom=max_env/global)\n",
                       k, alpha, min_env, max_env, n_env,
                       fabs(alpha - min_env) / std::max(alpha, 1e-30));
            }   // [perf] end diagnostic host path
        }

        // [S4-dev] derive next iter's active mask from the freeze decision already on device
        // (m_env_alpha == 0 ⇔ env converged this iter). Zero D2H — replaces the S4 host detection.
        // Frozen set is final here: the S3 per-env backtrack only halves nonzero alphas (never → 0).
        if(s4_dev_mask && m_env_alpha_valid)
            _mask_from_env_alpha<<<(kEnvAlphaSlots + 255) / 256, 256>>>(
                m_env_active, m_env_alpha, kEnvAlphaSlots);

        cudaEventRecord(e2b);   // [phase-time] end of S1 per-env-alpha block / start of lineSearch
        bool isStop = lineSearch(TetMesh, alpha, alpha_CFL);

        cudaEventRecord(end3);
        postLineSearch(TetMesh, alpha);
        //computeGradientAndHessian(TetMesh);
        cudaEventRecord(end4);

        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        float time00, time11, time22, time33, time44;
        cudaEventElapsedTime(&time00, start, end0);
        cudaEventElapsedTime(&time11, end0, end1);
        //total_Cg_time += time1;
        cudaEventElapsedTime(&time22, end1, end2);
        cudaEventElapsedTime(&time33, end2, end3);
        cudaEventElapsedTime(&time44, end3, end4);
        {   // [phase-time] time3 sub-split: S1 per-env alpha (end2->e2b) vs lineSearch (e2b->end3)
            float t3a = 0, t3b = 0;
            cudaEventElapsedTime(&t3a, end2, e2b);
            cudaEventElapsedTime(&t3b, e2b, end3);
            g_t3_s1_ms += t3a;
            g_t3_ls_ms += t3b;
        }
        time0 += time00;
        time1 += time11;
        time2 += time22;
        time3 += time33;
        time4 += time44;
        ////*cflTime = ptime;
        //printf("time0 = %f,  time1 = %f,  time2 = %f,  time3 = %f,  time4 = %f\n",
        //       time00,
        //       time11,
        //       time22,
        //       time33,
        //       time44);
        (cudaEventDestroy(start));
        (cudaEventDestroy(end0));
        (cudaEventDestroy(end1));
        (cudaEventDestroy(end2));
        (cudaEventDestroy(end3));
        (cudaEventDestroy(end4));
        (cudaEventDestroy(e2b));
        totalTimeStep += alpha;

        // Semi-implicit early exit (ref: arXiv 2512.12151, Algorithm 1)
        // beta tracks cumulative line-search progress; when alpha≈1 (good step),
        // beta decays fast -> early exit.  When alpha is small, beta stays large.
        if(semi_implicit_enabled && k >= semi_implicit_min_iter)
        {
            semi_beta *= (1.0 - alpha);
            if(semi_beta <= semi_implicit_beta_tol)
            {
                printf("  [semi-implicit] early exit at Newton iter %d (beta=%.6e, tol=%.6e)\n",
                       k, semi_beta, semi_implicit_beta_tol);
                k++;
                break;
            }
        }
    }
    // [multi-env S4] clear the linear-system mask so later/other solves are unmasked
    if(s4_mask_on) m_global_linear_system->set_env_mask(nullptr, nullptr, 0);
    //iterV.push_back(k);
    //std::ofstream outiter("iterCount.txt");
    //for(int ii = 0; ii < iterV.size(); ii++)
    //{
    //    outiter << iterV[ii] << std::endl;
    //}
    //outiter.close();
    if(g_gipc_log_level >= 1)
        printf("\n\n      Kappa: %f                               iteration k:  %d\n", Kappa, k);
    // [phase-time] cumulative GPU ms per phase (across frames). time0=Hessian/grad assembly,
    // time1=PCG linear solve, time2=CCD-BVH build, time3=line-search(+per-env alpha), time4=κ update.
    // Compare modes to localize the isolated/strict slowdown. Also reports Newton iters this frame.
    if(getenv("STIFF_PHASE_TIME"))
        printf("[phase-time cum-ms] Hess=%.0f PCG=%.0f ccdBVH=%.0f lineSearch=%.0f kappaUpd=%.0f | ls-split[s1=%.0f ls=%.0f] ls-inner[e=%.0f bvh=%.0f cp=%.0f step=%.0f] | this-frame-newton-iters=%d cum-pcg-iters=%lld\n",
               time0, time1, time2, time3, time4, g_t3_s1_ms, g_t3_ls_ms,
               g_ls_e_ms, g_ls_bvh_ms, g_ls_cp_ms, g_ls_step_ms, k, (long long)total_Cg_count);
    return k;
}

void GIPC::updateVelocities(device_TetraData& TetMesh)
{
    int numbers = vertexNum;
    if(numbers <= 0)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _updateVelocities<<<blockNum, threadNum>>>(
        TetMesh.vertexes, TetMesh.o_vertexes, TetMesh.velocities, TetMesh.BoundaryType, IPC_dt, numbers);

    m_abd_system->update_velocity(*m_abd_sim_data);
}

void GIPC::updateBoundary(device_TetraData& TetMesh, double alpha)
{
    int numbers = vertexNum;
    if(numbers <= 0)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _updateBoundary<<<blockNum, threadNum>>>(
        TetMesh.vertexes, TetMesh.BoundaryType, _moveDir, alpha, numbers);
}

void GIPC::updateBoundaryMoveDir(device_TetraData& TetMesh, double alpha, int fid)
{
    int numbers = vertexNum;
    if(numbers <= 0)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _updateBoundaryMoveDir<<<blockNum, threadNum>>>(
        TetMesh.vertexes, TetMesh.BoundaryType, _moveDir, IPC_dt, FEM::PI, alpha, numbers, fid);
}


void GIPC::computeXTilta(device_TetraData& TetMesh, const double& rate)
{
    int numbers = vertexNum;
    if(numbers <= 0)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _computeXTilta<<<blockNum, threadNum>>>(TetMesh.BoundaryType,
                                            TetMesh.velocities,
                                            TetMesh.o_vertexes,
                                            TetMesh.xTilta,
                                            TetMesh.apply_gravity,
                                            IPC_dt,
                                            rate,
                                            gravity,
                                            numbers);

    m_abd_system->cal_q_tilde(*m_abd_sim_data);
}

extern int total_Frames;   // file-scope frame counter (defined below); drives the stitch/soft target
                           // (update_soft_constraint_target_position(total_Frames+1)) → MUST be in the
                           // checkpoint or the restart's stitch target is for the wrong frame.

// [decouple debug] full-state checkpoint. Persistent cross-frame state only (friction/contact is
// ephemeral, rebuilt each step from positions): FEM vertexes/o_vertexes/velocities/xTilta +
// ABD q/q_prev/q_v + Kappa + total_Frames. Binary: [magic u32][vN u32][nb u32][4*vN double3 FEM]
// [3*nb Vector12 ABD][Kappa double][total_Frames i32]. Load restores them → next step() bit-identical.
void GIPC::save_checkpoint(device_TetraData& tm, const char* path)
{
    const int vN = (int)vertexNum;
    const int nb = (int)abd_fem_count_info.abd_body_num;
    std::vector<double3> hv(vN), ho(vN), hvel(vN), hxt(vN);
    CUDA_SAFE_CALL(cudaMemcpy(hv.data(),   tm.vertexes,   vN*sizeof(double3), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(ho.data(),   tm.o_vertexes, vN*sizeof(double3), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(hvel.data(), tm.velocities, vN*sizeof(double3), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(hxt.data(),  tm.xTilta,     vN*sizeof(double3), cudaMemcpyDeviceToHost));
    std::vector<double> hq(12*nb), hqp(12*nb), hqv(12*nb);
    if(nb > 0 && m_abd_sim_data)
    {
        auto& d = m_abd_sim_data->device;
        CUDA_SAFE_CALL(cudaMemcpy(hq.data(),  reinterpret_cast<const double*>(d.body_id_to_q.data()),      12*nb*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(hqp.data(), reinterpret_cast<const double*>(d.body_id_to_q_prev.data()), 12*nb*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(hqv.data(), reinterpret_cast<const double*>(d.body_id_to_q_v.data()),    12*nb*sizeof(double), cudaMemcpyDeviceToHost));
    }
    FILE* f = fopen(path, "wb");
    if(!f) { printf("[ckpt] cannot open %s for write\n", path); return; }
    uint32_t magic = 0x53544B50u, uvN = (uint32_t)vN, unb = (uint32_t)nb;
    fwrite(&magic,1,4,f); fwrite(&uvN,1,4,f); fwrite(&unb,1,4,f);
    fwrite(hv.data(),sizeof(double3),vN,f); fwrite(ho.data(),sizeof(double3),vN,f);
    fwrite(hvel.data(),sizeof(double3),vN,f); fwrite(hxt.data(),sizeof(double3),vN,f);
    fwrite(hq.data(),sizeof(double),12*nb,f); fwrite(hqp.data(),sizeof(double),12*nb,f);
    fwrite(hqv.data(),sizeof(double),12*nb,f);
    fwrite(&Kappa,sizeof(double),1,f);
    fwrite(&total_Frames,sizeof(int),1,f);
    fclose(f);
    printf("[ckpt] saved %s (vN=%d nb=%d Kappa=%.6e total_Frames=%d)\n", path, vN, nb, Kappa, total_Frames);
}

void GIPC::load_checkpoint(device_TetraData& tm, const char* path)
{
    FILE* f = fopen(path, "rb");
    if(!f) { printf("[ckpt] cannot open %s for read\n", path); return; }
    uint32_t magic=0, uvN=0, unb=0;
    size_t rd = fread(&magic,1,4,f); rd += fread(&uvN,1,4,f); rd += fread(&unb,1,4,f);
    if(magic != 0x53544B50u || (int)uvN != (int)vertexNum || (int)unb != (int)abd_fem_count_info.abd_body_num)
    { printf("[ckpt] MISMATCH magic=%x vN=%u(exp %u) nb=%u(exp %u)\n", magic, uvN, (uint32_t)vertexNum, unb, (uint32_t)abd_fem_count_info.abd_body_num); fclose(f); return; }
    const int vN = (int)uvN, nb = (int)unb;
    std::vector<double3> hv(vN), ho(vN), hvel(vN), hxt(vN);
    std::vector<double> hq(12*nb), hqp(12*nb), hqv(12*nb); double kap=0;
    rd += fread(hv.data(),sizeof(double3),vN,f); rd += fread(ho.data(),sizeof(double3),vN,f);
    rd += fread(hvel.data(),sizeof(double3),vN,f); rd += fread(hxt.data(),sizeof(double3),vN,f);
    rd += fread(hq.data(),sizeof(double),12*nb,f); rd += fread(hqp.data(),sizeof(double),12*nb,f);
    rd += fread(hqv.data(),sizeof(double),12*nb,f);
    rd += fread(&kap,sizeof(double),1,f);
    int tf = 0; rd += fread(&tf,sizeof(int),1,f); fclose(f); (void)rd;
    CUDA_SAFE_CALL(cudaMemcpy(tm.vertexes,   hv.data(),   vN*sizeof(double3), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(tm.o_vertexes, ho.data(),   vN*sizeof(double3), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(tm.velocities, hvel.data(), vN*sizeof(double3), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(tm.xTilta,     hxt.data(),  vN*sizeof(double3), cudaMemcpyHostToDevice));
    if(nb > 0 && m_abd_sim_data)
    {
        auto& d = m_abd_sim_data->device;
        CUDA_SAFE_CALL(cudaMemcpy(reinterpret_cast<double*>(d.body_id_to_q.data()),      hq.data(),  12*nb*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(reinterpret_cast<double*>(d.body_id_to_q_prev.data()), hqp.data(), 12*nb*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(reinterpret_cast<double*>(d.body_id_to_q_v.data()),    hqv.data(), 12*nb*sizeof(double), cudaMemcpyHostToDevice));
    }
    Kappa = kap;
    total_Frames = tf;
    printf("[ckpt] loaded %s (vN=%d nb=%d Kappa=%.6e total_Frames=%d)\n", path, vN, nb, Kappa, total_Frames);
}


int    totalNT          = 0;
double totalTime        = 0;
int    total_Frames     = 0;
double ttime0           = 0;
double ttime1           = 0;
double ttime2           = 0;
double ttime3           = 0;
double ttime4           = 0;
bool   isUpdateBoundary = false;
void   GIPC::IPC_Solver(device_TetraData& TetMesh)
{
    //double animation_fullRate = 0;
    cudaEvent_t start, end0;
    cudaEventCreate(&start);
    cudaEventCreate(&end0);
    double alpha = 1;
    cudaEventRecord(start);
    //    if(isRotate&&total_Frames*IPC_dt>=2.2){
    //        isRotate = false;
    //        updateBoundary2(TetMesh);
    //    }
    if(isUpdateBoundary)
    {
        updateBoundaryMoveDir(TetMesh, alpha, total_Frames);
        buildBVH_FULLCCD(alpha);
        buildFullCP(alpha);
        if(h_ccd_cpNum > 0)
        {
            double slackness_m = 0.8;
            alpha              = std::min(alpha,
                             self_largestFeasibleStepSize(slackness_m, ensure_reduce_scratch(h_ccd_cpNum), h_ccd_cpNum));
        }
        //updateBoundary(TetMesh, alpha);

        CUDA_SAFE_CALL(cudaMemcpy(TetMesh.temp_double3Mem,
                                  TetMesh.vertexes,
                                  vertexNum * sizeof(double3),
                                  cudaMemcpyDeviceToDevice));
        updateBoundaryMoveDir(TetMesh, alpha, total_Frames);
        stepForward(TetMesh.vertexes, TetMesh.temp_double3Mem, _moveDir, TetMesh.BoundaryType, 1, true, vertexNum);
        //step_forward(TetMesh, 1, true);

        bool rehash = true;

        buildBVH();
        int numOfIntersect = 0;
        while(isIntersected(TetMesh))
        {
            printf("type 6 intersection happened:    %f\n", alpha);
            alpha /= 2.0;
            updateBoundaryMoveDir(TetMesh, alpha, total_Frames);
            numOfIntersect++;
            stepForward(TetMesh.vertexes,
                        TetMesh.temp_double3Mem,
                        _moveDir,
                        TetMesh.BoundaryType,
                        1,
                        true,
                        vertexNum);
            //step_forward(TetMesh, 1, true);
            buildBVH();
        }

        buildCP();
        printf("boundary alpha: %f\n  finished a step\n", alpha);
    }

    TetMesh.update_soft_constraint_target_position(total_Frames + 1, IPC_dt);
    //suggestKappa(Kappa);
    upperBoundKappa(Kappa);
    if(Kappa < 1e-16)
    {
        suggestKappa(Kappa);
    }
    initKappa(TetMesh);
    //Kappa = 1e4;
#ifdef USE_FRICTION
    ensure_frictionBuffers();  // [0be8da3-port] grow-only, no per-frame malloc
    buildFrictionSets();
#endif
    animation_fullRate = animation_subRate;
    int    k           = 0;
    double time0       = 0;
    double time1       = 0;
    double time2       = 0;
    double time3       = 0;
    double time4       = 0;
    while(true)
    {
        //if (h_cpNum[0] > 0) return;
        tempMalloc_closeConstraint();
        CUDA_SAFE_CALL(cudaMemset(_close_cpNum, 0, sizeof(uint32_t)));
        CUDA_SAFE_CALL(cudaMemset(_close_gpNum, 0, sizeof(uint32_t)));

        totalNT += solve_subIP(TetMesh, time0, time1, time2, time3, time4);

        double2 minMaxDist1 = minMaxGroundDist();
        double2 minMaxDist2 = minMaxSelfDist();

        double minDist = std::min(minMaxDist1.x, minMaxDist2.x);
        double maxDist = std::max(minMaxDist1.y, minMaxDist2.y);


        bool finishMotion = animation_fullRate > 0.99 ? true : false;

        if(finishMotion)
        {
            tempFree_closeConstraint();
            break;
            //}
        }
        else
        {
            tempFree_closeConstraint();
        }

        animation_fullRate += animation_subRate;
        //updateVelocities(TetMesh);

        //computeXTilta(TetMesh, 1);
#ifdef USE_FRICTION
        ensure_frictionBuffers();  // [0be8da3-port] grow-only, no sub-iter realloc
        buildFrictionSets();
#endif
    }

#ifdef USE_FRICTION
    // [0be8da3-port] friction buffers persist across frames; freed in FREE_DEVICE_MEM.
#endif

    updateVelocities(TetMesh);

    computeXTilta(TetMesh, 1);
    cudaEventRecord(end0);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    float tttime;
    cudaEventElapsedTime(&tttime, start, end0);
    cudaEventDestroy(start);
    cudaEventDestroy(end0);
    totalTime += tttime;
    total_Frames++;
    if(g_gipc_log_level >= 1)
        printf("average time cost:     %f,    frame id:   %d\n", totalTime / totalNT, total_Frames);

    // [multi-env P3a] validate the segmented per-env reduction primitive on real
    // data: per-env vertex count (must match the substrate, e.g. 11433/env) and a
    // real per-env quantity (velocity norm). Read-only diagnostic, gated.
    if(getenv("STIFF_PENV_STATS") && TetMesh.d_point_to_group)
    {
        const int NG = 256;
        static double* d_sq = nullptr;
        static int*    d_cnt = nullptr;
        if(!d_sq)
        {
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_sq, NG * sizeof(double)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_cnt, NG * sizeof(int)));
        }
        CUDA_SAFE_CALL(cudaMemset(d_sq, 0, NG * sizeof(double)));
        CUDA_SAFE_CALL(cudaMemset(d_cnt, 0, NG * sizeof(int)));
        int bs = 256, gs = (vertexNum + bs - 1) / bs;
        _per_env_sqnorm_accum<<<gs, bs>>>(TetMesh.d_point_to_group, TetMesh.velocities,
                                          d_sq, d_cnt, vertexNum, NG);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        std::vector<double> h_sq(NG);
        std::vector<int> h_cnt(NG);
        CUDA_SAFE_CALL(cudaMemcpy(h_sq.data(), d_sq, NG * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(h_cnt.data(), d_cnt, NG * sizeof(int), cudaMemcpyDeviceToHost));
        printf("[P3a-reduce] frame %d per-env:", total_Frames);
        for(int g = 0; g < NG; ++g)
            if(h_cnt[g] > 0) printf(" g%d[n=%d |v|=%.4e]", g, h_cnt[g], sqrt(h_sq[g]));
        printf("\n");
    }

    ttime0 += time0;
    ttime1 += time1;
    ttime2 += time2;
    ttime3 += time3;
    ttime4 += time4;


    std::ofstream outTime("timeCost.txt");

    outTime << "time0: " << ttime0 / 1000.0 << std::endl;
    outTime << "time1: " << ttime1 / 1000.0 << std::endl;
    outTime << "time2: " << ttime2 / 1000.0 << std::endl;
    outTime << "time3: " << ttime3 / 1000.0 << std::endl;
    outTime << "time4: " << ttime4 / 1000.0 << std::endl;
    outTime << "time_makePD: " << timemakePd / 1000.0 << std::endl;

    outTime << "totalTime: " << totalTime / 1000.0 << std::endl;
    outTime << "total iter: " << totalNT << std::endl;
    outTime << "frames: " << total_Frames << std::endl;
    outTime << "totalCollisionNum: " << totalCollisionPairs << std::endl;
    outTime << "averageCollision: " << totalCollisionPairs / totalNT << std::endl;
    outTime << "maxCOllisionPairNum: " << maxCOllisionPairNum << std::endl;
    outTime << "totalCgTime: " << total_Cg_count << std::endl;
    outTime.close();


    auto& stats = gipc::Statistics::instance();

    stats.at_current_frame()["timer"] =
        gipc::GlobalTimer::current()->report_merged_as_json();
    if(g_gipc_log_level >= 1)
        gipc::GlobalTimer::current()->print_merged_timings();
    gipc::GlobalTimer::current()->clear();
    stats.write_to_file(std::string{gipc::output_dir()} + "/stats.json");

    auto f = stats.frame();
    stats.frame(f + 1);
}
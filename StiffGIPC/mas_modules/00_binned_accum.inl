//
// MASPreconditioner.cu
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#include "MASPreconditioner.cuh"
#include <linear_system/utils/binned_reduce.cuh>  // [4.3] MAS determinism: binned reproducible FP
#include "cuda_tools/cuda_tools.h"
#include "device_launch_parameters.h"
#include <math_constants.h>
#include <muda/launch/launch.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <vector>
#include <bitset>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "cooperative_groups.h"
using namespace cooperative_groups;
//#include "Eigen/Eigen"
using namespace std;
#define SYME
#define GROUP

// [4.3] MAS determinism: binned reproducible-FP accumulation for the three non-deterministic
// float/double atomicAdd targets — d_multiLevelR (restrict), d_multiLevelZ (Schwarz apply),
// d_inverseMatMas (coarse Hessian aggregation). Order-independent ⇒ bit-identical. The bins are
// set via cudaMemcpyToSymbol before the depositing kernels; combined back by the kernels below.
#define MAS_NB (BANKSIZE * (BANKSIZE + 1) / 2)
__device__ double* g_mRbin  = nullptr;
__device__ double* g_mZbin  = nullptr;
__device__ double* g_matbin = nullptr;

__global__ void _mas_comb_mZ(Precision_T3* mZ,
                             const double* bin,
                             int n,
                             const int2* extent)
{
    if(extent)
        n = extent->y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    mZ[i].x = binned_combine(bin + ((size_t)i * 3 + 0) * BINNED_K);
    mZ[i].y = binned_combine(bin + ((size_t)i * 3 + 1) * BINNED_K);
    mZ[i].z = binned_combine(bin + ((size_t)i * 3 + 2) * BINNED_K);
}
__global__ void _mas_comb_mR(Eigen::Vector3f* mR,
                             const double* bin,
                             int start,
                             int end,
                             const int2* extent)
{
    if(extent)
        end = extent->y;
    int i = start + blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= end) return;
    mR[i][0] = (float)binned_combine(bin + ((size_t)i * 3 + 0) * BINNED_K);
    mR[i][1] = (float)binned_combine(bin + ((size_t)i * 3 + 1) * BINNED_K);
    mR[i][2] = (float)binned_combine(bin + ((size_t)i * 3 + 2) * BINNED_K);
}
__global__ void _mas_comb_mat(__GEIGEN__::MasMatrixSymT* mat,
                              const double* bin,
                              int startC,
                              int endC,
                              const int2* extent)
{
    if(extent)
        endC = extent->y / BANKSIZE;
    int t    = blockIdx.x * blockDim.x + threadIdx.x;
    int nblk = (endC - startC) * MAS_NB;
    if(t >= nblk) return;
    int cPid  = startC + t / MAS_NB;
    int index = t % MAS_NB;
#pragma unroll
    for(int i = 0; i < 3; i++)
#pragma unroll
        for(int j = 0; j < 3; j++)
            mat[cPid].M[index](i, j) =
                binned_combine(bin + (((size_t)cPid * MAS_NB + index) * 9 + i * 3 + j) * BINNED_K);
}

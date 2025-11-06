//
// PCG_SOLVER.cu
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#include "PCG_SOLVER.cuh"
#include "device_launch_parameters.h"
#include "gpu_eigen_libs.cuh"
#include "cuda_tools/cuda_tools.h"

void PCG_Data::Malloc_DEVICE_MEM(const int& vertexNum, const int& tetrahedraNum)
{

    CUDA_SAFE_CALL(cudaMalloc((void**)&squeue,
                              std::max(vertexNum, tetrahedraNum) * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&dx, vertexNum * sizeof(double3)));
}

void PCG_Data::FREE_DEVICE_MEM()
{
    CUDA_SAFE_CALL(cudaFree(squeue));
    //CUDA_SAFE_CALL(cudaFree(b));
    CUDA_SAFE_CALL(cudaFree(P));
    CUDA_SAFE_CALL(cudaFree(r));
    CUDA_SAFE_CALL(cudaFree(c));
    CUDA_SAFE_CALL(cudaFree(z));
    CUDA_SAFE_CALL(cudaFree(q));
    CUDA_SAFE_CALL(cudaFree(s));
    CUDA_SAFE_CALL(cudaFree(dx));
    //CUDA_SAFE_CALL(cudaFree(tempDx));
    if(P_type > 0)
    {
        CUDA_SAFE_CALL(cudaFree(filterTempVec3));
        CUDA_SAFE_CALL(cudaFree(preconditionTempVec3));
    }
    if(P_type == 1)
    {
        MP.FreeMAS();
    }
}

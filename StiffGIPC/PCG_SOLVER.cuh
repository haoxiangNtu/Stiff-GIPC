//
// PCG_SOLVER.cuh
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#pragma once
#ifndef _PCG_SOLVER_CUH_
#define _PCG_SOLVER_CUH_
#include <cuda_runtime.h>
#include "device_fem_data.cuh"
#include <cstdint>
#include "MASPreconditioner.cuh"

class PCG_Data
{
  public:
    double*                 squeue;
    double3*                b;
    __GEIGEN__::Matrix3x3d* P;
    double3*                r;
    double3*                c;
    double3*                q;
    double3*                s;
    double3*                z;
    double3*                dx;
    double3*                tempDx;

    double3*          filterTempVec3;
    double3*          preconditionTempVec3;
    MASPreconditioner MP;

    int P_type = 1;

  public:
    void Malloc_DEVICE_MEM(const int& vertex_num, const int& tetradedra_num);
    void FREE_DEVICE_MEM();
};
#endif  // ! _PCG_SOLVER_CUH_

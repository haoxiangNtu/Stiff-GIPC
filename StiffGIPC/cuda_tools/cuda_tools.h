//
// cuda_tool.h
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang and Jiming Ruan. All rights reserved.
//

#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include<vector>
#define CUDA_SAFE_CALL(err) cuda_safe_call_(err, __FILE__, __LINE__)
const static int default_threads = 256;
//#define CUDA_KERNEL_CHECK(err)  cuda_kernel_check_(err, __FILE__, __LINE__)

inline unsigned long long LogIte(unsigned long long value)
{
    if(value == 0)
    {
        return 0;
    }
    return 1 + LogIte(value >> 1);
}
inline unsigned long long Log2(unsigned long long value)
{
    value -= 1;
    if(value == 0)
    {
        return 1;
    }
    return LogIte(value);
}


// [C-3 probe] when set (capture-probe scope), CUDA errors THROW instead of
// aborting so the graph-capture harness can terminate the capture, report the
// blocking site, and fall back gracefully.
inline bool& cuda_safe_call_throws()
{
    static thread_local bool v = false;
    return v;
}

// [C-3] true while the Newton/GH graph harness is recording: capture-illegal
// conveniences (defensive full-device syncs, diagnostics) gate on this.
inline bool& gipc_in_graph_capture()
{
    static thread_local bool v = false;
    return v;
}

inline void cuda_safe_call_(cudaError err, const char* file_name, const int num_line)
{
    if(cudaSuccess != err)
    {
        // CUDA runtime teardown at process / Python-interpreter exit: the driver
        // is already unloading, so cudaFree() (and friends) in static / global
        // object destructors return cudaErrorCudartUnloading. The allocations are
        // reclaimed by the driver anyway, so swallow this instead of aborting —
        // otherwise every clean exit ends in "Aborted (core dumped)".
        if(err == cudaErrorCudartUnloading)
            return;

        if(cuda_safe_call_throws())
            throw std::runtime_error(std::string(file_name) + "["
                                     + std::to_string(num_line) + "]: CUDA error "
                                     + std::to_string((int)err) + " ("
                                     + cudaGetErrorString(err) + ")");
        std::cerr << file_name << "[" << num_line << "]: "
                  << "CUDA Running API error[" << (int)err
                  << "]: " << cudaGetErrorString(err) << std::endl;

        std::abort();
    }
}



template <typename... Arguments>
void LaunchCudaKernal(int gs, int bs, size_t mem, void (*f)(Arguments...), Arguments... args)
{
    if(gs < 1)
        return;
    if(!mem)
    {
        f<<<gs, bs>>>(args...);
    }
    else
    {
        f<<<gs, bs, mem>>>(args...);
    }
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        std::cerr << __FILE__ << "[" << __LINE__ << "]: "
                  << "CUDA Running API error[" << (int)err
                  << "]: " << cudaGetErrorString(err) << std::endl;
        exit(0);
    }
}

template <typename... Arguments>
void LaunchCudaKernal_default(int total, int bs, size_t mem, void (*f)(Arguments...), Arguments... args)
{
    int gs = (total + bs - 1) / bs;
    if(gs < 1)
        return;
    if(!mem)
    {
        f<<<gs, bs>>>(args...);
    }
    else
    {
        f<<<gs, bs, mem>>>(args...);
    }
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        std::cerr << __FILE__ << "[" << __LINE__ << "]: "
                  << "CUDA Running API error[" << (int)err
                  << "]: " << cudaGetErrorString(err) << std::endl;
        exit(0);
    }
}
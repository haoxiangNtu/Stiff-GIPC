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
#include <cstdlib>   // std::getenv for STIFF_CUDA_DEBUG
#define CUDA_SAFE_CALL(err) cuda_safe_call_(err, __FILE__, __LINE__)
// L2 diagnostics: synchronous per-phase CUDA error checkpoint. Default OFF
// (one cached-bool branch, ~zero cost). Set STIFF_CUDA_DEBUG=1 to localize
// asynchronous kernel-launch errors to the phase that produced them instead
// of having them surface at a later innocent cudaFree/cudaMemcpy.
#define GIPC_CUDA_CHECKPOINT(tag) gipc_cuda_checkpoint((tag), __FILE__, __LINE__)
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

        std::cerr << file_name << "[" << num_line << "]: "
                  << "CUDA Running API error[" << (int)err
                  << "]: " << cudaGetErrorString(err) << std::endl;

        // ---- L1 diagnostics (only runs on failure; zero cost on success path) ----
        // 1) Sticky/last error: this CUDA_SAFE_CALL may merely be the first
        //    synchronizing call AFTER an asynchronous kernel launch that actually
        //    failed. cudaGetLastError() surfaces that prior error.
        cudaError_t last = cudaGetLastError();
        if(last != cudaSuccess && last != err)
            std::cerr << "  [diag] prior/sticky error: [" << (int)last << "] "
                      << cudaGetErrorString(last) << std::endl;
        // 2) GPU memory (an OOM often manifests as invalidArgument[1] downstream).
        size_t mem_free = 0, mem_total = 0;
        if(cudaMemGetInfo(&mem_free, &mem_total) == cudaSuccess)
            std::cerr << "  [diag] GPU memory: " << (mem_free >> 20) << " MiB free / "
                      << (mem_total >> 20) << " MiB total" << std::endl;
        // 3) Async-error hint: tell the user how to localize the true failure.
        std::cerr << "  [diag] NOTE: CUDA reports kernel-launch errors at the NEXT"
                     " synchronizing call, so the file:line above can be an innocent"
                     " bystander (e.g. a cudaFree/cudaMemcpy). Re-run with"
                     " STIFF_CUDA_DEBUG=1 to synchronize after each solver phase and"
                     " pinpoint the phase that actually failed." << std::endl;

        std::abort();
    }
}

// L2: env-gated synchronous checkpoint. Reads STIFF_CUDA_DEBUG exactly once
// (cached static bool), so when disabled this is a single perfectly-predicted
// branch — no measurable cost on the hot path. When enabled, forces a
// cudaDeviceSynchronize + error check so an async kernel error is attributed
// to the phase tag instead of surfacing at a later innocent call.
inline bool gipc_cuda_debug_enabled()
{
    static const bool dbg = []{
        const char* v = std::getenv("STIFF_CUDA_DEBUG");
        return v && v[0] && v[0] != '0';
    }();
    return dbg;
}

inline void gipc_cuda_checkpoint(const char* tag, const char* file_name, const int num_line)
{
    if(!gipc_cuda_debug_enabled())
        return;  // default: cached-bool branch, ~zero cost

    cudaError_t err = cudaDeviceSynchronize();
    if(err == cudaSuccess)
        err = cudaGetLastError();
    if(err != cudaSuccess && err != cudaErrorCudartUnloading)
    {
        std::cerr << file_name << "[" << num_line << "]: "
                  << "[CUDA-DEBUG] error after phase '" << tag << "': ["
                  << (int)err << "] " << cudaGetErrorString(err) << std::endl;
        size_t mem_free = 0, mem_total = 0;
        if(cudaMemGetInfo(&mem_free, &mem_total) == cudaSuccess)
            std::cerr << "  [diag] GPU memory: " << (mem_free >> 20) << " MiB free / "
                      << (mem_total >> 20) << " MiB total" << std::endl;
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
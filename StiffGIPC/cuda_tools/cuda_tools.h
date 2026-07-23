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
#include <vector>
#include <unordered_map>
#include <string>
#include <mutex>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
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


// [audit] STIFF_API_AUDIT=1: count every CUDA_SAFE_CALL site (file:line) and
// dump the sorted table at exit. Off by default; one static-bool test per call.
namespace gipc_audit
{
struct SiteTable
{
    std::unordered_map<std::string, long long> counts;
    std::mutex                                 mu;
    ~SiteTable()
    {
        std::vector<std::pair<std::string, long long>> v(counts.begin(), counts.end());
        std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.second > b.second; });
        fprintf(stderr, "[api-audit] %zu sites\n", v.size());
        for(size_t i = 0; i < v.size() && i < 60; ++i)
            fprintf(stderr, "[api-audit] %8lld  %s\n", v[i].second, v[i].first.c_str());
    }
};
inline void note_site(const char* file, int line)
{
    static const bool enabled = []
    {
        const char* v = getenv("STIFF_API_AUDIT");
        return v && atoi(v) != 0;
    }();
    if(!enabled)
        return;
    static SiteTable table;
    char             key[512];
    snprintf(key, sizeof(key), "%s:%d", file, line);
    std::lock_guard<std::mutex> lk(table.mu);
    ++table.counts[key];
}
}  // namespace gipc_audit

inline void cuda_safe_call_(cudaError err, const char* file_name, const int num_line)
{
    gipc_audit::note_site(file_name, num_line);
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
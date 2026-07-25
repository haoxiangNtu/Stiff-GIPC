//
// mlbvh.cu
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#include "mlbvh.cuh"
#include <cub/device/device_radix_sort.cuh>   // [perenv-parallel #2] malloc-free Morton sort
#include <cmath>
#include <cstdlib>   // [ee-lb] getenv/atoi for the launch-variant pick
#include "cuda_tools/cuda_tools.h"
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include <iostream>
#include <fstream>
#include "gpu_eigen_libs.cuh"

// --- Emit-overflow guard (dynamic, never out-of-bounds) ---------------------
// Pair-emit kernels do `buf[atomicAdd(_cpNum,1)] = ...`. If detected pairs exceed
// the allocated buffer this writes OOB and corrupts GPU memory. g_*_cp_cap are the
// allocated logical capacities (set by GIPC::set via set_emit_caps); buffers are
// allocated with +1 slot, and any emit whose index reaches the cap is redirected
// to that single trash slot [cap]. _cpNum still counts the TRUE total so the host
// can detect overflow, grow the buffer and re-run detection (no pairs lost, no OOB).
__device__ int g_dcd_cp_cap = 0x7fffffff;
__device__ int g_ccd_cp_cap = 0x7fffffff;
// [xenv pin] when 1, drop the EE once-only dedup (obj_idx<self_eid) → emit BOTH directions.
// If counts become env-symmetric ⇒ the dedup line is the asymmetry; if still asymmetric ⇒
// the edge-tree candidate enumeration (Morton) is.
__device__ int g_ee_nodedup = 0;
void set_ee_nodedup(int v) { CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_ee_nodedup, &v, sizeof(int))); }
// [xenv pin/fix] when 1, canonicalize each EE edge's endpoint order by POSITION before _dType_EE
// (env-invariant since geometry is bit-identical) → kills the flipped-edge-order asymmetry.
__device__ int g_ee_canon = 0;
void set_ee_canon(int v) { CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_ee_canon, &v, sizeof(int))); }
__device__ __forceinline__ bool _pos_lt(const double3& a, const double3& b)
{ if(a.x != b.x) return a.x < b.x; if(a.y != b.y) return a.y < b.y; return a.z < b.z; }
// [env-det] global→env-local vertex id (mirror across identical envs); FINAL tie-break in canon.
__device__ const int* g_vloc = nullptr;
void set_ee_vloc(const int* p) { CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_vloc, &p, sizeof(int*))); }
// [env-det] TOTAL env-invariant vertex order: position lexicographic, ties broken by env-local id.
__device__ __forceinline__ bool _vless(const double3& a, uint32_t ia, const double3& b, uint32_t ib)
{ if(a.x != b.x) return a.x < b.x; if(a.y != b.y) return a.y < b.y; if(a.z != b.z) return a.z < b.z;
  return g_vloc ? (g_vloc[ia] < g_vloc[ib]) : (ia < ib); }
// [xenv crack] exact clamped segment-segment distance (geometric, order-invariant) for the trace.
__device__ __forceinline__ double _seg_seg_d(double3 p1,double3 q1,double3 p2,double3 q2)
{
    double3 d1=__GEIGEN__::__minus(q1,p1), d2=__GEIGEN__::__minus(q2,p2), r=__GEIGEN__::__minus(p1,p2);
    double a=__GEIGEN__::__v_vec_dot(d1,d1), e=__GEIGEN__::__v_vec_dot(d2,d2), f=__GEIGEN__::__v_vec_dot(d2,r);
    double s,t;
    if(a<=1e-30 && e<=1e-30) return __GEIGEN__::__norm(r);
    if(a<=1e-30){ s=0; t=f/e; t=t<0?0:(t>1?1:t); }
    else { double c=__GEIGEN__::__v_vec_dot(d1,r);
        if(e<=1e-30){ t=0; s=-c/a; s=s<0?0:(s>1?1:s); }
        else { double b=__GEIGEN__::__v_vec_dot(d1,d2); double den=a*e-b*b;
            s = den>1e-30 ? (b*f-c*e)/den : 0; s=s<0?0:(s>1?1:s);
            t=(b*s+f)/e;
            if(t<0){t=0; s=-c/a; s=s<0?0:(s>1?1:s);}
            else if(t>1){t=1; s=(b-c)/a; s=s<0?0:(s>1?1:s);} } }
    double3 c1=__GEIGEN__::__add(p1,__GEIGEN__::__s_vec_multiply(d1,s));
    double3 c2=__GEIGEN__::__add(p2,__GEIGEN__::__s_vec_multiply(d2,t));
    return __GEIGEN__::__norm(__GEIGEN__::__minus(c1,c2));
}
// [xenv pin] when 1, force the near-parallel EE mollifier OFF (add_e=-1) → drops the -obj_idx-2
// global-edge-index encoding. Tests whether the residual barrier-gradient asymmetry is the mollifier.
__device__ int g_ee_nomollify = 0;
void set_ee_nomollify(int v) { CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_ee_nomollify, &v, sizeof(int))); }
// [xenv crack] log near-threshold EE candidate TESTS (full 4 ids = both edges) to localize the
// membership residual: did env1 TEST a candidate env0 emitted? (enumeration vs classification)
__device__ int g_ee_trace = 0;
__device__ int g_max_stack = 0;  // [ovf] max traversal stack depth reached
// [audit-gate] the per-pop atomicMax below is a whole-grid same-address GLOBAL atomic inside the
// hottest traversal loops (selfQuery_* = top-2 frame cost). The report side was already gated on
// STIFF_STACK_DIAG (GIPC.cu) — the probe itself never was. Default OFF.
__device__ int g_bvh_audit = 0;
void set_bvh_audit(int v){ CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_bvh_audit,&v,sizeof(int))); }
void reset_max_stack(){ int z=0; CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_max_stack,&z,sizeof(int))); }
int get_max_stack(){ int v=0; CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&v,g_max_stack,sizeof(int))); return v; }
void set_ee_trace(int v) { CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_ee_trace, &v, sizeof(int))); }
__device__ int g_ee_tgt0 = -1; __device__ int g_ee_tgt1 = -1;
void set_ee_tgt(int a, int b){ CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_ee_tgt0,&a,sizeof(int))); CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_ee_tgt1,&b,sizeof(int))); }
// [env-det] env-LOCAL EE dedup: the once-only ownership `obj < self` uses GLOBAL edge indices, which
// are not env-mirror ⇒ co-located envs keep different directed instances. Compare the two edges by
// their env-local vertex-id key instead (mirror-invariant). g_ee_canon gates; needs g_vloc.
__device__ __forceinline__ uint64_t _edge_lkey(const uint2& e)
{   // sorted (env-local vert id) pair packed; falls back to global ids if g_vloc unset
    uint32_t a = g_vloc ? (uint32_t)g_vloc[e.x] : e.x;
    uint32_t b = g_vloc ? (uint32_t)g_vloc[e.y] : e.y;
    uint32_t lo = a<b?a:b, hi = a<b?b:a;
    return ((uint64_t)lo<<32) | hi;
}
// [env-det] deterministic EE emit gate: decide d<dHat on the ORDER-INVARIANT true segment-segment
// distance (geometric) instead of the order-sensitive dtype-selected sub-distance ⇒ identical-geometry
// envs make the SAME emit decision for near-dHat-threshold contacts (the last bit-identity layer).
__device__ int g_ee_detgate = 0;
void set_ee_detgate(int v){ CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_ee_detgate, &v, sizeof(int))); }
// [env-det BVH] Morton sort for the per-env (active) build. Default = thrust::sort_by_key (unstable:
// equal-Morton ties resolved by index VALUE ⇒ env-asymmetric for co-located near-degenerate geometry).
// STIFF_BVH_ENVDET ⇒ stable_sort_by_key: equal-Morton ties keep the (env-local-canonical) active-list
// order ⇒ identical envs build identical trees ⇒ identical traversal ⇒ identical candidate sets.
// [perenv-parallel #2] iota on an explicit stream (replaces thrust::sequence in the active paths —
// keeps the whole per-env build submission stream-pure).
__global__ void _iota_u32(uint32_t* a, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) a[i] = (uint32_t)i;
}
void lbvh::ensure_sort_scratch(int N)
{
    if(_sort_tmp && N <= _sort_cap) return;
    int cap = N + N / 4 + 64;
    if(_sort_tmp) cudaFree(_sort_tmp);
    if(_mch_alt)  cudaFree(_mch_alt);
    if(_idx_alt)  cudaFree(_idx_alt);
    size_t bytes = 0;   // cub size query (host-only)
    cub::DeviceRadixSort::SortPairs((void*)nullptr, bytes, (const uint64_t*)nullptr, (uint64_t*)nullptr,
                                    (const uint32_t*)nullptr, (uint32_t*)nullptr, cap, 0, 64);
    CUDA_SAFE_CALL(cudaMalloc(&_sort_tmp, bytes));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_mch_alt, (size_t)cap * sizeof(uint64_t)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_idx_alt, (size_t)cap * sizeof(uint32_t)));
    _sort_tmp_bytes = bytes;
    _sort_cap       = cap;
}
// [env-det BVH → perenv-parallel #2] per-env (active) Morton sort. cub radix SortPairs on the
// instance's PRE-ALLOCATED scratch: (a) NO cudaMalloc/cudaFree inside (thrust's internal alloc/free
// are device-wide syncs — they serialized the per-env pool streams, so the expensive per-env detect
// kernels could never overlap); (b) LSD radix is STABLE — bit-identical to the previous
// thrust::stable_sort_by_key (STIFF_BVH_ENVDET path: equal-Morton ties keep the canonical
// active-list order ⇒ identical envs build identical trees). The old non-ENVDET unstable variant is
// subsumed by the stable sort (only reachable via manual flag combos; stable is a determinism
// superset).
static inline void _mc_sort_active(lbvh& b, uint64_t* mc, uint32_t* idx, int N, cudaStream_t stream = 0)
{
    b.ensure_sort_scratch(N);   // no-op when pre-sized (pool slots are sized at allocPerEnvPool)
    size_t bytes = b._sort_tmp_bytes;
    cub::DeviceRadixSort::SortPairs(b._sort_tmp, bytes, mc, b._mch_alt, idx, b._idx_alt,
                                    N, 0, 64, stream);
    CUDA_SAFE_CALL(cudaMemcpyAsync(mc,  b._mch_alt, (size_t)N * sizeof(uint64_t),
                                   cudaMemcpyDeviceToDevice, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(idx, b._idx_alt, (size_t)N * sizeof(uint32_t),
                                   cudaMemcpyDeviceToDevice, stream));
}

#include "linear_system/utils/pcg_capacity_mode.h"  // [C-1]
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

#ifndef STIFF_BVH_STACK_CAP
#define STIFF_BVH_STACK_CAP 2048
#endif
static_assert(STIFF_BVH_STACK_CAP >= 16,
              "STIFF_BVH_STACK_CAP is too small for a useful traversal");

// The production/default capacity keeps the original push instruction exactly.
// Experimental smaller caps are bounds-checked and fail loudly with a CUDA
// trap instead of dropping a subtree and silently violating IPC completeness.
#if STIFF_BVH_STACK_CAP == 2048
#define BVH_STACK_PUSH(node_) (*stack_ptr++ = (node_))
#else
#define BVH_STACK_PUSH(node_)                                                     \
    do                                                                            \
    {                                                                             \
        if(stack_ptr - stack >= STIFF_BVH_STACK_CAP)                              \
            asm volatile("trap;");                                                \
        *stack_ptr++ = (node_);                                                   \
    } while(0)
#endif

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
// [B3 tosymbol-cache] These setters republish the SAME value on every
// buildCP (twice per detection pass) — a measured 6-14 cudaMemcpyToSymbol
// round trips per Newton iteration. Under the single-engine lease exactly one
// engine writes these process-global symbols, so a host-side last-value cache
// is sound (mode changes and buffer growth change the value -> write-through).
// Device-WRITTEN counters (reset_max_stack) are deliberately NOT cached.
void set_ee_nodedup(int v) { static int last = -999; if(v == last) return; CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_ee_nodedup, &v, sizeof(int))); last = v; }
// [xenv pin/fix] when 1, canonicalize each EE edge's endpoint order by POSITION before _dType_EE
// (env-invariant since geometry is bit-identical) → kills the flipped-edge-order asymmetry.
__device__ int g_ee_canon = 0;
void set_ee_canon(int v) { static int last = -999; if(v == last) return; CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_ee_canon, &v, sizeof(int))); last = v; }
__device__ __forceinline__ bool _pos_lt(const double3& a, const double3& b)
{ if(a.x != b.x) return a.x < b.x; if(a.y != b.y) return a.y < b.y; return a.z < b.z; }
// [env-det] global→env-local vertex id (mirror across identical envs); FINAL tie-break in canon.
__device__ const int* g_vloc = nullptr;
void set_ee_vloc(const int* p) { static const int* last = (const int*)-1; if(p == last) return; CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_vloc, &p, sizeof(int*))); last = p; }
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
void set_ee_nomollify(int v) { static int last = -999; if(v == last) return; CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_ee_nomollify, &v, sizeof(int))); last = v; }
// [xenv crack] log near-threshold EE candidate TESTS (full 4 ids = both edges) to localize the
// membership residual: did env1 TEST a candidate env0 emitted? (enumeration vs classification)
__device__ int g_ee_trace = 0;
__device__ int g_max_stack = 0;  // [ovf] max traversal stack depth reached
// [audit-gate] the per-pop atomicMax below is a whole-grid same-address GLOBAL atomic inside the
// hottest traversal loops (selfQuery_* = top-2 frame cost). The report side was already gated on
// STIFF_STACK_DIAG (GIPC.cu) — the probe itself never was. Default OFF.
__device__ int g_bvh_audit = 0;
void set_bvh_audit(int v){ static int last = -999; if(v == last) return; CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_bvh_audit,&v,sizeof(int))); last = v; }
void reset_max_stack(){ int z=0; CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_max_stack,&z,sizeof(int))); }
int get_max_stack(){ int v=0; CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&v,g_max_stack,sizeof(int))); return v; }
int get_bvh_stack_capacity(){ return STIFF_BVH_STACK_CAP; }

// Validation build only: aggregate traversal work without a global atomic in
// the inner loop.  Each participating thread keeps local counters and commits
// four atomics after its traversal.  With the compile definition absent the
// macros erase completely, so this experiment cannot perturb release register
// pressure or timings.
enum BvhAuditFamily
{
    kBvhVfDcd = 0,
    kBvhEeDcd = 1,
    kBvhVfCcd = 2,
    kBvhEeCcd = 3,
};
#ifdef STIFF_BVH_TRAVERSAL_AUDIT_BUILD
__device__ int g_bvh_traversal_audit = 0;
__device__ int g_bvh_pair_work_audit = 0;
__device__ double g_bvh_traversal_margin_scale = 1.0;
__device__ unsigned long long g_bvh_traversal_counters[4][4];
__device__ unsigned long long
    g_bvh_traversal_body_counters[4][kBvhAuditBodyCapacity][4];
// Raw broad-phase leaf pairs, before exact distance/CCD classification.  The
// matrix is canonicalized by body id so it lines up with the conservative
// body-pair displacement states in GIPC's shadow cache audit.
__device__ unsigned long long
    g_bvh_traversal_pair_primitive_counters
        [4][kBvhAuditBodyCapacity][kBvhAuditBodyCapacity];
void set_bvh_traversal_audit(int v)
{
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_bvh_traversal_audit, &v, sizeof(int)));
}
void set_bvh_pair_work_audit(int v)
{
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_bvh_pair_work_audit, &v, sizeof(int)));
}
void set_bvh_traversal_margin_scale(double scale)
{
    static double last = -1.0;
    if(scale == last)
        return;
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_bvh_traversal_margin_scale,
                                     &scale,
                                     sizeof(double)));
    last = scale;
}
void reset_bvh_traversal_audit()
{
    unsigned long long zero[4][4] = {};
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_bvh_traversal_counters, zero, sizeof(zero)));
    unsigned long long body_zero[4][kBvhAuditBodyCapacity][4] = {};
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(
        g_bvh_traversal_body_counters,
        body_zero,
        sizeof(body_zero)));
    static const unsigned long long
        pair_zero[4][kBvhAuditBodyCapacity][kBvhAuditBodyCapacity] = {};
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(
        g_bvh_traversal_pair_primitive_counters,
        pair_zero,
        sizeof(pair_zero)));
}
void get_bvh_traversal_audit(BvhTraversalAudit out[4])
{
    static_assert(sizeof(BvhTraversalAudit) == 4 * sizeof(unsigned long long),
                  "audit host/device layout mismatch");
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(out,
                                       g_bvh_traversal_counters,
                                       4 * sizeof(BvhTraversalAudit)));
}
void get_bvh_traversal_body_audit(
    BvhTraversalAudit out[4][kBvhAuditBodyCapacity])
{
    static_assert(sizeof(BvhTraversalAudit) == 4 * sizeof(unsigned long long),
                  "audit host/device layout mismatch");
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(
        out,
        g_bvh_traversal_body_counters,
        sizeof(g_bvh_traversal_body_counters)));
}
void get_bvh_traversal_pair_primitive_audit(
    unsigned long long out[4][kBvhAuditBodyCapacity]
                               [kBvhAuditBodyCapacity])
{
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(
        out,
        g_bvh_traversal_pair_primitive_counters,
        sizeof(g_bvh_traversal_pair_primitive_counters)));
}
void print_bvh_traversal_audit()
{
    BvhTraversalAudit rows[4] = {};
    get_bvh_traversal_audit(rows);
    static const char* names[4] = {"vf_dcd", "ee_dcd", "vf_ccd", "ee_ccd"};
    for(int i = 0; i < 4; ++i)
        printf("[bvh-audit] family=%s queries=%llu node_pops=%llu "
               "overlapping_children=%llu primitive_tests=%llu\n",
               names[i],
               rows[i].queries,
               rows[i].node_pops,
               rows[i].overlapping_children,
               rows[i].primitive_tests);
}
#define BVH_TRAVERSAL_AUDIT_BEGIN(family_)                                      \
    const bool _bvh_count = g_bvh_traversal_audit != 0;                         \
    unsigned long long _bvh_pops = 0;                                           \
    unsigned long long _bvh_overlaps = 0;                                       \
    unsigned long long _bvh_primitive_tests = 0;                                \
    int _bvh_body = -1;                                                         \
    constexpr int _bvh_family = family_
#define BVH_TRAVERSAL_AUDIT_SET_BODY(body_)                                     \
    do { if(_bvh_count) _bvh_body = (body_); } while(0)
#define BVH_TRAVERSAL_AUDIT_POP()                                                \
    do { if(_bvh_count) ++_bvh_pops; } while(0)
#define BVH_TRAVERSAL_AUDIT_OVERLAP()                                            \
    do { if(_bvh_count) ++_bvh_overlaps; } while(0)
#define BVH_TRAVERSAL_AUDIT_PRIMITIVE()                                          \
    do { if(_bvh_count) ++_bvh_primitive_tests; } while(0)
#define BVH_TRAVERSAL_AUDIT_PRIMITIVE_PAIR(target_body_)                         \
    do                                                                           \
    {                                                                            \
        if(_bvh_count)                                                           \
        {                                                                        \
            ++_bvh_primitive_tests;                                              \
            if(g_bvh_pair_work_audit)                                            \
            {                                                                    \
                const int _bvh_target = (target_body_);                          \
                if(_bvh_body >= 0 && _bvh_body < kBvhAuditBodyCapacity           \
                   && _bvh_target >= 0                                           \
                   && _bvh_target < kBvhAuditBodyCapacity)                       \
                {                                                                \
                    const int _bvh_a =                                              \
                        _bvh_body < _bvh_target ? _bvh_body : _bvh_target;       \
                    const int _bvh_b =                                              \
                        _bvh_body < _bvh_target ? _bvh_target : _bvh_body;       \
                    atomicAdd(                                                   \
                        &g_bvh_traversal_pair_primitive_counters                 \
                            [_bvh_family][_bvh_a][_bvh_b],                       \
                        1ULL);                                                    \
                }                                                                \
            }                                                                    \
        }                                                                        \
    } while(0)
#define BVH_TRAVERSAL_AUDIT_COMMIT()                                             \
    do                                                                           \
    {                                                                            \
        if(_bvh_count)                                                           \
        {                                                                        \
            atomicAdd(&g_bvh_traversal_counters[_bvh_family][0], 1ULL);          \
            atomicAdd(&g_bvh_traversal_counters[_bvh_family][1], _bvh_pops);     \
            atomicAdd(&g_bvh_traversal_counters[_bvh_family][2], _bvh_overlaps); \
            atomicAdd(&g_bvh_traversal_counters[_bvh_family][3],                 \
                      _bvh_primitive_tests);                                     \
            if(_bvh_body >= 0 && _bvh_body < kBvhAuditBodyCapacity)              \
            {                                                                    \
                atomicAdd(&g_bvh_traversal_body_counters[_bvh_family]            \
                                                        [_bvh_body][0],          \
                          1ULL);                                                  \
                atomicAdd(&g_bvh_traversal_body_counters[_bvh_family]            \
                                                        [_bvh_body][1],          \
                          _bvh_pops);                                             \
                atomicAdd(&g_bvh_traversal_body_counters[_bvh_family]            \
                                                        [_bvh_body][2],          \
                          _bvh_overlaps);                                         \
                atomicAdd(&g_bvh_traversal_body_counters[_bvh_family]            \
                                                        [_bvh_body][3],          \
                          _bvh_primitive_tests);                                  \
            }                                                                    \
        }                                                                        \
    } while(0)
#define BVH_TRAVERSAL_MARGIN(gap_) ((gap_) * g_bvh_traversal_margin_scale)
#else
void set_bvh_traversal_audit(int) {}
void set_bvh_pair_work_audit(int) {}
void set_bvh_traversal_margin_scale(double) {}
void reset_bvh_traversal_audit() {}
void get_bvh_traversal_audit(BvhTraversalAudit out[4])
{
    memset(out, 0, 4 * sizeof(BvhTraversalAudit));
}
void get_bvh_traversal_body_audit(
    BvhTraversalAudit out[4][kBvhAuditBodyCapacity])
{
    memset(out, 0, sizeof(BvhTraversalAudit) * 4 * kBvhAuditBodyCapacity);
}
void get_bvh_traversal_pair_primitive_audit(
    unsigned long long out[4][kBvhAuditBodyCapacity]
                               [kBvhAuditBodyCapacity])
{
    memset(out,
           0,
           sizeof(unsigned long long) * 4 * kBvhAuditBodyCapacity
               * kBvhAuditBodyCapacity);
}
void print_bvh_traversal_audit() {}
#define BVH_TRAVERSAL_AUDIT_BEGIN(family_) ((void)0)
#define BVH_TRAVERSAL_AUDIT_SET_BODY(body_) ((void)0)
#define BVH_TRAVERSAL_AUDIT_POP() ((void)0)
#define BVH_TRAVERSAL_AUDIT_OVERLAP() ((void)0)
#define BVH_TRAVERSAL_AUDIT_PRIMITIVE() ((void)0)
#define BVH_TRAVERSAL_AUDIT_PRIMITIVE_PAIR(target_body_) ((void)0)
#define BVH_TRAVERSAL_AUDIT_COMMIT() ((void)0)
#define BVH_TRAVERSAL_MARGIN(gap_) (gap_)
#endif

// Experimental VF-DCD cache state.  The cache owns only raw point/face
// broad-phase candidates; every reuse calls the normal exact PT classifier.
// Body-pair validity is published by the conservative displacement audit.
__device__ const unsigned char* g_bvh_vf_cache_valid = nullptr;
__device__ const int* g_bvh_vf_cache_index = nullptr;
__device__ int g_bvh_vf_cache_body_count = 0;
__device__ int g_bvh_vf_cache_pair_count = 0;
__device__ int2* g_bvh_vf_cache_candidates = nullptr;
__device__ uint32_t* g_bvh_vf_cache_counts = nullptr;
__device__ int g_bvh_vf_cache_segment_capacity = 0;
__device__ int* g_bvh_vf_cache_overflow = nullptr;
__device__ uint32_t* g_bvh_vf_front_nodes = nullptr;
__device__ uint32_t* g_bvh_vf_front_counts = nullptr;
__device__ int g_bvh_vf_front_capacity = 0;
__device__ int* g_bvh_vf_front_overflow = nullptr;
static int h_bvh_vf_cache_pair_count = 0;
static int h_bvh_vf_cache_body_count = 0;
static int h_bvh_vf_front_capacity = 0;
static uint32_t* h_bvh_vf_front_counts = nullptr;
static int* h_bvh_vf_front_overflow = nullptr;

void set_bvh_vf_pair_cache(const unsigned char* pair_valid,
                           const int*           pair_index,
                           int                  body_count,
                           int                  pair_count,
                           int2*                candidates,
                           uint32_t*            counts,
                           int                  segment_capacity,
                           int*                 overflow)
{
    h_bvh_vf_cache_pair_count = pair_count;
    h_bvh_vf_cache_body_count = body_count;
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(
        g_bvh_vf_cache_valid, &pair_valid, sizeof(pair_valid)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(
        g_bvh_vf_cache_index, &pair_index, sizeof(pair_index)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(
        g_bvh_vf_cache_body_count, &body_count, sizeof(body_count)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(
        g_bvh_vf_cache_pair_count, &pair_count, sizeof(pair_count)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(
        g_bvh_vf_cache_candidates, &candidates, sizeof(candidates)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(
        g_bvh_vf_cache_counts, &counts, sizeof(counts)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_bvh_vf_cache_segment_capacity,
                                     &segment_capacity,
                                     sizeof(segment_capacity)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(
        g_bvh_vf_cache_overflow, &overflow, sizeof(overflow)));
}

void set_bvh_vf_pair_front(uint32_t* front_nodes,
                           uint32_t* front_counts,
                           int       front_capacity,
                           int*      front_overflow)
{
    h_bvh_vf_front_capacity = front_capacity;
    h_bvh_vf_front_counts = front_counts;
    h_bvh_vf_front_overflow = front_overflow;
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(
        g_bvh_vf_front_nodes, &front_nodes, sizeof(front_nodes)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(
        g_bvh_vf_front_counts, &front_counts, sizeof(front_counts)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(
        g_bvh_vf_front_capacity, &front_capacity, sizeof(front_capacity)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(
        g_bvh_vf_front_overflow, &front_overflow, sizeof(front_overflow)));
}

__global__ void _buildBvhVfPairFront(const Node* nodes,
                                      const int*  node_body,
                                      int         primitive_count)
{
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    const int node_count = 2 * primitive_count - 1;
    if(node >= node_count)
        return;
    const int body = node_body[node];
    if(body < 0 || body >= g_bvh_vf_cache_body_count)
        return;
    const uint32_t parent = nodes[node].parent_idx;
    if(parent != 0xFFFFFFFFu && node_body[parent] == body)
        return;

    // The traversal front currently seeds existing internal-node traversal
    // code.  A one-primitive body would expose a leaf root; fall back to the
    // exhaustive root rather than risk interpreting leaf payload as children.
    if(nodes[node].element_idx != 0xFFFFFFFFu)
    {
        atomicExch(g_bvh_vf_front_overflow, 1);
        return;
    }
    const uint32_t slot = atomicAdd(&g_bvh_vf_front_counts[body], 1u);
    if(slot < static_cast<uint32_t>(g_bvh_vf_front_capacity))
        g_bvh_vf_front_nodes[
            static_cast<size_t>(body) * g_bvh_vf_front_capacity + slot] =
            static_cast<uint32_t>(node);
    else
        atomicExch(g_bvh_vf_front_overflow, 1);
}

void rebuild_bvh_vf_pair_front(const Node* nodes,
                               const int*  node_body,
                               int         primitive_count,
                               cudaStream_t stream)
{
    if(!getenv("STIFF_BVH_PAIR_CACHE") || h_bvh_vf_cache_body_count <= 0
       || h_bvh_vf_front_capacity <= 0 || !nodes || !node_body
       || primitive_count < 1)
        return;
    CUDA_SAFE_CALL(cudaMemsetAsync(h_bvh_vf_front_counts,
                                   0,
                                   (size_t)h_bvh_vf_cache_body_count
                                       * sizeof(uint32_t),
                                   stream));
    CUDA_SAFE_CALL(cudaMemsetAsync(
        h_bvh_vf_front_overflow, 0, sizeof(int), stream));
    const int node_count = 2 * primitive_count - 1;
    _buildBvhVfPairFront<<<(node_count + 255) / 256, 256, 0, stream>>>(
        nodes, node_body, primitive_count);
}

__global__ void _resetInvalidVfPairCacheCounts()
{
    const int pair = blockIdx.x * blockDim.x + threadIdx.x;
    if(pair >= g_bvh_vf_cache_pair_count)
        return;
    if(!g_bvh_vf_cache_valid[pair])
        g_bvh_vf_cache_counts[pair] = 0;
}

void reset_bvh_vf_pair_cache_counts(cudaStream_t stream)
{
    if(!getenv("STIFF_BVH_PAIR_CACHE") || h_bvh_vf_cache_pair_count <= 0)
        return;
    // Device-validity mode clears an invalid segment's count in the same
    // pair-owned block that decides validity and rebases its references.
    if(getenv("STIFF_BVH_PAIR_CACHE_DEVICE"))
        return;
    // Pointer publication and allocation happen before the first detector.
    // Launch only the armed pair range; the previous body-capacity launch
    // added dozens of empty blocks to every VF query.
    _resetInvalidVfPairCacheCounts<<<
        (h_bvh_vf_cache_pair_count + 255) / 256,
        256,
        0,
        stream>>>();
}

__device__ __forceinline__ int _bvhVfCachePairIndex(int body_a, int body_b)
{
    if(!g_bvh_vf_cache_index || body_a < 0 || body_b < 0
       || body_a >= g_bvh_vf_cache_body_count
       || body_b >= g_bvh_vf_cache_body_count)
        return -1;
    const int a = body_a < body_b ? body_a : body_b;
    const int b = body_a < body_b ? body_b : body_a;
    return g_bvh_vf_cache_index[a * g_bvh_vf_cache_body_count + b];
}

__device__ __forceinline__ bool _bvhVfCacheSkipNode(int query_body,
                                                     int node_body)
{
    const int pair = _bvhVfCachePairIndex(query_body, node_body);
    return pair >= 0 && g_bvh_vf_cache_valid
           && g_bvh_vf_cache_valid[pair] != 0;
}

// Seed traversal directly at the maximal uniform-body subtrees for every
// invalid target pair.  Valid pairs are served by cached-list replay; skipped
// body pairs have no index entry.  Returning false requests the exact legacy
// root traversal (front unavailable/overflow/local-stack bound).
__device__ __forceinline__ bool _bvhVfCacheSeedFront(
    int query_body, uint32_t* stack, uint32_t*& stack_ptr)
{
    if(!g_bvh_vf_front_nodes || !g_bvh_vf_front_counts
       || !g_bvh_vf_front_overflow || *g_bvh_vf_front_overflow
       || !g_bvh_vf_cache_valid || !g_bvh_vf_cache_index)
        return false;
    for(int target_body = 0; target_body < g_bvh_vf_cache_body_count;
        ++target_body)
    {
        const int pair = _bvhVfCachePairIndex(query_body, target_body);
        if(pair < 0 || g_bvh_vf_cache_valid[pair])
            continue;
        const uint32_t count = g_bvh_vf_front_counts[target_body];
        if(count > static_cast<uint32_t>(g_bvh_vf_front_capacity)
           || stack_ptr - stack + count > STIFF_BVH_STACK_CAP)
        {
            stack_ptr = stack;
            return false;
        }
        const size_t begin =
            static_cast<size_t>(target_body) * g_bvh_vf_front_capacity;
        for(uint32_t i = 0; i < count; ++i)
            *stack_ptr++ = g_bvh_vf_front_nodes[begin + i];
    }
    return true;
}

__device__ __forceinline__ void _bvhVfCacheRecord(int query_body,
                                                   int target_body,
                                                   int vertex,
                                                   int face)
{
    const int pair = _bvhVfCachePairIndex(query_body, target_body);
    if(pair < 0 || !g_bvh_vf_cache_valid || g_bvh_vf_cache_valid[pair]
       || !g_bvh_vf_cache_counts || !g_bvh_vf_cache_candidates)
        return;
    const uint32_t slot = atomicAdd(&g_bvh_vf_cache_counts[pair], 1u);
    if(slot < static_cast<uint32_t>(g_bvh_vf_cache_segment_capacity))
        g_bvh_vf_cache_candidates[
            static_cast<size_t>(pair) * g_bvh_vf_cache_segment_capacity
            + slot] = make_int2(vertex, face);
    else if(g_bvh_vf_cache_overflow)
        atomicExch(g_bvh_vf_cache_overflow, 1);
}

void set_ee_trace(int v) { static int last = -999; if(v == last) return; CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_ee_trace, &v, sizeof(int))); last = v; }
__device__ int g_ee_tgt0 = -1; __device__ int g_ee_tgt1 = -1;
void set_ee_tgt(int a, int b){ static int la = -999, lb = -999; if(a == la && b == lb) return; CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_ee_tgt0,&a,sizeof(int))); CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_ee_tgt1,&b,sizeof(int))); la = a; lb = b; }
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
void set_ee_detgate(int v){ static int last = -999; if(v == last) return; CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_ee_detgate, &v, sizeof(int))); last = v; }
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
    ++pcg_buffer_generation();   // [C-1] sort scratch baked in the LS graph moves
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

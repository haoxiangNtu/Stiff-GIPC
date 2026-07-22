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
__device__ __forceinline__ uint32_t _emit_slot(uint32_t* cnt, int cap)
{
    uint32_t i = atomicAdd(cnt, 1u);
    return (i < (uint32_t)cap) ? i : (uint32_t)cap;   // overflow -> trash slot [cap]
}
void set_emit_caps(int dcd_cap, int ccd_cap)
{
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_dcd_cp_cap, &dcd_cap, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_ccd_cp_cap, &ccd_cap, sizeof(int)));
}




__device__ __host__ inline AABB merge(const AABB& lhs, const AABB& rhs) noexcept
{
    AABB merged;
    merged.upper.x = std::max(lhs.upper.x, rhs.upper.x);
    merged.upper.y = std::max(lhs.upper.y, rhs.upper.y);
    merged.upper.z = std::max(lhs.upper.z, rhs.upper.z);
    merged.lower.x = std::min(lhs.lower.x, rhs.lower.x);
    merged.lower.y = std::min(lhs.lower.y, rhs.lower.y);
    merged.lower.z = std::min(lhs.lower.z, rhs.lower.z);
    return merged;
}

__device__ __host__ inline bool overlap(const AABB& lhs, const AABB& rhs, const double& gapL) noexcept
{
    if((rhs.lower.x - lhs.upper.x) >= gapL || (lhs.lower.x - rhs.upper.x) >= gapL)
        return false;
    if((rhs.lower.y - lhs.upper.y) >= gapL || (lhs.lower.y - rhs.upper.y) >= gapL)
        return false;
    if((rhs.lower.z - lhs.upper.z) >= gapL || (lhs.lower.z - rhs.upper.z) >= gapL)
        return false;
    return true;
}

// Check if collision between bodyA and bodyB should be skipped
// according to the collision exclusion matrix.
//
// [multi-FEM-bodyid] Previously mapped body_id == -1 (legacy FEM sentinel)
// to the last matrix row, which aliased all FEM bodies into a single slot.
// Now every body (ABD or FEM) carries its own real body_id and indexes the
// matrix directly.
__device__ inline bool _is_collision_excluded(int bodyA, int bodyB,
                                              const int* _collision_skip_matrix,
                                              int _collision_body_count)
{
    if(_collision_skip_matrix == nullptr || _collision_body_count <= 0)
        return false;
    if(bodyA < 0 || bodyB < 0 || bodyA >= _collision_body_count || bodyB >= _collision_body_count)
        return false;
    return _collision_skip_matrix[bodyA * _collision_body_count + bodyB] != 0;
}

// [multi-FEM-bodyid] Should we run narrow-phase contact / sanity check
// between two vertices/edges/faces with body IDs (bodyA, bodyB)?
// Rules:
//   bodyA != bodyB                       -> YES (different bodies)
//   bodyA == bodyB && is_fem[bodyA]      -> YES (FEM body self-collision)
//   bodyA == bodyB && !is_fem[bodyA]     -> NO (ABD body, no self-collision)
//   bodyA == -1 (unassigned)             -> NO (defensive)
// Replaces the legacy `(A != B) || (A == -1)` pattern that hardcoded
// "all FEM share body_id -1, FEM-self always on" assumption.
__device__ inline bool _should_check_pair(int bodyA, int bodyB,
                                          const int* _body_id_to_is_fem)
{
    if(bodyA != bodyB) return true;
    if(bodyA < 0 || _body_id_to_is_fem == nullptr) return false;
    return _body_id_to_is_fem[bodyA] != 0;
}

__device__ __host__ inline double3 centroid(const AABB& box) noexcept
{
    double3 c;
    c.x = (box.upper.x + box.lower.x) * 0.5;
    c.y = (box.upper.y + box.lower.y) * 0.5;
    c.z = (box.upper.z + box.lower.z) * 0.5;
    return c;
}

__device__ __host__ inline std::uint32_t expand_bits(std::uint32_t v) noexcept
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ __host__ inline std::uint32_t morton_code(double x,
                                                     double y,
                                                     double z,
                                                     double resolution = 1024.0) noexcept
{
    x = std::min(std::max(x * resolution, 0.0), resolution - 1.0);
    y = std::min(std::max(y * resolution, 0.0), resolution - 1.0);
    z = std::min(std::max(z * resolution, 0.0), resolution - 1.0);

    const std::uint32_t xx = expand_bits(static_cast<std::uint32_t>(x));
    const std::uint32_t yy = expand_bits(static_cast<std::uint32_t>(y));
    const std::uint32_t zz = expand_bits(static_cast<std::uint32_t>(z));

    std::uint32_t mchash = ((xx << 2) + (yy << 1) + zz);

    return mchash;
}

__device__ __host__ void AABB::combines(const double& x, const double& y, const double& z)
{
    lower = make_double3(std::min(lower.x, x), std::min(lower.y, y), std::min(lower.z, z));
    upper = make_double3(std::max(upper.x, x), std::max(upper.y, y), std::max(upper.z, z));
}

__device__ __host__ void AABB::combines(const double& x,
                                        const double& y,
                                        const double& z,
                                        const double& xx,
                                        const double& yy,
                                        const double& zz)
{
    lower = make_double3(std::min(lower.x, x), std::min(lower.y, y), std::min(lower.z, z));
    upper =
        make_double3(std::max(upper.x, xx), std::max(upper.y, yy), std::max(upper.z, zz));
}

__host__ __device__ void AABB::combines(const AABB& aabb)
{
    lower = make_double3(std::min(lower.x, aabb.lower.x),
                         std::min(lower.y, aabb.lower.y),
                         std::min(lower.z, aabb.lower.z));
    upper = make_double3(std::max(upper.x, aabb.upper.x),
                         std::max(upper.y, aabb.upper.y),
                         std::max(upper.z, aabb.upper.z));
}

__host__ __device__ double3 AABB::center()
{
    return make_double3((upper.x + lower.x) * 0.5,
                        (upper.y + lower.y) * 0.5,
                        (upper.z + lower.z) * 0.5);
}

__device__ __host__ AABB::AABB()
{
    lower = make_double3(1e32, 1e32, 1e32);
    upper = make_double3(-1e32, -1e32, -1e32);
}

//__device__
//inline int common_upper_bits(const unsigned int lhs, const unsigned int rhs) noexcept
//{
//    return ::__clz(lhs ^ rhs);
//}
__device__ inline int common_upper_bits(const unsigned long long int lhs,
                                        const unsigned long long int rhs) noexcept
{
    return ::__clzll(lhs ^ rhs);
}


__device__ inline uint2 determine_range(const uint64_t*    node_code,
                                        const unsigned int num_leaves,
                                        unsigned int       idx)
{
    if(idx == 0)
    {
        return make_uint2(0, num_leaves - 1);
    }

    // determine direction of the range
    const uint64_t self_code = node_code[idx];
    const int      L_delta   = common_upper_bits(self_code, node_code[idx - 1]);
    const int      R_delta   = common_upper_bits(self_code, node_code[idx + 1]);
    const int      d         = (R_delta > L_delta) ? 1 : -1;

    // Compute upper bound for the length of the range

    const int delta_min = std::min(L_delta, R_delta);
    int       l_max     = 2;
    int       delta     = -1;
    int       i_tmp     = idx + d * l_max;
    if(0 <= i_tmp && i_tmp < num_leaves)
    {
        delta = common_upper_bits(self_code, node_code[i_tmp]);
    }
    while(delta > delta_min)
    {
        l_max <<= 1;
        i_tmp = idx + d * l_max;
        delta = -1;
        if(0 <= i_tmp && i_tmp < num_leaves)
        {
            delta = common_upper_bits(self_code, node_code[i_tmp]);
        }
    }

    // Find the other end by binary search
    int l = 0;
    int t = l_max >> 1;
    while(t > 0)
    {
        i_tmp = idx + (l + t) * d;
        delta = -1;
        if(0 <= i_tmp && i_tmp < num_leaves)
        {
            delta = common_upper_bits(self_code, node_code[i_tmp]);
        }
        if(delta > delta_min)
        {
            l += t;
        }
        t >>= 1;
    }
    unsigned int jdx = idx + l * d;
    if(d < 0)
    {
        unsigned int temp_jdx = jdx;
        jdx                   = idx;
        idx                   = temp_jdx;
    }
    return make_uint2(idx, jdx);
}

__device__ inline unsigned int find_split(const uint64_t*    node_code,
                                          const unsigned int num_leaves,
                                          const unsigned int first,
                                          const unsigned int last) noexcept
{
    const uint64_t first_code = node_code[first];
    const uint64_t last_code  = node_code[last];
    if(first_code == last_code)
    {
        return (first + last) >> 1;
    }
    const int delta_node = common_upper_bits(first_code, last_code);

    // binary search...
    int split  = first;
    int stride = last - first;
    do
    {
        stride           = (stride + 1) >> 1;
        const int middle = split + stride;
        if(middle < last)
        {
            const int delta = common_upper_bits(first_code, node_code[middle]);
            if(delta > delta_node)
            {
                split = middle;
            }
        }
    } while(stride > 1);

    return split;
}

__device__ void _d_PP(const double3& v0, const double3& v1, double& d)
{
    d = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v0, v1));
}

__device__ void _d_PT(const double3& v0, const double3& v1, const double3& v2, const double3& v3, double& d)
{
    double3 b    = __GEIGEN__::__v_vec_cross(__GEIGEN__::__minus(v2, v1),
                                          __GEIGEN__::__minus(v3, v1));
    double3 test = __GEIGEN__::__minus(v0, v1);
    double aTb = __GEIGEN__::__v_vec_dot(__GEIGEN__::__minus(v0, v1), b);  //(v0 - v1).dot(b);
    //printf("%f   %f   %f          %f   %f   %f   %f\n", b.x, b.y, b.z, test.x, test.y, test.z, aTb);
    d = aTb * aTb / __GEIGEN__::__squaredNorm(b);
}

__device__ void _d_PE(const double3& v0, const double3& v1, const double3& v2, double& d)
{
    d = __GEIGEN__::__squaredNorm(__GEIGEN__::__v_vec_cross(
            __GEIGEN__::__minus(v1, v0), __GEIGEN__::__minus(v2, v0)))
        / __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v2, v1));
}

__device__ void _d_EE(const double3& v0, const double3& v1, const double3& v2, const double3& v3, double& d)
{
    double3 b = __GEIGEN__::__v_vec_cross(__GEIGEN__::__minus(v1, v0),
                                          __GEIGEN__::__minus(v3, v2));  //(v1 - v0).cross(v3 - v2);
    double aTb = __GEIGEN__::__v_vec_dot(__GEIGEN__::__minus(v2, v0), b);  //(v2 - v0).dot(b);
    d = aTb * aTb / __GEIGEN__::__squaredNorm(b);
}


__device__ void _d_EEParallel(const double3& v0,
                              const double3& v1,
                              const double3& v2,
                              const double3& v3,
                              double&        d)
{
    double3 b = __GEIGEN__::__v_vec_cross(
        __GEIGEN__::__v_vec_cross(__GEIGEN__::__minus(v1, v0), __GEIGEN__::__minus(v2, v0)),
        __GEIGEN__::__minus(v1, v0));
    double aTb = __GEIGEN__::__v_vec_dot(__GEIGEN__::__minus(v2, v0), b);  //(v2 - v0).dot(b);
    d = aTb * aTb / __GEIGEN__::__squaredNorm(b);
}

__device__ double _compute_epx(const double3& v0, const double3& v1, const double3& v2, const double3& v3)
{
    return 1e-3 * __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v0, v1))
           * __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v2, v3));
}

__device__ double _compute_epx_cp(const double3& v0,
                                  const double3& v1,
                                  const double3& v2,
                                  const double3& v3)
{
    return 1e-3 * __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v0, v1))
           * __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v2, v3));
}

__device__ int _dType_PT(const double3& v0, const double3& v1, const double3& v2, const double3& v3)
{
    double3 basis0 = __GEIGEN__::__minus(v2, v1);
    double3 basis1 = __GEIGEN__::__minus(v3, v1);
    double3 basis2 = __GEIGEN__::__minus(v0, v1);

    const double3 nVec = __GEIGEN__::__v_vec_cross(basis0, basis1);

    basis1 = __GEIGEN__::__v_vec_cross(basis0, nVec);
    __GEIGEN__::Matrix3x3d D, D1, D2;

    __GEIGEN__::__set_Mat_val(D,
                              basis0.x,
                              basis1.x,
                              nVec.x,
                              basis0.y,
                              basis1.y,
                              nVec.y,
                              basis0.z,
                              basis1.z,
                              nVec.z);
    __GEIGEN__::__set_Mat_val(D1,
                              basis2.x,
                              basis1.x,
                              nVec.x,
                              basis2.y,
                              basis1.y,
                              nVec.y,
                              basis2.z,
                              basis1.z,
                              nVec.z);
    __GEIGEN__::__set_Mat_val(D2,
                              basis0.x,
                              basis2.x,
                              nVec.x,
                              basis0.y,
                              basis2.y,
                              nVec.y,
                              basis0.z,
                              basis2.z,
                              nVec.z);

    double2 param[3];
    param[0].x = __GEIGEN__::__Determiant(D1) / __GEIGEN__::__Determiant(D);
    param[0].y = __GEIGEN__::__Determiant(D2) / __GEIGEN__::__Determiant(D);

    if(param[0].x > 0 && param[0].x < 1 && param[0].y >= 0)
    {
        return 3;  // PE v1v2
    }
    else
    {
        basis0 = __GEIGEN__::__minus(v3, v2);
        basis1 = __GEIGEN__::__v_vec_cross(basis0, nVec);
        basis2 = __GEIGEN__::__minus(v0, v2);

        __GEIGEN__::__set_Mat_val(D,
                                  basis0.x,
                                  basis1.x,
                                  nVec.x,
                                  basis0.y,
                                  basis1.y,
                                  nVec.y,
                                  basis0.z,
                                  basis1.z,
                                  nVec.z);
        __GEIGEN__::__set_Mat_val(D1,
                                  basis2.x,
                                  basis1.x,
                                  nVec.x,
                                  basis2.y,
                                  basis1.y,
                                  nVec.y,
                                  basis2.z,
                                  basis1.z,
                                  nVec.z);
        __GEIGEN__::__set_Mat_val(D2,
                                  basis0.x,
                                  basis2.x,
                                  nVec.x,
                                  basis0.y,
                                  basis2.y,
                                  nVec.y,
                                  basis0.z,
                                  basis2.z,
                                  nVec.z);

        param[1].x = __GEIGEN__::__Determiant(D1) / __GEIGEN__::__Determiant(D);
        param[1].y = __GEIGEN__::__Determiant(D2) / __GEIGEN__::__Determiant(D);

        if(param[1].x > 0.0 && param[1].x < 1.0 && param[1].y >= 0.0)
        {
            return 4;  // PE v2v3
        }
        else
        {
            basis0 = __GEIGEN__::__minus(v1, v3);
            basis1 = __GEIGEN__::__v_vec_cross(basis0, nVec);
            basis2 = __GEIGEN__::__minus(v0, v3);

            __GEIGEN__::__set_Mat_val(D,
                                      basis0.x,
                                      basis1.x,
                                      nVec.x,
                                      basis0.y,
                                      basis1.y,
                                      nVec.y,
                                      basis0.z,
                                      basis1.z,
                                      nVec.z);
            __GEIGEN__::__set_Mat_val(D1,
                                      basis2.x,
                                      basis1.x,
                                      nVec.x,
                                      basis2.y,
                                      basis1.y,
                                      nVec.y,
                                      basis2.z,
                                      basis1.z,
                                      nVec.z);
            __GEIGEN__::__set_Mat_val(D2,
                                      basis0.x,
                                      basis2.x,
                                      nVec.x,
                                      basis0.y,
                                      basis2.y,
                                      nVec.y,
                                      basis0.z,
                                      basis2.z,
                                      nVec.z);

            param[2].x = __GEIGEN__::__Determiant(D1) / __GEIGEN__::__Determiant(D);
            param[2].y = __GEIGEN__::__Determiant(D2) / __GEIGEN__::__Determiant(D);

            if(param[2].x > 0.0 && param[2].x < 1.0 && param[2].y >= 0.0)
            {
                return 5;  // PE v3v1
            }
            else
            {
                if(param[0].x <= 0.0 && param[2].x >= 1.0)
                {
                    return 0;  // PP v1
                }
                else if(param[1].x <= 0.0 && param[0].x >= 1.0)
                {
                    return 1;  // PP v2
                }
                else if(param[2].x <= 0.0 && param[1].x >= 1.0)
                {
                    return 2;  // PP v3
                }
                else
                {
                    return 6;  // PT
                }
            }
        }
    }
}

__device__ int _dType_EE(const double3& v0, const double3& v1, const double3& v2, const double3& v3)
{
    double3 u = __GEIGEN__::__minus(v1, v0);
    double3 v = __GEIGEN__::__minus(v3, v2);
    double3 w = __GEIGEN__::__minus(v0, v2);

    double a = __GEIGEN__::__squaredNorm(u);
    double b = __GEIGEN__::__v_vec_dot(u, v);
    double c = __GEIGEN__::__squaredNorm(v);
    double d = __GEIGEN__::__v_vec_dot(u, w);
    double e = __GEIGEN__::__v_vec_dot(v, w);

    double D  = a * c - b * b;  // always >= 0
    double tD = D;              // tc = tN / tD, default tD = D >= 0
    double sN, tN;
    int    defaultCase = 8;
    sN                 = (b * e - c * d);
    if(sN <= 0.0)
    {  // sc < 0 => the s=0 edge is visible
        tN          = e;
        tD          = c;
        defaultCase = 2;
    }
    else if(sN >= D)
    {  // sc > 1  => the s=1 edge is visible
        tN          = e + b;
        tD          = c;
        defaultCase = 5;
    }
    else
    {
        tN = (a * e - b * d);
        if(tN > 0.0 && tN < tD
           && (__GEIGEN__::__v_vec_dot(w, __GEIGEN__::__v_vec_cross(u, v)) == 0.0
               || __GEIGEN__::__squaredNorm(__GEIGEN__::__v_vec_cross(u, v)) < 1.0e-20 * a * c))
        {
            if(sN < D / 2)
            {
                tN          = e;
                tD          = c;
                defaultCase = 2;
            }
            else
            {
                tN          = e + b;
                tD          = c;
                defaultCase = 5;
            }
        }
    }

    if(tN <= 0.0)
    {
        if(-d <= 0.0)
        {
            return 0;
        }
        else if(-d >= a)
        {
            return 3;
        }
        else
        {
            return 6;
        }
    }
    else if(tN >= tD)
    {
        if((-d + b) <= 0.0)
        {
            return 1;
        }
        else if((-d + b) >= a)
        {
            return 4;
        }
        else
        {
            return 7;
        }
    }

    return defaultCase;
}


__device__ inline bool _checkPTintersection(const double3*  _vertexes,
                                            const uint32_t& id0,
                                            const uint32_t& id1,
                                            const uint32_t& id2,
                                            const uint32_t& id3,
                                            const double&   dHat,
                                            uint32_t*       _cpNum,
                                            int*            _mInx,
                                            int4*           _collisionPair,
                                            int4* _ccd_collisionPair) noexcept
{
    double3 v0 = _vertexes[id0];
    double3 v1 = _vertexes[id1];
    double3 v2 = _vertexes[id2];
    double3 v3 = _vertexes[id3];

    int dtype = _dType_PT(v0, v1, v2, v3);

    double d = 100;
    switch(dtype)
    {
        case 0: {
            _d_PP(v0, v1, d);
            if(d < dHat)
            {
                //printf("%d   %d   %d   %d   %d   %f\n", dtype, idx, _faces[obj_idx].x, _faces[obj_idx].y, _faces[obj_idx].z, d);
                int cdp_idx = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                _ccd_collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, id3);
                _collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, -1, -1);
                _mInx[cdp_idx]          = atomicAdd(_cpNum + 2, 1);
            }
            break;
        }

        case 1: {
            _d_PP(v0, v2, d);
            if(d < dHat)
            {
                //printf("%d   %d   %d   %d   %d   %f\n", dtype, idx, _faces[obj_idx].x, _faces[obj_idx].y, _faces[obj_idx].z, d);
                int cdp_idx = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                _ccd_collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, id3);
                _collisionPair[cdp_idx] = make_int4(-id0 - 1, id2, -1, -1);
                _mInx[cdp_idx]          = atomicAdd(_cpNum + 2, 1);
            }
            break;
        }

        case 2: {
            _d_PP(v0, v3, d);
            if(d < dHat)
            {
                //printf("%d   %d   %d   %d   %d   %f\n", dtype, idx, _faces[obj_idx].x, _faces[obj_idx].y, _faces[obj_idx].z, d);
                int cdp_idx = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                _ccd_collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, id3);
                _collisionPair[cdp_idx] = make_int4(-id0 - 1, id3, -1, -1);
                _mInx[cdp_idx]          = atomicAdd(_cpNum + 2, 1);
            }
            break;
        }

        case 3: {
            _d_PE(v0, v1, v2, d);
            if(d < dHat)
            {
                //printf("%d   %d   %d   %d   %d   %f\n", dtype, idx, _faces[obj_idx].x, _faces[obj_idx].y, _faces[obj_idx].z, d);
                int cdp_idx = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                _ccd_collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, id3);
                _collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, -1);
                _mInx[cdp_idx]          = atomicAdd(_cpNum + 3, 1);
            }
            break;
        }

        case 4: {
            _d_PE(v0, v2, v3, d);
            if(d < dHat)
            {
                //printf("%d   %d   %d   %d   %d   %f\n", dtype, idx, _faces[obj_idx].x, _faces[obj_idx].y, _faces[obj_idx].z, d);
                int cdp_idx = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                _ccd_collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, id3);
                _collisionPair[cdp_idx] = make_int4(-id0 - 1, id2, id3, -1);
                _mInx[cdp_idx]          = atomicAdd(_cpNum + 3, 1);
            }
            break;
        }

        case 5: {
            _d_PE(v0, v3, v1, d);
            if(d < dHat)
            {
                //printf("%d   %d   %d   %d   %d   %f\n", dtype, idx, _faces[obj_idx].x, _faces[obj_idx].y, _faces[obj_idx].z, d);
                int cdp_idx = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                _ccd_collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, id3);
                _collisionPair[cdp_idx] = make_int4(-id0 - 1, id3, id1, -1);
                _mInx[cdp_idx]          = atomicAdd(_cpNum + 3, 1);
            }
            break;
        }

        case 6: {
            _d_PT(v0, v1, v2, v3, d);
            if(d < dHat)
            {
                //printf("%d   %d   %d   %d   %d   %f\n", dtype, idx, _faces[obj_idx].x, _faces[obj_idx].y, _faces[obj_idx].z, d);
                int cdp_idx = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                _ccd_collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, id3);
                _collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, id3);
                //printf("ccbcbcbcbbcbcbbcbcb  %d  %d  %d  %d\n", -id0 - 1, id1, id2, id3);
                _mInx[cdp_idx] = atomicAdd(_cpNum + 4, 1);
            }
            break;
        }

        default:
            break;
    }
}

__device__ inline bool _checkPTintersection_fullCCD(const double3*  _vertexes,
                                                    const uint32_t& id0,
                                                    const uint32_t& id1,
                                                    const uint32_t& id2,
                                                    const uint32_t& id3,
                                                    const double&   dHat,
                                                    uint32_t*       _cpNum,
                                                    int4* _ccd_collisionPair) noexcept
{
    double3 v0 = _vertexes[id0];
    double3 v1 = _vertexes[id1];
    double3 v2 = _vertexes[id2];
    double3 v3 = _vertexes[id3];

    int dtype = _dType_PT(v0, v1, v2, v3);

    double3 basis0 = __GEIGEN__::__minus(v2, v1);
    double3 basis1 = __GEIGEN__::__minus(v3, v1);
    double3 basis2 = __GEIGEN__::__minus(v0, v1);

    const double3 nVec = __GEIGEN__::__v_vec_cross(basis0, basis1);

    double sign = __GEIGEN__::__v_vec_dot(nVec, basis2);

    if(dtype == 6 && (sign < 0))
    {
        return;
    }

    _ccd_collisionPair[_emit_slot(_cpNum, g_ccd_cp_cap)] = make_int4(-id0 - 1, id1, id2, id3);
}

__device__ inline bool _checkEEintersection(const double3*  _vertexes,
                                            const double3*  _rest_vertexes,
                                            uint32_t        id0,
                                            uint32_t        id1,
                                            uint32_t        id2,
                                            uint32_t        id3,
                                            const uint32_t& obj_idx,
                                            const double&   dHat,
                                            uint32_t*       _cpNum,
                                            int*            MatIndex,
                                            int4*           _collisionPair,
                                            int4*           _ccd_collisionPair,
                                            int             edgeNum) noexcept
{
    double3 v0 = _vertexes[id0];
    double3 v1 = _vertexes[id1];
    double3 v2 = _vertexes[id2];
    double3 v3 = _vertexes[id3];
    if(g_ee_trace) {
        double ssd = _seg_seg_d(v0,v1,v2,v3);
        if(ssd < 1.5*sqrt(dHat)) printf("EE %u %u %u %u %.17e\n", id0,id1,id2,id3, ssd);
    }

    // [xenv pin/fix] canonicalize each edge's endpoint order by POSITION (env-invariant) so
    // _dType_EE is order-independent → identical envs classify identically.
    if(g_ee_canon)
    {
        // (1) internal endpoint order within each edge — TOTAL order (position, env-local id)
        if(_vless(v1, id1, v0, id0)) { double3 t=v0; v0=v1; v1=t; uint32_t s=id0; id0=id1; id1=s; }
        if(_vless(v3, id3, v2, id2)) { double3 t=v2; v2=v3; v3=t; uint32_t s=id2; id2=id3; id3=s; }
        // (2) edge-pair order (which edge is self vs obj) — by smaller endpoint, TOTAL order
        if(_vless(v2, id2, v0, id0))
        {
            double3 t0=v0,t1=v1; v0=v2; v1=v3; v2=t0; v3=t1;
            uint32_t s0=id0,s1=id1; id0=id2; id1=id3; id2=s0; id3=s1;
        }
    }

    int    dtype  = _dType_EE(v0, v1, v2, v3);
    if(g_ee_trace && (id0==(uint32_t)g_ee_tgt0||id1==(uint32_t)g_ee_tgt0||id2==(uint32_t)g_ee_tgt0||id3==(uint32_t)g_ee_tgt0
                    ||id0==(uint32_t)g_ee_tgt1||id1==(uint32_t)g_ee_tgt1||id2==(uint32_t)g_ee_tgt1||id3==(uint32_t)g_ee_tgt1))
        printf("DT ids %u %u %u %u dtype %d pos %.15e %.15e %.15e | %.15e %.15e %.15e | %.15e %.15e %.15e | %.15e %.15e %.15e\n",
               id0,id1,id2,id3,dtype, v0.x,v0.y,v0.z, v1.x,v1.y,v1.z, v2.x,v2.y,v2.z, v3.x,v3.y,v3.z);
    // [env-det] order-invariant true seg-seg squared distance for the deterministic emit gate.
    double dsg2_ = 0.0;
    if(g_ee_detgate) { double sd_ = _seg_seg_d(v0,v1,v2,v3); dsg2_ = sd_*sd_; }
    int    add_e  = -1;
    double d      = 100.0;
    bool   smooth = false;
    switch(dtype)
    {
        case 0: {
            _d_PP(v0, v2, d);
            if((g_ee_detgate ? dsg2_ : d) < dHat)
            {

                double eeSqureNCross = __GEIGEN__::__squaredNorm(__GEIGEN__::__v_vec_cross(
                    __GEIGEN__::__minus(v0, v1), __GEIGEN__::__minus(v2, v3))) /* / __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v0, v1))*/;
                double eps_x = _compute_epx_cp(_rest_vertexes[id0],
                                               _rest_vertexes[id1],
                                               _rest_vertexes[id2],
                                               _rest_vertexes[id3]);
                add_e        = g_ee_nomollify ? -1 : ((eeSqureNCross < eps_x) ? -obj_idx - 2 : -1);

                if(add_e <= -2)
                {
                    int cdp_idx                 = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    if(smooth)
                    {
                        _collisionPair[cdp_idx] =
                            make_int4(-id0 - 1, -id2 - 1, -id1 - 1, -id3 - 1);
                        MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);

                        break;
                    }
                    _collisionPair[cdp_idx] = make_int4(-id0 - 1, id2, -1, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 2, 1);
                }
                else
                {
                    int cdp_idx                 = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    _collisionPair[cdp_idx] = make_int4(-id0 - 1, id2, -1, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 2, 1);
                }
            }
            break;
        }

        case 1: {
            _d_PP(v0, v3, d);
            if((g_ee_detgate ? dsg2_ : d) < dHat)
            {

                double eeSqureNCross = __GEIGEN__::__squaredNorm(__GEIGEN__::__v_vec_cross(
                    __GEIGEN__::__minus(v0, v1), __GEIGEN__::__minus(v2, v3))) /* / __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v0, v1))*/;
                double eps_x = _compute_epx_cp(_rest_vertexes[id0],
                                               _rest_vertexes[id1],
                                               _rest_vertexes[id2],
                                               _rest_vertexes[id3]);
                add_e        = g_ee_nomollify ? -1 : ((eeSqureNCross < eps_x) ? -obj_idx - 2 : -1);

                if(add_e <= -2)
                {
                    int cdp_idx                 = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    if(smooth)
                    {
                        _collisionPair[cdp_idx] =
                            make_int4(-id0 - 1, -id3 - 1, -id1 - 1, -id2 - 1);
                        MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
                        break;
                    }
                    _collisionPair[cdp_idx] = make_int4(-id0 - 1, id3, -1, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 2, 1);
                }
                else
                {
                    int cdp_idx                 = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    _collisionPair[cdp_idx] = make_int4(-id0 - 1, id3, -1, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 2, 1);
                }
            }
            break;
        }

        case 2: {
            _d_PE(v0, v2, v3, d);
            if((g_ee_detgate ? dsg2_ : d) < dHat)
            {

                double eeSqureNCross = __GEIGEN__::__squaredNorm(__GEIGEN__::__v_vec_cross(
                    __GEIGEN__::__minus(v0, v1), __GEIGEN__::__minus(v2, v3))) /* / __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v0, v1))*/;
                double eps_x = _compute_epx_cp(_rest_vertexes[id0],
                                               _rest_vertexes[id1],
                                               _rest_vertexes[id2],
                                               _rest_vertexes[id3]);
                add_e        = g_ee_nomollify ? -1 : ((eeSqureNCross < eps_x) ? -obj_idx - 2 : -1);


                if(add_e <= -2)
                {
                    int cdp_idx                 = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    if(smooth)
                    {
                        _collisionPair[cdp_idx] =
                            make_int4(-id0 - 1, -id2 - 1, id3, -id1 - 1);
                        MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
                        break;
                    }
                    _collisionPair[cdp_idx] = make_int4(-id0 - 1, id2, id3, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 3, 1);
                }
                else
                {
                    int cdp_idx                 = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    _collisionPair[cdp_idx] = make_int4(-id0 - 1, id2, id3, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 3, 1);
                }
            }
            break;
        }

        case 3: {
            _d_PP(v1, v2, d);
            if((g_ee_detgate ? dsg2_ : d) < dHat)
            {

                double eeSqureNCross = __GEIGEN__::__squaredNorm(__GEIGEN__::__v_vec_cross(
                    __GEIGEN__::__minus(v0, v1), __GEIGEN__::__minus(v2, v3))) /* / __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v0, v1))*/;
                double eps_x = _compute_epx_cp(_rest_vertexes[id0],
                                               _rest_vertexes[id1],
                                               _rest_vertexes[id2],
                                               _rest_vertexes[id3]);
                add_e        = g_ee_nomollify ? -1 : ((eeSqureNCross < eps_x) ? -obj_idx - 2 : -1);

                if(add_e <= -2)
                {
                    int cdp_idx                 = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    if(smooth)
                    {
                        _collisionPair[cdp_idx] =
                            make_int4(-id1 - 1, -id2 - 1, -id0 - 1, -id3 - 1);
                        MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
                        break;
                    }
                    _collisionPair[cdp_idx] = make_int4(-id1 - 1, id2, -1, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 2, 1);
                }
                else
                {
                    int cdp_idx                 = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    _collisionPair[cdp_idx] = make_int4(-id1 - 1, id2, -1, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 2, 1);
                }
            }
            break;
        }

        case 4: {
            _d_PP(v1, v3, d);
            if((g_ee_detgate ? dsg2_ : d) < dHat)
            {

                double eeSqureNCross = __GEIGEN__::__squaredNorm(__GEIGEN__::__v_vec_cross(
                    __GEIGEN__::__minus(v0, v1), __GEIGEN__::__minus(v2, v3))) /* / __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v0, v1))*/;
                double eps_x = _compute_epx_cp(_rest_vertexes[id0],
                                               _rest_vertexes[id1],
                                               _rest_vertexes[id2],
                                               _rest_vertexes[id3]);
                add_e        = g_ee_nomollify ? -1 : ((eeSqureNCross < eps_x) ? -obj_idx - 2 : -1);

                if(add_e <= -2)
                {
                    int cdp_idx                 = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    if(smooth)
                    {
                        _collisionPair[cdp_idx] =
                            make_int4(-id1 - 1, -id3 - 1, -id0 - 1, -id2 - 1);
                        MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
                        break;
                    }
                    _collisionPair[cdp_idx] = make_int4(-id1 - 1, id3, -1, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 2, 1);
                }
                else
                {
                    int cdp_idx                 = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    _collisionPair[cdp_idx] = make_int4(-id1 - 1, id3, -1, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 2, 1);
                }
            }
            break;
        }

        case 5: {
            _d_PE(v1, v2, v3, d);
            if((g_ee_detgate ? dsg2_ : d) < dHat)
            {

                double eeSqureNCross = __GEIGEN__::__squaredNorm(__GEIGEN__::__v_vec_cross(
                    __GEIGEN__::__minus(v0, v1), __GEIGEN__::__minus(v2, v3))) /* / __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v0, v1))*/;
                double eps_x = _compute_epx_cp(_rest_vertexes[id0],
                                               _rest_vertexes[id1],
                                               _rest_vertexes[id2],
                                               _rest_vertexes[id3]);
                add_e        = g_ee_nomollify ? -1 : ((eeSqureNCross < eps_x) ? -obj_idx - 2 : -1);

                if(add_e <= -2)
                {
                    int cdp_idx                 = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    if(smooth)
                    {
                        _collisionPair[cdp_idx] =
                            make_int4(-id1 - 1, -id2 - 1, id3, -id0 - 1);
                        MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
                        break;
                    }
                    _collisionPair[cdp_idx] = make_int4(-id1 - 1, id2, id3, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 3, 1);
                }
                else
                {
                    int cdp_idx                 = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    _collisionPair[cdp_idx] = make_int4(-id1 - 1, id2, id3, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 3, 1);
                }
            }
            break;
        }

        case 6: {
            _d_PE(v2, v0, v1, d);
            if((g_ee_detgate ? dsg2_ : d) < dHat)
            {

                double eeSqureNCross = __GEIGEN__::__squaredNorm(__GEIGEN__::__v_vec_cross(
                    __GEIGEN__::__minus(v2, v3), __GEIGEN__::__minus(v0, v1))) /* / __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v2, v3))*/;
                double eps_x = _compute_epx_cp(_rest_vertexes[id2],
                                               _rest_vertexes[id3],
                                               _rest_vertexes[id0],
                                               _rest_vertexes[id1]);
                add_e        = g_ee_nomollify ? -1 : ((eeSqureNCross < eps_x) ? -obj_idx - 2 : -1);


                if(add_e <= -2)
                {
                    int cdp_idx                 = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    if(smooth)
                    {
                        _collisionPair[cdp_idx] =
                            make_int4(-id2 - 1, -id0 - 1, id1, -id3 - 1);
                        MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
                        break;
                    }
                    _collisionPair[cdp_idx] = make_int4(-id2 - 1, id0, id1, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 3, 1);
                }
                else
                {
                    int cdp_idx                 = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    _collisionPair[cdp_idx] = make_int4(-id2 - 1, id0, id1, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 3, 1);
                }
            }
            break;
        }

        case 7: {
            _d_PE(v3, v0, v1, d);
            if((g_ee_detgate ? dsg2_ : d) < dHat)
            {

                double eeSqureNCross = __GEIGEN__::__squaredNorm(__GEIGEN__::__v_vec_cross(
                    __GEIGEN__::__minus(v2, v3), __GEIGEN__::__minus(v0, v1))) /* / __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v2, v3))*/;
                double eps_x = _compute_epx_cp(_rest_vertexes[id2],
                                               _rest_vertexes[id3],
                                               _rest_vertexes[id0],
                                               _rest_vertexes[id1]);
                add_e        = g_ee_nomollify ? -1 : ((eeSqureNCross < eps_x) ? -obj_idx - 2 : -1);


                if(add_e <= -2)
                {
                    int cdp_idx                 = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    if(smooth)
                    {
                        _collisionPair[cdp_idx] =
                            make_int4(-id3 - 1, -id0 - 1, id1, -id2 - 1);
                        MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
                        break;
                    }
                    _collisionPair[cdp_idx] = make_int4(-id3 - 1, id0, id1, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 3, 1);
                }
                else
                {
                    int cdp_idx                 = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    _collisionPair[cdp_idx] = make_int4(-id3 - 1, id0, id1, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 3, 1);
                }
            }
            break;
        }

        case 8: {
            _d_EE(v0, v1, v2, v3, d);

            double eeSqureNCross = __GEIGEN__::__squaredNorm(__GEIGEN__::__v_vec_cross(
                __GEIGEN__::__minus(v0, v1), __GEIGEN__::__minus(v2, v3))) /* / __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v0, v1))*/;
            double eps_x = _compute_epx_cp(_rest_vertexes[id0],
                                           _rest_vertexes[id1],
                                           _rest_vertexes[id2],
                                           _rest_vertexes[id3]);
            add_e        = g_ee_nomollify ? -1 : ((eeSqureNCross < eps_x) ? -obj_idx - 2 : -1);

            if((g_ee_detgate ? dsg2_ : d) < dHat)
            {
                if(add_e <= -2)
                {
                    //printf("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\nxxxxxxxxxxx\n");
                    int cdp_idx                 = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                    MatIndex[cdp_idx]           = atomicAdd(_cpNum + 4, 1);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    if(smooth)
                    {
                        _collisionPair[cdp_idx] = make_int4(id0, id1, id2, -id3 - 1);
                        break;
                    }
                    _collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                }
                else
                {

                    int cdp_idx                 = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    _collisionPair[cdp_idx]     = make_int4(id0, id1, id2, id3);
                    MatIndex[cdp_idx]           = atomicAdd(_cpNum + 4, 1);
                }
            }
            break;
        }

        default:
            break;
    }
}

__global__ void _reduct_max_box(AABB* _leafBoxes, int number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ AABB tep[];

    if(idx >= number)
        return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    AABB temp = _leafBoxes[idx];

    __threadfence();

    double xmin = temp.lower.x, ymin = temp.lower.y, zmin = temp.lower.z;
    double xmax = temp.upper.x, ymax = temp.upper.y, zmax = temp.upper.z;
    //printf("%f   %f    %f   %f   %f    %f\n", xmin, ymin, zmin, xmax, ymax, zmax);
    //printf("%f   %f    %f\n", xmax, ymax, zmax);
    int    warpTid = threadIdx.x % 32;
    int    warpId  = (threadIdx.x >> 5);
    double nextTp;
    int    warpNum;
    int    tidNum = 32;
    if(blockIdx.x == gridDim.x - 1)
    {
        warpNum = ((number - idof + 31) >> 5);
        if(warpId == warpNum - 1)
        {
            tidNum = number - idof - (warpNum - 1) * 32;
        }
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < tidNum; i = (i << 1))
    {
        temp.combines(__shfl_down_sync(0xffffffff, xmin, i),
                      __shfl_down_sync(0xffffffff, ymin, i),
                      __shfl_down_sync(0xffffffff, zmin, i),
                      __shfl_down_sync(0xffffffff, xmax, i),
                      __shfl_down_sync(0xffffffff, ymax, i),
                      __shfl_down_sync(0xffffffff, zmax, i));
        if(warpTid + i < tidNum)
        {
            xmin = temp.lower.x, ymin = temp.lower.y, zmin = temp.lower.z;
            xmax = temp.upper.x, ymax = temp.upper.y, zmax = temp.upper.z;
        }
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
        xmin = temp.lower.x, ymin = temp.lower.y, zmin = temp.lower.z;
        xmax = temp.upper.x, ymax = temp.upper.y, zmax = temp.upper.z;
        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp.combines(__shfl_down_sync(0xffffffff, xmin, i),
                          __shfl_down_sync(0xffffffff, ymin, i),
                          __shfl_down_sync(0xffffffff, zmin, i),
                          __shfl_down_sync(0xffffffff, xmax, i),
                          __shfl_down_sync(0xffffffff, ymax, i),
                          __shfl_down_sync(0xffffffff, zmax, i));
            if(threadIdx.x + i < warpNum)
            {
                xmin = temp.lower.x, ymin = temp.lower.y, zmin = temp.lower.z;
                xmax = temp.upper.x, ymax = temp.upper.y, zmax = temp.upper.z;
            }
        }
    }
    if(threadIdx.x == 0)
    {
        _leafBoxes[blockIdx.x] = temp;
    }
}

template <class element_type>
__global__ void _calcLeafBvs(const double3*      _vertexes,
                             const element_type* _elements,
                             AABB*               _bvs,
                             int                 faceNum,
                             int                 type = 0,
                             const int*          _bodyID = nullptr,
                             const int*          _collision_skip_matrix = nullptr,
                             int                 _collision_body_count = 0)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= faceNum)
        return;
    AABB _bv;

    element_type _e = _elements[idx];

    // BVH-skip optimization (audit/perf-bvh-skip-isolated): if the element's
    // body has all collisions excluded (diag matrix[B][B]==1), leave _bv as
    // default (empty: lower>upper). Cloth/other-body queries' overlap tests
    // will return false, so the entire isolated-body subtree is naturally
    // pruned during traversal — no descent into these leaves at all.
    if(_bodyID && _collision_skip_matrix && _collision_body_count > 0) {
        int B = _bodyID[_e.x];
        if(B >= 0 && B < _collision_body_count
           && _collision_skip_matrix[B * _collision_body_count + B] != 0) {
            _bvs[idx] = _bv;  // default empty bbox
            return;
        }
    }

    double3      _v = _vertexes[_e.x];
    _bv.combines(_v.x, _v.y, _v.z);
    _v = _vertexes[_e.y];
    _bv.combines(_v.x, _v.y, _v.z);
    if(type == 0)
    {
        _v = _vertexes[*((uint32_t*)(&_e) + 2)];
        _bv.combines(_v.x, _v.y, _v.z);
    }
    _bvs[idx] = _bv;
}

template <class element_type>
__global__ void _calcLeafBvs_ccd(const double3*      _vertexes,
                                 const double3*      _moveDir,
                                 double              alpha,
                                 const element_type* _elements,
                                 AABB*               _bvs,
                                 int                 faceNum,
                                 int                 type = 0,
                                 const int*          _bodyID = nullptr,
                                 const int*          _collision_skip_matrix = nullptr,
                                 int                 _collision_body_count = 0,
                                 const double*       alpha_dev = nullptr)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= faceNum)
        return;
    if(alpha_dev) alpha = *alpha_dev;
    AABB _bv;

    element_type _e   = _elements[idx];

    // BVH-skip optimization (audit/perf-bvh-skip-isolated)
    if(_bodyID && _collision_skip_matrix && _collision_body_count > 0) {
        int B = _bodyID[_e.x];
        if(B >= 0 && B < _collision_body_count
           && _collision_skip_matrix[B * _collision_body_count + B] != 0) {
            _bvs[idx] = _bv;
            return;
        }
    }

    double3      _v   = _vertexes[_e.x];
    double3      _mvD = _moveDir[_e.x];
    _bv.combines(_v.x, _v.y, _v.z);
    _bv.combines(_v.x - _mvD.x * alpha, _v.y - _mvD.y * alpha, _v.z - _mvD.z * alpha);


    _v   = _vertexes[_e.y];
    _mvD = _moveDir[_e.y];
    _bv.combines(_v.x, _v.y, _v.z);
    _bv.combines(_v.x - _mvD.x * alpha, _v.y - _mvD.y * alpha, _v.z - _mvD.z * alpha);
    if(type == 0)
    {
        _v   = _vertexes[*((uint32_t*)(&_e) + 2)];
        _mvD = _moveDir[*((uint32_t*)(&_e) + 2)];
        _bv.combines(_v.x, _v.y, _v.z);
        _bv.combines(_v.x - _mvD.x * alpha, _v.y - _mvD.y * alpha, _v.z - _mvD.z * alpha);
    }
    _bvs[idx] = _bv;
}

// BVH-skip #3: indirect leaf-bbox kernel.
// Builds bbox for n_active leaves; thread t reads element via _active_idx[t]
// and writes _bvs[t]. Combined with calcLeafNodes_indirect, the final BVH stores
// ORIGINAL face/edge indices in element_idx (so query kernels can dereference
// _faces[element_idx] correctly), while topology size is reduced to n_active.
template <class element_type>
__global__ void _calcLeafBvs_indirect(const double3*      _vertexes,
                                      const element_type* _elements,
                                      const int*          _active_idx,
                                      AABB*               _bvs,
                                      int                 n_active,
                                      int                 type)
{
    int t = threadIdx.x + blockIdx.x * blockDim.x;
    if(t >= n_active)
        return;
    int          orig = _active_idx[t];
    element_type _e   = _elements[orig];
    AABB         _bv;
    double3      _v = _vertexes[_e.x];
    _bv.combines(_v.x, _v.y, _v.z);
    _v = _vertexes[_e.y];
    _bv.combines(_v.x, _v.y, _v.z);
    if(type == 0)
    {
        _v = _vertexes[*((uint32_t*)(&_e) + 2)];
        _bv.combines(_v.x, _v.y, _v.z);
    }
    _bvs[t] = _bv;
}

template <class element_type>
__global__ void _calcLeafBvs_ccd_indirect(const double3*      _vertexes,
                                          const double3*      _moveDir,
                                          double              alpha,
                                          const element_type* _elements,
                                          const int*          _active_idx,
                                          AABB*               _bvs,
                                          int                 n_active,
                                          int                 type,
                                          const double*       alpha_dev = nullptr)
{
    int t = threadIdx.x + blockIdx.x * blockDim.x;
    if(t >= n_active)
        return;
    if(alpha_dev) alpha = *alpha_dev;   // [de-CPU] per-env CCD search alpha read on device
    int          orig = _active_idx[t];
    element_type _e   = _elements[orig];
    AABB         _bv;
    double3      _v   = _vertexes[_e.x];
    double3      _mvD = _moveDir[_e.x];
    _bv.combines(_v.x, _v.y, _v.z);
    _bv.combines(_v.x - _mvD.x * alpha, _v.y - _mvD.y * alpha, _v.z - _mvD.z * alpha);

    _v   = _vertexes[_e.y];
    _mvD = _moveDir[_e.y];
    _bv.combines(_v.x, _v.y, _v.z);
    _bv.combines(_v.x - _mvD.x * alpha, _v.y - _mvD.y * alpha, _v.z - _mvD.z * alpha);
    if(type == 0)
    {
        _v   = _vertexes[*((uint32_t*)(&_e) + 2)];
        _mvD = _moveDir[*((uint32_t*)(&_e) + 2)];
        _bv.combines(_v.x, _v.y, _v.z);
        _bv.combines(_v.x - _mvD.x * alpha, _v.y - _mvD.y * alpha, _v.z - _mvD.z * alpha);
    }
    _bvs[t] = _bv;
}

// Variant of _calcLeafNodes that maps the (sorted) leaf-array index back to
// the ORIGINAL face/edge index via _active_idx, so query kernels work unchanged.
__global__ void _calcLeafNodes_indirect(Node*           _nodes,
                                        const uint32_t* _indices,
                                        const int*      _active_idx,
                                        int             number)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= number)
        return;
    if(idx < number - 1)
    {
        _nodes[idx].left_idx    = 0xFFFFFFFF;
        _nodes[idx].right_idx   = 0xFFFFFFFF;
        _nodes[idx].parent_idx  = 0xFFFFFFFF;
        _nodes[idx].element_idx = 0xFFFFFFFF;
    }
    int l_idx                 = idx + number - 1;
    _nodes[l_idx].left_idx    = 0xFFFFFFFF;
    _nodes[l_idx].right_idx   = 0xFFFFFFFF;
    _nodes[l_idx].parent_idx  = 0xFFFFFFFF;
    _nodes[l_idx].element_idx = _active_idx[_indices[idx]];
}

// [env-det] env-major Morton: put the prim's env id in the HIGH bits so co-located identical envs
// sort into separate contiguous blocks (env-symmetric tree), instead of the default global-index
// tie-break that interleaves equal-Morton co-located prims non-deterministically. STIFF_BVH_ENVDET.
__device__ int g_bvh_envmajor = 0;
void set_bvh_envmajor(int v){ CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_bvh_envmajor, &v, sizeof(int))); }
__global__ void _calcMChash(uint64_t* _MChash, AABB* _bvs, int number, const int* prim_env,
                            const int* prim_localid, const double3* env_offset,
                            const uint32_t* prim_v0)
{
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= number)
        return;
    AABB    maxBv     = _bvs[0];
    double3 SceneSize = make_double3(maxBv.upper.x - maxBv.lower.x,
                                     maxBv.upper.y - maxBv.lower.y,
                                     maxBv.upper.z - maxBv.lower.z);
    double3 centerP   = _bvs[idx + number - 1].center();
    // [env-det] subtract the per-env world offset so the Morton code is computed in the LOCAL frame
    // ⇒ co-located identical envs get MIRROR morton ⇒ identical intra-env tree structure. The offset
    // AABBs are kept for traversal efficiency / env separation; only the sort key is localized.
    if(g_bvh_envmajor && env_offset && prim_v0)
    { double3 o = env_offset[prim_v0[idx]]; centerP.x -= o.x; centerP.y -= o.y; centerP.z -= o.z; }
    double3 offset    = make_double3(centerP.x - maxBv.lower.x,
                                  centerP.y - maxBv.lower.y,
                                  centerP.z - maxBv.lower.z);

    //printf("%d   %f     %f     %f\n", offset.x, offset.y, offset.z);
    uint64_t mc32 = morton_code(
        offset.x / SceneSize.x, offset.y / SceneSize.y, offset.z / SceneSize.z);
    uint64_t mc64;
    if(g_bvh_envmajor && prim_env)
    {   // [env(high), morton(30), ENV-LOCAL-prim-id(20 low)] — env-blocked AND the low-bits tie-break
        // is ENV-LOCAL (mirror across identical envs) so find_split/determine_range give IDENTICAL
        // subtree structure for co-located envs (global idx in the low bits made them differ → the
        // 5th hidden global-index dependence).
        uint64_t env = (uint64_t)(prim_env[idx] < 0 ? 1023 : prim_env[idx]);
        // localid MUST be derived from env-local VERTEX ids (mirror), not global prim rank (NOT
        // mirror — edge/face global numbering is env-asymmetric). 26-bit slot (0-25).
        uint64_t loc = prim_localid ? ((uint64_t)prim_localid[idx] & 0x3FFFFFFULL) : ((uint64_t)idx & 0x3FFFFFFULL);
        mc64 = (env << 56) | ((mc32 & 0x3FFFFFFFULL) << 26) | loc;
    }
    else
        mc64 = ((mc32 << 32) | idx);
    _MChash[idx]  = mc64;
}

__global__ void _calcLeafNodes(Node* _nodes, const uint32_t* _indices, int number)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= number)
        return;
    if(idx < number - 1)
    {
        _nodes[idx].left_idx    = 0xFFFFFFFF;
        _nodes[idx].right_idx   = 0xFFFFFFFF;
        _nodes[idx].parent_idx  = 0xFFFFFFFF;
        _nodes[idx].element_idx = 0xFFFFFFFF;
    }
    int l_idx                 = idx + number - 1;
    _nodes[l_idx].left_idx    = 0xFFFFFFFF;
    _nodes[l_idx].right_idx   = 0xFFFFFFFF;
    _nodes[l_idx].parent_idx  = 0xFFFFFFFF;
    _nodes[l_idx].element_idx = _indices[idx];
}


__global__ void _calcInternalNodes(Node* _nodes, const uint64_t* _MChash, int number)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= number - 1)
        return;
    const uint2        ij    = determine_range(_MChash, number, idx);
    const unsigned int gamma = find_split(_MChash, number, ij.x, ij.y);

    _nodes[idx].left_idx  = gamma;
    _nodes[idx].right_idx = gamma + 1;
    if(std::min(ij.x, ij.y) == gamma)
    {
        _nodes[idx].left_idx += number - 1;
    }
    if(std::max(ij.x, ij.y) == gamma + 1)
    {
        _nodes[idx].right_idx += number - 1;
    }
    _nodes[_nodes[idx].left_idx].parent_idx  = idx;
    _nodes[_nodes[idx].right_idx].parent_idx = idx;
}

__global__ void _calcInternalAABB(const Node* _nodes, AABB* _bvs, uint32_t* flags, int number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    idx = idx + number - 1;

    uint32_t parent = _nodes[idx].parent_idx;
    while(parent != 0xFFFFFFFF)  // means idx == 0
    {
        const int old = atomicCAS(flags + parent, 0xFFFFFFFF, 0);
        if(old == 0xFFFFFFFF)
        {
            return;
        }

        const uint32_t lidx = _nodes[parent].left_idx;
        const uint32_t ridx = _nodes[parent].right_idx;

        const AABB lbox = _bvs[lidx];
        const AABB rbox = _bvs[ridx];
        _bvs[parent]    = merge(lbox, rbox);

        __threadfence();

        parent = _nodes[parent].parent_idx;
    }
}

__global__ void _sortBvs(const uint32_t* _indices, AABB* _bvs, AABB* _temp_bvs, int number)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= number)
        return;
    _bvs[idx] = _temp_bvs[_indices[idx]];
}

// [env-part B] traversal pruning gate: skip other-env subtrees by env-id (no cross-env candidates).
__device__ int g_bvh_envpart = 0;
void set_bvh_envpart(int v){ CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_bvh_envpart, &v, sizeof(int))); }

// [perenv-par] per-vertex env id (= d_point_to_group); cross-env self-collision pairs skipped at
// emission when set. -1 = ungrouped/static (never skipped). Null = gate off (byte-for-byte legacy).
__device__ const int* g_self_p2g = nullptr;
__device__ unsigned long long g_xskip = 0;   // [debug] count of cross-env pairs skipped
void set_self_p2g(const int* p){
    if(getenv("STIFF_XSKIP_DBG")) fprintf(stderr, "[self_p2g] set to %p\n", (const void*)p);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_self_p2g, &p, sizeof(const int*)));
}
unsigned long long get_xskip(){ unsigned long long h=0; cudaMemcpyFromSymbol(&h, g_xskip, sizeof(h)); return h; }

// [multi-env subscene, v0.6.7 API] user-facing per-vertex env filter
// (SimEngine::set_vertex_env_ids). SEPARATE symbol from g_self_p2g on purpose:
// buildCP RESETS g_self_p2g every call (DECOUPLE_THRESH-gated internal filter),
// which would wipe an API-set array (caught by test_env_isolation: isolate mode
// stacked instead of passing through). The two filters compose (skip if either
// says skip). Semantics: pair skipped iff both env ids >= 0 and different.
__device__ const int* g_vertex_env_id = nullptr;
__device__ inline bool _same_env(int vA, int vB)
{
    if(g_vertex_env_id == nullptr) return true;
    int ea = g_vertex_env_id[vA], eb = g_vertex_env_id[vB];
    return ea < 0 || eb < 0 || ea == eb;
}
void mlbvh_set_vertex_env_id(const int* d_vertex_env_id)
{
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_vertex_env_id, &d_vertex_env_id, sizeof(const int*)));
}

// [perenv-par] skip a self-collision pair iff both representative verts are in DIFFERENT non-negative
// envs. Gate-off (g_self_p2g==null) or any static (-1) endpoint => never skip.
__device__ inline bool _cross_env_skip(int va, int vb)
{
    if(!g_self_p2g) return false;
    int ga = g_self_p2g[va], gb = g_self_p2g[vb];
    bool sk = (ga >= 0 && gb >= 0 && ga != gb);
    if(sk) atomicAdd(&g_xskip, 1ULL);
    return sk;
}

// [env-part B] node_env[leaf] = prim env (via the leaf's element_idx). Internal init to -2 (unset);
// _propagateNodeEnv fills them bottom-up.
__global__ void _setLeafEnv(int* node_env, const Node* _nodes, const int* prim_env, int number)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= number) return;
    if(idx < number - 1) node_env[idx] = -2;
    int l = idx + number - 1;
    uint32_t e = _nodes[l].element_idx;
    node_env[l] = (e != 0xFFFFFFFF && prim_env) ? prim_env[e] : -1;
}

// [env-part B] bottom-up env propagation (mirrors _calcInternalAABB's atomicCAS climb): a parent's
// env = its children's common env, or -1 (MIXED) if they differ. Only the 2nd child to reach a parent
// proceeds (both children's env are then known). Leaves must be set (via _setLeafEnv) first.
__global__ void _propagateNodeEnv(int* node_env, const Node* _nodes, uint32_t* flags, int number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number) return;
    idx = idx + number - 1;
    uint32_t parent = _nodes[idx].parent_idx;
    while(parent != 0xFFFFFFFF)
    {
        const int old = atomicCAS(flags + parent, 0xFFFFFFFF, 0);
        if(old == 0xFFFFFFFF) return;  // first child arrives -> wait for sibling
        const uint32_t l = _nodes[parent].left_idx;
        const uint32_t r = _nodes[parent].right_idx;
        int le = node_env[l], re = node_env[r];
        node_env[parent] = (le == re) ? le : -1;  // uniform env, or MIXED
        __threadfence();
        parent = _nodes[parent].parent_idx;
    }
}

void computeNodeEnv(int* node_env, const Node* _nodes, const int* prim_env, uint32_t* flags, int number, cudaStream_t stream)
{
    if(number < 1 || !node_env || !prim_env) return;
    const unsigned int tn = default_threads;
    int bn = (number + tn - 1) / tn;
    _setLeafEnv<<<bn, tn, 0, stream>>>(node_env, _nodes, prim_env, number);
    if(number > 1)
    {
        CUDA_SAFE_CALL(cudaMemsetAsync(flags, 0xFFFFFFFF, sizeof(uint32_t) * (number - 1), stream));
        _propagateNodeEnv<<<bn, tn, 0, stream>>>(node_env, _nodes, flags, number);
    }
}

__global__ void _selfQuery_vf(const int*      _bodyID,
                              const int*      _btype,
                              const double3*  _vertexes,
                              const uint3*    _faces,
                              const uint32_t* _surfVerts,
                              const AABB*     _bvs,
                              const Node*     _nodes,
                              int4*           _collisionPair,
                              int4*           _ccd_collisionPair,
                              uint32_t*       _cpNum,
                              int*            MatIndex,
                              double          dHat,
                              int             number,
                              const int*      _collision_skip_matrix,
                              int             _collision_body_count,
                              const int*      _body_id_to_is_fem)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    uint32_t  stack[2048];
    uint32_t* stack_ptr = stack;
    *stack_ptr++        = 0;

    AABB _bv;
    idx       = _surfVerts[idx];

    // BVH-skip: query vertex's body has no possible collisions → exit early.
    // (audit/perf-bvh-skip-isolated: diag[B][B]==1 marks isolated body)
    if(_collision_skip_matrix && _collision_body_count > 0) {
        int B = _bodyID[idx];
        if(B >= 0 && B < _collision_body_count
           && _collision_skip_matrix[B * _collision_body_count + B] != 0)
            return;
    }

    _bv.upper = _vertexes[idx];
    _bv.lower = _vertexes[idx];
    //double bboxDiagSize2 = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(_bvs[0].upper, _bvs[0].lower));
    //printf("%f\n", bboxDiagSize2);
    double gapl = sqrt(dHat);  //0.001 * sqrt(bboxDiagSize2);
    //double dHat = gapl * gapl;// *bboxDiagSize2;
    unsigned int num_found = 0;
    do
    {
        const uint32_t node_id = *--stack_ptr;
        if(g_bvh_audit) { int _d=(int)(stack_ptr-stack); atomicMax(&g_max_stack,_d); }
        const uint32_t L_idx   = _nodes[node_id].left_idx;
        const uint32_t R_idx   = _nodes[node_id].right_idx;

        if(overlap(_bv, _bvs[L_idx], gapl))
        {
            const auto obj_idx = _nodes[L_idx].element_idx;
            if(obj_idx != 0xFFFFFFFF)
            {
                if(_should_check_pair(_bodyID[idx], _bodyID[_faces[obj_idx].x], _body_id_to_is_fem)
                   && !_is_collision_excluded(_bodyID[idx], _bodyID[_faces[obj_idx].x],
                                             _collision_skip_matrix, _collision_body_count)
                   && !_cross_env_skip(idx, _faces[obj_idx].x)
                   && _same_env(idx, _faces[obj_idx].x))
                {
                    if(idx != _faces[obj_idx].x && idx != _faces[obj_idx].y
                       && idx != _faces[obj_idx].z)
                    {
                        if(!(_btype[idx] >= 2 && _btype[_faces[obj_idx].x] >= 2
                             && _btype[_faces[obj_idx].y] >= 2
                             && _btype[_faces[obj_idx].z] >= 2))
                            _checkPTintersection(_vertexes,
                                                 idx,
                                                 _faces[obj_idx].x,
                                                 _faces[obj_idx].y,
                                                 _faces[obj_idx].z,
                                                 dHat,
                                                 _cpNum,
                                                 MatIndex,
                                                 _collisionPair,
                                                 _ccd_collisionPair);
                    }
                }
            }
            else  // the node is not a leaf.
            {
                *stack_ptr++ = L_idx;
            }
        }
        if(overlap(_bv, _bvs[R_idx], gapl))
        {
            const auto obj_idx = _nodes[R_idx].element_idx;
            if(obj_idx != 0xFFFFFFFF)
            {
                if(_should_check_pair(_bodyID[idx], _bodyID[_faces[obj_idx].x], _body_id_to_is_fem)
                   && !_is_collision_excluded(_bodyID[idx], _bodyID[_faces[obj_idx].x],
                                             _collision_skip_matrix, _collision_body_count)
                   && !_cross_env_skip(idx, _faces[obj_idx].x)
                   && _same_env(idx, _faces[obj_idx].x))
                {
                    if(idx != _faces[obj_idx].x && idx != _faces[obj_idx].y
                       && idx != _faces[obj_idx].z)
                    {
                        if(!(_btype[idx] >= 2 && _btype[_faces[obj_idx].x] >= 2
                             && _btype[_faces[obj_idx].y] >= 2
                             && _btype[_faces[obj_idx].z] >= 2))
                            _checkPTintersection(_vertexes,
                                                 idx,
                                                 _faces[obj_idx].x,
                                                 _faces[obj_idx].y,
                                                 _faces[obj_idx].z,
                                                 dHat,
                                                 _cpNum,
                                                 MatIndex,
                                                 _collisionPair,
                                                 _ccd_collisionPair);
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

__global__ void _selfQuery_vf_ccd(const int*      _bodyID,
                                  const int*      _btype,
                                  const double3*  _vertexes,
                                  const double3*  moveDir,
                                  double          alpha,
                                  const uint3*    _faces,
                                  const uint32_t* _surfVerts,
                                  const AABB*     _bvs,
                                  const Node*     _nodes,
                                  int4*           _ccd_collisionPair,
                                  uint32_t*       _cpNum,
                                  double          dHat,
                                  int             number,
                                  const int*      _collision_skip_matrix,
                                  int             _collision_body_count,
                                  const int*      _body_id_to_is_fem,
                                  const double*   alpha_dev = nullptr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    if(alpha_dev) alpha = *alpha_dev;   // [de-CPU] per-env CCD search alpha read on device

    uint32_t  stack[2048];
    uint32_t* stack_ptr = stack;
    *stack_ptr++        = 0;

    AABB _bv;
    idx                    = _surfVerts[idx];

    // BVH-skip (audit/perf-bvh-skip-isolated)
    if(_collision_skip_matrix && _collision_body_count > 0) {
        int B = _bodyID[idx];
        if(B >= 0 && B < _collision_body_count
           && _collision_skip_matrix[B * _collision_body_count + B] != 0)
            return;
    }

    double3 current_vertex = _vertexes[idx];
    double3 mvD            = moveDir[idx];
    _bv.upper              = current_vertex;
    _bv.lower              = current_vertex;
    _bv.combines(current_vertex.x - mvD.x * alpha,
                 current_vertex.y - mvD.y * alpha,
                 current_vertex.z - mvD.z * alpha);
    //double bboxDiagSize2 = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(_bvs[0].upper, _bvs[0].lower));
    //printf("%f\n", bboxDiagSize2);
    double gapl = sqrt(dHat);  //0.001 * sqrt(bboxDiagSize2);
    //double dHat = gapl * gapl;// *bboxDiagSize2;
    unsigned int num_found = 0;
    do
    {
        const uint32_t node_id = *--stack_ptr;
        if(g_bvh_audit) { int _d=(int)(stack_ptr-stack); atomicMax(&g_max_stack,_d); }
        const uint32_t L_idx   = _nodes[node_id].left_idx;
        const uint32_t R_idx   = _nodes[node_id].right_idx;

        if(overlap(_bv, _bvs[L_idx], gapl))
        {
            const auto obj_idx = _nodes[L_idx].element_idx;
            if(obj_idx != 0xFFFFFFFF)
            {
                if(_should_check_pair(_bodyID[idx], _bodyID[_faces[obj_idx].x], _body_id_to_is_fem)
                   && !_is_collision_excluded(_bodyID[idx], _bodyID[_faces[obj_idx].x],
                                             _collision_skip_matrix, _collision_body_count)
                   && !_cross_env_skip(idx, _faces[obj_idx].x)
                   && _same_env(idx, _faces[obj_idx].x))
                {

                    if(!(_btype[idx] >= 2 && _btype[_faces[obj_idx].x] >= 2
                         && _btype[_faces[obj_idx].y] >= 2
                         && _btype[_faces[obj_idx].z] >= 2))
                        if(idx != _faces[obj_idx].x && idx != _faces[obj_idx].y
                           && idx != _faces[obj_idx].z)
                        {
                            _ccd_collisionPair[_emit_slot(_cpNum, g_ccd_cp_cap)] =
                                make_int4(-idx - 1,
                                          _faces[obj_idx].x,
                                          _faces[obj_idx].y,
                                          _faces[obj_idx].z);
                            //_checkPTintersection_fullCCD(_vertexes, idx, _faces[obj_idx].x, _faces[obj_idx].y, _faces[obj_idx].z, dHat, _cpNum, _ccd_collisionPair);
                        }
                }
            }
            else  // the node is not a leaf.
            {
                *stack_ptr++ = L_idx;
            }
        }
        if(overlap(_bv, _bvs[R_idx], gapl))
        {
            const auto obj_idx = _nodes[R_idx].element_idx;
            if(obj_idx != 0xFFFFFFFF)
            {
                if(_should_check_pair(_bodyID[idx], _bodyID[_faces[obj_idx].x], _body_id_to_is_fem)
                   && !_is_collision_excluded(_bodyID[idx], _bodyID[_faces[obj_idx].x],
                                             _collision_skip_matrix, _collision_body_count)
                   && !_cross_env_skip(idx, _faces[obj_idx].x)
                   && _same_env(idx, _faces[obj_idx].x))
                {
                    if(!(_btype[idx] >= 2 && _btype[_faces[obj_idx].x] >= 2
                         && _btype[_faces[obj_idx].y] >= 2
                         && _btype[_faces[obj_idx].z] >= 2))
                        if(idx != _faces[obj_idx].x && idx != _faces[obj_idx].y
                           && idx != _faces[obj_idx].z)
                        {
                            _ccd_collisionPair[_emit_slot(_cpNum, g_ccd_cp_cap)] =
                                make_int4(-idx - 1,
                                          _faces[obj_idx].x,
                                          _faces[obj_idx].y,
                                          _faces[obj_idx].z);
                            //_checkPTintersection_fullCCD(_vertexes, idx, _faces[obj_idx].x, _faces[obj_idx].y, _faces[obj_idx].z, dHat, _cpNum, _ccd_collisionPair);
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


// [ee-lb] traversal body shared by the __launch_bounds__ occupancy variants below (same code,
// different register budgets — selected at launch via STIFF_EE_LB).
static __device__ __forceinline__ void _selfQuery_ee_body(const int*     _bodyID,
                              const int*     _btype,
                              const double3* _vertexes,
                              const double3* _rest_vertexes,
                              const uint2*   _edges,
                              const AABB*    _bvs,
                              const Node*    _nodes,
                              int4*          _collisionPair,
                              int4*          _ccd_collisionPair,
                              uint32_t*      _cpNum,
                              int*           MatIndex,
                              double         dHat,
                              int            number,
                              const int*     _collision_skip_matrix,
                              int            _collision_body_count,
                              const int*     _body_id_to_is_fem,
                              const int* node_env)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    uint32_t  stack[2048];
    uint32_t* stack_ptr = stack;
    *stack_ptr++        = 0;

    idx               = idx + number - 1;
    AABB     _bv      = _bvs[idx];
    uint32_t self_eid = _nodes[idx].element_idx;
    int qenv = (g_bvh_envpart && node_env) ? node_env[idx] : -1;  // [env-part B] query edge env

    // BVH-skip (audit/perf-bvh-skip-isolated): if both edge endpoints' body
    // is isolated, no collision is possible — exit early.
    if(_collision_skip_matrix && _collision_body_count > 0) {
        int B = _bodyID[_edges[self_eid].x];
        if(B >= 0 && B < _collision_body_count
           && _collision_skip_matrix[B * _collision_body_count + B] != 0)
            return;
    }

    //double bboxDiagSize2 = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(_bvs[0].upper, _bvs[0].lower));
    //printf("%f\n", bboxDiagSize2);
    double gapl = sqrt(dHat);  //0.001 * sqrt(bboxDiagSize2);
    //double dHat = gapl * gapl;// *bboxDiagSize2;
    unsigned int num_found = 0;
    do
    {
        const uint32_t node_id = *--stack_ptr;
        if(g_bvh_audit) { int _d=(int)(stack_ptr-stack); atomicMax(&g_max_stack,_d); }
        const uint32_t L_idx   = _nodes[node_id].left_idx;
        const uint32_t R_idx   = _nodes[node_id].right_idx;

        if((qenv < 0 || node_env[L_idx] < 0 || node_env[L_idx] == qenv) && overlap(_bv, _bvs[L_idx], gapl))
        {
            const auto obj_idx = _nodes[L_idx].element_idx;
            if(obj_idx != 0xFFFFFFFF)
            {
                if(self_eid != obj_idx)
                {
                    if(_should_check_pair(_bodyID[_edges[self_eid].x], _bodyID[_edges[obj_idx].x], _body_id_to_is_fem)
                       && !_is_collision_excluded(_bodyID[_edges[self_eid].x], _bodyID[_edges[obj_idx].x],
                                                 _collision_skip_matrix, _collision_body_count)
                       && !_cross_env_skip(_edges[self_eid].x, _edges[obj_idx].x)
                       && _same_env(_edges[self_eid].x, _edges[obj_idx].x))
                    {


                        if(!(_edges[self_eid].x == _edges[obj_idx].x
                             || _edges[self_eid].x == _edges[obj_idx].y
                             || _edges[self_eid].y == _edges[obj_idx].x
                             || _edges[self_eid].y == _edges[obj_idx].y || (!g_ee_nodedup && (g_ee_canon ? (_edge_lkey(_edges[obj_idx]) < _edge_lkey(_edges[self_eid])) : (obj_idx < self_eid)))))
                        {
                            //printf("%d   %d   %d   %d\n", _edges[self_eid].x, _edges[self_eid].y, _edges[obj_idx].x, _edges[obj_idx].y);
                            if(!(_btype[_edges[self_eid].x] >= 2
                                 && _btype[_edges[self_eid].y] >= 2
                                 && _btype[_edges[obj_idx].x] >= 2
                                 && _btype[_edges[obj_idx].y] >= 2))
                                _checkEEintersection(_vertexes,
                                                     _rest_vertexes,
                                                     _edges[self_eid].x,
                                                     _edges[self_eid].y,
                                                     _edges[obj_idx].x,
                                                     _edges[obj_idx].y,
                                                     obj_idx,
                                                     dHat,
                                                     _cpNum,
                                                     MatIndex,
                                                     _collisionPair,
                                                     _ccd_collisionPair,
                                                     number);
                        }
                    }
                }
            }
            else  // the node is not a leaf.
            {
                *stack_ptr++ = L_idx;
            }
        }
        if((qenv < 0 || node_env[R_idx] < 0 || node_env[R_idx] == qenv) && overlap(_bv, _bvs[R_idx], gapl))
        {
            const auto obj_idx = _nodes[R_idx].element_idx;
            if(obj_idx != 0xFFFFFFFF)
            {
                if(self_eid != obj_idx)
                {
                    if(_should_check_pair(_bodyID[_edges[self_eid].x], _bodyID[_edges[obj_idx].x], _body_id_to_is_fem)
                       && !_is_collision_excluded(_bodyID[_edges[self_eid].x], _bodyID[_edges[obj_idx].x],
                                                 _collision_skip_matrix, _collision_body_count)
                       && !_cross_env_skip(_edges[self_eid].x, _edges[obj_idx].x)
                       && _same_env(_edges[self_eid].x, _edges[obj_idx].x))
                    {
                        if(!(_edges[self_eid].x == _edges[obj_idx].x
                             || _edges[self_eid].x == _edges[obj_idx].y
                             || _edges[self_eid].y == _edges[obj_idx].x
                             || _edges[self_eid].y == _edges[obj_idx].y || (!g_ee_nodedup && (g_ee_canon ? (_edge_lkey(_edges[obj_idx]) < _edge_lkey(_edges[self_eid])) : (obj_idx < self_eid)))))
                        {
                            //printf("%d   %d   %d   %d\n", _edges[self_eid].x, _edges[self_eid].y, _edges[obj_idx].x, _edges[obj_idx].y);
                            if(!(_btype[_edges[self_eid].x] >= 2
                                 && _btype[_edges[self_eid].y] >= 2
                                 && _btype[_edges[obj_idx].x] >= 2
                                 && _btype[_edges[obj_idx].y] >= 2))
                                _checkEEintersection(_vertexes,
                                                     _rest_vertexes,
                                                     _edges[self_eid].x,
                                                     _edges[self_eid].y,
                                                     _edges[obj_idx].x,
                                                     _edges[obj_idx].y,
                                                     obj_idx,
                                                     dHat,
                                                     _cpNum,
                                                     MatIndex,
                                                     _collisionPair,
                                                     _ccd_collisionPair,
                                                     number);
                        }
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

// [ee-lb] launch shells. Baseline compiles to ~168 reg → 1 block/SM (8 warps, 16.7% theoretical
// occupancy; 11.8% achieved) while DRAM sits at ~0.3% — latency-bound with nothing in flight to
// hide it. The capped variants trade registers (spills land in an idle L1/L2) for resident warps:
//   lb2: __launch_bounds__(256,2) → ≤128 reg → 16 warps/SM;  lb3: (256,3) → ≤85 reg → 24 warps/SM.
#define _SQEE_PARAMS                                                                               \
    const int *_bodyID, const int *_btype, const double3 *_vertexes,                               \
        const double3 *_rest_vertexes, const uint2 *_edges, const AABB *_bvs,                      \
        const Node *_nodes, int4 *_collisionPair, int4 *_ccd_collisionPair, uint32_t *_cpNum,      \
        int *MatIndex, double dHat, int number, const int *_collision_skip_matrix,                 \
        int _collision_body_count, const int *_body_id_to_is_fem, const int *node_env
#define _SQEE_ARGS                                                                                 \
    _bodyID, _btype, _vertexes, _rest_vertexes, _edges, _bvs, _nodes, _collisionPair,              \
        _ccd_collisionPair, _cpNum, MatIndex, dHat, number, _collision_skip_matrix,                \
        _collision_body_count, _body_id_to_is_fem, node_env
__global__ void _selfQuery_ee(_SQEE_PARAMS)
{
    _selfQuery_ee_body(_SQEE_ARGS);
}
__global__ void __launch_bounds__(256, 2) _selfQuery_ee_lb2(_SQEE_PARAMS)
{
    _selfQuery_ee_body(_SQEE_ARGS);
}
__global__ void __launch_bounds__(256, 3) _selfQuery_ee_lb3(_SQEE_PARAMS)
{
    _selfQuery_ee_body(_SQEE_ARGS);
}

__global__ void _selfQuery_ee_ccd(const int*     _bodyID,
                                  const int*     _btype,
                                  const double3* _vertexes,
                                  const double3* moveDir,
                                  double         alpha,
                                  const uint2*   _edges,
                                  const AABB*    _bvs,
                                  const Node*    _nodes,
                                  int4*          _ccd_collisionPair,
                                  uint32_t*      _cpNum,
                                  double         dHat,
                                  int            number,
                                  const int*     _collision_skip_matrix,
                                  int            _collision_body_count,
                                  const int*     _body_id_to_is_fem,
                              const int* node_env,
                                  const double*  alpha_dev = nullptr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    if(alpha_dev) alpha = *alpha_dev;   // [de-CPU] per-env CCD search alpha read on device

    uint32_t  stack[2048];
    uint32_t* stack_ptr   = stack;
    *stack_ptr++          = 0;
    idx                   = idx + number - 1;
    AABB     _bv          = _bvs[idx];
    uint32_t self_eid     = _nodes[idx].element_idx;
    int qenv = (g_bvh_envpart && node_env) ? node_env[idx] : -1;  // [env-part B] query edge env
    uint2    current_edge = _edges[self_eid];

    // BVH-skip (audit/perf-bvh-skip-isolated)
    if(_collision_skip_matrix && _collision_body_count > 0) {
        int B = _bodyID[current_edge.x];
        if(B >= 0 && B < _collision_body_count
           && _collision_skip_matrix[B * _collision_body_count + B] != 0)
            return;
    }
    //double3 edge_tvert0 = __GEIGEN__::__minus(_vertexes[current_edge.x], __GEIGEN__::__s_vec_multiply(moveDir[current_edge.x], alpha));
    //double3 edge_tvert1 = __GEIGEN__::__minus(_vertexes[current_edge.y], __GEIGEN__::__s_vec_multiply(moveDir[current_edge.y], alpha));
    //_bv.combines(edge_tvert0.x, edge_tvert0.y, edge_tvert0.z);
    //_bv.combines(edge_tvert1.x, edge_tvert1.y, edge_tvert1.z);
    double gapl = sqrt(dHat);

    unsigned int num_found = 0;
    do
    {
        const uint32_t node_id = *--stack_ptr;
        if(g_bvh_audit) { int _d=(int)(stack_ptr-stack); atomicMax(&g_max_stack,_d); }
        const uint32_t L_idx   = _nodes[node_id].left_idx;
        const uint32_t R_idx   = _nodes[node_id].right_idx;

        if((qenv < 0 || node_env[L_idx] < 0 || node_env[L_idx] == qenv) && overlap(_bv, _bvs[L_idx], gapl))
        {
            const auto obj_idx = _nodes[L_idx].element_idx;
            if(obj_idx != 0xFFFFFFFF)
            {
                if(self_eid != obj_idx)
                {
                    if(_should_check_pair(_bodyID[_edges[self_eid].x], _bodyID[_edges[obj_idx].x], _body_id_to_is_fem)
                       && !_is_collision_excluded(_bodyID[_edges[self_eid].x], _bodyID[_edges[obj_idx].x],
                                                 _collision_skip_matrix, _collision_body_count)
                       && !_cross_env_skip(_edges[self_eid].x, _edges[obj_idx].x)
                       && _same_env(_edges[self_eid].x, _edges[obj_idx].x))
                    {
                        if(!(_btype[_edges[self_eid].x] >= 2
                             && _btype[_edges[self_eid].y] >= 2
                             && _btype[_edges[obj_idx].x] >= 2
                             && _btype[_edges[obj_idx].y] >= 2))
                            if(!(current_edge.x == _edges[obj_idx].x
                                 || current_edge.x == _edges[obj_idx].y
                                 || current_edge.y == _edges[obj_idx].x
                                 || current_edge.y == _edges[obj_idx].y || (!g_ee_nodedup && (g_ee_canon ? (_edge_lkey(_edges[obj_idx]) < _edge_lkey(current_edge)) : (obj_idx < self_eid)))))
                            {
                                _ccd_collisionPair[_emit_slot(_cpNum, g_ccd_cp_cap)] =
                                    make_int4(current_edge.x,
                                              current_edge.y,
                                              _edges[obj_idx].x,
                                              _edges[obj_idx].y);
                            }
                    }
                }
            }
            else  // the node is not a leaf.
            {
                *stack_ptr++ = L_idx;
            }
        }
        if((qenv < 0 || node_env[R_idx] < 0 || node_env[R_idx] == qenv) && overlap(_bv, _bvs[R_idx], gapl))
        {
            const auto obj_idx = _nodes[R_idx].element_idx;
            if(obj_idx != 0xFFFFFFFF)
            {
                if(self_eid != obj_idx)
                {
                    if(_should_check_pair(_bodyID[_edges[self_eid].x], _bodyID[_edges[obj_idx].x], _body_id_to_is_fem)
                       && !_is_collision_excluded(_bodyID[_edges[self_eid].x], _bodyID[_edges[obj_idx].x],
                                                 _collision_skip_matrix, _collision_body_count)
                       && !_cross_env_skip(_edges[self_eid].x, _edges[obj_idx].x)
                       && _same_env(_edges[self_eid].x, _edges[obj_idx].x))
                    {
                        if(!(_btype[_edges[self_eid].x] >= 2
                             && _btype[_edges[self_eid].y] >= 2
                             && _btype[_edges[obj_idx].x] >= 2
                             && _btype[_edges[obj_idx].y] >= 2))
                            if(!(current_edge.x == _edges[obj_idx].x
                                 || current_edge.x == _edges[obj_idx].y
                                 || current_edge.y == _edges[obj_idx].x
                                 || current_edge.y == _edges[obj_idx].y || (!g_ee_nodedup && (g_ee_canon ? (_edge_lkey(_edges[obj_idx]) < _edge_lkey(current_edge)) : (obj_idx < self_eid)))))
                            {
                                _ccd_collisionPair[_emit_slot(_cpNum, g_ccd_cp_cap)] =
                                    make_int4(current_edge.x,
                                              current_edge.y,
                                              _edges[obj_idx].x,
                                              _edges[obj_idx].y);
                            }
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

///////////////////////////////////////host//////////////////////////////////////////////


AABB calcMaxBV(AABB* _leafBoxes, AABB* _tempLeafBox, const int& number)
{

    int                numbers   = number;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(AABB) * (threadNum >> 5);

    //AABB* _tempLeafBox;
    //CUDA_SAFE_CALL(cudaMalloc((void**)&_tempLeafBox, number * sizeof(AABB)));
    CUDA_SAFE_CALL(cudaMemcpy(
        _tempLeafBox, _leafBoxes + number - 1, number * sizeof(AABB), cudaMemcpyDeviceToDevice));

    _reduct_max_box<<<blockNum, threadNum, sharedMsize>>>(_tempLeafBox, numbers);

    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while(numbers > 1)
    {
        _reduct_max_box<<<blockNum, threadNum, sharedMsize>>>(_tempLeafBox, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    AABB h_bv;
    cudaMemcpy(&h_bv, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToHost);
    //CUDA_SAFE_CALL(cudaFree(_tempLeafBox));
    return h_bv;
}

// [perenv-parallel #1] async, no-host-sync scene bbox: reduces on `stream`, writes _leafBoxes[0]
// (device, read by calcMChash on the same stream) — NO D2H. Used by the per-env Construct so the
// per-env loop never syncs the host. (Host `scene`/getSceneSize is not needed mid per-env build.)
void calcMaxBV_async(AABB* _leafBoxes, AABB* _tempLeafBox, int number, cudaStream_t stream)
{
    int numbers = number;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    unsigned int sharedMsize = sizeof(AABB) * (threadNum >> 5);
    cudaMemcpyAsync(_tempLeafBox, _leafBoxes + number - 1, number * sizeof(AABB),
                    cudaMemcpyDeviceToDevice, stream);
    _reduct_max_box<<<blockNum, threadNum, sharedMsize, stream>>>(_tempLeafBox, numbers);
    numbers = blockNum; blockNum = (numbers + threadNum - 1) / threadNum;
    while(numbers > 1)
    {
        _reduct_max_box<<<blockNum, threadNum, sharedMsize, stream>>>(_tempLeafBox, numbers);
        numbers = blockNum; blockNum = (numbers + threadNum - 1) / threadNum;
    }
    cudaMemcpyAsync(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice, stream);
}

template <class element_type>
void calcLeafBvs(const double3*      _vertexes,
                 const element_type* _faces,
                 AABB*               _bvs,
                 const int&          faceNum,
                 const int&          type,
                 const int*          _bodyID = nullptr,
                 const int*          _collision_skip_matrix = nullptr,
                 int                 _collision_body_count = 0)
{
    int numbers = faceNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calcLeafBvs<<<blockNum, threadNum>>>(_vertexes, _faces, _bvs + numbers - 1, faceNum, type,
                                          _bodyID, _collision_skip_matrix, _collision_body_count);
}

template <class element_type>
void calcLeafBvs_fullCCD(const double3*      _vertexes,
                         const double3*      _moveDir,
                         const double&       alpha,
                         const element_type* _faces,
                         AABB*               _bvs,
                         const int&          faceNum,
                         const int&          type,
                         const int*          _bodyID = nullptr,
                         const int*          _collision_skip_matrix = nullptr,
                         int                 _collision_body_count = 0,
                         const double*       alpha_dev = nullptr)
{
    int numbers = faceNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calcLeafBvs_ccd<<<blockNum, threadNum>>>(
        _vertexes, _moveDir, alpha, _faces, _bvs + numbers - 1, faceNum, type,
        _bodyID, _collision_skip_matrix, _collision_body_count, alpha_dev);
}

// BVH-skip #3 launchers: write n_active leaves at _bvs+(n_active-1), saving
// sort/tree-build work proportional to the fraction of isolated faces/edges.
template <class element_type>
void calcLeafBvs_indirect(const double3*      _vertexes,
                          const element_type* _faces,
                          const int*          _active_idx,
                          AABB*               _bvs,
                          int                 n_active,
                          int                 type,
                          cudaStream_t        stream = 0)
{
    if(n_active < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (n_active + threadNum - 1) / threadNum;
    _calcLeafBvs_indirect<<<blockNum, threadNum, 0, stream>>>(
        _vertexes, _faces, _active_idx, _bvs + n_active - 1, n_active, type);
}

template <class element_type>
void calcLeafBvs_fullCCD_indirect(const double3*      _vertexes,
                                  const double3*      _moveDir,
                                  const double&       alpha,
                                  const element_type* _faces,
                                  const int*          _active_idx,
                                  AABB*               _bvs,
                                  int                 n_active,
                                  int                 type,
                                  cudaStream_t        stream = 0,
                                  const double*       alpha_dev = nullptr)
{
    if(n_active < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (n_active + threadNum - 1) / threadNum;
    _calcLeafBvs_ccd_indirect<<<blockNum, threadNum, 0, stream>>>(
        _vertexes, _moveDir, alpha, _faces, _active_idx, _bvs + n_active - 1, n_active, type, alpha_dev);
}

void calcLeafNodes_indirect(Node*           _nodes,
                            const uint32_t* _indices,
                            const int*      _active_idx,
                            int             n_active,
                            cudaStream_t    stream = 0)
{
    if(n_active < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (n_active + threadNum - 1) / threadNum;
    _calcLeafNodes_indirect<<<blockNum, threadNum, 0, stream>>>(_nodes, _indices, _active_idx, n_active);
}

void calcMChash(uint64_t* _MChash, AABB* _bvs, int number, const int* prim_env = nullptr,
                const int* prim_localid = nullptr, const double3* env_offset = nullptr,
                const uint32_t* prim_v0 = nullptr, cudaStream_t stream = 0)
{
    int numbers = number;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calcMChash<<<blockNum, threadNum, 0, stream>>>(_MChash, _bvs, number, prim_env, prim_localid, env_offset, prim_v0);
}

void calcLeafNodes(Node* _nodes, const uint32_t* _indices, int number)
{
    int numbers = number;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calcLeafNodes<<<blockNum, threadNum>>>(_nodes, _indices, number);
}

void calcInternalNodes(Node* _nodes, const uint64_t* _MChash, int number, cudaStream_t stream = 0)
{
    int numbers = number;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calcInternalNodes<<<blockNum, threadNum, 0, stream>>>(_nodes, _MChash, number);
}

void calcInternalAABB(const Node* _nodes, AABB* _bvs, uint32_t* flags, int number, cudaStream_t stream = 0)
{
    int numbers = number;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    //uint32_t* flags;
    //CUDA_SAFE_CALL(cudaMalloc((void**)&flags, (numbers-1) * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemsetAsync(flags, 0xFFFFFFFF, sizeof(uint32_t) * (numbers - 1), stream));
    _calcInternalAABB<<<blockNum, threadNum, 0, stream>>>(_nodes, _bvs, flags, numbers);
    //CUDA_SAFE_CALL(cudaFree(flags));
}

void sortBvs(const uint32_t* _indices, AABB* _bvs, AABB* _temp_bvs, int number, cudaStream_t stream = 0)
{
    int numbers = number;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    //AABB* _temp_bvs = _tempLeafBox;
    // CUDA_SAFE_CALL(cudaMalloc((void**)&_temp_bvs, (number) * sizeof(AABB)));
    cudaMemcpyAsync(_temp_bvs, _bvs + number - 1, sizeof(AABB) * number, cudaMemcpyDeviceToDevice, stream);
    _sortBvs<<<blockNum, threadNum, 0, stream>>>(_indices, _bvs + number - 1, _temp_bvs, number);
    //CUDA_SAFE_CALL(cudaFree(_temp_bvs));
}


void selfQuery_ee(const int*     _bodyID,
                  const int*     _btype,
                  const double3* _vertexes,
                  const double3* _rest_vertexes,
                  const uint2*   _edges,
                  const AABB*    _bvs,
                  const Node*    _nodes,
                  int4*          _collisonPairs,
                  int4*          _ccd_collisonPairs,
                  uint32_t*      _cpNum,
                  int*           MatIndex,
                  double         dHat,
                  int            number,
                  const int*     _collision_skip_matrix,
                  int            _collision_body_count,
                  const int*     _body_id_to_is_fem,
                  const int* node_env,
                  cudaStream_t   stream = 0)
{
    int numbers = number;
    if(numbers < 1)
        return;
    const unsigned int threadNum = 256;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    // [ee-lb] STIFF_EE_LB: 0/unset = baseline (168 reg, 8 warps/SM), 2 = ≤128 reg (16 warps),
    // 3 = ≤85 reg (24 warps). Same body — occupancy/spill A/B without swapping wheels.
    static int s_ee_lb = -1;
    if(s_ee_lb < 0)
    {
        const char* e = getenv("STIFF_EE_LB");
        s_ee_lb       = e ? atoi(e) : 0;
    }
    auto* kern = s_ee_lb == 2 ? _selfQuery_ee_lb2 : s_ee_lb == 3 ? _selfQuery_ee_lb3 : _selfQuery_ee;
    kern<<<blockNum, threadNum, 0, stream>>>(_bodyID,
                                           _btype,
                                           _vertexes,
                                           _rest_vertexes,
                                           _edges,
                                           _bvs,
                                           _nodes,
                                           _collisonPairs,
                                           _ccd_collisonPairs,
                                           _cpNum,
                                           MatIndex,
                                           dHat,
                                           numbers,
                                           _collision_skip_matrix,
                                           _collision_body_count,
                                           _body_id_to_is_fem, node_env);
}

void fullCCDselfQuery_ee(const int*     _bodyID,
                         const int*     _btype,
                         const double3* _vertexes,
                         const double3* moveDir,
                         const double&  alpha,
                         const uint2*   _edges,
                         const AABB*    _bvs,
                         const Node*    _nodes,
                         int4*          _ccd_collisonPairs,
                         uint32_t*      _cpNum,
                         double         dHat,
                         int            number,
                         const int*     _collision_skip_matrix,
                         int            _collision_body_count,
                         const int*     _body_id_to_is_fem,
                         const int* node_env,
                         cudaStream_t   stream = 0,
                         const double*  alpha_dev = nullptr)
{
    int numbers = number;
    if(numbers < 1)
        return;
    const unsigned int threadNum = 256;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    _selfQuery_ee_ccd<<<blockNum, threadNum, 0, stream>>>(
        _bodyID, _btype, _vertexes, moveDir, alpha, _edges, _bvs, _nodes, _ccd_collisonPairs, _cpNum, dHat, numbers,
        _collision_skip_matrix, _collision_body_count, _body_id_to_is_fem, node_env, alpha_dev);
}

void selfQuery_vf(const int*      _bodyID,
                  const int*      _btype,
                  const double3*  _vertexes,
                  const uint3*    _faces,
                  const uint32_t* _surfVerts,
                  const AABB*     _bvs,
                  const Node*     _nodes,
                  int4*           _collisonPairs,
                  int4*           _ccd_collisonPairs,
                  uint32_t*       _cpNum,
                  int*            MatIndex,
                  double          dHat,
                  int             number,
                  const int*      _collision_skip_matrix,
                  int             _collision_body_count,
                  const int*      _body_id_to_is_fem,
                  cudaStream_t    stream = 0)
{
    int numbers = number;
    if(numbers < 1)
        return;
    const unsigned int threadNum = 256;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    _selfQuery_vf<<<blockNum, threadNum, 0, stream>>>(_bodyID,
                                           _btype,
                                           _vertexes,
                                           _faces,
                                           _surfVerts,
                                           _bvs,
                                           _nodes,
                                           _collisonPairs,
                                           _ccd_collisonPairs,
                                           _cpNum,
                                           MatIndex,
                                           dHat,
                                           numbers,
                                           _collision_skip_matrix,
                                           _collision_body_count,
                                           _body_id_to_is_fem);
}

void fullCCDselfQuery_vf(const int*      _bodyID,
                         const int*      _btype,
                         const double3*  _vertexes,
                         const double3*  moveDir,
                         const double&   alpha,
                         const uint3*    _faces,
                         const uint32_t* _surfVerts,
                         const AABB*     _bvs,
                         const Node*     _nodes,
                         int4*           _ccd_collisonPairs,
                         uint32_t*       _cpNum,
                         double          dHat,
                         int             number,
                         const int*      _collision_skip_matrix,
                         int             _collision_body_count,
                         const int*      _body_id_to_is_fem,
                         cudaStream_t    stream = 0,
                         const double*   alpha_dev = nullptr)
{
    int numbers = number;
    if(numbers < 1)
        return;
    const unsigned int threadNum = 256;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    _selfQuery_vf_ccd<<<blockNum, threadNum, 0, stream>>>(
        _bodyID, _btype, _vertexes, moveDir, alpha, _faces, _surfVerts, _bvs, _nodes, _ccd_collisonPairs, _cpNum, dHat, numbers,
        _collision_skip_matrix, _collision_body_count, _body_id_to_is_fem, alpha_dev);
}

void lbvh::FREE_DEVICE_MEM()
{
    CUDA_SAFE_CALL(cudaFree(_indices));
    CUDA_SAFE_CALL(cudaFree(_MChash));
    CUDA_SAFE_CALL(cudaFree(_nodes));
    CUDA_SAFE_CALL(cudaFree(_bvs));
    CUDA_SAFE_CALL(cudaFree(_flags));
    CUDA_SAFE_CALL(cudaFree(_tempLeafBox));
}

void lbvh::MALLOC_DEVICE_MEM(const int& number)
{
    CUDA_SAFE_CALL(cudaMalloc((void**)&_indices, (number) * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_MChash, (number) * sizeof(uint64_t)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_nodes, (2 * number - 1) * sizeof(Node)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_node_env, (2 * number - 1) * sizeof(int)));  // [env-part B]
    CUDA_SAFE_CALL(cudaMalloc((void**)&_bvs, (2 * number - 1) * sizeof(AABB)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_tempLeafBox, number * sizeof(AABB)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_flags, (number - 1) * sizeof(uint32_t)));
    //CUDA_SAFE_CALL(cudaMalloc((void**)&_cpNum, sizeof(uint32_t)));ye
    //CUDA_SAFE_CALL(cudaMemset(_cpNum, 0, sizeof(uint32_t)));
}

lbvh::~lbvh()
{
    //FREE_DEVICE_MEM();
}


void lbvh_f::init(int*       _mbodyID,
                  int*       _mbtype,
                  double3*   _mVerts,
                  uint3*     _mFaces,
                  uint32_t*  _mSurfVert,
                  int4*      _mCollisonPairs,
                  int4*      _ccd_mCollisonPairs,
                  uint32_t*  _mcpNum,
                  int*       _mMatIndex,
                  const int& faceNum,
                  const int& vertNum,
                  int*       collision_skip_matrix,
                  int        collision_body_count)
{
    _bodyId            = _mbodyID;
    _faces             = _mFaces;
    _surfVerts         = _mSurfVert;
    _vertexes          = _mVerts;
    _collisionPair     = _mCollisonPairs;
    _ccd_collisionPair = _ccd_mCollisonPairs;
    _cpNum             = _mcpNum;
    _MatIndex          = _mMatIndex;
    face_number        = faceNum;
    vert_number        = vertNum;
    _btype             = _mbtype;
    _collision_skip_matrix = collision_skip_matrix;
    _collision_body_count  = collision_body_count;
    MALLOC_DEVICE_MEM(face_number);
}

void lbvh_e::init(int*       _mbodyID,
                  int*       _mbtype,
                  double3*   _mVerts,
                  double3*   _mRest_vertexes,
                  uint2*     _mEdges,
                  int4*      _mCollisonPairs,
                  int4*      _ccd_mCollisonPairs,
                  uint32_t*  _mcpNum,
                  int*       _mMatIndex,
                  const int& edgeNum,
                  const int& vertNum,
                  int*       collision_skip_matrix,
                  int        collision_body_count)
{
    _bodyId            = _mbodyID;
    _rest_vertexes     = _mRest_vertexes;
    _edges             = _mEdges;
    _vertexes          = _mVerts;
    _cpNum             = _mcpNum;
    _collisionPair     = _mCollisonPairs;
    _ccd_collisionPair = _ccd_mCollisonPairs;
    _MatIndex          = _mMatIndex;
    edge_number        = edgeNum;
    vert_number        = vertNum;
    _btype             = _mbtype;
    _collision_skip_matrix = collision_skip_matrix;
    _collision_body_count  = collision_body_count;
    MALLOC_DEVICE_MEM(edge_number);
}

AABB* lbvh_f::getSceneSize()
{
    calcLeafBvs(_vertexes, _faces, _bvs, face_number, 0,
                _bodyId, _collision_skip_matrix, _collision_body_count);

    calcMaxBV(_bvs, _tempLeafBox, face_number);
    return _bvs;
}

double lbvh_f::Construct(cudaStream_t stream)
{
    // BVH-skip #3: when _active_idx is set and shrinks the input, build BVH on
    // n_active leaves instead of full face_number — saves work in calcMaxBV,
    // sort_by_key, sortBvs, internal-node + AABB passes proportional to (1 - n_active/face_number).
    if(_active_idx != nullptr && face_number_active > 0
       && face_number_active <= (int)face_number)
    {
        const int N = face_number_active;
        // [perenv-parallel #1] fully async on `stream` (no host sync) so per-env builds overlap.
        calcLeafBvs_indirect(_vertexes, _faces, _active_idx, _bvs, N, 0, stream);
        calcMaxBV_async(_bvs, _tempLeafBox, N, stream);
        calcMChash(_MChash, _bvs, N, nullptr, nullptr, nullptr, nullptr, stream);
        _iota_u32<<<(N + 255) / 256, 256, 0, stream>>>(_indices, N);
        _mc_sort_active(*this, _MChash, _indices, N, stream);
        sortBvs(_indices, _bvs, _tempLeafBox, N, stream);
        calcLeafNodes_indirect(_nodes, _indices, _active_idx, N, stream);
        calcInternalNodes(_nodes, _MChash, N, stream);
        calcInternalAABB(_nodes, _bvs, _flags, N, stream);
        return 0;
    }
    calcLeafBvs(_vertexes, _faces, _bvs, face_number, 0,
                _bodyId, _collision_skip_matrix, _collision_body_count);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    scene = calcMaxBV(_bvs, _tempLeafBox, face_number);
    calcMChash(_MChash, _bvs, face_number, m_prim_env, m_prim_localid, m_env_offset, m_prim_v0);
    thrust::sequence(thrust::device_ptr<uint32_t>(_indices),
                     thrust::device_ptr<uint32_t>(_indices) + face_number);
    thrust::sort_by_key(thrust::device_ptr<uint64_t>(_MChash),
                        thrust::device_ptr<uint64_t>(_MChash) + face_number,
                        thrust::device_ptr<uint32_t>(_indices));
    sortBvs(_indices, _bvs, _tempLeafBox, face_number);
    calcLeafNodes(_nodes, _indices, face_number);
    calcInternalNodes(_nodes, _MChash, face_number);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    calcInternalAABB(_nodes, _bvs, _flags, face_number);
    computeNodeEnv(m_node_env, _nodes, m_prim_env, _flags, face_number);  // [env-part B]
    return 0;  //time0 + time1 + time2;
}

double lbvh_f::ConstructFullCCD(const double3* moveDir, const double& alpha, cudaStream_t stream,
                                const double* alpha_dev)
{
    if(_active_idx != nullptr && face_number_active > 0
       && face_number_active <= (int)face_number)
    {
        const int N = face_number_active;
        // [perenv-parallel #2] fully async on `stream` (mirrors the DCD active path): swept-leaf
        // build -> async max-BV -> Morton -> cub sort (pre-alloc scratch) -> tree. No host sync,
        // no malloc/free -> concurrent per-env swept builds+queries actually overlap.
        calcLeafBvs_fullCCD_indirect(_vertexes, moveDir, alpha, _faces,
                                     _active_idx, _bvs, N, 0, stream, alpha_dev);
        calcMaxBV_async(_bvs, _tempLeafBox, N, stream);
        calcMChash(_MChash, _bvs, N, nullptr, nullptr, nullptr, nullptr, stream);
        _iota_u32<<<(N + 255) / 256, 256, 0, stream>>>(_indices, N);
        _mc_sort_active(*this, _MChash, _indices, N, stream);
        sortBvs(_indices, _bvs, _tempLeafBox, N, stream);
        calcLeafNodes_indirect(_nodes, _indices, _active_idx, N, stream);
        calcInternalNodes(_nodes, _MChash, N, stream);
        calcInternalAABB(_nodes, _bvs, _flags, N, stream);
        return 0;
    }
    calcLeafBvs_fullCCD(_vertexes, moveDir, alpha, _faces, _bvs, face_number, 0,
                        _bodyId, _collision_skip_matrix, _collision_body_count,
                        alpha_dev);
    scene = calcMaxBV(_bvs, _tempLeafBox, face_number);
    calcMChash(_MChash, _bvs, face_number, m_prim_env, m_prim_localid, m_env_offset, m_prim_v0);
    thrust::sequence(thrust::device_ptr<uint32_t>(_indices),
                     thrust::device_ptr<uint32_t>(_indices) + face_number);

    thrust::sort_by_key(thrust::device_ptr<uint64_t>(_MChash),
                        thrust::device_ptr<uint64_t>(_MChash) + face_number,
                        thrust::device_ptr<uint32_t>(_indices));
    sortBvs(_indices, _bvs, _tempLeafBox, face_number);

    calcLeafNodes(_nodes, _indices, face_number);

    calcInternalNodes(_nodes, _MChash, face_number);
    calcInternalAABB(_nodes, _bvs, _flags, face_number);
    computeNodeEnv(m_node_env, _nodes, m_prim_env, _flags, face_number);  // [env-part B]

    return 0;
}

double lbvh_e::Construct(cudaStream_t stream)
{
    // BVH-skip #3: when _active_idx is set (face_number_active reused as edge active count)
    if(_active_idx != nullptr && face_number_active > 0
       && face_number_active <= (int)edge_number)
    {
        const int N = face_number_active;
        // [perenv-parallel #1] fully async on `stream` (no host sync) so per-env builds overlap.
        calcLeafBvs_indirect(_vertexes, _edges, _active_idx, _bvs, N, 1, stream);
        calcMaxBV_async(_bvs, _tempLeafBox, N, stream);
        calcMChash(_MChash, _bvs, N, nullptr, nullptr, nullptr, nullptr, stream);
        _iota_u32<<<(N + 255) / 256, 256, 0, stream>>>(_indices, N);
        _mc_sort_active(*this, _MChash, _indices, N, stream);
        sortBvs(_indices, _bvs, _tempLeafBox, N, stream);
        calcLeafNodes_indirect(_nodes, _indices, _active_idx, N, stream);
        calcInternalNodes(_nodes, _MChash, N, stream);
        calcInternalAABB(_nodes, _bvs, _flags, N, stream);
        return 0;
    }

    /*cudaEvent_t start, end0, end1, end2;
    cudaEventCreate(&start);
    cudaEventCreate(&end0);
    cudaEventCreate(&end1);
    cudaEventCreate(&end2);

    cudaEventRecord(start);*/
    calcLeafBvs(_vertexes, _edges, _bvs, edge_number, 1,
                _bodyId, _collision_skip_matrix, _collision_body_count);
    scene = calcMaxBV(_bvs, _tempLeafBox, edge_number);
    calcMChash(_MChash, _bvs, edge_number, m_prim_env, m_prim_localid, m_env_offset, m_prim_v0);
    thrust::sequence(thrust::device_ptr<uint32_t>(_indices),
                     thrust::device_ptr<uint32_t>(_indices) + edge_number);
    //cudaEventRecord(end0);

    thrust::sort_by_key(thrust::device_ptr<uint64_t>(_MChash),
                        thrust::device_ptr<uint64_t>(_MChash) + edge_number,
                        thrust::device_ptr<uint32_t>(_indices));
    sortBvs(_indices, _bvs, _tempLeafBox, edge_number);

    //cudaEventRecord(end1);

    calcLeafNodes(_nodes, _indices, edge_number);

    calcInternalNodes(_nodes, _MChash, edge_number);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    calcInternalAABB(_nodes, _bvs, _flags, edge_number);
    computeNodeEnv(m_node_env, _nodes, m_prim_env, _flags, edge_number);  // [env-part B]
    //selfQuery(_vertexes, _edges, _bvs, _nodes, _collisionPair, _cpNum, edge_number);
    //cudaEventRecord(end2);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    /*float time0 = 0, time1 = 0, time2 = 0;
    cudaEventElapsedTime(&time0, start, end0);
    cudaEventElapsedTime(&time1, end0, end1);
    cudaEventElapsedTime(&time2, end1, end2);
    (cudaEventDestroy(start));
    (cudaEventDestroy(end0));
    (cudaEventDestroy(end1));
    (cudaEventDestroy(end2));*/
    //std::cout << "sort time: " << time1 << std::endl;
    return 0;  //time0 + time1 + time2;
    //std::cout << "generation done: " << time0 + time1 + time2 << std::endl;
}

double lbvh_e::ConstructFullCCD(const double3* moveDir, const double& alpha, cudaStream_t stream,
                                const double* alpha_dev)
{
    if(_active_idx != nullptr && face_number_active > 0
       && face_number_active <= (int)edge_number)
    {
        const int N = face_number_active;
        // [perenv-parallel #2] fully async on `stream` (mirrors the DCD active path).
        calcLeafBvs_fullCCD_indirect(_vertexes, moveDir, alpha, _edges,
                                     _active_idx, _bvs, N, 1, stream, alpha_dev);
        calcMaxBV_async(_bvs, _tempLeafBox, N, stream);
        calcMChash(_MChash, _bvs, N, nullptr, nullptr, nullptr, nullptr, stream);
        _iota_u32<<<(N + 255) / 256, 256, 0, stream>>>(_indices, N);
        _mc_sort_active(*this, _MChash, _indices, N, stream);
        sortBvs(_indices, _bvs, _tempLeafBox, N, stream);
        calcLeafNodes_indirect(_nodes, _indices, _active_idx, N, stream);
        calcInternalNodes(_nodes, _MChash, N, stream);
        calcInternalAABB(_nodes, _bvs, _flags, N, stream);
        return 0;
    }
    calcLeafBvs_fullCCD(_vertexes, moveDir, alpha, _edges, _bvs, edge_number, 1,
                        _bodyId, _collision_skip_matrix, _collision_body_count,
                        alpha_dev);
    scene = calcMaxBV(_bvs, _tempLeafBox, edge_number);
    calcMChash(_MChash, _bvs, edge_number, m_prim_env, m_prim_localid, m_env_offset, m_prim_v0);
    thrust::sequence(thrust::device_ptr<uint32_t>(_indices),
                     thrust::device_ptr<uint32_t>(_indices) + edge_number);

    thrust::sort_by_key(thrust::device_ptr<uint64_t>(_MChash),
                        thrust::device_ptr<uint64_t>(_MChash) + edge_number,
                        thrust::device_ptr<uint32_t>(_indices));
    sortBvs(_indices, _bvs, _tempLeafBox, edge_number);

    calcLeafNodes(_nodes, _indices, edge_number);

    calcInternalNodes(_nodes, _MChash, edge_number);

    calcInternalAABB(_nodes, _bvs, _flags, edge_number);
    computeNodeEnv(m_node_env, _nodes, m_prim_env, _flags, edge_number);  // [env-part B]

    return 0;
}


void lbvh_f::SelfCollitionDetect(double dHat, cudaStream_t stream)
{

    selfQuery_vf(_bodyId,
                 _btype,
                 _vertexes,
                 _faces,
                 _surfVerts,
                 _bvs,
                 _nodes,
                 _collisionPair,
                 _ccd_collisionPair,
                 _cpNum,
                 _MatIndex,
                 dHat,
                 vert_number,
                 _collision_skip_matrix,
                 _collision_body_count,
                 _body_id_to_is_fem,
                 stream);
}

void lbvh_e::SelfCollitionDetect(double dHat, cudaStream_t stream)
{
    // BVH-skip #3: EE self-query reads leaves at offset [N-1, 2N-1). When
    // indirect BVH is active, leaves live at [n_active-1, 2*n_active-1), so
    // we must launch with N = n_active edges, not the full edge_number.
    int N = (_active_idx != nullptr && face_number_active > 0
             && face_number_active <= (int)edge_number)
                ? face_number_active
                : (int)edge_number;
    selfQuery_ee(_bodyId,
                 _btype,
                 _vertexes,
                 _rest_vertexes,
                 _edges,
                 _bvs,
                 _nodes,
                 _collisionPair,
                 _ccd_collisionPair,
                 _cpNum,
                 _MatIndex,
                 dHat,
                 N,
                 _collision_skip_matrix,
                 _collision_body_count,
                 _body_id_to_is_fem,
                 m_node_env,
                 stream);
}

void lbvh_f::SelfCollitionFullDetect(double dHat, const double3* moveDir, const double& alpha,
                                     cudaStream_t stream, const double* alpha_dev)
{

    fullCCDselfQuery_vf(
        _bodyId, _btype, _vertexes, moveDir, alpha, _faces, _surfVerts, _bvs, _nodes, _ccd_collisionPair, _cpNum, dHat, vert_number,
        _collision_skip_matrix, _collision_body_count, _body_id_to_is_fem, stream, alpha_dev);
}

void lbvh_e::SelfCollitionFullDetect(double dHat, const double3* moveDir, const double& alpha,
                                     cudaStream_t stream, const double* alpha_dev)
{
    // Same fix as SelfCollitionDetect: launch count must match leaf count.
    int N = (_active_idx != nullptr && face_number_active > 0
             && face_number_active <= (int)edge_number)
                ? face_number_active
                : (int)edge_number;
    fullCCDselfQuery_ee(
        _bodyId, _btype, _vertexes, moveDir, alpha, _edges, _bvs, _nodes, _ccd_collisionPair, _cpNum, dHat, N,
        _collision_skip_matrix, _collision_body_count, _body_id_to_is_fem, m_node_env, stream, alpha_dev);
}


//#include <cstdio>
//#include <cstdlib>
//#include <vector>
//
//#include <cuda_runtime.h>
//#include <cusolverDn.h>
//#include <random>
//
//#include <cstdlib>
//
//int main2() {
//    cusolverDnHandle_t cusolverH = NULL;
//    cudaStream_t stream = NULL;
//
//    const int m = 12;
//    const int lda = m;
//    /*
//     *       | 3.5 0.5 0.0 |
//     *   A = | 0.5 3.5 0.0 |
//     *       | 0.0 0.0 2.0 |
//     *
//     */
//    std::vector<double> A;// = { 3.5, 0.5, 0.0, 0.5, 3.5, 0.0, 0.0, 0.0, 2.0 };
//    //const std::vector<double> lambda = { 2.0, 3.0, 4.0 };
//    for (int i = 0;i < m;i++) {
//        for (int j = 0;j < m;j++) {
//            A.push_back((double)rand() / RAND_MAX);
//        }
//    }
//
//    std::vector<double> V(lda * m, 0); // eigenvectors
//    std::vector<double> W(m, 0);       // eigenvalues
//
//    double* d_A = nullptr;
//    double* d_W = nullptr;
//    int* d_info = nullptr;
//
//    int info = 0;
//
//    int lwork = 0;            /* size of workspace */
//    double* d_work = nullptr; /* device workspace*/
//
//    std::printf("A = (matlab base-1)\n");
//    //print_matrix(m, m, A.data(), lda);
//    std::printf("=====\n");
//
//    cudaEvent_t start, end0;
//    cudaEventCreate(&start);
//    cudaEventCreate(&end0);
//
//
//    /* step 1: create cusolver handle, bind a stream */
//    (cusolverDnCreate(&cusolverH));
//
//    (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
//    (cusolverDnSetStream(cusolverH, stream));
//
//    (cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(double) * A.size()));
//    (cudaMalloc(reinterpret_cast<void**>(&d_W), sizeof(double) * W.size()));
//    (cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int)));
//
//    (
//        cudaMemcpyAsync(d_A, A.data(), sizeof(double) * A.size(), cudaMemcpyHostToDevice, stream));
//
//    // step 3: query working space of syevd
//    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
//    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
//    cudaEventRecord(start);
//    (cusolverDnDsyevd_bufferSize(cusolverH, jobz, uplo, m, d_A, lda, d_W, &lwork));
//
//    (cudaMalloc(reinterpret_cast<void**>(&d_work), sizeof(double) * lwork));
//
//    // step 4: compute spectrum
//    (
//        cusolverDnDsyevd(cusolverH, jobz, uplo, m, d_A, lda, d_W, d_work, lwork, d_info));
//    cudaEventRecord(end0);
//    (
//        cudaMemcpyAsync(V.data(), d_A, sizeof(double) * V.size(), cudaMemcpyDeviceToHost, stream));
//    (
//        cudaMemcpyAsync(W.data(), d_W, sizeof(double) * W.size(), cudaMemcpyDeviceToHost, stream));
//    (cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));
//
//    (cudaStreamSynchronize(stream));
//
//
//
//    CUDA_SAFE_CALL(cudaDeviceSynchronize());
//
//    float time0 = 0, time1 = 0, time2 = 0;
//    cudaEventElapsedTime(&time0, start, end0);
//
//    (cudaEventDestroy(start));
//    (cudaEventDestroy(end0));
//
//    std::printf("after syevd: info = %d  %f\n", info, time0);
//    if (0 > info) {
//        std::printf("%d-th parameter is wrong \n", -info);
//        exit(1);
//    }
//
//    std::printf("eigenvalue = (matlab base-1), ascending order\n");
//    int idx = 1;
//    for (auto const& i : W) {
//        std::printf("W[%i] = %E\n", idx, i);
//        idx++;
//    }
//
//
//    (cudaFree(d_A));
//    (cudaFree(d_W));
//    (cudaFree(d_info));
//    (cudaFree(d_work));
//
//    (cusolverDnDestroy(cusolverH));
//
//    (cudaStreamDestroy(stream));
//
//    (cudaDeviceReset());
//
//    return EXIT_SUCCESS;
//}

// [liquid] see liquid_system.cuh
#include "liquid_system.cuh"
#include "liquid_validation.h"
#include "device_fem_data.cuh"
#include <linear_system/linear_system/global_matrix.h>
#include <cuda_tools/cuda_tools.h>
#include <linear_system/utils/binned_reduce.cuh>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <Eigen/Dense>
#include <stdexcept>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <fstream>

extern __device__ double* g_gbin;  // binned elastic-gradient accumulator (GIPC.cu)

namespace gipc_liquid
{

// ----------------------------------------------------------------------------
// lattice helpers
// ----------------------------------------------------------------------------
__device__ __host__ __forceinline__ int cell_lin(int i, int j, int k, int3 d)
{
    return (i * d.y + j) * d.z + k;
}
__device__ __host__ __forceinline__ int node_lin(int i, int j, int k, int3 d)
{
    return (i * (d.y + 1) + j) * (d.z + 1) + k;
}
__device__ __host__ __forceinline__ void cell_ijk(int c, int3 d, int& i, int& j, int& k)
{
    k = c % d.z;
    int t = c / d.z;
    j = t % d.y;
    i = t / d.y;
}
__device__ __host__ __forceinline__ void node_ijk(int n, int3 d, int& i, int& j, int& k)
{
    k = n % (d.z + 1);
    int t = n / (d.z + 1);
    j = t % (d.y + 1);
    i = t / (d.y + 1);
}
// corner q = a*4 + b*2 + c  ->  sign vector (2a-1, 2b-1, 2c-1)
__device__ __forceinline__ double3 corner_sign(int q)
{
    return make_double3((q & 4) ? 1.0 : -1.0, (q & 2) ? 1.0 : -1.0, (q & 1) ? 1.0 : -1.0);
}

__device__ __forceinline__ double3 d3add(double3 a, double3 b) { return make_double3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __forceinline__ double3 d3sub(double3 a, double3 b) { return make_double3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ __forceinline__ double3 d3scale(double3 a, double s) { return make_double3(a.x * s, a.y * s, a.z * s); }
__device__ __forceinline__ bool d3finite(double3 a) { return isfinite(a.x) && isfinite(a.y) && isfinite(a.z); }
// clamp / fallback event counters (traceability: a run that silently clamps J or positions must not pass a physics test)
__device__ int* g_liq_fbk = nullptr;
#define LIQ_FBK(i) do { if(g_liq_fbk) atomicAdd(&g_liq_fbk[(i)], 1); } while(0)
// debug switches are NUMERIC: LIQ_NO_EXPAND=0 must not enable the ablation (the old `getenv != nullptr` test did)
static bool liq_env_flag(const char* k) { const char* v = getenv(k); return v != nullptr && atof(v) != 0.0; }

// particle -> cell-centre trilinear weights (8 centres), with clamping to the
// interior so that every particle always has 8 valid centres.
struct CentreStencil
{
    int    b[3];
    double w[3][2];
};
__device__ __forceinline__ CentreStencil centre_stencil(double3 x, double3 org, double dx, int3 d)
{
    CentreStencil s;
    double r[3] = {(x.x - org.x) / dx - 0.5, (x.y - org.y) / dx - 0.5, (x.z - org.z) / dx - 0.5};
    int    n[3] = {d.x, d.y, d.z};
    for(int a = 0; a < 3; ++a)
    {
        int    bi = (int)floor(r[a]);
        double f  = r[a] - bi;
        if(bi < 0)
        {
            bi = 0;
            f  = 0.0;
        }
        if(bi > n[a] - 2)
        {
            bi = n[a] - 2;
            f  = 1.0;
        }
        s.b[a]    = bi;
        s.w[a][0] = 1.0 - f;
        s.w[a][1] = f;
    }
    return s;
}

// ----------------------------------------------------------------------------
// particle embedding: 27-node stencil, separable weights (f0/2, 1/2, f1/2)
// ----------------------------------------------------------------------------
struct NodeStencil
{
    int    b[3];
    double w[3][3];
};
__device__ int g_liq_tri = 0;   // LIQ_UG: the embedding is the trilinear stencil of the particle's cell, packed into the 27-node layout
__device__ __forceinline__ NodeStencil node_stencil(int3 base, double3 frac)
{
    NodeStencil s;
    s.b[0] = base.x; s.b[1] = base.y; s.b[2] = base.z;
    double f[3] = {frac.x, frac.y, frac.z};
    if(g_liq_tri == 2)
    {   // LIQ_UG_BSPLINE: quadratic B-spline at node coordinate b + 1/2 + f (= ug_weights_q of liquid_ug.cuh)
        for(int a = 0; a < 3; ++a)
        {
            s.w[a][0] = 0.5 * (1.0 - f[a]) * (1.0 - f[a]);
            s.w[a][1] = 0.75 - (f[a] - 0.5) * (f[a] - 0.5);
            s.w[a][2] = 0.5 * f[a] * f[a];
        }
        return s;
    }
    if(g_liq_tri)
    {   // particle at node coordinate b + 1/2 + f: nodes (b, b+1) for f < 1/2, (b+1, b+2) otherwise
        for(int a = 0; a < 3; ++a)
        {
            if(f[a] < 0.5) { s.w[a][0] = 0.5 - f[a]; s.w[a][1] = 0.5 + f[a]; s.w[a][2] = 0.0; }
            else           { s.w[a][0] = 0.0; s.w[a][1] = 1.5 - f[a]; s.w[a][2] = f[a] - 0.5; }
        }
        return s;
    }
    for(int a = 0; a < 3; ++a)
    {
        s.w[a][0] = 0.5 * (1.0 - f[a]);
        s.w[a][1] = 0.5;
        s.w[a][2] = 0.5 * f[a];
    }
    return s;
}
// CPIC-style compatibility mask: bit k = node (a,b,c) of the 27-node stencil, k = (a*3+b)*3+c
#define LIQ_MASK_ALL 0x7FFFFFF
__device__ __forceinline__ double masked_w(const NodeStencil& st, int a, int b, int c, int mask, double winv)
{
    int k = (a * 3 + b) * 3 + c;
    return ((mask >> k) & 1) ? st.w[0][a] * st.w[1][b] * st.w[2][c] * winv : 0.0;
}
__device__ __forceinline__ NodeStencil node_stencil_of(const CentreStencil& s)
{
    return node_stencil(make_int3(s.b[0], s.b[1], s.b[2]), make_double3(s.w[0][1], s.w[1][1], s.w[2][1]));
}
// per-side node slots: the copy of node nid a particle at x uses (0 = '+' side of the node's reference plane / no plane)
// label of a particle at a node: its own label if it has one (near wall); a far particle takes the node's single label,
// or the geometric side of the node's reference plane when the node carries both labels (two-sided: thin wall)
__device__ __forceinline__ int vside_l(int nid, double3 x, int plab, const int* nlab, const int* has, const double3* X, const double3* nn, int stride)
{
    if(stride <= 0) return 0;
    if(plab >= 0) return plab;
    int L = nlab ? nlab[nid] : 0;
    if(L == 2) return 1;
    if(L == 3 && has && has[nid]) return (nn[nid].x * (x.x - X[nid].x) + nn[nid].y * (x.y - X[nid].y) + nn[nid].z * (x.z - X[nid].z) < 0.0) ? 1 : 0;
    return 0;
}
__device__ __forceinline__ int vnode(int nid, double3 x, const int* has, const double3* X, const double3* nn, int stride, int plab = -1, const int* nlab = nullptr)
{
    if(stride <= 0) return nid;
    return nid + vside_l(nid, x, plab, nlab, has, X, nn, stride) * stride;
}
__device__ __forceinline__ int vside(int nid, double3 x, const int* has, const double3* X, const double3* nn, int stride, int plab = -1, const int* nlab = nullptr)
{
    return vside_l(nid, x, plab, nlab, has, X, nn, stride);
}
__global__ void k_plabel(int n, const int* near, const double3* wn, int* plab)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    if(!near[p]) { plab[p] = -1; return; }
    double3 v = wn[p];
    double  ax = fabs(v.x), ay = fabs(v.y), az = fabs(v.z);
    double  dom = (ax >= ay && ax >= az) ? v.x : ((ay >= az) ? v.y : v.z);
    plab[p] = dom < 0.0 ? 1 : 0;
}
__global__ void k_nlab(int n, const int* near, const int* plab, const int3* pbase, int3 dims, int* nlab)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n || !near[p]) return;
    int3 b = pbase[p];
    int  bit = 1 << plab[p];
    for(int a = 0; a < 3; ++a)
        for(int bb = 0; bb < 3; ++bb)
            for(int c = 0; c < 3; ++c) atomicOr(&nlab[node_lin(b.x + a, b.y + bb, b.z + c, dims)], bit);
}
// node reference planes: among the near particles whose 27-node stencil contains the node, the one whose wall point is
// CLOSEST to the node (a plate node takes the plate, not the floor a few cells away); ties broken by particle index
__global__ void k_nref_min(int n, const int* near, const int3* pbase, const double3* wX, int3 dims, double3 org, double dx, unsigned long long* key)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n || !near[p]) return;
    int3    b = pbase[p];
    double3 X = wX[p];
    for(int a = 0; a < 3; ++a)
        for(int bb = 0; bb < 3; ++bb)
            for(int c = 0; c < 3; ++c)
            {
                double3 xn = make_double3(org.x + (b.x + a) * dx, org.y + (b.y + bb) * dx, org.z + (b.z + c) * dx);
                double  d  = sqrt((xn.x - X.x) * (xn.x - X.x) + (xn.y - X.y) * (xn.y - X.y) + (xn.z - X.z) * (xn.z - X.z)) / dx;
                unsigned long long q = (unsigned long long)fmin(d * 1.0e6, 4.0e9);
                atomicMin(&key[node_lin(b.x + a, b.y + bb, b.z + c, dims)], (q << 32) | (unsigned long long)p);
            }
}
__global__ void k_nref_fill(int nnode, const unsigned long long* key, const double3* wX, const double3* wn, int* has, double3* X, double3* nn)
{
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if(nid >= nnode) return;
    unsigned long long kk = key[nid];
    if(kk == 0xffffffffffffffffULL) { has[nid] = 0; return; }
    int p = (int)(kk & 0xffffffffULL);
    double3 v = wn[p];
    double  ax = fabs(v.x), ay = fabs(v.y), az = fabs(v.z);
    double  dom = (ax >= ay && ax >= az) ? v.x : ((ay >= az) ? v.y : v.z);
    if(dom < 0.0) v = make_double3(-v.x, -v.y, -v.z);   // normalise the orientation: positive dominant axis
    has[nid] = 1; X[nid] = wX[p]; nn[nid] = v;
}
__global__ void k_count_twosided(int nnode, const int* flag, int* out)
{
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if(nid >= nnode) return;
    if(flag[nid] && flag[nid + nnode]) atomicAdd(out, 1);
}
// nearest-wall side test for the particle's 27 nodes: compatible iff the node lies on the particle's side of the
// local wall plane (X_w, n). Particles without a wall point keep the full stencil. Weights are renormalised.
__global__ void k_build_mask(int n, const double3* xp, const int* near, const double3* wX, const double3* wn, const int3* pbase, const double3* pfrac,
                             double3 org, double dx, int* mask, double* winv, int* nmasked)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    if(!near || !near[p]) { mask[p] = LIQ_MASK_ALL; winv[p] = 1.0; return; }
    NodeStencil st = node_stencil(pbase[p], pfrac[p]);
    double3     X = wX[p], nn = wn[p];
    int         m = 0;
    double      ws = 0.0, wall = 0.0;
    for(int a = 0; a < 3; ++a)
        for(int b = 0; b < 3; ++b)
            for(int c = 0; c < 3; ++c)
            {
                double  w  = st.w[0][a] * st.w[1][b] * st.w[2][c];
                double3 xi = make_double3(org.x + (st.b[0] + a) * dx, org.y + (st.b[1] + b) * dx, org.z + (st.b[2] + c) * dx);
                double  d  = nn.x * (xi.x - X.x) + nn.y * (xi.y - X.y) + nn.z * (xi.z - X.z);
                wall += w;
                if(d >= -1e-12) { m |= 1 << ((a * 3 + b) * 3 + c); ws += w; }
            }
    if(ws <= 1e-12 * wall)
    {   // no compatible node (particle inside the solid or degenerate normal): keep the full stencil, count it
        mask[p] = LIQ_MASK_ALL; winv[p] = 1.0; LIQ_FBK(9); return;
    }
    mask[p] = m; winv[p] = 1.0 / ws;
    if(m != LIQ_MASK_ALL && nmasked) atomicAdd(nmasked, 1);
}
// 2b: per-cell cut axis and per-side wall data. Pass 1 sums |n| of the near-wall particles over their 8 centre-stencil
// cells; the cell's dominant axis e defines its two sides (sign of n[e]). Pass 2 accumulates the weighted wall point and
// normal of each side (for the corner-compatibility test of the per-side voxel elements).
__global__ void k_side_nabs(int n, const double3* xp, const int* near, const double3* wn, double3 org, double dx, int3 dims, double3* nabs)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n || !near[p]) return;
    CentreStencil s = centre_stencil(xp[p], org, dx, dims);
    double3 nn = wn[p];
    for(int a = 0; a < 2; ++a)
        for(int b = 0; b < 2; ++b)
            for(int c = 0; c < 2; ++c)
            {
                int cid = cell_lin(s.b[0] + a, s.b[1] + b, s.b[2] + c, dims);
                atomicAdd(&nabs[cid].x, fabs(nn.x)); atomicAdd(&nabs[cid].y, fabs(nn.y)); atomicAdd(&nabs[cid].z, fabs(nn.z));
            }
}
__global__ void k_cell_axis(int ncell, const double3* nabs, int* axis)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if(c >= ncell) return;
    double3 v = nabs[c];
    if(v.x + v.y + v.z <= 0.0) { axis[c] = -1; return; }
    axis[c] = (v.x >= v.y && v.x >= v.z) ? 0 : ((v.y >= v.z) ? 1 : 2);
}
__device__ __forceinline__ int cell_side(const int* axis, int cid, double3 nn)
{
    int e = axis[cid];
    if(e < 0) return 0;
    double ne = (e == 0) ? nn.x : ((e == 1) ? nn.y : nn.z);
    return ne >= 0.0 ? 0 : 1;
}
// masked direct 27-node transfer of mass and (APIC) momentum: algebraically identical to the cell-centre route
// (particle -> 8 centres -> 8 corners) when nothing is masked
__global__ void k_p2g_nodes(int n, const double3* xp, const double3* vp, const double* Cp, const double* mp, const int3* pbase, const double3* pfrac,
                            const int* pmask, const double* pwinv, double3 org, double dx, int3 dims, int apic, double* mi, double3* mvi, int* ref,
                            const int* nref_has = nullptr, const double3* nref_X = nullptr, const double3* nref_n = nullptr, int stride = 0, const int* plabel = nullptr, const int* nlab = nullptr)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    double3 x = xp[p], v = vp[p];
    double  m = mp[p];
    double  C[9];
#pragma unroll
    for(int r = 0; r < 9; ++r) C[r] = apic ? Cp[9 * p + r] : 0.0;
    NodeStencil st   = node_stencil(pbase[p], pfrac[p]);
    const int   mask = pmask[p];
    const double winv = pwinv[p];
    for(int a = 0; a < 3; ++a)
        for(int b = 0; b < 3; ++b)
            for(int c = 0; c < 3; ++c)
            {
                int nid = vnode(node_lin(st.b[0] + a, st.b[1] + b, st.b[2] + c, dims), x, nref_has, nref_X, nref_n, stride, plabel ? plabel[p] : -1, nlab);
                if(st.w[0][a] * st.w[1][b] * st.w[2][c] > 0.0) ref[nid] = 1;   // in a stencil (masked or not): a corner of some voxel
                double w = masked_w(st, a, b, c, mask, winv);
                if(w <= 0.0) continue;
                double3 xi  = make_double3(org.x + (st.b[0] + a) * dx, org.y + (st.b[1] + b) * dx, org.z + (st.b[2] + c) * dx);
                double3 d   = d3sub(xi, x);
                double3 va  = v;
                if(apic)
                {
                    va.x += C[0] * d.x + C[1] * d.y + C[2] * d.z;
                    va.y += C[3] * d.x + C[4] * d.y + C[5] * d.z;
                    va.z += C[6] * d.x + C[7] * d.y + C[8] * d.z;
                }
                double wm = w * m;
                atomicAdd(&mi[nid], wm);
                atomicAdd(&mvi[nid].x, wm * va.x);
                atomicAdd(&mvi[nid].y, wm * va.y);
                atomicAdd(&mvi[nid].z, wm * va.z);
            }
}
// active = carries masked mass OR is a referenced (voxel-corner) node without mass -> a GHOST inside the solid, held fixed at
// its lattice point for this step (a cut voxel keeps its full element; the fluid cannot expand into the wall)
__global__ void k_node_flag(int nnode, const double* mi, const int* ref, int make_ghost, int* nflag, int* ghost)
{
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if(nid >= nnode) return;
    bool hasm = mi[nid] > 0.0;
    nflag[nid] = (hasm || ref[nid]) ? 1 : 0;
    ghost[nid] = (make_ghost && !hasm && ref[nid]) ? 1 : 0;
}

// ----------------------------------------------------------------------------
// P2G
// ----------------------------------------------------------------------------
__global__ void k_p2g(int            n,
                      const double3* xp,
                      const double3* vp,
                      const double*  Cp,
                      const double*  Jp,
                      const double*  mp,
                      const double*  Vp0,
                      double         dx,
                      double3        org,
                      int3           dims,
                      double         kappa,
                      double         tension_ratio,
                      int            apic,
                      double*        mc,
                      double3*       mvc,
                      double*        Gc,
                      double*        Vc,
                      double*        piVc,
                      const int*     wnear = nullptr,   // CPIC: cell centres behind the particle's wall plane get no deposit
                      const double3* wX = nullptr,
                      const double3* wn = nullptr,
                      const int*     axis = nullptr,    // 2b: with axis + the second set, a near-wall particle's volume / pressure state
                      const int*     near2 = nullptr,   //     goes to the cell's '-' set when its wall normal is negative along the cell axis
                      const double3* wn2 = nullptr,
                      double*        mc2 = nullptr,
                      double*        Vc2 = nullptr,
                      double*        piVc2 = nullptr,
                      const int*     pmask = nullptr,   // 2b corner votes: each depositing near-wall particle votes with its own mask
                      float*         vote = nullptr,
                      const int*     nref_has = nullptr, // per-side slots: set / corner copies by the node reference planes
                      const double3* nref_X = nullptr,
                      const double3* nref_n = nullptr,
                      int            stride = 0,
                      int*           corner_side = nullptr,
                      const int*     plabel = nullptr,
                      const int*     nlab = nullptr,
                      double*        ybar = nullptr,     // LIQ_WB_YBAR: sum w V (up . x_p) per cell -> the height the deposited J belongs to
                      double3        upv  = make_double3(0, 1, 0))
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    double3 x = xp[p];
    double3 v = vp[p];
    double  m = mp[p];
    double  V = Vp0[p];
    double  J = Jp[p];
    double  wsum = 1.0;
    bool    masked = wnear && wnear[p];
    double3 X = masked ? wX[p] : make_double3(0, 0, 0), nn = masked ? wn[p] : make_double3(0, 0, 0);
    double  pi = (J < 1.0) ? kappa * J * (J - 1.0) : kappa * tension_ratio * J * (J - 1.0);
    double  C[9];
#pragma unroll
    for(int r = 0; r < 9; ++r) C[r] = apic ? Cp[9 * p + r] : 0.0;

    CentreStencil s = centre_stencil(x, org, dx, dims);
    if(masked)
    {   // renormalise over the compatible centres (volume is conserved on the particle's side)
        wsum = 0.0;
        for(int a = 0; a < 2; ++a)
            for(int b = 0; b < 2; ++b)
                for(int c = 0; c < 2; ++c)
                {
                    double3 xc = make_double3(org.x + (s.b[0] + a + 0.5) * dx, org.y + (s.b[1] + b + 0.5) * dx, org.z + (s.b[2] + c + 0.5) * dx);
                    if(nn.x * (xc.x - X.x) + nn.y * (xc.y - X.y) + nn.z * (xc.z - X.z) >= -1e-12) wsum += s.w[0][a] * s.w[1][b] * s.w[2][c];
                }
        if(wsum <= 1e-12) { masked = false; wsum = 1.0; }
    }
    for(int a = 0; a < 2; ++a)
        for(int b = 0; b < 2; ++b)
            for(int c = 0; c < 2; ++c)
            {
                double w = s.w[0][a] * s.w[1][b] * s.w[2][c];
                if(w <= 0.0) continue;
                int     ci = s.b[0] + a, cj = s.b[1] + b, ck = s.b[2] + c;
                int     cid = cell_lin(ci, cj, ck, dims);
                double3 xc  = make_double3(org.x + (ci + 0.5) * dx, org.y + (cj + 0.5) * dx, org.z + (ck + 0.5) * dx);
                if(masked)
                {
                    if(nn.x * (xc.x - X.x) + nn.y * (xc.y - X.y) + nn.z * (xc.z - X.z) < -1e-12) continue;
                    w /= wsum;
                }
                double3 dxc = d3sub(xc, x);
                double3 va  = v;
                if(apic)
                {
                    va.x += C[0] * dxc.x + C[1] * dxc.y + C[2] * dxc.z;
                    va.y += C[3] * dxc.x + C[4] * dxc.y + C[5] * dxc.z;
                    va.z += C[6] * dxc.x + C[7] * dxc.y + C[8] * dxc.z;
                }
                double wm = w * m;
                atomicAdd(&mc[cid], wm);
                atomicAdd(&mvc[cid].x, wm * va.x);
                atomicAdd(&mvc[cid].y, wm * va.y);
                atomicAdd(&mvc[cid].z, wm * va.z);
                if(apic)
                {
#pragma unroll
                    for(int r = 0; r < 9; ++r) atomicAdd(&Gc[9 * cid + r], wm * C[r]);
                }
                bool cut_here = false;   // 2b: does THIS particle see this cell as cut (one of its corners behind the particle's wall)?
                int  side     = 0;
                if(near2 && near2[p] && vote && pmask)
                {
                    int mk = pmask[p];
                    for(int q = 0; q < 8; ++q)
                    {
                        int qa = (q >> 2) & 1, qb = (q >> 1) & 1, qc = q & 1;
                        int kbit = ((a + qa) * 3 + (b + qb)) * 3 + (c + qc);
                        if(!((mk >> kbit) & 1)) cut_here = true;
                    }
                    if(cut_here)
                    {   // side split and corner votes only where the wall actually passes this cell for this particle;
                        // cells it sees as whole stay single-element (set 0) -- no overlapping elements in the interior
                        side = axis ? cell_side(axis, cid, wn2[p]) : 0;
                        for(int q = 0; q < 8; ++q)
                        {
                            int qa = (q >> 2) & 1, qb = (q >> 1) & 1, qc = q & 1;
                            int kbit = ((a + qa) * 3 + (b + qb)) * 3 + (c + qc);
                            atomicAdd(&vote[(cid * 2 + side) * 16 + q + (((mk >> kbit) & 1) ? 0 : 8)], (float)w);
                        }
                    }
                }
                if(stride > 0 && corner_side && mc2)
                {   // two-field: the set is the particle's side at the cell's anchor node; record the copies it uses at the corners
                    int anchor = node_lin(ci, cj, ck, dims);
                    side = vside(anchor, x, nref_has, nref_X, nref_n, stride, plabel ? plabel[p] : -1, nlab);
                    for(int q = 0; q < 8; ++q)
                    {
                        int qa = (q >> 2) & 1, qb = (q >> 1) & 1, qc = q & 1;
                        corner_side[(cid * 2 + side) * 8 + q] = vside(node_lin(ci + qa, cj + qb, ck + qc, dims), x, nref_has, nref_X, nref_n, stride, plabel ? plabel[p] : -1, nlab);
                    }
                    cut_here = true;
                }
                if(cut_here && side == 1 && mc2)
                {   // '-' side of a cut cell: its own volume state (the mass also, for the activity test)
                    atomicAdd(&mc2[cid], wm);
                    atomicAdd(&Vc2[cid], w * V);
                    atomicAdd(&piVc2[cid], w * V * pi);
                    continue;
                }
                atomicAdd(&Vc[cid], w * V);
                atomicAdd(&piVc[cid], w * V * pi);
                if(ybar) atomicAdd(&ybar[cid], w * V * (upv.x * x.x + upv.y * x.y + upv.z * x.z));
            }
}

__global__ void k_cell_finalize(int           ncell,
                                const double* mc,
                                double3*      mvc,
                                double*       Gc,
                                const double* Vc,
                                const double* piVc,
                                double*       Jbc,
                                int*          flag,
                                double        kappa,
                                double        tension_ratio,
                                int           j_mode,
                                double        rho_cell,
                                const double* mc2 = nullptr,   // 2b second set (nullptr = none)
                                const double* Vc2 = nullptr,
                                const double* piVc2 = nullptr,
                                double*       Jbc2 = nullptr)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if(c >= ncell) return;
    bool act2 = false;
    if(mc2 && Jbc2)
    {
        act2 = mc2[c] > 0.0 && Vc2[c] > 0.0;
        double Jb2 = 1.0;
        if(act2)
        {
            if(j_mode == 2 || j_mode == 4) Jb2 = fmin(fmax(rho_cell / mc2[c], 0.25), 8.0);
            else
            {
                double pi = piVc2[c] / Vc2[c];
                double keff = (pi > 0.0 && tension_ratio > 0.0) ? kappa * tension_ratio : kappa;
                double sq = 1.0 + 4.0 * pi / keff;
                if(sq < 0.0) sq = 0.0;
                Jb2 = 0.5 * (1.0 + sqrt(sq));
            }
        }
        Jbc2[c] = Jb2;
    }
    double m = mc[c];
    if(m > 0.0 && Vc[c] > 0.0)
    {
        double inv = 1.0 / m;
        mvc[c]     = d3scale(mvc[c], inv);
#pragma unroll
        for(int r = 0; r < 9; ++r) Gc[9 * c + r] *= inv;
        if(j_mode == 2 || j_mode == 4)
        {
            double Jb = rho_cell / m;  // rho0 dx^3 / m_c: cell-level volume ratio (mode 4: x fill, see fill_fraction_pass)
            { double Jb_c = fmin(fmax(Jb, 0.25), 8.0); if(Jb_c != Jb) LIQ_FBK(0); Jbc[c] = Jb_c; }
        }
        else
        {
            double pi = piVc[c] / Vc[c];
            // inverse of the SAME two-branch law k_p2g deposited: pi = kappa J (J-1) for J < 1 (pi < 0),
            // pi = kappa r J (J-1) for J > 1 (pi > 0). Inverting the tensile branch with kappa instead of
            // kappa r under-read the cell's expansion by the factor r (2026-10-18 audit, LIQ_TENSION > 0 only).
            double keff = (pi > 0.0 && tension_ratio > 0.0) ? kappa * tension_ratio : kappa;
            double s  = 1.0 + 4.0 * pi / keff;
            if(s < 0.0) s = 0.0;
            Jbc[c] = 0.5 * (1.0 + sqrt(s));
        }
        flag[c] = 1;
    }
    else
    {
        Jbc[c]  = 1.0;
        flag[c] = act2 ? 1 : 0;
    }
}

__global__ void k_node_gather(int            nnode,
                              int3           dims,
                              double         dx,
                              const double*  mc,
                              const double3* vc,
                              const double*  Gc,
                              const int*     cflag,
                              double*        mi,
                              double3*       mvi,
                              int*           nflag)
{
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if(nid >= nnode) return;
    int i, j, k;
    node_ijk(nid, dims, i, j, k);
    double  m  = 0.0;
    double3 mv = make_double3(0, 0, 0);
    for(int a = 0; a < 2; ++a)
        for(int b = 0; b < 2; ++b)
            for(int c = 0; c < 2; ++c)
            {
                int ci = i - a, cj = j - b, ck = k - c;
                if(ci < 0 || cj < 0 || ck < 0 || ci >= dims.x || cj >= dims.y || ck >= dims.z) continue;
                int cid = cell_lin(ci, cj, ck, dims);
                if(!cflag[cid]) continue;
                double  mcc = 0.125 * mc[cid];
                double3 d   = make_double3((a - 0.5) * dx, (b - 0.5) * dx, (c - 0.5) * dx);  // x_i - x_c
                const double* G = Gc + 9 * cid;
                double3 v = vc[cid];
                v.x += G[0] * d.x + G[1] * d.y + G[2] * d.z;
                v.y += G[3] * d.x + G[4] * d.y + G[5] * d.z;
                v.z += G[6] * d.x + G[7] * d.y + G[8] * d.z;
                m += mcc;
                mv = d3add(mv, d3scale(v, mcc));
            }
    mi[nid]    = m;
    mvi[nid]   = mv;
    nflag[nid] = (m > 0.0) ? 1 : 0;
}

// LIQ_BDF2: the previous step's particle velocity field to the grid, through the same route as the momentum
// (particle -> cell centres with the APIC affine term, normalise by the cell mass, cell centres -> nodes)
__global__ void k_p2g_hist(int n, const double3* xp, const double3* v2, const double* C2, const double* mp, double dx, double3 org, int3 dims,
                           int apic, double3* mvc2, double* Gc2)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    double3       x = xp[p];
    double3       v = v2[p];
    double        m = mp[p];
    CentreStencil s = centre_stencil(x, org, dx, dims);
    for(int a = 0; a < 2; ++a)
        for(int b = 0; b < 2; ++b)
            for(int c = 0; c < 2; ++c)
            {
                double w = s.w[0][a] * s.w[1][b] * s.w[2][c];
                if(w <= 0.0) continue;
                int     ci = s.b[0] + a, cj = s.b[1] + b, ck = s.b[2] + c;
                int     cid = cell_lin(ci, cj, ck, dims);
                double3 dxc = d3sub(make_double3(org.x + (ci + 0.5) * dx, org.y + (cj + 0.5) * dx, org.z + (ck + 0.5) * dx), x);
                double3 va  = v;
                if(apic)
                {
                    const double* C = C2 + 9 * p;
                    va.x += C[0] * dxc.x + C[1] * dxc.y + C[2] * dxc.z;
                    va.y += C[3] * dxc.x + C[4] * dxc.y + C[5] * dxc.z;
                    va.z += C[6] * dxc.x + C[7] * dxc.y + C[8] * dxc.z;
                }
                double wm = w * m;
                atomicAdd(&mvc2[cid].x, wm * va.x);
                atomicAdd(&mvc2[cid].y, wm * va.y);
                atomicAdd(&mvc2[cid].z, wm * va.z);
                if(apic)
                {
#pragma unroll
                    for(int r = 0; r < 9; ++r) atomicAdd(&Gc2[9 * cid + r], wm * C2[9 * p + r]);
                }
            }
}
__global__ void k_cell_norm_hist(int ncell, const double* mc, const int* cflag, double3* mvc2, double* Gc2)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if(c >= ncell || !cflag[c] || mc[c] <= 0.0) return;
    double inv = 1.0 / mc[c];
    mvc2[c]    = d3scale(mvc2[c], inv);
#pragma unroll
    for(int r = 0; r < 9; ++r) Gc2[9 * c + r] *= inv;
}
__global__ void k_node_gather_hist(int nnode, int3 dims, double dx, const double* mc, const double3* vc2, const double* Gc2, const int* cflag, double3* mvi2)
{
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if(nid >= nnode) return;
    int i, j, k;
    node_ijk(nid, dims, i, j, k);
    double3 mv = make_double3(0, 0, 0);
    for(int a = 0; a < 2; ++a)
        for(int b = 0; b < 2; ++b)
            for(int c = 0; c < 2; ++c)
            {
                int ci = i - a, cj = j - b, ck = k - c;
                if(ci < 0 || cj < 0 || ck < 0 || ci >= dims.x || cj >= dims.y || ck >= dims.z) continue;
                int cid = cell_lin(ci, cj, ck, dims);
                if(!cflag[cid]) continue;
                double        mcc = 0.125 * mc[cid];
                double3       d   = make_double3((a - 0.5) * dx, (b - 0.5) * dx, (c - 0.5) * dx);
                const double* G   = Gc2 + 9 * cid;
                double3       v   = vc2[cid];
                v.x += G[0] * d.x + G[1] * d.y + G[2] * d.z;
                v.y += G[3] * d.x + G[4] * d.y + G[5] * d.z;
                v.z += G[6] * d.x + G[7] * d.y + G[8] * d.z;
                mv = d3add(mv, d3scale(v, mcc));
            }
    mvi2[nid] = mv;
}
// LIQ_BDF2_WALL_BE: nodes near a wall stay implicit Euler -- mode 1: the 27-node stencil of every wall-point particle (~1.5 cells);
// mode 2: only the 8 corners of the cell holding each wall point (the cut cells, ~0.5 cell)
__global__ void k_bh_wall_mark(int n, const int* valid, const int3* pbase, const int3* wcell, int3 dims, int mode, int* flag)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n || !valid[p]) return;
    int3 b    = mode == 2 ? wcell[p] : pbase[p];
    int  span = mode == 2 ? 2 : 3;
    for(int a = 0; a < span; ++a)
        for(int bb = 0; bb < span; ++bb)
            for(int c = 0; c < span; ++c)
            {
                int ni = b.x + a, nj = b.y + bb, nk = b.z + c;
                if(ni < 0 || nj < 0 || nk < 0 || ni > dims.x || nj > dims.y || nk > dims.z) continue;
                flag[node_lin(ni, nj, nk, dims)] = 1;
            }
}
// LIQ_BDF2 after the solve: the displacement field (x1 - X) / dt moves the particles and their J, the BDF2 velocity
// v1 = (4/3 v - 1/3 v_prev + 2/3 dt g) + 3/2 (x1 - x~) / dt  (from m' (x1 - x~) / dt^2 = f, m' = 9/4 m) is what they carry
__global__ void k_read_node_vel_bdf2(int n_active, const int* active_nodes, const double3* vert, const double3* xt, const double3* vb, int3 dims,
                                     double dx, double3 org, int base, double dt, double3* vi_pos, double3* vi_vel,   // xt: the liquid's own copy, per active slot
                                     const int* wall_be = nullptr)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if(s >= n_active) return;
    int n = active_nodes[s];
    int i, j, k;
    node_ijk(n, dims, i, j, k);
    double3 x0 = make_double3(org.x + i * dx, org.y + j * dx, org.z + k * dx);
    double3 x1 = vert[base + s];
    vi_pos[n]  = d3scale(d3sub(x1, x0), 1.0 / dt);
    vi_vel[n]  = (wall_be && wall_be[n]) ? vi_pos[n] : d3add(vb[s], d3scale(d3sub(x1, xt[s]), 1.5 / dt));   // near-wall nodes: implicit Euler
}
__global__ void k_p2c_contact(int n, const double3* xp, const double* mp, const int* pflag, double3 org, double dx, int3 dims, double* mcc)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n || !pflag[p]) return;
    CentreStencil s = centre_stencil(xp[p], org, dx, dims);
    for(int a = 0; a < 2; ++a)
        for(int b = 0; b < 2; ++b)
            for(int c = 0; c < 2; ++c)
            {
                double w = s.w[0][a] * s.w[1][b] * s.w[2][c];
                if(w > 0.0) atomicAdd(&mcc[cell_lin(s.b[0] + a, s.b[1] + b, s.b[2] + c, dims)], w * mp[p]);
            }
}
__global__ void k_node_contact(int nnode, int3 dims, const double* mcc, const int* cflag, double* mci)
{
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if(nid >= nnode) return;
    int i, j, k;
    node_ijk(nid, dims, i, j, k);
    double m = 0.0;
    for(int a = 0; a < 2; ++a)
        for(int b = 0; b < 2; ++b)
            for(int c = 0; c < 2; ++c)
            {
                int ci = i - a, cj = j - b, ck = k - c;
                if(ci < 0 || cj < 0 || ck < 0 || ci >= dims.x || cj >= dims.y || ck >= dims.z) continue;
                int cid = cell_lin(ci, cj, ck, dims);
                if(cflag[cid]) m += 0.125 * mcc[cid];
            }
    mci[nid] = m;
}

__global__ void k_compact_cells(int ncell, const int* flag, const int* scan, int* active)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if(c >= ncell) return;
    if(flag[c]) active[scan[c]] = c;
}
__global__ void k_compact_nodes(int nnode, const int* flag, const int* scan, int* active, int* slot)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if(n >= nnode) return;
    if(flag[n])
    {
        active[scan[n]] = n;
        slot[n]         = scan[n];
    }
    else
        slot[n] = -1;
}
__global__ void k_counts(int ncell, int nnode, const int* cflag, const int* cscan, const int* nflag, const int* nscan, int* out)
{
    out[0] = cscan[ncell - 1] + cflag[ncell - 1];
    out[1] = nscan[nnode - 1] + nflag[nnode - 1];
}

// ---------------------------------------------------------------------------
// grid wall boundary condition: classify a lattice node against the containers
// ---------------------------------------------------------------------------
__device__ __forceinline__ double cont_r(const LiquidContainer& c, double y, double* slope)
{
    if(y <= c.py[0]) { *slope = 0.0; return c.pr[0]; }
    for(int k = 1; k < c.n; ++k)
        if(y <= c.py[k])
        {
            double dy = c.py[k] - c.py[k - 1];
            double s  = dy > 0.0 ? (c.pr[k] - c.pr[k - 1]) / dy : 0.0;
            *slope    = s;
            return c.pr[k - 1] + s * (y - c.py[k - 1]);
        }
    *slope = 0.0;
    return c.pr[c.n - 1];
}
__device__ __forceinline__ bool inv3(const double A[9], double B[9])
{
    double d = A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) + A[2] * (A[3] * A[7] - A[4] * A[6]);
    if(!(fabs(d) > 1e-12)) return false;
    double id = 1.0 / d;
    B[0] = (A[4] * A[8] - A[5] * A[7]) * id; B[1] = (A[2] * A[7] - A[1] * A[8]) * id; B[2] = (A[1] * A[5] - A[2] * A[4]) * id;
    B[3] = (A[5] * A[6] - A[3] * A[8]) * id; B[4] = (A[0] * A[8] - A[2] * A[6]) * id; B[5] = (A[2] * A[3] - A[0] * A[5]) * id;
    B[6] = (A[3] * A[7] - A[4] * A[6]) * id; B[7] = (A[1] * A[6] - A[0] * A[7]) * id; B[8] = (A[0] * A[4] - A[1] * A[3]) * id;
    return true;
}
// returns 0 = free, 1 = slip (normal n follows the body), 2 = follow the body; vs = body velocity at x
__device__ int wall_classify(double3 x, const LiquidContainer* cont, int ncont, const double* q, const double* qv,
                             int ground_bc, double ground_y, double3& n, double3& vs)
{
    for(int ci = 0; ci < ncont; ++ci)
    {
        const LiquidContainer& c = cont[ci];
        const double* Q  = q + 12 * c.body;
        const double* QV = qv + 12 * c.body;
        // ABD map x = p + R xbar with the ROWS of R = a1, a2, a3 = q[3:6], q[6:9], q[9:12] (x_i = p_i + a_i . xbar;
        // get_abd_body_transforms() returns R^T in its 3x3 block, see lab_scene.abd_world_tf)
        double A[9] = {Q[3], Q[4], Q[5], Q[6], Q[7], Q[8], Q[9], Q[10], Q[11]};   // row-major R
        double Ai[9];
        if(!inv3(A, Ai)) continue;
        double d[3]  = {x.x - Q[0], x.y - Q[1], x.z - Q[2]};
        double xb[3] = {Ai[0] * d[0] + Ai[1] * d[1] + Ai[2] * d[2], Ai[3] * d[0] + Ai[4] * d[1] + Ai[5] * d[2],
                        Ai[6] * d[0] + Ai[7] * d[1] + Ai[8] * d[2]};
        double X[3]  = {c.Rd[0] * xb[0] + c.Rd[1] * xb[1] + c.Rd[2] * xb[2] + c.td[0], c.Rd[3] * xb[0] + c.Rd[4] * xb[1] + c.Rd[5] * xb[2] + c.td[1],
                        c.Rd[6] * xb[0] + c.Rd[7] * xb[1] + c.Rd[8] * xb[2] + c.td[2]};
        double y = X[1], r = sqrt(X[0] * X[0] + X[2] * X[2]);
        if(y > c.y_top || y < c.py[0] - c.band) continue;
        double s, rin = cont_r(c, y, &s);
        if(y >= c.py[0] && r < rin) return 0;           // inside the cavity: fluid region, free
        if(r > rin + c.band) continue;                  // beyond the shell
        int    type;
        double nX[3];
        if(y >= c.py[0])
        {   // wall: inward normal of r = rin(y) is (-e_r, s) (any sign: only the line matters)
            double er0 = r > 1e-12 ? X[0] / r : 1.0, er2 = r > 1e-12 ? X[2] / r : 0.0;
            double nn  = sqrt(1.0 + s * s);
            nX[0] = -er0 / nn; nX[1] = s / nn; nX[2] = -er2 / nn;
            type  = c.slip ? 1 : 2;
        }
        else if(r <= c.pr[0])
        {
            nX[0] = 0.0; nX[1] = 1.0; nX[2] = 0.0;        // floor
            type  = c.slip ? 1 : 2;
        }
        else
        {
            nX[0] = 0.0; nX[1] = 1.0; nX[2] = 0.0;
            type  = 2;                                    // corner below the wall: follow the body
        }
        // world normal ~ A^-T Rd^T nX
        double t[3] = {c.Rd[0] * nX[0] + c.Rd[3] * nX[1] + c.Rd[6] * nX[2], c.Rd[1] * nX[0] + c.Rd[4] * nX[1] + c.Rd[7] * nX[2],
                       c.Rd[2] * nX[0] + c.Rd[5] * nX[1] + c.Rd[8] * nX[2]};
        double w[3] = {Ai[0] * t[0] + Ai[3] * t[1] + Ai[6] * t[2], Ai[1] * t[0] + Ai[4] * t[1] + Ai[7] * t[2],
                       Ai[2] * t[0] + Ai[5] * t[1] + Ai[8] * t[2]};
        double wn = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
        n  = make_double3(w[0] / wn, w[1] / wn, w[2] / wn);
        // body velocity at x: pdot + Adot xbar
        vs = make_double3(QV[0] + QV[3] * xb[0] + QV[4] * xb[1] + QV[5] * xb[2], QV[1] + QV[6] * xb[0] + QV[7] * xb[1] + QV[8] * xb[2],
                          QV[2] + QV[9] * xb[0] + QV[10] * xb[1] + QV[11] * xb[2]);
        return type;
    }
    if(ground_bc && x.y < ground_y)
    {
        n  = make_double3(0, 1, 0);
        vs = make_double3(0, 0, 0);
        return 1;
    }
    return 0;
}

__global__ void k_write_node_slots(int            pool,
                                   int            n_active,
                                   const int*     active_nodes,
                                   const double*  mi,
                                   const double3* mvi,
                                   int3           dims,
                                   double         dx,
                                   double3        org,
                                   double3        park,
                                   double         mfloor,
                                   int            floor_gravity,
                                   const double*  mci,
                                   double         ghost_floor,
                                   double         ghost_frac,
                                   int            cav_on,
                                   double         c0, double c1, double c2, double c3, double c4, double c5, double c6,
                                   const int*     ghost,
                                   int            n_ghost_slots,     // 2b private ghost corners after the active nodes
                                   const int3*    ghost_corner,
                                   int            n_lattice,         // per-side slots: node id = nid % n_lattice
                                   const LiquidContainer* cont,
                                   int            ncont,
                                   const double*  abd_q,
                                   const double*  abd_qv,
                                   int            ground_bc,
                                   double         ground_y,
                                   int*           wall_t,
                                   double3*       wall_n,
                                   double3*       wall_xt,
                                   int*           wall_count,
                                   double         rho,
                                   double         rest_k,
                                   double         rest_kv,
                                   double         dt,
                                   double3        g,
                                   int            base,
                                   double3*       vert,
                                   double3*       overt,
                                   double3*       vel,
                                   double3*       xt,
                                   double*        mass,
                                   int*           btype,
                                   const double3* bh_mvi = nullptr,   // LIQ_BDF2: previous-step node momentum (nullptr = implicit Euler)
                                   double3*       bh_vb  = nullptr,
                                   double3*       bh_xt  = nullptr,
                                   const int*     bh_wall = nullptr,   // LIQ_BDF2_WALL_BE: these nodes keep the implicit-Euler predictor
                                   const double*  node_w  = nullptr,   // LIQ_WB: consistent nodal weight; gravity acts on it, not on the deposited mass
                                   const double*  node_mfl = nullptr,  // LIQ_WB_MFLOOR: per-node extra inertia floor (x rho dx^3), weightless
                                   int            air_grav = 0)       // LIQ_WB_AIRGRAV: nodes with deposited mass but no geometric weight (above the free surface) weigh their deposit
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if(s >= pool) return;
    int slot = base + s;
    if(s < n_active)
    {
        int n = active_nodes[s];
        int i, j, k;
        node_ijk(n_lattice > 0 ? n % n_lattice : n, dims, i, j, k);
        double3 x = make_double3(org.x + i * dx, org.y + j * dx, org.z + k * dx);
        if(ghost && ghost[n])
        {   // CPIC ghost: a voxel corner behind the wall seen from every particle that references it -> fixed this step
            vert[slot] = x; overt[slot] = x; vel[slot] = make_double3(0, 0, 0); xt[slot] = x; mass[slot] = rho * dx * dx * dx; btype[slot] = 1;
            if(wall_t) wall_t[slot - base] = 0;
            return;
        }
        double  m = mi[n];
        double3 v = (m > 0.0) ? d3scale(mvi[n], 1.0 / m) : make_double3(0, 0, 0);
        // near-massless free-surface nodes (jittered seeds, grazing stencils) make the
        // Jacobi-PCG stall: floor the node mass (interior nodes are unaffected)
        double mr = m;
        m = fmax(m, mfloor * rho * dx * dx * dx);
        // the floor is for PCG conditioning: it must add inertia, not weight (a weighed floor makes every
        // boundary node heavier than the water it carries -> the wall layer sinks and drives a steady roll)
        double gs = floor_gravity ? 1.0 : mr / m;
        if(ghost_floor > 0.0 && mci && mci[n] > ghost_frac * mr && mr < ghost_floor * rho * dx * dx * dx)
        {   // wall ghost: heavy (resists drifting into the solid within the step) but weightless
            m  = ghost_floor * rho * dx * dx * dx;
            gs = mr / m;
        }
        if(node_mfl && node_mfl[n] > 0.0) m = fmax(m, node_mfl[n] * rho * dx * dx * dx);   // LIQ_WB_MFLOOR: near-wall inertia floor
        if(node_w) gs = (m > 0.0) ? ((air_grav && !(node_w[n] > 0.0)) ? mr / m : node_w[n] / m) : 0.0;   // LIQ_WB: weight = rho x geometric fluid volume around the node (0 behind a wall)
        if(wall_t)
        {   // grid wall boundary condition: the normal displacement is pinned to the body by a penalty
            // (k_wall_assemble); here the node only starts with the body's normal velocity
            double3 nn = make_double3(0, 1, 0), vsol = make_double3(0, 0, 0);
            int     wt = (ncont > 0 || ground_bc) ? wall_classify(x, cont, ncont, abd_q, abd_qv, ground_bc, ground_y, nn, vsol) : 0;
            wall_t[s] = wt;
            if(wt)
            {
                if(wt == 1)
                {
                    double dn = (v.x - vsol.x) * nn.x + (v.y - vsol.y) * nn.y + (v.z - vsol.z) * nn.z;
                    v         = make_double3(v.x - dn * nn.x, v.y - dn * nn.y, v.z - dn * nn.z);
                }
                else
                    v = vsol;
                wall_n[s]  = nn;
                wall_xt[s] = make_double3(x.x + dt * vsol.x, x.y + dt * vsol.y, x.z + dt * vsol.z);
                if(wall_count) atomicAdd(wall_count, 1);
            }
        }
        if(cav_on)
        {   // TEST: grid node on the solid side of the cavity wall / floor -> fixed (grid-level wall boundary condition)
            double r   = sqrt((x.x - c0) * (x.x - c0) + (x.z - c1) * (x.z - c1));
            double rin = (x.y < c4) ? c3 : (x.y > c6 ? c5 : c3 + (c5 - c3) * (x.y - c4) / (c6 - c4));
            if(x.y < c2 || r > rin)
            {
                vert[slot] = x; overt[slot] = x; vel[slot] = make_double3(0, 0, 0); xt[slot] = x; mass[slot] = m; btype[slot] = 1;
                return;
            }
        }
        double3 xtn = make_double3(x.x + dt * v.x + dt * dt * gs * g.x, x.y + dt * v.y + dt * dt * gs * g.y, x.z + dt * v.z + dt * dt * gs * g.z);
        if(bh_mvi && !(bh_wall && bh_wall[n]))
        {   // LIQ_BDF2 (velocity form): x1 = x + dt (19/18 v - 1/18 v_prev) + 4/9 dt^2 a(x1)  <=>  inertia 9/4 m around this predictor
            double3 v2 = (mr > 0.0) ? d3scale(bh_mvi[n], 1.0 / mr) : v;
            double  c2 = 4.0 / 9.0 * dt * dt * gs;
            xtn = make_double3(x.x + dt * (19.0 / 18.0 * v.x - 1.0 / 18.0 * v2.x) + c2 * g.x, x.y + dt * (19.0 / 18.0 * v.y - 1.0 / 18.0 * v2.y) + c2 * g.y,
                               x.z + dt * (19.0 / 18.0 * v.z - 1.0 / 18.0 * v2.z) + c2 * g.z);
            m *= 2.25;
            bh_vb[s] = make_double3(4.0 / 3.0 * v.x - 1.0 / 3.0 * v2.x + 2.0 / 3.0 * dt * gs * g.x, 4.0 / 3.0 * v.y - 1.0 / 3.0 * v2.y + 2.0 / 3.0 * dt * gs * g.y,
                                    4.0 / 3.0 * v.z - 1.0 / 3.0 * v2.z + 2.0 / 3.0 * dt * gs * g.z);
            bh_xt[s] = xtn;
        }
        // rest drag inside the solve: still water is pushed afresh every step by the residual wall-layer imbalance, and damping
        // the carried velocity cannot reach that push. A slow node instead gets (1 + k) times its inertia around a target pulled
        // toward rest, m' = m (1 + k), x~' = (x~ + k x) / (1 + k): the gradient at the start of the step is unchanged (weight and
        // pressure balance as before), stiff (pressure) modes are unaffected, and soft modes move 1/(1 + k) as far per step.
        if(rest_k > 0.0)
        {
            double sp = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
            if(sp < rest_kv)
            {
                double w  = 1.0 - (sp / rest_kv) * (sp / rest_kv);
                double kk = rest_k * w * w;
                xtn = make_double3((xtn.x + kk * x.x) / (1.0 + kk), (xtn.y + kk * x.y) / (1.0 + kk), (xtn.z + kk * x.z) / (1.0 + kk));
                m *= 1.0 + kk;
            }
        }
        vert[slot]  = x;
        overt[slot] = x;
        vel[slot]   = v;
        xt[slot]    = xtn;
        mass[slot]  = m;
        btype[slot] = 0;
    }
    else if(s < n_active + n_ghost_slots)
    {   // 2b private ghost corner: fixed at its lattice point this step
        int3    gc = ghost_corner[s - n_active];
        double3 x  = make_double3(org.x + gc.x * dx, org.y + gc.y * dx, org.z + gc.z * dx);
        vert[slot] = x; overt[slot] = x; vel[slot] = make_double3(0, 0, 0); xt[slot] = x; mass[slot] = rho * dx * dx * dx; btype[slot] = 1;
        if(wall_t) wall_t[s] = 0;
    }
    else
    {
        vert[slot]  = park;
        overt[slot] = park;
        vel[slot]   = make_double3(0, 0, 0);
        xt[slot]    = park;
        mass[slot]  = rho * dx * dx * dx;
        btype[slot] = 1;
    }
}

// 2b: one voxel element per (active cell, active side). A corner that is not on the element's side of its wall (or has no
// node slot) is replaced by a PRIVATE fixed ghost slot at the lattice corner (allocated after the active nodes).
__device__ __forceinline__ bool corner_real(const float* vote, int c, int sd, int q)
{
    if(!vote) return true;   // ablation: private ghosts disabled
    const float* v = vote + (c * 2 + sd) * 16;
    return !(v[q] <= 0.0f && v[8 + q] > 0.0f);   // ghost only when every depositing particle of this side has it masked
}
__global__ void k_elem_plan(int nact, const int* active_cells, const double* mc, const double* Vc, const double* mc2, const double* Vc2,
                            const float* vote, const int* node_slot, int3 dims, double3 org, double dx,
                            int* elem_cnt, int* ghost_cnt)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if(v >= nact) return;
    int c = active_cells[v];
    int i, j, k;
    cell_ijk(c, dims, i, j, k);
    int ne = 0, ng = 0;
    for(int sd = 0; sd < 2; ++sd)
    {
        bool act = sd == 0 ? (mc[c] > 0.0 && Vc[c] > 0.0) : (mc2 && mc2[c] > 0.0 && Vc2[c] > 0.0);
        if(!act) continue;
        ++ne;
        for(int q = 0; q < 8; ++q)
        {
            int a = (q >> 2) & 1, b = (q >> 1) & 1, cc = q & 1;
            int sl = node_slot[node_lin(i + a, j + b, k + cc, dims)];
            if(sl < 0 || !corner_real(vote, c, sd, q)) ++ng;
        }
    }
    elem_cnt[v] = ne;
    ghost_cnt[v] = ng;
}
__global__ void k_vox_build(int nact, const int* active_cells, const double* mc, const double* Vc, const double* Jbc, const double* mc2, const double* Vc2,
                            const double* Jbc2, const float* vote, const int* node_slot, int3 dims, double3 org,
                            double dx, int base, int ghost_base, const int* elem_off, const int* ghost_off, int* vox_nodes, double* voxV, double* voxJb,
                            int3* voxijk, int3* ghost_corner)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if(v >= nact) return;
    int c = active_cells[v];
    int i, j, k;
    cell_ijk(c, dims, i, j, k);
    int e = elem_off[v], g = ghost_off[v];
    for(int sd = 0; sd < 2; ++sd)
    {
        bool act = sd == 0 ? (mc[c] > 0.0 && Vc[c] > 0.0) : (mc2 && mc2[c] > 0.0 && Vc2[c] > 0.0);
        if(!act) continue;
        for(int q = 0; q < 8; ++q)
        {
            int a = (q >> 2) & 1, b = (q >> 1) & 1, cc = q & 1;
            int sl = node_slot[node_lin(i + a, j + b, k + cc, dims)];
            if(sl >= 0 && corner_real(vote, c, sd, q))
                vox_nodes[8 * e + q] = base + sl;
            else
            {
                vox_nodes[8 * e + q] = base + ghost_base + g;
                ghost_corner[g]      = make_int3(i + a, j + b, k + cc);
                ++g;
            }
        }
        voxV[e]   = sd == 0 ? Vc[c] : Vc2[c];
        voxJb[e]  = sd == 0 ? Jbc[c] : Jbc2[c];
        voxijk[e] = make_int3(i, j, k);
        ++e;
    }
}
// 2b: an active node that no voxel element references any more (its corners were taken by private ghosts) has no
// volume stiffness but still receives ghost-penalty / wall forces -> it must be held fixed this step (orphan rule)
__global__ void k_mark_used(int nelem, const int* vox_nodes, int base, int nact, int* used)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if(t >= 8 * nelem) return;
    int s = vox_nodes[t] - base;
    if(s >= 0 && s < nact) used[s] = 1;
}
__global__ void k_orphan_ghost(int nact, const int* active_nodes, const int* used, int* ghost, int* count)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if(s >= nact) return;
    if(!used[s]) { ghost[active_nodes[s]] = 1; atomicAdd(count, 1); }
}
// per-side node slots: one element per (active cell, active set); corner slot = the copy the set's particles use there
__global__ void k_elem_plan2(int nact, const int* active_cells, const double* mc, const double* Vc, const double* mc2, const double* Vc2, int* elem_cnt)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if(v >= nact) return;
    int c = active_cells[v];
    elem_cnt[v] = ((mc[c] > 0.0 && Vc[c] > 0.0) ? 1 : 0) + ((mc2[c] > 0.0 && Vc2[c] > 0.0) ? 1 : 0);
}
__global__ void k_vox_build2(int nact, const int* active_cells, const double* mc, const double* Vc, const double* Jbc, const double* mc2, const double* Vc2, const double* Jbc2,
                             const int* corner_side, const int* node_slot, int3 dims, int base, int stride, const int* elem_off, int* vox_nodes, double* voxV,
                             double* voxJb, int3* voxijk)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if(v >= nact) return;
    int c = active_cells[v];
    int i, j, k;
    cell_ijk(c, dims, i, j, k);
    int e = elem_off[v];
    for(int sd = 0; sd < 2; ++sd)
    {
        bool act = sd == 0 ? (mc[c] > 0.0 && Vc[c] > 0.0) : (mc2[c] > 0.0 && Vc2[c] > 0.0);
        if(!act) continue;
        for(int q = 0; q < 8; ++q)
        {
            int a = (q >> 2) & 1, b = (q >> 1) & 1, cc = q & 1;
            int nid = node_lin(i + a, j + b, k + cc, dims) + corner_side[(c * 2 + sd) * 8 + q] * stride;
            int sl  = node_slot[nid];
            if(sl < 0) { LIQ_FBK(10); sl = node_slot[nid % stride]; if(sl < 0) sl = 0; }   // must not happen: the set's particles referenced it
            vox_nodes[8 * e + q] = base + sl;
        }
        voxV[e]   = sd == 0 ? Vc[c] : Vc2[c];
        voxJb[e]  = sd == 0 ? Jbc[c] : Jbc2[c];
        voxijk[e] = make_int3(i, j, k);
        ++e;
    }
}
__global__ void k_vox_data(int           nact,
                           const int*    active_cells,
                           const int*    node_slot,
                           int3          dims,
                           const double* Vc,
                           const double* Jbc,
                           int           base,
                           int*          vox_nodes,
                           double*       voxV,
                           double*       voxJb,
                           int3*         voxijk)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if(v >= nact) return;
    int c = active_cells[v];
    int i, j, k;
    cell_ijk(c, dims, i, j, k);
    for(int q = 0; q < 8; ++q)
    {
        int a = (q >> 2) & 1, b = (q >> 1) & 1, cc = q & 1;
        int n = node_lin(i + a, j + b, k + cc, dims);
        int s = node_slot[n];
        if(s < 0) LIQ_FBK(10);   // voxel corner without a node slot: silently rerouted to slot 0 (must stay 0)
        vox_nodes[8 * v + q] = base + (s < 0 ? 0 : s);
    }
    voxV[v]   = Vc[c];
    voxJb[v]  = Jbc[c];
    voxijk[v] = make_int3(i, j, k);
}

// LIQ_WB_YBAR: the kernel deposit gives the J of the depositing particles' mean height, not of the element's fluid centroid -- next to a
// floor (particles only above) or under the free surface (particles only below) the two differ by ~h/4 and the hydrostatic J step over
// that offset is ~11 % of a cell's pressure difference (a hydrostatic start is out of balance only in those rows). Shift by the
// hydrostatic gradient: Jb(centroid) = Jb + rho g / kappa (y_c - ybar).
__global__ void k_wb_ybar_fix(int nact, const int* active_cells, double* voxJb, const double* vox_yc, const double* ybar, const double* Vc, double rgk)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if(v >= nact) return;
    const int c = active_cells[v];
    const double yc = vox_yc[v];
    if(!(yc > -1e299) || !(Vc[c] > 0.0)) return;
    voxJb[v] += rgk * (yc - ybar[c] / Vc[c]);
}
// LIQ_WB_GEOV: rest volume of each element from its geometric volume (sum of the consistent corner weights / rho) and its state Jb,
// so that the voxel force is exactly pressure x geometric area -- the same geometry the node weights use. The deposited volume
// (kernel-weighted particle rest volume) is 12.5 % short next to a wall (spill behind it) and noisy with the particle arrangement.
__global__ void k_wb_vr(int nvox, const double* vox_w, const double* voxJb, const double* voxV, const int* vox_cc, double rho, double dx3, double* vox_Vr,
                        double fill_guard = 0.0)   // LIQ_WB_GEOV=2: a full cell whose deposit fills < fill_guard of it keeps the deposit (sparse cells in a splash)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if(v >= nvox) return;
    double w = 0.0;
    for(int q = 0; q < 8; ++q) w += vox_w[8 * v + q];
    const double Jb = voxJb[v];
    if(vox_cc[v] == 0 && fabs(w / rho - dx3) > 1e-9 * dx3) { vox_Vr[v] = voxV[v]; return; }   // separate water: weights are the deposit (rest volume) -> keep it
    if(fill_guard > 0.0 && vox_cc[v] == 0 && voxV[v] * Jb < fill_guard * dx3) { vox_Vr[v] = voxV[v]; return; }
    vox_Vr[v] = (w > 0.0 && Jb > 0.0) ? w / (rho * Jb) : 0.0;                                 // full / cut: geometric current volume / Jb
}
__global__ void k_cell2elem(int nact, const int* active_cells, int* cell2elem)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if(v >= nact) return;
    cell2elem[active_cells[v]] = v;
}
__global__ void k_particle_Jcell(int n, const double3* xp, const double* Jbc, const int* cflag, double3 org, double dx, int3 dims, double* Jp)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    CentreStencil s = centre_stencil(xp[p], org, dx, dims);
    double J = 0.0, wsum = 0.0;
    for(int a = 0; a < 2; ++a)
        for(int b = 0; b < 2; ++b)
            for(int c = 0; c < 2; ++c)
            {
                double w = s.w[0][a] * s.w[1][b] * s.w[2][c];
                int    cid = cell_lin(s.b[0] + a, s.b[1] + b, s.b[2] + c, dims);
                if(w <= 0.0 || !cflag[cid]) continue;
                J += w * Jbc[cid];
                wsum += w;
            }
    Jp[p] = wsum > 0.0 ? J / wsum : 1.0;
}

// LIQ_JFILT: hydrostatically corrected null-space filter of the particle volume state. The particles carry more J values than
// the cell-level pressure can represent; the excess (particle-to-particle scatter inside a cell, ~4e-4 = 40 Pa at rest, as large as
// the hydrostatic J step between neighbouring cells) is invisible to the solve until the particles move, then every move re-deposits
// it as a pressure kick. Filter only the deviation from the hydrostatic J of the particle's column (delta-SPH with the hydrostatic
// part excluded, Antuono et al. 2012; null-space filter, Tran et al. 2019): a = J - J_hyd -> cell-centre average -> back to the
// particle; J += beta (a_smooth - a). A hydrostatic state (or any grid-resolved anomaly) is a fixed point; total sum V0 J conserved.
__device__ __forceinline__ bool jf_hyd(double3 x, int3 dims, double3 org, double dx, int U, double3 up, const double* hcol, const int* blo,
                                       double rgk, double& Jh)
{
    const int ijk[3] = {(int)floor((x.x - org.x) / dx), (int)floor((x.y - org.y) / dx), (int)floor((x.z - org.z) / dx)};
    const int n[3]   = {dims.x, dims.y, dims.z};
    for(int a = 0; a < 3; ++a)
        if(ijk[a] < 0 || ijk[a] >= n[a]) return false;
    const int col = (U == 0) ? ijk[1] * dims.z + ijk[2] : (U == 1) ? ijk[0] * dims.z + ijk[2] : ijk[0] * dims.y + ijk[1];
    if(blo[col] < 0) return false;
    const double h = hcol[col];
    if(!(h > -1e299)) return false;
    const double y = up.x * x.x + up.y * x.y + up.z * x.z;
    if(y > h + 0.5 * dx) return false;   // above the column's water body (spray, a droplet): left alone
    Jh = 1.0 - rgk * fmax(h - y, 0.0);
    return true;
}
__global__ void k_jf_deposit(int n, const double3* xp, const double* Jp, const double* Vp0, int3 dims, double3 org, double dx, int U, double3 up,
                             const double* hcol, const int* blo, double rgk, double* num, double* den)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    double Jh;
    if(!jf_hyd(xp[p], dims, org, dx, U, up, hcol, blo, rgk, Jh)) return;
    const double  a = Jp[p] - Jh, V = Vp0[p];
    CentreStencil s = centre_stencil(xp[p], org, dx, dims);
    for(int i = 0; i < 2; ++i)
        for(int j = 0; j < 2; ++j)
            for(int k = 0; k < 2; ++k)
            {
                const double w = s.w[0][i] * s.w[1][j] * s.w[2][k];
                if(w <= 0.0) continue;
                const int cid = cell_lin(s.b[0] + i, s.b[1] + j, s.b[2] + k, dims);
                atomicAdd(&num[cid], w * V * a);
                atomicAdd(&den[cid], w * V);
            }
}
__global__ void k_jf_apply(int n, const double3* xp, double* Jp, const double* Vp0, int3 dims, double3 org, double dx, int U, double3 up,
                           const double* hcol, const int* blo, double rgk, const double* num, const double* den, double beta, int* flag, double* red)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    flag[p] = 0;
    double Jh;
    if(!jf_hyd(xp[p], dims, org, dx, U, up, hcol, blo, rgk, Jh)) return;
    CentreStencil s = centre_stencil(xp[p], org, dx, dims);
    double as = 0.0, ws = 0.0;
    for(int i = 0; i < 2; ++i)
        for(int j = 0; j < 2; ++j)
            for(int k = 0; k < 2; ++k)
            {
                const double w = s.w[0][i] * s.w[1][j] * s.w[2][k];
                if(w <= 0.0) continue;
                const int cid = cell_lin(s.b[0] + i, s.b[1] + j, s.b[2] + k, dims);
                if(!(den[cid] > 0.0)) continue;
                as += w * num[cid] / den[cid];
                ws += w;
            }
    if(!(ws > 0.0)) return;
    const double dJ = beta * (as / ws - (Jp[p] - Jh));
    Jp[p] += dJ;
    flag[p] = 1;
    atomicAdd(&red[0], Vp0[p] * dJ);
    atomicAdd(&red[1], Vp0[p]);
}
__global__ void k_jf_conserve(int n, double* Jp, const int* flag, const double* red)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n || !flag[p] || !(red[1] > 0.0)) return;
    Jp[p] -= red[0] / red[1];
}

// j_mode 3: particles whose stencil cells are all (nearly) full take the exact cell
// volume ratio; particles touching under-filled (free-surface) cells keep their
// tracked J so that compaction of the surface layer still builds pressure.
__global__ void k_reanchor_J(int n, const double3* xp, const double* mc, const int* cflag, double3 org, double dx, int3 dims, double rho_cell, double Jmin, double beta, double* Jp)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    CentreStencil s = centre_stencil(xp[p], org, dx, dims);
    double ratio = 0.0, wsum = 0.0, rmax = 0.0;
    for(int a = 0; a < 2; ++a)
        for(int b = 0; b < 2; ++b)
            for(int c = 0; c < 2; ++c)
            {
                double w   = s.w[0][a] * s.w[1][b] * s.w[2][c];
                int    cid = cell_lin(s.b[0] + a, s.b[1] + b, s.b[2] + c, dims);
                if(w <= 1e-3 || !cflag[cid]) continue;
                double r = rho_cell / mc[cid];
                ratio += w * r;
                wsum += w;
                rmax = fmax(rmax, r);
            }
    if(wsum <= 0.0) return;
    ratio /= wsum;
    if(rmax < 1.05)  // interior: all stencil cells full -> move toward the exact cell-level state
    {
        double J = Jp[p] + beta * (ratio - Jp[p]);
        { double Jc = fmin(fmax(J, Jmin), 2.0); if(Jc != J) LIQ_FBK(1); Jp[p] = Jc; }
    }
}

// ----------------------------------------------------------------------------
// j_mode 4: geometric fill fraction of the active cells.
// The plain cell ratio (j_mode 2) reads a half-full surface cell as "expanded
// 2x" (no pressure until its top layer has compacted into the cell below), and
// the tent-smeared mass of the top layer makes the cell below look light: the
// free surface and every wall layer sink by up to a cell, and stirring compacts
// further.  Tracked J (modes 0/3) has no such bias but integrates the velocity
// noise of the free-surface nodes, which the no-tension branch rectifies into a
// net pressure (the slow "foam" creep).  Here the water region is rebuilt from
// the positions: (particles (+) r_d) (-) r_e with r_d - r_e = h/2, i.e. the union
// of particle balls closed by r_e: gap-free inside, boundary exactly on the
// particle cube faces, plus each particle's own h^3-equivalent ball so that a
// sparse spray never reads as compressed.  Sampled on a dx/4 sub-grid and
// tent-smoothed like the mass, J_c = fill_c rho0 dx^3 / m_c is 1 for a nominal
// lattice cut by any plane and < 1 only where particles pack denser than nominal.
// ----------------------------------------------------------------------------
#define LIQ_ERODE_MAX 300
__constant__ signed char c_erode_off[3 * LIQ_ERODE_MAX];
__constant__ int         c_erode_n;

__device__ __forceinline__ bool sub_in_domain(int I, int J, int K, int3 d)
{
    return I >= 0 && J >= 0 && K >= 0 && I < 4 * d.x && J < 4 * d.y && K < 4 * d.z;
}
__device__ __forceinline__ size_t sub_index(int I, int J, int K, int3 d)
{   // dense sub-grid, cell-major: 64 samples per cell, local index si*16 + sj*4 + sk
    return (size_t)cell_lin(I >> 2, J >> 2, K >> 2, d) * 64 + ((I & 3) << 4) + ((J & 3) << 2) + (K & 3);
}
__global__ void k_near_flag(int ncell, const int* cflag, int3 d, int* near)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if(c >= ncell) return;
    int i, j, k;
    cell_ijk(c, d, i, j, k);
    int f = 0;
    for(int a = -1; a <= 1 && !f; ++a)
        for(int b = -1; b <= 1 && !f; ++b)
            for(int e = -1; e <= 1 && !f; ++e)
            {
                int ii = i + a, jj = j + b, kk = k + e;
                if(ii < 0 || jj < 0 || kk < 0 || ii >= d.x || jj >= d.y || kk >= d.z) continue;
                if(cflag[cell_lin(ii, jj, kk, d)]) f = 1;
            }
    near[c] = f;
}
__global__ void k_count1(int n, const int* flag, const int* scan, int* out) { out[0] = scan[n - 1] + flag[n - 1]; }
// pass A: squared distance of every sub-sample of the near cells to the nearest particle
__global__ void k_fill_sample(int nnear, const int* near_cells, int3 d, double3 org, double dx, const double3* xp,
                              const int* pidx, const int* cstart, const int* cend, double rd2, double ro2, unsigned char* chi)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if(t >= nnear * 64) return;
    int c = near_cells[t >> 6], s = t & 63;
    int i, j, k;
    cell_ijk(c, d, i, j, k);
    double3 x = make_double3(org.x + (i + ((s >> 4) + 0.5) * 0.25) * dx, org.y + (j + (((s >> 2) & 3) + 0.5) * 0.25) * dx,
                             org.z + (k + ((s & 3) + 0.5) * 0.25) * dx);
    double best = 1e300;
    for(int a = -1; a <= 1; ++a)
        for(int b = -1; b <= 1; ++b)
            for(int e = -1; e <= 1; ++e)
            {
                int ii = i + a, jj = j + b, kk = k + e;
                if(ii < 0 || jj < 0 || kk < 0 || ii >= d.x || jj >= d.y || kk >= d.z) continue;
                int cid = cell_lin(ii, jj, kk, d);
                int s0  = cstart[cid];
                if(s0 < 0) continue;
                int s1 = cend[cid];
                for(int q = s0; q < s1; ++q)
                {
                    double3 dv = d3sub(x, xp[pidx[q]]);
                    best       = fmin(best, dv.x * dv.x + dv.y * dv.y + dv.z * dv.z);
                }
            }
    chi[(size_t)c * 64 + s] = (unsigned char)((best < rd2 ? 1 : 0) | (best < ro2 ? 2 : 0));
}
// pass B: erode bit0 by the sub-grid ball (c_erode_off), OR the own-ball bit
__global__ void k_fill_erode(int nnear, const int* near_cells, int3 d, const unsigned char* chi, unsigned char* chi2)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if(t >= nnear * 64) return;
    int           c = near_cells[t >> 6], s = t & 63;
    unsigned char v = chi[(size_t)c * 64 + s];
    int           i, j, k;
    cell_ijk(c, d, i, j, k);
    int I = 4 * i + (s >> 4), J = 4 * j + ((s >> 2) & 3), K = 4 * k + (s & 3);
    int in = (v & 1);
    for(int o = 0; o < c_erode_n && in; ++o)
    {
        int II = I + c_erode_off[3 * o], JJ = J + c_erode_off[3 * o + 1], KK = K + c_erode_off[3 * o + 2];
        if(!sub_in_domain(II, JJ, KK, d) || !(chi[sub_index(II, JJ, KK, d)] & 1)) in = 0;
    }
    chi2[(size_t)c * 64 + s] = (unsigned char)(in | ((v >> 1) & 1));
}
// pass C: tent-smoothed fill of each active cell -> J_c = fill rho0 dx^3 / m_c
__global__ void k_fill_apply(int nactive, const int* active, int3 d, const unsigned char* chi2, const double* mc, double rho_cell,
                             double Jmin, double* Jbc, double* fill_out)
{
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if(a >= nactive) return;
    int c = active[a];
    int i, j, k;
    cell_ijk(c, d, i, j, k);
    double acc = 0.0, wsum = 0.0;
    // tent support of cell i is (i-0.5, i+1.5) in cell units: sub-samples I with (I+0.5)/4 in it, i.e. 4i-2 .. 4i+5
    for(int I = 4 * i - 2; I <= 4 * i + 5; ++I)
    {
        double wx = 1.0 - fabs((I + 0.5) * 0.25 - (i + 0.5));
        if(wx <= 0.0 || I < 0 || I >= 4 * d.x) continue;
        for(int J = 4 * j - 2; J <= 4 * j + 5; ++J)
        {
            double wy = 1.0 - fabs((J + 0.5) * 0.25 - (j + 0.5));
            if(wy <= 0.0 || J < 0 || J >= 4 * d.y) continue;
            for(int K = 4 * k - 2; K <= 4 * k + 5; ++K)
            {
                double wz = 1.0 - fabs((K + 0.5) * 0.25 - (k + 0.5));
                if(wz <= 0.0 || K < 0 || K >= 4 * d.z) continue;
                double w = wx * wy * wz;
                wsum += w;
                if(chi2[sub_index(I, J, K, d)]) acc += w;
            }
        }
    }
    double fill = wsum > 0.0 ? acc / wsum : 1.0;
    fill_out[a] = fill;
    double m    = mc[c];
    double Jb   = (m > 0.0) ? fill * rho_cell / m : 1.0;
    { double Jb_c = fmin(fmax(Jb, Jmin), 8.0); if(Jb_c != Jb) LIQ_FBK(2); Jbc[c] = Jb_c; }
}

__global__ void k_write_particle_slots(int n, const double3* xp, int base, double3* vert, double3* overt, double3* xt)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    vert[base + p]  = xp[p];
    overt[base + p] = xp[p];
    xt[base + p]    = xp[p];
}

// ----------------------------------------------------------------------------
// voxel water element: E = V psi(J), J = det(A) Jb, A = I + sum_q u_q g_q^T
// ----------------------------------------------------------------------------
__device__ __forceinline__ void cofactor3(const double A[3][3], double C[3][3])
{
    C[0][0] = A[1][1] * A[2][2] - A[1][2] * A[2][1];
    C[0][1] = A[1][2] * A[2][0] - A[1][0] * A[2][2];
    C[0][2] = A[1][0] * A[2][1] - A[1][1] * A[2][0];
    C[1][0] = A[0][2] * A[2][1] - A[0][1] * A[2][2];
    C[1][1] = A[0][0] * A[2][2] - A[0][2] * A[2][0];
    C[1][2] = A[0][1] * A[2][0] - A[0][0] * A[2][1];
    C[2][0] = A[0][1] * A[1][2] - A[0][2] * A[1][1];
    C[2][1] = A[0][2] * A[1][0] - A[0][0] * A[1][2];
    C[2][2] = A[0][0] * A[1][1] - A[0][1] * A[1][0];
}
__device__ __forceinline__ double levi(int a, int b, int c)
{
    if(a == b || b == c || a == c) return 0.0;
    return ((a == 0 && b == 1 && c == 2) || (a == 1 && b == 2 && c == 0) || (a == 2 && b == 0 && c == 1)) ? 1.0 : -1.0;
}

// per-voxel state used by both assembly and energy
struct VoxState
{
    double3 g[8];
    double  A[3][3];
    double  detA;
    double  J;
};
// LIQ_VOX_BAR: barrier on the volume ratio J = Jb det A of a voxel, b(J) = -(J - Jb)^2 ln(J / Jb) for 0 < J < Jb (C2 at the
// threshold Jb = LIQ_VOX_BAR, zero above it). The quadratic model 1/2 kappa (J - 1)^2 is finite at J = 0, so when a sparse cell
// (falling jet, film, free surface) is folded by its independently moving corners the Newton direction keeps pushing it into the
// line search's det A <= 0 wall (energy 1e300): every step size is rejected and the same direction comes back until the iteration
// cap. With the barrier the model sees the wall. It is on the PHYSICAL ratio J, not on the in-step det A: water never reaches
// J < 0.3 legitimately, while a sparse cell (J_base >> 1) legitimately reaches det A ~ 0.3 at J ~ 1 in an impact -- a barrier on
// det A flung such cells apart (spill crossings 13 -> 76, pour blow-up at 100 deg). Returns b, db/dJ, d2b/dJ2.
__device__ __forceinline__ void vox_barrier(double a, double ab, double& b0, double& b1, double& b2)
{
    b0 = b1 = b2 = 0.0;
    if(!(ab > 0.0) || a >= ab || a <= 0.0) return;
    const double d = a - ab, l = log(a / ab);
    b0 = -d * d * l;
    b1 = -2.0 * d * l - d * d / a;
    b2 = -2.0 * l - 4.0 * d / a + d * d / (a * a);
}
// LIQ_WB_DET: the same mean deformation gradient for a cut element, with g_q = a_q / V_g (the sub-volume's non-wall boundary
// vectors): first-order force identical to the linear well-balanced operator (balance preserved), but J = Jb det(I + sum u_q g_q^T)
// carries the pressure-weighted geometric stiffness (psi1 Jb d^2det) that the linearised J lacks -- the only implicit restoring
// stiffness of divergence-free (surface-wave) modes. Full cells already have it (vox_state).
__device__ __forceinline__ void vox_state_g(const int* nodes, const int3 ijk, const double3* vert, double dx, double3 org, double Jb, const double3* a, double Vg, VoxState& st)
{
    for(int r = 0; r < 3; ++r)
        for(int s = 0; s < 3; ++s) st.A[r][s] = (r == s) ? 1.0 : 0.0;
    const double inv = 1.0 / Vg;
    for(int q = 0; q < 8; ++q)
    {
        int     ia = (q >> 2) & 1, ib = (q >> 1) & 1, ic = q & 1;
        double3 x0 = make_double3(org.x + (ijk.x + ia) * dx, org.y + (ijk.y + ib) * dx, org.z + (ijk.z + ic) * dx);
        double3 u  = d3sub(vert[nodes[q]], x0);
        st.g[q]    = d3scale(a[q], inv);
        double ua[3] = {u.x, u.y, u.z};
        double gb[3] = {st.g[q].x, st.g[q].y, st.g[q].z};
        for(int r = 0; r < 3; ++r)
            for(int s = 0; s < 3; ++s) st.A[r][s] += ua[r] * gb[s];
    }
    double C[3][3];
    cofactor3(st.A, C);
    st.detA = st.A[0][0] * C[0][0] + st.A[0][1] * C[0][1] + st.A[0][2] * C[0][2];
    st.J    = st.detA * Jb;
}
__device__ __forceinline__ void vox_state(const int* nodes, const int3 ijk, const double3* vert, double dx, double3 org, double Jb, VoxState& st)
{
    for(int a = 0; a < 3; ++a)
        for(int b = 0; b < 3; ++b) st.A[a][b] = (a == b) ? 1.0 : 0.0;
    for(int q = 0; q < 8; ++q)
    {
        int     a = (q >> 2) & 1, b = (q >> 1) & 1, c = q & 1;
        double3 x0 = make_double3(org.x + (ijk.x + a) * dx, org.y + (ijk.y + b) * dx, org.z + (ijk.z + c) * dx);
        double3 u  = d3sub(vert[nodes[q]], x0);
        double3 sg = corner_sign(q);
        st.g[q]    = d3scale(sg, 1.0 / (4.0 * dx));
        double ua[3] = {u.x, u.y, u.z};
        double gb[3] = {st.g[q].x, st.g[q].y, st.g[q].z};
        for(int r = 0; r < 3; ++r)
            for(int s = 0; s < 3; ++s) st.A[r][s] += ua[r] * gb[s];
    }
    double C[3][3];
    cofactor3(st.A, C);
    st.detA = st.A[0][0] * C[0][0] + st.A[0][1] * C[0][1] + st.A[0][2] * C[0][2];
    st.J    = st.detA * Jb;
}

#include "liquid_wb.cuh"
// ---- LIQ_CUTCELL: fluid sub-volume of wall-cut cells (see LIQUID_CUTCELL_DESIGN_2026-10-23.md) ----
__global__ void k_cc_mark(int n, const int* valid, const int3* wcell, int3 d, const double3* wX, const double3* wn, double3* nsum, double3* xsum, int* cnt)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n || !valid[p]) return;
    int3 c = wcell[p];
    if(c.x < 0 || c.y < 0 || c.z < 0 || c.x >= d.x || c.y >= d.y || c.z >= d.z) return;
    int cl = cell_lin(c.x, c.y, c.z, d);
    atomicAdd(&nsum[cl].x, wn[p].x);
    atomicAdd(&nsum[cl].y, wn[p].y);
    atomicAdd(&nsum[cl].z, wn[p].z);
    atomicAdd(&xsum[cl].x, wX[p].x);
    atomicAdd(&xsum[cl].y, wX[p].y);
    atomicAdd(&xsum[cl].z, wX[p].z);
    atomicAdd(&cnt[cl], 1);
}
__device__ __forceinline__ double cc_N(int q, double3 x)   // trilinear shape function of corner q on the unit cell
{
    int i = (q >> 2) & 1, j = (q >> 1) & 1, k = q & 1;
    return (i ? x.x : 1.0 - x.x) * (j ? x.y : 1.0 - x.y) * (k ? x.z : 1.0 - x.z);
}
// per element: area vectors a_q of the fluid-side face caps (unit cell, scaled by dx^2) and the fluid sub-volume (divergence theorem;
// the wall polygon's area follows from closure, A_w = n . sum_q a_q)
__global__ void k_cc_elem(int nvox, const int3* voxijk, int3 d, const double3* nsum, const double3* xsum, const int* cnt, double3 org, double dx,
                          double vmin, int* vox_cc, double3* vox_a, double* vox_vg, int* ncut)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if(v >= nvox) return;
    vox_cc[v] = 0;
    int3 c  = voxijk[v];
    int  cl = cell_lin(c.x, c.y, c.z, d);
    int  k  = cnt[cl];
    if(k <= 0) return;
    double3 ns = nsum[cl];
    double  ln = sqrt(ns.x * ns.x + ns.y * ns.y + ns.z * ns.z);
    if(ln < 0.5 * k) return;   // wall points facing opposite ways in one cell (thin wall): keep the plain element
    double3 n  = d3scale(ns, 1.0 / ln);
    double3 xw = d3scale(xsum[cl], 1.0 / k);
    double3 X  = make_double3((xw.x - org.x) / dx - c.x, (xw.y - org.y) / dx - c.y, (xw.z - org.z) / dx - c.z);   // unit-cell frame
    double3 a[8];
    for(int q = 0; q < 8; ++q) a[q] = make_double3(0, 0, 0);
    double vol3 = 0.0;
    for(int ax = 0; ax < 3; ++ax)
        for(int side = 0; side < 2; ++side)
        {
            double3 nf = make_double3(ax == 0 ? (side ? 1.0 : -1.0) : 0.0, ax == 1 ? (side ? 1.0 : -1.0) : 0.0, ax == 2 ? (side ? 1.0 : -1.0) : 0.0);
            double3 P[4];
            int     Q[4];
            const int uv[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
            for(int t = 0; t < 4; ++t)
            {
                double cc[3];
                int    ii[3];
                int    o0 = ax == 0 ? 1 : 0, o1 = ax == 2 ? 1 : 2;
                cc[ax] = side; cc[o0] = uv[t][0]; cc[o1] = uv[t][1];
                ii[ax] = side; ii[o0] = uv[t][0]; ii[o1] = uv[t][1];
                P[t] = make_double3(cc[0], cc[1], cc[2]);
                Q[t] = (ii[0] << 2) | (ii[1] << 1) | ii[2];
            }
            double3 cap[8];
            int     m = 0;
            for(int t = 0; t < 4; ++t)
            {
                double3 pa = P[t], pb = P[(t + 1) & 3];
                double  da = n.x * (pa.x - X.x) + n.y * (pa.y - X.y) + n.z * (pa.z - X.z);
                double  db = n.x * (pb.x - X.x) + n.y * (pb.y - X.y) + n.z * (pb.z - X.z);
                if(da >= 0.0) cap[m++] = pa;
                if((da >= 0.0) != (db >= 0.0))
                {
                    double tt = da / (da - db);
                    cap[m++]  = make_double3(pa.x + tt * (pb.x - pa.x), pa.y + tt * (pb.y - pa.y), pa.z + tt * (pb.z - pa.z));
                }
            }
            for(int t = 1; t + 1 < m; ++t)
            {   // fan triangle (cap0, cap_t, cap_t+1), edge-midpoint rule (exact for the bilinear shape functions on the face)
                double3 A = cap[0], B = cap[t], C = cap[t + 1];
                double3 e1 = d3sub(B, A), e2 = d3sub(C, A);
                double3 cr = make_double3(e1.y * e2.z - e1.z * e2.y, e1.z * e2.x - e1.x * e2.z, e1.x * e2.y - e1.y * e2.x);
                double  ar = 0.5 * sqrt(cr.x * cr.x + cr.y * cr.y + cr.z * cr.z);
                double3 m1 = d3scale(d3add(A, B), 0.5), m2 = d3scale(d3add(B, C), 0.5), m3 = d3scale(d3add(C, A), 0.5);
                for(int u = 0; u < 4; ++u)
                {
                    double w = ar / 3.0 * (cc_N(Q[u], m1) + cc_N(Q[u], m2) + cc_N(Q[u], m3));
                    a[Q[u]]  = d3add(a[Q[u]], d3scale(nf, w));
                }
                vol3 += ar / 3.0 * ((m1.x * nf.x + m1.y * nf.y + m1.z * nf.z) + (m2.x * nf.x + m2.y * nf.y + m2.z * nf.z) + (m3.x * nf.x + m3.y * nf.y + m3.z * nf.z));
            }
        }
    double3 sa = make_double3(0, 0, 0);
    for(int q = 0; q < 8; ++q) sa = d3add(sa, a[q]);
    double Aw = n.x * sa.x + n.y * sa.y + n.z * sa.z;
    double Vg = (vol3 - (n.x * X.x + n.y * X.y + n.z * X.z) * Aw) / 3.0;   // wall polygon: x.(-n) = -n.X on the plane
    if(!(Vg > 1e-6) || Vg > 0.999) return;                                 // no fluid side / essentially uncut: plain element
    vox_cc[v] = 1;
    for(int q = 0; q < 8; ++q) vox_a[8 * v + q] = d3scale(a[q], dx * dx);
    vox_vg[v] = fmax(Vg, vmin) * dx * dx * dx;
    atomicAdd(ncut, 1);
}
// cut element: J = Jb (1 + sum_q a_q . u_q / V_g)
__device__ __forceinline__ double cc_J(const int* nodes, const int3 ijk, const double3* vert, double dx, double3 org, double Jb, const double3* a, double Vg)
{
    double dv = 0.0;
    for(int q = 0; q < 8; ++q)
    {
        int     ia = (q >> 2) & 1, ib = (q >> 1) & 1, ic = q & 1;
        double3 x0 = make_double3(org.x + (ijk.x + ia) * dx, org.y + (ijk.y + ib) * dx, org.z + (ijk.z + ic) * dx);
        double3 u  = d3sub(vert[nodes[q]], x0);
        dv += a[q].x * u.x + a[q].y * u.y + a[q].z * u.z;
    }
    return Jb * (1.0 + dv / Vg);
}

__global__ void k_vox_assemble(int              nvox,
                               const int*       vox_nodes,
                               const double*    voxV,
                               const double*    voxJb,
                               const int3*      voxijk,
                               const double3*   vert,
                               double           dx,
                               double3          org,
                               double           kappa,
                               double           tension_ratio,
                               double           dt,
                               int              fem_off,
                               int              global_offset,
                               Eigen::Matrix3d* vals,
                               int*             rows,
                               int*             cols,
                               const int*       vox_cc = nullptr,   // LIQ_CUTCELL
                               const double3*   vox_a  = nullptr,
                               const double*    vox_vg = nullptr,
                               double           vbar   = 0.0,       // LIQ_VOX_BAR threshold
                               double           vbar_kk = 0.0,      // barrier stiffness (kappa units already applied)
                               const double3*   vox_f  = nullptr,   // LIQ_WB: constant hydrostatic forces (gradient -dt^2 f)
                               const double*    vox_hg = nullptr,   // LIQ_WB_HG: per element 6 x (V_T, a^T_q) sub-cell records
                               double           hgk    = 0.0,       // hg x kappa
                               int              wb_det = 0,         // LIQ_WB_DET: cut elements use the exact determinant (vox_state_g)
                               const double*    vox_s  = nullptr,   // LIQ_WB_SURFK: per element int_S N_q N_r dA_h
                               double           sk     = 0.0,       // surfk x rho g
                               double3          upv    = make_double3(0, 1, 0),
                               const double*    vox_Vr = nullptr,   // LIQ_WB_GEOV: rest volume = geometric volume / Jb (force = pressure x geometric area)
                               int              wb_fullf = 0)       // LIQ_WB_FULLF: full cells add their exact hydrostatic term too (gradient -dt^2 f)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if(v >= nvox) return;
    const int* nodes = vox_nodes + 8 * v;
    int        toff  = global_offset + 36 * v;
    double     V     = voxV[v];
    double     Jb    = voxJb[v];
    const bool det_cut = vox_cc && vox_cc[v] && wb_det && vox_f;
    if(vox_cc && vox_cc[v] && !det_cut)
    {   // cut element: energy V psi(J), J linear in the node displacements -> gradient V psi' (Jb / V_g) a_q, Hessian V psi'' (Jb / V_g)^2 a_q a_r^T
        const double3* a  = vox_a + 8 * v;
        double         Vg = vox_vg[v];
        double         J  = cc_J(nodes, voxijk[v], vert, dx, org, Jb, a, Vg);
        int            kk = 0;
        double         keff = (J < 1.0) ? kappa : kappa * tension_ratio;
        double         bb0, bb1, bb2;
        vox_barrier(J, vbar, bb0, bb1, bb2);
        const bool     bar = bb1 != 0.0 || bb2 != 0.0;
        // LIQ_WB: the element IS its geometric fluid sub-volume. LIQ_CUTCELL (research): weight min(V_c, V_g) -- a kernel-spread deposit
        // larger than the geometric fluid sub-volume must not multiply the pressure force (V_c / V_g up to 4 at the 0.25 dx^3 floor)
        V = vox_Vr ? vox_Vr[v] : (vox_f ? Vg : fmin(V, Vg));
        bool           ok   = V > 0.0 && isfinite(J) && (keff > 0.0 || bar);
        const double   psi1 = keff * (J - 1.0) + (bar ? vbar_kk * bb1 : 0.0), psi2 = keff + (bar ? vbar_kk * bb2 : 0.0);
        double         dt2  = dt * dt, c1 = ok ? dt2 * V * psi1 * Jb / Vg : 0.0, c2 = ok ? dt2 * V * psi2 * (Jb / Vg) * (Jb / Vg) : 0.0;
        // LIQ_WB_HG: sub-cell volumetric consistency. delta_T = sum_q a^T_q.u_q / V_T (piece T), delta_c = sum_q a_q.u_q / V_g (cell):
        // E = 1/2 hgk sum_T V_T (delta_T - delta_c)^2 -- the second-order part of per-piece quadrature of the volumetric energy
        // around the cell mean. Zero for any affine motion (uniform divergence); it only resists the non-affine (hourglass) part.
        int     nT = 0;
        double  cT[6], sT[6];
        double3 dT[6][8];
        if(vox_hg && hgk > 0.0 && Vg > 0.0)
        {
            double3 uu[8];
            for(int q = 0; q < 8; ++q)
            {
                const int     ia = (q >> 2) & 1, ib = (q >> 1) & 1, ic = q & 1;
                const double3 x0 = make_double3(org.x + (voxijk[v].x + ia) * dx, org.y + (voxijk[v].y + ib) * dx, org.z + (voxijk[v].z + ic) * dx);
                uu[q] = d3sub(vert[nodes[q]], x0);
            }
            const double* rec = vox_hg + 150 * v;
            for(int T = 0; T < 6; ++T)
            {
                const double VT = rec[25 * T];
                if(!(VT > 1e-12 * Vg)) continue;
                double sum = 0.0;
                for(int q = 0; q < 8; ++q)
                {
                    const double3 aT = make_double3(rec[25 * T + 1 + 3 * q], rec[25 * T + 2 + 3 * q], rec[25 * T + 3 + 3 * q]);
                    dT[nT][q] = make_double3(aT.x / VT - a[q].x / Vg, aT.y / VT - a[q].y / Vg, aT.z / VT - a[q].z / Vg);
                    sum += dT[nT][q].x * uu[q].x + dT[nT][q].y * uu[q].y + dT[nT][q].z * uu[q].z;
                }
                cT[nT] = dt2 * hgk * VT;
                sT[nT] = sum;
                ++nT;
            }
        }
        // LIQ_WB_SURFK: E_s = 1/2 sk sum_qr S_qr (up.u_q)(up.u_r); gradient sk sum_r S_qr (up.u_r) up; blocks sk S_qr up up^T
        double sv[8];
        const double* Sm = (vox_s && sk > 0.0) ? vox_s + 36 * v : nullptr;
        if(Sm)
        {
            double un[8];
            for(int q = 0; q < 8; ++q)
            {
                const int     ia = (q >> 2) & 1, ib = (q >> 1) & 1, ic = q & 1;
                const double3 x0 = make_double3(org.x + (voxijk[v].x + ia) * dx, org.y + (voxijk[v].y + ib) * dx, org.z + (voxijk[v].z + ic) * dx);
                const double3 u  = d3sub(vert[nodes[q]], x0);
                un[q] = upv.x * u.x + upv.y * u.y + upv.z * u.z;
            }
            for(int q = 0; q < 8; ++q)
            {
                double acc = 0.0;
                for(int r = 0; r < 8; ++r) { const int qq = q < r ? q : r, rr = q < r ? r : q; acc += Sm[qq * 8 - qq * (qq + 1) / 2 + rr] * un[r]; }
                sv[q] = dt2 * sk * acc;
            }
        }
        if(ok || (vox_f && V > 0.0) || nT > 0 || Sm)
            for(int q = 0; q < 8; ++q)
            {
                double f[3] = {c1 * a[q].x, c1 * a[q].y, c1 * a[q].z};
                if(vox_f && V > 0.0) { f[0] -= dt2 * vox_f[8 * v + q].x; f[1] -= dt2 * vox_f[8 * v + q].y; f[2] -= dt2 * vox_f[8 * v + q].z; }
                for(int T = 0; T < nT; ++T) { const double cs = cT[T] * sT[T]; f[0] += cs * dT[T][q].x; f[1] += cs * dT[T][q].y; f[2] += cs * dT[T][q].z; }
                if(Sm) { f[0] += sv[q] * upv.x; f[1] += sv[q] * upv.y; f[2] += sv[q] * upv.z; }
                for(int b = 0; b < 3; ++b) binned_deposit(g_gbin + ((size_t)nodes[q] * 3 + b) * BINNED_K, f[b]);
            }
        for(int q = 0; q < 8; ++q)
            for(int r = q; r < 8; ++r)
            {
                int row = nodes[q] + fem_off, col = nodes[r] + fem_off;
                Eigen::Matrix3d B;
                double aq[3] = {a[q].x, a[q].y, a[q].z}, ar[3] = {a[r].x, a[r].y, a[r].z};
                for(int i = 0; i < 3; ++i)
                    for(int j = 0; j < 3; ++j) B(i, j) = c2 * aq[i] * ar[j];
                for(int T = 0; T < nT; ++T)
                {
                    const double dq[3] = {dT[T][q].x, dT[T][q].y, dT[T][q].z}, dr[3] = {dT[T][r].x, dT[T][r].y, dT[T][r].z};
                    for(int i = 0; i < 3; ++i)
                        for(int j = 0; j < 3; ++j) B(i, j) += cT[T] * dq[i] * dr[j];
                }
                if(Sm)
                {
                    const double cs = dt2 * sk * Sm[q * 8 - q * (q + 1) / 2 + r];
                    const double uv[3] = {upv.x, upv.y, upv.z};
                    for(int i = 0; i < 3; ++i)
                        for(int j = 0; j < 3; ++j) B(i, j) += cs * uv[i] * uv[j];
                }
                if(row <= col) { rows[toff + kk] = row; cols[toff + kk] = col; vals[toff + kk] = B; }
                else { rows[toff + kk] = col; cols[toff + kk] = row; vals[toff + kk] = B.transpose(); }
                ++kk;
            }
        return;
    }
    VoxState   st;
    if(det_cut)
    {   // LIQ_WB_DET cut element: the sub-volume's own mean deformation gradient + the well-balanced constant hydrostatic force
        const double Vg = vox_vg[v];
        V = Vg;
        if(Vg > 0.0)
        {
            vox_state_g(nodes, voxijk[v], vert, dx, org, Jb, vox_a + 8 * v, Vg, st);
            const double dt2f = dt * dt;
            for(int q = 0; q < 8; ++q)
            {
                const double f[3] = {-dt2f * vox_f[8 * v + q].x, -dt2f * vox_f[8 * v + q].y, -dt2f * vox_f[8 * v + q].z};
                for(int b = 0; b < 3; ++b) binned_deposit(g_gbin + ((size_t)nodes[q] * 3 + b) * BINNED_K, f[b]);
            }
        }
        else { st.J = 1.0; st.detA = 1.0; }
    }
    else
    {
        vox_state(nodes, voxijk[v], vert, dx, org, Jb, st);
        if(vox_Vr) V = vox_Vr[v];   // LIQ_WB_GEOV
        if(wb_fullf && vox_f)
        {   // LIQ_WB_FULLF: the full cell's exact hydrostatic term (zero for separate water and dead elements)
            const double dt2f = dt * dt;
            for(int q = 0; q < 8; ++q)
            {
                const double f[3] = {-dt2f * vox_f[8 * v + q].x, -dt2f * vox_f[8 * v + q].y, -dt2f * vox_f[8 * v + q].z};
                for(int b = 0; b < 3; ++b) binned_deposit(g_gbin + ((size_t)nodes[q] * 3 + b) * BINNED_K, f[b]);
            }
        }
    }

    // write all 36 slots (zero blocks keep the layout stable)
    int kk = 0;
    for(int q = 0; q < 8; ++q)
        for(int r = q; r < 8; ++r)
        {
            int row = nodes[q] + fem_off, col = nodes[r] + fem_off;
            if(row <= col)
            {
                rows[toff + kk] = row;
                cols[toff + kk] = col;
            }
            else
            {
                rows[toff + kk] = col;
                cols[toff + kk] = row;
            }
            vals[toff + kk].setZero();
            ++kk;
        }
    if(!(V > 0.0) || !isfinite(st.J)) return;
    double keff = (st.J < 1.0) ? kappa : kappa * tension_ratio;
    double bb0, bb1, bb2;
    vox_barrier(st.J, vbar, bb0, bb1, bb2);
    const bool bar = bb1 != 0.0 || bb2 != 0.0;
    if(keff <= 0.0 && !bar) return;

    double psi1 = keff * (st.J - 1.0);
    double psi2 = keff;
    if(bar)
    {   // psi(J) += kb b(J)
        psi1 += vbar_kk * bb1;
        psi2 += vbar_kk * bb2;
    }
    double C[3][3];
    cofactor3(st.A, C);
    double dt2 = dt * dt;

    // gradient: f_q = dt^2 * V psi1 Jb * C g_q
    double s0 = dt2 * V * psi1 * Jb;
    for(int q = 0; q < 8; ++q)
    {
        double gq[3] = {st.g[q].x, st.g[q].y, st.g[q].z};
        for(int a = 0; a < 3; ++a)
        {
            double f = s0 * (C[a][0] * gq[0] + C[a][1] * gq[1] + C[a][2] * gq[2]);
            binned_deposit(g_gbin + ((size_t)nodes[q] * 3 + a) * BINNED_K, f);
        }
    }

    // Hessian w.r.t. A (9x9), index (a,b) -> 3a+b
    Eigen::Matrix<double, 9, 9> HA;
    for(int a = 0; a < 3; ++a)
        for(int b = 0; b < 3; ++b)
            for(int c = 0; c < 3; ++c)
                for(int d = 0; d < 3; ++d)
                {
                    double dcof = 0.0;
                    for(int n = 0; n < 3; ++n)
                        for(int q = 0; q < 3; ++q)
                        {
                            double e1 = levi(a, c, n);
                            if(e1 == 0.0) continue;
                            double e2 = levi(b, d, q);
                            if(e2 == 0.0) continue;
                            dcof += e1 * e2 * st.A[n][q];
                        }
                    HA(3 * a + b, 3 * c + d) = V * (psi2 * Jb * Jb * C[a][b] * C[c][d] + psi1 * Jb * dcof);
                }
    HA = 0.5 * (HA + HA.transpose().eval());
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 9, 9>> es(HA);
        if(es.eigenvalues()[0] < 0.0)
        {
            Eigen::Matrix<double, 9, 9> D = Eigen::Matrix<double, 9, 9>::Zero();
            for(int i = 0; i < 9; ++i)
                if(es.eigenvalues()[i] > 0.0) D(i, i) = es.eigenvalues()[i];
            HA = es.eigenvectors() * D * es.eigenvectors().transpose();
        }
    }
    HA *= dt2;

    // node blocks B_qr(a,c) = sum_{b,d} g_q[b] HA[(a,b),(c,d)] g_r[d]
    kk = 0;
    for(int q = 0; q < 8; ++q)
        for(int r = q; r < 8; ++r)
        {
            double gq[3] = {st.g[q].x, st.g[q].y, st.g[q].z};
            double gr[3] = {st.g[r].x, st.g[r].y, st.g[r].z};
            Eigen::Matrix3d B;
            for(int a = 0; a < 3; ++a)
                for(int c = 0; c < 3; ++c)
                {
                    double acc = 0.0;
                    for(int b = 0; b < 3; ++b)
                        for(int d = 0; d < 3; ++d) acc += gq[b] * HA(3 * a + b, 3 * c + d) * gr[d];
                    B(a, c) = acc;
                }
            int row = nodes[q] + fem_off, col = nodes[r] + fem_off;
            if(row <= col)
                vals[toff + kk] = B;
            else
                vals[toff + kk] = B.transpose();
            ++kk;
        }
}

// CPIC bookkeeping: the voxel (pressure) force the fluid exerts on the FIXED ghost nodes = the part of the liquid load that a
// Dirichlet ghost absorbs instead of the wall it stands in. Reported so the support-force budget closes (acceptance test 1).
__global__ void k_vox_ghost_force(int nvox, const int* vox_nodes, const double* voxV, const double* voxJb, const int3* voxijk, const double3* vert,
                                  double dx, double3 org, double kappa, double tension_ratio, int base, const int* active_nodes, const int* ghost, int n_active, double* out)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if(v >= nvox) return;
    const int* nodes = vox_nodes + 8 * v;
    double     V     = voxV[v];
    if(!(V > 0.0)) return;
    VoxState st;
    vox_state(nodes, voxijk[v], vert, dx, org, voxJb[v], st);
    if(!isfinite(st.J)) return;
    double keff = (st.J < 1.0) ? kappa : kappa * tension_ratio;
    if(keff <= 0.0) return;
    double psi1 = keff * (st.J - 1.0);
    double C[3][3];
    cofactor3(st.A, C);
    double s0 = V * psi1 * voxJb[v];
    for(int q = 0; q < 8; ++q)
    {
        int slot = nodes[q] - base;
        if(slot < 0) continue;
        bool isghost = slot >= n_active ? true : (ghost[active_nodes[slot]] != 0);   // private 2b ghost, or a fixed massless node
        if(!isghost) continue;
        double gq[3] = {st.g[q].x, st.g[q].y, st.g[q].z};
        for(int a = 0; a < 3; ++a)
        {
            double grad = s0 * (C[a][0] * gq[0] + C[a][1] * gq[1] + C[a][2] * gq[2]);   // dE/dx (per dt^2) -> force = -grad
            atomicAdd(&out[a], -grad);
        }
    }
}
static inline double3 liquid_up(double3 g) { const double m = sqrt(g.x * g.x + g.y * g.y + g.z * g.z); return m > 0.0 ? make_double3(-g.x / m, -g.y / m, -g.z / m) : make_double3(0, 1, 0); }
__device__ __forceinline__ double pdev_dot(double3 a, double3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
// ---- LIQ_VISC: implicit cell-mean Newtonian viscosity (see Params::visc) ----
// Per element: mean displacement gradient G = sum_q u_q g_q^T (g_q = corner-sign/(4dx) for a full cell, a_q / V_g for a cut one),
// S = sym(G). Incremental potential Psi = (mu/dt) V |S|^2 (energy slot; the Newton weights it by dt^2), gradient 2 mu dt V S g_q,
// Hessian block (q,r)(a,c) = mu dt V [delta_ac g_q.g_r + g_r[a] g_q[c]]. Added after the voxel assembly into the same 36 blocks.
__device__ __forceinline__ bool visc_setup(int v, const int* vox_nodes, const int3* voxijk, const int* vox_cc, const double3* vox_a, const double* vox_vg, const double3* vert,
                                           double dx, double3 org, double3* g, double3* uu, double& V)
{
    const int* nodes = vox_nodes + 8 * v;
    const int3 ijk   = voxijk[v];
    const bool cut   = vox_cc && vox_cc[v];
    if(cut)
    {
        V = vox_vg[v];
        if(!(V > 0.0)) return false;
        for(int q = 0; q < 8; ++q) g[q] = d3scale(vox_a[8 * v + q], 1.0 / V);
    }
    else
    {
        V = dx * dx * dx;
        for(int q = 0; q < 8; ++q) g[q] = d3scale(corner_sign(q), 1.0 / (4.0 * dx));
    }
    for(int q = 0; q < 8; ++q)
    {
        const int ia = (q >> 2) & 1, ib = (q >> 1) & 1, ic = q & 1;
        uu[q] = d3sub(vert[nodes[q]], make_double3(org.x + (ijk.x + ia) * dx, org.y + (ijk.y + ib) * dx, org.z + (ijk.z + ic) * dx));
    }
    return true;
}
__global__ void k_visc_assemble(int nvox, const int* vox_nodes, const int3* voxijk, const int* vox_cc, const double3* vox_a, const double* vox_vg, const double3* vert,
                                double dx, double3 org, double mu, double dt, int fem_off, int global_offset, Eigen::Matrix3d* vals)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if(v >= nvox) return;
    double3 g[8], uu[8];
    double  V;
    if(!visc_setup(v, vox_nodes, voxijk, vox_cc, vox_a, vox_vg, vert, dx, org, g, uu, V)) return;
    double S[3][3];
    for(int a = 0; a < 3; ++a) for(int b = 0; b < 3; ++b) S[a][b] = 0.0;
    for(int q = 0; q < 8; ++q)
    {
        const double ua[3] = {uu[q].x, uu[q].y, uu[q].z}, gb[3] = {g[q].x, g[q].y, g[q].z};
        for(int a = 0; a < 3; ++a) for(int b = 0; b < 3; ++b) S[a][b] += 0.5 * (ua[a] * gb[b] + ua[b] * gb[a]);
    }
    const int* nodes = vox_nodes + 8 * v;
    const double cv = 2.0 * mu * dt * V;
    for(int q = 0; q < 8; ++q)
    {
        const double gb[3] = {g[q].x, g[q].y, g[q].z};
        for(int a = 0; a < 3; ++a)
        {
            const double f = cv * (S[a][0] * gb[0] + S[a][1] * gb[1] + S[a][2] * gb[2]);
            binned_deposit(g_gbin + ((size_t)nodes[q] * 3 + a) * BINNED_K, f);
        }
    }
    const int toff = global_offset + 36 * v;
    int kk = 0;
    const double ch = mu * dt * V;
    for(int q = 0; q < 8; ++q)
        for(int r = q; r < 8; ++r)
        {
            const double gq[3] = {g[q].x, g[q].y, g[q].z}, gr[3] = {g[r].x, g[r].y, g[r].z};
            const double gg = gq[0] * gr[0] + gq[1] * gr[1] + gq[2] * gr[2];
            Eigen::Matrix3d B;
            for(int a = 0; a < 3; ++a)
                for(int c = 0; c < 3; ++c) B(a, c) = ch * ((a == c ? gg : 0.0) + gr[a] * gq[c]);
            const int row = nodes[q] + fem_off, col = nodes[r] + fem_off;
            if(row <= col) vals[toff + kk] += B;
            else vals[toff + kk] += B.transpose();
            ++kk;
        }
}
__global__ void k_visc_energy(int nvox, const int* vox_nodes, const int3* voxijk, const int* vox_cc, const double3* vox_a, const double* vox_vg, const double3* vert,
                              double dx, double3 org, double mu, double dt, double* partial)
{
    extern __shared__ double sh[];
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    double e = 0.0;
    if(v < nvox)
    {
        double3 g[8], uu[8];
        double  V;
        if(visc_setup(v, vox_nodes, voxijk, vox_cc, vox_a, vox_vg, vert, dx, org, g, uu, V))
        {
            double S[3][3];
            for(int a = 0; a < 3; ++a) for(int b = 0; b < 3; ++b) S[a][b] = 0.0;
            for(int q = 0; q < 8; ++q)
            {
                const double ua[3] = {uu[q].x, uu[q].y, uu[q].z}, gb[3] = {g[q].x, g[q].y, g[q].z};
                for(int a = 0; a < 3; ++a) for(int b = 0; b < 3; ++b) S[a][b] += 0.5 * (ua[a] * gb[b] + ua[b] * gb[a]);
            }
            double s2 = 0.0;
            for(int a = 0; a < 3; ++a) for(int b = 0; b < 3; ++b) s2 += S[a][b] * S[a][b];
            e = (mu / dt) * V * s2;
        }
    }
    sh[threadIdx.x] = e;
    __syncthreads();
    for(int k = blockDim.x / 2; k > 0; k >>= 1)
    {
        if(threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k];
        __syncthreads();
    }
    if(threadIdx.x == 0) partial[blockIdx.x] = sh[0];
}
// ---- LIQ_PDEV: particle volumetric-deviation energy (see Params::pdev) ----
// Per element: particles of its cell (sorted lists). Trilinear field of the 8 corners: d_p(u) = sum_q gradN_q(xi_p).u_q,
// J_p(u) = J_p (1 + d_p), J~(u) = sum_r w_r J_r (1 + d_r), w_r = V_r / sum V. eps_p(u) = eps_p^0 + sum_q c_pq.u_q with
// c_pq = J_p gradN_q(xi_p) - sum_r w_r J_r gradN_q(xi_r). Energy 1/2 K sum_p V_p eps_p^2; gradient K sum_p V_p eps_p c_pq;
// Hessian (q,r) K sum_p V_p c_pq c_pr^T. sum_p V_p eps_p(u) == 0: orthogonal to the cell-mean (one-point) energy.
__device__ __forceinline__ bool pdev_cell_setup(int c, const int* cstart, const int* cend, const int* pidx, const double3* xp, const double* Jp, const double* Vp0,
                                                double3 o, double dx, double3 g, double rho_over_kappa, double& sumV, double& Jt, double3& xt, double3* G)
{
    const int st = cstart[c];
    if(st < 0) return false;
    const int en = cend[c];
    if(en - st < 2) return false;
    sumV = 0.0; Jt = 0.0; xt = make_double3(0, 0, 0);
    for(int q = 0; q < 8; ++q) G[q] = make_double3(0, 0, 0);
    for(int e = st; e < en; ++e)
    {
        const int p = pidx[e];
        const double V = Vp0[p] * Jp[p];
        sumV += V; Jt += V * Jp[p];
        xt = d3add(xt, d3scale(xp[p], V));
        const double3 s = make_double3((xp[p].x - o.x) / dx, (xp[p].y - o.y) / dx, (xp[p].z - o.z) / dx);
        for(int q = 0; q < 8; ++q) G[q] = d3add(G[q], d3scale(gipc_liquid_wb::wb_gradN(q, s, dx), V * Jp[p]));
    }
    if(!(sumV > 0.0)) return false;
    Jt /= sumV; xt = d3scale(xt, 1.0 / sumV);
    for(int q = 0; q < 8; ++q) G[q] = d3scale(G[q], 1.0 / sumV);
    return true;
}
__global__ void k_pdev_assemble(int nvox, const int* vox_nodes, const int3* voxijk, const int* vox_cc, const double* vox_vg, int3 dims, const double3* vert, double dx, double3 org,
                                const int* cstart, const int* cend, const int* pidx, const double3* xp, const double* Jp, const double* Vp0, double3 g, double rho, double kappa,
                                double K, int all, double dt, int fem_off, int global_offset, Eigen::Matrix3d* vals)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if(v >= nvox) return;
    const bool cut = vox_cc && vox_cc[v];
    if(cut && !(vox_vg[v] > 0.0)) return;   // dead element
    if(!cut && !all) return;
    const int3 ijk = voxijk[v];
    const int  c   = cell_lin(ijk.x, ijk.y, ijk.z, dims);
    const double3 o = make_double3(org.x + ijk.x * dx, org.y + ijk.y * dx, org.z + ijk.z * dx);
    double sumV, Jt; double3 xt, G[8];
    if(!pdev_cell_setup(c, cstart, cend, pidx, xp, Jp, Vp0, o, dx, g, rho / kappa, sumV, Jt, xt, G)) return;
    const int* nodes = vox_nodes + 8 * v;
    double3 uu[8];
    for(int q = 0; q < 8; ++q)
    {
        const int ia = (q >> 2) & 1, ib = (q >> 1) & 1, ic = q & 1;
        uu[q] = d3sub(vert[nodes[q]], make_double3(o.x + ia * dx, o.y + ib * dx, o.z + ic * dx));
    }
    const double dt2 = dt * dt, rk = rho / kappa;
    double3 grad[8];
    for(int q = 0; q < 8; ++q) grad[q] = make_double3(0, 0, 0);
    double H[36][9];
    for(int kk = 0; kk < 36; ++kk) for(int a = 0; a < 9; ++a) H[kk][a] = 0.0;
    const int st = cstart[c], en = cend[c];
    for(int e = st; e < en; ++e)
    {
        const int p = pidx[e];
        const double V = Vp0[p] * Jp[p];
        const double3 s = make_double3((xp[p].x - o.x) / dx, (xp[p].y - o.y) / dx, (xp[p].z - o.z) / dx);
        double3 cpq[8];
        double  eps = (Jp[p] - Jt) + rk * pdev_dot(g, d3sub(xp[p], xt));
        for(int q = 0; q < 8; ++q)
        {
            cpq[q] = d3sub(d3scale(gipc_liquid_wb::wb_gradN(q, s, dx), Jp[p]), G[q]);
            eps += pdev_dot(cpq[q], uu[q]);
        }
        const double w = dt2 * K * V;
        for(int q = 0; q < 8; ++q) grad[q] = d3add(grad[q], d3scale(cpq[q], w * eps));
        int kk = 0;
        for(int q = 0; q < 8; ++q)
            for(int r = q; r < 8; ++r)
            {
                const double cq[3] = {cpq[q].x, cpq[q].y, cpq[q].z}, cr[3] = {cpq[r].x, cpq[r].y, cpq[r].z};
                for(int a = 0; a < 3; ++a)
                    for(int b = 0; b < 3; ++b) H[kk][3 * a + b] += w * cq[a] * cr[b];
                ++kk;
            }
    }
    for(int q = 0; q < 8; ++q)
    {
        binned_deposit(g_gbin + ((size_t)nodes[q] * 3 + 0) * BINNED_K, grad[q].x);
        binned_deposit(g_gbin + ((size_t)nodes[q] * 3 + 1) * BINNED_K, grad[q].y);
        binned_deposit(g_gbin + ((size_t)nodes[q] * 3 + 2) * BINNED_K, grad[q].z);
    }
    const int toff = global_offset + 36 * v;
    int kk = 0;
    for(int q = 0; q < 8; ++q)
        for(int r = q; r < 8; ++r)
        {
            const int row = nodes[q] + fem_off, col = nodes[r] + fem_off;
            Eigen::Matrix3d& B = vals[toff + kk];
            if(row <= col) { for(int a = 0; a < 3; ++a) for(int b = 0; b < 3; ++b) B(a, b) += H[kk][3 * a + b]; }
            else { for(int a = 0; a < 3; ++a) for(int b = 0; b < 3; ++b) B(b, a) += H[kk][3 * a + b]; }
            ++kk;
        }
}
__global__ void k_pdev_energy(int nvox, const int* vox_nodes, const int3* voxijk, const int* vox_cc, const double* vox_vg, int3 dims, const double3* vert, double dx, double3 org,
                              const int* cstart, const int* cend, const int* pidx, const double3* xp, const double* Jp, const double* Vp0, double3 g, double rho, double kappa,
                              double K, int all, double* partial)
{
    extern __shared__ double sh[];
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    double e = 0.0;
    if(v < nvox)
    {
        const bool cut = vox_cc && vox_cc[v];
        if(!(cut && !(vox_vg[v] > 0.0)) && (cut || all))
        {
            const int3 ijk = voxijk[v];
            const int  c   = cell_lin(ijk.x, ijk.y, ijk.z, dims);
            const double3 o = make_double3(org.x + ijk.x * dx, org.y + ijk.y * dx, org.z + ijk.z * dx);
            double sumV, Jt; double3 xt, G[8];
            if(pdev_cell_setup(c, cstart, cend, pidx, xp, Jp, Vp0, o, dx, g, rho / kappa, sumV, Jt, xt, G))
            {
                const int* nodes = vox_nodes + 8 * v;
                double3 uu[8];
                for(int q = 0; q < 8; ++q)
                {
                    const int ia = (q >> 2) & 1, ib = (q >> 1) & 1, ic = q & 1;
                    uu[q] = d3sub(vert[nodes[q]], make_double3(o.x + ia * dx, o.y + ib * dx, o.z + ic * dx));
                }
                const double rk = rho / kappa;
                for(int ee = cstart[c]; ee < cend[c]; ++ee)
                {
                    const int p = pidx[ee];
                    const double V = Vp0[p] * Jp[p];
                    const double3 s = make_double3((xp[p].x - o.x) / dx, (xp[p].y - o.y) / dx, (xp[p].z - o.z) / dx);
                    double eps = (Jp[p] - Jt) + rk * pdev_dot(g, d3sub(xp[p], xt));
                    for(int q = 0; q < 8; ++q) eps += pdev_dot(d3sub(d3scale(gipc_liquid_wb::wb_gradN(q, s, dx), Jp[p]), G[q]), uu[q]);
                    e += 0.5 * K * V * eps * eps;
                }
            }
        }
    }
    sh[threadIdx.x] = e;
    __syncthreads();
    for(int k = blockDim.x / 2; k > 0; k >>= 1)
    {
        if(threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k];
        __syncthreads();
    }
    if(threadIdx.x == 0) partial[blockIdx.x] = sh[0];
}

__global__ void k_vox_energy(int            nvox,
                             const int*     vox_nodes,
                             const double*  voxV,
                             const double*  voxJb,
                             const int3*    voxijk,
                             const double3* vert,
                             double         dx,
                             double3        org,
                             double         kappa,
                             double         tension_ratio,
                             double*        partial,
                             const int*     vox_cc = nullptr,
                             const double3* vox_a  = nullptr,
                             const double*  vox_vg = nullptr,
                             double         vbar   = 0.0,
                             double         vbar_kk = 0.0,
                             const double3* vox_f  = nullptr,
                             const double*  vox_hg = nullptr,
                             double         hgk    = 0.0,
                             int            invert_ok = 0,
                             int            wb_det = 0,
                             const double*  vox_s = nullptr,
                             double         sk = 0.0,
                             double3        upv = make_double3(0, 1, 0),
                             const double*  vox_Vr = nullptr,
                             int            wb_fullf = 0)
{
    extern __shared__ double sh[];
    int    v = blockIdx.x * blockDim.x + threadIdx.x;
    double e = 0.0;
    if(v < nvox && vox_cc && vox_cc[v])
    {
        double V = vox_Vr ? vox_Vr[v] : (vox_f ? vox_vg[v] : fmin(voxV[v], vox_vg[v]));   // same weight as the assembly
        double J;
        if(wb_det && vox_f)
        {
            if(vox_vg[v] > 0.0) { VoxState st; vox_state_g(vox_nodes + 8 * v, voxijk[v], vert, dx, org, voxJb[v], vox_a + 8 * v, vox_vg[v], st); J = st.J; }
            else J = 1.0;
        }
        else
            J = cc_J(vox_nodes + 8 * v, voxijk[v], vert, dx, org, voxJb[v], vox_a + 8 * v, vox_vg[v]);
        if(V > 0.0 && isfinite(J))
        {
            double d    = J - 1.0;
            double keff = (J < 1.0) ? kappa : kappa * tension_ratio;
            e           = 0.5 * keff * d * d * V;
            if(vbar > 0.0 && J < vbar)
            {
                double b0, b1, b2;
                vox_barrier(J, vbar, b0, b1, b2);
                e += V * vbar_kk * b0;
            }
            if(vox_f)
            {   // constant hydrostatic forces: E_lin = -sum_q f_q . u_q (u_q = node displacement from its lattice point)
                const int* nd = vox_nodes + 8 * v;
                for(int q = 0; q < 8; ++q)
                {
                    int     ia = (q >> 2) & 1, ib = (q >> 1) & 1, ic = q & 1;
                    double3 x0 = make_double3(org.x + (voxijk[v].x + ia) * dx, org.y + (voxijk[v].y + ib) * dx, org.z + (voxijk[v].z + ic) * dx);
                    double3 u  = d3sub(vert[nd[q]], x0);
                    e -= vox_f[8 * v + q].x * u.x + vox_f[8 * v + q].y * u.y + vox_f[8 * v + q].z * u.z;
                }
            }
            if(vox_s && sk > 0.0)
            {   // LIQ_WB_SURFK
                const int*    nd = vox_nodes + 8 * v;
                const double* Sm = vox_s + 36 * v;
                double un[8];
                for(int q = 0; q < 8; ++q)
                {
                    const int     ia = (q >> 2) & 1, ib = (q >> 1) & 1, ic = q & 1;
                    const double3 x0 = make_double3(org.x + (voxijk[v].x + ia) * dx, org.y + (voxijk[v].y + ib) * dx, org.z + (voxijk[v].z + ic) * dx);
                    const double3 u  = d3sub(vert[nd[q]], x0);
                    un[q] = upv.x * u.x + upv.y * u.y + upv.z * u.z;
                }
                double es = 0.0;
                for(int q = 0; q < 8; ++q)
                    for(int r = 0; r < 8; ++r) { const int qq = q < r ? q : r, rr = q < r ? r : q; es += Sm[qq * 8 - qq * (qq + 1) / 2 + rr] * un[q] * un[r]; }
                e += 0.5 * sk * es;
            }
            if(vox_hg && hgk > 0.0 && vox_vg[v] > 0.0)
            {   // LIQ_WB_HG (see k_vox_assemble)
                const int*     nd  = vox_nodes + 8 * v;
                const double3* a   = vox_a + 8 * v;
                const double   Vg  = vox_vg[v];
                const double*  rec = vox_hg + 150 * v;
                double3 uu[8];
                for(int q = 0; q < 8; ++q)
                {
                    const int     ia = (q >> 2) & 1, ib = (q >> 1) & 1, ic = q & 1;
                    const double3 x0 = make_double3(org.x + (voxijk[v].x + ia) * dx, org.y + (voxijk[v].y + ib) * dx, org.z + (voxijk[v].z + ic) * dx);
                    uu[q] = d3sub(vert[nd[q]], x0);
                }
                for(int T = 0; T < 6; ++T)
                {
                    const double VT = rec[25 * T];
                    if(!(VT > 1e-12 * Vg)) continue;
                    double sum = 0.0;
                    for(int q = 0; q < 8; ++q)
                        sum += (rec[25 * T + 1 + 3 * q] / VT - a[q].x / Vg) * uu[q].x + (rec[25 * T + 2 + 3 * q] / VT - a[q].y / Vg) * uu[q].y
                               + (rec[25 * T + 3 + 3 * q] / VT - a[q].z / Vg) * uu[q].z;
                    e += 0.5 * hgk * VT * sum * sum;
                }
            }
            if(J <= 0.0 && !invert_ok) e = 1e300;
        }
    }
    else if(v < nvox)
    {
        VoxState st;
        vox_state(vox_nodes + 8 * v, voxijk[v], vert, dx, org, voxJb[v], st);
        double V = vox_Vr ? vox_Vr[v] : voxV[v];
        if(V > 0.0 && isfinite(st.J))
        {
            {
                double d    = st.J - 1.0;
                double keff = (st.J < 1.0) ? kappa : kappa * tension_ratio;
                e           = 0.5 * keff * d * d * V;
            }
            if(vbar > 0.0 && st.J < vbar)
            {
                double b0, b1, b2;
                vox_barrier(st.J, vbar, b0, b1, b2);
                e += V * vbar_kk * b0;
            }
            if(st.detA <= 0.0 && !invert_ok) e = 1e300;  // inverted voxel: reject the step (LIQ_VOX_INVERT_OK=1: let the energy handle it)
        }
        if(wb_fullf && vox_f && e < 1e299)
        {   // LIQ_WB_FULLF: E_lin = -sum_q f_q . u_q for the full cell's exact hydrostatic term
            const int* nd = vox_nodes + 8 * v;
            for(int q = 0; q < 8; ++q)
            {
                int     ia = (q >> 2) & 1, ib = (q >> 1) & 1, ic = q & 1;
                double3 x0 = make_double3(org.x + (voxijk[v].x + ia) * dx, org.y + (voxijk[v].y + ib) * dx, org.z + (voxijk[v].z + ic) * dx);
                double3 uu = d3sub(vert[nd[q]], x0);
                e -= vox_f[8 * v + q].x * uu.x + vox_f[8 * v + q].y * uu.y + vox_f[8 * v + q].z * uu.z;
            }
        }
    }
    sh[threadIdx.x] = e;
    __syncthreads();
    for(int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if(threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if(threadIdx.x == 0) partial[blockIdx.x] = sh[0];
}
#include "liquid_ug.cuh"

__global__ void k_sum_partials(int n, const double* partial, double* out)
{
    double e = 0.0;
    for(int i = 0; i < n; ++i) e += partial[i];
    *out = e;
}

// ----------------------------------------------------------------------------
// post-step: nodes -> particles
// ----------------------------------------------------------------------------
__global__ void k_read_node_start_vel(int n_active, const int* active_nodes, const double3* vel, int base, double3* vi_old)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if(s >= n_active) return;
    vi_old[active_nodes[s]] = vel[base + s];
}

__global__ void k_read_node_vel(int n_active, const int* active_nodes, const double3* vert, int3 dims, double dx, double3 org, int base, double dt, double3* vi, int n_lattice = 0)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if(s >= n_active) return;
    int n = active_nodes[s];
    int i, j, k;
    node_ijk(n_lattice > 0 ? n % n_lattice : n, dims, i, j, k);
    double3 x0 = make_double3(org.x + i * dx, org.y + j * dx, org.z + k * dx);
    vi[n]      = d3scale(d3sub(vert[base + s], x0), 1.0 / dt);
}

__global__ void k_g2p(int            n,
                      double3*       xp,
                      double3*       vp,
                      double*        Cp,
                      double*        Jp,
                      const double3* vi,
                      const double3* vi_old,
                      double         flip,
                      const int*     nflag,
                      int3           dims,
                      double         dx,
                      double3        org,
                      double         dt,
                      int            apic,
                      double         apic_ratio,
                      double         rest_damp,
                      double         rest_v,
                      int*           sleep,
                      double         sleep_v,
                      double         wake_v,
                      int            sleep_n,
                      double*        vnmag,
                      double         Jmin,
                      double         Jmax,
                      double3        lo,
                      double3        hi,
                      const int*     pmask,
                      const double*  pwinv,
                      const int*     wnear,      // CPIC wall sample: nearest wall point X_w, normal n (into the fluid),
                      const double3* wX,         //   triangle + barycentric -> the wall's velocity over this step
                      const double3* wn,
                      const int*     wtri,
                      const double3* wb,
                      const uint3*   faces,
                      const double3* xsolid,     // solid positions after the solve
                      const double3* xsolid0,    // solid positions at the step start
                      int            cpic_c,     // 1 = ghost-velocity substitution for masked-out nodes, 0 = their actual velocity
                      const int*     nref_has = nullptr, const double3* nref_X = nullptr, const double3* nref_n = nullptr, int stride = 0, const int* plabel = nullptr, const int* nlab = nullptr,
                      const double3* vi_vel = nullptr,   // LIQ_BDF2: node velocity the particles carry (vi then only moves them and their J)
                      double3*       vp_prev = nullptr,  // LIQ_BDF2: receives this step's starting particle velocity / affine velocity
                      double*        Cp_prev = nullptr,
                      const int*     cell2elem = nullptr, // LIQ_WB_G2P: cut cells transfer with the SAME sub-volume geometry the energy uses:
                      const int*     vox_cc = nullptr,    //   cell velocity = the sub-volume's nodal-weight mean (rho int N_q dV), cell gradient =
                      const double*  vox_w = nullptr,     //   (1/V_g) sum_q v_q a_q^T. The plain 8-corner mean/gradient treats the fixed ghosts as
                      const double3* vox_a = nullptr,     //   fluid at rest: the particles then move and compress 1/f times less than the energy's
                      const double*  vox_vg = nullptr,    //   sub-volume does, the wall cells end up under-pressured and drive a circulation (10-25)
                      const int*     vox_wc = nullptr,    // LIQ_WB_G2P=2: only wall-cut elements with fill >= g2p_fmin
                      double         g2p_fmin = 0.0,
                      double         dx3 = 0.0,
                      const int*     fl_c2e = nullptr,    // LIQ_G2P_FLUID: cell -> element map; stencil cells that are not fluid elements (dead: behind a wall,
                      const int*     fl_cc  = nullptr,    //   above the column surface; inactive) are dropped and the weights renormalised -- their fixed /
                      const double*  fl_vg  = nullptr,    //   inactive corners otherwise pull the particle's velocity and J update toward rest
                      int            jcell  = 0,          // LIQ_G2P_JCELL: J update = the volume ratio of the particle's OWN element over the step
                      const double3* jc_a   = nullptr,    //   (full: det(I + dt sum v_q g_q^T); cut: 1 + dt sum a_q.v_q / V_g) instead of the interpolated C
                      int            fluid_only = 0)     // LIQ_G2P_FLUID proper (the map may be passed for JCELL alone)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    double3       x = xp[p];
    CentreStencil s = centre_stencil(x, org, dx, dims);
    double3       vn = make_double3(0, 0, 0), vo = make_double3(0, 0, 0), vn2 = make_double3(0, 0, 0);
    double        Cn[9], Cn2[9];
#pragma unroll
    for(int r = 0; r < 9; ++r) Cn[r] = Cn2[r] = 0.0;
    double inv4dx = 1.0 / (4.0 * dx);
    double3       fvn = make_double3(0, 0, 0), fvo = make_double3(0, 0, 0), fvn2 = make_double3(0, 0, 0);   // LIQ_G2P_FLUID sums
    double        fCn[9], fCn2[9], fw = 0.0;
#pragma unroll
    for(int r = 0; r < 9; ++r) fCn[r] = fCn2[r] = 0.0;
    const int    mask = pmask ? pmask[p] : LIQ_MASK_ALL;
    const double winv = pwinv ? pwinv[p] : 1.0;
    double       jr_sum = 0.0, jr_w = 0.0;   // LIQ_G2P_JCELL=2
    if(mask == LIQ_MASK_ALL)
    {   // unmasked: the original cell-centre route (velocity = centre interpolation of the 8-corner averages,
        // affine velocity = centre interpolation of the cell-centre gradients)
        for(int a = 0; a < 2; ++a)
            for(int b = 0; b < 2; ++b)
                for(int c = 0; c < 2; ++c)
                {
                    double w = s.w[0][a] * s.w[1][b] * s.w[2][c];
                    if(w <= 0.0) continue;
                    int     ci = s.b[0] + a, cj = s.b[1] + b, ck = s.b[2] + c;
                    double3 vc = make_double3(0, 0, 0), vco = make_double3(0, 0, 0), vc2 = make_double3(0, 0, 0);
                    double  G[9], G2[9];
#pragma unroll
                    for(int r = 0; r < 9; ++r) G[r] = G2[r] = 0.0;
                    // LIQ_WB_G2P: a cut cell's transfer weights and gradient from its fluid sub-volume
                    int    ev = -1;
                    double wsumq = 0.0, invVg = 0.0;
                    if(cell2elem && vox_cc && vox_w && vox_a && vox_vg)
                    {
                        const int cl = cell_lin(ci, cj, ck, dims);
                        const int e  = cell2elem[cl];
                        if(e >= 0 && vox_cc[e] && vox_vg[e] > 0.0 && (!vox_wc || (vox_wc[e] && vox_vg[e] >= g2p_fmin * dx3)))
                        {
                            for(int q = 0; q < 8; ++q) wsumq += vox_w[8 * e + q];
                            if(wsumq > 0.0) { ev = e; invVg = 1.0 / vox_vg[e]; }
                        }
                    }
                    for(int q = 0; q < 8; ++q)
                    {
                        int     qa = (q >> 2) & 1, qb = (q >> 1) & 1, qc = q & 1;
                        int     nid = vnode(node_lin(ci + qa, cj + qb, ck + qc, dims), x, nref_has, nref_X, nref_n, stride, plabel ? plabel[p] : -1, nlab);
                        double3 v   = nflag[nid] ? vi[nid] : make_double3(0, 0, 0);
                        const double wq = (ev >= 0) ? vox_w[8 * ev + q] / wsumq : 0.125;
                        if(flip > 0.0 && nflag[nid]) vco = d3add(vco, d3scale(vi_old[nid], wq));
                        double g[3];
                        if(ev >= 0) { const double3 aq = vox_a[8 * ev + q]; g[0] = aq.x * invVg; g[1] = aq.y * invVg; g[2] = aq.z * invVg; }
                        else { double3 sg = corner_sign(q); g[0] = sg.x * inv4dx; g[1] = sg.y * inv4dx; g[2] = sg.z * inv4dx; }
                        vc = d3add(vc, d3scale(v, wq));
                        double va[3] = {v.x, v.y, v.z};
                        for(int r = 0; r < 3; ++r)
                            for(int t = 0; t < 3; ++t) G[3 * r + t] += va[r] * g[t];
                        if(vi_vel)
                        {
                            double3 u  = nflag[nid] ? vi_vel[nid] : make_double3(0, 0, 0);
                            vc2        = d3add(vc2, d3scale(u, wq));
                            double ua[3] = {u.x, u.y, u.z};
                            for(int r = 0; r < 3; ++r)
                                for(int t = 0; t < 3; ++t) G2[3 * r + t] += ua[r] * g[t];
                        }
                    }
                    vn  = d3add(vn, d3scale(vc, w));
                    vo  = d3add(vo, d3scale(vco, w));
                    vn2 = d3add(vn2, d3scale(vc2, w));
#pragma unroll
                    for(int r = 0; r < 9; ++r) { Cn[r] += w * G[r]; Cn2[r] += w * G2[r]; }
                    if(jcell >= 2 && fl_c2e)
                    {   // LIQ_G2P_JCELL=2: this stencil cell's element volume ratio over the step (the J the energy gave it);
                        // =3: weighted by the element's fluid volume as well (a sliver / thin surface cell barely steers a particle's J)
                        const int e = fl_c2e[cell_lin(ci, cj, ck, dims)];
                        if(e >= 0 && (fl_cc[e] == 0 || fl_vg[e] > 0.0))
                        {
                            double rc;
                            if(fl_cc[e] == 1 && jc_a)
                            {
                                double dvol = 0.0;
                                for(int q = 0; q < 8; ++q)
                                {
                                    const int     nid = node_lin(ci + ((q >> 2) & 1), cj + ((q >> 1) & 1), ck + (q & 1), dims);
                                    const double3 v   = nflag[nid] ? vi[nid] : make_double3(0, 0, 0);
                                    dvol += jc_a[8 * e + q].x * v.x + jc_a[8 * e + q].y * v.y + jc_a[8 * e + q].z * v.z;
                                }
                                rc = 1.0 + dt * dvol / fl_vg[e];
                            }
                            else
                            {
                                double Mc[3][3];
                                for(int r = 0; r < 3; ++r) for(int t2 = 0; t2 < 3; ++t2) Mc[r][t2] = (r == t2) ? 1.0 : 0.0;
                                for(int q = 0; q < 8; ++q)
                                {
                                    const int     nid = node_lin(ci + ((q >> 2) & 1), cj + ((q >> 1) & 1), ck + (q & 1), dims);
                                    const double3 v   = nflag[nid] ? vi[nid] : make_double3(0, 0, 0);
                                    const double3 sg  = corner_sign(q);
                                    const double  g3[3] = {sg.x * inv4dx, sg.y * inv4dx, sg.z * inv4dx}, va[3] = {v.x, v.y, v.z};
                                    for(int r = 0; r < 3; ++r) for(int t2 = 0; t2 < 3; ++t2) Mc[r][t2] += dt * va[r] * g3[t2];
                                }
                                rc = Mc[0][0] * (Mc[1][1] * Mc[2][2] - Mc[1][2] * Mc[2][1]) - Mc[0][1] * (Mc[1][0] * Mc[2][2] - Mc[1][2] * Mc[2][0])
                                     + Mc[0][2] * (Mc[1][0] * Mc[2][1] - Mc[1][1] * Mc[2][0]);
                            }
                            const double wv = (jcell == 3) ? w * ((fl_cc[e] == 1) ? fl_vg[e] / (dx * dx * dx) : 1.0) : w;
                            jr_sum += wv * rc; jr_w += wv;
                        }
                    }
                    if(fl_c2e)
                    {
                        const int e = fl_c2e[cell_lin(ci, cj, ck, dims)];
                        if(e >= 0 && (fl_cc[e] == 0 || fl_vg[e] > 0.0))
                        {
                            fvn = d3add(fvn, d3scale(vc, w)); fvo = d3add(fvo, d3scale(vco, w)); fvn2 = d3add(fvn2, d3scale(vc2, w)); fw += w;
#pragma unroll
                            for(int r = 0; r < 9; ++r) { fCn[r] += w * G[r]; fCn2[r] += w * G2[r]; }
                        }
                    }
                }
        if(fl_c2e && fluid_only && fw > 0.0)
        {   // LIQ_G2P_FLUID: only the fluid elements of the stencil, renormalised
            const double iw = 1.0 / fw;
            vn = d3scale(fvn, iw); vo = d3scale(fvo, iw); vn2 = d3scale(fvn2, iw);
#pragma unroll
            for(int r = 0; r < 9; ++r) { Cn[r] = fCn[r] * iw; Cn2[r] = fCn2[r] * iw; }
        }
    }
    else
    {   // masked (near a wall), CPIC-style:
        //  (a) velocity: the compatible nodes with renormalised weights -- the SAME operator as the position embedding
        //      used by the contact chain rule and the CCD direction;
        //  (b) ghost velocity for the masked-out nodes: the wall's normal velocity + the fluid's own tangential velocity
        //      (no-penetration, slip);
        //  (c) affine velocity: the original cell-centre-gradient operator over all 27 nodes, with the ghost velocity at
        //      the masked-out nodes (no least squares: a single compatible node layer stays well-posed).
        NodeStencil ns   = node_stencil_of(s);
        double      wsum = 0.0;
        for(int a = 0; a < 3; ++a)
            for(int b = 0; b < 3; ++b)
                for(int c = 0; c < 3; ++c)
                {
                    double w = masked_w(ns, a, b, c, mask, winv);
                    if(w <= 0.0) continue;
                    int nid = vnode(node_lin(ns.b[0] + a, ns.b[1] + b, ns.b[2] + c, dims), x, nref_has, nref_X, nref_n, stride, plabel ? plabel[p] : -1, nlab);
                    if(!nflag[nid]) continue;
                    vn = d3add(vn, d3scale(vi[nid], w));
                    if(flip > 0.0) vo = d3add(vo, d3scale(vi_old[nid], w));
                    wsum += w;
                }
        if(wsum > 0.0)
        {
            vn = d3scale(vn, 1.0 / wsum);
            vo = d3scale(vo, 1.0 / wsum);
        }
        double3 vs = vn;
        if(wnear && wnear[p] && wtri && faces)
        {
            uint3   F  = faces[wtri[p]];
            double3 bb = wb[p];
            double3 vw = make_double3(0, 0, 0);
            if(xsolid && xsolid0 && dt > 0.0)
            {
                double3 dv = d3add(d3add(d3scale(d3sub(xsolid[F.x], xsolid0[F.x]), bb.x), d3scale(d3sub(xsolid[F.y], xsolid0[F.y]), bb.y)),
                                   d3scale(d3sub(xsolid[F.z], xsolid0[F.z]), bb.z));
                vw = d3scale(dv, 1.0 / dt);
            }
            double3 nn = wn[p];
            double  dn = nn.x * (vn.x - vw.x) + nn.y * (vn.y - vw.y) + nn.z * (vn.z - vw.z);
            vs = d3sub(vn, d3scale(nn, dn));
        }
        else
            LIQ_FBK(8);   // masked particle without wall information (should not happen)
        for(int a = 0; a < 2; ++a)
            for(int b = 0; b < 2; ++b)
                for(int c = 0; c < 2; ++c)
                {
                    double w = s.w[0][a] * s.w[1][b] * s.w[2][c];
                    if(w <= 0.0) continue;
                    int    ci = s.b[0] + a, cj = s.b[1] + b, ck = s.b[2] + c;
                    double G[9];
#pragma unroll
                    for(int r = 0; r < 9; ++r) G[r] = 0.0;
                    for(int q = 0; q < 8; ++q)
                    {
                        int     qa = (q >> 2) & 1, qb = (q >> 1) & 1, qc = q & 1;
                        int     kk = ((a + qa) * 3 + (b + qb)) * 3 + (c + qc);
                        int     nid = vnode(node_lin(ci + qa, cj + qb, ck + qc, dims), x, nref_has, nref_X, nref_n, stride, plabel ? plabel[p] : -1, nlab);
                        bool    compat = ((mask >> kk) & 1) != 0;
                        double3 v = (compat || !cpic_c) ? (nflag[nid] ? vi[nid] : make_double3(0, 0, 0)) : vs;
                        double3 sg  = corner_sign(q);
                        double  g[3] = {sg.x * inv4dx, sg.y * inv4dx, sg.z * inv4dx};
                        double  va[3] = {v.x, v.y, v.z};
                        for(int r = 0; r < 3; ++r)
                            for(int t = 0; t < 3; ++t) G[3 * r + t] += va[r] * g[t];
                    }
#pragma unroll
                    for(int r = 0; r < 9; ++r) Cn[r] += w * G[r];
                }
    }
    // J update: exact determinant of (I + dt C)
    double M[3][3];
    for(int r = 0; r < 3; ++r)
        for(int t = 0; t < 3; ++t) M[r][t] = (r == t ? 1.0 : 0.0) + dt * Cn[3 * r + t];
    double det = M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
                 + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
    if(jcell >= 2 && jr_w > 0.0) det = jr_sum / jr_w;   // the transpose of the deposit: the stencil's element ratios with the deposit weights (live elements)
    else if(jcell == 1 && fl_c2e)
    {   // LIQ_G2P_JCELL: the element containing the particle (same cell the energy compresses); dead / missing -> keep the interpolated update
        const int ci = (int)floor((x.x - org.x) / dx), cj = (int)floor((x.y - org.y) / dx), ck = (int)floor((x.z - org.z) / dx);
        if(ci >= 0 && cj >= 0 && ck >= 0 && ci < dims.x && cj < dims.y && ck < dims.z)
        {
            const int e = fl_c2e[cell_lin(ci, cj, ck, dims)];
            if(e >= 0 && (fl_cc[e] == 0 || fl_vg[e] > 0.0))
            {
                double Mc[3][3];
                for(int r = 0; r < 3; ++r) for(int t2 = 0; t2 < 3; ++t2) Mc[r][t2] = (r == t2) ? 1.0 : 0.0;
                double dvol = 0.0;
                for(int q = 0; q < 8; ++q)
                {
                    const int qa = (q >> 2) & 1, qb = (q >> 1) & 1, qc = q & 1;
                    const int nid = node_lin(ci + qa, cj + qb, ck + qc, dims);
                    const double3 v = nflag[nid] ? (vi_vel ? vi[nid] : vi[nid]) : make_double3(0, 0, 0);
                    if(fl_cc[e] == 1 && jc_a) dvol += jc_a[8 * e + q].x * v.x + jc_a[8 * e + q].y * v.y + jc_a[8 * e + q].z * v.z;
                    else
                    {
                        const double3 sg = corner_sign(q);
                        const double g3[3] = {sg.x * inv4dx, sg.y * inv4dx, sg.z * inv4dx}, va[3] = {v.x, v.y, v.z};
                        for(int r = 0; r < 3; ++r) for(int t2 = 0; t2 < 3; ++t2) Mc[r][t2] += dt * va[r] * g3[t2];
                    }
                }
                if(fl_cc[e] == 1 && jc_a) det = 1.0 + dt * dvol / fl_vg[e];
                else det = Mc[0][0] * (Mc[1][1] * Mc[2][2] - Mc[1][2] * Mc[2][1]) - Mc[0][1] * (Mc[1][0] * Mc[2][2] - Mc[1][2] * Mc[2][0])
                           + Mc[0][2] * (Mc[1][0] * Mc[2][1] - Mc[1][1] * Mc[2][0]);
            }
        }
    }
    double J = Jp[p] * det;
    if(!isfinite(J)) { LIQ_FBK(3); J = Jmin; }
    else if(J < Jmin) { LIQ_FBK(4); J = Jmin; }
    if(J > Jmax) { LIQ_FBK(4); J = Jmax; }
    double3 xn = d3add(x, d3scale(vn, dt));
    // FLIP blend: carry the particle's own velocity plus the grid's velocity CHANGE over the step (no resampling loss);
    // positions still move with the smooth grid velocity
    double3 vs = vi_vel ? vn2 : vn;
    if(flip > 0.0)
    {
        double3 vf = d3add(vp[p], d3sub(vn, vo));
        vs         = d3add(d3scale(vf, flip), d3scale(vn, 1.0 - flip));
    }
    // rest damping: only slow particles (|v| < rest_v) lose velocity, weight 1 - (|v|/rest_v)^2, so pours and sloshing
    // (cm/s and up) are untouched while the slow residual circulation of resting water is braked
    if(rest_damp > 0.0)
    {
        double sp = sqrt(vs.x * vs.x + vs.y * vs.y + vs.z * vs.z);
        if(sp < rest_v)
        {
            double w = 1.0 - (sp / rest_v) * (sp / rest_v);
            vs       = d3scale(vs, exp(-rest_damp * dt * w));
        }
    }
    if(!d3finite(xn) || !d3finite(vn) || !d3finite(vs))
    {
        xn = x;
        vn = make_double3(0, 0, 0);
        vs = vn;
#pragma unroll
        for(int r = 0; r < 9; ++r) Cn[r] = Cn2[r] = 0.0;
    }
    // rest sleep (position-level): the residual creep of resting water is re-created by every step's grid solve, and
    // positions advance with that solved velocity (vn), so damping the carried velocity (vs) cannot stop it. A particle
    // that has stayed slower than sleep_v for sleep_n steps is held (x and J frozen, v = C = 0) until the solved velocity
    // at it exceeds wake_v (container moved, liquid poured onto it, ...).
    if(vnmag) vnmag[p] = sqrt(vn.x * vn.x + vn.y * vn.y + vn.z * vn.z);
    if(sleep_v > 0.0)
    {
        double sp = sqrt(vn.x * vn.x + vn.y * vn.y + vn.z * vn.z);
        int    c  = sleep[p];
        if(c >= sleep_n && sp <= wake_v)
        {
            xn = x;
            J  = Jp[p];
            vs = make_double3(0, 0, 0);
#pragma unroll
            for(int r = 0; r < 9; ++r) Cn[r] = 0.0;
        }
        else
            c = (c >= sleep_n) ? 0 : (sp < sleep_v ? c + 1 : 0);
        sleep[p] = c;
    }
    double3 xn_pre = xn;
    if(xn.x < lo.x) { xn.x = lo.x; vs.x = fmax(vs.x, 0.0); }
    if(xn.x > hi.x) { xn.x = hi.x; vs.x = fmin(vs.x, 0.0); }
    if(xn.y < lo.y) { xn.y = lo.y; vs.y = fmax(vs.y, 0.0); }
    if(xn.y > hi.y) { xn.y = hi.y; vs.y = fmin(vs.y, 0.0); }
    if(xn.z < lo.z) { xn.z = lo.z; vs.z = fmax(vs.z, 0.0); }
    if(xn.z > hi.z) { xn.z = hi.z; vs.z = fmin(vs.z, 0.0); }
    if(xn.x != xn_pre.x || xn.y != xn_pre.y || xn.z != xn_pre.z) LIQ_FBK(7);
    if(vp_prev)
    {
        vp_prev[p] = vp[p];
#pragma unroll
        for(int r = 0; r < 9; ++r) Cp_prev[9 * p + r] = Cp[9 * p + r];
    }
    xp[p] = xn;
    vp[p] = vs;
#pragma unroll
    for(int r = 0; r < 9; ++r) Cp[9 * p + r] = apic ? apic_ratio * (vi_vel ? Cn2[r] : Cn[r]) : 0.0;
    Jp[p] = J;
}

__global__ void k_particle_stencil(int n, const double3* xp, double3 org, double dx, int3 dims, int3* pbase, double3* pfrac)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    CentreStencil s = centre_stencil(xp[p], org, dx, dims);
    pbase[p] = make_int3(s.b[0], s.b[1], s.b[2]);
    pfrac[p] = make_double3(s.w[0][1], s.w[1][1], s.w[2][1]);
}
// x_p = temp_p + sum W (x_i - temp_i)
__global__ void k_particle_positions(int n, const int3* pbase, const double3* pfrac, const int* pmask, const double* pwinv, const int* node_slot, int3 dims, int base, int pbase_v,
                                     double3* vert, const double3* temp, double3* xt, const double3* xp = nullptr, const int* nref_has = nullptr,
                                     const double3* nref_X = nullptr, const double3* nref_n = nullptr, int stride = 0, const int* plabel = nullptr, const int* nlab = nullptr)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    double3 xq = xp ? xp[p] : make_double3(0, 0, 0);
    NodeStencil st = node_stencil(pbase[p], pfrac[p]);
    const int mask = pmask[p]; const double winv = pwinv[p];
    double3     d  = make_double3(0, 0, 0);
    for(int a = 0; a < 3; ++a)
        for(int b = 0; b < 3; ++b)
            for(int c = 0; c < 3; ++c)
            {
                double w = masked_w(st, a, b, c, mask, winv);
                if(w <= 0.0) continue;
                int sl = node_slot[vnode(node_lin(st.b[0] + a, st.b[1] + b, st.b[2] + c, dims), xq, nref_has, nref_X, nref_n, stride, plabel ? plabel[p] : -1, nlab)];
                if(sl < 0) continue;
                int v = base + sl;
                d = d3add(d, d3scale(d3sub(vert[v], temp[v]), w));
            }
    double3 x = d3add(temp[pbase_v + p], d);
    vert[pbase_v + p] = x;
    xt[pbase_v + p]   = x;  // zero kinetic energy/gradient on the (btype 4) particle rows
}
__global__ void k_particle_movedir(int n, const int3* pbase, const double3* pfrac, const int* pmask, const double* pwinv, const int* node_slot, int3 dims, int base, int pbase_v, double3* md,
                                   const double3* xp = nullptr, const int* nref_has = nullptr, const double3* nref_X = nullptr, const double3* nref_n = nullptr, int stride = 0, const int* plabel = nullptr, const int* nlab = nullptr)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    double3 xq = xp ? xp[p] : make_double3(0, 0, 0);
    NodeStencil st = node_stencil(pbase[p], pfrac[p]);
    const int mask = pmask[p]; const double winv = pwinv[p];
    double3     d  = make_double3(0, 0, 0);
    for(int a = 0; a < 3; ++a)
        for(int b = 0; b < 3; ++b)
            for(int c = 0; c < 3; ++c)
            {
                double w = masked_w(st, a, b, c, mask, winv);
                if(w <= 0.0) continue;
                int sl = node_slot[vnode(node_lin(st.b[0] + a, st.b[1] + b, st.b[2] + c, dims), xq, nref_has, nref_X, nref_n, stride, plabel ? plabel[p] : -1, nlab)];
                if(sl < 0) continue;
                d = d3add(d, d3scale(md[base + sl], w));
            }
    md[pbase_v + p] = d;
}
__global__ void k_contact_grad_chain(int n, const int3* pbase, const double3* pfrac, const int* pmask, const double* pwinv, const int* node_slot, int3 dims, int base, int pbase_v, double3* fb,
                                     const double3* xp = nullptr, const int* nref_has = nullptr, const double3* nref_X = nullptr, const double3* nref_n = nullptr, int stride = 0, const int* plabel = nullptr, const int* nlab = nullptr)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    double3 xq = xp ? xp[p] : make_double3(0, 0, 0);
    double3 g = fb[pbase_v + p];
    if(g.x == 0.0 && g.y == 0.0 && g.z == 0.0) return;
    NodeStencil st = node_stencil(pbase[p], pfrac[p]);
    const int mask = pmask[p]; const double winv = pwinv[p];
    for(int a = 0; a < 3; ++a)
        for(int b = 0; b < 3; ++b)
            for(int c = 0; c < 3; ++c)
            {
                double w = masked_w(st, a, b, c, mask, winv);
                if(w <= 0.0) continue;
                int sl = node_slot[vnode(node_lin(st.b[0] + a, st.b[1] + b, st.b[2] + c, dims), xq, nref_has, nref_X, nref_n, stride, plabel ? plabel[p] : -1, nlab)];
                if(sl < 0) continue;
                int v = base + sl;
                atomicAdd(&fb[v].x, w * g.x);
                atomicAdd(&fb[v].y, w * g.y);
                atomicAdd(&fb[v].z, w * g.z);
            }
}
// ---- LIQ_LOCAL_STEP: per-node scale of the move direction from the step bound of the particles that reference the node ----
__device__ inline void liq_atomic_min_double(double* addr, double val)
{
    unsigned long long* bits = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long  old  = *bits, assumed;
    do
    {
        assumed    = old;
        double cur = __longlong_as_double(static_cast<long long>(assumed));
        if(cur <= val) break;
        old = atomicCAS(bits, assumed, static_cast<unsigned long long>(__double_as_longlong(val)));
    } while(assumed != old);
}
__global__ void k_fill_double(int n, double* a, double v)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) a[i] = v;
}
// alpha_p: the largest feasible fraction of particle p's direction against its active contact pairs (filled by the engine's CCD kernel,
// 1 = unlimited); the ground plane (normal +y at ground_y, IPC convention x_new = x - alpha d) is added here. Every node of a limited
// particle's stencil takes the minimum.
__global__ void k_node_scale_min(int n, const double* alpha_p, const int3* pbase, const double3* pfrac, const int* pmask, const double* pwinv, const int* node_slot,
                                 int3 dims, int base, int pbase_v, const double3* md, const double3* vert, double ground_y, double ground_slack, double cut,
                                 double* scale, int* count,
                                 const double3* xp = nullptr, const int* nref_has = nullptr, const double3* nref_X = nullptr, const double3* nref_n = nullptr, int stride = 0,
                                 const int* plabel = nullptr, const int* nlab = nullptr)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    double a = alpha_p[p];
    if(ground_y > -1e299)
    {
        double dist = vert[pbase_v + p].y - ground_y, coef = md[pbase_v + p].y;   // approaching the plane when d.y > 0
        if(dist > 0.0 && coef > 0.0) a = fmin(a, ground_slack * dist / coef);
    }
    if(!(a < cut)) return;
    if(a < 0.0) a = 0.0;
    atomicAdd(count, 1);
    double3 xq = xp ? xp[p] : make_double3(0, 0, 0);
    NodeStencil st = node_stencil(pbase[p], pfrac[p]);
    const int mask = pmask[p]; const double winv = pwinv[p];
    for(int aa = 0; aa < 3; ++aa)
        for(int b = 0; b < 3; ++b)
            for(int c = 0; c < 3; ++c)
            {
                double w = masked_w(st, aa, b, c, mask, winv);
                if(w <= 0.0) continue;
                int sl = node_slot[vnode(node_lin(st.b[0] + aa, st.b[1] + b, st.b[2] + c, dims), xq, nref_has, nref_X, nref_n, stride, plabel ? plabel[p] : -1, nlab)];
                if(sl < 0) continue;
                liq_atomic_min_double(&scale[sl], a);
            }
}
__global__ void k_node_scale_apply(int n, const double* scale, int base, double3* md)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if(s >= n) return;
    double f = scale[s];
    if(f < 1.0) { double3 v = md[base + s]; md[base + s] = make_double3(v.x * f, v.y * f, v.z * f); }
}
// pass 1: count cross blocks (one side particle), accumulate diagonal particle blocks, flag particles
__global__ void k_contact_scan(int nblk, const int* rows, const int* cols, const Eigen::Matrix3d* vals, int pbase_v, int n_particles, double* Hpp, int* pflag, int* counters)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if(b >= nblk) return;
    int r = rows[b], c = cols[b];
    bool pr = r >= pbase_v && r < pbase_v + n_particles;
    bool pc = c >= pbase_v && c < pbase_v + n_particles;
    if(!pr && !pc) return;
    if(pr && pc)
    {
        if(r != c) return;  // particle-particle blocks do not arise (particles have no faces)
        int p = r - pbase_v;
        const Eigen::Matrix3d& H = vals[b];
        for(int i = 0; i < 3; ++i)
            for(int j = 0; j < 3; ++j) atomicAdd(&Hpp[9 * p + 3 * i + j], H(i, j));
        pflag[p] = 1;
        return;
    }
    atomicAdd(&counters[0], 1);
}
// pass 2a: emit cross blocks (t, p) -> (t, node_i) * W_i
__global__ void k_emit_cross(int nblk, const int* rows, const int* cols, const Eigen::Matrix3d* vals, int pbase_v, int n_particles,
                             const int3* pbase, const double3* pfrac, const int* pmask, const double* pwinv, const int* node_slot, int3 dims, int base, int* cursor,
                             int* out_rows, int* out_cols, Eigen::Matrix3d* out_vals, const double3* xp = nullptr, const int* nref_has = nullptr,
                             const double3* nref_X = nullptr, const double3* nref_n = nullptr, int stride = 0, const int* plabel = nullptr, const int* nlab = nullptr)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if(b >= nblk) return;
    int r = rows[b], c = cols[b];
    bool pr = r >= pbase_v && r < pbase_v + n_particles;
    bool pc = c >= pbase_v && c < pbase_v + n_particles;
    if(pr == pc) return;
    int             p = (pc ? c : r) - pbase_v;
    double3         xq = xp ? xp[p] : make_double3(0, 0, 0);
    int             t = pc ? r : c;
    Eigen::Matrix3d H;
    if(pc) H = vals[b]; else H = vals[b].transpose();  // H_{t p}
    int             slot = atomicAdd(cursor, 27);
    NodeStencil     st   = node_stencil(pbase[p], pfrac[p]);
    const int mask = pmask[p]; const double winv = pwinv[p];
    int             k    = 0;
    for(int a = 0; a < 3; ++a)
        for(int bb = 0; bb < 3; ++bb)
            for(int cc = 0; cc < 3; ++cc, ++k)
            {
                double w  = masked_w(st, a, bb, cc, mask, winv);
                int    sl = node_slot[vnode(node_lin(st.b[0] + a, st.b[1] + bb, st.b[2] + cc, dims), xq, nref_has, nref_X, nref_n, stride, plabel ? plabel[p] : -1, nlab)];
                int    o  = slot + k;
                if(w <= 0.0 || sl < 0)
                {   // zero block with a valid index keeps the stream well-formed
                    out_rows[o] = t; out_cols[o] = t; out_vals[o].setZero();
                    continue;
                }
                int v = base + sl;
                if(t <= v) { out_rows[o] = t; out_cols[o] = v; out_vals[o] = w * H; }
                else       { out_rows[o] = v; out_cols[o] = t; out_vals[o] = w * H.transpose(); }
            }
}
// pass 2b: emit (p,p) -> W_i W_j Hpp over the 27-node stencil (378 upper blocks)
__global__ void k_emit_diag(int n, const int* pflag, const double* Hpp, const int3* pbase, const double3* pfrac, const int* pmask, const double* pwinv, const int* node_slot, int3 dims, int base,
                            int* cursor, int* out_rows, int* out_cols, Eigen::Matrix3d* out_vals, const double3* xp = nullptr, const int* nref_has = nullptr,
                            const double3* nref_X = nullptr, const double3* nref_n = nullptr, int stride = 0, const int* plabel = nullptr, const int* nlab = nullptr)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n || !pflag[p]) return;
    double3 xq = xp ? xp[p] : make_double3(0, 0, 0);
    Eigen::Matrix3d H;
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j) H(i, j) = Hpp[9 * p + 3 * i + j];
    NodeStencil st = node_stencil(pbase[p], pfrac[p]);
    const int mask = pmask[p]; const double winv = pwinv[p];
    int         vid[27];
    double      wgt[27];
    int         k = 0;
    for(int a = 0; a < 3; ++a)
        for(int bb = 0; bb < 3; ++bb)
            for(int cc = 0; cc < 3; ++cc, ++k)
            {
                double w  = masked_w(st, a, bb, cc, mask, winv);
                int    sl = node_slot[vnode(node_lin(st.b[0] + a, st.b[1] + bb, st.b[2] + cc, dims), xq, nref_has, nref_X, nref_n, stride, plabel ? plabel[p] : -1, nlab)];
                wgt[k]    = (sl < 0) ? 0.0 : w;
                vid[k]    = (sl < 0) ? -1 : base + sl;
            }
    int slot = atomicAdd(cursor, 378);
    int o    = slot;
    int fallback = -1;
    for(int i = 0; i < 27; ++i) if(vid[i] >= 0) { fallback = vid[i]; break; }
    for(int i = 0; i < 27; ++i)
        for(int j = i; j < 27; ++j, ++o)
        {
            double w = wgt[i] * wgt[j];
            if(w <= 0.0 || vid[i] < 0 || vid[j] < 0)
            {
                out_rows[o] = fallback; out_cols[o] = fallback; out_vals[o].setZero();
                continue;
            }
            int ri = vid[i], cj = vid[j];
            if(ri <= cj) { out_rows[o] = ri; out_cols[o] = cj; out_vals[o] = w * H; }
            else         { out_rows[o] = cj; out_cols[o] = ri; out_vals[o] = w * H.transpose(); }
        }
}

// 8-node Hessian-only stencil (LIQ_H8): the particle's enclosing cell corners with
// trilinear weights. Inexact (the gradient keeps the exact 27-node embedding) but a
// congruence => PSD; 36 blocks per particle instead of 378, 8 per cross block.
struct Stencil8
{
    int    vid[8];
    double w[8];
};
__device__ __forceinline__ Stencil8 stencil8(double3 x, double3 org, double dx, int3 dims, const int* node_slot, int base)
{
    Stencil8 s;
    double r[3] = {(x.x - org.x) / dx, (x.y - org.y) / dx, (x.z - org.z) / dx};
    int    n[3] = {dims.x, dims.y, dims.z};
    int    b[3];
    double f[3];
    for(int a = 0; a < 3; ++a)
    {
        b[a] = (int)floor(r[a]);
        if(b[a] < 0) b[a] = 0;
        if(b[a] > n[a] - 1) b[a] = n[a] - 1;
        f[a] = fmin(fmax(r[a] - b[a], 0.0), 1.0);
    }
    for(int q = 0; q < 8; ++q)
    {
        int    a = (q >> 2) & 1, bb = (q >> 1) & 1, c = q & 1;
        double w = (a ? f[0] : 1.0 - f[0]) * (bb ? f[1] : 1.0 - f[1]) * (c ? f[2] : 1.0 - f[2]);
        int    sl = node_slot[node_lin(b[0] + a, b[1] + bb, b[2] + c, dims)];
        s.w[q]   = (sl < 0) ? 0.0 : w;
        s.vid[q] = (sl < 0) ? -1 : base + sl;
    }
    return s;
}
__global__ void k_emit_diag8(int n, const int* pflag, const double* Hpp, const double3* xp, double3 org, double dx, int3 dims, const int* node_slot, int base,
                             int* cursor, int* out_rows, int* out_cols, Eigen::Matrix3d* out_vals)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n || !pflag[p]) return;
    Eigen::Matrix3d H;
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j) H(i, j) = Hpp[9 * p + 3 * i + j];
    Stencil8 st = stencil8(xp[p], org, dx, dims, node_slot, base);
    int fallback = -1;
    for(int i = 0; i < 8; ++i) if(st.vid[i] >= 0) { fallback = st.vid[i]; break; }
    int slot = atomicAdd(cursor, 36);
    int o    = slot;
    for(int i = 0; i < 8; ++i)
        for(int j = i; j < 8; ++j, ++o)
        {
            double w = st.w[i] * st.w[j];
            if(w <= 0.0 || st.vid[i] < 0 || st.vid[j] < 0) { out_rows[o] = fallback; out_cols[o] = fallback; out_vals[o].setZero(); continue; }
            int ri = st.vid[i], cj = st.vid[j];
            if(ri <= cj) { out_rows[o] = ri; out_cols[o] = cj; out_vals[o] = w * H; }
            else         { out_rows[o] = cj; out_cols[o] = ri; out_vals[o] = w * H.transpose(); }
        }
}
__global__ void k_emit_cross8(int nblk, const int* rows, const int* cols, const Eigen::Matrix3d* vals, int pbase_v, int n_particles, const double3* xp,
                              double3 org, double dx, int3 dims, const int* node_slot, int base, int* cursor, int* out_rows, int* out_cols, Eigen::Matrix3d* out_vals)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if(b >= nblk) return;
    int r = rows[b], c = cols[b];
    bool pr = r >= pbase_v && r < pbase_v + n_particles;
    bool pc = c >= pbase_v && c < pbase_v + n_particles;
    if(pr == pc) return;
    int             p = (pc ? c : r) - pbase_v;
    int             t = pc ? r : c;
    Eigen::Matrix3d H;
    if(pc) H = vals[b]; else H = vals[b].transpose();
    Stencil8 st   = stencil8(xp[p], org, dx, dims, node_slot, base);
    int      slot = atomicAdd(cursor, 8);
    for(int k = 0; k < 8; ++k)
    {
        int o = slot + k;
        if(st.w[k] <= 0.0 || st.vid[k] < 0) { out_rows[o] = t; out_cols[o] = t; out_vals[o].setZero(); continue; }
        int v = st.vid[k];
        if(t <= v) { out_rows[o] = t; out_cols[o] = v; out_vals[o] = st.w[k] * H; }
        else       { out_rows[o] = v; out_cols[o] = t; out_vals[o] = st.w[k] * H.transpose(); }
    }
}

// row-sum lumped variant of k_emit_diag: 27 diagonal blocks W_i * Hpp (PSD, cheaper)
__global__ void k_emit_diag_lumped(int n, const int* pflag, const double* Hpp, const int3* pbase, const double3* pfrac, const int* pmask, const double* pwinv, const int* node_slot, int3 dims, int base,
                                   int* cursor, int* out_rows, int* out_cols, Eigen::Matrix3d* out_vals, const double3* xp = nullptr, const int* nref_has = nullptr,
                                   const double3* nref_X = nullptr, const double3* nref_n = nullptr, int stride = 0, const int* plabel = nullptr, const int* nlab = nullptr)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n || !pflag[p]) return;
    double3 xq = xp ? xp[p] : make_double3(0, 0, 0);
    Eigen::Matrix3d H;
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j) H(i, j) = Hpp[9 * p + 3 * i + j];
    NodeStencil st   = node_stencil(pbase[p], pfrac[p]);
    const int mask = pmask[p]; const double winv = pwinv[p];
    int         slot = atomicAdd(cursor, 27);
    int         k    = 0;
    int         fallback = -1;
    for(int a = 0; a < 3 && fallback < 0; ++a)
        for(int bb = 0; bb < 3 && fallback < 0; ++bb)
            for(int cc = 0; cc < 3 && fallback < 0; ++cc)
            {
                int sl = node_slot[vnode(node_lin(st.b[0] + a, st.b[1] + bb, st.b[2] + cc, dims), xq, nref_has, nref_X, nref_n, stride, plabel ? plabel[p] : -1, nlab)];
                if(sl >= 0) fallback = base + sl;
            }
    for(int a = 0; a < 3; ++a)
        for(int bb = 0; bb < 3; ++bb)
            for(int cc = 0; cc < 3; ++cc, ++k)
            {
                double w  = masked_w(st, a, bb, cc, mask, winv);
                int    sl = node_slot[vnode(node_lin(st.b[0] + a, st.b[1] + bb, st.b[2] + cc, dims), xq, nref_has, nref_X, nref_n, stride, plabel ? plabel[p] : -1, nlab)];
                int    o  = slot + k;
                if(w <= 0.0 || sl < 0) { out_rows[o] = fallback; out_cols[o] = fallback; out_vals[o].setZero(); continue; }
                out_rows[o] = base + sl; out_cols[o] = base + sl; out_vals[o] = w * H;
            }
}

// ----------------------------------------------------------------------------
// particle neighbour grid (lattice cells), push-apart and SPH density
// ----------------------------------------------------------------------------
__device__ __forceinline__ int particle_cell(double3 x, double3 org, double dx, int3 d)
{
    int i = (int)floor((x.x - org.x) / dx), j = (int)floor((x.y - org.y) / dx), k = (int)floor((x.z - org.z) / dx);
    i = max(0, min(d.x - 1, i)); j = max(0, min(d.y - 1, j)); k = max(0, min(d.z - 1, k));
    return cell_lin(i, j, k, d);
}
__global__ void k_pkeys(int n, const double3* xp, double3 org, double dx, int3 d, int* key, int* idx)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    key[p] = particle_cell(xp[p], org, dx, d);
    idx[p] = p;
}
__global__ void k_cell_ranges(int n, const int* key, int* cstart, int* cend)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    int k = key[i];
    if(i == 0 || key[i - 1] != k) cstart[k] = i;
    if(i == n - 1 || key[i + 1] != k) cend[k] = i + 1;
}
// cubic spline kernel (3D), support 2h
__device__ __forceinline__ double W_cubic(double r, double h)
{
    double q = r / h;
    double s = 1.0 / (3.141592653589793 * h * h * h);
    if(q < 1.0) return s * (1.0 - 1.5 * q * q + 0.75 * q * q * q);
    if(q < 2.0) { double t = 2.0 - q; return s * 0.25 * t * t * t; }
    return 0.0;
}
// gradient magnitude dW/dr of W_cubic(r, h) (support 2h)
__device__ __forceinline__ double dW_cubic(double r, double h)
{
    double q = r / h;
    double s = 1.0 / (3.141592653589793 * h * h * h * h);
    if(q < 1.0) return s * (-3.0 * q + 2.25 * q * q);
    if(q < 2.0) { double t = 2.0 - q; return -s * 0.75 * t * t; }
    return 0.0;
}
// Akinci 2013 cohesion spline (support c): attraction at mid range, mild repulsion near 0
__device__ __forceinline__ double C_akinci(double r, double c)
{
    if(r >= c || r <= 0.0) return 0.0;
    double c3 = c * c * c, k = 32.0 / (3.141592653589793 * c3 * c3 * c3);
    double a  = (c - r) * (c - r) * (c - r) * r * r * r;
    return (2.0 * r > c) ? k * a : k * (2.0 * a - c3 * c3 / 64.0);
}
// surface normal n_i = (2 hs) sum_j (m_j / rho_j) grad W(x_i - x_j)   (zero inside, points out of the liquid at the surface)
__global__ void k_st_normals(int n, const double3* xp, const double* mp, const double* rho, const int* pidx, const int* cstart, const int* cend,
                             double3 org, double dx, int3 d, double hs, double3* nrm)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    double3 x = xp[p], acc = make_double3(0, 0, 0);
    int ci = (int)floor((x.x - org.x) / dx), cj = (int)floor((x.y - org.y) / dx), ck = (int)floor((x.z - org.z) / dx);
    for(int a = -1; a <= 1; ++a)
        for(int b = -1; b <= 1; ++b)
            for(int c = -1; c <= 1; ++c)
            {
                int i = ci + a, j = cj + b, k = ck + c;
                if(i < 0 || j < 0 || k < 0 || i >= d.x || j >= d.y || k >= d.z) continue;
                int cid = cell_lin(i, j, k, d), s0 = cstart[cid];
                if(s0 < 0) continue;
                for(int s = s0; s < cend[cid]; ++s)
                {
                    int q = pidx[s];
                    if(q == p) continue;
                    double3 dv = d3sub(x, xp[q]);
                    double  r  = sqrt(dv.x * dv.x + dv.y * dv.y + dv.z * dv.z);
                    if(r <= 1e-12 || r >= 2.0 * hs || rho[q] <= 0.0) continue;
                    acc = d3add(acc, d3scale(dv, mp[q] / rho[q] * dW_cubic(r, hs) / r));
                }
            }
    nrm[p] = d3scale(acc, -2.0 * hs);   // grad W points toward x_i - x_j decreasing; minus makes n point outward
}
// v_i += dt / m_i * sum_j K_ij ( F_cohesion + F_curvature ),  K_ij = 2 rho0 / (rho_i + rho_j)
__global__ void k_st_apply(int n, double3* vp, const double3* xp, const double* mp, const double* rho, const double3* nrm, const int* pidx,
                           const int* cstart, const int* cend, double3 org, double dx, int3 d, double gamma, double rho0, double c, double dt,
                           const int* stuck, double thin_rho, const int* wnear)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n || (stuck && stuck[p])) return;
    if(thin_rho > 0.0 && rho[p] >= thin_rho) return;   // bulk particle: no explicit cohesion (LIQ_ST_THIN)
    if(thin_rho > 0.0 && wnear && wnear[p]) return;    // wall-adjacent particle: its low SPH density is the wall, not a thin layer
    double3 x = xp[p], f = make_double3(0, 0, 0), np_ = nrm[p];
    double  mpp = mp[p];
    int ci = (int)floor((x.x - org.x) / dx), cj = (int)floor((x.y - org.y) / dx), ck = (int)floor((x.z - org.z) / dx);
    const int R = (int)ceil(c / dx - 1e-9);   // cells to search: every neighbour within the cohesion support c
    for(int a = -R; a <= R; ++a)
        for(int b = -R; b <= R; ++b)
            for(int cc = -R; cc <= R; ++cc)
            {
                int i = ci + a, j = cj + b, k = ck + cc;
                if(i < 0 || j < 0 || k < 0 || i >= d.x || j >= d.y || k >= d.z) continue;
                int cid = cell_lin(i, j, k, d), s0 = cstart[cid];
                if(s0 < 0) continue;
                for(int s = s0; s < cend[cid]; ++s)
                {
                    int q = pidx[s];
                    if(q == p) continue;
                    double3 dv = d3sub(x, xp[q]);
                    double  r  = sqrt(dv.x * dv.x + dv.y * dv.y + dv.z * dv.z);
                    if(r <= 1e-12 || r >= c) continue;
                    double K = 2.0 * rho0 / fmax(1e-9, rho[p] + rho[q]);
                    double3 coh  = d3scale(dv, -gamma * mpp * mp[q] * C_akinci(r, c) / r);
                    double3 curv = d3scale(d3sub(np_, nrm[q]), -gamma * mpp);
                    f = d3add(f, d3scale(d3add(coh, curv), K));
                }
            }
    double3 v = vp[p];
    vp[p] = d3add(v, d3scale(f, dt / mpp));
}
template <int MODE>  // 0 = push-apart displacement, 1 = density
__global__ void k_neighbors(int n, const double3* xp, const double* mp, const int* key, const int* pidx, const int* cstart, const int* cend,
                            double3 org, double dx, int3 d, double r0, double cap, double hs, double3* push, double* rho, const int* pflag,
                            const double* mc = nullptr, const int* cflag = nullptr, double rho_cell = 0.0, double surf_ratio = 0.0,
                            const int* wnear = nullptr, const double3* wX = nullptr, const double3* wn = nullptr,
                            int film = 0, const int* fnear = nullptr, double ground_y = -1e300, double film_h = 0.0, double r0f = 0.0,
                            double3* vp = nullptr)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    // film mode: water lying on the ground plane (centre within film_h of it, no solid triangle within the wall search
    // radius at the last wall pass). Thinner than a cell, it has no grid pressure; without the push its particles settle
    // into one dense layer on the plane and the rendered puddle thins away. Its ground-contact particles may slide along
    // the plane or lift off (the downward component is dropped), which cannot cross the plane; mesh contacts stay frozen.
    const bool fp = MODE == 0 && film && fnear && !fnear[p] && xp[p].y - ground_y < film_h;
    if(MODE == 0 && pflag && pflag[p] && !fp)
    {
        push[p] = make_double3(0, 0, 0);
        return;
    }
    if(fp) r0 = r0f;
    double3 x = xp[p];
    int     ci = (int)floor((x.x - org.x) / dx), cj = (int)floor((x.y - org.y) / dx), ck = (int)floor((x.z - org.z) / dx);
    if(MODE == 0 && mc && cflag && surf_ratio > 0.0 && !fp)
    {   // free-surface particles (under-filled cell) are not pushed: with no pressure to
        // hold them, a position push only lifts the surface layer ("evaporation")
        int cc = particle_cell(x, org, dx, d);
        if(!cflag[cc] || rho_cell / mc[cc] > surf_ratio) { push[p] = make_double3(0, 0, 0); return; }
    }
    double3 acc = make_double3(0, 0, 0);
    double  dens = 0.0;
    for(int a = -1; a <= 1; ++a)
        for(int b = -1; b <= 1; ++b)
            for(int c = -1; c <= 1; ++c)
            {
                int i = ci + a, j = cj + b, k = ck + c;
                if(i < 0 || j < 0 || k < 0 || i >= d.x || j >= d.y || k >= d.z) continue;
                int cid = cell_lin(i, j, k, d);
                int s0 = cstart[cid];
                if(s0 < 0) continue;
                int s1 = cend[cid];
                for(int s = s0; s < s1; ++s)
                {
                    int q = pidx[s];
                    double3 dv = d3sub(x, xp[q]);
                    double  r  = sqrt(dv.x * dv.x + dv.y * dv.y + dv.z * dv.z);
                    if(MODE == 0)
                    {
                        if(q == p || r >= r0) continue;
                        if(pflag && pflag[q] && !(film && fnear && !fnear[q] && xp[q].y - ground_y < film_h)) continue;
                        if(wnear)
                        {   // CPIC: a pair must not straddle a wall (the other particle behind this one's wall plane, or vice versa);
                            // wall points are the previous step's (positions moved < dhat since)
                            if(wnear[p]) { double3 X = wX[p], nn = wn[p]; if(nn.x * (xp[q].x - X.x) + nn.y * (xp[q].y - X.y) + nn.z * (xp[q].z - X.z) < 0.0) continue; }
                            if(wnear[q]) { double3 X = wX[q], nn = wn[q]; if(nn.x * (x.x - X.x) + nn.y * (x.y - X.y) + nn.z * (x.z - X.z) < 0.0) continue; }
                        }
                        if(r < 1e-9)
                        {   // coincident: deterministic tiny split along the index order
                            double sgn = (q > p) ? 1.0 : -1.0;
                            acc.x += 0.5 * r0 * sgn * 0.57735;
                            acc.y += 0.5 * r0 * sgn * 0.57735;
                            acc.z += 0.5 * r0 * sgn * 0.57735;
                            continue;
                        }
                        double f = 0.5 * (r0 - r) / r;
                        acc = d3add(acc, d3scale(dv, f));
                    }
                    else
                    {
                        if(r < 2.0 * hs) dens += mp[q] * W_cubic(r, hs);
                    }
                }
            }
    if(MODE == 0)
    {
        if(fp && pflag && pflag[p]) acc.y = fmax(acc.y, 0.0);   // touching the plane: slide or lift off only
        double l = sqrt(acc.x * acc.x + acc.y * acc.y + acc.z * acc.z);
        if(l > cap) acc = d3scale(acc, cap / l);
        push[p] = acc;
        if(fp && film == 2 && vp && l > 0.0)
        {   // inelastic: the velocity component against the push (the approach the push undid) is dropped
            double3 u  = d3scale(acc, 1.0 / fmin(l, cap));
            double  vn = vp[p].x * u.x + vp[p].y * u.y + vp[p].z * u.z;
            if(vn < 0.0) vp[p] = d3sub(vp[p], d3scale(u, vn));
        }
    }
    else
        rho[p] = dens;
}

// ----------------------------------------------------------------------------
// implicit surface tension (LIQ_STI): pair cohesion energy E = sum_pairs w_pq k psi(|x_p - x_q|) between free-surface /
// thin-layer particles, assembled into the Newton solve: energy (line search), gradient on the particle rows (mapped to
// the stencil nodes by the contact chain rule), Gauss-Newton Hessian on the 8-node trilinear stencils of BOTH particles
// (d = x_p - x_q = sum_i (W_ip - W_iq) u_i: a congruence of the PSD-projected 3x3 block => PSD).
// psi (unit = bond energy k = 2 gamma h^2): -1 at the rest distance r0 (no force), repulsive below (-1 + A (1 - r/r0)^2,
// A matching the curvature at r0), attractive up to the cutoff rc (smoothstep back to 0, zero force at rc).
// ----------------------------------------------------------------------------
__device__ int g_coh_mode = 0;   // 0 = LIQ_STI cohesion, 1 = LIQ_UG_REP repulsion (1/2 (r0 - r)^2 below r0, nothing beyond)
__device__ __forceinline__ void coh_psi(double r, double r0, double rc, double& e, double& d1, double& d2)
{
    if(g_coh_mode == 1)
    {   // cubic onset (C2 at r0: the Hessian does not jump when a pair enters the range, which kept the Newton chattering)
        if(r >= r0) { e = 0.0; d1 = 0.0; d2 = 0.0; return; }
        const double u = r0 - r;
        e = u * u * u / (3.0 * r0); d1 = -u * u / r0; d2 = 2.0 * u / r0;
        return;
    }
    if(r >= rc) { e = 0.0; d1 = 0.0; d2 = 0.0; return; }
    const double L = rc - r0;
    if(r >= r0)
    {
        const double s = (r - r0) / L;
        e  = -1.0 + s * s * (3.0 - 2.0 * s);
        d1 = 6.0 * s * (1.0 - s) / L;
        d2 = 6.0 * (1.0 - 2.0 * s) / (L * L);
        return;
    }
    const double A = 3.0 * r0 * r0 / (L * L), u = 1.0 - r / r0;
    e  = -1.0 + A * u * u;
    d1 = -2.0 * A * u / r0;
    d2 = 2.0 * A / (r0 * r0);
}
// gate, pass 1: cells holding a wall-adjacent particle
__global__ void k_coh_wallcells(int n, const double3* xp, const int* near, double3 org, double dx, int3 d, int* cwall)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n || !near || !near[p]) return;
    cwall[particle_cell(xp[p], org, dx, d)] = 1;
}
// gate, pass 2: a surface particle sits in a cell with an empty in-domain neighbour (air) and no wall-adjacent particle
__global__ void k_coh_gate(int n, const double3* xp, const int* cstart, const int* cwall, double3 org, double dx, int3 d, double* s)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    double3 x = xp[p];
    int     c = particle_cell(x, org, dx, d);
    double  g = 0.0;
    if(!cwall[c])
    {
        int ci = (int)floor((x.x - org.x) / dx), cj = (int)floor((x.y - org.y) / dx), ck = (int)floor((x.z - org.z) / dx);
        for(int a = -1; a <= 1 && g == 0.0; ++a)
            for(int b = -1; b <= 1 && g == 0.0; ++b)
                for(int e = -1; e <= 1 && g == 0.0; ++e)
                {
                    int i = ci + a, j = cj + b, k = ck + e;
                    if(i < 0 || j < 0 || k < 0 || i >= d.x || j >= d.y || k >= d.z) continue;   // outside the grid: not air
                    if(cstart[cell_lin(i, j, k, d)] < 0) g = 1.0;
                }
    }
    s[p] = g;
}
// pairs (p < q) of gated particles closer than r_list at the step start, weight min(s_p, s_q)
__global__ void k_coh_pairs(int n, const double3* xp, const double* s, const int* pidx, const int* cstart, const int* cend,
                            double3 org, double dx, int3 d, double r_list, int cap, int2* pairs, double* w, int* count)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n || s[p] <= 0.0) return;
    double3 x = xp[p];
    int     ci = (int)floor((x.x - org.x) / dx), cj = (int)floor((x.y - org.y) / dx), ck = (int)floor((x.z - org.z) / dx);
    for(int a = -1; a <= 1; ++a)
        for(int b = -1; b <= 1; ++b)
            for(int c = -1; c <= 1; ++c)
            {
                int i = ci + a, j = cj + b, k = ck + c;
                if(i < 0 || j < 0 || k < 0 || i >= d.x || j >= d.y || k >= d.z) continue;
                int cid = cell_lin(i, j, k, d);
                int s0  = cstart[cid];
                if(s0 < 0) continue;
                int s1 = cend[cid];
                for(int t = s0; t < s1; ++t)
                {
                    int q = pidx[t];
                    if(q <= p || s[q] <= 0.0) continue;
                    double3 dv = d3sub(x, xp[q]);
                    if(dv.x * dv.x + dv.y * dv.y + dv.z * dv.z >= r_list * r_list) continue;
                    int o = atomicAdd(count, 1);
                    if(o < cap) { pairs[o] = make_int2(p, q); w[o] = fmin(s[p], s[q]); }
                }
            }
}
__global__ void k_coh_energy(int np, const int2* pairs, const double* w, const double3* x, int pslot, double k, double r0, double rc, double* part)
{
    extern __shared__ double sh[];
    int    i = blockIdx.x * blockDim.x + threadIdx.x;
    double e = 0.0;
    if(i < np)
    {
        int2    pq = pairs[i];
        double3 dv = d3sub(x[pslot + pq.x], x[pslot + pq.y]);
        double  r  = sqrt(dv.x * dv.x + dv.y * dv.y + dv.z * dv.z), ps, d1, d2;
        coh_psi(r, r0, rc, ps, d1, d2);
        e = w[i] * k * ps;
    }
    sh[threadIdx.x] = e;
    __syncthreads();
    for(int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if(threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if(threadIdx.x == 0) part[blockIdx.x] = sh[0];
}
// gradient (Newton units: x dt^2) on the particle rows; the contact chain rule then carries it to the stencil nodes
__global__ void k_coh_grad(int np, const int2* pairs, const double* w, const double3* x, int pslot, double kdt2, double r0, double rc, double3* fb)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= np) return;
    int2    pq = pairs[i];
    double3 dv = d3sub(x[pslot + pq.x], x[pslot + pq.y]);
    double  r  = sqrt(dv.x * dv.x + dv.y * dv.y + dv.z * dv.z), e, d1, d2;
    if(r < 1e-12) return;
    coh_psi(r, r0, rc, e, d1, d2);
    if(d1 == 0.0) return;
    double sc = w[i] * kdt2 * d1 / r;
    atomicAdd(&fb[pslot + pq.x].x, sc * dv.x);
    atomicAdd(&fb[pslot + pq.x].y, sc * dv.y);
    atomicAdd(&fb[pslot + pq.x].z, sc * dv.z);
    atomicAdd(&fb[pslot + pq.y].x, -sc * dv.x);
    atomicAdd(&fb[pslot + pq.y].y, -sc * dv.y);
    atomicAdd(&fb[pslot + pq.y].z, -sc * dv.z);
}
// Hessian: node coefficients c = W8_p - W8_q over the union of the two step-start cells (<= 15 nodes: the pair is closer
// than one cell, so the cells are neighbours); the upper blocks c_i c_j H of the union, none beyond the cutoff
__device__ __forceinline__ int coh_union(int2 pq, const double3* xp, double3 org, double dx, int3 dims, const int* node_slot, int base, int* vid, double* c)
{
    Stencil8 sp = stencil8(xp[pq.x], org, dx, dims, node_slot, base), sq = stencil8(xp[pq.y], org, dx, dims, node_slot, base);
    int m = 0;
    for(int k = 0; k < 8; ++k)
        if(sp.vid[k] >= 0 && sp.w[k] > 0.0) { vid[m] = sp.vid[k]; c[m] = sp.w[k]; ++m; }
    for(int k = 0; k < 8; ++k)
    {
        if(sq.vid[k] < 0 || sq.w[k] <= 0.0) continue;
        int f = -1;
        for(int l = 0; l < m; ++l)
            if(vid[l] == sq.vid[k]) { f = l; break; }
        if(f >= 0) c[f] -= sq.w[k];
        else if(m < 15) { vid[m] = sq.vid[k]; c[m] = -sq.w[k]; ++m; }
    }
    return m;
}
__global__ void k_coh_hess_count(int np, const int2* pairs, const double3* x, int pslot, const double3* xp, double3 org, double dx, int3 dims,
                                 const int* node_slot, int base, double rc, int* nblk)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= np) return;
    int2    pq = pairs[i];
    double3 dv = d3sub(x[pslot + pq.x], x[pslot + pq.y]);
    double  r2 = dv.x * dv.x + dv.y * dv.y + dv.z * dv.z;
    if(r2 >= rc * rc || r2 < 1e-24) { nblk[i] = 0; return; }
    int    vid[15];
    double c[15];
    int    m = coh_union(pq, xp, org, dx, dims, node_slot, base, vid, c);
    nblk[i] = m * (m + 1) / 2;
}
__global__ void k_coh_hess(int np, const int2* pairs, const double* w, const double3* x, int pslot, const double3* xp, double3 org, double dx,
                           int3 dims, const int* node_slot, int base, double kdt2, double r0, double rc, const int* nblk, const int* off, int live,
                           int* rows, int* cols, Eigen::Matrix3d* vals)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= np || nblk[i] == 0) return;
    int2    pq = pairs[i];
    double3 dv = d3sub(x[pslot + pq.x], x[pslot + pq.y]);
    double  r  = sqrt(dv.x * dv.x + dv.y * dv.y + dv.z * dv.z), e = 0.0, d1 = 0.0, d2 = 0.0;
    coh_psi(r, r0, rc, e, d1, d2);
    const double a = w[i] * kdt2 * fmax(d2, 0.0);         // along the pair (PSD projection)
    const double b = w[i] * kdt2 * fmax(d1 / r, 0.0);     // across it
    int    vid[15];
    double c[15];
    int    m = coh_union(pq, xp, org, dx, dims, node_slot, base, vid, c);
    const double nv[3] = {dv.x / r, dv.y / r, dv.z / r};
    Eigen::Matrix3d H;
    for(int u = 0; u < 3; ++u)
        for(int v = 0; v < 3; ++v) H(u, v) = (a - b) * nv[u] * nv[v] + (u == v ? b : 0.0);
    int o = live + off[i];
    for(int k = 0; k < m; ++k)
        for(int l = k; l < m; ++l, ++o)
        {
            int ri = vid[k], cj = vid[l];
            if(ri > cj) { int t = ri; ri = cj; cj = t; }
            rows[o] = ri; cols[o] = cj; vals[o] = (c[k] * c[l]) * H;
        }
}
// ---- LIQ_STM: membrane surface tension ----------------------------------------------------------------------------------
// Every free-surface particle carries a surface element of area A0 (= stm_area h^2) with outward normal n, spanned by
// tangents a, b (|a| = |b| = sqrt(A0), a ⟂ b ⟂ n). Its energy is sigma |F a x F b|, with F = sum_q x_q (grad w_q)^T over the 8
// nodes of the particle's step-start cell (trilinear MPM Lite kernel; the nodes start on the lattice, so F = I at the step
// start). At F = I the gradient on node q is sigma A0 (I - n n^T) grad w_q: the surface stress, whose divergence is the
// Laplace (curvature) force. No particle pairs and no rest length, unlike the pair cohesion of LIQ_STI.
__device__ __forceinline__ void stm_stencil(double3 x, double3 org, double dx, int3 dims, const int* node_slot, int base, int* vid, double3* gw)
{
    double r[3] = {(x.x - org.x) / dx, (x.y - org.y) / dx, (x.z - org.z) / dx};
    int    n[3] = {dims.x, dims.y, dims.z}, b[3];
    double f[3];
    for(int a = 0; a < 3; ++a)
    {
        b[a] = (int)floor(r[a]);
        if(b[a] < 0) b[a] = 0;
        if(b[a] > n[a] - 1) b[a] = n[a] - 1;
        f[a] = fmin(fmax(r[a] - b[a], 0.0), 1.0);
    }
    for(int q = 0; q < 8; ++q)
    {
        int    i = (q >> 2) & 1, j = (q >> 1) & 1, k = q & 1;
        double A = i ? f[0] : 1.0 - f[0], B = j ? f[1] : 1.0 - f[1], C = k ? f[2] : 1.0 - f[2];
        double dA = i ? 1.0 : -1.0, dB = j ? 1.0 : -1.0, dC = k ? 1.0 : -1.0;
        gw[q] = make_double3(dA * B * C / dx, A * dB * C / dx, A * B * dC / dx);
        int sl = node_slot[node_lin(b[0] + i, b[1] + j, b[2] + k, dims)];
        vid[q] = (sl < 0) ? -1 : base + sl;
    }
}
// pass 1: surface flag + outward normal from the neighbourhood asymmetry (weighted centroid offset of the neighbours within R).
// Solids count as occupied: every particle with a wall point (within wfc_r h of a wall or the ground) also contributes its mirror image
// across its wall plane, so the liquid-solid interface carries no surface element and a wall-adjacent particle under the free surface
// gets the free-surface normal. With no liquid-solid energy this is a neutral (90 degree) contact angle.
__global__ void k_stm_flag(int n, const double3* xp, const int* pidx, const int* cstart, const int* cend, const int* wv, const double3* wX,
                           const double3* wn, double3 org, double dx, int3 d, double R, double thr, int* flag, double3* nrm)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    flag[p] = 0;
    double3 x  = xp[p];
    int     ci = (int)floor((x.x - org.x) / dx), cj = (int)floor((x.y - org.y) / dx), ck = (int)floor((x.z - org.z) / dx);
    if(ci < 0 || cj < 0 || ck < 0 || ci >= d.x || cj >= d.y || ck >= d.z) return;
    double sx = 0.0, sy = 0.0, sz = 0.0, W = 0.0;
    auto add = [&](double3 y) {
        double3 dv = d3sub(x, y);
        double  r  = sqrt(dv.x * dv.x + dv.y * dv.y + dv.z * dv.z);
        if(r >= R || r < 1e-12) return;
        double w = 1.0 - r / R;
        sx += w * dv.x; sy += w * dv.y; sz += w * dv.z; W += w;
    };
    auto mirror = [&](int q) {   // reflection of particle q across its own wall plane (into the solid)
        double3 nq = wn[q], Xq = wX[q], yq = xp[q];
        double  dist = (yq.x - Xq.x) * nq.x + (yq.y - Xq.y) * nq.y + (yq.z - Xq.z) * nq.z;
        return make_double3(yq.x - 2.0 * dist * nq.x, yq.y - 2.0 * dist * nq.y, yq.z - 2.0 * dist * nq.z);
    };
    if(wv && wv[p]) add(mirror(p));
    for(int a = -1; a <= 1; ++a)
        for(int b = -1; b <= 1; ++b)
            for(int e = -1; e <= 1; ++e)
            {
                int i = ci + a, j = cj + b, k = ck + e;
                if(i < 0 || j < 0 || k < 0 || i >= d.x || j >= d.y || k >= d.z) continue;
                int c = cell_lin(i, j, k, d);
                if(cstart[c] < 0) continue;
                for(int t = cstart[c]; t < cend[c]; ++t)
                {
                    int q = pidx[t];
                    if(q != p) add(xp[q]);
                    if(wv && wv[q] && q != p) add(mirror(q));
                }
            }
    if(W <= 0.0) return;   // isolated particle: no surface element (a one-particle droplet)
    sx /= W; sy /= W; sz /= W;
    double as = sqrt(sx * sx + sy * sy + sz * sz);
    if(as < thr) return;
    flag[p] = 1;
    nrm[p]  = make_double3(sx / as, sy / as, sz / as);
}
// pass 1b (amode 1): element area = 1 / areal density of the surface elements around p, density = sum of a 2D tent kernel
// 3 (1 - r/Rs) / (pi Rs^2) over the flagged particles within Rs (p itself and, with walls, the mirror images of flagged particles
// across their wall planes, so the free surface continues into the solid and a contact-line element is not over-weighted).
// Wherever the detection flags one layer, two layers or a sparse layer, the elements share the local surface area between
// them, so the total tracks the true area (offline: jittered slab 1.00; spheres of radius 4 / 6 / 8 h: 0.81 / 0.89 / 0.90
// of 4 pi R^2, the rest being the outer particle centres sitting h/2 inside). The normal is the kernel-weighted average of
// the flagged neighbours' normals (median error 12 -> 5 degrees on a jittered sphere).
__global__ void k_stm_area(int n, const double3* xp, const int* flag, const double3* nrm, const int* pidx, const int* cstart, const int* cend,
                           const int* wv, const double3* wX, const double3* wn, double3 org, double dx, int3 d, double Rs, double mult,
                           double amax, double* A, double3* n2)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n || !flag[p]) return;
    double3 x  = xp[p];
    int     ci = (int)floor((x.x - org.x) / dx), cj = (int)floor((x.y - org.y) / dx), ck = (int)floor((x.z - org.z) / dx);
    double  dens = 0.0, sx = 0.0, sy = 0.0, sz = 0.0;
    auto add = [&](double3 y, double3 m) {
        double3 dv = d3sub(x, y);
        double  r  = sqrt(dv.x * dv.x + dv.y * dv.y + dv.z * dv.z);
        if(r >= Rs) return;
        double w = 1.0 - r / Rs;
        dens += w; sx += w * m.x; sy += w * m.y; sz += w * m.z;
    };
    auto add_mirror = [&](int q) {   // flagged q reflected across its own wall plane, with its normal reflected too
        double3 nq = wn[q], Xq = wX[q], yq = xp[q], mq = nrm[q];
        double  dist = (yq.x - Xq.x) * nq.x + (yq.y - Xq.y) * nq.y + (yq.z - Xq.z) * nq.z;
        double  mn   = mq.x * nq.x + mq.y * nq.y + mq.z * nq.z;
        add(make_double3(yq.x - 2.0 * dist * nq.x, yq.y - 2.0 * dist * nq.y, yq.z - 2.0 * dist * nq.z),
            make_double3(mq.x - 2.0 * mn * nq.x, mq.y - 2.0 * mn * nq.y, mq.z - 2.0 * mn * nq.z));
    };
    for(int a = -1; a <= 1; ++a)
        for(int b = -1; b <= 1; ++b)
            for(int e = -1; e <= 1; ++e)
            {
                int i = ci + a, j = cj + b, k = ck + e;
                if(i < 0 || j < 0 || k < 0 || i >= d.x || j >= d.y || k >= d.z) continue;
                int c = cell_lin(i, j, k, d);
                if(cstart[c] < 0) continue;
                for(int t = cstart[c]; t < cend[c]; ++t)
                {
                    int q = pidx[t];
                    if(!flag[q]) continue;
                    add(xp[q], nrm[q]);   // includes p itself (r = 0)
                    if(wv && wv[q]) add_mirror(q);
                }
            }
    A[p] = mult * fmin(3.14159265358979323846 * Rs * Rs / (3.0 * dens), amax);   // dens >= 1 (the self term)
    double l = sqrt(sx * sx + sy * sy + sz * sz);
    n2[p] = l > 1e-12 ? make_double3(sx / l, sy / l, sz / l) : nrm[p];
}
// pass 2: compacted surface elements in a deterministic order (exclusive scan of the flags): particle index + tangents a, b
// (|a| = |b| = sqrt(A0): L for amode 0, sqrt(A[p]) for amode 1). The node stencil is rebuilt from the step-start position by
// every assembly (node slots are assigned after this pass).
__global__ void k_stm_fill(int n, const int* flag, const int* scan, const double3* nrm, double L, int* sp, double3* sa, double3* sb,
                           const double* A = nullptr)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n || !flag[p]) return;
    int     k   = scan[p];
    double3 nv  = nrm[p];
    if(A) L = sqrt(A[p]);
    double3 ref = fabs(nv.x) < 0.9 ? make_double3(1.0, 0.0, 0.0) : make_double3(0.0, 1.0, 0.0);
    double3 t1  = make_double3(nv.y * ref.z - nv.z * ref.y, nv.z * ref.x - nv.x * ref.z, nv.x * ref.y - nv.y * ref.x);
    double  l1  = sqrt(t1.x * t1.x + t1.y * t1.y + t1.z * t1.z);
    t1          = d3scale(t1, 1.0 / l1);
    double3 t2  = make_double3(nv.y * t1.z - nv.z * t1.y, nv.z * t1.x - nv.x * t1.z, nv.x * t1.y - nv.y * t1.x);
    sp[k] = p; sa[k] = d3scale(t1, L); sb[k] = d3scale(t2, L);
}
// the element's 8 node slots and coefficients c_a = grad w . a, c_b = grad w . b (missing nodes -> a valid node, zero weight)
__device__ __forceinline__ bool stm_elem(int i, const int* sp, const double3* sa, const double3* sb, const double3* xp, double3 org, double dx,
                                         int3 d, const int* node_slot, int base, int* vid, double* ca, double* cb)
{
    double3 gw[8];
    stm_stencil(xp[sp[i]], org, dx, d, node_slot, base, vid, gw);
    double3 a = sa[i], b = sb[i];
    int     fb = -1;
    for(int q = 0; q < 8; ++q)
        if(vid[q] >= 0) { fb = vid[q]; break; }
    if(fb < 0) return false;
    for(int q = 0; q < 8; ++q)
    {
        ca[q] = gw[q].x * a.x + gw[q].y * a.y + gw[q].z * a.z;
        cb[q] = gw[q].x * b.x + gw[q].y * b.y + gw[q].z * b.z;
        if(vid[q] < 0) { vid[q] = fb; ca[q] = cb[q] = 0.0; }
    }
    return true;
}
__device__ __forceinline__ void stm_uv(const int* vid, const double* ca, const double* cb, const double3* x, double3& u, double3& v)
{
    u = make_double3(0, 0, 0); v = make_double3(0, 0, 0);
    for(int q = 0; q < 8; ++q)
    {
        double3 xq = x[vid[q]];
        u.x += ca[q] * xq.x; u.y += ca[q] * xq.y; u.z += ca[q] * xq.z;
        v.x += cb[q] * xq.x; v.y += cb[q] * xq.y; v.z += cb[q] * xq.z;
    }
}
#define STM_ARGS const int* sp, const double3* sa, const double3* sb, const double3* xp, double3 org, double dx, int3 d, const int* node_slot, int base
#define STM_ELEM(i) int vid[8]; double ca[8], cb[8]; bool ok_ = stm_elem(i, sp, sa, sb, xp, org, dx, d, node_slot, base, vid, ca, cb)
__global__ void k_stm_energy(int m, STM_ARGS, const double3* x, double sigma, double* part)
{
    extern __shared__ double sh[];
    int    i = blockIdx.x * blockDim.x + threadIdx.x;
    double e = 0.0;
    if(i < m)
    {
        STM_ELEM(i);
        double3 u = make_double3(0, 0, 0), v = make_double3(0, 0, 0);
        if(ok_) stm_uv(vid, ca, cb, x, u, v);
        double cx = u.y * v.z - u.z * v.y, cy = u.z * v.x - u.x * v.z, cz = u.x * v.y - u.y * v.x;
        e = sigma * sqrt(cx * cx + cy * cy + cz * cz);
    }
    sh[threadIdx.x] = e;
    __syncthreads();
    for(int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if(threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if(threadIdx.x == 0) part[blockIdx.x] = sh[0];
}
// gradient (Newton units: x dt^2) straight onto the node rows
__global__ void k_stm_grad(int m, STM_ARGS, const double3* x, double sdt2, double3* fb)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= m) return;
    STM_ELEM(i);
    if(!ok_) return;
    double3 u, v;
    stm_uv(vid, ca, cb, x, u, v);
    double uu = u.x * u.x + u.y * u.y + u.z * u.z, vv = v.x * v.x + v.y * v.y + v.z * v.z, uv = u.x * v.x + u.y * v.y + u.z * v.z;
    double A2 = uu * vv - uv * uv;
    if(A2 <= 1e-40) return;
    double  A  = sqrt(A2), s = sdt2 / A;
    double3 gu = make_double3(s * (vv * u.x - uv * v.x), s * (vv * u.y - uv * v.y), s * (vv * u.z - uv * v.z));
    double3 gv = make_double3(s * (uu * v.x - uv * u.x), s * (uu * v.y - uv * u.y), s * (uu * v.z - uv * u.z));
    for(int q = 0; q < 8; ++q)
    {
        atomicAdd(&fb[vid[q]].x, ca[q] * gu.x + cb[q] * gv.x);
        atomicAdd(&fb[vid[q]].y, ca[q] * gu.y + cb[q] * gv.y);
        atomicAdd(&fb[vid[q]].z, ca[q] * gu.z + cb[q] * gv.z);
    }
}
// symmetric 6x6 -> its PSD projection (cyclic Jacobi)
__device__ void stm_psd6(double H[6][6])
{
    double V[6][6];
    for(int i = 0; i < 6; ++i)
        for(int j = 0; j < 6; ++j) V[i][j] = (i == j) ? 1.0 : 0.0;
    for(int sweep = 0; sweep < 12; ++sweep)
    {
        double off = 0.0, dia = 0.0;
        for(int i = 0; i < 6; ++i)
        {
            dia += H[i][i] * H[i][i];
            for(int j = i + 1; j < 6; ++j) off += H[i][j] * H[i][j];
        }
        if(off <= 1e-30 * dia + 1e-300) break;
        for(int p = 0; p < 5; ++p)
            for(int q = p + 1; q < 6; ++q)
            {
                if(fabs(H[p][q]) < 1e-300) continue;
                double th = (H[q][q] - H[p][p]) / (2.0 * H[p][q]);
                double t  = (th >= 0.0 ? 1.0 : -1.0) / (fabs(th) + sqrt(th * th + 1.0));
                double c = 1.0 / sqrt(t * t + 1.0), s = t * c;
                for(int k = 0; k < 6; ++k)
                {   // H <- J^T H J (columns, then rows)
                    double hp = H[k][p], hq = H[k][q];
                    H[k][p] = c * hp - s * hq; H[k][q] = s * hp + c * hq;
                }
                for(int k = 0; k < 6; ++k)
                {
                    double hp = H[p][k], hq = H[q][k];
                    H[p][k] = c * hp - s * hq; H[q][k] = s * hp + c * hq;
                }
                for(int k = 0; k < 6; ++k)
                {
                    double vp = V[k][p], vq = V[k][q];
                    V[k][p] = c * vp - s * vq; V[k][q] = s * vp + c * vq;
                }
            }
    }
    double lam[6];
    for(int i = 0; i < 6; ++i) lam[i] = fmax(H[i][i], 0.0);
    for(int i = 0; i < 6; ++i)
        for(int j = 0; j < 6; ++j)
        {
            double s = 0.0;
            for(int k = 0; k < 6; ++k) s += V[i][k] * lam[k] * V[j][k];
            H[i][j] = s;
        }
}
// Hessian: the area's 6x6 Hessian in (u, v), PSD-projected, mapped to the 8 nodes (36 upper blocks per element)
__global__ void k_stm_hess(int m, STM_ARGS, const double3* x, double sdt2, int live, int* rows, int* cols, Eigen::Matrix3d* vals)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= m) return;
    STM_ELEM(i);
    if(!ok_) { for(int q = 0; q < 8; ++q) { vid[q] = base; ca[q] = cb[q] = 0.0; } }   // never expected: zero blocks on a valid row
    double3 u, v;
    stm_uv(vid, ca, cb, x, u, v);
    double U[3] = {u.x, u.y, u.z}, Vv[3] = {v.x, v.y, v.z};
    double uu = u.x * u.x + u.y * u.y + u.z * u.z, vv = v.x * v.x + v.y * v.y + v.z * v.z, uv = u.x * v.x + u.y * v.y + u.z * v.z;
    double A2 = uu * vv - uv * uv;
    double H[6][6];
    for(int r = 0; r < 6; ++r)
        for(int c = 0; c < 6; ++c) H[r][c] = 0.0;
    if(A2 > 1e-40)
    {
        double A = sqrt(A2), gu[3], gv[3];
        for(int k = 0; k < 3; ++k) { gu[k] = (vv * U[k] - uv * Vv[k]) / A; gv[k] = (uu * Vv[k] - uv * U[k]) / A; }
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c)
            {
                double d = (r == c) ? 1.0 : 0.0;
                H[r][c]         = ((vv * d - Vv[r] * Vv[c]) - gu[r] * gu[c]) / A;                         // uu
                H[3 + r][3 + c] = ((uu * d - U[r] * U[c]) - gv[r] * gv[c]) / A;                           // vv
                H[r][3 + c]     = ((2.0 * U[r] * Vv[c] - Vv[r] * U[c] - uv * d) - gu[r] * gv[c]) / A;     // uv
                H[3 + c][r]     = H[r][3 + c];
            }
        stm_psd6(H);
    }
    int o = live + 36 * i;
    for(int q = 0; q < 8; ++q)
        for(int r = q; r < 8; ++r, ++o)
        {
            double aq = ca[q], bq = cb[q], ar = ca[r], br = cb[r];
            Eigen::Matrix3d B;
            for(int a = 0; a < 3; ++a)
                for(int b = 0; b < 3; ++b)
                    B(a, b) = sdt2 * (aq * ar * H[a][b] + aq * br * H[a][3 + b] + bq * ar * H[3 + a][b] + bq * br * H[3 + a][3 + b]);
            int ri = vid[q], cj = vid[r];
            if(ri > cj) { int t = ri; ri = cj; cj = t; B.transposeInPlace(); }
            rows[o] = ri; cols[o] = cj; vals[o] = B;
        }
}
__global__ void k_apply_push(int n, double3* xp, const double3* push, double3 lo, double3 hi)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    double3 x = d3add(xp[p], push[p]);
    double3 xc = make_double3(fmin(fmax(x.x, lo.x), hi.x), fmin(fmax(x.y, lo.y), hi.y), fmin(fmax(x.z, lo.z), hi.z));
    if(xc.x != x.x || xc.y != x.y || xc.z != x.z) LIQ_FBK(5);
    xp[p] = xc;
}
// pump: dropper / pipette suction, hold and squeeze (see LiquidSystem::set_pump)
// MODE 0: velocity term (before P2G); MODE 1: position term (after G2P): conveyor along the
// axis (particles within 5 mm of the cavity top are not conveyed further up) + follow the tube
// velocity terms act only INSIDE the bore (t in [0, L]); the position conveyor also on the
// 10 mm intake below the tip, so water is drawn in through the opening
__device__ __forceinline__ bool pump_region(double3 x, double3 a, double3 b, double r, double t_lo, double3& ax, double& t, double& L, double& rad)
{
    ax = d3sub(b, a);
    L  = sqrt(ax.x * ax.x + ax.y * ax.y + ax.z * ax.z);
    if(L < 1e-9) return false;
    ax = d3scale(ax, 1.0 / L);
    double3 d = d3sub(x, a);
    t         = d.x * ax.x + d.y * ax.y + d.z * ax.z;      // along the axis, 0 at the tip
    if(t < t_lo || t > L) return false;
    double3 rv = d3sub(d, d3scale(ax, t));
    rad        = sqrt(rv.x * rv.x + rv.y * rv.y + rv.z * rv.z);
    return rad <= r;
}
// bore_clear: for the position conveyor in the intake region (t < 0) only particles within
// bore_clear = bore radius - dhat of the axis are drawn in. The region radius is deliberately a bit
// larger than the bore, so without this a particle under the glass annulus (bore radius .. outer
// radius) was conveyed straight into the tube's end face: the conveyor moves positions before the
// step, outside the barrier/CCD, and a particle that starts inside the wall stays there (lab stage 1
// at 5 cm/s: 15-22 particles embedded 0-12 mm above the tip at radius 3.2-4.3 mm, bore 3.0 mm).
template <int MODE>
__global__ void k_pump(int n, double3* xp, double3* vp, double3 a, double3 b, double r, double accel, double conveyor, double3 v_tube,
                       double follow, double track, double dt, double bore_clear, int* count)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    double3 ax; double t, L, rad;
    if(!pump_region(xp[p], a, b, r, MODE == 0 ? 0.0 : -0.010, ax, t, L, rad)) return;
    if(MODE == 1 && t < 0.0 && rad > bore_clear) return;   // under the end face, not under the opening
    if(MODE == 0)
    {   // track the tube's velocity FIRST, then add the gravity-cancelling acceleration (tracking
        // after it would remove part of the cancellation: the column sagged at exactly g*dt per step)
        double3 v = vp[p];
        if(track > 0.0) v = d3add(v, d3scale(d3sub(v_tube, v), track));
        vp[p] = d3add(v, d3scale(ax, accel * dt));
    }
    else
    {
        double c = conveyor;
        if(c > 0.0 && t > L - 0.005) c = 0.0;              // do not convey into the cavity ceiling
        double3 dx = d3add(d3scale(ax, c * dt), d3scale(v_tube, follow * dt));
        xp[p] = d3add(xp[p], dx);
    }
    if(count) atomicAdd(count, 1);
}
// chamber-pressure pump: column census (count, volume, mean axial velocity, liquid at the opening) and the axial kick
__global__ void k_pump_column(int n, const double3* xp, const double3* vp, const double* Jp, const double* Vp0, double3 a, double3 b, double r,
                              double t_lo, double t_open, double* red)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    double3 ax; double t, L, rad;
    if(!pump_region(xp[p], a, b, r, t_lo, ax, t, L, rad)) return;
    atomicAdd(&red[0], 1.0);
    atomicAdd(&red[1], Vp0[p] * Jp[p]);
    atomicAdd(&red[2], vp[p].x * ax.x + vp[p].y * ax.y + vp[p].z * ax.z);
    if(t <= t_open) atomicAdd(&red[3], 1.0);
}
__global__ void k_pump_kick(int n, const double3* xp, double3* vp, double3 a, double3 b, double r, double t_lo, double dv)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    double3 ax; double t, L, rad;
    if(!pump_region(xp[p], a, b, r, t_lo, ax, t, L, rad)) return;
    vp[p] = d3add(vp[p], d3scale(ax, dv));
}
// LIQ_PUMP_TRACK_RIGID (2026-11-12): velocity tracking of the bore column with the tube's RIGID-BODY velocity.
// k_pump<0> relaxes every particle in the bore towards one velocity, the tip's (tip displacement between two set_pump calls).
// That is right for a translating tube. A dropper held at the bulb swings about the grip, 200+ mm above the tip, so the wall
// at height s moves slower than the tip; particles given the tip's velocity run AHEAD of the wall there, hit the leading wall and
// push the tube further along its own motion: a self-excited swing. Rokae replay with 66 particles in the bore, palm at rest:
// tip swing in the palm frame 16 -> 43 mm peak-to-peak in 1.5 s at 3 Hz, dying out as the bore empties; particle offset along
// the wall's motion -0.2 mm at 0-15 mm from the tip, +0.4 mm at 60-90 mm.
// Here the target velocity is the tube's own at the particle: v(t) = v_tip + (t / L) (v_top - v_tip), exact on the axis for any
// rigid motion (the bore radius times the unknown spin about the axis is neglected). The top's previous position is kept in
// this file (the class has no slot for it and its layout must not change for the isolated builds).
static int     g_pump_track_rigid      = -1;      // LIQ_PUMP_TRACK_RIGID (-1 = not read yet)
static bool    g_pump_top_has_prev     = false;
static double3 g_pump_top_prev         = make_double3(0, 0, 0);
__global__ void k_pump_track_rigid(int n, const double3* xp, double3* vp, double3 a, double3 b, double r, double accel, double3 v_tip, double3 v_top,
                                   double track, double dt)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    double3 ax; double t, L, rad;
    if(!pump_region(xp[p], a, b, r, 0.0, ax, t, L, rad)) return;
    const double3 vt = d3add(v_tip, d3scale(d3sub(v_top, v_tip), t / L));
    double3       v  = vp[p];
    if(track > 0.0) v = d3add(v, d3scale(d3sub(vt, v), track));   // track first, then the gravity-cancelling acceleration (as k_pump<0>)
    vp[p] = d3add(v, d3scale(ax, accel * dt));
}
// LIQ_PUMP_SINK (2026-11-12): supply for the position conveyor of the kinematic pump.
// The plain conveyor (k_pump<1>) lifts only the column of radius bore - dhat under the opening. That column holds ~6 particles
// (bore 3.2 mm, spacing 2.5 mm) and is emptied in 0.2 s; the hole it leaves is narrower than a grid cell, the grid sees neither
// a mass nor a pressure deficit there (the volume state follows the velocity divergence, not the particle density), so the
// surrounding water never refills it. The intake then depends on how far the tip happens to wander under water, not on the
// commanded flow (Rokae dropper replay: target 61-79 particles, drawn 16-47), and the conveyor, still running, carries the
// short slug to the top of the tube.
// With LIQ_PUMP_SINK=1 and the tip immersed, the conveyor moves the pool particles in the slab -h <= t <= 0 below the tip plane
// along a divergence-free field that carries exactly the conveyed flux Q = c pi rc^2 (rc = bore - dhat) into the opening:
//   capture cylinder rad <= rc:  axial  w = c (1 + t / h)   (0 at the bottom of the slab, c at the tip plane)
//                                radial u = -c rad / (2 h)
//   around it, rc < rad <= R:    radial u = -c rc^2 / (2 h rad)   (line sink of strength Q / h; faded out between R1 and R)
// so the particle density under the opening stays that of the pool. Displacements are <= c dt rc / (2 h) (0.08 mm per 5 ms step
// at 15 cm/s) outside the capture cylinder; inside it particles converge towards the axis, i.e. away from the glass.
// c < 0 with liquid leaving the bore is the same field reversed (source). In air (no liquid around the opening) the plain
// conveyor is used unchanged. Two guards keep the column attached to the opening: the bore column is lifted only while a
// particle is ready to follow it through the opening, and the source field runs only while a particle is leaving the bore.
struct PumpSinkCfg
{
    bool   init  = false;
    int    on    = 0;        // LIQ_PUMP_SINK
    double h     = 0.010;    // LIQ_PUMP_SINK_H: slab thickness below the tip plane (m)
    double R1    = 0.012;    // LIQ_PUMP_SINK_R1: the line sink is at full strength up to this radius ...
    double R     = 0.020;    // LIQ_PUMP_SINK_R: ... and zero beyond this one (m)
    double top   = 0.003;    // LIQ_PUMP_SINK_TOP: layer below the tip plane whose particle count tells whether the tip is immersed (m)
    int    n_min = 12;       // LIQ_PUMP_SINK_NMIN: particles in that layer (rc < rad <= R1) for "immersed"
    double gate  = -1.0;     // LIQ_PUMP_SINK_GATE: reach of the attach guards along the axis (m); < 0 = one seed spacing, 0 = guards off
};
static PumpSinkCfg g_pump_sink;
static PumpSinkCfg& pump_sink_cfg()
{
    PumpSinkCfg& S = g_pump_sink;
    if(!S.init)
    {
        S.init  = true;
        S.on    = liquid_env("LIQ_PUMP_SINK", S.on);
        S.h     = liquid_env("LIQ_PUMP_SINK_H", S.h);
        S.R1    = liquid_env("LIQ_PUMP_SINK_R1", S.R1);
        S.R     = liquid_env("LIQ_PUMP_SINK_R", S.R);
        S.top   = liquid_env("LIQ_PUMP_SINK_TOP", S.top);
        S.n_min = liquid_env("LIQ_PUMP_SINK_NMIN", S.n_min);
        S.gate  = liquid_env("LIQ_PUMP_SINK_GATE", S.gate);
        liquid_require(S.on == 0 || S.on == 1, "LIQ_PUMP_SINK must be 0 or 1");
        liquid_require(S.h > 0.0 && S.R1 > 0.0 && S.R > S.R1 && S.top > 0.0 && S.top <= S.h && S.n_min >= 1,
                       "LIQ_PUMP_SINK_H / _R1 / _R / _TOP / _NMIN: need h > 0, 0 < R1 < R, 0 < top <= h, n_min >= 1");
    }
    return S;
}
// census for the sink conveyor: red[0] = liquid around the opening (-top <= t < 0, rc < rad <= R1): the tip is immersed;
// red[1] = particles ready to enter (capture cylinder, within `gate` below the tip plane); red[2] = bore particles within
// `gate` above the tip plane (ready to leave); red[3] = particles in the bore
__global__ void k_pump_sink_census(int n, const double3* xp, double3 a, double3 b, double r, double rc, double R1, double top, double gate, double* red)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    double3 ax = d3sub(b, a);
    const double L = sqrt(ax.x * ax.x + ax.y * ax.y + ax.z * ax.z);
    if(L < 1e-9) return;
    ax = d3scale(ax, 1.0 / L);
    const double3 d = d3sub(xp[p], a);
    const double  t = d.x * ax.x + d.y * ax.y + d.z * ax.z;
    if(t < -fmax(top, gate) || t > L) return;
    const double3 rv  = d3sub(d, d3scale(ax, t));
    const double  rad = sqrt(rv.x * rv.x + rv.y * rv.y + rv.z * rv.z);
    if(t >= 0.0)
    {
        if(rad > r) return;
        atomicAdd(&red[3], 1.0);
        if(t <= gate) atomicAdd(&red[2], 1.0);
        return;
    }
    if(rad <= rc)
    {
        if(t >= -gate) atomicAdd(&red[1], 1.0);
    }
    else if(rad <= R1 && t >= -top)
        atomicAdd(&red[0], 1.0);
}
// the sink / source conveyor (see above). c_bore moves the particles in the bore (0 <= t <= L, rad <= r), c_slab the pool below
// the tip plane; the two differ only when one of the attach guards holds one of them back.
__global__ void k_pump_sink(int n, double3* xp, double3 a, double3 b, double r, double rc, double h, double R1, double R, double c_slab,
                            double c_bore, double3 v_tube, double follow, double dt)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    double3 ax = d3sub(b, a);
    const double L = sqrt(ax.x * ax.x + ax.y * ax.y + ax.z * ax.z);
    if(L < 1e-9) return;
    ax = d3scale(ax, 1.0 / L);
    const double3 d = d3sub(xp[p], a);
    const double  t = d.x * ax.x + d.y * ax.y + d.z * ax.z;
    if(t < -h || t > L) return;
    const double3 rv  = d3sub(d, d3scale(ax, t));
    const double  rad = sqrt(rv.x * rv.x + rv.y * rv.y + rv.z * rv.z);
    if(t >= 0.0)
    {   // bore: uniform conveyor, as k_pump<1>
        if(rad > r) return;
        double c = c_bore;
        if(c > 0.0 && t > L - 0.005) c = 0.0;              // do not convey into the cavity ceiling
        xp[p] = d3add(xp[p], d3add(d3scale(ax, c * dt), d3scale(v_tube, follow * dt)));
        return;
    }
    if(rad > R) return;
    double w = 0.0, u = 0.0;
    if(rad <= rc)
    {
        w = c_slab * (1.0 + t / h);
        u = -0.5 * c_slab * rad / h;
    }
    else
    {
        double s = rad <= R1 ? 1.0 : (R - rad) / (R - R1);
        s        = s * s * (3.0 - 2.0 * s);
        u        = -0.5 * c_slab * rc * rc / (h * rad) * s;
    }
    double3 dx = d3scale(ax, w * dt);
    if(rad > 1e-12) dx = d3add(dx, d3scale(rv, u * dt / rad));
    if(rad <= rc) dx = d3add(dx, d3scale(v_tube, follow * dt));   // the plain conveyor's intake column follows the tube too
    xp[p] = d3add(xp[p], dx);
}
// table stick: freeze particles that touch the bare table (see LiquidParams::stick_h)
__global__ void k_stick(int n, double3* xp, double3* vp, double* Cp, int* stuck, double3* xs, double ylo, double stick_h, int* count)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    if(stuck[p])
    {
        xp[p] = xs[p];
    }
    else if(xp[p].y < ylo + stick_h)
    {
        stuck[p] = 1;
        xs[p]    = xp[p];
        atomicAdd(count, 1);
    }
    else
        return;
    vp[p] = make_double3(0, 0, 0);
#pragma unroll
    for(int r = 0; r < 9; ++r) Cp[9 * p + r] = 0.0;
}
// ---- island sleep: connected components of the occupied cells (26-neighbour), label = min cell index ----
__global__ void k_isl_init(int na, const int* act, int* label)
{
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if(a >= na) return;
    label[act[a]] = act[a];
}
__global__ void k_isl_prop(int na, const int* act, int3 d, const int* cflag, int* label, int* changed)
{
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if(a >= na) return;
    int c = act[a], i, j, k;
    cell_ijk(c, d, i, j, k);
    int l = label[c];
    for(int di = -1; di <= 1; ++di)
        for(int dj = -1; dj <= 1; ++dj)
            for(int dk = -1; dk <= 1; ++dk)
            {
                int ii = i + di, jj = j + dj, kk = k + dk;
                if(ii < 0 || jj < 0 || kk < 0 || ii >= d.x || jj >= d.y || kk >= d.z) continue;
                int nb = cell_lin(ii, jj, kk, d);
                if(!cflag[nb]) continue;
                l = min(l, label[nb]);
            }
    if(l < label[c])
    {
        atomicMin(&label[c], l);
        *changed = 1;
    }
}
__global__ void k_isl_jump(int na, const int* act, int* label)
{
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if(a >= na) return;
    int c = act[a], l = label[c];
    for(int r = 0; r < 4; ++r)
    {
        int ll = label[l];
        if(ll >= l) break;
        l = ll;
    }
    if(l < label[c]) atomicMin(&label[c], l);
}
__global__ void k_isl_reset(int na, const int* act, int* cnt, int* miss, int* fast, int* minrest)
{
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if(a >= na) return;
    int c = act[a];
    cnt[c] = 0; miss[c] = 0; fast[c] = 0; minrest[c] = 0x7fffffff;
}
__global__ void k_isl_accum(int n, const double3* xold, double3 org, double dx, int3 d, const int* cflag, const int* label, const double* vnmag, const int* prest,
                            const int* stuck, double v_slow, double v_wake, int pump_on, double3 pa, double3 pb, double pr, int* pisl, int* cnt,
                            int* miss, int* fast, int* minrest)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    int c = particle_cell(xold[p], org, dx, d), r = cflag[c] ? label[c] : c;
    if(r < 0) r = c;
    pisl[p] = r;
    atomicAdd(&cnt[r], 1);
    atomicMin(&minrest[r], prest[p]);
    if(stuck && stuck[p]) return;   // frozen on the table anyway
    double v = vnmag[p];
    bool   pumped = false;
    if(pump_on)
    {
        double3 ax; double t, L, rad;
        pumped = pump_region(xold[p], pa, pb, pr, -0.010, ax, t, L, rad);
    }
    if(v >= v_slow) atomicAdd(&miss[r], 1);
    if(v > v_wake || pumped) atomicAdd(&fast[r], 1);
}
__global__ void k_isl_apply(int n, const int* pisl, const int* cnt, const int* miss, const int* fast, const int* minrest, double allow, int N,
                            const int* stuck, int* prest, double3* xp, const double3* xold, double* Jp, const double* Jold, double3* vp, double* Cp,
                            int* asleep)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    int  r = pisl[p];
    int  tol = max(1, (int)(allow * cnt[r]));
    bool slow = fast[r] == 0 && miss[r] <= tol;
    int  c = slow ? min(minrest[r] + 1, N + 1) : 0;    // one counter per body: a merged newcomer restarts the whole body
    prest[p] = c;
    if(c < N || (stuck && stuck[p])) return;
    xp[p] = xold[p];
    Jp[p] = Jold[p];
    vp[p] = make_double3(0, 0, 0);
#pragma unroll
    for(int q = 0; q < 9; ++q) Cp[9 * p + q] = 0.0;
    atomicAdd(asleep, 1);
}
__global__ void k_zero_push_asleep(int n, const int* sleep, int sleep_n, double3* push)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    if(sleep[p] >= sleep_n) push[p] = make_double3(0, 0, 0);
}
__global__ void k_count_asleep(int n, const int* sleep, int sleep_n, int* count)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    if(sleep[p] >= sleep_n) atomicAdd(count, 1);
}
__global__ void k_zero_push_stuck(int n, const int* stuck, double3* push)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    if(stuck[p]) push[p] = make_double3(0, 0, 0);
}
__global__ void k_reinit_J(int n, const double* rho, double rho_ref, double Jmin, double* Jp)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    double r = rho[p];
    double J = (r > 0.0) ? rho_ref / r : 1.0;
    if(!isfinite(J)) { LIQ_FBK(3); J = 1.0; }
    { double Jc = fmin(fmax(J, Jmin), 2.0); if(Jc != J) LIQ_FBK(6); Jp[p] = Jc; }
}

__global__ void k_count_nonfinite3(int n, const double3* a, int* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    if(!(isfinite(a[i].x) && isfinite(a[i].y) && isfinite(a[i].z))) atomicAdd(out, 1);
}
__global__ void k_count_nonfinite_blocks(int n, const Eigen::Matrix3d* v, const int* rows, const int* cols, int maxid, int* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    const Eigen::Matrix3d& m = v[i];
    bool bad = false;
    for(int a = 0; a < 3; ++a)
        for(int b = 0; b < 3; ++b) bad |= !isfinite(m(a, b));
    if(rows[i] < 0 || cols[i] < 0 || rows[i] >= maxid || cols[i] >= maxid || rows[i] > cols[i]) bad = true;
    if(bad) atomicAdd(out, 1);
}

__global__ void k_scale_rows(int n, const double3* src, double3* dst, int lo, int hi, double sc)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    double3 v = src[i];
    if(i >= lo && i < hi) v = make_double3(v.x * sc, v.y * sc, v.z * sc);
    dst[i] = v;
}

// [profile] LIQ_PROFILE=1: accumulate kernel time per phase and print every 200 calls
struct LiqProf
{
    // NO CUDA calls in the static constructor: a context created during the shared
    // library's static initialisation breaks the module's device-symbol registration
    // ("invalid device symbol" at the first cudaMemcpyToSymbol). Everything is lazy.
    int         state = -1;  // -1 unknown, 0 off, 1 on
    double      acc[6] = {0, 0, 0, 0, 0, 0};
    long long   calls = 0, steps = 0;   // steps: pre_step+post_step pairs in the current window
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    bool on()
    {
        if(state < 0)
        {
            state = getenv("LIQ_PROFILE") ? 1 : 0;
            if(state) { cudaEventCreate(&e0); cudaEventCreate(&e1); }
        }
        return state == 1;
    }
    void start() { if(on()) cudaEventRecord(e0); }
    void stop(int slot)
    {
        if(!on()) return;
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1); acc[slot] += ms;
    }
    void step_done() { if(on()) ++steps; }
    void tick()
    {
        if(!on()) return;
        if(++calls % 200 == 0)
        {
            // slot 5 is accumulated once per pre_step and once per post_step; it used to be printed
            // as the raw window sum under a "per step" label (about 15 frames' worth in the lab scene)
            printf("[liq-prof] per Newton iter (ms): assemble %.3f | chain: grad %.3f scan+count %.3f emit %.3f | energy %.3f | pre+post per step %.3f (%lld steps in window)\n",
                   acc[0] / 200, acc[1] / 200, acc[2] / 200, acc[3] / 200, acc[4] / 200,
                   steps > 0 ? acc[5] / (double)steps : 0.0, steps);
            for(double& a : acc) a = 0;
            steps = 0;
        }
    }
};
static LiqProf g_prof;

// ----------------------------------------------------------------------------
// host side
// ----------------------------------------------------------------------------
static inline int blocks_for(int n, int t = 256) { return (n + t - 1) / t; }

LiquidSystem::~LiquidSystem() { free_device(); }

void LiquidSystem::configure(const LiquidParams& p, const std::vector<double3>& x0, double pv)
{
    P               = p;
    P.tension_ratio = liquid_env("LIQ_TENSION", P.tension_ratio);
    P.vbar = liquid_env("LIQ_VOX_BAR", P.vbar);
    P.vbar_k = liquid_env("LIQ_VOX_BAR_K", P.vbar_k);
    P.invert_ok = liquid_env("LIQ_VOX_INVERT_OK", P.invert_ok);
    P.flip = liquid_env("LIQ_FLIP", P.flip);
    P.bdf2 = liquid_env("LIQ_BDF2", P.bdf2);
    P.bdf2_wall_be = liquid_env("LIQ_BDF2_WALL_BE", P.bdf2_wall_be);
    P.st_gamma = liquid_env("LIQ_ST", P.st_gamma);
    P.rest_damp = liquid_env("LIQ_REST_DAMP", P.rest_damp);
    P.rest_v = liquid_env("LIQ_REST_V", P.rest_v);
    P.sleep_v = liquid_env("LIQ_SLEEP_V", P.sleep_v);
    P.sleep_n = liquid_env("LIQ_SLEEP_N", P.sleep_n);
    P.wake_v = liquid_env("LIQ_WAKE_V", P.wake_v);
    P.rest_k = liquid_env("LIQ_REST_K", P.rest_k);
    P.isl_v = liquid_env("LIQ_ISL_V", P.isl_v);
    P.isl_wake = liquid_env("LIQ_ISL_WAKE", P.isl_wake);
    P.isl_t = liquid_env("LIQ_ISL_T", P.isl_t);
    P.isl_allow = liquid_env("LIQ_ISL_ALLOW", P.isl_allow);
    P.rest_kv = liquid_env("LIQ_REST_KV", P.rest_kv);
    P.st_c = liquid_env("LIQ_ST_C", P.st_c);
    P.st_thin = liquid_env("LIQ_ST_THIN", P.st_thin);
    P.sti_gamma = liquid_env("LIQ_STI", P.sti_gamma);
    P.sti_r0 = liquid_env("LIQ_STI_R0", P.sti_r0);
    P.sti_rc = liquid_env("LIQ_STI_RC", P.sti_rc);
    P.sti_freeze = liquid_env("LIQ_STI_FREEZE", P.sti_freeze);
    P.stm_sigma = liquid_env("LIQ_STM", P.stm_sigma);
    P.stm_area = liquid_env("LIQ_STM_AREA", P.stm_area);
    P.stm_thr = liquid_env("LIQ_STM_THR", P.stm_thr);
    P.stm_amode = liquid_env("LIQ_STM_AMODE", P.stm_amode);
    P.stm_amax = liquid_env("LIQ_STM_AMAX", P.stm_amax);
    P.reanchor_beta = liquid_env("LIQ_REANCHOR_BETA", P.reanchor_beta);
    P.conv_scale = liquid_env("LIQ_CONV_SCALE", P.conv_scale);
    liquid_require(liquid_env("LIQ_PARTICLE_QUAD", 0) == 0,
                   "LIQ_PARTICLE_QUAD prototype not physically validated; use the archived experimental build only");
    liquid_require(liquid_env("LIQ_GHOST_EXT", 0.0) == 0.0,
                   "LIQ_GHOST_EXT prototype rejected; use the archived experimental build only");
    P.push_surf_ratio = liquid_env("LIQ_PUSH_SURF", P.push_surf_ratio);
    P.push_film = liquid_env("LIQ_PUSH_FILM", P.push_film);
    P.push_film_r = liquid_env("LIQ_PUSH_FILM_R", P.push_film_r);
    P.J_track_max = liquid_env("LIQ_JMAX", P.J_track_max);
    P.fill_dilate = liquid_env("LIQ_FILL_DILATE", P.fill_dilate);
    P.fill_erode = liquid_env("LIQ_FILL_ERODE", P.fill_erode);
    P.fill_own = liquid_env("LIQ_FILL_OWN", P.fill_own);
    P.stick_h = liquid_env("LIQ_STICK_H", P.stick_h);
    P.floor_gravity = liquid_env("LIQ_FLOOR_GRAVITY", P.floor_gravity);
    P.ghost_floor = liquid_env("LIQ_GHOST_FLOOR", P.ghost_floor);
    P.ghost_frac = liquid_env("LIQ_GHOST_FRAC", P.ghost_frac);
    P.wall_bc = liquid_env("LIQ_WALL_BC", P.wall_bc);
    P.wfc = liquid_env("LIQ_WFC", P.wfc);
    P.wfc_pre = (int)liquid_env("LIQ_WFC_PRE", (double)P.wfc_pre);
    P.wfc_al = (int)liquid_env("LIQ_WFC_AL", (double)P.wfc_al);
    liquid_require(P.wfc_al == 0 || P.wfc_al == 1, "LIQ_WFC_AL must be 0 or 1");
    liquid_require(P.wfc_pre >= 0 && P.wfc_pre <= 3, "LIQ_WFC_PRE must be 0 (off), 1 (previous step or predictor), 2 (predictor) or 3 (previous step)");
    P.wfc_ground = liquid_env("LIQ_WFC_GROUND", P.wfc_ground);
    P.cpic = liquid_env("LIQ_CPIC", P.cpic);
    P.cpic_r = liquid_env("LIQ_CPIC_R", P.cpic_r);
    P.cpic_ghost = liquid_env("LIQ_CPIC_GHOST", P.cpic_ghost);
    P.cpic_p2g = liquid_env("LIQ_CPIC_P2G", P.cpic_p2g);
    P.cpic_c = liquid_env("LIQ_CPIC_C", P.cpic_c);
    P.cpic_cell = liquid_env("LIQ_CPIC_CELL", P.cpic_cell);
    P.cpic_split = liquid_env("LIQ_CPIC_SPLIT", P.cpic_split);
    P.cpic_side1 = liquid_env("LIQ_CPIC_SIDE1", P.cpic_side1);
    P.cpic_fields = liquid_env("LIQ_CPIC_FIELDS", P.cpic_fields);
    liquid_require(P.cpic_fields == 1 || P.cpic_fields == 2, "LIQ_CPIC_FIELDS must be 1 or 2");
    P.cpic_pghost = liquid_env("LIQ_CPIC_PGHOST", P.cpic_pghost);
    P.wfc_k = liquid_env("LIQ_WFC_K", P.wfc_k);
    P.wfc_r = liquid_env("LIQ_WFC_R", P.wfc_r);
    P.gp = liquid_env("LIQ_GP", P.gp);
    P.gp_full = liquid_env("LIQ_GP_FULL", P.gp_full);
    P.gp_vlo = liquid_env("LIQ_GP_VLO", P.gp_vlo);
    P.gp_vhi = liquid_env("LIQ_GP_VHI", P.gp_vhi);
    P.gp_rel = liquid_env("LIQ_GP_REL", P.gp_rel);
    P.cutcell = liquid_env("LIQ_CUTCELL", P.cutcell);
    P.cutcell_vmin = liquid_env("LIQ_CUTCELL_VMIN", P.cutcell_vmin);
    P.wb = liquid_env("LIQ_WB", P.wb);
    P.wb_gate = liquid_env("LIQ_WB_GATE", P.wb_gate);
    P.wb_slip = liquid_env("LIQ_WB_SLIP", P.wb_slip);
    P.wb_slip_phi = liquid_env("LIQ_WB_SLIP_PHI", P.wb_slip_phi);
    P.wb_slip_phi_mov = liquid_env("LIQ_WB_SLIP_PHI_MOVING", P.wb_slip_phi_mov);
    P.wb_moving_eps = liquid_env("LIQ_WB_MOVING_EPS", P.wb_moving_eps);
    P.wb_ghost = liquid_env("LIQ_WB_GHOST", P.wb_ghost);
    P.wb_wmin = liquid_env("LIQ_WB_WMIN", P.wb_wmin);
    P.wb_mfloor = liquid_env("LIQ_WB_MFLOOR", P.wb_mfloor);
    P.wb_aggr = liquid_env("LIQ_WB_AGGR", P.wb_aggr);
    P.wb_aggr_depth = liquid_env("LIQ_WB_AGGR_DEPTH", P.wb_aggr_depth);
    P.wb_aggr_wall = liquid_env("LIQ_WB_AGGR_WALL", P.wb_aggr_wall);
    P.wb_aggr_ghost = liquid_env("LIQ_WB_AGGR_GHOST", P.wb_aggr_ghost);
    P.wb_aggr_debug = liquid_env("LIQ_WB_AGGR_DEBUG", P.wb_aggr_debug);
    P.wb_lat = liquid_env("LIQ_WB_LAT", P.wb_lat);
    P.wb_hg = liquid_env("LIQ_WB_HG", P.wb_hg);
    P.pdev = liquid_env("LIQ_PDEV", P.pdev);
    P.wb_det = liquid_env("LIQ_WB_DET", P.wb_det);
    P.wb_surfk = liquid_env("LIQ_WB_SURFK", P.wb_surfk);
    P.wb_hg_wallonly = liquid_env("LIQ_WB_HG_WALLONLY", P.wb_hg_wallonly);
    P.wb_topface = liquid_env("LIQ_WB_TOPFACE", P.wb_topface);
    P.visc = liquid_env("LIQ_VISC", P.visc);
    P.wb_g2p = liquid_env("LIQ_WB_G2P", P.wb_g2p);
    P.wb_phi_euclid = liquid_env("LIQ_WB_PHI_EUCLID", P.wb_phi_euclid);
    P.wb_hole_lat = liquid_env("LIQ_WB_HOLE_LAT", P.wb_hole_lat);
    P.wb_phi_sign = (int)liquid_env("LIQ_WB_PHI_SIGN", (double)P.wb_phi_sign);
    P.jfilt = liquid_env("LIQ_JFILT", P.jfilt);
    P.wb_airgrav = (int)liquid_env("LIQ_WB_AIRGRAV", (double)P.wb_airgrav);
    P.g2p_fluid = (int)liquid_env("LIQ_G2P_FLUID", (double)P.g2p_fluid);
    P.wb_geov = (int)liquid_env("LIQ_WB_GEOV", (double)P.wb_geov);
    P.wb_geov_fill = liquid_env("LIQ_WB_GEOV_FILL", P.wb_geov_fill);
    P.wb_phi_tri = (int)liquid_env("LIQ_WB_PHI_TRI", (double)P.wb_phi_tri);
    P.g2p_jcell = (int)liquid_env("LIQ_G2P_JCELL", (double)P.g2p_jcell);
    P.wb_ybar = (int)liquid_env("LIQ_WB_YBAR", (double)P.wb_ybar);
    P.wb_g2p_fmin = liquid_env("LIQ_WB_G2P_FMIN", P.wb_g2p_fmin);
    P.wb_snap = liquid_env("LIQ_WB_SNAP", P.wb_snap);
    P.pdev_all = liquid_env("LIQ_PDEV_ALL", P.pdev_all);
    P.wb_near = liquid_env("LIQ_WB_NEAR", P.wb_near);
    P.wb_topk = liquid_env("LIQ_WB_TOPK", P.wb_topk);
    P.wb_phi_dtol = liquid_env("LIQ_WB_PHI_DTOL", P.wb_phi_dtol);
    P.wb_colmin = liquid_env("LIQ_WB_COLMIN", P.wb_colmin);
    P.wb_fullf = (int)liquid_env("LIQ_WB_FULLF", (double)P.wb_fullf);
    P.wb_phi_freeze = (int)liquid_env("LIQ_WB_PHI_FREEZE", (double)P.wb_phi_freeze);
    P.wb_hcol_freeze = (int)liquid_env("LIQ_WB_HCOL_FREEZE", (double)P.wb_hcol_freeze);
    P.wb_cap = liquid_env("LIQ_WB_CAP", P.wb_cap);
    P.wb_phi_ghost = liquid_env("LIQ_WB_PHI_GHOST", P.wb_phi_ghost);
    P.wb_thin = liquid_env("LIQ_WB_THIN", P.wb_thin);
    P.wb_vis_r = liquid_env("LIQ_WB_VIS_R", P.wb_vis_r);
    P.wb_phi_ghost_side = (int)liquid_env("LIQ_WB_PHI_GHOST_SIDE", (double)P.wb_phi_ghost_side);
    P.wb_ovh = liquid_env("LIQ_WB_OVH", P.wb_ovh);
    P.wb_ovh_cos = liquid_env("LIQ_WB_OVH_COS", P.wb_ovh_cos);
    P.wb_hsmooth_w = (int)liquid_env("LIQ_WB_HSMOOTH_W", (double)P.wb_hsmooth_w);
    P.wb_colmin_pool = (int)liquid_env("LIQ_WB_COLMIN_POOL", (double)P.wb_colmin_pool);
    P.wb_hmode = (int)liquid_env("LIQ_WB_HMODE", (double)P.wb_hmode);
    P.wb_hwin = liquid_env("LIQ_WB_HWIN", P.wb_hwin);
    liquid_require(P.wb_hmode >= 0 && P.wb_hmode <= 3, "LIQ_WB_HMODE must be 0 (top-k), 1 (per-column particles per layer), 2 (top-layer window) or 3 (soft window)");
    P.wb_surf = liquid_env("LIQ_WB_SURF", P.wb_surf);
    P.wb_hsmooth = liquid_env("LIQ_WB_HSMOOTH", P.wb_hsmooth);
    P.gm = liquid_env("LIQ_GHOST_MIRROR", P.gm);
    P.gm_k = liquid_env("LIQ_GHOST_MIRROR_K", P.gm_k);
    P.wall_mass = liquid_env("LIQ_WALL_MASS", P.wall_mass);
    P.wall_static_mass = liquid_env("LIQ_WALL_STATIC_MASS", P.wall_static_mass);
    P.wall_static_mode = (int)liquid_env("LIQ_WALL_STATIC_MODE", (double)P.wall_static_mode);
    P.wall_static_normal = (int)liquid_env("LIQ_WALL_STATIC_NORMAL", (double)P.wall_static_normal);
    P.wall_tang = liquid_env("LIQ_WALL_TANG", P.wall_tang);
    P.wb_nrm_exact = liquid_env("LIQ_WB_NRM_EXACT", P.wb_nrm_exact);
    P.wb_trim = liquid_env("LIQ_WB_TRIM", P.wb_trim);
    P.ground_bc = liquid_env("LIQ_GROUND_BC", P.ground_bc);
    if(getenv("LIQ_TEST_CAVITY"))
    {
        P.cav_on = sscanf(getenv("LIQ_TEST_CAVITY"), "%lf,%lf,%lf,%lf,%lf,%lf,%lf", &P.cav[0], &P.cav[1], &P.cav[2], &P.cav[3], &P.cav[4], &P.cav[5], &P.cav[6]) == 7;
        printf("[liquid] TEST cavity grid BC %s: axis (%.4f, %.4f) floor %.4f r %.4f->%.4f over y %.4f..%.4f\n", P.cav_on ? "ON" : "PARSE FAILED",
               P.cav[0], P.cav[1], P.cav[2], P.cav[3], P.cav[5], P.cav[4], P.cav[6]);
    }
    liquid_require(finite_positive(P.dx) && finite_positive(P.rho) && finite_positive(P.kappa)
                       && finite_positive(pv) && finite_positive(P.rho * pv), "invalid liquid scale, material or particle volume");
    liquid_require(!x0.empty() && x0.size() <= std::numeric_limits<int>::max() / 27,
                   "invalid particle count for 32-bit stencils");
    liquid_require(P.gp_full == 0 || P.gp_full == 1 || P.gp_full == 2 || P.gp_full == 3 || P.gp_full == 6 || P.gp_full == 7,
                   "LIQ_GP_FULL must be 0, 1, 2, 3 or 6 (4 and 5 have inconsistent derivatives)");
    liquid_require(P.j_mode >= 0 && P.j_mode <= 4, "j_mode must be 0, 1, 2, 3 or 4");
    for(double v : {P.tension_ratio, P.st_gamma, P.rest_damp, P.sleep_v, P.isl_v, P.rest_k,
                    P.push_surf_ratio, P.fill_dilate, P.fill_erode, P.fill_own, P.stick_h,
                    P.ghost_floor, P.gp, P.mass_floor, P.push_apart})
        liquid_require(finite_nonnegative(v), "liquid strengths, radii and sleep thresholds must be finite and nonnegative");
    for(double v : {P.rest_v, P.rest_kv, P.isl_t, P.st_c, P.conv_scale, P.wfc_k, P.wfc_r, P.wall_mass})
        liquid_require(finite_positive(v), "liquid velocity scales, duration, support and penalty scales must be positive");
    for(double v : {P.flip, P.reanchor_beta, P.ghost_frac, P.isl_allow, P.apic_ratio})
        liquid_require(std::isfinite(v) && v >= 0 && v <= 1, "liquid blend fractions must be in [0, 1]");
    liquid_require(P.sleep_n > 0 && std::isfinite(P.wake_v) && P.wake_v >= P.sleep_v
                       && std::isfinite(P.isl_wake) && P.isl_wake >= P.isl_v,
                   "sleep duration must be positive and wake thresholds cannot be below sleep thresholds");
    liquid_require(finite_positive(P.J_track_min) && std::isfinite(P.J_track_max) && P.J_track_max >= P.J_track_min,
                   "invalid tracked-J interval");
    liquid_require(P.invert_ok == 0 || P.invert_ok == 1, "LIQ_VOX_INVERT_OK must be 0 or 1");
    liquid_require(finite_nonnegative(P.vbar) && P.vbar < 1.0 && std::isfinite(P.vbar_k) && P.vbar_k > 0.0,
                   "LIQ_VOX_BAR must be in [0, 1) and LIQ_VOX_BAR_K > 0");
    liquid_require(finite_nonnegative(P.gp_vlo) && std::isfinite(P.gp_vhi) && P.gp_vhi > P.gp_vlo,
                   "LIQ_GP_VHI must exceed nonnegative LIQ_GP_VLO");
    liquid_require(P.st_c <= 2.0, "LIQ_ST_C cannot exceed 2 (the cohesion search spans at most 5x5x5 cells)");
    liquid_require(finite_nonnegative(P.sti_gamma), "LIQ_STI must be finite and nonnegative (N/m)");
    liquid_require(finite_nonnegative(P.stm_sigma) && finite_positive(P.stm_area) && finite_positive(P.stm_thr),
                   "LIQ_STM must be finite and nonnegative, LIQ_STM_AREA and LIQ_STM_THR positive");
    liquid_require(P.stm_amode == 0 || P.stm_amode == 1, "LIQ_STM_AMODE must be 0 (fixed element area) or 1 (calibrated)");
    liquid_require(P.bdf2 == 0 || P.bdf2 == 1, "LIQ_BDF2 must be 0 or 1");
    liquid_require(P.bdf2_wall_be >= 0 && P.bdf2_wall_be <= 2, "LIQ_BDF2_WALL_BE must be 0, 1 or 2");
    liquid_require(P.cutcell == 0 || P.cutcell == 1, "LIQ_CUTCELL must be 0 or 1");
    liquid_require(finite_positive(P.cutcell_vmin) && P.cutcell_vmin <= 1.0, "LIQ_CUTCELL_VMIN must be in (0, 1]");
    liquid_require(!P.cutcell || (P.wfc && !P.cpic), "LIQ_CUTCELL needs LIQ_WFC=1 (its wall points) and LIQ_CPIC=0");
    liquid_require(P.wb == 0 || P.wb == 1, "LIQ_WB must be 0 or 1");
    liquid_require(!P.wb || (P.wfc && !P.cpic && !P.cutcell), "LIQ_WB needs LIQ_WFC=1 (its wall points), LIQ_CPIC=0 and LIQ_CUTCELL=0");
    liquid_require(finite_positive(P.wb_gate) && P.wb_gate <= 1.0, "LIQ_WB_GATE must be in (0, 1]");
    liquid_require(P.wb_slip == 0 || P.wb_slip == 1, "LIQ_WB_SLIP must be 0 or 1");
    liquid_require(finite_nonnegative(P.wb_mfloor) && P.wb_mfloor <= 10.0, "LIQ_WB_MFLOOR in [0, 10]");
    liquid_require(finite_nonnegative(P.wb_lat) && P.wb_lat <= 4.0, "LIQ_WB_LAT in [0, 4]");
    liquid_require(finite_nonnegative(P.wb_hg) && P.wb_hg <= 100.0, "LIQ_WB_HG in [0, 100]");
    liquid_require(finite_nonnegative(P.pdev) && P.pdev <= 100.0, "LIQ_PDEV in [0, 100]");
    liquid_require(finite_nonnegative(P.visc) && P.visc <= 100.0, "LIQ_VISC in [0, 100] Pa s");
    liquid_require(P.wb_phi_sign >= 0 && P.wb_phi_sign <= 11, "LIQ_WB_PHI_SIGN must be 0 (nearest wall point), 1 (any containing triangle), 2 (nearest containing triangle), 3 (any wall point plane, sti48) , 4 (3 with a lateral gate LIQ_WB_HOLE_LAT on the behind decision) , 5 (visibility: the segment to the nearest particle crosses a solid), 6 (3, counting only wall points whose particle is seen through a solid), 7 (visibility OR 6, magnitude from the mesh), 8 (rule 3 + unseen nodes behind + distance capped at exact + ghost), 9 (8 with the planes filtered by water connectivity), 10 (8 with planes counted only over their own triangle) or 11 (8 with the plane dead zone only at thin walls)");
    liquid_require(finite_nonnegative(P.jfilt) && P.jfilt <= 1.0, "LIQ_JFILT in [0, 1]");
    liquid_require(finite_nonnegative(P.wb_surfk) && P.wb_surfk <= 100.0 && finite_nonnegative(P.wb_snap) && P.wb_snap < 0.5, "LIQ_WB_SURFK in [0, 100], LIQ_WB_SNAP in [0, 0.5)");
    liquid_require(finite_nonnegative(P.wb_aggr) && P.wb_aggr < 1.0 && P.wb_aggr_depth >= 1 && P.wb_aggr_depth <= 6, "LIQ_WB_AGGR in [0, 1), LIQ_WB_AGGR_DEPTH in [1, 6]");
    liquid_require(finite_nonnegative(P.wb_wmin) && P.wb_wmin < 1.0 && finite_nonnegative(P.wb_near) && P.wb_near <= 1.0, "LIQ_WB_WMIN in [0, 1), LIQ_WB_NEAR in [0, 1]");
    liquid_require(P.wb_ghost >= 1 && P.wb_ghost <= 3, "LIQ_WB_GHOST must be 1 (slip), 2 (behind fixed, on-wall slip) or 3 (all wall nodes fixed; diagnostic)");
    liquid_require(std::isfinite(P.wb_slip_phi) && P.wb_slip_phi < 1.0, "LIQ_WB_SLIP_PHI must be < 1.0 (lattice units)");
    liquid_require(std::isfinite(P.wall_tang) && P.wall_tang >= 0.0 && P.wall_tang <= 1.0, "LIQ_WALL_TANG must be in [0, 1]");
    liquid_require(P.wall_static_normal == 0 || P.wall_static_normal == 1, "LIQ_WALL_STATIC_NORMAL must be 0 or 1");
    liquid_require(!(P.wb && P.wb_slip && P.bdf2) || P.bdf2_wall_be != 0, "LIQ_WB slip nodes need LIQ_BDF2_WALL_BE >= 1 under LIQ_BDF2 (they are implicit-Euler nodes)");
    P.ug = (int)liquid_env("LIQ_UG", (double)P.ug);
    P.ug_jact = liquid_env("LIQ_UG_JACT", P.ug_jact);
    P.ug_kb = liquid_env("LIQ_UG_KB", P.ug_kb);
    P.ug_mfloor = liquid_env("LIQ_UG_MFLOOR", P.ug_mfloor);
    P.ug_bspline = (int)liquid_env("LIQ_UG_BSPLINE", (double)P.ug_bspline);
    P.ug_reanchor = liquid_env("LIQ_UG_REANCHOR", P.ug_reanchor);
    P.ug_feasible = liquid_env("LIQ_UG_FEASIBLE", P.ug_feasible);
    P.ug_hg       = liquid_env("LIQ_UG_HG", P.ug_hg);
    liquid_require(finite_nonnegative(P.ug_hg), "LIQ_UG_HG must be a nonnegative viscosity (Pa s)");
    P.ug_rep      = liquid_env("LIQ_UG_REP", P.ug_rep);
    P.ug_rep_r0   = liquid_env("LIQ_UG_REP_R0", P.ug_rep_r0);
    liquid_require(finite_nonnegative(P.ug_rep) && finite_positive(P.ug_rep_r0) && P.ug_rep_r0 < 1.0, "LIQ_UG_REP must be nonnegative and LIQ_UG_REP_R0 in (0, 1) seed spacings");
    liquid_require(P.ug == 0 || P.ug == 1, "LIQ_UG must be 0 or 1");
    liquid_require(P.ug_bspline == 0 || P.ug_bspline == 1, "LIQ_UG_BSPLINE must be 0 or 1");
    liquid_require(finite_nonnegative(P.ug_reanchor) && P.ug_reanchor <= 1.0, "LIQ_UG_REANCHOR must be in [0, 1]");
    liquid_require(finite_nonnegative(P.ug_feasible) && P.ug_feasible <= 1.0, "LIQ_UG_FEASIBLE must be in [0, 1]");
    liquid_require(finite_nonnegative(P.ug_jact) && P.ug_jact < 1.0 && finite_positive(P.ug_kb) && finite_nonnegative(P.ug_mfloor),
                   "LIQ_UG_JACT must be in [0, 1), LIQ_UG_KB positive, LIQ_UG_MFLOOR nonnegative");
    if(P.ug)
    {   // the recovered engine's liquid: no grid wall BC, no rest / regularisation layers; only the particle barrier meets the solids
        P.wfc = P.wfc_ground = P.gp = 0; P.cpic = 0; P.wb = 0; P.cutcell = 0; P.wall_bc = P.ground_bc = 0;
        P.rest_damp = 0.0; P.sleep_v = 0.0; P.isl_v = 0.0; P.push_film = 0;   // LIQ_REST_K / LIQ_REST_KV (in-solve rest drag of slow nodes, k_write_node_slots) stay available in UG mode
        P.push_apart = liquid_env("LIQ_UG_PUSH", 0.0);   // explicit pre-solve push-apart (main-path push_apart_pass) as the cheap sub-cell density control of UG mode; 0 = off
        P.st_gamma = 0.0; P.sti_gamma = 0.0; P.stm_sigma = 0.0; P.stick_h = 0.0; P.ghost_floor = 0.0; P.jfilt = 0.0;
        P.vbar = 0.0; P.visc = 0.0; P.pdev = 0.0; P.flip = 0.0; P.j_mode = 0; P.gm = 0;   // LIQ_BDF2 stays available (grid-node BDF2, no wall arrays in UG)
        P.g2p_jcell = 0; P.g2p_fluid = 0; P.wb_g2p = 0; P.wb_aggr = 0.0;
        P.mass_floor = P.ug_mfloor;
        P.apic = 1;
    }
    liquid_require(P.gm == 0 || P.gm == 1, "LIQ_GHOST_MIRROR must be 0 or 1");
    liquid_require(P.gm == 0 || liquid_env("LIQ_GHOST_MIRROR_RESEARCH", 0) == 1,
                   "LIQ_GHOST_MIRROR rejected 2026-10-23 (still water worse: flask 1.1 mm/s, tanks 6-14 mm/s; WATER_REALISM_REVIEW_2026-10-23.md); "
                   "set LIQ_GHOST_MIRROR_RESEARCH=1 only to reproduce that experiment");
    liquid_require(P.cutcell == 0 || liquid_env("LIQ_CUTCELL_RESEARCH", 0) == 1,
                   "LIQ_CUTCELL is an unfinished prototype (first version churns still water; LIQUID_CUTCELL_DESIGN_2026-10-23.md); "
                   "set LIQ_CUTCELL_RESEARCH=1 to run it");
    liquid_require(finite_positive(P.gm_k), "LIQ_GHOST_MIRROR_K must be positive");
    liquid_require(!P.gm || (P.wfc && !P.cpic), "LIQ_GHOST_MIRROR needs LIQ_WFC=1 (its wall points) and LIQ_CPIC=0");
    liquid_require(!P.bdf2 || (!P.cpic && P.flip == 0.0 && P.rest_k == 0.0),
                   "LIQ_BDF2 needs LIQ_CPIC 0, LIQ_FLIP 0 and LIQ_REST_K 0 (their node / particle updates are implicit-Euler only)");
    liquid_require(finite_positive(P.stm_amax), "LIQ_STM_AMAX must be positive");
    liquid_require(P.stm_sigma == 0.0 || 2.0 * std::cbrt(pv) <= P.dx * (1.0 + 1e-9),
                   "LIQ_STM needs 2 h <= dx (8 or more particles per cell): its neighbourhood search covers one cell");
    liquid_require(finite_positive(P.sti_r0) && std::isfinite(P.sti_rc) && P.sti_rc > P.sti_r0, "LIQ_STI_RC must exceed LIQ_STI_R0 > 0");
    liquid_require(P.sti_gamma == 0.0 || (P.sti_rc + 0.25) * std::cbrt(pv) <= P.dx,
                   "LIQ_STI_RC + 0.25 (list margin) seed spacings must fit in one cell (the pair search spans 3x3x3 cells)");
    liquid_require(P.push_film >= 0 && P.push_film <= 2, "LIQ_PUSH_FILM must be 0, 1 or 2");
    liquid_require(P.push_film_r >= 0.0 && P.push_film_r <= 2.0, "LIQ_PUSH_FILM_R must be in [0, 2] (the push search spans one cell)");
    liquid_require(finite_positive(P.cpic_r), "LIQ_CPIC_R must be positive (cells)");
    for(int v : {P.floor_gravity, P.wall_bc, P.ground_bc, P.wfc, P.apic, P.cpic, P.cpic_ghost, P.cpic_p2g, P.cpic_c, P.cpic_cell, P.cpic_split, P.cpic_side1, P.cpic_pghost})
        liquid_require(v == 0 || v == 1, "liquid boolean options must be 0 or 1");
    n_cells = checked_grid_count(P.dims.x, P.dims.y, P.dims.z);
    n_nodes = checked_grid_count(P.dims.x, P.dims.y, P.dims.z, 1);
    n_vnodes = (P.cpic && P.cpic_fields == 2) ? 2 * n_nodes : n_nodes;
    if(P.j_mode == 4)
        liquid_require(n_cells <= std::numeric_limits<int>::max() / 64, "fill reconstruction exceeds 32-bit sub-grid indexing");
    h_x0            = x0;
    n_particles     = (int)x0.size();
    particle_volume = pv;
}

int LiquidSystem::estimate_seed_nodes() const
{
    std::vector<char> cell(n_cells, 0);
    for(const auto& x : h_x0)
    {
        double r[3] = {(x.x - P.origin.x) / P.dx - 0.5, (x.y - P.origin.y) / P.dx - 0.5, (x.z - P.origin.z) / P.dx - 0.5};
        int    n[3] = {P.dims.x, P.dims.y, P.dims.z};
        int    b[3];
        for(int a = 0; a < 3; ++a)
        {
            b[a] = (int)std::floor(r[a]);
            b[a] = std::max(0, std::min(n[a] - 2, b[a]));
        }
        for(int a = 0; a < 2; ++a)
            for(int bb = 0; bb < 2; ++bb)
                for(int c = 0; c < 2; ++c) cell[cell_lin(b[0] + a, b[1] + bb, b[2] + c, P.dims)] = 1;
    }
    std::vector<char> node(n_nodes, 0);
    int               cnt = 0;
    for(int c = 0; c < n_cells; ++c)
    {
        if(!cell[c]) continue;
        int i, j, k;
        cell_ijk(c, P.dims, i, j, k);
        for(int q = 0; q < 8; ++q)
        {
            int nid = node_lin(i + ((q >> 2) & 1), j + ((q >> 1) & 1), k + (q & 1), P.dims);
            if(!node[nid])
            {
                node[nid] = 1;
                ++cnt;
            }
        }
    }
    return cnt;
}

void LiquidSystem::alloc_device()
{
    free_device();
    {
        int tri = P.ug ? (P.ug_bspline ? 2 : 1) : 0;   // LIQ_UG: trilinear (1) / quadratic B-spline (2) embedding for every node_stencil user (contact chain rule, particle slots)
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_liq_tri, &tri, sizeof(int)));
    }
    const int n = n_particles;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_xp, n * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_vp, n * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_Cp, 9 * n * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_Jp, n * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_mp, n * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_Vp0, n * sizeof(double)));
    CUDA_SAFE_CALL(cudaMemcpy(d_xp, h_x0.data(), n * sizeof(double3), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemset(d_vp, 0, n * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMemset(d_Cp, 0, 9 * n * sizeof(double)));
    if(P.bdf2)
    {
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_bh_vp, n * sizeof(double3)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_bh_Cp, 9 * (size_t)n * sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_bh_mvc, (size_t)n_cells * sizeof(double3)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_bh_Gc, 9 * (size_t)n_cells * sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_bh_mvi, (size_t)n_vnodes * sizeof(double3)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_bh_vel, (size_t)n_vnodes * sizeof(double3)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_bh_vb, (size_t)std::max(1, P.node_pool) * sizeof(double3)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_bh_xt, (size_t)std::max(1, P.node_pool) * sizeof(double3)));
        CUDA_SAFE_CALL(cudaMemset(d_bh_vel, 0, (size_t)n_vnodes * sizeof(double3)));
        bh_init = false;
        printf("[liquid] BDF2 (velocity form) time integration for the grid nodes: inertia x 9/4 around x + dt (19/18 v - 1/18 v_prev) + 4/9 dt^2 g\n");
    }
    std::vector<double> ones(n, 1.0), mp(n, P.rho * particle_volume), vp0(n, particle_volume);
    CUDA_SAFE_CALL(cudaMemcpy(d_Jp, ones.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_mp, mp.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_Vp0, vp0.data(), n * sizeof(double), cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMalloc((void**)&d_mc, n_cells * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_mvc, n_cells * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_Gc, 9 * n_cells * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_Vc, n_cells * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_piVc, n_cells * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_Jbc, n_cells * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_cell_nabs, n_cells * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_cell_axis, n_cells * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_corner_vote, 32 * (size_t)n_cells * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_mc2, n_cells * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_Vc2, n_cells * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_piVc2, n_cells * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_Jbc2, n_cells * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_elem_cnt, n_cells * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_elem_off, n_cells * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_ghost_cnt, n_cells * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_ghost_off, n_cells * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(d_cell_axis, 0xff, n_cells * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_cell_flag, n_cells * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_cell_scan, n_cells * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_active_cells, n_cells * sizeof(int)));

    CUDA_SAFE_CALL(cudaMalloc((void**)&d_mi, n_vnodes * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_mci, n_vnodes * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_mcc, n_cells * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_mvi, n_vnodes * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_vi, n_vnodes * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_vi_old, n_vnodes * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_st_n, n_particles * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_node_flag, n_vnodes * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_node_ref, n_vnodes * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_node_ghost, n_vnodes * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(d_node_ghost, 0, n_vnodes * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_node_scan, n_vnodes * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_node_slot, n_vnodes * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_active_nodes, n_vnodes * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_counts, 2 * sizeof(int)));

    CUDA_SAFE_CALL(cudaMalloc((void**)&d_vox_nodes, 8 * (size_t)max_voxels * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_vox_V, max_voxels * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_vox_Jb, max_voxels * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_vox_ijk, max_voxels * sizeof(int3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_ghost_corner, (size_t)std::max(1, P.node_pool) * sizeof(int3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_slot_used, (size_t)std::max(1, P.node_pool) * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_reduce, (blocks_for(max_voxels) + 1) * sizeof(double)));
    CUDA_SAFE_CALL(cudaMemset(d_node_flag, 0, n_vnodes * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(d_vi, 0, n_vnodes * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMemset(d_vi_old, 0, n_vnodes * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_nref_key, n_nodes * sizeof(unsigned long long)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_nref_has, n_nodes * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_nref_X, n_nodes * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_nref_n, n_nodes * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMemset(d_nref_has, 0, n_nodes * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_corner_side, 16 * (size_t)n_cells * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_plabel, n * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_nlab, n_nodes * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(d_nlab, 0, n_nodes * sizeof(int)));

    CUDA_SAFE_CALL(cudaMalloc((void**)&d_pkey, n * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_pidx, n * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_cstart, n_cells * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_cend, n_cells * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_push, n * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_rho, n * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_pbase, n * sizeof(int3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_pfrac, n * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_Hpp, 9 * (size_t)n * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_pflag, n * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_counters, 4 * sizeof(int)));   // [cross blocks, emit cursor, LIQ_WFC_PRE count, spare]
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_fallback, 12 * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(d_fallback, 0, 12 * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_pmask, n * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_pwinv, n * sizeof(double)));
    { std::vector<int> all(n, LIQ_MASK_ALL); std::vector<double> one(n, 1.0);
      CUDA_SAFE_CALL(cudaMemcpy(d_pmask, all.data(), n * sizeof(int), cudaMemcpyHostToDevice));
      CUDA_SAFE_CALL(cudaMemcpy(d_pwinv, one.data(), n * sizeof(double), cudaMemcpyHostToDevice)); }
    { int* fp = d_fallback; CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_liq_fbk, &fp, sizeof(int*))); }
    CUDA_SAFE_CALL(cudaMemset(d_pflag, 0, n * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_stuck, n * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(d_stuck, 0, n * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_xstuck, n * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_stuck_count, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(d_stuck_count, 0, sizeof(int)));
    if(P.stick_h > 0.0) printf("[liquid] table stick: particles within %.2f mm of the ground plane freeze\n", P.stick_h * 1000.0);
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_sleep, n * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(d_sleep, 0, n * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_sleep_count, sizeof(int)));
    if(P.isl_v > 0.0)
    {
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_isl_label, n_cells * sizeof(int)));
        CUDA_SAFE_CALL(cudaMemset(d_isl_label, 0xff, n_cells * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_isl_cnt, n_cells * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_isl_miss, n_cells * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_isl_fast, n_cells * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_isl_minrest, n_cells * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_isl_flag, 2 * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_pisl, n * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_prest, n * sizeof(int)));
        CUDA_SAFE_CALL(cudaMemset(d_prest, 0, n * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_xold, n * sizeof(double3)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_Jold, n * sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_vnmag, n * sizeof(double)));
        CUDA_SAFE_CALL(cudaMemset(d_vnmag, 0, n * sizeof(double)));
        printf("[liquid] island sleep: a connected body slower than %.2f mm/s (%.1f%% may exceed, none above %.2f mm/s) for %.2f s is held as a whole\n",
               P.isl_v * 1000.0, P.isl_allow * 100.0, P.isl_wake * 1000.0, P.isl_t);
    }
    if(P.rest_k > 0.0)
        printf("[liquid] rest drag: nodes slower than %.2f mm/s carry up to %.0fx inertia toward rest (inside the solve)\n", P.rest_kv * 1000.0, 1.0 + P.rest_k);
    if(P.sleep_v > 0.0)
        printf("[liquid] rest sleep: particles slower than %.2f mm/s for %d steps are held until faster than %.2f mm/s\n",
               P.sleep_v * 1000.0, P.sleep_n, P.wake_v * 1000.0);
    h_spacing = std::cbrt(particle_volume);
    if(P.j_mode == 4)
    {
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_chi, (size_t)n_cells * 64));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_chi2, (size_t)n_cells * 64));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_near_flag, n_cells * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_near_scan, n_cells * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_near_cells, n_cells * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_near_count, sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_fill, max_voxels * sizeof(double)));
        // erosion ball on the dx/4 sub-grid (radius fill_erode h / (dx/4); 1.5 h = 3 steps at 8 ppc)
        double rs = P.fill_erode * h_spacing / (0.25 * P.dx);
        if(rs > 4.0)
        {
            printf("[liquid] fill_erode %.2f h = %.1f sub-steps > 4: clamped (use more particles per cell)\n", P.fill_erode, rs);
            rs = 4.0;
        }
        std::vector<signed char> off;
        int                      nr = (int)std::ceil(rs);
        for(int a = -nr; a <= nr; ++a)
            for(int b = -nr; b <= nr; ++b)
                for(int c = -nr; c <= nr; ++c)
                {
                    if(a == 0 && b == 0 && c == 0) continue;
                    if(a * a + b * b + c * c <= rs * rs + 1e-9) { off.push_back((signed char)a); off.push_back((signed char)b); off.push_back((signed char)c); }
                }
        int ne = (int)off.size() / 3;
        if(ne > LIQ_ERODE_MAX) throw std::runtime_error("[liquid] erosion stencil too large");
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_erode_off, off.data(), off.size()));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_erode_n, &ne, sizeof(int)));
        double rd = P.fill_dilate * h_spacing;
        if(rd > P.dx + 1e-12)
            printf("[liquid] fill_dilate %.2f h = %.2f dx > 1 dx: the 27-cell search misses particles beyond one cell (use >= 8 particles per cell)\n",
                   P.fill_dilate, rd / P.dx);
        printf("[liquid] j_mode 4: fill reconstruction dilate %.2f h, erode %.2f h (%d sub-grid offsets), own %.2f h; sub-grid %zu MB x2\n",
               P.fill_dilate, P.fill_erode, ne, P.fill_own, (size_t)n_cells * 64 >> 20);
    }
    // calibrate the SPH density on the seed (interior value = max over particles)
    build_neighbor_grid();
    density_pass(true);
}

void LiquidSystem::build_neighbor_grid()
{
    const int n = n_particles;
    k_pkeys<<<blocks_for(n), 256>>>(n, d_xp, P.origin, P.dx, P.dims, d_pkey, d_pidx);
    thrust::sort_by_key(thrust::device, thrust::device_pointer_cast(d_pkey), thrust::device_pointer_cast(d_pkey) + n, thrust::device_pointer_cast(d_pidx));
    CUDA_SAFE_CALL(cudaMemsetAsync(d_cstart, 0xff, n_cells * sizeof(int)));  // -1
    CUDA_SAFE_CALL(cudaMemsetAsync(d_cend, 0, n_cells * sizeof(int)));
    k_cell_ranges<<<blocks_for(n), 256>>>(n, d_pkey, d_cstart, d_cend);
}

void LiquidSystem::push_apart_pass()
{
    if(P.push_apart <= 0.0) return;
    const int n  = n_particles;
    double    r0 = P.push_apart * h_spacing;
    // per-step cap: never more than a quarter spacing, and (when the engine has a
    // barrier) less than half the barrier gap, so a particle outside the dhat zone
    // can never be pushed across a solid surface; particles in contact are skipped.
    double    cap = 0.25 * h_spacing;
    if(P.dhat > 0.0) cap = fmin(cap, 0.45 * P.dhat);
    // film mode needs the near-wall flags of the last wall pass, a +y ground plane, and a barrier narrower than the wall
    // search radius (a contact with no triangle within that radius is then the ground plane's)
    const double rq_wall = (P.cpic ? fmax(P.wfc_r * h_spacing, P.cpic_r * P.dx) : P.wfc_r * h_spacing);
    const int    film    = (P.push_film && d_wp_near && P.clamp_lo_y > -1e299 && P.dhat < rq_wall) ? P.push_film : 0;
    const double r0f     = (P.push_film_r > 0.0 ? P.push_film_r : P.push_apart) * h_spacing;
    if(P.push_film && steps_done == 1)
        printf("[liquid] push-apart film mode %d: water within %.1f mm of the ground plane, spacing %.2f mm%s\n", P.push_film, P.dx * 1e3, r0f * 1e3,
               film ? "" : " -- INACTIVE (needs LIQ_WFC or LIQ_CPIC, a +y ground plane and dhat < the wall search radius)");
    // LIQ_UG_PUSH_CONTACT=1 (UG mode): particles in solid contact are pushed too (the per-step cap < 0.45 dhat keeps them
    // outside the surface; the barrier + CCD of this step then hold them), so the contact layer at the floor / walls is
    // regularised like the interior instead of being skipped
    static const int s_push_contact = getenv("LIQ_UG_PUSH_CONTACT") ? atoi(getenv("LIQ_UG_PUSH_CONTACT")) : 0;
    const int* pflag_push = (P.ug && s_push_contact) ? nullptr : d_pflag;
    k_neighbors<0><<<blocks_for(n), 256>>>(n, d_xp, d_mp, d_pkey, d_pidx, d_cstart, d_cend, P.origin, P.dx, P.dims, r0, cap, 0.0, d_push, d_rho, pflag_push,
                                           d_mc, d_cell_flag, P.rho * P.dx * P.dx * P.dx, P.push_surf_ratio,
                                           (P.cpic && d_wp_near) ? d_wp_near : nullptr, d_wp_X, d_wp_n,
                                           film, film ? d_wp_near : nullptr, P.clamp_lo_y - 1e-4, P.dx, r0f, d_vp);
    if(P.stick_h > 0.0) k_zero_push_stuck<<<blocks_for(n), 256>>>(n, d_stuck, d_push);
    if(P.sleep_v > 0.0) k_zero_push_asleep<<<blocks_for(n), 256>>>(n, d_sleep, P.sleep_n, d_push);
    if(P.isl_v > 0.0 && isl_n_steps > 0) k_zero_push_asleep<<<blocks_for(n), 256>>>(n, d_prest, isl_n_steps, d_push);
    double3 lo = make_double3(P.origin.x, fmax(P.origin.y - 0.5 * P.dx, P.clamp_lo_y), P.origin.z);
    double3 hi = make_double3(P.origin.x + P.dims.x * P.dx - 1e-9, P.origin.y + P.dims.y * P.dx - 1e-9, P.origin.z + P.dims.z * P.dx - 1e-9);
    k_apply_push<<<blocks_for(n), 256>>>(n, d_xp, d_push, lo, hi);
}

void LiquidSystem::island_pass(double dt)
{
    const int n = n_particles;
    if(h_active_cells <= 0) return;
    isl_n_steps = max(1, (int)ceil(P.isl_t / dt));
    k_isl_init<<<blocks_for(h_active_cells), 256>>>(h_active_cells, d_active_cells, d_isl_label);
    for(int it = 0; it < 512; ++it)
    {
        if(it % 4 == 0) CUDA_SAFE_CALL(cudaMemsetAsync(d_isl_flag, 0, sizeof(int)));
        k_isl_prop<<<blocks_for(h_active_cells), 256>>>(h_active_cells, d_active_cells, P.dims, d_cell_flag, d_isl_label, d_isl_flag);
        k_isl_jump<<<blocks_for(h_active_cells), 256>>>(h_active_cells, d_active_cells, d_isl_label);
        if(it % 4 == 3)
        {
            int changed = 0;
            CUDA_SAFE_CALL(cudaMemcpy(&changed, d_isl_flag, sizeof(int), cudaMemcpyDeviceToHost));
            if(!changed) break;
        }
    }
    CUDA_SAFE_CALL(cudaMemsetAsync(d_isl_cnt, 0, n_cells * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemsetAsync(d_isl_miss, 0, n_cells * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemsetAsync(d_isl_fast, 0, n_cells * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemsetAsync(d_isl_minrest, 0x7f, n_cells * sizeof(int)));   // 0x7f7f7f7f: larger than any counter
    const int pump_on = (pump_r > 0.0 && (pump_conveyor != 0.0 || pump_accel != 0.0 || pump_follow != 0.0 || pump_track > 0.0)) ? 1 : 0;
    k_isl_accum<<<blocks_for(n), 256>>>(n, d_xold, P.origin, P.dx, P.dims, d_cell_flag, d_isl_label, d_vnmag, d_prest, P.stick_h > 0.0 ? d_stuck : nullptr,
                                        P.isl_v, P.isl_wake, pump_on, pump_a, pump_b, pump_r, d_pisl, d_isl_cnt, d_isl_miss, d_isl_fast, d_isl_minrest);
    CUDA_SAFE_CALL(cudaMemsetAsync(d_isl_flag + 1, 0, sizeof(int)));
    k_isl_apply<<<blocks_for(n), 256>>>(n, d_pisl, d_isl_cnt, d_isl_miss, d_isl_fast, d_isl_minrest, P.isl_allow, isl_n_steps,
                                        P.stick_h > 0.0 ? d_stuck : nullptr, d_prest, d_xp, d_xold, d_Jp, d_Jold, d_vp, d_Cp, d_isl_flag + 1);
    CUDA_SAFE_CALL(cudaGetLastError());
    if(steps_done % 200 == 0)
    {
        int h_asleep = 0;
        CUDA_SAFE_CALL(cudaMemcpy(&h_asleep, d_isl_flag + 1, sizeof(int), cudaMemcpyDeviceToHost));
        printf("[liquid] island sleep: %d / %d particles held (step %d)\n", h_asleep, n, steps_done);
    }
}

void LiquidSystem::density_pass(bool calibrate)
{
    const int n  = n_particles;
    double    hs = 0.5 * P.dx;  // support 2h = dx (one lattice cell -> 27-cell search is exact)
    k_neighbors<1><<<blocks_for(n), 256>>>(n, d_xp, d_mp, d_pkey, d_pidx, d_cstart, d_cend, P.origin, P.dx, P.dims, 0.0, 0.0, hs, d_push, d_rho, nullptr);
    if(calibrate)
    {
        auto   ptr = thrust::device_pointer_cast(d_rho);
        rho_ref    = *thrust::max_element(thrust::device, ptr, ptr + n);
        if(!(rho_ref > 0.0)) rho_ref = P.rho;
        printf("[liquid] SPH density calibration: rho_ref %.1f (nominal %.1f), spacing %.2f mm\n", rho_ref, P.rho, h_spacing * 1000.0);
    }
    else if(P.j_mode == 1)
        k_reinit_J<<<blocks_for(n), 256>>>(n, d_rho, rho_ref, P.J_min, d_Jp);
}

void LiquidSystem::set_containers(const std::vector<LiquidContainer>& c, const double* abd_q, const double* abd_qv)
{
    if(c.size() > LIQ_MAX_CONT) throw std::runtime_error("[liquid] too many containers for the wall BC");
    h_cont   = c;
    d_abd_q  = abd_q;
    d_abd_qv = abd_qv;
    if(!d_cont) CUDA_SAFE_CALL(cudaMalloc((void**)&d_cont, LIQ_MAX_CONT * sizeof(LiquidContainer)));
    if(!d_wall_count) CUDA_SAFE_CALL(cudaMalloc((void**)&d_wall_count, sizeof(int)));
    if(!d_wall_t)
    {
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_wall_t, P.node_pool * sizeof(int)));
        CUDA_SAFE_CALL(cudaMemset(d_wall_t, 0, P.node_pool * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_wall_n, P.node_pool * sizeof(double3)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_wall_xt, P.node_pool * sizeof(double3)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_wall_red, (blocks_for(P.node_pool) + 1) * sizeof(double)));
    }
    if(!c.empty()) CUDA_SAFE_CALL(cudaMemcpy(d_cont, c.data(), c.size() * sizeof(LiquidContainer), cudaMemcpyHostToDevice));
    P.wall_bc = 1;
    P.wall_bc = liquid_env("LIQ_WALL_BC", P.wall_bc);
    printf("[liquid] wall BC %s: %zu containers, normal penalty %.0f rho dx^3 (slip), ground %s\n", P.wall_bc ? "ON" : "off (LIQ_WALL_BC=0)",
           c.size(), P.wall_mass, P.ground_bc ? "on" : "off");
    for(size_t k = 0; k < c.size(); ++k)
        printf("[liquid]   container %zu: ABD body %d, %d profile samples, floor %.4f, free above %.4f, shell %.4f, %s\n", k, c[k].body, c[k].n,
               c[k].py[0], c[k].y_top, c[k].band, c[k].slip ? "slip" : "no-slip");
}

void LiquidSystem::free_device()
{
    if(d_ptet) { cudaFree(d_ptet); d_ptet = nullptr; }
    if(d_pbary) { cudaFree(d_pbary); d_pbary = nullptr; }
    if(d_tet_cnt) { cudaFree(d_tet_cnt); d_tet_cnt = nullptr; }
    if(d_porous_count) { cudaFree(d_porous_count); d_porous_count = nullptr; }
    if(d_porous_vol) { cudaFree(d_porous_vol); d_porous_vol = nullptr; }
    if(d_vwater) { cudaFree(d_vwater); d_vwater = nullptr; }
    if(d_vcap) { cudaFree(d_vcap); d_vcap = nullptr; }
    void* ptrs[] = {d_xp, d_vp, d_Cp, d_Jp, d_mp, d_Vp0, d_mc, d_mvc, d_Gc, d_Vc, d_piVc, d_Jbc, d_cell_flag, d_cell_scan,
                    d_active_cells, d_mi, d_mvi, d_vi, d_node_flag, d_node_ref, d_node_ghost, d_node_scan, d_node_slot, d_active_nodes, d_counts,
                    d_vox_nodes, d_vox_V, d_vox_Jb, d_vox_ijk, d_reduce, d_pkey, d_pidx, d_cstart, d_cend, d_push, d_rho,
                    d_pbase, d_pfrac, d_Hpp, d_pflag, d_counters, d_fallback, d_pmask, d_pwinv, d_md_scaled,
                    d_cell_nabs, d_cell_axis, d_corner_vote, d_mc2, d_Vc2, d_piVc2, d_Jbc2, d_elem_cnt, d_elem_off, d_ghost_cnt, d_ghost_off, d_ghost_corner, d_slot_used, d_nref_key, d_nref_has, d_nref_X, d_nref_n, d_corner_side, d_plabel, d_nlab, d_chi, d_chi2, d_near_flag, d_near_scan, d_near_cells,
                    d_near_count, d_fill, d_stuck, d_xstuck, d_stuck_count, d_mcc, d_mci, d_cont, d_wall_count, d_wall_t, d_wall_n, d_wall_xt, d_wall_red,
                    d_tri_near, d_tri_cnt, d_tri_start, d_tri_list, d_wp_valid, d_wp_near, d_wp_X, d_wp_tri, d_wp_b, d_wp_n, d_wp_cell, d_wp_xi, d_wp_vid, d_wp_red, d_wp_act, d_wp_lam,
                    d_gp_cut, d_gp_flag, d_gp_scan, d_gp_pairs, d_gp_red, d_gp_cutn, d_gp_w, d_vi_old, d_st_n, d_sleep, d_sleep_count, d_isl_label, d_isl_cnt, d_isl_miss, d_isl_fast,
                    d_isl_minrest, d_isl_flag, d_pisl, d_prest, d_xold, d_Jold, d_vnmag, d_coh_pairs, d_coh_w, d_coh_s, d_coh_red, d_coh_count, d_coh_cwall, d_coh_nblk, d_coh_off,
                    d_stm_flag, d_stm_scan, d_stm_nrm, d_stm_p, d_stm_a, d_stm_b, d_stm_red, d_stm_A, d_stm_n2,
                    d_bh_vp, d_bh_Cp, d_bh_mvc, d_bh_Gc, d_bh_mvi, d_bh_vel, d_bh_vb, d_bh_xt, d_gp_cutv, d_gp_cutc, d_gm_key, d_gm_list, d_gm_cnt, d_gm_red, d_gm_front, d_cc_nsum, d_cc_xsum, d_cc_cnt, d_vox_cc, d_vox_a, d_vox_vg, d_bh_wall};
    for(void* p : ptrs)
        if(p) cudaFree(p);
    d_tri_near = d_tri_cnt = d_tri_start = d_tri_list = nullptr; tri_list_cap = 0;
    if(d_wfc_fsolid) { cudaFree(d_wfc_fsolid); d_wfc_fsolid = nullptr; } wfc_fsolid_n = 0;
    if(d_pump_red) { cudaFree(d_pump_red); d_pump_red = nullptr; }
    if(d_ls_alpha) { cudaFree(d_ls_alpha); d_ls_alpha = nullptr; }
    if(d_ls_scale) { cudaFree(d_ls_scale); d_ls_scale = nullptr; } ls_scale_n = 0;
    d_wp_valid = d_wp_near = d_wp_tri = d_wp_vid = nullptr; d_wp_b = d_wp_n = d_wp_xi = d_wp_X = nullptr; d_wp_cell = nullptr; d_wp_red = nullptr; d_wp_act = nullptr; d_wp_lam = nullptr;
    d_gp_cut = d_gp_flag = d_gp_scan = nullptr; d_gp_pairs = nullptr; d_gp_red = nullptr; d_gp_cutn = nullptr; d_gp_w = nullptr; gp_cap = 0;
    d_mcc = d_mci = nullptr;
    d_cont = nullptr; d_wall_count = nullptr; d_wall_t = nullptr; d_wall_n = d_wall_xt = nullptr; d_wall_red = nullptr;
    d_xp = d_vp = nullptr;
    d_Cp = d_Jp = d_mp = d_Vp0 = d_mc = d_Vc = d_piVc = d_Jbc = d_Gc = d_mi = d_vox_V = d_vox_Jb = d_reduce = nullptr;
    d_mvc = d_mvi = d_vi = d_vi_old = d_st_n = nullptr;
    d_cell_flag = d_cell_scan = d_active_cells = d_node_flag = d_node_scan = d_node_slot = d_active_nodes = d_counts = d_vox_nodes = nullptr;
    d_node_ref = d_node_ghost = nullptr;
    d_cell_nabs = nullptr; d_cell_axis = nullptr; d_corner_vote = nullptr;
    d_mc2 = d_Vc2 = d_piVc2 = d_Jbc2 = nullptr; d_elem_cnt = d_elem_off = d_ghost_cnt = d_ghost_off = nullptr; d_ghost_corner = nullptr; d_slot_used = nullptr;
    d_nref_key = nullptr; d_nref_has = nullptr; d_nref_X = d_nref_n = nullptr; d_corner_side = nullptr; d_plabel = nullptr; d_nlab = nullptr;
    d_vox_ijk = nullptr;
    d_pkey = d_pidx = d_cstart = d_cend = nullptr;
    d_push = nullptr;
    d_rho  = nullptr;
    for(void* q : {(void*)d_ug_nodes, (void*)d_ug_w, (void*)d_ug_grad, (void*)d_ug_red, (void*)d_ug_hmass, (void*)d_ug_href})
        if(q) cudaFree(q);
    d_ug_nodes = nullptr; d_ug_w = nullptr; d_ug_grad = nullptr; d_ug_red = nullptr; d_ug_hmass = nullptr; d_ug_href = nullptr;
    d_pbase = nullptr; d_pfrac = nullptr; d_Hpp = nullptr; d_pflag = nullptr; d_counters = nullptr; d_fallback = nullptr; d_pmask = nullptr; d_pwinv = nullptr;
    { int* fp = nullptr; cudaMemcpyToSymbol(g_liq_fbk, &fp, sizeof(int*)); }
    d_md_scaled = nullptr;
    d_chi = d_chi2 = nullptr;
    d_near_flag = d_near_scan = d_near_cells = d_near_count = nullptr;
    d_fill = nullptr;
    d_stuck = nullptr; d_xstuck = nullptr; d_stuck_count = nullptr;
    d_sleep = nullptr; d_sleep_count = nullptr;
    d_isl_label = d_isl_cnt = d_isl_miss = d_isl_fast = d_isl_minrest = d_isl_flag = d_pisl = d_prest = nullptr;
    d_xold = nullptr; d_Jold = d_vnmag = nullptr;
    d_coh_pairs = nullptr; d_coh_w = d_coh_s = d_coh_red = nullptr; d_coh_count = d_coh_cwall = d_coh_nblk = d_coh_off = nullptr; coh_cap = 0; h_coh = 0;
    d_stm_flag = d_stm_scan = d_stm_p = nullptr; d_stm_nrm = d_stm_a = d_stm_b = d_stm_n2 = nullptr; d_stm_red = d_stm_A = nullptr; h_stm = 0;
    d_bh_vp = d_bh_mvc = d_bh_mvi = d_bh_vel = d_bh_vb = d_bh_xt = d_gp_cutv = nullptr; d_gp_cutc = nullptr;
    d_gm_key = nullptr; d_gm_list = nullptr; d_gm_cnt = nullptr; d_gm_red = nullptr; d_gm_front = nullptr; h_gm = 0; gm_cap = 0;
    d_cc_nsum = d_cc_xsum = nullptr; d_cc_cnt = nullptr; d_vox_cc = nullptr; d_vox_a = nullptr; d_vox_vg = nullptr; h_cc = 0;
    if(d_vox_f) cudaFree(d_vox_f); if(d_vox_w) cudaFree(d_vox_w); if(d_wb_phi) cudaFree(d_wb_phi); if(d_node_w) cudaFree(d_node_w); if(d_wb_cnt) cudaFree(d_wb_cnt);
    if(d_wb_n) cudaFree(d_wb_n); if(d_wb_agroot) cudaFree(d_wb_agroot); if(d_wb_hg) cudaFree(d_wb_hg); d_wb_hg = nullptr; if(d_vox_s) cudaFree(d_vox_s); d_vox_s = nullptr; if(d_pdev_red) cudaFree(d_pdev_red); d_pdev_red = nullptr; if(d_visc_red) cudaFree(d_visc_red); d_visc_red = nullptr; if(d_jf_num) cudaFree(d_jf_num); d_jf_num = nullptr; if(d_jf_den) cudaFree(d_jf_den); d_jf_den = nullptr; if(d_jf_flag) cudaFree(d_jf_flag); d_jf_flag = nullptr; if(d_jf_red) cudaFree(d_jf_red); d_jf_red = nullptr; if(d_cell2elem) cudaFree(d_cell2elem); d_cell2elem = nullptr; if(d_vox_Vr) cudaFree(d_vox_Vr); d_vox_Vr = nullptr; if(d_vox_yc) cudaFree(d_vox_yc); d_vox_yc = nullptr; if(d_ybar) cudaFree(d_ybar); d_ybar = nullptr; if(d_vox_wc) cudaFree(d_vox_wc); d_vox_wc = nullptr; if(d_wb_agdbg) cudaFree(d_wb_agdbg); d_wb_agdbg = nullptr; if(d_wb_agtype) cudaFree(d_wb_agtype); if(d_wb_agt) cudaFree(d_wb_agt); if(d_wb_agnrm) cudaFree(d_wb_agnrm); d_wb_agtype = d_wb_agt = nullptr; d_wb_agnrm = nullptr; if(d_wb_agflag) cudaFree(d_wb_agflag); if(d_wb_agscan) cudaFree(d_wb_agscan); if(d_wb_agn) cudaFree(d_wb_agn); if(d_wb_agr) cudaFree(d_wb_agr); if(d_wb_agc) cudaFree(d_wb_agc); if(d_wb_agred) cudaFree(d_wb_agred);
    d_wb_agroot = d_wb_agflag = d_wb_agscan = d_wb_agn = d_wb_agr = nullptr; d_wb_agc = d_wb_agred = nullptr; h_ag = ag_cap = 0; if(d_wb_mfl) cudaFree(d_wb_mfl); d_wb_mfl = nullptr; if(d_wb_dead) cudaFree(d_wb_dead); d_wb_dead = nullptr; if(d_wb_pmin) cudaFree(d_wb_pmin); d_wb_pmin = nullptr; if(d_wb_hraw) cudaFree(d_wb_hraw); d_wb_hraw = nullptr; if(d_wb_cflag) cudaFree(d_wb_cflag); if(d_wb_cvgeo) cudaFree(d_wb_cvgeo); if(d_wb_hcol) cudaFree(d_wb_hcol); if(d_wb_blo) cudaFree(d_wb_blo); if(d_wb_bhi) cudaFree(d_wb_bhi); if(d_wb_hwt) cudaFree(d_wb_hwt); d_wb_hwt = nullptr; if(d_wb_frz) cudaFree(d_wb_frz); d_wb_frz = nullptr; if(d_wb_colD) cudaFree(d_wb_colD); if(d_wb_colG) cudaFree(d_wb_colG); d_wb_colD = d_wb_colG = nullptr; if(d_wb_ovh) cudaFree(d_wb_ovh); if(d_wb_ovr) cudaFree(d_wb_ovr); if(d_wb_ovr2) cudaFree(d_wb_ovr2); if(d_wb_hcol2) cudaFree(d_wb_hcol2); d_wb_ovh = d_wb_ovr = d_wb_ovr2 = nullptr; d_wb_hcol2 = nullptr; if(d_wb_thin) cudaFree(d_wb_thin); d_wb_thin = nullptr; wb_thin_cap = 0; if(d_wb_lab) cudaFree(d_wb_lab); if(d_wb_blk) cudaFree(d_wb_blk); if(d_wb_pcomp) cudaFree(d_wb_pcomp); if(d_wb_ccflag) cudaFree(d_wb_ccflag); d_wb_lab = d_wb_pcomp = d_wb_ccflag = nullptr; d_wb_blk = nullptr; wb_pcomp_cap = 0;
    d_wb_cflag = d_wb_blo = d_wb_bhi = nullptr; d_wb_cvgeo = d_wb_hcol = nullptr;
    d_vox_f = nullptr; d_vox_w = nullptr; d_wb_phi = nullptr; d_node_w = nullptr; d_wb_cnt = nullptr; d_wb_n = nullptr; h_wb = h_wb_surf = h_wb_allneg = h_wb_empty = h_wb_slip = 0;
    d_bh_wall = nullptr; d_bh_Cp = d_bh_Gc = nullptr; bh_init = false;
}

void LiquidSystem::fill_fraction_pass()
{
    build_neighbor_grid();  // particles may have moved since the push-apart grid
    k_near_flag<<<blocks_for(n_cells), 256>>>(n_cells, d_cell_flag, P.dims, d_near_flag);
    thrust::exclusive_scan(thrust::device, thrust::device_pointer_cast(d_near_flag), thrust::device_pointer_cast(d_near_flag) + n_cells,
                           thrust::device_pointer_cast(d_near_scan));
    k_compact_cells<<<blocks_for(n_cells), 256>>>(n_cells, d_near_flag, d_near_scan, d_near_cells);
    k_count1<<<1, 1>>>(n_cells, d_near_flag, d_near_scan, d_near_count);
    CUDA_SAFE_CALL(cudaMemcpy(&h_near_cells, d_near_count, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemsetAsync(d_chi, 0, (size_t)n_cells * 64));
    CUDA_SAFE_CALL(cudaMemsetAsync(d_chi2, 0, (size_t)n_cells * 64));
    double rd = fmin(P.fill_dilate * h_spacing, P.dx), ro = P.fill_own * h_spacing;
    if(h_near_cells > 0)
    {
        k_fill_sample<<<blocks_for(h_near_cells * 64), 256>>>(h_near_cells, d_near_cells, P.dims, P.origin, P.dx, d_xp, d_pidx, d_cstart, d_cend,
                                                               rd * rd, ro * ro, d_chi);
        k_fill_erode<<<blocks_for(h_near_cells * 64), 256>>>(h_near_cells, d_near_cells, P.dims, d_chi, d_chi2);
    }
    if(h_active_cells > 0)
        k_fill_apply<<<blocks_for(h_active_cells), 256>>>(h_active_cells, d_active_cells, P.dims, d_chi2, d_mc, P.rho * P.dx * P.dx * P.dx, P.J_min,
                                                          d_Jbc, d_fill);
    if(getenv("LIQ_DEBUG") && steps_done % 50 == 0 && h_active_cells > 0)
    {
        std::vector<double> fill(h_active_cells), mc(n_cells), Jb(n_cells);
        std::vector<int>    act(h_active_cells);
        CUDA_SAFE_CALL(cudaMemcpy(fill.data(), d_fill, h_active_cells * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(act.data(), d_active_cells, h_active_cells * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(mc.data(), d_mc, n_cells * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(Jb.data(), d_Jbc, n_cells * sizeof(double), cudaMemcpyDeviceToHost));
        double rc = P.rho * P.dx * P.dx * P.dx, fsum = 0, jmin = 1e300, mmin = 0;
        int    n05 = 0, n09 = 0, imin = -1;
        for(int a = 0; a < h_active_cells; ++a)
        {
            int c = act[a];
            fsum += fill[a];
            double J = Jb[c];
            if(J < 0.5) ++n05;
            if(J < 0.9) ++n09;
            if(J < jmin) { jmin = J; imin = c; mmin = mc[c] / rc; }
        }
        int i, j, k;
        if(imin >= 0) cell_ijk(imin, P.dims, i, j, k);
        printf("[liquid] step %d: active %d near %d cells, mean fill %.3f | cell J min %.3f (cell %d,%d,%d mass %.4f fill %.4f), J<0.5: %d, J<0.9: %d\n",
               steps_done, h_active_cells, h_near_cells, fsum / h_active_cells, jmin, i, j, k, mmin, (imin >= 0 ? jmin * mmin : 0.0), n05, n09);
    }
    CUDA_SAFE_CALL(cudaGetLastError());
}

// ---- porous body (open-cell sponge) ------------------------------------------------------------------
__device__ __forceinline__ double d3trip(double3 a, double3 b, double3 c)
{   // a . (b x c)
    return a.x * (b.y * c.z - b.z * c.y) + a.y * (b.z * c.x - b.x * c.z) + a.z * (b.x * c.y - b.y * c.x);
}
// barycentric coordinates of p in tet (a,b,c,d); true when inside (small tolerance on the faces)
__device__ __forceinline__ bool tet_bary(double3 p, double3 a, double3 b, double3 c, double3 d, double4& w)
{
    double3 e1 = d3sub(b, a), e2 = d3sub(c, a), e3 = d3sub(d, a), q = d3sub(p, a);
    double  det = d3trip(e1, e2, e3);
    if(fabs(det) < 1e-18) return false;
    double inv = 1.0 / det;
    double w1 = d3trip(q, e2, e3) * inv, w2 = d3trip(e1, q, e3) * inv, w3 = d3trip(e1, e2, q) * inv;
    double w0 = 1.0 - w1 - w2 - w3;
    const double eps = -1e-6;
    if(w0 < eps || w1 < eps || w2 < eps || w3 < eps) return false;
    w = make_double4(w0, w1, w2, w3);
    return true;
}
// brute force over the body's tets (a sponge has O(1e3) tets; AABB prefilter first)
__global__ void k_porous_locate(int n, const double3* xp, const uint4* tets, const double3* x, int tet0, int tet1, int* ptet, double4* pbary,
                                int* tet_cnt, int* inside)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    double3 q     = xp[p];
    int     found = -1;
    double4 w     = make_double4(0, 0, 0, 0);
    for(int t = tet0; t < tet1; ++t)
    {
        uint4   T = tets[t];
        double3 a = x[T.x], b = x[T.y], c = x[T.z], d = x[T.w];
        if(q.x < fmin(fmin(a.x, b.x), fmin(c.x, d.x)) || q.x > fmax(fmax(a.x, b.x), fmax(c.x, d.x))) continue;
        if(q.y < fmin(fmin(a.y, b.y), fmin(c.y, d.y)) || q.y > fmax(fmax(a.y, b.y), fmax(c.y, d.y))) continue;
        if(q.z < fmin(fmin(a.z, b.z), fmin(c.z, d.z)) || q.z > fmax(fmax(a.z, b.z), fmax(c.z, d.z))) continue;
        if(tet_bary(q, a, b, c, d, w)) { found = t; break; }
    }
    ptet[p] = found;
    if(found >= 0)
    {
        pbary[p] = w;
        atomicAdd(&tet_cnt[found - tet0], 1);
        atomicAdd(inside, 1);
    }
}
__device__ __forceinline__ double porous_len3(double3 v) { return sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }
// pore capacity of a tet: phi x current volume (default) or the open-cell FOAM rule (LIQ_POROUS_FOAM=1): the skeleton keeps
// its volume (1 - phi) V0 and the pores take the rest of V_eff = V0 x min(J, lambda_min), lambda_min = the smallest edge
// stretch. A uniaxial pinch then collapses the pores even when the FEM volume barely changes: at nu 0.35 the block bulges
// instead of losing volume (60 -> 24 mm under the pads lost 25 % volume, s rose only to ~1.05 and nothing was released).
__device__ __forceinline__ double porous_tet_capacity(uint4 T, const double3* x, const double3* x0, double phi, int foam)
{
    double3 a = x[T.x], b = x[T.y], c = x[T.z], d = x[T.w];
    double  vol = fabs(d3trip(d3sub(b, a), d3sub(c, a), d3sub(d, a))) / 6.0;
    if(!foam) return phi * vol;
    double3 a0 = x0[T.x], b0 = x0[T.y], c0 = x0[T.z], d0 = x0[T.w];
    double  vol0 = fabs(d3trip(d3sub(b0, a0), d3sub(c0, a0), d3sub(d0, a0))) / 6.0;
    if(vol0 <= 0.0) return 0.0;
    double        lam  = vol / vol0;
    const double3 P[4]  = {a, b, c, d};
    const double3 P0[4] = {a0, b0, c0, d0};
    for(int i = 0; i < 4; ++i)
        for(int j = i + 1; j < 4; ++j)
        {
            double l0 = porous_len3(d3sub(P0[j], P0[i]));
            if(l0 > 0.0) lam = fmin(lam, porous_len3(d3sub(P[j], P[i])) / l0);
        }
    return vol0 * fmax(0.02, lam - (1.0 - phi));
}
// per-vertex pore saturation: every tet spreads its water volume and its capacity (porosity x current volume) over
// its 4 vertices; a particle reads the saturation interpolated with its barycentric weights. One tet holds only
// ~4 particles at 8 mm water (a per-tet gate cascaded: over-full tets drained into their neighbours until the
// sponge emptied), one vertex aggregates ~24 tets; and the GLOBAL saturation missed a local squeeze entirely
// (the pads compress a 2 x 2 cm patch of a 10 x 3.5 cm sponge: total volume -10%, nothing was expelled).
__global__ void k_porous_vertex_sat(int nt, const uint4* tets, const double3* x, const double3* x0, const int* tet_cnt, double Vp, double phi,
                                    int foam, int v0, double* vwater, double* vcap)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= nt) return;
    uint4   T   = tets[i];
    double  wat = 0.25 * tet_cnt[i] * Vp, cap = 0.25 * porous_tet_capacity(T, x, x0, phi, foam);
    atomicAdd(&vwater[T.x - v0], wat); atomicAdd(&vcap[T.x - v0], cap);
    atomicAdd(&vwater[T.y - v0], wat); atomicAdd(&vcap[T.y - v0], cap);
    atomicAdd(&vwater[T.z - v0], wat); atomicAdd(&vcap[T.z - v0], cap);
    atomicAdd(&vwater[T.w - v0], wat); atomicAdd(&vcap[T.w - v0], cap);
}
__device__ __forceinline__ double porous_local_factor(uint4 T, double4 w, const double* vwater, const double* vcap, int v0, double rel)
{   // retention factor from s = interpolated water / interpolated capacity: 1 at or below capacity; above it min(1, 1/s)
    // (rel <= 0, the soft gate) or the sharp release 1 - (s - 1)/rel clamped to [0, 1] (LIQ_POROUS_REL = rel > 0).
    // Soft gate: the excess of a squeezed patch (s ~ 2) still had half of its momentum reset and half of the position
    // pull every step and crept out at ~10 mm/s (sponge_full14/16: 4-6 ml reached the cup); the sharp gate frees it.
    double wat = w.x * vwater[T.x - v0] + w.y * vwater[T.y - v0] + w.z * vwater[T.z - v0] + w.w * vwater[T.w - v0];
    double cap = w.x * vcap[T.x - v0] + w.y * vcap[T.y - v0] + w.z * vcap[T.z - v0] + w.w * vcap[T.w - v0];
    if(cap <= 0.0) return 0.0;
    double s = wat / cap;
    if(s <= 1.0) return 1.0;
    if(rel <= 0.0) return 1.0 / s;
    double f = 1.0 - (s - 1.0) / rel;
    return f > 0.0 ? f : 0.0;
}
__global__ void k_porous_tet_capacity(int nt, const uint4* tets, const double3* x, const double3* x0, double phi, int foam, double* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= nt) return;
    atomicAdd(out, porous_tet_capacity(tets[i], x, x0, phi, foam));
}
// retention factor f = min(1, 1/s) with the GLOBAL saturation s of the body: a per-tet gate was too coarse (4 particles per
// tet at 8 mm water) and cascaded: over-full tets drained into their neighbours, which then overflowed, until the sponge
// emptied during the lift. With the global factor the whole pore water is weightless while the body holds <= its capacity
// and sinks at (1 - 1/s) g when squeezed over capacity (volume down), i.e. the excess drains and the rest stays.
__global__ void k_porous_apply(int n, double3* vp, const int* ptet, const double4* pbary, const uint4* tets, const double3* vel,
                               const double* vwater, const double* vcap, int v0, double drag, double dt, double3 g, double drag_free, double rel)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    int t = ptet[p];
    if(t < 0) return;
    uint4   T  = tets[t];
    double4 w  = pbary[p];
    const double retain = porous_local_factor(T, w, vwater, vcap, v0, rel);
    double3 vm = d3add(d3add(d3scale(vel[T.x], w.x), d3scale(vel[T.y], w.y)), d3add(d3scale(vel[T.z], w.z), d3scale(vel[T.w], w.w)));
    double3 v  = vp[p];
    // Darcy-like drag toward the material velocity; with the sharp gate the released excess (retain -> 0) only keeps the
    // free-flow drag drag_free (LIQ_POROUS_DRAG_FREE, default 0 = the solved motion), the held water the full drag
    const double kd = rel > 0.0 ? fmin(1.0, drag * dt) * retain + fmin(1.0, drag_free * dt) * (1.0 - retain) : fmin(1.0, drag * dt);
    v          = d3add(v, d3scale(d3sub(vm, v), kd));
    v          = d3sub(v, d3scale(g, dt * retain));                       // capillary retention: cancels (part of) the dt^2 g of the node predictor
    vp[p] = v;
}
static double porous_rel_env()
{   // LIQ_POROUS_REL: width of the sharp release gate (0 = the original soft min(1, 1/s) gate)
    static const double v = getenv("LIQ_POROUS_REL") ? atof(getenv("LIQ_POROUS_REL")) : 0.0;
    return v;
}
static int porous_foam_env()
{   // LIQ_POROUS_FOAM=1: open-cell foam pore capacity V0 x max(0.02, min(J, lambda_min) - (1 - phi)) instead of phi x V
    static const int v = getenv("LIQ_POROUS_FOAM") ? atoi(getenv("LIQ_POROUS_FOAM")) : 0;
    return v;
}
static double porous_drag_free_env()
{   // LIQ_POROUS_DRAG_FREE: Darcy drag [1/s] kept by the released excess (sharp gate only)
    static const double v = getenv("LIQ_POROUS_DRAG_FREE") ? atof(getenv("LIQ_POROUS_DRAG_FREE")) : 0.0;
    return v;
}
void LiquidSystem::porous_pre(device_TetraData& m, double dt, double3 gravity)
{
    if(porous_body < 0 || porous_tet1 <= porous_tet0) return;
    const int n = n_particles, nt = porous_tet1 - porous_tet0;
    if(!d_ptet)
    {
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_ptet, n * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_pbary, n * sizeof(double4)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_tet_cnt, nt * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_porous_count, sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_porous_vol, sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_vwater, std::max(porous_nv, 1) * sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_vcap, std::max(porous_nv, 1) * sizeof(double)));
        printf("[liquid] porous gate: %s (LIQ_POROUS_REL %.2f, LIQ_POROUS_DRAG_FREE %.0f /s), capacity %s\n",
               porous_rel_env() > 0.0 ? "sharp release" : "soft min(1, 1/s)", porous_rel_env(), porous_drag_free_env(),
               porous_foam_env() ? "open-cell foam (LIQ_POROUS_FOAM)" : "phi x volume");
    }
    CUDA_SAFE_CALL(cudaMemsetAsync(d_tet_cnt, 0, nt * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemsetAsync(d_porous_count, 0, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemsetAsync(d_porous_vol, 0, sizeof(double)));
    k_porous_locate<<<blocks_for(n), 256>>>(n, d_xp, m.tetrahedras, m.vertexes, porous_tet0, porous_tet1, d_ptet, d_pbary, d_tet_cnt, d_porous_count);
    const int foam = porous_foam_env();
    k_porous_tet_capacity<<<blocks_for(nt), 256>>>(nt, m.tetrahedras + porous_tet0, m.vertexes, m.rest_vertexes, porous_phi, foam, d_porous_vol);
    double cap = 0.0;   // total pore capacity [m^3]
    CUDA_SAFE_CALL(cudaMemcpy(&h_porous_inside, d_porous_count, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(&cap, d_porous_vol, sizeof(double), cudaMemcpyDeviceToHost));
    h_porous_sat     = cap > 0.0 ? h_porous_inside * particle_volume / cap : 0.0;   // global, for logging
    CUDA_SAFE_CALL(cudaMemsetAsync(d_vwater, 0, porous_nv * sizeof(double)));
    CUDA_SAFE_CALL(cudaMemsetAsync(d_vcap, 0, porous_nv * sizeof(double)));
    k_porous_vertex_sat<<<blocks_for(nt), 256>>>(nt, m.tetrahedras + porous_tet0, m.vertexes, m.rest_vertexes, d_tet_cnt, particle_volume, porous_phi, foam, porous_v0, d_vwater, d_vcap);
    k_porous_apply<<<blocks_for(n), 256>>>(n, d_vp, d_ptet, d_pbary, m.tetrahedras, m.velocities, d_vwater, d_vcap, porous_v0, porous_drag, dt, gravity, porous_drag_free_env(), porous_rel_env());
}

// After G2P the inside particles are pulled to their material point x_m = sum(w_i x_i(new)) of the tet they were
// located in at the start of the step (the drag velocity set before P2G is only a momentum hint: the grid solve
// compromises it with the surrounding water and the water lagged the lifted sponge at ~half its speed, then fell
// out). beta = 1 below capacity (capillary hold: the water travels with the foam) and 1/s above it, so the excess
// of a squeezed sponge keeps the grid-solved motion (pressure pushes it out through the open surface).
__global__ void k_porous_follow(int n, double3* xp, double3* vp, const int* ptet, const double4* pbary, const uint4* tets, const double3* x,
                                const double3* vel, const double* vwater, const double* vcap, int v0, double beta0, double rel)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    int t = ptet[p];
    if(t < 0) return;
    uint4   T  = tets[t];
    double4 w  = pbary[p];
    const double beta = beta0 * porous_local_factor(T, w, vwater, vcap, v0, rel);
    double3 xm = d3add(d3add(d3scale(x[T.x], w.x), d3scale(x[T.y], w.y)), d3add(d3scale(x[T.z], w.z), d3scale(x[T.w], w.w)));
    double3 vm = d3add(d3add(d3scale(vel[T.x], w.x), d3scale(vel[T.y], w.y)), d3add(d3scale(vel[T.z], w.z), d3scale(vel[T.w], w.w)));
    xp[p] = d3add(xp[p], d3scale(d3sub(xm, xp[p]), beta));
    vp[p] = d3add(vp[p], d3scale(d3sub(vm, vp[p]), beta));
}
void LiquidSystem::porous_post(device_TetraData& m, double dt)
{
    if(porous_body < 0 || !d_ptet || porous_drag <= 0.0) return;
    const double beta0 = fmin(1.0, porous_drag * dt);
    k_porous_follow<<<blocks_for(n_particles), 256>>>(n_particles, d_xp, d_vp, d_ptet, d_pbary, m.tetrahedras, m.vertexes, m.velocities,
                                                      d_vwater, d_vcap, porous_v0, beta0, porous_rel_env());
}

void LiquidSystem::pre_step(device_TetraData& m, double dt, double3 gravity)
{
    last_gravity = gravity;
    last_dt = dt;
    const int n = n_particles;
    g_prof.start();
    CUDA_SAFE_CALL(cudaMemsetAsync(d_fallback, 0, 12 * sizeof(int)));
    // particle regularisation BEFORE the solve: any particle nudged into the barrier
    // zone is then handled by the barrier + CCD of this step
    if(P.push_apart > 0.0 && steps_done > 0)
    {
        build_neighbor_grid();
        if(P.ug)
        {   // LIQ_UG_PUSH: the free-surface gate of k_neighbors<0> reads the tent cell masses, which only the main-path P2G
            // (or the UG re-anchor) fills -> deposit them here, otherwise every particle counts as surface and nothing moves
            CUDA_SAFE_CALL(cudaMemsetAsync(d_mc, 0, n_cells * sizeof(double)));
            CUDA_SAFE_CALL(cudaMemsetAsync(d_cell_flag, 0, n_cells * sizeof(int)));
            k_ug_cellmass<<<blocks_for(n_particles), 256>>>(n_particles, d_xp, d_mp, P.origin, P.dx, P.dims, d_mc, d_cell_flag);
        }
        push_apart_pass();
    }
    if(pump_r > 0.0 && pump_mode == 2)
    {   // chamber pressure -> column: census, gas state, 1-DOF implicit axial update
        if(!d_pump_red) CUDA_SAFE_CALL(cudaMalloc((void**)&d_pump_red, 4 * sizeof(double)));
        CUDA_SAFE_CALL(cudaMemsetAsync(d_pump_red, 0, 4 * sizeof(double)));
        const double t_lo = -2.0 * h_spacing, t_open = 2.0 * h_spacing;   // bore + the liquid at the opening
        k_pump_column<<<blocks_for(n), 256>>>(n, d_xp, d_vp, d_Jp, d_Vp0, pump_a, pump_b, pump_r, t_lo, t_open, d_pump_red);
        double red[4];
        CUDA_SAFE_CALL(cudaMemcpy(red, d_pump_red, 4 * sizeof(double), cudaMemcpyDeviceToHost));
        const double ncol = red[0], Vliq = red[1], vmean = ncol > 0.0 ? red[2] / ncol : 0.0;
        const bool   sealed = red[3] > 0.0;
        const double Vair = fmax(pump_Vcav - Vliq, 1e-9);
        if(!sealed || pump_V0 <= 0.0) pump_V0 = Vair;   // open tip: air enters / leaves freely, p = p0
        const double pg = pump_p0 * (pump_V0 / Vair - 1.0);
        const double bore = pump_bore > 0.0 ? pump_bore : pump_r;
        const double A    = M_PI * bore * bore;
        double dv = 0.0;
        if(sealed && ncol > 0.0)
        {
            const double mcol = ncol * P.rho * particle_volume;
            const double Fax  = -pg * A;                                   // +axis = up the tube: p_gauge > 0 pushes OUT
            const double k    = pump_p0 * pump_V0 * A * A / (Vair * Vair); // gas spring on the column's axial position
            const double vnew = (vmean + dt * Fax / mcol) / (1.0 + dt * dt * k / mcol);
            dv = vnew - vmean;
            k_pump_kick<<<blocks_for(n), 256>>>(n, d_xp, d_vp, pump_a, pump_b, pump_r, t_lo, dv);
        }
        h_pump_state[0] = Vair; h_pump_state[1] = pg; h_pump_state[2] = Vliq; h_pump_state[3] = ncol; h_pump_state[4] = pump_V0; h_pump_state[5] = sealed ? 1.0 : 0.0;
    }
    else if(pump_r > 0.0)
    {
        double3 v_tube = make_double3((pump_a.x - pump_a_prev.x) / dt, (pump_a.y - pump_a_prev.y) / dt, (pump_a.z - pump_a_prev.z) / dt);
        // positions first (conveyor + follow the tube), in the same frame as the tube's own motion
        const double bore       = pump_bore > 0.0 ? pump_bore : pump_r;
        const double bore_clear = fmax(0.0, bore - P.dhat);
        bool         sink_done  = false;
        PumpSinkCfg& S          = pump_sink_cfg();
        if(S.on && pump_conveyor != 0.0 && bore_clear > 1e-6)
        {   // LIQ_PUMP_SINK: with the tip immersed the conveyor draws from / delivers to the pool around the opening
            static bool announced = false;
            if(!announced)
            {
                announced = true;
                printf("[liquid] pump sink conveyor ON: capture radius %.2f mm, slab %.1f mm, line sink full to %.1f mm / zero at %.1f mm, "
                       "immersed = %d particles within %.1f mm below the tip plane, attach guards %s\n",
                       bore_clear * 1e3, S.h * 1e3, S.R1 * 1e3, S.R * 1e3, S.n_min, S.top * 1e3, S.gate == 0.0 ? "off" : "on");
            }
            if(!d_pump_red) CUDA_SAFE_CALL(cudaMalloc((void**)&d_pump_red, 4 * sizeof(double)));
            CUDA_SAFE_CALL(cudaMemsetAsync(d_pump_red, 0, 4 * sizeof(double)));
            const double gate = S.gate < 0.0 ? h_spacing : S.gate;
            k_pump_sink_census<<<blocks_for(n), 256>>>(n, d_xp, pump_a, pump_b, pump_r, bore_clear, S.R1, S.top, gate > 0.0 ? gate : h_spacing, d_pump_red);
            double red[4];
            CUDA_SAFE_CALL(cudaMemcpy(red, d_pump_red, 4 * sizeof(double), cudaMemcpyDeviceToHost));
            double c_slab = 0.0, c_bore = 0.0;
            if(red[0] >= S.n_min)
            {
                c_slab = c_bore = pump_conveyor;
                if(gate > 0.0)
                {   // attach guards, proportional: n_full = particles in the feed zone (capture cylinder, `gate` long) at the pool's density.
                    // Intake: the column is lifted at the conveyor speed times the feed zone's filling, so it moves only as fast as water
                    // follows it through the opening and the bore fills at the pool's density (lifting at the full speed whenever one
                    // particle was ready stretched the column to 0.6 of it: 64 particles over 100 mm). Nothing ready = no lift.
                    // Expulsion: the source flow in the pool is scaled by the filling of the bore's lowest `gate`; nothing leaving = none.
                    const double n_full = M_PI * bore_clear * bore_clear * gate / particle_volume;
                    if(pump_conveyor > 0.0) c_bore = pump_conveyor * fmin(1.0, red[1] / n_full);
                    if(pump_conveyor < 0.0) c_slab = pump_conveyor * fmin(1.0, red[2] / n_full);
                }
                k_pump_sink<<<blocks_for(n), 256>>>(n, d_xp, pump_a, pump_b, pump_r, bore_clear, S.h, S.R1, S.R, c_slab, c_bore, v_tube, pump_follow, dt);
                sink_done = true;
            }
            // diagnostics through get_liquid_pump_state(): around the opening, ready to enter, ready to leave, in the bore, c_slab, c_bore
            h_pump_state[0] = red[0]; h_pump_state[1] = red[1]; h_pump_state[2] = red[2]; h_pump_state[3] = red[3];
            h_pump_state[4] = sink_done ? c_slab : 0.0; h_pump_state[5] = sink_done ? c_bore : pump_conveyor;
        }
        if(!sink_done && (pump_conveyor != 0.0 || pump_follow != 0.0))
            k_pump<1><<<blocks_for(n), 256>>>(n, d_xp, d_vp, pump_a, pump_b, pump_r, 0.0, pump_conveyor, v_tube, pump_follow, 0.0, dt, bore_clear, nullptr);
        if(g_pump_track_rigid < 0)
        {
            g_pump_track_rigid = liquid_env("LIQ_PUMP_TRACK_RIGID", 0);
            liquid_require(g_pump_track_rigid == 0 || g_pump_track_rigid == 1, "LIQ_PUMP_TRACK_RIGID must be 0 or 1");
            if(g_pump_track_rigid) printf("[liquid] pump velocity tracking: rigid-body velocity of the tube (tip and top), not the tip's alone\n");
        }
        // top velocity from the top's displacement since the previous step (first step: the tip's, i.e. pure translation)
        const double3 v_top = g_pump_top_has_prev ? make_double3((pump_b.x - g_pump_top_prev.x) / dt, (pump_b.y - g_pump_top_prev.y) / dt, (pump_b.z - g_pump_top_prev.z) / dt)
                                                  : v_tube;
        g_pump_top_prev     = pump_b;
        g_pump_top_has_prev = true;
        if(pump_accel != 0.0 || pump_track > 0.0)
        {
            if(g_pump_track_rigid)
                k_pump_track_rigid<<<blocks_for(n), 256>>>(n, d_xp, d_vp, pump_a, pump_b, pump_r, pump_accel, v_tube, v_top, pump_track, dt);
            else
                k_pump<0><<<blocks_for(n), 256>>>(n, d_xp, d_vp, pump_a, pump_b, pump_r, pump_accel, 0.0, v_tube, 0.0, pump_track, dt, bore_clear, nullptr);
        }
    }
    porous_pre(m, dt, gravity);   // sponge: drag + capillary retention on the particles inside the porous body
    if(P.st_gamma > 0.0)
    {   // explicit surface tension (Akinci 2013) on the particle velocities before P2G; stable for dt < sqrt(rho dx^3 / (2 pi sigma))
        build_neighbor_grid();
        const double hs = 0.5 * P.dx;
        CUDA_SAFE_CALL(cudaGetLastError());
        k_neighbors<1><<<blocks_for(n), 256>>>(n, d_xp, d_mp, d_pkey, d_pidx, d_cstart, d_cend, P.origin, P.dx, P.dims, 0.0, 0.0, hs, d_push, d_rho, nullptr);
        CUDA_SAFE_CALL(cudaGetLastError());
        k_st_normals<<<blocks_for(n), 256>>>(n, d_xp, d_mp, d_rho, d_pidx, d_cstart, d_cend, P.origin, P.dx, P.dims, hs, d_st_n);
        CUDA_SAFE_CALL(cudaGetLastError());
        k_st_apply<<<blocks_for(n), 256>>>(n, d_vp, d_xp, d_mp, d_rho, d_st_n, d_pidx, d_cstart, d_cend, P.origin, P.dx, P.dims, P.st_gamma,
                                           rho_ref > 0.0 ? rho_ref : P.rho, P.st_c * P.dx, dt, P.stick_h > 0.0 ? d_stuck : nullptr,
                                           P.st_thin > 0.0 ? P.st_thin * (rho_ref > 0.0 ? rho_ref : P.rho) : 0.0,
                                           (steps_done > 0 && d_wp_near) ? d_wp_near : nullptr);   // near-wall flags from the previous step
        CUDA_SAFE_CALL(cudaGetLastError());
        if(steps_done == 0) printf("[liquid] surface tension ON: gamma %.4g, cohesion support %.2f dx, rho0 %.1f, thin-layer gate %s\n", P.st_gamma, P.st_c, rho_ref,
                                   P.st_thin > 0.0 ? (std::to_string(P.st_thin) + " rho0").c_str() : "off");
    }
    // particle stencils are final here (push-apart, pump, porous, surface tension are done): build them before the
    // wall geometry (compatibility masks need them) and the transfers
    k_particle_stencil<<<blocks_for(n), 256>>>(n, d_xp, P.origin, P.dx, P.dims, d_pbase, d_pfrac);
    if(P.ug)
    {
        if(rep_on()) cohesion_pairs();   // LIQ_UG_REP: step-start pair list (the main-path call below is not reached in UG mode)
        ug_pre(m, dt, gravity);
        g_prof.stop(5);
        return;
    }
    if((P.wfc || P.cpic) && d_faces && n_faces > 0) wall_geometry(m);   // nearest wall per particle (+ masks), step-start solid positions
    if(P.sti_gamma > 0.0 || rep_on()) cohesion_pairs();
    if(P.stm_sigma > 0.0) stm_setup();
    if(P.bdf2)
    {
        if(wall_active() && !P.wb) throw std::runtime_error("[liquid] LIQ_BDF2 does not support the grid wall BC (LIQ_WALL_BC): its wall nodes are implicit-Euler only");
        if(!bh_init)
        {   // first step: no history yet -> v_prev = v (the predictor reduces to x + dt v)
            CUDA_SAFE_CALL(cudaMemcpyAsync(d_bh_vp, d_vp, n * sizeof(double3), cudaMemcpyDeviceToDevice));
            CUDA_SAFE_CALL(cudaMemcpyAsync(d_bh_Cp, d_Cp, 9 * (size_t)n * sizeof(double), cudaMemcpyDeviceToDevice));
            bh_init = true;
        }
    }
    CUDA_SAFE_CALL(cudaMemsetAsync(d_mc, 0, n_cells * sizeof(double)));
    CUDA_SAFE_CALL(cudaMemsetAsync(d_mvc, 0, n_cells * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMemsetAsync(d_Gc, 0, 9 * n_cells * sizeof(double)));
    CUDA_SAFE_CALL(cudaMemsetAsync(d_Vc, 0, n_cells * sizeof(double)));
    if(P.wb && P.wb_ybar)
    {
        if(!d_ybar) CUDA_SAFE_CALL(cudaMalloc((void**)&d_ybar, (size_t)n_cells * sizeof(double)));
        CUDA_SAFE_CALL(cudaMemsetAsync(d_ybar, 0, (size_t)n_cells * sizeof(double)));
    }
    CUDA_SAFE_CALL(cudaMemsetAsync(d_piVc, 0, n_cells * sizeof(double)));

    const bool split = P.cpic && P.cpic_split && d_wp_near;
    const bool twof  = P.cpic && P.cpic_fields == 2 && d_wp_near;
    if(twof)
    {
        CUDA_SAFE_CALL(cudaMemsetAsync(d_corner_side, 0, 16 * (size_t)n_cells * sizeof(int)));
        CUDA_SAFE_CALL(cudaMemsetAsync(d_mc2, 0, n_cells * sizeof(double)));
        CUDA_SAFE_CALL(cudaMemsetAsync(d_Vc2, 0, n_cells * sizeof(double)));
        CUDA_SAFE_CALL(cudaMemsetAsync(d_piVc2, 0, n_cells * sizeof(double)));
    }
    if(split)
    {
        CUDA_SAFE_CALL(cudaMemsetAsync(d_corner_vote, 0, 32 * (size_t)n_cells * sizeof(float)));
        CUDA_SAFE_CALL(cudaMemsetAsync(d_mc2, 0, n_cells * sizeof(double)));
        CUDA_SAFE_CALL(cudaMemsetAsync(d_Vc2, 0, n_cells * sizeof(double)));
        CUDA_SAFE_CALL(cudaMemsetAsync(d_piVc2, 0, n_cells * sizeof(double)));
    }
    k_p2g<<<blocks_for(n), 256>>>(n, d_xp, d_vp, d_Cp, d_Jp, d_mp, d_Vp0, P.dx, P.origin, P.dims, P.kappa, P.tension_ratio, P.apic, d_mc, d_mvc, d_Gc, d_Vc, d_piVc,
                                  (P.cpic && P.cpic_cell && d_wp_near) ? d_wp_near : nullptr, d_wp_X, d_wp_n,
                                  (split && P.cpic_side1) ? d_cell_axis : nullptr, split ? d_wp_near : nullptr, d_wp_n, (split || twof) ? d_mc2 : nullptr, d_Vc2, d_piVc2, d_pmask, split ? d_corner_vote : nullptr,
                                  d_nref_has, d_nref_X, d_nref_n, twof ? n_nodes : 0, twof ? d_corner_side : nullptr, d_plabel, d_nlab,
                                  (P.wb && P.wb_ybar && d_ybar) ? d_ybar : nullptr, liquid_up(gravity));
    k_cell_finalize<<<blocks_for(n_cells), 256>>>(n_cells, d_mc, d_mvc, d_Gc, d_Vc, d_piVc, d_Jbc, d_cell_flag, P.kappa, P.tension_ratio, P.j_mode,
                                                  P.rho * P.dx * P.dx * P.dx, (split || twof) ? d_mc2 : nullptr, d_Vc2, d_piVc2, (split || twof) ? d_Jbc2 : nullptr);
    if(P.cpic && P.cpic_p2g)
    {   // masked direct particle -> node transfer (identical to the cell route where nothing is masked)
        CUDA_SAFE_CALL(cudaMemsetAsync(d_mi, 0, n_vnodes * sizeof(double)));
        CUDA_SAFE_CALL(cudaMemsetAsync(d_mvi, 0, n_vnodes * sizeof(double3)));
        CUDA_SAFE_CALL(cudaMemsetAsync(d_node_ref, 0, n_vnodes * sizeof(int)));
        k_p2g_nodes<<<blocks_for(n), 256>>>(n, d_xp, d_vp, d_Cp, d_mp, d_pbase, d_pfrac, d_pmask, d_pwinv, P.origin, P.dx, P.dims, P.apic, d_mi, d_mvi, d_node_ref,
                                            d_nref_has, d_nref_X, d_nref_n, twof ? n_nodes : 0, d_plabel, d_nlab);
        k_node_flag<<<blocks_for(n_vnodes), 256>>>(n_vnodes, d_mi, d_node_ref, P.cpic_ghost, d_node_flag, d_node_ghost);
    }
    else
        k_node_gather<<<blocks_for(n_nodes), 256>>>(n_nodes, P.dims, P.dx, d_mc, d_mvc, d_Gc, d_cell_flag, d_mi, d_mvi, d_node_flag);
    if(P.bdf2)
    {   // previous-step velocity field on this step's grid (same weights and cell masses as the momentum)
        CUDA_SAFE_CALL(cudaMemsetAsync(d_bh_mvc, 0, n_cells * sizeof(double3)));
        CUDA_SAFE_CALL(cudaMemsetAsync(d_bh_Gc, 0, 9 * (size_t)n_cells * sizeof(double)));
        k_p2g_hist<<<blocks_for(n), 256>>>(n, d_xp, d_bh_vp, d_bh_Cp, d_mp, P.dx, P.origin, P.dims, P.apic, d_bh_mvc, d_bh_Gc);
        k_cell_norm_hist<<<blocks_for(n_cells), 256>>>(n_cells, d_mc, d_cell_flag, d_bh_mvc, d_bh_Gc);
        k_node_gather_hist<<<blocks_for(n_nodes), 256>>>(n_nodes, P.dims, P.dx, d_mc, d_bh_mvc, d_bh_Gc, d_cell_flag, d_bh_mvi);
        if(P.bdf2_wall_be)
        {   // near-wall nodes keep implicit Euler (wall points are from this step's wall pass)
            if(!d_bh_wall) CUDA_SAFE_CALL(cudaMalloc((void**)&d_bh_wall, (size_t)n_nodes * sizeof(int)));
            CUDA_SAFE_CALL(cudaMemsetAsync(d_bh_wall, 0, (size_t)n_nodes * sizeof(int)));
            if(d_wp_valid) k_bh_wall_mark<<<blocks_for(n), 256>>>(n, d_wp_valid, d_pbase, d_wp_cell, P.dims, P.bdf2_wall_be, d_bh_wall);
        }
    }
    if(P.ghost_floor > 0.0)
    {   // contact flags are from the previous step's last assembly (lagged by one step)
        CUDA_SAFE_CALL(cudaMemsetAsync(d_mcc, 0, n_cells * sizeof(double)));
        k_p2c_contact<<<blocks_for(n), 256>>>(n, d_xp, d_mp, d_pflag, P.origin, P.dx, P.dims, d_mcc);
        k_node_contact<<<blocks_for(n_nodes), 256>>>(n_nodes, P.dims, d_mcc, d_cell_flag, d_mci);
    }

    thrust::exclusive_scan(thrust::device, thrust::device_pointer_cast(d_cell_flag), thrust::device_pointer_cast(d_cell_flag) + n_cells, thrust::device_pointer_cast(d_cell_scan));
    thrust::exclusive_scan(thrust::device, thrust::device_pointer_cast(d_node_flag), thrust::device_pointer_cast(d_node_flag) + n_vnodes, thrust::device_pointer_cast(d_node_scan));
    k_compact_cells<<<blocks_for(n_cells), 256>>>(n_cells, d_cell_flag, d_cell_scan, d_active_cells);
    k_compact_nodes<<<blocks_for(n_vnodes), 256>>>(n_vnodes, d_node_flag, d_node_scan, d_active_nodes, d_node_slot);
    k_counts<<<1, 1>>>(n_cells, n_vnodes, d_cell_flag, d_cell_scan, d_node_flag, d_node_scan, d_counts);
    int counts[2] = {0, 0};
    CUDA_SAFE_CALL(cudaMemcpy(counts, d_counts, 2 * sizeof(int), cudaMemcpyDeviceToHost));
    h_active_cells = counts[0];
    h_active_nodes = counts[1];
    if(h_active_nodes > P.node_pool)
    {
        char buf[256];
        snprintf(buf, sizeof(buf), "[liquid] %d active grid nodes exceed the node pool (%d): raise node_pool", h_active_nodes, P.node_pool);
        throw std::runtime_error(buf);
    }
    if(h_active_cells > max_voxels)
        throw std::runtime_error("[liquid] active voxels exceed max_voxels");
    if(P.j_mode == 4)  // geometric fill -> rewrites d_Jbc of the active cells (before k_vox_data copies it)
        fill_fraction_pass();
    h_elements = h_active_cells; h_ghost_slots = 0; h_live_nodes = h_active_nodes;
    if(split && h_active_cells > 0)
    {   // 2b: elements per (cell, side) and the private ghost corners they need
        k_elem_plan<<<blocks_for(h_active_cells), 256>>>(h_active_cells, d_active_cells, d_mc, d_Vc, d_mc2, d_Vc2, P.cpic_pghost ? d_corner_vote : nullptr, d_node_slot, P.dims,
                                                           P.origin, P.dx, d_elem_cnt, d_ghost_cnt);
        thrust::exclusive_scan(thrust::device, thrust::device_pointer_cast(d_elem_cnt), thrust::device_pointer_cast(d_elem_cnt) + h_active_cells, thrust::device_pointer_cast(d_elem_off));
        thrust::exclusive_scan(thrust::device, thrust::device_pointer_cast(d_ghost_cnt), thrust::device_pointer_cast(d_ghost_cnt) + h_active_cells, thrust::device_pointer_cast(d_ghost_off));
        k_count1<<<1, 1>>>(h_active_cells, d_elem_cnt, d_elem_off, d_counts);
        k_count1<<<1, 1>>>(h_active_cells, d_ghost_cnt, d_ghost_off, d_counts + 1);
        int tot[2] = {0, 0};
        CUDA_SAFE_CALL(cudaMemcpy(tot, d_counts, 2 * sizeof(int), cudaMemcpyDeviceToHost));
        h_elements = tot[0]; h_ghost_slots = tot[1]; h_live_nodes = h_active_nodes + h_ghost_slots;
        if(h_elements > max_voxels) throw std::runtime_error("[liquid] per-side voxel elements exceed max_voxels");
        if(h_active_nodes + h_ghost_slots > P.node_pool)
        {
            char buf[256];
            snprintf(buf, sizeof(buf), "[liquid] %d active nodes + %d ghost corners exceed the node pool (%d): raise node_pool", h_active_nodes, h_ghost_slots, P.node_pool);
            throw std::runtime_error(buf);
        }
        // build the elements now: the private ghost corners must be known before the node slots are written
        k_vox_build<<<blocks_for(h_active_cells), 256>>>(h_active_cells, d_active_cells, d_mc, d_Vc, d_Jbc, d_mc2, d_Vc2, d_Jbc2, P.cpic_pghost ? d_corner_vote : nullptr, d_node_slot,
                                                          P.dims, P.origin, P.dx, node_slot_base, h_active_nodes, d_elem_off, d_ghost_off, d_vox_nodes, d_vox_V, d_vox_Jb,
                                                          d_vox_ijk, d_ghost_corner);
        // orphan rule: active nodes referenced by no element are fixed this step
        CUDA_SAFE_CALL(cudaMemsetAsync(d_slot_used, 0, (size_t)P.node_pool * sizeof(int)));
        if(h_elements > 0) k_mark_used<<<blocks_for(8 * h_elements), 256>>>(h_elements, d_vox_nodes, node_slot_base, h_active_nodes, d_slot_used);
        CUDA_SAFE_CALL(cudaMemsetAsync(d_counters, 0, sizeof(int)));
        if(h_active_nodes > 0) k_orphan_ghost<<<blocks_for(h_active_nodes), 256>>>(h_active_nodes, d_active_nodes, d_slot_used, d_node_ghost, d_counters);
        CUDA_SAFE_CALL(cudaMemcpy(&h_orphans, d_counters, sizeof(int), cudaMemcpyDeviceToHost));
    }

    if(P.wb && !split && h_active_cells > 0 && d_wp_valid && d_cstart)
    {   // LIQ_WB: well-balanced cut cells (liquid_wb.cuh). Nodal wall distance from this step's wall points, then per active cell the
        // fluid sub-volume operator (a_q, rho (b - beta)_q, w_q) and the consistent nodal weights the node slots use for gravity.
        using namespace gipc_liquid_wb;
        if(!d_wb_phi)
        {
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_wb_phi, (size_t)n_nodes * sizeof(double)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_node_w, (size_t)n_nodes * sizeof(double)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_wb_cnt, 8 * sizeof(int)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_wb_n, (size_t)n_nodes * sizeof(double3)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_wb_pmin, (size_t)n_nodes * sizeof(int)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_wb_mfl, (size_t)n_nodes * sizeof(double)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_wb_frz, (size_t)n_nodes * sizeof(int)));
            CUDA_SAFE_CALL(cudaMemset(d_wb_frz, 0, (size_t)n_nodes * sizeof(int)));
            if(P.wb_slip)
            {   // the grid wall BC buffers (k_wall_assemble / k_wall_energy), normally allocated with analytic containers
                if(!d_wall_count) CUDA_SAFE_CALL(cudaMalloc((void**)&d_wall_count, sizeof(int)));
                if(!d_wall_t)
                {
                    CUDA_SAFE_CALL(cudaMalloc((void**)&d_wall_t, P.node_pool * sizeof(int)));
                    CUDA_SAFE_CALL(cudaMemset(d_wall_t, 0, P.node_pool * sizeof(int)));
                    CUDA_SAFE_CALL(cudaMalloc((void**)&d_wall_n, P.node_pool * sizeof(double3)));
                    CUDA_SAFE_CALL(cudaMalloc((void**)&d_wall_xt, P.node_pool * sizeof(double3)));
                    CUDA_SAFE_CALL(cudaMalloc((void**)&d_wall_red, (blocks_for(P.node_pool) + 1) * sizeof(double)));
                }
            }
            if(!d_vox_cc) CUDA_SAFE_CALL(cudaMalloc((void**)&d_vox_cc, (size_t)max_voxels * sizeof(int)));
            if(!d_vox_a) CUDA_SAFE_CALL(cudaMalloc((void**)&d_vox_a, 8 * (size_t)max_voxels * sizeof(double3)));
            if(!d_vox_vg) CUDA_SAFE_CALL(cudaMalloc((void**)&d_vox_vg, (size_t)max_voxels * sizeof(double)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_vox_f, 8 * (size_t)max_voxels * sizeof(double3)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_vox_wc, (size_t)max_voxels * sizeof(int)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_vox_yc, (size_t)max_voxels * sizeof(double)));
            if(P.wb_surfk > 0.0) { CUDA_SAFE_CALL(cudaMalloc((void**)&d_vox_s, 36 * (size_t)max_voxels * sizeof(double))); printf("[liquid] WB free-surface gravity stiffness ON: %.2f rho g int (u.up)^2 dA over the surface faces (snap %.3f dx)\n", P.wb_surfk, P.wb_snap); }
            if(P.wb_hg > 0.0) { CUDA_SAFE_CALL(cudaMalloc((void**)&d_wb_hg, 150 * (size_t)max_voxels * sizeof(double))); printf("[liquid] WB sub-cell volumetric consistency ON: %.2f kappa over the 6 Kuhn-tet pieces of every cut cell\n", P.wb_hg); }
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_vox_w, 8 * (size_t)max_voxels * sizeof(double)));
            printf("[liquid] well-balanced cut cells ON (LIQ_WB): wall = nodal distance from the wall points, free surface = per-cell plane "
                   "(gate %.2f); pressure areas, hydrostatic term and nodal weights from one fluid sub-volume per cell\n", P.wb_gate);
        }
        CUDA_SAFE_CALL(cudaMemsetAsync(d_node_w, 0, (size_t)n_nodes * sizeof(double)));
        CUDA_SAFE_CALL(cudaMemsetAsync(d_wb_mfl, 0, (size_t)n_nodes * sizeof(double)));
        CUDA_SAFE_CALL(cudaMemsetAsync(d_wb_cnt, 0, 6 * sizeof(int)));
        const bool wb_thinw = P.wb_phi_sign == 11 && d_tri_start && d_tri_list && n_particles > 0;
        if(wb_thinw)
        {
            if(wb_thin_cap < n_particles)
            {
                if(d_wb_thin) cudaFree(d_wb_thin);
                wb_thin_cap = n_particles + n_particles / 4 + 1024;
                CUDA_SAFE_CALL(cudaMalloc((void**)&d_wb_thin, (size_t)wb_thin_cap * sizeof(int)));
            }
            k_wb_thin<<<blocks_for(n_particles), 256>>>(n_particles, d_wp_valid, d_wp_near, d_wp_X, d_wp_n, d_wp_tri, P.origin, P.dx, P.dims, d_tri_start, d_tri_list,
                                                         d_faces, m.vertexes, P.wb_thin * P.dx, d_wb_thin);
        }
        const bool wb_conn = P.wb_phi_sign == 9 && d_tri_start && d_tri_list && h_active_nodes > 0;
        if(wb_conn)
        {   // LIQ_WB_PHI_SIGN=9: open-space nodes (see their nearest particle), connected through grid edges that cross no solid; each particle's
            // water = the label of the nearest open corner of its cell that it sees
            if(!d_wb_lab)
            {
                CUDA_SAFE_CALL(cudaMalloc((void**)&d_wb_lab, (size_t)n_nodes * sizeof(int)));
                CUDA_SAFE_CALL(cudaMalloc((void**)&d_wb_blk, (size_t)n_nodes));
                CUDA_SAFE_CALL(cudaMalloc((void**)&d_wb_ccflag, sizeof(int)));
            }
            if(wb_pcomp_cap < n_particles)
            {
                if(d_wb_pcomp) cudaFree(d_wb_pcomp);
                wb_pcomp_cap = n_particles + n_particles / 4 + 1024;
                CUDA_SAFE_CALL(cudaMalloc((void**)&d_wb_pcomp, (size_t)wb_pcomp_cap * sizeof(int)));
            }
            CUDA_SAFE_CALL(cudaMemsetAsync(d_wb_lab, 0xff, (size_t)n_nodes * sizeof(int)));
            k_wb_open<<<blocks_for(h_active_nodes), 256>>>(h_active_nodes, d_active_nodes, P.dims, P.origin, P.dx, d_cstart, d_cend, d_pidx, d_xp,
                                                            d_tri_start, d_tri_list, d_faces, m.vertexes, d_wb_lab);
            k_wb_edges<<<blocks_for(h_active_nodes), 256>>>(h_active_nodes, d_active_nodes, P.dims, P.origin, P.dx, d_wb_lab, d_tri_start, d_tri_list,
                                                             d_faces, m.vertexes, d_wb_blk);
            int it = 0;
            for(; it < 64; ++it)
            {
                CUDA_SAFE_CALL(cudaMemsetAsync(d_wb_ccflag, 0, sizeof(int)));
                for(int r = 0; r < 8; ++r)
                    k_wb_cc_iter<<<blocks_for(h_active_nodes), 256>>>(h_active_nodes, d_active_nodes, P.dims, d_wb_blk, d_wb_lab, d_wb_ccflag);
                int changed = 0;
                CUDA_SAFE_CALL(cudaMemcpy(&changed, d_wb_ccflag, sizeof(int), cudaMemcpyDeviceToHost));
                if(!changed) break;
            }
            if(it == 64) printf("[liquid] LIQ_WB_PHI_SIGN=9: open-space labels did not converge in 512 sweeps (step %d)\n", steps_done);
            k_wb_pcomp<<<blocks_for(n_particles), 256>>>(n_particles, d_xp, P.origin, P.dx, P.dims, d_wb_lab, d_tri_start, d_tri_list, d_faces, m.vertexes,
                                                          d_wb_pcomp);
        }
        k_wb_phi<<<blocks_for(h_active_nodes), 256>>>(h_active_nodes, d_active_nodes, P.dims, P.origin, P.dx, d_cstart, d_cend, d_pidx, d_wp_valid, d_wp_near,
                                                       d_wp_X, d_wp_n, P.wb_lat * P.wb_lat * P.dx * P.dx, d_wb_phi, d_wb_n, d_wb_pmin, P.wb_phi_euclid, P.wb_hole_lat * P.wb_hole_lat * P.dx * P.dx, d_wp_tri, d_faces, m.vertexes,
                                                       P.wb_phi_sign, P.wb_phi_tri, P.wb_phi_dtol * P.dx,
                                                       P.wb_phi_freeze > 0 ? d_wb_frz : nullptr, (P.wb_phi_freeze > 0 && steps_done == P.wb_phi_freeze) ? 1 : 0,
                                                       d_xp, d_tri_start, d_tri_list, P.wb_phi_ghost, P.wb_phi_ghost_side,
                                                       wb_conn ? d_wb_lab : nullptr, wb_conn ? d_wb_pcomp : nullptr, wb_thinw ? d_wb_thin : nullptr, P.wb_vis_r);
        if(P.wb_phi_freeze > 0 && steps_done == P.wb_phi_freeze) printf("[liquid] LIQ_WB_PHI_FREEZE: nodal wall distances frozen at step %d\n", steps_done);
        {   // per-cell geometry, per-column water height, then the operator
            const double  gm = sqrt(gravity.x * gravity.x + gravity.y * gravity.y + gravity.z * gravity.z);
            const double3 up = gm > 0.0 ? make_double3(-gravity.x / gm, -gravity.y / gm, -gravity.z / gm) : make_double3(0, 0, 0);
            const double  ax = fabs(up.x), ay = fabs(up.y), az = fabs(up.z);
            const int     U  = (ay > ax && ay >= az) ? 1 : ((az > ax && az > ay) ? 2 : 0);
            const double  uc = (U == 0) ? up.x : (U == 1) ? up.y : up.z;
            const int     sg = uc >= 0.0 ? 1 : -1;
            const int     ncol = (U == 0) ? P.dims.y * P.dims.z : (U == 1) ? P.dims.x * P.dims.z : P.dims.x * P.dims.y;
            if(!d_wb_cflag)
            {
                CUDA_SAFE_CALL(cudaMalloc((void**)&d_wb_cflag, (size_t)n_cells * sizeof(int)));
                CUDA_SAFE_CALL(cudaMalloc((void**)&d_wb_dead, (size_t)n_cells * sizeof(int)));
                CUDA_SAFE_CALL(cudaMalloc((void**)&d_wb_cvgeo, (size_t)n_cells * sizeof(double)));
                CUDA_SAFE_CALL(cudaMalloc((void**)&d_wb_hcol, (size_t)ncol * sizeof(double)));
                CUDA_SAFE_CALL(cudaMalloc((void**)&d_wb_hraw, (size_t)ncol * sizeof(double)));
                CUDA_SAFE_CALL(cudaMalloc((void**)&d_wb_blo, (size_t)ncol * sizeof(int)));
                CUDA_SAFE_CALL(cudaMalloc((void**)&d_wb_hwt, (size_t)ncol * sizeof(double)));
                CUDA_SAFE_CALL(cudaMalloc((void**)&d_wb_colD, (size_t)ncol * sizeof(double)));
                CUDA_SAFE_CALL(cudaMalloc((void**)&d_wb_colG, (size_t)ncol * sizeof(double)));
                CUDA_SAFE_CALL(cudaMalloc((void**)&d_wb_ovh, (size_t)ncol * sizeof(int)));
                CUDA_SAFE_CALL(cudaMalloc((void**)&d_wb_ovr, (size_t)ncol * sizeof(int)));
                CUDA_SAFE_CALL(cudaMalloc((void**)&d_wb_ovr2, (size_t)ncol * sizeof(int)));
                CUDA_SAFE_CALL(cudaMalloc((void**)&d_wb_hcol2, (size_t)ncol * sizeof(double)));
                CUDA_SAFE_CALL(cudaMemset(d_wb_ovh, 0, (size_t)ncol * sizeof(int)));
                CUDA_SAFE_CALL(cudaMalloc((void**)&d_wb_bhi, (size_t)ncol * sizeof(int)));
            }
            CUDA_SAFE_CALL(cudaMemsetAsync(d_wb_cflag, 0, (size_t)n_cells * sizeof(int)));
            CUDA_SAFE_CALL(cudaMemsetAsync(d_wb_dead, 0, (size_t)n_cells * sizeof(int)));
            k_wb_cellgeo<<<blocks_for(h_active_cells), 256>>>(h_active_cells, d_active_cells, P.dims, P.origin, P.dx, d_wb_phi, up, gravity, d_wb_cflag, d_wb_cvgeo);
            k_wb_column<<<blocks_for(ncol), 256>>>(ncol, P.dims, P.origin, P.dx, U, sg, d_wb_cflag, d_wb_cvgeo, d_Vc, d_wb_phi, up, gravity, d_xp, d_cstart, d_cend, d_pidx,
                                                    h_spacing, P.wb_topk, d_wb_hraw, d_wb_blo, d_wb_bhi, P.wb_hmode, P.wb_hwin, P.wb_colmin,
                                                    P.wb_hsmooth_w ? d_wb_hwt : nullptr,
                                                    (P.wb_colmin > 0.0 && P.wb_colmin_pool) ? d_wb_colD : nullptr, (P.wb_colmin > 0.0 && P.wb_colmin_pool) ? d_wb_colG : nullptr,
                                                    (P.wb_hcol_freeze > 0 && steps_done > P.wb_hcol_freeze) ? 1 : 0, P.wb_hsmooth_w >= 2 ? 2 : 1, P.wb_cap,
                                                    P.wb_ovh > 0.0 ? d_wb_ovh : nullptr, P.wb_ovh, P.wb_ovh_cos, d_tri_start, d_tri_list, d_faces, m.vertexes, P.wb_trim);
            if(P.wb_colmin > 0.0 && P.wb_colmin_pool)
                k_wb_colguard<<<blocks_for(ncol), 256>>>(ncol, P.dims, U, d_wb_colD, d_wb_colG, P.wb_colmin, d_wb_hraw, d_wb_blo, d_wb_bhi);
            if(P.wb_hsmooth)
            {
                const bool ovh = P.wb_ovh > 0.0;
                k_wb_hsmooth<<<blocks_for(ncol), 256>>>(ncol, P.dims, U, P.wb_hsmooth, d_wb_hraw, d_wb_blo, d_wb_hcol, P.wb_hsmooth_w ? d_wb_hwt : nullptr,
                                                         P.wb_hsmooth_w == 3 ? 1 : 0, ovh ? d_wb_ovh : nullptr, ovh ? d_wb_ovr : nullptr);
                if(ovh)
                    for(int pass = 0; pass < 2; ++pass)
                    {   // LIQ_WB_OVH: 2 x 2 Jacobi passes (ping-pong) carry the surface up to 4 flagged columns deep under an overhang
                        k_wb_ovh_fill<<<blocks_for(ncol), 256>>>(ncol, P.dims, U, d_wb_hraw, d_wb_blo, d_wb_ovh, d_wb_hcol, d_wb_ovr, d_wb_hcol2, d_wb_ovr2);
                        k_wb_ovh_fill<<<blocks_for(ncol), 256>>>(ncol, P.dims, U, d_wb_hraw, d_wb_blo, d_wb_ovh, d_wb_hcol2, d_wb_ovr2, d_wb_hcol, d_wb_ovr);
                    }
            }
            else CUDA_SAFE_CALL(cudaMemcpyAsync(d_wb_hcol, d_wb_hraw, (size_t)ncol * sizeof(double), cudaMemcpyDeviceToDevice));
            k_wb_geom<<<blocks_for(h_active_cells), 256>>>(h_active_cells, d_active_cells, P.dims, P.origin, P.dx, U, sg, d_wb_phi, d_Vc, d_wb_cflag, d_wb_hcol, d_wb_blo,
                                                            d_wb_bhi, gravity, P.rho, d_vox_cc, d_vox_a, d_vox_vg, d_vox_f, d_vox_w, d_node_w, d_wb_cnt,
                                                            steps_done == 0 ? (int)liquid_env("LIQ_WB_DUMP", 0) : 0, P.wb_surf, d_wb_dead, d_wb_hg, d_vox_s, P.wb_snap, P.wb_hg_wallonly, P.wb_topface, d_vox_wc,
                                                            P.wb_ybar ? d_vox_yc : nullptr, P.wb_fullf);
        }
        h_ag = 0;
        if(P.wb_aggr > 0.0 && h_active_nodes > 0)
        {   // LIQ_WB_AGGR: poorly supported nodes -> extrapolation of a well-supported root cell (liquid_wb.cuh)
            if(!d_wb_agroot)
            {
                CUDA_SAFE_CALL(cudaMalloc((void**)&d_wb_agroot, (size_t)n_nodes * sizeof(int)));
                CUDA_SAFE_CALL(cudaMalloc((void**)&d_wb_agtype, (size_t)n_nodes * sizeof(int)));
                CUDA_SAFE_CALL(cudaMalloc((void**)&d_wb_agflag, (size_t)P.node_pool * sizeof(int)));
                CUDA_SAFE_CALL(cudaMalloc((void**)&d_wb_agscan, (size_t)P.node_pool * sizeof(int)));
                printf("[liquid] WB aggregation ON: nodes with fluid share < %.2f rho dx^3 follow the trilinear extrapolation of a root cell (search depth %d, on-wall tangential %d, ghosts %d)\n", P.wb_aggr, P.wb_aggr_depth, P.wb_aggr_wall, P.wb_aggr_ghost);
            }
            CUDA_SAFE_CALL(cudaMemsetAsync(d_wb_cnt + 6, 0, sizeof(int)));
            k_wb_aggr_mark<<<blocks_for(h_active_nodes), 256>>>(h_active_nodes, d_active_nodes, P.dims, P.dx, d_wb_phi, d_wb_n, P.wb_slip_phi * P.dx, d_node_w,
                                                                 P.wb_aggr * P.rho * P.dx * P.dx * P.dx, P.wb_aggr_depth, P.wb_aggr_wall, P.wb_aggr_ghost, d_wb_cflag,
                                                                 d_wb_agroot, d_wb_agtype, d_wb_cnt + 6);
            CUDA_SAFE_CALL(cudaMemcpy(&h_ag, d_wb_cnt + 6, sizeof(int), cudaMemcpyDeviceToHost));
            if(h_ag > 0)
            {
                if(h_ag > ag_cap)
                {
                    if(d_wb_agn) cudaFree(d_wb_agn); if(d_wb_agr) cudaFree(d_wb_agr); if(d_wb_agc) cudaFree(d_wb_agc); if(d_wb_agred) cudaFree(d_wb_agred);
                    if(d_wb_agt) cudaFree(d_wb_agt); if(d_wb_agnrm) cudaFree(d_wb_agnrm);
                    ag_cap = (int)(1.5 * h_ag) + 1024;
                    CUDA_SAFE_CALL(cudaMalloc((void**)&d_wb_agn, ag_cap * sizeof(int)));
                    CUDA_SAFE_CALL(cudaMalloc((void**)&d_wb_agr, ag_cap * sizeof(int)));
                    CUDA_SAFE_CALL(cudaMalloc((void**)&d_wb_agt, ag_cap * sizeof(int)));
                    CUDA_SAFE_CALL(cudaMalloc((void**)&d_wb_agnrm, ag_cap * sizeof(double3)));
                    CUDA_SAFE_CALL(cudaMalloc((void**)&d_wb_agc, 8 * (size_t)ag_cap * sizeof(double)));
                    CUDA_SAFE_CALL(cudaMalloc((void**)&d_wb_agred, (blocks_for(ag_cap) + 1) * sizeof(double)));
                }
                k_wb_aggr_flag<<<blocks_for(h_active_nodes), 256>>>(h_active_nodes, d_active_nodes, d_wb_agroot, d_wb_agflag);
                thrust::exclusive_scan(thrust::device, thrust::device_pointer_cast(d_wb_agflag), thrust::device_pointer_cast(d_wb_agflag) + h_active_nodes,
                                       thrust::device_pointer_cast(d_wb_agscan));
                k_wb_aggr_compact<<<blocks_for(h_active_nodes), 256>>>(h_active_nodes, d_active_nodes, d_wb_agroot, d_wb_agtype, d_wb_n, d_wb_agscan, P.dims, d_wb_agn, d_wb_agr, d_wb_agt, d_wb_agnrm, d_wb_agc);
            }
        }
        if(P.wb_nrm_exact > 0.0 && d_tri_start && d_tri_list && d_wall_count && h_active_nodes > 0)
        {
            CUDA_SAFE_CALL(cudaMemsetAsync(d_wall_count, 0, sizeof(int)));
            gipc_liquid_wb::k_wb_nrm_exact<<<blocks_for(h_active_nodes), 256>>>(h_active_nodes, d_active_nodes, P.dims, P.origin, P.dx, d_wb_phi, P.wb_nrm_exact * P.dx,
                                                                                 d_tri_start, d_tri_list, d_faces, m.vertexes, d_wb_n, d_wall_count);
            if(steps_done % 200 == 0)
            {
                int ne = 0;
                CUDA_SAFE_CALL(cudaMemcpy(&ne, d_wall_count, sizeof(int), cudaMemcpyDeviceToHost));
                printf("[liquid] LIQ_WB_NRM_EXACT: %d band nodes take the mesh normal (step %d)\n", ne, steps_done);
            }
        }
        if(P.wb_slip)
        {   // slip wall BC on the nodes on / behind the wall (+ implicit Euler for them under BDF2); k_write_node_slots leaves wall_t alone when LIQ_WB
            CUDA_SAFE_CALL(cudaMemsetAsync(d_wall_t, 0, P.node_pool * sizeof(int)));
            CUDA_SAFE_CALL(cudaMemsetAsync(d_wall_count, 0, sizeof(int)));
            k_wb_slip_mark<<<blocks_for(h_active_nodes), 256>>>(h_active_nodes, d_active_nodes, P.dims, P.origin, P.dx, d_wb_phi, d_wb_n, P.wb_slip_phi * P.dx, (P.wb_slip_phi_mov >= 0.0) ? P.wb_slip_phi_mov * P.dx : -1.0, P.wb_moving_eps * P.wb_moving_eps, P.wb_ghost,
                                                                 d_wb_pmin, d_wp_tri, d_wp_b, d_faces, m.vertexes, d_overt, m.velocities, dt,
                                                                 d_node_w, P.wb_wmin * P.rho * P.dx * P.dx * P.dx, P.wb_near * P.dx, P.wb_mfloor, d_wb_mfl,
                                                                 (P.wb_aggr > 0.0) ? d_wb_agroot : nullptr, (P.wb_aggr > 0.0) ? d_wb_agtype : nullptr,
                                                                 d_wall_t, d_wall_n, d_wall_xt, (P.bdf2 && P.bdf2_wall_be) ? d_bh_wall : nullptr, d_wall_count,
                                                                 P.wall_static_mass > 0.0 ? P.wall_static_mode : 0, P.wall_tang < 1.0);
            CUDA_SAFE_CALL(cudaMemcpy(&h_wb_slip, d_wall_count, sizeof(int), cudaMemcpyDeviceToHost));
            h_wall_nodes = h_wb_slip;
        }
        int cnt[6] = {0, 0, 0, 0, 0, 0};
        CUDA_SAFE_CALL(cudaMemcpy(cnt, d_wb_cnt, 6 * sizeof(int), cudaMemcpyDeviceToHost));
        h_wb = cnt[0]; h_wb_surf = cnt[1]; h_wb_allneg = cnt[2]; h_wb_empty = cnt[3]; h_wb_sep = cnt[4]; h_cc = h_wb;
        if(steps_done % 200 == 0)
            printf("[liquid] well-balanced cells: %d cut of %d active (%d with a free-surface plane, %d outside the wall, %d above the column surface; both dead; %d separate water), %d slip wall nodes, %d aggregated nodes (step %d)\n",
                   h_wb, h_active_cells, h_wb_surf, h_wb_allneg, h_wb_empty, h_wb_sep, h_wb_slip, h_ag, steps_done);
        CUDA_SAFE_CALL(cudaGetLastError());
    }
    if(d_wall_count) CUDA_SAFE_CALL(cudaMemsetAsync(d_wall_count, 0, sizeof(int)));
    k_write_node_slots<<<blocks_for(P.node_pool), 256>>>(P.node_pool, h_active_nodes, d_active_nodes, d_mi, d_mvi, P.dims, P.dx, P.origin, P.park,
                                                         P.mass_floor, P.floor_gravity, P.ghost_floor > 0.0 ? d_mci : nullptr, P.ghost_floor, P.ghost_frac,
                                                         P.cav_on, P.cav[0], P.cav[1], P.cav[2], P.cav[3], P.cav[4], P.cav[5], P.cav[6],
                                                         P.cpic ? d_node_ghost : nullptr, h_ghost_slots, d_ghost_corner, n_nodes,
                                                         (P.wall_bc && d_cont && d_abd_q) ? d_cont : nullptr,
                                                         (P.wall_bc && d_cont && d_abd_q) ? (int)h_cont.size() : 0, d_abd_q, d_abd_qv,
                                                         P.wall_bc ? P.ground_bc : 0, P.clamp_lo_y,
                                                         (wall_active() && !P.wb) ? d_wall_t : nullptr, d_wall_n, d_wall_xt, (P.wb && P.wb_slip) ? nullptr : d_wall_count,
                                                         P.rho, P.rest_k, P.rest_kv, dt, gravity, node_slot_base, m.vertexes, m.o_vertexes, m.velocities,
                                                         m.xTilta, m.masses, m.BoundaryType, P.bdf2 ? d_bh_mvi : nullptr, P.bdf2 ? d_bh_vb : nullptr,
                                                         P.bdf2 ? d_bh_xt : nullptr, (P.bdf2 && P.bdf2_wall_be) ? d_bh_wall : nullptr,
                                                         (P.wb && d_node_w) ? d_node_w : nullptr, (P.wb && P.wb_mfloor > 0.0) ? d_wb_mfl : nullptr, P.wb_airgrav);
    fp_xt = m.xTilta; fp_mass = m.masses;   // LIQ_FORCE_PROBE
    if(P.wb && P.wb_slip && d_wall_t && h_active_nodes > 0)
        gipc_liquid_wb::k_wb_slip_project<<<blocks_for(h_active_nodes), 256>>>(h_active_nodes, node_slot_base, d_wall_t, d_wall_n, m.velocities, P.wall_tang < 1.0);
    if(P.flip > 0.0 && h_active_nodes > 0)
    {
        CUDA_SAFE_CALL(cudaMemsetAsync(d_vi_old, 0, n_vnodes * sizeof(double3)));
        k_read_node_start_vel<<<blocks_for(h_active_nodes), 256>>>(h_active_nodes, d_active_nodes, m.velocities, node_slot_base, d_vi_old);
    }
    if(d_wall_count && (steps_done % 200 == 0))
    {
        CUDA_SAFE_CALL(cudaMemcpy(&h_wall_nodes, d_wall_count, sizeof(int), cudaMemcpyDeviceToHost));
        printf("[liquid] wall BC: %d of %d active grid nodes constrained (step %d)\n", h_wall_nodes, h_active_nodes, steps_done);
    }
    wfc_x = m.vertexes;
    wfc_xt = m.xTilta; wfc_vel = m.velocities;
    d_overt = m.o_vertexes;
    if((P.wfc || P.cpic) && d_faces && n_faces > 0) wfc_points(m);   // bind the wall points to this step's node slots (+ ghost-penalty pairs)
    if(twof && h_active_cells > 0)
    {   // per-side node slots: elements per (cell, side)
        k_elem_plan2<<<blocks_for(h_active_cells), 256>>>(h_active_cells, d_active_cells, d_mc, d_Vc, d_mc2, d_Vc2, d_elem_cnt);
        thrust::exclusive_scan(thrust::device, thrust::device_pointer_cast(d_elem_cnt), thrust::device_pointer_cast(d_elem_cnt) + h_active_cells, thrust::device_pointer_cast(d_elem_off));
        k_count1<<<1, 1>>>(h_active_cells, d_elem_cnt, d_elem_off, d_counts);
        CUDA_SAFE_CALL(cudaMemcpy(&h_elements, d_counts, sizeof(int), cudaMemcpyDeviceToHost));
        if(h_elements > max_voxels) throw std::runtime_error("[liquid] per-side voxel elements exceed max_voxels");
        k_vox_build2<<<blocks_for(h_active_cells), 256>>>(h_active_cells, d_active_cells, d_mc, d_Vc, d_Jbc, d_mc2, d_Vc2, d_Jbc2, d_corner_side, d_node_slot, P.dims,
                                                           node_slot_base, n_nodes, d_elem_off, d_vox_nodes, d_vox_V, d_vox_Jb, d_vox_ijk);
        CUDA_SAFE_CALL(cudaMemsetAsync(d_counters, 0, sizeof(int)));
        k_count_twosided<<<blocks_for(n_nodes), 256>>>(n_nodes, d_node_flag, d_counters);
        CUDA_SAFE_CALL(cudaMemcpy(&h_twosided, d_counters, sizeof(int), cudaMemcpyDeviceToHost));
    }
    else if(!split && h_active_cells > 0)
        k_vox_data<<<blocks_for(h_active_cells), 256>>>(h_active_cells, d_active_cells, d_node_slot, P.dims, d_Vc, d_Jbc, node_slot_base, d_vox_nodes,
                                                        d_vox_V, d_vox_Jb, d_vox_ijk);
    if(P.wb && P.wb_ybar && !split && h_active_cells > 0 && d_ybar && d_vox_yc)
    {
        const double gm = sqrt(last_gravity.x * last_gravity.x + last_gravity.y * last_gravity.y + last_gravity.z * last_gravity.z);
        if(gm > 0.0) k_wb_ybar_fix<<<blocks_for(h_active_cells), 256>>>(h_active_cells, d_active_cells, d_vox_Jb, d_vox_yc, d_ybar, d_Vc, P.rho * gm / P.kappa);
    }
    if(P.wb && P.wb_geov && !split && h_active_cells > 0 && d_vox_w)
    {
        if(!d_vox_Vr) { CUDA_SAFE_CALL(cudaMalloc((void**)&d_vox_Vr, (size_t)max_voxels * sizeof(double))); printf("[liquid] WB geometric rest volume ON (LIQ_WB_GEOV): voxel force = pressure x geometric area\n"); }
        k_wb_vr<<<blocks_for(h_active_cells), 256>>>(h_active_cells, d_vox_w, d_vox_Jb, d_vox_V, d_vox_cc, P.rho, P.dx * P.dx * P.dx, d_vox_Vr,
                                                       P.wb_geov == 2 ? P.wb_geov_fill : 0.0);
    }
    if(P.wb && (P.wb_g2p || P.g2p_fluid || P.g2p_jcell) && !split && h_active_cells > 0)
    {   // LIQ_WB_G2P: cell -> element map for the transfers (non-split path: element v is active cell v)
        if(!d_cell2elem) CUDA_SAFE_CALL(cudaMalloc((void**)&d_cell2elem, (size_t)n_cells * sizeof(int)));
        CUDA_SAFE_CALL(cudaMemsetAsync(d_cell2elem, 0xFF, (size_t)n_cells * sizeof(int)));
        k_cell2elem<<<blocks_for(h_active_cells), 256>>>(h_active_cells, d_active_cells, d_cell2elem);
    }
    if(!P.wb) h_cc = 0;
    if(P.cutcell && h_elements > 0 && d_wp_valid && d_wp_cell && d_wp_X && d_wp_n)
    {   // LIQ_CUTCELL: fluid sub-volume of the wall-cut elements (wall points are from this step's wall pass)
        if(!d_cc_cnt)
        {
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_cc_nsum, n_cells * sizeof(double3)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_cc_xsum, n_cells * sizeof(double3)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_cc_cnt, n_cells * sizeof(int)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_vox_cc, (size_t)max_voxels * sizeof(int)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_vox_a, 8 * (size_t)max_voxels * sizeof(double3)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_vox_vg, (size_t)max_voxels * sizeof(double)));
            printf("[liquid] cut cells ON: wall-cut elements measure only their fluid sub-volume (face-cap area vectors), sub-volume floor %.2f dx^3\n",
                   P.cutcell_vmin);
        }
        CUDA_SAFE_CALL(cudaMemsetAsync(d_cc_nsum, 0, n_cells * sizeof(double3)));
        CUDA_SAFE_CALL(cudaMemsetAsync(d_cc_xsum, 0, n_cells * sizeof(double3)));
        CUDA_SAFE_CALL(cudaMemsetAsync(d_cc_cnt, 0, n_cells * sizeof(int)));
        CUDA_SAFE_CALL(cudaMemsetAsync(d_counters, 0, sizeof(int)));
        k_cc_mark<<<blocks_for(n), 256>>>(n, d_wp_valid, d_wp_cell, P.dims, d_wp_X, d_wp_n, d_cc_nsum, d_cc_xsum, d_cc_cnt);
        k_cc_elem<<<blocks_for(h_elements), 256>>>(h_elements, d_vox_ijk, P.dims, d_cc_nsum, d_cc_xsum, d_cc_cnt, P.origin, P.dx, P.cutcell_vmin, d_vox_cc,
                                                    d_vox_a, d_vox_vg, d_counters);
        CUDA_SAFE_CALL(cudaMemcpy(&h_cc, d_counters, sizeof(int), cudaMemcpyDeviceToHost));
        if(steps_done % 200 == 0) printf("[liquid] cut cells: %d of %d elements use the fluid sub-volume (step %d)\n", h_cc, h_elements, steps_done);
    }
    if(P.j_mode == 2 || P.j_mode == 4)  // diagnostic: particle J := cell volume ratio (the state the voxel energy uses)
        k_particle_Jcell<<<blocks_for(n), 256>>>(n, d_xp, d_Jbc, d_cell_flag, P.origin, P.dx, P.dims, d_Jp);
    if(P.j_mode == 3)  // re-anchor interior particles to the cell ratio (takes effect from the next P2G)
        k_reanchor_J<<<blocks_for(n), 256>>>(n, d_xp, d_mc, d_cell_flag, P.origin, P.dx, P.dims, P.rho * P.dx * P.dx * P.dx, P.J_min, P.reanchor_beta, d_Jp);
    k_write_particle_slots<<<blocks_for(n), 256>>>(n, d_xp, particle_slot_base, m.vertexes, m.o_vertexes, m.xTilta);
    CUDA_SAFE_CALL(cudaGetLastError());
    g_prof.stop(5);
}

void LiquidSystem::apply_particle_positions(double3* vertexes, const double3* temp, double3* xTilta)
{
    const int n = n_particles;
    k_particle_positions<<<blocks_for(n), 256>>>(n, d_pbase, d_pfrac, d_pmask, d_pwinv, d_node_slot, P.dims, node_slot_base, particle_slot_base, vertexes, temp, xTilta, d_xp, d_nref_has, d_nref_X, d_nref_n, (P.cpic && P.cpic_fields == 2) ? n_nodes : 0, d_plabel, d_nlab);
}

void LiquidSystem::apply_particle_movedir(double3* md)
{
    const int n = n_particles;
    k_particle_movedir<<<blocks_for(n), 256>>>(n, d_pbase, d_pfrac, d_pmask, d_pwinv, d_node_slot, P.dims, node_slot_base, particle_slot_base, md, d_xp, d_nref_has, d_nref_X, d_nref_n, (P.cpic && P.cpic_fields == 2) ? n_nodes : 0, d_plabel, d_nlab);
    static const bool dbg = liq_env_flag("LIQ_DEBUG");
    if(dbg)
    {
        int bad = 0;
        CUDA_SAFE_CALL(cudaMemset(d_counters, 0, sizeof(int)));
        k_count_nonfinite3<<<blocks_for(node_slot_base + P.node_pool + n), 256>>>(node_slot_base + P.node_pool + n, md, d_counters);
        CUDA_SAFE_CALL(cudaMemcpy(&bad, d_counters, sizeof(int), cudaMemcpyDeviceToHost));
        std::vector<double3> h(node_slot_base + P.node_pool + n);
        CUDA_SAFE_CALL(cudaMemcpy(h.data(), md, h.size() * sizeof(double3), cudaMemcpyDeviceToHost));
        double mn = 0, mp = 0;
        for(int i = node_slot_base; i < node_slot_base + P.node_pool; ++i) mn = std::max(mn, sqrt(h[i].x * h[i].x + h[i].y * h[i].y + h[i].z * h[i].z));
        for(int i = particle_slot_base; i < particle_slot_base + n; ++i) mp = std::max(mp, sqrt(h[i].x * h[i].x + h[i].y * h[i].y + h[i].z * h[i].z));
        printf("[liq-dbg] moveDir nonfinite %d  max|d| nodes %.3e particles %.3e\n", bad, mn, mp);
    }
}

// ---------------------------------------------------------------------------
// wall field constraint (WFC): the grid displacement field may not move into a solid at the wetted wall
// ---------------------------------------------------------------------------
__device__ __forceinline__ double3 wfc_closest_tri(double3 p, double3 a, double3 b, double3 c, double3& bary)
{   // Ericson, Real-Time Collision Detection 5.1.5
    double3 ab = d3sub(b, a), ac = d3sub(c, a), ap = d3sub(p, a);
    double  d1 = ab.x * ap.x + ab.y * ap.y + ab.z * ap.z, d2 = ac.x * ap.x + ac.y * ap.y + ac.z * ap.z;
    if(d1 <= 0.0 && d2 <= 0.0) { bary = make_double3(1, 0, 0); return a; }
    double3 bp = d3sub(p, b);
    double  d3 = ab.x * bp.x + ab.y * bp.y + ab.z * bp.z, d4 = ac.x * bp.x + ac.y * bp.y + ac.z * bp.z;
    if(d3 >= 0.0 && d4 <= d3) { bary = make_double3(0, 1, 0); return b; }
    double vc = d1 * d4 - d3 * d2;
    if(vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0)
    {
        double v = d1 / (d1 - d3);
        bary     = make_double3(1.0 - v, v, 0.0);
        return d3add(a, d3scale(ab, v));
    }
    double3 cp = d3sub(p, c);
    double  d5 = ab.x * cp.x + ab.y * cp.y + ab.z * cp.z, d6 = ac.x * cp.x + ac.y * cp.y + ac.z * cp.z;
    if(d6 >= 0.0 && d5 <= d6) { bary = make_double3(0, 0, 1); return c; }
    double vb = d5 * d2 - d1 * d6;
    if(vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0)
    {
        double w = d2 / (d2 - d6);
        bary     = make_double3(1.0 - w, 0.0, w);
        return d3add(a, d3scale(ac, w));
    }
    double va = d3 * d6 - d5 * d4;
    if(va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
    {
        double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        bary     = make_double3(0.0, 1.0 - w, w);
        return d3add(b, d3scale(d3sub(c, b), w));
    }
    double den = 1.0 / (va + vb + vc), v = vb * den, w = vc * den;
    bary = make_double3(1.0 - v - w, v, w);
    return d3add(a, d3add(d3scale(ab, v), d3scale(ac, w)));
}
__global__ void k_wfc_tri_bin(int nf, const uint3* faces, const double3* x, double3 org, double dx, int3 d, const int* near, int* cnt,
                              const int* start, int* list, int pass, const int* bodyid, int skip_body)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if(f >= nf) return;
    uint3   F = faces[f];
    if(skip_body >= 0 && bodyid && bodyid[F.x] == skip_body) return;   // porous body: water passes through its surface
    double3 a = x[F.x], b = x[F.y], c = x[F.z];
    double  lo[3] = {fmin(a.x, fmin(b.x, c.x)), fmin(a.y, fmin(b.y, c.y)), fmin(a.z, fmin(b.z, c.z))};
    double  hi[3] = {fmax(a.x, fmax(b.x, c.x)), fmax(a.y, fmax(b.y, c.y)), fmax(a.z, fmax(b.z, c.z))};
    double  o[3]  = {org.x, org.y, org.z};
    int     n[3]  = {d.x, d.y, d.z}, i0[3], i1[3];
    for(int k = 0; k < 3; ++k)
    {
        i0[k] = (int)floor((lo[k] - o[k]) / dx);
        i1[k] = (int)floor((hi[k] - o[k]) / dx);
        if(i1[k] < 0 || i0[k] >= n[k]) return;   // outside the liquid domain
        i0[k] = max(i0[k], 0);
        i1[k] = min(i1[k], n[k] - 1);
    }
    for(int i = i0[0]; i <= i1[0]; ++i)
        for(int j = i0[1]; j <= i1[1]; ++j)
            for(int k = i0[2]; k <= i1[2]; ++k)
            {
                int cid = cell_lin(i, j, k, d);
                if(!near[cid]) continue;
                int s = atomicAdd(&cnt[cid], 1);
                if(pass == 1) list[start[cid] + s] = f;
            }
}
__global__ void k_wfc_points(int n, const double3* xp, const uint3* faces, const double3* x, double3 org, double dx, int3 d, const int* start,
                             const int* list, double rq, double rq_valid, int* valid, int* near, double3* wX, int* wtri, double3* wb, double3* wn, int3* wcell,
                             double3* wxi, double ground_y = -1e300)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    valid[p]  = 0;
    near[p]   = 0;
    double3 q = xp[p];
    int     ci = (int)floor((q.x - org.x) / dx), cj = (int)floor((q.y - org.y) / dx), ck = (int)floor((q.z - org.z) / dx);
    double  best = rq * rq;
    int     bt   = -1;
    double3 bb = make_double3(0, 0, 0), bX = make_double3(0, 0, 0);
    for(int a = -1; a <= 1; ++a)
        for(int b = -1; b <= 1; ++b)
            for(int c = -1; c <= 1; ++c)
            {
                int i = ci + a, j = cj + b, k = ck + c;
                if(i < 0 || j < 0 || k < 0 || i >= d.x || j >= d.y || k >= d.z) continue;
                int cid = cell_lin(i, j, k, d);
                for(int e = start[cid]; e < start[cid + 1]; ++e)
                {
                    int     f = list[e];
                    uint3   F = faces[f];
                    double3 br;
                    double3 X  = wfc_closest_tri(q, x[F.x], x[F.y], x[F.z], br);
                    double3 dv = d3sub(q, X);
                    double  d2 = dv.x * dv.x + dv.y * dv.y + dv.z * dv.z;
                    if(d2 < best) { best = d2; bt = f; bb = br; bX = X; }
                }
            }
    bool gnd = false;
    if(bt < 0)
    {   // no mesh triangle within rq: the analytic ground plane (LIQ_WFC_GROUND; ground_y = -1e300 when off) is this particle's
        // wall, static with normal +y. `near` keeps meaning "a mesh triangle within rq" (cohesion exclusion, CPIC, film push).
        double hg = q.y - ground_y;
        if(!(hg > 1e-9 && hg < rq)) return;
        gnd  = true;
        best = hg * hg;
        bX   = make_double3(q.x, ground_y, q.z);
    }
    if(best < 1e-18) return;
    double  L  = sqrt(best);
    double3 nn = gnd ? make_double3(0, 1, 0) : d3scale(d3sub(q, bX), 1.0 / L);          // wall -> particle: into the fluid
    near[p]    = gnd ? 0 : 1;
    wX[p]      = bX;
    valid[p]   = (L <= rq_valid) ? 1 : 0;
    double  g[3] = {(bX.x - org.x) / dx, (bX.y - org.y) / dx, (bX.z - org.z) / dx};
    int     ii[3];
    double  xi[3];
    int     dd[3] = {d.x, d.y, d.z};
    for(int k = 0; k < 3; ++k)
    {
        ii[k] = (int)floor(g[k]);
        if(ii[k] < 0) ii[k] = 0;
        if(ii[k] > dd[k] - 1) ii[k] = dd[k] - 1;
        xi[k] = fmin(fmax(g[k] - ii[k], 0.0), 1.0);
    }
    wtri[p]  = gnd ? -2 : bt;   // -2: the ground plane
    wb[p]    = bb;
    wn[p]    = nn;
    wcell[p] = make_int3(ii[0], ii[1], ii[2]);
    wxi[p]   = make_double3(xi[0], xi[1], xi[2]);
}
// node binding of the wall points once the node slots of this step are final
__global__ void k_wfc_bind(int n, const int* near, const int3* wcell, const int* node_slot, int3 d, int base, int* wvid, const double3* xp = nullptr,
                           const int* nref_has = nullptr, const double3* nref_X = nullptr, const double3* nref_n = nullptr, int stride = 0, const int* plabel = nullptr, const int* nlab = nullptr,
                           const int* valid = nullptr)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    double3 xq = xp ? xp[p] : make_double3(0, 0, 0);
    if(!near[p] && !(valid && valid[p])) { for(int qc = 0; qc < 8; ++qc) wvid[8 * p + qc] = -1; return; }   // valid && !near: a ground point
    int3 ii = wcell[p];
    for(int qc = 0; qc < 8; ++qc)
    {
        int a = (qc >> 2) & 1, b = (qc >> 1) & 1, c = qc & 1;
        int s = node_slot[vnode(node_lin(ii.x + a, ii.y + b, ii.z + c, d), xq, nref_has, nref_X, nref_n, stride, plabel ? plabel[p] : -1, nlab)];
        wvid[8 * p + qc] = s >= 0 ? base + s : -1;
    }
}
__global__ void k_mark_particle_cells(int n, const double3* xp, double3 org, double dx, int3 d, int* cflag)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    cflag[particle_cell(xp[p], org, dx, d)] = 1;
}
// gap g = n.(u_field(X_w) - u_solid(X_w)) at the current iterate; also returns the 11 vertex ids and weights
__device__ __forceinline__ double wfc_gap(int p, const int* wtri, const double3* wb, const double3* wn, const int3* wcell, const double3* wxi,
                                          const int* wvid, const uint3* faces, const double3* x, const double3* x0, double3 org, double dx, int* v,
                                          double* c)
{
    int3    ce = wcell[p];
    double3 xi = wxi[p], n = wn[p], bb = wb[p];
    double3 uf = make_double3(0, 0, 0);
    for(int qc = 0; qc < 8; ++qc)
    {
        int    a = (qc >> 2) & 1, b = (qc >> 1) & 1, cc = qc & 1;
        double N = (a ? xi.x : 1.0 - xi.x) * (b ? xi.y : 1.0 - xi.y) * (cc ? xi.z : 1.0 - xi.z);
        int    id = wvid[8 * p + qc];
        v[qc] = id;
        c[qc] = id >= 0 ? N : 0.0;
        if(id >= 0 && N > 0.0)
        {
            double3 X = make_double3(org.x + (ce.x + a) * dx, org.y + (ce.y + b) * dx, org.z + (ce.z + cc) * dx);
            uf        = d3add(uf, d3scale(d3sub(x[id], X), N));
        }
    }
    double3 us = make_double3(0, 0, 0);
    if(wtri[p] < 0)
    {   // the ground plane: static, no solid vertices
        for(int j = 0; j < 3; ++j) { v[8 + j] = -1; c[8 + j] = 0.0; }
    }
    else
    {
        uint3  F    = faces[wtri[p]];
        int    t[3] = {(int)F.x, (int)F.y, (int)F.z};
        double bw[3] = {bb.x, bb.y, bb.z};
        for(int j = 0; j < 3; ++j)
        {
            v[8 + j] = t[j];
            c[8 + j] = -bw[j];
            us       = d3add(us, d3scale(d3sub(x[t[j]], x0[t[j]]), bw[j]));
        }
    }
    double3 du = d3sub(uf, us);
    return n.x * du.x + n.y * du.y + n.z * du.z;
}
// predicted gap of the free motion at a wall point: n.(u~_field(X_w) - dt v_solid(X_w)), u~ = inertial predictor of the grid nodes (incl. gravity);
// the wall is taken at its start-of-step velocity WITHOUT its own free fall (a held / supported wall does not fall)
__device__ __forceinline__ double wfc_gap_pred(int p, const int* wtri, const double3* wb, const double3* wn, const int3* wcell, const double3* wxi, const int* wvid,
                                               const uint3* faces, const double3* xt, const double3* vel, double dt, double3 org, double dx)
{
    int3    ce = wcell[p];
    double3 xi = wxi[p], n = wn[p], bb = wb[p];
    double3 uf = make_double3(0, 0, 0);
    for(int qc = 0; qc < 8; ++qc)
    {
        int    a = (qc >> 2) & 1, b = (qc >> 1) & 1, cc = qc & 1;
        double N = (a ? xi.x : 1.0 - xi.x) * (b ? xi.y : 1.0 - xi.y) * (cc ? xi.z : 1.0 - xi.z);
        int    id = wvid[8 * p + qc];
        if(id >= 0 && N > 0.0)
        {
            double3 X = make_double3(org.x + (ce.x + a) * dx, org.y + (ce.y + b) * dx, org.z + (ce.z + cc) * dx);
            uf        = d3add(uf, d3scale(d3sub(xt[id], X), N));
        }
    }
    double3 us = make_double3(0, 0, 0);
    if(wtri[p] >= 0)
    {
        uint3  F    = faces[wtri[p]];
        int    t[3] = {(int)F.x, (int)F.y, (int)F.z};
        double bw[3] = {bb.x, bb.y, bb.z};
        for(int j = 0; j < 3; ++j) us = d3add(us, d3scale(vel[t[j]], bw[j] * dt));
    }
    double3 du = d3sub(uf, us);
    return n.x * du.x + n.y * du.y + n.z * du.z;
}
__global__ void k_wfc_assemble(int n, const int* valid, const int* wtri, const double3* wb, const double3* wn, const int3* wcell, const double3* wxi,
                               const int* wvid, const uint3* faces, const double3* x, const double3* x0, double3 org, double dx, double K,
                               double3* fb, int* cursor, int* rows, int* cols, Eigen::Matrix3d* vals, double3* fsolid, int pslot = -1,
                               int pre = 0, const int* act_prev = nullptr, const double3* xt = nullptr, const double3* vel = nullptr, double dt = 0.0,
                               int* pre_count = nullptr, const double* lam = nullptr)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n || !valid[p]) return;
    int    v[11];
    double c[11];
    double g    = wfc_gap(p, wtri, wb, wn, wcell, wxi, wvid, faces, x, x0, org, dx, v, c);
    if(lam) g -= lam[p] / K;                      // LIQ_WFC_AL: shifted gap, energy K/2 min(0, g - lambda/K)^2
    int    slot = atomicAdd(cursor, 66);
    int    anyv = v[8] >= 0 ? v[8] : pslot + p;   // zero padding blocks sit on a real vertex (a ground point has no triangle: its particle)
    double3 nrm = wn[p];
    double  nv[3] = {nrm.x, nrm.y, nrm.z};
    bool    act = g < 0.0;
    if(!act && pre && g <= 1e-12 * dx)
    {   // LIQ_WFC_PRE: the constraint sits at its boundary (every wall point in the first Newton iteration of a step). Semismooth Newton:
        // the gradient K g c is zero there either way; take the active-side Hessian where the wall is expected to load
        bool a_prev = (pre == 1 || pre == 3) && act_prev && act_prev[p];
        bool a_pred = (pre == 1 || pre == 2) && xt && vel && wfc_gap_pred(p, wtri, wb, wn, wcell, wxi, wvid, faces, xt, vel, dt, org, dx) < 0.0;
        act = a_prev || a_pred;
        if(act && pre_count) atomicAdd(pre_count, 1);
    }
    if(act)
        for(int j = 0; j < 11; ++j)
        {
            if(v[j] < 0 || c[j] == 0.0) continue;
            double s = K * g * c[j];
            atomicAdd(&fb[v[j]].x, s * nrm.x);
            atomicAdd(&fb[v[j]].y, s * nrm.y);
            atomicAdd(&fb[v[j]].z, s * nrm.z);
            if(fsolid && j >= 8)
            {   // the same gradient on the wall triangle's vertices, kept for the force export
                atomicAdd(&fsolid[v[j]].x, s * nrm.x);
                atomicAdd(&fsolid[v[j]].y, s * nrm.y);
                atomicAdd(&fsolid[v[j]].z, s * nrm.z);
            }
        }
    int kk = 0;
    for(int j = 0; j < 11; ++j)
        for(int l = j; l < 11; ++l)
        {
            int o = slot + kk++;
            if(!act || v[j] < 0 || v[l] < 0 || c[j] == 0.0 || c[l] == 0.0)
            {
                rows[o] = anyv; cols[o] = anyv; vals[o].setZero();
                continue;
            }
            int r = v[j], cc = v[l];
            if(r > cc) { int tmp = r; r = cc; cc = tmp; }
            Eigen::Matrix3d H;
            double s = K * c[j] * c[l];
            for(int a = 0; a < 3; ++a)
                for(int b = 0; b < 3; ++b) H(a, b) = s * nv[a] * nv[b];
            rows[o] = r; cols[o] = cc; vals[o] = H;
        }
}
__global__ void k_wfc_energy(int n, const int* valid, const int* wtri, const double3* wb, const double3* wn, const int3* wcell, const double3* wxi,
                             const int* wvid, const uint3* faces, const double3* x, const double3* x0, double3 org, double dx, double Kp, double* part,
                             const double* lam = nullptr, double Kraw = 1.0)
{
    extern __shared__ double sh[];
    int    p = blockIdx.x * blockDim.x + threadIdx.x;
    double e = 0.0;
    if(p < n && valid[p])
    {
        int    v[11];
        double c[11];
        double g = wfc_gap(p, wtri, wb, wn, wcell, wxi, wvid, faces, x, x0, org, dx, v, c);
        if(lam) g -= lam[p] / Kraw;               // LIQ_WFC_AL (the constant -lambda^2 / 2K is dropped: it does not depend on the iterate)
        if(g < 0.0) e = 0.5 * Kp * g * g;
    }
    sh[threadIdx.x] = e;
    __syncthreads();
    for(int k = blockDim.x / 2; k > 0; k >>= 1)
    {
        if(threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k];
        __syncthreads();
    }
    if(threadIdx.x == 0) part[blockIdx.x] = sh[0];
}
// LIQ_WFC_PRE: which wall points carry a wall reaction at the accepted iterate of this step (start-of-step solid positions x0 still in place)
__global__ void k_wfc_final_act(int n, const int* valid, const int* wtri, const double3* wb, const double3* wn, const int3* wcell, const double3* wxi,
                                const int* wvid, const uint3* faces, const double3* x, const double3* x0, double3 org, double dx, int* act,
                                double* lam = nullptr, double K = 1.0)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    int a = 0;
    double l = 0.0;
    if(valid[p])
    {
        int    v[11];
        double c[11];
        double g = wfc_gap(p, wtri, wb, wn, wcell, wxi, wvid, faces, x, x0, org, dx, v, c);
        if(lam)
        {   // multiplier update of the method of multipliers: lambda <- max(0, lambda - K g)
            l = lam[p] - K * g;
            if(!(l > 0.0)) l = 0.0;
            a = l > 0.0 ? 1 : 0;
        }
        else a = g < 0.0 ? 1 : 0;
    }
    act[p] = a;
    if(lam) lam[p] = l;
}


// ---- LIQ_GHOST_MIRROR: solid-side grid nodes slaved to the mirror image of the fluid field ----
// per lattice node: the closest wall point among the wall-point particles whose 27-node stencil holds the node
__global__ void k_gm_key(int n, const int* valid, const int3* pbase, const double3* wX, int3 dims, double3 org, double dx, unsigned long long* key)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n || !valid[p]) return;
    int3    b = pbase[p];
    double3 X = wX[p];
    for(int a = 0; a < 3; ++a)
        for(int bb = 0; bb < 3; ++bb)
            for(int c = 0; c < 3; ++c)
            {
                int ni = b.x + a, nj = b.y + bb, nk = b.z + c;
                if(ni < 0 || nj < 0 || nk < 0 || ni > dims.x || nj > dims.y || nk > dims.z) continue;
                double3 xn = make_double3(org.x + ni * dx, org.y + nj * dx, org.z + nk * dx);
                double  d  = sqrt((xn.x - X.x) * (xn.x - X.x) + (xn.y - X.y) * (xn.y - X.y) + (xn.z - X.z) * (xn.z - X.z)) / dx;
                unsigned long long q = (unsigned long long)fmin(d * 1.0e6, 4.0e9);
                atomicMin(&key[node_lin(ni, nj, nk, dims)], (q << 32) | (unsigned long long)p);
            }
}
// thin walls: a node behind its closest wall plane but IN FRONT of a nearby wall plane facing the other way (n_q . n_closest < 0)
// is a fluid node of the other side -> no ghost (curved one-sided walls have similar normals and are not affected)
__global__ void k_gm_front(int n, const int* valid, const int3* pbase, const double3* wX, const double3* wn, const unsigned long long* key, int3 dims,
                           double3 org, double dx, int* front)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n || !valid[p]) return;
    int3    b = pbase[p];
    double3 X = wX[p], nn = wn[p];
    for(int a = 0; a < 3; ++a)
        for(int bb = 0; bb < 3; ++bb)
            for(int c = 0; c < 3; ++c)
            {
                int ni = b.x + a, nj = b.y + bb, nk = b.z + c;
                if(ni < 0 || nj < 0 || nk < 0 || ni > dims.x || nj > dims.y || nk > dims.z) continue;
                int                nl = node_lin(ni, nj, nk, dims);
                unsigned long long kk = key[nl];
                if(kk == 0xffffffffffffffffULL) continue;
                double3 nc = wn[(int)(kk & 0xffffffffULL)];
                if(nn.x * nc.x + nn.y * nc.y + nn.z * nc.z >= 0.0) continue;
                double3 xn = make_double3(org.x + ni * dx, org.y + nj * dx, org.z + nk * dx);
                if(nn.x * (xn.x - X.x) + nn.y * (xn.y - X.y) + nn.z * (xn.z - X.z) > 1e-9 * dx) atomicOr(&front[nl], 1);
            }
}
// ghost terms: active nodes on or behind the plane of their closest wall point and not on the far side of a thin wall
__global__ void k_gm_list(int n_active, const int* active_nodes, const unsigned long long* key, const int* front, const double3* wX, const double3* wn,
                          int3 dims, double3 org, double dx, int2* list, int cap, int* count)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if(s >= n_active) return;
    int                nid = active_nodes[s];
    unsigned long long kk  = key[nid];
    if(kk == 0xffffffffffffffffULL || front[nid]) return;
    int p = (int)(kk & 0xffffffffULL);
    int i, j, k;
    node_ijk(nid, dims, i, j, k);
    double3 nn = wn[p], X = wX[p];
    double  sd = nn.x * (org.x + i * dx - X.x) + nn.y * (org.y + j * dx - X.y) + nn.z * (org.z + k * dx - X.z);
    if(sd > 1e-9 * dx) return;   // in front of the wall: a fluid node
    int t = atomicAdd(count, 1);
    if(t < cap) list[t] = make_int2(s, p);
}
// one ghost term: up to 12 vertices (ghost, 8 corners of the cell holding the mirror point, 3 wall-triangle vertices) with
// coefficient matrices A_k = alpha_k I + beta_k n n^T, residual r = sum_k A_k u_k = u_g - M u_f(x_m) - 2 n n^T u_wall
__device__ bool gm_term(int t, const int2* list, const int* active_nodes, const int* node_slot, int base, int3 dims, double3 org, double dx,
                        const double3* x, const double3* x0s, const double3* wX, const double3* wn, const int* wtri, const double3* wb, const uint3* faces,
                        int* v, double* al, double* be, double3& nrm, double3& r)
{
    int2 e = list[t];
    int  nid = active_nodes[e.x], p = e.y;
    int  i, j, k;
    node_ijk(nid, dims, i, j, k);
    double3 xg0 = make_double3(org.x + i * dx, org.y + j * dx, org.z + k * dx);
    double3 nn = wn[p], X = wX[p];
    nrm = nn;
    double  sd = nn.x * (xg0.x - X.x) + nn.y * (xg0.y - X.y) + nn.z * (xg0.z - X.z);
    double3 xm = make_double3(xg0.x - 2.0 * sd * nn.x, xg0.y - 2.0 * sd * nn.y, xg0.z - 2.0 * sd * nn.z);
    double  rr[3] = {(xm.x - org.x) / dx, (xm.y - org.y) / dx, (xm.z - org.z) / dx};
    int     bq[3], nd[3] = {dims.x, dims.y, dims.z};
    double  f[3];
    for(int a = 0; a < 3; ++a)
    {
        bq[a] = (int)floor(rr[a]);
        if(bq[a] < 0 || bq[a] > nd[a] - 1) return false;
        f[a] = fmin(fmax(rr[a] - bq[a], 0.0), 1.0);
    }
    v[0] = base + e.x; al[0] = 1.0; be[0] = 0.0;
    double3 xk0[12];
    xk0[0] = xg0;
    for(int q = 0; q < 8; ++q)
    {
        int qa = (q >> 2) & 1, qb = (q >> 1) & 1, qc = q & 1;
        int sl = node_slot[node_lin(bq[0] + qa, bq[1] + qb, bq[2] + qc, dims)];
        if(sl < 0) return false;   // a mirror corner without a node: no term
        double w = (qa ? f[0] : 1.0 - f[0]) * (qb ? f[1] : 1.0 - f[1]) * (qc ? f[2] : 1.0 - f[2]);
        v[1 + q] = base + sl; al[1 + q] = -w; be[1 + q] = 2.0 * w;
        xk0[1 + q] = make_double3(org.x + (bq[0] + qa) * dx, org.y + (bq[1] + qb) * dx, org.z + (bq[2] + qc) * dx);
        if(v[1 + q] == v[0]) { al[0] += -w; be[0] += 2.0 * w; v[1 + q] = -1; }   // the ghost is one of the corners: merge
    }
    for(int jj = 0; jj < 3; ++jj) v[9 + jj] = -1;
    if(wtri[p] >= 0)
    {   // the wall's own motion over the step (two-way: these become body DOFs through the engine's reduction)
        uint3   F = faces[wtri[p]];
        double3 b = wb[p];
        int     tv[3] = {(int)F.x, (int)F.y, (int)F.z};
        double  tb[3] = {b.x, b.y, b.z};
        for(int jj = 0; jj < 3; ++jj) { v[9 + jj] = tv[jj]; al[9 + jj] = 0.0; be[9 + jj] = -2.0 * tb[jj]; xk0[9 + jj] = x0s[tv[jj]]; }
    }
    r = make_double3(0, 0, 0);
    for(int q = 0; q < 12; ++q)
    {
        if(v[q] < 0) continue;
        double3 u  = d3sub(x[v[q]], xk0[q]);
        double  un = nn.x * u.x + nn.y * u.y + nn.z * u.z;
        r = d3add(r, make_double3(al[q] * u.x + be[q] * un * nn.x, al[q] * u.y + be[q] * un * nn.y, al[q] * u.z + be[q] * un * nn.z));
    }
    return true;
}
__global__ void k_gm_assemble(int m, const int2* list, const int* active_nodes, const int* node_slot, int base, int3 dims, double3 org, double dx,
                              const double3* x, const double3* x0s, const double3* wX, const double3* wn, const int* wtri, const double3* wb, const uint3* faces,
                              double K, double3* fb, int live, int* rows, int* cols, Eigen::Matrix3d* vals)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if(t >= m) return;
    int     v[12];
    double  al[12], be[12];
    double3 nn, r;
    bool    ok   = gm_term(t, list, active_nodes, node_slot, base, dims, org, dx, x, x0s, wX, wn, wtri, wb, faces, v, al, be, nn, r);
    int     slot = live + 78 * t;
    int     anyv = base + list[t].x;
    if(!ok)
    {
        for(int o = 0; o < 78; ++o) { rows[slot + o] = anyv; cols[slot + o] = anyv; vals[slot + o].setZero(); }
        return;
    }
    double rn = nn.x * r.x + nn.y * r.y + nn.z * r.z;
    double c  = rn < 0.0 ? 0.0 : 1.0;   // normal part one-sided: only penetration of the mirrored field is resisted
    double3 Dr = d3scale(make_double3(r.x - c * rn * nn.x, r.y - c * rn * nn.y, r.z - c * rn * nn.z), K);
    double  Drn = nn.x * Dr.x + nn.y * Dr.y + nn.z * Dr.z;
    for(int q = 0; q < 12; ++q)
    {
        if(v[q] < 0) continue;
        double3 g = make_double3(al[q] * Dr.x + be[q] * Drn * nn.x, al[q] * Dr.y + be[q] * Drn * nn.y, al[q] * Dr.z + be[q] * Drn * nn.z);
        atomicAdd(&fb[v[q]].x, g.x);
        atomicAdd(&fb[v[q]].y, g.y);
        atomicAdd(&fb[v[q]].z, g.z);
    }
    double nv[3] = {nn.x, nn.y, nn.z};
    int    kk    = 0;
    for(int a = 0; a < 12; ++a)
        for(int b = a; b < 12; ++b)
        {
            int o = slot + kk++;
            if(v[a] < 0 || v[b] < 0) { rows[o] = anyv; cols[o] = anyv; vals[o].setZero(); continue; }
            // A_a^T D A_b = K [al_a al_b I + gam n n^T],  gam = (al_a + be_a)(be_b - c (al_b + be_b)) + be_a al_b
            double gam = (al[a] + be[a]) * (be[b] - c * (al[b] + be[b])) + be[a] * al[b];
            Eigen::Matrix3d H;
            for(int i = 0; i < 3; ++i)
                for(int j = 0; j < 3; ++j) H(i, j) = K * ((i == j ? al[a] * al[b] : 0.0) + gam * nv[i] * nv[j]);
            int rr_ = v[a], cc = v[b];
            if(rr_ > cc) { int tmp = rr_; rr_ = cc; cc = tmp; H.transposeInPlace(); }
            if(rr_ == cc) H = 0.5 * (H + H.transpose().eval()) * (a == b ? 1.0 : 2.0);   // diagonal block of a vertex listed once
            rows[o] = rr_; cols[o] = cc; vals[o] = H;
        }
}
__global__ void k_gm_energy(int m, const int2* list, const int* active_nodes, const int* node_slot, int base, int3 dims, double3 org, double dx,
                            const double3* x, const double3* x0s, const double3* wX, const double3* wn, const int* wtri, const double3* wb, const uint3* faces,
                            double Kp, double* part)
{
    extern __shared__ double sh[];
    int    t = blockIdx.x * blockDim.x + threadIdx.x;
    double e = 0.0;
    if(t < m)
    {
        int     v[12];
        double  al[12], be[12];
        double3 nn, r;
        if(gm_term(t, list, active_nodes, node_slot, base, dims, org, dx, x, x0s, wX, wn, wtri, wb, faces, v, al, be, nn, r))
        {
            double rn = nn.x * r.x + nn.y * r.y + nn.z * r.z;
            double c  = rn < 0.0 ? 0.0 : 1.0;
            e = 0.5 * Kp * (r.x * r.x + r.y * r.y + r.z * r.z - c * rn * rn);
        }
    }
    sh[threadIdx.x] = e;
    __syncthreads();
    for(int k = blockDim.x / 2; k > 0; k >>= 1)
    {
        if(threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k];
        __syncthreads();
    }
    if(threadIdx.x == 0) part[blockIdx.x] = sh[0];
}

// volumetric ghost penalty: pairs (cut cell, active face neighbour); each unordered pair once (owner = lower cell along +axis)
__global__ void k_gp_mark_cut(int n, const int* valid, const int3* wcell, int3 d, int* cut, const double3* wn, double3* cutn)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n || !valid[p]) return;
    int3 c  = wcell[p];
    int  cl = cell_lin(c.x, c.y, c.z, d);
    cut[cl] = 1;
    if(cutn)
    {
        atomicAdd(&cutn[cl].x, wn[p].x);
        atomicAdd(&cutn[cl].y, wn[p].y);
        atomicAdd(&cutn[cl].z, wn[p].z);
    }
}
// wall normal of a face pair (gp_full 2/3): normalised sum over both cells; false if no usable normal
__device__ __forceinline__ bool gp_pair_normal(int2 pr, const double3* cutn, double nv[3])
{
    double3 a = cutn[pr.x], b = cutn[pr.y];
    double  x = a.x + b.x, y = a.y + b.y, z = a.z + b.z, l = sqrt(x * x + y * y + z * z);
    if(l < 1e-12) return false;
    nv[0] = x / l; nv[1] = y / l; nv[2] = z / l;
    return true;
}
// LIQ_GP_REL: per cut cell, the summed velocity of the wall under its wall points (triangle vertices' velocities of the last step)
__global__ void k_gp_cut_vel(int n, const int* valid, const int3* wcell, int3 d, const int* wtri, const double3* wb, const uint3* faces, const double3* vel,
                             double3* cutv, int* cutc)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n || !valid[p]) return;
    int3    c  = wcell[p];
    int     cl = cell_lin(c.x, c.y, c.z, d);
    double3 vw = make_double3(0, 0, 0);
    if(wtri[p] >= 0)
    {   // a ground point has no triangle: the ground is at rest
        uint3   F = faces[wtri[p]];
        double3 b = wb[p];
        vw = d3add(d3add(d3scale(vel[F.x], b.x), d3scale(vel[F.y], b.y)), d3scale(vel[F.z], b.z));
    }
    atomicAdd(&cutv[cl].x, vw.x);
    atomicAdd(&cutv[cl].y, vw.y);
    atomicAdd(&cutv[cl].z, vw.z);
    atomicAdd(&cutc[cl], 1);
}
__global__ void k_gp_flag(int ncell, int3 d, const int* cflag, const int* cut, const double3* cutn, int* flag, const int* dead = nullptr)
{   // flag[3*c + a] = 1 if the pair (c, c + e_a) qualifies: both active and at least one cut; with cutn (CPIC): not when
    // both are cut by walls facing opposite ways (the pair would straddle a thin wall and couple its two sides)
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if(c >= ncell) return;
    int i, j, k;
    cell_ijk(c, d, i, j, k);
    for(int a = 0; a < 3; ++a)
    {
        int ii = i + (a == 0), jj = j + (a == 1), kk = k + (a == 2);
        int f  = 0;
        if(ii < d.x && jj < d.y && kk < d.z && cflag[c])
        {
            int c2 = cell_lin(ii, jj, kk, d);
            f      = cflag[c2] && (cut[c] || cut[c2]);
            if(f && dead && (dead[c] || dead[c2])) f = 0;   // LIQ_WB: a dead element (behind the wall, kernel spill) is not fluid: no jump to penalise
            if(f && cutn && cut[c] && cut[c2])
            {
                double3 u = cutn[c], w = cutn[c2];
                if(u.x * w.x + u.y * w.y + u.z * w.z < 0.0) f = 0;
            }
        }
        flag[3 * c + a] = f;
    }
}
__global__ void k_gp_compact(int n3, const int* flag, const int* scan, int3 d, int2* pairs)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if(t >= n3 || !flag[t]) return;
    int c = t / 3, a = t % 3, i, j, k;
    cell_ijk(c, d, i, j, k);
    pairs[scan[t]] = make_int2(c, cell_lin(i + (a == 0), j + (a == 1), k + (a == 2), d));
}
// the 12 nodes of a face pair and their trace weights d_i = grad w_ic - grad w_ic'  (tr G = sum_i u_i . grad w_ic)
__device__ __forceinline__ void gp_stencil(int2 pr, int3 d, const int* node_slot, int base, double dx, double3 org, int* v, double3* w, double3* X, int stride = 0, const int* plabel = nullptr, const int* nlab = nullptr)
{
    int i0, j0, k0, i1, j1, k1;
    cell_ijk(pr.x, d, i0, j0, k0);
    cell_ijk(pr.y, d, i1, j1, k1);
    int    ax = (i1 != i0) ? 0 : ((j1 != j0) ? 1 : 2);
    double s  = 1.0 / (4.0 * dx);
    // union of corners: node (i0..i0+1+[ax==0], ...) -> 12 nodes laid out as 3 x 2 x 2 along the pair axis
    int n = 0;
    for(int a = 0; a <= 1 + (ax == 0); ++a)
        for(int b = 0; b <= 1 + (ax == 1); ++b)
            for(int c = 0; c <= 1 + (ax == 2); ++c)
            {
                int    ni = i0 + a, nj = j0 + b, nk = k0 + c;
                int    sl = node_slot[node_lin(ni, nj, nk, d)];
                if(sl < 0 && stride > 0) sl = node_slot[node_lin(ni, nj, nk, d) + stride];   // per-side slots: the populated copy
                double g0[3] = {0, 0, 0}, g1[3] = {0, 0, 0};
                int    la[3] = {a, b, c};
                // corner of cell 0 if all local indices in {0,1}
                bool in0 = la[0] <= 1 && la[1] <= 1 && la[2] <= 1;
                // corner of cell 1: shift along ax by one
                int  lb[3] = {a - (ax == 0), b - (ax == 1), c - (ax == 2)};
                bool in1 = lb[0] >= 0 && lb[1] >= 0 && lb[2] >= 0 && lb[0] <= 1 && lb[1] <= 1 && lb[2] <= 1;
                for(int q = 0; q < 3; ++q)
                {
                    if(in0) g0[q] = (2 * la[q] - 1) * s;
                    if(in1) g1[q] = (2 * lb[q] - 1) * s;
                }
                v[n] = sl >= 0 ? base + sl : -1;
                w[n] = make_double3(g0[0] - g1[0], g0[1] - g1[1], g0[2] - g1[2]);
                X[n] = make_double3(org.x + ni * dx, org.y + nj * dx, org.z + nk * dx);
                ++n;
            }
}
// gp_full 6: per-pair full-gradient weight from the start-of-step transferred node velocities (max over the 12 pair nodes)
__global__ void k_gp_weight(int np, const int2* pairs, int3 d, const double* mi, const double3* mvi, double vlo, double vhi, double* wout, int stride = 0, const int* plabel = nullptr, const int* nlab = nullptr,
                            const double3* cutv = nullptr, const int* cutc = nullptr)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if(t >= np) return;
    int i0, j0, k0, i1, j1, k1;
    cell_ijk(pairs[t].x, d, i0, j0, k0);
    cell_ijk(pairs[t].y, d, i1, j1, k1);
    int    ax   = (i1 != i0) ? 0 : ((j1 != j0) ? 1 : 2);
    double vmax = 0.0;
    double3 vref = make_double3(0, 0, 0);   // LIQ_GP_REL: the wall's velocity at this pair (mean over the cut cell(s) of the pair)
    if(cutv && cutc)
    {
        int c0 = cutc[pairs[t].x], c1 = cutc[pairs[t].y];
        if(c0 + c1 > 0) vref = d3scale(d3add(cutv[pairs[t].x], cutv[pairs[t].y]), 1.0 / (c0 + c1));
    }
    for(int a = 0; a <= 1 + (ax == 0); ++a)
        for(int b = 0; b <= 1 + (ax == 1); ++b)
            for(int c = 0; c <= 1 + (ax == 2); ++c)
            {
                int    nid = node_lin(i0 + a, j0 + b, k0 + c, d);
                double m   = mi[nid];
                if(m <= 0.0 && stride > 0) { nid += stride; m = mi[nid]; }   // per-side slots: the populated copy
                if(m <= 0.0) continue;
                double3 v = d3sub(d3scale(mvi[nid], 1.0 / m), vref);
                vmax      = fmax(vmax, sqrt(v.x * v.x + v.y * v.y + v.z * v.z));
            }
    double wf;
    if(vmax <= vlo) wf = 1.0;
    else if(vmax >= vhi) wf = 0.0;
    else
    {
        double u = (vmax - vlo) / (vhi - vlo);
        wf       = 1.0 - u * u * (3.0 - 2.0 * u);
    }
    wout[t] = wf;
}
__global__ void k_gp_assemble(int np, const int2* pairs, int3 d, const int* node_slot, int base, double dx, double3 org, const double3* x, double K,
                              int full, const double3* cutn, const double* gpw, double3* fb, int* cursor, int* rows, int* cols, Eigen::Matrix3d* vals, int stride = 0, const int* plabel = nullptr, const int* nlab = nullptr)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if(t >= np) return;
    int     v[12];
    double3 w[12], X[12];
    gp_stencil(pairs[t], d, node_slot, base, dx, org, v, w, X, stride);
    // dG = sum_i u_i (x) w_i  (3x3);  trace mode uses only tr(dG)
    double G[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    for(int i = 0; i < 12; ++i)
        if(v[i] >= 0)
        {
            double3 u = d3sub(x[v[i]], X[i]);
            double  uu[3] = {u.x, u.y, u.z}, ww[3] = {w[i].x, w[i].y, w[i].z};
            for(int a = 0; a < 3; ++a)
                for(int b = 0; b < 3; ++b) G[a][b] += uu[a] * ww[b];
        }
    double dtr = G[0][0] + G[1][1] + G[2][2];
    // gp_full 2: only the gradient of the wall-NORMAL displacement (n^T dG): normal motion of the ghost
    // nodes must extend the fluid's smoothly, tangential slip / shear along the wall stays free.
    // gp_full 3: only the normal stretch n^T dG n (the pivot-into-the-wall relief mode).
    double nv[3] = {0, 0, 0}, gn[3] = {0, 0, 0}, snn = 0.0;
    int    mode = full;
    // gp_full 6: blend w1 * full + w3 * normal stretch (w1 = per-pair rest weight, w3 = 1 - w1 when a wall normal exists)
    double w1 = 0.0, w3 = 0.0;
    if(full >= 2)
    {
        bool hn = cutn && gp_pair_normal(pairs[t], cutn, nv);
        if(hn)
        {
            for(int b = 0; b < 3; ++b) gn[b] = nv[0] * G[0][b] + nv[1] * G[1][b] + nv[2] * G[2][b];
            snn = gn[0] * nv[0] + gn[1] * nv[1] + gn[2] * nv[2];
        }
        if(full == 6 || full == 7)
        {   // 6: blend rest weight x full gradient + (1 - w1) x normal stretch; 7: rest weight x full gradient only (nothing when flowing)
            w1 = gpw ? gpw[t] : 0.0;
            w3 = (full == 6 && hn) ? 1.0 - w1 : 0.0;
            if(w1 <= 0.0 && w3 <= 0.0) mode = -1;
            else mode = 6;
        }
        else if(!hn) mode = -1;      // no normal: no contribution
    }
    for(int i = 0; i < 12; ++i)
        if(v[i] >= 0 && mode >= 0)
        {
            double ww[3] = {w[i].x, w[i].y, w[i].z}, gv[3];
            double gw = gn[0] * ww[0] + gn[1] * ww[1] + gn[2] * ww[2], nw = nv[0] * ww[0] + nv[1] * ww[1] + nv[2] * ww[2];
            for(int a = 0; a < 3; ++a)
                gv[a] = mode == 1 ? K * (G[a][0] * ww[0] + G[a][1] * ww[1] + G[a][2] * ww[2])
                      : mode == 2 ? K * gw * nv[a]
                      : mode == 3 ? K * snn * nw * nv[a]
                      : mode == 6 ? K * (w1 * (G[a][0] * ww[0] + G[a][1] * ww[1] + G[a][2] * ww[2]) + w3 * snn * nw * nv[a])
                                  : K * dtr * ww[a];
            atomicAdd(&fb[v[i]].x, gv[0]);
            atomicAdd(&fb[v[i]].y, gv[1]);
            atomicAdd(&fb[v[i]].z, gv[2]);
        }
    int slot = atomicAdd(cursor, 78), kk = 0, anyv = v[0] >= 0 ? v[0] : v[11];
    for(int i = 0; i < 12; ++i)
        for(int j = i; j < 12; ++j)
        {
            int o = slot + kk++;
            if(v[i] < 0 || v[j] < 0 || mode < 0) { rows[o] = anyv; cols[o] = anyv; vals[o].setZero(); continue; }
            double wi[3] = {w[i].x, w[i].y, w[i].z}, wj[3] = {w[j].x, w[j].y, w[j].z};
            int    r = v[i], cc = v[j];
            Eigen::Matrix3d H;
            if(mode == 6)
            {   // w1 K (w_i.w_j) I + w3 K (n.w_i)(n.w_j) n n^T
                double dd = wi[0] * wj[0] + wi[1] * wj[1] + wi[2] * wj[2];
                double sc = (nv[0] * wi[0] + nv[1] * wi[1] + nv[2] * wi[2]) * (nv[0] * wj[0] + nv[1] * wj[1] + nv[2] * wj[2]);
                for(int a = 0; a < 3; ++a)
                    for(int b = 0; b < 3; ++b) H(a, b) = K * (w3 * sc * nv[a] * nv[b] + (a == b ? w1 * dd : 0.0));
            }
            else if(mode >= 2)
            {   // K (w_i.w_j) n n^T  (mode 2)  or  K (n.w_i)(n.w_j) n n^T  (mode 3): symmetric in i <-> j
                double sc = mode == 2 ? (wi[0] * wj[0] + wi[1] * wj[1] + wi[2] * wj[2])
                                      : (nv[0] * wi[0] + nv[1] * wi[1] + nv[2] * wi[2]) * (nv[0] * wj[0] + nv[1] * wj[1] + nv[2] * wj[2]);
                for(int a = 0; a < 3; ++a)
                    for(int b = 0; b < 3; ++b) H(a, b) = K * sc * nv[a] * nv[b];
            }
            else if(full)
            {
                double dd = wi[0] * wj[0] + wi[1] * wj[1] + wi[2] * wj[2];
                H.setZero();
                for(int a = 0; a < 3; ++a) H(a, a) = K * dd;
            }
            else
            {
                for(int a = 0; a < 3; ++a)
                    for(int b = 0; b < 3; ++b) H(a, b) = K * wi[a] * wj[b];
                if(r > cc) H.transposeInPlace();
            }
            if(r > cc) { int tmp = r; r = cc; cc = tmp; }
            rows[o] = r; cols[o] = cc; vals[o] = H;
        }
}
__global__ void k_gp_energy(int np, const int2* pairs, int3 d, const int* node_slot, int base, double dx, double3 org, const double3* x, double Kp,
                            int full, const double3* cutn, const double* gpw, double* part, int stride = 0, const int* plabel = nullptr, const int* nlab = nullptr)
{
    extern __shared__ double sh[];
    int    t = blockIdx.x * blockDim.x + threadIdx.x;
    double e = 0.0;
    if(t < np)
    {
        int     v[12];
        double3 w[12], X[12];
        gp_stencil(pairs[t], d, node_slot, base, dx, org, v, w, X, stride);
        double G[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        for(int i = 0; i < 12; ++i)
            if(v[i] >= 0)
            {
                double3 u = d3sub(x[v[i]], X[i]);
                double  uu[3] = {u.x, u.y, u.z}, ww[3] = {w[i].x, w[i].y, w[i].z};
                for(int a = 0; a < 3; ++a)
                    for(int b = 0; b < 3; ++b) G[a][b] += uu[a] * ww[b];
            }
        double nv[3];
        if(full == 6)
        {
            double w1 = gpw ? gpw[t] : 0.0, f2 = 0.0, snn = 0.0;
            for(int a = 0; a < 3; ++a)
                for(int b = 0; b < 3; ++b) f2 += G[a][b] * G[a][b];
            bool hn = cutn && gp_pair_normal(pairs[t], cutn, nv);
            if(hn)
                for(int a = 0; a < 3; ++a)
                    for(int b = 0; b < 3; ++b) snn += nv[a] * G[a][b] * nv[b];
            e = 0.5 * Kp * (w1 * f2 + (hn ? 1.0 - w1 : 0.0) * snn * snn);
        }
        else if(full == 7)
        {   // rest-only: the full-gradient penalty weighted by the rest weight, nothing when flowing
            const double w1 = gpw ? gpw[t] : 1.0;
            double f2 = 0.0;
            for(int a = 0; a < 3; ++a)
                for(int b = 0; b < 3; ++b) f2 += G[a][b] * G[a][b];
            e = 0.5 * Kp * w1 * f2;
        }
        else if(full >= 2)
        {
            if(cutn && gp_pair_normal(pairs[t], cutn, nv))
            {
                double gn[3];
                for(int b = 0; b < 3; ++b) gn[b] = nv[0] * G[0][b] + nv[1] * G[1][b] + nv[2] * G[2][b];
                if(full == 2) e = 0.5 * Kp * (gn[0] * gn[0] + gn[1] * gn[1] + gn[2] * gn[2]);
                else { double snn = gn[0] * nv[0] + gn[1] * nv[1] + gn[2] * nv[2]; e = 0.5 * Kp * snn * snn; }
            }
        }
        else if(full)
        {
            double f2 = 0.0;
            for(int a = 0; a < 3; ++a)
                for(int b = 0; b < 3; ++b) f2 += G[a][b] * G[a][b];
            e = 0.5 * Kp * f2;
        }
        else
        {
            double dtr = G[0][0] + G[1][1] + G[2][2];
            e          = 0.5 * Kp * dtr * dtr;
        }
    }
    sh[threadIdx.x] = e;
    __syncthreads();
    for(int k = blockDim.x / 2; k > 0; k >>= 1)
    {
        if(threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k];
        __syncthreads();
    }
    if(threadIdx.x == 0) part[blockIdx.x] = sh[0];
}

void LiquidSystem::wall_geometry(device_TetraData& m)
{
    const int n = n_particles;
    if(!d_wp_valid)
    {
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_tri_near, n_cells * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_tri_cnt, n_cells * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_tri_start, (n_cells + 1) * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_wp_valid, n * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_wp_near, n * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_wp_X, n * sizeof(double3)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_wp_tri, n * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_wp_b, n * sizeof(double3)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_wp_n, n * sizeof(double3)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_wp_cell, n * sizeof(int3)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_wp_xi, n * sizeof(double3)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_wp_vid, 8 * (size_t)n * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_wp_act, n * sizeof(int)));
        CUDA_SAFE_CALL(cudaMemset(d_wp_act, 0, n * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_wp_lam, n * sizeof(double)));
        CUDA_SAFE_CALL(cudaMemset(d_wp_lam, 0, n * sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_wp_red, (blocks_for(n) + 1) * sizeof(double)));
        CUDA_SAFE_CALL(cudaMemset(d_wp_valid, 0, n * sizeof(int)));
        CUDA_SAFE_CALL(cudaMemset(d_wp_near, 0, n * sizeof(int)));
        CUDA_SAFE_CALL(cudaMemset(d_wp_vid, 0xff, 8 * (size_t)n * sizeof(int)));
        if(P.wfc) printf("[liquid] wall field constraint ON: K %.0f rho dx^3 per wall point, search radius %.2f h, %d surface triangles%s\n", P.wfc_k, P.wfc_r, n_faces,
                         !P.wfc_ground ? "" : P.clamp_lo_y > -1e299 ? " + the ground plane" : " (LIQ_WFC_GROUND ignored: no +y ground plane)");
        if(P.wfc && P.wfc_pre) printf("[liquid] wall field constraint: boundary activity mode %d (1 = previous step or predictor, 2 = predictor, 3 = previous step)\n", P.wfc_pre);
        if(P.wfc && P.wfc_al) printf("[liquid] wall field constraint: augmented Lagrangian (per wall point multiplier, one update per step)\n");
        if(P.cpic) printf("[liquid] CPIC compatibility ON: particles only couple to grid nodes on their side of the nearest wall (mask radius %.2f dx, %d surface triangles)\n", P.cpic_r, n_faces);
    }
    d_overt = m.o_vertexes;
    // bin the solid surface triangles into the cells near the fluid (cells of THIS step's particles: the active flags are
    // rewritten by k_cell_finalize after P2G)
    CUDA_SAFE_CALL(cudaMemsetAsync(d_cell_flag, 0, n_cells * sizeof(int)));
    k_mark_particle_cells<<<blocks_for(n), 256>>>(n, d_xp, P.origin, P.dx, P.dims, d_cell_flag);
    k_near_flag<<<blocks_for(n_cells), 256>>>(n_cells, d_cell_flag, P.dims, d_tri_near);
    CUDA_SAFE_CALL(cudaMemsetAsync(d_tri_cnt, 0, n_cells * sizeof(int)));
    k_wfc_tri_bin<<<blocks_for(n_faces), 256>>>(n_faces, d_faces, m.vertexes, P.origin, P.dx, P.dims, d_tri_near, d_tri_cnt, d_tri_start, d_tri_list, 0,
                                                m.point_id_to_body_id, porous_body);
    thrust::exclusive_scan(thrust::device, thrust::device_pointer_cast(d_tri_cnt), thrust::device_pointer_cast(d_tri_cnt) + n_cells,
                           thrust::device_pointer_cast(d_tri_start));
    int last[2] = {0, 0};
    CUDA_SAFE_CALL(cudaMemcpy(&last[0], d_tri_start + n_cells - 1, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(&last[1], d_tri_cnt + n_cells - 1, sizeof(int), cudaMemcpyDeviceToHost));
    int total = last[0] + last[1];
    CUDA_SAFE_CALL(cudaMemcpy(d_tri_start + n_cells, &total, sizeof(int), cudaMemcpyHostToDevice));
    if((size_t)total + 1 > tri_list_cap)
    {
        if(d_tri_list) cudaFree(d_tri_list);
        tri_list_cap = (size_t)(1.5 * total) + 1024;
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_tri_list, tri_list_cap * sizeof(int)));
    }
    CUDA_SAFE_CALL(cudaMemsetAsync(d_tri_cnt, 0, n_cells * sizeof(int)));
    if(total > 0)
        k_wfc_tri_bin<<<blocks_for(n_faces), 256>>>(n_faces, d_faces, m.vertexes, P.origin, P.dx, P.dims, d_tri_near, d_tri_cnt, d_tri_start, d_tri_list, 1,
                                                    m.point_id_to_body_id, porous_body);
    const double rq_wfc  = P.wfc_r * h_spacing;
    const double rq_mask = P.cpic ? fmax(rq_wfc, P.cpic_r * P.dx) : rq_wfc;
    const double ground_y = (P.wfc && P.wfc_ground && P.clamp_lo_y > -1e299) ? P.clamp_lo_y - 1e-4 : -1e300;
    k_wfc_points<<<blocks_for(n), 256>>>(n, d_xp, d_faces, m.vertexes, P.origin, P.dx, P.dims, d_tri_start, d_tri_list, rq_mask, rq_wfc,
                                         d_wp_valid, d_wp_near, d_wp_X, d_wp_tri, d_wp_b, d_wp_n, d_wp_cell, d_wp_xi, ground_y);
    h_masked = 0;
    if(P.cpic)
    {
        CUDA_SAFE_CALL(cudaMemsetAsync(d_counters, 0, sizeof(int)));
        k_build_mask<<<blocks_for(n), 256>>>(n, d_xp, d_wp_near, d_wp_X, d_wp_n, d_pbase, d_pfrac, P.origin, P.dx, d_pmask, d_pwinv, d_counters);
        CUDA_SAFE_CALL(cudaMemcpy(&h_masked, d_counters, sizeof(int), cudaMemcpyDeviceToHost));
        if(P.cpic_fields == 2)
        {   // per-side node slots: reference plane per node from the lowest-index near particle referencing it
            CUDA_SAFE_CALL(cudaMemsetAsync(d_nref_key, 0xff, n_nodes * sizeof(unsigned long long)));
            k_nref_min<<<blocks_for(n), 256>>>(n, d_wp_near, d_pbase, d_wp_X, P.dims, P.origin, P.dx, d_nref_key);
            k_nref_fill<<<blocks_for(n_nodes), 256>>>(n_nodes, d_nref_key, d_wp_X, d_wp_n, d_nref_has, d_nref_X, d_nref_n);
            CUDA_SAFE_CALL(cudaMemsetAsync(d_nlab, 0, n_nodes * sizeof(int)));
            k_plabel<<<blocks_for(n), 256>>>(n, d_wp_near, d_wp_n, d_plabel);
            k_nlab<<<blocks_for(n), 256>>>(n, d_wp_near, d_plabel, d_pbase, P.dims, d_nlab);
        }
        if(P.cpic_split)
        {   // 2b: cut axis and per-side wall data of the cells the near-wall particles deposit into
            CUDA_SAFE_CALL(cudaMemsetAsync(d_cell_nabs, 0, n_cells * sizeof(double3)));
            k_side_nabs<<<blocks_for(n), 256>>>(n, d_xp, d_wp_near, d_wp_n, P.origin, P.dx, P.dims, d_cell_nabs);
            k_cell_axis<<<blocks_for(n_cells), 256>>>(n_cells, d_cell_nabs, d_cell_axis);
        }
    }
    h_tri_total = total;
    CUDA_SAFE_CALL(cudaGetLastError());
}

void LiquidSystem::wfc_points(device_TetraData& m)
{
    const int n = n_particles;
    if(!d_wp_valid) return;
    const int total = h_tri_total;
    k_wfc_bind<<<blocks_for(n), 256>>>(n, d_wp_near, d_wp_cell, d_node_slot, P.dims, node_slot_base, d_wp_vid, d_xp, d_nref_has, d_nref_X, d_nref_n,
                                       (P.cpic && P.cpic_fields == 2) ? n_nodes : 0, d_plabel, d_nlab, d_wp_valid);
    h_wp = thrust::reduce(thrust::device, thrust::device_pointer_cast(d_wp_valid), thrust::device_pointer_cast(d_wp_valid) + n, 0);
    h_gp = 0;
    h_gm = 0;
    if(P.gm && h_wp > 0 && h_active_nodes > 0)
    {
        if(!d_gm_key)
        {
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_gm_key, (size_t)n_nodes * sizeof(unsigned long long)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_gm_front, (size_t)n_nodes * sizeof(int)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_gm_cnt, sizeof(int)));
            printf("[liquid] ghost mirror ON: grid nodes on / behind a wall follow the mirror image of the fluid field (free slip, one-sided normal), "
                   "K %.0f rho dx^3\n", P.gm_k);
        }
        if(gm_cap < h_active_nodes)
        {
            if(d_gm_list) cudaFree(d_gm_list);
            if(d_gm_red) cudaFree(d_gm_red);
            gm_cap = std::max(1024, (int)(1.25 * P.node_pool));
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_gm_list, (size_t)gm_cap * sizeof(int2)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_gm_red, (size_t)(blocks_for(gm_cap) + 1) * sizeof(double)));
        }
        CUDA_SAFE_CALL(cudaMemsetAsync(d_gm_key, 0xff, (size_t)n_nodes * sizeof(unsigned long long)));
        CUDA_SAFE_CALL(cudaMemsetAsync(d_gm_front, 0, (size_t)n_nodes * sizeof(int)));
        CUDA_SAFE_CALL(cudaMemsetAsync(d_gm_cnt, 0, sizeof(int)));
        k_gm_key<<<blocks_for(n), 256>>>(n, d_wp_valid, d_pbase, d_wp_X, P.dims, P.origin, P.dx, d_gm_key);
        k_gm_front<<<blocks_for(n), 256>>>(n, d_wp_valid, d_pbase, d_wp_X, d_wp_n, d_gm_key, P.dims, P.origin, P.dx, d_gm_front);
        k_gm_list<<<blocks_for(h_active_nodes), 256>>>(h_active_nodes, d_active_nodes, d_gm_key, d_gm_front, d_wp_X, d_wp_n, P.dims, P.origin, P.dx, d_gm_list,
                                                       gm_cap, d_gm_cnt);
        CUDA_SAFE_CALL(cudaMemcpy(&h_gm, d_gm_cnt, sizeof(int), cudaMemcpyDeviceToHost));
        if(h_gm > gm_cap) throw std::runtime_error("[liquid] ghost-mirror terms exceed their buffer");
        if(steps_done % 200 == 0) printf("[liquid] ghost mirror: %d solid-side grid nodes slaved (step %d)\n", h_gm, steps_done);
    }
    if(P.gp > 0.0)
    {
        if(!d_gp_cut)
        {
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_gp_cut, n_cells * sizeof(int)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_gp_flag, 3 * (size_t)n_cells * sizeof(int)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_gp_scan, 3 * (size_t)n_cells * sizeof(int)));
            if(P.gp_full >= 2) CUDA_SAFE_CALL(cudaMalloc((void**)&d_gp_cutn, n_cells * sizeof(double3)));
            printf("[liquid] ghost penalty ON: %.3g kappa on wall-cut cell / face-neighbour jumps, mode %d (%s)\n", P.gp, P.gp_full,
                   P.gp_full == 0 ? "trace" : P.gp_full == 1 ? "full gradient" : P.gp_full == 2 ? "normal-displacement gradient"
                   : P.gp_full == 3 ? "normal stretch" : "rest-adaptive: full gradient when slow, normal stretch when flowing");
            if(P.gp_full == 6 || P.gp_full == 7) printf("[liquid]   full gradient below %.1f mm/s, normal stretch above %.1f mm/s\n", P.gp_vlo * 1000.0, P.gp_vhi * 1000.0);
        }
        CUDA_SAFE_CALL(cudaMemsetAsync(d_gp_cut, 0, n_cells * sizeof(int)));
        if(d_gp_cutn) CUDA_SAFE_CALL(cudaMemsetAsync(d_gp_cutn, 0, n_cells * sizeof(double3)));
        k_gp_mark_cut<<<blocks_for(n), 256>>>(n, d_wp_valid, d_wp_cell, P.dims, d_gp_cut, d_wp_n, d_gp_cutn);
        k_gp_flag<<<blocks_for(n_cells), 256>>>(n_cells, P.dims, d_cell_flag, d_gp_cut, (P.cpic && d_gp_cutn) ? d_gp_cutn : nullptr, d_gp_flag, (P.wb && d_wb_dead) ? d_wb_dead : nullptr);
        const int n3 = 3 * n_cells;
        thrust::exclusive_scan(thrust::device, thrust::device_pointer_cast(d_gp_flag), thrust::device_pointer_cast(d_gp_flag) + n3,
                               thrust::device_pointer_cast(d_gp_scan));
        int lastf = 0, lasts = 0;
        CUDA_SAFE_CALL(cudaMemcpy(&lastf, d_gp_flag + n3 - 1, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(&lasts, d_gp_scan + n3 - 1, sizeof(int), cudaMemcpyDeviceToHost));
        h_gp = lastf + lasts;
        if(h_gp > gp_cap)
        {
            if(d_gp_pairs) cudaFree(d_gp_pairs);
            if(d_gp_red) cudaFree(d_gp_red);
            if(d_gp_w) cudaFree(d_gp_w);
            gp_cap = (int)(1.5 * h_gp) + 1024;
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_gp_pairs, gp_cap * sizeof(int2)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_gp_w, gp_cap * sizeof(double)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_gp_red, (blocks_for(gp_cap) + 1) * sizeof(double)));
        }
        if(h_gp > 0) k_gp_compact<<<blocks_for(n3), 256>>>(n3, d_gp_flag, d_gp_scan, P.dims, d_gp_pairs);
        if(h_gp > 0 && (P.gp_full == 6 || P.gp_full == 7))
        {
            const bool rel = P.gp_rel && d_wp_tri && d_wp_b && d_faces && m.velocities;
            if(rel)
            {
                if(!d_gp_cutv)
                {
                    CUDA_SAFE_CALL(cudaMalloc((void**)&d_gp_cutv, n_cells * sizeof(double3)));
                    CUDA_SAFE_CALL(cudaMalloc((void**)&d_gp_cutc, n_cells * sizeof(int)));
                }
                CUDA_SAFE_CALL(cudaMemsetAsync(d_gp_cutv, 0, n_cells * sizeof(double3)));
                CUDA_SAFE_CALL(cudaMemsetAsync(d_gp_cutc, 0, n_cells * sizeof(int)));
                k_gp_cut_vel<<<blocks_for(n), 256>>>(n, d_wp_valid, d_wp_cell, P.dims, d_wp_tri, d_wp_b, d_faces, m.velocities, d_gp_cutv, d_gp_cutc);
            }
            k_gp_weight<<<blocks_for(h_gp), 256>>>(h_gp, d_gp_pairs, P.dims, d_mi, d_mvi, P.gp_vlo, P.gp_vhi, d_gp_w, (P.cpic && P.cpic_fields == 2) ? n_nodes : 0, d_plabel, d_nlab,
                                                   rel ? d_gp_cutv : nullptr, rel ? d_gp_cutc : nullptr);
            if(steps_done % 200 == 0)
            {
                double ws = thrust::reduce(thrust::device, thrust::device_pointer_cast(d_gp_w), thrust::device_pointer_cast(d_gp_w) + h_gp, 0.0);
                printf("[liquid] rest-adaptive ghost penalty: mean full-gradient weight %.3f over %d pairs (step %d)\n", ws / h_gp, h_gp, steps_done);
            }
        }
    }
    CUDA_SAFE_CALL(cudaGetLastError());
    if(steps_done % 200 == 0) printf("[liquid] wall field constraint: %d wall points, %d ghost-penalty pairs, %d binned triangle-cell entries, %d masked particles, %d cells -> %d elements, %d private ghost corners, %d orphan nodes fixed, %d two-sided nodes (step %d)\n", h_wp, h_gp, total, h_masked, h_active_cells, h_elements, h_ghost_slots, h_orphans, h_twosided, steps_done);
}

void LiquidSystem::chain_rule_contact(double3* fb, GIPCTripletMatrix& T)
{
    const bool   coh  = coh_active();
    const double kdt2 = pair_k() * last_dt * last_dt;   // bond (LIQ_STI) / repulsion (LIQ_UG_REP) energy scale in Newton units
    if(coh)   // on the particle rows: the particle chain rule below maps it with the contact gradient
        k_coh_grad<<<blocks_for(h_coh), 256>>>(h_coh, d_coh_pairs, d_coh_w, wfc_x, particle_slot_base, kdt2, pair_r0(), pair_rc(), fb);
    chain_rule_contact_particles(fb, T);
    if(coh)
    {   // exact block counts: only the pairs inside the cutoff, each with its union's upper triangle (36..120 blocks)
        const double rc = pair_rc();
        k_coh_hess_count<<<blocks_for(h_coh), 256>>>(h_coh, d_coh_pairs, wfc_x, particle_slot_base, d_xp, P.origin, P.dx, P.dims, d_node_slot, node_slot_base,
                                                     rc, d_coh_nblk);
        thrust::exclusive_scan(thrust::device, thrust::device_pointer_cast(d_coh_nblk), thrust::device_pointer_cast(d_coh_nblk) + h_coh,
                               thrust::device_pointer_cast(d_coh_off));
        int lastc = 0, lasto = 0;
        CUDA_SAFE_CALL(cudaMemcpy(&lastc, d_coh_nblk + h_coh - 1, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(&lasto, d_coh_off + h_coh - 1, sizeof(int), cudaMemcpyDeviceToHost));
        const long long need = (long long)lasto + lastc;
        const int       live = T.global_triplet_offset;
        if(need > 0)
        {
            T.ensure_capacity_preserve((size_t)live, (size_t)(live + need));
            k_coh_hess<<<blocks_for(h_coh), 256>>>(h_coh, d_coh_pairs, d_coh_w, wfc_x, particle_slot_base, d_xp, P.origin, P.dx, P.dims, d_node_slot, node_slot_base,
                                                   kdt2, pair_r0(), rc, d_coh_nblk, d_coh_off, live, T.block_row_indices(), T.block_col_indices(),
                                                   T.block_values());
            T.global_triplet_offset = live + (int)need;
        }
        CUDA_SAFE_CALL(cudaGetLastError());
    }
    if(stm_active())
    {   // membrane surface tension: gradient straight onto the node rows, 36 upper node blocks per surface element
        const double sdt2 = P.stm_sigma * last_dt * last_dt;
        k_stm_grad<<<blocks_for(h_stm), 256>>>(h_stm, d_stm_p, d_stm_a, d_stm_b, d_xp, P.origin, P.dx, P.dims, d_node_slot, node_slot_base, wfc_x, sdt2, fb);
        const int       live = T.global_triplet_offset;
        const long long need = 36LL * h_stm;
        T.ensure_capacity_preserve((size_t)live, (size_t)(live + need));
        k_stm_hess<<<blocks_for(h_stm), 256>>>(h_stm, d_stm_p, d_stm_a, d_stm_b, d_xp, P.origin, P.dx, P.dims, d_node_slot, node_slot_base, wfc_x, sdt2, live,
                                               T.block_row_indices(), T.block_col_indices(), T.block_values());
        T.global_triplet_offset = live + (int)need;
        CUDA_SAFE_CALL(cudaGetLastError());
    }
    if(!wfc_active() || h_wp <= 0 || !d_overt || !wfc_x) return;
    const int  n    = n_particles;
    const int  live = T.global_triplet_offset;
    const long long need = 66LL * h_wp;
    T.ensure_capacity_preserve((size_t)live, (size_t)(live + need));
    int cursor = live;
    CUDA_SAFE_CALL(cudaMemcpy(d_counters + 1, &cursor, sizeof(int), cudaMemcpyHostToDevice));
    const double K = P.wfc_k * P.rho * P.dx * P.dx * P.dx;
    if(!d_wfc_fsolid || wfc_fsolid_n != node_slot_base)
    {
        if(d_wfc_fsolid) cudaFree(d_wfc_fsolid);
        wfc_fsolid_n = node_slot_base;
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_wfc_fsolid, (size_t)std::max(1, wfc_fsolid_n) * sizeof(double3)));
    }
    CUDA_SAFE_CALL(cudaMemsetAsync(d_wfc_fsolid, 0, (size_t)std::max(1, wfc_fsolid_n) * sizeof(double3)));
    const bool pre_log = P.wfc_pre && (steps_done % 200 == 0);
    if(pre_log) CUDA_SAFE_CALL(cudaMemsetAsync(d_counters + 2, 0, sizeof(int)));
    k_wfc_assemble<<<blocks_for(n), 256>>>(n, d_wp_valid, d_wp_tri, d_wp_b, d_wp_n, d_wp_cell, d_wp_xi, d_wp_vid, d_faces, wfc_x, d_overt, P.origin,
                                           P.dx, K, fb, d_counters + 1, T.block_row_indices(), T.block_col_indices(), T.block_values(), d_wfc_fsolid,
                                           particle_slot_base, P.wfc_pre, d_wp_act, wfc_xt, wfc_vel, last_dt, pre_log ? d_counters + 2 : nullptr,
                                           P.wfc_al ? d_wp_lam : nullptr);
    if(pre_log)
    {
        int hpre = 0;
        CUDA_SAFE_CALL(cudaMemcpy(&hpre, d_counters + 2, sizeof(int), cudaMemcpyDeviceToHost));
        if(hpre > h_wp_pre) h_wp_pre = hpre;   // the first assembly of the step is the one that pre-activates
    }
    T.global_triplet_offset = live + (int)need;
    CUDA_SAFE_CALL(cudaGetLastError());
    if(P.gp > 0.0 && h_gp > 0)
    {
        const int       live2 = T.global_triplet_offset;
        const long long need2 = 78LL * h_gp;
        T.ensure_capacity_preserve((size_t)live2, (size_t)(live2 + need2));
        int cursor2 = live2;
        CUDA_SAFE_CALL(cudaMemcpy(d_counters + 1, &cursor2, sizeof(int), cudaMemcpyHostToDevice));
        const double Kgp = P.gp * P.kappa * P.dx * P.dx * P.dx * last_dt * last_dt;
        k_gp_assemble<<<blocks_for(h_gp), 256>>>(h_gp, d_gp_pairs, P.dims, d_node_slot, node_slot_base, P.dx, P.origin, wfc_x, Kgp, P.gp_full, d_gp_cutn, (P.gp_full == 6 || P.gp_full == 7) ? d_gp_w : nullptr, fb, d_counters + 1,
                                                 T.block_row_indices(), T.block_col_indices(), T.block_values(), (P.cpic && P.cpic_fields == 2) ? n_nodes : 0, d_plabel, d_nlab);
        T.global_triplet_offset = live2 + (int)need2;
        CUDA_SAFE_CALL(cudaGetLastError());
    }
    if(P.gm && h_gm > 0)
    {   // ghost mirror: 78 upper blocks per solid-side node (12 vertices)
        const int       live3 = T.global_triplet_offset;
        const long long need3 = 78LL * h_gm;
        T.ensure_capacity_preserve((size_t)live3, (size_t)(live3 + need3));
        k_gm_assemble<<<blocks_for(h_gm), 256>>>(h_gm, d_gm_list, d_active_nodes, d_node_slot, node_slot_base, P.dims, P.origin, P.dx, wfc_x, d_overt,
                                                 d_wp_X, d_wp_n, d_wp_tri, d_wp_b, d_faces, P.gm_k * P.rho * P.dx * P.dx * P.dx, fb, live3,
                                                 T.block_row_indices(), T.block_col_indices(), T.block_values());
        T.global_triplet_offset = live3 + (int)need3;
        CUDA_SAFE_CALL(cudaGetLastError());
    }
}

void LiquidSystem::wfc_commit_activity(const double3* vertexes)
{   // called by the engine after the Newton solve of a step, BEFORE it commits o_vertexes = x (the solid displacement of the step is still x - o_vertexes)
    if((!P.wfc_pre && !P.wfc_al) || P.ug || !wfc_active() || !d_wp_act || !d_overt || n_particles <= 0) return;
    if(h_wp <= 0)
    {
        CUDA_SAFE_CALL(cudaMemsetAsync(d_wp_act, 0, n_particles * sizeof(int)));
        if(d_wp_lam) CUDA_SAFE_CALL(cudaMemsetAsync(d_wp_lam, 0, n_particles * sizeof(double)));
        return;
    }
    const double K = P.wfc_k * P.rho * P.dx * P.dx * P.dx;
    k_wfc_final_act<<<blocks_for(n_particles), 256>>>(n_particles, d_wp_valid, d_wp_tri, d_wp_b, d_wp_n, d_wp_cell, d_wp_xi, d_wp_vid, d_faces, vertexes, d_overt,
                                                     P.origin, P.dx, d_wp_act, P.wfc_al ? d_wp_lam : nullptr, K);
    if(steps_done % 200 == 0)
    {
        const int loaded = thrust::reduce(thrust::device, thrust::device_pointer_cast(d_wp_act), thrust::device_pointer_cast(d_wp_act) + n_particles, 0);
        if(P.wfc_al)
        {
            const double lsum = thrust::reduce(thrust::device, thrust::device_pointer_cast(d_wp_lam), thrust::device_pointer_cast(d_wp_lam) + n_particles, 0.0);
            const double lmax = thrust::reduce(thrust::device, thrust::device_pointer_cast(d_wp_lam), thrust::device_pointer_cast(d_wp_lam) + n_particles, 0.0, thrust::maximum<double>());
            printf("[liquid] wall constraint multipliers: %d of %d wall points loaded, total reaction %.4g N, largest %.4g N = %.3g um of penalty displacement (step %d)\n",
                   loaded, h_wp, last_dt > 0.0 ? lsum / (last_dt * last_dt) : 0.0, last_dt > 0.0 ? lmax / (last_dt * last_dt) : 0.0, lmax / K * 1e6, steps_done);
        }
        else
            printf("[liquid] wall constraint activity: %d of %d wall points loaded at the end of the step, %d counted active at the boundary in the first assembly (step %d)\n",
                   loaded, h_wp, h_wp_pre, steps_done);
        h_wp_pre = 0;
    }
}

void LiquidSystem::cohesion_pairs()
{
    const int n = n_particles;
    h_coh = 0;
    if(!d_coh_s)
    {
        coh_cap = 24 * n;   // a gated particle has at most ~12 partners within 1.75 h; each pair is stored once
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_coh_s, (size_t)n * sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_coh_pairs, (size_t)coh_cap * sizeof(int2)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_coh_w, (size_t)coh_cap * sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_coh_red, (size_t)blocks_for(coh_cap) * sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_coh_count, sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_coh_nblk, (size_t)coh_cap * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_coh_off, (size_t)coh_cap * sizeof(int)));
    }
    if(!d_coh_cwall) CUDA_SAFE_CALL(cudaMalloc((void**)&d_coh_cwall, (size_t)n_cells * sizeof(int)));
    build_neighbor_grid();   // final step-start positions (push-apart, pump, porous moved them)
    if(rep_on())
    {   // LIQ_UG_REP: every particle takes part (no surface gate); the potential switches to repulsion in coh_psi
        if(!coh_gate_done)
        {
            const int mode = 1;
            CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_coh_mode, &mode, sizeof(int)));
            thrust::fill(thrust::device, thrust::device_pointer_cast(d_coh_s), thrust::device_pointer_cast(d_coh_s) + n, 1.0);
            coh_gate_done = true;
        }
    }
    else if(!(P.sti_freeze && coh_gate_done))
    {   // LIQ_STI_FREEZE keeps the first step's gate (diagnostic: no pair switches on or off between steps)
        CUDA_SAFE_CALL(cudaMemsetAsync(d_coh_cwall, 0, (size_t)n_cells * sizeof(int)));
        k_coh_wallcells<<<blocks_for(n), 256>>>(n, d_xp, (P.wfc || P.cpic) ? d_wp_near : nullptr, P.origin, P.dx, P.dims, d_coh_cwall);
        k_coh_gate<<<blocks_for(n), 256>>>(n, d_xp, d_cstart, d_coh_cwall, P.origin, P.dx, P.dims, d_coh_s);
        coh_gate_done = true;
    }
    CUDA_SAFE_CALL(cudaMemsetAsync(d_coh_count, 0, sizeof(int)));
    k_coh_pairs<<<blocks_for(n), 256>>>(n, d_xp, d_coh_s, d_pidx, d_cstart, d_cend, P.origin, P.dx, P.dims, pair_rc() + 0.25 * h_spacing, coh_cap,
                                        d_coh_pairs, d_coh_w, d_coh_count);
    CUDA_SAFE_CALL(cudaGetLastError());
    int cnt = 0;
    CUDA_SAFE_CALL(cudaMemcpy(&cnt, d_coh_count, sizeof(int), cudaMemcpyDeviceToHost));
    if(cnt > coh_cap)
    {
        static bool warned = false;
        if(!warned) printf("[liquid] WARNING implicit surface tension: %d pairs exceed the capacity %d, extra pairs dropped\n", cnt, coh_cap);
        warned = true;
        cnt = coh_cap;
    }
    h_coh = cnt;
    if(rep_on())
    {
        if(steps_done == 0 || steps_done % 200 == 0)
            printf("[liquid] LIQ_UG_REP sub-cell density control: pair repulsion k %.3g N/m below %.2f mm (%.2f seed spacings), %d pairs (step %d)\n",
                   pair_k(), pair_r0() * 1e3, P.ug_rep_r0, h_coh, steps_done);
    }
    else if(steps_done == 0)
        printf("[liquid] implicit surface tension ON: gamma %.4g N/m (bond %.3g J), rest %.2f mm, cutoff %.2f mm, surface particles (air-adjacent cells, wall layer excluded), %d pairs\n",
               P.sti_gamma, 2.0 * P.sti_gamma * h_spacing * h_spacing, P.sti_r0 * h_spacing * 1e3, P.sti_rc * h_spacing * 1e3, h_coh);
    else if(steps_done % 200 == 0)
        printf("[liquid] implicit surface tension: %d pairs (step %d)\n", h_coh, steps_done);
}

void LiquidSystem::stm_setup()
{
    const int n = n_particles;
    h_stm = 0;
    if(!d_stm_flag)
    {
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_stm_flag, (size_t)n * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_stm_scan, (size_t)n * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_stm_nrm, (size_t)n * sizeof(double3)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_stm_p, (size_t)n * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_stm_a, (size_t)n * sizeof(double3)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_stm_b, (size_t)n * sizeof(double3)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_stm_red, (size_t)blocks_for(n) * sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_stm_A, (size_t)n * sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_stm_n2, (size_t)n * sizeof(double3)));
    }
    build_neighbor_grid();   // final step-start positions
    const bool walls = (P.wfc || P.cpic) && d_faces && n_faces > 0 && d_wp_valid && d_wp_X && d_wp_n;   // the wall pass ran just before (same condition as in pre_step)
    k_stm_flag<<<blocks_for(n), 256>>>(n, d_xp, d_pidx, d_cstart, d_cend, walls ? d_wp_valid : nullptr, d_wp_X, d_wp_n, P.origin, P.dx, P.dims,
                                       2.0 * h_spacing, P.stm_thr * h_spacing, d_stm_flag, d_stm_nrm);
    thrust::exclusive_scan(thrust::device, thrust::device_pointer_cast(d_stm_flag), thrust::device_pointer_cast(d_stm_flag) + n,
                           thrust::device_pointer_cast(d_stm_scan));
    int lf = 0, ls = 0;
    CUDA_SAFE_CALL(cudaMemcpy(&lf, d_stm_flag + n - 1, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(&ls, d_stm_scan + n - 1, sizeof(int), cudaMemcpyDeviceToHost));
    h_stm = ls + lf;
    const bool cal = P.stm_amode == 1;
    if(h_stm > 0 && cal)
    {
        CUDA_SAFE_CALL(cudaMemsetAsync(d_stm_A, 0, (size_t)n * sizeof(double)));
        k_stm_area<<<blocks_for(n), 256>>>(n, d_xp, d_stm_flag, d_stm_nrm, d_pidx, d_cstart, d_cend, walls ? d_wp_valid : nullptr, d_wp_X, d_wp_n,
                                           P.origin, P.dx, P.dims, 2.0 * h_spacing, P.stm_area, P.stm_amax * h_spacing * h_spacing, d_stm_A, d_stm_n2);
    }
    if(h_stm > 0)
        k_stm_fill<<<blocks_for(n), 256>>>(n, d_stm_flag, d_stm_scan, cal ? d_stm_n2 : d_stm_nrm, sqrt(P.stm_area) * h_spacing, d_stm_p, d_stm_a, d_stm_b,
                                           cal ? d_stm_A : nullptr);
    CUDA_SAFE_CALL(cudaGetLastError());
    if(steps_done == 0)
    {
        double area = h_stm * P.stm_area * h_spacing * h_spacing;
        if(cal && h_stm > 0) area = thrust::reduce(thrust::device, thrust::device_pointer_cast(d_stm_A), thrust::device_pointer_cast(d_stm_A) + n, 0.0);
        printf("[liquid] membrane surface tension ON: sigma %.4g N/m, %s (h %.2f mm), surface = neighbour-centroid offset > %.2f h "
               "within 2 h (solids count as occupied: neutral contact angle): %d of %d particles, total element area %.4g cm2\n", P.stm_sigma,
               cal ? "element area = 1 / local areal density of the surface elements" : "fixed element area", h_spacing * 1e3, P.stm_thr, h_stm, n,
               area * 1e4);
        if(!cal) printf("[liquid]   fixed element area %.2f h^2\n", P.stm_area);
        else if(P.stm_area != 1.0) printf("[liquid]   calibrated area x %.3f\n", P.stm_area);
    }
}

void LiquidSystem::chain_rule_contact_particles(double3* fb, GIPCTripletMatrix& T)
{
    const int n    = n_particles;
    const int live = T.global_triplet_offset;
    g_prof.start();
    k_contact_grad_chain<<<blocks_for(n), 256>>>(n, d_pbase, d_pfrac, d_pmask, d_pwinv, d_node_slot, P.dims, node_slot_base, particle_slot_base, fb, d_xp, d_nref_has, d_nref_X, d_nref_n, (P.cpic && P.cpic_fields == 2) ? n_nodes : 0, d_plabel, d_nlab);
    g_prof.stop(1);
    g_prof.start();
    CUDA_SAFE_CALL(cudaMemsetAsync(d_Hpp, 0, 9 * (size_t)n * sizeof(double)));
    CUDA_SAFE_CALL(cudaMemsetAsync(d_pflag, 0, n * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemsetAsync(d_counters, 0, 2 * sizeof(int)));
    h_contact_particles = 0;
    h_expanded_blocks   = 0;
    if(live <= 0) return;
    k_contact_scan<<<blocks_for(live), 256>>>(live, T.block_row_indices(), T.block_col_indices(), T.block_values(), particle_slot_base, n, d_Hpp, d_pflag, d_counters);
    int cnt[2] = {0, 0};
    CUDA_SAFE_CALL(cudaMemcpy(cnt, d_counters, 2 * sizeof(int), cudaMemcpyDeviceToHost));
    int nflag = thrust::reduce(thrust::device, thrust::device_pointer_cast(d_pflag), thrust::device_pointer_cast(d_pflag) + n, 0);
    static const bool dbg       = liq_env_flag("LIQ_DEBUG");
    static const bool no_expand = liq_env_flag("LIQ_NO_EXPAND");
    static const bool no_cross  = liq_env_flag("LIQ_NO_CROSS");
    static const bool no_diag   = liq_env_flag("LIQ_NO_DIAG");
    static const bool lump_diag = liq_env_flag("LIQ_LUMP_DIAG");
    static const bool h8        = liq_env_flag("LIQ_H8");
    const long long   per_diag  = h8 ? 36 : (lump_diag ? 27 : 378);
    const long long   per_cross = h8 ? 8 : 27;
    long long need = (no_cross ? 0 : per_cross * cnt[0]) + (no_diag ? 0 : per_diag * nflag);
    h_contact_particles = nflag;
    h_expanded_blocks   = need;
    if(dbg)
    {
        int bad = 0;
        CUDA_SAFE_CALL(cudaMemset(d_counters, 0, sizeof(int)));
        k_count_nonfinite3<<<blocks_for(node_slot_base + P.node_pool + n), 256>>>(node_slot_base + P.node_pool + n, fb, d_counters);
        CUDA_SAFE_CALL(cudaMemcpy(&bad, d_counters, sizeof(int), cudaMemcpyDeviceToHost));
        printf("[liq-dbg] chain: live %d cross %d flagged %d need %lld cap %zu  nonfinite fb %d\n", live, cnt[0], nflag, need, T.triplet_capacity(), bad);
    }
    g_prof.stop(2);
    if(need <= 0 || no_expand) return;
    g_prof.start();
    T.ensure_capacity_preserve((size_t)live, (size_t)(live + need));
    int cursor = live;
    CUDA_SAFE_CALL(cudaMemcpy(d_counters + 1, &cursor, sizeof(int), cudaMemcpyHostToDevice));
    if(!no_cross)
    {
        if(h8)
            k_emit_cross8<<<blocks_for(live), 256>>>(live, T.block_row_indices(), T.block_col_indices(), T.block_values(), particle_slot_base, n, d_xp, P.origin, P.dx,
                                                     P.dims, d_node_slot, node_slot_base, d_counters + 1, T.block_row_indices(), T.block_col_indices(), T.block_values());
        else
            k_emit_cross<<<blocks_for(live), 256>>>(live, T.block_row_indices(), T.block_col_indices(), T.block_values(), particle_slot_base, n, d_pbase, d_pfrac,
                                                    d_pmask, d_pwinv, d_node_slot, P.dims, node_slot_base, d_counters + 1, T.block_row_indices(), T.block_col_indices(), T.block_values(), d_xp, d_nref_has, d_nref_X, d_nref_n, (P.cpic && P.cpic_fields == 2) ? n_nodes : 0, d_plabel, d_nlab);
    }
    if(!no_diag)
    {
        if(h8)
            k_emit_diag8<<<blocks_for(n), 256>>>(n, d_pflag, d_Hpp, d_xp, P.origin, P.dx, P.dims, d_node_slot, node_slot_base, d_counters + 1, T.block_row_indices(),
                                                 T.block_col_indices(), T.block_values());
        else if(lump_diag)
            k_emit_diag_lumped<<<blocks_for(n), 256>>>(n, d_pflag, d_Hpp, d_pbase, d_pfrac, d_pmask, d_pwinv, d_node_slot, P.dims, node_slot_base, d_counters + 1, T.block_row_indices(),
                                                       T.block_col_indices(), T.block_values(), d_xp, d_nref_has, d_nref_X, d_nref_n, (P.cpic && P.cpic_fields == 2) ? n_nodes : 0, d_plabel, d_nlab);
        else
            k_emit_diag<<<blocks_for(n), 256>>>(n, d_pflag, d_Hpp, d_pbase, d_pfrac, d_pmask, d_pwinv, d_node_slot, P.dims, node_slot_base, d_counters + 1, T.block_row_indices(),
                                                T.block_col_indices(), T.block_values(), d_xp, d_nref_has, d_nref_X, d_nref_n, (P.cpic && P.cpic_fields == 2) ? n_nodes : 0, d_plabel, d_nlab);
    }
    T.global_triplet_offset = live + (int)need;
    CUDA_SAFE_CALL(cudaGetLastError());
    g_prof.stop(3);
    if(dbg)
    {
        int bad = 0, cur = 0;
        CUDA_SAFE_CALL(cudaMemcpy(&cur, d_counters + 1, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemset(d_counters, 0, sizeof(int)));
        k_count_nonfinite_blocks<<<blocks_for((int)need), 256>>>((int)need, T.block_values() + live, T.block_row_indices() + live, T.block_col_indices() + live,
                                                                 node_slot_base + P.node_pool + n, d_counters);
        CUDA_SAFE_CALL(cudaMemcpy(&bad, d_counters, sizeof(int), cudaMemcpyDeviceToHost));
        printf("[liq-dbg]   emitted: cursor end %d (expected %lld)  bad blocks %d\n", cur, live + need, bad);
    }
}

// grid wall boundary condition: per active node slot one 3x3 diagonal block (zero for free nodes),
// slip: E = K/2 (n.(x - xt))^2, follow: E = K/2 |x - xt|^2 (inertia units: no dt^2 factor)
// wall_t bits: 1|2 = slip | fixed, 4 = static wall node selected for the stiff penalty (LIQ_WALL_STATIC_MASS), 8 = static wall node,
// 16 = behind the wall (phi < 0). Penalty E = Kt/2 |d|^2 + (Kn - Kt)/2 (n.d)^2: Kn on the wall normal, Kt tangential (0 for slip nodes).
__device__ __forceinline__ void wall_pen_k(int t, double K, double kmul, int knorm, double ktang, double& Kn, double& Kt)
{
    const bool stiff = t & 4;
    Kn = stiff ? K * kmul : K;
    Kt = (stiff && !knorm) ? K * kmul : K;
    if((t & 8) && !(t & 16)) Kt *= ktang;   // LIQ_WALL_TANG: static wall, fluid-side band node
    if((t & 3) == 1) Kt = 0.0;
}
__global__ void k_wall_assemble(int nact, int base, int fem_off, const int* wt, const double3* wn, const double3* wxt, const double3* vert,
                                double K, int off, Eigen::Matrix3d* vals, int* rows, int* cols, double kmul = 1.0,   // kmul: static-wall nodes (bit 4)
                                int knorm = 0, double ktang = 1.0)   // LIQ_WALL_STATIC_NORMAL, LIQ_WALL_TANG
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if(s >= nact) return;
    int vtx = base + s, idx = off + s;
    rows[idx] = vtx + fem_off;
    cols[idx] = vtx + fem_off;
    Eigen::Matrix3d H;
    H.setZero();
    const int t = wt[s];
    if(t & 3)
    {
        double Kn, Kt;
        wall_pen_k(t, K, kmul, knorm, ktang, Kn, Kt);
        double3 x = vert[vtx], xt = wxt[s];
        double  d[3] = {x.x - xt.x, x.y - xt.y, x.z - xt.z};
        double3 n  = wn[s];
        double  nv[3] = {n.x, n.y, n.z};
        double  dn = nv[0] * d[0] + nv[1] * d[1] + nv[2] * d[2];
        double  g[3];
        for(int a = 0; a < 3; ++a)
        {
            g[a] = Kt * d[a] + (Kn - Kt) * dn * nv[a];
            for(int b = 0; b < 3; ++b) H(a, b) = (Kn - Kt) * nv[a] * nv[b] + (a == b ? Kt : 0.0);
        }
        for(int a = 0; a < 3; ++a) binned_deposit(g_gbin + ((size_t)vtx * 3 + a) * BINNED_K, g[a]);
    }
    vals[idx] = H;
}
__global__ void k_wall_energy(int nact, int base, const int* wt, const double3* wn, const double3* wxt, const double3* vert, double Kp, double* part,
                              double kmul = 1.0, int knorm = 0, double ktang = 1.0)
{
    extern __shared__ double sh[];
    int    s = blockIdx.x * blockDim.x + threadIdx.x;
    double e = 0.0;
    if(s < nact && (wt[s] & 3))
    {
        double Kn, Kt;
        wall_pen_k(wt[s], Kp, kmul, knorm, ktang, Kn, Kt);
        double3 x = vert[base + s], xt = wxt[s];
        double  d[3] = {x.x - xt.x, x.y - xt.y, x.z - xt.z};
        double3 n  = wn[s];
        double  dn = n.x * d[0] + n.y * d[1] + n.z * d[2];
        e = 0.5 * Kt * (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) + 0.5 * (Kn - Kt) * dn * dn;
    }
    sh[threadIdx.x] = e;
    __syncthreads();
    for(int k = blockDim.x / 2; k > 0; k >>= 1)
    {
        if(threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k];
        __syncthreads();
    }
    if(threadIdx.x == 0) part[blockIdx.x] = sh[0];
}
__global__ void k_add_partials(int nb, const double* part, double* out)
{
    double s = 0.0;
    for(int i = 0; i < nb; ++i) s += part[i];
    out[0] += s;
}

__global__ void k_fp_combine(int n, int base, const double* bins, double3* out)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if(s >= n) return;
    const size_t v = (size_t)base + s;
    out[s] = make_double3(binned_combine(bins + (v * 3 + 0) * BINNED_K), binned_combine(bins + (v * 3 + 1) * BINNED_K), binned_combine(bins + (v * 3 + 2) * BINNED_K));
}
__global__ void k_fp_nodes(int n, int base, const int* active_nodes, const double3* vert, const double3* xt, const double* mass, const double* node_w, const int* wall_t,
                           double* out)   // 12 doubles per node: x(3) xt(3) mass node_w wall_t lattice_id pad pad
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if(s >= n) return;
    const int nid = active_nodes[s];
    const double3 x = vert[base + s], t = xt[base + s];
    double* o = out + 12 * (size_t)s;
    o[0] = x.x; o[1] = x.y; o[2] = x.z; o[3] = t.x; o[4] = t.y; o[5] = t.z;
    o[6] = mass[base + s]; o[7] = node_w ? node_w[nid] : 0.0; o[8] = wall_t ? (double)wall_t[s] : 0.0; o[9] = (double)nid; o[10] = 0.0; o[11] = 0.0;
}
void LiquidSystem::force_probe(const double3* vertexes, int fem_off, double dt, int global_offset, Eigen::Matrix3d* vals, int* rows, int* cols, const char* prefix)
{
    // own triplet buffers (the Hessian blocks are discarded): callable outside the assembly, e.g. at the converged iterate
    const size_t ntrip = 36 * (size_t)h_elements + (size_t)h_active_nodes + 16;
    Eigen::Matrix3d* tv = nullptr; int *tr = nullptr, *tc = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void**)&tv, ntrip * sizeof(Eigen::Matrix3d)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&tr, ntrip * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&tc, ntrip * sizeof(int)));
    vals = tv; rows = tr; cols = tc; global_offset = 0;
    const size_t nv = (size_t)node_slot_base + h_active_nodes, nb = nv * 3 * BINNED_K;
    double *tmp = nullptr, *orig = nullptr, *nd = nullptr;
    double3* comb = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void**)&tmp, nb * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&comb, (size_t)h_active_nodes * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&nd, 12 * (size_t)h_active_nodes * sizeof(double)));
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&orig, g_gbin, sizeof(double*)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_gbin, &tmp, sizeof(double*)));
    CUDA_SAFE_CALL(cudaMemset(tmp, 0, nb * sizeof(double)));
    const std::string stem = std::string(prefix) + "_" + std::to_string(steps_done);
    std::ofstream data(stem + "_force.bin", std::ios::binary);
    std::vector<double3> h(h_active_nodes);
    auto grab = [&]() {
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        k_fp_combine<<<blocks_for(h_active_nodes), 256>>>(h_active_nodes, node_slot_base, tmp, comb);
        CUDA_SAFE_CALL(cudaMemcpy(h.data(), comb, (size_t)h_active_nodes * sizeof(double3), cudaMemcpyDeviceToHost));
        data.write((const char*)h.data(), (size_t)h_active_nodes * sizeof(double3));
        CUDA_SAFE_CALL(cudaMemset(tmp, 0, nb * sizeof(double)));
    };
    // (1) voxel pressure + hydrostatic term (+ J barrier), (2) viscosity (0 at u = 0, kept for the layout), (3) wall band penalty
    k_vox_assemble<<<blocks_for(h_elements, 128), 128>>>(h_elements, d_vox_nodes, d_vox_V, d_vox_Jb, d_vox_ijk, vertexes, P.dx, P.origin, P.kappa, P.tension_ratio, dt,
                                                             fem_off, global_offset, vals, rows, cols, h_cc > 0 ? d_vox_cc : nullptr, d_vox_a, d_vox_vg,
                                                             P.vbar, P.vbar_k * P.kappa, P.wb ? d_vox_f : nullptr, (P.wb && P.wb_hg > 0.0) ? d_wb_hg : nullptr, P.wb_hg * P.kappa, P.wb_det,
                                                             (P.wb && P.wb_surfk > 0.0) ? d_vox_s : nullptr, P.wb_surfk * P.rho * sqrt(last_gravity.x * last_gravity.x + last_gravity.y * last_gravity.y + last_gravity.z * last_gravity.z), liquid_up(last_gravity),
                                                             (P.wb && P.wb_geov && d_vox_Vr) ? d_vox_Vr : nullptr, P.wb ? P.wb_fullf : 0);
    grab();
    if(P.visc > 0.0)
        k_visc_assemble<<<blocks_for(h_elements, 128), 128>>>(h_elements, d_vox_nodes, d_vox_ijk, h_cc > 0 ? d_vox_cc : nullptr, d_vox_a, d_vox_vg, vertexes, P.dx, P.origin,
                                                              P.visc, dt, fem_off, global_offset, vals);
    grab();
    if(wall_active() && h_active_nodes > 0)
        k_wall_assemble<<<blocks_for(h_active_nodes), 256>>>(h_active_nodes, node_slot_base, fem_off, d_wall_t, d_wall_n, d_wall_xt, vertexes,
                                                             P.wall_mass * P.rho * P.dx * P.dx * P.dx, global_offset + 36 * h_elements, vals, rows, cols,
                                                             P.wall_static_mass > 0.0 ? P.wall_static_mass / P.wall_mass : 1.0, P.wall_static_normal, P.wall_tang);
    grab();
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_gbin, &orig, sizeof(double*)));
    k_fp_nodes<<<blocks_for(h_active_nodes), 256>>>(h_active_nodes, node_slot_base, d_active_nodes, vertexes, fp_xt, fp_mass, P.wb ? d_node_w : nullptr,
                                                    wall_active() ? d_wall_t : nullptr, nd);
    std::vector<double> hn(12 * (size_t)h_active_nodes);
    CUDA_SAFE_CALL(cudaMemcpy(hn.data(), nd, hn.size() * sizeof(double), cudaMemcpyDeviceToHost));
    data.write((const char*)hn.data(), hn.size() * sizeof(double));
    data.flush();
    std::ofstream meta(stem + "_force.json");
    meta.precision(17);
    meta << "{\"step\":" << steps_done << ",\"dt\":" << dt << ",\"dx\":" << P.dx << ",\"origin\":[" << P.origin.x << ',' << P.origin.y << ',' << P.origin.z
         << "],\"dims\":[" << P.dims.x << ',' << P.dims.y << ',' << P.dims.z << "],\"n\":" << h_active_nodes << ",\"gravity\":[" << last_gravity.x << ',' << last_gravity.y << ','
         << last_gravity.z << "],\"rho\":" << P.rho << ",\"kappa\":" << P.kappa << ",\"layout\":\"3 x n double3 gradients (vox, visc, wall; dt^2-scaled), then n x 12 doubles"
         << " (x, xtilde, mass, node_w, wall_t, lattice id, pad, pad)\"}";
    CUDA_SAFE_CALL(cudaFree(tmp)); CUDA_SAFE_CALL(cudaFree(comb)); CUDA_SAFE_CALL(cudaFree(nd));
    CUDA_SAFE_CALL(cudaFree(tv)); CUDA_SAFE_CALL(cudaFree(tr)); CUDA_SAFE_CALL(cudaFree(tc));
    printf("[liquid] force probe: %s_force.bin (%d nodes, step %d)\n", stem.c_str(), h_active_nodes, steps_done);
}

void LiquidSystem::assemble(const double3* vertexes, int fem_off, double dt, int global_offset, Eigen::Matrix3d* vals, int* rows, int* cols)
{
    if(P.ug)
    {
        if(n_particles <= 0 || h_active_nodes <= 0) return;
        g_prof.start();
        if(P.ug_bspline)
            k_ug_assemble<3><<<blocks_for(n_particles, 128), 128>>>(n_particles, d_ug_nodes, d_ug_grad, d_Vp0, d_Jp, vertexes, UgGrid{d_active_nodes, node_slot_base, P.dims, P.origin, P.dx}, P.kappa, P.tension_ratio,
                                                                   P.ug_jact, P.ug_kb * P.kappa, dt, fem_off, global_offset, vals, rows, cols, P.ug_hg);
        else
            k_ug_assemble<2><<<blocks_for(n_particles, 128), 128>>>(n_particles, d_ug_nodes, d_ug_grad, d_Vp0, d_Jp, vertexes, UgGrid{d_active_nodes, node_slot_base, P.dims, P.origin, P.dx}, P.kappa, P.tension_ratio,
                                                                   P.ug_jact, P.ug_kb * P.kappa, dt, fem_off, global_offset, vals, rows, cols, P.ug_hg);
        CUDA_SAFE_CALL(cudaGetLastError());
        g_prof.stop(0);
        g_prof.tick();
        return;
    }
    if(h_elements <= 0) return;
    {   // LIQ_FORCE_PROBE=<prefix>, LIQ_FORCE_PROBE_STEP=<n>: per-term node forces at the first Newton iterate (u = 0) of step n
        const char* fpre = getenv("LIQ_FORCE_PROBE");
        if(fpre && *fpre && fp_done != steps_done && steps_done == (int)liquid_env("LIQ_FORCE_PROBE_STEP", -1) && h_active_nodes > 0)
        {
            fp_done = steps_done;
            force_probe(vertexes, fem_off, dt, global_offset, vals, rows, cols, (std::string(fpre) + "_u0").c_str());
        }
    }
    g_prof.start();
    k_vox_assemble<<<blocks_for(h_elements, 128), 128>>>(h_elements, d_vox_nodes, d_vox_V, d_vox_Jb, d_vox_ijk, vertexes, P.dx, P.origin, P.kappa, P.tension_ratio, dt,
                                                             fem_off, global_offset, vals, rows, cols, h_cc > 0 ? d_vox_cc : nullptr, d_vox_a, d_vox_vg,
                                                             P.vbar, P.vbar_k * P.kappa, P.wb ? d_vox_f : nullptr, (P.wb && P.wb_hg > 0.0) ? d_wb_hg : nullptr, P.wb_hg * P.kappa, P.wb_det,
                                                             (P.wb && P.wb_surfk > 0.0) ? d_vox_s : nullptr, P.wb_surfk * P.rho * sqrt(last_gravity.x * last_gravity.x + last_gravity.y * last_gravity.y + last_gravity.z * last_gravity.z), liquid_up(last_gravity),
                                               (P.wb && P.wb_geov && d_vox_Vr) ? d_vox_Vr : nullptr, P.wb ? P.wb_fullf : 0);
    CUDA_SAFE_CALL(cudaGetLastError());
    if(P.visc > 0.0)
    {
        k_visc_assemble<<<blocks_for(h_elements, 128), 128>>>(h_elements, d_vox_nodes, d_vox_ijk, h_cc > 0 ? d_vox_cc : nullptr, d_vox_a, d_vox_vg, vertexes, P.dx, P.origin,
                                                              P.visc, dt, fem_off, global_offset, vals);
        CUDA_SAFE_CALL(cudaGetLastError());
    }
    if(P.pdev > 0.0)
    {
        k_pdev_assemble<<<blocks_for(h_elements, 128), 128>>>(h_elements, d_vox_nodes, d_vox_ijk, h_cc > 0 ? d_vox_cc : nullptr, d_vox_vg, P.dims, vertexes, P.dx, P.origin,
                                                              d_cstart, d_cend, d_pidx, d_xp, d_Jp, d_Vp0, last_gravity, P.rho, P.kappa, P.pdev * P.kappa, P.pdev_all, dt,
                                                              fem_off, global_offset, vals);
        CUDA_SAFE_CALL(cudaGetLastError());
    }
    if(wall_active() && h_active_nodes > 0)
        k_wall_assemble<<<blocks_for(h_active_nodes), 256>>>(h_active_nodes, node_slot_base, fem_off, d_wall_t, d_wall_n, d_wall_xt, vertexes,
                                                             P.wall_mass * P.rho * P.dx * P.dx * P.dx, global_offset + 36 * h_elements, vals, rows, cols,
                                                             P.wall_static_mass > 0.0 ? P.wall_static_mass / P.wall_mass : 1.0, P.wall_static_normal, P.wall_tang);
    if(h_ag > 0)
        gipc_liquid_wb::k_wb_aggr_assemble<<<blocks_for(h_ag), 256>>>(h_ag, d_wb_agn, d_wb_agr, d_wb_agt, d_wb_agnrm, d_wb_agc, P.dims, P.origin, P.dx, d_node_slot, node_slot_base, fem_off, vertexes,
                                                                     P.wall_mass * P.rho * P.dx * P.dx * P.dx, global_offset + 36 * h_elements + (wall_active() ? h_active_nodes : 0),
                                                                     vals, rows, cols);
    g_prof.stop(0);
    g_prof.tick();
}

void LiquidSystem::energy_device_out(const double3* vertexes, double* out_slot)
{
    if(P.ug)
    {
        if(n_particles <= 0 || h_active_nodes <= 0) { CUDA_SAFE_CALL(cudaMemsetAsync(out_slot, 0, sizeof(double))); return; }
        const int t = 256, b = blocks_for(n_particles, t);
        if(P.ug_bspline)
            k_ug_energy<3><<<b, t, t * sizeof(double)>>>(n_particles, d_ug_nodes, d_ug_grad, d_Vp0, d_Jp, vertexes, UgGrid{d_active_nodes, node_slot_base, P.dims, P.origin, P.dx}, P.kappa, P.tension_ratio,
                                                         P.ug_jact, P.ug_kb * P.kappa, d_ug_red, P.ug_hg, last_dt > 0.0 ? last_dt : 1.0);
        else
            k_ug_energy<2><<<b, t, t * sizeof(double)>>>(n_particles, d_ug_nodes, d_ug_grad, d_Vp0, d_Jp, vertexes, UgGrid{d_active_nodes, node_slot_base, P.dims, P.origin, P.dx}, P.kappa, P.tension_ratio,
                                                         P.ug_jact, P.ug_kb * P.kappa, d_ug_red, P.ug_hg, last_dt > 0.0 ? last_dt : 1.0);
        k_sum_partials<<<1, 1>>>(b, d_ug_red, out_slot);
        CUDA_SAFE_CALL(cudaGetLastError());
        return;
    }
    if(h_elements <= 0)
    {
        CUDA_SAFE_CALL(cudaMemsetAsync(out_slot, 0, sizeof(double)));
        return;
    }
    const int t = 256;
    int       b = blocks_for(h_elements, t);
    g_prof.start();
    k_vox_energy<<<b, t, t * sizeof(double)>>>(h_elements, d_vox_nodes, d_vox_V, d_vox_Jb, d_vox_ijk, vertexes, P.dx, P.origin, P.kappa, P.tension_ratio, d_reduce,
                                               h_cc > 0 ? d_vox_cc : nullptr, d_vox_a, d_vox_vg, P.vbar,
                                               P.vbar_k * P.kappa, P.wb ? d_vox_f : nullptr, (P.wb && P.wb_hg > 0.0) ? d_wb_hg : nullptr, P.wb_hg * P.kappa, P.invert_ok, P.wb_det,
                                               (P.wb && P.wb_surfk > 0.0) ? d_vox_s : nullptr, P.wb_surfk * P.rho * sqrt(last_gravity.x * last_gravity.x + last_gravity.y * last_gravity.y + last_gravity.z * last_gravity.z), liquid_up(last_gravity),
                                               (P.wb && P.wb_geov && d_vox_Vr) ? d_vox_Vr : nullptr, P.wb ? P.wb_fullf : 0);
    k_sum_partials<<<1, 1>>>(b, d_reduce, out_slot);
    if(P.visc > 0.0 && h_elements > 0 && last_dt > 0.0)
    {
        if(!d_visc_red) CUDA_SAFE_CALL(cudaMalloc((void**)&d_visc_red, (blocks_for(max_voxels, 256) + 1) * sizeof(double)));
        const int bv = blocks_for(h_elements, 256);
        k_visc_energy<<<bv, 256, 256 * sizeof(double)>>>(h_elements, d_vox_nodes, d_vox_ijk, h_cc > 0 ? d_vox_cc : nullptr, d_vox_a, d_vox_vg, vertexes, P.dx, P.origin, P.visc, last_dt, d_visc_red);
        k_add_partials<<<1, 1>>>(bv, d_visc_red, out_slot);
    }
    if(P.pdev > 0.0 && h_elements > 0)
    {
        if(!d_pdev_red) CUDA_SAFE_CALL(cudaMalloc((void**)&d_pdev_red, (blocks_for(max_voxels, 256) + 1) * sizeof(double)));
        const int bp = blocks_for(h_elements, 256);
        k_pdev_energy<<<bp, 256, 256 * sizeof(double)>>>(h_elements, d_vox_nodes, d_vox_ijk, h_cc > 0 ? d_vox_cc : nullptr, d_vox_vg, P.dims, vertexes, P.dx, P.origin,
                                                          d_cstart, d_cend, d_pidx, d_xp, d_Jp, d_Vp0, last_gravity, P.rho, P.kappa, P.pdev * P.kappa, P.pdev_all, d_pdev_red);
        k_add_partials<<<1, 1>>>(bp, d_pdev_red, out_slot);
    }
    if(wall_active() && h_active_nodes > 0 && last_dt > 0.0)
    {
        int bw = blocks_for(h_active_nodes, t);
        k_wall_energy<<<bw, t, t * sizeof(double)>>>(h_active_nodes, node_slot_base, d_wall_t, d_wall_n, d_wall_xt, vertexes,
                                                     P.wall_mass * P.rho * P.dx * P.dx * P.dx / (last_dt * last_dt), d_wall_red,
                                                     P.wall_static_mass > 0.0 ? P.wall_static_mass / P.wall_mass : 1.0, P.wall_static_normal, P.wall_tang);
        k_add_partials<<<1, 1>>>(bw, d_wall_red, out_slot);
    }
    if(h_ag > 0 && last_dt > 0.0)
    {
        int bw = blocks_for(h_ag, t);
        gipc_liquid_wb::k_wb_aggr_energy<<<bw, t, t * sizeof(double)>>>(h_ag, d_wb_agn, d_wb_agr, d_wb_agt, d_wb_agnrm, d_wb_agc, P.dims, P.origin, P.dx, d_node_slot, node_slot_base, vertexes,
                                                                        P.wall_mass * P.rho * P.dx * P.dx * P.dx / (last_dt * last_dt), d_wb_agred);
        k_add_partials<<<1, 1>>>(bw, d_wb_agred, out_slot);
    }
    if(wfc_active() && h_wp > 0 && d_overt && last_dt > 0.0)
    {
        int bw = blocks_for(n_particles, t);
        k_wfc_energy<<<bw, t, t * sizeof(double)>>>(n_particles, d_wp_valid, d_wp_tri, d_wp_b, d_wp_n, d_wp_cell, d_wp_xi, d_wp_vid, d_faces, vertexes,
                                                    d_overt, P.origin, P.dx, P.wfc_k * P.rho * P.dx * P.dx * P.dx / (last_dt * last_dt), d_wp_red,
                                                    P.wfc_al ? d_wp_lam : nullptr, P.wfc_k * P.rho * P.dx * P.dx * P.dx);
        k_add_partials<<<1, 1>>>(bw, d_wp_red, out_slot);
    }
    if(wfc_active() && P.gp > 0.0 && h_gp > 0)
    {
        int bw = blocks_for(h_gp, t);
        k_gp_energy<<<bw, t, t * sizeof(double)>>>(h_gp, d_gp_pairs, P.dims, d_node_slot, node_slot_base, P.dx, P.origin, vertexes,
                                                   P.gp * P.kappa * P.dx * P.dx * P.dx, P.gp_full, d_gp_cutn, (P.gp_full == 6 || P.gp_full == 7) ? d_gp_w : nullptr, d_gp_red, (P.cpic && P.cpic_fields == 2) ? n_nodes : 0, d_plabel, d_nlab);
        k_add_partials<<<1, 1>>>(bw, d_gp_red, out_slot);
    }
    if(wfc_active() && P.gm && h_gm > 0 && d_overt && last_dt > 0.0)
    {
        const int t = 256;
        int bw = blocks_for(h_gm, t);
        k_gm_energy<<<bw, t, t * sizeof(double)>>>(h_gm, d_gm_list, d_active_nodes, d_node_slot, node_slot_base, P.dims, P.origin, P.dx, vertexes, d_overt,
                                                   d_wp_X, d_wp_n, d_wp_tri, d_wp_b, d_faces, P.gm_k * P.rho * P.dx * P.dx * P.dx / (last_dt * last_dt), d_gm_red);
        k_add_partials<<<1, 1>>>(bw, d_gm_red, out_slot);
    }
    if(coh_active())
    {
        int bw = blocks_for(h_coh, t);
        k_coh_energy<<<bw, t, t * sizeof(double)>>>(h_coh, d_coh_pairs, d_coh_w, vertexes, particle_slot_base, pair_k(),
                                                    pair_r0(), pair_rc(), d_coh_red);
        k_add_partials<<<1, 1>>>(bw, d_coh_red, out_slot);
    }
    if(stm_active())
    {
        int bw = blocks_for(h_stm, t);
        k_stm_energy<<<bw, t, t * sizeof(double)>>>(h_stm, d_stm_p, d_stm_a, d_stm_b, d_xp, P.origin, P.dx, P.dims, d_node_slot, node_slot_base,
                                                    vertexes, P.stm_sigma, d_stm_red);
        k_add_partials<<<1, 1>>>(bw, d_stm_red, out_slot);
    }
    g_prof.stop(4);
}

void LiquidSystem::hydrostatic_probe(const device_TetraData& m) const
{
    {   // LIQ_FORCE_PROBE at the converged iterate of the same step (tag _conv)
        const char* fpre = getenv("LIQ_FORCE_PROBE");
        if(fpre && *fpre && steps_done == (int)liquid_env("LIQ_FORCE_PROBE_STEP", -1) && h_active_nodes > 0 && last_dt > 0.0)
            const_cast<LiquidSystem*>(this)->force_probe(m.vertexes, 0, last_dt, 0, nullptr, nullptr, nullptr, (std::string(fpre) + "_conv").c_str());
    }
    const char* prefix = getenv("LIQ_HYDRO_PROBE");
    if(!prefix || !*prefix || steps_done != liquid_env("LIQ_HYDRO_PROBE_STEP", 2000)) return;
    // Binary layout is described by the adjacent JSON. Copies do not mutate solver state.
    // All vertex ids in the stencils remain GLOBAL ids, including ABD wall vertices.
    std::string stem = std::string(prefix) + "_" + std::to_string(steps_done);
    std::ofstream meta(stem + ".json"), data(stem + ".bin", std::ios::binary);
    if(!meta || !data) throw std::runtime_error("cannot open liquid hydrostatic probe output");
    meta.precision(17);
    meta << "{\"step\":" << steps_done << ",\"dt\":" << last_dt << ",\"particle_quadrature\":0"
         << ",\"dx\":" << P.dx << ",\"origin\":[" << P.origin.x << ',' << P.origin.y << ',' << P.origin.z
         << "],\"dims\":[" << P.dims.x << ',' << P.dims.y << ',' << P.dims.z
         << "],\"node_base\":" << node_slot_base << ",\"kappa\":" << P.kappa
         << ",\"wfc_K\":" << P.wfc_k * P.rho * P.dx * P.dx * P.dx
         << ",\"tension_ratio\":" << P.tension_ratio << ",\"arrays\":[";
    bool first = true;
    auto dump = [&](const char* name, const char* type, const void* ptr, size_t count, size_t width) {
        if(!first) meta << ',';
        first = false;
        meta << "{\"name\":\"" << name << "\",\"dtype\":\"" << type << "\",\"count\":" << count << '}';
        std::vector<char> buffer(count * width);
        if(count) CUDA_SAFE_CALL(cudaMemcpy(buffer.data(), ptr, buffer.size(), cudaMemcpyDeviceToHost));
        data.write(buffer.data(), buffer.size());
    };
    const size_t nv = node_slot_base + h_active_nodes;
    dump("x", "<f8", m.vertexes, 3 * nv, sizeof(double));
    dump("x0", "<f8", m.o_vertexes, 3 * nv, sizeof(double));
    dump("xt", "<f8", m.xTilta, 3 * nv, sizeof(double));
    dump("mass", "<f8", m.masses, nv, sizeof(double));
    dump("active_nodes", "<i4", d_active_nodes, h_active_nodes, sizeof(int));
    dump("real_mass", "<f8", d_mi, n_nodes, sizeof(double));
    dump("vox_nodes", "<i4", d_vox_nodes, 8 * h_elements, sizeof(int));
    dump("vox_ijk", "<i4", d_vox_ijk, 3 * h_elements, sizeof(int));
    dump("vox_V", "<f8", d_vox_V, h_elements, sizeof(double));
    dump("vox_Jb", "<f8", d_vox_Jb, h_elements, sizeof(double));
    dump("xp", "<f8", d_xp, 3 * n_particles, sizeof(double));
    dump("Jp", "<f8", d_Jp, n_particles, sizeof(double));
    if(P.wb && d_vox_a && d_vox_vg && d_vox_f && d_vox_w && d_node_w)
    {   // WB element data (cut flag, area vectors, sub-volume, hydrostatic forces, corner weights) + per-lattice-node weights
        dump("wb_vox_cc", "<i4", d_vox_cc, h_elements, sizeof(int));
        dump("wb_vox_a", "<f8", d_vox_a, 3 * 8 * (size_t)h_elements, sizeof(double));
        dump("wb_vox_vg", "<f8", d_vox_vg, h_elements, sizeof(double));
        dump("wb_vox_f", "<f8", d_vox_f, 3 * 8 * (size_t)h_elements, sizeof(double));
        dump("wb_vox_w", "<f8", d_vox_w, 8 * (size_t)h_elements, sizeof(double));
        dump("wb_node_w", "<f8", d_node_w, n_nodes, sizeof(double));
        if(P.wb_geov && d_vox_Vr) dump("wb_vox_Vr", "<f8", d_vox_Vr, h_elements, sizeof(double));
        if(d_wb_phi) dump("wb_phi", "<f8", d_wb_phi, n_nodes, sizeof(double));             // nodal signed wall distance (valid on active nodes)
        if(d_wb_n) dump("wb_nrm", "<f8", d_wb_n, 3 * (size_t)n_nodes, sizeof(double));
        if(d_wb_cvgeo) dump("wb_cvgeo", "<f8", d_wb_cvgeo, n_cells, sizeof(double));       // per cell geometric fluid volume without the surface cut
        if(d_wb_cflag) dump("wb_cflag", "<i4", d_wb_cflag, n_cells, sizeof(int));
        {   // per lattice column (along up): raw / final surface height, LIQ_WB_OVH flag
            const double ax = fabs(last_gravity.x), ay = fabs(last_gravity.y), az = fabs(last_gravity.z);
            const int    U  = (ay > ax && ay >= az) ? 1 : ((az > ax && az > ay) ? 2 : 0);
            const size_t ncol = (U == 0) ? (size_t)P.dims.y * P.dims.z : (U == 1) ? (size_t)P.dims.x * P.dims.z : (size_t)P.dims.x * P.dims.y;
            if(d_wb_hraw) dump("wb_hraw", "<f8", d_wb_hraw, ncol, sizeof(double));
            if(d_wb_hcol) dump("wb_hcol", "<f8", d_wb_hcol, ncol, sizeof(double));
            if(d_wb_ovh) dump("wb_ovh", "<i4", d_wb_ovh, ncol, sizeof(int));
            if(d_wb_lab) dump("wb_lab", "<i4", d_wb_lab, n_nodes, sizeof(int));
            if(d_wb_pcomp) dump("wb_pcomp", "<i4", d_wb_pcomp, n_particles, sizeof(int));
        }
    }
    if(P.cpic && d_pmask)
    {   // 2b debugging: masks, near flags, cut axes, corner votes, second-set volumes, ghost corners
        dump("pmask", "<i4", d_pmask, n_particles, sizeof(int));
        dump("wp_near", "<i4", d_wp_near, n_particles, sizeof(int));
        dump("cell_axis", "<i4", d_cell_axis, n_cells, sizeof(int));
        dump("corner_vote", "<f4", d_corner_vote, 32 * (size_t)n_cells, sizeof(float));
        dump("Vc", "<f8", d_Vc, n_cells, sizeof(double));
        dump("Vc2", "<f8", d_Vc2, n_cells, sizeof(double));
        dump("mc2", "<f8", d_mc2, n_cells, sizeof(double));
        dump("ghost_corner", "<i4", d_ghost_corner, 3 * h_ghost_slots, sizeof(int));
        dump("node_slot", "<i4", d_node_slot, n_nodes, sizeof(int));
    }
    if(wfc_active())
    {
        dump("wp_valid", "<i4", d_wp_valid, n_particles, sizeof(int));
        dump("wp_tri", "<i4", d_wp_tri, n_particles, sizeof(int));
        dump("wp_b", "<f8", d_wp_b, 3 * n_particles, sizeof(double));
        dump("wp_n", "<f8", d_wp_n, 3 * n_particles, sizeof(double));
        dump("wp_cell", "<i4", d_wp_cell, 3 * n_particles, sizeof(int));
        dump("wp_xi", "<f8", d_wp_xi, 3 * n_particles, sizeof(double));
        dump("wp_vid", "<i4", d_wp_vid, 8 * n_particles, sizeof(int));
        if(d_wp_X) dump("wp_X", "<f8", d_wp_X, 3 * n_particles, sizeof(double));
        dump("faces", "<u4", d_faces, 3 * n_faces, sizeof(unsigned));
    }
    meta << "]}";
    data.flush(); meta.flush();
    if(!data || !meta) throw std::runtime_error("failed writing liquid hydrostatic probe output");
    printf("[liquid] converged-state hydrostatic snapshot: %s\n", stem.c_str());
}

void LiquidSystem::post_step(device_TetraData& m, double dt)
{
    if(P.wb_aggr_debug && h_ag > 0 && d_node_slot)
    {   // LIQ_WB_AGGR_DEBUG: how well the aggregation constraints hold after the solve
        if(!d_wb_agdbg) CUDA_SAFE_CALL(cudaMalloc((void**)&d_wb_agdbg, 4 * sizeof(double)));
        CUDA_SAFE_CALL(cudaMemsetAsync(d_wb_agdbg, 0, 4 * sizeof(double)));
        gipc_liquid_wb::k_wb_aggr_residual<<<blocks_for(h_ag), 256>>>(h_ag, d_wb_agn, d_wb_agr, d_wb_agt, d_wb_agnrm, d_wb_agc, P.dims, P.origin, P.dx, d_node_slot,
                                                                      node_slot_base, m.vertexes, d_wb_agdbg);
        if(steps_done % 200 == 0)
        {
            double hd[4];
            CUDA_SAFE_CALL(cudaMemcpy(hd, d_wb_agdbg, 4 * sizeof(double), cudaMemcpyDeviceToHost));
            printf("[liquid] aggregation residual (step %d, %d constraints): max %.3e m, mean %.3e m, max on-wall %.3e m; max |u_s| %.3e m (dt %.1e)\n", steps_done, h_ag,
                   hd[0], hd[1] / h_ag, hd[2], hd[3], dt);
        }
    }
    if(P.ug)
    {
        ug_post(m, dt);
        return;
    }
    const int n = n_particles;
    g_prof.start();
    CUDA_SAFE_CALL(cudaMemsetAsync(d_vi, 0, n_vnodes * sizeof(double3)));
    if(h_active_nodes > 0 && P.bdf2)
    {
        CUDA_SAFE_CALL(cudaMemsetAsync(d_bh_vel, 0, n_vnodes * sizeof(double3)));
        k_read_node_vel_bdf2<<<blocks_for(h_active_nodes), 256>>>(h_active_nodes, d_active_nodes, m.vertexes, d_bh_xt, d_bh_vb, P.dims, P.dx, P.origin,
                                                                  node_slot_base, dt, d_vi, d_bh_vel, P.bdf2_wall_be ? d_bh_wall : nullptr);
    }
    else if(h_active_nodes > 0)
        k_read_node_vel<<<blocks_for(h_active_nodes), 256>>>(h_active_nodes, d_active_nodes, m.vertexes, P.dims, P.dx, P.origin, node_slot_base, dt, d_vi, n_nodes);
    double3 lo = make_double3(P.origin.x, fmax(P.origin.y - 0.5 * P.dx, P.clamp_lo_y), P.origin.z);
    double3 hi = make_double3(P.origin.x + P.dims.x * P.dx - 1e-9, P.origin.y + P.dims.y * P.dx - 1e-9, P.origin.z + P.dims.z * P.dx - 1e-9);
    if(P.isl_v > 0.0)
    {
        CUDA_SAFE_CALL(cudaMemcpyAsync(d_xold, d_xp, n * sizeof(double3), cudaMemcpyDeviceToDevice));
        CUDA_SAFE_CALL(cudaMemcpyAsync(d_Jold, d_Jp, n * sizeof(double), cudaMemcpyDeviceToDevice));
    }
    k_g2p<<<blocks_for(n), 256>>>(n, d_xp, d_vp, d_Cp, d_Jp, d_vi, d_vi_old, P.flip, d_node_flag, P.dims, P.dx, P.origin, dt, P.apic, P.apic_ratio,
                                  P.rest_damp, P.rest_v, d_sleep, P.sleep_v, P.wake_v, P.sleep_n, P.isl_v > 0.0 ? d_vnmag : nullptr,
                                  (P.j_mode == 0 || P.j_mode == 3) ? P.J_track_min : P.J_min,
                                  (P.j_mode == 0 || P.j_mode == 3) ? P.J_track_max : 2.0, lo, hi, d_pmask, d_pwinv,
                                  P.cpic ? d_wp_near : nullptr, d_wp_X, d_wp_n, d_wp_tri, d_wp_b, d_faces, m.vertexes, m.o_vertexes, P.cpic_c,
                                  d_nref_has, d_nref_X, d_nref_n, (P.cpic && P.cpic_fields == 2) ? n_nodes : 0, d_plabel, d_nlab,
                                  P.bdf2 ? d_bh_vel : nullptr, P.bdf2 ? d_bh_vp : nullptr, P.bdf2 ? d_bh_Cp : nullptr,
                                  (P.wb && P.wb_g2p && d_cell2elem) ? d_cell2elem : nullptr, d_vox_cc, d_vox_w, d_vox_a, d_vox_vg,
                                  (P.wb_g2p == 2) ? d_vox_wc : nullptr, P.wb_g2p_fmin, P.dx * P.dx * P.dx,
                                  (P.wb && (P.g2p_fluid || P.g2p_jcell) && d_cell2elem) ? d_cell2elem : nullptr, d_vox_cc, d_vox_vg,
                                  P.g2p_jcell, d_vox_a, P.g2p_fluid);
    porous_post(m, dt);   // sponge: pore water follows the foam (kinematic retention), excess keeps the solved motion
    if(P.jfilt > 0.0 && P.wb && d_wb_hcol && d_wb_blo)
    {   // LIQ_JFILT (see k_jf_deposit): hydrostatically corrected null-space filter of the tracked particle J
        const double gm = sqrt(last_gravity.x * last_gravity.x + last_gravity.y * last_gravity.y + last_gravity.z * last_gravity.z);
        if(gm > 0.0)
        {
            const double3 up = make_double3(-last_gravity.x / gm, -last_gravity.y / gm, -last_gravity.z / gm);
            const double  ax = fabs(up.x), ay = fabs(up.y), az = fabs(up.z);
            const int     U  = (ay > ax && ay >= az) ? 1 : ((az > ax && az > ay) ? 2 : 0);
            const double  rgk = P.rho * gm / P.kappa;
            if(!d_jf_num)
            {
                CUDA_SAFE_CALL(cudaMalloc((void**)&d_jf_num, (size_t)n_cells * sizeof(double)));
                CUDA_SAFE_CALL(cudaMalloc((void**)&d_jf_den, (size_t)n_cells * sizeof(double)));
                CUDA_SAFE_CALL(cudaMalloc((void**)&d_jf_flag, (size_t)n_particles * sizeof(int)));
                CUDA_SAFE_CALL(cudaMalloc((void**)&d_jf_red, 2 * sizeof(double)));
                printf("[liquid] J null-space filter ON (LIQ_JFILT %.2f): particle J deviation from the column hydrostatic J relaxed to its cell-centre average\n", P.jfilt);
            }
            CUDA_SAFE_CALL(cudaMemsetAsync(d_jf_num, 0, (size_t)n_cells * sizeof(double)));
            CUDA_SAFE_CALL(cudaMemsetAsync(d_jf_den, 0, (size_t)n_cells * sizeof(double)));
            CUDA_SAFE_CALL(cudaMemsetAsync(d_jf_red, 0, 2 * sizeof(double)));
            k_jf_deposit<<<blocks_for(n), 256>>>(n, d_xp, d_Jp, d_Vp0, P.dims, P.origin, P.dx, U, up, d_wb_hcol, d_wb_blo, rgk, d_jf_num, d_jf_den);
            k_jf_apply<<<blocks_for(n), 256>>>(n, d_xp, d_Jp, d_Vp0, P.dims, P.origin, P.dx, U, up, d_wb_hcol, d_wb_blo, rgk, d_jf_num, d_jf_den, P.jfilt, d_jf_flag, d_jf_red);
            k_jf_conserve<<<blocks_for(n), 256>>>(n, d_Jp, d_jf_flag, d_jf_red);
            CUDA_SAFE_CALL(cudaGetLastError());
        }
    }
    if(P.isl_v > 0.0) island_pass(dt);
    if(P.sleep_v > 0.0 && steps_done % 200 == 0)
    {
        int h_asleep = 0;
        CUDA_SAFE_CALL(cudaMemsetAsync(d_sleep_count, 0, sizeof(int)));
        k_count_asleep<<<blocks_for(n), 256>>>(n, d_sleep, P.sleep_n, d_sleep_count);
        CUDA_SAFE_CALL(cudaMemcpy(&h_asleep, d_sleep_count, sizeof(int), cudaMemcpyDeviceToHost));
        printf("[liquid] rest sleep: %d / %d particles asleep (step %d)\n", h_asleep, n, steps_done);
    }
    if(P.j_mode == 1)
    {
        build_neighbor_grid();
        density_pass(false);
    }
    if(P.stick_h > 0.0)
    {
        k_stick<<<blocks_for(n), 256>>>(n, d_xp, d_vp, d_Cp, d_stuck, d_xstuck, lo.y, P.stick_h, d_stuck_count);
        if(getenv("LIQ_DEBUG") && steps_done % 50 == 0)
        {
            CUDA_SAFE_CALL(cudaMemcpy(&h_stuck, d_stuck_count, sizeof(int), cudaMemcpyDeviceToHost));
            printf("[liquid] step %d: %d particles stuck on the table\n", steps_done, h_stuck);
        }
    }
    k_write_particle_slots<<<blocks_for(n), 256>>>(n, d_xp, particle_slot_base, m.vertexes, m.o_vertexes, m.xTilta);
    CUDA_SAFE_CALL(cudaGetLastError());
    g_prof.stop(5);
    g_prof.step_done();
    CUDA_SAFE_CALL(cudaMemcpy(h_fallback, d_fallback, 12 * sizeof(int), cudaMemcpyDeviceToHost));
    steps_done++;
}

// ---------------------------------------------------------------------------- LIQ_UG (liquid_ug.cuh)
void LiquidSystem::ug_pre(device_TetraData& m, double dt, double3 gravity)
{
    const int n = n_particles;
    last_dt     = dt;   // k_ug_energy (LIQ_UG_HG) needs the step size
    if(!d_ug_nodes)
    {
        int tri = P.ug_bspline ? 2 : 1;
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_liq_tri, &tri, sizeof(int)));
        const size_t ns = (size_t)ug_ns();
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_ug_nodes, ns * (size_t)n * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_ug_w, ns * (size_t)n * sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_ug_grad, ns * (size_t)n * sizeof(double3)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_ug_red, (blocks_for(n) + 1) * sizeof(double)));
        if((long long)ug_nb() * n > 37ll * max_voxels)
            throw std::runtime_error("[liquid] LIQ_UG needs 36 (trilinear) / 378 (B-spline) triplet blocks per particle: the liquid triplet reservation (37 x max_voxels) is too small");
        printf("[liquid] LIQ_UG particle-quadrature liquid (recovered Unified_GIPC method): %s, kappa %.3g, tension ratio %.2f, "
               "volume barrier %s (J_act %.3f, k %.2f kappa), node mass floor %.3f rho dx^3; wall / rest / push-apart layers off\n",
               P.ug_bspline ? "quadratic B-spline 27-node stencil (LIQ_UG_BSPLINE=1)" : "trilinear 8-node stencil",
               P.kappa, P.tension_ratio, P.ug_jact > 0.0 ? "ON" : "off", P.ug_jact, P.ug_kb, P.mass_floor);
        if(P.ug_reanchor > 0.0) printf("[liquid] LIQ_UG density re-anchor of interior particles: beta %.2f per step\n", P.ug_reanchor);
        if(P.ug_hg > 0.0) printf("[liquid] LIQ_UG hourglass viscosity %.3g Pa s on the trilinear stencil's zero-energy modes%s\n", P.ug_hg, P.ug_bspline ? " (IGNORED with the B-spline stencil)" : "");
        printf("[liquid] LIQ_UG volume feasible step: %s\n", P.ug_feasible > 0.0 ? "ON (line search bounded at the first inverted particle, safety factor applied)" : "off (infinite-energy rejection only)");
    }
    CUDA_SAFE_CALL(cudaMemsetAsync(d_mi, 0, n_vnodes * sizeof(double)));
    CUDA_SAFE_CALL(cudaMemsetAsync(d_mvi, 0, n_vnodes * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMemsetAsync(d_node_ref, 0, n_vnodes * sizeof(int)));
    if(P.ug_bspline) k_ug_p2g<3><<<blocks_for(n), 256>>>(n, d_xp, d_vp, d_Cp, d_mp, P.origin, P.dx, P.dims, P.apic, d_mi, d_mvi, d_node_ref);
    else             k_ug_p2g<2><<<blocks_for(n), 256>>>(n, d_xp, d_vp, d_Cp, d_mp, P.origin, P.dx, P.dims, P.apic, d_mi, d_mvi, d_node_ref);
    if(P.bdf2)
    {   // LIQ_BDF2 in UG mode: previous-step particle velocities on this step's stencils -> d_bh_mvi (same weights as d_mvi)
        if(!bh_init)
        {
            CUDA_SAFE_CALL(cudaMemcpyAsync(d_bh_vp, d_vp, n * sizeof(double3), cudaMemcpyDeviceToDevice));
            CUDA_SAFE_CALL(cudaMemcpyAsync(d_bh_Cp, d_Cp, 9 * (size_t)n * sizeof(double), cudaMemcpyDeviceToDevice));
            bh_init = true;
        }
        if(!d_ug_hmass)
        {
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_ug_hmass, n_vnodes * sizeof(double)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_ug_href, n_vnodes * sizeof(int)));
        }
        CUDA_SAFE_CALL(cudaMemsetAsync(d_bh_mvi, 0, n_vnodes * sizeof(double3)));
        CUDA_SAFE_CALL(cudaMemsetAsync(d_ug_hmass, 0, n_vnodes * sizeof(double)));
        if(P.ug_bspline) k_ug_p2g<3><<<blocks_for(n), 256>>>(n, d_xp, d_bh_vp, d_bh_Cp, d_mp, P.origin, P.dx, P.dims, P.apic, d_ug_hmass, d_bh_mvi, d_ug_href);
        else             k_ug_p2g<2><<<blocks_for(n), 256>>>(n, d_xp, d_bh_vp, d_bh_Cp, d_mp, P.origin, P.dx, P.dims, P.apic, d_ug_hmass, d_bh_mvi, d_ug_href);
    }
    k_node_flag<<<blocks_for(n_vnodes), 256>>>(n_vnodes, d_mi, d_node_ref, 0, d_node_flag, d_node_ghost);
    CUDA_SAFE_CALL(cudaMemsetAsync(d_cell_flag, 0, n_cells * sizeof(int)));
    thrust::exclusive_scan(thrust::device, thrust::device_pointer_cast(d_cell_flag), thrust::device_pointer_cast(d_cell_flag) + n_cells, thrust::device_pointer_cast(d_cell_scan));
    thrust::exclusive_scan(thrust::device, thrust::device_pointer_cast(d_node_flag), thrust::device_pointer_cast(d_node_flag) + n_vnodes, thrust::device_pointer_cast(d_node_scan));
    k_compact_nodes<<<blocks_for(n_vnodes), 256>>>(n_vnodes, d_node_flag, d_node_scan, d_active_nodes, d_node_slot);
    k_counts<<<1, 1>>>(n_cells, n_vnodes, d_cell_flag, d_cell_scan, d_node_flag, d_node_scan, d_counts);
    int counts[2] = {0, 0};
    CUDA_SAFE_CALL(cudaMemcpy(counts, d_counts, 2 * sizeof(int), cudaMemcpyDeviceToHost));
    h_active_cells = 0;
    h_active_nodes = counts[1];
    if(h_active_nodes > P.node_pool)
    {
        char buf[256];
        snprintf(buf, sizeof(buf), "[liquid] %d active grid nodes exceed the node pool (%d): raise node_pool", h_active_nodes, P.node_pool);
        throw std::runtime_error(buf);
    }
    h_elements = 0; h_ghost_slots = 0; h_live_nodes = h_active_nodes; h_ag = 0; h_cc = 0;
    if(d_wall_count) CUDA_SAFE_CALL(cudaMemsetAsync(d_wall_count, 0, sizeof(int)));
    k_write_node_slots<<<blocks_for(P.node_pool), 256>>>(P.node_pool, h_active_nodes, d_active_nodes, d_mi, d_mvi, P.dims, P.dx, P.origin, P.park,
                                                         P.mass_floor, P.floor_gravity, nullptr, 0.0, 0.0,
                                                         P.cav_on, P.cav[0], P.cav[1], P.cav[2], P.cav[3], P.cav[4], P.cav[5], P.cav[6],
                                                         nullptr, 0, d_ghost_corner, n_nodes, nullptr, 0, d_abd_q, d_abd_qv, 0, P.clamp_lo_y,
                                                         nullptr, d_wall_n, d_wall_xt, nullptr,
                                                         P.rho, 0.0, P.rest_kv, dt, gravity, node_slot_base, m.vertexes, m.o_vertexes, m.velocities,
                                                         m.xTilta, m.masses, m.BoundaryType, P.bdf2 ? d_bh_mvi : nullptr, P.bdf2 ? d_bh_vb : nullptr,
                                                         P.bdf2 ? d_bh_xt : nullptr, nullptr, nullptr, nullptr, P.wb_airgrav);
    fp_xt = m.xTilta; fp_mass = m.masses;
    wfc_x = m.vertexes;
    d_overt = m.o_vertexes;
    CUDA_SAFE_CALL(cudaMemsetAsync(d_counters, 0, sizeof(int)));
    if(P.ug_bspline) k_ug_stencil<3><<<blocks_for(n), 256>>>(n, d_xp, P.origin, P.dx, P.dims, d_node_slot, node_slot_base, d_ug_nodes, d_ug_w, d_ug_grad, d_counters);
    else             k_ug_stencil<2><<<blocks_for(n), 256>>>(n, d_xp, P.origin, P.dx, P.dims, d_node_slot, node_slot_base, d_ug_nodes, d_ug_w, d_ug_grad, d_counters);
    int bad = 0;
    CUDA_SAFE_CALL(cudaMemcpy(&bad, d_counters, sizeof(int), cudaMemcpyDeviceToHost));
    if(bad) throw std::runtime_error("[liquid] LIQ_UG: a particle stencil node with positive weight is not active");
    k_write_particle_slots<<<blocks_for(n), 256>>>(n, d_xp, particle_slot_base, m.vertexes, m.o_vertexes, m.xTilta);
    CUDA_SAFE_CALL(cudaGetLastError());
    if(steps_done % 200 == 0) printf("[liquid] LIQ_UG step %d: %d active grid nodes, %d particles; volume feasible step engaged in %lld Newton iterations so far\n", steps_done, h_active_nodes, n, feasible_hits);
}

__global__ void k_min_partials(int n, const double* partial, double* out)
{
    double a = 1.0;
    for(int i = 0; i < n; ++i) a = fmin(a, partial[i]);
    out[0] = a;
}
__global__ void k_set_one(double* out) { out[0] = 1.0; }

// device-side variant: writes the bound into one slot of the engine's CCD alpha chain (no extra D2H per Newton iteration)
void LiquidSystem::feasible_alpha_device(const double3* vertexes, const double3* dir, double* out_slot)
{
    if(!P.ug || P.ug_feasible <= 0.0 || n_particles <= 0 || h_active_nodes <= 0 || !d_ug_nodes) { k_set_one<<<1, 1>>>(out_slot); return; }
    const int t = 256, b = blocks_for(n_particles, t);
    const UgGrid G{d_active_nodes, node_slot_base, P.dims, P.origin, P.dx};
    if(P.ug_bspline) k_ug_feasible<3><<<b, t, t * sizeof(double)>>>(n_particles, d_ug_nodes, d_ug_grad, vertexes, dir, G, P.ug_feasible, d_ug_red);
    else             k_ug_feasible<2><<<b, t, t * sizeof(double)>>>(n_particles, d_ug_nodes, d_ug_grad, vertexes, dir, G, P.ug_feasible, d_ug_red);
    k_min_partials<<<1, 1>>>(b, d_ug_red, out_slot);
}

double LiquidSystem::feasible_alpha(const double3* vertexes, const double3* dir)
{
    if(!P.ug || P.ug_feasible <= 0.0 || n_particles <= 0 || h_active_nodes <= 0 || !d_ug_nodes) return 1.0;
    const int t = 256, b = blocks_for(n_particles, t);
    const UgGrid G{d_active_nodes, node_slot_base, P.dims, P.origin, P.dx};
    if(P.ug_bspline) k_ug_feasible<3><<<b, t, t * sizeof(double)>>>(n_particles, d_ug_nodes, d_ug_grad, vertexes, dir, G, P.ug_feasible, d_ug_red);
    else             k_ug_feasible<2><<<b, t, t * sizeof(double)>>>(n_particles, d_ug_nodes, d_ug_grad, vertexes, dir, G, P.ug_feasible, d_ug_red);
    std::vector<double> h(b);
    CUDA_SAFE_CALL(cudaMemcpy(h.data(), d_ug_red, b * sizeof(double), cudaMemcpyDeviceToHost));
    double a = 1.0;
    for(double v : h) a = std::min(a, v);
    return a;
}

void LiquidSystem::ug_post(device_TetraData& m, double dt)
{
    const int n = n_particles;
    g_prof.start();
    double3 lo = make_double3(P.origin.x, fmax(P.origin.y - 0.5 * P.dx, P.clamp_lo_y), P.origin.z);
    double3 hi = make_double3(P.origin.x + P.dims.x * P.dx - 1e-9, P.origin.y + P.dims.y * P.dx - 1e-9, P.origin.z + P.dims.z * P.dx - 1e-9);
    CUDA_SAFE_CALL(cudaMemsetAsync(d_counters, 0, sizeof(int)));
    if(h_active_nodes > 0)
    {
        const double3* vvel = nullptr;
        if(P.bdf2)
        {   // BDF2 velocity form: v_i = vb_i + 3/2 (x_i - x~_i) / dt per active node, scattered per lattice node id
            CUDA_SAFE_CALL(cudaMemsetAsync(d_bh_vel, 0, n_vnodes * sizeof(double3)));
            CUDA_SAFE_CALL(cudaMemsetAsync(d_vi, 0, n_vnodes * sizeof(double3)));
            k_read_node_vel_bdf2<<<blocks_for(h_active_nodes), 256>>>(h_active_nodes, d_active_nodes, m.vertexes, d_bh_xt, d_bh_vb, P.dims, P.dx, P.origin,
                                                                      node_slot_base, dt, d_vi, d_bh_vel, nullptr);
            vvel = d_bh_vel;
        }
        const UgGrid G{d_active_nodes, node_slot_base, P.dims, P.origin, P.dx};
        if(P.ug_bspline) k_ug_commit<3><<<blocks_for(n), 256>>>(n, d_ug_nodes, d_ug_w, d_ug_grad, m.vertexes, G, dt, P.apic, lo, hi, d_xp, d_vp, d_Cp, d_Jp, d_counters,
                                                                vvel, P.bdf2 ? d_bh_vp : nullptr, P.bdf2 ? d_bh_Cp : nullptr);
        else             k_ug_commit<2><<<blocks_for(n), 256>>>(n, d_ug_nodes, d_ug_w, d_ug_grad, m.vertexes, G, dt, P.apic, lo, hi, d_xp, d_vp, d_Cp, d_Jp, d_counters,
                                                                vvel, P.bdf2 ? d_bh_vp : nullptr, P.bdf2 ? d_bh_Cp : nullptr);
    }
    int bad = 0;
    CUDA_SAFE_CALL(cudaMemcpy(&bad, d_counters, sizeof(int), cudaMemcpyDeviceToHost));
    if(bad) throw std::runtime_error("[liquid] LIQ_UG commit: non-positive or non-finite particle volume / displacement");
    if(P.ug_reanchor > 0.0 && h_active_nodes > 0)
    {   // density re-anchor (LIQ_UG_REANCHOR): interior particles take the tent-smeared cell volume ratio (j_mode 3 of the main path)
        CUDA_SAFE_CALL(cudaMemsetAsync(d_mc, 0, n_cells * sizeof(double)));
        CUDA_SAFE_CALL(cudaMemsetAsync(d_cell_flag, 0, n_cells * sizeof(int)));
        k_ug_cellmass<<<blocks_for(n), 256>>>(n, d_xp, d_mp, P.origin, P.dx, P.dims, d_mc, d_cell_flag);
        k_reanchor_J<<<blocks_for(n), 256>>>(n, d_xp, d_mc, d_cell_flag, P.origin, P.dx, P.dims, P.rho * P.dx * P.dx * P.dx, P.J_min, P.ug_reanchor, d_Jp);
    }
    porous_post(m, dt);
    k_write_particle_slots<<<blocks_for(n), 256>>>(n, d_xp, particle_slot_base, m.vertexes, m.o_vertexes, m.xTilta);
    CUDA_SAFE_CALL(cudaGetLastError());
    g_prof.stop(5);
    g_prof.step_done();
    CUDA_SAFE_CALL(cudaMemcpy(h_fallback, d_fallback, 12 * sizeof(int), cudaMemcpyDeviceToHost));
    steps_done++;
}

double* LiquidSystem::local_step_alpha_buffer()
{   // LIQ_LOCAL_STEP: per-particle step bound, reset to 1 (the engine's pair kernel takes atomic minima into it)
    if(n_particles <= 0) return nullptr;
    if(!d_ls_alpha) CUDA_SAFE_CALL(cudaMalloc((void**)&d_ls_alpha, (size_t)n_particles * sizeof(double)));
    k_fill_double<<<blocks_for(n_particles), 256>>>(n_particles, d_ls_alpha, 1.0);
    return d_ls_alpha;
}
int LiquidSystem::local_step_filter(double3* md, const double3* vertexes, double cut, bool count_out)
{   // scale the direction of the grid nodes around step-limited particles (contact pairs / ground plane), then refresh the particle rows
    if(n_particles <= 0 || !d_ls_alpha || node_slot_base < 0 || P.ug) return 0;
    const int nslots = particle_slot_base - node_slot_base;
    if(nslots <= 0) return 0;
    if(!d_ls_scale || ls_scale_n != nslots)
    {
        if(d_ls_scale) cudaFree(d_ls_scale);
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_ls_scale, (size_t)nslots * sizeof(double)));
        ls_scale_n = nslots;
    }
    k_fill_double<<<blocks_for(nslots), 256>>>(nslots, d_ls_scale, 1.0);
    CUDA_SAFE_CALL(cudaMemsetAsync(d_counters + 3, 0, sizeof(int)));
    k_node_scale_min<<<blocks_for(n_particles), 256>>>(n_particles, d_ls_alpha, d_pbase, d_pfrac, d_pmask, d_pwinv, d_node_slot, P.dims, node_slot_base, particle_slot_base, md, vertexes,
                                                      P.clamp_lo_y, 0.9, cut, d_ls_scale, d_counters + 3,
                                                      d_xp, d_nref_has, d_nref_X, d_nref_n, (P.cpic && P.cpic_fields == 2) ? n_nodes : 0, d_plabel, d_nlab);
    k_node_scale_apply<<<blocks_for(nslots), 256>>>(nslots, d_ls_scale, node_slot_base, md);
    apply_particle_movedir(md);
    int limited = 0;
    if(count_out) CUDA_SAFE_CALL(cudaMemcpy(&limited, d_counters + 3, sizeof(int), cudaMemcpyDeviceToHost));
    return limited;
}

void LiquidSystem::completion_dir(double3* md, int vertex_num, double scale)
{   // LIQ_SPLIT_STEP: the liquid-only remainder of a Newton direction: solid rows zero, grid-node rows x scale (the particle rows are
    // re-derived from the node rows by apply_particle_movedir before the CCD)
    if(vertex_num <= 0 || node_slot_base < 0) return;
    k_scale_rows<<<blocks_for(vertex_num), 256>>>(vertex_num, md, md, 0, node_slot_base, 0.0);
    k_scale_rows<<<blocks_for(vertex_num), 256>>>(vertex_num, md, md, node_slot_base, particle_slot_base, scale);
}

void LiquidSystem::zero_rows(double3* md, int vertex_num)
{
    if(vertex_num <= 0) return;
    k_scale_rows<<<blocks_for(vertex_num), 256>>>(vertex_num, md, md, node_slot_base, particle_slot_base + n_particles, 0.0);
}

const double3* LiquidSystem::scaled_movedir(const double3* md, int vertex_num)
{
    static const int s_conv_particle = getenv("LIQ_CONV_PARTICLE") ? atoi(getenv("LIQ_CONV_PARTICLE")) : 0;
    if(P.conv_scale >= 1.0 && !s_conv_particle) return md;
    if(!d_md_scaled || n_vertices_total != vertex_num)
    {
        if(d_md_scaled) cudaFree(d_md_scaled);
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_md_scaled, vertex_num * sizeof(double3)));
        n_vertices_total = vertex_num;
    }
    if(s_conv_particle)
    {   // LIQ_CONV_PARTICLE=1: the liquid enters the Newton exit test through the PARTICLE-interpolated direction
        // (sum_i w_ip d_i, what the recovered engine's "physical direction" norm measures), not through the node rows.
        // Null-space (hourglass / sub-cell) node directions are nearly invisible at the particles, so the Newton stops
        // before resolving them: they stay unresolved = damped every step, which is how the recovered engine settles
        // while the particle-visible (physical) motion is still converged to the tolerance.
        CUDA_SAFE_CALL(cudaMemcpyAsync(d_md_scaled, md, (size_t)vertex_num * sizeof(double3), cudaMemcpyDeviceToDevice));
        apply_particle_movedir(d_md_scaled);                                                                      // particle rows <- interpolated node rows
        k_scale_rows<<<blocks_for(vertex_num), 256>>>(vertex_num, d_md_scaled, d_md_scaled, node_slot_base, particle_slot_base, 0.0);
        k_scale_rows<<<blocks_for(vertex_num), 256>>>(vertex_num, d_md_scaled, d_md_scaled, particle_slot_base, particle_slot_base + n_particles, P.conv_scale);
        return d_md_scaled;
    }
    k_scale_rows<<<blocks_for(vertex_num), 256>>>(vertex_num, md, d_md_scaled, node_slot_base, particle_slot_base + n_particles, P.conv_scale);
    return d_md_scaled;
}

int LiquidSystem::get_wall_forces(double* out3, int n, double dt) const
{
    // WFC gradient of the LAST Newton assembly on the solid vertices, converted to Newtons (-grad / dt^2). The
    // assembly precedes the final line search, so this is the reaction at the last iterate, not the accepted one
    // (the difference vanishes at convergence). Zero when WFC is off or no wall point was active.
    if(n <= 0) return 0;
    std::fill(out3, out3 + 3 * (size_t)n, 0.0);
    if(!d_wfc_fsolid || wfc_fsolid_n <= 0 || !(dt > 0.0)) return 0;
    const int m = std::min(n, wfc_fsolid_n);
    std::vector<double3> h(m);
    CUDA_SAFE_CALL(cudaMemcpy(h.data(), d_wfc_fsolid, (size_t)m * sizeof(double3), cudaMemcpyDeviceToHost));
    const double sc = -1.0 / (dt * dt);
    for(int i = 0; i < m; ++i) { out3[3 * i] = sc * h[i].x; out3[3 * i + 1] = sc * h[i].y; out3[3 * i + 2] = sc * h[i].z; }
    return m;
}

void LiquidSystem::get_ghost_reaction(const double3* vertexes, double* out3)
{
    out3[0] = out3[1] = out3[2] = 0.0;
    if(!P.cpic || !d_node_ghost || h_elements <= 0 || !vertexes) return;
    if(!d_pump_red) CUDA_SAFE_CALL(cudaMalloc((void**)&d_pump_red, 4 * sizeof(double)));
    CUDA_SAFE_CALL(cudaMemset(d_pump_red, 0, 4 * sizeof(double)));
    k_vox_ghost_force<<<blocks_for(h_elements), 256>>>(h_elements, d_vox_nodes, d_vox_V, d_vox_Jb, d_vox_ijk, vertexes, P.dx, P.origin, P.kappa,
                                                          P.tension_ratio, node_slot_base, d_active_nodes, d_node_ghost, h_active_nodes, d_pump_red);
    CUDA_SAFE_CALL(cudaMemcpy(out3, d_pump_red, 3 * sizeof(double), cudaMemcpyDeviceToHost));
}

int LiquidSystem::count_in_pump()
{
    int* cnt = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void**)&cnt, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(cnt, 0, sizeof(int)));
    k_pump<0><<<blocks_for(n_particles), 256>>>(n_particles, d_xp, d_vp, pump_a, pump_b, pump_r, 0.0, 0.0, make_double3(0, 0, 0), 0.0, 0.0, 0.0, 0.0, cnt);
    int h = 0;
    CUDA_SAFE_CALL(cudaMemcpy(&h, cnt, sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(cnt);
    return h;
}

void LiquidSystem::get_particles(double* out) const
{
    CUDA_SAFE_CALL(cudaMemcpy(out, d_xp, n_particles * sizeof(double3), cudaMemcpyDeviceToHost));
}
void LiquidSystem::get_particle_velocities(double* out) const
{
    CUDA_SAFE_CALL(cudaMemcpy(out, d_vp, n_particles * sizeof(double3), cudaMemcpyDeviceToHost));
}
void LiquidSystem::get_particle_J(double* out) const
{
    CUDA_SAFE_CALL(cudaMemcpy(out, d_Jp, n_particles * sizeof(double), cudaMemcpyDeviceToHost));
}
void LiquidSystem::set_particle_J(const double* in)
{   // test hook: overwrite the tracked particle volume ratios (e.g. a hydrostatic initial state)
    CUDA_SAFE_CALL(cudaMemcpy(d_Jp, in, n_particles * sizeof(double), cudaMemcpyHostToDevice));
}
void LiquidSystem::get_particle_density(double* out) const
{
    CUDA_SAFE_CALL(cudaMemcpy(out, d_rho, n_particles * sizeof(double), cudaMemcpyDeviceToHost));
}

}  // namespace gipc_liquid

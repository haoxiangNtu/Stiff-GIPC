__global__ void _reduct_max_box(AABB* _leafBoxes, int number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ AABB tep[];

    // [audit lens-B fix] NO early return: the exited lanes of a partial last
    // warp fed UNDEFINED register values into the full-mask __shfl_down_sync
    // below (participation-mismatch UB), and AABB::combines() min/max-absorbs
    // whatever comes back — on this HW typically 0.0, which silently clamps the
    // scene box to include the origin (visible only for scenes whose true box
    // excludes it; the box feeds Morton quantization AND bboxDiagSize2→dHat).
    // c3087a7 swept this template in GIPC.cu but this mlbvh.cu twin was missed.
    // All lanes now stay resident; out-of-range lanes carry the NEUTRAL
    // (inverted 1e32) AABB so every shuffle reads a defined absorbing value.
    // Shared tep[] is sized blockDim/32 (calcMaxBV), so phantom-warp lane-0
    // writes stay in bounds and are never read (stage 2 reads [0, warpNum)).
    AABB temp;   // default ctor = (+1e32, -1e32) = neutral element for combines
    if(idx < number)
        temp = _leafBoxes[idx];

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
    // [audit lens-B fix] stage 2: whole warps other than warp 0 may retire (no
    // barrier and no cross-warp op remains), but EVERY lane of warp 0 must stay
    // resident for the full-mask shuffles — padding lanes read the neutral AABB
    // instead of the old `threadIdx.x >= warpNum` early return that exited
    // warp-0 lanes mid-warp (same participation-mismatch UB as stage 1).
    if(warpId != 0)
        return;
    if(warpNum > 1)
    {
        temp = (warpTid < warpNum) ? tep[warpTid] : AABB();
        xmin = temp.lower.x, ymin = temp.lower.y, zmin = temp.lower.z;
        xmax = temp.upper.x, ymax = temp.upper.y, zmax = temp.upper.z;
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp.combines(__shfl_down_sync(0xffffffff, xmin, i),
                          __shfl_down_sync(0xffffffff, ymin, i),
                          __shfl_down_sync(0xffffffff, zmin, i),
                          __shfl_down_sync(0xffffffff, xmax, i),
                          __shfl_down_sync(0xffffffff, ymax, i),
                          __shfl_down_sync(0xffffffff, zmax, i));
            if(warpTid + i < warpNum)
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
template <bool TrackMax>
static __device__ __forceinline__ void _calcLeafNodes_indirect_body(
    Node*           _nodes,
    const uint32_t* _indices,
    const int*      _active_idx,
    uint32_t*       _node_max_element,
    bool            _node_max_sorted,
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
    if constexpr(TrackMax)
        _node_max_element[l_idx] =
            _node_max_sorted ? (uint32_t)idx : _nodes[l_idx].element_idx;
}

__global__ void _calcLeafNodes_indirect(Node*           _nodes,
                                        const uint32_t* _indices,
                                        const int*      _active_idx,
                                        int             number)
{
    _calcLeafNodes_indirect_body<false>(
        _nodes, _indices, _active_idx, nullptr, false, number);
}

__global__ void _calcLeafNodes_indirect_with_max(
    Node*           _nodes,
    const uint32_t* _indices,
    const int*      _active_idx,
    uint32_t*       _node_max_element,
    bool            _node_max_sorted,
    int             number)
{
    _calcLeafNodes_indirect_body<true>(_nodes,
                                       _indices,
                                       _active_idx,
                                       _node_max_element,
                                       _node_max_sorted,
                                       number);
}

// [env-det] env-major Morton: put the prim's env id in the HIGH bits so co-located identical envs
// sort into separate contiguous blocks (env-symmetric tree), instead of the default global-index
// tie-break that interleaves equal-Morton co-located prims non-deterministically. STIFF_BVH_ENVDET.
__device__ int g_bvh_envmajor = 0;
void set_bvh_envmajor(int v){ static int last = -999; if(v == last) return; CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_bvh_envmajor, &v, sizeof(int))); last = v; }
template <bool Morton14>
static __device__ __forceinline__ void _calcMChash_body(
    uint64_t*       _MChash,
    AABB*           _bvs,
    int             number,
    const int*      prim_env,
    const int*      prim_localid,
    const double3*  env_offset,
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
    const uint64_t mc32 = morton_code(
        offset.x / SceneSize.x, offset.y / SceneSize.y, offset.z / SceneSize.z);
    uint64_t mc64;
    if constexpr(Morton14)
    {
        // Research probe: use the requested 42 Morton bits even when the
        // production env-major layout is enabled.  The previous ordering put
        // this branch after g_bvh_envmajor, making STIFF_BVH_MORTON14 a no-op
        // in every FOLD-SHIRT run.  The global primitive index is a unique
        // 22-bit tie-break.  This deliberately does not preserve env-major
        // grouping, so multi-env performance/determinism must be judged by the
        // candidate gates; it is not a production key layout.
        if(number < (1 << 22))
        {
            const uint64_t mc42 = morton_code_14(offset.x / SceneSize.x,
                                                 offset.y / SceneSize.y,
                                                 offset.z / SceneSize.z);
            mc64 = (mc42 << 22) | idx;
        }
        else
            mc64 = (mc32 << 32) | idx;
    }
    else if(g_bvh_envmajor && prim_env)
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

__global__ void _calcMChash(uint64_t*       _MChash,
                            AABB*           _bvs,
                            int             number,
                            const int*      prim_env,
                            const int*      prim_localid,
                            const double3*  env_offset,
                            const uint32_t* prim_v0)
{
    _calcMChash_body<false>(
        _MChash, _bvs, number, prim_env, prim_localid, env_offset, prim_v0);
}

__global__ void _calcMChash14(uint64_t*       _MChash,
                              AABB*           _bvs,
                              int             number,
                              const int*      prim_env,
                              const int*      prim_localid,
                              const double3*  env_offset,
                              const uint32_t* prim_v0)
{
    _calcMChash_body<true>(
        _MChash, _bvs, number, prim_env, prim_localid, env_offset, prim_v0);
}

template <bool TrackMax>
static __device__ __forceinline__ void _calcLeafNodes_body(
    Node*           _nodes,
    const uint32_t* _indices,
    uint32_t*       _node_max_element,
    bool            _node_max_sorted,
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
    _nodes[l_idx].element_idx = _indices[idx];
    if constexpr(TrackMax)
        _node_max_element[l_idx] =
            _node_max_sorted ? (uint32_t)idx : _nodes[l_idx].element_idx;
}

__global__ void _calcLeafNodes(Node*           _nodes,
                               const uint32_t* _indices,
                               int             number)
{
    _calcLeafNodes_body<false>(_nodes, _indices, nullptr, false, number);
}

__global__ void _calcLeafNodes_with_max(Node*           _nodes,
                                        const uint32_t* _indices,
                                        uint32_t*       _node_max_element,
                                        bool            _node_max_sorted,
                                        int             number)
{
    _calcLeafNodes_body<true>(
        _nodes, _indices, _node_max_element, _node_max_sorted, number);
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

template <bool TrackMax>
static __device__ __forceinline__ void _calcInternalAABB_body(
    const Node* _nodes,
    AABB*       _bvs,
    uint32_t*   flags,
    uint32_t*   _node_max_element,
    int         number)
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
        if constexpr(TrackMax)
        {
            const uint32_t lmax = _node_max_element[lidx];
            const uint32_t rmax = _node_max_element[ridx];
            _node_max_element[parent] = lmax > rmax ? lmax : rmax;
        }

        __threadfence();

        parent = _nodes[parent].parent_idx;
    }
}

__global__ void _calcInternalAABB(const Node* _nodes,
                                  AABB*       _bvs,
                                  uint32_t*   flags,
                                  int         number)
{
    _calcInternalAABB_body<false>(
        _nodes, _bvs, flags, nullptr, number);
}

__global__ void _calcInternalAABB_with_max(const Node* _nodes,
                                           AABB*       _bvs,
                                           uint32_t*   flags,
                                           uint32_t*   _node_max_element,
                                           int         number)
{
    _calcInternalAABB_body<true>(
        _nodes, _bvs, flags, _node_max_element, number);
}

// Experimental graph-safe LBVH treelet optimizer.  A binary rotation at P
// replaces one internal child C=(A,B) and P's other child S with either
// C'=(A,S), P'=(C',B), or C'=(S,B), P'=(A,C').  P still covers exactly the
// same primitives, so only C's box (and optional subtree maximum) changes.
//
// Nodes whose depths have the same value modulo three have disjoint write
// footprints: a rotation touches depths d..d+2, while the next selected
// descendant starts at d+3.  Depths are recomputed before every phase because
// earlier rotations can move a subtree by one level.  This makes each phase
// race-free and deterministic without locks or host/device communication.
__global__ void _calcInternalDepths(const Node* nodes,
                                    uint32_t*   depths,
                                    int         number)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number - 1)
        return;
    uint32_t depth  = 0;
    uint32_t parent = nodes[idx].parent_idx;
    while(parent != 0xFFFFFFFFu && depth < static_cast<uint32_t>(number))
    {
        ++depth;
        parent = nodes[parent].parent_idx;
    }
    depths[idx] = depth;
}

__device__ __forceinline__ double _bvh_half_surface_area(const AABB& box)
{
    const double x = max(0.0, box.upper.x - box.lower.x);
    const double y = max(0.0, box.upper.y - box.lower.y);
    const double z = max(0.0, box.upper.z - box.lower.z);
    return x * y + y * z + z * x;
}

// One PLOC workgroup. Representatives must remain Morton ordered. Every round
// performs the range-nearest search, mutual merge, and stable compaction. The
// helper is used both by the diagnostic all-in-one-block implementation and by
// the hierarchical implementation below.
static __device__ __forceinline__ void _buildPlocWorkgroup(
    Node*           nodes,
    AABB*           boxes,
    const uint32_t* initial_clusters,
    uint32_t*       clusters_a,
    uint32_t*       clusters_b,
    uint32_t*       nearest,
    uint32_t*       flags,
    uint32_t*       offsets,
    uint32_t*       assigned,
    uint32_t*       node_max_element,
    uint32_t*       root_output,
    int             number,
    int             leaf_base,
    int             next_internal,
    int             radius,
    int             ploc_plus_plus,
    int*            shared_count,
    int*            shared_new_count,
    int*            shared_next_internal,
    int*            shared_round_merges)
{
    for(int i = threadIdx.x; i < number; i += blockDim.x)
        clusters_a[i] = initial_clusters
                            ? initial_clusters[i]
                            : static_cast<uint32_t>(leaf_base + i);
    if(threadIdx.x == 0)
    {
        *shared_count         = number;
        *shared_new_count     = number;
        *shared_next_internal = next_internal;
        *shared_round_merges  = 0;
    }
    __syncthreads();

    uint32_t* input  = clusters_a;
    uint32_t* output = clusters_b;
    int count = number;

    // Symmetric merge distance guarantees at least one mutual-nearest pair
    // per iteration.  The number guard is a corruption fail-safe; the forced
    // adjacent pair below also guarantees forward progress in a tie/NaN
    // corner case without involving the host.
    for(int round = 0; count > 1 && round < number; ++round)
    {
        const int local_radius = min(radius, count - 1);
        for(int i = threadIdx.x; i < count; i += blockDim.x)
        {
            int best = -1;
            if(ploc_plus_plus)
            {
                // PLOC++'s equal-box corner-case rule: seed even/odd adjacent
                // pairs so identical representatives reduce logarithmically.
                best = ((i & 1) == 0 && i + 1 < count) ? i + 1 : i - 1;
                if(best < 0)
                    best = i + 1;
            }
            else
            {
                best = max(0, i - local_radius);
                if(best == i)
                    best = min(count - 1, i + 1);
            }

            double best_cost = _bvh_half_surface_area(
                merge(boxes[input[i]], boxes[input[best]]));
            const int begin = max(0, i - local_radius);
            const int end   = min(count - 1, i + local_radius);
            for(int j = begin; j <= end; ++j)
            {
                if(j == i || j == best)
                    continue;
                const double cost = _bvh_half_surface_area(
                    merge(boxes[input[i]], boxes[input[j]]));
                // Original PLOC is left-first on exact ties.  PLOC++ keeps
                // the adjacent seed on ties, which is its documented fix for
                // coincident primitives.
                if(cost < best_cost
                   || (!ploc_plus_plus && cost == best_cost && j < best))
                {
                    best      = j;
                    best_cost = cost;
                }
            }
            nearest[i] = static_cast<uint32_t>(best);
        }
        __syncthreads();

        if(threadIdx.x == 0)
        {
            int out_count   = 0;
            int merge_count = 0;
            for(int i = 0; i < count; ++i)
            {
                const int j = static_cast<int>(nearest[i]);
                const bool mutual = j >= 0 && j < count
                                    && nearest[j] == static_cast<uint32_t>(i);
                const bool merge_left  = mutual && i < j;
                const bool merge_right = mutual && i > j;
                flags[i]   = merge_right ? 0u : 1u;
                offsets[i] = static_cast<uint32_t>(out_count);
                assigned[i] = 0xFFFFFFFFu;
                if(!merge_right)
                    ++out_count;
                if(merge_left)
                {
                    assigned[i] = static_cast<uint32_t>(
                        *shared_next_internal - merge_count);
                    ++merge_count;
                }
            }

            if(merge_count == 0)
            {
                // Defensive deterministic progress for non-finite AABBs or a
                // future distance change that breaks the mutual-pair lemma.
                nearest[0] = 1u;
                nearest[1] = 0u;
                out_count = 0;
                for(int i = 0; i < count; ++i)
                {
                    if(i >= 2)
                        nearest[i] = static_cast<uint32_t>(i);
                    flags[i]   = i == 1 ? 0u : 1u;
                    offsets[i] = static_cast<uint32_t>(out_count);
                    assigned[i] = i == 0
                                      ? static_cast<uint32_t>(
                                            *shared_next_internal)
                                      : 0xFFFFFFFFu;
                    if(i != 1)
                        ++out_count;
                }
                merge_count = 1;
            }
            *shared_new_count     = out_count;
            *shared_round_merges  = merge_count;
        }
        __syncthreads();

        for(int i = threadIdx.x; i < count; i += blockDim.x)
        {
            if(flags[i] == 0u)
                continue;
            const uint32_t out = offsets[i];
            const int j = static_cast<int>(nearest[i]);
            const bool merge_left = j > i && j < count
                                    && nearest[j] == static_cast<uint32_t>(i);
            if(merge_left)
            {
                const uint32_t node = assigned[i];
                const uint32_t left = input[i];
                const uint32_t right = input[j];
                nodes[node].parent_idx  = 0xFFFFFFFFu;
                nodes[node].left_idx    = left;
                nodes[node].right_idx   = right;
                nodes[node].element_idx = 0xFFFFFFFFu;
                nodes[left].parent_idx  = node;
                nodes[right].parent_idx = node;
                boxes[node]             = merge(boxes[left], boxes[right]);
                if(node_max_element)
                {
                    const uint32_t a = node_max_element[left];
                    const uint32_t b = node_max_element[right];
                    node_max_element[node] = a > b ? a : b;
                }
                output[out] = node;
            }
            else
                output[out] = input[i];
        }
        __syncthreads();

        if(threadIdx.x == 0)
        {
            *shared_next_internal -= *shared_round_merges;
            *shared_count = *shared_new_count;
        }
        __syncthreads();
        count = *shared_count;
        uint32_t* swap = input;
        input  = output;
        output = swap;
    }

    if(threadIdx.x == 0 && root_output)
        *root_output = input[0];
}

// Capture-safe diagnostic implementation that starts the entire tree in one
// workgroup. It is intentionally retained as an oracle/performance control;
// large FOLD trees demonstrate why PLOC++ only recommends this path for upper
// levels.
__global__ void _buildPlocSingleWorkgroup(Node*           nodes,
                                           AABB*           boxes,
                                           uint32_t*       clusters_a,
                                           uint32_t*       clusters_b,
                                           uint32_t*       nearest,
                                           uint32_t*       flags,
                                           uint32_t*       offsets,
                                           uint32_t*       assigned,
                                           uint32_t*       node_max_element,
                                           int             number,
                                           int             radius,
                                           int             ploc_plus_plus)
{
    if(number <= 1)
        return;
    __shared__ int shared_count;
    __shared__ int shared_new_count;
    __shared__ int shared_next_internal;
    __shared__ int shared_round_merges;
    _buildPlocWorkgroup(nodes,
                        boxes,
                        nullptr,
                        clusters_a,
                        clusters_b,
                        nearest,
                        flags,
                        offsets,
                        assigned,
                        node_max_element,
                        nullptr,
                        number,
                        number - 1,
                        number - 2,
                        radius,
                        ploc_plus_plus,
                        &shared_count,
                        &shared_new_count,
                        &shared_next_internal,
                        &shared_round_merges);
}

// Hierarchical PLOC++ lower level. Morton-contiguous chunks are independent,
// so one workgroup can agglomerate each chunk without a global barrier. A FOLD
// chunk
// needs only 256 representatives, so both representative IDs and AABBs fit in
// shared memory. Nearest-neighbour rounds then touch global memory only when a
// new internal node is committed. Internal ID ranges are disjoint and reserve
// [0, chunk_count-2] for the upper tree.
__global__ void _buildPlocChunksShared(Node*       nodes,
                                       AABB*       boxes,
                                       uint32_t*   roots,
                                       uint32_t*   node_max_element,
                                       int         number,
                                       int         chunk_size,
                                       int         radius)
{
    constexpr int kMaxChunk = 256;
    const int chunk = static_cast<int>(blockIdx.x);
    const int start = chunk * chunk_size;
    if(start >= number)
        return;
    const int initial_count = min(chunk_size, number - start);
    if(initial_count == 1)
    {
        if(threadIdx.x == 0)
            roots[chunk] = static_cast<uint32_t>(number - 1 + start);
        return;
    }

    __shared__ uint32_t ids_a[kMaxChunk];
    __shared__ uint32_t ids_b[kMaxChunk];
    __shared__ AABB     boxes_a[kMaxChunk];
    __shared__ AABB     boxes_b[kMaxChunk];
    __shared__ uint32_t nearest[kMaxChunk];
    __shared__ uint32_t offsets[kMaxChunk];
    __shared__ uint32_t assigned[kMaxChunk];
    __shared__ int      shared_count;
    __shared__ int      shared_new_count;
    __shared__ int      shared_next_internal;
    __shared__ int      shared_round_merges;

    const int chunk_count = (number + chunk_size - 1) / chunk_size;
    const int internal_first = chunk_count - 1 + start - chunk;
    for(int i = threadIdx.x; i < initial_count; i += blockDim.x)
    {
        const uint32_t leaf = static_cast<uint32_t>(number - 1 + start + i);
        ids_a[i]   = leaf;
        boxes_a[i] = boxes[leaf];
    }
    if(threadIdx.x == 0)
    {
        shared_count         = initial_count;
        shared_new_count     = initial_count;
        shared_next_internal = internal_first + initial_count - 2;
        shared_round_merges  = 0;
    }
    __syncthreads();

    uint32_t* input_ids  = ids_a;
    uint32_t* output_ids = ids_b;
    AABB* input_boxes    = boxes_a;
    AABB* output_boxes   = boxes_b;
    int count = initial_count;
    for(int round = 0; count > 1 && round < initial_count; ++round)
    {
        const int local_radius = min(radius, count - 1);
        for(int i = threadIdx.x; i < count; i += blockDim.x)
        {
            int best = ((i & 1) == 0 && i + 1 < count) ? i + 1 : i - 1;
            if(best < 0)
                best = i + 1;
            double best_cost = _bvh_half_surface_area(
                merge(input_boxes[i], input_boxes[best]));
            const int begin = max(0, i - local_radius);
            const int end   = min(count - 1, i + local_radius);
            for(int j = begin; j <= end; ++j)
            {
                if(j == i || j == best)
                    continue;
                const double cost = _bvh_half_surface_area(
                    merge(input_boxes[i], input_boxes[j]));
                if(cost < best_cost)
                {
                    best      = j;
                    best_cost = cost;
                }
            }
            nearest[i] = static_cast<uint32_t>(best);
        }
        __syncthreads();

        if(threadIdx.x == 0)
        {
            int out_count   = 0;
            int merge_count = 0;
            for(int i = 0; i < count; ++i)
            {
                const int j = static_cast<int>(nearest[i]);
                const bool mutual = j >= 0 && j < count
                                    && nearest[j] == static_cast<uint32_t>(i);
                const bool merge_left  = mutual && i < j;
                const bool merge_right = mutual && i > j;
                offsets[i]  = static_cast<uint32_t>(out_count);
                assigned[i] = 0xFFFFFFFFu;
                if(!merge_right)
                    ++out_count;
                if(merge_left)
                {
                    assigned[i] = static_cast<uint32_t>(
                        shared_next_internal - merge_count);
                    ++merge_count;
                }
            }
            if(merge_count == 0)
            {
                nearest[0] = 1u;
                nearest[1] = 0u;
                out_count = 0;
                for(int i = 0; i < count; ++i)
                {
                    if(i >= 2)
                        nearest[i] = static_cast<uint32_t>(i);
                    offsets[i] = static_cast<uint32_t>(out_count);
                    assigned[i] = i == 0
                                      ? static_cast<uint32_t>(
                                            shared_next_internal)
                                      : 0xFFFFFFFFu;
                    if(i != 1)
                        ++out_count;
                }
                merge_count = 1;
            }
            shared_new_count    = out_count;
            shared_round_merges = merge_count;
        }
        __syncthreads();

        for(int i = threadIdx.x; i < count; i += blockDim.x)
        {
            const int j = static_cast<int>(nearest[i]);
            const bool mutual = j >= 0 && j < count
                                && nearest[j] == static_cast<uint32_t>(i);
            if(mutual && i > j)
                continue;
            const uint32_t out = offsets[i];
            if(mutual && i < j)
            {
                const uint32_t node  = assigned[i];
                const uint32_t left  = input_ids[i];
                const uint32_t right = input_ids[j];
                const AABB merged_box = merge(input_boxes[i], input_boxes[j]);
                nodes[node].parent_idx  = 0xFFFFFFFFu;
                nodes[node].left_idx    = left;
                nodes[node].right_idx   = right;
                nodes[node].element_idx = 0xFFFFFFFFu;
                nodes[left].parent_idx  = node;
                nodes[right].parent_idx = node;
                boxes[node]             = merged_box;
                if(node_max_element)
                {
                    const uint32_t a = node_max_element[left];
                    const uint32_t b = node_max_element[right];
                    node_max_element[node] = a > b ? a : b;
                }
                output_ids[out]   = node;
                output_boxes[out] = merged_box;
            }
            else
            {
                output_ids[out]   = input_ids[i];
                output_boxes[out] = input_boxes[i];
            }
        }
        __syncthreads();

        if(threadIdx.x == 0)
        {
            shared_next_internal -= shared_round_merges;
            shared_count = shared_new_count;
        }
        __syncthreads();
        count = shared_count;
        uint32_t* ids_swap = input_ids;
        input_ids  = output_ids;
        output_ids = ids_swap;
        AABB* boxes_swap = input_boxes;
        input_boxes  = output_boxes;
        output_boxes = boxes_swap;
    }
    if(threadIdx.x == 0)
        roots[chunk] = input_ids[0];
}

// PLOC++ upper level: after the lower kernel completes, only chunk_count
// representatives remain. Their tree consumes the reserved leading internal
// IDs, ending at root node zero. This separate launch is stream ordered and is
// fully CUDA Graph capture-safe.
__global__ void _buildPlocUpper(Node*           nodes,
                                AABB*           boxes,
                                const uint32_t* roots,
                                uint32_t*       clusters_a,
                                uint32_t*       clusters_b,
                                uint32_t*       nearest,
                                uint32_t*       flags,
                                uint32_t*       offsets,
                                uint32_t*       assigned,
                                uint32_t*       node_max_element,
                                int             chunk_count,
                                int             radius,
                                int             ploc_plus_plus)
{
    if(chunk_count <= 1)
        return;
    __shared__ int shared_count;
    __shared__ int shared_new_count;
    __shared__ int shared_next_internal;
    __shared__ int shared_round_merges;
    _buildPlocWorkgroup(nodes,
                        boxes,
                        roots,
                        clusters_a,
                        clusters_b,
                        nearest,
                        flags,
                        offsets,
                        assigned,
                        node_max_element,
                        nullptr,
                        chunk_count,
                        0,
                        chunk_count - 2,
                        radius,
                        ploc_plus_plus,
                        &shared_count,
                        &shared_new_count,
                        &shared_next_internal,
                        &shared_round_merges);
}

__global__ void _rotateSahTreelets(Node*           nodes,
                                   AABB*           boxes,
                                   const uint32_t* depths,
                                   uint32_t*       node_max_element,
                                   int             number,
                                   int             phase)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= number - 1 || static_cast<int>(depths[p] % 3u) != phase)
        return;

    const uint32_t children[2] = {nodes[p].left_idx, nodes[p].right_idx};
    double         best_gain   = 0.0;
    int            best_side   = -1;
    int            best_inside = -1;

#pragma unroll
    for(int side = 0; side < 2; ++side)
    {
        const uint32_t c = children[side];
        const uint32_t s = children[1 - side];
        if(c >= static_cast<uint32_t>(number - 1)
           || s == 0xFFFFFFFFu)
            continue;
        const uint32_t a = nodes[c].left_idx;
        const uint32_t b = nodes[c].right_idx;
        if(a == 0xFFFFFFFFu || b == 0xFFFFFFFFu)
            continue;

        const double old_cost = _bvh_half_surface_area(boxes[c]);
        const double gain_a =
            old_cost - _bvh_half_surface_area(merge(boxes[a], boxes[s]));
        const double gain_b =
            old_cost - _bvh_half_surface_area(merge(boxes[s], boxes[b]));
        if(gain_a > best_gain)
        {
            best_gain   = gain_a;
            best_side   = side;
            best_inside = 0;
        }
        if(gain_b > best_gain)
        {
            best_gain   = gain_b;
            best_side   = side;
            best_inside = 1;
        }
    }

    const double tolerance =
        1.0e-12 * (_bvh_half_surface_area(boxes[p]) + 1.0);
    if(best_side < 0 || !(best_gain > tolerance))
        return;

    const uint32_t c = children[best_side];
    const uint32_t s = children[1 - best_side];
    const uint32_t a = nodes[c].left_idx;
    const uint32_t b = nodes[c].right_idx;
    const uint32_t inside  = best_inside == 0 ? a : b;
    const uint32_t outside = best_inside == 0 ? b : a;

    // Keep C in its original P slot.  Keep the selected original grandchild
    // in its original C slot, insert S in the other, and promote `outside`.
    if(best_side == 0)
        nodes[p].right_idx = outside;
    else
        nodes[p].left_idx = outside;
    if(best_inside == 0)
        nodes[c].right_idx = s;
    else
        nodes[c].left_idx = s;

    nodes[outside].parent_idx = static_cast<uint32_t>(p);
    nodes[s].parent_idx       = c;
    boxes[c]                  = merge(boxes[inside], boxes[s]);
    if(node_max_element)
    {
        const uint32_t lhs = node_max_element[inside];
        const uint32_t rhs = node_max_element[s];
        node_max_element[c] = lhs > rhs ? lhs : rhs;
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
void set_bvh_envpart(int v){ static int last = -999; if(v == last) return; CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_bvh_envpart, &v, sizeof(int))); last = v; }

// [perenv-par] per-vertex env id (= d_point_to_group); cross-env self-collision pairs skipped at
// emission when set. -1 = ungrouped/static (never skipped). Null = gate off (byte-for-byte legacy).
__device__ const int* g_self_p2g = nullptr;
__device__ unsigned long long g_xskip = 0;   // [debug] count of cross-env pairs skipped
void set_self_p2g(const int* p){
    if(getenv("STIFF_XSKIP_DBG")) fprintf(stderr, "[self_p2g] set to %p\n", (const void*)p);
    static const int* last_p2g = (const int*)-1;
    if(p == last_p2g)
        return;
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_self_p2g, &p, sizeof(const int*)));
    last_p2g = p;
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
    static const int* last_veid = (const int*)-1;
    if(d_vertex_env_id == last_veid)
        return;
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_vertex_env_id, &d_vertex_env_id, sizeof(const int*)));
    last_veid = d_vertex_env_id;
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

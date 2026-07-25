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


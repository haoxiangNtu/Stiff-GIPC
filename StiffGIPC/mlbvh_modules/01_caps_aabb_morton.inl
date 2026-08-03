// [B3 trial-defer] monotone overflow counter: emission bumps it on the trash-
// slot branch (cold path, zero hot cost). Consumers compare against the last
// value they saw — no resets, no ToSymbol churn. Cross-TU via RDC extern.
__device__ uint32_t g_pair_overflow_count = 0u;
__device__ __forceinline__ uint32_t _emit_slot(uint32_t* cnt, int cap)
{
    uint32_t i = atomicAdd(cnt, 1u);
    if(i >= (uint32_t)cap)
    {
        atomicAdd(&g_pair_overflow_count, 1u);
        return (uint32_t)cap;   // overflow -> trash slot [cap]
    }
    return i;
}
void set_emit_caps(int dcd_cap, int ccd_cap)
{
    // [audit v0.8.5.1] centralize the DCD<=CCD cap invariant at the single
    // publish point. The detect kernels dual-write the CCD mirror at the
    // DCD-clamped slot index, so dcd_cap > ccd_cap means the mirror write can
    // run past its allocation (the exact OOB class bdd0776 closed at the two
    // known grow sites — this catches any FUTURE third grow site loudly).
    if(dcd_cap > ccd_cap)
        fprintf(stderr,
                "[set_emit_caps][INVARIANT VIOLATION] dcd_cap=%d > ccd_cap=%d: "
                "the CCD mirror buffer can overflow. A pair-buffer grow site "
                "raised the DCD cap without growing the CCD mirror.\n",
                dcd_cap, ccd_cap);
    // [B3 tosymbol-cache] caps change only on pair-buffer growth
    static int last_dcd = -1, last_ccd = -1;
    if(dcd_cap == last_dcd && ccd_cap == last_ccd)
        return;
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_dcd_cp_cap, &dcd_cap, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_ccd_cp_cap, &ccd_cap, sizeof(int)));
    last_dcd = dcd_cap; last_ccd = ccd_cap;
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

// Validation candidate: 14 bits/axis (42-bit Morton key).  The production
// LBVH above quantizes each axis to 10 bits.  This deliberately simple loop is
// off the default path and tests whether FOLD-SHIRT traversal is tree-quality
// limited before investing in PLOC/treelet restructuring.
__device__ __host__ inline std::uint64_t morton_code_14(double x,
                                                        double y,
                                                        double z) noexcept
{
    constexpr double resolution = 16384.0;
    x = std::min(std::max(x * resolution, 0.0), resolution - 1.0);
    y = std::min(std::max(y * resolution, 0.0), resolution - 1.0);
    z = std::min(std::max(z * resolution, 0.0), resolution - 1.0);
    const std::uint32_t xi = static_cast<std::uint32_t>(x);
    const std::uint32_t yi = static_cast<std::uint32_t>(y);
    const std::uint32_t zi = static_cast<std::uint32_t>(z);
    std::uint64_t code = 0;
#pragma unroll
    for(int bit = 0; bit < 14; ++bit)
    {
        code |= static_cast<std::uint64_t>((xi >> bit) & 1u) << (3 * bit + 2);
        code |= static_cast<std::uint64_t>((yi >> bit) & 1u) << (3 * bit + 1);
        code |= static_cast<std::uint64_t>((zi >> bit) & 1u) << (3 * bit);
    }
    return code;
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

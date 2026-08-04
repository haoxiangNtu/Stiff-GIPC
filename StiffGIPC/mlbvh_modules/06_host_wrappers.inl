#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{
// Validation-only upper bound for tree quality.  This deliberately uses a
// host binned-SAH builder, then uploads the topology before the unchanged GPU
// traversal.  It is NOT a production path (it synchronizes and cannot be
// captured); its purpose is to answer the cheaper question before a PLOC or
// treelet implementation: can a materially better binary tree reduce enough
// node visits on the exact same IPC queries to repay a GPU builder?
constexpr int kSahOracleBuckets = 16;

struct SahOracleItem
{
    AABB     box;
    uint32_t element = 0;
};

struct SahOracleBucket
{
    AABB box;
    int  count = 0;
};

static bool bvh_sah_oracle_enabled()
{
    static const bool enabled = []
    {
        const char* value = std::getenv("STIFF_BVH_SAH_ORACLE");
        return value && std::atoi(value) > 0;
    }();
    return enabled;
}

static double sah_oracle_area(const AABB& box)
{
    const double x = std::max(0.0, box.upper.x - box.lower.x);
    const double y = std::max(0.0, box.upper.y - box.lower.y);
    const double z = std::max(0.0, box.upper.z - box.lower.z);
    return 2.0 * (x * y + x * z + y * z);
}

static double sah_oracle_center_axis(const SahOracleItem& item, int axis)
{
    if(axis == 0)
        return (item.box.lower.x + item.box.upper.x) * 0.5;
    if(axis == 1)
        return (item.box.lower.y + item.box.upper.y) * 0.5;
    return (item.box.lower.z + item.box.upper.z) * 0.5;
}

static double sah_oracle_lower_axis(const AABB& box, int axis)
{
    return axis == 0 ? box.lower.x : axis == 1 ? box.lower.y : box.lower.z;
}

static double sah_oracle_upper_axis(const AABB& box, int axis)
{
    return axis == 0 ? box.upper.x : axis == 1 ? box.upper.y : box.upper.z;
}

static int sah_oracle_bucket(double center, double lo, double hi)
{
    if(!(hi > lo))
        return 0;
    int bucket = static_cast<int>(
        (center - lo) * static_cast<double>(kSahOracleBuckets) / (hi - lo));
    return std::max(0, std::min(kSahOracleBuckets - 1, bucket));
}

class SahOracleBuilder
{
  public:
    SahOracleBuilder(std::vector<SahOracleItem> input, int count)
        : items(std::move(input))
        , nodes(static_cast<size_t>(2 * count - 1))
        , boxes(static_cast<size_t>(2 * count - 1))
        , maxima(static_cast<size_t>(2 * count - 1), 0)
        , leaf_base(count - 1)
    {
        for(Node& node : nodes)
        {
            node.parent_idx = 0xFFFFFFFFu;
            node.left_idx   = 0xFFFFFFFFu;
            node.right_idx  = 0xFFFFFFFFu;
            node.element_idx = 0xFFFFFFFFu;
        }
    }

    void build()
    {
        const uint32_t root = build_node(0, static_cast<int>(items.size()),
                                         0xFFFFFFFFu, 0);
        if(root != 0 || internal_next != leaf_base || leaf_next != (int)items.size())
            throw std::runtime_error("SAH oracle produced an invalid node layout");
    }

    std::vector<Node>     nodes;
    std::vector<AABB>     boxes;
    std::vector<uint32_t> maxima;
    int                   max_depth = 0;
    double                internal_area_sum = 0.0;

  private:
    AABB range_box(int begin, int end) const
    {
        AABB box;
        for(int i = begin; i < end; ++i)
            box.combines(items[i].box);
        return box;
    }

    int partition(int begin, int end)
    {
        AABB centroid_box;
        for(int i = begin; i < end; ++i)
        {
            const double x = sah_oracle_center_axis(items[i], 0);
            const double y = sah_oracle_center_axis(items[i], 1);
            const double z = sah_oracle_center_axis(items[i], 2);
            centroid_box.combines(x, y, z);
        }

        double best_cost = std::numeric_limits<double>::infinity();
        int    best_axis = -1;
        int    best_split = -1;
        for(int axis = 0; axis < 3; ++axis)
        {
            const double lo = sah_oracle_lower_axis(centroid_box, axis);
            const double hi = sah_oracle_upper_axis(centroid_box, axis);
            if(!(hi > lo))
                continue;

            SahOracleBucket buckets[kSahOracleBuckets];
            for(int i = begin; i < end; ++i)
            {
                const int bucket = sah_oracle_bucket(
                    sah_oracle_center_axis(items[i], axis), lo, hi);
                ++buckets[bucket].count;
                buckets[bucket].box.combines(items[i].box);
            }

            AABB left_box[kSahOracleBuckets - 1];
            AABB right_box[kSahOracleBuckets - 1];
            int  left_count[kSahOracleBuckets - 1] = {};
            int  right_count[kSahOracleBuckets - 1] = {};
            AABB prefix;
            AABB suffix;
            int  prefix_count = 0;
            int  suffix_count = 0;
            for(int i = 0; i < kSahOracleBuckets - 1; ++i)
            {
                prefix_count += buckets[i].count;
                if(buckets[i].count)
                    prefix.combines(buckets[i].box);
                left_count[i] = prefix_count;
                left_box[i]   = prefix;

                const int r = kSahOracleBuckets - 1 - i;
                suffix_count += buckets[r].count;
                if(buckets[r].count)
                    suffix.combines(buckets[r].box);
                right_count[r - 1] = suffix_count;
                right_box[r - 1]   = suffix;
            }

            for(int split = 0; split < kSahOracleBuckets - 1; ++split)
            {
                if(left_count[split] == 0 || right_count[split] == 0)
                    continue;
                const double cost =
                    sah_oracle_area(left_box[split]) * left_count[split]
                    + sah_oracle_area(right_box[split]) * right_count[split];
                if(cost < best_cost)
                {
                    best_cost  = cost;
                    best_axis  = axis;
                    best_split = split;
                }
            }
        }

        if(best_axis >= 0)
        {
            const double lo = sah_oracle_lower_axis(centroid_box, best_axis);
            const double hi = sah_oracle_upper_axis(centroid_box, best_axis);
            auto middle = std::stable_partition(
                items.begin() + begin,
                items.begin() + end,
                [=](const SahOracleItem& item)
                {
                    return sah_oracle_bucket(
                               sah_oracle_center_axis(item, best_axis), lo, hi)
                           <= best_split;
                });
            const int split = static_cast<int>(middle - items.begin());
            if(split > begin && split < end)
                return split;
        }

        int fallback_axis = 0;
        double longest = sah_oracle_upper_axis(centroid_box, 0)
                         - sah_oracle_lower_axis(centroid_box, 0);
        for(int axis = 1; axis < 3; ++axis)
        {
            const double extent = sah_oracle_upper_axis(centroid_box, axis)
                                  - sah_oracle_lower_axis(centroid_box, axis);
            if(extent > longest)
            {
                longest = extent;
                fallback_axis = axis;
            }
        }
        std::stable_sort(items.begin() + begin,
                         items.begin() + end,
                         [=](const SahOracleItem& lhs, const SahOracleItem& rhs)
                         {
                             const double a = sah_oracle_center_axis(lhs, fallback_axis);
                             const double b = sah_oracle_center_axis(rhs, fallback_axis);
                             return a == b ? lhs.element < rhs.element : a < b;
                         });
        return begin + (end - begin) / 2;
    }

    uint32_t build_node(int begin, int end, uint32_t parent, int depth)
    {
        max_depth = std::max(max_depth, depth);
        if(end - begin == 1)
        {
            const uint32_t node_idx = static_cast<uint32_t>(leaf_base + leaf_next++);
            nodes[node_idx].parent_idx  = parent;
            nodes[node_idx].element_idx = items[begin].element;
            boxes[node_idx]              = items[begin].box;
            maxima[node_idx]             = items[begin].element;
            return node_idx;
        }

        const uint32_t node_idx = static_cast<uint32_t>(internal_next++);
        const int split = partition(begin, end);
        const uint32_t left  = build_node(begin, split, node_idx, depth + 1);
        const uint32_t right = build_node(split, end, node_idx, depth + 1);
        nodes[node_idx].parent_idx = parent;
        nodes[node_idx].left_idx   = left;
        nodes[node_idx].right_idx  = right;
        boxes[node_idx] = boxes[left];
        boxes[node_idx].combines(boxes[right]);
        maxima[node_idx] = std::max(maxima[left], maxima[right]);
        internal_area_sum += sah_oracle_area(boxes[node_idx]);
        return node_idx;
    }

    std::vector<SahOracleItem> items;
    int internal_next = 0;
    int leaf_next = 0;
    int leaf_base = 0;
};

static bool build_sah_oracle(lbvh&          tree,
                             int            count,
                             const int*     active_idx,
                             cudaStream_t   stream,
                             uint32_t*      node_max_element,
                             const char*    label)
{
    if(!bvh_sah_oracle_enabled() || count < 2)
        return false;

    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    CUDA_SAFE_CALL(cudaStreamIsCapturing(stream, &capture_status));
    if(capture_status != cudaStreamCaptureStatusNone)
        throw std::runtime_error(
            "STIFF_BVH_SAH_ORACLE is a host-synchronized validation path and "
            "cannot run inside CUDA stream capture");

    const auto start = std::chrono::steady_clock::now();
    // Some legacy full-world leaf launchers still use PTDS even when their
    // caller supplies an auxiliary stream.  The oracle is intentionally
    // synchronous, so make that cross-stream boundary explicit before reading
    // the just-produced leaves.
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    std::vector<AABB> leaf_boxes(static_cast<size_t>(count));
    std::vector<int>  original_indices(static_cast<size_t>(count));
    CUDA_SAFE_CALL(cudaMemcpyAsync(leaf_boxes.data(),
                                   tree._bvs + count - 1,
                                   static_cast<size_t>(count) * sizeof(AABB),
                                   cudaMemcpyDeviceToHost,
                                   stream));
    if(active_idx)
        CUDA_SAFE_CALL(cudaMemcpyAsync(original_indices.data(),
                                       active_idx,
                                       static_cast<size_t>(count) * sizeof(int),
                                       cudaMemcpyDeviceToHost,
                                       stream));
    CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
    if(!active_idx)
        std::iota(original_indices.begin(), original_indices.end(), 0);

    std::vector<SahOracleItem> items(static_cast<size_t>(count));
    for(int i = 0; i < count; ++i)
    {
        items[i].box     = leaf_boxes[i];
        items[i].element = static_cast<uint32_t>(original_indices[i]);
    }
    SahOracleBuilder builder(std::move(items), count);
    builder.build();

    CUDA_SAFE_CALL(cudaMemcpyAsync(tree._nodes,
                                   builder.nodes.data(),
                                   builder.nodes.size() * sizeof(Node),
                                   cudaMemcpyHostToDevice,
                                   stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(tree._bvs,
                                   builder.boxes.data(),
                                   builder.boxes.size() * sizeof(AABB),
                                   cudaMemcpyHostToDevice,
                                   stream));
    if(node_max_element)
        CUDA_SAFE_CALL(cudaMemcpyAsync(node_max_element,
                                       builder.maxima.data(),
                                       builder.maxima.size() * sizeof(uint32_t),
                                       cudaMemcpyHostToDevice,
                                       stream));
    CUDA_SAFE_CALL(cudaStreamSynchronize(stream));

    const double elapsed_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - start)
                                  .count();
    static int reports = 0;
    if(reports++ < 12)
    {
        const double root_area = sah_oracle_area(builder.boxes[0]);
        std::fprintf(stderr,
                     "[bvh-sah-oracle] %s leaves=%d depth=%d "
                     "normalized_internal_area=%.6f host_build_upload_ms=%.3f\n",
                     label,
                     count,
                     builder.max_depth,
                     root_area > 0.0 ? builder.internal_area_sum / root_area : 0.0,
                     elapsed_ms);
    }
    return true;
}
}  // namespace

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
                            cudaStream_t    stream = 0,
                            uint32_t*       node_max_element = nullptr,
                            bool            node_max_sorted = false)
{
    if(n_active < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (n_active + threadNum - 1) / threadNum;
    if(node_max_element)
        _calcLeafNodes_indirect_with_max<<<blockNum, threadNum, 0, stream>>>(
            _nodes,
            _indices,
            _active_idx,
            node_max_element,
            node_max_sorted,
            n_active);
    else
        _calcLeafNodes_indirect<<<blockNum, threadNum, 0, stream>>>(
            _nodes, _indices, _active_idx, n_active);
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
    static const bool use_morton14 = []
    {
        const char* value = getenv("STIFF_BVH_MORTON14");
        return value && atoi(value) > 0;
    }();
    if(use_morton14)
        _calcMChash14<<<blockNum, threadNum, 0, stream>>>(
            _MChash,
            _bvs,
            number,
            prim_env,
            prim_localid,
            env_offset,
            prim_v0);
    else
        _calcMChash<<<blockNum, threadNum, 0, stream>>>(_MChash,
                                                       _bvs,
                                                       number,
                                                       prim_env,
                                                       prim_localid,
                                                       env_offset,
                                                       prim_v0);
}

void calcLeafNodes(Node*           _nodes,
                   const uint32_t* _indices,
                   int             number,
                   uint32_t*       node_max_element = nullptr,
                   bool            node_max_sorted = false)
{
    int numbers = number;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    if(node_max_element)
        _calcLeafNodes_with_max<<<blockNum, threadNum>>>(
            _nodes, _indices, node_max_element, node_max_sorted, number);
    else
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

void calcInternalAABB(const Node* _nodes,
                      AABB*       _bvs,
                      uint32_t*   flags,
                      int         number,
                      cudaStream_t stream = 0,
                      uint32_t*   node_max_element = nullptr)
{
    int numbers = number;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    //uint32_t* flags;
    //CUDA_SAFE_CALL(cudaMalloc((void**)&flags, (numbers-1) * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemsetAsync(flags, 0xFFFFFFFF, sizeof(uint32_t) * (numbers - 1), stream));
    if(node_max_element)
        _calcInternalAABB_with_max<<<blockNum, threadNum, 0, stream>>>(
            _nodes, _bvs, flags, node_max_element, numbers);
    else
        _calcInternalAABB<<<blockNum, threadNum, 0, stream>>>(
            _nodes, _bvs, flags, numbers);
    //CUDA_SAFE_CALL(cudaFree(flags));
}

static int bvh_sah_rotation_cycles()
{
    static const int cycles = []
    {
        const char* value = getenv("STIFF_BVH_SAH_ROTATIONS");
        if(!value)
            return 0;
        const int parsed = atoi(value);
        return parsed < 0 ? 0 : (parsed > 8 ? 8 : parsed);
    }();
    return cycles;
}

enum BvhSahRotationFamily
{
    kSahFaceDcd = 1,
    kSahEdgeDcd = 2,
    kSahFaceCcd = 4,
    kSahEdgeCcd = 8,
};

static int bvh_refit_interval()
{
    static const int interval = []
    {
        const char* value = getenv("STIFF_BVH_REFIT_INTERVAL");
        const int parsed = value ? atoi(value) : 1;
        return std::max(1, std::min(1000000, parsed));
    }();
    return interval;
}

static int bvh_refit_mask()
{
    static const int mask = []
    {
        const char* value = getenv("STIFF_BVH_REFIT_MASK");
        return value ? (static_cast<int>(strtol(value, nullptr, 0)) & 0xF)
                     : 0xF;
    }();
    return mask;
}

static bool shouldRefitTopology(lbvh&      tree,
                                int        number,
                                int        family,
                                const int* active_identity,
                                cudaStream_t stream)
{
    const int interval = bvh_refit_interval();
    if(interval <= 1 || !(bvh_refit_mask() & family))
    {
        tree.m_refit_topology_ready = false;
        return false;
    }
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    CUDA_SAFE_CALL(cudaStreamIsCapturing(stream, &capture_status));
    if(capture_status == cudaStreamCaptureStatusActive)
    {
        unsigned long long capture_id = 0;
        CUDA_SAFE_CALL(cudaStreamGetCaptureInfo(
            stream, &capture_status, &capture_id));
        // A graph replay repeats the captured build/refit pattern forever;
        // if capture happened to contain only refits, quality would degrade
        // without bound across frames.  Force the first construction of every
        // distinct capture to be a complete build, then refit within the body.
        if(tree.m_refit_capture_id != capture_id)
        {
            tree.m_refit_capture_id = capture_id;
            tree.m_refit_topology_ready = false;
            tree.m_refit_since_rebuild = 0;
        }
    }
    const bool same_storage = tree.m_refit_nodes_identity == tree._nodes
                              && tree.m_refit_active_identity
                                     == active_identity
                              && tree.m_refit_number == number;
    if(!same_storage)
    {
        tree.m_refit_topology_ready = false;
        tree.m_refit_since_rebuild = 0;
        tree.m_refit_nodes_identity = tree._nodes;
        tree.m_refit_active_identity = active_identity;
        tree.m_refit_number = number;
    }
    if(tree.m_refit_topology_ready
       && tree.m_refit_since_rebuild + 1 < interval)
    {
        ++tree.m_refit_since_rebuild;
        ++tree.m_refit_reuses;
        return true;
    }
    tree.m_refit_topology_ready = true;
    tree.m_refit_since_rebuild = 0;
    ++tree.m_refit_rebuilds;
    return false;
}

template <class element_type, bool Swept>
static void refitTopology(lbvh&               tree,
                          const element_type* elements,
                          const double3*      move_dir,
                          double              alpha,
                          int                 number,
                          int                 type,
                          cudaStream_t        stream,
                          const double*       alpha_dev,
                          uint32_t*           node_max_element = nullptr)
{
    constexpr int threads = 256;
    _refitLeafBvs<element_type, Swept><<<
        (number + threads - 1) / threads, threads, 0, stream>>>(
        tree._vertexes,
        move_dir,
        alpha,
        elements,
        tree._nodes,
        tree._bvs,
        number,
        type,
        tree._bodyId,
        tree._collision_skip_matrix,
        tree._collision_body_count,
        alpha_dev);
    calcInternalAABB(tree._nodes,
                     tree._bvs,
                     tree._flags,
                     number,
                     stream,
                     node_max_element);
}

static int bvh_ploc_mode()
{
    static const int mode = []
    {
        const char* value = getenv("STIFF_BVH_PLOC");
        if(!value)
            return 0;
        const int parsed = atoi(value);
        return parsed <= 0 ? 0 : std::min(parsed, 3);
    }();
    return mode;
}

static int bvh_wide8_mode()
{
    static const int mode = []
    {
        const char* value = getenv("STIFF_BVH_WIDE8");
        const int parsed = value ? atoi(value) : 0;
        return std::max(0, std::min(3, parsed));
    }();
    return mode;
}

// Bit 0 = face trees, bit 1 = edge trees.  Keeping the families separable is
// essential for attribution: a VF-only cache must not pay the measured
// body-major regression in the uncached EE tree.
static int bvh_body_major_mask()
{
    static const int mask = []
    {
        const char* value = getenv("STIFF_BVH_BODY_MAJOR");
        if(!value || atoi(value) <= 0)
            return 0;
        const char* explicit_mask = getenv("STIFF_BVH_BODY_MAJOR_MASK");
        return explicit_mask ? (atoi(explicit_mask) & 3) : 3;
    }();
    return mask;
}

static void makeFaceKeysBodyMajor(lbvh_f& tree,
                                  int     number,
                                  const int* active_idx,
                                  cudaStream_t stream)
{
    if(!(bvh_body_major_mask() & 1) || number < 1)
        return;
    _bodyMajorFaceKeys<<<(number + 255) / 256, 256, 0, stream>>>(
        tree._MChash, tree._faces, active_idx, tree._bodyId, number);
}

static void makeEdgeKeysBodyMajor(lbvh_e& tree,
                                  int     number,
                                  const int* active_idx,
                                  cudaStream_t stream)
{
    if(!(bvh_body_major_mask() & 2) || number < 1)
        return;
    _bodyMajorEdgeKeys<<<(number + 255) / 256, 256, 0, stream>>>(
        tree._MChash, tree._edges, active_idx, tree._bodyId, number);
}

static void computeNodeBodyLabels(lbvh& tree,
                                  int   number,
                                  cudaStream_t stream,
                                  bool publish_vf_front = false)
{
    if(!tree.m_node_body || !tree.m_prim_body || number < 1)
        return;
    computeNodeEnv(tree.m_node_body,
                   tree._nodes,
                   tree.m_prim_body,
                   tree._flags,
                   number,
                   stream);
    if(publish_vf_front)
        rebuild_bvh_vf_pair_front(
            tree._nodes, tree.m_node_body, number, stream);
}

static int bvh_wide8_mask()
{
    static const int mask = []
    {
        const char* value = getenv("STIFF_BVH_WIDE8_MASK");
        return value ? (static_cast<int>(strtol(value, nullptr, 0)) & 0xF)
                     : 0xF;
    }();
    return mask;
}

static const uint32_t* buildBvh8Children(const Node* nodes,
                                         const AABB* boxes,
                                         AABB*       temp_leaf_boxes,
                                         int         number,
                                         int         family,
                                         cudaStream_t stream)
{
    const int mode = bvh_wide8_mode();
    if(mode == 0 || number < 2 || (bvh_wide8_mask() & family) == 0)
        return nullptr;
    static_assert(sizeof(AABB) >= 8 * sizeof(uint32_t),
                  "temporary leaf boxes cannot hold BVH8 child fronts");
    auto* children = reinterpret_cast<uint32_t*>(temp_leaf_boxes);
    constexpr int threads = 256;
    const int blocks = (number - 1 + threads - 1) / threads;
    _buildBvh8Children<<<blocks, threads, 0, stream>>>(
        nodes, boxes, children, number, mode);
    return children;
}

static int bvh_ploc_mask()
{
    static const int mask = []
    {
        const char* value = getenv("STIFF_BVH_PLOC_MASK");
        return value ? (static_cast<int>(strtol(value, nullptr, 0)) & 0xF)
                     : 0xF;
    }();
    return mask;
}

static int bvh_ploc_radius()
{
    static const int radius = []
    {
        const char* value = getenv("STIFF_BVH_PLOC_RADIUS");
        const int parsed = value ? atoi(value) : 16;
        return std::max(1, std::min(64, parsed));
    }();
    return radius;
}

static int bvh_ploc_chunk_size()
{
    static const int chunk_size = []
    {
        const char* value = getenv("STIFF_BVH_PLOC_CHUNK");
        const int parsed = value ? atoi(value) : 256;
        return std::max(32, std::min(256, parsed));
    }();
    return chunk_size;
}

static bool buildPlocTopology(lbvh&         tree,
                              int           number,
                              int           family,
                              cudaStream_t  stream,
                              uint32_t*     node_max_element = nullptr)
{
    const int mode = bvh_ploc_mode();
    if(mode == 0 || number < 2 || (bvh_ploc_mask() & family) == 0)
        return false;

    auto* mch_u32 = reinterpret_cast<uint32_t*>(tree._MChash);
    auto* temp_u32 = reinterpret_cast<uint32_t*>(tree._tempLeafBox);
    uint32_t* clusters_b = mch_u32;
    uint32_t* nearest    = mch_u32 + number;
    uint32_t* flags      = temp_u32;
    uint32_t* offsets    = temp_u32 + number;
    uint32_t* assigned   = temp_u32 + 2 * number;
    if(mode < 3)
    {
        _buildPlocSingleWorkgroup<<<1, 256, 0, stream>>>(
            tree._nodes,
            tree._bvs,
            tree._indices,
            clusters_b,
            nearest,
            flags,
            offsets,
            assigned,
            node_max_element,
            number,
            bvh_ploc_radius(),
            mode == 2 ? 1 : 0);
    }
    else
    {
        const int chunk_size  = bvh_ploc_chunk_size();
        const int chunk_count = (number + chunk_size - 1) / chunk_size;
        uint32_t* roots = temp_u32 + 3 * number;
        _buildPlocChunksShared<<<chunk_count, 256, 0, stream>>>(
            tree._nodes,
            tree._bvs,
            roots,
            node_max_element,
            number,
            chunk_size,
            bvh_ploc_radius());
        if(chunk_count > 1)
            _buildPlocUpper<<<1, 256, 0, stream>>>(
                tree._nodes,
                tree._bvs,
                roots,
                tree._indices,
                clusters_b,
                nearest,
                flags,
                offsets,
                assigned,
                node_max_element,
                chunk_count,
                bvh_ploc_radius(),
                1);
    }
    return true;
}

static int bvh_sah_rotation_mask()
{
    static const int mask = []
    {
        const char* value = getenv("STIFF_BVH_SAH_ROTATION_MASK");
        return value ? (static_cast<int>(strtol(value, nullptr, 0)) & 0xF)
                     : 0xF;
    }();
    return mask;
}

static int bvh_sah_rotation_phase()
{
    static const int phase = []
    {
        const char* value = getenv("STIFF_BVH_SAH_ROTATION_PHASE");
        if(!value)
            return -1;
        const int parsed = atoi(value);
        return parsed >= 0 && parsed < 3 ? parsed : -1;
    }();
    return phase;
}

void optimizeSahTreelets(Node*         nodes,
                         AABB*         boxes,
                         uint32_t*     depths,
                         int           number,
                         int           family,
                         cudaStream_t  stream = 0,
                         uint32_t*     node_max_element = nullptr)
{
    const int cycles = bvh_sah_rotation_cycles();
    if(cycles == 0 || number < 3
       || (bvh_sah_rotation_mask() & family) == 0)
        return;
    const unsigned int threads = default_threads;
    const int blocks = (number - 1 + threads - 1) / threads;
    const int selected_phase = bvh_sah_rotation_phase();
    for(int cycle = 0; cycle < cycles; ++cycle)
    {
        for(int phase = 0; phase < 3; ++phase)
        {
            if(selected_phase >= 0 && phase != selected_phase)
                continue;
            _calcInternalDepths<<<blocks, threads, 0, stream>>>(
                nodes, depths, number);
            _rotateSahTreelets<<<blocks, threads, 0, stream>>>(
                nodes, boxes, depths, node_max_element, number, phase);
        }
    }
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


static int ee_range_prune_mode();

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
                  const uint32_t* node_max_element,
                  const uint32_t* wide_children,
                  cudaStream_t   stream = 0)
{
    int numbers = number;
    if(numbers < 1)
        return;
    const unsigned int threadNum = 256;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    if(node_max_element)
    {
        const int range_prune_mode = ee_range_prune_mode();
        if(range_prune_mode > 0)
        {
            if(wide_children)
            {
                if(range_prune_mode == 2)
                    _selfQuery_ee_sorted_prune_wide8<<<
                        blockNum, threadNum, 0, stream>>>(
                        _bodyID, _btype, _vertexes, _rest_vertexes, _edges,
                        _bvs, _nodes, _collisonPairs, _ccd_collisonPairs,
                        _cpNum, MatIndex, dHat, numbers,
                        _collision_skip_matrix, _collision_body_count,
                        _body_id_to_is_fem, node_env, wide_children,
                        node_max_element);
                else
                    _selfQuery_ee_range_prune_wide8<<<
                        blockNum, threadNum, 0, stream>>>(
                        _bodyID, _btype, _vertexes, _rest_vertexes, _edges,
                        _bvs, _nodes, _collisonPairs, _ccd_collisonPairs,
                        _cpNum, MatIndex, dHat, numbers,
                        _collision_skip_matrix, _collision_body_count,
                        _body_id_to_is_fem, node_env, wide_children,
                        node_max_element);
            }
            else
            {
                auto* range_kernel = range_prune_mode == 2
                                         ? _selfQuery_ee_sorted_prune
                                         : _selfQuery_ee_range_prune;
                range_kernel<<<blockNum, threadNum, 0, stream>>>(
                    _bodyID, _btype, _vertexes, _rest_vertexes, _edges,
                    _bvs, _nodes, _collisonPairs, _ccd_collisonPairs,
                    _cpNum, MatIndex, dHat, numbers,
                    _collision_skip_matrix, _collision_body_count,
                    _body_id_to_is_fem, node_env, node_max_element);
            }
            return;
        }
    }

    if(wide_children)
    {
        _selfQuery_ee_wide8<<<blockNum, threadNum, 0, stream>>>(
            _bodyID, _btype, _vertexes, _rest_vertexes, _edges, _bvs,
            _nodes, _collisonPairs, _ccd_collisonPairs, _cpNum, MatIndex,
            dHat, numbers, _collision_skip_matrix, _collision_body_count,
            _body_id_to_is_fem, node_env, wide_children);
        return;
    }

    // [ee-lb] STIFF_EE_LB occupancy A/B.  In the 2026-08-04 sm_89 DLTO
    // validation image, baseline and lb2 both link at 106 registers while lb3
    // links at 80; do not infer occupancy from the older 168-register comment.
    // Frozen-state timing retains lb2 and rejects lb3 on 4090.
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
                         const uint32_t* node_max_element,
                         const uint32_t* wide_children,
                         cudaStream_t   stream = 0,
                         const double*  alpha_dev = nullptr)
{
    int numbers = number;
    if(numbers < 1)
        return;
    const unsigned int threadNum = 256;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    if(node_max_element && ee_range_prune_mode() == 1)
    {
        if(wide_children)
            _selfQuery_ee_ccd_range_prune_wide8<<<
                blockNum, threadNum, 0, stream>>>(
                _bodyID, _btype, _vertexes, moveDir, alpha, _edges, _bvs,
                _nodes, _ccd_collisonPairs, _cpNum, dHat, numbers,
                _collision_skip_matrix, _collision_body_count,
                _body_id_to_is_fem, node_env, alpha_dev, wide_children,
                node_max_element);
        else
            _selfQuery_ee_ccd_range_prune<<<blockNum, threadNum, 0, stream>>>(
                _bodyID, _btype, _vertexes, moveDir, alpha, _edges, _bvs,
                _nodes, _ccd_collisonPairs, _cpNum, dHat, numbers,
                _collision_skip_matrix, _collision_body_count,
                _body_id_to_is_fem, node_env, alpha_dev, node_max_element);
    }
    else if(wide_children)
        _selfQuery_ee_ccd_wide8<<<blockNum, threadNum, 0, stream>>>(
            _bodyID, _btype, _vertexes, moveDir, alpha, _edges, _bvs,
            _nodes, _ccd_collisonPairs, _cpNum, dHat, numbers,
            _collision_skip_matrix, _collision_body_count,
            _body_id_to_is_fem, node_env, alpha_dev, wide_children);
    else
        _selfQuery_ee_ccd<<<blockNum, threadNum, 0, stream>>>(
            _bodyID, _btype, _vertexes, moveDir, alpha, _edges, _bvs, _nodes,
            _ccd_collisonPairs, _cpNum, dHat, numbers, _collision_skip_matrix,
            _collision_body_count, _body_id_to_is_fem, node_env, alpha_dev);
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
                  const int*      node_body,
                  const uint32_t* wide_children,
                  cudaStream_t    stream = 0)
{
    int numbers = number;
    if(numbers < 1)
        return;
    const unsigned int threadNum = 256;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    if(wide_children)
        _selfQuery_vf_wide8<<<blockNum, threadNum, 0, stream>>>(_bodyID,
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
                                           _body_id_to_is_fem,
                                           wide_children,
                                           node_body);
    else
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
                                           _body_id_to_is_fem,
                                           node_body);
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
                         const uint32_t* wide_children,
                         cudaStream_t    stream = 0,
                         const double*   alpha_dev = nullptr)
{
    int numbers = number;
    if(numbers < 1)
        return;
    const unsigned int threadNum = 256;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    if(wide_children)
        _selfQuery_vf_ccd_wide8<<<blockNum, threadNum, 0, stream>>>(
            _bodyID, _btype, _vertexes, moveDir, alpha, _faces, _surfVerts,
            _bvs, _nodes, _ccd_collisonPairs, _cpNum, dHat, numbers,
            _collision_skip_matrix, _collision_body_count,
            _body_id_to_is_fem, alpha_dev, wide_children);
    else
        _selfQuery_vf_ccd<<<blockNum, threadNum, 0, stream>>>(
            _bodyID, _btype, _vertexes, moveDir, alpha, _faces, _surfVerts,
            _bvs, _nodes, _ccd_collisonPairs, _cpNum, dHat, numbers,
            _collision_skip_matrix, _collision_body_count,
            _body_id_to_is_fem, alpha_dev);
}

void lbvh::FREE_DEVICE_MEM()
{
    if(getenv("STIFF_BVH_REFIT_STATS")
       && (m_refit_rebuilds || m_refit_reuses))
        printf("[bvh-refit-stats] primitives=%u rebuilds=%llu refits=%llu "
               "queries_per_rebuild=%.6f\n",
               static_cast<unsigned int>(m_refit_number),
               m_refit_rebuilds,
               m_refit_reuses,
               m_refit_rebuilds
                   ? (double)(m_refit_rebuilds + m_refit_reuses)
                         / (double)m_refit_rebuilds
                   : 0.0);
    auto release = [](auto*& pointer)
    {
        if(pointer)
        {
            CUDA_SAFE_CALL(cudaFree(pointer));
            pointer = nullptr;
        }
    };
    release(_indices);
    release(_MChash);
    release(_nodes);
    release(_bvs);
    release(_flags);
    release(_tempLeafBox);
    release(m_node_env);
    release(m_node_body);
    release(m_node_max_element);
    release(_sort_tmp);
    release(_mch_alt);
    release(_idx_alt);
    _sort_tmp_bytes = 0;
    _sort_cap       = 0;
    m_refit_nodes_identity = nullptr;
    m_refit_active_identity = nullptr;
    m_refit_number = 0;
    m_refit_since_rebuild = 0;
    m_refit_topology_ready = false;
    m_refit_capture_id = 0;
}

void lbvh::MALLOC_DEVICE_MEM(const int& number, bool allocate_node_max)
{
    CUDA_SAFE_CALL(cudaMalloc((void**)&_indices, (number) * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_MChash, (number) * sizeof(uint64_t)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_nodes, (2 * number - 1) * sizeof(Node)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_node_env, (2 * number - 1) * sizeof(int)));  // [env-part B]
    if(getenv("STIFF_BVH_PAIR_CACHE"))
        CUDA_SAFE_CALL(cudaMalloc((void**)&m_node_body,
                                  (2 * number - 1) * sizeof(int)));
    if(allocate_node_max)
        CUDA_SAFE_CALL(cudaMalloc((void**)&m_node_max_element,
                                  (2 * number - 1) * sizeof(uint32_t)));
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
    MALLOC_DEVICE_MEM(face_number, false);
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
    MALLOC_DEVICE_MEM(edge_number, ee_range_prune_mode() > 0);
}

AABB* lbvh_f::getSceneSize()
{
    calcLeafBvs(_vertexes, _faces, _bvs, face_number, 0,
                _bodyId, _collision_skip_matrix, _collision_body_count);

    // [B3 bbox-async] the init-only entry keeps the blocking reduction and now
    // also owns the host `scene` mirror (GIPC::init reads it for dHat); the
    // hot Construct paths never touch the host again.
    scene = calcMaxBV(_bvs, _tempLeafBox, face_number);
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
        if(shouldRefitTopology(*this, N, kSahFaceDcd, _active_idx, stream))
        {
            refitTopology<uint3, false>(
                *this, _faces, nullptr, 0.0, N, 0, stream, nullptr);
            return 0;
        }
        // [perenv-parallel #1] fully async on `stream` (no host sync) so per-env builds overlap.
        calcLeafBvs_indirect(_vertexes, _faces, _active_idx, _bvs, N, 0, stream);
        if(build_sah_oracle(*this, N, _active_idx, stream, nullptr,
                            "face-dcd-active"))
        {
            computeNodeEnv(m_node_env, _nodes, m_prim_env, _flags, N, stream);
            computeNodeBodyLabels(*this, N, stream, true);
            return 0;
        }
        calcMaxBV_async(_bvs, _tempLeafBox, N, stream);
        calcMChash(_MChash, _bvs, N, nullptr, nullptr, nullptr, nullptr, stream);
        makeFaceKeysBodyMajor(*this, N, _active_idx, stream);
        _iota_u32<<<(N + 255) / 256, 256, 0, stream>>>(_indices, N);
        _mc_sort_active(*this, _MChash, _indices, N, stream);
        sortBvs(_indices, _bvs, _tempLeafBox, N, stream);
        calcLeafNodes_indirect(_nodes, _indices, _active_idx, N, stream);
        if(!buildPlocTopology(*this, N, kSahFaceDcd, stream))
        {
            calcInternalNodes(_nodes, _MChash, N, stream);
            calcInternalAABB(_nodes, _bvs, _flags, N, stream);
        }
        optimizeSahTreelets(
            _nodes, _bvs, _flags, N, kSahFaceDcd, stream);
        computeNodeBodyLabels(*this, N, stream, true);
        return 0;
    }
    if(shouldRefitTopology(*this, face_number, kSahFaceDcd, nullptr, stream))
    {
        refitTopology<uint3, false>(*this,
                                    _faces,
                                    nullptr,
                                    0.0,
                                    face_number,
                                    0,
                                    stream,
                                    nullptr);
        return 0;
    }
    calcLeafBvs(_vertexes, _faces, _bvs, face_number, 0,
                _bodyId, _collision_skip_matrix, _collision_body_count);
    if(build_sah_oracle(*this, face_number, nullptr, stream, nullptr,
                        "face-dcd"))
    {
        computeNodeEnv(m_node_env, _nodes, m_prim_env, _flags, face_number, stream);
        computeNodeBodyLabels(*this, face_number, stream, true);
        return 0;
    }
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    calcMaxBV_async(_bvs, _tempLeafBox, face_number, 0);  // [B3 bbox-async] root AABB stays device-resident
    calcMChash(_MChash, _bvs, face_number, m_prim_env, m_prim_localid, m_env_offset, m_prim_v0);
    makeFaceKeysBodyMajor(*this, face_number, nullptr, 0);
    // [C-1 capture-safe sort] cub stable radix on pre-allocated instance
    // scratch (bit-identical order; no thrust internal malloc/free, so the
    // build can be recorded into a CUDA graph).
    _iota_u32<<<(face_number + 255) / 256, 256>>>(_indices, face_number);
    _mc_sort_active(*this, _MChash, _indices, face_number, 0);
    sortBvs(_indices, _bvs, _tempLeafBox, face_number);
    calcLeafNodes(_nodes, _indices, face_number);
    if(!buildPlocTopology(*this, face_number, kSahFaceDcd, 0))
    {
        calcInternalNodes(_nodes, _MChash, face_number);
        //CUDA_SAFE_CALL(cudaDeviceSynchronize());
        calcInternalAABB(_nodes, _bvs, _flags, face_number);
    }
    optimizeSahTreelets(
        _nodes, _bvs, _flags, face_number, kSahFaceDcd);
    computeNodeEnv(m_node_env, _nodes, m_prim_env, _flags, face_number);  // [env-part B]
    computeNodeBodyLabels(*this, face_number, 0, true);
    return 0;  //time0 + time1 + time2;
}

double lbvh_f::ConstructFullCCD(const double3* moveDir, const double& alpha, cudaStream_t stream,
                                const double* alpha_dev)
{
    if(_active_idx != nullptr && face_number_active > 0
       && face_number_active <= (int)face_number)
    {
        const int N = face_number_active;
        if(shouldRefitTopology(*this, N, kSahFaceCcd, _active_idx, stream))
        {
            refitTopology<uint3, true>(
                *this, _faces, moveDir, alpha, N, 0, stream, alpha_dev);
            return 0;
        }
        // [perenv-parallel #2] fully async on `stream` (mirrors the DCD active path): swept-leaf
        // build -> async max-BV -> Morton -> cub sort (pre-alloc scratch) -> tree. No host sync,
        // no malloc/free -> concurrent per-env swept builds+queries actually overlap.
        calcLeafBvs_fullCCD_indirect(_vertexes, moveDir, alpha, _faces,
                                     _active_idx, _bvs, N, 0, stream, alpha_dev);
        if(build_sah_oracle(*this, N, _active_idx, stream, nullptr,
                            "face-ccd-active"))
        {
            computeNodeEnv(m_node_env, _nodes, m_prim_env, _flags, N, stream);
            computeNodeBodyLabels(*this, N, stream);
            return 0;
        }
        calcMaxBV_async(_bvs, _tempLeafBox, N, stream);
        calcMChash(_MChash, _bvs, N, nullptr, nullptr, nullptr, nullptr, stream);
        makeFaceKeysBodyMajor(*this, N, _active_idx, stream);
        _iota_u32<<<(N + 255) / 256, 256, 0, stream>>>(_indices, N);
        _mc_sort_active(*this, _MChash, _indices, N, stream);
        sortBvs(_indices, _bvs, _tempLeafBox, N, stream);
        calcLeafNodes_indirect(_nodes, _indices, _active_idx, N, stream);
        if(!buildPlocTopology(*this, N, kSahFaceCcd, stream))
        {
            calcInternalNodes(_nodes, _MChash, N, stream);
            calcInternalAABB(_nodes, _bvs, _flags, N, stream);
        }
        optimizeSahTreelets(
            _nodes, _bvs, _flags, N, kSahFaceCcd, stream);
        computeNodeBodyLabels(*this, N, stream);
        return 0;
    }
    if(shouldRefitTopology(*this, face_number, kSahFaceCcd, nullptr, stream))
    {
        refitTopology<uint3, true>(*this,
                                   _faces,
                                   moveDir,
                                   alpha,
                                   face_number,
                                   0,
                                   stream,
                                   alpha_dev);
        return 0;
    }
    calcLeafBvs_fullCCD(_vertexes, moveDir, alpha, _faces, _bvs, face_number, 0,
                        _bodyId, _collision_skip_matrix, _collision_body_count,
                        alpha_dev);
    if(build_sah_oracle(*this, face_number, nullptr, stream, nullptr,
                        "face-ccd"))
    {
        computeNodeEnv(m_node_env, _nodes, m_prim_env, _flags, face_number, stream);
        computeNodeBodyLabels(*this, face_number, stream);
        return 0;
    }
    calcMaxBV_async(_bvs, _tempLeafBox, face_number, 0);  // [B3 bbox-async] root AABB stays device-resident
    calcMChash(_MChash, _bvs, face_number, m_prim_env, m_prim_localid, m_env_offset, m_prim_v0);
    makeFaceKeysBodyMajor(*this, face_number, nullptr, 0);
    // [C-1 capture-safe sort] cub stable radix on pre-allocated instance
    // scratch (bit-identical order; no thrust internal malloc/free, so the
    // build can be recorded into a CUDA graph).
    _iota_u32<<<(face_number + 255) / 256, 256>>>(_indices, face_number);
    _mc_sort_active(*this, _MChash, _indices, face_number, 0);
    sortBvs(_indices, _bvs, _tempLeafBox, face_number);

    calcLeafNodes(_nodes, _indices, face_number);
    if(!buildPlocTopology(*this, face_number, kSahFaceCcd, 0))
    {
        calcInternalNodes(_nodes, _MChash, face_number);
        calcInternalAABB(_nodes, _bvs, _flags, face_number);
    }
    optimizeSahTreelets(
        _nodes, _bvs, _flags, face_number, kSahFaceCcd);
    computeNodeEnv(m_node_env, _nodes, m_prim_env, _flags, face_number);  // [env-part B]
    computeNodeBodyLabels(*this, face_number, 0);

    return 0;
}

static int ee_range_prune_mode()
{
    static const int mode = []
    {
        const char* value = getenv("STIFF_EE_RANGE_PRUNE");
        if(!value)
            return 0;
        const int parsed = atoi(value);
        return parsed <= 0 ? 0 : (parsed == 2 ? 2 : 1);
    }();
    return mode;
}

double lbvh_e::Construct(cudaStream_t stream)
{
    // BVH-skip #3: when _active_idx is set (face_number_active reused as edge active count)
    if(_active_idx != nullptr && face_number_active > 0
       && face_number_active <= (int)edge_number)
    {
        const int N = face_number_active;
        const int range_mode = m_node_max_element ? ee_range_prune_mode() : 0;
        uint32_t* node_max = range_mode ? m_node_max_element : nullptr;
        if(shouldRefitTopology(*this, N, kSahEdgeDcd, _active_idx, stream))
        {
            refitTopology<uint2, false>(
                *this, _edges, nullptr, 0.0, N, 1, stream, nullptr, node_max);
            return 0;
        }
        // [perenv-parallel #1] fully async on `stream` (no host sync) so per-env builds overlap.
        calcLeafBvs_indirect(_vertexes, _edges, _active_idx, _bvs, N, 1, stream);
        if(build_sah_oracle(*this, N, _active_idx, stream, node_max,
                            "edge-dcd-active"))
        {
            computeNodeEnv(m_node_env, _nodes, m_prim_env, _flags, N, stream);
            computeNodeBodyLabels(*this, N, stream);
            return 0;
        }
        calcMaxBV_async(_bvs, _tempLeafBox, N, stream);
        calcMChash(_MChash, _bvs, N, nullptr, nullptr, nullptr, nullptr, stream);
        makeEdgeKeysBodyMajor(*this, N, _active_idx, stream);
        _iota_u32<<<(N + 255) / 256, 256, 0, stream>>>(_indices, N);
        _mc_sort_active(*this, _MChash, _indices, N, stream);
        sortBvs(_indices, _bvs, _tempLeafBox, N, stream);
        calcLeafNodes_indirect(
            _nodes,
            _indices,
            _active_idx,
            N,
            stream,
            node_max,
            range_mode == 2);
        if(!buildPlocTopology(*this, N, kSahEdgeDcd, stream, node_max))
        {
            calcInternalNodes(_nodes, _MChash, N, stream);
            calcInternalAABB(_nodes, _bvs, _flags, N, stream, node_max);
        }
        optimizeSahTreelets(
            _nodes, _bvs, _flags, N, kSahEdgeDcd, stream, node_max);
        computeNodeBodyLabels(*this, N, stream);
        return 0;
    }

    /*cudaEvent_t start, end0, end1, end2;
    cudaEventCreate(&start);
    cudaEventCreate(&end0);
    cudaEventCreate(&end1);
    cudaEventCreate(&end2);

    cudaEventRecord(start);*/
    const int range_mode = m_node_max_element ? ee_range_prune_mode() : 0;
    uint32_t* node_max = range_mode ? m_node_max_element : nullptr;
    if(shouldRefitTopology(*this, edge_number, kSahEdgeDcd, nullptr, stream))
    {
        refitTopology<uint2, false>(*this,
                                    _edges,
                                    nullptr,
                                    0.0,
                                    edge_number,
                                    1,
                                    stream,
                                    nullptr,
                                    node_max);
        return 0;
    }
    calcLeafBvs(_vertexes, _edges, _bvs, edge_number, 1,
                _bodyId, _collision_skip_matrix, _collision_body_count);
    if(build_sah_oracle(*this, edge_number, nullptr, stream, node_max,
                        "edge-dcd"))
    {
        computeNodeEnv(m_node_env, _nodes, m_prim_env, _flags, edge_number, stream);
        computeNodeBodyLabels(*this, edge_number, stream);
        return 0;
    }
    calcMaxBV_async(_bvs, _tempLeafBox, edge_number, 0);  // [B3 bbox-async] root AABB stays device-resident
    calcMChash(_MChash, _bvs, edge_number, m_prim_env, m_prim_localid, m_env_offset, m_prim_v0);
    makeEdgeKeysBodyMajor(*this, edge_number, nullptr, 0);
    // [C-1 capture-safe sort] see face variant.
    _iota_u32<<<(edge_number + 255) / 256, 256>>>(_indices, edge_number);
    _mc_sort_active(*this, _MChash, _indices, edge_number, 0);
    sortBvs(_indices, _bvs, _tempLeafBox, edge_number);

    //cudaEventRecord(end1);

    calcLeafNodes(
        _nodes, _indices, edge_number, node_max, range_mode == 2);
    if(!buildPlocTopology(
           *this, edge_number, kSahEdgeDcd, 0, node_max))
    {
        calcInternalNodes(_nodes, _MChash, edge_number);
        //CUDA_SAFE_CALL(cudaDeviceSynchronize());
        calcInternalAABB(_nodes, _bvs, _flags, edge_number, 0, node_max);
    }
    optimizeSahTreelets(
        _nodes, _bvs, _flags, edge_number, kSahEdgeDcd, 0, node_max);
    computeNodeEnv(m_node_env, _nodes, m_prim_env, _flags, edge_number);  // [env-part B]
    computeNodeBodyLabels(*this, edge_number, 0);
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
        const bool range_prune = m_node_max_element
                                 && ee_range_prune_mode() == 1;
        uint32_t* node_max = range_prune ? m_node_max_element : nullptr;
        if(shouldRefitTopology(*this, N, kSahEdgeCcd, _active_idx, stream))
        {
            refitTopology<uint2, true>(*this,
                                       _edges,
                                       moveDir,
                                       alpha,
                                       N,
                                       1,
                                       stream,
                                       alpha_dev,
                                       node_max);
            return 0;
        }
        // [perenv-parallel #2] fully async on `stream` (mirrors the DCD active path).
        calcLeafBvs_fullCCD_indirect(_vertexes, moveDir, alpha, _edges,
                                     _active_idx, _bvs, N, 1, stream, alpha_dev);
        if(build_sah_oracle(*this, N, _active_idx, stream, node_max,
                            "edge-ccd-active"))
        {
            computeNodeEnv(m_node_env, _nodes, m_prim_env, _flags, N, stream);
            computeNodeBodyLabels(*this, N, stream);
            return 0;
        }
        calcMaxBV_async(_bvs, _tempLeafBox, N, stream);
        calcMChash(_MChash, _bvs, N, nullptr, nullptr, nullptr, nullptr, stream);
        makeEdgeKeysBodyMajor(*this, N, _active_idx, stream);
        _iota_u32<<<(N + 255) / 256, 256, 0, stream>>>(_indices, N);
        _mc_sort_active(*this, _MChash, _indices, N, stream);
        sortBvs(_indices, _bvs, _tempLeafBox, N, stream);
        calcLeafNodes_indirect(
            _nodes, _indices, _active_idx, N, stream, node_max, false);
        if(!buildPlocTopology(*this, N, kSahEdgeCcd, stream, node_max))
        {
            calcInternalNodes(_nodes, _MChash, N, stream);
            calcInternalAABB(_nodes, _bvs, _flags, N, stream, node_max);
        }
        optimizeSahTreelets(
            _nodes, _bvs, _flags, N, kSahEdgeCcd, stream, node_max);
        computeNodeBodyLabels(*this, N, stream);
        return 0;
    }
    const bool range_prune = m_node_max_element
                             && ee_range_prune_mode() == 1;
    uint32_t* node_max = range_prune ? m_node_max_element : nullptr;
    if(shouldRefitTopology(*this, edge_number, kSahEdgeCcd, nullptr, stream))
    {
        refitTopology<uint2, true>(*this,
                                   _edges,
                                   moveDir,
                                   alpha,
                                   edge_number,
                                   1,
                                   stream,
                                   alpha_dev,
                                   node_max);
        return 0;
    }
    calcLeafBvs_fullCCD(_vertexes, moveDir, alpha, _edges, _bvs, edge_number, 1,
                        _bodyId, _collision_skip_matrix, _collision_body_count,
                        alpha_dev);
    if(build_sah_oracle(*this, edge_number, nullptr, stream, node_max,
                        "edge-ccd"))
    {
        computeNodeEnv(m_node_env, _nodes, m_prim_env, _flags, edge_number, stream);
        computeNodeBodyLabels(*this, edge_number, stream);
        return 0;
    }
    calcMaxBV_async(_bvs, _tempLeafBox, edge_number, 0);  // [B3 bbox-async] root AABB stays device-resident
    calcMChash(_MChash, _bvs, edge_number, m_prim_env, m_prim_localid, m_env_offset, m_prim_v0);
    makeEdgeKeysBodyMajor(*this, edge_number, nullptr, 0);
    // [C-1 capture-safe sort] see face variant.
    _iota_u32<<<(edge_number + 255) / 256, 256>>>(_indices, edge_number);
    _mc_sort_active(*this, _MChash, _indices, edge_number, 0);
    sortBvs(_indices, _bvs, _tempLeafBox, edge_number);

    calcLeafNodes(_nodes, _indices, edge_number, node_max, false);
    if(!buildPlocTopology(
           *this, edge_number, kSahEdgeCcd, 0, node_max))
    {
        calcInternalNodes(_nodes, _MChash, edge_number);
        calcInternalAABB(_nodes, _bvs, _flags, edge_number, 0, node_max);
    }
    optimizeSahTreelets(
        _nodes, _bvs, _flags, edge_number, kSahEdgeCcd, 0, node_max);
    computeNodeEnv(m_node_env, _nodes, m_prim_env, _flags, edge_number);  // [env-part B]
    computeNodeBodyLabels(*this, edge_number, 0);

    return 0;
}


void lbvh_f::SelfCollitionDetect(double dHat, cudaStream_t stream)
{
    const int tree_number = (_active_idx != nullptr && face_number_active > 0
                             && face_number_active <= (int)face_number)
                                ? face_number_active
                                : (int)face_number;
    const uint32_t* wide_children =
        buildBvh8Children(
            _nodes, _bvs, _tempLeafBox, tree_number, kSahFaceDcd, stream);
    reset_bvh_vf_pair_cache_counts(stream);
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
                 m_node_body,
                 wide_children,
                 stream);
    replay_bvh_vf_pair_cache(_vertexes,
                             _faces,
                             _cpNum,
                             _MatIndex,
                             _collisionPair,
                             _ccd_collisionPair,
                             dHat,
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
    const uint32_t* wide_children =
        buildBvh8Children(
            _nodes, _bvs, _tempLeafBox, N, kSahEdgeDcd, stream);
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
                 m_node_max_element,
                 wide_children,
                 stream);
}

void lbvh_f::SelfCollitionFullDetect(double dHat, const double3* moveDir, const double& alpha,
                                     cudaStream_t stream, const double* alpha_dev)
{
    const int tree_number = (_active_idx != nullptr && face_number_active > 0
                             && face_number_active <= (int)face_number)
                                ? face_number_active
                                : (int)face_number;
    const uint32_t* wide_children =
        buildBvh8Children(
            _nodes, _bvs, _tempLeafBox, tree_number, kSahFaceCcd, stream);
    fullCCDselfQuery_vf(
        _bodyId, _btype, _vertexes, moveDir, alpha, _faces, _surfVerts, _bvs, _nodes, _ccd_collisionPair, _cpNum, dHat, vert_number,
        _collision_skip_matrix, _collision_body_count, _body_id_to_is_fem,
        wide_children, stream, alpha_dev);
}

void lbvh_e::SelfCollitionFullDetect(double dHat, const double3* moveDir, const double& alpha,
                                     cudaStream_t stream, const double* alpha_dev)
{
    // Same fix as SelfCollitionDetect: launch count must match leaf count.
    int N = (_active_idx != nullptr && face_number_active > 0
             && face_number_active <= (int)edge_number)
                ? face_number_active
                : (int)edge_number;
    const uint32_t* wide_children =
        buildBvh8Children(
            _nodes, _bvs, _tempLeafBox, N, kSahEdgeCcd, stream);
    fullCCDselfQuery_ee(
        _bodyId, _btype, _vertexes, moveDir, alpha, _edges, _bvs, _nodes, _ccd_collisionPair, _cpNum, dHat, N,
        _collision_skip_matrix, _collision_body_count, _body_id_to_is_fem,
        m_node_env, m_node_max_element, wide_children, stream, alpha_dev);
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

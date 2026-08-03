#include "linear_system/utils/graph_node_resize.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace gipc
{
namespace graph_resize
{
namespace
{
constexpr int kMaxSlots = 64;

struct Registry
{
    cudaGraphDeviceNode_t* d_slots = nullptr;   // device array [kMaxSlots]
    std::vector<cudaGraphDeviceNode_t> pending;   // host mirror, index = slot
    int next_slot = 0;
    bool dirty = false;
};

Registry& registry()
{
    static Registry r;
    return r;
}

cudaGraphDeviceNode_t* device_slots(bool may_allocate)
{
    Registry& r = registry();
    if(!r.d_slots)
    {
        if(!may_allocate)
            return nullptr;
        if(cudaMalloc((void**)&r.d_slots,
                      sizeof(cudaGraphDeviceNode_t) * kMaxSlots)
           != cudaSuccess)
        {
            r.d_slots = nullptr;
            return nullptr;
        }
        cudaMemset(r.d_slots, 0, sizeof(cudaGraphDeviceNode_t) * kMaxSlots);
        r.pending.assign(kMaxSlots, nullptr);
    }
    return r.d_slots;
}
}  // namespace

// One thread reads the live count and resizes the target node for this replay.
// A null handle (never published, or a capture whose bind failed) leaves the
// node at its recorded capacity width -- always a correct, just wider, launch.
__global__ void _resize_grid_from_count(const cudaGraphDeviceNode_t* slots,
                                        int            slot,
                                        const int*     d_count,
                                        int            multiplier,
                                        int            block_size,
                                        int            capacity_blocks)
{
    if(threadIdx.x || blockIdx.x)
        return;
    cudaGraphDeviceNode_t node = slots[slot];
    if(!node)
        return;
    long long items =
        static_cast<long long>(*d_count) * static_cast<long long>(multiplier);
    if(items < 0)
        items = 0;
    long long blocks = (items + block_size - 1) / block_size;
    if(blocks < 1)
        blocks = 1;
    if(blocks > capacity_blocks)
        blocks = capacity_blocks;
    cudaGraphKernelNodeSetGridDim(node,
                                  dim3(static_cast<unsigned>(blocks), 1, 1));
}

bool enabled()
{
    static const bool on = []() {
        // Default ON. It measured 0% while the atomic convoy (C6-w) still
        // dominated; with that gone the padded-width cost surfaced and this
        // is where it lives -- not in kernel launch extents but in the graph's
        // MEMSET nodes, which CUDA replays as memset32 KERNELS. Measured on
        // forcegrip (15 frames, node-granularity nsys):
        //   memset32 total      136 ms -> 83 ms   (-39%)
        //   of which >10k-block  87 ms -> 35 ms   (-60%)
        //   all-kernel GPU time 1.060 s -> 0.951 s (-10%)
        // Wall moved only ~2% (inside noise) because this scene is not bound
        // by kernel time. Default stays OFF: resizing a node changes the
        // occupancy of everything downstream, which reorders the binned
        // scatter's atomicAdds -- gpu_native_rl_gate measured a 1.1e-14
        // velocity delta against its 3.6e-15 floor. A 27% win (the zero-skip)
        // is worth relitigating a gate's premise; a 2% one is not.
        const char* value = std::getenv("STIFF_GRAPH_DEVICE_RESIZE");
        return value && std::atoi(value) != 0;
    }();
    return on;
}

bool capturing()
{
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if(cudaStreamIsCapturing(cudaStreamPerThread, &status) != cudaSuccess)
    {
        cudaGetLastError();
        return false;
    }
    return status != cudaStreamCaptureStatusNone;
}

int arm(const int* d_count, int multiplier, int block_size, int capacity_blocks)
{
    if(!enabled() || !d_count || multiplier < 1 || block_size < 1
       || capacity_blocks < 1 || !capturing())
        return -1;
    // Never allocate here: arm() only runs during capture, and a cudaMalloc
    // inside a capture invalidates it. prewarm() owns the allocation.
    cudaGraphDeviceNode_t* slots = device_slots(false);
    if(!slots)
        return -1;
    Registry& r = registry();
    if(r.next_slot >= kMaxSlots)
        return -1;
    const int slot = r.next_slot++;
    _resize_grid_from_count<<<1, 1, 0, cudaStreamPerThread>>>(
        slots, slot, d_count, multiplier, block_size, capacity_blocks);
    return slot;
}

void bind_last(int slot)
{
    if(slot < 0)
        return;
    Registry& r = registry();
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    unsigned long long      id     = 0;
    cudaGraph_t             graph  = nullptr;
    const cudaGraphNode_t*  front  = nullptr;
    size_t                  count  = 0;
    if(cudaStreamGetCaptureInfo(
           cudaStreamPerThread, &status, &id, &graph, &front, &count)
           != cudaSuccess
       || status != cudaStreamCaptureStatusActive || count != 1 || !front)
    {
        cudaGetLastError();
        return;   // slot stays null -> node keeps its capacity width
    }
    cudaLaunchAttributeValue value{};
    value.deviceUpdatableKernelNode.deviceUpdatable = 1;
    if(cudaGraphKernelNodeSetAttribute(
           front[0], cudaLaunchAttributeDeviceUpdatableKernelNode, &value)
       != cudaSuccess)
    {
        cudaGetLastError();
        return;
    }
    r.pending[slot] = value.deviceUpdatableKernelNode.devNode;
    r.dirty         = true;
    if(std::getenv("STIFF_GRAPH_RESIZE_DIAG"))
        fprintf(stderr, "[graph-resize] bound slot %d -> node %p\n", slot,
                (void*)r.pending[slot]);
}

void publish()
{
    Registry& r = registry();
    if(!r.dirty || !r.d_slots)
    {
        r.next_slot = 0;
        return;
    }
    if(cudaMemcpy(r.d_slots,
                  r.pending.data(),
                  sizeof(cudaGraphDeviceNode_t) * kMaxSlots,
                  cudaMemcpyHostToDevice)
       != cudaSuccess)
        cudaGetLastError();
    if(std::getenv("STIFF_GRAPH_RESIZE_DIAG"))
    {
        int live = 0;
        for(auto h : r.pending) if(h) ++live;
        fprintf(stderr, "[graph-resize] published %d live handles\n", live);
    }
    r.dirty     = false;
    r.next_slot = 0;
}

void prewarm()
{
    if(!enabled() || capturing())
        return;
    device_slots(true);
}

void discard()
{
    Registry& r = registry();
    if(r.d_slots)
    {
        std::fill(r.pending.begin(), r.pending.end(), nullptr);
        if(cudaMemset(r.d_slots, 0,
                      sizeof(cudaGraphDeviceNode_t) * kMaxSlots)
           != cudaSuccess)
            cudaGetLastError();
    }
    r.dirty     = false;
    r.next_slot = 0;
}
}  // namespace graph_resize
}  // namespace gipc

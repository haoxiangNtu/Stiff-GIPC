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
        // Default OFF: the mechanism is verified (see the header) but on the
        // measured scenes the padded launch width was never the bottleneck --
        // resizing bought 0% while the atomic convoy fixed by
        // STIFF_SKIP_ZERO_DEPOSIT bought 27%. Opt-in so the capability is
        // there for a workload whose width actually dominates.
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

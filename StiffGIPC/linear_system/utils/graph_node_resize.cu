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
// [C6-aa] One slot array PER GRAPH CONSTRUCTION, held in a ring. The first
// version kept a single global array: every construction re-armed slot 0..k,
// and publish() overwrote the array the PREVIOUS construction's recorded
// resizers read. Concrete corruption path (audited, not hypothetical):
// step() records the full frame graph (publishes its handles), then
// prepare_gpu_rl() records the episode graph through the same registry —
// its publish was missing, so the episode replay's resizers read the FULL
// graph's handles and resized THAT graph's nodes with episode counts; after
// end_gpu_rl() the next step() replayed the full exec with a mergebin-zero
// grid sized for the episode's (smaller) unique count -> under-zeroed bins
// -> stale values entering the Hessian, silently. A ring entry per
// construction makes the arrays disjoint: each recorded resizer bakes the
// pointer of ITS OWN construction's array, and later constructions write
// elsewhere. Ring reuse only becomes a hazard after kRing constructions
// whose execs are ALL still alive — far beyond the 3-4 live execs the
// engine keeps (frame full graph, episode/gpu_rl; re-records destroy their
// predecessors).
constexpr int kRing = 16;

struct Entry
{
    cudaGraphDeviceNode_t* d_slots = nullptr;   // device array [kMaxSlots]
    std::vector<cudaGraphDeviceNode_t> pending;  // host mirror, index = slot
    int next_slot = 0;
};

struct Registry
{
    Entry entries[kRing];
    int   current    = -1;   // active construction, -1 = none
    int   next_index = 0;
    bool  allocated  = false;
};

Registry& registry()
{
    static Registry r;
    return r;
}
}  // namespace

// One thread reads the live count and resizes the target node for this replay.
// A null handle (bind failed, or the construction was discarded) leaves the
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
        // Default ON since C6-y (the RL gates' velocity floor was the only
        // blocker, and it was a gate defect). Measured: memset32 136 ms ->
        // 83 ms, all-kernel GPU time -10% on the forcegrip profile.
        const char* value = std::getenv("STIFF_GRAPH_DEVICE_RESIZE");
        return !value || std::atoi(value) != 0;
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

void prewarm()
{
    Registry& r = registry();
    if(!enabled() || r.allocated || capturing())
        return;
    for(Entry& entry : r.entries)
    {
        if(cudaMalloc((void**)&entry.d_slots,
                      sizeof(cudaGraphDeviceNode_t) * kMaxSlots)
           != cudaSuccess)
        {
            cudaGetLastError();
            entry.d_slots = nullptr;
            return;   // partial allocation: begin() will refuse to activate
        }
        cudaMemset(entry.d_slots, 0, sizeof(cudaGraphDeviceNode_t) * kMaxSlots);
        entry.pending.assign(kMaxSlots, nullptr);
    }
    r.allocated = true;
}

void begin()
{
    Registry& r = registry();
    // [C6-aa] begin() is guaranteed to run OUTSIDE any capture (guard below),
    // so it can own the allocation itself. Relying on try_launch_full_graph's
    // prewarm() missed the episode-first flow entirely: the warm-up frame is
    // the legacy boundary (no full-graph record), so prepare_gpu_rl reached
    // this point with nothing allocated and every episode/gpu_rl graph
    // silently recorded without resizers.
    prewarm();
    if(!enabled() || !r.allocated || capturing())
    {
        if(std::getenv("STIFF_GRAPH_RESIZE_DIAG"))
            fprintf(stderr,
                    "[graph-resize] begin() inert: enabled=%d allocated=%d "
                    "capturing=%d\n",
                    (int)enabled(), (int)r.allocated, (int)capturing());
        return;   // arm() stays inert (current == -1)
    }
    const int index = r.next_index;
    r.next_index    = (r.next_index + 1) % kRing;
    Entry& entry    = r.entries[index];
    std::fill(entry.pending.begin(), entry.pending.end(), nullptr);
    entry.next_slot = 0;
    // Outside capture by the guard above; a stale array from kRing
    // constructions ago must not leak old handles into the new recording.
    if(cudaMemsetAsync(entry.d_slots,
                       0,
                       sizeof(cudaGraphDeviceNode_t) * kMaxSlots,
                       cudaStreamPerThread)
       != cudaSuccess)
        cudaGetLastError();
    r.current = index;
}

int arm(const int* d_count, int multiplier, int block_size, int capacity_blocks)
{
    Registry& r = registry();
    if(!enabled() || r.current < 0 || !d_count || multiplier < 1
       || block_size < 1 || capacity_blocks < 1 || !capturing())
        return -1;
    Entry& entry = r.entries[r.current];
    if(entry.next_slot >= kMaxSlots)
        return -1;
    const int slot = entry.next_slot++;
    _resize_grid_from_count<<<1, 1, 0, cudaStreamPerThread>>>(
        entry.d_slots, slot, d_count, multiplier, block_size, capacity_blocks);
    return slot;
}

void bind_last(int slot)
{
    Registry& r = registry();
    if(slot < 0 || r.current < 0)
        return;
    Entry& entry = r.entries[r.current];
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
    entry.pending[slot] = value.deviceUpdatableKernelNode.devNode;
    if(std::getenv("STIFF_GRAPH_RESIZE_DIAG"))
        fprintf(stderr,
                "[graph-resize] construction %d bound slot %d -> node %p\n",
                r.current, slot, (void*)entry.pending[slot]);
}

void publish()
{
    Registry& r = registry();
    if(r.current < 0)
        return;
    Entry& entry = r.entries[r.current];
    if(cudaMemcpy(entry.d_slots,
                  entry.pending.data(),
                  sizeof(cudaGraphDeviceNode_t) * kMaxSlots,
                  cudaMemcpyHostToDevice)
       != cudaSuccess)
        cudaGetLastError();
    if(std::getenv("STIFF_GRAPH_RESIZE_DIAG"))
    {
        int live = 0;
        for(auto h : entry.pending)
            if(h)
                ++live;
        fprintf(stderr,
                "[graph-resize] construction %d published %d live handles\n",
                r.current, live);
    }
    r.current = -1;
}

void discard()
{
    Registry& r = registry();
    if(r.current < 0)
        return;
    Entry& entry = r.entries[r.current];
    std::fill(entry.pending.begin(), entry.pending.end(), nullptr);
    entry.next_slot = 0;
    // The abandoned recording may still be instantiated by a caller that
    // swallows the failure; zeroed slots turn its resizers into no-ops.
    if(entry.d_slots
       && cudaMemsetAsync(entry.d_slots,
                          0,
                          sizeof(cudaGraphDeviceNode_t) * kMaxSlots,
                          cudaStreamPerThread)
              != cudaSuccess)
        cudaGetLastError();
    r.current = -1;
}
}  // namespace graph_resize
}  // namespace gipc

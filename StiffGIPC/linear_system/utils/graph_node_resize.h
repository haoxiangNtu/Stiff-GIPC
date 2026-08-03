#pragma once
// [C6-v] Device-driven grid sizing for kernels recorded into a CUDA graph.
//
// A recorded kernel node carries the grid dimension it was captured with, and
// the contact-driven counts this engine works on (unique matrix blocks, pair
// counts) are only known on the device at REPLAY time.  The usual answer --
// train a host-side capacity tier, replay at that width, retry when it
// overflows -- is still the host guessing a size in advance: it pays for the
// slack every frame and pays a retry plus a re-record when the guess is low.
//
// CUDA 12 offers the exact primitive instead: a kernel node marked with
// cudaLaunchAttributeDeviceUpdatableKernelNode hands back a
// cudaGraphDeviceNode_t handle, and DEVICE code can call
// cudaGraphKernelNodeSetGridDim() on it.  A one-thread "resizer" node recorded
// ahead of the target therefore reads the live device counter and sets the
// target's grid for that same replay -- no host, no tier, no retry.
//
// Verified on sm_89 / CUDA 12.8 (scratchpad probes):
//   * the resize takes effect in the SAME replay, no re-instantiate
//   * it works for nodes inside conditional (WHILE/IF) bodies
//   * it works under STREAM CAPTURE, which is how this engine builds graphs:
//     cudaStreamGetCaptureInfo right after a launch yields exactly that node
//
// Handles only exist once the node does, i.e. after the resizer was already
// recorded, so the resizer reads its handle from a device slot that the host
// fills in after cudaStreamEndCapture (publish()).  Nothing is copied into the
// graph: no H2D/D2H node is added, so the zero-transfer episode audit holds.
#include <cuda_runtime.h>

namespace gipc
{
namespace graph_resize
{
// Master switch (STIFF_GRAPH_DEVICE_RESIZE, default on).  When off, or when
// the stream is not capturing, arm() returns -1 and callers keep their static
// width.
bool enabled();

// True while cudaStreamPerThread is capturing a graph.
bool capturing();

// Record a resizer node.  At replay it sets the target node's grid to
// ceil((*d_count * multiplier) / block_size) blocks, clamped to
// [1, capacity_blocks].  Returns a slot id to pass to bind_last(), or -1 if
// resizing is unavailable (caller then keeps the capacity-width launch).
int arm(const int* d_count, int multiplier, int block_size, int capacity_blocks);

// Call immediately after launching the target kernel: takes the node just
// added to the capture and marks it device-updatable, remembering its handle
// for publish().  A frontier that is not a single node means the launch was
// not the sole successor and the slot is abandoned (the target keeps its
// capacity width, which is always correct).
void bind_last(int slot);

// Called once after cudaStreamEndCapture succeeds: copies the collected node
// handles into the device slot array the resizers read.  Safe to call when
// nothing was armed.
void publish();

// Drop pending arm/bind state (a capture that was abandoned).
void discard();

// Allocate the device slot array. MUST be called outside any capture:
// cudaMalloc during capture invalidates it (observed as error 901 on the very
// first armed frame). Cheap and idempotent.
void prewarm();
}  // namespace graph_resize
}  // namespace gipc

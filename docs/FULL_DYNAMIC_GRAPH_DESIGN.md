# Full dynamic CUDA-Graph design (Phase C/D extension)

This document is the implementation contract for the independent
`codex/full-dynamic-graph` branch, based on Claude's `686a94d`.

## What "fully dynamic" means here

There are three different properties and they must not be conflated:

1. **GPU-native correctness**: frame counts, Newton/PCG/line-search decisions,
   reward/done/reset and fault state are device-resident. The host submits a
   graph launch and does not query a count inside the steady-state frame.
2. **Device-resized launch width**: an already-recorded kernel node changes its
   grid from a device counter. This is useful only when the consumer reads a
   contiguous live prefix or when the work is otherwise safely maskable.
3. **Unrestricted dynamic graph topology**: nodes, pointers, allocations and
   CUB algorithm plans can be created or destroyed during replay. CUDA Graphs
   do not provide this property. `cudaGraphKernelNodeSetGridDim/Param/Enabled`
   can update existing device-updatable nodes, but cannot add a node or perform
   an in-graph `cudaMalloc`/pointer replacement.

The target for this branch is (1) for the complete qualified IPC frame and (2)
for measurable contiguous-prefix work. It does **not** claim (3), because that
would contradict the CUDA Graph execution model.

## Current Claude baseline

`686a94d` already supplies:

- conditional Newton/PCG/line-search graph control;
- per-construction device resize handle storage;
- capacity tiers and device live-count masks;
- merged and isolated graph paths in their qualified envelopes;
- episode/GPU-RL device ABI and device-side semantics.

The baseline still records several CUB and segmented contact/triplet operations
with host-sized capacity arguments. Those operations are GPU-native in the
correctness sense: the data beyond the live count is neutralized or masked.
They are not all exact-width in the performance sense.

## Required architecture for the next level

### A. One device frame descriptor

All graph-consumed metadata belongs to one preallocated descriptor:

```text
FrameDeviceDescriptor {
  live_dcd/live_ccd/live_ground;
  live_arity[5];
  live_contact_class[4];
  segment_start[4], segment_capacity[4];
  unique_count, scan_total, generation;
  status/overflow/fault/reset masks;
}
```

Every consumer must read this descriptor or a device view derived from it. A
host mirror may be used only at a legal frame/episode boundary for telemetry,
recapture, or capacity growth.

### B. Preallocated arena and generation contract

All pointers used by a resident graph come from a preallocated arena. Capacity
growth is a boundary operation:

```text
device overflow -> status bit -> graph terminates -> boundary adjudication
                 -> grow/rebuild/recapture OR fail-closed
```

There is no allocation, resize, pointer swap, or CUB temp-storage growth inside
capture or replay. Every pointer-generation change invalidates the old exec.

### C. Dynamic contiguous-prefix kernels

Add resize handles only to kernels with a proof that lanes past the live prefix
perform no observable writes. First candidates are merge-bin clear/combine,
linear reductions, and compacted contact exports. Each candidate requires:

- graph-vs-host numerical parity;
- sanitizer coverage for zero, one, tier-boundary and overflow counts;
- evidence that the resize node itself adds less work than it removes.

### D. CUB and sorting

Do not pass a device pointer as CUB's `num_items`: the CUB host API is evaluated
while recording and its internal node plan is not a stable public device ABI.
Use one of these explicit strategies per operation:

1. **Fixed-capacity masked CUB** (current safe fallback): always record the
   capacity plan; sentinel/neutral lanes and live-count masks preserve results.
2. **Bucket graph**: record a finite set of power-of-two CUB plans and select one
   with device conditional nodes. This keeps topology fixed and makes the chosen
   plan exact within a bucket, at the cost of graph size and scratch memory.
3. **Custom device-controlled primitive**: replace CUB with a persistent or
   multi-pass kernel whose live count is read from the descriptor. This is the
   only route to arbitrary exact device lengths, and must be proven separately
   for determinism and memory traffic.

The implementation must choose per operation based on measured wall time, not
on the word "dynamic" alone.

### E. Contact/triplet layout

The current tiered layout is intentionally strided. Its live triplets are not a
single contiguous prefix, so blindly shrinking every launch to the total live
count is incorrect. The extension needs device-written segment offsets and
segment capacities, followed by kernels that read those offsets on device.
Padding remains legal only when it is neutral and when every downstream class
consumer is bounded by its own live segment.

### F. Failure and RL episode semantics

An episode cannot silently recapture in the middle of resident replay. For each
overflow, invalid, nonfinite, or capture fault, the device publishes:

- terminal status and error taxonomy;
- committed frame count and telemetry;
- whether the last frame was committed or rolled back;
- current/lagged contact history generation.

The caller may then end the episode and recapture at a boundary. A failed
episode is fail-closed; it must not resume ordinary `step()` using a failed
attempt's contact payload or stale host mirrors.

## Acceptance gates

No claim of "complete dynamic graph" is allowed until all of these pass on the
exact source commit:

1. merged and isolated FOLD-SHIRT: full 1550-frame replay;
2. graph-vs-host parity on a fixed 24-frame prefix plus end-state parity;
3. injected capacity overflow, recapture and fail-closed tests;
4. repeated prepare/end/step lifecycle with no cross-graph handle aliasing;
5. compute-sanitizer zero invalid accesses on zero/live/tier-boundary counts;
6. nsys on the full contact-rich window: zero H2D, zero D2H and zero host sync
   during resident frames;
7. A800 and 4090 evidence with the same descriptor ABI and graph signature;
8. performance comparison against v0.8.5 on both host-step and resident-RL
   workloads. A graph is a success only if the measured workload improves;
   correctness alone is not a performance claim.

## Feasibility and expected result

The target is feasible as a **fixed-topology, preallocated, device-controlled
graph**. It can exceed v0.8.5 in the RL regime because v0.8.5 has no episode
resident graph at all. It is not guaranteed to beat v0.8.5 on every large
contact replay: graph capture, capacity padding and CUB plans can cost more
than the host loop. The branch therefore treats the v0.8.5 comparison as a
benchmark gate, not as an assumption.

## Current replay evidence (2026-08-03)

The independent `codex/full-dynamic-graph` worktree was built with CUDA 12.8
and replayed the bundled `episode_fold_shirt_umi.hdf5` (1551 action rows) in
merged mode with one environment. The 7187-vertex/42-ABD scene completed all
1551 frames without a solver exception. The replay now has an optional
`CASE39_GRAPH_STATS=1` audit which counts full-graph, fallback and overflow
frames from the terminal status packet; this is an audit mode, not part of the
zero-sync steady state. Isolated mode with one environment is not a valid
isolated qualification (the resolver requires multiple declared environment
groups); a four-environment smoke reached the isolated path, but the full
1551-frame four-environment run remains a separate acceptance job.

CUDA Graph cannot perform arbitrary in-graph allocation or graph recapture. A
device overflow can publish status and stop/finish the current episode, but
rebuilding the executable still requires a host CUDA API call. The strict
no-CPU steady-state contract is therefore: no host work inside a running
frame/episode, preallocated capacity sufficient for that episode, and
recapture only between episodes. Calling this “CPU-free recapture” would be
incorrect.

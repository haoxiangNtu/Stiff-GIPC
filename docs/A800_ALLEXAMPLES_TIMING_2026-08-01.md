# A800 all-examples timing — 2026-08-01, binary 454fb59 (C6-l complete)

Method: every replay example plus towel, run PAIRED on the A800 (graph off,
then graph on = `STIFF_FRAME_GRAPH/FULL_GRAPH/C4/C6` knobs), merged mode,
sequential and exclusive on the card. 60-frame window where the example
accepts `CASE39*_FRAME_END`; towel runs its native 220+40 frames. "mean" is
each example's own per-step wall-clock report; fps = 1000/mean. The local
4090 was untouched throughout.

| example | off ms | off fps | on ms | on/off | on status |
|---|---|---|---|---|---|
| towel_scramble (961v) | 25.8 | 38.8 | ~25 | ~1.0x | C6-n: declined to release path (whole run 6s vs off 7s; was 2126.1ms/frame, 502s) |
| case39 | 113.2 | 8.8 | 203.9 | 1.80x | pass |
| case39_multienv | 114.5 | 8.7 | 155.3 | 1.36x | pass |
| cupshirt_finray | 124.8 | 8.0 | 240.4 | 1.93x | pass |
| case39_UMI_forcegrip | 138.0 | 7.2 | 379.7 | 2.75x | pass after C6-m (97% full-graph) |
| beaker_finray | 152.4 | 6.6 | 198.5 | 1.30x | pass |
| case39_UMI_beaker | 192.9 | 5.2 | 399.8 | 2.07x | pass after C6-m (95% full-graph) |
| foldshirt_finray | 319.1 | 3.1 | 475.1 | 1.49x | pass |
| beaker_finray_me | 326.2 | 3.1 | 460.8 | 1.41x | pass |
| cupshirt_finray_me | 362.4 | 2.8 | 702.5 | 1.94x | pass |
| foldshirt_stats | 391.1 | 2.6 | 538.8 | 1.38x | pass |
| foldshirt_multienv | 393.1 | 2.5 | 557.2 | 1.42x | pass |
| case39_UMI_sf_obb | 459.1 | 2.2 | 498.7 | **1.09x** | pass |
| case39_UMI_sf | 609.9 | 1.6 | 819.8 | 1.34x | pass |
| foldshirt_finray_me | 1167.2 | 0.9 | 1868.4 | 1.60x | pass |

## Readings

- **13 of 15 examples pass graph-on**, with the on/off ratio between 1.09x
  and 1.94x — consistently BETTER than the 4090's 1.6-3.3x band, matching
  the expectation that the A800's slow host link makes saved launch
  round-trips worth more. The heaviest-contact scenes amortize best
  (sf_obb 1.09x, beaker 1.30x); the lightest pay the most (cupshirt 1.9x).
- **towel-on anatomy — RESOLVED by C6-n**. The 82x outlier decomposed into
  three layers: 19 re-record-storm frames ate 219 of 468 s (a capture on
  the A800 costs ~20-30 s against ~1-3 s on the 4090); the remaining
  frames paid the fixed capacity-width launch cost; and the two-graph gate
  was no refuge (its inner graphs re-record on every capacity-generation
  bump — one hard crumple frame spent 15.7 s recording for the same 567
  Newton iterations the release solver finishes in 0.5 s). step() now
  declines ALL graph machinery below STIFF_FULL_GRAPH_MIN_VERTS (default
  1024) and runs the release frame (episodes exempt; gates pin 0).
  A800 after C6-n: the whole 220+40-frame towel run takes 6 s with the
  graph env vs 7 s off (was 502 s), footprint fingerprint unchanged
  (0.904). Measurement caveat recorded in the C6-n commit: per-frame
  [bench] timers understate the release path (async queue drains in the
  untimed get_vertices); total wall is the honest metric.
- **C6-o then cured the pathology at the root** (the threshold is now a
  performance choice, not a shield). nsys attribution overturned the
  re-record hypothesis: the two-graph towel run contained NO
  capture/instantiate calls at all -- it burned 150k PCG iterations (8x
  the committed work) inside FAILED attempts, because the tier-shaped
  partition flags its own truncation on the device at the first poisoned
  iteration (OVF_TRIPLETS, result=RETRY) but the host loop never looked
  before the frame terminal and spun starved to the 1000-Newton budget.
  Fix 1: solve_subIP polls an async 12-byte status snapshot per
  iteration and aborts the attempt immediately (committed trajectory
  proven bitwise-identical to the release path). Fix 2: per-axis
  growth-streak escalation (same axis re-crossing within 8 frames earns
  2x/4x extra headroom; contact-class tiers previously grew with NO
  headroom and re-crossed every ratchet frame). A800 with the threshold
  forced OFF: towel two-graph 20 s wall with zero runaway frames; towel
  full-graph 51 s (was 502 s); UMI forcegrip/beaker unchanged (97%/95%,
  RC=0).
- **C6-p cured the steady-state step-mode overhead at its root.** The
  forensics eliminated every fashionable suspect with measurements: PCG
  iteration counts and launch widths are IDENTICAL graph vs host, node
  dispatch gaps are 0.1us median, pre-launch host work is 12.5ms. The
  real cost was BAKED CAPACITY WIDTH: recorded kernels launch at
  tier x headroom extents, and the 2x training headroom doubled every
  capacity-wide pass (the converter unique-block reduction alone was
  86ms/frame at 8.4M slots over a 4.2M payload). With C6-o's escalation
  bounding growth storms, the default headroom flipped 2 -> 1:
  forcegrip 4090 2.08x -> 1.39x, A800 3.25x -> 1.75x (26s -> 14s wall,
  off 8s); beaker 4090 1.61x -> 1.21x, A800 20s RC=0. Cost: 1-2 extra
  growth frames per run (97% -> 94% coverage). All gates PASS under the
  new default; towel det stack bitwise x2 and committed trace identical
  to the release path across tier widths (pad-neutrality proven). The
  remaining 1.2-1.75x is the structural floor of baked-extent whole-
  frame graphs: tier-quantization slack plus re-record training frames.
- Pod interpreter gotcha: the canonical interpreter is plain `python3`
  (3.12, has polyscope + matches tools/numpy). Prefixing the venv into
  PATH for cmake leaks it into run sections (venv python lacks
  polyscope), and /usr/bin/python3 is 3.10 (incompatible tools/numpy).
  Keep build PATH exports out of run sections.
- **Two graph-on failures — RESOLVED by C6-m** (fallback runs the true
  graph-off frame: `s_layout_override_off` RAII forces `device_count_mode()`
  false during a capacity-fallback attempt). Root cause: the C6-i fallback
  still ran the tier-shaped partition + capacity guard, so a frame whose
  class counts exceeded the trained tiers re-flagged OVF_TRIPLETS on the
  fallback itself and burned the retry budget (`case39_UMI_beaker` frame
  20); the same capacity-shaped launches truncate/overrun in that state
  (`case39_UMI_forcegrip`'s sticky 700 surfacing at converter.cu:311).
  After C6-m both scenes pass on the A800: forcegrip 379.7 ms (97%
  full-graph), beaker 399.8 ms (95%). Both were re-verified locally too
  (G18/G19/D4/G17e PASS, towel det stack bitwise x2).
- Environment notes: the pod needed `polyscope` (now installed alongside
  trimesh). SOLVED: the missing `[graph-stats]` lines in the suite logs
  were a stale remote binary — its knob registry predated
  STIFF_GRAPH_STATS/STIFF_C6_ABD_STEP_GRAPH and the log headers show
  `[knob-registry][WARN] unknown STIFF_* knob` for both (silent no-op).
  Physics and timing were unaffected; the C6-m rerun on a fresh build
  prints coverage normally.

## C6-r: replay scenes through the EPISODE channel (4090, beaker replay)

The step-mode tables above beg the question "would the episode channel be
faster for replay?" Measured (UMI beaker grasp, 94-frame recorded
trajectory, actions baked through the same joint mapping the host loop
uses, episode boundary-resume on capacity growth):

| channel | ms/frame | vs host |
|---|---|---|
| host step() (release) | 156.4 | 1.00x |
| step() whole-frame graph | 244.4 | 1.56x |
| episode (device-resident, resume pattern) | 319-326 | ~2.05x |

Episode is the SLOWEST for heavy replay, and the decomposition says why:
prepare+launch is nearly free (0.15s), the cost is the replay itself
(29.1s for 91 frames = 319 ms/frame pure device rate). Episode residency
removes per-frame HOST overhead (a few ms) but charges the baked
capacity-width tax on every kernel -- and episodes bake 2x tier headroom
on purpose (no per-frame fallback, C6-q). On a 150 ms frame the saved
host milliseconds are ~2%, the width tax is +50-100%.

The channel economics in one line: episode residency pays when frames are
SMALL and host overhead is comparable to GPU work (D4 articulated
micro-frames: 4.1 ms/frame device vs 7.6-11.5 host-driven = 1.8-2.8x
WIN, growing with per-frame host cost); it loses when frames are big
(replay scenes: 150-400 ms of GPU work per frame). Contact-ramp
trajectories additionally need the boundary-resume pattern (first
launch died at frame 1 on tiers trained from a contact-free warmup;
after 2 host bridge frames the re-prepared episode ran the remaining
91/91 frames clean).

## C6-s: device-side sort dispatch research (three probes, 4090, CUDA 12.8 / cub 2.7)

Question: can the contact pipeline's sorts be dispatched FROM DEVICE code
(exact live counts, no pads, no D2H), bypassing "cub is host-only"?

Probe 1 (device-side cub, bare): cub::DeviceRadixSort::SortPairs called
from a __global__ parent COMPILES AND WORKS under RDC (which this build
already enables) + CDP2, and costs the same as host dispatch:
64-bit keys + 32-bit payload, N=4.2M: host 1.039 ms, device 1.076 ms
(+3.6%). Padded 2N host sort: 1.943 ms (1.87x — the pad tax measured in
isolation, linear as expected). The earlier claim "cub device-scope
algorithms cannot be called from device code" is REFUTED for cub 2.7.

Probe 2 (inside a CUDA graph): capturing the CDP parent into a graph is
LEGAL (capture/instantiate/replay all clean, and the replay honors a
device-resident count change without re-record) BUT the graph's node
edges only order against the PARENT kernel — downstream graph nodes race
with the parent's device-launched children (check kernel saw unsorted
data). Raw CDP inside whole-frame/episode graphs is therefore UNUSABLE;
in-graph device-side shape freedom remains limited to prebaked
size-class subgraph selection (the PCG self-tail pattern).

Probe 3 (bare stream pipeline): work enqueued on the SAME stream behind
the CDP parent IS fenced behind the parent's device-launched children
(CDP2 tail-launch semantics): 0/20 iterations raced. The step-mode
hybrid could adopt device-dispatched sorts today; the benefit there is
only removing frame-level count readbacks (~2%), since the host path is
already pad-free.

Net: the sort library is NOT the blocker; the GRAPH DEPENDENCY SEMANTICS
are. Exact-width inside graphs still needs either size-class bucketing
(~20% residual tax) or a future driver feature (device-side rewrite of
graph-node grid dims).

## C6-t: device-side GRID DIM update — the pad tax is NOT structural

CORRECTION of a claim made twice in the C6-p/C6-s writeups ("CUDA offers no
device-side rewrite of graph-node grid dims; only the host can
cudaGraphExecKernelNodeSetParams"). That claim is FALSE for CUDA 12.8.
The API exists and works:

  driver_types.h:  cudaGraphKernelNodeFieldGridDim  /**< Grid dimension update */
  cuda_device_runtime_api.h:401
      __device__ cudaError_t cudaGraphKernelNodeSetGridDim(
          cudaGraphDeviceNode_t node, dim3 gridDim);
  node handle: mark the node with launch attribute
      cudaLaunchAttributeDeviceUpdatableKernelNode (attr fills in devNode)

Probe results (4090, sm_89, CUDA 12.8, capacity 8.4M / block 256):

1. Plain graph, resizer kernel node -> payload kernel node: the resizer
   reads the DEVICE-RESIDENT live count and sets the payload node's grid;
   the change takes effect IN THE SAME REPLAY, no re-record, no host.
   live=8.4M/4.2M/1M/1024 all ran EXACT-WIDTH (blocks run == ceil(live/256)).
2. DECISIVE: the same thing INSIDE a conditional WHILE body -- the exact
   shape of our whole-frame graph -- also works. Marking a node inside a
   conditional body graph device-updatable succeeds, instantiate succeeds,
   and across 3 body iterations the payload ran exact-width every time
   (live=2.1M: 24576 blocks run vs 98304 padded).

Attribution of the graph-on extra GPU time (forcegrip, 15 frames, node
granularity), i.e. what this could recover:

  our capacity-width kernels (grid-update fixable) 2248.2 ms  77.9%
  cub device-scope algorithms (separate approach)   247.8 ms   8.6%
  PCG inner loop (static shape, not pad tax)        390.8 ms  13.5%

So ~78% of the whole-frame graph's overhead is our own launch-width padding
on kernels we author (the converter unique-block reduction dominates), and
that portion is addressable with device-side grid updates: a small resizer
node ahead of each capacity-width node, reading the live count the device
already maintains. The cub sorts (8.6%) need either the CDP path (legal
outside graphs, C6-s probe 3) or bucketing.

Implication: the "baked width" floor asserted in C6-p is not a CUDA
limitation, it is an unexploited API. Step-mode ratios (1.39x forcegrip /
1.21x beaker) and the episode large-frame penalty could both move
substantially toward parity. NOT yet implemented -- this entry records the
proven mechanism and the measured size of the prize.

### C6-t implementation, step 1: narrow the dominant kernel to `length`

The device-update API turned out not to be needed for the biggest item.
`Converter::_make_unique_block_warp_reduction`'s uniqueness pass already
opens with `if(i >= length) return;` yet was launched over `capacity`,
which under device_count_mode is `assembly_capacity_tier(length)` — up to
2x (measured on forcegrip: payload 4.2M launched at 8.4M). Both bounds are
record-time constants, so launching at `length` is equally graph-legal and
bit-identical: the removed threads performed no writes. This kernel was
the single largest overhead item in the nsys attribution (1284 ms of the
2248 ms "ours" bucket over 15 frames).

A third probe closed the wiring question for the remaining layers: our
graphs are built by STREAM CAPTURE, and the pattern works there too —
`cudaStreamGetCaptureInfo` right after a launch yields exactly that node,
`cudaGraphKernelNodeSetAttribute` marks it device-updatable MID-CAPTURE,
and a resizer node recorded ahead of it (reading the handle from device
memory, published after EndCapture) resizes it at replay: live=2.1M ran
8192 blocks against a 32768-block capacity.

What the API can NOT reach here, and why the machinery is deliberately NOT
built yet: the converter's remaining capacity-width passes (sentinel fill,
partition flags, sort, scan) genuinely need `capacity`, because the live
triplets are NOT a contiguous prefix — the partition stages them into four
class segments at TIER strides with zero pads between (13_kappa_partition:
segment_start[s] = segment_start[s-1] + class_tier[s-1]). "Shrink to the
true live count" is meaningless against a strided layout; that padding is
the price of the baked segment layout, not of launch width. A device
resizer would need a consumer whose live data IS a contiguous prefix; the
remaining profile items of that shape are small (barrier gradient/hessian
15.8 ms). Building the resizer with no measured consumer would be
speculative complexity — the mechanism is proven and recorded here, to be
used when a large contiguous-prefix consumer appears (e.g. if the sort
width is ever narrowed from `capacity` to `length`, which is the next
candidate and carries real risk against the sentinel-pad design).

## C6-v: what actually costs in the whole-frame graph (4090, clean GPU)

Every earlier attribution in this document was taken while an unrelated
20 GB training job shared the card, and cross-mode nsys comparisons had a
structural blind spot: `--cuda-graph-trace=node` does not itemise
DEVICE-launched graphs, so graph-off's PCG self-tail loop was invisible
and its kernel counts read 30x too low. Re-measured with the card idle and
the device loop disabled on the off side (STIFF_PCG_DEVICE_LOOP=0), the
comparison is finally like-for-like:

    graph-on  1.20 s GPU kernel time / 173288 kernels
    graph-off 0.68 s GPU kernel time / 132995 kernels

i.e. 1.3x the kernels but 1.76x the time. The single dominant item:

    delta 270.7 ms (50% of the whole gap)
    binned_block_merge_scatter: on 93x @ 2999 us, off 62x @ 133 us

22x per instance on an essentially identical element count. The cause is
not launch width: `binned_deposit` is an `atomicAdd`, the tier-shaped
payload is roughly half ZERO PAD triplets, and every pad carries the same
(0,0) key -- so they all dedup to ONE unique index and every padded
thread piled onto the same 9 bins. An atomic convoy, not a wide launch.

Skipping zero components (a no-op: adding 0.0 cannot change a bin)
removes it: forcegrip 60f, median of 3, 13.34 s -> 9.78 s (-27%), against
6.81 s graph-off, so the step-mode ratio moves 1.96x -> 1.44x.

Two lessons worth keeping:

* Padded WORK is expensive; padded LAUNCH WIDTH is nearly free. Three
  separate width-narrowing attempts (uniqueness pass to `length`, convert
  capacity to `length`, device-side grid resizing) measured 0%, -4% and
  0%. The device-resize machinery works exactly as designed and buys
  nothing here.
* The gates' bitwise graph-vs-baseline digest is brittle by nature. It
  rests on atomic arrival order, which is a property of the generated
  code: adding a never-taken bounds check to the scatter lambda alone
  took frame_graph_gate from 3/3 to 1/3 passes. Anything touching that
  kernel must be a separate kernel, not a runtime branch.

## C6-w: zero-skip becomes the default, and the bitwise gates move to the det stack

The C6-v fast path shipped opt-in because it flipped frame_graph_gate's
graph-vs-baseline BITWISE digest. Four measurements showed that objection
was misplaced:

1. **merged is not run-to-run deterministic, and never claimed to be.**
   `binned_deposit`'s default path is a bare `atomicAdd`
   ("merged/isolated never pay the bit-identity tax" — 13_kappa_partition).
   Measured on the plain release path (no graph, no det, no zero-skip),
   towel diverges from ITSELF at frame 2 (2.2e-14 relative) and reaches
   1.1e-4 by frame 119.
2. **Zero-skip perturbs by ~2 ULP**: 3.6e-16 absolute / 7e-16 relative on
   the gate fixture — two orders of magnitude BELOW the noise merged
   already carries.
3. **On the deterministic stack it is exactly neutral**: with
   STIFF_SPMV_DET the full-graph-vs-release divergence signature is
   bit-identical with zero-skip on and off (both frame 31, 1.816e-07 —
   that residue is the pre-existing capacity-grid reassociation).
4. **The gates only held because their fixture is a toy** (8 vertices, 4
   steps): chaos has no time to amplify and the atomic arrival order
   happens to repeat. That is a property of the scene, not a contract.

So the gates were rebuilt as a two-layer check that is STRICTLY stronger
than what they replaced:

* **det layer** — the bitwise digest comparison now runs under
  STIFF_SPMV_DET, where the order-free binned cascade makes "graph must
  not change a single bit" a mathematical statement about the code rather
  than an artefact of GPU scheduling.
* **default layer** — the non-det stack is checked against the baseline's
  OWN run-to-run envelope (3 baseline runs, error = min distance to any of
  them, budget = max(4x noise, 1e-11 x scale)), the same instrument G18
  already uses for contact scenes. Measured: G16 error=0.0 against
  noise=3.3e-16; G17a error=0.0 against noise=4.5e-15.

Shipped result (4090, clean GPU, 60 frames, median of 3):

| scene | graph off | graph on | ratio (was) |
|---|---|---|---|
| case39_UMI_forcegrip | 7.10 s | 10.37 s | **1.46x** (1.96x) |
| case39_UMI_beaker | 10.92 s | 13.57 s | **1.24x** (1.49x) |

verify_gates.sh: ALL 22 GATES GREEN with the new default, gold anchor
0544461bd82123ae unmoved (strict runs the det stack, where the change is
provably neutral).

## Cross-version reality check vs v0.8.5 (4090, clean GPU)

Two workloads, both built from a v0.8.5-compatible script so the
comparison is apples to apples (the two UMI replay wrappers are
byte-identical between the tag and HEAD).

**Large-frame replay (60 frames, median of 3).** The whole-frame graph is
a validation vehicle here, not a performance mode:

| scene | v0.8.5 | HEAD default (graph off) | HEAD graph on |
|---|---|---|---|
| case39_UMI_forcegrip | 7.26 s | 7.10 s (-2%) | 10.37 s (+43% vs tag) |
| case39_UMI_beaker | 10.09 s | 10.78 s (**+7% regression**) | 13.57 s (+35% vs tag) |

**RL micro-step workload** (the D4 articulated ABD chain pressed into the
ground, 300 steps after contact priming, dt=0.01). v0.8.5 has no episode
or gpu_rl API at all, so its only option is the host-driven step() loop:

| path | ms / step | vs v0.8.5 |
|---|---|---|
| v0.8.5 host step() | 19.3 (18.9 / 19.3 / 19.7) | 1.00x |
| HEAD host step() | 9.7 (9.2 / 9.7 / 10.3) | **2.0x faster** |
| HEAD gpu_rl residency | 3.85 (3.80 / 3.84 / 3.93) | **5.0x faster** |

That is the honest summary of the whole campaign: on big replay frames
the residency work is a wash (and the graph costs 35-43%), while on the
small-frame RL regime it is 2x from the residency work alone and 5x with
device residency on top. The regime, not the feature, decides.

Open item: the ~7% beaker regression in the DEFAULT path against v0.8.5
is a real regression on the path every user takes, measured over 6
interleaved runs with non-overlapping ranges.

## C6-x: where the remaining whole-frame-graph gap actually is (profiled clean)

Per-frame, engine-reported (forcegrip, 60 frames, setup excluded):
graph on 140.0 ms vs graph off 99.5 ms. Per-frame medians from the dump:
66.7 ms vs 48.4 ms (both distributions are heavy-tailed on the same hard
contact frames 27-29 / 52-58; the graph is 1.8x worse on those, so the
overhead scales with Newton iterations, not with frame count).

Timeline attribution at node granularity (15 frames):

| | graph on | graph off |
|---|---|---|
| GPU idle gaps | 279 ms | 541 ms |
| GPU busy share | 69% | 52% |
| host<->device copies | 1197 / 2 ms | 11533 / 16 ms |
| **GPU time actually computing** | **0.86 s** | **0.64 s** |

The graph wins every axis a graph is supposed to win -- half the idle
gaps, a tenth of the transfers, 1.3 ms of host work before the launch
(the 12.5 ms measured earlier was contention from an unrelated job on the
card). It loses because it asks the GPU to do **34% more work**.

Kernel-level split of the 376 ms extra (15 frames):

* 136 ms (36%) memset nodes replayed as memset32 kernels: graph mode
  zeroes 5.4 GB in 136 ms (38 GB/s) where stream mode zeroes 23.3 GB in
  48 ms (486 GB/s). Half that time is the 50 largest, up to 151 MB each --
  the mergebin cleared at tier capacity instead of the live unique count.
* 68 ms (18%) extra BVH collision queries: 33 passes vs 20, identical per
  pass. This is the pre-capture dry run plus re-records, i.e. the price of
  having to know buffer sizes before recording.
* 97 ms (26%) padded DATA inside O(N) passes -- converter lambdas
  (415 us vs 188 us per instance), cub sorts, and the tier staging
  kernels that graph-off does not run at all.

STIFF_GRAPH_DEVICE_RESIZE targets the first bucket and does work
(memset32 136 ms -> 83 ms, >10k-block ones -60%, all-kernel GPU time
-10%), but wall moved only ~2% and it perturbs the binned scatter's
atomic order enough to trip gpu_native_rl_gate (velocity delta 1.1e-14
against a 3.6e-15 floor). It therefore stays default OFF: a 27% win (the
zero-skip) is worth relitigating a gate's premise, a 2% one is not.

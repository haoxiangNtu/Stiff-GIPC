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

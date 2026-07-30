# Whole-frame graph: should it be default-on? — evidence, 2026-07-30

**Verdict: no. Keep it opt-in.** The graph is slower on every scene measured
(1.6x to 10.8x), two of six scenes still fail, and one of those failures is a
physics divergence on a pure-cloth scene.

## Method

`scripts`-free driver (kept in the session scratchpad, results copied to
`graph_default_on_evidence.csv`). For each scene: run the example twice
back-to-back on the same binary, first with the graph knobs unset, then with
`STIFF_FRAME_GRAPH=1 STIFF_FRAME_FULL_GRAPH=1 STIFF_C4_COLLISION_GRAPH=1
STIFF_C6_ABD_STEP_GRAPH=1`. Everything else identical (`merged`, N=1, 60 frames
where the example accepts `CASE39_FRAME_END`).

**What the milliseconds are.** Wall clock around `eng.step()`, measured inside
the example itself (`time.perf_counter()`), arithmetic mean over the frames the
example ran, frame 0 included (it is legacy in both configurations). Both paths
fully synchronise inside `step()` — the graph path at
`frame_transaction.cu:2395/3981`, the host path via its per-Newton-iteration
blocking reads — so this is end-to-end frame time, not CPU time. "host" means
the release `IPC_Solver` path where the host loop drives Newton and launches
kernels per iteration; it is **not** "time spent on the CPU".

Back-to-back matters: an earlier pair of hand-run measurements taken ~20 minutes
apart gave 1047 ms/frame for the host and 754 ms for the graph on the same scene
and binary, i.e. the opposite conclusion. Re-measured back-to-back the host is
324 ms. Machine state drifts by 3x on this box; only paired runs are usable.

## Results

| scene | graph off | graph on | ratio | coverage | on-status |
|---|---|---|---|---|---|
| towel_scramble (961 verts) | 22.5 ms | 244.1 ms | **10.8x slower** | 98% | **FAIL — physics** |
| cupshirt_finray | 109.4 ms | 351.9 ms | 3.22x slower | 98% | pass |
| case39_cup_softgripper | 98.6 ms | 263.8 ms | 2.68x slower | 98% | pass |
| foldshirt_finray | 268.5 ms | 604.5 ms | 2.25x slower | 98% | pass |
| foldshirt_multienv | 323.8 ms | 526.0 ms | 1.62x slower | 98% | pass |
| beaker_finray | 134.4 ms | — | — | 80% | **FAIL — OVF_CCD retry budget** |

Coverage is the fraction of frames actually recorded into the whole-frame graph;
98% everywhere means eligibility is not the limiter (frame 0 is legacy by
design).

## The towel divergence is the important one

Same 961-vertex towel, same drop pose (tilt 75.6, yaw 323.0, h 0.33):

    graph off:  footprint ratio vs flat = 0.905  -> folded    PASS
    graph on:   footprint ratio vs flat = 1.001  -> stayed flat  FAIL

The cloth does not crumple inside the graph. This is a pure-FEM scene with no
ABD grasp, so none of the C6 ABD machinery is involved. It is the cleanest
reproduction of an in-graph physics difference we have and should be the next
target.

## Why it is slower

The whole-frame graph trades **launch width** for **host round-trips**: a
recorded executable must be reusable across frames, so every launch is shaped by
a capacity tier rather than the live count. On these replay scenes the width
costs more than the round-trips it saves, and the smaller the scene the worse the
trade — towel (961 verts) is 10.8x, foldshirt_multienv (7187 verts, contact-rich)
is 1.62x.

Two width decisions were already walked back after measuring:

- `m_graph_train_cp[0]` was pinned to `MAX_COLLITION_PAIRS_NUM` in C6-d on the
  argument that slot 0 does not feed the triplet envelope so widening it is free.
  Free in memory, not in time: it is the launch extent for every per-pair contact
  kernel, so a ~28k-pair scene ran 737196-wide grids. It also OOM'd the two small
  scenes outright (towel, 961 verts, was being asked for 737196-pair capacity).
  Re-tiered to peak * headroom; that alone took foldshirt_multienv from 629.5 to
  526.0 ms and turned both OOMs into passes.
- The ground extent stays at worst case (`surf_vertexNum`) on purpose: the ground
  axis has no `OVF_*` bit, so a truncation there is undetectable and can never be
  adjudicated into a retry. Cheap in absolute terms and not worth the risk.

The next lever, not yet pulled: `m_contact_class_tier[0]` sits at 1048576 against
a 250171 observation (4.2x) because of the zero-pad allowance
(`sort_capacity - contact_tier`), which makes the radix sort and partition run
~2.5x wider than the honest payload. It cannot simply be removed: zero-pad
triplets hash to (0,0) and sort into class 0, so a class-0 tier below the pad
count would displace real class 1..3 entries — a wrong partition boundary, not an
overflow. The real fix is for the capacity mirror not to emit zero pads into the
sorted payload at all, which is a design change.

## What this means for the default

Keep all four knobs opt-in. The whole-frame graph's value proposition is the
GPU-native RL path — episode residency, zero host blocking, action sequences
pre-uploaded and observations streamed back — not per-frame throughput on
single-env replays. Revisit the default only when (a) both remaining failures are
fixed and (b) the graph is at least at parity on the scenes above.

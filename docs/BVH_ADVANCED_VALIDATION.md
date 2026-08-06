# Advanced BVH validation: tree quality, wide traversal, and coherence

Date: 2026-08-04

Branch: `codex/bvh-full-campaign`

Campaign base: `7e0bdab` (`codex/bvh-advanced-validation`)

Worktree: `/home/ps/Downloads/Stiff-GIPC-bvh-full-campaign`

This worktree is isolated from
`/home/ps/Downloads/Stiff-GIPC-c1-ls-graph`.  The Claude worktree is not
modified by this experiment.

## Experiment order

1. Measure an upper bound for binary-tree quality before implementing PLOC or
   GPU treelets.
2. Implement BVH8 only if a better binary tree materially reduces traversal
   work.  Compressing a weak tree first would confound topology quality with
   wide-node traversal quality.
3. Measure per-body/per-family temporal candidate caches.  A cache contains
   only raw primitive candidates; distance type, barrier activation,
   mollification, friction, and CCD classification are always recomputed.
4. Treat VF/EE/CCD unification as a broad-phase redesign, not a local kernel
   fusion.  VF and EE use different primitive trees, while DCD and swept CCD
   use different boxes and run at different solver stages.

## RBS-UIPC `merge_public` reference audit

The relevant branch is `origin/lhx/merge_public` in `/home/ps/rbs-uipc`
(checked out at `/home/ps/rbs-uipc-cmp-public`).  The relevant history is:

- `0b8345e3`: add `InfoStacklessBVH` and make it a broad-phase option;
- `e66210d5`: stabilize node-level BID/CID pruning;
- `b7ba74a5`: preload query BID/CID into shared memory and retain a V0
  comparison path.

This is **not** PLOC, SAH, or BVH8.  It still builds a 30-bit Morton LBVH.  Its
three transferable ideas are:

1. Sort external query AABBs by Morton code as well as sorting tree
   primitives.  See
   `src/backends/cuda/collision_detection/details/info_stackless_bvh.inl`
   (`QueryBuffer::build`, around lines 858--870).
2. Propagate a uniform body ID and contact ID into internal nodes, then reject
   a whole subtree when body-self-collision or the contact mask makes every
   leaf ineligible.  See
   `filters/info_stackless_bvh_simplex_trajectory_filter.cu` around lines
   284--300.
3. Load per-query BID/CID once per thread before the hot traversal loop.

Stiff-GIPC already launches EE self queries in the Morton-sorted leaf order
(`_nodes[N-1+thread].element_idx`), so idea 1 is already present for EE.  VF
still follows `_surfVerts`; adding another per-frame radix sort for VF must pay
for that sort and preserve CUDA-Graph capture.  Stiff-GIPC also already carries
uniform environment metadata (`m_node_env`) and exact EE ownership metadata
(`m_node_max_element`).  RBS's BID/contact-mask summaries remain a possible
additional pruning experiment, but are not evidence of a better geometric
tree.

The RBS branch's own 6400-vertex cloth sweep reports only about 0.4--3.6%
isolated improvement from `stackless_bvh` to `info_stackless_bvh`, with some
configurations slower and substantial run-to-run noise.  It is a useful design
reference, not a portable speedup claim.

## SAH tree-quality oracle

`STIFF_BVH_SAH_ORACLE=1` replaces only BVH construction with a deterministic
16-bin, three-axis, one-primitive-leaf SAH builder on the host.  It then uploads
the binary topology and runs the unchanged GPU VF/EE DCD and swept-CCD kernels.
The same collision filters, pair encodings, capacity protocol, and exact narrow
phase remain in force.

This path deliberately synchronizes the CUDA stream and is rejected during
stream capture.  It is not a candidate production implementation.  It answers
one question: how much traversal work could a substantially better binary tree
remove on frozen FOLD-SHIRT states?  If the reduction is small, PLOC/treelet and
BVH8 work should stop.  If it is large, the next implementation is a
capture-safe GPU treelet/PLOC builder with the same frozen-pair gates.

### Frozen FOLD-SHIRT results

The gate first evolved a baseline checkpoint, then loaded that exact state in
fresh baseline/oracle processes.  Geometry, physical pair multiplicity, and
the original encoded DCD and swept-CCD rows were exact in every cell below.
Numbers are the reduction in traversal `node_pops` relative to the 30-bit
Morton LBVH; primitive-test counts were unchanged.

| mode | frame | VF-DCD | EE-DCD | VF-CCD | EE-CCD |
|---|---:|---:|---:|---:|---:|
| merged | 1 | 17.28% | 16.05% | 16.24% | 15.53% |
| merged | 10 | 17.99% | 15.46% | 16.83% | 14.87% |
| merged | 30 | 14.75% | 14.08% | 14.66% | 13.45% |
| isolated | 1 | 17.28% | 16.02% | 16.10% | 15.63% |
| isolated | 10 | 17.57% | 15.41% | 16.79% | 14.98% |
| isolated | 30 | 14.74% | 13.90% | 14.27% | 13.43% |

The SAH trees had depths 18--21 for 12,819 active faces and 19,228 active
edges.  The deliberately non-production host build plus upload took roughly
10--11 ms for faces and 15--18 ms for edges.  Therefore the oracle proves a
real topology-quality opportunity, but is itself far more expensive than the
query work it saves and cannot be captured in a CUDA Graph.

## Graph-safe GPU treelet rotations

`STIFF_BVH_SAH_ROTATIONS=N` enables deterministic local binary rotations after
LBVH construction.  Every rotation preserves the primitive set under its
parent; three depth colors give disjoint write footprints.  Depths are rebuilt
on device, the optional EE subtree maximum is updated with the topology, and
the implementation has no allocation, readback, or synchronization.  It is
wired into face/edge, DCD/CCD, full/active construction and remains off by
default.

Two full three-phase cycles reduced node pops by about 6.2--7.7% over frozen
frames 1/10/30.  It passed:

- exact physical and encoded DCD/CCD multisets for merged and isolated at
  frames 1, 10, and 30;
- the 50-frame merged/isolated/strict trajectory gate;
- strict gold `0544461bd82123ae` exactly (merged maximum position delta was
  `6.005e-10`, below the existing `1e-8` gate).

Blindly applying even one full cycle is not profitable.  In a 30-frame nsys
capture it added about 98 ms of depth/rotation kernels; a fixed-call estimate
from query means saved only about 52 ms.  This candidate is therefore rejected
as an all-family optimization despite its lower node count.

The family/phase controls exist specifically to test the cost model:

- `STIFF_BVH_SAH_ROTATION_MASK`: bit 0 face-DCD, bit 1 edge-DCD, bit 2
  face-CCD, bit 3 edge-CCD;
- `STIFF_BVH_SAH_ROTATION_PHASE`: one of the three depth colors, or unset for
  all three.

On one frozen frame-30 FOLD checkpoint, 202 identical DCD rebuilds produced:

| candidate | total CUDA kernels | VF-DCD | known BVH build | encoded set |
|---|---:|---:|---:|---|
| LBVH | 608.296 ms | 383.034 ms | 19.421 ms | reference |
| face-DCD phase 0 | 594.466 ms | 363.464 ms | 21.842 ms | exact |
| face-DCD phase 1 | **590.085 ms** | **359.295 ms** | 21.831 ms | exact |
| face-DCD phase 2 | 601.080 ms | 370.714 ms | 21.776 ms | exact |

Phase 1 is a credible narrow candidate (about 3.0% for this DCD-only frozen
workload), not yet a whole-simulator speedup claim.  A freely evolved 30-frame
capture changed the number of Newton/query launches, so its much larger
apparent gain is deliberately not credited.  At this stage of the campaign,
repeated deterministic long-run and A800 measurements were still required;
the complete cross-architecture result is reported below.

The repository's 22-segment `verify_gates.sh` passed once with all new knobs
unset and once with face-DCD/phase-1 enabled.  Both runs kept strict gold
`0544461bd82123ae` and passed towel, quarantine, MAS, checkpoint, physics,
frame/episode/articulated-RL graphs, C4 collision graph, isolated whole-frame
graph, GPU-native RL, knob registry, and the FOLD smoke test.  The candidate
kernels are therefore demonstrated capture-safe on the existing Phase-C/D
graph gates; this did not replace the A800 performance run, which was
subsequently completed in the cross-architecture closure below.

### Decision threshold

A GPU tree-quality implementation is justified only if all of the following
hold:

- DCD and swept-CCD physical and encoded candidate multisets remain exact;
- VF and/or EE node pops fall materially (target at least 15% on the dominant
  families, not just a lower abstract SAH score);
- the predicted query saving exceeds the measured build overhead with margin;
- the default 22 gates, strict gold, complete 1550-frame merged/isolated
  trajectories, and CUDA-Graph capture remain valid.

The oracle crosses the tree-quality threshold, while the cheap local-rotation
implementation recovers only part of that headroom.  A PLOC/treelet rebuild is
therefore justified as further research.  BVH8 should be evaluated on that
better tree (and against a binary traversal of the same topology), not credited
with the SAH result.  Per-body temporal caching and a unified broad-phase
redesign remain independent experiments.

## Full-campaign continuation: body-pair coherence and PLOC/PLOC++

The continuation branch is `codex/bvh-full-campaign` in the separate worktree
`/home/ps/Downloads/Stiff-GIPC-bvh-full-campaign`, based on this report's
validated revision.  It still does not modify the Claude worktree.

### Conservative body-pair coherence census

The validation-only `STIFF_BVH_COHERENCE_AUDIT` path snapshots vertices on the
device and measures three strict invalidation rules without changing solver
results.  A global table remains valid while the global maximum displacement
is at most `delta/2`.  A table for distinct bodies A/B remains valid while
`max_disp(A) + max_disp(B) <= delta`; a self-body table uses
`2 * max_disp(A) <= delta`.  Collision masks and FEM self-collision semantics
are applied before counting eligible body pairs.

On the first 30 FOLD frames with a 1.5x broad-phase radius, global queries per
build were only 1.127 (merged) and 1.141 (isolated).  Body-pair granularity was
better: merged FEM-self/FEM-FEM/ABD-FEM/ABD-ABD ratios were
2.048/1.764/1.784/2.631, and isolated ratios were
2.204/1.809/1.836/2.923.  However, the dominant cloth body and its principal
pairs remained near one query per build.  On frozen frame 30, the same 1.5x
margin increased VF/EE DCD node pops by about 19% and swept VF/EE node pops by
about 18%; the raw swept CCD candidate list grew by about 43%.  Therefore the
census supports segmented body-pair/family caching, but does not yet establish
a net win for the dominant FOLD work.  Any implementation must cache only raw
broad-phase candidates and rerun distance type, barrier, mollification,
friction, and refined CCD on every reuse.

### Capture-safe PLOC implementations

Three opt-in construction controls were implemented for a direct cost test:

- `STIFF_BVH_PLOC=1`: original PLOC tie semantics, entire tree in one
  workgroup;
- `STIFF_BVH_PLOC=2`: PLOC++ coincident-box tie semantics, entire tree in one
  workgroup;
- `STIFF_BVH_PLOC=3`: Morton-contiguous chunks agglomerated in parallel from
  shared memory, followed by the PLOC++ upper-level single-workgroup pass.

`STIFF_BVH_PLOC_RADIUS`, `STIFF_BVH_PLOC_CHUNK`, and
`STIFF_BVH_PLOC_MASK` control the search radius, chunk size, and the four
face/edge DCD/CCD families.  All paths use memory that is already dead after
Morton sorting, perform no allocation/readback/synchronization, and are CUDA
Graph capture-safe.

The full PLOC++ R16 tree recovers most of the host SAH oracle's quality on
frozen merged frame 30: VF-DCD/EE-DCD/VF-CCD/EE-CCD node pops fall by roughly
13.4%/12.9%/13.6%/13.9%, with identical primitive-test counts and exact encoded
pair multisets.  R32 reaches roughly 14--15%.  The complete frozen FOLD pair
gate passed for merged and isolated frames 1, 10, and 30, and the 50-frame
merged/isolated/strict trajectory gate kept the strict gold exactly.

Construction cost rejects rebuilding either form for every query.  The table
below profiles 50 DCD plus 50 swept-CCD rebuilds on the same merged frame-30
checkpoint (406 total tree constructions, RTX 4090):

| builder | all CUDA kernels | builder kernels | four query kernels |
|---|---:|---:|---:|
| Morton LBVH | 441.323 ms | 12.043 ms topology+AABB | 401.154 ms |
| full PLOC++ R16 | 8,024.044 ms | 7,623.613 ms | 372.845 ms |
| full PLOC++ R32 | 13,725.297 ms | 13,348.522 ms | 349.241 ms |
| hierarchical shared PLOC++ C256/R16 | 640.414 ms | 226.577 ms | 386.227 ms |

Thus the better tree is real, but even the optimized hierarchical builder adds
about 214.5 ms more construction work than LBVH to save about 14.9 ms of query
work in this test.  It remains off by default.  It should be retested only when
the topology is amortized across exact AABB refits; a topology refit remains
collision-complete because every leaf and internal AABB is recomputed, while
tree quality affects performance rather than correctness.

### Exact topology refit changes the PLOC decision

`STIFF_BVH_REFIT_INTERVAL=N` now performs that missing amortization.  A full
build persists each leaf's original primitive `element_idx` and the binary
topology.  A refit reconstructs every current or swept leaf AABB directly in
that persistent leaf slot, then recomputes every internal AABB bottom-up.  It
does not reuse an old box and cannot omit a primitive; an arbitrarily poor
tree remains collision-complete and only traverses more nodes.  Checkpoint
restore and FEM/ABD teleport explicitly invalidate topology quality and force
one complete rebuild.  Scratch-pointer, active-list, and primitive-count
identity checks prevent one isolated/per-env scratch slot from borrowing
another slot's topology.

The initial frozen frame-30 scan used 200 DCD plus 200 deterministic swept-CCD
rebuilds.  Pure LBVH refit reduced all CUDA kernel time from 1,726.755 ms to
1,634.149 ms (5.36%) with exact DCD and CCD physical/encoded multisets.
Hierarchical PLOC++ rebuilt every 32 constructions used 1,627.996 ms; keeping
one PLOC++ topology for the whole frozen interval used 1,551.486 ms (10.15%
below baseline).  Full single-workgroup PLOC++ still lost most of its query
gain to even a few expensive builds and is rejected.

Frozen geometry hides tree degradation, so intervals 16/32/64/128/192/256
and effectively infinite were then profiled on the exact same deterministic
30-frame FOLD trajectory.  All candidates produced the same vertex hash
`642ddf01b9afc6fb`, pair hash `5dc74cc216dd2a8e`, 52,761 encoded pairs, and
per-frame Newton sequence.  Hierarchical PLOC++ interval 128 was the measured
optimum:

| candidate | all CUDA kernels | four query families | PLOC builds | refit leaves |
|---|---:|---:|---:|---:|
| rebuild Morton LBVH every query | 18,002.144 ms | 3,445.366 ms | 0 | 0 |
| pure LBVH, effectively infinite refit | 17,940.421 ms | 4,395.125 ms | 0 | 12.355 ms |
| hierarchical PLOC++, interval 64 | 17,147.050 ms | 3,331.408 ms | 27.214 ms | 12.201 ms |
| hierarchical PLOC++, interval 128 | **16,944.391 ms** | **3,308.522 ms** | 14.303 ms | 12.322 ms |
| hierarchical PLOC++, interval 192 | 17,483.807 ms | 3,357.927 ms | 9.923 ms | 12.318 ms |
| hierarchical PLOC++, interval 256 | 17,505.885 ms | 3,384.903 ms | 7.837 ms | 12.226 ms |
| hierarchical PLOC++, effectively infinite | 17,323.274 ms | 3,554.668 ms | 2.379 ms | 12.331 ms |

The interval-128 observed whole-kernel reduction is 5.88%; directly
attributable query/build/refit work supports a more conservative roughly
2.7% gain.  A separate IsaacSim process occupied the RTX 4090 at 97--99%
during these captures, so these particular samples were not used for a
default decision.  The later free-GPU 4090 and A800 matrices provide that
decision evidence.

The frozen FOLD gate passed at frames 1/10/30 for merged and isolated with
exact original DCD and CCD encodings.  The 50-frame merged/isolated/strict gate
also passed and retained strict gold `0544461bd82123ae`.  CUDA stream capture
forces the first construction in each distinct capture to be a full build, so
replay cannot accidentally repeat an all-refit graph forever.  The candidate
then passed the frame-graph transaction gate, all three C4 collision scenarios,
the C5 isolated whole-frame gate, episode graph, articulated episode graph,
GPU-RL, and GPU-native-RL gates.  The traversal-audit validation build exposed
and fixed a pre-existing capture hazard as part of that run: disabled audit
symbols are now initialized by their device-zero value and process-fixed audit
settings are copied only when they actually change, rather than issuing a
synchronous `cudaMemcpyToSymbol` from every captured `buildCP()`.

Isolated execution originally reported 43 full rebuilds and zero refits in the
C5 gate because one scalar topology state was shared while execution rotated
among per-environment scratch/active-list addresses.  The implementation now
keeps a topology generation for every `_nodes` scratch allocation.  A slot is
reused only when its active-list pointer and primitive count still match; if a
different environment rotates onto the slot, the mismatch forces an exact full
rebuild.  Checkpoint load and FEM/ABD teleport invalidate all generations.

After that change, ordinary C5 branches measured 5 rebuilds and 328 refits
(66.6 queries per rebuild), while graph-capture branches measured 13 rebuilds
and 30 refits.  The complete C5 numerical/isolation gate passed.  A four-env
frozen FOLD gate at frames 1, 10, and 30 also retained exact geometry and exact
physical and original-encoding multisets for both DCD and swept CCD.

The exact same isolated four-env frame-30 checkpoint was then replayed for 200
DCD and 200 CCD queries under Nsight Systems:

| metric | rebuild Morton LBVH | PLOC++ / refit-128 | delta |
|---|---:|---:|---:|
| all CUDA kernels | 15,168.870 ms | 14,262.041 ms | **-5.98%** |
| four query families | 14,372.751 ms | 13,973.022 ms | **-2.78%** |
| known BVH build kernels, excluding CUB | 380.123 ms | 280.913 ms | **-26.10%** |
| VF DCD | 3,408.726 ms | 3,330.423 ms | -2.30% |
| EE DCD | 5,161.698 ms | 5,124.504 ms | -0.72% |
| VF CCD | 792.702 ms | 720.843 ms | -9.07% |
| EE CCD | 5,009.625 ms | 4,797.252 ms | -4.24% |

The candidate performed 33 rebuilds and 3,177 refits, or 97.27 queries per
rebuild.  Raw pair order changed, as expected for concurrent atomic output,
but the physical and original encoded DCD/CCD multisets were exact.  Freely
evolved isolated runs are intrinsically noisy here: two baseline replicas
already diverged by `8.57e-2` in position and followed different Newton/query
counts.  Their wall times therefore are not used as evidence.  The frozen
fixed-workload result is the credited isolated speedup; a long trajectory and
A800 repeat were still required at this point.  Both are closed below.

### Same-topology BVH8 traversal

`STIFF_BVH_WIDE8` builds a maximum-eight-child traversal front from the
already constructed binary tree.  Child references and AABBs remain those of
the exact binary topology, so a BVH2/BVH8 comparison cannot accidentally
credit BVH8 with PLOC/SAH tree quality.  The child table reuses the post-sort
temporary leaf-AABB buffer, performs no allocation or host operation, and is
rebuilt on the caller's stream immediately before the selected query.

Three collapse policies were measured: fixed three-level fronts (mode 1),
minimum SAH-increase expansion (mode 2), and repeatedly expanding the largest
frontier box (mode 3).  Fixed levels made the four query kernels about 16%
slower.  Minimum-increase expansion was worse because its front construction
was expensive and it retained large overlapping parents.  Largest-box-first
was the only useful policy.  `STIFF_BVH_WIDE8_MASK` uses the same family bits
as the PLOC/rotation masks.

The profitable slice is face DCD only (`STIFF_BVH_WIDE8=3`,
`STIFF_BVH_WIDE8_MASK=1`).  Two 1,000-rebuild frozen frame-30 profiles produced:

| metric | binary LBVH mean | face-DCD BVH8 mean | delta |
|---|---:|---:|---:|
| all CUDA kernels | 5,844.237 ms | 5,571.858 ms | -4.66% |
| DCD query kernels | 5,462.235 ms | 5,155.349 ms | -5.62% |
| VF-DCD query | 3,616.976 ms | 3,302.339 ms | -8.70% |
| BVH8 front construction | 0 | 34.050 ms | +34.050 ms |

Enabling edge DCD as well reduced its frozen query kernel by roughly 3%, but
changed the freely evolved merged 50-frame result by `1.101e-7`, above the
existing `1e-8` candidate contract.  Face DCD alone passed the
merged/isolated/strict 50-frame gate (`1.531e-9` maximum merged position delta,
strict gold exact).  Frozen FOLD merged/isolated frames 1, 10, and 30 retained
exact DCD and swept-CCD physical multisets and exact original encodings.

BVH8 does not stack materially with the expensive full PLOC tree.  On the same
PLOC++ R16 topology and 100 DCD rebuilds, binary queries used 492.491 ms;
BVH8 queries used 485.380 ms plus 6.340 ms to build the fronts.  That is nearly
break-even before the roughly 7.47-second PLOC construction cost.

The final fair stacking test used the same isolated four-env frame-30
checkpoint for 200 DCD and 200 CCD rebuilds.  Both physical and original
encoded pair multisets were exact.  PLOC++/refit-128 alone used 13,726.146 ms
of total kernel time; adding face-DCD BVH8 used 13,816.766 ms (**+0.66%**).
VF-DCD itself increased from 3,266.159 to 3,345.213 ms, plus 26.279 ms for
1,608 front constructions.  Therefore BVH8 is rejected from the PLOC/refit
winner bundle.  Its earlier gain applies only to the cheap Morton LBVH branch,
which remains an independent opt-in candidate rather than a cumulative win.

## Body-pair VF cache and shared traversal front

The first real per-body/per-family cache is now implemented for VF-DCD behind
`STIFF_BVH_PAIR_CACHE=1`.  It stores only raw `(vertex, face)` broad-phase
candidates in fixed body-pair segments.  Every reuse calls the unchanged exact
PT classifier, so distance type, active-barrier status, contact encoding, and
all later mollification/friction work are recomputed.  A maximal uniform-body
subtree front lets an invalid pair start at only that target body's roots;
valid pairs skip tree traversal and replay their own segment.  EE and all CCD
semantics use independent candidate families and are never inferred from a VF
hit.

`STIFF_BVH_PAIR_CACHE_DEVICE=1` removes the validation prototype's per-query
host decision.  Pair-specific reference positions, two-sided maximum
displacement reductions, the conservative
`max_disp(A) + max_disp(B) <= delta` decision, invalid-generation rebasing,
segment-count reset, and overflow invalidation all execute in one device
kernel.  Static topology discovery and allocation happen once before steady
state; terminal statistics are copied only after the run.  The implementation
has 33 eligible FOLD body pairs and 56,314 pair-specific reference entries.

Correctness evidence for margin 1.5 includes:

- a deterministic merged 30-frame FOLD run where two baselines, the old
  host-audit cache, and device validity all produced the same vertex hash
  `642ddf01b9afc6fb`, pair hash `5dc74cc216dd2a8e`, 52,761 encoded pairs, and
  identical per-frame Newton counts;
- the 50-frame merged/isolated/strict candidate gate, with strict gold
  `0544461bd82123ae` exact and non-strict deltas inside measured multi-baseline
  noise envelopes;
- a frozen frame-30, 1,000-rebuild test with exactly the same 30,571 encoded
  DCD rows and no traversal-front fallback.

The frozen-state upper bound is large: VF-DCD traversal plus cache replay fell
from 3,614.698 ms to 1,105.441 ms (about 69.4%), while the paired EE control
changed by only 0.2%.  Real evolution is much less favorable.  In an nsys
capture of the exact deterministic 30-frame workload, ordinary VF-DCD used
1,252.761 ms.  The candidate used 860.298 ms of fresh traversal, 267.449 ms of
exact replay, 16.053 ms of device validity, 1.661 ms of front construction,
and 1.544 ms in a now-removed redundant reset kernel: 1,147.005 ms total, only
8.44% less.  Total CUDA kernel time fell from 17,335.744 ms to 17,194.224 ms
(0.82%); unrelated paired kernels show that this whole-process delta is near
the machine's current contention/noise floor.  The structural VF saving is
positive, but not large enough to default this feature.

Stacking the cache with PLOC++/refit-128 preserves that frozen structural
gain but is not an end-to-end winner.  On one exact merged frame-30
checkpoint, 1,000 DCD rebuilds retained the exact physical and original
encoded contact multiset.  PLOC/refit plus body-major traversal used
5,262.613 ms of total kernel time and 3,202.398 ms in VF-DCD.  Adding the
device cache used 3,257.577 ms total (**-38.10%**); its VF path was 26.010 ms
fresh traversal + 1,143.954 ms exact replay + 28.007 ms validity + 0.030 ms
front construction (**-62.59%** versus the uncached VF query).

Freely evolved 30-frame replicas exposed the missing cost: two uncached
body-major runs completed 730/734 Newton iterations, while two cached runs
completed 776/856.  The uncached replicas already differed by `3.26e-2` in
maximum final position, so raw trajectory deltas are not a completeness
oracle; frozen pair gates remain exact.  Nevertheless, the cache consistently
changed emission/assembly order enough to follow a more expensive Newton
path.  It is therefore rejected from the PLOC/refit winner bundle unless a
future order-preserving publication step removes that solver-side penalty.

Margin 1.5 was the only credible tested point.  It reused 43.95% of pair uses
in the deterministic workload and reduced VF-DCD node pops from 306.7 million
to 95.2 million without changing results.  Margin 1.25 reused too little;
margin 2.0 replayed 21.4 million raw candidates in a 30-frame free run and
changed the trajectory beyond the observed two-baseline RMS envelope.  The
next cache tests must be pair-selective and must measure EE-DCD and swept CCD
independently rather than assuming the VF result transfers.

### EE-DCD extension and the shared-front concurrency boundary

`STIFF_BVH_PAIR_CACHE_MASK` now selects VF-DCD with bit 0 and EE-DCD with bit
1.  EE stores raw edge-index pairs and reruns the unchanged exact
`_checkEEintersection<false>` path on every reuse.  A first combined prototype
incorrectly shared one mutable front between the default-stream VF detector
and auxiliary-stream EE detector.  The frozen oracle caught the resulting
race: only 25,531 of the baseline's 30,832 encoded pairs survived.  VF and EE
now own separate front arrays, counters, overflow flags, device symbols, and
front-build kernels.  The same counterexample then retained all 30,832 rows,
with exact physical and original-encoding multisets and zero front overflow.
The complete frozen FOLD gate also passed for merged and isolated frames 1,
10, and 30 with exact geometry and exact DCD/CCD physical and encoded
multisets; isolated presently exercises the honest cache-disabled path.
This establishes the safe scope of a shared traversal runtime: code and
read-only interfaces may be shared, but mutable fronts must be isolated by
tree family and concurrent stream.

EE-only caching is a clear performance loss on the PLOC/refit tree.  In the
same frozen frame-30 checkpoint repeated for 1,000 DCD rebuilds, ordinary EE
traversal used 2,236.166 ms.  The candidate used 37.124 ms of residual
traversal, 3,513.837 ms of exact replay, 27.783 ms of shared validity, and
3.680 ms of front construction: 3,582.424 ms (**+60.20%**).  Total kernel time
rose from 5,925.296 to 7,284.339 ms (**+22.94%**).  The reason is direct: each
query replayed about 89,621 raw EE candidates, and exact EE reclassification
cost more than traversing the already-good tree.

The corrected combined VF+EE cache is still a frozen structural win because
VF's saving is larger than EE's loss.  Against a body-major VF+EE baseline,
1,000 rebuilds kept the exact 30,832 encoded rows while total kernel time fell
5,526.666 to 4,885.925 ms (**-11.59%**) and the complete DCD path fell
5,399.152 to 4,757.572 ms (**-11.88%**).  Two free-run replicas did not show a
consistent Newton penalty (baseline 750/812, candidate 764/797), but baseline
nondeterminism and large wall-time variance prevent crediting an end-to-end
speedup.  EE-only is rejected; the combination remains diagnostic/opt-in.

### Swept CCD endpoint-coherence audit

A DCD reference position is not a valid proof for a swept query.  The new
`STIFF_BVH_CCD_COHERENCE_AUDIT` records both endpoints of every vertex sweep
for each body pair.  A cached list is considered complete only while
`max_endpoint_disp(A) + max_endpoint_disp(B) <= delta`; linear interpolation
then bounds every point of each swept primitive by the same per-body maximum.
The audit is host-synchronized diagnostic code only and is never active in a
timed candidate or CUDA Graph.

On three freely evolved 30-frame FOLD runs, pair/work reuse was:

| margin scale | pair reuse | VF-CCD weighted work reuse | EE-CCD weighted work reuse |
|---:|---:|---:|---:|
| 1.25 | 30.80% | 27.10% | 25.60% |
| 1.50 | 35.35% | 31.45% | 29.29% |
| 2.00 | 45.43% | 39.99% | 37.71% |

These values are high enough to justify one real base-gap-filtered replay
prototype, but too low to claim a win: the earlier margin census found that a
2.0 swept list grows about 43%.  A production candidate must therefore cache
expanded raw pairs, revalidate both endpoints on device, and filter every
replay against the current ordinary gap so the published CCD candidate set
remains exactly unchanged.

The current cache is intentionally disabled for the isolated per-env BVH
path.  That path point-swaps one BVH object across stream-local scratch slots;
correct support needs independent candidate generations, references, counts,
and fronts per environment.  It will be implemented only if the combined
family ROI survives the remaining tests.  No 1550-frame or A800 claim is made
yet.

### Swept replay result

The endpoint audit was followed by a real device-resident VF/EE swept-cache
replay.  Each body-pair segment stores only expanded raw candidates, validates
both sweep endpoints on device, and invokes the unchanged exact CCD filter on
every reuse.  Frozen frame-30 DCD/CCD physical and original-encoding multisets
were exact in merged and isolated modes at frames 1, 10, and 30.  The isolated
path deliberately remains cache-disabled because it has no per-slot cache
generation yet.

On the fixed short CCD profile, including device validity, reset, front build,
fresh traversal, and exact replay, the selected CCD-family path changed as
follows:

| cache mask | baseline path | candidate path | delta | decision |
|---|---:|---:|---:|---|
| VF-CCD | 47.069 ms | 40.136 ms | -14.73% | structural only |
| EE-CCD | 47.069 ms | 44.692 ms | -5.05% | reject alone |
| VF+EE-CCD | 47.069 ms | 38.816 ms | -17.53% | opt-in only |

The combined whole-profile kernel sum changed only 176.824 to 175.901 ms
(-0.52%).  Freely evolved contact publication order is still not canonical,
and no end-to-end solver gain was established.  Therefore no swept cache is
part of the winner bundle.

### Reusing traversal fronts across exact refits

`STIFF_BVH_FRONT_REUSE=1` preserves the already-built uniform-body subtree
roots while PLOC topology is only refitted.  DCD and swept boxes change, but
the binary topology and every subtree's body membership do not, so the front
is still exact.  VF front builds fell from 83 to 3 and EE builds from 82 to 2
in the frozen probe.  This saved only about 0.375 ms out of roughly 149 ms
(about 0.25%).  It is a correct shared-runtime primitive, but too small to
include in the performance bundle.

## RBS-style VF query ordering and isolated query domains

The RBS audit's transferable Morton idea is now implemented behind
`STIFF_BVH_QUERY_ORDER`.  Bit 0 sorts VF-DCD query vertices; bit 2 sorts
VF-CCD queries.  Keys are `(30-bit Morton, global vertex id)`, so ties have a
unique deterministic order.  The implementation uses preallocated CUB key,
value, and temporary buffers on the caller's stream: no hot allocation,
readback, or synchronization is introduced, and stream capture remains valid.
EE is unchanged because it already assigns lanes in Morton-sorted leaf order.

For merged one-environment FOLD frame 30 with PLOC++/refit-128, 500 DCD and
500 swept rebuilds gave:

| metric | two-baseline mean | VF-DCD ordered | delta |
|---|---:|---:|---:|
| all CUDA kernels | 3,849.372 ms | 3,290.131 ms | **-14.53%** |
| four query families | 3,711.971 ms | 3,044.489 ms | **-17.98%** |
| VF-DCD | 1,704.666 ms | 1,035.710 ms | **-39.24%** |
| query-key kernels | 0 | 3.935 ms | +3.935 ms |

Sorting VF-CCD alone reduced that kernel by about 14.4%, but radix cost made
the complete fixed workload 0.5--1.0% slower.  Enabling both DCD and CCD was
also worse than DCD alone (3,332.117 versus 3,290.131 ms).  The accepted slice
is therefore mask 1 only; mask 4 and mask 5 are rejected.

The isolated implementation exposed a separate inefficiency: each per-env
face tree queried every scene surface vertex and rejected foreign environments
only at the leaves.  `STIFF_BVH_PERENV_QUERY_SUBSET=1` builds topology-static
lists of global surface-vertex ids for each environment and launches only that
exact subset.  This does not replace any geometric or IPC filter.  A four-env
frame-30 traversal audit changed VF query launches from 220,000 to 55,000
(-75%) while retaining exactly the same overlap counts, primitive tests, DCD
rows, and swept rows.  Node pops fell only 4.74% for DCD and 4.27% for CCD,
showing that the removed foreign queries were cheap.

That distinction matters for performance.  The subset alone was equal to its
adjacent baseline (11.148 versus 11.145 seconds for 500 DCD + 500 CCD
rebuilds).  Sorting the full global list alone was only about 1--2% faster in
clean interleaved runs.  The combination sorts each smaller per-env list and
was repeatably useful: three clean samples were 10.389, 10.451, and 10.403
seconds, against a 11.162-second adjacent-baseline median, or **-6.80%**.
Samples overlapped by an unrelated container CUDA diagnostic were discarded,
not averaged into this result.

Correctness coverage for the combined PLOC/refit/query-order/subset bundle now
includes:

- four-env frozen FOLD frames 1, 10, and 30 in merged and isolated modes, with
  exact geometry and exact DCD/CCD physical and original-encoding multisets;
- the 50-frame merged/isolated/strict candidate gate; strict retained gold
  `0544461bd82123ae` bitwise;
- merged frame-graph normal, rollback, unique retry, retry exhaustion, and
  full-digest gates, plus all three collision-graph scenarios;
- the isolated whole-frame graph gate (19 graph frames after one capture
  fallback), numerical parity, and perturbation isolation.

### Complete FOLD-SHIRT trajectory coverage

`scripts/bvh_fold_long_gate.py` now runs the real bundled 1550-action replay,
writes the terminal checkpoint and vertex array, rejects non-finite state and
CUDA/retry failures, and audits whole-frame-graph coverage.  The first version
of this gate exposed an important false proof: FOLD only declares body groups
when it instantiates at least two environments.  Asking for `isolated` with one
environment printed `per-env machinery disabled, running merged-equivalent`
and produced 1550 fallback frames.  The gate now rejects that configuration
before launch and also rejects a graph run with zero full-graph frames or more
than a 1% fallback budget.

On the RTX 4090, the complete candidate bundle
(`PLOC=3`, refit interval 128, VF-DCD query order, per-env query subset) has the
following long-horizon evidence:

| replay | graph coverage | elapsed | result |
|---|---:|---:|---|
| merged, 1 env, graph off | n/a | 133.1 s (85.5 ms/frame) | 1550/1550, finite |
| merged, 1 env, graph on | 1544 full + 6 boundary fallback | 183.0 s (117.7 ms/frame) | 1550/1550, finite |
| isolated, 4 real groups, graph on | 1545 full + 5 boundary fallback | 674.0 s (434.3 ms/batch, 108.6 ms/env) | 1550/1550, finite |

For the real isolated run, the exact fallback ids were `[0, 1, 10, 12, 13]`:
frame 0 is the documented lazy-workspace warm-up, while device capacity status
was published at frames `[1, 10, 12, 13]`.  Every frame after 13 used the C5
full conditional graph.  This is the intended fixed-topology contract -- the
device detects insufficient capacity, that frame is finished on the audited
boundary fallback, and the next graph is recorded at the grown tier.  There
were no late-episode overflows, exhausted retries, NaNs, or CUDA errors.

The long run crosses repeated 128-construction rebuild/refit generations and
therefore closes the obvious lifetime-risk gap for PLOC topology state.  A
same-binary, graph-off merged matrix then ran every variant twice on the whole
1550-frame episode:

| merged 1-env variant | two runs (ms/frame) | mean | versus baseline |
|---|---:|---:|---:|
| all candidates off | 98.5 / 93.6 | 96.05 | baseline |
| PLOC++ / refit-128 | 88.2 / 87.3 | **87.75** | **-8.64%** |
| VF-DCD query order | 91.1 / 89.6 | **90.35** | **-5.93%** |
| PLOC/refit + query order | 89.1 / 89.6 | 89.35 | -6.98% |

Both individual candidates are therefore real merged-trajectory wins: their
slowest repeat is faster than the fastest baseline.  They do not stack
additively.  The combined bundle is about 1.8% slower than PLOC/refit alone,
because both target overlapping VF traversal work and the changed publication
order also changes the chaotic Newton path.  PLOC/refit alone is the merged
winner; query order remains a useful independent opt-in, not part of the
default merged bundle.

Terminal positions support rather than weaken that interpretation.  The two
baseline runs differ by 0.4442 maximum absolute position and 0.1100 RMS over
the complete chaotic trajectory.  Every candidate terminal state is closer
to at least one baseline than the baselines are to each other (maximum
0.0652--0.3751), while the frozen-frame gates still prove exact DCD/CCD
physical and encoded candidate multisets.

The same two-repeat matrix is complete for four real isolated environments:

| isolated 4-env variant | two runs (ms/batch) | mean | versus baseline |
|---|---:|---:|---:|
| all candidates off | 326.5 / 318.4 | 322.45 | baseline |
| PLOC++ / refit-128 | 326.5 / 311.4 | 318.95 | -1.09% |
| VF-DCD query order + per-env subset | 319.7 / 313.1 | 316.40 | -1.88% |
| PLOC/refit + query order + subset | 315.7 / 303.9 | **309.80** | **-3.92%** |

PLOC/refit alone and query-order/subset alone are smaller than, or overlap,
the baseline's 2.5% run spread; they remain isolated opt-ins rather than
standalone winners.  The combination is different: both repeats beat the
fastest baseline and its mean is 3.92% lower.  It is the measured isolated
bundle, although the modest effect still warrants an A800 repeat before any
default change.  The repeat below shows that this 4090 result does not
generalize to A800 isolated mode.

The terminal-state envelope remains conservative.  Baseline-to-baseline max
position difference is 0.3907 (RMS 0.0291); the two combined runs are only
0.1025--0.1112 from their nearest baseline (RMS 0.0064--0.0068) and 0.1120
from each other.  Frozen frames continue to carry the stronger exact pair-set
proof.  All candidate knobs remain opt-in pending the cross-architecture
decision below.

Whole-frame Graph is a residency/correctness result on this 4090, not the
throughput winner.  The one graph-on sample was 117.7 ms/frame for merged,
versus 89.35 ms for the matching graph-off combined-bundle mean (+31.7%).  For
four isolated environments it was 434.3 versus 309.80 ms/batch (+40.2%).
Those comparisons are not credited as two-repeat performance estimates, but
the gap is too large to describe conditional Graph as an acceleration on this
machine.  The BVH winner claims above therefore use graph-off A/B runs; Graph
coverage separately proves that the candidates remain capture-safe.

### A800 cross-architecture closure (2026-08-06)

The complete campaign was repeated on the managed
`stiffgipc-v083-a800-sm80` worker with one NVIDIA A800-SXM4-80GB.  The tested
Release build used CUDA 12.8, DLTO, Python bindings, no viewer, audit knobs
off, and `CMAKE_CUDA_ARCHITECTURES=80-real`.  `cuobjdump --list-elf` and
`--dump-sass` report only `lto.sm_80.cubin` and `code for sm_80` in both the
core and Python shared objects.  The strict anchor remains bit-identical at
`0544461bd82123ae`.

A clean audit-off build first exposed a real configuration defect hidden by
the incremental audit build: the face/edge body-order metadata pointers were
declared inside `STIFF_BVH_COHERENCE_AUDIT_BUILD` even though production query
ordering uses them.  Moving those two declarations to production state fixes
both clean 4090 and A800 builds; it does not alter any numerical path.  The
managed worker also reports a host PID through NVML that is different from
the container PID.  The long gate therefore gained an explicit, default-off
single-alias claim: it may associate exactly one initially unknown NVML PID
with its just-launched child, but any simultaneous or later unknown CUDA
process still terminates and rejects the sample.  All measurements below used
that idle audit, and no additional CUDA process appeared.

Before timing, the combined candidate passed the 50-frame
merged/isolated/strict gate.  More importantly, the frozen FOLD pair oracle at
frames 1, 10, and 30 passed in both modes.  It compared 26,741--31,826 active
DCD rows and 77,507--105,970 swept CCD rows per checkpoint: geometry,
canonical multisets, and original encodings were all exact between baseline
and candidate.  This closes contact completeness on a real contact-rich
state, rather than relying on the zero-pair strict anchor.

Every graph-off cell below is two independent, GPU-idle-audited 1550-frame
runs.  All 16 runs completed 1550/1550 with finite state, and their logs have
zero quarantine, failure, retry-exhaustion, overflow, illegal-access, NaN, or
CUDA-error matches.

| A800 merged, 1 env | two runs (ms/frame) | mean | versus all-off |
|---|---:|---:|---:|
| all candidates off | 116.6 / 117.2 | 116.90 | baseline |
| PLOC++ / refit-128 | 114.7 / 115.0 | 114.85 | -1.75% |
| VF-DCD query order | 111.4 / 112.2 | 111.80 | -4.36% |
| PLOC/refit + query order | 108.8 / 113.0 | **110.90** | **-5.13%** |

All six candidate samples beat the fastest baseline sample.  The A800 merged
winner is therefore the combined bundle, unlike the 4090 where PLOC/refit
alone wins.  Baseline terminal replicas differ by max 0.413831 and RMS
0.106016; every candidate is closer to one baseline than that envelope (max
at most 0.289620, RMS at most 0.040946).  Exact frozen pair sets remain the
strong correctness oracle because the freely evolved trajectory is chaotic.

| A800 isolated, 4 env | two runs (ms/batch) | mean | versus all-off |
|---|---:|---:|---:|
| all candidates off | 461.0 / 470.2 | **465.60** | baseline |
| PLOC++ / refit-128 | 459.3 / 498.9 | 479.10 | +2.90% |
| VF-DCD query order + per-env subset | 498.5 / 446.4 | 472.45 | +1.47% |
| PLOC/refit + query order + subset | 480.1 / 496.7 | 488.40 | +4.90% |

No isolated candidate earns a speed claim on A800: every two-run mean
regresses and the first two candidates have 8.3--11.0% repeat spread.  One
query-order sample is the fastest isolated sample, but its other repeat is the
slowest, which is precisely why isolated all-off remains the A800 decision.
Candidate terminal deviations are in the same chaotic scale as the baseline
replicas; the frozen-frame exact pair oracle above is the non-statistical
correctness proof.

Complete whole-frame Graph coverage also passed on contact-rich FOLD:

| A800 replay | graph coverage | graph-on | matching graph-off mean | result |
|---|---:|---:|---:|---|
| merged, combined bundle | 1544 full + 6 boundary fallback | 128.5 ms/frame | 110.9 ms/frame | 1550/1550, finite |
| isolated, all-off | 1544 full + 6 boundary fallback | 480.2 ms/batch | 465.6 ms/batch | 1550/1550, finite |

The merged fallback ids are `[0, 1, 6, 11, 13, 14]`, with capacity status on
the latter five.  Isolated uses `[0, 1, 5, 10, 12, 13]`, again with five
capacity frames after warm-up.  Thus both modes return to the graph after
boundary growth and execute every frame after 14 or 13 in the graph.  The
single graph runs are residency/correctness evidence, not speed estimates:
they are respectively 15.87% and 3.14% slower than their matching two-run
graph-off means.

The cross-architecture decision is consequently mode-specific and remains
opt-in: retain PLOC/refit and query order as useful merged candidates, use the
combined bundle for the measured A800 merged workload, and leave A800
isolated all-off.  There is no evidence for one universal default across the
4090/A800 and merged/isolated matrix.

### Clean v0.8.5 comparison

A separate clean worktree at tag `v0.8.5` (`2efaa72`) was built for sm_89 in
Release mode with the viewer disabled and Python bindings enabled.  The
historical tree has no `STIFFGIPC_DLTO` option; the campaign binary uses its
normal DLTO-on Release configuration.  This is therefore a product-version
comparison, not a claim that every cross-version delta comes from BVH alone.
The causal BVH numbers remain the same-current-binary all-off/candidate A/B
matrices above.

Both historical variants completed two uncontaminated 1550-frame runs:

| mode | v0.8.5 two runs | v0.8.5 mean | current all-off | current winner |
|---|---:|---:|---:|---:|
| merged, 1 env | 96.8 / 104.5 ms/frame | 100.65 | 96.05 (-4.57%) | PLOC/refit 87.75 (**-12.82%**) |
| isolated, 4 env | 346.4 / 348.9 ms/batch | 347.65 | 322.45 (-7.25%) | combined 309.80 (**-10.89%**) |

All four historical runs completed 1550/1550 with finite terminal vertices.
The v0.8.5 isolated repeats differ by max 0.3756 and RMS 0.0239 at the
terminal state, inside the same chaotic long-horizon scale already observed
for current baselines.  Cross-version terminal hashes are deliberately not a
correctness oracle; exact frozen-frame pair multisets, strict gold, and the
same-binary terminal envelopes carry that proof.

## Current conclusions

- Better topology is real: 13.4--18.0% fewer node pops is available without
  changing a single exact primitive test or pair encoding.
- More Morton bits are not the answer; the prior corrected 42-bit probe was
  essentially flat.  PLOC/treelets/SAH rotations change topology and are the
  relevant family.
- Lightweight rotations must be selective.  Whole-tree/all-family application
  loses to its own build cost; face-DCD phase 1 is the only measured positive
  slice so far.
- Full PLOC/PLOC++ remains too expensive.  Hierarchical PLOC++ becomes a real
  candidate when amortized with exact topology refit; interval 128 measured
  about 5.9% observed and at least 2.7% directly attributable whole-kernel
  improvement on deterministic 30-frame FOLD, with exact state and pairs.
- Same-topology software BVH8 is implemented.  Whole-pipeline and PLOC+BVH8
  variants lose, while largest-box-first face-DCD-only wide traversal is a
  measured 4.7% win in the query-heavy frozen test and remains a candidate.
- Conservative body-pair/family coherence and a GPU-resident VF-DCD raw
  candidate cache/shared body front are implemented.  VF is a small positive
  slice in real evolution (about 8.4% of its own path, under 1% whole-kernel
  time).  It has a much larger frozen gain when stacked with PLOC/refit, but
  changes contact publication order and increased Newton work in every tested
  VF-only free run, so it remains opt-in and outside the winner bundle.
  EE-DCD is implemented and rejected alone (+60% family-path cost); corrected
  VF+EE is a frozen -11.9% DCD-path experiment but has no credited free-run
  win.  Swept endpoint coherence led to a filtered CCD replay prototype, but
  that implementation saves only 0.52% of the short whole profile, so it
  remains outside the bundle.  Isolated
  per-env cache generations remain pending.
- The RBS-style query-order idea is useful specifically for VF-DCD: about
  39.2% off that merged kernel and 14.5% off the fixed merged workload.
  VF-CCD sorting loses after radix overhead.  In isolated mode, an exact
  per-env query subset has no standalone wall benefit, but makes query sorting
  cheap enough that the combination measured about 6.8% end-to-end on the
  fixed four-env workload.  This is the second winner candidate after
  PLOC++/refit-128.  Both survive complete 1550-frame 4090 and A800
  trajectories and exact frozen contact gates.  On A800 merged, their combined
  bundle wins by 5.13%; on A800 isolated every candidate mean regresses, so
  all-off remains the decision.  The architecture/mode disagreement rules out
  a universal default and keeps the candidates explicit opt-ins.

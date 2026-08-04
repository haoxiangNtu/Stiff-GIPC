# Advanced BVH validation: tree quality, wide traversal, and coherence

Date: 2026-08-04

Branch: `codex/bvh-advanced-validation`

Base: `96b52fb` (`codex/bvh-accel-validation`)

Worktree: `/home/ps/Downloads/Stiff-GIPC-bvh-advanced-validation`

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
apparent gain is deliberately not credited.  Repeated deterministic long-run
and A800 measurements are still required before changing any default.

The repository's 22-segment `verify_gates.sh` passed once with all new knobs
unset and once with face-DCD/phase-1 enabled.  Both runs kept strict gold
`0544461bd82123ae` and passed towel, quarantine, MAS, checkpoint, physics,
frame/episode/articulated-RL graphs, C4 collision graph, isolated whole-frame
graph, GPU-native RL, knob registry, and the FOLD smoke test.  The candidate
kernels are therefore demonstrated capture-safe on the existing Phase-C/D
graph gates; this does not replace the still-missing A800 performance run.

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

## Current conclusions

- Better topology is real: 13.4--18.0% fewer node pops is available without
  changing a single exact primitive test or pair encoding.
- More Morton bits are not the answer; the prior corrected 42-bit probe was
  essentially flat.  PLOC/treelets/SAH rotations change topology and are the
  relevant family.
- Lightweight rotations must be selective.  Whole-tree/all-family application
  loses to its own build cost; face-DCD phase 1 is the only measured positive
  slice so far.
- Full and hierarchical GPU PLOC/PLOC++ recover useful tree quality but lose
  when rebuilt for every query; topology amortization/refit is the remaining
  condition under which they may become profitable.
- BVH8 is not yet implemented, and no speedup is claimed for it.
- Conservative body-pair/family coherence is now measured, but the actual raw
  candidate cache and unified VF/EE/CCD broad phase remain pending.  The
  dominant cloth pairs invalidate far more often than static ABD pairs, so a
  useful cache must segment both validity and stored candidate work.

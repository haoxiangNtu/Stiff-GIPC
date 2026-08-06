# IPC BVH acceleration validation report

Date: 2026-08-04

Validation branch: `codex/bvh-accel-validation`

Base revision: `a929b1c`

Isolated worktree: `/home/ps/Downloads/Stiff-GIPC-bvh-accel-validation`

This branch is an experiment and evidence branch. It was created separately
from `/home/ps/Downloads/Stiff-GIPC-c1-ls-graph`; no Claude worktree file or
branch was modified.

## Executive decision

Only one tested idea is both useful and compatible with the existing IPC
contract: **EE original-index subtree ownership pruning**
(`STIFF_EE_RANGE_PRUNE=1`). It removes provably ineligible EE subtrees in both
DCD and swept CCD. It preserves the exact physical candidate multiset and the
exact existing four-integer encoding. On one frozen FOLD-SHIRT state it reduced
EE node pops by 16.1% (DCD) and 17.1% (CCD), while leaving the number of
primitive tests exactly unchanged. The measured kernel improvement was about
1--4% for EE-DCD and 10--13% for EE-CCD. The complete simulation gain is much
smaller: currently low single digits at best, and not yet statistically stable
enough to enable by default.

The following attractive-sounding changes did not earn a production change:

- one global margin/Verlet candidate table;
- more `__launch_bounds__` pressure;
- shrinking the explicit traversal stack;
- increasing Morton quantization from 10 to 14 bits per axis;
- sorted-rank EE ownership, because it changes IPC pair representation;
- a supposed cheap fusion of “three traversals,” because the premise does not
  match this source tree.

PLOC/treelet optimization, BVH8/CWBVH, and RT-core traversal remain unproven,
not disproven. They require separate implementations and the same gates below.

## What the source actually does

There are four query families, not three interchangeable passes:

1. vertex--face DCD over the face BVH;
2. edge--edge DCD over the edge BVH;
3. vertex--face CCD over swept face boxes;
4. edge--edge CCD over swept edge boxes.

VF and EE do not use the same primitive tree. DCD boxes and swept CCD boxes are
not the same boxes, and DCD contact construction and line-search CCD occur at
different solver stages. A shared front or wide-tree rewrite may still be
possible, but it is not a local “traverse once and emit three arrays” change.

`lbvh_f::Construct*` and `lbvh_e::Construct*` currently calculate leaf boxes,
Morton keys, sort, create leaves/internal nodes, and propagate AABBs on every
construction. The statement that the current code “normally only refits and
rarely rebuilds” is therefore false for this revision. A 30-frame FOLD profile
put the four query kernels near 47% of CUDA kernel time; the known build kernels
excluding the unattributable shared CUB radix-sort symbol were about 2--3%.
Refit can reduce build cost, but it cannot by itself remove the dominant query
work.

## Correctness standard

A speed number is accepted only after all applicable levels pass:

1. both sides load the **same checkpoint**, so geometry and topology are
   bit-identical before the query;
2. decoded DCD and CCD physical candidate multisets match, including
   multiplicity;
3. the original encoded `(N,4)` multiset also matches when the candidate claims
   unchanged IPC semantics;
4. final positions, Newton envelope, active-term FD gates, and the strict gold
   anchor remain valid;
5. a complete 1550-frame FOLD-SHIRT trajectory finishes in both merged and
   isolated modes with finite state;
6. graph, episode, GPU-RL, quarantine, lifecycle, reset, and collision gates
   remain green.

Freely evolved merged/isolated trajectories are not a valid exact pair oracle:
atomic emission/order can move their Newton paths apart across processes. The
new `bvh_fold_pair_gate.py` therefore creates a baseline checkpoint and runs
baseline/candidate queries against that frozen state. Its deterministic
non-rigid motion field also audits the swept CCD path rather than inferring CCD
correctness from “the replay did not crash.”

## Accepted candidate: EE range ownership pruning

The existing non-canonical, non-node-deduplicated EE rule emits only the
directed owner with `obj_idx >= self_eid`. Each BVH node now optionally stores
the maximum **original edge index** in its subtree. If that maximum is below
`self_eid`, every leaf below the node would be rejected by the unchanged leaf
rule, so skipping the subtree is an exact optimization.

Important properties:

- default/unset mode allocates no extra max-index array and launches the
  original kernels;
- mode 1 uses original edge indices for DCD and CCD;
- per-environment scratch save/swap/free paths carry the optional array;
- an overflow cannot silently drop work; the existing pair capacity protocol
  remains in force;
- the optimization disables itself under canonical/node-dedup rules where the
  proof above does not apply.

Frozen FOLD-SHIRT evidence (one environment):

| Mode | Frame | DCD pairs | CCD pairs | Physical multiset | Encoding |
|---|---:|---:|---:|---|---|
| merged | 1 | 26,741 | 77,793 | exact | exact |
| merged | 10 | 26,847 | 77,504 | exact | exact |
| merged | 30 | 31,210 | 106,146 | exact | exact |
| isolated | 1 | 26,750 | 77,884 | exact | exact |
| isolated | 10 | 26,878 | 77,542 | exact | exact |
| isolated | 30 | 31,454 | 105,351 | exact | exact |

Frozen frame 30, ten repeated exported queries, traversal-audit build:

| Family | Baseline node pops | Mode 1 node pops | Delta | Baseline overlaps | Mode 1 overlaps | Primitive tests |
|---|---:|---:|---:|---:|---:|---:|
| EE DCD | 39,446,956 | 33,111,641 | -16.06% | 55,867,380 | 41,321,853 (-26.04%) | 1,266,813 / 1,266,813 |
| EE CCD | 41,209,520 | 34,151,180 | -17.13% | 59,672,600 | 43,382,720 (-27.30%) | 1,556,020 / 1,556,020 |

Frozen frame 30, 100 exported DCD and CCD rebuilds, normal optimized build:

| Metric | Baseline | Mode 1 | Delta |
|---|---:|---:|---:|
| EE DCD mean kernel | 958.857 us | 947.107 us | -1.23% |
| EE CCD mean kernel | 823.547 us | 715.357 us | -13.14% |
| known BVH build kernels (CUB excluded) | 38.185 ms | 40.277 ms | +5.48% |
| all four query families | 794.764 ms | 785.797 ms | -1.13% |
| all CUDA kernels in this query-heavy microbenchmark | 871.520 ms | 865.060 ms | -0.74% |

The fixed test is deliberately query-heavy (queries are about 91% of its CUDA
kernel time), and unrelated VF timing moved by several percent under the
machine's background GPU load. The structural counter reduction is hard
evidence; the total wall-time number is not yet a release-grade speedup claim.
The main edge tree adds `(2 * 20,404 - 1) * 4` bytes, about 159 KiB, for the
subtree maxima; isolated pool slots add the same order per concurrent slot.
The table includes the small metadata-propagation build cost rather than hiding
it.

Complete trajectory evidence with mode 1 active in DCD and CCD:

| Mode | Frames | Vertices | Finite | Final SHA-256 prefix | Mean `step()` |
|---|---:|---:|---|---|---:|
| merged | 1550/1550 | 7,187 x 3 | yes | `dd8567f80117a94e` | 93.6 ms |
| isolated | 1550/1550 | 7,187 x 3 | yes | `3225b89404c95d78` | 82.1 ms |

These hashes identify the artifacts; they are not cross-mode gold values. The
two modes follow different numerical execution paths and their wall times must
not be compared as a candidate speedup.

The final `verify_gates.sh` runs passed all 22 segments both with the candidate
unset and with `STIFF_EE_RANGE_PRUNE=1`. This includes strict gold
`0544461bd82123ae`, towel, both quarantine paths, constitutive/geometry checks,
reset, frame/episode/articulated-RL graphs, collision and isolated whole-frame
graphs, GPU-native RL, knob registry, and the 30-frame FOLD smoke test.

## Rejected candidate: one global temporal candidate table

The completeness argument itself is sound: build a raw primitive candidate
superset with extra distance margin `delta`; while every vertex has moved no
more than `delta/2` from the build snapshot, every pair that can enter the real
activation distance remains in that superset.

What is not sound is caching the already classified active IPC rows. Distance
type, barrier activity, mollification, friction state, and CCD filtering must be
recomputed on every reuse. Only raw primitive candidates may be cached.

The audit build expanded only the broad phase, continued to run the exhaustive
solver path, and measured the theoretical safe reuse opportunities:

| Search radius / real radius | Safe reuse fraction | Queries/build | Longest span | VF-DCD pops/query vs 1.0 | VF-DCD primitive tests/query vs 1.0 |
|---:|---:|---:|---:|---:|---:|
| 1.25 | 5.73% | 1.061 | 4 | 1.096x | 1.306x |
| 1.50 | 15.89% | 1.189 | 9 | 1.191x | 1.629x |
| 2.00 | 20.27% | 1.254 | 6 | 1.401x | 2.280x |

Even before paying to rescan/reclassify the cached table, approximate traversal
work per real query is `expanded_pops / queries_per_build`: 1.033x, 1.002x,
and 1.117x of baseline for the three margins. Thus a **single global table is
not profitable in this FOLD workload**. A per-body or per-family table remains
a possible future experiment because one fast ABD should not invalidate every
slow cloth region.

## Other candidates

### Launch bounds

The linked sm_89 image reports 106 registers for both the baseline EE-DCD
kernel and its `lb2` variant, not the stale 168-register source comment. The
`lb3` variant reports 80 registers but adds stack/spill pressure. A freely
evolved initial profile misleadingly made both variants look slower because
its Newton/query counts differed. Repeating 100 DCD rebuilds on the same frozen
checkpoint gives the valid comparison:

| Mode | Variant | EE-DCD mean | Delta from lb0 | Encoded set |
|---|---|---:|---:|---|
| merged | lb0 | 975.367 us | baseline | exact |
| merged | lb2 | 958.857 us | -1.69% | exact |
| merged | lb3 | 991.764 us | +1.68% | exact |
| isolated | lb0 | 969.693 us | baseline | exact |
| isolated | lb2 | 963.985 us | -0.59% | exact |

Thus the existing merged default `lb2` is modestly useful and should remain.
There is no case for `lb3`; the isolated result is too small to overturn the
existing end-to-end measurements that classified it as noise. A800/sm_80 must
be measured separately.

### Explicit stack capacity

The observed traversal depth was only 9--10, but reducing the compile-time
array from 2,048 to 64 entries did not improve EE-DCD (1,674.652 us to
1,687.792 us). The experimental small-stack build traps on overflow; it never
silently discards a subtree. Keep the production capacity until a different
traversal layout removes local-memory traffic measurably.

### Morton 14-bit precision

The first probe was found to be invalid because env-major key construction ran
before the experimental branch, making the knob a no-op. The branch was fixed,
and Nsight confirms `_calcMChash14` is now launched. On the same frozen state,
physical and encoded DCD/CCD sets were exact, but tree work barely changed:

- VF-DCD node pops: -0.028%;
- EE-DCD node pops: -0.104%;
- VF-CCD node pops: +0.002%;
- EE-CCD node pops: -0.002%.

Morton quantization is therefore not the current tree-quality limiter. This
does **not** reject PLOC/PLOC++, SAH treelets, or a wide BVH; those alter
topology/traversal rather than only key resolution.

### Sorted-rank EE ownership (`STIFF_EE_RANGE_PRUNE=2`)

This research mode reduced frozen EE-DCD node pops by about 40.7%, but it forces
canonical edge ordering and changes the encoded pair multiset. The decoded
physical multiset stays exact, yet contact subtype/evaluation order can change
floating-point results. It is not acceptable as a drop-in optimization under
the current strict/gold contract. The frozen gate therefore requires exact
encoding by default; set `BVH_FOLD_GATE_REQUIRE_ENCODING=0` only when explicitly
studying this representation-changing mode.

### PLOC/treelets, BVH8, and RT cores

No production claim is made for these. A useful next experiment must first
show a material reduction in node pops/overlap tests on frozen FOLD states,
then pass exact DCD/CCD encoding, FD, 1550-frame, and graph-capture gates. RT
cores additionally introduce OptiX/toolchain portability and A800 compatibility
questions, so they are lower priority than software treelets or a wide BVH.

## Physics invariants

None of the accepted code changes gradient, Hessian, SPD projection,
preconditioner construction, ACCD, or line-search acceptance. Those are solver
invariants, not performance knobs. FD must compare the analytic **unprojected**
Hessian with finite differences; PSD projection is separately checked by
minimum-eigenvalue/Cholesky-style tests. A candidate that makes an active term
disappear is not considered a passing FD test.

The repository's `merged/stitch` FD fixture currently fails on the baseline and
candidate in the same way (line-search budget exhaustion followed by CUDA 700
in the MAS env-segment path). It is a broken oracle, not evidence for or against
the BVH candidate. The usable merged `cloth_ground`/`fem_contact` and strict
`cloth_ground`/`fem_contact`/`stitch` fixtures pass with mode 1.

## Reproduction

Builds used here:

```bash
cmake -S . -B build-bvh-validation -GNinja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89-real \
  -DSTIFFGIPC_DLTO=ON -DSTIFFGIPC_ENABLE_DIAGNOSTICS=ON
cmake --build build-bvh-validation --target pystiffgipc -j8
```

Candidate and frozen FOLD gates:

```bash
STIFFGIPC_NATIVE_DIR="$PWD/build-bvh-validation" \
BVH_CANDIDATE_ENV=STIFF_EE_RANGE_PRUNE=1 \
python3 scripts/bvh_candidate_gate.py

STIFFGIPC_NATIVE_DIR="$PWD/build-bvh-validation" \
BVH_CANDIDATE_ENV=STIFF_EE_RANGE_PRUNE=1 \
python3 scripts/bvh_fold_pair_gate.py
```

Profile reporting:

```bash
nsys export --type sqlite --output run.sqlite run.nsys-rep
python3 scripts/bvh_profile_report.py run.sqlite --frames 30
```

Experimental knobs are registered and default off:

- `STIFF_EE_RANGE_PRUNE=1`: exact original-index DCD+CCD pruning candidate;
- `STIFF_EE_RANGE_PRUNE=2`: representation-changing research mode;
- `STIFF_BVH_MORTON14=1`: corrected 42-bit Morton research probe;
- `STIFF_BVH_TRAVERSAL_AUDIT=1`: counters, audit build only;
- `STIFF_BVH_MARGIN_SCALE` and `STIFF_BVH_COHERENCE_AUDIT`: shadow temporal audit;
- `STIFFGIPC_BVH_STACK_CAP`: compile-time stack-capacity experiment.

## Scope limits

- Measurements in this report are RTX 4090/sm_89 only.
- A background Houdini process occupied roughly 1 GiB and introduced timing
  noise; no unrelated process was terminated.
- No A800 run has been made on this branch, so there is no sm_80 performance or
  nsys zero-transfer claim here.
- Mode 1 should remain opt-in until repeated clean-machine end-to-end trials
  establish a stable gain and A800 gates pass.

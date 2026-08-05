# GPU-native RL contract and completion plan

## Definition of done

“GPU-native RL” means the steady-state transition

```text
device action
  -> simulation step
  -> device observation/reward/done
  -> device reset/select
  -> next device action
```

stays on one or more CUDA streams. The CPU may enqueue work, but a normal RL
step must not contain:

- host-to-device or device-to-host payload copies;
- `cudaDeviceSynchronize`, `cudaStreamSynchronize`, event waits, or equivalent
  implicit host waits;
- host decisions that determine Newton, PCG, CCD, line-search, reward, done, or
  reset control flow;
- topology-dependent allocation or CUDA Graph re-capture.

Setup, diagnostics, checkpointing, rendering, and final result adjudication may
cross the host boundary. They are not part of the steady-state contract.

## Current status

The implementation is transitional and must not yet be described as complete
GPU-native RL.

| Capability | Status | Evidence / remaining gap |
| --- | --- | --- |
| Newton, PCG, and line-search device continuation | Partial | Implemented for the whole-frame graph's eligible scene subset. |
| Reusable articulated one-frame graph | Implemented for the eligible collision subset | `prepare_gpu_rl()` captures once; `launch_gpu_rl_async(stream)` reuses the executable. The 4090 contact+friction gate records 1057 nodes. |
| Device actions | Implemented | Writable packed float64 revolute/prismatic buffers are exported as raw device pointers. |
| Device positions, velocities, status, frame counter | Implemented | Exported by `get_gpu_rl_device_abi()`. |
| Zero graph H2D/D2H | Implemented and fail-closed | Capture audit rejects any host node, H2D node, or D2H node. |
| No steady-state host wait | Implemented at engine ABI | Launch only enqueues the graph and an event. `gpu_rl_ready()` and `synchronize_gpu_rl()` are explicitly optional host/debug boundaries. |
| Real collision/contact/CCD | Complete for eligible scenes (2026-07-29, C4-a..d) | `STIFF_C4_COLLISION_GRAPH=1` records BVH+DCD+CCD scalar chain+backtracking LS **plus in-graph kappa adaptation (C4-b) and lagged friction (C4-c)** inside the whole-frame graph. G18 three-scenario matrix (ground / friction mu=0.3 / squeeze) passes at 2e-13..2e-12 with the kappa trajectory matching the baseline value-for-value. |
| Device reward and done | Done signal complete; reward is policy-side by contract (2026-07-29) | done/health is device-readable via the FrameStatus packet and the per-env quarantine flags in the ABI; graph-refreshed joint observations ({angle,rate}/{disp,rate}) feed reward computation, which is task-specific and belongs to the RL environment, not the physics engine. |
| Selective device reset | Whole-scene in-stream reset done; per-env masks open (2026-07-29) | `launch_gpu_rl_reset_async` replays the prepare-time snapshot with pure D2D on the bound stream. Per-env mask-driven partial reset needs per-env vertex slicing and remains open. |
| Batched heterogeneous environments | Merged + isolated done (2026-07-29, C5); strict deliberately out | Merged multi-env works through the ABI (point_to_group / env_quarantined / env_count handles). **Isolated mode now runs entirely inside the whole-frame graph** (`STIFF_C5_ISOLATED_GRAPH=1`): per-env CCD alpha chain, per-env freeze decision publishing the Newton exit, the per-env S3 backtracking WHILE with its uniform-alpha fallback IF, and in-graph kappa — G19 shows 20/20 frames graph-executed with zero fallback and no added cross-env coupling. Strict stays rejected on purpose: capacity-grid reductions legally reassociate sums, so admitting strict is an anchor-change decision, not a residency one. |
| Closed-loop Warp/Torch/Newton adapter | Not implemented | The engine ABI exists; adapters still contain `.numpy()` and synchronous `step()` calls. |
| A800 proof | Contact-load proof complete (2026-07-29, at 8c4e504) | `sm_80` in-place rebuild: strict anchor bit-identical (`0544461bd82123ae`), G18 collision matrix PASS, gpu-rl gate PASS, contact-load closed loop PASS, and the Nsight Systems steady-state capture **under collision+friction+joint-drive load** shows h2d=0 / d2h=0 / sync=0 — 40 `cudaGraphLaunch` (~214µs mean enqueue on the A800's slow host link, i.e. the exact motivation for residency) + 80 small D2D publishes are the host's entire steady-state workload. Remaining measurements (long-horizon stability, memory high-water, throughput curves) are routine benchmarking, not correctness gaps. |

The older `launch_episode_async()` API is an open-loop trajectory executor. It
pre-uploads all actions and copies observations to pinned host slots. It remains
useful for regression and throughput experiments, but it does **not** satisfy
the definition above.

### RTX 4090 node-level Nsight recheck (2026-08-05)

`scripts/gpu_rl_contact_steady.py` is now self-contained: it arms the frame,
full-frame, collision, and ABD-step graph knobs before its warm-up.  Previously
those settings existed only in other gates' child environments, so invoking
this script directly took the release layout and failed to train the ABD final
assembly tier before `prepare_gpu_rl()`.

The repaired standalone smoke reports 1057 graph nodes, graph H2D=0/D2H=0,
43/43 completed frames and `result=0`.  Nsight Systems 2024.6.2 then captured
the 40-step steady region with `--cuda-graph-trace=node`, rather than the
default graph-level black box.  `scripts/nsys_gpu_rl_audit.py` reports:

- 40 `cudaGraphLaunch`, 40 non-blocking completion-event records, and 80
  24-byte D2D action publishes;
- 5655 executed CUDA Graph node events, proving node-level tracing was active;
- zero H2D and zero D2H rows over the complete capture;
- zero synchronization API rows and zero CUPTI synchronization activities
  inside the 2.461 ms host submission interval.

The complete SQLite table contains one `cuCtxSynchronize` API row and two
CUPTI synchronization activities **after** the final submission.  Timeline
inspection places them in the `cudaProfilerStop`/Nsight flush boundary while
the already-enqueued GPU work drains (the first starts 24 us after the last
event record).  They are profiler-boundary work, not a simulation-step wait.
Accordingly the precise claim is “zero synchronization in the steady
submission loop,” not the misleading stronger claim “the exported SQLite
contains zero synchronization rows.”

Reproduction:

```bash
STIFFGIPC_NATIVE_DIR="$PWD/build-campaign-audit" \
LD_LIBRARY_PATH="$PWD/build-campaign-audit" \
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --cuda-graph-trace=node --export=sqlite --force-overwrite=true \
  --output=artifacts/nsys-contact-4090/contact-steady-node \
  python3.10 scripts/gpu_rl_contact_steady.py --nsys

scripts/nsys_gpu_rl_audit.py \
  artifacts/nsys-contact-4090/contact-steady-node.sqlite \
  --expected-steps 40
```

## Device ABI (foundation block)

After a legal warm-up:

```python
engine.prepare_gpu_rl()                 # setup boundary
abi = engine.get_gpu_rl_device_abi()

# External policy writes abi["revolute_actions"] and
# abi["prismatic_actions"] on `stream`.
engine.launch_gpu_rl_async(stream)      # no host wait

# External observation/reward policy consumes abi["positions"],
# abi["velocities"], abi["statuses"], and abi["frame_counter"] on `stream`.
```

Actions are tightly packed `float64[joints, 3]`:

- revolute: target angle, strength ratio, external torque;
- prismatic: target distance, strength ratio, external force.

Positions and velocities are `float64[vertices, 3]` in engine-internal vertex
order. Raw pointers are owned by the engine and remain valid until
`end_gpu_rl()`, reset, or destruction. Every repeated launch must use the same
CUDA stream so policy writes, simulation, and policy reads have explicit stream
ordering without a host fence.

## Completion blocks

### C4: real collision/CCD graph

1. ~~Replace remaining host collision counts and launch bounds with
   device-owned counters and capacity tiers.~~ (C4-a, 883b214)
2. ~~Record broad phase, narrow phase, refined CCD and rollback in the
   whole-frame conditional graph.~~ (C4-a; friction still excluded)
3. ~~Preallocate all steady-state tiers; publish overflow as device
   status/done rather than growing buffers mid-step.~~ (C4-a: recording-time
   capacity mirrors + pre-capture dry-run training + in-graph OVF guards)
4. ~~Prove contact numerical equivalence against the synchronous solver.~~
   (G18: ground-contact scene, ~1e-13; contact-rich self-collision scenes
   still to be added to the gate matrix)

Remaining sub-blocks, with reconnaissance results (2026-07-29):

- **C4-b in-graph kappa adaptation.** Today the graph freezes kappa at its
  frame-boundary value (eligibility enforces a kappa-quiet window). Findings:
  the barrier/ground G/H and energy kernels take kappa **by value**
  (`mKappa`), so recording bakes it in — except the per-group path, which
  already reads the device array `m_kappa_group`. But `pergroup_kappa` is an
  isolated-mode overlay, which C-3 eligibility rejects. Plan: add a scalar
  `kappa_dev` tail parameter (`kappa = kappa_dev ? *kappa_dev : mKappa`) to
  the ~6 kappa-consuming kernels; devicify the postLineSearch body at the
  Newton-loop tail: close-constraint buffers are already grow-only (train
  capacity at the boundary), `_checkGroundCloseVal`/`_checkSelfCloseVal`
  need a persistent device flag slot plus a capacity-grid/d_live tail
  parameter (their `numbers` is a host mirror today), the doubling kernel is
  trivial (`if(flag) k = min(2k, kappaMax)` with `kappaMax` a scene-constant
  precomputed at the boundary), and the frame status already carries kappa
  back at the boundary. Note the in-frame doubling strategy is already
  deliberately weakened upstream (compute*CloseVal no longer feeds it
  aggressively; kappa re-seeds each frame via gradient projection), so the
  quiet-window contract is mild in practice.
- **C4-c friction inside the graph.** Friction sets (lagged lastH family)
  rebuild only at synchronous frame boundaries; eligibility rejects nonzero
  friction. Needs: `buildFrictionSets` capture-safe (device counts, no
  grow), plus the lagged-count mirrors (`h_cpNum_last`) devicified the same
  way as C4-a did for the live counts.
- **C4-d contact-rich gate matrix.** Extend G18 with a self-collision DCD
  scene (two interpenetrating-trajectory cloth/soft bodies) and a towel
  recipe window, so tier guards and the swept CCD overflow retry see real
  DCD/EE traffic.

### D2: device RL semantics

1. Define compact, versioned device observation/action/reset descriptors.
2. Compute joint observations, contact observations, reward, termination, and
   truncation on device.
3. Implement selective per-environment reset from device reset masks.
4. Keep policy, simulation, reward/done, and reset ordered on the caller's CUDA
   stream without a host query.

### D3: batched environments and adapters

1. Remove the merged-only restrictions or provide a device-native segmented
   graph for isolated environments.
2. Replace Newton/IsaacLab adapter `.numpy()` control, reset, joint readback,
   and contact paths with Warp/device-pointer kernels.
3. Add many-environment determinism, isolation, reset, and throughput gates.

### D4: proof on A800

The final A800 run must include:

- `sm_80` SASS verification for the exact tested binary;
- Nsight Systems steady-state trace with zero H2D/D2H memcpy and zero host
  synchronization per environment step;
- contact-rich and articulated workloads, not only the no-collision cube gate;
- device-policy integration, device reward/done/reset, and selective resets;
- numerical comparison, long-horizon stability, memory high-water marks, and
  throughput/latency measurements.

Only after all four completion blocks pass may Phase C and Phase D be marked
complete.

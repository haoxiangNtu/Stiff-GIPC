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
| Reusable articulated one-frame graph | Implemented for the current no-collision gate | `prepare_gpu_rl()` captures once; `launch_gpu_rl_async(stream)` reuses the executable. |
| Device actions | Implemented | Writable packed float64 revolute/prismatic buffers are exported as raw device pointers. |
| Device positions, velocities, status, frame counter | Implemented | Exported by `get_gpu_rl_device_abi()`. |
| Zero graph H2D/D2H | Implemented and fail-closed | Capture audit rejects any host node, H2D node, or D2H node. |
| No steady-state host wait | Implemented at engine ABI | Launch only enqueues the graph and an event. `gpu_rl_ready()` and `synchronize_gpu_rl()` are explicitly optional host/debug boundaries. |
| Real collision/contact/CCD | C4-a landed (2026-07-29) | `STIFF_C4_COLLISION_GRAPH=1` records BVH+DCD+CCD scalar chain+backtracking LS inside the whole-frame graph (G18 gate: 20 contact frames, zero fallback, ~1e-13 parity). Contract limits remain: kappa/close-set frozen at frame-boundary values, zero friction only — C4-b (in-graph kappa) and C4-c (friction set rebuild) still open. |
| Device reward and done | Not implemented | Consumers can read device state/status, but reward/done kernels and stable ABI are still required. |
| Selective device reset | Not implemented | Reset/teleport and episode bookkeeping still use host paths in the RL adapters. |
| Batched heterogeneous environments | Not implemented | Current graph requires merged mode and rejects per-env/strict overlays. |
| Closed-loop Warp/Torch/Newton adapter | Not implemented | The engine ABI exists; adapters still contain `.numpy()` and synchronous `step()` calls. |
| A800 proof | Partial (2026-07-29) | `sm_80` in-place build of a2c8b33: strict anchor bit-identical (`0544461bd82123ae`), all five graph gates PASS, gpu-rl parity bitwise (error 0.0), and the Nsight Systems steady-state capture shows h2d=0 / d2h=0 / sync=0 with 40 `cudaGraphLaunch` + 40 D2D publishes for 40 steps. Remaining for full D4: contact-rich workloads, device-policy integration, reward/done/reset on device, long-horizon/throughput/memory measurements. |

The older `launch_episode_async()` API is an open-loop trajectory executor. It
pre-uploads all actions and copies observations to pinned host slots. It remains
useful for regression and throughput experiments, but it does **not** satisfy
the definition above.

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

// ============================================================================
// core/frame_pipeline.h — THE in-frame stage-order document (v0.8.6 Phase 2d).
//
// Doc-only header: no code, no macros. This is the single authoritative
// statement of how a frame walks. If you reorder stages in IPC_Solver /
// solve_subIP / lineSearch (core/ipc_solver.inl), update THIS file in the
// SAME commit — a stale pipeline doc is worse than none.
//
// ─── One frame — GIPC::IPC_Solver ───────────────────────────────────────────
//   0. boundary/animation move-dir      updateBoundaryMoveDir        (gipc_modules/14 + 08)
//   1. broadphase rebuild               buildBVH / buildBVH_FULLCCD  (gipc_modules/12 → mlbvh_modules)
//   2. contact pair emission            buildCP / buildFullCP        (gipc_modules/10;
//         growth/redo + DCD≤CCD invariant owned by contact/pair_buffers.cuh)
//   3. kappa init                       upperBoundKappa / suggestKappa / initKappa (gipc_modules/13)
//   4. lagged friction sets             ensure_frictionBuffers + buildFrictionSets (gipc_modules/09 + 14)
//   5. frame-start iron-law probe       quarantineGroundInfeasibleAtFrameStart (multienv/isolation.cuh)
//         ORDER CONTRACT: must run BEFORE any CCD-alpha kernel of the frame —
//         a mid-gap teleport otherwise hits a CCD fail-fast before detection.
//   6. Newton loop                      solve_subIP (below)
//         re-runs with fresh friction sets while the outer kappa/friction
//         lag loop demands it
//   7. state advance                    updateVelocities + computeXTilta (gipc_modules/14)
//
// ─── One Newton iteration — GIPC::solve_subIP ───────────────────────────────
//   a. frame-start triplet grow         computeGradientAndHessian preamble (gipc_modules/13):
//         ensure_capacity_discard is legal ONLY there (offsets just reset,
//         no live matrix; conv_pred = 27*prev/10). Everywhere else growth
//         must be ensure_capacity_preserve (towel-strict root cause).
//   b. un-freeze recheck                per-env hmx vs threshold, BEFORE the solve
//   c. gradient + Hessian assembly      computeGradientAndHessian (gipc_modules/13;
//         fused barrier assembly in 04, kinetic/soft/ground in 06)
//   d. convergence decide               _newton_convergence_decide (device) / per-env
//         freeze mask + DECOUPLE_THRESH all-envs latch (gipc_modules/11 + core)
//   e. solve                            calculateMovingDirection (gipc_modules/14 →
//         MAS or diag preconditioner + PCG, linear_system/)
//   f. post-PCG quarantine inertness    NaN direction scan + zero-direction
//         (multienv/isolation.cuh; keeps quarantined envs bit-frozen)
//   g. CCD alpha chain                  cfl / ground / self largestFeasibleStepSize
//         (reduction kernels gipc_modules/07, host wrappers 09/10;
//         ground honors _ground_skip_body of quarantined envs)
//   h. per-env S1 alpha block           per-env alpha compute/halve (gipc_modules/11;
//         strict layout-fixed policy)
//   i. line search                      lineSearch (core/ipc_solver.inl): energy
//         backtrack via computeEnergy_DeviceOut + _global_ls_decide/_s3_decide,
//         0×NaN step-forward guard in gipc_modules/08, mid-search BVH/CP
//         rebuild on intersection
//   j. post line search                 postLineSearch (core/ipc_solver.inl):
//         close-val refresh + ground-infeasible iron-law demotion
//
// Invariants live with their owners, not here:
//   DCD≤CCD mirror lockstep ........ contact/pair_buffers.cuh
//   quarantine semantics / status .. multienv/isolation.cuh (ISOLATION CONTRACT)
//   triplet growth discipline ...... linear_system ensure_capacity_{preserve,discard}
//   reduction shapes / neutrality .. device_common/reductions.cuh
// ============================================================================

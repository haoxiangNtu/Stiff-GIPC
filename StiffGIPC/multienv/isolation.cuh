#pragma once
// ============================================================================
// multienv/isolation.cuh — the single owner of the ENV-ISOLATION machinery
// (v0.8.6 Phase 2c; docs/V086_REFACTOR_PLAN.md). Product iron law: a
// pathological env must NEVER disturb healthy envs.
//
// THE ISOLATION CONTRACT (single source of truth):
//  - Availability gate = GIPC::perEnvIsolationLive(): multi-env groups
//    declared (m_active_group_count > 1) AND the host telemetry path is on
//    (env_newton_iter_cap > 0 or STIFF_PERENV_TELEM) AND per-env alpha is on
//    (STIFF_PERENV_ALPHA — set by isolated/strict modes). The pure-device
//    fast path deliberately has NO NaN defense: isolation promises REQUIRE
//    this gate (documented contract, see comprehensive audit P1-2).
//  - Status codes (m_env_status): 0 running, 1 converged, 2 timeout-frozen
//    (per-solve), 3 quarantined/diverged. m_env_status RESETS each solve;
//    m_env_quarantined is the PERSISTENT flag (survives frames) with a
//    device mirror (m_d_env_quarantined) for the direction-zero kernel.
//  - A quarantined env is made fully INERT: positions frozen (alpha==0 keeps
//    the last accepted state verbatim — the 0*NaN guard in step-forward),
//    direction zeroed each iteration (below), its bodies removed from ground
//    detection AND ground-CCD alpha via the skip table, its slot pinned
//    status 3 in the solve loop. Init-time violations still throw.
//  - Mid-frame entry points: the frame-start flag-only probe (teleports can
//    make an env infeasible BETWEEN frames, and CCD fail-fasts run before
//    any detection — the probe must run first), the detection-time demotion
//    in throwIfGroundDistanceInvalid, and the post-PCG non-finite-direction
//    scan in solve_subIP. Policy stays at those sites; mechanics live here.
// ============================================================================

// [TU split] The four kernels and four GIPC method bodies live in
// multienv/isolation.cu since the v0.8.6 physical-separation pass — this
// header carries THE CONTRACT and stays included from the composite for
// discoverability. Kernel consumers outside isolation.cu (the post-PCG scan
// and direction-zero launches in core/ipc_solver.cu) reach the kernels via
// extern __global__ declarations under relocatable device code.

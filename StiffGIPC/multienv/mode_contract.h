// ============================================================================
// multienv/mode_contract.h — THE promise table for merged / isolated / strict
// (v0.8.6 mode-design hardening, tier C2). Doc-only header.
//
// A mode is a point on TWO independent axes:
//   axis A (coupling)     — how much solver state the envs share
//   axis B (determinism)  — what reproducibility is promised
// The fourth corner (coupled + deterministic) has no user story; do not add
// modes without one.
//
// ─── PROMISES (what users may rely on; tests pin these) ─────────────────────
// MERGED
//   + maximum throughput; one global solve (global line-search alpha, global
//     Newton convergence, shared kappa)
//   - NO isolation promise: a pathological env throws and kills the batch
//     (fail-fast; the fail-isolate extension is designed but awaits owner
//     review — see the audit note, tier C4)
//   - NO reproducibility promise (atomic emission/reduction order)
//   pinned by: kick + foldshirt-smoke behavioral gates (numeric envelope
//   gate planned, tier C3)
// ISOLATED
//   + per-env fairness: per-env line-search alpha, per-env convergence
//     freeze, per-group kappa, env-local broadphase (no cross-env contact)
//   + IRON LAW: a pathological env is quarantined fully INERT mid-run;
//     healthy envs continue (contract text: multienv/isolation.cuh)
//   - NO reproducibility promise
//   pinned by: midrun + startup quarantine gates (envelope gate planned)
// STRICT
//   + everything ISOLATED promises
//   + BITWISE reproducibility: run-to-run AND cross-architecture (proven
//     sm_80 == sm_89, anchor f7fb5a786c2d7935)
//   - costs: canonical emission orders + layout-fixed policies (measured
//     single-digit % on anchor-class scenes; workload-dependent)
//   pinned by: G1 bitwise anchor (armed MIRROR+SLOT)
//
// ─── SINGLE SOURCE OF TRUTH ─────────────────────────────────────────────────
// Bundle definition lives in stiff_physics/engine.py (resolve_multienv_mode):
//   isolated = BVH_ENVDET PERENV_BVH DECOUPLE_THRESH PERGROUP_KAPPA
//              SEGMENTED_PCG PERENV_ALPHA PERENV_PAR
//   strict   = isolated + EE_CANON EE_DETGATE CCD_CANON SPMV_DET
//   merged   = (none)
// Individual STIFF_* env vars remain per-feature DEV OVERRIDES (an explicit
// env var wins over the bundle; the resolver retracts only its own flags).
// The C++ runtime truth is GIPC::m_mode_config — a ModeConfig snapshot
// captured ONCE at finalize (multienv/mode_config.h): printable, and
// coherence-checked so half-configured bundles warn loudly instead of
// running silently (the half-config bug class).
//
// Preconditioner defaults by mode (measured, foldshirt study): MAS pairs
// best with MERGED; diag with ISOLATED/STRICT. Not auto-selected yet —
// candidates for ModeConfig-driven defaults once C3 envelope gates exist.
// ============================================================================
#pragma once

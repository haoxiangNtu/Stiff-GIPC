// ============================================================================
// core/solver_stats.h — single declaration source for the solver's cross-TU
// frame/perf counters (v0.8.6 Phase 4).
//
// DEFINITIONS live in core/ipc_solver.inl (the orchestration owns the
// counters; IPC_Solver/solve_subIP update them). Every OTHER consumer —
// engine_modules/04 stats getters, gipc_modules/14 checkpoint save/load —
// includes THIS header instead of hand-rolling `extern` declarations: a
// hand-rolled extern whose type drifts from the definition is silent UB,
// while a shared declaration header turns the mismatch into a compile error
// (core/ipc_solver.cu includes this header BEFORE the definitions, so
// decl/def divergence cannot compile).
//
// Deliberately NOT here: totalTime/ttime0..4/timemakePd/isUpdateBoundary/
// iterV/g_ls_*/g_t3_* — private to the orchestration TU; nobody else may
// grow a dependency on them.
// ============================================================================
#pragma once

extern int    totalNT;              // Newton iterations, cumulative
extern double total_Cg_count;       // PCG iterations, cumulative
extern double totalCollisionPairs;  // DCD pairs, cumulative
extern double maxCOllisionPairNum;  // DCD pairs, per-frame max
extern int    total_Frames;         // frames completed; checkpointed (drives the
                                    // stitch/soft-constraint target frame index)

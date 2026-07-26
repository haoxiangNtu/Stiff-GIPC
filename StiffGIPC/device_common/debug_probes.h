// ============================================================================
// device_common/debug_probes.h — single declaration source for the debug /
// probe globals (v0.8.6 diagnostics hardening, tier A4).
//
// WHY: C++ variables do NOT carry their type in the mangled symbol name — a
// hand-rolled `extern` whose type drifts from the definition links fine and
// is silent UB (kernels are safe: __global__ signatures mangle their
// parameter types, so a drifted kernel extern fails at device link). Every
// cross-TU consumer of these globals must include THIS header; the defining
// modules include it too, so decl/def divergence is a compile error.
// ============================================================================
#pragma once

extern int g_gipc_log_level;  // defined gipc_modules/00; log verbosity,
                              // set from Python via SimEngine::set_log_level
extern int g_dec_frame;       // defined gipc_modules/13; [decouple probe]
extern int g_dec_k;           //   frame/k dump gating (pcg_solver reads both)

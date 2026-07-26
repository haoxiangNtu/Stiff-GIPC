// ============================================================================
// energy/energy_terms.h — THE energy-term table (single source, doc-only).
// v0.8.6 energy separation E1a; blueprint: docs/ENERGY_SEPARATION_PLAN.md.
//
// The type-switch lives in energy/01_energy_host_dispatch.inl — change it and
// this table in the SAME commit. "Kernel" = the reduction in gipc_modules/07
// (moves into per-term files in later E1 slices); "G/H" = gradient/Hessian
// assembly owner today.
//
//  type  term                    size source      kernel                                G/H owner
//  ----  ----------------------  ---------------  ------------------------------------  ---------
//   0    kinetic                 fem_point_num    _getKineticEnergy_Reduction_3D        gipc 06
//   1    FEM tet elastic         fem_tet_num      _getFEMEnergy_Reduction_3D            gipc 12 + femEnergy.cuh
//   2    IPC barrier             h_cpNum[0]       _getBarrierEnergy_Reduction_3D        gipc 01/04/05
//   3    delta (line-search)     fem_point_num    _getDeltaEnergy_Reduction             —
//   4    ground barrier          h_gpNum          _computeGroundEnergy_Reduction        gipc 06
//   5    friction (lagged)       h_cpNum_last[0]  _getFrictionEnergy_Reduction_3D       gipc 02
//   6    ground friction         h_gpNum_last     _getFrictionEnergy_gd_Reduction_3D    gipc 02
//   7    rest-stable NHK         fem_tet_num      _getRestStableNHKEnergy_Reduction_3D  femEnergy.cuh
//   8    triangle membrane       triangleNum      _get_triangleFEMEnergy_Reduction_3D   gipc 12
//   9    soft constraints        softNum          _computeSoftConstraintEnergy_Reduction gipc 06
//  10    bending (quad|angle)    tri_edge_num     _getQuadBending/_getBendingEnergy     gipc 12
//  11    VESTIGIAL — sized (triangleNum) but never dispatched; E2 cleanup
//
// Constitutive libraries: femEnergy.cuh carries BOTH ARAP and SNK (USE_SNK1);
// bending has USE_QUADRATIC_BENDING. ABD affine energy lives in abd_system/
// (already its own TU). The smooth/mollifier branches inside the barrier
// family and the close-set chain are FROZEN (G0.5 tripwire).
//
// Adding a constitutive model (end state, phase E2/E3): one new per-term
// file implementing energy/gradient/hessian + ONE row here + ONE registry
// entry — nothing else.
// ============================================================================
#pragma once

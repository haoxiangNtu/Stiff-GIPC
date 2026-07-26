// ============================================================================
// energy/composite_externs.cuh — extern __global__ declarations the COMPOSITE
// needs for kernels whose definitions moved into per-term TUs (E3). Kernel
// externs self-check at device link (signatures mangle); grow this file one
// rung at a time, verbatim signatures only.
// ============================================================================
#pragma once

// ── rung 1: kinetic (energy/kinetic.cu) ──
// reduction kernel: still launched by the composite's BLOCKING dispatcher
// (single live caller, type 3) — extern carries the default args (legal:
// defaults may differ per TU declaration; the defining TU keeps its own).
extern __global__ void _getKineticEnergy_Reduction_3D(
    double3* _vertexes, double3* _xTilta, double* _energy, double* _masses, int number,
    double* penv = nullptr, const int* p2g = nullptr, int ng = 0);
extern __global__ void _calKineticGradient(
    double3* vertexes, double3* xTilta, double3* gradient, double* masses, int numbers);

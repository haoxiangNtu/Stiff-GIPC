#!/bin/bash
# [frame-fsm P0] Repeatable audit of host-only residues on the frame path.
# Run from the repo root. Each section lists constructs that are illegal (or
# a host round-trip) inside a captured frame; the counts are the burn-down
# metric for P1..P3b. Finalize/warm-up files are excluded where practical.
cd "$(dirname "$0")/.."
echo "=== frame-path host residues (target: 0 by P3b) ==="

section() { echo; echo "--- $1 ---"; }

section "blocking D2H memcpy in solver frame files (GIPC.cu, pcg_solver, mlbvh, MASPreconditioner, converter, global_linear_system)"
grep -n 'cudaMemcpy(.*DeviceToHost' \
    StiffGIPC/GIPC.cu StiffGIPC/linear_system/solver/pcg_solver.cu \
    StiffGIPC/mlbvh.cu StiffGIPC/MASPreconditioner.cu \
    StiffGIPC/linear_system/utils/converter.cu \
    StiffGIPC/linear_system/linear_system/global_linear_system.cu 2>/dev/null \
    | grep -v 'sceneToHost\|STIFF_MAS_DUMP\|_tdumped\|frame_timing\|// finalize' | wc -l

section "device-wide synchronize on the frame path"
grep -n 'cudaDeviceSynchronize' \
    StiffGIPC/GIPC.cu StiffGIPC/linear_system/solver/pcg_solver.cu \
    StiffGIPC/mlbvh.cu StiffGIPC/MASPreconditioner.cu 2>/dev/null \
    | grep -vE '^\s*//|dump|DUMP|ksum' | wc -l

section "thrust calls (internal alloc/free + device sync) outside finalize"
grep -n 'thrust::' StiffGIPC/GIPC.cu StiffGIPC/mlbvh.cu \
    StiffGIPC/MASPreconditioner.cu StiffGIPC/linear_system/utils/converter.cu 2>/dev/null \
    | grep -v '^\s*//' | wc -l

section "cudaMalloc/cudaFree reachable per frame (candidates; verify each)"
grep -n 'cudaMalloc\|cudaFree' StiffGIPC/GIPC.cu 2>/dev/null \
    | grep -iE 'isIntersected|intersect|grow|redo' | wc -l

section "per-frame host file writes"
grep -n 'ofstream\|write_to_file' StiffGIPC/GIPC.cu 2>/dev/null | grep -v '^\s*//' | wc -l

echo
echo "=== detail (top offenders, for the burn-down list) ==="
grep -n 'cudaMemcpy(.*DeviceToHost' StiffGIPC/GIPC.cu 2>/dev/null \
    | grep -v 'STIFF_MAS_DUMP\|frame_timing' | head -25

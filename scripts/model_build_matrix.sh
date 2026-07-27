#!/usr/bin/env bash
# Production-build and invariant matrix for every supported FEM model.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
LOG_DIR="${GATE_LOG_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/stiffgipc-model-matrix.XXXXXX")}"
BUILD_ROOT="${MODEL_MATRIX_BUILD_ROOT:-$ROOT/build-model-matrix}"
MODES="${MODEL_MATRIX_MODES:-merged}"
JOBS="${GATE_BUILD_JOBS:-$(nproc)}"

mkdir -p "$LOG_DIR" "$BUILD_ROOT"

for model in SNK1 SNK2 ARAP; do
    lower="${model,,}"
    build_dir="$BUILD_ROOT/$lower"
    echo "== production model $model =="
    cmake -S "$ROOT" -B "$build_dir" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_PYTHON_BINDINGS=ON \
        -DBUILD_GL_VIEWER=OFF \
        -DSTIFFGIPC_ENABLE_DIAGNOSTICS=OFF \
        -DSTIFFGIPC_FEM_MODEL="$model" \
        > "$LOG_DIR/model-$lower-configure.log" 2>&1
    cmake --build "$build_dir" -j"$JOBS" --target pystiffgipc \
        > "$LOG_DIR/model-$lower-build.log" 2>&1

    env STIFFGIPC_NATIVE_DIR="$build_dir" PYTHONPATH="$ROOT" \
        MODEL_EXPECT="$model" python3 - "$build_dir" <<'PY'
import os
import sys

expected_dir = os.path.realpath(sys.argv[1])
from stiff_physics.engine import _C

native = os.path.realpath(_C.__file__)
if os.path.commonpath((native, expected_dir)) != expected_dir:
    raise SystemExit(
        f"native module escaped model build: got={native} expected={expected_dir}"
    )
if _C.fem_model() != os.environ["MODEL_EXPECT"]:
    raise SystemExit(
        f"compiled model {_C.fem_model()} != {os.environ['MODEL_EXPECT']}"
    )
for name in (
    "debug_fd_gradient_check",
    "debug_fd_hessian_check",
    "debug_fd_activity",
):
    if hasattr(_C.SimEngine, name):
        raise SystemExit(f"production build unexpectedly exports {name}")
print(
    f"production import PASS model={_C.fem_model()} native={native}",
    flush=True,
)
PY

    env STIFFGIPC_NATIVE_DIR="$build_dir" PYTHONPATH="$ROOT" \
        MODEL_EXPECT="$model" MODEL_MODES="$MODES" \
        timeout 1200 python3 "$ROOT/scripts/model_validation.py" \
        > "$LOG_DIR/model-$lower-validation.log" 2>&1
    grep -qF "MODEL-VALIDATION: PASS" \
        "$LOG_DIR/model-$lower-validation.log"
done

echo "MODEL-BUILD-MATRIX: PASS"

#!/usr/bin/env bash
# Shared helpers for push-time and manually invoked verification.

gate_reset_test_environment()
{
    local name
    while IFS='=' read -r name _; do
        case "$name" in
            STIFF_*|STIFFGIPC_NATIVE_DIR|GIPC_*|BVHSKIP*|NAN_*|FD_DEBUG|\
            CASE*|SCENE_*|FD_*|DEMO_*|BENCH_*|CHECKPOINT_*|MODEL_*|\
            GRASP_*|GRIP_*|DUCK_*|BEAKER_*|CUP_*|FEM_*|\
            AUTO_STEP|BAR|BUFF|CLEAN|DUMP_POS|EXCLUDE_FEM|MESH|RECORD|\
            SKIP_COL|SKIP_EXCLUSIONS|NO_CUP_TABLE|NO_FEM|NO_STITCH|\
            POS_K|SC|USE_HARD_PIN|HARD_PIN_*|MAX_REVOLUTE_*|YM|\
            GUI_*|UI_*|SANITIZER_TOOLS)
                unset "$name"
                ;;
        esac
    done < <(env)
}

gate_absolute_dir()
{
    local path=$1
    mkdir -p "$path"
    (cd "$path" && pwd -P)
}

gate_prepare_paths()
{
    local root=$1

    if [ -n "${GATE_BUILD_DIR:-}" ]; then
        GATE_BUILD_DIR=$(gate_absolute_dir "$GATE_BUILD_DIR")
    else
        GATE_BUILD_DIR=$(gate_absolute_dir "$root/build")
    fi

    if [ -n "${GATE_LOG_DIR:-}" ]; then
        GATE_LOG_DIR=$(gate_absolute_dir "$GATE_LOG_DIR")
    else
        GATE_LOG_DIR=$(mktemp -d "${TMPDIR:-/tmp}/stiffgipc-gates.XXXXXX")
    fi

    if [ -n "${GATE_RESULTS_DIR:-}" ]; then
        GATE_RESULTS_DIR=$(gate_absolute_dir "$GATE_RESULTS_DIR")
    else
        GATE_RESULTS_DIR=$(gate_absolute_dir "$GATE_LOG_DIR/results")
    fi

    export GATE_BUILD_DIR GATE_LOG_DIR GATE_RESULTS_DIR
    export STIFFGIPC_NATIVE_DIR="$GATE_BUILD_DIR"
    export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
}

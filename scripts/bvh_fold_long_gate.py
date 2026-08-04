#!/usr/bin/env python3
"""Run the complete FOLD-SHIRT trajectory for BVH release candidates.

The short candidate gate proves frozen pair completeness.  This gate supplies
the complementary long-horizon evidence: merged and isolated replays must each
finish the requested number of real frames, publish finite state, and avoid
CUDA/overflow failures.  Optional whole-frame-graph mode additionally requires
every frame to be accounted for and no overflow status.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from bvh_candidate_gate import candidate_environment


ROOT = Path(__file__).resolve().parent.parent
FRAMES = int(os.environ.get("BVH_FOLD_LONG_FRAMES", "1550"))
ENVS = int(os.environ.get("BVH_FOLD_LONG_ENVS", "1"))
TIMEOUT = int(os.environ.get("BVH_FOLD_LONG_TIMEOUT", "10800"))
GRAPH = bool(int(os.environ.get("BVH_FOLD_LONG_GRAPH", "0")))
MODES = tuple(
    item.strip()
    for item in os.environ.get(
        "BVH_FOLD_LONG_MODES", "merged,isolated"
    ).split(",")
    if item.strip()
)
ARTIFACT_DIR = Path(
    os.environ.get("BVH_FOLD_LONG_ARTIFACT_DIR", "/tmp/stiff-bvh-long")
).resolve()

BAD_OUTPUT = re.compile(
    r"Traceback|CUDA error|illegal memory|budget exhausted.*nan|"
    r"abd-kinetic-nan|retry budget exhausted",
    re.I,
)
SUMMARY = re.compile(
    r"\[fs-hl\]\s+(\d+) envs,\s+(\d+) frames: mean\s+([0-9.]+)ms"
)
GRAPH_SUMMARY = re.compile(
    r"\[fs-graph-audit\] full=(\d+) fallback=(\d+) overflow=(\d+)"
)


def run_mode(mode: str, candidate: dict[str, str]) -> dict[str, object]:
    if mode not in ("merged", "isolated"):
        raise RuntimeError(f"unknown mode {mode!r}")
    stem = f"fold-{mode}-{FRAMES}f-{ENVS}e-{'graph' if GRAPH else 'step'}"
    log_path = ARTIFACT_DIR / f"{stem}.log"
    vertex_path = ARTIFACT_DIR / f"{stem}-vertices.npy"
    checkpoint_path = ARTIFACT_DIR / f"{stem}.ckpt"

    env = os.environ.copy()
    env.update(candidate)
    env.update(
        CASE39ME_HEADLESS="1",
        CASE39ME_NUM_ENVS=str(ENVS),
        CASE39_FRICTION="0.8",
        CASE39_TRACE_EVERY=os.environ.get("BVH_FOLD_LONG_TRACE_EVERY", "100"),
        CASE39_FRAME_START="0",
        CASE39_FRAME_END=str(FRAMES),
        CASE39ME_DUMP_VERTS=str(vertex_path),
        CASE39ME_SAVE_CHECKPOINT=str(checkpoint_path),
        STIFF_MULTIENV_MODE=mode,
        GIPC_LOG_LEVEL="0",
    )
    graph_knobs = (
        "STIFF_FRAME_GRAPH",
        "STIFF_FRAME_FULL_GRAPH",
        "STIFF_C4_COLLISION_GRAPH",
        "STIFF_C5_ISOLATED_GRAPH",
        "STIFF_C6_ABD_STEP_GRAPH",
    )
    for name in graph_knobs:
        env.pop(name, None)
    if GRAPH:
        env.update(
            STIFF_FRAME_GRAPH="1",
            STIFF_FRAME_FULL_GRAPH="1",
            STIFF_C4_COLLISION_GRAPH="1",
            STIFF_C6_ABD_STEP_GRAPH="1",
            CASE39_GRAPH_STATS="1",
        )
        if mode == "isolated":
            env["STIFF_C5_ISOLATED_GRAPH"] = "1"
    else:
        env.pop("CASE39_GRAPH_STATS", None)

    command = [sys.executable, str(ROOT / "examples/replay_foldshirt_multienv.py")]
    started = time.monotonic()
    output_lines: list[str] = []
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                output_lines.append(line)
                log.write(line)
                if line.startswith(("[fs-hl] frame", "[fs-graph-audit]")):
                    print(f"[{mode}] {line.rstrip()}", flush=True)
                if time.monotonic() - started > TIMEOUT:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    raise RuntimeError(
                        f"{mode}: exceeded {TIMEOUT}s; see {log_path}"
                    )
            return_code = process.wait()
        finally:
            if process.poll() is None:
                process.terminate()

    elapsed = time.monotonic() - started
    output = "".join(output_lines)
    if return_code or BAD_OUTPUT.search(output):
        raise RuntimeError(
            f"{mode}: replay failed rc={return_code}; see {log_path}"
        )
    summaries = SUMMARY.findall(output)
    if not summaries:
        raise RuntimeError(f"{mode}: missing final frame summary; see {log_path}")
    env_count, frame_count, mean_ms = summaries[-1]
    if int(env_count) != ENVS or int(frame_count) != FRAMES:
        raise RuntimeError(
            f"{mode}: completed {frame_count} frames/{env_count} envs, "
            f"expected {FRAMES}/{ENVS}"
        )
    if not vertex_path.is_file() or not checkpoint_path.is_file():
        raise RuntimeError(f"{mode}: final artifacts were not written")
    vertices = np.load(vertex_path)
    if not np.isfinite(vertices).all():
        raise RuntimeError(f"{mode}: final state contains non-finite vertices")

    graph_full = graph_fallback = graph_overflow = None
    if GRAPH:
        graph_rows = GRAPH_SUMMARY.findall(output)
        if not graph_rows:
            raise RuntimeError(f"{mode}: missing graph coverage summary")
        graph_full, graph_fallback, graph_overflow = map(int, graph_rows[-1])
        if graph_full + graph_fallback != FRAMES or graph_overflow != 0:
            raise RuntimeError(
                f"{mode}: invalid graph accounting full={graph_full} "
                f"fallback={graph_fallback} overflow={graph_overflow}"
            )

    digest = hashlib.sha256(vertices.tobytes()).hexdigest()
    result: dict[str, object] = {
        "mode": mode,
        "frames": FRAMES,
        "envs": ENVS,
        "graph": GRAPH,
        "elapsed_s": elapsed,
        "mean_step_ms": float(mean_ms),
        "vertex_shape": list(vertices.shape),
        "vertex_sha256": digest,
        "finite": True,
        "graph_full": graph_full,
        "graph_fallback": graph_fallback,
        "graph_overflow": graph_overflow,
        "log": str(log_path),
        "vertices": str(vertex_path),
        "checkpoint": str(checkpoint_path),
    }
    print(
        f"FOLD-LONG: {mode} PASS frames={FRAMES} elapsed={elapsed:.1f}s "
        f"mean={float(mean_ms):.1f}ms hash={digest[:16]}",
        flush=True,
    )
    return result


def main() -> int:
    if FRAMES <= 0 or ENVS <= 0 or not MODES:
        print("BVH-FOLD-LONG-GATE: invalid frames/envs/modes")
        return 2
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        candidate = candidate_environment()
        print(
            "candidate: "
            + " ".join(f"{key}={value}" for key, value in candidate.items()),
            flush=True,
        )
        results = [run_mode(mode, candidate) for mode in MODES]
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"BVH-FOLD-LONG-GATE: FAIL: {error}")
        return 1

    summary_path = ARTIFACT_DIR / (
        f"fold-{FRAMES}f-{ENVS}e-{'graph' if GRAPH else 'step'}-summary.json"
    )
    summary_path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"BVH-FOLD-LONG-GATE: PASS summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Frozen-state FOLD-SHIRT contact-completeness gate for BVH candidates.

Merged/isolated FOLD replays are not bitwise repeatable across independent
processes, so comparing two freely evolved trajectories is not a valid pair-set
oracle.  This gate first creates baseline checkpoints at representative frames.
It then loads each exact checkpoint in fresh baseline and candidate processes,
advances zero frames, and rebuilds the active IPC contact set on both sides.
Thus both traversals see bit-identical geometry and topology.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from bvh_candidate_gate import (
    canonical_pair_multiset,
    candidate_environment,
    encoded_pair_multiset,
)


ROOT = Path(__file__).resolve().parent.parent
MODES = ("merged", "isolated")
CHECKPOINT_FRAMES = tuple(
    int(item)
    for item in os.environ.get("BVH_FOLD_GATE_CHECKPOINTS", "1,10,30").split(",")
    if item.strip()
)
REQUIRE_ENCODING = bool(
    int(os.environ.get("BVH_FOLD_GATE_REQUIRE_ENCODING", "1"))
)
BAD_OUTPUT = re.compile(
    r"Traceback|CUDA error|illegal memory|budget exhausted.*nan", re.I
)


def base_environment(
    mode: str, candidate_knobs: set[str]
) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in candidate_knobs
    }
    env.update(
        CASE39ME_HEADLESS="1",
        CASE39ME_NUM_ENVS=os.environ.get("BVH_FOLD_GATE_ENVS", "1"),
        CASE39_FRICTION="0.8",
        CASE39_TRACE_EVERY="0",
        STIFF_MULTIENV_MODE=mode,
        STIFF_LOG_LEVEL="0",
        GIPC_LOG_LEVEL="0",
    )
    return env


def invoke(env: dict[str, str], label: str) -> str:
    process = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "replay_foldshirt_multienv.py")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    output = process.stdout + process.stderr
    if process.returncode or BAD_OUTPUT.search(output):
        sys.stderr.write(output[-6000:])
        raise RuntimeError(f"{label} failed with rc={process.returncode}")
    return output


def create_checkpoint(
    mode: str,
    frame: int,
    candidate_knobs: set[str],
    checkpoint: Path,
) -> None:
    env = base_environment(mode, candidate_knobs)
    env.update(
        CASE39_FRAME_START="0",
        CASE39_FRAME_END=str(frame),
        CASE39ME_SAVE_CHECKPOINT=str(checkpoint),
    )
    invoke(env, f"{mode} checkpoint@{frame}")
    if not checkpoint.is_file():
        raise RuntimeError(f"{mode}@{frame}: checkpoint was not created")


def frozen_oracle(
    mode: str,
    frame: int,
    candidate: dict[str, str] | None,
    candidate_knobs: set[str],
    checkpoint: Path,
    vertex_path: Path,
    pair_path: Path,
    ccd_pair_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    env = base_environment(mode, candidate_knobs)
    env.update(
        CASE39_FRAME_START=str(frame),
        CASE39_FRAME_END=str(frame),
        CASE39ME_LOAD_CHECKPOINT=str(checkpoint),
        CASE39ME_DUMP_VERTS=str(vertex_path),
        CASE39ME_DUMP_PAIRS=str(pair_path),
        CASE39ME_DUMP_CCD_PAIRS=str(ccd_pair_path),
    )
    if candidate:
        env.update(candidate)
    invoke(env, f"{mode}@{frame} {'candidate' if candidate else 'baseline'}")
    if (
        not vertex_path.is_file()
        or not pair_path.is_file()
        or not ccd_pair_path.is_file()
    ):
        raise RuntimeError(f"{mode}@{frame}: missing vertex, DCD, or CCD dump")
    return np.load(vertex_path), np.load(pair_path), np.load(ccd_pair_path)


def main() -> int:
    if not CHECKPOINT_FRAMES or min(CHECKPOINT_FRAMES) < 1:
        print("BVH-FOLD-PAIR-GATE: ERROR: checkpoint frames must be positive")
        return 2
    try:
        candidate = candidate_environment()
    except RuntimeError as error:
        print(f"BVH-FOLD-PAIR-GATE: ERROR: {error}")
        return 2

    candidate_knobs = set(candidate)
    failures: list[str] = []
    print(
        f"checkpoints={','.join(map(str, CHECKPOINT_FRAMES))} "
        f"envs={os.environ.get('BVH_FOLD_GATE_ENVS', '1')} candidate="
        + " ".join(f"{key}={value}" for key, value in candidate.items())
        + f" require_encoding={int(REQUIRE_ENCODING)}"
    )
    with tempfile.TemporaryDirectory(prefix="stiff-bvh-fold-pairs.") as tmp:
        directory = Path(tmp)
        for mode in MODES:
            for frame in CHECKPOINT_FRAMES:
                stem = f"{mode}-{frame}"
                checkpoint = directory / f"{stem}.ckpt"
                try:
                    create_checkpoint(
                        mode, frame, candidate_knobs, checkpoint
                    )
                    base_vertices, base_pairs, base_ccd_pairs = frozen_oracle(
                        mode,
                        frame,
                        None,
                        candidate_knobs,
                        checkpoint,
                        directory / f"{stem}-base.npy",
                        directory / f"{stem}-base-pairs.npy",
                        directory / f"{stem}-base-ccd-pairs.npy",
                    )
                    cand_vertices, cand_pairs, cand_ccd_pairs = frozen_oracle(
                        mode,
                        frame,
                        candidate,
                        candidate_knobs,
                        checkpoint,
                        directory / f"{stem}-candidate.npy",
                        directory / f"{stem}-candidate-pairs.npy",
                        directory / f"{stem}-candidate-ccd-pairs.npy",
                    )
                except RuntimeError as error:
                    failures.append(str(error))
                    continue

                geometry_exact = np.array_equal(base_vertices, cand_vertices)
                base_set = canonical_pair_multiset(base_pairs)
                cand_set = canonical_pair_multiset(cand_pairs)
                pairs_exact = np.array_equal(base_set, cand_set)
                encoding_exact = np.array_equal(
                    encoded_pair_multiset(base_pairs),
                    encoded_pair_multiset(cand_pairs),
                )
                base_ccd_set = canonical_pair_multiset(base_ccd_pairs)
                cand_ccd_set = canonical_pair_multiset(cand_ccd_pairs)
                ccd_pairs_exact = np.array_equal(base_ccd_set, cand_ccd_set)
                ccd_encoding_exact = np.array_equal(
                    encoded_pair_multiset(base_ccd_pairs),
                    encoded_pair_multiset(cand_ccd_pairs),
                )
                print(
                    f"{mode:8s} frame={frame:4d} "
                    f"geometry={'exact' if geometry_exact else 'DIFF'} "
                    f"pairs={len(base_set)}/{len(cand_set)} "
                    f"pair_multiset={'exact' if pairs_exact else 'DIFF'} "
                    f"encoding={'exact' if encoding_exact else 'permuted'} "
                    f"ccd={len(base_ccd_set)}/{len(cand_ccd_set)} "
                    f"ccd_multiset={'exact' if ccd_pairs_exact else 'DIFF'} "
                    f"ccd_encoding={'exact' if ccd_encoding_exact else 'permuted'}"
                )
                if not geometry_exact:
                    failures.append(
                        f"{mode}@{frame}: checkpoint geometry differs before query"
                    )
                if not pairs_exact:
                    failures.append(
                        f"{mode}@{frame}: active contact multiset changed"
                    )
                if REQUIRE_ENCODING and not encoding_exact:
                    failures.append(
                        f"{mode}@{frame}: DCD pair encoding changed"
                    )
                if not ccd_pairs_exact:
                    failures.append(
                        f"{mode}@{frame}: swept CCD candidate multiset changed"
                    )
                if REQUIRE_ENCODING and not ccd_encoding_exact:
                    failures.append(
                        f"{mode}@{frame}: CCD pair encoding changed"
                    )

    if failures:
        for failure in failures:
            print("FAIL:", failure)
        print("BVH-FOLD-PAIR-GATE: FAIL")
        return 1
    print("BVH-FOLD-PAIR-GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

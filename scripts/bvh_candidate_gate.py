#!/usr/bin/env python3
"""Equivalence gate for opt-in BVH acceleration candidates.

This gate compares a candidate against the same binary with its candidate
knobs removed.  It is intentionally stricter than a smoke test: all three
multi-environment modes run in separate processes, final positions are
compared, Newton envelopes must match, and the strict bitwise anchor must stay
at the repository gold value.

The default candidate is the validation branch's exact original-index EE range
pruning (mode 1).
Override BVH_CANDIDATE_ENV with comma-separated NAME=VALUE entries to test a
different candidate.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
GOLD = "0544461bd82123ae"
MODES = ("merged", "isolated", "strict")
FRAMES = int(os.environ.get("BVH_GATE_FRAMES", "50"))
POSITION_TOL = float(os.environ.get("BVH_GATE_POSITION_TOL", "1e-8"))
BASELINE_RUNS = max(2, int(os.environ.get("BVH_GATE_BASELINE_RUNS", "3")))
NOISE_FACTOR = float(os.environ.get("BVH_GATE_NOISE_FACTOR", "1.05"))
BAD_OUTPUT = re.compile(r"Traceback|CUDA error|budget exhausted.*nan", re.I)


def candidate_environment() -> dict[str, str]:
    value = os.environ.get(
        "BVH_CANDIDATE_ENV", "STIFF_EE_RANGE_PRUNE=1"
    )
    result: dict[str, str] = {}
    for assignment in value.split(","):
        assignment = assignment.strip()
        if not assignment:
            continue
        name, separator, item = assignment.partition("=")
        if not separator or not name.startswith("STIFF_"):
            raise RuntimeError(
                "BVH_CANDIDATE_ENV entries must be STIFF_NAME=VALUE"
            )
        result[name] = item
    if not result:
        raise RuntimeError("BVH_CANDIDATE_ENV selected no candidate knobs")
    return result


def run_anchor(
    mode: str,
    candidate: dict[str, str] | None,
    candidate_knobs: set[str],
    dump_path: Path,
    pair_path: Path,
) -> tuple[str, int, np.ndarray, np.ndarray]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in candidate_knobs
    }
    env.update(
        SCENE_MODE=mode,
        SCENE_N="2",
        SCENE_FRAMES=str(FRAMES),
        DUMP_POS=str(dump_path),
        DUMP_PAIRS=str(pair_path),
        GIPC_LOG_LEVEL="0",
    )
    if candidate:
        env.update(candidate)
    process = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "anchor_scene.py")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    output = process.stdout + process.stderr
    if process.returncode or BAD_OUTPUT.search(output):
        sys.stderr.write(output[-4000:])
        raise RuntimeError(
            f"{mode} {'candidate' if candidate else 'baseline'} failed "
            f"with rc={process.returncode}"
        )
    vhash = next(
        (line.split()[1] for line in output.splitlines() if line.startswith("VHASH ")),
        None,
    )
    newton = next(
        (int(line.split()[1]) for line in output.splitlines() if line.startswith("NEWTON ")),
        None,
    )
    if (
        vhash is None
        or newton is None
        or not dump_path.is_file()
        or not pair_path.is_file()
    ):
        raise RuntimeError(
            f"{mode}: missing VHASH, NEWTON, position dump, or pair dump"
        )
    return vhash, newton, np.load(dump_path), np.load(pair_path)


def canonical_pair_multiset(pairs: np.ndarray) -> np.ndarray:
    """Canonicalize clean PP/PE/PT/EE rows without erasing multiplicity.

    Clean rows contain 2, 3, or 4 non-negative physical vertex ids followed by
    -1 padding.  Candidate mode 2 may reverse an edge or swap the two edges;
    sorting the participating ids erases only those representation choices.
    Arity is preserved by the padding, rows are then lexicographically sorted,
    and duplicate rows remain duplicate rows.
    """
    rows = np.asarray(pairs, dtype=np.int64)
    if rows.ndim != 2 or rows.shape[1] != 4:
        raise RuntimeError(f"invalid clean-pair shape {rows.shape}, expected (N,4)")
    canonical = np.full(rows.shape, -1, dtype=np.int64)
    for index, row in enumerate(rows):
        vertices = np.sort(row[row >= 0])
        canonical[index, : len(vertices)] = vertices
    if len(canonical):
        order = np.lexsort(tuple(canonical[:, column] for column in range(3, -1, -1)))
        canonical = canonical[order]
    return canonical


def encoded_pair_multiset(pairs: np.ndarray) -> np.ndarray:
    """Sort clean rows while preserving their within-row IPC representation."""
    rows = np.asarray(pairs, dtype=np.int64)
    if rows.ndim != 2 or rows.shape[1] != 4:
        raise RuntimeError(f"invalid clean-pair shape {rows.shape}, expected (N,4)")
    if len(rows):
        order = np.lexsort(tuple(rows[:, column] for column in range(3, -1, -1)))
        rows = rows[order]
    return rows


def main() -> int:
    try:
        candidate = candidate_environment()
    except RuntimeError as error:
        print(f"BVH-CANDIDATE-GATE: ERROR: {error}")
        return 2

    failures: list[str] = []
    candidate_knobs = set(candidate)
    print("candidate:", " ".join(f"{k}={v}" for k, v in candidate.items()))
    with tempfile.TemporaryDirectory(prefix="stiff-bvh-candidate.") as tmp:
        directory = Path(tmp)
        for mode in MODES:
            try:
                baseline_runs = 1 if mode == "strict" else BASELINE_RUNS
                baselines = [
                    run_anchor(
                        mode,
                        None,
                        candidate_knobs,
                        directory / f"{mode}-base-{run}.npy",
                        directory / f"{mode}-base-{run}-pairs.npy",
                    )
                    for run in range(baseline_runs)
                ]
                cand_hash, cand_newton, cand_positions, cand_pairs = run_anchor(
                    mode,
                    candidate,
                    candidate_knobs,
                    directory / f"{mode}-candidate.npy",
                    directory / f"{mode}-candidate-pairs.npy",
                )
            except RuntimeError as error:
                failures.append(str(error))
                continue

            base_hash, base_newton, base_positions, base_pairs = baselines[0]
            baseline_differences = [
                float(np.max(np.abs(a[2] - b[2])))
                for index, a in enumerate(baselines)
                for b in baselines[index + 1 :]
            ]
            noise_envelope = max(baseline_differences, default=0.0)
            candidate_differences = [
                float(np.max(np.abs(cand_positions - run[2])))
                for run in baselines
            ]
            difference = min(candidate_differences)
            allowed_difference = max(
                POSITION_TOL, NOISE_FACTOR * noise_envelope
            )
            base_pair_sets = [canonical_pair_multiset(run[3]) for run in baselines]
            base_pair_set = base_pair_sets[0]
            cand_pair_set = canonical_pair_multiset(cand_pairs)
            pair_equal = any(
                np.array_equal(pair_set, cand_pair_set)
                for pair_set in base_pair_sets
            )
            baseline_hashes = ",".join(run[0] for run in baselines)
            print(
                f"{mode:8s} baseline={baseline_hashes}/"
                f"{sorted(set(run[1] for run in baselines))} "
                f"candidate={cand_hash}/{cand_newton} "
                f"nearest|max dpos|={difference:.3e} "
                f"baseline_noise={noise_envelope:.3e} "
                f"allowed={allowed_difference:.3e} "
                f"pairs={len(base_pair_set)}/{len(cand_pair_set)} "
                f"pair_multiset={'exact' if pair_equal else 'DIFF'}"
            )
            baseline_newtons = {run[1] for run in baselines}
            if cand_newton not in baseline_newtons:
                failures.append(
                    f"{mode}: Newton envelope changed "
                    f"{sorted(baseline_newtons)}->{cand_newton}"
                )
            if difference > allowed_difference:
                failures.append(
                    f"{mode}: max position delta {difference:.3e} "
                    f"> noise-aware limit {allowed_difference:.3e} "
                    f"(baseline noise {noise_envelope:.3e})"
                )
            if not pair_equal:
                baseline_only = len(
                    set(map(tuple, base_pair_set)) - set(map(tuple, cand_pair_set))
                )
                candidate_only = len(
                    set(map(tuple, cand_pair_set)) - set(map(tuple, base_pair_set))
                )
                failures.append(
                    f"{mode}: active contact multiset changed "
                    f"(baseline-only unique={baseline_only}, "
                    f"candidate-only unique={candidate_only})"
                )
            if mode == "strict":
                if base_hash != cand_hash:
                    failures.append(
                        f"strict candidate changed the baseline hash: "
                        f"base={base_hash}, candidate={cand_hash}"
                    )
                # The repository gold is defined at the gate's canonical
                # 50-frame horizon.  A shortened developer smoke has a
                # different legitimate hash and must not be reported as a
                # physics failure.
                if FRAMES == 50 and (
                    base_hash != GOLD or cand_hash != GOLD
                ):
                    failures.append(
                        f"strict gold changed: base={base_hash}, "
                        f"candidate={cand_hash}, expected={GOLD}"
                    )

    if failures:
        for failure in failures:
            print("FAIL:", failure)
        print("BVH-CANDIDATE-GATE: FAIL")
        return 1
    print("BVH-CANDIDATE-GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

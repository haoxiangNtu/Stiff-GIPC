#!/usr/bin/env python3
"""P3b transactional-frame bitwise and failure-context regression.

The worker hashes the native checkpoint format, not only rendered vertices:
FEM x/x_prev/v/x_tilde, ABD q/q_prev/q_v, Kappa, and the committed frame id
are therefore compared byte for byte.  Every case runs in a fresh process so
the legacy global telemetry counters cannot mask a rollback defect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ASSETS = str(ROOT / "Assets") + "/"
MARKER = "FRAME_FSM_JSON="


def checkpoint_hash(engine) -> str:
    fd, path = tempfile.mkstemp(prefix="frame_fsm_", suffix=".bin")
    os.close(fd)
    try:
        engine.native.save_checkpoint(path)
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    finally:
        Path(path).unlink(missing_ok=True)


def make_engine():
    sys.path.insert(0, str(ROOT))
    from stiff_physics.engine import Config, Engine

    engine = Engine(Config(
        dt=0.020,
        ground_offset=-0.5,
        assets_dir=ASSETS,
        preconditioner_type=0,
        collision_detection_buff_scale=float(
            os.environ.get("STIFF_FRAME_TEST_BUFF_SCALE", "6.0")
        ),
    ))
    transform = np.eye(4)
    transform[:3, :3] *= 0.1
    transform[1, 3] = 0.5
    engine.load_mesh("sim_data/tetmesh/cube.msh",
                     dimensions=3,
                     body_type="ABD",
                     transform=transform,
                     young_modulus=1e8)
    transform = np.eye(4)
    transform[:3, :3] *= 0.1
    engine.load_mesh("sim_data/tetmesh/cube.msh",
                     dimensions=3,
                     body_type="FEM",
                     transform=transform,
                     young_modulus=1e6)
    engine.finalize()
    return engine


def worker() -> None:
    engine = make_engine()
    before = checkpoint_hash(engine)
    error = ""
    try:
        engine.step()
    except RuntimeError as exc:
        error = str(exc)
    after = checkpoint_hash(engine)
    status = engine.get_frame_status()
    fields = (
        "result", "phase", "error_code", "invalid_bits", "err_env",
        "err_primitive", "err_newton_iter", "err_ls_iter", "path_flags",
        "graph_launches", "host_boundaries", "newton_iters", "pcg_iters",
        "ls_trials", "hw_dcd_pairs", "hw_ccd_pairs", "hw_triplets",
        "required_dcd_pairs", "required_ccd_pairs", "required_triplets",
        "root_graph_nodes", "root_d2h_nodes", "terminal_graph_nodes",
        "terminal_d2h_nodes", "frame_id", "attempt", "retry_count",
        "retry_invalid_bits",
    )
    payload = {name: int(getattr(status, name)) for name in fields}
    payload.update(before=before, after=after, error=error)
    print(MARKER + json.dumps(payload, sort_keys=True), flush=True)


def run_case(name: str, updates: dict[str, str]) -> dict:
    env = dict(os.environ)
    for key in (
        "STIFF_FRAME_GRAPH", "STIFF_FRAME_FORCE_ROLLBACK",
        "STIFF_FRAME_TEST_DCD_CAP", "STIFF_FRAME_TEST_CCD_CAP",
        "STIFF_FRAME_TEST_TRIPLET_CAP", "STIFF_FRAME_TEST_NAN_VERTEX",
        "STIFF_FRAME_TEST_BUFF_SCALE",
    ):
        env.pop(key, None)
    env.update(PYTHONPATH=str(ROOT),
               STIFFGIPC_NATIVE_DIR=str(ROOT / "build"),
               STIFF_LOG_LEVEL="0",
               **updates)
    result = subprocess.run([sys.executable, __file__, "--worker"],
                            cwd=ROOT,
                            env=env,
                            text=True,
                            capture_output=True,
                            timeout=120)
    output = result.stdout + result.stderr
    if result.returncode != 0:
        raise AssertionError(f"{name} worker failed ({result.returncode}):\n{output}")
    lines = [line for line in output.splitlines() if line.startswith(MARKER)]
    if len(lines) != 1:
        raise AssertionError(f"{name} worker emitted no unique result:\n{output}")
    payload = json.loads(lines[0][len(MARKER):])
    print(f"{name:18s} {payload['after']} result={payload['result']} "
          f"attempt={payload['attempt']} retries={payload['retry_count']}")
    return payload


def parent() -> None:
    legacy = run_case("legacy", {"STIFF_FRAME_GRAPH": "0"})
    graph_a = run_case("graph run A", {"STIFF_FRAME_GRAPH": "1"})
    graph_b = run_case("graph run B", {"STIFF_FRAME_GRAPH": "1"})
    forced = run_case("forced rollback", {
        "STIFF_FRAME_GRAPH": "1", "STIFF_FRAME_FORCE_ROLLBACK": "1"})
    small = run_case("triplet cap=0", {
        "STIFF_FRAME_GRAPH": "1", "STIFF_FRAME_TEST_TRIPLET_CAP": "0"})
    physical_small = run_case("physical cap small", {
        "STIFF_FRAME_GRAPH": "1", "STIFF_FRAME_TEST_BUFF_SCALE": "0.001"})
    nan = run_case("NaN rollback", {
        "STIFF_FRAME_GRAPH": "1", "STIFF_FRAME_TEST_NAN_VERTEX": "0"})

    reference = legacy["after"]
    for name, payload in (("graph A", graph_a), ("graph B", graph_b),
                          ("forced rollback", forced), ("small cap", small),
                          ("physical small cap", physical_small)):
        assert payload["result"] == 0, (name, payload)
        assert payload["after"] == reference, (name, payload)
    assert graph_a["after"] == graph_b["after"]
    assert forced["attempt"] == forced["retry_count"] == 1
    assert small["attempt"] == small["retry_count"] == 1
    assert small["required_triplets"] > 0
    assert small["retry_invalid_bits"] & (1 << 18)
    assert physical_small["attempt"] == physical_small["retry_count"] == 1
    assert physical_small["required_ccd_pairs"] > 0
    assert physical_small["retry_invalid_bits"] & (1 << 17)
    assert graph_a["root_d2h_nodes"] == 0
    assert graph_a["terminal_d2h_nodes"] == 1
    # This fixture has one accepted Newton step followed by the terminal
    # convergence probe.  P3b-1 leaves exactly one phase-only host stitch for
    # each probe; the former grad/CCD/LS decision payloads are not boundaries.
    assert graph_a["newton_iters"] == graph_a["ls_trials"] == 1
    assert graph_a["host_boundaries"] == 2
    assert graph_b["host_boundaries"] == graph_a["host_boundaries"]

    assert nan["result"] == 2
    assert nan["phase"] == 9
    assert nan["error_code"] == 2
    assert nan["invalid_bits"] & (1 << 8)
    assert nan["err_primitive"] == 0
    assert nan["before"] == nan["after"]
    assert nan["error"]

    print("FRAME TRANSACTION: PASS (four-way byte identity + NaN rollback)")
    print(f"GRAPH AUDIT: root D2H=0, terminal D2H=1, "
          f"launches={graph_a['graph_launches']}, "
          f"phase stitches={graph_a['host_boundaries']}, "
          f"newton={graph_a['newton_iters']}, ls={graph_a['ls_trials']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    worker() if args.worker else parent()

#!/usr/bin/env python3
"""Phase-D episode-resident CUDA Graph gate.

The parent compares two clean subprocesses:

* baseline: one audited whole-frame graph launch plus one host boundary/frame;
* episode: one outer graph launch per episode, a device WHILE over every
  frame, two asynchronous pinned-host observation slots, and executable reuse
  across two consecutive episodes.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FRAMES = int(os.environ.get("EPISODE_GATE_FRAMES", "6"))
EPISODES = int(os.environ.get("EPISODE_GATE_EPISODES", "2"))


def make_engine():
    from stiff_physics.engine import Config, Engine

    cfg = Config(
        dt=0.01,
        density=1e3,
        young_modulus=1e6,
        friction_rate=0.4,
        relative_dhat=1e-3,
        absolute_dhat=1e-3,
        ground_offset=0.0,
        assets_dir=os.path.join(ROOT, "Assets") + "/",
        multienv_mode="merged",
        preconditioner_type=1,
        skip_all_collision=True,
    )
    engine = Engine(cfg)
    engine.load_mesh("tetMesh/cube.msh", 3, "FEM", np.eye(4))
    engine.finalize()
    engine.step()  # allocation/lazy-workspace warm-up
    return engine


def digest(positions: np.ndarray, velocities: np.ndarray) -> str:
    payload = np.ascontiguousarray(
        np.concatenate(
            (positions.reshape(-1), velocities.reshape(-1))
        )
    )
    return hashlib.sha256(payload.tobytes()).hexdigest()


def trace_frame_digests(
    label: str, positions: np.ndarray, velocities: np.ndarray
) -> None:
    if os.environ.get("EPISODE_GATE_TRACE") != "1":
        return
    for frame, (position, velocity) in enumerate(
        zip(positions, velocities)
    ):
        print(
            "EPISODE-GRAPH-FRAME-DIGEST: "
            f"{label} {frame} {digest(position, velocity)}"
        )


def baseline() -> None:
    engine = make_engine()
    positions = []
    velocities = []
    for _ in range(FRAMES * EPISODES):
        engine.step()
        status = engine.native.get_frame_status()
        assert status.result == 0
        assert status.path_flags & (1 << 4)
        positions.append(np.asarray(engine.get_vertices()).copy())
        velocities.append(
            np.asarray(engine.native.get_vertex_velocities()).copy()
        )
    positions_array = np.asarray(positions)
    velocities_array = np.asarray(velocities)
    trace_frame_digests(
        "baseline", positions_array, velocities_array
    )
    result = digest(positions_array, velocities_array)
    print(f"EPISODE-GRAPH-DIGEST: {result}")


def episode() -> None:
    engine = make_engine()
    position_chunks = []
    velocity_chunks = []
    for episode_index in range(EPISODES):
        engine.launch_episode_async(FRAMES)
        assert engine.episode_in_flight()

        engine.wait_episode_observation(0)
        slot0 = engine.get_episode_observation(0)
        assert slot0["first_frame"] == 0
        assert slot0["positions"].shape[0] == (FRAMES + 1) // 2

        engine.wait_episode_observation(1)
        slot1 = engine.get_episode_observation(1)
        assert slot1["first_frame"] == (FRAMES + 1) // 2
        assert (
            slot0["positions"].shape[0]
            + slot1["positions"].shape[0]
            == FRAMES
        )

        statuses = list(slot0["statuses"]) + list(slot1["statuses"])
        assert len(statuses) == FRAMES
        first_frame = episode_index * FRAMES
        for local_index, status in enumerate(statuses):
            assert status.result == 0
            assert status.frame_id == first_frame + local_index + 1
            assert status.path_flags & (1 << 4)
            assert status.path_flags & (1 << 8)
            assert status.path_flags & (1 << 9)
            assert status.path_flags & (1 << 10)
            assert not (status.path_flags & (1 << 3))
            assert status.graph_launches == 1
            assert status.host_boundaries == 0
            assert status.root_graph_nodes > 0
            assert status.root_d2h_nodes >= 4
            assert status.terminal_graph_nodes == 0
            assert status.terminal_d2h_nodes == 0

        position_chunks.extend(
            (slot0["positions"], slot1["positions"])
        )
        velocity_chunks.extend(
            (slot0["velocities"], slot1["velocities"])
        )
        assert engine.get_episode_attempted_frame_count() == FRAMES
        assert engine.finish_episode() == FRAMES
        assert not engine.episode_in_flight()

    positions = np.concatenate(position_chunks, axis=0)
    velocities = np.concatenate(velocity_chunks, axis=0)
    assert np.array_equal(np.asarray(engine.get_vertices()), positions[-1])
    trace_frame_digests("episode", positions, velocities)
    result = digest(positions, velocities)
    print(f"EPISODE-GRAPH-DIGEST: {result}")


def child_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("STIFF_MIRROR_AUDIT", None)
    env.pop("STIFF_SLOT_AUDIT", None)
    env["STIFFGIPC_NATIVE_DIR"] = os.environ.get(
        "STIFFGIPC_NATIVE_DIR", os.path.join(ROOT, "build")
    )
    env["STIFF_FRAME_GRAPH"] = "1"
    env["STIFF_FRAME_FULL_GRAPH"] = "1"
    # [C6-q] gate fixtures are tiny; pin the size threshold off so the warmup
    # step exercises the whole-frame graph this gate audits
    env["STIFF_FULL_GRAPH_MIN_VERTS"] = "0"
    env["STIFF_MULTIENV_MODE"] = "merged"
    for key in (
        "STIFF_BVH_ENVDET",
        "STIFF_PERENV_BVH",
        "STIFF_DECOUPLE_THRESH",
        "STIFF_PERGROUP_KAPPA",
        "STIFF_SEGMENTED_PCG",
        "STIFF_PERENV_ALPHA",
        "STIFF_PERENV_PAR",
        "STIFF_EE_CANON",
        "STIFF_EE_DETGATE",
        "STIFF_CCD_CANON",
        "STIFF_SPMV_DET",
        "STIFF_PERENV_MASK",
    ):
        env[key] = "0"
    return env


def run_child(mode: str) -> str:
    completed = subprocess.run(
        [sys.executable, __file__, f"--child={mode}"],
        check=True,
        cwd=ROOT,
        env=child_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(completed.stdout, end="")
    prefix = "EPISODE-GRAPH-DIGEST: "
    return next(
        line[len(prefix):]
        for line in completed.stdout.splitlines()
        if line.startswith(prefix)
    )


if __name__ == "__main__":
    child = next(
        (
            arg.split("=", 1)[1]
            for arg in sys.argv[1:]
            if arg.startswith("--child=")
        ),
        None,
    )
    if child == "baseline":
        baseline()
    elif child == "episode":
        episode()
    else:
        baseline_digest = run_child("baseline")
        episode_digest = run_child("episode")
        assert episode_digest == baseline_digest
        print("EPISODE-GRAPH-GATE: PASS")

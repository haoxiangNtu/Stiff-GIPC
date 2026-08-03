#!/usr/bin/env python3
"""Phase-D articulated RL episode gate.

The baseline applies the public per-joint controls before every synchronous
step.  The episode path uploads the same controls in two episode-sized chunks,
captures only at legal episode boundaries, and compares every position and
velocity returned through the two pinned observation slots.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FRAMES = int(os.environ.get("EPISODE_RL_GATE_FRAMES", "3"))
EPISODES = 2


def transform(z: float) -> np.ndarray:
    value = np.eye(4)
    value[:3, :3] *= 0.4
    value[2, 3] = z
    return value


def make_engine():
    from stiff_physics.engine import Config, Engine

    cfg = Config(
        dt=0.01,
        gravity=(0.0, -9.8, 0.0),
        ground_offset=-100.0,
        assets_dir=os.path.join(ROOT, "Assets") + "/",
        multienv_mode="merged",
        preconditioner_type=1,
        skip_all_collision=True,
    )
    engine = Engine(cfg)
    for z, boundary in (
        (-0.8, "Fixed"),
        (0.0, "Free"),
        (0.8, "Free"),
    ):
        engine.load_mesh(
            "tetMesh/cube.msh",
            dimensions=3,
            body_type="ABD",
            transform=transform(z),
            boundary_type=boundary,
        )
    engine.native.add_revolute_joint(
        0,
        1,
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, -0.4]),
        -100.0,
        100.0,
        0.0,
        "revolute",
    )
    engine.native.add_prismatic_joint(
        1,
        2,
        np.array([0.0, 0.0, 0.2]),
        np.array([0.0, 0.0, 1.0]),
        -2.0,
        2.0,
        "prismatic",
    )
    engine.finalize()
    engine.native.set_log_level(0)
    engine.step()  # train fixed-topology workspaces and ABD layout tiers
    return engine


def actions() -> tuple[np.ndarray, np.ndarray]:
    count = FRAMES * EPISODES
    index = np.arange(1, count + 1, dtype=np.float64)
    sign = np.where((index.astype(np.int64) & 1) == 0, -1.0, 1.0)

    revolute = np.empty((count, 1, 3), dtype=np.float64)
    revolute[:, 0, 0] = 0.0025 * index
    revolute[:, 0, 1] = 0.75 + 0.025 * index
    revolute[:, 0, 2] = 0.001 * sign

    prismatic = np.empty((count, 1, 3), dtype=np.float64)
    prismatic[:, 0, 0] = 0.00025 * index
    prismatic[:, 0, 1] = 0.65 + 0.02 * index
    prismatic[:, 0, 2] = 0.01 * sign
    return revolute, prismatic


def digest(positions: np.ndarray, velocities: np.ndarray) -> str:
    payload = np.ascontiguousarray(
        np.concatenate((positions.reshape(-1), velocities.reshape(-1)))
    )
    return hashlib.sha256(payload.tobytes()).hexdigest()


def emit(positions: list[np.ndarray], velocities: list[np.ndarray]) -> None:
    position_array = np.asarray(positions)
    velocity_array = np.asarray(velocities)
    if os.environ.get("EPISODE_RL_GATE_TRACE") == "1":
        for frame, (position, velocity) in enumerate(
            zip(positions, velocities)
        ):
            print(
                "EPISODE-RL-FRAME-DIGEST: "
                f"{frame} {digest(position, velocity)}"
            )
    dump_path = os.environ.get("EPISODE_RL_GATE_DUMP")
    if dump_path:
        np.savez(
            dump_path,
            positions=position_array,
            velocities=velocity_array,
        )
    result = digest(position_array, velocity_array)
    print(f"EPISODE-RL-GRAPH-DIGEST: {result}")


def baseline() -> None:
    engine = make_engine()
    revolute, prismatic = actions()
    positions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    for frame in range(FRAMES * EPISODES):
        engine.native.set_revolute_target(0, revolute[frame, 0, 0])
        engine.native.set_revolute_strength(0, revolute[frame, 0, 1])
        engine.native.set_revolute_torque(0, revolute[frame, 0, 2])
        engine.native.set_prismatic_target(0, prismatic[frame, 0, 0])
        engine.native.set_prismatic_strength(0, prismatic[frame, 0, 1])
        engine.native.set_prismatic_force(0, prismatic[frame, 0, 2])
        engine.step()
        if os.environ.get("EPISODE_RL_GATE_TRACE") == "1":
            status = engine.native.get_frame_status()
            print(
                "EPISODE-RL-STATUS: baseline "
                f"{frame} {status.newton_iters} {status.pcg_iters}"
            )
        positions.append(np.asarray(engine.get_vertices()).copy())
        velocities.append(
            np.asarray(engine.native.get_vertex_velocities()).copy()
        )
    emit(positions, velocities)


def episode() -> None:
    engine = make_engine()
    revolute, prismatic = actions()
    positions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []

    for episode_index in range(EPISODES):
        first = episode_index * FRAMES
        last = first + FRAMES
        engine.launch_episode_async(
            FRAMES, revolute[first:last], prismatic[first:last]
        )
        assert engine.episode_in_flight()

        observed = 0
        for slot in (0, 1):
            engine.wait_episode_observation(slot)
            observation = engine.get_episode_observation(slot)
            assert observation["first_frame"] == observed
            slot_positions = observation["positions"]
            slot_velocities = observation["velocities"]
            statuses = list(observation["statuses"])
            assert slot_positions.shape[0] == len(statuses)
            for local, status in enumerate(statuses):
                expected_frame = first + observed + local + 1
                assert status.result == 0
                assert status.frame_id == expected_frame
                assert status.path_flags & (1 << 4)
                assert status.path_flags & (1 << 8)
                assert status.path_flags & (1 << 9)
                assert status.path_flags & (1 << 10)
                assert status.graph_launches == 1
                assert status.host_boundaries == 0
                if os.environ.get("EPISODE_RL_GATE_TRACE") == "1":
                    print(
                        "EPISODE-RL-STATUS: episode "
                        f"{first + observed + local} "
                        f"{status.newton_iters} {status.pcg_iters}"
                    )
            positions.extend(np.asarray(slot_positions))
            velocities.extend(np.asarray(slot_velocities))
            observed += len(statuses)

        assert observed == FRAMES
        assert engine.get_episode_attempted_frame_count() == FRAMES
        assert engine.finish_episode() == FRAMES
        assert not engine.episode_in_flight()

    assert len(positions) == FRAMES * EPISODES
    assert np.array_equal(
        np.asarray(engine.get_vertices()), np.asarray(positions[-1])
    )
    emit(positions, velocities)


def child_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("STIFF_MIRROR_AUDIT", None)
    env.pop("STIFF_SLOT_AUDIT", None)
    env["STIFFGIPC_NATIVE_DIR"] = os.environ.get(
        "STIFFGIPC_NATIVE_DIR", os.path.join(ROOT, "build")
    )
    env["STIFF_FRAME_GRAPH"] = "1"
    env["STIFF_FRAME_FULL_GRAPH"] = "1"
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


def run_child(mode: str, dump_path: str) -> str:
    env = child_environment()
    env["EPISODE_RL_GATE_DUMP"] = dump_path
    completed = subprocess.run(
        [sys.executable, __file__, f"--child={mode}"],
        check=True,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(completed.stdout, end="")
    prefix = "EPISODE-RL-GRAPH-DIGEST: "
    return next(
        line[len(prefix) :]
        for line in completed.stdout.splitlines()
        if line.startswith(prefix)
    )


def load_observations(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {
            "positions": np.asarray(archive["positions"]).copy(),
            "velocities": np.asarray(archive["velocities"]).copy(),
        }


# dt of the fixture scene (gpu_rl_gate.make_engine); the velocity floor
# below is the position floor amplified by this backward difference.
SCENE_DT = 0.01


def assert_numerical_equivalence(
    baseline_a: dict[str, np.ndarray],
    baseline_b: dict[str, np.ndarray],
    episode_result: dict[str, np.ndarray],
) -> None:
    for field in ("positions", "velocities"):
        first = baseline_a[field]
        second = baseline_b[field]
        observed = episode_result[field]
        assert first.shape == second.shape == observed.shape
        assert np.isfinite(first).all()
        assert np.isfinite(second).all()
        assert np.isfinite(observed).all()

        baseline_noise = float(np.max(np.abs(first - second)))
        episode_error = min(
            float(np.max(np.abs(observed - first))),
            float(np.max(np.abs(observed - second))),
        )
        scale = max(
            1.0,
            float(np.max(np.abs(first))),
            float(np.max(np.abs(second))),
        )
        precision_floor = 16.0 * np.finfo(first.dtype).eps * scale
        if field == "velocities":
            # [C6-y] Velocities are a backward difference of positions over
            # dt, so whatever last-ulp noise the positions carry arrives here
            # multiplied by 1/dt (dt=0.01 in this fixture -> x100). Applying
            # the same raw precision floor to both fields therefore holds the
            # velocity field to 1/100th of the position field's standard --
            # visible in every passing run, where the two errors differ by
            # exactly 100x (2.776e-17 vs 2.776e-15). collision_graph_gate
            # already carries this correction ([C6-j A800], same reasoning);
            # this gate was missing it.
            precision_floor /= SCENE_DT
        tolerance = max(8.0 * baseline_noise, precision_floor)
        assert episode_error <= tolerance, (
            f"{field} differs beyond the baseline nondeterminism envelope: "
            f"error={episode_error:.17g}, baseline_noise="
            f"{baseline_noise:.17g}, tolerance={tolerance:.17g}"
        )
        print(
            "EPISODE-RL-NUMERICS: "
            f"{field} error={episode_error:.3e} "
            f"baseline_noise={baseline_noise:.3e} "
            f"tolerance={tolerance:.3e}"
        )


if __name__ == "__main__":
    child = next(
        (
            argument.split("=", 1)[1]
            for argument in sys.argv[1:]
            if argument.startswith("--child=")
        ),
        None,
    )
    if child == "baseline":
        baseline()
    elif child == "episode":
        episode()
    else:
        with tempfile.TemporaryDirectory(prefix="stiff-episode-rl-gate-") as tmp:
            baseline_a_path = os.path.join(tmp, "baseline-a.npz")
            baseline_b_path = os.path.join(tmp, "baseline-b.npz")
            episode_path = os.path.join(tmp, "episode.npz")
            run_child("baseline", baseline_a_path)
            run_child("baseline", baseline_b_path)
            run_child("episode", episode_path)
            assert_numerical_equivalence(
                load_observations(baseline_a_path),
                load_observations(baseline_b_path),
                load_observations(episode_path),
            )
        print("EPISODE-RL-GRAPH-GATE: PASS")

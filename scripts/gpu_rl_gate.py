#!/usr/bin/env python3
"""GPU-native RL device-ABI gate.

The Phase-D episode path still pays one H2D per episode and streams
observations back through pinned slots.  GPU-native RL mode removes even
that: an external CUDA agent (here: cudart memcpys standing in for a GPU
policy) writes the packed action buffers in device memory, launches the
reusable one-frame graph on its own stream, and reads observations from
device pointers.  The steady-state graph must audit to zero host nodes,
zero H2D nodes and zero D2H nodes.

Phase 1 (closed-loop parity) synchronizes after every launch and compares
each frame against the public per-joint-control baseline within the
baseline's own nondeterminism envelope.  Phase 2 (zero-sync residency)
enqueues action writes and launches back-to-back with no host wait until
one final event synchronization, then checks the device frame counter and
the terminal state.  Lifecycle guards (step() lockout, stream affinity,
end_gpu_rl() recovery) are asserted along the way.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FRAMES = int(os.environ.get("GPU_RL_GATE_FRAMES", "3"))  # per phase
TOTAL = FRAMES * 2

# Mirrors frame_fsm::FrameStatus (frame_status.cuh).  alignas(16) pads the
# 200 payload bytes to 208; the ABI reports the padded size.
STATUS_BYTES = 208
STATUS_RESULT_OFFSET = 0
STATUS_PATH_FLAGS_OFFSET = 36
STATUS_HOST_BOUNDARIES_OFFSET = 44
STATUS_NEWTON_OFFSET = 52
STATUS_PCG_OFFSET = 56
STATUS_FRAME_ID_OFFSET = 176

PATH_FULL_CONDITIONAL_GRAPH = 1 << 4
PATH_PCG_DEVICE_CONTINUATION = 1 << 8
PATH_LS_DEVICE_LOOP = 1 << 9
PATH_EPISODE_RESIDENT = 1 << 10
PATH_GPU_NATIVE_RL = 1 << 11

MEMCPY_H2D = 1
MEMCPY_D2H = 2


class Cudart:
    """Minimal cudart surface standing in for the external GPU policy."""

    def __init__(self) -> None:
        candidates = [
            "libcudart.so.12",
            "libcudart.so",
            "/usr/local/cuda/lib64/libcudart.so.12",
        ]
        found = ctypes.util.find_library("cudart")
        if found:
            candidates.insert(0, found)
        last_error: Exception | None = None
        for name in candidates:
            try:
                self.lib = ctypes.CDLL(name)
                break
            except OSError as error:
                last_error = error
        else:
            raise RuntimeError(f"could not load cudart: {last_error}")
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p

    def check(self, status: int, what: str) -> None:
        if status != 0:
            message = self.lib.cudaGetErrorString(status).decode()
            raise RuntimeError(f"{what} failed: {message} ({status})")

    def stream_create(self) -> int:
        stream = ctypes.c_void_p()
        # cudaStreamNonBlocking: prove there is no hidden legacy-stream
        # coupling in the steady state.
        self.check(
            self.lib.cudaStreamCreateWithFlags(ctypes.byref(stream), 1),
            "cudaStreamCreateWithFlags",
        )
        return stream.value or 0

    def stream_destroy(self, stream: int) -> None:
        self.check(
            self.lib.cudaStreamDestroy(ctypes.c_void_p(stream)),
            "cudaStreamDestroy",
        )

    def memcpy(self, dst: int, src: int, size: int, kind: int) -> None:
        self.check(
            self.lib.cudaMemcpy(
                ctypes.c_void_p(dst),
                ctypes.c_void_p(src),
                ctypes.c_size_t(size),
                ctypes.c_int(kind),
            ),
            "cudaMemcpy",
        )

    def memcpy_async(
        self, dst: int, src: int, size: int, kind: int, stream: int
    ) -> None:
        self.check(
            self.lib.cudaMemcpyAsync(
                ctypes.c_void_p(dst),
                ctypes.c_void_p(src),
                ctypes.c_size_t(size),
                ctypes.c_int(kind),
                ctypes.c_void_p(stream),
            ),
            "cudaMemcpyAsync",
        )

    def read_doubles(self, pointer: int, count: int) -> np.ndarray:
        out = np.empty(count, dtype=np.float64)
        self.memcpy(
            out.ctypes.data, pointer, out.nbytes, MEMCPY_D2H
        )
        return out

    def read_bytes(self, pointer: int, count: int) -> bytes:
        out = (ctypes.c_ubyte * count)()
        self.memcpy(
            ctypes.addressof(out), pointer, count, MEMCPY_D2H
        )
        return bytes(out)

    def read_int64(self, pointer: int) -> int:
        value = ctypes.c_int64()
        self.memcpy(
            ctypes.addressof(value), pointer, 8, MEMCPY_D2H
        )
        return value.value


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
    index = np.arange(1, TOTAL + 1, dtype=np.float64)
    sign = np.where((index.astype(np.int64) & 1) == 0, -1.0, 1.0)

    revolute = np.empty((TOTAL, 1, 3), dtype=np.float64)
    revolute[:, 0, 0] = 0.0025 * index
    revolute[:, 0, 1] = 0.75 + 0.025 * index
    revolute[:, 0, 2] = 0.001 * sign

    prismatic = np.empty((TOTAL, 1, 3), dtype=np.float64)
    prismatic[:, 0, 0] = 0.00025 * index
    prismatic[:, 0, 1] = 0.65 + 0.02 * index
    prismatic[:, 0, 2] = 0.01 * sign
    return revolute, prismatic


def expect_raises(what: str, callback, *fragments: str) -> None:
    try:
        callback()
    except Exception as error:  # noqa: BLE001 - message-matched below
        text = str(error)
        for fragment in fragments:
            assert fragment in text, (
                f"{what}: expected {fragment!r} in {text!r}"
            )
        return
    raise AssertionError(f"{what}: expected an exception")


def parse_status(blob: bytes) -> dict[str, int]:
    return {
        "result": struct.unpack_from("<i", blob, STATUS_RESULT_OFFSET)[0],
        "path_flags": struct.unpack_from(
            "<I", blob, STATUS_PATH_FLAGS_OFFSET
        )[0],
        "host_boundaries": struct.unpack_from(
            "<i", blob, STATUS_HOST_BOUNDARIES_OFFSET
        )[0],
        "newton_iters": struct.unpack_from(
            "<i", blob, STATUS_NEWTON_OFFSET
        )[0],
        "pcg_iters": struct.unpack_from("<i", blob, STATUS_PCG_OFFSET)[0],
        "frame_id": struct.unpack_from(
            "<q", blob, STATUS_FRAME_ID_OFFSET
        )[0],
    }


def baseline(dump_path: str) -> None:
    engine = make_engine()
    revolute, prismatic = actions()
    positions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    for frame in range(TOTAL):
        engine.native.set_revolute_target(0, revolute[frame, 0, 0])
        engine.native.set_revolute_strength(0, revolute[frame, 0, 1])
        engine.native.set_revolute_torque(0, revolute[frame, 0, 2])
        engine.native.set_prismatic_target(0, prismatic[frame, 0, 0])
        engine.native.set_prismatic_strength(0, prismatic[frame, 0, 1])
        engine.native.set_prismatic_force(0, prismatic[frame, 0, 2])
        engine.step()
        positions.append(np.asarray(engine.get_vertices()).copy())
        velocities.append(
            np.asarray(engine.native.get_vertex_velocities()).copy()
        )
    np.savez(
        dump_path,
        positions=np.asarray(positions),
        velocities=np.asarray(velocities),
    )
    print("GPU-RL-BASELINE: done")


def gpu_native(dump_path: str) -> None:
    engine = make_engine()
    native = engine.native
    revolute, prismatic = actions()
    cuda = Cudart()

    engine.prepare_gpu_rl()
    assert engine.gpu_rl_prepared()
    assert not engine.gpu_rl_ready()

    abi = engine.get_gpu_rl_device_abi()
    print(
        "GPU-RL-ABI: nodes={graph_nodes} h2d={graph_h2d} d2h={graph_d2h} "
        "revolute={revolute_joints} prismatic={prismatic_joints} "
        "vertices={vertices}".format(**abi)
    )
    assert abi["graph_nodes"] > 0
    assert abi["graph_h2d"] == 0, "steady-state graph must contain no H2D"
    assert abi["graph_d2h"] == 0, "steady-state graph must contain no D2H"
    assert abi["action_dtype"] == "float64"
    assert abi["action_components"] == 3
    assert abi["revolute_joints"] == 1
    assert abi["prismatic_joints"] == 1
    assert abi["status_bytes"] == STATUS_BYTES
    vertex_count = abi["vertices"]
    assert vertex_count == native.get_vertex_count()

    # While GPU-native mode owns the device, the synchronous frame entry
    # points must refuse to run.
    expect_raises(
        "step lockout", engine.step, "GPU-native RL", "end_gpu_rl"
    )
    expect_raises(
        "episode lockout",
        lambda: engine.launch_episode_async(
            1, revolute[:1], prismatic[:1]
        ),
        "GPU-native RL",
    )

    stream = cuda.stream_create()
    action_bytes = 3 * 8

    def write_actions(frame: int, asynchronous: bool) -> None:
        revolute_row = np.ascontiguousarray(revolute[frame, 0])
        prismatic_row = np.ascontiguousarray(prismatic[frame, 0])
        if asynchronous:
            cuda.memcpy_async(
                abi["revolute_actions"],
                revolute_row.ctypes.data,
                action_bytes,
                MEMCPY_H2D,
                stream,
            )
            cuda.memcpy_async(
                abi["prismatic_actions"],
                prismatic_row.ctypes.data,
                action_bytes,
                MEMCPY_H2D,
                stream,
            )
        else:
            cuda.memcpy(
                abi["revolute_actions"],
                revolute_row.ctypes.data,
                action_bytes,
                MEMCPY_H2D,
            )
            cuda.memcpy(
                abi["prismatic_actions"],
                prismatic_row.ctypes.data,
                action_bytes,
                MEMCPY_H2D,
            )

    def read_observation() -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        flat = cuda.read_doubles(abi["positions"], vertex_count * 3)
        velocity = cuda.read_doubles(abi["velocities"], vertex_count * 3)
        status = parse_status(
            cuda.read_bytes(abi["statuses"], STATUS_BYTES)
        )
        return (
            flat.reshape(vertex_count, 3),
            velocity.reshape(vertex_count, 3),
            status,
        )

    # ---- Phase 1: closed-loop parity, one synchronization per frame ----
    positions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    for frame in range(FRAMES):
        write_actions(frame, asynchronous=False)
        engine.launch_gpu_rl_async(stream)
        engine.synchronize_gpu_rl()
        assert engine.gpu_rl_ready()
        frame_positions, frame_velocities, status = read_observation()
        assert status["result"] == 0, f"frame {frame}: {status}"
        for bit in (
            PATH_FULL_CONDITIONAL_GRAPH,
            PATH_PCG_DEVICE_CONTINUATION,
            PATH_LS_DEVICE_LOOP,
            PATH_EPISODE_RESIDENT,
            PATH_GPU_NATIVE_RL,
        ):
            assert status["path_flags"] & bit, hex(status["path_flags"])
        assert status["host_boundaries"] == 0
        assert status["frame_id"] == frame
        assert cuda.read_int64(abi["frame_counter"]) == frame + 1
        print(
            "GPU-RL-STATUS: parity "
            f"{frame} {status['newton_iters']} {status['pcg_iters']}"
        )
        positions.append(frame_positions)
        velocities.append(frame_velocities)

    # ---- Phase 2: zero-sync residency ----
    for frame in range(FRAMES, TOTAL):
        write_actions(frame, asynchronous=True)
        engine.launch_gpu_rl_async(stream)
    engine.synchronize_gpu_rl()
    final_positions, final_velocities, status = read_observation()
    assert status["result"] == 0, f"final: {status}"
    assert status["path_flags"] & PATH_GPU_NATIVE_RL
    assert status["frame_id"] == TOTAL - 1
    assert cuda.read_int64(abi["frame_counter"]) == TOTAL
    print(
        "GPU-RL-STATUS: final "
        f"{TOTAL - 1} {status['newton_iters']} {status['pcg_iters']}"
    )

    # The stored observation must be the engine's committed state.
    assert np.array_equal(
        np.asarray(engine.get_vertices()), final_positions
    )

    # Stream affinity is part of the ABI: a second stream must be refused.
    other = cuda.stream_create()
    expect_raises(
        "stream affinity",
        lambda: engine.launch_gpu_rl_async(other),
        "same CUDA stream",
    )
    cuda.stream_destroy(other)

    engine.end_gpu_rl()
    assert not engine.gpu_rl_prepared()
    cuda.stream_destroy(stream)

    # The engine must come back alive as an ordinary synchronous stepper.
    engine.step()
    assert np.isfinite(np.asarray(engine.get_vertices())).all()
    print("GPU-RL-RECOVERY: post-end step ok")

    np.savez(
        dump_path,
        positions=np.asarray(positions),
        velocities=np.asarray(velocities),
        final_positions=final_positions,
        final_velocities=final_velocities,
    )
    print("GPU-RL-NATIVE: done")


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


def run_child(mode: str, dump_path: str) -> None:
    completed = subprocess.run(
        [sys.executable, __file__, f"--child={mode}", dump_path],
        check=True,
        cwd=ROOT,
        env=child_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(completed.stdout, end="")


def load(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]).copy() for key in archive}


def envelope_compare(
    label: str,
    first: np.ndarray,
    second: np.ndarray,
    observed: np.ndarray,
) -> None:
    assert first.shape == second.shape == observed.shape
    assert np.isfinite(first).all()
    assert np.isfinite(second).all()
    assert np.isfinite(observed).all()
    baseline_noise = float(np.max(np.abs(first - second)))
    error = min(
        float(np.max(np.abs(observed - first))),
        float(np.max(np.abs(observed - second))),
    )
    scale = max(
        1.0, float(np.max(np.abs(first))), float(np.max(np.abs(second)))
    )
    precision_floor = 16.0 * np.finfo(first.dtype).eps * scale
    tolerance = max(8.0 * baseline_noise, precision_floor)
    assert error <= tolerance, (
        f"{label} differs beyond the baseline nondeterminism envelope: "
        f"error={error:.17g}, baseline_noise={baseline_noise:.17g}, "
        f"tolerance={tolerance:.17g}"
    )
    print(
        f"GPU-RL-NUMERICS: {label} error={error:.3e} "
        f"baseline_noise={baseline_noise:.3e} tolerance={tolerance:.3e}"
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
        baseline(sys.argv[2])
    elif child == "native":
        gpu_native(sys.argv[2])
    else:
        with tempfile.TemporaryDirectory(prefix="stiff-gpu-rl-gate-") as tmp:
            baseline_a_path = os.path.join(tmp, "baseline-a.npz")
            baseline_b_path = os.path.join(tmp, "baseline-b.npz")
            native_path = os.path.join(tmp, "native.npz")
            run_child("baseline", baseline_a_path)
            run_child("baseline", baseline_b_path)
            run_child("native", native_path)
            baseline_a = load(baseline_a_path)
            baseline_b = load(baseline_b_path)
            native_result = load(native_path)
            for field in ("positions", "velocities"):
                envelope_compare(
                    f"parity {field}",
                    baseline_a[field][:FRAMES],
                    baseline_b[field][:FRAMES],
                    native_result[field],
                )
                envelope_compare(
                    f"final {field}",
                    baseline_a[field][TOTAL - 1],
                    baseline_b[field][TOTAL - 1],
                    native_result[f"final_{field}"],
                )
        print("GPU-RL-GATE: PASS")

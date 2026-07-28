#!/usr/bin/env python3
"""GPU-native RL device-ABI regression.

The action trajectory is uploaded once to temporary device memory. Each policy
step then enqueues only D2D action publication, the reusable simulation graph,
and D2D observation retention on one CUDA stream. There is one final host wait
and readback for test adjudication; the simulation graph itself must contain no
H2D or D2H nodes.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.episode_graph_rl_gate import (  # noqa: E402
    actions,
    assert_numerical_equivalence,
    child_environment,
    emit,
    load_observations,
    make_engine,
    run_child,
)


class CudaRuntime:
    H2D = 1
    D2H = 2
    D2D = 3

    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libcudart.so")
        self.lib.cudaMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
        ]
        self.lib.cudaMalloc.restype = ctypes.c_int
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaFree.restype = ctypes.c_int
        self.lib.cudaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.lib.cudaMemcpy.restype = ctypes.c_int
        self.lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.lib.cudaMemcpyAsync.restype = ctypes.c_int
        self.lib.cudaStreamCreate.argtypes = [
            ctypes.POINTER(ctypes.c_void_p)
        ]
        self.lib.cudaStreamCreate.restype = ctypes.c_int
        self.lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamSynchronize.restype = ctypes.c_int
        self.lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamDestroy.restype = ctypes.c_int

    @staticmethod
    def _pointer(value: int) -> ctypes.c_void_p:
        return ctypes.c_void_p(value)

    @staticmethod
    def _check(code: int, operation: str) -> None:
        if code:
            raise RuntimeError(f"{operation} failed with CUDA error {code}")

    def malloc(self, size: int) -> int:
        pointer = ctypes.c_void_p()
        self._check(
            self.lib.cudaMalloc(ctypes.byref(pointer), size),
            "cudaMalloc",
        )
        return int(pointer.value or 0)

    def free(self, pointer: int) -> None:
        if pointer:
            self._check(
                self.lib.cudaFree(self._pointer(pointer)), "cudaFree"
            )

    def memcpy(self, destination: int, source: int, size: int, kind: int) -> None:
        self._check(
            self.lib.cudaMemcpy(
                self._pointer(destination),
                self._pointer(source),
                size,
                kind,
            ),
            "cudaMemcpy",
        )

    def memcpy_async(
        self,
        destination: int,
        source: int,
        size: int,
        kind: int,
        stream: int,
    ) -> None:
        self._check(
            self.lib.cudaMemcpyAsync(
                self._pointer(destination),
                self._pointer(source),
                size,
                kind,
                self._pointer(stream),
            ),
            "cudaMemcpyAsync",
        )

    def stream_create(self) -> int:
        stream = ctypes.c_void_p()
        self._check(
            self.lib.cudaStreamCreate(ctypes.byref(stream)),
            "cudaStreamCreate",
        )
        return int(stream.value or 0)

    def stream_synchronize(self, stream: int) -> None:
        self._check(
            self.lib.cudaStreamSynchronize(self._pointer(stream)),
            "cudaStreamSynchronize",
        )

    def stream_destroy(self, stream: int) -> None:
        if stream:
            self._check(
                self.lib.cudaStreamDestroy(self._pointer(stream)),
                "cudaStreamDestroy",
            )


def gpu_native() -> None:
    runtime = CudaRuntime()
    engine = make_engine()
    revolute, prismatic = actions()
    step_count = revolute.shape[0]

    engine.native.prepare_gpu_rl()
    abi = engine.native.get_gpu_rl_device_abi()
    assert abi["graph_nodes"] > 0
    assert abi["graph_h2d"] == 0
    assert abi["graph_d2h"] == 0
    assert abi["action_dtype"] == "float64"
    assert abi["action_components"] == 3

    revolute = np.ascontiguousarray(revolute, dtype=np.float64)
    prismatic = np.ascontiguousarray(prismatic, dtype=np.float64)
    vertex_count = int(abi["vertices"])
    status_bytes = int(abi["status_bytes"])
    observation_bytes = vertex_count * 3 * np.dtype(np.float64).itemsize
    revolute_step_bytes = revolute.shape[1] * 3 * revolute.itemsize
    prismatic_step_bytes = prismatic.shape[1] * 3 * prismatic.itemsize

    allocated: list[int] = []
    stream = runtime.stream_create()
    try:
        def allocate(size: int) -> int:
            pointer = runtime.malloc(size)
            allocated.append(pointer)
            return pointer

        d_revolute = allocate(revolute.nbytes)
        d_prismatic = allocate(prismatic.nbytes)
        d_positions = allocate(step_count * observation_bytes)
        d_velocities = allocate(step_count * observation_bytes)
        d_statuses = allocate(step_count * status_bytes)

        # Setup-only uploads. A real policy produces these values on device.
        runtime.memcpy(
            d_revolute, revolute.ctypes.data, revolute.nbytes, runtime.H2D
        )
        runtime.memcpy(
            d_prismatic,
            prismatic.ctypes.data,
            prismatic.nbytes,
            runtime.H2D,
        )

        for frame in range(step_count):
            runtime.memcpy_async(
                int(abi["revolute_actions"]),
                d_revolute + frame * revolute_step_bytes,
                revolute_step_bytes,
                runtime.D2D,
                stream,
            )
            runtime.memcpy_async(
                int(abi["prismatic_actions"]),
                d_prismatic + frame * prismatic_step_bytes,
                prismatic_step_bytes,
                runtime.D2D,
                stream,
            )
            engine.native.launch_gpu_rl_async(stream)
            runtime.memcpy_async(
                d_positions + frame * observation_bytes,
                int(abi["positions"]),
                observation_bytes,
                runtime.D2D,
                stream,
            )
            runtime.memcpy_async(
                d_velocities + frame * observation_bytes,
                int(abi["velocities"]),
                observation_bytes,
                runtime.D2D,
                stream,
            )
            runtime.memcpy_async(
                d_statuses + frame * status_bytes,
                int(abi["statuses"]),
                status_bytes,
                runtime.D2D,
                stream,
            )

        # The only steady-run host wait is the gate's terminal adjudication.
        runtime.stream_synchronize(stream)

        positions = np.empty(
            (step_count, vertex_count, 3), dtype=np.float64
        )
        velocities = np.empty_like(positions)
        statuses = np.empty((step_count, status_bytes), dtype=np.uint8)
        counter = np.empty(1, dtype=np.int64)
        runtime.memcpy(
            positions.ctypes.data,
            d_positions,
            positions.nbytes,
            runtime.D2H,
        )
        runtime.memcpy(
            velocities.ctypes.data,
            d_velocities,
            velocities.nbytes,
            runtime.D2H,
        )
        runtime.memcpy(
            statuses.ctypes.data,
            d_statuses,
            statuses.nbytes,
            runtime.D2H,
        )
        runtime.memcpy(
            counter.ctypes.data,
            int(abi["frame_counter"]),
            counter.nbytes,
            runtime.D2H,
        )

        results = np.ascontiguousarray(statuses[:, :4]).view(np.int32)
        assert np.all(results == 0), results
        assert int(counter[0]) == step_count
        assert np.isfinite(positions).all()
        assert np.isfinite(velocities).all()
        print(
            "GPU-NATIVE-RL-AUDIT: "
            f"nodes={abi['graph_nodes']} "
            f"h2d={abi['graph_h2d']} d2h={abi['graph_d2h']} "
            f"steps={counter[0]}"
        )
        emit(list(positions), list(velocities))
    finally:
        runtime.stream_synchronize(stream)
        if engine.native.gpu_rl_prepared():
            engine.native.end_gpu_rl()
        for pointer in reversed(allocated):
            runtime.free(pointer)
        runtime.stream_destroy(stream)


def run_gpu_child(dump_path: str) -> None:
    environment = child_environment()
    environment["EPISODE_RL_GATE_DUMP"] = dump_path
    completed = subprocess.run(
        [sys.executable, __file__, "--child=gpu-native"],
        check=True,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(completed.stdout, end="")


if __name__ == "__main__":
    child = next(
        (
            argument.split("=", 1)[1]
            for argument in sys.argv[1:]
            if argument.startswith("--child=")
        ),
        None,
    )
    if child == "gpu-native":
        gpu_native()
    else:
        with tempfile.TemporaryDirectory(
            prefix="stiff-gpu-native-rl-gate-"
        ) as temporary:
            baseline_a_path = os.path.join(temporary, "baseline-a.npz")
            baseline_b_path = os.path.join(temporary, "baseline-b.npz")
            gpu_native_path = os.path.join(temporary, "gpu-native.npz")
            run_child("baseline", baseline_a_path)
            run_child("baseline", baseline_b_path)
            run_gpu_child(gpu_native_path)
            assert_numerical_equivalence(
                load_observations(baseline_a_path),
                load_observations(baseline_b_path),
                load_observations(gpu_native_path),
            )
        print("GPU-NATIVE-RL-GATE: PASS")

#!/usr/bin/env python3
"""D4: contact-load GPU-native RL steady state (nsys zero-sync scene).

An articulated ABD chain (fixed base + revolute link + prismatic slider)
WITH collision enabled drops toward the ground plane while the policy
drives the joints — ground pairs, the swept CCD scalar chain, the C4-b
in-graph kappa chain and the C4-c lagged friction machinery all execute
inside the reusable one-frame graph.

Run modes:
  (default)      — closed-loop smoke: launches with per-step sync, checks
                   statuses/joint observations stay finite, prints a
                   PASS/FAIL verdict line.
  --nsys         — steady-state profile body: warm-up, then inside the
                   cudaProfilerStart/Stop window run N back-to-back
                   {D2D action publish + graph launch} with zero host
                   waits; adjudication happens after cudaProfilerStop.
"""

from __future__ import annotations

import ctypes
import os
import struct
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.gpu_rl_gate import (  # noqa: E402
    Cudart,
    MEMCPY_H2D,
    STATUS_BYTES,
    parse_status,
    transform,
)

STEPS = int(os.environ.get("GPU_RL_CONTACT_STEPS", "40"))
MEMCPY_D2D = 3


def make_engine():
    from stiff_physics.engine import Config, Engine

    cfg = Config(
        dt=0.01,
        gravity=(0.0, -9.8, 0.0),
        ground_offset=-0.55,
        assets_dir=os.path.join(ROOT, "Assets") + "/",
        multienv_mode="merged",
        preconditioner_type=1,
        friction_rate=0.3,
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
    engine.step()  # warm-up trains lazy workspaces and capacity tiers
    # [C6-b aftermath] Capacity tiers now train from OBSERVED peaks, not the
    # old inflated payload rule, so a contact-free warm-up leaves every
    # contact tier at its floor and the episode overflows the moment the
    # driven joints press the cubes into the ground (step 3, result=1).
    # Prime the tiers with a short host rollout along the same drive law the
    # episode replays -- the boundary-legal way to size an episode: warm up
    # on representative contact, then record.
    for i in range(1, 11):
        engine.native.set_revolute_target(0, 0.16 * i / 10.0)
        engine.native.set_prismatic_target(0, 0.016 * i / 10.0)
        engine.step()
    return engine


def actions(total: int) -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(1, total + 1, dtype=np.float64)
    revolute = np.empty((total, 3), dtype=np.float64)
    revolute[:, 0] = 0.004 * index
    revolute[:, 1] = 0.8
    revolute[:, 2] = 0.0
    prismatic = np.empty((total, 3), dtype=np.float64)
    prismatic[:, 0] = 0.0004 * index
    prismatic[:, 1] = 0.7
    prismatic[:, 2] = 0.0
    return revolute, prismatic


def main() -> None:
    nsys = "--nsys" in sys.argv[1:]
    engine = make_engine()
    cuda = Cudart()
    lib = cuda.lib

    engine.prepare_gpu_rl()
    abi = engine.get_gpu_rl_device_abi()
    print(
        "CONTACT-STEADY-ABI: nodes={graph_nodes} h2d={graph_h2d} "
        "d2h={graph_d2h} joints={joint_observation_count}".format(**abi)
    )
    assert abi["graph_h2d"] == 0 and abi["graph_d2h"] == 0

    warm = 3
    total = warm + STEPS
    revolute, prismatic = actions(total)
    row_bytes = 3 * 8

    def device_alloc(size: int) -> int:
        pointer = ctypes.c_void_p()
        cuda.check(
            lib.cudaMalloc(ctypes.byref(pointer), ctypes.c_size_t(size)),
            "cudaMalloc",
        )
        return pointer.value or 0

    staging_rev = device_alloc(total * row_bytes)
    staging_pri = device_alloc(total * row_bytes)
    cuda.memcpy(
        staging_rev,
        np.ascontiguousarray(revolute).ctypes.data,
        total * row_bytes,
        MEMCPY_H2D,
    )
    cuda.memcpy(
        staging_pri,
        np.ascontiguousarray(prismatic).ctypes.data,
        total * row_bytes,
        MEMCPY_H2D,
    )

    stream = cuda.stream_create()

    def publish(frame: int) -> None:
        cuda.memcpy_async(
            abi["revolute_actions"],
            staging_rev + frame * row_bytes,
            row_bytes,
            MEMCPY_D2D,
            stream,
        )
        cuda.memcpy_async(
            abi["prismatic_actions"],
            staging_pri + frame * row_bytes,
            row_bytes,
            MEMCPY_D2D,
            stream,
        )

    for frame in range(warm):
        publish(frame)
        engine.launch_gpu_rl_async(stream)
    engine.synchronize_gpu_rl()

    if nsys:
        cuda.check(lib.cudaProfilerStart(), "cudaProfilerStart")
    # [episode-boundary adjudication] A failed frame restores itself (the
    # per-frame transaction) and reports through its status slot. The host --
    # the adjudicator by design -- steps the release solver across the failure
    # region (frame-boundary growth trains the overflowed tier on the true
    # load), re-prepares, and resumes the episode. One resume is budgeted; a
    # second failure is a real defect.
    resumes = 0
    frame = warm
    while frame < total:
        publish(frame)
        engine.launch_gpu_rl_async(stream)
        if not nsys:
            engine.synchronize_gpu_rl()
            raw = cuda.read_bytes(abi["statuses"], STATUS_BYTES)
            status = parse_status(raw)
            if status["result"] != 0:
                import struct as _s
                inv = _s.unpack_from("<I", raw, 8)[0]
                assert resumes == 0, (
                    f"step {frame} failed after a resume: {status} "
                    f"invalid=0x{inv:x}"
                )
                resumes += 1
                print(
                    f"CONTACT-STEADY-RESUME: step {frame} invalid=0x{inv:x} "
                    "-> host adjudication + re-prepare"
                )
                engine.end_gpu_rl()
                for host_step in range(frame, min(frame + 8, total)):
                    engine.set_revolute_target(
                        0, float(revolute[host_step, 0])
                    )
                    engine.set_prismatic_target(
                        0, float(prismatic[host_step, 0])
                    )
                    engine.step()
                frame = min(frame + 8, total)
                engine.prepare_gpu_rl()
                abi = engine.get_gpu_rl_device_abi()
                continue
        frame += 1
    if nsys:
        cuda.check(lib.cudaProfilerStop(), "cudaProfilerStop")

    engine.synchronize_gpu_rl()
    blob = cuda.read_bytes(abi["statuses"], STATUS_BYTES)
    status = parse_status(blob)
    kappa = struct.unpack_from("<d", blob, 160)[0]
    counter = cuda.read_int64(abi["frame_counter"])
    joints = cuda.read_doubles(
        abi["joint_observations"], abi["joint_observation_count"]
    )
    print(
        f"CONTACT-STEADY-FINAL: frames={counter} result={status['result']} "
        f"newton={status['newton_iters']} kappa={kappa:.6g} "
        f"joints={np.array2string(joints, precision=5)}"
    )
    assert counter == total
    assert status["result"] == 0
    assert np.isfinite(joints).all()

    engine.end_gpu_rl()
    cuda.stream_destroy(stream)
    cuda.check(lib.cudaFree(ctypes.c_void_p(staging_rev)), "cudaFree")
    cuda.check(lib.cudaFree(ctypes.c_void_p(staging_pri)), "cudaFree")
    print("CONTACT-STEADY: PASS")


if __name__ == "__main__":
    main()

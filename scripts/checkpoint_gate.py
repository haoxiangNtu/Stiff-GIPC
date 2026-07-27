#!/usr/bin/env python3
"""Checkpoint format, transactionality, and restart gate."""

from __future__ import annotations

import argparse
import gc
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stiff_physics import CheckpointError, Config, Engine  # noqa: E402

ASSETS = str(ROOT / "Assets") + "/"
MODES = tuple(
    mode.strip()
    for mode in os.environ.get("CHECKPOINT_MODES", "merged,strict").split(",")
    if mode.strip()
)


def transform(x: float, y: float) -> np.ndarray:
    value = np.eye(4)
    value[0, 3] = x
    value[1, 3] = y
    return value


def build(mode: str, shift: float = 0.0) -> Engine:
    engine = Engine(
        Config(
            dt=0.01,
            density=1e3,
            young_modulus=1e6,
            gravity=(0.0, -9.8, 0.0),
            ground_offset=-2.0,
            multienv_mode=mode,
            per_env_exit=(mode == "strict"),
            env_newton_iter_cap=30 if mode == "strict" else 0,
            assets_dir=ASSETS,
        )
    )
    engine.set_log_level(0)
    engine.load_mesh(
        "tetMesh/cube.msh",
        3,
        "ABD",
        transform(-0.7 + shift, 0.3),
    )
    engine.load_mesh(
        "tetMesh/cube.msh", 3, "FEM", transform(0.7 + shift, 0.3)
    )
    engine.set_body_groups([0, 0])
    engine.finalize()
    engine.native.set_body_external_force(0, 1.0, 2.0, -0.5)
    return engine


def build_contact(mode: str) -> Engine:
    """Stacked cubes in sustained frictional contact.

    The plain build() separates its bodies by 1.4 units, so its round-trip
    proves nothing about contact/friction-dependent state. Here an FEM cube
    drops 0.02 onto an ABD cube resting on the ground, with friction and a
    lateral drive engaged, so the save boundary sits inside live contact
    (asserted via the collision-pair delta of the final pre-save step).
    Friction anchors and Kappa are deliberately NOT in the checkpoint —
    both rebuild each frame from the frame-start configuration; this scene
    is what proves that reconstruction actually round-trips.
    """
    engine = Engine(
        Config(
            dt=0.01,
            density=1e3,
            young_modulus=1e6,
            gravity=(0.0, -9.8, 0.0),
            ground_offset=0.05,
            friction_rate=0.5,
            gd_friction_rate=0.4,
            multienv_mode=mode,
            per_env_exit=(mode == "strict"),
            env_newton_iter_cap=30 if mode == "strict" else 0,
            assets_dir=ASSETS,
        )
    )
    engine.set_log_level(0)
    # cube.msh natively spans y in [0.1, 0.5]; bottom cube rests near the
    # ground, top cube starts 0.02 above it and lands within a few steps.
    engine.load_mesh("tetMesh/cube.msh", 3, "ABD", transform(0.0, 0.0))
    engine.load_mesh("tetMesh/cube.msh", 3, "FEM", transform(0.0, 0.42))
    engine.set_body_groups([0, 0])
    engine.finalize()
    engine.native.set_body_external_force(1, 1.5, 0.0, 0.0)
    return engine


def build_quarantine_scene() -> Engine:
    """Build two independent environments so per-env quarantine is live."""
    engine = Engine(
        Config(
            dt=0.01,
            density=1e3,
            young_modulus=1e6,
            gravity=(0.0, -9.8, 0.0),
            ground_offset=-2.0,
            multienv_mode="strict",
            per_env_exit=True,
            env_newton_iter_cap=30,
            assets_dir=ASSETS,
        )
    )
    engine.set_log_level(0)
    engine.load_mesh(
        "tetMesh/cube.msh", 3, "FEM", transform(-0.7, 0.3)
    )
    engine.load_mesh(
        "tetMesh/cube.msh", 3, "FEM", transform(0.7, 0.3)
    )
    engine.set_body_groups([0, 1])
    engine.finalize()
    return engine


def expect_rejected(engine: Engine, path: Path, label: str) -> None:
    before = engine.get_vertices().copy()
    before_velocity = engine.get_vertex_velocities().copy()
    frame = engine.native.get_total_frames_done()
    try:
        engine.load_checkpoint(path)
    except CheckpointError as exc:
        print(f"    reject {label}: {exc}", flush=True)
    else:
        raise AssertionError(f"{label} checkpoint was accepted")
    if not np.array_equal(before, engine.get_vertices()):
        raise AssertionError(f"{label} rejection mutated vertex state")
    if not np.array_equal(before_velocity, engine.get_vertex_velocities()):
        raise AssertionError(f"{label} rejection mutated velocity state")
    if frame != engine.native.get_total_frames_done():
        raise AssertionError(f"{label} rejection mutated frame state")


def check_quarantine_restart(directory: Path) -> None:
    checkpoint = directory / "quarantine.ckpt"
    source = build_quarantine_scene()
    # Quarantine is deliberately a mid-run recovery mechanism.  An infeasible
    # initial scene must fail loudly, so first complete one healthy frame and
    # only then inject the between-frame teleport that isolation should absorb.
    source.step()
    fem = source.get_load_records()[0]
    fem_positions = source.get_vertices()[
        fem.vertex_offset : fem.vertex_offset + fem.vertex_count
    ].copy()
    fem_positions[:, 1] -= 3.0
    source.teleport_fem_vertices(fem_positions)
    source.step()
    status = list(source.native.get_per_env_status())
    if not status or status[0] != 3:
        raise AssertionError(f"quarantine did not engage: status={status}")
    source.save_checkpoint(checkpoint)
    source.step()
    expected = source.get_vertices().copy()
    del source
    gc.collect()

    restored = build_quarantine_scene()
    restored.load_checkpoint(checkpoint)
    restored.step()
    if not np.array_equal(expected, restored.get_vertices()):
        delta = float(np.max(np.abs(expected - restored.get_vertices())))
        raise AssertionError(
            f"quarantine restart is not bitwise: {delta:.3e}"
        )
    status = list(restored.native.get_per_env_status())
    if not status or status[0] != 3:
        raise AssertionError(
            f"restored quarantine is not persistent: status={status}"
        )
    print("    quarantine/ground-skip restart bitwise PASS", flush=True)


def run_case(mode: str, contact: bool = False) -> int:
    print(f"[checkpoint] mode={mode} contact={contact}", flush=True)
    with tempfile.TemporaryDirectory(prefix="stiffgipc-checkpoint-gate.") as temp:
        directory = Path(temp)
        checkpoint = directory / "state.ckpt"

        if contact:
            # Both cubes free-fall together, so the 0.02 inter-cube gap only
            # starts closing once the bottom cube grounds (~step 10); contact
            # forms ~step 17 and is resting-stable well before step 25.
            source = build_contact(mode)
            for _ in range(24):
                source.step()
            pairs_before = source.native.get_total_collision_pairs()
            source.step()
            pair_delta = source.native.get_total_collision_pairs() - pairs_before
            if pair_delta <= 0:
                raise AssertionError(
                    "contact scene has no live collision pairs at the save "
                    f"boundary (delta={pair_delta}) — scene drifted, fix it"
                )
        else:
            source = build(mode)
            source.step()
            source.step()
        source.save_checkpoint(checkpoint)
        saved_positions = source.get_vertices().copy()
        saved_velocities = source.get_vertex_velocities().copy()
        saved_frame = source.native.get_total_frames_done()
        if checkpoint.read_bytes()[:8] != b"STIFFCP2":
            raise AssertionError("versioned magic missing")
        if list(directory.glob("*.tmp.*")):
            raise AssertionError("atomic-save temporary file leaked")

        source.step()
        expected_positions = source.get_vertices().copy()
        expected_velocities = source.get_vertex_velocities().copy()
        del source
        gc.collect()

        restored = build_contact(mode) if contact else build(mode)
        restored.load_checkpoint(checkpoint)
        if not np.array_equal(saved_positions, restored.get_vertices()):
            raise AssertionError("saved positions did not round-trip bitwise")
        if not np.array_equal(saved_velocities, restored.get_vertex_velocities()):
            raise AssertionError("saved velocities did not round-trip bitwise")
        if restored.native.get_total_frames_done() != saved_frame:
            raise AssertionError("frame index did not round-trip")

        restored.step()
        actual_positions = restored.get_vertices()
        actual_velocities = restored.get_vertex_velocities()
        position_delta = float(np.max(np.abs(expected_positions - actual_positions)))
        velocity_delta = float(
            np.max(np.abs(expected_velocities - actual_velocities))
        )
        # strict promises deterministic kernels and must restart bit-for-bit -
        # in the contact case this is the completeness oracle: bitwise restart
        # under live friction proves NO state is missing from the checkpoint.
        # merged/isolated allow last-ulp atomic-order noise, hence allclose.
        # History: before load_checkpoint rebuilt the frame-entry contact pair
        # set (see checkpoint_io.cu [frame-entry pair set]), the contact case
        # diverged ~5e-6 - the first Newton iteration ran on the finalize-time
        # pair set instead of the previous frame's final one. The rebuild
        # reconstructs it from restored positions (measured restart delta
        # ~5e-17), so every case now holds the same tight bound.
        if mode == "strict":
            restart_ok = np.array_equal(
                expected_positions, actual_positions
            ) and np.array_equal(expected_velocities, actual_velocities)
        else:
            restart_ok = np.allclose(
                expected_positions, actual_positions, rtol=1e-12, atol=1e-12
            ) and np.allclose(
                expected_velocities, actual_velocities, rtol=1e-12, atol=1e-12
            )
        if not restart_ok:
            raise AssertionError(
                "restart next-step differs: "
                f"positions={position_delta:.3e} velocities={velocity_delta:.3e}"
            )
        print(
            f"    restart {'bitwise' if mode == 'strict' else 'numeric'} PASS "
            f"pos={position_delta:.2e} vel={velocity_delta:.2e}",
            flush=True,
        )

        original = checkpoint.read_bytes()
        expect_rejected(restored, directory / "missing.ckpt", "missing")
        try:
            restored.save_checkpoint("")
        except CheckpointError as exc:
            print(f"    reject empty save path: {exc}", flush=True)
        else:
            raise AssertionError("empty checkpoint save path was accepted")

        corrupt = directory / "corrupt.ckpt"
        damaged = bytearray(original)
        damaged[-1] ^= 0x01
        corrupt.write_bytes(damaged)
        expect_rejected(restored, corrupt, "checksum")

        wrong_model = directory / "wrong-model.ckpt"
        incompatible = bytearray(original)
        # v2 fixed header: constitutive model id is the reserved u32 at 36.
        struct.pack_into("<I", incompatible, 36, 0xFFFFFFFF)
        wrong_model.write_bytes(incompatible)
        expect_rejected(restored, wrong_model, "constitutive-model")

        truncated = directory / "truncated.ckpt"
        truncated.write_bytes(original[:-13])
        expect_rejected(restored, truncated, "truncated")

        legacy = directory / "legacy.ckpt"
        legacy.write_bytes(struct.pack("=III", 0x53544B50, 16, 1))
        expect_rejected(restored, legacy, "legacy")

        # The native solver exposes process-global CUDA symbols and therefore
        # permits only one finalized Engine at a time. Release the restart
        # engine before constructing the same-count/different-geometry probe.
        del restored
        gc.collect()
        different_scene = build(mode, shift=0.01)
        expect_rejected(different_scene, checkpoint, "scene-signature")
        del different_scene
        gc.collect()
        if mode == "strict":
            check_quarantine_restart(directory)

    print(f"[checkpoint] mode={mode} PASS", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("merged", "isolated", "strict"))
    parser.add_argument("--contact", action="store_true")
    args = parser.parse_args()
    if args.case:
        return run_case(args.case, contact=args.contact)
    ok = True
    for mode in MODES:
        for extra in ((), ("--contact",)):
            result = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--case", mode, *extra],
                env=os.environ.copy(),
            )
            ok &= result.returncode == 0
    print("CHECKPOINT-GATE:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

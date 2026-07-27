#!/usr/bin/env python3
"""GPU lifecycle and process-scoped mode regression gate."""

from __future__ import annotations

import gc
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stiff_physics import Config, Engine, LifecycleError

ASSETS = str(ROOT / "Assets") + "/"


def expect_lifecycle(call, label: str) -> None:
    try:
        call()
    except LifecycleError:
        print(f"    {label}: rejected", flush=True)
    else:
        raise AssertionError(f"{label} unexpectedly succeeded")


def expect_value_error(call, label: str) -> None:
    try:
        call()
    except ValueError:
        print(f"    {label}: rejected", flush=True)
    else:
        raise AssertionError(f"{label} unexpectedly succeeded")


def main() -> int:
    engine = Engine(Config(multienv_mode="merged"))
    with tempfile.TemporaryDirectory(prefix="stiffgipc-lifecycle-gate.") as tmp:
        expect_lifecycle(
            lambda: engine.save_checkpoint(os.path.join(tmp, "invalid.ckpt")),
            "unfinalized checkpoint",
        )
        expect_lifecycle(
            lambda: engine.native.save_checkpoint(
                os.path.join(tmp, "invalid-native.ckpt")
            ),
            "unfinalized native checkpoint",
        )
        expect_lifecycle(
            lambda: engine.native.load_checkpoint(
                os.path.join(tmp, "missing-native.ckpt")
            ),
            "unfinalized native restore",
        )
    expect_lifecycle(engine.step, "unfinalized step")
    del engine
    gc.collect()
    print("    unfinalized destruction: safe", flush=True)

    first = Engine(Config(multienv_mode="merged", assets_dir=ASSETS))
    second = Engine(Config(multienv_mode="merged", assets_dir=ASSETS))
    expect_lifecycle(
        lambda: first.native.set_config(
            Config(multienv_mode="merged", assets_dir=ASSETS).native
        ),
        "native set_config after init",
    )
    for cycle in range(2):
        first.load_mesh("tetMesh/cube.msh", 3, "FEM", np.eye(4))
        first.set_body_groups([0])
        first.finalize()
        stress = first.get_fem_von_mises_stress()
        if len(stress) != len(first.get_vertices()):
            raise AssertionError("stress scratch readback has the wrong size")
        expect_lifecycle(first.finalize, f"double finalize cycle {cycle + 1}")
        first.reset()
        expect_lifecycle(first.step, f"step after reset cycle {cycle + 1}")
        print(f"    finalized reset/reuse cycle {cycle + 1}: safe", flush=True)
    expect_lifecycle(
        lambda: first.native.set_vertex_env_ids([0]),
        "unfinalized vertex-env filter",
    )
    for candidate in (first, second):
        candidate.load_mesh("tetMesh/cube.msh", 3, "FEM", np.eye(4))
        candidate.set_body_groups([0])
    first.finalize()
    first_vertex_count = len(first.get_vertices())
    expect_value_error(
        lambda: first.native.set_vertex_env_ids([0]),
        "wrong-size vertex-env filter",
    )
    first.native.set_vertex_env_ids([0] * first_vertex_count)
    expect_lifecycle(
        second.finalize,
        "concurrent finalized Engine",
    )
    first.native.set_vertex_env_ids([])
    first.reset()
    second.finalize()
    second_vertex_count = len(second.get_vertices())
    second.native.set_vertex_env_ids([0] * second_vertex_count)
    second.native.set_vertex_env_ids([])
    print("    runtime + vertex-env owner handoff: safe", flush=True)
    expect_lifecycle(
        lambda: Engine(Config(multienv_mode="strict")),
        "cross-mode Engine",
    )
    expect_lifecycle(
        lambda: Engine(Config(multienv_mode="merged", per_env_exit=True)),
        "per_env_exit change",
    )
    expect_lifecycle(
        lambda: Engine(Config(multienv_mode="merged", cuda_device=1)),
        "cross-device Engine",
    )
    os.environ["STIFF_SPMV_DET"] = "1"
    expect_lifecycle(
        lambda: Engine(Config(multienv_mode="merged")),
        "mutated STIFF_* signature",
    )
    expect_lifecycle(
        first.finalize,
        "mutated STIFF_* before finalize",
    )
    del first, second
    gc.collect()
    print("LIFECYCLE-GATE: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

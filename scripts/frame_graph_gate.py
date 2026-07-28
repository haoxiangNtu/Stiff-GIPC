#!/usr/bin/env python3
"""Phase-C frame transaction gate.

Checks the honest warm-up fallback, graph audit counts, and byte-exact
terminal rollback.  The rollback injection is process-scoped, so the parent
runs the two cases in clean subprocesses.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def make_engine():
    from stiff_physics.engine import Config, Engine

    cfg = Config(
        dt=0.01,
        density=1e3,
        young_modulus=1e6,
        friction_rate=0.4,
        relative_dhat=1e-3,
        ground_offset=0.0,
        assets_dir=os.path.join(ROOT, "Assets") + "/",
        multienv_mode="strict",
        preconditioner_type=1,
    )
    cfg._cfg.absolute_dhat = 1e-3
    engine = Engine(cfg)
    engine.load_mesh("tetMesh/cube.msh", 3, "FEM", np.eye(4))
    engine.native.set_body_groups([0])
    engine.finalize()
    return engine


def normal_case() -> None:
    engine = make_engine()
    engine.step()
    warm = engine.native.get_frame_status()
    assert warm.path_flags & 1
    assert warm.path_flags & 4
    assert not (warm.path_flags & 2)

    engine.step()
    status = engine.native.get_frame_status()
    assert status.result == 0
    assert status.path_flags & 1
    assert status.path_flags & 2
    assert status.path_flags & 8
    assert not (status.path_flags & 16)
    assert status.graph_launches == 2
    assert status.host_boundaries == 1
    assert status.root_graph_nodes > 0
    assert status.root_d2h_nodes == 0
    assert status.terminal_graph_nodes > 0
    assert status.terminal_d2h_nodes == 1
    print("FRAME-GRAPH-NORMAL: PASS")


def rollback_case() -> None:
    engine = make_engine()
    engine.step()
    before = np.asarray(engine.get_vertices()).copy()
    try:
        engine.step()
    except RuntimeError:
        pass
    else:
        raise AssertionError("forced frame rollback did not surface an error")
    after = np.asarray(engine.get_vertices()).copy()
    status = engine.native.get_frame_status()
    assert np.array_equal(before, after)
    assert status.result == 1
    assert status.phase == 9
    assert status.error_code == 1
    assert status.invalid_bits & (1 << 16)
    assert status.terminal_d2h_nodes == 1
    print("FRAME-GRAPH-ROLLBACK: PASS")


def run_child(mode: str) -> None:
    env = os.environ.copy()
    env["STIFF_FRAME_GRAPH"] = "1"
    env["STIFF_MULTIENV_MODE"] = "strict"
    if mode == "rollback":
        env["STIFF_FRAME_FORCE_ROLLBACK"] = "1"
    else:
        env.pop("STIFF_FRAME_FORCE_ROLLBACK", None)
    subprocess.run(
        [sys.executable, __file__, f"--child={mode}"],
        check=True,
        cwd=ROOT,
        env=env,
    )


if __name__ == "__main__":
    child = next(
        (arg.split("=", 1)[1] for arg in sys.argv[1:]
         if arg.startswith("--child=")),
        None,
    )
    if child == "normal":
        normal_case()
    elif child == "rollback":
        rollback_case()
    else:
        run_child("normal")
        run_child("rollback")
        print("FRAME-GRAPH-GATE: PASS")

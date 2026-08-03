#!/usr/bin/env python3
"""Phase-C frame transaction gate.

Checks the honest warm-up fallback, graph audit counts, and byte-exact
terminal rollback.  The rollback injection is process-scoped, so the parent
runs the two cases in clean subprocesses.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def make_engine(hybrid: bool = False):
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
    if hybrid:
        engine.load_mesh("tetMesh/cube.msh", 3, "ABD", np.eye(4))
        transform = np.eye(4)
        transform[0, 3] = 0.4005
        engine.load_mesh("tetMesh/cube.msh", 3, "ABD", transform)
        engine.native.set_body_groups([0, 0])
    else:
        engine.load_mesh("tetMesh/cube.msh", 3, "FEM", np.eye(4))
        engine.native.set_body_groups([0])
    engine.finalize()
    return engine


def make_full_graph_engine():
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


def unique_overflow_case() -> None:
    engine = make_engine(hybrid=True)
    engine.step()
    before = np.asarray(engine.get_vertices()).copy()
    engine.step()
    after = np.asarray(engine.get_vertices()).copy()
    status = engine.native.get_frame_status()
    assert not np.array_equal(before, after)
    assert status.result == 0
    assert status.phase == 8
    assert status.error_code == 0
    assert status.attempt == 1
    assert status.retry_count == 1
    assert status.retry_invalid_bits & (1 << 19)
    assert status.path_flags & (1 << 6)
    assert status.path_flags & (1 << 7)
    assert status.terminal_d2h_nodes == 1
    print("FRAME-GRAPH-UNIQUE-RETRY: PASS")


def unique_exhaustion_case() -> None:
    engine = make_engine(hybrid=True)
    engine.step()
    before = np.asarray(engine.get_vertices()).copy()
    try:
        engine.step()
    except RuntimeError:
        pass
    else:
        raise AssertionError("zero retry budget did not surface overflow")
    after = np.asarray(engine.get_vertices()).copy()
    status = engine.native.get_frame_status()
    assert np.array_equal(before, after)
    assert status.result == 1
    assert status.phase == 9
    assert status.error_code == 4
    assert status.retry_count == 0
    assert status.retry_invalid_bits & (1 << 19)
    print("FRAME-GRAPH-RETRY-EXHAUSTION: PASS")


def full_digest_case(require_full: bool) -> None:
    engine = make_full_graph_engine()
    for _ in range(4):
        engine.step()
    status = engine.native.get_frame_status()
    assert status.result == 0
    if require_full:
        assert status.path_flags & (1 << 4)
        assert status.path_flags & (1 << 8)
        assert status.path_flags & (1 << 9)
        assert not (status.path_flags & (1 << 3))
        assert status.graph_launches == 1
        assert status.host_boundaries == 1
        assert status.root_graph_nodes > 0
        assert status.root_d2h_nodes == 1
        assert status.terminal_graph_nodes == 0
        assert status.terminal_d2h_nodes == 0
    else:
        assert not (status.path_flags & (1 << 4))
        assert status.path_flags & (1 << 3)
    vertices = np.asarray(engine.get_vertices())
    digest = hashlib.sha256(vertices.tobytes()).hexdigest()
    dump = os.environ.get("FRAME_GRAPH_DUMP")
    if dump:
        np.save(dump, np.asarray(vertices, dtype=np.float64))
    print(f"FRAME-GRAPH-FULL-DIGEST: {digest}")


def audit_fallback_case() -> None:
    engine = make_full_graph_engine()
    engine.step()
    engine.step()
    status = engine.native.get_frame_status()
    assert status.result == 0
    assert not (status.path_flags & (1 << 4))
    assert status.path_flags & (1 << 3)
    assert status.graph_launches == 2
    assert status.host_boundaries == 1
    print("FRAME-GRAPH-AUDIT-FALLBACK: PASS")


def run_child(mode: str) -> None:
    env = os.environ.copy()
    env["STIFF_FRAME_GRAPH"] = "1"
    env["STIFF_FRAME_FULL_GRAPH"] = "0"
    env["STIFF_MULTIENV_MODE"] = "strict"
    if mode == "rollback":
        env["STIFF_FRAME_FORCE_ROLLBACK"] = "1"
    else:
        env.pop("STIFF_FRAME_FORCE_ROLLBACK", None)
    if mode.startswith("unique-"):
        env["STIFF_FRAME_FORCE_UNIQUE_TIER"] = "1"
    else:
        env.pop("STIFF_FRAME_FORCE_UNIQUE_TIER", None)
    if mode == "unique-exhaustion":
        env["STIFF_FRAME_MAX_RETRIES"] = "0"
    else:
        env.pop("STIFF_FRAME_MAX_RETRIES", None)
    subprocess.run(
        [sys.executable, __file__, f"--child={mode}"],
        check=True,
        cwd=ROOT,
        env=env,
    )


def run_audit_fallback_child() -> None:
    env = os.environ.copy()
    env["STIFF_FRAME_GRAPH"] = "1"
    env["STIFF_FRAME_FULL_GRAPH"] = "1"
    # [C6-q] tiny fixture; pin the size threshold off for the full-graph audit
    env["STIFF_FULL_GRAPH_MIN_VERTS"] = "0"
    env["STIFF_MULTIENV_MODE"] = "merged"
    env["STIFF_MIRROR_AUDIT"] = "1"
    env["STIFF_SLOT_AUDIT"] = "1"
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
    subprocess.run(
        [sys.executable, __file__, "--child=audit-fallback"],
        check=True,
        cwd=ROOT,
        env=env,
    )


def run_full_child(mode: str, *, det: bool = True, dump: str | None = None) -> str:
    env = os.environ.copy()
    # The repository-wide gate arms these host-side audits globally. They are
    # intentionally ineligible for whole-frame capture and have their own
    # two-graph coverage above; isolate the full-graph fingerprint subprocess.
    env.pop("STIFF_MIRROR_AUDIT", None)
    env.pop("STIFF_SLOT_AUDIT", None)
    env["STIFF_FRAME_GRAPH"] = "1"
    env["STIFF_FRAME_FULL_GRAPH"] = (
        "1" if mode == "full-enabled" else "0"
    )
    # [C6-q] tiny fixture; pin the size threshold off for the full-graph audit
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
    # [C6-w] The bitwise digest comparison runs on the DETERMINISTIC reduction
    # stack. On the default merged stack `binned_deposit` is a bare atomicAdd,
    # so equal bits between two processes is an empirical property of GPU
    # scheduling, not a guarantee -- measured: the plain release path diverges
    # run-to-run from frame 2 (2.2e-14) to 1.1e-4 by frame 119 on a contact
    # scene, and this gate only held because its fixture is 8 vertices over 4
    # steps. Under STIFF_SPMV_DET the deposits go through the order-free binned
    # cascade, where "graph must not change a single bit" is a mathematical
    # statement about the code, which is exactly the property this gate exists
    # to defend. The default stack is still covered, by the tolerance envelope
    # below.
    if det:
        env["STIFF_SPMV_DET"] = "1"
    if dump:
        env["FRAME_GRAPH_DUMP"] = dump
    else:
        env.pop("FRAME_GRAPH_DUMP", None)
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
    prefix = "FRAME-GRAPH-FULL-DIGEST: "
    digest = next(
        line[len(prefix):]
        for line in completed.stdout.splitlines()
        if line.startswith(prefix)
    )
    return digest


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
    elif child == "unique-overflow":
        unique_overflow_case()
    elif child == "unique-exhaustion":
        unique_exhaustion_case()
    elif child == "full-baseline":
        full_digest_case(False)
    elif child == "full-enabled":
        full_digest_case(True)
    elif child == "audit-fallback":
        audit_fallback_case()
    else:
        run_child("normal")
        run_child("rollback")
        run_child("unique-overflow")
        run_child("unique-exhaustion")
        run_audit_fallback_child()
        baseline_digest = run_full_child("full-baseline")
        enabled_digest = run_full_child("full-enabled")
        assert enabled_digest == baseline_digest, (
            "whole-frame graph changed the result on the deterministic stack"
        )
        print("FRAME-GRAPH-FULL-DET: PASS (bitwise)")

        # [C6-w] Default (non-deterministic) stack: assert the graph lands
        # inside the baseline's OWN run-to-run envelope. Bitwise equality is
        # not available here by construction -- bare atomicAdd ordering -- so
        # the honest question is whether graph-vs-baseline is distinguishable
        # from baseline-vs-baseline.
        with tempfile.TemporaryDirectory() as tmp:
            base_runs = []
            for index in range(3):
                path = os.path.join(tmp, f"base{index}.npy")
                run_full_child("full-baseline", det=False, dump=path)
                base_runs.append(np.load(path))
            graph_path = os.path.join(tmp, "graph.npy")
            run_full_child("full-enabled", det=False, dump=graph_path)
            graph_run = np.load(graph_path)
        scale = float(np.abs(base_runs[0]).max()) or 1.0
        noise = max(
            float(np.abs(a - b).max())
            for i, a in enumerate(base_runs)
            for b in base_runs[i + 1:]
        )
        error = min(float(np.abs(graph_run - b).max()) for b in base_runs)
        # Floor: legal reassociation on a 4-step fixture stays within a few
        # thousand ULP. A real semantic drift (the class of bug this gate has
        # actually caught) is many orders larger.
        budget = max(noise * 4.0, 1e-11 * scale)
        print(
            f"FRAME-GRAPH-FULL-ENVELOPE: error={error:.3e} "
            f"baseline_noise={noise:.3e} budget={budget:.3e}"
        )
        assert error <= budget, "graph-vs-baseline exceeds the baseline's own noise"
        print("FRAME-GRAPH-FULL: PASS")
        print("FRAME-GRAPH-GATE: PASS")

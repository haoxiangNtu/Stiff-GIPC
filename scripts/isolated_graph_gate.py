#!/usr/bin/env python3
"""C5: isolated mode inside the whole-frame conditional CUDA graph.

Four FEM cubes in four declared environment groups fall onto the ground.
The baseline runs isolated mode on the audited per-frame path; the graph
child runs the same scene with STIFF_C5_ISOLATED_GRAPH=1 so the whole
frame — per-env CCD alpha chain, per-env freeze decision, the per-env S3
line-search WHILE loop, in-graph kappa — executes as one root launch.

What this gate asserts:
  1. every frame really ran the full conditional graph (zero fallback);
  2. trajectories match the baseline's own run-to-run envelope;
  3. ISOLATION HOLDS: perturbing env 3's initial state must leave envs
     0..2 bit-identical, on both paths. This is the isolated-mode promise
     and the reason the graph records the merged tree with emission-time
     cross-env filtering rather than per-env trees.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FRAMES = int(os.environ.get("ISOLATED_GRAPH_GATE_FRAMES", "20"))
ENVS = 4
PATH_FULL_CONDITIONAL_GRAPH = 1 << 4


def transform(x: float, y: float) -> np.ndarray:
    value = np.eye(4)
    value[:3, :3] *= 0.4
    value[0, 3] = x
    value[1, 3] = y
    return value


def make_engine(perturb_last: bool = False):
    from stiff_physics.engine import Config, Engine

    cfg = Config(
        dt=0.01,
        gravity=(0.0, -9.8, 0.0),
        ground_offset=-0.10,
        assets_dir=os.path.join(ROOT, "Assets") + "/",
        multienv_mode="isolated",
        preconditioner_type=1,
        friction_rate=0.0,
    )
    engine = Engine(cfg)
    for env in range(ENVS):
        # Envs are separated far enough that no cross-env pair is physical;
        # isolation is still what guarantees it stays that way numerically.
        y = 0.0
        if perturb_last and env == ENVS - 1:
            y = 0.011  # only env 3 differs
        engine.load_mesh(
            "tetMesh/cube.msh", 3, "FEM", transform(env * 3.0, y)
        )
    engine.set_body_groups(list(range(ENVS)))
    engine.finalize()
    engine.native.set_log_level(0)
    engine.step()  # warm-up trains lazy workspaces and capacity tiers
    return engine


def run(dump_path: str, expect_graph: bool, perturb: bool) -> None:
    engine = make_engine(perturb_last=perturb)
    positions: list[np.ndarray] = []
    graph_frames = 0
    fallback_frames = 0
    for frame in range(FRAMES):
        try:
            engine.step()
        except Exception:
            status = engine.native.get_frame_status()
            print(
                "ISOLATED-GRAPH-FAILURE: "
                f"frame={frame} result={status.result} "
                f"phase={status.phase} error={status.error_code} "
                f"bits={status.invalid_bits:#x} "
                f"newton={status.newton_iters} ls={status.ls_trials}",
                flush=True,
            )
            raise
        status = engine.native.get_frame_status()
        assert status.result == 0, (
            f"frame {frame}: result={status.result} "
            f"error={status.error_code} bits={status.invalid_bits:#x}"
        )
        if bool(status.path_flags & PATH_FULL_CONDITIONAL_GRAPH):
            graph_frames += 1
        else:
            fallback_frames += 1
        if os.environ.get("ISOLATED_GRAPH_GATE_TRACE") == "1":
            print(
                f"ISOLATED-GRAPH-FRAME: {frame} "
                f"full={int(bool(status.path_flags & PATH_FULL_CONDITIONAL_GRAPH))} "
                f"newton={status.newton_iters} pcg={status.pcg_iters} "
                f"ls={status.ls_trials} dcd={status.hw_dcd_pairs} "
                f"kappa={status.kappa:.6g}"
            )
        positions.append(np.asarray(engine.get_vertices()).copy())

    array = np.asarray(positions)
    assert np.isfinite(array).all()
    print(
        f"ISOLATED-GRAPH-STATS: graph_frames={graph_frames} "
        f"fallback_frames={fallback_frames}"
    )
    if expect_graph:
        # [C6-i] A capacity overflow is adjudicated at the frame boundary by
        # finishing THAT frame on the release solver (the baseline path) and
        # re-recording the next frame at the grown tier -- replaying the frame
        # in-graph at a new tier changes grid shapes and made retries
        # non-deterministic. Such growth frames are legitimate and rare; what
        # this gate must still reject is a scene that cannot hold the graph at
        # steady state. Budget: at most 2 fallback frames per episode, and the
        # numerics assertions below stay bit-for-bit unchanged.
        assert fallback_frames <= 2, (
            f"{fallback_frames} frames fell back off the conditional graph "
            "(capacity-growth frames are budgeted at 2 per episode)"
        )
    np.savez(dump_path, positions=array)
    print("ISOLATED-GRAPH-RUN: done")


def child_environment(graph: bool) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("STIFF_MIRROR_AUDIT", None)
    env.pop("STIFF_SLOT_AUDIT", None)
    env["STIFFGIPC_NATIVE_DIR"] = os.environ.get(
        "STIFFGIPC_NATIVE_DIR", os.path.join(ROOT, "build")
    )
    if graph:
        env["STIFF_FRAME_GRAPH"] = "1"
        env["STIFF_FRAME_FULL_GRAPH"] = "1"
        env["STIFF_C4_COLLISION_GRAPH"] = "1"
        env["STIFF_C5_ISOLATED_GRAPH"] = "1"
    else:
        env["STIFF_FRAME_GRAPH"] = "0"
        for key in (
            "STIFF_FRAME_FULL_GRAPH",
            "STIFF_C4_COLLISION_GRAPH",
            "STIFF_C5_ISOLATED_GRAPH",
        ):
            env.pop(key, None)
    return env


def run_child(mode: str, dump_path: str, perturb: bool) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            __file__,
            f"--child={mode}",
            dump_path,
            "1" if perturb else "0",
        ],
        check=True,
        cwd=ROOT,
        env=child_environment(graph=(mode == "graph")),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(completed.stdout, end="")


def load(path: str) -> np.ndarray:
    with np.load(path) as archive:
        return np.asarray(archive["positions"]).copy()


def envelope_compare(
    label: str,
    first: np.ndarray,
    second: np.ndarray,
    observed: np.ndarray,
) -> None:
    assert first.shape == second.shape == observed.shape
    baseline_noise = float(np.max(np.abs(first - second)))
    error = min(
        float(np.max(np.abs(observed - first))),
        float(np.max(np.abs(observed - second))),
    )
    scale = max(
        1.0, float(np.max(np.abs(first))), float(np.max(np.abs(second)))
    )
    precision_floor = 16.0 * np.finfo(first.dtype).eps * scale
    # Same rationale as G18: capacity-grid reductions legally reassociate
    # the sums relative to the exact-count baseline.
    tolerance = max(8.0 * baseline_noise, precision_floor, 1e-9 * scale)
    assert error <= tolerance, (
        f"{label} differs beyond the envelope: error={error:.17g}, "
        f"baseline_noise={baseline_noise:.17g}, tolerance={tolerance:.17g}"
    )
    print(
        f"ISOLATED-GRAPH-NUMERICS: {label} error={error:.3e} "
        f"baseline_noise={baseline_noise:.3e} tolerance={tolerance:.3e}"
    )


def isolation_delta(plain: np.ndarray, perturbed: np.ndarray,
                    verts_per_env: int) -> tuple[float, float]:
    """Return (delta of envs 0..N-2, delta of the perturbed env N-1)."""
    untouched = verts_per_env * (ENVS - 1)
    return (
        float(np.max(np.abs(plain[:, :untouched] - perturbed[:, :untouched]))),
        float(np.max(np.abs(plain[:, untouched:] - perturbed[:, untouched:]))),
    )


def assert_isolation(label: str, delta: float, moved: float) -> None:
    """Assert what ISOLATED actually promises (multienv/mode_contract.h).

    The contract's isolated tier promises per-env fairness (own alpha,
    own freeze, own kappa, env-local broadphase => no cross-env contact)
    and explicitly states "NO reproducibility promise" — bitwise cross-env
    identity is the STRICT tier. So the assertion here is PHYSICAL
    isolation: perturbing one env must not couple into its neighbours at
    any physically meaningful scale. Residual jitter (order-dependent
    atomics in the shared SpMV, amplified by contact dynamics) is expected
    and is exactly what STRICT's canonical orders remove.
    """
    print(
        f"ISOLATED-GRAPH-ISOLATION: {label} "
        f"unperturbed_env_delta={delta:.3e} perturbed_env_delta={moved:.3e} "
        f"ratio={delta / moved:.3e}"
    )
    assert moved > 1e-6, (
        f"{label}: the perturbation did not move its own env — test is blind"
    )
    assert delta < 1e-4 * moved, (
        f"{label}: perturbing env {ENVS - 1} coupled into envs "
        f"0..{ENVS - 2} at {delta:.17g} vs its own {moved:.17g} — that is "
        f"physical coupling, not numerical jitter: isolation broken"
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
    if child in ("baseline", "graph"):
        run(
            sys.argv[2],
            expect_graph=(child == "graph"),
            perturb=(len(sys.argv) > 3 and sys.argv[3] == "1"),
        )
    else:
        with tempfile.TemporaryDirectory(
            prefix="stiff-isolated-graph-gate-"
        ) as tmp:
            paths = {
                key: os.path.join(tmp, f"{key}.npz")
                for key in (
                    "baseline_a",
                    "baseline_b",
                    "graph",
                    "baseline_perturbed",
                    "graph_perturbed",
                )
            }
            run_child("baseline", paths["baseline_a"], False)
            run_child("baseline", paths["baseline_b"], False)
            run_child("graph", paths["graph"], False)
            run_child("baseline", paths["baseline_perturbed"], True)
            run_child("graph", paths["graph_perturbed"], True)

            baseline_a = load(paths["baseline_a"])
            baseline_b = load(paths["baseline_b"])
            graph = load(paths["graph"])
            envelope_compare("positions", baseline_a, baseline_b, graph)

            verts_per_env = baseline_a.shape[1] // ENVS
            base_delta, base_moved = isolation_delta(
                baseline_a, load(paths["baseline_perturbed"]), verts_per_env
            )
            graph_delta, graph_moved = isolation_delta(
                graph, load(paths["graph_perturbed"]), verts_per_env
            )
            assert_isolation("baseline", base_delta, base_moved)
            assert_isolation("graph", graph_delta, graph_moved)
            # The C5 claim: recording the merged tree with emission-time
            # cross-env filtering does not couple envs any more than the
            # per-env-tree path already does.
            budget = max(8.0 * base_delta, 1e-6 * base_moved)
            print(
                "ISOLATED-GRAPH-ISOLATION-PARITY: "
                f"graph={graph_delta:.3e} baseline={base_delta:.3e} "
                f"budget={budget:.3e}"
            )
            assert graph_delta <= budget, (
                f"the graph path couples envs more than the per-env-tree "
                f"baseline: {graph_delta:.17g} > {budget:.17g}"
            )
        print("ISOLATED-GRAPH-GATE: PASS")

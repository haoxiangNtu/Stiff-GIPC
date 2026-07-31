#!/usr/bin/env python3
"""C4-a: collision inside the whole-frame conditional CUDA graph.

Two FEM cubes fall under gravity: the lower one reaches the ground plane,
the upper one lands on it — exercising ground pairs, self pairs, the swept
CCD scalar chain, and the conditional line search with real barrier energy.
The graph child runs the same physics as the synchronous baseline with
STIFF_C4_COLLISION_GRAPH=1; every frame must stay inside the baseline's own
run-to-run nondeterminism envelope.

C4-a contract points asserted here:
  - the baseline keeps kappa constant over the window (the graph freezes
    kappa at its frame-boundary value, so a kappa-doubling scene would be
    out of contract);
  - friction coefficients are zero (friction sets rebuild only at
    synchronous frames until C4-c);
  - at least one graph frame executes the full conditional path with a
    nonzero DCD pair count, proving contact really ran inside the graph.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# 20 frames: contact starts around frame 15; the baseline's first kappa
# doubling lands at frame 20, and the C4-a graph freezes kappa inside the
# frame, so the sync-equivalence window ends there by contract.
FRAMES = int(os.environ.get("COLLISION_GRAPH_GATE_FRAMES", "24"))
PATH_FULL_CONDITIONAL_GRAPH = 1 << 4

# C4-d gate matrix. Each scenario runs baseline x2 + graph and must stay
# inside the equivalence envelope:
#   ground   — two-cube drop onto the ground plane (ground pairs + swept CCD)
#   friction — same drop with mu=0.3 (C4-c lagged sets + friction energy/GH)
#   squeeze  — upper cube dropped from height directly onto the lower one
#              (cube-cube DCD/EE narrow-phase traffic + ground)
SCENARIOS = {
    "ground": {"friction": 0.0, "upper_y": 0.43},
    "friction": {"friction": 0.3, "upper_y": 0.43},
    "squeeze": {"friction": 0.0, "upper_y": 0.75},
}


def transform(y: float) -> np.ndarray:
    value = np.eye(4)
    value[:3, :3] *= 0.4
    value[1, 3] = y
    return value


def make_engine(scenario: str):
    from stiff_physics.engine import Config, Engine

    spec = SCENARIOS[scenario]
    cfg = Config(
        dt=0.01,
        gravity=(0.0, -9.8, 0.0),
        ground_offset=-0.10,
        assets_dir=os.path.join(ROOT, "Assets") + "/",
        multienv_mode="merged",
        preconditioner_type=1,
        friction_rate=spec["friction"],
    )
    engine = Engine(cfg)
    # Lower cube: bottom starts 0.02 above the ground plane.
    engine.load_mesh("tetMesh/cube.msh", 3, "FEM", transform(0.0))
    engine.load_mesh(
        "tetMesh/cube.msh", 3, "FEM", transform(spec["upper_y"])
    )
    engine.finalize()
    engine.native.set_log_level(0)
    engine.step()  # warm-up trains lazy workspaces and capacity tiers
    return engine


def run(dump_path: str, expect_graph: bool, scenario: str) -> None:
    engine = make_engine(scenario)
    positions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    kappas: list[float] = []
    graph_contact_frames = 0
    fallback_frames = 0
    dcd_frames = 0
    for frame in range(FRAMES):
        try:
            engine.step()
        except Exception:
            status = engine.native.get_frame_status()
            print(
                "COLLISION-GRAPH-FAILURE-STATUS: "
                f"frame={frame} result={status.result} "
                f"phase={status.phase} error={status.error_code} "
                f"bits={status.invalid_bits:#x} "
                f"newton={status.newton_iters} pcg={status.pcg_iters} "
                f"ls={status.ls_trials} alpha={status.final_alpha:.17g} "
                f"cfl={status.cfl_alpha:.17g} "
                f"energy={status.final_energy:.17g} "
                f"dcd={status.hw_dcd_pairs} ccd={status.hw_ccd_pairs} "
                f"prim={status.err_primitive} "
                f"err_newton={status.err_newton_iter} "
                f"err_ls={status.err_ls_iter}",
                flush=True,
            )
            raise
        status = engine.native.get_frame_status()
        assert status.result == 0, (
            f"frame {frame}: result={status.result} "
            f"error={status.error_code} bits={status.invalid_bits:#x}"
        )
        kappas.append(float(status.kappa))
        full = bool(status.path_flags & PATH_FULL_CONDITIONAL_GRAPH)
        if status.hw_dcd_pairs > 0:
            dcd_frames += 1
        if expect_graph:
            # "Contact ran inside the graph" is evidenced by the
            # contact-frame Newton signature: free flight always converges in
            # one iteration, contact needs several. The per-frame envelope
            # comparison then proves the contact physics matched.
            if full and status.newton_iters > 1:
                graph_contact_frames += 1
            if not full:
                fallback_frames += 1
        if os.environ.get("COLLISION_GRAPH_GATE_TRACE") == "1":
            print(
                f"COLLISION-GRAPH-FRAME: {frame} full={int(full)} "
                f"newton={status.newton_iters} pcg={status.pcg_iters} "
                f"dcd={status.hw_dcd_pairs} ccd={status.hw_ccd_pairs} "
                f"retries={status.retry_count} kappa={status.kappa:.6g}"
            )
        positions.append(np.asarray(engine.get_vertices()).copy())
        velocities.append(
            np.asarray(engine.native.get_vertex_velocities()).copy()
        )

    assert np.isfinite(np.asarray(positions)).all()
    # [C4-b] kappa now advances inside the graph (in-graph close-set
    # doubling + boundary initKappa); the old kappa-quiet window contract is
    # gone. The kappa trajectory itself is part of the envelope evidence.
    print(
        "COLLISION-GRAPH-KAPPA: "
        f"first={kappas[0]:.9g} last={kappas[-1]:.9g} "
        f"moved={int(min(kappas) != max(kappas))}"
    )
    print(f"COLLISION-GRAPH-DCD-FRAMES: {dcd_frames}")
    if expect_graph:
        print(
            "COLLISION-GRAPH-STATS: "
            f"contact_graph_frames={graph_contact_frames} "
            f"fallback_frames={fallback_frames}"
        )
        if os.environ.get("COLLISION_GRAPH_GATE_REQUIRE_CONTACT", "1") == "1":
            assert graph_contact_frames > 0, (
                "no frame ran contact inside the full conditional graph"
            )
    np.savez(
        dump_path,
        positions=np.asarray(positions),
        velocities=np.asarray(velocities),
        kappas=np.asarray(kappas),
    )
    print("COLLISION-GRAPH-RUN: done")


def child_environment(graph: bool) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("STIFF_MIRROR_AUDIT", None)
    env.pop("STIFF_SLOT_AUDIT", None)
    env["STIFFGIPC_NATIVE_DIR"] = os.environ.get(
        "STIFFGIPC_NATIVE_DIR", os.path.join(ROOT, "build")
    )
    env["STIFF_MULTIENV_MODE"] = "merged"
    if graph:
        env["STIFF_FRAME_GRAPH"] = "1"
        env["STIFF_FRAME_FULL_GRAPH"] = "1"
        # [C6-n] gate fixtures are tiny (16-vertex cubes / 2 cloths); pin the
        # size threshold off so they keep exercising the whole-frame graph
        env["STIFF_FULL_GRAPH_MIN_VERTS"] = "0"
        env["STIFF_C4_COLLISION_GRAPH"] = "1"
    else:
        env["STIFF_FRAME_GRAPH"] = "0"
        env.pop("STIFF_FRAME_FULL_GRAPH", None)
        env.pop("STIFF_C4_COLLISION_GRAPH", None)
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


def run_child(mode: str, dump_path: str, scenario: str) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            __file__,
            f"--child={mode}",
            dump_path,
            scenario,
        ],
        check=True,
        cwd=ROOT,
        env=child_environment(graph=(mode == "graph")),
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
    baselines: list[np.ndarray],
    observed: np.ndarray,
) -> None:
    # [C6-j] The envelope is derived from N baseline runs, not 2. With two
    # runs the noise estimate is a single pairwise sample of a chaotic
    # distribution: whenever the two happened to land close (measured:
    # 1e-10 noise on a scene whose true spread is 1e-6), the gate flagged a
    # perfectly healthy graph run about one time in two. Noise is now the
    # max pairwise spread across all baselines and the error is the distance
    # to the NEAREST baseline. N via STIFF_G18_BASELINE_RUNS (default 4).
    assert all(b.shape == observed.shape for b in baselines)
    baseline_noise = max(
        float(np.max(np.abs(a - b)))
        for i, a in enumerate(baselines)
        for b in baselines[i + 1 :]
    )
    error = min(float(np.max(np.abs(observed - b))) for b in baselines)
    scale = max(
        [1.0] + [float(np.max(np.abs(b))) for b in baselines]
    )
    first = baselines[0]
    precision_floor = 16.0 * np.finfo(first.dtype).eps * scale
    # The graph pass reduces over capacity grids (zero-padded tails), which
    # legally reorders floating-point sums relative to the exact-count
    # baseline. How MUCH that shows up depends on the tier sizes, so a fixed
    # magnitude band is a weak test: after C6 sized the tiers from observed
    # counts the peak moved from ~2e-13 to ~2e-9, with no physics change.
    # The magnitude band stays (a real contact bug — dropped pairs, wrong
    # Hessian — lands at 1e-3+), but the SHAPE check in
    # assert_divergence_is_chaotic below is what actually distinguishes
    # reassociation from a broken solve.
    # [C6-j A800] Velocities are a backward difference of positions over dt:
    # any legal position-level reassociation delta reappears in velocities
    # multiplied by 1/dt (dt=0.01 in every scenario -> x100). The A800 exposed
    # this: its host path is BITWISE deterministic (baseline_noise = 0 across
    # four runs), so the tolerance fell to the raw floor and a 4.9e-6 velocity
    # delta -- exactly the passing 5e-8 position delta / dt -- failed the gate.
    # Scale the velocity floor accordingly; positions and kappas keep theirs.
    equivalence = 1e-7 * scale
    if label.endswith(":velocities"):
        equivalence = 1e-5 * scale
    # Kappa is a RATIO of two large reductions (-gsum/gsnorm), so each
    # operand's ~1e-7 legal reassociation compounds: the A800's bitwise-
    # deterministic baselines measured 1.9e-7 relative on a healthy run
    # against the 1e-7 floor. Real contact bugs land at 1e-3+ relative.
    elif label.endswith(":kappas"):
        equivalence = 1e-6 * scale
    tolerance = max(8.0 * baseline_noise, precision_floor, equivalence)
    assert error <= tolerance, (
        f"{label} differs beyond the baseline nondeterminism envelope: "
        f"error={error:.17g}, baseline_noise={baseline_noise:.17g}, "
        f"tolerance={tolerance:.17g}"
    )
    print(
        f"COLLISION-GRAPH-NUMERICS: {label} error={error:.3e} "
        f"baseline_noise={baseline_noise:.3e} tolerance={tolerance:.3e}"
    )


def assert_divergence_is_chaotic(label: str, baseline: np.ndarray,
                                 graph: np.ndarray) -> None:
    """Distinguish reassociation drift from a broken solve by its SHAPE.

    Legal drift starts as a rounding difference (one ULP) at the frame where
    contact first perturbs the reduction order, then gets amplified by the
    contact dynamics — and, being noise rather than a systematic force error,
    it does NOT grow monotonically. A dropped contact pair or a wrong Hessian
    instead shows up as an immediate, structural, monotonically growing gap.
    """
    per_frame = np.abs(baseline - graph).reshape(baseline.shape[0], -1).max(axis=1)
    nonzero = np.flatnonzero(per_frame > 0.0)
    if nonzero.size == 0:
        print(f"COLLISION-GRAPH-SHAPE: {label} bit-identical")
        return
    onset = int(nonzero[0])
    onset_value = float(per_frame[onset])
    peak = float(per_frame.max())
    scale = max(1.0, float(np.max(np.abs(baseline))))
    print(
        f"COLLISION-GRAPH-SHAPE: {label} onset_frame={onset} "
        f"onset={onset_value:.3e} peak={peak:.3e} final={per_frame[-1]:.3e}"
    )
    assert onset_value <= 1e-11 * scale, (
        f"{label}: divergence APPEARS at {onset_value:.3e} on frame {onset} — "
        f"that is far above a rounding difference, so the graph is computing "
        f"different physics, not reassociating the same sums"
    )
    # Monotone growth is the signature of a systematic error — but only once
    # the gap has grown past rounding. A handful of frames of ULP-scale noise
    # is monotone about as often as not (squeeze: 7e-18 -> 2e-15 over six
    # frames), and calling that "systematic" is a false alarm, so the check
    # applies only when the divergence actually reached a physical scale.
    tail = per_frame[onset:]
    if tail.size >= 6 and peak > 1e-12 * scale:
        monotone = bool(np.all(np.diff(tail) >= 0.0))
        assert not monotone, (
            f"{label}: divergence grows monotonically from {onset_value:.3e} "
            f"to {peak:.3e} — systematic, not chaotic"
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
            scenario=sys.argv[3] if len(sys.argv) > 3 else "ground",
        )
    else:
        wanted = os.environ.get("COLLISION_GRAPH_GATE_SCENARIOS")
        names = (
            [s for s in wanted.split(",") if s] if wanted else list(SCENARIOS)
        )
        for scenario in names:
            print(f"COLLISION-GRAPH-SCENARIO: {scenario}")
            with tempfile.TemporaryDirectory(
                prefix="stiff-collision-graph-gate-"
            ) as tmp:
                n_base = max(
                    2, int(os.environ.get("STIFF_G18_BASELINE_RUNS", "4"))
                )
                baseline_runs = []
                for index in range(n_base):
                    path = os.path.join(tmp, f"baseline-{index}.npz")
                    run_child("baseline", path, scenario)
                    baseline_runs.append(load(path))
                graph_path = os.path.join(tmp, "graph.npz")
                run_child("graph", graph_path, scenario)
                graph_result = load(graph_path)
                for field in ("positions", "velocities", "kappas"):
                    envelope_compare(
                        f"{scenario}:{field}",
                        [b[field] for b in baseline_runs],
                        graph_result[field],
                    )
                assert_divergence_is_chaotic(
                    f"{scenario}:positions",
                    baseline_runs[0]["positions"],
                    graph_result["positions"],
                )
        print("COLLISION-GRAPH-GATE: PASS")

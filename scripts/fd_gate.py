#!/usr/bin/env python3
"""Derivative gate for the intrusive diagnostics build.

Every scene declares the physical terms that must be live before derivatives
are checked.  This prevents a numerically green result from silently testing an
inactive contact, friction, ground, bending, or stitch path.

For each live scene and execution mode the gate checks:

* total energy versus the assembled FEM gradient over a small h sweep; and
* the assembled block-Hessian diagonal versus finite differences of that
  gradient (pure-FEM scenes only).

The h sweep is intentional: one fixed step is not reliable across cloth,
barrier, and stiff volumetric energy scales.  At least two step sizes must pass,
so a single cancellation accident cannot make the gate green.
"""

from __future__ import annotations

import gc
import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from stiff_physics.engine import Config, Engine  # noqa: E402

ASSETS = os.path.join(ROOT, "Assets") + "/"
MODES = tuple(
    value.strip()
    for value in os.environ.get("FD_MODES", "merged,strict").split(",")
    if value.strip()
)
NPROBES = int(os.environ.get("FD_NPROBES", "64"))
H_NPROBES = int(os.environ.get("FD_H_NPROBES", "32"))
GRAD_P95_TOL = float(os.environ.get("FD_GRAD_P95_TOL", "1e-3"))
GRAD_MAX_TOL = float(os.environ.get("FD_GRAD_MAX_TOL", "5e-2"))
HESS_P95_TOL = float(os.environ.get("FD_HESS_P95_TOL", "5e-3"))
HESS_MAX_TOL = float(os.environ.get("FD_HESS_MAX_TOL", "2e-2"))


def _steps(name: str, default: str) -> tuple[float, ...]:
    legacy = os.environ.get("FD_H") if name == "FD_GRAD_H_VALUES" else None
    raw = os.environ.get(name, legacy or default)
    values = tuple(float(value.strip()) for value in raw.split(",") if value.strip())
    if len(values) < 2 or any(value <= 0.0 for value in values):
        raise ValueError(f"{name} must contain at least two positive values")
    return values


GRAD_H_VALUES = _steps("FD_GRAD_H_VALUES", "1e-7,3e-8,1e-8")
HESS_H_VALUES = _steps("FD_HESS_H_VALUES", "1e-6,3e-7,1e-7")

ACTIVITY_NAMES = (
    "fem_tets",
    "triangles",
    "bending_edges",
    "soft",
    "contact",
    "ground",
    "friction",
    "ground_friction",
)


@dataclass(frozen=True)
class Scene:
    name: str
    build: Callable[[str], Engine]
    required_activity: tuple[str, ...]
    check_hessian: bool = True


def _transform(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    value = np.eye(4)
    value[:3, 3] = (x, y, z)
    return value


def _quiet(engine: Engine) -> Engine:
    engine.set_log_level(0)
    return engine


def scene_cloth_ground(mode: str) -> Engine:
    engine = _quiet(
        Engine(
            Config(
                dt=0.01,
                cloth_thickness=1e-3,
                cloth_young_modulus=1e4,
                bend_young_modulus=1e3,
                cloth_density=200,
                friction_rate=0.4,
                relative_dhat=1e-3,
                ground_offset=0.0,
                multienv_mode=mode,
                assets_dir=ASSETS,
            )
        )
    )
    engine.load_mesh(
        "triMesh/cloth_30x30.obj",
        dimensions=2,
        body_type="FEM",
        transform=_transform(y=0.03),
    )
    engine.set_body_groups([0])
    engine.finalize()
    for _ in range(10):
        engine.step()
    return engine


def scene_fem_contact_friction(mode: str) -> Engine:
    engine = _quiet(
        Engine(
            Config(
                dt=0.005,
                density=1e3,
                young_modulus=1e6,
                friction_rate=0.4,
                relative_dhat=2e-3,
                ground_offset=-2.0,
                gravity=(0.0, 0.0, 0.0),
                multienv_mode=mode,
                assets_dir=ASSETS,
            )
        )
    )
    # cube.msh spans y=[0.1, 0.5].  The 0.6 mm gap is inside dHat but
    # non-intersecting; the offset avoids coincident face triangulations.
    engine.load_mesh("tetMesh/cube.msh", 3, "FEM", _transform(y=-0.09))
    engine.load_mesh(
        "tetMesh/cube.msh", 3, "FEM", _transform(x=0.025, y=0.3106)
    )
    engine.set_body_groups([0, 0])
    engine.finalize()

    positions = engine.get_vertices().copy()
    velocities = np.zeros_like(positions)
    velocities[8:, 0] = 0.03
    velocities[8:, 1] = -0.005
    engine.teleport_fem_vertices(positions, velocities)
    engine.step()
    return engine


def scene_stitch(mode: str) -> Engine:
    engine = _quiet(
        Engine(
            Config(
                dt=0.01,
                density=1e3,
                young_modulus=1e5,
                soft_motion_rate=1e4,
                skip_all_collision=True,
                gravity=(0.0, 0.0, 0.0),
                ground_offset=-2.0,
                multienv_mode=mode,
                assets_dir=ASSETS,
            )
        )
    )
    engine.load_mesh(
        "tetMesh/cube.msh",
        3,
        "ABD",
        _transform(x=-0.6),
        boundary_type="Fixed",
    )
    engine.load_mesh("tetMesh/cube.msh", 3, "FEM", _transform(x=0.6))
    abd, fem = engine.get_load_records()
    anchor = abd.vertex_offset
    stitched = fem.vertex_offset
    anchor_x = np.asarray(engine.get_vertex_position_host(anchor))
    stitched_x = np.asarray(engine.get_vertex_position_host(stitched))
    # Deliberately non-resting by 3.7 cm so the soft gradient cannot be zero.
    rest = stitched_x - anchor_x + np.array([0.03, -0.01, 0.02])
    engine.add_stitch_spring(
        stitched, anchor, abd.body_offset, tuple(float(v) for v in rest)
    )
    engine.set_body_groups([0, 0])
    engine.finalize()
    engine.step()
    return engine


SCENES = (
    Scene(
        "cloth_ground",
        scene_cloth_ground,
        ("triangles", "bending_edges", "ground", "ground_friction"),
    ),
    Scene(
        "fem_contact",
        scene_fem_contact_friction,
        ("fem_tets", "contact", "friction"),
    ),
    # The Hessian helper intentionally rejects ABD q-space.  E-G still probes
    # the FEM side of the live ABD/FEM stitch coupling.
    Scene("stitch", scene_stitch, ("fem_tets", "soft"), check_hessian=False),
)


def _activity(engine: Engine) -> dict[str, int]:
    raw = engine.native.debug_fd_activity()
    return {name: int(value) for name, value in zip(ACTIVITY_NAMES, raw)}


def _result(raw: list[float]) -> dict[str, float]:
    return dict(
        max_rel=raw[0],
        mean_rel=raw[1],
        n=int(raw[2]),
        worst_v=int(raw[3]),
        worst_axis=int(raw[4]),
        sign=raw[5],
        p50=raw[6],
        p95=raw[7],
        nonfinite=int(raw[8]),
    )


def _sweep(
    label: str,
    values: tuple[float, ...],
    call: Callable[[float], list[float]],
    expected_n: int,
    p95_tol: float,
    max_tol: float,
) -> bool:
    passing = 0
    for h in values:
        result = _result(call(h))
        passed = (
            result["nonfinite"] == 0
            and result["n"] == expected_n
            and abs(result["sign"]) == 1.0
            and result["p95"] < p95_tol
            and result["max_rel"] < max_tol
        )
        passing += int(passed)
        verdict = "PASS" if passed else "FAIL"
        print(
            f"    {label} h={h:.0e} {verdict} "
            f"p50={result['p50']:.2e} p95={result['p95']:.2e} "
            f"max={result['max_rel']:.2e} nonfinite={result['nonfinite']} "
            f"sign={result['sign']:+.0f} "
            f"worst=(v{result['worst_v']},axis{result['worst_axis']})",
            flush=True,
        )
    # Two distinct numerical scales must agree.  Requiring all values would
    # turn normal truncation/cancellation at one end of the sweep into flakes.
    return passing >= 2


def run_case(mode: str, scene: Scene, scene_index: int) -> int:
    ok = True
    print(f"[FD] mode={mode} scene={scene.name}", flush=True)
    engine = scene.build(mode)
    try:
        activity = _activity(engine)
        missing = [name for name in scene.required_activity if activity[name] <= 0]
        print(
            "    activity "
            + " ".join(f"{name}={value}" for name, value in activity.items()),
            flush=True,
        )
        if missing:
            print(
                "    ACTIVITY FAIL inactive required terms: " + ", ".join(missing),
                flush=True,
            )
            ok = False
        else:
            seed = 12345 + 101 * scene_index
            grad_ok = _sweep(
                "E-G",
                GRAD_H_VALUES,
                lambda h: engine.native.debug_fd_gradient_check(h, NPROBES, seed),
                NPROBES,
                GRAD_P95_TOL,
                GRAD_MAX_TOL,
            )
            ok &= grad_ok
            if scene.check_hessian:
                hess_ok = _sweep(
                    "G-H",
                    HESS_H_VALUES,
                    lambda h: engine.native.debug_fd_hessian_check(
                        h, H_NPROBES, seed + 17
                    ),
                    H_NPROBES,
                    HESS_P95_TOL,
                    HESS_MAX_TOL,
                )
                ok &= hess_ok
    finally:
        del engine
        gc.collect()
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        nargs=2,
        metavar=("MODE", "SCENE"),
        help="internal: run one mode/scene in this process",
    )
    args = parser.parse_args()
    if args.case:
        mode, scene_name = args.case
        for scene_index, scene in enumerate(SCENES):
            if scene.name == scene_name:
                return run_case(mode, scene, scene_index)
        parser.error(f"unknown scene {scene_name!r}")

    if not MODES:
        raise ValueError("FD_MODES selected no execution modes")
    ok = True
    # Mode flags are process-scoped in the native implementation.  A clean
    # child per case also turns CUDA aborts into an ordinary red gate while
    # preserving the exact child output for diagnosis.
    for mode in MODES:
        for scene in SCENES:
            result = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--case", mode, scene.name],
                env=os.environ.copy(),
            )
            if result.returncode != 0:
                print(
                    f"[FD] child failed mode={mode} scene={scene.name} "
                    f"rc={result.returncode}",
                    flush=True,
                )
                ok = False

    print("FD-GATE:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

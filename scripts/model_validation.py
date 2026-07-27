#!/usr/bin/env python3
"""Constitutive/physics invariants for the active tetrahedral model."""

from __future__ import annotations

import argparse
import gc
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stiff_physics import Config, Engine  # noqa: E402
from stiff_physics.engine import _C  # noqa: E402

ASSETS = str(ROOT / "Assets") + "/"
YOUNG_MODULUS = 1e6
POISSON_RATE = 0.3
MODES = tuple(
    mode.strip()
    for mode in os.environ.get("MODEL_MODES", "merged,strict").split(",")
    if mode.strip()
)


def transform(x: float = 0.0, y: float = 0.0) -> np.ndarray:
    value = np.eye(4)
    value[0, 3] = x
    value[1, 3] = y
    return value


def cube_engine(
    mode: str,
    *,
    dt: float = 0.01,
    gravity=(0.0, 0.0, 0.0),
) -> Engine:
    engine = Engine(
        Config(
            dt=dt,
            density=1e3,
            young_modulus=YOUNG_MODULUS,
            poisson_rate=POISSON_RATE,
            gravity=gravity,
            ground_offset=-2.0,
            skip_all_collision=True,
            multienv_mode=mode,
            assets_dir=ASSETS,
        )
    )
    engine.set_log_level(0)
    # Material stiffness is a per-body load parameter.  Keep it explicit here:
    # Config.young_modulus configures the solver fallback, while load_mesh()
    # deliberately has its own backwards-compatible default (1e7).
    engine.load_mesh(
        "tetMesh/cube.msh",
        3,
        "FEM",
        np.eye(4),
        young_modulus=YOUNG_MODULUS,
    )
    engine.set_body_groups([0])
    engine.finalize()
    return engine


def constitutive_von_mises(deformation: np.ndarray, model: str) -> float:
    """Independent analytic P(F) -> Cauchy -> von Mises oracle."""
    mu = YOUNG_MODULUS / (2.0 * (1.0 + POISSON_RATE))
    lam = (
        YOUNG_MODULUS
        * POISSON_RATE
        / ((1.0 + POISSON_RATE) * (1.0 - 2.0 * POISSON_RATE))
    )
    length_rate = 4.0 * mu / 3.0
    volume_rate = lam + 5.0 * mu / 6.0
    jacobian = float(np.linalg.det(deformation))
    cofactor = jacobian * np.linalg.inv(deformation).T
    if model == "SNK1":
        first_piola = (
            length_rate * deformation
            + volume_rate
            * (jacobian - 1.0 - length_rate / volume_rate)
            * cofactor
        )
    elif model == "SNK2":
        invariant_2 = float(np.sum(deformation * deformation))
        first_piola = (
            length_rate
            * (1.0 - 1.0 / (invariant_2 + 1.0))
            * deformation
            + volume_rate
            * (
                jacobian
                - 1.0
                - 3.0 * length_rate / (4.0 * volume_rate)
            )
            * cofactor
        )
    elif model == "ARAP":
        left, _, right_t = np.linalg.svd(deformation)
        rotation = left @ right_t
        first_piola = length_rate * (deformation - rotation)
    else:
        raise AssertionError(f"unhandled constitutive model {model!r}")
    cauchy = first_piola @ deformation.T / jacobian
    deviator = cauchy - np.eye(3) * np.trace(cauchy) / 3.0
    return float(np.sqrt(1.5 * np.sum(deviator * deviator)))


def check_rest_and_objectivity(mode: str) -> None:
    engine = cube_engine(mode)
    rest = engine.get_vertices().copy()
    engine.step()
    rest_error = float(np.max(np.abs(engine.get_vertices() - rest)))
    if rest_error > 1e-12:
        raise AssertionError(f"rest-state drift {rest_error:.3e}")

    angle = 0.63
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    rigid = rest @ rotation.T + np.array([0.37, -0.21, 0.19])
    engine.teleport_fem_vertices(rigid, np.zeros_like(rigid))
    engine.step()
    objectivity_error = float(np.max(np.abs(engine.get_vertices() - rigid)))
    if objectivity_error > 1e-11:
        raise AssertionError(
            f"rigid-transform objectivity error {objectivity_error:.3e}"
        )
    print(
        f"    rest/objectivity PASS rest={rest_error:.2e} "
        f"rigid={objectivity_error:.2e}",
        flush=True,
    )


def check_uniform_motion(mode: str) -> None:
    engine = cube_engine(mode)
    initial = engine.get_vertices().copy()
    velocity = np.tile(np.array([0.1, -0.03, 0.02]), (len(initial), 1))
    engine.teleport_fem_vertices(initial, velocity)
    for _ in range(10):
        engine.step()
    expected = initial + 0.1 * velocity
    position_error = float(np.max(np.abs(engine.get_vertices() - expected)))
    velocity_error = float(
        np.max(np.abs(engine.get_vertex_velocities() - velocity))
    )
    # Strict's order-independent binned reductions can leave deterministic
    # O(1e-8 m/s) roundoff after ten implicit solves. Relative to this
    # 0.1 m/s trajectory that is below 5e-7, while still making any visible
    # momentum leak a hard failure.
    uniform_tol = float(os.environ.get("MODEL_UNIFORM_TOL", "5e-8"))
    if position_error > uniform_tol or velocity_error > uniform_tol:
        raise AssertionError(
            f"uniform-motion conservation errors pos={position_error:.3e} "
            f"vel={velocity_error:.3e}"
        )
    print(
        f"    uniform motion PASS pos={position_error:.2e} "
        f"vel={velocity_error:.2e}",
        flush=True,
    )


def check_stress_export(mode: str) -> None:
    model = _C.fem_model()
    engine = cube_engine(mode)
    rest = engine.get_vertices().copy()
    rest_stress = engine.get_fem_von_mises_stress()
    rest_max = float(np.max(np.abs(rest_stress)))
    center = np.mean(rest, axis=0)
    deformation = np.diag([1.08, 0.97, 1.02])
    deformed = (rest - center) @ deformation.T + center
    engine.teleport_fem_vertices(deformed, np.zeros_like(deformed))
    stress = engine.get_fem_von_mises_stress()
    peak = float(np.max(stress))
    expected = constitutive_von_mises(deformation, model)
    oracle_error = float(np.max(np.abs(stress - expected)))
    oracle_tolerance = max(1e-6, abs(expected) * 1e-9)
    if (
        not np.isfinite(rest_stress).all()
        or not np.isfinite(stress).all()
        or rest_max > 1e-6
        or peak <= 1.0
        or oracle_error > oracle_tolerance
    ):
        raise AssertionError(
            f"constitutive stress export invalid: rest={rest_max:.3e} "
            f"deformed={peak:.3e} expected={expected:.3e} "
            f"oracle_error={oracle_error:.3e}"
        )
    print(
        f"    stress export PASS rest={rest_max:.2e} "
        f"deformed={peak:.2e} oracle={oracle_error:.2e}",
        flush=True,
    )

    # Solver buffers are published through process-global CUDA symbols. Drop
    # the first finalized engine before building the mixed routing scene.
    del engine
    gc.collect()
    mixed = Engine(
        Config(
            gravity=(0.0, 0.0, 0.0),
            ground_offset=-2.0,
            skip_all_collision=True,
            multienv_mode=mode,
            assets_dir=ASSETS,
        )
    )
    mixed.set_log_level(0)
    mixed.load_mesh(
        "tetMesh/cube.msh",
        3,
        "ABD",
        transform(x=-0.7),
        boundary_type="Fixed",
    )
    mixed.load_mesh("tetMesh/cube.msh", 3, "FEM", transform(x=0.7))
    mixed.set_body_groups([0, 0])
    mixed.finalize()
    records = mixed.get_load_records()
    fem = records[1]
    vertices = mixed.get_vertices().copy()
    fem_slice = slice(fem.vertex_offset, fem.vertex_offset + fem.vertex_count)
    fem_center = np.mean(vertices[fem_slice], axis=0)
    vertices[fem_slice] = (
        (vertices[fem_slice] - fem_center) @ deformation.T + fem_center
    )
    mixed.teleport_fem_vertices(vertices[fem_slice], np.zeros_like(vertices[fem_slice]))
    mixed_stress = mixed.get_fem_von_mises_stress()
    abd_end = records[0].vertex_offset + records[0].vertex_count
    abd_peak = float(np.max(np.abs(mixed_stress[:abd_end])))
    fem_peak = float(np.max(mixed_stress[fem_slice]))
    if abd_peak != 0.0 or not np.isfinite(fem_peak) or fem_peak <= 1.0:
        raise AssertionError(
            f"mixed stress routing invalid: ABD={abd_peak:.3e} "
            f"FEM={fem_peak:.3e}"
        )
    print(
        f"    mixed stress routing PASS ABD={abd_peak:.2e} "
        f"FEM={fem_peak:.2e}",
        flush=True,
    )


def check_density_independent_gravity(mode: str) -> None:
    engine = Engine(
        Config(
            dt=0.01,
            gravity=(0.0, -9.8, 0.0),
            ground_offset=-2.0,
            skip_all_collision=True,
            multienv_mode=mode,
            assets_dir=ASSETS,
        )
    )
    engine.set_log_level(0)
    engine.load_mesh(
        "tetMesh/cube.msh", 3, "FEM", transform(x=-1.0), density=500.0
    )
    engine.load_mesh(
        "tetMesh/cube.msh", 3, "FEM", transform(x=1.0), density=2000.0
    )
    engine.set_body_groups([0, 0])
    engine.finalize()
    initial = engine.get_vertices().copy()
    engine.step()
    displacement = engine.get_vertices() - initial
    difference = float(np.max(np.abs(displacement[:8] - displacement[8:])))
    expected_dy = -9.8 * 0.01**2
    gravity_error = abs(float(np.mean(displacement[:, 1])) - expected_dy)
    gravity_tol = float(os.environ.get("MODEL_GRAVITY_TOL", "1e-10"))
    if difference > gravity_tol or gravity_error > gravity_tol:
        raise AssertionError(
            f"density/gravity mismatch cross={difference:.3e} "
            f"analytic={gravity_error:.3e}"
        )
    print(
        f"    density-independent gravity PASS cross={difference:.2e} "
        f"step={gravity_error:.2e}",
        flush=True,
    )


def ballistic_error(mode: str, dt: float, frames: int) -> float:
    engine = cube_engine(mode, dt=dt, gravity=(0.0, -9.8, 0.0))
    initial_y = float(np.mean(engine.get_vertices()[:, 1]))
    for _ in range(frames):
        engine.step()
    elapsed = dt * frames
    analytic_y = initial_y - 0.5 * 9.8 * elapsed**2
    return abs(float(np.mean(engine.get_vertices()[:, 1])) - analytic_y)


def check_timestep_refinement(mode: str) -> None:
    coarse = ballistic_error(mode, 0.02, 5)
    fine = ballistic_error(mode, 0.01, 10)
    ratio = fine / coarse if coarse else 0.0
    if not (fine < coarse and 0.35 < ratio < 0.65):
        raise AssertionError(
            f"first-order timestep refinement failed: coarse={coarse:.3e} "
            f"fine={fine:.3e} ratio={ratio:.3f}"
        )
    print(
        f"    timestep refinement PASS coarse={coarse:.2e} "
        f"fine={fine:.2e} ratio={ratio:.3f}",
        flush=True,
    )


def check_ground_friction(mode: str) -> None:
    engine = Engine(
        Config(
            dt=0.005,
            density=1e3,
            young_modulus=1e6,
            gravity=(0.0, -9.8, 0.0),
            ground_offset=0.0,
            relative_dhat=1e-3,
            friction_rate=0.4,
            multienv_mode=mode,
            assets_dir=ASSETS,
        )
    )
    engine.set_log_level(0)
    for x, mu in ((-1.0, 0.0), (1.0, 0.8)):
        engine.load_mesh(
            "tetMesh/cube.msh", 3, "FEM", transform(x=x, y=-0.099)
        )
        body = len(engine.get_load_records()) - 1
        engine.set_body_friction(body, mu, mu)
    engine.set_body_groups([0, 0])
    engine.finalize()
    positions = engine.get_vertices().copy()
    velocity = np.zeros_like(positions)
    velocity[:, 0] = 0.2
    engine.teleport_fem_vertices(positions, velocity)
    for _ in range(4):
        engine.step()
    final_velocity = engine.get_vertex_velocities()
    frictionless = float(np.mean(final_velocity[:8, 0]))
    frictional = float(np.mean(final_velocity[8:, 0]))
    if abs(frictionless - 0.2) > 1e-5 or not (0.0 < frictional < 0.1):
        raise AssertionError(
            f"ground friction response invalid: mu0={frictionless:.6f} "
            f"mu0.8={frictional:.6f}"
        )
    print(
        f"    ground friction PASS mu0={frictionless:.6f} "
        f"mu0.8={frictional:.6f}",
        flush=True,
    )


def run_case(mode: str) -> int:
    model = _C.fem_model()
    expected = os.environ.get("MODEL_EXPECT")
    if model not in ("SNK1", "SNK2", "ARAP"):
        raise AssertionError(f"unknown compiled model {model!r}")
    if expected and model != expected.upper():
        raise AssertionError(f"compiled model {model} != expected {expected.upper()}")
    print(f"[model] mode={mode} constitutive={model}", flush=True)
    check_rest_and_objectivity(mode)
    check_uniform_motion(mode)
    check_stress_export(mode)
    check_density_independent_gravity(mode)
    check_timestep_refinement(mode)
    check_ground_friction(mode)
    gc.collect()
    print(f"[model] mode={mode} PASS", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("merged", "isolated", "strict"))
    args = parser.parse_args()
    if args.case:
        return run_case(args.case)
    ok = True
    for mode in MODES:
        result = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--case", mode],
            env=os.environ.copy(),
        )
        ok &= result.returncode == 0
    print("MODEL-VALIDATION:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

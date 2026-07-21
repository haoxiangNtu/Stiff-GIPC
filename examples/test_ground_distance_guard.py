#!/usr/bin/env python3
"""Regression: relative ground CCD stays strictly in the IPC barrier domain."""

from pathlib import Path
import sys

import numpy as np

from stiff_physics.engine import Config, Engine


ASSETS_DIR = Path(__file__).resolve().parent.parent / "Assets"


def make_engine(mode: str, clearance: float, ground_offset: float = 0.0) -> Engine:
    engine = Engine(
        Config(
            dt=0.01,
            ground_offset=ground_offset,
            assets_dir=str(ASSETS_DIR) + "/",
            multienv_mode=mode,
        )
    )
    transform = np.eye(4)
    transform[1, 3] = ground_offset - 0.1 + clearance
    engine.load_mesh(
        "tetMesh/cube.msh",
        dimensions=3,
        body_type="FEM",
        transform=transform,
    )
    engine.set_body_groups([0])
    return engine


def check_exact_zero(mode: str) -> None:
    engine = make_engine(mode, 0.0)

    try:
        engine.finalize()
        engine.step()
    except RuntimeError as error:
        message = str(error)
        if "ground distance is non-finite or non-positive" not in message:
            raise AssertionError(f"unexpected ground guard error: {message}") from error
        print(f"PASS [{mode}]: exact-zero initial ground distance failed immediately")
        return

    raise AssertionError("exact-zero initial ground distance was accepted")


def check_near_boundary(mode: str) -> None:
    ground_offset = -1.0
    clearance = 2.0 * (np.nextafter(ground_offset, np.inf) - ground_offset)
    engine = make_engine(mode, clearance, ground_offset)
    engine.finalize()
    min_distance = float(np.min(engine.get_vertices()[:, 1]) - ground_offset)
    if not np.isfinite(min_distance) or min_distance <= 0.0 or min_distance >= 1e-12:
        raise AssertionError(f"expected a positive ULP-scale initial gap: {min_distance}")
    for _ in range(20):
        engine.step()
        distance = float(np.min(engine.get_vertices()[:, 1]) - ground_offset)
        min_distance = min(min_distance, distance)
    if not np.isfinite(min_distance) or min_distance <= 0.0:
        raise AssertionError(f"ground trial position left the strict domain: {min_distance}")
    print(f"PASS [{mode}]: near-boundary trajectory stayed strictly feasible; min={min_distance:.3e} m")


def main() -> None:
    scenario = sys.argv[1] if len(sys.argv) > 1 else "exact"
    mode = sys.argv[2] if len(sys.argv) > 2 else "merged"
    if scenario == "exact":
        check_exact_zero(mode)
    elif scenario in {"near", "margin"}:
        check_near_boundary(mode)
    else:
        raise ValueError(f"unknown scenario {scenario!r}; use exact or near")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Regression: exact ground contact is outside the IPC barrier domain."""

from pathlib import Path

import numpy as np

from stiff_physics.engine import Config, Engine


ASSETS_DIR = Path(__file__).resolve().parent.parent / "Assets"


def main() -> None:
    engine = Engine(
        Config(
            dt=0.01,
            ground_offset=0.0,
            assets_dir=str(ASSETS_DIR) + "/",
        )
    )
    transform = np.eye(4)
    transform[1, 3] = -0.1
    engine.load_mesh(
        "tetMesh/cube.msh",
        dimensions=3,
        body_type="FEM",
        transform=transform,
    )
    engine.set_body_groups([0])

    try:
        engine.finalize()
        engine.step()
    except RuntimeError as error:
        message = str(error)
        if "ground distance is non-finite or non-positive" not in message:
            raise AssertionError(f"unexpected ground guard error: {message}") from error
        print("PASS: exact-zero initial ground distance failed immediately")
        return

    raise AssertionError("exact-zero initial ground distance was accepted")


if __name__ == "__main__":
    main()

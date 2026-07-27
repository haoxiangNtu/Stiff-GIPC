#!/usr/bin/env python3
"""Negative-test gate for the init-feasibility GeometryError taxonomy.

Three cases, each in its own subprocess (one Engine per process):
  below_ground     — FEM cube spawned through the ground plane must raise
                     GeometryError at finalize() (host-side init scan).
  interpenetrating — two overlapping FEM cubes pass finalize but must raise
                     GeometryError on the first step (frame-0 line search
                     exhausted with non-finite incremental potential).
  clean            — a well-posed scene must NOT raise (false-positive guard).

Historically both broken configurations produced a silent NaN cascade with no
nameable error; this gate pins the typed-throw contract.
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ASSETS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Assets"
) + "/"
CASES = ("below_ground", "interpenetrating", "clean")


def transform(x: float, y: float):
    import numpy as np

    value = np.eye(4)
    value[0, 3] = x
    value[1, 3] = y
    return value


def run_case(case: str) -> int:
    from stiff_physics import Config, Engine, GeometryError

    def build(ground: float) -> Engine:
        engine = Engine(
            Config(
                dt=0.01,
                density=1e3,
                young_modulus=1e6,
                gravity=(0.0, -9.8, 0.0),
                ground_offset=ground,
                multienv_mode="merged",
                assets_dir=ASSETS,
            )
        )
        engine.set_log_level(0)
        return engine

    if case == "below_ground":
        # cube.msh natively spans y in [0.1, 0.5]; a ground plane at 0.3 cuts
        # straight through it -> finalize must refuse.
        engine = build(ground=0.3)
        engine.load_mesh("tetMesh/cube.msh", 3, "FEM", transform(0.0, 0.0))
        engine.set_body_groups([0])
        try:
            engine.finalize()
        except GeometryError as exc:
            print(f"    below_ground: typed refusal at finalize: {exc}")
            return 0
        print("    below_ground: finalize ACCEPTED a ground-penetrating body")
        return 1

    if case == "interpenetrating":
        # Two FEM cubes half-overlapped in x; ground far below so only the
        # frame-0 infeasibility path can fire.
        engine = build(ground=-2.0)
        engine.load_mesh("tetMesh/cube.msh", 3, "FEM", transform(0.0, 0.0))
        engine.load_mesh("tetMesh/cube.msh", 3, "FEM", transform(0.2, 0.0))
        engine.set_body_groups([0, 0])
        engine.finalize()
        try:
            engine.step()
        except GeometryError as exc:
            print(f"    interpenetrating: typed refusal on first step: {exc}")
            return 0
        print(
            "    interpenetrating: first step ACCEPTED an interpenetrating "
            "initial state (silent-NaN regression)"
        )
        return 1

    # clean
    engine = build(ground=-0.5)
    engine.load_mesh("tetMesh/cube.msh", 3, "FEM", transform(0.0, 0.0))
    engine.set_body_groups([0])
    engine.finalize()
    engine.step()
    engine.step()
    print("    clean: two steps, no exception")
    return 0


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES)
    args = parser.parse_args()
    if args.case:
        return run_case(args.case)
    ok = True
    for case in CASES:
        print(f"[geometry] case={case}", flush=True)
        result = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--case", case],
            env=os.environ.copy(),
        )
        ok &= result.returncode == 0
    print("GEOMETRY-GATE:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

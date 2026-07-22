#!/usr/bin/env python3
"""Persistent-LS regression: real multi-million-pair, multi-grow jaw scene.

The earlier overflow probe used two cubes and tiny injected capacities (tens of
pairs).  That did not exercise an independently growing friction last-H buffer
after the LS graph family had already been cached.  This worker drives the
single-env finray jaw long enough to cross both boundaries:

* at least two CCD grow-redo passes with a required count above three million;
* a second friction-buffer growth, followed by an LS graph-family rebuild;
* a finite accepted state after the old stale-address failure point.

The parent captures C/C++ stdout from a fresh process so the assertions verify
that the intended growth path actually ran instead of merely checking exit 0.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _worker() -> None:
    import numpy as np

    sys.path.insert(0, str(ROOT / "examples"))
    import umi_finray_lib as L

    prep = L.prepare_scene("foldshirt")
    params = L._drive_params()
    engine = L.make_engine(prep, 1)
    sides = L.load_finray_sides()
    envs, _ = L.build_world(engine, prep, 1, 4.0, sides, dict(abd_cursor=0))
    L._stitch_seg_arrays(envs)
    engine.finalize()

    robot = L._setup_after_finalize(engine, envs, params)
    L.arm_force_barrier(engine, robot, params)
    env_joints = L.slice_env_joints(robot, 1)
    groups = L.build_drive_groups(envs, env_joints)
    jaw_indices = next(g for g in groups if g["key"] == (0, "L"))["pris"]
    close_force = params.get("barrier_force", params.get("k_grip", 3.0))

    for _ in range(20):
        for index in jaw_indices:
            opened, closed = L._open_close(robot, index)
            close_direction = 1.0 if closed - opened > 0.0 else -1.0
            engine.native.set_prismatic_strength(index, 0.0)
            engine.native.set_prismatic_force(
                index, close_direction * close_force
            )
        engine.step()

    vertices = np.asarray(engine.get_vertices())
    if not np.isfinite(vertices).all():
        raise AssertionError("large-grow jaw produced a non-finite vertex")
    print("LARGE_GROW_FINITE")


def _parent() -> None:
    env = dict(os.environ)
    env.update(
        PYTHONUNBUFFERED="1",
        STIFF_LS_GRAPH="1",
        STIFF_LS_GRAPH_DIAG="1",
        CASE39_QUIET="1",
        # State the production-scale starting cap explicitly.  Pair counts, not
        # an artificial one-pair cap, must force the repeated growth here.
        CASE39ME_BUFF_SCALE="4.0",
    )
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = result.stdout + result.stderr
    if result.returncode != 0:
        print(output, end="")
        raise SystemExit(result.returncode)

    ccd_grows = re.findall(
        r"\[CCD-grow\] h_ccd_cpNum=(\d+) > cap=(\d+) -> grow to (\d+)",
        output,
    )
    if len(ccd_grows) < 2:
        raise AssertionError(f"expected repeated CCD grow-redo, saw {ccd_grows}")
    if max(int(required) for required, _, _ in ccd_grows) < 3_000_000:
        raise AssertionError(f"growth stayed in synthetic-small range: {ccd_grows}")

    friction_grows = [
        match.start() for match in re.finditer(r"\[friction-grow\]", output)
    ]
    if len(friction_grows) < 2:
        raise AssertionError(
            "scene did not cross the cached friction-buffer capacity twice"
        )
    rebound = output.find("[ls-graph-cache]", friction_grows[-1])
    if rebound < 0:
        raise AssertionError("LS graph family was not rebound after friction grow")
    if "LARGE_GROW_FINITE" not in output:
        raise AssertionError("large-grow worker did not publish a finite terminal state")

    for required, old_cap, new_cap in ccd_grows:
        print(f"CCD grow {required} > {old_cap} -> {new_cap}")
    print(f"friction grows={len(friction_grows)}, post-grow LS rebind=yes")
    print("LS LARGE-GROW: PASS")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--worker":
        _worker()
    else:
        _parent()

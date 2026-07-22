#!/usr/bin/env python3
"""Small N-env contact scene used by the frame-FSM sanitizer gates."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stiff_physics.engine import Config, Engine


mode = sys.argv[1] if len(sys.argv) > 1 else "strict"
num_envs = int(os.environ.get("SAN_N", "3"))
config = Config(
    dt=0.01,
    density=1e3,
    young_modulus=1e6,
    friction_rate=0.4,
    relative_dhat=1e-3,
    ground_offset=0.0,
    assets_dir=str(ROOT / "Assets") + "/",
    multienv_mode=mode,
    preconditioner_type=1,
)
config._cfg.absolute_dhat = 1e-3
engine = Engine(config)

for _ in range(num_envs):
    engine.load_mesh("tetMesh/cube.msh", 3, "ABD", np.eye(4))
for _ in range(num_envs):
    transform = np.eye(4)
    transform[0, 3] = 0.6
    transform[1, 3] = -0.05
    engine.load_mesh("tetMesh/cube.msh", 3, "FEM", transform)

engine.native.set_body_groups(list(range(num_envs)) * 2)
engine.finalize()
for _ in range(3):
    engine.step()

vertices = np.asarray(engine.get_vertices())
assert np.isfinite(vertices).all()
status = engine.get_frame_status()
assert int(status.result) == 0
if os.environ.get("STIFF_FRAME_GRAPH", "0") not in ("", "0"):
    assert int(status.path_flags) & (1 << 1)
print(
    f"FRAME_SAN_SCENE_DONE mode={mode} N={num_envs} "
    f"newton={int(status.newton_iters)} pcg={int(status.pcg_iters)}"
)

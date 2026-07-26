#!/usr/bin/env python3
"""Strict bitwise anchor scene (v0.8.6 gate suite).

2 ABD cubes + 2 FEM cubes, groups [0,1,0,1], strict flags, 50 frames.
Prints VHASH <sha256[:16]> of the final vertex buffer. The gold value is
pinned in scripts/verify_gates.sh — any bit-level drift of the strict stack
changes it. Env knobs: SCENE_N (default 2), SCENE_FRAMES (default 50).
"""
import os, sys
import hashlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from stiff_physics.engine import Engine, Config

cfg = Config(dt=0.01, density=1e3, young_modulus=1e6, friction_rate=0.4,
             relative_dhat=1e-3, ground_offset=0.0,
             assets_dir=os.path.join(ROOT, "Assets") + "/",
             multienv_mode=os.environ.get("SCENE_MODE", "strict"),
             preconditioner_type=1)
cfg._cfg.absolute_dhat = 1e-3
eng = Engine(cfg)
N = int(os.environ.get("SCENE_N", "2"))
for _ in range(N):
    eng.load_mesh("tetMesh/cube.msh", 3, "ABD", np.eye(4))
for _ in range(N):
    tf = np.eye(4); tf[0, 3] = 0.6; tf[1, 3] = -0.05
    eng.load_mesh("tetMesh/cube.msh", 3, "FEM", tf)
eng.native.set_body_groups(list(range(N)) * 2)
eng.finalize()
for _ in range(int(os.environ.get("SCENE_FRAMES", "50"))):
    eng.step()
V = np.asarray(eng.get_vertices())
assert np.isfinite(V).all(), "non-finite!"
print("VHASH", hashlib.sha256(V.tobytes()).hexdigest()[:16])
print("NEWTON", eng.native.get_total_newton_iters())   # [C3] envelope input
if os.environ.get("DUMP_POS"):                          # [C3] equivalence input
    np.save(os.environ["DUMP_POS"], V)
print("SCENE_OK")

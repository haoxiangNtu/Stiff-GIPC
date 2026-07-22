#!/usr/bin/env python3
"""[P3] strict-mode determinism quad-gate (self-contained, no golden files).

Small scene per env: one ABD cube + one FEM cube dropped onto the ground
(contact + friction + two body classes), N envs CO-LOCATED (identical physics
coordinates -> identical rounding), separated only by virtual BVH offsets +
body-group exclusions; absolute_dhat pinned so contact width cannot inflate
with the all-env bounding box. This is the v0.8.3 bitwise-trio configuration.

Five gates, all BIT-IDENTICAL requirements:
  G1 run-to-run @ N=2 : two full runs agree exactly
  G2 run-to-run @ N=8 : two full runs agree exactly
  G3 cross-env  @ N=8 : env_i == env_0 exactly (co-located)
  G4 batch-invariance : env_0 @ N=2 == env_0 @ N=8 exactly
  G5 diagonal-PC cross-check: batch-invariance + cross-env repeat under
     preconditioner_type=0 (run-to-run is owned by G1/G2 on the MAS path)

G1-G4 run under the DEFAULT preconditioner (MAS, preconditioner_type=1).
v0.8.4.2 made MAS itself batch-invariant: hierarchy depth is computed from
the per-env node count, and the two coarse-aggregation fast paths whose
warp-shuffle trees produced launch-shape-dependent partial sums (one of them
reading CUDA-undefined lanes outside __activemask) were replaced on the
STRICT path with exact order-independent binned deposits (non-strict keeps a
corrected full-participation tree). Before that fix env_0 drifted ~1e-9
between N=2 and N=8; any regression re-fires G4 bit-exactly.

These four caught the v0.8.4 DCD/CCD shared-buffer race (fixed by the DCD
snapshot); any regression in per-env alpha isolation re-fires here.

Run:  python3 examples/test_strict_quadgate.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stiff_physics.engine import Engine, Config

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Assets") + "/"
FRAMES = 50
BVH_SPACING = 5.0  # virtual broadphase separation; physics stays co-located


def run(num_envs: int, pc_type: int = 0) -> np.ndarray:
    cfg = Config(dt=0.01, density=1e3, young_modulus=1e6, friction_rate=0.4,
                 relative_dhat=1e-3, ground_offset=0.0, assets_dir=ASSETS,
                 multienv_mode="strict", preconditioner_type=pc_type)
    # pin contact width: relative_dhat would scale with the all-env bbox and
    # break batch-invariance (env0@N2 vs env0@N8) by design, not by bug.
    cfg._cfg.absolute_dhat = 1e-3
    eng = Engine(cfg)
    for _ in range(num_envs):  # ABD cube per env (co-located across envs)
        eng.load_mesh("tetMesh/cube.msh", 3, "ABD", np.eye(4))
    for _ in range(num_envs):  # FEM cube per env, beside it, 5 cm lower drop
        tf = np.eye(4); tf[0, 3] = 0.6; tf[1, 3] = -0.05
        eng.load_mesh("tetMesh/cube.msh", 3, "FEM", tf)
    # env groups: ABD bodies first, then FEM bodies (engine body order)
    eng.native.set_body_groups(list(range(num_envs)) + list(range(num_envs)))
    eng.finalize()
    if num_envs > 1:  # virtual BVH offsets keep broadphase per-env (AFTER finalize)
        flat = []
        for i in range(num_envs):
            flat += [BVH_SPACING * i, 0.0, 0.0]
        eng.native.set_env_offsets(flat)
    for _ in range(FRAMES):
        eng.step()
    V = np.asarray(eng.get_vertices()).copy()
    assert np.isfinite(V).all()
    return V


def env_slice(V: np.ndarray, num_envs: int, i: int) -> np.ndarray:
    """Vertices of env i (co-located: no offset removal). [all ABD][all FEM]."""
    abd = V[8 * i: 8 * (i + 1)]
    fem = V[8 * num_envs + 8 * i: 8 * num_envs + 8 * (i + 1)]
    return np.vstack([abd, fem])


def bitwise(a, b):
    return a.shape == b.shape and bool((a == b).all())


def main():
    a2, b2 = run(2, pc_type=1), run(2, pc_type=1)
    a8, b8 = run(8, pc_type=1), run(8, pc_type=1)
    d2, d8 = run(2, pc_type=0), run(8, pc_type=0)

    g1 = bitwise(a2, b2)
    g2 = bitwise(a8, b8)
    g3 = all(bitwise(env_slice(a8, 8, i), env_slice(a8, 8, 0)) for i in range(1, 8))
    g4 = bitwise(env_slice(a2, 2, 0), env_slice(a8, 8, 0))

    def report(name, ok, aa, bb):
        if ok:
            print(f"{name}: PASS (bit-identical)")
        else:
            d = np.abs(aa - bb).max() if aa.shape == bb.shape else float("inf")
            print(f"{name}: FAIL (max |diff| = {d:.3e})")

    report("G1 run-to-run N=2   ", g1, a2, b2)
    report("G2 run-to-run N=8   ", g2, a8, b8)
    report("G3 cross-env  N=8   ", g3, env_slice(a8, 8, 1), env_slice(a8, 8, 0))
    report("G4 batch N=2 vs N=8 ", g4, env_slice(a2, 2, 0), env_slice(a8, 8, 0))

    g5 = (bitwise(env_slice(d2, 2, 0), env_slice(d8, 8, 0))
          and all(bitwise(env_slice(d8, 8, i), env_slice(d8, 8, 0)) for i in range(1, 8)))
    print(f"G5 diag-PC cross-check: {'PASS (bit-identical)' if g5 else 'FAIL'}")

    ok = g1 and g2 and g3 and g4 and g5
    print("QUAD-GATE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

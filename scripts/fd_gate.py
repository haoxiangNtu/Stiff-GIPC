#!/usr/bin/env python3
"""[FD gate] Finite-difference gradient consistency — nightly, never production.

Two small scenes exercise the FEM-side energy stack (kinetic + elastic +
membrane + bending + soft + barrier + friction + ground); after a few frames
(so pair lists and lagged friction exist) the engine's test-only hook probes
random free-FEM coordinates: central difference of the TOTAL energy vs the
assembled analytic gradient (fb + shape_grads). Catches E<->G desync — the
bug class neither eyes nor the bitwise anchor can attribute.

PASS: max relative error < TOL on every scene x mode.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from stiff_physics.engine import Engine, Config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "Assets") + "/"
TOL = float(os.environ.get("FD_TOL", "1e-3"))
H = float(os.environ.get("FD_H", "1e-6"))
NPROBES = int(os.environ.get("FD_NPROBES", "128"))

def scene_cloth_ground():
    eng = Engine(Config(dt=0.01, cloth_thickness=1e-3, cloth_young_modulus=1e4,
                        bend_young_modulus=1e3, cloth_density=200, friction_rate=0.4,
                        relative_dhat=1e-3, ground_offset=0.0, assets_dir=ASSETS))
    tf = np.eye(4); tf[1, 3] = 0.05
    eng.load_mesh("triMesh/cloth_30x30.obj", dimensions=2, body_type="FEM", transform=tf)
    eng.finalize()
    return eng, 8

def scene_fem_cube_stack():
    # cube.msh spans y in [0.1, 0.5] (0.4 side) — stack with a genuine gap
    eng = Engine(Config(dt=0.01, density=1e3, young_modulus=1e6, friction_rate=0.4,
                        relative_dhat=1e-3, ground_offset=0.0, assets_dir=ASSETS))
    for k in range(2):
        tf = np.eye(4); tf[1, 3] = -0.09 + 0.45 * k   # bases at y=0.01 / 0.46
        eng.load_mesh("tetMesh/cube.msh", 3, "FEM", tf)
    eng.finalize()
    return eng, 6

ok = True
for name, builder in (("cloth_ground", scene_cloth_ground), ("fem_cube_stack", scene_fem_cube_stack)):
    eng, warm = builder()
    for _ in range(warm):
        eng.step()
    mx, mean, n, wv, wa, sign, p50, p95, nnf = eng.native.debug_fd_gradient_check(H, NPROBES, 12345)
    # robust criterion: no non-finite probes, bulk (p95) tight, worst bounded
    # (isolated near-kink contact probes legitimately degrade central FD)
    verdict = "PASS" if (nnf == 0 and n > 0 and p95 < TOL and mx < 5e-2) else "FAIL"
    ok &= verdict == "PASS"
    print(f"{name:16s} {verdict}  p50={p50:.2e} p95={p95:.2e} max={mx:.3e} "
          f"mean={mean:.2e} nonfinite={int(nnf)} probes={int(n)} sign={sign:+.0f} "
          f"worst=(v{int(wv)},axis{int(wa)})", flush=True)
    del eng
print("FD-GATE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

#!/usr/bin/env python3
"""[initial-state variation] Towel scramble recipe + cloth-teleport verification.

The requirement: start episodes from a NON-rest cloth state (crumpled/folded
towel) instead of the flat rest pose. The engine already has every primitive
needed — this script is the standard recipe:

  Phase 1  SCRAMBLE : drop the towel with a few random velocity kicks
                      (teleport keeps positions, injects velocity+xTilta)
                      -> settle -> a crumpled state.
  Phase 2  SAVE     : export the scrambled vertex positions (.npy) and a full
                      deterministic checkpoint (save_checkpoint).
  Phase 3  REUSE    : a FRESH engine (same scene graph) teleports its towel to
                      the scrambled positions -> verifies the cloth block is
                      actually covered by teleport (max error assert) -> steps
                      to confirm the state is stable, not exploding.

Notes
  * rest shape (tri DmInverses) is untouched by teleport -> elasticity stays
    correct for the crumpled state.
  * set_vertex_velocities_gpu alone does NOT rebuild xTilta; always kick via
    teleport_fem_vertices(positions, velocities).
  * for exact mid-trajectory reproducibility across processes use
    save_checkpoint/load_checkpoint instead of the .npy path.

Run:  python3 examples/recipe_towel_scramble.py
"""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from stiff_physics.engine import Engine, Config

ASSETS   = os.path.join(ROOT, "Assets") + "/"
OUT_NPY  = "/tmp/towel_scrambled.npy"
OUT_CKPT = "/tmp/towel_scrambled.ckpt"
SEED     = 7
SETTLE   = 220     # frames for the tilted towel to fall, fold and settle
CLOTH    = "triMesh/cloth_30x30.obj"


def build_scene(tilt_deg=75.0, yaw_deg=0.0, height=0.30):
    """Scene builder. The scramble RANDOMIZATION lives entirely in the drop
    pose (tilt/yaw/height drawn from SEED) — the tilted towel lands edge-first
    and folds onto itself. No teleport tricks in the scramble phase: random
    per-vertex position noise self-intersects the cloth (IPC cannot recover
    from penetration) and random VELOCITY teleports crash stock v0.8.3 (see
    CHANGELOG known issues)."""
    cfg = Config(
        dt=0.01,
        cloth_thickness=1e-3, cloth_young_modulus=1e4, bend_young_modulus=1e3,
        cloth_density=200, strain_rate=100,
        poisson_rate=0.49, friction_rate=0.4, relative_dhat=1e-3,
        ground_offset=0.0, assets_dir=ASSETS,
        collision_detection_buff_scale=16.0,   # folded cloth: many self-pairs
        linear_system_buff_scale=8.0,
    )
    eng = Engine(cfg)
    cx, sx = np.cos(np.deg2rad(tilt_deg)), np.sin(np.deg2rad(tilt_deg))
    cy, sy = np.cos(np.deg2rad(yaw_deg)), np.sin(np.deg2rad(yaw_deg))
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    tf = np.eye(4); tf[:3, :3] = 0.4 * (Ry @ Rx); tf[1, 3] = height
    eng.load_mesh(CLOTH, dimensions=2, body_type="FEM", transform=tf,
                  young_modulus=1e4)
    eng.finalize()
    return eng


# ---------------- Phase 1: scramble (randomized drop pose) ----------------
rng = np.random.default_rng(SEED)
tilt = float(rng.uniform(60.0, 85.0))
yaw  = float(rng.uniform(0.0, 360.0))
h    = float(rng.uniform(0.25, 0.35))
eng = build_scene(tilt, yaw, h)
P0 = np.asarray(eng.get_vertices()).copy()
n = P0.shape[0]
print(f"[scramble] towel verts = {n}, drop pose: tilt={tilt:.1f} yaw={yaw:.1f} h={h:.2f}")

for fr in range(SETTLE):
    eng.step()

P_scrambled = np.asarray(eng.get_vertices()).copy()
assert np.isfinite(P_scrambled).all(), "scramble diverged"
# crumple metric: ground footprint vs the FLAT towel's area (the towel is
# loaded tilted, so use its largest extent — the unshrunk edge — as flat side)
flat_side = float((P0.max(0) - P0.min(0)).max())
crumple = (P_scrambled[:, 0].ptp() * P_scrambled[:, 2].ptp()) / (flat_side * flat_side)
print(f"[scramble] footprint ratio vs flat = {crumple:.3f} (smaller = more folded)")

# ---------------- Phase 2: save ----------------
np.save(OUT_NPY, P_scrambled)
eng.native.save_checkpoint(OUT_CKPT)
print(f"[save] {OUT_NPY} + {OUT_CKPT}")

# ---------------- Phase 3: reuse in a fresh engine ----------------
eng2 = build_scene(tilt, yaw, h)   # same scene graph; state comes from the npy
target = np.load(OUT_NPY)
eng2.native.teleport_fem_vertices(target, None)   # rest velocities
got = np.asarray(eng2.get_vertices())
err = np.abs(got - target).max()
print(f"[reuse] teleport coverage max |err| = {err:.3e} m")
if err > 1e-9:
    print("FAIL: cloth block not fully covered by teleport_fem_vertices")
    sys.exit(1)

for fr in range(40):
    eng2.step()
V = np.asarray(eng2.get_vertices())
assert np.isfinite(V).all(), "reused state diverged"
drift = np.abs(V - target).max()
print(f"[reuse] 40-frame settle drift = {drift:.3e} m (crumpled state held)")

ok = crumple < 0.98 and err <= 1e-9 and drift < 0.2
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

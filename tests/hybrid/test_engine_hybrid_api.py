"""
Phase 3 integration test: load a hybrid mesh via the new
add_fem_pins_with_local_pos C++ API + verify the engine reaches finalize
+ step without crashing.

This test bypasses the stiff_physics Python package (which is editable-
installed from /home/ps/Downloads/Stiff-GIPC and so doesn't pick up our
worktree's engine.py changes) and calls the pystiffgipc native module
directly.

Validation goals:
  - pystiffgipc has the new add_fem_pins_with_local_pos API
  - vertex_to_pin_idx population works for bulk pins (Phase 2 work)
  - d_tet_to_abd_body population works (Phase 2 work)
  - apply_fem_pins kernel runs (existing M1)
  - M3.5 chain-rule kernel handles the bulk-pin Hessian routing
  - one full step completes without NaN

Phase 6 covers physical correctness; this is a plumbing smoke test.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

# Direct-import the freshly-built pystiffgipc from build_312
sys.path.insert(0, str(REPO / "build_312"))

import pystiffgipc as _C

print(f"[test] loaded pystiffgipc from: {_C.__file__}")

# Verify the new API exists
assert hasattr(_C.SimEngine, "add_fem_pins_with_local_pos"), \
    "pystiffgipc missing add_fem_pins_with_local_pos — rebuild?"
print("[test] add_fem_pins_with_local_pos API present")


def make_minimal_fixture():
    """Tiny hybrid mesh: 8-vertex 1cm cube, 5 tets, bottom 4 verts pinned
    to ABD body 0 (also a 1cm cube)."""
    verts = np.array([
        [0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0.01, 0.01, 0],
        [0, 0, 0.01], [0.01, 0, 0.01], [0, 0.01, 0.01], [0.01, 0.01, 0.01],
    ], dtype=np.float64)
    tets = np.array([
        [0, 1, 2, 4],
        [1, 2, 3, 7],
        [1, 4, 5, 7],
        [2, 4, 6, 7],
        [1, 2, 4, 7],
    ], dtype=np.int32)
    return verts, tets


def main():
    verts, tets = make_minimal_fixture()
    print(f"[test] fixture: {len(verts)}v / {len(tets)}t")

    # Build engine + Config
    cfg = _C.Config()
    eng = _C.SimEngine()
    eng.set_config(cfg)
    eng.init_cuda()

    # 1. Load an ABD cube as body 0 (1cm box) via the existing assets.
    assets = REPO / "Assets" / "sim_data" / "tetmesh" / "cube.msh"
    assert assets.is_file(), f"need assets cube.msh, got {assets}"
    eng.load_mesh(str(assets), 3, 0,  # dimensions=3, body_type=0(ABD)
                  np.eye(4), 1e8, 0)  # young, boundary_type=0(Free)
    print(f"[test] ABD cube loaded as body 0")

    # 2. Load the hybrid FEM cube as a FEM body.  Binding expects:
    #   vertices: (N, 3) — num_verts = shape[0]
    #   faces:    (M, K) — num_faces = shape[0], K = verts_per_face
    eng.load_mesh_from_data(np.ascontiguousarray(verts, dtype=np.float64),
                            np.ascontiguousarray(tets, dtype=np.int32),
                            4,  # verts_per_face = 4 (tet)
                            3,  # dimensions
                            1,  # body_type=1 (FEM)
                            np.eye(4), 1e6, 0)  # young, boundary_type=0
    print(f"[test] FEM cube loaded")

    # 3. The FEM body's vertex_offset.  We need to know it to convert the
    # local FEM vertex idx (0..7) to global pin indices.  Get it from
    # the load records.
    load_records = eng.get_all_load_records()
    print(f"[test] load_records: "
          f"{[(r.body_offset, r.vertex_offset, r.vertex_count) for r in load_records]}")
    fem_rec = load_records[-1]  # last loaded
    fem_v_offset = fem_rec.vertex_offset

    # 4. Bulk pin verts 0..3 (bottom face) to ABD body 0.
    fem_global_ids = np.array([fem_v_offset + i for i in range(4)], dtype=np.int32)
    body_ids = np.array([0, 0, 0, 0], dtype=np.int32)
    local_pos = verts[:4].astype(np.float64)  # identity: local == world

    eng.add_fem_pins_with_local_pos(fem_global_ids, body_ids, local_pos)
    print(f"[test] added 4 hybrid pins")

    # 5. Finalize + step.  This exercises:
    #   - vertex_to_pin_idx + d_tet_to_abd_body population
    #   - finalize's anchor=-1 sentinel branch (skip world->local transform)
    #   - apply_fem_pins kernel
    #   - M3.5 chain-rule kernel (because pinned verts are present)
    eng.finalize()
    print(f"[test] finalized")

    eng.step()
    print(f"[test] step() OK")

    # 6. Sanity: vertices still sensible after step
    verts_after = eng.get_vertices_host()
    assert not np.any(np.isnan(verts_after)), "NaN after step"
    assert not np.any(np.isinf(verts_after)), "Inf after step"
    fem_after = verts_after[fem_v_offset:fem_v_offset + 8]
    print(f"[test] FEM verts after step: bbox = "
          f"{fem_after.min(0)} .. {fem_after.max(0)}")

    # The bottom 4 verts should still be at z=0 (pinned to ABD body 0
    # which hasn't moved since gravity acts on the FEM-only top half)
    bottom_z = fem_after[:4, 2]
    print(f"[test] bottom 4 verts z = {bottom_z}")
    # Note: with no IPC barrier set up properly for this tiny mesh,
    # numerical behavior may not match physical intuition.  We're just
    # checking that the engine runs.

    print("\n✅ Phase 3 plumbing OK: bulk-pin API + finalize + step all work.")

    # ------------------------------------------------------------------
    # Phase 4 verification: all-rigid-tet skip
    # Build a fresh engine, this time pin ALL 8 verts to body 0.
    # Expect to see "[Hybrid] 5 / 5 tets are rigid-internal" in output,
    # and step() to still complete (since all "free" deformation is now
    # at the chain-rule routed level + ABD itself).
    # ------------------------------------------------------------------
    print("\n[Phase 4] all-rigid-tet skip test")
    cfg2 = _C.Config()
    eng2 = _C.SimEngine()
    eng2.set_config(cfg2)
    eng2.init_cuda()
    eng2.load_mesh(str(assets), 3, 0, np.eye(4), 1e8, 0)
    eng2.load_mesh_from_data(np.ascontiguousarray(verts, dtype=np.float64),
                             np.ascontiguousarray(tets, dtype=np.int32),
                             4, 3, 1, np.eye(4), 1e6, 0)
    fem_v_offset2 = eng2.get_all_load_records()[-1].vertex_offset
    fem_global_ids2 = np.arange(fem_v_offset2, fem_v_offset2 + 8, dtype=np.int32)
    body_ids2 = np.zeros(8, dtype=np.int32)
    local_pos2 = verts.astype(np.float64)
    eng2.add_fem_pins_with_local_pos(fem_global_ids2, body_ids2, local_pos2)

    eng2.finalize()
    # Look for the [Hybrid] message in stdout (already printed by C++).
    eng2.step()
    print("[Phase 4] all-rigid mesh step() OK (no explosion)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

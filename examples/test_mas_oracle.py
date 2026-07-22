#!/usr/bin/env python3
"""[MAS oracle] CPU P^T H P cross-check of the coarse-matrix aggregation.

The engine dumps (STIFF_MAS_DUMP) the first preconditioning call's packed
fine+coarse matrices, goingNext chains, partId map and level sizes. The CPU
rebuilds EVERY coarse level independently: expand all directed fine pairs of
each bank (upper-tri packed storage, lower triangle = transpose), walk both
sides' goingNext chains to level L, keep same-bank pairs, exact-sum with
math.fsum into the canonical packed slot, compare element-wise with the GPU.
This validates transpose handling, diagonal indexing, partial-bank padding
and the per-level chain on every goingNext transition.

Scope note: the oracle re-derives the INTRA-bank aggregation (kernel-2 path),
so scenes are built from mutually disconnected single-bank bodies — no
cross-bank triplets (kernel-1 path) exist to contaminate the coarse blocks.
The kernel-1 (cross-bank triplet) path is NOT independently covered here:
strict and non-strict share its goingNext / same-bank / transpose structure,
so their mutual agreement cannot rule out a common indexing error. A
dedicated two-bank cross-Hessian CPU fixture is future work.

Scenes (all preconditioner_type=1):
  cube k=8   partial bank, 2 levels      - strict
  k7   k=7   non-power-of-two partial    - strict   [generated .msh]
  20x cube   disconnected, 4 levels      - strict (exact) + merged (tree)

Run:  python3 examples/test_mas_oracle.py
"""
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BANK = 16
NB = BANK * (BANK + 1) // 2


def gen_k7_msh(path):
    """7-vertex, 4-tet connected mesh (non-power-of-two partial bank)."""
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                  [1, 1, 1], [-0.6, 1.1, 1.0], [1.2, 1.4, 2.0]], dtype=float) * 0.1
    T = [(1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 6), (4, 5, 6, 7)]  # 1-based
    for t in T:
        a, b, c, d = (V[i - 1] for i in t)
        vol = np.dot(np.cross(b - a, c - a), d - a) / 6.0
        assert abs(vol) > 1e-9, f"degenerate tet {t}"
    with open(path, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n7\n")
        for i, v in enumerate(V, 1):
            f.write(f"{i} {v[0]} {v[1]} {v[2]}\n")
        f.write("$EndNodes\n$Elements\n4\n")
        for i, t in enumerate(T, 1):
            f.write(f"{i} 4 2 0 1 {t[0]} {t[1]} {t[2]} {t[3]}\n")
        f.write("$EndElements\n")


RUN_CODE = """
import sys, os
import numpy as np
sys.path.insert(0, {root!r})
from stiff_physics.engine import Engine, Config
cfg = Config(dt=0.01, density=1e3, young_modulus=1e6, ground_offset=-10.0,
             assets_dir={root!r} + "/Assets/",
             multienv_mode={mode!r}, preconditioner_type=1)
eng = Engine(cfg)
for k in range({nbodies}):
    tf = np.eye(4); tf[0, 3] = 2.0 * k      # far apart: no contact pairs
    eng.load_mesh({mesh!r}, 3, "FEM", tf)
eng.finalize()
eng.step()
"""


def dump_scene(mesh, mode, nbodies=1):
    d = tempfile.mkdtemp(prefix="mas_oracle_")
    env = dict(os.environ, STIFF_MAS_DUMP=d)
    r = subprocess.run([sys.executable, "-c",
                        RUN_CODE.format(root=ROOT, mesh=mesh, mode=mode, nbodies=nbodies)],
                       env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, f"scene subprocess failed rc={r.returncode}: {r.stderr[-500:]}"
    if not any("[mas-dump]" in l for l in r.stdout.splitlines()):
        raise AssertionError(f"no dump: {r.stdout[-500:]}\n{r.stderr[-500:]}")
    return d


def directed_block(bank_flat, r, c):
    """directed (r,c) 3x3 from upper-tri packed storage (Eigen col-major)."""
    if r <= c:
        i = BANK * r - r * (r + 1) // 2 + c
        return bank_flat[i * 9:(i + 1) * 9].reshape(3, 3, order="F")
    i = BANK * c - c * (c + 1) // 2 + r
    return bank_flat[i * 9:(i + 1) * 9].reshape(3, 3, order="F").T


def oracle(dumpdir, tag, tol, expect_levels=None, expect_blocks=None):
    imat = np.fromfile(f"{dumpdir}/mas_imat.bin", dtype=np.float64).reshape(-1, NB * 9)
    gnext = np.fromfile(f"{dumpdir}/mas_goingNext.bin", dtype=np.uint32).astype(np.int64)
    pmap = np.fromfile(f"{dumpdir}/mas_map.bin", dtype=np.int32).astype(np.int64)
    lsz = np.fromfile(f"{dumpdir}/mas_levelSize.bin", dtype=np.int32).reshape(-1, 2)
    map_nodes = pmap.shape[0]
    n_banks_fine = map_nodes // BANK
    levels = lsz.shape[0] - 1
    valid = pmap >= 0

    # node id of each map slot at level 1..levels-1
    chains = []
    node = np.full(map_nodes, -1, dtype=np.int64)
    node[valid] = gnext[pmap[valid]]                 # real-vertex -> L1 cluster
    chains.append(node.copy())
    for _ in range(2, levels):
        nxt = np.full(map_nodes, -1, dtype=np.int64)
        ok = chains[-1] >= 0
        nxt[ok] = gnext[chains[-1][ok]]              # cluster -> next level
        chains.append(nxt.copy())

    checked = 0
    worst = 0.0
    for node_of_slot in chains:
        cpu = {}
        for b in range(n_banks_fine):
            base = b * BANK
            slots = [s for s in range(BANK) if valid[base + s]]
            for sr in slots:
                nr = node_of_slot[base + sr]
                if nr < 0:
                    continue
                for sc in slots:
                    nc = node_of_slot[base + sc]
                    if nc < 0 or nr // BANK != nc // BANK:
                        continue
                    B = directed_block(imat[b], sr, sc)
                    cb = int(nr // BANK)
                    r0, c0 = int(nr % BANK), int(nc % BANK)
                    # canonical packed slot: (min,max); lower triangle stored
                    # as the transpose of its mirror directed block
                    M = B if c0 >= r0 else B.T
                    lo, hi = (r0, c0) if c0 >= r0 else (c0, r0)
                    key = (cb, BANK * lo - lo * (lo + 1) // 2 + hi)
                    e = cpu.setdefault(key, [[] for _ in range(9)])
                    for i in range(3):
                        for j in range(3):
                            e[i * 3 + j].append(M[i, j])
        for (cb, pi), lists in cpu.items():
            gpu = imat[cb][pi * 9:(pi + 1) * 9].reshape(3, 3, order="F")
            ref = np.array([math.fsum(x) for x in lists]).reshape(3, 3)
            scale = max(np.abs(ref).max(), np.abs(gpu).max(), 1e-30)
            rel = np.abs(ref - gpu).max() / scale
            worst = max(worst, rel)
            checked += 1
    ok = worst < tol and checked > 0
    if expect_levels is not None and levels != expect_levels:
        print(f"[{tag}] COVERAGE FAIL: levels={levels} != expected {expect_levels}")
        ok = False
    if expect_blocks is not None and checked < expect_blocks:
        print(f"[{tag}] COVERAGE FAIL: coarse_blocks={checked} < expected {expect_blocks}")
        ok = False
    print(f"[{tag}] levels={levels} fine_banks={n_banks_fine} coarse_blocks={checked} "
          f"worst rel={worst:.3e} (tol {tol:.0e}) {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ok = True
    k7dir = tempfile.mkdtemp(prefix="mas_k7_")
    k7 = os.path.join(k7dir, "k7.msh")
    gen_k7_msh(k7)
    ok &= oracle(dump_scene("tetMesh/cube.msh", "strict"), "cube k=8 strict", 1e-11,
                 expect_levels=2, expect_blocks=1)
    ok &= oracle(dump_scene(k7, "strict"), "k=7 strict", 1e-11,
                 expect_levels=2, expect_blocks=1)
    ok &= oracle(dump_scene("tetMesh/cube.msh", "strict", nbodies=20),
                 "20x cube 4-level strict", 1e-11, expect_levels=4, expect_blocks=60)
    ok &= oracle(dump_scene("tetMesh/cube.msh", "merged", nbodies=20),
                 "20x cube 4-level tree", 1e-8, expect_levels=4, expect_blocks=60)
    print("MAS ORACLE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

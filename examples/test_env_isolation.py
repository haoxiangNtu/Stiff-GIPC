#!/usr/bin/env python3
"""Per-vertex env isolation necessity test (multi-env subscene contact filter).

Two FEM cubes are spawned OVERLAPPING in space (cube2 dropped into cube1). With no
env tags the broad-phase sees them as different bodies and IPC keeps them apart, so
cube2 rests ON cube1 (stacked, dy ~ one cube height). With per-vertex env tags
(cube1 -> env 0, cube2 -> env 1) the broad-phase skips every cross-env pair, so
cube2 falls THROUGH cube1 and both settle on the analytical ground (dy ~ 0) — cross
-env isolation WITHOUT any spatial separation.

Run one mode:
    python examples/test_env_isolation.py --mode collide
    python examples/test_env_isolation.py --mode isolate
Prints a JSON line with the discriminator dy = center_y(cube2) - center_y(cube1).
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pathlib import Path
from stiff_physics.engine import Engine, Config

ASSETS_DIR = str(Path(__file__).resolve().parent.parent / "Assets") + "/"


def run(mode: str) -> dict:
    cfg = Config(dt=0.020, ground_offset=0.0, assets_dir=ASSETS_DIR, preconditioner_type=0)
    eng = Engine(cfg)

    # cube1 (FEM) near the ground; cube2 (FEM) held well ABOVE it so nothing starts
    # in penetration (IPC cannot recover from initial overlap), then dropped on top.
    T1 = np.eye(4); T1[:3, :3] *= 0.1; T1[1, 3] = 0.10
    eng.load_mesh("sim_data/tetmesh/cube.msh", dimensions=3, body_type="FEM",
                  transform=T1, young_modulus=1e6)
    T2 = np.eye(4); T2[:3, :3] *= 0.1; T2[1, 3] = 0.35
    eng.load_mesh("sim_data/tetmesh/cube.msh", dimensions=3, body_type="FEM",
                  transform=T2, young_modulus=1e6)

    eng.finalize()

    s0, c0 = eng.get_fem_body_vertex_range(0)
    s1, c1 = eng.get_fem_body_vertex_range(1)

    if mode == "isolate":
        nverts = len(eng.get_vertices())
        env = np.zeros(nverts, dtype=np.int32)  # default env 0
        env[s1:s1 + c1] = 1                      # cube2 -> env 1
        eng.set_vertex_env_ids(env)

    for _ in range(160):
        eng.step()

    V = eng.get_vertices()
    cy1 = float(V[s0:s0 + c0, 1].mean())
    cy2 = float(V[s1:s1 + c1, 1].mean())
    # penetration probe: does cube2's lowest vertex sit below cube1's highest?
    overlap = float(V[s0:s0 + c0, 1].max() - V[s1:s1 + c1, 1].min())
    return {"mode": mode, "cy1": cy1, "cy2": cy2, "dy": cy2 - cy1, "yoverlap": overlap}


def _self_check() -> int:
    """Run both modes in fresh subprocesses (the env pointer is a process-global
    device symbol, so modes must not share a process) and assert the contrast."""
    import subprocess, re
    out = {}
    for m in ("collide", "isolate"):
        p = subprocess.run([sys.executable, __file__, "--mode", m],
                           capture_output=True, text=True)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        if line is None:
            print(f"[FAIL] {m}: no RESULT\n{p.stdout}\n{p.stderr}")
            return 1
        out[m] = json.loads(line[len("RESULT "):])
        print(f"  {m}: {out[m]}")
    c, i = out["collide"], out["isolate"]
    ok = (c["dy"] > 0.06 and c["yoverlap"] < 0.01          # collide: stacked, no penetration
          and abs(i["dy"]) < 0.02 and i["yoverlap"] > 0.06)  # isolate: interpenetrated
    print(("[PASS]" if ok else "[FAIL]") +
          " per-vertex env isolation: collide stacks (dy=%.3f, pen=%.4f); "
          "isolate passes through (dy=%.3f, overlap=%.3f)"
          % (c["dy"], c["yoverlap"], i["dy"], i["yoverlap"]))
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["collide", "isolate"])
    args = ap.parse_args()
    if args.mode:
        print("RESULT " + json.dumps(run(args.mode)), flush=True)
    else:
        sys.exit(_self_check())

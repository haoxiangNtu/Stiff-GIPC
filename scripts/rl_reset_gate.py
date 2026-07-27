#!/usr/bin/env python3
"""RL episode-reset gate: teleport semantics, health telemetry, quarantine revival.

Cases (one subprocess each):
  teleport_continuity — merged: ABD teleport must be visible in get_vertices()
      IMMEDIATELY (regression for the q->x desync), FEM teleport + continued
      stepping stays finite, health counters stay zero.
  bad_reset_overlap   — merged: teleporting the FEM cube INTO the ABD cube is
      not refused (no ground invariant), but the next step's degradation must
      surface through the step-health counters (the RL discard signal).
  bad_reset_ground    — merged: teleporting below ground must raise
      GeometryError AT the teleport (typed refusal via the rebuild's
      buildCP invariant).
  quarantine_revival  — strict 2-env: a bad reset quarantines env0 (env1
      unaffected), and a subsequent GOOD teleport of env0 revives it
      (status clears, env0 steps healthily again).
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ASSETS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Assets"
) + "/"
CASES = (
    "teleport_continuity",
    "bad_reset_overlap",
    "bad_reset_ground",
    "quarantine_revival",
)


def transform(x: float, y: float):
    import numpy as np

    value = np.eye(4)
    value[0, 3] = x
    value[1, 3] = y
    return value


def build_pair(mode: str):
    """ABD cube + FEM cube, apart, above ground."""
    from stiff_physics import Config, Engine

    engine = Engine(
        Config(
            dt=0.01,
            density=1e3,
            young_modulus=1e6,
            gravity=(0.0, -9.8, 0.0),
            ground_offset=-0.5,
            friction_rate=0.4,
            multienv_mode=mode,
            assets_dir=ASSETS,
        )
    )
    engine.set_log_level(0)
    engine.load_mesh("tetMesh/cube.msh", 3, "ABD", transform(-0.7, 0.0))
    engine.load_mesh("tetMesh/cube.msh", 3, "FEM", transform(0.7, 0.0))
    engine.set_body_groups([0, 0])
    engine.finalize()
    return engine


def build_two_env():
    from stiff_physics import Config, Engine

    engine = Engine(
        Config(
            dt=0.01,
            density=1e3,
            young_modulus=1e6,
            gravity=(0.0, -9.8, 0.0),
            ground_offset=-2.0,
            multienv_mode="strict",
            per_env_exit=True,
            env_newton_iter_cap=30,
            assets_dir=ASSETS,
        )
    )
    engine.set_log_level(0)
    engine.load_mesh("tetMesh/cube.msh", 3, "FEM", transform(-0.7, 0.3))
    engine.load_mesh("tetMesh/cube.msh", 3, "FEM", transform(0.7, 0.3))
    engine.set_body_groups([0, 1])
    engine.finalize()
    return engine


def run_case(case: str) -> int:
    import numpy as np
    from stiff_physics import GeometryError

    if case == "teleport_continuity":
        e = build_pair("merged")
        abd = e.get_load_records()[0]
        for _ in range(5):
            e.step()
        # ABD teleport: new pose must be visible IMMEDIATELY (q->x sync).
        e.teleport_abd_bodies(np.array([0], dtype=np.int32),
                              transform(0.5, 1.0)[None, :, :])
        verts = e.get_vertices()
        abd_c = verts[abd.vertex_offset:abd.vertex_offset + abd.vertex_count]
        centroid = abd_c.mean(axis=0)
        # ABD local frames are COM-centered (x = J*q), so the world centroid
        # equals the transform translation exactly.
        if abs(centroid[0] - 0.5) > 0.05 or abs(centroid[1] - 1.0) > 0.05:
            print(f"    FAIL: ABD teleport not visible pre-step "
                  f"(centroid={centroid})")
            return 1
        # FEM teleport to a clean spot well clear of the ABD cube (which now
        # sits around x=0.5): shift x by +1.0, keep height.
        fem = e.get_load_records()[1]
        cur = e.get_vertices()[fem.vertex_offset:
                               fem.vertex_offset + fem.vertex_count].copy()
        cur[:, 0] += 1.0
        e.native.teleport_fem_vertices(cur, None)
        for _ in range(5):
            e.step()
        p = e.get_vertices()
        if not np.isfinite(p).all():
            print("    FAIL: non-finite positions after healthy resets")
            return 1
        if e.native.get_ls_exhausted_count() != 0:
            print("    FAIL: healthy resets tripped the health counters")
            return 1
        print("    teleport continuity PASS (abd visible pre-step, "
              "5+5 steps finite, health clean)")
        return 0

    if case == "bad_reset_overlap":
        e = build_pair("merged")
        for _ in range(2):
            e.step()
        abd = e.get_load_records()[0]
        fem = e.get_load_records()[1]
        # park the FEM cube inside the ABD cube (half-overlap in x)
        overlap = e.get_vertices()[abd.vertex_offset:
                                   abd.vertex_offset + abd.vertex_count].copy()
        overlap[:, 0] += 0.2
        e.native.teleport_fem_vertices(overlap, None)
        before = e.native.get_ls_exhausted_count()
        e.step()
        tripped = e.native.get_ls_exhausted_count() - before
        nonfin = e.native.get_ls_nonfinite_count()
        if tripped <= 0:
            print("    FAIL: overlapping reset did not trip the "
                  "step-health counter (silent poisoned episode)")
            return 1
        print(f"    bad overlap reset PASS (exhausted+={tripped} "
              f"nonfinite={nonfin} -> RL loop can discard)")
        return 0

    if case == "bad_reset_ground":
        e = build_pair("merged")
        for _ in range(2):
            e.step()
        fem = e.get_load_records()[1]
        below = e.get_vertices()[fem.vertex_offset:
                                 fem.vertex_offset + fem.vertex_count].copy()
        below[:, 1] -= 5.0
        try:
            e.native.teleport_fem_vertices(below, None)
        except GeometryError as exc:
            print(f"    bad ground reset PASS (typed refusal at teleport: "
                  f"{str(exc)[:80]}...)")
            return 0
        print("    FAIL: below-ground reset was accepted at teleport")
        return 1

    # quarantine_revival
    e = build_two_env()
    env0 = e.get_load_records()[0]
    good = None
    e.step()
    good = e.get_vertices()[env0.vertex_offset:
                            env0.vertex_offset + env0.vertex_count].copy()
    bad = good.copy()
    bad[:, 1] -= 3.0
    # strict/per-env buildCP returns before the merged-path ground validation,
    # so the quarantine fires at the NEXT step's frame-start scan (iron-law),
    # not at the teleport itself — assert after one step, like the checkpoint
    # quarantine gate does.
    e.native.teleport_fem_vertices(bad, None)
    e.step()
    status = list(e.native.get_per_env_status())
    if not status or status[0] != 3:
        print(f"    FAIL: bad reset did not quarantine env0 (status={status})")
        return 1
    import numpy as np
    v1 = e.get_vertices()
    env1 = e.get_load_records()[1]
    if not np.isfinite(
        v1[env1.vertex_offset:env1.vertex_offset + env1.vertex_count]
    ).all():
        print("    FAIL: healthy env1 corrupted by env0 quarantine")
        return 1
    # the revival: teleport env0 back to a good pose
    e.native.teleport_fem_vertices(good, None)
    status = list(e.native.get_per_env_status())
    if status and status[0] == 3:
        print(f"    FAIL: good reset did not clear quarantine (status={status})")
        return 1
    y_before = float(
        e.get_vertices()[env0.vertex_offset:env0.vertex_offset
                         + env0.vertex_count][:, 1].mean()
    )
    for _ in range(3):
        e.step()
    seg = e.get_vertices()[env0.vertex_offset:
                           env0.vertex_offset + env0.vertex_count]
    y_after = float(seg[:, 1].mean())
    status = list(e.native.get_per_env_status())
    if not np.isfinite(seg).all() or (status and status[0] == 3):
        print(f"    FAIL: revived env0 not healthy (status={status})")
        return 1
    if not (y_after < y_before - 1e-4):
        print(f"    FAIL: revived env0 frozen (y {y_before:.4f} -> "
              f"{y_after:.4f})")
        return 1
    print(f"    quarantine revival PASS (quarantine -> revive -> falls "
          f"{y_before:.3f}->{y_after:.3f}, env1 clean throughout)")
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
        print(f"[rl-reset] case={case}", flush=True)
        result = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--case", case],
            env=os.environ.copy(),
        )
        ok &= result.returncode == 0
    print("RL-RESET-GATE:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

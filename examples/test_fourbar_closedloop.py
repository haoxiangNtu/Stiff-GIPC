#!/usr/bin/env python3
"""[P3] closed-loop four-bar linkage: kinematic-loop joint constraint test.

A parallelogram four-bar (fixed ground link + two 0.25 m side bars + 0.4 m
coupler) is assembled at 30 deg off vertical with FOUR passive revolute
joints — the fourth closes the loop, which open-chain trees never exercise.
Under gravity the mechanism must swing like a 1-DOF pendulum.

Loop-specific assertions:
  * the coupler stays TRANSLATING (both ends same height within 2 cm): if the
    loop-closing joint failed, the chain degenerates to a double pendulum and
    the coupler rotates;
  * both side bars move in sync (parallelogram);
  * each bar stays rigid (diagonal length drift < 1%);
  * Newton iterations stay far below the cap (no active-set chattering).

Run:  python3 examples/test_fourbar_closedloop.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stiff_physics.engine import Engine, Config

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Assets") + "/"
FRAMES = 80
ITER_PEAK_LIMIT = 300  # cap is 1000; healthy loop closure stays way below

# cube.msh spans [-0.2,0.2]x[0.1,0.5]x[-0.2,0.2]: center (0,0.3,0), edge 0.4
CUBE_CENTER = np.array([0.0, 0.3, 0.0])


def bar_transform(center, angle_deg, length, thick=0.05):
    """4x4 placing a cube as a bar: long axis = x rotated by angle (about z)."""
    S = np.diag([length / 0.4, thick / 0.4, thick / 0.4, 1.0])
    C = np.eye(4); C[:3, 3] = -CUBE_CENTER
    a = np.deg2rad(angle_deg)
    R = np.eye(4)
    R[0, 0], R[0, 1] = np.cos(a), -np.sin(a)
    R[1, 0], R[1, 1] = np.sin(a), np.cos(a)
    T = np.eye(4); T[:3, 3] = center
    return T @ R @ S @ C


def main():
    cfg = Config(dt=0.01, density=1e3, ground_offset=0.0, assets_dir=ASSETS)
    eng = Engine(cfg)

    # Parallelogram: side bars 0.25 m at 30 deg off vertical (u = (0.5, 0.866))
    u = np.array([np.sin(np.deg2rad(30)), np.cos(np.deg2rad(30))])
    P1 = np.array([-0.2, 0.5]); P4 = np.array([0.2, 0.5])   # ground-link hinges
    P2 = P1 + 0.25 * u;         P3 = P4 + 0.25 * u          # coupler hinges

    def c3(p2):  # lift 2D layout point to 3D center
        return np.array([p2[0], p2[1], 0.0])

    # body 0: ground link (fixed), bodies 1..3 free
    eng.load_mesh("tetMesh/cube.msh", 3, "ABD", bar_transform(c3((P1 + P4) / 2), 0, 0.4),
                  boundary_type="Fixed")
    eng.load_mesh("tetMesh/cube.msh", 3, "ABD", bar_transform(c3((P1 + P2) / 2), 60, 0.25))
    eng.load_mesh("tetMesh/cube.msh", 3, "ABD", bar_transform(c3((P2 + P3) / 2), 0, 0.4))
    eng.load_mesh("tetMesh/cube.msh", 3, "ABD", bar_transform(c3((P3 + P4) / 2), 60, 0.25))

    for a in range(4):
        for b in range(a + 1, 4):
            eng.add_collision_exclusion(a, b)

    z = (0, 0, 1)
    eng.add_revolute_joint(0, 1, z, (*P1, 0.0), -3.0, 3.0, passive=True, name="j1")
    eng.add_revolute_joint(1, 2, z, (*P2, 0.0), -3.0, 3.0, passive=True, name="j2")
    eng.add_revolute_joint(2, 3, z, (*P3, 0.0), -3.0, 3.0, passive=True, name="j3")
    eng.add_revolute_joint(3, 0, z, (*P4, 0.0), -3.0, 3.0, passive=True, name="j4-loop")

    eng.finalize()

    V0 = np.asarray(eng.get_vertices()).copy()
    # per-bar vertex slices: 8 verts each, load order
    bars = [slice(8 * i, 8 * (i + 1)) for i in range(4)]
    diag0 = [np.linalg.norm(V0[s][0] - V0[s][6]) for s in bars]

    it_prev = eng.native.get_total_newton_iters()
    it_peak = 0
    coupler_tilt_max = 0.0
    for fr in range(FRAMES):
        eng.step()
        tot = eng.native.get_total_newton_iters()
        it_peak = max(it_peak, tot - it_prev)
        it_prev = tot
        V = np.asarray(eng.get_vertices())
        assert np.isfinite(V).all(), f"non-finite vertices at frame {fr}"
        # coupler (bar 2) end heights: mean y of left-end vs right-end verts
        c = V[bars[2]]
        left = c[c[:, 0].argsort()[:4]][:, 1].mean()
        right = c[c[:, 0].argsort()[-4:]][:, 1].mean()
        coupler_tilt_max = max(coupler_tilt_max, abs(left - right))

    V1 = np.asarray(eng.get_vertices())
    assert np.abs(V1).max() < 3.0, "mechanism flew apart"

    disp = [np.linalg.norm((V1[s] - V0[s]).mean(axis=0)) for s in bars]
    diag1 = [np.linalg.norm(V1[s][0] - V1[s][6]) for s in bars]
    rigid_err = max(abs(d1 - d0) / d0 for d0, d1 in zip(diag0, diag1))

    print(f"iter peak/frame       = {it_peak}")
    print(f"bar displacements     = {[f'{d:.4f}' for d in disp]} m (ground, crank, coupler, rocker)")
    print(f"coupler max end tilt  = {coupler_tilt_max * 100:.2f} cm")
    print(f"max bar rigidity err  = {rigid_err * 100:.3f} %")

    ok = True
    if it_peak >= ITER_PEAK_LIMIT:
        print(f"FAIL: Newton peak {it_peak} >= {ITER_PEAK_LIMIT} (loop constraint ill-conditioned?)")
        ok = False
    if disp[0] > 1e-6:
        print("FAIL: fixed ground link moved")
        ok = False
    if disp[1] < 0.03:
        print("FAIL: mechanism did not swing (crank displacement < 3 cm)")
        ok = False
    if abs(disp[1] - disp[3]) > 0.3 * max(disp[1], disp[3]) + 0.005:
        print("FAIL: side bars out of sync — loop not transmitting motion")
        ok = False
    if coupler_tilt_max > 0.02:
        print("FAIL: coupler rotated (> 2 cm end-height split) — loop-closing joint j4 broken, "
              "chain degenerated to a double pendulum")
        ok = False
    if rigid_err > 0.01:
        print("FAIL: bar rigidity violated")
        ok = False

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

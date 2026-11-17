#!/usr/bin/env python3
"""Physics smoke test for the lab scene: build, check pad/prop placement,
step a few frames, print timing / pad forces / dropper settling.

    PYTHONPATH=. <solver-python> examples/lab_tactile/smoke_test.py [--frames 40]
"""
from __future__ import annotations

import argparse
import os.path as osp
import sys
import time

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from lab_scene import LabScene  # noqa: E402
from lab_task import top_down_R, T_of  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--precond", type=int, default=0)
    ap.add_argument("--no-tactile-gpu", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    sc = LabScene(dt=0.005, precond=args.precond, tactile_gpu=not args.no_tactile_gpu, log_level=0)
    print(f"[smoke] scene built in {time.time()-t0:.1f}s")
    # placement check: every prop's rest surface vs the planned transform
    for n, b in sc.bodies.items():
        if b.kind == "arm":
            continue
        p = sc.props[n]
        T = sc.prop_T[n]
        want = p.verts @ T[:3, :3].T + T[:3, 3]
        have = b.rest_verts
        err = np.abs(want.mean(0) - have.mean(0)).max() * 1e3 if len(want) == len(have) else float("nan")
        print(f"[smoke] prop {n:12s} centroid err {err:6.2f} mm  y-range {have[:,1].min()*1e3:6.1f}..{have[:,1].max()*1e3:6.1f} mm")
    dv = sc.dropper_verts()
    print(f"[smoke] dropper y-range {dv[:,1].min()*1e3:.1f}..{dv[:,1].max()*1e3:.1f} mm, bulb centre {np.round(sc.bulb_centre()*1e3,1)} mm")
    print(f"[smoke] TCP at load {np.round(sc.tcp_pose()[:3,3],4)} vs FK {np.round(sc.arm.fk(sc.q)[:3,3],4)}")
    # hold the load pose, open the fingers, step
    sc.gripper.command(mode="pos", gap_mm=40, vmax_mm_s=60)
    q0 = sc.q.copy()
    times = []
    for k in range(args.frames):
        sc.drive_arm(q0)
        gi = sc.gripper.update()
        t1 = time.time()
        sc.eng.step()
        times.append(time.time() - t1)
        if k % 10 == 0 or k == args.frames - 1:
            bc = sc.bulb_centre()
            try:
                print(f"[smoke]   collision pairs total {sc.eng.native.get_total_collision_pairs()} max {sc.eng.native.get_max_collision_pairs()} newton {sc.eng.native.get_total_newton_iters()}")
            except Exception as exc:
                print("[smoke]   (pair stats n/a)", exc)
            print(f"[smoke] fr {k:3d} step {times[-1]*1e3:6.1f} ms  gap {gi['gap_mm']:5.1f} mm  Fn {gi['fn_left']:.3f}/{gi['fn_right']:.3f} N"
                  f"  bulb y {bc[1]*1e3:.2f} mm  tcp err {np.linalg.norm(sc.tcp_pose()[:3,3]-sc.arm.fk(q0)[:3,3])*1e3:.2f} mm")
    print(f"[smoke] mean step {np.mean(times[5:])*1e3:.1f} ms  (newton total {sc.eng.native.get_total_newton_iters()})")
    for side in ("left", "right"):
        if sc.gripper.pads[side].get("gpu") is not None:
            im = sc.gripper.indentation(side)
            print(f"[smoke] pad {side} indentation max {im.max()*1e3:.3f} mm, shape {im.shape}")
    print("[smoke] OK")


if __name__ == "__main__":
    main()

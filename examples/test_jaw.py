#!/usr/bin/env python3
"""Isolate Problem 2: in force mode the negative-range jaw (d1) won't open.
Hypothesis: the closed-end barrier (slot0) is armed AT cl=0 while d1 rests at
exactly 0 -> singular -> pins d1.  Toggle BAR=0/1 to arm/skip the barrier.

  BAR=1 (default): arm both-end barrier (reproduce stuck d1)
  BAR=0          : do NOT arm barrier -> does d1 open now?
"""
import os, numpy as np
import umi_finray_lib as L

BAR = int(os.environ.get("BAR", "1"))
prep = L.prepare_scene("foldshirt"); P = L._drive_params()
eng = L.make_engine(prep, 1); sides = L.load_finray_sides()
envs, _ = L.build_world(eng, prep, 1, 4.0, sides, dict(abd_cursor=0))
seg = L._stitch_seg_arrays(envs); eng.finalize()
robot = L._setup_after_finalize(eng, envs, P)
if BAR:
    L.arm_force_barrier(eng, robot, P)
ejs = L.slice_env_joints(robot, 1); groups = L.build_drive_groups(envs, ejs)
Lg = [g for g in groups if g['key'] == (0, 'L')][0]
pis = Lg['pris']
print(f"L jaws pis={pis}  BAR={BAR}")
for pi in pis:
    op, cl = L._open_close(robot, pi)
    print(f"  pi={pi} limits op={op:+.4f} cl={cl:+.4f}  start_d={eng.native.get_prismatic_current_distance(pi):+.5f}")

def cmd_open():
    for pi in pis:
        op, cl = L._open_close(robot, pi)
        eng.native.set_prismatic_force(pi, 0.0)
        eng.native.set_prismatic_strength(pi, P['pos_k'])
        eng.native.set_prismatic_target(pi, op)

def cmd_close():
    for pi in pis:
        op, cl = L._open_close(robot, pi)
        cd = 1.0 if (cl - op) > 0 else -1.0
        eng.native.set_prismatic_strength(pi, 0.0)
        eng.native.set_prismatic_force(pi, cd * P['barrier_force'])

print("\n-- phase 1: command CLOSE (pure force) and step --")
for fr in range(40):
    cmd_close(); eng.step()
    if fr % 10 == 9:
        ds = [eng.native.get_prismatic_current_distance(pi) for pi in pis]
        print(f"  fr{fr+1:3d}  d0={ds[0]:+.5f}  d1={ds[1]:+.5f}")

print("\n-- phase 2: command OPEN (force path) and step --")
for fr in range(60):
    cmd_open(); eng.step()
    if fr % 10 == 9:
        ds = [eng.native.get_prismatic_current_distance(pi) for pi in pis]
        print(f"  fr{fr+1:3d}  d0={ds[0]:+.5f}  d1={ds[1]:+.5f}")
ds = [eng.native.get_prismatic_current_distance(pi) for pi in pis]
print(f"\n  verdict: d0 {'OPENED' if abs(ds[0])>0.02 else 'STUCK'}, "
      f"d1 {'OPENED' if abs(ds[1])>0.02 else 'STUCK'}")

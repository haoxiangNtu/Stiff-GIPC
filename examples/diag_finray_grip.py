#!/usr/bin/env python3
"""Headless per-env grip diagnostic for the UMI finray suite.

Runs one scene/mode at N envs through its grasp window and prints, per env:
the gripper opening (closed?), whether the object was lifted (rigid) or the
gripper closed (cloth), and any over-limit (fly-out) of the prismatic joint.

  SC=beaker|cupshirt|foldshirt   GRIP_MODE=pos|stitch|force   CASE39ME_NUM_ENVS=N

Plus any of the tuning knobs (GRIP_STITCH_DEBOUNCE, GRIP_CLOSE_DS,
GRIP_STITCH_MIN_S, GRIP_FORCE_STRENGTH, CASE39_FRICTION, ...).

Run each invocation as its OWN process (avoids the metis sort-cache trap that
bites only when several scenes are built in one process).
"""
import os, numpy as np
import umi_finray_lib as L

scene = os.environ.get("SC", "beaker")
mode = os.environ.get("GRIP_MODE", "pos")
N = int(os.environ.get("CASE39ME_NUM_ENVS", "4"))
END = {"beaker": 94, "cupshirt": 950, "foldshirt": 640}[scene]

prep = L.prepare_scene(scene); P = L._drive_params()
if "GRIP_FORCE_LOCK" not in os.environ:   # lock for rigid grasps; pure-force close for cloth-only
    P['force_lock'] = 1 if prep['abd_obj'] else 0
eng = L.make_engine(prep, N); sides = L.load_finray_sides()
envs, _ = L.build_world(eng, prep, N, 4.0, sides, dict(abd_cursor=0))
seg = L._stitch_seg_arrays(envs); eng.finalize()
robot = L._setup_after_finalize(eng, envs, P)
if mode == "force":
    L.arm_force_barrier(eng, robot, P)
ejs = L.slice_env_joints(robot, N); groups = L.build_drive_groups(envs, ejs); gs = {}
A = prep["actions"]
objr = [(e.get("abd_obj_rec") or e.get("fem_obj_rec")) for e in envs]
Lp = [[g for g in groups if g['key'] == (e, 'L')][0]['pris'] for e in range(N)]
ymin = [9e9]*N; dmin = [9e9]*N; over = [False]*N
v0 = eng.get_vertices()
for fr in range(END):
    L.drive_frame(eng, robot, ejs, groups, prep, A[fr], mode, gs, P, seg)
    eng.step()
    v = eng.get_vertices()
    for e in range(N):
        r = objr[e]
        y = v[r.vertex_offset:r.vertex_offset+r.vertex_count, 1].mean()
        ymin[e] = min(ymin[e], y)
        d0 = eng.native.get_prismatic_current_distance(Lp[e][0])
        d1 = eng.native.get_prismatic_current_distance(Lp[e][1])
        if d0 > 0.0415 or d1 < -0.0415:
            over[e] = True
        if (scene == "foldshirt" and fr >= 480) or (scene != "foldshirt" and fr >= END*0.7):
            dmin[e] = min(dmin[e], abs(d0))
v = eng.get_vertices()
print(f"\n[{scene}/{mode}]  N={N}  debounce={P['stitch_debounce']} ds={P['close_ds']} "
      f"min_s={P['stitch_min_s']} force_strength={P['force_strength']} friction={prep['friction']}")
for e in range(N):
    yf = v[objr[e].vertex_offset:objr[e].vertex_offset+objr[e].vertex_count, 1].mean()
    rose = yf - ymin[e]
    if scene == "foldshirt":
        verd = "CLOSED" if dmin[e] < 0.012 else "NOT-CLOSED"
    else:
        verd = "GRIP+LIFT" if rose > 0.02 else "slip/no-lift"
    flag = "  *** FLY-OUT (over limit) ***" if over[e] else ""
    print(f"   e{e}  dmin={dmin[e]:.3f}  obj_rose={rose:+.3f}  -> {verd}{flag}")

#!/usr/bin/env python3
"""Why does a PURE-FORCE grip slip when the arm lifts/moves?

Records, per frame, for env0's LEFT hand on cupshirt/force:
  d0,d1  = the two finger prismatic openings (the squeeze DOF)
  cup_y  = grasped object centroid height (world)
  grip_y = LEFT-hand finger reference height (moves WITH the arm)
  rel    = cup - gripper  (x,y,z): if this drifts during transport, the object
           is sliding RELATIVE to the gripper (tangential slip)

We then split the run into CLOSE (before lift) and LIFT/TRANSPORT (arm rising)
and report how much the opening (d) drifts and how much the object slips
relative to the gripper in each phase. Pure force has zero positional
stiffness on the squeeze DOF -> the grip is a neutral force balance ->
inertial load during transport walks the opening and the object lags.
"""
import os, numpy as np
import umi_finray_lib as L

scene = os.environ.get("SC", "beaker"); mode = os.environ.get("GRIP_MODE", "force"); N = 1
prep = L.prepare_scene(scene); P = L._drive_params()
END = int(os.environ.get("CASE39_FRAME_END", str(len(prep["actions"]))))  # FULL trajectory by default
eng = L.make_engine(prep, N); sides = L.load_finray_sides()
envs, _ = L.build_world(eng, prep, N, 4.0, sides, dict(abd_cursor=0))
seg = L._stitch_seg_arrays(envs); eng.finalize()
robot = L._setup_after_finalize(eng, envs, P)
if mode == "force":
    L.arm_force_barrier(eng, robot, P)
ejs = L.slice_env_joints(robot, N); groups = L.build_drive_groups(envs, ejs); gs = {}
A = prep["actions"]

env0 = envs[0]
objr = env0.get("abd_obj_rec") or env0.get("fem_obj_rec")
Lpris = [g for g in groups if g['key'] == (0, 'L')][0]['pris']
# left-hand finger FEM verts = gripper-fixed reference (moves with arm)
Lfem = [(g['fem_rec'].vertex_offset, g['fem_rec'].vertex_count)
        for g in env0['grippers'] if g['side'] == 'L']

def cen(v, off, cnt): return v[off:off+cnt].mean(axis=0)

rows = []
for fr in range(END):
    L.drive_frame(eng, robot, ejs, groups, prep, A[fr], mode, gs, P, seg)
    eng.step()
    v = eng.get_vertices()
    cup = cen(v, objr.vertex_offset, objr.vertex_count)
    grip = np.mean([cen(v, o, c) for o, c in Lfem], axis=0)
    d0 = eng.native.get_prismatic_current_distance(Lpris[0])
    d1 = eng.native.get_prismatic_current_distance(Lpris[1])
    rows.append((fr, d0, d1, cup[1], grip[1], *(cup - grip)))

R = np.array(rows)
fr, d0, d1, cupy, gripy, rx, ry, rz = (R[:, i] for i in range(8))

# LIFT/TRANSPORT = from the cup's lowest point (grasped at the bottom) to the end.
# (Before that the cup is just falling to the floor + the arm descending -- not slip.)
lift_start = int(np.argmin(cupy))
close = slice(0, lift_start); lift = slice(lift_start, END)

print(f"\n[{scene}/{mode}]  barrier_force={P['barrier_force']} anchor={P['force_anchor']}  friction={prep['friction']}")
print(f"  lift/transport = frame {lift_start} (cup lowest, just grasped) -> {END}")
print(f"  opening d0 at grasp={d0[lift_start]:.4f}  at end={d0[-1]:.4f}")
def rng(a, s): return f"{a[s].min():+.4f}..{a[s].max():+.4f}  (drift {a[s].max()-a[s].min():.4f})"
for name, ph in [("CLOSE ", close), ("LIFT  ", lift)]:
    print(f"\n {name}frames {ph.start}-{ph.stop}")
    print(f"   opening d0     {rng(d0, ph)}")
    print(f"   gripper_y      {rng(gripy, ph)}")
    print(f"   cup_y          {rng(cupy, ph)}")
    print(f"   cup-grip dy    {rng(ry, ph)}   <- object slipping DOWN relative to gripper")
    print(f"   cup-grip dx/dz {rng(rx, ph)} / {rng(rz, ph)}")
# sampled trajectory (see the final lift)
print("\n  frame   d0      d1     gripY    objY    obj-grip_dy")
step = max(1, END // 24)
for i in range(0, END, step):
    print(f"  {int(fr[i]):5d}  {d0[i]:.4f}  {d1[i]:+.4f}  {gripy[i]:+.3f}  {cupy[i]:+.3f}  {ry[i]:+.4f}")

# headline: relative drop of object vs gripper across the lift
rel_slip = ry[lift][-1] - ry[lift][0] if lift.stop > lift.start else 0.0
open_walk = d0[lift].max() - d0[lift].min() if lift.stop > lift.start else 0.0
print(f"\n  >> during LIFT: object moved {rel_slip*1000:+.1f} mm relative to gripper in y")
print(f"  >> during LIFT: finger opening d0 walked {open_walk*1000:.1f} mm (no position stiffness to pin it)")

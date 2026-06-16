#!/usr/bin/env python3
"""URDF dual-arm mixed control: LEFT arm position-controlled, RIGHT arm
torque-controlled, in ONE simulation. Validates set_revolute_torque on a real
robot + that position(penalty) and torque(generalized force) coexist per-joint.

Gravity off to isolate the control. Left arm joints get a position target; one
right-arm joint gets pure torque (strength=0 + set_revolute_torque).
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stiff_physics import Engine, Config
from stiff_physics.robot import Robot

URDF = "sim_data/urdf/ridgeback_dual_panda_UMI/ridgeback_dual_panda2.urdf"

cfg = Config(gravity=(0.0, 0.0, 0.0), dt=0.01)
eng = Engine(cfg)
eng.load_urdf(URDF, root_fixed=True, revolute_as_motor=True)
eng.finalize()
robot = Robot(eng)

joints = robot.revolute_joints
left  = [j for j in joints if "left_arm_joint"  in j.name]
right = [j for j in joints if "right_arm_joint" in j.name]
print(f"revolute joints: {len(joints)} (left_arm={len(left)}, right_arm={len(right)})")

# LEFT arm: POSITION control -> target 0.3 rad on every left joint (PD penalty).
POS_TGT = 0.3
for j in left:
    eng.native.set_revolute_target(j.index, POS_TGT)

# RIGHT arm: TORQUE control -> pick one mid joint, kill PD, apply pure torque.
torque_joint = right[1] if len(right) > 1 else right[0]
for j in right:
    eng.native.set_revolute_target(j.index, 0.0)     # hold others at 0 (position)
eng.native.set_revolute_strength(torque_joint.index, 0.0)  # pure torque on this one
eng.native.set_revolute_torque(torque_joint.index, 0.05)
print(f"torque joint: {torque_joint.name} (idx {torque_joint.index}), tau=+80")

a0 = np.asarray(eng.get_revolute_current_angles())
for _ in range(80):
    eng.step()
a1 = np.asarray(eng.get_revolute_current_angles())

def avg(js): return float(np.mean([abs(a1[j.index]) for j in js]))
left_err = float(np.mean([abs(a1[j.index] - POS_TGT) for j in left]))
tq_move  = float(a1[torque_joint.index] - a0[torque_joint.index])
right_other = float(np.mean([abs(a1[j.index]-a0[j.index]) for j in right
                             if j.index != torque_joint.index]))

print(f"\nLEFT (position, target {POS_TGT}): mean |angle| = {avg(left):.3f} rad, "
      f"mean err = {left_err:.3f}")
print(f"RIGHT torque joint moved: {tq_move:+.3f} rad")
print(f"RIGHT other joints moved (should be ~0): {right_other:.4f} rad")
print("\n=== Checks ===")
print(f"left arm tracked target (err<0.1): {left_err < 0.1}")
print(f"right torque joint rotated (|move|>0.05): {abs(tq_move) > 0.05}")
print(f"torque sign correct (+tau -> +angle): {tq_move > 0}")
print(f"mixed pos+torque coexist on one robot: "
      f"{left_err < 0.1 and abs(tq_move) > 0.05}")

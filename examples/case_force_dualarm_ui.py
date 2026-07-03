#!/usr/bin/env python3
"""GUI: mixed position/torque control on a URDF dual-arm robot.

LEFT arm  = POSITION control  (drag "left target" slider -> set_revolute_target)
RIGHT arm = TORQUE   control  (drag "right torque" slider -> set_revolute_torque
                               on one joint; the rest of the right arm is held
                               by position control so it doesn't flop).

Click "Run / stop". Watch the left arm track the target while the right joint
spins under torque — two control modes coexisting in one sim.

Run:
    STIFF_SKIP_CCD_SANITY=1 python examples/case_force_dualarm_ui.py
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stiff_physics import Engine, Config
from stiff_physics.robot import Robot
import polyscope as ps
import polyscope.imgui as psim

from pathlib import Path
ASSETS_DIR = str(Path(__file__).resolve().parent.parent / "Assets") + "/"


URDF = "sim_data/urdf/ridgeback_dual_panda_UMI/ridgeback_dual_panda2.urdf"

eng = Engine(Config(gravity=(0.0, 0.0, 0.0), dt=0.01, assets_dir=ASSETS_DIR))
eng.load_urdf(URDF, root_fixed=True, revolute_as_motor=True)
eng.finalize()
robot = Robot(eng)

joints = robot.revolute_joints
left  = [j for j in joints if "left_arm_joint"  in j.name]
right = [j for j in joints if "right_arm_joint" in j.name]
torque_joint = right[1] if len(right) > 1 else right[0]
for j in right:                                   # hold non-torque right joints
    eng.native.set_revolute_target(j.index, 0.0)
eng.native.set_revolute_strength(torque_joint.index, 0.0)   # pure torque on this one
print(f"left_arm joints={len(left)}, right torque joint = {torque_joint.name}")

state = {"run": False, "left_tgt": 0.0, "torque": 0.0}

ps.init()
ps.set_up_dir("y_up")
verts = eng.get_vertices(); faces = eng.get_surface_faces()
mesh = ps.register_surface_mesh("robot", verts, faces, color=(0.6, 0.7, 0.85))


def callback():
    changed_l, state["left_tgt"] = psim.SliderFloat("left target (rad)", state["left_tgt"], -1.5, 1.5)
    changed_t, state["torque"]   = psim.SliderFloat("right torque (N*m)", state["torque"], -0.2, 0.2)
    if changed_l:
        for j in left:
            eng.native.set_revolute_target(j.index, state["left_tgt"])
    if changed_t:
        eng.native.set_revolute_torque(torque_joint.index, state["torque"])
    if psim.Button("Run / stop"):
        state["run"] = not state["run"]
    psim.SameLine()
    psim.TextUnformatted("RUNNING" if state["run"] else "paused")
    if state["run"]:
        eng.step()
        mesh.update_vertex_positions(eng.get_vertices())
        ang = np.asarray(eng.get_revolute_current_angles())
        psim.TextUnformatted(f"left mean angle: {np.mean([ang[j.index] for j in left]):+.3f} rad")
        psim.TextUnformatted(f"right torque-joint angle: {ang[torque_joint.index]:+.3f} rad")


ps.set_user_callback(callback)
ps.show()

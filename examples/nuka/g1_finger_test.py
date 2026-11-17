"""Diagnostic: do the Dex3 finger joints follow set_revolute_target? G1 fixed in the air, left hand joints commanded to a closed pose."""
import os.path as osp, sys, numpy as np
sys.path.insert(0, "examples/robotgym"); sys.path.insert(0, "examples/nuka/tools")
from common import ASSETS, Scene, _load_urdf_capture, Body
R_CONV = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])
sc = Scene(dict(dt=0.005, preconditioner_type=0, newton_iter_cap=40, absolute_dhat=2e-3), log_level=0)
eng = sc.eng
T = np.eye(4); T[:3, :3] = R_CONV; T[1, 3] = 1.2
_load_urdf_capture(eng, osp.join(ASSETS, "nuka", "g1", "g1_hull.urdf"), T, True, False, 1e7, {})
eng.set_log_level(0)
bodies = {}
for r in eng.get_load_records():
    if r.body_type == 0:
        bodies[r.label] = Body(r, r.label, "prop"); sc.bodies[r.label] = bodies[r.label]; eng.add_ground_collision_skip(r.body_offset)
sc.finalize()
n = eng.native.get_num_revolute_joints(); rev = {eng.get_revolute_joint_info(i).name: i for i in range(n)}
closed = {"left_hand_thumb_0_joint": -0.5, "left_hand_thumb_1_joint": 0.6, "left_hand_thumb_2_joint": 0.8, "left_hand_index_0_joint": -1.2, "left_hand_index_1_joint": -1.5, "left_hand_middle_0_joint": -1.2, "left_hand_middle_1_joint": -1.5}
strength = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
eng.set_max_revolute_step_per_frame(100.0)
for nme in closed:
    eng.set_revolute_strength(rev[nme], strength)
infos = [(nme, eng.get_revolute_joint_info(rev[nme])) for nme in closed]
print("joint info:", [(nme, round(i.lower_limit, 2), round(i.upper_limit, 2), i.strength_ratio) for nme, i in infos])
def idx_pos():
    V = np.asarray(eng.get_abd_body(bodies["left_hand_index_1_link"].body_id).get_vertices(), float); return V.mean(0)
p0 = idx_pos()
for k in range(201):
    for nme, v in closed.items():
        eng.set_revolute_target(rev[nme], v)
    if k % 40 == 0:
        q = np.asarray(eng.get_revolute_current_angles_abs(), float)
        print(f"t {k*0.005:.2f}s strength {strength}: " + " ".join(f"{nme.replace('left_hand_','').replace('_joint','')}={q[rev[nme]]:+.2f}" for nme in closed) + f" | index_1 moved {np.linalg.norm(idx_pos()-p0)*1e3:.1f} mm | drive tau index_0 {eng.get_revolute_drive_torque(rev['left_hand_index_0_joint']):+.3f}", flush=True)
    sc.step()

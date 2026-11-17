import os.path as osp, sys, numpy as np
sys.path.insert(0, "examples/robotgym"); sys.path.insert(0, "examples/nuka/tools")
from common import ASSETS, Scene, _load_urdf_capture
from urdf_kin import UrdfKin
R_CONV = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])
sc = Scene(dict(dt=0.005, preconditioner_type=0, newton_iter_cap=40, absolute_dhat=2e-3), log_level=0)
T = np.eye(4); T[:3, :3] = R_CONV; T[1, 3] = 1.2
# load at a NON-zero pose (like the fold scene) so the initial-angle-offset path is exercised
init = {"left_shoulder_pitch_joint": -0.8, "left_elbow_joint": 1.2, "right_shoulder_pitch_joint": -0.8, "right_elbow_joint": 1.2, "left_shoulder_roll_joint": 0.3, "right_shoulder_roll_joint": -0.3, "waist_pitch_joint": 0.1}
log = _load_urdf_capture(sc.eng, osp.join(ASSETS, "nuka", "g1", "g1_hull.urdf"), T, True, False, 1e7, init)
for l in log.splitlines():
    if "revolute-limit" in l: print(l[:230])
kin = UrdfKin(osp.join(ASSETS, "nuka", "g1", "g1_hull.urdf"))
n = sc.eng.native.get_num_revolute_joints()
for i in range(n):
    info = sc.eng.get_revolute_joint_info(i); j = kin.joints.get(info.name)
    flag = "" if j is None or (abs(info.lower_limit - j["lower"]) < 1e-6 and abs(info.upper_limit - j["upper"]) < 1e-6) else "  <-- differs from URDF (%.2f, %.2f)" % (j["lower"], j["upper"])
    if i in (26, 39, 41) or flag:
        print(f"joint {i:2d} {info.name:32s} engine limits ({info.lower_limit:+.3f}, {info.upper_limit:+.3f}){flag}")

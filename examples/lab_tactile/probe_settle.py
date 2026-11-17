"""Per-frame cost probe of the pre-settle phase (joint-space move + gripper open)."""
import os.path as osp, sys, time
import numpy as np
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from lab_scene import LabScene, T_of
from lab_task import top_down_R, minjerk
from run_lab_demo import HOME_TCP

import os
DT = float(os.environ.get("LAB_DT", "0.005")); PRE = int(os.environ.get("LAB_PRECOND", "0"))
sc = LabScene(dt=DT, precond=PRE, tactile_gpu=False, log_level=0)
print("[probe] dt", DT, "precond", PRE, "joint ratio", os.environ.get("LAB_JOINT_RATIO", "1000"), flush=True)
q0 = sc.q.copy()
q1, ok = sc.arm.ik(T_of(top_down_R("z"), HOME_TCP), sc.arm.HOME, iters=80)
print("[probe] home ik", ok, q1.round(3), flush=True)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
N = int(round(2.5 / DT))
move = "--no-move" not in sys.argv
grip = "--no-grip" not in sys.argv
if grip:
    sc.gripper.command(mode="pos", gap_mm=60, vmax_mm_s=60)
nt0, cp0 = 0, 0
for k in range(n):
    s = (k + 1) / N
    q = q0 + minjerk(s) * (q1 - q0) if move else q0
    sc.drive_arm(q)
    t0 = time.time(); gi = sc.gripper.update(); tg = time.time() - t0
    t0 = time.time(); sc.eng.step(); ts = time.time() - t0
    nt, cp = sc.eng.native.get_total_newton_iters(), sc.eng.native.get_total_collision_pairs()
    if k % 5 == 0:
        print(f"[probe] fr {k:3d} step {ts*1e3:7.1f} ms (grip-update {tg*1e3:5.1f} ms) newton +{nt-nt0:3d} pairs +{cp-cp0:7.0f} max {sc.eng.native.get_max_collision_pairs():.0f} gap {gi['gap_mm']:5.1f} Fn {gi['fn_left']:.3f}/{gi['fn_right']:.3f}", flush=True)
    nt0, cp0 = nt, cp
print("[probe] done", flush=True)

"""Joint tracking probe: move slowly to the home pose, hold, compare the
engine's measured joint angles / link poses with the commanded FK."""
import os, os.path as osp, sys, time
import numpy as np
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from lab_scene import LabScene, T_of, HAND
from lab_task import top_down_R, minjerk
from run_lab_demo import HOME_TCP
DT = float(os.environ.get("LAB_DT", "0.005"))
sc = LabScene(dt=DT, precond=int(os.environ.get("LAB_PRECOND", "0")), tactile_gpu=False, log_level=0)
q0 = sc.q.copy()
q1, ok = sc.arm.ik(T_of(top_down_R("z"), HOME_TCP), sc.arm.HOME, iters=80)
move_s = float(os.environ.get("LAB_MOVE_S", "5.0")); hold_s = 1.5
N = int(round(move_s / DT)); H = int(round(hold_s / DT))
sc.gripper.command(mode="pos", gap_mm=60, vmax_mm_s=60)
tw = time.time(); nt0 = 0
for k in range(N + H):
    s = min(1.0, (k + 1) / N)
    q = q0 + minjerk(s) * (q1 - q0)
    sc.drive_arm(q); sc.gripper.update(); sc.eng.step()
    if k % 100 == 0 or k == N + H - 1:
        qm = sc.arm_q()
        T_fk = sc.arm.fk(q); T_m = sc.tcp_pose()
        nt = sc.eng.native.get_total_newton_iters()
        tau = [sc.eng.get_revolute_drive_torque(j) for j in sc.arm_joint_ids]
        print(f"[track] fr {k:4d} ({(time.time()-tw)/(k+1)*1e3:6.1f} ms/fr, newton/fr {(nt-nt0)/100:.1f}) "
              f"dq(cmd-meas) {np.round(q - qm, 3)} tcp err {np.linalg.norm(T_fk[:3,3]-T_m[:3,3])*1e3:6.1f} mm "
              f"tau {np.round(tau, 1)}", flush=True)
        nt0 = nt
lp = sc.arm.link_poses(q1)
for n in ["fr3_link2", "fr3_link4", "fr3_link6", HAND]:
    b = sc.bodies[n]; Tm = b.pose(sc.eng) @ sc.link_T0[n]
    print(f"[track] {n:9s} FK {lp[n][:3,3].round(4)} meas {Tm[:3,3].round(4)} err {np.linalg.norm(lp[n][:3,3]-Tm[:3,3])*1e3:.1f} mm", flush=True)
print("[track] done", flush=True)

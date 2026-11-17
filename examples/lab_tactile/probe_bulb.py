"""Bulb-grasp stability probe: grasp, lift, hold, transport; log tube tilt."""
import os, os.path as osp, sys, time
if __name__ == "__main__": import os as _os; _os.environ.setdefault("LIQ_UG", "0")   # main-path example, when run as a script: pinned to the main liquid path (the engine default is the UG mode since 2026-11-18)
import numpy as np
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from lab_scene import LabScene, T_of
from lab_task import top_down_R, minjerk, best_top_down, lerp_T
from run_lab_demo import HOME_TCP
DY = float(os.environ.get("PB_DY", "0.007"))        # pad centre height above bulb centre
FG = float(os.environ.get("PB_FORCE", "2.5"))        # grip force per pad
AX = os.environ.get("PB_AXIS", "x")                  # closing axis
sc = LabScene(dt=0.005, precond=0, tactile_gpu=False, log_level=0)
q0 = sc.q.copy()
sc.home_pose = T_of(top_down_R("z"), HOME_TCP); sc.cmd_pose = sc.arm.fk(q0)
q1, _ = sc.arm.ik(sc.home_pose, sc.arm.HOME, iters=80)
sc.gripper.command(mode="pos", gap_mm=40, vmax_mm_s=60)
N = 400
for k in range(N):
    sc.drive_arm(q0 + minjerk((k + 1) / N) * (q1 - q0)); sc.gripper.update(); sc.eng.step()
sc.q = q1
bc = sc.bulb_centre(); p_g = bc + [0, DY, 0]
R = best_top_down(sc, AX, p_g); Tg = T_of(R, p_g)
tip_i = sc._tip_idx()
def tilt():
    v = sc.dropper_verts(); b = v[sc.dropper["bulb_mask"]].mean(0); t = v[tip_i]; a = b - t
    return np.degrees(np.arccos(a[1] / np.linalg.norm(a))), b
def run(label, T_from, T_to, dur, grip=None):
    if grip: sc.gripper.command(**grip)
    n = int(dur / 0.005); t0 = time.time()
    for k in range(n):
        T = lerp_T(T_from, T_to, (k + 1) / n)
        q, ok = sc.arm.ik(T, sc.q, iters=15); sc.q = q; sc.drive_arm(q)
        gi = sc.gripper.update(); sc.eng.step()
        if k % 40 == 0 or k == n - 1:
            th, b = tilt()
            print(f"[bulb] {label:14s} {k*0.005:4.2f}s tilt {th:5.1f} deg  gap {gi['gap_mm']:5.1f}  Fn {gi['fn_left']:.2f}/{gi['fn_right']:.2f}  bulb y {b[1]*1e3:6.1f}", flush=True)
    print(f"[bulb] {label} done ({(time.time()-t0)/n*1e3:.0f} ms/fr)", flush=True)
    return T_to
print(f"[bulb] config DY={DY} F={FG} axis={AX} pcg={os.environ.get('LAB_PCG_TOL')} newton={os.environ.get('LAB_NEWTON_TOL')} precond={os.environ.get('LAB_PRECOND')}", flush=True)
T = run("approach", sc.arm.fk(q1), T_of(R, p_g + [0, 0.12, 0]), 1.2)
T = run("descend", T, Tg, 1.0)
T = run("grasp", T, Tg, 2.2, grip=dict(mode="safe", force_N=FG))
T = run("lift", T, T_of(R, p_g + [0, 0.14, 0]), 2.0)
T = run("hold", T, T, 0.5)
T = run("transport", T, T_of(R, p_g + [0, 0.14, -0.25]), 2.2)
T = run("hold2", T, T, 1.5)
print("[bulb] done", flush=True)

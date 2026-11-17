"""Dropper-lag probe: replay stage-1 grasp / lift / transport and log, EVERY
physics frame, the dropper tilt, where the tip leans relative to the gripper
velocity, Newton iterations and wall time.  Env:
  LAB_NEWTON_TOL (1e-3)   PB_DT (0.005)   PB_FORCE (6 N secure grip)
  PB_TDUR (3.2 s transport)  PB_TDIST (0.25 m)  PB_TAG (name)  PB_OUT (dir)
"""
import os, os.path as osp, sys, time
import numpy as np
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from lab_scene import LabScene, T_of
from lab_task import top_down_R, minjerk, best_top_down, lerp_T
from run_lab_demo import HOME_TCP

DT = float(os.environ.get("PB_DT", "0.005"))
FG = float(os.environ.get("PB_FORCE", "6.0"))
TDUR = float(os.environ.get("PB_TDUR", "3.2"))
TDIST = float(os.environ.get("PB_TDIST", "0.25"))
TAG = os.environ.get("PB_TAG", "probe")
LOCK = int(os.environ.get("PB_LOCK", "0"))   # 1: freeze finger targets after the secure grip (no servo)
OUT = os.environ.get("PB_OUT", osp.join(osp.dirname(__file__), "..", "..", "out", "lag"))
os.makedirs(OUT, exist_ok=True)

sc = LabScene(dt=DT, precond=0, tactile_gpu=False, log_level=0)
eng = sc.eng
print(f"[lag] {TAG}: dt={DT} F={FG} tol={os.environ.get('LAB_NEWTON_TOL','1e-3')} "
      f"vtol={os.environ.get('LAB_NEWTON_VTOL')} pcg={os.environ.get('LAB_PCG_TOL')} lock={LOCK} tdur={TDUR} tdist={TDIST}", flush=True)

q0 = sc.q.copy()
sc.home_pose = T_of(top_down_R("z"), HOME_TCP)
q1, _ = sc.arm.ik(sc.home_pose, sc.arm.HOME, iters=80)
sc.gripper.command(mode="pos", gap_mm=40, vmax_mm_s=60)
N = int(round(3.0 / DT))
for k in range(N):
    sc.drive_arm(q0 + minjerk((k + 1) / N) * (q1 - q0)); sc.gripper.update(); eng.step()
sc.q = q1
bc = sc.bulb_centre(); p_g = bc + [0, 0.007, 0]
R = best_top_down(sc, "x", p_g); Tg = T_of(R, p_g)

rows = []
def measure(label, k, T_cmd, ms, nt, gi):
    v = sc.dropper_verts(); b = v[sc.dropper["bulb_mask"]].mean(0); tip = v[sc._tip_idx()]
    a = b - tip; L = np.linalg.norm(a)
    tilt = np.degrees(np.arccos(np.clip(a[1] / L, -1, 1)))
    rows.append((label, k * DT, tilt, *(tip - b), *T_cmd[:3, 3], ms, nt,
                 float(gi.get("fn_left", np.nan)), float(gi.get("fn_right", np.nan)), float(gi.get("gap_mm", np.nan))))
    return tilt

def run(label, T_from, T_to, dur, grip=None, smooth=True):
    if grip: sc.gripper.command(**grip)
    n = int(round(dur / DT)); t0 = time.time(); nt0 = eng.native.get_total_newton_iters()
    for k in range(n):
        s = (k + 1) / n
        T = lerp_T(T_from, T_to, s)   # lerp_T applies min-jerk itself
        q, ok = sc.arm.ik(T, sc.q, iters=15); sc.q = q; sc.drive_arm(q)
        gi = sc.gripper.update(); ta = time.time(); eng.step(); ms = (time.time() - ta) * 1e3
        nt1 = eng.native.get_total_newton_iters(); nt = nt1 - nt0; nt0 = nt1
        tilt = measure(label, len(rows), T, ms, nt, gi)
        if k % int(round(0.2 / DT)) == 0 or k == n - 1:
            print(f"[lag] {label:10s} {k*DT:5.2f}s tilt {tilt:5.2f} deg  nt {nt:3d}  {ms:6.0f} ms  "
                  f"Fn {gi['fn_left']:.2f}/{gi['fn_right']:.2f} gap {gi['gap_mm']:5.2f}", flush=True)
    print(f"[lag] {label} done ({(time.time()-t0)/n*1e3:.0f} ms/fr)", flush=True)
    return T_to

T = run("approach", sc.arm.fk(q1), T_of(R, p_g + [0, 0.12, 0]), 1.6)
T = run("descend", T, Tg, 1.6)
T = run("grasp", T, Tg, 2.2, grip=dict(mode="safe", force_N=2.5))
T = run("secure", T, Tg, 0.8, grip=dict(mode="safe", force_N=FG))
if LOCK:
    sc.gripper.mode = "hold"      # update() leaves _target_d untouched in this mode
    print("[lag] finger targets frozen (no force servo)", flush=True)
T = run("lift", T, T_of(R, p_g + [0, 0.14, 0]), 2.6)
T = run("hold", T, T, 1.0)
SCEN = os.environ.get("PB_SCENARIO", "transport")
if SCEN == "beaker":
    # contact-heavy: carry the dropper over the mixing beaker and lower the tip 30 mm into it
    bk = np.asarray(sc.prop_T["beaker_mix"][:3, 3], float)
    T = run("transport", T, T_of(R, [bk[0], p_g[1] + 0.14, bk[2]]), TDUR)
    T = run("lower", T, T_of(R, [bk[0], p_g[1] + 0.14 - 0.124, bk[2]]), 1.6)
    T = run("hold2", T, T, 1.0)
else:
    T = run("transport", T, T_of(R, p_g + [0, 0.14, -TDIST]), TDUR)
    T = run("hold2", T, T, 2.0)
arr = np.array([r[1:] for r in rows], np.float64); lab = np.array([r[0] for r in rows])
np.savez(osp.join(OUT, f"{TAG}.npz"), label=lab, data=arr, dt=DT, force=FG,
         cols=np.array("t tilt lx ly lz tx ty tz ms nt fnl fnr gap".split()))
print(f"[lag] {TAG} saved {len(rows)} frames -> {OUT}/{TAG}.npz", flush=True)

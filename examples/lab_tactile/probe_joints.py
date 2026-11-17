"""Per-joint FK consistency: move one joint at a time, compare the engine's
hand/link7 pose with the URDF FK (and with the sign-flipped angle)."""
import os, os.path as osp, sys
import numpy as np
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from lab_scene import LabScene, HAND
from lab_task import minjerk
sc = LabScene(dt=0.005, precond=0, tactile_gpu=False, log_level=0, with_props=False,
              with_dropper=False, with_gels=False)
q0 = sc.q.copy()
joints = [int(a) for a in sys.argv[1].split(",")] if len(sys.argv) > 1 else list(range(7))
print("[joints] q0", q0.round(3), "limits", sc.arm.lo.round(2), sc.arm.hi.round(2), flush=True)

def meas(n):
    b = sc.bodies[n]
    return b.pose(sc.eng) @ sc.link_T0[n]

for j in joints:
    amp = 0.5
    if q0[j] + amp > sc.arm.hi[j] - 0.05:
        amp = -0.5
    N, H = 150, 60
    for k in range(N + H):
        s = min(1.0, (k + 1) / N)
        q = q0.copy(); q[j] = q0[j] + amp * minjerk(s)
        sc.drive_arm(q); sc.eng.step()
    qm = sc.arm_q()
    Tm = meas(HAND)
    T_fk = sc.arm.fk(q, HAND)
    qf = q.copy(); qf[j] = q0[j] - amp
    T_fl = sc.arm.fk(qf, HAND)
    print(f"[joints] j{j+1} amp {amp:+.2f}: meas-angle err {qm[j]-q[j]:+.4f} | hand meas {Tm[:3,3].round(4)} "
          f"FK {T_fk[:3,3].round(4)} err {np.linalg.norm(Tm[:3,3]-T_fk[:3,3])*1e3:6.1f} mm | "
          f"flipped-FK err {np.linalg.norm(Tm[:3,3]-T_fl[:3,3])*1e3:6.1f} mm | rot err {np.degrees(np.arccos(np.clip((np.trace(Tm[:3,:3].T @ T_fk[:3,:3])-1)/2,-1,1))):5.1f} deg", flush=True)
    for k in range(N):   # back to q0
        s = (k + 1) / N
        q = q0.copy(); q[j] = q0[j] + amp * (1 - minjerk(s))
        sc.drive_arm(q); sc.eng.step()
print("[joints] done", flush=True)

"""Kinematic clearance of the FR3 wrist/hand/fingers to the fixed slab along a bed-sheet timeline (no engine, CPU).
Link geometry = the rest vertices recorded by any previous bedsheet run, moved by FK(q_t) @ inv(FK(q_rec0)); prints the
minimum link-to-slab distance per segment (negative = the commanded path drives that link into the slab).
The fingers point 25 deg up, so the wrist (link7) hangs ~9 cm below the TCP: with pinch_x 0.40 the lay-down drove it
~10 mm into the slab's side edge (arm jammed, Newton at the cap, released cloth frozen) while the joint angles still
tracked (soft joints) -- check the path with this before running the solver.
  python examples/robotgym/check_bedsheet_clearance.py REC.npz [--pinch-x 0.46] [--end liftoff] [--lay-y 0.06] [--lay-z 0.54] [--side right]"""
import sys, argparse, numpy as np
import os.path as osp
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '..', 'lab_tactile'))
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from lab_rec import load_recording
from common import Arm
import bedsheet_fold as bf
ap = argparse.ArgumentParser(); ap.add_argument('rec'); ap.add_argument('--pinch-x', type=float, default=bf.PINCH_X)
ap.add_argument('--end', default='liftoff'); ap.add_argument('--lay-y', type=float, default=0.06); ap.add_argument('--lay-z', type=float, default=0.54)
ap.add_argument('--dt', type=float, default=0.02); ap.add_argument('--side', default='both')
a = ap.parse_args()
z = load_recording(a.rec); names = [str(n) for n in z['body_names']]
lo = np.array([-0.5, 0.28, 0.15]); hi = np.array([0.5, 0.30, 0.55])
def boxd(P):
    q = np.maximum(np.maximum(lo - P, P - hi), 0); out = np.linalg.norm(q, axis=1)
    inside = np.all((P > lo) & (P < hi), axis=1)
    return np.where(inside, -np.minimum(P - lo, hi - P).min(axis=1), out)

src = open(bf.__file__).read()
LINKS = ['fr3_link5', 'fr3_link6', 'fr3_link7', 'fr3_hand', 'fr3_leftfinger', 'fr3_rightfinger']
for ai, arm_meta in enumerate(z['meta']['arms']):
    side = arm_meta['name']
    if a.side != 'both' and side != a.side:
        continue
    arm = Arm(side, np.asarray(arm_meta['base_T'], float)); kin = arm.kin
    q_rec0 = np.asarray(z['q_arms'][0][ai], float); fd0 = 0.5 * float(z['gap_arms'][0][ai]) * (1e-3 if z['gap_arms'][0][ai] > 1 else 1.0)
    LP0 = kin.link_poses(q_rec0, fd0)
    rest = {l: np.asarray(z['body_rest'][names.index(f'{side}/{l}')], float) for l in LINKS}
    tl = bf.timeline(side, 5e-4, a.end, a.lay_y, a.lay_z, a.pinch_x)
    q = kin.ik(tl.segs[0].T_to, kin.HOME, iters=200)[0]
    fd = 0.0; worst = {}; ikerr = 0.0
    for t in np.arange(0.0, tl.total + 1e-9, a.dt):
        seg, s = tl.at(t); T = seg.pose(s)
        q, ok = kin.ik(T, q, iters=60)
        ikerr = max(ikerr, np.linalg.norm(kin.fk(q)[:3, 3] - T[:3, 3]))
        if seg.finger is not None:
            fd += float(np.clip(seg.finger - fd, -0.04 * a.dt, 0.04 * a.dt))
        LP = kin.link_poses(q, fd)
        for l in LINKS:
            M = LP[l] @ np.linalg.inv(LP0[l]); P = rest[l] @ M[:3, :3].T + M[:3, 3]
            d = boxd(P).min()
            key = (seg.label, l)
            if key not in worst or d < worst[key][0]:
                worst[key] = (d, t)
    print(f"== {side}: pinch_x {a.pinch_x} end {a.end} lay_y {a.lay_y} lay_z {a.lay_z}; IK worst {ikerr*1e3:.1f} mm")
    labels = []
    for (lab, l) in worst:
        if lab not in labels:
            labels.append(lab)
    for lab in labels:
        row = [f"{l[4:]:>11s} {worst[(lab, l)][0]*1e3:6.1f}" for l in LINKS]
        print(f"  {lab[:22]:22s} " + ' '.join(row))

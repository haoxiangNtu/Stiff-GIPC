"""Where does the recorded right-hand pinch hold the dropper bulb? (gives RIGHT_BULB_IN_PALM)

For candidate points around the thumb-index midpoint (palm frame) it takes the nearest thumb surface point
and the nearest index/middle-finger surface point; a point counts as caged when those two directions are
opposed (cos < -0.7) and the palm is not closer than 4 mm. The radius needed to touch both sides is
max(d_thumb, d_finger); the bulb centre is the point that minimises the worst such radius over the loosest
part of the hold (17-23 s), then the per-time clearances are printed for the whole grasp.
"""
import os.path as osp
import sys

import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import robot_assets as RA  # noqa: E402
import rokae_dropper as R  # noqa: E402


def main():
    kin = RA.URDFKin(RA.build()["sim"])
    tele = R.Teleop(kin=kin)
    samples = {}

    def clouds(t):
        f = kin.fk(tele.at(t))
        P = np.linalg.inv(f["right_palm_link"])
        th, fg, other = [], [], []
        for link, (path, T) in kin.meshes.items():
            if not (link.startswith("right_finger") or link == "right_palm_link"):
                continue
            if link not in samples:
                samples[link] = trimesh.load(path).sample(4000)
            v = (P @ f[link] @ T @ np.c_[samples[link], np.ones(len(samples[link]))].T).T[:, :3]
            (th if "finger1" in link else fg if ("finger2" in link or "finger3" in link) else other).append(v)
        return [cKDTree(np.vstack(x)) for x in (th, fg, other)], f

    hold = [17.0, 18.0, 20.0, 23.0]
    f = kin.fk(tele.at(hold[0]))
    pin = np.linalg.inv(f["right_palm_link"]) @ np.r_[R.pinch(f, "right"), 1.0]
    g = np.mgrid[-0.04:0.04:0.0015, -0.04:0.04:0.0015, -0.04:0.04:0.0015].reshape(3, -1).T + pin[:3]
    worst = np.zeros(len(g))
    for t in hold:
        (tt, tf, to), _ = clouds(t)
        dt, it = tt.query(g)
        df, jf = tf.query(g)
        do, _ = to.query(g)
        ut = (tt.data[it] - g) / dt[:, None]
        uf = (tf.data[jf] - g) / df[:, None]
        ok = ((ut * uf).sum(1) < -0.7) & (do > 0.004)
        worst = np.maximum(worst, np.where(ok, np.maximum(dt, df), np.inf))
    c = g[np.argmin(worst)]
    print(f"RIGHT_BULB_IN_PALM = {np.round(c, 4).tolist()}  (radius touching both sides over the hold: {worst.min()*1e3:.1f} mm)")
    for t in [14.5, 15, 15.5, 16, 16.5, 17, 18, 20, 23, 24, 25, 26, 26.5, 27, 30, 31.5, 32.5]:
        (tt, tf, to), f = clouds(t)
        print(f"t {t:5.1f}  thumb {tt.query(c)[0]*1e3:5.1f}  index/middle {tf.query(c)[0]*1e3:5.1f}  palm {to.query(c)[0]*1e3:5.1f} mm")


if __name__ == "__main__":
    main()

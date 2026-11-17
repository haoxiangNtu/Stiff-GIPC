"""Minimal URDF kinematics for the FR3 arm: FK of the joint chain + damped
least-squares IK with joint limits and a posture bias (pure numpy).

The chain is read from the URDF (origins, axes, limits), so the FK matches the
engine's importer convention: joint angles are ABSOLUTE URDF angles, and the
engine's targets are absolute too (its importer clamps the zero pose into the
limits and stores the clamp as the initial offset; see
``urdf_scene_importer.cpp``).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np


def rpy_to_R(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def xyz_rpy_T(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = rpy_to_R(rpy)
    T[:3, 3] = xyz
    return T


def axis_angle_R(axis, ang):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def rot_log(R):
    """Rotation vector of R (3,)."""
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v / (2 * np.sin(th)) * th


class UrdfChain:
    def __init__(self, urdf_path: str, base_T: np.ndarray | None = None):
        root = ET.parse(urdf_path).getroot()
        self.joints = {}
        self.children = {}
        for j in root.findall("joint"):
            name = j.get("name")
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            ax = j.find("axis")
            axis = [float(v) for v in (ax.get("xyz", "1 0 0") if ax is not None else "1 0 0").split()]
            lim = j.find("limit")
            lo = float(lim.get("lower", "-1e9")) if lim is not None else -1e9
            hi = float(lim.get("upper", "1e9")) if lim is not None else 1e9
            info = dict(name=name, type=j.get("type"), T=xyz_rpy_T(xyz, rpy), axis=np.array(axis),
                        lo=lo, hi=hi, parent=j.find("parent").get("link"),
                        child=j.find("child").get("link"))
            self.joints[name] = info
            self.children.setdefault(info["parent"], []).append(name)
        self.parent_joint = {info["child"]: n for n, info in self.joints.items()}
        self.base_T = np.eye(4) if base_T is None else np.asarray(base_T, float)

    def path_to(self, link: str):
        """Joint names from the root to `link`."""
        out = []
        while link in self.parent_joint:
            jn = self.parent_joint[link]
            out.append(jn)
            link = self.joints[jn]["parent"]
        return out[::-1]

    def fk(self, q: dict, link: str) -> np.ndarray:
        """World pose of `link` for joint values q (dict name -> value)."""
        T = self.base_T.copy()
        for jn in self.path_to(link):
            j = self.joints[jn]
            T = T @ j["T"]
            v = q.get(jn, 0.0)
            if j["type"] in ("revolute", "continuous"):
                R = np.eye(4)
                R[:3, :3] = axis_angle_R(j["axis"], v)
                T = T @ R
            elif j["type"] == "prismatic":
                P = np.eye(4)
                P[:3, 3] = j["axis"] / np.linalg.norm(j["axis"]) * v
                T = T @ P
        return T


class Fr3Arm:
    ARM_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
    TCP = "fr3_hand_tcp"
    HOME = np.array([0.0, -0.6, 0.0, -2.2, 0.0, 1.6, 0.785])   # a comfortable posture

    def __init__(self, urdf_path: str, base_T: np.ndarray | None = None):
        self.chain = UrdfChain(urdf_path, base_T)
        self.lo = np.array([self.chain.joints[j]["lo"] for j in self.ARM_JOINTS])
        self.hi = np.array([self.chain.joints[j]["hi"] for j in self.ARM_JOINTS])
        # StiffGIPC's URDF importer clamps every revolute limit to +-179 deg
        # (kSafeAngleLimit = 3.124 rad, atan2-based angle): FR3 joint 6's URDF
        # range 0.54..4.52 is therefore 0.54..3.10 in the engine.  Plan inside it.
        self.lo = np.maximum(self.lo, -3.10)
        self.hi = np.minimum(self.hi, 3.10)

    def clamp(self, q):
        return np.clip(q, self.lo + 0.02, self.hi - 0.02)

    def fk(self, q, link: str | None = None) -> np.ndarray:
        qd = {j: float(v) for j, v in zip(self.ARM_JOINTS, q)}
        return self.chain.fk(qd, link or self.TCP)

    def link_poses(self, q, finger_d: float = 0.0) -> dict:
        qd = {j: float(v) for j, v in zip(self.ARM_JOINTS, q)}
        qd["fr3_finger_joint1"] = finger_d
        qd["fr3_finger_joint2"] = finger_d
        links = [f"fr3_link{i}" for i in range(8)] + ["fr3_hand", "fr3_leftfinger", "fr3_rightfinger",
                                                     "fr3_hand_tcp"]
        return {l: self.chain.fk(qd, l) for l in links}

    def jacobian(self, q, eps=1e-6) -> np.ndarray:
        """6x7 geometric Jacobian by central differences (pos; rotation vector)."""
        T0 = self.fk(q)
        J = np.zeros((6, 7))
        for i in range(7):
            dq = np.zeros(7)
            dq[i] = eps
            Tp, Tm = self.fk(q + dq), self.fk(q - dq)
            J[:3, i] = (Tp[:3, 3] - Tm[:3, 3]) / (2 * eps)
            dR = Tp[:3, :3] @ Tm[:3, :3].T
            J[3:, i] = rot_log(dR) / (2 * eps)
        return J

    def ik(self, T_goal: np.ndarray, q0: np.ndarray, iters: int = 60, tol_pos=2e-4,
           tol_rot=2e-3, damping=5e-2, posture=None, w_post=0.05, gain=0.8,
           max_step=0.15) -> tuple[np.ndarray, bool]:
        """Damped least squares with a null-space posture bias and joint limits
        (a joint sitting on a limit cannot be pushed further out).  Returns (q, ok)."""
        q = np.array(q0, float)
        post = self.HOME if posture is None else np.asarray(posture, float)
        lo, hi = self.lo + 0.02, self.hi - 0.02
        for _ in range(iters):
            T = self.fk(q)
            e_p = T_goal[:3, 3] - T[:3, 3]
            e_r = rot_log(T_goal[:3, :3] @ T[:3, :3].T)
            if np.linalg.norm(e_p) < tol_pos and np.linalg.norm(e_r) < tol_rot:
                return q, True
            e = np.concatenate([e_p, e_r]) * gain
            en = np.linalg.norm(e_p) + 0.1 * np.linalg.norm(e_r)
            lam = damping if en > 0.02 else max(damping * 0.1, 2e-3)   # sharpen near the goal
            wp = w_post * min(1.0, en / 0.02)                           # posture pull fades out
            J = self.jacobian(q)
            for _k in range(2):
                JJt = J @ J.T + (lam ** 2) * np.eye(6)
                dq = J.T @ np.linalg.solve(JJt, e)
                N = np.eye(7) - J.T @ np.linalg.solve(JJt, J)
                dq += wp * (N @ (post - q))
                blocked = ((q <= lo + 1e-9) & (dq < 0)) | ((q >= hi - 1e-9) & (dq > 0))
                if not blocked.any():
                    break
                J = J.copy()
                J[:, blocked] = 0.0            # drop the blocked joints and resolve
            n = np.linalg.norm(dq)
            if n > max_step:
                dq *= max_step / n
            q = np.clip(q + dq, lo, hi)
        T = self.fk(q)
        e_p = T_goal[:3, 3] - T[:3, 3]
        e_r = rot_log(T_goal[:3, :3] @ T[:3, :3].T)
        return q, (np.linalg.norm(e_p) < 5 * tol_pos and np.linalg.norm(e_r) < 5 * tol_rot)


def make_pose(pos, R) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


if __name__ == "__main__":
    import os.path as osp
    here = osp.dirname(osp.abspath(__file__))
    urdf = osp.join(here, "..", "..", "Assets", "fr3", "fr3_franka_hand.urdf")
    arm = Fr3Arm(urdf)
    q_ready = np.array([0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4])
    T = arm.fk(q_ready)
    print("ready pose TCP (URDF base frame, Z-up):", T[:3, 3].round(4), "\nR=\n", T[:3, :3].round(3))
    print("limits lo", arm.lo.round(3), "\n       hi", arm.hi.round(3))
    # IK round trip
    goal = make_pose([0.45, 0.10, 0.25], np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float))
    q, ok = arm.ik(goal, arm.HOME)
    Tq = arm.fk(q)
    print("IK ok", ok, "q", q.round(3), "\n pos err mm", np.linalg.norm(Tq[:3, 3] - goal[:3, 3]) * 1e3,
          "rot err", np.linalg.norm(rot_log(goal[:3, :3] @ Tq[:3, :3].T)))
    lp = arm.link_poses(q, 0.04)
    for k in ["fr3_hand", "fr3_leftfinger", "fr3_rightfinger", "fr3_hand_tcp"]:
        print(f"  {k:16s} {lp[k][:3,3].round(4)}")

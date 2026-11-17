"""Minimal URDF kinematics: forward kinematics over the joint tree and a damped-least-squares IK for one chain.
Used to place floating-base robots (foot clearance) and to plan arm targets (G1 shirt fold). No external dependencies."""
import math
import xml.etree.ElementTree as ET
import numpy as np


def rpy_to_R(rpy):
    r, p, y = [float(v) for v in rpy]
    Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def axis_angle_R(axis, th):
    a = np.asarray(axis, float); n = np.linalg.norm(a)
    if n < 1e-12 or abs(th) < 1e-15:
        return np.eye(3)
    a = a / n
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def so3_log(R):
    c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * th / (2.0 * math.sin(th))


def _vec(el, key, default="0 0 0"):
    return np.array([float(v) for v in (el.get(key, default) if el is not None else default).split()])


class UrdfKin:
    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.links = [l.get("name") for l in root.findall("link")]
        self.joints, self.child_joints, self.parent_joint = {}, {}, {}
        self.effort, self.velocity_limit = {}, {}
        for j in root.findall("joint"):
            name, typ = j.get("name"), j.get("type")
            o = j.find("origin")
            T = np.eye(4); T[:3, :3] = rpy_to_R(_vec(o, "rpy")); T[:3, 3] = _vec(o, "xyz")
            ax = j.find("axis"); axis = _vec(ax, "xyz", "1 0 0")
            lim = j.find("limit")
            lower = float(lim.get("lower", "-3.14159")) if lim is not None else -math.pi
            upper = float(lim.get("upper", "3.14159")) if lim is not None else math.pi
            self.effort[name] = float(lim.get("effort", "0")) if lim is not None else 0.0
            self.velocity_limit[name] = float(lim.get("velocity", "0")) if lim is not None else 0.0
            parent, child = j.find("parent").get("link"), j.find("child").get("link")
            self.joints[name] = dict(type=typ, parent=parent, child=child, T=T, axis=axis, lower=lower, upper=upper)
            self.child_joints.setdefault(parent, []).append(name)
            self.parent_joint[child] = name
        roots = [l for l in self.links if l not in self.parent_joint]
        self.root = roots[0]
        self.revolute = [n for n, j in self.joints.items() if j["type"] in ("revolute", "continuous")]

    # ---- forward kinematics
    def fk(self, q=None, base_T=None):
        q = q or {}
        T = {self.root: np.eye(4) if base_T is None else np.asarray(base_T, float)}
        stack = [self.root]
        while stack:
            l = stack.pop()
            for jn in self.child_joints.get(l, []):
                j = self.joints[jn]
                Tj = T[l] @ j["T"]
                if j["type"] in ("revolute", "continuous"):
                    R = np.eye(4); R[:3, :3] = axis_angle_R(j["axis"], float(q.get(jn, 0.0))); Tj = Tj @ R
                elif j["type"] == "prismatic":
                    P = np.eye(4); P[:3, 3] = np.asarray(j["axis"], float) * float(q.get(jn, 0.0)); Tj = Tj @ P
                T[j["child"]] = Tj
                stack.append(j["child"])
        return T

    def chain(self, tip, base=None):
        """Revolute joint names from `base` (or the root) down to `tip`, in order."""
        out = []
        l = tip
        while l in self.parent_joint and l != base:
            jn = self.parent_joint[l]
            if self.joints[jn]["type"] in ("revolute", "continuous"):
                out.append(jn)
            l = self.joints[jn]["parent"]
        return out[::-1]

    # ---- inverse kinematics (damped least squares, finite-difference Jacobian)
    def ik(self, tip, T_target, q, joints, iters=150, pos_w=1.0, rot_w=0.3, lam=0.05, step=0.8, tol=1e-4, base_T=None, limits=True):
        q = dict(q); T_target = np.asarray(T_target, float)
        n = len(joints)
        for _ in range(iters):
            T = self.fk(q, base_T)[tip]
            e = np.concatenate([pos_w * (T_target[:3, 3] - T[:3, 3]), rot_w * so3_log(T_target[:3, :3] @ T[:3, :3].T)])
            if np.linalg.norm(e[:3]) < tol * pos_w and np.linalg.norm(e[3:]) < 5e-3 * rot_w:
                break
            J = np.zeros((6, n)); h = 1e-5
            for i, jn in enumerate(joints):
                q2 = dict(q); q2[jn] = q2.get(jn, 0.0) + h
                T2 = self.fk(q2, base_T)[tip]
                J[:3, i] = pos_w * (T2[:3, 3] - T[:3, 3]) / h
                J[3:, i] = rot_w * so3_log(T2[:3, :3] @ T[:3, :3].T) / h
            dq = J.T @ np.linalg.solve(J @ J.T + lam * lam * np.eye(6), e)
            for i, jn in enumerate(joints):
                v = q.get(jn, 0.0) + step * dq[i]
                if limits:
                    v = min(max(v, self.joints[jn]["lower"]), self.joints[jn]["upper"])
                q[jn] = v
        T = self.fk(q, base_T)[tip]
        return q, float(np.linalg.norm(T_target[:3, 3] - T[:3, 3])), float(np.linalg.norm(so3_log(T_target[:3, :3] @ T[:3, :3].T)))

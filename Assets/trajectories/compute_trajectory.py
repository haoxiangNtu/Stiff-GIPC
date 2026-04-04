#!/usr/bin/env python3
"""Pre-compute joint trajectories for Franka Panda and XArm7.

Reads URDF joint parameters directly (no external URDF library needed),
builds an FK chain, runs Jacobian IK to track the 32 key poses from the
Isaac Lab T-shirt folding demo, and writes trajectory files consumable by
Stiff-GIPC's trajectory playback system.

Usage:
    cd /home/ps/Downloads/Stiff-GIPC/Assets/trajectories
    python3 compute_trajectory.py            # generates both files
    python3 compute_trajectory.py --franka   # Franka only
    python3 compute_trajectory.py --xarm     # XArm7 only
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = SCRIPT_DIR.parent

GRIP_OPEN = 0.04
GRIP_CLOSE = 0.002

KEY_POSES = np.array([
    # (duration_s, x, y, z, qw, qx, qy, qz, gripper_width)
    [2.0, 0.9, -0.08, 0.60, 1, 0, 0, 0, GRIP_OPEN],
    [1.5, 0.9, -0.08, 0.40, 1, 0, 0, 0, GRIP_OPEN],
    [1.0, 0.9, -0.08, 0.40, 1, 0, 0, 0, GRIP_CLOSE],
    [1.5, 0.92, -0.08, 0.62, 1, 0, 0, 0, GRIP_CLOSE],
    [2.0, 0.48, 0.12, 0.62, 1, 0, 0, 0, GRIP_CLOSE],
    [1.0, 0.48, 0.12, 0.62, 1, 0, 0, 0, GRIP_OPEN],
    [2.0, 0.43, 0.12, 0.62, 1, 0, 0, 0, GRIP_OPEN],
    [1.5, 0.43, 0.12, 0.52, 1, 0, 0, 0, GRIP_OPEN],
    [1.0, 0.43, 0.12, 0.52, 1, 0, 0, 0, GRIP_CLOSE],
    [1.5, 0.43, 0.12, 0.60, 1, 0, 0, 0, GRIP_CLOSE],
    [2.0, 0.55, 0.12, 0.60, 1, 0, 0, 0, GRIP_CLOSE],
    [1.0, 0.55, 0.12, 0.60, 1, 0, 0, 0, GRIP_OPEN],
    [2.0, 0.67, -0.12, 0.60, 1, 0, 0, 0, GRIP_OPEN],
    [1.5, 0.67, -0.12, 0.52, 1, 0, 0, 0, GRIP_OPEN],
    [1.0, 0.67, -0.12, 0.52, 1, 0, 0, 0, GRIP_CLOSE],
    [1.5, 0.62, -0.12, 0.62, 1, 0, 0, 0, GRIP_CLOSE],
    [2.0, 0.48, -0.12, 0.62, 1, 0, 0, 0, GRIP_CLOSE],
    [1.0, 0.48, -0.12, 0.62, 1, 0, 0, 0, GRIP_OPEN],
    [2.0, 0.43, -0.12, 0.52, 1, 0, 0, 0, GRIP_OPEN],
    [1.0, 0.43, -0.12, 0.52, 1, 0, 0, 0, GRIP_CLOSE],
    [1.5, 0.50, -0.12, 0.62, 1, 0, 0, 0, GRIP_CLOSE],
    [2.0, 0.55, -0.12, 0.62, 1, 0, 0, 0, GRIP_CLOSE],
    [1.0, 0.55, -0.12, 0.62, 1, 0, 0, 0, GRIP_OPEN],
    [2.0, 0.43, 0.00, 0.60, 1, 0, 0, 0, GRIP_OPEN],
    [1.5, 0.43, 0.00, 0.52, 1, 0, 0, 0, GRIP_OPEN],
    [1.0, 0.43, 0.00, 0.52, 1, 0, 0, 0, GRIP_CLOSE],
    [1.5, 0.43, 0.00, 0.67, 1, 0, 0, 0, GRIP_CLOSE],
    [1.5, 0.55, 0.00, 0.67, 1, 0, 0, 0, GRIP_CLOSE],
    [1.5, 0.62, 0.00, 0.67, 1, 0, 0, 0, GRIP_CLOSE],
    [1.0, 0.62, 0.00, 0.67, 1, 0, 0, 0, GRIP_OPEN],
    [2.0, 0.45, 0.00, 0.72, 1, 0, 0, 0, GRIP_OPEN],
    [2.0, 0.38, 0.00, 0.55, 1, 0, 0, 0, GRIP_OPEN],
], dtype=np.float64)

# ─────────────────────────────────────────────────────────────
# Rotation helpers
# ─────────────────────────────────────────────────────────────

def rotx(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def roty(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rotz(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def rpy_to_mat(r, p, y):
    return rotz(y) @ roty(p) @ rotx(r)

def make_tf(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def rot_axis(axis_str, theta):
    """4x4 rotation matrix about named axis."""
    if axis_str == "Z":
        return make_tf(rotz(theta), [0, 0, 0])
    elif axis_str == "Y":
        return make_tf(roty(theta), [0, 0, 0])
    elif axis_str == "X":
        return make_tf(rotx(theta), [0, 0, 0])
    raise ValueError(f"Unknown axis {axis_str}")


# ─────────────────────────────────────────────────────────────
# FK Chain — manually specified from URDF data
# ─────────────────────────────────────────────────────────────

class FKChain:
    """Stores joint-frame transforms and computes FK + Jacobian."""

    def __init__(self, joint_origins, joint_axes, ee_offset, joint_limits):
        """
        joint_origins: list of 4x4 transforms (parent-to-joint-frame)
        joint_axes: list of axis strings ("X", "Y", "Z")
        ee_offset: 4x4 transform from last joint frame to EE
        joint_limits: list of (lower, upper) tuples
        """
        self.origins = joint_origins
        self.axes = joint_axes
        self.ee_offset = ee_offset
        self.limits = np.array(joint_limits)
        self.n_joints = len(joint_origins)

    def fk(self, q, base_T=np.eye(4)):
        """Forward kinematics: returns 4x4 EE transform."""
        T = base_T.copy()
        for i in range(self.n_joints):
            T = T @ self.origins[i] @ rot_axis(self.axes[i], q[i])
        T = T @ self.ee_offset
        return T

    def fk_position(self, q, base_T=np.eye(4)):
        return self.fk(q, base_T)[:3, 3]

    def jacobian(self, q, base_T=np.eye(4), eps=1e-6):
        """Numerical 3xN position Jacobian."""
        p0 = self.fk_position(q, base_T)
        J = np.zeros((3, self.n_joints))
        for i in range(self.n_joints):
            q_plus = q.copy()
            q_plus[i] += eps
            J[:, i] = (self.fk_position(q_plus, base_T) - p0) / eps
        return J

    def clamp(self, q):
        return np.clip(q, self.limits[:, 0], self.limits[:, 1])


def build_franka_chain():
    """Build FK chain for single-arm Franka Panda (FR3 variant)."""
    hp = math.pi / 2.0
    origins = [
        make_tf(rpy_to_mat(0, 0, 0), [0, 0, 0.333]),
        make_tf(rpy_to_mat(-hp, 0, 0), [0, 0, 0]),
        make_tf(rpy_to_mat(hp, 0, 0), [0, -0.316, 0]),
        make_tf(rpy_to_mat(hp, 0, 0), [0.0825, 0, 0]),
        make_tf(rpy_to_mat(-hp, 0, 0), [-0.0825, 0.384, 0]),
        make_tf(rpy_to_mat(hp, 0, 0), [0, 0, 0]),
        make_tf(rpy_to_mat(hp, 0, 0), [0.088, 0, 0]),
    ]
    axes = ["Z"] * 7
    limits = [
        (-2.7437, 2.7437),
        (-1.7837, 1.7837),
        (-2.9007, 2.9007),
        (-3.0421, -0.1518),
        (-2.8065, 2.8065),
        (0.5445, 4.5169),
        (-3.0159, 3.0159),
    ]
    # EE offset: link7 -> link8 (0,0,0.107) -> hand (rot -pi/4 about Z) -> TCP (0,0,0.1034)
    ee = (make_tf(np.eye(3), [0, 0, 0.107])
          @ make_tf(rotz(-math.pi / 4), [0, 0, 0])
          @ make_tf(np.eye(3), [0, 0, 0.1034]))
    return FKChain(origins, axes, ee, limits)


def build_xarm7_chain():
    """Build FK chain for XArm7 (7-DOF arm, gripper not included in IK)."""
    hp = math.pi / 2.0
    origins = [
        make_tf(rpy_to_mat(0, 0, 0), [0, 0, 0.267]),
        make_tf(rpy_to_mat(-hp, 0, 0), [0, 0, 0]),
        make_tf(rpy_to_mat(hp, 0, 0), [0, -0.293, 0]),
        make_tf(rpy_to_mat(hp, 0, 0), [0.0525, 0, 0]),
        make_tf(rpy_to_mat(hp, 0, 0), [0.0775, -0.3425, 0]),
        make_tf(rpy_to_mat(hp, 0, 0), [0, 0, 0]),
        make_tf(rpy_to_mat(-hp, 0, 0), [0.076, 0.097, 0]),
    ]
    axes = ["Z"] * 7
    limits = [
        (-6.283, 6.283),
        (-2.059, 2.0944),
        (-6.283, 6.283),
        (-0.19198, 3.927),
        (-6.283, 6.283),
        (-1.69297, 3.14159),
        (-6.283, 6.283),
    ]
    # link7 -> link_eef (identity) -> gripper_base (identity) -> TCP (0,0,0.172)
    ee = make_tf(np.eye(3), [0, 0, 0.172])
    return FKChain(origins, axes, ee, limits)


# ─────────────────────────────────────────────────────────────
# IK Solver
# ─────────────────────────────────────────────────────────────

def ik_step(chain, q, target_pos, q_init, base_T,
            velocity_gain=1.2, null_gain=0.3,
            lambda_damp=0.01, max_ee_vel=0.2, max_joint_vel=2.0):
    """One-step damped least-squares IK with null-space posture bias."""
    ee_pos = chain.fk_position(q, base_T)
    err = target_pos - ee_pos
    ee_vel = velocity_gain * err
    speed = np.linalg.norm(ee_vel)
    if speed > max_ee_vel:
        ee_vel *= max_ee_vel / speed

    J = chain.jacobian(q, base_T)
    JJT = J @ J.T + lambda_damp * np.eye(3)
    J_inv = J.T @ np.linalg.inv(JJT)
    dq_task = J_inv @ ee_vel

    n = chain.n_joints
    N = np.eye(n) - J_inv @ J
    dq_null = N @ (null_gain * (q_init - q))

    dq = dq_task + dq_null
    max_dq = np.max(np.abs(dq))
    if max_dq > max_joint_vel:
        dq *= max_joint_vel / max_dq
    return dq


def compute_trajectory(chain, base_T, q_init, key_poses, dt=0.005,
                       n_revolute_total=None, n_prismatic_total=None,
                       gripper_rev_indices=None, gripper_width_to_angle=None):
    """
    Run IK on key_poses and return (times, joint_values) arrays.

    n_revolute_total / n_prismatic_total: total number of revolute/prismatic
    joints in the URDF (including gripper). The IK only controls the first
    chain.n_joints revolute DOFs.

    gripper_rev_indices: list of revolute joint indices that should be driven
                         by the gripper_width from key_poses (for robots with
                         revolute gripper joints like XArm7).
    gripper_width_to_angle: callable converting gripper width (m) to joint angle (rad).
    """
    if n_revolute_total is None:
        n_revolute_total = chain.n_joints
    if n_prismatic_total is None:
        n_prismatic_total = 0

    cumulative_times = np.cumsum(key_poses[:, 0])
    total_time = cumulative_times[-1]

    q = q_init.copy()
    t = 0.0
    rows = []

    while t <= total_time + 0.5:
        kp_idx = int(np.searchsorted(cumulative_times, t, side="left"))
        kp_idx = min(kp_idx, len(key_poses) - 1)

        tgt_pos = key_poses[kp_idx, 1:4]
        tgt_grip = float(key_poses[kp_idx, 8])

        dq = ik_step(chain, q, tgt_pos, q_init, base_T)
        q = chain.clamp(q + dq * dt)

        rev_vals = np.zeros(n_revolute_total)
        rev_vals[:chain.n_joints] = q

        if gripper_rev_indices is not None and gripper_width_to_angle is not None:
            grip_angle = gripper_width_to_angle(tgt_grip)
            for idx in gripper_rev_indices:
                rev_vals[idx] = grip_angle

        pris_vals = np.full(n_prismatic_total, tgt_grip)

        row = [t] + rev_vals.tolist() + pris_vals.tolist()
        rows.append(row)
        t += dt

    return np.array(rows)


def write_trajectory(path, data, revolute_names, prismatic_names):
    header = "# time " + " ".join(revolute_names + prismatic_names)
    fmt = ["%.6f"] + ["%.8f"] * (data.shape[1] - 1)
    np.savetxt(path, data, header=header, fmt=fmt, comments="")
    print(f"  Written {len(data)} keyframes to {path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--franka", action="store_true", help="Franka only")
    parser.add_argument("--xarm", action="store_true", help="XArm7 only")
    parser.add_argument("--dt", type=float, default=0.005, help="IK timestep")
    args = parser.parse_args()

    do_both = not args.franka and not args.xarm

    out_dir = SCRIPT_DIR
    os.makedirs(out_dir, exist_ok=True)

    # ── Franka Panda ──
    if args.franka or do_both:
        print("=== Franka Panda trajectory ===")
        chain = build_franka_chain()
        q_init = np.array([0.0, -0.569, 0.0, -2.110, 0.0, 3.037, 0.741])
        # No global transform scaling for Franka (1:1 real-world scale in URDF)
        base_T = np.eye(4)

        ee0 = chain.fk_position(q_init, base_T)
        print(f"  Initial EE position: ({ee0[0]:.4f}, {ee0[1]:.4f}, {ee0[2]:.4f})")

        data = compute_trajectory(
            chain, base_T, q_init, KEY_POSES, dt=args.dt,
            n_revolute_total=7, n_prismatic_total=2,
        )
        rev_names = [f"panda_joint{i}" for i in range(1, 8)]
        pris_names = ["panda_finger_joint1", "panda_finger_joint2"]
        write_trajectory(out_dir / "franka_fold.txt", data, rev_names, pris_names)

    # ── XArm7 ──
    if args.xarm or do_both:
        print("=== XArm7 trajectory ===")
        chain = build_xarm7_chain()
        q_init = np.zeros(7)

        # The XArm7 in Stiff-GIPC is scaled by 0.3, but the joint-angle IK is
        # independent of the global scale. The Cartesian targets however must be
        # expressed in the URDF's own coordinate frame (un-scaled).
        # Since the Isaac Lab key poses are in real-world metres for Franka,
        # we shift them to be reachable by XArm7's workspace (arm reach ~0.7m).
        # A simple re-centering: XArm7 base is at origin, table in front.
        xarm_key_poses = KEY_POSES.copy()
        # Shift targets so the centre is within XArm7 reach
        xarm_key_poses[:, 1] -= 0.35   # X: shift closer
        xarm_key_poses[:, 3] += 0.10   # Z: lift a bit

        base_T = np.eye(4)
        ee0 = chain.fk_position(q_init, base_T)
        print(f"  Initial EE position: ({ee0[0]:.4f}, {ee0[1]:.4f}, {ee0[2]:.4f})")

        # XArm7 gripper joints are revolute (drive_joint + 5 mimic joints).
        # 7 arm + 6 gripper = 13 revolute total, 0 prismatic.
        n_revolute_total = 13

        # Map Franka gripper_width (metres) -> XArm drive_joint angle (rad).
        # XArm gripper: at drive_joint=0.85 rad the opening is ~55mm per side.
        # Approximate linear mapping.
        def xarm_grip_width_to_angle(w):
            return np.clip(w / 0.055 * 0.85, 0.0, 0.85)

        # Indices 7-12 are the 6 gripper revolute joints (all mimic drive_joint)
        gripper_indices = list(range(7, 13))

        data = compute_trajectory(
            chain, base_T, q_init, xarm_key_poses, dt=args.dt,
            n_revolute_total=n_revolute_total,
            n_prismatic_total=0,
            gripper_rev_indices=gripper_indices,
            gripper_width_to_angle=xarm_grip_width_to_angle,
        )
        rev_names = ([f"joint{i}" for i in range(1, 8)]
                     + ["drive_joint", "left_finger_joint", "left_inner_knuckle_joint",
                        "right_outer_knuckle_joint", "right_finger_joint",
                        "right_inner_knuckle_joint"])
        write_trajectory(out_dir / "xarm7_fold.txt", data, rev_names, [])

    print("Done.")


if __name__ == "__main__":
    main()

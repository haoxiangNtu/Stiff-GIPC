#!/usr/bin/env python3
"""Regression: joint assembly must not depend on parent/child body ID order."""

import os
import sys

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from stiff_physics import Config, Engine


CUBE = os.path.join(ROOT, "Assets", "sim_data", "tetmesh", "cube.msh")
FRAMES = 12


def run(parent_body, child_body):
    engine = Engine(
        Config(
            gravity=(0.0, 0.0, 0.0),
            dt=0.01,
            ground_offset=-100.0,
            skip_all_collision=True,
            joint_strength_ratio=1e3,
        )
    )

    for x_offset in (-0.12, 0.12):
        transform = np.eye(4)
        transform[:3, :3] *= 0.25
        transform[0, 3] = x_offset
        engine.load_mesh(
            CUBE,
            dimensions=3,
            body_type="ABD",
            transform=transform,
            young_modulus=1e8,
            boundary_type="Free",
        )

    engine.add_fixed_joint(
        parent_body,
        child_body,
        world_anchor=[0.0, 0.075, 0.0],
        world_normal=[0.0, 1.0, 0.0],
        world_bitangent=[0.0, 0.0, 1.0],
    )
    engine.finalize()
    engine.native.set_body_external_force(1, 20.0, 7.0, -3.0)

    trajectory = []
    for _ in range(FRAMES):
        engine.step()
        trajectory.append(np.asarray(engine.get_vertices()).copy())
    return np.asarray(trajectory)


forward = run(0, 1)
reversed_order = run(1, 0)
max_error = float(np.max(np.abs(forward - reversed_order)))

print(f"[joint-order] max trajectory error = {max_error:.3e}")
assert np.isfinite(forward).all() and np.isfinite(reversed_order).all()
assert max_error < 1e-8, (
    "joint response depends on parent/child body ID order: "
    f"max_error={max_error:.3e}"
)

invalid_engine = Engine(Config(skip_all_collision=True))
invalid_engine.load_mesh(CUBE, dimensions=3, body_type="ABD")
for parent_body, child_body in ((0, 0), (0, 1)):
    try:
        invalid_engine.add_fixed_joint(
            parent_body,
            child_body,
            world_anchor=[0.0, 0.0, 0.0],
            world_normal=[0.0, 1.0, 0.0],
            world_bitangent=[0.0, 0.0, 1.0],
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            f"invalid joint IDs were accepted: parent={parent_body}, child={child_body}"
        )
invalid_engine.finalize()
print("PASS")

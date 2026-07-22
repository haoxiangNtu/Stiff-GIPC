#!/usr/bin/env python3
"""[B1] regression: URDF primitive collision -> conservative proxy meshes.

v0.8.4 skipped box/sphere/cylinder collision elements with a warning (the link
silently lost its collision shape).  v0.8.5 converts them to watertight,
conservatively CIRCUMSCRIBED triangle meshes (proxy contains the exact
primitive, so contact can only fire early, never be missed):

  box       -> exact 8v/12f corners
  sphere    -> icosphere subdiv 2, scaled so every face plane >= radius
  cylinder  -> 24-segment circumscribed polygon prism + cap fans
  multi-element links -> all primitives merged into ONE body, each element's
                         own <origin> baked into the vertices

Checks: body/vertex counts, containment bounds (r <= |v-c| <= r*1.06), origin
baking, and 5 stable steps with a fixed root (zero drift expected).

Run:  python3 examples/test_urdf_primitives.py
"""
import os, sys, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stiff_physics.engine import Engine, Config

URDF_XML = """<?xml version="1.0"?>
<robot name="prim_test">
  <link name="base">
    <collision>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry><box size="0.1 0.1 0.1"/></geometry>
    </collision>
  </link>
  <link name="arm">
    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry><cylinder radius="0.02" length="0.2"/></geometry>
    </collision>
  </link>
  <link name="tip">
    <collision>
      <geometry><sphere radius="0.03"/></geometry>
    </collision>
  </link>
  <link name="multi">
    <collision>
      <origin xyz="-0.05 0 0" rpy="0 0 0"/>
      <geometry><box size="0.04 0.04 0.04"/></geometry>
    </collision>
    <collision>
      <origin xyz="0.05 0 0" rpy="0 0 0"/>
      <geometry><sphere radius="0.02"/></geometry>
    </collision>
  </link>
  <joint name="j1" type="revolute">
    <parent link="base"/>
    <child link="arm"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1"/>
  </joint>
  <joint name="j2" type="fixed">
    <parent link="arm"/>
    <child link="tip"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
  </joint>
  <joint name="j3" type="fixed">
    <parent link="tip"/>
    <child link="multi"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
  </joint>
</robot>
"""


def main():
    tmpdir = tempfile.mkdtemp(prefix="stiffgipc_prim_urdf_")
    urdf = os.path.join(tmpdir, "prim_test.urdf")
    with open(urdf, "w") as f:
        f.write(URDF_XML)

    eng = Engine(Config(ground_offset=0.0))
    eng.load_urdf(urdf, translation=(0.0, 0.6, 0.0), root_fixed=True)
    eng.finalize()

    raw = eng._engine
    n_bodies = raw.get_abd_body_count()
    n = raw.get_vertex_count_host()
    V = raw.get_vertices_host()
    print(f"abd_bodies={n_bodies} num_vertices={n}")

    assert n_bodies == 4, f"body count {n_bodies} != 4 (multi-element link must merge)"
    EXPECT_V = 8 + 50 + 162 + (8 + 162)  # box + cyl(24seg) + icosphere(sub2) + merged
    assert n == EXPECT_V, f"vertex count {n} != expected {EXPECT_V}"

    ty = 0.6  # world y translation

    # base box: exact corners around origin (0,0,0.05), half-extent 0.05
    box = V[0:8]
    assert np.allclose(np.abs(box[:, 0]), 0.05, atol=1e-12), "box x extent wrong"
    assert np.allclose(np.abs(box[:, 1] - ty), 0.05, atol=1e-12), "box y extent wrong"
    assert np.allclose(np.sort(np.unique(np.round(box[:, 2], 12))), [0.0, 0.1]), "box z extent wrong"

    # arm cylinder: joint z=0.1 + collision origin z=0.1 -> center z=0.2, r=0.02
    cyl = V[8:58]
    rad = np.sqrt(cyl[:48, 0] ** 2 + (cyl[:48, 1] - ty) ** 2)  # 48 ring verts
    assert rad.min() >= 0.02 - 1e-12, f"cylinder under-approximates: {rad.min()}"
    assert rad.max() <= 0.02 * 1.02, f"cylinder over-inflated: {rad.max()}"
    assert abs(cyl[:, 2].min() - 0.1) < 1e-9 and abs(cyl[:, 2].max() - 0.3) < 1e-9

    # tip sphere: center z=0.3, r=0.03 -> conservative band [r, 1.06r]
    sph = V[58:220]
    d = np.linalg.norm(sph - np.array([0.0, ty, 0.3]), axis=1)
    assert d.min() >= 0.03 - 1e-12, f"sphere under-approximates: {d.min()}"
    assert d.max() <= 0.03 * 1.06, f"sphere over-inflated: {d.max()}"

    # multi link (center z=0.35): per-element origins baked into vertices
    multi = V[220:390]
    mbox, msph = multi[:8], multi[8:]
    assert np.allclose(np.abs(mbox[:, 0] + 0.05), 0.02, atol=1e-9), "multi box origin not baked"
    dm = np.linalg.norm(msph - np.array([0.05, ty, 0.35]), axis=1)
    assert dm.min() >= 0.02 - 1e-12 and dm.max() <= 0.02 * 1.06, "multi sphere origin/size wrong"
    print("geometry containment checks PASS")

    # fixed-root robot must stay put through a few solves
    for _ in range(5):
        eng.step()
    drift = np.abs(raw.get_vertices_host() - V).max()
    print(f"5-step max drift = {drift:.3e} m")
    assert drift < 5e-3, f"unexpected drift {drift}"

    print("B1 URDF primitive proxy: ALL PASS")


if __name__ == "__main__":
    main()

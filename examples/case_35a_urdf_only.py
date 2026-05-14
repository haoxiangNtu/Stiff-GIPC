#!/usr/bin/env python3
"""case_35a_urdf_only.py — ridgeback dual-panda URDF, no gripper, no
hybrid mesh.  Just GUI joint sliders to verify arm motion works cleanly
before integrating the hybrid gripper (case_35).
"""
import sys, os, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _use_dailyv2_engine  # noqa: F401

import numpy as np
from scipy.spatial.transform import Rotation
import polyscope as ps
import polyscope.imgui as psim

from stiff_physics import Engine, Config


URDF_PATH = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/urdf/ridgeback_dual_panda_soft/ridgeback_softgripper_no_finger.urdf"
ARM_SCALE = 0.3


def make_arm_tf(scale: float) -> np.ndarray:
    tf = np.eye(4)
    tf[:3, :3] = scale * Rotation.from_rotvec([-math.pi/2, 0, 0]).as_matrix()
    tf[0, 3] = 0.0
    tf[1, 3] = -0.9
    tf[2, 3] = 0.0
    return tf


def main():
    cfg = Config(
        dt=0.020,
        cloth_thickness=1e-3, cloth_young_modulus=1e4, bend_young_modulus=1e3,
        cloth_density=200, strain_rate=100, soft_motion_rate=1.0,
        poisson_rate=0.49, friction_rate=0.4, relative_dhat=1e-3,
        joint_strength_ratio=100.0, revolute_driving_strength_ratio=100.0,
        semi_implicit_enabled=True, semi_implicit_beta_tol=5e-2,
        semi_implicit_min_iter=1, newton_tol=5e-2,
        preconditioner_type=0, ground_offset=-0.5,
        assets_dir="/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/",
    )
    cfg._cfg.collision_detection_buff_scale = 64.0

    eng = Engine(cfg)
    print(f"\n[case35a] === ridgeback URDF only (no gripper) ===", flush=True)

    arm_tf = make_arm_tf(ARM_SCALE)
    eng.native.load_urdf(URDF_PATH, arm_tf, True, False, 1e7, {})
    n_urdf = eng.abd_body_count
    print(f"[case35a] URDF: {n_urdf} ABD bodies", flush=True)
    for b in range(n_urdf):
        eng.add_ground_collision_skip(b)

    eng.finalize()

    # Identify left_arm joints
    panda_joints = []
    for i in range(eng.native.get_num_revolute_joints()):
        info = eng.native.get_revolute_joint_info(i)
        if info.name.startswith("left_arm_joint") and info.name != "left_arm_joint8":
            panda_joints.append((info.name, i, info.lower_limit, info.upper_limit))
    print(f"[case35a] {len(panda_joints)} left-panda joints", flush=True)
    print(f"[case35a] finalized\n", flush=True)

    auto_n = int(os.environ.get("AUTO_STEP", "0"))
    if auto_n > 0:
        for i in range(auto_n):
            t0 = time.perf_counter()
            eng.step()
            print(f"[case35a] step {i}: {(time.perf_counter()-t0)*1000:.1f} ms",
                  flush=True)
        return

    # GUI
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")

    verts_world = eng.get_vertices()
    all_faces = eng.get_surface_faces()
    recs = eng.get_load_records()
    hand_id = next(r.body_offset for r in recs if r.label == "left_arm_hand")

    body_meshes = []
    for r in recs:
        v_off = r.vertex_offset
        v_end = v_off + r.vertex_count
        face_mask = np.all((all_faces >= v_off) & (all_faces < v_end), axis=1)
        if not face_mask.any():
            continue
        faces_local = all_faces[face_mask] - v_off
        name = (r.label or f"body{r.body_offset}").replace(" ", "_")[:32]
        if r.body_offset == hand_id:
            color = (0.3, 0.5, 0.85)         # blue — left_arm_hand
        elif "left_arm" in (r.label or ""):
            color = (0.6, 0.7, 0.85)         # light blue — left arm
        elif "right_arm" in (r.label or ""):
            color = (0.85, 0.7, 0.6)         # tan — right arm
        else:
            color = (0.55, 0.55, 0.6)        # grey — ridgeback body
        m = ps.register_surface_mesh(name, verts_world[v_off:v_end], faces_local,
                                      smooth_shade=True)
        m.set_color(color)
        body_meshes.append((m, v_off, v_end))

    state = dict(
        running=False, step_count=0, last_step_ms=0.0,
        slider_deg={nm: 0.0 for nm,_,_,_ in panda_joints},
        driven_deg={nm: 0.0 for nm,_,_,_ in panda_joints},
        strength=1.0,
    )
    MAX_DEG_PER_STEP = float(os.environ.get("CASE35_MAX_DEG", "0.5"))

    def do_step():
        for nm, idx, lo, hi in panda_joints:
            target = state['slider_deg'][nm]
            current = state['driven_deg'][nm]
            delta = target - current
            if abs(delta) > MAX_DEG_PER_STEP:
                current += MAX_DEG_PER_STEP * np.sign(delta)
            else:
                current = target
            state['driven_deg'][nm] = current
            eng.native.set_revolute_target(idx, math.radians(current))
            eng.native.set_revolute_strength(idx, state['strength'])
        t0 = time.perf_counter()
        eng.step()
        state['last_step_ms'] = (time.perf_counter() - t0) * 1000.0
        state['step_count'] += 1
        cur = eng.get_vertices()
        for m, v0, v1 in body_meshes:
            m.update_vertex_positions(cur[v0:v1])

    def callback():
        psim.SetNextWindowPos((10, 10), psim.ImGuiCond_Once)
        psim.SetNextWindowSize((480, 0), psim.ImGuiCond_Once)
        psim.Begin("case_35a — ridgeback URDF only")
        psim.Text(f"step #{state['step_count']}: {state['last_step_ms']:.1f} ms")
        psim.Text(f"URDF: {n_urdf} ABD bodies, no hybrid gripper")
        psim.Separator()
        if state['running']:
            if psim.Button("Pause"): state['running'] = False
        else:
            if psim.Button("Run"): state['running'] = True
        psim.SameLine()
        if psim.Button("Step"): do_step()
        psim.SameLine()
        if psim.Button("Reset all"):
            for k in state['slider_deg']:
                state['slider_deg'][k] = 0.0

        chg, val = psim.SliderFloat("joint stiffness##s", state['strength'],
                                    v_min=0.1, v_max=10.0)
        if chg: state['strength'] = val

        psim.Separator()
        psim.Text(f"Panda joint angles (deg) — rate {MAX_DEG_PER_STEP}°/step:")
        for nm, idx, lo, hi in panda_joints:
            short = nm.replace("left_arm_", "")
            chg, val = psim.SliderFloat(
                f"{short}##{idx}", state['slider_deg'][nm],
                v_min=math.degrees(lo), v_max=math.degrees(hi))
            if chg:
                state['slider_deg'][nm] = val
        psim.End()

        chasing = any(abs(state['slider_deg'][n] - state['driven_deg'][n]) > 1e-3
                      for n,_,_,_ in panda_joints)
        if state['running'] or chasing:
            do_step()

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""case_30_gripper_with_cubes.py — case_29 + 1 ABD cube + 1 FEM cube.

Same Strategy D unified mesh as gripper (Fixed-vert + set_vertex_positions_gpu
to drive rigid region).  Adds:
  - 1 ABD cube (Free, falls under gravity)
  - 1 FEM cube (Free, falls under gravity, soft)

Both cubes start above ground.  They fall and rest on ground.
User drags gripper via sliders to push / squeeze / grab them.

Run:
    cd /home/ps/Downloads/Stiff-GIPC-hybrid-mesh
    ./run examples/case_30_gripper_with_cubes.py
"""
import sys, os, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _use_dailyv2_engine  # noqa: F401

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from stiff_physics import Engine, Config


UNIFIED_NPZ = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/hybrid_d/STRATEGY_F_unified.npz"
CUBE_MSH    = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/tetmesh/cube.msh"


def main():
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("tile_reflection")

    cfg = Config(
        dt=0.020,
        cloth_thickness=1e-3, cloth_young_modulus=1e4, bend_young_modulus=1e3,
        cloth_density=200, strain_rate=100, soft_motion_rate=1e4,
        poisson_rate=0.49, friction_rate=0.4, relative_dhat=1e-4,
        joint_strength_ratio=100.0, revolute_driving_strength_ratio=100.0,
        semi_implicit_enabled=True, semi_implicit_beta_tol=5e-2,
        semi_implicit_min_iter=1, newton_tol=5e-2,
        preconditioner_type=0, ground_offset=-0.1,
        assets_dir="/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/",
    )
    cfg._cfg.collision_detection_buff_scale = 8.0

    eng = Engine(cfg)
    print(f"\n[case30] === gripper + ABD cube + FEM cube ===", flush=True)

    # ---- 1. ABD cube — Free, ~3cm at (-0.05, -0.3, 0.05) — left of gripper, above ground
    abd_T = np.eye(4)
    abd_T[:3, :3] *= 0.03  # 3cm cube
    abd_T[3, 3] = 1.0  # last diagonal stays 1 (homogeneous)
    abd_T[:3, 3] = [-0.05, 0.0, 0.05]
    eng.load_mesh(CUBE_MSH, dimensions=3, body_type="ABD",
                  transform=abd_T, young_modulus=1e8)
    abd_cube_rec = eng.get_load_records()[-1]
    print(f"[case30] ABD cube body={abd_cube_rec.body_offset} verts={abd_cube_rec.vertex_count}",
          flush=True)

    # ---- 2. FEM cube — soft Young, falls and squishes
    fem_cube_T = np.eye(4)
    fem_cube_T[:3, :3] *= 0.03  # 3cm cube
    fem_cube_T[3, 3] = 1.0
    fem_cube_T[:3, 3] = [0.10, 0.0, 0.05]   # right of gripper
    eng.load_mesh(CUBE_MSH, dimensions=3, body_type="FEM",
                  transform=fem_cube_T, young_modulus=5e6)
    fem_cube_rec = eng.get_load_records()[-1]
    print(f"[case30] FEM cube body={fem_cube_rec.body_offset} verts={fem_cube_rec.vertex_count}",
          flush=True)

    # ---- 3. Gripper unified mesh as FEM body (case_29 style)
    data = np.load(UNIFIED_NPZ)
    verts = np.ascontiguousarray(data['vertices'], dtype=np.float64)
    tets = np.ascontiguousarray(data['tets'], dtype=np.int32)
    vertex_region = data['vertex_region']
    n_rigid_v = int((vertex_region == 1).sum())
    n_fem_v = int((vertex_region == 0).sum())
    fem_young = float(os.environ.get("HYBRID_D_YOUNG", "1e8"))
    print(f"[case30] gripper mesh V={len(verts)} T={len(tets)}, "
          f"rigid_v={n_rigid_v}, FEM_v={n_fem_v}, Young={fem_young:.1e}",
          flush=True)

    # Place gripper above ground (at Y=0 by default; offset further up so it
    # doesn't immediately hit cubes).
    gripper_T = np.eye(4)
    gripper_T[:3, 3] = [0.025, -0.05, 0.025]  # gripper centered above cubes area
    verts_world_init = verts + gripper_T[:3, 3]  # apply transform manually
    eng.native.load_mesh_from_data(np.ascontiguousarray(verts_world_init, dtype=np.float64),
                                   tets, 4, 3, 1, np.eye(4), fem_young, 0)
    gripper_rec = eng.get_load_records()[-1]
    gripper_v_offset = gripper_rec.vertex_offset
    print(f"[case30] gripper body={gripper_rec.body_offset} verts={gripper_rec.vertex_count}",
          flush=True)

    # Mark gripper rigid verts as Fixed (Dirichlet — case_29 fix)
    rigid_local_idx = np.nonzero(vertex_region == 1)[0]
    for i in rigid_local_idx:
        eng.native.set_vertex_boundary(int(gripper_v_offset + i), 1)
    print(f"[case30] {len(rigid_local_idx)} gripper rigid verts → Fixed", flush=True)

    eng.finalize()
    print(f"[case30] finalized\n", flush=True)

    rest_world = eng.get_vertices().copy()
    rigid_global_idx = gripper_v_offset + rigid_local_idx
    rest_rigid = rest_world[rigid_global_idx].copy()

    auto_n = int(os.environ.get("AUTO_STEP", "0"))
    if auto_n > 0:
        for i in range(auto_n):
            t0 = time.perf_counter()
            eng.step()
            print(f"[case30] step {i}: {(time.perf_counter()-t0)*1000:.1f} ms",
                  flush=True)
        return

    # GUI
    verts_world = eng.get_vertices()
    faces_surf = eng.get_surface_faces()
    mesh = ps.register_surface_mesh("scene", verts_world, faces_surf,
                                    smooth_shade=True)
    mesh.set_color((0.7, 0.75, 0.85))
    region_marker = np.zeros(len(verts_world), dtype=np.float32)
    region_marker[rigid_global_idx] = 1.0
    mesh.add_scalar_quantity("rigid (red) — drag this", region_marker,
                             enabled=True, cmap='reds')

    state = dict(running=False, step_count=0,
                 x=0.0, y=0.0, z=0.0,
                 dx=0.0, dy=0.0, dz=0.0,
                 last_step_ms=0.0)

    MAX_DRIVE_PER_STEP_CM = float(os.environ.get("HYBRID_D_MAX_DRIVE_CM", "0.05"))
    SLIDER_MAX_CM = float(os.environ.get("HYBRID_D_SLIDER_MAX_CM", "30.0"))

    def drive_rigid():
        for axis in ('x', 'y', 'z'):
            target, current = state[axis], state['d' + axis]
            delta = target - current
            if abs(delta) > MAX_DRIVE_PER_STEP_CM:
                current += MAX_DRIVE_PER_STEP_CM * np.sign(delta)
            else:
                current = target
            state['d' + axis] = current
        cur_all = eng.get_vertices().copy()
        offset = np.array([state['dx']*0.01, state['dy']*0.01, state['dz']*0.01])
        cur_all[rigid_global_idx] = rest_rigid + offset
        eng.native.set_vertex_positions_gpu(np.ascontiguousarray(cur_all))

    def do_step():
        drive_rigid()
        t0 = time.perf_counter()
        eng.step()
        state['last_step_ms'] = (time.perf_counter() - t0) * 1000.0
        state['step_count'] += 1
        mesh.update_vertex_positions(eng.get_vertices())

    def callback():
        psim.SetNextWindowPos((10, 10), psim.ImGuiCond_Once)
        psim.SetNextWindowSize((400, 0), psim.ImGuiCond_Once)
        psim.Begin("case_30 — gripper + cubes (drag rigid red region)")
        psim.Text(f"ABD cube V={abd_cube_rec.vertex_count}, "
                  f"FEM cube V={fem_cube_rec.vertex_count}")
        psim.Text(f"Gripper V={gripper_rec.vertex_count} rigid={n_rigid_v}")
        psim.Text(f"step #{state['step_count']}: {state['last_step_ms']:.1f} ms")
        psim.Separator()

        if state['running']:
            if psim.Button("Pause"): state['running'] = False
        else:
            if psim.Button("Run (gravity)"): state['running'] = True
        psim.SameLine()
        if psim.Button("Step Once"): do_step()
        psim.SameLine()
        if psim.Button("Reset"):
            state['x'] = state['y'] = state['z'] = 0.0
            state['dx'] = state['dy'] = state['dz'] = 0.0
            drive_rigid()
            mesh.update_vertex_positions(eng.get_vertices())

        psim.Separator()
        psim.Text(f"Drag gripper rigid (cm) — rate {MAX_DRIVE_PER_STEP_CM}cm/step:")
        chg, val = psim.SliderFloat("X##cx", state['x'],
                                    v_min=-SLIDER_MAX_CM, v_max=SLIDER_MAX_CM)
        if chg: state['x'] = val
        chg, val = psim.SliderFloat("Y##cy", state['y'],
                                    v_min=-SLIDER_MAX_CM, v_max=SLIDER_MAX_CM)
        if chg: state['y'] = val
        chg, val = psim.SliderFloat("Z##cz", state['z'],
                                    v_min=-SLIDER_MAX_CM, v_max=SLIDER_MAX_CM)
        if chg: state['z'] = val
        psim.Text(f"  driven: x={state['dx']:+.2f} y={state['dy']:+.2f} z={state['dz']:+.2f}")

        psim.End()

        # Auto-step when slider differs from driven OR Run is on
        slider_changed = (abs(state['x']-state['dx']) > 1e-4 or
                          abs(state['y']-state['dy']) > 1e-4 or
                          abs(state['z']-state['dz']) > 1e-4)
        if state['running'] or slider_changed:
            do_step()

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""case_33_arm_cloth_gravity.py — arm + cloth + gravity comparison.

Tests user's hypothesis: disabling gravity on a kinematically driven ABD
arm should make cloth interaction smoother (fewer Newton iters, fewer
line search hangs, no drift accumulation).

Scene (intentionally minimal — no hybrid gripper, no URDF — to isolate
the gravity variable):
    arm_base (Fixed ABD cube, 4cm)  ── above scene
            |
    [revolute joint, Z axis, anchor at midpoint]
            |
    arm_link (Free ABD cube, 4cm)  ── pendulum, swings sideways via slider
            (poking down at cloth below)

    cloth (FEM 2D shell, 25×25)  ── horizontal patch, 4 corners pinned

Run modes:
    GUI (default):
        ./run examples/case_33_arm_cloth_gravity.py
        - drag joint slider, toggle "arm gravity" button, watch ms/step

    Headless A/B benchmark:
        AUTO_BENCH=1 ./run examples/case_33_arm_cloth_gravity.py
        - runs identical motion script with gravity ON vs OFF
        - reports: ms/step, Newton iter/step, line search hang count
"""
import sys, os, math, time, io, contextlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _use_dailyv2_engine  # noqa: F401

import numpy as np

from stiff_physics import Engine, Config

CUBE_MSH = "/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/sim_data/tetmesh/cube.msh"


def make_cloth_mesh_vertical(nx=25, ny=25, size=0.30, center=(0.0, 0.0, 0.05)):
    """Generate a flat vertical cloth patch in XY plane at Z=center[2].
    Returns (verts (N,3), tris (M,3)).
    Top edge (y=max) typically pinned.
    """
    cx, cy, cz = center
    xs = np.linspace(-size/2, size/2, nx) + cx
    ys = np.linspace(-size/2, size/2, ny) + cy
    verts = np.zeros((nx*ny, 3), dtype=np.float64)
    for j in range(ny):
        for i in range(nx):
            verts[j*nx + i] = [xs[i], ys[j], cz]
    tris = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            v0 = j*nx + i
            v1 = j*nx + (i+1)
            v2 = (j+1)*nx + i
            v3 = (j+1)*nx + (i+1)
            tris.append([v0, v1, v3])
            tris.append([v0, v3, v2])
    return verts, np.array(tris, dtype=np.int32)


def setup_scene(disable_arm_gravity: bool, verbose: bool = True,
                semi_implicit: bool = True):
    cfg = Config(
        dt=0.020,
        cloth_thickness=1e-3, cloth_young_modulus=5e3, bend_young_modulus=5e2,
        cloth_density=200, strain_rate=100, soft_motion_rate=1e4,
        poisson_rate=0.49, friction_rate=0.4, relative_dhat=1e-4,
        joint_strength_ratio=200.0, revolute_driving_strength_ratio=300.0,
        semi_implicit_enabled=semi_implicit, semi_implicit_beta_tol=5e-2,
        semi_implicit_min_iter=1, newton_tol=5e-2,
        preconditioner_type=0, ground_offset=-0.5,
        assets_dir="/home/ps/Downloads/Stiff-GIPC-hybrid-mesh/Assets/",
    )
    cfg._cfg.collision_detection_buff_scale = 8.0

    eng = Engine(cfg)

    # ---- arm_base: Fixed ABD cube, 4cm at (0, 0.05, 0) ----
    base_T = np.eye(4); base_T[:3, :3] *= 0.04; base_T[3, 3] = 1.0
    base_T[:3, 3] = [0.0, 0.05, 0.0]
    eng.load_mesh(CUBE_MSH, dimensions=3, body_type="ABD",
                  transform=base_T, young_modulus=1e8, boundary_type="Fixed")
    base_id = eng.get_load_records()[-1].body_offset       # global = 0

    # ---- arm_link: Free ABD cube at (0.05, -0.05, 0), OFF-AXIS for Y rotation
    # Y-axis revolute → arm sweeps in XZ plane.  arm_link offset (0.05, -0.05, 0)
    # from anchor (0,0,0) gives moment arm in X.  At θ=-90°, arm is at
    # (0, -0.05, 0.05) → cube extends Z=[0.03, 0.07], hits cloth at Z=0.05.
    link_T = np.eye(4); link_T[:3, :3] *= 0.04; link_T[3, 3] = 1.0
    link_T[:3, 3] = [0.05, -0.05, 0.0]
    eng.load_mesh(CUBE_MSH, dimensions=3, body_type="ABD",
                  transform=link_T, young_modulus=1e8, boundary_type="Free")
    link_id = eng.get_load_records()[-1].body_offset       # global = 1

    # ---- revolute joint about Y axis at origin ----
    # θ=0: arm at +X.  θ→-90°: arm sweeps to +Z (toward cloth).
    joint_idx = eng.native.add_revolute_joint(
        parent_body=base_id, child_body=link_id,
        world_axis=np.array([0.0, 1.0, 0.0]),
        joint_pos=np.array([0.0, 0.0, 0.0]),
        lower_limit=-math.radians(120), upper_limit=math.radians(120),
        initial_angle=0.0, name="base_to_link",
    )

    # ---- cloth: vertical sheet in XY plane at Z = 0.05 ----
    # Top edge (y=max) Fixed (hanging); arm sweeps from +X side toward +Z
    # and contacts the cloth around θ ∈ [-30°, -100°].
    cloth_verts, cloth_tris = make_cloth_mesh_vertical(nx=25, ny=25, size=0.30,
                                                        center=(0.0, 0.0, 0.05))
    eng.native.load_mesh_from_data(cloth_verts, cloth_tris, 3, 2, 1,
                                    np.eye(4), 5e3, 0)
    cloth_rec = eng.get_load_records()[-1]
    cloth_v_offset = cloth_rec.vertex_offset
    n_abd = sum(1 for r in eng.get_load_records() if r.body_type == 0)
    cloth_global = n_abd + cloth_rec.body_offset           # global = 2

    # Pin top edge of vertical cloth (j=ny-1, all i) so it hangs.
    nx, ny = 25, 25
    pinned_local = [(ny-1)*nx + i for i in range(nx)]
    for c in pinned_local:
        eng.native.set_vertex_boundary(int(cloth_v_offset + c), 1)  # Fixed

    # ---- collision exclusions: arm_base shouldn't touch link or cloth ----
    eng.native.add_collision_exclusion(base_id, link_id)
    eng.native.add_collision_exclusion(base_id, cloth_global)
    # link ↔ cloth: keep collision ON (this is the contact we're testing)

    eng.finalize()

    if disable_arm_gravity:
        eng.native.set_body_apply_gravity(link_id, False)
        if verbose:
            print(f"[case33] disabled gravity on arm_link (body {link_id})", flush=True)

    eng.native.set_revolute_target(joint_idx, 0.0)
    eng.native.set_revolute_strength(joint_idx, 1.0)

    return dict(
        eng=eng,
        base_id=base_id, link_id=link_id, joint_idx=joint_idx,
        cloth_global=cloth_global, cloth_v_offset=cloth_v_offset,
        n_cloth=cloth_verts.shape[0], n_cloth_tris=cloth_tris.shape[0],
        cloth_pinned=[cloth_v_offset + c for c in pinned_local],
    )


# ============================================================
# Headless A/B benchmark
# ============================================================

class _FdCapture:
    """Redirect file descriptor 1 (real stdout, including C printf) to a
    pipe.  Use `with _FdCapture() as cap: ...` then `cap.text` to read."""
    def __enter__(self):
        import os as _os
        self._saved_fd = _os.dup(1)
        self._read_fd, self._write_fd = _os.pipe()
        _os.dup2(self._write_fd, 1)
        _os.close(self._write_fd)
        return self

    def __exit__(self, *exc):
        import os as _os, sys as _sys
        _sys.stdout.flush()
        _os.dup2(self._saved_fd, 1)
        _os.close(self._saved_fd)
        # drain the pipe
        chunks = []
        try:
            import fcntl, errno
            fcntl.fcntl(self._read_fd, fcntl.F_SETFL,
                        fcntl.fcntl(self._read_fd, fcntl.F_GETFL) | _os.O_NONBLOCK)
            while True:
                try:
                    data = _os.read(self._read_fd, 65536)
                    if not data: break
                    chunks.append(data.decode('utf-8', 'replace'))
                except (OSError, BlockingIOError):
                    break
        finally:
            _os.close(self._read_fd)
        self.text = ''.join(chunks)


def benchmark_motion_script(scene, n_warmup=10, n_steps=200,
                             ramp_target_deg=-60.0, rate_deg_per_step=0.5,
                             return_path=True):
    """Scripted joint motion + per-step metrics.

    Uses fd-level stdout redirect to capture engine's C printf lines,
    parses `iteration k: N` (Newton iter count) and `lineSearchCount=9`
    (line search saturation indicator).
    """
    eng = scene['eng']
    joint_idx = scene['joint_idx']

    for _ in range(n_warmup):
        eng.step()

    half = n_steps // 2
    angles = []
    driven = 0.0
    for s in range(half):
        delta = ramp_target_deg - driven
        if abs(delta) > rate_deg_per_step:
            driven += rate_deg_per_step * np.sign(delta)
        else:
            driven = ramp_target_deg
        angles.append(driven)
    if return_path:
        for s in range(n_steps - half):
            delta = 0.0 - driven
            if abs(delta) > rate_deg_per_step:
                driven += rate_deg_per_step * np.sign(delta)
            else:
                driven = 0.0
            angles.append(driven)
    else:
        for _ in range(n_steps - half):
            angles.append(driven)

    step_times_ms = []
    newton_iters = []
    ls_hangs = []

    for s in range(len(angles)):
        eng.native.set_revolute_target(joint_idx, math.radians(angles[s]))
        with _FdCapture() as cap:
            t0 = time.perf_counter()
            eng.step()
            elapsed_ms = (time.perf_counter() - t0) * 1000
        so = cap.text
        step_times_ms.append(elapsed_ms)
        # Parse "iteration k:  NN" — engine prints once per step at end
        max_iter = 0
        for line in so.split('\n'):
            if 'iteration k:' in line:
                try:
                    n = int(line.split('iteration k:')[1].strip().split()[0])
                    max_iter = max(max_iter, n)
                except Exception: pass
            if 'Newton iter' in line:
                try:
                    n = int(line.split('Newton iter')[1].strip().split()[0])
                    max_iter = max(max_iter, n)
                except Exception: pass
        newton_iters.append(max_iter)
        ls_hangs.append(so.count('lineSearchCount=9'))

    return dict(
        step_times_ms=np.array(step_times_ms),
        newton_iters=np.array(newton_iters),
        ls_hangs=np.array(ls_hangs),
        angles=np.array(angles),
    )


def run_benchmark():
    """Headless A/B: run identical script with gravity ON vs OFF, report.

    BENCH_MODE=full → disable semi-implicit (forces full Newton; biggest
                      gravity effect, but slow)
    BENCH_MODE=semi (default) → semi-implicit on (production setting; smaller
                                 effect, but realistic)
    BENCH_RAMP=-60 → swing angle target (default -60°, ensures cloth contact)
    """
    semi = os.environ.get("BENCH_MODE", "semi") == "semi"
    ramp = float(os.environ.get("BENCH_RAMP", "-60.0"))
    n_steps = int(os.environ.get("BENCH_STEPS", "240"))

    print("\n" + "="*70, flush=True)
    print(f"CASE_33 BENCHMARK: arm gravity ON vs OFF "
          f"(semi_implicit={semi}, ramp={ramp}°, steps={n_steps})",
          flush=True)
    print("="*70, flush=True)

    results = {}
    for label, gravity_off in [("gravity_ON", False), ("gravity_OFF", True)]:
        print(f"\n[bench] === {label} ===", flush=True)
        scene = setup_scene(disable_arm_gravity=gravity_off, verbose=False,
                             semi_implicit=semi)
        # Suppress engine stdout during setup to keep output clean
        cap = io.StringIO()
        with contextlib.redirect_stdout(cap):
            r = benchmark_motion_script(scene, n_warmup=5, n_steps=n_steps,
                                        ramp_target_deg=ramp,
                                        rate_deg_per_step=0.3)
        results[label] = r

        st = r['step_times_ms']
        ni = r['newton_iters']
        lh = r['ls_hangs']
        print(f"   step time:      mean={st.mean():.2f}  median={np.median(st):.2f}  "
              f"p95={np.percentile(st,95):.2f}  max={st.max():.2f}  ms",
              flush=True)
        print(f"   Newton iter:    mean={ni.mean():.1f}  median={np.median(ni):.1f}  "
              f"p95={np.percentile(ni,95):.1f}  max={ni.max()}",
              flush=True)
        print(f"   line search 9:  total={lh.sum()}  steps_with_hang={(lh>0).sum()}/{len(lh)}",
              flush=True)

    print("\n" + "="*70, flush=True)
    print("COMPARISON (OFF vs ON)", flush=True)
    print("="*70, flush=True)
    on, off = results["gravity_ON"], results["gravity_OFF"]
    def pct(a, b):
        return (a - b) / max(b, 1e-9) * 100
    metric_pairs = [
        ("step time mean",      on['step_times_ms'].mean(),      off['step_times_ms'].mean(),      "ms"),
        ("step time p95",       np.percentile(on['step_times_ms'], 95), np.percentile(off['step_times_ms'], 95), "ms"),
        ("step time max",       on['step_times_ms'].max(),        off['step_times_ms'].max(),        "ms"),
        ("Newton iter mean",    on['newton_iters'].mean(),         off['newton_iters'].mean(),         ""),
        ("Newton iter p95",     np.percentile(on['newton_iters'], 95), np.percentile(off['newton_iters'], 95), ""),
        ("Newton iter max",     on['newton_iters'].max(),          off['newton_iters'].max(),          ""),
        ("line search 9 total", on['ls_hangs'].sum(),              off['ls_hangs'].sum(),              ""),
    ]
    for name, a, b, unit in metric_pairs:
        delta = pct(b, a)  # negative = OFF improved (smaller)
        sign = "↓" if delta < 0 else ("↑" if delta > 0 else "=")
        print(f"  {name:<22} ON={a:>8.2f}{unit:<3}  OFF={b:>8.2f}{unit:<3}  Δ={delta:+6.1f}% {sign}",
              flush=True)

    print("\nInterpretation:", flush=True)
    print("  Δ < 0 = OFF is better (smaller).  Bigger negative = bigger smoothness gain.", flush=True)
    print("  Look at p95/max — these capture the worst-case stalls user feels in GUI.", flush=True)


# ============================================================
# Interactive GUI
# ============================================================

def run_gui():
    import polyscope as ps
    import polyscope.imgui as psim

    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("tile_reflection")

    # Default: gravity OFF (per case_32 lesson; user can toggle)
    state = dict(
        gravity_off=True,
        scene=None,
        running=False, step_count=0,
        angle_deg=0.0, driven_deg=0.0,
        last_step_ms=0.0,
        last_newton_iters=0, last_ls_hangs=0,
        cumulative_ls_hangs=0,
    )

    def rebuild_scene():
        state['scene'] = setup_scene(disable_arm_gravity=state['gravity_off'])
        state['step_count'] = 0
        state['angle_deg'] = 0.0
        state['driven_deg'] = 0.0
        state['cumulative_ls_hangs'] = 0
        # Render meshes
        scene = state['scene']
        eng = scene['eng']
        verts_world = eng.get_vertices()
        all_faces = eng.get_surface_faces()

        # Filter cloth faces (cloth body global id = scene['cloth_global'])
        # Easier to query by FEM body's vertex range
        cloth_v_off = scene['cloth_v_offset']
        cloth_v_end = cloth_v_off + scene['n_cloth']
        cloth_face_mask = np.all((all_faces >= cloth_v_off) & (all_faces < cloth_v_end), axis=1)
        cloth_faces_local = all_faces[cloth_face_mask] - cloth_v_off
        cloth_verts_world = verts_world[cloth_v_off:cloth_v_end]

        cloth_mesh = ps.register_surface_mesh("cloth", cloth_verts_world,
                                               cloth_faces_local,
                                               smooth_shade=True)
        cloth_mesh.set_color((0.85, 0.7, 0.5))

        # Arm bodies (cubes)
        for body_idx, color, name in [(0, (0.4,0.4,0.5), 'arm_base'),
                                       (1, (0.6,0.7,0.9), 'arm_link')]:
            # rough: take all faces with verts in body's range (scan point_id_to_body_id)
            # Simpler: since cubes are loaded first sequentially, use vertex_offset
            recs = eng.get_load_records()
            v_off = recs[body_idx].vertex_offset
            v_end = v_off + recs[body_idx].vertex_count
            face_mask = np.all((all_faces >= v_off) & (all_faces < v_end), axis=1)
            faces_local = all_faces[face_mask] - v_off
            cube_mesh = ps.register_surface_mesh(name, verts_world[v_off:v_end],
                                                  faces_local, smooth_shade=False)
            cube_mesh.set_color(color)

        state['cloth_mesh'] = cloth_mesh
        state['cloth_v_off'] = cloth_v_off
        state['cloth_v_end'] = cloth_v_end
        state['arm_meshes'] = []
        for name in ('arm_base', 'arm_link'):
            state['arm_meshes'].append((name, ps.get_surface_mesh(name)))
        state['arm_recs'] = [(eng.get_load_records()[i].vertex_offset,
                              eng.get_load_records()[i].vertex_offset + eng.get_load_records()[i].vertex_count)
                             for i in range(2)]
        print(f"[case33-gui] rebuilt scene: gravity_off={state['gravity_off']}", flush=True)

    rebuild_scene()
    MAX_ANGLE_PER_STEP_DEG = float(os.environ.get("CASE33_MAX_DEG", "0.4"))
    SLIDER_MAX = float(os.environ.get("CASE33_SLIDER_MAX", "60.0"))

    def do_step():
        scene = state['scene']
        eng = scene['eng']
        # Rate-limit
        delta = state['angle_deg'] - state['driven_deg']
        if abs(delta) > MAX_ANGLE_PER_STEP_DEG:
            state['driven_deg'] += MAX_ANGLE_PER_STEP_DEG * np.sign(delta)
        else:
            state['driven_deg'] = state['angle_deg']
        eng.native.set_revolute_target(scene['joint_idx'], math.radians(state['driven_deg']))

        # Capture stdout to count Newton/LS metrics
        cap = io.StringIO()
        with contextlib.redirect_stdout(cap):
            t0 = time.perf_counter()
            eng.step()
            state['last_step_ms'] = (time.perf_counter() - t0) * 1000
        so = cap.getvalue()
        max_n = 0
        for line in so.split('\n'):
            if 'Newton iter' in line:
                try:
                    n = int(line.split('Newton iter')[1].strip().split()[0])
                    max_n = max(max_n, n)
                except Exception: pass
        state['last_newton_iters'] = max_n
        state['last_ls_hangs'] = so.count('lineSearchCount=9')
        state['cumulative_ls_hangs'] += state['last_ls_hangs']
        state['step_count'] += 1

        # Update meshes
        cur_verts = eng.get_vertices()
        state['cloth_mesh'].update_vertex_positions(cur_verts[state['cloth_v_off']:state['cloth_v_end']])
        for (name, mesh), (v0, v1) in zip(state['arm_meshes'], state['arm_recs']):
            mesh.update_vertex_positions(cur_verts[v0:v1])

    def callback():
        psim.SetNextWindowPos((10, 10), psim.ImGuiCond_Once)
        psim.SetNextWindowSize((460, 0), psim.ImGuiCond_Once)
        psim.Begin("case_33 — arm + cloth + gravity toggle")

        psim.Text(f"step #{state['step_count']}: {state['last_step_ms']:.1f} ms  "
                  f"newton_iter={state['last_newton_iters']}  "
                  f"ls_hang={state['last_ls_hangs']}  "
                  f"cum_ls_hang={state['cumulative_ls_hangs']}")

        gravity_label = "arm_link gravity: OFF (no drift, smooth)" if state['gravity_off'] \
                        else "arm_link gravity: ON (joint penalty wrestles g)"
        psim.Text(gravity_label)
        if psim.Button("Toggle gravity (rebuild scene)"):
            state['gravity_off'] = not state['gravity_off']
            rebuild_scene()

        psim.Separator()
        if state['running']:
            if psim.Button("Pause"): state['running'] = False
        else:
            if psim.Button("Run"): state['running'] = True
        psim.SameLine()
        if psim.Button("Step"): do_step()
        psim.SameLine()
        if psim.Button("Reset angle"):
            state['angle_deg'] = 0.0

        psim.Separator()
        psim.Text(f"Joint angle (deg) — rate {MAX_ANGLE_PER_STEP_DEG}°/step:")
        chg, val = psim.SliderFloat("angle##j", state['angle_deg'],
                                    v_min=-SLIDER_MAX, v_max=SLIDER_MAX)
        if chg: state['angle_deg'] = val
        psim.Text(f"  driven: {state['driven_deg']:+.1f}°  target: {state['angle_deg']:+.1f}°")
        psim.End()

        chasing = abs(state['angle_deg'] - state['driven_deg']) > 1e-3
        if state['running'] or chasing:
            do_step()

    ps.set_user_callback(callback)
    ps.show()


def main():
    if int(os.environ.get("AUTO_BENCH", "0")):
        run_benchmark()
    elif int(os.environ.get("AUTO_STEP", "0")):
        # quick smoke test
        n = int(os.environ["AUTO_STEP"])
        scene = setup_scene(disable_arm_gravity=True)
        for i in range(n):
            t0 = time.perf_counter()
            scene['eng'].step()
            print(f"[case33-smoke] step {i}: {(time.perf_counter()-t0)*1000:.1f} ms",
                  flush=True)
    else:
        run_gui()


if __name__ == "__main__":
    main()

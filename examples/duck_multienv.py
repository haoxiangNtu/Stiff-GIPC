"""Multi-env rubber-duck drop/settle — StiffGIPC env-ceiling test vs uipc 512.

Each env = one rubber duck (light: 274 tet verts / 931 tets, extracted from the
Newton rubber_duck USD via assets/duck/duck_tet.npz) tiled in a grid and dropped
onto the ground. Mirrors the SCALE question from the uipc 512-env duck video:
how many independent grasp-class bodies can StiffGIPC's MERGED world hold on 24GB?

NOTE: StiffGIPC has no GPU geometry instancing — load_mesh_instanced parses the
mesh once but stores N FULL copies in GPU memory. The duck is light, so this is a
fair best-case ceiling for rigid/light-deformable grasp bodies (unlike the heavy
6436v cloth which caps ~20).

Env vars:
  DUCK_N=64        number of envs (ducks)
  DUCK_TYPE=ABD    ABD (rigid, cheapest) or FEM (soft, faithful to the video)
  DUCK_SPACING=0.25  grid spacing [m]
  DUCK_FRAMES=30   frames to simulate
  DUCK_HEADLESS=1  1=headless (ceiling sweep), 0=polyscope GUI
"""
import os, sys, time, math
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
from stiff_physics import Engine, Config

DUCK_TET = os.path.join(_ROOT, "Assets", "duck",
                        "duck_tet_light.npz" if int(os.environ.get("DUCK_LIGHT", "0"))
                        else "duck_tet.npz")


def gpu_mem_mb():
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        return int(out.decode().split("\n")[0])
    except Exception:
        return -1


def main():
    N        = int(os.environ.get("DUCK_N", "64"))
    dtype    = os.environ.get("DUCK_TYPE", "ABD").upper()
    spacing  = float(os.environ.get("DUCK_SPACING", "0.25"))
    frames   = int(os.environ.get("DUCK_FRAMES", "30"))
    headless = int(os.environ.get("DUCK_HEADLESS", "1"))

    d = np.load(DUCK_TET)
    verts, cells = d["verts"].astype(np.float64), d["cells"].astype(np.int32)
    # duck is ~6x8x5.6 cm centered at origin; lift above ground
    print(f"[duck] N={N} type={dtype} duck={verts.shape[0]}v/{cells.shape[0]}tet "
          f"spacing={spacing} frames={frames}", flush=True)

    cfg = Config(dt=0.02, poisson_rate=0.45, friction_rate=0.4,
                 relative_dhat=1e-3, newton_tol=5e-2, newton_iter_cap=50,
                 preconditioner_type=1, ground_offset=0.0,
                 assets_dir=_ROOT + "/Assets/")
    cfg._cfg.absolute_dhat = 0.0019
    # isolated ducks -> few contacts; default triplet_internal_margin=32 blows the
    # FEM triplet/contact buffer at large N (OOM). Use a small margin + tight buffers.
    cfg._cfg.collision_detection_buff_scale = float(os.environ.get("DUCK_BUFF", "1.0"))
    cfg._cfg.linear_system_buff_scale       = float(os.environ.get("DUCK_LSYS", "1.0"))
    cfg._cfg.triplet_internal_margin        = float(os.environ.get("DUCK_MARGIN", "2.0"))
    eng = Engine(cfg)

    # grid of N transforms, ducks lifted 0.15 m above ground
    side = int(math.ceil(math.sqrt(N)))
    Ts = []
    for i in range(N):
        r, c = divmod(i, side)
        T = np.eye(4); T[0, 3] = c * spacing; T[1, 3] = 0.15; T[2, 3] = r * spacing
        Ts.append(T)

    young = 1e8 if dtype == "ABD" else 3e5  # rigid stiff vs soft duck (~300 kPa)
    t0 = time.perf_counter()
    res = eng.native.load_mesh_instanced(verts, cells, 4, 3,
                                         0 if dtype == "ABD" else 1,
                                         Ts, young, 0)
    eng.finalize()
    t_load = time.perf_counter() - t0
    print(f"[duck] loaded {N} ducks in {t_load:.1f}s  GPU mem={gpu_mem_mb()} MB", flush=True)

    if headless:
        ms = []
        for f in range(frames):
            t = time.perf_counter(); eng.step(); ms.append((time.perf_counter()-t)*1000.0)
            if f % 10 == 0:
                print(f"[duck] frame {f:3d} step={ms[-1]:6.0f}ms  GPU={gpu_mem_mb()}MB", flush=True)
        mm = float(np.mean(ms))
        print(f"\n[duck-RESULT] N={N} type={dtype}: mean {mm:.1f}ms/step "
              f"({1000.0/mm:.2f} fps, {N*1000.0/mm:.0f} env-steps/s)  "
              f"peak GPU={gpu_mem_mb()}MB  load={t_load:.1f}s", flush=True)
        return

    import polyscope as ps
    v = eng.get_vertices(); fa = eng.get_surface_faces()
    ps.init(); ps.set_up_dir("y_up"); ps.set_ground_plane_mode("shadow_only")
    st = dict(run=False, mesh=ps.register_surface_mesh("ducks", v, fa, color=(0.95,0.85,0.2)))
    import polyscope.imgui as psim
    def cb():
        if psim.Button("Start/Pause"): st['run'] = not st['run']
        psim.Text(f"{N} ducks ({dtype})  GPU {gpu_mem_mb()} MB")
        if st['run']:
            t=time.perf_counter(); eng.step(); dtms=(time.perf_counter()-t)*1000.0
            psim.Text(f"step {dtms:.0f} ms")
            st['mesh'].update_vertex_positions(eng.get_vertices())
    ps.set_user_callback(cb); ps.show()


if __name__ == "__main__":
    main()

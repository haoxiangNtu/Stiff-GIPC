"""CPU/GPU test: water inside an open box made of triangles (a 'cup'), then
the cup slides sideways.  Checks containment (no leak through the walls) and
that the water follows the moving cup."""
import os, sys, time, numpy as np, warp as wp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
wp.config.quiet = True
from stiff_physics.liquid import LiquidMPM

dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
dx = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
# open box: inner 0.08 x 0.10 x 0.08 m, wall thickness 0.004 -> closed solid shell (outer minus inner), outward winding
import trimesh
outer = trimesh.creation.box(extents=[0.088, 0.104, 0.088]); outer.apply_translation([0, 0.052, 0])
inner = trimesh.creation.box(extents=[0.080, 0.104, 0.080]); inner.apply_translation([0, 0.052 + 0.004 + 0.002, 0])
try:
    cup = outer.difference(inner)      # boolean (needs manifold3d / blender)
except Exception:
    cup = None
if cup is None or cup.is_empty or not cup.is_watertight:
    # fallback: 5 thick slabs as separate closed boxes
    parts = []
    t = 0.004
    parts.append(trimesh.creation.box(extents=[0.088, t, 0.088]).apply_translation([0, t / 2, 0]))
    for sx in (-1, 1):
        parts.append(trimesh.creation.box(extents=[t, 0.104, 0.088]).apply_translation([sx * (0.040 + t / 2), 0.052, 0]))
        parts.append(trimesh.creation.box(extents=[0.088, 0.104, t]).apply_translation([0, 0.052, sx * (0.040 + t / 2)]))
    cup = trimesh.util.concatenate(parts)
print("cup: %d verts %d faces, signed volume %.3e" % (len(cup.vertices), len(cup.faces), LiquidMPM.check_orientation(cup.vertices, cup.faces)))
origin = np.array([-0.16, -0.02, -0.16]); extent = np.array([0.32, 0.24, 0.32])
SOLVER = os.environ.get("LIQ_SOLVER", "lite_explicit"); SUB = float(os.environ.get("LIQ_SUBSTEP", "1e-4"))
liq = LiquidMPM(origin, extent, dx, device=dev, gravity=(0, -9.81, 0), kappa=1e5, ppc=8, substep_dt=SUB, verbose=True, solver=SOLVER)
print("solver", SOLVER, "substep", SUB)
liq.band_ext = float(os.environ.get("LIQ_BAND_EXT", "0")) * dx
n = liq.seed_box([-0.035, 0.012, -0.035], [0.035, 0.062, 0.035], jitter=0.5)
print("seeded", n, "particles; grid", liq.grid_size)
V0 = cup.vertices.copy(); F = cup.faces
liq.set_surface_mesh(V0, F)
dt_solid = 0.005
def report(tag):
    x = liq.positions()
    inside = (np.abs(x[:, 0]) < 0.040) & (np.abs(x[:, 2]) < 0.040) & (x[:, 1] > 0.004)
    print(f"[{tag}] t={liq.sim_time:.3f}s  inside {inside.mean()*100:5.1f}%  y min/max {x[:,1].min():.4f}/{x[:,1].max():.4f}  "
          f"x mean {x[:,0].mean():+.4f}  bc nodes {liq.n_bc}  projected {liq.projected_last()}", flush=True)
t0 = time.time()
# phase 1: settle 0.3 s
for k in range(60):
    liq.update_surface(V0, dt_solid); liq.step(dt_solid)
    if k % 20 == 19: report("settle")
# phase 2: slide the cup +x at 0.2 m/s for 0.3 s, then stop 0.2 s
vx = 0.2; V = V0.copy()
for k in range(100):
    if k < 60: V = V + np.array([vx * dt_solid, 0, 0])
    liq.update_surface(V, dt_solid); liq.step(dt_solid)
    if k % 20 == 19:
        x = liq.positions(); cx = V[:, 0].mean()
        inside = (np.abs(x[:, 0] - cx) < 0.040) & (np.abs(x[:, 2]) < 0.040) & (x[:, 1] > 0.004)
        print(f"[slide ] t={liq.sim_time:.3f}s  cup x {cx:+.4f}  water x mean {x[:,0].mean():+.4f}  inside {inside.mean()*100:5.1f}%  y min/max {x[:,1].min():.4f}/{x[:,1].max():.4f}  bc {liq.n_bc}  proj {liq.projected_last()}", flush=True)
print("wall %.1f s total (%d particles, dx %.3f, device %s)" % (time.time() - t0, n, dx, dev))

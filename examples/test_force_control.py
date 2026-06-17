#!/usr/bin/env python3
"""Prototype validation for set_body_external_force (ABD external body force).

Free ABD cube, NO gravity. Apply a constant linear force F in +x and check the
centroid follows x(t) = 1/2 a t^2 with a = F/m (constant accel, linear in F,
correct sign). Run for F in {0, +10, +20, -10}.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stiff_physics import Engine, Config

CUBE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    ("assets" if os.path.isdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")) else "Assets") + "/sim_data/tetmesh/cube.msh")
DT = 0.01
N  = 40


def run(Fx):
    cfg = Config(gravity=(0.0, 0.0, 0.0), dt=DT)
    eng = Engine(cfg)
    eng.load_mesh(CUBE, dimensions=3, body_type="ABD", young_modulus=1e8,
                  boundary_type="Free")
    rec = eng.get_load_records()[-1]
    bid, voff, vcnt = rec.body_offset, rec.vertex_offset, rec.vertex_count
    eng.finalize()
    eng.native.set_body_external_force(bid, float(Fx), 0.0, 0.0)

    def cx():
        v = eng.get_vertices()
        return float(np.asarray(v[voff:voff + vcnt]).mean(axis=0)[0])

    x0 = cx()
    xs = []
    for f in range(N):
        eng.step()
        xs.append(cx() - x0)
    return np.array(xs)


print(f"{'F(N)':>6} {'x@end(mm)':>12} {'a=2x/t^2':>12} {'quad-fit a':>12}")
results = {}
for F in [0.0, 10.0, 20.0, -10.0]:
    xs = run(F)
    t = np.arange(1, N + 1) * DT
    x_end = xs[-1]
    a_end = 2 * x_end / (t[-1] ** 2) if abs(x_end) > 1e-12 else 0.0
    # quadratic fit x = 0.5 a t^2  -> a = 2 * slope of (x vs t^2)
    a_fit = 2 * np.polyfit(t**2, xs, 1)[0]
    results[F] = a_fit
    print(f"{F:>6.1f} {1000*x_end:>12.4f} {a_end:>12.4f} {a_fit:>12.4f}")

print("\n=== Checks ===")
print(f"F=0 stays put (|x|<0.1mm): {abs(results[0.0])<1e-3 and True}  "
      f"(a={results[0.0]:.5f})")
print(f"F=+10 moves +x (a>0): {results[10.0] > 0}")
print(f"F=-10 moves -x (a<0): {results[-10.0] < 0}")
print(f"linear in F (a(20)/a(10) ~ 2): "
      f"{results[20.0]/results[10.0]:.3f}" if results[10.0] else "n/a")
print(f"symmetric (a(-10) ~ -a(10)): "
      f"{results[-10.0]/results[10.0]:.3f}" if results[10.0] else "n/a")

# Liquids in StiffGIPC (implicit MPM Lite + IPC, monolithic)

Water is simulated **inside** the IPC Newton solve: the active grid nodes of
an MPM Lite background grid are extra unknowns, every active voxel is an
8-node water element (`ψ = κ/2 (J−1)²`, no tension), and the particles are
embedded in their 27-node stencil. Particles are also IPC collision points:
the particle–triangle barrier (and CCD) of the engine applies to them, with
the barrier gradient/Hessian chain-ruled onto the grid nodes. Solids feel the
water through the same contact blocks, so the coupling is two-way and
implicit (one 5 ms step, no liquid sub-stepping).

Engine code: `StiffGIPC/mpm/liquid_system.{cuh,cu}`; design notes (internal):
`MPM_IPC_MONOLITHIC_DESIGN.md`.

## API

```python
from stiff_physics.engine import Engine, Config
eng = Engine(Config(dt=0.005, preconditioner_type=0, absolute_dhat=5e-4, ...))
# ... load the solids (ABD / FEM bodies) ...
eng.add_liquid(points,            # (N,3) particle seeds (m); keep >= 1 cell of grid margin
               dx=0.005,          # grid spacing (m); 8 particles per cell = spacing dx/2
               domain_lo=lo, domain_hi=hi,   # grid box (m), may extend below the ground
               rho=1000.0, kappa=1e5,        # density, bulk stiffness (Pa)
               particle_volume=(dx/2)**3,    # rest volume per particle
               node_pool=0,       # grid-node slots reserved, FIXED at finalize (0 = auto: max(8192, 12x the seed's nodes);
                                  #   a pour that spreads the water over several vessels needs ~4x the seed, splashes more)
               mass_floor=0.05,   # node mass floor as a fraction of rho*dx^3 (free-surface conditioning)
               apic=True, push_apart=0.9,    # APIC transfers; particle min-spacing floor (fraction of spacing)
               j_mode=3)          # 3 = tracked J re-anchored to the cell volume ratio in full cells;
                                  # 4 = cell volume ratio x geometric fill fraction (position-based, no surface creep)
eng.finalize()
eng.step()
x = eng.get_liquid_particles()    # (N,3)
v = eng.get_liquid_velocities()
J = eng.get_liquid_J()
eng.get_liquid_stats()            # active cells / nodes, slot bases, body ids
```

Constraints: one liquid per engine; `add_liquid` before `finalize()`;
`preconditioner_type=0` (the MAS preconditioner reorders FEM vertices);
use `Config(energy_rel_tol=1e-9)` with curved thin meshes (roundoff ties at convergence).
Water–solid friction is 0 (inviscid). Liquid grid nodes never collide; the
particles collide with every solid body and the ground.

## Tests (run on a GPU; ~10–60 s each)

| script | what it checks |
|---|---|
| `m1_dambreak.py` | 10 cm water column collapses on the ground: front speed, volume (final sheet thickness), nearest-neighbour spacing (clumping), Newton/step |
| `m2_box.py` | 0.8 L in a static open box (Fixed ABD): containment (0 outside, 0 penetration), rest depth (55.6 mm expected), spacing, Newton/step; options `--jitter --mass-floor --far-box --pcg-tol` reproduce the lab-scene solver settings |
| `m4_beaker_tilt.py` | the lab beaker (kinematic, teleported) filled to 0.6 and tilted about its lip: spill onset 52–54° (geometric 50°); stalls at the rim crossing with newton_tol 1e-4 on a small scene (use 1e-3) |
| `m3_float.py` | 4 cm free ABD cube (ρ = 500) dropped in: floats half submerged, water rises by the displaced volume (two-way coupling); `--cube-fixed` isolates the contact; `--push --jmode --dhat --pcg-tol` sweeps |

Known issues: `apic=False` (PIC transfers) produces NaN particle positions (untested path, use APIC);
`j_mode 3` grows a thin low-density layer at a resting free surface (~0.6 mm/s at dx 5 mm; `j_mode 4` = cell density x geometric fill fraction removes it (10 s box: surface flat) but is EXPERIMENTAL: it loses ~7 % level under a dropped floating cube (sub-grid clumping) and stalls the Newton solve on the dam-break ground impact);
the Newton solve needs ~10 iterations/step whenever a free body sits in the water.

Debug switches: `LIQ_DEBUG=1` (chain-rule counts, energy-slot dumps on line-search failures, direction norms, CCD alphas), `LIQ_NO_EXPAND / LIQ_NO_CROSS / LIQ_NO_DIAG / LIQ_LUMP_DIAG` (Hessian expansion ablations), `LIQ_TENSION`, `LIQ_REANCHOR_BETA`, `LIQ_CONV_SCALE`.

Lab demo: `python examples/lab_tactile/run_lab_demo.py --liquid beaker_mix --liquid-engine --liquid-fill 0.6 --stages 4`.

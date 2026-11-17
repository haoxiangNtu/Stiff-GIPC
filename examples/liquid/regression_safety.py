"""Native input rejection and real-engine tail-block CUDA regression.

Run on the simulation pod. Use Compute Sanitizer for --mode smoke;
test --particles 256, 257, 10524, and repeat with LIQ_ST=2.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stiff_physics.engine import Engine, Config, _C
from m2_box import open_box


def validation():
    # Deliberately bypass the Python wrapper's reshape: the native API is public.
    engine = _C.SimEngine()
    base = dict(xyz=np.array([[.2, .03, .2]]), dx=.005,
                domain_lo=np.array([.1, .001, .1]), domain_hi=np.array([.3, .2, .3]))
    cases = [
        ("scalar", dict(xyz=np.array(1.))),
        ("one_dimensional", dict(xyz=np.ones(3))),
        ("two_columns", dict(xyz=np.ones((5, 2)))),
        ("four_columns", dict(xyz=np.ones((5, 4)))),
        ("empty", dict(xyz=np.empty((0, 3)))),
        ("nan_position", dict(xyz=np.array([[np.nan, .03, .2]]))),
        ("infinite_position", dict(xyz=np.array([[np.inf, .03, .2]]))),
        ("outside_domain", dict(xyz=np.array([[.4, .03, .2]]))),
        ("zero_dx", dict(dx=0.)), ("negative_dx", dict(dx=-.1)),
        ("nan_dx", dict(dx=np.nan)), ("tiny_dx", dict(dx=1e-300)),
        ("too_small_grid", dict(dx=1.)),
        ("overflow_grid", dict(dx=1e-5)),
        ("reversed_domain", dict(domain_hi=np.array([0., 0., 0.]))),
        ("infinite_domain", dict(domain_hi=np.array([np.inf, .2, .3]))),
        ("negative_rho", dict(rho=-1.)), ("nan_kappa", dict(kappa=np.nan)),
        ("negative_volume", dict(particle_volume=-1.)),
        ("infinite_volume", dict(particle_volume=np.inf)),
        ("negative_pool", dict(node_pool=-1)),
        ("invalid_jmode", dict(j_mode=5)),
        ("negative_mass_floor", dict(mass_floor=-1.)),
        ("nan_push", dict(push_apart=np.nan)),
        ("negative_ground_mu", dict(ground_mu=-.1)),
        ("negative_stick", dict(stick_h=-1.)),
        ("invalid_apic_ratio", dict(apic_ratio=1.1)),
    ]
    for name, changes in cases:
        try:
            engine.add_liquid(**(base | changes))
        except (ValueError, RuntimeError) as exc:
            print(f"PASS reject {name}: {exc}", flush=True)
        else:
            raise AssertionError(f"accepted invalid input: {name}")
    engine.add_liquid(**base)
    print(json.dumps(dict(test="native_validation", rejected=len(cases), valid_retry=True)), flush=True)


def validate_runtime(engine):
    base = dict(body=np.array([0], np.int32), Rd=np.eye(3).ravel(), td=np.zeros(3),
                nprof=np.array([2], np.int32), py=np.r_[.02, .18, np.zeros(30)],
                pr=np.r_[.06, .06, np.zeros(30)], y_top=np.array([.18]),
                band=np.array([.005]), slip=np.array([1], np.int32))
    cases = [
        ("negative_body", dict(body=np.array([-1], np.int32))),
        ("body_out_of_range", dict(body=np.array([1], np.int32))),
        ("empty_profile", dict(nprof=np.array([0], np.int32))),
        ("oversized_profile", dict(nprof=np.array([33], np.int32))),
        ("unordered_profile", dict(py=np.r_[.18, .02, np.zeros(30)])),
        ("negative_radius", dict(pr=np.r_[-.06, .06, np.zeros(30)])),
        ("nan_transform", dict(Rd=np.full(9, np.nan))),
        ("short_transform", dict(Rd=np.zeros(8))),
        ("negative_band", dict(band=np.array([-.005]))),
        ("invalid_slip", dict(slip=np.array([2], np.int32))),
        ("nan_wall_mass", dict(wall_mass=np.nan)),
        ("invalid_ground", dict(ground_bc=2)),
    ]
    for name, change in cases:
        try:
            engine._engine.set_liquid_containers(**(base | change))
        except (ValueError, RuntimeError) as exc:
            print(f"PASS reject {name}: {exc}", flush=True)
        else:
            raise AssertionError(f"accepted invalid container: {name}")
    pump = dict(tip=np.array([.2, .03, .2]), top=np.array([.2, .1, .2]), radius=0., accel=0.)
    for change in [dict(radius=np.nan), dict(accel=np.inf), dict(track=2.), dict(follow=-1.),
                   dict(tip=np.array([np.nan, .03, .2]))]:
        try:
            engine._engine.set_liquid_pump(**(pump | change))
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError(f"accepted invalid pump: {change}")
    # Valid retries must still work, without introducing a wall penalty into this smoke test.
    os.environ["LIQ_WALL_BC"] = "0"
    engine._engine.set_liquid_containers(**base)
    engine._engine.set_liquid_pump(**pump)
    print(json.dumps(dict(test="runtime_validation", rejected=len(cases) + 5, valid_retry=True)), flush=True)


def smoke(n, steps, expect_invalid_env=False, runtime_validation=False):
    engine = Engine(Config(dt=.005, preconditioner_type=0, absolute_dhat=.0005,
                           newton_tol=1e-3, pcg_tol=1e-6, newton_iter_cap=80,
                           friction_rate=0., gd_friction_rate=0.))
    engine.set_log_level(-1)
    vertices, faces = open_box((.2, .2), .12, .005, .02, .16, .01)
    engine.load_mesh_from_data(vertices, faces, 3, 3, "ABD", np.eye(4), 1e7, "Fixed")
    side = int(np.ceil(n ** (1 / 3)))
    xyz = np.stack(np.meshgrid(*([np.arange(side)] * 3), indexing="ij"), axis=-1).reshape(-1, 3)[:n]
    xyz = xyz * .0025 + np.array([.17, .022, .17])
    # Close neighbours exercise actual push-apart, in addition to the tail guard.
    xyz += np.random.default_rng(5).uniform(-.0002, .0002, xyz.shape)
    engine.add_liquid(xyz, .005, [.1, .001, .1], [.3, .2, .3], j_mode=0, push_apart=.9)
    if expect_invalid_env:
        try:
            engine.finalize()
        except (ValueError, RuntimeError) as exc:
            print(f"PASS reject environment: {exc}", flush=True)
            return
        raise AssertionError("accepted invalid environment")
    engine.finalize()
    if runtime_validation:
        validate_runtime(engine)
    for _ in range(steps):
        engine.step()
        x = engine.get_liquid_particles()
        v = engine.get_liquid_velocities()
        j = engine.get_liquid_J()
        assert x.shape == (n, 3) and np.isfinite(x).all() and np.isfinite(v).all()
        assert np.isfinite(j).all() and (j > 0).all()
        assert (x[:, 1] >= .02).all(), "particle crossed container floor"
    print(json.dumps(dict(test="engine_smoke", particles=n, steps=steps,
                          surface_tension=os.environ.get("LIQ_ST", "0"),
                          max_speed=float(np.linalg.norm(v, axis=1).max()),
                          mean_J=float(j.mean()))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validation", "smoke", "invalid-env", "runtime-validation"], default="validation")
    parser.add_argument("--particles", type=int, default=257)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    if args.mode == "validation":
        validation()
    else:
        smoke(args.particles, args.steps, args.mode == "invalid-env", args.mode == "runtime-validation")

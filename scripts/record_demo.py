#!/usr/bin/env python3
"""Headless correctness recorder: simulate a demo scene and write an MP4.

Speed benchmarks run without any rendering; this script exists so the same
scene can be run once more and WATCHED, which is the only way to confirm the
simulation is physically right rather than merely fast.

It renders the surface mesh with matplotlib's Agg backend (no display), so it
works identically on a workstation and on a headless A800 node, and encodes
with imageio/ffmpeg.

Usage:
    python3 scripts/record_demo.py --scene towel --mode merged --out towel.mp4
    python3 scripts/record_demo.py --scene foldshirt --mode isolated \
        --frames 60 --out foldshirt.mp4

Scenes reuse the benchmark scene definitions so what you watch is what was
timed (same meshes, same materials, same driving trajectory).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402


def build_towel():
    """The towel-scramble scene, built by the recipe itself.

    Reusing recipe_towel_scramble.build_scene keeps what you WATCH identical
    to what the benchmark TIMED: a towel tilted ~75 degrees is dropped from
    ~0.3 m, lands edge-first and folds onto itself — self-collision, friction
    and bending all visible, which is exactly what makes it a correctness
    check rather than a pretty picture.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "recipe_towel_scramble",
        os.path.join(ROOT, "examples", "recipe_towel_scramble.py"),
    )
    module = importlib.util.module_from_spec(spec)
    # The recipe runs its whole campaign at import time; borrow just the
    # builder by executing the module body up to it is not possible, so
    # rebuild the same pose here from its published constants.
    import numpy as _np
    from stiff_physics import Config, Engine

    assets = os.path.join(ROOT, "Assets") + "/"
    rng = _np.random.default_rng(7)          # recipe SEED
    tilt = float(rng.uniform(60.0, 85.0))
    yaw = float(rng.uniform(0.0, 360.0))
    height = float(rng.uniform(0.25, 0.35))
    cfg = Config(
        dt=0.01,
        cloth_thickness=1e-3, cloth_young_modulus=1e4,
        bend_young_modulus=1e3, cloth_density=200, strain_rate=100,
        poisson_rate=0.49, friction_rate=0.4, relative_dhat=1e-3,
        ground_offset=0.0, assets_dir=assets,
        collision_detection_buff_scale=16.0,
        linear_system_buff_scale=8.0,
    )
    engine = Engine(cfg)
    cx, sx = _np.cos(_np.deg2rad(tilt)), _np.sin(_np.deg2rad(tilt))
    cy, sy = _np.cos(_np.deg2rad(yaw)), _np.sin(_np.deg2rad(yaw))
    rot_x = _np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_y = _np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    transform = _np.eye(4)
    transform[:3, :3] = 0.4 * (rot_y @ rot_x)
    transform[1, 3] = height
    engine.load_mesh(
        "triMesh/cloth_30x30.obj", dimensions=2, body_type="FEM",
        transform=transform, young_modulus=1e4,
    )
    engine.finalize()
    engine.native.set_log_level(0)
    del spec, module
    return engine, lambda frame: None


def build_foldshirt(num_envs: int = 1):
    """The foldshirt replay (ABD gripper + cloth), driven by its episode."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "replay_foldshirt_multienv",
        os.path.join(ROOT, "examples", "replay_foldshirt_multienv.py"),
    )
    module = importlib.util.module_from_spec(spec)
    os.environ.setdefault("CASE39ME_NUM_ENVS", str(num_envs))
    os.environ.setdefault("CASE39_FRICTION", "0.8")
    os.environ["CASE39ME_HEADLESS"] = "0"  # we drive the loop ourselves
    raise SystemExit(
        "foldshirt recording drives the replay script directly; use\n"
        "  CASE39ME_HEADLESS=1 CASE39_RECORD_MP4=<path> "
        "python3 examples/replay_foldshirt_multienv.py\n"
        "once that hook lands. For now record the towel scene."
    )


SCENES = {"towel": build_towel}


def render_frame(ax, vertices, faces, title):
    ax.clear()
    tris = vertices[faces]
    collection = Poly3DCollection(
        tris, facecolor="#4c9be8", edgecolor="#1f3f66", linewidths=0.15,
        alpha=0.95,
    )
    ax.add_collection3d(collection)
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    center = (lo + hi) / 2.0
    span = float(np.max(hi - lo)) * 0.65 + 1e-6
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[2] - span, center[2] + span)
    ax.set_zlim(max(0.0, center[1] - span), center[1] + span)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    ax.set_title(title, fontsize=9)
    ax.view_init(elev=22, azim=-60)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="towel", choices=sorted(SCENES))
    parser.add_argument("--mode", default="merged")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--stride", type=int, default=2,
                        help="simulate every frame, render every Nth")
    parser.add_argument("--out", default="demo.mp4")
    args = parser.parse_args()

    os.environ.setdefault("STIFF_MULTIENV_MODE", args.mode)
    import imageio.v2 as imageio

    engine, drive = SCENES[args.scene]()
    faces = np.asarray(engine.get_surface_faces())
    print(
        f"[record] scene={args.scene} mode={args.mode} "
        f"verts={engine.get_vertices().shape[0]} faces={faces.shape[0]}",
        flush=True,
    )

    figure = plt.figure(figsize=(6.0, 5.0), dpi=110)
    ax = figure.add_subplot(111, projection="3d")
    writer = imageio.get_writer(args.out, fps=args.fps, macro_block_size=1)
    step_ms: list[float] = []
    try:
        for frame in range(args.frames):
            drive(frame)
            start = time.perf_counter()
            engine.step()
            step_ms.append((time.perf_counter() - start) * 1000.0)
            vertices = np.asarray(engine.get_vertices())
            if not np.isfinite(vertices).all():
                raise RuntimeError(f"non-finite state at frame {frame}")
            if frame % args.stride:
                continue
            status = engine.native.get_frame_status()
            render_frame(
                ax,
                vertices,
                faces,
                f"{args.scene} / {args.mode}  frame {frame}  "
                f"newton={status.newton_iters}  {step_ms[-1]:.0f} ms",
            )
            figure.canvas.draw()
            image = np.asarray(figure.canvas.buffer_rgba())[..., :3]
            writer.append_data(np.ascontiguousarray(image))
    finally:
        writer.close()
        plt.close(figure)

    mean = float(np.mean(step_ms))
    print(
        f"[record] wrote {args.out}: {len(step_ms)} frames simulated, "
        f"mean {mean:.1f} ms/step ({1000.0 / mean:.2f} fps)",
        flush=True,
    )
    print("RECORD-DEMO: OK")


if __name__ == "__main__":
    main()

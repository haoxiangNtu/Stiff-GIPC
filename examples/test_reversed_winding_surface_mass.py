#!/usr/bin/env python3
"""Regression: a closed, globally inward-wound ABD mesh has physical mass."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stiff_physics.engine import Config, Engine


ASSETS_DIR = Path(__file__).resolve().parent.parent / "Assets"


def load_triangle_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    triangles = []
    with path.open("r", encoding="utf-8") as obj_file:
        for line in obj_file:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("f "):
                face = [int(value.split("/")[0]) - 1 for value in line.split()[1:]]
                if len(face) != 3:
                    raise ValueError(f"expected triangles in {path}")
                triangles.append(face)
    return np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.int32)


def main() -> None:
    vertices, outward_faces = load_triangle_obj(ASSETS_DIR / "triMesh/cube_outward.obj")
    inward_faces = outward_faces[:, [0, 2, 1]].copy()
    signed_volume = float(
        np.sum(
            np.einsum(
                "ij,ij->i",
                vertices[inward_faces[:, 0]],
                np.cross(vertices[inward_faces[:, 1]], vertices[inward_faces[:, 2]]),
            )
        )
        / 6.0
    )
    if not signed_volume < 0.0:
        raise AssertionError(f"test mesh is not inward-wound: volume={signed_volume}")

    engine = Engine(
        Config(
            dt=0.01,
            ground_offset=-2.0,
            assets_dir=str(ASSETS_DIR) + "/",
        )
    )
    transform = np.eye(4)
    transform[1, 3] = 0.5
    engine.load_mesh_from_data(
        vertices=vertices * 0.1,
        faces=inward_faces,
        verts_per_face=3,
        dimensions=3,
        body_type="ABD",
        transform=transform,
        young_modulus=1e5,
        boundary_type="Free",
    )
    engine.finalize()
    engine.step()

    positions = engine.get_vertices()
    if not np.isfinite(positions).all():
        raise AssertionError("inward-wound ABD produced non-finite positions")
    print(
        "PASS: globally reversed closed ABD mesh finalized and stepped; "
        f"signed_volume={signed_volume:.6g}"
    )


if __name__ == "__main__":
    main()

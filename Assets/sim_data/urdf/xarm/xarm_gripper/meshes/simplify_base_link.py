#!/usr/bin/env python3
"""Simplify base_link.STL for the xarm gripper.

Original: 14412 vertices, 24227 triangles
Target:   ~2000 faces (reduce by ~12x)

Saves the original as base_link_original.STL and writes the
simplified mesh to base_link.STL.
"""

import shutil
from pathlib import Path
import trimesh

src = Path(__file__).parent / "base_link.STL"
backup = Path(__file__).parent / "base_link_original.STL"
target_faces = 2000

if not backup.exists():
    shutil.copy2(src, backup)
    print(f"[backup] {src.name} -> {backup.name}")
else:
    print(f"[backup] {backup.name} already exists, skipping copy")

mesh = trimesh.load(str(src))
print(f"[original] vertices={len(mesh.vertices)}, faces={len(mesh.faces)}")

simplified = mesh.simplify_quadric_decimation(face_count=target_faces)
print(f"[simplified] vertices={len(simplified.vertices)}, faces={len(simplified.faces)}")

simplified.export(str(src))
print(f"[saved] {src.name} ({src.stat().st_size} bytes)")

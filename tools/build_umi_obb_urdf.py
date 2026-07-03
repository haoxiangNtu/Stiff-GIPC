#!/usr/bin/env python3
"""Generate an OBB-collision variant of the UMI ridgeback_dual_panda2 URDF.

For the UMI dual-panda robot, this script:
  1. For every link with a collision EXCEPT the 4 finray finger links:
       - <mesh> collision -> load mesh, compute trimesh bounding_box_oriented,
         export 8-vert OBB .obj into meshes/obb_umi/, rewrite the collision mesh
         filename to that obj (strip any 'scale').
       - <box>/<cylinder>/<sphere> primitive collision -> convert to a trimesh,
         OBB it, export, replace the primitive geometry with a <mesh>.
       The original <collision><origin> (xyz/rpy) is PRESERVED as-is. The OBB obj
       is in mesh-local coords; the URDF origin still applies on top.
  2. The 4 finray finger collisions (left/right_arm_{left,right}finger) are left
     untouched — they are the rigid mount seats of the hybrid finray gripper.
  3. Writes ridgeback_dual_panda2_OBB.urdf in the same dir (original untouched).

Usage:
    python build_umi_obb_urdf.py
"""

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh


URDF_DIR = Path(
    "/home/ps/Downloads/assets/sim_data/urdf/ridgeback_dual_panda_UMI"
)
URDF_PATH = URDF_DIR / "ridgeback_dual_panda2.urdf"
# OBB_OUT lets the caller pick the output filename (so we can emit both the
# correct variant and a deliberately-broken hand-OBB variant for comparison).
OUT_URDF_PATH = URDF_DIR / os.environ.get("OBB_OUT", "ridgeback_dual_panda2_OBB.urdf")
OBB_SUBDIR = "meshes/obb_umi"

# Links to KEEP detailed (do NOT OBB).  Mirrors case_39's softgripper URDF,
# which OBBs the FAR arm links (link0-7) + mobile base but keeps the HAND and
# fingers as detailed collision meshes.  A fat OBB box on a link that sits right
# next to the deformables (finray FEM + grasped cloth) pollutes the local CCD
# broad-phase and destabilises the soft finray (it balloons).  So the hand —
# the link the finray gripper mounts to — must stay detailed, exactly like
# case_39 (meshes/franka_hand/collision/hand.stl).
KEEP_LINKS = {
    "left_arm_leftfinger",
    "left_arm_rightfinger",
    "right_arm_leftfinger",
    "right_arm_rightfinger",
}
# The finray gripper MOUNTS on the hand; OBB-ifying the hand (and link7) turns
# them into solid boxes whose flat faces slice through the finray FEM (1179/2076
# verts land inside the box, all within the ~2.85 mm dHat shell), producing a
# dense thicket of vert-vs-box candidate contacts that hangs the broad phase and
# balloons the soft finray. case_39 avoids this by carrying NO collision on
# link6/7/hand at all.
#   OBB_KEEP_HAND=1 (default, CORRECT) keeps hand+link7 detailed (= the working
#       non-OBB geometry near the gripper) while the FAR arm links stay OBB.
#   OBB_KEEP_HAND=0 reproduces the BROKEN variant (hand OBB-ified) for comparison.
if int(os.environ.get("OBB_KEEP_HAND", "1")):
    KEEP_LINKS |= {
        "left_arm_hand", "right_arm_hand",
        "left_arm_link7", "right_arm_link7",
    }


def resolve_mesh_path(filename: str, urdf_dir: Path):
    """Resolve URDF mesh filename relative to the URDF dir (with fallback)."""
    p = Path(filename)
    if p.is_absolute() and p.exists():
        return str(p)
    candidate = urdf_dir / filename
    if candidate.exists():
        return str(candidate)
    if p.exists():
        return str(p)
    parts = p.parts
    for start in range(len(parts)):
        suffix = Path(*parts[start:])
        for ancestor in [urdf_dir] + list(urdf_dir.parents)[:4]:
            candidate = ancestor / suffix
            if candidate.exists():
                return str(candidate)
    return None


def primitive_to_mesh(geom_elem):
    """Convert a URDF primitive geometry element to a trimesh."""
    box = geom_elem.find("box")
    if box is not None:
        size = [float(x) for x in box.get("size", "0.01 0.01 0.01").split()]
        return trimesh.creation.box(extents=size)

    cyl = geom_elem.find("cylinder")
    if cyl is not None:
        r = float(cyl.get("radius", "0.01"))
        h = float(cyl.get("length", "0.01"))
        return trimesh.creation.cylinder(radius=r, height=h)

    sphere = geom_elem.find("sphere")
    if sphere is not None:
        r = float(sphere.get("radius", "0.01"))
        return trimesh.creation.icosphere(radius=r)

    return None


def main():
    if not URDF_PATH.exists():
        print(f"ERROR: URDF not found: {URDF_PATH}")
        sys.exit(1)

    urdf_dir = URDF_PATH.parent
    obb_dir = urdf_dir / OBB_SUBDIR
    obb_dir.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(str(URDF_PATH))
    root = tree.getroot()

    mesh_count = 0
    prim_count = 0
    kept_count = 0

    for link_elem in root.iter("link"):
        link_name = link_elem.get("name", "")

        collisions = list(link_elem.iter("collision"))
        if not collisions:
            continue

        if link_name in KEEP_LINKS:
            kept_count += 1
            print(f"  KEEP {link_name}: collision untouched")
            continue

        for collision in collisions:
            geom = collision.find("geometry")
            if geom is None:
                continue

            mesh_elem = geom.find("mesh")
            if mesh_elem is not None:
                # Mesh collision -> OBB obj
                src_filename = mesh_elem.get("filename", "")
                src_path = resolve_mesh_path(src_filename, urdf_dir)
                if src_path is None or not os.path.exists(src_path):
                    print(f"  WARN {link_name}: mesh not found: {src_filename}")
                    continue

                mesh = trimesh.load(src_path, force="mesh")
                obb = mesh.bounding_box_oriented

                obj_name = f"{link_name}_obb.obj"
                obj_path = obb_dir / obj_name
                obb.export(str(obj_path))

                # rewrite filename relative to urdf dir; strip scale
                rel = os.path.join(OBB_SUBDIR, obj_name)
                mesh_elem.set("filename", rel)
                if "scale" in mesh_elem.attrib:
                    del mesh_elem.attrib["scale"]

                print(
                    f"  OBB  {link_name}: {len(mesh.vertices)} -> 8 verts "
                    f"({rel})"
                )
                mesh_count += 1

            elif (
                geom.find("box") is not None
                or geom.find("cylinder") is not None
                or geom.find("sphere") is not None
            ):
                # Primitive collisions (wheels, riser) are NOT loaded as ABD
                # bodies by the importer in the original URDF — leave them as-is
                # so the OBB variant has the SAME ABD set as the original.
                # Converting them to mesh boxes would add 5 new ABDs (4 wheels +
                # riser) that overlap the chassis/plate at the initial pose and
                # blow up IPC on the first step.
                print(f"  SKIP {link_name}: primitive collision left untouched")
                prim_count += 1

    tree.write(str(OUT_URDF_PATH), xml_declaration=True, encoding="unicode")

    print("\n--- Summary ---")
    print(f"  OBB-ified (mesh):  {mesh_count}")
    print(f"  OBB-ified (prim):  {prim_count}")
    print(f"  OBB-ified total:   {mesh_count + prim_count}")
    print(f"  Kept (fingers):    {kept_count}")
    print(f"  OBB meshes dir:    {obb_dir}")
    print(f"  Output URDF:       {OUT_URDF_PATH}")


if __name__ == "__main__":
    main()

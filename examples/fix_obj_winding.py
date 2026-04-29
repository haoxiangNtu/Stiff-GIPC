#!/usr/bin/env python3
"""Standalone utility — fix face winding in a .obj / .STL collision mesh.

The Stiff-GIPC engine treats user-supplied collision geometry as
authoritative: it does NOT silently rewrite face indices. If your asset
has inconsistent triangle orientation (winding flipped on some subset
of faces), use this script to produce a winding-fixed copy once and
point your URDF at the fixed file.

Limitations
-----------
This utility uses `trimesh.fix_normals()`, which can only fix winding
on a closed manifold. It CANNOT fix:
  - Non-closed meshes (holes, gaps, interior boundaries) — the
    underlying volume is undefined
  - Self-intersecting meshes
  - Non-manifold edges (>2 faces sharing one edge)

A notable real-world example that this utility canNOT fix:
  xarm7's `xarm_gripper_base_link.STL` reports `is_volume=no` (not a
  closed manifold), so its signed volume is mathematically meaningless
  regardless of winding. The Stiff-GIPC engine still computes a number
  for ABD mass/centroid/inertia from this mesh, but those values are
  garbage. The fix is to either:
    (a) Re-export the collision mesh from CAD with proper closure
    (b) Replace it with a convex hull or VHACD decomposition
    (c) Accept that body 8 in xarm7+gripper has physically wrong
        inertia properties (engine has a `mass = ρ·|V|` fallback that
        keeps free-fall trajectories sensible but joint dynamics are
        subtly off)

Why we don't auto-fix in the engine
-----------------------------------
Asset quality is upstream's responsibility (URDF author, robot vendor,
mesh-processing pipeline). Auto-fixing in the engine would:
  - Silently change physics behavior depending on input quirks
  - Hide asset bugs from the people who can actually fix them upstream
  - Add a per-load CPU cost on every simulation startup

This script makes the fix explicit: you run it once, you commit the
output, you know exactly which files were modified.

Algorithm
---------
Uses trimesh's `fix_normals()`:
  1. Build face-adjacency graph
  2. Flood-fill: ensure neighbors traverse shared edge in OPPOSITE
     directions (consistent winding)
  3. Check signed volume sign; if negative, flip all faces so outward
     normals are correct
  4. Verify is_winding_consistent + signed-volume > 0

Usage
-----
    # Fix one mesh, write to <name>_fixed.<ext> next to the input
    python examples/fix_obj_winding.py path/to/link.obj

    # Custom output path
    python examples/fix_obj_winding.py path/to/link.obj -o /tmp/fixed.obj

    # Batch-fix all collision meshes for a URDF (writes alongside originals)
    python examples/fix_obj_winding.py path/to/robot.urdf --in-place

    # Just analyze, don't write anything (dry-run)
    python examples/fix_obj_winding.py path/to/link.obj --dry-run

Requires `trimesh` (pip install trimesh).
"""
import argparse, sys, os, re
from pathlib import Path

try:
    import trimesh
    import numpy as np
except ImportError as ex:
    print(f"ERROR: {ex}. Install trimesh: pip install trimesh", file=sys.stderr)
    sys.exit(1)


def signed_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    """∑ (1/6) p0·(p1×p2). Sign reveals overall orientation; magnitude
    matches mesh volume only if winding is consistent."""
    p0 = verts[faces[:, 0]]
    p1 = verts[faces[:, 1]]
    p2 = verts[faces[:, 2]]
    return float(np.einsum("ij,ij->i", p0, np.cross(p1, p2)).sum() / 6.0)


def winding_stats(mesh: trimesh.Trimesh) -> dict:
    return {
        "n_verts": len(mesh.vertices),
        "n_faces": len(mesh.faces),
        "consistent_winding": bool(mesh.is_winding_consistent),
        "is_volume": bool(mesh.is_volume),
        "signed_volume": signed_volume(np.asarray(mesh.vertices),
                                       np.asarray(mesh.faces)),
    }


def fix_one(input_path: Path, output_path: Path | None, dry_run: bool,
            convex_hull: bool = False) -> dict:
    """Returns {input, before, after, output, changed}."""
    # process=True merges duplicate vertices (essential for STL files which
    # store 3 verts per triangle independently — without merging there are
    # no shared edges and the winding check is vacuous).
    mesh = trimesh.load(input_path, force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"{input_path}: did not load as a single trimesh "
                         f"(got {type(mesh).__name__})")
    before = winding_stats(mesh)
    if convex_hull:
        # Replace with convex hull — guaranteed closed manifold, well-oriented.
        # Loses concavity detail but for ABD collision meshes that are
        # approximately convex (gripper fingers/knuckles, link cylinders),
        # the loss is negligible vs the gain of correct mass/centroid/inertia.
        fixed = mesh.convex_hull
    else:
        fixed = mesh.copy()
        fixed.fix_normals()
    after = winding_stats(fixed)
    changed = (
        before["consistent_winding"] != after["consistent_winding"]
        or before["is_volume"] != after["is_volume"]
        or abs(before["signed_volume"] - after["signed_volume"]) > 1e-12
    )

    saved_to = None
    if not dry_run and changed:
        if output_path is None:
            stem = input_path.stem
            ext = input_path.suffix
            output_path = input_path.parent / f"{stem}_fixed{ext}"
        # trimesh.export honors file extension
        fixed.export(str(output_path))
        saved_to = output_path

    return {
        "input": input_path,
        "before": before,
        "after": after,
        "changed": changed,
        "saved_to": saved_to,
    }


def collect_urdf_meshes(urdf_path: Path) -> list[Path]:
    """Find all <mesh filename="..."/> references in a URDF and resolve them."""
    text = urdf_path.read_text()
    base_dir = urdf_path.parent
    refs = re.findall(r'<mesh\s+filename="([^"]+)"', text)
    paths = []
    for ref in refs:
        # URDF may use package://, ROS-style. Strip and try relative.
        ref = re.sub(r"^package://[^/]+/", "", ref)
        ref = re.sub(r"^file://", "", ref)
        cand = (base_dir / ref).resolve()
        if cand.exists():
            paths.append(cand)
        else:
            print(f"  WARNING: missing mesh '{ref}' (looked at {cand})",
                  file=sys.stderr)
    return paths


def fmt_stats(s: dict) -> str:
    return (f"verts={s['n_verts']:>5d}  faces={s['n_faces']:>5d}  "
            f"winding_ok={'yes' if s['consistent_winding'] else 'NO'}  "
            f"is_volume={'yes' if s['is_volume'] else 'no'}  "
            f"V_signed={s['signed_volume']:+.4e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help=".obj / .STL mesh OR a .urdf to batch-process")
    ap.add_argument("-o", "--output", help="output path (single-mesh mode only)")
    ap.add_argument("--in-place", action="store_true",
                    help="for URDF batch mode: overwrite originals "
                         "(otherwise writes <name>_fixed.<ext>)")
    ap.add_argument("--dry-run", action="store_true",
                    help="just analyze, don't write any files")
    ap.add_argument("--convex-hull", action="store_true",
                    help="replace mesh with its convex hull. Guaranteed to "
                         "produce a closed manifold — use for ABD collision "
                         "meshes that trimesh.fix_normals() can't repair "
                         "(non-closed source). Loses concavity detail; only "
                         "appropriate when the original is approximately "
                         "convex (gripper fingers, link cylinders, etc.).")
    args = ap.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"ERROR: input not found: {input_path}", file=sys.stderr)
        sys.exit(2)

    if input_path.suffix.lower() == ".urdf":
        if args.output:
            print("ERROR: --output not supported in URDF batch mode "
                  "(use --in-place or default)", file=sys.stderr)
            sys.exit(2)
        meshes = collect_urdf_meshes(input_path)
        if not meshes:
            print("No meshes found in URDF.")
            return
        print(f"Scanning {len(meshes)} mesh(es) referenced by {input_path.name}\n")
        n_fixed = 0
        for m_path in meshes:
            try:
                # In-place means write to same path as input. Otherwise default
                # rule (<stem>_fixed.<ext> next to input) applies.
                out = m_path if args.in_place else None
                r = fix_one(m_path, out, args.dry_run, args.convex_hull)
            except Exception as ex:
                print(f"  {m_path.name}: ERROR — {type(ex).__name__}: {ex}")
                continue
            tag = "MODIFIED" if r["changed"] else "ok      "
            if r["changed"]: n_fixed += 1
            print(f"  [{tag}] {m_path.name}")
            print(f"    before: {fmt_stats(r['before'])}")
            if r["changed"]:
                print(f"    after : {fmt_stats(r['after'])}")
                if r["saved_to"]:
                    print(f"    wrote : {r['saved_to']}")
                elif args.dry_run:
                    print(f"    (dry-run — not written)")
        print(f"\n{n_fixed}/{len(meshes)} meshes needed winding fix.")
    else:
        out_path = Path(args.output).resolve() if args.output else None
        r = fix_one(input_path, out_path, args.dry_run, args.convex_hull)
        print(f"  before: {fmt_stats(r['before'])}")
        if r["changed"]:
            print(f"  after : {fmt_stats(r['after'])}")
            if r["saved_to"]:
                print(f"  wrote : {r['saved_to']}")
            else:
                print(f"  (dry-run — not written)")
        else:
            print(f"  no changes needed (winding already consistent + outward).")


if __name__ == "__main__":
    main()

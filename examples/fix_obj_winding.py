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
import argparse, sys, os, re, shutil
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


def boundary_edge_count(faces: np.ndarray) -> int:
    """Count edges that appear in != 2 triangles (= holes)."""
    from collections import Counter
    e = Counter()
    for tri in faces:
        for k in range(3):
            a, b = int(tri[k]), int(tri[(k + 1) % 3])
            e[(min(a, b), max(a, b))] += 1
    return sum(1 for c in e.values() if c != 2)


def smart_repair(verts: np.ndarray, faces: np.ndarray,
                 min_vertex_keep: float = 0.95,
                 verbose: bool = False) -> tuple[np.ndarray, np.ndarray, str]:
    """Repair a possibly-non-manifold mesh, preferring shape-preserving methods.

    Cascade (each tried in order, accepts first that succeeds):
      1. Already closed (boundary_edges == 0) → return as-is
      2. pymeshfix.fill_holes(refine=True) only — gentlest, adds new
         triangles to close boundaries, leaves rest of mesh untouched
      3. pymeshfix.clean() + fill_holes — also removes degenerate /
         self-intersecting triangles before filling
      4. pymeshfix.repair() — aggressive (may delete chunks)
      5. trimesh.convex_hull — last-resort fallback

    Acceptance criteria for steps 2-4:
      - output is closed (boundary_edges == 0)
      - output vertex count >= input * min_vertex_keep
      - output volume > 0

    Returns: (verts, faces, method_used)
    """
    n_in = len(verts)

    # Step 1: already closed?
    if boundary_edge_count(faces) == 0:
        return verts, faces, "already_closed"

    # Steps 2-4 require pymeshfix
    try:
        import pymeshfix
        v_in = np.ascontiguousarray(verts, dtype=np.float64)
        f_in = np.ascontiguousarray(faces, dtype=np.int32)

        # Step 2: fill_holes only
        try:
            fix = pymeshfix.MeshFix(v_in, f_in)
            fix.fill_holes(refine=True)
            v_out = np.asarray(fix.points, dtype=np.float64)
            f_out = np.asarray(fix.faces, dtype=np.int32)
            if (len(v_out) >= n_in * min_vertex_keep
                and boundary_edge_count(f_out) == 0
                and signed_volume(v_out, f_out) > 0):
                return v_out, f_out, "fill_holes"
            if verbose:
                print(f"    fill_holes failed: v={len(v_out)} bd={boundary_edge_count(f_out)}")
        except Exception as e:
            if verbose:
                print(f"    fill_holes raised: {e}")

        # Step 3: clean (degeneracy + intersection removal) + fill_holes
        try:
            fix = pymeshfix.MeshFix(v_in, f_in)
            fix.clean(max_iters=10, inner_loops=3)
            fix.fill_holes(refine=True)
            v_out = np.asarray(fix.points, dtype=np.float64)
            f_out = np.asarray(fix.faces, dtype=np.int32)
            if (len(v_out) >= n_in * min_vertex_keep
                and boundary_edge_count(f_out) == 0
                and signed_volume(v_out, f_out) > 0):
                return v_out, f_out, "clean+fill_holes"
            if verbose:
                print(f"    clean+fill_holes: v={len(v_out)} bd={boundary_edge_count(f_out)}")
        except Exception as e:
            if verbose:
                print(f"    clean+fill_holes raised: {e}")

        # Step 4: full repair (aggressive — may delete a lot)
        try:
            fix = pymeshfix.MeshFix(v_in, f_in)
            fix.repair()
            v_out = np.asarray(fix.points, dtype=np.float64)
            f_out = np.asarray(fix.faces, dtype=np.int32)
            if (len(v_out) >= n_in * min_vertex_keep
                and boundary_edge_count(f_out) == 0
                and signed_volume(v_out, f_out) > 0):
                return v_out, f_out, "full_repair"
            if verbose:
                print(f"    full_repair: v={len(v_out)} bd={boundary_edge_count(f_out)} (rejected)")
        except Exception as e:
            if verbose:
                print(f"    full_repair raised: {e}")
    except ImportError:
        if verbose:
            print("    pymeshfix not installed; skipping shape-preserving repair")

    # Step 5: convex hull fallback
    tm = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    hull = tm.convex_hull
    return (np.asarray(hull.vertices, dtype=np.float64),
            np.asarray(hull.faces, dtype=np.int32),
            "convex_hull")


def fix_one(input_path: Path, output_path: Path | None, dry_run: bool,
            convex_hull: bool = False,
            backup_suffix: str = ".orig",
            auto_fix: bool = False,
            min_vertex_keep: float = 0.95) -> dict:
    """Returns {input, before, after, output, changed}.

    When `output_path == input_path` (in-place rewrite), copies the
    original to ``input_path + backup_suffix`` first so a later
    `--restore` can recover. Skips backup if one already exists (so
    re-running the fix doesn't clobber the original).
    """
    # process=True merges duplicate vertices (essential for STL files which
    # store 3 verts per triangle independently — without merging there are
    # no shared edges and the winding check is vacuous).
    mesh = trimesh.load(input_path, force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"{input_path}: did not load as a single trimesh "
                         f"(got {type(mesh).__name__})")
    before = winding_stats(mesh)
    method_used = "fix_normals"
    if auto_fix:
        # Cascade: closed → fill_holes → clean+fill_holes → repair → convex hull
        v_in = np.asarray(mesh.vertices, dtype=np.float64)
        f_in = np.asarray(mesh.faces, dtype=np.int32)
        v_out, f_out, method_used = smart_repair(
            v_in, f_in, min_vertex_keep=min_vertex_keep, verbose=False
        )
        fixed = trimesh.Trimesh(vertices=v_out, faces=f_out, process=False)
    elif convex_hull:
        # Replace with convex hull — guaranteed closed manifold, well-oriented.
        # Loses concavity detail but for ABD collision meshes that are
        # approximately convex (gripper fingers/knuckles, link cylinders),
        # the loss is negligible vs the gain of correct mass/centroid/inertia.
        fixed = mesh.convex_hull
        method_used = "convex_hull"
    else:
        fixed = mesh.copy()
        fixed.fix_normals()
    after = winding_stats(fixed)
    changed = (
        before["consistent_winding"] != after["consistent_winding"]
        or before["is_volume"] != after["is_volume"]
        or abs(before["signed_volume"] - after["signed_volume"]) > 1e-12
        or len(mesh.vertices) != len(fixed.vertices)
        or len(mesh.faces) != len(fixed.faces)
    )

    saved_to = None
    backed_up = None
    if not dry_run and changed:
        if output_path is None:
            stem = input_path.stem
            ext = input_path.suffix
            output_path = input_path.parent / f"{stem}_fixed{ext}"
        # If overwriting in place, snapshot original first (idempotent —
        # don't overwrite an existing backup).
        if output_path.resolve() == input_path.resolve():
            backup = input_path.with_suffix(input_path.suffix + backup_suffix)
            if not backup.exists():
                shutil.copy2(input_path, backup)
                backed_up = backup
        # trimesh.export honors file extension
        fixed.export(str(output_path))
        saved_to = output_path

    return {
        "input": input_path,
        "before": before,
        "after": after,
        "changed": changed,
        "saved_to": saved_to,
        "backed_up": backed_up,
        "method": method_used,
    }


def restore_one(target_path: Path, backup_suffix: str = ".orig") -> bool:
    """Move <target><backup_suffix> back to <target>. Returns True if a
    backup existed and was restored, False otherwise."""
    backup = target_path.with_suffix(target_path.suffix + backup_suffix)
    if not backup.exists():
        return False
    if target_path.exists():
        target_path.unlink()
    backup.rename(target_path)
    return True


def collect_urdf_meshes(urdf_path: Path,
                        collision_only: bool = False) -> list[Path]:
    """Find <mesh filename="..."/> references in a URDF and resolve them.

    collision_only: when True, return only meshes inside <collision> blocks
        (skip <visual>). Crucial for convex-hull replacement — visual
        meshes drive rendering and must not be simplified.
    """
    text = urdf_path.read_text()
    base_dir = urdf_path.parent
    if collision_only:
        # Pull the substring inside each <collision> ... </collision> block.
        blocks = re.findall(r"<collision>(.*?)</collision>", text, flags=re.DOTALL)
        refs = []
        for blk in blocks:
            refs.extend(re.findall(r'<mesh\s+filename="([^"]+)"', blk))
    else:
        refs = re.findall(r'<mesh\s+filename="([^"]+)"', text)
    paths = []
    for ref in refs:
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
    ap.add_argument("--auto-fix", action="store_true",
                    help="Smart per-mesh repair cascade (recommended): "
                         "1) closed mesh → leave alone, "
                         "2) pymeshfix.fill_holes (preserve shape), "
                         "3) clean+fill_holes (more aggressive), "
                         "4) full repair, "
                         "5) convex hull (last-resort fallback). "
                         "First step that produces closed mesh with vertex "
                         "count >= input * --min-vertex-keep is accepted. "
                         "Requires pymeshfix (pip install pymeshfix).")
    ap.add_argument("--min-vertex-keep", type=float, default=0.95,
                    help="--auto-fix: minimum vertex retention ratio "
                         "(default 0.95 = at most 5%% loss before falling "
                         "back to the next repair stage)")
    ap.add_argument("--collision-only", action="store_true",
                    help="URDF batch mode: only process <collision> meshes, "
                         "skip <visual>. Strongly recommended with "
                         "--convex-hull (don't replace render-only assets).")
    ap.add_argument("--backup-suffix", default=".orig",
                    help="suffix for original-mesh backup written next to "
                         "each modified file (default: '.orig'). The "
                         "backup is created only when the mesh actually "
                         "changes and only if no backup already exists "
                         "(safe to re-run).")
    ap.add_argument("--restore", action="store_true",
                    help="Reverse mode: rename each <name><backup-suffix> "
                         "back to <name>, undoing a prior --convex-hull "
                         "or --fix-normals run. Works in URDF batch mode "
                         "too (restores all meshes referenced by the URDF).")
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
        meshes = collect_urdf_meshes(input_path, collision_only=args.collision_only)
        if not meshes:
            print("No meshes found in URDF.")
            return

        if args.restore:
            print(f"Restoring originals for {len(meshes)} mesh(es) "
                  f"using suffix '{args.backup_suffix}'")
            n_done = 0
            for m_path in meshes:
                if restore_one(m_path, args.backup_suffix):
                    print(f"  RESTORED: {m_path.name}")
                    n_done += 1
                else:
                    print(f"  no backup: {m_path.name}")
            print(f"\n{n_done}/{len(meshes)} files restored.")
            return

        print(f"Scanning {len(meshes)} mesh(es) referenced by {input_path.name}\n")
        n_fixed = 0
        for m_path in meshes:
            try:
                # In-place means write to same path as input. Otherwise default
                # rule (<stem>_fixed.<ext> next to input) applies.
                out = m_path if args.in_place else None
                r = fix_one(m_path, out, args.dry_run, args.convex_hull,
                            args.backup_suffix, args.auto_fix,
                            args.min_vertex_keep)
            except Exception as ex:
                print(f"  {m_path.name}: ERROR — {type(ex).__name__}: {ex}")
                continue
            tag = "MODIFIED" if r["changed"] else "ok      "
            method = r.get("method", "")
            if r["changed"]: n_fixed += 1
            print(f"  [{tag}] {m_path.name}  (method: {method})")
            print(f"    before: {fmt_stats(r['before'])}")
            if r["changed"]:
                print(f"    after : {fmt_stats(r['after'])}")
                if r["saved_to"]:
                    print(f"    wrote : {r['saved_to']}")
                elif args.dry_run:
                    print(f"    (dry-run — not written)")
        print(f"\n{n_fixed}/{len(meshes)} meshes needed winding fix.")
    else:
        if args.restore:
            ok = restore_one(input_path, args.backup_suffix)
            print(f"  RESTORED: {input_path.name}" if ok
                  else f"  no backup found ({input_path.name}{args.backup_suffix})")
            return
        out_path = Path(args.output).resolve() if args.output else None
        r = fix_one(input_path, out_path, args.dry_run, args.convex_hull,
                    args.backup_suffix, args.auto_fix,
                    args.min_vertex_keep)
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

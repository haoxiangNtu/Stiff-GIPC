"""Parse an Omniverse USD stage and translate it to StiffGIPC engine load calls.

Replaces ``rbs_physics.UsdParser`` for the StiffGIPC integration.
Uses the single-engine multi-instance pattern: only env_0 (template) is
parsed, then each body is loaded N times with per-environment spatial offsets.

Articulations are parsed directly from UsdPhysics joint schemas
(RevoluteJoint, PrismaticJoint, FixedJoint) -- no URDF file required.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from stiff_physics.engine import Engine

_approx_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


@dataclass
class SceneInfo:
    """Tracking structure returned by the parser for views and manager."""

    num_envs: int = 1
    env_offsets: list[np.ndarray] = field(default_factory=list)

    rigid_body_map: dict[str, list[int]] = field(default_factory=dict)
    deformable_body_map: dict[str, list[tuple[int, int]]] = field(default_factory=dict)

    # robot_name -> dict with keys:
    #   "joint_offset_rev": per-env start index in revolute joint array
    #   "joint_count_rev": number of revolute joints per env
    #   "joint_offset_pri": per-env start index in prismatic joint array
    #   "joint_count_pri": number of prismatic joints per env
    #   "body_offsets": per-env list of ABD body offsets for links
    #   "link_names": list of link names in load order
    articulation_map: dict[str, dict] = field(default_factory=dict)

    geometry_dict: dict[str, dict] = field(default_factory=dict)
    body_scales: dict[int, np.ndarray] = field(default_factory=dict)

    # ABD body ID -> 4x4 transform used at load time (for Fabric sync correction)
    body_init_transforms: dict[int, np.ndarray] = field(default_factory=dict)

    # ABD body ID -> {"vertices": (V,3) float64, "faces": (F,3) int32}
    # Stored at load time (already in world space) for collision mesh export.
    body_meshes: dict[int, dict] = field(default_factory=dict)


class StiffGipcUsdParser:
    """Parse USD stage and load bodies into a StiffGIPC engine."""

    ENV_PATTERN = re.compile(r"env_(\d+)$")

    MAX_COLLISION_VERTS = 500
    MAX_STATIC_COLLISION_VERTS = 5000

    # USD physics:approximation -> internal method name (same mapping as rbs-physics)
    APPROXIMATION_MAP = {
        "convexdecomposition": "coacd",
        "convexhull": "convex_hull",
        "boundingsphere": "bounding_sphere",
        "boundingcube": "bounding_box",
        "meshsimplification": "quadratic",
    }

    # Set before parse_and_build() to skip prims whose path contains any of these substrings.
    skip_prim_patterns: list[str] = []

    def __init__(self, engine: Engine, stage, debug_verbose: bool = False):
        self.engine = engine
        self.stage = stage
        self.debug_verbose = debug_verbose
        self.mesh_approximation_map: dict[str, dict] = {}

    def parse_and_build(
        self,
        env_scope_path: str = "/World/envs",
        skip_mesh_approximation: bool = False,
    ) -> SceneInfo:
        envs = self._detect_envs(env_scope_path)
        num_envs = len(envs)
        offsets = self._compute_env_offsets(envs)

        info = SceneInfo(num_envs=num_envs, env_offsets=offsets)

        # Build mesh approximation map from USD physics:approximation
        # attributes, following the same approach as rbs-physics/usd_parser.py
        if not skip_mesh_approximation:
            self._build_mesh_approximation_map()

        template = envs[0]
        template_path = template["env_path"]

        articulations = self._find_articulations(template_path)
        art_exclude_paths = set()
        for a in articulations:
            art_exclude_paths.add(a["prim_path"])
            parent_path = str(a["prim"].GetParent().GetPath()) if a["prim"].GetParent() else None
            if parent_path and parent_path != "/":
                art_exclude_paths.add(parent_path)
        if self.debug_verbose:
            print(f"[UsdParser] Articulation exclusion paths: {art_exclude_paths}", flush=True)

        rigid_bodies = self._find_rigid_bodies(template_path, exclude_under=art_exclude_paths)
        static_colliders = self._find_static_colliders(template_path, exclude_under=art_exclude_paths)
        deformables = self._find_deformable_bodies(template_path)
        has_ground = self._find_ground()

        if has_ground:
            self.engine.native.add_ground(0.0)

        for art in articulations:
            self._load_articulation_from_usd(art, offsets, info)

        if self.debug_verbose:
            print(f"[UsdParser] Loading {len(rigid_bodies)} independent rigid bodies...", flush=True)
        rb_body_ids_per_env: list[list[int]] = [[] for _ in range(num_envs)]
        kinematic_rb_ids_per_env: list[list[int]] = [[] for _ in range(num_envs)]
        for i, rb in enumerate(rigid_bodies):
            self._load_rigid_body(rb, offsets, info)
            for env_idx in range(num_envs):
                bid = info.rigid_body_map[rb["prim_path"]][env_idx]
                rb_body_ids_per_env[env_idx].append(bid)
                if rb.get("boundary_type", 0) == 1:
                    kinematic_rb_ids_per_env[env_idx].append(bid)

        art_body_ids_per_env: list[list[int]] = [[] for _ in range(num_envs)]
        for art_key, art_val in info.articulation_map.items():
            for env_idx, body_list in enumerate(art_val.get("body_offsets", [])):
                art_body_ids_per_env[env_idx].extend(body_list)

        sc_body_ids_per_env: list[list[int]] = [[] for _ in range(num_envs)]
        if static_colliders:
            if self.debug_verbose:
                print(f"[UsdParser] Loading {len(static_colliders)} static colliders...", flush=True)
            for sc in static_colliders:
                self._load_rigid_body(sc, offsets, info,
                                      max_verts=self.MAX_STATIC_COLLISION_VERTS)
                for env_idx in range(num_envs):
                    sc_body_ids_per_env[env_idx].append(
                        info.rigid_body_map[sc["prim_path"]][env_idx])

        excl_count = 0
        for env_idx in range(num_envs):
            for art_bid in art_body_ids_per_env[env_idx]:
                for sc_bid in sc_body_ids_per_env[env_idx]:
                    self.engine.add_collision_exclusion(art_bid, sc_bid)
                    excl_count += 1
                for rb_bid in kinematic_rb_ids_per_env[env_idx]:
                    self.engine.add_collision_exclusion(art_bid, rb_bid)
                    excl_count += 1
        if self.debug_verbose and excl_count:
            print(f"[UsdParser] Excluded {excl_count} per-env articulation-external pairs "
                  f"(arm vs container + arm vs kinematic rigid bodies)", flush=True)

        if deformables:
            if self.debug_verbose:
                print(f"[UsdParser] Loading {len(deformables)} deformable (FEM) bodies...", flush=True)
            for db in deformables:
                self._load_deformable_body(db, offsets, info)
                if self.debug_verbose:
                    print(f"[UsdParser]   FEM: {db['prim_path']}, "
                          f"verts={len(db['mesh']['vertices'])}, "
                          f"youngs={db['youngs_modulus']}", flush=True)

        if self.debug_verbose:
            print("[UsdParser] parse_and_build() returning", flush=True)
        return info

    # ------------------------------------------------------------------
    # Environment detection
    # ------------------------------------------------------------------

    def _detect_envs(self, scope_path: str) -> list[dict]:
        scope = self.stage.GetPrimAtPath(scope_path)
        if not scope or not scope.IsValid():
            return [{"env_id": 0, "env_path": "/", "transform": np.eye(4)}]

        envs = []
        for child in scope.GetChildren():
            m = self.ENV_PATTERN.match(child.GetName())
            if m is None:
                continue
            envs.append({
                "env_id": int(m.group(1)),
                "env_path": str(child.GetPath()),
                "transform": self._get_transform(child),
            })
        envs.sort(key=lambda e: e["env_id"])
        if not envs:
            return [{"env_id": 0, "env_path": scope_path, "transform": np.eye(4)}]
        return envs

    def _compute_env_offsets(self, envs: list[dict]) -> list[np.ndarray]:
        t0_inv = np.linalg.inv(envs[0]["transform"])
        return [e["transform"] @ t0_inv for e in envs]

    # ------------------------------------------------------------------
    # Prim type detection
    # ------------------------------------------------------------------

    def _find_rigid_bodies(self, template_path: str, exclude_under: set[str] | None = None) -> list[dict]:
        from pxr import Usd, UsdPhysics

        exclude_under = exclude_under or set()
        excluded_count = 0
        results = []
        predicate = Usd.TraverseInstanceProxies()
        for prim in self.stage.Traverse(predicate):
            path = str(prim.GetPath())
            if not path.startswith(template_path):
                continue
            if self.skip_prim_patterns and any(pat in path for pat in self.skip_prim_patterns):
                if self.debug_verbose:
                    print(f"[UsdParser] Skipping (skip_prim_patterns): {path}", flush=True)
                continue
            if self.debug_verbose and "box" in path.lower() and "/box" in path:
                has_rb = prim.HasAPI(UsdPhysics.RigidBodyAPI)
                has_col = prim.HasAPI(UsdPhysics.CollisionAPI) if hasattr(UsdPhysics, 'CollisionAPI') else False
                print(f"[UsdParser] DEBUG box prim: {path} type={prim.GetTypeName()} "
                      f"hasRigidBody={has_rb} hasCollision={has_col}", flush=True)
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                continue
            if any(path.startswith(ep + "/") or path == ep for ep in exclude_under):
                excluded_count += 1
                continue
            result = self._extract_mesh_from_prim(prim)
            if result is None:
                if self.debug_verbose:
                    print(f"[UsdParser]   WARNING: RigidBody {path} has no mesh, skipping", flush=True)
                continue
            mesh_data, mesh_prim = result

            boundary = 0
            if prim.GetAttribute("physics:kinematicEnabled"):
                ke = prim.GetAttribute("physics:kinematicEnabled").Get()
                if ke:
                    boundary = 1

            if self.debug_verbose:
                print(f"[UsdParser]   Found rigid body: {path}, boundary={boundary}", flush=True)
            results.append({
                "prim_path": path,
                "prim": prim,
                "mesh": mesh_data,
                "mesh_prim": mesh_prim,
                "mesh_prim_path": str(mesh_prim.GetPath()),
                "boundary_type": boundary,
                "transform": self._get_transform(mesh_prim),
            })
        if self.debug_verbose:
            if excluded_count > 0:
                print(f"[UsdParser] Excluded {excluded_count} rigid bodies under articulation(s)", flush=True)
            print(f"[UsdParser] Found {len(results)} independent rigid bodies", flush=True)
        return results

    def _find_static_colliders(self, template_path: str, exclude_under: set[str] | None = None) -> list[dict]:
        """Find mesh prims with CollisionAPI but no RigidBodyAPI (static collision geometry)."""
        from pxr import Usd, UsdPhysics, UsdGeom

        exclude_under = exclude_under or set()
        rigid_body_paths = set()
        for prim in self.stage.Traverse(Usd.TraverseInstanceProxies()):
            path = str(prim.GetPath())
            if path.startswith(template_path) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_body_paths.add(path)

        results = []
        seen_parents = set()
        for prim in self.stage.Traverse(Usd.TraverseInstanceProxies()):
            path = str(prim.GetPath())
            if not path.startswith(template_path):
                continue
            if self.skip_prim_patterns and any(pat in path for pat in self.skip_prim_patterns):
                if self.debug_verbose:
                    print(f"[UsdParser] Skipping static collider (skip_prim_patterns): {path}", flush=True)
                continue
            if not prim.IsA(UsdGeom.Mesh):
                continue
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if any(path.startswith(rb + "/") or path == rb for rb in rigid_body_paths):
                continue
            if any(path.startswith(ep + "/") or path == ep for ep in exclude_under):
                continue
            parent_path = str(prim.GetParent().GetPath()) if prim.GetParent() else path
            top_path = parent_path
            for rb_path in rigid_body_paths:
                if path.startswith(rb_path + "/"):
                    top_path = rb_path
                    break
            if top_path in seen_parents:
                continue
            seen_parents.add(top_path)

            mesh_data = self._read_mesh_attrs(prim)
            if mesh_data is None:
                continue

            if self.debug_verbose:
                print(f"[UsdParser]   Found static collider: {path} (boundary=1)", flush=True)
            results.append({
                "prim_path": top_path,
                "prim": prim,
                "mesh": mesh_data,
                "mesh_prim": prim,
                "mesh_prim_path": path,
                "boundary_type": 1,
                "transform": self._get_transform(prim),
            })
        if self.debug_verbose and results:
            print(f"[UsdParser] Found {len(results)} static colliders", flush=True)
        return results

    @staticmethod
    def _has_deformable_body_api(prim) -> bool:
        """Multi-level check mirroring rbs-physics DeformableBuilder.

        USD's ``GetAppliedSchemas()`` filters out schema names that are not
        in the global schema registry.  ``DeformableBodyAPI`` is a custom
        schema, so we need fallbacks.
        """
        if "DeformableBodyAPI" in prim.GetAppliedSchemas():
            return True

        from pxr import Sdf
        stage = prim.GetStage()
        for layer in stage.GetUsedLayers():
            spec = layer.GetPrimAtPath(prim.GetPath())
            if spec is None:
                continue
            op = spec.GetInfo("apiSchemas")
            if op is None:
                continue
            all_items = []
            if hasattr(op, "isExplicit") and op.isExplicit:
                all_items = list(op.explicitItems)
            else:
                all_items = list(getattr(op, "prependedItems", [])) + list(getattr(op, "appendedItems", []))
            if "DeformableBodyAPI" in all_items:
                return True

        if prim.GetAttribute("deform:youngsModulus").IsValid():
            return True
        return False

    def _find_deformable_bodies(self, template_path: str) -> list[dict]:
        from pxr import Usd, UsdGeom
        results = []
        prim_count = 0
        if self.debug_verbose:
            print(f"[UsdParser] Searching deformable bodies under {template_path}...", flush=True)
        for prim in self.stage.Traverse(Usd.TraverseInstanceProxies()):
            path = str(prim.GetPath())
            if not path.startswith(template_path):
                continue
            prim_count += 1

            if not self._has_deformable_body_api(prim):
                continue

            if self.debug_verbose:
                print(f"[UsdParser]   Deformable candidate: {path} "
                      f"isMesh={prim.IsA(UsdGeom.Mesh)}", flush=True)

            if not prim.IsA(UsdGeom.Mesh):
                if self.debug_verbose:
                    print(f"[UsdParser]   SKIP {path}: not a UsdGeom.Mesh", flush=True)
                continue

            ym_attr = prim.GetAttribute("deform:youngsModulus")
            if ym_attr and ym_attr.IsValid():
                youngs = float(ym_attr.Get()) if ym_attr.Get() is not None else 1e5
            else:
                youngs = 1e5
                if self.debug_verbose:
                    print(f"[UsdParser]   WARN {path}: no youngsModulus, default {youngs}", flush=True)

            result = self._extract_mesh_from_prim(prim)
            if result is None:
                if self.debug_verbose:
                    print(f"[UsdParser]   SKIP {path}: _extract_mesh_from_prim returned None", flush=True)
                continue
            mesh_data, mesh_prim = result

            if self.debug_verbose:
                print(f"[UsdParser]   Accepted deformable: {path} "
                      f"verts={len(mesh_data['vertices'])} youngs={youngs}", flush=True)

            results.append({
                "prim_path": path,
                "prim": prim,
                "mesh": mesh_data,
                "mesh_prim": mesh_prim,
                "youngs_modulus": youngs,
                "transform": self._get_transform(mesh_prim),
            })
        if self.debug_verbose:
            print(f"[UsdParser] Scanned {prim_count} prims under {template_path}, "
                  f"found {len(results)} deformable bodies", flush=True)
        return results

    def _find_articulations(self, template_path: str) -> list[dict]:
        from pxr import UsdPhysics

        results = []
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            if not path.startswith(template_path):
                continue
            if not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                continue

            root_fixed = True
            rif_attr = prim.GetAttribute("rbs:root_is_fixed")
            if rif_attr and rif_attr.IsValid() and rif_attr.Get() is not None:
                root_fixed = bool(rif_attr.Get())

            results.append({
                "prim_path": path,
                "prim": prim,
                "root_fixed": root_fixed,
                "transform": self._get_transform(prim),
            })
        return results

    # ------------------------------------------------------------------
    # Mesh approximation (ported from rbs-physics mesh_factory.py)
    # ------------------------------------------------------------------

    def _build_mesh_approximation_map(self):
        """Scan the stage for ``physics:approximation`` and ``rbs:coacd_threshold``.

        Builds ``self.mesh_approximation_map`` which maps prim paths to
        ``{"method": str, "params": float | None}``.
        This mirrors ``rbs_physics.usd_parser.UsdParser.parse_usd``.
        """
        from pxr import Usd, UsdPhysics

        self.mesh_approximation_map = {}
        for prim in self.stage.Traverse(Usd.TraverseInstanceProxies()):
            prim_path = str(prim.GetPath())

            has_mesh_col = prim.HasAPI(UsdPhysics.MeshCollisionAPI)
            has_rigid = prim.HasAPI(UsdPhysics.RigidBodyAPI)
            if not has_mesh_col and not has_rigid:
                continue

            approx_attr = prim.GetAttribute("physics:approximation")
            if approx_attr and approx_attr.HasAuthoredValue():
                approximation = str(approx_attr.Get()).lower()
            else:
                approximation = "convexdecomposition"

            method = self.APPROXIMATION_MAP.get(approximation)
            if method is None:
                continue

            coacd_threshold = None
            threshold_attr = prim.GetAttribute("rbs:coacd_threshold")
            if threshold_attr and threshold_attr.HasAuthoredValue():
                coacd_threshold = float(threshold_attr.Get())

            self.mesh_approximation_map[prim_path] = {
                "method": method,
                "params": coacd_threshold,
            }

        if self.debug_verbose and self.mesh_approximation_map:
            print(f"[UsdParser] Mesh approximation map: {len(self.mesh_approximation_map)} entries",
                  flush=True)
            for p, cfg in self.mesh_approximation_map.items():
                print(f"  {p}: method={cfg['method']}, params={cfg['params']}", flush=True)

    def _get_approximation_config(self, prim_path: str) -> Optional[dict]:
        """Look up the approximation config for *prim_path* or any ancestor."""
        if not self.mesh_approximation_map:
            return None
        if prim_path in self.mesh_approximation_map:
            return self.mesh_approximation_map[prim_path]
        for ancestor_path, config in self.mesh_approximation_map.items():
            if prim_path.startswith(ancestor_path + "/") or prim_path == ancestor_path:
                return config
        return None

    @staticmethod
    def _approximate_mesh(mesh_data: dict, method: str,
                          params: Optional[float] = None,
                          prim_path: str = "",
                          debug_verbose: bool = False) -> dict:
        """Apply a collision approximation method to a mesh.

        Implements the same methods as ``rbs_physics.mesh_factory.approximate_mesh``:
        coacd, convex_hull, bounding_box, bounding_sphere, quadratic.
        CoACD parts are **merged** into a single mesh (not split).

        Results are cached by ``(vertex_count, face_count, method, params)`` so
        identical meshes across cloned environments are only computed once
        (mirrors ``rbs_physics.mesh_factory._approx_cache``).
        """
        global _approx_cache
        verts = mesh_data["vertices"]
        faces = mesh_data["faces"]
        orig_nverts = len(verts)
        orig_nfaces = len(faces)

        cache_key = (orig_nverts, orig_nfaces, method, params)
        if cache_key in _approx_cache:
            cached_v, cached_f = _approx_cache[cache_key]
            if debug_verbose:
                print(f"[UsdParser] {method} [cached] ({prim_path}): "
                      f"{orig_nverts}→{len(cached_v)} verts", flush=True)
            return {"vertices": cached_v.copy(), "faces": cached_f.copy(),
                    "verts_per_face": 3}

        result = None

        if method == "coacd":
            try:
                import coacd as _coacd
                _coacd.set_log_level("off")
                cmesh = _coacd.Mesh(
                    np.asarray(verts, dtype=np.float64),
                    np.asarray(faces, dtype=np.int32),
                )
                threshold = params if params is not None else 0.05
                parts = _coacd.run_coacd(cmesh,
                                         threshold=threshold,
                                         mcts_nodes=20,
                                         mcts_iterations=30,
                                         mcts_max_depth=1,
                                         merge=True,
                                         max_convex_hull=16)
                if parts:
                    all_v, all_f, voff = [], [], 0
                    for pv, pf in parts:
                        v = np.asarray(pv, dtype=np.float64)
                        f = np.asarray(pf, dtype=np.int32)
                        all_v.append(v)
                        all_f.append(f + voff)
                        voff += len(v)
                    new_v = np.concatenate(all_v, axis=0)
                    new_f = np.concatenate(all_f, axis=0)
                    if debug_verbose:
                        print(f"[UsdParser] CoACD ({prim_path}): {orig_nverts}→{len(new_v)} verts, "
                              f"{len(parts)} parts merged, threshold={threshold}", flush=True)
                    result = {"vertices": new_v, "faces": new_f, "verts_per_face": 3}
                else:
                    if debug_verbose:
                        print(f"[UsdParser] CoACD empty for {prim_path}, falling back to convex_hull",
                              flush=True)
                    return StiffGipcUsdParser._approximate_mesh(
                        mesh_data, "convex_hull", prim_path=prim_path, debug_verbose=debug_verbose)
            except ImportError:
                if debug_verbose:
                    print(f"[UsdParser] coacd not installed, falling back to convex_hull for {prim_path}",
                          flush=True)
                return StiffGipcUsdParser._approximate_mesh(
                    mesh_data, "convex_hull", prim_path=prim_path, debug_verbose=debug_verbose)
            except Exception as e:
                if debug_verbose:
                    print(f"[UsdParser] CoACD failed for {prim_path}: {e}, falling back to convex_hull",
                          flush=True)
                return StiffGipcUsdParser._approximate_mesh(
                    mesh_data, "convex_hull", prim_path=prim_path, debug_verbose=debug_verbose)

        elif method == "convex_hull":
            try:
                import trimesh
                hull = trimesh.Trimesh(verts, faces).convex_hull
                if debug_verbose:
                    print(f"[UsdParser] Convex hull ({prim_path}): {orig_nverts}→{len(hull.vertices)} verts",
                          flush=True)
                result = {"vertices": np.asarray(hull.vertices, dtype=np.float64),
                          "faces": np.asarray(hull.faces, dtype=np.int32),
                          "verts_per_face": 3}
            except Exception as e:
                if debug_verbose:
                    print(f"[UsdParser] Convex hull failed for {prim_path}: {e}, falling back to bounding_box",
                          flush=True)
                return StiffGipcUsdParser._approximate_mesh(
                    mesh_data, "bounding_box", prim_path=prim_path, debug_verbose=debug_verbose)

        elif method == "bounding_box":
            try:
                import trimesh
                obb = trimesh.Trimesh(verts, faces).bounding_box_oriented
                result = {"vertices": np.asarray(obb.vertices, dtype=np.float64),
                          "faces": np.asarray(obb.faces, dtype=np.int32),
                          "verts_per_face": 3}
            except Exception as e:
                if debug_verbose:
                    print(f"[UsdParser] Bounding box failed for {prim_path}: {e}", flush=True)
                return mesh_data

        elif method == "bounding_sphere":
            try:
                import trimesh
                tmesh = trimesh.Trimesh(verts, faces)
                center = tmesh.bounding_sphere.primitive.center
                radius = tmesh.bounding_sphere.primitive.radius
                sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
                sphere.apply_translation(center)
                result = {"vertices": np.asarray(sphere.vertices, dtype=np.float64),
                          "faces": np.asarray(sphere.faces, dtype=np.int32),
                          "verts_per_face": 3}
            except Exception as e:
                if debug_verbose:
                    print(f"[UsdParser] Bounding sphere failed for {prim_path}: {e}", flush=True)
                return StiffGipcUsdParser._approximate_mesh(
                    mesh_data, "bounding_box", prim_path=prim_path, debug_verbose=debug_verbose)

        elif method == "quadratic":
            try:
                import trimesh
                tm = trimesh.Trimesh(verts, faces)
                simplified = tm.simplify_quadric_decimation(face_count=len(faces) // 4)
                result = {"vertices": np.asarray(simplified.vertices, dtype=np.float64),
                          "faces": np.asarray(simplified.faces, dtype=np.int32),
                          "verts_per_face": 3}
            except Exception as e:
                if debug_verbose:
                    print(f"[UsdParser] Quadratic simplification failed for {prim_path}: {e}", flush=True)
                return mesh_data

        if result is not None:
            _approx_cache[cache_key] = (result["vertices"].copy(),
                                        result["faces"].copy())
            return result

        return mesh_data

    def _find_ground(self) -> bool:
        from pxr import UsdGeom

        for prim in self.stage.Traverse():
            name = prim.GetName().lower()
            if "ground" in name or "groundplane" in name:
                return True
            if prim.IsA(UsdGeom.Plane):
                return True
        return False

    # ------------------------------------------------------------------
    # Mesh extraction
    # ------------------------------------------------------------------

    def _extract_mesh_from_prim(self, prim) -> Optional[dict]:
        """Extract **all** mesh data under *prim* and return (merged_mesh, prim).

        When a link contains multiple sub-meshes (e.g. wrist + palm + hand back),
        all meshes are merged into one.  Vertices are transformed into the
        *prim*'s local coordinate frame so callers can use ``_get_transform(prim)``
        to obtain the correct local-to-world matrix.

        Prefers meshes under a ``collisions/`` child to avoid double-counting
        visual and collision copies.
        """
        from pxr import Usd, UsdGeom

        if prim.IsA(UsdGeom.Mesh):
            data = self._read_mesh_attrs(prim)
            return (data, prim) if data is not None else None

        col_child = prim.GetPrimAtPath("collisions")
        search_root = col_child if col_child and col_child.IsValid() else prim

        all_verts = []
        all_faces = []
        voff = 0
        T_link = self._get_transform(prim)
        T_link_inv = np.linalg.inv(T_link)

        for child in Usd.PrimRange(search_root, Usd.TraverseInstanceProxies()):
            if "visuals" in str(child.GetPath()):
                continue
            if not child.IsA(UsdGeom.Mesh):
                continue
            data = self._read_mesh_attrs(child)
            if data is None:
                continue

            v_local = data["vertices"]
            T_mesh = self._get_transform(child)
            T_rel = T_link_inv @ T_mesh
            v_in_link = (T_rel[:3, :3] @ v_local.T).T + T_rel[:3, 3]

            faces = data["faces"]
            if faces.ndim == 1:
                faces = faces.reshape(-1, data["verts_per_face"])
            all_faces.append(faces + voff)
            all_verts.append(v_in_link)
            voff += len(v_local)

        if not all_verts:
            return None

        merged = {
            "vertices": np.concatenate(all_verts, axis=0),
            "faces": np.concatenate(all_faces, axis=0),
            "verts_per_face": 3,
        }
        if self.debug_verbose and len(all_verts) > 1:
            print(f"[UsdParser]   Merged {len(all_verts)} sub-meshes under {prim.GetPath()}: "
                  f"{voff} verts total", flush=True)
        return (merged, prim)

    def _read_mesh_attrs(self, mesh_prim) -> Optional[dict]:
        from pxr import UsdGeom

        mesh = UsdGeom.Mesh(mesh_prim)
        points = mesh.GetPointsAttr().Get()
        fvc = mesh.GetFaceVertexCountsAttr().Get()
        fvi = mesh.GetFaceVertexIndicesAttr().Get()
        if points is None or fvc is None or fvi is None:
            return None

        vertices = np.array(points, dtype=np.float64)
        face_vertex_counts = np.array(fvc, dtype=np.int32)
        face_vertex_indices = np.array(fvi, dtype=np.int32)

        if np.all(face_vertex_counts == 3):
            faces = face_vertex_indices.reshape(-1, 3)
            verts_per_face = 3
        elif np.all(face_vertex_counts == 4):
            faces = face_vertex_indices.reshape(-1, 4)
            verts_per_face = 4
        else:
            tris = []
            idx = 0
            for c in face_vertex_counts:
                for j in range(1, c - 1):
                    tris.append([face_vertex_indices[idx],
                                 face_vertex_indices[idx + j],
                                 face_vertex_indices[idx + j + 1]])
                idx += c
            faces = np.array(tris, dtype=np.int32)
            verts_per_face = 3

        return {
            "vertices": vertices,
            "faces": faces,
            "verts_per_face": verts_per_face,
        }

    # ------------------------------------------------------------------
    # Articulation loading from USD joint schemas
    # ------------------------------------------------------------------

    def _load_articulation_from_usd(self, art: dict, offsets: list[np.ndarray], info: SceneInfo):
        from pxr import UsdPhysics

        art_prim = art["prim"]
        art_path = art["prim_path"]
        root_fixed = art["root_fixed"]

        # Search for links/joints under the articulation prim itself first.
        # Fall back to the parent scope for Isaac Lab layouts where the
        # ArticulationRootAPI lives on a child of the robot scope.
        robot_path = art_path
        link_prims = self._collect_articulation_links(robot_path)
        joint_prims = self._collect_articulation_joints(robot_path)

        if not link_prims:
            robot_scope = art_prim.GetParent()
            robot_path = str(robot_scope.GetPath())
            link_prims = self._collect_articulation_links(robot_path)
            joint_prims = self._collect_articulation_joints(robot_path)

        if self.debug_verbose:
            print(f"[UsdParser] Loading articulation from USD: {robot_path}", flush=True)

        if not link_prims:
            print(f"[UsdParser] WARNING: No link prims found for articulation {robot_path}", flush=True)
            return

        link_names = [prim.GetName() for _, prim in link_prims]
        root_link_path = self._find_root_link(link_prims, joint_prims)

        if self.debug_verbose:
            print(f"[UsdParser] Articulation {robot_path}: {len(link_prims)} links, "
                  f"{len(joint_prims)} joints, root={root_link_path}", flush=True)

        prev_rev = self.engine.num_revolute_joints
        prev_pri = self.engine.num_prismatic_joints

        art_info = {
            "link_names": link_names,
            "joint_offset_rev": [],
            "joint_count_rev": 0,
            "joint_offset_pri": [],
            "joint_count_pri": 0,
            "body_offsets": [],
            "joint_fk_data": [],
            "link_path_order": [lp for lp, _ in link_prims],
            "root_link_path": root_link_path,
        }

        # Phase 1: Load all link meshes as instanced (one call per link, N instances)
        link_instance_results = {}  # link_path -> instanced load result
        link_mesh_data = {}         # link_path -> processed mesh data
        link_boundaries = {}        # link_path -> boundary type
        link_mesh_prim_paths = {}   # link_path -> mesh prim path

        for link_path, link_prim in link_prims:
            is_root = (link_path == root_link_path)
            boundary = 1 if (is_root and root_fixed) else 0

            result = self._extract_mesh_from_prim(link_prim)
            if result is None:
                if self.debug_verbose:
                    print(f"[UsdParser]   WARNING: No mesh for link {link_path}, skipping", flush=True)
                continue
            mesh_data, mesh_prim = result
            mesh_prim_path = str(mesh_prim.GetPath())

            if self.debug_verbose:
                print(f"[UsdParser]   Link {link_prim.GetName()}: "
                      f"mesh={mesh_prim_path}, boundary={boundary}", flush=True)

            approx_cfg = self._get_approximation_config(mesh_prim_path)
            if approx_cfg is None:
                approx_cfg = self._get_approximation_config(str(link_prim.GetPath()))
            if approx_cfg is not None:
                mesh_data = self._approximate_mesh(
                    mesh_data, approx_cfg["method"],
                    params=approx_cfg.get("params"),
                    prim_path=mesh_prim_path,
                    debug_verbose=self.debug_verbose,
                )
            else:
                mesh_data = self._simplify_mesh(mesh_data, self.MAX_COLLISION_VERTS,
                                                 debug_verbose=self.debug_verbose)

            mesh_world = self._get_transform(mesh_prim)
            transforms = [offset @ mesh_world for offset in offsets]

            inst_result = self.engine.load_mesh_instanced(
                vertices=mesh_data["vertices"],
                faces=mesh_data["faces"],
                transforms_list=transforms,
                verts_per_face=mesh_data["verts_per_face"],
                dimensions=3,
                body_type="ABD",
                young_modulus=1e8,
                boundary_type=boundary,
            )

            link_instance_results[link_path] = inst_result
            link_mesh_data[link_path] = mesh_data
            link_boundaries[link_path] = boundary
            link_mesh_prim_paths[link_path] = mesh_prim_path

            for env_idx in range(len(offsets)):
                body_id = inst_result["body_offsets"][env_idx]
                info.body_init_transforms[body_id] = transforms[env_idx]
                info.body_meshes[body_id] = {
                    "vertices": mesh_data["vertices"].copy(),
                    "faces": mesh_data["faces"].copy() if mesh_data["faces"].ndim == 2 else mesh_data["faces"].reshape(-1, mesh_data["verts_per_face"]),
                }

        # Build parent/child link maps from joints for resolve_body_id fallback.
        # Needed when intermediate links have no mesh and are skipped.
        from pxr import UsdPhysics as _UsdPhy
        child_to_parent: dict[str, str] = {}
        parent_to_children: dict[str, list[str]] = {}
        for _, jp, _ in joint_prims:
            ja = _UsdPhy.Joint(jp)
            t0, t1 = ja.GetBody0Rel().GetTargets(), ja.GetBody1Rel().GetTargets()
            if t0 and t1:
                p, c = str(t0[0]), str(t1[0])
                child_to_parent[c] = p
                parent_to_children.setdefault(p, []).append(c)

        # Phase 2: Build per-env body mappings, create joints, exclusions, geometry_dict
        for env_idx, offset in enumerate(offsets):
            env_transform = offset
            prev_rev_j = self.engine.num_revolute_joints

            path_to_body_id = {}
            abd_bodies_this_env = []

            for link_path, _ in link_prims:
                if link_path not in link_instance_results:
                    continue
                body_id = link_instance_results[link_path]["body_offsets"][env_idx]
                path_to_body_id[link_path] = body_id
                abd_bodies_this_env.append(body_id)

            art_info["body_offsets"].append(abd_bodies_this_env)
            art_info["joint_offset_rev"].append(prev_rev_j)

            n_bodies = len(abd_bodies_this_env)
            for i in range(n_bodies):
                for j in range(i + 1, n_bodies):
                    self.engine.add_collision_exclusion(abd_bodies_this_env[i],
                                                       abd_bodies_this_env[j])

            for joint_path, joint_prim, joint_type in joint_prims:
                fk_entry = self._create_joint_from_usd(
                    joint_prim, joint_type, path_to_body_id,
                    env_transform, robot_path,
                    child_to_parent, parent_to_children,
                )
                if env_idx == 0 and fk_entry is not None:
                    art_info["joint_fk_data"].append(fk_entry)

            for link_path, _ in link_prims:
                if link_path not in path_to_body_id:
                    continue
                bid = path_to_body_id[link_path]
                clone_link_path = self._remap_path(link_path, offsets, env_idx)
                asset_id = link_instance_results[link_path]["asset_id"]
                mesh_pp = link_mesh_prim_paths.get(link_path, link_path)
                clone_mesh_path = self._remap_path(mesh_pp, offsets, env_idx)
                info.geometry_dict[clone_link_path] = {
                    "type": "rigid_body",
                    "abd_body_offset": bid,
                    "asset_id": asset_id,
                    "prim_path": clone_link_path,
                    "mesh_prim_path": clone_mesh_path,
                    "instance_id": env_idx,
                    "robot_name": art_path,
                }

        total_rev = self.engine.num_revolute_joints
        total_pri = self.engine.num_prismatic_joints
        if len(offsets) > 0:
            art_info["joint_count_rev"] = (total_rev - prev_rev) // len(offsets)
            art_info["joint_count_pri"] = (total_pri - prev_pri) // len(offsets)
        info.articulation_map[art_path] = art_info

        if self.debug_verbose:
            print(f"[UsdParser] Articulation loaded: {len(link_prims)} links, "
                  f"{art_info['joint_count_rev']} revolute joints, "
                  f"{art_info['joint_count_pri']} prismatic joints per env", flush=True)

    def _collect_articulation_links(self, robot_path: str) -> list[tuple[str, object]]:
        """Find all RigidBodyAPI prims under the robot scope (these are links)."""
        from pxr import UsdPhysics

        links = []
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            if not (path.startswith(robot_path + "/") or path == robot_path):
                continue
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                links.append((path, prim))
        return links

    def _collect_articulation_joints(self, robot_path: str) -> list[tuple[str, object, str]]:
        """Find all UsdPhysics joint prims under the robot scope.
        Returns (path, prim, joint_type_str) tuples."""
        from pxr import UsdPhysics

        joints = []
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            if not (path.startswith(robot_path + "/") or path == robot_path):
                continue
            if prim.IsA(UsdPhysics.RevoluteJoint):
                joints.append((path, prim, "revolute"))
            elif prim.IsA(UsdPhysics.PrismaticJoint):
                joints.append((path, prim, "prismatic"))
            elif prim.IsA(UsdPhysics.FixedJoint):
                joints.append((path, prim, "fixed"))
            elif prim.IsA(UsdPhysics.Joint):
                joints.append((path, prim, "fixed"))
        return joints

    def _find_root_link(self, link_prims, joint_prims) -> Optional[str]:
        """Find the root link: appears as body0 but never as body1 in any joint."""
        from pxr import UsdPhysics

        body0_paths = set()
        body1_paths = set()
        for _, joint_prim, _ in joint_prims:
            joint_api = UsdPhysics.Joint(joint_prim)
            targets0 = joint_api.GetBody0Rel().GetTargets()
            targets1 = joint_api.GetBody1Rel().GetTargets()
            if targets0:
                body0_paths.add(str(targets0[0]))
            if targets1:
                body1_paths.add(str(targets1[0]))

        root_candidates = body0_paths - body1_paths
        if root_candidates:
            return next(iter(root_candidates))

        if link_prims:
            return link_prims[0][0]
        return None

    @staticmethod
    def _resolve_body_up(path: str, child_to_parent: dict, path_to_body_id: dict,
                         depth: int = 20) -> Optional[int]:
        """Walk UP the parent chain to find the nearest link with a body."""
        cur = path
        for _ in range(depth):
            if cur in path_to_body_id:
                return path_to_body_id[cur]
            cur = child_to_parent.get(cur)
            if cur is None:
                break
        return None

    @staticmethod
    def _resolve_body_down(path: str, parent_to_children: dict, path_to_body_id: dict,
                           depth: int = 20) -> Optional[int]:
        """Walk DOWN the child chain (BFS) to find the nearest link with a body."""
        if path in path_to_body_id:
            return path_to_body_id[path]
        queue = list(parent_to_children.get(path, []))
        for _ in range(depth):
            if not queue:
                break
            nxt = []
            for c in queue:
                if c in path_to_body_id:
                    return path_to_body_id[c]
                nxt.extend(parent_to_children.get(c, []))
            queue = nxt
        return None

    def _create_joint_from_usd(self, joint_prim, joint_type: str,
                               path_to_body_id: dict, env_transform: np.ndarray,
                               robot_path: str,
                               child_to_parent: Optional[dict] = None,
                               parent_to_children: Optional[dict] = None):
        """Create a StiffGIPC joint constraint from a UsdPhysics joint prim.

        Returns FK data dict for env_0 (None if joint skipped).
        """
        from pxr import UsdPhysics, Gf

        joint_api = UsdPhysics.Joint(joint_prim)
        targets0 = joint_api.GetBody0Rel().GetTargets()
        targets1 = joint_api.GetBody1Rel().GetTargets()

        if not targets0 or not targets1:
            return None

        body0_path = str(targets0[0])
        body1_path = str(targets1[0])

        parent_id = path_to_body_id.get(body0_path)
        child_id = path_to_body_id.get(body1_path)

        if parent_id is None and child_to_parent is not None:
            parent_id = self._resolve_body_up(body0_path, child_to_parent, path_to_body_id)
        if child_id is None and parent_to_children is not None:
            child_id = self._resolve_body_down(body1_path, parent_to_children, path_to_body_id)

        if parent_id is None or child_id is None or parent_id == child_id:
            return None

        local_pos0 = np.array(joint_api.GetLocalPos0Attr().Get(), dtype=np.float64)
        local_rot0 = joint_api.GetLocalRot0Attr().Get()
        local_pos1_raw = joint_api.GetLocalPos1Attr().Get()
        local_rot1 = joint_api.GetLocalRot1Attr().Get()
        local_pos1 = np.array(local_pos1_raw, dtype=np.float64) if local_pos1_raw is not None else np.zeros(3)

        body0_world = env_transform @ self._get_transform(
            self.stage.GetPrimAtPath(body0_path))

        joint_world_pos = (body0_world[:3, :3] @ local_pos0) + body0_world[:3, 3]

        rot0_mat = self._gf_quat_to_mat3(local_rot0)
        rot1_mat = self._gf_quat_to_mat3(local_rot1) if local_rot1 is not None else np.eye(3)
        joint_frame = body0_world[:3, :3] @ rot0_mat

        joint_name = joint_prim.GetName()

        # Build FK data: parent_to_joint and joint_to_child transforms
        parent_to_joint = np.eye(4)
        parent_to_joint[:3, :3] = rot0_mat
        parent_to_joint[:3, 3] = local_pos0

        joint_to_child = np.eye(4)
        joint_to_child[:3, :3] = rot1_mat.T
        joint_to_child[:3, 3] = -rot1_mat.T @ local_pos1

        fk_entry = {
            "joint_name": joint_name,
            "joint_type": joint_type,
            "body0_path": body0_path,
            "body1_path": body1_path,
            "parent_to_joint": parent_to_joint.copy(),
            "joint_to_child": joint_to_child.copy(),
            "axis_str": "X",
        }

        if joint_type == "revolute":
            rev_api = UsdPhysics.RevoluteJoint(joint_prim)
            axis_str = str(rev_api.GetAxisAttr().Get()) if rev_api.GetAxisAttr().Get() else "X"
            fk_entry["axis_str"] = axis_str

            local_axis = self._axis_str_to_vec(axis_str)
            world_axis = joint_frame @ local_axis

            lower_attr = rev_api.GetLowerLimitAttr()
            upper_attr = rev_api.GetUpperLimitAttr()
            lower = float(lower_attr.Get()) if lower_attr and lower_attr.Get() is not None else -180.0
            upper = float(upper_attr.Get()) if upper_attr and upper_attr.Get() is not None else 180.0

            lower_rad = np.deg2rad(lower)
            upper_rad = np.deg2rad(upper)

            self.engine.add_revolute_joint(
                parent_body=parent_id,
                child_body=child_id,
                world_axis=world_axis,
                joint_pos=joint_world_pos,
                lower_limit=lower_rad,
                upper_limit=upper_rad,
                initial_angle=0.0,
                name=joint_name,
            )

        elif joint_type == "prismatic":
            pri_api = UsdPhysics.PrismaticJoint(joint_prim)
            axis_str = str(pri_api.GetAxisAttr().Get()) if pri_api.GetAxisAttr().Get() else "X"
            fk_entry["axis_str"] = axis_str

            local_axis = self._axis_str_to_vec(axis_str)
            world_axis = joint_frame @ local_axis

            lower_attr = pri_api.GetLowerLimitAttr()
            upper_attr = pri_api.GetUpperLimitAttr()
            lower = float(lower_attr.Get()) if lower_attr and lower_attr.Get() is not None else 0.0
            upper = float(upper_attr.Get()) if upper_attr and upper_attr.Get() is not None else 0.04

            self.engine.add_prismatic_joint(
                parent_body=parent_id,
                child_body=child_id,
                world_center=joint_world_pos,
                world_axis=world_axis,
                lower_limit=lower,
                upper_limit=upper,
                name=joint_name,
            )

        elif joint_type == "fixed":
            world_normal = joint_frame[:, 1]
            world_bitangent = joint_frame[:, 2]

            self.engine.add_fixed_joint(
                parent_body=parent_id,
                child_body=child_id,
                world_anchor=joint_world_pos,
                world_normal=world_normal,
                world_bitangent=world_bitangent,
            )

        return fk_entry

    # ------------------------------------------------------------------
    # Loading standalone rigid bodies / deformable bodies
    # ------------------------------------------------------------------

    def _load_rigid_body(self, rb: dict, offsets: list[np.ndarray], info: SceneInfo,
                         max_verts: int | None = None):
        prim_path = rb["prim_path"]
        approx_cfg = self._get_approximation_config(prim_path)
        if approx_cfg is not None:
            mesh = self._approximate_mesh(
                rb["mesh"], approx_cfg["method"],
                params=approx_cfg.get("params"),
                prim_path=prim_path,
                debug_verbose=self.debug_verbose,
            )
        else:
            mesh = self._simplify_mesh(rb["mesh"], max_verts or self.MAX_COLLISION_VERTS,
                                     debug_verbose=self.debug_verbose)
        info.rigid_body_map[prim_path] = []

        transforms = [offset @ rb["transform"] for offset in offsets]
        result = self.engine.load_mesh_instanced(
            vertices=mesh["vertices"],
            faces=mesh["faces"],
            transforms_list=transforms,
            verts_per_face=mesh["verts_per_face"],
            dimensions=3,
            body_type="ABD",
            young_modulus=1e8,
            boundary_type=rb["boundary_type"],
        )

        for env_idx in range(len(offsets)):
            body_offset = result["body_offsets"][env_idx]
            info.rigid_body_map[prim_path].append(body_offset)
            info.body_init_transforms[body_offset] = transforms[env_idx]
            info.body_meshes[body_offset] = {
                "vertices": mesh["vertices"].copy(),
                "faces": mesh["faces"].copy() if mesh["faces"].ndim == 2 else mesh["faces"].reshape(-1, mesh["verts_per_face"]),
            }

            clone_path = self._remap_path(prim_path, offsets, env_idx)
            mesh_pp = rb.get("mesh_prim_path", prim_path)
            clone_mesh_path = self._remap_path(mesh_pp, offsets, env_idx)
            info.geometry_dict[clone_path] = {
                "type": "rigid_body",
                "abd_body_offset": body_offset,
                "asset_id": result["asset_id"],
                "prim_path": clone_path,
                "mesh_prim_path": clone_mesh_path,
                "instance_id": env_idx,
            }

    @staticmethod
    def _tetrahedralize_surface(vertices: np.ndarray, faces: np.ndarray):
        """Convert a closed surface triangle mesh to a tetrahedral volume mesh.

        Mirrors ``rbs_physics.utils.generate_tetrahedral_grid`` using the same
        TetGen parameters (order=1, mindihedral=20, minratio=1.5).
        """
        import pyvista as pv
        from tetgen import TetGen

        pv_faces = np.hstack([
            np.full((len(faces), 1), 3, dtype=np.int32),
            np.asarray(faces, dtype=np.int32),
        ]).flatten()
        pv_mesh = pv.PolyData(np.asarray(vertices, dtype=np.float64), pv_faces)
        tri_mesh = pv_mesh.triangulate()
        tet = TetGen(tri_mesh)
        tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
        grid = tet.grid
        tet_cells = grid.cells.reshape(-1, 5)[:, 1:]
        return grid.points.astype(np.float64), tet_cells.astype(np.int32)

    def _load_deformable_body(self, db: dict, offsets: list[np.ndarray], info: SceneInfo):
        prim_path = db["prim_path"]
        mesh = db["mesh"]
        info.deformable_body_map[prim_path] = []

        if mesh["verts_per_face"] == 3:
            tet_verts, tet_cells = self._tetrahedralize_surface(
                mesh["vertices"], mesh["faces"])
            if self.debug_verbose:
                print(f"[UsdParser] Tetrahedralized {prim_path}: "
                      f"{len(mesh['vertices'])} surface verts -> "
                      f"{len(tet_verts)} verts, {len(tet_cells)} tets", flush=True)
            mesh = {"vertices": tet_verts, "faces": tet_cells, "verts_per_face": 4}

        transforms = [offset @ db["transform"] for offset in offsets]
        result = self.engine.load_mesh_instanced(
            vertices=mesh["vertices"],
            faces=mesh["faces"],
            transforms_list=transforms,
            verts_per_face=mesh["verts_per_face"],
            dimensions=3,
            body_type="FEM",
            young_modulus=db["youngs_modulus"],
        )

        for env_idx in range(len(offsets)):
            vert_offset = result["vertex_offsets"][env_idx]
            vert_count = result["vertex_counts"][env_idx]
            info.deformable_body_map[prim_path].append((vert_offset, vert_count))

            clone_path = self._remap_path(prim_path, offsets, env_idx)
            info.geometry_dict[clone_path] = {
                "type": "deformable_body",
                "vertex_offset": vert_offset,
                "vertex_count": vert_count,
                "asset_id": result["asset_id"],
                "prim_path": clone_path,
                "instance_id": env_idx,
                "load_transform": transforms[env_idx].copy(),
            }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _get_transform(self, prim) -> np.ndarray:
        from pxr import UsdGeom

        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            return np.eye(4)
        mat = xformable.ComputeLocalToWorldTransform(0)
        return np.array(mat, dtype=np.float64).T  # pxr is column-major

    def _remap_path(self, path: str, offsets: list, env_idx: int) -> str:
        """Remap env_0 path to env_N.

        Only replaces the first occurrence of env_0 in the path (the
        environment scope component), avoiding false positives in prim
        names that happen to contain 'env_0'.
        """
        if env_idx == 0:
            return path
        return re.sub(r"/env_0(/|$)", f"/env_{env_idx}\\1", path, count=1)

    @staticmethod
    def _gf_quat_to_mat3(gf_quat) -> np.ndarray:
        """Convert a Gf.Quatf/Quatd/Quath to a 3x3 rotation matrix."""
        from scipy.spatial.transform import Rotation
        real = float(gf_quat.GetReal())
        imag = gf_quat.GetImaginary()
        r = Rotation.from_quat([float(imag[0]), float(imag[1]), float(imag[2]), real])
        return r.as_matrix()

    @staticmethod
    def _axis_str_to_vec(axis: str) -> np.ndarray:
        """Convert USD axis string ('X','Y','Z') to unit vector."""
        axis = axis.upper()
        if axis == "X":
            return np.array([1.0, 0.0, 0.0])
        elif axis == "Y":
            return np.array([0.0, 1.0, 0.0])
        else:
            return np.array([0.0, 0.0, 1.0])

    @staticmethod
    def _simplify_mesh(mesh_data: dict, max_verts: int, debug_verbose: bool = False) -> dict:
        """Reduce a mesh to at most *max_verts* while preserving topology.

        Uses quadric decimation first (keeps concavity / hollow interior).
        Falls back to convex hull only when decimation is unavailable or the
        mesh is too degenerate to decimate.
        """
        verts = mesh_data["vertices"]
        if len(verts) <= max_verts:
            return mesh_data
        try:
            import trimesh
            tm = trimesh.Trimesh(vertices=verts, faces=mesh_data["faces"])

            target_faces = max(max_verts, 2 * max_verts)
            if len(tm.faces) > target_faces:
                ratio = max(0.01, 1.0 - target_faces / len(tm.faces))
                try:
                    dec = tm.simplify_quadric_decimation(ratio)
                    if len(dec.vertices) > 0 and len(dec.faces) > 0:
                        if debug_verbose:
                            print(f"[UsdParser] Decimated mesh: {len(verts)}→{len(dec.vertices)} verts, "
                                  f"{len(mesh_data['faces'])}→{len(dec.faces)} faces", flush=True)
                        return {
                            "vertices": np.array(dec.vertices, dtype=np.float64),
                            "faces": np.array(dec.faces, dtype=np.int32),
                            "verts_per_face": 3,
                        }
                except Exception:
                    pass

            hull = tm.convex_hull
            if debug_verbose:
                print(f"[UsdParser] Convex hull: {len(verts)}→{len(hull.vertices)} verts", flush=True)
            return {
                "vertices": np.array(hull.vertices, dtype=np.float64),
                "faces": np.array(hull.faces, dtype=np.int32),
                "verts_per_face": 3,
            }
        except Exception as e:
            if debug_verbose:
                print(f"[UsdParser] Mesh simplification failed ({e}), using original", flush=True)
            return mesh_data

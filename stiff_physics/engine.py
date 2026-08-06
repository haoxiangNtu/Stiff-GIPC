"""Thin Python wrapper around the pystiffgipc C++ module."""

from __future__ import annotations

import os
import sys
import sysconfig
import time as _time
import numpy as np
from pathlib import Path
from typing import Optional


_INSTALLED_MODE = False

def _import_native():
    """Import pystiffgipc from the installed _native/ sub-package first,
    falling back to the development build/ directory.

    Override path via env var STIFFGIPC_NATIVE_DIR — useful when working
    in a worktree whose build/ should be used instead of the venv's
    installed _native/ (which may point at a different worktree).
    """
    global _INSTALLED_MODE

    # 0. Env-var override (worktree dev workflow) — always wins
    override = os.environ.get("STIFFGIPC_NATIVE_DIR")
    if override:
        if os.path.isdir(override) and override not in sys.path:
            sys.path.insert(0, override)
        try:
            import pystiffgipc
            return pystiffgipc
        except ImportError as exc:
            raise ImportError(
                f"STIFFGIPC_NATIVE_DIR={override} but pystiffgipc not "
                f"importable from there ({exc})") from exc

    # 1. Try installed location (wheel / pip install -e .)
    try:
        from stiff_physics._native import pystiffgipc
        _INSTALLED_MODE = True
        return pystiffgipc
    except ImportError:
        pass

    # 2. Fallback: dev build directory next to the project root.
    #
    # Keep ABI-specific build dirs ahead of the generic "build/" dir.  This
    # worktree may contain both py3.11 and py3.12 extension modules; loading a
    # stale py3.12 module from build/ against a freshly rebuilt core library can
    # corrupt SimEngineConfig layout (e.g. cuda_device is read from the wrong
    # offset).  Newton/IsaacLab py3.12 should therefore resolve build_312 first.
    _project_root = Path(__file__).resolve().parent.parent
    py_tag = f"{sys.version_info.major}{sys.version_info.minor}"
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ""
    build_dirs = [
        _project_root / f"build_{py_tag}",
        _project_root / "build",
    ]
    for _build_dir in build_dirs:
        if not _build_dir.is_dir():
            continue
        if ext_suffix and not any(_build_dir.glob(f"pystiffgipc*{ext_suffix}")):
            continue
        bd = str(_build_dir)
        if bd not in sys.path:
            sys.path.insert(0, bd)
        try:
            import pystiffgipc
            return pystiffgipc
        except ImportError:
            try:
                sys.path.remove(bd)
            except ValueError:
                pass

    raise ImportError(
        "pystiffgipc C++ module not found. "
        "Install the stiff-physics wheel, or build with: "
        "cmake -DBUILD_PYTHON_BINDINGS=ON .. && make pystiffgipc"
    )


_C = _import_native()

_PACKAGE_DATA_DIR = Path(__file__).resolve().parent / "data"


class BodyView:
    """A read-only view of one body's slice of the global vertex/face arrays.

    Lightweight — holds no data of its own; every accessor re-slices the
    engine's current state. Cheap because slicing is a numpy view, but the
    `get_vertices()` slice DOES reflect the latest engine.step() (you don't
    need to fetch a fresh BodyView each frame).

    Use `Engine.get_bodies()` / `get_abd_body(id)` / `get_fem_body(id)` to
    obtain BodyViews — don't construct directly.
    """

    __slots__ = ("_engine", "_record")

    def __init__(self, engine: "Engine", record):
        self._engine = engine
        self._record = record

    @property
    def kind(self) -> str:
        """'ABD' (rigid affine body) or 'FEM' (deformable)."""
        return "ABD" if self._record.body_type == 0 else "FEM"

    @property
    def body_id(self) -> int:
        """Index within this body's kind (e.g. 0..N-1 for ABD bodies)."""
        return self._record.body_offset

    @property
    def label(self) -> str:
        """URDF link name (e.g. 'link_base'), .obj filename, or other source label."""
        return self._record.label

    @property
    def asset_id(self) -> int:
        """Shared MeshAsset id (-1 if not from a shared asset)."""
        return self._record.asset_id

    @property
    def instance_id(self) -> int:
        """Instance index for instanced loads (0 if not instanced)."""
        return self._record.instance_id

    @property
    def vertex_offset(self) -> int:
        """Start index in engine.get_vertices()."""
        return self._record.vertex_offset

    @property
    def vertex_count(self) -> int:
        return self._record.vertex_count

    def get_vertices(self) -> np.ndarray:
        """Current deformed vertices for this body, shape (vertex_count, 3)."""
        v = self._engine.get_vertices()
        s = self._record.vertex_offset
        return v[s : s + self._record.vertex_count]

    def get_vertex_velocities(self) -> np.ndarray:
        v = self._engine.get_vertex_velocities()
        s = self._record.vertex_offset
        return v[s : s + self._record.vertex_count]

    def get_surface_faces(self, local_indices: bool = True) -> np.ndarray:
        """Surface triangles belonging to this body.

        Args:
            local_indices: when True (default), vertex refs are 0-based for
                this body so faces directly index into self.get_vertices().
                Set False to keep global indices that match
                engine.get_vertices().
        """
        all_faces = self._engine.get_surface_faces()
        s = self._record.vertex_offset
        e = s + self._record.vertex_count
        if self._record.vertex_count == 0:
            return np.empty((0, all_faces.shape[1]), dtype=all_faces.dtype)
        in_range = ((all_faces >= s) & (all_faces < e)).all(axis=1)
        body_faces = all_faces[in_range]
        if local_indices:
            body_faces = (body_faces - s).astype(all_faces.dtype, copy=False)
        return body_faces

    def __repr__(self) -> str:
        return (f"BodyView(kind={self.kind} body_id={self.body_id} "
                f"label='{self.label}' verts={self.vertex_count})")


# ---------------------------------------------------------------------------
# Multi-env execution modes. Three tiers, each a superset of the previous:
#   "merged"   (A) — baseline v0.6.x: all envs in one merged solve. Fastest, but
#                    NO per-env isolation (cross-env contact possible, merged-bbox
#                    dHat). Correct only for single-env or well-separated envs.
#   "isolated" (B) — per-env DECOUPLED physics: env-id collision isolation, per-env
#                    BVH, per-env κ (absolute dHat), per-env line-search + segmented
#                    per-env PCG. Each env is a physically-correct independent sim.
#                    Does NOT guarantee bit-identical / batch-invariant results.
#   "strict"   (C) — isolated + deterministic kernels/canonical contact ordering.
#                    Guarantees repeatability for a fixed scene and batch layout.
#                    Cross-batch-size identity remains a validation target, not a
#                    public contract. Slowest.
# The mode resolves to the low-level STIFF_* flags below (which stay available as
# per-feature debug overrides). An explicitly-set STIFF_* env var always wins
# (setdefault). STIFF_MULTIENV_MODE overrides the Config field.
_MULTIENV_ISOLATED_FLAGS = [
    "STIFF_BVH_ENVDET",     # env-id collision isolation (no cross-env contact)
    "STIFF_PERENV_BVH",     # per-env broad-phase BVH
    "STIFF_DECOUPLE_THRESH",  # absolute dHat + per-group κ gating (+ N-invariant meanMass)
    "STIFF_PERGROUP_KAPPA",   # per-env barrier/friction/init κ
    "STIFF_SEGMENTED_PCG",  # block-diagonal per-env PCG (own α/β/convergence)
    "STIFF_PERENV_ALPHA",   # per-env line-search α
    # K-stream concurrent per-env BVH build+query (perenv-parallel #2). QUALIFIED FOR STRICT
    # 2026-07-04: run-to-run + cross-env + batch-size N=2/4/8 all 0.000 with PAR on (the
    # canon/order-free stack absorbs the stream-interleaved pair emission). =0 opts out.
    "STIFF_PERENV_PAR",
]
_MULTIENV_STRICT_EXTRA = [
    "STIFF_EE_CANON",       # canonical edge-edge emission order
    "STIFF_EE_DETGATE",     # deterministic EE dedup gate
    "STIFF_CCD_CANON",      # canonical CCD emission order
    "STIFF_SPMV_DET",       # deterministic (order-independent) SpMV
]
_MULTIENV_ISOLATED_ONLY = []   # (empty: PAR qualified for strict and moved into the base list)
_MULTIENV_MODE_ALIASES = {
    "0": "merged", "merged": "merged", "a": "merged",
    "1": "isolated", "isolated": "isolated", "decoupled": "isolated", "b": "isolated",
    "2": "strict", "strict": "strict", "deterministic": "strict", "c": "strict",
}


# STIFF_* controls are process-global and several native hot paths cache them on
# first use.  Track the values the wrapper owns so repeated Engines in the SAME
# mode are idempotent, and lock the native mode after the first Engine: changing
# modes in one process used to mix cached and live flags and can corrupt CUDA
# graph execution.  Cross-mode comparisons must use subprocesses.
_MODE_FLAGS_SET_BY_US: dict = {}
_PEE_FLAGS_SET_BY_US: dict = {}
_PROCESS_MULTIENV_MODE: str | None = None
_PROCESS_PER_ENV_EXIT: bool | None = None
_PROCESS_MODE_SIGNATURE: tuple[tuple[str, str | None], ...] | None = None
_PER_ENV_EXIT_FLAGS = (
    "STIFF_DECOUPLE_THRESH",
    "STIFF_PERENV_ALPHA",
    "STIFF_PERENV_MASK",
    "STIFF_PERENV_TELEM",
)
_MODE_SIGNATURE_KEYS = tuple(
    sorted(
        set(
            _MULTIENV_ISOLATED_FLAGS
            + _MULTIENV_STRICT_EXTRA
            + _MULTIENV_ISOLATED_ONLY
            + list(_PER_ENV_EXIT_FLAGS)
            + ["STIFF_EE_LB", "STIFF_PERENV_MASK_DEV"]
        )
    )
)


def _env_enabled(key: str) -> bool:
    """Return whether a low-level feature flag is explicitly enabled."""
    value = os.environ.get(key)
    return value is not None and value != "" and not value.startswith("0")


def _mode_signature() -> tuple[tuple[str, str | None], ...]:
    return tuple((key, os.environ.get(key)) for key in _MODE_SIGNATURE_KEYS)


def _assert_process_mode_signature() -> None:
    if (
        _PROCESS_MODE_SIGNATURE is not None
        and _mode_signature() != _PROCESS_MODE_SIGNATURE
    ):
        raise _C.LifecycleError(
            "process-scoped STIFF_* mode flags changed after the first "
            "Engine was created. Restore them or start a new subprocess."
        )


def _setdefault_tracked(registry: dict, key: str, value: str) -> None:
    import os
    if key not in os.environ:
        os.environ[key] = value
        registry[key] = value


def _retract_our_flags(registry: dict, keep: set = frozenset()) -> None:
    import os
    for key, val in list(registry.items()):
        if key in keep:
            continue
        if os.environ.get(key) == val:   # untouched since we set it
            del os.environ[key]
        del registry[key]                # user changed it -> it is theirs now


def resolve_multienv_mode(mode: str = "merged") -> str:
    """Set the STIFF_* env flags for the requested multi-env mode (setdefault, so
    explicit STIFF_* env vars win). STIFF_MULTIENV_MODE overrides `mode`. Returns
    the canonical mode name. Idempotent for repeated Engines in one mode.

    Native kernels cache some STIFF_* values, so :class:`Engine` rejects a
    cross-mode switch in the same process before calling this resolver."""
    import os
    raw = os.environ.get("STIFF_MULTIENV_MODE", mode)
    canon = _MULTIENV_MODE_ALIASES.get(str(raw).strip().lower())
    if canon is None:
        raise ValueError(
            f"unknown multienv_mode {raw!r}; use merged/isolated/strict (or 0/1/2)")
    flags = []
    if canon == "isolated":
        flags = _MULTIENV_ISOLATED_FLAGS + _MULTIENV_ISOLATED_ONLY
    elif canon == "strict":
        flags = _MULTIENV_ISOLATED_FLAGS + _MULTIENV_STRICT_EXTRA
    mode_keep = set(flags) | ({"STIFF_EE_LB"} if canon == "merged" else set())
    _retract_our_flags(_MODE_FLAGS_SET_BY_US, keep=mode_keep)
    for f in flags:
        _setdefault_tracked(_MODE_FLAGS_SET_BY_US, f, "1")
    # [0.8.2] determinism is POSITIVE-gated in the binary now (strict sets STIFF_SPMV_DET above;
    # merged/isolated run the fast paths by default) — the old STIFF_FAST_GRAD hint is no longer
    # read by the engine and is not set anymore.
    # merged defaults the selfQuery_ee lb2 variant.  The 2026-08-04 sm_89 DLTO
    # image links baseline/lb2 at the same 106 registers (so the old 128-reg
    # explanation was stale), but a frozen-state 100-query test still measured
    # lb2 -1.69% for EE-DCD. NOT defaulted for strict; isolated measured only
    # -0.59% in that microtest and prior end-to-end trials classified it as
    # noise. Both modes can opt in explicitly.
    if canon == "merged":
        _setdefault_tracked(_MODE_FLAGS_SET_BY_US, "STIFF_EE_LB", "2")
    return canon


class Config:
    """Simulation configuration mirroring gipc::SimEngineConfig."""

    def __init__(
        self,
        dt: float = 0.01,
        density: float = 1e3,
        young_modulus: float = 1e7,
        poisson_rate: float = 0.49,
        friction_rate: float = 0.4,
        # Ground friction coefficient. None (default) = follow friction_rate —
        # the historic behavior. Pass a value to decouple ground friction from
        # object-object friction (the A2 "gd_friction_rate wrapper bug" fix:
        # previously this could only be set by poking cfg._cfg after
        # construction).
        gd_friction_rate: float | None = None,
        newton_tol: float = 1e-2,
        # [uipc-style, opt-in] physical Newton exit: max step displacement <=
        # newton_velocity_tol * dt (m/s; uipc default 0.05). 0 = legacy
        # newton_tol*length*dt exit (exact current behavior). Scene-size and
        # env-count independent; makes relative_dhat inert for the exit check.
        newton_velocity_tol: float = 0.0,
        # 1e-6 (was 1e-4, inherited from upstream, never tuned): with absolute-dhat
        # kappa (correct contact stiffness) loose PCG directions explode Newton
        # counts (measured 1407 vs 507 total Newton over 30f at 1e-4 vs 1e-6 on the
        # multi-env grasp scene; net time strictly worse at 1e-4). Stiff-contact
        # scenes may benefit from 1e-8 (env STIFF_PCG_TOL or this arg).
        pcg_tol: float = 1e-4,  # [0.8.2] back to the 0.6.x default; 1e-6 cost ~22% at N=1 for no accuracy need
        relative_dhat: float = 1e-3,
        absolute_dhat: float = 0.0,
        joint_strength_ratio: float = 100.0,
        revolute_driving_strength_ratio: float = 100.0,
        semi_implicit_enabled: bool = False,
        semi_implicit_beta_tol: float = 1e-3,
        semi_implicit_min_iter: int = 1,
        newton_iter_cap: int = 1000,
        # Optional shared merged/per-env energy acceptance band. Defaults to
        # strict non-increase (E1 <= E0); set nonzero values explicitly to use
        # E1 <= E0 + energy_abs_tol + energy_rel_tol*abs(E0).
        energy_abs_tol: float = 0.0,
        energy_rel_tol: float = 0.0,
        skip_all_collision: bool = False,
        preconditioner_type: int = 1,
        cuda_device: int = 0,
        assets_dir: str = "",
        prismatic_strength_ratio: float = 100.0,
        prismatic_driving_strength_ratio: float = 100.0,
        max_revolute_step_per_frame: float = 0.1,
        max_prismatic_step_per_frame: float = 0.002,
        gravity: tuple[float, float, float] = (0.0, -9.8, 0.0),
        ground_normal: tuple[float, float, float] = (0.0, 1.0, 0.0),
        ground_offset: float = -1.0,
        velocity_damping: float = 0.0,
        multienv_mode: str = "merged",
        # Initial DCD pair-buffer capacity multiplier. Overflow self-heals (the
        # engine grows the buffers 1.5x and redoes detection), so this only
        # tunes how often the grow-redo cost is paid at startup vs memory used.
        collision_detection_buff_scale: float = 6.0,
        # [per-env exit] Productized switch for the per-env decoupled Newton
        # exit: each env converges by its OWN criterion and is frozen/masked
        # out (resources released) instead of being coupled to the batch.
        # Resolves to STIFF_DECOUPLE_THRESH + STIFF_PERENV_ALPHA +
        # STIFF_PERENV_MASK at Engine() time (explicit env vars still win).
        # Only meaningful for multi-env scenes.
        per_env_exit: bool = False,
        **kwargs,
    ):
        # Multi-env execution tier: "merged" (baseline) / "isolated" (per-env decoupled,
        # not bit-identical) / "strict" (fixed-layout repeatable). Resolved to
        # STIFF_* flags by Engine(). STIFF_MULTIENV_MODE env var overrides this.
        self.multienv_mode = multienv_mode
        self.per_env_exit = per_env_exit
        self._cfg = _C.Config()
        self._cfg.dt = dt
        self._cfg.density = density
        self._cfg.young_modulus = young_modulus
        self._cfg.poisson_rate = poisson_rate
        self._cfg.friction_rate = friction_rate
        self._cfg.gd_friction_rate = (friction_rate if gd_friction_rate is None
                                      else gd_friction_rate)
        self._cfg.newton_tol = newton_tol
        if hasattr(self._cfg, "newton_velocity_tol"):
            self._cfg.newton_velocity_tol = newton_velocity_tol
        self._cfg.pcg_tol = pcg_tol
        self._cfg.relative_dhat = relative_dhat
        # absolute_dhat>0 makes dHat = absolute_dhat^2 (fixed), independent of the
        # full-scene bbox → consistent contact across num_envs (no cross-env coupling).
        if hasattr(self._cfg, "absolute_dhat"):
            self._cfg.absolute_dhat = absolute_dhat
        self._cfg.joint_strength_ratio = joint_strength_ratio
        self._cfg.revolute_driving_strength_ratio = revolute_driving_strength_ratio
        self._cfg.prismatic_strength_ratio = prismatic_strength_ratio
        self._cfg.prismatic_driving_strength_ratio = prismatic_driving_strength_ratio
        if hasattr(self._cfg, "max_revolute_step_per_frame"):
            self._cfg.max_revolute_step_per_frame = max_revolute_step_per_frame
        if hasattr(self._cfg, "max_prismatic_step_per_frame"):
            self._cfg.max_prismatic_step_per_frame = max_prismatic_step_per_frame
        self._cfg.semi_implicit_enabled = semi_implicit_enabled
        self._cfg.semi_implicit_beta_tol = semi_implicit_beta_tol
        self._cfg.semi_implicit_min_iter = semi_implicit_min_iter
        self._cfg.newton_iter_cap = newton_iter_cap
        if energy_abs_tol < 0.0 or energy_rel_tol < 0.0:
            raise ValueError("energy tolerances must be non-negative")
        self._cfg.energy_abs_tol = energy_abs_tol
        self._cfg.energy_rel_tol = energy_rel_tol
        self._cfg.skip_all_collision = skip_all_collision
        self._cfg.preconditioner_type = preconditioner_type
        self._cfg.cuda_device = cuda_device
        self._cfg.collision_detection_buff_scale = collision_detection_buff_scale
        self._cfg.velocity_damping = velocity_damping
        if not assets_dir and _INSTALLED_MODE and _PACKAGE_DATA_DIR.is_dir():
            self._cfg.assets_dir = str(_PACKAGE_DATA_DIR) + "/"
        else:
            self._cfg.assets_dir = assets_dir
        import numpy as np
        self._cfg.gravity = np.array(gravity, dtype=np.float64)
        self._cfg.ground_normal = np.array(ground_normal, dtype=np.float64)
        self._cfg.ground_offset = ground_offset

        for k, v in kwargs.items():
            if hasattr(self._cfg, k):
                setattr(self._cfg, k, v)

    @property
    def native(self) -> _C.Config:
        return self._cfg

    def __repr__(self) -> str:
        return f"Config(dt={self._cfg.dt}, density={self._cfg.density})"


class Engine:
    """High-level Python interface to the StiffGIPC simulation engine.

    Lifecycle::

        engine = Engine(config)
        engine.load_urdf("path/to/robot.urdf", scale=0.3)
        engine.finalize()
        for _ in range(1000):
            engine.step()
            verts = engine.get_vertices()
    """

    def __init__(self, config: Optional[Config] = None):
        self._config = config or Config()
        # Resolve the multi-env mode → STIFF_* flags BEFORE any load/finalize/step, so the
        # gated engine paths (finalize meanMass, per-frame κ/BVH/PCG) see them. Env vars win.
        requested_mode = os.environ.get(
            "STIFF_MULTIENV_MODE",
            getattr(self._config, "multienv_mode", "merged"),
        )
        canonical_mode = _MULTIENV_MODE_ALIASES.get(
            str(requested_mode).strip().lower()
        )
        if canonical_mode is None:
            raise ValueError(
                f"unknown multienv_mode {requested_mode!r}; "
                "use merged/isolated/strict (or 0/1/2)"
            )
        requested_per_env_exit = bool(
            getattr(self._config, "per_env_exit", False)
        )
        global _PROCESS_MULTIENV_MODE
        global _PROCESS_PER_ENV_EXIT
        global _PROCESS_MODE_SIGNATURE
        if (
            _PROCESS_MULTIENV_MODE is not None
            and canonical_mode != _PROCESS_MULTIENV_MODE
        ):
            raise _C.LifecycleError(
                "multi-environment mode is process-scoped: this process "
                f"already initialized {_PROCESS_MULTIENV_MODE!r}, then requested "
                f"{canonical_mode!r}. Run different modes in separate subprocesses."
            )
        if (
            _PROCESS_PER_ENV_EXIT is not None
            and requested_per_env_exit != _PROCESS_PER_ENV_EXIT
        ):
            raise _C.LifecycleError(
                "per_env_exit is process-scoped: this process already "
                f"initialized per_env_exit={_PROCESS_PER_ENV_EXIT}, then "
                f"requested {requested_per_env_exit}. Use a separate subprocess."
            )
        if (
            _PROCESS_MODE_SIGNATURE is not None
            and _mode_signature() != _PROCESS_MODE_SIGNATURE
        ):
            raise _C.LifecycleError(
                "process-scoped STIFF_* mode flags changed after the first "
                "Engine was created. Restore them or start a new subprocess."
            )
        self.multienv_mode = resolve_multienv_mode(canonical_mode)
        if requested_per_env_exit:
            # [per-env exit] productized switch — see Config docstring. Tracked
            # setdefault so explicitly-set env vars (incl. "0" overrides) still
            # win. The process lock above rejects a later Engine with a
            # different per_env_exit setting.
            for flag in _PER_ENV_EXIT_FLAGS:
                _setdefault_tracked(_PEE_FLAGS_SET_BY_US, flag, "1")
            # telemetry (per-env iters/status, NaN quarantine) lives in the host
            # S1 path — make it part of the productized switch. Set
            # STIFF_PERENV_TELEM=0 explicitly to opt back into the zero-D2H
            # device fast path (no telemetry).
        else:
            # First Engine with per_env_exit=False retracts only wrapper-owned
            # setup left by an explicit resolver call; user flags remain.
            _retract_our_flags(_PEE_FLAGS_SET_BY_US)
        # [audit] half-configuration traps (documented in the engine): warn
        # loudly instead of running with silently-degraded semantics.
        if _env_enabled("STIFF_DECOUPLE_THRESH") and not _env_enabled("STIFF_PERENV_ALPHA"):
            print("[stiff-physics][WARN] STIFF_DECOUPLE_THRESH without "
                  "STIFF_PERENV_ALPHA: per-env freezing cannot run, so the Newton "
                  "loop runs to its iteration cap EVERY frame (worse than default). "
                  "Set both, or use multienv_mode='isolated'/'strict'.")
        if _env_enabled("STIFF_PERGROUP_KAPPA") and not _env_enabled("STIFF_DECOUPLE_THRESH"):
            print("[stiff-physics][WARN] STIFF_PERGROUP_KAPPA without "
                  "STIFF_DECOUPLE_THRESH: per-group kappa is only a broadcast of "
                  "the global kappa (stub) — environments are NOT kappa-isolated.")
        signature = _mode_signature()
        native_engine = _C.SimEngine()
        native_engine.set_config(self._config.native)
        native_engine.init_cuda()
        self._engine = native_engine
        # Commit the process lock only after native construction and CUDA
        # initialization succeed. A failed constructor must not poison all
        # later Engine attempts in this Python process.
        _PROCESS_MULTIENV_MODE = canonical_mode
        _PROCESS_PER_ENV_EXIT = requested_per_env_exit
        _PROCESS_MODE_SIGNATURE = signature
        self._finalized = False

    @property
    def native(self) -> _C.SimEngine:
        return self._engine

    def load_urdf(
        self,
        urdf_path: str,
        scale: float = 1.0,
        translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        root_fixed: bool = True,
        revolute_as_motor: bool = False,
        default_young: float = 1e7,
        initial_joint_angles: dict[str, float] | None = None,
    ) -> None:
        """Load a URDF robot model into the scene.

        Args:
            initial_joint_angles: Optional dict mapping joint names to angles
                (radians). When provided, the arm is loaded at the FK target
                pose instead of the zero pose, avoiding pass-through collisions.
        """
        transform = np.eye(4)
        transform[:3, :3] *= scale
        transform[0, 3] = translation[0]
        transform[1, 3] = translation[1]
        transform[2, 3] = translation[2]

        resolved = urdf_path
        if not os.path.isabs(resolved):
            assets = self._engine.get_assets_dir()
            candidate = os.path.join(assets, resolved)
            if os.path.exists(candidate):
                resolved = candidate

        self._engine.load_urdf(resolved, transform, root_fixed,
                               revolute_as_motor, default_young,
                               initial_joint_angles or {})

    _BODY_TYPE_MAP = {"ABD": 0, "abd": 0, "FEM": 1, "fem": 1}
    _BOUNDARY_MAP  = {"Free": 0, "free": 0, "Fixed": 1, "fixed": 1,
                      "Motor": 2, "motor": 2, "Animated": 3, "animated": 3}

    def load_mesh(
        self,
        mesh_path: str,
        dimensions: int = 3,
        body_type: str | int = "FEM",
        transform: Optional[np.ndarray] = None,
        young_modulus: float = 1e7,
        boundary_type: str | int = "Free",
        density: float | None = None,
    ) -> None:
        """Load a raw mesh (.msh for 3D tet, .obj for 2D cloth/shell).

        Args:
            mesh_path: Path to mesh file, resolved against assets_dir if relative.
            dimensions: 2 for triangle shell/cloth, 3 for tet volume.
            body_type: "ABD" (0) for rigid or "FEM" (1) for deformable.
            transform: 4x4 transformation matrix (identity if None).
            young_modulus: Young's modulus for this body.
            boundary_type: "Free" (0) or "Fixed" (1).
            density: Per-body density override (kg/m^3 for 3D FEM, surface
                density for cloth). None = global Config.density /
                cloth_density. FEM/cloth bodies only.
        """
        if transform is None:
            transform = np.eye(4)

        bt = self._BODY_TYPE_MAP.get(body_type, body_type) if isinstance(body_type, str) else body_type
        bb = self._BOUNDARY_MAP.get(boundary_type, boundary_type) if isinstance(boundary_type, str) else boundary_type

        resolved = mesh_path
        if not os.path.isabs(resolved):
            assets = self._engine.get_assets_dir()
            candidate = os.path.join(assets, resolved)
            if os.path.exists(candidate):
                resolved = candidate

        self._engine.load_mesh(resolved, dimensions, bt, transform, young_modulus, bb)
        if density is not None:
            # The body just loaded is the last load record.
            self._engine.set_soft_body_density(
                len(self._engine.get_all_load_records()) - 1, float(density))

    def load_mesh_from_data(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        verts_per_face: int = 3,
        dimensions: int = 3,
        body_type: str | int = "FEM",
        transform: Optional[np.ndarray] = None,
        young_modulus: float = 1e7,
        boundary_type: str | int = "Free",
    ) -> None:
        """Load mesh from in-memory vertex/face arrays (no file I/O)."""
        if transform is None:
            transform = np.eye(4)
        bt = self._BODY_TYPE_MAP.get(body_type, body_type) if isinstance(body_type, str) else body_type
        bb = self._BOUNDARY_MAP.get(boundary_type, boundary_type) if isinstance(boundary_type, str) else boundary_type
        verts = np.ascontiguousarray(vertices, dtype=np.float64)
        fcs = np.ascontiguousarray(faces, dtype=np.int32)
        self._engine.load_mesh_from_data(verts, fcs, verts_per_face,
                                         dimensions, bt, transform,
                                         young_modulus, bb)

    def load_mesh_instanced(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        transforms_list: list[np.ndarray],
        verts_per_face: int = 3,
        dimensions: int = 3,
        body_type: str | int = "ABD",
        young_modulus: float = 1e7,
        boundary_type: str | int = "Free",
    ) -> dict:
        """Load one mesh as N instances with different transforms.

        Args:
            vertices: (V, 3) float64 rest-pose vertex positions.
            faces: (F, vpf) int32 face index array.
            transforms_list: List of N (4, 4) float64 transform matrices.
            verts_per_face: vertices per face (3 for tri, 4 for tet).
            dimensions: 2 for shell, 3 for volume.
            body_type: "ABD" / "FEM" (or 0 / 1).
            young_modulus: Young's modulus.
            boundary_type: "Free" / "Fixed" (or 0 / 1).

        Returns:
            dict with keys: body_offsets, vertex_offsets, vertex_counts, asset_id
        """
        bt = self._BODY_TYPE_MAP.get(body_type, body_type) if isinstance(body_type, str) else body_type
        bb = self._BOUNDARY_MAP.get(boundary_type, boundary_type) if isinstance(boundary_type, str) else boundary_type
        verts = np.ascontiguousarray(vertices, dtype=np.float64)
        fcs = np.ascontiguousarray(faces, dtype=np.int32)
        tf_list = [np.ascontiguousarray(t, dtype=np.float64) for t in transforms_list]

        result = self._engine.load_mesh_instanced(
            verts, fcs, verts_per_face, dimensions, bt,
            tf_list, young_modulus, bb
        )
        return {
            "body_offsets":   list(result.body_offsets),
            "vertex_offsets": list(result.vertex_offsets),
            "vertex_counts":  list(result.vertex_counts),
            "asset_id":       result.asset_id,
        }

    def get_mesh_asset(self, asset_id: int):
        """Return the MeshAsset object for a given asset_id."""
        return self._engine.get_mesh_asset(asset_id)

    @property
    def mesh_asset_count(self) -> int:
        return self._engine.get_mesh_asset_count()

    def add_collision_exclusion(self, body_a: int, body_b: int) -> None:
        self._engine.add_collision_exclusion(body_a, body_b)

    def set_body_groups(self, groups) -> None:
        """Set per-collision-body group (environment) id, length collision_body_num
        (ABD bodies first [0,n_abd), then FEM [n_abd, n_abd+n_fem)).

        Bodies in different groups (both >= 0) never collide — folded into the
        collision-skip matrix at finalize(), so spatially-tiled environments are
        guaranteed isolated regardless of spacing. Non-negative ids must be dense
        ``0..N-1`` and N must not exceed 256. ``-1`` is a merged-mode wildcard;
        isolated/strict require every collision body to belong to a group. Invalid
        declarations fail at finalize() instead of silently degrading physics.
        """
        self._engine.set_body_groups([int(g) for g in groups])

    def set_vertex_env_ids(self, env_ids) -> None:
        """Set per-VERTEX env id, length = engine vertex count (ABD-body vertices in
        load order, then FEM particles).

        The broad-phase skips any contact pair whose two vertices carry different
        (>= 0) env ids — cross-env contact isolation WITHOUT spatial separation,
        applied uniformly to FEM particles and ABD-body vertices. Unlike
        set_body_groups (per-body, folded into the O(body^2) skip matrix at
        finalize), this handles a single FEM body whose particles span many envs,
        and is decoupled from the block-diagonal solve (contact filtering only).
        env id < 0 = shared geometry that collides with every env. May be called
        any time after finalize(); the device array is (re)uploaded on each call.
        The underlying CUDA symbol is process-scoped, so only one Engine may own
        this optional filter at a time. Pass an empty sequence to release it.
        """
        self._engine.set_vertex_env_ids([int(e) for e in env_ids])

    def add_ground_collision_skip(self, body_id: int) -> None:
        self._engine.add_ground_collision_skip(body_id)

    def add_hybrid_fem_body(self, hybrid_data, transform=None,
                            target_abd_body_offset=None):
        """Load a hybrid ABD-FEM tet mesh produced by tools/build_hybrid_mesh.py.

        Parameters
        ----------
        hybrid_data : str | dict | numpy.lib.npyio.NpzFile
            Either a path to a .npz file produced by build_hybrid_mesh.py,
            or an already-loaded mapping with the same field names
            (vertices, tets, vertex_abd_body_id, vertex_local_pos, density,
            young_modulus, poisson_ratio, ...).
        transform : np.ndarray of shape (4,4), optional
            World-frame transform applied to the FEM mesh on load.  Pass
            this when the .npz vertices are in a per-body local frame and
            you need to place the hybrid mesh somewhere specific in the
            world.  Default: identity (.npz vertices used as-is).
        target_abd_body_offset : int, optional
            If given, ALL pin body_ids in the .npz are shifted by this
            offset before being passed to the engine.  Useful when the
            .npz was authored against a different ABD body indexing.
            Default: 0 (use .npz body_ids verbatim).

        Returns
        -------
        fem_body_offset : int
            The vertex_offset of the loaded FEM body (use this to convert
            local FEM vertex indices to global if you need them later).
        """
        if isinstance(hybrid_data, (str, os.PathLike)):
            data = np.load(str(hybrid_data))
        else:
            data = hybrid_data

        verts = np.ascontiguousarray(data["vertices"], dtype=np.float64)
        tets = np.ascontiguousarray(data["tets"], dtype=np.int32)
        v_body = np.ascontiguousarray(data["vertex_abd_body_id"], dtype=np.int32)
        v_lo = np.ascontiguousarray(data["vertex_local_pos"], dtype=np.float64)
        young = float(data["young_modulus"])
        # density/poisson currently unused by the FEM loader (engine derives
        # mass from volume & a fixed density); kept here for completeness.

        T = np.eye(4) if transform is None else np.asarray(transform, dtype=np.float64)

        # Add FEM body via load_mesh_from_data.  Tet mesh: verts (N,3),
        # tets (M,4), pass as-is (engine binding expects faces shape[0]==M
        # and a separate verts_per_face int).  body_type='FEM', initial
        # boundary_type='Free' — pinned verts are upgraded to 'Fixed' in
        # finalize() via the add_fem_pins_with_local_pos → existing M1 path.
        n_v = verts.shape[0]
        n_t = tets.shape[0]
        self.load_mesh_from_data(
            verts, tets, verts_per_face=4, dimensions=3,
            body_type="FEM", transform=T, young_modulus=young,
            boundary_type="Free",
        )
        rec = self.get_load_records()[-1]
        body_offset_v = rec.vertex_offset

        # Build pin arrays — only rigid verts (vertex_abd_body_id >= 0).
        rigid_mask = v_body >= 0
        if not np.any(rigid_mask):
            print(f"[add_hybrid_fem_body] no rigid verts in npz — pure FEM body")
            return body_offset_v
        rigid_local_idx = np.nonzero(rigid_mask)[0].astype(np.int32)
        fem_global_ids = (rigid_local_idx + body_offset_v).astype(np.int32)
        body_ids_arr = v_body[rigid_mask].astype(np.int32)
        if target_abd_body_offset is not None:
            body_ids_arr = body_ids_arr + int(target_abd_body_offset)
        local_pos_arr = np.ascontiguousarray(v_lo[rigid_mask], dtype=np.float64)

        self._engine.add_fem_pins_with_local_pos(
            fem_global_ids, body_ids_arr, local_pos_arr)

        n_rigid = int(rigid_mask.sum())
        n_iface_t = int((data["tet_region"] == 1).sum())
        n_rigid_t = int((data["tet_region"] == 2).sum())
        print(f"[add_hybrid_fem_body] {n_v}v/{n_t}t loaded; "
              f"{n_rigid} rigid pins to bodies {sorted(set(body_ids_arr.tolist()))}; "
              f"{n_iface_t} interface tets, {n_rigid_t} rigid-internal tets")
        return body_offset_v

    def add_stitch_spring(self, fem_vertex_id: int, abd_anchor_vertex_id: int,
                          abd_body_id: int,
                          rest_offset_world=(0.0, 0.0, 0.0)) -> None:
        """Stitch a FEM vertex to an ABD body via a soft-spring constraint.

        At each step the FEM vertex is pulled toward
            target_world = abd_anchor_vertex_world_pos + rest_offset_world

        For best behavior under ABD body rotation, place the FEM vertex
        coincident with the ABD anchor vertex at finalize time and pass
        rest_offset_world = (0, 0, 0). Then the spring naturally tracks
        both translation and rotation of the ABD body.

        Spring stiffness is governed by Config.soft_motion_rate.

        Must be called BEFORE finalize().
        """
        self._engine.add_stitch_spring(
            int(fem_vertex_id), int(abd_anchor_vertex_id), int(abd_body_id),
            tuple(float(x) for x in rest_offset_world))

    # ---- libuipc-style per-face orient labels (pre-finalize) ----

    def set_abd_body_face_orient(self, body_id: int, orient) -> bool:
        """Override per-face orientation for an ABD surface body.

        `orient` is one int per triangle in {-1, 0, +1}: -1 flips that
        face's normal at integration time (mass / centroid / inertia),
        0 or +1 leaves it untouched. Topology (face vertex order) is
        NOT modified — collision / BVH / render code see the original.

        Must be called BEFORE finalize() (after that, surface mesh data
        has already been copied to ABDSystem and changes are ignored).
        Returns True on success, False if body_id has no surface body.
        """
        arr = np.ascontiguousarray(np.asarray(orient, dtype=np.int32))
        return self._engine.set_abd_body_face_orient(body_id, arr.tolist())

    def get_abd_body_face_orient(self, body_id: int) -> np.ndarray:
        """Read current per-face orient labels (empty if none set)."""
        return np.asarray(self._engine.get_abd_body_face_orient(body_id),
                          dtype=np.int32)

    def get_abd_surface_body_vertices(self, body_id: int) -> np.ndarray:
        """Pre-finalize: per-body local surface vertices. Shape (N, 3) float64."""
        return self._engine.get_abd_surface_body_vertices(body_id)

    def get_abd_surface_body_triangles(self, body_id: int) -> np.ndarray:
        """Pre-finalize: per-body local triangle indices. Shape (M, 3) int32."""
        return self._engine.get_abd_surface_body_triangles(body_id)

    def label_face_orient_for_abd_body(self, body_id: int, method: str = "flood_fill") -> int:
        """Compute per-face orient labels for an ABD body and write them
        into the engine. Must be called BEFORE finalize().

        method: only 'flood_fill' supported currently. Builds edge
            adjacency, BFS-propagates winding consistency from face 0
            (handles multi-component meshes by restarting from each
            unvisited face), then checks signed volume sign — if
            negative, flips the orient label of every face so that
            outward normal convention is restored.

        Returns: number of faces marked as inverted (orient = -1).
        Returns 0 if no fix was needed (mesh already correctly wound)
        or if body_id has no surface body.
        """
        from stiff_physics.mesh_utils import compute_face_orient_flood_fill
        verts = self.get_abd_surface_body_vertices(body_id)
        faces = self.get_abd_surface_body_triangles(body_id)
        if verts.size == 0 or faces.size == 0:
            return 0
        orient = compute_face_orient_flood_fill(verts, faces)
        self.set_abd_body_face_orient(body_id, orient)
        return int((orient == -1).sum())

    def add_fixed_joint(self, parent_body: int, child_body: int,
                        world_anchor, world_normal, world_bitangent) -> int:
        """Create a fixed joint between two ABD bodies. Must call before finalize().

        Parent/child IDs may be in either numeric order; the engine canonicalizes
        Hessian storage internally. Invalid or identical body IDs raise an error.
        """
        import numpy as np
        a = np.asarray(world_anchor, dtype=np.float64).ravel()
        n = np.asarray(world_normal, dtype=np.float64).ravel()
        b = np.asarray(world_bitangent, dtype=np.float64).ravel()
        return self._engine.add_fixed_joint(parent_body, child_body, a, n, b)

    def add_revolute_joint(self, parent_body: int, child_body: int,
                           world_axis, joint_pos,
                           lower_limit: float, upper_limit: float,
                           initial_angle: float = 0.0,
                           name: str = "",
                           passive: bool = False) -> int:
        """Create a revolute joint between two ABD bodies. Must call before finalize().

        Parent/child IDs may be in either numeric order; the engine canonicalizes
        Hessian storage internally. Invalid or identical body IDs raise an error.

        passive=True -> a FREE hinge (no position servo; limits still enforced)
        for articulated objects like doors/scissors. Default False keeps the
        historic hold-at-initial-angle behavior.
        """
        import numpy as np
        ax = np.asarray(world_axis, dtype=np.float64).ravel()
        p = np.asarray(joint_pos, dtype=np.float64).ravel()
        return self._engine.add_revolute_joint(parent_body, child_body,
                                               ax, p, lower_limit, upper_limit,
                                               initial_angle, name, passive)

    def add_prismatic_joint(self, parent_body: int, child_body: int,
                            world_center, world_axis,
                            lower_limit: float, upper_limit: float,
                            name: str = "",
                            passive: bool = False) -> int:
        """Create a prismatic joint between two ABD bodies. Must call before finalize().

        Parent/child IDs may be in either numeric order; the engine canonicalizes
        Hessian storage internally. Invalid or identical body IDs raise an error.

        passive=True -> a free slider (no position servo; limits still act)."""
        import numpy as np
        c = np.asarray(world_center, dtype=np.float64).ravel()
        ax = np.asarray(world_axis, dtype=np.float64).ravel()
        return self._engine.add_prismatic_joint(parent_body, child_body,
                                                c, ax, lower_limit, upper_limit,
                                                name, passive)

    def set_vertex_boundary(self, vertex_index: int, boundary_type: int) -> None:
        """Set per-vertex boundary type (0=Free, 1=Fixed). Must call before finalize()."""
        self._engine.set_vertex_boundary(vertex_index, boundary_type)

    def set_vertex_boundaries(self, indices, boundary_type: int) -> None:
        """Batch version of set_vertex_boundary."""
        for idx in indices:
            self._engine.set_vertex_boundary(int(idx), boundary_type)

    @property
    def abd_body_count(self) -> int:
        return self._engine.get_abd_body_count()

    @property
    def fem_body_count(self) -> int:
        return self._engine.get_fem_body_count()

    @property
    def vertex_count_host(self) -> int:
        """Vertex count on host (available before finalize)."""
        return self._engine.get_vertex_count_host()

    def get_vertex_position_host(self, idx: int) -> tuple[float, float, float]:
        """Read a single vertex position from host memory (before finalize)."""
        return self._engine.get_vertex_position_host(idx)

    def finalize(self) -> None:
        """Finalize the scene: compute FEM data, upload to GPU, build BVH.

        Only one finalized Engine may be active in a process because legacy
        solver buffers are still published through process-global CUDA symbols.
        Reset or destroy it before finalizing another Engine.
        """
        _assert_process_mode_signature()
        self._engine.finalize()
        self._finalized = True

    def step(self) -> None:
        """Advance simulation by one timestep (dt).

        After :meth:`prepare_gpu_rl` this becomes a thin asynchronous
        enqueue of the recorded RL graph (Isaac ``simulate()`` semantics):
        host-set joint targets are published to the device action slab and
        the graph launch returns without any host wait.  Reads that go
        through device memory on the engine stream are ordered behind the
        queued frames; call :meth:`synchronize_gpu_rl` for an explicit
        barrier.  :meth:`end_gpu_rl` restores the ordinary host-driven
        transaction.
        """
        _assert_process_mode_signature()
        # [prepare-timeline] env-gated test knobs, zero-cost when unset.
        # STIFF_AUTO_PREPARE_AT=k: before the k-th step() of this Engine,
        # call prepare_gpu_rl() (self-contained); afterwards every step()
        # thin-routes and synchronizes so per-frame timing stays honest.
        # An ineligible scene prints the refusal reason once and stays on
        # the host path. STIFF_MS_DUMP=path: np.save per-step wall ms.
        _auto_at = os.environ.get("STIFF_AUTO_PREPARE_AT")
        _ms_dump = os.environ.get("STIFF_MS_DUMP")
        _t0 = _time.perf_counter() if _ms_dump else None
        if _auto_at is not None:
            _n = getattr(self, "_auto_prepare_count", 0)
            self._auto_prepare_count = _n + 1
            if _n == int(_auto_at) and not getattr(
                self, "_auto_prepare_done", False
            ):
                self._auto_prepare_done = True
                try:
                    _tp = _time.perf_counter()
                    self._engine.prepare_gpu_rl()
                    print(
                        f"[auto-prepare] engaged at step {_n} "
                        f"({(_time.perf_counter() - _tp) * 1000:.0f}ms)",
                        flush=True,
                    )
                    self._auto_prepared = True
                except Exception as exc:  # noqa: BLE001 - report & fall back
                    print(f"[auto-prepare] REFUSED: {exc}", flush=True)
        self._engine.step()
        if getattr(self, "_auto_prepared", False):
            self._engine.synchronize_gpu_rl()
            # Health audit: a replay that overflows or no-ops must not be
            # reported as a fast frame. Read the one-frame status packet.
            if not hasattr(self, "_rl_status_reader"):
                import ctypes

                _abi = self.get_gpu_rl_device_abi()
                _rt = ctypes.CDLL("libcudart.so")
                _buf = (ctypes.c_byte * int(_abi["status_bytes"]))()

                def _read_status():
                    _rt.cudaMemcpy(
                        ctypes.byref(_buf),
                        ctypes.c_void_p(int(_abi["statuses"])),
                        ctypes.c_size_t(len(_buf)),
                        ctypes.c_int(2),
                    )
                    import struct

                    b = bytes(_buf)
                    return (
                        struct.unpack_from("<i", b, 0)[0],
                        struct.unpack_from("<I", b, 8)[0],
                        struct.unpack_from("<i", b, 32)[0],
                    )  # (result, invalid_bits, error_code)

                self._rl_status_reader = _read_status
                self._rl_status_fail = 0
                self._rl_status_frames = 0
                import atexit

                atexit.register(
                    lambda: print(
                        f"[auto-prepare] health: "
                        f"{self._rl_status_frames - self._rl_status_fail}"
                        f"/{self._rl_status_frames} frames ok, "
                        f"{self._rl_status_fail} failed",
                        flush=True,
                    )
                )
            self._rl_status_frames += 1
            _res, _inv, _ec = self._rl_status_reader()
            if _res != 0:
                # Boundary protocol (design doc §5): a capacity overflow is
                # the rare event where the CPU re-enters — the failed frame
                # did not commit, so end residency, re-prepare (whose
                # internal training frame re-runs THIS frame's targets on
                # the host at the grown capacity) and continue thin.
                self._rl_status_fail += 1
                _n_rec = getattr(self, "_rl_recoveries", 0)
                if _res == 1 and _ec == 1 and _n_rec < 64:
                    self._rl_recoveries = _n_rec + 1
                    _tr = _time.perf_counter()
                    self._engine.end_gpu_rl()
                    del self._rl_status_reader
                    # Contact-class tiers only grow through the step
                    # transaction's OVF-required feedback (the machinery the
                    # foldshirt 1551-frame audit proved), so replay the
                    # failed frame through ONE step-transaction frame — it
                    # rolls back, grows from the actual requireds with the
                    # escalation streak, and completes the physics — then
                    # re-capture at the grown tiers and resume thin.
                    _graph_knobs = (
                        "STIFF_FRAME_GRAPH",
                        "STIFF_FRAME_FULL_GRAPH",
                        "STIFF_C4_COLLISION_GRAPH",
                        "STIFF_C6_ABD_STEP_GRAPH",
                    )
                    _saved = {k: os.environ.get(k) for k in _graph_knobs}
                    for k in _graph_knobs:
                        os.environ[k] = "1"
                    try:
                        self._engine.step()
                    finally:
                        for k, v in _saved.items():
                            if v is None:
                                os.environ.pop(k, None)
                            else:
                                os.environ[k] = v
                    self._engine.prepare_gpu_rl()
                    print(
                        f"[auto-prepare] OVF recovery #{_n_rec + 1} at "
                        f"rl-frame {self._rl_status_frames} "
                        f"(inv=0x{_inv:x}, "
                        f"{(_time.perf_counter() - _tr) * 1000:.0f}ms)",
                        flush=True,
                    )
                else:
                    print(
                        f"[auto-prepare] frame FAILED (no recovery): "
                        f"result={_res} invalid=0x{_inv:x} "
                        f"error_code={_ec}",
                        flush=True,
                    )
        if _t0 is not None:
            _log = getattr(self, "_ms_dump_log", None)
            if _log is None:
                _log = self._ms_dump_log = []
                import atexit

                atexit.register(
                    lambda: np.save(
                        _ms_dump, np.asarray(self._ms_dump_log)
                    )
                )
            _log.append((_time.perf_counter() - _t0) * 1000.0)
        # [release gate] STIFF_ITER_LOG=1: per-frame Newton-iteration telemetry
        # for ANY example without touching the example (peak / anomaly audits).
        if os.environ.get("STIFF_ITER_LOG"):
            try:
                total = self._engine.get_total_newton_iters()
            except AttributeError:
                return
            prev = getattr(self, "_iterlog_prev", 0)
            fr = getattr(self, "_iterlog_frame", 0)
            print(f"[iterlog] fr={fr} newton={total - prev}", flush=True)
            self._iterlog_prev, self._iterlog_frame = total, fr + 1

    def launch_episode_async(
        self,
        frames: int,
        revolute_actions=None,
        prismatic_actions=None,
    ) -> None:
        """Launch a device-resident RL episode without per-frame host waits.

        A normal :meth:`step` must run first to train lazy CUDA workspaces.
        Each non-empty action array has shape ``(frames, joints, 3)`` and
        stores target, strength, and external torque/force respectively.
        Use :meth:`wait_episode_observation` or poll
        :meth:`episode_observation_ready` before reading either observation
        slot, then call :meth:`finish_episode`.
        """
        _assert_process_mode_signature()
        self._engine.launch_episode_async(
            int(frames), revolute_actions, prismatic_actions
        )

    def episode_in_flight(self) -> bool:
        """Return whether an asynchronous episode still needs finishing."""
        return bool(self._engine.episode_in_flight())

    def episode_observation_ready(self, slot: int) -> bool:
        """Non-blocking readiness query for pinned observation slot 0 or 1."""
        return bool(self._engine.episode_observation_ready(int(slot)))

    def wait_episode_observation(self, slot: int) -> None:
        """Wait only for observation slot 0 or 1 and its CUDA event fence."""
        self._engine.wait_episode_observation(int(slot))

    def get_episode_observation(self, slot: int):
        """Return one ready episode observation slot.

        The result contains ``first_frame``, ``positions``, ``velocities``,
        and per-frame ``statuses``. Position and velocity arrays have shape
        ``(slot_frames, vertices, 3)``.
        """
        return self._engine.get_episode_observation(int(slot))

    def get_episode_attempted_frame_count(self) -> int:
        """Return the number of frames published by ready observation slots."""
        return int(self._engine.get_episode_attempted_frame_count())

    def finish_episode(self) -> int:
        """Wait for the terminal slot and return the successful frame count."""
        return int(self._engine.finish_episode())

    # ---- GPU-native RL device ABI ----

    def prepare_gpu_rl(self) -> None:
        """Capture the reusable one-frame GPU-native RL graph.

        This is a setup boundary and may allocate, capture and synchronize;
        run warm-up :meth:`step` calls with representative contact first so
        the capacity tiers observe realistic peaks.  No environment knobs
        are required: when the process never enabled STIFF_FRAME_GRAPH the
        engine forces the capacity layout on for its own lifecycle and runs
        one internal training step before capturing (so prepare advances the
        simulation by one frame in that case).  Afterwards either keep
        calling :meth:`step` (now a thin async enqueue) or drive the packed
        float64 action buffers from :meth:`get_gpu_rl_device_abi` directly
        in device memory with :meth:`launch_gpu_rl_async` — the steady
        state performs no host synchronization and its graph contains zero
        H2D/D2H nodes.  :meth:`launch_episode_async` stays locked out until
        :meth:`end_gpu_rl`.
        """
        _assert_process_mode_signature()
        self._engine.prepare_gpu_rl()

    def prepare_gpu_rl_episode(self, frames: int) -> None:
        """Capture a multi-frame GPU-native RL episode.

        After one warm-up :meth:`step`, this allocates/captures a fixed-size
        device episode.  Write the returned action slab directly on device,
        then call :meth:`launch_gpu_rl_episode_async` once; the graph's outer
        conditional loop consumes all ``frames`` without a host launch per
        frame.  ``end_gpu_rl`` remains the explicit teardown boundary.
        """
        _assert_process_mode_signature()
        self._engine.prepare_gpu_rl_episode(int(frames))

    def launch_gpu_rl_async(self, cuda_stream: int = 0) -> None:
        """Enqueue one RL simulation step without any host wait.

        Action writes and observation reads must be issued on this same
        CUDA stream (0 = the engine's per-thread default stream); repeated
        launches must keep using the stream chosen first.
        """
        self._engine.launch_gpu_rl_async(int(cuda_stream))

    def launch_gpu_rl_episode_async(self, cuda_stream: int = 0) -> None:
        """Launch the prepared multi-frame GPU-native episode once."""
        self._engine.launch_gpu_rl_episode_async(int(cuda_stream))

    def gpu_rl_prepared(self) -> bool:
        """Return whether the GPU-native RL graph is captured and armed."""
        return bool(self._engine.gpu_rl_prepared())

    def gpu_rl_ready(self) -> bool:
        """Non-blocking completion query; not part of the steady RL loop."""
        return bool(self._engine.gpu_rl_ready())

    def synchronize_gpu_rl(self) -> None:
        """Block until the last launched step finished (debug/teardown)."""
        self._engine.synchronize_gpu_rl()

    def end_gpu_rl(self) -> None:
        """Synchronize, release episode resources and re-enable step()."""
        self._engine.end_gpu_rl()

    def launch_gpu_rl_reset_async(self, cuda_stream: int = 0) -> None:
        """Enqueue an in-stream reset to the prepare-time state snapshot.

        Pure device copies on the bound stream — no host synchronization.
        The next launched step rebuilds collision state in its own prologue.
        Episode bookkeeping (the device frame counter, any reward
        accumulators) is deliberately left to the caller's policy.
        """
        self._engine.launch_gpu_rl_reset_async(int(cuda_stream))

    def launch_gpu_rl_reset_masked_async(
        self, env_mask_device_ptr: int, cuda_stream: int = 0
    ) -> None:
        """Selective per-env reset: envs whose int32 mask entry (device
        memory, indexed by env group id) is nonzero snap back to the
        prepare-time snapshot; other envs are untouched. Pure device-side --
        the mask can be written by a device done-flag kernel, so an RL loop
        resets finished envs with zero host transfers."""
        self._engine.launch_gpu_rl_reset_masked_async(
            int(env_mask_device_ptr), int(cuda_stream)
        )

    def get_gpu_rl_device_abi(self) -> dict:
        """Return raw device pointers and the audited graph ABI.

        Keys include ``revolute_actions``/``prismatic_actions`` (packed
        ``(joints, 3)`` float64: target, strength, external torque/force),
        ``positions``/``velocities`` (``(vertices, 3)`` for the one-frame API
        or ``(frames, vertices, 3)`` for a multi-frame capture, float64 in
        engine-internal order), ``statuses`` (one ``status_bytes`` packet per
        captured frame), ``frame_counter`` (int64), ``episode_frame_count``,
        the audited
        ``graph_nodes``/``graph_h2d``/``graph_d2h`` counts,
        ``joint_observations``/``joint_observation_count`` (float64 block
        written by the graph itself: {angle, rate} per revolute driving
        joint then {displacement, rate} per prismatic driving joint), and
        the multi-env handles ``point_to_group`` (int32 per vertex, -1 =
        wildcard), ``env_quarantined`` (int32 per env, nonzero = poisoned)
        and ``env_count``.
        """
        return dict(self._engine.get_gpu_rl_device_abi())

    def gpu_rl_tensors(self) -> dict:
        """Zero-copy torch views of the GPU-native RL device buffers.

        Returns a dict of ``torch.Tensor`` objects aliasing the ABI device
        memory from :meth:`get_gpu_rl_device_abi` — no ``.numpy()`` staging,
        no copies: ``actions_revolute``/``actions_prismatic`` (``(frames,
        joints, 3)`` float64, squeezed to ``(joints, 3)`` for the one-frame
        API; write targets here instead of memcpy), ``positions``/
        ``velocities`` (``(vertices, 3)`` float64, or with a leading frames
        dim for multi-frame captures), ``joint_observations`` (float64,
        {angle, rate} per revolute then {displacement, rate} per prismatic),
        ``statuses`` (``(frames, status_bytes)`` uint8) and
        ``frame_counter`` (int64 scalar).

        Stream discipline: the engine enqueues on CUDA's per-thread default
        stream.  Torch work on its default (legacy) stream is implicitly
        ordered against it; if you use a non-default torch stream, call
        :meth:`synchronize_gpu_rl` before reading observations.
        """
        import torch  # local import: torch is an optional dependency

        abi = self.get_gpu_rl_device_abi()
        frames = max(1, int(abi.get("episode_frame_count", 1)))

        class _DeviceArray:
            def __init__(self, ptr: int, shape: tuple, typestr: str):
                self.__cuda_array_interface__ = {
                    "shape": tuple(shape),
                    "typestr": typestr,
                    "data": (int(ptr), False),
                    "version": 3,
                    "strides": None,
                }

        def view(key: str, shape: tuple, typestr: str = "<f8"):
            return torch.as_tensor(
                _DeviceArray(abi[key], shape, typestr), device="cuda"
            )

        def framed(shape: tuple) -> tuple:
            return shape if frames == 1 else (frames, *shape)

        rev = int(abi["revolute_joints"])
        pri = int(abi["prismatic_joints"])
        verts = int(abi["vertices"])
        out = {
            "positions": view("positions", framed((verts, 3))),
            "velocities": view("velocities", framed((verts, 3))),
            "joint_observations": view(
                "joint_observations",
                (int(abi["joint_observation_count"]),),
            ),
            "statuses": view(
                "statuses", (frames, int(abi["status_bytes"])), "|u1"
            ),
            "frame_counter": view("frame_counter", (1,), "<i8"),
        }
        if rev:
            out["actions_revolute"] = view(
                "revolute_actions", framed((rev, 3))
            )
        if pri:
            out["actions_prismatic"] = view(
                "prismatic_actions", framed((pri, 3))
            )
        return out

    def set_log_level(self, level: int) -> None:
        """Control per-frame solver log verbosity.

        0 = silent (suppress solver banner, Newton-iteration, Kappa,
        ``average time cost``, timer breakdown and one-time setup prints);
        >=1 = verbose (default).  Use 0 for co-simulation / piped tooling that
        needs a clean stdout.
        """
        self._engine.set_log_level(int(level))

    def reset(self) -> None:
        """Tear down the entire world and return to a fresh empty state.

        Frees all loaded bodies (FEM/ABD), constraints and GPU buffers while
        keeping the current :class:`Config`.  After ``reset()`` re-run the usual
        ``load_mesh()`` / ``load_urdf()`` / ``add_*`` + :meth:`finalize` sequence
        in the same process — no need to recreate the Engine.
        """
        self._engine.reset()
        self._finalized = False

    def save_checkpoint(self, path: str | os.PathLike) -> None:
        """Atomically save a versioned, checksummed frame-boundary checkpoint.

        The destination is replaced only after the complete checkpoint has
        reached disk. Invalid/non-finite live state raises ``CheckpointError``.
        """
        if not self._finalized:
            raise _C.LifecycleError(
                "save_checkpoint requires a finalized Engine"
            )
        self._engine.save_checkpoint(os.fspath(path))

    def load_checkpoint(self, path: str | os.PathLike) -> None:
        """Load a checkpoint for this exact scene/material/mode configuration.

        Format, checksum, topology and finite-value validation all finish
        before live GPU state is changed.
        """
        if not self._finalized:
            raise _C.LifecycleError(
                "load_checkpoint requires a finalized Engine"
            )
        self._engine.load_checkpoint(os.fspath(path))

    # ---- State queries ----

    def get_frame_status(self):
        """Return the latest Phase-C frame-boundary status packet.

        The object has one stable layout for legacy, graph-fallback and
        whole-frame graph execution; inspect ``path_flags`` to distinguish the
        path actually taken.
        """
        return self._engine.get_frame_status()

    def get_vertices(self) -> np.ndarray:
        """Return vertex positions as (N, 3) float64 array."""
        return self._engine.get_vertices()

    def get_vertices_device_ptr(self) -> int:
        """[gpu-direct] Raw CUDA device pointer (int) to the (N,3) float64 vertex
        buffer, N = :meth:`get_vertex_count`. Wrap with a zero-copy GPU array
        (e.g. ``warp.array(ptr=..., dtype=wp.vec3d, length=N)``) to read FEM
        vertex positions without a host round-trip. Valid after finalize()."""
        return self._engine.get_vertices_device_ptr()

    def get_vertex_velocities(self) -> np.ndarray:
        """Return vertex velocities as (N, 3) float64 array."""
        return self._engine.get_vertex_velocities()

    def set_vertex_positions_gpu(self, positions: np.ndarray) -> None:
        """Write vertex positions to GPU from (N, 3) float64."""
        self._engine.set_vertex_positions_gpu(
            np.ascontiguousarray(positions, dtype=np.float64))

    def set_vertex_velocities_gpu(self, velocities: np.ndarray) -> None:
        """Write vertex velocities to GPU from (N, 3) float64.

        .. warning::
            This writes the velocity buffer ONLY — it does NOT rebuild the
            inertial prediction (``xTilta``), so a bare velocity write does not
            move the body on the next step. To set a full kinematic state
            (positions and/or velocities) use :meth:`teleport_fem_vertices`,
            which rebuilds the prediction consistently.
        """
        self._engine.set_vertex_velocities_gpu(
            np.ascontiguousarray(velocities, dtype=np.float64))

    def teleport_fem_vertices(self, positions: np.ndarray,
                              velocities: Optional[np.ndarray] = None) -> None:
        """Teleport FEM vertices: writes new positions to _vertexes (current),
        o_vertexes (previous-step committed), and xTilta (predictor). Use this
        instead of set_vertex_positions_gpu for handoff/reset, otherwise the
        next engine.step() reverts to the stale rest pose via xTilta.

        If ``velocities`` is provided, also writes velocities and extends
        xTilta = x + v*dt + g*dt^2 so inertia is preserved across handoff.
        Default (None) zeros velocities per teleport_abd_bodies semantics.
        """
        pos = np.ascontiguousarray(positions, dtype=np.float64)
        if velocities is None:
            self._engine.teleport_fem_vertices(pos)
        else:
            self._engine.teleport_fem_vertices(
                pos,
                np.ascontiguousarray(velocities, dtype=np.float64))

    def get_surface_faces(self) -> np.ndarray:
        """Return surface triangle indices as (F, 3) uint32 array."""
        return self._engine.get_surface_faces()

    def get_surface_vertex_indices(self) -> np.ndarray:
        """Return indices of surface vertices as (S,) uint32 array."""
        return self._engine.get_surface_vertex_indices()

    @property
    def vertex_count(self) -> int:
        return self._engine.get_vertex_count()

    @property
    def surface_face_count(self) -> int:
        return self._engine.get_surface_face_count()

    # ---- ABD body state ----

    def get_abd_body_transforms(self, body_offsets: np.ndarray) -> np.ndarray:
        """Return (N, 4, 4) float64 transforms for ABD bodies at given offsets."""
        offsets = np.ascontiguousarray(body_offsets, dtype=np.int32)
        return self._engine.get_abd_body_transforms(offsets)

    def set_abd_body_transforms(self, body_offsets: np.ndarray,
                                transforms: np.ndarray) -> None:
        """Set ABD body transforms from (N, 4, 4) float64."""
        offsets = np.ascontiguousarray(body_offsets, dtype=np.int32)
        tfs = np.ascontiguousarray(transforms, dtype=np.float64)
        self._engine.set_abd_body_transforms(offsets, tfs)

    def teleport_abd_bodies(self, body_offsets: np.ndarray,
                            transforms: np.ndarray) -> None:
        """Teleport ABD bodies: sets q/q_prev/q_tilde/q_temp and zeros velocity.

        Use this instead of set_abd_body_transforms for init/reset to avoid
        phantom velocities from stale q_prev.
        """
        offsets = np.ascontiguousarray(body_offsets, dtype=np.int32)
        tfs = np.ascontiguousarray(transforms, dtype=np.float64)
        self._engine.teleport_abd_bodies(offsets, tfs)

    def get_abd_body_velocities(self, body_offsets: np.ndarray) -> np.ndarray:
        """Return (N, 4, 4) float64 velocity matrices for ABD bodies."""
        offsets = np.ascontiguousarray(body_offsets, dtype=np.int32)
        return self._engine.get_abd_body_velocities(offsets)

    def set_abd_body_velocities(self, body_offsets: np.ndarray,
                                velocities: np.ndarray) -> None:
        """Set ABD body velocities from (N, 4, 4) float64."""
        offsets = np.ascontiguousarray(body_offsets, dtype=np.int32)
        vels = np.ascontiguousarray(velocities, dtype=np.float64)
        self._engine.set_abd_body_velocities(offsets, vels)

    # ---- FEM body state ----

    def get_fem_body_vertex_range(self, fem_body_idx: int) -> tuple[int, int]:
        """Return (vertex_start, vertex_count) for a FEM body."""
        return self._engine.get_fem_body_vertex_range(fem_body_idx)

    # ---- Load record tracking ----

    def get_load_records(self) -> list:
        """Return all BodyLoadRecord objects."""
        return self._engine.get_all_load_records()

    # ---- Per-body views ----

    def get_bodies(self) -> list[BodyView]:
        """All bodies (ABD + FEM mixed) in load-record order."""
        return [BodyView(self, r) for r in self._engine.get_all_load_records()]

    def get_abd_body(self, body_id: int) -> BodyView:
        """Get the ABD body with the given index (0..abd_body_count-1)."""
        for r in self._engine.get_all_load_records():
            if r.body_type == 0 and r.body_offset == body_id:
                return BodyView(self, r)
        raise IndexError(f"ABD body {body_id} not found")

    def get_fem_body(self, body_id: int) -> BodyView:
        """Get the FEM body with the given index (0..fem_body_count-1)."""
        for r in self._engine.get_all_load_records():
            if r.body_type == 1 and r.body_offset == body_id:
                return BodyView(self, r)
        raise IndexError(f"FEM body {body_id} not found")

    def get_vertex_body_ids(self) -> np.ndarray:
        """Per-vertex body identification, shape (N, 2) int32.

        Column 0: body_type (0=ABD, 1=FEM)
        Column 1: body_offset (index within its kind)

        Useful for mask-based filtering across the global vertex array::

            ids = engine.get_vertex_body_ids()
            verts = engine.get_vertices()
            fem_verts = verts[ids[:, 0] == 1]      # all FEM vertices
            link3_verts = verts[(ids[:, 0] == 0) & (ids[:, 1] == 3)]
        """
        n = len(self.get_vertices())
        ids = np.zeros((n, 2), dtype=np.int32)
        for r in self._engine.get_all_load_records():
            s = r.vertex_offset
            e = s + r.vertex_count
            ids[s:e, 0] = r.body_type
            ids[s:e, 1] = r.body_offset
        return ids

    def get_body_contact_force(self, vertex_offset: int, vertex_count: int) -> np.ndarray:
        """Net body-body barrier GRADIENT sum over a vertex range — LEGACY UNITS.

        .. warning::
            This legacy accessor returns the raw incremental-potential
            gradient summed over ``[vertex_offset, vertex_offset+vertex_count)``
            — i.e. ``-force x dt^2``, NOT Newtons — and includes ONLY the
            body-body barrier term (no ground contact, no friction). It is kept
            for backward compatibility with pre-0.8.4 callers. For physical
            per-vertex contact forces in Newtons (with optional ground term)
            use :meth:`get_vertex_contact_forces` and aggregate per body via
            :meth:`get_load_records` vertex ranges.

        Returns a ``(3,)`` float64 array.  Must be called AFTER :meth:`step`.
        """
        return self._engine.get_body_contact_force(int(vertex_offset), int(vertex_count))

    def get_pair_contact_force(self, a_off: int, a_cnt: int, b_off: int, b_cnt: int) -> np.ndarray:
        """Net IPC contact force (world frame, raw IP scaling) on body A FROM body B.

        Barrier gradient summed over A's vertex range ``[a_off, a_off+a_cnt)``,
        restricted to collision pairs that connect A and B's vertex range
        ``[b_off, b_off+b_cnt)``.  Apply the same ``-1/dt**2`` + sign convention
        as :meth:`get_body_contact_force` to get Newtons.  Call AFTER :meth:`step`.
        """
        return self._engine.get_pair_contact_force(int(a_off), int(a_cnt), int(b_off), int(b_cnt))

    def get_collision_pairs_clean(self) -> np.ndarray:
        """Current body-body collision pairs as an ``(N, 4)`` int array of plain
        vertex indices (``-1`` padded for PP/PE), UIPC-style — the "clean export
        layer".  The solver keeps its MMCVID packing internally; this is a
        read-only decoded view for sensors / inspection.  Call AFTER :meth:`step`.
        """
        return self._engine.get_collision_pairs_clean()

    def get_contacts_device(self):
        """[Step B] GPU-resident per-contact export. Returns
        ``(count, pair_ptr, force_ptr)`` — device pointers (uintptr) to an int2
        ``(bodyA, bodyB)`` (bodyB=-1 for ground) and a double3 world contact force
        (N) on bodyA. Wrap with ``warp.array(ptr=..., copy=False)``. Read-only,
        valid until the next call. Call AFTER :meth:`step`.
        """
        return self._engine.get_contacts_device()

    def get_contacts(self):
        """[Step B] Host readback of the per-contact export: ``(pair (N,2) int,
        force (N,3) double world N on bodyA)``. For validation; the GPU-direct
        path uses :meth:`get_contacts_device`. Call AFTER :meth:`step`.
        """
        return self._engine.get_contacts()

    def get_vertex_contact_forces(self, include_ground: bool = True,
                                  components: str = "normal") -> np.ndarray:
        """Per-vertex IPC contact force (N, 3) in Newtons of the current state.

        ``components`` selects what is included:

        - ``"normal"`` (default, historic behavior): body-body barrier forces,
          plus ground contact when ``include_ground``.
        - ``"friction_lagged"``: the friction forces the solver ACTUALLY used
          this step. Positions are current; the normal force (lambda) and
          tangent basis are lagged one step by IPC's semi-implicit friction —
          this is the honest label for what the engine applies.
        - ``"total"``: normal + friction_lagged (e.g. an object at rest on a
          slope now sums to ~zero net contact force).

        Slice with :meth:`get_load_records` vertex ranges for per-body maps.
        Rebuilds normal contacts once when requested; the friction path is
        read-only on the solver's frozen friction set.
        """
        comp = {"normal": 0, "friction_lagged": 1, "total": 2}[components]
        return np.asarray(self._engine.get_vertex_contact_forces(include_ground, comp))

    def get_fem_von_mises_stress(self) -> np.ndarray:
        """Per-vertex von Mises stress (Pa) for the configured tetrahedral
        constitutive law. Non-tet vertices (cloth/ABD) are 0."""
        return np.asarray(self._engine.get_fem_von_mises_stress())

    def get_per_env_newton_iters(self) -> np.ndarray:
        """Newton iter at which each env froze last solve (-1 = ran to loop
        end / absent). Requires the host per-env path (``per_env_exit=True``
        plus ``env_newton_iter_cap`` or STIFF_PERENV_TELEM=1)."""
        return np.asarray(self._engine.get_per_env_newton_iters())

    def get_per_env_status(self) -> np.ndarray:
        """Per-env status of the last solve: 0 active/absent, 1 converged,
        2 timeout (env_newton_iter_cap), 3 diverged (NaN quarantined)."""
        return np.asarray(self._engine.get_per_env_status())

    def get_total_energy_tolerance_accepts(self) -> int:
        """Return the cumulative count of tolerance-assisted energy accepts."""
        return int(self._engine.get_total_energy_tolerance_accepts())

    def set_body_friction(self, body_offset: int, mu: float,
                          ground_mu: float | None = None) -> None:
        """Override one body's friction coefficient (per-body friction).

        ``body_offset`` indexes :meth:`get_load_records`. Self-contact pairs
        use the geometric mean of the two sides' mu; ground contact uses
        ``ground_mu`` for this body (None = keep the global
        ``gd_friction_rate``). Call after loading the body, before
        :meth:`finalize`. Scenes that never call this run the legacy global-mu
        path bit-identically. Example: GRIP's UMI soft finger (mu=3.5) grasping
        a mu=0.4 object in one scene.
        """
        self._engine.set_body_friction(int(body_offset), float(mu),
                                       -1.0 if ground_mu is None else float(ground_mu))

    def set_soft_body_density(self, body_offset: int, density: float) -> None:
        """Override one SOFT body's density (FEM tets or cloth shell).

        ``body_offset`` indexes :meth:`get_load_records`. Call after loading
        the body and before :meth:`finalize`. Unset bodies keep the global
        ``Config.density`` / ``cloth_density`` — this is what makes a
        light-towel + heavy-soft-plate scene possible.
        """
        self._engine.set_soft_body_density(int(body_offset), float(density))

    def set_abd_body_density(self, body_id: int, density: float) -> None:
        """Override one ABD body's density (mass = density × volume).

        Lets a scene mix per-body densities (the global ``Config.density`` is
        the default for bodies without an override).  Must be called AFTER the
        body is loaded and BEFORE :meth:`finalize`.
        """
        self._engine.set_abd_body_density(int(body_id), float(density))

    def set_abd_body_mass(self, body_id: int, mass: float) -> None:
        """Override one surface-mesh ABD body's total mass in kilograms.

        This API is intentionally distinct from :meth:`set_abd_body_density`.
        Call it after loading the body and before :meth:`finalize`.
        """
        self._engine.set_abd_body_mass(int(body_id), float(mass))

    def set_abd_body_inertia(self, body_id: int, mass: float,
                             com, inertia) -> None:
        """Override an ABD body's mass / COM / inertia (load frame).

        ``com`` is a length-3 array (world/load frame), ``inertia`` a 3x3 (about
        the COM). Uses authored values (e.g. URDF inertial tags via Newton
        body_mass/body_com/body_inertia) instead of the welded collision-mesh
        geometry. Must be called AFTER loading the body and BEFORE finalize().
        """
        import numpy as _np
        c = _np.ascontiguousarray(com, dtype=_np.float64).reshape(3)
        I = _np.ascontiguousarray(inertia, dtype=_np.float64).reshape(9)
        self._engine.set_abd_body_inertia(int(body_id), float(mass), c, I)

    # ---- Joint control ----

    @property
    def num_revolute_joints(self) -> int:
        return self._engine.get_num_revolute_joints()

    @property
    def num_prismatic_joints(self) -> int:
        return self._engine.get_num_prismatic_joints()

    def get_revolute_joint_info(self, idx: int):
        return self._engine.get_revolute_joint_info(idx)

    def get_prismatic_joint_info(self, idx: int):
        return self._engine.get_prismatic_joint_info(idx)

    def set_revolute_target(self, idx: int, angle_rad: float) -> None:
        self._engine.set_revolute_target(idx, angle_rad)

    def set_revolute_initial_offset(self, idx: int, offset_rad: float) -> None:
        self._engine.set_revolute_initial_offset(idx, offset_rad)

    def set_prismatic_target(self, idx: int, distance_m: float) -> None:
        self._engine.set_prismatic_target(idx, distance_m)

    def set_revolute_strength(self, idx: int, strength: float) -> None:
        """Set per-joint driving strength multiplier.

        Effective K = Config.revolute_driving_strength_ratio * strength *
        (m_parent + m_child). Default 1.0 (no change vs global ratio).
        Lower = joint yields under contact.
        """
        self._engine.set_revolute_strength(idx, strength)

    def set_prismatic_strength(self, idx: int, strength: float) -> None:
        self._engine.set_prismatic_strength(idx, strength)

    def get_prismatic_drive_force(self, idx: int) -> float:
        return self._engine.get_prismatic_drive_force(idx)

    def get_prismatic_current_distance(self, idx: int) -> float:
        return self._engine.get_prismatic_current_distance(idx)

    def get_body_contact_force_batched(self, offsets, counts) -> np.ndarray:
        """BATCHED body-body barrier GRADIENT sums — LEGACY UNITS.

        .. warning::
            Returns raw incremental-potential gradients (``-force x dt^2``, NOT
            Newtons) and includes ONLY the body-body barrier term (no ground,
            no friction). Kept for pre-0.8.4 callers. For physical per-vertex
            forces in Newtons use :meth:`get_vertex_contact_forces` and
            aggregate per segment.

        `offsets`/`counts` are per-finger vertex ranges; returns (n_seg, 3).
        Call AFTER step()."""
        return self._engine.get_body_contact_force_batched(
            np.ascontiguousarray(offsets, dtype=np.int32),
            np.ascontiguousarray(counts, dtype=np.int32))

    def set_max_revolute_step_per_frame(self, rad: float) -> None:
        """Set the per-step revolute target slew limit in radians."""
        self._engine.set_max_revolute_step_per_frame(float(rad))

    def set_max_prismatic_step_per_frame(self, meters: float) -> None:
        """Set the per-step prismatic target slew limit in meters."""
        self._engine.set_max_prismatic_step_per_frame(float(meters))

    def get_revolute_current_angles(self) -> np.ndarray:
        """Read actual joint angles from GPU state.

        Returns (N,) float64 in radians.  IMPORTANT: this is the angle
        RELATIVE TO THE LOAD-TIME POSE.  If the URDF was loaded via
        `load_urdf(initial_joint_angles=...)`, this returns 0 immediately
        after load (not the offset values).  For ABSOLUTE URDF angles
        use `get_revolute_current_angles_abs()`.
        """
        return self._engine.get_revolute_current_angles()

    def get_revolute_initial_offsets(self) -> np.ndarray:
        """Per-joint URDF angle at URDF load time.

        Returns (N,) float64 in radians.  Comes from the `initial_joint_angles`
        dict passed to `load_urdf`; all zeros if it wasn't passed.
        """
        return self._engine.get_revolute_initial_offsets()

    def get_revolute_current_angles_abs(self) -> np.ndarray:
        """Absolute URDF joint angles = relative-from-load + initial-offset.

        Returns (N,) float64 in radians.  Use this when you need 'what
        angle does the URDF think the joint is at right now' — e.g. for
        RL state observation, saving to disk, comparison with joint limits.
        """
        return (self._engine.get_revolute_current_angles()
                + self._engine.get_revolute_initial_offsets())

    def get_all_joint_infos(self):
        return self._engine.get_all_joint_infos()

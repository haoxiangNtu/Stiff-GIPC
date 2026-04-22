# CLAUDE.md — guide for AI assistants working on Stiff-GIPC

## What this repo is

**StiffGIPC** is a GPU-accelerated IPC (Incremental Potential Contact) physics
engine written in CUDA/C++. Python bindings are exposed via `pybind11` and
distributed as the **`stiff-physics`** wheel.

- Engine: `StiffGIPC/` (CUDA / C++)
- Bindings: `bindings/pystiffgipc.cu`
- Python package: `stiff_physics/` (Engine / Config / Robot wrappers)
- Examples: `examples/` (cloth, rigid, URDF arm, etc.)
- Scene assets: `Assets/`

## Build, test, run

### Build the Python module (development)

```bash
cmake -B build -DBUILD_PYTHON_BINDINGS=ON -DBUILD_GL_VIEWER=OFF
cmake --build build -j$(nproc) --target pystiffgipc
```

Produces `build/pystiffgipc.cpython-<ver>-x86_64-linux-gnu.so` and
`build/libstiffgipc_core.so`. The Python loader (`stiff_physics/engine.py`)
finds `.so` files in (in order): the installed `_native/` sub-package, then
`./build_venv/`, `./build_311/`, `./build/`.

### Build the OpenGL viewer + tests (full)

```bash
cmake -B build -DBUILD_PYTHON_BINDINGS=ON -DBUILD_GL_VIEWER=ON
cmake --build build -j$(nproc)
```

Requires GLEW, GLUT, OpenGL, ImGui (vcpkg or system).

### Build a release wheel

Use the documented commit→tag→worktree→build flow.
**See `STIFF_PHYSICS_RELEASE_HANDBOOK.md` §4.2 — do not skip it.**

### Run an example

```bash
PYTHONPATH=. python examples/case_26_arm_cloth_semi_implicit.py
```

## Branch policy

| Branch | Purpose |
|---|---|
| `main` | Upstream KemengHuang/Stiff-GIPC (read-only mirror) |
| `release/stable` | **Release source-of-truth.** Only audited commits. Each release is tagged from this branch. |
| `lhx/multi-env-instance` | Daily development |
| `lhx/wip-multi-env-instance` | WIP staging (post-v0.1.0 additions to be audited) |
| `lhx/case26-*`, `perf/*`, `bench/*` | **Experimental — not for release.** Must not be cherry-picked into `release/stable` without independent audit. |

When building from a specific commit/tag, use a `git worktree` so the main
working tree stays clean.

## Documentation map

| File | Audience |
|---|---|
| `CHANGELOG.md` | Public. User-facing release notes. |
| `STIFF_PHYSICS_RELEASE_HANDBOOK.md` | Maintainer. Wheel build, public-repo update, security rules. |
| `docs/internal/` | Maintainer-only. Gitignored — not pushed to remote. Per-issue handovers and decision rationale (`INDEX.md` for the index). |

## Conventions

- **Never commit** wheel binaries (`dist/`), build artefacts (`build*/`),
  runtime config dumps (`imgui.ini`, `.polyscope.ini`), or large media (`*.mp4`).
- **Never push** the private repo's working branches to a public remote
  without checking what handover/PLAN/RESEARCH docs they include — see the
  `.gitignore` patterns.
- Each release wheel must correspond to a **tagged commit** (no working-tree
  builds; v0.1.0 already had this problem and is now reconstructed via tag
  `v0.1.0-source`).
- Detailed release-process rules: handbook §4.2.

## Reporting issues

External users → GitHub issues on `haoxiangNtu/stiff-physics`.
Internal handovers → `docs/internal/` (gitignored).

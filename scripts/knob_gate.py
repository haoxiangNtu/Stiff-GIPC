#!/usr/bin/env python3
"""Static consistency gate for the STIFF_* knob registry (no GPU needed).

Asserts:
  1. every STIFF_* env read by C++/CUDA code has a row in
     StiffGIPC/config/knob_registry.h  (typo/omission = FAIL, names listed);
  2. every STIFF_* env read by the stiff_physics python layer has a row;
  3. every flag in the python mode resolver bundles has a row.

The registry plus the finalize-time tripwire turn "misspelled env var =
silent no-op" into a loud, nameable event; this gate keeps the registry
from drifting behind the code.
"""

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cpp_reads() -> set:
    out = subprocess.run(
        r"grep -rhoE '(getenv|env_on)\(\"STIFF_[A-Z0-9_]+\"' "
        "StiffGIPC/ bindings/ "
        "--include=*.cu --include=*.cuh --include=*.inl --include=*.h",
        shell=True, capture_output=True, text=True, cwd=ROOT,
    ).stdout
    return set(re.findall(r"STIFF_[A-Z0-9_]+", out))


def python_reads() -> set:
    out = subprocess.run(
        "grep -rhoE '\"STIFF_[A-Z0-9_]+\"' stiff_physics/",
        shell=True, capture_output=True, text=True, cwd=ROOT,
    ).stdout
    return set(re.findall(r"STIFF_[A-Z0-9_]+", out))


def resolver_flags() -> set:
    src = open(os.path.join(ROOT, "stiff_physics", "engine.py")).read()
    flags = set()
    for list_name in ("_MULTIENV_ISOLATED_FLAGS", "_MULTIENV_STRICT_EXTRA",
                      "_MULTIENV_ISOLATED_ONLY"):
        m = re.search(list_name + r"\s*=\s*\[(.*?)\]", src, re.S)
        if m:
            flags |= set(re.findall(r"STIFF_[A-Z0-9_]+", m.group(1)))
    return flags


def registry_names() -> set:
    src = open(
        os.path.join(ROOT, "StiffGIPC", "config", "knob_registry.h")
    ).read()
    names = set(re.findall(r"^\s*X\((STIFF_[A-Z0-9_]+),", src, re.M))
    names.add("STIFF_KNOB_STRICT")  # registry meta-knob, matched in code
    return names


def main() -> int:
    registry = registry_names()
    ok = True
    for label, reads in (
        ("C++/CUDA", cpp_reads()),
        ("python layer", python_reads()),
        ("mode resolver", resolver_flags()),
    ):
        missing = sorted(reads - registry)
        print(f"  {label}: {len(reads)} knobs read, "
              f"{len(missing)} missing from registry")
        if missing:
            ok = False
            for name in missing:
                print(f"    MISSING ROW: {name}")
    print("KNOB-GATE:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Summarize Stiff-GIPC BVH work from an Nsight Systems SQLite export.

The report intentionally separates traversal families.  VF/EE DCD and VF/EE
CCD do not share one traversal kernel in the current implementation, so a
single aggregate number can hide regressions in one of the four paths.

Example:
  nsys export --type sqlite --output run.sqlite run.nsys-rep
  python scripts/bvh_profile_report.py run.sqlite --frames 30
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path


QUERY_PREFIXES = {
    "vf_dcd": ("_selfQuery_vf(",),
    "ee_dcd": (
        "_selfQuery_ee(",
        "_selfQuery_ee_lb2(",
        "_selfQuery_ee_lb3(",
        "_selfQuery_ee_range_prune(",
        "_selfQuery_ee_sorted_prune(",
    ),
    "vf_ccd": ("_selfQuery_vf_ccd(",),
    "ee_ccd": (
        "_selfQuery_ee_ccd(",
        "_selfQuery_ee_ccd_range_prune(",
    ),
}

CACHE_PREFIXES = {
    "vf_dcd_replay": ("_replayVfPairCache(",),
    "vf_cache_validity": ("_updateBvhVfPairCacheValidityDevice(",),
    "vf_cache_front": ("_buildBvhVfPairFront(",),
    "vf_cache_reset": ("_resetInvalidVfPairCacheCounts(",),
}

# Kernels that belong uniquely to construction/refit of this LBVH.  CUB radix
# sort is deliberately excluded: the same generated name is also used by
# matrix/triplet sorting, and attributing all of it to BVH would be false.
BUILD_PREFIXES = (
    "void _calcLeafBvs",
    "_reduct_max_box(",
    "_calcMChash",
    "_iota_u32(",
    "_sortBvs(",
    "_calcLeafNodes",
    "_calcInternalNodes(",
    "_calcInternalAABB",
    "_calcInternalDepths(",
    "_rotateSahTreelets(",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path, help="SQLite file from `nsys export`")
    parser.add_argument("--frames", type=int, help="completed simulation frames")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser.parse_args()


def _classify(name: str) -> str | None:
    for category, prefixes in QUERY_PREFIXES.items():
        if name.startswith(prefixes):
            return category
    return None


def _record(duration_ns: int) -> dict[str, float | int]:
    return {
        "calls": 0,
        "time_ms": 0.0,
        "mean_us": 0.0,
        "fraction_of_kernel_time": 0.0,
    }


def load_report(path: Path, frames: int | None) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"profile does not exist: {path}")

    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        table_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        required = {"CUPTI_ACTIVITY_KIND_KERNEL", "StringIds"}
        missing = sorted(required - table_names)
        if missing:
            raise RuntimeError(
                "not a CUDA-kernel nsys export; missing tables: " + ", ".join(missing)
            )

        rows = connection.execute(
            """
            SELECT StringIds.value, kernel.end - kernel.start
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernel
            JOIN StringIds ON StringIds.id = kernel.demangledName
            """
        )

        total_ns = 0
        query_ns: dict[str, int] = defaultdict(int)
        query_calls: dict[str, int] = defaultdict(int)
        build_ns = 0
        build_calls = 0
        observed_query_names: dict[str, set[str]] = defaultdict(set)
        cache_ns: dict[str, int] = defaultdict(int)
        cache_calls: dict[str, int] = defaultdict(int)

        for name, duration_ns in rows:
            duration_ns = int(duration_ns)
            total_ns += duration_ns
            category = _classify(name)
            if category is not None:
                query_ns[category] += duration_ns
                query_calls[category] += 1
                observed_query_names[category].add(name.split("(", 1)[0])
            for cache_category, prefixes in CACHE_PREFIXES.items():
                if name.startswith(prefixes):
                    cache_ns[cache_category] += duration_ns
                    cache_calls[cache_category] += 1
                    break
            if name.startswith(BUILD_PREFIXES):
                build_ns += duration_ns
                build_calls += 1

        if total_ns == 0:
            raise RuntimeError("profile contains no CUDA kernel duration")

        query_total_ns = sum(query_ns.values())
        categories: dict[str, dict[str, object]] = {}
        for category in QUERY_PREFIXES:
            calls = query_calls[category]
            duration_ns = query_ns[category]
            categories[category] = {
                "calls": calls,
                "time_ms": duration_ns / 1e6,
                "mean_us": duration_ns / calls / 1e3 if calls else 0.0,
                "fraction_of_kernel_time": duration_ns / total_ns,
                "kernel_symbols": sorted(observed_query_names[category]),
            }

        report: dict[str, object] = {
            "profile": str(path.resolve()),
            "frames": frames,
            "total_kernel_time_ms": total_ns / 1e6,
            "query_total": {
                "calls": sum(query_calls.values()),
                "time_ms": query_total_ns / 1e6,
                "fraction_of_kernel_time": query_total_ns / total_ns,
            },
            "query_families": categories,
            "cache_kernels": {
                category: {
                    "calls": cache_calls[category],
                    "time_ms": cache_ns[category] / 1e6,
                    "mean_us": (
                        cache_ns[category] / cache_calls[category] / 1e3
                        if cache_calls[category]
                        else 0.0
                    ),
                    "fraction_of_kernel_time": cache_ns[category] / total_ns,
                }
                for category in CACHE_PREFIXES
            },
            "bvh_build_excluding_cub_sort": {
                "calls": build_calls,
                "time_ms": build_ns / 1e6,
                "fraction_of_kernel_time": build_ns / total_ns,
                "note": "CUB radix sort excluded because its generated symbol is shared",
            },
        }
        if frames is not None:
            if frames <= 0:
                raise RuntimeError("--frames must be positive")
            report["per_frame"] = {
                "kernel_ms": total_ns / frames / 1e6,
                "query_ms": query_total_ns / frames / 1e6,
                "bvh_build_excluding_cub_sort_ms": build_ns / frames / 1e6,
            }
        return report
    finally:
        connection.close()


def print_human(report: dict[str, object]) -> None:
    total_ms = float(report["total_kernel_time_ms"])
    query_total = report["query_total"]
    assert isinstance(query_total, dict)
    print(f"profile: {report['profile']}")
    print(f"total CUDA kernel time: {total_ms:.3f} ms")
    print(
        "BVH queries: "
        f"{float(query_total['time_ms']):.3f} ms, "
        f"{100.0 * float(query_total['fraction_of_kernel_time']):.2f}% of kernels, "
        f"{int(query_total['calls'])} launches"
    )
    print("family   calls      time_ms     mean_us   kernel_%")
    families = report["query_families"]
    assert isinstance(families, dict)
    for category in QUERY_PREFIXES:
        item = families[category]
        assert isinstance(item, dict)
        print(
            f"{category:8s} {int(item['calls']):6d} "
            f"{float(item['time_ms']):12.3f} {float(item['mean_us']):11.3f} "
            f"{100.0 * float(item['fraction_of_kernel_time']):10.2f}"
        )
    cache = report["cache_kernels"]
    assert isinstance(cache, dict)
    for category in CACHE_PREFIXES:
        item = cache[category]
        assert isinstance(item, dict)
        if int(item["calls"]):
            print(
                f"{category:16s} {int(item['calls']):6d} "
                f"{float(item['time_ms']):12.3f} "
                f"{float(item['mean_us']):11.3f} "
                f"{100.0 * float(item['fraction_of_kernel_time']):10.2f}"
            )
    build = report["bvh_build_excluding_cub_sort"]
    assert isinstance(build, dict)
    print(
        "BVH build (known kernels, CUB excluded): "
        f"{float(build['time_ms']):.3f} ms, {int(build['calls'])} launches"
    )
    per_frame = report.get("per_frame")
    if isinstance(per_frame, dict):
        print(
            f"per frame ({int(report['frames'])}): "
            f"kernels={float(per_frame['kernel_ms']):.3f} ms, "
            f"queries={float(per_frame['query_ms']):.3f} ms, "
            "known-build="
            f"{float(per_frame['bvh_build_excluding_cub_sort_ms']):.3f} ms"
        )


def main() -> int:
    args = parse_args()
    try:
        report = load_report(args.sqlite, args.frames)
    except (RuntimeError, sqlite3.Error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        print()
    else:
        print_human(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

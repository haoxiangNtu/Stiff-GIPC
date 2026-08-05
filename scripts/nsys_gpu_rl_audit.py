#!/usr/bin/env python3
"""Audit a node-level Nsight Systems GPU-native RL steady-state trace.

The cudaProfilerStop boundary may make Nsight synchronize outstanding GPU
work while it flushes the capture.  That profiler-owned boundary is outside
the host submission loop and must not be confused with a per-step wait.  This
gate therefore requires zero H2D/D2H over the complete capture and zero host
or CUPTI synchronization inside the interval spanning the first action publish
through the last completion-event record.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def scalar(connection: sqlite3.Connection, query: str) -> int:
    row = connection.execute(query).fetchone()
    if row is None or row[0] is None:
        raise RuntimeError(f"query returned no scalar: {query}")
    return int(row[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("--expected-steps", type=int, default=40)
    args = parser.parse_args()
    if args.expected_steps <= 0:
        parser.error("--expected-steps must be positive")
    if not args.database.is_file():
        parser.error(f"database does not exist: {args.database}")

    connection = sqlite3.connect(args.database)
    try:
        h2d_rows = scalar(
            connection,
            "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY "
            "WHERE copyKind IN (1, 11)",
        )
        d2h_rows = scalar(
            connection,
            "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY "
            "WHERE copyKind IN (2, 12)",
        )
        d2d_rows = scalar(
            connection,
            "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY "
            "WHERE copyKind IN (8, 13)",
        )
        graph_launches = scalar(
            connection,
            "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r "
            "JOIN StringIds AS s ON s.id=r.nameId "
            "WHERE s.value LIKE '%GraphLaunch%'",
        )
        event_records = scalar(
            connection,
            "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r "
            "JOIN StringIds AS s ON s.id=r.nameId "
            "WHERE s.value LIKE '%EventRecord%'",
        )
        graph_node_events = scalar(
            connection,
            "SELECT COUNT(*) FROM CUDA_GRAPH_NODE_EVENTS",
        )

        bounds = connection.execute(
            "SELECT MIN(r.start), MAX(r.end) "
            "FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r "
            "JOIN StringIds AS s ON s.id=r.nameId "
            "WHERE s.value LIKE '%MemcpyAsync%' "
            "OR s.value LIKE '%GraphLaunch%' "
            "OR s.value LIKE '%EventRecord%'"
        ).fetchone()
        if bounds is None or bounds[0] is None or bounds[1] is None:
            raise RuntimeError("steady-state submission interval is empty")
        submit_start, submit_end = map(int, bounds)

        sync_api_inside = int(
            connection.execute(
                "SELECT COUNT(*) "
                "FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r "
                "JOIN StringIds AS s ON s.id=r.nameId "
                "WHERE (lower(s.value) LIKE '%synchronize%' "
                "OR lower(s.value) LIKE '%wait%') "
                "AND r.start < ? AND r.end > ?",
                (submit_end, submit_start),
            ).fetchone()[0]
        )
        cupti_sync_inside = int(
            connection.execute(
                "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION "
                "WHERE start < ? AND end > ?",
                (submit_end, submit_start),
            ).fetchone()[0]
        )
        sync_api_total = scalar(
            connection,
            "SELECT COUNT(*) "
            "FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r "
            "JOIN StringIds AS s ON s.id=r.nameId "
            "WHERE lower(s.value) LIKE '%synchronize%' "
            "OR lower(s.value) LIKE '%wait%'",
        )
        cupti_sync_total = scalar(
            connection,
            "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
        )
    finally:
        connection.close()

    evidence = {
        "database": str(args.database.resolve()),
        "expected_steps": args.expected_steps,
        "graph_launches": graph_launches,
        "event_records": event_records,
        "d2d_rows": d2d_rows,
        "h2d_rows": h2d_rows,
        "d2h_rows": d2h_rows,
        "graph_node_events": graph_node_events,
        "submit_window_us": (submit_end - submit_start) / 1000.0,
        "sync_api_inside_submit": sync_api_inside,
        "cupti_sync_inside_submit": cupti_sync_inside,
        "sync_api_total_including_profiler_boundary": sync_api_total,
        "cupti_sync_total_including_profiler_boundary": cupti_sync_total,
    }
    print(json.dumps(evidence, indent=2, sort_keys=True))

    failures: list[str] = []
    expected_d2d = 2 * args.expected_steps
    if graph_launches != args.expected_steps:
        failures.append(
            f"graph launches {graph_launches} != {args.expected_steps}"
        )
    if event_records != args.expected_steps:
        failures.append(
            f"event records {event_records} != {args.expected_steps}"
        )
    if d2d_rows != expected_d2d:
        failures.append(f"D2D rows {d2d_rows} != {expected_d2d}")
    if h2d_rows or d2h_rows:
        failures.append(f"host/device transfers H2D={h2d_rows} D2H={d2h_rows}")
    if sync_api_inside or cupti_sync_inside:
        failures.append(
            "steady submission synchronized: "
            f"API={sync_api_inside} CUPTI={cupti_sync_inside}"
        )
    if graph_node_events <= 0:
        failures.append("trace contains no CUDA Graph node events")

    if failures:
        print("NSYS-GPU-RL-AUDIT: FAIL: " + "; ".join(failures))
        return 1
    print("NSYS-GPU-RL-AUDIT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

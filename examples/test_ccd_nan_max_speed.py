#!/usr/bin/env python3
"""Regression: device CCD must not discard a NaN max-speed candidate."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build"))

import pystiffgipc


def main() -> None:
    try:
        pystiffgipc._test_ccd_nan_max_speed_fail_fast()
    except RuntimeError as exc:
        message = str(exc)
        assert "maxSpeed" in message, message
        assert "fail-fast" in message, message
        print("PASS: NaN max-speed is rejected by the device CCD tail")
        return
    raise AssertionError("NaN max-speed unexpectedly produced an accepted CCD step")


if __name__ == "__main__":
    main()

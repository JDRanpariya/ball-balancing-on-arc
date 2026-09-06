"""Sensor recalibration utility for the ball-on-arc ToF sensors.

Usage (from evaluation/):
    PYTHONPATH=. python scripts/recalibrate_sensors.py

Or from Python:
    from balancer.hardware.calibration import recalibrate
    recalibrate(n_samples=500)

The function:
0. Keep the ball at physical center of the arc.
1. Reads N raw samples from the distance sensor with ball at physical center.
2. Computes median LEFT and RIGHT mm values.
3. Updates constants.py LEFT_CENTER_MM, RIGHT_CENTER_MM, SENSOR_OFFSET_RAD.
4. Appends a timestamped entry to calibration_log.json.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import serial
from datetime import datetime
from pathlib import Path
from typing import Optional

_LOG_PATH = Path(__file__).parent / "calibration_log.json"
_CONSTANTS_PATH = Path(__file__).parent / "constants.py"

BAUD = 115200
DIST_PORT = "/dev/distance_sensor"


def _read_raw_samples(n: int, port: str = DIST_PORT, baud: int = BAUD) -> tuple[list[int], list[int]]:
    """Read n valid raw (left_mm, right_mm) samples from the distance sensor."""
    from balancer.hardware.sensor_utils import parse_distance_line

    lefts, rights = [], []
    ser = serial.Serial(port, baud, timeout=2.0)
    try:
        while len(lefts) < n:
            line = ser.readline()
            if not line:
                continue
            decoded = line.decode("utf-8", errors="ignore").strip()
            result = parse_distance_line(decoded)
            if result is None:
                continue
            l, r, _ = result
            if l == 65535 or r == 65535:
                continue
            if l > 300 or r > 300:
                continue
            lefts.append(l)
            rights.append(r)
    finally:
        ser.close()
    return lefts, rights


def _update_constants(left_mm: int, right_mm: int) -> None:
    """Patch LEFT_CENTER_MM, RIGHT_CENTER_MM and SENSOR_OFFSET_RAD in constants.py."""
    src = _CONSTANTS_PATH.read_text()

    src = re.sub(
        r"(LEFT_CENTER_MM:\s*int\s*=\s*)\d+",
        lambda m: f"{m.group(1)}{left_mm}",
        src,
    )
    src = re.sub(
        r"(RIGHT_CENTER_MM:\s*int\s*=\s*)\d+",
        lambda m: f"{m.group(1)}{right_mm}",
        src,
    )
    src = re.sub(
        r"(SENSOR_OFFSET_RAD:\s*float\s*=\s*)[0-9.]+",
        lambda m: f"{m.group(1)}0.0",
        src,
    )
    _CONSTANTS_PATH.write_text(src)


def _append_log(left_mm: int, right_mm: int, note: str) -> None:
    """Append a new calibration entry to calibration_log.json."""
    with open(_LOG_PATH) as f:
        log = json.load(f)

    new_id = max(e["id"] for e in log["entries"]) + 1
    entry = {
        "id": new_id,
        "date": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "note": note,
        "LEFT_CENTER_MM": left_mm,
        "RIGHT_CENTER_MM": right_mm,
        "equilibrium_left_mm": left_mm,
        "equilibrium_right_mm": right_mm,
        "SENSOR_OFFSET_RAD": 0.0,
    }
    log["entries"].append(entry)

    with open(_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


def recalibrate(
    n_samples: int = 500,
    port: str = DIST_PORT,
    note: str = "",
    dry_run: bool = False,
) -> dict:
    """Recalibrate sensor centers from live readings.

    Place the ball at the physical arc center before calling this.

    Args:
        n_samples: Number of samples to average (default 500, ~25s at 20Hz).
        port: Serial port for distance sensor.
        note: Optional note stored in the calibration log.
        dry_run: If True, print results but do not write any files.

    Returns:
        Dict with keys: left_mm, right_mm, old_left, old_right, offset_rad.
    """
    from balancer.hardware.constants import SYSTEM

    print(f"Reading {n_samples} samples from {port} ... (keep ball at physical center)")
    lefts, rights = _read_raw_samples(n_samples, port=port)

    new_left = int(round(statistics.median(lefts)))
    new_right = int(round(statistics.median(rights)))
    old_left = SYSTEM.LEFT_CENTER_MM
    old_right = SYSTEM.RIGHT_CENTER_MM

    arc_r = SYSTEM.ARC_RADIUS_M
    # theta at new equilibrium using OLD constants (= residual offset being fixed)
    if new_right <= old_right:
        residual = -(new_right - old_right) / 1000.0 / arc_r
    else:
        residual = (new_left - old_left) / 1000.0 / arc_r

    print(f"\n--- Calibration results ---")
    print(f"  Samples : {len(lefts)}")
    print(f"  LEFT  : old={old_left}mm  new={new_left}mm  drift={new_left-old_left:+d}mm")
    print(f"  RIGHT : old={old_right}mm  new={new_right}mm  drift={new_right-old_right:+d}mm")
    print(f"  Residual offset (old constants): {residual*1000:+.2f} mrad")
    print(f"  SENSOR_OFFSET_RAD -> 0.0 (absorbed into center_mm)")

    if not note:
        note = (f"Recalibration. Ball at physical center: LEFT={new_left}mm RIGHT={new_right}mm. "
                f"Drift from previous: LEFT={new_left-old_left:+d}mm RIGHT={new_right-old_right:+d}mm. "
                f"SENSOR_OFFSET_RAD set to 0.0.")

    if dry_run:
        print("\n[dry_run] No files written.")
    else:
        _update_constants(new_left, new_right)
        _append_log(new_left, new_right, note)
        print(f"\nUpdated constants.py: LEFT_CENTER_MM={new_left}, RIGHT_CENTER_MM={new_right}, SENSOR_OFFSET_RAD=0.0")
        print(f"Appended entry to {_LOG_PATH}")

    return {
        "left_mm": new_left,
        "right_mm": new_right,
        "old_left": old_left,
        "old_right": old_right,
        "residual_offset_rad": residual,
    }

#!/usr/bin/env python3
"""CLI entry point for sensor recalibration.

Place the ball at the physical arc center, then run:
    cd evaluation
    python scripts/recalibrate_sensors.py

Options:
    --samples N     Number of samples to average (default 500)
    --note TEXT     Optional note for the log entry
    --dry-run       Print results without writing any files
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from balancer.hardware.calibration import recalibrate

parser = argparse.ArgumentParser(description="Recalibrate ToF sensor centers")
parser.add_argument("--samples", type=int, default=500)
parser.add_argument("--note", type=str, default="")
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

print("=" * 60)
print("  ToF Sensor Recalibration")
print("  Ball must be at physical arc center and at rest.")
print("=" * 60)
result = recalibrate(n_samples=args.samples, note=args.note, dry_run=args.dry_run)
print("\nDone.")

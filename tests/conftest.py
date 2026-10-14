"""Puts the repository root (packages occupancy, bounds), experiments/ (runner, required_fleet, dock_mouth, capacity_bounds)
and tests/ (helpers) on sys.path."""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT, ROOT / "experiments", ROOT / "tests"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

csv.field_size_limit(10 ** 9)        # stored points.csv: the deadlock-time column of long runs is very long

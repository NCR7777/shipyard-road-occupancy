"""Reproduction of stored results: points of two stored runs are recomputed with the released code from the run's own
config.json at their fixed seeds (experiments/runner.compute, derive, point_row) and must equal the stored rows of
points.csv in every column except the run time (secs), compared as text, so floats to the last bit. The two label columns
util_top1_name and busy_top1_name follow experiments/rerun.same: runs stored before junction names got their bracketed tag
[building·entrance, short id] match the rerun name up to that tag.

  Okpo, main scenario sectioned at the stops, safe release, saturated, K = 5, seed 101: sectioning, the Banker test, the
  per-point check against T1' of the statistics window and the steady-state readings;
  Yantai (run on revision 395, whose road network is that of the released revision 396), main scenario, stepwise requests
  with teleport recovery, saturated, K = 3, seed 101: deadlock detection and recovery, deadlock histograms and times.
Also the tables of the screen's lower end and the chain statistics, rebuilt from the released files by
experiments/chain_statistics.py."""
import csv
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

import runner as R
from helpers import ROOT
from rerun import NAME_COLS, same

KEY = ("scen", "model", "level", "K", "seed")
POINTS = [("okpo_rules", "main_s|segment_safe|saturated|5|101"), ("yantai_rev395_saturation", "main|segment|saturated|3|101")]


def test_same_rule():
    """Every column must be equal as text; only the two name columns accept a stored junction name without the tag."""
    tagged = "junction entrance001 node [building014·paint shop (inferred)·building014-entrance1, 3fa2]"
    assert same("throughput", "91.0", "91.0") is True and same("throughput", "91.0", "91.00000000000001") is False
    assert same("util_top1_name", "junction entrance001 node", tagged) == "name"
    assert same("util_top1_name", "junction entrance002 node", tagged) is False
    assert same("util_top1_name", tagged.replace("3fa2", "3fa3"), tagged) is False
    assert same("flow_mean", "junction entrance001 node", tagged) is False and NAME_COLS == {"util_top1_name", "busy_top1_name"}


@pytest.mark.parametrize("run,key", POINTS)
def test_rerun_stored_point(run, key):
    d = ROOT / "results" / "runs" / run
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    text = (d / "points.csv").read_text(encoding="utf-8-sig")
    columns = [c for c in next(csv.reader(io.StringIO(text))) if c != "secs"]
    stored = [r for r in csv.DictReader(io.StringIO(text)) if "|".join(r[c] for c in KEY) == key]
    assert len(stored) == 1, key
    got = R.point_row(R.derive(R.compute(cfg, key)), columns)
    diff = {c: (stored[0][c][:60], got[c][:60]) for c in columns if same(c, stored[0][c], got[c]) is False}
    assert not diff, diff
    assert all(got[c] for c in ("throughput", "T1_tight_mix", "mix_shift", "thr_by_day", "util_top1_name"))
    assert sum(bool(got[c]) for c in columns) > 40


def test_screen_tables_from_release():
    """stable_low.csv and intra_own.csv byte for byte; best and align of chain_stats.json in every released field."""
    rel = ROOT / "results" / "screen"
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        subprocess.run([sys.executable, "-B", "experiments/chain_statistics.py", "--out", d], cwd=ROOT, check=True)
        for f in ("stable_low.csv", "intra_own.csv"):
            assert (out / f).read_bytes() == (rel / f).read_bytes(), f
        a, b = (json.loads(p.read_text(encoding="utf-8")) for p in (rel / "chain_stats.json", out / "chain_stats.json"))
    assert all(b["best"][k] == v for k, v in a["best"].items())
    assert len(a["align"]) == len(b["align"]) == 25
    assert all(y[k] == v for x, y in zip(a["align"], b["align"]) for k, v in x.items())

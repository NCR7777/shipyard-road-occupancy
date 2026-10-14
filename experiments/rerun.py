"""Rerun stored points of a run at their fixed seeds and compare them with the stored rows, cell by cell.

A run directory holds config.json and points.csv (results/runs/<run>/, results/synthetic_dock/); a large point table is
split into points_<part>.csv files with the same columns (by granularity, see results/README.md), read here together.
Each selected row is
recomputed with the released code (experiments/runner.py for the yard runs, experiments/dock_mouth.py for the synthetic dock
mouth) and written as a row of the same file; every column of the stored file except the run time (secs) must be equal as
text, which for the floats means equal to the last bit.

Run:   python experiments/rerun.py results/runs/okpo_rules --pick scen=main,model=reserve,K=30,seed=101 [--pick ...]
       python experiments/rerun.py results/runs/okpo_rules --sample 4 [--procs N]
--sample n picks n rows spread evenly over the stored file (sorted by key); --max-k drops rows with larger fleets.
Exit code 0 if every compared row is identical.
"""
import argparse
import csv
import io
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import runner as R          # noqa: E402

csv.field_size_limit(10 ** 9)
SKIP = {"secs"}             # run time of a point
# Runs stored before junction names got their bracketed tag ([building·entrance, short id]) name junctions without it; for
# these two label columns a stored name without the tag is compared with the rerun name up to the tag (every other column,
# and every number, must be equal as text)
NAME_COLS = {"util_top1_name", "busy_top1_name"}


def same(col, stored, rerun):
    """True if equal; "name" if equal only up to the junction tag that the stored (earlier) format lacks."""
    if stored == rerun:
        return True
    if col in NAME_COLS and " [" not in stored and rerun.split(" [")[0] == stored:
        return "name"
    return False


def select(rows, picks, sample, max_k, key_cols):
    if max_k is not None:
        rows = [r for r in rows if int(float(r["K"])) <= max_k]
    out = []
    for p in picks or []:
        want = dict(x.split("=", 1) for x in p.split(","))
        hit = [r for r in rows if all(r[k] == v for k, v in want.items())]
        if not hit:
            sys.exit(f"no stored row matches {p}")
        out += hit
    if sample:
        rows = sorted(rows, key=lambda r: tuple(r[c] for c in key_cols))
        step = max(1, len(rows) // sample)
        out += rows[::step][:sample]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("--pick", action="append")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--max-k", type=int, default=None)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    run = Path(a.run)
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    texts = [f.read_text(encoding="utf-8-sig") for f in sorted(run.glob("points*.csv"))]
    stored = [r for t in texts for r in csv.DictReader(io.StringIO(t))]
    heads = {tuple(next(csv.reader(io.StringIO(t)))) for t in texts}
    assert len(heads) == 1, f"{run}: point tables with different columns"
    columns = [c for c in heads.pop() if c not in SKIP]
    synthetic = "network" not in cfg
    if synthetic:
        import dock_mouth as M
        key_cols = ("rule", "W", "rho", "K", "seed")
        key = lambda r: "|".join(f"{float(r[c]):g}" if c == "W" else r[c] for c in key_cols)
    else:
        key_cols = ("scen", "model", "level", "K", "seed")
        key = lambda r: "|".join(r[c] for c in key_cols)
    rows = select(stored, a.pick, a.sample, a.max_k, key_cols)
    keys = list(dict.fromkeys(key(r) for r in rows))
    print(f"{run}: {len(keys)} stored points to rerun", flush=True)
    t0 = time.time()
    if synthetic:
        res = M.rerun(cfg, keys, a.procs)
        got = {k: M.point_row(r, columns) for k, r in res.items()}
    else:
        res = R.run_pool(cfg, None, min(a.procs, len(keys)), keys)
        got = {k: R.point_row(R.derive(r), columns) for k, r in res.items()}
    bad = named = 0
    for r in rows:
        k = key(r)
        cmp = {c: same(c, r[c], got[k][c]) for c in columns}
        diff = [c for c, v in cmp.items() if v is False]
        tag = [c for c, v in cmp.items() if v == "name"]
        bad += bool(diff)
        named += bool(tag) and not diff
        print(("IDENTICAL " if not diff else "DIFFERENT ") + k
              + ("" if diff or not tag else f"  (stored junction names in the earlier format without the tag: {', '.join(tag)})")
              + ("" if not diff else "  " + "; ".join(f"{c}: stored {r[c][:40]!r}, rerun {got[k][c][:40]!r}" for c in diff[:4])),
              flush=True)
    print(f"{len(rows) - bad} of {len(rows)} rows identical in {len(columns)} columns ({named} with junction names in the earlier "
          f"format); {time.time() - t0:.0f} s", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

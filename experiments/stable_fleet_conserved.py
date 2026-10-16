"""K_stable of the conserved inter-area mapping (the main case): the 30-day criterion of experiments/daily.py with the search
of experiments/stable_fleet.py, for the 30-day batches of the conserved mapping (loading and unloading 10 min, and 30 min;
the levels of a scenario as listed in its configuration: the points of the scan over the share of intra-area moves only at
the plant demand). The search starts at K_req - 2 as in stable_fleet; where the result of a criterion falls exactly on the
start (nothing below it computed, so K_stable could be smaller: censored at the lower end), the start moves down by 4 until
1 (k0_of); cells whose result is not at the start keep the start of stable_fleet. K_req is the required fleet of the same
cells in four days (experiments/required_fleet.py; --kreq as in stable_fleet).

Run:   python experiments/stable_fleet_conserved.py results/runs/<run>/config.json --out <dir> --kreq <demand.csv> [...]
       [--procs N] [--max-tasks-per-child N] [--slots DIR]
       e.g. results/runs/okpo_main_stable/config.json with --kreq results/runs/okpo_main_fleet/demand.csv, and
       results/runs/okpo_main_stable_30min/config.json with --kreq results/runs/okpo_main_fleet_30min/demand.csv
(--max-tasks-per-child: a worker process exits after this many points and frees its memory; --slots as in
experiments/runner.py). The saturation curves of the conserved mapping (whole and sectioned roads, 30-min handling) are
ordinary batches of experiments/runner.py. Writes <dir>/points/ (cache) and <dir>/stable.json
({"<scenario>|<rule>|<level>": {K_req, k0, stable, stable_tol}}).
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import runner as F                        # noqa: E402
import stable_fleet as R                  # noqa: E402


def cells(cfg):
    return [(s, m, lvl) for s in cfg["scenarios"] for m in cfg["models"][s] for lvl in F.levels_of(cfg, s)]


def k0_of(ev, kreq):
    """Start of the search: K_req - 2 (as stable_fleet); while the result of a criterion falls exactly on the start (nothing
    below it computed, K_stable may be smaller: censored at the lower end) the start moves down by 4, until 1. Cells whose
    result is not at the start keep the start of stable_fleet, cell by cell."""
    k0 = max(1, kreq - 2)
    while k0 > 1 and k0 in R.search(ev, k0)[1].values():
        k0 = max(1, k0 - 4)
    return k0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kreq", nargs="+", required=True, help="demand tables with K_req of the same cells (demand.csv or demand.json)")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--max-tasks-per-child", type=int, default=None, help="points per worker process before it exits and frees "
                                                                         "its memory (runner.run_pool); default: no limit")
    ap.add_argument("--slots", default=os.environ.get("CPU_SLOTS"), help="directory of machine-wide CPU slots (cpu_slots.Slots); "
                                                                         "default: environment variable CPU_SLOTS, else off")
    a = ap.parse_args()
    cfg = F.load_config(a.config)
    out = Path(a.out)
    (out / "points").mkdir(parents=True, exist_ok=True)
    kreq = R.kreq_table(a.kreq, cfg.get("yard"))
    pts = R.load(out)
    while True:
        need = []
        for s, m, lvl in cells(cfg):
            ev = R.evaluate(cfg, pts, s, m, lvl)
            ks, _ = R.search(ev, k0_of(ev, kreq[(s, m, lvl)]))
            need += [f"{s}|{m}|{lvl}|{K}|{sd}" for K in ks for sd in cfg["seeds"] if (s, m, lvl, K, sd) not in pts]
        if not need:
            break
        need.sort(key=lambda k: -int(k.split("|")[3]))            # large fleets first
        F.run_pool(cfg, out, min(a.procs, len(need)), need, slots_dir=a.slots, max_tasks=a.max_tasks_per_child)
        pts = R.load(out)
    res = {}
    for s, m, lvl in cells(cfg):
        ev = R.evaluate(cfg, pts, s, m, lvl)
        k0 = k0_of(ev, kreq[(s, m, lvl)])
        res[f"{s}|{m}|{lvl}"] = dict(K_req=kreq[(s, m, lvl)], k0=k0, **R.search(ev, k0)[1])
    F.write_json(out / "stable.json", res)
    print(f"{len(res)} cells, {len(pts)} points; stable.json in {out}", flush=True)


if __name__ == "__main__":
    main()

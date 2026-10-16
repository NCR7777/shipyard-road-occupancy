"""The lower end of the screen checked without a long-run backlog: representative low-load cells of the screen of demand over
the bound, each run for 30 days of continuous release under free flow and full-path reservation with 10 seeds. K_stable is
the 30-day criterion of experiments/stable_fleet.py (tolerance criterion: the mean over the 10 seeds of the slope of the
end-of-day backlog over days 6-30 at most 0.1 % of the daily demand per day, for K and K + 1), searched from K_req - 2 with
the start of experiments/stable_fleet_conserved.py (k0_of: while the result falls on the start, the start moves down by 4).
The configurations are the stored ones of the runs (the 30-day versions of the screen cells, same parameters otherwise):
  results/runs/okpo_screen_stable_main/config.json            conserved mapping of the main case, Okpo 75, 85, 100 tasks a
                                                               day (16 h); K_req: results/runs/okpo_main_screen/demand.csv
  results/runs/yantai_screen_stable_main/config.json          the same, Yantai 60, 70 tasks a day;
                                                               K_req: results/runs/yantai_main_screen/demand.csv
  results/runs/okpo_screen_stable_proportional/config.json    proportional mapping, Okpo: road width -1.5 m at 75, +1.5 m at
                                                               60, intra-area moves at 150 tasks a day; K_req:
                                                               results/runs/okpo_screen_widths/demand.csv (-1.5 m),
                                                               results/runs/okpo_screen_low/demand.csv (+1.5 m),
                                                               results/runs/okpo_intra_screen/demand.csv (intra-area moves)
K_req (the start of the search) is the four-day required fleet of the same cell in the screen runs (10 seeds, criterion of
experiments/required_fleet.py); the scenario s of a demand table is the cell s_30d (stable_fleet.kreq_table).

Run:   python experiments/screen_stable.py results/runs/<run>/config.json --out <dir> --kreq <demand.csv> [...]
                [--procs 6] [--max-tasks-per-child 2] [--max-minutes M] [--slots DIR]
Writes <dir>/config.json, stable.json ({"<scenario>|<rule>|<level>": {K_req, k0, stable, stable_tol}}), points.csv
(runner.write_points) and daily.csv (end-of-day backlog of each day of each point); <dir>/points/ is the cache for resuming.
--slots: machine-wide CPU slots as in experiments/runner.py (default: the environment variable CPU_SLOTS, off when neither is
given). At the deadline of --max-minutes the run stops with exit code 3; the same command resumes.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import runner as F                           # noqa: E402
import stable_fleet as R                     # noqa: E402
import stable_fleet_conserved as R61         # noqa: E402


def cells(cfg):
    return [(s, m, lvl) for s in cfg["scenarios"] for m in cfg["models"][s] for lvl in F.levels_of(cfg, s)]


def write_files(out):
    """points.csv (runner.write_points) and daily.csv (end-of-day backlog of each day of each point)."""
    xs = sorted((json.loads(f.read_text(encoding="utf-8")) for f in (out / "points").glob("*.json")),
                key=lambda x: (x["scen"], x["model"], x["level"], x["K"], x["seed"]))
    with open(out / "points.csv", "w", newline="", encoding="utf-8-sig") as f:
        F.write_points(f, [F.derive(x) for x in xs])
    with open(out / "daily.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["scen", "model", "level", "K", "seed", "backlog"])
        for x in xs:
            w.writerow([x["scen"], x["model"], x["level"], x["K"], x["seed"], " ".join(map(str, x["daily"]["backlog"]))])
    return len(xs)


def run(cfg, out, a):
    if (out / "config.json").exists():
        if json.loads((out / "config.json").read_text(encoding="utf-8")) != cfg:
            sys.exit(f"refusing to reuse {out}: its config.json differs from the configuration")
    else:
        out.mkdir(parents=True, exist_ok=True)
        F.write_json(out / "config.json", cfg)
    (out / "points").mkdir(exist_ok=True)
    print(f"run directory: {out}", flush=True)
    deadline = time.time() + 60 * a.max_minutes if a.max_minutes else None
    kreq, pts = R.kreq_table(a.kreq, cfg.get("yard")), R.load(out)
    prog = F.Progress(out / "progress.json", max(len(pts), len(cells(cfg)) * 6 * len(cfg["seeds"])), len(pts), a.procs,
                      cfg.get("title", ""), "screen_stable")
    while True:                                  # the 30-day search of stable_fleet_conserved
        need = []
        for s, m, lvl in cells(cfg):
            ev = R.evaluate(cfg, pts, s, m, lvl)
            ks, _ = R.search(ev, R61.k0_of(ev, kreq[(s, m, lvl)]))
            need += [f"{s}|{m}|{lvl}|{K}|{sd}" for K in ks for sd in cfg["seeds"] if (s, m, lvl, K, sd) not in pts]
        if not need:
            break
        need.sort(key=lambda k: -int(k.split("|")[3]))
        prog.d["total"] = max(prog.d["total"], len(pts) + len(need))
        _, failed, stopped = F.pool_run(cfg, out, a.procs, need, prog, a.slots, a.max_tasks_per_child, deadline)
        if failed:
            prog.write(state="failed", note=f"{failed} failed")
            sys.exit(1)
        if stopped:
            prog.write(state="paused", note="stopped at the deadline (--max-minutes); the same command resumes")
            sys.exit(3)
        pts = R.load(out)
    res = {}
    for s, m, lvl in cells(cfg):
        ev = R.evaluate(cfg, pts, s, m, lvl)
        k0 = R61.k0_of(ev, kreq[(s, m, lvl)])
        res[f"{s}|{m}|{lvl}"] = dict(K_req=kreq[(s, m, lvl)], k0=k0, **R.search(ev, k0)[1])
    F.write_json(out / "stable.json", res)
    n = write_files(out)
    prog.write(state="done", total=len(pts), done=len(pts), note=f"{len(res)} cells")
    print(f"done: {len(res)} cells, {n} points", flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", help="the stored run configuration (results/runs/*_screen_stable_*/config.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kreq", nargs="+", required=True, help="demand tables with K_req of the same cells (demand.csv of the "
                                                             "four-day screen runs, or demand.json)")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--max-tasks-per-child", type=int, default=None)
    ap.add_argument("--slots", default=os.environ.get("CPU_SLOTS"), help="directory of machine-wide CPU slots (cpu_slots.Slots); "
                                                                         "default: environment variable CPU_SLOTS, else off")
    a = ap.parse_args()
    run(F.load_config(a.config), Path(a.out), a)


if __name__ == "__main__":
    main()

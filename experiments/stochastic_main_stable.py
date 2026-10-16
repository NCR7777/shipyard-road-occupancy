"""K_stable of the main case under random handling and driving times: the 30-day criterion of experiments/daily.py for the
conserved inter-area mapping with random times (CV 0.3 and 0.6: execution in booked order with booking on projected occupancy
at mean durations, reserve_op3, and free flow) and the deterministic case at the mean durations (m15: full-path reservation
and free flow); Okpo 339, 350, 678, 700 tasks a day, 16 h and 24 h, 30 days of continuous release, 10 seeds.
For every cell (scenario x rule x daily demand) the smallest K at which K and K + 1 both meet a criterion is searched for each
of the two criteria, from K_req - 2 (stable_fleet.search; while the result falls on the start, the start moves down by 4:
stable_fleet_conserved.k0_of). K_req is the four-day required fleet of the same cell (experiments/required_fleet.py); cells
whose K_req is ">150" are not run over 30 days (K_stable ">150").

Run:   python experiments/stochastic_main_stable.py results/runs/okpo_stoch_main_stable/config.json --out <dir>
                --kreq results/runs/okpo_stoch_main_fleet/demand.csv [--procs 6] [--max-tasks-per-child 2]
                [--max-minutes M] [--slots DIR]
--kreq: demand.csv of the four-day search (the stored one is results/runs/okpo_stoch_main_fleet/demand.csv; its scenario s is
the cell s_30d). Writes <dir>/config.json, stable.json ({"<scenario>|<rule>|<level>": {K_req, k0, stable, stable_tol}}),
points.csv (runner.write_points) and daily.csv (end-of-day backlog of each day of each point); <dir>/points/ is the cache
for resuming. --slots: machine-wide CPU slots as in experiments/runner.py (default: the environment variable CPU_SLOTS, off
when neither is given). At the deadline of --max-minutes the run stops with exit code 3; the same command resumes.
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
import runner as F                        # noqa: E402
import stable_fleet as R                  # noqa: E402
import stable_fleet_conserved as R61      # noqa: E402


def pack(cfg, out, pts):
    """The per-point tables: points.csv and daily.csv."""
    order = lambda x: (x["scen"], x["model"], x["level"], x["K"], x["seed"])
    sims = sorted((F.derive(x) for x in pts.values()), key=order)
    with open(out / "points.csv", "w", newline="", encoding="utf-8-sig") as f:
        F.write_points(f, sims)
    with open(out / "daily.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, ["scen", "model", "level", "K", "seed", "backlog"])
        w.writeheader()
        w.writerows(dict(scen=x["scen"], model=x["model"], level=x["level"], K=x["K"], seed=x["seed"], backlog=" ".join(map(str, x["daily"]["backlog"])))
                    for x in sims)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", help="the stored run configuration (results/runs/okpo_stoch_main_stable/config.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kreq", nargs="+", required=True, help="demand.csv of the four-day search of the same cells")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--max-tasks-per-child", type=int, default=2)
    ap.add_argument("--slots", default=os.environ.get("CPU_SLOTS"), help="directory of machine-wide CPU slots (cpu_slots.Slots); "
                                                                         "default: environment variable CPU_SLOTS, else off")
    a = ap.parse_args()
    cfg = F.load_config(a.config)
    out = Path(a.out)
    if (out / "config.json").exists():
        if json.loads((out / "config.json").read_text(encoding="utf-8")) != cfg:
            sys.exit(f"refusing to reuse {out}: its config.json differs from the configuration")
    else:
        out.mkdir(parents=True, exist_ok=True)
        F.write_json(out / "config.json", cfg)
    (out / "points").mkdir(exist_ok=True)
    print(f"run directory: {out}", flush=True)
    deadline = time.time() + 60 * a.max_minutes if a.max_minutes else None
    kreq = R.kreq_table(a.kreq, cfg.get("yard"))         # integer K_req only: cells with ">150" are not in it
    cells = [c for c in R61.cells(cfg) if c in kreq]
    pts = R.load(out)
    prog = F.Progress(out / "progress.json", max(len(pts), len(cells) * 6 * len(cfg["seeds"])), len(pts), a.procs,
                      cfg.get("title", ""), "stochastic_main_stable")
    while True:
        need = []
        for s, m, lvl in cells:
            ev = R.evaluate(cfg, pts, s, m, lvl)
            ks, _ = R.search(ev, R61.k0_of(ev, kreq[(s, m, lvl)]))
            need += [f"{s}|{m}|{lvl}|{K}|{sd}" for K in ks for sd in cfg["seeds"] if (s, m, lvl, K, sd) not in pts]
        if not need:
            break
        need.sort(key=lambda k: -int(k.split("|")[3]))            # large fleets first
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
    for s, m, lvl in R61.cells(cfg):
        if (s, m, lvl) not in kreq:
            res[f"{s}|{m}|{lvl}"] = dict(K_req=None, note="K_req > 150 (4 days), not run over 30 days")
            continue
        ev = R.evaluate(cfg, pts, s, m, lvl)
        k0 = R61.k0_of(ev, kreq[(s, m, lvl)])
        res[f"{s}|{m}|{lvl}"] = dict(K_req=kreq[(s, m, lvl)], k0=k0, **R.search(ev, k0)[1])
    F.write_json(out / "stable.json", res)
    pack(cfg, out, pts)
    prog.write(state="done", total=len(pts), done=len(pts), note=f"{len(res)} cells")
    print(f"done: {len(res)} cells, {len(pts)} points", flush=True)


if __name__ == "__main__":
    main()

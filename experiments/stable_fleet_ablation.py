"""K_stable of the ablation of the occupancy representation (the column "fleet without backlog K_stable" of the paper's table
"ablation of the occupancy representation, main case" and of the supplementary table "ablation ..., every move scaled with
the flows"): the 30-day criterion of experiments/daily.py with the search of experiments/stable_fleet.py, for the 30-day
batches of the three representations (a vehicle count per road, loaded and empty the same width: experiments/ablation.py and
ablation_conserved.py; empty vehicles take no width: empty_rule = "free"), at Okpo's own volume on 16-h days.
  every move scaled  results/runs/okpo_ablation_stable (scenarios d16_<rep>_30d); K_req of results/runs/okpo_ablation_fleet.
  main case          results/runs/okpo_main_ablation_stable (conserved mapping, d16_c_<rep>_30d); K_req of
                     results/runs/okpo_main_ablation_fleet.
Each cell (scenario x rule x level) searches the smallest K at which K and K + 1 both meet each of the two criteria
(stable_fleet.search), from K_req - 2; where the result of a criterion falls exactly on the start, the start moves down by 4
(stable_fleet_conserved.k0_of; cells whose result is not at the start search as stable_fleet does). K_req is the four-day
required fleet of the same cell (the scenario s of the demand table is the cell s_30d; integers only).

Run:   python experiments/stable_fleet_ablation.py results/runs/okpo_ablation_stable/config.json --out <dir>
           --kreq results/runs/okpo_ablation_fleet/demand.csv [--procs N] [--max-tasks-per-child N] [--slots DIR]
           [--max-minutes M]
       python experiments/stable_fleet_ablation.py results/runs/okpo_main_ablation_stable/config.json --out <dir>
           --kreq results/runs/okpo_main_ablation_fleet/demand.csv [...]
(--max-tasks-per-child: a worker process exits after this many points and frees its memory; --slots as in
experiments/runner.py; --max-minutes: no new points after this time, exit code 3, the same command resumes.) Points are written
one by one and the search reads only the points written so far, so an interrupted run resumes by computing the missing ones.
Writes <dir>/points/ (cache), progress.json and stable.json ({"<scenario>|<rule>|<level>": {K_req, k0, stable, stable_tol}};
without k0 for the batch of the proportional mapping, the format of its stored stable.json).
"""
import argparse
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


def kreq_table(cfg, paths):
    """{(30-day scenario, rule, level): K_req}: the four-day required fleets of the demand tables `paths` (the scenario s of a
    table is the cell s_30d; integers, cells without a passing fleet are left out), for the scenarios of cfg."""
    scen = cfg["scenarios"]
    return {k: v for k, v in R.kreq_table(paths, cfg.get("yard")).items() if k[0] in scen}


def results(cfg, pts, kreq):
    """{cell: dict(K_req, k0, stable, stable_tol)} (the content of stable.json; while fleets remain to be computed a criterion
    may be missing). The batch of the proportional mapping (no demand_map) does not record k0, as its stored stable.json."""
    out = {}
    for s, m, lvl in R61.cells(cfg):
        ev = R.evaluate(cfg, pts, s, m, lvl)
        k0 = R61.k0_of(ev, kreq[(s, m, lvl)])
        out[f"{s}|{m}|{lvl}"] = dict(K_req=kreq[(s, m, lvl)], **({} if not mapped(cfg) else {"k0": k0}), **R.search(ev, k0)[1])
    return out


def mapped(cfg):
    """The batch of the conserved mapping (its scenarios set demand_map)."""
    return any(sp.get("demand_map") for sp in cfg["scenarios"].values())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kreq", nargs="+", required=True, help="demand tables with K_req of the same cells (demand.csv)")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--max-tasks-per-child", type=int, default=None, help="points per worker process before it exits")
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--slots", default=os.environ.get("CPU_SLOTS"), help="optional CPU-slot directory shared by runs")
    a = ap.parse_args()
    cfg = F.load_config(a.config)
    out = Path(a.out)
    (out / "points").mkdir(parents=True, exist_ok=True)
    print(f"run directory: {out}", flush=True)
    deadline = time.time() + 60 * a.max_minutes if a.max_minutes else None
    kreq = kreq_table(cfg, a.kreq)
    pts = R.load(out)
    prog = F.Progress(out / "progress.json", max(len(pts), len(R61.cells(cfg)) * 6 * len(cfg["seeds"])), len(pts), a.procs,
                      "K_stable of the ablation of the occupancy representation", "stable_ablation")
    while True:
        need = []
        for s, m, lvl in R61.cells(cfg):
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
            prog.write(state="paused", note="stopped at the time limit (--max-minutes); the same command resumes")
            sys.exit(3)
        pts = R.load(out)
    res = results(cfg, pts, kreq)
    F.write_json(out / "stable.json", res)
    prog.write(state="done", total=len(pts), done=len(pts), note=f"{len(res)} cells")
    print(f"done: {len(res)} cells, {len(pts)} points", flush=True)


if __name__ == "__main__":
    main()

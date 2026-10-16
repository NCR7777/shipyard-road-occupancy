"""Saturation curves, required fleet and 30-day stable fleet under the interventions on the binding element (the scenarios of
experiments/intervention_net.py): the main case (conserved mapping, 16 h) with the road changes of each scenario added and
full-path reservation only. runner.net_for builds intervention_net.DoorNet where a scenario has a door_stop change.
The batch configurations are the stored ones (results/runs/<yard>_interv_<part>/config.json); the base scenarios of Okpo are
the runs of the main case (okpo_main_saturation, okpo_main_fleet, okpo_main_stable), the Yantai base saturation curve is
yantai_main_saturation and its K_req at 150 and 300 tasks/day comes from the screen of the main case (yantai_main_screen).
  sat     two yards x intervention scenarios x full-path reservation, the K grid of the main-case saturation curves, 10 seeds and
          one bound sample per seed (4 days): results/runs/{okpo,yantai}_interv_saturation;
  demand  required fleet (4 days, criterion of experiments/required_fleet.py, which computes it): Okpo 339, 350, 678, 700
          tasks/day; Yantai its own demand of 23 tasks/day and 150 tasks/day (demand ratio 12.0 %, the level of the screen
          nearest the 12.44 % upper end of 16 h), with the Yantai base at 23 tasks/day (not in the main-case runs):
          results/runs/{okpo,yantai}_interv_fleet; Yantai also at 300 tasks/day (demand ratio 24.0 %; under the current layout
          full-path reservation needs 21 vehicles and free flow 11, and only at this level does it show whether the door
          binds), intervention scenarios only: results/runs/yantai_interv_fleet_high;
  stable  30-day stable fleet under continuous release (criterion and search of experiments/stable_fleet.py and
          stable_fleet_conserved.py: from K_req - 2, the smallest K at which K and K + 1 both meet the criterion) on the cells of
          demand, the Yantai base included (no other 30-day run has it): results/runs/{okpo,yantai}_interv_stable. K_req comes from
          the demand tables (--kreq).

Run:   python experiments/intervention_runs.py --part sat|demand|stable results/runs/<run>/config.json --out <dir>
           [--kreq TABLE ...] [--procs N] [--max-minutes M] [--max-tasks-per-child N] [--slots DIR]
       e.g. --part stable results/runs/yantai_interv_stable/config.json --kreq results/runs/yantai_interv_fleet/demand.csv
           results/runs/yantai_interv_fleet_high/demand.csv results/runs/yantai_main_screen/demand.csv
       and  --part stable results/runs/okpo_interv_stable/config.json --kreq results/runs/okpo_interv_fleet/demand.csv
(--kreq as in stable_fleet.py: rows of the configuration's yard with an integer K_req; --max-tasks-per-child: a worker process
exits after this many points and frees its memory; --slots: machine-wide CPU slots as in experiments/runner.py, default the
environment variable CPU_SLOTS, off when neither is given; --max-minutes: no new points after this time, exit code 3, the same
command resumes; not passed on to required_fleet.py). Writes to <dir>: points/ (cache) and points.csv; sat also
bound_points.json (the bound samples); demand: demand.csv (required_fleet.py); stable: stable.json and daily.csv (end-of-day
backlog of every 30-day point).
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
import stable_fleet_conserved as SC       # noqa: E402


def write_points(out):
    sims = sorted(F.load_points(out)[0], key=lambda x: (x["scen"], x["model"], x["level"], x["K"], x["seed"]))
    with open(out / "points.csv", "w", newline="", encoding="utf-8-sig") as f:
        F.write_points(f, sims)
    return sims


def stable(cfg, out, a, deadline):
    """As the 30-day search of stable_fleet_conserved (search stable_fleet.search, start stable_fleet_conserved.k0_of), K_req
    from the tables --kreq; writes stable.json, points.csv, daily.csv."""
    kreq = R.kreq_table(a.kreq, cfg.get("yard"))
    pts = R.load(out)
    cells = SC.cells(cfg)
    prog = F.Progress(out / "progress.json", max(len(pts), len(cells) * 6 * len(cfg["seeds"])), len(pts), a.procs,
                      cfg.get("title", ""), "intervention_runs")
    while True:
        need = []
        for s, m, lvl in cells:
            ev = R.evaluate(cfg, pts, s, m, lvl)
            ks, _ = R.search(ev, SC.k0_of(ev, kreq[(s, m, lvl)]))
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
            prog.write(state="paused", note="stopped at --max-minutes; the same command resumes")
            sys.exit(3)
        pts = R.load(out)
    res = {}
    for s, m, lvl in cells:
        ev = R.evaluate(cfg, pts, s, m, lvl)
        k0 = SC.k0_of(ev, kreq[(s, m, lvl)])
        res[f"{s}|{m}|{lvl}"] = dict(K_req=kreq[(s, m, lvl)], k0=k0, **R.search(ev, k0)[1])
    F.write_json(out / "stable.json", res)
    sims = write_points(out)
    import csv                        # end-of-day backlogs, one row per point (K_stable can be recomputed from it and points.csv)
    with open(out / "daily.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, ["scen", "model", "level", "K", "seed", "backlog"])
        w.writeheader()
        w.writerows(dict(scen=x["scen"], model=x["model"], level=x["level"], K=x["K"], seed=x["seed"],
                         backlog=" ".join(map(str, pts[(x["scen"], x["model"], x["level"], x["K"], x["seed"])]["daily"]["backlog"]))) for x in sims)
    prog.write(state="done", total=len(pts), done=len(pts), note=f"{len(res)} cells")
    print(f"done: {len(res)} cells, {len(pts)} points", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", help="stored configuration of the batch (results/runs/<yard>_interv_<part>/config.json)")
    ap.add_argument("--part", choices=("sat", "demand", "stable"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kreq", nargs="+", default=None, help="stable: demand tables with K_req of the same cells (demand.csv)")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--max-tasks-per-child", type=int, default=None)
    ap.add_argument("--slots", default=os.environ.get("CPU_SLOTS"), help="directory of machine-wide CPU slots (cpu_slots.Slots); "
                                                                         "default: environment variable CPU_SLOTS, else off")
    a = ap.parse_args()
    if a.part == "stable" and not a.kreq:
        ap.error("--part stable needs --kreq")
    cfg = F.load_config(a.config)
    out = Path(a.out)
    (out / "points").mkdir(parents=True, exist_ok=True)
    deadline = time.time() + 60 * a.max_minutes if a.max_minutes else None
    if a.part == "demand":                    # required_fleet.py searches on the same configuration
        import required_fleet
        sys.argv = [sys.argv[0], a.config, "--out", str(out), "--procs", str(a.procs)] + (["--slots", a.slots] if a.slots else [])
        return required_fleet.main()
    if a.part == "stable":
        return stable(cfg, out, a, deadline)
    keys = F.point_keys(cfg)
    todo = sorted((k for k in keys if not (out / "points" / F.fname(k)).exists()),
                  key=lambda k: -int(k.split("|")[3]) if k.count("|") == 4 else 0)
    prog = F.Progress(out / "progress.json", len(keys), len(keys) - len(todo), a.procs, cfg.get("title", ""), "intervention_runs")
    _, failed, stopped = F.pool_run(cfg, out, a.procs, todo, prog, a.slots, a.max_tasks_per_child, deadline)
    if stopped:
        prog.write(state="paused", note="stopped at --max-minutes; the same command resumes")
        sys.exit(3)
    sims = write_points(out)
    bpts = sorted(F.load_points(out)[1].values(), key=lambda x: (x["scen"], x["seed"]))
    F.write_json(out / "bound_points.json", bpts)
    prog.write(state="failed" if failed else "done", note=f"{failed} failed" if failed else f"{len(sims)} points")
    print(f"done: {len(sims)} points, {len(bpts)} bound samples, {failed} failed", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

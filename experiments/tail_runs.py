"""Batches of booking the tail (conserved inter-area mapping of the main case, deterministic 10-min loading and unloading,
16 h): full-path reservation booking the tail (reserve_tlw) and the current rule, release by timer, with the readings counted
by the same definition (reserve_tlw_timer); the rules are those of experiments/tail_booking.py. The configurations are the
stored ones of the runs (results/runs/okpo_tail_saturation, yantai_tail_saturation, okpo_tail_fleet, okpo_tail_stable).
  sat     saturation curves of both yards: the K grid of the conserved-mapping saturation runs x 10 seeds x both rules; no
          sampled bound. -> points.csv, tail.csv
  demand  K_req of Okpo at 339 and 350 tasks a day (4 days), both rules: search and criterion of required_fleet, run on
          this directory. -> points.csv, demand.csv, tail.csv
  stable  K_stable of the same two levels (30 days of continuous release) for reserve_tlw, searched from K_req - 2 of the
          demand run (--kreq: its demand.csv, the stored one results/runs/okpo_tail_fleet/demand.csv); criterion and search
          of stable_fleet and stable_fleet_conserved. -> stable.json, points.csv, daily.csv, tail.csv
Every per-point reading the summaries need is written to these files; points/ is the cache for resuming. tail.csv holds the
tail readings of each point (TAIL_COLS, metrics()["tail"] of tail_booking).
Only the complete definition of booking the tail is here: the earlier definitions of the tail booking and the control
without its two additions are not part of the release.

Run:   python experiments/tail_runs.py --part sat results/runs/okpo_tail_saturation/config.json --out <dir> [--procs N]
                [--max-minutes M] [--slots DIR]
       python experiments/tail_runs.py --part demand results/runs/okpo_tail_fleet/config.json --out <dir> [--procs N]
                [--slots DIR]
       python experiments/tail_runs.py --part stable results/runs/okpo_tail_stable/config.json --out <dir>
                --kreq <demand.csv of the demand run> [--procs N] [--max-tasks-per-child 2] [--max-minutes M] [--slots DIR]
At the deadline of --max-minutes the run stops with exit code 3; the same command resumes.
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
import runner as R                           # noqa: E402
import stable_fleet as S                     # noqa: E402
import stable_fleet_conserved as SC          # noqa: E402

TAIL_COLS = ("tasks", "tail_tasks", "res", "res_o", "res_d", "node", "road", "stop", "other", "hold_s", "bridge", "rev", "regrow", "behind_zero",
             "behind_short", "zero")
js = lambda c: json.loads(json.dumps(c, ensure_ascii=False))
BOOK = "reserve_tlw"            # the rule booking the tail; its K_req starts the 30-day search


def prepare(cfg, out):
    if (out / "config.json").exists():
        if json.loads((out / "config.json").read_text(encoding="utf-8")) != js(cfg):
            sys.exit(f"refusing to reuse {out}: its config.json differs from the configuration")
    else:
        out.mkdir(parents=True, exist_ok=True)
        R.write_json(out / "config.json", cfg)
    (out / "points").mkdir(exist_ok=True)
    print(f"run directory: {out}", flush=True)
    return out


def write_files(out, daily=False):
    """points.csv (runner.write_points), tail.csv (the tail readings of each point); for 30 days also daily.csv (the
    end-of-day backlog of each day)."""
    xs = sorted((json.loads(f.read_text(encoding="utf-8")) for f in (out / "points").glob("*.json")),
                key=lambda x: (x["scen"], x["model"], x["level"], x["K"], x["seed"]))
    with open(out / "points.csv", "w", newline="", encoding="utf-8-sig") as f:
        R.write_points(f, [R.derive(x) for x in xs])
    with open(out / "tail.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["scen", "model", "level", "K", "seed", "throughput", "T1_tight_mix"] + list(TAIL_COLS))
        for x in xs:
            t = x["tail"]
            w.writerow([x["scen"], x["model"], x["level"], x["K"], x["seed"], x["throughput"], x["T1_tight_mix"]] + [t.get(k, 0) for k in TAIL_COLS])
    if daily:
        with open(out / "daily.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["scen", "model", "level", "K", "seed", "backlog"])
            for x in xs:
                w.writerow([x["scen"], x["model"], x["level"], x["K"], x["seed"], " ".join(map(str, x["daily"]["backlog"]))])
    return len(xs)


def kreq_table(path):
    """{(30-day scenario, rule, level): K_req}: the results of the booking rule BOOK in the demand table of the demand run
    (demand.csv; K_req ">150" etc. left out)."""
    rows = csv.DictReader(open(path, encoding="utf-8-sig"))
    return {("d16_c_30d", r["model"], r["level"]): int(r["K_req"]) for r in rows
            if r["model"] == BOOK and r["K_req"].isdigit()}


def stable(cfg, out, a, deadline):
    """The 30-day search of stable_fleet_conserved (from K_req - 2; while the result is censored at the lower end the start
    moves down by 4)."""
    kreq = kreq_table(a.kreq)
    cells = [(s, m, lvl) for s in cfg["scenarios"] for m in cfg["models"][s] for lvl in cfg["demand"]]
    pts = S.load(out)
    prog = R.Progress(out / "progress.json", max(len(pts), len(cells) * 6 * len(cfg["seeds"])), len(pts), a.procs,
                      cfg.get("title", ""), "tail_runs")
    while True:
        need = []
        for s, m, lvl in cells:
            ev = S.evaluate(cfg, pts, s, m, lvl)
            ks, _ = S.search(ev, SC.k0_of(ev, kreq[(s, m, lvl)]))
            need += [f"{s}|{m}|{lvl}|{K}|{sd}" for K in ks for sd in cfg["seeds"] if (s, m, lvl, K, sd) not in pts]
        if not need:
            break
        need.sort(key=lambda k: -int(k.split("|")[3]))
        prog.d["total"] = max(prog.d["total"], len(pts) + len(need))
        _, failed, stopped = R.pool_run(cfg, out, a.procs, need, prog, a.slots, a.max_tasks_per_child, deadline)
        if failed:
            prog.write(state="failed", note=f"{failed} failed")
            sys.exit(1)
        if stopped:
            prog.write(state="paused", note="stopped at the deadline (--max-minutes); the same command resumes")
            sys.exit(3)
        pts = S.load(out)
    res = {}
    for s, m, lvl in cells:
        ev = S.evaluate(cfg, pts, s, m, lvl)
        k0 = SC.k0_of(ev, kreq[(s, m, lvl)])
        res[f"{s}|{m}|{lvl}"] = dict(K_req=kreq[(s, m, lvl)], k0=k0, **S.search(ev, k0)[1])
    R.write_json(out / "stable.json", res)
    n = write_files(out, daily=True)
    prog.write(state="done", total=len(pts), done=len(pts), note=f"{len(res)} cells")
    print(f"done: {len(res)} cells, {n} points", flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--part", choices=("sat", "demand", "stable"), required=True)
    ap.add_argument("config", help="the stored run configuration")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kreq", help="stable: demand.csv of the demand run (K_req of reserve_tlw)")
    ap.add_argument("--procs", type=int, default=23)
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--max-tasks-per-child", type=int, default=None)
    ap.add_argument("--slots", default=os.environ.get("CPU_SLOTS"), help="optional CPU-slot directory shared by runs "
                    "(default: environment variable CPU_SLOTS; without it an ordinary process pool)")
    a = ap.parse_args()
    if a.part == "stable" and not a.kreq:
        ap.error("--part stable needs --kreq")
    cfg = R.load_config(a.config)
    deadline = time.time() + 60 * a.max_minutes if a.max_minutes else None
    if a.part == "sat":
        out = prepare(cfg, Path(a.out))
        failed, stopped = R.sweep(cfg, out, a.procs, deadline, a.slots)
        if stopped:
            print("stopped at the deadline; the same command resumes", flush=True)
            sys.exit(3)
        print(f"done; {write_files(out)} points, {failed} failed", flush=True)
        sys.exit(1 if failed else 0)
    if a.part == "demand":            # required_fleet searches on this directory by its config.json, then tail.csv is added
        out = prepare(cfg, Path(a.out))
        import required_fleet
        sys.argv = [sys.argv[0], a.config, "--out", str(out), "--procs", str(a.procs)] + \
            (["--slots", a.slots] if a.slots else [])
        required_fleet.main()
        write_files(out)
        return
    stable(cfg, prepare(cfg, Path(a.out)), a, deadline)


if __name__ == "__main__":
    main()

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

Plateau of the saturation curves under random times (the lower-end marks of the capacity interval of the main case):
       python experiments/stochastic_main_stable.py --plateau --out <dir> [--curves <run dir> <run dir>] [--scan FILE]
                [--saturation FILE]
For both yards x time level (CV 0.3, CV 0.6, m15) x rule: the plateau (mean over K = 100, 105 ... 150 of the per-seed means;
+-95 % = t interval over the seeds), its share of lambda^S, the maximum of the curve and K = 150; with the deterministic 10-min
rows (d10) of the main case beside them. Inputs (released defaults, relative to the repository root): --curves
results/runs/okpo_stoch_main_curves and yantai_stoch_main_curves (points.csv), --scan results/main_case/scan.csv (lambda^S of
the conserved mapping, q = 0, 16 h), --saturation results/main_case/saturation.csv (the d10 rows, map conserved). Writes
<dir>/main_case_plateau.csv (UTF-8 without BOM, LF; released as results/stochastic/main_case_plateau.csv).
"""
import argparse
import collections
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
csv.field_size_limit(1 << 30)
import runner as F                        # noqa: E402
import stable_fleet as R                  # noqa: E402
import stable_fleet_conserved as R61      # noqa: E402

YARDS = ("yupu", "yantai")
LEVELS = ("cv3", "cv6", "m15")
PLAT_KS = tuple(range(100, 151, 5))
T975 = {9: 2.262, 7: 2.365, 4: 2.776}


# ---------------------------------------------------------------- plateau of the saturation curves (--plateau)
def read(path):
    return list(csv.DictReader(open(path, encoding="utf-8-sig"))) if path.exists() else []


def num(v):
    return None if v in ("", None) else float(v)


def mean_ci(xs):
    n = len(xs)
    if n < 2:
        return (xs[0] if xs else None), None
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, T975.get(n - 1, 2.0) * sd / math.sqrt(n)


def lam_s(scan):
    """lambda^S of the conserved mapping per yard (q = 0, whole roads, 16 h): scan.csv of conserved_bounds or the released
    results/main_case/scan.csv."""
    import conserved_bounds as TB
    q0 = TB.q0_lam(scan)
    return {y: q0[y] for y in YARDS}


def curves(y, pts):
    by = collections.defaultdict(list)
    for r in pts:
        by[(r["scen"], r["model"], int(r["K"]))].append(float(r["throughput"]))
    return [dict(yard=y, level=s.split("_")[0], model=m, K=K, n=len(v), mean=sum(v) / len(v), lo=min(v), hi=max(v))
            for (s, m, K), v in sorted(by.items(), key=lambda kv: (LEVELS.index(kv[0][0].split("_")[0]), kv[0][1], kv[0][2]))]


def plateau(sat, cv, lam, saturation):
    """sat: {yard: rows of points.csv}. Plateau = mean over K = 100, 105 ... 150; +-95 % = t interval of the per-seed
    plateaus. saturation: the table of the deterministic main case (rows of map conserved)."""
    out = []
    for y in YARDS:
        for lv in LEVELS:
            for m in ("reserve_op3", "reserve", "free"):
                c = {r["K"]: r for r in cv if (r["yard"], r["level"], r["model"]) == (y, lv, m)}
                if not c or any(K not in c for K in PLAT_KS):
                    continue
                per = collections.defaultdict(list)
                for r in sat[y]:
                    if (r["scen"], r["model"]) == (f"{lv}_c", m) and int(r["K"]) in PLAT_KS:
                        per[int(r["seed"])].append(float(r["throughput"]))
                plat, ci = mean_ci([sum(v) / len(v) for v in per.values()])
                kmax = max(c, key=lambda K: c[K]["mean"])
                out.append(dict(yard=y, level=lv, model=m, lam_S=lam[y], plateau=plat, plateau_ci=ci, ratio=plat / lam[y], max=c[kmax]["mean"], K_at_max=kmax,
                                at_150=c[150]["mean"], lo_150=c[150]["lo"], hi_150=c[150]["hi"]))
    det = {(r["yard"], r["model"]): r for r in read(saturation) if r.get("map", "conserved") == "conserved"}
    for y in YARDS:
        for m in ("reserve", "free"):
            r = det.get((y, m))
            if r:
                out.append(dict(yard=y, level="d10", model=m, lam_S=lam[y], plateau=float(r["plateau"]), ratio=float(r["plateau"]) / lam[y],
                                max=num(r.get("max")), K_at_max=r.get("K_at_max"), at_150=num(r.get("at_150"))))
    return out


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:          # UTF-8 without BOM, LF: as the released tables
        w = csv.DictWriter(f, keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def plateau_main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lam = lam_s(ROOT / a.scan)
    cv, sat = [], {}
    for y, d in zip(YARDS, a.curves):
        sat[y] = read(ROOT / d / "points.csv")
        cv += curves(y, sat[y])
    pl = plateau(sat, cv, lam, ROOT / a.saturation)
    write_csv(out / "main_case_plateau.csv", pl)
    print(f"{len(pl)} rows -> {out / 'main_case_plateau.csv'}", flush=True)


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
    ap.add_argument("config", nargs="?", help="the stored run configuration (results/runs/okpo_stoch_main_stable/config.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kreq", nargs="+", help="demand.csv of the four-day search of the same cells")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--max-tasks-per-child", type=int, default=2)
    ap.add_argument("--slots", default=os.environ.get("CPU_SLOTS"), help="directory of machine-wide CPU slots (cpu_slots.Slots); "
                                                                         "default: environment variable CPU_SLOTS, else off")
    ap.add_argument("--plateau", action="store_true", help="write the plateau table of the saturation curves instead")
    ap.add_argument("--curves", nargs=2, default=["results/runs/okpo_stoch_main_curves", "results/runs/yantai_stoch_main_curves"],
                    help="--plateau: the saturation runs of Okpo and Yantai")
    ap.add_argument("--scan", default="results/main_case/scan.csv", help="--plateau: lambda^S of the conserved mapping (q = 0)")
    ap.add_argument("--saturation", default="results/main_case/saturation.csv", help="--plateau: the deterministic 10-min rows")
    a = ap.parse_args()
    if a.plateau:
        return plateau_main(a)
    if not (a.config and a.kreq):
        ap.error("the 30-day search needs the configuration and --kreq")
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

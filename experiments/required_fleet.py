"""Required fleet of a demand level: the smallest fleet K whose ten-seed mean throughput (statistics window: days 2-4 after one
day of warm-up) reaches 0.99 x the daily vehicle demand (daily demand less the lifts by gantry cranes; without lifts, the
daily demand). Points are written one by one and the search is stateless (it is read from the points written so far), so an
interrupted run resumes by computing the missing points.

Finite demand: each day's tasks are released in 8 batches at equal intervals over the working day (16 or 24 h), for 4 days.
The search, on integer K:
  free flow, full-path reservation and the reference rule (throughput rises with K): first K = 150; not reached -> ">150";
    otherwise bisection on the integers between the last fleet that fails and the first that reaches the target; then
    K_req + 1 ... + 3 are tested and marked if the target fails again above K_req (fail_above);
  safe release (throughput rises and then falls with K): SAFE_LOW, SAFE_PEAK of the yard and SAFE_SCAN first; if some
    K <= 150 reaches the target, every integer from 1 to the smallest one is computed (so K_req is the smallest integer that
    reaches it); if none does, SAFE_HIGH (100-200) is scanned for the lowest level whose target fails at K <= 150, with
    ">200" if it fails there too; finally K_req + 1 ... + 5 are tested.
Run:   python experiments/required_fleet.py results/runs/<run>/config.json --out <dir> [--procs N]
(the configurations of the stored searches: results/runs/*_fleet_*/config.json). Writes <dir>/points/, points.csv and
demand.csv (one row per search: scenario, rule, level, K_req, the fleets probed and their mean throughputs).
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import runner as R      # noqa: E402

SAFE_SCAN = (20, 30, 40, 50, 60, 80, 100, 120)
SAFE_LOW = (10, 15)
SAFE_PEAK = {"yupu": (31,), "yantai": (18, 20)}        # fleet of the largest saturated throughput of safe release, main scenario
SAFE_HIGH = tuple(range(100, 201, 10))
ABOVE = {"segment_safe": 5}                            # fleets tested above K_req (default 3)


def truck_demand(level, lift_rel):
    """Daily vehicle demand: daily demand less the lifts released in the statistics window (none without crane spans)."""
    return level - lift_rel


def load(out):
    pts = {}
    for f in (out / "points").glob("*.json"):
        x = json.loads(f.read_text(encoding="utf-8"))
        pts[(x["scen"], x["model"], x["level"], x["K"], x["seed"])] = x
    return pts


def evaluate(cfg, pts, s, m, lvl, K):
    """(mean throughput, target) when all seeds of the cell are computed, else None. Target = 0.99 x daily vehicle demand."""
    xs = [pts.get((s, m, lvl, K, sd)) for sd in cfg["seeds"]]
    if any(x is None for x in xs):
        return None
    lift = sum((x.get("lift") or {}).get("released_per_day", 0.0) for x in xs) / len(xs)
    return sum(x["throughput"] for x in xs) / len(xs), 0.99 * truck_demand(cfg["demand"][lvl], lift)


def search(cfg, pts, s, m, lvl):
    """(fleets to compute next, empty when decided; the result or None), from the points written so far."""
    Ks = sorted({k[3] for k in pts if k[:3] == (s, m, lvl)} | {150})
    ev = {K: e for K in Ks if (e := evaluate(cfg, pts, s, m, lvl, K)) is not None}
    safe = m == "segment_safe"
    probes = sorted(set(SAFE_LOW + SAFE_PEAK[cfg["yard"]] + SAFE_SCAN + (150,))) if safe else (150,)
    todo = [K for K in probes if K not in ev]
    if todo:
        return todo, None
    ok = {K: e[0] >= e[1] for K, e in ev.items()}
    target = ev[150][1]
    passing = [K for K in ev if ok[K]]

    # the K = 100-200 scan is done only for the lowest level of a scenario and rule whose target fails at K <= 150 (the
    # lower levels have a fleet that reaches theirs); higher levels are recorded as ">150" with high_scan = False
    def reached(lv):
        es = (evaluate(cfg, pts, s, m, lv, k) for k in {k[3] for k in pts if k[:3] == (s, m, lv)})
        return any(e[0] >= e[1] for e in es if e is not None)
    high = safe and all(reached(lv) for lv, x in cfg["demand"].items() if x < cfg["demand"][lvl])
    if high and not any(ok[K] for K in ev if K <= 150):
        todo = [K for K in SAFE_HIGH if K not in ev]
        if todo:
            return todo, None
    probed = {K: round(e[0], 2) for K, e in sorted(ev.items())}
    if not passing:
        kmax = max(ev, key=lambda K: ev[K][0])
        return [], dict(K_req=None, over=True, K_top=max(ev), thr_top=ev[max(ev)][0], thr_K150=ev[150][0], target=target,
                        K_best=kmax, thr_best=ev[kmax][0], after_fail=[], fail_above=[], nonmono=False, probed=probed,
                        high_scan=high if safe else None)
    K1 = min(passing)
    lo = max((K for K in ev if K < K1), default=0)            # every fleet below K1 fails (K1 is the smallest that passes)
    if K1 - lo > 1:
        if not safe:                                         # monotone rules: bisection on the integers
            return [(lo + K1) // 2], None
        # safe release: every integer below K1 (from 1 when K1 <= 150, so K_req is the smallest integer that reaches the
        # target); above 150 only those between K1 and the previous fleet
        return [K for K in range(1 if K1 <= 150 else lo + 1, K1) if K not in ev], None
    above = range(K1 + 1, K1 + 1 + ABOVE.get(m, 3))
    todo = [K for K in above if K not in ev]
    if todo:
        return todo, None
    after = [K for K in above if not ok[K]]
    fail_above = [K for K in ev if K > K1 and not ok[K]]
    return [], dict(K_req=K1, over=False, thr=ev[K1][0], target=target, K_below=lo or None, thr_below=ev[lo][0] if lo else None,
                    thr_K150=ev[150][0], after_fail=after, fail_above=fail_above, nonmono=bool(fail_above), probed=probed)


def summarize(cfg, out, searches, pts):
    rows = []
    for s, m, lvl in searches:
        _, r = search(cfg, pts, s, m, lvl)
        sp = R.scen_params(cfg, s)
        rows.append(dict(yard=cfg["yard"], scen=s, day_h=sp["day_s"] / 3600, section=sp.get("section") or "whole", model=m, level=lvl,
                         demand=cfg["demand"][lvl], thr=r.get("thr"),
                         K_req=r["K_req"] if not r["over"] else f">{r['K_top'] if r.get('high_scan') is not False else 150}",
                         target=r["target"], K_below=r.get("K_below"), thr_below=r.get("thr_below"), thr_K150=r["thr_K150"],
                         K_best=r.get("K_best"), thr_best=r.get("thr_best"), after_fail=json.dumps(r["after_fail"]),
                         fail_above=json.dumps(r["fail_above"]), nonmono=r["nonmono"], high_scan=r.get("high_scan"),
                         probed=json.dumps(r["probed"])))
    with open(out / "demand.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    sims = sorted((R.derive(x) for x in pts.values()), key=lambda x: (x["scen"], x["model"], x["level"], x["K"], x["seed"]))
    with open(out / "points.csv", "w", newline="", encoding="utf-8-sig") as f:
        R.write_points(f, sims)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    cfg = R.load_config(a.config)
    out = Path(a.out)
    (out / "points").mkdir(parents=True, exist_ok=True)
    searches = [(s, m, lvl) for s in cfg["scenarios"] for m in cfg["models"][s] for lvl in R.levels_of(cfg, s)]
    pts = load(out)
    while True:
        need = []
        for s, m, lvl in searches:
            for k in search(cfg, pts, s, m, lvl)[0]:
                need += [f"{s}|{m}|{lvl}|{k}|{sd}" for sd in cfg["seeds"] if (s, m, lvl, k, sd) not in pts]
        if not need:
            break
        need.sort(key=lambda k: -int(k.split("|")[3]))           # large fleets first
        R.run_pool(cfg, out, min(a.procs, len(need)), need)
        pts = load(out)
    rows = summarize(cfg, out, searches, pts)
    print(f"{len(rows)} searches, {len(pts)} points; demand.csv in {out}")


if __name__ == "__main__":
    main()

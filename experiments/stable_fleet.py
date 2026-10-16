"""The 30-day criterion K_stable: under continuous release for 30 days, the smallest fleet whose end-of-day backlog has no
positive trend (experiments/daily.py), for every cell (scenario x rule x daily demand) of a configuration whose scenarios have
days = 30 and daily_stats (Okpo, the proportional mapping and the mapping with intra-area moves, 16 / 24 h, free flow and
full-path reservation). Points are written one by one and the search reads only the points written so far, so an interrupted
run resumes by computing the missing points.

Each of the two criteria (main and tolerance; K_stable is the smallest K at which K and K + 1 both meet it) is searched from
K_req - 2: first K_req - 2 ... K_req + 1 one by one, then the jumps and equal splits of search(); K at most KMAX. K_req is the
required fleet of the same cell in four days (experiments/required_fleet.py), read from the demand tables of those searches
(--kreq: demand.csv of required_fleet.py, a table with the same columns such as results/fleet/required_fleet.csv, or a
demand.json with the same keys; rows with an integer K_req, of the configuration's yard where the table has a yard column;
the table's scenario s is the cell s_30d).

Run:   python experiments/stable_fleet.py results/runs/<run>/config.json --out <dir> --kreq <demand.csv> [<demand.csv> ...]
       [--procs N] [--slots DIR]
       e.g. results/runs/okpo_stable_30day/config.json with --kreq results/fleet/required_fleet.csv
       results/runs/okpo_intra_fleet/demand.csv (the proportional mapping and the mapping with intra-area moves)
(--slots: machine-wide CPU slots as in experiments/runner.py, default the environment variable CPU_SLOTS, off when neither is
given). The saturation curves of the same mappings are ordinary batches of experiments/runner.py. Writes <dir>/points/ (cache)
and <dir>/stable.json ({"<scenario>|<rule>|<level>": {K_req, stable, stable_tol}}).
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
import runner as F                        # noqa: E402
import daily as Dly                       # noqa: E402

KMAX = 150            # top of the K grid of the saturation curves and the required-fleet table (">150"); with the proportional
# mapping at 16 h and peak demand, full-path reservation has a plateau of 656 < 678 at K = 150


def kreq_table(paths, yard=None):
    """{(30-day scenario, rule, level): K_req}: the required fleets of the demand tables `paths` (integers; cells without a
    passing fleet are left out); rows of tables with a yard column only for `yard`."""
    out = {}
    for f in map(Path, paths):
        if f.suffix == ".json":
            rows = json.loads(f.read_text(encoding="utf-8"))
        else:
            with open(f, encoding="utf-8-sig") as fh:
                rows = [dict(r, K_req=int(r["K_req"]) if r["K_req"].isdigit() else r["K_req"]) for r in csv.DictReader(fh)]
        for r in rows:
            if isinstance(r["K_req"], int) and r.get("yard", yard) == yard:
                out[(f"{r['scen']}_30d", r["model"], r["level"])] = r["K_req"]
    return out


def load(out):
    """{(scenario, rule, level, K, seed): point result}."""
    pts = {}
    for f in (out / "points").glob("*.json"):
        x = json.loads(f.read_text(encoding="utf-8"))
        pts[(x["scen"], x["model"], x["level"], x["K"], x["seed"])] = x
    return pts


def evaluate(cfg, pts, s, m, lvl):
    """{K: daily.trend}: the fleets whose ten seeds are all computed."""
    ev = {}
    for K in sorted({k[3] for k in pts if k[:3] == (s, m, lvl)}):
        xs = [pts.get((s, m, lvl, K, sd)) for sd in cfg["seeds"]]
        if all(xs):
            ev[K] = Dly.trend([Dly.slope(x["daily"]["backlog"]) for x in xs], cfg["demand"][lvl])
    return ev


def search(ev, k0, kmax=KMAX):
    """(fleets still to compute, {criterion: K_stable or None}). For each criterion:
      1. first k0 ... k0 + 3 (K_req - 2 ... K_req + 1, one by one);
      2. lo = the largest computed fleet that fails (read as monotone in K: every fleet below lo fails); with lo + 1 and
         lo + 2 both computed (and so both passing) K_stable = lo + 1;
      3. a passing fleet hi above lo with a gap in between: three equal split points of (lo, hi) (only lo + 1, lo + 2 when
         adjacent);
      4. nothing computed above lo: jump to lo + 4 and lo + 12, still failing then kmax; kmax failing too gives None (no
         stable fleet with K <= kmax).
    In the monotone case the result equals a scan of every integer from K_req - 2 until two adjacent fleets pass, with fewer
    intermediate fleets for cells far above K_req (a 30-day point of a large fleet takes 5-15 min). Otherwise it takes the
    smallest K that passes together with every computed fleet above it, which is more conservative than a scan."""
    need, res = set(), {}
    for crit in ("stable", "stable_tol"):
        first = [K for K in range(k0, k0 + 4) if K not in ev]
        if first:
            need |= set(first)
            continue
        lo = max((K for K in ev if K >= k0 and not ev[K][crit]), default=k0 - 1)
        above = sorted(K for K in ev if K > lo)
        if above and above[0] == lo + 1:
            if lo + 2 in ev or lo + 1 == kmax:
                res[crit] = lo + 1
            else:
                need.add(lo + 2)
        elif above:
            need |= {lo + (above[0] - lo) * j // 4 for j in (1, 2, 3)} - {lo, above[0]}
        elif lo >= kmax:
            res[crit] = None
        elif lo < k0 + 3 + 12:
            need |= {min(lo + 4, kmax), min(lo + 12, kmax)}
        else:
            need.add(kmax)
    return sorted(need), res


def cells(cfg):
    return [(s, m, lvl) for s in cfg["scenarios"] for m in cfg["models"][s] for lvl in cfg["demand"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kreq", nargs="+", required=True, help="demand tables with K_req of the same cells (demand.csv or demand.json)")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--slots", default=os.environ.get("CPU_SLOTS"), help="directory of machine-wide CPU slots (cpu_slots.Slots); "
                                                                         "default: environment variable CPU_SLOTS, else off")
    a = ap.parse_args()
    cfg = F.load_config(a.config)
    out = Path(a.out)
    (out / "points").mkdir(parents=True, exist_ok=True)
    kreq = kreq_table(a.kreq, cfg.get("yard"))
    pts = load(out)
    while True:
        need = []
        for s, m, lvl in cells(cfg):
            ks, _ = search(evaluate(cfg, pts, s, m, lvl), max(1, kreq[(s, m, lvl)] - 2))
            need += [f"{s}|{m}|{lvl}|{K}|{sd}" for K in ks for sd in cfg["seeds"] if (s, m, lvl, K, sd) not in pts]
        if not need:
            break
        need.sort(key=lambda k: -int(k.split("|")[3]))            # large fleets first
        F.run_pool(cfg, out, min(a.procs, len(need)), need, slots_dir=a.slots)
        pts = load(out)
    res = {f"{s}|{m}|{lvl}": dict(K_req=kreq[(s, m, lvl)], **search(evaluate(cfg, pts, s, m, lvl), max(1, kreq[(s, m, lvl)] - 2))[1])
           for s, m, lvl in cells(cfg)}
    F.write_json(out / "stable.json", res)
    print(f"{len(res)} cells, {len(pts)} points; stable.json in {out}", flush=True)


if __name__ == "__main__":
    main()

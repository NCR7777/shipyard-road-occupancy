"""Tables of the holding ratios with matched methods and of the computing cost, from the per-point tables only (no points/):
the sub-network and whole-network tables of hold_ratio_matched.py, the points of intervention_dispatch.py, intervention_subnet.py
and intervention_bounds.py, and the holding ratios of the interventions (only as a check).
  hold_ratio_matched.csv  one row per scenario: the share of time the binding element is held, on the sub-network for the three
                          methods (full-path reservation, offline dispatch with the 6.4-s drive-off and with Definition 1(c))
                          and the best schedule, and on the whole network in the same definitions (range and mean); the ratio
                          with matched methods (whole network / sub-network, the same method) beside the ratio of the
                          interventions (whole-network offline dispatch / best sub-network schedule); ratio = ratio of the
                          means, lo / hi = the two ranges divided crosswise (as results/interventions/hold_ratio.csv); the
                          throughput of the three methods on the sub-network relative to lambda_W^N of the sub-network point
  computing_times.csv     computing cost (wall-clock seconds): the system-bound linear programme, full-path reservation and
                          offline dispatch once on the whole network, one sub-network point with CP-SAT and annealing, one
                          sub-network point with the matched methods

Run:   python experiments/hold_ratio_matched_report.py --out <dir>
                [--points FILE] [--whole FILE] [--subnet FILE] [--dispatch FILE] [--bounds FILE] [--hold-ratio FILE]
with the released tables as defaults (paths relative to the repository root): --points
results/interventions/hold_ratio_matched_points.csv and --whole hold_ratio_matched_whole.csv (points.csv and whole.csv of
hold_ratio_matched.py), --subnet subnet_points.csv (intervention_subnet.py), --dispatch dispatch_points.csv
(intervention_dispatch.py), --bounds bound_points.csv (intervention_bounds.py), --hold-ratio hold_ratio.csv (the ratios of the
interventions, recomputed here from the same readings and checked). Writes <dir>/hold_ratio_matched.csv and
computing_times.csv (UTF-8 without BOM, LF; released as results/interventions/hold_ratio_matched.csv, computing_times.csv);
the column source names each input by its path relative to results/.
"""
import argparse
import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import intervention_dispatch as TE        # noqa: E402
import intervention_subnet as TS          # noqa: E402
import hold_ratio_matched as M            # noqa: E402

METHODS = (("fpr", "full-path reservation"), ("clr", "offline dispatch, 6.4 s"), ("def1c", "offline dispatch, Definition 1(c)"))
DEFAULTS = dict(points="results/interventions/hold_ratio_matched_points.csv", whole="results/interventions/hold_ratio_matched_whole.csv",
                subnet="results/interventions/subnet_points.csv", dispatch="results/interventions/dispatch_points.csv",
                bounds="results/interventions/bound_points.csv", hold_ratio="results/interventions/hold_ratio.csv")


def rd(path):
    return list(csv.DictReader(open(path, encoding="utf-8-sig")))


def rel(path):
    """An input named by its path relative to results/ (else as given)."""
    p = Path(path).resolve()
    try:
        return p.relative_to((ROOT / "results").resolve()).as_posix()
    except ValueError:
        return str(path)


def rng(xs):
    return dict(min=min(xs), max=max(xs), mean=statistics.mean(xs))


def ratio(whole, sub):
    return dict(r=statistics.mean(whole) / statistics.mean(sub), lo=min(whole) / max(sub), hi=max(whole) / min(sub))


def load(a):
    pts, whole = rd(a.points), rd(a.whole)
    subs = {(r["inst"], r["seed"], r["N"]): r for r in rd(a.subnet)}
    ect = {(r["yard"], r["scen"], r["seed"]): r for r in rd(a.dispatch)}
    assert len(pts) == len(M.sub_keys()) and len(whole) == len(M.whole_keys())
    for r in pts:                              # the same instance: full-path reservation as in intervention_subnet (completion
        s = subs[(r["inst"], r["seed"], r["N"])]    # recomputed from t2 + handling, at most the last digit apart); the same binding
        assert abs(float(r["fpr_thr"]) / float(s["fpr_thr"]) - 1) < 1e-12 and r["watch1"] == s["watch1"], r["inst"]   # element; every
        assert r["fpr_check"] == r["clr_check"] == r["def1c_check"] == "True", r["inst"]                            # schedule passes its checker
    for r in whole:                            # the same whole-network point: throughputs equal those of intervention_dispatch
        e = ect[(r["yard"], r["scen"], r["seed"])]
        assert all(float(r[f"{v}_thr"]) == float(e[f"{v}_thr"]) for v, _ in METHODS), r["key"]
    return pts, whole, subs, ect


def hold_ratios(pts, subs, ect, a):
    old = {(r["yard"], r["scen"]): r for r in rd(a.hold_ratio)}
    out = []
    for name in M.SUBNETS:
        y, s = TS.INSTANCES[name]["yard"], TS.INSTANCES[name]["scen"]
        g = [r for r in pts if r["inst"] == name]
        best = [subs[(r["inst"], r["seed"], r["N"])] for r in g]
        e = [ect[(y, s, str(sd))] for sd in TE.SEEDS]
        assert all(x["watch1"] == g[0]["watch1"] for x in e), name
        sub = {v: [float(r[f"{v}_hold1"]) for r in g] for v, _ in METHODS}
        sub["best"] = [float(r["hold1"]) for r in best]
        wh = {v: [float(x[f"{v}_hold1"]) for x in e] for v, _ in METHODS}
        row = dict(yard=y, scen=s, subnet=name, element=g[0]["watch1"], n_sub=len(g), n_whole=len(e))
        for v in sub:
            row.update({f"sub_{v}_{k}": x for k, x in rng(sub[v]).items()})
        for v in wh:
            row.update({f"whole_{v}_{k}": x for k, x in rng(wh[v]).items()})
        for v, _ in METHODS:
            row.update({f"ratio_{v}{'' if k == 'r' else '_' + k}": x for k, x in ratio(wh[v], sub[v]).items()})
        row.update({f"ratio_clr_best{'' if k == 'r' else '_' + k}": x for k, x in ratio(wh["clr"], sub["best"]).items()})
        row.update({f"ratio_def1c_best{'' if k == 'r' else '_' + k}": x for k, x in ratio(wh["def1c"], sub["best"]).items()})
        o = old[(y, s)]                         # the ratio of the interventions recomputed from the same readings, equal
        assert abs(row["ratio_clr_best"] - float(o["ratio_clr"])) < 1e-12 and abs(row["ratio_clr_best_lo"] - float(o["ratio_clr_lo"])) < 1e-12, name
        lam = [float(b["lamN"]) for b in best]
        for v, _ in METHODS:
            row.update({f"sub_{v}_N_{k}": x for k, x in rng([float(r[f"{v}_thr"]) / l for r, l in zip(g, lam)]).items()})
        row.update({f"sub_best_N_{k}": x for k, x in rng([float(b["thr"]) / l for b, l in zip(best, lam)]).items()})
        out.append(row)
    return out


def cost(pts, whole, subs, a):
    yards = list(dict.fromkeys(TS.INSTANCES[n]["yard"] for n in M.SUBNETS))
    scen = {(TS.INSTANCES[n]["yard"], TS.INSTANCES[n]["scen"]) for n in M.SUBNETS}
    lp = [r for r in rd(a.bounds) if (r["yard"], r["scen"]) in scen]
    best = [r for k, r in subs.items() if k[0] in M.SUBNETS]
    ect = [r for r in rd(a.dispatch) if (r["yard"], r["scen"]) in scen]
    out = []

    def add(item, src, rows, col, note=""):
        for y in yards:
            xs = [float(r[col]) for r in rows if r["yard"] == y]
            out.append(dict(item=item, yard=y, n=len(xs), min_s=min(xs), max_s=max(xs), mean_s=statistics.mean(xs), source=src, note=note))
    add("system_lp", rel(a.bounds), lp, "secs",
        "system bound: building and solving the linear programme (map reading excluded), once per scenario")
    add("whole_fpr", rel(a.whole), whole, "fpr_secs",
        "full-path reservation over two days, K = 100 (map reading and network building included); it gives the tasks of offline dispatch")
    add("whole_ect_clr", rel(a.whole), whole, "clr_secs", "offline dispatch once, 6.4-s drive-off")
    add("whole_ect_def1c", rel(a.whole), whole, "def1c_secs", "offline dispatch once, Definition 1(c)")
    add("whole_dispatch_point", rel(a.dispatch), ect, "secs",
        "one point of intervention_dispatch.py in all: full-path reservation, upper bound, both offline dispatches, checker and "
        "held shares")
    add("sub_best", rel(a.subnet), best, "secs",
        "one sub-network point: the exact search (CP-SAT, 15 s per task), annealing in two rounds of 15 min and the CP-SAT polish; "
        "CP-SAT 1-4 threads, annealing single-threaded")
    for v, label in METHODS:
        add(f"sub_{v}", rel(a.points), pts, f"{v}_secs", f"one sub-network point: {label} (schedule only)")
    add("sub_matched_point", rel(a.points), pts, "secs", "one sub-network point in all: instance, three schedules, checker and held shares")
    return out


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:          # UTF-8 without BOM, LF: as the released tables
        w = csv.DictWriter(f, list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", default=v, help=f"default {v}")
    a = ap.parse_args()
    for k in DEFAULTS:
        setattr(a, k, ROOT / getattr(a, k))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pts, whole, subs, ect = load(a)
    hold, cst = hold_ratios(pts, subs, ect, a), cost(pts, whole, subs, a)
    write(out / "hold_ratio_matched.csv", hold)
    write(out / "computing_times.csv", cst)
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()

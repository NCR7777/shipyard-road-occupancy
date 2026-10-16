"""Whole-network offline dispatch under each intervention scenario (the method and checker of experiments/offline_dispatch.py)
and the whole-network holding of the binding elements.

Each point = yard x scenario (intervention_net.SCEN) x seed 101-103, K = 100, 2 working days, the main case (conserved mapping;
at the saturated level the tasks come from conserved_mapping.stream, as in retime.run_fpr_conserved, with only the road changes
of the scenario added). For the original schedule of full-path reservation and the offline dispatch with the two destination
definitions (offline_dispatch.ect: the 6.4 s drive-off as the main reading, definition 1(c) beside it) it records:
  throughput lambda = N·16 h / completion, the mid-run rate, and the independent checker (each in its own definition);
  holding = the share of [0, completion] during which a resource is held (at least one vehicle on it; exact.check.busy, the same
  function as for the sub-network of intervention_subnet.py): for the top three binding elements of the system bound of the
  scenario (the points.csv of experiments/intervention_bounds.py) and the binding element of lambda_W^N of the batch of these two
  days.
The three base points have the same schedules as the main-case points of offline_dispatch.py (the same throughputs to the last
digit; results/retiming/offline_dispatch.csv).

Run:   python experiments/intervention_dispatch.py --out <dir> --bounds <points.csv of intervention_bounds.py> [--procs N]
           [--slots DIR]
       with the released table: --bounds results/interventions/bound_points.csv
(--slots: machine-wide CPU slots as in experiments/runner.py, default the environment variable CPU_SLOTS, off when neither is
given). Writes to <dir>: config.json, progress.json, points/ (one JSON per point with the schedules, a cache for resuming) and
points.csv, released as results/interventions/dispatch_points.csv (the scenario table results/interventions/dispatch.csv and the
table of holding ratios results/interventions/hold_ratio.csv, whole network over sub-network of intervention_subnet.py, are drawn
from it).
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import runner as F                        # noqa: E402
import intervention_net as N              # noqa: E402

SEEDS, K, DAYS = (101, 102, 103), 100, 2
VARIANTS = (("clr", True), ("def1c", False))
BOUNDS = "INTERVENTION_BOUNDS"            # environment variable naming the bounds table (--bounds; child processes inherit it)


def code_sha():
    h = hashlib.sha256()
    for d in ("occupancy", "bounds", "exact"):
        for f in sorted((ROOT / d).glob("*.py")):
            h.update(f"{d}/{f.name}".encode() + f.read_bytes())
    for n in ("demand_mapping.py", "conserved_mapping.py", "intervention_net.py", "intervention_dispatch.py", "offline_dispatch.py",
              "retime.py"):
        h.update(f"experiments/{n}".encode() + (ROOT / "experiments" / n).read_bytes())
    return h.hexdigest()


def keys():
    return [f"{y}|{s}|{sd}" for y, ss in N.SCEN.items() for s in ss for sd in SEEDS]


def watch(y, s):
    """The top three binding elements of the system bound of the scenario (the points.csv of intervention_bounds.py named by
    --bounds): [(name, resource tuple)]."""
    for r in csv.DictReader(open(os.environ[BOUNDS], encoding="utf-8-sig")):
        if (r["yard"], r["scen"]) == (y, s):
            return [(r[f"top{i}"], tuple(r[f"res{i}"].split("|"))) for i in (1, 2, 3)]
    raise KeyError((y, s))


def run_fpr(y, s, seed):
    """Two-day saturated schedule of full-path reservation in the main case with the road changes of the scenario (as
    retime.run_fpr_conserved, the network from intervention_net.net)."""
    import conserved_mapping
    import retime as R
    from exact.model import FPR
    from occupancy.tasks import vehicle_starts
    p = N.params(y, s, days=DAYS)
    net = N.net(y, p)
    sim = FPR(net, p, "reserve", K, seed, None)
    sim.stream = conserved_mapping.stream(net, seed, p)
    sim.run()
    return R.collect(net, p, sim, vehicle_starts(net, seed, K))


def compute(arg):
    key, out = arg
    import offline_dispatch as E
    import retime as R
    from exact import check as C
    from occupancy.bound import bounds
    y, s, seed = key.split("|")
    t0 = time.time()
    D = run_fpr(y, s, int(seed))
    T, p, net, starts = D["T"], D["p"], D["net"], D["starts"]
    n = len(T)
    b = bounds(net, p, T, tight_only=True)
    w = watch(y, s)
    bind2 = (b["binding_tight"][0]["name"], tuple(map(str, b["binding_tight"][0]["res"])))
    res = [tuple(x[1]) for x in w] + [bind2[1]]
    names = [x[0] for x in w] + [bind2[0]]
    fix = lambda r: {str(tuple(map(str, k))): v for k, v in r.items()}

    def read(seq, clr):
        x = np.zeros(2 * n)
        for lst in seq.values():
            for j, a, b_ in lst:
                x[2 * j], x[2 * j + 1] = a, b_
        done = np.array([x[2 * j + 1] + D["DL"][j] for j in range(n)])
        W = float(done.max())
        c = C.check(net, p, T, starts, seq, clr=clr)
        hold = C.busy(net, p, T, starts, seq, res, clr=clr)
        return dict(W=W, thr=R.thr(n, W), mid=R.mid_rate(done), check=c["ok"], check_errors=c["errors"][:5], max_ratio=c["max_ratio"],
                    hold=fix(hold), seq={str(kk): v for kk, v in seq.items()})
    out_d = dict(key=key, yard=y, scen=s, seed=int(seed), K=K, days=DAYS, N=n, lamN=b["network_tight"], watch=names,
                 watch_res=["|".join(r) for r in res], fpr=read(D["seq"], True))
    for v, keep in VARIANTS:
        out_d[v] = read(E.ect(D, keep_clr=keep), keep)
    out_d["secs"] = time.time() - t0
    F.write_json(Path(out) / "points" / F.fname(key), out_d)
    return key, out_d["fpr"]["thr"], out_d["clr"]["thr"]


def summarize(out):
    rows = []
    for k in keys():
        x = json.loads((out / "points" / F.fname(k)).read_text(encoding="utf-8"))
        row = dict(yard=x["yard"], scen=x["scen"], seed=x["seed"], N=x["N"], lamN=x["lamN"], **{f"watch{i + 1}": x["watch"][i] for i in range(4)},
                   **{f"watch{i + 1}_res": x["watch_res"][i] for i in range(4)})
        for v in ("fpr", "clr", "def1c"):
            r = x[v]
            row.update({f"{v}_thr": r["thr"], f"{v}_mid": r["mid"], f"{v}_W": r["W"], f"{v}_check": r["check"]})
            row.update({f"{v}_hold{i + 1}": r["hold"][str(tuple(x["watch_res"][i].split("|")))] for i in range(4)})
        row["secs"] = x["secs"]
        rows.append(row)
    with open(out / "points.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--bounds", required=True, help="points.csv of experiments/intervention_bounds.py "
                                                    "(released: results/interventions/bound_points.csv)")
    ap.add_argument("--procs", type=int, default=18)
    ap.add_argument("--slots", default=os.environ.get("CPU_SLOTS"), help="directory of machine-wide CPU slots (cpu_slots.Slots); "
                                                                         "default: environment variable CPU_SLOTS, else off")
    a = ap.parse_args()
    os.environ[BOUNDS] = str(Path(a.bounds).resolve())
    out = Path(a.out)
    cfg = dict(title="offline dispatch and holding on the whole network under the interventions", keys=keys(), K=K, days=DAYS,
               variants=[v for v, _ in VARIANTS], scenarios=N.SCEN, bounds=a.bounds, code_sha256=code_sha())
    if (out / "config.json").exists() and json.loads((out / "config.json").read_text(encoding="utf-8")) != json.loads(json.dumps(cfg)):
        sys.exit(f"{out} holds a run of another configuration")
    (out / "points").mkdir(parents=True, exist_ok=True)
    F.write_json(out / "config.json", cfg)
    todo = [k for k in keys() if not (out / "points" / F.fname(k)).exists()]
    prog = F.Progress(out / "progress.json", len(keys()), len(keys()) - len(todo), a.procs, cfg["title"], "intervention_dispatch")
    if todo:
        for key, thr0, thr in F.slot_imap(compute, [(k, str(out)) for k in todo], min(a.procs, len(todo)), slots_dir=a.slots,
                                          tag="intervention_dispatch"):
            prog.log(f"{key}: full-path reservation {thr0:.0f}/day -> offline dispatch {thr:.0f}", done=prog.d["done"] + 1)
    summarize(out)
    prog.write(state="done", processes=0)
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()

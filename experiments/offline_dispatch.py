"""Whole-network offline dispatch: the same batch of tasks and the same vehicles as experiments/retime.py, with dispatch also
free (earliest completion, offline).

Retiming with the precedence fixed does not change full-path reservation (it already is the earliest solution of its own
precedence), and the rolling search under the same dispatch with the precedence on the resources free recovers almost nothing
(experiments/retime.py): the loss lies in dispatch and in the booking order. This module gives a feasible whole-network
schedule with dispatch free as well:
  the tasks are handled one by one in release order (the order in which full-path reservation dispatched them); each task
  goes to the vehicle, among M candidates, that completes it earliest -- the candidates are the M vehicles with the least
  "available time + empty run to the origin"; the completion time is found by trying the booking of full-path reservation on
  the reservation table of that moment (waiting only at the origin, no stops en route), and the chosen one is booked.
  The reservation table drops only the reservations that end before the earliest available time of all vehicles (all later
  bookings start no earlier).
Both destination bookings are computed (VARIANTS): clr = the booking of full-path reservation (the destination is held for the
6.4 s drive-off of the empty vehicle after unloading, as in the original schedule; the main reading); def1c = Definition 1(c)
(released at the end of unloading, as in the bounds). All tasks known, offline: not an online rule. Each schedule is checked
by the independent checker under its own booking; the readings are those of retime.py (lambda = N·16 h / completion, mid-run
rate, busy share of the binding resource and attribution of its idle time), compared with the same point (the same batch of
tasks) of retime.py.
Demand mapping: --map conserved takes the inter-area mapping with erection flow conserved (the main case; compute_conserved);
the default is the proportional inter-area mapping.

Run:   python experiments/offline_dispatch.py --out <dir> [--map conserved] [--procs N] [--slots DIR] [--aggregate-only]
           [--system FILE --plateau-okpo FILE --plateau-yantai FILE] [--conserved-report DIR]
Writes <dir>/config.json, progress.json, points/ and, when the released readings are given (as for retime.py, e.g.
--map conserved --conserved-report results/main_case), points.csv; the released one is results/retiming/offline_dispatch.csv
(main case). retime.py --dispatch takes either <dir> (points/) or that points.csv for its offline-dispatch columns.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import runner as F                        # noqa: E402
import retime as R                        # noqa: E402

M_CAND = 10                               # number of candidate vehicles tried for each task
# two destination bookings: clr = the booking of full-path reservation (destination held for the 6.4 s drive-off of the empty
# vehicle after unloading, as in the original schedule; the main reading); def1c = Definition 1(c) (released at the end of
# unloading, as in the bounds)
VARIANTS = (("clr", True), ("def1c", False))


def ect(D, m_cand=M_CAND, keep_clr=False):
    """Offline earliest-completion dispatch (see the module docstring). keep_clr: the destination keeps the 6.4 s drive-off of
    the empty vehicle after unloading (the booking of full-path reservation); otherwise it ends at the end of unloading
    (Definition 1(c)). Returns {vehicle: [(task, t1, t2)]}."""
    from occupancy.sim import Sim, Vehicle
    net, p, T, starts = D["net"], D["p"], D["T"], D["starts"]
    load, unload = p["load_s"], p["unload_s"]

    class _S(Sim):
        floor = 0.0

        def windows(self, v, task):
            *a, clr = super().windows(v, task)
            return (*a, clr if keep_clr else 0.0)

        def commit(self, vid, now, W):           # drops only the reservations that end before the earliest available time of all vehicles
            return super().commit(vid, self.floor, W)

    s = _S(net, dict(p, days=400), "reserve", len(starts), 1, None, tasks=[])
    ready, loc = [0.0] * len(starts), list(starts)
    seq = {}

    def probe(k, task):
        """Completion time of task by vehicle k under the booking of full-path reservation (not booked; the first half of
        Sim.plan)."""
        v = Vehicle(k, loc[k])
        so, sd, d_so, d_sd, TE, TL, WE, WL, ao, ad, tail_l, clr = s.windows(v, task)
        tmin = ready[k]
        while True:
            t1 = s.earliest(WE + [(so["res"], ao, TE + load + tail_l, d_so)], tmin, k, False, [])
            t2, tc, _, _ = s.book_l(sd, d_sd, TL, WL, ad, clr, t1 + TE + load, k, False, [])
            o2 = s.conflict(so["res"], t1 + ao, t2 + tail_l, d_so, k)
            if o2 is None:
                return tc + unload
            tmin = o2 - ao

    for j in sorted(range(len(T)), key=lambda j: T[j]["id"]):
        task = dict(T[j], id=j)
        o = net.stops[task["o"]]["node"]
        cand = sorted(range(len(starts)), key=lambda k: (ready[k] + net.dist(loc[k], o) / p["v_empty"], k))[:m_cand]
        k = min(cand, key=lambda k: (probe(k, task), k))
        s.floor = min(ready)
        s.plan(Vehicle(k, loc[k]), task, ready[k])
        seq.setdefault(k, []).append((j, task["plan"]["t1"], task["plan"]["t2"]))
        ready[k] = task["plan"]["t2"] + D["DL"][j]
        loc[k] = net.stops[task["d"]]["node"]
    return seq


def compute(arg):
    """One point (the keys of retime.py): offline dispatch on the same batch of tasks as full-path reservation, checked by the
    independent checker, writes points/<key>.json."""
    key, out = arg
    from exact import check as C
    from occupancy.bound import bounds
    yard, seed, k, days = key.split("|")
    t0 = time.time()
    D = R.run_fpr(yard, int(seed), int(k), float(days))
    T, p, net = D["T"], D["p"], D["net"]
    N = len(T)
    b = bounds(net, p, T, tight_only=True)
    rstar = tuple(b["binding_tight"][0]["res"])
    W0 = float(max(D["x0"][2 * j + 1] + D["DL"][j] for j in range(N)))
    res = dict(key=key, yard=yard, seed=int(seed), K=int(k), N=N, m_cand=M_CAND, binding=b["binding_tight"][0]["name"], lamN=b["network_tight"],
               W_fpr=W0, thr_fpr=R.thr(N, W0))
    inst = R.bare_instance(D)
    clr_s = (p["plat_len"] + p["long_clear"]) / p["v_empty"]
    for name, keep in VARIANTS:
        seq = ect(D, keep_clr=keep)
        x = np.zeros(2 * N)
        for kk, lst in seq.items():
            for j, a, b_ in lst:
                x[2 * j], x[2 * j + 1] = a, b_
        done = np.array([x[2 * j + 1] + D["DL"][j] for j in range(N)])
        W = float(done.max())
        c = C.check(net, p, T, D["starts"], seq, clr=keep)
        # busy share and idle attribution on the binding resource (as in retime.py): the same schedule and the same windows
        # (of this booking) -- its own precedence is fixed and the earliest solution computed, both readings are taken on it
        D2 = R.from_seq(inst, seq)
        mw, mres = R.make_win(R.model_windows(inst, clr_s if keep else 0.0), D2["seq"], D2["x0"], inst.cap)
        xr, st, edges, pe = R.retime(mw, D2["base"], D2["x0"], "all")
        W_ret = float(max(xr[2 * j + 1] + D2["DL"][j] for j in range(N)))
        r_id = mres.index(rstar)
        idle = R.idle_causes(mw, xr, pe, edges, r_id, W_ret)
        res[name] = dict(W=W, thr=R.thr(N, W), mid=R.mid_rate(done), check=c["ok"], check_errors=c["errors"][:5], max_ratio=c["max_ratio"],
                         busy=R.busy_frac(mw, xr, r_id, W_ret), W_retimed=W_ret, idle={kk: v / W_ret for kk, v in idle.items()},
                         retime_rounds=st["rounds"], vehicles_used=len(seq), seq={str(kk): v for kk, v in seq.items()})
    res["secs"] = time.time() - t0
    F.write_json(Path(out) / "points" / F.fname(key), res)
    return key, res["clr"]["thr"], res["thr_fpr"]


def compute_conserved(arg):
    """One point of the conserved mapping (main case): retime.run_fpr is replaced by run_fpr_conserved and the point is
    computed by compute (compute unchanged)."""
    R.run_fpr = R.run_fpr_conserved
    return compute(arg)


def aggregate(out, dmap=None, refs=None):
    """points.csv of the run in out; refs: released readings (see retime.committed)."""
    pts = [json.loads((out / "points" / F.fname(k)).read_text(encoding="utf-8")) for k in R.keys() if (out / "points" / F.fname(k)).exists()]
    if not pts:
        return
    sysv, plat = R.committed(dmap, refs)
    rows = []
    for r in pts:
        S = sysv[r["yard"]]
        row = dict(yard=r["yard"], seed=r["seed"], N=r["N"], fpr=r["thr_fpr"], lamN=r["lamN"], lamS=S, fpr_S=r["thr_fpr"] / S,
                   plat_res_S=plat[r["yard"]]["reserve"] / S, plat_ref_S=plat[r["yard"]]["ref"] / S, binding=r["binding"], secs=r["secs"])
        for v, _ in VARIANTS:
            x = r[v]
            row.update({f"{v}": x["thr"], f"{v}_mid": x["mid"], f"{v}_S": x["thr"] / S, f"{v}_mid_S": x["mid"] / S, f"{v}_N": x["thr"] / r["lamN"],
                        f"{v}_gain": x["thr"] / r["thr_fpr"] - 1, f"{v}_check": x["check"], f"{v}_busy": x["busy"],
                        f"{v}_retimed_gain": x["W"] / x["W_retimed"] - 1, f"{v}_idle": sum(x["idle"].values()),
                        **{f"{v}_idle_{c}": x["idle"].get(c, 0.0) for c in ("start", "vehicle", "chain", "self", "tail")}})
        row["def1c_vs_clr"] = row["def1c"] / row["clr"] - 1
        rows.append(row)
    import csv
    with open(out / "points.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def config(dmap=None):
    """Configuration of a run (config.json), with the hash of the functions the points depend on (without docstrings) and of
    occupancy/, bounds/, exact/. dmap = "conserved": the conserved mapping also hashes compute_conserved and run_fpr_conserved
    and records the code of its task stream (gen_sha256)."""
    cons = dmap == "conserved"
    h = hashlib.sha256()
    for d in ("occupancy", "bounds", "exact"):
        for f in sorted((ROOT / d).glob("*.py")):
            h.update(f"{d}/{f.name}".encode() + f.read_bytes())
    R.func_sha(__file__, ("ect", "compute") + (("compute_conserved",) if cons else ()), h)
    R.func_sha(ROOT / "experiments" / "retime.py", R.COMPUTE + ("from_seq",) + (("run_fpr_conserved",) if cons else ()), h)
    cfg = dict(title="whole-network offline dispatch (earliest completion)", yards=list(R.YARDS), seeds=list(R.SEEDS), K=R.K, days=R.DAYS,
               m_cand=M_CAND, variants=[v for v, _ in VARIANTS],
               networks={y: R.P.YARDS[y]["network"] for y in R.YARDS},
               code_sha256=h.hexdigest())
    if cons:
        cfg.update(title=cfg["title"] + ": inter-area mapping with erection flow conserved (main case)", demand_map="conserved",
                   gen_sha256=F.conserved_sha())
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--slots", default=None, help="CPU slot directory (default: environment variable CPU_SLOTS, else none)")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--map", choices=("current", "conserved"), default="current", help="demand mapping (as retime.py --map)")
    ap.add_argument("--system", help="system bounds, e.g. results/bounds/system_by_granularity.csv")
    ap.add_argument("--plateau-okpo", help="plateaus of Okpo rev 1833, e.g. results/saturation/okpo_readings.csv")
    ap.add_argument("--plateau-yantai", help="Yantai saturation summary, e.g. results/saturation/rules_summary.json")
    ap.add_argument("--conserved-report", help="main-case directory with bounds.csv and saturation.csv, e.g. results/main_case")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    dmap = "conserved" if a.map == "conserved" else None
    cfg, out = config(dmap), Path(a.out)
    (out / "points").mkdir(parents=True, exist_ok=True)
    if (out / "config.json").exists() and json.loads((out / "config.json").read_text(encoding="utf-8")) != json.loads(json.dumps(cfg)):
        sys.exit(f"{out} holds a run of another configuration")
    F.write_json(out / "config.json", cfg)
    ks = R.keys()
    todo = [k for k in ks if not (out / "points" / F.fname(k)).exists()]
    prog = F.Progress(out / "progress.json", len(ks), len(ks) - len(todo), 0, cfg["title"], "offline_dispatch")
    if not a.aggregate_only and todo:
        for key, thr, thr0 in F.slot_imap(compute_conserved if dmap else compute, [(k, str(out)) for k in todo], min(a.procs, len(todo)),
                                          slots_dir=a.slots, tag="offline_dispatch"):
            prog.log(f"{key}: full-path reservation {thr0:.0f}/day -> offline dispatch {thr:.0f}", done=prog.d["done"] + 1)
    if a.conserved_report if dmap else a.system and a.plateau_okpo and a.plateau_yantai:
        aggregate(out, dmap, dict(system=a.system, plateau=dict(yupu=a.plateau_okpo, yantai=a.plateau_yantai), conserved=a.conserved_report))
    else:
        print("points.csv needs the released readings (--system, --plateau-okpo, --plateau-yantai; --conserved-report with --map conserved)")
    prog.write(state="done" if all((out / "points" / F.fname(k)).exists() for k in ks) else "paused", processes=0)
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()

"""Holding ratios with matched methods. The holding ratio of the interventions (results/interventions/hold_ratio.csv) divides
the whole-network holding of offline dispatch (intervention_dispatch.py, 6.4-s drive-off) by the holding of the best schedule
of the sub-network (the exact search and the annealing with CP-SAT polish of intervention_subnet.py): the two sides use
methods of different strength. Here the sub-network points of the four scenarios where the holding ratio applies (the
instances of intervention_subnet.py as they are: N = 20, 30, 60, K = 8, seeds 101-103) are scheduled with the methods of the
whole network:
  fpr    full-path reservation (exact.model.run_fpr, the full-path reservation of subnet_exact.compute; destination including the
         6.4-s drive-off of the empty vehicle, checker and holding in the same definition, as intervention_dispatch.py);
  clr    offline dispatch (offline_dispatch.ect, as intervention_dispatch.py), 6.4-s drive-off (the main reading);
  def1c  offline dispatch, Definition 1(c) (destination released at the end of unloading).
Holding = exact.check.busy (as intervention_dispatch.py and intervention_subnet.py) for the top three binding elements of the
system bound of the scenario (intervention_subnet.watch, from the bounds table --bounds).
Also the computing time of one whole-network point in its steps (whole keys = the points of intervention_dispatch.py:
scenario x seed, K = 100, two days): full-path reservation and offline dispatch in both definitions once each; the
throughputs equal the dispatch points to the last digit (checked by hold_ratio_matched_report.py).

Run:   python experiments/hold_ratio_matched.py --out <dir> --bounds results/interventions/bound_points.csv [--procs 23]
                [--slots DIR] [--only-seed SEED]
--bounds: points.csv of experiments/intervention_bounds.py (released: results/interventions/bound_points.csv).
--only-seed: compute and summarise only the sub-network and whole-network points of this seed (101, 102 or 103).
--slots: machine-wide CPU slots as in experiments/runner.py (default: the environment variable CPU_SLOTS, off when neither is
given; then a plain process pool). A sub-network point takes about a second, a whole-network point 2-5 min. CP-SAT is not
used, but exact.model needs ortools to be importable.
Writes to <dir>: config.json, progress.json, points/ (one JSON per point with the schedules, a cache for resuming), points.csv
(sub-network; released as results/interventions/hold_ratio_matched_points.csv) and whole.csv (whole-network computing times;
released as results/interventions/hold_ratio_matched_whole.csv). The tables hold_ratio_matched.csv and computing_times.csv are
made from these by hold_ratio_matched_report.py.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import runner as F                        # noqa: E402
import intervention_dispatch as TE        # noqa: E402
import intervention_subnet as TS          # noqa: E402

SUBNETS = ("yupu_corridor_c", "yupu_corridor_c_w13", "yantai_paint_c", "yantai_paint_c_door1")   # where the holding ratio applies
VARIANTS = (("clr", True), ("def1c", False))


def code_sha():
    h = hashlib.sha256()
    for d in ("occupancy", "bounds", "exact"):
        for f in sorted((ROOT / d).glob("*.py")):
            h.update(f"{d}/{f.name}".encode() + f.read_bytes())
    for n in ("demand_mapping.py", "conserved_mapping.py", "intervention_net.py", "intervention_subnet.py", "intervention_dispatch.py",
              "offline_dispatch.py", "retime.py", "hold_ratio_matched.py"):
        h.update(f"experiments/{n}".encode() + (ROOT / "experiments" / n).read_bytes())
    return h.hexdigest()


def sub_keys(seed=None):
    return [k for k in TS.keys(SUBNETS) if seed is None or k.split("|")[1] == str(seed)]


def whole_keys(seed=None):
    return [f"whole|{y}|{s}|{sd}" for y, s in sorted({(TS.INSTANCES[n]["yard"], TS.INSTANCES[n]["scen"]) for n in SUBNETS}) for sd in TE.SEEDS
            if seed is None or sd == seed]


def sub_point(key):
    """One sub-network point: full-path reservation and offline dispatch in both definitions on the same instance, each with
    its checker, throughput and the holding of the binding elements."""
    import offline_dispatch as E
    from exact import check as C, model as X
    name, seed, n, k = key.split("|")
    t0 = time.time()
    inst = TS.instance(name, int(seed), int(n))
    net, p, T, starts = inst.net, inst.p, inst.tasks, inst.starts
    wl = TS.watch(name)
    res = [tuple(x[1]) for x in wl]
    out = dict(key=key, inst=name, yard=TS.INSTANCES[name]["yard"], scen=TS.INSTANCES[name]["scen"], seed=int(seed), N=int(n), K=int(k),
               watch=[x[0] for x in wl], watch_res=["|".join(r) for r in res], inst_secs=time.time() - t0)

    def read(seq, clr, secs):
        W = max(b + inst.dur_l(j) for v in seq.values() for j, a, b in v)
        c = C.check(net, p, T, starts, seq, clr=clr)
        hold = C.busy(net, p, T, starts, seq, res, clr=clr)
        return dict(W=W, thr=int(n) * TS.DAY / W, check=c["ok"], check_errors=c["errors"][:5], hold=[hold[r] for r in res],
                    vehicles=len(seq), secs=secs, seq={str(kk): v for kk, v in seq.items()})
    t1 = time.time()
    mk, seq, _ = X.run_fpr(inst, int(k))
    out["fpr"] = read(seq, True, time.time() - t1)
    assert abs(out["fpr"]["W"] - mk) < 1e-6, (key, out["fpr"]["W"], mk)          # completion as the full-path reservation of subnet_exact
    D = dict(net=net, p=p, T=T, starts=starts, DL=[inst.dur_l(j) for j in range(len(T))])
    for v, keep in VARIANTS:
        t1 = time.time()
        seq = E.ect(D, keep_clr=keep)
        out[v] = read(seq, keep, time.time() - t1)
    out["secs"] = time.time() - t0
    return out


def whole_point(key):
    """One whole-network point (the same point as intervention_dispatch.py): the computing time and throughput of two days of
    full-path reservation and of one offline dispatch in each definition (checker and holding not recomputed)."""
    import offline_dispatch as E
    import retime as R
    _, y, s, seed = key.split("|")
    t0 = time.time()
    D = TE.run_fpr(y, s, int(seed))
    n = len(D["T"])
    out = dict(key=key, yard=y, scen=s, seed=int(seed), N=n, fpr_secs=time.time() - t0,
               fpr_thr=R.thr(n, float(max(D["x0"][2 * j + 1] + D["DL"][j] for j in range(n)))))
    for v, keep in VARIANTS:
        t1 = time.time()
        seq = E.ect(D, keep_clr=keep)
        out[f"{v}_secs"] = time.time() - t1
        out[f"{v}_thr"] = R.thr(n, max(b + D["DL"][j] for lst in seq.values() for j, a, b in lst))
    out["secs"] = time.time() - t0
    return out


def compute(arg):
    key, out = arg
    r = whole_point(key) if key.startswith("whole|") else sub_point(key)
    F.write_json(Path(out) / "points" / F.fname(key), r)
    return key, r["secs"]


def summarize(out, seed=None):
    rows, whole = [], []
    for key in sub_keys(seed):
        x = json.loads((out / "points" / F.fname(key)).read_text(encoding="utf-8"))
        row = {c: x[c] for c in ("inst", "yard", "scen", "seed", "N", "K")}
        row.update({f"watch{i + 1}": x["watch"][i] for i in range(3)})
        for v in ("fpr", "clr", "def1c"):
            r = x[v]
            row.update({f"{v}_W": r["W"], f"{v}_thr": r["thr"], f"{v}_check": r["check"], f"{v}_vehicles": r["vehicles"], f"{v}_secs": r["secs"]})
            row.update({f"{v}_hold{i + 1}": r["hold"][i] for i in range(3)})
        row.update(inst_secs=x["inst_secs"], secs=x["secs"])
        rows.append(row)
    for key in whole_keys(seed):
        whole.append(json.loads((out / "points" / F.fname(key)).read_text(encoding="utf-8")))
    for name, rr in (("points.csv", rows), ("whole.csv", whole)):
        with open(out / name, "w", newline="", encoding="utf-8") as f:           # UTF-8 without BOM, LF: as the released tables
            w = csv.DictWriter(f, list(rr[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rr)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--bounds", required=True, help="points.csv of experiments/intervention_bounds.py "
                                                    "(released: results/interventions/bound_points.csv)")
    ap.add_argument("--procs", type=int, default=23)
    ap.add_argument("--slots", default=os.environ.get("CPU_SLOTS"), help="directory of machine-wide CPU slots (cpu_slots.Slots); "
                                                                         "default: environment variable CPU_SLOTS, else off")
    ap.add_argument("--only-seed", type=int, default=None, help="only the sub-network and whole-network points of this seed")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    os.environ[TE.BOUNDS] = str(Path(a.bounds).resolve())      # intervention_subnet.watch reads it, also in the child processes
    out = Path(a.out)
    ks = whole_keys() + sub_keys()                    # the whole-network points are long: started first
    cfg = dict(title="holding ratios with matched methods", keys=ks, subnets=list(SUBNETS), variants=[v for v, _ in VARIANTS],
               bounds=a.bounds, code_sha256=code_sha())
    if (out / "config.json").exists() and json.loads((out / "config.json").read_text(encoding="utf-8")) != json.loads(json.dumps(cfg, ensure_ascii=False)):
        sys.exit(f"{out} holds a run of another configuration")
    (out / "points").mkdir(parents=True, exist_ok=True)
    F.write_json(out / "config.json", cfg)
    print(f"run directory: {out}", flush=True)
    run = whole_keys(a.only_seed) + sub_keys(a.only_seed)
    todo = [k for k in run if not (out / "points" / F.fname(k)).exists()]
    prog = F.Progress(out / "progress.json", len(run), len(run) - len(todo), a.procs, cfg["title"], "hold_ratio_matched")
    if todo:
        for key, secs in F.slot_imap(compute, [(k, str(out)) for k in todo], min(a.procs, len(todo)), slots_dir=a.slots,
                                     tag="hold_ratio_matched"):
            prog.log(f"{key} ({secs:.0f} s)", done=prog.d["done"] + 1)
    summarize(out, a.only_seed)
    prog.write(state="done", processes=0)
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()

"""Sub-network instances of the main case under the interventions: the best known schedule that waits only at the origin, the
finite-window upper bounds and the holding of the binding elements (sub-network instances drawn from the task stream of the
main case).

Definition, size and seeds of the sub-networks as in the exact sub-network study (the ball sub-networks of
subnet_exact.INSTANCES: the roads within 400 m of both ends of the centre road, or within 200 m of the doors of a building;
tasks = the first N tasks of the task stream of the main case (conserved_mapping.stream, the same seed) whose origin and
destination are both in the sub-network and whose loaded shortest path lies in it (corridor: the path also passes road010-6;
building: origin or destination at that building), released at time 0; N = 20, 30, 60, K = 8, seeds 101-103), on the network
with the road changes of the scenario (intervention_net). Only the task stream differs: exact.model.sample_tasks draws from
TaskStream (the proportional mapping); sample_tasks here is written the same way with conserved_mapping.stream (exact/ is
unchanged).
  yupu_corridor_c        through corridor road010-6, current (how tight the bottleneck of the main case is; the same
                         sub-network under the proportional mapping is the instance yupu_corridor of subnet_exact.py / corridor.py)
  yupu_corridor_c_w13    the same instances, corridor widened to 13 m
  yupu_r0994_c_w14       with the corridor widened to 14 m the binding element of the system bound is road099-4 (12 m, 463 m):
                         the roads within 400 m of its ends, tasks passing it; it has no comparable instance (all of
                         them are long through flows, experiments/intervention_diag.py), so it is defined but not scheduled
  yantai_paint_c         Yantai building014 doors, current (binding in the system bound: entrance001)
  yantai_paint_c_door1   the same instances, entrance001 for passing only (2 in-door bays); the binding element moves to the
                         other door (building014-entrance1)
  yantai_b007_c_door2    the binding element of the system bound with both doors relaxed: the doors of building007 (entrance002
                         and two more doors tied at 1.000), the roads within 200 m, tasks with origin or destination at building007
Each point:
  1. the steps of the exact sub-network study (subnet_exact.compute called as it is, with the instance of this module): full-path
     reservation, the finite-window linear programme, beta, lambda_W^N, the occupancy-order lower bound (chain), CP-SAT with the
     dispatch of full-path reservation fixed and the optimisation with it as a hint (15 s per task), the independent checker;
  2. as in the corridor study (corridor.py): simulated annealing from the best feasible schedule of step 1 (serial booking,
     corridor.anneal) and the same-dispatch CP-SAT polish, two rounds of 15 min (the second in 3 segments with reheating);
  3. on the bound side also subnet_bounds.chain2 and subnet_gap.chain3 (binding resource = the binding element of lambda_W^N in
     the window of the best schedule); the best upper bound is the smallest of the bounds;
  4. the best schedule: independent checker (destination until the end of unloading, definition 1(c)); holding = the share of the
     completion time during which a resource is held (exact.check.busy, the same function as for the whole-network holding of
     intervention_dispatch.py; also the split of subnet_gap.gap_split): for the top three binding elements of the system bound
     of the scenario (the points.csv of intervention_bounds.py).

Run:   python experiments/intervention_subnet.py --out <dir> --bounds <points.csv of intervention_bounds.py> [--only INSTANCE ...]
           [--max-minutes M] [--max-threads N] [--slots DIR] [--aggregate-only]
       with the released table: --bounds results/interventions/bound_points.csv
Each point runs in its own child process (subnet_exact.schedule; 1-4 CP-SAT threads per point, the annealing is single
threaded; --slots: the CPU slots it takes, default the environment variable CPU_SLOTS, off when neither is given). The run of the
paper used experiments/intervention_subnet_run.py (one CPU slot per point, CP-SAT threads leased step by step). Writes to <dir>:
config.json, progress.json, points/ (one JSON per point with the best schedule, a cache for resuming) and points.csv, released as
results/interventions/subnet_points.csv (the scenario table results/interventions/subnet.csv is drawn from it, and with the
whole-network holding of intervention_dispatch.py the ratios of results/interventions/hold_ratio.csv).
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
import subnet_exact as SUB                # noqa: E402
import intervention_net as IV             # noqa: E402

INSTANCES = {
    "yupu_corridor_c": dict(yard="yupu", scen="base", ball=("road010-6", 400.0), via=["road010-6"]),
    "yupu_corridor_c_w13": dict(yard="yupu", scen="w13", ball=("road010-6", 400.0), via=["road010-6"]),
    "yupu_r0994_c_w14": dict(yard="yupu", scen="w14", ball=("road099-4", 400.0), via=["road099-4"]),
    "yantai_paint_c": dict(yard="yantai", scen="base", ball=("building014", 200.0), ents=["building014"]),
    "yantai_paint_c_door1": dict(yard="yantai", scen="door1", ball=("building014", 200.0), ents=["building014"]),
    "yantai_b007_c_door2": dict(yard="yantai", scen="door2", ball=("building007", 200.0), ents=["building007"]),
}
SEEDS, NS, K = SUB.SEEDS, SUB.NS, 8
SECS = 15.0                               # CP-SAT seconds per task (as secs_per_task of the exact sub-network study)
ROUNDS = ((15.0, 1), (15.0, 3))           # two rounds of annealing (minutes, segments), as in the corridor study
DAY = SUB.DAY
_NET, _CACHE = {}, {}


def code_sha():
    h = hashlib.sha256()
    for d in ("occupancy", "bounds", "exact"):
        for f in sorted((ROOT / d).glob("*.py")):
            h.update(f"{d}/{f.name}".encode() + f.read_bytes())
    for n in ("demand_mapping.py", "conserved_mapping.py", "intervention_net.py", "intervention_subnet.py", "subnet_exact.py",
              "subnet_bounds.py", "corridor.py", "subnet_gap.py", "retime.py"):
        h.update(f"experiments/{n}".encode() + (ROOT / "experiments" / n).read_bytes())
    return h.hexdigest()


# no comparable instance (intervention_diag: the tasks passing road099-4 are all long through flows): defined, not scheduled
SKIP = ("yupu_r0994_c_w14",)


def keys(only=None):
    return [f"{n}|{s}|{N}|{K}" for n in INSTANCES if n not in SKIP and (not only or n in only) for N in NS for s in SEEDS]


def sample_tasks(net, p, seed, roads, n, ents=None, via=None, max_draws=10 ** 7):
    """As exact.model.sample_tasks, with the task stream of the conserved mapping, conserved_mapping.stream (the same seed)."""
    import conserved_mapping
    from exact import model as X
    stops, ts, out = X.sub_stops(net, roads), conserved_mapping.stream(net, seed, p), []
    for _ in range(max_draws):
        t = ts.next(0.0)
        if ents and not {net.stops[t["o"]]["ent"], net.stops[t["d"]]["ent"]} & set(ents):
            continue
        if t["o"] not in stops or t["d"] not in stops:
            continue
        pr = X.path_roads(net, net.stops[t["o"]]["node"], net.stops[t["d"]]["node"])
        if via and not pr & set(via):
            continue
        if not t.get("lift") and pr <= roads:
            out.append(dict(t, id=len(out), src_id=t["id"], release=0.0))
            if len(out) == n:
                return out
    raise RuntimeError(f"fewer than {n} tasks in the sub-network ({len(out)})")


def instance(name, seed, N):
    """As subnet_exact.instance: the first N tasks (the smaller N are prefixes of the larger), start positions drawn at the
    largest N; network and parameters of the scenario (conserved mapping + road changes)."""
    from exact import model as X
    s = INSTANCES[name]
    if name not in _NET:
        p = IV.params(s["yard"], s["scen"])
        _NET[name] = IV.net(s["yard"], p), p
    net, p = _NET[name]
    if (name, seed) not in _CACHE:
        ent, R = s["ball"]
        seeds = {net.stops[x]["node"] for e, v in net.ents.items() if v["name"].startswith(ent) for x in v["stops"]} | \
            {x for r in net.road.values() if r["name"] == ent for x in (r["a"], r["b"])}
        roads = X.ball_roads(net, seeds, R)
        ents = [e for e, v in net.ents.items() if v["name"].startswith(tuple(s.get("ents", ())))] if s.get("ents") else None
        via = [rid for rid, r in net.road.items() if r["name"] in s.get("via", ())] or None
        tasks = sample_tasks(net, p, seed, roads, max(NS), ents, via)
        _CACHE[(name, seed)] = (tasks, X.vehicle_starts(net, tasks, seed, K), roads)
    tasks, starts, roads = _CACHE[(name, seed)]
    return X.Instance(f"{name}|{seed}|{N}", net, p, tasks[:N], starts, roads)


# subnet_exact.compute takes the instance by its global name: on import of this module (child processes included) it becomes
# the main-case instance
SUB.instance = instance


def watch(name):
    """The top three binding elements of the system bound of the scenario [(name, resource tuple)] (the points.csv of
    intervention_bounds.py, --bounds)."""
    import intervention_dispatch as E
    s = INSTANCES[name]
    return E.watch(s["yard"], s["scen"])


def polish(inst, k, seq, secs, w):
    """Same-dispatch offline rebooking (the polish of corridor.run_point): dispatch and the order of each vehicle fixed, CP-SAT
    minimises the completion, then a left shift in continuous time. Returns (completion, schedule) or None."""
    import retime as R
    D = R.from_seq(inst, seq)
    mw, _ = R.make_win(R.model_windows(inst), D["seq"], D["x0"], inst.cap)
    xs, _ = R.int_start(D, mw)
    r = R.same_dispatch(mw, D["base"], D["DL"], xs, max(10.0, secs), w)
    if "x" not in r:
        return None
    x2, *_ = R.retime(mw, D["base"], r["x"], "all")
    n = len(inst.tasks)
    return max(x2[2 * j + 1] + D["DL"][j] for j in range(n)), {kk: [(j, float(x2[2 * j]), float(x2[2 * j + 1])) for j, _, _ in v] for kk, v in D["seq"].items()}


def run_point(key, minutes, w, out):
    """Child process: one point (minutes is not used, the time limits are SECS and ROUNDS; w CP-SAT threads), writes
    points/<key>.json."""
    out = Path(out)
    try:
        import subnet_bounds as B2
        import subnet_gap as B3
        import corridor as Co
        from exact import check as C, model as X
        name, seed, n, k = key.split("|")
        seed, n, k = int(seed), int(n), int(k)
        t0 = time.time()
        rsub = SUB.compute(key, SECS, w)
        inst = instance(name, seed, n)
        net, p = inst.net, inst.p
        cand = [(rsub["opt"]["W"], rsub["opt"]["seq"], "subnet_opt")] if rsub["opt"]["check"]["ok"] else []
        if rsub["fix"]["check"]["ok"]:
            cand.append((rsub["fix"]["W"], rsub["fix"]["seq"], "subnet_fix"))
        cand.append((rsub["fpr"]["W"], rsub["fpr"]["seq"], "fpr"))
        W, seq, src = min(cand, key=lambda c: c[0])
        seq = {int(kk): [tuple(y) for y in v] for kk, v in seq.items()}
        start = dict(W=W, src=src)
        rounds = []
        for i, (mins, rst) in enumerate(ROUNDS):
            t1 = time.time()
            pol = min(120.0, 0.1 * 60 * mins)
            sa, W_sa, ev, W_book0 = Co.anneal(Co.Booker(inst, k), seq, mins - pol / 60, seed * 1000 + n + 7 * i, rst)
            got = [(W_sa, sa, "anneal")]
            pr = polish(inst, k, sa, t1 + 60 * mins - time.time(), w)
            if pr:
                got.append((pr[0], pr[1], "polish"))
            Wr, sr, srcr = min(got, key=lambda c: c[0])
            ok = C.check(net, p, inst.tasks, inst.starts, sr)["ok"]
            rounds.append(dict(W_sa=W_sa, W_polish=pr[0] if pr else None, evals=ev, W_book0=W_book0, check=ok, secs=time.time() - t1))
            if ok and Wr < W - 1e-6:
                W, seq, src = Wr, {int(kk): [tuple(y) for y in v] for kk, v in sr.items()}, f"r{i + 1}_{srcr}"
        c = C.check(net, p, inst.tasks, inst.starts, seq)
        lam, bind = X.lam_window(inst, W)
        bres = tuple(bind[0]["res"])
        c2, c3 = B2.chain2(inst, k, bres), B3.chain3(inst, k, bres)
        Ws = dict(lp=rsub["lp"]["W"], chain=rsub["chain"]["W"], chain2=c2["W"] if c2 else None, chain3=c3["W"] if c3 else None)
        ub = min([rsub["opt"]["thr_ub"]] + [n * DAY / v for v in Ws.values() if v])
        wl = watch(name)
        wres = [tuple(x[1]) for x in wl] + [tuple(map(str, bres))]
        hold = C.busy(net, p, inst.tasks, inst.starts, seq, wres)
        split = B3.gap_split(inst, seq, wres[0])
        F.write_json(out / "points" / F.fname(key), dict(
            key=key, inst=name, yard=INSTANCES[name]["yard"], scen=INSTANCES[name]["scen"], seed=seed, N=n, K=k, threads=w,
            flows=rsub["flows"], fpr=dict(W=rsub["fpr"]["W"], thr=rsub["fpr"]["thr"], check=rsub["fpr"]["check"]["ok"]),
            subnet=dict(opt_W=rsub["opt"]["W"], opt_status=rsub["opt"]["status"], opt_thr_ub=rsub["opt"]["thr_ub"],
                        fix_W=rsub["fix"]["W"], lamN=rsub["lamN"], binding=rsub["binding"], beta=rsub["beta"]),
            start=start, rounds=rounds, W=W, thr=n * DAY / W, src=src, status=src, check=c["ok"], check_errors=c["errors"][:5], lamN=lam, binding=bind[0]["name"],
            bounds_W=Ws, thr_ub=ub, gap=1 - (n * DAY / W) / ub, watch=[x[0] for x in wl] + [bind[0]["name"]], watch_res=["|".join(r) for r in wres],
            hold={"|".join(r): hold[r] for r in wres}, split=split, secs=time.time() - t0, seq={str(kk): v for kk, v in seq.items()}))
    except Exception:
        import traceback
        (out / "failures.log").open("a", encoding="utf-8").write(f"{key}\n{traceback.format_exc()}\n")


def summarize(out, ks):
    rows = []
    for key in ks:
        f = out / "points" / F.fname(key)
        if not f.exists():
            continue
        x = json.loads(f.read_text(encoding="utf-8"))
        rows.append(dict(inst=x["inst"], yard=x["yard"], scen=x["scen"], seed=x["seed"], N=x["N"], K=x["K"], threads=x["threads"],
                         fpr_thr=x["fpr"]["thr"], start_src=x["start"]["src"], start_thr=x["N"] * DAY / x["start"]["W"], thr=x["thr"], src=x["src"],
                         check=x["check"], lamN=x["lamN"], binding=x["binding"], thr_ub=x["thr_ub"], gap=x["gap"],
                         subnet_opt_thr=x["N"] * DAY / x["subnet"]["opt_W"], subnet_status=x["subnet"]["opt_status"],
                         subnet_thr_ub=x["subnet"]["opt_thr_ub"],
                         **{f"ub_{k}": (x["N"] * DAY / v if v else None) for k, v in x["bounds_W"].items()},
                         **{f"watch{i + 1}": x["watch"][i] for i in range(4)}, **{f"watch{i + 1}_res": x["watch_res"][i] for i in range(4)},
                         **{f"hold{i + 1}": x["hold"][x["watch_res"][i]] for i in range(4)},
                         split_busy=x["split"]["busy"] / x["split"]["W"], split_forced=x["split"]["forced"] / x["split"]["W"],
                         split_vehicle=x["split"]["vehicle"] / x["split"]["W"], split_other=x["split"]["other"] / x["split"]["W"],
                         split_start=x["split"]["start"] / x["split"]["W"],
                         evals=sum(r["evals"] for r in x["rounds"]), secs=x["secs"]))
    if rows:
        with open(out / "points.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return rows


def main():
    import intervention_dispatch as E
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--bounds", required=True, help="points.csv of experiments/intervention_bounds.py "
                                                    "(released: results/interventions/bound_points.csv)")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--max-threads", type=int, default=4)
    ap.add_argument("--slots", default=None, help="CPU slot directory (default: environment variable CPU_SLOTS, else none)")
    ap.add_argument("--aggregate-only", action="store_true")
    a = ap.parse_args()
    for n in sorted(set(a.only or INSTANCES) & set(SKIP)):
        print(f"{n}: no comparable instance (experiments/intervention_diag.py), not scheduled", flush=True)
    os.environ[E.BOUNDS] = str(Path(a.bounds).resolve())      # watch() reads it, also in the child processes
    out = Path(a.out)
    cfg = dict(title="best schedules, upper bounds and holding on the sub-networks of the main case under the interventions",
               instances=INSTANCES, seeds=list(SEEDS), ns=list(NS), K=K, secs_per_task=SECS, rounds=[list(r) for r in ROUNDS],
               bounds=a.bounds, code_sha256=code_sha())
    if (out / "config.json").exists() and json.loads((out / "config.json").read_text(encoding="utf-8")) != json.loads(json.dumps(cfg)):
        sys.exit(f"{out} holds a run of another configuration")
    (out / "points").mkdir(parents=True, exist_ok=True)
    F.write_json(out / "config.json", cfg)
    print(f"run directory: {out}", flush=True)
    ks = keys()
    pf = lambda k: out / "points" / F.fname(k)
    prog = F.Progress(out / "progress.json", len(ks), sum(pf(k).exists() for k in ks), 0, cfg["title"], "intervention_subnet")
    if not a.aggregate_only:
        SUB.OUT, SUB.MIN_W, SUB.MAX_W = out, 1, a.max_threads
        deadline = time.time() + 60 * a.max_minutes if a.max_minutes else None
        todo = sorted((k for k in keys(a.only) if not pf(k).exists()), key=lambda k: -int(k.split("|")[2]))
        if SUB.schedule(todo, 0.0, pf, prog, deadline, a.slots, target=run_point):
            summarize(out, ks)
            prog.write(state="paused", note="stopped at --max-minutes; the same command resumes")
            sys.exit(3)
    rows = summarize(out, ks)
    prog.write(state="done" if all(pf(k).exists() for k in ks) else "paused", processes=0, note=f"{len(rows)} points")
    print(f"-> {out} ({len(rows)} points)", flush=True)


if __name__ == "__main__":
    main()

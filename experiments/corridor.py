"""Okpo through-corridor sub-network: feasible schedules (strongly feasible) close to the bound.

The instances are those of experiments/subnet_exact.py (instance("yupu_corridor", seed, N): the roads within 400 m of both
ends of road010-6 (12 m, 168 m), the tasks whose loaded route passes it, K = 8, N = 20, 30, 60, seeds 101-103), and so is the
definition of a schedule: waiting only at the origin, no stops on the empty and loaded runs, the destination held only until
the end of unloading (Definition 1(c)). The start is the best known solution of the sub-network run (the better of its first
and its extended solve, checked by the independent checker). Two steps per point:
  1. simulated annealing (anneal): a solution = the route of each vehicle + a booking order; serial booking (Booker) turns a
     solution into a schedule -- in booking order each task is booked in the reservation table in the earliest slot of "the
     whole path, waiting only at the origin, no earlier than its vehicle finishes its previous task" (the booking of
     Sim.plan, destination until the end of unloading). Neighbourhood: move a task to some position of some vehicle's
     route, swap two tasks, change the booking key of a task; objective = completion + 1e-3 x sum of completion times.
  2. same-dispatch offline rebooking (polish): the dispatch and vehicle orders of the best annealing solution fixed, the
     precedence on the resources free, CP-SAT minimises the completion (experiments/retime.same_dispatch, rounded to
     0.01 s), followed by a left shift in continuous time.
The shortest completion of the sub-network solution, annealing and polish is taken; the best solution is checked by the
independent checker (exact.check.check, destination until the end of unloading).
On the bound side the upper bound of the optimum of the sub-network run is taken (the tightest of the CP-SAT bound, the
finite-window linear programme and the sharpened occupancy-order bound), and the generalised occupancy-order bound and the
vehicle integrality bound of experiments/subnet_gap.py are recorded. The idle time of the best solution on road010-6 is split
by experiments/subnet_gap.gap_split (gaps the pairwise precedence cannot avoid / the rest).
A large-neighbourhood search with the CP-SAT model of exact.model as subsolver (freeing the dispatch of 6-30 tasks, the
predecessors of the others kept) found no better solution within 40 s per round on one thread, hence the method above.

Run:   python experiments/corridor.py --subnet <run of subnet_exact.py> --out <dir> [--gap <run of subnet_gap.py>]
           [--minutes M] [--slots DIR] [--max-minutes M] [--aggregate-only] [--warm DIR] [--restarts N] [--resplit DIR]
--subnet is the --out directory of a run of subnet_exact.py (and subnet_bounds.py) made first, read for the starting solutions
and bounds (points/ with the schedules, ext/, bounds2.json; these point files are not released, the released table of the
paper's run is results/subnet/points.csv, configuration results/subnet/config.json); --gap for the bounds of subnet_gap.py in points.csv (NaN without it). The released tables of
this module are results/subnet/corridor.csv (points.csv of a run) and results/subnet/corridor_split.csv (points.csv of
--resplit). Each point runs in its own child process on one thread (experiments/subnet_exact.schedule).
Writes <dir>/config.json, progress.json, points/ (one JSON per point: best solution and readings) and points.csv. --warm:
an earlier run of this module, each point starts from the better of its solution and the sub-network one. --resplit: an
earlier run of this module, whose idle split is recomputed with the current gap_split into points.csv of --out.
"""
import argparse
import csv
import hashlib
import heapq
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import runner as F                        # noqa: E402
import subnet_exact as SUB                # noqa: E402

INST = "yupu_corridor"
SEEDS, NS, K = (101, 102, 103), (20, 30, 60), 8
SUBNET_DIR = None                         # output directory of experiments/subnet_exact.py (--subnet; the child processes read it from config.json)
DAY = SUB.DAY
POLISH = 0.1                              # share of the time limit of a point for the polish (same-dispatch offline rebooking), at most 120 s


def keys():
    return [f"{INST}|{s}|{N}|{K}" for N in NS for s in SEEDS]


def subnet_best(key):
    """The point file of the sub-network run: the best known solution (the better of the first and the extended solve that
    passes the checker), the lower bounds (finite-window linear programme, occupancy-order bounds), the upper bound of the
    optimum and lambda_W^N."""
    r = json.loads((SUBNET_DIR / "points" / F.fname(key)).read_text(encoding="utf-8"))
    seq, W, src = r["opt"]["seq"], r["opt"]["W"], "subnet_opt"
    ub = r["opt"]["thr_ub"]
    ef = SUBNET_DIR / "ext" / F.fname(key)
    if ef.exists():
        x = json.loads(ef.read_text(encoding="utf-8"))
        if x["check"] and x["W"] and x["W"] < W:
            seq, W, src = x["seq"], x["W"], "subnet_ext"
        ub = min(ub, x["thr_ub"])
    c2 = (json.loads((SUBNET_DIR / "bounds2.json").read_text(encoding="utf-8"))["points"].get(key) or {}).get("chain2")
    lb = max(v for v in (r["lp"]["W"], r["chain"]["W"], c2, 0.0) if v)
    ub = min([ub] + [r["N"] * DAY / v for v in (r["chain"]["W"], c2) if v])
    return dict(seq={int(k): [tuple(y) for y in v] for k, v in seq.items()}, W=W, src=src, lb=lb, thr_ub=ub, lamN=r["lamN"],
                fpr=r["fpr"]["thr"], N=r["N"], W_lp=r["lp"]["W"])


class Booker:
    """Serial booking: given the route of each vehicle and the booking order, books each task one after another in the
    reservation table in the earliest slot of "the whole path, waiting only at the origin, no earlier than its vehicle
    finishes its previous task" (the booking and reservation table of Sim.plan, just-in-time departure as there), the
    destination held only until the end of unloading (Definition 1(c), as in the model of exact.model). One Booker is reused,
    the reservation table is cleared each time."""

    def __init__(self, inst, K):
        from occupancy.sim import Sim

        class _S(Sim):
            def windows(self, v, task):
                *a, clr = super().windows(v, task)
                return (*a, 0.0)

            def commit(self, vid, now, W):          # the booking order is not in time order, so finished reservations cannot be dropped by "now"
                return super().commit(vid, -math.inf, W)
        self.inst, self.K = inst, K
        self.sim = _S(inst.net, dict(inst.p, days=400), "reserve", K, 1, None, tasks=[])
        self.dl = [inst.dur_l(j) for j in range(len(inst.tasks))]

    def run(self, routes, prio):
        """routes {vehicle: [task ...]}; prio {task: key} (lower keys are booked first, on one vehicle in route order). Returns
        (completion, sum of completion times, {vehicle: [(task, t1, t2)]})."""
        import collections as co
        from occupancy.sim import Vehicle
        s, inst = self.sim, self.inst
        s.table, s.ev = co.defaultdict(list), []
        s.occ, s.ivs, s.wait_by = co.defaultdict(float), co.defaultdict(list), co.defaultdict(float)
        out, heap, nxt = {}, [], {k: 0 for k in routes}
        ready, loc = {k: 0.0 for k in routes}, {k: inst.starts[k] for k in routes}
        for k, r in routes.items():
            if r:
                heapq.heappush(heap, (prio[r[0]], k))
        W = S = 0.0
        while heap:
            _, k = heapq.heappop(heap)
            j = routes[k][nxt[k]]
            task = dict(inst.tasks[j], id=j)
            s.plan(Vehicle(k, loc[k]), task, ready[k])
            out.setdefault(k, []).append((j, task["plan"]["t1"], task["plan"]["t2"]))
            ready[k] = task["plan"]["t2"] + self.dl[j]
            loc[k] = inst.net.stops[inst.tasks[j]["d"]]["node"]
            W, S = max(W, ready[k]), S + ready[k]
            nxt[k] += 1
            if nxt[k] < len(routes[k]):
                heapq.heappush(heap, (prio[routes[k][nxt[k]]], k))
        return W, S, out


def anneal(booker, seq, minutes, seed, restarts=1):
    """Simulated annealing: a solution = the route of each vehicle + booking keys; neighbourhood = move a task to some position
    of some vehicle's route, swap two tasks (also across vehicles), change the booking key of a task; objective = completion
    + 1e-3 x sum of completion times; the temperature falls geometrically from 0.5 % to 0.005 % of the completion. Starts from
    seq (routes and the order by t1).
    restarts > 1: the time limit is split into legs, each reheating from the current best solution (a few fast-cooling legs did
    better than one slow leg in trial runs).
    Returns (best {vehicle: [(task, t1, t2)]}, completion, number of evaluations, completion of the start under serial
    booking)."""
    rng = random.Random(seed)
    routes = {k: [j for j, _, _ in v] for k, v in seq.items()}
    for k in range(booker.K):
        routes.setdefault(k, [])
    prio = {j: a for v in seq.values() for j, a, _ in v}
    f = lambda W, S: W + 1e-3 * S
    W, S, out = booker.run(routes, prio)
    best, W0 = [W, out, routes, prio], W
    n, jobs = 0, list(prio)
    for r in range(restarts):
        routes, prio = {k: list(v) for k, v in best[2].items()}, dict(best[3])
        W, S, _ = booker.run(routes, prio)
        cur = f(W, S)
        T0, T1 = 5e-3 * W, 5e-5 * W
        t_start = time.time()
        t_end = t_start + 60 * minutes / restarts
        n += _anneal_leg(booker, rng, routes, prio, cur, best, f, T0, T1, t_start, t_end, jobs, W0)
    return best[1], best[0], n, W0


def _anneal_leg(booker, rng, routes, prio, cur, best, f, T0, T1, t_start, t_end, jobs, W0):
    """One leg of anneal (updates best = [completion, schedule, routes, booking keys] in place). Returns the number of
    evaluations."""
    n = 0
    while time.time() < t_end:
        temp = T0 * (T1 / T0) ** ((time.time() - t_start) / (t_end - t_start))
        r2, p2 = {k: list(v) for k, v in routes.items()}, dict(prio)
        mv = rng.random()
        if mv < 0.4:                                   # move
            j = rng.choice(jobs)
            r2[next(k for k, v in r2.items() if j in v)].remove(j)
            k1 = rng.randrange(booker.K)
            r2[k1].insert(rng.randint(0, len(r2[k1])), j)
        elif mv < 0.7:                                 # swap
            a, b = rng.sample(jobs, 2)
            ka, kb = (next(k for k, v in r2.items() if x in v) for x in (a, b))
            ia, ib = r2[ka].index(a), r2[kb].index(b)
            r2[ka][ia], r2[kb][ib] = b, a
        else:                                          # change the booking key
            j = rng.choice(jobs)
            p2[j] += rng.gauss(0, 0.05 * W0)
        W2, S2, out2 = booker.run(r2, p2)
        n += 1
        c2 = f(W2, S2)
        if c2 <= cur or rng.random() < math.exp((cur - c2) / temp):
            routes, prio, cur = r2, p2, c2
            if W2 < best[0] - 1e-6:
                best[:] = [W2, out2, routes, prio]
    return n


def run_point(key, minutes, w, out):
    """Child process: one point (time limit minutes minutes, w threads), writes points/<key>.json."""
    global SUBNET_DIR
    out = Path(out)
    try:
        import subnet_gap as B
        import retime as R
        from exact import check as C
        name, seed, N, k = key.split("|")
        seed, N, k = int(seed), int(N), int(k)
        inst = SUB.instance(name, seed, N)
        cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
        SUBNET_DIR = Path(cfg["start_from"])
        b = subnet_best(key)
        warm = (b["W"], b["seq"], b["src"])                # start: the best sub-network solution, or the better solution of an earlier run (warm_from)
        wf = Path(cfg["warm_from"]) / "points" / F.fname(key) if cfg.get("warm_from") else None
        if wf and wf.exists():
            x = json.loads(wf.read_text(encoding="utf-8"))
            if x["check"] and x["W"] < warm[0] - 1e-6:
                warm = (x["W"], {int(kk): [tuple(y) for y in v] for kk, v in x["seq"].items()}, "previous run")
        t0 = time.time()
        pol = min(120.0, POLISH * 60 * minutes)
        sa, W_sa, n, W_book0 = anneal(Booker(inst, k), warm[1], minutes - pol / 60, seed * 1000 + N, cfg.get("restarts", 1))
        cand = [(b["W"], b["seq"], b["src"]), warm, (W_sa, sa, "anneal")]
        D = R.from_seq(inst, sa)                       # polish: the dispatch and vehicle orders of the best annealing solution fixed
        mw, _ = R.make_win(R.model_windows(inst), D["seq"], D["x0"], inst.cap)
        xs, how = R.int_start(D, mw)
        r = R.same_dispatch(mw, D["base"], D["DL"], xs, max(10.0, t0 + 60 * minutes - time.time()), w)
        if "x" in r:
            x2, *_ = R.retime(mw, D["base"], r["x"], "all")
            cand.append((max(x2[2 * j + 1] + D["DL"][j] for j in range(N)),
                         {kk: [(j, float(x2[2 * j]), float(x2[2 * j + 1])) for j, _, _ in v] for kk, v in D["seq"].items()}, "polish"))
        W, seq, src = min(cand, key=lambda c: c[0])
        c = C.check(inst.net, inst.p, inst.tasks, inst.starts, seq)
        res = ("r", next(r_ for r_, v in inst.net.road.items() if v["name"] == "road010-6"))
        split = {nm: B.gap_split(inst, s_, res) for nm, s_ in (("start", b["seq"]), ("best", seq))}
        F.write_json(out / "points" / F.fname(key), dict(
            key=key, seed=seed, N=N, K=k, threads=w, minutes=minutes, evals=n, src=src, W_book0=W_book0, W_sa=W_sa,
            warm=dict(src=warm[2], W=warm[0]), W_polish=cand[3][0] if len(cand) > 3 else None, polish_status=r["status"],
            start=dict(src=b["src"], W=b["W"], thr=N * DAY / b["W"]), W=W, thr=N * DAY / W, thr_ub=b["thr_ub"],
            lamN=b["lamN"], fpr=b["fpr"], lb=b["lb"], W_lp=b["W_lp"], split=split, check=c["ok"], check_errors=c["errors"][:5],
            status=src, secs=time.time() - t0, seq={str(kk): v for kk, v in seq.items()}))
    except Exception:
        import traceback
        (out / "failures.log").open("a", encoding="utf-8").write(f"{key}\n{traceback.format_exc()}\n")


def aggregate(cfg, out, src=None, gap_dir=None):
    """Aggregation -> points.csv in out. The point files are read from src (default out); the idle split is recomputed with
    the current subnet_gap.gap_split from the best solution of each point file and the best sub-network solution (not taken
    from the split stored in the point file). gap_dir: output directory of experiments/subnet_gap.py (bounds3.json:
    generalised occupancy-order bound and vehicle integrality bound); its columns are NaN without it."""
    import subnet_gap as B
    pts = [json.loads(((src or out) / "points" / F.fname(k)).read_text(encoding="utf-8")) for k in keys()
           if ((src or out) / "points" / F.fname(k)).exists()]
    if not pts:
        return
    for r in pts:
        inst = SUB.instance(INST, r["seed"], r["N"])
        res = ("r", next(r_ for r_, v in inst.net.road.items() if v["name"] == "road010-6"))
        r["split"] = {nm: B.gap_split(inst, s_, res) for nm, s_ in
                      (("start", subnet_best(r["key"])["seq"]), ("best", {int(k): [tuple(y) for y in v] for k, v in r["seq"].items()}))}
    f = Path(gap_dir) / "bounds3.json" if gap_dir else None
    b3 = json.loads(f.read_text(encoding="utf-8"))["points"] if f and f.exists() else {}
    rows = []
    for r in pts:
        ln = r["lamN"]
        x3 = b3.get(r["key"], {})
        row = dict(seed=r["seed"], N=r["N"], K=r["K"], fpr_N=r["fpr"] / ln, start_N=r["start"]["thr"] / ln, best_N=r["thr"] / ln,
                   sa_N=r["N"] * DAY / r["W_sa"] / ln, book0_N=r["N"] * DAY / r["W_book0"] / ln, ub_N=r["thr_ub"] / ln,
                   chain3_N=(x3.get("thr3") or float("nan")) / ln, veh_N=(x3.get("thr_veh") or float("nan")) / ln,
                   gap=1 - r["thr"] / r["thr_ub"], start_gap=1 - r["start"]["thr"] / r["thr_ub"], gain=r["thr"] / r["start"]["thr"] - 1,
                   lamN=ln, best=r["thr"], start=r["start"]["thr"], thr_ub=r["thr_ub"], src=r["src"], evals=r["evals"],
                   threads=r["threads"], minutes=r["minutes"], check=r["check"],
                   warm_src=r.get("warm", {}).get("src", r["start"]["src"]), warm_N=r["N"] * DAY / r.get("warm", r["start"])["W"] / ln,
                   rule=(r["thr"] - r["fpr"]) / (ln - r["fpr"]), rule_first_search=(r["start"]["thr"] - r["fpr"]) / (ln - r["fpr"]))
        for nm in ("start", "best"):         # excess of the completion W over the finite-window linear programme W_lp = extra occupancy of road010-6 + gaps the pairwise precedence cannot avoid + other gaps (including the head)
            g = r["split"][nm]
            ex = g["W"] - r["W_lp"]
            row.update({f"{nm}_busy": g["busy"] / g["W"], f"{nm}_forced": g["forced"] / g["W"], f"{nm}_other": (g["other"] + g["vehicle"]) / g["W"],
                        f"{nm}_head": g["start"] / g["W"], f"{nm}_ex_occ": (g["busy"] - r["W_lp"]) / ex, f"{nm}_ex_forced": g["forced"] / ex,
                        f"{nm}_ex_other": (g["other"] + g["vehicle"] + g["start"]) / ex})
            row[f"{nm}_sum"] = row[f"{nm}_busy"] + row[f"{nm}_forced"] + row[f"{nm}_other"] + row[f"{nm}_head"]
        rows.append(row)
    rows.sort(key=lambda x: (x["N"], x["seed"]))
    with open(out / "points.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    global SUBNET_DIR
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--subnet", required=True, help="output directory of experiments/subnet_exact.py (points/, ext/, bounds2.json)")
    ap.add_argument("--gap", default=None, help="output directory of experiments/subnet_gap.py (bounds3.json)")
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--slots", default=None, help="CPU slot directory (default: environment variable CPU_SLOTS, else none)")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--warm", default=None, help="output directory of an earlier run of this module: each point starts from the better of its solution and the sub-network one")
    ap.add_argument("--restarts", type=int, default=1, help="number of annealing legs (each reheats from the current best solution)")
    ap.add_argument("--resplit", default=None, help="output directory of an earlier run of this module: recompute its idle split with the current gap_split into --out")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    SUBNET_DIR = Path(a.subnet)
    import retime as R                     # the points depend only on these functions (without docstrings); changes of the aggregation do not change the hash
    out = Path(a.out)
    if a.resplit:
        h = R.func_sha(ROOT / "experiments" / "subnet_gap.py", ("gap_split", "chain3", "veh_bound"), hashlib.sha256())
        cfg = dict(title="corridor: idle split of the best solutions on road010-6 recomputed (head included)", source=str(a.resplit), code_sha256=h.hexdigest())
        out.mkdir(parents=True, exist_ok=True)
        F.write_json(out / "config.json", cfg)
        aggregate(cfg, out, src=Path(a.resplit), gap_dir=a.gap)
        print(f"-> {out}", flush=True)
        return
    code = hashlib.sha256()
    for f in sorted((ROOT / "exact").glob("*.py")) + [ROOT / "experiments" / "subnet_exact.py"]:
        code.update(f.name.encode() + f.read_bytes())
    R.func_sha(__file__, ("subnet_best", "Booker", "anneal", "_anneal_leg", "run_point"), code)
    R.func_sha(ROOT / "experiments" / "subnet_gap.py", ("gap_split",), code)
    R.func_sha(ROOT / "experiments" / "retime.py", R.COMPUTE + ("from_seq",), code)
    cfg = dict(title="through-corridor sub-network: feasible schedules (simulated annealing + same-dispatch offline rebooking)", inst=INST,
               seeds=list(SEEDS), ns=list(NS), K=K, minutes=a.minutes, polish=POLISH, start_from=str(SUBNET_DIR), code_sha256=code.hexdigest(),
               ortools=__import__("ortools").__version__,
               **(dict(warm_from=a.warm, restarts=a.restarts) if a.warm or a.restarts > 1 else {}))
    (out / "points").mkdir(parents=True, exist_ok=True)
    if (out / "config.json").exists() and json.loads((out / "config.json").read_text(encoding="utf-8")) != json.loads(json.dumps(cfg)):
        sys.exit(f"{out} holds a run of another configuration")
    F.write_json(out / "config.json", cfg)
    ks = keys()
    pf = lambda k: out / "points" / F.fname(k)
    prog = F.Progress(out / "progress.json", len(ks), sum(pf(k).exists() for k in ks), 0, cfg["title"], "corridor")
    if not a.aggregate_only:
        SUB.OUT, SUB.MIN_W, SUB.MAX_W = out, 1, 1     # one thread per point (single-threaded annealing); points run in parallel
        deadline = time.time() + 60 * a.max_minutes if a.max_minutes else None
        if SUB.schedule([k for k in ks if not pf(k).exists()], a.minutes, pf, prog, deadline, a.slots, target=run_point):
            prog.write(state="paused", note="stopped at --max-minutes; the same command resumes")
            aggregate(cfg, out, gap_dir=a.gap)
            sys.exit(3)
    aggregate(cfg, out, gap_dir=a.gap)
    prog.write(state="done" if all(pf(k).exists() for k in ks) else "paused", processes=0)
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()

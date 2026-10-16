"""Optimal throughput on small sub-network instances (CP-SAT): how close full-path reservation and the bounds come to the
best schedule that waits only at the origin (the exact sub-network study of the paper).

Each point = instance x seed x number of tasks N x fleet size K:
  1. instance (exact.model): sub-network + the first N tasks of the task stream of the simulation conditioned on the
     sub-network (released at time 0) + start positions of the K vehicles;
  2. full-path reservation (simulation, FPR): completion time and the sequence of each vehicle; checked by the independent
     checker in its own way (destination plus the drive-off of the empty vehicle);
  3. bounds: lambda_W^N of Proposition 2 (the mix of this batch, truncation of Eq. (5)) and beta; the finite-window linear
     programme from an empty system (the same occupancy rows, boundary as in the instance);
  4. CP-SAT: first the dispatch and order of full-path reservation are fixed and only the times are set (checks that it is
     feasible in the model and gives an integer feasible solution), then, with that as a hint and the linear programme as a
     lower bound, the optimal schedule that waits only at the origin is solved; the independent checker checks the solution.
Throughput is always lambda = N·H / W (H = 16 h, W = time at which all N tasks are completed).

Run:   python experiments/subnet_exact.py --out <dir> [--secs S] [--only INSTANCE ...] [--max-minutes M] [--aggregate-only]
           [--slots DIR] [--extend INSTANCE ... [--ext-secs S] [--ext-gap G]]
Writes to <dir>: config.json, progress.json, points/ (one JSON per point with the schedules and the number of CP-SAT threads;
a cache for resuming), ext/ (extended solves, --extend), points.csv and summary.csv. experiments/subnet_bounds.py adds
bounds2.json (the sharpened occupancy-order bound), which the aggregation reads. The released table of the run of the paper
is results/subnet/points.csv (configuration results/subnet/config.json); the per-point files (points/, ext/, bounds2.json) are
caches of a run and are not released, so --extend, subnet_bounds.py, subnet_gap.py and corridor.py read the --out directory of
a run of this module made first.
Computing: processes are pinned to the cores other than core 0; each point runs in its own process with MAX_W CP-SAT threads
(with --slots: the CPU slots it takes, between MIN_W and MAX_W); the number of threads is recorded per point and is not part
of the configuration.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import runner as F                        # noqa: E402
import dock_mouth as D                    # noqa: E402
from occupancy import params as P         # noqa: E402
from occupancy.mapgraph import to_graph   # noqa: E402
from occupancy.network import Net         # noqa: E402

CORES = list(range(1, os.cpu_count()))       # leave core 0 free
DAY = 16 * 3600.0
INSTANCES = {
    # Okpo dock 1 front road (road161, single lane of 9 m, 1,095 m) with the roads and stops at both ends: the binding element
    # of the whole yard under the main flow mix
    "yupu_dock1": dict(label="Okpo dock 1 front road", yard="yupu", roads=["road161", "road160", "road162", "road008", "road291", "road007-4"]),
    # Yantai building014 (paint shop, two doors), the area at its doors: the roads within 200 m of the doors; tasks with origin
    # or destination at building014 (its doors bind the whole yard under the main flow mix)
    "yantai_paint": dict(label="Yantai building014 doors", yard="yantai", ball=("building014", 200.0), ents=["building014"]),
    # Okpo through corridor: the roads within 400 m of both ends of road010-6 (12 m, 168 m); tasks whose loaded route passes it
    # (the through corridor that binds the whole yard under sectioning)
    "yupu_corridor": dict(label="Okpo through corridor road010-6", yard="yupu", ball=("road010-6", 400.0), via=["road010-6"]),
    # synthetic dock-mouth network (the network of the dock-mouth figure of the paper; single-lane dock road of 9 m, 300 m, one
    # crane at the dock, four places in the adjacent area): background tasks + erections (with crane windows)
    "synth_dock": dict(label="synthetic dock mouth (with crane)", synthetic=True),
}
SEEDS = (101, 102, 103)
NS = (20, 30, 60)
N_SEEDS = NS          # numbers of tasks computed for all three seeds (the others only for the first seed)
KS = (8,)             # K = 8: all four instances are network-bound (lambda^F > lambda_W^N in trial runs), the case of the saturated level of the paper
SYN_P6 = 30           # synthetic network: one erection every 30 tasks (inserted at equal intervals), the others are background tasks
UNIT = 0.1            # time unit of CP-SAT (s)
FIX_S = 60.0          # time limit (s) of the solve with the dispatch and order of full-path reservation fixed


def net_of(name):
    s = INSTANCES[name]
    if s.get("synthetic"):
        p = dict(P.DEFAULTS, **P.SCENARIOS["crane"], p6_per_day=1.0)
        return Net(to_graph(D.dock_map(9.0, yard_s=(230.0, 245.0, 260.0, 275.0))), p), p
    Y = P.YARDS[s["yard"]]
    p = P.scenario("main")
    return Net(Y["network"], p, Y["roles"]), p


_CACHE = {}


def instance(name, seed, N):
    """The first N tasks of the instance (for one seed the smaller N are prefixes of the larger, "from small to large"); the
    start positions are drawn at the largest N and are the same for every N."""
    from exact import model as X
    if name not in _CACHE:
        _CACHE[name] = net_of(name)
    net, p = _CACHE[name]
    s, Nmax = INSTANCES[name], max(NS)
    key = (name, seed)
    if key not in _CACHE:
        if s.get("synthetic"):
            rng = random.Random(seed)
            yards = sorted(x for x in net.stops if x.startswith("zyard"))
            six = {round((i + 0.5) * SYN_P6) for i in range(Nmax // SYN_P6)}
            tasks = [dict(id=j, flow="P6" if j in six else "BG", o=rng.choice(D.SRC_IDS), d="zdock#0" if j in six else yards[j % len(yards)],
                          B=7.0, Lv=20.0, release=0.0) for j in range(Nmax)]       # the four places of the adjacent area in turn, so that no single place becomes the bottleneck
            roads = {a[3] for a in net.arc}
        else:
            if "roads" in s:
                roads = {rid for rid, r in net.road.items() if r["name"] in s["roads"]}
            else:
                ent, R = s["ball"]
                seeds = {net.stops[x]["node"] for e, v in net.ents.items() if v["name"].startswith(ent) for x in v["stops"]} | \
                    {x for r in net.road.values() if r["name"] == ent for x in (r["a"], r["b"])}
                roads = X.ball_roads(net, seeds, R)
            ents = [e for e, v in net.ents.items() if v["name"].startswith(tuple(s.get("ents", ())))] if s.get("ents") else None
            via = [rid for rid, r in net.road.items() if r["name"] in s.get("via", ())] or None
            tasks = X.sample_tasks(net, p, seed, roads, Nmax, ents, via)
        _CACHE[key] = (tasks, X.vehicle_starts(net, tasks, seed, max(KS)), roads)
    tasks, starts, roads = _CACHE[key]
    return X.Instance(f"{name}|{seed}|{N}", net, p, tasks[:N], starts, roads)


def compute(key, secs, workers):
    """One point: instance|seed|N|K."""
    from exact import check as C, model as X
    name, seed, N, K = key.split("|")
    seed, N, K = int(seed), int(N), int(K)
    inst = instance(name, seed, N)
    net, p = inst.net, inst.p
    thr = lambda W: N * DAY / W
    secs = secs * N                     # the time limit scales with the number of tasks (--secs is seconds per task)
    t0 = time.time()
    mk, seq, _ = X.run_fpr(inst, K)
    c_fpr = C.check(net, p, inst.tasks, inst.starts, seq, clr=True)
    t_fpr = time.time() - t0
    W_lp = X.lp_finite(inst, K)
    beta = X.beta(inst, K)
    lamN_fpr, bind = X.lam_window(inst, mk)
    W_ch = X.chain_bound(inst, K, tuple(bind[0]["res"]))
    lb = max(W_lp, W_ch or 0.0)
    rb = X.rebook(inst, K, seq, UNIT)
    W_rb = max(b + inst.dur_l(j) for v in rb.values() for j, a, b in v)
    t0 = time.time()
    fx = X.solve(inst, K, min(FIX_S, secs), workers, fix=rb, lb=lb, horizon=W_rb + 60, unit=UNIT)
    t_fix = time.time() - t0
    t0 = time.time()
    op = X.solve(inst, K, secs, workers, hint=fx["seq"], lb=lb, horizon=fx["M"], unit=UNIT)
    t_opt = time.time() - t0
    c_opt = C.check(net, p, inst.tasks, inst.starts, op["seq"], clr=False)
    c_fix = C.check(net, p, inst.tasks, inst.starts, fx["seq"], clr=False)
    W_opt = op["makespan"]
    lamN, bind = X.lam_window(inst, W_opt)
    flows = {}
    for t in inst.tasks:
        flows[t["flow"]] = flows.get(t["flow"], 0) + 1
    busy = {}
    for nm, sq in (("fpr", seq), ("opt", op["seq"])):
        b = C.busy(net, p, inst.tasks, inst.starts, sq, [tuple(x["res"]) for x in bind[:1]])
        busy[nm] = {str(k): v for k, v in b.items()}
    return dict(key=key, inst=name, seed=seed, N=N, K=K, flows=flows, starts=inst.starts[:K], added_roads=sorted(inst.added),
                n_roads=len(inst.roads),
                fpr=dict(W=mk, thr=thr(mk), check=c_fpr, secs=t_fpr, seq=seq),
                rebook=dict(W=W_rb),
                fix=dict(status=fx["status"], M=fx["M"], W=fx["makespan"], thr=thr(fx["makespan"]), secs=t_fix, check=c_fix, seq=fx["seq"]),
                opt=dict(status=op["status"], M=op["M"], bound=op["bound"], W=W_opt, thr=thr(W_opt), thr_ub=thr(max(op["bound"], 1e-9)),
                         gap=(op["M"] - op["bound"]) / op["M"], secs=t_opt, check=c_opt, seq=op["seq"], scale=op["scale"], unit=UNIT),
                lp=dict(W=W_lp, thr=thr(W_lp)), chain=dict(W=W_ch, thr=thr(W_ch) if W_ch else None),
                lamN=lamN, lamN_fpr=lamN_fpr, beta=beta,
                lamNb=lamN * (1 + beta / W_opt), binding=[x["name"] for x in bind[:3]], busy=busy,
                cpsat=dict(workers=workers, secs=secs))


OUT = None


def code_sha():
    """SHA-256 of the code the points depend on (occupancy/, bounds/, exact/ of this repository), recorded in config.json."""
    h = hashlib.sha256()
    for d in ("occupancy", "bounds", "exact"):
        for f in sorted((ROOT / d).glob("*.py")):
            h.update(f"{d}/{f.name}".encode() + f.read_bytes())
    return h.hexdigest()


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--secs", type=float, default=15.0, help="CP-SAT time limit per task (s)")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--slots", default=None, help="CPU slot directory (default: environment variable CPU_SLOTS, else none)")
    ap.add_argument("--extend", nargs="*", default=None, help="after all points: extend the solve of the points of these instances (ext/, see extend_point)")
    ap.add_argument("--ext-secs", type=float, default=30.0, help="time limit per task of the extended solve (s)")
    ap.add_argument("--ext-gap", type=float, default=0.05, help="extend only the points whose gap from the best solution to the best bound exceeds this")
    a = ap.parse_args()
    F.pin(CORES)
    cfg = dict(title="optimal throughput on small sub-network instances (CP-SAT)", instances=INSTANCES, seeds=list(SEEDS), ns=list(NS),
               ks=list(KS), n_seeds=list(N_SEEDS), syn_p6=SYN_P6, unit=UNIT, fix_s=FIX_S, secs_per_task=a.secs,
               networks={y: P.YARDS[y]["network"] for y in sorted({s["yard"] for s in INSTANCES.values() if "yard" in s})},
               code_sha256=code_sha(), ortools=__import__("ortools").__version__, day_s=DAY)
    OUT = Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "config.json").exists() and json.loads((OUT / "config.json").read_text(encoding="utf-8")) != json.loads(json.dumps(cfg)):
        sys.exit(f"{OUT} holds a run of another configuration")
    F.write_json(OUT / "config.json", cfg)
    (OUT / "points").mkdir(exist_ok=True)
    keys = [f"{n}|{s}|{N}|{K}" for n in INSTANCES for N in NS for s in SEEDS for K in KS
            if (a.only is None or n in a.only) and (s == SEEDS[0] or N in N_SEEDS)]
    pf = lambda k: OUT / "points" / F.fname(k)
    done = [k for k in keys if pf(k).exists()]
    prog = F.Progress(OUT / "progress.json", len(keys), len(done), 0, cfg["title"], "subnet_exact")
    deadline = time.time() + 60 * a.max_minutes if a.max_minutes else None
    if not a.aggregate_only:
        stopped = schedule([k for k in keys if not pf(k).exists()], a.secs, pf, prog, deadline, a.slots)
        if not stopped and a.extend:
            ef = lambda k: OUT / "ext" / F.fname(k)
            (OUT / "ext").mkdir(exist_ok=True)
            ek = [k for k in keys if k.split("|")[0] in a.extend and pf(k).exists() and (ef(k).exists() or cur_gap(k) > a.ext_gap)]
            prog.write(total=len(keys) + len(ek), done=prog.d["done"] + sum(1 for k in ek if ef(k).exists()))
            stopped = schedule([k for k in ek if not ef(k).exists()], a.ext_secs, ef, prog, deadline, a.slots, target=extend_point)
        if stopped:
            prog.write(state="paused", note="stopped at --max-minutes; the same command resumes")
            aggregate(cfg, keys)
            sys.exit(3)
    aggregate(cfg, keys)
    prog.write(state="done" if all(pf(k).exists() for k in keys) else "paused", processes=0)


MIN_W, MAX_W = 4, 8      # CPU slots a point needs before it starts and at most takes (CP-SAT threads); with fewer than 4 threads CP-SAT is weak


def cur_gap(key):
    """Current gap of a point: 1 - best known (the better of the CP-SAT best solution and full-path reservation) / best bound
    (the smaller of the CP-SAT bound and the occupancy-order bounds)."""
    r = json.loads((OUT / "points" / F.fname(key)).read_text(encoding="utf-8"))
    b2f = OUT / "bounds2.json"
    c2 = (json.loads(b2f.read_text(encoding="utf-8"))["points"].get(key) or {}).get("chain2") if b2f.exists() else None
    ub = min([r["opt"]["thr_ub"]] + [x for x in (r["chain"]["thr"], r["N"] * DAY / c2 if c2 else None) if x])
    return 1 - max(r["opt"]["thr"], r["fpr"]["thr"]) / ub


def run_point(key, secs, w, out):
    """Child process: computes one point and writes its JSON."""
    global OUT
    OUT = Path(out)
    try:
        r = compute(key, secs, w)
        F.write_json(OUT / "points" / F.fname(key), r)
    except Exception:
        import traceback
        (OUT / "failures.log").open("a", encoding="utf-8").write(f"{key}\n{traceback.format_exc()}\n")


def extend_point(key, secs, w, out):
    """Child process: extended solve (ext/<key>.json). With the best solution of the point file as the starting solution and
    the finite-window linear programme and the occupancy-order bounds (including bounds2.json) as the lower bound, solves once
    more with time limit secs x N on w threads; the independent checker checks the solution. The aggregation takes the better
    best solution and the better bound of the two solves."""
    global OUT
    OUT = Path(out)
    try:
        from exact import check as C, model as X
        r = json.loads((OUT / "points" / F.fname(key)).read_text(encoding="utf-8"))
        inst = instance(r["inst"], r["seed"], r["N"])
        b2 = json.loads((OUT / "bounds2.json").read_text(encoding="utf-8"))["points"] if (OUT / "bounds2.json").exists() else {}
        lb = max(x for x in (r["lp"]["W"], r["chain"]["W"], (b2.get(key) or {}).get("chain2"), 0.0) if x)
        hint = {int(k): [tuple(x) for x in v] for k, v in r["opt"]["seq"].items()}
        t0 = time.time()
        op = X.solve(inst, r["K"], secs * r["N"], w, hint=hint, lb=lb, horizon=r["opt"]["M"], unit=UNIT)
        ok = C.check(inst.net, inst.p, inst.tasks, inst.starts, op["seq"])["ok"] if "seq" in op else False
        F.write_json(OUT / "ext" / F.fname(key), dict(key=key, status=op["status"], M=op.get("M"), bound=op["bound"], W=op.get("makespan"),
                                                       thr=r["N"] * DAY / op["makespan"] if "makespan" in op else None,
                                                       thr_ub=r["N"] * DAY / max(op["bound"], 1e-9), secs=time.time() - t0, threads=w,
                                                       secs_per_task=secs, check=ok, seq=op.get("seq")))
    except Exception:
        import traceback
        (OUT / "failures.log").open("a", encoding="utf-8").write(f"ext {key}\n{traceback.format_exc()}\n")


def schedule(todo, secs, pf, prog, deadline, slots_dir=None, target=None):
    """Computes the points in parallel, each in its own process (target, default run_point). With a CPU slot directory
    (slots_dir or the environment variable CPU_SLOTS): a point starts once MIN_W slots are taken and takes up to MAX_W, the
    number of CP-SAT threads = the slots taken, all returned when the point is done. Without slots: MAX_W threads per point
    and as many points at once as fit on the cores other than core 0. After the deadline no new point starts; the running ones
    finish. Returns whether it stopped at the deadline."""
    import multiprocessing as mp
    sd = slots_dir or os.environ.get("CPU_SLOTS")
    slots = None
    if sd:
        sys.path.insert(0, sd)
        from cpu_slots import Slots
        slots = Slots(f"subnet-{os.getpid()}")
    running, stopped, pend = {}, False, 0          # pend: slots taken, not yet given to a point
    while todo or running:
        for k, (pr, w) in list(running.items()):
            if not pr.is_alive():
                pr.join()
                del running[k]
                for _ in range(w if slots else 0):
                    slots.give()
                if pf(k).exists():
                    prog.log(f"{k} ({w} threads)", done=prog.d["done"] + 1)
                else:
                    prog.log(f"{k}: failed (see failures.log)", failed=prog.d["failed"] + 1)
        stopped = stopped or (deadline is not None and time.time() > deadline)
        w = 0
        if slots:
            slots.want(0 if stopped else len(todo) * MAX_W + len(slots.held))     # heartbeat
            while todo and not stopped and pend < MAX_W and slots.take():
                pend += 1
            if pend >= MIN_W:
                w, pend = pend, 0
        elif todo and not stopped and sum(x for _, x in running.values()) + MAX_W <= max(MAX_W, len(CORES)):
            w = MAX_W
        if w:
            k = todo.pop(0)
            pr = mp.Process(target=target or run_point, args=(k, secs, w, str(OUT)))
            pr.start()
            running[k] = (pr, w)
        prog.write(processes=sum(x for _, x in running.values()), note=" ".join(f"{k}:{x}" for k, (_, x) in running.items()))
        if not w:
            time.sleep(3)
        if stopped and not running:
            break
    if slots:
        slots.close()
    return stopped and bool(todo)


def fpr_in_model(inst, seq):
    """Consistency of the definitions: the continuous-time schedule of full-path reservation put into the windows of the model
    (exact.model.Instance.chain: empty run, origin window, loaded run and destination), the windows of one vehicle not merged
    but added (as the cumulative constraints of CP-SAT write them), capacity checked resource by resource. Returns the number
    of overloaded resources (overload longer than 1e-6 s); 0 means the schedule of full-path reservation is a feasible solution
    of the model."""
    import collections
    by = collections.defaultdict(list)
    for k, lst in seq.items():
        loc = inst.starts[int(k)]
        for j, t1, t2 in lst:
            c = inst.chain(j, loc)
            by[c["so"]].append((t1 + c["ao"], t2 + c["oend"], c["d_so"]))
            for r, a, b, d in c["WE"]:
                by[r].append((t1 + a, t1 + b, d))
            for r, a, b, d in c["WL"]:
                by[r].append((t2 + a, t2 + b, d))
            loc = inst.net.stops[inst.tasks[j]["d"]]["node"]
    bad = 0
    for r, iv in by.items():
        ts = sorted({x for s, e, _ in iv for x in (s, e)})
        if any(b - a > 1e-6 and sum(d for s, e, d in iv if s <= a + 1e-9 and e >= b - 1e-9) > inst.cap[r] + 1e-6 for a, b in zip(ts, ts[1:])):
            bad += 1
    return bad


def aggregate(cfg, keys):
    pts = [json.loads((OUT / "points" / F.fname(k)).read_text(encoding="utf-8")) for k in keys if (OUT / "points" / F.fname(k)).exists()]
    b2f = OUT / "bounds2.json"                   # sharpened lower bounds computed afterwards (experiments/subnet_bounds.py)
    b2 = json.loads(b2f.read_text(encoding="utf-8"))["points"] if b2f.exists() else {}
    rows = []
    from exact import check as C, model as X
    import ast
    for r in pts:
        inst = instance(r["inst"], r["seed"], r["N"])
        ext_used = False
        ef = OUT / "ext" / F.fname(r["key"])         # extended solve (extend_point): the better best solution and the better bound
        if ef.exists():
            x = json.loads(ef.read_text(encoding="utf-8"))
            if x["check"] and x["W"] and x["W"] < r["opt"]["W"]:
                r["opt"].update(thr=x["thr"], W=x["W"], status=x["status"], seq=x["seq"])
                ext_used = True
            r["opt"]["thr_ub"] = min(r["opt"]["thr_ub"], x["thr_ub"])
            r["opt"]["secs"] += x["secs"]
            r["opt"]["gap"] = 1 - r["opt"]["thr"] / r["opt"]["thr_ub"]
            r["cpsat"]["workers"] = f"{r['cpsat']['workers']}+{x['threads']}"
        if ext_used:                                  # busy share of the binding resource recomputed for the extended solution (busy in the point file is that of the first solve)
            res = [ast.literal_eval(k) for k in r["busy"]["opt"]]
            seq = {int(k): [tuple(y) for y in v] for k, v in r["opt"]["seq"].items()}
            r["busy"]["opt"] = {str(k): v for k, v in C.busy(inst.net, inst.p, inst.tasks, inst.starts, seq, res).items()}
        r["lamF"] = X.lam_fleet(inst, r["K"])        # fleet bound of Proposition 1 (least empty driving on the sub-network)
        r["fpr_model_bad"] = fpr_in_model(inst, r["fpr"]["seq"])
        lF, lX, lO, lN, l0 = r["fpr"]["thr"], r["fix"]["thr"], r["opt"]["thr"], r["lamN"], r["lp"]["thr"]
        # the schedule of full-path reservation is itself a feasible solution of the model (continuous time; checked by
        # fpr_in_model), so the best known schedule is at least as good; at points where CP-SAT did not recover it within the
        # time limit (offsets of the start after rounding and rebooking, too little search), the optimum and the same-dispatch
        # offline rebooking take it
        cp_below = lO < lF
        if r["fpr_model_bad"] == 0:
            lX, lO = max(lX, lF), max(lO, lF)
        c2 = (b2.get(r["key"]) or {}).get("chain2")
        # upper bound of the optimum: the tighter of the CP-SAT bound (which includes the finite-window linear programme and
        # the occupancy-order bound) and the sharpened occupancy-order bound
        cands = [x for x in (r["chain"]["thr"], r["N"] * DAY / c2 if c2 else None) if x]
        r["chain"]["thr"] = min(cands) if cands else None          # throughput limit of the occupancy-order bounds, the tighter one
        lU = min([r["opt"]["thr_ub"]] + cands)
        w = lN - lF
        # three-part split for the door type (occupancy-order bound binding; origin by Definition 1; parts of bounds2):
        # lambda_W^N -> finite-window linear programme is all boundary term; linear programme -> occupancy-order bound is shared
        # between occupancy order and boundary term in the ratio of "sum of adjacent gaps" to "first entry + last task from
        # leaving to the end of unloading" in the excess of the bound's completion time over the linear programme;
        # occupancy-order bound -> best bound (where CP-SAT proves a tighter one) is "other"; best bound -> best solution is
        # undetermined; best solution -> FPR is the rule
        pa = (b2.get(r["key"]) or {}).get("parts")
        split = {}
        if r["inst"] == "yantai_paint" and pa and c2 and c2 > r["lp"]["W"]:
            lc = r["N"] * DAY / c2
            ex = c2 - r["lp"]["W"]
            split = dict(sh_order=(l0 - lc) * pa["order"] / ex / w, sh_boundary=((lN - l0) + (l0 - lc) * (ex - pa["order"]) / ex) / w,
                         sh_other=(lc - lU) / w, sh_rule=(lO - lF) / w, sh_und=(lU - lO) / w, steady=pa["base"] / (pa["base"] + pa["order"]),
                         gap_mean=pa["gap_mean"], gap_min=pa["gap_min"], gap_max=pa["gap_max"], cyc=pa["base"] / pa["J"])
        rows.append(dict(inst=r["inst"], seed=r["seed"], N=r["N"], K=r["K"], flows=" ".join(f"{k}:{v}" for k, v in sorted(r["flows"].items())),
                         fpr=lF, fix=lX, opt=lO, opt_ub=lU, status=r["opt"]["status"], gap=r["opt"]["gap"], gap2=1 - lO / lU,
                         secs=r["opt"]["secs"], threads=r["cpsat"]["workers"], reused_from=r.get("reused_from", ""), cp_below_fpr=cp_below,
                         cp_opt=r["opt"]["thr"],
                         lamN=lN, lamNb=r["lamNb"], beta_h=r["beta"] / 3600, lp0=l0, chain=r["chain"]["thr"],
                         fpr_N=lF / lN, fix_N=lX / lN, opt_N=lO / lN, optub_N=lU / lN, lp0_N=l0 / lN,
                         chain_N=r["chain"]["thr"] / lN if r["chain"]["thr"] else None,
                         # split of the interval [FPR, lambda_W^N]: timing under the same dispatch (FPR -> optimum with fixed
                         # dispatch), dispatch and order (-> optimum), slack of the bound (optimum -> lambda); when the optimum
                         # is not proven, the rule side takes the lower end of [best solution, bound] and the bound side the
                         # upper end (the slack of the bound is at least lambda - bound)
                         timing_share=(lX - lF) / w, dispatch_share=(lO - lX) / w, rule_share=(lO - lF) / w, rule_share_hi=(lU - lF) / w,
                         bound_share=(lN - lO) / w, bound_share_lo=(lN - lU) / w,
                         # the same split with the upper end replaced by the finite-window linear programme (from an empty
                         # system, boundary as in the instance; a strict upper bound for these instances, always >= the optimum)
                         rule_share0=(lO - lF) / (l0 - lF), rule_share0_hi=(lU - lF) / (l0 - lF),
                         bound_share0=(l0 - lO) / (l0 - lF), bound_share0_lo=(l0 - lU) / (l0 - lF),
                         fpr_ok=r["fpr"]["check"]["ok"] and r["fpr_model_bad"] == 0, fix_ok=r["fix"]["check"]["ok"], opt_ok=r["opt"]["check"]["ok"],
                         fpr_model_bad=r["fpr_model_bad"], fpr_minus_opt_s=r["fpr"]["W"] - r["opt"]["W"],
                         fix_vs_fpr_s=r["fix"]["W"] - r["fpr"]["W"], binding=r["binding"][0] if r["binding"] else "",
                         busy_fpr=next(iter(r["busy"]["fpr"].values()), None), busy_opt=next(iter(r["busy"]["opt"].values()), None),
                         ext_used=ext_used, lamF=r["lamF"], lamF_N=r["lamF"] / lN,
                         **{k: split.get(k) for k in ("sh_order", "sh_boundary", "sh_other", "sh_rule", "sh_und", "steady",
                                                      "gap_mean", "gap_min", "gap_max", "cyc")}))
    if not rows:
        return
    with open(OUT / "points.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summ = []
    for name in INSTANCES:
        for N in NS:
            for K in KS:
                g = [x for x in rows if x["inst"] == name and x["N"] == N and x["K"] == K]
                if not g:
                    continue
                mean = lambda c: statistics.mean(x[c] for x in g)
                summ.append(dict(inst=name, N=N, K=K, seeds=len(g), fpr_N=mean("fpr_N"), fix_N=mean("fix_N"), opt_N=mean("opt_N"),
                                 optub_N=mean("optub_N"), lp0_N=mean("lp0_N"),
                                 timing_share=mean("timing_share"), dispatch_share=mean("dispatch_share"),
                                 rule_share=mean("rule_share"), rule_share_hi=mean("rule_share_hi"),
                                 bound_share=mean("bound_share"), bound_share_lo=mean("bound_share_lo"),
                                 rule_share0=mean("rule_share0"), rule_share0_hi=mean("rule_share0_hi"),
                                 bound_share0=mean("bound_share0"), bound_share0_lo=mean("bound_share0_lo"),
                                 opt_lp0=mean("opt") / mean("lp0"), fpr_lp0=mean("fpr") / mean("lp0"),
                                 gap_max=max(x["gap2"] for x in g), optimal=sum(x["status"] == "OPTIMAL" for x in g),
                                 secs_max=max(x["secs"] for x in g), lamN=mean("lamN"), fpr=mean("fpr"), opt=mean("opt"),
                                 all_ok=all(x["fpr_ok"] and x["fix_ok"] and x["opt_ok"] for x in g)))
    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, list(summ[0]))
        w.writeheader()
        w.writerows(summ)


if __name__ == "__main__":
    main()

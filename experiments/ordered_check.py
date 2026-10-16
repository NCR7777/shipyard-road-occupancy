"""Extended self-check of execution in booked order: independent checks and adversarial instances.

Runs on the final code (ordered_proj.OPE2: execution in booked order + lifts inside the crane span + audit record,
optionally booking on projected occupancy):
  base   the self-check of ordered.py (its 15 instances x 4 time models x adm / rel, 100 seeds each, with the tasks and the
         control of ordered.job); optionally compared row by row with an earlier run of ordered.py (--base, all equal
         apart from the time used);
  adv    adversarial instances and time models: wide road 17 / 19 / 23 m (two roadside stops on the shared road, mixed
         vehicle widths), ring 3 / 5, sectioned corridor, random grids 1-8 (10 vehicles, 60 tasks) x nominal / shorter /
         mixed / grid-aligned / crawling;
  crane  crane instances: dock front 9 / 15 m, 1 / 2 cranes, dock-side off-road bay;
  lift   the crane instances with a third of the P6 marked as lifts inside the crane span (the gantry crane lifts the
         block straight into the dock, holding only the crane);
  each suite is also run with booking on projected occupancy (rule adm_proj, proj=True of ordered_proj) and with its
  version booking and projecting with mean durations (adm_proj_mean, the yard rule reserve_op3; non-nominal times only).
Checks independent of the executor: C1 the actual occupancy, rescanned time by time from the [entry, release) of the
released bookings (per vehicle the largest width, cranes by count), stays within capacity; C2 (adm, adm_proj) the entry
order on every resource agrees with the keys; C3 every booking is entered and released, every chain completes. At nominal
times the completion times are also compared with reserve (lifts by their start); where they differ, the completion
events of reserve at equal times are replayed in the order execution in booked order processed them, and after the replay
they must agree one by one.

  python experiments/ordered_check.py --out DIR [--seeds 100] [--procs 10] [--slots DIR] [--base DIR]
  -> DIR/cases.csv (one row per suite, instance, seed, time model and rule), DIR/summary.json, DIR/tables.md.
--base: output directory of a run of ordered.py (its cases.csv) to compare the base suite with."""
import argparse
import collections
import csv
import hashlib
import heapq
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "tests"))
import ordered as O                                  # noqa: E402
import ordered_proj as Q                             # noqa: E402
from occupancy import params as P                    # noqa: E402
from occupancy import sim as S                       # noqa: E402

RULES = {"adm": dict(rule="adm"), "rel": dict(rule="rel"), "adm_proj": dict(rule="adm", proj=True), "adm_proj_mean": dict(rule="adm", proj=True)}
# adm_proj_mean (the yard rule reserve_op3): booking and projection with mean durations (book="mean" of ordered.py: handling
# 1.5 times, mean driving time); actual times by each time model, only the non-nominal ones (the booked durations are not
# nominal, so the nominal comparison with reserve does not apply)
MEAN_BOOK = dict(mean=1.5, cv=0.3, speed=[0.9, 1.0], book_q=None, book="mean")


def independent_checks(sim):
    """C1, C2, C3. Returns the largest excess over capacity (should be <= 0)."""
    by = collections.defaultdict(list)
    for b in sim.log_bk:
        assert b.ta is not None and b.tr is not None and b.tr >= b.ta - 1e-9, "C3 booking not entered or not released"
        by[b.res].append(b)
    left = sum(len(v) for v in sim.bk.values())
    assert left == 0, f"C3 {left} bookings not released"
    worst = 0.0
    for res, L in by.items():
        cap = sim.cap[res]
        for t in sorted({b.ta for b in L}):
            held = [b for b in L if b.ta <= t < b.tr]
            if res[0] == "c":
                load = len(held)
            else:
                per = collections.defaultdict(float)
                for b in held:
                    per[b.vid] = max(per[b.vid], b.d)
                load = sum(per.values())
            worst = max(worst, load - cap)
            assert load <= cap + 1e-9, ("C1 over capacity", res, t, load, cap)
        if sim.rule == "adm":
            seqd = sorted(L, key=lambda b: sim.b_seq[b])
            for b1, b2 in zip(seqd, seqd[1:]):
                assert (b1.cid == b2.cid and b1.ev == b2.ev) or b1.key() < b2.key(), ("C2 entry order", res, b1.key(), b2.key())
    return worst


# ---------------------------------------------------------------- adversarial time models (some shorter than nominal)
def fast(sim, task):
    rng = random.Random(f"fast|{sim.seed}|{task['id']}")
    return sim.p["load_s"] * rng.uniform(0.05, 1), sim.p["unload_s"] * rng.uniform(0.05, 1), rng.uniform(1, 4), rng.uniform(1, 4)


def mixed(sim, task):
    rng = random.Random(f"mixed|{sim.seed}|{task['id']}")
    if rng.random() < 0.5:
        return fast(sim, task)
    h = lambda h0: h0 * (1 + rng.expovariate(0.3))
    return h(sim.p["load_s"]), h(sim.p["unload_s"]), rng.uniform(0.05, 1.0), rng.uniform(0.05, 1.0)


def grid(sim, task):
    """Durations aligned to a grid, to create events at equal times: handling 600 or 1200 s, speed 1 or 1/2."""
    rng = random.Random(f"grid|{sim.seed}|{task['id']}")
    return rng.choice((600.0, 1200.0)), rng.choice((600.0, 1200.0)), rng.choice((1.0, 0.5)), rng.choice((1.0, 0.5))


def crawl(sim, task):
    """A fifth of the tasks crawl (speed 1/20, loading + 2 h): long-overdue vehicles mixed with punctual ones."""
    rng = random.Random(f"crawl|{sim.seed}|{task['id']}")
    if rng.random() < 0.2:
        return sim.p["load_s"] + 7200.0, sim.p["unload_s"], 0.05, 0.05
    return sim.p["load_s"], sim.p["unload_s"], 1.0, 1.0


ADV_TIMES = {"nominal": None, "fast": fast, "mixed": mixed, "grid": grid, "crawl": crawl}


# ---------------------------------------------------------------- adversarial instances
def wide_stops_net(W):
    import helpers as TO
    m = TO.make_map(dict(A=(0, 0), n1=(20, 0), n2=(420, 0), B=(440, 0), C=(20, 40), D=(420, 40)),
                    [("A", "n1"), ("n1", "n2"), ("n2", "B"), ("n1", "C"), ("n2", "D")],
                    {"fA": ("A", "assembly_workshop"), "fB": ("B", "paint_workshop"), "fC": ("C", "outfitting_workshop"),
                     "fD": ("D", "assembly_workshop")},
                    width={"r0": 31.0, "r1": W, "r2": 31.0, "r3": 31.0, "r4": 31.0}, zones={"zbuf": ("buffer", "r1", (120.0, 260.0))})
    return TO.load_net(m)


def adv_case(name, seed):
    import helpers as TO
    rng = random.Random(f"{name}|{seed}")
    if name.startswith("wide"):
        net = wide_stops_net(float(name[4:]))
        stops, K = sorted(net.stops), 7
        tasks = [dict(TO.task(i, *rng.sample(stops, 2), release=600.0 * rng.randint(0, 40)), B=float(rng.choice((5, 5, 6, 8)))) for i in range(45)]
    elif name.startswith("rnd"):
        net = TO.random_net(int(name[3:]))
        stops, K = sorted(net.stops), 10
        tasks = [dict(TO.task(i, *rng.sample(stops, 2), release=rng.uniform(0, 20000)), B=float(rng.randint(5, 15))) for i in range(60)]
    elif name.startswith("ring"):
        n = int(name[4:])
        net = TO.ring(n)
        stops, K = sorted(net.stops), n + 2
        tasks = [TO.task(i, *rng.sample(stops, 2), release=300.0 * rng.randint(0, 30)) for i in range(5 * n)]
    else:
        net = TO.sectioned_corridor()[1]
        stops, K = sorted(net.stops), 5
        tasks = [dict(TO.task(i, *rng.sample(stops, 2), release=600.0 * rng.randint(0, 30)), B=float(rng.choice((5, 8)))) for i in range(30)]
    locs = [net.stops[s]["node"] for s in (rng.choice(stops) for _ in range(K))]
    return net, P.scenario("main", days=1, warmup_days=0), tasks, locs


ADV = ["wide17", "wide19", "wide23", "ring3", "ring5", "sectioned"] + [f"rnd{i}" for i in range(1, 9)]


def crane_net(p, W):
    """Crane instance: A -(40 m)- n1 -(400 m, W wide, two docks along it)- n2 -(40 m)- B, a yard on the spur n2 - Y (12 m)
    and an outfitting shop at C. The network is built from the synthetic map (occupancy.mapgraph.to_graph)."""
    import helpers as TO
    from occupancy.mapgraph import to_graph
    from occupancy.network import Net
    m = TO.make_map(dict(A=(0, 0), n1=(40, 0), n2=(440, 0), B=(480, 0), Y=(440, 80), C=(40, -60)),
                    [("A", "n1"), ("n1", "n2"), ("n2", "B"), ("n2", "Y"), ("n1", "C")],
                    {"fA": ("A", "assembly_workshop"), "fB": ("B", "paint_workshop"), "fC": ("C", "outfitting_workshop")},
                    width={"r0": 31.0, "r1": W, "r2": 31.0, "r3": 12.0, "r4": 31.0},
                    zones={"zdock1": ("dry_dock", "r1", ()), "zdock2": ("dry_dock", "r1", ()), "zyard": ("yard", "r3", (30.0,))})
    m["zones"]["zdock1"]["boundary"] = dict(outer=[[120, 5], [200, 5], [200, 60], [120, 60]])
    m["zones"]["zdock2"]["boundary"] = dict(outer=[[300, 5], [380, 5], [380, 60], [300, 60]])
    return Net(to_graph(m), p)


def crane_case(name, seed):
    """Crane instances; names starting with lift also mark a third of the P6 as lifts inside the crane span (holding only
    the crane)."""
    import helpers as TO
    base = name[4:] if name.startswith("lift") else name
    bay, W, nz = base.endswith("bay"), (9.0 if "9" in base else 15.0), (2 if "n2" in base else 1)
    over = dict(days=1, warmup_days=0, crane_n=nz, crane_cycle_s=1800.0, crane_hook_s=P.CRANE["crane_hook_s"])
    if bay:
        over["retrofit"] = [dict(kind="dock_bay", cap=1)]
    p = P.scenario("main", **over)
    net = crane_net(p, W)
    rng = random.Random(f"{base}|{seed}")
    docks = [s for s in net.stops if s.startswith("zdock")]
    others = [s for s in net.stops if not s.startswith("zdock")]
    tasks = []
    for i in range(36):
        if rng.random() < 0.5:
            t = dict(TO.task(i, rng.choice(others), rng.choice(docks), release=300.0 * rng.randint(0, 50)), flow="P6", B=float(rng.choice((5, 8))))
        else:
            t = dict(TO.task(i, *rng.sample(others + docks[:1], 2), release=300.0 * rng.randint(0, 50)), B=float(rng.choice((5, 8))))
        tasks.append(t)
    locs = [net.stops[rng.choice(others)]["node"] for _ in range(6)]
    if name.startswith("lift"):
        for t in [t for t in tasks if t["flow"] == "P6"][::3]:
            t["lift"] = True
    return net, p, tasks, locs


CRANE = ["crane9", "crane15", "crane15n2", "crane9bay", "crane15n2bay"]
LIFT = ["lift15", "lift15n2", "lift9bay"]


# ---------------------------------------------------------------- one run
def run(net, p, tasks, locs, seed, times, rule, cmp_nominal):
    if rule == "adm_proj_mean":
        assert times is not None and not cmp_nominal
        p = dict(p, stoch=MEAN_BOOK)
    sim = Audited(net, p, len(locs), seed, tasks=[dict(t) for t in tasks], times=times, audit=True, **RULES[rule])
    for v, loc in zip(sim.vs, locs):
        v.loc = loc
    sim.run()
    out = dict(sim.summary(), lifts=len(sim.lifts), crane_bk=sum(b.res[0] == "c" for b in sim.log_bk),
               bay_bk=sum(b.res[0] == "b" for b in sim.log_bk), over_cap=independent_checks(sim))
    if cmp_nominal:
        ref = S.Sim(net, p, "reserve", len(locs), seed, None, tasks=[dict(t) for t in tasks])
        for v, loc in zip(ref.vs, locs):
            v.loc = loc
        ref.run()
        a = sorted((r["id"], r["done"]) for r in sim.done if r["done"] <= sim.H)
        b = sorted((r["id"], r["done"]) for r in ref.done)
        la = sorted((r["id"], r["start"]) for r in sim.lifts if r["start"] <= sim.H)
        lb = sorted((r["id"], r["start"]) for r in ref.lifts)
        same = lambda x, y: len(x) == len(y) and all(u[0] == w[0] and abs(u[1] - w[1]) < 1e-6 for u, w in zip(x, y))
        out.update(nominal_late_max_s=sim.late_max, same_as_reserve=same(a, b) and same(la, lb), dispatch_tie=False)
        if not out["same_as_reserve"]:
            # Dispatch ties: when two vehicles of reserve complete at exactly the same time, the queued tasks
            # go to them in the
            # order the events are processed, which may differ between the two implementations (reserve in the
            # order pushed at
            # dispatch, execution in booked order in the order of the arrival events at that time). The completion events of
            # reserve at equal times are replayed in the order of execution in booked order; then all must agree one by one
            rank = {ch["task"]["id"]: i for i, ch in enumerate(sim.fin_order)}
            rep = Replay(net, p, "reserve", len(locs), seed, None, tasks=[dict(t) for t in tasks], rank=rank)
            for v, loc in zip(rep.vs, locs):
                v.loc = loc
            rep.run()
            b2 = sorted((r["id"], r["done"]) for r in rep.done)
            lb2 = sorted((r["id"], r["start"]) for r in rep.lifts)
            out["dispatch_tie"] = same(a, b2) and same(la, lb2)
        assert sim.late_max < 1e-6 and (out["same_as_reserve"] or out["dispatch_tie"]), ("P4", sim.late_max)
    return out


class Replay(S.Sim):
    """reserve, with the completion events at equal times processed after the other events of that time, in the completion
    order of execution in booked order (rank: task id -> place)."""

    def __init__(self, *a, rank, **k):
        self.rank = rank
        super().__init__(*a, **k)

    def push(self, t, kind, x):
        self.seq += 1
        key = (1, self.rank.get(x[1]["id"], len(self.rank)), self.seq) if kind == "done" else (0, 0, self.seq)
        heapq.heappush(self.ev, (t, key, kind, x))


class Audited(Q.OPE2):
    """OPE2 that also records the completion order of the vehicle chains (for the tie replay)."""

    def __init__(self, *a, **k):
        self.fin_order = []
        super().__init__(*a, **k)

    def on_fin(self, t, ch):
        if not ch.get("lift"):
            self.fin_order.append(ch)
        return super().on_fin(t, ch)


def base_job(name, seed, times, rule):
    """One row of the self-check of ordered.py (tasks and control of ordered.job, executor replaced by the final code OPE2)."""
    cs, _ = O.cases()
    net, tasks, locs = cs[name]
    p = P.scenario("main", days=1, warmup_days=0)
    if name.startswith("random"):
        rng = random.Random(1000 * seed + int(name[6:]))
        stops = sorted(net.stops)
        tasks = [dict(t, o=a, d=b, release=rng.uniform(0, 30000)) for t in tasks for a, b in [rng.sample(stops, 2)]]
    out = run(net, p, tasks, locs, seed, O.TIMES[times], rule, times == "nominal")
    if times == "stoch_cv6" and rule == "adm":
        import stochastic
        q = dict(p, stoch=dict(mean=1.5, cv=0.6, speed=[0.9, 1.0], book_q=None))
        rb = stochastic.Sim(net, q, "reserve_rb", len(locs), seed, None, tasks=[dict(t) for t in tasks])
        for v, loc in zip(rb.vs, locs):
            v.loc = loc
        rb.run()
        out.update(rb_deadlocks=len(rb.deadlocks), rb_done=len(rb.done))
    return out


def job(args):
    suite, name, seed, times, rule = args
    t0 = time.time()
    out = dict(suite=suite, case=name, seed=seed, times=times, rule=rule)
    try:
        if suite == "base":
            out.update(base_job(name, seed, times, rule))
        else:
            net, p, tasks, locs = (adv_case if suite == "adv" else crane_case)(name, seed)
            out.update(run(net, p, tasks, locs, seed, ADV_TIMES[times], rule, times == "nominal"))
        out["error"] = ""
    except AssertionError as e:
        out["error"] = repr(e)[:300]
    out["secs"] = round(time.time() - t0, 2)
    return out


def code_sha():
    h = hashlib.sha256()
    for f in sorted((ROOT / "occupancy").glob("*.py")) + [ROOT / "experiments" / n for n in ("ordered.py", "ordered_proj.py", "ordered_check.py", "stochastic.py")]:
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def compare_base(rows, base_dir):
    """The adm / rel rows of the base suite compared row by row with an earlier run of ordered.py (its columns, except the
    time used); base_dir: the output directory of that run."""
    old = {(r["case"], r["seed"], r["times"], r["rule"]): r for r in csv.DictReader(open(Path(base_dir) / "cases.csv", encoding="utf-8-sig"))}
    n = same = 0
    diff = []
    for r in rows:
        if r["suite"] != "base" or r["rule"] not in ("adm", "rel"):
            continue
        o = old.get((r["case"], str(r["seed"]), r["times"], r["rule"]))
        n += 1
        if o and all(str(r.get(k, "")) == v for k, v in o.items() if k not in ("secs", "case", "seed", "times", "rule")):
            same += 1
        elif len(diff) < 10:
            diff.append((r["case"], r["seed"], r["times"], r["rule"]))
    return dict(rows=n, same=same, old_rows=len(old), first_diff=diff)


def main():
    ap = argparse.ArgumentParser(description="Extended self-check of execution in booked order.")
    ap.add_argument("--out", required=True, help="output directory (cases.csv, summary.json, tables.md)")
    ap.add_argument("--slots", default=os.environ.get("CPU_SLOTS"), help="CPU-slot directory (optional; default $CPU_SLOTS)")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--base", default=None, help="output directory of a run of ordered.py to compare the base suite with")
    a = ap.parse_args()
    n, nc = a.seeds, a.seeds // 2
    todo = [("base", c, s, tm, r) for c in O.cases()[0] for s in range(1, n + 1) for tm in O.TIMES for r in ("adm", "rel")]
    ok = lambda tm, r: not (r == "adm_proj_mean" and tm == "nominal")
    todo += [("base", c, s, tm, r) for c in O.cases()[0] for s in range(1, nc + 1) for tm in O.TIMES for r in ("adm_proj", "adm_proj_mean") if ok(tm, r)]
    todo += [("adv", c, s, tm, r) for c in ADV for s in range(1, n + 1) for tm in ADV_TIMES for r in RULES if ok(tm, r)]
    todo += [("crane", c, s, tm, r) for c in CRANE + LIFT for s in range(1, nc + 1) for tm in ADV_TIMES for r in RULES if ok(tm, r)]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = O.pool_map(job, todo, a.procs, a.slots, "ordered_check")
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "cases.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    agg = collections.defaultdict(collections.Counter)
    for r in rows:
        g = agg[(r["suite"], r["times"], r["rule"])]
        g["runs"] += 1
        g["errors"] += bool(r["error"])
        for k in ("tasks", "wait_order", "wait_cap", "wait_rel", "deadlock", "lifts", "crane_bk", "bay_bk", "rb_deadlocks"):
            g[k] += r.get(k, 0) or 0
        g["nominal_same"] += bool(r.get("same_as_reserve"))
        g["nominal_tie_diff"] += bool(r.get("dispatch_tie")) and not r.get("same_as_reserve") and r["times"] == "nominal"
        g["over_cap_max"] = max(g["over_cap_max"], r.get("over_cap", 0) or 0)
    tot = collections.Counter()
    for g in agg.values():
        tot.update({k: v for k, v in g.items() if k != "over_cap_max"})
    summ = dict(code_sha=code_sha(), totals=dict(tot), over_cap_max=max((g["over_cap_max"] for g in agg.values()), default=0.0),
                base_vs_ordered=compare_base(rows, a.base) if a.base else None, groups={"|".join(k): dict(v) for k, v in sorted(agg.items())},
                errors=[r for r in rows if r["error"]][:20])
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    L = ["| suite | times | rule | runs | tasks | order waits | capacity waits | release waits | deadlocks | assertion failures | "
         "lifts | crane bookings | bay bookings | nominal same as reserve |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (s, tm, r), g in sorted(agg.items()):
        tie = f" (plus {g['nominal_tie_diff']} dispatch ties)" if g["nominal_tie_diff"] else ""
        nom = f"{g['nominal_same']}{tie}" if tm == "nominal" else "-"
        L.append(f"| {s} | {tm} | {r} | {g['runs']} | {g['tasks']} | {g['wait_order']} | {g['wait_cap']} | {g['wait_rel']} | {g['deadlock']} | "
                 f"{g['errors']} | {g['lifts']} | {g['crane_bk']} | {g['bay_bk']} | {nom} |")
    (out / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps({k: v for k, v in summ.items() if k not in ("groups",)}, ensure_ascii=False))
    print(out)


if __name__ == "__main__":
    main()

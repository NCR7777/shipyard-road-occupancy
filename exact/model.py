"""Sub-network instances, full-path reservation (simulation), the optimal schedule that waits only at the origin (CP-SAT) and
the bounds on the same instance.

Finite window: the same batch of N tasks is released at time 0 in the release order of the simulation (saturated), the K
vehicles start at the same positions; all three (full-path reservation, the optimal schedule, the bounds) complete all N tasks,
the window is the respective completion time W and the throughput is lambda_W = N·H / W (H = length of the working day). The
completed set of tasks is the same; there is no room to pick tasks.

Windows and occupancy are always taken from the simulation (Sim.windows -> Net.leg), not written a second time:
  empty run (relative to the empty departure t1), origin stop resource [t1 + ao, t2 + tail], loaded run and crane (relative to
  the loaded departure t2), destination [t2 + ad, t2 + TL + unloading].
The only difference from full-path reservation: the destination is held only until unloading is done (as in the bounds and
the theorem), without the (L0 + s)/v0 = 6.4 s in which the empty vehicle drives off after unloading. So every schedule of
full-path reservation is a feasible solution here (it only holds these 6.4 s more), and conversely every schedule here
satisfies Definition 1 (admissible) and can be executed by a rule that waits only at the origin.
Rounding of time (the unit of solve; 0.1 s in the batches, 1 s for the small test instances): window starts are rounded down,
ends up, the durations empty run + loading and loaded run + unloading up -- an integer solution is always feasible in
continuous time (conservative).
"""
import collections
import copy
import math
import random
from fractions import Fraction

import numpy as np
from ortools.sat.python import cp_model
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, vstack

from bounds.system import stop_ret
from occupancy.bound import _system, bounds, share, task_windows
from occupancy.network import EPS
from occupancy.sim import Sim, Vehicle
from occupancy.tasks import TaskStream


# ---------------------------------------------------------------- sub-network and instance
def ball_roads(net, seeds, R):
    """Roads (road ids of the network) whose two ends can be reached from the seed nodes within a road distance <= R."""
    import heapq
    dist = {n: 0.0 for n in seeds}
    heap = [(0.0, n) for n in sorted(seeds)]
    while heap:
        d, n = heapq.heappop(heap)
        if d > dist[n]:
            continue
        for i in net.out[n]:
            h, L = net.arc[i][1], net.arc[i][2]
            if d + L <= R and d + L < dist.get(h, math.inf):
                dist[h] = d + L
                heapq.heappush(heap, (d + L, h))
    return {a[3] for a in net.arc if a[0] in dist and a[1] in dist}


def sub_stops(net, roads):
    nodes = {x for a in net.arc if a[3] in roads for x in a[:2]}
    return {sid for sid, st in net.stops.items() if st["node"] in nodes}


def path_roads(net, src, dst):
    return {net.arc[a][3] for a in (net.path(src, dst) if src != dst else [])}


def sample_tasks(net, p, seed, roads, n, ents=None, via=None, max_draws=10 ** 7):
    """From the task stream of the simulation (TaskStream, same seed), the first n tasks whose origin and destination are in
    the sub-network, whose loaded shortest path lies entirely in the sub-network and that are not lifts by a gantry crane
    (the same release stream conditioned on the sub-network); renumbered 0...n-1, all released at time 0. With ents only the
    tasks whose origin or destination belongs to these entities; with via only the tasks whose loaded shortest path passes
    one of these roads."""
    stops, ts, out = sub_stops(net, roads), TaskStream(net, seed, p), []
    for _ in range(max_draws):
        t = ts.next(0.0)
        if ents and not {net.stops[t["o"]]["ent"], net.stops[t["d"]]["ent"]} & set(ents):
            continue
        if t["o"] not in stops or t["d"] not in stops:
            continue
        pr = path_roads(net, net.stops[t["o"]]["node"], net.stops[t["d"]]["node"])
        if via and not pr & set(via):
            continue
        if not t.get("lift") and pr <= roads:
            out.append(dict(t, id=len(out), src_id=t["id"], release=0.0))
            if len(out) == n:
                return out
    raise RuntimeError(f"cannot draw {n} tasks in the sub-network (got {len(out)})")


def vehicle_starts(net, tasks, seed, K):
    """Start positions of the K vehicles: drawn from the origin and destination stop nodes of the instance (random stream
    seed·100 + 13, the seed rule of tasks.vehicle_starts)."""
    pool = sorted({net.stops[t[k]]["node"] for t in tasks for k in ("o", "d")})
    rng = random.Random(seed * 100 + 13)
    return [rng.choice(pool) for _ in range(K)]


def close_roads(net, tasks, starts, roads):
    """Empty-run closure: the roads on the shortest paths from any destination or start position to any origin join the
    sub-network. Returns (set of roads, roads added)."""
    o = {net.stops[t["o"]]["node"] for t in tasks}
    src = {net.stops[t["d"]]["node"] for t in tasks} | set(starts)
    need = set().union(*(path_roads(net, a, b) for a in src for b in o))
    return roads | need, need - roads


class Instance:
    """Sub-network instance: net (whole yard or synthetic network), p, tasks (released at time 0, in release order), start
    positions, roads of the sub-network."""

    def __init__(self, name, net, p, tasks, starts, roads):
        self.name, self.net, self.p, self.tasks, self.starts = name, net, p, tasks, starts
        self.roads, self.added = close_roads(net, tasks, starts, roads)
        self.keep = [i for i, a in enumerate(net.arc) if a[3] in self.roads]
        self.cap, self.size = net.capacities(p)
        self.sim = Sim(net, p, "reserve", 1, 1, None, tasks=[])        # only its windows(): the same windows as full-path reservation
        self._chain = {}

    # ------------------------------------------------------------ windows (from Sim.windows)
    def chain(self, j, loc):
        """Task j with the vehicle departing from loc: dict(TE, TL, ao, oend, WE, WL, so, d_so). WE: windows of the empty run
        (relative to t1); WL: loaded run, crane and destination (relative to t2, the destination only until unloading is
        done); origin window [t1 + ao, t2 + oend], merged with the windows of the loaded run on the origin stop resource that
        adjoin it. Windows on the same resource are merged with the piecewise maximum width."""
        key = (j, loc)
        if key in self._chain:
            return self._chain[key]
        t, p = self.tasks[j], self.p
        so, sd, d_so, d_sd, TE, TL, WE, WL, ao, ad, tail_l, clr = self.sim.windows(Vehicle(-1, loc), t)
        WL = merge(WL + [(sd["res"], ad, TL + p["unload_s"], d_sd)])
        oend, rest = tail_l, []
        for w in sorted(WL, key=lambda w: w[1]):
            if w[0] == so["res"] and w[1] <= oend + EPS:
                assert w[3] <= d_so + EPS, (self.name, j, w, d_so)
                oend = max(oend, w[2])
            else:
                rest.append(w)
        WE = merge(WE)
        assert all(w[0] != so["res"] for w in WE), (self.name, j, loc)
        c = self._chain[key] = dict(TE=TE, TL=TL, ao=ao, oend=oend, WE=WE, WL=rest, so=so["res"], d_so=d_so)
        return c

    def locs(self, j, K):
        """Possible departure positions of the empty run of task j: the destinations of the other tasks and the start
        positions of the first K vehicles."""
        return sorted({self.net.stops[t["d"]]["node"] for i, t in enumerate(self.tasks) if i != j} | set(self.starts[:K]))

    def dur_l(self, j):
        return self.chain(j, self.net.stops[self.tasks[j]["o"]]["node"])["TL"] + self.p["unload_s"]


def merge(ws):
    """[(resource, start, end, width)] -> windows on the same resource that overlap or adjoin are merged (the widths must be
    equal; the windows within a task belong to one vehicle with one load and are equal by construction)."""
    by = collections.defaultdict(list)
    for r, s, e, d in ws:
        if e > s and d > EPS:
            by[r].append([s, e, d])
    out = []
    for r, iv in by.items():
        iv.sort()
        cur = iv[0]
        for s, e, d in iv[1:]:
            if s <= cur[1] + EPS and r[0] != "c":
                assert abs(d - cur[2]) <= EPS, (r, cur, (s, e, d))
                cur[1] = max(cur[1], e)
            else:
                out.append((r, *cur))
                cur = [s, e, d]
        out.append((r, *cur))
    return out


# ---------------------------------------------------------------- full-path reservation (simulation)
class FPR(Sim):
    """Full-path reservation (reserve, just-in-time departure) that records one more item: the vehicle that does each task
    (plan["vid"])."""

    def plan(self, v, task, t0):
        super().plan(v, task, t0)
        task["plan"]["vid"] = v.id


def run_fpr(inst, K):
    """Full-path reservation on the same batch of tasks and the same start positions. Returns (completion time, task sequence
    of each vehicle {vehicle: [(task, t1, t2)]}, sim)."""
    p = dict(inst.p, days=400)
    sim = FPR(inst.net, p, "reserve", K, 1, None, record=True, tasks=[dict(t) for t in inst.tasks])
    for v, loc in zip(sim.vs, inst.starts[:K]):
        v.loc = loc
    sim.run()
    assert len(sim.done) == len(inst.tasks), (inst.name, K, len(sim.done))
    seq = collections.defaultdict(list)
    for r in sim.done:
        seq[r["plan"]["vid"]].append((r["id"], r["plan"]["t1"], r["plan"]["t2"]))
    return max(r["done"] for r in sim.done), {k: sorted(v, key=lambda x: x[1]) for k, v in seq.items()}, sim


# ---------------------------------------------------------------- CP-SAT
def scale_of(xs):
    """Common integer multiplier of the widths and capacities (exact)."""
    s = 1
    for x in xs:
        f = Fraction(x).limit_denominator(1000)
        assert abs(float(f) - x) < 1e-9, x
        s = s * f.denominator // math.gcd(s, f.denominator)
    return s


def solve(inst, K, time_s, workers, hint=None, fix=None, lb=None, horizon=None, seed=0, unit=0.1, log=False, local=None):
    """Optimal schedule that waits only at the origin: minimises the completion time of all N tasks. Time is rounded to unit
    seconds (window starts down, ends up, durations up: an integer solution is always feasible in continuous time).
    hint / fix: {vehicle: [(task, t1, t2)]} (seconds), as a starting hint / as fixed dispatch and order (CP-SAT then only sets
    the times; used to check that full-path reservation is feasible in the model).
    lb: lower bound of the completion time (seconds; finite-window linear programme or occupancy-order bound, valid for every
    feasible schedule in continuous time, and so after rounding). horizon: seconds.
    local (set of resources): local relaxation for a lower bound of the completion time -- only the capacity constraints on
    these resources and "tasks in progress at the same time <= K" are kept, without dispatch and order (the empty run of each
    task may start from any possible position); time is rounded inwards (window starts up plus one unit, ends down, durations
    down), so every feasible schedule in continuous time maps to a feasible solution of it, and its optimum (or the solver's
    bound) x unit does not exceed the optimal completion time in continuous time.
    Returns dict(status, M (completion time, seconds, rounded model), bound (seconds), makespan (continuous time), seq
    (seconds), wall)."""
    net, p, T = inst.net, inst.p, inst.tasks
    N = len(T)
    load = p["load_s"]
    if local is None:
        ce = lambda x: math.ceil(x / unit - 1e-7)
        fl = lambda x: math.floor(x / unit + 1e-7)
        ws = fl
    else:                                   # inwards: durations floor - 1, windows [ceil(a) + 1, floor(b)]
        ce = lambda x: math.floor(x / unit + 1e-7)
        fl = lambda x: math.ceil(x / unit - 1e-7) + 1
        ws = lambda x: math.floor(x / unit + 1e-7) - 1
    H = ce(horizon) + 1
    m = cp_model.CpModel()
    t1 = [m.new_int_var(math.floor(T[j]["release"] / unit + 1e-7), H, f"t1_{j}") for j in range(N)]
    t2 = [m.new_int_var(0, H, f"t2_{j}") for j in range(N)]
    DL = [ce(inst.dur_l(j)) for j in range(N)]
    m.add_cumulative([m.new_interval_var(t1[j], m.new_int_var(0, H, ""), t2[j] + DL[j], "") for j in range(N)], [1] * N, K)   # redundant: tasks in progress at the same time <= K
    dnode = [net.stops[t["d"]]["node"] for t in T]
    if local is not None:
        return _relaxed(inst, K, m, t1, t2, DL, ws, fl, ce, H, local, lb, time_s, workers, seed, unit, log)
    x = {(i, j): m.new_bool_var(f"x{i}_{j}") for i in range(N) for j in range(N) if i != j}
    s = {(k, j): m.new_bool_var(f"s{k}_{j}") for k in range(K) for j in range(N)}
    for j in range(N):
        m.add_exactly_one([x[i, j] for i in range(N) if i != j] + [s[k, j] for k in range(K)])
        m.add_at_most_one([x[j, i] for i in range(N) if i != j])
        for i in range(N):
            if i != j:
                m.add(t1[j] >= t2[i] + DL[i]).only_enforce_if(x[i, j])
    used = []
    for k in range(K):
        m.add_at_most_one([s[k, j] for j in range(N)])
        used.append(sum(s[k, j] for j in range(N)))
    for k in range(K):               # vehicles at the same start position are interchangeable: used in order of their number, first task numbers increasing
        for k2 in range(k + 1, K):
            if inst.starts[k] == inst.starts[k2] and not fix:
                m.add(used[k] >= used[k2])
                b = m.new_bool_var("")
                m.add(used[k2] == 1).only_enforce_if(b)
                m.add(used[k2] == 0).only_enforce_if(~b)
                m.add(sum(j * s[k, j] for j in range(N)) < sum(j * s[k2, j] for j in range(N))).only_enforce_if(b)
    by = collections.defaultdict(list)          # resource -> [(interval, width)]
    for j in range(N):
        os_ = m.new_int_var(0, H, f"os_{j}")
        src = collections.defaultdict(list)     # departure position -> arcs into task j
        for i in range(N):
            if i != j:
                src[dnode[i]].append(x[i, j])
        for k in range(K):
            src[inst.starts[k]].append(s[k, j])
        oend = None
        for l, arcs in sorted(src.items()):
            y = m.new_bool_var(f"y{j}_{l[-6:]}")
            m.add(y == sum(arcs))
            c = inst.chain(j, l)
            oend = c["oend"]
            m.add(t2[j] >= t1[j] + ce(c["TE"] + load)).only_enforce_if(y)
            m.add(os_ == t1[j] + fl(c["ao"])).only_enforce_if(y)
            for r, a, b, d in c["WE"]:
                by[r].append((m.new_optional_fixed_size_interval_var(t1[j] + fl(a), ce(b) - fl(a), y, ""), d))
        c = inst.chain(j, net.stops[T[j]["o"]]["node"])
        by[c["so"]].append((m.new_interval_var(os_, m.new_int_var(0, H, ""), t2[j] + ce(c["oend"]), f"o{j}"), c["d_so"]))
        for r, a, b, d in c["WL"]:
            by[r].append((m.new_fixed_size_interval_var(t2[j] + fl(a), ce(b) - fl(a), ""), d))
    S = scale_of([inst.cap[r] for r in by] + [d for iv in by.values() for _, d in iv])
    for r, iv in by.items():
        C = inst.cap[r]
        if sum(d for _, d in iv) <= C + EPS:
            continue
        m.add_cumulative([a for a, _ in iv], [round(d * S) for _, d in iv], round(C * S))
    M = m.new_int_var(0, H + max(DL), "M")
    for j in range(N):
        m.add(M >= t2[j] + DL[j])
    if lb is not None:
        m.add(M >= ce(lb - 1e-6))
    m.minimize(M)
    route = fix or (hint and canon(hint, inst.starts[:K]))
    if route:
        prev = {}
        for k, lst in route.items():
            for n, (j, a, b) in enumerate(lst):
                prev[j] = ("s", k) if n == 0 else ("x", lst[n - 1][0])
                m.add_hint(t1[j], min(H, ce(a)))
                m.add_hint(t2[j], min(H, ce(b)))
        for (i, j), v in x.items():
            val = prev.get(j) == ("x", i)
            (m.add(v == val) if fix else m.add_hint(v, val))
        for (k, j), v in s.items():
            val = prev.get(j) == ("s", k)
            (m.add(v == val) if fix else m.add_hint(v, val))
    sol = cp_model.CpSolver()
    sol.parameters.max_time_in_seconds = time_s
    sol.parameters.num_workers = workers
    sol.parameters.random_seed = seed
    sol.parameters.log_search_progress = log
    st = sol.solve(m)
    out = dict(status=sol.status_name(st), wall=sol.wall_time, bound=sol.best_objective_bound * unit, scale=S, unit=unit,
               n_bool=len(x) + len(s), n_res=len(by))
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return out
    seq = {}
    for k in range(K):
        j = next((j for j in range(N) if sol.value(s[k, j])), None)
        lst = []
        while j is not None:
            lst.append((j, sol.value(t1[j]) * unit, sol.value(t2[j]) * unit))
            j = next((i for i in range(N) if i != j and sol.value(x[j, i])), None)
        if lst:
            seq[k] = lst
    assert sum(len(v) for v in seq.values()) == N
    out.update(M=sol.value(M) * unit, seq=seq, makespan=max(b + inst.dur_l(j) for v in seq.values() for j, a, b in v))
    return out


def _relaxed(inst, K, m, t1, t2, DL, ws, fl, ce, H, local, lb, time_s, workers, seed, unit, log):
    """The local relaxation of solve (see local in solve). Durations ws (down), window starts fl (up + 1), ends ce (down)."""
    net, T, load = inst.net, inst.tasks, inst.p["load_s"]
    N = len(T)
    by = collections.defaultdict(list)
    for j in range(N):
        os_ = m.new_int_var(0, H, f"os_{j}")
        ys = []
        for l in inst.locs(j, K):
            y = m.new_bool_var("")
            ys.append(y)
            c = inst.chain(j, l)
            m.add(t2[j] >= t1[j] + ws(c["TE"] + load) + 1).only_enforce_if(y)     # floor(t2/u) - floor(t1/u) >= floor(D/u)
            m.add(os_ == t1[j] + fl(c["ao"])).only_enforce_if(y)
            for r, a, b, d in c["WE"]:
                if r in local and ce(b) > fl(a):
                    by[r].append((m.new_optional_fixed_size_interval_var(t1[j] + fl(a), ce(b) - fl(a), y, ""), d))
        m.add_exactly_one(ys)
        c = inst.chain(j, net.stops[T[j]["o"]]["node"])
        if c["so"] in local:
            by[c["so"]].append((m.new_interval_var(os_, m.new_int_var(0, H, ""), t2[j] + ce(c["oend"]), ""), c["d_so"]))
        for r, a, b, d in c["WL"]:
            if r in local and ce(b) > fl(a):
                by[r].append((m.new_fixed_size_interval_var(t2[j] + fl(a), ce(b) - fl(a), ""), d))
    S = scale_of([inst.cap[r] for r in by] + [d for iv in by.values() for _, d in iv])
    for r, iv in by.items():
        if sum(d for _, d in iv) > inst.cap[r] + EPS:
            m.add_cumulative([a for a, _ in iv], [round(d * S) for _, d in iv], round(inst.cap[r] * S))
    M = m.new_int_var(0, H + max(DL), "M")
    for j in range(N):
        m.add(M >= t2[j] + DL[j])
    if lb is not None:
        m.add(M >= math.floor(lb / unit + 1e-7) - 1)
    m.minimize(M)
    sol = cp_model.CpSolver()
    sol.parameters.max_time_in_seconds = time_s
    sol.parameters.num_workers = workers
    sol.parameters.random_seed = seed
    sol.parameters.log_search_progress = log
    st = sol.solve(m)
    return dict(status=sol.status_name(st), wall=sol.wall_time, bound=sol.best_objective_bound * unit,
                M=sol.value(M) * unit if st in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None, n_res=len(by))


def near(inst, res, depth):
    """Neighbourhood of resource res: the resources that share a node with it (both ends of a road, junctions, the stops on a
    road and the cranes of a dock), extended ring by ring to depth rings; only those in the sub-network."""
    net = inst.net
    ends = lambda rid: {net.road[rid]["a"], net.road[rid]["b"]}
    nodes = {}
    for r in inst.cap:
        if r[0] == "r" and r[1] in inst.roads:
            nodes[r] = ends(r[1])
        elif r[0] == "n":
            nodes[r] = {r[1]}
    for st in net.stops.values():
        if st["res"][0] == "b" and st["node"].startswith("S|"):
            nodes[st["res"]] = ends(st["node"].split("|")[1])
    for z in net.by_cls.get("dry_dock", []):
        if ("c", z) in inst.cap:
            nodes[("c", z)] = set().union(*(ends(net.stops[s]["node"].split("|")[1]) for s in net.ents[z]["stops"]))
    out, ring = {res}, {res}
    for _ in range(depth):
        ns = set().union(*(nodes.get(r, set()) for r in ring))
        ring = {r for r, x in nodes.items() if x & ns} - out
        out |= ring
    return out


def rebook(inst, K, seq, unit=0.1):
    """Rebook a schedule of full-path reservation (continuous time) with the same dispatch and order, in the order of the
    empty departures, on the windows of the rounded model: each task starts from its original times (rounded up); where a
    window does not fit, the leg it belongs to (empty run and origin window -> t1, loaded run -> t2) is pushed one unit later
    until all fit. Gives a feasible solution of the rounded model (a hint for CP-SAT) whose completion time exceeds that of the
    original schedule only by the few seconds of rounding. Returns {vehicle: [(task, t1, t2)]} (seconds)."""
    p, net = inst.p, inst.net
    ce = lambda x: math.ceil(x / unit - 1e-7)
    fl = lambda x: math.floor(x / unit + 1e-7)
    placed = collections.defaultdict(list)

    def over(r, s, e, d):
        iv = [x for x in placed[r] if x[0] < e and x[1] > s]
        if sum(x[2] for x in iv) + d <= inst.cap[r] + EPS:
            return False
        return any(sum(x[2] for x in iv if x[0] <= t < x[1]) + d > inst.cap[r] + EPS for t in {s} | {x[0] for x in iv if x[0] > s})

    order = sorted(((a, k, n) for k, lst in seq.items() for n, (j, a, b) in enumerate(lst)))
    loc = {k: inst.starts[k] for k in seq}
    done, out = {k: 0 for k in seq}, {k: [] for k in seq}
    for _, k, n in order:
        j, a, b = seq[k][n]
        c = inst.chain(j, loc[k])
        dE = ce(c["TE"] + p["load_s"])
        t1 = max(ce(a), done[k], ce(inst.tasks[j]["release"]))
        t2 = max(ce(b), t1 + dE)
        while True:
            W1 = [(r, t1 + fl(x), t1 + ce(y), d) for r, x, y, d in c["WE"]] + [(c["so"], t1 + fl(c["ao"]), t2 + ce(c["oend"]), c["d_so"])]
            W2 = [(r, t2 + fl(x), t2 + ce(y), d) for r, x, y, d in c["WL"]]
            if any(over(*w) for w in W1):
                t1 += 1
                t2 = max(t2, t1 + dE)
            elif any(over(*w) for w in W2):
                t2 += 1
            else:
                break
        for r, s, e, d in W1 + W2:
            placed[r].append((s, e, d))
        done[k] = t2 + ce(inst.dur_l(j))
        out[k].append((j, t1 * unit, t2 * unit))
        loc[k] = net.stops[inst.tasks[j]["d"]]["node"]
    return out


def canon(seq, starts):
    """Reassign the sequences among the vehicles at the same start position so that they satisfy the symmetry constraints of
    solve (lower-numbered vehicles first, first task numbers increasing)."""
    out = {}
    for loc in set(starts):
        ks = [k for k, x in enumerate(starts) if x == loc]
        lst = sorted((seq[k] for k in ks if seq.get(k)), key=lambda v: v[0][0])
        out.update(zip(ks, lst))
    return out


# ---------------------------------------------------------------- bounds
def sub_net(inst):
    """View of the network with only the arcs of the sub-network (empty flows may only use the sub-network), for the linear
    programme; shortest paths and windows still follow the original network (net.leg is bound to the original object)."""
    sub = copy.copy(inst.net)
    sub.arc = [inst.net.arc[i] for i in inst.keep]
    sub.leg = inst.net.leg
    return sub


def ret_sub(inst):
    """Empty-run tail truncation of Eq. (5) (bounds.system.stop_ret, computed on the original network: smaller, that is looser,
    and still valid), arc numbers mapped to the sub-network."""
    p = inst.p
    full = stop_ret(inst.net, p["plat_len"] + p["long_clear"])
    return {n: full[i] for n, i in enumerate(inst.keep) if i in full}


def lam_window(inst, W):
    """lambda_W^N of Proposition 2: the mix of these N tasks completed in a window of length W put into the linear programme
    (truncation of Eq. (5); each erection counts in the crane rows as tau·W/(W + tau)), tasks/day. Written like
    occupancy.bound.bounds_window, plus the truncation ret. Returns (lambda_W^N, binding resources)."""
    p = inst.p
    if p.get("crane_n"):
        c = p["crane_cycle_s"]
        p = dict(p, crane_cycle_s=c * W / (W + c))
    b = bounds(sub_net(inst), p, inst.tasks, tight_only=True, ret=ret_sub(inst))
    return b["network_tight"], b["binding_tight"]


def lam_fleet(inst, K):
    """Fleet bound of Proposition 1, lambda^F = K·H / zeta (zeta = mean handling and loaded driving + least empty driving on the
    sub-network), tasks/day."""
    b = bounds(sub_net(inst), inst.p, inst.tasks, ret=ret_sub(inst))
    return K * inst.p["day_s"] / b["cycle_s"]


def lp_finite(inst, K, fleet=True):
    """Finite-window linear programme from an empty system, a lower bound of the completion time: the same occupancy rows
    (k = 0, 1, 2) and conservation of empty vehicles, with the boundary written as in the instance -- empty flows start from
    the start positions of the K vehicles and may end at any node; W is the completion time, the right-hand side of the crane
    rows is W + tau - t_U (a crane window starts at the start of unloading <= W - t_U).
    min W, s.t. N·a + sum e f <= W (crane rows + tau - t_U), conservation: in - out = N(o - d) - s + z, sum z = K, f, z >= 0;
    with fleet also the vehicle-time row of Proposition 1.
    For every schedule that waits only at the origin (or any schedule satisfying Assumptions 1-3) the completion time satisfies
    these rows (the proof of Theorem 1 holds directly on [0, W]: no vehicle is under way before the start, and all windows of
    the completed tasks lie within [0, W]). Returns the optimum of W (seconds)."""
    p, net = inst.p, sub_net(inst)
    N = len(inst.tasks)
    Aeq, Aub, res_list, abar, *_ = _system(net, p, inst.cap, inst.size, inst.tasks, True, None, ret_sub(inst))
    nodes = sorted({x for a in net.arc for x in a[:2]})
    ni = {n: i for i, n in enumerate(nodes)}
    A, nn, nr = Aub.shape[1] - 1, len(nodes), len(res_list)
    crane = np.array([res_list[i % nr][0] == "c" for i in range(Aub.shape[0])])
    # variables: [W, f (A), z (nn)]
    ub = hstack([csr_matrix(-np.ones((Aub.shape[0], 1))), Aub[:, 1:], csr_matrix((Aub.shape[0], nn))]).tocsr()
    b_ub = -N * np.asarray(Aub[:, 0].todense()).ravel() + np.where(crane, p.get("crane_cycle_s", 0.0) - p["unload_s"], 0.0)
    on = -np.asarray(Aeq[:, 0].todense()).ravel() * N                  # number of origins - number of destinations
    s = np.zeros(nn)
    for loc in inst.starts[:K]:
        s[ni[loc]] += 1
    eq = vstack([hstack([csr_matrix((nn, 1)), Aeq[:, 1:], csr_matrix(-np.eye(nn))]),
                 csr_matrix(np.concatenate([[0.0], np.zeros(A), np.ones(nn)]).reshape(1, -1))]).tocsr()
    b_eq = np.concatenate([on - s, [K]])
    if fleet:      # vehicle-time row of Proposition 1: sum(loading + loaded driving + unloading) + sum empty driving <= K·W (each vehicle does its tasks and empty runs one after another within [0, W])
        hand = sum(p["load_s"] + inst.dur_l(j) for j in range(N))
        lens = np.array([a[2] for a in net.arc]) / p["v_empty"]
        ub = vstack([ub, csr_matrix(np.concatenate([[-K], lens, np.zeros(nn)]).reshape(1, -1))]).tocsr()
        b_ub = np.concatenate([b_ub, [-hand]])
    c = np.zeros(1 + A + nn)
    c[0] = 1.0
    lp = linprog(c, A_ub=ub, b_ub=b_ub, A_eq=eq, b_eq=b_eq, bounds=[(0, None)] * len(c), method="highs")
    assert lp.status == 0, lp.message
    return float(lp.x[0])


def chain_bound(inst, K, res):
    """Lower bound of the completion time by the occupancy order when the binding resource is an origin stop resource (such as
    the door of a workshop): the part the linear programme does not see.
    These tasks each hold res only with their origin window, exclusively, so the windows line up in time. Between two
    consecutive windows, where the loaded windows of the earlier vehicle after it leaves the door and the empty-run windows of
    the later vehicle before it enters the door do not fit together on the same resource (such as the junction in front of
    the door), the two vehicles cannot hold that resource at the same time: from these two vehicles alone follows a gap
    gap(i, j) by which the later window starts at least after the earlier one (the smallest over all possible departure
    positions of the later vehicle). So W >= earliest start + sum of the shortest window lengths + sum_j min_i gap(i, j)
    - max_j min_i gap(i, j) + shortest time from leaving the door to the end of unloading.
    Uses only a part of these two kinds of constraints, so it holds for every feasible schedule (a relaxation). Returns None
    when the premise does not hold (res also carries empty-run or loaded windows, or is not exclusive)."""
    N, load = len(inst.tasks), inst.p["load_s"]
    J = [j for j in range(N) if inst.chain(j, inst.net.stops[inst.tasks[j]["o"]]["node"])["so"] == res]
    if len(J) < 2:
        return None
    for j in range(N):
        for l in inst.locs(j, K):
            c = inst.chain(j, l)
            if any(w[0] == res for w in c["WE"] + c["WL"]) or (j in J and 2 * c["d_so"] <= inst.cap[res] + EPS):
                return None
    ln, r0, gin = [], [], []
    for j in J:
        cs = {l: inst.chain(j, l) for l in inst.locs(j, K)}
        ln.append(min(c["TE"] - c["ao"] for c in cs.values()) + load + cs[next(iter(cs))]["oend"])
        r0.append(inst.tasks[j]["release"] + min(c["ao"] for c in cs.values()))
        best = math.inf
        for i in J:
            if i == j:
                continue
            ci = inst.chain(i, inst.net.stops[inst.tasks[i]["o"]]["node"])
            for c in cs.values():
                lo = ci["oend"] - c["ao"]                 # delta = t1_j - t2_i: the door (res) is exclusive
                bad = sorted((a1 - b2, a2 - b1) for r, a1, a2, d in ci["WL"] for r2, b1, b2, e in c["WE"]
                             if r == r2 and r[0] != "c" and d + e > inst.cap[r] + EPS)
                for a, b in bad:
                    if a < lo < b:
                        lo = b
                best = min(best, lo + c["ao"] - ci["oend"])
        gin.append(best)
    q = min(inst.dur_l(j) - inst.chain(j, inst.net.stops[inst.tasks[j]["o"]]["node"])["oend"] for j in J)
    return min(r0) + sum(ln) + sum(gin) - max(gin) + q


def beta(inst, K):
    """Boundary constant of Proposition 2, beta = K·max_{r,k}(a^(k),max_r + e^(k),max_r) + t_clr^max (for this instance: the
    tasks are these N, the empty runs any shortest path from a destination or start position to an origin, the clearing time
    the largest junction and road clearing time in the sub-network)."""
    p, net = inst.p, sub_net(inst)
    Aeq, Aub, res_list, *_ = _system(net, p, inst.cap, inst.size, inst.tasks[:1], True, None, ret_sub(inst))
    nr = len(res_list)
    ri = {r: i for i, r in enumerate(res_list)}
    amax = np.zeros(3 * nr)
    for t in inst.tasks:
        by, _ = task_windows(inst.net, p, inst.cap, inst.size, t, tight=True)
        for r, iv in by.items():
            for k in range(3):
                v = sum((b - a) * share(d, inst.cap[r], k) for a, b, d in _pieces_pos(iv))
                amax[k * nr + ri[r]] = max(amax[k * nr + ri[r]], v)
    pos = {i: n for n, i in enumerate(inst.keep)}
    E = Aub[:, 1:].tocsc()
    emax = np.zeros(3 * nr)
    o = {net.stops[t["o"]]["node"] for t in inst.tasks}
    src = {net.stops[t["d"]]["node"] for t in inst.tasks} | set(inst.starts[:K])
    for a in src:
        for b in o:
            if a != b:
                cols = [pos[i] for i in inst.net.path(a, b)]
                emax = np.maximum(emax, np.asarray(E[:, cols].sum(axis=1)).ravel())
    nodes = {x for i in inst.keep for x in inst.net.arc[i][:2] if x in inst.net.node_res}
    Lmax = max(t["Lv"] for t in inst.tasks)
    s, sw = p["long_clear"], p["turn_sweep_s"]
    clr = max([(Lmax + s) / p["v_loaded"]] + [(inst.size[n] + Lmax + s) / p["v_loaded"] + sw for n in nodes] +
              [(inst.size[n] + p["plat_len"] + s) / p["v_empty"] + sw for n in nodes])
    return K * float((amax + emax).max()) + clr


def _pieces_pos(iv):
    """bound.pieces with positions: [(start, end, maximum width)]."""
    ts = sorted({x for s, e, _ in iv for x in (s, e)})
    return [(a, b, max(d for s, e, d in iv if s <= a and e >= b)) for a, b in zip(ts, ts[1:])
            if any(s <= a and e >= b for s, e, _ in iv)]

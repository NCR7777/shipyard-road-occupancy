"""System bound (system T1'): without sampling, computes the expected occupancy directly from the sampling law of the task
stream and puts it into the same linear programme as T1'.

Mix = [(w, task, bl)]: w is a probability, task = dict(flow, o, d[, lift]) (o, d are stops), bl = (((B, Lv), prob), ...)
(joint distribution). A sampled mix (the first N tasks of a seed) is the special case w = 1/N with a one-point bl, and the same
code reproduces the per-seed T1' of occupancy.bound (tested).
The time a task holds resource r depends only on Lv (start and end of the windows), its width only on (r, B): all windows of
a task on one resource have the same width (network.demand: loaded on a road min(C, B + c), on junctions and bays C, crane 1), so
    ā[k, r] = Σ w · Σ_{(B, Lv)} p · time_r(task, Lv) · u^(k)(width(r, B) / C_r).
The times V are first summed per (bl, Lv) (independent of road widths) and then multiplied by the share table; changing a road
width or a capacity recomputes only the shares, not the windows.
"""
import collections
import heapq

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack

from occupancy.bound import DFF_K, _system, res_name, share, task_windows
from occupancy.params import DOCK_CLASS, FLOW
from occupancy.tasks import crane_p6, flow_weights, mark_lift


def bl_joint(p):
    """Joint distribution of (B, Lv) (tasks.TaskStream): bw ~ U{block_w}, bl ~ U{max(bw, block_l[0]), ..., block_l[1]};
    B = max(bw, plat_w), Lv = max(bl, plat_len)."""
    out = collections.Counter()
    lo, hi = p["block_w"]
    for bw in range(lo, hi + 1):
        a, b = max(bw, p["block_l"][0]), p["block_l"][1]
        for bl in range(a, b + 1):
            out[(float(max(bw, p["plat_w"])), float(max(bl, p["plat_len"])))] += 1 / (hi - lo + 1) / (b - a + 1)
    return tuple(sorted(out.items()))


def od_mix(net, p):
    """Origin-destination distribution of the task stream (sampling law of tasks.TaskStream, both od_sampling variants):
    [(probability, task)], task = dict(flow, o, d[, lift])."""
    w = flow_weights(p)
    pairs, out = net.lift_pairs(p), collections.Counter()
    for (fid, oc, dc, _), wf in zip(FLOW, w):
        if wf <= 0:
            continue
        pf = wf / sum(w)
        if p.get("od_sampling", "entity") == "stop":      # every stop of the role equally likely; destination not in the origin's entity
            S = [s for c in oc.split("|") for e in net.by_cls[c] for s in net.ents[e]["stops"]]
            for so in S:
                o = net.stops[so]["ent"]
                D = [s for c in dc.split("|") for e in net.by_cls[c] if e != o for s in net.ents[e]["stops"]]
                for sd in D:
                    out[(fid, so, sd)] += pf / len(S) / len(D)
            continue
        O = [e for c in oc.split("|") for e in net.by_cls[c]]   # entities equally likely, stops within an entity equally likely
        for o in O:
            D = [e for c in dc.split("|") for e in net.by_cls[c] if e != o]
            for d in D:
                q = pf / len(O) / len(D) / len(net.ents[o]["stops"]) / len(net.ents[d]["stops"])
                for so in net.ents[o]["stops"]:
                    for sd in net.ents[d]["stops"]:
                        out[(fid, so, sd)] += q
    return [(q, mark_lift(net, dict(flow=f, o=o, d=d), pairs)) for (f, o, d), q in out.items()]


def crane_mix(net, p):
    """P6 fixed in absolute number under crane-paced erection (sampling law of tasks.crane_tasks): docks weighted by their
    crane counts (the long-run proportions of crane_split), origin entities equally likely, stops within an entity equally
    likely, stops within the dock equally likely; pairs inside a crane span are marked lift."""
    _, oc, _, _ = next(f for f in FLOW if f[0] == "P6")
    srcs = [e for c in oc.split("|") for e in net.by_cls[c]]
    n, docks, pairs = net.crane_counts(p), net.by_cls[DOCK_CLASS], net.lift_pairs(p)
    return [(n[z] / sum(n[y] for y in docks) / len(srcs) / len(net.ents[o]["stops"]) / len(net.ents[z]["stops"]),
             mark_lift(net, dict(flow="P6", o=so, d=sd), pairs))
            for z in docks for o in srcs for so in net.ents[o]["stops"] for sd in net.ents[z]["stops"]]


def scenario_mix(net, p):
    """(mix, fixed part). In crane-paced scenarios (p has crane_n) the P6 are fixed in absolute number (as in
    bound.bounds_fixed): fixed part = (P6 mix, number per day), and the mix contains only the other flows (flow_share P6 = 0)."""
    bl = bl_joint(p)
    mix = [(w, t, bl) for w, t in od_mix(net, p)]
    fixed = ([(w, t, bl) for w, t in crane_mix(net, p)], crane_p6(net, p)) if p.get("crane_n") else None
    return mix, fixed


def sampled_mix(tasks):
    """Sampled mix: weight 1/N per task, one-point bl."""
    return [(1 / len(tasks), {k: t[k] for k in ("flow", "o", "d", "lift") if k in t}, (((t["B"], t["Lv"]), 1.0),)) for t in tasks]


def union(iv):
    """Length of the union of the windows (same merging as bound.task_occupancy)."""
    iv = sorted(iv)
    tot, (cs, ce, _) = 0.0, iv[0]
    for s, e, _ in iv[1:]:
        if s > ce:
            tot, cs = tot + ce - cs, s
        ce = max(ce, e)
    return tot + ce - cs


def t1_times(net, p, cap, size):
    """Windows of T1' (bound.task_windows, destination counted only until the end of unloading): task -> {resource: length
    of the union}."""
    def f(t):
        by, _ = task_windows(net, p, cap, size, t, tight=True)
        return {r: union(iv) for r, iv in by.items()}
    return f


def width(net, p, cap, r, B):
    """Width a loaded vehicle takes on resource r (network.demand); crane windows have width 1 (bound.task_windows)."""
    return 1.0 if r[0] == "c" else net.demand(r, B, True, p, cap)


class Loads:
    """Integrated occupancy time of a mix on each resource (grouped by (bl, Lv), independent of road widths) and the net node
    outflow on (origin +1, destination −1; direct lifts inside a crane span are not counted).
    times(task) -> {resource: seconds}, for a task carrying B and Lv (times must not depend on B)."""

    def __init__(self, net, p, mix, times, res_list):
        self.net, self.p, self.res_list = net, p, res_list
        ri = {r: i for i, r in enumerate(res_list)}
        self.nodes = sorted({x for a in net.arc for x in a[:2]})         # same node order as bound._system
        ni = {n: i for i, n in enumerate(self.nodes)}
        self.V, self.PB = collections.defaultdict(lambda: np.zeros(len(res_list))), {}
        self.on = np.zeros(len(self.nodes))
        for w, t, bl in mix:
            for Lv in sorted({lv for (_, lv), _ in bl}):
                key = (bl, Lv)
                if key not in self.PB:
                    self.PB[key] = [(B, q) for (B, lv), q in bl if lv == Lv]
                x = self.V[key]
                for r, s in times(dict(t, B=self.PB[key][0][0], Lv=Lv)).items():
                    x[ri[r]] += w * s
            if not t.get("lift"):
                self.on[ni[net.stops[t["o"]]["node"]]] += w
                self.on[ni[net.stops[t["d"]]["node"]]] -= w

    def abar(self, cap):
        """ā[k, r] (row k = DFF_K): integrated time × share u^(k)(width(r, B) / C_r), weighted by the probabilities of (B, Lv)."""
        Bs = sorted({B for pb in self.PB.values() for B, _ in pb})
        S = share_table(self.net, self.p, cap, self.res_list, Bs)
        out = np.zeros((len(DFF_K), len(self.res_list)))
        for key, x in self.V.items():
            pB = np.zeros(len(Bs))
            for B, q in self.PB[key]:
                pB[Bs.index(B)] += q
            out += (x[:, None] * np.einsum("ijk,j->ik", S, pB)).T
        return out


def share_table(net, p, cap, res_list, Bs):
    """S[i, j, k] = u^(k)(width(resource i, B_j) / C_i)."""
    S = np.zeros((len(res_list), len(Bs), len(DFF_K)))
    for i, r in enumerate(res_list):
        for j, B in enumerate(Bs):
            d = width(net, p, cap, r, B)
            S[i, j] = [share(d, cap[r], k) for k in DFF_K]
    return S


def ret_road(net, a, limit):
    """Shortest distance driven after leaving the road of arc a (through a turn successor of a) before entering the same
    road again, capped at limit (obeying turn permissions)."""
    rid, seen = net.arc[a][3], set()
    heap = [(0.0, b) for b in net.succ[a]]
    while heap:
        d, b = heapq.heappop(heap)
        if d >= limit:
            break
        if b in seen:
            continue
        seen.add(b)
        if net.arc[b][3] == rid:
            return d
        for c in net.succ[b]:
            heapq.heappush(heap, (d + net.arc[b][2], c))
    return limit


def stop_ret(net, limit):
    """{arc: ret_road}: arcs that leave a resource carrying stops (the road or section of an out-arc of a stop node) at a map
    node (arc head not an S| node). On whole-road networks these resources are the roads of net.cuts; on sectioned networks
    they are the sections that contain the stops (the original road ids are no longer resources)."""
    res = {net.arc[a][3] for st in net.stops.values() if st["node"].startswith("S|") for a in net.out[st["node"]]}
    return {a: ret_road(net, a, limit) for a, x in enumerate(net.arc) if not x[1].startswith("S|") and x[3] in res}


def empty_block(net, p, cap, size, dummy, ret=None):
    """Empty-run part built by bound._system (conservation rows, the empty-run columns of the occupancy rows and the tail
    clearance, identical term by term to T1'); the caller replaces the λ column (column 0).
    ret = {arc: distance} (stop_ret): on resources that carry stops, the tail clearance of arcs that leave at a map node is
    truncated to min(vehicle length + longitudinal clearance, ret) (the truncated clearing distance κ_a of the paper; proof in
    the paper's appendix); the truncation is implemented in bound._system and shared by the system bound and T1''."""
    Aeq, Aub, res_list, *_ = _system(net, p, cap, size, [dummy], True, ret=ret)
    return Aeq[:, 1:], Aub[:, 1:], res_list


def solve_t1(net, p, cap, size, dummy, abar, on, fixed=None, top=10, rhs=None, ret=None, lp_out=None):
    """The same T1' linear programme as occupancy.bound (the rows and columns of _system, the same linprog call as
    _max_lambda), with the λ column set to the given ā and on.
    fixed = (ā6, on6, number per day): the part that takes resources in absolute number (as in bounds_fixed); the return
    value includes it.
    rhs = {resource: factor}: the capacity H of all rows of that resource is multiplied by the factor (occupancy times 10%
    shorter ⇔ × 1/0.9); ret: see empty_block. Returns (λ, price list price_list).
    If lp_out is a dict, also writes into it the solution x, the dual prices mu of the occupancy rows (tasks/day per second),
    the activity act (including the fixed part) and the capacity H (for first-order predictions)."""
    Aeq, Aub, res_list = empty_block(net, p, cap, size, dummy, ret)
    nr = len(res_list)
    Aub = hstack([csr_matrix(abar.reshape(-1, 1)), Aub]).tocsr()
    Aeq = hstack([csr_matrix(-on.reshape(-1, 1)), Aeq]).tocsr()
    fix = fixed[2] * fixed[0].reshape(-1) if fixed else np.zeros(len(DFF_K) * nr)
    beq = fixed[2] * fixed[1] if fixed else np.zeros(Aeq.shape[0])
    Hvec = capacity_vec(p, res_list, rhs)
    c = np.zeros(Aub.shape[1])
    c[0] = -1.0
    lp = linprog(c, A_ub=Aub, b_ub=Hvec - fix, A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * Aub.shape[1], method="highs")
    if lp.status != 0:
        raise RuntimeError(f"system bound LP failed: {lp.message}")
    if lp_out is not None:
        lp_out.update(x=lp.x, mu=-lp.ineqlin.marginals, act=Aub @ lp.x + fix, H=Hvec, res_list=res_list)
    return float(lp.x[0]) + (fixed[2] if fixed else 0.0), price_list(net, res_list, -lp.ineqlin.marginals, Aub @ lp.x + fix, Hvec, top)


def capacity_vec(p, res_list, rhs=None):
    """Capacity of each occupancy row (row k·nr + i): H, multiplied by the factor for the resources given in rhs."""
    nr = len(res_list)
    H = np.full(len(DFF_K) * nr, p["day_s"])
    for r, m in (rhs or {}).items():
        H[[k * nr + res_list.index(r) for k in range(len(DFF_K))]] *= m
    return H


def price_list(net, res_list, mu, act, Hvec, top=10):
    """The top resources by dual price: price = sum of the dual prices of the resource's rows × 3600 (tasks/day per
    additional 1 h/day, as in bound._max_lambda); utilisation = maximum over its rows of activity (including the fixed part)
    / capacity. The solution is degenerate and often only one or two prices are positive: if there are fewer than top, the
    list is filled up, by utilisation, with resources of price 0, i.e. resources that are binding or nearly binding at the
    same time but whose price falls elsewhere."""
    nr = len(res_list)
    price = mu[:len(DFF_K) * nr].reshape(len(DFF_K), nr).sum(axis=0) * 3600
    util = (act[:len(DFF_K) * nr] / Hvec).reshape(len(DFF_K), nr).max(axis=0)
    pos = [i for i in np.argsort(-price, kind="stable") if price[i] > 1e-9][:top]
    rest = [i for i in np.argsort(-util, kind="stable") if i not in pos][:top - len(pos)]
    return [dict(res=list(res_list[i]), name=res_name(net, res_list[i]), price_per_h=float(price[i]) if i in pos else 0.0,
                 util=float(util[i])) for i in pos + rest]


class SystemT1:
    """System bound (system T1') of a scenario (or T1' of any given mix). value(cap, rhs) recomputes it for the given
    capacities and right-hand sides (only the shares and the linear programme are recomputed)."""

    def __init__(self, net, p, mix, fixed=None):
        self.net, self.p = net, p
        self.cap, self.size = net.capacities(p)
        self.res_list = sorted(self.cap)
        f = t1_times(net, p, self.cap, self.size)
        self.L = Loads(net, p, mix, f, self.res_list)
        self.L6 = Loads(net, p, fixed[0], f, self.res_list) if fixed else None
        self.n6 = fixed[1] if fixed else None
        w, t, bl = next(x for x in mix if not x[1].get("lift"))
        self.dummy = dict(t, B=bl[0][0][0], Lv=bl[0][0][1])

    def value(self, cap=None, top=10, rhs=None, ret=None, lp_out=None):
        """(T1', price list); cap, rhs, ret, lp_out: see solve_t1 (ret = stop_ret(net, ...) truncates the empty-run tail
        clearance to the paper's κ_a)."""
        cap = cap or self.cap
        fixed = (self.L6.abar(cap), self.L6.on, self.n6) if self.L6 else None
        return solve_t1(self.net, self.p, cap, self.size, self.dummy, self.L.abar(cap), self.L.on, fixed, top, rhs, ret, lp_out)

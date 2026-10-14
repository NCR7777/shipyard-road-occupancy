"""Network bound T1 and its tightened form T1' (paper, Section 3 and Appendix A).

Network bound (T1): max lambda (tasks/day) such that no resource is occupied more than the day H:
    lambda * abar_r + sum_a f_a * e_ar <= H        (every resource r)
    sum_in f - sum_out f = lambda * (o_n - d_n)    (every node n: empty vehicles flow from unloading back to loading)
  abar_r: the least occupancy of one task of the mix on r (loading, loaded run, unloading, without waiting; seconds x width
          share), on the loaded routes of the simulation;
  f_a:    daily empty flow on arc a, routes and pairing chosen by the linear programme (relaxed: no turn permissions, no turn
          sweep, no tail clearance of empty runs and no entrance node at the start), so it bounds every dispatch the
          simulation could use.
Tightened (T1'): same variables and conservation constraints, two more "width packing" rows per resource, and the tail
clearance of empty runs on roads:
    lambda * abar_r^(k) + sum_a f_a * e_ar^(k) <= H   (k = 1, 2)
  abar^(k), e^(k) replace the width share x = width taken / capacity by the Fekete-Schepers dual-feasible function u^(k)(x).
  Vehicles in one resource at one time have sum x <= 1, hence sum u^(k)(x) <= 1; integrating over time gives the rows. k = 1:
  a vehicle with a share above one half counts 1 (two such cannot be together); k = 2: shares in (1/3, 2/3) count 1/2, above
  2/3 count 1. Shares are exact rationals (the float widths converted exactly to fractions), with no tolerance band at the
  thresholds. The destination window ends when unloading ends: the clearance after unloading may overlap the empty run of the
  same vehicle (roadside stop) or the origin window of its next task (loading at the stop where it unloaded); leaving it out
  keeps the windows of one vehicle disjoint. The original T1 row (with the clearance) is kept for comparison only.
Fleet bound: K H / cbar, cbar = mean (loading + loaded run + unloading) + the least empty running (minimum-cost flow under the
same conservation constraints).
Dual prices: d lambda / d H_r, the extra tasks/day of the network bound per extra hour/day of resource r; for T1' the sum over
the rows of r.
"""
import collections
import functools
import math
from fractions import Fraction

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

from .network import EPS

DFF_K = (0, 1, 2)          # 0 = identity (the T1 row); 1, 2 = tightening rows


def dff(x, k):
    """Fekete-Schepers dual-feasible function (x a Fraction): x if (k+1)x is an integer, floor((k+1)x)/k otherwise; k = 0 is
    the identity. Non-decreasing. Exact, no tolerance band."""
    if k == 0:
        return x
    y = (k + 1) * x
    return x if y.denominator == 1 else Fraction(math.floor(y), k)


@functools.lru_cache(maxsize=None)
def share(d, C, k):
    """u^(k) of the share min(1, d/C) of width d on capacity C. d and C are converted to exact fractions before dividing, so a
    share exactly at a threshold (such as 6.5/19.5 = 1/3) is not affected by rounding; the result equals the correctly rounded
    float value of d/C."""
    return float(dff(min(Fraction(1), Fraction(d) / Fraction(C)), k))


def task_windows(net, p, cap, size, task, tight=False):
    """Occupancy windows of one task (without waiting) on each resource {resource: [(start, end, width)]} (relative to the
    loaded departure), and the loaded travel time.
    tight: the destination window ends when unloading ends. The clearance after unloading may overlap the empty run of the same
    vehicle (roadside stop) or the origin window of its next task (loading where it unloaded); counting both would count it
    twice. The T1 row keeps the clearance.
    A P6 inside a crane span (lift) uses no vehicle: it only holds the dock's cranes for crane_cycle_s, loaded travel time 0."""
    if task.get("lift"):
        return {("c", net.stops[task["d"]]["ent"]): [(0.0, p["crane_cycle_s"], 1.0)]}, 0.0
    so, sd = net.stops[task["o"]], net.stops[task["d"]]
    d_so = net.demand(so["res"], task["B"], True, p, cap)
    d_sd = net.demand(sd["res"], task["B"], True, p, cap)
    TL, st, uses, _ = net.leg(so["node"], sd["node"], p["v_loaded"], task["Lv"], task["B"], True, p, cap, size, sd["res"], d_sd)
    tail = (task["Lv"] + p["long_clear"]) / p["v_loaded"]
    clr = 0.0 if tight else (p["plat_len"] + p["long_clear"]) / p["v_empty"]
    ad = next(st[a] for _, _, a, r, _ in uses if r is None)
    W = [(so["res"], -p["load_s"], tail, d_so)] + [(res, st[a], st[r] + dl, d) for res, d, a, r, dl in uses if r is not None] \
        + [(sd["res"], ad, TL + p["unload_s"] + clr, d_sd)]
    if task.get("flow") == "P6" and ("c", sd["ent"]) in cap:      # crane row: one crane for crane_cycle_s from the start of unloading
        W.append((("c", sd["ent"]), TL, TL + p["crane_cycle_s"], 1.0))
    by = collections.defaultdict(list)
    for res, s, e, d in W:
        if d > EPS and e > s:
            by[res].append((s, e, d))
    return by, TL


def task_occupancy(net, p, cap, size, task):
    """Least occupancy of one task (without waiting) on each resource (seconds x width share) and the loaded travel time
    (T1 row)."""
    by, TL = task_windows(net, p, cap, size, task)
    out = {}
    for res, iv in by.items():
        iv.sort()
        tot, (cs, ce, _) = 0.0, iv[0]
        for s, e, _ in iv[1:]:
            if s > ce:
                tot, cs = tot + ce - cs, s
            ce = max(ce, e)
        out[res] = (tot + ce - cs) * max(d for *_, d in iv) / cap[res]
    return out, TL


def pieces(iv):
    """The union of windows cut into elementary pieces [(length, largest width of the windows covering the piece)]."""
    ts = sorted({x for s, e, _ in iv for x in (s, e)})
    out = []
    for a, b in zip(ts, ts[1:]):
        w = max((d for s, e, d in iv if s <= a and e >= b), default=0.0)
        if w > EPS:
            out.append((b - a, w))
    return out


def _loads(net, p, cap, size, tasks, tight, ri, ni, ks):
    """Mean occupancy abar[k, resource] of a task of the mix, mean handling and loaded time, net outflow per node on (origin +1,
    destination -1). A lift (P6 inside a crane span) only occupies the cranes: no handling, no flow conservation (no vehicle)."""
    abar = np.zeros((len(ks), len(ri)))
    hand = 0.0
    on = np.zeros(len(ni))
    for t in tasks:
        if tight:
            by, TL = task_windows(net, p, cap, size, t, tight=True)
            for r, iv in by.items():
                for ln, d in pieces(iv):
                    for j, k in enumerate(ks):
                        abar[j, ri[r]] += ln * share(d, cap[r], k)
        else:
            occ, TL = task_occupancy(net, p, cap, size, t)
            for r, x in occ.items():
                abar[0, ri[r]] += x
        if t.get("lift"):
            continue
        hand += p["load_s"] + TL + p["unload_s"]
        on[ni[net.stops[t["o"]]["node"]]] += 1
        on[ni[net.stops[t["d"]]["node"]]] -= 1
    n = len(tasks)
    return abar / n, hand / n, on / n


def _system(net, p, cap, size, tasks, tight, fixed=None, ret=None):
    """tight=False: the T1 system; tight=True: three rows per resource (k = 0, 1, 2), tail clearance on empty runs.
    fixed = (tasks, per day): this part is fixed in absolute terms (P6 under crane pacing): subtracted from the right-hand
    side of the occupancy rows and added to the conservation right-hand side; lambda counts only `tasks`. The returned b_ub,
    b_eq are the corrections to the right-hand sides (0 without fixed).
    ret = {arc: distance}: on resources with a stop, the tail clearance of an empty run leaving a graph node is capped at
    min(vehicle length + longitudinal clearance, ret) (paper, eq. (5); bounds.system.stop_ret). None: no cap."""
    nodes = sorted({x for a in net.arc for x in a[:2]})
    ni = {n: i for i, n in enumerate(nodes)}
    res_list = sorted(cap)
    ri = {r: i for i, r in enumerate(res_list)}
    nr, A = len(res_list), len(net.arc)
    ks = DFF_K if tight else (0,)
    abar, hand, on = _loads(net, p, cap, size, tasks, tight, ri, ni, ks)
    fix_ub, fix_eq = np.zeros(len(ks) * nr), np.zeros(len(nodes))
    if fixed:
        fa, _, fo = _loads(net, p, cap, size, fixed[0], tight, ri, ni, ks)
        fix_ub, fix_eq = fixed[1] * fa.reshape(-1), fixed[1] * fo
    # conservation: sum_in f - sum_out f - lambda (o - d) = 0
    rows, cols, vals = [], [], []
    for j, (tl, hd, *_ ) in enumerate(net.arc):
        rows += [ni[hd], ni[tl]]
        cols += [1 + j, 1 + j]
        vals += [1.0, -1.0]
    for i in np.nonzero(on)[0]:
        rows.append(i); cols.append(0); vals.append(-on[i])
    Aeq = csr_matrix((vals, (rows, cols)), shape=(len(nodes), 1 + A))
    # occupancy: lambda abar_r^(k) + sum f_a e_ar^(k) <= H; row group j belongs to ks[j], row j*nr + r
    v, Le = p["v_empty"], p["plat_len"] + p["long_clear"]
    rows, cols, vals = [], [], []
    for j in range(len(ks)):
        for r in np.nonzero(abar[j])[0]:
            rows.append(j * nr + r); cols.append(0); vals.append(abar[j, r])
    for a, (tl, hd, L, rid, _) in enumerate(net.arc):
        res = ("r", rid)
        d = net.demand(res, p["plat_w"], False, p, cap)
        # tail clearance only on arcs leaving a road (head is a graph node); at a stop the vehicle may just pass, so none
        # (still a lower occupancy)
        hold = L / v + ((min(Le, ret[a]) if ret and a in ret else Le) / v if tight and not hd.startswith("S|") else 0.0)
        for j, k in enumerate(ks):
            x = share(d, cap[res], k) if d > EPS else 0.0
            if x > EPS:
                rows.append(j * nr + ri[res]); cols.append(1 + a); vals.append(hold * x)
        if hd in net.node_res and hd not in net.access:
            res = ("n", hd)
            d = net.demand(res, p["plat_w"], False, p, cap)
            for j, k in enumerate(ks):
                x = share(d, cap[res], k) if d > EPS else 0.0
                if x > EPS:
                    rows.append(j * nr + ri[res]); cols.append(1 + a); vals.append((size[hd] + Le) / v * x)
    Aub = csr_matrix((vals, (rows, cols)), shape=(len(ks) * nr, 1 + A))
    lens = np.array([a[2] for a in net.arc])
    return Aeq, Aub, res_list, abar, hand, lens, fix_ub, fix_eq


def _max_lambda(Aeq, Aub, res_list, abar, H, top, keep=None, fix_ub=None, fix_eq=None):
    """Solve max lambda. keep: keep only the occupancy rows of these resources (local bound); fix_ub, fix_eq: right-hand sides
    taken by the fixed part (_system's fixed). Returns (lambda, binding resources)."""
    nr = len(res_list)
    nk = Aub.shape[0] // nr
    rows = np.arange(Aub.shape[0])
    if keep is not None:
        keep_i = [i for i, r in enumerate(res_list) if r in keep]
        rows = np.array([j * nr + i for j in range(nk) for i in keep_i])
    A = Aub[rows]
    nvar = Aeq.shape[1]
    c = np.zeros(nvar); c[0] = -1.0
    b_ub = np.full(A.shape[0], H) if fix_ub is None else H - fix_ub[rows]
    b_eq = np.zeros(Aeq.shape[0]) if fix_eq is None else fix_eq
    lp = linprog(c, A_ub=A, b_ub=b_ub, A_eq=Aeq, b_eq=b_eq, bounds=[(0, None)] * nvar, method="highs")
    if lp.status != 0:
        raise RuntimeError(f"network bound LP failed: {lp.message}")
    lam = lp.x[0]
    price = collections.defaultdict(float)          # tasks/day per extra hour/day: sum of the dual prices of the rows of a resource
    util = collections.defaultdict(float)
    for i, row in enumerate(rows):
        r = row % nr
        price[r] += -lp.ineqlin.marginals[i] * 3600
        util[r] = max(util[r], 1 - lp.ineqlin.residual[i] / H)
    order = sorted(price, key=lambda r: -price[r])
    return lam, [dict(res=list(res_list[r]), price_per_h=float(price[r]), util=float(util[r]),
                      loaded_share=float(lam * abar[:, r].max() / H)) for r in order[:top] if price[r] > 1e-9]


def bounds_fixed(net, p, tasks, fixed, top=10, local=None):
    """T1' with a part fixed in absolute terms: fixed = (tasks, per day), the P6 arriving at the crane pace; tasks is the mix of
    the other flows and lambda their daily bound. Returns network_tight_fixed = lambda + per day (the bound of the whole yard),
    the same with only the `local` resources, and the binding resources."""
    cap, size = net.capacities(p)
    H = p["day_s"]
    Aeq, Aub, res_list, abar, _, _, fu, fe = _system(net, p, cap, size, tasks, True, fixed)
    lam, binding = _max_lambda(Aeq, Aub, res_list, abar, H, top, fix_ub=fu, fix_eq=fe)
    for b in binding:
        b["name"] = res_name(net, tuple(b["res"]))
    out = dict(network_tight_fixed=float(lam) + fixed[1], binding_tight_fixed=binding)
    if local:
        out["network_tight_fixed_local"] = float(_max_lambda(Aeq, Aub, res_list, abar, H, top, keep=set(local), fix_ub=fu, fix_eq=fe)[0]) + fixed[1]
    return out


def bounds(net, p, tasks, top=10, local=None, tight_only=False, fixed=None, ret=None):
    """Returns a dict: network (T1, tasks/day), network_tight (T1'), cycle_s (cbar of the fleet bound), empty_min_s,
    binding / binding_tight (binding resources and dual prices); with a set of resources `local` also T1' using only these
    (network_tight_local). tight_only: only T1' (the per-point check of simulated points). Crane-paced scenarios (p has crane_n)
    get one crane row per dock (task_windows). fixed = (tasks, per day): a part fixed in absolute terms (_system; in the
    per-point check, the P6 lifted directly by a gantry crane), lambda counts only `tasks`. ret: cap of the empty tail
    clearance (_system)."""
    cap, size = net.capacities(p)
    H = p["day_s"]
    out = dict(n_tasks=len(tasks))
    for tight in (True,) if tight_only else (False, True):
        Aeq, Aub, res_list, abar, hand, lens, fu, fe = _system(net, p, cap, size, tasks, tight, fixed, ret)
        lam, binding = _max_lambda(Aeq, Aub, res_list, abar, H, top, **(dict(fix_ub=fu, fix_eq=fe) if fixed else {}))
        for b in binding:
            b["name"] = res_name(net, tuple(b["res"]))
        sfx = "_tight" if tight else ""
        out["network" + sfx], out["binding" + sfx] = float(lam), binding
        if tight and local:
            out["network_tight_local"] = float(_max_lambda(Aeq, Aub, res_list, abar, H, top, keep=set(local),
                                                           **(dict(fix_ub=fu, fix_eq=fe) if fixed else {}))[0])
    if tight_only:
        return out
    # fleet bound: the least empty running at lambda = 1 (minimum-cost flow)
    nvar = Aeq.shape[1]
    c2 = np.concatenate([[0.0], lens / p["v_empty"]])
    lp2 = linprog(c2, A_eq=Aeq, b_eq=np.zeros(Aeq.shape[0]), bounds=[(1, 1)] + [(0, None)] * (nvar - 1), method="highs")
    if lp2.status != 0:
        raise RuntimeError(f"least empty running LP failed: {lp2.message}")
    out.update(cycle_s=float(hand + lp2.fun), empty_min_s=float(lp2.fun), handling_travel_s=float(hand))
    return out


def bounds_window(net, p, tasks, lifts, W):
    """T1' (tasks/day, counting only `tasks`, the vehicle tasks) for the per-point check: the mix is the tasks completed in a
    statistics window of length W; lifts completed in the same window (gantry crane, inside the span) hold the cranes in
    absolute terms.
    Crane window edge term: a P6 completed in the window started its crane period within an interval of length W (start of
    unloading = completion - unload_s; start of a lift = completion - crane_hook_s, both 10 min), so each crane starts at most
    floor(W / cycle) + 1 times in it, at most one more than the rate W / cycle; the dual-feasible values of the crane periods at
    one time sum to <= 1, and integrating over [start, end + cycle] (length W + cycle) gives sum cycle x u <= W + cycle. Hence
    the crane rows use cycle x W / (W + cycle). The same edge term of road resources is covered by the tolerance K / days."""
    if not tasks:
        return None
    if p.get("crane_n"):
        c = p["crane_cycle_s"]
        p = dict(p, crane_cycle_s=c * W / (W + c))
    fixed = (lifts, len(lifts) * p["day_s"] / W) if lifts else None
    return bounds(net, p, tasks, tight_only=True, fixed=fixed)["network_tight"]


def fleet_bound(K, cycle_s, p):
    return K * p["day_s"] / cycle_s


def node_tags(net):
    """Entrance node -> "building·entrance name" (facility and name of the accessPoints; several entrance nodes share a name,
    so the tag tells them apart)."""
    if not hasattr(net, "_node_tags"):
        tags = {}
        for a in net.D["accessPoints"].values():
            f = net.D["facilities"].get(a.get("facilityId"), {})
            tags.setdefault(a["nodeId"], []).append("·".join(x for x in (f.get("name"), a.get("name")) if x))
        net._node_tags = {n: "; ".join(sorted(x)) for n, x in tags.items()}
    return net._node_tags


def res_name(net, res):
    """Readable name of a resource. Junctions and entrance nodes get [building·entrance name, short id] after the node name
    (ordinary junctions only the short id)."""
    if res[0] == "r":
        r = net.road[res[1]]
        return f"{r['name']} (W={r['W'] if r['W'] is not None else 'unrestricted'}, {r['L']:.0f} m)"
    if res[0] == "n":
        n = res[1]
        sid = n[5:] if n.startswith("node_") else n
        tag = ", ".join(x for x in (node_tags(net).get(n), sid if len(sid) <= 12 else sid[:8]) if x)
        return f"junction {net.D['nodes'][n].get('name') or n} [{tag}]"
    if res[0] == "c":
        return f"crane {net.ents[res[1]]['name']}"
    return f"stop {net.ents[res[1]]['name']}#{res[2]}"

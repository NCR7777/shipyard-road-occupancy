"""Capacity bounds of a yard (paper, Section 3 and Table 7): the system bound (network bound of the exact task mix, without
sampling noise) and its convergence in the sample size, the route-free bound T1'' with its Lagrangian certificate and the
route-fixed value R, the network bound of each task sample, the back-substitution (a feasible value of the route-free
programme, the lower end of the route margin), the read-out of the binding resource, the dual-price table and the
attribution of R - T1'. keep_rows (with solve_rows) solves the system bound with only some groups of occupancy rows (the
width-packing rows dropped; packing_rows.py).

The configuration is results/bounds/<yard>/config.json (network, parameters, scenarios main / erection / crane, seeds, sample
size n_mix and the sizes of the convergence table; ref names the runs whose sampled bounds the system bound is compared with).
Run:   python experiments/capacity_bounds.py results/bounds/okpo/config.json --out <dir> [--scens main ...] [--items system route_free seeds ...]
Writes <dir>/parts/<scenario>__<item>.json (one file per item, a cache for resuming) and, from the parts available,
system.csv, route_free.csv, seeds.csv, convergence.csv, readout.csv, prices.csv and attribution.csv (the stored tables are in
results/bounds/<yard>/).
The route-free programme of a large instance takes minutes (column generation; the paper reports the times).
"""
import argparse
import collections
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack
from scipy.sparse.csgraph import dijkstra

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bounds.route_free import NK, RouteFree, t1_on_path     # noqa: E402
from bounds.system import Loads, SystemT1, capacity_vec, empty_block, sampled_mix, scenario_mix, solve_t1, width   # noqa: E402
from occupancy.bound import bounds, bounds_fixed, res_name, share      # noqa: E402
from occupancy.network import Net                            # noqa: E402
from occupancy.tasks import TaskStream, crane_p6, crane_tasks   # noqa: E402

SCENS = ("main", "erection", "crane")
STEPS = (("none", {}), ("sweep", dict(sweep=True)), ("lv", dict(sweep=True, lv_route=True)),
         ("dest", dict(sweep=True, lv_route=True, dest=True)), ("caps", dict(sweep=True, lv_route=True, dest=True, caps=False)))
STEP_NAME = {"none": "R (all relaxations)", "sweep": "+ turn sweep", "lv": "+ actual vehicle length",
             "dest": "+ destination from entering the last road", "caps": "+ no cap (= T1')"}
CTX = {}


def ctx(cfg, scen):
    """(network, parameters, system mix and fixed part) per configuration and scenario."""
    key = (cfg["network"], scen, json.dumps(cfg.get("roles"), sort_keys=True), json.dumps(cfg["params"], sort_keys=True),
           json.dumps(cfg["scenarios"][scen], sort_keys=True))
    if key not in CTX:
        p = dict(cfg["params"], **cfg["scenarios"][scen])
        net = Net(cfg["network"], p, cfg.get("roles"))
        CTX[key] = dict(net=net, p=p, mix=scenario_mix(net, p))
    return CTX[key]


def sample(net, p, seed, n):
    """The first n tasks of a seed; under crane pacing also the P6 sample fixed in absolute terms (about n / 10)."""
    s = TaskStream(net, seed, p)
    tasks = [s.next(0.0) for _ in range(n)]
    if not p.get("crane_n"):
        return tasks, None
    n6 = crane_p6(net, p)
    return tasks, (crane_tasks(net, seed, dict(p, days=math.ceil(n / 10 / n6)), n6), n6)


def official_t1(net, p, tasks, fx):
    """T1' of occupancy.bound for one task sample (under crane pacing network_tight_fixed of bounds_fixed)."""
    return bounds_fixed(net, p, tasks, fx)["network_tight_fixed"] if fx else bounds(net, p, tasks, tight_only=True)["network_tight"]


def item_system(cfg, scen):
    c = ctx(cfg, scen)
    t = time.time()
    st = SystemT1(c["net"], c["p"], *c["mix"])
    c["sys"] = st
    v, binding = st.value()
    mix, fixed = c["mix"]
    return dict(T1=v, binding=binding, secs=round(time.time() - t, 1), n_types=len(mix), n_fixed_types=len(fixed[0]) if fixed else 0,
                lp=dict(rows=len(st.L.nodes) + 3 * len(st.res_list), cols=1 + len(c["net"].arc)))


def item_route_free(cfg, scen):
    c = ctx(cfg, scen)
    t = time.time()
    rf = RouteFree(c["net"], c["p"], *c["mix"])
    init = time.time() - t
    out = rf.solve()
    c["rf"], c["pool"], c["sol"] = rf, out.pop("pool"), out.pop("sol")
    out.pop("duals")
    return dict(out, init_secs=round(init, 1))


def need(cfg, scen, *keys):
    """Recompute the objects of earlier items that a later item needs (system bound, route-free solution)."""
    c = ctx(cfg, scen)
    if "sys" in keys and "sys" not in c:
        c["sys"] = SystemT1(c["net"], c["p"], *c["mix"])
    if "rf" in keys and "sol" not in c:
        item_route_free(cfg, scen)
    return c


def com_tasks(net, mix, fixed):
    """Commodity (part, origin node, destination node, B) -> [(weight, task with B and Lv)]; no lifts, no origin = destination."""
    com = collections.defaultdict(list)
    for prt, m in enumerate([mix] + ([fixed[0]] if fixed else [])):
        for w, t, bl in m:
            o, d = net.stops[t["o"]]["node"], net.stops[t["d"]]["node"]
            if t.get("lift") or o == d:
                continue
            for (B, lv), pr in bl:
                com[(prt, o, d, B)].append((w * pr, dict(t, B=B, Lv=lv)))
    return com


def item_backsub(cfg, scen):
    """Back-substitution: keep the route split of every commodity in the optimal solution of the route-free master problem,
    put back the unrelaxed occupancy of T1' on those routes (actual vehicle length, turn sweep, destination from entering the
    last road, uncapped tail; empty rows as in T1'), re-optimise the empty flows and solve for lambda. This is a feasible
    value of "T1' with detours allowed", so the route margin in the bound is at least feasible - T1'. Also reported: the
    largest row utilisation of the original solution put back, and the number of rows above 1 (where the relaxation on detour
    routes raises T1'')."""
    c = need(cfg, scen, "sys", "rf")
    return backsub(c["net"], c["p"], c["sys"], c["rf"], c["sol"], *c["mix"])


def backsub(net, p, st, rf, sol, mix, fixed=None):
    t0 = time.time()
    cap, size, nr = st.cap, st.size, len(st.res_list)
    ri = {r: i for i, r in enumerate(st.res_list)}
    n6 = st.n6 or 0.0
    ab0 = st.L.abar(cap).reshape(-1)
    ab6 = st.L6.abar(cap).reshape(-1) if st.L6 else np.zeros(NK * nr)

    def vec(t, arcs):
        x = np.zeros(NK * nr)
        for r, u in t1_on_path(net, p, cap, size, t, arcs).items():
            w = width(net, p, cap, r, t["B"])
            for k in range(NK):
                x[k * nr + ri[r]] += u * share(w, cap[r], k)
        return x
    com, cache = com_tasks(net, mix, fixed), {}
    a_route, a6_route, dz0, dz6 = ab0.copy(), ab6.copy(), np.zeros(NK * nr), np.zeros(NK * nr)
    for (prt, i, j, arcs), z in sol["z"].items():
        o, d = rf.parts[prt]["pairs"][i]
        lst, P0 = com[(prt, o, d, rf.Bs[j])], rf.parts[prt]["paths"][i]
        q = sum(w for w, _ in lst)
        diff = np.zeros(NK * nr)
        for w, t in lst:
            ck = (prt, i, t["flow"], t["o"], t["d"], t["B"], t["Lv"])
            if ck not in cache:
                cache[ck] = vec(t, P0)
            diff += w / q * (vec(t, arcs) - cache[ck])
        if prt == 0:
            a_route += z / sol["lam"] * diff
            dz0 += z * diff
        else:
            a6_route += z / n6 * diff
            dz6 += z * diff
    fx6 = (a6_route.reshape(NK, nr), st.L6.on, n6) if st.L6 else None
    lam_f = solve_t1(net, p, cap, size, rf.dummy, a_route.reshape(NK, nr), st.L.on, fx6)[0]
    _, Ae0, _ = empty_block(net, p, cap, size, rf.dummy)
    act = sol["lam"] * ab0 + Ae0 @ sol["f"] + dz0 + n6 * ab6 + dz6
    ratio = act / (p["day_s"])
    return dict(feasible=float(lam_f), T1=float(st.value()[0]), cols=len(sol["z"]), ratio_max=float(ratio.max()),
                rows_over_1=int((ratio > 1 + 1e-9).sum()), secs=round(time.time() - t0, 1))


def item_readout(cfg, scen):
    """Two testable quantities of the binding resource (highest dual price of the system bound, loaded runs on shortest
    paths): pass_share, the share of the loaded flow on it that only passes (neither origin nor destination on it); alt_slack,
    the remaining utilisation 1 - utilisation of the tightest resource on the shortest alternative route avoiding it,
    weighted by the passing flow; alt_extra_m, the extra length of that route (weighted mean)."""
    c = need(cfg, scen, "sys", "rf")
    return readout(c["net"], c["sys"], c["rf"])


def readout(net, st, rf):
    T1, full = st.value(top=len(st.res_list))
    util = {tuple(b["res"]): b["util"] for b in full}
    r = tuple(full[0]["res"])
    n6 = st.n6 or 0.0
    ends = {net.road[r[1]]["a"], net.road[r[1]]["b"]} if r[0] == "r" else set()

    def uses(path, o, d):
        if r[0] == "r":
            return any(net.arc[a][3] == r[1] for a in path)
        return r[1] in (o, d) or any(net.arc[a][1] == r[1] for a in path[:-1])

    def local(o, d):
        if r[0] == "r":
            return any(x.startswith(f"S|{r[1]}|") or (x in ends and x in net.stops) for x in (o, d))
        return r[1] in (o, d)
    flows = []
    for prt, P_ in enumerate(rf.parts):
        scale = (T1 - n6) if prt == 0 else n6
        for (o, d), path, qq in zip(P_["pairs"], P_["paths"], P_["Q"].sum(axis=1)):
            if uses(path, o, d):
                flows.append((o, d, path, scale * qq, local(o, d)))
    on = sum(f for *_, f, _ in flows)
    pas = [(o, d, path, f) for o, d, path, f, loc in flows if not loc]
    out = dict(res=list(r), name=full[0]["name"], price_per_h=full[0]["price_per_h"], flow_on=on,
               pass_share=sum(f for *_, f in pas) / on if on else 0.0, n_pass_pairs=len(pas))
    if not pas:
        return dict(out, alt_slack=None, alt_extra_m=None, unreachable_share=None)
    keep = [a for a, x in enumerate(net.arc) if (x[3] != r[1] if r[0] == "r" else r[1] not in x[:2])]
    t_, h_, L = rf.tidx[keep], rf.hidx[keep], rf.alen[keep]
    order = np.lexsort((L, h_, t_))
    first = np.r_[True, (t_[order][1:] != t_[order][:-1]) | (h_[order][1:] != h_[order][:-1])]
    sel = order[first]
    G = csr_matrix((L[sel], (t_[sel], h_[sel])), shape=(len(rf.nodes),) * 2)
    arc_of = {(int(a), int(b)): keep[int(s)] for a, b, s in zip(t_[sel], h_[sel], sel)}
    srcs = sorted({rf.ni[o] for o, *_ in pas})
    dist, pred = dijkstra(G, directed=True, indices=srcs, return_predecessors=True)
    sl, ex, wt, lost = 0.0, 0.0, 0.0, 0.0
    for o, d, path, f in pas:
        k, n = srcs.index(rf.ni[o]), rf.ni[d]
        if not np.isfinite(dist[k, n]):
            lost += f
            continue
        arcs = []
        while pred[k, n] >= 0:
            arcs.append(arc_of[(int(pred[k, n]), n)])
            n = int(pred[k, n])
        arcs.reverse()
        res = {("r", net.arc[a][3]) for a in arcs} | {("n", net.arc[a][0]) for a in arcs[1:] if net.arc[a][0] in net.node_res}
        sl += f * (1 - max(util.get(x, 0.0) for x in res))
        ex += f * (dist[k, rf.ni[d]] - sum(net.arc[a][2] for a in path))
        wt += f
    return dict(out, alt_slack=sl / wt if wt else None, alt_extra_m=ex / wt if wt else None,
                unreachable_share=lost / sum(f for *_, f in pas))


def item_seeds(cfg, scen):
    c, rows = ctx(cfg, scen), []
    for seed in cfg["seeds"]:
        t = time.time()
        tasks, fx = sample(c["net"], c["p"], seed, cfg["n_mix"])
        t1 = official_t1(c["net"], c["p"], tasks, fx)
        out = RouteFree(c["net"], c["p"], sampled_mix(tasks), (sampled_mix(fx[0]), fx[1]) if fx else None).solve()
        rows.append(dict(seed=seed, T1=t1, R=out["R"], T1pp=out["T1pp"], gap=out["gap"], iters=out["iters"],
                         detour=out["detour"]["share"], top=out["binding"][0]["name"] if out["binding"] else "", secs=round(time.time() - t, 1)))
    return rows


def item_conv(cfg, scen, n):
    c, rows = ctx(cfg, scen), []
    for seed in cfg["seeds"]:
        t = time.time()
        tasks, fx = sample(c["net"], c["p"], seed, n)
        rows.append(dict(seed=seed, N=n, T1=official_t1(c["net"], c["p"], tasks, fx), secs=round(time.time() - t, 1)))
    return rows


def item_attribution(cfg, scen):
    """Sources of R - T1': on the system mix, restore the T1' occupancy step by step (the four switches of
    RouteFree.route_times), routes fixed to the shortest paths."""
    c = ctx(cfg, scen)
    rf = c.get("rf") or RouteFree(c["net"], c["p"], *c["mix"])
    mix, fixed = c["mix"]
    out = {}
    for name, kw in STEPS:
        t = time.time()
        f = lambda task: rf.route_times(task, **kw)
        L = Loads(c["net"], c["p"], mix, f, rf.res_list)
        L6 = Loads(c["net"], c["p"], fixed[0], f, rf.res_list) if fixed else None
        v, _ = solve_t1(c["net"], c["p"], rf.cap, rf.size, rf.dummy, L.abar(rf.cap), L.on,
                        (L6.abar(rf.cap), L6.on, fixed[1]) if fixed else None, ret=rf.ret if kw.get("caps", True) else None)
        out[name] = dict(value=v, secs=round(time.time() - t, 1))
    return out


def res_info(net, cap, size, p, r):
    """Type and size of a resource: road (width, length), junction / entrance (size D), stop, cranes (number)."""
    if r[0] == "r":
        x = net.road[r[1]]
        return dict(type="road", W=x["W"] if x["W"] is not None else f"unrestricted (as {p['unrestricted_w']:.0f})", L=round(x["L"], 1))
    if r[0] == "n":
        return dict(type="entrance" if r[1] in net.access else "junction", W=f"D = {size[r[1]]:g}", L="")
    if r[0] == "c":
        return dict(type="crane", W=f"{cap[r]:g} crane" + ("" if cap[r] == 1 else "s"), L="")
    return dict(type="stop", W="", L="")


def item_prices(cfg, scen):
    """Dual-price table: the union of the ten resources with the highest dual prices of the system bound and of T1'', each with
    both columns, "shortest-path dispatch (T1')" and "detours allowed (T1'')": price (tasks/day per extra h/day), util,
    d10_first (first-order gain of 10 % less occupancy = price x 16 h x (1/0.9 - 1)), d10 (the same change re-solved), w1
    (road 1 m wider, re-solved; w1_D marks roads whose end junctions would grow by the junction rule). Resources with
    utilisation < 1 - 1e-6 are slack in every row at the optimum, so relaxing them changes nothing: 0 without re-solving."""
    c = need(cfg, scen, "sys", "rf")
    return price_rows(c["net"], c["p"], c["sys"], c["rf"], c["pool"], scen)


def price_rows(net, p, st, rf, pool, scen, ret=None):
    nr = len(st.res_list)
    b1, l1 = st.value(top=nr, ret=ret)
    out2 = rf.solve(pool=pool, top=nr)
    b2, l2 = out2["T1pp"], out2["binding"]
    info = {"T1′": (b1, {tuple(x["res"]): x for x in l1}, lambda **kw: st.value(ret=ret, **kw)[0]),
            "T1″": (b2, {tuple(x["res"]): x for x in l2}, lambda **kw: rf.solve(pool=pool, **kw)["T1pp"])}
    rows = []
    for rank, r in enumerate(dict.fromkeys([tuple(x["res"]) for x in l1[:10]] + [tuple(x["res"]) for x in l2[:10]]), 1):
        row = dict(scen=scen, rank=rank, res="|".join(map(str, r)), name=info["T1′"][1][r]["name"], **res_info(net, st.cap, st.size, p, r))
        if r[0] == "r":
            W = net.road[r[1]]["W"]
            row["w1_D"] = W is not None and any(n in st.size and W + 1 + p["width_delta"] > st.size[n]
                                                 for n in (net.road[r[1]]["a"], net.road[r[1]]["b"]))
        for tag, (base, lst, solve) in info.items():
            x = lst[r]
            tight = x["util"] >= 1 - 1e-6
            row.update({f"{tag}_base": base, f"{tag}_price": x["price_per_h"], f"{tag}_util": x["util"],
                        f"{tag}_d10_first": x["price_per_h"] * p["day_s"] / 3600 * (1 / 0.9 - 1),
                        f"{tag}_d10": solve(rhs={r: 1 / 0.9}) - base if tight else 0.0})
            if r[0] == "r":
                cap = dict(st.cap)
                cap[r] += 1.0
                row[f"{tag}_w1"] = solve(cap=cap) - base if tight else 0.0
        rows.append(row)
    return rows


def solve_rows(net, p, cap, res_list, Aeq, Aub, abar, on, fix6, on6, n6, keep):
    """The same linear programme as bounds.system.solve_t1 with only the occupancy rows of the groups `keep` of DFF_K. abar,
    fix6 have the shape (len(DFF_K), nr) or hold only the group k = 0. Returns (lambda + fixed part, name of the resource with
    the highest dual price, its price (tasks/day per extra h/day))."""
    nr = len(res_list)
    rows = np.array([k * nr + i for k in keep for i in range(nr)])
    A = hstack([csr_matrix(abar.reshape(-1, 1)), Aub]).tocsr()[rows]
    E = hstack([csr_matrix(-on.reshape(-1, 1)), Aeq]).tocsr()
    fix = (n6 * fix6.reshape(-1))[rows] if n6 else np.zeros(len(rows))
    beq = n6 * on6 if n6 else np.zeros(E.shape[0])
    H = capacity_vec(p, res_list)[rows]
    c = np.zeros(A.shape[1])
    c[0] = -1.0
    lp = linprog(c, A_ub=A, b_ub=H - fix, A_eq=E, b_eq=beq, bounds=[(0, None)] * A.shape[1], method="highs")
    if lp.status != 0:
        raise RuntimeError(f"linear programme failed: {lp.message}")
    price = np.zeros(nr)
    for j, r in enumerate(rows):
        price[r % nr] += -lp.ineqlin.marginals[j] * 3600
    top = int(np.argmax(price))
    return float(lp.x[0]) + (n6 or 0.0), res_name(net, res_list[top]), float(price[top])


def keep_rows(st, keep, ret):
    """The system bound with only the occupancy rows of the groups `keep` (keep = (0,): the width-packing rows dropped);
    windows, empty-run tail clearance and truncation as in the system bound (used by packing_rows.py and the bound
    sensitivity)."""
    cap = st.cap
    Aeq, Aub, res_list = empty_block(st.net, st.p, cap, st.size, st.dummy, ret)
    six = (st.L6.abar(cap), st.L6.on, st.n6) if st.L6 else (None, None, None)
    return solve_rows(st.net, st.p, cap, res_list, Aeq, Aub, st.L.abar(cap), st.L.on, six[0], six[1], six[2], keep)


ITEMS = {"system": item_system, "route_free": item_route_free, "backsub": item_backsub, "readout": item_readout,
         "seeds": item_seeds, "attribution": item_attribution, "prices": item_prices}


def ref_values(cfg, scen):
    """Network bound of each task sample of the reference runs (main, erection: the bound runs; crane: the crane-paced runs)."""
    d = cfg["ref"]["rho" if scen == "crane" else "lp"]
    key = "network_tight_fixed" if scen == "crane" else "network_tight"
    rows = csv.DictReader(open(ROOT / d / "network_bound_seeds.csv", encoding="utf-8"))
    vals = {int(r["seed"]): float(r[key]) for r in rows if r["scen"] == scen}
    return {seed: vals[seed] for seed in cfg["seeds"]}, d


def facility(net, res):
    """The building or zone of an entrance node, stop or crane (many entrance nodes share a name); empty for roads and
    ordinary junctions."""
    if res[0] == "n" and res[1] in net.stops:
        return net.ents[net.stops[res[1]]["ent"]]["name"]
    if res[0] in ("b", "c"):
        return net.ents[res[1]]["name"]
    return ""


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def aggregate(cfg, out):
    parts = {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in (out / "parts").glob("*.json")}
    get = lambda s, i: parts.get(f"{s}__{i}")
    net = Net(cfg["network"], cfg["params"], cfg.get("roles"))
    tag = lambda b: b["name"] + (f" [{facility(net, tuple(b['res']))}]" if facility(net, tuple(b["res"])) else "")
    for k, v in parts.items():
        for b in v.get("binding", []) if isinstance(v, dict) else []:
            b["name"] = tag(b)
        if k.endswith("__prices"):
            for row in v:
                row["facility"] = facility(net, tuple(row["res"].split("|")[:2]))
    sysrows, convrows, seedrows, rfrows, attrows, prices, readrows = [], [], [], [], [], [], []
    for s in cfg["scenarios"]:
        sy, rf, sd = get(s, "system"), get(s, "route_free"), get(s, "seeds")
        if sy:
            ref, d = ref_values(cfg, s)
            T = sy["T1"]
            r10 = list(ref.values())
            sysrows.append(dict(scen=s, T1_system=T, ref_dir=d, ref_mean=sum(r10) / len(r10), ref_min=min(r10), ref_max=max(r10),
                                sys_vs_mean=T / (sum(r10) / len(r10)) - 1, top1=sy["binding"][0]["name"],
                                top1_price=sy["binding"][0]["price_per_h"], secs=sy["secs"], n_types=sy["n_types"],
                                n_fixed_types=sy["n_fixed_types"]))
            for x in sd or []:
                seedrows.append(dict(scen=s, **x, T1_ref=ref[x["seed"]], same_as_ref=abs(x["T1"] / ref[x["seed"]] - 1) < 1e-9,
                                     order_ok=x["T1"] <= x["R"] * (1 + 1e-9) and x["R"] <= x["T1pp"] * (1 + 1e-9)))
            for n in cfg["conv_ns"]:
                xs = [x["T1"] for x in (sd or [] if n == cfg["conv_ns"][0] else get(s, f"conv{n}") or [])]
                if xs:
                    dev = [abs(x / T - 1) for x in xs]
                    hm = len(xs) / sum(1 / x for x in xs)
                    inv = np.array([1 / x for x in xs])
                    hm_se = float(inv.std(ddof=1) / np.sqrt(len(xs)) / inv.mean() ** 2)
                    convrows.append(dict(scen=s, N=n, seeds=len(xs), mean=sum(xs) / len(xs), min=min(xs), max=max(xs), system=T,
                                         mean_minus_system=sum(xs) / len(xs) - T, hmean=hm, hmean_minus_system=hm - T, hmean_se=hm_se,
                                         hmean_z=(hm - T) / hm_se, mean_abs_rel=sum(dev) / len(dev), max_abs_rel=max(dev)))
        if sy and rf:
            T = sy["T1"]
            dt, bs = rf["detour"], get(s, "backsub") or {}
            lo = bs.get("feasible", float("nan"))
            rfrows.append(dict(scen=s, T1_system=T, R=rf["R"], T1pp=rf["T1pp"], feasible=lo, margin_lo=lo - T, margin_hi=rf["T1pp"] - T,
                               margin_lo_rel=lo / T - 1, margin_hi_rel=rf["T1pp"] / T - 1, backsub_ratio_max=bs.get("ratio_max"),
                               backsub_rows_over_1=bs.get("rows_over_1"), backsub_cols=bs.get("cols"), backsub_secs=bs.get("secs"),
                               R_minus_T1=rf["R"] - T, gap=rf["gap"], iters=rf["iters"], cols=rf["cols"], mu_min=rf.get("mu_min"),
                               detour_share=dt["share"], detoured_per_day=dt["detoured_per_day"], extra_m=dt["extra_m"],
                               extra_rel=dt["extra_rel"], extra_km_per_day=dt["extra_km_per_day"],
                               avoided=" / ".join(f"{a['name']} {a['detoured_per_day']:.1f}" for a in dt["avoided"]),
                               top1=rf["binding"][0]["name"], top1_price=rf["binding"][0]["price_per_h"], rows=rf["lp"]["rows"],
                               lp_cols=rf["lp"]["cols"], nnz=rf["lp"]["nnz"], keys=rf["lp"]["keys"], secs=rf["secs"],
                               init_secs=rf["init_secs"], dual_infeas=rf["dual_infeas"]))
            at = get(s, "attribution")
            if at:
                prev = None
                for name, _ in STEPS:
                    v = at[name]["value"]
                    ref_v = rf["R"] if name == "none" else (T if name == "caps" else None)
                    attrows.append(dict(scen=s, step=name, name=STEP_NAME[name], value=v, delta=None if prev is None else v - prev,
                                        minus_T1=v - T, check_rel=None if ref_v is None else v / ref_v - 1, secs=at[name]["secs"]))
                    prev = v
            rd = get(s, "readout")
            if rd:
                readrows.append(dict(scen=s, name=tag(rd), type=res_info(net, *net.capacities(cfg["params"]), cfg["params"], tuple(rd["res"]))["type"],
                                     **{k: v for k, v in rd.items() if k not in ("res", "name")},
                                     margin_lo_rel=rfrows[-1]["margin_lo_rel"], margin_hi_rel=rfrows[-1]["margin_hi_rel"]))
        prices += get(s, "prices") or []
    for name, rows in (("system", sysrows), ("convergence", convrows), ("seeds", seedrows), ("route_free", rfrows),
                       ("attribution", attrows), ("prices", prices), ("readout", readrows)):
        if rows:
            write_csv(out / f"{name}.csv", rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scens", nargs="*", default=None)
    ap.add_argument("--items", nargs="*", default=None, help=f"items of {list(ITEMS)} and conv<N> (default: all)")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    out = Path(a.out)
    (out / "parts").mkdir(parents=True, exist_ok=True)
    items = dict(ITEMS, **{f"conv{n}": (lambda n: lambda c, s: item_conv(c, s, n))(n) for n in cfg["conv_ns"][1:]})
    for s in a.scens or [x for x in SCENS if x in cfg["scenarios"]]:
        for name in a.items or list(items):
            f = out / "parts" / f"{s}__{name}.json"
            if f.exists():
                continue
            t = time.time()
            f.write_text(json.dumps(items[name](cfg, s), ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"{s} {name}: {time.time() - t:.0f} s", flush=True)
    aggregate(cfg, out)


if __name__ == "__main__":
    main()

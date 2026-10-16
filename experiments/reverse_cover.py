"""Reversing coverage: the admissible rules cover reversing, i.e. the cut-off distance eta_a of the tail clearance kappa_a of
empty vehicles no longer depends on turning permissions.

bounds/ and occupancy/ are not modified: only the cut-off table ret ({arc: distance}, with the definition and the arc set
of bounds.system.stop_ret) passed to SystemT1.value and RouteFree is replaced. Serves the paper's statement that the
capacity bounds hold when vehicles may reverse or turn back.

Four cut-offs (all only on resources that carry a stop, on arcs whose head is a map node; kappa_a = min(vehicle length +
longitudinal clearance, eta_a)):
  cur   current: turning permissions respected, shortest distance after leaving until the vehicle first drives onto the
        same resource again (bounds.system.stop_ret; at a dead-end entrance turning back is allowed, 0);
  star  the paper's eta*_a: shortest distance from the head of a to the other end of the resource without passing the
        resource, turning permissions not required (0 for a loop road whose two ends are the same node);
  free  the direct formulation: turning permissions not required (turning back allowed at every node), shortest distance
        after leaving until the vehicle first drives onto the same resource again;
  zero  0 on every such arc ("take 0 where needed", the extreme case).
Also checks eta*_a >= eta_cur(a) (holds arc by arc when turning permissions forbid only turning back). The system bound is
computed on the whole and on the sectioned network, the route-free bound on the whole network.

  python experiments/reverse_cover.py --out DIR [--slots DIR]
  -> DIR/bounds.csv (one row per yard, network, bound and cut-off), DIR/eta.csv (the cut-offs arc by arc), DIR/summary.json,
     DIR/tables.md."""
import argparse
import csv
import hashlib
import heapq
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bounds.route_free as RFm                                  # noqa: E402
from bounds.system import SystemT1, scenario_mix, stop_ret       # noqa: E402
from occupancy import params as P                                # noqa: E402
from occupancy.network import Net                                # noqa: E402

YARDS = ("yupu", "yantai")
VARIANTS = ("cur", "star", "free", "zero")


def stop_arcs(net):
    """The arc set of stop_ret: arcs on resources that carry a stop, whose head is a map node."""
    res = {net.arc[a][3] for st in net.stops.values() if st["node"].startswith("S|") for a in net.out[st["node"]]}
    return [a for a, x in enumerate(net.arc) if not x[1].startswith("S|") and x[3] in res]


def eta_star(net, a, limit):
    """eta*_a: shortest distance from the head n of a to the other end of resource r(a) without passing r(a) (any arc
    sequence, turning permissions ignored), capped at limit; 0 when both ends are the same node."""
    rid, n = net.arc[a][3], net.arc[a][1]
    ends = {net.road[rid]["a"], net.road[rid]["b"]}
    assert n in ends, ("arc head is not an end of the resource", a)
    other = ends - {n}
    if not other:
        return 0.0
    (m,) = other
    dist, heap = {n: 0.0}, [(0.0, n)]
    while heap:
        d, x = heapq.heappop(heap)
        if d >= limit:
            break
        if x == m:
            return d
        if d > dist[x]:
            continue
        for b in net.out[x]:
            if net.arc[b][3] == rid:
                continue
            y, nd = net.arc[b][1], d + net.arc[b][2]
            if nd < dist.get(y, math.inf):
                dist[y] = nd
                heapq.heappush(heap, (nd, y))
    return limit


def eta_free(net, a, limit):
    """The direct formulation: after leaving r(a) (first arc not on r(a)), shortest distance driven by any arc sequence
    (turning back allowed) before first driving onto r(a) again, capped at limit."""
    rid, seen = net.arc[a][3], set()
    heap = [(0.0, b) for b in net.out[net.arc[a][1]] if net.arc[b][3] != rid]
    heapq.heapify(heap)
    while heap:
        d, b = heapq.heappop(heap)
        if d >= limit:
            break
        if b in seen:
            continue
        seen.add(b)
        if net.arc[b][3] == rid:
            return d
        for c in net.out[net.arc[b][1]]:
            heapq.heappush(heap, (d + net.arc[b][2], c))
    return limit


def ret_of(kind, net, limit):
    if kind == "cur":
        return stop_ret(net, limit)
    arcs = stop_arcs(net)
    if kind == "zero":
        return {a: 0.0 for a in arcs}
    f = eta_star if kind == "star" else eta_free
    return {a: f(net, a, limit) for a in arcs}


def net_of(yard, p):
    Y = P.YARDS[yard]
    return Net(Y["network"], p, Y["roles"])


def job(args):
    """One (yard, sectioning): the system bound under the four cut-offs; for the whole network also the route-free bound.
    Returns (rows, eta table, seconds)."""
    yard, section = args
    t0 = time.time()
    p = P.scenario("main", section=section) if section else P.scenario("main")
    net = net_of(yard, p)
    mix, fixed = scenario_mix(net, p)
    st = SystemT1(net, p, mix, fixed)
    Le = p["plat_len"] + p["long_clear"]
    rets = {k: ret_of(k, net, Le) for k in VARIANTS}
    rows = []
    for k in VARIANTS:
        v, b = st.value(top=3, ret=rets[k])
        rows.append(dict(yard=yard, section=section or "whole", bound="system", variant=k, value=v,
                         top1=b[0]["name"] if b else "", top1_price=b[0]["price_per_h"] if b else None,
                         top2=b[1]["name"] if len(b) > 1 else "", top3=b[2]["name"] if len(b) > 2 else "",
                         n_arcs=len(rets[k]), n_short=sum(x < Le - 1e-9 for x in rets[k].values())))
    if not section:
        for k in VARIANTS:
            orig = RFm.stop_ret
            RFm.stop_ret = lambda net_, limit, k=k: ret_of(k, net_, limit)
            try:
                sol = RFm.RouteFree(net, p, mix, fixed).solve(top=3)
            finally:
                RFm.stop_ret = orig
            rows.append(dict(yard=yard, section="whole", bound="route_free", variant=k, value=sol["T1pp"],
                             top1=sol["binding"][0]["name"] if sol["binding"] else "",
                             top1_price=sol["binding"][0]["price_per_h"] if sol["binding"] else None,
                             top2=sol["binding"][1]["name"] if len(sol["binding"]) > 1 else "",
                             top3=sol["binding"][2]["name"] if len(sol["binding"]) > 2 else "",
                             n_arcs=len(rets[k]), n_short=None, gap=sol["gap"]))
    eta = []
    for a in sorted(rets["cur"]):
        x = net.arc[a]
        eta.append(dict(yard=yard, section=section or "whole", arc=a, tail=x[0], head=x[1], road=x[3], length=x[2],
                        **{f"eta_{k}": rets[k][a] for k in ("cur", "star", "free")}))
        assert rets["star"][a] >= rets["cur"][a] - 1e-9, ("eta* < eta_cur", yard, section, a, rets["star"][a], rets["cur"][a])
    return rows, eta, round(time.time() - t0, 1)


def code_sha():
    h = hashlib.sha256()
    for f in sorted((ROOT / "occupancy").glob("*.py")) + sorted((ROOT / "bounds").glob("*.py")) + [Path(__file__)]:
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser(description="Capacity bounds under the four cut-offs of the empty-vehicle tail clearance.")
    ap.add_argument("--out", required=True, help="output directory (bounds.csv, eta.csv, summary.json, tables.md)")
    ap.add_argument("--slots", default=os.environ.get("CPU_SLOTS"), help="CPU-slot directory (optional; default $CPU_SLOTS)")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "experiments"))
    from ordered import pool_map
    todo = [(y, s) for y in YARDS for s in (None, "stops")]
    res = pool_map(job, todo, 4, a.slots, "reverse_cover")
    rows = [r for x in res for r in x[0]]
    eta = [r for x in res for r in x[1]]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("bounds.csv", rows), ("eta.csv", eta)):
        keys = list(dict.fromkeys(k for r in data for k in r))
        with open(out / name, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, keys)
            w.writeheader()
            w.writerows(data)
    base = {(r["yard"], r["section"], r["bound"]): r["value"] for r in rows if r["variant"] == "cur"}
    lines = ["| yard | network | bound | cut-off | value (blocks / 16 h) | vs current | binding elements (top three) | "
             "arcs cut below L0 + s |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        b = base[(r["yard"], r["section"], r["bound"])]
        short = "" if r["n_short"] is None else f"{r['n_short']} / {r['n_arcs']}"
        lines.append(f"| {r['yard']} | {r['section']} | {r['bound']} | {r['variant']} | {r['value']:.2f} | {100 * (r['value'] / b - 1):+.4f}% | "
                     f"{r['top1']} / {r['top2']} / {r['top3']} | {short} |")
    (out / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summ = dict(code_sha=code_sha(), secs={f"{y}|{s}": t for (y, s), (_, _, t) in zip(todo, res)},
                star_ge_cur=all(r["eta_star"] >= r["eta_cur"] - 1e-9 for r in eta),
                values={f"{r['yard']}|{r['section']}|{r['bound']}|{r['variant']}": r["value"] for r in rows})
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(lines))
    print(out)


if __name__ == "__main__":
    main()

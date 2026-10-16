"""Roads that blocks wider than the road pass, and the system bound under the ban on blocks wider than the passable width
(supplementary material, sensitivity to the inputs: the row "ban on blocks wider than the passable width"; the check of the
clear width beside roads in the table of inputs). The ban, its rerouting and the sampling law under it are those of
experiments/width_ban.py.

  --part roads   both yards, main mix (the exact sampling law of the system bound, bounds.system.od_mix x bl_joint, ordinary
                 shortest paths): for every graph road the share of tasks whose loaded run passes it, the share with block
                 width B > W (capped by Eq. (1), B + c > C_r) and the share with B + c > W; under the ban (G = W, and the
                 passable widths read from imagery) the shares of tasks without a feasible route and of rerouted tasks (by
                 flow).
                 -> <out>/roads_<yard>.csv, summary.json
  --part bounds  system T1' (Eq. (5), the tail of the empty run cut): both yards x main mix / erection-heavy mix / crane-paced
                 erection x whole roads x ban level (none = no ban, recomputed through the subclasses of width_ban, which must
                 equal the stored system bound of the same case to the last bit, --earlier; road = passable width taken as the
                 road width; img = passable widths read from imagery) x handling of tasks without a feasible route (redraw,
                 clamp). One file per case in <out>/parts/, resumable. -> <out>/bounds.csv, summary.json
Run:   python experiments/width_ban_bounds.py --part roads --out <dir>
       python experiments/width_ban_bounds.py --part bounds --out <dir> --earlier results/bounds/sensitivity.csv [--procs 4]
--earlier: a table with the columns yard, scen, section, variant, T1_sys of the system bounds without the ban (the stored
results/bounds/sensitivity.csv; rows whole / base). The passable widths are those of runner.clear_widths
(results/width_ban/clear_width.json).
"""
import argparse
import collections
import csv
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import runner as F                                                # noqa: E402
import width_ban as BAN                                           # noqa: E402
from bounds.system import SystemT1, bl_joint, od_mix, scenario_mix, stop_ret   # noqa: E402
from occupancy import params as P                                 # noqa: E402
from occupancy.network import Net                                 # noqa: E402

YARDS = ("yupu", "yantai")
SCENS = ("main", "erection", "crane")
# the binding elements of the two yards (the paper's table of the bounds)
BINDING = {"yupu": {"road_trace_027": "main and erection-heavy mixes bind (road 161 at Dock 1)",
                    "road_trace_044": "crane-paced erection binds (through corridor 010-6)"},
           "yantai": {"road_trace_007": "erection-heavy mix binds (road 007 at Dock 2)"}}


def scen(yard, name):
    """Parameters of the three mixes: P.DEFAULTS with the main or erection-heavy scenario, or crane-paced erection with the
    yard's peak P6 per day and peak demand (as the stored crane-paced runs)."""
    s = dict(P.SCENARIOS[name])
    if name == "crane":
        s.update(p6_per_day=P.CRANE_YARD[P.base(yard)]["p6_peak"], erect_peak=P.YARDS[yard].get("demand", P.DEMAND)["peak"])
    return dict(P.DEFAULTS, **s)


def clear_map():
    """Passable widths read from imagery (runner.clear_widths): {yard: {graph road: G}}."""
    return F.clear_widths()


def levels(yard):
    return {"road": "road", "img": clear_map()[P.base(yard)]}


def make_config(part):
    return dict(part=part, yards=list(YARDS), scens=list(SCENS), networks={y: P.YARDS[y]["network"] for y in YARDS},
                clear={y: clear_map()[y] for y in YARDS}, params=P.DEFAULTS)


def net_of(yard, p, ban=None):
    Y = P.YARDS[yard]
    return (BAN.BanNet if ban else Net)(Y["network"], dict(p, **({"width_ban": ban} if ban else {})), Y["roles"])


# ---------------------------------------------------------------- roads passed by blocks wider than the road
def roads(yard):
    """(rows per road, summary). Shares = by the sampling law of the task stream, the share of tasks whose loaded run passes
    the graph road (lifts inside a crane span not counted)."""
    p = scen(yard, "main")
    net = net_of(yard, p)
    bnet = net_of(yard, p, {"clear": "road", "infeasible": "redraw"})
    inet = net_of(yard, p, {"clear": clear_map()[yard], "infeasible": "redraw"})
    img = [0.0, 0.0]                                                 # imagery level: without a feasible route, rerouted
    pB = collections.Counter()
    for (B, _), q in bl_joint(p):
        pB[B] += q
    c = p["lat_clear"]
    W = {r["orig"]: r["W"] for r in net.road.values() if r["W"] is not None}
    thru, over, over_c = collections.Counter(), collections.Counter(), collections.Counter()
    tot = any_over = infeas = rerouted = 0.0
    by_flow = collections.defaultdict(lambda: [0.0, 0.0, 0.0])       # flow -> [share, without a feasible route, rerouted]
    for w, t in od_mix(net, p):
        if t.get("lift"):
            continue
        so, sd = BAN.stop_nodes(net, t)
        route = net.path(so, sd)
        rs = {net.road[net.arc[a][3]]["orig"] for a in route}
        tot += w
        by_flow[t["flow"]][0] += w
        wmin = min((W[r] for r in rs if r in W), default=float("inf"))
        for B, q in pB.items():
            any_over += w * q * (B > wmin)
            if not bnet.reach(so, sd, B):
                infeas += w * q
                by_flow[t["flow"]][1] += w * q
            elif bnet.route(so, sd, B) != tuple(route):
                rerouted += w * q
                by_flow[t["flow"]][2] += w * q
            if not inet.reach(so, sd, B):
                img[0] += w * q
            elif inet.route(so, sd, B) != tuple(route):
                img[1] += w * q
        for r in rs:
            thru[r] += w
            if r in W:
                over[r] += w * sum(q for B, q in pB.items() if B > W[r])
                over_c[r] += w * sum(q for B, q in pB.items() if B + c > W[r])
    stops = collections.defaultdict(list)
    for st in net.stops.values():
        if st["node"].startswith("S|"):
            stops[st["node"].split("|")[1]].append(net.ents[st["ent"]]["name"])
    L = collections.Counter()
    for r in net.road.values():
        L[r["orig"]] += r["L"]
    rows = [dict(road=r, name=net.D["roads"][r]["name"], W=W[r], L_m=round(L[r], 1), stops="; ".join(sorted(set(stops.get(r, [])))),
                 through=thru[r] / tot, over=over[r] / tot, over_c=over_c[r] / tot, binding=BINDING[yard].get(r, ""))
            for r in W if over_c[r] > 0]
    rows.sort(key=lambda x: (-x["over"], -x["over_c"], x["road"]))
    for i, x in enumerate(rows, 1):
        x["rank"] = i
    tot_over = sum(x["over"] for x in rows)
    cum = 0.0
    for x in rows:
        cum += x["over"]
        x["cum_over_share"] = cum / tot_over if tot_over else 0.0
    summ = dict(yard=yard, any_over=any_over / tot, infeasible_road=infeas / tot, rerouted_road=rerouted / tot,
                infeasible_img=img[0] / tot, rerouted_img=img[1] / tot,
                roads_over=sum(x["over"] > 0 for x in rows), roads_over_c=len(rows), over_passages_per_task=tot_over,
                by_flow={f: dict(share=v[0] / tot, infeasible=v[1] / tot, rerouted=v[2] / tot) for f, v in sorted(by_flow.items())})
    return rows, summ


def part_roads(cfg, out):
    summ = {}
    for y in YARDS:
        rows, summ[y] = roads(y)
        keys = ["rank", "road", "name", "W", "L_m", "stops", "through", "over", "over_c", "cum_over_share", "binding"]
        with open(out / f"roads_{y}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, keys)
            w.writeheader()
            w.writerows(rows)
        print(y, json.dumps(summ[y], ensure_ascii=False), flush=True)
    F.write_json(out / "summary.json", summ)


# ---------------------------------------------------------------- system T1'
def cases():
    return [(y, s, lv, h) for y in YARDS for s in SCENS for lv, hs in (("none", ("-",)), ("road", ("redraw", "clamp")), ("img", ("redraw", "clamp")))
            for h in hs]


def job(a):
    out, (y, s, lv, h) = a
    F.pin(list(range(1, mp.cpu_count())))
    t0 = time.time()
    p = scen(y, s)
    ban = None if lv == "none" else {"clear": levels(y)[lv], "infeasible": h}
    if ban is None:                       # no ban: width_ban's BanLoads with nothing banned, equal to SystemT1 bit for bit
        net = net_of(y, p, {"clear": {}, "infeasible": "redraw"})
        net.G = {}
        mix, fixed = BAN.ban_mix(net, dict(p, width_ban={"clear": {}, "infeasible": "redraw"}))
        st = BAN.BanSystemT1(net, p, mix, fixed)
    else:
        net = net_of(y, p, ban)
        pb = dict(p, width_ban=ban)
        mix, fixed = BAN.ban_mix(net, pb)
        st = BAN.BanSystemT1(net, pb, mix, fixed)
    ret = stop_ret(net, p["plat_len"] + p["long_clear"])
    v, b = st.value(top=5, ret=ret)
    w0 = sum(w for w, t, _ in scenario_mix(net, p)[0])
    res = dict(yard=y, scen=s, level=lv, handling=h, T1_sys=v, mix_weight=sum(w for w, *_ in mix), mix_weight0=w0,
               binding=[dict(name=x["name"], res=x["res"], price_per_h=x["price_per_h"], util=x["util"]) for x in b], secs=round(time.time() - t0, 1))
    F.write_json(out / "parts" / f"{y}__{s}__{lv}__{h}.json", res)
    return y, s, lv, h, res["secs"]


def aggregate(cfg, out, earlier):
    """bounds.csv and summary.json; earlier: the table of the system bounds without the ban (whole roads, variant base)."""
    parts = {tuple(f.stem.split("__")): json.loads(f.read_text(encoding="utf-8")) for f in (out / "parts").glob("*.json")}
    before = {(r["yard"], r["scen"]): float(r["T1_sys"]) for r in csv.DictReader(open(earlier, encoding="utf-8-sig"))
              if r["section"] == "whole" and r["variant"] == "base"}
    rows = []
    for y, s, lv, h in cases():
        r = parts.get((y, s, lv, h))
        if r is None:
            continue
        base = parts.get((y, s, "none", "-"))
        b = r["binding"] + [{}, {}]
        rows.append(dict(yard=y, scen=s, level=lv, handling=h, T1_sys=r["T1_sys"], rel_none=r["T1_sys"] / base["T1_sys"] - 1 if base else None,
                         top1=b[0].get("name"), price1=b[0].get("price_per_h"), top2=b[1].get("name"), price2=b[1].get("price_per_h"),
                         util2=b[1].get("util"), earlier=before.get((y, s)) if lv == "none" else None,
                         same_as_earlier=(r["T1_sys"] == before[(y, s)]) if lv == "none" else None, secs=r["secs"]))
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "bounds.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    F.write_json(out / "summary.json", dict(rows=rows, n_parts=len(parts), earlier=Path(earlier).name))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--part", choices=("roads", "bounds"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--earlier", default=None, help="system bounds without the ban (yard, scen, section, variant, T1_sys); "
                                                    "required for --part bounds")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    if a.part == "bounds" and not a.earlier:
        ap.error("--part bounds needs --earlier")
    cfg = make_config(a.part)
    out = Path(a.out)
    if (out / "config.json").exists() and json.loads((out / "config.json").read_text(encoding="utf-8")) != json.loads(json.dumps(cfg, ensure_ascii=False)):
        sys.exit(f"refusing to reuse {out}: its config.json differs from this configuration")
    (out / "parts").mkdir(parents=True, exist_ok=True)
    F.write_json(out / "config.json", cfg)
    print(f"run directory: {out}", flush=True)
    if a.part == "roads":
        return part_roads(cfg, out)
    todo = [c for c in cases() if not (out / "parts" / ("__".join(c) + ".json")).exists()]
    prog = F.Progress(out / "progress.json", len(cases()), len(cases()) - len(todo), a.procs, "System bound under the ban", "ban")
    with mp.Pool(a.procs) as pool:
        for y, s, lv, h, secs in pool.imap_unordered(job, [(out, c) for c in todo]):
            prog.log(f"{y} {s} {lv} {h} ({secs:.0f} s)", done=prog.d["done"] + 1)
    prog.write(state="done")
    aggregate(cfg, out, a.earlier)
    print("done", flush=True)


if __name__ == "__main__":
    main()

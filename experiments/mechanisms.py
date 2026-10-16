"""Mechanism readings of the main case (conserved inter-area mapping) against the proportional mapping (paper, appendix of
the instances: erections within the cranes of each dock; supplementary material, onset of saturation and required fleets: why
the main case needs more vehicles at the 16-h peak; the plateau of full-path reservation under the two mappings). Reads stored
results and the networks, and builds the task-law loads of two system bounds.
  crane_dock.csv  per dock, deliveries to the dock / crane capacity of the dock (16 h): the dock shares of the erection
                  destinations from the task law (the P6 law of bounds.system.od_mix summed per dock; the task stream draws the
                  entities equally likely), the cranes from the network (params.CRANE, crane_n = "map"), capacity = cranes
                  x 16 h / 190 min; conserved mapping (2 % of the moves go to the docks) and proportional mapping
                  (0.2 / 3.45), Okpo 339, 350, 600, 678 and 700 moves/day, Yantai 23 and 46.
  loaded.csv      loaded occupancy per move (without a linear programme: the largest expected occupancy a-bar[k, r] over the
                  packing rows, s per move; unique for a task law, independent of degenerate solutions of lambda^S): the roads
                  around Okpo's through corridor and road 161, conserved against proportional; also the loaded utilisation at
                  each mapping's own lambda^S, lambda^S a-bar / H.
  waits.csv       simulated waiting attributed to resources (wait_by of Sim.metrics, min/day; mean of 10 seeds): 678 moves/day,
                  16 h, full-path reservation, K = 121, conserved against proportional. The two rings of the dock-mouth zone
                  (dock.ring0 / ring1) and road 161 (watch, complete); the other roads from the top 5 of each point (wait_top),
                  a seed in whose top 5 a road is missing counting 0, so these are lower bounds, with the number of seeds that
                  have the road in their top 5 (waits_by_resource.csv).
  platform.csv    plateau of full-path reservation (mean over K = 100-150) paired by seed: conserved minus proportional, mean
                  +- standard error.
Run:   python experiments/mechanisms.py --out <dir> --mapping-bounds <system.csv> --conserved-scan <scan.csv>
           --peak-current <run dir> --peak-conserved <run dir>
           --plat-current-yupu <run dir> --plat-conserved-yupu <run dir>
           --plat-current-yantai <run dir> --plat-conserved-yantai <run dir>
--mapping-bounds: system.csv of mapping_bounds.py or the released results/intra_area/bounds.csv (row Okpo, mapping main,
handling 10 min, 16 h: lambda^S of the proportional mapping); --conserved-scan: scan.csv of conserved_bounds.py or the released
results/main_case/scan.csv (Okpo, q = 0: lambda^S of the conserved mapping); --plat-*: run directories whose points.csv holds
the saturated full-path reservation points of K = 100-150 (proportional mapping, scenario main: results/runs/okpo_rules and
results/runs/yantai_rev395_saturation; conserved, scenario main_c: results/runs/okpo_main_saturation and
results/runs/yantai_main_saturation).
--peak-*: waits() reads the full point results (wait_top, watch, dock.ring1), which points.csv does not hold, so the ten points
(Okpo, full-path reservation, level n678 = 678 moves/day, K = 121, seeds 101-110, 16-h day, whole roads, main mix, handling
10 min) are recomputed: a configuration with network okpo/rev1833, roles {}, params = occupancy.params.DEFAULTS, scenarios
{<scen>: <its parameters>}, demand {"n678": 678}, models {<scen>: ["reserve"]}, ks [121], seeds 101-110, n_mix 2000, bounds
false, run as runner.py <config.json> --out <dir>; --peak-* names <dir> (waits() reads
<dir>/points/<scen>_reserve_n678_121_<seed>.json). The points are those of the required-fleet searches:
  --peak-conserved  scenario d16_c = {"demand_map": "conserved"}, as in results/runs/okpo_main_fleet/config.json;
  --peak-current    scenario d16 = {} (proportional mapping), from the required-fleet run at the peak, which is not released;
                    these points rebuild the "current" row of results/main_case/peak_waits.csv.
Writes crane_dock.csv, loaded.csv, waits.csv, waits_by_resource.csv, platform.csv and mech.json in <dir>.
"""
import argparse
import collections
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import runner as F                           # noqa: E402
import mapping_bounds as B                   # noqa: E402
import conserved_bounds as TB                # noqa: E402
from occupancy import params as P            # noqa: E402

YARDS = ("yupu", "yantai")
LEVELS = {"yupu": (339, 350, 600, 678, 700), "yantai": (23, 46)}
SHARE = {"current": 0.2 / 3.45, "conserved": 0.02}
ROADS = {"010-6": "road_trace_044", "026": "road_3477f15b-70d4-450e-ba80-a35d6273db92", "099-4": "road_trace_145", "098-1": "road_trace_118",
         "005-3": "road_trace_661", "013-5": "road_trace_547", "013-7": "road_trace_454", "102": "road_trace_122", "161": "road_trace_027"}
PEAK = dict(level="n678", K=121, current="d16", conserved="d16_c")
PLAT = {"yupu": dict(current="main", conserved="main_c"), "yantai": dict(current="main", conserved="main_c")}
PLAT_KS = tuple(range(100, 151, 5))
OUT = None                                   # output directory (--out)


def write(name, rows):
    with open(OUT / name, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def crane_dock():
    from bounds.system import od_mix
    rows = []
    p = dict(P.DEFAULTS)
    pc = dict(P.DEFAULTS, **P.CRANE)
    for y in YARDS:
        net = B.net_of(y, p)
        share = collections.Counter()
        for w, t in od_mix(net, dict(p, flow_share={"P6": 1.0})):
            share[net.stops[t["d"]]["ent"]] += w
        n = net.crane_counts(pc)
        for z in net.by_cls[P.DOCK_CLASS]:
            cap = n[z] * pc["day_s"] / pc["crane_cycle_s"]
            row = dict(yard=y, dock=z, dock_name=net.ents[z]["name"], cranes=n[z], cap16=cap, dest_share=share[z])
            for m, s in SHARE.items():
                for lv in LEVELS[y]:
                    row[f"{m}_{lv}"] = lv * s * share[z] / cap
            rows.append(row)
        tot = sum(n.values()) * pc["day_s"] / pc["crane_cycle_s"]
        row = dict(yard=y, dock="all", dock_name="whole yard", cranes=sum(n.values()), cap16=tot, dest_share=sum(share.values()))
        for m, s in SHARE.items():
            for lv in LEVELS[y]:
                row[f"{m}_{lv}"] = lv * s / tot
        rows.append(row)
    return rows


def loaded(lam_current, lam_conserved):
    """lam_current: system.csv of the system bounds under the demand mappings; lam_conserved: scan.csv of the system bounds of
    the conserved mapping."""
    from bounds.system import SystemT1, bl_joint, od_mix
    p = dict(P.DEFAULTS)
    # the network of the conserved saturation runs of Okpo (their configuration, scenario main_c)
    cfg = dict(network=P.YARDS["yupu"]["network"], roles=P.YARDS["yupu"]["roles"], params=P.DEFAULTS,
               scenarios={"main_c": {"demand_map": "conserved"}})
    net = F.net_for(cfg, F.scen_params(cfg, "main_c"))[0]
    bl = bl_joint(p)
    st = {k: SystemT1(net, p, [(w, t, bl) for w, t in od_mix(net, dict(p, flow_share={"P6": v}))]) for k, v in (("P6", 1.0), ("ND", 0.0))}
    cap = st["P6"].cap
    A = {k: s.L.abar(cap) for k, s in st.items()}
    rl = st["P6"].res_list
    lam = {"current": float(next(r["T1_sys"] for r in csv.DictReader(open(lam_current, encoding="utf-8-sig"))
                                 if (r["yard"], r["map"], r["load_min"], r["day_h"]) == ("yupu", "main", "10", "16"))),
           "conserved": TB.q0_lam(lam_conserved)["yupu"]}
    rows = []
    for name, rid in ROADS.items():
        i = rl.index(("r", rid))
        per = {m: float((c * A["P6"] + (1 - c) * A["ND"])[:, i].max()) for m, c in SHARE.items()}
        rows.append(dict(road=name, res=f"r|{rid}", per_move_current_s=per["current"], per_move_conserved_s=per["conserved"],
                         rel_change=per["conserved"] / per["current"] - 1,
                         loaded_util_current=lam["current"] * per["current"] / p["day_s"], loaded_util_conserved=lam["conserved"] * per["conserved"] / p["day_s"]))
    return rows, lam


def waits(dirs):
    """dirs = {"current": run directory, "conserved": run directory} with the point files of PEAK in points/."""
    rows, tops = [], {}
    for m in ("current", "conserved"):
        d, s = dirs[m], PEAK[m]
        xs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((Path(d) / "points").glob(f"{s}_reserve_{PEAK['level']}_{PEAK['K']}_*.json"))]
        assert len(xs) == 10, (d, len(xs))
        mean = lambda f: sum(f(x) for x in xs) / len(xs)
        top, seen = collections.Counter(), collections.Counter()
        for x in xs:
            for w in x["wait_top"]:
                key = w["name"].split(" (")[0]
                top[key] += w["value"] / len(xs)
                seen[key] += 1
        tops[m] = (top, seen)
        rows.append(dict(map=m, dir=Path(d).name, scen=s, K=PEAK["K"], level=PEAK["level"], throughput=mean(lambda x: x["throughput"]),
                         wait_total=mean(lambda x: x["wait_attr_min_per_day"]), ring0=mean(lambda x: x["dock"]["ring0"]["wait_min_per_day"]),
                         ring1=mean(lambda x: x["dock"]["ring1"]["wait_min_per_day"]), ring_all=mean(lambda x: x["dock"]["all"]["wait_min_per_day"]),
                         road161=mean(lambda x: x["watch"].get("r|road_trace_027", [0, 0, 0])[2])))
    names = sorted(set(tops["current"][0]) | set(tops["conserved"][0]), key=lambda k: -max(tops["current"][0][k], tops["conserved"][0][k]))
    res = [dict(resource=k, current=tops["current"][0][k], current_in_top5=tops["current"][1][k], conserved=tops["conserved"][0][k],
                conserved_in_top5=tops["conserved"][1][k]) for k in names]
    return rows, res


def platform(dirs):
    """dirs = {yard: {"current": run directory, "conserved": run directory}} whose points.csv holds the saturated points."""
    rows = []
    for y in YARDS:
        per = {}
        for m, s in PLAT[y].items():
            pts = {}
            for x in csv.DictReader(open(Path(dirs[y][m]) / "points.csv", encoding="utf-8-sig")):
                if (x["scen"], x["model"], x["level"]) == (s, "reserve", "saturated") and int(x["K"]) in PLAT_KS:
                    pts[(int(x["K"]), int(x["seed"]))] = float(x["throughput"])
            per[m] = {sd: sum(pts[(K, sd)] for K in PLAT_KS) / len(PLAT_KS) for sd in F.SEEDS}
        diff = [per["conserved"][sd] - per["current"][sd] for sd in F.SEEDS]
        mu = sum(diff) / len(diff)
        se = (sum((v - mu) ** 2 for v in diff) / (len(diff) - 1) / len(diff)) ** 0.5
        rows.append(dict(yard=y, plat_current=sum(per["current"].values()) / len(F.SEEDS), plat_conserved=sum(per["conserved"].values()) / len(F.SEEDS),
                         diff_mean=mu, diff_se=se, t=mu / se, n=len(diff), diffs=json.dumps([round(v, 2) for v in diff])))
    return rows


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--mapping-bounds", required=True, help="system.csv of mapping_bounds or results/intra_area/bounds.csv")
    ap.add_argument("--conserved-scan", required=True, help="scan.csv of conserved_bounds or results/main_case/scan.csv")
    for m in ("current", "conserved"):
        ap.add_argument(f"--peak-{m}", required=True, help=f"run directory with the peak points ({m} mapping) in points/")
        for y in YARDS:
            ap.add_argument(f"--plat-{m}-{y}", required=True, help=f"run directory, saturated points ({m} mapping, {y})")
    a = ap.parse_args()
    OUT = Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    cd = crane_dock()
    write("crane_dock.csv", cd)
    print("crane_dock", flush=True)
    ld, lam = loaded(a.mapping_bounds, a.conserved_scan)
    write("loaded.csv", ld)
    print("loaded", flush=True)
    wr, wres = waits({m: getattr(a, f"peak_{m}") for m in ("current", "conserved")})
    write("waits.csv", wr)
    write("waits_by_resource.csv", wres)
    print("waits", flush=True)
    pl = platform({y: {m: getattr(a, f"plat_{m}_{y}") for m in ("current", "conserved")} for y in YARDS})
    write("platform.csv", pl)
    F.write_json(OUT / "mech.json", dict(crane_dock=cd, loaded=ld, lam=lam, waits=wr, waits_by_resource=wres, platform=pl))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()

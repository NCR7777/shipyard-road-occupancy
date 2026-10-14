"""Write numbers.tex (one LaTeX macro per number of the text) and tables/*.tex from the result data of the repository.

Every number is read from results/ (paths relative to the repository root, read through src.py; results/MANIFEST.csv
lists each file) and checked with assertions as it is read. Sources, by directory of results/:
  network/         topology of the two yard networks (okpo_topology.json, yantai_topology.json), the Okpo map revisions
                   (okpo_revisions.json), sectioned roads, roads shorter than a vehicle (short_roads.csv), crane spans
  bounds/          system, route-free and sampled bounds of each yard ({okpo,yantai}/: system.csv, route_free.csv,
                   convergence.csv, readout.csv, prices.csv, config.json); system bounds by granularity
                   (system_by_granularity.csv), their sensitivity to the inputs, sampling of origins and destinations
  saturation/      saturation curves, plateaus and per-point checks (rules_summary.json; Okpo on rev 1833:
                   okpo_curves.csv, okpo_readings.csv, okpo_curves_check.json), onset of saturation K* (kstar.csv), all
                   rules at saturation (rules.csv), refusal and teleport diagnostics, sensitivity of the plateaus
  interval/        capacity interval (interval.csv), crane-paced interval with Okpo's 16-day plateau and Yantai's
                   eight-day check, 30-min handling, erection load
  fleet/           required fleets (required_fleet.csv, with fleet_8day.csv for Okpo's headline cells; src.demand), the
                   K* marks, crane-paced and 30-min fleets, the 10 % screen (screen.csv, screen_cross.csv, screen_band.json)
  dock/            dock-mouth thresholds (thresholds.csv, summary.json), crane hours, utilisation levels, one crane at Dock 1
  anchors/         blocks per vehicle-day against handling time and by rule (the external anchor)
  synthetic_dock/  synthetic dock mouth: thresholds rho_10 (rho_star.csv), per-point throughput, waiting term
  runs/<run>/      configuration (config.json), per-point results (points.csv) and per-point bound check (check.json) of
                   the simulation batches; the run names are in src.py
The model parameters, task flows and the geometry of the synthetic dock mouth are read from occupancy/params.py,
occupancy/tasks.py and experiments/dock_mouth.py.
Okpo's readings are on map revision 1833 (with the floating docks). Some inputs come from runs on revision 1825, which
differs only by three spur roads at the floating docks (model parameters, bounds, utilisation levels, per-point checks);
where a reading differs between the revisions, the rev 1833 value is used and the other is only checked.
Only the macros and tables that the paper uses are written: the macros of the text, the tables and the Highlights, those
the figures check, and the computations and checks they rest on. The tables are those printed: tab_bounds,
tab_fleethead, tab_flows.

Run from paper/: python scripts/make_numbers.py (or, from the repository root, python paper/reproduce.py).
"""
import re
import statistics
from pathlib import Path

from src import (BOUNDS_DIR, CRANE_FLEET, CRANE_RUNS, FLAG_RUNS, MAIN_SAFE, OD_RUNS, RHO_SAFE, SEC_FLEET, SYN_DOCK, SYN_WAIT,
                 demand, interval, js, res_name, rows, show)

ROOT = Path(__file__).resolve().parents[1]
YARDS = ("yupu", "yantai")
MAC = {}            # macro name -> LaTeX text


def fmt(x, nd=0):
    s = f"{float(x):.{nd}f}"
    if abs(float(s)) >= 10000:                       # AMM: a thousands separator from five digits on, none in 1736
        s = f"{float(x):,.{nd}f}"
    return s.replace(",", "{,}")


def put(name, value):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    assert name not in MAC, name
    MAC[name] = value


def num(name, x, nd=0):
    put(name, fmt(x, nd))


def pct(name, x, nd=0):          # x is a fraction
    put(name, fmt(100 * float(x), nd) + r"\%")


Y = {"yupu": "Yu", "yantai": "Ya"}


# ---------------------------------------------------------------- saturation summary (rules_summary.json)
S = js("results/saturation/rules_summary.json")
SAT = S["sat"]
# onset of saturation K* (kstar.csv), read on the saturation curves of Figure 4 (Okpo rev 1833, Yantai the curve of
# rules_summary.json) at 20 % (the criterion of the paper), 10 % and 30 %; the interval is (lo, hi). The point estimates
# are those of rules_summary.json (50 and 25); Okpo's interval 45-55 is read on rev 1833 (the curve of the earlier map,
# yupu_rev1824, gives 50-60)
KST = {(r["curve"], float(r["theta"])): r for r in rows("results/saturation/kstar.csv")}
KSTAR = {y: (int(KST[(y, 0.2)]["lo"]), int(KST[(y, 0.2)]["hi"])) for y in YARDS}
assert {y: int(KST[(y, 0.2)]["K_star"]) for y in YARDS} == {"yupu": 50, "yantai": 25} == \
    {y: S["K_star"][y][0] for y in YARDS}
assert KSTAR == {"yupu": (45, 55), "yantai": (25, 25)} and KST[("yupu_rev1824", 0.2)]["lo"] == str(S["K_star"]["yupu"][0])


def rule(block, r):
    return SAT[block]["rules"][r]


# topology and parameters
topo = {"yupu": js("results/network/okpo_topology.json")["yards"]["yupu"]["topology"],
        "yantai": js("results/network/yantai_topology.json")["change"]["r395"]["topology"]}
sections = rows("results/network/sectioned_roads.csv")
# Okpo on the map with floating docks (rev 1833, the map of the headline numbers): three short dead-end roads more than
# rev 1825, with the same node resources and stops (okpo_revisions.json); road count and length from short_roads.csv,
# which counts the same map
mapdiff = js("results/network/okpo_revisions.json")["None"]
SHORT = {(r["yard"], r["gran"]): r for r in rows("results/network/short_roads.csv")}
for y in YARDS:
    t = topo[y]
    cut = [s for s in sections if s["yard"] == {"yupu": "Okpo", "yantai": "Yantai"}[y]]
    nsec = sum(int(s["n"]) - 1 for s in cut)
    whole, sect = SHORT[(y, "whole")], SHORT[(y, "stops")]
    if y == "yupu":
        assert mapdiff["revision"] == [1825, 1833] and mapdiff["road_res"] == [t["roads"], int(whole["n_roads"])]
        assert mapdiff["node_res"][0] == mapdiff["node_res"][1] and mapdiff["stops_same"] and whole["map"] == "rev 1833"
        roads, km = int(whole["n_roads"]), float(whole["km"])
    else:
        roads, km = t["roads"], t["km"]
        assert roads == int(whole["n_roads"]) and abs(km - float(whole["km"])) < 0.05
    assert int(sect["n_roads"]) == roads + nsec
    num(Y[y] + "Roads", roads)
    num(Y[y] + "RoadKm", km, 1)
    num(Y[y] + "NodeRes", t["junction_res"] + t["access"])
    num(Y[y] + "Stops", t["stops"])
    num(Y[y] + "RoadsSectioned", roads + nsec)

P = js(MAIN_SAFE["yupu"] + "/config.json")["params"]
num("SpeedEmpty", P["v_empty"] * 3.6)
num("SpeedLoaded", P["v_loaded"] * 3.6)
num("PlatLen", P["plat_len"], 2)
num("PlatW", P["plat_w"], 1)
num("LatClear", P["lat_clear"])
num("LongClear", P["long_clear"])
num("TurnSweep", P["turn_sweep_s"])
num("TurnDeg", P["turn_deg"])
num("LoadMin", P["load_s"] / 60)
num("BlockWLo", P["block_w"][0])
num("BlockWHi", P["block_w"][1])
num("BlockLLo", P["block_l"][0])
num("BlockLHi", P["block_l"][1])
num("FreeWidth", P["unrestricted_w"])
num("FreeWidthB", P["unrestricted_w"] / 2 - P["lat_clear"], 1)    # widest block for which two loaded vehicles meet there
num("DayH", P["day_s"] / 3600)
num("SimDays", P["days"])
num("Batches", P["batches"])
num("RecoveryS", P["deadlock_recovery_s"])
num("SectionMin", P["section_min_m"])
num("SectionGap", P["section_gap_m"])
num("Seeds", len(js(MAIN_SAFE["yupu"] + "/config.json")["seeds"]))
num("MixN", js(MAIN_SAFE["yupu"] + "/config.json")["n_mix"])
# widest junction for which every clearing time stays within loading and unloading (occupancy lemma of the appendix):
# loaded passage of the longest vehicle with a swept turn, (D + L + s)/v1 + sigma <= min(t_L, t_U); rounded down to 100 m
_t = min(P["load_s"], P["unload_s"]) - P["turn_sweep_s"]
_L = max(P["block_l"][1], P["plat_len"])
num("ClearJunctionMax", (_t * P["v_loaded"] - _L - P["long_clear"]) // 100 * 100)
assert _t * P["v_empty"] - P["plat_len"] - P["long_clear"] >= _t * P["v_loaded"] - _L - P["long_clear"]   # empty passage is not tighter
assert (_L + P["long_clear"]) / P["v_loaded"] <= _t                                                    # nor is a tail clearance

# cranes and erection cadence
C = js(RHO_SAFE["yupu"] + "/config.json")["scenarios"]["crane_8d"]
cycle_min = C["crane_cycle_s"] / 60
num("CraneCycleMin", cycle_min)
num("CraneHookMin", C["crane_hook_s"] / 60)
num("CraneSetMin", cycle_min - C["crane_hook_s"] / 60)
num("RhoSimDays", C["days"])
spans = js("results/network/crane_spans.json")["spans"]
for y, zh in (("yupu", "Okpo"), ("yantai", "Yantai")):
    docks = {s["dock"]: s["cranes"] for s in spans if s["yard"] == zh}
    n = sum(docks.values())
    p6 = js(RHO_SAFE[y] + "/config.json")["scenarios"]["crane_8d"]["p6_per_day"]
    num(Y[y] + "Cranes", n)
    num(Y[y] + "ErectPerDay", p6, 1 if y == "yupu" else 0)
    num(Y[y] + "Rho", p6 * cycle_min / (n * P["day_s"] / 60), 2)
# erections per peak day = 2 (peak factor, as for demand) x LNG carriers a year x erection units / working days
# (CRANE_YARD in occupancy/params.py)
p6yu = js(RHO_SAFE["yupu"] + "/config.json")["scenarios"]["crane_8d"]["p6_per_day"]
assert abs(p6yu - 2 * 20 * 100 / 300) < 1e-9 and "p6_peak=2 * 20 * P6_UNITS[1] / 300" in show("occupancy/params.py")
num("PeakFactor", 2)
num("YuErectAvg", 20 * 100 / 300, 1)

# saturation curves, plateaus, K*: Yantai from rules_summary.json; Okpo on the map with floating docks
# (okpo_readings.csv, column r1833)
SEC61 = {r["macro"]: r for r in rows("results/saturation/okpo_readings.csv")}
kst = next(r for m, r in SEC61.items() if "K*" in m)
assert float(kst["r1833"]) == float(kst["r1825"]) == int(KST[("yupu", 0.2)]["K_star"])   # point estimate on both maps
for y in YARDS:
    b = f"{y}|main"
    for r, tag, suf, col in (("whole|reserve", "Res", "Plateau", "plateau"), ("whole|reserve", "Res", "AtMaxK", "at_150"),
                             ("whole|segment_safe", "Safe", "Max", "max"), ("whole|segment", "Ref", "Plateau", "plateau")):
        name = Y[y] + tag + suf
        num(name, float(SEC61[name]["r1833"]) if y == "yupu" else rule(b, r)[col])
    num(Y[y] + "FreeSlope", float(SEC61["YuFreeSlope"]["r1833"]) if y == "yupu"
        else rule(b, "whole|free")["curve"]["10"] / 10, 1)
    num(Y[y] + "KStarLo", KSTAR[y][0])
num("YuKStarHi", KSTAR["yupu"][1])
# full-path reservation is still rising at the largest fleet simulated: its largest mean is the one at K = 150, above
# the plateau (mean over K = 100-150), and no step between adjacent fleets falls by 1 % or more (seed noise); so the
# plateau states the lower end conservatively (Okpo: map with floating docks, okpo_curves.csv; Yantai: the curves of
# rules_summary.json)
RES_CURVE = {"yupu": {int(r["K"]): float(r["mean"]) for r in rows("results/saturation/okpo_curves.csv")
                      if r["model"] == "reserve"},
             "yantai": {int(k): v for k, v in rule("yantai|main", "whole|reserve")["curve"].items()}}
for y, c in RES_CURVE.items():
    ks = sorted(c)
    plat = statistics.mean(c[k] for k in ks if k >= 100)
    assert max(c, key=c.get) == ks[-1] == 150 and MAC[Y[y] + "ResAtMaxK"] == fmt(c[150]) and c[150] > plat
    src_plat = float(SEC61["YuResPlateau"]["r1833"]) if y == "yupu" else rule("yantai|main", "whole|reserve")["plateau"]
    assert abs(plat - src_plat) < 0.6 and all(c[b] > 0.99 * c[a] for a, b in zip(ks, ks[1:]))
# waits of safe release that start with a refusal while capacity was available (refusal.csv: the formula of
# rules_summary.json on the current maps, Okpo rev 1833; its share_prev column reproduces rules_summary.json value by
# value)
REF = rows("results/saturation/refusal.csv")
shares22 = sorted(v for d in S["refusal"].values() for v in d.values())
assert len(REF) == len(shares22) and all(abs(a - b) < 1e-12 for a, b in zip(sorted(float(r["share_prev"]) for r in REF),
                                                                           shares22))
assert {r["map"] for r in REF if r["yard"] == "yupu"} == {"rev 1833"}
shares = [float(r["share"]) for r in REF]
pct("RefusalLo", min(shares))
pct("RefusalHi", max(shares))
nv = S["naive"]
num("SafeChecks", nv["checks"])
assert nv["mismatches"] == 0 and nv["deadlocks"] == 0

# per-instance check of the network bound over all physical-rule simulation points
batches = [RHO_SAFE["yupu"], RHO_SAFE["yantai"], "results/runs/okpo_rev1825_erection_safe",
           "results/runs/okpo_rev1825_dock_diagnosis", "results/runs/yantai_dock_diagnosis", MAIN_SAFE["yupu"],
           MAIN_SAFE["yantai"], "results/runs/okpo_rev1825_anchor", "results/runs/yantai_anchor"]
checks = [js(d + "/check.json")["findings"]["T1_tight_check"] for d in batches] + [S["t1_check_demand"]]
# required fleets rerun on the Okpo map with floating docks: the check of those runs (okpo_curves_check.json), and the
# sectioned full-path reservation fleets (SEC_FLEET), checked here point by point against the bound of each point's
# completed mix
check25f = js("results/saturation/okpo_curves_check.json")["t1_check"]
g_pts = [r for r in rows(SEC_FLEET + "/points.csv") if r["model"] != "free"]
g_ratio = [float(r["throughput"]) / float(r["T1_tight_mix"]) for r in g_pts]
check25g = {"points": len(g_pts), "max_ratio": max(g_ratio), "over": sum(x > 1 for x in g_ratio)}
checks += [check25f, check25g]
assert all(c.get("over_strict", c.get("over", 0)) == 0 for c in checks)

# dock mouth: thresholds (10 % drop) by K (thresholds.csv): Okpo on rev 1833 with the current code, Yantai from the
# earlier runs (reproduced by the current code; the sectioned reference rule from the current code); a cell reads eight
# simulated days, or 16 where the eight-day runs were not steady (thr_table, steady_table, days_table)
RK = {r["key"]: r for r in rows("results/dock/thresholds.csv")}
assert len(RK) == 60 and {r["map"] for k, r in RK.items() if k.startswith("yupu")} == {"rev 1833"}
for k, r in RK.items():
    assert (r["days_table"] == "16") == bool(r["thr10"] and r["thr10_steady"] != "True"), k   # 16 days where starred
    assert r["thr_table"] == (r["thr10_16d"] if r["days_table"] == "16" else r["thr10"]), k
    assert r["cell_tex"] == ("--" if not r["thr_table"] else f"{float(r['thr_table']):.2f}"
                             + ("" if r["steady_table"] == "True" else r"$^{*}$")), k
DOCK_SUM = js("results/dock/summary.json")["dock_checks"]
assert DOCK_SUM["rho_same"]                      # the utilisation levels are the same on both Okpo maps
# Section 4.4 and the conclusions: at Okpo the whole-road threshold of safe release does not rise with the fleet,
# sectioning raises it for both rules at every fleet, and the readings with 20 and 150 vehicles are 16-day readings
# whose throughput beyond the threshold kept falling
assert DOCK_SUM["safe_whole_falls"] and all(v["up"] == v["n"] for v in DOCK_SUM["sectioning_up"].values())
assert all(RK[f"yupu|whole|segment_safe|{K}"]["days_table"] == "16" and RK[f"yupu|whole|segment_safe|{K}"]["steady_table"]
           == "False" for K in (20, 150))


def thr(key):
    r = RK[key]
    return (float(r["thr_table"]), r["steady_table"] == "True") if r["thr_table"] else (None, True)


for y in YARDS:
    for sec, s, K, w in (("whole", "", 20, "Twenty"), ("whole", "", 40, "Forty"), ("whole", "", 150, "OneFifty"),
                         ("stops", "Sec", 40, "Forty")):
        v, _ = thr(f"{y}|{sec}|segment_safe|{K}")
        num(Y[y] + "SafeThr" + s + w, v, 2)
res_drop = max(-float(RK[k]["min_rel"]) for k in RK if "|reserve|" in k)
assert not any(RK[k]["thr_table"] for k in RK if "|reserve|" in k) and abs(res_drop - DOCK_SUM["reserve_max_drop_r1833"]) < 1e-9
put("ResMaxDrop", fmt(res_drop, 1) + r"\%")
# the seven thresholds that are steady over eight days, one by one
steady_keys = sorted(k for k in RK if RK[k]["steady_table"] == "True")
assert steady_keys == sorted(["yupu|whole|segment_safe|60", "yupu|whole|segment|150", "yantai|whole|segment_safe|60",
                              "yantai|stops|segment_safe|60", "yantai|stops|segment_safe|100", "yantai|stops|segment|60",
                              "yantai|whole|segment|150"]), steady_keys
# 16-day reruns of the thresholds that were not steady over eight days: three become steady and move little, in the
# others the drift persists; in most of those every level below the threshold is steady and every level beyond it keeps
# losing throughput (column reading_16d); over all of them the loss at the highest utilisation deepens from 8 to 16
# days and the threshold itself moves little
star = [r for r in RK.values() if r["days_table"] == "16"]
became = [r for r in star if r["thr10_steady_16d"] == "True"]
assert all(r["reading_16d"] == "steady, new value" for r in became)
assert all(r["reading_16d"].startswith("drift persists over 16 days") for r in star if r not in became)
above = [r for r in star if r["reading_16d"].endswith("(steady below the threshold, falling beyond it)")]
deepen = [float(r["min_rel"]) - float(r["min_rel_16d"]) for r in star]          # percentage points, positive = deeper
move = [float(r["thr10_16d"]) - float(r["thr10"]) for r in star]
assert min(deepen) > 0
num("DockStarred", len(star))
num("DockBecameSteady", len(became))
num("DockDrift", len(star) - len(became))
num("DockDriftAbove", len(above))
put("DockSteadyMove", fmt(max(abs(float(r["thr10_16d"]) - float(r["thr10"])) for r in became), 3))   # 0.0135: not 0.01
num("DockDeepenLo", min(deepen))
num("DockDeepenHi", max(deepen))
num("DockMoveMax", max(abs(m) for m in move), 2)
num("DockMoveMed", statistics.median(abs(m) for m in move), 2)

rho_rows = rows("results/dock/rho_levels.csv")
top = {}
for r in rho_rows:
    if r["scen"] == "rho15":
        top[(r["yard"], r["combo"])] = float(r["crane_util"])
# crane hours at the top utilisation with 150 vehicles: Okpo on rev 1833 (crane_hours.csv, whose r1825 column is the
# value of rho_levels.csv), Yantai from rho_levels.csv
for r in rows("results/dock/crane_hours.csv"):
    assert abs(float(r["crane_util_r1825"]) - top[("yupu", r["combo"])]) < 1e-9
    top[("yupu", r["combo"])] = float(r["crane_util_r1833"])
for y in YARDS:
    safe = [top[(y, "whole|segment_safe")], top[(y, "stops|segment_safe")]]
    resv = [top[(y, "whole|reserve")], top[(y, "stops|reserve")]]
    for tag, v in (("CraneSafe", safe), ("CraneRes", resv)):
        lo, hi = fmt(100 * min(v)), fmt(100 * max(v))
        put(Y[y] + tag, (lo if lo == hi else f"{lo}--{hi}") + r"\%")
num("RhoTop", max(float(r["rho"]) for r in rho_rows), 2)
# share of the reference rule's loaded erection tasks teleported at least once (teleport.csv: the diagnosis of
# teleport_previous.csv rerun on the current maps; Yantai value by value as there)
TEL = rows("results/saturation/teleport.csv")
tele22 = sorted(float(r["share_ever"]) for r in rows("results/saturation/teleport_previous.csv") if r["kind"] == "teleport")
assert len(TEL) == len(tele22) and all(abs(a - b) < 1e-12 for a, b in zip(sorted(float(r["share_ever_prev"]) for r in TEL),
                                                                            tele22))
assert all(r["same_as_stored"] == "True" for r in TEL)
tele = [float(r["share_ever"]) for r in TEL]
pct("TeleLo", min(tele))
pct("TeleHi", max(tele))

# ---------------------------------------------------------------- required fleet (Okpo on the map with floating docks)
DEM = demand()


def kreq(y, h, sec, model, n):
    for r in DEM:
        if (r["yard"], float(r["day_h"]), r["section"], r["model"], int(float(r["demand"]))) == (y, h, sec, model, n):
            return r
    return None


def kval(r):
    return r["K_req"]


for y, n, h, name in (("yupu", 350, 24.0, "YuFleetAnchorHTwentyFourRes"), ("yantai", 450, 16.0, "YaFleetFourFiftyHSixteenRes")):
    put(name, kval(kreq(y, h, "whole", "reserve", n)).replace(">", r"$>$"))
extra = [int(kreq("yupu", 16.0, "whole", "reserve", n)["K_req"]) / int(kreq("yupu", 16.0, "whole", "free", n)["K_req"]) - 1
         for n in (339, 350)]
pct("YuOwnExtraLo", min(extra))
pct("YuOwnExtraHi", max(extra))
# the conclusions say it in words: more than half on 16-h days, a quarter on 24-h days (eight-day values)
extra24 = [int(kreq("yupu", 24.0, "whole", "reserve", n)["K_req"]) / int(kreq("yupu", 24.0, "whole", "free", n)["K_req"]) - 1
           for n in (339, 350)]
assert min(extra) > 0.5 and all(abs(e - 0.25) < 1e-9 for e in extra24)
# headline fleets over eight simulated days (fleet_8day.csv, applied in src.demand): seven of the eight cells keep the
# four-day value; full-path reservation at 350 blocks/day on 24-h days needs one vehicle less, inside the four-day seed
# interval
F8 = rows("results/fleet/fleet_8day.csv")
d8 = {s["days"] for s in js(f"results/{F8[0]['batch']}/config.json")["scenarios"].values()}
assert len(d8) == 1 and {r["batch"] for r in F8} == {F8[0]["batch"]}
num("FleetEightDays", d8.pop())                     # simulated days of the headline-fleet reruns (parameter table)
ch8 = [r for r in F8 if r["changed"] == "True"]
assert len(F8) == 8 and len(ch8) == 1 and (ch8[0]["day_h"], ch8[0]["model"], ch8[0]["demand"]) == ("24", "reserve", "350")
assert int(ch8[0]["K_8d"]) == int(ch8[0]["K_4d"]) - 1 and ch8[0]["K_4d_ci"] == f"[{ch8[0]['K_8d']}, {ch8[0]['K_4d']}]"
assert MAC["YuFleetAnchorHTwentyFourRes"] == ch8[0]["K_8d"]
num("YuEightDaySame", len(F8) - len(ch8))
num("YuEightDayCells", len(F8))
put("YuFleetAnchorHTwentyFourResFourDay", ch8[0]["K_4d"])
# the onset of saturation at 10 % and 30 % instead of 20 % (kstar.csv), and the two readings of the paper that hold
# with either: Okpo's own volume (free flow and full-path reservation, both working days) stays below K*, its 16-h peak
# (full-path reservation) at or above its upper end
def kst_text(y, theta):
    r = KST[(y, theta)]
    return r["K_star"] if r["lo"] == r["hi"] == r["K_star"] else f"{r['K_star']} ({r['lo']}--{r['hi']})"


for y in YARDS:                                     # the criterion of the paper, written as the other two
    put(Y[y] + "KStarTwenty", kst_text(y, 0.2))
# the 16-h peak (full-path reservation) lies past K* and the 24-h peak within it, as Section 4.3 says
for h, where in ((16.0, "past"), (24.0, "within")):
    pk = [int(kreq("yupu", h, "whole", "reserve", n)["K_req"]) for n in (678, 700)]
    assert (min(pk) >= KSTAR["yupu"][1]) if where == "past" else all(KSTAR["yupu"][0] <= k <= KSTAR["yupu"][1] for k in pk), (h, pk)
for theta, tn in ((0.1, "Ten"), (0.3, "Thirty")):
    for y in YARDS:
        put(Y[y] + "KStar" + tn, kst_text(y, theta))
    own_k = [int(kreq("yupu", h, "whole", m, n)["K_req"]) for h in (16.0, 24.0) for m in ("free", "reserve") for n in (339, 350)]
    peak_k = [int(kreq("yupu", 16.0, "whole", "reserve", n)["K_req"]) for n in (678, 700)]
    assert max(own_k) < int(KST[("yupu", theta)]["lo"]) and min(peak_k) >= int(KST[("yupu", theta)]["hi"]), theta
# the bold and underline marks of the required-fleet table at 20 % are those of kstar_marks.csv (column mark_20)
DEM4 = demand(eight_day=False)
for r in rows("results/fleet/kstar_marks.csv"):
    t = next(d for d in DEM4 if (d["yard"], float(d["day_h"]), d["section"], d["model"], int(float(d["demand"])))
             == (r["yard"], float(r["day_h"]), "whole", r["model"], int(r["demand"])))
    assert t["K_req"] == r["K_req"], r
    k = int(r["K_req"]) if r["K_req"].isdigit() else 10 ** 6
    lo, hi = KSTAR[r["yard"]]
    assert r["mark_20"] == ("bold" if k >= hi else "underline" if k >= lo else ""), r
# sectioned against whole dock roads, both from four-day runs
kreq4 = lambda y, h, sec, model, n: next((r for r in DEM4 if (r["yard"], float(r["day_h"]), r["section"], r["model"],
                                                               int(float(r["demand"]))) == (y, h, sec, model, n)), None)
diff = [abs(int(kreq4("yupu", float(r["day_h"]), "whole", "reserve", int(float(r["demand"])))["K_req"]) - int(r["K_req"]))
        for r in DEM if r["yard"] == "yupu" and r["section"] == "stops" and r["model"] == "reserve"
        and r["K_req"].isdigit()]
num("SecFleetMaxDiff", max(diff))
assert max(diff) == js("results/fleet/sectioned_summary.json")["SecFleetMaxDiff"]["r1833"]
# blocks per vehicle-day at Okpo with 30 vehicles, on the map with floating docks (per_vehicle_rates.csv, column r1833):
# 16-h days in the text, 24-h days ("Day") in the anchor figure
SEC72 = {r["macro"]: r for r in rows("results/anchors/per_vehicle_rates.csv")}
assert set(SEC72) == {f"YuPerTruck{m}{d}" for m in ("Free", "Res", "Ref", "Safe") for d in ("", "Day")}
for m_ in ("Free", "Res", "Ref", "Safe"):
    num("YuPerTruck" + m_, SEC72["YuPerTruck" + m_]["r1833"], 1)
num("HHIRate", 500 / 24, 1)        # Park and Seo (2006): about 500 blocks/day with 24 transporters, 24-h days
num("ShenRate", 600 / 30)          # Shen et al. (2018): about 600 blocks/day with about 30 transporters

# ---------------------------------------------------------------- bounds (results/bounds)
SYS = {y: {r["scen"]: r for r in rows(BOUNDS_DIR[y] + "/system.csv")} for y in YARDS}
RF = {y: {r["scen"]: r for r in rows(BOUNDS_DIR[y] + "/route_free.csv")} for y in YARDS}
CONV = {y: rows(BOUNDS_DIR[y] + "/convergence.csv") for y in YARDS}
RO = {y: {r["scen"]: r for r in rows(BOUNDS_DIR[y] + "/readout.csv")} for y in YARDS}
PR = {y: rows(BOUNDS_DIR[y] + "/prices.csv") for y in YARDS}
SC = {"main": "Main", "erection": "Er", "crane": "Crane"}
num("YuSysMain", SYS["yupu"]["main"]["T1_system"])
num("YuSysCrane", SYS["yupu"]["crane"]["T1_system"])
# Table 6 note: sampled bounds are ten task samples of MixN tasks each (the ten seeds of the bound-sample batches that
# each bound run reads, listed under "ref" in its config.json); the route margin runs from a feasible value of the
# route-free programme (column generation, checked
# by back-substitution) to the certificate, the minimum over pricing rounds that the table reports as route-free bound
for y in YARDS:
    c23 = js(BOUNDS_DIR[y] + "/config.json")
    assert len(c23["seeds"]) == 10 and fmt(c23["n_mix"]) == MAC["MixN"]
    for d in c23["ref"].values():
        cr = js(f"{d}/config.json")
        assert len(cr["seeds"]) == 10 and cr["n_mix"] == c23["n_mix"]
    for sc in SC:
        f = RF[y][sc]
        assert abs(float(f["margin_lo_rel"]) - (float(f["feasible"]) / float(f["T1_system"]) - 1)) < 1e-9
        assert abs(float(f["margin_hi_rel"]) - (float(f["T1pp"]) / float(f["T1_system"]) - 1)) < 1e-9
num("YuCorridorExtra", RO["yupu"]["crane"]["alt_extra_m"])
c2k = [abs(float(r["mean_abs_rel"])) for y in YARDS for r in CONV[y] if r["N"] == "2000"]
c200k = [abs(float(r["mean_abs_rel"])) for y in YARDS for r in CONV[y] if r["N"] == "200000"]
pct("ConvTwoKLo", min(c2k), 1)
pct("ConvTwoKHi", max(c2k), 1)
pct("ConvBigLo", min(c200k), 1)
pct("ConvBigHi", max(c200k), 1)
jen = [r for y in YARDS for r in CONV[y] if r["scen"] != "crane"]
num("JensenGroups", len(jen))
num("JensenBelow", sum(float(r["hmean_minus_system"]) < 0 for r in jen))
num("JensenMaxZ", max(float(r["hmean_z"]) for r in jen), 1)
big = RF["yupu"]["crane"]
num("LPRows", big["rows"])
num("LPCols", big["lp_cols"])
num("LPIters", big["iters"])
num("LPMin", (float(big["secs"]) + float(big["init_secs"])) / 60)
num("YuSysSecs", max(float(SYS["yupu"][sc]["secs"]) for sc in SC))
num("LPOtherSecs", max(float(RF[y][sc]["secs"]) + float(RF[y][sc]["init_secs"])
                       for y in YARDS for sc in SC if (y, sc) != ("yupu", "crane")))


def price_row(y, sc, key):
    return next(r for r in PR[y] if r["scen"] == sc and key in r["name"])


for y in YARDS:     # fleet size at which free flow first exceeds the system bound
    c = rule(f"{y}|main", "whole|free")["curve"]
    num(Y[y] + "FreeCross", float(SEC61["YuFreeCross"]["r1833"]) if y == "yupu"
        else min(int(k) for k, v in c.items() if v > float(SYS[y]["main"]["T1_system"])))
num("YuCorrUtil", price_row("yupu", "main", "road010-6")["T1′_util"], 2)
doors = [float(r["T1′_util"]) for r in PR["yantai"] if r["scen"] == "main" and "entrance001 node" in r["name"]
         and "building014" in r["facility"]]
num("YaDoorUtilB", sorted(doors)[-2], 3)
sysm = {y: float(SYS[y]["main"]["T1_system"]) for y in YARDS}
pct("YuDensOwn", 339 / sysm["yupu"])
pct("YaDensOwn", round(68512 / 100 * 10 / 300) / sysm["yantai"], 1)
num("YuDockPrice", price_row("yupu", "main", "road161")["T1′_price"], 1)

# ---------------------------------------------------------------- rules at saturation, capacity interval
# Okpo map with the floating docks and spurs (the earlier saturation and bound runs used rev 1825, without them): same
# stops and node resources, same system bounds, four-rule plateaus within MapChangeMax
md = js("results/network/okpo_revisions.json")["None"]
assert md["revision"] == [1825, 1833] and md["stops_same"] and md["ents_same"]
assert md["node_res"][0] == md["node_res"][1] and md["access"][0] == md["access"][1]
SYS27 = {(r["yard"], r["scen"], r["section"]): float(r["T1_sys"]) for r in rows("results/bounds/system_by_granularity.csv")}
for y in YARDS:
    for sc in SC:
        assert abs(SYS27[(y, sc, "whole")] - float(SYS[y][sc]["T1_system"])) < 0.05, (y, sc)
I27 = interval()                                                                                  # system bounds by kind
R27 = {r["key"]: r for r in rows("results/saturation/rules_main_mix.csv")}                 # main mix, by rule and granularity
R27E = {r["case"] + "|" + r["combo"]: r for r in rows("results/saturation/rules.csv")}       # all rules, both mixes
# sectioning the dock roads, read on one map: system bound and full-path reservation plateau
plat27 = lambda sc, g: float(R27E[f"yupu|{sc}|{g}|reserve"]["plateau"])
for sc, tag in (("main", "Main"), ("erection", "Er")):
    pct("Yu" + tag + "BoundGain", SYS27[("yupu", sc, "stops")] / SYS27[("yupu", sc, "whole")] - 1, 1)
    pct("Yu" + tag + "ResGain", plat27(sc, "stops") / plat27(sc, "whole") - 1, 1)
assert abs(plat27("main", "stops") / plat27("main", "whole") - 1 - float(SEC61["YuMainResGain"]["r1833"])) < 1e-4
for (y, kind, gran), r in I27.items():         # interval rows agree with the system bounds on the same map
    assert abs(float(r["T1_sys"]) - SYS27[(y, kind, gran)]) <= 0.05 + 1e-9      # csv rounds to 0.1
# decline of safe release beyond its peak, K = 150 (both yards, both granularities)
pd_ = [f"{y}|{g}" for y in YARDS for g in ("whole", "stops")]
drops = [-float(R27[f"{k}|segment_safe|inf"]["drop_150"]) / 100 for k in pd_]
pct("SafeDropLo", min(drops))
pct("SafeDropHi", max(drops))
# required fleets with integer-K-filled searches (crane-pacing fleet runs): crane-paced erection in the required-fleet table
E_DEM = {y: {(r["scen"], int(float(r["demand"])), r["model"]): r for r in rows(CRANE_FLEET[y] + "/demand.csv")}
         for y in YARDS}

# ---------------------------------------------------------------- public-source inputs (not results)
# Conversion inputs as in occupancy/params.py, from the published sources that the paper cites with the parameters
# (Appendix B).
num("OkSteelT", 1018067)       # Hanwha Ocean sustainability report 2025, p.110, mean 2022-2025, t/yr
num("BlockT", 100)             # mean block weight, t (Yantai conversion; Zhang 2025 thesis 50-150 t)
num("MovesPerBlock", 10)       # moves per block (Meng et al. 2018; Tao et al. 2019)
num("WorkDays", 300)           # working days per year (Yantai conversion)
num("OkMovesYr", 105000)       # Hanwha Systems customer case 2026: 105,000 block moves per year at Okpo
num("YaSteelT", 68512)         # Yantai steel, t/yr (source in Appendix B)
num("OkLNG", 20)               # LNG carriers per year in the two dry docks (base scenario)
# derived: 2025 deliveries = backlog on a delivery basis at end-2024 + 2025 orders - backlog at end-2025, LNG carriers
# (Hanwha Ocean earnings presentations 4Q24 and 4Q25, p.17; read 2026-10-13)
num("OkLNGBacklogStart", 72)
num("OkLNGOrders", 13)
num("OkLNGBacklogEnd", 65)
assert 72 + 13 - 65 == 20
num("ErectUnits", 100)         # erection units per ship, main value (patent: 90-110)
num("ErectUnitsLo", 89)        # Roh and Lee (2006)
num("ErectUnitsHi", 110)
num("HHIMoves", 500)           # Park and Seo (2006)
num("HHIFleet", 24)
num("ShenMoves", 600)          # Shen et al. (2018)
num("ShenFleet", 30)
sens = {r["item"]: r for r in rows("results/saturation/sensitivity.csv")}
w = sens["plateau with widths +1.5 / −1.5 m"]
put("WidthSensYu", w["Okpo"].replace(" / ", "/").replace("%", r"\%").replace("-", "$-$"))
put("WidthSensYa", w["Yantai (by entity)"].replace(" / ", "/").replace("%", r"\%").replace("-", "$-$"))
own = round(1018067 / 100 * 10 / 300)
direct = round(105000 / 300)
levels = sorted({int(float(r["demand"])) for r in DEM})
assert own in levels and direct in levels and 2 * own in levels and 2 * direct in levels
num("OkOwn", own)
num("OkDirect", direct)
num("OkPeakLo", 2 * own)
num("OkPeakHi", 2 * direct)
num("YaOwn", round(68512 / 100 * 10 / 300))
reg = [n for n in levels if n in (300, 450, 600)]
put("RegLevels", ", ".join(fmt(n) for n in reg[:-1]) + " and " + fmt(reg[-1]))
put("PeakLevels", f"{fmt(2 * reg[-2])} and {fmt(2 * reg[-1])}")
assert 2 * reg[-2] in levels and 2 * reg[-1] in levels
er_cfg = js("results/runs/okpo_rev1825_erection_safe/config.json")
pct("ErShare", er_cfg["scenarios"]["erection"]["flow_share"]["P6"])
DOCK_KS = sorted({int(k.split("|")[-1]) for k in RK})        # fleets of the thresholds the paper reports (thresholds.csv)
put("DockKs", ", ".join(str(k) for k in DOCK_KS[:-1]) + f" and {DOCK_KS[-1]}")
num("YuRhoLo", min(float(r["rho"]) for r in rho_rows if r["yard"] == "yupu"), 2)
num("YaRhoLo", min(float(r["rho"]) for r in rho_rows if r["yard"] == "yantai"), 2)

# ---------------------------------------------------------------- block length, dock-mouth network
BLOCK = rows("results/bounds/block_length.csv")
num("BootN", js("results/fleet/bootstrap.json")["bootstrap_resamples"])     # size of the paired seed bootstrap


def klist(ks):
    ks = [str(k) for k in ks]
    return ", ".join(ks[:-1]) + " and " + ks[-1]


def rng(name, vals, nd=0):
    lo, hi = fmt(min(vals), nd), fmt(max(vals), nd)
    put(name, lo if lo == hi else f"{lo}--{hi}")


# block length (granularity) at Okpo
BL = {(r["scen"], r["key"], r["gran"]): r for r in BLOCK}


def blk_row(scen, key):
    return next(r for (s, k, g), r in BL.items() if s == scen and k == key)


e0, e1 = blk_row("erection-heavy mix", "erection"), blk_row("erection-heavy mix", "erection_sect")
m0, m1 = blk_row("main", "main"), blk_row("main", "main_s")
# the bound gains of sectioning are those of the text (same map, same system bounds)
for (a_, b_), sc in (((e0, e1), "erection"), ((m0, m1), "main")):    # (the block table rounds bounds to 0.1)
    assert abs(float(b_["T1"]) / float(a_["T1"]) - SYS27[("yupu", sc, "stops")] / SYS27[("yupu", sc, "whole")]) < 5e-4
for scen, base in (("main", m1), ("erection-heavy mix", e1)):
    for r in BLOCK:
        if r["scen"] == scen and r["block_m"] and "010-6" not in r["gran"]:
            assert r["T1"] == base["T1"]                 # block length leaves the bound unchanged
gap = [r for r in BLOCK if r["scen"] == "erection-heavy mix · dock-front section length"]
pct("GapBound", max(abs(float(r["T1"]) / float(e1["T1"]) - 1) for r in gap), 1)
blen = sorted({int(r["block_m"]) for r in BLOCK if r["block_m"] and "010-6" not in r["gran"]})
num("BlockLenMin", blen[0])
num("BlockLenMax", blen[-1])
glen = sorted(2 * int(r["gap_m"]) for r in gap)
num("GapLenLo", glen[0])
num("GapLenHi", glen[-1])

# dock-mouth network: threshold rho_10 by fleet and passable margin
DOCK_PY = show("experiments/dock_mouth.py")
DCFG = js(SYN_DOCK + "/config.json")
dp = DCFG["params"]
L1_, DOCK_S_, YARD_S_ = map(float, re.search(r"^L1, DOCK_S, YARD_S = ([\d.]+), ([\d.]+), ([\d.]+)", DOCK_PY, re.M).groups())
nsrc = len(re.search(r"^SRC = \(([^)]*)\)", DOCK_PY, re.M).group(1).split(","))
B_, LV_ = map(float, re.search(r'flow="P6", .*B=([\d.]+), Lv=([\d.]+)', DOCK_PY).groups())
assert f'B={B_}, Lv={LV_}' in DOCK_PY.split("class Background")[1]                 # background blocks the same
d1, d0 = B_ + dp["lat_clear"], dp["plat_w"] + dp["lat_clear"]
num("NetLen", L1_)
num("NetAt", DOCK_S_)
num("YardAt", YARD_S_)
num("NetSrcBays", nsrc)
num("NetDone", d1)
num("NetDzero", d0, 1)
num("NetDzeroTwo", 2 * d0)
num("NetMarginLo", min(DCFG["widths"]) + dp["lat_clear"] - d1)
num("NetMarginHi", max(DCFG["widths"]) + dp["lat_clear"] - d1)
num("NetMarginN", len(DCFG["widths"]))
num("NetRhoLo", min(DCFG["rhos"]), 1)
num("NetRhoHi", max(DCFG["rhos"]), 2)
num("NetRhoN", len(DCFG["rhos"]))
put("NetKs", klist(DCFG["ks"]))
num("NetKHi", max(DCFG["ks"]))
num("NetDays", DCFG["days"])
num("NetFail", DCFG["fail"], 2)
# Proposition: share of the dock road's day that erections take without waiting on it
tau, v1_, tU = dp["crane_cycle_s"], dp["v_loaded"], dp["unload_s"]
pct("PropShareRoadside", (DOCK_S_ / v1_ + tU) / tau, 1)
pct("PropShareBay", (DOCK_S_ / v1_) / tau, 1)
RS25 = rows(SYN_DOCK + "/rho_star.csv")
assert all(not r["rho_star_P6"] for r in RS25)                  # erections completed / released >= fail everywhere
assert all(not r["rho_star_thr10"] for r in RS25 if r["rule"] in ("seg_bay", "res"))    # waiting off the road
STAR = {(r["rule"], float(r["margin"]), int(r["K"])): (float(r["rho_star_thr10"]) if r["rho_star_thr10"] else None)
        for r in RS25}
for r in RS25:
    assert abs(float(r["W"]) + dp["lat_clear"] - d1 - float(r["margin"])) < 1e-9
MARG = sorted({m for _, m, _ in STAR})
KS_D = sorted({k for _, _, k in STAR})
cls = lambda m: int(m // d0)                           # empty vehicles that fit beside a standing loaded one
CLS = {}
for K in KS_D:                                         # whole road: identical within each class at every fleet size
    for c in sorted({cls(m) for m in MARG}):
        vals = {STAR[("seg", m, K)] for m in MARG if cls(m) == c}
        assert len(vals) == 1, (K, c, vals)
        CLS[(K, c)] = vals.pop()
assert sorted({cls(m) for m in MARG}) == [0, 1, 2]
nodrop = [K for K in KS_D if all(CLS[(K, c)] is None for c in (0, 1, 2))]
num("NetNoDropK", max(nodrop))
assert nodrop == KS_D[:len(nodrop)]
k0 = [K for K in KS_D if CLS[(K, 0)] is not None]
num("NetZeroFirstK", k0[0])
num("NetZeroFirst", CLS[(k0[0], 0)], 2)
num("NetZeroLast", CLS[(KS_D[-1], 0)], 2)
assert all(CLS[(a, 0)] > CLS[(b, 0)] for a, b in zip(k0, k0[1:])) and all(CLS[(a, 1)] > CLS[(b, 1)] for a, b in zip(k0, k0[1:]))
assert CLS[(KS_D[-1], 2)] > CLS[(KS_D[-2], 2)]
kk = [K for K in KS_D if K >= 16]                        # the K = 12 cell of the middle class sits at the grid top
step = [1 - (1 - CLS[(K, 1)]) / (1 - CLS[(K, 0)]) for K in kk]
assert all(a > b for a, b in zip(step, step[1:]))
pct("StepGainHi", step[0])
pct("StepGainLo", step[-1])
num("StepKLo", kk[0])
num("StepKHi", kk[-1])
assert 24 in KS_D and CLS[(24, 0)] is not None
num("NetMechK", 24)        # fleet whose throughput curves the dock-mouth figure draws (make_figures.py reads this macro)
# sectioned dock road, waiting on the road: earlier or later than the whole road, by margin, at K >= 16
side = {}
for m in MARG:
    s = {("E" if STAR[("seg_s", m, K)] < (CLS[(K, cls(m))] or 1.0) else "L") if STAR[("seg_s", m, K)] is not None
         else ("L" if CLS[(K, cls(m))] is not None else "=") for K in kk}
    side[m] = s.pop() if len(s) == 1 else "mixed"


def mlist(ms):
    ms = [fmt(m) for m in ms]
    return ms[0] if len(ms) == 1 else ", ".join(ms[:-1]) + " and " + ms[-1]


put("SecEarlier", mlist([m for m in MARG if side[m] == "E"]))
put("SecLater", mlist([m for m in MARG if side[m] == "L"]))
put("SecMixed", mlist([m for m in MARG if side[m] == "mixed"]))
assert all(STAR[("seg_s", m, 8)] is not None for m in MARG if cls(m) == 0) and CLS[(8, 0)] is None
# absolute throughput (ten-seed means of the per-point results, synthetic_dock/points.csv): sectioning raises the level at low
# utilisation, against which each road's threshold is read, and where its threshold comes later it can collapse deeper
PTS = rows(SYN_DOCK + "/points.csv")
TPT = {}
for r in PTS:
    TPT.setdefault((r["rule"], float(r["margin"]), int(r["K"]), float(r["rho"])), []).append(float(r["thr"]))
assert all(len(v) == len(DCFG["seeds"]) for v in TPT.values())
mthr = lambda rule, m, K, rho: sum(TPT[(rule, m, K, rho)]) / len(TPT[(rule, m, K, rho)])
rlo, rhi = min(DCFG["rhos"]), max(DCFG["rhos"])
rng("SecBaseGain", [100 * (mthr("seg_s", m, K, rlo) / mthr("seg", m, K, rlo) - 1) for m in MARG for K in kk])
MAC["SecBaseGain"] += r"\%"
mL, KL = [m for m in MARG if side[m] == "L"][0], KS_D[-1]
assert STAR[("seg_s", mL, KL)] > CLS[(KL, cls(mL))] and mthr("seg_s", mL, KL, rhi) < mthr("seg", mL, KL, rhi)
num("SecDeepM", mL)
num("SecDeepK", KL)
num("SecDeepSec", mthr("seg_s", mL, KL, rhi), 1)
num("SecDeepWhole", mthr("seg", mL, KL, rhi), 1)
# cells (K >= 16) in which the sectioned road reaches its threshold later: at the top utilisation it then carries
# more than the whole road with few vehicles and collapses deeper with many (both kinds occur at K = 24)
later = [(m, K, mthr("seg_s", m, K, rhi) < mthr("seg", m, K, rhi)) for m in MARG for K in kk
         if CLS[(K, cls(m))] is not None and (STAR[("seg_s", m, K)] is None or STAR[("seg_s", m, K)] > CLS[(K, cls(m))])]
assert (len(later), sum(d for _, _, d in later)) == (10, 5)                       # 5 cells deeper, 5 higher
assert not any(d for _, K, d in later if K == kk[0]) and all(d for _, K, d in later if K == kk[-1])
assert (mL, kk[0], False) in later and (mL, KL, True) in later
num("SecShallowK", kk[0])
num("SecShallowSec", mthr("seg_s", mL, kk[0], rhi), 1)
num("SecShallowWhole", mthr("seg", mL, kk[0], rhi), 1)
# waiting term of Proposition 5(ii): ten-seed means of the dock-mouth runs that log occupancy (safe release, whole
# dock road, margins below d1 where at most one loaded vehicle is on the road, so N-bar is the union of the waits)
NBR = [r for r in rows("results/synthetic_dock/waiting_term.csv") if r["rule"] == "seg"]
NPT = rows(SYN_WAIT + "/points.csv")
NB = {(float(r["margin"]), int(r["K"]), float(r["rho"])): r for r in NBR}
same = lambda p, m, K, rho: (p["rule"], float(p["margin"]), int(p["K"]), float(p["rho"])) == ("seg", m, K, rho)
for (m, K, rho), r in NB.items():
    assert m < d1 and int(r["n"]) == len(DCFG["seeds"])
    nv = [float(p["nbar"]) for p in NPT if same(p, m, K, rho)]           # the mean is over the released per-seed values
    assert len(nv) == int(r["n"]) and abs(sum(nv) / len(nv) - float(r["nbar_mean"])) < 1e-12
    for col, f in (("done_BG_mean", "done_BG"), ("done_P6_mean", "done_P6")):   # same runs as the released dock-mouth
        v = [float(p[f]) for p in PTS if same(p, m, K, rho)]                      # points (logging changes nothing)
        assert len(v) == int(r["n"]) and abs(sum(v) / len(v) - float(r[col])) < 1e-9
at = lambda m, K, rho=rhi: float(NB[(m, K, rho)]["nbar_mean"])
KR, RLOW = sorted({K for _, K, _ in NB}), min(rho for _, _, rho in NB)
M0 = sorted({m for m, _, rho in NB if rho == rhi and cls(m) == 0})
M1 = sorted({m for m, _, rho in NB if rho == rhi and cls(m) == 1})
assert KR == [16, 32] and RLOW == 0.6 and M0 and M1
for K in KR:
    assert all(at(m, K) == at(M0[0], K) for m in M0) and all(at(m, K) == at(M1[0], K) for m in M1)   # same step, same runs
    assert at(M0[0], K) > at(M1[0], K)                                  # falls with the step of margin ...
for m in M0 + M1:
    assert at(m, KR[1]) > at(m, KR[0])                                  # ... and rises with the fleet
rng("NbarZero", [at(M0[0], K) for K in KR], 2)
rng("NbarOne", [at(M1[0], K) for K in KR], 2)
num("NbarCi", max(float(r["nbar_ci95"]) for (m, K, rho), r in NB.items() if rho == rhi), 2)
put("NbarKs", klist(KR))
put("NbarMOne", mlist(M1))
num("NbarSeeds", len(DCFG["seeds"]))
# bound of Proposition 5(ii) on the adjacent traffic, from the ten-seed means (linear in N-bar and lambda_E, so the bound at the means
# is the mean of the per-seed bounds), over the measured adjacent traffic
assert YARD_S_ > DOCK_S_
ratio14 = [(dp["day_s"] * (1 - at(m, K)) - float(NB[(m, K, rhi)]["done_P6_mean"]) * (DOCK_S_ / v1_ + tU)) * v1_ / YARD_S_
           / float(NB[(m, K, rhi)]["done_BG_mean"]) for m, K, rho in NB if rho == rhi]
assert min(ratio14) > 1

# one crane at Dock 1: the current erection volume does not fit 190 min per erection; at times that fit, the bound and its
# binding element stay
CR1 = {r["name"]: r for r in rows("results/dock/one_crane.csv")}
one = [CR1[f"crane_{p}_dock1_1"] for p in ("p6lo", "peak", "p6hi")]
assert all(r["infeasible"] == "True" and float(r["tau_min"]) == cycle_min and float(r["rho"]) > 1 for r in one)
pk = CR1["crane_peak_dock1_1"]
assert abs(float(pk["n6"]) - float(MAC["YuErectPerDay"])) < 0.05
assert all(abs(float(CR1[f"crane_{p}_dock1_1"]["n6"]) / float(pk["n6"]) - u / int(MAC["ErectUnits"])) < 0.005
           for p, u in (("p6lo", int(MAC["ErectUnitsLo"])), ("p6hi", int(MAC["ErectUnitsHi"]))))
rng("CraneOneRho", [float(r["rho"]) for r in one], 2)
num("CraneOneTauMax", pk["tau_max"])
fit = [CR1["crane_peak_dock1_1_tau100"], CR1["crane_peak_dock1_1_tau50"]]
mp = CR1["crane_peak_map"]
assert all(r["infeasible"] == "False" and float(r["tau_min"]) <= float(r["tau_max"]) and "road010-6" in r["top1"]
           for r in fit + [mp])
rng("CraneOneBound", [float(r["T1_sys"]) for r in fit])
assert fmt(mp["T1_sys"]) == MAC["YuSysCrane"]          # the map's cranes: the crane-paced system bound of the text
put("CraneOneTaus", klist(sorted({int(float(r["tau_min"])) for r in fit}, reverse=True)))

# ---------------------------------------------------------------- crane-paced erection, bound sensitivity
# erection load of the main mix against the cranes of the map (cranes are resources only under crane pacing)
cap_day = int(MAC["YuCranes"]) * P["day_s"] / C["crane_cycle_s"]      # erections a day that Okpo's cranes can lift
EL = {(r["yard"], r["level"], r["day_h"]): r for r in rows("results/interval/erection_load.csv")}
el = [EL[("yupu", lv, "16")] for lv in ("own", "600")]
assert abs(float(el[0]["crane_cap"]) - cap_day) < 1e-9 and abs(float(el[0]["P6_share"]) - 0.2 / 3.45) < 1e-12
pct("MainErShare", el[0]["P6_share"], 1)
rng("MainCraneLoad", [float(r["ratio_main"]) for r in el], 1)
# capacity interval under crane pacing (whole roads; sectioned roads give the same bound)
CI = {(r["yard"], r["gran"], r["mix"], r["kind"]): r for r in rows("results/interval/crane_interval.csv")}
# Okpo's crane-paced whole-road plateau of full-path reservation is read over 16 days (crane_16day_plateau.csv): the
# four-day runs are not
# steady there (one of eleven fleets), all eleven 16-day runs are; the 16-day plateau (days 9-16) lies below the
# four-day interval. Yantai keeps its four-day plateau (all eleven fleets steady; eight-day check below).
CP16 = rows("results/interval/crane_16day_plateau.csv")
assert [(r["yard"], r["gran"]) for r in CP16] == [("yupu", "whole")]
CP16 = CP16[0]
assert CP16["n_K"] == CP16["steady_K"] == "11" and CP16["in_4d_interval"] == "False"
assert abs(float(CP16["plat_4d"]) - float(CI[("yupu", "whole", "crane", "reserve")]["plat"])) < 1e-9
assert abs(float(CP16["T1"]) - float(CI[("yupu", "whole", "crane", "reserve")]["T1"])) < 1e-9
assert CI[("yupu", "whole", "crane", "reserve")]["steady_K"] == "1"
PLAT_CRANE = {"yupu": float(CP16["plat_16d"]), "yantai": float(CI[("yantai", "whole", "crane", "reserve")]["plat"])}
num("YuCraneSixteenPlat", CP16["plat_16d"])
num("YuCraneSixteenCI", CP16["plat_16d_ci"])
pct("YuCraneSixteenDrop", -float(CP16["rel"]), 1)
num("YuCraneFourPlat", CP16["plat_4d"])
for y in YARDS:                                     # the crane-paced bounds are the system bounds of the bound table
    best = CI[(y, "whole", "crane", "best")]
    assert fmt(best["T1"]) == fmt(SYS[y]["crane"]["T1_system"]) and float(best["T1"]) == float(CI[(y, "stops", "crane", "best")]["T1"])
pct("YuCraneBoundGain", float(CI[("yupu", "whole", "crane", "best")]["T1"]) / float(CI[("yupu", "whole", "main", "best")]["T1"]) - 1, 1)
assert "010-6" in CI[("yupu", "whole", "crane", "best")]["top1"]                # through corridor binds under crane pacing
pct("YaCraneBoundDrop", 1 - float(CI[("yantai", "whole", "crane", "best")]["T1"]) / float(CI[("yantai", "whole", "main", "best")]["T1"]), 1)
# Yantai's crane-paced plateaus over eight days (crane_8day.csv): at K = 100 and 150 on both kinds of road the runs are steady
# and days 5-8 stay within CraneLongYaMax of the four-day readings (days 2-4 of the same runs)
C8 = rows("results/interval/crane_8day.csv")
assert len(C8) == 8 and all(r["same_as_4d"] == "10" for r in C8)
C8YA = [r for r in C8 if r["yard"] == "yantai"]
assert len(C8YA) == 4 and all(r["steady_8d"] == "True" for r in C8YA)
pct("CraneLongYaMax", max(abs(float(r["rel_d5_8"])) for r in C8YA), 1)
# required fleets: crane-paced against the main mix, same map, same seeds, paired bootstrap of the difference
FL = rows("results/fleet/crane_fleet.csv")
civ = lambda s: [None if x.strip() == "None" else int(x) for x in s.strip("[]").split(",")]
FLK = {(r["yard"], r["day_h"], r["n"], r["cell"]): r for r in FL}
# bound sensitivity: road widths +-1.5 m, 30-min handling, without the width-packing rows (system bounds, both yards)
BS = {(r["yard"], r["scen"], r["section"], r["variant"]): r for r in rows("results/bounds/sensitivity.csv")}
base = {k[:3]: r for k, r in BS.items() if k[3] == "base"}
for k, r in base.items():                                          # base bounds are the system bounds of the text
    assert abs(float(r["T1_sys"]) - SYS27[k]) < 0.05
moved = [k for k in base if float(BS[k + ("w+1.5",)]["rel_base"]) != 0]
assert all(("010-6" in base[k]["top1"]) or ("road007" in base[k]["top1"]) for k in moved)   # 12-m roads bind there
assert all(("road161" in base[k]["top1"]) or ("entrance" in base[k]["top1"]) for k in base if k not in moved)
rng("WidthBoundUp", [100 * float(BS[k + ("w+1.5",)]["rel_base"]) for k in moved], 1)
rng("WidthBoundDown", [-100 * float(BS[k + ("w-1.5",)]["rel_base"]) for k in moved], 1)
MAC["WidthBoundUp"] += r"\%"
MAC["WidthBoundDown"] += r"\%"
assert all(float(BS[("yupu", "main", "whole", v)]["rel_base"]) == 0 for v in ("w+1.5", "w-1.5"))
assert all(float(base[k]["T1_nopack"]) == float(base[k]["T1_sys"]) for k in base if "entrance" in base[k]["top1"])  # doors: none
rng("LoadBoundDrop", [-100 * float(r["rel_base"]) for k, r in BS.items() if k[3] == "l30"])
MAC["LoadBoundDrop"] += r"\%"
L30 = {(r["yard"], r["gran"], r["kind"]): r for r in rows("results/interval/handling_30min.csv")}

# ---------------------------------------------------------------- this paper: lower end at full-path reservation
# width-packing rows: by how much they tighten the bound, 1 - bound / bound without the rows (the column pack_tighten)
cut_ = lambda k: 100 * (1 - float(base[k]["T1_sys"]) / float(base[k]["T1_nopack"]))
assert all(abs(cut_(k) / 100 - float(base[k]["pack_tighten"])) < 1e-9 for k in base)
rng("PackTightCut", [cut_(k) for k in base if k[0] == "yupu" and k[1] in ("main", "crane")])
rng("PackTightCutEr", [cut_(k) for k in base if k[0] == "yupu" and k[1] == "erection"])
MAC["PackTightCut"] += r"\%"
MAC["PackTightCutEr"] += r"\%"
# capacity interval with full-path reservation as the lower end (plateau of saturation/rules.csv, map with floating docks)
TK, TG = {"main": "", "erection": "Er"}, {"whole": "Whole", "stops": "Sec"}
RESLOW = {}
for (y, kind, gran), r in I27.items():
    if kind not in TK:
        continue
    res = float(R27E[f"{y}|{kind}|{gran}|reserve"]["plateau"])
    RESLOW[(y, kind, gran)] = (res, float(r["T1_sys"]))
    num(Y[y] + TK[kind] + "ResLow" + TG[gran], res)
    pct(Y[y] + TK[kind] + "ResLow" + TG[gran] + "Ratio", res / float(r["T1_sys"]))
assert abs(RESLOW[("yupu", "main", "whole")][0] - float(SEC61["YuResPlateau"]["r1833"])) < 0.5     # plateau of Section 4.2
assert abs(RESLOW[("yantai", "main", "whole")][0] - float(rule("yantai|main", "whole|reserve")["plateau"])) < 0.5
for y in YARDS:
    r_ = CI[(y, "whole", "crane", "reserve")]
    pct(Y[y] + "CraneResRatio", PLAT_CRANE[y] / float(r_["T1"]))      # Okpo: 16-day plateau
assert abs(PLAT_CRANE["yupu"] / float(CP16["T1"]) - float(CP16["ratio_16d"])) < 1e-12
# both ends with 30-min loading and unloading (handling_30min.csv, full-path reservation)
for (y, g, kind), r in L30.items():
    if kind != "reserve":
        continue
    assert abs(float(r["T1_10"]) - SYS27[(y, "main", g)]) < 0.05 and abs(float(r["plat_10"]) - RESLOW[(y, "main", g)][0]) < 0.5
    pct(Y[y] + "LoadThirty" + TG[g] + "Ratio", r["ratio_30"])
num("YaLoadThirtyWholeBound", L30[("yantai", "whole", "reserve")]["T1_30"])
res30 = [r for (y, g, kind), r in L30.items() if kind == "reserve"]
rng("LoadResDrop", [-100 * float(r["dplat_rel"]) for r in res30])
rng("YaLoadThirtyResRatios", [100 * float(L30[("yantai", g, "reserve")]["ratio_30"]) for g in TG])   # by yard (Okpo:
MAC["LoadResDrop"] += r"\%"                                                         # YuLoadThirtyWholeRatio, ...Sec...)
MAC["YaLoadThirtyResRatios"] += r"\%"
# crane pacing against the main mix: free flow and full-path reservation only
yu_res = [r for r in FL if r["yard"] == "yupu" and r["cell"] in ("free", "reserve")]
assert all(civ(r["dK_ci"])[0] <= 0 <= civ(r["dK_ci"])[1] for r in yu_res)
num("YuCraneFleetDiffRes", max(abs(int(r["dK"])) for r in yu_res))
num("YuCraneCellsRes", len(yu_res))
more_res = [r for r in FL if r["yard"] == "yantai" and (r["day_h"], r["n"]) in (("16", "450"), ("24", "600"))
            and r["cell"] == "reserve"]
assert more_res and all(civ(r["dK_ci"])[1] < 0 for r in more_res)
rng("YaCraneMoreRes", [-int(r["dK"]) for r in more_res])
rest_res = [r for r in FL if r["yard"] == "yantai" and r["cell"] in ("free", "reserve") and r not in more_res
            and not (r["n"] == "600" and r["day_h"] == "16" and r["cell"] == "reserve")]   # >150 in both scenarios
num("YaCraneRestMaxRes", max(abs(int(r["dK"])) for r in rest_res))
# per-run checks of this paper: besides the batches checked above, the saturated full-path reservation points of the
# crane-paced and 30-min batches (CRANE_RUNS)
sens = {}
for d in CRANE_RUNS:
    for r in rows(f"results/runs/{d}/points.csv"):
        if r["model"] == "reserve" and r["T1_tight_mix"]:
            sens[(d, r["scen"], r["K"], r["seed"])] = float(r["throughput"]) / float(r["T1_tight_mix"])
assert {k[1] for k in sens} == {"crane", "crane_s", "main_l30", "main_s_l30"} and max(sens.values()) <= 1
num("CheckSensPoints", len(sens))
L30_SUM = js("results/fleet/fleet_30min_summary.json")
assert L30_SUM["t1_check"]["over"] == 0                       # required-fleet runs with 30-min handling, anchor runs
num("CheckLThirtyPoints", L30_SUM["t1_check"]["points"])
SCREEN_SUM = js("results/fleet/screen_summary.json")              # the 30-min points above plus the low-demand fleet runs
assert SCREEN_SUM["t1_check"]["over"] == 0 and SCREEN_SUM["t1_check"]["points"] > L30_SUM["t1_check"]["points"]
num("CheckScreenPoints", SCREEN_SUM["t1_check"]["points"] - L30_SUM["t1_check"]["points"])
# per-point checks of the main experiments, from the batches whose results the paper reads, rules of the paper only
# (full-path reservation, safe release, the reference rule): Yantai from the batches of rules_summary.json (dock-mouth
# runs and the K = 150 runs of the other two rules, safe release curves, required fleets) and its saturation curves
# (FLAG_RUNS, main scenario); Okpo on the map with floating docks: saturation curves of the main and erection-heavy mixes
# (okpo_rules, okpo_rules_erection), required fleets (okpo_curves_check.json, SEC_FLEET), and from
# refusal_summary.json the eight-day fleets and 16-day dock-mouth runs, the dock-mouth runs, the refusal and teleport
# runs. The Okpo batches on the earlier map and the earlier anchor runs are not counted here.
P1_RULES = {"reserve", "segment_safe", "segment"}


def point_check(d, keep=lambda r: True):
    pts = [r for r in rows(d + "/points.csv") if r["model"] in P1_RULES and r["T1_tight_mix"] and keep(r)]
    ratio = [float(r["throughput"]) / float(r["T1_tight_mix"]) for r in pts]
    return {"points": len(pts), "max_ratio": max(ratio), "over": sum(x > 1 for x in ratio)}


D22 = S["dirs"]
dem22 = {d: point_check("results/" + d) for d in (D22["demand"]["yupu"], D22["demand"]["yantai"],
                                                       D22["demand_anchor"], D22["demand_peak"])}
assert sum(c["points"] for c in dem22.values()) == S["t1_check_demand"]["points"]     # the check of rules_summary.json, split by batch
assert [js(f"results/{d}/config.json")["yard"] for d in dem22] == ["yupu", "yantai", "yupu", "yupu"]
assert js(FLAG_RUNS["yantai"] + "/config.json")["yard"] == "yantai"
RULES_P1 = {"okpo_rules": {"main", "main_s"}, "okpo_rules_erection": {"erection", "erection_s"}}
for b, sc in RULES_P1.items():
    assert js(f"results/runs/{b}/config.json")["yard"] == "yupu" and set(js(f"results/runs/{b}/config.json")["scenarios"]) == sc
REFUSAL_SUM = js("results/saturation/refusal_summary.json")
p1_checks = ([js(d + "/check.json")["findings"]["T1_tight_check"]
              for d in (RHO_SAFE["yantai"], "results/" + D22["diag"]["yantai"], MAIN_SAFE["yantai"])]
             + [dem22[D22["demand"]["yantai"]], point_check(FLAG_RUNS["yantai"], lambda r: r["scen"] == "main")]
             + [point_check(f"results/runs/{b}") for b in RULES_P1] + [check25f, check25g]
             + [REFUSAL_SUM[k] for k in ("t1_check", "t1_check_dock", "t1_check_refusal")])
assert all(c.get("over_strict", c.get("over", 0)) == 0 for c in p1_checks)
num("CheckPoints", sum(c["points"] for c in p1_checks))
num("CheckMaxRatioPone", max([c["max_ratio"] for c in p1_checks] + list(sens.values())
                             + [L30_SUM["t1_check"]["max_ratio"], SCREEN_SUM["t1_check"]["max_ratio"]]), 3)
# screening by demand over the bound (day length scaled): extra fleet of full-path reservation over free flow
SCR = []
for y in YARDS:
    for h in (16.0, 24.0):
        for n in sorted({int(float(r["demand"])) for r in DEM if r["yard"] == y}):
            f_, r_ = kreq(y, h, "whole", "free", n), kreq(y, h, "whole", "reserve", n)
            if f_ and r_ and f_["K_req"].isdigit() and r_["K_req"].isdigit():
                SCR.append((n / (sysm[y] * h / 16), int(r_["K_req"]) / int(f_["K_req"]) - 1))
lo_ = min(SCR)
num("ScreenCells", len(SCR))
num("ScreenRatioLo", lo_[0], 2)
pct("ScreenExtraMin", min(e for _, e in SCR))
# the 10 % screen located with the low-demand levels (screen.csv: the cells of the required-fleet table and the
# low-demand cells, integer fleets; screen_cross.csv: crossings interpolated linearly in the ratio between adjacent levels of each yard and day length)
SCRN = [r for r in rows("results/fleet/screen.csv") if r["gap"]]          # cells where both rules reach demand
T5 = [(float(r["ratio"]), float(r["gap"])) for r in SCRN if r["src"] == "table5"]
# the table cells of screen.csv are the four-day fleets; Okpo's headline cells are eight-day values here, which differ in
# one cell (350 blocks/day, 24 h: 16 -> 15 under full-path reservation), far above every crossing and still above 10 %
odd = [(a, b) for a, b in T5 if not any(abs(a - c) < 1e-9 and abs(b - e) < 1e-9 for c, e in SCR)]
r350 = next(r for r in SCRN if (r["yard"], r["day_h"], r["demand"]) == ("yupu", "24", "350"))
assert len(T5) == len(SCR) and odd == [(float(r350["ratio"]), float(r350["gap"]))]
new350 = int(MAC["YuFleetAnchorHTwentyFourRes"]) / int(kreq("yupu", 24.0, "whole", "free", 350)["K_req"]) - 1
assert any(abs(c - float(r350["ratio"])) < 1e-9 and abs(e - new350) < 1e-9 for c, e in SCR) and new350 > 0.10
CRS = {(r["yard"], r["day_h"]): r for r in rows("results/fleet/screen_cross.csv") if r["measure"] == "gap"}
first = [float(r["first"]) for k, r in CRS.items() if k[0] != "all"]
last = [float(r["last"]) for k, r in CRS.items() if k[0] != "all"]
assert abs(float(CRS[("all", "all")]["first"]) - min(first)) < 1e-12
# below the earliest crossing every cell needs the same fleet under both rules; above the latest every cell needs more
# than 10 % more vehicles under full-path reservation (the screen holds as a band, not as one value)
assert all(float(r["gap"]) == 0 for r in SCRN if float(r["ratio"]) < min(first))
assert all(float(r["gap"]) > 0.10 for r in SCRN if float(r["ratio"]) > max(last))
assert float(r350["ratio"]) > max(last)
pct("ScreenFirstLo", min(first), 1)
pct("ScreenFirstHi", max(first))
pct("ScreenFirstHiOne", max(first), 1)
pct("ScreenAllAbove", max(last))
# fractional fleets, interpolated between K_req - 1 and K_req (measure gap_frac; screen_band.json checks them against
# the integer crossings): the first 10 % crossing lies at the same ratio for Okpo on both working days, and spans a range at Yantai
CRF = {(r["yard"], r["day_h"]): r for r in rows("results/fleet/screen_cross.csv") if r["measure"] == "gap_frac"}
yu_f = [float(CRF[("yupu", h)]["first"]) for h in ("16", "24")]
ya_f = [float(CRF[("yantai", h)]["first"]) for h in ("16", "24")]
assert fmt(100 * yu_f[0], 1) == fmt(100 * yu_f[1], 1)
pct("ScreenFracYu", yu_f[0], 1)
pct("ScreenFracYaLo", min(ya_f), 1)
pct("ScreenFracYaHi", max(ya_f), 1)
BAND33 = js("results/fleet/screen_band.json")["screen_band"]["band"]
assert abs(BAND33["frac"]["first_lo"] - min(yu_f + ya_f)) < 1e-12 and abs(BAND33["frac"]["first_hi"] - max(yu_f + ya_f)) < 1e-12
assert abs(BAND33["int"]["first_lo"] - min(first)) < 1e-12 and abs(BAND33["int"]["first_hi"] - max(first)) < 1e-12
band = [int(r["K_free"]) for r in SCRN if min(first) <= float(r["ratio"]) <= max(last)]
put("ScreenOneVehicle", f"{fmt(100 / max(band))}--{fmt(100 / min(band))}" + r"\%")   # one vehicle, % of the free-flow fleet
LOWN = {y: sorted({int(r["demand"]) for r in SCRN if r["src"] == "low_demand" and r["yard"] == y}) for y in YARDS}
assert len(LOWN["yupu"]) == len(LOWN["yantai"]) and all(max(v) < 300 for v in LOWN.values())   # below the required-fleet table
num("ScreenLevels", len(LOWN["yupu"]))                               # low-demand levels per yard, both day lengths
for y in YARDS:
    rng("ScreenDemand" + Y[y], LOWN[y])
# external anchor and handling time: driving part of the free-flow cycle at Okpo (30 vehicles, 24-h days), and the handling
# time each at which free flow would fall to Hyundai's rate (Park and Seo 2006: about 500 blocks/day with 24 transporters)
fr24 = float(SEC72["YuPerTruckFreeDay"]["r1833"])
drive = 24 * 60 / fr24 - 2 * P["load_s"] / 60
HND = L30_SUM["anchor_reads"]["handling"]
assert abs(HND["cycle_at_anchor_min"] - 24 * 60 / (500 / 24)) < 0.01 and abs(HND["travel_free_min"] - drive) < 0.1
num("AnchorDriveMin", drive)
num("AnchorHandleMax", (24 * 60 / (500 / 24) - drive) / 2)
num("AnchorCycleMin", HND["cycle_at_anchor_min"])
num("MaxVehClear", max(P["block_l"][1], P["plat_len"]) + P["long_clear"])
# the anchor framed by handling time (handling_time.csv): blocks per vehicle-day at Okpo, 30 vehicles, 24-h days, t min each
AF = rows("results/anchors/handling_time.csv")
assert abs(float(AF[0]["free"]) - fr24) < 1e-9 and abs(float(AF[0]["anchor"]) - 500 / 24) < 1e-9
assert all((float(r["free"]) > float(r["anchor"])) == (r["free_above"] == "True") for r in AF)
# the anchor figure draws the same numbers: its first column is the main handling time, the rates of
# per_vehicle_rates.csv
assert float(AF[0]["load_min"]) == float(MAC["LoadMin"])
for col, m in (("free", "Free"), ("reference", "Ref"), ("reserve", "Res")):
    assert fmt(AF[0][col], 1) == fmt(SEC72[f"YuPerTruck{m}Day"]["r1833"], 1)
# required fleets with 30-min loading and unloading (fleet_30min.csv), whole roads, both yards: rows of the
# required-fleet table
FL30 = {(r["yard"], r["day_h"], r["model"], int(r["demand"])): r for r in rows("results/fleet/fleet_30min.csv")}
for (y, g), r in ((("yupu", "whole"), L30[("yupu", "whole", "reserve")]), (("yantai", "whole"), L30[("yantai", "whole", "reserve")])):
    for h in ("16", "24"):
        assert all(abs(float(f["T1_30"]) - float(r["T1_30"]) * int(h) / 16) < 0.01 for k, f in FL30.items() if k[:2] == (y, h))
assert not any(FL30[("yupu", h, m, n)]["fail_above"] not in ("", "[]") for h in ("16", "24") for m in ("free", "reserve")
               for n in (339, 350))                                                  # own volume: no a-marks needed
# Okpo's own volume and its anchor (339 and 350 blocks/day) as one range, printed as a single number where the two
# cells agree (e.g. 15 for full-path reservation on 24-h days)
for h, hn in ((16.0, "Sixteen"), (24.0, "TwentyFour")):
    for model, mn in (("free", "Free"), ("reserve", "Res")):
        a, b = (kval(kreq("yupu", h, "whole", model, n)).replace(">", r"$>$") for n in (339, 350))
        put(f"YuFleetOwnRangeH{hn}{mn}", a if a == b else f"{a}--{b}")
# roads shorter than a vehicle with its clearance (short_roads.csv): share of the road length, none a binding resource
assert all(r["n_binding"] == "0" and float(r["thr"]) == max(P["block_l"][1], P["plat_len"]) + P["long_clear"]
           for r in SHORT.values())
for y in YARDS:
    pct(Y[y] + "ShortShare", SHORT[(y, "whole")]["share_len"])
# why the lower end is low: Okpo, 30 vehicles, 16-h days, main mix on the map with floating docks (okpo_rules batch,
# per-point cycle parts in min per task): off-road wait for the booking under full-path reservation, the no-wait cycle of
# free flow, and the loaded and empty hauls that the cycle implies at the rated speeds
CYC = [r for r in rows("results/runs/okpo_rules/points.csv") if r["scen"] == "main" and r["K"] == "30"]
cyc = {m: [r for r in CYC if r["model"] == m] for m in ("free", "reserve")}
assert all(len(v) == 10 for v in cyc.values())
mean = lambda m, c: statistics.mean(float(r[c]) for r in cyc[m])
assert abs(mean("free", "cyc_wait")) < 1e-6
num("YuResWaitMin", mean("reserve", "cyc_wait"), 1)
num("YuNoWaitCycleMin", mean("free", "cyc_total"), 1)
num("YuHaulLoaded", mean("free", "cyc_loaded") * 60 * P["v_loaded"] / 1000, 1)
num("YuHaulEmpty", mean("free", "cyc_empty") * 60 * P["v_empty"] / 1000, 1)
num("OkHaul", 6)                 # Hanwha Systems customer case: average haul 6 km, 5 km after the improvement
num("OkHaulAfter", 5)
# seed-bootstrap ranges of the required fleet at Okpo's own volume, 16-h days (K_lo/K_hi of src.demand: the eight-day
# cells of fleet_8day.csv)
for model, mn in (("free", "Free"), ("reserve", "Res")):
    lo_b = [int(kreq("yupu", 16.0, "whole", model, n)["K_lo"]) for n in (339, 350)]
    hi_b = [int(kreq("yupu", 16.0, "whole", model, n)["K_hi"]) for n in (339, 350)]
    put("YuOwnBoot" + mn, f"{min(lo_b)}--{max(hi_b)}")
assert max(int(kreq("yupu", 16.0, "whole", "free", n)["K_hi"]) for n in (339, 350)) < \
    min(int(kreq("yupu", 16.0, "whole", "reserve", n)["K_lo"]) for n in (339, 350))   # the ranges do not overlap
# origin-destination sampling by stop instead of by entity (OD_RUNS, bounds only, the current maps: Okpo rev 1833, Yantai
# rev 396; settings and seeds of the earlier sampled-bound batches): change of the mean sampled bound; Okpo stays bound
# by roads, Yantai by building doors, under od_stop by those of another workshop (the steel-processing shop, building 007)
OD = {}
for y in YARDS:
    b = rows(OD_RUNS[y] + "/bounds.csv")
    for sc in ("main", "od_stop"):
        rs = [r for r in b if r["scen"] == sc]
        assert len(rs) == 10
        OD[(y, sc)] = statistics.mean(float(r["T1_tight_per_day"]) for r in rs)
        door = ["entrance" in r["tight_binding_1"] for r in rs]
        assert all(door) if y == "yantai" else not any(door)
        if (y, sc) == ("yantai", "od_stop"):
            assert all("building007" in r["tight_binding_1"] for r in rs)
        if (y, sc) == ("yantai", "main"):
            assert sum("building014" in r["tight_binding_1"] for r in rs) == 9
assert OD[("yupu", "od_stop")] < OD[("yupu", "main")] and OD[("yantai", "od_stop")] > OD[("yantai", "main")]
pct("OdYuDrop", 1 - OD[("yupu", "od_stop")] / OD[("yupu", "main")], 1)
pct("OdYaRise", OD[("yantai", "od_stop")] / OD[("yantai", "main")] - 1)
assert round(OD[("yupu", "main")]) == 1720 and round(OD[("yantai", "main")]) == 1252     # mean sampled bounds, main scenario
ODS = {r["yard"]: r for r in rows("results/bounds/od_sampling.csv")}
assert all(abs(float(ODS[y]["main_mean"]) - OD[(y, "main")]) < 1e-3 and abs(float(ODS[y]["od_mean"]) - OD[(y, "od_stop")])
           < 1e-3 for y in YARDS)

# ---------------------------------------------------------------- tables
TAB = ROOT / "tables"
TAB.mkdir(exist_ok=True)


def write(name, lines):
    (TAB / name).write_text("% generated by scripts/make_numbers.py; do not edit\n" + "\n".join(lines) + "\n",
                            encoding="utf-8")


# bounds table
yname = {"yupu": "Okpo", "yantai": "Yantai"}
scname = {"main": "Main mix", "erection": "Erection-heavy mix", "crane": "Crane-paced erection"}
lines = []
for y in YARDS:
    for sc in SC:
        s, f = SYS[y][sc], RF[y][sc]
        lo, hi = max(0.0, float(f["margin_lo_rel"])), max(0.0, float(f["margin_hi_rel"]))
        margin = "0" if hi < 1e-9 else f"{100*lo:.1f}--{100*hi:.1f}"
        typ = RO[y][sc]["type"]
        typ = {"road": "road", "entrance": "door"}[typ]
        lines.append(f"{yname[y]} & {scname[sc]} & {fmt(s['T1_system'])} & {fmt(s['ref_min'])}--{fmt(s['ref_max'])}"
                     f" & {fmt(f['T1pp'])} & {margin} & {res_name(s['top1'])} ({typ}) & "
                     f"{100*float(RO[y][sc]['pass_share']):.1f} \\\\")
write("tab_bounds.tex", lines)


# required fleet table
def cell(y, h, model, n):
    r = kreq(y, h, "whole", model, n)
    if r is None:
        return "--"
    v = r["K_req"]
    lo, hi = KSTAR[y]
    k = 10 ** 6 if not v.isdigit() else int(v)
    s = v.replace(">", "$>$")
    if r["fail_above"] not in ("", "[]"):
        s += r"$^{a}$"
    if k >= hi:
        return r"\textbf{" + s + "}"
    if k >= lo:
        return r"\underline{" + s + "}"
    return s


def crane_cell(y, h, model, n):
    """Crane-paced erection (CRANE_FLEET demand.csv): the same search, unmarked (K* was read for the main mix)."""
    r = E_DEM[y][(f"crane_d{int(h)}", n, model)]
    k = FLK[(y, str(int(h)), str(n), model)]["K_crane"]
    assert r["K_req"] == (k or r["K_req"]) and (k or r["K_req"].startswith(">"))
    return r["K_req"].replace(">", r"$>$") + (r"$^{a}$" if r["fail_above"] not in ("", "[]") else "")


def fleet_rows(y, levels, crane, l30, first):
    """Rows of a required-fleet table for one yard: main mix (marked against K*, the onset of saturation of full-path
    reservation), crane-paced erection and 30-min loading and unloading (both not marked)."""
    lines = [r"\multicolumn{9}{l}{" + first + r" (onset of saturation $K^{*}$ = "
             + (f"{KSTAR[y][0]}--{KSTAR[y][1]}" if KSTAR[y][0] != KSTAR[y][1] else f"{KSTAR[y][0]}") + r")}\\"]
    for n in levels:
        c = [cell(y, h, m, n) for h in (16.0, 24.0) for m in ("free", "reserve", "segment_safe", "segment")]
        lines.append(f"{fmt(n)} & " + " & ".join(c) + r" \\")
    lines.append(r"\multicolumn{9}{l}{\quad\emph{crane-paced erection}}\\")
    for n in crane:
        c = [crane_cell(y, h, m, n) if m in ("free", "reserve") else "--"
             for h in (16.0, 24.0) for m in ("free", "reserve", "segment_safe", "segment")]
        lines.append(f"{fmt(n)} & " + " & ".join(c) + r" \\")
    # 30-min loading and unloading (fleet_30min.csv), free flow and full-path reservation, not marked for K*; dagger: demand above
    # the system bound with 30-min handling
    lines.append(r"\multicolumn{9}{l}{\quad\emph{30-min loading and unloading}}\\")
    for n in l30:
        c = []
        for h in ("16", "24"):
            for m in ("free", "reserve", "segment_safe", "segment"):
                r = FL30.get((y, h, m, n))
                if r is None:
                    c.append("--")
                    continue
                s = r["cell"].replace("ᵃ", "").replace("> ", ">").replace(">", r"$>$")
                assert s.replace("$>$", "") in (r["K_req"].lstrip(">"), "150") and r["cell"].endswith("ᵃ") == (
                    r["fail_above"] not in ("", "[]"))
                s += (r"$^{a}$" if r["cell"].endswith("ᵃ") else "") + (r"$^{\dagger}$" if r["above_bound"] == "True" else "")
                c.append(s)
        lines.append(f"{fmt(n)} & " + " & ".join(c) + r" \\")
    return lines


# the headline rows of the main text: Okpo's own volume and peak, both working days; crane pacing and 30-min handling at
# the own volume
assert {339, 350, 678, 700} == {own, direct, 2 * own, 2 * direct}
write("tab_fleethead.tex", fleet_rows("yupu", (339, 350, 678, 700), (339,), (339, 350), r"\emph{Main mix}"))
# dock mouth: full-path reservation loses less than 10 % everywhere (stated in the text)
assert not any(RK[f"{y}|{sec}|reserve|{K}"]["thr_table"] for y in YARDS for sec in ("whole", "stops") for K in (20, 40, 60, 100, 150))

# task flows and their weights (FLOW in occupancy/params.py): rows of the parameter table (Table B.1)
ROLE = {"assembly_workshop": "assembly workshop", "outfitting_workshop": "outfitting workshop",
        "paint_workshop": "paint shop", "yard|buffer": "stockyard or buffer", "assembly_yard": "assembly area",
        "dry_dock": "dry dock"}
FLOWS = re.findall(r'\("P(\d)", "([\w|]+)", "([\w|]+)", ([\d.]+)\)', show("occupancy/params.py"))
assert [int(f[0]) for f in FLOWS] == list(range(1, 9))
wsum = sum(float(f[3]) for f in FLOWS)
assert abs(float(FLOWS[5][3]) / wsum - float(el[0]["P6_share"])) < 1e-12 and FLOWS[5][2] == "dry_dock"
fl = [rf"{ROLE[a]} $\to$ {ROLE[b_]}: {fmt(w, 2 if len(w.split('.')[-1]) > 1 else 1)} ({100 * float(w) / wsum:.1f}\%)"
      for _, a, b_, w in FLOWS]
write("tab_flows.tex", [rf"\multicolumn{{4}}{{@{{}}p{{14.5cm}}@{{}}}}{{\quad {fl[i]}; {fl[i + 1]}}} \\" for i in range(0, 8, 2)])
# the weights are mean passes of a block along each flow; tasks are drawn in proportion to them (occupancy/tasks.py:
# rng.choices(FLOW, weights=...)), so only their shares enter the model, and demand counts moves directly
assert "rng.choices(FLOW, weights=self.w)" in show("occupancy/tasks.py")
num("FlowTotal", wsum, 2)

# ---------------------------------------------------------------- numbers.tex
out = ["% generated by make_numbers.py from the result files; do not edit"]    # neutral: the file goes to the journal
out += [rf"\newcommand{{\{k}}}{{{v}\xspace}}" for k, v in MAC.items()]
(ROOT / "numbers.tex").write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"{len(MAC)} macros, tables in {TAB}")

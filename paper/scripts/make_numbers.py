"""Write numbers.tex (one LaTeX macro per number of the text) and tables/*.tex from the result data of the repository.

Every number is read from results/ (paths relative to the repository root, read through src.py; results/MANIFEST.csv
lists each file) and checked with assertions as it is read. Sources, by directory of results/:
  main_case/       the main case (erection flow conserved, every other move between areas): system bounds of the mappings
                   and demand ratios, binding elements, saturation curves and plateaus, capacity interval, fleets K_req
                   and K_stable of the four rules, the scan of moves within areas, crane load by dock, peak waits
  intra_area/      moves within areas: bounds, fleets, screen, placement variants; saturation; fleets without growing
                   backlog over 30 days (stable.csv, both inter-area mappings and this one)
  network/         topology of the two yard networks (okpo_topology.json, yantai_topology.json), the Okpo map revisions
                   (okpo_revisions.json), sectioned roads, roads shorter than a vehicle (short_roads*.csv), crane spans
  bounds/          system, route-free and sampled bounds of each yard with every move scaled ({okpo,yantai}/: system.csv,
                   route_free.csv, convergence.csv, readout.csv, prices.csv, config.json); system bounds by granularity
                   (system_by_granularity.csv), their sensitivity to the inputs, sampling of origins and destinations;
                   route-free bounds of the main case, the bound with the utilisation rows only, platform size
  saturation/      saturation curves, plateaus and per-point checks with every move scaled (rules_summary.json; Okpo on
                   rev 1833: okpo_curves.csv, okpo_readings.csv, okpo_curves_check.json), onset of saturation K*
                   (kstar.csv), all rules at saturation (rules.csv), refusal and teleport diagnostics, sensitivity
  interval/        capacity interval (interval.csv), crane-paced interval with Okpo's 16-day plateau and Yantai's
                   eight-day check, 30-min handling, erection load
  fleet/           required fleets with every move scaled (required_fleet.csv, with fleet_8day.csv for Okpo's headline
                   cells; src.demand), the K* marks, crane-paced and 30-min fleets, the 10 % screen (screen.csv,
                   screen_cross.csv, screen_band.json)
  ablation/        the ablation of the occupancy representation (both mappings)
  retiming/        whole-network retiming and offline dispatch
  subnet/          sub-network instances: best schedules, the through corridor
  screen/          the screen in two levels (demand ratio and chain load), across yards and road widths, low levels
  stochastic/      random handling and driving times, late rebooking, execution in booked order
  width_ban/       the ban on blocks wider than the passable width
  tail/            release at tail clearance: roadside stops, plateaus and fleets when the tail is booked
  hand/            the hand-computed instances
  dock/            dock-mouth thresholds (thresholds.csv, summary.json), crane hours, utilisation levels, one crane at
                   Dock 1, off-road unloading bays (bays.csv)
  anchors/         moves per vehicle-day against handling time and by rule (the external anchor)
  synthetic_dock/  synthetic dock mouth: thresholds rho_10 (rho_star.csv), per-point throughput, waiting term
  runs/<run>/      configuration (config.json), per-point results (points.csv) and per-point bound check (check.json) of
                   the simulation batches; the run names are in src.py
The model parameters, task flows, the geometry of the synthetic dock mouth and the 30-day criterion are read from
occupancy/params.py, occupancy/tasks.py, experiments/dock_mouth.py and experiments/daily.py.
Okpo's readings are on map revision 1833 (with the floating docks). Some inputs come from runs on revision 1825, which
differs only by three spur roads at the floating docks (model parameters, bounds, utilisation levels, per-point checks);
where a reading differs between the revisions, the rev 1833 value is used and the other is only checked.
Only the macros and tables that the paper uses are written: the macros of the text, the tables and the Highlights, those
the figures check, and the computations and checks they rest on. The tables are those printed: tab_bounds, tab_ablation,
tab_fleethead (main text), tab_flows (Appendix B), tab_fleethead_prop, tab_ablation_prop, tab_subnet, tab_whole,
tab_stoch and tab_dockbay (supplementary material).

Run from paper/: python scripts/make_numbers.py (or, from the repository root, python paper/reproduce.py).
"""
import json
import re
import statistics
from pathlib import Path

from src import (ABL_BOUNDS, ABL_SUMMARY, BOUNDS_DIR, CRANE_FLEET, CRANE_RUNS, FLAG_RUNS, MAIN_SAFE, OD_RUNS, RETIME, RHO_SAFE,
                 SEC_FLEET, SYN_DOCK, SYN_WAIT, demand, interval, js, res_name, rows, show)

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


close = lambda a, b, tol=1e-9: abs(float(a) - float(b)) < tol


def prange(name, vals, nd=0):
    """A range of shares (fractions) as one macro, 'lo--hi%', or 'v%' where both ends round alike."""
    lo, hi = fmt(100 * min(vals), nd), fmt(100 * max(vals), nd)
    put(name, (lo if lo == hi else f"{lo}--{hi}") + r"\%")


def irange(name, vals):
    """A range of integers (fleets, counts) as one macro."""
    lo, hi = min(vals), max(vals)
    put(name, f"{lo}" if lo == hi else f"{lo}--{hi}")


def pair(name, vals, nd=0):
    put(name, " and ".join(fmt(v, nd) for v in vals))


Y = {"yupu": "Yu", "yantai": "Ya"}


# ---------------------------------------------------------------- saturation summary (rules_summary.json)
# (with the safe-release fleet searches filled below the first success)
S = js("results/saturation/rules_summary.json")
SAT = S["sat"]
# onset of saturation K* (kstar.csv), read on the saturation curves with every move scaled (Okpo rev 1833, Yantai the curve of
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
assert check25f["points"] == 11790             # with the safe-release searches filled below the first success
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
# the reference rule (optimistic: it teleports out of deadlocks) against free flow at the own volume on 16-h days, as
# above: the lower limit of the headline (Section 4.3, conclusions); its cells are four-day values, and free flow's
# eight-day values equal its four-day ones (fleet8d.csv changes one full-path reservation cell only, asserted below)
extra_ref = [int(kreq("yupu", 16.0, "whole", "segment", n)["K_req"]) / int(kreq("yupu", 16.0, "whole", "free", n)["K_req"])
             - 1 for n in (339, 350)]
pct("YuOwnRefExtraLo", min(extra_ref))
pct("YuOwnRefExtraHi", max(extra_ref))
# the conclusions say it in words: more than half on 16-h days, a quarter on 24-h days (eight-day values)
extra24 = [int(kreq("yupu", 24.0, "whole", "reserve", n)["K_req"]) / int(kreq("yupu", 24.0, "whole", "free", n)["K_req"]) - 1
           for n in (339, 350)]
assert min(extra) > 0.5 and all(abs(e - 0.25) < 1e-9 for e in extra24)
# headline fleets over eight simulated days (fleet_8day.csv, applied in src.demand): seven of the eight cells keep the
# four-day value; full-path reservation at 350 moves/day on 24-h days needs one vehicle less, inside the four-day seed
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
for r in rows("results/fleet/kstar_marks.csv"):          # filled searches: Yantai, 16 h, 300 moves/day, safe release
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
# moves per vehicle-day at Okpo with 30 vehicles, on the map with floating docks (per_vehicle_rates.csv, column r1833):
# 16-h days in the text, 24-h days ("Day") in the anchor figure
SEC72 = {r["macro"]: r for r in rows("results/anchors/per_vehicle_rates.csv")}
assert set(SEC72) == {f"YuPerTruck{m}{d}" for m in ("Free", "Res", "Ref", "Safe") for d in ("", "Day")}
for m_ in ("Free", "Res", "Ref", "Safe"):
    num("YuPerTruck" + m_, SEC72["YuPerTruck" + m_]["r1833"], 1)
num("HHIRate", 500 / 24, 1)        # Park and Seo (2006): about 500 moves/day with 24 transporters, 24-h days
num("ShenRate", 600 / 30)          # Shen et al. (2018): about 600 moves/day with about 30 transporters

# ---------------------------------------------------------------- bounds (results/bounds)
SYS = {y: {r["scen"]: r for r in rows(BOUNDS_DIR[y] + "/system.csv")} for y in YARDS}
RF = {y: {r["scen"]: r for r in rows(BOUNDS_DIR[y] + "/route_free.csv")} for y in YARDS}
CONV = {y: rows(BOUNDS_DIR[y] + "/convergence.csv") for y in YARDS}
RO = {y: {r["scen"]: r for r in rows(BOUNDS_DIR[y] + "/readout.csv")} for y in YARDS}
PR = {y: rows(BOUNDS_DIR[y] + "/prices.csv") for y in YARDS}
SC = {"main": "Main", "erection": "Er", "crane": "Crane"}
num("YuSysMain", SYS["yupu"]["main"]["T1_system"])
num("YuSysCrane", SYS["yupu"]["crane"]["T1_system"])
# Table 7 note: sampled bounds are ten task samples of MixN tasks each (the ten seeds of the bound-sample batches that
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
# the route-free bound (certificate) where the route margin is large, Okpo under crane-paced erection: the upper end of
# the capacity interval for rules that may reroute (Section 4.3, Fig. 5)
num("YuRFCrane", RF["yupu"]["crane"]["T1pp"])
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
    return ks[0] if len(ks) == 1 else ", ".join(ks[:-1]) + " and " + ks[-1]


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
# one cell (350 moves/day, 24 h: 16 -> 15 under full-path reservation), far above every crossing and still above 10 %
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
BAND33 = js("results/fleet/screen_band.json")["screen_band"]["band"]
assert abs(BAND33["frac"]["first_lo"] - min(yu_f + ya_f)) < 1e-12 and abs(BAND33["frac"]["first_hi"] - max(yu_f + ya_f)) < 1e-12
# three further levels on 16-h days at each yard, at 2-6 % of the bound (low.csv, variant base = the paper's roads):
# both rules need the same fleet at each, below every first crossing; they move Okpo's fractional first crossing on
# 16-h days from 5.29 % to 5.48 % (low_cross.csv, levels all against earlier = the earlier screen's) and leave the other crossings
LC = {(r["yard"], r["variant"], r["measure"], r["levels"]): float(r["first"]) for r in rows("results/screen/low_cross.csv")}
LL = {(r["yard"], r["variant"], r["measure"], r["levels"]): float(r["last"]) for r in rows("results/screen/low_cross.csv")}
LOW53 = [r for r in rows("results/screen/low_levels.csv") if r["variant"] == "base"]
idx31 = {(r["yard"], int(r["demand"])): r for r in SCRN if r["day_h"] == "16"}
for r in LOW53:
    if r["new"] == "False":                          # the old levels are the cells of fleet/screen.csv
        q = idx31[(r["yard"], int(r["n"]))]
        assert all(close(r[c], q[c]) for c in ("ratio", "gap", "gap_frac")) and (r["K_free"], r["K_res"]) == (q["K_free"], q["K_res"])
    else:
        assert float(r["gap"]) == 0 and float(r["ratio"]) < min(first)
for y, f_ in (("yupu", yu_f), ("yantai", ya_f)):
    assert close(LC[(y, "base", "gap_frac", "earlier")], f_[0]) and close(LC[(y, "base", "gap", "all")], float(CRS[(y, "16")]["first"]))
    assert close(LL[(y, "base", "gap", "all")], float(CRS[(y, "16")]["last"]))
    f_[0] = LC[(y, "base", "gap_frac", "all")]
assert ya_f[0] == float(CRF[("yantai", "16")]["first"]) and fmt(100 * yu_f[0], 2) == "5.48"
prange("ScreenFracYu", yu_f, 1)
pct("ScreenFracYaLo", min(ya_f), 1)
pct("ScreenFracYaHi", max(ya_f), 1)
DENSE = {y: sorted(int(r["n"]) for r in LOW53 if r["yard"] == y and r["new"] == "True") for y in YARDS}
assert len(DENSE["yupu"]) == len(DENSE["yantai"])
num("ScreenDenseLevels", len(DENSE["yupu"]))
for y in YARDS:
    put("ScreenDense" + Y[y], ", ".join(map(str, DENSE[y][:-1])) + f" and {DENSE[y][-1]}")
assert abs(BAND33["int"]["first_lo"] - min(first)) < 1e-12 and abs(BAND33["int"]["first_hi"] - max(first)) < 1e-12
band = [int(r["K_free"]) for r in SCRN if min(first) <= float(r["ratio"]) <= max(last)]
put("ScreenOneVehicle", f"{fmt(100 / max(band))}--{fmt(100 / min(band))}" + r"\%")   # one vehicle, % of the free-flow fleet
LOWN = {y: sorted({int(r["demand"]) for r in SCRN if r["src"] == "low_demand" and r["yard"] == y}) for y in YARDS}
assert len(LOWN["yupu"]) == len(LOWN["yantai"]) and all(max(v) < 300 for v in LOWN.values())   # below the required-fleet table
num("ScreenLevels", len(LOWN["yupu"]))                               # low-demand levels per yard, both day lengths
for y in YARDS:
    rng("ScreenDemand" + Y[y], LOWN[y])
# external anchor and handling time: driving part of the free-flow cycle at Okpo (30 vehicles, 24-h days), and the handling
# time each at which free flow would fall to Hyundai's rate (Park and Seo 2006: about 500 moves/day with 24 transporters)
fr24 = float(SEC72["YuPerTruckFreeDay"]["r1833"])
drive = 24 * 60 / fr24 - 2 * P["load_s"] / 60
HND = L30_SUM["anchor_reads"]["handling"]
assert abs(HND["cycle_at_anchor_min"] - 24 * 60 / (500 / 24)) < 0.01 and abs(HND["travel_free_min"] - drive) < 0.1
num("AnchorDriveMin", drive)
num("AnchorHandleMax", (24 * 60 / (500 / 24) - drive) / 2)
num("AnchorCycleMin", HND["cycle_at_anchor_min"])
num("MaxVehClear", max(P["block_l"][1], P["plat_len"]) + P["long_clear"])
# the anchor framed by handling time (handling_time.csv): moves per vehicle-day at Okpo, 30 vehicles, 24-h days, t min each
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
# Okpo's own volume and its anchor (339 and 350 moves/day) as one range, printed as a single number where the two
# cells agree (e.g. 15 for full-path reservation on 24-h days); the reference rule for the headline range
for h, hn in ((16.0, "Sixteen"), (24.0, "TwentyFour")):
    for model, mn in (("free", "Free"), ("reserve", "Res"), ("segment", "Ref")):
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


yname = {"yupu": "Okpo", "yantai": "Yantai"}       # the bounds table (Table 7) is written with the main case below


# required fleet table
def marked(v, fail="[]", ks=None):
    """A fleet as printed: '>' in maths, a when a larger fleet tested above it fails again; against K* = (lo, hi) bold
    at or above hi, underlined from lo (ks None: not marked)."""
    s = v.replace(">", "$>$") + (r"$^{a}$" if fail not in ("", "[]") else "")
    if ks is None:
        return s
    k = int(v) if v.isdigit() else 10 ** 6
    return r"\textbf{" + s + "}" if k >= ks[1] else r"\underline{" + s + "}" if k >= ks[0] else s


def cell(y, h, model, n):
    r = kreq(y, h, "whole", model, n)
    return "--" if r is None else marked(r["K_req"], r["fail_above"], KSTAR[y])


def crane_cell(y, h, model, n):
    """Crane-paced erection (CRANE_FLEET demand.csv): the same search, unmarked (K* was read for the main mix)."""
    r = E_DEM[y][(f"crane_d{int(h)}", n, model)]
    k = FLK[(y, str(int(h)), str(n), model)]["K_crane"]
    assert r["K_req"] == (k or r["K_req"]) and (k or r["K_req"].startswith(">"))
    return r["K_req"].replace(">", r"$>$") + (r"$^{a}$" if r["fail_above"] not in ("", "[]") else "")


def l30_cell(y, h, m, n):
    """30-min loading and unloading (fleet_30min.csv), not marked for K*; dagger: demand above the system bound at 30 min."""
    r = FL30.get((y, h, m, n))
    if r is None:
        return "--"
    s = r["cell"].replace("ᵃ", "").replace("> ", ">").replace(">", r"$>$")
    assert s.replace("$>$", "") in (r["K_req"].lstrip(">"), "150") and r["cell"].endswith("ᵃ") == (
        r["fail_above"] not in ("", "[]"))
    return s + (r"$^{a}$" if r["cell"].endswith("ᵃ") else "") + (r"$^{\dagger}$" if r["above_bound"] == "True" else "")


assert {339, 350, 678, 700} == {own, direct, 2 * own, 2 * direct}     # the rows of the headline tables (Tables 9, S2)
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

# ================================================================ demand mappings, random times, sub-networks, ban, screen
# Each block reads the runs' own summaries, checks them against the readings of the paper they extend, and writes the
# macros of Appendix B and the supplementary material and the tables tab_subnet, tab_stoch and tab_dockbay.
# --- demand mapping. Inter-area mapping (the paper's): each of the MovesPerBlock moves of a block runs along one
# of the eight flows. Intra-area mapping: FlowTotal of them do; the others run between two stops of one entity (area or
# workshop), or load and unload at its only stop. Bounds, ratios and fleets under both (bounds.csv, ratio.csv,
# fleet.csv, fleet_l30.csv, var.csv, var_bounds.csv, sampled.csv)
NMOVE = int(MAC["MovesPerBlock"])
num("IntraMoves", NMOVE - wsum, 2)
B48 = {(r["yard"], r["map"], int(r["load_min"]), int(r["day_h"])): r for r in rows("results/intra_area/bounds.csv")}
lam48 = lambda y, m="intra", lm=10, h=16: float(B48[(y, m, lm, h)]["T1_sys"])
for y in YARDS:
    assert close(lam48(y, "main"), SYS[y]["main"]["T1_system"], 0.05)            # the system bound of the paper
    assert close(lam48(y, "main", 30), L30[(y, "whole", "reserve")]["T1_30"], 0.05)
    for lm in (10, 30):
        assert lam48(y, "inter_only", lm) == lam48(y, "main", lm)                # same law of the flows, same bound
        assert all(close(lam48(y, m, lm, 24) / lam48(y, m, lm), 1.5) for m in ("main", "intra", "inter_only"))
for y, tag in (("yupu", "Yu"), ("yantai", "Ya")):
    num(tag + "IntraSys", lam48(y))
# binding under the intra-area mapping: both drive-in bays of one assembly area at Okpo; doors of the steel-processing
# and panel-line hall (building 007) at Yantai, Yantai's only assembly workshop, through which its rehandling passes
top48 = lambda y: [B48[(y, "intra", 10, 16)][f"top{i}"] for i in (1, 2, 3)]
assert all("zone012" in t and "assembly area" in t for t in top48("yupu")[:2]) and all("building007" in t for t in top48("yantai"))
R48 = {(r["yard"], int(r["day_h"]), int(r["level"]), r["map"]): float(r["ratio"])
       for r in rows("results/intra_area/ratio.csv") if r["load_min"] == "10"}
for (y, h, n, m), v in R48.items():                       # demand over the bound, both counted in moves of the mapping
    assert close(v, (n * wsum / NMOVE if m == "inter_only" else n) / lam48(y, m, 10, h))
prange("YuIntraDens", [R48[("yupu", 16, n, "intra")] for n in (339, 350)], 1)
prange("YuIntraDensDay", [R48[("yupu", 24, n, "intra")] for n in (339, 350)], 1)
prange("YuIntraDensPeak", [R48[("yupu", 16, n, "intra")] for n in (678, 700)], 1)
prange("YuInterOnlyDens", [R48[("yupu", 16, n, "inter_only")] for n in (339, 350)], 1)
pct("YaIntraDens", R48[("yantai", 16, 23, "intra")], 1)
assert fmt(100 * R48[("yupu", 16, 339, "main")]) == MAC["YuDensOwn"].replace(r"\%", "")
# the sampled bound (ten samples of MixN tasks) lies about a tenth below the system bound under the intra-area mapping:
# single-stop paint shops take a varying number of same-stop moves from sample to sample
SMP = {(r["yard"], int(r["load_min"]), r["bound"]): r for r in rows("results/intra_area/sampled.csv")}
smp = float(SMP[("yupu", 10, "network_tight")]["mean"])
assert all(close(SMP[(y, 10, "network_tight")]["T1_sys"], lam48(y)) for y in YARDS)
pct("YuIntraSampledBelow", 1 - smp / lam48("yupu"), 1)
prange("YuIntraDensSampled", [n / smp for n in (339, 350)], 1)
# intra-area moves handled at an entity's only stop, share of the intra-area moves of the samples (Okpo: mostly the
# doors of single-door paint shops; the share under the task law is not in a result file)
for y, tag in (("yupu", "Yu"), ("yantai", "Ya")):
    s_ = SMP[(y, 10, "network_tight")]
    pct(tag + "IntraSameStop", float(s_["same_stop_mean"]) / float(s_["n_intra_mean"]), 1)
# required fleets (Okpo, whole roads, main mix): the inter-area rows are the paper's (four days; eight days where the
# paper uses them); the intra-area rows read the same over eight days as over four
F48 = {(int(r["level"]), int(r["day_h"]), r["map"], int(r["days"])): r for r in rows("results/intra_area/fleet.csv")}
for (n, h, m, d), r in F48.items():
    if m == "main":
        for model in ("free", "reserve", "segment"):
            if r["K_" + model]:
                ref = (kreq if d == 8 else kreq4)("yupu", float(h), "whole", model, n)
                assert ref["K_req"] == r["K_" + model] and (ref["fail_above"] not in ("", "[]")) == (
                    r["fail_" + model] not in ("", "[]")), (n, h, d, model)
    elif d == 8:
        assert (r["K_free"], r["K_reserve"]) == (F48[(n, h, m, 4)]["K_free"], F48[(n, h, m, 4)]["K_reserve"])
assert all(r["fail_" + mo] in ("", "[]") for (n, h, m, d), r in F48.items() if m == "intra" for mo in ("free", "reserve", "segment"))
k48 = lambda n, h, model, m="intra": int(F48[(n, h, m, 4)]["K_" + model])
irange("YuIntraFleetFree", [k48(n, 16, "free") for n in (339, 350)])
irange("YuIntraFleetRes", [k48(n, 16, "reserve") for n in (339, 350)])
ext48 = [k48(n, 16, "reserve") / k48(n, 16, "free") - 1 for n in (339, 350)]
pct("YuIntraExtraLo", min(ext48))
pct("YuIntraExtraHi", max(ext48))
irange("YuIntraPeakFree", [k48(n, 16, "free") for n in (678, 700)])
irange("YuIntraPeakRes", [k48(n, 16, "reserve") for n in (678, 700)])
ext48p = [k48(n, 16, "reserve") / k48(n, 16, "free") - 1 for n in (678, 700)]
pct("YuIntraPeakExtraLo", min(ext48p))
pct("YuIntraPeakExtraHi", max(ext48p))
irange("YuIntraDayFree", [k48(n, 24, "free") for n in (339, 350)])
irange("YuIntraDayRes", [k48(n, 24, "reserve") for n in (339, 350)])
irange("YuInterOnlyFree", [k48(n, 16, "free", "inter_only") for n in (339, 350)])
irange("YuInterOnlyRes", [k48(n, 16, "reserve", "inter_only") for n in (339, 350)])
FL48 = {(int(r["level"]), int(r["day_h"]), r["map"]): r for r in rows("results/intra_area/fleet_30min.csv")}
for (n, h, m), r in FL48.items():
    if m == "main":                                       # the paper's 30-min cells (fleet_30min.csv)
        assert (r["K_free"], r["K_res"]) == (FL30[("yupu", str(h), "free", n)]["K_req"], FL30[("yupu", str(h), "reserve", n)]["K_req"])
irange("YuIntraLThirtyFree", [int(FL48[(n, 16, "intra")]["K_free"]) for n in (339, 350)])
irange("YuIntraLThirtyRes", [int(FL48[(n, 16, "intra")]["K_res"]) for n in (339, 350)])
# where the intra-area moves are placed (entities alike, stops alike, open areas only, both): the bound moves, Okpo's
# own-volume fleets stay within one vehicle
V48 = [r for r in rows("results/intra_area/variants.csv") if r["level"] in ("339", "350") and r["day_h"] == "16"]
VB48 = {(r["yard"], r["var"]): float(r["T1_sys"]) for r in rows("results/intra_area/variant_bounds.csv")
        if r["load_min"] == "10" and r["day_h"] == "16"}
assert len({r["var"] for r in V48}) == 4 and all(close(r["T1_sys"], lam48("yupu")) for r in V48 if r["var"] == "main")
assert {int(r["K_free"]) for r in V48} == {k48(n, 16, "free") for n in (339, 350)}
assert {int(r["K_res"]) for r in V48} == {k48(n, 16, "reserve") for n in (339, 350)}
vb = [lam48("yupu")] + [v for (y, _), v in VB48.items() if y == "yupu"]
assert len(vb) == 4
num("YuIntraVarBoundLo", min(vb))
num("YuIntraVarBoundHi", max(vb))
irange("YuIntraOpenFree", [int(r["K_free"]) for r in V48 if r["var"] == "open"])
irange("YuIntraOpenRes", [int(r["K_res"]) for r in V48 if r["var"] == "open"])
# the screen under the intra-area mapping (screen_cross.csv): fractional crossings inside the band of the paper; with
# integer fleets one low level, one vehicle apart, exceeds 10 %
X48 = {(int(r["day_h"]), r["measure"]): r for r in rows("results/intra_area/screen_cross.csv")}
for h, tag in ((16, "Sixteen"), (24, "TwentyFour")):
    pct("YuIntraCross" + tag, X48[(h, "gap_frac")]["first"], 1)
SC48 = [r for r in rows("results/intra_area/screen.csv") if r["day_h"] == "16"]
odd48 = min((float(r["ratio"]), r) for r in SC48 if float(r["gap"]) > 0.10)[1]
assert float(X48[(16, "gap")]["first"]) < float(odd48["ratio"]) and int(odd48["K_res"]) - int(odd48["K_free"]) == 1
pct("YuIntraOddRatio", odd48["ratio"], 1)
assert float(MAC["ScreenFirstLo"].replace(r"\%", "")) / 100 <= float(X48[(24, "gap_frac")]["first"]) \
    <= float(X48[(16, "gap_frac")]["first"]) <= float(MAC["ScreenAllAbove"].replace(r"\%", "")) / 100   # inside the band

# --- saturation under the intra-area mapping (sat_summary.csv) and fleets without backlog (stable.csv)
SS = {(r["yard"], r["scen"]): r for r in rows("results/intra_area/saturation.csv")}
cur = SS[("yupu", "main (proportional)")]
assert close(cur["plateau_reserve"], RESLOW[("yupu", "main", "whole")][0], 0.5) and close(cur["lam_S"], lam48("yupu", "main"))
assert (cur["K_star_20"], cur["K_star_20_lo"], cur["K_star_20_hi"]) == ("50", "45", "55") == \
    tuple(str(x) for x in (KST[("yupu", 0.2)]["K_star"], *KSTAR["yupu"]))
for y, tag in (("yupu", "Yu"), ("yantai", "Ya")):
    s_ = SS[(y, "main_in")]
    assert close(s_["lam_S"], lam48(y))
    num(tag + "IntraResPlat", s_["plateau_reserve"])
    pct(tag + "IntraResRatio", s_["ratio_lam"])
    put(tag + "IntraKStar", f"{s_['K_star_20']} ({s_['K_star_20_lo']}--{s_['K_star_20_hi']})")
pct("YuIntraResRatioSampled", SS[("yupu", "main_in")]["ratio_sampled"])
num("YuIntraStopPlat", SS[("yupu", "main_in_stop")]["plateau_reserve"])
pct("YuIntraStopRatio", SS[("yupu", "main_in_stop")]["ratio_lam"])
INTRA_KS = (int(SS[("yupu", "main_in")]["K_star_20_lo"]), int(SS[("yupu", "main_in")]["K_star_20_hi"]))
MK49 = rows("results/intra_area/marks.csv")
assert all(not (r["mark_10"] or r["mark_20"] or r["mark_30"]) for r in MK49) and max(int(r["K_req"]) for r in MK49) < INTRA_KS[0]
# fleets without backlog: 30-day runs with continuous release, both mappings, Okpo's own volume and peak
STB = {(r["map"], int(r["day_h"]), r["model"], int(r["demand"])): r for r in rows("results/intra_area/stable.csv")}
MP = {"inter": "proportional", "intra": "intra-area"}
STAB = ("free", "reserve")                                 # the rules of the 30-day runs
assert len(STB) == 32 and all(r["pending"] == "[]" for r in STB.values())
for (mp, h, model, n), r in STB.items():                   # their K_req are the four-day values of the paper and of the intra-area runs
    assert int(r["K_req"]) == (int(kreq4("yupu", float(h), "whole", model, n)["K_req"]) if mp == MP["inter"]
                               else k48(n, h, model))
CFG49 = js("results/runs/okpo_stable_30day/config.json")
assert {s["days"] for s in CFG49["scenarios"].values()} == {30} and len(CFG49["seeds"]) == int(MAC["Seeds"])
D49 = show("experiments/daily.py")
win = tuple(int(x) for x in re.search(r"^WIN = \((\d+), (\d+)\)", D49, re.M).groups())
tol = float(re.search(r"^TOL = ([\d.]+)", D49, re.M).group(1))
assert win[1] == 30 and "stable_tol=m <= TOL * n" in D49
num("StableDays", 30)
num("StableFrom", win[0])
pct("StableTol", tol, 1)


def kreq_tab(mp, h, model, n):
    """K_req as Table S2 prints it: eight-day values at the own volume under the inter-area mapping."""
    return int(kreq("yupu", float(h), "whole", model, n)["K_req"]) if mp == MP["inter"] else k48(n, h, model)


dstab = {k: int(r["K_stable"]) - kreq_tab(*k) for k, r in STB.items()}
assert min(dstab.values()) >= 0 and all(d <= 1 for (mp, h, mo, n), d in dstab.items() if n in (339, 350))
exc = sorted(k for k, d in dstab.items() if d > 1)
assert exc == sorted((MP["inter"], h, "reserve", n) for h in (16, 24) for n in (678, 700)), exc
num("StableCells", len(dstab))
num("StableSameOrOne", sum(d <= 1 for d in dstab.values()))
num("StableOwnCells", sum(1 for k in dstab if k[3] in (339, 350)))
ex16 = [STB[(MP["inter"], 16, "reserve", n)] for n in (678, 700)]
ex24 = [STB[(MP["inter"], 24, "reserve", n)] for n in (678, 700)]
pair("YuStableExcKreq", [int(r["K_req"]) for r in ex16])
pair("YuStableExc", [int(r["K_stable"]) for r in ex16])
pair("YuStableExcSlope", [float(r["slope_Kreq"]) for r in ex16])
pair("YuStableExcBacklog", [float(r["backlog30_Kreq"]) for r in ex16])
pair("YuStableExcFlowKreq", [float(r["flow_mean_Kreq"]) for r in ex16])
pair("YuStableExcFlow", [float(r["flow_mean_Kstable"]) for r in ex16])
pair("YuStableExcKreqDay", [int(r["K_req"]) for r in ex24])
pair("YuStableExcDay", [int(r["K_stable"]) for r in ex24])
assert all(float(r["slope_lo_Kreq"]) > 0 for r in ex16 + ex24)                 # a significant rise at K_req
assert all(abs(float(r["slope_Kstable"])) <= tol * n for r, n in zip(ex16 + ex24, (678, 700) * 2))
ci = [abs(int(r["K_stable_ci"]) - int(r["K_stable"])) for r in STB.values()]
num("StableCiDiff", sum(d > 0 for d in ci))
num("StableCiOne", sum(d == 1 for d in ci))
# headline fleets read with K_stable (own volume, 16-h days), both mappings
for mp, tag in (("inter", "Yu"), ("intra", "YuIntra")):
    ks_ = {m: [int(STB[(MP[mp], 16, m, n)]["K_stable"]) for n in (339, 350)] for m in STAB}
    irange(tag + "StableOwnFree", ks_["free"])
    irange(tag + "StableOwnRes", ks_["reserve"])
    e_ = [a / b - 1 for a, b in zip(ks_["reserve"], ks_["free"])]
    pct(tag + "StableOwnExtraLo", min(e_))
    pct(tag + "StableOwnExtraHi", max(e_))


# ================================================================ the main case (erection conserved) and its readings
# --- three mappings of demand to moves. A block makes MovesPerBlock moves; the eight flows weigh FlowTotal
# moves, the erection flow (assembly area -> dry dock) ErectPerBlock of them. Conserved mapping (the main case):
# ErectPerBlock erections a block, the other moves between areas along the other seven flows by their weights.
# Proportional mapping (the main case before): every move along the flows in proportion to their weights, so the
# erections are scaled by MovesPerBlock / FlowTotal. Intra-area mapping: IntraMoves moves within an area.
# bounds.csv (map: main = proportional, conserved, intra, crane = crane-paced erection), 16-h days
B61 = {(r["yard"], r["map"]): r for r in rows("results/main_case/bounds.csv")}
erect = float(FLOWS[5][3])
num("ErectPerBlock", erect, 1)
num("ConsOtherMoves", NMOVE - erect, 1)
num("PropErectPerBlock", NMOVE * erect / wsum, 2)
num("PropErectFactor", NMOVE / wsum, 1)                  # the erection flow scaled 2.9-fold
for y in YARDS:
    assert close(B61[(y, "main")]["lam16"], SYS[y]["main"]["T1_system"], 0.05)         # the paper's proportional bound
    assert close(B61[(y, "intra")]["lam16"], lam48(y), 1e-6) and close(B61[(y, "crane")]["lam16"], SYS[y]["crane"]["T1_system"], 0.05)
    assert all(close(float(B61[(y, m)]["lam24"]) / float(B61[(y, m)]["lam16"]), 1.5, 1e-9) for m in ("main", "conserved", "intra"))
    # the conserved bound is the proportional programme with the erection flow at its weight (summary.json checks)
    assert abs(js("results/main_case/summary.json")["bound_checks"][y]["rel_flow_share"]) < 1e-12
lamC = {y: float(B61[(y, "conserved")]["lam16"]) for y in YARDS}
pct("YuIntraBoundRise", lam48("yupu") / lamC["yupu"] - 1, 1)        # Appendix B.2: against the main case
pct("YaIntraBoundDrop", 1 - lam48("yantai") / lamC["yantai"], 1)
num("YuSysCons", lamC["yupu"])
num("YaSysCons", lamC["yantai"])
num("YuSysConsDay", B61[("yupu", "conserved")]["lam24"])
pct("YuConsRise", lamC["yupu"] / float(SYS["yupu"]["main"]["T1_system"]) - 1, 1)
pct("YaConsDrop", 1 - lamC["yantai"] / float(SYS["yantai"]["main"]["T1_system"]), 1)
assert "010-6" in B61[("yupu", "conserved")]["top1"] and "building014" in B61[("yantai", "conserved")]["top1"]
# demand over the bound: Okpo's own volume, its peak (16-h days) and own volume on 24-h days; Yantai's own volume
for m, tag in (("conserved", "Cons"), ("main", "Prop"), ("crane", "Crane")):
    bb = B61[("yupu", m)]
    prange(f"Yu{tag}DensOwn", [float(bb["ratio16_339"]), float(bb["ratio16_350"])], 1)
    if m != "crane":
        prange(f"Yu{tag}DensPeak", [float(bb["ratio16_678"]), float(bb["ratio16_700"])], 1)
        prange(f"Yu{tag}DensDay", [float(bb["ratio24_339"]), float(bb["ratio24_350"])], 1)
    pct(f"Ya{tag}DensOwn", B61[("yantai", m)]["ratio16_23"], 1)
bb = B61[("yupu", "intra")]                    # the intra-area ratios are those of intra_area/ (YuIntraDens, ...Day, ...Peak, YaIntraDens)
assert [MAC[k] for k in ("YuIntraDens", "YuIntraDensDay", "YuIntraDensPeak", "YaIntraDens")] == [
    f"{100*float(bb[a]):.1f}--{100*float(bb[b_]):.1f}\\%" for a, b_ in (("ratio16_339", "ratio16_350"),
                                                                        ("ratio24_339", "ratio24_350"),
                                                                        ("ratio16_678", "ratio16_700"))] + \
    [f"{100*float(B61[('yantai', 'intra')]['ratio16_23']):.1f}\\%"]
assert MAC["YaOwn"] == "23" and MAC["YaPropDensOwn"] == MAC["YaDensOwn"]
# erections per day at Okpo's own volume and the cranes of each dock (crane_by_dock.csv, 16-h days; capacity = cranes
# x 16 h / crane cycle; erections split evenly between the two docks, as the task stream draws them)
num("YuConsErectDay", B61[("yupu", "conserved")]["p6_own"], 1)
num("YuPropErectDay", B61[("yupu", "main")]["p6_own"], 1)
CD = {(r["yard"], r["cranes"]): r for r in rows("results/main_case/crane_by_dock.csv")}
d2, all3 = CD[("yupu", "1")], CD[("yupu", "3")]
assert close(d2["cap16"], 16 * 60 / cycle_min) and close(d2["dest_share"], 0.5)
two = lambda r, a, b_: f"{float(r[a]):.2f}--{float(r[b_]):.2f}"
put("YuDockTwoOwn", two(d2, "conserved_339", "conserved_350"))
put("YuDockTwoPeak", two(d2, "conserved_678", "conserved_700"))
put("YuCranesPeak", two(all3, "conserved_678", "conserved_700"))
put("YuDockTwoProp", two(d2, "current_339", "current_350"))
assert max(float(d2[f"conserved_{n}"]) for n in (339, 350)) < 1 < min(float(d2[f"conserved_{n}"]) for n in (678, 700))
# the share of rehandling moved into areas (q): system bound and binding element from 0 to 80 % (scan.csv), one switch
# per yard where 1/lambda^S, convex and piecewise linear in q, has its breakpoint (switch.csv)
SCAN = [r for r in rows("results/main_case/scan.csv")]
SW = {r["yard"]: r for r in rows("results/main_case/scan_switch.csv")}
for y in YARDS:
    s_ = [r for r in SCAN if r["yard"] == y]
    assert close(s_[0]["lam16"], lamC[y], 1e-6) and float(s_[0]["q"]) == 0
    assert close(next(r for r in s_ if abs(float(r["q"]) - 0.655) < 1e-9)["lam16"], lam48(y), 1e-6)
    pct(Y[y] + "ScanSwitch", SW[y]["q_star"])
    num(Y[y] + "ScanMax", SW[y]["lam_star"])
    num(Y[y] + "ScanEnd", s_[-1]["lam16"])
    assert float(SW[y]["lam_star"]) >= max(float(r["lam16"]) for r in s_) - 1e-6     # the largest bound of the scan
pct("ScanQMax", max(float(r["q"]) for r in SCAN))
# --- Table 7: the system bound of each scenario, its binding element and the demand ratio at the own volume; sampled
# bounds of ten task samples of MixN tasks (proportional, erection-heavy and crane-paced: bounds/<yard>/; conserved: the
# ten seeds of the main case's saturation runs; intra-area: intra_area/sampled.csv); route-free bound and margin, and the
# through share, where the route-free programme was solved (bounds/<yard>/route_free.csv)
SAT61 = {(r["yard"], r["map"], r["model"]): r for r in rows("results/main_case/saturation.csv")}
assert js("results/" + js("results/main_case/summary.json")["sat_dirs"]["yupu"] + "/config.json")["n_mix"] == \
    int(MAC["MixN"].replace("{,}", ""))
TYPE = {"road": "road", "entrance": "door", "stop": "bay"}
# route-free bound (Lagrangian certificate), route margin (back-substituted feasible value to certificate) and through
# share of the main case and of the proportional mapping, whole roads, on the current map (rev 1833; route_free_main_case.csv);
# the proportional certificate and through share are those of bounds/<yard>/, its feasible value differs (that run was on
# rev 1825)
RF69 = {(r["yard"], r["map"], r["section"]): r for r in rows("results/bounds/route_free_main_case.csv")}
for k, r in RF69.items():
    b_ = next(x for x in rows("results/bounds/route_free_mappings.csv") if (x["yard"], x["map"], x["section"]) == k)
    assert close(r["certificate"], b_["T1pp"], 1e-9) and close(r["feasible"], b_["feasible"], 1e-9)
    assert close(r["margin_lo"], float(r["feasible"]) / float(r["system"]) - 1, 1e-12)
    assert close(r["margin_hi"], float(r["certificate"]) / float(r["system"]) - 1, 1e-12)
for y in YARDS:
    rc, rp = RF69[(y, "conserved", "whole")], RF69[(y, "proportional", "whole")]
    assert close(rc["system"], lamC[y], 1e-6) and close(rc["certificate"], RF69[(y, "conserved", "stops")]["certificate"], 1e-6)
    assert close(rp["system"], SYS[y]["main"]["T1_system"], 1e-6) and close(rp["certificate"], RF[y]["main"]["T1pp"], 1e-6)
    assert abs(float(rp["through"]) - float(RO[y]["main"]["pass_share"])) < 1e-3
RFMAP = {"conserved": "conserved", "main": "proportional"}
rcY = RF69[("yupu", "conserved", "whole")]
num("YuRFCons", rcY["certificate"])
pct("YuRFConsShare", float(rcY["certificate"]) / float(rcY["system"]))
pct("YuConsThrough", rcY["through"])
assert RF69[("yantai", "conserved", "whole")]["binding_type"] == "entrance" and \
    abs(float(RF69[("yantai", "conserved", "whole")]["margin_hi"])) < 1e-9          # Yantai: a door, no route margin
lines = []
for y in YARDS:
    rows6 = [("Main case: erection conserved", "conserved"), ("Every move scaled", "main"),
             (MAC["IntraMoves"] + " of " + MAC["MovesPerBlock"] + " moves within areas", "intra"),
             ("Erection-heavy mix", "erection"), ("Crane-paced erection", "crane")]
    for i, (label, m) in enumerate(rows6):
        sc = {"conserved": None, "intra": None}.get(m, m)
        if sc:                                                # scenarios of bounds/<yard>/: system, sampled, route-free
            s, f = SYS[y][sc], RF[y][sc]
            lam, top, typ = float(s["T1_system"]), s["top1"], {"road": "road", "entrance": "door"}[RO[y][sc]["type"]]
            smp = f"{fmt(s['ref_min'])}--{fmt(s['ref_max'])}"
            lo, hi = max(0.0, float(f["margin_lo_rel"])), max(0.0, float(f["margin_hi_rel"]))
            rf = f"{fmt(f['T1pp'])} & " + ("0" if hi < 1e-9 else f"{100*lo:.1f}--{100*hi:.1f}")
            thr = f"{100*float(RO[y][sc]['pass_share']):.1f}"
        else:
            b = B61[(y, m)]
            lam, top, typ = float(b["lam16"]), b["top1"], TYPE[b["type1"]]
            if m == "conserved":
                s_ = SAT61[(y, "conserved", "reserve")]
                smp = f"{fmt(s_['sampled_lo'])}--{fmt(s_['sampled_hi'])}"
            else:
                s_ = SMP[(y, 10, "network_tight")]
                smp = f"{fmt(s_['lo'])}--{fmt(s_['hi'])}"
            rf, thr = "-- & --", "--"
        if m in RFMAP:                                        # route_free_main_case.csv: the current map
            f = RF69[(y, RFMAP[m], "whole")]
            lo, hi = max(0.0, float(f["margin_lo"])), max(0.0, float(f["margin_hi"]))
            rf = f"{fmt(f['certificate'])} & " + ("0" if hi < 1e-9 else f"{100*lo:.1f}--{100*hi:.1f}")
            thr = f"{100*float(f['through']):.1f}"
        bm = B61.get((y, m))                                # no own-volume ratio under the erection-heavy mix
        if bm is not None:
            assert close(bm["lam16"], lam, 0.05)
            ratio = (f"{100*float(bm['ratio16_339']):.1f}--{100*float(bm['ratio16_350']):.1f}" if y == "yupu"
                     else f"{100*float(bm['ratio16_23']):.1f}")
        else:
            ratio = "--"
        lines.append(f"{yname[y] if i == 0 else ''} & {label} & {fmt(lam)} & {res_name(top)} ({typ}) & {ratio} & "
                     f"{smp} & {rf} & {thr} \\\\")
write("tab_bounds.tex", lines)
# the binding element moves with the scenario at Okpo (Dock 1 road, through corridor, assembly-area bays); at Yantai the
# paint-shop doors bind under both inter-area mappings and crane pacing, the panel-line hall under the intra-area one
BIND6 = {(y, m): res_name(B61[(y, m)]["top1"]) for y in YARDS for m in ("conserved", "main", "intra", "crane")}
assert [BIND6[("yupu", m)] for m in ("conserved", "main", "intra", "crane")] == \
    ["through corridor", "Dock 1 road", "assembly-area bays", "through corridor"]
assert [BIND6[("yantai", m)] for m in ("conserved", "main", "intra", "crane")] == \
    ["paint-shop doors", "paint-shop doors", "panel-line hall doors", "paint-shop doors"]
# --- the main case at saturation (saturation.csv): plateaus, peaks, onset of saturation, crossing of free flow;
# no simulated point over its own bound (check.csv)
for y in YARDS:
    gsat = lambda mo, c: float(SAT61[(y, "conserved", mo)][c])
    num(Y[y] + "ConsResPlateau", gsat("reserve", "plateau"))
    pct(Y[y] + "ConsResRatio", gsat("reserve", "ratio"))
    num(Y[y] + "ConsResAtMaxK", gsat("reserve", "at_150"))
    num(Y[y] + "ConsSafeMax", gsat("segment_safe", "max"))
    num(Y[y] + "ConsSafePlateau", gsat("segment_safe", "plateau"))
    num(Y[y] + "ConsRefPlateau", gsat("segment", "plateau"))
    pct(Y[y] + "ConsRefRatio", gsat("segment", "ratio"))
    num(Y[y] + "ConsFreeCross", gsat("free", "free_cross"))
    num(Y[y] + "ConsSampledMean", gsat("reserve", "sampled"))
    sr = SAT61[(y, "conserved", "reserve")]
    for th, tn in (("20", ""), ("10", "Ten"), ("30", "Thirty")):
        put(Y[y] + "ConsKStar" + tn, sr[f"K_star_{th}"] if sr[f"K_star_{th}_lo"] == sr[f"K_star_{th}_hi"] == sr[f"K_star_{th}"]
            else f"{sr[f'K_star_{th}']} ({sr[f'K_star_{th}_lo']}--{sr[f'K_star_{th}_hi']})")
KSTAR_C = (int(SAT61[("yupu", "conserved", "reserve")]["K_star_20_lo"]), int(SAT61[("yupu", "conserved", "reserve")]["K_star_20_hi"]))
# capacity interval of the main case (interval.csv): whole and sectioned dock roads, 10- and 30-min handling;
# sectioning leaves the bounds unchanged, since neither binding element is a road with stops
IV61 = {(r["yard"], r["section"], int(r["load_min"])): r for r in rows("results/main_case/interval.csv")}
for y in YARDS:
    assert close(IV61[(y, "whole", 10)]["plat_reserve"], SAT61[(y, "conserved", "reserve")]["plateau"], 1e-9)
    assert all(close(IV61[(y, g, lm)]["lam_S"], IV61[(y, "whole", lm)]["lam_S"], 1e-9) for g in ("stops",) for lm in (10, 30))
    for g in TG:
        i10, i30 = IV61[(y, g, 10)], IV61[(y, g, 30)]
        num(Y[y] + "ConsResLow" + TG[g], i10["plat_reserve"])
        pct(Y[y] + "ConsResLow" + TG[g] + "Ratio", i10["ratio_reserve"])
        pct(Y[y] + "ConsLoadThirty" + TG[g] + "Ratio", i30["ratio_reserve"])
    num(Y[y] + "ConsLoadThirtyBound", IV61[(y, "whole", 30)]["lam_S"])
assert res_name(IV61[("yupu", "whole", 30)]["binding"]) == "paint-shop doors"
CK61 = rows("results/main_case/check.csv") + rows("results/main_case/check_rules.csv")
assert all(r["over"] == "0" for r in CK61)
num("ConsCheckPoints", sum(int(r["checked"]) for r in CK61))
num("ConsCheckMax", max(float(r["max_ratio"]) for r in CK61), 2)
num("ConsSatCheckMax", max(float(r["max_ratio"]) for r in CK61 if r["batch"] == "main_saturation"), 2)
# --- the main case's fleets (fleet.csv: K_req and K_stable of free flow and full-path reservation; fleet_rules.csv
# adds safe release and the reference rule, fleet_30min.csv the 30-min rows); marks against the main case's K* (20 %)
F61 = {(int(r["day_h"]), r["model"], int(r["demand"])): r for r in rows("results/main_case/fleet.csv") if r["q"] == "0.0"}
T761 = {(int(r["day_h"]), int(r["demand"])): r for r in rows("results/main_case/fleet_rules.csv")}
L3061 = {(int(r["day_h"]), int(r["demand"])): r for r in rows("results/main_case/fleet_30min.csv")}
for (h, n), r in T761.items():
    for mo in ("free", "reserve"):
        assert r["Kreq_" + mo] == F61[(h, mo, n)]["K_req"] and r["Kstable_" + mo] == F61[(h, mo, n)]["K_stable"]
    k_ = lambda c: int(c.replace("ᵃ", "")) if c.replace("ᵃ", "").isdigit() else 10 ** 6
    for c, mk in (("Kreq_free", "mark_free"), ("Kreq_reserve", "mark_reserve"), ("Kreq_segment_safe", "mark_segment_safe"),
                  ("Kreq_segment", "mark_segment"), ("Kstable_free", "mark_stable_free"), ("Kstable_reserve", "mark_stable_reserve")):
        k = k_(r[c])
        assert r[mk] == ("bold" if k >= KSTAR_C[1] else "underline" if k >= KSTAR_C[0] else ""), (h, n, c)


def cons_cell(h, n, col):
    """A cell of the main case as Table 9 prints it, marked against its K* (fleet_rules.csv: 'a' and '> 200' as there)."""
    v = T761[(h, n)][col]
    return marked(v.replace("ᵃ", "").replace(" ", ""), "[a]" if v.endswith("ᵃ") else "[]", KSTAR_C)


cK = lambda h, mo, n, c="K_stable": int(F61[(h, mo, n)][c])
for th in ("10", "30"):              # with 10 % or 30 % in place of 20 %: own volume below K*, 16-h peak at or above it
    sr = SAT61[("yupu", "conserved", "reserve")]
    assert max(cK(h, m, n, c) for h in (16, 24) for m in ("free", "reserve") for n in (339, 350) for c in ("K_req", "K_stable")) \
        < int(sr[f"K_star_{th}_lo"]) and min(cK(16, "reserve", n, "K_req") for n in (678, 700)) >= int(sr[f"K_star_{th}_hi"]), th
irange("YuConsStableOwnFree", [cK(16, "free", n) for n in (339, 350)])
irange("YuConsStableOwnRes", [cK(16, "reserve", n) for n in (339, 350)])
irange("YuConsOwnFree", [cK(16, "free", n, "K_req") for n in (339, 350)])
irange("YuConsOwnRes", [cK(16, "reserve", n, "K_req") for n in (339, 350)])
for tag, c in (("Stable", "K_stable"), ("", "K_req")):
    e_ = [cK(16, "reserve", n, c) / cK(16, "free", n, c) - 1 for n in (339, 350)]
    pct(f"YuCons{tag}OwnExtraLo", min(e_))
    pct(f"YuCons{tag}OwnExtraHi", max(e_))
irange("YuConsStablePeakRes", [cK(16, "reserve", n) for n in (678, 700)])
irange("YuConsPeakRes", [cK(16, "reserve", n, "K_req") for n in (678, 700)])
irange("YuConsStableDayRes", [cK(24, "reserve", n) for n in (339, 350)])
irange("YuConsStableDayFree", [cK(24, "free", n) for n in (339, 350)])
put("YuConsSafeOwn", "--".join(T761[(16, n)]["Kreq_segment_safe"].replace("ᵃ", "") for n in (339, 350)))
put("YuConsRefOwn", "--".join(T761[(16, n)]["Kreq_segment"] for n in (339, 350)))
assert all(T761[(16, n)]["Kreq_segment_safe"].endswith("ᵃ") for n in (339, 350))
# within one vehicle of the proportional mapping at the own volume (K_req and K_stable, both days); K_req at 339
# moves/day equal to that under crane-paced erection
num("YuConsOwnMaxDiff", max(abs(cK(h, mo, n, c) - int(F61[(h, mo, n)][c + "_cur"])) for h in (16, 24)
                             for mo in ("free", "reserve") for n in (339, 350) for c in ("K_req", "K_stable")))
assert int(MAC["YuConsOwnMaxDiff"]) == 1
assert all(cK(h, "reserve", 339, "K_req") == int(E_DEM["yupu"][(f"crane_d{h}", 339, "reserve")]["K_req"]) for h in (16, 24))
# K_stable against K_req in the main case (the 16 cells of Table 9): within one vehicle except full-path reservation at
# the 16-h peak
dC = {k: int(r["K_stable"]) - int(r["K_req"]) for k, r in F61.items()}
num("ConsStableCells", len(dC))
num("ConsStableWithinOne", sum(abs(d) <= 1 for d in dC.values()))
assert sorted(k for k, d in dC.items() if abs(d) > 1) == [(16, "reserve", 678), (16, "reserve", 700)]
irange("YuConsLThirtyFree", [int(L3061[(16, n)]["Kstable_free"]) for n in (339, 350)])
irange("YuConsLThirtyRes", [int(L3061[(16, n)]["Kstable_reserve"]) for n in (339, 350)])


# --- Tables 9 and S2: both mappings; K_req of the four rules and K_stable of free flow and full-path reservation, per day
NC = 13


def stab_cell(mp, h, model, n, ks):
    r = STB.get((MP[mp], h, model, n))
    return "--" if r is None else marked(r["K_stable"], ks=ks)


def head(text):
    return rf"\multicolumn{{{NC}}}{{l}}{{{text}}}\\"


def kst(ks):
    return f"{ks[0]}--{ks[1]}" if ks[0] != ks[1] else f"{ks[0]}"


# the proportional mapping (every move scaled with the flows): the earlier main case, now in the supplementary material
T7P = [head(r"\emph{Every move scaled} (onset of saturation $K^{*}$ = " + kst(KSTAR["yupu"]) + ")")]
for n in (339, 350, 678, 700):
    c = []
    for h in (16, 24):
        c += [cell("yupu", float(h), m, n) for m in ("free", "reserve", "segment_safe", "segment")]
        c += [stab_cell("inter", h, m, n, KSTAR["yupu"]) for m in STAB]
    T7P.append(f"{fmt(n)} & " + " & ".join(c) + r" \\")
T7P.append(head(r"\quad\emph{30-min loading and unloading}"))
for n in (339, 350):
    c = []
    for h in ("16", "24"):
        c += [l30_cell("yupu", h, m, n) for m in ("free", "reserve", "segment_safe", "segment")] + ["--", "--"]
    T7P.append(f"{fmt(n)} & " + " & ".join(c) + r" \\")
write("tab_fleethead_prop.tex", T7P)
# Table 9: the main case (erection conserved), with crane-paced erection and 30-min handling below it, then the
# intra-area mapping
T7 = [head(r"\emph{Main case: erection conserved} (onset of saturation $K^{*}$ = " + kst(KSTAR_C) + ")")]
for n in (339, 350, 678, 700):
    c = []
    for h in (16, 24):
        c += [cons_cell(h, n, "Kreq_" + m) for m in ("free", "reserve", "segment_safe", "segment")]
        c += [cons_cell(h, n, "Kstable_" + m) for m in STAB]
    T7.append(f"{fmt(n)} & " + " & ".join(c) + r" \\")
T7.append(head(r"\quad\emph{crane-paced erection}"))
c = []
for h in (16, 24):
    c += [crane_cell("yupu", float(h), m, 339) if m in ("free", "reserve") else "--"
          for m in ("free", "reserve", "segment_safe", "segment")] + ["--", "--"]
T7.append("339 & " + " & ".join(c) + r" \\")
T7.append(head(r"\quad\emph{30-min loading and unloading}"))
for n in (339, 350):
    c = []
    for h in (16, 24):
        r_ = L3061[(h, n)]
        c += [r_["Kreq_free"], r_["Kreq_reserve"], "--", "--", r_["Kstable_free"], r_["Kstable_reserve"]]
    T7.append(f"{fmt(n)} & " + " & ".join(c) + r" \\")
T7.append(head(r"\emph{" + MAC["IntraMoves"] + r" of " + MAC["MovesPerBlock"] + r" moves per block within an area} "
               r"(onset of saturation $K^{*}$ = " + kst(INTRA_KS) + ")"))
for n in (339, 350, 678, 700):
    c = []
    for h in (16, 24):
        c += [str(k48(n, h, "free")), str(k48(n, h, "reserve")), "--", str(k48(n, h, "segment"))]
        c += [stab_cell("intra", h, m, n, None) for m in STAB]
    T7.append(f"{fmt(n)} & " + " & ".join(c) + r" \\")
T7.append(head(r"\quad\emph{30-min loading and unloading}"))
for n in (339, 350):
    c = []
    for h in (16, 24):
        c += [FL48[(n, h, "intra")]["K_free"], FL48[(n, h, "intra")]["K_res"], "--", "--", "--", "--"]
    T7.append(f"{fmt(n)} & " + " & ".join(c) + r" \\")
assert all(len(l.split(" & ")) == NC for l in T7 if not l.startswith(r"\multicolumn"))
assert max(k48(n, h, m) for n in (339, 350, 678, 700) for h in (16, 24) for m in ("free", "reserve", "segment")) \
    < INTRA_KS[0] and max(int(STB[(MP["intra"], h, m, n)]["K_stable"]) for h in (16, 24) for m in STAB
                          for n in (339, 350, 678, 700)) < INTRA_KS[0]           # the intra-area rows stay unmarked
write("tab_fleethead.tex", T7)


def signed(x, nd=1):
    """A relative change in per cent with its sign in maths ('$-$54.0\\%'), 0.0 without a sign."""
    s = fmt(100 * float(x), nd)
    return s.lstrip("-") + r"\%" if float(s) == 0 else ("$-$" + s[1:] if s.startswith("-") else "$+$" + s) + r"\%"


# --- random handling and driving times (numbers.json, plateau.csv, curves.csv, deadlocks.csv, demand.csv, long8.csv)
N50 = js("results/stochastic/summary.json")
C50 = js("results/" + N50["runs"]["yupu"]["stoch_curves"] + "/config.json")["scenarios"]
C50V = js("results/" + N50["runs"]["yupu"]["stoch_variants"] + "/config.json")["scenarios"]
assert C50["cv3"]["stoch"] == {"mean": 1.5, "cv": 0.3, "speed": [0.9, 1.0], "book_q": None}
assert C50["cv6"]["stoch"] == {"mean": 1.5, "cv": 0.6, "speed": [0.9, 1.0], "book_q": None}
assert C50["cv6q"]["stoch"]["book_q"] == 0.9 and C50["cv6pm"]["stoch"]["speed"] == [0.9, 1.1]
assert C50["m15"]["load_s"] == C50["m15"]["unload_s"] == 1.5 * P["load_s"] == C50V["r15"]["load_s"]   # deterministic
assert C50V["v2"]["stoch"] == {"mean": 1.5, "cv": 0.0, "speed": [1.0, 1.0], "book_q": None}   # 15 min, booked at 10
assert C50V["v1"]["stoch"] == {"mean": 1.0, "cv": 0.0, "speed": [0.999, 1.0], "book_q": None}  # 0.1 % slower only
num("StochMeanMin", 1.5 * P["load_s"] / 60)
num("StochSpeedLo", C50["cv3"]["stoch"]["speed"][0], 1)
num("StochCvLo", C50["cv3"]["stoch"]["cv"], 1)
num("StochCvHi", C50["cv6"]["stoch"]["cv"], 1)
num("StochBookQ", C50["cv6q"]["stoch"]["book_q"], 1)
pct("StochPmHi", C50["cv6pm"]["stoch"]["speed"][1] - 1)
pct("StochVOneSlow", 1 - C50V["v1"]["stoch"]["speed"][0], 1)
pair("StochQNinety", [N50["handling_quantiles"][c]["q90_min"] for c in ("cv3", "cv6")], 1)
pair("StochQNinetyNine", [N50["handling_quantiles"][c]["q99_min"] for c in ("cv3", "cv6")], 1)
CK50 = N50["check"]
assert all(c["over"] == 0 for c in CK50.values()) and CK50["all"]["points"] == sum(CK50[k]["points"] for k in ("stoch", "pm", "m15"))
assert sum(int(r["points"]) for r in rows("results/stochastic/check.csv")) == CK50["all"]["points"]
num("StochCheckPoints", CK50["stoch"]["points"])
num("StochCheckMax", CK50["stoch"]["max_ratio"], 3)
num("StochPmPoints", CK50["pm"]["points"])
num("StochAllPoints", CK50["all"]["points"])
PL50 = {(r["yard"], r["rule"], r["level"]): r for r in rows("results/stochastic/plateau.csv")}
CU50 = {(r["yard"], r["level"], r["rule"], int(r["K"])): r for r in rows("results/stochastic/curves.csv")}
for y in YARDS:     # the nominal deterministic curves are the paper's runs (its value at K = 150)
    assert fmt(CU50[(y, "h0", "reserve", 150)]["thr"]) == MAC[Y[y] + "ResAtMaxK"]
    assert PL50[(y, "reserve", "cv6q")]["plateau"] < PL50[(y, "reserve", "cv6")]["plateau"]  # 0.9-quantile booking: lower
var_ = lambda rule, lvs: [float(PL50[(y, rule, lv)]["vs_ref_rel"]) for y in YARDS for lv in lvs]
assert all(PL50[(y, rule, lv)]["ref"] == "m15" for y in YARDS for rule in ("free", "safe") for lv in ("cv3", "cv6", "cv6pm"))
prange("StochFreeMax", [max(abs(x) for x in var_("free", ("cv3", "cv6")))], 1)
prange("StochSafeMax", [max(abs(x) for x in var_("safe", ("cv3", "cv6", "cv6pm")))], 1)
DL50 = rows("results/stochastic/deadlocks.csv")
safe_dl = [r for r in DL50 if r["rule"] == "safe" and r["level"] in ("cv3", "cv6", "cv6pm")]
assert all(r["runs_with_dl"] == "0" for r in DL50 if r["rule"] == "safe" or r["level"] in ("m15", "r15"))
num("StochSafeRuns", sum(int(r["points"]) for r in safe_dl))
for y in YARDS:      # rebooking: against deterministic times with the same handling (m15 for random, r15 for V2)
    assert PL50[(y, "reserve", "v2")]["ref"] == "r15" and all(PL50[(y, "reserve", lv)]["ref"] == "m15" for lv in ("cv3", "cv6"))
    prange(Y[y] + "StochResDrop", [-float(PL50[(y, "reserve", lv)]["vs_ref_rel"]) for lv in ("cv3", "cv6", "v2")])
    prange(Y[y] + "StochVOne", [-float(PL50[(y, "reserve", "v1")]["vs_ref_rel"])])
KDL = sorted({K for (y, lv, rule, K) in CU50 if lv == "v2"})            # the fleets of the V2 runs, 20-150
for (y, lv, rule, K), r in CU50.items():
    if rule == "reserve" and lv in ("cv3", "cv6", "v2") and K >= KDL[0]:
        assert r["runs_with_dl"] == r["n"]                                     # every run deadlocks
put("StochDlK", f"{KDL[0]}--{KDL[-1]}")
dlm = lambda y, lv: statistics.mean(float(r["dl_per_day"]) for (y_, lv_, ru, K), r in CU50.items()
                                    if (y_, lv_, ru) == (y, lv, "reserve") and K >= 20)
dl = [dlm(y, lv) for y in YARDS for lv in ("cv3", "cv6")]                 # random handling and driving times
put("StochDl", f"{fmt(min(dl))}--{fmt(max(dl))}")
dlv = [dlm(y, "v2") for y in YARDS]                                        # deterministic, longer than booked
put("StochDlLonger", f"{fmt(min(dlv))}--{fmt(max(dlv))}" if fmt(min(dlv)) != fmt(max(dlv)) else fmt(min(dlv)))
L8 = rows("results/stochastic/eight_day.csv")                  # days 5-8 of eight-day runs, against m15 at the same fleet
assert all(r["same_as_4d"] == r["n"] for r in L8)
l8 = [float(r["thr_d5_8"]) / float(CU50[(r["yard"], "m15", "reserve", int(r["K"]))]["thr"]) - 1 for r in L8]
assert max(l8) < -0.25
DM50 = {(r["level"], r["day"], r["rule"], int(r["demand"])): r for r in rows("results/stochastic/fleet.csv")}
for d_, h in (("d16", 16.0), ("d24", 24.0)):
    for n in (339, 350):
        assert DM50[("h0", d_, "free", n)]["K_req"] == kreq4("yupu", h, "whole", "free", n)["K_req"]
        assert DM50[("m15", d_, "free", n)]["K_req"] == DM50[("cv3", d_, "free", n)]["K_req"] == DM50[("cv6", d_, "free", n)]["K_req"]
irange("StochFreeFleetSixteen", [int(DM50[("cv6", "d16", "free", n)]["K_req"]) for n in (339, 350)])
irange("StochFreeFleetTwentyFour", [int(DM50[("cv6", "d24", "free", n)]["K_req"]) for n in (339, 350)])
irange("StochRebookTwentyFour", [int(DM50[(lv, "d24", "reserve", n)]["K_req"]) for lv in ("cv3", "cv6") for n in (339, 350)])
irange("StochMeanResTwentyFour", [int(DM50[("m15", "d24", "reserve", n)]["K_req"]) for n in (339, 350)])
reb16, KREB = [], 110
for lv, n in (("cv3", 339), ("cv6", 339), ("cv6", 350)):    # 16-h days: around the target with 110-200 vehicles
    pr_ = {int(k): v for k, v in json.loads(DM50[(lv, "d16", "reserve", n)]["probed"]).items()}
    assert max(pr_) == 200
    reb16 += [v / n for k, v in pr_.items() if k >= KREB]
num("StochRebookKLo", KREB)
assert DM50[("cv3", "d16", "reserve", 350)]["K_req"] == ">150" and not any(
    DM50[(lv, "d16", "reserve", n)]["K_req"].isdigit() and DM50[(lv, "d16", "reserve", n)]["after_fail"] == "[]"
    for lv in ("cv3", "cv6") for n in (339, 350))                    # no 16-h cell reaches the target for good
prange("StochRebookShare", reb16, 1)
# Table C (random times): plateaus over K = 100, 120 and 150, deadlocks a day over K = 20-150 (free flow does not check
# capacity, so it has none to count); the rebooking rows under a heading row
DET, MEAN = "deterministic, " + MAC["LoadMin"] + " min", "deterministic, " + MAC["StochMeanMin"] + " min"
RND = [("random, CV 0.3", "cv3"), ("random, CV 0.6", "cv6")]
STOCH_ROWS = [("Free flow", "free", [(DET, "h0"), (MEAN, "m15")] + RND),
           ("Safe release", "safe", [(DET, "h0"), (MEAN, "m15")] + RND),
           ("Full-path reservation", "reserve", [(DET, "h0"), (MEAN, "m15"), (MEAN + ", nominal speeds", "r15")]),
           (None, "reserve", RND + [(MEAN + ", booked at " + MAC["LoadMin"] + " min", "v2")])]
lines = []
for rname, rule, levels in STOCH_ROWS:
    if rname is None:
        lines.append(r"\multicolumn{7}{@{}l}{\quad\emph{Full-path reservation booked at nominal times, rebooked when late}}\\")
    for i, (tname, lv) in enumerate(levels):
        c = [rname if i == 0 and rname else "", tname]
        for y in YARDS:
            r = PL50[(y, rule, lv)]
            c += [fmt(r["plateau"]), "--" if lv == "h0" else signed(r["vs_h0_rel"] if lv in ("m15", "r15") else r["vs_ref_rel"])]
        dl_ = [statistics.mean(float(q["dl_per_day"]) for (y_, lv_, ru, K), q in CU50.items()
                               if (y_, lv_, ru) == (y, lv, rule) and K >= 20) for y in YARDS]
        c.append("--" if rule == "free" else " / ".join(fmt(v) for v in dl_) if max(dl_) > 0 else "0")
        lines.append(" & ".join(c) + r" \\")
# --- full-path reservation executed in booked order (no event of a resource before one of smaller booked key):
# booked at nominal times, on projected occupancy (overdue bookings kept in the tables at their projected release),
# and on projected occupancy at mean durations; plateaus over K = 100, 120 and 150 (plateau.csv), the change against
# the deterministic row with the same handling as above (15 min with mean speeds for random times, nominal speeds for
# the row booked at 10 min); no run deadlocked or exceeded its bound (check.csv). With deterministic 15-min handling,
# booking at mean times books the actual times: that row is full-path reservation at 15 min and is not repeated.
PL65 = {(r["yard"], r["rule"], r["level"]): r for r in rows("results/stochastic/ordered_plateau.csv")}
for (y, ru, lv), r in PL65.items():
    assert lv in ("r15", "cv3m", "cv6m") or close(r["rebooked"], PL50[(y, "reserve", lv)]["plateau"], 1e-6)
assert all(close(PL65[(y, "reserve_op", "r15")]["plateau"], PL50[(y, "reserve", "r15")]["plateau"], 0.05) for y in YARDS)
CK65 = rows("results/stochastic/ordered_check.csv")
assert all(r["over"] == "0" and r["deadlocks"] == "0" for r in CK65)
OPE = [("booked at nominal times", "reserve_op", ("cv3", "cv6", "v2"), "a"),
       ("booked on projected occupancy", "reserve_op2", ("cv3", "cv6", "v2"), "b"),
       ("booked on projected occupancy at mean times", "reserve_op3", ("cv3", "cv6"), "b")]
LVN = {"cv3": RND[0][0], "cv6": RND[1][0], "v2": MEAN + ", booked at " + MAC["LoadMin"] + " min"}
for title, ru, lvs, note in OPE:
    lines.append(r"\multicolumn{7}{@{}l}{\quad\emph{Full-path reservation executed in booked order, " + title
                 + r"}$^{" + note + r"}$}\\")
    for lv in lvs:
        c = ["", LVN[lv]]
        for y in YARDS:
            p, ref = float(PL65[(y, ru, lv)]["plateau"]), float(PL50[(y, "reserve", "r15" if lv == "v2" else "m15")]["plateau"])
            c += [fmt(p), signed(p / ref - 1)]
        lines.append(" & ".join(c + ["0"]) + r" \\")
write("tab_stoch.tex", lines)
num("OpeRuns", sum(int(r["points"]) for r in CK65 if r["rule"].startswith("reserve_op")))
num("OpeCheckMax", max(float(r["max_ratio"]) for r in CK65), 2)
# throughput as a share of the deterministic plateau that the Change column of Table S6 uses: the CV levels against
# deterministic mean durations (m15), the slower speeds (v2) against deterministic 15-min handling at nominal speeds
# (r15); ordered_plateau.csv over plateau.csv, paired seeds: projected occupancy, and at mean times; booked at nominal
# times the throughput fell from day to day (eight-day runs, long8.csv)
REF65 = lambda y, lv: float(PL50[(y, "reserve", "r15" if lv == "v2" else "m15")]["plateau"])
SH65 = lambda ru, y, lv: float(PL65[(y, ru, lv)]["plateau"]) / REF65(y, lv)
prange("OpeProjShare", [SH65("reserve_op2", y, lv) for y in YARDS for lv in ("cv3", "cv6", "v2")])
prange("OpeMeanShare", [SH65("reserve_op3", y, lv) for y in YARDS for lv in ("cv3", "cv6")])
assert all(f"{signed(SH65(ru, y, lv) - 1)}" in " ".join(lines) for _, ru, lvs, _ in OPE for y in YARDS for lv in lvs)
num("OpeRefMeanYu", REF65("yupu", "cv3"))                 # the reference rows, Table S6 (m15)
num("OpeRefMeanYa", REF65("yantai", "cv3"))
L865 = rows("results/stochastic/ordered_eight_day.csv")
nom8 = [-float(r["rel_d5_8"]) for r in L865 if r["rule"] == "reserve_op"]
proj8 = [r for r in L865 if r["rule"] in ("reserve_op2", "reserve_op3")]
assert min(nom8) > 0.25 and all(abs(float(r["drift_d2_8"])) <= float(r["drift_ci"]) for r in proj8)
prange("OpeDecline", nom8)
pct("OpeEightBelow", -min(float(r["rel_d5_8"]) for r in proj8))
pct("OpeEightAbove", max(float(r["rel_d5_8"]) for r in proj8))
assert sorted({(r["rule"], r["level"]) for r in proj8}) == [("reserve_op2", "cv3"), ("reserve_op2", "cv6"), ("reserve_op2", "v2"),
                                                            ("reserve_op3", "cv3"), ("reserve_op3", "cv6")]
assert {int(r["K"]) for r in L865} == {100, 150} and {r["n"] for r in L865} == {MAC["Seeds"]}
# required fleets at Okpo's own volume, 24-h days, CV 0.3 (demand.csv): at mean times, on projected occupancy, at
# nominal times; deterministic 15-min handling (r15)
DM65 = {(r["level"], r["day"], r["rule"], int(r["demand"])): r["K_req"] for r in rows("results/stochastic/ordered_fleet.csv")}
for ru, tag in (("reserve_op3", "Mean"), ("reserve_op2", "Proj"), ("reserve_op", "Nom")):
    irange("Ope" + tag + "Fleet", [int(DM65[("cv3", "d24", ru, n)]) for n in (339, 350)])
irange("OpeDetFleet", [int(DM65[("r15", "d24", "reserve", n)]) for n in (339, 350)])
# safe release under random times at Okpo: its plateau as a share of the system bound of those runs (every move scaled
# with the flows, the mapping of the random-time runs)
prange("YuStochSafeShare", [float(PL50[("yupu", "safe", lv)]["plateau"]) / float(SYS["yupu"]["main"]["T1_system"])
                            for lv in ("cv3", "cv6")])
irange("YuStochSafePlat", [round(float(PL50[("yupu", "safe", lv)]["plateau"])) for lv in ("cv3", "cv6")])

# --- schedules that wait only at origins on sub-networks (points.csv; CP-SAT best schedules, none proved optimal)
PT51 = rows("results/subnet/points.csv")
C51 = js("results/subnet/config.json")               # the paper's maps and day; windows from the simulation's own code
assert C51["day_s"] == P["day_s"] and C51["networks"] == {"yupu": "okpo/rev1833", "yantai": "yantai/rev396"}
assert C51["ks"] == [8] and sorted({int(r["N"]) for r in rows("results/subnet/points.csv")}) == C51["ns"]
assert len(PT51) == 36 and all(r["fpr_ok"] == r["fix_ok"] == r["opt_ok"] == "True" and r["fpr_model_bad"] == "0" for r in PT51)
assert not any(r["status"] == "OPTIMAL" for r in PT51)
INST = {"yupu_dock1": "Okpo, " + res_name("road161"), "yantai_paint": "Yantai, " + res_name("entrance001 node"),
        "yupu_corridor": "Okpo, " + res_name("road010-6"), "synth_dock": "Synthetic dock road (crane)"}
by51 = {i: [r for r in PT51 if r["inst"] == i] for i in INST}
for i, rs in by51.items():
    assert len(rs) == 9 and (i == "synth_dock" or all(res_name(r["binding"]) == INST[i].split(", ")[1] for r in rs))
put("SubNs", klist(sorted({int(r["N"]) for r in PT51})))
assert {r["K"] for r in PT51} == {"8"}
num("SubK", 8)
num("SubSeeds", len({r["seed"] for r in PT51}))
num("SubPoints", len(PT51))
v51 = lambda i, c: [float(r[c]) for r in by51[i]]
share = lambda i, num_, den: [(float(r[num_[0]]) - float(r[num_[1]])) / (float(r[den[0]]) - float(r[den[1]])) for r in by51[i]]
rng2 = lambda vals: f"{min(vals):.2f}--{max(vals):.2f}"
# Okpo's Dock 1 road: the best schedule within about 1 % of the bound; the rule's share of the interval to the
# finite-window programme started from an empty system (a strict bound on these instances)
prange("SubDockMore", [float(r["opt"]) / float(r["fpr"]) - 1 for r in by51["yupu_dock1"]])
prange("SubDockRule", share("yupu_dock1", ("opt_N", "fpr_N"), ("lp0_N", "fpr_N")))
prange("SubDockBusyOpt", v51("yupu_dock1", "busy_opt"))
prange("SubDockBusyFpr", v51("yupu_dock1", "busy_fpr"))
ext_ = []
for s in sorted({r["seed"] for r in by51["yupu_dock1"]}):      # opt / bound = s + a / N, least squares over N
    pts = [(1 / int(r["N"]), float(r["opt_N"])) for r in by51["yupu_dock1"] if r["seed"] == s]
    mx, my = statistics.mean(x for x, _ in pts), statistics.mean(v for _, v in pts)
    a_ = sum((x - mx) * (v - my) for x, v in pts) / sum((x - mx) ** 2 for x, _ in pts)
    ext_.append(my - a_ * mx)
put("SubDockExtrap", f"{min(ext_):.3f}--{max(ext_):.3f}")
# without the factor 1 + beta/W the per-point check is stricter than Proposition 2: on these short windows the best
# schedules exceed lambda_W^N (the boundary term), while no simulated run did
pct("SubOverMax", max(float(r["opt_N"]) for r in PT51) - 1, 1)
assert all(float(r["opt_N"]) <= float(r["lamNb"]) / float(r["lamN"]) for r in PT51)   # within (1 + beta/W) lambda_W^N
# Yantai's paint-shop doors: the order of occupations at the junction in front of a door; three-part split of the
# interval (bounds2.json, columns sh_*), steady-state limit, gaps between consecutive users and the door occupation
for c, name in (("sh_order", "SubOrder"), ("sh_boundary", "SubBoundary"), ("sh_rule", "SubDoorRule"), ("sh_und", "SubOpen")):
    prange(name, v51("yantai_paint", c))
assert all(close(sum(float(r[c]) for c in ("sh_order", "sh_boundary", "sh_rule", "sh_und", "sh_other")), 1, 1e-6)
           and float(r["sh_other"]) < 1e-4 for r in by51["yantai_paint"])   # sh_other: rounding of the time unit
put("SubDoorSteady", rng2(v51("yantai_paint", "steady")))
irange("SubDoorGap", [round(float(r["gap_min"])) for r in by51["yantai_paint"] if float(r["gap_min"]) > 0])
irange("SubDoorCycle", [round(float(r["cyc"])) for r in by51["yantai_paint"]])
# corridor and synthetic dock road: the rule's share of the interval to the bound from below (best schedule found)
# (corridor: the best schedules of corridor.csv, which improve on those of points.csv by a simulated annealing over dispatch and booking order
# whose schedules full-path reservation's booking builds, polished by constraint programming with the dispatch fixed;
# full-path reservation and the upper limit are those of points.csv; each best schedule passed the independent checker)
PT63 = {(int(r["seed"]), int(r["N"])): r for r in rows("results/subnet/corridor.csv")}
for r in by51["yupu_corridor"]:
    q = PT63[(int(r["seed"]), int(r["N"]))]
    assert close(q["fpr_N"], r["fpr_N"], 1e-9) and close(q["ub_N"], r["optub_N"], 1e-9) and q["check"] == "True"
    assert float(q["best_N"]) >= float(r["opt_N"]) - 1e-9
    assert close(q["rule"], (float(q["best_N"]) - float(q["fpr_N"])) / (1 - float(q["fpr_N"])), 1e-9)
assert len(PT63) == len(by51["yupu_corridor"])
prange("SubCorrRule", [float(q["rule"]) for q in PT63.values()])
# in the best schedules the corridor is held most of the time; part of the rest is spent in gaps that two consecutive
# users cannot avoid given their pairing (clearing the approach roads and end junctions), split of the corridor's idle
# time (corridor split points.csv: shares of the makespan, and of its excess over the finite-window programme)
SPL = [r for r in rows("results/subnet/corridor_split.csv")]
assert {(int(r["seed"]), int(r["N"])) for r in SPL} == set(PT63) and all(close(r["best_N"], PT63[(int(r["seed"]), int(r["N"]))]["best_N"], 1e-12) for r in SPL)
assert all(close(r["best_sum"], 1, 1e-9) for r in SPL)
prange("SubCorrBusy", [float(r["best_busy"]) for r in SPL])
prange("SubCorrForced", [float(r["best_forced"]) for r in SPL])
prange("SubCorrForcedExcess", [float(r["best_ex_forced"]) for r in SPL])
prange("SubSynthRule", [(float(r["opt_N"]) - float(r["fpr_N"])) / (1 - float(r["fpr_N"])) for r in by51["synth_dock"]])
# the optimum releases the destination when unloading ends (Definition 1(c)); full-path reservation holds it while the
# empty vehicle drives off, (platform length + clearance) / empty speed: on the Dock 1 road the bound falls by that time
# per task, a share drive-off x lambda / H of the bound
drive = (P["plat_len"] + P["long_clear"]) / P["v_empty"]
num("SubDriveOff", drive, 1)
dsh = [drive * float(r["lamN"]) / P["day_s"] for r in by51["yupu_dock1"]]
prange("SubDriveOffShare", dsh, 1)
prange("SubDriveOffGap", [d / (1 - float(r["fpr_N"])) for d, r in zip(dsh, by51["yupu_dock1"])])
lines = []
for i, name in INST.items():
    split = {"yupu_dock1": r"rule $\ge$ " + MAC["SubDockRule"] + " of the interval to the finite-window bound",
             "yantai_paint": "bound: order of occupations " + MAC["SubOrder"] + ", finite-window terms "
                             + MAC["SubBoundary"] + r"; rule $\ge$ " + MAC["SubDoorRule"] + "; open " + MAC["SubOpen"],
             "yupu_corridor": r"rule $\ge$ " + MAC["SubCorrRule"] + "; bound side open",
             "synth_dock": r"rule $\ge$ " + MAC["SubSynthRule"] + "; bound side open"}[i]
    v_ = {c: v51(i, c) for c in ("fpr_N", "fix_N", "opt_N", "optub_N")}
    if i == "yupu_corridor":
        v_["opt_N"] = [float(q["best_N"]) for q in PT63.values()]
    lines.append(f"{name} & " + " & ".join(rng2(v_[c]) for c in v_) + f" & {split} \\\\")
assert lines[2] == r"Okpo, through corridor & 0.57--0.70 & 0.60--0.81 & 0.73--0.89 & 0.97--1.00 & rule $\ge$ 33--76\%; bound side open \\"
write("tab_subnet.tex", lines)

# --- ban on blocks wider than the passable width (width_ban/summary.json, clear_width.json)
S52 = js("results/width_ban/summary.json")
none52 = {(b["yard"], b["scen"]): b for b in S52["bounds"] if b["level"] == "none"}
assert all(b["same_as_earlier"] == "True" and close(b["T1_sys"], SYS[y][sc]["T1_system"], 0.05) for (y, sc), b in none52.items())
ban = [b for b in S52["bounds"] if b["level"] != "none"]
same52 = [b for b in ban if b["top1"] == none52[(b["yard"], b["scen"])]["top1"]]
assert all((b["yard"], b["scen"], b["handling"]) == ("yupu", "main", "redraw") and "010-6" in b["top1"]
           for b in ban if b not in same52)
num("BanCells", len(ban))
num("BanSameBinding", len(same52))
for h, tag in (("clamp", "Clamp"), ("redraw", "Redraw")):
    v = [float(b["rel_none"]) for b in ban if b["handling"] == h]
    num("Ban" + tag + "Lo", -100 * min(v), 1)
    put("Ban" + tag + "Hi", fmt(100 * max(v), 1) + r"\%")
FG52 = S52["fleet_gap"]
assert {k: (v["free"], v["reserve"]) for k, v in FG52.items() if k.endswith("_none")} == {
    f"n{n}_none": (int(kreq("yupu", 16.0, "whole", "free", n)["K_req"]), int(kreq("yupu", 16.0, "whole", "reserve", n)["K_req"]))
    for n in (339, 350)}
assert all(v["free"] == FG52[k.split("_")[0] + "_none"]["free"] for k, v in FG52.items())   # free flow unchanged
irange("BanResOwn", [v["reserve"] for k, v in FG52.items() if not k.endswith("_none")])
irange("BanExtra", [v["extra"] for k, v in FG52.items() if not k.endswith("_none")])
CV52 = S52["coverage"]["yupu"]
num("BanImgChecked", sum(n for n, _ in CV52.values()))
pct("BanImgShare", sum(s for _, s in CV52.values()))
num("BanImgFull", CV52["all pass"][0])
pct("BanImgFullShare", CV52["all pass"][1])
CW52 = js("results/width_ban/clear_width.json")
assert sum(1 for e in CW52["yupu"].values() if e["G"] >= 15) == CV52["all pass"][0]       # G >= 15: 15-m blocks pass
num("BanClearFull", 15 + P["lat_clear"])
dock1 = [e for e in CW52["yupu"].values() if e["name"] == "road161"][0]
num("BanDockLegs", dock1["G"] + P["lat_clear"], 1)                                   # clear width between crane legs
assert S52["check"]["over_strict"] == 0

# --- the screen across yards and road widths, the dock mouth with off-road places at both yards
S53 = js("results/screen/widths_summary.json")
S53B = js("results/screen/low_summary.json")
assert S53["reuse"]["ok"] and S53["reuse"]["low_demand_rows_differ"] == []
SEG53 = {(s["measure"], s["cal"], s["test"], s["h"]): s["cls"] for s in S53["loo_segments"]}
hit = lambda cal, test, cls: (SEG53[("gap", cal, test, "16+24")][cls]["hit"], SEG53[("gap", cal, test, "16+24")][cls]["n"])
for cal, test, tag in (("yupu", "yantai", "Ya"), ("yantai", "yupu", "Yu")):
    h_, n_ = hit(cal, test, "same fleet")
    assert h_ == n_
    put("LooSame" + tag, f"{h_} of {n_}")
    h_, n_ = hit(cal, test, "certainly different")
    put("LooAbove" + tag, f"{h_} of {n_}")
    assert all(m["gap"] == 0 for m in SEG53[("gap", cal, test, "16+24")]["certainly different"]["miss"])   # misses: conservative
LX53 = {(r["measure"], r["cal"], int(r["h"])): r for r in rows("results/screen/leave_one_out.csv")}
num("LooErrSixteen", 100 * abs(float(LX53[("gap", "yupu", 16)]["err_first"])), 1)
num("LooErrSixteenFrac", 100 * abs(float(LX53[("gap_frac", "yupu", 16)]["err_first"])), 1)
num("LooErrTwentyFour", 100 * abs(float(LX53[("gap", "yupu", 24)]["err_first"])), 1)
for (k, cal, h), r in LX53.items():  # the test starts from the crossings of the paper's screen (fleet/screen_cross.csv)
    test = "yantai" if cal == "yupu" else "yupu"
    xs = CRS if k == "gap" else CRF
    assert close(r["pred_first"], xs[(cal, str(h))]["first"]) and close(r["act_first"], xs[(test, str(h))]["first"])
assert len(set(S53["bounds"].values())) == 2 and all(v["equal"] == v["points"] for v in S53["free_same"].values())
WIDTHS = ("base", "w-1.5", "w+1.5")                  # LC, LL: low_cross.csv, read with the screen above
prange("BandFracLo", [LC[("yupu", w, "gap_frac", "all")] for w in WIDTHS], 1)
assert min(LC[(y, w, "gap_frac", "all")] for y in YARDS for w in WIDTHS) == min(LC[("yupu", w, "gap_frac", "all")] for w in WIDTHS)
pct("BandLastLo", min(LL[(y, w, "gap", "all")] for y in YARDS for w in WIDTHS), 1)
pct("BandLastHi", max(LL[(y, w, "gap", "all")] for y in YARDS for w in WIDTHS), 1)
pct("WidthIntFirst", LC[("yupu", "w-1.5", "gap", "all")], 1)
# the one level that puts it there: full-path reservation needs one vehicle more than free flow, between levels with
# equal fleets (width.csv, Okpo, roads 1.5 m narrower)
W53 = sorted((float(r["ratio"]), r) for r in rows("results/screen/widths.csv") if (r["yard"], r["variant"]) == ("yupu", "w-1.5"))
odd53 = [(x, r) for x, r in W53 if float(r["gap"]) > 0.10][0][1]
assert int(odd53["K_res"]) - int(odd53["K_free"]) == 1 and float(odd53["ratio"]) < float(MAC["ScreenFirstLo"].replace(r"\%", "")) / 100
pct("WidthOddRatio", odd53["ratio"], 1)
for y in YARDS:                       # the paper's road widths: the densified levels change no integer crossing
    assert LC[(y, "base", "gap", "all")] == LC[(y, "base", "gap", "earlier")] and LL[(y, "base", "gap", "all")] == LL[(y, "base", "gap", "earlier")]
# --- dock mouth: safe release, whole dock roads, crane-paced erection, eight days (dock4.csv); the on-road column is
# that of Fig. 7 (16 days where eight were not steady)
D4 = {(r["yard"], int(r["K"])): r for r in rows("results/dock/bays.csv")}
for (y, K), r in D4.items():
    t = RK[f"{y}|whole|segment_safe|{K}"]
    assert close(r["road_thr16"] if r["road_days"] == "16" else r["road_thr10"], t["thr_table"])


def bay_cell(r, col):
    v = r[col + "_thr10"]
    return "none" if not v else f"{float(v):.2f}" + ("" if r[col + "_steady"] == "True" else r"$^{*}$")


lines = []
for y in YARDS:
    for i, K in enumerate(sorted(K for y_, K in D4 if y_ == y)):
        r = D4[(y, K)]
        lines.append(f"{yname[y] if i == 0 else ''} & {K} & " + " & ".join(bay_cell(r, c) for c in ("road", "bay1", "bay2", "spot"))
                     + r" \\")
write("tab_dockbay.tex", lines)
for y in YARDS:
    rs = [D4[(y, K)] for K in sorted(K for y_, K in D4 if y_ == y)]
    both = [float(r["bay1_thr10"]) - float(r["road_thr10"]) for r in rs if r["bay1_thr10"] and r["road_thr10"]]
    assert all(d > 0 for d in both) and all(r["road_thr10"] for r in rs)          # later at every fleet
    put(Y[y] + "BayShift", rng2(both))
    put(Y[y] + "BayNoLoss", klist([int(r["K"]) for r in rs if not r["bay1_thr10"]]))
    r100 = D4[(y, 100)]
    put(Y[y] + "BayHundred", f"{float(r100['bay1_thr10']):.2f} to {float(r100['bay2_thr10']):.2f}")
    num(Y[y] + "BayQueueOne", r100["bay1_queue_top"])            # waits for a bay standing on the dock road, min/day
    irange(Y[y] + "BayQueueTwo", [round(float(r100[c + "_queue_top"])) for c in ("bay2", "spot")])
    sixteen = [r for r in rs if r["bay1_thr16"]]
    put(Y[y] + "BaySixteenLater", rng2([float(r["bay1_thr16"]) - float(r["road_thr16"] or r["road_thr10"]) for r in sixteen]))
spot = [abs(float(r["bay2_thr10"]) - float(r["spot_thr10"])) for r in D4.values() if r["bay2_thr10"] and r["spot_thr10"]]
put("BaySpotDiff", f"{max(spot):.2f}")
drop16 = [float(r["bay1_thr10"]) - float(r["bay1_thr16"]) for r in D4.values() if r["bay1_thr16"]]
assert min(drop16) >= 0
put("BaySixteenDrop", f"{max(drop16):.2f}")

# --- Appendix B.4: the required-fleet searches of the cells the paper reads: every integer below the first
# success of safe release when it is at most 150, five fleets above it; three above for the monotone rules
for r in DEM:
    if not r["K_req"].isdigit():
        continue
    k, pr_ = int(r["K_req"]), {int(x) for x in json.loads(r["probed"])}
    if r["model"] == "segment_safe":
        assert (k > 150 or set(range(1, k)) <= pr_) and set(range(k + 1, min(k + 5, 200) + 1)) <= pr_, r
    else:
        assert set(range(k + 1, k + 4)) <= pr_, r

# --- points of the revision runs checked against the network bound of their own completed mix
rev = [{"points": sum(int(r["checked"]) for r in rows("results/intra_area/check.csv")),
        "over": sum(int(r["over"]) for r in rows("results/intra_area/check.csv")),
        "max_ratio": max(float(r["max_ratio"]) for r in rows("results/intra_area/check.csv"))},
       {"points": sum(int(r["checked"]) for r in rows("results/intra_area/check_saturation.csv")),
        "over": sum(int(r["over"]) for r in rows("results/intra_area/check_saturation.csv")),
        "max_ratio": max(float(r["max_ratio"]) for r in rows("results/intra_area/check_saturation.csv"))},
       {"points": S52["check"]["reserve_points"], "over": S52["check"]["over_strict"], "max_ratio": S52["check"]["max_ratio"]},
       S53["t1_check"], S53B["t1_check"]]
assert all(c["over"] == 0 for c in rev)
num("RevCheckPoints", sum(c["points"] for c in rev))
num("RevCheckMax", max(c["max_ratio"] for c in rev), 3)

# --- ablation of the occupancy representation. (a) vehicles per road: a road narrower than B_max + w + c
# exclusive, any other road two vehicles, loaded or empty; (b) an empty run as wide as the block it goes to fetch;
# (c) empty vehicles take no width. Each recomputes the system bound with its own occupancy coefficients
# (bounds.csv) and the fleets at Okpo's own volume on 16-h days (summary.json: K_stable by the tolerance criterion);
# main case (conserved) in the paper, the proportional mapping in the supplementary material
REP = (("base", "This paper: width by load and block, narrower empty runs"),
       ("count", r"(a) Vehicles per road: one below $B_{\max} + w + c$, else two"),
       ("same", "(b) Empty runs as wide as the block they fetch"),
       ("efree", "(c) Empty vehicles take no width"))
ABL = {}
for mp in ("conserved", "proportional"):
    bb = {(r["yard"], r["scen"], r["rep"]): r for r in rows(ABL_BOUNDS[mp])}
    fs = js(ABL_SUMMARY[mp])["fleet"]
    ref_lam = {y: lamC[y] if mp == "conserved" else float(SYS[y]["main"]["T1_system"]) for y in YARDS}
    lines = []
    for rep, label in REP:
        c = [label]
        for y in YARDS:
            r = bb[(y, "main", rep)]
            lam = float(r["T1_sys"])
            assert rep != "base" or close(lam, ref_lam[y], 1e-6)
            w_ = re.search(r"W=([\d.]+)", r["name1"])
            c += [fmt(lam) + ("" if rep == "base" else f" ({signed(lam / ref_lam[y] - 1)})"),
                  res_name(r["name1"]) + (f" ({float(w_.group(1)):g}~m)" if w_ else "")]
        ks = [fs[f"{rep}|{mo}|n{n}"]["K_stable"] for mo in ("reserve",) for n in (339, 350)]
        assert all(fs[f"{rep}|free|n{n}"]["K_stable"] == int(MAC["YuConsStableOwnFree"] if mp == "conserved"
                                                               else MAC["YuStableOwnFree"]) for n in (339, 350) if f"{rep}|free|n{n}" in fs)
        c.append(" / ".join(str(k) for k in ks))
        ABL[(mp, rep)] = dict(lam={y: float(bb[(y, "main", rep)]["T1_sys"]) for y in YARDS}, ks=ks,
                              bind={y: res_name(bb[(y, "main", rep)]["name1"]) for y in YARDS},
                              er=float(bb[("yupu", "erection", rep)]["T1_sys"]))
        lines.append(" & ".join(c) + r" \\")
    write("tab_ablation.tex" if mp == "conserved" else "tab_ablation_prop.tex", lines)
# the base rows are the fleets of Tables 9 and S2 (main_case/fleet.csv; intra_area/stable.csv for the proportional mapping)
assert ABL[("conserved", "base")]["ks"] == [cK(16, "reserve", n) for n in (339, 350)]
assert ABL[("proportional", "base")]["ks"] == [int(STB[(MP["inter"], 16, "reserve", n)]["K_stable"]) for n in (339, 350)]
A_ = {rep: ABL[("conserved", rep)] for rep, _ in REP}
assert [A_[r]["bind"]["yupu"] for r in ("base", "count", "same", "efree")] == \
    ["through corridor", "stockyard road", "through corridor", "through corridor"]
assert all(A_[r]["bind"]["yantai"] == "paint-shop doors" and close(A_[r]["lam"]["yantai"], lamC["yantai"], 1e-6) for r in A_)
pct("AblCountDrop", 1 - A_["count"]["lam"]["yupu"] / lamC["yupu"], 1)
pct("AblCountDropProp", 1 - ABL[("proportional", "count")]["lam"]["yupu"] / float(SYS["yupu"]["main"]["T1_system"]), 1)
pct("AblEfreeRiseEr", A_["efree"]["er"] / A_["base"]["er"] - 1, 1)               # erection-heavy mix: Dock 1 road binds
pct("AblEfreeRiseProp", ABL[("proportional", "efree")]["lam"]["yupu"] / float(SYS["yupu"]["main"]["T1_system"]) - 1, 1)
erYa = {r["rep"]: float(r["T1_sys"]) for r in rows("results/ablation/bounds_main_case.csv")
        if (r["yard"], r["scen"]) == ("yantai", "erection")}
pct("AblCountDropErYa", 1 - erYa["count"] / erYa["base"], 1)
# the width of the road that binds Okpo when vehicles are counted per road, under both mappings (from its name)
wc = {re.search(r"W=([\d.]+)", r["name1"]).group(1) for mp in ("conserved", "proportional")
      for r in rows(ABL_BOUNDS[mp]) if (r["yard"], r["scen"], r["rep"]) == ("yupu", "main", "count")}
assert len(wc) == 1
num("AblCountRoadW", wc.pop())
dev = [abs(k - b) for rep in ("count", "same", "efree") for k, b in zip(A_[rep]["ks"], A_["base"]["ks"])]
irange("AblFleetDev", dev)
irange("AblCountFleet", A_["count"]["ks"])
irange("AblSameFleet", A_["same"]["ks"])
irange("AblEfreeFleet", A_["efree"]["ks"])
free_ = int(MAC["YuConsStableOwnFree"])
prange("AblExtra", [k / free_ - 1 for rep, _ in REP for k in A_[rep]["ks"]])
# points of the ablation runs against the bound of their own representation (summary.json, check)
CKA = [js(ABL_SUMMARY[mp])["check"] for mp in ("conserved", "proportional")]
assert all(c[k]["over"] == 0 and c[k]["over_tol"] == 0 for c in CKA for k in ("demand", "stable"))
num("AblCheckPoints", sum(c[k]["reserve"] for c in CKA for k in ("demand", "stable")))

# --- the whole network. Full-path reservation's two-day schedules (100 vehicles, three seeds a yard, main case)
# retimed with every resource order and vehicle sequence fixed (difference constraints, longest path), retimed with the
# dispatch fixed (rolling constraint programming, 25 min), and dispatched offline to the vehicle that completes each
# task earliest under the same booking; shares of the system bound, destinations held for the empty vehicle's
# drive-off as in full-path reservation (Definition 1(c) beside it); every schedule passed the independent checker
WH = {mp: rows(RETIME[mp]) for mp in RETIME}
for mp, rs in WH.items():
    assert len(rs) == 6 and all(r[k] == "True" for r in rs for k in ("orig_check", "all_check", "same_check", "ect_check",
                                                                    "ect_def1c_check"))
    assert all(int(r["all_rounds"]) == 0 and int(r["window_mismatch"]) == 0 for r in rs)
WC = WH["conserved"]
CW = js("results/retiming/main_case_config.json")
assert CW["demand_map"] == "conserved" and CW["scenario"] == "main" and {int(r["K"]) for r in WC} == {CW["K"]}
num("WholeK", CW["K"])
num("WholeDays", CW["days"])
num("WholeSeeds", len(CW["seeds"]))
num("WholeSameMin", CW["same_minutes"])
assert all(close(r["lamS"], lamC[r["yard"]], 0.01) for r in WC)
assert all(close(r["plat_res"], SAT61[(r["yard"], "conserved", "reserve")]["plateau"], 1e-6) for r in WC)
rng3 = lambda vals: f"{min(vals):.2f}--{max(vals):.2f}" if f"{min(vals):.2f}" != f"{max(vals):.2f}" else f"{min(vals):.2f}"
lines = []
for y in YARDS:
    rs = [r for r in WC if r["yard"] == y]
    lines.append(f"{yname[y]} & " + " & ".join(rng3([float(r[c]) for r in rs]) for c in
                                                ("plat_res_S", "orig_S", "all_S", "same_S", "ect_S", "ect_def1c_S", "plat_ref_S"))
                 + r" \\")
write("tab_whole.tex", lines)
ECT = rows("results/retiming/offline_dispatch.csv")
assert {(r["yard"], r["seed"]) for r in ECT} == {(r["yard"], r["seed"]) for r in WC}
assert all(close(e["clr_S"], next(r["ect_S"] for r in WC if (r["yard"], r["seed"]) == (e["yard"], e["seed"])), 1e-9) for e in ECT)
prange("WholeEct", [float(r["ect_S"]) for r in WC])
prange("WholeEctGain", [float(r["clr_gain"]) for r in ECT])
num("WholeConsLo", min(int(r["all_edges"]) for r in WC))
num("WholeConsHi", max(int(r["all_edges"]) for r in WC))
num("WholeRetimeLo", min(float(r["rt_done_sum"]) / int(r["N"]) for r in WC), 1)
num("WholeRetimeHi", max(float(r["rt_done_sum"]) / int(r["N"]) for r in WC), 1)
put("WholeRetimeThr", fmt(100 * max(abs(float(r["all_S"]) / float(r["orig_S"]) - 1) for r in WC), 2) + r"\%")
put("WholeSameMax", fmt(100 * max(float(r["same_search"]) for r in WC), 3) + r"\%")
prange("WholeDisp", [float(r["sp_dispatch"]) for r in WC])
prange("WholeRest", [float(r["sp_rest"]) for r in WC])
prange("WholeIdle", [float(r["ect_idle"]) for r in WC])
prange("WholeChain", [float(r["ect_idle_chain"]) for r in WC])
pct("WholeVehMax", max(float(r["ect_idle_vehicle"]) for r in WC), 1)
corr = [r for r in WC if res_name(r["binding"]) == "through corridor"]
assert [(r["yard"], r["seed"]) for r in corr] == [("yupu", "101"), ("yupu", "103")]
prange("WholeCorrBusy", [float(r["ect_busy"]) for r in corr])
put("WholeLamRatio", rng3([float(r["lamN"]) / float(r["lamS"]) for r in WC]))
prange("WholeEctScaled", [float(r["ect_S"]) for r in WH["proportional"]])
put("WholeDefOneC", f"$-${fmt(-100 * min(float(e['def1c_vs_clr']) for e in ECT), 1)} to $+${fmt(100 * max(float(e['def1c_vs_clr']) for e in ECT), 1)}\\%")

# --- the screen in two levels. Lower end: the demand ratio (for a fixed mix the utilisation of the binding
# element) below which full-path reservation needed no more vehicles than free flow in any tested cell, the lowest first
# 10 % crossing of the fractional fleets over the ten series (two yards x 16/24 h, roads 1.5 m narrower or wider, the
# intra-area mapping); upper end: per yard, read on the chain load S_d (the expected sum of utilisations along the
# chain a vehicle books, empty leg included), on which the three road widths of a yard coincide (fig7.csv, cross.csv)
F7 = rows("results/screen/series.csv")
CR7 = {(r["series"], r["axis"], r["measure"]): r for r in rows("results/screen/cross.csv")}
SER = sorted({r["series"] for r in F7})
assert len(SER) == 10 and len(F7) == 136
low = min(float(CR7[(s, "ratio", "gap_frac")]["first"]) for s in SER)
assert fmt(100 * low, 1) == "5.3"
pct("ScreenLowEnd", low, 1)
below = [r for r in F7 if float(r["ratio"]) < low]
assert all(float(r["gap_frac"]) <= 0.10 for r in below)
# with ten seeds two cells below it needed one vehicle more; with 30 seeds both need equal fleets (seeds_levels.csv:
# Okpo's 16-h levels of 2.9-5.8 % at three road widths, 30 seeds; summary round2: the intra-area cell)
odd7 = sorted((r["series"], int(r["n"])) for r in below if int(r["K_res"]) > int(r["K_free"]))
assert odd7 == [("intra_16", 150), ("okpo_w-1.5", 75)], odd7
S64 = js("results/screen/summary.json")
SL = [r for r in rows("results/screen/seeds_levels.csv") if float(r["ratio"]) < low]
assert next(r for r in SL if (r["scen"], r["n"]) == ("d16_w-1.5", "75"))["K_res"] == "4" and all(r["K_free"] == r["K_res"] for r in SL)
(rerun,) = set(S64["round2"]["intra"]["fleets"]) - {MAC["Seeds"]}      # the seed counts run: the paper's and the rerun's
num("ScreenSeedsRerun", rerun)
I30 = S64["round2"]["intra"]["fleets"][rerun]
assert I30["K_free"] == I30["K_res"] and abs(S64["round2"]["intra"]["ratio"] - 0.0474) < 1e-3
# the main case (series_main_case.csv, cross_main_case.csv; ten seeds) and its cells up to 6 % rerun with 30 seeds
# (seeds_levels.csv): below the lower end, read on the ten series, one cell, Okpo at 16 h and 75 moves/day, needs one
# vehicle more than free flow with ten and with 30 seeds (four vehicles complete 98.8 % of the demand against the 99 %
# criterion); with the 48 cells of the ten series, whose two odd cells need equal fleets with 30 seeds, the same fleet in 63 of 64
F7C = rows("results/screen/series_main_case.csv")
CR7C = {(r["series"], r["axis"], r["measure"]): r for r in rows("results/screen/cross_main_case.csv")}
S69 = js("results/screen/summary_main_case.json")
assert close(S69["low_end"]["low"], low, 1e-12) and len(below) == S69["low_end"]["ten_series_cells"]
belowC = [r for r in F7C if float(r["ratio"]) < low]
assert len(belowC) == S69["low_end"]["cells"] and {r["series"] for r in belowC} == {"okpo_c16", "okpo_c24", "yantai_c16",
                                                                                     "yantai_c24"}
oddC = [r for r in belowC if int(r["K_res"]) > int(r["K_free"])]
SLC = [r for r in rows("results/screen/seeds_levels_main_case.csv") if float(r["ratio"]) < low]
assert sorted((r["series"], r["n"]) for r in SLC) == sorted((r["series"], r["n"]) for r in belowC)
oddC30 = [r for r in SLC if int(r["K_res"]) > int(r["K_free"])]
assert [(r["series"], int(r["n"])) for r in oddC] == [(r["series"], int(r["n"])) for r in oddC30] == [("okpo_c16", 75)]
assert int(oddC30[0]["K_res"]) == int(oddC30[0]["K_free"]) + 1 and float(oddC30[0]["res_below_share"]) < 0.99
num("ScreenLowCells", len(below) + len(belowC))
num("ScreenLowCellsTen", len(below))                     # the ten series
num("ScreenLowCellsMain", len(belowC))                   # the four series of the main case
num("ScreenLowSame", len(below) + len(belowC) - len(oddC30))
pct("ScreenOddRatio", oddC[0]["ratio"], 1)
pct("ScreenOddShare", oddC30[0]["res_below_share"], 1)
num("ScreenOddK", oddC30[0]["K_free"])
num("ScreenOddKPlus", oddC30[0]["K_res"])
num("ScreenOddFracDiff", oddC30[0]["dfrac"], 2)          # its fractional fleets, 30 seeds
assert len(odd7) + len(oddC) == 3                         # one vehicle more with ten seeds: three cells of 64
# the fractional fleets of the two rules below the lower end differ by at most ... vehicles (ten seeds, all series;
# 30 seeds where rerun)
dfr = [float(r["K_res_frac"]) - float(r["K_free_frac"]) for r in below] + [float(r["dfrac"]) for r in SL] + \
      [I30["frac_res"] - I30["frac_free"]] + [float(r["dfrac"]) for r in belowC + SLC]
num("ScreenLowFracMax", max(dfr), 2)
assert max(dfr) < 0.5 and close(max(float(r["dfrac"]) for r in belowC), S69["low_end"]["dfrac_max"], 1e-12)
num("ScreenLowFracMaxTen", max([float(r["K_res_frac"]) - float(r["K_free_frac"]) for r in below] +
                               [float(r["dfrac"]) for r in belowC]), 2)          # ten seeds, all 64 cells
up16 = [float(CR7[(s, "ratio", "gap_frac")]["last"]) for s in SER if s.endswith("16") or "1.5" in s] + \
       [float(CR7C[(s, "ratio", "gap_frac")]["last"]) for s in ("okpo_c16", "yantai_c16")]
# the main case's upper ends at 16 h, on the ratio and on the chain load, inside each yard's band of three road widths
for y, s in (("Yu", "okpo_c16"), ("Ya", "yantai_c16")):
    pct(y + "ConsUp", CR7C[(s, "ratio", "gap_frac")]["last"], 1)
    num(y + "ConsChainUp", CR7C[(s, "S_d", "gap_frac")]["last"], 2)
    pct(y + "ConsUpTwentyFour", CR7C[(s.replace("16", "24"), "ratio", "gap_frac")]["last"], 1)   # supplementary only
pct("ScreenUpLo", min(up16), 1)
pct("YuIntraUp", CR7[("intra_16", "ratio", "gap_frac")]["last"], 1)    # Okpo's intra-area series on 16-h days
pct("ScreenUpHi", max(up16), 1)
assert MAC["ScreenUpHi"] == fmt(100 * max(float(CR7[(s, "ratio", "gap_frac")]["last"]) for s in SER), 1) + r"\%"
for y, s3 in (("Yu", ("okpo_16", "okpo_w-1.5", "okpo_w+1.5")), ("Ya", ("yantai_16", "yantai_w-1.5", "yantai_w+1.5"))):
    sd = [float(CR7[(s, "S_d", "gap_frac")]["last"]) for s in s3]
    put(y + "ChainUp", f"{min(sd):.2f}--{max(sd):.2f}")
    assert min(sd) <= float(CR7C[({"Yu": "okpo_c16", "Ya": "yantai_c16"}[y], "S_d", "gap_frac")]["last"]) <= max(sd)
    if y == "Yu":
        pct("YuChainSpread", (max(sd) - min(sd)) / statistics.median(sd), 1)
        xr = [float(CR7[(s, "ratio", "gap_frac")]["last"]) for s in s3]
        pct("YuRatioSpread", (max(xr) - min(xr)) / statistics.median(xr), 1)

# why the upper ends differ between the yards (waits_pairs.csv, wide.csv): at equal demand ratio, the wait for a
# booking relative to the no-wait cycle (Okpo over Yantai) and the pushes on roads; widening every Okpo road of up to
# 15 m to 18 m (same seeds and fleets) cuts the road pushes, the pushes per road passage fall towards Yantai's, and at
# the higher pair of levels the relative wait equals Yantai's
WP = rows("results/screen/waits_pairs.csv")
assert len(WP) == 2
put("ScreenWaitRatio", f"{min(float(r['W_C0']) for r in WP):.1f}--{max(float(r['W_C0']) for r in WP):.1f}")
put("ScreenRoadPush", f"{min(float(r['road_push']) for r in WP):.1f}--{max(float(r['road_push']) for r in WP):.1f}")
put("ScreenOtherPush", f"{min(float(r['other_push']) for r in WP):.1f}--{max(float(r['other_push']) for r in WP):.1f}")
WD = rows("results/screen/wide.csv")
cut = [1 - float(r["wide_road_push"]) / float(r["base_road_push"]) for r in WD]
prange("WideCut", cut)
ppv = lambda r, k: float(r[k + "_road_push_per_visit"]) / float(r["yt_road_push_per_visit"])
put("WidePassBefore", f"{min(ppv(r, 'base') for r in WD):.1f}--{max(ppv(r, 'base') for r in WD):.1f}")
put("WidePassAfter", f"{min(ppv(r, 'wide') for r in WD):.1f}--{max(ppv(r, 'wide') for r in WD):.1f}")
put("WideSeries", f"{min(float(r['base_road_visits']) / float(r['yt_road_visits']) for r in WD):.1f}")
hi_ = max(WD, key=lambda r: float(r["base_x"]))
pct("WideEqLevel", hi_["base_x"], 1)
num("WideEqWait", float(hi_["wide_W_C0"]) / float(hi_["yt_W_C0"]), 2)
assert abs(float(hi_["wide_W_C0"]) / float(hi_["yt_W_C0"]) - 1) < 0.02
pct("WideBoundRise", S64["round2"]["wide"]["lam_wide"] / S64["round2"]["wide"]["lam_base"] - 1, 1)
assert res_name(S64["round2"]["wide"]["binding"][0]["res"]) == "stockyard road"
# the widening: every Okpo road of at most WideFrom m to WideTo m (the retrofit of the run's config; the threshold is in
# its description, and the retrofit lists as many roads as the summary)
CW = js("results/screen/wide_config.json")
(RW,) = CW["wide"]["retrofit"]
assert RW["kind"] == "widen" and len(RW["roads"]) == S64["round2"]["wide"]["n_roads"]
num("WideTo", RW["to"])
num("WideFrom", re.search(r"at most (\d+) m widened to (\d+) m", CW["about"]).group(1))
assert re.search(r"widened to (\d+) m", CW["about"]).group(1) == MAC["WideTo"]
num("WideBindW", re.search(r"W=([\d.]+)", S64["round2"]["wide"]["binding"][0]["res"]).group(1))

# the main case: free flow's rate per vehicle at small fleets (the fleet bound, Corollary fleet) and the dual price of
# the binding element (saturation_curves.csv at K = 10, binding.csv)
CUR = {(r["yard"], r["model"], int(r["K"])): float(r["mean"]) for r in rows("results/main_case/saturation_curves.csv")}
BND = [r for r in rows("results/main_case/binding.csv") if float(r["q"]) == 0]
for y in YARDS:
    num(Y[y] + "ConsFreeSlope", CUR[(y, "free", 10)] / 10, 1)
    pos_ = [r for r in BND if r["yard"] == y and float(r["price_per_h"]) > 0]
    assert len(pos_) == 1                              # one priced element per yard; the text quotes Okpo's price
    if y == "yupu":
        num("YuConsPrice", pos_[0]["price_per_h"], 1)
num("YaConsDoorUtilA", next(float(r["util"]) for r in BND if r["yard"] == "yantai" and r["rank"] == "1"), 3)
num("YaConsDoorUtilB", next(float(r["util"]) for r in BND if r["yard"] == "yantai" and r["rank"] == "2"), 3)
# the 16-h peak needs more vehicles in the main case than with every move scaled because of the roads, not the docks:
# per move, the loaded occupancy of the corridor roads rises and that of the Dock 1 road falls (loaded_occupancy.csv); the
# waiting on the two dock-front roads falls (waits.csv, 678 moves/day, 121 vehicles)
LD = {r["road"]: float(r["rel_change"]) for r in rows("results/main_case/loaded_occupancy.csv")}
prange("PeakCorrRise", [v for v in LD.values() if v > 0], 1)
pct("PeakDockFall", -LD["161"])
WT = {r["map"]: r for r in rows("results/main_case/peak_waits.csv")}
assert WT["current"]["K"] == WT["conserved"]["K"] and WT["current"]["level"] == f"n{MAC['OkPeakLo']}"
num("PeakWaitK", WT["current"]["K"])
num("PeakDockWaitProp", WT["current"]["ring0"])
num("PeakDockWaitCons", WT["conserved"]["ring0"])

# roads shorter than a vehicle with its clearance (network/short_roads_by_road.csv): none of them is among the elements that bind,
# or come next, in any system bound reported here. Checked: the top elements (three where the result file lists three,
# else two) of the bounds of every scenario, mapping, granularity, handling time, road width, ablation, ban and scan
SR = rows("results/network/short_roads_by_road.csv")
thr_s = {float(r["thr"]) for r in SHORT.values()}
assert thr_s == {float(MAC["MaxVehClear"])}
short_names = {(r["yard"], r["name"]) for r in SR if r["kind"] == "road"}
BTOP = []                                       # one entry per reported bound: (yard, its top elements)
for y in YARDS:                                 # bounds/<yard>/prices.csv: the ten highest of each scenario, by rank
    BTOP += [(y, [r["name"] for r in PR[y] if r["scen"] == sc and int(r["rank"]) <= 3]) for sc in SC]
BTOP += [(r["yard"], [r[c] for c in ("top1", "top2", "nopack_top") if r.get(c)])
         for r in rows("results/bounds/sensitivity.csv")]
BTOP += [(r["yard"], [r[c] for c in ("binding1", "binding2") if r.get(c)]) for r in rows("results/bounds/system_by_granularity.csv")]
BTOP += [(r["yard"], [r[c] for c in ("top1", "top2") if r.get(c)]) for r in rows("results/interval/crane_interval.csv")]
BTOP += [(r["yard"], [r["top1"], r["top2"], r["top3"]]) for r in rows("results/intra_area/bounds.csv")]
BTOP += [(r["yard"], [r[c] for c in ("top1", "top2", "top3") if r[c] != "—"]) for r in B61.values()]
BTOP += [(r["yard"], [r["top1"], r["top2"], r["top3"]]) for r in SCAN]
BTOP += [(r["yard"], [r["top1"], r["top2"], r["top3"]])
         for r in rows("results/main_case/bounds_sections.csv")]
BTOP += [(r["yard"], [r["name1"], r["name2"], r["name3"]]) for mp in ABL_BOUNDS for r in rows(ABL_BOUNDS[mp])]
BTOP += [(b["yard"], [b[c] for c in ("top1", "top2") if b.get(c)]) for b in S52["bounds"]]
BTOP += [("yupu", [t["name"] for t in S64["round2"]["wide"]["top"]])]
road_of = lambda s: s.split(" (")[0] if s.startswith("road") else None      # node resources (junctions, doors, bays) aside
hits_ = [(y, n) for y, ns in BTOP for n in ns if road_of(n) and (y, road_of(n)) in short_names]
lens = [float(m.group(1)) for _, ns in BTOP for n in ns if road_of(n) for m in [re.search(r", ([\d.]+) m\)", n)] if m]
assert not hits_ and min(lens) > float(MAC["MaxVehClear"]) and all(len(ns) >= 2 for _, ns in BTOP), (hits_, min(lens))
num("ShortCheckBounds", len(BTOP))
num("ShortCheckElems", len({(y, n.split(" (")[0]) for y, ns in BTOP for n in ns}))
num("ShortRoadMinTop", min(lens))

# ---------------------------------------------------------------- packing rows, platform size, tail clearance
# the main case with the utilisation rows alone (k = 0): the bound, the tightening by the width-packing rows and
# the row that carries the price of the binding element (nopack.csv); Yantai's door rows coincide, no tightening
NP = {(r["map"], r["yard"], r["section"]): r for r in rows("results/bounds/packing_rows.csv")}
npY, npA = NP[("conserved", "yupu", "whole")], NP[("conserved", "yantai", "whole")]
assert close(npY["T1_sys"], lamC["yupu"], 1e-6) and close(npA["T1_sys"], lamC["yantai"], 1e-6)
assert float(npY["price_k0"]) == 0 == float(npY["price_k2"]) and close(npY["price_k1"], MAC["YuConsPrice"], 0.05)
assert close(npY["T1_k01"], npY["T1_sys"], 1e-9) and float(npA["pack_tighten"]) == 0
assert close(NP[("proportional", "yupu", "whole")]["pack_tighten"], 0.124, 1e-3)              # \PackTightCut (bounds/sensitivity.csv)
num("YuNoPack", npY["T1_nopack"])
pct("YuPackTight", npY["pack_tighten"], 1)
# platform width and length, main case, whole roads, 16 h (plat.csv); the reported 21 m x 7.5 m platform
PLT = {(r["yard"], float(r["plat_w"]), float(r["plat_len"])): r for r in rows("results/bounds/platform_size.csv")}
assert all(close(PLT[(y, 5.5, 16.25)]["T1_sys"], lamC[y], 1e-6) for y in YARDS)
assert all(PLT[k]["top1"] == PLT[(k[0], 5.5, 16.25)]["top1"] for k in PLT)            # binding elements unchanged
for y in YARDS:
    pct(Y[y] + "PlatBig", -float(PLT[(y, 7.5, 21.0)]["rel"]), 1)                         # 21 m x 7.5 m
assert PLT[("yupu", 6.5, 16.25)]["T1_sys"] == PLT[("yupu", 7.5, 16.25)]["T1_sys"]     # any width above 5.5 m alike
pct("YuPlatWide", -float(PLT[("yupu", 6.5, 16.25)]["rel"]), 1)
pct("YuPlatLong", -float(PLT[("yupu", 5.5, 21.0)]["rel"]), 1)
pct("YaPlatLong", -float(PLT[("yantai", 5.5, 21.0)]["rel"]), 1)
assert PLT[("yantai", 7.5, 16.25)]["T1_sys"] == PLT[("yantai", 5.5, 16.25)]["T1_sys"]
assert fmt(5.5, 1) == MAC["PlatW"] and fmt(16.25, 2) == MAC["PlatLen"]             # the platform of the paper
num("PlatRepW", max(k[1] for k in PLT), 1)  # the transporters reported by KED Global (kim2024hanwha): the widest and
num("PlatRepLen", max(k[2] for k in PLT))   # longest platform of plat.csv
assert (MAC["PlatRepW"], MAC["PlatRepLen"]) == ("7.5", "21") and PLT[("yupu", 7.5, 21.0)]["rel"] == PLT[("yupu", 6.5, 21.0)]["rel"]
# the main case's 16-h series at Okpo: first 10 % crossing with fractional fleets (cross_conserved.csv), below the
# lower end read on the ten series; and the main-case cells below the lower end with equal fleets (30 seeds)
pct("YuConsFirstFrac", CR7C[("okpo_c16", "ratio", "gap_frac")]["first"], 1)
num("ScreenLowSameMain", len(belowC) - len(oddC30))
# the best schedule around the Dock 1 road: its distance from the bound, extrapolated to long runs (Section 4.3)
pct("SubDockGapMax", 1 - float(MAC["SubDockExtrap"].split("--")[0]), 1)
# roadside stops less than a clearing length beyond the node behind them, whole-road networks (tail/roadside_stops.json)
RS72 = js("results/tail/roadside_stops.json")
stops = {f"{Y[y]}{k}": RS72[y][c] for y in YARDS for k, c in (("TailStops", "within_platform_clearing"),
                                                             ("TailStopsL", "within_block_clearing"),
                                                             ("RoadStops", "roadside_stops"))}
for k, v in stops.items():
    num(k, v)
assert all(stops[f"{y}TailStops"] <= stops[f"{y}TailStopsL"] <= stops[f"{y}RoadStops"] for y in ("Yu", "Ya"))
# full-path reservation booking the body of a vehicle at a roadside stop until its tail clears (complete rule,
# "lw"): plateaus (sat_summary.csv) and Okpo's own-volume fleets on 16-h days (fleet.csv), against release by timer
S74 = {r["yard"]: r for r in rows("results/tail/saturation.csv")}
for y in YARDS:
    assert close(S74[y]["plateau_cur"], SAT61[(y, "conserved", "reserve")]["plateau"], 1e-6)
    num(Y[y] + "TailPlat", S74[y]["plateau_lw"])
    pct(Y[y] + "TailDrop", -float(S74[y]["rel_lw"]), 1)
F74 = {int(r["demand"]): r for r in rows("results/tail/fleet.csv")}
assert [int(F74[n]["K_stable_cur"]) for n in (339, 350)] == [cK(16, "reserve", n) for n in (339, 350)]
irange("YuTailStable", [int(F74[n]["K_stable_lw"]) for n in (339, 350)])
prange("YuTailStableExtra", [float(F74[n]["more_stable_lw"]) for n in (339, 350)])

# ---------------------------------------------------------------- the hand-computed instances (Appendix A.6, Section S4)
# Their geometry is that of tests/test_occupancy.py (the hand values of the corridor, the crossroads and the reuse of a
# stop); the bounds are computed here from the paper's parameters and checked against the values computed by hand
# (hand/instances.json), which also gives the simulated throughputs of the three rules on the same instances
HAND = js("results/hand/instances.json")
H_, v1_, v0_, c_, s_ = P["day_s"], P["v_loaded"], P["v_empty"], P["lat_clear"], P["long_clear"]
L0_, w_, tH = P["plat_len"], P["plat_w"], P["load_s"]
assert P["unload_s"] == tH and H_ == 16 * 3600
Lv = 20.0                                         # block length of the test tasks, longer than the platform
# (1) single-lane corridor: road 12 m wide (capacity 13), middle segment 2000 m between nodes 50 m from each end;
# every task A -> B with a block 7 m wide (loaded width 8, empty 6.5)
C1, B1, D1 = 12 + c_, 7.0, 2000.0
win1 = (D1 + Lv + s_) / v1_                       # loaded window on the middle segment
emp1 = D1 / v0_ + (L0_ + s_) / v0_                # empty traversal and its tail
lam1 = H_ / (win1 * 1 + 0.5 * emp1)               # k = 1: loaded share 8/13 > 1/2 counts 1, empty 6.5/13 = 1/2
assert (B1 + c_) / C1 > 0.5 and (w_ + c_) / C1 == 0.5
# (2) four-arm crossroads: arms 1000 m, 15 m wide (capacity 16), blocks 10 m wide (11 of 16), empty runs take no width,
# handling 60 s; each arm is held by half of the tasks for its loaded window
C2, B2, D2 = 15 + c_, 10.0, 1000.0
win2 = (D2 + Lv + s_) / v1_
lam2 = H_ / (0.5 * win2)
# (3) one stop served by consecutive tasks: entrance X behind a 10-m road; a vehicle that unloads at X loads there next
# (Y -> X, then X -> Y); X, held exclusively, is busy per pair of tasks for the drive along the 10-m road, unloading,
# loading and the tail clearance, and the empty flow is zero
D3 = 10.0
pair3 = D3 / v1_ + tH + tH + (Lv + s_) / v1_
lam3 = 2 * H_ / pair3
doc_vals = HAND["hand_values"]
for v in (lam1, lam2, lam3):
    assert any(abs(v - d) < 0.006 for d in doc_vals), (v, doc_vals)
num("HandBlockLen", Lv)
num("HandOneRoadW", C1 - c_)
num("HandOneBlockW", B1)
num("HandOneLen", D1)
num("HandTwoRoadW", C2 - c_)
num("HandTwoBlockW", B2)
num("HandTwoArm", D2)
num("HandTwoHandling", 60)                         # handling time of the crossroads test, seconds
num("HandThreeShort", D3)
num("HandOneWin", win1)
num("HandOneEmpty", emp1, 3)
num("HandOneBound", lam1, 2)
num("HandTwoWin", win2)
num("HandTwoMean", 0.5 * win2, 1)
num("HandTwoBound", lam2, 2)
num("HandThreePair", pair3)
num("HandThreeDrive", D3 / v1_)
num("HandThreeTail", (Lv + s_) / v1_)
num("HandThreeBound", lam3, 2)
num("HandThreeTaskMean", pair3 / 2, 1)
# the simulated throughputs of the three rules (the segment-wise rule there is the reference rule: teleport recovery)
r1, r2, r3 = HAND["corridor"], HAND["crossroads"], HAND["consecutive"]
lst = lambda v: ", ".join(str(x) for x in v)
put("HandOneK", lst(r1["K"]))
put("HandOneFree", lst(r1["free"]))
put("HandOneRes", lst(r1["reserve"]))
put("HandOneSeg", lst(r1["reference"]))
put("HandTwoK", lst(r2["K"]))
put("HandTwoFree", lst(r2["free"]))
put("HandTwoRes", lst(r2["reserve"]))
put("HandTwoSeg", lst(r2["reference"]))
num("HandThreeKRes", r3["K"]["reserve"])         # full-path reservation, segment-wise rule
num("HandThreeKSeg", r3["K"]["reference"])
num("HandLongDays", r3["days"])                    # days run, days counted
num("HandLongCount", r3["counted_days"])
assert HAND["recovery_s"] > 0
num("HandThreeRes", r3["reserve"], 2)
num("HandThreeSeg", r3["reference"], 2)
assert abs(float(MAC["HandThreeSeg"]) - lam3) < 0.075 and fmt(r1["bound"], 2) == MAC["HandOneBound"] and \
    fmt(r2["bound"], 2) == MAC["HandTwoBound"]
assert r3["reference_deadlocks"] == 0 and fmt(r3["bound"], 2) == MAC["HandThreeBound"]

# ---------------------------------------------------------------- numbers.tex
out = ["% generated by make_numbers.py from the result files; do not edit"]    # neutral: the file goes to the journal
out += [rf"\newcommand{{\{k}}}{{{v}\xspace}}" for k, v in MAC.items()]
(ROOT / "numbers.tex").write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"{len(MAC)} macros, tables in {TAB}")

"""Read the result data of the repository (results/) for the scripts that write the paper's numbers, tables and figures.

Every number, table of results and data figure of the paper comes from the files read here; results/MANIFEST.csv lists
each file with its SHA-256 and where the paper uses it. Paths are relative to the repository root.

Yard keys: "yupu" is Okpo (the Hanwha Ocean yard on Geoje; Yupu is its Chinese name), "yantai" the CIMC Raffles yard at Yantai.
Okpo's readings are on network revision 1833 (with the floating docks); some runs are on revision 1825, which differs only by
three spur roads at the floating docks and the end points of three roads (results/network/okpo_revisions.json).
Mappings of demand to moves: the main case (erection flow conserved, every other move between areas; results/main_case/),
every move scaled with the flows (the proportional mapping: the bound tables in results/bounds/okpo, results/bounds/yantai and
most runs under results/runs/), and moves within areas (results/intra_area/).
"""
import csv
import io
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# functional names of map elements (no map identifiers in displayed text), shared by the bound table and the figures
# (the specific keys first: Yantai's building 007, the hall that Yantai's intra-area moves pass, has a door named
# entrance001 node too; Okpo's building 006 is one of its paint shops)
RES_NAME = {"building007·steel processing": "panel-line hall doors", "zone012": "assembly-area bays",
            "building006·paint": "paint-shop doors", "road098-1": "stockyard road", "road161": "Dock 1 road",
            "road010-6": "through corridor", "road013-7": "bypass corridor", "road007": "Dock 2 road",
            "road099-4": "east corridor", "road026": "central road", "entrance001 node": "paint-shop doors"}


def res_name(s):
    for k, v in RES_NAME.items():
        if k in s:
            return v
    raise KeyError(s)


# run directories (results/runs/<run>: config.json, points.csv where released, check.json = the per-point check of the run)
BOUNDS_DIR = {"yupu": "results/bounds/okpo", "yantai": "results/bounds/yantai"}        # system, route-free bounds, prices
MAIN_SAFE = {"yupu": "results/runs/okpo_rev1825_safe_sectioned",                      # safe release, sectioned
             "yantai": "results/runs/yantai_safe_sectioned"}
RHO_SAFE = {"yupu": "results/runs/okpo_rev1825_dock_safe",                            # crane-paced rho curves (dock mouth)
            "yantai": "results/runs/yantai_dock_safe"}
FLAG_RUNS = {"yantai": "results/runs/yantai_rev395_saturation"}                       # Yantai saturation curves, three rules
CRANE_RUNS = ["okpo_crane", "yantai_crane", "okpo_handling_30min",
              "yantai_handling_30min"]                                                # crane-paced and 30-min saturated runs
CRANE_FLEET = {"yupu": "results/runs/okpo_crane_fleet", "yantai": "results/runs/yantai_crane_fleet"}   # fleets, crane pacing
SEC_FLEET = "results/runs/okpo_fleet_sectioned"                                       # sectioned full-path reservation fleets
OD_RUNS = {"yupu": "results/runs/okpo_od_sampling", "yantai": "results/runs/yantai_od_sampling"}  # bounds, sampling by stop
SYN_DOCK = "results/synthetic_dock"                                                   # synthetic dock-mouth network
SYN_WAIT = "results/runs/synthetic_dock_waiting"                                      # its waiting term (per seed)
# the revision's tables kept per mapping: the main case ("conserved") and every move scaled ("proportional")
ABL_BOUNDS = {"conserved": "results/ablation/bounds_main_case.csv",                   # ablation: system bounds
              "proportional": "results/ablation/bounds_proportional.csv"}
ABL_SUMMARY = {"conserved": "results/ablation/summary_main_case.json",                # ablation: fleets and checks
               "proportional": "results/ablation/summary_proportional.json"}
RETIME = {"conserved": "results/retiming/main_case.csv",                              # whole-network retiming
          "proportional": "results/retiming/proportional.csv"}
csv.field_size_limit(10 ** 9)                     # some point files carry long per-day fields


def show(path):
    return (REPO / path).read_text(encoding="utf-8-sig")


def rows(path):
    return list(csv.DictReader(io.StringIO(show(path))))


def js(path):
    return json.loads(show(path))


def interval():
    """Capacity-interval rows {(yard, kind, gran): row}; the paper reads their system bounds (T1_sys)."""
    return {(r["yard"], r["kind"], r["gran"]): dict(r) for r in rows("results/interval/interval.csv")}


def demand(eight_day=True):
    """Required-fleet rows of every move scaled with the flows (results/fleet/required_fleet.csv): Okpo on revision 1833,
    Yantai on revision 396, with the Yantai full-path reservation cells taken from the integer-K-filled main-mix searches of
    the crane-pacing runs where they differ (only 450 moves/day on 16-h days: 69 -> 66, a fleet above which the target fails
    again), and Okpo's headline cells (339 and 350 moves/day, free flow and full-path reservation, whole roads, both working
    days) replaced by their eight-day values with the eight-day seed-bootstrap ranges (fleet/fleet_8day.csv; only 350
    moves/day, 24 h, full-path reservation changes: 16 -> 15); eight_day=False keeps the four-day values, for comparisons
    with other four-day runs. The safe-release searches have every integer below the first success simulated."""
    dem = rows("results/fleet/required_fleet.csv")
    assert {r["map"] for r in dem if r["yard"] == "yupu" and r["section"] == "whole"} == {"rev 1833"}
    key = lambda r: (r["yard"], float(r["day_h"]), r["section"], r["model"], int(float(r["demand"])))
    idx = {key(r): r for r in dem}
    changed = []
    for r in rows(CRANE_FLEET["yantai"] + "/demand.csv"):
        if r["scen"] in ("main_d16", "main_d24") and r["model"] == "reserve":
            t = idx[key(r)]
            if t["K_req"] != r["K_req"]:
                assert int(r["K_req"]) < int(t["K_req"]) and r["fail_above"] not in ("", "[]")
                t.update(K_req=r["K_req"], fail_above=r["fail_above"])
                changed.append((r["scen"], r["demand"]))
    assert changed == [("main_d16", "450")], changed
    if not eight_day:
        return dem
    changed = []
    for r in rows("results/fleet/fleet_8day.csv"):
        t = idx[("yupu", float(r["day_h"]), "whole", r["model"], int(r["demand"]))]
        assert t["K_req"] == r["K_4d"] and r["K_4d_ci"] == f"[{t['K_lo']}, {t['K_hi']}]", (r, t)   # same four-day cell
        assert r["fail_8d"] == "[]" and r["missing_8d"] == "[]" and r["K_8d_cens"] == "0"
        if r["K_8d"] != t["K_req"]:
            changed.append((r["day_h"], r["model"], r["demand"]))
        t.update(K_req=r["K_8d"], K_lo=r["K_8d_lo"], K_hi=r["K_8d_hi"], fail_above=r["fail_8d"])
    assert changed == [("24", "reserve", "350")], changed
    return dem

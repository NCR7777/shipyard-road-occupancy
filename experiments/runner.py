"""Batch runner: simulated points (fleet size vs throughput under the traffic rules) and sampled network bounds of one batch
configuration, as stored with every run in results/runs/<run>/config.json.

A configuration names the network (data/networks/<yard>/<revision>), the base parameters, the scenarios (parameter
overrides), the traffic rules per scenario (models), the demand levels (tasks a day; None = saturated, an endless supply of
tasks), the fleet sizes K, the seeds and the size of the task samples of the bound (n_mix). A point is
"<scenario>|<rule>|<level>|<K>|<seed>"; a bound sample is "<scenario>|bound|<seed>".

Scenario keys of the revision turn to modules outside occupancy/ (without them every code path is that of the earlier runs):
demand_map (task stream of a demand mapping: demand_mapping, demand_mapping_var with intra_var, conserved_mapping),
daily_stats (daily readings under continuous release: daily), stoch (random handling and driving times: stochastic),
width_ban (ban on blocks wider than the passable width: width_ban), bay_spots (unloading bay and off-road waiting place at the
dock: dock_spots), occ_rep (ablation of the occupancy representation: ablation, ablation_conserved), a retrofit change of kind
door_stop (doors for passage only, with handling bays inside: the network class intervention_net.DoorNet). Rules outside
occupancy.sim: reserve_rb (full-path reservation with late rebooking: stochastic), reserve_op (execution in booked order:
ordered), reserve_op2 (booking on projected occupancy: ordered_proj), reserve_op3 (the same with mean durations: op3_sim),
reserve_tlw and reserve_tlw_timer (booking the tail, and release by timer counted by the same definition: tail_booking).

Each simulated point of an occupancy rule is checked against the network bound T1' of the tasks completed in its own
statistics window (T1_tight_mix, occupancy.bound.bounds_window: lifts by a gantry crane in the same window hold the cranes in
absolute terms, crane rows with the window edge term); the throughput must not exceed it. Under random times the bound is
taken at the nominal times (bound_params). Saturated points also get the steady-state readings (mix shift, daily drift).

Run:   python experiments/runner.py results/runs/<run>/config.json --out <dir> [--procs N] [--only KEY_PREFIX ...] [--slots DIR]
Writes <dir>/points/<key>.json (one file per point, a cache for resuming), <dir>/points.csv (one row per simulated point, the
layout of the stored points.csv files) and <dir>/summary.json (sampled bounds and the per-point check). Processes are pinned
to the cores other than core 0. --slots DIR (default: the environment variable CPU_SLOTS; off when neither is given): a
directory holding a module cpu_slots whose class Slots shares the cores of the machine between several jobs (methods want,
take, give, close; attribute held); one slot is taken per point. Without it an ordinary process pool is used.
Other modules use compute, sweep, run_pool and slot_imap, the constants of the batches (SCREEN_LEVELS, STOCH_SCEN, ...) and
the hash helpers (mapping_sha, ...) of this module.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import os
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from occupancy import params as P                        # noqa: E402
from occupancy.network import Net                         # noqa: E402

STEADY_TOL = 0.02         # steady state: |mix shift| <= 2 % (saturated level; mean over the seeds)
STEADY_DRIFT_TOL = 0.05   # and |daily drift| <= 5 %
# parameters that decide the stops and sections of a network (the cache key of net_for)
NET_KEYS = ("frontage_gap", "frontage_min_len", "frontage_merge", "dock_stops", "section", "section_min_m", "section_gap_m", "block_m",
            "retrofit")

# ---------------------------------------------------------------- constants of the batches
SEEDS = tuple(range(101, 111))
N_MIX = 2000                     # task mix of the network bound: the first N_MIX tasks of the seed's task stream
# daily demands of external references: Okpo by the same conversion 339, the Hanwha system 350, HHI about 500 in 24 h
REFERENCE_DEMAND = (339, 350, 500)
DEMAND_PEAK = (678, 700)         # Okpo erection peak = regular x 2 (as the peak of PLANT_LEVELS): 2 x 339, 2 x 350
LOAD30 = {"load_s": 1800.0, "unload_s": 1800.0}
# required fleet with 30-min loading and unloading: every level of the required-fleet table (Okpo 10 levels, Yantai 5)
ANCHOR_LEVELS = {"yupu": tuple(sorted(set(P.PLANT_LEVELS) | set(REFERENCE_DEMAND) | set(DEMAND_PEAK))), "yantai": P.PLANT_LEVELS}
# daily moves of the two external anchors (fleets under the conserved mapping, 24 h): HHI about 500 a day (24 vehicles, 24 h;
# Park and Seo 2006), an unnamed Korean yard about 600 a day (about 30 vehicles; Shen et al. 2018)
ANCHOR_MOVES = (500, 600)
# screen (fleet gap against daily demand / system bound) at its low end. System bound (main flow mix, whole roads, handling
# 10 min) Okpo 1,736, Yantai 1,298 tasks/16 h (24 h: x 1.5): Okpo 50-250 -> 0.03-0.14 at 16 h, 0.02-0.10 at 24 h; Yantai
# 25-200 -> 0.02-0.15 at 16 h, 0.01-0.10 at 24 h; continues the lowest plant level (ratio from 0.12)
SCREEN_LEVELS = {"yupu": (50, 100, 150, 200, 250), "yantai": (25, 50, 100, 150, 200)}
# finer levels (a separate batch): the 10 % crossing of the first batch lies at Okpo 50-200, Yantai 100-200, between levels
# 0.02-0.04 apart (as ratios); three levels are inserted
SCREEN_MID_LEVELS = {"yupu": (75, 125, 175), "yantai": (75, 125, 175)}
# screen with intra-area moves: the low Okpo levels (both batches) plus the plant levels 300, 450, 500, 600: with the system
# bound at 3,168 tasks/16 h the 24-h ratio has no level between 350 (7.4 %) and 678 (14.3 %), and the 10 % crossing must lie
# between levels (339, 350, 678, 700 are in the required-fleet batch)
MAPPING_SCREEN_LEVELS = tuple(sorted(SCREEN_LEVELS["yupu"] + SCREEN_MID_LEVELS["yupu"] + (300, 450, 500, 600)))
MAPPING_VARS = ("open", "stop", "open_stop")       # variants of experiments/demand_mapping_var.py
# fleet points of the scan over the share of intra-area moves (Okpo plant demand, 16 h); 0 and 0.655 are the conserved
# mapping and the intra mapping
CONSERVED_FLEET_Q = (0.25, 0.5)
DOCK_KS = (20, 40, 60, 100, 150)          # fleets at which the dock-mouth thresholds are read (the columns of the dock table)
# screen: finer levels at demand ratios 2 %-6 % (tasks/16 h; Okpo bound 1,735.6 -> 3.5 %, 4.0 %, 4.9 %, Yantai 1,297.9 ->
# 2.7 %, 4.6 %, 5.4 %), not overlapping SCREEN_LEVELS
SCREEN_LOW_LEVELS = {"yupu": (60, 70, 85), "yantai": (35, 60, 70)}
# fleets at which the dock with one off-road unloading bay (8 days) is not steady at rho_10; all rho levels of these fleets are
# run for 16 days (as for the on-road side of the dock figure)
DOCK_BAY_16D_KS = {"yupu": (40, 60, 100, 150), "yantai": (60, 100, 150)}
# random handling and driving times (experiments/stochastic.py, Sim.draw; occupancy/ unchanged). Handling = nominal value h0 +
# lognormal, E[T] = 1.5 h0 (15 min), CV 0.3 / 0.6; the speed of each trip is multiplied by U[0.9, 1.0] (never faster than
# nominal, so the bound still holds). Variants: speed +-10 % (U[0.9, 1.1], checked against the bound at 1.1 x speed,
# bound_params) and booking at the 0.9 quantile (book_q). Deterministic mean case m15: handling 15 min, speed x 1/E[1/f]
# (driving time at its mean). Under random times full-path reservation uses late rebooking (reserve_rb; at nominal durations
# event by event equal to reserve). The deterministic h0 case is not recomputed (the runs of the earlier version serve).
STOCH_KS = (5, 10, 20, 30, 40, 60, 80, 100, 120, 150)
STOCH_SPD = 0.1 / math.log(10 / 9)          # f ~ U[0.9, 1]: E[1/f] = ln(10/9) / 0.1, mean driving time = nominal / STOCH_SPD


def stoch_params(cv, hi=1.0, q=None):
    return dict(mean=1.5, cv=cv, speed=[0.9, hi], book_q=q)


STOCH_SCEN = {"m15": dict(load_s=900.0, unload_s=900.0, v_empty=P.DEFAULTS["v_empty"] * STOCH_SPD,
                          v_loaded=P.DEFAULTS["v_loaded"] * STOCH_SPD),
              "cv3": dict(stoch=stoch_params(0.3)), "cv6": dict(stoch=stoch_params(0.6)),
              "cv6pm": dict(stoch=stoch_params(0.6, 1.1)), "cv6q": dict(stoch=stoch_params(0.6, q=0.9))}
# control rows: V1 = nominal handling, speed U[0.999, 1]; V2 = handling 15 min without variation (booked at the nominal
# 10 min), nominal speed; r15 = full-path reservation, handling 15 min, nominal speed
STOCH_VAR = {"v1": dict(stoch=dict(mean=1.0, cv=0.0, speed=[0.999, 1.0], book_q=None)),
             "v2": dict(stoch=dict(mean=1.5, cv=0.0, speed=[1.0, 1.0], book_q=None)), "r15": dict(load_s=900.0, unload_s=900.0)}
STOCH_RULES = ("free", "reserve_rb", "segment_safe")
STOCH_MODELS = {"m15": ("free", "reserve", "segment_safe"), "cv3": STOCH_RULES, "cv6": STOCH_RULES, "cv6pm": STOCH_RULES,
                "cv6q": ("reserve_rb",)}
# passable widths from the image check (per yard and map road: name, road width W, passable width G)
CLEAR_WIDTH = ROOT / "results" / "width_ban" / "clear_width.json"


def clear_widths():
    """{yard: {map road: passable width G}} (roads not listed: the road width); keys other than yards (about) are skipped."""
    raw = CLEAR_WIDTH.read_bytes()
    return {y: {r: float(v["G"]) for r, v in rs.items()} for y, rs in json.loads(raw).items() if y in ("yupu", "yantai")}


def bound_params(p):
    """Nominal parameters of the per-point check under random times: with a speed factor above 1 (the +-10 % variant) the bound
    is taken at the fastest speed; otherwise p itself (the nominal times of the random cases are lower limits)."""
    hi = (p.get("stoch") or {}).get("speed", [0, 1.0])[1]
    return dict(p, v_empty=p["v_empty"] * hi, v_loaded=p["v_loaded"] * hi) if hi > 1 else p


# ---------------------------------------------------------------- hashes of the modules a batch also depends on
def ordered_proj_sha():
    """Points of booking on projected occupancy also depend on experiments/ordered_proj.py; its SHA-256 goes into the
    configuration."""
    import ordered_proj
    return ordered_proj.code_sha256()


def ordered_sha():
    """Points of execution in booked order also depend on experiments/ordered.py and stochastic.py (how the random durations
    are drawn); their SHA-256 goes into the configuration."""
    import ordered
    return ordered.code_sha256()


def mapping_sha(var=False, *more):
    """Points under a demand mapping also depend on experiments/demand_mapping.py (the task stream; the variant batches also
    on demand_mapping_var.py), which is not in occupancy/; the SHA-256 goes into the configuration (gen_sha256). Saturation
    curves of the two placements of intra-area moves as the variant batches; 30-day batches add daily.py (more)."""
    names = (("demand_mapping_var.py", "demand_mapping.py") if var else ("demand_mapping.py",)) + more
    return hashlib.sha256(b"".join((ROOT / "experiments" / n).read_bytes() for n in names)).hexdigest()


def conserved_sha(daily=False):
    """Points under the conserved mapping also depend on experiments/conserved_mapping.py and demand_mapping.py (the task
    stream), 30-day batches also on daily.py; written into the configuration (gen_sha256)."""
    names = ("conserved_mapping.py", "demand_mapping.py") + (("daily.py",) if daily else ())
    return hashlib.sha256(b"".join((ROOT / "experiments" / n).read_bytes() for n in names)).hexdigest()


def stoch_sha():
    """Points under random times also depend on experiments/stochastic.py (random handling and driving times, late
    rebooking), which is not in occupancy/; the SHA-256 goes into the configuration."""
    return hashlib.sha256((ROOT / "experiments" / "stochastic.py").read_bytes()).hexdigest()


def load_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def scen_params(cfg, scen):
    p = dict(cfg["params"])
    p.update(cfg["scenarios"][scen])
    return p


def levels_of(cfg, scen):
    return cfg.get("levels", {}).get(scen, list(cfg["demand"]))


def ks_of(cfg, lvl):
    """Fleet sizes of a level (kmax limits a level; ks_level lists the fleet sizes of a level)."""
    only = cfg.get("ks_level", {}).get(lvl)
    return [K for K in cfg["ks"] if K <= cfg.get("kmax", {}).get(lvl, K) and (only is None or K in only)]


def point_keys(cfg):
    keys = []
    for scen in cfg["scenarios"]:
        if cfg.get("bounds", True):
            keys += [f"{scen}|bound|{s}" for s in cfg["seeds"]]
        for model in cfg["models"].get(scen, ("reserve",)):
            keys += [f"{scen}|{model}|{lvl}|{K}|{s}" for lvl in levels_of(cfg, scen) for K in ks_of(cfg, lvl) for s in cfg["seeds"]]
    return keys


def fname(key):
    return key.replace("|", "_") + ".json"


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


NETS = {}


def net_for(cfg, p):
    """(network, dock-mouth resources), cached by the parameters that decide stops and sections."""
    k = (cfg["network"], json.dumps(cfg.get("roles") or {}, sort_keys=True)) + tuple(json.dumps(p.get(x), sort_keys=True) for x in NET_KEYS)
    if k not in NETS:
        cls = Net
        if any(s.get("kind") == "door_stop" for s in p.get("retrofit") or []):    # door retrofit acting per door
            from intervention_net import DoorNet as cls                             # (experiments/intervention_net.py; else Net)
        net = cls(cfg["network"], p, cfg.get("roles"))
        NETS[k] = net, set().union(*net.near(P.DOCK_CLASS))
    return NETS[k]


def map_stream(net, seed, p):
    """Task stream of a demand mapping, for saturated levels and bound samples. The variants of where the intra-area moves
    are placed (intra_var) take demand_mapping_var.Stream; the conserved mapping takes conserved_mapping."""
    import demand_mapping
    import demand_mapping_var
    import conserved_mapping
    if p.get("demand_map") == "conserved":
        return conserved_mapping.stream(net, seed, p)
    return demand_mapping_var.Stream(net, seed, p) if p.get("intra_var") else demand_mapping.stream(net, seed, p)


def op3_sim(net, p, mode, K, seed, level_n, record=False, tasks=None):
    """reserve_op3: booking on projected occupancy with execution in booked order (ordered_proj.OPE2, proj=True), bookings and
    projection at mean durations (book="mean" of ordered: handling at its mean, speed at the speed of the mean driving time);
    actual durations and common random numbers as reserve_op2. Only the booked durations change, not the execution rule."""
    import ordered_proj
    assert mode == "reserve_op3"
    return ordered_proj.OPE2(net, dict(p, stoch=dict(p["stoch"], book="mean")), K, seed, level_n, tasks, proj=True)


def compute(cfg, key):
    """The result of one point (bound sample or simulation), without its run time."""
    from occupancy.bound import bounds, bounds_fixed
    from occupancy.sim import Sim
    from occupancy.tasks import TaskStream, crane_p6, crane_tasks
    parts = key.split("|")
    p = scen_params(cfg, parts[0])
    if p.get("width_ban"):              # ban on blocks wider than the passable width (experiments/width_ban.py; else below)
        import width_ban
        return width_ban.compute(cfg, key)
    if p.get("bay_spots"):              # unloading bay plus off-road waiting place at the dock (experiments/dock_spots.py; else below)
        import dock_spots
        return dock_spots.compute(cfg, key)
    if p.get("occ_rep"):                # ablation of the occupancy representation (experiments/ablation.py; under the conserved
        import ablation                 # mapping experiments/ablation_conserved.py; else below)
        import ablation_conserved
        return (ablation_conserved if p.get("demand_map") else ablation).compute(cfg, key)
    net, dock = net_for(cfg, p)
    if parts[1] == "bound":
        seed = int(parts[2])
        if p.get("crane_n"):        # crane pacing: crane rows; the cranes join the dock-mouth resources
            n6, local = crane_p6(net, p), dock | {("c", z) for z in net.by_cls[P.DOCK_CLASS]}
            pb = dict(p, flow_share={"P6": n6 / p["erect_peak"]})          # peak-day mix: P6 takes n6 / peak demand
            s, o = TaskStream(net, seed, pb), TaskStream(net, seed, p)     # o: the other flows (no P6 in the task stream)
            res = dict(key=key, scen=parts[0], seed=seed, **bounds(net, pb, [s.next(0.0) for _ in range(cfg["n_mix"])], local=local))
            fixed = crane_tasks(net, seed, dict(p, days=math.ceil(200 / n6)), n6)  # saturated: P6 fixed at n6 a day, other flows free
            res.update(bounds_fixed(net, p, [o.next(0.0) for _ in range(cfg["n_mix"])], (fixed, n6), local=local))
            return res
        s = map_stream(net, seed, p) if p.get("demand_map") else TaskStream(net, seed, p)     # task mix of a demand mapping
        return dict(key=key, scen=parts[0], seed=seed,
                    **bounds(net, bound_params(p), [s.next(0.0) for _ in range(cfg["n_mix"])], local=dock))
    scen, model, lvl, K, seed = parts[0], parts[1], parts[2], int(parts[3]), int(parts[4])
    if model == "reserve_op":                        # full-path reservation with execution in booked order (experiments/ordered.py)
        from ordered import yard_sim as Sim
    elif model == "reserve_op2":                     # booking on projected occupancy, execution in booked order (ordered_proj.py)
        from ordered_proj import yard_sim2 as Sim
    elif model == "reserve_op3":                     # the same, booking and projection at mean durations
        Sim = op3_sim
    elif model in ("reserve_tlw", "reserve_tlw_timer"):   # booking the tail / release by timer counted alike (tail_booking.py)
        from tail_booking import yard_sim as Sim
    elif p.get("stoch") or model == "reserve_rb":    # random handling and driving times, late rebooking (stochastic.py)
        from stochastic import Sim
    tasks = None
    if p.get("demand_map") and cfg["demand"][lvl] is not None:     # finite demand of a demand mapping
        import demand_mapping
        import demand_mapping_var       # variants of where the intra-area moves are placed (intra_var)
        import conserved_mapping        # the conserved inter-area mapping (demand_map = conserved)
        mod = conserved_mapping if p["demand_map"] == "conserved" else \
            (demand_mapping_var if p.get("intra_var") else demand_mapping)
        tasks = mod.demand_tasks(net, seed, cfg["demand"][lvl], p)
    elif p.get("daily_stats"):          # the daily readings need the released tasks; these equal the ones Sim generates itself
        from occupancy.tasks import demand_tasks
        tasks = demand_tasks(net, seed, cfg["demand"][lvl], p)
    sim = Sim(net, p, model, K, seed, cfg["demand"][lvl], tasks=tasks)
    if p.get("demand_map") and cfg["demand"][lvl] is None:         # saturated level: idle vehicles draw from the mapping
        sim.stream = map_stream(net, seed, p)
    sim.run()
    res = dict(key=key, scen=scen, model=model, level=lvl, K=K, seed=seed, **sim.metrics())
    if p.get("daily_stats"):            # released, completed, end-of-day backlog, flow times, waits per day (experiments/daily.py)
        import daily
        res["daily"] = daily.daily(sim, p, tasks)
    if model != "free":     # per-point check: throughput <= T1' of the tasks completed in the statistics window
        pb = bound_params(p)        # random times: at the nominal times (a lower limit; at the fastest speed under +-10 % speeds)
        res["T1_tight_mix"] = t1_window(net, pb, sim, sim.H)
        if cfg["demand"][lvl] is None:
            res["steady"] = steady(net, pb, sim, res["T1_tight_mix"])
    return res


def t1_window(net, p, sim, t_end, tasks=None):
    """T1' (occupancy.bound.bounds_window) of the statistics window [end of warm-up, t_end]: tasks are by default the vehicle
    tasks completed in the window; lifts completed in the window hold the cranes in absolute terms."""
    from occupancy.bound import bounds_window
    win = lambda rs: [r for r in rs if sim.t0m <= r["done"] <= t_end]
    return bounds_window(net, p, win(sim.done) if tasks is None else tasks, win(sim.lifts), t_end - sim.t0m)


def drift(days):
    """Daily drift of throughput: least-squares slope x (days - 1) / mean, the fitted change across the window."""
    n = len(days)
    if n < 2 or not sum(days):
        return 0.0
    mx, my = (n - 1) / 2, sum(days) / n
    return sum((x - mx) * (y - my) for x, y in enumerate(days)) / sum((x - mx) ** 2 for x in range(n)) * (n - 1) / my


def steady(net, p, sim, t1_done_H):
    """Steady-state readings of a saturated point, for windows [warm-up, E] with E = 4 days and the run length:
    mix shift = T1'(tasks completed in the window) / T1'(tasks dispatched in the window) - 1 (about 0 in steady state; > 0
    when queues grow at a binding resource and the hard tasks stay queued); long-run estimate = throughput / (1 + shift);
    daily drift of throughput (drift()). Both must be within their tolerances for a steady reading."""
    disp = sim.dispatched()
    out = {}
    for E in sorted({P.DEFAULTS["days"], p["days"]}):
        if E > p["days"]:
            continue
        t_end = E * p["day_s"]
        done = [r for r in sim.done if sim.t0m <= r["done"] <= t_end]
        dsp = [r for r in disp if sim.t0m <= r["dispatch"] <= t_end]
        a = t1_done_H if E == p["days"] else t1_window(net, p, sim, t_end)
        b = t1_window(net, p, sim, t_end, dsp)
        thr = len(done) * p["day_s"] / (t_end - sim.t0m)
        shift = a / b - 1 if a and b else None
        per = collections.Counter(min(int(r["done"] // p["day_s"]), E - 1) for r in done)
        out[str(E)] = dict(thr=thr, T1t_done=a, T1t_disp=b, mix_shift=shift, longrun=thr / (1 + shift) if shift is not None else None,
                           drift=drift([per[d] for d in range(p["warmup_days"], E)]))
    return out


def load_points(out):
    """(derived simulated points, {(scenario, seed): bound sample}) of <out>/points."""
    pts = [json.loads(f.read_text(encoding="utf-8")) for f in (out / "points").glob("*.json")]
    return [derive(x) for x in pts if "network" not in x], {(x["scen"], x["seed"]): x for x in pts if "network" in x}


def derive(r):
    """Scalars of a point for points.csv: P6 flow times, dock-mouth readings (ring 0 = the dock stop roads, all = both rings),
    the most occupied resources, crane, lift, starvation, safe-release, cycle and late-rebooking readings."""
    nan = float("nan")
    f6 = r.get("flows", {}).get("P6", {})
    d0, da = r["dock"]["ring0"], r["dock"]["all"]
    st = r.get("steady") or {}
    last, w4 = (st[max(st, key=int)] if st else {}), st.get("4", {})
    cr = r.get("crane") or {}
    lf = r.get("lift") or {}
    sv, vd, sf, cy = (r.get(k) or {} for k in ("starve", "victims_diag", "safe", "cycle"))
    rb = r.get("rebook") or {}         # late rebooking (reserve_rb)
    return dict(r, rb_late=rb.get("late_per_day"), rb_rebook=rb.get("rebook_per_day"), rb_late_mean_s=rb.get("late_mean_s"),
                starve=sv.get("starve_max"), starve_blk=sv.get("blocked_max"), starve_road=sv.get("blocked_road_max"),
                victims_dock=vd.get("at_dock_per_day"), victims_p6=vd.get("p6_loaded_per_day"), frozen_end=vd.get("frozen_end"),
                safe_wait_n=sf.get("wait_n_per_day"), safe_wait_min=sf.get("wait_min_per_day"), safe_banker=sf.get("banker"),
                cyc_empty=cy.get("empty"), cyc_loaded=cy.get("loaded"), cyc_wait=cy.get("wait"), cyc_total=cy.get("total"),
                lift_rel=lf.get("released_per_day"), lift_n=lf.get("n_per_day"), lift_wait=lf.get("wait_mean"),
                lift_flow_mean=lf.get("flow_mean"), lift_flow_p95=lf.get("flow_p95"), lift_queue_end=lf.get("queue_end"),
                crane_waits=cr.get("waits_per_day"), crane_wait_min=cr.get("wait_min_per_day"), crane_wait_attr=cr.get("wait_attr_min_per_day"),
                crane_util=cr.get("util_max"), T1_tight_disp=last.get("T1t_disp"), mix_shift=last.get("mix_shift"), longrun_est=last.get("longrun"),
                drift=last.get("drift"), drift_w4=w4.get("drift"), thr_w4=w4.get("thr"), mix_shift_w4=w4.get("mix_shift"), P6_per_day=f6.get("n_per_day", 0.0), P6_flow_mean=f6.get("flow_mean", nan), P6_flow_p95=f6.get("flow_p95", nan),
                dock0_busy=d0["busy_max"], busy_top1=r["busy_top"][0]["value"] if r["busy_top"] else nan,
                busy_top1_name=r["busy_top"][0]["name"] if r["busy_top"] else "",
                dock0_util=d0["util_max"], dock0_wait=d0["wait_min_per_day"], dock_wait=da["wait_min_per_day"],
                dock_util_mean=da["util_mean"], dock_dl=da["deadlocks_per_day"], dock_near=da["near_per_day"],
                dock0_dl=d0["deadlocks_per_day"], util_top1=r["util_top"][0]["value"] if r["util_top"] else nan,
                util_top1_name=r["util_top"][0]["name"] if r["util_top"] else "")


# columns of points.csv (stored files of earlier runs have a prefix of these)
FIELDS = ["scen", "model", "level", "K", "seed", "throughput", "completed", "flow_mean", "flow_p95", "wait_mean",
          "wait_p95", "deadlocks", "deadlocks_per_day", "victims", "near_deadlocks", "near_per_day", "backlog",
          "waiting_end", "T1_tight_mix", "P6_per_day", "P6_flow_mean", "P6_flow_p95", "dock0_util", "dock0_busy",
          "dock0_wait", "dock_wait", "dock_util_mean", "dock_dl", "dock0_dl", "dock_near", "util_top1", "util_top1_name",
          "busy_top1", "busy_top1_name",
          "wait_attr_min_per_day", "deadlock_stuck_hist", "deadlock_core_hist", "deadlock_times_h", "secs",
          # steady-state readings of saturated points
          "T1_tight_disp", "mix_shift", "longrun_est", "thr_w4", "mix_shift_w4", "thr_by_day",
          # cranes (crane pacing): starts beyond the number of cranes (a self-check, 0 for the occupancy rules), waiting
          # attributed to the cranes (min/day), utilisation; daily drift over the run and over four days
          "crane_waits", "crane_wait_min", "crane_wait_attr", "crane_util", "drift", "drift_w4",
          # lifts by a gantry crane inside its span (no vehicle): released and completed per day, waits and flow times (min),
          # queue at the end
          "lift_rel", "lift_n", "lift_wait", "lift_flow_mean", "lift_flow_p95", "lift_queue_end",
          # crane starvation and the part blocked by the roads (the larger of the two docks); recovered vehicles of the
          # reference rule holding a dock stop / carrying P6 to a dock (per day), vehicles stuck at the end without recovery;
          # refusals of safe release (per day, min/day) and full reductions; cycle parts (min/task, split by the nominal
          # handling durations and speeds: under random times the handling and driving beyond the nominal values count
          # into cyc_wait, which is then no waiting reading)
          "starve", "starve_blk", "victims_dock", "victims_p6", "frozen_end", "safe_wait_n", "safe_wait_min", "safe_banker",
          "cyc_empty", "cyc_loaded", "cyc_wait", "cyc_total",
          # starvation while a vehicle waits for a road
          "starve_road",
          # full-path reservation with late rebooking (reserve_rb): late arrivals (per day), rebookings (per day), mean lateness (s)
          "rb_late", "rb_rebook", "rb_late_mean_s"]


def write_points(f, sims, fields=FIELDS):
    """points.csv layout (also used to compare reruns with stored rows). sims are results after derive(); fields default to
    FIELDS (the header of an existing points.csv when points are added to it)."""
    w = csv.DictWriter(f, fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(dict(r, deadlock_times_h=" ".join(map(str, r["deadlock_times_h"])),
                     deadlock_stuck_hist=json.dumps(r["deadlock_stuck_hist"]),
                     deadlock_core_hist=json.dumps(r["deadlock_core_hist"]),
                     thr_by_day=" ".join(map(str, r.get("thr_by_day", [])))) for r in sims)


def point_row(r, columns):
    """One derived point as the string cells of points.csv, restricted to `columns` (the header of a stored file)."""
    import io
    buf = io.StringIO()
    write_points(buf, [r])
    rd = list(csv.DictReader(io.StringIO(buf.getvalue())))[0]
    return {c: rd.get(c, "") for c in columns}


def pin(cores):
    try:
        import psutil
        psutil.Process().cpu_affinity(cores)
    except Exception:     # without psutil (or where affinity is not supported) the processes are not pinned
        pass


CFG, OUT = None, None


def init(cfg, out, cores):
    global CFG, OUT
    pin(cores)
    CFG, OUT = cfg, Path(out) if out else None


def run_point(key):
    t = time.time()
    try:
        res = compute(CFG, key)
        res["secs"] = round(time.time() - t, 2)
        if OUT:
            write_json(OUT / "points" / (key.replace("|", "_") + ".json"), res)
        return key, res, None
    except Exception:
        return key, None, traceback.format_exc(limit=4)


class Progress:
    """Progress of a run in a small JSON file (path): counts of points, processes, state, the last lines; log() also prints
    the line."""

    def __init__(self, path, total, done, procs, title, task="run"):
        self.path, self.lock = Path(path), threading.Lock()
        self.d = dict(task=task, title=title, total=total, done=done, reused=done, failed=0, processes=procs,
                      started=time.time(), updated=time.time(), state="running", note="", recent=[])
        self.write()

    def write(self, **kw):
        with self.lock:
            self.d.update(kw, updated=time.time())
            try:
                write_json(self.path, self.d)
            except PermissionError:      # the progress is for display only: if the file is busy, the next write follows
                pass

    def log(self, line, **kw):
        print(line, flush=True)
        with self.lock:
            self.d["recent"] = (self.d["recent"] + [line])[-8:]
        self.write(**kw)

    def heartbeat(self, stop, every=30.0):
        while not stop.wait(every):
            self.write()


def describe(key, res):
    if "network" in res:
        return f"{key}: T1 {res['network']:.1f}/day, T1' {res['network_tight']:.1f}/day ({res['secs']} s)"
    return f"{key}: throughput {res['throughput']:.1f}/day, deadlocks {res['deadlocks']} ({res['secs']} s)"


def pool_run(cfg, out, procs, todo, log=print, slots_dir=None, max_tasks=None, deadline=None):
    """run_pool that returns ({key: result}, number of failed points, stopped at the deadline with points left); no new
    points are started after `deadline` (a time stamp), the points under way are finished."""
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    prog = log if isinstance(log, Progress) else None
    say = prog.log if prog else (lambda line, **kw: log(line))
    slots, sd = None, slots_dir or os.environ.get("CPU_SLOTS")
    if sd:
        sys.path.insert(0, sd)
        from cpu_slots import Slots
        slots = Slots(f"{cfg.get('task', 'run')}-{os.getpid()}")
    stop = threading.Event()
    if prog:
        threading.Thread(target=prog.heartbeat, args=(stop,), daemon=True).start()
    cores = list(range(1, os.cpu_count())) or [0]
    pin(cores)
    res, failed, stopped, n, todo, live, shown = {}, 0, False, len(todo), list(todo), {}, None
    cap = procs if slots else 2 * procs
    with ProcessPoolExecutor(procs, initializer=init, initargs=(cfg, str(out) if out else None, cores),
                             **({"max_tasks_per_child": max_tasks} if max_tasks else {})) as ex:
        while todo or live:
            stopped = stopped or (deadline is not None and time.time() > deadline)
            if slots:
                slots.want(0 if stopped else len(todo) + len(live))
            while todo and not stopped and len(live) < cap and (slots is None or slots.take()):
                k = todo.pop(0)
                live[ex.submit(run_point, k)] = k
            if prog and (len(slots.held) if slots else procs) != shown:
                shown = len(slots.held) if slots else procs
                prog.write(processes=shown)
            if not live:
                if stopped:
                    break
                time.sleep(0.5)            # no slot free: try again shortly
                continue
            done, _ = wait(live, timeout=0.5, return_when=FIRST_COMPLETED)
            for f in done:
                k = live.pop(f)
                if slots:
                    slots.give()
                try:
                    key, r, err = f.result()
                except Exception as e:     # a worker process died
                    key, r, err = k, None, repr(e)
                if err:
                    failed += 1
                    say(f"failed {key}: {err.strip().splitlines()[-1]}", failed=failed)
                    if out:
                        with (Path(out) / "failures.log").open("a", encoding="utf-8") as fl:
                            fl.write(f"{key}\n{err}\n")
                else:
                    res[key] = r
                    say(f"{len(res)}/{n} " + describe(key, r), **({"done": prog.d["done"] + 1} if prog else {}))
    if slots:
        slots.close()
    stop.set()
    return res, failed, stopped and bool(todo)


def run_pool(cfg, out, procs, todo, log=print, slots_dir=None, max_tasks=None):
    """Compute the points `todo` in at most `procs` processes (started as needed) pinned to the cores other than core 0;
    returns {key: result} and writes each point to <out>/points/ (out None: nothing written). log: a print-like function for
    one line per point, or a Progress (lines, counts, processes, heartbeat). slots_dir (default: the environment variable
    CPU_SLOTS; off when neither is given): one machine-wide CPU slot is taken per point and given back when it is done, so
    that several jobs share the cores; without it at most 2 x procs points are under way. max_tasks: a worker process exits
    after this many points and a new one takes over (frees the network cache and the memory of the simulations). Failed
    points are logged (and appended to <out>/failures.log); RuntimeError at the end if any failed."""
    res, failed, _ = pool_run(cfg, out, procs, todo, log, slots_dir, max_tasks)
    if failed:
        raise RuntimeError(f"{failed} points failed")
    return res


def slot_imap(func, items, procs, initializer=None, initargs=(), slots_dir=None, tag="run", ordered=False):
    """Replacement of mp.Pool(procs, initializer, initargs).imap_unordered(func, items) for modules with their own process
    pool: with machine-wide CPU slots (slots_dir or the environment variable CPU_SLOTS) one slot is taken per item and given
    back when it is done; without them at most procs items are under way. Returns a generator of the results of func in the
    order of completion (ordered: a list in the order of items, once all are done, in place of pool.map). To stop early the
    caller closes the generator: no new items are started, the items under way are finished and the slots given back."""
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    def gen(index):
        slots, sd = None, slots_dir or os.environ.get("CPU_SLOTS")
        if sd:
            sys.path.insert(0, sd)
            from cpu_slots import Slots
            slots = Slots(f"{tag}-{os.getpid()}")
        todo, live = list(enumerate(items)), {}
        try:
            with ProcessPoolExecutor(procs, initializer=initializer, initargs=initargs) as ex:
                while todo or live:
                    if slots:
                        slots.want(len(todo) + len(live))
                    while todo and len(live) < procs and (slots is None or slots.take()):
                        i, x = todo.pop(0)
                        live[ex.submit(func, x)] = i
                    if not live:
                        time.sleep(0.5)            # no slot free: try again shortly
                        continue
                    done, _ = wait(live, timeout=0.5, return_when=FIRST_COMPLETED)
                    for f in done:
                        i = live.pop(f)
                        if slots:
                            slots.give()
                        yield (i, f.result()) if index else f.result()
        finally:
            if slots:
                slots.close()
    if ordered:
        return [r for _, r in sorted(gen(True), key=lambda x: x[0])]
    return gen(False)


def sweep(cfg, out, procs, deadline=None, slots_dir=None):
    """Compute every point of the configuration that is not yet in <out>/points (a resumed run adds only the missing ones),
    with the progress in <out>/progress.json; no new points after `deadline` (a time stamp), the points under way are
    finished. Returns (failed points, stopped at the deadline with points left)."""
    (out / "points").mkdir(parents=True, exist_ok=True)
    keys = point_keys(cfg)
    todo = [k for k in keys if not (out / "points" / fname(k)).exists()]
    prog = Progress(out / "progress.json", len(keys), len(keys) - len(todo), procs, cfg.get("title", ""), cfg.get("task", "run"))
    _, failed, stopped = pool_run(cfg, out, procs, todo, prog, slots_dir, deadline=deadline)
    if stopped:
        prog.write(state="paused", note="stopped at the deadline; the same command resumes")
    else:
        prog.write(state="failed" if failed else "done", note=f"{failed} failed" if failed else "")
    return failed, stopped


def stats(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return dict(mean=sum(xs) / len(xs), lo=min(xs), hi=max(xs)) if xs else dict(mean=float("nan"), lo=float("nan"), hi=float("nan"))


def aggregate(cfg, out):
    """points.csv and summary.json (sampled bounds per scenario and seed; per-point check against T1' of the window)."""
    pts = [json.loads(f.read_text(encoding="utf-8")) for f in (out / "points").glob("*.json")]
    sims = sorted((derive(x) for x in pts if "network" not in x), key=lambda x: (x["scen"], x["model"], x["level"], x["K"], x["seed"]))
    bnds = {f"{x['scen']}|{x['seed']}": {k: v for k, v in x.items() if k not in ("key", "scen", "seed")} for x in pts if "network" in x}
    with open(out / "points.csv", "w", newline="", encoding="utf-8-sig") as f:
        write_points(f, sims)
    mix = [r for r in sims if r.get("T1_tight_mix")]
    check = dict(points=len(mix), max_ratio=max((r["throughput"] / r["T1_tight_mix"] for r in mix), default=None),
                 over_strict=sum(r["throughput"] > r["T1_tight_mix"] for r in mix))
    write_json(out / "summary.json", dict(findings=dict(T1_tight_check=check), bounds=bnds))
    return check


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--only", nargs="*", default=None, help="compute only the points whose key starts with one of these")
    ap.add_argument("--slots", default=os.environ.get("CPU_SLOTS"), help="directory of machine-wide CPU slots (cpu_slots.Slots); "
                                                                         "default: environment variable CPU_SLOTS, else off")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = Path(a.out)
    (out / "points").mkdir(parents=True, exist_ok=True)
    keys = [k for k in point_keys(cfg) if a.only is None or k.startswith(tuple(a.only))]
    todo = [k for k in keys if not (out / "points" / (k.replace("|", "_") + ".json")).exists()]
    print(f"{len(todo)} of {len(keys)} points to compute in {out}", flush=True)
    if todo:
        run_pool(cfg, out, min(a.procs, len(todo)), todo, slots_dir=a.slots)
    print("per-point check:", aggregate(cfg, out))


if __name__ == "__main__":
    main()

"""Batch runner: simulated points (fleet size vs throughput under the traffic rules) and sampled network bounds of one batch
configuration, as stored with every run in results/runs/<run>/config.json.

A configuration names the network (data/networks/<yard>/<revision>), the base parameters, the scenarios (parameter
overrides), the traffic rules per scenario (models), the demand levels (tasks a day; None = saturated, an endless supply of
tasks), the fleet sizes K, the seeds and the size of the task samples of the bound (n_mix). A point is
"<scenario>|<rule>|<level>|<K>|<seed>"; a bound sample is "<scenario>|bound|<seed>".

Each simulated point of an occupancy rule is checked against the network bound T1' of the tasks completed in its own
statistics window (T1_tight_mix, occupancy.bound.bounds_window: lifts by a gantry crane in the same window hold the cranes in
absolute terms, crane rows with the window edge term); the throughput must not exceed it. Saturated points also get the
steady-state readings (mix shift, daily drift).

Run:   python experiments/runner.py results/runs/<run>/config.json --out <dir> [--procs N] [--only KEY_PREFIX ...]
Writes <dir>/points/<key>.json (one file per point, a cache for resuming), <dir>/points.csv (one row per simulated point, the
layout of the stored points.csv files) and <dir>/summary.json (sampled bounds and the per-point check). Processes are pinned
to the cores other than core 0.
"""
import argparse
import collections
import csv
import json
import math
import multiprocessing as mp
import os
import sys
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


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


NETS = {}


def net_for(cfg, p):
    """(network, dock-mouth resources), cached by the parameters that decide stops and sections."""
    k = (cfg["network"], json.dumps(cfg.get("roles") or {}, sort_keys=True)) + tuple(json.dumps(p.get(x), sort_keys=True) for x in NET_KEYS)
    if k not in NETS:
        net = Net(cfg["network"], p, cfg.get("roles"))
        NETS[k] = net, set().union(*net.near(P.DOCK_CLASS))
    return NETS[k]


def compute(cfg, key):
    """The result of one point (bound sample or simulation), without its run time."""
    from occupancy.bound import bounds, bounds_fixed
    from occupancy.sim import Sim
    from occupancy.tasks import TaskStream, crane_p6, crane_tasks
    parts = key.split("|")
    p = scen_params(cfg, parts[0])
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
        s = TaskStream(net, seed, p)
        return dict(key=key, scen=parts[0], seed=seed, **bounds(net, p, [s.next(0.0) for _ in range(cfg["n_mix"])], local=dock))
    scen, model, lvl, K, seed = parts[0], parts[1], parts[2], int(parts[3]), int(parts[4])
    sim = Sim(net, p, model, K, seed, cfg["demand"][lvl]).run()
    res = dict(key=key, scen=scen, model=model, level=lvl, K=K, seed=seed, **sim.metrics())
    if model != "free":     # per-point check: throughput <= T1' of the tasks completed in the statistics window
        res["T1_tight_mix"] = t1_window(net, p, sim, sim.H)
        if cfg["demand"][lvl] is None:
            res["steady"] = steady(net, p, sim, res["T1_tight_mix"])
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


def derive(r):
    """Scalars of a point for points.csv: P6 flow times, dock-mouth readings (ring 0 = the dock stop roads, all = both rings),
    the most occupied resources, crane, lift, starvation, safe-release and cycle readings."""
    nan = float("nan")
    f6 = r.get("flows", {}).get("P6", {})
    d0, da = r["dock"]["ring0"], r["dock"]["all"]
    st = r.get("steady") or {}
    last, w4 = (st[max(st, key=int)] if st else {}), st.get("4", {})
    cr = r.get("crane") or {}
    lf = r.get("lift") or {}
    sv, vd, sf, cy = (r.get(k) or {} for k in ("starve", "victims_diag", "safe", "cycle"))
    return dict(r, starve=sv.get("starve_max"), starve_blk=sv.get("blocked_max"), starve_road=sv.get("blocked_road_max"),
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
          # refusals of safe release (per day, min/day) and full reductions; cycle parts (min/task)
          "starve", "starve_blk", "victims_dock", "victims_p6", "frozen_end", "safe_wait_n", "safe_wait_min", "safe_banker",
          "cyc_empty", "cyc_loaded", "cyc_wait", "cyc_total",
          # starvation while a vehicle waits for a road
          "starve_road"]


def write_points(f, sims):
    """points.csv layout (also used to compare reruns with stored rows). sims are results after derive()."""
    w = csv.DictWriter(f, FIELDS, extrasaction="ignore")
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


def run_pool(cfg, out, procs, todo, log=print):
    """Compute the points `todo` in `procs` processes pinned to the cores other than core 0; returns {key: result}."""
    cores = list(range(1, os.cpu_count())) or [0]
    pin(cores)
    res, failed = {}, 0
    with mp.Pool(procs, initializer=init, initargs=(cfg, str(out) if out else None, cores)) as pool:
        for i, (key, r, err) in enumerate(pool.imap_unordered(run_point, todo), 1):
            if err:
                failed += 1
                log(f"failed {key}: {err.strip().splitlines()[-1]}")
            else:
                res[key] = r
                log(f"{i}/{len(todo)} {key}: " + (f"T1' {r['network_tight']:.1f}/day" if "network_tight" in r else
                                                     f"throughput {r['throughput']:.1f}/day") + f" ({r['secs']} s)")
    if failed:
        raise RuntimeError(f"{failed} points failed")
    return res


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
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = Path(a.out)
    (out / "points").mkdir(parents=True, exist_ok=True)
    keys = [k for k in point_keys(cfg) if a.only is None or k.startswith(tuple(a.only))]
    todo = [k for k in keys if not (out / "points" / (k.replace("|", "_") + ".json")).exists()]
    print(f"{len(todo)} of {len(keys)} points to compute in {out}", flush=True)
    if todo:
        run_pool(cfg, out, min(a.procs, len(todo)), todo)
    print("per-point check:", aggregate(cfg, out))


if __name__ == "__main__":
    main()

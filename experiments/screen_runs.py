"""Reruns for the screen of demand over the bound (the screen in two levels: demand ratio and chain load). occupancy/ is not
changed: the rules and code are those of the stored batches.
  seeds  Okpo, 16 h, six levels at demand ratios of 2.9-5.8 % (50, 60, 70, 75, 85, 100 tasks a day) x three road widths x
         free flow and full-path reservation, seeds 101-130, the full grid K = 1-9 (stored configuration:
         results/runs/okpo_screen_30seeds/config.json; the points of seeds 101-110 equal those of the low screen levels
         exactly). runner.sweep point by point (with the per-point bound as usual), points.csv as written by runner.
  waits  the waiting of full-path reservation split by resource: both yards, 16 h, the low screen levels (Okpo 100, 150,
         200, 250; Yantai 75, 100, 125, 150, 200 tasks a day) x the required fleet K_res of the screen table x 10 seeds,
         rerun with the current code (throughput equal to the stored points). The subclass WaitSim records at each dispatch
         the increment of wait_by, i.e. the resources and seconds by which the booking moves the departure later (their sum =
         the wait of the task, t2 - t0 - empty run - loading), by position on the chain (empty leg, origin stop, loaded leg,
         destination stop, both legs, crane); also the number of times each resource is passed and pushes a task later, and
         per task the dispatch time (since the release of its batch), the number of tasks dispatched at the same time, the
         wait and the number of pushes -> <out>/parts/.
  wide   every Okpo road with a mapped width of at most 15 m widened to 18 m (the widen change of the road-change menu,
         occupancy/ unchanged; the list of roads is in the configuration), 100 and 200 tasks a day x the same K_res and seeds
         as waits (paired), waiting split by resource; also the system bound and per-resource quantities after widening
         (chain_load.compute) -> <out>/parts/, chain_wide.json. The widened roads include roads along docks, whose dock stops
         move with the width (frontage rule on the dock outline); the derived graphs of data/networks do not hold the
         outlines, so on them occupancy.network stops with NotImplementedError. This part shows how the stored readings were
         computed on the original map.
  intra  intra-area mapping, 16 h, 150 tasks a day with 30 seeds (stored configuration:
         results/runs/okpo_intra_screen_30seeds/config.json, K = 1-10) -> points.csv.

Run:   python experiments/screen_runs.py --part seeds|intra --config <run config.json> --out <dir> [--procs N] [--slots DIR]
       python experiments/screen_runs.py --part waits|wide --screen <screen cells> --wait-configs yupu=<config.json>
                yantai=<config.json> --out <dir> [--sensitivity <bound-sensitivity table>] [--procs N] [--slots DIR]
--screen: the screen cells with yard, day_h, demand, K_res (results/fleet/screen.csv). --wait-configs: the run configurations
whose scenario d16 the wait readings rerun (the batches of the screen cells: results/runs/okpo_screen_low/config.json and
results/runs/yantai_screen_low/config.json have the same d16 parameters). --sensitivity (wide only): the reference
lambda^S of chain_load (results/bounds/sensitivity.csv). --max-minutes (seeds): no new points after that time; exit code 3,
the same command resumes.
"""
import argparse
import collections
import csv
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
csv.field_size_limit(1 << 30)
import runner as R                           # noqa: E402

WAIT_LEVELS = {"yupu": (100, 150, 200, 250), "yantai": (75, 100, 125, 150, 200)}


# ---------------------------------------------------------------- batches from stored configurations (seeds, intra)
def run_batch(a, deadline=None):
    """The batch of the stored configuration a.config with runner.sweep: <out>/points/, points.csv."""
    cfg = R.load_config(a.config)
    out = Path(a.out)
    if (out / "config.json").exists() and json.loads((out / "config.json").read_text(encoding="utf-8")) != cfg:
        sys.exit(f"refusing to reuse {out}: its config.json differs from the configuration")
    out.mkdir(parents=True, exist_ok=True)
    R.write_json(out / "config.json", cfg)
    print(f"run directory: {out}", flush=True)
    failed, stopped = R.sweep(cfg, out, a.procs, deadline, a.slots)
    if stopped:
        sys.exit(3)
    sims = sorted(R.load_points(out)[0], key=lambda x: (x["scen"], x["model"], x["level"], x["K"], x["seed"]))
    with open(out / "points.csv", "w", newline="", encoding="utf-8-sig") as f:
        R.write_points(f, sims)
    print(f"done; {len(sims)} points, {failed} failed", flush=True)


def run_seeds(a):
    run_batch(a, time.time() + 60 * a.max_minutes if a.max_minutes else None)


def run_intra(a):
    run_batch(a)


# ---------------------------------------------------------------- waiting split by resource
def wait_sim():
    from occupancy.sim import Sim

    class WaitSim(Sim):
        """Read-only bookkeeping: the increment of wait_by around plan (the full-path reservation of a task) is the split of
        the task's wait by resource (tasks dispatched in the statistics window). No dispatch decision changes."""

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.rec = []

        def plan(self, v, task, t0):
            so, sd, _, _, TE, _, WE, WL, *_ = self.windows(v, task)
            before = dict(self.wait_by)
            super().plan(v, task, t0)
            if self.in_win(t0):
                delta = {r: x - before.get(r, 0.0) for r, x in self.wait_by.items() if x - before.get(r, 0.0) > 1e-9}
                self.rec.append(dict(t0=t0, wait=task["plan"]["t2"] - t0 - TE - self.p["load_s"], delta=delta, so=so["res"], sd=sd["res"],
                                     E={r for r, *_ in WE}, L={r for r, *_ in WL}))
    return WaitSim


def leg(r, x):
    if r[0] == "c":
        return "crane"
    if r == x["so"]:
        return "origin"
    if r == x["sd"]:
        return "dest"
    return {(True, True): "both", (True, False): "empty", (False, True): "loaded"}.get((r in x["E"], r in x["L"]), "other")


def rkey(r):
    return "|".join(map(str, r))


def wait_point(job):
    """job = (yard, tasks a day, K, seed, run configuration, extra scenario parameters); the scenario is d16 of that
    configuration, the extra parameters (such as the widening) on top of it."""
    yard, n, K, seed, d, extra = job
    cfg = json.loads(Path(d).read_text(encoding="utf-8"))
    p = dict(R.scen_params(cfg, "d16"), **extra)
    net, _ = R.net_for(cfg, p)
    t = time.time()
    sim = wait_sim()(net, p, "reserve", K, seed, n)
    sim.run()
    m = sim.metrics()
    by_res, pushed, visits, by_leg = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    # tasks dispatched at the same time (idle vehicles book together when a batch is released)
    same_t0 = collections.Counter(x["t0"] for x in sim.rec)
    batch = p["day_s"] / p["batches"]
    tasks = [[round(x["t0"] % batch, 1), same_t0[x["t0"]], round(x["wait"], 1), len(x["delta"]), sum(r[0] == "r" for r in x["delta"]),
              round(sum(s for r, s in x["delta"].items() if r[0] == "r"), 1)] for x in sim.rec]
    for x in sim.rec:
        for r in x["E"] | x["L"] | {x["so"], x["sd"]}:
            visits[rkey(r)] += 1
        for r, s in x["delta"].items():
            by_res[rkey(r)] += s
            pushed[rkey(r)] += 1
            by_leg[leg(r, x)] += s
    return dict(key=f"{yard}|{n}|{K}|{seed}", yard=yard, n=n, K=K, seed=seed, dir=d, throughput=m["throughput"], completed=m["completed"],
                T1_tight_mix=R.t1_window(net, R.bound_params(p), sim, sim.H), cycle=m.get("cycle"),
                n_tasks=len(sim.rec), wait_s=sum(x["wait"] for x in sim.rec), attr_s=sum(sum(x["delta"].values()) for x in sim.rec),
                waited=sum(x["wait"] > 1e-9 for x in sim.rec), by_leg=dict(by_leg), by_res=dict(by_res), pushed=dict(pushed), visits=dict(visits),
                tasks=tasks, tasks_cols=["since_batch_s", "same_t0", "wait_s", "pushes", "road_pushes", "road_wait_s"], secs=round(time.time() - t, 1))


def wait_jobs(screen, configs, yards=WAIT_LEVELS, extra=None):
    """Jobs of the wait readings: K_res of the screen table (16 h) and the run configuration of the yard (configs)."""
    rows = {(r["yard"], int(r["day_h"]), int(r["demand"])): r for r in csv.DictReader(open(screen, encoding="utf-8-sig"))}
    return [(y, n, int(rows[(y, 16, n)]["K_res"]), s, str(configs[y]), extra or {})
            for y, ns in yards.items() for n in ns for s in R.SEEDS]


def wide_roads():
    """The Okpo roads with a mapped width of at most 15 m (widened to 18 m; with the 9-m dock road road161)."""
    from occupancy import params as P
    from occupancy.network import Net
    Y = P.YARDS["yupu"]
    net = Net(Y["network"], P.DEFAULTS, Y["roles"])
    return sorted(rid for rid, r in net.road.items() if r["W"] is not None and r["W"] <= 15.0)


def wide_params():
    """Widening by the existing road-change menu (retrofit widen: only the mapped road widths change, the end junctions
    follow the size rule of the map); occupancy/ is not changed."""
    return {"retrofit": [{"kind": "widen", "roads": wide_roads(), "to": 18.0}]}


def waits_config(screen, configs):
    return dict(title="Waiting of full-path reservation split by resource: both yards, 16 h, low screen levels x K_res x 10 "
                      "seeds", jobs=[list(j) for j in wait_jobs(screen, configs)])


def wide_config(screen, configs):
    """Okpo roads of at most 15 m widened to 18 m, waiting split by resource at two levels (100, 200 tasks a day, the K_res
    and seeds of waits, paired); also the system bound after widening."""
    from occupancy import params as P
    w = wide_params()
    return dict(title="Okpo roads of at most 15 m widened to 18 m: waiting split by resource (100, 200 tasks a day x K_res "
                      "x 10 seeds) and the system bound", wide=w,
                jobs=[list(j) for j in wait_jobs(screen, configs, {"yupu": (100, 200)}, w)],
                inputs=dict(network=P.YARDS["yupu"]["network"]))


def wjob(a):
    out, job = a
    R.pin(list(range(1, os.cpu_count())))
    r = wait_point(job)
    R.write_json(out / "parts" / (r["key"].replace("|", "_") + ".json"), r)
    return r["key"], r["secs"]


def run_jobs(cfg, out, a, tag):
    if (out / "config.json").exists() and json.loads((out / "config.json").read_text(encoding="utf-8")) != json.loads(json.dumps(cfg, ensure_ascii=False)):
        sys.exit(f"refusing to reuse {out}: its config.json differs from the current configuration")
    (out / "parts").mkdir(parents=True, exist_ok=True)
    R.write_json(out / "config.json", cfg)
    print(f"run directory: {out}", flush=True)
    jobs = [tuple(j) for j in cfg["jobs"]]
    todo = [j for j in jobs if not (out / "parts" / f"{j[0]}_{j[1]}_{j[2]}_{j[3]}.json").exists()]
    prog = R.Progress(out / "progress.json", len(jobs), len(jobs) - len(todo), a.procs, cfg["title"], tag)
    stop = threading.Event()
    threading.Thread(target=prog.heartbeat, args=(stop, 60.0), daemon=True).start()
    for key, secs in R.slot_imap(wjob, [(out, j) for j in todo], a.procs, slots_dir=a.slots, tag=tag):
        prog.log(f"{key} ({secs:.0f} s)", done=prog.d["done"] + 1)
    stop.set()
    prog.write(state="done")


def run_waits(a):
    run_jobs(waits_config(a.screen, a.wait_configs), Path(a.out), a, "screen_waits")
    print("done", flush=True)


def run_wide(a):
    """Widening: the waits split by resource, then the system bound and per-resource quantities (chain_load.compute; the
    case adds only retrofit)."""
    import chain_load as C
    cfg, out = wide_config(a.screen, a.wait_configs), Path(a.out)
    run_jobs(cfg, out, a, "screen_wide")
    if not (out / "chain_wide.json").exists():
        C.SCENS["yupu_wide18"] = ("yupu", cfg["wide"])
        C.REFS.update(sensitivity=a.sensitivity)
        R.write_json(out / "chain_wide.json", C.compute("yupu_wide18"))
    print("done", flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--part", choices=("seeds", "waits", "wide", "intra"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", help="seeds, intra: the stored run configuration")
    ap.add_argument("--screen", help="waits, wide: the screen cells (yard, day_h, demand, K_res)")
    ap.add_argument("--wait-configs", nargs="+", metavar="YARD=FILE",
                    help="waits, wide: run configuration of each yard whose scenario d16 is rerun")
    ap.add_argument("--sensitivity", help="wide: bound-sensitivity table, the reference lambda^S of chain_load")
    ap.add_argument("--procs", type=int, default=23)
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--slots", default=os.environ.get("CPU_SLOTS"), help="optional CPU-slot directory shared by runs "
                    "(default: environment variable CPU_SLOTS; without it an ordinary process pool)")
    a = ap.parse_args()
    need = {"seeds": ("config",), "intra": ("config",), "waits": ("screen", "wait_configs"),
            "wide": ("screen", "wait_configs", "sensitivity")}[a.part]
    missing = [f"--{x.replace('_', '-')}" for x in need if getattr(a, x) is None]
    if missing:
        ap.error(f"--part {a.part} needs {' '.join(missing)}")
    if a.wait_configs:
        a.wait_configs = dict(x.split("=", 1) for x in a.wait_configs)
    dict(seeds=run_seeds, waits=run_waits, wide=run_wide, intra=run_intra)[a.part](a)


if __name__ == "__main__":
    main()

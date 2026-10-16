"""Screen grid of the main case, the conserved inter-area mapping (paper, the screening figure: the main case's curves; the
lower end read in whole vehicles; supplementary material, onset of saturation, required fleets and the screen).

The configurations are those of the conserved-mapping required-fleet runs (scenarios d16_c and d24_c: conserved mapping with
intra-area share q = 0, whole roads, main mix, handling 10 min), with other levels, seeds, fleets and title:
  screen  one yard x 16 / 24 h x free flow and full-path reservation x 10 seeds: the levels of the screening curves of the
          proportional mapping (SCREEN_LEVELS, SCREEN_MID_LEVELS and ANCHOR_LEVELS; on 16-h days also SCREEN_LOW_LEVELS) whose
          demand ratio (n / lambda^S of the conserved mapping) is at most 25 %, without the levels already computed in a stored
          run (--done; at Okpo 339, 350, 678 and 700). K_req is searched as in required_fleet.py (ten-seed mean throughput
          >= 0.99 x daily demand). Writes <out>/config.json, points/, points.csv and demand.csv.
  seeds   the cells with demand ratio <= 6 % x seeds 101-130 x the whole K grid (Okpo 1-9, Yantai 1-6), every point computed
          (runner.sweep); the points of seeds 101-110 equal those of the screen with the same key. Writes <out>/config.json,
          points/ and points.csv.
Run:   python experiments/screen_conserved.py --part screen|seeds --yard yupu|yantai --conserved-scan <scan.csv> --out <dir>
           [--done <config.json>] [--procs N] [--slots DIR] [--max-minutes M]
--conserved-scan: scan.csv of conserved_bounds.py or the released results/main_case/scan.csv (lambda^S of the conserved
mapping: the q = 0 rows); --done: config.json of the stored conserved-mapping required-fleet run of the yard
(results/runs/okpo_main_fleet), whose levels are not computed again.
--max-minutes (seeds only): no new points after this time; the run then exits with code 3 and the same command resumes it.
Both parts resume an interrupted run from the points written so far.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import runner as F                           # noqa: E402
import conserved_bounds as TB                # noqa: E402
from occupancy import params as P            # noqa: E402

TOP_RATIO, LOW_RATIO = 0.25, 0.06
SEEDS30 = tuple(range(101, 131))
KS30 = {"yupu": tuple(range(1, 10)), "yantai": tuple(range(1, 7))}
KS = tuple(range(1, 41)) + tuple(range(45, 151, 5))     # K grid of the required-fleet configurations
SCAN, DONE = None, None                                 # --conserved-scan, --done (set by main)


def conserved_lam(y):
    """The stored lambda^S of the conserved mapping (moves/day, 16-h day, whole roads, q = 0): the q = 0 row of the scan."""
    return TB.q0_lam(SCAN)[y]


def lam(y, h):
    """lambda^S of the conserved mapping (moves/day, h-hour working day; on 24-h days 1.5 times that of 16 h, checked to the
    last bit)."""
    return conserved_lam(y) * h / 16


def committed(y):
    """Levels of the conserved mapping already computed in a stored run (--done; d16_c and d24_c have the same levels)."""
    return set(F.load_config(DONE)["demand"].values()) if DONE else set()


def levels(y, top=TOP_RATIO):
    """{working day: levels to compute}: the levels of the screening curves of the proportional mapping (SCREEN_LEVELS,
    SCREEN_MID_LEVELS, ANCHOR_LEVELS; on 16-h days also SCREEN_LOW_LEVELS) with demand ratio <= top, without the stored ones."""
    cand = set(F.SCREEN_LEVELS[y]) | set(F.SCREEN_MID_LEVELS[y]) | set(F.ANCHOR_LEVELS[y])
    return {h: tuple(sorted(n for n in cand | (set(F.SCREEN_LOW_LEVELS[y]) if h == 16 else set()) if n / lam(y, h) <= top and n not in committed(y)))
            for h in (16, 24)}


def base_config(y):
    """The configuration of the conserved-mapping required-fleet runs of a yard (scenarios d16_c and d24_c; levels to be
    set)."""
    Y = P.YARDS[y]
    scen = {"d16_c": {"demand_map": "conserved"}, "d24_c": {"day_s": 24 * 3600.0, "demand_map": "conserved"}}
    return dict(network=Y["network"], yard=y, roles=Y["roles"], params=P.DEFAULTS, scenarios=scen, demand={}, levels={},
                ks=list(KS), seeds=list(F.SEEDS), models={s: ["free", "reserve"] for s in scen}, n_mix=2000, bounds=False)


def screen_config(y):
    c = base_config(y)
    lv = levels(y)
    sc = {s: c["scenarios"][s] for s in ("d16_c", "d24_c")}
    c.update(scenarios=sc, models={s: c["models"][s] for s in sc},
             demand={f"n{x}": x for x in sorted(set(lv[16]) | set(lv[24]))}, levels={f"d{h}_c": [f"n{x}" for x in lv[h]] for h in (16, 24)},
             title=f"{P.YARDS[y]['name']}, main case: screen grid, demand ratio <= 25 % x 16 / 24 h x free flow, reservation")
    return c


def seeds_config(y):
    c = screen_config(y)
    lv = levels(y, LOW_RATIO)
    c.update(seeds=list(SEEDS30), ks=list(KS30[y]), demand={f"n{x}": x for x in sorted(set(lv[16]) | set(lv[24]))},
             levels={f"d{h}_c": [f"n{x}" for x in lv[h]] for h in (16, 24)},
             title=f"{P.YARDS[y]['name']}, main case: screen lower end, ratio <= 6 % x seeds 101-130 x K = 1-{KS30[y][-1]}")
    return c


def prepare(cfg, out):
    out = Path(out)
    if (out / "config.json").exists():
        if json.loads((out / "config.json").read_text(encoding="utf-8")) != json.loads(json.dumps(cfg, ensure_ascii=False)):
            sys.exit(f"refusing to reuse {out}: its config.json differs from this configuration")
    else:
        out.mkdir(parents=True, exist_ok=True)
        F.write_json(out / "config.json", cfg)
    return out


def main():
    global SCAN, DONE
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--part", choices=("screen", "seeds"), required=True)
    ap.add_argument("--yard", choices=("yupu", "yantai"), required=True)
    ap.add_argument("--conserved-scan", required=True, help="scan.csv of conserved_bounds or results/main_case/scan.csv")
    ap.add_argument("--done", default=None, help="config.json of the stored conserved-mapping required-fleet run of the yard")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--max-minutes", type=float, default=None, help="time limit of the seeds part")
    ap.add_argument("--slots", default=os.environ.get("CPU_SLOTS"), help="optional CPU-slot directory shared by runs")
    a = ap.parse_args()
    SCAN, DONE = a.conserved_scan, a.done
    if a.part == "screen":           # required_fleet searches K_req on the configuration written to the run directory
        out = prepare(screen_config(a.yard), a.out)
        import required_fleet
        sys.argv = [sys.argv[0], str(out / "config.json"), "--out", str(out), "--procs", str(a.procs)] + \
            (["--slots", a.slots] if a.slots else [])
        return required_fleet.main()
    cfg = seeds_config(a.yard)
    out = prepare(cfg, a.out)
    print(f"run directory: {out}", flush=True)
    failed, stopped = F.sweep(cfg, out, a.procs, time.time() + 60 * a.max_minutes if a.max_minutes else None, a.slots)
    if stopped:
        print("stopped at the time limit; run the same command to resume", flush=True)
        sys.exit(3)
    sims = sorted(F.load_points(out)[0], key=lambda x: (x["scen"], x["model"], x["level"], x["K"], x["seed"]))
    with open(out / "points.csv", "w", newline="", encoding="utf-8-sig") as f:
        F.write_points(f, sims)
    print(f"done; {len(sims)} points, {failed} failed", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

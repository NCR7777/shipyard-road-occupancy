"""Bounds under the demand mappings: the system bound and the sampled bound of the proportional mapping, of the mapping
with intra-area moves and of inter-area moves only.

Cases: two yards x mapping (main = the proportional mapping, the earlier one; intra = intra-area moves; inter_only =
inter-area moves only) x loading and unloading 10 or 30 min each x working day 16 or 24 h; whole roads, main flow mix.
  system bound lambda^S: bounds.system.SystemT1 on the task law of demand_mapping.system_mix (intra = 0.345 x inter-area law
      + 0.655 x intra-area law), the empty-run tail clearance truncated at the stop roads (stop_ret, as in the bound
      sensitivity); in moves a day (both kinds together). main and inter_only have the same task law; the main values at 16 h
      are checked against the stored bound-sensitivity table.
  sampled bound lambda^N: the first n_mix tasks of the task stream of each seed under the intra mapping
      (demand_mapping.stream), T1 and T1' of occupancy.bound.bounds (as the bound points of runner.compute).

Run:   python experiments/mapping_bounds.py --out <dir> --sensitivity <bound-sensitivity table> [--procs 11] [--slots DIR]
--sensitivity is the table of the bound sensitivity (CSV with the columns yard, scen, section, variant, T1_sys; the stored
one is results/bounds/sensitivity.csv), read for the check of the main values.
Writes, case by case and resumable, <dir>/config.json, progress.json, points/<key>.json (one file per case), and the tables
system.csv (system bounds), sampled.csv (sampled bounds) and system.json (both).
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from occupancy import params as P            # noqa: E402
import runner as R                           # noqa: E402
import demand_mapping as M                   # noqa: E402

YARDS = ("yupu", "yantai")
MAPS = ("main", "intra", "inter_only")
LOADS = (10, 30)


def params(m, load, h):
    p = dict(P.DEFAULTS, load_s=60.0 * load, unload_s=60.0 * load, day_s=h * 3600.0)
    if m != "main":
        p["demand_map"] = m
    return p


def make_config():
    keys = [f"sys|{y}|{m}|{l}|{h}" for y in YARDS for m in MAPS for l in LOADS for h in P.DAY_HOURS]
    keys += [f"smp|{y}|intra|{l}|16|{s}" for y in YARDS for l in LOADS for s in R.SEEDS]
    return dict(title="System and sampled bounds under the demand mappings", yards=list(YARDS), maps=list(MAPS),
                loads=list(LOADS), days=list(P.DAY_HOURS), seeds=list(R.SEEDS), n_mix=R.N_MIX, params=P.DEFAULTS, keys=keys,
                inputs={y: dict(network=P.YARDS[y]["network"], roles=P.YARDS[y]["roles"]) for y in YARDS})


def net_of(y, p):
    from occupancy.network import Net
    Y = P.YARDS[y]
    return Net(Y["network"], p, Y["roles"])


def compute(key):
    from bounds.system import SystemT1, scenario_mix, stop_ret
    from occupancy.bound import bounds
    kind, y, m, load, h, *seed = key.split("|")
    p = params(m, int(load), int(h))
    net = net_of(y, p)
    t = time.time()
    if kind == "sys":
        st = SystemT1(net, p, M.system_mix(net, p) if m != "main" else scenario_mix(net, p)[0])
        v, b = st.value(top=3, ret=stop_ret(net, p["plat_len"] + p["long_clear"]))
        res = dict(T1_sys=v, binding=[dict(name=x["name"], res=x["res"], price_per_h=x["price_per_h"], util=x["util"]) for x in b])
    else:
        s = M.stream(net, int(seed[0]), p)
        tasks = [s.next(0.0) for _ in range(R.N_MIX)]
        dock = set().union(*net.near(P.DOCK_CLASS))
        r = bounds(net, p, tasks, local=dock)
        res = {k: r[k] for k in ("network", "network_tight", "network_tight_local", "cycle_s", "empty_min_s")}
        res.update(n_intra=sum(t["flow"] == M.INTRA for t in tasks), same_stop=sum(t["o"] == t["d"] for t in tasks),
                   binding_tight=[dict(name=x["name"], price_per_h=x["price_per_h"]) for x in r["binding_tight"][:3]])
    return dict(key=key, kind=kind, yard=y, map=m, load_min=int(load), day_h=int(h), seed=int(seed[0]) if seed else None,
                secs=round(time.time() - t, 1), **res)


def work(args):
    key, out = args
    r = compute(key)
    R.write_json(Path(out) / "points" / (key.replace("|", "_") + ".json"), r)
    return key, r["secs"]


def summarize(cfg, out, sens):
    """points/ -> system.csv, sampled.csv and system.json. sens: the bound-sensitivity table (main scenario, whole roads,
    variants base = 10 min and l30 = 30 min handling)."""
    pts = {x["key"]: x for x in (json.loads(f.read_text(encoding="utf-8")) for f in (out / "points").glob("*.json"))}
    ref = {(r["yard"], r["variant"]): float(r["T1_sys"]) for r in csv.DictReader(open(sens, encoding="utf-8-sig"))
           if r["scen"] == "main" and r["section"] == "whole" and r["variant"] in ("base", "l30")}
    rows = []
    for y in YARDS:
        for l in LOADS:
            # the sensitivity table has 16-h values only: lambda^S of the proportional mapping at 16 h equals them exactly
            # (==); 24 h is solved by the same linear programme with H = 24 h, its ratio to the 16-h value is scale_16h
            ref_main = ref[(y, "base" if l == 10 else "l30")]
            same = pts[f"sys|{y}|main|{l}|16"]["T1_sys"] == ref_main
            for h in P.DAY_HOURS:
                base = pts[f"sys|{y}|main|{l}|{h}"]["T1_sys"]
                for m in MAPS:
                    x = pts[f"sys|{y}|{m}|{l}|{h}"]
                    b = x["binding"] + [{}] * 3
                    rows.append(dict(yard=y, map=m, load_min=l, day_h=h, T1_sys=x["T1_sys"], rel_main=x["T1_sys"] / base - 1,
                                     main_16h_earlier=ref_main, same_as_earlier=same,
                                     scale_16h=x["T1_sys"] / pts[f"sys|{y}|{m}|{l}|16"]["T1_sys"],
                                     **{f"top{i + 1}": b[i].get("name") for i in range(3)},
                                     **{f"price{i + 1}": b[i].get("price_per_h") for i in range(3)},
                                     **{f"util{i + 1}": b[i].get("util") for i in range(3)}))
    smp = []
    for y in YARDS:
        for l in LOADS:
            xs = [pts[f"smp|{y}|intra|{l}|16|{s}"] for s in cfg["seeds"]]
            T1s = pts[f"sys|{y}|intra|{l}|16"]["T1_sys"]
            for k in ("network", "network_tight"):
                v = [x[k] for x in xs]
                smp.append(dict(yard=y, map="intra", load_min=l, day_h=16, bound=k, mean=sum(v) / len(v), lo=min(v), hi=max(v), T1_sys=T1s,
                                n_intra_mean=sum(x["n_intra"] for x in xs) / len(xs), same_stop_mean=sum(x["same_stop"] for x in xs) / len(xs),
                                top1=json.dumps(sorted({x["binding_tight"][0]["name"] for x in xs}), ensure_ascii=False)))
    for name, rs in (("system.csv", rows), ("sampled.csv", smp)):
        with open(out / name, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, list(rs[0]))
            w.writeheader()
            w.writerows(rs)
    R.write_json(out / "system.json", dict(system=rows, sampled=smp))
    return rows, smp


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--sensitivity", required=True, help="bound-sensitivity table (CSV), for the check of the main values")
    ap.add_argument("--procs", type=int, default=11)
    ap.add_argument("--slots", default=os.environ.get("CPU_SLOTS"), help="optional CPU-slot directory shared by runs "
                    "(default: environment variable CPU_SLOTS; without it an ordinary process pool)")
    a = ap.parse_args()
    cfg = make_config()
    out = Path(a.out)
    if (out / "config.json").exists():
        if json.loads((out / "config.json").read_text(encoding="utf-8")) != json.loads(json.dumps(cfg, ensure_ascii=False)):
            sys.exit(f"refusing to reuse {out}: its config.json differs from the current configuration")
    else:
        out.mkdir(parents=True, exist_ok=True)
        R.write_json(out / "config.json", cfg)
    (out / "points").mkdir(exist_ok=True)
    print(f"run directory: {out}", flush=True)
    todo = [k for k in cfg["keys"] if not (out / "points" / (k.replace("|", "_") + ".json")).exists()]
    todo.sort(key=lambda k: (not (k.startswith("sys|yupu")), k))          # the system bounds of Okpo take longest: first
    prog = R.Progress(out / "progress.json", len(cfg["keys"]), len(cfg["keys"]) - len(todo), a.procs, cfg["title"],
                      "mapping_bounds")
    cores = list(range(1, os.cpu_count()))
    for key, secs in R.slot_imap(work, [(k, str(out)) for k in todo], min(a.procs, max(1, len(todo))), initializer=R.pin,
                                 initargs=(cores,), slots_dir=a.slots, tag="mapping_bounds"):
        prog.log(f"{key} ({secs} s)", done=prog.d["done"] + 1)
    rows, _ = summarize(cfg, out, a.sensitivity)
    prog.write(state="done", note=f"{len(rows)} rows")
    for r in rows:
        print(r["yard"], r["map"], r["load_min"], r["day_h"], round(r["T1_sys"], 1), r["top1"], flush=True)


if __name__ == "__main__":
    main()

"""System bound under the conserved inter-area mapping (the main case) with dock roads sectioned at the stops and with 30-min
loading and unloading: the rows of the capacity interval for these two variants.

Cases: two yards x resource granularity (whole roads / dock roads sectioned at the stops) x loading and unloading 10 or 30
min x 16 h; task law conserved_mapping.system_mix (q = 0), the empty-run tail clearance truncated at the stop roads (as in
conserved_bounds). The whole-road 10-min value is checked against the q = 0 value of conserved_bounds (exact equality,
column same_as_scan). Each point keeps the ten resources with the highest dual prices.

Run:   python experiments/conserved_bounds_sections.py --out <dir> --conserved-scan <scan.csv>
                [--procs 10] [--slots DIR]
--conserved-scan: scan.csv of conserved_bounds or the released results/main_case/scan.csv (conserved_bounds.q0_lam).
Writes, case by case and resumable, <dir>/config.json, progress.json, points/<key>.json and bounds.csv.
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
import mapping_bounds as B                   # noqa: E402
import conserved_bounds as TB                # noqa: E402

YARDS = ("yupu", "yantai")
SECS = ("whole", "stops")
LOADS = (10, 30)


def make_config():
    keys = [f"{y}|{s}|{l}" for y in YARDS for s in SECS for l in LOADS]
    return dict(title="System bound of the conserved mapping: sectioned dock roads and 30-min handling", yards=list(YARDS),
                secs=list(SECS), loads=list(LOADS), params=P.DEFAULTS, keys=keys, inputs=B.make_config()["inputs"])


def params(sec, load):
    p = dict(P.DEFAULTS, demand_map="conserved", load_s=60.0 * load, unload_s=60.0 * load)
    return dict(p, section="stops") if sec == "stops" else p


def compute(key):
    from bounds.system import SystemT1, stop_ret
    from capacity_bounds import facility, res_info
    import conserved_mapping
    y, sec, load = key.split("|")
    p = params(sec, int(load))
    net = B.net_of(y, p)
    t = time.time()
    st = SystemT1(net, p, conserved_mapping.system_mix(net, p))
    v, b = st.value(top=10, ret=stop_ret(net, p["plat_len"] + p["long_clear"]))
    rows = [dict(name=x["name"], res="|".join(map(str, x["res"])), price_per_h=x["price_per_h"], util=x["util"],
                 facility=facility(net, tuple(x["res"])), **res_info(net, st.cap, st.size, p, tuple(x["res"]))) for x in b]
    return dict(key=key, yard=y, section=sec, load_min=int(load), day_h=16, T1_sys=v, binding=rows, secs=round(time.time() - t, 1))


def work(args):
    key, out = args
    r = compute(key)
    R.write_json(Path(out) / "points" / (key.replace("|", "_") + ".json"), r)
    return key, r["secs"]


def summarize(cfg, out, scan):
    """points/ -> bounds.csv. scan: scan.csv of conserved_bounds or the released one (its q = 0 rows are the whole-road
    10-min values)."""
    pts = {x["key"]: x for x in (json.loads(f.read_text(encoding="utf-8")) for f in (out / "points").glob("*.json"))}
    ref = TB.q0_lam(scan)
    rows = []
    for k in cfg["keys"]:
        x = pts[k]
        b = x["binding"]
        top = next((r for r in b if r["price_per_h"] > 0), b[0])
        rows.append(dict(yard=x["yard"], section=x["section"], load_min=x["load_min"], T1_sys=x["T1_sys"], T1_sys_24h=1.5 * x["T1_sys"],
                         priced=top["name"], priced_type=top["type"], **{f"top{i + 1}": b[i]["name"] for i in range(3)},
                         **{f"util{i + 1}": b[i]["util"] for i in range(3)},
                         same_as_scan=(x["T1_sys"] == ref[x["yard"]]) if (x["section"], x["load_min"]) == ("whole", 10)
                         else None))
    with open(out / "bounds.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--conserved-scan", required=True,
                    help="scan.csv of conserved_bounds (or the released one), for the check of the whole-road 10-min value")
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--slots", default=os.environ.get("CPU_SLOTS"), help="optional CPU-slot directory shared by runs "
                    "(default: environment variable CPU_SLOTS; without it an ordinary process pool)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
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
    prog = R.Progress(out / "progress.json", len(cfg["keys"]), len(cfg["keys"]) - len(todo), a.procs, cfg["title"],
                      "conserved_bounds_sections")
    for key, secs in R.slot_imap(work, [(k, str(out)) for k in todo], min(a.procs, max(1, len(todo))), initializer=R.pin,
                                 initargs=(list(range(1, os.cpu_count())),), slots_dir=a.slots,
                                 tag="conserved_bounds_sections"):
        prog.log(f"{key} ({secs} s)", done=prog.d["done"] + 1)
    rows = summarize(cfg, out, a.conserved_scan)
    prog.write(state="done", note=f"{len(rows)} rows")
    for r in rows:
        print(r["yard"], r["section"], r["load_min"], round(r["T1_sys"], 1), r["priced"], r["same_as_scan"], flush=True)


if __name__ == "__main__":
    main()

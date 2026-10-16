"""System bound under the inter-area mapping with the erection flow conserved (the main case) and the scan of the intra-area
share.

Cases: two yards x intra-area share q (0-80 % in steps of 2.5 %, plus 65.5 %, the share of the intra mapping of
mapping_bounds) x loading and unloading 10 min x 16 h; whole roads, main flow mix, the empty-run tail clearance truncated at
the stop roads (stop_ret, as in mapping_bounds). Task law: conserved_mapping.system_mix (0.02 to the docks + (0.98 - q) other
inter-area moves + q intra-area moves).
Three checks: q = 0 at 24 h (its ratio to 16 h should be 1.5); q = 0 against the proportional task law with
flow_share = {P6: 0.02} (should be the same); q = 0.655 against the intra mapping of mapping_bounds (read in the summary).
Each point keeps the ten resources with the highest dual prices (resource id, type, size, building or zone, price,
utilisation): the data of the binding resources along the scan of the intra-area share.

Run:   python experiments/conserved_bounds.py --out <dir> --mapping-bounds <system.csv of mapping_bounds>
                [--procs 11] [--slots DIR]
--mapping-bounds: system.csv of mapping_bounds or the released results/intra_area/bounds.csv (yard, map, load_min, day_h,
T1_sys). q0_lam reads the q = 0 rows of scan.csv (or of the released results/main_case/scan.csv) for the other modules.
Writes, case by case and resumable, <dir>/config.json, progress.json, points/<key>.json, and the tables scan.csv (one row
per yard and q), binding.csv (the ten priced resources at q = 0) and checks.json.
"""
import argparse
import csv
import json
import os
import sys
import time
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from occupancy import params as P            # noqa: E402
import runner as R                           # noqa: E402
import mapping_bounds as B                   # noqa: E402

YARDS = B.YARDS
QS = sorted({Fraction(i, 40) for i in range(33)} | {Fraction(131, 200)})     # 0, 2.5 %, ..., 80 %, plus 65.5 %
qs = lambda q: f"{float(q):g}"


def make_config():
    keys = [f"sys|{y}|{qs(q)}|16" for y in YARDS for q in QS] + [f"sys|{y}|0|24" for y in YARDS] + [f"fs|{y}|0|16" for y in YARDS]
    return dict(title="System bound of the conserved mapping and the scan of the intra-area share", yards=list(YARDS),
                qs=[qs(q) for q in QS], load_min=10, params=P.DEFAULTS, keys=keys, inputs=B.make_config()["inputs"])


def params(q, h):
    p = dict(P.DEFAULTS, day_s=h * 3600.0, demand_map="conserved")
    return dict(p, intra_share=float(q)) if q else p


def compute(key):
    from bounds.system import SystemT1, scenario_mix, stop_ret
    from capacity_bounds import facility, res_info
    import conserved_mapping
    kind, y, q, h = key.split("|")
    q, h = Fraction(q), int(h)
    p = params(q, h)
    net = B.net_of(y, p)
    t = time.time()
    if kind == "fs":           # check: the proportional task law with the share of the flow to the docks set to 2 %
        pm = dict(P.DEFAULTS, day_s=h * 3600.0, flow_share={"P6": 0.02})
        st = SystemT1(net, pm, scenario_mix(net, pm)[0])
    else:
        st = SystemT1(net, p, conserved_mapping.system_mix(net, p))
    v, b = st.value(top=10, ret=stop_ret(net, p["plat_len"] + p["long_clear"]))
    rows = [dict(name=x["name"], res="|".join(map(str, x["res"])), price_per_h=x["price_per_h"], util=x["util"],
                 facility=facility(net, tuple(x["res"])), **res_info(net, st.cap, st.size, p, tuple(x["res"]))) for x in b]
    return dict(key=key, kind=kind, yard=y, q=float(q), day_h=h, T1_sys=v, binding=rows, secs=round(time.time() - t, 1))


def work(args):
    key, out = args
    r = compute(key)
    R.write_json(Path(out) / "points" / (key.replace("|", "_") + ".json"), r)
    return key, r["secs"]


def summarize(cfg, out, mapping):
    """points/ -> scan.csv, binding.csv and checks.json. mapping: system.csv of mapping_bounds (the intra and main values at
    10 min and 16 h)."""
    pts = {x["key"]: x for x in (json.loads(f.read_text(encoding="utf-8")) for f in (out / "points").glob("*.json"))}
    ref = {(r["yard"], r["map"]): float(r["T1_sys"]) for r in csv.DictReader(open(mapping, encoding="utf-8-sig"))
           if r["load_min"] == "10" and r["day_h"] == "16"}
    scan, binding, checks = [], [], {}
    for y in YARDS:
        prev = None
        for q in QS:
            x = pts[f"sys|{y}|{qs(q)}|16"]
            b = x["binding"]
            top = next((r["name"] for r in b if r["price_per_h"] > 0), b[0]["name"])
            scan.append(dict(yard=y, q=float(q), T1_sys=x["T1_sys"], T1_sys_24h=1.5 * x["T1_sys"], priced=top, switch=prev is not None and top != prev,
                             **{f"top{i + 1}": b[i]["name"] for i in range(3)}, **{f"type{i + 1}": b[i]["type"] for i in range(3)},
                             **{f"util{i + 1}": b[i]["util"] for i in range(3)}, **{f"price{i + 1}": b[i]["price_per_h"] for i in range(3)},
                             secs=x["secs"]))
            prev = top
        x0 = pts[f"sys|{y}|0|16"]
        binding += [dict(yard=y, q=0.0, day_h=16, rank=i + 1, **r) for i, r in enumerate(x0["binding"])]
        checks[y] = dict(scale_24h=pts[f"sys|{y}|0|24"]["T1_sys"] / x0["T1_sys"],
                         flow_share_002=pts[f"fs|{y}|0|16"]["T1_sys"], rel_flow_share=pts[f"fs|{y}|0|16"]["T1_sys"] / x0["T1_sys"] - 1,
                         intra_mapping=ref[(y, "intra")], q0655=pts[f"sys|{y}|0.655|16"]["T1_sys"],
                         rel_intra_mapping=pts[f"sys|{y}|0.655|16"]["T1_sys"] / ref[(y, "intra")] - 1,
                         main_mapping=ref[(y, "main")])
    for name, rs in (("scan.csv", scan), ("binding.csv", binding)):
        with open(out / name, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, list(rs[0]))
            w.writeheader()
            w.writerows(rs)
    R.write_json(out / "checks.json", checks)
    return scan, checks


def q0_lam(path):
    """{yard: lambda^S at q = 0, 16 h} from scan.csv of this module (column T1_sys) or from the released
    results/main_case/scan.csv (the same rows, the column named lam16)."""
    rows = csv.DictReader(open(path, encoding="utf-8-sig"))
    return {r["yard"]: float(r["T1_sys"] if "T1_sys" in r else r["lam16"]) for r in rows if float(r["q"]) == 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--mapping-bounds", required=True, help="system.csv of mapping_bounds, for the check at q = 0.655")
    ap.add_argument("--procs", type=int, default=11)
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
    todo.sort(key=lambda k: "|yupu|" not in k)          # Okpo takes longest: first
    prog = R.Progress(out / "progress.json", len(cfg["keys"]), len(cfg["keys"]) - len(todo), a.procs, cfg["title"],
                      "conserved_bounds")
    for key, secs in R.slot_imap(work, [(k, str(out)) for k in todo], min(a.procs, max(1, len(todo))), initializer=R.pin,
                                 initargs=(list(range(1, os.cpu_count())),), slots_dir=a.slots, tag="conserved_bounds"):
        prog.log(f"{key} ({secs} s)", done=prog.d["done"] + 1)
    scan, checks = summarize(cfg, out, a.mapping_bounds)
    prog.write(state="done", note=f"{len(scan)} rows")
    for r in scan:
        print(r["yard"], f"{r['q']:.3f}", round(r["T1_sys"], 1), r["priced"], "<- switch" if r["switch"] else "", flush=True)
    print(json.dumps(checks, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

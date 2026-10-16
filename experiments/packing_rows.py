"""Packing rows: how much the width-packing rows tighten the system bound under the conserved mapping of the main case (the
bound with the utilisation rows only). Linear programmes only.

Computed as T1_nopack of the bound sensitivity (capacity_bounds.keep_rows, called unchanged): with the same windows,
empty-run constraints and truncation of the tail clearance (stop_ret),
  lambda^S        the system bound (all occupancy rows k = 0, 1, 2; SystemT1.value);
  lambda^S (k=0)  only the utilisation rows k = 0 kept, the two groups of Fekete-Schepers packing rows k = 1, 2 dropped,
                  re-solved; tightening = 1 - lambda^S / lambda^S (k=0);
  also solved with only the group k = 1 added (k01) and only the group k = 2 added (k02), to see which rows tighten.
Binding rows: dual price (tasks/day per extra h/day) and utilisation (activity / capacity) of the binding element of
lambda^S (the resource with the highest price) in each row k.
Cases: the conserved mapping (conserved_mapping.system_mix, q = 0) at Okpo with whole and sectioned roads and at Yantai with
whole roads; the proportional mapping (the proportional task law) for the same three cases, for comparison. Parameters:
route_free_mappings.params, the same as the main scenario of the bound sensitivity, so lambda^S and T1_nopack of the
proportional mapping should equal its stored values exactly (checked in the summary); lambda^S is also checked against
route_free_mappings.

Run:   python experiments/packing_rows.py --out <dir> --sensitivity <bound-sensitivity table>
                --route-free <points.json or route_free.csv of route_free_mappings> [--procs 4] [--slots DIR]
--sensitivity: CSV with the columns yard, scen, section, variant, T1_sys, T1_nopack (the stored one is
results/bounds/sensitivity.csv). --route-free: points.json of route_free_mappings, or a CSV with the columns map, yard,
section, T1_sys (its route_free.csv; the stored one is results/bounds/route_free_mappings.csv).
Writes <dir>/config.json, progress.json, points/<key>.json (cache for resuming), points.json and nopack.csv.
"""
import argparse
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
from occupancy import params as P            # noqa: E402
import runner as R                           # noqa: E402
import mapping_bounds as B                   # noqa: E402
import route_free_mappings as RF             # noqa: E402

KEYS = [f"{m}|{y}|{s}" for m in ("conserved", "proportional") for y, s in (("yupu", "whole"), ("yupu", "stops"), ("yantai", "whole"))]


def make_config():
    return dict(title="Tightening of the system bound by the width-packing rows under the conserved mapping", keys=KEYS,
                load_min=10, params=P.DEFAULTS, inputs=B.make_config()["inputs"])


def compute(key):
    from bounds.system import SystemT1, scenario_mix, stop_ret
    from occupancy.bound import DFF_K
    import capacity_bounds as EB
    import conserved_mapping
    t0 = time.time()
    m, y, sec = key.split("|")
    p = RF.params(m, sec)
    net = B.net_of(y, p)
    mix = conserved_mapping.system_mix(net, p) if m == "conserved" else scenario_mix(net, p)[0]
    st = SystemT1(net, p, mix)
    ret = stop_ret(net, p["plat_len"] + p["long_clear"])
    lo = {}
    v, b = st.value(top=3, ret=ret, lp_out=lo)
    nr, i = len(lo["res_list"]), lo["res_list"].index(tuple(b[0]["res"]))
    k_rows = [k * nr + i for k in range(len(DFF_K))]
    out = dict(key=key, map=m, yard=y, section=sec, T1_sys=v, binding=[dict(name=x["name"], price_per_h=x["price_per_h"], util=x["util"]) for x in b],
               top_price_k=[float(lo["mu"][r]) * 3600 for r in k_rows], top_util_k=[float(lo["act"][r] / lo["H"][r]) for r in k_rows])
    for name, keep in (("nopack", (0,)), ("k01", (0, 1)), ("k02", (0, 2))):
        lam, top, price = EB.keep_rows(st, keep, ret)
        out[name] = dict(T1=lam, top=top, price_per_h=price)
    return dict(out, secs=round(time.time() - t0, 1))


def work(args):
    key, out = args
    r = compute(key)
    R.write_json(Path(out) / "points" / (key.replace("|", "_") + ".json"), r)
    return key, r["secs"]


def summarize(cfg, out, sens, route_free):
    """points/ -> points.json and nopack.csv (without the run times secs: byte-identical when recomputed with points/ moved
    away); the proportional mapping is checked against T1_sys and T1_nopack of the bound-sensitivity table sens (main
    scenario, base), lambda^S against route_free_mappings (route_free: its points.json, or its route_free.csv = the released
    results/bounds/route_free_mappings.csv)."""
    pts = [{k: v for k, v in json.loads((out / "points" / (key.replace("|", "_") + ".json")).read_text(encoding="utf-8")).items() if k != "secs"}
           for key in cfg["keys"]]
    R.write_json(out / "points.json", pts)
    sens_rows = {(r["yard"], r["section"]): r for r in csv.DictReader(open(sens, encoding="utf-8-sig"))
                 if r["scen"] == "main" and r["variant"] == "base"}
    if str(route_free).endswith(".json"):
        rf_rows = json.loads(Path(route_free).read_text(encoding="utf-8"))
    else:
        rf_rows = [dict(r, T1_sys=float(r["T1_sys"])) for r in csv.DictReader(open(route_free, encoding="utf-8-sig"))]
    rf = {(x["map"], x["yard"], x["section"]): x["T1_sys"] for x in rf_rows}
    rows = []
    for x in pts:
        n, ref = x["nopack"], sens_rows.get((x["yard"], x["section"])) if x["map"] == "proportional" else None
        rows.append(dict(map=x["map"], yard=x["yard"], section=x["section"], T1_sys=x["T1_sys"], top1=x["binding"][0]["name"],
                         top1_price=x["binding"][0]["price_per_h"], **{f"price_k{k}": v for k, v in enumerate(x["top_price_k"])},
                         **{f"util_k{k}": v for k, v in enumerate(x["top_util_k"])}, T1_nopack=n["T1"], nopack_top=n["top"],
                         nopack_price=n["price_per_h"], pack_tighten=1 - x["T1_sys"] / n["T1"], T1_k01=x["k01"]["T1"], k01_top=x["k01"]["top"],
                         T1_k02=x["k02"]["T1"], k02_top=x["k02"]["top"],
                         same_as_route_free_run=x["T1_sys"] == rf[(x["map"], x["yard"], x["section"])],
                         same_as_earlier=(float(ref["T1_sys"]) == x["T1_sys"] and float(ref["T1_nopack"]) == n["T1"])
                         if ref else ""))
    with open(out / "nopack.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--sensitivity", required=True,
                    help="bound-sensitivity table (CSV), for the check of the proportional mapping")
    ap.add_argument("--route-free", required=True,
                    help="points.json or route_free.csv of route_free_mappings, for the check of lambda^S")
    ap.add_argument("--procs", type=int, default=4)
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
                      "packing_rows")
    stop = threading.Event()
    threading.Thread(target=prog.heartbeat, args=(stop, 60.0), daemon=True).start()
    for key, secs in R.slot_imap(work, [(k, str(out)) for k in todo], min(a.procs, max(1, len(todo))), initializer=R.pin,
                                 initargs=(list(range(1, os.cpu_count())),), slots_dir=a.slots, tag="packing_rows"):
        prog.log(f"{key} ({secs} s)", done=prog.d["done"] + 1)
        print(key, secs, flush=True)
    stop.set()
    rows = summarize(cfg, out, a.sensitivity, a.route_free)
    prog.write(state="done", note=f"{len(rows)} rows")
    for r in rows:
        print(r["map"], r["yard"], r["section"], round(r["T1_sys"], 1), round(r["T1_nopack"], 1), f"{100 * r['pack_tighten']:.2f}%",
              r["top1"], [round(v, 2) for v in (r["price_k0"], r["price_k1"], r["price_k2"])], "|", r["nopack_top"],
              "route-free run", r["same_as_route_free_run"], "sensitivity", r["same_as_earlier"], flush=True)


if __name__ == "__main__":
    main()

"""Platform size: sensitivity of the system bound to the width and length of the vehicle platform (public reports give a
platform of about 21 m x 7.5 m for the transporters of Hanwha Ocean; the model takes the 16.25 m x 5.5 m of the DCY270).
Linear programmes only.

Two yards, conserved mapping of the main case (conserved_mapping.system_mix, q = 0; whole roads, 16 h, loading and unloading
10 min) x platform width plat_w in {5.5, 6.5, 7.5} m x platform length plat_len in {16.25, 21} m: the system bound lambda^S
(SystemT1, the empty-run tail clearance truncated at the stop roads; the truncation limit of stop_ret moves with plat_len),
the binding element and its dual prices and utilisation in the rows k = 0, 1, 2.
Only plat_w and plat_len of the parameters change; the model follows by its rules: loaded outline B = max(block width, w),
vehicle length Lv = max(block length, L) (bl_joint), empty vehicle width w + c, empty vehicle length L (bound._system), width
on each resource min(C, width + c) (network.demand); the distributions of block width and length, road widths, junction
sizes, stops and turn sweep do not change. The current platform (5.5, 16.25) gives the lambda^S of route_free_mappings and
packing_rows exactly.

Run:   python experiments/platform_size.py --out <dir> [--procs 6] [--slots DIR]
Writes <dir>/config.json, progress.json, points/<key>.json (cache for resuming), points.json and plat.csv.
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

WIDTHS, LENGTHS = (5.5, 6.5, 7.5), (16.25, 21.0)
KEYS = [f"{y}|{w}|{L}" for y in ("yupu", "yantai") for L in LENGTHS for w in WIDTHS]


def make_config():
    return dict(title="Platform size: system bound of the conserved mapping of the main case", keys=KEYS,
                widths=list(WIDTHS), lengths=list(LENGTHS), load_min=10, params=P.DEFAULTS, inputs=B.make_config()["inputs"])


def compute(key):
    from bounds.system import SystemT1, stop_ret
    from occupancy.bound import DFF_K
    import conserved_mapping
    t0 = time.time()
    y, w, L = key.split("|")
    p = dict(RF.params("conserved", "whole"), plat_w=float(w), plat_len=float(L))
    net = B.net_of(y, p)
    st = SystemT1(net, p, conserved_mapping.system_mix(net, p))
    lo = {}
    v, b = st.value(top=3, ret=stop_ret(net, p["plat_len"] + p["long_clear"]), lp_out=lo)
    nr, i = len(lo["res_list"]), lo["res_list"].index(tuple(b[0]["res"]))
    k_rows = [k * nr + i for k in range(len(DFF_K))]
    return dict(key=key, yard=y, plat_w=float(w), plat_len=float(L), T1_sys=v,
                binding=[dict(name=x["name"], price_per_h=x["price_per_h"], util=x["util"]) for x in b],
                top_price_k=[float(lo["mu"][r]) * 3600 for r in k_rows], top_util_k=[float(lo["act"][r] / lo["H"][r]) for r in k_rows],
                secs=round(time.time() - t0, 1))


def work(args):
    key, out = args
    r = compute(key)
    R.write_json(Path(out) / "points" / (key.replace("|", "_") + ".json"), r)
    return key, r["secs"]


def summarize(cfg, out):
    """points/ -> points.json and plat.csv (without the run times: byte-identical when recomputed with points/ moved away);
    rel is the change against the current platform of the yard (5.5 m, 16.25 m)."""
    pts = [{k: v for k, v in json.loads((out / "points" / (key.replace("|", "_") + ".json")).read_text(encoding="utf-8")).items() if k != "secs"}
           for key in cfg["keys"]]
    R.write_json(out / "points.json", pts)
    base = {x["yard"]: x["T1_sys"] for x in pts if (x["plat_w"], x["plat_len"]) == (WIDTHS[0], LENGTHS[0])}
    rows = [dict(yard=x["yard"], plat_w=x["plat_w"], plat_len=x["plat_len"], T1_sys=x["T1_sys"], rel=x["T1_sys"] / base[x["yard"]] - 1,
                 top1=x["binding"][0]["name"], top1_price=x["binding"][0]["price_per_h"], **{f"price_k{k}": v for k, v in enumerate(x["top_price_k"])},
                 **{f"util_k{k}": v for k, v in enumerate(x["top_util_k"])}, top2=x["binding"][1]["name"], top2_util=x["binding"][1]["util"])
            for x in pts]
    with open(out / "plat.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=6)
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
    todo.sort(key=lambda k: not k.startswith("yupu"))          # Okpo is slow: first
    prog = R.Progress(out / "progress.json", len(cfg["keys"]), len(cfg["keys"]) - len(todo), a.procs, cfg["title"],
                      "platform_size")
    stop = threading.Event()
    threading.Thread(target=prog.heartbeat, args=(stop, 60.0), daemon=True).start()
    for key, secs in R.slot_imap(work, [(k, str(out)) for k in todo], min(a.procs, max(1, len(todo))), initializer=R.pin,
                                 initargs=(list(range(1, os.cpu_count())),), slots_dir=a.slots, tag="platform_size"):
        prog.log(f"{key} ({secs} s)", done=prog.d["done"] + 1)
        print(key, secs, flush=True)
    stop.set()
    rows = summarize(cfg, out)
    prog.write(state="done", note=f"{len(rows)} rows")
    for r in rows:
        print(r["yard"], r["plat_w"], r["plat_len"], round(r["T1_sys"], 1), f"{100 * r['rel']:+.2f}%", r["top1"],
              [round(v, 2) for v in (r["price_k0"], r["price_k1"], r["price_k2"])], [round(v, 3) for v in (r["util_k0"], r["util_k1"], r["util_k2"])],
              "|", r["top2"], round(r["top2_util"], 3), flush=True)


if __name__ == "__main__":
    main()

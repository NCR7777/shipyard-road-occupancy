"""System bound lambda^S and its binding elements (top 10) of every intervention scenario, written as in
conserved_bounds.compute: task law conserved_mapping.system_mix, the empty tail truncated by Eq. (5). Two yards x the scenarios
of intervention_net.py (SCEN and the two scenarios of the bound only, BOUND_ONLY) x 16 h. The base scenarios equal the system
bounds of the main case (Okpo 1,813.29, Yantai 1,247.60 tasks/16 h; results/main_case/bounds.csv).

Run:   python experiments/intervention_bounds.py --out <dir> [--procs N] [--slots DIR]
(--slots: machine-wide CPU slots as in experiments/runner.py, default the environment variable CPU_SLOTS, off when neither is
given). Writes to <dir>: config.json, progress.json, points/ (one JSON per scenario, a cache for resuming), points.csv (one row
per scenario: T1_sys and the top three binding elements with name, resource, utilisation and price) and binding.csv (the top
10 of every scenario). points.csv is released as results/interventions/bound_points.csv; the scenario table of the paper,
results/interventions/bounds.csv, is drawn from it. experiments/intervention_dispatch.py and intervention_subnet.py read the
binding elements of each scenario from this points.csv (--bounds).
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import runner as F                           # noqa: E402
import intervention_net as N                 # noqa: E402


def code_sha():
    h = hashlib.sha256()
    for d in ("bounds", "occupancy"):
        for f in sorted((ROOT / d).glob("*.py")):
            h.update(f"{d}/{f.name}".encode() + f.read_bytes())
    for n in ("demand_mapping.py", "conserved_mapping.py", "intervention_net.py", "intervention_bounds.py", "capacity_bounds.py"):
        h.update(f"experiments/{n}".encode() + (ROOT / "experiments" / n).read_bytes())
    return h.hexdigest()


def make_config():
    keys = [f"sys|{y}|{s}" for y, ss in N.SCEN.items() for s in list(ss) + list(N.BOUND_ONLY.get(y, {}))]
    return dict(task="intervention_bounds", title="system bounds and binding elements of the intervention scenarios", keys=keys,
                scenarios=N.SCEN, bound_only=N.BOUND_ONLY, load_min=10, day_h=16, code_sha256=code_sha())


def compute(key):
    import conserved_mapping
    from bounds.system import SystemT1, stop_ret
    from capacity_bounds import facility, res_info
    _, y, s = key.split("|")
    p = N.params(y, s)
    net = N.net(y, p)
    t = time.time()
    st = SystemT1(net, p, conserved_mapping.system_mix(net, p))
    v, b = st.value(top=10, ret=stop_ret(net, p["plat_len"] + p["long_clear"]))
    rows = [dict(name=x["name"], res="|".join(map(str, x["res"])), price_per_h=x["price_per_h"], util=x["util"],
                 facility=facility(net, tuple(x["res"])), **res_info(net, st.cap, st.size, p, tuple(x["res"]))) for x in b]
    return dict(key=key, yard=y, scen=s, T1_sys=v, binding=rows, secs=round(time.time() - t, 1))


def work(args):
    key, out = args
    r = compute(key)
    F.write_json(Path(out) / "points" / (key.replace("|", "_") + ".json"), r)
    return key, r["secs"]


def summarize(cfg, out):
    pts = [json.loads((out / "points" / (k.replace("|", "_") + ".json")).read_text(encoding="utf-8")) for k in cfg["keys"]]
    rows = [dict(yard=x["yard"], scen=x["scen"], T1_sys=x["T1_sys"], **{f"top{i + 1}": x["binding"][i]["name"] for i in range(3)},
                 **{f"res{i + 1}": x["binding"][i]["res"] for i in range(3)}, **{f"util{i + 1}": x["binding"][i]["util"] for i in range(3)},
                 **{f"price{i + 1}": x["binding"][i]["price_per_h"] for i in range(3)}, secs=x["secs"]) for x in pts]
    binding = [dict(yard=x["yard"], scen=x["scen"], rank=i + 1, **r) for x in pts for i, r in enumerate(x["binding"])]
    for name, rs in (("points.csv", rows), ("binding.csv", binding)):
        with open(out / name, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, list(rs[0]))
            w.writeheader()
            w.writerows(rs)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--slots", default=os.environ.get("CPU_SLOTS"), help="directory of machine-wide CPU slots (cpu_slots.Slots); "
                                                                         "default: environment variable CPU_SLOTS, else off")
    a = ap.parse_args()
    cfg, out = make_config(), Path(a.out)
    if (out / "config.json").exists() and json.loads((out / "config.json").read_text(encoding="utf-8")) != json.loads(json.dumps(cfg)):
        sys.exit(f"{out} holds a run of another configuration")
    (out / "points").mkdir(parents=True, exist_ok=True)
    F.write_json(out / "config.json", cfg)
    pf = lambda k: out / "points" / (k.replace("|", "_") + ".json")
    todo = sorted((k for k in cfg["keys"] if not pf(k).exists()), key=lambda k: "|yupu|" not in k)
    prog = F.Progress(out / "progress.json", len(cfg["keys"]), len(cfg["keys"]) - len(todo), a.procs, cfg["title"], cfg["task"])
    for key, secs in F.slot_imap(work, [(k, str(out)) for k in todo], min(a.procs, max(1, len(todo))), slots_dir=a.slots,
                                 tag="intervention_bounds"):
        prog.log(f"{key} ({secs} s)", done=prog.d["done"] + 1)
    for r in summarize(cfg, out):
        print(r["yard"], r["scen"], round(r["T1_sys"], 2), r["top1"], round(r["util1"], 3), "|", r["top2"], round(r["util2"], 3), "|",
              r["top3"], round(r["util3"], 3), flush=True)
    prog.write(state="done", processes=0)


if __name__ == "__main__":
    main()

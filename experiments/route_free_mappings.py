"""Route-free bounds and certificates under the demand mappings: the route-free bound, the route margin and the through share
of the binding resource for the conserved mapping of the main case (the columns Route-free, Route margin and Through of the
table of bounds, and the upper end "if vehicles could reroute" of the main-case rows of the capacity interval), and the
per-resource quantities of the chain load of the main-case curve. Linear programmes only, no simulation.

Route-free bounds: two yards x resource granularity (whole roads / dock roads sectioned at the stops) x demand mapping
(conserved = conserved_mapping.system_mix, q = 0; proportional = the proportional task law, for comparison), loading and
unloading 10 min, 16 h (24 h: x 1.5). As in capacity_bounds:
  system bound       bounds.system.SystemT1, the empty-run tail clearance truncated at the stop roads (stop_ret, as in
                     conserved_bounds); also the untruncated value;
  route-free bound   bounds.route_free.RouteFree.solve: column generation with a Lagrangian certificate (mu+, nu') from the
                     dual prices of every round; T1pp = the least certificate over the rounds, master = the value of the last
                     master problem, gap = (T1pp - master) / master; R = the first master problem (loaded runs on shortest
                     paths);
  back-substitution  capacity_bounds.backsub: the route split of the optimal solution of T1'', coefficients put back to the
                     unrelaxed T1', empty flows re-optimised -> a feasible value (the lower end of the route margin);
  through share      capacity_bounds.readout: the share of the loaded flow on the binding resource with neither origin nor
                     destination on it (Through), the remaining utilisation on the alternative routes and their extra length.
  route margin = [feasible / system bound - 1, T1pp / system bound - 1] (system bound truncated: the System column).
Chain load: chain_load.compute applied unchanged to the conserved mapping (cases yupu_cons, yantai_cons; only inside this
  process demand_mapping.system_mix is routed to conserved_mapping.system_mix and the stored reference value is the
  conserved lambda^S of conserved_bounds; chain_load.py itself is not changed); the output has the format of chain_load.

Run:   python experiments/route_free_mappings.py --out <dir> --conserved-scan <scan.csv>
                [--procs 10] [--slots DIR]
--conserved-scan: scan.csv of conserved_bounds or the released results/main_case/scan.csv; its q = 0 rows give the stored
lambda^S of the conserved mapping (16 h, whole roads), the reference of the chain-load cases.
Writes, case by case and resumable, <dir>/config.json, progress.json, points/<key>.json, and points.json (every field of the
route-free cases), chain.json (the two chain-load cases) and route_free.csv.
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
import conserved_bounds as TB                # noqa: E402

YARDS = ("yupu", "yantai")
MAPS = ("conserved", "proportional")
SECS = ("whole", "stops")
# stored table read by the chain-load cases (set in every worker by init): conserved = scan.csv of conserved_bounds
REFS = {}


def keys():
    return [f"rf|{y}|{m}|{s}" for y in YARDS for m in MAPS for s in SECS] + [f"chain|{y}" for y in YARDS]


def make_config():
    return dict(title="Route-free bounds and chain load of the conserved mapping", yards=list(YARDS), maps=list(MAPS),
                secs=list(SECS), load_min=10, params=P.DEFAULTS, keys=keys(), inputs=B.make_config()["inputs"])


def params(m, sec):
    p = dict(P.DEFAULTS, **({"demand_map": "conserved"} if m == "conserved" else P.SCENARIOS["main"]))
    return dict(p, section="stops") if sec == "stops" else p


def conserved_lam(y):
    """The stored lambda^S of the conserved mapping (16 h, whole roads, q = 0): the q = 0 row of scan.csv of conserved_bounds
    or of the released results/main_case/scan.csv."""
    return TB.q0_lam(REFS["conserved"])[y]


def rf_case(y, m, sec):
    from bounds.route_free import RouteFree
    from bounds.system import SystemT1, scenario_mix, stop_ret
    import capacity_bounds as T
    import conserved_mapping
    p = params(m, sec)
    net = B.net_of(y, p)
    t0 = time.time()
    mix = conserved_mapping.system_mix(net, p) if m == "conserved" else scenario_mix(net, p)[0]
    st = SystemT1(net, p, mix)
    v, b = st.value(top=5, ret=stop_ret(net, p["plat_len"] + p["long_clear"]))
    v0 = st.value(top=1)[0]
    rf = RouteFree(net, p, mix)
    sol = rf.solve(top=5)
    t_rf = time.time() - t0
    bs = T.backsub(net, p, st, rf, sol["sol"], mix)
    rd = T.readout(net, st, rf)
    cap, size = net.capacities(p)
    typ = lambda r: T.res_info(net, cap, size, p, tuple(r))["type"]
    pick = lambda lst: [dict(name=x["name"], res="|".join(map(str, x["res"])), type=typ(x["res"]), facility=T.facility(net, tuple(x["res"])),
                             price_per_h=x["price_per_h"], util=x["util"]) for x in lst]
    return dict(yard=y, map=m, section=sec, T1_sys=v, T1_sys_notrunc=v0, binding=pick(b),
                R=sol["R"], master=sol["master"], T1pp=sol["T1pp"], gap=sol["gap"], iters=sol["iters"], cols=sol["cols"], mu_min=sol["mu_min"],
                dual_infeas=sol["dual_infeas"], history=sol["history"], lp=sol["lp"], binding_pp=pick(sol["binding"]), detour=sol["detour"],
                rf_secs=round(t_rf, 1), backsub=bs, readout=dict({k: x for k, x in rd.items() if k != "res"}, res="|".join(map(str, rd["res"])),
                                                                   type=typ(rd["res"])))


def chain_case(y):
    """chain_load.compute applied to the conserved mapping (see the module docstring): the patch holds only inside this
    process and is undone afterwards."""
    import demand_mapping
    import conserved_mapping
    import chain_load as C
    name = f"{y}_cons"
    C.SCENS[name] = (y, {"demand_map": "conserved"})
    mix0, ref0 = demand_mapping.system_mix, C.reference
    demand_mapping.system_mix = lambda net, p: (conserved_mapping.system_mix(net, p) if p.get("demand_map") == "conserved"
                                                else mix0(net, p))
    C.reference = lambda n: (conserved_lam(y), str(REFS["conserved"])) if n == name else ref0(n)
    try:
        return C.compute(name)
    finally:                       # SCENS is restored too (chain_load.make_config lists its cases)
        demand_mapping.system_mix, C.reference = mix0, ref0
        C.SCENS.pop(name)


def compute(key):
    t = time.time()
    kind, y, *rest = key.split("|")
    r = rf_case(y, *rest) if kind == "rf" else chain_case(y)
    return dict(r, key=key, secs=round(time.time() - t, 1))


def init(cores, refs):
    R.pin(cores)
    REFS.update(refs)


def work(args):
    key, out = args
    r = compute(key)
    R.write_json(Path(out) / "points" / (key.replace("|", "_") + ".json"), r)
    return key, r["secs"]


def summarize(cfg, out):
    """points/ -> points.json (route-free cases), chain.json (chain-load cases, in the format of chain_load) and
    route_free.csv."""
    pts = {x["key"]: x for x in (json.loads(f.read_text(encoding="utf-8")) for f in sorted((out / "points").glob("*.json")))}
    rf = [pts[k] for k in cfg["keys"] if k.startswith("rf|")]
    R.write_json(out / "points.json", rf)
    (out / "chain.json").write_text(json.dumps({x["name"]: {k: v for k, v in x.items() if k != "key"} for k, x in pts.items()
                                                if k.startswith("chain|")}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    rows = []
    for x in rf:
        T, bs, rd = x["T1_sys"], x["backsub"], x["readout"]
        rows.append(dict(yard=x["yard"], map=x["map"], section=x["section"], T1_sys=T, T1_sys_notrunc=x["T1_sys_notrunc"], R=x["R"],
                         feasible=bs["feasible"], master=x["master"], T1pp=x["T1pp"], gap=x["gap"], margin_lo_rel=bs["feasible"] / T - 1,
                         margin_hi_rel=x["T1pp"] / T - 1, pass_share=rd["pass_share"], alt_slack=rd["alt_slack"], alt_extra_m=rd["alt_extra_m"],
                         top1=x["binding"][0]["name"], top1_type=x["binding"][0]["type"], top1_pp=x["binding_pp"][0]["name"],
                         top1_pp_type=x["binding_pp"][0]["type"], readout_res=rd["name"], detour_share=x["detour"]["share"],
                         extra_m=x["detour"]["extra_m"], backsub_rows_over_1=bs["rows_over_1"], backsub_ratio_max=bs["ratio_max"],
                         iters=x["iters"], cols=x["cols"], mu_min=x["mu_min"], dual_infeas=x["dual_infeas"], rf_secs=x["rf_secs"], secs=x["secs"]))
    with open(out / "route_free.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--conserved-scan", required=True,
                    help="scan.csv of conserved_bounds or the released one (reference lambda^S of the chain-load cases)")
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
    todo.sort(key=lambda k: ("|yupu" not in k, "|stops" not in k))          # Okpo sectioned takes longest: first
    prog = R.Progress(out / "progress.json", len(cfg["keys"]), len(cfg["keys"]) - len(todo), a.procs, cfg["title"],
                      "route_free_mappings")
    stop = threading.Event()        # long items (column generation): refresh the progress file every minute
    threading.Thread(target=prog.heartbeat, args=(stop, 60.0), daemon=True).start()
    for key, secs in R.slot_imap(work, [(k, str(out)) for k in todo], min(a.procs, max(1, len(todo))), initializer=init,
                                 initargs=(list(range(1, os.cpu_count())), dict(conserved=a.conserved_scan)),
                                 slots_dir=a.slots, tag="route_free_mappings"):
        prog.log(f"{key} ({secs} s)", done=prog.d["done"] + 1)
        print(key, secs, flush=True)
    stop.set()
    rows = summarize(cfg, out)
    prog.write(state="done", note=f"{len(rows)} rows")
    for r in rows:
        print(r["yard"], r["map"], r["section"], round(r["T1_sys"], 1), round(r["feasible"], 1), round(r["T1pp"], 1),
              f"{100 * r['margin_lo_rel']:.1f}–{100 * r['margin_hi_rel']:.1f}%", f"through {100 * r['pass_share']:.1f}%", r["top1"], flush=True)


if __name__ == "__main__":
    main()

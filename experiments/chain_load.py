"""Chain load: the per-resource quantities of the chain-load statistic, the mechanism behind the screen of demand over the
bound. Linear programmes only, no simulation.

For each case (two yards x three road-width levels, main flow mix; Okpo under the intra mapping; whole roads, 10-min
handling; 16 h, plus both yards at 24 h, a check that the length of the working day does not matter) the system bound is
solved once (bounds.system.SystemT1, the empty-run tail clearance truncated at the stop roads, as in mapping_bounds), giving
  lam            lambda^S, checked against the stored values (bound-sensitivity table, mapping_bounds);
  util_lp[k][r]  utilisation of resource r in row k (k = 0 width-weighted, 1, 2 packing rows) of the LP solution at lambda^S
                 = activity / H; mu the dual prices (> 0 on binding rows);
  beta[k][r]     utilisation of row k at demand = lambda^S, empty runs by the minimum empty flow (as the step of the fleet
                 bound in occupancy.bound.bounds, one unit per task); at demand n the utilisation is (n / lambda^S) x beta,
                 independent of the length of the working day (lambda^S scales with H);
  pi[r]          probability that the loaded chain of a task of the task law (origin stop, loaded route, destination stop:
                 the resources of bound.task_windows) passes r;
  eps[r]         expected number of passes of r by the empty chain of a task under the minimum empty flow: a road counts
                 on leaving it (arc head a map node), once more when the empty run ends at a roadside stop on it; a junction
                 counts on entering it, entrance nodes do not count (as the empty-run columns of the LP);
  dur[r]         loaded occupancy time of a task on r (s, not weighted by shares; the windows of a-bar); dur / pi is the
                 occupancy time per pass;
  cycle0_s       cycle without waiting = loading + unloading + E[loaded drive] + minimum empty run (s per task, written as
                 c-bar of the fleet bound);
  *_d            the same quantities under dispatch-like empty runs (pairing_flow: the previous destination and the next
                 origin paired independently by the task law, on the shortest paths of the simulation): beta_d (empty
                 occupancy by this flow), eps_d, dur_d (empty occupancy time on each resource), empty_d_s, cycle0_d_s.
                 Full-path reservation books the empty leg together with the loaded one, so the chain includes the empty
                 leg; the minimum empty flow is only the lower end inside the bound (0.6-1.2 min of empty driving per task,
                 against 3.6-9.5 min in the simulations with few vehicles, checked with the simulated empty-run times).

Run:   python experiments/chain_load.py --out <dir> --sensitivity <bound-sensitivity table>
                --mapping-bounds <system.csv of mapping_bounds> [--procs 9] [--slots DIR]
--sensitivity: CSV with the columns yard, scen, section, variant, T1_sys (the stored one is results/bounds/sensitivity.csv;
main flow mix, whole roads, 16 h, variants base and w+-1.5); --mapping-bounds: system.csv of mapping_bounds or the
released results/intra_area/bounds.csv (intra, 16 h).
Writes, case by case and resumable, <dir>/config.json, progress.json and parts/<case>.json.
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

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import runner as R                           # noqa: E402
from occupancy import params as P            # noqa: E402

SCENS = {"yupu_base": ("yupu", {}), "yupu_w-1.5": ("yupu", {"width_delta": -1.5}), "yupu_w+1.5": ("yupu", {"width_delta": 1.5}),
         "yantai_base": ("yantai", {}), "yantai_w-1.5": ("yantai", {"width_delta": -1.5}), "yantai_w+1.5": ("yantai", {"width_delta": 1.5}),
         "yupu_intra": ("yupu", {"demand_map": "intra"}),
         "yupu_base_d24": ("yupu", {"day_s": 24 * 3600.0}), "yantai_base_d24": ("yantai", {"day_s": 24 * 3600.0})}
# stored tables the check of lambda^S reads (set in every worker by init): sensitivity = the bound-sensitivity table (main
# flow mix, whole roads, 16 h; base / w+-1.5), mapping = system.csv of mapping_bounds (intra mapping; 16 h, 24 h)
REFS = {}


def init(refs):
    REFS.update(refs)


def params(name):
    return dict(P.DEFAULTS, **SCENS[name][1])


def make_config():
    ys = sorted({y for y, _ in SCENS.values()})
    return dict(title="Per-resource quantities of the chain load: utilisation of the system bound, pass probabilities of "
                      "the chains and empty passes", scens={k: list(v) for k, v in SCENS.items()}, params=P.DEFAULTS,
                inputs={y: dict(network=P.YARDS[y]["network"], roles=P.YARDS[y]["roles"]) for y in ys})


def reference(name):
    """The stored lambda^S (tasks/day, for the working day of the case)."""
    y, kw = SCENS[name]
    if kw.get("demand_map"):
        r = {(x["yard"], x["map"], x["load_min"], x["day_h"]): float(x["T1_sys"])
             for x in csv.DictReader(open(REFS["mapping"], encoding="utf-8-sig"))}
        return r[(y, "intra", "10", "16")], str(REFS["mapping"])
    var = {-1.5: "w-1.5", 1.5: "w+1.5"}.get(kw.get("width_delta"), "base")
    rows = csv.DictReader(open(REFS["sensitivity"], encoding="utf-8-sig"))
    v = next(float(r["T1_sys"]) for r in rows
             if (r["yard"], r["scen"], r["section"], r["variant"]) == (y, "main", "whole", var))
    return v * params(name)["day_s"] / 57600.0, str(REFS["sensitivity"])


def chain_sets(net, p, cap, size, mix, ri):
    """pi: probability that the loaded chain (the resources of task_windows, tight windows) passes each resource; E[loaded
    drive time] (s). The resource set and the drive time do not depend on B and Lv: each task takes the first item of bl."""
    from occupancy.bound import task_windows
    pi, tl = np.zeros(len(ri)), 0.0
    for w, t, bl in mix:
        by, TL = task_windows(net, p, cap, size, dict(t, B=bl[0][0][0], Lv=bl[0][0][1]), tight=True)
        pi[[ri[r] for r in by]] += w
        tl += w * TL
    return pi, tl


def empty_visits(net, f1, term, ri):
    """eps: expected number of passes of each resource by the empty chain of a task (see the module docstring);
    term = {node: empty flow ending there}."""
    eps = np.zeros(len(ri))
    for a, (_, hd, _, rid, _) in enumerate(net.arc):
        if f1[a] <= 0:
            continue
        if not hd.startswith("S|"):
            eps[ri[("r", rid)]] += f1[a]
        if hd in net.node_res and hd not in net.access:
            eps[ri[("n", hd)]] += f1[a]
    road_of = {hd: rid for _, hd, _, rid, _ in net.arc if hd.startswith("S|")}
    for n, q in term.items():
        if n.startswith("S|") and q > 0:              # empty run ending at a roadside stop: the last road has no leaving arc
            eps[ri[("r", road_of[n])]] += q
    return eps


def pairing_flow(net, mix):
    """Dispatch-like empty runs (independent pairing): the empty vehicle drives from the destination of the previous task
    (by the destination distribution of the task law) to the origin of this task (by the origin distribution), the two
    independent, on the same shortest path as the simulation (net.path, obeying the turn permissions). Returns the arc flow
    per task and the end amounts {origin node: amount}. With few vehicles "the nearest free vehicle" is about a random one;
    the mean is checked with the simulated empty-run times."""
    po, pd = collections.Counter(), collections.Counter()
    for w, t, _ in mix:
        if not t.get("lift"):
            po[net.stops[t["o"]]["node"]] += w
            pd[net.stops[t["d"]]["node"]] += w
    f, term = np.zeros(len(net.arc)), collections.Counter()
    for d, qd in pd.items():
        for o, qo in po.items():
            if d != o:
                np.add.at(f, net.path(d, o), qd * qo)
                term[o] += qd * qo
    return f, term


def compute(name):
    from bounds.system import SystemT1, empty_block, scenario_mix, stop_ret
    from occupancy.bound import DFF_K, res_name
    from occupancy.network import Net
    import demand_mapping as M
    y, kw = SCENS[name]
    p, Y, t0 = params(name), P.YARDS[y], time.time()
    net = Net(Y["network"], p, Y["roles"])
    mix = M.system_mix(net, p) if kw.get("demand_map") else scenario_mix(net, p)[0]
    st = SystemT1(net, p, mix)
    ret = stop_ret(net, p["plat_len"] + p["long_clear"])
    lp = {}
    lam, top = st.value(top=3, ret=ret, lp_out=lp)
    nk, nr, H = len(DFF_K), len(st.res_list), p["day_s"]
    util_lp = (lp["act"] / lp["H"]).reshape(nk, nr)
    mu = lp["mu"][:nk * nr].reshape(nk, nr) * 3600          # tasks/day per extra h/day
    Aeq, Aub, res_list = empty_block(net, p, st.cap, st.size, st.dummy, ret)
    assert res_list == st.res_list
    lens = np.array([a[2] for a in net.arc])
    mc = linprog(lens / p["v_empty"], A_eq=Aeq, b_eq=st.L.on, bounds=[(0, None)] * len(lens), method="highs")
    if mc.status != 0:
        raise RuntimeError(f"linear programme of the minimum empty flow failed: {mc.message}")
    abar = st.L.abar(st.cap)                                  # loaded occupancy per task (s x share), rows k
    beta = lam * (abar + (Aub @ mc.x).reshape(nk, nr)) / H
    ri = {r: i for i, r in enumerate(st.res_list)}
    pi, tl = chain_sets(net, p, st.cap, st.size, mix, ri)
    eps = empty_visits(net, mc.x, {n: q for n, q in zip(st.L.nodes, st.L.on) if q > 0}, ri)
    # loaded occupancy time per task on each resource (s, not weighted by shares)
    dur = sum(x * sum(q for _, q in st.L.PB[key]) for key, x in st.L.V.items())
    fd, term = pairing_flow(net, mix)                          # dispatch-like empty runs (see pairing_flow)
    eps_d = empty_visits(net, fd, term, ri)
    occ_d = (Aub @ fd).reshape(nk, nr)
    beta_d = lam * (abar + occ_d) / H
    se = np.array([min(1.0, net.demand(r, p["plat_w"], False, p, st.cap) / st.cap[r]) if r[0] in "rn" else 0.0 for r in st.res_list])
    # empty occupancy time on each resource (s per task, as the empty-run columns of the LP)
    dur_d = np.where(se > 0, occ_d[0] / np.where(se > 0, se, 1.0), 0.0)
    empty_d = float(lens @ fd / p["v_empty"])
    ref, src = reference(name)
    W = lambda r: (net.road[r[1]]["W"] + p["width_delta"]) if r[0] == "r" and net.road[r[1]]["W"] is not None else None
    return dict(name=name, yard=y, kw=kw, day_s=H, lam=lam, lam_ref=ref, lam_ref_src=src, lam_same=lam == ref, lam_rel=lam / ref - 1,
                top=[dict(name=x["name"], price_per_h=x["price_per_h"], util=x["util"]) for x in top],
                empty_s=float(mc.fun), loaded_s=tl, cycle0_s=p["load_s"] + p["unload_s"] + tl + float(mc.fun),
                empty_d_s=empty_d, cycle0_d_s=p["load_s"] + p["unload_s"] + tl + empty_d, eps_d=eps_d.tolist(), dur_d=dur_d.tolist(),
                beta_d=beta_d.tolist(), max_beta_d=beta_d.max(axis=1).tolist(),
                loaded_pi_sum=float(pi.sum()), eps_sum=float(eps.sum()), eps_d_sum=float(eps_d.sum()), dur=dur.tolist(),
                res=[list(r) for r in st.res_list], names=[res_name(net, r) for r in st.res_list], W=[W(r) for r in st.res_list],
                cap=[st.cap[r] for r in st.res_list], pi=pi.tolist(), eps=eps.tolist(), beta=beta.tolist(), util_lp=util_lp.tolist(),
                mu=mu.tolist(), secs=round(time.time() - t0, 1))


def job(a):
    out, name = a
    R.pin(list(range(1, os.cpu_count())))
    r = compute(name)
    R.write_json(out / "parts" / f"{name}.json", r)
    return name, r["secs"], r["lam_same"]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--sensitivity", required=True, help="bound-sensitivity table (CSV), for the check of lambda^S")
    ap.add_argument("--mapping-bounds", required=True, help="system.csv of mapping_bounds, for the check of lambda^S")
    ap.add_argument("--procs", type=int, default=len(SCENS))
    ap.add_argument("--slots", default=os.environ.get("CPU_SLOTS"), help="optional CPU-slot directory shared by runs "
                    "(default: environment variable CPU_SLOTS; without it an ordinary process pool)")
    a = ap.parse_args()
    cfg = make_config()
    out = Path(a.out)
    if (out / "config.json").exists() and json.loads((out / "config.json").read_text(encoding="utf-8")) != json.loads(json.dumps(cfg, ensure_ascii=False)):
        sys.exit(f"refusing to reuse {out}: its config.json differs from the current configuration")
    (out / "parts").mkdir(parents=True, exist_ok=True)
    R.write_json(out / "config.json", cfg)
    print(f"run directory: {out}", flush=True)
    todo = [s for s in SCENS if not (out / "parts" / f"{s}.json").exists()]
    todo.sort(key=lambda s: not s.startswith("yupu"))          # Okpo is slow: first
    prog = R.Progress(out / "progress.json", len(SCENS), len(SCENS) - len(todo), a.procs, cfg["title"], "chain_load")
    stop = threading.Event()
    threading.Thread(target=prog.heartbeat, args=(stop, 60.0), daemon=True).start()
    refs = dict(sensitivity=a.sensitivity, mapping=a.mapping_bounds)
    for name, secs, same in R.slot_imap(job, [(out, s) for s in todo], a.procs, initializer=init, initargs=(refs,),
                                        slots_dir=a.slots, tag="chain_load"):
        prog.log(f"{name} ({secs:.0f} s, lambda^S {'equal to' if same else 'differs from'} the stored value)",
                 done=prog.d["done"] + 1)
        print(name, secs, same, flush=True)
    stop.set()
    prog.write(state="done")


if __name__ == "__main__":
    main()

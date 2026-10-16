"""Sharpened lower bound of the completion time on the sub-network instances, computed afterwards (the point files are not
changed): reads <run>/points/ of experiments/subnet_exact.py, computes for every point the occupancy-order bound chain2 and
its parts and writes them to <run>/bounds2.json ({key: {chain2: seconds or None, parts: {...}}}, plus the SHA-256 of this
script), which the aggregation of subnet_exact.py reads.
--converge: for the three seeds of the Yantai doors, the same bound, lambda_W^N and full-path reservation at N = 60, 120, 240
(the convergence curve) -> <run>/converge.json.

Run:   python experiments/subnet_bounds.py <run> [--converge] [--slots DIR]
<run> is the --out directory of a run of subnet_exact.py made first (its point files are not released).
The process is pinned to the cores other than core 0; with a CPU slot directory (--slots or CPU_SLOTS) it takes one slot.

Occupancy-order bound (generalises exact.model.chain_bound):
  on the binding resource res each task has at most one loaded window: the origin window (the origin stop resource is res,
  type O) or a window of its loaded run (passing, or the destination on res, type L); those of more than half the width (no
  two fit together) form a list J, which do not overlap on res and line up in time.
  The origin window of type O counts from the arrival, by Definition 1 ([t1 + TE, t2 + oend]): shorter than in the model
  (the booking of full-path reservation, from the entry into the last road), so the bound holds for every admissible
  schedule that waits only at the origin, not just for this booking.
  Minimum gap gap(i, j, l) between consecutive windows i -> j: looking at these two tasks only, the loaded windows of i after
  it leaves res (relative to t2_i) and the windows of j before it enters res -- for type O the empty run from the departure
  position l (relative to t1_j), for type L the windows of its loaded run before res (relative to t2_j) -- cannot be held at
  the same time where they do not fit together on one resource.
  Departure-position quota: the vehicle of j departs from l, the destination of its previous task or its start position;
  each task's destination serves at most once, each start position at most as often as vehicles start there.
  So the sum of the gaps over j other than the first >= min over f of the assignment of J without f: each of these tasks
  takes one departure-position slot at cost min_i gap(i, j, l) (i ranges over the tasks of J other than j).
  W >= earliest start + sum of the shortest window lengths + that sum + shortest time from leaving res to the end of
  unloading.
  Empty windows on res, windows of at most half the width, a second window of a task and slots taken by tasks outside J are
  not counted: fewer constraints only loosen the bound, which still holds.
"""
import argparse
import collections
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import runner as F                        # noqa: E402
import subnet_exact as T                  # noqa: E402
from occupancy.network import EPS         # noqa: E402


def forbid(Wi, Wj, cap):
    """Forbidden zone of delta (anchor of j - t2 of i): where window Wi of i and window Wj of j do not fit together on one
    resource, delta is not in (a1 - b2, a2 - b1)."""
    return sorted((a1 - b2, a2 - b1) for r, a1, a2, d in Wi for r2, b1, b2, e in Wj if r == r2 and d + e > cap[r] + EPS)


def first_ok(lo, bad):
    for a, b in bad:
        if a < lo < b:
            lo = b
    return lo


def chain2(inst, K, res):
    """Returns dict(W, J, r0, q, base, order, gap_min, gap_max, gap_mean, start_only_slots_used) or None (see the module
    docstring). base = sum of the shortest window lengths (type O = loading + oend, type L = window length); order = the sum
    of the gaps of the assignment; W = r0 + base + order + q."""
    N, load, cap = len(inst.tasks), inst.p["load_s"], inst.cap
    onode = lambda j: inst.net.stops[inst.tasks[j]["o"]]["node"]
    dnode = lambda j: inst.net.stops[inst.tasks[j]["d"]]["node"]
    info = {}
    for j in range(N):
        c0 = inst.chain(j, onode(j))
        mine = [w for w in c0["WL"] if w[0] == res]
        if c0["so"] == res and not mine and 2 * c0["d_so"] > cap[res] + EPS:
            info[j] = ("O", None)
        elif c0["so"] != res and len(mine) == 1 and 2 * mine[0][3] > cap[res] + EPS:
            info[j] = ("L", mine[0])
    J = sorted(info)
    if len(J) < 2:
        return None
    sig = lambda i: (inst.tasks[i]["o"], inst.tasks[i]["d"], inst.tasks[i]["B"], inst.tasks[i]["Lv"])
    rep = {}                                   # signature -> one representative task; cnt: number of tasks of each signature in J
    for i in J:
        rep.setdefault(sig(i), i)
    cnt = collections.Counter(sig(i) for i in J)

    def after(i):                              # time at which i leaves res (relative to t2_i) and the loaded windows after it
        ci = inst.chain(i, onode(i))
        end = ci["oend"] if info[i][0] == "O" else info[i][1][2]
        return end, [w for w in ci["WL"] if w[0] != res and w[0][0] != "c" and w[2] > end - EPS]
    aft = {s: after(i) for s, i in rep.items()}
    starts = collections.Counter(inst.starts[:K])
    ends = collections.Counter(dnode(j) for j in range(N))
    locs = sorted(set(starts) | set(ends))
    slots = [l for l in locs for _ in range(min(len(J), starts[l] + ends[l]))]
    big = 1e9
    cost = np.full((len(J), len(slots)), big)
    base, r0, q = [], [], []
    memo = {}
    for row, j in enumerate(J):
        c0 = inst.chain(j, onode(j))
        if info[j][0] == "O":
            base.append(load + c0["oend"])
            q.append(inst.dur_l(j) - c0["oend"])
            cand = {l: inst.chain(j, l) for l in locs if l in inst.locs(j, K) or l == onode(j)}
            r0.append(inst.tasks[j]["release"] + min(c["TE"] for c in cand.values()))
        else:
            _, s, e, _ = info[j][1]
            base.append(e - s)
            q.append(inst.dur_l(j) - e)
            cand = {l: None for l in locs}
            r0.append(inst.tasks[j]["release"] + min(inst.chain(j, l)["TE"] for l in inst.locs(j, K)) + load + s)
        val = {}
        for l, c in cand.items():
            key = (sig(j), l if info[j][0] == "O" else None)
            if key not in memo:
                if info[j][0] == "O":                  # Definition 1: the door from the arrival (t1 + TE) on
                    start, Wj = c["TE"], [w for w in c["WE"] if w[0] != res]
                else:
                    start = info[j][1][1]
                    Wj = [w for w in c0["WL"] if w[0] != res and w[0][0] != "c" and w[1] < start + EPS]
                memo[key] = sorted((first_ok(e_i - start, forbid(W_i, Wj, cap)) + start - e_i, s_i)
                                   for s_i, (e_i, W_i) in aft.items())
            for g, s_i in memo[key]:                   # i ranges over the tasks of J other than j
                if cnt[s_i] - (s_i == sig(j)) > 0:
                    val[l] = g
                    break
        for col, l in enumerate(slots):
            if l in val:
                cost[row, col] = val[l]
    best, arg = math.inf, None
    for f in range(len(J)):                            # the first window f has no gap and takes no slot
        rows = [r for r in range(len(J)) if r != f]
        ri, ci = linear_sum_assignment(cost[rows])
        v = cost[rows][ri, ci].sum()
        if v < best:
            best, arg = v, (rows, ri, ci)
    rows, ri, ci = arg
    g = [cost[rows[a], b] for a, b in zip(ri, ci)]
    used = sum(1 for a, b in zip(ri, ci) if starts[slots[b]] and not ends[slots[b]])
    if best >= big:
        return None
    return dict(W=min(r0) + sum(base) + best + min(q), J=len(J), r0=min(r0), q=min(q), base=sum(base), order=best,
                gap_min=min(g), gap_max=max(g), gap_mean=sum(g) / len(g), start_only_slots_used=used)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", help="output directory of experiments/subnet_exact.py")
    ap.add_argument("--converge", action="store_true")
    ap.add_argument("--slots", default=None, help="CPU slot directory (default: environment variable CPU_SLOTS, else none)")
    a = ap.parse_args()
    F.pin(T.CORES)
    out = Path(a.run)
    T.OUT = out
    sd, slots = a.slots or os.environ.get("CPU_SLOTS"), None
    if sd:
        sys.path.insert(0, sd)
        from cpu_slots import Slots
        slots = Slots(f"subnet_bounds-{os.getpid()}")
        slots.want(1)
        while not slots.take():
            slots.want(1)
            time.sleep(3)
    from exact import model as X
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if a.converge:
        converge(out, sha, X)
    else:
        f = out / "bounds2.json"
        old = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        res = old.get("points", {}) if old.get("sha256") == sha else {}
        for p in sorted((out / "points").glob("*.json")):
            r = json.loads(p.read_text(encoding="utf-8"))
            if r["key"] in res:
                continue
            inst = T.instance(r["inst"], r["seed"], r["N"])
            lam, bind = X.lam_window(inst, r["opt"]["W"])
            pr = chain2(inst, r["K"], tuple(bind[0]["res"]))
            res[r["key"]] = dict(chain2=pr["W"] if pr else None, binding=bind[0]["name"], parts=pr)
            if slots:
                slots.want(1)                  # heartbeat
            print(r["key"], {k: (round(v, 1) if isinstance(v, float) else v) for k, v in (pr or {}).items()}, flush=True)
        F.write_json(f, dict(script="experiments/subnet_bounds.py", sha256=sha, points=res))
    if slots:
        slots.close()


def converge(out, sha, X):
    """Yantai doors: the first N tasks of one seed (N = 60, 120, 240, from the same release stream), start positions as at the
    point N = 60, K = 8."""
    name, K, rows = "yantai_paint", 8, []
    s = T.INSTANCES[name]
    for seed in T.SEEDS:
        i60 = T.instance(name, seed, 60)
        net, p = i60.net, i60.p
        ents = [e for e, v in net.ents.items() if v["name"].startswith(tuple(s["ents"]))]
        big = X.sample_tasks(net, p, seed, i60.roads, 240, ents)
        assert [t["src_id"] for t in big[:60]] == [t["src_id"] for t in i60.tasks]
        for N in (60, 120, 240):
            inst = X.Instance(f"{name}|{seed}|{N}", net, p, big[:N], i60.starts, i60.roads)
            mk, _, _ = X.run_fpr(inst, K)
            lam, bind = X.lam_window(inst, mk)
            pr = chain2(inst, K, tuple(bind[0]["res"]))
            W_lp = X.lp_finite(inst, K)
            thr = lambda W: N * T.DAY / W
            row = dict(seed=seed, N=N, lamN=lam, fpr=thr(mk), lp=thr(W_lp), chain=thr(pr["W"]), parts=pr,
                       steady=pr["base"] / (pr["base"] + pr["order"]), binding=bind[0]["name"])
            rows.append(row)
            print(seed, N, f"FPR/λ {row['fpr'] / lam:.3f}  LP/λ {row['lp'] / lam:.3f}  chain/λ {row['chain'] / lam:.3f}  "
                  f"steady base/(base+order) {row['steady']:.3f}  J {pr['J']}", flush=True)
    F.write_json(out / "converge.json", dict(script="experiments/subnet_bounds.py --converge", sha256=sha, rows=rows))


if __name__ == "__main__":
    main()

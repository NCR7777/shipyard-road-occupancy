"""Occupancy-order bound on the through-corridor sub-network (generalises chain2 of experiments/subnet_bounds.py: between two
consecutive windows it looks at all windows of the two tasks at fixed offsets), the vehicle integrality bound and the split of
the idle time on the corridor, used by experiments/corridor.py.

chain2 was written for the door type: between consecutive windows i -> j it looks only at "the loaded windows after i leaves
the binding resource res" and "the windows of j before it enters res" -- the earlier vehicle leaving the door and the later
one entering it pass the junction in front of the door one after the other. On a corridor (road010-6) there are two more
kinds of precedence:
  following in the same direction: both vehicles come from the same approach road, and the later one can enter that road
  only after the tail of the earlier one has left it (the windows of the two vehicles before res collide);
  alternating in opposite directions: the road the earlier vehicle takes after leaving res is the road the later one comes
  from (a window of i after res collides with a window of j before res; already in chain2).
This generalisation puts all windows of the two tasks that are fixed relative to an anchor into the forbidden zone of
gap(i, j):
  i (anchor t2_i): all windows of its loaded run (other than res, without cranes); where the origin stop resource is not res,
    also the stretch it must hold at least, [t2_i - loading, t2_i + oend] (Definition 1: the origin is held from the
    arrival, which is no later than t2 - loading);
  j of type L (anchor t2_j): the same; j of type O (anchor t1_j, the origin is res): the empty-run windows from the departure
    position l (as in chain2, assigned by the quota).
gap(i, j) = the minimum gap between the two windows under the condition that "the res window of j starts after the end of
the res window of i" and the other windows of the two fit together on every resource (the forbidden zones are open intervals,
passed from the lower limit one after the other). Only a part of the pairwise constraints is used, so W >= earliest start +
sum of the shortest window lengths + the sum of the gaps of the assignment + shortest time from leaving res to the end of
unloading holds for every admissible schedule that waits only at the origin (the argument of chain2). Empty windows on res
and windows of at most half the width are not counted (they only loosen the bound).

Run:   python experiments/subnet_gap.py --subnet <--out dir of a subnet_exact.py run> --out <dir> [KEY ...]
(default: the nine points of the through corridor). Reads <subnet>/points/<key>.json of a run of subnet_exact.py made first
(completion time of the best solution, whose binding resource is used; the point files are not released, the released table
of the paper's run is results/subnet/points.csv) and writes <dir>/bounds3.json.
"""
import argparse
import collections
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import runner as F                        # noqa: E402
import subnet_exact as T                  # noqa: E402
from subnet_bounds import first_ok, forbid   # noqa: E402
from occupancy.network import EPS         # noqa: E402


def chain3(inst, K, res):
    """Returns dict(W, J, r0, q, base, order, gap_min, gap_max, gap_mean, gaps_pos) or None (see the module docstring). The
    structure is that of subnet_bounds.chain2; only the forbidden zones take all windows of the two tasks at fixed offsets."""
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

    def fixed(i):
        """Windows of task i fixed relative to t2_i (other than res, without cranes), with the stretch the origin must hold
        at least."""
        ci = inst.chain(i, onode(i))
        ws = [w for w in ci["WL"] if w[0] != res and w[0][0] != "c"]
        if ci["so"] != res:
            ws.append((ci["so"], -load, ci["oend"], ci["d_so"]))
        return ws

    def res_end(i):
        ci = inst.chain(i, onode(i))
        return ci["oend"] if info[i][0] == "O" else info[i][1][2]

    W_i = {i: (res_end(i), fixed(i)) for i in J}
    starts = collections.Counter(inst.starts[:K])
    ends = collections.Counter(dnode(j) for j in range(N))
    locs = sorted(set(starts) | set(ends))
    slots = [l for l in locs for _ in range(min(len(J), starts[l] + ends[l]))]
    big = 1e9
    cost = np.full((len(J), len(slots)), big)
    base, r0, q = [], [], []
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
            if info[j][0] == "O":
                start, Wj = c["TE"], [w for w in c["WE"] if w[0] != res]
            else:
                start, Wj = info[j][1][1], fixed(j)
            g = min((first_ok(e_i - start, forbid(Wi, Wj, cap)) + start - e_i for i, (e_i, Wi) in W_i.items() if i != j), default=big)
            val[l] = g
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
    if best >= big:
        return None
    rows, ri, ci = arg
    g = [float(cost[rows[a], b]) for a, b in zip(ri, ci)]
    return dict(W=float(min(r0) + sum(base) + best + min(q)), J=len(J), r0=float(min(r0)), q=float(min(q)), base=float(sum(base)),
                order=float(best), gap_min=min(g), gap_max=max(g), gap_mean=sum(g) / len(g), gaps_pos=sum(x > 1e-6 for x in g))


def veh_bound(inst, K):
    """Vehicle integrality bound: some vehicle does at least ceil(N/K) tasks; each task needs at least c_j = empty run from
    some possible departure position to the origin + loading + loaded run + unloading, so W >= the sum of the ceil(N/K)
    smallest c_j. The vehicle-time row of the linear programme only gives the mean (sum c <= K·W); with few tasks this row is
    tighter."""
    N, load = len(inst.tasks), inst.p["load_s"]
    c = sorted(min(inst.chain(j, l)["TE"] for l in inst.locs(j, K)) + load + inst.dur_l(j) for j in range(N))
    return float(sum(c[:math.ceil(N / K)]))


def gap_split(inst, seq, res):
    """Split of the idle time of a schedule on the binding resource res: each gap between two consecutive holds (end of the
    earlier hold, start of the later one) is split into "the minimum gap that the pairwise precedence of these two tasks alone
    cannot avoid" (gap, as in chain3: windows of the two tasks fixed relative to their anchors do not fit together on another
    resource) and the rest.
    Anchors of the holds: windows of the empty run are anchored at t1, of the loaded run at t2; the origin window (the origin
    is res) starts at t1 and ends at t2. For the earlier hold the windows fixed at the anchor of its end are taken, for the
    later one those fixed at the anchor of its start.
    Returns dict(W, busy, idle, forced, vehicle, other, start, n_gaps) (seconds; vehicle = the two holds belong to the same
    vehicle, decided by the vehicle order; start = from 0 to the start of the first hold; other includes the tail from the
    last hold to completion). busy + forced + vehicle + other + start = W."""
    cap, load = inst.cap, inst.p["load_s"]
    occ = []                                    # (start, end, vehicle, task, end anchor, end offset, start anchor, start offset, position)
    W = 0.0
    for k, lst in seq.items():
        loc = inst.starts[int(k)]
        for j, t1, t2 in lst:
            c = inst.chain(j, loc)
            T = {1: t1, 2: t2}
            for r, a, b, d in c["WE"]:
                if r == res:
                    occ.append((t1 + a, t1 + b, k, j, 1, b, 1, a, loc))
            if c["so"] == res:
                occ.append((t1 + c["ao"], t2 + c["oend"], k, j, 2, c["oend"], 1, c["ao"], loc))
            for r, a, b, d in c["WL"]:
                if r == res:
                    occ.append((t2 + a, t2 + b, k, j, 2, b, 2, a, loc))
            W = max(W, t2 + inst.dur_l(j))
            loc = inst.net.stops[inst.tasks[j]["d"]]["node"]

    def fixed(j, loc, anchor):
        c = inst.chain(j, loc)
        if anchor == 1:
            ws = [w for w in c["WE"] if w[0] != res]
            if c["so"] != res:
                ws.append((c["so"], c["ao"], c["TE"] + load + c["oend"], c["d_so"]))
        else:
            ws = [w for w in c["WL"] if w[0] != res and w[0][0] != "c"]
            if c["so"] != res:
                ws.append((c["so"], -load, c["oend"], c["d_so"]))
        return ws

    occ.sort()
    out = dict(W=W, busy=0.0, idle=0.0, forced=0.0, vehicle=0.0, other=0.0, start=0.0, n_gaps=0)
    if occ:                                     # head: from 0 to the first hold (the vehicles must drive here first), listed apart, counted as idle
        out["start"] = out["idle"] = occ[0][0]
    end, last = 0.0, None
    for o in occ:
        if last is not None and o[0] > end + 1e-6:
            g = o[0] - end
            out["idle"] += g
            out["n_gaps"] += 1
            if last[2] == o[2]:
                out["vehicle"] += g
            else:
                lo = last[5] - o[7]
                f = first_ok(lo, forbid(fixed(last[3], last[8], last[4]), fixed(o[3], o[8], o[6]), cap)) - lo
                out["forced"] += min(g, max(0.0, f))
                out["other"] += g - min(g, max(0.0, f))
        if last is None or o[1] > end:
            out["busy"] += o[1] - max(o[0], end) if last is not None else o[1] - o[0]
            end, last = max(end, o[1]), o
    out["idle"] += W - end
    out["other"] += W - end
    return out


def code_sha():
    """SHA-256 of this module and experiments/subnet_bounds.py (recorded in bounds3.json)."""
    return hashlib.sha256(Path(__file__).read_bytes() + (ROOT / "experiments" / "subnet_bounds.py").read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("keys", nargs="*", help="points instance|seed|N|K (default: the nine points of the through corridor)")
    ap.add_argument("--subnet", required=True, help="output directory of experiments/subnet_exact.py")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    F.pin(T.CORES)
    keys = a.keys or [f"yupu_corridor|{s}|{N}|8" for N in T.NS for s in T.SEEDS]
    from exact import model as X
    out, sha = Path(a.out), code_sha()
    out.mkdir(parents=True, exist_ok=True)
    f = out / "bounds3.json"
    res = json.loads(f.read_text(encoding="utf-8"))["points"] if f.exists() else {}
    for key in keys:
        if key in res:
            continue
        name, seed, N, k = key.split("|")
        inst = T.instance(name, int(seed), int(N))
        r = json.loads((Path(a.subnet) / "points" / F.fname(key)).read_text(encoding="utf-8"))
        lam, bind = X.lam_window(inst, r["opt"]["W"])
        pr = chain3(inst, int(k), tuple(bind[0]["res"]))
        vb = veh_bound(inst, int(k))
        res[key] = dict(chain3=pr["W"] if pr else None, veh=vb, lp=r["lp"]["W"], lamN=lam, binding=bind[0]["name"], parts=pr,
                        thr3=int(N) * T.DAY / pr["W"] if pr else None, thr_veh=int(N) * T.DAY / vb)
        F.write_json(f, dict(script="experiments/subnet_gap.py", sha256=sha, points=res))
        print(key, f"LP {r['lp']['W']:.0f} s  chain3 {pr['W'] if pr else float('nan'):.0f} s  vehicle {vb:.0f} s  λ3/λ_W^N "
                   f"{(int(N) * T.DAY / pr['W'] / lam) if pr else float('nan'):.3f}  λveh/λ_W^N {int(N) * T.DAY / vb / lam:.3f}", flush=True)
    print(f"-> {f}")


if __name__ == "__main__":
    main()

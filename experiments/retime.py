"""Whole-network offline retiming: how much of the gap between full-path reservation and the system bound on the whole yard
is lost to timing, to the precedence on the resources and to dispatch.

Takes a saturated simulation schedule of full-path reservation (reserve, just-in-time departure; exact.model.FPR in addition
records the vehicle of each task) on the whole yard: main flow mix, K = 100 (plateau), 2 working days (32 h, from an empty
system), seeds 101-103, Okpo (rev 1833) and Yantai (rev 396). The task order of each vehicle and the precedence of the holds
on every resource are fixed and only the times are optimised (offline retiming with all tasks known, not an online rule):
  variables: the empty departure t1 and the loaded departure t2 of each task. The hold windows are taken from Sim.windows
        (the same windows as full-path reservation, the destination including the 6.4 s drive-off of the empty vehicle after
        unloading), at fixed offsets from t1 or t2 (no stops en route); the length of the origin stop resource window
        [t1 + ao, t2 + tail] varies with the wait (waiting only at the origin).
  constraints: t2 >= t1 + empty run + loading; consecutive tasks of a vehicle t1_j >= t2_i + loaded run + unloading; t >= 0;
        for every pair of holds on a resource that do not overlap in the original schedule, a before b: start of b >= end
        of a. Written reduced to "immediate predecessors": if c is the predecessor of b that starts latest, the predecessors
        that end no later than the start of c follow through c (end of a <= start of c <= end of c <= start of b) and are not
        written separately.
These are difference constraints; the earliest solution (longest path, queue-based Bellman-Ford) makes every variable minimal
at once: it is both the shortest completion and the solution with the most tasks completed before any instant (under the
fixed precedence), in polynomial time. The original schedule satisfies all constraints, so there is a solution and no
positive cycle.
Feasibility: holds that overlap pairwise after retiming also overlap pairwise in the original schedule (pairs that do not
overlap have their order and gap fixed); the Helly property of intervals gives them a common instant in the original
schedule, where the sum with the maximum width per vehicle is <= capacity, so capacity holds everywhere. The code still scans
resource by resource (maximum width per vehicle); on an overload it adds the original order of the violating pairs and solves
again (lazily), recording the rounds; finally the independent checker (exact/check.py, clr=True: the booking of full-path
reservation) checks the result.
Comparison (relaxed): only the conflicting pairs, which do not fit together, are fixed (with the same reduction); pairs that
can share are not fixed, and overloads add the original order lazily until the scan passes.
First step: full-path reservation books each task in "the earliest slot that fits between the reservations booked before
it", which is the longest path of its own precedence, so retiming with the precedence fixed does not change it (Okpo: the same
completion to 1e-10 s); the loss lies wholly in the precedence on the resources and in dispatch.
Same-dispatch offline rebooking (second step; the same-dispatch offline rebooking of the sub-network study scaled up to the
whole yard): dispatch and the order of each vehicle unchanged, the precedence on the resources free. Windows as in the model
of exact.model (model_windows, destination until the end of unloading); time rounded to 0.01 s (int_start: at 0.1 s the
precedence of full-path reservation has positive cycles after rounding); starting solution = the rounded earliest solution
with the original precedence fixed; rolling large-neighbourhood search (lns_full): each round frees the times of 40
consecutive tasks in the order of their empty departures (the precedence on the resources may change with them), the
variables of the other tasks are fixed at their current values, the completion may not exceed the current value, and the sum
of the loaded departures of the freed tasks is minimised (CP-SAT, on the threads of the point); if better, all tasks are
shifted left in the rounded model (int_shift); the window moves on by half a block. Finally a left shift in continuous time
(retime) and the independent checker (destination until the end of unloading).
Readings: throughput lambda = N·16 h / completion (the same batch of N tasks; the original schedule and the retimed ones all
from an empty system); mid-run rate = rate between the 10 % and the 90 % completion (without the boundary at start and end);
lambda_W^N = T1' of the mix of this batch (bound.bounds, as in the per-point check); lambda^S = system T1' (bounds.system,
compared with the released value); the plateaus of full-path reservation and of the reference rule (stepwise request +
recovery, "teleport") are the released ones. The idle time on the binding resource is attributed to "the constraint that
pushes the next hold later": the previous task of the same vehicle (dispatch and vehicle order), precedence on another
resource (chain coupling and the order there), the start and the end.
Demand mapping: --map conserved takes the inter-area mapping with erection flow conserved (the main case; the task stream of
conserved_mapping, compute_conserved), with lambda^S and the plateaus of the two rules of that case; the default is the
proportional inter-area mapping.

Run:   python experiments/retime.py --out <dir> [--map conserved] [--minutes M] [--slots DIR] [--max-minutes M]
           [--max-threads N] [--aggregate-only] [--system FILE --plateau-okpo FILE --plateau-yantai FILE]
           [--conserved-report DIR] [--dispatch DIR_OR_CSV]
For the main case with the released readings:
       python experiments/retime.py --out <dir> --map conserved --conserved-report results/main_case
           --dispatch results/retiming/offline_dispatch.csv
and for the proportional mapping: --system results/bounds/system_by_granularity.csv --plateau-okpo
results/saturation/okpo_readings.csv --plateau-yantai results/saturation/rules_summary.json (the released offline-dispatch table
is that of the main case; for the proportional mapping its columns stay NaN unless --dispatch names a run with that map).
Each point runs in its own child process (experiments/subnet_exact.schedule; with --slots on the CPU slots it takes). Writes
<dir>/config.json, progress.json, points/ (one JSON per point with all schedules), system.json and, when the released
readings are given, points.csv. The released tables are results/retiming/main_case.csv (--map conserved; configuration
results/retiming/main_case_config.json) and results/retiming/proportional.csv (proportional mapping).
Released readings read by the aggregation: --system: system bounds by granularity (results/bounds/system_by_granularity.csv;
columns yard, scen, section, T1_sys); --plateau-okpo: plateaus of Okpo rev 1833 (results/saturation/okpo_readings.csv; columns
macro, r1833; rows YuResPlateau, YuRefPlateau); --plateau-yantai: summary of the Yantai saturation runs
(results/saturation/rules_summary.json; ["sat"]["yantai|main"]["rules"]["whole|reserve" and "whole|segment"]["plateau"]);
--conserved-report (with --map conserved, instead of the three): directory of the main case (results/main_case/) with
bounds.csv (yard, map, lam16) and saturation.csv (yard, model, map, plateau; sat_summary.csv in a report directory of the
main case is read when present); --dispatch: the output directory of experiments/offline_dispatch.py (points/) or its
points.csv (results/retiming/offline_dispatch.csv), with the same map (its columns are NaN without it).
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import runner as F                        # noqa: E402
from occupancy import params as P         # noqa: E402

YARDS = ("yupu", "yantai")
SEEDS = (101, 102, 103)
K, DAYS = 100, 2
MODES = ("all", "conflict")
TOL = 1e-7                                # "do not overlap" in the original schedule: end of a <= start of b + TOL (0 where the simulation times are equal)
DAY = 16 * 3600.0
VEH, TASK, ORD = 0, 1, 2                  # kinds of edges
WS = 1000                                 # integer scale of widths and capacities in the cumulative constraints of CP-SAT (1/WS m; the mapped road widths have many decimals, a common multiple cannot make them exact)
UNIT = 0.01                               # time unit (s) of the same-dispatch offline rebooking: at 0.1 s the precedence of full-path reservation has positive cycles after rounding (int_start)


def keys():
    return [f"{y}|{s}|{K}|{DAYS}" for y in YARDS for s in SEEDS]


# ---------------------------------------------------------------- original schedule and windows
def run_fpr(yard, seed, k, days):
    """One saturated simulation of full-path reservation. Returns a dict: net, p, sim, T (only the tasks unloaded within days,
    renumbered 0...N-1 in id order, with the departure position of the empty run start, TE, TL, release = 0), seq {vehicle:
    [(task, t1, t2)]}, starts (start positions of the vehicles), x0 (original times; variable 2j = t1_j, 2j + 1 = t2_j), base
    (constraints of the vehicle order and within tasks (u, v, c, kind): x_v >= x_u + c), DL (loaded run + unloading of each
    task)."""
    from exact.model import FPR
    from occupancy.network import Net
    from occupancy.tasks import vehicle_starts
    Y = P.YARDS[yard]
    p = P.scenario("main", days=days)
    net = Net(Y["network"], p, Y["roles"])
    sim = FPR(net, p, "reserve", k, seed, None)
    sim.run()
    return collect(net, p, sim, vehicle_starts(net, seed, k))


def run_fpr_conserved(yard, seed, k, days):
    """As run_fpr, with the inter-area mapping with erection flow conserved (the main case, demand_map = "conserved"): at the
    saturated level a free vehicle takes its task from conserved_mapping.stream (as runner.compute does at the saturated
    level)."""
    import conserved_mapping
    from exact.model import FPR
    from occupancy.network import Net
    from occupancy.tasks import vehicle_starts
    Y = P.YARDS[yard]
    p = P.scenario("main", days=days, demand_map="conserved")
    net = Net(Y["network"], p, Y["roles"])
    sim = FPR(net, p, "reserve", k, seed, None)
    sim.stream = conserved_mapping.stream(net, seed, p)
    sim.run()
    return collect(net, p, sim, vehicle_starts(net, seed, k))


def collect(net, p, sim, starts):
    """A finished full-path reservation (exact.model.FPR) -> the dict of run_fpr (the tests use it directly on small
    networks)."""
    from occupancy.sim import Vehicle
    done = sorted(sim.done, key=lambda r: r["id"])
    pos = {r["id"]: j for j, r in enumerate(done)}
    T = [dict(id=r["id"], flow=r["flow"], o=r["o"], d=r["d"], B=r["B"], Lv=r["Lv"], start=r["start"], release=0.0) for r in done]
    seq = collections.defaultdict(list)
    for r in done:
        assert "tc" not in r["plan"], "dock-side waiting places are outside the scope of this experiment"
        seq[r["plan"]["vid"]].append((pos[r["id"]], r["plan"]["t1"], r["plan"]["t2"]))
    seq = {kk: sorted(v, key=lambda x: x[1]) for kk, v in sorted(seq.items())}
    x0 = np.zeros(2 * len(T))
    for lst in seq.values():
        for j, a, b in lst:
            x0[2 * j], x0[2 * j + 1] = a, b
    for t in T:
        so, sd, d_so, d_sd, t["TE"], t["TL"], WE, WL, *_ = sim.windows(Vehicle(-1, t["start"]), t)
        assert all(r[0] != "c" for r, *_ in WL), "cranes are outside the scope of this experiment (main flow mix)"
    DL = [t["TL"] + p["unload_s"] for t in T]
    base = []
    for lst in seq.values():
        for n, (j, a, b) in enumerate(lst):
            base.append((2 * j, 2 * j + 1, T[j]["TE"] + p["load_s"], TASK))
            if n:
                i = lst[n - 1][0]
                base.append((2 * i + 1, 2 * j, DL[i], VEH))
    return dict(net=net, p=p, sim=sim, T=T, seq=seq, starts=starts, x0=x0, base=base, DL=DL)


def from_seq(inst, seq):
    """exact.model.Instance (sub-network) and one of its schedules {vehicle: [(task, t1, t2)]} -> the same dict as run_fpr
    (windows by model_windows; used for the corridor). The task dicts get the departure position of the empty run start, TE
    and TL."""
    p, T = inst.p, inst.tasks
    x0, base = np.zeros(2 * len(T)), []
    DL = [inst.dur_l(j) for j in range(len(T))]
    for k, lst in seq.items():
        loc = inst.starts[int(k)]
        for n, (j, a, b) in enumerate(lst):
            c = inst.chain(j, loc)
            T[j].update(start=loc, TE=c["TE"], TL=c["TL"])
            x0[2 * j], x0[2 * j + 1] = a, b
            base.append((2 * j, 2 * j + 1, c["TE"] + p["load_s"], TASK))
            if n:
                base.append((2 * lst[n - 1][0] + 1, 2 * j, DL[lst[n - 1][0]], VEH))
            loc = inst.net.stops[T[j]["d"]]["node"]
    return dict(net=inst.net, p=p, T=T, seq={int(k): v for k, v in seq.items()}, starts=inst.starts, x0=x0, base=base, DL=DL)


def fpr_windows(R):
    """The windows of full-path reservation itself (Sim.windows, not merged within a task, destination until the end of
    unloading + the 6.4 s drive-off): for each task [(resource, start anchor, start offset, end anchor, end offset, width)],
    anchor 0 = t1, 1 = t2."""
    from occupancy.sim import Vehicle
    out = []
    for t in R["T"]:
        so, sd, d_so, d_sd, TE, TL, WE, WL, ao, ad, tail_l, clr = R["sim"].windows(Vehicle(-1, t["start"]), t)
        out.append([(r, 0, a, 0, b, d) for r, a, b, d in WE] + [(so["res"], 0, ao, 1, tail_l, d_so)] +
                   [(r, 1, a, 1, b, d) for r, a, b, d in WL] + [(sd["res"], 1, ad, 1, TL + R["p"]["unload_s"] + clr, d_sd)])
    return out


def bare_instance(R):
    """The window part of exact.model.Instance (chain, dur_l, cap, starts), without the empty-run closure of a sub-network (not
    needed on the whole network, and too slow for many tasks)."""
    from exact.model import Instance
    from occupancy.sim import Sim
    inst = object.__new__(Instance)
    inst.name, inst.net, inst.p, inst.tasks, inst.starts = "whole network", R["net"], R["p"], R["T"], R["starts"]
    inst.cap, inst.size = R["net"].capacities(R["p"])
    inst.sim, inst._chain = Sim(R["net"], R["p"], "reserve", 1, 1, None, tasks=[]), {}
    return inst


def model_windows(inst, clr=0.0):
    """The windows of the model of exact.model (Instance.chain: merged per task and resource, the origin window merged with the
    adjoining windows of the loaded run, the destination only until the end of unloading -- Definition 1(c)).
    clr > 0: the booking of full-path reservation, the destination window extended to clr seconds after the end of unloading
    (drive-off of the empty vehicle). The same window is extended, no separate stretch is listed: a separate stretch would
    overlap the destination window by one unit after rounding and count the width of the same task twice."""
    out = []
    for j, t in enumerate(inst.tasks):
        c = inst.chain(j, t["start"])
        ws = [(r, 0, a, 0, b, d) for r, a, b, d in c["WE"]] + [(c["so"], 0, c["ao"], 1, c["oend"], c["d_so"])] + \
            [(r, 1, a, 1, b, d) for r, a, b, d in c["WL"]]
        if clr:
            sd, end = inst.net.stops[t["d"]]["res"], inst.dur_l(j)
            k = next(i for i, w in enumerate(ws) if w[0] == sd and w[3] == 1 and abs(w[4] - end) < 1e-6)
            ws[k] = ws[k][:4] + (end + clr, ws[k][5])
        out.append(ws)
    return out


def drive_off_trims(win, D, clr):
    """Under the booking of full-path reservation the drive-off after unloading of a vehicle's previous task (clr seconds on
    the destination resource, the extended window of model_windows) may overlap a window of its next empty run on the same
    resource (a just-in-time departure starts right from here); per vehicle the maximum width applies at each instant (as in
    the reservation table of the simulation and in the checker), whereas the cumulative constraints of CP-SAT add the widths.
    Each such pair is marked here with the stretch to cut, so that adding equals taking the maximum: when the window of the
    next task is no wider than the destination window, the window of the next task starts at the end of the drive-off (lo_v,
    lo_o: start = max(original start, x[lo_v] + lo_o)); when it is wider, the drive-off part of the destination window is cut
    at the start of the next task's window (hi_v, hi_o, hi_f: end = max(x[ev] + hi_f, min(original end, x[hi_v] + hi_o)), hi_f
    is the offset of the end of unloading, nothing before it is cut). The destination window is the one extended to the end
    of the drive-off (model_windows).
    Only for the CP-SAT models (same_dispatch, lns_full); the difference constraints and the scan already take the maximum
    width per vehicle. Adds lo_v, lo_o, hi_v, hi_o, hi_f to win in place."""
    n = len(win["res"])
    win["lo_v"], win["lo_o"] = np.full(n, -1, np.int64), np.zeros(n)
    win["hi_v"], win["hi_o"], win["hi_f"] = np.full(n, -1, np.int64), np.zeros(n), np.zeros(n)
    by_ev, by_sv = collections.defaultdict(list), collections.defaultdict(list)
    for w in range(n):
        by_ev[int(win["ev"][w])].append(w)
        by_sv[int(win["sv"][w])].append(w)
    pairs = 0
    for lst in D["seq"].values():
        for (i, _, _), (j, _, _) in zip(lst, lst[1:]):
            dest = [w for w in by_ev[2 * i + 1] if abs(win["eo"][w] - D["DL"][i] - clr) < 1e-6]
            if not dest:
                continue
            t = dest[0]
            nxt = [w for w in by_sv[2 * j] if win["res"][w] == win["res"][t]]
            if not nxt:
                continue
            pairs += 1
            if max(win["d"][w] for w in nxt) <= win["d"][t] + 1e-9:
                for w in nxt:
                    win["lo_v"][w], win["lo_o"][w] = 2 * i + 1, D["DL"][i] + clr
            else:
                win["hi_v"][t], win["hi_o"][t], win["hi_f"][t] = 2 * j, min(win["so"][w] for w in nxt), D["DL"][i]
    return pairs


def cp_intervals(m, x, win, H, fl, ce):
    """The intervals of CP-SAT (shared by same_dispatch and lns_full): windows rounded as in the rounded model (starts down,
    ends up); the length of the origin window varies with two variables; windows marked by drive_off_trims are cut by their
    mark. Returns {resource number: [(interval, width, capacity)]}."""
    by, big = collections.defaultdict(list), 4 * H + 10 ** 7
    lo_v, hi_v = win.get("lo_v"), win.get("hi_v")
    for w in range(len(win["res"])):
        sv, ev, a, b = int(win["sv"][w]), int(win["ev"][w]), fl(win["so"][w]), ce(win["eo"][w])
        if lo_v is not None and lo_v[w] >= 0:              # from the end of the drive-off of the previous task
            s, e = m.new_int_var(0, big, ""), m.new_int_var(0, big, "")
            m.add_max_equality(s, [x[sv] + a, x[int(lo_v[w])] + ce(win["lo_o"][w])])
            m.add_max_equality(e, [s, x[ev] + b])
            iv = m.new_interval_var(s, m.new_int_var(0, big, ""), e, "")
        elif hi_v is not None and hi_v[w] >= 0:            # drive-off part of the destination window cut at the start of the next task's window (nothing before the end of unloading)
            mn, e = m.new_int_var(0, big, ""), m.new_int_var(0, big, "")
            m.add_min_equality(mn, [x[ev] + b, x[int(hi_v[w])] + fl(win["hi_o"][w])])
            m.add_max_equality(e, [x[ev] + ce(win["hi_f"][w]), mn])
            iv = m.new_interval_var(x[sv] + a, m.new_int_var(0, big, ""), e, "")
        elif sv == ev:
            iv = m.new_fixed_size_interval_var(x[sv] + a, b - a, "")
        else:
            iv = m.new_interval_var(x[sv] + a, m.new_int_var(0, big, ""), x[ev] + b, "")
        by[int(win["res"][w])].append((iv, float(win["d"][w]), float(win["cap"][w])))
    return by


def make_win(per_task, seq, x0, cap):
    """Window table (dict of arrays): res resource number, sv/so variable and offset of the start, ev/eo variable and offset of
    the end, d width, owner vehicle, cap capacity; res_list resource number -> resource. Windows of length 0 at the original
    times x0 are dropped (an origin window is no shorter than the loading time, the other windows have fixed lengths)."""
    rid, res_list, W = {}, [], collections.defaultdict(list)
    for kk, lst in seq.items():
        for j, _, _ in lst:
            for r, ia, oa, ib, ob, d in per_task[j]:
                if x0[2 * j + ib] + ob - x0[2 * j + ia] - oa <= 1e-6 or d <= 1e-9:
                    continue
                if r not in rid:
                    rid[r] = len(res_list)
                    res_list.append(r)
                for key, v in (("res", rid[r]), ("sv", 2 * j + ia), ("so", oa), ("ev", 2 * j + ib), ("eo", ob), ("d", d), ("owner", kk)):
                    W[key].append(v)
    win = {kk: np.array(v, dtype=float if kk in ("so", "eo", "d") else np.int64) for kk, v in W.items()}
    win["cap"] = np.array([cap[r] for r in res_list])[win["res"]]
    return win, res_list


def times(win, x):
    return x[win["sv"]] + win["so"], x[win["ev"]] + win["eo"]


# ---------------------------------------------------------------- difference constraints
def order_pairs(win, xr, tol=TOL):
    """Reduced representation (immediate predecessors) of the pairs of holds (a before b) on each resource that do not overlap
    in schedule xr. Returns (array a, array b, total number of non-overlapping pairs)."""
    s0, e0 = times(win, xr)
    out_a, out_b, total = [], [], 0
    order = np.argsort(win["res"], kind="stable")
    bounds_ = np.flatnonzero(np.diff(win["res"][order])) + 1
    for idx in np.split(order, bounds_):
        s, e = s0[idx], e0[idx]
        oe = np.argsort(e, kind="stable")
        E, PM = e[oe], np.maximum.accumulate(s[oe])
        hi = np.searchsorted(E, s + tol, "right")              # number of predecessors with end <= start of b (prefix in end order)
        total += int(hi.sum())
        lo = np.searchsorted(E, PM[np.maximum(hi - 1, 0)] + tol, "right")
        cnt = np.where(hi > 0, hi - lo, 0)
        if not cnt.sum():
            continue
        b_loc = np.repeat(np.arange(len(idx)), cnt)
        off = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        out_a.append(idx[oe[np.repeat(lo, cnt) + off]])
        out_b.append(idx[b_loc])
    return np.concatenate(out_a), np.concatenate(out_b), total


def pair_edges(win, a, b, xr):
    """a before b (the order in xr): x[sv_b] + so_b >= x[ev_a] + eo_a + min(0, start of b - end of a in xr) (xr itself
    satisfies it)."""
    s0, e0 = times(win, xr)
    c = win["eo"][a] - win["so"][b] + np.minimum(0.0, s0[b] - e0[a])
    return win["ev"][a], win["sv"][b], c


def earliest(n, U, V, C, Kd, R, order):
    """Least solution of the difference constraints x_V >= x_U + C and x >= 0 (longest path, queue-based Bellman-Ford). For
    equal (U, V) only the largest C is kept. Returns (x, the binding edge of each variable (-1 for x = 0), U, C, kind,
    resource after removing duplicates). order: initial queue order (original times)."""
    o = np.lexsort((-C, V, U))
    U, V, C, Kd, R = U[o], V[o], C[o], Kd[o], R[o]
    keep = np.r_[True, (U[1:] != U[:-1]) | (V[1:] != V[:-1])]
    U, V, C, Kd, R = U[keep], V[keep], C[keep], Kd[keep], R[keep]
    ptr = np.searchsorted(U, np.arange(n + 1))
    x, pe, cnt = np.zeros(n), np.full(n, -1), np.zeros(n, np.int64)
    inq = np.ones(n, bool)
    q = collections.deque(order.tolist())
    while q:
        u = q.popleft()
        inq[u] = False
        a, b = ptr[u], ptr[u + 1]
        if a == b:
            continue
        vs = V[a:b]
        cand = x[u] + C[a:b]
        m = cand > x[vs] + 1e-9
        if not m.any():
            continue
        vv = vs[m]
        x[vv], pe[vv] = cand[m], np.arange(a, b)[m]
        cnt[vv] += 1
        if cnt[vv].max() > n:
            raise RuntimeError("positive cycle in the difference constraints (the original schedule should satisfy all of them)")
        new = vv[~inq[vv]]
        inq[new] = True
        q.extend(new.tolist())
    return x, pe, U, C, Kd, R


def overloads(win, x, tol=1e-6, per_owner=True):
    """Scan resource by resource after retiming (maximum width per vehicle): overloaded stretches -> [(resource number, the
    widest window of each vehicle present)]. A first pass adds the widths directly to find candidate stretches (overlapping
    windows of one vehicle are then counted more than once), then the midpoint of each candidate is checked with the maximum
    width per vehicle. per_owner=False: the widths are added directly (as the cumulative constraints of CP-SAT write them),
    the candidate stretches are the overloads."""
    s, e = x[win["sv"]] + win["so"], x[win["ev"]] + win["eo"]
    t = np.concatenate([s, e])
    dl = np.concatenate([win["d"], -win["d"]])
    rr = np.concatenate([win["res"], win["res"]])
    o = np.lexsort((dl, t, rr))
    t, dl, rr = t[o], dl[o], rr[o]
    cs = np.cumsum(dl)
    gs = np.r_[True, rr[1:] != rr[:-1]]
    first = np.maximum.accumulate(np.where(gs, np.arange(len(o)), 0))
    lvl = cs - (cs - dl)[first]
    nxt_same = np.r_[rr[1:] == rr[:-1], False]
    gap = np.r_[t[1:] - t[:-1], 0.0]
    capv = np.zeros(win["res"].max() + 1)
    capv[win["res"]] = win["cap"]
    cand = np.flatnonzero(nxt_same & (gap > 1e-9) & (lvl > capv[rr] + tol))
    if not len(cand) or not per_owner:
        return [(int(rr[i]), None) for i in cand]
    by = collections.defaultdict(list)
    for i in cand:
        by[int(rr[i])].append(0.5 * (t[i] + t[i + 1]))
    order = np.argsort(win["res"], kind="stable")
    rs = win["res"][order]
    out = []
    for r, mids in by.items():
        idx = order[np.searchsorted(rs, r, "left"):np.searchsorted(rs, r, "right")]
        for m in mids:
            best = {}
            for w in idx[(s[idx] < m) & (e[idx] > m)]:
                k = int(win["owner"][w])
                if k not in best or win["d"][w] > win["d"][best[k]]:
                    best[k] = w
            if sum(win["d"][w] for w in best.values()) > capv[r] + tol:
                out.append((r, sorted(best.values())))
    return out


def retime(win, base, x0, mode, max_rounds=500):
    """Retiming (see the module docstring): the vehicle orders and the precedence on the resources of schedule x0 are fixed and
    the earliest solution is computed. mode = all: all pairs on each resource that do not overlap in x0 are fixed; conflict:
    only the pairs that do not fit together. On an overload the order of x0 is added for the violating pairs (lazily) until
    the scan finds no overload. Returns (x, readings dict, edge table (U, C, kind, resource), pe)."""
    n = len(x0)
    s0, e0 = times(win, x0)
    a, b, total = order_pairs(win, x0)
    stats = dict(nonoverlap_pairs=total, reduced_pairs=len(a))
    if mode == "conflict":
        k = win["d"][a] + win["d"][b] > win["cap"][a] + 1e-9
        a, b = a[k], b[k]
    stats["initial_pairs"] = len(a)
    have = set(zip(a.tolist(), b.tolist()))
    BU, BV, BC, BK = (np.array(c) for c in zip(*base))
    order = np.argsort(x0, kind="stable")
    rounds, added = 0, 0
    while True:
        eu, ev, ec = pair_edges(win, a, b, x0)
        U = np.concatenate([BU, eu]).astype(np.int64)
        V = np.concatenate([BV, ev]).astype(np.int64)
        C = np.concatenate([BC, ec])
        Kd = np.concatenate([BK, np.full(len(a), ORD)]).astype(np.int64)
        R = np.concatenate([np.full(len(BU), -1), win["res"][a]]).astype(np.int64)
        x, pe, U2, C2, K2, R2 = earliest(n, U, V, C, Kd, R, order)
        bad = overloads(win, x)
        if not bad:
            break
        rounds += 1
        if rounds > max_rounds:
            raise RuntimeError(f"more than {max_rounds} rounds of lazily added constraints")
        new = set()
        for r, S in bad:
            cands = [(u, w) if s0[u] <= s0[w] else (w, u) for i, u in enumerate(S) for w in S[i + 1:]]
            cands = [(u, w) for u, w in cands if e0[u] <= s0[w] + TOL and (u, w) not in have]
            conf = [(u, w) for u, w in cands if win["d"][u] + win["d"][w] > win["cap"][u] + 1e-9]
            pick = conf or cands[:1]
            if not pick:
                raise RuntimeError(f"resource {r} overloaded, but no pair that does not overlap in the original schedule (excluded by the Helly property)")
            new.update(pick)
        have |= new
        na, nb = zip(*sorted(new))
        a, b = np.concatenate([a, na]), np.concatenate([b, nb])
        added += len(new)
    stats.update(rounds=rounds, lazy_pairs=added, final_pairs=len(a), edges=len(U2),
                 edges_by_kind={nm: int((K2 == c).sum()) for nm, c in (("veh", VEH), ("task", TASK), ("order", ORD))})
    return x, stats, (U2, C2, K2, R2), pe


def int_start(D, win, unit=UNIT, clr=0.0):
    """Feasible starting solution in the rounded model: the vehicle orders of the original schedule x0 and all pairs that do
    not overlap on each resource (true windows) are fixed, the windows are rounded outwards as in the rounded model (starts
    down, ends up, durations up), and the earliest solution is computed -- the set of overlapping holds can only be a subset of
    those overlapping pairwise in x0, so it is feasible. In full-path reservation many holds adjoin exactly, and after rounding
    some cycles may become positive (already at 0.1 s); at unit = 0.01 s there are none, and the completion differs from the
    original schedule by less than 0.05 %. With a positive cycle it falls back to replay_start (clr: the booking with the
    destination drive-off). Returns (seconds, method)."""
    x0, base = D["x0"], D["base"]
    ce = lambda v: np.ceil(np.asarray(v) / unit - 1e-7) * unit
    fl = lambda v: np.floor(np.asarray(v) / unit + 1e-7) * unit
    a, b, _ = order_pairs(win, x0)
    BU, BV, BC, BK = (np.array(c) for c in zip(*base))
    U = np.concatenate([BU, win["ev"][a]]).astype(np.int64)
    V = np.concatenate([BV, win["sv"][b]]).astype(np.int64)
    C = np.concatenate([ce(BC), ce(win["eo"][a]) - fl(win["so"][b])])
    z = np.zeros(len(U), np.int64)
    try:
        x, *_ = earliest(len(x0), U, V, C, z, z, np.argsort(x0, kind="stable"))
        return np.round(x / unit) * unit, "order"
    except RuntimeError:
        return replay_start(D, unit, clr), "replay"


def replay_start(D, unit=UNIT, clr=0.0):
    """Feasible starting solution in the rounded model (the fallback of int_start): replays full-path reservation with the same
    dispatch and the same order on each vehicle (Replay), the windows rounded outwards as in the rounded model, each task
    booked in the earliest slot that fits. With clr > 0 the destination keeps the drive-off after unloading (the booking of
    full-path reservation), otherwise it ends at the end of unloading."""
    from occupancy.sim import Sim
    T, seq = D["T"], D["seq"]
    ce = lambda v: math.ceil(v / unit - 1e-7) * unit
    fl = lambda v: math.floor(v / unit + 1e-7) * unit
    load, unload = D["p"]["load_s"], D["p"]["unload_s"]
    t1_0 = {j: a for lst in seq.values() for j, a, _ in lst}

    class Replay(Sim):
        def windows(self, v, task):
            so, sd, d_so, d_sd, TE, TL, WE, WL, ao, ad, tail_l, clr = super().windows(v, task)
            return (so, sd, d_so, d_sd, ce(TE + load) - load, ce(TL + unload) - unload, [(r, fl(a), ce(b), d) for r, a, b, d in WE],
                    [(r, fl(a), ce(b), d) for r, a, b, d in WL], fl(ao), fl(ad), ce(tail_l), ce(clr) if clr else 0.0)

        def dispatch(self, t):
            for k in sorted(self.idle, key=lambda k: t1_0[todo[k][0]] if todo.get(k) else math.inf):
                if todo.get(k):
                    j = todo[k].popleft()
                    self.idle.remove(k)
                    task = dict(T[j], id=j, dispatch=t, start=self.vs[k].loc)
                    self.plan(self.vs[k], task, t)

    todo = {k: collections.deque(j for j, _, _ in lst) for k, lst in seq.items()}
    sim = Replay(D["net"], dict(D["p"], days=400), "reserve", len(D["starts"]), 1, None, tasks=[])
    for v, loc in zip(sim.vs, D["starts"]):
        v.loc = loc
    sim.run()
    assert len(sim.done) == len(T)
    x = np.zeros(2 * len(T))
    for r in sim.done:
        assert r["start"] == T[r["id"]]["start"]
        x[2 * r["id"]], x[2 * r["id"] + 1] = r["plan"]["t1"], r["plan"]["t2"]
    return np.round(x / unit) * unit


def same_dispatch(win, base, DL, xh, secs, workers, unit=UNIT, seed=0, log=False):
    """Same-dispatch offline rebooking (whole network; the same-dispatch offline rebooking of the sub-network study scaled up
    to the whole yard): dispatch and the order of each vehicle fixed (base), the precedence on the resources free (cumulative
    constraints of CP-SAT on the widths, widths and capacities scaled to integers), the completion minimised. Windows as in
    the model of exact.model (model_windows). Time rounded to unit: window starts down, ends up, durations up, so an integer
    solution is always feasible in continuous time. xh: feasible starting solution in the rounded model (seconds; the result
    of exact.model.rebook).
    Returns dict(status, x (seconds), M (seconds), bound (seconds), wall, intervals, cumulatives)."""
    from ortools.sat.python import cp_model
    ce = lambda v: math.ceil(v / unit - 1e-7)
    fl = lambda v: math.floor(v / unit + 1e-7)
    n, N = len(xh), len(DL)
    H = max(round(xh[2 * j + 1] / unit) + ce(DL[j]) for j in range(N))
    m = cp_model.CpModel()
    x = [m.new_int_var(0, H, "") for _ in range(n)]
    for u, v, c, _ in base:
        m.add(x[v] >= x[u] + ce(c))
    by = cp_intervals(m, x, win, H, fl, ce)
    ncum = 0
    for lst in by.values():               # widths rounded up, capacities down to 1/WS m (conservative; a width equal to the capacity takes the capacity itself)
        if sum(d for _, d, _ in lst) > lst[0][2] + 1e-6:
            cap = math.floor(lst[0][2] * WS + 1e-6)
            m.add_cumulative([iv for iv, _, _ in lst], [min(cap, math.ceil(d * WS - 1e-6)) for _, d, _ in lst], cap)
            ncum += 1
    M = m.new_int_var(0, H, "M")
    for j in range(N):
        m.add(M >= x[2 * j + 1] + ce(DL[j]))
    m.minimize(M)
    for v, val in zip(x, xh):
        m.add_hint(v, round(val / unit))
    m.add_hint(M, H)
    sol = cp_model.CpSolver()
    sol.parameters.max_time_in_seconds = secs
    sol.parameters.num_workers = workers
    sol.parameters.random_seed = seed
    sol.parameters.log_search_progress = log
    st = sol.solve(m)
    out = dict(status=sol.status_name(st), wall=sol.wall_time, bound=sol.best_objective_bound * unit, intervals=len(win["res"]), cumulatives=ncum,
               hint_M=H * unit)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        out.update(x=np.array([sol.value(v) for v in x], dtype=float) * unit, M=sol.value(M) * unit)
    return out


def int_shift(win, base, x, unit=UNIT):
    """Left shift in the rounded model: the vehicle orders of x (feasible in the rounded model) and all pairs of rounded windows
    that do not overlap on each resource are fixed, and the earliest solution is computed. x itself satisfies all constraints
    (no positive cycle), the set of overlapping holds can only be a subset of those overlapping pairwise in x, so it stays
    feasible; the result is no later than x and still a multiple of unit."""
    ce = lambda v: np.ceil(np.asarray(v) / unit - 1e-7) * unit
    fl = lambda v: np.floor(np.asarray(v) / unit + 1e-7) * unit
    rw = dict(win, so=fl(win["so"]), eo=ce(win["eo"]))
    a, b, _ = order_pairs(rw, x, tol=1e-6)
    s0, e0 = times(rw, x)
    BU, BV, BC, BK = (np.array(c) for c in zip(*base))
    U = np.concatenate([BU, rw["ev"][a]]).astype(np.int64)
    V = np.concatenate([BV, rw["sv"][b]]).astype(np.int64)
    C = np.concatenate([ce(BC), rw["eo"][a] - rw["so"][b]])
    z = np.zeros(len(U), np.int64)
    y, *_ = earliest(len(x), U, V, C, z, z, np.argsort(x, kind="stable"))
    return np.round(y / unit) * unit


def lns_full(win, base, DL, xs, minutes, workers, unit=UNIT, block=40, iter_secs=20.0, seed=0, log=None):
    """Rolling large-neighbourhood search for the whole-network same-dispatch offline rebooking: the model is built once (as
    in same_dispatch); each round frees the times of block consecutive tasks in the current order of empty departures (the
    precedence on the resources may change with them), the variables of the other tasks are fixed at their current values
    (by changing their domains), the completion may not exceed the current value, and the sum of the loaded departures of
    the freed tasks is minimised; a better solution is accepted, and all tasks are then shifted left in the rounded model
    (int_shift), passing the time saved on to the later tasks. The window moves on by half a block each round and starts
    again from the beginning after a full pass.
    xs: feasible starting solution in the rounded model (seconds). Returns (x (seconds), log of the rounds)."""
    from ortools.sat.python import cp_model
    ce = lambda v: math.ceil(v / unit - 1e-7)
    fl = lambda v: math.floor(v / unit + 1e-7)
    n, N = len(xs), len(DL)
    H = max(round(xs[2 * j + 1] / unit) + ce(DL[j]) for j in range(N))
    m = cp_model.CpModel()
    x = [m.new_int_var(0, H, "") for _ in range(n)]
    for u, v, c, _ in base:
        m.add(x[v] >= x[u] + ce(c))
    by = cp_intervals(m, x, win, H, fl, ce)
    for lst in by.values():
        if sum(d for _, d, _ in lst) > lst[0][2] + 1e-6:
            cap = math.floor(lst[0][2] * WS + 1e-6)
            m.add_cumulative([iv for iv, _, _ in lst], [min(cap, math.ceil(d * WS - 1e-6)) for _, d, _ in lst], cap)
    M = m.new_int_var(0, H, "M")
    for j in range(N):
        m.add(M >= x[2 * j + 1] + ce(DL[j]))
    def dom(var, lo, hi):                 # domain of a variable [lo, hi] (one interval of two numbers, as built)
        d = m.proto.variables[var.index].domain
        d[0], d[1] = lo, hi
    cur = np.round(np.asarray(xs) / unit).astype(np.int64)
    Mk = lambda c: max(int(c[2 * j + 1]) + ce(DL[j]) for j in range(N))
    log = [] if log is None else log
    t_end, pos, it = time.time() + 60 * minutes, 0, 0
    while time.time() + 5 < t_end:
        order = np.argsort(cur[0::2], kind="stable")
        F = set(order[pos:pos + block].tolist())
        for j in range(N):
            for v in (2 * j, 2 * j + 1):
                dom(x[v], 0, H) if j in F else dom(x[v], int(cur[v]), int(cur[v]))
        dom(M, 0, Mk(cur))
        m.minimize(sum(x[2 * j + 1] for j in F))
        m.clear_hints()
        for v, val in zip(x, cur):
            m.add_hint(v, int(val))
        sol = cp_model.CpSolver()
        sol.parameters.max_time_in_seconds = min(iter_secs, max(1.0, t_end - time.time()))
        sol.parameters.num_workers = workers
        sol.parameters.random_seed = seed + it
        t0 = time.time()
        st = sol.solve(m)
        rec = dict(it=it, pos=pos, status=sol.status_name(st), secs=time.time() - t0)
        if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            new = np.array([sol.value(v) for v in x], dtype=np.int64)
            gain = int(sum(cur[2 * j + 1] - new[2 * j + 1] for j in F))
            if gain > 0:
                cur = np.round(int_shift(win, base, new * unit, unit) / unit).astype(np.int64)
            rec.update(gain_s=gain * unit, M=Mk(cur) * unit)
        log.append(rec)
        pos = pos + block // 2 if pos + block < N else 0
        it += 1
    return cur * unit, log


# ---------------------------------------------------------------- readings
def thr(N, W):
    return N * DAY / W


def mid_rate(done):
    """Rate (tasks/day) between the 10 % and the 90 % completion."""
    t = np.sort(done)
    i, j = math.ceil(0.1 * len(t)), math.floor(0.9 * len(t)) - 1
    return (j - i) * DAY / (t[j] - t[i])


def idle_causes(win, x, pe, edges, r, W1):
    """Idle time on the binding resource r after retiming, attributed by cause (seconds): start (0 to the first hold), tail
    (last hold to completion); each idle stretch in between is given to the hold that starts right after it, tracing back
    along the binding edges of the earliest solution (through the within-task edge "depart after loading" further back): the
    previous task of the same vehicle -> vehicle; precedence on another resource -> chain; precedence on this resource ->
    self; x = 0 -> start."""
    U, C, Kd, R = edges
    idx = np.flatnonzero(win["res"] == r)
    s, e = x[win["sv"][idx]] + win["so"][idx], x[win["ev"][idx]] + win["eo"][idx]
    o = np.argsort(s, kind="stable")
    out = collections.Counter()
    end, first = 0.0, True
    for i in o:
        if s[i] > end + 1e-9:
            gap = s[i] - end
            if first:
                out["start"] += gap
            else:
                var, cause = int(win["sv"][idx[i]]), None
                while cause is None:
                    k = pe[var]
                    if k < 0:
                        cause = "start"
                    elif Kd[k] == TASK:
                        var = int(U[k])
                    elif Kd[k] == VEH:
                        cause = "vehicle"
                    else:
                        cause = "self" if R[k] == r else "chain"
                out[cause] += gap
        first = False
        end = max(end, e[i])
    out["tail"] += max(0.0, W1 - end)
    return dict(out)


def busy_frac(win, x, r, W):
    idx = np.flatnonzero(win["res"] == r)
    iv = sorted(zip(x[win["sv"][idx]] + win["so"][idx], x[win["ev"][idx]] + win["eo"][idx]))
    tot, end = 0.0, -math.inf
    for a, b in iv:
        if b > end:
            tot += b - max(a, end)
            end = b
    return tot / W


def window_check(net, p, T, seq, starts, sim):
    """Windows consistent item by item: each task rebuilt by Sim.windows (the booking of full-path reservation, destination
    with the 6.4 s drive-off) and by task_holds of the independent checker (clr=True), compared after taking the piecewise
    maximum width on each resource (norm). Returns the number of tasks that differ."""
    from exact import check as Cx
    from occupancy.sim import Vehicle
    cap, size = net.capacities(p)
    bad = 0
    for kk, lst in seq.items():
        for j, t1, t2 in lst:
            t = T[j]
            so, sd, d_so, d_sd, TE, TL, WE, WL, ao, ad, tail_l, clr = sim.windows(Vehicle(-1, t["start"]), t)
            mine = [(r, t1 + a, t1 + b, d) for r, a, b, d in WE] + [(so["res"], t1 + ao, t2 + tail_l, d_so)] + \
                [(r, t2 + a, t2 + b, d) for r, a, b, d in WL] + [(sd["res"], t2 + ad, t2 + TL + p["unload_s"] + clr, d_sd)]
            hs, TE2, DL = Cx.task_holds(net, cap, size, p, t, t["start"], True)
            TT = {"e": t1, "l": t2}
            ref = [(r, TT[a1] + a, TT[b1] + b, w) for r, w, a1, a, b1, b, cr in hs]
            bad += norm(mine) != norm(ref) or abs(TE2 - TE) > 1e-9
    return bad


def norm(ws):
    """Pieces after taking the piecewise maximum width on each resource (crane pieces listed separately), rounded to 1e-6."""
    import collections
    by = collections.defaultdict(list)
    for r, s, e, d in ws:
        by[r].append((s, e, d))
    out = set()
    for r, iv in by.items():
        ts = sorted({x for s, e, _ in iv for x in (s, e)})
        segs = []
        for a, b in zip(ts, ts[1:]):
            w = max((d for s, e, d in iv if s <= a + 1e-9 and e >= b - 1e-9), default=0.0)
            if w > 0:
                if segs and abs(segs[-1][1] - a) < 1e-9 and segs[-1][2] == w:
                    segs[-1][1] = b
                else:
                    segs.append([a, b, w])
        out |= {(r, round(a, 6), round(b, 6), w) for a, b, w in segs}
    return out


def compute(key, minutes, w, out):
    """One point: yard|seed|K|days (child process; w CP-SAT threads, the same-dispatch offline rebooking limited to minutes
    minutes), writes points/<key>.json."""
    try:
        from exact import check as Cx
        from occupancy.bound import bounds
        yard, seed, k, days = key.split("|")
        t0 = time.time()
        D = run_fpr(yard, int(seed), int(k), float(days))
        net, p, T, seq, starts, x0, base = D["net"], D["p"], D["T"], D["seq"], D["starts"], D["x0"], D["base"]
        N, t_sim = len(T), time.time() - t0
        cap, _ = net.capacities(p)
        dl = lambda x: np.array([x[2 * j + 1] + D["DL"][j] for j in range(N)])
        to_seq = lambda x: {kk: [(j, float(x[2 * j]), float(x[2 * j + 1])) for j, _, _ in lst] for kk, lst in seq.items()}
        win, res_list = make_win(fpr_windows(D), seq, x0, cap)
        b = bounds(net, p, T, tight_only=True)
        rstar = tuple(b["binding_tight"][0]["res"])
        r_id = res_list.index(rstar)
        c0 = Cx.check(net, p, T, starts, seq, clr=True)
        done0 = dl(x0)
        W0 = float(done0.max())
        res = dict(key=key, yard=yard, seed=int(seed), K=int(k), days=float(days), N=N, flows=dict(collections.Counter(t["flow"] for t in T)),
                   windows=len(win["res"]), resources=len(res_list), window_mismatch=window_check(net, p, T, seq, starts, D["sim"]),
                   sim_secs=t_sim, lamN=b["network_tight"], binding=[x["name"] for x in b["binding_tight"][:3]], binding_res=str(rstar),
                   orig=dict(W=W0, thr=thr(N, W0), mid=mid_rate(done0), check=c0["ok"], max_ratio=c0["max_ratio"],
                             busy=busy_frac(win, x0, r_id, W0), seq={str(kk): v for kk, v in seq.items()}))
        # retiming with the precedence fixed (the windows of full-path reservation itself, destination with the 6.4 s drive-off)
        for mode in MODES:
            t1 = time.time()
            x, st, edges, pe = retime(win, base, x0, mode)
            done = dl(x)
            W1 = float(done.max())
            sq = to_seq(x)
            c1 = Cx.check(net, p, T, starts, sq, clr=True)
            assert np.all(x <= x0 + 1e-6), "the earliest solution must not be later than the original schedule"
            res[mode] = dict(W=W1, thr=thr(N, W1), mid=mid_rate(done), check=c1["ok"], check_errors=c1["errors"][:5], max_ratio=c1["max_ratio"],
                             busy=busy_frac(win, x, r_id, W1), idle=idle_causes(win, x, pe, edges, r_id, W1), secs=time.time() - t1,
                             seq={str(kk): v for kk, v in sq.items()}, **st)
        # same-dispatch offline rebooking: dispatch and the order of each vehicle unchanged, the precedence on the resources free
        # (the windows of the model of exact.model). Once for each destination booking: same = the booking of full-path
        # reservation (destination held for clr seconds of drive-off after unloading, as in the original schedule; the main
        # reading); same_def1c = Definition 1(c) (released at the end of unloading, as in the bounds). Each is checked under its
        # own booking
        inst = bare_instance(D)
        clr_s = (p["plat_len"] + p["long_clear"]) / p["v_empty"]
        for name, clr in (("same", clr_s), ("same_def1c", 0.0)):
            t1 = time.time()
            mw, mres = make_win(model_windows(inst, clr), seq, x0, cap)
            n_trim = drive_off_trims(mw, D, clr) if clr else 0
            m_id = mres.index(rstar)
            xs, how = int_start(D, mw, clr=clr)
            assert not overloads(dict(mw, so=np.floor(mw["so"] / UNIT + 1e-7) * UNIT, eo=np.ceil(mw["eo"] / UNIT - 1e-7) * UNIT), xs,
                                 per_owner=bool(clr)), "starting solution overloaded in the rounded model"
            x1 = int_shift(mw, base, xs)
            lg = []
            xl, lg = lns_full(mw, base, D["DL"], x1, minutes, w, log=lg)
            x2, st, edges, pe = retime(mw, base, xl, "all")              # left shift in continuous time (the gaps left by rounding)
            done = dl(x2)
            W2 = float(done.max())
            sq = to_seq(x2)
            c2 = Cx.check(net, p, T, starts, sq, clr=bool(clr))
            res[name] = dict(W=W2, thr=thr(N, W2), mid=mid_rate(done), check=c2["ok"], check_errors=c2["errors"][:5], max_ratio=c2["max_ratio"],
                             busy=busy_frac(mw, x2, m_id, W2), idle=idle_causes(mw, x2, pe, edges, m_id, W2), secs=time.time() - t1, threads=w,
                             minutes=minutes, start=how, clr=clr, drive_off_pairs=n_trim, W_replay=float(dl(xs).max()), W_shift=float(dl(x1).max()),
                             W_lns=float(dl(xl).max()), iters=len(lg), accepted=sum(1 for r in lg if r.get("gain_s", 0) > 0), log=lg,
                             seq={str(kk): v for kk, v in sq.items()})
        res.update(secs=time.time() - t0, thr=res["same"]["thr"], thr_ub=res["lamN"], status=f"retimed {res['all']['thr']:.0f}, same dispatch "
                   f"{res['same']['thr']:.0f} (originally {res['orig']['thr']:.0f})")
        F.write_json(Path(out) / "points" / F.fname(key), res)
    except Exception:
        import traceback
        (Path(out) / "failures.log").open("a", encoding="utf-8").write(f"{key}\n{traceback.format_exc()}\n")


def compute_conserved(key, minutes, w, out):
    """One point of the conserved mapping (main case), in its own child process: run_fpr of this module is replaced by
    run_fpr_conserved and the point is computed by compute (compute and the functions in COMPUTE are unchanged)."""
    global run_fpr
    run_fpr = run_fpr_conserved
    compute(key, minutes, w, out)


CONSERVED = ("run_fpr_conserved", "compute_conserved")     # functions additionally hashed for the conserved mapping


# ---------------------------------------------------------------- aggregation
def committed(dmap=None, refs=None):
    """Released readings: system T1' and the plateaus of the two rules. refs: the paths given on the command line, system
    (system.csv of the system bounds), plateau = {yard: path} (Okpo: CSV with the macros YuResPlateau, YuRefPlateau in column
    r1833; Yantai: JSON summary of the saturation runs), conserved (directory of the conserved mapping: the conserved rows of
    bounds.csv and of sat_summary.csv, released as saturation.csv; used when dmap = "conserved")."""
    if dmap == "conserved":
        d = Path(refs["conserved"])
        rd = lambda n: list(csv.DictReader(open(d / n, encoding="utf-8-sig")))
        sysv = {r["yard"]: float(r["lam16"]) for r in rd("bounds.csv") if r["map"] == "conserved"}
        sat_f = "sat_summary.csv" if (d / "sat_summary.csv").exists() else "saturation.csv"
        sat = {(r["yard"], r["model"]): float(r["plateau"]) for r in rd(sat_f) if r["map"] == "conserved"}
        return sysv, {y: dict(reserve=sat[y, "reserve"], ref=sat[y, "segment"]) for y in YARDS}
    sysv = {r["yard"]: float(r["T1_sys"]) for r in csv.DictReader(open(refs["system"], encoding="utf-8-sig"))
            if r["scen"] == "main" and r["section"] == "whole" and r["yard"] in YARDS}
    sec = {r["macro"]: float(r["r1833"]) for r in csv.DictReader(open(refs["plateau"]["yupu"], encoding="utf-8-sig"))}
    yt = json.loads(Path(refs["plateau"]["yantai"]).read_text(encoding="utf-8"))["sat"]["yantai|main"]["rules"]
    plat = {"yupu": dict(reserve=sec["YuResPlateau"], ref=sec["YuRefPlateau"]),
            "yantai": dict(reserve=yt["whole|reserve"]["plateau"], ref=yt["whole|segment"]["plateau"])}
    return sysv, plat


def system_now(out, dmap=None):
    """System T1' recomputed with the current code and network (bounds.system), to be compared with the released value
    (conserved mapping: task mix conserved_mapping.system_mix)."""
    f = out / "system.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    import conserved_mapping
    from bounds.system import SystemT1, scenario_mix
    from occupancy.network import Net
    d = {}
    for y in YARDS:
        Y = P.YARDS[y]
        p = P.scenario("main", **({"demand_map": dmap} if dmap else {}))
        net = Net(Y["network"], p, Y["roles"])
        mix = (conserved_mapping.system_mix(net, p),) if dmap == "conserved" else scenario_mix(net, p)
        v, bind = SystemT1(net, p, *mix).value(top=3)
        d[y] = dict(T1_sys=v, binding=[x["name"] for x in bind[:3]])
    F.write_json(f, d)
    return d


def dispatch_points(path):
    """Offline-dispatch readings by point key (the fields of the point files of experiments/offline_dispatch.py that aggregate
    reads), from its output directory (points/<key>.json) or from its points.csv (released as
    results/retiming/offline_dispatch.csv; a row is matched to the key by yard and seed, K and days are those of this module)."""
    if not path:
        return {}
    p = Path(path)
    if p.is_dir():
        return {k: json.loads((p / "points" / F.fname(k)).read_text(encoding="utf-8")) for k in keys() if (p / "points" / F.fname(k)).exists()}
    out = {}
    for r in csv.DictReader(open(p, encoding="utf-8-sig")):
        out[f"{r['yard']}|{r['seed']}|{K}|{DAYS}"] = dict(N=int(r["N"]), thr_fpr=float(r["fpr"]), **{v: dict(
            thr=float(r[v]), mid=float(r[f"{v}_mid"]), busy=float(r[f"{v}_busy"]), check=r[f"{v}_check"] == "True",
            idle={c: float(r[f"{v}_idle_{c}"]) for c in IDLE}) for v in ("clr", "def1c")})
    return out


def aggregate(cfg, out, refs, dispatch=None):
    """points.csv of the run in out. refs: released readings (see committed); dispatch: output directory or points.csv of
    experiments/offline_dispatch.py with the same map (see dispatch_points; its columns are NaN without it)."""
    pts = [json.loads((out / "points" / F.fname(k)).read_text(encoding="utf-8")) for k in keys() if (out / "points" / F.fname(k)).exists()]
    if not pts:
        return
    dmap = cfg.get("demand_map")
    sysv, plat = committed(dmap, refs)
    now = system_now(out, dmap)
    for y in YARDS:
        assert abs(now[y]["T1_sys"] - sysv[y]) < 0.01, (y, now[y]["T1_sys"], sysv[y])
    disp = dispatch_points(dispatch)
    rows = []
    for r in pts:
        y, S = r["yard"], sysv[r["yard"]]
        row = dict(yard=y, seed=r["seed"], K=r["K"], N=r["N"], windows=r["windows"], W0_h=r["orig"]["W"] / 3600, lamS=S, lamN=r["lamN"],
                   plat_res=plat[y]["reserve"], plat_ref=plat[y]["ref"], binding=r["binding"][0], window_mismatch=r["window_mismatch"],
                   orig=r["orig"]["thr"], orig_mid=r["orig"]["mid"], orig_check=r["orig"]["check"], orig_busy=r["orig"]["busy"])
        for m in MODES + SAME:
            x = r[m]
            row.update({f"{m}": x["thr"], f"{m}_mid": x["mid"], f"{m}_W_h": x["W"] / 3600, f"{m}_check": x["check"], f"{m}_busy": x["busy"]})
            if m in MODES:
                row.update({f"{m}_rounds": x["rounds"], f"{m}_lazy": x["lazy_pairs"], f"{m}_pairs": x["final_pairs"],
                            f"{m}_nonoverlap": x["nonoverlap_pairs"], f"{m}_edges": x["edges"], f"{m}_edges_order": x["edges_by_kind"]["order"]})
            else:
                row.update({f"{m}_threads": x["threads"], f"{m}_minutes": x["minutes"], f"{m}_iters": x["iters"], f"{m}_accepted": x["accepted"],
                            f"{m}_start": x["start"], f"{m}_W_start_h": x["W_replay"] / 3600, f"{m}_W_lns_h": x["W_lns"] / 3600,
                            f"{m}_pairs_trim": x.get("drive_off_pairs", 0)})
            for c in IDLE:
                row[f"{m}_idle_{c}"] = x["idle"].get(c, 0.0) / x["W"]
            row[f"{m}_idle"] = sum(x["idle"].values()) / x["W"]
        row["same_search"] = 1 - row["same_W_lns_h"] / row["same_W_start_h"]          # share by which the rolling search shortens the completion
        row["same_def1c_search"] = 1 - row["same_def1c_W_lns_h"] / row["same_def1c_W_start_h"]
        # what retiming with the precedence fixed moves earlier (compared directly between the two schedules): empty departures
        # (the postponement of just-in-time departure is dropped, the vehicle waits at the origin instead) and completions
        # (= loaded departure + fixed length)
        o = {j: (a, b) for v in r["orig"]["seq"].values() for j, a, b in v}
        n = {j: (a, b) for v in r["all"]["seq"].values() for j, a, b in v}
        d1 = [o[j][0] - n[j][0] for j in o]
        d2 = [o[j][1] - n[j][1] for j in o]
        row.update(rt_t1_moved=sum(v > 1e-6 for v in d1), rt_t1_max=max(d1), rt_done_moved=sum(v > 1e-6 for v in d2), rt_done_sum=sum(d2),
                   rt_done_max=max(d2), rt_thr_diff=row["all"] / row["orig"] - 1)
        e = disp.get(r["key"])           # offline dispatch (experiments/offline_dispatch.py, the same batch of tasks; both destination bookings)
        assert e is None or (e["N"] == r["N"] and abs(e["thr_fpr"] - r["orig"]["thr"]) < 1e-6), r["key"]
        for v, col in (("clr", "ect"), ("def1c", "ect_def1c")):
            x = e[v] if e else None
            row.update({col: x["thr"] if x else float("nan"), f"{col}_mid": x["mid"] if x else float("nan"),
                        f"{col}_busy": x["busy"] if x else float("nan"), f"{col}_check": x["check"] if x else None,
                        **{f"{col}_idle_{c}": (x["idle"].get(c, 0.0) if x else float("nan")) for c in IDLE}})
            row[f"{col}_idle"] = sum(row[f"{col}_idle_{c}"] for c in IDLE)
        for c in ("orig", "orig_mid", "all", "all_mid", "conflict", "conflict_mid", "same", "same_mid", "same_def1c", "same_def1c_mid", "ect",
                  "ect_mid", "ect_def1c", "ect_def1c_mid", "plat_res", "plat_ref"):
            row[f"{c}_S"] = row[c] / S
            row[f"{c}_N"] = row[c] / row["lamN"]
        row["ect_def1c_vs_clr"] = row["ect_def1c"] / row["ect"] - 1
        # interval [two-day value of full-path reservation on the same batch, lambda^S] = 100 %: timing + precedence (rolling
        # search under the same dispatch) + dispatch + rest, adding up to 100 %
        w = S - row["orig"]
        for sfx, same, ect in (("", "same", "ect"), ("_def1c", "same_def1c", "ect_def1c")):
            row[f"sp_timing{sfx}"] = (row["all"] - row["orig"]) / w
            row[f"sp_order{sfx}"] = (row[same] - row["all"]) / w
            row[f"sp_dispatch{sfx}"] = (row[ect] - row[same]) / w
            row[f"sp_rest{sfx}"] = (S - row[ect]) / w
        row["plat_gap_S"] = (row["orig"] - row["plat_res"]) / S
        rows.append(row)
    rows.sort(key=lambda x: (x["yard"] != "yupu", x["seed"]))
    with open(out / "points.csv", "w", newline="", encoding="utf-8-sig") as f:
        w_ = csv.DictWriter(f, list(rows[0]))
        w_.writeheader()
        w_.writerows(rows)


SAME = ("same", "same_def1c")
IDLE = ("start", "vehicle", "chain", "self", "tail")


COMPUTE = ("run_fpr", "collect", "fpr_windows", "bare_instance", "model_windows", "drive_off_trims", "cp_intervals", "make_win", "times",
           "order_pairs", "pair_edges", "earliest",
           "overloads", "retime", "int_start", "replay_start", "same_dispatch", "int_shift", "lns_full", "thr", "mid_rate", "idle_causes",
           "busy_frac", "window_check", "compute")


def func_sha(path, names, h):
    """Adds these top-level functions of the file (their AST without docstrings) to the hash h: the points depend only on them,
    so changes of the aggregation do not change the hash."""
    import ast
    for n in ast.parse(Path(path).read_text(encoding="utf-8")).body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names:
            for x in ast.walk(n):
                if isinstance(x, (ast.FunctionDef, ast.ClassDef)) and x.body and isinstance(x.body[0], ast.Expr) and \
                        isinstance(x.body[0].value, ast.Constant) and isinstance(x.body[0].value.value, str):
                    x.body = x.body[1:] or [ast.Pass()]
            h.update(n.name.encode() + ast.dump(n).encode())
    return h


def code_sha(extra=()):
    h = hashlib.sha256()
    for d in ("occupancy", "bounds", "exact"):
        for f in sorted((ROOT / d).glob("*.py")):
            h.update(f"{d}/{f.name}".encode() + f.read_bytes())
    return func_sha(__file__, COMPUTE + extra, h).hexdigest()


def config(minutes, dmap=None):
    """Configuration of a run (config.json). dmap = "conserved": the conserved mapping also hashes the functions in CONSERVED
    and records the code of its task stream (gen_sha256)."""
    cfg = dict(title="whole-network offline retiming", yards=list(YARDS), seeds=list(SEEDS), K=K, days=DAYS,
               modes=list(MODES), tol=TOL, scenario="main", rule="reserve (just-in-time departure)", same_minutes=minutes, unit=UNIT, width_scale=WS,
               same_dests=["booking of full-path reservation (destination drive-off 6.4 s, main reading)", "Definition 1(c) (released at the end of unloading)"],
               networks={y: P.YARDS[y]["network"] for y in YARDS},
               code_sha256=code_sha(CONSERVED if dmap else ()), ortools=__import__("ortools").__version__)
    if dmap:
        cfg.update(title=cfg["title"] + ": inter-area mapping with erection flow conserved (main case)", demand_map=dmap,
                   gen_sha256=F.conserved_sha())
    return cfg


def main():
    import subnet_exact as SUB
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=30.0, help="minutes of the same-dispatch offline rebooking (rolling large-neighbourhood search) per point")
    ap.add_argument("--slots", default=None, help="CPU slot directory (default: environment variable CPU_SLOTS, else none)")
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--max-threads", type=int, default=8, help="CP-SAT threads (CPU slots) per point at most; 1 runs the points in parallel")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--map", choices=("current", "conserved"), default="current",
                    help="demand mapping: current = proportional inter-area mapping; conserved = inter-area mapping with erection flow conserved (main case)")
    ap.add_argument("--system", help="system bounds, e.g. results/bounds/system_by_granularity.csv")
    ap.add_argument("--plateau-okpo", help="plateaus of Okpo rev 1833, e.g. results/saturation/okpo_readings.csv")
    ap.add_argument("--plateau-yantai", help="Yantai saturation summary, e.g. results/saturation/rules_summary.json")
    ap.add_argument("--conserved-report", help="main-case directory with bounds.csv and saturation.csv, e.g. results/main_case")
    ap.add_argument("--dispatch", help="output directory or points.csv of experiments/offline_dispatch.py with the same map, "
                                       "e.g. results/retiming/offline_dispatch.csv")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    cons = a.map == "conserved"
    cfg, out = config(a.minutes, "conserved" if cons else None), Path(a.out)
    (out / "points").mkdir(parents=True, exist_ok=True)
    if (out / "config.json").exists() and json.loads((out / "config.json").read_text(encoding="utf-8")) != json.loads(json.dumps(cfg)):
        sys.exit(f"{out} holds a run of another configuration")
    F.write_json(out / "config.json", cfg)
    ks = keys()
    pf = lambda k: out / "points" / F.fname(k)
    prog = F.Progress(out / "progress.json", len(ks), sum(pf(k).exists() for k in ks), 0, cfg["title"], "retime")
    refs = dict(system=a.system, plateau=dict(yupu=a.plateau_okpo, yantai=a.plateau_yantai), conserved=a.conserved_report)

    def agg():
        if a.conserved_report if cons else a.system and a.plateau_okpo and a.plateau_yantai:
            aggregate(cfg, out, refs, a.dispatch)
        else:
            print("points.csv needs the released readings (--system, --plateau-okpo, --plateau-yantai; --conserved-report with --map conserved)")
    if not a.aggregate_only:
        # each point in its own process on up to --max-threads CP-SAT threads (subnet_exact.schedule); a point starts with one
        # slot (the subproblems of the rolling search are small, a single thread makes progress)
        SUB.OUT, SUB.MIN_W, SUB.MAX_W = out, 1, a.max_threads
        deadline = time.time() + 60 * a.max_minutes if a.max_minutes else None
        if SUB.schedule([k for k in ks if not pf(k).exists()], a.minutes, pf, prog, deadline, a.slots,
                        target=compute_conserved if cons else compute):
            prog.write(state="paused", note="stopped at --max-minutes; the same command resumes")
            agg()
            sys.exit(3)
    prog.write(state="aggregating")
    agg()
    prog.write(state="done" if all(pf(k).exists() for k in ks) else "paused", processes=0)
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()

"""Full-path reservation with execution in booked order (order-preserving execution, rule reserve_op), and a self-checking
executor for its deadlock-freedom proposition.

A subclass of occupancy.sim.Sim (occupancy/ is not modified): dispatch and booking are those of full-path reservation (the
search of plan, just-in-time departure, the nominal reservation table); the execution is new.

Rule (the paper's definition of execution in booked order): at dispatch the whole chain is booked into the reservation table
with nominal durations (as under reserve) and never rebooked. The bookings on each resource are totally ordered by the key
(booked start, dispatch number). After reaching decision point x (the decision points of the empty and the loaded run are
numbered consecutively) the vehicle leaves x at the earliest time at which all of the following hold:
  R1 not before the booked time beta(x); R2 the vehicle has arrived (previous event + actual driving or loading time);
  adm (main rule): R3 for every booking b that x acquires, all bookings on the same resource with a smaller key have been
      acquired (entry in booked order); R4 the width fits (actual occupancy).
  rel (variant): R3' all bookings on the same resource with a smaller key whose booked end <= the booked start of b have been
      released; the width is not checked (capacity follows from a lemma of the proposition; asserted here).
A booking b is released at the actual time of its release event + the actual timer (>= nominal): roads / junctions after
the clearance / passage time scaled by the speed of this run, the origin at loaded departure + tail, the destination at start
of unloading + actual unloading + clearance to leave, the crane at start of unloading + actual unloading + (tau - t_U).

Checks of the proof, step by step (a violation raises AssertionError):
  P1 capacity: after every acquisition the actual occupancy (per vehicle the largest width, cranes by count) is within capacity;
  P2 waiting edges: under adm, if the width does not fit, everything held on the resource has a smaller key, among it an
     overdue booking with booked end <= s_b, the rest without the overdue ones plus b fits, and the release event of the
     overdue booking is booked strictly before beta(x); under rel the release event waited for is booked strictly before
     beta(x); an order wait points to an acquisition event with a smaller key;
  P3 actual times are never earlier than booked; P4 at nominal times every event is at its booked time (completion times
  equal those of reserve); P5 after the end of the horizon dispatching stops and the network drains: every dispatched task
  completes.
A blocked chain is retried only when a resource it is to acquire is acquired or released (wait_on); at equal times in the
order of the keys.

Self-check on small instances (the synthetic networks of tests/helpers.py):
  python experiments/ordered.py --out DIR [--seeds 100] [--procs 23] [--slots DIR]
  -> DIR/cases.csv (one row per instance, seed, times and rule), DIR/summary.json, DIR/tables.md.
Yard runs: runner.compute with the rule name reserve_op (yard_sim); random times as in stochastic.py (p["stoch"], common
random numbers)."""
import argparse
import collections
import csv
import functools
import hashlib
import heapq
import json
import os
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from occupancy import sim as S                      # noqa: E402
from occupancy.network import EPS                   # noqa: E402
from stochastic import hand_time, speed_factor      # noqa: E402

TOL = 1e-6          # the tolerance of the reservation table (EPS of Sim.conflict)


class Bk:
    """A booking: resource, width, booked period [s, e), chain and acquisition event it belongs to, booked time rb of its
    release event; ta / tr are the actual times of acquisition / release."""
    __slots__ = ("res", "d", "s", "e", "cid", "vid", "ev", "rb", "ta", "tr")

    def __init__(self, res, d, s, e, cid, vid, ev):
        self.res, self.d, self.s, self.e, self.cid, self.vid, self.ev = res, d, s, e, cid, vid, ev
        self.rb, self.ta, self.tr = None, None, None

    def key(self):
        return (self.s, self.cid)


class OPE(S.Sim):
    """Full-path reservation + execution in booked order. times(sim, task) -> (loading, unloading, empty speed factor,
    loaded speed factor); None: by p["stoch"] (drawn as in stochastic.py) or nominal. Bookings use nominal durations (load_s,
    unload_s and speeds of p); with p["stoch"]["book"] = "mean" they use the mean durations."""

    def __init__(self, net, p, K, seed, level_n=None, tasks=None, rule="adm", times=None):
        assert rule in ("adm", "rel")
        self.rule, self.times, self.seed = rule, times, seed
        # lower bounds of random handling (nominal); factor of the booking speed over the nominal speed
        self.h0, self.fbar = (p["load_s"], p["unload_s"]), 1.0
        s = p.get("stoch") or {}
        # booking with mean durations: mean handling, the speed whose driving time is the mean driving time
        if s.get("book") == "mean":
            lo, hi = s["speed"]
            self.fbar = (hi - lo) / math.log(hi / lo) if hi > lo else lo
            p = dict(p, load_s=s["mean"] * p["load_s"], unload_s=s["mean"] * p["unload_s"],
                     v_empty=p["v_empty"] * self.fbar, v_loaded=p["v_loaded"] * self.fbar)
        super().__init__(net, p, "reserve", K, seed, level_n, False, tasks)
        # resource -> bookings not yet released (released ones block nobody and are removed)
        self.bk = collections.defaultdict(list)
        self.chains, self.blocked, self.draining = [], set(), False
        # resource -> chains blocked at a decision point that acquires it (retried only on its acquisitions and
        # releases)
        self.wait_on = collections.defaultdict(set)
        self.n = collections.Counter()                # self-check counters
        self.late_max = 0.0
        self._cap, self._booking, self._got = None, False, set()

    # ---------------------------------------------------------------- durations
    def draw(self, task):
        p = self.p
        if self.times is not None:
            return self.times(self, task)
        return stoch_times(self, task) if p.get("stoch") else (p["load_s"], p["unload_s"], 1.0, 1.0)

    def account(self, W):
        """The occupancy readings record actual occupancy (at release, on_rel), not the booked windows. Two overlapping
        bookings of one vehicle on one resource (clearance at a roadside destination and the next empty run) are counted
        twice; only seconds, and the readings are not used for conclusions."""
        if not self._booking:
            super().account(W)

    # ------------------- event queue: at equal times releases first, then arrivals and dispatch, last the departure attempts
    def push(self, t, kind, x, prio=1):
        # reserve's own completion event: under execution in booked order completion is at the actual end of
        # unloading (on_fin)
        if kind == "done":
            return
        self.seq += 1
        heapq.heappush(self.ev, (t, prio, self.seq, kind, x))

    def dispatch(self, t):
        if not self.draining:
            super().dispatch(t)

    def run(self):
        self.dispatch(0.0)
        while self.ev:
            t, _, _, kind, x = heapq.heappop(self.ev)
            if t > self.H:
                # after the horizon: no more dispatch, the dispatched tasks are completed (P5)
                self.draining = True
            getattr(self, "on_" + kind)(t, x)
        stuck = [c for c in self.chains if not c["done"]]
        self.n["deadlock"] += bool(stuck)
        assert not stuck and not self.blocked, f"deadlock: {len(stuck)} chains not completed"
        return self

    # ---------------------------------------------------------------- dispatch: the bookings of reserve, then the chain
    def commit(self, vid, now, W):
        if self._cap is not None:
            self._cap += [w for w in W if w[2] > w[1] and w[3] > EPS]
        super().commit(vid, now, W)

    def plan(self, v, task, t0):
        assert not task.get("lift"), "the self-checking executor has no lifts inside the crane span"
        legs = self.legs(v, task)                  # plan moves v.loc to the destination: take the trips first
        self._cap, self._booking = [], True
        super().plan(v, task, t0)
        W, self._cap, self._booking = self._cap, None, False
        assert "tc" not in task["plan"], "the self-checking executor has no dock-side waiting place"
        self.chains.append(self.build(v, task, t0, legs, W))

    def build(self, v, task, t0, legs, W):
        so, sd, d_so, d_sd, (TE, stE, uE, aE), (TL, stL, uL, aL), tail, clr = legs
        t1, t2 = task["plan"]["t1"], task["plan"]["t2"]
        load, unload, fE, fL = self.draw(task)
        p, cid = self.p, len(self.chains)
        ev = [dict(beta=t1 + stE[k], dur=(stE[k] - stE[k - 1]) / fE if k else 0.0, acq=[], rel=[]) for k in range(aE + 1)]
        nE = len(ev)
        ev += [dict(beta=t2 + stL[k], dur=(stL[k] - stL[k - 1]) / fL if k else load, acq=[], rel=[]) for k in range(aL + 1)]
        new = lambda res, d, s, e, a: Bk(res, d, s, e, cid, v.id, a)

        def add(b, r, timer):
            ev[b.ev]["acq"].append(b)
            ev[r]["rel"].append((b, timer))
            b.rb = ev[r]["beta"]
        for res, d, a, r, dl in uE:
            if d <= EPS:
                continue
            if r is None:           # origin stop resource: released at loaded departure + tail
                add(new(res, d_so, t1 + stE[a], t2 + tail, a), nE, tail / fL)
            else:
                add(new(res, d, t1 + stE[a], t1 + stE[r] + dl, a), r, dl / fE)
        tau, U = p.get("crane_cycle_s"), p["unload_s"]
        for res, d, a, r, dl in uL:
            if d <= EPS:
                continue
            if res[0] == "c":       # crane: acquired at the start of unloading, held tau - t_U after unloading
                add(new(res, d, t2 + TL, t2 + TL + tau, nE + aL), nE + aL, unload + tau - U)
            elif r is None:         # destination stop resource: released at unloading done + clearance to leave
                add(new(res, d_sd, t2 + stL[a], t2 + TL + U + clr, nE + a), nE + aL, unload + clr / fE)
            else:
                add(new(res, d, t2 + stL[a], t2 + stL[r] + dl, nE + a), nE + r, dl / fL)
        mine = sorted((b.res, b.s, b.e, b.d) for x in ev for b, _ in x["rel"])
        assert len(mine) == len(W) and all(a[0] == b[0] and abs(a[1] - b[1]) < TOL and abs(a[2] - b[2]) < TOL and abs(a[3] - b[3]) < 1e-9
                                           for a, b in zip(mine, sorted(W))), "chain does not match the booked windows of reserve"
        for x in ev:
            for b in x["acq"]:
                self.bk[b.res].append(b)
        ch = dict(cid=cid, v=v, task=task, ev=ev, k=0, ready=t0, U=unload, done=False, t0=t0,
                  # actual driving, loading and unloading time (the rest is waiting)
                  active=sum(x["dur"] for x in ev) + unload)
        self.push(max(t0, ev[0]["beta"]), "try", ch, 2)
        return ch

    # ---------------------------------------------------------------- execution
    def width_ok(self, res, add):
        """add = [(vehicle, width)]: does the actual occupancy of res fit with it added (per vehicle the largest width;
        cranes by count, an earlier one of the same vehicle included)?"""
        act = [b for b in self.bk[res] if b.ta is not None]
        if res[0] == "c":
            return len(act) + len(add) <= self.cap[res] + EPS
        per = collections.defaultdict(float)
        for b in act:
            per[b.vid] = max(per[b.vid], b.d)
        for vid, d in add:
            per[vid] = max(per[vid], d)
        return sum(per.values()) <= self.cap[res] + EPS

    def blockers(self, ch, x):
        """Why x cannot depart: [(kind, booking)]; empty = may depart. Checks P2 item by item on the way."""
        out, beta = [], x["beta"]
        for b in x["acq"]:
            same = lambda b2: b2.cid == b.cid and b2.ev == b.ev
            earlier = [b2 for b2 in self.bk[b.res] if b2.key() < b.key() and not same(b2)]
            if self.rule == "rel":
                for b2 in earlier:
                    # booked to end before b, not yet released (bk holds only unreleased ones)
                    if b2.e <= b.s + TOL:
                        assert b2.rb < beta - 1e-9, ("P2: waiting edge does not point to an earlier event", b2.rb, beta)
                        out.append(("rel", b2))
                continue
            for b2 in earlier:
                if b2.ta is None:                     # order wait: a booking earlier in the order has not entered yet
                    assert b2.s <= b.s
                    out.append(("order", b2))
        if out or self.rule == "rel":
            return out
        for res in {b.res for b in x["acq"]}:
            new = [(ch["v"].id, b.d) for b in x["acq"] if b.res == res]
            if self.width_ok(res, new):
                continue
            s_b = min(b.s for b in x["acq"] if b.res == res)
            act = [b2 for b2 in self.bk[res] if b2.ta is not None]
            assert all(b2.key() < (s_b, ch["cid"]) or (b2.cid == ch["cid"]) for b2 in act), "P2: a booking with a larger key is held"
            over = [b2 for b2 in act if b2.e <= s_b + TOL]
            assert over, "P2: does not fit but no booking is overdue"
            assert all(b2.rb < beta - 1e-9 for b2 in over), "P2: release event of an overdue booking not earlier than beta(x)"
            rest = [b2 for b2 in act if b2.e > s_b + TOL]
            per = collections.defaultdict(float)
            for b2 in rest:
                per[b2.vid] = max(per[b2.vid], b2.d)
            for vid, d in new:
                per[vid] = max(per[vid], d)
            assert (len(rest) + len(new) if res[0] == "c" else sum(per.values())) <= self.cap[res] + EPS, "P2: does not fit without the overdue ones"
            out += [("cap", b2) for b2 in over]
        return out

    def attempt(self, ch, t):
        """Tries to leave the current decision point; on success self._got = the resources acquired. A blocked chain is
        filed in wait_on under each resource it is to acquire."""
        x, cid = ch["ev"][ch["k"]], ch["cid"]
        if t < x["beta"] - 1e-9 or t < ch["ready"] - 1e-9:
            return False
        why = self.blockers(ch, x)
        if why:
            if cid not in self.blocked:
                self.n["wait_" + why[0][0]] += 1
                self.blocked.add(cid)
                ch["wres"] = {b.res for b in x["acq"]}
                for r in ch["wres"]:
                    self.wait_on[r].add(cid)
            return False
        if cid in self.blocked:
            self.blocked.discard(cid)
            for r in ch.pop("wres"):
                self.wait_on[r].discard(cid)
        assert t >= x["beta"] - 1e-9                                   # P3
        self.late_max = max(self.late_max, t - x["beta"])
        self.n["events"] += 1
        self.n["late_events"] += t > x["beta"] + 1e-6
        x["t"] = t
        for b in x["acq"]:
            b.ta = t
        self._got = {b.res for b in x["acq"]}
        for res in self._got:                                          # P1
            assert self.width_ok(res, []), ("P1: over capacity", res)
        self.n["acq"] += len(x["acq"])
        for b, timer in x["rel"]:
            self.push(t + timer, "rel", b, 0)
        ch["k"] += 1
        if ch["k"] == len(ch["ev"]):
            self.push(t + ch["U"], "fin", ch, 1)
        else:
            nx = ch["ev"][ch["k"]]
            ch["ready"] = t + nx["dur"]
            self.push(max(ch["ready"], nx["beta"]), "try", ch, 2)
        return True

    def wake(self, t, res):
        """An acquisition or release on res: retry the chains blocked on it (in key order), then on the resources they
        acquire, until none can depart."""
        todo = set(res)
        while todo:
            cids = {c for r in todo for c in self.wait_on.get(r, ())}
            todo = set()
            for cid in sorted(cids, key=lambda c: (self.chains[c]["ev"][self.chains[c]["k"]]["beta"], c)):
                if cid in self.blocked and self.attempt(self.chains[cid], t):
                    todo |= self._got

    def on_try(self, t, ch):
        if ch["cid"] in self.blocked or ch["done"] or not self.attempt(ch, t):
            return
        self.wake(t, self._got)

    def on_rel(self, t, b):
        b.tr = t
        self.bk[b.res].remove(b)
        self.account([(b.res, b.ta, t, b.d)])
        self.wake(t, (b.res,))

    def on_fin(self, t, ch):
        # events and bookings of the chain are no longer needed (unreleased bookings are held by their release
        # events)
        ch["done"], ch["ev"] = True, None
        v, task = ch["v"], ch["task"]
        self.finish(v, task, t, max(0.0, t - ch["t0"] - ch["active"]))
        self.idle.append(v.id)
        self.dispatch(t)

    def dispatched(self):
        """All dispatched tasks: the completed records + the chains not completed (for the steady-state criterion)."""
        return list(self.done) + [c["task"] for c in self.chains if not c["done"]]

    def metrics(self):
        m = super().metrics()
        m["ope"] = dict(wait_order=self.n["wait_order"], wait_cap=self.n["wait_cap"], wait_rel=self.n["wait_rel"],
                        late_max_s=self.late_max, late_events=self.n["late_events"], events=self.n["events"], tasks=len(self.chains))
        return m

    def summary(self):
        return dict(tasks=len(self.chains), events=self.n["events"], acquisitions=self.n["acq"], wait_order=self.n["wait_order"],
                    wait_cap=self.n["wait_cap"], wait_rel=self.n["wait_rel"], deadlock=self.n["deadlock"],
                    late_max_s=self.late_max, late_events=self.n["late_events"])


def yard_sim(net, p, mode, K, seed, level_n, record=False, tasks=None):
    """Sim interface of runner.compute (rule name reserve_op): full-path reservation + execution in booked order, times by
    p["stoch"] (as in stochastic.py)."""
    assert mode == "reserve_op"
    return OPE(net, p, K, seed, level_n, tasks)


def code_sha256():
    """The results of reserve_op depend on this file and on stochastic.py (how random times are drawn); this SHA-256 goes
    into the configuration of the yard batches (runner.ordered_sha)."""
    return hashlib.sha256(b"".join((ROOT / "experiments" / n).read_bytes() for n in ("ordered.py", "stochastic.py"))).hexdigest()


# ---------------------------------------------------------------- random times (never shorter than nominal)
def stoch_times(sim, task, s=None):
    """As stochastic.Sim.draw (common random numbers): one random stream per task (seed, task id), handling h0 + lognormal,
    uniform speed factors; s defaults to p["stoch"]."""
    s = s or sim.p["stoch"]
    rng = random.Random(f"stoch|{sim.seed}|{task['id']}")
    u = [min(max(rng.random(), 1e-12), 1 - 1e-12) for _ in range(4)]
    hl, hu = sim.h0
    # trips are built with the booking speed; the actual speed factor is taken relative to it
    b = sim.fbar
    return hand_time(hl, s, u[0]), hand_time(hu, s, u[1]), speed_factor(s, u[2]) / b, speed_factor(s, u[3]) / b


def stoch_cv_times(cv, lo=0.9):
    """Handling with mean 1.5 h0 and CV cv, speed factor U[lo, 1] (for the small instances)."""
    return functools.partial(stoch_times, s=dict(mean=1.5, cv=cv, speed=[lo, 1.0]))


def heavy_times(sim, task):
    """Heavy tail, occasionally very slow: 30 % of the tasks get Exp(mean 2 h0) added to loading and to unloading, speed
    factor U[0.2, 1] (a run now and then slowed to a fifth)."""
    rng = random.Random(f"heavy|{sim.seed}|{task['id']}")
    h = lambda h0: h0 * (1 + (rng.expovariate(0.5) if rng.random() < 0.3 else 0.0))
    return h(sim.p["load_s"]), h(sim.p["unload_s"]), rng.uniform(0.2, 1.0), rng.uniform(0.2, 1.0)


def one_late(sim, task):
    """Only the vehicles of even-numbered tasks are slow: loading + 1 h, loaded speed factor 0.25 ("one vehicle overdue",
    the others nominal)."""
    p = sim.p
    return (p["load_s"] + 3600.0, p["unload_s"], 1.0, 0.25) if task["id"] % 2 == 0 else (p["load_s"], p["unload_s"], 1.0, 1.0)


TIMES = {"nominal": None, "stoch_cv6": stoch_cv_times(0.6), "heavy": heavy_times, "one_late": one_late}


# -------------------------------------------------------------- small instances (the synthetic networks of tests/helpers.py)
@functools.lru_cache(maxsize=None)
def cases():
    sys.path.insert(0, str(ROOT / "tests"))
    import helpers as TO
    from occupancy import params as P
    corridor = TO.load_net(TO.make_map(TO.CORRIDOR, TO.CORRIDOR_EDGES, {"fA": ("A", "assembly_workshop"), "fB": ("B", "paint_workshop")}))
    out = {"corridor_head_on": (corridor, [TO.task(0, "A", "B"), TO.task(1, "B", "A")], ["A", "B"])}
    wide = wide_net()
    rng = random.Random(7)
    out["wide_road"] = (wide, [dict(TO.task(i, *rng.sample("ABCD", 2), release=rng.uniform(0, 4000)), B=5.0) for i in range(30)],
                        ["A", "B", "C", "D", "A"])
    out["wide_overdue"] = (wide, WIDE3, ["A", "B", "C"])
    for n in (3, 4, 6):
        out[f"ring{n}"] = (TO.ring(n), [TO.task(i, f"N{i}", f"N{(i + 1) % n}") for i in range(n)] +
                           [TO.task(n + i, f"N{(i + 2) % n}", f"N{i % n}", release=600.0 * (i + 1)) for i in range(2 * n)], [f"N{i}" for i in range(n)])
    for sd in range(1, 9):
        out[f"random{sd}"] = TO.random_case(sd)
    sec = TO.sectioned_corridor()[1]
    out["sectioned"] = (sec, [dict(TO.task(i, *random.Random(i).sample(sorted(sec.stops), 2), release=900.0 * i), B=8.0) for i in range(24)],
                        [sec.stops[s]["node"] for s in sorted(sec.stops)][:4])
    return out, P


# Three vehicles on the wide road, "two bookings overlap, one vehicle overdue" (worked by hand): v0 A->D, v1 B->A (released
# at 200 s), v2 C->B (released at 300 s); on the 15 m road the bookings of the three touch or overlap in pairs, never more
# than two at a time
WIDE3 = [dict(id=0, flow="T", o="A", d="D", B=5.0, Lv=20.0, release=0.0), dict(id=1, flow="T", o="B", d="A", B=5.0, Lv=20.0, release=200.0),
         dict(id=2, flow="T", o="C", d="B", B=5.0, Lv=20.0, release=300.0)]


def wide_net():
    """Wide-road instance: A -(20 m)- n1 -(400 m, 15 m wide)- n2 -(20 m)- B, n1 also joined to C and n2 to D (40 m); the
    connecting roads are 31 m wide. The 15 m road has capacity 16 m and a loaded vehicle with B = 5 m takes 6 m: two
    vehicles pass side by side, three do not fit, so bookings often overlap in pairs, and when one vehicle is overdue a
    third waits."""
    sys.path.insert(0, str(ROOT / "tests"))
    import helpers as TO
    m = TO.make_map(dict(A=(0, 0), n1=(20, 0), n2=(420, 0), B=(440, 0), C=(20, 40), D=(420, 40)),
                    [("A", "n1"), ("n1", "n2"), ("n2", "B"), ("n1", "C"), ("n2", "D")],
                    {"fA": ("A", "assembly_workshop"), "fB": ("B", "paint_workshop"), "fC": ("C", "outfitting_workshop"),
                     "fD": ("D", "assembly_workshop")},
                    width={"r0": 31.0, "r1": 15.0, "r2": 31.0, "r3": 31.0, "r4": 31.0})
    return TO.load_net(m)


def run_case(name, net, tasks, locs, p, seed, times, rule):
    sim = OPE(net, p, len(locs), seed, tasks=[dict(t) for t in tasks], rule=rule, times=TIMES[times])
    for v, loc in zip(sim.vs, locs):
        v.loc = loc
    sim.run()
    out = dict(case=name, seed=seed, times=times, rule=rule, **sim.summary())
    # P4: at nominal times every event is at its booked time, completions as under reserve
    if times == "nominal":
        ref = S.Sim(net, p, "reserve", len(locs), seed, None, tasks=[dict(t) for t in tasks])
        for v, loc in zip(ref.vs, locs):
            v.loc = loc
        ref.run()
        a = sorted((r["id"], r["done"]) for r in sim.done if r["done"] <= sim.H)
        b = sorted((r["id"], r["done"]) for r in ref.done)
        out["nominal_late_max_s"] = sim.late_max
        out["same_as_reserve"] = len(a) == len(b) and all(x[0] == y[0] and abs(x[1] - y[1]) < 1e-6 for x, y in zip(a, b))
        assert sim.late_max < 1e-6 and out["same_as_reserve"], ("P4", name, seed, sim.late_max)
    return out


def job(args):
    name, seed, times, rule = args
    cs, P = cases()
    net, tasks, locs = cs[name]
    p = P.scenario("main", days=1, warmup_days=0)
    # random small networks: origins, destinations and releases redrawn per seed
    if name.startswith("random"):
        rng = random.Random(1000 * seed + int(name[6:]))
        stops = sorted(net.stops)
        tasks = [dict(t, o=a, d=b, release=rng.uniform(0, 30000)) for t in tasks for a, b in [rng.sample(stops, 2)]]
    t0 = time.time()
    try:
        out = run_case(name, net, tasks, locs, p, seed, times, rule)
        out["error"] = ""
    except AssertionError as e:
        out = dict(case=name, seed=seed, times=times, rule=rule, error=repr(e))
    # control: same instance, same random times, nominal booking + late rebooking (stochastic.py)
    if times == "stoch_cv6" and rule == "adm":
        import stochastic
        q = dict(p, stoch=dict(mean=1.5, cv=0.6, speed=[0.9, 1.0], book_q=None))
        rb = stochastic.Sim(net, q, "reserve_rb", len(locs), seed, None, tasks=[dict(t) for t in tasks])
        for v, loc in zip(rb.vs, locs):
            v.loc = loc
        rb.run()
        out.update(rb_deadlocks=len(rb.deadlocks), rb_done=len(rb.done))
    out["secs"] = round(time.time() - t0, 2)
    return out


def code_sha():
    h = hashlib.sha256()
    for f in sorted((ROOT / "occupancy").glob("*.py")) + [Path(__file__), ROOT / "experiments" / "stochastic.py"]:
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser(description="Self-check of execution in booked order on the small instances.")
    ap.add_argument("--out", required=True, help="output directory (cases.csv, summary.json, tables.md)")
    ap.add_argument("--slots", default=os.environ.get("CPU_SLOTS"), help="CPU-slot directory (optional; default $CPU_SLOTS)")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--procs", type=int, default=23)
    a = ap.parse_args()
    names = list(cases()[0])
    todo = [(n, sd, tm, r) for n in names for sd in range(1, a.seeds + 1) for tm in TIMES for r in ("adm", "rel")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = pool_map(job, todo, a.procs, a.slots)
    rows.sort(key=lambda r: (r["case"], r["times"], r["rule"], r["seed"]))
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("case", "seed", "times", "rule"), k))
    with open(out / "cases.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    agg = collections.defaultdict(collections.Counter)
    for r in rows:
        g = agg[(r["case"], r["times"], r["rule"])]
        g["runs"] += 1
        g["errors"] += bool(r["error"])
        for k in ("tasks", "events", "acquisitions", "wait_order", "wait_cap", "wait_rel", "deadlock", "late_events", "rb_deadlocks"):
            g[k] += r.get(k, 0)
        g["late_max_s"] = max(g["late_max_s"], r.get("late_max_s", 0.0))
    summ = dict(code_sha=code_sha(), runs=len(rows), errors=sum(bool(r["error"]) for r in rows),
                deadlocks=sum(r.get("deadlock", 0) for r in rows), tasks=sum(r.get("tasks", 0) for r in rows),
                wait_cap=sum(r.get("wait_cap", 0) for r in rows), wait_order=sum(r.get("wait_order", 0) for r in rows),
                wait_rel=sum(r.get("wait_rel", 0) for r in rows),
                nominal_same_as_reserve=sum(bool(r.get("same_as_reserve")) for r in rows if r["times"] == "nominal"),
                nominal_runs=sum(r["times"] == "nominal" for r in rows),
                rb_runs=sum("rb_deadlocks" in r for r in rows), rb_runs_deadlocked=sum(r.get("rb_deadlocks", 0) > 0 for r in rows),
                rb_deadlocks=sum(r.get("rb_deadlocks", 0) for r in rows),
                groups={"|".join(k): dict(v) for k, v in sorted(agg.items())})
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["| instance | times | rule | runs | tasks | order waits | capacity waits (blocked by an overdue vehicle) | release waits | "
             "deadlocks | assertion failures | max lateness (s) | deadlocks of late rebooking |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (c, tm, r), g in sorted(agg.items()):
        rb = g["rb_deadlocks"] if tm == "stoch_cv6" and r == "adm" else "-"
        lines.append(f"| {c} | {tm} | {r} | {g['runs']} | {g['tasks']} | {g['wait_order']} | {g['wait_cap']} | {g['wait_rel']} | "
                     f"{g['deadlock']} | {g['errors']} | {g['late_max_s']:.0f} | {rb} |")
    (out / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summ.items() if k != "groups"}, ensure_ascii=False))
    print(out)


def pool_map(fn, todo, procs, slots_dir, tag="ordered"):
    """Computes fn(x) for x in todo in at most procs processes and returns the results in the order of todo; with a CPU-slot
    directory, takes a slot for every item submitted and returns it when done (like runner.slot_imap)."""
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    slots = None
    if slots_dir:
        sys.path.insert(0, slots_dir)
        from cpu_slots import Slots
        slots = Slots(f"{tag}-{os.getpid()}")
    out, todo, live = {}, list(enumerate(todo)), {}
    with ProcessPoolExecutor(procs) as ex:
        while todo or live:
            if slots:
                slots.want(len(todo) + len(live))
            while todo and len(live) < procs and (slots is None or slots.take()):
                i, k = todo.pop(0)
                live[ex.submit(fn, k)] = i
            if not live:
                time.sleep(0.5)
                continue
            done, _ = wait(live, timeout=0.5, return_when=FIRST_COMPLETED)
            for f in done:
                out[live.pop(f)] = f.result()
                if slots:
                    slots.give()
    if slots:
        slots.close()
    return [out[i] for i in sorted(out)]


if __name__ == "__main__":
    main()

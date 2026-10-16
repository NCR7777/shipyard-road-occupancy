"""Full-path reservation + execution in booked order, booking on projected occupancy (rule reserve_op2); in addition lifts
inside the crane span and an audit record for execution in booked order.

A subclass OPE2 of ordered.OPE (ordered.py is not changed by it):

Booking on projected occupancy (proj=True): at dispatch the actual period of every unreleased booking is first projected,
then the search of full-path reservation looks for a gap in the projected occupancy table; execution keeps the key order
((O1)-(O3) unchanged). The projection is a longest path over the pending decision points, computed in one pass in the order
of the key (booked time, chain number):
  projected time of decision point x = max(booked time beta(x), projected arrival at x, on every resource x enters: the
  projected entry of the bookings with a smaller key ((O2)) and the projected release of the bookings with a smaller key
  that should end before it starts by the booking (an overdue one is waited for, a conservative estimate of (O3)));
  projected arrival at x = projected time of the previous decision point + nominal duration; for the current decision
  point: arrived (observable) -> now, not arrived -> max(now, actual time of the previous decision point + nominal
  duration); projected release of a booking = projected time of its release event + nominal timer, for a running timer
  max(now, start + nominal timer).
In the occupancy table every unreleased booking takes [booked start, max(booked end, projected release)) (per vehicle and
resource merged with the piecewise maximum width, cranes each on their own), and a new booking is placed only where it
fits; every unreleased booking takes at least its booked period, so the lemma that unreleased bookings fit within their
booked periods still holds and the proposition covers the rule. At nominal times projection = booking, event by event the
same as full-path reservation.
Lifts inside the crane span: a chain of one decision point (booked time = start of the booking); after entering the crane
booking the block is hooked, the release is at the actual hook time + (tau - nominal hook time).
Audit record (audit=True): released bookings and the entry order on each resource (checks C1, C2 of ordered_check.py).

Import only (no command line): runner.compute runs it for the rule reserve_op2 (yard_sim2); ordered_check.py and
tail_release.py use OPE2."""
import collections
import hashlib
import heapq
import math
import time
from pathlib import Path

import ordered as O
from occupancy.network import EPS

ROOT = Path(__file__).resolve().parents[1]


class _Lift:
    """The "vehicle" of a lift inside the crane span: only an id (negative, as the lift bookings of occupancy)."""
    def __init__(self, k):
        self.id = k


class OPE2(O.OPE):
    def __init__(self, net, p, K, seed, level_n=None, tasks=None, rule="adm", times=None, proj=False, audit=False):
        self.proj, self.audit = proj, audit
        super().__init__(net, p, K, seed, level_n, tasks, rule, times)
        self.active = {}                     # chains not completed (lifts included)
        self.b_ty = {}                       # booking -> time its release timer started (release event passed)
        self.nlift = 0
        self.pstat = collections.Counter()   # projections: calls, decision points, bookings, time used
        self.log_bk, self.b_seq, self.nseq = [], {}, 0

    # ---------------------------------------------------------------- dispatch
    def plan(self, v, task, t0):
        if self.proj:
            self.table = self.projected_table(t0)
        super().plan(v, task, t0)
        ch = self.chains[-1]
        so, sd, d_so, d_sd, (TE, stE, uE, aE), (TL, stL, uL, aL), *_ = self.legs_of_last
        nE = aE + 1
        # nominal durations (for the projection; the actual duration dur is drawn at random and unknown to the
        # dispatcher)
        for k, x in enumerate(ch["ev"]):
            x["dnom"] = (stE[k] - stE[k - 1] if k else 0.0) if k < nE else (stL[k - nE] - stL[k - nE - 1] if k > nE else self.p["load_s"])
        self.active[ch["cid"]] = ch

    def legs(self, v, task):
        out = super().legs(v, task)
        self.legs_of_last = out
        return out

    def lift(self, t, task):
        """Lift inside the crane span: books the crane period (as full-path reservation), enters in booked order; completed
        when hooked."""
        res, c = self.cranes[self.net.stops[task["d"]]["ent"]], self.p["crane_cycle_s"]
        self.lift_rel += self.t0m <= t < self.H
        if self.proj:
            self.table = self.projected_table(t)
        s = self.earliest([(res, 0.0, c, 1.0)], t, -1, False, [])
        self.nlift += 1
        vid = -self.nlift
        self.commit(vid, t, [(res, s, s + c, 1.0)])
        cid = len(self.chains)
        b = O.Bk(res, 1.0, s, s + c, cid, vid, 0)
        b.rb = s
        hook = self.draw(task)[1] if (self.times is not None or self.p.get("stoch")) else self.p["crane_hook_s"]
        ev = [dict(beta=s, dur=0.0, dnom=0.0, acq=[b], rel=[(b, hook + c - self.p["crane_hook_s"])])]
        self.bk[res].append(b)
        ch = dict(cid=cid, v=_Lift(vid), task=task, ev=ev, k=0, ready=t, U=hook, done=False, t0=t, active=hook, lift=True)
        self.chains.append(ch)
        self.active[cid] = ch
        self.push(max(t, s), "try", ch, 2)

    # ---------------------------------------------------------------- execution (records only)
    def attempt(self, ch, t):
        ok = super().attempt(ch, t)
        if ok:
            x = ch["ev"][ch["k"] - 1]
            for b, _ in x["rel"]:
                self.b_ty[b] = t
            if self.audit:
                for b in x["acq"]:
                    self.nseq += 1
                    self.b_seq[b] = self.nseq
        return ok

    def on_rel(self, t, b):
        super().on_rel(t, b)
        self.b_ty.pop(b, None)
        if self.audit:
            self.log_bk.append(b)

    def on_fin(self, t, ch):
        self.active.pop(ch["cid"], None)
        if not ch.get("lift"):
            return super().on_fin(t, ch)
        start = ch["ev"][0]["t"]
        ch["done"], ch["ev"] = True, None
        self.lifts.append(dict(ch["task"], start=start, done=t))

    def dispatched(self):
        return list(self.done) + [c["task"] for c in self.chains if not c["done"] and not c.get("lift")]

    def metrics(self):
        m = super().metrics()
        n = max(1, self.pstat["calls"])
        m["proj"] = dict(calls=self.pstat["calls"], events_mean=self.pstat["events"] / n, bookings_mean=self.pstat["bookings"] / n,
                         ms_mean=1000 * self.pstat["secs"] / n, secs=self.pstat["secs"])
        return m

    # ---------------------------------------------------------------- projected occupancy
    def projected_table(self, t0):
        """{resource: [(start, end, width, vehicle)]}: every unreleased booking takes [booked start, max(booked end, projected
        release))."""
        tic = time.perf_counter()
        pend = {}
        heaps, M, last_in = collections.defaultdict(list), collections.defaultdict(lambda: -math.inf), collections.defaultdict(lambda: -math.inf)
        for b, ty in self.b_ty.items():                 # timer running: nominal timer
            pend[b] = max(t0, ty + b.e - b.rb)
            heapq.heappush(heaps[b.res], (b.e, pend[b], id(b)))
        evs = []
        for ch in self.active.values():
            for k in range(ch["k"], len(ch["ev"])):
                evs.append((ch["ev"][k]["beta"], ch["cid"], k, ch))
        evs.sort(key=lambda e: (e[0], e[1]))
        proj = {}
        for beta, cid, k, ch in evs:
            x = ch["ev"][k]
            if k > ch["k"]:
                ready = proj[(cid, k - 1)] + x["dnom"]
            elif ch["ready"] <= t0 + 1e-9:
                ready = t0
            else:
                prev = ch["ev"][k - 1]["t"] if k else ch["t0"]
                ready = max(t0, prev + x["dnom"])
            tt = max(beta, ready)
            for b in x["acq"]:
                h, r = heaps[b.res], b.res
                while h and h[0][0] <= b.s + O.TOL:
                    M[r] = max(M[r], heapq.heappop(h)[1])
                tt = max(tt, last_in[r], M[r])
            for b in x["acq"]:
                last_in[b.res] = max(last_in[b.res], tt)
            proj[(cid, k)] = tt
            for b, _ in x["rel"]:
                pend[b] = tt + b.e - b.rb
                heapq.heappush(heaps[b.res], (b.e, pend[b], id(b)))
        table = collections.defaultdict(list)
        nb = 0
        for r, L in self.bk.items():
            by = collections.defaultdict(list)
            for b in L:
                e = max(b.e, pend.get(b, b.e))
                if e > t0:
                    by[b.vid if r[0] != "c" else id(b)].append((b.s, e, b.d))
                nb += 1
            for key, ws in by.items():
                vid = key if r[0] != "c" else next(b.vid for b in L if id(b) == key)
                table[r] += merge_max(ws, vid)
        self.pstat.update(calls=1, events=len(evs), bookings=nb)
        self.pstat["secs"] += time.perf_counter() - tic
        return table


def merge_max(ws, vid):
    """Merges several periods of one vehicle on one resource with the piecewise maximum width (as occupancy.sim.Sim.commit)."""
    if len(ws) == 1:
        s, e, d = ws[0]
        return [(s, e, d, vid)]
    ts = sorted({x for s, e, _ in ws for x in (s, e)})
    out = []
    for a, b in zip(ts, ts[1:]):
        w = max((d for s, e, d in ws if s <= a and e >= b), default=0.0)
        if w <= EPS:
            continue
        if out and out[-1][1] == a and out[-1][2] == w:
            out[-1] = (out[-1][0], b, w, vid)
        else:
            out.append((a, b, w, vid))
    return out


def yard_sim2(net, p, mode, K, seed, level_n, record=False, tasks=None):
    """Sim interface of runner.compute: reserve_op2 = booking on projected occupancy + execution in booked order."""
    assert mode == "reserve_op2"
    return OPE2(net, p, K, seed, level_n, tasks, proj=True)


def code_sha256():
    """The results of reserve_op2 depend on this file, ordered.py and stochastic.py; this SHA-256 goes into the
    configuration of the yard batches (runner.ordered_proj_sha)."""
    return hashlib.sha256(b"".join((ROOT / "experiments" / n).read_bytes() for n in ("ordered_proj.py", "ordered.py", "stochastic.py"))).hexdigest()

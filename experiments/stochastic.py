"""Random handling and driving times, and full-path reservation with late rebooking (rule reserve_rb).

A subclass of occupancy.sim.Sim (occupancy/ itself is not modified). It is switched on by p["stoch"] or by the rule name
reserve_rb (runner.compute picks this class then); the batch configuration records the SHA-256 of this file
(runner.stoch_sha). Serves the robustness of the reservation rule to random handling and driving times in the paper.

Random times (p["stoch"] = {"mean": m, "cv": c, "speed": [lo, hi], "book_q": q or None}, Sim.draw): handling = nominal
value h0 + X with X lognormal, E[h0 + X] = m h0 and CV(h0 + X) = c (hand_time); the speed of every run (one factor for the
empty run, one for the loaded run) is multiplied by a factor uniform on [lo, hi] (speed_factor). Each task has its own random
stream (seed, task id) from which four uniform numbers are drawn in a fixed order and inverted: the task sequence and the
vehicle starts are untouched, and a task sits at the same quantiles under every rule, fleet size and CV (common random
numbers). Used only with free, reserve_rb, segment and segment_safe (the Banker criterion of safe release does not look at
times), not with crane pacing. The handling times of the task enter the existing occupancy code through "the vehicle's
parameter table" (self.p is replaced, during grant and plan, by a copy whose handling times are those drawn for the task),
the speed factors through legs; everything else is unchanged.

reserve_rb (full-path reservation + late rebooking): at dispatch the whole chain is booked into the reservation table with
the booking durations (nominal; the quantile when book_q is given), with the search and just-in-time departure of reserve;
afterwards the vehicle proceeds decision point by decision point as under stepwise request: it moves on at the booked time
and only if the next part fits under the actual occupancy (capacity is still checked event by event). When the vehicle
reaches a decision point later than booked, it withdraws its bookings not yet ended and rebooks the rest of the chain; the
resources it holds are extended in the table to their expected release, without a conflict check (the vehicle is there),
so the table may be overloaded, and vehicles already booked there wait for it by capacity when they arrive. Circular waits
on the road are counted with the deadlock test of stepwise request (wait-for graph reduction) and resolved by the
teleport recovery of the reference rule (deadlock_recovery_s). At nominal times it is event by event the same as reserve
(up to floating-point rounding).

Import only (no command line): runner.compute runs it for the rule reserve_rb and for p["stoch"]."""
import collections
import math
import random
from statistics import NormalDist

from occupancy import sim as S
from occupancy.network import EPS

STOCH_MODES = ("free", "reserve_rb", "segment", "segment_safe")      # rules that take random times
MODES = ("reserve_rb",)                                              # the new rule of this module


def hand_time(h0, s, u):
    """u-quantile of the random handling time: h0 + X, X lognormal, E[h0 + X] = mean h0, CV(h0 + X) = cv (s = p["stoch"])."""
    m, c = s["mean"], s["cv"]
    if c == 0:
        return h0 * m
    r = c * m / (m - 1)                     # coefficient of variation of X
    sig = math.sqrt(math.log1p(r * r))
    return h0 + math.exp(math.log((m - 1) * h0) - sig * sig / 2 + sig * NormalDist().inv_cdf(u))


def speed_factor(s, u):
    """u-quantile of the speed factor: uniform on [lo, hi]."""
    lo, hi = s["speed"]
    return lo + (hi - lo) * u


class Sim(S.Sim):
    def __init__(self, net, p, mode, K, seed, level_n, record=False, tasks=None):
        rb = mode == "reserve_rb"
        if p.get("stoch") and mode not in STOCH_MODES:
            raise ValueError(f"random times are only for {STOCH_MODES}, not for {mode}")
        if (rb or p.get("stoch")) and p.get("crane_n"):
            # ponytail: under random unloading the crane window would become "unloading done + positioning and release",
            # and a rebooking would have to carry the crane window; add when random times under crane pacing are needed
            raise ValueError("reserve_rb and random times are not combined with crane pacing")
        # reserve_rb is built with the execution of stepwise request (stepwise family, no Banker, no direction lock), then
        # given its own rule name back
        super().__init__(net, p, "segment" if rb else mode, K, seed, level_n, record, tasks)
        self.mode, self.rb, self.seed, self.lock = mode, rb, seed, False if rb else self.lock
        self.rb_n = collections.Counter()   # rebookings and late seconds (in the statistics window)
        self._tm = {}
        for v in self.vs:
            # tm = (loading, unloading, empty speed factor, loaded speed factor) of the current task; pt = the vehicle's
            # parameter table (handling of the current task); bt = booked times of the decision points {phase: [time]};
            # bk = trips and durations used for booking; bres = resources on which the vehicle has bookings in the table;
            # rel = resource -> scheduled release time (the latest); hw = resource -> [width, scheduled release] of each
            # hold; first = dispatched, full-path booking not yet made
            v.tm, v.pt, v.bt, v.bk, v.bres, v.rel, v.hw, v.first = None, None, None, None, set(), {}, {}, False

    # ---------------------------------------------------------------- task times
    def draw(self, task):
        """(loading, unloading, empty speed factor, loaded speed factor) of the task. Without p["stoch"] the nominal values
        (load_s, unload_s, 1, 1)."""
        p, s = self.p, self.p.get("stoch")
        if not s:
            return p["load_s"], p["unload_s"], 1.0, 1.0
        k = task["id"]
        if k not in self._tm:
            rng = random.Random(f"stoch|{self.seed}|{k}")
            u = [min(max(rng.random(), 1e-12), 1 - 1e-12) for _ in range(4)]
            self._tm[k] = (hand_time(p["load_s"], s, u[0]), hand_time(p["unload_s"], s, u[1]), speed_factor(s, u[2]), speed_factor(s, u[3]))
        return self._tm[k]

    def book_times(self):
        """(loading, unloading, speed factor) used for booking under reserve_rb: nominal; with p["stoch"]["book_q"] = q the
        q-quantile of handling and the (1 - q)-quantile of the speed factor."""
        p, s = self.p, self.p.get("stoch")
        q = (s or {}).get("book_q")
        if q is None:
            return p["load_s"], p["unload_s"], 1.0
        return hand_time(p["load_s"], s, q), hand_time(p["unload_s"], s, q), speed_factor(s, 1 - q)

    def task_p(self, tm):
        """The vehicle's parameter table: handling times of the task (draw); p itself when they equal the nominal values
        (no stoch)."""
        p = self.p
        return p if (tm[0], tm[1]) == (p["load_s"], p["unload_s"]) else dict(p, load_s=tm[0], unload_s=tm[1])

    def legs(self, v, task, fE=None, fL=None):
        """As legs of occupancy, with the speeds multiplied by the factors of the task (fE, fL default to draw; the trips
        used for booking pass their own). Multiplying by 1.0 does not change a floating-point value."""
        net, p, cap, size = self.net, self.p, self.cap, self.size
        if fE is None:
            fE, fL = self.draw(task)[2:]
        vE, vL = p["v_empty"] * fE, p["v_loaded"] * fL
        so, sd = net.stops[task["o"]], net.stops[task["d"]]
        d_so = net.demand(so["res"], task["B"], True, p, cap)
        d_sd = net.demand(sd["res"], task["B"], True, p, cap)
        E = net.leg(v.loc, so["node"], vE, p["plat_len"], p["plat_w"], False, p, cap, size, so["res"], d_so, True)
        L = net.leg(so["node"], sd["node"], vL, task["Lv"], task["B"], True, p, cap, size, sd["res"], d_sd)
        if task.get("flow") == "P6" and sd["ent"] in self.cranes:
            L = L[:2] + (L[2] + [(self.cranes[sd["ent"]], 1.0, L[3], None, 0.0)], L[3])
        tail_l = (task["Lv"] + p["long_clear"]) / vL
        clr = (p["plat_len"] + p["long_clear"]) / vE
        return so, sd, d_so, d_sd, E, L, tail_l, clr

    def plan(self, v, task, t0):
        """Free flow (under random times the only rule that comes here): handling times of the task, speeds through legs."""
        base, self.p = self.p, self.task_p(self.draw(task))
        try:
            return super().plan(v, task, t0)
        finally:
            self.p = base

    def start_segment(self, v, task, t):
        v.tm = self.draw(task)
        v.pt = self.task_p(v.tm)
        # trips and durations for booking (nominal or quantile): same resources and decision points, other times
        if self.rb:
            lb, ub, fb = self.book_times()
            _, _, _, _, Eb, Lb, tb, cb = self.legs(v, task, fb, fb)
            v.bk = dict(E=Eb, L=Lb, tail=tb, clr=cb, load=lb, unload=ub)
            v.first = True
        super().start_segment(v, task, t)

    def grant(self, v, t):
        base, self.p = self.p, v.pt
        try:
            super().grant(v, t)
        finally:
            self.p = base

    # ---------------------------------------------------------------- late rebooking (reserve_rb)
    def request(self, v, t):
        if self.rb:
            if v.first:             # full-path booking at dispatch
                v.first = False
                self.rebook(v, t, "E", 0, first=True)
            if not self.rb_gate(v, t):
                return
        super().request(v, t)

    def wake(self, t):
        if not self.rb:
            return super().wake(t)
        for k in list(self.waiting):
            v = self.vs[k]
            if self.fits(v, v.req + v.restore) and not self.locked(v) and (not self.safe or self.safe_to_grant(v)):
                self.waiting.remove(k)
                if not self.rb_gate(v, t):      # fits, but the rebooked time lies ahead: leave the queue, request again then
                    continue
                self.grant(v, t)

    def acquire(self, v, res, d, t):
        if self.rb:                 # [width, scheduled release] of each hold (narrow)
            v.hw.setdefault(res, []).append([d, None])
        super().acquire(v, res, d, t)

    def release(self, v, res, t, all_=False):
        h = v.hold.get(res)
        if self.rb and h is not None:
            if h[0] > 1 and not all_:       # one of several holds released: narrow (occupancy's release only counts down)
                h[0] -= 1
                self.narrow(v, res, t)
                return
            v.hw.pop(res, None)
        super().release(v, res, t, all_)

    def free_later(self, v, t, res):
        if self.rb:
            v.rel[res] = max(t, v.rel.get(res, t))
            next(x for x in v.hw[res] if x[1] is None)[1] = t        # the hold acquired first is scheduled for release first
        super().free_later(v, t, res)

    def on_recover(self, t, k):
        # recovery voids the scheduled releases (check_deadlock clears pending); clear them here too
        if self.rb:
            self.vs[k].rel = {}
        super().on_recover(t, k)

    def narrow(self, v, res, t):
        """When one of the vehicle's several holds on a resource is released (the one scheduled at t), the width held drops to
        the largest of the others, as the full-path booking takes the piecewise maximum width in the table (commit): after
        unloading at a roadside stop, an empty run leaving by the same road takes the empty width once the clearance is
        passed. The occupancy before narrowing is recorded as usual."""
        hw = v.hw[res]
        hw.remove(min((x for x in hw if x[1] is not None), key=lambda x: abs(x[1] - t)))
        h, w = v.hold[res], max(x[0] for x in hw)
        if w < h[1] - EPS:
            s = v.since[res]
            self.occ[res] += max(0.0, min(t, self.H) - max(s, self.t0m)) * min(1.0, h[1] / self.cap[res])
            if min(t, self.H) > max(s, self.t0m):
                self.ivs[res].append((max(s, self.t0m), min(t, self.H)))
            if self.log is not None:
                self.log.append((res, s, t, h[1], v.id))
            self.used[res] -= h[1] - w
            h[1], v.since[res] = w, t

    def rb_gate(self, v, t):
        """At a decision point (or when the request fits) the booked time is checked: later than booked, the rest of the chain
        is rebooked; earlier than the (new) booked time, the vehicle waits where it is until then (off-network waiting
        place, origin stop or the current road, all within its booking) and False is returned. The time waited for the
        booked time counts as waiting of the task (as under reserve)."""
        th = v.bt[v.phase][v.step]
        if t > th + EPS:
            if self.in_win(t):
                self.rb_n["late"] += 1
                self.rb_n["late_s"] += t - th
            self.rebook(v, t, v.phase, v.step)
            th = v.bt[v.phase][v.step]
        if t < th - EPS:
            if v.wait_t is None:
                v.waited += th - t
            self.push(th, "step", v.id)
            return False
        return True

    def rebook(self, v, now, phase, k, first=False):
        """Books the rest of the chain from decision point (phase, k) on with the booking durations (v.bk: nominal or
        quantile); first = the full-path booking at dispatch.
        (1) Withdraw the vehicle's bookings in the table that have not ended.
        (2) New times for the resources still to be acquired (at decision points >= k; after a teleport recovery also those
            to be re-acquired, held from the departure). A gap is searched only at the two places where a vehicle can wait:
            k = 0, i.e. leaving the off-network waiting place (dispatch; as in plan: empty run and origin stop resource
            booked until loaded departure + tail, shifted as a whole when loading cannot be followed by departure; at
            dispatch also just-in-time departure) and the loaded departure from the origin stop. En route (k > 0) no new gap
            is awaited on the road: the rest of the current run is shifted from now (no conflict check, the table may be
            overloaded) and the vehicle moves on when the next part fits; when late on the empty run, the loaded run still
            searches its gap at the origin (waiting on the origin stop). Waiting for new gaps on the road as well would let
            stopped vehicles block others from passing as booked, those would rebook later still, pushing each other away,
            and the wait-for graph does not see such waits through the table (a probe run: with speeds only 0.1 % slower,
            throughput fell to 1 block a day).
        (3) Resources held now are booked until their expected release: scheduled ones at the scheduled time, the others by
            the new booking (road / junction until the decision point that releases it + clearance, origin until loaded
            departure + tail, destination until unloading done + clearance to leave); no conflict check, as the vehicle is
            there; the table may be overloaded by it, other vehicles cannot book into the overloaded period, and vehicles
            already booked there wait for it by capacity on arrival."""
        b, vid = v.bk, v.id
        TE, stE, uE, arrE = b["E"]
        TL, stL, uL, arrL = b["L"]
        load, unload, tail, clr = b["load"], b["unload"], b["tail"], b["clr"]
        so, sd = v.so["res"], v.sd["res"]
        for res in v.bres:                                                                    # (1)
            self.table[res] = [x for x in self.table[res] if x[3] != vid or x[1] <= now]
        st, uses = (stL, uL) if phase == "L" else (stE, uE)
        sk, Tk = st[k], (TL if phase == "L" else TE) - st[k]
        # current run: fwd = road / junction windows ahead (relative to leaving k); ext = held resources whose
        # release depends
        # on a later decision point {resource: (releasing decision point, clearance)}; end = the stop resource at the end of
        # this run (origin for the empty run, destination for the loaded run): (start of its window ahead, width, held now)
        fwd, ext, end = [], [], None
        for res, d, a, r, dl in uses:
            if d <= EPS or (a < k and r is not None and r < k):
                continue
            held = a < k and res in v.hold
            if r is None:
                end = (st[max(a, k)] - sk, d, held)
            elif held:
                ext.append((res, st[r] + dl - sk, d))
            else:
                fwd.append((res, st[max(a, k)] - sk, st[r] + dl - sk, d))
        if phase == "E":                                                                       # (2)
            WL = [(res, stL[a], stL[r] + dl, d) for res, d, a, r, dl in uL if r is not None and d > EPS]
            WL.append((sd, next(stL[a] for _, _, a, r, _ in uL if r is None), TL + unload + clr, v.d_sd))
            o_s, o_d, o_held = end
            tmin = now
            # as in plan: origin until loaded departure + tail; shifted as a whole if departure cannot follow
            # loading
            while True:
                t1 = now if k else self.earliest(fwd + ([] if o_held else [(so, o_s, Tk + load + tail, o_d)]), tmin, vid, False, [])
                t2 = self.earliest(WL, t1 + Tk + load, vid, False, [])
                o2 = None if o_held or k else self.conflict(so, t1 + o_s, t2 + tail, o_d, vid)
                if o2 is None:
                    break
                tmin = o2 - o_s
            if first and self.p["reserve_jit"]:          # just-in-time departure (as in plan)
                tj = t2 - load - Tk
                if tj > t1 + EPS and all(self.conflict(r_, tj + s, tj + e, d, vid) is None for r_, s, e, d in fwd):
                    t1 = tj
            x = t1
            W = [(res, t1 + s, t1 + e, d) for res, s, e, d in fwd] + [(res, t2 + s, t2 + e, d) for res, s, e, d in WL]
            if o_held:
                W.append((so, now, t2 + tail, o_d))
            else:
                W.append((so, t1 + o_s, t2 + tail, o_d))
            v.bt = dict(E=[t1 + y - sk for y in stE], L=[t2 + y for y in stL])
        else:
            d_s, d_d, d_held = end
            dest = (sd, d_s, Tk + unload + clr, d_d)
            # origin stop (acquired on the empty run): until loaded departure + tail
            if k == 0 and so in v.hold:
                ext.append((so, tail, v.d_so))
            elif k == 0 and v.restore:                   # requested again at the origin after a teleport recovery
                fwd.append((so, 0.0, tail, v.d_so))
            x = now if k else self.earliest(fwd + ([] if d_held else [dest]), now, vid, False, [])
            W = [(res, x + s, x + e, d) for res, s, e, d in fwd]
            W.append((sd, now if d_held else x + d_s, x + dest[2], d_d))
            v.bt = dict(v.bt, L=[x + y - sk for y in stL])
        W += [(res, now, x + e, d) for res, e, d in ext]                                       # (3)
        # scheduled releases (junction just passed, clearance of the last destination): until then
        for res, h in v.hold.items():
            if v.pending[res]:
                W.append((res, now, v.rel[res], h[1]))
        self.commit(vid, now, W)
        v.bres = {w[0] for w in W}
        if not first and self.in_win(now):
            self.rb_n["rebook"] += 1

    def metrics(self):
        m = super().metrics()
        if self.rb:                      # late arrivals and rebookings per day in the statistics window, mean late seconds
            n, n_days = self.rb_n, (self.H - self.t0m) / self.p["day_s"]
            m["rebook"] = dict(late_per_day=n["late"] / n_days, rebook_per_day=n["rebook"] / n_days,
                               late_mean_s=n["late_s"] / n["late"] if n["late"] else 0.0)
        return m

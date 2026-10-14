"""Discrete-event simulation: dispatch = nearest idle vehicle, first come first served. Traffic rules:

  free     free flow: vehicles do not interact; travel time = distance / speed.
  reserve  full-path reservation before departure: at dispatch, the earliest departure time is searched in the reservation
           tables of all resources with the occupancy windows of a "no-wait pass"; empty run + loading at the origin +
           loaded run + unloading at the destination are booked at once. Vehicles wait only at stops (off the road or in
           the bay), never en route, so no deadlock can occur.
           reserve_jit (part of the main model): once the loaded run is booked, the empty run is postponed so that the
           vehicle "finishes loading at the origin just in time to depart", if that is still feasible; this holds the
           origin stop resource for less time.
  segment  stepwise request: on reaching a junction a vehicle requests "junction + next road (the last road together
           with the destination stop resource)" at once; if refused, it waits while holding its current road.
           Each time a vehicle starts waiting a deadlock check (graph reduction) is run; on a deadlock one vehicle on the
           wait cycle is recovered manually (it gives up all its resources and requests again from its original position
           after deadlock_recovery_s), and the count, time and size are recorded. During recovery the vehicle holds no
           resources, i.e. it leaves the network, so the throughput of this rule must be read as a function of the
           recovery time (the "reference rule (teleport recovery)").
           deadlock_recovery_s = None ("no recovery" switch): a deadlock is only recorded; the stuck vehicles keep
           holding the road.
           With sectioning (net.sec non-empty) a direction lock applies: when a vehicle is about to enter a sectioned map
           road (holding none of its sections yet) and the road carries oncoming vehicles whose widths together with its
           own do not fit, it waits until they have left the whole road; vehicles in the same direction may follow.
  segment_safe  stepwise request + safe release: still requests "junction + next road" step by step, but a request is
           granted only if the system is still in a safe state afterwards, i.e. there is an order in which every vehicle on
           the network can finish its remaining task (empty run + loading + loaded run + unloading, until it leaves the
           network as an idle vehicle), each using only the currently free resources plus those released by the vehicles
           ahead of it in the order (Banker-type deadlock avoidance, corresponding to release by a yard dispatcher or
           signaller). Cranes and scheduled releases (tail leaving, clearance given up) are released on time and are
           treated as "available sooner or later"; a vehicle waiting for a crane still holds its stop road. Vehicles do
           not leave the network; the deadlock check is kept as an assertion, and a detected deadlock is a program error.
           Implementation in safe_to_grant: first check whether this vehicle alone can finish, then check the last safe
           order, and only then run a full reduction; a request judged unsafe is cached by "whether the stuck set has
           changed" and not recomputed while it has not (exact, not an approximation).

An idle vehicle stays in the yard next to its last unloading stop and does not occupy the road; after unloading at the
destination it holds the stop for a further (platform length + longitudinal clearance) / empty speed to vacate the bay.

Crane-paced erection (p["crane_n"]): the cranes of each dock are a resource ('c', dock) with capacity = number of cranes
(Net.capacities); from the start of a P6 unloading (= hooking, hoisting) one crane is held for crane_cycle_s (hooking and
hoisting + positioning and unhooking). Reservation books the crane interval together with the whole path: if the crane
forces a later departure, the departure time is shifted; with just-in-time departure the vehicle stays at its previous
unloading point (if the empty run cannot be postponed, it waits at the origin stop bay), and never stops en route.
Stepwise request asks for the crane on reaching the dock; if refused, the vehicle waits while holding its stop road
(waiting for a crane is not a deadlock; the crane is released on time). Free flow does not check cranes.
P6 inside the crane span (task["lift"]): the gantry crane lifts directly from the assembly area into the dock; no vehicle
is dispatched and no road is occupied, only a crane of that dock for crane_cycle_s. Reservation books the earliest free
crane interval (the same reservation table as the crane intervals of vehicles, first booked first served); stepwise
request lifts if a crane is free, otherwise it queues first come first served with the vehicles waiting for the crane at
the dock side. Free flow does not check. Lifts are recorded in lifts and not counted in the transporter throughput.

Local readings: dock-mouth zone = the two rings of resources of network.Net.near(dock). Within the statistics window the
following are recorded per resource: occupancy (width share × seconds; for free flow the no-wait windows, i.e. the
"needed" occupancy) and busy-time fraction (fraction of time with at least one vehicle on it); waiting attributed to the
resource that blocks it (reservation: the resources that push the departure time later; stepwise: the resource refused
when the wait starts; refused by safe release: the road the vehicle is about to enter); deadlocks and near-deadlocks
stuck in the dock-mouth zone.
Further readings: crane starvation rate (starvation); diagnostics of the vehicles recovered under the reference rule
(teleport recovery) (whether, when recovered, the vehicle held a dock stop resource, and whether it was carrying a P6 to
the dock); number of refusals and waiting under safe release; cycle-time breakdown (cycle).
"""
import collections
import heapq

from .bound import pieces, res_name
from .network import EPS
from .params import DOCK_CLASS
from .tasks import TaskStream, crane_p6, crane_tasks, demand_tasks, vehicle_starts

MODES = ("free", "reserve", "segment", "segment_safe")


def union_len(iv):
    """Total length of a union of time intervals."""
    tot, end = 0.0, -float("inf")
    for s, e in sorted(iv):
        if e > end:
            tot += e - max(s, end)
            end = e
    return tot


def suffix_needs(uses, arr):
    """Resources a leg still has to request from decision point k on, {resource: max width}, k = 0 … arr + 1 (for safe
    release; cranes are "available sooner or later" and not counted). Adjacent decision points without new requests share
    the same dict (read-only)."""
    by = collections.defaultdict(list)
    for res, d, a, _, _ in uses:
        if res[0] != "c" and d > EPS:
            by[a].append((res, d))
    out, cur = [{}] * (arr + 2), {}
    for k in range(arr, -1, -1):
        if by[k]:
            cur = dict(cur)
            for res, d in by[k]:
                if d > cur.get(res, 0.0):
                    cur[res] = d
        out[k] = cur
    return out


def merge_max(a, b):
    """Merge two demand tables, taking the max width per resource."""
    out = dict(b)
    for r, d in a.items():
        if d > out.get(r, 0.0):
            out[r] = d
    return out


class Vehicle:
    def __init__(self, k, loc):
        self.id, self.loc = k, loc
        self.hold = {}            # resource → [count, width]
        self.since = {}           # resource → time acquired
        self.gen = 0              # invalidates pending release events after a manual recovery
        self.pending = collections.Counter()   # resource → number of scheduled release events
        self.req, self.restore, self.wait_t, self.block = [], [], None, None
        self.victim = False       # recovered manually during the current wait
        # safe release (segment_safe): sticky = occupancy not released on time {resource: width} (count exceeds scheduled
        # releases); need = resources still to be requested in the remaining task {resource: max width}; needs[phase][k] =
        # need from decision point k on; unsafe = (phase, decision point, stuck set S) when last judged unsafe; dirty = the
        # occupancy, scheduled releases or remaining demand of some vehicle in S has changed (Sim.touch), recompute
        self.sticky, self.need, self.needs, self.unsafe, self.dirty = {}, {}, None, None, False
        self.unsafe_wait = False  # the current wait is due to a refusal by safe release (capacity suffices)


class Sim:
    def __init__(self, net, p, mode, K, seed, level_n, record=False, tasks=None):
        assert mode in MODES
        self.net, self.p, self.mode = net, p, mode
        self.seg = mode in ("segment", "segment_safe")           # stepwise request family
        self.safe = mode == "segment_safe"
        self.res = mode == "reserve"                             # reservation table
        self.lock = mode == "segment" and bool(net.sec)          # direction lock: reference rule with sectioning
        self.sused = collections.defaultdict(float)              # safe release: per resource, sum of occupancy not released on time
        self.order = []                                          # safe release: last checked safe order (vehicle ids)
        self.watchers = collections.defaultdict(set)             # safe release: vehicle → waiting vehicles whose cached stuck set contains it
        self.onroad = collections.defaultdict(dict)              # direction lock: sectioned map road → {vehicle: (direction, width)}
        self.frozen = set()                                      # vehicles stuck for good when there is no recovery
        self.safe_n = collections.Counter()                      # safe release: number of checks (diagnostics), by the step that decided
        self.cap, self.size = net.capacities(p)
        self.H, self.t0m = p["days"] * p["day_s"], p["warmup_days"] * p["day_s"]
        self.vs = [Vehicle(k, loc) for k, loc in enumerate(vehicle_starts(net, seed, K))]
        self.idle = list(range(K))
        self.queue = collections.deque()
        self.stream = TaskStream(net, seed, p)
        self.saturated = level_n is None and tasks is None
        self.ev, self.seq = [], 0
        if tasks is None and level_n is None:       # saturated level: with crane-paced erection P6 still arrives at the crane pace, the other flows never run dry
            tasks = crane_tasks(net, seed, p, crane_p6(net, p)) if p.get("crane_n") else []
        for t in tasks if tasks is not None else demand_tasks(net, seed, level_n, p):
            self.push(t["release"], "arrive", t)
        self.table = collections.defaultdict(list)    # reserve: resource → [(start, end, width, vehicle)]
        self.used = collections.defaultdict(float)    # segment: resource → width occupied
        self.waiting = []
        self.done, self.deadlocks, self.near = [], [], []          # near: (wait start time, blocking resource)
        self.log = [] if record else None              # (resource, start, end, width, vehicle), for tests
        self.zone = net.near(DOCK_CLASS)
        self.watch = set().union(*self.zone) | {s["res"] for s in net.stops.values() if s["mode"] in ("node", "door")} | \
            {("n", n) for n in net.doors}
        # crane-paced erection: the gantry cranes of each dock are one resource ('c', dock) with capacity = number of cranes;
        # a P6 holds one for crane_cycle_s from the start of unloading. Not attached to the vehicle; released automatically on time
        self.cranes = {z: ("c", z) for z in net.by_cls[DOCK_CLASS]} if p.get("crane_n") else {}    # capacity: see Net.capacities
        self.watch |= set(self.cranes.values())
        self.lifts, self.lift_q, self.lift_rel = [], collections.defaultdict(collections.deque), 0   # lifts inside the crane span
        self.occ = collections.defaultdict(float)      # resource → occupancy in the statistics window (width share × seconds)
        self.ivs = collections.defaultdict(list)       # resource → occupied intervals in the statistics window (for the busy-time fraction)
        self.wait_by = collections.defaultdict(float)  # resource → waiting attributed to it in the statistics window (s)

    def push(self, t, kind, x):
        self.seq += 1
        heapq.heappush(self.ev, (t, self.seq, kind, x))

    def run(self):
        self.dispatch(0.0)
        while self.ev and self.ev[0][0] <= self.H:
            t, _, kind, x = heapq.heappop(self.ev)
            getattr(self, "on_" + kind)(t, x)
        return self

    # ---------------------------------------------------------------- dispatch
    def on_arrive(self, t, task):
        if task.get("lift"):
            return self.lift(t, task)
        self.queue.append(task)
        self.dispatch(t)

    # ---------------------------------------------------------------- P6 inside the crane span
    def lift(self, t, task):
        res = self.cranes[self.net.stops[task["d"]]["ent"]]
        self.lift_rel += self.t0m <= t < self.H
        if self.res:
            c = self.p["crane_cycle_s"]
            s = self.earliest([(res, 0.0, c, 1.0)], t, -1, False, [])
            self.commit(-1 - len(self.lifts), t, [(res, s, s + c, 1.0)])     # negative id: not a vehicle (each lift counts separately in the time-by-time test check)
            self.lift_start(task, s, res)
        elif self.seg and (self.lift_q[res] or self.used[res] + 1.0 > self.cap[res] + EPS):
            self.lift_q[res].append((t, task))
        else:
            self.lift_start(task, t, res)

    def lift_start(self, task, s, res):
        c = self.p["crane_cycle_s"]
        self.account([(res, s, s + c, 1.0)])
        if self.seg:
            self.used[res] += 1.0
            self.push(s + c, "crane_free", (res, 1.0))
        self.lifts.append(dict(task, start=s, done=s + self.p["crane_hook_s"]))

    def serve_lifts(self, t, res):
        """Stepwise request: while a crane is free, lift the queued in-span P6 in turn; if a vehicle that arrived earlier is
        waiting for this crane at the dock side, give way to it first (first come first served; the vehicle is handled by wake)."""
        q = self.lift_q[res]
        while q and self.used[res] + 1.0 <= self.cap[res] + EPS:
            if any(self.vs[k].block == res and self.vs[k].wait_t <= q[0][0] for k in self.waiting):
                return
            self.lift_start(q.popleft()[1], t, res)

    def dispatch(self, t):
        while self.idle:
            if not self.queue:
                if not self.saturated:
                    return
                self.queue.append(self.stream.next(t))
            task = self.queue.popleft()
            o = self.net.stops[task["o"]]["node"]
            k = min(self.idle, key=lambda k: (self.net.dist(self.vs[k].loc, o), k))
            self.idle.remove(k)
            task["dispatch"], task["start"] = t, self.vs[k].loc
            (self.start_segment if self.seg else self.plan)(self.vs[k], task, t)

    def legs(self, v, task):
        net, p, cap, size = self.net, self.p, self.cap, self.size
        so, sd = net.stops[task["o"]], net.stops[task["d"]]
        d_so = net.demand(so["res"], task["B"], True, p, cap)
        d_sd = net.demand(sd["res"], task["B"], True, p, cap)
        E = net.leg(v.loc, so["node"], p["v_empty"], p["plat_len"], p["plat_w"], False, p, cap, size, so["res"], d_so, True)
        L = net.leg(so["node"], sd["node"], p["v_loaded"], task["Lv"], task["B"], True, p, cap, size, sd["res"], d_sd)
        if task.get("flow") == "P6" and sd["ent"] in self.cranes:     # crane: requested on reaching the dock (start of unloading), released by the task logic
            L = L[:2] + (L[2] + [(self.cranes[sd["ent"]], 1.0, L[3], None, 0.0)], L[3])
        tail_l = (task["Lv"] + p["long_clear"]) / p["v_loaded"]
        clr = (p["plat_len"] + p["long_clear"]) / p["v_empty"]
        return so, sd, d_so, d_sd, E, L, tail_l, clr

    def finish(self, v, task, t, wait):
        # start: vehicle position at dispatch; cs: start of crane lift (P6 to a dock with cranes); blk: intervals in which the
        # vehicle waits for road resources (component of the crane starvation rate)
        self.done.append(dict(id=task["id"], flow=task.get("flow"), o=task["o"], d=task["d"], B=task["B"], Lv=task["Lv"],
                              release=task["release"], dispatch=task["dispatch"], done=t, wait=wait, plan=task.get("plan"),
                              start=task.get("start"), cs=task.get("crane_start"), blk=task.get("blk"), blk_road=task.get("blk_road")))

    def in_win(self, t):
        return self.t0m <= t <= self.H

    def account(self, W):
        """Record occupancy: W = [(resource, start, end, width)], clipped to the statistics window; windows on the same
        resource are merged with the piecewise maximum width."""
        by = collections.defaultdict(list)
        for res, s, e, d in W:
            s, e = max(s, self.t0m), min(e, self.H)
            if e > s and d > EPS:
                by[res].append((s, e, d))
        for res, iv in by.items():
            self.occ[res] += sum(ln * min(1.0, d / self.cap[res]) for ln, d in pieces(iv))
            self.ivs[res] += [(s, e) for s, e, _ in iv]

    # ---------------------------------------------------------------- free flow and full-path reservation
    def windows(self, v, task):
        """No-wait occupancy windows of the reservation rules (relative to the departure of each leg):
        (so, sd, d_so, d_sd, TE, TL, WE, WL, ao, ad, tail_l, clr).
        WE, WL exclude the stop resources at both ends; WL includes the crane (held for crane_cycle_s from the start of unloading)."""
        so, sd, d_so, d_sd, (TE, stE, uE, _), (TL, stL, uL, _), tail_l, clr = self.legs(v, task)
        ao = next(stE[a] for _, _, a, r, _ in uE if r is None)     # time the origin stop resource is requested (relative to the empty-run departure)
        ad = next(stL[a] for _, _, a, r, _ in uL if r is None)
        WE = [(res, stE[a], stE[r] + dl, d) for res, d, a, r, dl in uE if r is not None]
        WL = [(res, stL[a], stL[r] + dl, d) for res, d, a, r, dl in uL if r is not None]
        WL += [(res, TL, TL + self.p["crane_cycle_s"], d) for res, d, a, r, dl in uL if res[0] == "c"]   # crane: from the start of unloading
        return so, sd, d_so, d_sd, TE, TL, WE, WL, ao, ad, tail_l, clr

    @staticmethod
    def split_blk(blk, jumps):
        """Under reservation, split the intervals blk in which "the vehicle waits at a stop to depart" by the resources that
        pushed the departure time later: the shifts in jumps are laid onto blk in order (the shifts sum to the total length
        of blk); return the parts caused by roads and stop resources (not by cranes)."""
        out, k, pos = [], 0, None
        for res, x in jumps:
            while x > EPS and k < len(blk):
                s, e = blk[k]
                cur = s if pos is None or pos < s else pos
                take = min(x, e - cur)
                if take > EPS and res[0] != "c":
                    if out and abs(out[-1][1] - cur) <= EPS:
                        out[-1] = (out[-1][0], cur + take)
                    else:
                        out.append((cur, cur + take))
                pos, x = cur + take, x - take
                if pos >= e - EPS:
                    k, pos = k + 1, None
        return out

    def book_l(self, sd, d_sd, TL, WL, ad, clr, tmin, vid, free, jumps):
        """Reservation of the loaded run: returns (t2 loaded departure, tc start of unloading, absolute windows of the loaded
        run and destination, wait in the waiting place (arrival, tc) or None).
        Without a dock-side waiting place tc = t2 + TL and the windows are identical, item by item, to earlier versions.
        With a dock-side waiting place at the destination (net.wait_at): the road windows are booked from t2; after
        reaching the dock (a = t2 + TL) the vehicle waits in the waiting place until the stop resource and the crane are both
        free (tc ≥ a) and then unloads; the waiting place is booked for [a, tc); if it is full, the departure is postponed."""
        unload = self.p["unload_s"]
        wres = None if free else self.net.wait_at.get(sd["node"])
        if wres is None:
            t2 = self.earliest(WL + [(sd["res"], ad, TL + unload + clr, d_sd)], tmin, vid, free, jumps)
            return t2, t2 + TL, [(r, t2 + s, t2 + e, d) for r, s, e, d in WL] + [(sd["res"], t2 + ad, t2 + TL + unload + clr, d_sd)], None
        road = [w for w in WL if w[0][0] != "c"]
        crane = [(r, 0.0, e - s, d) for r, s, e, d in WL if r[0] == "c"]          # crane windows start at the start of unloading
        while True:
            t2 = self.earliest(road, tmin, vid, False, jumps)
            a = t2 + TL
            jc = []
            tc = self.earliest([(sd["res"], 0.0, unload + clr, d_sd)] + crane, a, vid, False, jc)
            o2 = self.conflict(wres, a, tc, 1.0, vid) if tc > a + EPS else None
            if o2 is None:
                jumps += jc
                W = [(r, t2 + s, t2 + e, d) for r, s, e, d in road] + [(r, tc + s, tc + e, d) for r, s, e, d in crane] \
                    + [(sd["res"], tc, tc + unload + clr, d_sd)]
                if tc > a + EPS:
                    W.append((wres, a, tc, 1.0))
                return t2, tc, W, ((a, tc) if tc > a + EPS else None)
            jumps.append((wres, o2 - a))
            tmin = t2 + (o2 - a)

    def plan(self, v, task, t0):
        """Full-path reservation (reserve): empty run, loading, loaded run and unloading are booked at once; if the vehicle
        cannot depart right after loading, it waits while holding the origin stop resource. Free flow uses the same windows
        without booking them.
        With a dock-side waiting place at the destination, the vehicle may wait there for unloading (book_l)."""
        so, sd, d_so, d_sd, TE, TL, WE, WL, ao, ad, tail_l, clr = self.windows(v, task)
        load, unload = self.p["load_s"], self.p["unload_s"]
        free = self.mode == "free"
        tmin, jumps = t0, []                # jumps: (resource, seconds it shifts the departure later), for wait attribution
        while True:
            jL = []
            t1 = self.earliest(WE + [(so["res"], ao, TE + load + tail_l, d_so)], tmin, v.id, free, jumps)
            a = t1 + TE
            t2, tc, WLa, bay = self.book_l(sd, d_sd, TL, WL, ad, clr, a + load, v.id, free, jL)
            o2 = None if free else self.conflict(so["res"], t1 + ao, t2 + tail_l, d_so, v.id)
            if o2 is None:
                break
            jumps.append((so["res"], o2 - ao - t1))
            tmin = o2 - ao                  # waiting after loading holds the origin: shift everything later until the origin is free for that time
        jumps += jL
        if not free and self.p["reserve_jit"]:
            # just-in-time departure: the origin window [tj + ao, t2 + tail] is a subset of the feasible window, only the empty run needs checking
            tj = t2 - load - TE
            if tj > t1 + EPS and all(self.conflict(r, tj + s, tj + e, d, v.id) is None for r, s, e, d in WE):
                t1, a = tj, tj + TE
        W = [(r, t1 + s, t1 + e, d) for r, s, e, d in WE] + [(so["res"], t1 + ao, t2 + tail_l, d_so)] + WLa
        if not free:
            self.commit(v.id, t0, W)
        self.account(W)
        if self.in_win(t0):
            for res, x in jumps:
                self.wait_by[res] += x
        task["plan"] = dict(start=v.loc, t1=t1, a=a, t2=t2)     # times of empty-run departure, arrival at origin, loaded departure (tests check no stops en route)
        if bay:
            task["plan"]["tc"] = tc                              # waits in the dock-side waiting place until unloading starts at tc
        if any(r[0] == "c" for r, *_ in WL):                      # start of crane lift = start of unloading (crane starvation rate)
            task["crane_start"] = tc
        if task.get("flow") == "P6" and not free:                # vehicle waits at a stop for its reserved departure time ("blocked by the network" under reservation)
            task["blk"] = [(s, e) for s, e in ((t0, t1), (a + load, t2)) + ((bay,) if bay else ()) if e > s + EPS]
            task["blk_road"] = self.split_blk(task["blk"], jumps)     # the part caused by roads
        v.loc = sd["node"]
        self.push(tc + unload, "done", (v.id, task, max(0.0, (t1 - t0) + (t2 - a - load) + (tc - t2 - TL if bay else 0.0))))

    def on_done(self, t, x):
        k, task, wait = x
        self.finish(self.vs[k], task, t, wait)
        self.idle.append(k)
        self.dispatch(t)

    def earliest(self, W, t, vid, free, jumps):
        """Earliest departure time ≥ t at which all of W = [(resource, relative start, relative end, width)] fit; every shift
        is recorded in jumps.
        If a window intersects an overload interval [o1, o2), the departure is shifted at least until the window start passes
        o2; feasibility can only switch from no to yes at these points, so no feasible solution is skipped."""
        if free:
            return t
        while True:
            for res, s, e, d in W:
                o2 = self.conflict(res, t + s, t + e, d, vid)
                if o2 is not None:
                    jumps.append((res, o2 - s - t))
                    t = o2 - s
                    break
            else:
                return t

    def conflict(self, res, s, e, d, vid):
        """If adding width d in [s, e) overloads the resource, return the end of the first intersecting overload interval;
        otherwise None. The vehicle's own reservations do not count, except for cranes: the vehicle leaves after unloading
        while the crane stays busy for 180 min, so the same vehicle's earlier crane interval also holds the crane."""
        if d <= EPS or e <= s:
            return None
        C = self.cap[res]
        own = None if res[0] == "c" else vid
        lst = [(a, b, w) for a, b, w, k in self.table[res] if b > s + EPS and k != own]   # EPS: avoids getting stuck at interval ends through rounding errors
        ov = [w for a, b, w in lst if a < e - EPS]
        if not ov or sum(ov) + d <= C + EPS:
            return None
        over = False
        for tt in sorted({s} | {a for a, _, _ in lst if a > s} | {b for _, b, _ in lst}):
            load = sum(w for a, b, w in lst if a <= tt < b)
            if load + d > C + EPS:
                if not over and tt >= e - EPS:
                    return None
                over = True
            elif over:
                return tt
        return None

    def commit(self, vid, now, W):
        for res, s, e, d in W:
            if e <= s or d <= EPS:
                continue
            lst = [x for x in self.table[res] if x[1] > now]
            mine = [] if res[0] == "c" else [x for x in lst if x[3] == vid and x[0] <= e and x[1] >= s]   # crane intervals are independent of each other
            if not mine:
                lst.append((s, e, d, vid))
            else:                                           # adjacent or overlapping windows of this vehicle: piecewise max width, narrow pieces are not recorded as wide
                lst = [x for x in lst if not (x[3] == vid and x[0] <= e and x[1] >= s)]
                pieces = mine + [(s, e, d, vid)]
                ts = sorted({x[0] for x in pieces} | {x[1] for x in pieces})
                for a, b in zip(ts, ts[1:]):
                    w = max((x[2] for x in pieces if x[0] <= a and x[1] >= b), default=0.0)
                    if w <= EPS:
                        continue
                    if lst and lst[-1][3] == vid and lst[-1][1] == a and lst[-1][2] == w:
                        lst[-1] = (lst[-1][0], b, w, vid)
                    else:
                        lst.append((a, b, w, vid))
            self.table[res] = lst
            if self.log is not None:
                self.log.append((res, s, e, d, vid))

    # ---------------------------------------------------------------- stepwise request
    def start_segment(self, v, task, t):
        v.task = task
        v.so, v.sd, v.d_so, v.d_sd, v.E, v.L, v.tail_l, v.clr = self.legs(v, task)
        if self.safe:               # resources still to be requested from each decision point on (for the empty run including the whole loaded run)
            sl = suffix_needs(*v.L[2:4])
            v.needs = {"E": [merge_max(x, sl[0]) for x in suffix_needs(*v.E[2:4])], "L": sl}
            v.need = v.needs["E"][0]
            self.uncache(v)
        if self.lock:
            v.runs = {"E": self.net.runs(v.loc, v.so["node"]), "L": self.net.runs(v.so["node"], v.sd["node"])}
        v.phase, v.step, v.waited = "E", 0, 0.0
        self.request(v, t)

    def leg_of(self, v):
        return v.E if v.phase == "E" else v.L

    def request(self, v, t):
        uses = self.leg_of(v)[2]
        v.req = [(res, d) for res, d, a, _, _ in uses if a == v.step]
        fit = self.fits(v, v.req + v.restore) and not self.locked(v)
        if fit and (not self.safe or self.safe_to_grant(v)):
            self.grant(v, t)
        else:
            if v.wait_t is None:            # a new request after recovery is the same wait; attribution stays with the resource that blocked it when the wait started
                v.wait_t = t
                # the resource lacking capacity; if the direction lock blocks, the section it is about to enter; if safe
                # release refuses, the road it is about to enter (the first item that is not a crane)
                v.block = next((res for res, d in v.req + v.restore if not self.fits(v, [(res, d)])), None) \
                    or self.lock_res(v) or next(res for res, d in v.req if res[0] != "c")
                v.unsafe_wait = fit
            self.waiting.append(v.id)
            self.check_deadlock(t)

    def held(self, v, res):
        h = v.hold.get(res)
        return h[1] if h else 0.0

    def fits(self, v, req, used=None):
        used = self.used if used is None else used
        return all(used[res] + max(0.0, d - self.held(v, res)) <= self.cap[res] + EPS for res, d in req)

    def acquire(self, v, res, d, t):
        h = v.hold.get(res)
        if h is None:
            v.hold[res] = [1, d]
            self.used[res] += d
            v.since[res] = t
        else:
            extra = max(0.0, d - h[1])
            h[0] += 1
            h[1] += extra
            self.used[res] += extra

    def release(self, v, res, t, all_=False):
        h = v.hold.get(res)
        if h is None:
            return
        h[0] -= 1
        if h[0] <= 0 or all_:
            self.used[res] -= h[1]
            del v.hold[res]
            s = v.since.pop(res)
            self.occ[res] += max(0.0, min(t, self.H) - max(s, self.t0m)) * min(1.0, h[1] / self.cap[res])
            if min(t, self.H) > max(s, self.t0m):
                self.ivs[res].append((max(s, self.t0m), min(t, self.H)))
            if self.log is not None:
                self.log.append((res, s, t, h[1], v.id))
            if self.lock and res[0] == "r":            # direction lock: holding no section of this sectioned road any more means it has left the road
                o = self.net.road[res[1]]["orig"]
                if o in self.net.sec and not any(("r", x) in v.hold for x in self.net.sec[o]):
                    self.onroad[o].pop(v.id, None)

    def grant(self, v, t):
        if v.wait_t is not None:
            v.waited += t - v.wait_t
            if self.in_win(t):
                self.wait_by[v.block] += t - v.wait_t
                if v.unsafe_wait:
                    self.safe_n["wait_n"] += 1
                    self.safe_n["wait_s"] += t - v.wait_t
            # a recovered vehicle is already counted as a deadlock, not as a near-deadlock; waiting for a crane is not a deadlock
            # (the crane is released on time), not counted either
            if t - v.wait_t >= self.p["near_deadlock_s"] and not v.victim and v.block[0] != "c":
                self.near.append((v.wait_t, v.block))
            if v.block[0] != "c" and v.task.get("flow") == "P6":     # interval in which a P6 vehicle waits for road resources (crane starvation rate)
                v.task.setdefault("blk", []).append((v.wait_t, t))
            v.wait_t, v.victim, v.unsafe_wait = None, False, False
        for res, d in v.restore + v.req:
            if res[0] == "c":          # crane (requested only on reaching the dock): not attached to the vehicle, released automatically after crane_cycle_s
                self.used[res] += d
                self.account([(res, t, t + self.p["crane_cycle_s"], d)])
                self.push(t + self.p["crane_cycle_s"], "crane_free", (res, d))
                v.task["crane_start"] = t
            else:
                self.acquire(v, res, d, t)
                if self.safe:
                    self.stick(v, res)
        v.restore = []
        _, st, uses, arr = self.leg_of(v)
        if self.lock:
            self.mark_road(v)
        for res, d, a, r, dl in uses:
            if r == v.step:
                self.free_later(v, t + dl, res)
        if v.phase == "L" and v.step == 0:
            self.free_later(v, t + v.tail_l, v.so["res"])
        if self.safe:               # one step forward: the remaining demand shrinks; the cache of this request is invalid
            v.need = v.needs[v.phase][v.step + 1]
            self.touch(v)
            self.uncache(v)
        if v.step == arr:
            self.push(t + (self.p["load_s"] if v.phase == "E" else self.p["unload_s"]), "handled", v.id)
        else:
            v.step += 1
            self.push(t + st[v.step] - st[v.step - 1], "step", v.id)

    def free_later(self, v, t, res):
        v.pending[res] += 1
        if self.safe:
            self.stick(v, res)
        self.push(t, "free", (v.id, v.gen, res))

    def on_step(self, t, k):
        self.request(self.vs[k], t)

    def on_free(self, t, x):
        k, gen, res = x
        v = self.vs[k]
        if gen == v.gen:
            v.pending[res] -= 1
            self.release(v, res, t)
            if self.safe:
                self.stick(v, res)
            self.wake(t)

    # ---------------------------------------------------------------- safe release (segment_safe)
    def touch(self, v):
        """The occupancy, scheduled releases or remaining demand of v changed: every waiting vehicle whose cached stuck set
        contains v must recompute."""
        for w in self.watchers.get(v.id, ()):
            self.vs[w].dirty = True

    def uncache(self, v):
        if v.unsafe is not None:
            for u in v.unsafe[1]:
                self.watchers[u].discard(v.id)
            v.unsafe = None

    def stick(self, v, res):
        """The occupancy or scheduled releases of v on res changed (called on acquire, scheduling a release, release):
        recompute the occupancy not released on time (its width if the hold count exceeds the scheduled releases, else 0)
        and update sused; the post-grant occupancy of v (post_grant) changes with it, so always touch."""
        self.touch(v)
        h = v.hold.get(res)
        w = h[1] if h and h[0] > v.pending[res] else 0.0
        old = v.sticky.get(res, 0.0)
        if w != old:
            self.sused[res] += w - old
            if w > 0.0:
                v.sticky[res] = w
            else:
                del v.sticky[res]

    def post_grant(self, v):
        """(Occupancy not released on time, remaining demand) of v after its current request is granted; derived from the
        acquisitions of grant and the releases scheduled at this decision point, without changing the state."""
        _, _, uses, _ = self.leg_of(v)
        rel = collections.Counter(res for res, d, a, r, dl in uses if r == v.step)
        if v.phase == "L" and v.step == 0:
            rel[v.so["res"]] += 1
        add = collections.defaultdict(list)
        for res, d in v.req:
            if res[0] != "c":
                add[res].append(d)
        st = {}
        for res in set(v.hold) | set(add):
            n, w = v.hold[res] if res in v.hold else (0, 0.0)
            n, w = n + len(add.get(res, ())), max([w] + add.get(res, []))
            if n > v.pending[res] + rel[res]:
                st[res] = w
        return st, v.needs[v.phase][v.step + 1]

    def safe_to_grant(self, v):
        """Whether the state is still safe after v's current request is granted (Banker type): there is an order in which the
        vehicles on the network (those with occupancy not released on time) finish their remaining tasks one by one, each
        using only the currently free resources plus those released by the vehicles ahead of it. Vehicle u can finish ⇔ each
        remaining demand d of u satisfies d + (occupancy of the other unfinished vehicles) ≤ capacity. Cranes and scheduled
        releases are released on time and treated as "available sooner or later". Tried in turn:
          (0) the stuck set S found when this request was last judged unsafe (the sink strongly connected component of the
              wait graph containing v): keep only the vehicles in S (all others finished, the most favourable case for S);
              if still no vehicle in S can finish → still unsafe (exact, not an approximation);
          (1) insert into the last checked safe order σ: v goes to the earliest position j such that v can finish while the
              vehicles after j do not move; the vehicles before j can only get tighter on the resources where v now holds
              more than before, and are checked one by one (σ held before the grant, the other terms are unchanged).
              j = 0 means "v can finish while all other vehicles stay put";
          (2) full reduction (Kahn style: vehicles that can finish are removed one by one, releasing their occupancy); the
              remaining vehicles are the stuck set, and the sink strongly connected component containing v is kept for (0)."""
        c = v.unsafe                                                                             # (0)
        if c is not None and c[0] != (v.phase, v.step):
            self.uncache(v)
            c = None
        if c is not None and not v.dirty:     # no vehicle in S has changed: same conclusion
            self.safe_n["cached"] += 1
            return False
        st, nd = self.post_grant(v)
        cap, su, mine = self.cap, self.sused, v.sticky
        if c is not None:
            if self.still_stuck(v, c[1], c[2], st, nd):
                v.dirty = False
                self.safe_n["stuck"] += 1
                return False
            self.uncache(v)
        if self.order_insert(v, st, nd):                                                        # (1)
            return True
        return self.reduce_all(v, st, nd)

    def order_insert(self, v, st, nd):
        """(1) Insert into the last checked safe order σ (see safe_to_grant)."""
        cap, su, mine = self.cap, self.sused, v.sticky
        rest = [u for u in self.order if u != v.id and self.vs[u].sticky]
        # v was originally before the p-th vehicle of rest: the vehicles ahead of it already counted v's old occupancy before
        # the grant, so the extra is only st - mine; the other vehicles saw v as finished before the grant, so the extra is
        # the whole st. If v is on the network but not in σ (should not happen), check against the whole st
        if mine and v.id in self.order:
            k = self.order.index(v.id)
            p = sum(1 for u in self.order[:k] if self.vs[u].sticky)
        else:
            p = 0
        acc, j = collections.defaultdict(float), 0
        for i in range(len(rest) - 1, -1, -1):
            hit = False
            for r, w in self.vs[rest[i]].sticky.items():
                if r in nd:
                    acc[r] += w
                    hit = hit or nd[r] + acc[r] > cap[r] + EPS
            if hit:
                j = i + 1
                break
        rel, ok = dict.fromkeys(st, 0.0), True
        for i in range(j):
            u = self.vs[rest[i]]
            for r, w in st.items():
                add = w - (mine.get(r, 0.0) if i < p else 0.0)
                if add > EPS and r in u.need and u.need[r] + su[r] + w - mine.get(r, 0.0) - rel[r] - u.sticky.get(r, 0.0) > cap[r] + EPS:
                    ok = False
                    break
            if not ok:
                break
            for r in st:
                rel[r] += u.sticky.get(r, 0.0)
        if ok:
            self.safe_n["alone" if j == 0 else "order"] += 1
            self.order = rest[:j] + [v.id] + rest[j:]
        return ok

    def reduce_all(self, v, st, nd):
        """(2) Full reduction (see safe_to_grant)."""
        cap, su, mine = self.cap, self.sused, v.sticky
        self.safe_n["banker"] += 1                                                               # (2)
        items = {u.id: (u.sticky, u.need) for u in self.vs if u.sticky and u.id != v.id}
        if st:
            items[v.id] = (st, nd)
        used = {r: x for r, x in su.items() if x > EPS}                   # resources occupied by some vehicle after the grant
        for r in set(st) | set(mine):
            x = used.get(r, 0.0) + st.get(r, 0.0) - mine.get(r, 0.0)
            if x > EPS:
                used[r] = x
            else:
                used.pop(r, None)
        cnt, waiters, ready = {}, collections.defaultdict(list), []
        for uid, (s, n) in items.items():
            b = 0
            for r, d in n.items():
                x = used.get(r)
                if x is not None and d + x - s.get(r, 0.0) > cap[r] + EPS:
                    b += 1
                    waiters[r].append(uid)
            if b:
                cnt[uid] = b
            else:
                ready.append(uid)
        freed, order = collections.defaultdict(float), []
        while ready:
            w = ready.pop()
            order.append(w)
            for r, h in items[w][0].items():
                if r not in waiters:
                    continue
                freed[r] += h
                keep = []
                for uid in waiters[r]:
                    s, n = items[uid]
                    if n[r] + used[r] - freed[r] - s.get(r, 0.0) <= cap[r] + EPS:
                        cnt[uid] -= 1
                        if not cnt[uid]:
                            ready.append(uid)
                    else:
                        keep.append(uid)
                waiters[r] = keep
        if len(order) == len(items):
            self.order = order
            return True
        S, wit = self.stuck_core(v.id, set(items) - set(order), items, waiters)
        S = tuple(sorted(S))
        v.unsafe, v.dirty = ((v.phase, v.step), S, wit), False
        for u in S:
            self.watchers[u].add(v.id)
        self.safe_n["unsafe"] += 1
        return False

    def stuck_core(self, vid, stuck, items, waiters):
        """Certificate of unsafety: a "self-blocking" subset Q of the stuck vehicles containing v, and a witness resource per
        vehicle. Starting from v, each vehicle takes one demand r it cannot meet (at the end of the reduction the occupancy of
        the stuck vehicles on r makes it not fit; prefer resources whose holders are already in Q) and all stuck holders of r
        are added to Q, until Q is closed. The holders of every witness resource of a vehicle in Q are all in Q, so with only Q
        left still none of them can finish, which proves unsafety; Q is usually just a few vehicles on one cycle, much smaller
        than the whole stuck set, so cache invalidations (some vehicle in Q changed) are much rarer as well. If v is not among
        the stuck vehicles (should not happen), fall back to the whole stuck set. Returns (Q, {vehicle: witness resource})."""
        if vid not in stuck:
            return frozenset(stuck), {}
        holders = collections.defaultdict(set)
        for w in stuck:
            for r in items[w][0]:
                holders[r].add(w)
        blk = collections.defaultdict(list)
        for r, us in waiters.items():
            for u in us:
                blk[u].append(r)
        Q, wit, todo = {vid}, {}, [vid]
        while todo:
            u = todo.pop()
            r = min(blk[u], key=lambda x: (len(holders[x] - Q - {u}), str(x)))
            wit[u] = r
            for w in sorted(holders[r] - Q - {u}):
                Q.add(w)
                todo.append(w)
        return frozenset(Q), wit

    def still_stuck(self, v, S, wit, st, nd):
        """(0): with only the vehicles in S left (all others finished), still no vehicle in S can finish. Returns False
        (recompute) if some vehicle in S has left the network.
        Vehicle u cannot finish ⇔ some demand r is held by vehicles in S and does not fit (the witness resource);
        wit = {vehicle: last witness resource}, tried first, otherwise search again."""
        items = {}
        for uid in S:
            u = self.vs[uid]
            items[uid] = (st, nd) if uid == v.id else (u.sticky, u.need)
            if uid != v.id and not u.sticky:
                return False
        used = collections.defaultdict(float)
        for s, _ in items.values():
            for r, w in s.items():
                used[r] += w
        cap = self.cap
        for uid, (s, n) in items.items():
            r = wit.get(uid)
            if r is not None and r in n and n[r] + used[r] - s.get(r, 0.0) > cap[r] + EPS:
                continue
            for r, d in n.items():
                x = used.get(r)
                if x is not None and d + x - s.get(r, 0.0) > cap[r] + EPS:
                    wit[uid] = r
                    break
            else:
                return False
        return True

    # ---------------------------------------------------------------- direction lock (reference rule with sectioning)
    def entering(self, v):
        """Sectioned map roads that v's current request enters [(map road, direction, width, section resource)]: it enters if
        it holds no section of the road yet; it also enters if it already holds one but the direction of this request is
        opposite to the direction registered for it on this road (after unloading at a roadside stop it gets a task in the
        opposite direction and turns around)."""
        runs = v.runs[v.phase]
        if v.step >= len(runs):
            return []
        rid, fw = runs[v.step]
        o = self.net.road[rid]["orig"]
        if o not in self.net.sec:
            return []
        if any(("r", x) in v.hold for x in self.net.sec[o]):
            reg = self.onroad[o].get(v.id)
            if reg is None or reg[0] == fw:
                return []
        res = ("r", rid)
        return [(o, fw, next(d for r, d in v.req if r == res), res)]

    def locked(self, v, onroad=None):
        """The direction lock blocks v: the sectioned road it is about to enter carries an oncoming vehicle whose width together
        with its own does not fit (vehicles in the same direction may follow). onroad defaults to the vehicles currently on the road."""
        if not self.lock:
            return False
        onroad = self.onroad if onroad is None else onroad
        return any(u != v.id and fu != fw and d + du > self.cap[res] + EPS
                   for o, fw, d, res in self.entering(v) for u, (fu, du) in onroad.get(o, {}).items())

    def lock_res(self, v):
        return next((res for _, _, _, res in self.entering(v)), None) if self.locked(v) else None

    def mark_road(self, v):
        """v has just acquired a section of a sectioned road: record its direction and width on this map road (update the
        direction if already on the road)."""
        runs = v.runs[v.phase]
        if v.step < len(runs):
            rid, fw = runs[v.step]
            o = self.net.road[rid]["orig"]
            if o in self.net.sec:
                self.onroad[o][v.id] = (fw, v.hold[("r", rid)][1])

    def on_crane_free(self, t, x):
        res, d = x
        self.used[res] -= d
        self.serve_lifts(t, res)          # lifts that arrived before the vehicles waiting at the dock side
        self.wake(t)
        self.serve_lifts(t, res)          # cranes still free after the vehicles have taken theirs

    def on_handled(self, t, k):
        v = self.vs[k]
        if v.phase == "E":
            v.phase, v.step = "L", 0
            self.request(v, t)
            return
        self.free_later(v, t + v.clr, v.sd["res"])
        v.loc = v.sd["node"]
        self.finish(v, v.task, t, v.waited)
        self.idle.append(k)
        self.dispatch(t)

    def wake(self, t):
        for k in list(self.waiting):
            v = self.vs[k]
            if self.fits(v, v.req + v.restore) and not self.locked(v) and (not self.safe or self.safe_to_grant(v)):
                self.waiting.remove(k)
                self.grant(v, t)

    def deadlocked(self):
        """Returns (stuck vehicles, wait cycles, resources each stuck vehicle cannot obtain).
        Graph reduction: vehicles not waiting and vehicles whose requests can be met will release all their resources;
        resources of waiting vehicles with a scheduled release (e.g. a junction just passed) are released as well.
        The waiting vehicles left at the end are the stuck vehicles: both the vehicles on deadlock cycles and bystanders
        blocked by the cycles.
        Wait graph: a resource r that u cannot obtain is held by stuck vehicle w → u→w; if the direction lock blocks u,
        u → the oncoming stuck vehicles on that road. Every stuck vehicle has at least one outgoing edge, so each sink strongly
        connected component of the condensation graph has at least two vehicles that wait only for each other: these are
        the wait cycles; bystanders are in no sink component."""
        stuck, used, onr = self._reduce()
        edges, blocked = {}, {}
        for u in stuck:
            v = self.vs[u]
            blocked[u] = {res for res, d in v.req + v.restore if used[res] + max(0.0, d - self.held(v, res)) > self.cap[res] + EPS}
            edges[u] = {w for w in stuck if w != u and any(
                res in self.vs[w].hold and self.vs[w].pending[res] < self.vs[w].hold[res][0] for res in blocked[u])}
            for o, fw, d, res in self.entering(v) if self.lock else ():
                opp = {w for w, (fu, du) in onr.get(o, {}).items() if w != u and fu != fw and d + du > self.cap[res] + EPS}
                if opp:
                    blocked[u].add(res)
                    edges[u] |= opp
        reach = {}
        for u in stuck:                                   # at most a dozen or so stuck vehicles: reachable sets by one depth-first search each
            seen, todo = set(), list(edges[u])
            while todo:
                w = todo.pop()
                if w not in seen:
                    seen.add(w)
                    todo += edges[w]
            reach[u] = seen
        cores = []
        for u in sorted(stuck):
            scc = {u} | {w for w in reach[u] if u in reach[w]}
            if len(scc) > 1 and all(edges[w] <= scc for w in scc) and scc not in cores:
                cores.append(scc)
        assert not stuck or cores, "a stuck set must contain a wait cycle"
        return stuck, cores, blocked

    def _reduce(self):
        used = collections.defaultdict(float, self.used)
        for c in self.cranes.values():         # cranes are released automatically on time and block no one
            used[c] = 0.0
        stuck = set(self.waiting)
        for v in self.vs:
            for res, (n, d) in v.hold.items():
                if v.id not in stuck or v.pending[res] >= n:
                    used[res] -= d
        # direction lock: vehicles not waiting will all finish and leave the road, only the stuck vehicles remain on it
        onr = {o: {u: x for u, x in d.items() if u in stuck} for o, d in self.onroad.items()} if self.lock else None
        changed = True
        while changed:
            changed = False
            for k in sorted(stuck):
                v = self.vs[k]
                if self.fits(v, v.req + v.restore, used) and not self.locked(v, onr):
                    stuck.remove(k)
                    for res, (n, d) in v.hold.items():
                        if v.pending[res] < n:
                            used[res] -= d
                    for x in (onr or {}).values():
                        x.pop(k, None)
                    changed = True
        return stuck, used, onr

    def check_deadlock(self, t):
        stuck, cores, blocked = self.deadlocked()
        if not stuck:
            return
        if self.safe:
            raise AssertionError(f"deadlock detected under safe release: t = {t:.1f} s, stuck vehicles {sorted(stuck)}")
        if self.p["deadlock_recovery_s"] is None:      # no recovery: record only newly stuck vehicles; they keep holding the road and stay stuck for good
            new = stuck - self.frozen
            if new:
                nc = [c for c in cores if c & new]
                self.deadlocks.append(dict(t=t, size=len(new), core=len(nc[0]) if nc else 0, cores=len(nc), victims=[], flags=[],
                                           res={r for u in (nc[0] if nc else new) for r in blocked[u]}))
                self.frozen |= stuck
            return
        rec = dict(t=t, size=len(stuck), core=len(cores[0]), cores=len(cores), victims=[], flags=[],
                   res={r for u in cores[0] for r in blocked[u]})          # resources the vehicles on the wait cycle cannot obtain
        self.deadlocks.append(rec)
        while stuck:
            # manual recovery: chosen only on wait cycles, the lowest-numbered vehicle of the lowest-numbered cycle (cycles
            # ordered by their smallest vehicle id); it gives up all its resources, waits in a nearby open space for
            # deadlock_recovery_s and then requests again from its original position. Bystanders stay in the queue, no penalty
            v = self.vs[min(cores[0])]
            rec["victims"].append(v.id)
            # diagnostics: whether, when recovered, it held a dock stop resource (ring 0 of the dock-mouth zone), and whether it
            # was carrying a P6 to a dock with cranes
            rec["flags"].append((any(r in self.zone[0] for r in v.hold),
                                 v.phase == "L" and v.task.get("flow") == "P6" and v.sd["ent"] in self.cranes))
            v.victim = True
            self.waiting.remove(v.id)
            for res in list(v.hold):
                self.release(v, res, t, all_=True)
            v.gen += 1
            v.pending.clear()                              # wait_t is kept: the recovery time also counts as waiting for road
            self.push(t + self.p["deadlock_recovery_s"], "recover", v.id)
            self.wake(t)
            stuck, cores, blocked = self.deadlocked()

    def on_recover(self, t, k):
        v = self.vs[k]
        uses = self.leg_of(v)[2]
        v.restore = [(res, d) for res, d, a, r, _ in uses if a < v.step and (r is None or r >= v.step)]
        if v.phase == "L" and v.step == 0:
            v.restore.append((v.so["res"], v.d_so))
        self.request(v, t)

    # ---------------------------------------------------------------- metrics
    def metrics(self):
        day = self.p["day_s"]
        win = [r for r in self.done if self.t0m <= r["done"] <= self.H]
        n_days = (self.H - self.t0m) / day
        q = lambda xs, f: sorted(xs)[min(len(xs) - 1, int(f * len(xs)))] if xs else float("nan")
        flow = [(r["done"] - r["release"]) / 60 for r in win]
        wait = [r["wait"] / 60 for r in win]
        near = self.near + [(v.wait_t, v.block) for v in self.vs if v.wait_t is not None and not v.victim and v.block[0] != "c"
                            and v.id not in self.frozen and self.H - v.wait_t >= self.p["near_deadlock_s"]]
        dl_win = [d for d in self.deadlocks if self.t0m <= d["t"] <= self.H]
        hist = lambda key: dict(sorted(collections.Counter(str(d[key]) for d in self.deadlocks).items(), key=lambda x: int(x[0])))
        occ = collections.defaultdict(float, self.occ)              # stepwise request: resources still held at the end of the simulation are counted up to H
        ivs = collections.defaultdict(list, {r: list(x) for r, x in self.ivs.items()})
        for v in self.vs:
            for res, s in v.since.items():
                occ[res] += max(0.0, self.H - max(s, self.t0m)) * min(1.0, v.hold[res][1] / self.cap[res])
                if self.H > max(s, self.t0m):
                    ivs[res].append((max(s, self.t0m), self.H))
        span = self.H - self.t0m
        # busy-time fraction: time with at least one vehicle on the resource / statistics window. On resources where every
        # vehicle takes more than half the width (e.g. road 161) it is the measure of the k = 1 row that tightens T1';
        # util (weighted by width share) understates such resources
        busy = collections.defaultdict(float, {r: union_len(x) / span for r, x in ivs.items()})
        by_flow = collections.defaultdict(list)
        for r in win:
            by_flow[r["flow"]].append((r["done"] - r["release"]) / 60)
        flows = {f: dict(n_per_day=len(x) / n_days, flow_mean=sum(x) / len(x), flow_p95=q(x, 0.95))
                 for f, x in sorted(by_flow.items(), key=lambda kv: str(kv[0]))}
        zone = {}
        for name, ring in [(f"ring{i}", x) for i, x in enumerate(self.zone)] + [("all", set().union(*self.zone))]:
            u = [occ[r] / span for r in ring] or [0.0]           # synthetic maps have no dock
            bz = [busy[r] for r in ring] or [0.0]
            zone[name] = dict(n=len(ring), util_mean=sum(u) / len(u), util_max=max(u), busy_mean=sum(bz) / len(bz), busy_max=max(bz),
                                    wait_min_per_day=sum(self.wait_by[r] for r in ring) / 60 / n_days,
                                    deadlocks_per_day=sum(bool(d["res"] & ring) for d in dl_win) / n_days,
                                    near_per_day=sum(self.t0m <= x <= self.H and b in ring for x, b in near) / n_days)
        # list positive values only: the order of tied zeros depends on set iteration and changes with PYTHONHASHSEED; the
        # order of positive values is stable. Cranes are excluded from the top 5: they are reported in crane, the top 5 stay network resources
        top = lambda d, f: [dict(res=list(r), name=res_name(self.net, r), value=f(x))
                            for r, x in sorted(d.items(), key=lambda kv: -kv[1]) if x > 0 and r[0] != "c"][:5]
        # full readings of the dock-mouth zone and door resources (workshop entrance nodes): aggregating "most occupied /
        # most waited for" across seeds then need not rely on the top 5 alone
        key = lambda r: "|".join(map(str, r))
        watch = {key(r): [busy.get(r, 0.0), occ.get(r, 0.0) / span, self.wait_by.get(r, 0.0) / 60 / n_days]
                 for r in sorted(self.watch, key=key)}
        days_ = range(self.p["warmup_days"], self.p["days"])
        per_day = collections.Counter(min(int(r["done"] // day), self.p["days"] - 1) for r in win)    # tasks completed exactly at H count for the last day
        extra = {}
        if self.cranes:          # cranes: consistency check, waiting attributed to cranes, utilization (crane hours / available crane hours) and busy-time fraction, the higher of the two docks
            cs = self.cranes.values()
            extra["crane"] = dict(self.crane_check(n_days), wait_attr_min_per_day=sum(self.wait_by[c] for c in cs) / 60 / n_days,
                                  util_max=max(occ[c] / span for c in cs), busy_max=max(busy[c] for c in cs))
            # lifts inside the crane span: number released and completed in the statistics window (per day), crane wait and flow
            # time (release → hooking and hoisting done, min), number still queued at the end of the simulation
            lw = [r for r in self.lifts if self.in_win(r["done"])]
            extra["lift"] = dict(released_per_day=self.lift_rel / n_days, n_per_day=len(lw) / n_days,
                                 wait_mean=sum(r["start"] - r["release"] for r in lw) / 60 / len(lw) if lw else float("nan"),
                                 flow_mean=sum(r["done"] - r["release"] for r in lw) / 60 / len(lw) if lw else float("nan"),
                                 flow_p95=q([(r["done"] - r["release"]) / 60 for r in lw], 0.95),
                                 queue_end=sum(len(x) for x in self.lift_q.values()))
            extra["starve"] = self.starvation()
        if self.mode == "segment":       # diagnostics of the reference rule (teleport recovery): among the vehicles recovered in the statistics window, those holding a dock stop resource and those carrying a P6 to the dock (per day)
            fl = [f for d in dl_win for f in d["flags"]]
            extra["victims_diag"] = dict(per_day=len(fl) / n_days, at_dock_per_day=sum(a for a, _ in fl) / n_days,
                                         p6_loaded_per_day=sum(b for _, b in fl) / n_days, frozen_end=len(self.frozen))
        if self.safe:                    # safe release: number of decisions per step, number and duration of waits due to refusals in the statistics window
            extra["safe"] = dict(self.safe_n, wait_n_per_day=self.safe_n["wait_n"] / n_days,
                                 wait_min_per_day=self.safe_n["wait_s"] / 60 / n_days)
        extra["cycle"] = self.cycle(win)
        return dict(extra, throughput=len(win) / n_days, completed=len(win),
                    flow_mean=sum(flow) / len(flow) if flow else float("nan"), flow_p95=q(flow, 0.95),
                    wait_mean=sum(wait) / len(wait) if wait else float("nan"), wait_p95=q(wait, 0.95),
                    # deadlocks: deadlocks counts the whole simulation (including warm-up days); deadlocks_per_day only the
                    # statistics window (days 2-4), on the same basis as throughput
                    deadlocks=len(self.deadlocks), deadlocks_per_day=len(dl_win) / n_days,
                    deadlock_times_h=[round(d["t"] / 3600, 3) for d in self.deadlocks],
                    deadlock_stuck_hist=hist("size"), deadlock_core_hist=hist("core"),
                    victims=sum(len(d["victims"]) for d in self.deadlocks),
                    near_deadlocks=len(near), near_per_day=sum(self.t0m <= x <= self.H for x, _ in near) / n_days,
                    near_times_h=[round(x / 3600, 3) for x, _ in sorted(near)],
                    backlog=len(self.queue), waiting_end=len(self.waiting),
                    # flow time per flow, dock-mouth zone readings, the 5 resources with the highest utilization and attributed waiting (waiting in min per day)
                    flows=flows, dock=zone, util_top=top(occ, lambda x: x / span), busy_top=top(busy, lambda x: x),
                    wait_top=top(self.wait_by, lambda x: x / 60 / n_days),
                    wait_attr_min_per_day=sum(self.wait_by.values()) / 60 / n_days,
                    # watch[resource] = [busy-time fraction, width-weighted utilization, attributed waiting min/day]; thr_by_day = completions per day in the statistics window
                    watch=watch, thr_by_day=[per_day[d] for d in days_])

    def crane_check(self, n_days):
        """Consistency check of the number of cranes working at once: in order of unloading start (done - unload_s) the crane
        of that dock that becomes free first takes over, each time for crane_cycle_s; if all cranes of the dock are busy when
        unloading starts, i.e. the number working at once would exceed the number of cranes, one violation and its excess
        duration are recorded (in the statistics window, per day). The occupancy rules start unloading only when a crane is
        free, so this is 0 by construction (it checks the implementation of the crane capacity, not "vehicles never wait for
        cranes"; the time waiting for cranes is in wait_attr_min_per_day); free flow does not check capacity, so these two
        numbers are what free flow ignores about cranes. Lifts inside the crane span are scheduled in as well, by lift start time."""
        cnt = self.net.crane_counts(self.p)
        free = {z: [0.0] * n for z, n in cnt.items()}
        n = w = 0.0
        jobs = [(r["done"] - self.p["unload_s"], r["id"], r) for r in self.done if r["flow"] == "P6"] + \
            [(r["start"], r["id"], r) for r in self.lifts]
        for s, _, r in sorted(jobs, key=lambda x: x[:2]):
            f = free[self.net.stops[r["d"]]["ent"]]
            i = min(range(len(f)), key=f.__getitem__)
            x = max(0.0, f[i] - s)
            f[i] = s + x + self.p["crane_cycle_s"]
            if self.in_win(r["done"]):
                n, w = n + (x > EPS), w + x
        return dict(waits_per_day=n / n_days, wait_min_per_day=w / 60 / n_days)

    def starvation(self):
        """Crane starvation rate: fraction of the statistics window in which a dock "has a free crane while a released P6 has
        not started its lift", starve; the part in which "the vehicle of such a P6 is waiting for road resources", blocked
        (stepwise rules: the vehicle waits on the road for a road, junction or stop, including time out of the network after
        a manual recovery; reservation: the vehicle waits at a stop for its reserved departure time). The crane interval is
        [start of lift, + crane_cycle_s] (transporter: start of unloading; lift inside the span: start of lift); a P6 counts
        as "released, not started" from release to start of lift (queued for dispatch, empty run, loading, loaded run, waiting
        for a crane); P6 not started at the end of the simulation are counted up to H. Returns {dock: {starve, blocked}} and
        the higher of the two docks. starve measures how far effective capacity falls short although nominal ρ < 1; blocked
        is the part caused by the network."""
        cyc, H, inf = self.p["crane_cycle_s"], self.H, float("inf")
        jobs = collections.defaultdict(list)

        def add(task, cs, blk, since=None, road=None):
            """road: the intervals in which the vehicle waits for roads (not for a crane); for the reservation family split by
            the resources that shifted the departure, for the stepwise family blk never contains crane waits anyway."""
            z = self.net.stops[task["d"]]["ent"]
            if z in self.cranes and task["release"] <= H:
                b = list(blk or []) + ([(since, H)] if since is not None else [])
                jobs[z].append((task["release"], inf if cs is None else cs, b, b if road is None else list(road)))
        for r in self.done:
            if r["flow"] == "P6":
                add(r, r["cs"], r["blk"], road=r.get("blk_road") if self.res else None)
        for r in self.lifts:
            add(r, r["start"], None)
        for q in self.lift_q.values():
            for _, task in q:
                add(task, None, None)
        for task in self.queue:
            if task.get("flow") == "P6":
                add(task, None, None)
        for _, _, kind, x in self.ev:            # reservation, free flow: scheduled but not yet completed
            if kind == "done" and x[1].get("flow") == "P6":
                add(x[1], x[1].get("crane_start"), x[1].get("blk"), road=x[1].get("blk_road") if self.res else None)
        for v in self.vs if self.seg else ():
            if v.id not in self.idle and v.task.get("flow") == "P6":
                waiting = v.wait_t is not None and v.block[0] != "c"
                add(v.task, v.task.get("crane_start"), v.task.get("blk"), v.wait_t if waiting else None)
        cnt, span, out = self.net.crane_counts(self.p), H - self.t0m, {}
        def sweep(z, part):
            """Total time in which the dock "has a free crane and a P6 waiting for its lift", and the part of it in which some
            vehicle is in a `part` interval (blk or road)."""
            ev = []
            for job in jobs[z]:
                rel, cs = job[:2]
                end = min(cs, H)
                ev += [(rel, 1, 0, 0), (end, -1, 0, 0)]
                if cs < H:
                    ev += [(cs, 0, 1, 0), (cs + cyc, 0, -1, 0)]
                for s, e in job[part]:
                    s, e = max(s, rel), min(e, end)
                    if e > s:
                        ev += [(s, 0, 0, 1), (e, 0, 0, -1)]
            pend = busy = nb = 0
            st = sb = 0.0
            prev = 0.0
            for t, a, b, c in sorted(ev):
                lo, hi = max(prev, self.t0m), min(t, H)
                if hi > lo and pend > 0 and busy < cnt[z]:
                    st += hi - lo
                    sb += (hi - lo) if nb > 0 else 0.0
                prev, pend, busy, nb = t, pend + a, busy + b, nb + c
            return st, sb
        for z in self.cranes:
            st, sb = sweep(z, 2)
            out[z] = dict(starve=st / span, blocked=sb / span, blocked_road=sweep(z, 3)[1] / span)
        return dict(by_dock=out, starve_max=max((x["starve"] for x in out.values()), default=0.0),
                    blocked_max=max((x["blocked"] for x in out.values()), default=0.0),
                    blocked_road_max=max((x["blocked_road"] for x in out.values()), default=0.0))

    def cycle(self, win):
        """Cycle-time breakdown (min/task, transporter tasks completed in the statistics window; for external comparison):
        empty and loaded runs at no-wait travel time (shortest path / speed), loading and unloading at their durations;
        waiting = dispatch to end of unloading - the above (waiting at stops or on the road for road resources, for cranes,
        in recovery). Free flow has zero waiting."""
        rs = [r for r in win if r.get("start") is not None]
        if not rs:
            return {}
        p, net, n = self.p, self.net, len(rs)
        node = lambda s: net.stops[s]["node"]
        emp = sum(net.dist(r["start"], node(r["o"])) for r in rs) / p["v_empty"] / n / 60
        ld = sum(net.dist(node(r["o"]), node(r["d"])) for r in rs) / p["v_loaded"] / n / 60
        tot = sum(r["done"] - r["dispatch"] for r in rs) / n / 60
        hand = (p["load_s"] + p["unload_s"]) / 60
        return dict(empty=emp, load=p["load_s"] / 60, loaded=ld, unload=p["unload_s"] / 60, wait=tot - emp - ld - hand, total=tot)

    def dispatched(self):
        """All dispatched tasks (completed records + tasks still in progress at the end of the simulation), each with its
        dispatch time dispatch. Used by the steady-state criterion."""
        live = [x[1] for _, _, kind, x in self.ev if kind == "done"]      # free flow, reservation: scheduled completion events
        live += [v.task for v in self.vs if self.seg and v.id not in self.idle]
        return self.done + live

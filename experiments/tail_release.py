"""Execution in booked order with release at the later of timer and tail clearance.

ordered.py and ordered_proj.py are not changed by it; here TailOPE, a subclass of ordered_proj.OPE2, with three kinds of
release:
  timer  as before: a booking is released when its timer expires (= OPE2, bookings as in the paper's booking windows);
  later  released at the later of timer expiry and tail clearance, booked as before;
  book   as later, and at booking the resources on which the tail still lies while the vehicle loads or unloads at a
         roadside stop are booked until the body has cleared them without waiting, with the loaded width (booking the
         tail); a vehicle with an empty run of length zero that loads at a roadside origin also books the resources behind
         its body (condition (B5) and the tail lemma of the paper).
Body model: the front of the vehicle moves along the chain; the vehicle is L0 long when empty and L from the start of
loading; it is as wide as an empty vehicle when empty and takes the loaded width from the start of loading.
  On a resource: from the decision point that enters it until the front has passed the far end of the resource + vehicle
  length + longitudinal clearance (far end of a junction / entrance = node + D_n, the same definition as the timer).
  Stops: at off-road stops (entrance into a shop, drive-in bay) the vehicle drives in as before; at roadside stops it
  stands while loading or unloading: at the destination it leaves when unloading is done and it clears the stop (leaves
  the road), at the origin it continues on the loaded run after loading.
  A resource already cleared by the tail on the empty run may be covered again when the body grows on loading (situation
  1 of the check); with an empty run of length zero the vehicle stands on the stop facing the loaded direction and its
  body lies behind the stop (situation 3).
Independent check (without the booking and release logic of the executor): from the recorded actual departure times at
every decision point and the driving times, the front trajectory, vehicle length and width are rebuilt and the period of
the body on every resource is computed:
  B1 the body period is covered by holds of this vehicle on this resource of at least the body width; B2 the occupancy
  rescanned time by time from body periods and widths (per vehicle the largest width) stays within capacity;
  and as in ordered_check: C1 held occupancy within capacity, C2 entry order agrees with the keys (adm), C3 every booking
  entered and released.
Instances: short (this file, with the hand-made counterexample tailx), aud (the instances near / band / mix / zero of an
independent check, copied), same (its instances with tasks between stops on the same node, copied), turn (turning back at
a dead end and very short empty runs from roadside stops), adv, crane (ordered_check: adversarial instances with cranes,
lifts inside the crane span and the dock-side off-road bay), base (the self-check of ordered.py).

  python experiments/tail_release.py --out DIR [--seeds 100] [--procs 8] [--slots DIR] [--suites a,b]
  -> DIR/cases_<suite>.csv per suite (suites already written are skipped, so a run can be resumed), and once all are
     written DIR/cases.csv, DIR/summary.json, DIR/tables.md."""
import argparse
import collections
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "tests"))
import ordered_check as C                            # noqa: E402
import ordered as O                                  # noqa: E402
import ordered_proj as Q                             # noqa: E402
from occupancy import params as P                    # noqa: E402
from occupancy import sim as S                       # noqa: E402
from occupancy.network import EPS                    # noqa: E402

MODES = ("timer", "later", "book")
TOL = 1e-6


# -------------------------------------------------------------- geometry: chainage of the decision points and tail clearance
def geometry(p, legs, task):
    """Decision-point geometry of a chain: run (0 empty / 1 loaded), pos (chainage within the run, booked speed x booked
    time), whether the stops at the two ends are roadside, and the growth of the vehicle length on loading."""
    so, sd, d_so, d_sd, (TE, stE, uE, aE), (TL, stL, uL, aL), tail, clr = legs
    vE, vL = p["v_empty"], p["v_loaded"]
    nE = aE + 1
    return dict(nE=nE, n=nE + aL + 1, run=[0] * nE + [1] * (aL + 1), pos=[stE[k] * vE for k in range(nE)] + [stL[k] * vL for k in range(aL + 1)],
                v=(vE, vL), road_o=so.get("mode") == "on_road", road_d=sd.get("mode") == "on_road", dL=task["Lv"] - p["plat_len"])


def loaded_plan(g, rem):
    """After the loaded departure the front still has to cover rem: ("run", last decision point passed, remaining
    distance, 1); past a roadside destination -> ("dest",)."""
    pos, z, last = g["pos"], g["nE"], g["n"] - 1
    for k in range(g["nE"] + 1, last + 1):
        if pos[k] < rem - 1e-9:
            z = k
    if rem > pos[last] + 1e-9 and z == last and g["road_d"]:
        return ("dest",)
    return ("run", z, rem - pos[z], 1)


def tail_plan(g, y, lam):
    """For a booking released from y with clearance length lam (by the vehicle length of that run): the last decision
    point z the vehicle passes before the body has cleared, and the distance still to drive after it -> ("run", z, rem,
    run); the body leaves only when unloading at a roadside destination is done and the stop is cleared -> ("dest",).
    z == y and rem == lam: no decision point and no stop on the way, clearance exactly at timer expiry.
    Empty run at a roadside origin: the vehicle grows to L when loading starts; clearances by L reaching past the stop are
    carried over to the loaded run."""
    pos, r = g["pos"], g["run"][y]
    end = g["nE"] - 1 if r == 0 else g["n"] - 1
    target, z = pos[y] + lam, y
    for k in range(y + 1, end + 1):
        if pos[k] < target - 1e-9:
            z = k
    cross = target > pos[end] + 1e-9 if not (r == 0 and g["road_o"]) else target + g["dL"] > pos[end] + 1e-9
    if not cross:
        return ("run", z, target - pos[z], r)
    if r == 1:                                        # body past the destination stop
        return ("dest",) if g["road_d"] else ("run", end, target - pos[end], 1)
    if not g["road_o"]:                               # off-road origin: drives into the shop / bay
        return ("run", end, target - pos[end], 0)
    # roadside origin: continued on the loaded run after loading, length L
    return loaded_plan(g, target - pos[end] + g["dL"])


def clearing(sim, legs, task):
    """For every use of a booking (run, index) -> (releasing decision point y (chain index), clearance length lambda);
    destination stop resources, cranes and bays are not included (released by their own timers)."""
    so, sd, d_so, d_sd, (TE, stE, uE, aE), (TL, stL, uL, aL), tail, clr = legs
    p, nE = sim.p, aE + 1
    out = {}
    for run, uses, off, v, Lv in ((0, uE, 0, p["v_empty"], p["plat_len"]), (1, uL, nE, p["v_loaded"], task["Lv"])):
        for i, (res, d, a, r, dl) in enumerate(uses):
            if d <= EPS or res[0] in ("c", "b"):
                continue
            if r is None:
                if run == 0:                              # origin stop resource: L + s from the loaded departure
                    out[(0, i)] = (nE, task["Lv"] + p["long_clear"])
                continue
            lam = dl * v
            if res[0] == "n":
                lam = min(lam, sim.size[res[1]] + Lv + p["long_clear"])
            out[(run, i)] = (off + r, lam)
    return out


def _ins(net):
    if not hasattr(net, "_tail_ins"):
        ins = collections.defaultdict(list)
        for i, a in enumerate(net.arc):
            ins[a[1]].append(i)
        net._tail_ins = ins
    return net._tail_ins


def behind(sim, legs, task):
    """Resources behind a vehicle that loads at a roadside origin with its body (length L) reaching back past the start of
    the empty run (situations 3 and 2 of the check):
      empty run of length zero: the vehicle stands on stop S facing the loaded direction; walk back from S;
      very short empty run: the body first lies back along the empty route to its start (those resources are held by the
      bookings of the empty run, not listed), then the walk continues back from the start.
    Walking back: straight on along the same graph road, at nodes the incoming road with the smallest turn; when the
    start is an entrance (shop) the body is off the road behind it, not listed. Lists the resources covered by the body or
    its clearance: [(resource, distance the front still has to drive after the loaded departure, far end in S
    coordinates)]."""
    so = legs[0]
    net, p = sim.net, sim.p
    if so.get("mode") != "on_road":
        return []
    Sn, Lv, s = so["node"], task["Lv"], p["long_clear"]
    reach = Lv + s + max(sim.size.values(), default=0.0)
    ins = _ins(net)
    orig = lambda a: net.road[net.arc[a][3]]["orig"]
    straight = lambda node, ref: ([b for b in ins[node] if orig(b) == orig(ref) and net.arc[b][4] == net.arc[ref][4]] if node.startswith("S|")
                                  else sorted((b for b in ins[node] if not (net.arc[b][0] == net.arc[ref][1] and net.arc[b][3] == net.arc[ref][3])),
                                              key=lambda b: (net.turn_deg(b, ref), b)))
    if legs[4][3] == 0:
        sd_node = sim.net.stops[task["d"]]["node"]
        path = net.path(Sn, sd_node) if sd_node != Sn else []
        if path:
            cand = straight(Sn, path[0])
        # task between stops on the same node: the vehicle does not move, direction = forward on the road
        else:
            cand = sorted(a for a in ins[Sn] if net.arc[a][4])
        dist, seen = 0.0, {so["res"]}
    else:
        start = task.get("start")
        epath = net.path(start, Sn)
        dist = sum(net.arc[a][2] for a in epath)
        if dist >= reach or start in net.access:
            return []
        cand = straight(start, epath[0])
        seen = {u[0] for u in legs[4][2]} | {so["res"]}
    if not cand:
        return []
    a, out = cand[0], []
    while a is not None and dist < reach:
        t, h, ln, rid, fw = net.arc[a]
        res = ("r", rid)
        if res not in seen and dist < Lv + s:
            out.append((res, Lv + s - dist, -dist))
        seen.add(res)
        dist += ln
        if ("n", t) in sim.cap and ("n", t) not in seen:
            D = sim.size[t]
            if dist < D + Lv + s:
                out.append((("n", t), D + Lv + s - dist, -dist + D))
            seen.add(("n", t))
        prev = straight(t, a)
        a = prev[0] if prev else None
    return out


def extensions(sim, legs, task):
    """book: bookings on which the tail lies while the vehicle loads or unloads at a roadside stop, booked until the body
    has cleared them without waiting. Returns {(run, index): (new releasing decision point, booked end relative to t2,
    timer, width)}; timer ("run", remaining distance) or ("dest",) (unloading done + clearance); the extended bookings of
    the empty run take the loaded width (the body is loaded-wide after loading starts).
    Also returns the bookings behind a zero-length empty run [(resource, new releasing decision point, booked end relative
    to t2, timer, width)]."""
    so, sd, d_so, d_sd, (TE, stE, uE, aE), (TL, stL, uL, aL), tail, clr = legs
    g = geometry(sim.p, legs, task)
    vL, nE, U = sim.p["v_loaded"], g["nE"], sim.p["unload_s"]
    loaded = lambda res, d: max(d, sim.net.demand(res, task["B"], True, sim.p, sim.cap))
    out = {}
    for key, (y, lam) in clearing(sim, legs, task).items():
        pl = tail_plan(g, y, lam)
        d = (uE if key[0] == 0 else uL)[key[1]][1]
        res = (uE if key[0] == 0 else uL)[key[1]][0]
        w = loaded(res, d) if key[0] == 0 else d
        if pl[0] == "dest":
            out[key] = (g["n"] - 1, TL + U + clr, ("dest",), w)
        elif g["run"][y] == 0 and pl[3] == 1:
            out[key] = (pl[1], stL[pl[1] - nE] + pl[2] / vL, ("run", pl[2]), w)
    beh = []
    for res, rem, _ in behind(sim, legs, task):
        pl = loaded_plan(g, rem)
        w = loaded(res, 0.0)
        if pl[0] == "dest":
            beh.append((res, g["n"] - 1, TL + U + clr, ("dest",), w))
        else:
            beh.append((res, pl[1], stL[pl[1] - nE] + pl[2] / vL, ("run", pl[2]), w))
    return out, beh


# ------------------------------------------------ booking (book): tail windows stretch with t1 and t2 like the origin window
class _TailBook(S.Sim):
    def plan(self, v, task, t0):
        if getattr(self, "tail", "timer") != "book":
            return super().plan(v, task, t0)
        assert not self.net.wait_at and self.mode == "reserve", "book: only the basic form of full-path reservation"
        legs = self.legs(v, task)
        so, sd, d_so, d_sd, (TE, stE, uE, aE), (TL, stL, uL, aL), tail_l, clr = legs
        ext, beh = extensions(self, legs, task)
        load = self.p["load_s"]
        io = next(i for i, u in enumerate(uE) if u[3] is None)
        # the origin stop resource itself may have to be booked until the roadside destination is cleared
        if (0, io) in ext:
            tail_l, d_so = ext[(0, io)][1], ext[(0, io)][3]
        WE = [(res, stE[a], stE[r] + dl, d) for i, (res, d, a, r, dl) in enumerate(uE) if r is not None and (0, i) not in ext]
        OW = [(res, stE[a], ext[(0, i)][1], ext[(0, i)][3]) for i, (res, d, a, r, dl) in enumerate(uE) if r is not None and (0, i) in ext]
        # resources behind the start of the empty run: booked from the decision point of arrival (start of loading)
        OW += [(res, stE[aE], e2, w) for res, _, e2, _, w in beh]
        WL = [(res, stL[a], ext[(1, i)][1] if (1, i) in ext else stL[r] + dl, d) for i, (res, d, a, r, dl) in enumerate(uL) if r is not None]
        WL += [(res, TL, TL + self.p["crane_cycle_s"], d) for res, d, a, r, dl in uL if res[0] == "c"]
        ao = next(stE[a] for _, _, a, r, _ in uE if r is None)
        ad = next(stL[a] for _, _, a, r, _ in uL if r is None)
        tmin, jumps = t0, []
        while True:
            jL = []
            t1 = self.earliest(WE + [(so["res"], ao, TE + load + tail_l, d_so)] + [(r_, s_, TE + load + e_, d_) for r_, s_, e_, d_ in OW],
                               tmin, v.id, False, jumps)
            a = t1 + TE
            t2, tc, WLa, bay = self.book_l(sd, d_sd, TL, WL, ad, clr, a + load, v.id, False, jL)
            o2, off = self.conflict(so["res"], t1 + ao, t2 + tail_l, d_so, v.id), ao
            for r_, s_, e_, d_ in OW:
                if o2 is not None:
                    break
                o2, off = self.conflict(r_, t1 + s_, t2 + e_, d_, v.id), s_
            if o2 is None:
                break
            jumps.append((so["res"], o2 - off - t1))
            tmin = o2 - off
        jumps += jL
        if self.p["reserve_jit"]:
            tj = t2 - load - TE
            if tj > t1 + EPS and all(self.conflict(r, tj + s, tj + e, d, v.id) is None for r, s, e, d in WE):
                t1, a = tj, tj + TE
        W = [(r, t1 + s, t1 + e, d) for r, s, e, d in WE] + [(so["res"], t1 + ao, t2 + tail_l, d_so)] + \
            [(r, t1 + s, t2 + e, d) for r, s, e, d in OW] + WLa
        self.commit(v.id, t0, W)
        self.account(W)
        task["plan"] = dict(start=v.loc, t1=t1, a=a, t2=t2)
        if any(r[0] == "c" for r, *_ in WL):
            task["crane_start"] = tc
        v.loc = sd["node"]
        self.push(tc + self.p["unload_s"], "done", (v.id, task, 0.0))


# ---------------------------------------------------------------- executor
class TailOPE(Q.OPE2, _TailBook):
    def __init__(self, *a, tail="timer", **k):
        assert tail in MODES
        self.tail = tail
        self.tstate, self.zwait, self.dlink = {}, collections.defaultdict(list), collections.defaultdict(list)
        super().__init__(*a, **k)
        self.tn = collections.Counter()
        self.tail_late_max = 0.0

    def build(self, v, task, t0, legs, W):
        """As ordered.OPE.build (identical bookings under timer and later); under book the bookings on which the tail lies at
        a stop are extended and widened by extensions, and the bookings behind a zero-length empty run are added."""
        so, sd, d_so, d_sd, (TE, stE, uE, aE), (TL, stL, uL, aL), tail, clr = legs
        t1, t2 = task["plan"]["t1"], task["plan"]["t2"]
        load, unload, fE, fL = self.draw(task)
        p, cid = self.p, len(self.chains)
        ev = [dict(beta=t1 + stE[k], dur=(stE[k] - stE[k - 1]) / fE if k else 0.0, acq=[], rel=[]) for k in range(aE + 1)]
        nE = len(ev)
        ev += [dict(beta=t2 + stL[k], dur=(stL[k] - stL[k - 1]) / fL if k else load, acq=[], rel=[]) for k in range(aL + 1)]
        ext, beh = extensions(self, legs, task) if self.tail == "book" else ({}, [])
        made = {}
        timer_of = lambda kind: kind[1] / (p["v_loaded"] * fL) if kind[0] == "run" else unload + clr / fE

        def add(key, res, d, s, e, a, r, timer):
            b = O.Bk(res, d, s, e, cid, v.id, a)
            ev[a]["acq"].append(b)
            ev[r]["rel"].append((b, timer))
            b.rb = ev[r]["beta"]
            made[key] = b
            return b
        tau, U = p.get("crane_cycle_s"), p["unload_s"]
        dest = None
        for i, (res, d, a, r, dl) in enumerate(uE):
            if d <= EPS:
                continue
            if (0, i) in ext:
                z, e2, kind, w = ext[(0, i)]
                add((0, i), res, w, t1 + stE[a], t2 + e2, a, z, timer_of(kind))
            elif r is None:
                add((0, i), res, d_so, t1 + stE[a], t2 + tail, a, nE, tail / fL)
            else:
                add((0, i), res, d, t1 + stE[a], t1 + stE[r] + dl, a, r, dl / fE)
        for j, (res, z, e2, kind, w) in enumerate(beh):
            add(("B", j), res, w, t1 + stE[aE], t2 + e2, aE, z, timer_of(kind))
        for i, (res, d, a, r, dl) in enumerate(uL):
            if d <= EPS:
                continue
            if res[0] == "c":
                add((1, i), res, d, t2 + TL, t2 + TL + tau, nE + aL, nE + aL, unload + tau - U)
            elif r is None:
                dest = add((1, i), res, d_sd, t2 + stL[a], t2 + TL + U + clr, nE + a, nE + aL, unload + clr / fE)
            elif (1, i) in ext:
                add((1, i), res, d, t2 + stL[a], t2 + ext[(1, i)][1], nE + a, nE + aL, unload + clr / fE)
            else:
                add((1, i), res, d, t2 + stL[a], t2 + stL[r] + dl, nE + a, nE + r, dl / fL)
        mine = sorted((b.res, b.s, b.e, b.d) for x in ev for b, _ in x["rel"])
        assert len(mine) == len(W) and all(a[0] == b[0] and abs(a[1] - b[1]) < O.TOL and abs(a[2] - b[2]) < O.TOL and abs(a[3] - b[3]) < 1e-9
                                           for a, b in zip(mine, sorted(W))), "chain does not match the booked windows"
        for x in ev:
            for b in x["acq"]:
                self.bk[b.res].append(b)
        ch = dict(cid=cid, v=v, task=task, ev=ev, k=0, ready=t0, U=unload, done=False, t0=t0, active=sum(x["dur"] for x in ev) + unload)
        self.push(max(t0, ev[0]["beta"]), "try", ch, 2)
        ch["f"], ch["legs"], ch["made"] = (fE, fL), legs, made
        self._record(ch, legs, task, made, dest, ext)
        return ch

    def _record(self, ch, legs, task, made, dest, ext):
        """Under later / book, attaches the wait for tail clearance to the bookings whose tail has not cleared (bookings
        extended by book and bookings behind: no decision point after the releasing one, the timer is the clearance)."""
        g = geometry(self.p, legs, task)
        ch["g"], ch["clr_s"] = g, legs[-1]
        if self.tail == "timer":
            return
        cl = clearing(self, legs, task)
        for key, b in made.items():
            if key not in cl or key in ext:
                continue
            y, lam = cl[key]
            pl = tail_plan(g, y, lam)
            if pl[0] == "run" and pl[1] == y and abs(pl[2] - lam) < 1e-9:
                # no decision point and no stop on the way: clearance exactly at timer expiry
                continue
            self.tstate[b] = dict(tail=False, timer=False, plan=pl)
            if pl[0] == "dest":
                self.dlink[dest].append(b)
            else:
                self.zwait[(ch["cid"], pl[1])].append((b, pl[2], pl[3]))

    # ---------------------------------------------------------------- release: the later of timer expiry and tail clearance
    def attempt(self, ch, t):
        k0 = ch["k"]
        ok = super().attempt(ch, t)
        if ok and not ch.get("lift"):
            for b, rem, run in self.zwait.pop((ch["cid"], k0), ()):
                self.push(t + rem / (ch["g"]["v"][run] * ch["f"][run]), "tclr", b, 0)
        return ok

    def on_rel(self, t, b):                        # timer expired
        st = self.tstate.get(b)
        if st is not None and not st["tail"]:
            st["timer"], st["t_timer"] = True, t
            return
        self._release(t, b)

    def on_tclr(self, t, b):                       # tail cleared
        st = self.tstate[b]
        st["tail"] = True
        if st["timer"]:
            self.tn["tail_holds"] += t > st["t_timer"] + TOL
            self.tail_late_max = max(self.tail_late_max, t - st["t_timer"])
            self._release(t, b)

    def _release(self, t, b):
        super().on_rel(t, b)
        for b2 in self.dlink.pop(b, ()):
            self.push(t, "tclr", b2, 0)

    def blockers(self, ch, x):
        """Also checks (B5): an overdue booking pointed to by a capacity wait whose tail has not cleared must have the
        decision point its tail is still to pass booked before beta(x) (asserted under book)."""
        out = super().blockers(ch, x)
        for kind, b2 in out:
            st = self.tstate.get(b2)
            if kind != "cap" or st is None or st["tail"]:
                continue
            c2 = self.chains[b2.cid]
            if c2["ev"] is None:
                continue
            z = len(c2["ev"]) - 1 if st["plan"][0] == "dest" else st["plan"][1]
            if c2["k"] <= z and c2["ev"][z]["beta"] >= x["beta"] - 1e-9:
                self.tn["premise_viol"] += 1
                assert self.tail != "book", ("B5: decision point the tail is to pass not booked before beta(x)", c2["ev"][z]["beta"], x["beta"])
        return out

    def on_fin(self, t, ch):
        if not ch.get("lift"):
            ch["tl"] = [x.get("t") for x in ch["ev"]]
            ch["durs"] = [x["dur"] for x in ch["ev"]]
        return super().on_fin(t, ch)


# ---------------------------------------------------------- independent check: actual position, length and width of the body
class Motion:
    """Front trajectory of a completed chain (from the actual departure times at the decision points and the actual driving
    times), vehicle length L0 -> L (from the start of loading), time of leaving the road."""

    def __init__(self, sim, ch):
        g, p = ch["g"], sim.p
        self.ch, self.pos, self.tl, self.f, self.v = ch, g["pos"], ch["tl"], ch["f"], g["v"]
        self.nE, self.n, self.road_o, self.road_d = g["nE"], g["n"], g["road_o"], g["road_d"]
        self.L0, self.L, self.s = p["plat_len"], ch["task"]["Lv"], p["long_clear"]
        # leaving the arrival point of the empty run = start of loading
        self.t_load = self.tl[self.nE - 1]
        # roadside destination: unloading done + clearance to leave
        self.t_leave = self.tl[self.n - 1] + ch["U"] + ch["clr_s"] / self.f[0]

    def reach(self, run, k, X):
        """Time at which the front, having left decision point k (index within the run), reaches coordinate X of the run;
        at a roadside origin continued on the loaded run (X in loaded-run coordinates); stopped at a roadside destination
        -> None (leaves the road)."""
        off, end = (0, self.nE - 1) if run == 0 else (self.nE, self.n - 1 - self.nE)
        pos = lambda i: self.pos[off + i]
        while True:
            if k < end and X <= pos(k + 1) + 1e-9:
                return self.tl[off + k] + max(0.0, X - pos(k)) / (self.v[run] * self.f[run])
            if k < end:
                k += 1
                continue
            if run == 1:
                return None if self.road_d else self.tl[off + k] + (X - pos(k)) / (self.v[1] * self.f[1])
            if not self.road_o:
                return self.tl[off + k] + (X - pos(k)) / (self.v[0] * self.f[0])
            run, off, end, X, k = 1, self.nE, self.n - 1 - self.nE, X - pos(k), 0
            pos = lambda i: self.pos[self.nE + i]


def body_items(sim, ch):
    """Body periods of a chain on the spatial resources: ([(resource, start, end, width)], the use keys of these periods).
    At a roadside destination the body stays until unloading is done and the stop is cleared (leaves the road).
    Independent of the executor's bookings and clearance plans: only the route geometry (entry point and far end of every
    use) and the actual trajectory."""
    so, sd, d_so, d_sd, (TE, stE, uE, aE), (TL, stL, uL, aL), tail, clr = ch["legs"]
    m, net, p = Motion(sim, ch), sim.net, sim.p
    B = ch["task"]["B"]
    wl = lambda res: net.demand(res, B, True, p, sim.cap)
    out, keys = [], set()
    end_of = lambda t: m.t_leave if t is None else t
    for run, uses in ((0, uE), (1, uL)):
        off = 0 if run == 0 else m.nE
        for i, (res, d, a, r, dl) in enumerate(uses):
            if d <= EPS or res[0] in ("c", "b"):
                continue
            t_in = m.tl[off + a]
            if r is None:
                # roadside origin stop resource: the body leaves the stop + L + s; destination and off-road
                # stop resources by their holds
                if run == 0 and m.road_o:
                    out.append((res, t_in, end_of(m.reach(1, 0, m.L + m.s)), d_so))
                    keys.add((0, i))
                continue
            keys.add((run, i))
            if res[0] == "n":
                far = m.pos[off + a] + (0.0 if (run == 0 and a == 0) else sim.size[res[1]])
            # last road before a dock-side off-road bay: drives in on arrival
            elif a == r:
                far = m.pos[off + (aE if run == 0 else aL)]
            else:
                far = m.pos[off + r]
            if run == 1:
                out.append((res, t_in, end_of(m.reach(1, a, far + m.L + m.s)), d))
                continue
            if not m.road_o:
                out.append((res, t_in, m.reach(0, a, far + m.L0 + m.s), d))
                continue
            # clearance at length L0 (may reach into the loaded run: the vehicle is L long by then, recompute
            # with L)
            t0 = m.reach(0, a, far + m.L0 + m.s)
            if far + m.L0 + m.s > m.pos[aE] + 1e-9:
                t1 = end_of(m.reach(0, a, far + m.L + m.s))
                out += [(res, t_in, min(t1, m.t_load), d), (res, m.t_load, t1, wl(res))] if t1 > m.t_load else [(res, t_in, t1, d)]
            else:
                out.append((res, t_in, t0, d))
                if m.pos[aE] - m.L - m.s < far - 1e-9:        # the body grows on loading and covers the resource again
                    out.append((res, m.t_load, end_of(m.reach(0, aE, far + m.L + m.s)), wl(res)))
    for j, (res, rem, _) in enumerate(behind(sim, ch["legs"], ch["task"])):   # zero-length empty run: body behind the stop
        out.append((res, m.t_load, end_of(m.reach(1, 0, rem)), wl(res)))
        keys.add(("B", j))
    return out, keys


def body_checks(sim):
    """B1: the body periods are covered by holds of this vehicle on the resource of at least the body width; B2: the
    occupancy rescanned time by time from the body periods. Destination and off-road stop resources, bays, cranes and
    lifts inside the crane span enter B2 by their holds."""
    held = collections.defaultdict(list)
    for b in sim.log_bk:
        held[(b.res, b.vid)].append((b.ta, b.tr, b.d))
    ivs = collections.defaultdict(list)
    spatial = set()
    viol, excess = 0, 0.0
    for ch in sim.chains:
        if ch.get("lift") or "tl" not in ch:
            continue
        items, keys = body_items(sim, ch)
        spatial |= {id(ch["made"][k]) for k in keys if k in ch["made"]}
        vid = ch["v"].id
        for res, s, e, w in items:
            if e <= s + TOL:
                continue
            ivs[res].append((s, e, w, vid))
            cur = s
            for a, b, d in sorted(held[(res, vid)]):
                if d >= w - 1e-9 and a <= cur + TOL and b > cur:
                    cur = b
            if cur < e - TOL:
                viol += 1
                excess = max(excess, e - cur)
    for b in sim.log_bk:
        if id(b) not in spatial:
            ivs[b.res].append((b.ta, b.tr, b.d, b.vid))
    worst = 0.0
    for res, L in ivs.items():
        cap = sim.cap[res]
        for t in sorted({a for a, _, _, _ in L}):
            on = [x for x in L if x[0] <= t + 1e-9 and t < x[1] - 1e-9]
            if res[0] == "c":
                load = len(on)
            else:
                per = collections.defaultdict(float)
                for _, _, d, vid in on:
                    per[vid] = max(per[vid], d)
                load = sum(per.values())
            worst = max(worst, load - cap)
    return dict(body_viol=viol, body_excess_s=excess, body_over=worst)


# ---------------------------------------------------------------- instances
def short_net(seed):
    """Short roads and roadside stops close to junctions: a 3 x 3 grid, spacing between neighbouring nodes drawn from 12,
    18, 25, 60, 120 m, road widths 9-31 m; four shop entrances on connecting roads of 8-20 m; two buffer zones (roadside
    stops) with the stop 5-8 m from the start of the road or 7 m from its end."""
    import helpers as TO
    rng = random.Random(f"short|{seed}")
    xs, ys = [0.0], [0.0]
    for _ in range(2):
        xs.append(xs[-1] + rng.choice((12.0, 18.0, 25.0, 60.0, 120.0)))
        ys.append(ys[-1] + rng.choice((12.0, 18.0, 25.0, 60.0, 120.0)))
    pos = {f"g{i}{j}": (xs[i], ys[j]) for i in range(3) for j in range(3)}
    edges = [(f"g{i}{j}", f"g{i + 1}{j}") for i in range(2) for j in range(3)] + [(f"g{i}{j}", f"g{i}{j + 1}") for i in range(3) for j in range(2)]
    width = {f"r{i}": rng.choice((9.0, 12.0, 15.0, 20.0, 31.0)) for i in range(len(edges))}
    fac, classes = {}, ("assembly_workshop", "paint_workshop", "outfitting_workshop")
    for k, g in enumerate(("g00", "g20", "g02", "g22")):
        x, y = pos[g]
        ln = rng.choice((8.0, 12.0, 20.0))
        pos[f"F{k}"] = (x + (-ln if g[1] == "0" else ln), y)
        width[f"r{len(edges)}"] = 31.0
        edges.append((g, f"F{k}"))
        fac[f"f{k}"] = (f"F{k}", classes[k % 3])
    long_ = [i for i, (a, b) in enumerate(edges[:12]) if abs(pos[a][0] - pos[b][0]) + abs(pos[a][1] - pos[b][1]) >= 25.0]
    zones = {}
    for z, i in zip(("zbuf", "zbuf2"), rng.sample(long_, min(2, len(long_)))):
        a, b = edges[i]
        ln = abs(pos[a][0] - pos[b][0]) + abs(pos[a][1] - pos[b][1])
        zones[z] = ("buffer", f"r{i}", (rng.choice((0.0, 1.0, 3.0, ln - 12.0)),))
    return TO.load_net(TO.make_map(pos, edges, fac, width=width, zones=zones))


def tailx_case():
    """Hand-made counterexample: the wide road r1 (31 m) east of junction n has a roadside stop S 8 m from n. v drives empty
    from B via n to S and loads there with its tail still on n; w loads at A and drives via n and r1 to E1, booked on n
    after v's booking of n and before v's loaded departure, side by side with v on r1."""
    import helpers as TO
    m = TO.make_map(dict(A=(-50, 0), n=(0, 0), B=(0, 200), mm=(150, 0), E1=(170, 0), E2=(150, -20)),
                    [("A", "n"), ("n", "mm"), ("n", "B"), ("mm", "E1"), ("mm", "E2")],
                    {"fA": ("A", "assembly_workshop"), "fB": ("B", "paint_workshop"), "fE1": ("E1", "outfitting_workshop"),
                     "fE2": ("E2", "assembly_workshop")},
                    width={"r0": 31.0, "r1": 31.0, "r2": 31.0, "r3": 31.0, "r4": 31.0}, zones={"zbuf": ("buffer", "r1", (3.0,))})
    net = TO.load_net(m)
    tasks = [dict(TO.task(0, "A", "E1"), B=10.0), dict(TO.task(1, "zbuf#0", "E2"), B=10.0)]
    return net, P.scenario("main", days=1, warmup_days=0), tasks, ["A", "B"]


def short_case(name, seed):
    if name == "tailx":
        return tailx_case()
    import helpers as TO
    net = short_net(int(name[5:]) if name[5:] else seed)
    rng = random.Random(f"{name}|{seed}")
    stops = sorted(net.stops)
    tasks = [dict(TO.task(i, *rng.sample(stops, 2), release=rng.uniform(0, 20000)), B=float(rng.randint(5, 15)), Lv=rng.choice((17.0, 20.0, 30.0)))
             for i in range(50)]
    locs = [net.stops[rng.choice(stops)]["node"] for _ in range(8)]
    return net, P.scenario("main", days=1, warmup_days=0), tasks, locs


# Instances of the independent check (its stops_at, grid_case, case, same_case, copied)
def _stops_at(rng, edges, pos, deltas, kind="buffer", nz=4):
    zones = {}
    cand = list(range(len(edges)))
    rng.shuffle(cand)
    for z, i in zip(range(nz), cand):
        a, b = edges[i]
        ln = math.dist(pos[a], pos[b])
        ss = []
        for _ in range(rng.choice((1, 2))):
            dlt = rng.choice(deltas)
            if dlt >= ln - 1:
                continue
            at = dlt if rng.random() < 0.5 else ln - dlt
            ss.append(round(at - 5.0, 1))
        if ss:
            zones[f"z{z}"] = (kind, f"r{i}", tuple(sorted(set(ss))))
    return zones


def _grid_case(seed, deltas, spacing, widths, K=(5, 9), ntask=50, chain_zero=0.0):
    import helpers as TO
    rng = random.Random(f"grid|{seed}")
    xs, ys = [0.0], [0.0]
    for _ in range(2):
        xs.append(xs[-1] + rng.choice(spacing))
        ys.append(ys[-1] + rng.choice(spacing))
    pos = {f"g{i}{j}": (xs[i], ys[j]) for i in range(3) for j in range(3)}
    edges = [(f"g{i}{j}", f"g{i + 1}{j}") for i in range(2) for j in range(3)] + [(f"g{i}{j}", f"g{i}{j + 1}") for i in range(3) for j in range(2)]
    width = {f"r{i}": rng.choice(widths) for i in range(len(edges))}
    zones = _stops_at(rng, edges, pos, deltas)
    fac, classes = {}, ("assembly_workshop", "paint_workshop", "outfitting_workshop")
    for k, gg in enumerate(("g00", "g20", "g02", "g22")):
        x, y = pos[gg]
        ln = rng.choice((6.0, 12.0, 20.0))
        pos[f"F{k}"] = (x + (-ln if gg[0:2] == "g0" else ln), y)
        width[f"r{len(edges)}"] = rng.choice((12.0, 31.0))
        edges.append((gg, f"F{k}"))
        fac[f"f{k}"] = (f"F{k}", classes[k % 3])
    net = TO.load_net(TO.make_map(pos, edges, fac, width=width, zones=zones))
    stops = sorted(net.stops)
    road = [s for s in stops if net.stops[s]["mode"] == "on_road"] or stops
    tasks, last_d = [], None
    for i in range(ntask):
        if last_d is not None and rng.random() < chain_zero:
            o = last_d
            d = rng.choice([s for s in stops if s != o])
        else:
            o = rng.choice(road if rng.random() < 0.7 else stops)
            d = rng.choice([s for s in stops if s != o])
        last_d = d
        tasks.append(dict(TO.task(i, o, d, release=300.0 * rng.randint(0, 50)), B=float(rng.randint(5, 15)),
                          Lv=rng.choice((16.25, 20.0, 25.0, 30.0))))
    locs = [net.stops[rng.choice(stops)]["node"] for _ in range(rng.randint(*K))]
    return net, P.scenario("main", days=1, warmup_days=0), tasks, locs


def aud_case(name, seed):
    if name == "near":        # stops 2-12 m from a node, short roads
        return _grid_case(seed, (2.0, 4.0, 6.0, 8.0, 12.0), (10.0, 15.0, 22.0, 30.0, 45.0), (9.0, 12.0, 15.0, 20.0, 31.0))
    if name == "band":        # stops between the clearance lengths of the node behind by L0 and by L
        return _grid_case(seed, (22.0, 26.0, 30.0, 33.0, 36.0, 40.0, 44.0), (50.0, 60.0, 80.0), (9.0, 9.0, 12.0, 15.0))
    if name == "mix":
        return _grid_case(seed, (2.0, 6.0, 12.0, 20.0, 28.0, 36.0, 44.0, 60.0), (12.0, 25.0, 40.0, 60.0, 90.0), (15.0, 20.0, 31.0, 31.0))
    if name == "zero":        # chained tasks: zero-length empty runs at roadside origins
        return _grid_case(seed, (4.0, 8.0, 26.0, 33.0), (25.0, 40.0, 60.0), (9.0, 12.0, 31.0), K=(1, 3), chain_zero=0.8)
    raise KeyError(name)


def same_case(seed):
    """Two roadside zones sharing a stop node (tasks between stops on the same node), stops at various distances from the
    junction."""
    import helpers as TO
    rng = random.Random(f"same|{seed}")
    pos = {"n": (0.0, 0.0), "A": (-30.0, 0.0), "m": (rng.choice((40.0, 60.0, 90.0)), 0.0), "B": (0.0, 40.0), "C": (0.0, -40.0)}
    pos["E"] = (pos["m"][0] + 30.0, 0.0)
    edges = [("n", "A"), ("n", "m"), ("n", "B"), ("n", "C"), ("m", "E")]
    width = {f"r{i}": rng.choice((9.0, 12.0, 20.0, 31.0)) for i in range(len(edges))}
    s1 = rng.choice((3.0, 8.0, 20.0, 28.0))
    zones = {"za": ("buffer", "r1", (s1,)), "zb": ("buffer", "r1", (s1,)), "zc": ("buffer", "r1", (pos["m"][0] - 15.0,))}
    fac = {"fA": ("A", "paint_workshop"), "fB": ("B", "assembly_workshop"), "fC": ("C", "outfitting_workshop"), "fE": ("E", "assembly_workshop")}
    net = TO.load_net(TO.make_map(pos, edges, fac, width=width, zones=zones))
    stops = sorted(net.stops)
    pairs = [("za#0", "zb#0"), ("zb#0", "za#0")]
    tasks = []
    for i in range(40):
        o, d = rng.choice(pairs) if rng.random() < 0.4 else rng.sample(stops, 2)
        tasks.append(dict(TO.task(i, o, d, release=300.0 * rng.randint(0, 40)), B=float(rng.randint(5, 15)), Lv=rng.choice((16.25, 20.0, 30.0))))
    locs = [net.stops[rng.choice(stops)]["node"] for _ in range(rng.randint(3, 6))]
    return net, P.scenario("main", days=1, warmup_days=0), tasks, locs


def _uturns(m, nodes):
    """Synthetic maps allow no turning back by default; at these dead-end entrance nodes turning back along the incoming road
    is allowed (as in the two yard maps: turning back only at dead-end entrances)."""
    for n in nodes:
        for rid, r in m["roads"].items():
            for end, d_in, d_out in (("toNodeId", "forward", "backward"), ("fromNodeId", "backward", "forward")):
                if r[end] == n:
                    m["movements"][f"u{n}{rid}"] = dict(allowed=True, junctionId=f"j{n}", incomingArc=dict(roadId=rid, direction=d_in),
                                                       outgoingArc=dict(roadId=rid, direction=d_out))
    return m


def turn_case(seed):
    """Two situations found on the yards:
      turning back: a dead-end road east of junction n (30-90 m long) leads to the dead-end entrance E, with a roadside stop
            close to n or to E; a vehicle loading at the stop heading east must first turn back at E to go elsewhere and
            drives back the way it came, so that its body or its loaded run covers again resources passed on the empty run;
            the dead-end roads west and north to entrances A and B also allow turning back;
      very short empty run from a roadside stop: two stops on the same road 8-15 m apart, tasks mostly starting at a stop
            next to the destination of the previous task."""
    import helpers as TO
    rng = random.Random(f"turn|{seed}")
    Ld = rng.choice((30.0, 45.0, 60.0, 90.0))
    pos = {"n": (0.0, 0.0), "A": (-rng.choice((20.0, 40.0, 60.0)), 0.0), "B": (0.0, rng.choice((25.0, 50.0, 100.0))), "E": (Ld, 0.0),
           "C": (0.0, -rng.choice((30.0, 60.0)))}
    edges = [("n", "A"), ("n", "B"), ("n", "E"), ("n", "C")]
    width = {f"r{i}": rng.choice((9.0, 12.0, 20.0, 31.0)) for i in range(len(edges))}
    s1 = rng.choice((0.0, 3.0, 8.0, max(0.0, Ld - 20.0), max(0.0, Ld - 15.0)))
    s2 = min(Ld - 11.0, s1 + rng.choice((8.0, 12.0, 15.0)))
    zones = {"z1": ("buffer", "r2", (s1,)), "z2": ("buffer", "r2", (s2,)) if s2 > s1 + 1 else ("buffer", "r0", (1.0,)),
             "z3": ("buffer", "r1", (rng.choice((0.0, 5.0)),))}
    fac = {"fA": ("A", "paint_workshop"), "fB": ("B", "assembly_workshop"), "fE": ("E", "outfitting_workshop"), "fC": ("C", "assembly_workshop")}
    net = TO.load_net(_uturns(TO.make_map(pos, edges, fac, width=width, zones=zones), ("A", "B", "E", "C")))
    stops = sorted(net.stops)
    road = [x for x in stops if net.stops[x]["mode"] == "on_road"]
    near = lambda d: [x for x in road if x != d and abs(_spos(net, x) - _spos(net, d)) <= 20.0 and _sroad(net, x) == _sroad(net, d)]
    tasks, last_d = [], None
    for i in range(50):
        if last_d is not None and last_d in road and rng.random() < 0.6 and near(last_d):
            # very short empty run: from the stop next to the last destination
            o = rng.choice(near(last_d))
        elif last_d is not None and rng.random() < 0.3:
            o = last_d                                                # zero-length empty run
        else:
            o = rng.choice(road if rng.random() < 0.7 else stops)
        d = rng.choice([x for x in stops if x != o])
        last_d = d
        tasks.append(dict(TO.task(i, o, d, release=300.0 * rng.randint(0, 50)), B=float(rng.randint(5, 15)), Lv=rng.choice((16.25, 20.0, 25.0, 30.0))))
    locs = [net.stops[rng.choice(stops)]["node"] for _ in range(rng.randint(1, 4))]
    return net, P.scenario("main", days=1, warmup_days=0), tasks, locs


def _sroad(net, stop):
    return net.stops[stop]["node"].split("|")[1] if net.stops[stop]["node"].startswith("S|") else None


def _spos(net, stop):
    n = net.stops[stop]["node"]
    return float(n.split("|")[2]) if n.startswith("S|") else 0.0


SHORT = ["tailx"] + [f"short{i}" for i in range(1, 9)]
AUD = ["near", "band", "mix", "zero"]


def instance(suite, name, seed):
    if suite == "base":
        cs, _ = O.cases()
        net, tasks, locs = cs[name]
        if name.startswith("random"):
            rng = random.Random(1000 * seed + int(name[6:]))
            stops = sorted(net.stops)
            tasks = [dict(t, o=a, d=b, release=rng.uniform(0, 30000)) for t in tasks for a, b in [rng.sample(stops, 2)]]
        return net, P.scenario("main", days=1, warmup_days=0), tasks, locs
    if suite == "adv":
        return C.adv_case(name, seed)
    if suite == "crane":
        return C.crane_case(name, seed)
    if suite == "aud":
        return aud_case(name, seed)
    if suite == "same":
        return same_case(seed)
    if suite == "turn":
        return turn_case(seed)
    return short_case(name, seed)


TIMES = {"base": O.TIMES, "adv": C.ADV_TIMES, "crane": C.ADV_TIMES, "short": C.ADV_TIMES, "aud": C.ADV_TIMES, "same": C.ADV_TIMES,
         "turn": C.ADV_TIMES}


# ---------------------------------------------------------------- one run
def run(net, p, tasks, locs, seed, times, rule, tail):
    if rule == "adm_proj_mean":
        p = dict(p, stoch=C.MEAN_BOOK)
    sim = TailOPE(net, p, len(locs), seed, tasks=[dict(t) for t in tasks], times=times, audit=True, tail=tail, **C.RULES[rule])
    for v, loc in zip(sim.vs, locs):
        v.loc = loc
    out = dict(deadlock=0)
    try:
        sim.run()
    except AssertionError as e:
        if "deadlock" not in str(e):
            raise
        out.update(deadlock=1, stuck=sum(not c["done"] for c in sim.chains))
        out.update(premise_viol=sim.tn["premise_viol"], tail_holds=sim.tn["tail_holds"])
        return out
    out.update(sim.summary(), lifts=len(sim.lifts), over_cap=C.independent_checks(sim), **body_checks(sim),
               premise_viol=sim.tn["premise_viol"], tail_holds=sim.tn["tail_holds"], tail_late_max_s=sim.tail_late_max,
               nominal_late_max_s=sim.late_max if times is None else None,
               zero_road=sum(1 for c in sim.chains if not c.get("lift") and c["legs"][4][3] == 0 and c["legs"][0].get("mode") == "on_road"))
    if times is None and tail == "book" and rule != "adm_proj_mean":
        assert sim.late_max < 1e-6, ("(ii) at nominal times book must keep every booked time", sim.late_max)
    if tail == "book":
        assert out["body_viol"] == 0 and out["body_over"] <= 1e-9, ("body outside the holds", out["body_viol"], out["body_over"])
    return out


def job(args):
    suite, name, seed, times, rule, tail = args
    t0 = time.time()
    out = dict(suite=suite, case=name, seed=seed, times=times, rule=rule, tail=tail)
    try:
        net, p, tasks, locs = instance(suite, name, seed)
        out.update(run(net, p, tasks, locs, seed, TIMES[suite][times], rule, tail))
        out["error"] = ""
    except AssertionError as e:
        out["error"] = repr(e)[:300]
    out["secs"] = round(time.time() - t0, 2)
    return out


def code_sha():
    h = hashlib.sha256()
    for f in sorted((ROOT / "occupancy").glob("*.py")) + [ROOT / "experiments" / n for n in ("ordered.py", "ordered_proj.py", "ordered_check.py", "stochastic.py",
                                                                                           "tail_release.py")]:
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def todo_list(n):
    """Seeds: n each for the instances aud and same of the independent check, 2n for turn, n/2 for short, n/5 each for adv,
    crane and the base self-check (reduced so that the run fits the machine)."""
    ok = lambda tm, r: not (r == "adm_proj_mean" and tm == "nominal")
    rules = ("adm", "rel", "adm_proj", "adm_proj_mean")
    seeds = lambda k: range(1, max(1, k) + 1)
    out = [("aud", c, s, tm, r, m) for c in AUD for s in seeds(n) for tm in C.ADV_TIMES for r in rules for m in MODES if ok(tm, r)]
    out += [("same", "same", s, tm, r, m) for s in seeds(n) for tm in C.ADV_TIMES for r in rules for m in MODES if ok(tm, r)]
    out += [("turn", "turn", s, tm, r, m) for s in seeds(2 * n) for tm in C.ADV_TIMES for r in rules for m in MODES if ok(tm, r)]
    out += [("short", c, s, tm, r, m) for c in SHORT for s in seeds(n // 2) for tm in C.ADV_TIMES for r in rules for m in MODES if ok(tm, r)]
    out += [("adv", c, s, tm, r, m) for c in C.ADV for s in seeds(n // 5) for tm in C.ADV_TIMES for r in rules for m in MODES if ok(tm, r)]
    out += [("crane", c, s, tm, r, m) for c in C.CRANE + C.LIFT for s in seeds(n // 5) for tm in C.ADV_TIMES for r in rules for m in MODES
            if ok(tm, r)]
    out += [("base", c, s, tm, r, m) for c in O.cases()[0] for s in seeds(n // 5) for tm in O.TIMES for r in rules for m in MODES if ok(tm, r)]
    return out


def write_rows(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def main():
    """--suites a,b: compute only these suites, each written to cases_<suite>.csv (suites already written are skipped, so a
    run can be resumed); once all are written they are collected into cases.csv, summary.json and tables.md."""
    ap = argparse.ArgumentParser(description="Release at the later of timer and tail clearance: self-check.")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--slots", default=os.environ.get("CPU_SLOTS"), help="CPU-slot directory (optional; default $CPU_SLOTS)")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--suites", default=None)
    a = ap.parse_args()
    todo = todo_list(a.seeds)
    suites = list(dict.fromkeys(x[0] for x in todo))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for su in (a.suites.split(",") if a.suites else suites):
        part = out / f"cases_{su}.csv"
        if part.exists():
            continue
        mine = [x for x in todo if x[0] == su]
        print(f"suite {su}: {len(mine)} runs", flush=True)
        write_rows(part, O.pool_map(job, mine, a.procs, a.slots, "tail_release"))
    if not all((out / f"cases_{su}.csv").exists() for su in suites):
        print("suites not yet computed:", [su for su in suites if not (out / f"cases_{su}.csv").exists()])
        return
    rows = []
    for su in suites:
        for r in csv.DictReader(open(out / f"cases_{su}.csv", encoding="utf-8-sig")):
            for k, v in r.items():
                try:
                    r[k] = float(v) if v not in ("", None) and k not in ("suite", "case", "times", "rule", "tail", "error") else v
                except ValueError:
                    pass
            rows.append(r)
    write_rows(out / "cases.csv", rows)
    agg = collections.defaultdict(collections.Counter)
    for r in rows:
        g = agg[(r["suite"], r["tail"], r["rule"])]
        g["runs"] += 1
        g["errors"] += bool(r["error"])
        for k in ("tasks", "deadlock", "wait_order", "wait_cap", "wait_rel", "tail_holds", "premise_viol", "body_viol", "lifts", "zero_road"):
            g[k] += r.get(k, 0) or 0
        g["runs_body_viol"] += bool(r.get("body_viol"))
        g["runs_body_over"] += (r.get("body_over") or 0) > 1e-9
        g["nominal_late_runs"] += (r.get("nominal_late_max_s") or 0) > 1e-6
        for k in ("body_over", "body_excess_s", "over_cap", "tail_late_max_s"):
            g[k + "_max"] = max(g[k + "_max"], r.get(k) or 0.0)
    summ = dict(code_sha=code_sha(), runs=len(rows), errors=[r for r in rows if r["error"]][:20],
                groups={"|".join(k): dict(v) for k, v in sorted(agg.items())})
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    L = ["| suite | release | rule | runs | tasks | deadlocks | assertion failures | (B5) violated | release held for the tail | "
         "body not covered by holds (times / runs) | longest (s) | body rescan over capacity (runs / max m) | holds over capacity | "
         "runs late at nominal times |", "|---" * 14 + "|"]
    for (s, m, r), g in sorted(agg.items()):
        L.append(f"| {s} | {m} | {r} | {g['runs']} | {g['tasks']} | {g['deadlock']} | {g['errors']} | {g['premise_viol']} | {g['tail_holds']} | "
                 f"{g['body_viol']} / {g['runs_body_viol']} | {g['body_excess_s_max']:.0f} | {g['runs_body_over']} / {g['body_over_max']:.1f} | "
                 f"{g['over_cap_max']:.1f} | {g['nominal_late_runs']} |")
    (out / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(L))
    print(out)


if __name__ == "__main__":
    main()

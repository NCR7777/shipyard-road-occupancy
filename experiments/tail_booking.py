"""Full-path reservation booking the tail, with the body growing from the empty length L0 to the loaded length L when loading.

Rule names reserve_tlw (booking the tail) and reserve_tlw_timer (the current rule, with the readings counted by this
definition). Serves the paper's full-yard comparison of release by timer with booking the tail.
On the side of a roadside origin (the destination side is the plain tail booking):
  (1) Growing body: every booking of the empty run closer than l^c(L) to the roadside origin (road: L + s; junction /
      entrance: D_n + L + s) is cleared only on the loaded run, including those already cleared by L0 before arrival
      that the body covers again when it grows on loading; such a booking is held continuously from its acquisition until
      the tail has cleared it by L after loading.
  (2) Width: these extended bookings of the empty run are booked over their whole period at max(empty width, loaded
      width) (the vehicle carries the block from the start of loading).
  (3) Behind: with an empty run of length zero, loading directly at the roadside origin, the body is taken to lie on the
      side opposite to the loaded departure: walking back from the stop against the departure direction (straight on
      through stops and section boundaries; at junctions the incoming road with the smallest turn; behind an entrance
      lies the shop, stop), the junctions / entrances and roads within l^c(L) behind are booked from loading (decision
      point 0) with the loaded width until the tail has cleared them. Likewise when the empty run is very short, starts
      at a roadside stop (the roadside destination of the previous task) and the grown body reaches past the start of the
      empty run: walking back from that start against the empty-run direction (counted separately).
  (4) Roadside destination: resources not yet cleared by the tail are booked until unloading is done and the vehicle has
      left.
  Bridging after turning back (bridge): when the loaded run acquires a resource again no later than the front returns to
  its releasing node, the booking is extended until that acquisition.
  bridge=False, ahead=False is the plain booking of the tail with loaded length and width; bridge=False, ahead="zero"
  corresponds to the book mode of tail_release.py (compared one by one at nominal times in the self-check).
Readings (metrics()["tail"]): tasks (dispatched in the statistics window), tail_tasks (tasks with an extended booking),
res (extended bookings) split into res_o / res_d (origin / destination side) and stop / node / road / other (kind of
resource), hold_s (seconds held beyond timer expiry), rev (tasks whose loaded run leaves back along the last road of the
empty run), bridge (bridged bookings), regrow (bookings of (1) already cleared by L0 before arrival), behind_zero /
behind_short (bookings of the two cases of (3)), zero (tasks with a zero-length empty run at a roadside origin).

Import only (no command line): runner.compute runs it for the rules reserve_tlw and reserve_tlw_timer (yard_sim)."""
import collections
import hashlib
from pathlib import Path

from occupancy import sim as S
from occupancy.network import EPS

TOL = 1e-9
_PRED = {}


def ins(net):
    """Incoming arcs of every node, computed once per network."""
    if id(net) not in _PRED:
        out = collections.defaultdict(list)
        for i, a in enumerate(net.arc):
            out[a[1]].append(i)
        _PRED[id(net)] = (net, out)
    return _PRED[id(net)][1]


def behind(net, cap, node, road, fw, x0, reach):
    """Walks back from node, starting with the arc into node on the same graph road and in the same direction as (road
    road, direction fw): [(resource, distance of its releasing node from the stop)]. x0 = distance of node from the stop;
    the road node lies on itself (the stop resource or the first part of the empty run) is not listed. Conventions as
    behind of tail_release.py: at a node the incoming arc with the smallest turn other than turning back (equal angles: the
    smaller arc number), at stops and section boundaries only along the same graph road in the same direction, every
    resource listed once; stops where there is no incoming arc (behind an entrance lies the shop) or beyond reach."""
    orig = lambda a: net.road[net.arc[a][3]]["orig"]
    o = net.road[road]["orig"]
    into = ins(net)
    cur = min((j for j in into[node] if orig(j) == o and net.arc[j][4] == fw), default=None)
    out, x, seen = [], x0, {("r", road)}
    while cur is not None and x < reach:
        u, h, ln, rid, f = net.arc[cur]
        if ("r", rid) not in seen:
            out.append((("r", rid), x))
            seen.add(("r", rid))
        x += ln
        if ("n", u) in cap and ("n", u) not in seen:
            out.append((("n", u), x))
            seen.add(("n", u))
        prev = [b for b in into[u] if not (net.arc[b][0] == h and net.arc[b][3] == rid)]
        if u.startswith("S|"):
            prev = [b for b in prev if orig(b) == orig(cur) and net.arc[b][4] == f]
        # turn angle as stored per movement (Net.turn_deg); a stop node has no movement and a single candidate: 0
        cur = min(prev, key=lambda b: (0.0 if u.startswith("S|") else net.turn_deg(b, cur), b)) if prev else None
    return out


def tail_lw(sim, legs, task, bridge=True, ahead=True):
    """The bookings that booking the tail extends: (OW items [(key, resource, booked start (relative to t1), booked end
    (relative to t2), width)], destination side {(1, i): booked end (relative to t2)}, booked end of the origin stop
    resource (relative to t2) or None, readings Counter). Key (0, i) is the index in the uses of the empty run, ("b",
    resource) a resource behind. ahead: behind ((3)) - True for zero-length and short empty runs, "zero" for zero-length
    empty runs only (= the book mode of tail_release.py), False for none (= the plain booking of the tail)."""
    p, net = sim.p, sim.net
    so, sd, d_so, d_sd, (TE, stE, uE, aE), (TL, stL, uL, aL), tail_l, clr = legs
    vE, vL, s, L0, Lv = p["v_empty"], p["v_loaded"], p["long_clear"], p["plat_len"], task["Lv"]
    DE, DL = stE[aE] * vE, stL[aL] * vL
    road_o, road_d = so["mode"] == "on_road", sd["mode"] == "on_road"
    leave = TL + p["unload_s"] + clr
    dload = lambda res: net.demand(res, task["B"], True, p, sim.cap)
    n = collections.Counter()

    def lam(res, dl, v, L):
        return min(dl * v, sim.size[res[1]] + L + s) if res[0] == "n" else dl * v

    def after_t2(rem):
        return leave if rem > DL + TOL and road_d else rem / vL

    back = {}
    for res, d, a, r, dl in uL:
        back.setdefault(res, (stL[a] * vL, stL[a]))

    def end(res, over, dist):
        """The tail clears after another `over` on the loaded run; bridged when the vehicle turns back and the loaded run
        acquires the resource again no later than the front returns to its releasing node (dist from the stop)."""
        e = after_t2(over)
        if bridge and res in back and back[res][0] <= dist + 1e-6 and back[res][1] > e + EPS:
            n["bridge"] += 1
            return back[res][1]
        return e

    ow = []
    if road_o:
        for i, (res, d, a, r, dl) in enumerate(uE):
            if res[0] == "c" or d <= EPS or r is None:
                continue
            y = stE[r] * vE
            # by L0: clearance still missing when the front stops at the origin
            over0 = y + lam(res, dl, vE, L0) - DE
            if over0 + Lv - L0 > TOL:                              # by L: the body grows on loading and still covers it
                n["regrow"] += over0 <= TOL
                ow.append(((0, i), res, stE[a], end(res, over0 + Lv - L0, DE - y), max(d, dload(res))))
        node = None
        # loading directly at the roadside origin: body on the side opposite to the departure
        if ahead and aE == 0:
            a1 = net.arc[net.path(so["node"], sd["node"])[0]] if aL else None
            rid, fw = (a1[3], a1[4]) if a1 else (so["res"][1], True)
            node, x0, tag = so["node"], 0.0, "behind_zero"
            n["zero"] += 1
        # empty run from a roadside stop: the grown body may reach past the start
        elif ahead is True:
            st = task.get("start")
            if any(x["node"] == st and x["mode"] == "on_road" for x in net.stops.values()):
                e1 = net.arc[net.path(st, so["node"])[0]]
                node, rid, fw, x0, tag = st, e1[3], e1[4], DE, "behind_short"
        if node is not None:
            reach = Lv + s + max(sim.size.values())
            for res, dist in behind(net, sim.cap, node, rid, fw, x0, reach):
                over = (sim.size[res[1]] if res[0] == "n" else 0.0) + Lv + s - dist
                if over > TOL:
                    n[tag] += 1
                    ow.append((("b", res), res, TE, end(res, over, dist), dload(res)))
    dest = {}
    for i, (res, d, a, r, dl) in enumerate(uL):
        if res[0] == "c" or d <= EPS or r is None:
            continue
        if stL[r] * vL + lam(res, dl, vL, Lv) - DL > TOL and road_d:
            dest[(1, i)] = leave
    origin = leave if Lv + s > DL + TOL and road_d else None     # origin stop resource: L + s from the loaded departure
    return ow, dest, origin, n


class LWSim(S.Sim):
    """Full-path reservation + booking the tail. tail = "book": book the tail; "timer": the current rule (bookings identical
    one by one to reserve of Sim), only counted by this definition."""

    def __init__(self, *a, tail="book", bridge=True, ahead=True, **k):
        assert tail in ("book", "timer")
        self.tail, self.bridge, self.ahead = tail, bridge, ahead
        self.tn = collections.Counter()
        super().__init__(*a, **k)

    def plan(self, v, task, t0):
        """The windows OW start relative to t1 and end relative to t2; they are checked first with t2 = arrival + loading
        and, once the loaded run is booked, again with the actual t2, shifting everything later on a conflict; just-in-time
        departure checks only WE (the windows of OW are subsets of windows already feasible). With no extension it is step
        by step the same as Sim.plan."""
        assert self.mode == "reserve" and not self.net.wait_at, "booking the tail: only full-path reservation without dock-side waiting places"
        legs = self.legs(v, task)
        so, sd, d_so, d_sd, (TE, stE, uE, _), (TL, stL, uL, _), tail_l, clr = legs
        ow, dest, origin, info = tail_lw(self, legs, task, self.bridge, self.ahead)
        book = self.tail == "book"
        p = self.p
        load, unload = p["load_s"], p["unload_s"]
        ao = next(stE[a] for _, _, a, r, _ in uE if r is None)
        ad = next(stL[a] for _, _, a, r, _ in uL if r is None)
        ext0 = {k for k, *_ in ow} if book else set()
        WE = [(res, stE[a], stE[r] + dl, d) for i, (res, d, a, r, dl) in enumerate(uE) if r is not None and (0, i) not in ext0]
        OW = [(so["res"], ao, origin if book and origin is not None else tail_l, d_so)] + \
            ([(res, s0, e, d) for _, res, s0, e, d in ow] if book else [])
        WL = [(res, stL[a], dest[(1, i)] if book and (1, i) in dest else stL[r] + dl, d) for i, (res, d, a, r, dl) in enumerate(uL) if r is not None]
        WL += [(res, TL, TL + p["crane_cycle_s"], d) for res, d, a, r, dl in uL if res[0] == "c"]
        tmin, jumps = t0, []
        while True:
            jL = []
            t1 = self.earliest(WE + [(r, s, TE + load + e, d) for r, s, e, d in OW], tmin, v.id, False, jumps)
            a = t1 + TE
            t2, tc, WLa, bay = self.book_l(sd, d_sd, TL, WL, ad, clr, a + load, v.id, False, jL)
            hit = next(((r, s, o2) for r, s, e, d in OW if (o2 := self.conflict(r, t1 + s, t2 + e, d, v.id)) is not None), None)
            if hit is None:
                break
            r, s, o2 = hit
            jumps.append((r, o2 - s - t1))
            tmin = o2 - s
        jumps += jL
        if p["reserve_jit"]:
            tj = t2 - load - TE
            if tj > t1 + EPS and all(self.conflict(r, tj + s, tj + e, d, v.id) is None for r, s, e, d in WE):
                t1, a = tj, tj + TE
        W = [(r, t1 + s, t1 + e, d) for r, s, e, d in WE] + [(r, t1 + s, t2 + e, d) for r, s, e, d in OW] + WLa
        self.commit(v.id, t0, W)
        self.account(W)
        if self.in_win(t0):
            for res, x in jumps:
                self.wait_by[res] += x
            self.count(legs, task, ow, dest, origin, info, t1, t2)
        # vid: lets a check verify the body per vehicle; not a reading
        task["plan"] = dict(start=v.loc, t1=t1, a=a, t2=t2, vid=v.id)
        if bay:
            task["plan"]["tc"] = tc
        if any(r[0] == "c" for r, *_ in WL):
            task["crane_start"] = tc
        if task.get("flow") == "P6":
            task["blk"] = [(s, e) for s, e in ((t0, t1), (a + load, t2)) + ((bay,) if bay else ()) if e > s + EPS]
            task["blk_road"] = self.split_blk(task["blk"], jumps)
        v.loc = sd["node"]
        self.push(tc + unload, "done", (v.id, task, max(0.0, (t1 - t0) + (t2 - a - load) + (tc - t2 - TL if bay else 0.0))))

    def count(self, legs, task, ow, dest, origin, info, t1, t2):
        """Readings (see the module docstring). Seconds extended = tail clearance - timer expiry (for resources behind,
        timer expiry is taken as the start of loading, i.e. no booking under the current rule)."""
        so, sd, d_so, d_sd, (TE, stE, uE, aE), (TL, stL, uL, aL), tail_l, clr = legs
        n = self.tn
        n["tasks"] += 1
        n.update(info)
        items = [(k, res, (t1 + stE[uE[k[1]][3]] + uE[k[1]][4]) if k[0] == 0 else t1 + TE, e) for k, res, s0, e, d in ow] + \
            [((1, i), uL[i][0], t2 + stL[uL[i][3]] + uL[i][4], e) for (_, i), e in dest.items()] + \
            ([(("o",), so["res"], t2 + tail_l, origin)] if origin is not None else [])
        if not items:
            return
        n["tail_tasks"] += 1
        for k, res, timer, e in items:
            assert t2 + e >= timer - EPS, "tail clearance before timer expiry: the booked end must be the later of the two"
            n["res"] += 1
            n["res_d" if k[0] == 1 else "res_o"] += 1
            n["stop" if k == ("o",) else {"n": "node", "r": "road"}.get(res[0], "other")] += 1
            n["hold_s"] += t2 + e - timer
        if any(k[0] == 0 for k, *_ in ow) and aE >= 1 and aL >= 1:
            net = self.net
            lastE, firstL = net.arc[net.path(task["start"], so["node"])[-1]], net.arc[net.path(so["node"], sd["node"])[0]]
            n["rev"] += lastE[3] == firstL[3] and lastE[4] != firstL[4]

    def metrics(self):
        return dict(super().metrics(), tail=dict(self.tn, mode=self.tail))


def yard_sim(net, p, mode, K, seed, level_n, record=False, tasks=None):
    """Sim interface of runner.compute: reserve_tlw = booking the tail (growing body + bridging + behind); reserve_tlw_timer
    = current full-path reservation, counted by this definition."""
    return LWSim(net, p, "reserve", K, seed, level_n, record, tasks, tail={"reserve_tlw": "book", "reserve_tlw_timer": "timer"}[mode])


def code_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

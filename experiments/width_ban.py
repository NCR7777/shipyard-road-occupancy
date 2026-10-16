"""Ban on blocks wider than the passable width (supplementary material, sensitivity to the inputs: the row "ban on blocks wider
than the passable width"; the check of the clear width beside roads in the table of inputs).

Eq. (1) caps the occupied width at the resource capacity, d = min(B + c, C_r) with C_r = W_r + c: a loaded vehicle whose block
is wider than the road (B > W_r) still enters road r and only holds it alone (the implicit assumption: the block overhangs the
kerb; roadside obstacles are not in the model). This module adds a scenario parameter width_ban (absent by default; without it
every point takes the ordinary code, bit for bit unchanged). With it, a loaded vehicle may not enter a graph road whose passable
width G_r is less than its block width B:
  width_ban = {"clear": "road" | {graph road: G}, "infeasible": "redraw" | "clamp"}
    clear = "road": G_r = W_r + width_delta (the mapped road width: every passage that Eq. (1) caps is banned); a dict gives the
    passable widths read from imagery for the roads it lists (the stored configurations inline this table), the other roads
    as "road". Roads without a width limit (entrance access roads), junctions, doors and stops are not banned; empty runs are
    not affected. The loaded run takes the shortest path that avoids the banned roads and respects the turn permissions
    (BanNet.leg).
    Tasks without a feasible route (the stop of the origin or destination lies on a banned road, or banned roads separate them):
      redraw  the task is removed from the task stream and drawn again (the whole task, i.e. the sampling law conditioned on a
              feasible route; the number of tasks per day is unchanged);
      clamp   the block width is cut to the largest that a feasible route between the origin and destination allows (the
              bottleneck passable width of the widest feasible route, BanNet.bmax); the random numbers drawn are unchanged.
occupancy/ and bounds/ are not changed: the ban enters through the subclasses here. The simulation covers only the main mix
(no crane pacing, no concentrated erection; asserted); the system bound (ban_mix, BanSystemT1) covers all three mixes.

Run:   python experiments/required_fleet.py <config.json> --out <dir>     (or runner.py): runner.compute passes the points of
scenarios with width_ban to compute() here.
Inputs: the network of the configuration (data/networks/<yard>/<revision>: road widths and turn permissions per graph road) and
width_ban in the scenario parameters. Output: the point of runner.compute plus ban = dict(infeasible = tasks drawn without a
feasible route, infeasible_per_day, tasks).
"""
import collections
import heapq
import math

import numpy as np

from bounds.system import Loads, SystemT1, scenario_mix, t1_times
from occupancy.network import EPS, Net
from occupancy.tasks import TaskStream, releases


class BanNet(Net):
    """Net with the ban on loaded runs: G[graph road] = passable width; the loaded runs of leg with block width B use only the
    roads with G >= B. A B without banned roads (ban_of empty) uses the shortest-path trees of Net unchanged."""

    def __init__(self, G, p, roles=None):
        super().__init__(G, p, roles)
        assert not self.retro, "the ban is not combined with road changes (passing places have wider stretches)"
        clear = p["width_ban"]["clear"]
        clear = {} if clear == "road" else clear
        self.G = {}
        for r in self.road.values():
            if r["W"] is not None:
                self.G[r["orig"]] = float(clear.get(r["orig"], r["W"] + p["width_delta"]))
        assert set(clear) <= set(self.G), set(clear) - set(self.G)
        self.orig_of = [self.road[a[3]]["orig"] for a in self.arc]
        self.p_w = p["plat_w"]
        self._B, self._bans, self._btrees = None, {}, {}

    def ban_of(self, B):
        """Graph roads that a block of width B may not enter (G < B)."""
        if B is None:
            return frozenset()
        if B not in self._bans:
            self._bans[B] = frozenset(o for o, g in self.G.items() if B > g + EPS)
        return self._bans[B]

    def tree(self, src):
        ban = self.ban_of(self._B)
        if not ban:
            return super().tree(src)
        key = (src, ban)
        if key in self._btrees:
            return self._btrees[key]
        ok = lambda a: self.orig_of[a] not in ban
        dist, pred = {}, {}                       # as Net.tree, skipping the arcs on banned roads
        heap = [(self.arc[a][2], a, -1) for a in self.out[src] if ok(a)]
        heapq.heapify(heap)
        while heap:
            d, a, q = heapq.heappop(heap)
            if a in dist:
                continue
            dist[a], pred[a] = d, q
            for b in self.succ[a]:
                if b not in dist and ok(b):
                    heapq.heappush(heap, (d + self.arc[b][2], b, a))
        best = {src: (0.0, -1)}
        for a, d in dist.items():
            h = self.arc[a][1]
            if d < best.get(h, (math.inf,))[0]:
                best[h] = (d, a)
        self._btrees[key] = (best, pred)
        return best, pred

    def with_B(self, B, f, *a):
        self._B = B
        try:
            return f(*a)
        finally:
            self._B = None

    def leg(self, src, dst, v, Lv, b, loaded, *a, **kw):
        return self.with_B(b if loaded else None, super().leg, src, dst, v, Lv, b, loaded, *a, **kw)

    def reach(self, src, dst, B):
        return dst in self.with_B(B, self.tree, src)[0]

    def route(self, src, dst, B):
        return tuple(self.with_B(B, self.path, src, dst))

    def bmax(self, src, dst, B):
        """The largest width <= B of a load that can go from src to dst: feasibility is monotone in the width and changes only
        at values of G, so the largest feasible value of G is taken; None when none of them, from the empty vehicle's width
        (platform width) up, is feasible."""
        if self.reach(src, dst, B):
            return B
        return next((g for g in sorted({g for g in self.G.values() if self.p_w <= g < B}, reverse=True) if self.reach(src, dst, g)), None)


def stop_nodes(net, t):
    return net.stops[t["o"]]["node"], net.stops[t["d"]]["node"]


class BanStream(TaskStream):
    """TaskStream with the handling of tasks without a feasible route (module docstring). infeasible: number of draws without a
    feasible route (each redraw or clamp counts once)."""

    def __init__(self, net, seed, p):
        super().__init__(net, seed, p)
        self.infeasible = 0

    def next(self, release):
        net = self.net
        while True:
            t = super().next(release)
            so, sd = stop_nodes(net, t)
            if t.get("lift") or net.reach(so, sd, t["B"]):
                return t
            self.infeasible += 1
            if self.p["width_ban"]["infeasible"] == "clamp":
                b = net.bmax(so, sd, t["B"])
                if b is None:
                    raise RuntimeError(f"from {t['o']} to {t['d']} no route even for a load of the empty vehicle's width")
                t["B_drawn"], t["B"] = t["B"], b
                return t
            self.n -= 1                               # redraw: the task numbers do not skip


def ban_tasks(net, seed, level_n, p):
    """Task stream of a finite demand (the branch of tasks.demand_tasks without crane pacing and concentrated erection), tasks
    drawn by BanStream. Returns (tasks, number of draws without a feasible route)."""
    assert not p.get("crane_n") and not p.get("erect_batches") and level_n is not None
    s = BanStream(net, seed, p)
    tasks = [s.next(r) for r in releases(level_n, p)]
    return sorted(tasks, key=lambda t: (t["release"], t["id"])), s.infeasible


NETS = {}


def compute(cfg, key):
    """One simulated point (runner.compute passes the points of scenarios with width_ban here): finite demand, main mix; the
    per-point T1' as in runner.compute."""
    import runner as F
    from occupancy.sim import Sim
    scen, model, lvl, K, seed = key.split("|")
    K, seed, p = int(K), int(seed), F.scen_params(cfg, scen)
    k = (cfg["network"], scen)
    if k not in NETS:
        NETS[k] = BanNet(cfg["network"], p, cfg.get("roles"))
    net = NETS[k]
    tasks, n_inf = ban_tasks(net, seed, cfg["demand"][lvl], p)
    sim = Sim(net, p, model, K, seed, cfg["demand"][lvl], tasks=tasks).run()
    res = dict(key=key, scen=scen, model=model, level=lvl, K=K, seed=seed, **sim.metrics(),
               ban=dict(infeasible=n_inf, infeasible_per_day=n_inf / p["days"], tasks=len(tasks)))
    if model != "free":
        res["T1_tight_mix"] = F.t1_window(net, p, sim, sim.H)
    return res


# ---------------------------------------------------------------- system T1' (mix and times of bounds.system, routes by B)
def ban_mix(net, p):
    """(mix, fixed part): bounds.system.scenario_mix with the sampling law under the ban. clamp: in the (B, Lv) distribution of
    each task the infeasible B are replaced by bmax; redraw: the infeasible (B, Lv) are removed, the task weight is multiplied
    by the feasible probability, and the weights are normalised again over the whole mix (in the fixed part of crane pacing
    within each dock, so the shares of the docks are unchanged)."""
    mix, fixed = scenario_mix(net, p)
    clamp = p["width_ban"]["infeasible"] == "clamp"

    def cond(rows):
        out = []
        for w, t, bl in rows:
            if t.get("lift"):
                out.append((w, t, bl))
                continue
            so, sd = stop_nodes(net, t)
            ok = [net.reach(so, sd, B) for (B, _), _ in bl]
            if all(ok):
                out.append((w, t, bl))
            elif clamp:
                nb = collections.Counter()
                for ((B, Lv), q), y in zip(bl, ok):
                    nb[(B if y else net.bmax(so, sd, B), Lv)] += q
                assert None not in {B for B, _ in nb}, (t["o"], t["d"])
                out.append((w, t, tuple(sorted(nb.items()))))
            else:
                keep = tuple(x for x, y in zip(bl, ok) if y)
                f = sum(q for _, q in keep)
                if f > 0:
                    out.append((w * f / sum(q for _, q in bl), t, tuple((k, q / f) for k, q in keep)))
        return out

    def norm(rows, groups):
        tot0, tot1 = collections.Counter(), collections.Counter()
        for w, t, _ in rows[0]:
            tot0[groups(t)] += w
        for w, t, _ in rows[1]:
            tot1[groups(t)] += w
        s = {g: tot0[g] / tot1[g] for g in tot1}
        return [(w * s[groups(t)], t, bl) for w, t, bl in rows[1]]

    m2 = cond(mix)
    if not clamp:
        m2 = norm((mix, m2), lambda t: 0)
    f2 = None
    if fixed:
        f2 = cond(fixed[0])
        if not clamp:
            f2 = norm((fixed[0], f2), lambda t: net.stops[t["d"]]["ent"])
        f2 = (f2, fixed[1])
    return m2, f2


class BanLoads(Loads):
    """bounds.system.Loads, except that the (B, probability) pairs of a task and Lv are grouped by loaded route: one call of
    times per group (the route is fixed by the B of the group, the same within it), key (bl, Lv, the B of the group). Without a
    ban each (bl, Lv) has one group, the summation order is that of Loads and the result is identical to the last bit."""

    def __init__(self, net, p, mix, times, res_list):
        self.net, self.p, self.res_list = net, p, res_list
        ri = {r: i for i, r in enumerate(res_list)}
        self.nodes = sorted({x for a in net.arc for x in a[:2]})
        ni = {n: i for i, n in enumerate(self.nodes)}
        self.V, self.PB = collections.defaultdict(lambda: np.zeros(len(res_list))), {}
        self.on = np.zeros(len(self.nodes))
        for w, t, bl in mix:
            so, sd = stop_nodes(net, t)
            for Lv in sorted({lv for (_, lv), _ in bl}):
                groups = collections.defaultdict(list)
                for (B, lv), q in bl:
                    if lv == Lv:
                        groups[None if t.get("lift") else net.route(so, sd, B)].append((B, q))
                for g in groups.values():
                    key = (bl, Lv, tuple(B for B, _ in g))
                    if key not in self.PB:
                        self.PB[key] = g
                    x = self.V[key]
                    for r, s in times(dict(t, B=g[0][0], Lv=Lv)).items():
                        x[ri[r]] += w * s
            if not t.get("lift"):
                self.on[ni[so]] += w
                self.on[ni[sd]] -= w


class BanSystemT1(SystemT1):
    """SystemT1 with BanLoads (the mix from ban_mix)."""

    def __init__(self, net, p, mix, fixed=None):
        self.net, self.p = net, p
        self.cap, self.size = net.capacities(p)
        self.res_list = sorted(self.cap)
        f = t1_times(net, p, self.cap, self.size)
        self.L = BanLoads(net, p, mix, f, self.res_list)
        self.L6 = BanLoads(net, p, fixed[0], f, self.res_list) if fixed else None
        self.n6 = fixed[1] if fixed else None
        w, t, bl = next(x for x in mix if not x[1].get("lift"))
        self.dummy = dict(t, B=bl[0][0][0], Lv=bl[0][0][1])

"""Shared helpers of the tests: a simulator subclass that checks capacities event by event, small synthetic networks written
as maps (converted with occupancy.mapgraph.to_graph), the yard networks of data/networks, and the scenario parameters of
crane-paced erection used by the stored runs."""
import collections
import functools
import json
import math
import random
from pathlib import Path

from occupancy import params as P
from occupancy.mapgraph import to_graph
from occupancy.network import EPS, Net
from occupancy.sim import Sim as _Sim

ROOT = Path(__file__).resolve().parents[1]
V6, V12 = 6 / 3.6, 12 / 3.6
H = 57600.0                       # 16-h working day (s)


class Sim(_Sim):
    """Simulator that, for the stepwise family with record=True, checks the capacity after every acquisition and release:
    the occupancy of the resource recomputed from the widths the vehicles currently hold must equal `used` and stay within
    the capacity, and a vehicle holding the resource alone (width = capacity) leaves room for no other. (The log of the
    stepwise family records one entry per release as [acquired, released] x the largest width held, so checking it time by
    time could report a false overload when a vehicle widens its hold; check_capacity uses these counts instead.)"""
    exact_n = excl_n = 0

    def _exact(self, res):
        if self.log is None or not self.seg or res[0] == "c":
            return
        hs = [u.hold[res][1] for u in self.vs if res in u.hold]
        tot, C = sum(hs), self.cap[res]
        assert abs(tot - self.used[res]) < 1e-6 and tot <= C + 1e-6, (res, tot, self.used[res], C)
        if any(d >= C - EPS for d in hs):
            assert sum(d > EPS for d in hs) == 1, (res, hs)
            self.excl_n += 1
        self.exact_n += 1

    def acquire(self, v, res, d, t):
        super().acquire(v, res, d, t)
        self._exact(res)

    def release(self, v, res, t, all_=False):
        super().release(v, res, t, all_)
        self._exact(res)


def short(scen="main", days=1, warm=0, **kw):
    return P.scenario(scen, days=days, warmup_days=warm, **kw)


def check_capacity(sim):
    """Checks the actual occupancy time by time: the width of a vehicle on a resource at a time is the largest of its windows
    covering that time; the sum over vehicles stays within the capacity, and a vehicle holding a resource alone leaves room
    for no other. Returns the number of exclusive holds checked. For the stepwise family the check was done event by event
    during the run (Sim._exact); here it is only confirmed that it ran, and the number of exclusive events is returned."""
    if sim.seg:
        assert isinstance(sim, Sim) and sim.exact_n > 0, "stepwise rules need the checking Sim of the tests and record=True"
        return sim.excl_n
    by = collections.defaultdict(list)
    for res, s, e, d, vid in sim.log:
        by[res].append((s, e, d, vid))
    n_excl = 0
    for res, ws in by.items():
        C = sim.cap[res]
        for t in sorted({s for s, *_ in ws}):
            per = collections.defaultdict(float)
            for s, e, d, vid in ws:
                if s <= t < e - EPS:
                    per[vid] = max(per[vid], d)
            assert sum(per.values()) <= C + 1e-6, (res, t, dict(per))
            for vid, d in per.items():
                if d >= C - EPS:
                    n_excl += 1
                    assert all(w <= EPS for k, w in per.items() if k != vid), (res, t, dict(per))
    return n_excl


# ---------------------------------------------------------------- synthetic networks
def make_map(pos, edges, fac, width=9.0, zones=None):
    """Synthetic map: pos = node positions, edges = roads (end nodes; named r0, r1, ... in order), fac = {building: (entrance
    node, class)}; width = one width for all roads or {road: width}; zones = {zone: (class, road, [start chainages of the
    frontage stretches])}, stretches 10 m long. Junctions allow every turn except U-turns."""
    access = {n for n, _ in fac.values()}
    nodes = {k: dict(kind="access" if k in access else "ordinary", position=[x, y, 0], name=k) for k, (x, y) in pos.items()}
    roads = {f"r{i}": dict(fromNodeId=a, toNodeId=b, name=f"r{i}", direction="both",
                           geometry=dict(kind="path", anchors=[], spans=[dict(kind="line")]),
                           widthM=dict(state="known", value=width[f"r{i}"] if isinstance(width, dict) else width), extensions={})
             for i, (a, b) in enumerate(edges)}
    arcs_in, arcs_out = collections.defaultdict(list), collections.defaultdict(list)
    for rid, r in roads.items():
        arcs_out[r["fromNodeId"]].append((rid, "forward"))
        arcs_in[r["toNodeId"]].append((rid, "forward"))
        arcs_out[r["toNodeId"]].append((rid, "backward"))
        arcs_in[r["fromNodeId"]].append((rid, "backward"))
    junctions, movements = {}, {}
    for n in pos:
        junctions[f"j{n}"] = dict(nodeIds=[n], model="explicit_movements")
        for i in arcs_in[n]:
            for o in arcs_out[n]:
                if i[0] != o[0]:
                    movements[f"m{len(movements)}"] = dict(allowed=True, junctionId=f"j{n}",
                                                           incomingArc=dict(roadId=i[0], direction=i[1]),
                                                           outgoingArc=dict(roadId=o[0], direction=o[1]))
    ext = lambda c: {"org.shipyard.spatial_classification": {"classId": c}}
    zs = {z: dict(name=z, extensions=dict(ext(c), **{"org.shipyard.research_semantics": {
              "frontage": [dict(roadId=rid, sStartM=s, sEndM=s + 10) for s in ss]}}))
          for z, (c, rid, ss) in (zones or {}).items()}
    return dict(nodes=nodes, roads=roads, junctions=junctions, movements=movements, zones=zs,
                accessPoints={f"a{f}": dict(nodeId=n) for f, (n, _) in fac.items()},
                facilities={f: dict(name=f, accessPointIds=[f"a{f}"], extensions=ext(c)) for f, (n, c) in fac.items()})


def load_net(m, **over):
    """Network of a synthetic map (main scenario with the overrides that shape the network, e.g. section)."""
    return Net(to_graph(m), P.scenario("main", **over))


CORRIDOR = dict(A=(0, 0), n1=(50, 0), n2=(250, 0), n3=(450, 0), B=(500, 0))
CORRIDOR_EDGES = [("A", "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", "B")]


def corridor_net():
    """Single-lane two-way corridor: A (entrance) - n1 - n2 - n3 - B (entrance), all four roads 9 m wide (loaded and empty
    vehicles both hold a road alone)."""
    return load_net(make_map(CORRIDOR, CORRIDOR_EDGES, {"fA": ("A", "assembly_workshop"), "fB": ("B", "paint_workshop")}))


def task(i, o, d, release=0.0):
    return dict(id=i, flow="T", o=o, d=d, B=10.0, Lv=20.0, release=release)


def run_with(net, mode, tasks, locs, p=None, record=True):
    """Runs the given tasks with one vehicle per entry of locs, placed there."""
    sim = Sim(net, p or short(), mode, len(locs), 1, None, record=record, tasks=tasks)
    for v, loc in zip(sim.vs, locs):
        v.loc = loc
    return sim.run()


def ring(n, side=300.0, width=9.0):
    """Single-lane ring of n entrance nodes (regular n-gon, side `side`): N0 ... N(n-1), each the entrance of a building (an
    exclusive node resource)."""
    rad = side / (2 * math.sin(math.pi / n))
    pos = {f"N{i}": (rad * math.cos(2 * math.pi * i / n), rad * math.sin(2 * math.pi * i / n)) for i in range(n)}
    edges = [(f"N{i}", f"N{(i + 1) % n}") for i in range(n)]
    fac = {f"f{i}": (f"N{i}", "assembly_workshop") for i in range(n)}
    return load_net(make_map(pos, edges, fac, width=width))


def random_net(seed, n=4):
    """Random small network: an n x n grid (spacing 150 m) keeping a spanning tree plus random further edges, road widths
    drawn from 9, 12 and 15 m (mostly single-lane); five corner or edge nodes each get a building entrance (access roads
    20 m long, 31 m wide); one grid edge carries a roadside buffer stop."""
    rng = random.Random(seed)
    pos = {f"g{i}{j}": (150.0 * i, 150.0 * j) for i in range(n) for j in range(n)}
    cand = [(f"g{i}{j}", f"g{i + 1}{j}") for i in range(n - 1) for j in range(n)] + \
           [(f"g{i}{j}", f"g{i}{j + 1}") for i in range(n) for j in range(n - 1)]
    rng.shuffle(cand)
    comp = {k: k for k in pos}

    def root(x):
        while comp[x] != x:
            x = comp[x]
        return x
    edges = []
    for a, b in cand:                       # spanning tree + about half of the other edges
        if root(a) != root(b):
            comp[root(a)] = root(b)
            edges.append((a, b))
        elif rng.random() < 0.5:
            edges.append((a, b))
    width = {f"r{i}": rng.choice((9.0, 9.0, 12.0, 15.0)) for i in range(len(edges))}
    fac, classes = {}, ("assembly_workshop", "paint_workshop", "outfitting_workshop")
    for k, g in enumerate(rng.sample([f"g0{j}" for j in range(n)] + [f"g{n - 1}{j}" for j in range(n)], 5)):
        x, y = pos[g]
        pos[f"F{k}"] = (x + (-20.0 if g.startswith("g0") else 20.0), y)
        width[f"r{len(edges)}"] = 31.0
        edges.append((g, f"F{k}"))
        fac[f"f{k}"] = (f"F{k}", classes[k % 3])
    zones = {"zbuf": ("buffer", "r0", (60.0,))}
    return load_net(make_map(pos, edges, fac, width=width, zones=zones))


def random_case(seed):
    """Tasks on a random small network: 6 vehicles, 40 tasks (any origin and destination, block widths 5-15 m), released
    uniformly in the first 30,000 s."""
    net = random_net(seed)
    rng = random.Random(seed)
    stops = sorted(net.stops)
    tasks = [dict(task(i, *rng.sample(stops, 2), release=rng.uniform(0, 30000)), B=float(rng.randint(5, 15))) for i in range(40)]
    return net, tasks, [net.stops[s]["node"] for s in rng.sample(stops, 6)]


def sectioned_corridor():
    """A -(20 m, 31 m wide)- n1 -(600 m, 9 m wide, single lane)- n2 -(20 m, 31 m wide)- B, with a roadside buffer stop at
    300 m on the middle road. Returns the network whole and sectioned at the stops (cut at 275 and 325 m into 3 sections)."""
    m = make_map(dict(A=(0, 0), n1=(20, 0), n2=(620, 0), B=(640, 0)), [("A", "n1"), ("n1", "n2"), ("n2", "B")],
                 {"fA": ("A", "assembly_workshop"), "fB": ("B", "paint_workshop")}, width={"r0": 31.0, "r1": 9.0, "r2": 31.0},
                 zones={"zbuf": ("buffer", "r1", (295.0,))})
    G = to_graph(m)
    return [Net(G, P.scenario("main", section=s)) for s in (None, "stops")]


def two_routes():
    """A -10 m- J1 = J2 -10 m- B; between J1 and J2 a narrow road r1 (9 m, 500 m) and a wide route r3 + r4 (25 m, 400 m each,
    via K). Access roads r0, r2 are 31 m wide. All tasks A -> B, B = 10 m, Lv = 20 m; empty vehicles take no road room
    (empty_free), loading and unloading 60 s each."""
    y = math.sqrt(400 ** 2 - 250 ** 2)
    net = load_net(make_map(dict(A=(0, 0), J1=(10, 0), J2=(510, 0), B=(520, 0), K=(260, y)),
                            [("A", "J1"), ("J1", "J2"), ("J2", "B"), ("J1", "K"), ("K", "J2")],
                            {"fA": ("A", "assembly_workshop"), "fB": ("B", "paint_workshop")},
                            width={"r0": 31.0, "r1": 9.0, "r2": 31.0, "r3": 25.0, "r4": 25.0}))
    p = P.scenario("empty_free", load_s=60.0, unload_s=60.0)
    return net, p, [dict(id=i, flow="T", o="A", d="B", B=10.0, Lv=20.0, release=0.0) for i in range(10)]


# ---------------------------------------------------------------- yard networks and scenario parameters
@functools.lru_cache(maxsize=None)
def _yard_net(key, over):
    Y = P.YARDS[key]
    return Net(Y["network"], P.scenario("main", **json.loads(over)), Y["roles"])


def yard_net(key, **over):
    """Network of a yard key (yupu = Okpo rev 1833, yupu_r1825 = Okpo rev 1825, yantai = Yantai rev 396), cached; over =
    parameters that shape the network (section, block_m, retrofit, dock_stops)."""
    return _yard_net(key, json.dumps(over, sort_keys=True))


def crane_scen(yard, n6=None, crane_n=None, days=None, span_min=None):
    """Parameters of crane-paced erection as in the stored runs: cranes from the network (crane_n = "map"), the yard's peak
    P6 per day (params.CRANE_YARD) and peak demand; n6, crane_n, span_min, days override."""
    demand = P.YARDS[yard].get("demand", P.DEMAND)
    s = dict(P.SCENARIOS["crane"], p6_per_day=n6 or P.CRANE_YARD[P.base(yard)]["p6_peak"], erect_peak=demand["peak"])
    s.update({k: v for k, v in dict(crane_n=crane_n, span_min=span_min, days=days).items() if v})
    return s


def crane_p(yard, scen="crane", **over):
    """Scenario parameters under crane pacing; with days (and warm) a short run as short()."""
    kw = {k: over.pop(k) for k in ("n6", "crane_n", "span_min") if k in over}
    if over.get("days"):
        return short(scen, **dict(crane_scen(yard, **kw), **over))
    return P.scenario(scen, **crane_scen(yard, **kw))


def load_config(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- safe release: cross-check with a naive reduction
class NaiveCheck(Sim):
    """Cross-check of safe release: every safe_to_grant also runs a naive reduction without caches or orders (vehicles that
    can finish are removed batch by batch); both must agree. Also checks that the occupancy not released on time (sticky,
    sused) equals that recomputed from hold / pending."""
    naive_n = 0

    def safe_to_grant(self, v):
        got = super().safe_to_grant(v)
        tot = collections.defaultdict(float)
        for u in self.vs:
            assert u.sticky == {r: h[1] for r, h in u.hold.items() if h[0] > u.pending[r]}, u.id
            for r, w in u.sticky.items():
                tot[r] += w
        assert all(abs(self.sused[r] - tot[r]) < 1e-9 for r in set(tot) | set(self.sused))
        st, nd = self.post_grant(v)
        left = {u.id: (u.sticky, u.need) for u in self.vs if u.sticky and u.id != v.id}
        if st:
            left[v.id] = (st, nd)
        while True:
            used = collections.defaultdict(float)
            for s, _ in left.values():
                for r, w in s.items():
                    used[r] += w
            go = [u for u, (s, n) in left.items() if all(d + used[r] - s.get(r, 0.0) <= self.cap[r] + EPS for r, d in n.items())]
            if not go:
                break
            for u in go:
                del left[u]
        assert got == (not left), (v.id, got, sorted(left))
        self.naive_n += 1
        return got


def no_one_left(sim):
    """Safe release: no vehicle leaves the network - no deadlock record, no recovered vehicle, no voided release event."""
    return not sim.deadlocks and all(not v.victim and v.gen == 0 for v in sim.vs)


def naive_run(net, tasks, locs, p=None):
    """Safe release cross-checked against the naive reduction at every decision, one vehicle per entry of locs."""
    sim = NaiveCheck(net, p or short(), "segment_safe", len(locs), 1, None, record=True, tasks=tasks)
    for v, loc in zip(sim.vs, locs):
        v.loc = loc
    return sim.run()

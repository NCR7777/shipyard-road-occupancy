"""Demand mappings with intra-area moves and with inter-area moves only: how the daily demand of block moves becomes vehicle
tasks. Scenario parameter demand_map ("intra" or "inter_only"); without it this module is not used and the task stream is that
of occupancy.tasks, bit for bit. Serves the required fleet, the screen, the saturation curves and K_stable under these
mappings (experiments/runner.py: compute, map_stream) and their system bounds (system_mix).

The daily demand is converted at MOVES_PER_BLOCK = 10 moves per block (params.daily_tasks); the weights of the flows P1-P8 add
up to only 3.45 moves (the inter-area moves):
  intra       of the 10 moves per block, 3.45 are inter-area moves drawn with the flows and weights of the proportional mapping
              (the same random stream as occupancy.tasks.TaskStream, the inter-area tasks unchanged one by one); the other 6.55
              are intra-area moves: the area class is weighted by the number of stays of a block in areas of that class
              (intra_roles), the entity is equiprobable within its class, origin and destination are two different stops of
              that entity (ordered pairs equiprobable); an entity with a single stop has origin = destination, the vehicle only
              loads and unloads at that stop and holds the stop resources like any other loading and unloading (workshop
              entrance node, drive-in bay, road of the roadside stop), without driving. Block sizes as for the inter-area
              moves. The intra-area moves have their own random stream Random(seed * 100 + 71).
  inter_only  inter-area moves only: at the release positions of intra only the inter-area moves are kept (daily demand x
              INTER_SHARE); tasks and release times equal the inter-area moves of intra one by one.
Positions of the two kinds: the g-th move (from 0, continued across days) is an inter-area move iff floor((g+1) s) > floor(g s),
s = INTER_SHARE = 69/200 (integer arithmetic), i.e. the inter-area moves are spread evenly over the release positions of the
proportional mapping (level_n a day in `batches` batches, the i-th in batch i mod batches, tasks.releases).
Only for the main flow mix (no crane-paced erection, no concentrated erection); origins and destinations are sampled by entity
(od_sampling applies to the inter-area moves only).

Run:   imported by experiments/runner.py when a scenario has demand_map = "intra" or "inter_only" (and by the bound modules).
"""
import collections
import random
from fractions import Fraction

from occupancy.params import DOCK_CLASS, FLOW
from occupancy.tasks import TaskStream, flow_weights, releases

MOVES_PER_BLOCK = 10                                                    # transfers of params.daily_tasks
INTER_SHARE = sum(Fraction(str(f[3])) for f in FLOW) / MOVES_PER_BLOCK  # 3.45 / 10 = 69/200
INTRA = "IN"                                                            # flow name of the intra-area moves
MAPS = ("intra", "inter_only")


def inter_slot(g):
    """Whether the g-th move is an inter-area move."""
    a, b = INTER_SHARE.numerator, INTER_SHARE.denominator
    return (g + 1) * a // b > g * a // b


def intra_roles(p):
    """Area classes of the intra-area moves (the roles of FLOW, such as yard|buffer) and their weights: the number of stays of a
    block in an area of that class = max(sum of the inflow weights, sum of the outflow weights) (the assembly workshop at the
    start has outflow only); docks do not count (entering the dock is erection, where the vehicle's move ends). With the flow
    weights of params.FLOW: assembly workshop 1.05, outfitting 0.5, painting 1.0, stockyard and buffer 0.7, assembly area 1.05."""
    inn, out = collections.Counter(), collections.Counter()
    for (_, oc, dc, _), x in zip(FLOW, flow_weights(p)):
        out[oc] += x
        inn[dc] += x
    roles = [r for r in dict.fromkeys(c for f in FLOW for c in f[1:3]) if r != DOCK_CLASS]
    return roles, [max(inn[r], out[r]) for r in roles]


def check(p):
    if p.get("demand_map") not in MAPS:
        raise ValueError(f"unknown demand mapping: {p.get('demand_map')}")
    if p.get("crane_n") or p.get("erect_batches"):
        raise ValueError("the demand mappings apply to the main flow mix only (no crane-paced erection, no concentrated erection)")


class Stream(TaskStream):
    """Task stream of the intra mapping: the positions of inter-area moves are drawn as by TaskStream (the same random stream),
    the other positions are intra-area moves (their own random stream). The ids are consecutive over the positions."""

    def __init__(self, net, seed, p):
        super().__init__(net, seed, p)
        self.rng_in = random.Random(seed * 100 + 71)
        self.roles = intra_roles(p)

    def next(self, release):
        if inter_slot(self.n):
            return super().next(release)
        net, rng, p = self.net, self.rng_in, self.p
        role = rng.choices(self.roles[0], weights=self.roles[1])[0]
        e = rng.choice([x for c in role.split("|") for x in net.by_cls[c]])
        so = rng.choice(net.ents[e]["stops"])
        sd = rng.choice([s for s in net.ents[e]["stops"] if s != so] or [so])
        bw = rng.randint(*p["block_w"])
        bl = rng.randint(max(bw, p["block_l"][0]), p["block_l"][1])
        task = dict(id=self.n, flow=INTRA, o=so, d=sd, B=max(bw, p["plat_w"]), Lv=max(bl, p["plat_len"]), release=release)
        self.n += 1
        return task


def stream(net, seed, p):
    """Task stream of saturated levels and bound samples: Stream for intra; for inter_only the task stream of occupancy.tasks
    (inter-area moves only)."""
    check(p)
    return Stream(net, seed, p) if p["demand_map"] == "intra" else TaskStream(net, seed, p)


def demand_tasks(net, seed, level_n, p):
    """Finite demand (release times as tasks.demand_tasks for the main flow mix: tasks.releases): intra = every position,
    inter_only = only the positions of inter-area moves."""
    check(p)
    rel = releases(level_n, p)
    if p["demand_map"] == "inter_only":
        s = TaskStream(net, seed, p)
        tasks = [s.next(r) for g, r in enumerate(rel) if inter_slot(g)]
    else:
        s = Stream(net, seed, p)
        tasks = [s.next(r) for r in rel]
    return sorted(tasks, key=lambda t: (t["release"], t["id"]))


def share(p):
    """Share of the daily demand released to the vehicles: INTER_SHARE for inter_only (the target line of the required fleet
    is n x this share), otherwise 1."""
    return float(INTER_SHARE) if p.get("demand_map") == "inter_only" else 1


def intra_law(net, p):
    """Distribution of origin and destination of the intra-area moves (the sampling law of Stream): [(probability, task)],
    task = dict(flow, o, d)."""
    roles, w = intra_roles(p)
    out = collections.Counter()
    for r, x in zip(roles, w):
        E = [e for c in r.split("|") for e in net.by_cls[c]]
        for e in E:
            S = net.ents[e]["stops"]
            for so in S:
                D = [s for s in S if s != so] or [so]
                for sd in D:
                    out[(so, sd)] += x / sum(w) / len(E) / len(S) / len(D)
    return [(q, dict(flow=INTRA, o=o, d=d)) for (o, d), q in out.items()]


def system_mix(net, p):
    """Task law of the system bound (the mix [(probability, task, bl)] of bounds.system): intra = INTER_SHARE x inter-area law
    + (1 - INTER_SHARE) x intra-area law; inter_only equals the main flow mix of bounds.system.scenario_mix. lambda is in moves
    a day (both kinds together)."""
    from bounds.system import bl_joint, od_mix
    check(p)
    bl = bl_joint(p)
    if p["demand_map"] == "inter_only":
        return [(w, t, bl) for w, t in od_mix(net, p)]
    s = float(INTER_SHARE)
    return [(s * w, t, bl) for w, t in od_mix(net, p)] + [((1 - s) * w, t, bl) for w, t in intra_law(net, p)]

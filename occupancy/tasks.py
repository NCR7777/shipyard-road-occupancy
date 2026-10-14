"""Task stream: generates origins and destinations on the functional areas and entrances of a yard from the flows, their
weights and the block-size rules (the synthetic parts are scenario assumptions).

For a given seed the task sequence and the vehicle start points are fixed and do not change with the fleet size or the model
(common random numbers); scenarios with different flow shares (erection peak) have different task sequences.
"""
import collections
import random

from .params import DOCK_CLASS, FLOW, START_CLASSES

CRANE_ID0 = 10 ** 6        # first id of the P6 tasks under crane-paced erection (disjoint from the ids of the task stream)


def flow_weights(p):
    """Sampling weights of the flows. With p["flow_share"] = {flow: share}, these flows get the given shares and the other
    flows keep their FLOW weights, scaled in proportion; without it, the FLOW weights are returned unchanged."""
    fix = p.get("flow_share") or {}
    if not fix:
        return [f[3] for f in FLOW]
    scale = (1 - sum(fix.values())) / sum(f[3] for f in FLOW if f[0] not in fix)
    return [fix[f[0]] if f[0] in fix else f[3] * scale for f in FLOW]


def mark_lift(net, task, pairs):
    """Marks a P6 inside a crane span (net.lift_pairs) as lift: the gantry crane lifts it directly and no vehicle is
    dispatched. Only adds a key; draws no random numbers."""
    if task["flow"] == "P6" and (net.stops[task["o"]]["ent"], net.stops[task["d"]]["ent"]) in pairs:
        task["lift"] = True
    return task


class TaskStream:
    def __init__(self, net, seed, p):
        self.net, self.p = net, p
        self.rng = random.Random(seed)
        self.n = 0
        self.w = flow_weights(p)
        self.lifts = net.lift_pairs(p)

    def next(self, release):
        net, rng, p = self.net, self.rng, self.p
        fid, oc, dc, _ = rng.choices(FLOW, weights=self.w)[0]
        if p.get("od_sampling", "entity") == "stop":     # every stop of the role equally likely (entities weighted by their number of stops)
            so = rng.choice([s for c in oc.split("|") for e in net.by_cls[c] for s in net.ents[e]["stops"]])
            o = net.stops[so]["ent"]
            sd = rng.choice([s for c in dc.split("|") for e in net.by_cls[c] if e != o for s in net.ents[e]["stops"]])
        else:                                            # entities equally likely, stops within an entity equally likely (main model; same random-number use as earlier versions)
            o = rng.choice([e for c in oc.split("|") for e in net.by_cls[c]])
            d = rng.choice([e for c in dc.split("|") for e in net.by_cls[c] if e != o])
        bw = rng.randint(*p["block_w"])
        bl = rng.randint(max(bw, p["block_l"][0]), p["block_l"][1])
        if p.get("od_sampling", "entity") != "stop":
            so, sd = rng.choice(net.ents[o]["stops"]), rng.choice(net.ents[d]["stops"])
        task = dict(id=self.n, flow=fid, o=so, d=sd, B=max(bw, p["plat_w"]), Lv=max(bl, p["plat_len"]), release=release)
        self.n += 1
        return mark_lift(net, task, self.lifts)


def releases(level_n, p):
    """Daily demand level_n, released in `batches` batches per day (task i in batch i mod batches)."""
    day, nb = p["day_s"], p["batches"]
    return [d * day + (i % nb) * day / nb for d in range(p["days"]) for i in range(level_n)]


def crane_p6(net, p):
    """Number of P6 (erection units) per day under crane-paced erection: p["p6_per_day"], converted from published output
    figures (params.CRANE_YARD)."""
    return p["p6_per_day"]


def crane_rho(net, p):
    """Crane utilisation ρ = P6 per day × crane time per lift / (total number of cranes × working time). The P6 are split
    over the docks in proportion to their crane counts, so ρ is the same at every dock."""
    return crane_p6(net, p) * p["crane_cycle_s"] / (sum(net.crane_counts(p).values()) * p["day_s"])


def crane_split(net, p, n6, days):
    """Splits the n6 P6 of each day over the docks in proportion to their crane counts (smooth weighted round-robin over the
    j-th P6, continued across days; with equal crane counts the j-th P6 goes to dock j mod number of docks):
    returns [[indices of the P6 at dock k on day d]]. n6 need not be an integer (Okpo: 13.3): the first d days get
    round(n6 × d) in total, each day ⌊n6⌋ or ⌈n6⌉; an integer n6 gives exactly n6 per day. The number m per dock and day
    must not exceed ⌊cranes × day_s / time per lift⌋ (the crane capacity), otherwise a ValueError is raised."""
    docks = net.by_cls[DOCK_CLASS]
    w = [net.crane_counts(p)[z] for z in docks]
    cum = [round(n6 * d) for d in range(days + 1)]
    cur, seq = [0] * len(docks), []
    for _ in range(cum[-1]):
        cur = [a + b for a, b in zip(cur, w)]
        k = max(range(len(cur)), key=lambda i: (cur[i], -i))
        cur[k] -= sum(w)
        seq.append(k)
    out = [[[j for j in range(cum[d], cum[d + 1]) if seq[j] == k] for k in range(len(docks))] for d in range(days)]
    for per in out:
        for k, jj in enumerate(per):
            if len(jj) > int(w[k] * p["day_s"] // p["crane_cycle_s"]):
                raise ValueError(f"P6 exceeds the crane capacity: {docks[k]}, {len(jj)} per day, {w[k]} cranes")
    return out


def crane_tasks(net, seed, p, n6):
    """Crane-paced erection: the n6 P6 tasks of each day are split over the docks in proportion to their crane counts
    (crane_split); the m tasks of a dock on one day are released at equal intervals from the start of the day, every
    day_s / m, which is no shorter than the crane cycle / number of cranes, so the cranes are not overloaded. Origins
    (assembly areas; entities equally likely, stops within an entity equally likely), stops within the dock and block sizes
    use a separate random stream and leave the random numbers of the other tasks unchanged. (Area, dock) pairs inside a
    crane span are marked lift (mark_lift): the simulation dispatches no vehicle for them and they hold only the crane;
    random numbers are drawn as usual, so whether a pair lies inside or outside a span does not change the origin,
    destination or release time of any P6."""
    pairs = net.lift_pairs(p)
    rng = random.Random(seed * 100 + 61)
    _, oc, _, _ = next(f for f in FLOW if f[0] == "P6")
    srcs = [e for c in oc.split("|") for e in net.by_cls[c]]
    docks, day, out = net.by_cls[DOCK_CLASS], p["day_s"], []
    for d, per in enumerate(crane_split(net, p, n6, p["days"])):
        for k, jj in enumerate(per):
            for i, j in enumerate(jj):
                o = rng.choice(srcs)
                bw = rng.randint(*p["block_w"])
                bl = rng.randint(max(bw, p["block_l"][0]), p["block_l"][1])
                out.append(mark_lift(net, dict(id=CRANE_ID0 + j, flow="P6", o=rng.choice(net.ents[o]["stops"]),
                                               d=rng.choice(net.ents[docks[k]]["stops"]), B=max(bw, p["plat_w"]),
                                               Lv=max(bl, p["plat_len"]), release=d * day + i * day / len(jj)), pairs))
    return out


def demand_tasks(net, seed, level_n, p):
    """If p["erect_batches"] gives batch numbers (concentrated erection), the P6 tasks of each day are placed in these
    batches in turn; the release times of the other tasks are unchanged.
    If p["crane_n"] is given (crane-paced erection), the task stream produces only the other flows (per day, level_n − that
    day's number of P6, in batches as above), and crane_tasks releases the P6 at the crane pace (including direct lifts
    inside a crane span, with no vehicle); a P6 released at the same time as a batch comes first (the erection plan takes
    priority)."""
    s = TaskStream(net, seed, p)
    if p.get("crane_n"):
        c6 = crane_tasks(net, seed, p, crane_p6(net, p))
        per = collections.Counter(int(t["release"] // p["day_s"]) for t in c6)
        rel = [d * p["day_s"] + (i % p["batches"]) * p["day_s"] / p["batches"] for d in range(p["days"]) for i in range(level_n - per[d])]
        return sorted([s.next(r) for r in rel] + c6, key=lambda t: (t["release"], t["id"] < CRANE_ID0, t["id"]))
    tasks = [s.next(r) for r in releases(level_n, p)]
    hot = p.get("erect_batches")
    if hot:
        day, nb, n6 = p["day_s"], p["batches"], collections.Counter()
        for t in tasks:
            if t["flow"] == "P6":
                d = int(t["release"] // day)
                t["release"] = d * day + hot[n6[d] % len(hot)] * day / nb
                n6[d] += 1
    return sorted(tasks, key=lambda t: (t["release"], t["id"]))


def vehicle_starts(net, seed, K):
    """Start points of the first K vehicles: drawn at random among the first entrances of the assembly, painting and
    outfitting workshops (random stream seed·100+13)."""
    rng = random.Random(seed * 100 + 13)
    pool = [net.ents[e]["stops"][0] for c in START_CLASSES for e in net.by_cls[c]]
    return [net.stops[rng.choice(pool)]["node"] for _ in range(K)]

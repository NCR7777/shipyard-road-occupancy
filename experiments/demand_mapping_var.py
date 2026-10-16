"""Where the intra-area moves are placed: variants of the intra mapping of experiments/demand_mapping.py (scenario parameter
intra_var; demand_mapping itself is unchanged). Serves the sensitivity of the required fleet and of the saturation curve to the
placement of the intra-area moves (experiments/runner.py: compute, map_stream; system bound: system_mix).
  open       only in the open areas (stockyard and buffer, assembly area), class weights as demand_mapping.intra_roles, scaled
             up proportionally; entities equiprobable within a class;
  stop       all areas, stops equiprobable within a class (entities weighted by their number of stops: large areas and
             workshops with many doors get more rehandling);
  open_stop  both changes.
Positions of the inter-area moves and the random streams as in demand_mapping; the intra-area moves use the same separate
random stream Random(seed * 100 + 71).

Run:   imported by experiments/runner.py when a scenario has demand_map = "intra" and intra_var.
"""
import collections

import demand_mapping as M
from occupancy.tasks import releases

OPEN = ("yard|buffer", "assembly_yard")
VARS = ("open", "stop", "open_stop")


def roles(p):
    r, w = M.intra_roles(p)
    keep = [(a, x) for a, x in zip(r, w) if "open" not in p["intra_var"] or a in OPEN]
    return [a for a, _ in keep], [x for _, x in keep]


def pool(net, role):
    """Origin stops within a class: [(entity, stop)]."""
    return [(e, s) for c in role.split("|") for e in net.by_cls[c] for s in net.ents[e]["stops"]]


class Stream(M.Stream):
    def __init__(self, net, seed, p):
        if p.get("intra_var") not in VARS:
            raise ValueError(f"unknown variant of the intra-area moves: {p.get('intra_var')}")
        super().__init__(net, seed, p)
        self.roles = roles(p)
        self.by_stop = "stop" in p["intra_var"]

    def next(self, release):
        if M.inter_slot(self.n) or not self.by_stop:
            return super().next(release)
        net, rng, p = self.net, self.rng_in, self.p
        role = rng.choices(self.roles[0], weights=self.roles[1])[0]
        e, so = rng.choice(pool(net, role))
        sd = rng.choice([s for s in net.ents[e]["stops"] if s != so] or [so])
        bw = rng.randint(*p["block_w"])
        bl = rng.randint(max(bw, p["block_l"][0]), p["block_l"][1])
        task = dict(id=self.n, flow=M.INTRA, o=so, d=sd, B=max(bw, p["plat_w"]), Lv=max(bl, p["plat_len"]), release=release)
        self.n += 1
        return task


def demand_tasks(net, seed, level_n, p):
    M.check(p)
    s = Stream(net, seed, p)
    return sorted([s.next(r) for r in releases(level_n, p)], key=lambda t: (t["release"], t["id"]))


def intra_law(net, p):
    r, w = roles(p)
    out = collections.Counter()
    for role, x in zip(r, w):
        P_ = pool(net, role)
        E = sorted({e for e, _ in P_})
        for e in E:
            S = net.ents[e]["stops"]
            pe = len(S) / len(P_) if "stop" in p["intra_var"] else 1 / len(E)
            for so in S:
                D = [s for s in S if s != so] or [so]
                for sd in D:
                    out[(so, sd)] += x / sum(w) * pe / len(S) / len(D)
    return [(q, dict(flow=M.INTRA, o=o, d=d)) for (o, d), q in out.items()]


def system_mix(net, p):
    from bounds.system import bl_joint, od_mix
    M.check(p)
    bl, s = bl_joint(p), float(M.INTER_SHARE)
    return [(s * w, t, bl) for w, t in od_mix(net, p)] + [((1 - s) * w, t, bl) for w, t in intra_law(net, p)]

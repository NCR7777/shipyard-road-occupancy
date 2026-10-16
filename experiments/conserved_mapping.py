"""The conserved inter-area mapping of the main case (erection flow conserved) and the share of intra-area moves. Scenario
parameter demand_map = "conserved" (experiments/runner.py: compute and map_stream turn to this module); intra_share = q is the
share of intra-area moves among all moves (default 0). experiments/demand_mapping.py is unchanged; this is a subclass of it.
Serves the required fleet, the saturation curves and K_stable of the main case and their system bounds (system_mix).

Ten moves per block (params.daily_tasks):
  to the dock (P6)    0.2 moves per block, the value of its flow weight (Okpo 339 tasks a day -> 6.8 a day, close to the mean
                      of 6.7 erection lifts a day of the yard data); the proportional mapping scales all 10 moves by the flow
                      shares, which gives 10 x 0.2 / 3.45 = 0.58 moves per block to the dock;
  intra-area moves    10 q moves, placed as the intra-area moves of demand_mapping.Stream (area class weighted by the number of
                      stays, entity equiprobable within its class, two different stops of the entity);
  inter-area moves    the other 9.8 - 10 q moves, along the flows P1-P5, P7, P8 other than to the dock, split by the flow
                      weights of params.FLOW, origins, destinations and routes as in the proportional mapping (entities
                      equiprobable, shortest route between the two areas).
q = 0 is the conserved inter-area mapping (the main case); q = 0.655 equals the intra mapping of demand_mapping one by one. Why
the flow weights: they are the scaling rule of the proportional mapping with only the term of the flow to the dock removed, so
the two mappings differ only by the conservation of the flow to the dock; there are no operating data on the rehandling between
areas, and the flow weights are the only basis for the intensity of the moves between areas. The task law equals that of the
proportional mapping with flow_share = {P6: 0.02} (experiments/conserved_bounds.py checks the system bound).

Release positions as in demand_mapping: the g-th position (from 0, continued across days) is a basic inter-area move iff
demand_mapping.inter_slot(g) (69 of every 200), drawn from the same random stream as TaskStream (all eight flows, P6
included); of the other positions (the rehandling positions, 6.55 / 10) the j-th (from 0, continued across days) is an
inter-area rehandling move iff floor((j+1) r) > floor(j r), r = 1 - q / 0.655, otherwise an intra-area move. Inter-area
rehandling moves use their own random stream Random(seed * 100 + 73), drawn as by TaskStream with the flows other than to the
dock only; intra-area moves use Random(seed * 100 + 71) of demand_mapping.
Pairing (common random numbers): the basic inter-area moves are the same, at the same release times, under both mappings of
demand_mapping and every q here (the first tasks of the task stream of the proportional mapping); the intra-area moves are the
first ones of the intra-area sequence of the intra mapping; the inter-area rehandling moves are the first ones of the
rehandling sequence of q = 0. The mappings differ only in what is placed at the rehandling positions.
Simulation supports 0 <= q <= 0.655 only (a larger q would take positions of basic inter-area moves, and the flow to the dock
would no longer be 0.2 moves per block); the task law of the system bound is defined for 0 <= q < 0.98.

Run:   imported by experiments/runner.py when a scenario has demand_map = "conserved" (and by the bound modules).
"""
from fractions import Fraction

import demand_mapping as M
from occupancy.tasks import TaskStream, releases

P6_SHARE = Fraction("0.2") / M.MOVES_PER_BLOCK          # 1/50: share of the moves to the dock among all moves
BASE_INTRA = 1 - M.INTER_SHARE                           # 131/200: share of the rehandling positions (intra-area share of intra)
NONDOCK = {"P6": 0.0}                                    # flow_share: flows other than to the dock only


def q_of(p):
    return Fraction(str(p.get("intra_share", 0)))


def check(p, sim=True):
    if p.get("demand_map") != "conserved":
        raise ValueError(f"not the conserved mapping: {p.get('demand_map')}")
    if p.get("crane_n") or p.get("erect_batches") or p.get("flow_share") or p.get("intra_var"):
        raise ValueError("the conserved mapping applies to the main flow mix only (no crane-paced erection, concentrated erection, "
                         "flow shares or variants of the intra-area moves)")
    q = q_of(p)
    if q < 0 or (q > BASE_INTRA if sim else q >= 1 - P6_SHARE):
        raise ValueError(f"share of intra-area moves out of range: {q} (simulation <= 0.655, bound < 0.98)")


def reh_inter(j, r):
    """Whether the j-th rehandling position is an inter-area rehandling move (share r)."""
    return (j + 1) * r.numerator // r.denominator > j * r.numerator // r.denominator


class Stream(M.Stream):
    """Basic inter-area moves and intra-area moves as drawn by demand_mapping.Stream; the inter-area rehandling moves at the
    rehandling positions come from a separate task stream without the flow to the dock. The ids are consecutive over the
    positions."""

    def __init__(self, net, seed, p):
        check(p)
        super().__init__(net, seed, p)
        self.r = 1 - q_of(p) / BASE_INTRA
        self.j = 0
        self.reh = TaskStream(net, seed * 100 + 73, dict(p, flow_share=NONDOCK))

    def next(self, release):
        if not M.inter_slot(self.n):
            self.j += 1
            if reh_inter(self.j - 1, self.r):
                task = dict(self.reh.next(release), id=self.n)
                self.n += 1
                return task
        return super().next(release)


def stream(net, seed, p):
    return Stream(net, seed, p)


def demand_tasks(net, seed, level_n, p):
    """Finite demand: release times as tasks.releases (as in demand_mapping)."""
    s = Stream(net, seed, p)
    return sorted([s.next(r) for r in releases(level_n, p)], key=lambda t: (t["release"], t["id"]))


def system_mix(net, p):
    """Task law of the system bound: 0.02 x law of the flow to the dock + (0.98 - q) x inter-area law without the flow to the
    dock + q x intra-area law (demand_mapping.intra_law). lambda is in moves a day."""
    from bounds.system import bl_joint, od_mix
    check(p, sim=False)
    q, s6 = q_of(p), P6_SHARE
    bl = bl_joint(p)
    out = [(float(s6) * w, t, bl) for w, t in od_mix(net, dict(p, flow_share={"P6": 1.0}))]
    out += [(float(1 - s6 - q) * w, t, bl) for w, t in od_mix(net, dict(p, flow_share=NONDOCK))]
    return out + ([(float(q) * w, t, bl) for w, t in M.intra_law(net, p)] if q else [])

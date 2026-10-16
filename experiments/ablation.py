"""Ablation of the occupancy representation, every move scaled with the flows (supplementary material, the table "ablation of
the occupancy representation, every move scaled with the flows"): the occupancy representation is replaced by simpler ones,
and simulation and bound use the same representation.

Scenario parameter occ_rep (absent by default; without it every point takes the ordinary code, bit for bit unchanged):
  count  (a) a vehicle count per road (the representation of Park & Seo 2012 and Tao et al. 2017): on a road (roads with
         roadside stops included) every vehicle counts the same, loaded or empty, whatever its block width; a road of width
         W < B_max + w + c (the widest loaded vehicle and an empty one do not fit side by side; at Okpo and Yantai W < 21.5 m)
         is held alone (C_r), any other by at most 2 vehicles (C_r / 2 each); junctions, doors and drive-in places as in the
         occupancy model (a loaded vehicle holds a junction alone, an empty one by width; a stop takes one place per vehicle).
         CountNet.demand.
  same   (b) loaded and empty the same width: an empty run takes the occupancy of the loaded block width B of the task it goes
         to fetch (roads and junctions by width, min(C, B + c)); loaded runs as in the occupancy model.
         Simulation SameSim.legs; bound same_t1: the empty flow split by the B of the next task.
(c) empty vehicles take no width is the existing empty_rule = "free" (network.demand: an empty vehicle takes 0 on roads and
junctions, a stop still takes one place); it does not go through this module.

The bound changes with the representation only through the occupancy coefficients (d of Eq. (1)); the structure of the linear
programme is unchanged:
  count: the shares d / C of loaded and empty runs on roads become 1 (held alone) or 1/2; u^(k)(1) = 1 and u^(k)(1/2) = 1/2,
         so the three groups of rows coincide.
  same:  the empty flow f is split by the B of the next task into f^B, with conservation per class
         sum_in f^B - sum_out f^B = lambda o^B_n - s^B_n, s^B_n >= 0, sum_B s^B_n = lambda d_n (a vehicle that has unloaded
         leaves from the destination, and the linear programme chooses to which class of origins it goes); the occupancy
         coefficients of f^B use the width B. Under any dispatch the empty runs that fetch tasks of width B are a feasible
         decomposition of f^B, so this is still an upper bound; with the empty vehicle's width for every class (same_t1 with
         width = platform width) it has the same solution as the existing linear programme.
occupancy/ and bounds/ are not changed: the representations enter through the subclasses here. The simulation covers only the
main mix, whole roads (no sectioning) and finite demand (asserted).

Run:   python experiments/required_fleet.py <config.json> --out <dir>     (runner.compute passes the points of scenarios with
occ_rep here, the conserved mapping to ablation_conserved). system_bound(net, p, rep) gives the system bound of a
representation.
Inputs: the network and parameters of the configuration (occ_rep in the scenario). Output: the point of runner.compute (with
daily readings when daily_stats is set).
"""
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import bmat, csr_matrix, hstack, identity

from bounds.system import SystemT1, capacity_vec, empty_block, price_list, scenario_mix, stop_ret
from occupancy.bound import DFF_K, _loads
from occupancy.network import Net
from occupancy.sim import Sim

REPS = ("count", "same")


class CountNet(Net):
    """(a) A vehicle count per road: the occupancy on a road depends only on the road width, not on loaded or empty or the
    block width."""

    def demand(self, res, b, loaded, p, cap):
        if res[0] != "r":
            return Net.demand(res, b, loaded, p, cap)
        C, c = cap[res], p["lat_clear"]
        return C if C - c < max(p["block_w"][1], p["plat_w"]) + p["plat_w"] + c else C / 2


class SameSim(Sim):
    """(b) Loaded and empty the same width: an empty run takes the occupancy of the task's B (the b of Net.leg is B instead of
    the platform width)."""

    def legs(self, v, task):
        so, sd, d_so, d_sd, _, L, tail_l, clr = super().legs(v, task)
        p = self.p
        E = self.net.leg(v.loc, so["node"], p["v_empty"], p["plat_len"], task["B"], False, p, self.cap, self.size, so["res"], d_so, True)
        return so, sd, d_so, d_sd, E, L, tail_l, clr


def same_t1(net, p, cap, size, abar, rows, dummy, ret=None, top=3, width=None):
    """T1' of (b) (module docstring). abar: a-bar of the loaded part (the row order of bound._system); rows = [(weight, origin
    node, destination node, B)] (the share of each task); dummy: any task (used only to build the empty part,
    bounds.system.empty_block); width(B): the empty vehicle's width used for f^B, B by default (the platform width gives the
    existing linear programme, for checking). Returns (lambda, price list)."""
    Bs = sorted({B for *_, B in rows})
    blk = [empty_block(net, dict(p, plat_w=width(B) if width else B), cap, size, dummy, ret) for B in Bs]
    Aeq, res_list = blk[0][0], blk[0][2]
    nodes = sorted({x for a in net.arc for x in a[:2]})              # the node order of bound._system
    ni = {n: i for i, n in enumerate(nodes)}
    oB, dn = {B: np.zeros(len(nodes)) for B in Bs}, np.zeros(len(nodes))
    for w, o, d, B in rows:
        oB[B][ni[o]] += w
        dn[ni[d]] += w
    dest = np.nonzero(dn)[0]
    nd, m = len(dest), len(Bs)
    S = csr_matrix((np.ones(nd), (dest, np.arange(nd))), shape=(len(nodes), nd))
    col = lambda v: csr_matrix(v.reshape(-1, 1))
    # variables [lambda | f^B (one set of arcs per class) | s^B (one set of destinations per class)]; equalities: conservation
    # per class, and sum_B s^B = lambda d at each destination
    E = bmat([[col(-oB[B])] + [Aeq if i == j else None for i in range(m)] + [S if i == j else None for i in range(m)] for j, B in enumerate(Bs)]
             + [[col(-dn[dest])] + [None] * m + [identity(nd, format="csr")] * m], format="csr")
    A = hstack([col(abar.reshape(-1))] + [b[1] for b in blk] + [csr_matrix((blk[0][1].shape[0], m * nd))]).tocsr()
    H = capacity_vec(p, res_list)
    c = np.zeros(A.shape[1])
    c[0] = -1.0
    lp = linprog(c, A_ub=A, b_ub=H, A_eq=E, b_eq=np.zeros(E.shape[0]), bounds=[(0, None)] * A.shape[1], method="highs")
    if lp.status != 0:
        raise RuntimeError(f"linear programme of T1' with loaded and empty the same width failed: {lp.message}")
    return float(lp.x[0]), price_list(net, res_list, -lp.ineqlin.marginals, A @ lp.x, H, top)


def same_window(net, p, sim, width=None):
    """The per-point lambda_W^N of (b): the mix of the tasks completed in the statistics window (as runner.t1_window ->
    bound.bounds_window; the main mix has no cranes and lifts, and the tail of the empty run is not cut)."""
    assert not p.get("crane_n")
    win = [r for r in sim.done if sim.t0m <= r["done"] <= sim.H]
    if not win:
        return None
    cap, size = net.capacities(p)
    res_list = sorted(cap)
    nodes = sorted({x for a in net.arc for x in a[:2]})
    abar, _, _ = _loads(net, p, cap, size, win, True, {r: i for i, r in enumerate(res_list)}, {n: i for i, n in enumerate(nodes)}, DFF_K)
    rows = [(1 / len(win), net.stops[t["o"]]["node"], net.stops[t["d"]]["node"], t["B"]) for t in win]
    return same_t1(net, p, cap, size, abar, rows, win[0], None, width=width)[0]


def system_bound(net, p, rep, top=3, width=None):
    """System T1' (Eq. (5), the tail of the empty run cut): rep = None / "count" / "efree" through SystemT1 (the demand of net
    and the empty_rule of p fix the occupancy), "same" through same_t1. Returns (lambda^S, price list, seconds)."""
    t0 = time.time()
    mix, fixed = scenario_mix(net, p)
    assert fixed is None, "main mix only"
    st = SystemT1(net, p, mix)
    ret = stop_ret(net, p["plat_len"] + p["long_clear"])
    if rep != "same":
        v, b = st.value(top=top, ret=ret)
    else:
        rows = [(w * q, net.stops[t["o"]]["node"], net.stops[t["d"]]["node"], B) for w, t, bl in mix if not t.get("lift") for (B, _), q in bl]
        v, b = same_t1(net, p, st.cap, st.size, st.L.abar(st.cap), rows, st.dummy, ret, top, width)
    return v, b, round(time.time() - t0, 1)


NETS = {}


def net_of(cfg, p):
    assert not p.get("retrofit") and not p.get("section"), "only whole roads without road changes"
    k = (cfg["network"], p["occ_rep"])
    if k not in NETS:
        NETS[k] = (CountNet if p["occ_rep"] == "count" else Net)(cfg["network"], p, cfg.get("roles"))
    return NETS[k]


def compute(cfg, key):
    """One simulated point (runner.compute passes the points of scenarios with occ_rep here): finite demand, main mix; the
    per-point lambda_W^N in the same representation."""
    import runner as F
    from occupancy.tasks import demand_tasks
    scen, model, lvl, K, seed = key.split("|")
    K, seed, p = int(K), int(seed), F.scen_params(cfg, scen)
    n = cfg["demand"][lvl]
    assert p["occ_rep"] in REPS and n is not None and not p.get("crane_n") and not p.get("erect_batches") and not p.get("demand_map")
    net = net_of(cfg, p)
    tasks = demand_tasks(net, seed, n, p) if p.get("daily_stats") else None
    sim = (SameSim if p["occ_rep"] == "same" else Sim)(net, p, model, K, seed, n, tasks=tasks).run()
    res = dict(key=key, scen=scen, model=model, level=lvl, K=K, seed=seed, **sim.metrics())
    if p.get("daily_stats"):
        import daily
        res["daily"] = daily.daily(sim, p, tasks)
    if model != "free":
        res["T1_tight_mix"] = same_window(net, p, sim) if p["occ_rep"] == "same" else F.t1_window(net, p, sim, sim.H)
    return res

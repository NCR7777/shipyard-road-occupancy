"""Ablation of the occupancy representation in the main case (the paper's table "ablation of the occupancy representation, main
case"): the representations of ablation.py under the conserved inter-area mapping (demand_map = "conserved",
experiments/conserved_mapping.py). The subclasses and bounds of the representations are those of ablation.py; the mapping is
connected here:
  compute       simulated point: the task stream of conserved_mapping.demand_tasks (as the conserved-mapping branch of
                runner.compute), the rest as ablation.compute (CountNet, SameSim, per-point lambda_W^N in the same
                representation);
  system_bound  system T1' (Eq. (5), the tail of the empty run cut): the task law of conserved_mapping.system_mix (conserved
                mapping) or bounds.system.scenario_mix (other mixes), the rest as ablation.system_bound.
(c) empty vehicles take no width is still empty_rule = "free": its points take the conserved-mapping branch of runner.compute,
not this module.

Run:   python experiments/required_fleet.py <config.json> --out <dir>     (runner.compute passes the points of scenarios with
both occ_rep and demand_map here). system_bound(net, p, rep) gives the system bound of a representation.
Inputs: the network and parameters of the configuration (occ_rep and demand_map = "conserved" in the scenario). Output: the
point of runner.compute (with daily readings when daily_stats is set).
"""
import time

import ablation as A
from bounds.system import SystemT1, scenario_mix, stop_ret
from occupancy.sim import Sim


def mix_of(net, p):
    """Task law of the system bound: conserved_mapping.system_mix for the conserved mapping, scenario_mix otherwise (without the
    fixed part of crane pacing)."""
    if p.get("demand_map") == "conserved":
        import conserved_mapping
        return conserved_mapping.system_mix(net, p)
    mix, fixed = scenario_mix(net, p)
    assert fixed is None, "no crane pacing"
    return mix


def system_bound(net, p, rep, top=3, width=None):
    """As ablation.system_bound, with the task law of mix_of. Returns (lambda^S, price list, seconds)."""
    t0 = time.time()
    mix = mix_of(net, p)
    st = SystemT1(net, p, mix)
    ret = stop_ret(net, p["plat_len"] + p["long_clear"])
    if rep != "same":
        v, b = st.value(top=top, ret=ret)
    else:
        rows = [(w * q, net.stops[t["o"]]["node"], net.stops[t["d"]]["node"], B) for w, t, bl in mix if not t.get("lift") for (B, _), q in bl]
        v, b = A.same_t1(net, p, st.cap, st.size, st.L.abar(st.cap), rows, st.dummy, ret, top, width)
    return v, b, round(time.time() - t0, 1)


def compute(cfg, key):
    """One simulated point (runner.compute passes the points of scenarios with both occ_rep and demand_map here): conserved
    mapping, finite demand; the per-point lambda_W^N in the same representation."""
    import runner as F
    import conserved_mapping
    scen, model, lvl, K, seed = key.split("|")
    K, seed, p = int(K), int(seed), F.scen_params(cfg, scen)
    n = cfg["demand"][lvl]
    assert p["occ_rep"] in A.REPS and n is not None
    conserved_mapping.check(p)
    net = A.net_of(cfg, p)
    tasks = conserved_mapping.demand_tasks(net, seed, n, p)
    sim = (A.SameSim if p["occ_rep"] == "same" else Sim)(net, p, model, K, seed, n, tasks=tasks).run()
    res = dict(key=key, scen=scen, model=model, level=lvl, K=K, seed=seed, **sim.metrics())
    if p.get("daily_stats"):
        import daily
        res["daily"] = daily.daily(sim, p, tasks)
    if model != "free":
        res["T1_tight_mix"] = A.same_window(net, p, sim) if p["occ_rep"] == "same" else F.t1_window(net, p, sim, sim.H)
    return res

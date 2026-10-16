"""One unloading bay plus one off-road waiting place at each dock, under safe release (supplementary material, the dock-mouth
table: the column "one bay and one waiting place").

In the occupancy model the dock-side waiting place (road change dock_wait) serves only the reservation rules; the stepwise
rules still wait on the road. In the c places of a dock-side off-road bay (dock_bay) a vehicle may wait for the crane or unload,
several at once as the cranes allow. Here a dock_bay of capacity 2 plus an unloading-spot resource ('b', dock, 'spot') of
capacity bay_spots = 1 stands for "one unloading bay + one off-road waiting place":
  a P6 vehicle to the dock drives into the off-road places (2) and then, at the arrival decision point, requests the unloading
  spot together with the crane; it starts unloading only when it holds the spot, and when unloading is done it gives up the
  spot together with its place (after the tail clearance). A second vehicle waits in the other place, off the dock road.
At a dock with one crane this equals dock_bay of capacity 2; at a dock with two cranes the two vehicles cannot unload at once.
Parameter: bay_spots (an integer) in the scenario. Only safe release of the stepwise rules (asserted); without the key every
point takes the ordinary code (runner.compute passes points here only when the scenario has bay_spots). occupancy/ is not
changed: the subclass is here.

Run:   python experiments/runner.py <config.json> --out <dir>     (runner.compute passes the points with bay_spots here)
Inputs: the network and parameters of the configuration (retrofit = [dock_bay, cap 2], bay_spots = 1). Output: the point of
runner.compute.
"""
from occupancy.params import DOCK_CLASS
from occupancy.sim import Sim


def spot(z):
    return ("b", z, "spot")


class SpotSim(Sim):
    def __init__(self, net, p, mode, K, seed, level_n, **kw):
        assert mode == "segment_safe" and net.dock_bays, (mode, "the unloading spot needs safe release and dock-side bays")
        super().__init__(net, p, mode, K, seed, level_n, **kw)
        for z in net.by_cls[DOCK_CLASS]:
            self.cap[spot(z)] = float(p["bay_spots"])

    def legs(self, v, task):
        so, sd, d_so, d_sd, E, L, tail_l, clr = super().legs(v, task)
        # at the arrival decision point (before unloading) the unloading spot is requested with the crane; the task logic
        # releases it
        if sd["res"] in self.net.dock_bays:
            L = L[:2] + (L[2] + [(spot(sd["ent"]), 1.0, L[3], None, 0.0)], L[3])
        return so, sd, d_so, d_sd, E, L, tail_l, clr

    def on_handled(self, t, k):
        v = self.vs[k]
        if v.phase == "L" and spot(v.sd["ent"]) in v.hold:   # unloading done: spot and place freed after the tail clearance
            self.free_later(v, t + v.clr, spot(v.sd["ent"]))
        super().on_handled(t, k)


def compute(cfg, key):
    """One simulated point (the simulation branch of runner.compute, with SpotSim)."""
    import runner as F
    scen, model, lvl, K, seed = key.split("|")
    K, seed, p = int(K), int(seed), F.scen_params(cfg, scen)
    net, _ = F.net_for(cfg, p)
    sim = SpotSim(net, p, model, K, seed, cfg["demand"][lvl]).run()
    res = dict(key=key, scen=scen, model=model, level=lvl, K=K, seed=seed, **sim.metrics())
    res["T1_tight_mix"] = F.t1_window(net, p, sim, sim.H)
    if cfg["demand"][lvl] is None:
        res["steady"] = F.steady(net, p, sim, res["T1_tight_mix"])
    return res

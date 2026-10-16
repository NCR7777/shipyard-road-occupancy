"""Interventions on the binding element of the main case (inter-area mapping with erection flow conserved, deterministic
handling of 10 min, 16 h): the scenarios of the intervention study of the paper and the network class of the door retrofit.

Scenarios (p["retrofit"], the road changes of occupancy.network; without one a scenario equals the main case item by item):
  Okpo w13 / w14   only the through corridor road010-6 (road_trace_044, 12 m, 168 m) is widened to 13 m / 14 m (widen "to").
                   The junctions at its two ends take the widest adjacent road as their size, already 15 m and 18 m, so the
                   widening does not change them;
  Yantai door1     entrance001 of the paint shop building014 (node_3ae8b0e5..., binding in the system bound, utilisation
                   1.000) becomes a door for passing only, with 2 loading bays behind it: loading and unloading move behind
                   the door, and the door (entrance node) is held only while a vehicle passes, (D + vehicle length +
                   longitudinal clearance)/v; the other door (building014-entrance1, 0.998) is unchanged;
  Yantai door2     both doors for passing only, 2 in-door bays behind each.
                   This is the door change of occupancy.network ("door") acting per door, with the number of in-door bays as
                   a parameter (door_stop, DoorNet of this module). The door change as it is (in-door bays = number of
                   doors) only moves loading and unloading from the doorway to behind it: the binding element becomes the
                   in-door bay and the bound does not move (door1_b1, door2_b1, two scenarios of the bound only). The
                   occupancy of a door is almost all 10-min handling, so what limits is the number of bays, not the doorway.
                   Two bays per door make the bays no longer binding (once the binding element has moved elsewhere the
                   in-door bays are used about 0.6).
Neither the door change nor the widening changes the list of stops, so under one seed the task streams are the same task by
task (occupancy.tasks: first the entity is drawn, then one of its stops with equal probability) and the scenarios are paired
by task sequence. Adding a parallel door (the "entrance" change) would add a stop and change the equal-probability draw, so
the streams could not be paired; it is not used.
door_stop takes effect only where a scenario gives it (runner.net_for picks DoorNet by the scenario); every other scenario
uses occupancy.network.Net, and occupancy/ is unchanged.

The other intervention modules import this one: intervention_bounds.py (system bounds), intervention_runs.py (saturation,
required fleet and 30-day batches), intervention_dispatch.py (offline dispatch), intervention_subnet.py (sub-network) and
intervention_diag.py.
"""
from occupancy import params as P
from occupancy.network import Net

CORRIDOR = "road_trace_044"                                   # road010-6
PAINT = "facility_trace_014"                                  # building014, paint shop (inferred)
DOOR1 = "node_3ae8b0e5-3951-4995-9e8f-c6428b5a7d13"           # entrance001 (binding in the system bound, 1.000)
DOOR2 = "node_42c2c0ab-f07e-48ca-aa97-9b3ea59ebb28"           # building014-entrance1 (0.998)
SCEN = {"yupu": {"base": [], "w13": [dict(kind="widen", roads=[CORRIDOR], to=13.0)], "w14": [dict(kind="widen", roads=[CORRIDOR], to=14.0)]},
        "yantai": {"base": [], "door1": [dict(kind="door_stop", stops=[DOOR1], bays=2)], "door2": [dict(kind="door_stop", stops=[DOOR1, DOOR2], bays=2)]}}
BOUND_ONLY = {"yantai": {"door1_b1": [dict(kind="door_stop", stops=[DOOR1], bays=1)], "door2_b1": [dict(kind="door", facilities=[PAINT])]}}
LABEL = {"base": "current", "w13": "corridor widened to 13 m", "w14": "corridor widened to 14 m",
         "door1": "one door relaxed (entrance001, 2 in-door bays)", "door2": "two doors relaxed (2 in-door bays each)",
         "door1_b1": "entrance001: loading moved behind the door (1 in-door bay)",
         "door2_b1": "both doors: loading moved behind the doors (door change, 2 in-door bays in all)"}


def retro(yard, scen):
    return {**SCEN[yard], **BOUND_ONLY.get(yard, {})}[scen]


def params(yard, scen, **over):
    """Parameters of the main case (conserved mapping) plus the road changes of the scenario (the base scenario gets no
    retrofit key and equals the parameters of the main case item by item)."""
    p = P.scenario("main", demand_map="conserved", **over)
    return dict(p, retrofit=retro(yard, scen)) if retro(yard, scen) else p


class DoorNet(Net):
    """The door change acting per door (door_stop): the given doors are for passing only, and each of them has its own in-door
    bay resource (capacity bays). The other road changes as in Net."""

    def _retrofit(self):
        super()._retrofit()
        for s in self.retro:
            if s["kind"] == "door_stop":
                for sid in s["stops"]:
                    st = self.stops[sid]
                    assert st["mode"] == "node", sid
                    st.update(mode="door", res=("b", st["ent"], f"in:{sid}"))
                    self.doors.add(st["node"])
                    self.bay_cap[st["res"]] = float(s["bays"])


def net_class(p):
    return DoorNet if any(s.get("kind") == "door_stop" for s in p.get("retrofit") or []) else Net


def net(yard, p):
    Y = P.YARDS[yard]
    return net_class(p)(Y["network"], p, Y["roles"])

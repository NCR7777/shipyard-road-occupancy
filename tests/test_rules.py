"""Traffic rules beyond the basic ones: the U-turn at a stop under the direction lock, the split of reservation waits into
crane and road causes, unambiguous resource names, full-path reservation and safe release on a single-lane corridor with
tasks in both directions, and the differences between the two Okpo revisions."""
import re

import pytest

from helpers import Sim, check_capacity, corridor_net, crane_p, run_with, sectioned_corridor, short, task, yard_net
from occupancy import params as P
from occupancy.bound import res_name


def net(key, section=None):
    """Yard network (cached); section = "stops" sections the dock roads at the stops."""
    return yard_net(key) if section is None else yard_net(key, section=section)


# ================================================================ network revisions
def test_okpo_revision_diff():
    """Okpo revision 1825 -> 1833: three more roads (road001, road002, road005, 12 m wide, each from a former dead end to a
    new dead end, the spurs to the floating docks); their 3 new nodes are not junction resources; the 3 floating docks are
    zones of class floating_dock and not origin or destination entities (stops and entities are identical); the end points of
    three roads moved and their lengths changed (road_trace_031 by +30.7 m); all other roads, capacities and junction
    resources are the same. The new roads lie on no shortest path between stops."""
    a, b = net("yupu_r1825"), net("yupu")
    added = set(b.road) - set(a.road)
    assert added == {"road_trace_710", "road_trace_718", "road_trace_719"} and not set(a.road) - set(b.road)
    assert {b.road[r]["name"] for r in added} == {"road001", "road002", "road005"} and all(b.road[r]["W"] == 12.0 for r in added)
    assert a.stops == b.stops and a.ents == b.ents and a.node_res == b.node_res
    fd = [z for z, x in b.D["zones"].items() if x["classId"] == "floating_dock"]
    assert len(fd) == 3 and not set(fd) & set(b.ents)
    changed = {r for r in a.road if abs(a.road[r]["L"] - b.road[r]["L"]) > 1e-9}
    assert changed == {"road_trace_016", "road_trace_017", "road_trace_031"}
    assert b.road["road_trace_031"]["L"] - a.road["road_trace_031"]["L"] == pytest.approx(30.72, abs=0.01)
    ca, cb = a.capacities(P.scenario("main"))[0], b.capacities(P.scenario("main"))[0]
    assert all(ca[r] == cb[r] for r in ca) and set(cb) - set(ca) == {("r", r) for r in added}
    nodes = sorted({s["node"] for s in b.stops.values()})
    for u in nodes[::7]:
        for v in nodes[::11]:
            if u != v:
                assert not {b.arc[x][3] for x in b.path(u, v)} & added


# ================================================================ direction lock: U-turn at a stop
def test_lock_uturn_reverse():
    """Sectioned corridor: X drives east to the buffer in the middle, Y follows east (same direction, the lock lets it);
    X unloads and gets a task back to A (west). The U-turn is treated as an entry under the direction lock: with Y oncoming
    on the road and no room for both, X waits at the stop (in section r1@1, where the buffer is) and does not enter r1@0;
    the deadlock check counts the lock wait in the wait cycle as soon as X requests the U-turn (t = 1,992 s), and the
    reference rule recovers. Whole, the same instance does not deadlock. entering returns the section in the reverse
    direction on a U-turn."""
    whole, sec = sectioned_corridor()
    mk = lambda: [task(0, "A", "zbuf#0", 0.0), task(1, "A", "B", 10.0), task(2, "zbuf#0", "A", 1000.0)]

    class Watch(Sim):
        seen = []

        def request(self, v, t):
            if self.lock and v.id == 0 and v.task["id"] == 2 and v.phase == "L" and v.step == 0:
                self.seen.append((t, self.entering(v)))
            super().request(v, t)
    sim = Watch(sec, short(section="stops"), "segment", 2, 1, None, record=True, tasks=mk())
    for v in sim.vs:
        v.loc = "A"
    sim.run()
    d0 = sim.deadlocks[0]
    assert d0["t"] == pytest.approx(1992.0) and d0["res"] == {("r", "r1@1")}
    assert sim.seen[0][1] == [("r1", False, 10.0, ("r", "r1@1"))]          # 9 m single lane: a loaded vehicle holds it alone (10)
    assert not [x for x in sim.log if x[4] == 0 and x[0] == ("r", "r1@0") and 1392 < x[1] < 1992]     # X never entered r1@0 before the deadlock
    assert len(sim.done) == 3
    check_capacity(sim)
    w = run_with(whole, "segment", mk(), ["A", "A"])
    assert not w.deadlocks and len(w.done) == 3


# ================================================================ reservation waits split into crane and road causes
def test_split_blk_hand():
    """split_blk: the successive departure shifts are laid over the waiting spans in turn; those caused by the crane are not
    counted as road waits."""
    c, r = ("c", "z"), ("r", "x")
    assert Sim.split_blk([(0, 10)], [(r, 4), (c, 6)]) == [(0, 4)]
    assert Sim.split_blk([(0, 10)], [(c, 3), (r, 7)]) == [(3, 10)]
    assert Sim.split_blk([(0, 5), (20, 25)], [(r, 3), (c, 4), (r, 3)]) == [(0, 3), (22, 25)]
    assert Sim.split_blk([], [(r, 3)]) == []


def test_reserve_wait_split_crane_road():
    """Crane pacing at rho = 0.99, full-path reservation: in the "waiting to depart" span of every P6 the part caused by roads
    does not exceed the total wait, the difference is caused by the crane, and both causes occur; the "vehicle waiting for a
    road" part of crane starvation does not exceed the "blocked" part. For the stepwise family "blocked" never includes
    waiting for the crane, so the two are equal."""
    p = crane_p("yupu", n6=15, days=2, warm=1)
    sim = Sim(net("yupu"), p, "reserve", 25, 102, None).run()
    tot = road = 0.0
    for r in sim.done:
        if r["flow"] == "P6" and r["blk"]:
            b = sum(e - s for s, e in r["blk"])
            rd = sum(e - s for s, e in r.get("blk_road") or [])
            assert rd <= b + 1e-6
            tot, road = tot + b, road + rd
    assert tot > road > 0
    m = sim.metrics()
    for x in m["starve"]["by_dock"].values():
        assert 0.0 <= x["blocked_road"] <= x["blocked"] + 1e-12 <= x["starve"] + 1e-12
    seg = Sim(net("yupu"), p, "segment_safe", 25, 102, None).run().metrics()
    for x in seg["starve"]["by_dock"].values():
        assert x["blocked_road"] == pytest.approx(x["blocked"], abs=1e-12)


# ================================================================ resource names
def test_res_name_disambiguated():
    """Yantai has 12 junction resources named "entrance001 node": res_name appends [building·entrance name, short id], so
    they are all different; without the bracket they read "junction entrance001 node". Road names carry no bracket."""
    n = net("yantai")
    same = [x for x in n.node_res if n.D["nodes"][x].get("name") == "entrance001 node"]
    assert len(same) == 12
    names = [res_name(n, ("n", x)) for x in same]
    strip = lambda s: re.sub(r" \[[^\]]*\]", "", s)
    assert len(set(names)) == 12 and {strip(x) for x in names} == {"junction entrance001 node"}
    assert all("building" in x and "entrance" in x.split(" [")[1] for x in names)
    r = next(iter(n.road))
    assert res_name(n, ("r", r)) == strip(res_name(n, ("r", r))) == f"{n.road[r]['name']} (W={n.road[r]['W'] if n.road[r]['W'] is not None else 'unrestricted'}, {n.road[r]['L']:.0f} m)"


# ================================================================ single-lane corridor, tasks in both directions
def test_corridor_reserve_and_safe_release():
    """Single-lane corridor A - B, task 0 (vehicle at B, B -> A, released at 0 s) and task 1 (vehicle at A, A -> B, released
    at 1 s), buildings at both ends. Full-path reservation: task 0 books first (loading at B 0-600 s, corridor 600-900 s,
    unloading at A 900-1,500 s); task 1 would hold A from loading until departure, so its loading is shifted and it departs
    as soon as it has loaded. Safe release: task 1 cannot load (loading holds A while it still needs the corridor, and task 0
    needs A), so it loads after task 0 has unloaded (1,506 s) and completes at 3,006 s. Neither rule deadlocks or exceeds a
    capacity."""
    net_ = corridor_net()
    mk = lambda: [task(0, "B", "A", 0.0), task(1, "A", "B", 1.0)]
    out = {}
    for mode in ("reserve", "segment_safe"):
        sim = run_with(net_, mode, mk(), ["B", "A"])
        assert len(sim.done) == 2 and not sim.deadlocks
        check_capacity(sim)
        out[mode] = {r["id"]: r for r in sim.done}
    r, s = out["reserve"], out["segment_safe"]
    assert r[0]["done"] == pytest.approx(1500.0)
    assert r[1]["plan"]["t2"] - r[1]["plan"]["a"] == pytest.approx(600.0)   # departs on loading, loading itself is shifted
    assert s[1]["done"] == pytest.approx(3006.375)

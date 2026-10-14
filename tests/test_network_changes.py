"""Road changes (retrofits: widen, entrance, dock_bay, dock_wait, passing, blocks, door): the network transformations
(connectivity, capacities, stops) and their consistency in simulation, T1' and T1''; the empty-run tail cap on sectioned
networks; the synthetic dock mouth (experiments/dock_mouth.py): dock frontage from an outline, crane-paced P6, and a dock-side
bay letting a vehicle that waits for the crane leave the dock road."""
import math

import pytest

import dock_mouth as DM
from bounds.route_free import RouteFree, ret_road
from bounds.system import SystemT1, sampled_mix, scenario_mix, stop_ret, t1_times, union
from helpers import Sim, check_capacity, load_net, make_map, short, yard_net
from occupancy import params as P
from occupancy.bound import _system, bounds, task_windows
from occupancy.tasks import TaskStream

LE = P.DEFAULTS["plat_len"] + P.DEFAULTS["long_clear"]          # empty vehicle length + longitudinal clearance = 21.25 m
YP161, YP0106, YP026 = "road_trace_027", "road_trace_044", "road_3477f15b-70d4-450e-ba80-a35d6273db92"
YT014 = "facility_trace_014"
_W = lambda roads, **kw: [dict(kind="widen", roads=list(roads), **kw)]
RETRO = {"yupu": dict(bay1=[dict(kind="dock_bay", cap=1)], bay2=[dict(kind="dock_bay", cap=2)], pass161=[dict(kind="passing", road=YP161)],
                      w161p1=_W([YP161], add=1.0), w161to12=_W([YP161], to=12.0), w2p3=_W([YP0106, YP026], add=3.0)),
         "yantai": dict(door=[dict(kind="door", facilities=[YT014])], ent=[dict(kind="entrance", facility=YT014)],
                        entdoor=[dict(kind="entrance", facility=YT014), dict(kind="door", facilities=[YT014])])}


def net(yard, section=None, retrofit=None):
    """Network of the paper (Okpo revision 1833, Yantai revision 396) with sectioning and road changes (cached)."""
    over = {k: v for k, v in dict(section=section, retrofit=retrofit).items() if v is not None}
    return yard_net(yard, **over)


def pp(**kw):
    return P.scenario("main", **kw)


def sample(n, p, seed, N):
    s = TaskStream(n, seed, p)
    return [s.next(0.0) for _ in range(N)]


def test_names_of_the_changed_roads():
    """The roads and the building named by the road changes."""
    y, t = net("yupu"), net("yantai")
    assert (y.road[YP161]["name"], y.road[YP0106]["name"], y.road[YP026]["name"]) == ("road161", "road010-6", "road026")
    assert y.road[YP161]["W"] == 9.0 and round(y.road[YP161]["L"]) == 1095
    assert t.ents[YT014]["name"].startswith("building014")


# ================================================================ empty-run tail cap on sectioned networks
def test_stop_ret_whole_equals_cuts_and_sections():
    """The resources carrying stops are found from the out-arcs of stop nodes. On whole networks this equals the roads of
    net.cuts arc by arc and value by value; on sectioned networks it picks the sections that contain the stops: on Okpo both
    ends of such sections are section boundaries (no cap), on Yantai two sections end at a road end, with a way back of at
    least 218 m."""
    for yard in ("yupu", "yantai"):
        n = net(yard)
        old = {a: ret_road(n, a, 35.0) for a, x in enumerate(n.arc) if not x[1].startswith("S|") and x[3] in n.cuts}
        assert stop_ret(n, 35.0) == old
    s = net("yantai", "stops")
    sec = {a: r for a, r in stop_ret(s, 1000.0).items() if "@" in s.arc[a][3]}
    assert len(sec) == 2 and min(sec.values()) >= 218.0
    y = net("yupu", "stops")
    assert not [a for a in stop_ret(y, 1000.0) if "@" in y.arc[a][3]]


def test_stop_ret_short_loop_on_section():
    """A long road (300 m, sectioned) whose last section carries a stop and ends at a road end, beyond which a triangle of 5 m
    sides loops back onto the same section after 15 m. Looking only at the road ids of net.cuts misses this arc; stop_ret
    finds it and caps the empty tail of the empty-run row at 15 m: coefficient (L + 15)/v, all other arcs as without cap."""
    h = math.sqrt(3) / 2 * 5
    m = make_map(dict(A=(-20, 0), X=(0, 0), Y=(300, 0), Z=(305, 0), W=(302.5, h)),
                 [("A", "X"), ("X", "Y"), ("Y", "Z"), ("Z", "W"), ("W", "Y")],
                 {"fA": ("A", "assembly_workshop")}, width={"r0": 31.0, "r1": 9.0, "r2": 9.0, "r3": 9.0, "r4": 9.0},
                 zones={"zbuf": ("buffer", "r1", (275.0,))})
    n = load_net(m, section="stops")
    assert n.sec["r1"] == ["r1@0", "r1@1"] and n.stops["zbuf#0"]["res"] == ("r", "r1@1")
    a = next(i for i, x in enumerate(n.arc) if x[3] == "r1@1" and x[1] == "Y")
    ret = stop_ret(n, 50.0)
    assert ret[a] == pytest.approx(15.0, abs=1e-6) and a not in {i for i, x in enumerate(n.arc) if x[3] in n.cuts}
    p = pp(section="stops")
    cap, size = n.capacities(p)
    dummy = dict(id=0, flow="T", o="A", d="zbuf#0", B=6.0, Lv=20.0)
    _, A0, res, *_ = _system(n, p, cap, size, [dummy], True)
    _, A1, _, *_ = _system(n, p, cap, size, [dummy], True, ret=ret)
    D = (A0 - A1).tocsc()
    assert {j - 1 for j in range(1, D.shape[1]) if D[:, j].nnz} == {b for b, r in ret.items() if r < LE} == {a}
    i = res.index(("r", "r1@1"))
    v = p["v_empty"]
    assert A1[i, 1 + a] == pytest.approx((n.arc[a][2] + 15.0) / v * A0[i, 1 + a] / ((n.arc[a][2] + LE) / v), rel=1e-12)


@pytest.mark.parametrize("yard,value", [("yupu", 1869.8933406513631), ("yantai", 1297.888384763784)])
def test_system_T1_sectioned_unchanged_by_truncation(yard, value):
    """System T1' of the sectioned main scenario (Okpo 1,869.89, Yantai 1,297.89) is the same with the empty tails capped and
    without; so is the sampled T1' (seed 101)."""
    p = pp(section="stops")
    n = net(yard, "stops")
    st = SystemT1(n, p, *scenario_mix(n, p))
    ret = stop_ret(n, LE)
    assert st.value()[0] == pytest.approx(value, rel=1e-12) and st.value(ret=ret)[0] == pytest.approx(value, rel=1e-12)
    ts = sample(n, p, 101, 2000)
    assert bounds(n, p, ts, tight_only=True, ret=ret)["network_tight"] == pytest.approx(bounds(n, p, ts, tight_only=True)["network_tight"], rel=1e-12)


def test_route_free_builds_on_sections():
    """RouteFree builds on sectioned networks; the tail of pulling into a bay is counted on the section that contains the
    stop; on sectioned Yantai T1' <= R <= T1'' (sampled mix)."""
    for yard in ("yupu", "yantai"):
        n = net(yard, "stops")
        p = pp(section="stops")
        rf = RouteFree(n, p, sampled_mix(sample(n, p, 101, 200)))
        pulls = [s for s, v in n.stops.items() if v["mode"] == "pull_in" and v["node"].split("|")[1] in n.sec]
        assert pulls
        for s in pulls:
            node = n.stops[s]["node"]
            t = dict(id=0, flow="T", o=next(iter(n.stops)), d=s, B=6.0, Lv=20.0)
            seg = n.arc[rf.in_arcs[node][0]][3]
            assert seg.startswith(node.split("|")[1] + "@") and n.road[seg]["s0"] < float(node.split("|")[2]) < n.road[seg]["s1"]
            assert rf.fixed_times(t, 20.0)[("r", seg)] == pytest.approx((20.0 + p["long_clear"]) / p["v_loaded"])
    n = net("yantai", "stops")
    p = pp(section="stops")
    ts = sample(n, p, 101, 500)
    t1 = bounds(n, p, ts, tight_only=True)["network_tight"]
    out = RouteFree(n, p, sampled_mix(ts)).solve()
    assert t1 <= out["R"] * (1 + 1e-9) <= out["T1pp"] * (1 + 1e-9)


def restored_equals_windows(n, p, N=150, seed=101):
    """With every switch of R restored (the four switches + the tail at section boundaries, btail) the occupancy of each
    task equals the T1' windows (bound.task_windows) resource by resource: the route-independent part (doors and bays
    included) and the arc-by-arc part together are exactly the occupancy of the simulation."""
    ts = sample(n, p, seed, N)
    rf = RouteFree(n, p, sampled_mix(ts))
    f = t1_times(n, p, rf.cap, rf.size)
    k = 0
    for t in ts:
        a = rf.route_times(t, lv_route=True, caps=False, sweep=True, dest=True, btail=True)
        b = f(t)
        for r in set(a) | set(b):
            assert a.get(r, 0.0) == pytest.approx(b.get(r, 0.0), rel=1e-9, abs=1e-9), (t["o"], t["d"], r)
        k += 1
    return k


@pytest.mark.parametrize("yard", ["yupu", "yantai"])
def test_restored_equals_windows_on_sections(yard):
    assert restored_equals_windows(net(yard, "stops"), pp(section="stops")) == 150


# ================================================================ network transformations
def test_retrofit_empty_is_identity():
    """Without road changes (or with an empty list) the network is unchanged: arcs, stops, capacities."""
    for yard in ("yupu", "yantai"):
        a, b = net(yard), net(yard, retrofit=[])
        assert a.arc == b.arc and a.stops == b.stops and a.capacities(pp()) == b.capacities(pp(retrofit=[]))
        assert not a.bay_cap and not a.doors and not a.dock_bays


def test_widen():
    """Widening: a known width is raised by m (or to at least m); sections follow; the size of the end junctions follows the
    junction rule (largest known width of the adjacent roads); unrestricted roads do not change. Widening a road along a dock
    would move the dock stop, which the derived graph (frontage stored at the mapped widths) cannot do: it is refused."""
    sp = RETRO["yupu"]
    a, b = net("yupu", "stops"), net("yupu", "stops", sp["w2p3"])
    for rid in (YP0106, YP026):
        secs = a.sec.get(rid, [rid])
        assert all(b.road[s]["W"] == a.road[s]["W"] + 3.0 for s in secs)
    ca, sa = a.capacities(pp())
    cb, sb = b.capacities(pp())
    assert cb[("r", YP0106)] == ca[("r", YP0106)] + 3.0
    ends = {a.road[YP0106]["a"], a.road[YP0106]["b"]}
    assert all(sb[x] == max(sa[x], b.road[YP0106]["W"]) for x in ends if x in sa)
    assert set(a.road) == set(b.road) and a.arc == b.arc and a.stops == b.stops
    for k in ("w161to12", "w161p1"):
        with pytest.raises(NotImplementedError):
            net("yupu", None, sp[k])


def test_dock_bay():
    """Dock-side off-road bays: the roadside stops of the docks become drive-in bays (capacity 1 or 2, one place per vehicle);
    the dock road is no longer a stop resource; when sectioned the dock roads still get boundaries at the stops. A P6 holds
    the dock road only until its tail has entered the bay (arrival + tail clearance) and the bay until unloading ends."""
    for c in (1, 2):
        n = net("yupu", "stops", [dict(kind="dock_bay", cap=c)])
        cap, _ = n.capacities(pp())
        docks = [s for z in n.by_cls["dry_dock"] for s in n.ents[z]["stops"]]
        assert all(n.stops[s]["mode"] == "pull_in" and n.stops[s]["res"][0] == "b" and cap[n.stops[s]["res"]] == c for s in docks)
        assert n.sec.keys() == net("yupu", "stops").sec.keys() and n.dock_bays == {n.stops[s]["res"] for s in docks}
    n = net("yupu", None, [dict(kind="dock_bay", cap=1)])
    p = pp()
    cap, size = n.capacities(p)
    t = dict(id=0, flow="P6", o=n.ents[n.by_cls["assembly_yard"][0]]["stops"][0], d="zone_trace_019#0", B=8.0, Lv=20.0)
    by, TL = task_windows(n, p, cap, size, t, tight=True)
    tail = (20.0 + p["long_clear"]) / p["v_loaded"]
    assert max(e for _, e, _ in by[("r", YP161)]) == pytest.approx(TL + tail)
    assert max(e for _, e, _ in by[("b", "zone_trace_019", 0)]) == pytest.approx(TL + p["unload_s"])


def test_passing():
    """Passing place: road161 (stops at 104.5, 470.5 and 547.6 m) has its longest stretch 547.6-1,095.2 m; the 40 m wide
    section sits at its middle, 821.4 m, with the equivalent width of an unrestricted road (31 m, two loaded vehicles pass);
    its ends are section boundaries that can only be passed straight on; whole, the road is cut into 3 sections; sectioned,
    two boundaries are added to the existing ones."""
    n = net("yupu", None, RETRO["yupu"]["pass161"])
    secs = n.sec[YP161]
    assert len(secs) == 3 and n.edges[YP161] == [801.4, 841.4]
    mid = n.road[secs[1]]
    assert mid["W"] == P.DEFAULTS["unrestricted_w"] and mid["L"] == pytest.approx(40.0) and n.road[secs[0]]["W"] == 9.0
    assert n.stops["zone_trace_019#0"]["res"] == ("r", secs[0])
    s = net("yupu", "stops", RETRO["yupu"]["pass161"])
    assert set(net("yupu", "stops").edges[YP161]) | {801.4, 841.4} == set(s.edges[YP161])
    b = "S|road_trace_027|801.4"
    assert all(len(n.succ[i]) == 1 and n.arc[n.succ[i][0]][4] == n.arc[i][4] for i, x in enumerate(n.arc) if x[1] == b)


def test_entrance():
    """Another entrance: building 014 gets a copy of its first entrance (a new entrance node at the same place, an access
    road of the same length and width to the same junction, the same turn permissions): 3 stops; the new entrance and every
    other stop are reachable both ways; the new entrance node has the capacity of the original; junction resources are
    unchanged."""
    a, b = net("yantai"), net("yantai", None, RETRO["yantai"]["ent"])
    old = a.ents[YT014]["stops"]
    new = [s for s in b.ents[YT014]["stops"] if s not in old]
    assert len(new) == 1 and b.ents[YT014]["stops"][:2] == old and new[0] == old[0] + "~1"
    r0 = next(r for r in a.incident[old[0]])
    r1 = next(r for r in b.incident[new[0]])
    assert b.road[r1]["L"] == pytest.approx(a.road[r0]["L"]) and b.road[r1]["W"] == a.road[r0]["W"]
    other = lambda n_, r, x: n_.road[r]["b"] if n_.road[r]["a"] == x else n_.road[r]["a"]
    assert other(b, r1, new[0]) == other(a, r0, old[0])
    ca, cb = a.capacities(pp())[0], b.capacities(pp())[0]
    assert cb[("n", new[0])] == ca[("n", old[0])] and set(ca) <= set(cb) and set(cb) - set(ca) == {("n", new[0]), ("r", r1)}
    x = b.stops[new[0]]["node"]
    for s in b.stops.values():
        if s["node"] != x:
            assert math.isfinite(b.dist(x, s["node"])) and math.isfinite(b.dist(s["node"], x))


def test_door():
    """Door granularity: each of the two doors of building 014 is held while a vehicle passes, (D + vehicle length +
    longitudinal clearance)/v (requested at decision point arr on arrival, at decision point 0 on departure); loading and
    unloading take place at the bay inside (capacity 2, one place per vehicle); with another entrance the bay has capacity 3.
    Windows: origin bay [-loading, tail], door [0, passage]; destination door [arrival, arrival + passage]."""
    n = net("yantai", None, RETRO["yantai"]["door"])
    p = pp()
    cap, size = n.capacities(p)
    res = ("b", YT014, "in")
    assert cap[res] == 2.0 and all(n.stops[s]["mode"] == "door" and n.stops[s]["res"] == res for s in n.ents[YT014]["stops"])
    assert n.doors == set(n.ents[YT014]["stops"])
    door = n.ents[YT014]["stops"][0]
    other = next(s for s, v in n.stops.items() if v["ent"] != YT014 and v["mode"] == "node")
    for o, d in ((door, other), (other, door)):
        t = dict(id=0, flow="T", o=o, d=d, B=8.0, Lv=20.0)
        by, TL = task_windows(n, p, cap, size, t, tight=True)
        pas = (size[door] + 20.0 + p["long_clear"]) / p["v_loaded"]
        w = by[("n", door)]
        assert w == [(0.0, pytest.approx(pas), cap[("n", door)])] if o == door else w == [(pytest.approx(TL), pytest.approx(TL + pas), cap[("n", door)])]
        assert (min(s for s, _, _ in by[res]) == -p["load_s"]) if o == door else (max(e for _, e, _ in by[res]) == pytest.approx(TL + p["unload_s"]))
    assert net("yantai", None, RETRO["yantai"]["entdoor"]).capacities(p)[0][res] == 3.0


@pytest.mark.parametrize("yard,spec,section", [("yupu", "bay2", "stops"), ("yupu", "pass161", None), ("yantai", "entdoor", "stops"),
                                               ("yantai", "door", None)])
def test_restored_equals_windows_on_retrofits(yard, spec, section):
    """On changed networks the route-independent part of RouteFree (doors, bays, new entrances) and the arc-by-arc part
    together still equal the occupancy of the simulation (the T1' windows)."""
    n = net(yard, section, RETRO[yard][spec])
    assert restored_equals_windows(n, pp(section=section, retrofit=RETRO[yard][spec]), N=120) == 120


@pytest.mark.parametrize("yard,spec,mode", [("yupu", "bay2", "reserve"), ("yupu", "pass161", "reserve"),
                                            ("yantai", "entdoor", "reserve"), ("yantai", "door", "segment_safe")])
def test_sim_on_retrofits(yard, spec, mode):
    """Half a day on changed networks (K = 30): capacities hold (full-path reservation time by time from the log, safe
    release event by event), no deadlock (under safe release a detected deadlock raises), tasks complete; the bays inside
    the doors and at the docks are used."""
    p = dict(short(section="stops", retrofit=RETRO[yard][spec]), day_s=8 * 3600.0)
    n = net(yard, "stops", RETRO[yard][spec])
    sim = Sim(n, p, mode, 30, 101, None, record=True).run()
    assert len(sim.done) > 20 and not sim.deadlocks
    check_capacity(sim)
    bays = set(n.bay_cap)
    if bays:
        used = {x[0] for x in sim.log} & bays if sim.log else set()
        assert used, "the bays were not used"


def test_blocks_and_passing_at():
    """blocks: the whole road cut evenly into blocks of at most len (road010-6 is 167.6 m long; 50 m -> 4 blocks, boundaries
    at 41.9, 83.8 and 125.7 m, all more than 1 m from its stop). passing with `at` is centred there (road161 near end,
    287.5 m -> 267.5-307.5 m); a passing place over a stop is refused."""
    n = net("yupu", "stops", [dict(kind="blocks", road=YP0106, len=50.0)])
    assert n.edges[YP0106] == [41.9, 83.8, 125.7] and len(n.sec[YP0106]) == 4
    assert all(n.road[s]["W"] == 12.0 for s in n.sec[YP0106])
    m = net("yupu", None, [dict(kind="passing", road=YP161, at=287.5)])
    assert m.edges[YP161] == [267.5, 307.5] and m.road[m.sec[YP161][1]]["W"] == P.DEFAULTS["unrestricted_w"]
    with pytest.raises(AssertionError):
        net("yupu", None, [dict(kind="passing", road=YP161, at=470.5)])


# ================================================================ synthetic dock mouth
@pytest.mark.parametrize("W", DM.WIDTHS)
def test_dock_mouth_network(W):
    """Synthetic dock mouth: the dock frontage derived from the dock outline (mapgraph.dock_frontage: centre line within
    W/2 + 8 m of the outline, sampled every metre). The outline spans 140-160 m at W/2 + 2 m from the centre line, so the
    stretch reaches floor(sqrt((W/2 + 8)^2 - (W/2 + 2)^2)) = floor(sqrt(6 W + 60)) m beyond each end and its middle, the
    roadside stop, is at 150 m. The adjacent area has one drive-in bay at 250 m, the origin area six on rs; the dock has one
    crane; the depot entrance is the vehicles' start."""
    mode, p = DM.params("seg")
    n = DM.make_net(W, p)
    (rid, s0, s1), = n.D["zones"]["zdock"]["frontage"]
    ext = math.floor(math.sqrt(6 * W + 60))
    assert rid == "r1" and (s0, s1) == (140.0 - ext, 160.0 + ext)
    assert n.stops["zdock#0"] == dict(node="S|r1|150.0", res=("r", "r1"), mode="on_road", ent="zdock")
    assert n.stops["zyard#0"]["node"] == "S|r1|250.0" and n.stops["zyard#0"]["mode"] == "pull_in"
    assert n.ents["zsrc"]["stops"] == DM.SRC_IDS and n.crane_counts(p) == {"zdock": 1}
    assert n.road["r1"]["W"] == W and n.road["r1"]["L"] == pytest.approx(DM.L1)
    assert n.capacities(p)[0][("r", "r1")] == W + p["lat_clear"]


def test_dock_mouth_tasks():
    """P6 of the dock mouth are released at the crane pace, n6 = rho x 960 / 190 a day at equal intervals; the background
    stream draws the same sequence for a seed whatever rho or the rule (common random numbers)."""
    for rho in (0.3, 0.99):
        ts = DM.p6_tasks(101, rho)
        n6 = rho * 16 * 3600.0 / (190 * 60.0)
        assert len(ts) == int(DM.DAYS * n6) and all(t["d"] == "zdock#0" and t["o"] in DM.SRC_IDS for t in ts)
        gaps = [b["release"] - a["release"] for a, b in zip(ts, ts[1:])]
        assert all(g == pytest.approx(16 * 3600.0 / n6) for g in gaps)
    a, b = DM.Background(7), DM.Background(7)
    assert [a.next(0.0) for _ in range(30)] == [b.next(1.0) | dict(release=0.0) for _ in range(30)]
    assert DM.union_len([(0, 2), (1, 3), (5, 6)], 0.5, 5.5) == 3.0 and DM.union_len([], 0, 1) == 0.0


def test_dock_bay_frees_dock_road_while_waiting_for_crane():
    """Dock-side off-road bay (stepwise + safe release, synthetic dock mouth, 9 m single-lane dock road): after the first P6
    unloads the crane is busy for another 180 min; the second P6 reaches the dock while the crane is busy. Without the bay
    it waits on the dock road and the background vehicle to the adjacent area has to wait for it; with the bay it waits in
    the bay, the dock road is released on time and the background vehicle passes, finishing much earlier."""
    done = {}
    for rule in ("seg", "seg_bay"):
        mode, p = DM.params(rule, days=1)
        p = dict(p, warmup_days=0)
        n = DM.make_net(9.0, p)
        ts = [dict(id=1, flow="P6", o="zsrc#0", d="zdock#0", B=7.0, Lv=20.0, release=0.0),
              dict(id=2, flow="P6", o="zsrc#1", d="zdock#0", B=7.0, Lv=20.0, release=900.0),
              dict(id=3, flow="BG", o="zsrc#2", d="zyard#0", B=7.0, Lv=20.0, release=1500.0)]
        sim = Sim(n, p, mode, 3, 1, None, record=True, tasks=ts).run()
        assert len(sim.done) == 3 and not sim.deadlocks
        check_capacity(sim)
        done[rule] = {r["id"]: r["done"] for r in sim.done}
    assert done["seg"][3] > done["seg"][1] + 180 * 60 - 1          # the P6 waiting for the crane holds the dock road
    assert done["seg_bay"][3] < done["seg"][3] - 3600               # with the bay the background vehicle does not wait for the crane
    assert done["seg_bay"][2] == pytest.approx(done["seg"][2], abs=600)


def test_dock_wait_reserve_and_bound():
    """Dock-side waiting place (dock_wait): the dock stop stays on the road, one vehicle unloads at a time, with an off-road
    waiting place beside it.
    1. occupancy without waiting is unchanged: the union of the T1' windows of a P6 and the sampled T1' equal those without
       the waiting place;
    2. reservation rules: after the first P6 unloads the crane is busy for another 180 min; the second P6 no longer waits at
       its origin - it departs on its road windows, waits in the waiting place until the crane is free (tc) and then
       unloads; it completes at the same time as without the waiting place (crane-bound) but departs about 3 h earlier; the
       waiting-place window is in the log and within capacity; no stops en route other than the waiting place;
    3. the stepwise family does not use the waiting place: completion times are identical to those without it."""
    ts = lambda: [dict(id=1, flow="P6", o="zsrc#0", d="zdock#0", B=7.0, Lv=20.0, release=0.0),
                  dict(id=2, flow="P6", o="zsrc#1", d="zdock#0", B=7.0, Lv=20.0, release=900.0)]
    out = {}
    for wait in (False, True):
        mode, p = DM.params("res", days=1)
        p = dict(p, warmup_days=0, retrofit=[dict(kind="dock_wait", cap=1)] if wait else [])
        n = DM.make_net(9.0, p)
        cap, size = n.capacities(p)
        assert (("w", "zdock", 0) in cap) == wait
        by, TL = task_windows(n, p, cap, size, ts()[0], tight=True)
        out[(wait, "win")] = {r: union(iv) for r, iv in by.items()}
        out[(wait, "T1")] = bounds(n, p, [dict(t, flow="P6") for t in ts()] * 5, tight_only=True)["network_tight"]
        for m in ("reserve", "segment_safe"):
            sim = Sim(n, p, m, 2, 1, None, record=True, tasks=ts()).run()
            assert len(sim.done) == 2 and not sim.deadlocks
            check_capacity(sim)
            out[(wait, m)] = ({r["id"]: r["done"] for r in sim.done}, {r["id"]: r["plan"] for r in sim.done}, sim)
    assert out[(True, "win")] == pytest.approx(out[(False, "win")]) and out[(True, "T1")] == pytest.approx(out[(False, "T1")], rel=1e-12)
    d0, pl0, _ = out[(False, "reserve")]
    d1, pl1, sim = out[(True, "reserve")]
    assert d1 == pytest.approx(d0)                                      # crane-bound: completion times unchanged
    assert "tc" in pl1[2] and pl1[2]["t2"] < pl0[2]["t2"] - 2 * 3600     # departs about 3 h earlier and waits in the waiting place
    assert any(x[0] == ("w", "zdock", 0) for x in sim.log)
    assert out[(True, "segment_safe")][0] == out[(False, "segment_safe")][0]

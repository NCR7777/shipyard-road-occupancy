"""Model behaviour: the exclusive-occupancy invariant, free-flow consistency, throughput below the network bounds T1 and T1'
(which bind only the occupancy rules), reservation without deadlock or stops en route, stepwise requests reproducing
deadlocks and recovery of a vehicle on the wait cycle, junction holds and turn sweep, hand-computed bounds of a single lane
and a crossroads, dual-feasible functions, task generation of the erection peak and of crane pacing, lifts by gantry cranes,
just-in-time departure, dock-mouth readings and waiting attribution, seed reproducibility, safe release (cross-checked with a
naive reduction), the no-recovery switch, sectioning and the direction lock, crane starvation, the steady-state criterion,
the fleet search and the daily demand."""
import collections
import itertools
import math
import random
from fractions import Fraction

import pytest

import runner as R
import required_fleet as RF
from helpers import (CORRIDOR, CORRIDOR_EDGES, V6, V12, Sim, check_capacity, corridor_net, crane_p, load_config, load_net,
                     make_map, naive_run, no_one_left, random_case, ring, run_with, sectioned_corridor, short, task, yard_net)
from occupancy import params as P
from occupancy.bound import bounds, bounds_window, dff, fleet_bound, share
from occupancy.mapgraph import cover
from occupancy.network import Net, read_graph
from occupancy.tasks import TaskStream, crane_p6, crane_rho, crane_split, crane_tasks, demand_tasks, flow_weights


@pytest.fixture(scope="module")
def okpo():
    """Okpo, revision 1833 (the network of the paper)."""
    return yard_net("yupu")


@pytest.fixture(scope="module")
def yantai():
    """Yantai, revision 396 (the network of the paper)."""
    return yard_net("yantai")


@pytest.fixture(scope="module")
def corridor():
    return corridor_net()


# ================================================================ occupancy invariants and free flow
@pytest.mark.parametrize("mode", ["reserve", "segment"])
def test_exclusive_invariant(okpo, mode):
    """Okpo, K = 25, one day: the occupancy never exceeds a capacity and many exclusive holds are actually checked."""
    sim = Sim(okpo, short(), mode, 25, 101, None, record=True).run()
    assert len(sim.done) > 100
    assert check_capacity(sim) > 100


def test_free_flow_consistency(okpo):
    """Without interaction: completion = dispatch + empty distance / empty speed + loading + loaded distance / loaded speed +
    unloading. Full-path reservation with a single vehicle gives the same."""
    p = short()
    tasks = [dict(TaskStream(okpo, 7, p).next(100.0), release=100.0)]
    for mode in ("free", "reserve"):
        sim = Sim(okpo, p, mode, 1, 7, None, tasks=[dict(t) for t in tasks])
        start = sim.vs[0].loc
        sim.run()
        r = sim.done[0]
        so, sd = okpo.stops[r["o"]]["node"], okpo.stops[r["d"]]["node"]
        expect = 100.0 + okpo.dist(start, so) / p["v_empty"] + p["load_s"] + okpo.dist(so, sd) / p["v_loaded"] + p["unload_s"]
        assert r["done"] == pytest.approx(expect, abs=1e-6), mode
        assert r["wait"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("scen", ["main", "w-1.5", "empty_excl", "erection"])
def test_throughput_below_T1(okpo, scen):
    """Coarse check on Okpo: the throughput of the occupancy rules (reservation, stepwise) is at most min(T1', fleet bound);
    free flow is compared with the fleet bound only. The bounds are those of the task mix completed in the statistics window
    of the run; tolerance K / days of the window (each vehicle may have started at most one task before the window). With a
    single statistics day the tolerance is wide; the tight checks are the hand-computed instances and the reuse instance
    below. T1' does not count the clearance given up at the destination and T1 does, so T1' <= T1 is not guaranteed by
    construction and is not asserted."""
    p = short(scen, days=2, warm=1)
    for mode in ("free", "reserve", "segment"):
        for K in (5, 30):
            sim = Sim(okpo, p, mode, K, 103, None).run()
            m = sim.metrics()
            mix = [r for r in sim.done if sim.t0m <= r["done"] <= sim.H]
            b = bounds(okpo, p, mix)
            fleet = fleet_bound(K, b["cycle_s"], p)
            bound = fleet if mode == "free" else min(b["network_tight"], fleet)
            assert m["throughput"] <= bound + K / 1.0, (mode, K, m["throughput"], bound)


# ================================================================ deadlocks of stepwise requests; reservation
def test_segment_reproduces_head_on_deadlock(corridor):
    """Two vehicles head-on on the single-lane corridor: both load for 10 min, depart together and meet at n2; after the
    manual recovery both complete."""
    sim = run_with(corridor, "segment", [task(0, "A", "B"), task(1, "B", "A")], ["A", "B"])
    assert len(sim.deadlocks) >= 1
    assert sim.deadlocks[0]["size"] == 2 and sim.deadlocks[0]["core"] == 2
    assert sim.deadlocks[0]["t"] == pytest.approx(600 + 250 / V6, abs=1e-6)
    assert len(sim.done) == 2
    check_capacity(sim)


def test_victim_is_on_the_wait_cycle():
    """A stuck vehicle that is not on the wait cycle: v1 (A -> B) and v2 (B -> A) deadlock head-on at n2; v0 from the branch
    entrance C follows v1 and waits for n1-n2, blocked by the cycle but not on it. The recovered vehicle is chosen from the
    cycle {v1, v2} (the lowest id, v1), not the bystander v0 with a lower id. Afterwards v0 takes n1-n2 and meets v2 at n2,
    a new two-vehicle cycle {v0, v2} in which v0 is rightly chosen."""
    pos = dict(CORRIDOR, C=(50, 50))
    net = load_net(make_map(pos, CORRIDOR_EDGES + [("C", "n1")], {"fA": ("A", "assembly_workshop"),
                                                                 "fB": ("B", "paint_workshop"),
                                                                 "fC": ("C", "outfitting_workshop")}))
    tasks = [task(0, "A", "B"), task(1, "B", "A"), task(2, "C", "B", release=50.0)]
    sim = run_with(net, "segment", tasks, ["C", "A", "B"])
    d0 = sim.deadlocks[0]
    assert d0["t"] == pytest.approx(600 + 250 / V6, abs=1e-6)
    assert d0["size"] == 3 and d0["core"] == 2      # 3 stuck, 2 on the cycle
    assert d0["victims"] == [1]
    assert sim.deadlocks[1]["t"] == pytest.approx(750 + 200 / V6, abs=1e-6) and sim.deadlocks[1]["size"] == 2
    assert len(sim.done) == 3
    check_capacity(sim)


def test_reserve_never_deadlocks(corridor, okpo):
    """Reservation waits only at stops, never en route, so it cannot deadlock. Two non-trivial checks:
    1. the actual occupancy of the committed reservation windows stays within every capacity, time by time;
    2. every task takes exactly the free-flow time from the empty departure to the origin and from the loaded departure to
       the end of unloading, i.e. no task waits on any resource en route."""
    sim = run_with(corridor, "reserve", [task(0, "A", "B"), task(1, "B", "A")], ["A", "B"])
    assert len(sim.done) == 2 and not sim.deadlocks
    check_capacity(sim)
    t = sorted(r["done"] for r in sim.done)
    assert t[1] > t[0] + 10 * 60              # the second vehicle departs after the first has passed the whole corridor
    p = short()
    sim = Sim(okpo, p, "reserve", 40, 105, None, record=True).run()
    assert len(sim.done) > 300 and not sim.waiting
    assert check_capacity(sim) > 300
    waited = 0
    for r in sim.done:
        pl = r["plan"]
        so, sd = okpo.stops[r["o"]]["node"], okpo.stops[r["d"]]["node"]
        assert pl["a"] - pl["t1"] == pytest.approx(okpo.dist(pl["start"], so) / p["v_empty"], abs=1e-6)
        assert r["done"] - pl["t2"] == pytest.approx(okpo.dist(so, sd) / p["v_loaded"] + p["unload_s"], abs=1e-6)
        assert pl["t2"] >= pl["a"] + p["load_s"] - 1e-6
        waited += r["wait"] > 1.0
    assert waited > 100                       # many tasks did wait at a stop (occupancy binds), none en route


def test_junction_hold_and_turn_sweep():
    """Crossroads (four arms of 100 m, 15 m wide): two loaded vehicles load together and reach the junction together.
    Free flow: both complete at 600 + 200/v6 + 600 = 1320 s. With occupancy the junction is held alone for
    (15 + 20 + 5)/v6 = 24 s, plus 30 s for a turn of at least 60 degrees: the second vehicle completes at 1344 s when the first
    goes straight on and at 1374 s when it turns left."""
    pos = dict(X=(0, 0), W=(-100, 0), E=(100, 0), S=(0, -100), N=(0, 100))
    fac = {"fW": ("W", "assembly_workshop"), "fS": ("S", "assembly_workshop"),
           "fE": ("E", "paint_workshop"), "fN": ("N", "paint_workshop")}
    net = load_net(make_map(pos, [("X", "W"), ("X", "E"), ("X", "S"), ("X", "N")], fac, width=15.0))
    for d0, second in (("E", 1344.0), ("N", 1374.0)):        # vehicle 0: W -> E straight or W -> N left; vehicle 1: S -> N or S -> E
        d1 = "N" if d0 == "E" else "E"
        for mode, expect in (("free", 1320.0), ("reserve", second), ("segment", second)):
            sim = run_with(net, mode, [task(0, "W", d0), task(1, "S", d1)], ["W", "S"])
            done = sorted(r["done"] for r in sim.done)
            assert done == pytest.approx([1320.0, expect], abs=1e-6), (d0, mode, done)
            assert not sim.deadlocks


# ================================================================ hand-computed bounds
def test_corridor_T1_hand_value_and_free_flow_exceeds_it(corridor):
    """Single lane, all tasks A -> B: the entrance B is held alone for 50/v6 + 600 + (16.25 + 5)/v12 = 636.4 s per task, so
    the network bound is 57600/636.4 = 90.51 per day. Under the occupancy rules vehicles can only go one after another and
    reservation stays below the bound; free flow does not check capacities and with K = 4 exceeds it (about 140 per day):
    T1 does not bind free flow."""
    p = short(days=2, warm=1)
    tasks = [task(i, "A", "B") for i in range(600)]
    b = bounds(corridor, p, tasks)
    assert b["network"] == pytest.approx(57600 / (50 / V6 + 600 + 21.25 / V12), abs=1e-6)
    assert b["network"] == pytest.approx(90.51, abs=0.01)
    thr = {mode: run_with(corridor, mode, [dict(t) for t in tasks], ["A"] * 4, p, record=False).metrics()["throughput"]
           for mode in ("free", "reserve")}
    assert thr["free"] > b["network"] * 1.3
    assert thr["reserve"] <= b["network"]


def test_dff_is_dual_feasible():
    """u^(1), u^(2) are dual-feasible functions: sum x <= 1 implies sum u(x) <= 1; they are non-decreasing and u(x) >= 0. The
    shares are exact rationals, without a tolerance band. All multisets (up to 6 items) on the 1/24 grid, and 20,000 random
    rational sets (floats converted exactly to fractions)."""
    grid = [Fraction(i, 24) for i in range(1, 25)]
    for k in (1, 2):
        for n in range(1, 7):
            for xs in itertools.combinations_with_replacement(grid, n):
                if sum(xs) <= 1:
                    assert sum(dff(x, k) for x in xs) <= 1, (k, xs)
        rng = random.Random(0)
        for _ in range(20000):
            xs, s = [], Fraction(0)
            while True:
                x = Fraction(rng.uniform(0.0, 1.0) ** 2)
                if s + x > 1:
                    break
                xs.append(x)
                s += x
            assert sum(dff(x, k) for x in xs) <= 1, (k, xs)
        ys = [dff(x, k) for x in grid]
        assert ys == sorted(ys) and min(ys) >= 0
    F = Fraction
    assert dff(F(1, 2), 1) == F(1, 2) and dff(F(8, 13), 1) == 1 and dff(F(2, 5), 1) == 0
    assert dff(F(2, 5), 2) == F(1, 2) and dff(F(11, 16), 2) == 1 and dff(F(3, 10), 2) == 0


def test_dff_threshold_sides():
    """Both sides of a threshold (+-1e-10): an item just below is not raised, one just above counts at the next step, one
    exactly on it keeps x. Three items close to 1/3 with sum 1 - 2.6e-10 (k = 2) give sum u <= 1. Through share() the float
    widths are converted exactly: 6.5/19.5 = 1/3, 6.5/13 = 1/2 and 13/19.5 = 2/3 lie exactly on thresholds."""
    e = Fraction(1, 10 ** 10)
    for k, cuts in ((1, (Fraction(1, 2),)), (2, (Fraction(1, 3), Fraction(2, 3)))):
        for c in cuts:
            below, at, above = dff(c - e, k), dff(c, k), dff(c + e, k)
            assert below < at < above and at == c, (k, c, below, at, above)
    assert (dff(Fraction(1, 2) - e, 1), dff(Fraction(1, 2) + e, 1)) == (0, 1)
    assert (dff(Fraction(1, 3) - e, 2), dff(Fraction(1, 3) + e, 2), dff(Fraction(2, 3) - e, 2), dff(Fraction(2, 3) + e, 2)) == \
        (0, Fraction(1, 2), Fraction(1, 2), 1)
    xs = [Fraction(1 / 3 - 3e-10), Fraction(1 / 3 + 3.4e-10), Fraction(1 / 3 - 3e-10)]
    assert sum(xs) <= 1 and sum(dff(x, 2) for x in xs) <= 1
    assert share(6.5, 19.5, 2) == 1 / 3 and share(6.5, 13.0, 1) == 0.5 and share(13.0, 19.5, 2) == 2 / 3
    assert share(10.0, 10.0, 1) == 1.0 and share(12.0, 10.0, 2) == 1.0 and share(8.0, 13.0, 0) == 8.0 / 13.0


def test_T1_tight_corridor_hand_value():
    """Single-lane 12 m corridor A - n1 -(2000 m)- n2 - B, all tasks A -> B, B = 7 m (a loaded vehicle takes 8 of the
    capacity 13; an empty one 6.5). Middle road: loaded window [50/v6, 2050/v6 + 25/v6] = 1215 s, empty 2000/v12 = 600 s
    (T1' adds the tail 21.25/v12 = 6.375 s).
    T1 by width shares: lambda (1215 * 8/13 + 600 * 0.5) <= 57600 -> 54.98 per day;
    T1': 8 + 8 and 8 + 6.5 do not fit, so a loaded vehicle counts 1; two empty ones fit exactly (share 0.5):
    lambda (1215 + 0.5 * 606.375) <= 57600 -> 37.94 per day. The occupancy rules stay below it; free flow exceeds it."""
    pos = dict(A=(0, 0), n1=(50, 0), n2=(2050, 0), B=(2100, 0))
    net = load_net(make_map(pos, [("A", "n1"), ("n1", "n2"), ("n2", "B")],
                            {"fA": ("A", "assembly_workshop"), "fB": ("B", "paint_workshop")}, width=12.0))
    p = short(days=2, warm=1)
    tasks = [dict(task(i, "A", "B"), B=7.0) for i in range(400)]
    b = bounds(net, p, tasks)
    assert b["network"] == pytest.approx(57600 / (1215 * 8 / 13 + 300), abs=1e-6)
    assert b["network_tight"] == pytest.approx(57600 / (1215 + 0.5 * (600 + 21.25 / V12)), abs=1e-6)
    assert b["network_tight"] == pytest.approx(37.94, abs=0.01)
    assert b["binding_tight"][0]["res"] == ["r", "r1"]
    thr = {mode: run_with(net, mode, [dict(t) for t in tasks], ["A"] * 6, p, record=False).metrics()["throughput"]
           for mode in ("free", "reserve", "segment")}
    assert thr["reserve"] <= b["network_tight"] and thr["segment"] <= b["network_tight"]
    assert thr["free"] > b["network_tight"] * 2


def test_T1_tight_crossroads_hand_value():
    """Crossroads with four arms of 1000 m, 15 m wide (capacity 16; a loaded B = 10 takes 11, so two do not fit); empty
    vehicles take no room (empty_free); loading and unloading 60 s each. Tasks in turn W -> E, E -> W, S -> N, N -> S. Each
    arm is held by half of the tasks for 1000/v6 + 25/v6 = 615 s, i.e. 307.5 s per task on average.
    T1: 307.5 * 11/16 = 211.4 s -> 272.46 per day (an entrance node: (75 + 666.375)/4 = 185.3 s per task -> 310.8 per day).
    T1': the arms count as exclusive -> 57600/307.5 = 187.32 per day, binding at the four arms."""
    pos = dict(X=(0, 0), W=(-1000, 0), E=(1000, 0), S=(0, -1000), N=(0, 1000))
    fac = {"fW": ("W", "assembly_workshop"), "fS": ("S", "assembly_workshop"),
           "fE": ("E", "paint_workshop"), "fN": ("N", "paint_workshop")}
    net = load_net(make_map(pos, [("X", "W"), ("X", "E"), ("X", "S"), ("X", "N")], fac, width=15.0))
    p = short("empty_free", days=2, warm=1, load_s=60.0, unload_s=60.0)
    od = [("W", "E"), ("E", "W"), ("S", "N"), ("N", "S")]
    tasks = [task(i, *od[i % 4]) for i in range(800)]
    b = bounds(net, p, tasks)
    assert b["network"] == pytest.approx(57600 / (307.5 * 11 / 16), abs=1e-6)
    assert b["network_tight"] == pytest.approx(57600 / 307.5, abs=1e-6)
    assert tuple(b["binding_tight"][0]["res"]) in {("r", f"r{i}") for i in range(4)}   # four arms bind together (degenerate duals)
    thr = {mode: run_with(net, mode, [dict(t) for t in tasks], ["W", "E", "S", "N"] * 3, p, record=False).metrics()["throughput"]
           for mode in ("free", "reserve", "segment")}
    assert thr["reserve"] <= b["network_tight"] + 12 and thr["segment"] <= b["network_tight"] + 12   # tolerance K / statistics days
    assert thr["free"] > b["network_tight"]


class ReuseSim(Sim):
    """Only dispatch changes: an idle vehicle's next task starts at the stop where it is (after unloading at X it takes
    X -> Y_j, Y_j in turn; after unloading at Y_i it takes Y_i -> X). The traffic rules are unchanged."""
    def __init__(self, net, p, mode, K, locs):
        super().__init__(net, p, mode, K, 1, None, tasks=[])
        for v, loc in zip(self.vs, locs):
            v.loc = loc
        self.node2stop = {s["node"]: sid for sid, s in net.stops.items()}
        self.ys = sorted(sid for sid in net.stops if sid.startswith("zY#"))
        self.rr, self.nid, self.saturated = 0, 0, True

    def dispatch(self, t):
        while self.idle:
            k = self.idle.pop(0)
            here = self.node2stop[self.vs[k].loc]
            d = self.ys[self.rr % len(self.ys)] if here == "X" else "X"
            self.rr += here == "X"
            task_ = dict(id=self.nid, flow="T", o=here, d=d, B=5.5, Lv=20.0, release=t, dispatch=t)
            self.nid += 1
            (self.start_segment if self.mode == "segment" else self.plan)(self.vs[k], task_, t)


def test_T1_tight_reuse_regression():
    """Vehicles reloading at the stop where they unloaded. Entrance X - 10 m road (W = 31) - J (degree 2, not a resource) -
    wide road rY (W = 60) with 4 roadside buffer stops. Row X is actually busy 10/v6 + 600 + 600 + 25/v6 = 1,221 s per pair of
    tasks -> 2 * 57600/1221 = 94.349 per day. T1 counts the clearance given up at X at the destination of one task twice
    with the origin window of the next (1,227.375 s -> 93.859 per day): stepwise requests with K = 3 over 40 statistics days
    exceed it by 0.49 per day, 6.5 times the tolerance K/40, without any deadlock. T1' counts the destination only to the end
    of unloading and matches the busy cycle; both occupancy rules stay within T1' + K/statistics days."""
    net = load_net(make_map(dict(X=(0, 0), J=(10, 0), E=(210, 0)), [("X", "J"), ("J", "E")],
                            {"fX": ("X", "assembly_workshop")}, width={"r0": 31.0, "r1": 60.0},
                            zones={"zY": ("buffer", "r1", (60, 100, 140, 180))}))
    p = short(days=41, warm=1)
    ys = sorted(s["node"] for sid, s in net.stops.items() if sid.startswith("zY#"))
    for mode, K in (("segment", 3), ("reserve", 4)):
        sim = ReuseSim(net, p, mode, K, [ys[i % len(ys)] for i in range(K)]).run()
        m = sim.metrics()
        mix = [r for r in sim.done if sim.t0m <= r["done"] <= sim.H]
        b = bounds(net, p, mix)
        tol = K / 40
        assert b["network_tight"] == pytest.approx(2 * 57600 / (10 / V6 + 600 + 600 + 25 / V6), rel=1e-3)
        assert b["binding_tight"][0]["res"] == ["n", "X"]
        assert m["throughput"] <= b["network_tight"] + tol, (mode, m["throughput"], b["network_tight"])
        assert not sim.deadlocks
        if mode == "segment":       # T1, which double-counts the clearance, is exceeded here
            assert b["network"] == pytest.approx(93.859, abs=1e-3)
            assert m["throughput"] > b["network"] + tol


# ================================================================ task generation
def test_erection_peak_generation(okpo):
    """Erection peak: P6 takes 15 % of the weight, the other flows keep their relative weights; 194 tasks a day; P6 only in
    batches 1-4 (released 0-6 h into the day), the other tasks at the release times of the plain peak (task i in batch
    i mod 8). The heavier mix without concentration keeps the release times. The main scenario keeps the base weights."""
    raw = [f[3] for f in P.FLOW]
    assert flow_weights(P.scenario("main")) == raw
    w = flow_weights(P.scenario("erection"))
    i6 = [f[0] for f in P.FLOW].index("P6")
    assert w[i6] / sum(w) == pytest.approx(0.15)
    for i, (a, b) in enumerate(zip(w, raw)):
        if i != i6:
            assert a / b == pytest.approx(w[0] / raw[0])
    day, n6, n = P.DEFAULTS["day_s"], 0, 0
    for seed in range(101, 111):
        for scen in ("erection", "erection_even"):
            p = P.scenario(scen)
            ts = demand_tasks(okpo, seed, 194, p)
            assert collections.Counter(int(t["release"] // day) for t in ts) == {d: 194 for d in range(p["days"])}
            for t in ts:
                d, i = divmod(t["id"], 194)
                slot = t["release"] - d * day
                if t["flow"] == "P6" and scen == "erection":
                    assert slot in (0.0, 7200.0, 14400.0, 21600.0)
                else:
                    assert slot == (i % 8) * day / 8
            if scen == "erection":
                n6 += sum(t["flow"] == "P6" for t in ts)
                n += len(ts)
    assert n6 / n == pytest.approx(0.15, abs=0.02)


def test_reserve_jit_departs_just_in_time():
    """Just-in-time departure: once the loaded run is booked the empty run is postponed, so far more tasks leave as soon as
    loading ends than with the earlier heuristic (jit_off). The factor 0.5 was set on Okpo revision 1825 (K = 40, seed 105:
    131 against 283 late departures). No stops en route and capacities are checked by test_reserve_never_deadlocks, which
    runs with just-in-time departure (the default)."""
    late = {}
    net = yard_net("yupu_r1825")
    for scen in ("main", "jit_off"):
        sim = Sim(net, short(scen), "reserve", 40, 105, None).run()
        late[scen] = sum(r["plan"]["t2"] - r["plan"]["a"] - P.DEFAULTS["load_s"] > 1.0 for r in sim.done)
    assert late["main"] < 0.5 * late["jit_off"], late


@pytest.mark.parametrize("mode", ["reserve", "segment"])
def test_dock_zone_and_wait_attribution(okpo, mode):
    """Ring 0 of the dock mouth is the stop roads of the two docks; the waits attributed to resources sum to the waits of the
    tasks (statistics window = whole run)."""
    ring0, ring1 = okpo.near(P.DOCK_CLASS)
    assert ring0 == {("r", "road_trace_027"), ("r", "road_trace_028")}
    assert {("n", "node_trace_009"), ("n", "node_trace_179")} <= ring1
    sim = Sim(okpo, short("erection"), mode, 30, 102, None).run()
    m = sim.metrics()
    if mode == "reserve":
        total = sum(r["wait"] for r in sim.done) + sum(x[2] for _, _, kind, x in sim.ev if kind == "done")
    else:
        total = sum(r["wait"] for r in sim.done) + sum(v.waited for v in sim.vs if v.id not in sim.idle)
    assert sum(sim.wait_by.values()) == pytest.approx(total, rel=1e-9)
    assert m["flows"]["P6"]["n_per_day"] > 0 and m["dock"]["ring0"]["util_max"] > 0
    assert m["dock"]["all"]["wait_min_per_day"] >= m["dock"]["ring0"]["wait_min_per_day"]


# ================================================================ yards: entities, roles, dock stops
def test_okpo_dock_stops_single_stretch(okpo):
    """Every Okpo dock has a single frontage stretch, so one stop per dock (dock1) gives the same stops as the main
    scenario."""
    one = yard_net("yupu", dock_stops=1)
    assert one.stops == okpo.stops and one.cuts == okpo.cuts
    assert all(len(okpo.ents[z]["stops"]) == 1 for z in okpo.by_cls[P.DOCK_CLASS])


def test_yantai_entities_and_reachability(yantai):
    """Yantai: building 007 (5 entrances) is the only assembly workshop; building 014 has 2 entrances and building 015 has 3;
    every role is taken by entities of the class of the same name and every role has entities; every pair of origin and
    destination stops of a flow is reachable; the starting stops of vehicles include building 007; the daily demand derived
    from public data is 23 regular and 46 peak tasks a day."""
    assert yantai.D["revision"] == 396
    assert yantai.by_cls["assembly_workshop"] == ["facility_trace_007"]
    e = yantai.ents["facility_trace_007"]
    assert e["map_cls"] == "assembly_workshop" and len(e["stops"]) == 5
    assert len(yantai.ents["facility_trace_014"]["stops"]) == 2 and len(yantai.ents["facility_trace_015"]["stops"]) == 3
    assert all(c == yantai.ents[x]["map_cls"] for c, xs in yantai.by_cls.items() for x in xs)
    roles = {c for f in P.FLOW for x in f[1:3] for c in x.split("|")}
    assert all(yantai.by_cls[c] for c in roles)
    for _, oc, dc, _ in P.FLOW:
        for a in (s for c in oc.split("|") for x in yantai.by_cls[c] for s in yantai.ents[x]["stops"]):
            for b in (s for c in dc.split("|") for x in yantai.by_cls[c] for s in yantai.ents[x]["stops"]):
                assert math.isfinite(yantai.dist(yantai.stops[a]["node"], yantai.stops[b]["node"])), (a, b)
    starts = {yantai.stops[yantai.ents[x]["stops"][0]]["node"] for c in P.START_CLASSES for x in yantai.by_cls[c]}
    assert yantai.ents["facility_trace_007"]["stops"][0] in starts
    Y = P.YARDS["yantai"]
    assert Y["demand"] == {"regular": 23, "peak": 46, "saturated": None}
    assert round(68512 / 100 * 10 / 300) == Y["demand"]["regular"] and Y["demand"]["peak"] == 2 * Y["demand"]["regular"]


def test_roles_map_classes(yantai):
    """roles = {role: class}: entities of another class take part in the flows in that role. With building 007 reclassified
    as a steel processing shop and roles = {assembly_workshop: steel_processing_shop}, the entities, roles and stops equal
    those of the default mapping; only the map class differs. Without the mapping there is no assembly workshop."""
    import copy
    G = copy.deepcopy(read_graph(P.YARDS["yantai"]["network"]))
    G["facilities"]["facility_trace_007"]["classId"] = "steel_processing_shop"
    a = Net(G, P.scenario("main"), {"assembly_workshop": "steel_processing_shop"})
    core = lambda n: {k: (v["cls"], v["stops"]) for k, v in n.ents.items()}
    assert core(a) == core(yantai) and a.ents["facility_trace_007"]["map_cls"] == "steel_processing_shop"
    assert a.stops == yantai.stops and a.cuts == yantai.cuts and a.by_cls == yantai.by_cls
    assert a.capacities(P.scenario("main")) == yantai.capacities(P.scenario("main"))
    assert Net(G, P.scenario("main")).by_cls["assembly_workshop"] == []


def test_yantai_dock_stops(yantai):
    """Dock stops of Yantai, one roadside stop at the middle of each stored frontage stretch: zone 001 two (road007, road018),
    zone 002 six; with dock_stops = 1 only the longest stretch of each dock remains (road007, road014), ring 0 of the dock
    mouth is these two roads, and the other stops are unchanged."""
    def dock_roads(net):
        return {z: [net.road[net.stops[s]["res"][1]]["name"] for s in net.ents[z]["stops"]] for z in net.by_cls["dry_dock"]}
    assert dock_roads(yantai) == {"zone_trace_001": ["road007", "road018"],
                                  "zone_trace_002": ["road013-2", "road008", "road009", "road010", "road011", "road014"]}
    for z in yantai.by_cls["dry_dock"]:
        fr = yantai.D["zones"][z]["frontage"]
        assert len(fr) == len(yantai.ents[z]["stops"])
        for (rid, s0, s1), s in zip(fr, yantai.ents[z]["stops"]):
            st = yantai.stops[s]
            assert st["mode"] == "on_road" and st["res"] == ("r", rid)
            assert float(st["node"].split("|")[2]) == round((s0 + s1) / 2, 1) and s1 - s0 >= P.DEFAULTS["frontage_min_len"]
    one = yard_net("yantai", dock_stops=1)
    assert dock_roads(one) == {"zone_trace_001": ["road007"], "zone_trace_002": ["road014"]}
    assert one.near(P.DOCK_CLASS)[0] == {("r", "road_trace_008"), ("r", "road_trace_016")}
    assert {s for s in one.stops if not s.startswith("zone_trace_00")} == {s for s in yantai.stops if not s.startswith("zone_trace_00")}


def test_frontage_rule_is_fixed_by_the_graph():
    """The dock frontage of a derived graph was computed with the rule stored in it; other frontage parameters are refused."""
    with pytest.raises(ValueError):
        Net(P.YARDS["yupu"]["network"], P.scenario("main", frontage_gap=10.0))


# ================================================================ crane pacing, lifts
def test_crane_release_and_p6_count(okpo, yantai):
    """Crane pacing: P6 per day is the peak-day number of erection units from public output (Okpo 2 * 20 * 100 / 300 = 13.3,
    Yantai 2), split over the docks in proportion to their cranes (dock 1 two, dock 2 one) and released from the start of
    the day at equal intervals, 16 h / that dock's P6 of the day, not shorter than the crane cycle / cranes; beyond the crane
    capacity (floor(cranes * 960 / 190) per dock) it raises an error. With a fractional P6 per day the first d days release
    round(n6 * d) in total. The other tasks are level - the day's P6, at the 8 batch times; the task stream does not draw
    P6; at equal release times P6 comes first. rho = n6 * 190 / (3 * 960): Okpo 0.88, Yantai 0.13."""
    day, cyc = P.DEFAULTS["day_s"], 190 * 60
    for net, yard, peak, n6 in ((okpo, "yupu", 194, 2 * 20 * 100 / 300), (yantai, "yantai", 46, 2)):
        p = crane_p(yard)
        cnt = net.crane_counts(p)
        assert crane_p6(net, p) == pytest.approx(n6) and crane_rho(net, p) == pytest.approx(n6 * 190 / 2880)
        assert sorted(cnt.values()) == [1, 2] and sum(cnt.values()) == 3
        ts = demand_tasks(net, 103, peak, p)
        assert collections.Counter(int(t["release"] // day) for t in ts) == {d: peak for d in range(p["days"])}
        p6 = [t for t in ts if t["flow"] == "P6"]
        per = collections.Counter(int(t["release"] // day) for t in p6)
        assert [per[d] for d in range(p["days"])] == [round(n6 * (d + 1)) - round(n6 * d) for d in range(p["days"])]
        assert all(t["id"] >= 10 ** 6 for t in p6)
        for d in range(p["days"]):               # other tasks: peak - P6 of the day, task i in batch i mod 8
            rest = sorted(t["release"] - d * day for t in ts if t["flow"] != "P6" and int(t["release"] // day) == d)
            assert rest == sorted((i % 8) * day / 8 for i in range(peak - per[d]))
        for i, t in enumerate(ts):               # equal release times: P6 first
            if t["flow"] != "P6":
                assert not any(u["flow"] == "P6" and u["release"] == t["release"] for u in ts[i + 1:])
        by = collections.Counter(net.stops[t["d"]]["ent"] for t in p6)
        for z, n in cnt.items():                 # weighted by cranes: share of a dock = cranes / 3 (+-1 a day for rounding)
            assert abs(by[z] - n6 * p["days"] * n / 3) <= p["days"]
        for d in range(p["days"]):
            per = collections.defaultdict(list)
            for t in p6:
                if int(t["release"] // day) == d:
                    per[net.stops[t["d"]]["ent"]].append(t["release"] - d * day)
            for z, xs in per.items():
                xs.sort()
                gaps = [b - a for a, b in zip(xs, xs[1:])]
                assert xs[0] == 0.0 and all(g == pytest.approx(day / len(xs)) and g >= cyc / cnt[z] - 1e-9 for g in gaps)
                assert len(xs) <= int(cnt[z] * day // cyc)
        assert all(net.stops[t["o"]]["ent"] in net.by_cls["assembly_yard"] for t in p6)
        s = TaskStream(net, 103, p)
        assert not any(s.next(0.0)["flow"] == "P6" for _ in range(2000))
        assert [len(x) for x in crane_split(net, p, 15, 1)[0]] == [cnt[z] * 5 for z in net.by_cls["dry_dock"]]   # 15 = crane capacity
        with pytest.raises(ValueError):
            crane_split(net, p, 16, 1)
    # equal crane numbers (one crane per dock): the docks take turns
    p = crane_p("yupu", n6=10, crane_n=1)
    assert [[len(x) for x in d] for d in crane_split(okpo, p, 10, 2)] == [[5, 5], [5, 5]] and crane_rho(okpo, p) == pytest.approx(0.99, abs=0.005)
    # saturated level: P6 still arrive at the crane pace (n6 a day), the other flows are endless. The cranes are a dock
    # resource: the booked crane windows never exceed the cranes (check_capacity, including two crane windows of the same
    # vehicle); under both occupancy rules the cranes working at once never exceed their number (self-check); free flow does
    # not check the cranes
    p = crane_p("yupu", n6=15, days=2, warm=1)
    for mode in ("reserve", "segment", "free"):
        sim = Sim(okpo, p, mode, 25, 101, None, record=True).run()
        m = sim.metrics()
        p6 = [r for r in sim.done if r["flow"] == "P6"]
        assert all(r["id"] >= 10 ** 6 for r in p6) and 22 <= len(p6) + len(sim.lifts) <= 30 and len(sim.done) > 200   # 30 arrive in two days
        assert set(m["crane"]) == {"waits_per_day", "wait_min_per_day", "wait_attr_min_per_day", "util_max", "busy_max"}
        assert m["crane"]["util_max"] > 0.5 and not any(x["res"][0] == "c" for x in m["busy_top"])
        if mode != "free":         # free flow records the crane hours needed, which can exceed 100 %
            assert m["crane"]["util_max"] <= 1 + 1e-9
            assert m["crane"]["waits_per_day"] == 0 and m["crane"]["wait_min_per_day"] < 1e-6
        if mode == "reserve":
            check_capacity(sim)
            assert any(r[0][0] == "c" for r in sim.log) and m["crane"]["wait_attr_min_per_day"] > 0


def test_crane_rows_in_T1_tight(okpo):
    """Crane rows of T1': each P6 to dock z holds a crane for 190 min, row lambda * (share to z) * 190 min <= cranes * 16 h.
    With a large P6 share the cranes bind first: T1' = min_z cranes_z * 57600 / (share_z * 11400), binding at a crane. In
    the bound with P6 fixed at n6 a day equal to the crane capacity the other flows still have a positive bound and the
    binding resource is a road or junction again; the throughput under crane pacing stays below T1' with crane rows."""
    p = crane_p("yupu")
    pb = dict(p, flow_share={"P6": 0.3})
    s = TaskStream(okpo, 5, pb)
    mix = [s.next(0.0) for _ in range(800)]
    b = bounds(okpo, pb, mix, tight_only=True)
    sh = collections.Counter(okpo.stops[t["d"]]["ent"] for t in mix if t["flow"] == "P6")
    expect = min(n * 57600 / (sh[z] / len(mix) * 11400) for z, n in okpo.crane_counts(p).items())
    assert b["network_tight"] == pytest.approx(expect, rel=1e-6) and b["binding_tight"][0]["res"][0] == "c"
    o = TaskStream(okpo, 5, p)
    from occupancy.bound import bounds_fixed
    f = bounds_fixed(okpo, p, [o.next(0.0) for _ in range(800)], (crane_tasks(okpo, 5, dict(p, days=20), 15), 15),
                     local=set().union(*okpo.near(P.DOCK_CLASS)) | {("c", z) for z in okpo.by_cls[P.DOCK_CLASS]})
    assert f["network_tight_fixed"] > 15 and f["binding_tight_fixed"][0]["res"][0] != "c"
    assert f["network_tight_fixed_local"] >= f["network_tight_fixed"] - 1e-6
    q = crane_p("yupu", n6=15, days=2, warm=1)
    for mode in ("reserve", "segment"):
        sim = Sim(okpo, q, mode, 30, 102, None).run()
        done = [r for r in sim.done if sim.t0m <= r["done"] <= sim.H]
        assert sim.metrics()["throughput"] <= bounds(okpo, q, done, tight_only=True)["network_tight"] + 30


def test_spans_and_lift_pairs(okpo, yantai):
    """In-span or not: the gantry crane span of each dock (a zone without a class) and the stored share of each area inside
    it (>= 0.5 is in span).
    1. Cranes = those of the dock's crane list that count for erection: Okpo dock 1 two, dock 2 one; Yantai dock 1 two,
       dock 2 one;
    2. one span per dock, span zones are not origin or destination entities;
    3. Okpo in span: zone 018 -> dock 1 (1.00), zone 014 -> dock 2 (0.52); with threshold 0.6 only zone 018; no assembly
       area of Yantai lies in a span; scenarios without crane pacing, or span_min = None, have no lifts.
    The share is the clipped polygon area (mapgraph.cover, checked here on a hand case)."""
    assert okpo.crane_counts(crane_p("yupu")) == {"zone_trace_019": 2, "zone_trace_013": 1}
    assert yantai.crane_counts(crane_p("yantai")) == {"zone_trace_002": 2, "zone_trace_001": 1}
    assert cover([(0, 0), (10, 0), (10, 10), (0, 10)], [(2, -5), (8, -5), (8, 5), (2, 5), (2, -5)]) == pytest.approx(0.3)
    for net in (okpo, yantai):
        sp = net.spans()
        assert sorted(d for d, _ in sp.values()) == net.by_cls["dry_dock"] and all(z.startswith("zone_crane_span_") for z in sp)
        assert not set(sp) & set(net.ents)
        assert all(0.0 <= f <= 1.0 for _, cov in sp.values() for f in cov.values())
    cov = {e: f for _, c in okpo.spans().values() for e, f in c.items()}
    assert cov["zone_trace_018"] == 1.0 and 0.5 <= cov["zone_trace_014"] < 0.6
    p = crane_p("yupu")
    assert okpo.lift_pairs(p) == {("zone_trace_018", "zone_trace_019"), ("zone_trace_014", "zone_trace_013")}
    assert okpo.lift_pairs(dict(p, span_min=0.6)) == {("zone_trace_018", "zone_trace_019")}
    assert not okpo.lift_pairs(dict(p, span_min=None)) and not okpo.lift_pairs(dict(P.scenario("main"), span_min=0.5))
    assert not {e for e, _ in yantai.lift_pairs(crane_p("yantai"))} & set(yantai.by_cls["assembly_yard"])


def test_lifts_take_crane_not_road(okpo):
    """An in-span P6 is lifted by the gantry crane directly: no vehicle task, but it holds the crane for 190 min.
    1. exactly the P6 of in-span (area, dock) pairs are marked as lifts; without the in-span rule (span_min = None) the task
       sequence is the same apart from the lift key, i.e. the rule does not touch the random numbers;
    2. under all three rules lifts never appear among the vehicle completions; vehicle completions + lifts = released P6
       and other tasks;
    3. lifts hold the crane: under reservation each lift has a 190-min crane window in the booking table, and together with
       the vehicles' crane windows they never exceed the cranes (check_capacity); under both occupancy rules the cranes
       working at once never exceed their number; free flow does not check the cranes and lifts start at release."""
    p = crane_p("yupu", n6=15, days=2, warm=0)          # rho = 0.99; statistics window = whole run
    ts = demand_tasks(okpo, 103, 194, p)
    pairs = okpo.lift_pairs(p)
    ent = lambda s: okpo.stops[s]["ent"]
    lifts = {t["id"] for t in ts if t.get("lift")}
    assert lifts and all((ent(t["o"]), ent(t["d"])) in pairs for t in ts if t.get("lift"))
    assert not any((ent(t["o"]), ent(t["d"])) in pairs for t in ts if t["flow"] == "P6" and not t.get("lift"))
    plain = demand_tasks(okpo, 103, 194, dict(p, span_min=None))
    assert [{k: v for k, v in t.items() if k != "lift"} for t in ts] == plain and not any("lift" in t for t in plain)
    c = p["crane_cycle_s"]
    for mode in ("free", "reserve", "segment"):
        sim = Sim(okpo, p, mode, 25, 103, 194, record=True).run()
        m = sim.metrics()
        done = {r["id"] for r in sim.done}
        assert not done & lifts and {r["id"] for r in sim.lifts} <= lifts
        assert len(sim.lifts) + m["lift"]["queue_end"] == len(lifts) and m["lift"]["released_per_day"] == len(lifts) / 2
        assert all(r["start"] >= r["release"] and r["done"] == r["start"] + p["crane_hook_s"] for r in sim.lifts)
        if mode == "free":
            assert all(r["start"] == r["release"] for r in sim.lifts)
        else:
            assert m["crane"]["waits_per_day"] == 0
        if mode == "reserve":
            check_capacity(sim)
            cl = [(s, e) for res, s, e, d, vid in sim.log if res[0] == "c" and vid < 0]
            assert len(cl) == len(sim.lifts) and all(e - s == pytest.approx(c) for s, e in cl)


def test_crane_edge_term(okpo):
    """Window edge term of the crane rows (bound.bounds_window): dock 2 with one crane, a mix of P6 to this dock only. In a
    statistics window W = 3 days the crane starts at most floor(W / 190 min) + 1 = 16 times (the first exactly at the start),
    i.e. 16 / 3 = 5.33 a day; the rate-based crane row T1' = 960 / 190 = 5.05 would be exceeded; with the edge term
    T1' = 5.05 * (W + 190 min) / W = 5.39 is not. Lifts in the window hold the same crane in absolute terms: k lifts lower the
    bound of the vehicle tasks by exactly k / 3 a day."""
    p = dict(crane_p("yupu", crane_n=1), span_min=None)          # all out of span: no lifts in the mix
    c, day = p["crane_cycle_s"], p["day_s"]
    W = 3 * day
    mix = [t for t in crane_tasks(okpo, 7, dict(p, days=10), 10) if okpo.stops[t["d"]]["ent"] == "zone_trace_013"][:40]
    assert not any(t.get("lift") for t in mix)
    n = int(W // c) + 1
    assert n == 16
    plain = bounds(okpo, p, mix, tight_only=True)
    edge = bounds_window(okpo, p, mix, [], W)
    assert plain["network_tight"] == pytest.approx(day / c) and plain["binding_tight"][0]["res"] == ["c", "zone_trace_013"]
    assert edge == pytest.approx(day / c * (W + c) / W)
    assert plain["network_tight"] < n * day / W <= edge
    lifts = [dict(t, lift=True) for t in mix[:3]]
    assert bounds_window(okpo, p, mix, lifts, W) == pytest.approx(edge - 3 * day / W)
    assert bounds_window(okpo, P.scenario("main"), mix, [], W) == bounds(okpo, P.scenario("main"), mix, tight_only=True)["network_tight"]


def test_crane_check_counts_waits(okpo):
    """Self-check of the cranes working at once: one crane per dock, 190 min per erection. Two unloadings at the same dock
    starting at 0 and 60 min -> the second exceeds by 130 min; the other dock is unaffected; gaps >= 190 min do not exceed.
    Only completions in the statistics window count."""
    p = dict(crane_p("yupu", crane_n=1, days=1, warm=0))
    sim = Sim(okpo, p, "reserve", 1, 101, None, tasks=[])
    z0, z1 = okpo.by_cls["dry_dock"]
    s0, s1 = okpo.ents[z0]["stops"][0], okpo.ents[z1]["stops"][0]
    u = p["unload_s"]
    rec = lambda i, d, start: dict(id=i, flow="P6", d=d, done=start + u)
    sim.done = [rec(0, s0, 0.0), rec(1, s0, 3600.0), rec(2, s1, 3600.0), rec(3, s0, 3600.0 + 130 * 60 + 190 * 60), dict(id=9, flow="P1", d=s0, done=10.0)]
    c = sim.crane_check(1.0)
    assert c["waits_per_day"] == 1 and c["wait_min_per_day"] == pytest.approx(130.0)


# ================================================================ Yantai: bounds, sampling, readings
@pytest.mark.parametrize("scen", ["main", "dock1"])
def test_yantai_below_T1_tight(scen):
    """Yantai, small scale: the throughput of the occupancy rules (reservation, stepwise) is at most
    min(T1', fleet bound) + K / statistics days; free flow is compared with the fleet bound only. T1' is that of the task mix
    completed in the statistics window (days 2-3)."""
    p = short(scen, days=3, warm=1)
    net = yard_net("yantai", dock_stops=p["dock_stops"])
    for mode in ("free", "reserve", "segment"):
        for K in (5, 30):
            sim = Sim(net, p, mode, K, 104, None).run()
            m = sim.metrics()
            b = bounds(net, p, [r for r in sim.done if sim.t0m <= r["done"] <= sim.H])
            fleet = fleet_bound(K, b["cycle_s"], p)
            bound = fleet if mode == "free" else min(b["network_tight"], fleet)
            assert m["throughput"] <= bound + K / 2, (scen, mode, K, m["throughput"], bound)


def test_od_sampling_stop(yantai):
    """Sampling of origins and destinations: with entities equally likely the 4 paint shops each take 1/4 of the paint
    destinations; with stops equally likely they are weighted by their stops, building 014 (2 of the 14 paint stops) takes
    1/7. Under both the destination is never in the origin's entity, and the 5 doors of building 007 each take 1/5 of the
    origins of P1, P2 and P7."""
    n = 20000
    paint = yantai.by_cls["paint_workshop"]
    assert len(paint) == 4 and sum(len(yantai.ents[e]["stops"]) for e in paint) == 14
    sh = {}
    for how in ("entity", "stop"):
        p = P.scenario("main", od_sampling=how)
        s = TaskStream(yantai, 7, p)
        ts = [s.next(0.0) for _ in range(n)]
        assert all(yantai.stops[t["o"]]["ent"] != yantai.stops[t["d"]]["ent"] for t in ts)
        pt = [t for t in ts if t["flow"] in ("P2", "P3")]
        sh[how] = sum(yantai.stops[t["d"]]["ent"] == "facility_trace_014" for t in pt) / len(pt)
        src = collections.Counter(t["o"] for t in ts if t["flow"] in ("P1", "P2", "P7"))
        assert set(src) == set(yantai.ents["facility_trace_007"]["stops"])
        assert max(src.values()) / sum(src.values()) == pytest.approx(0.2, abs=0.02)
    assert sh["entity"] == pytest.approx(1 / 4, abs=0.02) and sh["stop"] == pytest.approx(1 / 7, abs=0.02)


def test_net_cache_key_includes_roles():
    """The network cache of the runner (runner.net_for) is keyed by the role mapping too: the same graph with two mappings
    gives two networks; the same configuration returns the cached one."""
    R.NETS.clear()
    a = dict(network=P.YARDS["yantai"]["network"], roles={"assembly_workshop": "outfitting_workshop"})
    b = dict(a, roles={})
    na, nb = R.net_for(a, P.scenario("main"))[0], R.net_for(b, P.scenario("main"))[0]
    assert na is not nb and na.by_cls["assembly_workshop"] != nb.by_cls["assembly_workshop"]
    assert nb.by_cls["assembly_workshop"] == ["facility_trace_007"]
    assert R.net_for(a, P.scenario("main"))[0] is na


def test_top_positive_and_watch(yantai):
    """Top-5 lists show positive values only (free flow attributes no waiting, so its waiting list is empty); the full
    readings (watch) cover the dock mouth and all building entrances, and the largest busy share of the dock stop roads
    equals dock.ring0.busy_max; the daily throughputs add up to the completed tasks."""
    p = short()
    m = Sim(yantai, p, "free", 10, 101, None).run().metrics()
    assert m["wait_top"] == [] and all(x["value"] > 0 for x in m["util_top"] + m["busy_top"])
    sim = Sim(yantai, p, "reserve", 20, 101, None).run()
    m = sim.metrics()
    key = lambda r: "|".join(map(str, r))
    doors = {key(s["res"]) for s in yantai.stops.values() if s["mode"] == "node"}
    ring0 = {key(r) for r in sim.zone[0]}
    assert doors | ring0 | {key(r) for r in sim.zone[1]} == set(m["watch"])
    assert max(m["watch"][r][0] for r in ring0) == m["dock"]["ring0"]["busy_max"]
    assert sum(m["thr_by_day"]) == m["completed"] and len(m["thr_by_day"]) == p["days"] - p["warmup_days"]


@pytest.mark.parametrize("mode", ["free", "reserve", "segment"])
def test_seed_reproducible(okpo, mode):
    """The same seed gives identical runs; another seed a different one."""
    a = Sim(okpo, short(), mode, 15, 104, 194).run()
    b = Sim(okpo, short(), mode, 15, 104, 194).run()
    assert a.done == b.done and a.metrics() == b.metrics()
    c = Sim(okpo, short(), mode, 15, 105, 194).run()
    assert c.done != a.done


# ================================================================ safe release, no recovery
def test_safe_head_on_single_lane(corridor):
    """Two vehicles head-on on the single lane: the reference rule (teleport recovery) deadlocks at n2 and recovers one
    vehicle (neither at a dock nor carrying P6); safe release refuses the later vehicle's entry (granting it would create
    "you hold what I need and I hold what you need"), both complete, no deadlock, no vehicle leaves the network, capacities
    hold at every time, and every decision agrees with the naive reduction."""
    seg = run_with(corridor, "segment", [task(0, "A", "B"), task(1, "B", "A")], ["A", "B"])
    assert len(seg.deadlocks) >= 1 and seg.deadlocks[0]["victims"]
    assert seg.deadlocks[0]["flags"] == [(False, False)]
    sim = naive_run(corridor, [task(0, "A", "B"), task(1, "B", "A")], ["A", "B"])
    assert len(sim.done) == 2 and no_one_left(sim) and sim.safe_n["unsafe"] >= 1 and sim.naive_n > 0
    check_capacity(sim)
    t = sorted(r["done"] for r in sim.done)
    assert t[1] > t[0] + 600            # one waits until the other has passed the whole corridor


@pytest.mark.parametrize("n", [3, 4])
def test_safe_circular_wait(n):
    """Three or four vehicles in a circular wait on a ring: vehicle i loads at entrance Ni and goes to the next entrance
    N(i+1), holding Ni while loading and requesting N(i+1) on departure. Stepwise requests (reference) form an n-vehicle wait
    cycle and recover; safe release refuses the n-th vehicle's entry (afterwards no order lets the vehicles finish in turn),
    every vehicle completes without deadlock. Without recovery (deadlock_recovery_s = None) the n vehicles stay stuck: one
    deadlock is recorded, all n are waiting at the end and no task completes."""
    net = ring(n)
    tasks = lambda: [task(i, f"N{i}", f"N{(i + 1) % n}") for i in range(n)]
    locs = [f"N{i}" for i in range(n)]
    seg = run_with(net, "segment", tasks(), locs)
    assert seg.deadlocks and seg.deadlocks[0]["core"] == n and len(seg.done) == n
    sim = naive_run(net, tasks(), locs)
    assert len(sim.done) == n and no_one_left(sim) and sim.safe_n["unsafe"] >= 1
    check_capacity(sim)
    frz = run_with(net, "segment", tasks(), locs, p=short(deadlock_recovery_s=None))
    m = frz.metrics()
    assert len(frz.deadlocks) == 1 and frz.deadlocks[0]["size"] == n and not frz.done
    assert len(frz.frozen) == n and m["waiting_end"] == n and m["victims_diag"]["frozen_end"] == n and m["near_deadlocks"] == 0


def test_safe_conservative_ring6():
    """Safe release is conservative because of the form of the Banker criterion: a single-lane ring of six entrances, three
    vehicles in the same direction each going two roads (N0 -> N2, N2 -> N4, N4 -> N0). The vehicle ahead frees its entrance
    on departure, so the three follow each other round the ring without meeting: stepwise requests have no deadlock and all
    three complete together; safe release, which lets each vehicle claim its whole remaining task while the others stand
    still, judges the third vehicle's loading unsafe and refuses it, so it completes later."""
    net = ring(6)
    tasks = lambda: [task(i, f"N{2 * i}", f"N{(2 * i + 2) % 6}") for i in range(3)]
    locs = [f"N{2 * i}" for i in range(3)]
    seg = run_with(net, "segment", tasks(), locs)
    assert not seg.deadlocks and len(seg.done) == 3 and len({round(r["done"], 1) for r in seg.done}) == 1
    sim = naive_run(net, tasks(), locs)
    assert len(sim.done) == 3 and no_one_left(sim) and sim.safe_n["unsafe"] >= 1 and sim.safe_n["wait_s"] > 0
    assert max(r["done"] for r in sim.done) > max(r["done"] for r in seg.done) + 300


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_safe_random_networks(seed):
    """One day on a random small network under safe release: the deadlock check never fires (it would raise), no vehicle
    leaves the network, capacities hold at every time, every decision agrees with the naive reduction, and the bookkeeping of
    occupancy not released on time agrees with hold / pending."""
    net, tasks, locs = random_case(seed)
    sim = naive_run(net, tasks, locs)
    assert no_one_left(sim) and sim.naive_n > 50 and len(sim.done) > 0
    check_capacity(sim)


def test_safe_random_networks_nontrivial():
    """The random instances above do contain deadlocks of stepwise requests (reference), so the cross-check has weight: at
    least one over the four seeds."""
    n = 0
    for seed in (1, 2, 3, 4):
        net, tasks, locs = random_case(seed)
        n += len(run_with(net, "segment", tasks, locs).deadlocks)
    assert n > 0


# ================================================================ sectioning and the direction lock
def test_section_corridor_rules():
    """Single lane, sectioned, two vehicles head-on: full-path reservation and safe release cannot deadlock head-on
    (reservation is a timetable; safe release refuses requests leading to a deadlock) and need no direction lock; stepwise
    requests (reference) with the direction lock wait outside the sectioned road until the oncoming vehicle has left the
    whole map road, and do not deadlock; without the lock they deadlock in the middle of the road."""
    whole, sec = sectioned_corridor()
    assert sec.sec == {"r1": ["r1@0", "r1@1", "r1@2"]} and sec.edges["r1"] == [275.0, 325.0]
    assert sec.stops["zbuf#0"]["res"] == ("r", "r1@1") and whole.stops["zbuf#0"]["res"] == ("r", "r1")
    p = short(section="stops")
    tasks = lambda: [task(0, "A", "B"), task(1, "B", "A")]
    for mode in ("reserve", "segment_safe", "segment"):
        sim = run_with(sec, mode, tasks(), ["A", "B"], p=p)
        assert len(sim.done) == 2 and not sim.deadlocks, mode
        check_capacity(sim)
        assert sim.lock == (mode == "segment")
    sim = Sim(sec, p, "segment", 2, 1, None, record=True, tasks=tasks())
    sim.lock = False
    for v, loc in zip(sim.vs, ["A", "B"]):
        v.loc = loc
    sim.run()
    assert sim.deadlocks and len(sim.done) == 2          # without the lock: head-on deadlock, recovered by teleport


def test_section_network_okpo(okpo):
    """Okpo sectioned at the stops: roads longer than 200 m with roadside stops get boundaries 25 m before and after each
    stop (roadside stops and drive-in points). road161 (9 m, 1,095 m) is cut at 79.5, 129.5, 445.5, 495.5, 522.6 and
    572.6 m into 7 sections; road020 into 6. Every section keeps the width and capacity of its road; a roadside stop's
    resource is its section (dock 1: the 50 m section 445.5-495.5 m); roads not sectioned are unchanged; the network stays
    connected and shortest distances between stops are unchanged; ring 0 of the dock mouth is the two dock sections and
    ring 1 holds the other sections of their roads; section = None gives the same network."""
    sec = yard_net("yupu", section="stops")
    assert sec.edges["road_trace_027"] == [79.5, 129.5, 445.5, 495.5, 522.6, 572.6] and len(sec.sec["road_trace_027"]) == 7
    assert okpo.road["road_trace_027"]["name"] == "road161" and len(sec.sec["road_trace_028"]) == 6
    on_road = {s["res"][1] for s in okpo.stops.values() if s["mode"] == "on_road"}
    assert set(sec.sec) == {r for r in on_road if okpo.road[r]["L"] > 200 and sec.edges.get(r)}
    p = P.scenario("main")
    c0, c1 = okpo.capacities(p)[0], sec.capacities(p)[0]
    for o, ss in sec.sec.items():
        assert sum(sec.road[s]["L"] for s in ss) == pytest.approx(okpo.road[o]["L"], abs=1e-9)
        assert all(sec.road[s]["W"] == okpo.road[o]["W"] and c1[("r", s)] == c0[("r", o)] for s in ss)
    assert all(sec.road[r] == okpo.road[r] for r in okpo.road if r not in sec.sec)
    d = sec.stops["zone_trace_019#0"]["res"][1]
    assert (sec.road[d]["s0"], sec.road[d]["s1"]) == (445.5, 495.5)
    for sid, st in sec.stops.items():
        if st["mode"] == "on_road":
            at = float(st["node"].split("|")[2])
            r = sec.road[st["res"][1]]
            assert r["orig"] == okpo.stops[sid]["res"][1] and r["s0"] < at < r["s1"]
    nodes = sorted({s["node"] for s in okpo.stops.values()})
    rng = random.Random(0)
    for a, b in [tuple(rng.sample(nodes, 2)) for _ in range(300)]:
        assert sec.dist(a, b) == pytest.approx(okpo.dist(a, b), abs=1e-6)
    r0, r1 = sec.near(P.DOCK_CLASS)
    assert r0 == {sec.stops[s]["res"] for z in sec.by_cls[P.DOCK_CLASS] for s in sec.ents[z]["stops"]}
    assert {("r", s) for s in sec.sec["road_trace_027"]} - r0 <= r1
    plain = Net(P.YARDS["yupu"]["network"], P.scenario("main", section=None))
    for attr in ("road", "stops", "cuts", "arc", "succ", "incident", "sec"):
        assert getattr(plain, attr) == getattr(okpo, attr), attr


def test_section_block_m(okpo):
    """Block length block_m: sectioned roads are further cut evenly between the stop boundaries into blocks of at most
    block_m; shortest distances are unchanged; a smaller block_m gives more sections."""
    nets = {B: yard_net("yupu", section="stops", block_m=B) for B in (100.0, 50.0)}
    for B, net in nets.items():
        for o, ss in net.sec.items():
            assert all(net.road[s]["L"] <= B + 1e-6 for s in ss), (B, o, [round(net.road[s]["L"], 1) for s in ss])
    assert len(nets[50.0].sec["road_trace_027"]) > len(nets[100.0].sec["road_trace_027"]) > 7
    nodes = sorted({s["node"] for s in okpo.stops.values()})
    rng = random.Random(1)
    for a, b in [tuple(rng.sample(nodes, 2)) for _ in range(100)]:
        assert nets[50.0].dist(a, b) == pytest.approx(okpo.dist(a, b), abs=1e-6)


def test_section_raises_T1_tight(okpo):
    """T1' of the sectioned network bounds the sectioned model (finer resources still satisfy the assumptions). The window
    of each section is a subset of the window of the whole road, so for the same mix the sectioned T1' is at least the
    whole-road one; the erection mix (P6 15 %) binds at road161 on the whole road, and sectioning raises the bound clearly,
    with another binding resource."""
    p = P.scenario("erection")
    sec = yard_net("yupu", section="stops")
    s1, s2 = TaskStream(okpo, 101, p), TaskStream(sec, 101, p)
    mix1, mix2 = [s1.next(0.0) for _ in range(800)], [s2.next(0.0) for _ in range(800)]
    assert [(t["o"], t["d"], t["B"]) for t in mix1] == [(t["o"], t["d"], t["B"]) for t in mix2]
    a = bounds(okpo, p, mix1, tight_only=True)
    b = bounds(sec, dict(p, section="stops"), mix2, tight_only=True)
    assert a["binding_tight"][0]["res"] == ["r", "road_trace_027"]
    assert b["network_tight"] > 1.3 * a["network_tight"] and b["binding_tight"][0]["res"] != ["r", "road_trace_027"]


def test_dock_crane_wait_and_yard_access(okpo):
    """A vehicle at the dock waiting for the crane while a vehicle must pass to the adjacent assembly area (Okpo dock 1, one
    crane, all P6 by road): vehicles A and B bring P6 from zone 018 to dock 1; A arrives first and holds the crane for
    190 min; B arrives and waits for the crane on the stop road. C goes from zone 001 along road161 from the other end to the
    drive-in point of zone 009 (547.6 m). With the whole road one resource, B waits on road161 and C is blocked until the
    crane is free and B has unloaded (under stepwise requests and safe release alike); sectioned, B holds only the 50 m dock
    section and under safe release C passes; the direction lock of the reference rule makes C wait for B on the whole map
    road, so C is blocked again (the lock is conservative); under full-path reservation B waits at its origin until its crane
    slot and C passes. Safe release treats the crane as available sooner or later and does not deadlock."""
    cr = dict(crane_p("yupu", crane_n={"zone_trace_019": 1, "zone_trace_013": 1}, days=1, warm=0), span_min=None)
    nets = {None: okpo, "stops": yard_net("yupu", section="stops")}
    out = {}
    for sec, net in nets.items():
        for mode in ("reserve", "segment_safe", "segment"):
            tasks = [dict(task(10 ** 6, "zone_trace_018#2", "zone_trace_019#0"), flow="P6"),
                     dict(task(10 ** 6 + 1, "zone_trace_018#1", "zone_trace_019#0"), flow="P6"),
                     dict(task(0, "zone_trace_001#0", "zone_trace_009#22", release=1800.0), flow="P5")]
            locs = [net.stops[s]["node"] for s in ("zone_trace_018#2", "zone_trace_018#1", "zone_trace_001#0")]
            sim = run_with(net, mode, tasks, locs, p=dict(cr, section=sec))
            done = {r["id"]: r["done"] for r in sim.done}
            assert len(done) == 3 and not sim.deadlocks, (sec, mode)
            out[(sec, mode)] = done
    crane_free = min(out[(None, "segment_safe")][10 ** 6], out[(None, "segment_safe")][10 ** 6 + 1]) - 600 + 190 * 60
    for key in ((None, "segment_safe"), (None, "segment"), ("stops", "segment")):
        assert out[key][0] > crane_free, key
    for key in (("stops", "segment_safe"), (None, "reserve"), ("stops", "reserve")):
        assert out[key][0] < 6000, key


def test_crane_starvation_hand_value(okpo):
    """Crane starvation by hand: full-path reservation, one vehicle, one P6 (zone 031 -> dock 1, released at 0), statistics
    window the whole day. The crane is idle from 0 until unloading starts (= empty run + loading + loaded run, the planned
    start of erection), so the starvation of dock 1 = start of erection / 16 h; the vehicle did not wait at a stop, so the
    part blocked by the network is 0; dock 2 has no P6 and gets 0."""
    p = dict(crane_p("yupu", days=1, warm=0), span_min=None)
    t = dict(task(10 ** 6, "zone_trace_031#0", "zone_trace_019#0"), flow="P6")
    sim = run_with(okpo, "reserve", [t], [okpo.stops["zone_trace_031#0"]["node"]], p=p)
    m = sim.metrics()
    cs = sim.done[0]["cs"]
    assert cs == pytest.approx(sim.done[0]["done"] - p["unload_s"])
    sv = m["starve"]["by_dock"]
    assert sv["zone_trace_019"]["starve"] == pytest.approx(cs / p["day_s"]) and sv["zone_trace_019"]["blocked"] == 0
    assert sv["zone_trace_013"] == dict(starve=0.0, blocked=0.0, blocked_road=0.0) and m["starve"]["starve_max"] == pytest.approx(cs / p["day_s"])
    c = m["cycle"]
    assert c["wait"] == pytest.approx(0.0, abs=1e-9) and c["total"] == pytest.approx((sim.done[0]["done"] - 0.0) / 60)


def test_crane_starvation_bounds(okpo):
    """Under crane pacing (rho = 0.99) the crane starvation of all three occupancy rules lies in [0, 1] and the part blocked
    by the network does not exceed it; the recovery diagnostics of the reference rule do not exceed the recoveries."""
    p = crane_p("yupu", n6=15, days=2, warm=1)
    for mode in ("reserve", "segment", "segment_safe"):
        m = Sim(okpo, p, mode, 25, 102, None).run().metrics()
        for x in m["starve"]["by_dock"].values():
            assert 0.0 <= x["blocked"] <= x["starve"] <= 1.0, mode
        if mode == "segment":
            d = m["victims_diag"]
            assert d["at_dock_per_day"] <= d["per_day"] and d["p6_loaded_per_day"] <= d["per_day"]


# ================================================================ steady state, drift, fleet search, demand
def test_drift():
    """Daily drift: about 0 when flat; a fall from 788 to 434 per day over 7 days (crane-paced reference rule, K = 150) is
    about -0.6, far beyond the 5 % tolerance."""
    assert R.drift([600, 600, 600]) == 0 and R.drift([100, 110]) == pytest.approx(10 / 105)
    assert R.drift([788.3, 642.5, 597.3, 535.5, 441.7, 436.1, 433.9]) < -10 * R.STEADY_DRIFT_TOL


def test_steady_criterion():
    """Steady-state criterion: entrance A by a short road to junction J; on one side a 20 m road to paint shop N, on the other
    a 2 km single-lane 9 m road to paint shop F (loaded and empty vehicles hold it alone). Only P2 (A -> paint), entities
    equally likely, so half of the tasks go to F. The narrow road passes one vehicle at a time and its queue grows.
    Statistics window from 0 (no warm-up), 2 days: with K = 40 the queue is far longer at the end than at the start, F is
    under-represented in the completed mix, the mix shift is clearly positive and the long-run estimate is below the
    measured throughput; with K = 1 there is no queue and the shift is about 0."""
    pos = dict(A=(0, 0), J=(10, 0), N=(10, 20), F=(10, -2000))
    net = load_net(make_map(pos, [("A", "J"), ("J", "N"), ("J", "F")], {"fA": ("A", "assembly_workshop"),
                                                                       "fN": ("N", "paint_workshop"), "fF": ("F", "paint_workshop")},
                            width={"r0": 31.0, "r1": 20.0, "r2": 9.0}))
    p = short(days=2, warm=0, load_s=60.0, unload_s=60.0, flow_share={"P2": 1.0})
    out = {}
    for K in (1, 40):
        sim = Sim(net, p, "reserve", K, 3, None).run()
        mix = [r for r in sim.done if sim.t0m <= r["done"] <= sim.H]
        st = R.steady(net, p, sim, bounds(net, p, mix, tight_only=True)["network_tight"])
        assert list(st) == ["2"] and st["2"]["thr"] == pytest.approx(sim.metrics()["throughput"])
        out[K] = st["2"]
    assert abs(out[1]["mix_shift"]) < R.STEADY_TOL
    assert out[40]["mix_shift"] > 5 * R.STEADY_TOL and out[40]["longrun"] < out[40]["thr"]


def test_fleet_search():
    """Required fleet (required_fleet.search; configuration of the stored Yantai plant-level search): the smallest integer K
    whose mean throughput reaches 0.99 x the daily demand. Monotone rules: K = 150 first, ">150" if not reached, otherwise
    bisection on the integers, then K_req + 1 ... + 3 are tested and a later failure is marked. Safe release: if a probe
    (including 10, 15 and the peak fleets) reaches the target, every integer below it is computed; if no K <= 150 does, 100-200
    is scanned and the integers between the first passing fleet and the previous one are filled in. The search reads only the
    points written so far (resuming only adds points)."""
    cfg = load_config("results/runs/yantai_fleet_plant/config.json")
    assert cfg["yard"] == "yantai" and cfg["demand"]["n300"] == 300
    lvl = "n300"

    def run(m, thr, lv=lvl, pts=None):
        pts, rounds = {} if pts is None else pts, []
        while True:
            ks, res = RF.search(cfg, pts, "d16", m, lv)
            if not ks:
                return res, rounds
            rounds.append(ks)
            for k in ks:
                for sd in cfg["seeds"]:
                    pts[("d16", m, lv, k, sd)] = dict(throughput=thr(k))

    for kstar in (1, 7, 40, 47, 95, 150, None):            # None: not even K = 150 reaches the target
        res, rounds = run("reserve", lambda K: 300.0 if kstar is not None and K >= kstar else 250.0 + K / 10)
        assert res["K_req"] == kstar and res["over"] == (kstar is None) and not res["nonmono"] and not res["after_fail"], (kstar, res)
        assert rounds[0] == [150] and len(rounds) <= 1 + math.ceil(math.log2(150)) + 1
        if kstar is not None and kstar > 1:
            assert res["K_below"] == kstar - 1 and res["thr_below"] < res["target"] <= res["thr"]
    res, _ = run("reserve", lambda K: 300.0 if K >= 47 and K != 49 else 250.0)     # passes, then fails again above
    assert res["K_req"] == 47 and res["after_fail"] == [49] and res["nonmono"]
    # safe release: (52, 53) and 55-70 pass (the smallest passing probe is 60; 1...59 are filled in and give 52, then 53...57
    # mark 54); 165-175 (nothing <= 150 passes, 170 passes in 100-200, 161...169 are filled in); 90-95 and 155-158 (missed by
    # the probes and the scan) give ">200"
    for ok, expect, after in ((lambda K: K in (52, 53) or 55 <= K <= 70, 52, [54]), (lambda K: 165 <= K <= 175, 165, []),
                              (lambda K: 90 <= K <= 95 or 155 <= K <= 158, None, [])):
        res, rounds = run("segment_safe", lambda K: 300.0 if ok(K) else 200.0)
        assert res["K_req"] == expect and res["after_fail"] == after, res
        assert set(rounds[0]) == {10, 15, 18, 20, 30, 40, 50, 60, 80, 100, 120, 150}
        if expect == 52:
            assert set(range(1, 52)) <= set(res["probed"]) and res["nonmono"]
        if expect is None:
            assert res["over"] and res["K_top"] == 200 and res["thr_best"] == 200.0 and res["high_scan"]
    # 100-200 is scanned only for the lowest level of a scenario whose target fails at K <= 150: n450 is scanned once n300
    # passed at 170; with n300 failing everywhere n450 is not scanned (">150", high_scan = False)
    pts = {}
    run("segment_safe", lambda K: 300.0 if K >= 170 else 200.0, pts=pts)
    res, _ = run("segment_safe", lambda K: 450.0 if K >= 180 else 200.0, lv="n450", pts=pts)
    assert res["K_req"] == 180 and set(range(171, 180)) <= set(res["probed"])
    res, _ = run("segment_safe", lambda K: 450.0 if K >= 180 else 200.0, lv="n450")
    assert res["over"] and res["K_top"] == 150 and res["high_scan"] is False


def test_truck_demand_less_lifts():
    """The vehicle demand of a level is the daily demand less the lifts by gantry cranes; the target of the fleet search is
    0.99 of it, from the mean released lifts of the seeds."""
    assert RF.truck_demand(339, 0.0) == 339 and RF.truck_demand(339, 6.5) == pytest.approx(332.5)
    cfg = dict(seeds=[1, 2], demand={"n339": 339})
    pts = {("s", "m", "n339", 30, 1): dict(throughput=320.0, lift=dict(released_per_day=6.0)),
           ("s", "m", "n339", 30, 2): dict(throughput=330.0, lift=dict(released_per_day=8.0))}
    assert RF.evaluate(cfg, pts, "s", "m", "n339", 30) == (pytest.approx(325.0), pytest.approx(0.99 * (339 - 7.0)))
    assert RF.evaluate(cfg, pts, "s", "m", "n339", 40) is None


def test_daily_tasks_shared():
    """One conversion for both yards: daily tasks = steel / mean block weight x transfers per block / working days; Yantai
    68,512 t -> 22.8 -> 23 / 46; Okpo 1,018,067 t -> 339.4 -> 339; plant-level grid 300, 450, 600 and the peaks at twice."""
    assert P.daily_tasks(68512) == pytest.approx(22.837, abs=1e-3) and P.YANTAI_DEMAND == {"regular": 23, "peak": 46, "saturated": None}
    assert P.daily_tasks(68512, block_t=50) == pytest.approx(45.67, abs=0.01) and P.STEEL_T["yupu"] == 1018067
    assert P.daily_tasks(P.STEEL_T["yupu"]) == pytest.approx(339.36, abs=0.01) and P.YUPU_DEMAND == 339
    assert P.PLANT_LEVELS == (300, 450, 600, 900, 1200) and P.DAY_HOURS == (16, 24)

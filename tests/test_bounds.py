"""Capacity bounds: the system bound T1' (exact task mix, no sampling noise), the route-free bound T1'' with its Lagrangian
certificate, the route-fixed value R, back-substitution, read-out of the binding resource and conversions of dual prices
(experiments/capacity_bounds.py, bounds/)."""
import collections
import math

import numpy as np
import pytest

import capacity_bounds as CB
from bounds.route_free import RouteFree, ret_road, t1_on_path
from bounds.system import Loads, SystemT1, bl_joint, crane_mix, empty_block, od_mix, sampled_mix, scenario_mix, solve_t1, t1_times
from helpers import H, V6, load_config, load_net, make_map, two_routes, yard_net
from occupancy import params as P
from occupancy.bound import res_name
from occupancy.tasks import TaskStream, crane_p6, crane_tasks

BOUNDS = {"yupu": "results/bounds/okpo/config.json", "yantai": "results/bounds/yantai/config.json"}


def cfg_of(yard):
    """Configuration of the stored capacity bounds of a yard (network, parameters, scenarios main / erection / crane, seeds,
    reference runs of the sampled bounds)."""
    return load_config(BOUNDS[yard])


def net_of(yard):
    """Network of the stored bounds: Okpo revision 1825, Yantai revision 396 (the three scenarios share it)."""
    cfg = cfg_of(yard)
    key = {"okpo/rev1825": "yupu_r1825", "yantai/rev396": "yantai"}[cfg["network"]]
    return yard_net(key)


def scen(yard, name):
    cfg = cfg_of(yard)
    return dict(cfg["params"], **cfg["scenarios"][name])


def mixes(tasks, fx):
    return sampled_mix(tasks), ((sampled_mix(fx[0]), fx[1]) if fx else None)


# ---------------------------------------------------------------- system T1'
@pytest.mark.parametrize("yard", ["yupu", "yantai"])
@pytest.mark.parametrize("name", ["main", "erection", "crane"])
def test_system_code_reproduces_seed_T1(yard, name):
    """The same code on a sampled mix (w = 1/N, a single (B, Lv) per task) reproduces the per-seed T1' of occupancy.bound
    (bounds_fixed under crane pacing), and equals the stored value of seed 101 in the reference runs (network_tight of the
    bound samples, network_tight_fixed of the crane-paced runs)."""
    net, p = net_of(yard), scen(yard, name)
    tasks, fx = CB.sample(net, p, 101, 2000)
    t1 = CB.official_t1(net, p, tasks, fx)
    assert SystemT1(net, p, *mixes(tasks, fx)).value()[0] == pytest.approx(t1, rel=1e-12)
    ref, _ = CB.ref_values(cfg_of(yard), name)
    assert t1 == pytest.approx(ref[101], rel=1e-12)


def test_bl_joint():
    """Joint distribution of (B, Lv): 165 combinations with probabilities summing to 1; B = 5.5 is bw = 5 (1/11); Lv has the
    platform length 16.25 as its lower bound."""
    bl = bl_joint(P.DEFAULTS)
    assert len(bl) == 165 and sum(q for _, q in bl) == pytest.approx(1.0, abs=1e-12)
    assert sum(q for (B, _), q in bl if B == 5.5) == pytest.approx(1 / 11)
    assert min(lv for (_, lv), _ in bl) == 16.25 and max(lv for (_, lv), _ in bl) == 30.0
    assert sum(q for (B, lv), q in bl if B == 15.0 and lv == 16.25) == pytest.approx(1 / 11 * 2 / 16)   # bw = 15: bl 15...30, 15 and 16 -> 16.25


@pytest.mark.parametrize("od", ["entity", "stop"])
def test_distribution_matches_task_stream(od):
    """The sampling law of the system mix agrees with TaskStream item by item: on Yantai the frequencies of flows, origin
    stops, destination stops and (B, Lv) of 100,000 sampled tasks lie within the expectation +- 5 sigma
    (sigma = sqrt(p(1 - p)/N)); the probabilities sum to 1. The crane-paced P6 (crane_tasks) are checked the same way by dock,
    origin stop and lift."""
    net = net_of("yantai")
    p = dict(scen("yantai", "main"), od_sampling=od)
    N = 100000
    s = TaskStream(net, 7, p)
    ts = [s.next(0.0) for _ in range(N)]
    mix = od_mix(net, p)
    assert sum(w for w, _ in mix) == pytest.approx(1.0, abs=1e-12)

    def check(emp, exp, n):
        for k in set(emp) | set(exp):
            q = exp.get(k, 0.0)
            assert abs(emp.get(k, 0) / n - q) <= 5 * math.sqrt(q * (1 - q) / n) + 1e-12, (k, emp.get(k, 0) / n, q)
    for key in (lambda t: t["flow"], lambda t: t["o"], lambda t: t["d"]):
        exp = collections.Counter()
        for w, t in mix:
            exp[key(t)] += w
        check(collections.Counter(key(t) for t in ts), exp, N)
    check(collections.Counter((t["B"], t["Lv"]) for t in ts), dict(bl_joint(p)), N)
    p = scen("yupu", "crane")
    net = net_of("yupu")
    n6 = crane_p6(net, p)
    ts = crane_tasks(net, 7, dict(p, days=3000), n6)
    cm = crane_mix(net, p)
    assert sum(w for w, _ in cm) == pytest.approx(1.0, abs=1e-12)
    for key in (lambda t: net.stops[t["d"]]["ent"], lambda t: t["o"], lambda t: bool(t.get("lift"))):
        exp = collections.Counter()
        for w, t in cm:
            exp[key(t)] += w
        check(collections.Counter(key(t) for t in ts), exp, len(ts))


def test_system_T1_converges():
    """Yantai main scenario: T1' of N tasks sampled by the same law converges to the system T1' - over 5 seeds the mean
    relative deviation at N = 20,000 is smaller than at N = 2,000."""
    net, p = net_of("yantai"), scen("yantai", "main")
    T = SystemT1(net, p, *scenario_mix(net, p)).value()[0]
    dev = {n: sum(abs(CB.official_t1(net, p, *CB.sample(net, p, s, n)) / T - 1) for s in range(101, 106)) / 5 for n in (2000, 20000)}
    assert dev[20000] < dev[2000] < 0.2


# ---------------------------------------------------------------- route-free T1''
def test_two_routes_hand_value():
    """By hand: the narrow road has capacity 10 and a loaded vehicle takes 10 -> exclusive (share 1 in row k = 1); the wide
    road has capacity 26 and a loaded vehicle takes 11 -> share 1/2 in row k = 2.
    T1' (shortest path, the narrow road only): 500/v6 + tail (20 + 5)/v6 = 315 s per task -> H / 315 = 182.86.
    R (narrow road, vehicle length at its lower bound 16.25): tail 21.25/v6 -> a = 312.75 s -> H / a = 184.17; R - T1' comes
    entirely from the vehicle length.
    T1'': x on the narrow road, 1 - x on the wide route: lambda x a <= H, lambda (1 - x) b <= H,
    b = (400 + 21.25)/v6 x 1/2 = 126.375 s (one row per wide road) -> T1'' = H/a + H/b = 639.96; detour share a/(a + b) =
    71.2 %, each detoured task drives 300 m more."""
    net, p, tasks = two_routes()
    a, b = (500 + 21.25) / V6, (400 + 21.25) / V6 / 2
    from occupancy.bound import bounds
    assert bounds(net, p, tasks, tight_only=True)["network_tight"] == pytest.approx(H / 315, rel=1e-12)
    assert SystemT1(net, p, sampled_mix(tasks)).value()[0] == pytest.approx(H / 315, rel=1e-12)
    out = RouteFree(net, p, sampled_mix(tasks)).solve()
    assert out["R"] == pytest.approx(H / a, rel=1e-12)
    assert out["T1pp"] == pytest.approx(H / a + H / b, rel=1e-9) and out["gap"] < 1e-9
    assert out["detour"]["share"] == pytest.approx(a / (a + b), rel=1e-6)
    assert out["detour"]["extra_m"] == pytest.approx(300.0, rel=1e-9)
    assert {x["name"].split(" (")[0] for x in out["binding"] if x["price_per_h"] > 0} >= {"r1"}
    assert out["binding"][0]["util"] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("yard", ["yupu", "yantai"])
def test_order_on_ten_seeds(yard):
    """Main scenario of both yards, the ten seeds of the stored bounds (2,000 tasks each): T1' <= R <= T1'' in every case;
    column generation converges (Lagrangian bound and master problem within 1e-7)."""
    net, p = net_of(yard), scen(yard, "main")
    for seed in cfg_of(yard)["seeds"]:
        tasks, fx = CB.sample(net, p, seed, 2000)
        t1 = CB.official_t1(net, p, tasks, fx)
        out = RouteFree(net, p, *mixes(tasks, fx)).solve()
        assert t1 <= out["R"] * (1 + 1e-9) and out["R"] <= out["T1pp"] * (1 + 1e-9), (seed, t1, out["R"], out["T1pp"])
        assert out["gap"] <= 1e-7 and out["master"] <= out["T1pp"] * (1 + 1e-12)


def test_order_with_fixed_part():
    """Crane pacing (P6 fixed in absolute terms, lifts included): T1' of bounds_fixed <= R <= T1'' on Okpo, seeds 101 and 102."""
    net, p = net_of("yupu"), scen("yupu", "crane")
    for seed in (101, 102):
        tasks, fx = CB.sample(net, p, seed, 2000)
        t1 = CB.official_t1(net, p, tasks, fx)
        out = RouteFree(net, p, *mixes(tasks, fx)).solve()
        assert t1 <= out["R"] * (1 + 1e-9) <= out["T1pp"] * (1 + 1e-9)


@pytest.mark.parametrize("yard", ["yupu", "yantai"])
def test_R_restored_equals_T1_windows(yard):
    """Implementation check: with every relaxation of R restored (true vehicle length, no cap, turn sweep, destination from
    entering the last road) the occupancy of each task equals the T1' windows (union of bound.task_windows), task by task and
    resource by resource; without restoring, R from two algorithms (first master problem of column generation; per-task
    times + the T1' linear programme) agrees. Main scenario and crane pacing, 2,000 tasks each."""
    net = net_of(yard)
    for name in ("main", "crane"):
        p = scen(yard, name)
        tasks, fx = CB.sample(net, p, 101, 2000)
        rf = RouteFree(net, p, *mixes(tasks, fx))
        f = t1_times(net, p, rf.cap, rf.size)
        for t in tasks + (fx[0] if fx else []):
            a, b = f(t), rf.route_times(t, lv_route=True, caps=False, sweep=True, dest=True)
            if net.stops[t["o"]]["node"] != net.stops[t["d"]]["node"]:
                assert max(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b)) < 1e-9, t
        L = Loads(net, p, sampled_mix(tasks), rf.route_times, rf.res_list)
        L6 = Loads(net, p, sampled_mix(fx[0]), rf.route_times, rf.res_list) if fx else None
        R2 = solve_t1(net, p, rf.cap, rf.size, rf.dummy, L.abar(rf.cap), L.on, (L6.abar(rf.cap), L6.on, fx[1]) if fx else None, ret=rf.ret)[0]
        assert rf.solve(R_only=True)["R"] == pytest.approx(R2, rel=1e-12)


def test_lagrangian_bound_and_duals():
    """The Lagrangian bound is an upper bound for any dual prices: on Okpo seed 101 the UB of the first round (shortest paths
    only) is at least the final T1''; the final UB equals the master problem; the dual feasibility of the empty-run part
    (nu_head - nu_tail <= mu e_a) holds within numerical tolerance."""
    net, p = net_of("yupu"), scen("yupu", "main")
    tasks, _ = CB.sample(net, p, 101, 2000)
    out = RouteFree(net, p, sampled_mix(tasks)).solve()
    assert out["history"][0]["ub"] >= out["T1pp"] * (1 - 1e-12)
    assert out["T1pp"] == pytest.approx(out["master"], rel=1e-7)
    assert out["dual_infeas"] < 1e-9


def test_certificate_robust():
    """The Lagrangian certificate does not depend on the accuracy of the solver's duals: on the Yantai erection mix (seed
    101), after convergence the certificate from the last duals equals the reported T1''; with a large negative component
    added to mu and a perturbation of nu that breaks dual feasibility, the certificate (mu+ first, then nu repaired) is still
    not below T1''."""
    net, p = net_of("yantai"), scen("yantai", "erection")
    tasks, _ = CB.sample(net, p, 101, 2000)
    rf = RouteFree(net, p, sampled_mix(tasks))
    out = rf.solve()
    mu, nu = out["duals"]
    assert rf.ub(mu, nu) == pytest.approx(out["T1pp"], rel=1e-12)
    rng = np.random.default_rng(0)
    for trial in range(5):
        mu2, nu2 = mu.copy(), nu + rng.normal(0, 10 * (np.abs(nu).max() + 1e-9), len(nu))
        mu2[rng.integers(len(mu2))] = -10 * mu.max()
        mu2 += rng.random(len(mu2)) * mu.max() * (trial / 4)
        assert rf.ub(mu2, nu2) >= out["T1pp"] * (1 - 1e-12), trial


def test_t1_on_path_matches_t1():
    """t1_on_path (the T1' windows on any route, used by back-substitution) equals system.t1_times on the simulation's
    shortest paths, task by task and resource by resource."""
    net, p = net_of("yantai"), scen("yantai", "main")
    cap, size = net.capacities(p)
    f = t1_times(net, p, cap, size)
    for t in CB.sample(net, p, 101, 300)[0]:
        a = f(t)
        b = t1_on_path(net, p, cap, size, t, net.path(net.stops[t["o"]]["node"], net.stops[t["d"]]["node"]))
        assert a.keys() == b.keys() and all(a[k] == pytest.approx(b[k], abs=1e-12) for k in a)


def test_backsub_two_routes():
    """Back-substitution: on the two parallel routes the route split of T1'' puts x = b/(a + b) on the narrow road (a, b the
    per-task occupancies of the relaxed rows). Putting back the unrelaxed occupancy (true vehicle length 20 m: narrow road
    315 s, wide road (400 + 25)/v6 x 1/2 = 127.5 s) with this split gives the feasible value
    min(H/(x 315), H/((1 - x) 127.5)) = 634.31, between T1' and T1''; the original solution put back exceeds 1 in a wide-road
    row (where the relaxation raises T1''). With an empty route split (all on shortest paths) back-substitution gives T1'."""
    net, p, tasks = two_routes()
    a, b = (500 + 21.25) / V6, (400 + 21.25) / V6 / 2
    x = b / (a + b)
    mix = sampled_mix(tasks)
    st, rf = SystemT1(net, p, mix), RouteFree(net, p, mix)
    sol = rf.solve()["sol"]
    out = CB.backsub(net, p, st, rf, sol, mix)
    assert out["feasible"] == pytest.approx(min(H / (x * 315), H / ((1 - x) * 127.5)), rel=1e-9)
    assert H / 315 < out["feasible"] < H / a + H / b
    assert out["ratio_max"] == pytest.approx((H / a + H / b) * (1 - x) * 127.5 / H, rel=1e-9) and out["rows_over_1"] >= 1
    assert CB.backsub(net, p, st, rf, dict(sol, z={}), mix)["feasible"] == pytest.approx(H / 315, rel=1e-12)


def test_readout_two_routes():
    """Read-out of the binding resource: on the two parallel routes T1' binds at the narrow road r1; every A -> B task only
    passes it (neither end on r1), share 1; the alternative route avoiding r1 drives 300 m more over the wide road, and the
    remaining slack = 1 - the largest utilisation of the resources on it."""
    net, p, tasks = two_routes()
    mix = sampled_mix(tasks)
    st, rf = SystemT1(net, p, mix), RouteFree(net, p, mix)
    out = CB.readout(net, st, rf)
    util = {tuple(x["res"]): x["util"] for x in st.value(top=len(st.res_list))[1]}
    alt = [("r", "r0"), ("r", "r3"), ("r", "r4"), ("r", "r2"), ("n", "J1"), ("n", "J2")]
    assert out["res"] == ["r", "r1"] and out["pass_share"] == pytest.approx(1.0)
    assert out["alt_extra_m"] == pytest.approx(300.0, abs=1e-9)
    assert out["alt_slack"] == pytest.approx(1 - max(util[r] for r in alt), abs=1e-12)


def test_return_caps():
    """Return caps ret (only on roads with stops): a triangle X, Y, Z of 5 m sides between entrances A and B. After leaving
    road XY at Y the shortest way back onto XY is Y -> Z -> X: 10 m; after A -> X, once round the triangle and back onto
    X -> A: 15 m; with no loop within the limit, the limit. On Okpo (revision 1825) the arcs that may turn back (dead-end
    entrances) have ret = 0; RouteFree caps only roads with stops (net.cuts)."""
    net = load_net(make_map(dict(A=(-100, 0), X=(0, 0), Y=(5, 0), Z=(2.5, 4.330127018922193), B=(100, 0)),
                            [("A", "X"), ("X", "Y"), ("Y", "Z"), ("Z", "X"), ("Y", "B")],
                            {"fA": ("A", "assembly_workshop"), "fB": ("B", "paint_workshop")}, width=12.0))
    xy = next(i for i, a in enumerate(net.arc) if (a[0], a[1]) == ("X", "Y"))
    ax = next(i for i, a in enumerate(net.arc) if (a[0], a[1]) == ("A", "X"))
    assert ret_road(net, xy, 50.0) == pytest.approx(10.0, abs=1e-9)
    assert ret_road(net, ax, 50.0) == pytest.approx(15.0, abs=1e-9)
    assert ret_road(net, ax, 12.0) == 12.0
    yp = net_of("yupu")
    p = scen("yupu", "main")
    rf = RouteFree(yp, p, sampled_mix(CB.sample(yp, p, 101, 50)[0]))
    uturn = [a for a, x in enumerate(yp.arc) if not x[1].startswith("S|") and any(yp.arc[b][3] == x[3] for b in yp.succ[a])]
    assert len(uturn) == 214 and all(ret_road(yp, a, 35.0) == 0.0 for a in uturn)
    assert rf.ret and all(yp.arc[a][3] in yp.cuts for a in rf.ret)
    assert all(rf.ret[a] == 0.0 for a in uturn if a in rf.ret)


def test_empty_tail_cap():
    """Capped empty-run rows (for T1''): on Okpo (revision 1825), 47 arcs on roads with stops have a way back shorter than
    the empty vehicle length + longitudinal clearance (21.25 m); only their coefficients in the road rows shrink, by
    (L + ret)/(L + 21.25); all other empty-run rows equal those of T1'."""
    net, p = net_of("yupu"), scen("yupu", "main")
    rf = RouteFree(net, p, sampled_mix(CB.sample(net, p, 101, 50)[0]))
    _, A0, res = empty_block(net, p, rf.cap, rf.size, rf.dummy)
    _, A1, _ = empty_block(net, p, rf.cap, rf.size, rf.dummy, rf.ret)
    D = (A0 - A1).tocsc()
    changed = {a for a in range(D.shape[1]) if D[:, a].nnz}
    short = {a for a, r in rf.ret.items() if r < 21.25}
    assert changed == short and len(short) == 47 and (A0 - A1).min() >= 0
    a = min(short)
    i = res.index(("r", net.arc[a][3]))
    assert A1[i, a] == pytest.approx(A0[i, a] * (net.arc[a][2] + rf.ret[a]) / (net.arc[a][2] + 21.25), rel=1e-12)


def test_price_conversions_two_routes():
    """Conversions: 10 % less occupancy of the narrow road r1 (capacity x 1/0.9). T1' is bound by r1 only: the exact gain
    H/(0.9 x 315) - H/315 equals the first-order estimate price x 16 h x (1/0.9 - 1) (price = 3600/315). T1'' = H/(0.9 a) + H/b,
    re-solved warm-started from the column pool; relaxing only the wide road r3 leaves r4 binding and T1'' unchanged
    (degenerate: the first-order estimate cannot replace re-solving). Widening r1 by 1 m (capacity 11, a loaded vehicle takes
    11, still exclusive) leaves T1' unchanged. The price table of experiments/capacity_bounds.py carries the same numbers."""
    net, p, tasks = two_routes()
    a, b = (500 + 21.25) / V6, (400 + 21.25) / V6 / 2
    st = SystemT1(net, p, sampled_mix(tasks))
    base, binding = st.value()
    r1 = ("r", "r1")
    top = next(x for x in binding if tuple(x["res"]) == r1)
    assert top["price_per_h"] == pytest.approx(3600 / 315, rel=1e-9)
    d10 = st.value(rhs={r1: 1 / 0.9})[0] - base
    assert d10 == pytest.approx(H / (0.9 * 315) - H / 315, rel=1e-9)
    assert d10 == pytest.approx(top["price_per_h"] * 16 * (1 / 0.9 - 1), rel=1e-9)
    rf = RouteFree(net, p, sampled_mix(tasks))
    out = rf.solve()
    assert rf.solve(rhs={r1: 1 / 0.9}, pool=out["pool"])["T1pp"] == pytest.approx(H / (0.9 * a) + H / b, rel=1e-9)
    assert rf.solve(rhs={("r", "r3"): 1 / 0.9}, pool=out["pool"])["T1pp"] == pytest.approx(out["T1pp"], rel=1e-9)
    cap = dict(st.cap)
    cap[r1] += 1.0
    assert st.value(cap=cap)[0] == pytest.approx(base, rel=1e-12)
    rows = {r["res"]: r for r in CB.price_rows(net, p, st, rf, out["pool"], "main")}
    row = rows["r|r1"]
    assert row["name"] == res_name(net, r1) and row["type"] == "road" and row["W"] == 9.0 and row["L"] == 500.0
    assert row["T1′_price"] == pytest.approx(3600 / 315, rel=1e-9) and row["T1′_d10"] == pytest.approx(d10, rel=1e-9)
    assert row["T1′_d10_first"] == pytest.approx(d10, rel=1e-9) and row["T1′_w1"] == pytest.approx(0.0, abs=1e-9)
    assert row["T1″_d10"] == pytest.approx(H / (0.9 * a) + H / b - out["T1pp"], rel=1e-9)

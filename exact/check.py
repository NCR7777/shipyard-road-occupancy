"""Independent schedule checker and exhaustive search on small instances (does not call exact.model, nor Net.leg / Net.demand /
Sim).

Rebuilds the occupancy of a vehicle from the shortest paths and distances by the definitions of the paper (Section 2), using
only facts of the network (shortest paths, arc lengths, junction sizes, capacities, turn angles, stops, number of cranes):
  decision points: departure, every entry into a new road (adjacent arcs of the same road in the same direction form one
  run), arrival; time = departure + distance driven / speed;
  road: from the decision point that enters it to the next decision point + (vehicle length + longitudinal clearance)/speed;
  junctions and entrance nodes (resources): from the decision point that passes them for (D + vehicle length + longitudinal
  clearance)/speed, plus the sweep time for a turn of at least 60 degrees; an empty run that departs from a junction or an
  entrance node holds it for one tail time;
  origin stop resource: from the decision point at which the empty run enters its last road (from the empty departure if the
  vehicle is already at the origin) to the loaded departure + tail;
  destination stop resource: from the decision point at which the loaded run enters its last road to the end of unloading
  (with clr=True plus the drive-off of the empty vehicle (L0 + s)/v0, as full-path reservation books it);
  crane: tau from the start of unloading, each erection holds one crane, not merged per vehicle.
Widths: road min(B + c, C) (empty vehicle: B = platform width); loaded vehicles hold junctions and entrances alone (C), empty
ones min(w + c, C); stops and cranes 1 each.
The holds of one vehicle on one resource are merged with the maximum width at each instant (Lemma A.3 (a) of the paper).

Scope (as Net.leg writes it, covering only the cases of the sub-network instances): empty-vehicle width rule
empty_rule = "width"; without the road changes at door granularity (door), dock-side waiting places (dock_wait), dock-side
off-road bays (dock_bay), passing places and blocks (passing, blocks); without lifts by a gantry crane (lift). Not checked
under sectioning (section). Outside this scope the windows of the checker no longer equal those of Net.leg item by item and
cannot be used directly."""
import collections
import itertools
import math

TOL = 1e-6


def width(cap, res, b, loaded, p):
    C = cap[res]
    if res[0] in ("b", "c"):
        return 1.0
    if loaded and res[0] == "n":
        return C
    return min(C, b + p["lat_clear"])


def runs(net, path):
    out, D = [], 0.0
    for a in path:
        _, _, L, rid, fw = net.arc[a]
        if out and out[-1][0] == rid and out[-1][1] == fw:
            out[-1][4] = a
        else:
            out.append([rid, fw, D, a, a])
        D += L
    return out, D


def leg(net, cap, size, p, src, dst, v, Lv, b, loaded, start_node):
    """Occupancy of one trip [(resource, width, relative start, relative end)], the driving time and the relative time of
    "entering the last road"."""
    if src == dst:
        return [], 0.0, 0.0
    R, D = runs(net, net.path(src, dst))
    tail = (Lv + p["long_clear"]) / v
    out = []
    if start_node and ("n", src) in cap:
        out.append((("n", src), width(cap, ("n", src), b, loaded, p), 0.0, tail))
    for i, (rid, fw, d0, a0, a1) in enumerate(R):
        th = d0 / v
        nxt = R[i + 1][2] / v if i + 1 < len(R) else D / v
        if i > 0 and net.arc[a0][0] in net.node_res:
            n = net.arc[a0][0]
            sw = p["turn_sweep_s"] if net.turn_deg(R[i - 1][4], a0) >= p["turn_deg"] else 0.0   # degrees, per movement
            out.append((("n", n), width(cap, ("n", n), b, loaded, p), th, th + (size[n] + Lv + p["long_clear"]) / v + sw))
        out.append((("r", rid), width(cap, ("r", rid), b, loaded, p), th, nxt + tail))
    return out, D / v, R[-1][2] / v


def task_holds(net, cap, size, p, task, loc, clr):
    """Occupancy of a task when the vehicle departs from loc (symbolic): [(resource, width, start anchor, start offset, end
    anchor, end offset, is crane)], anchor 'e' = empty departure t1, 'l' = loaded departure t2; plus TE (empty-run time) and DL
    (loaded run + unloading)."""
    so, sd = net.stops[task["o"]], net.stops[task["d"]]
    on, dn = so["node"], sd["node"]
    E, TE, accE = leg(net, cap, size, p, loc, on, p["v_empty"], p["plat_len"], p["plat_w"], False, True)
    L, TL, accL = leg(net, cap, size, p, on, dn, p["v_loaded"], task["Lv"], task["B"], True, False)
    tail = (task["Lv"] + p["long_clear"]) / p["v_loaded"]
    out = [(r, w, "e", a, "e", b, False) for r, w, a, b in E] + [(r, w, "l", a, "l", b, False) for r, w, a, b in L]
    out.append((so["res"], width(cap, so["res"], task["B"], True, p), "e", accE, "l", tail, False))
    end = TL + p["unload_s"] + ((p["plat_len"] + p["long_clear"]) / p["v_empty"] if clr else 0.0)
    out.append((sd["res"], width(cap, sd["res"], task["B"], True, p), "l", accL, "l", end, False))
    if task.get("flow") == "P6" and ("c", sd["ent"]) in cap:
        out.append((("c", sd["ent"]), 1.0, "l", TL, "l", TL + p["crane_cycle_s"], True))
    return out, TE, TL + p["unload_s"]


def check(net, p, tasks, starts, seq, clr=False):
    """Checks the timing vehicle by vehicle and task by task, and the capacity resource by resource at every instant.
    seq = {vehicle: [(task, t1, t2)]} (vehicle k starts at starts[k]). Returns dict(ok, makespan, max_ratio, errors)."""
    cap, size = net.capacities(p)
    errs, done_at, holds = [], {}, collections.defaultdict(list)       # resource -> [(start, end, width, owner)]
    for k, lst in seq.items():
        loc, prev = starts[k], 0.0
        for j, t1, t2 in lst:
            t = tasks[j]
            hs, TE, DL = task_holds(net, cap, size, p, t, loc, clr)
            if t1 < prev - TOL:
                errs.append(f"vehicle {k} task {j}: empty departure {t1} before the previous task finished unloading at {prev}")
            if t2 < t1 + TE + p["load_s"] - TOL:
                errs.append(f"vehicle {k} task {j}: loaded departure {t2} before arrival + loading {t1 + TE + p['load_s']}")
            T = {"e": t1, "l": t2}
            for r, w, a1, a, b1, b, cr in hs:
                holds[r].append((T[a1] + a, T[b1] + b, w, ("c", k, j) if cr else k))
            if j in done_at:
                errs.append(f"task {j} done twice")
            done_at[j] = prev = t2 + DL
            loc = net.stops[t["d"]]["node"]
    if set(done_at) != set(range(len(tasks))):
        errs.append(f"tasks not completed: {sorted(set(range(len(tasks))) - set(done_at))}")
    ratio = 0.0
    for r, iv in holds.items():
        ts = sorted({x for s, e, _, _ in iv for x in (s, e)})
        for a, b in zip(ts, ts[1:]):
            if b - a <= 1e-9:
                continue
            own = collections.defaultdict(float)
            for s, e, w, o in iv:
                if s <= a + 1e-9 and e >= b - 1e-9:
                    own[o] = max(own[o], w)
            load = sum(own.values())
            ratio = max(ratio, load / cap[r])
            if load > cap[r] + TOL:
                errs.append(f"{r} in [{a:.2f}, {b:.2f}] holds {load} > capacity {cap[r]}")
                break
    return dict(ok=not errs, makespan=max(done_at.values()) if done_at else 0.0, max_ratio=ratio, errors=errs[:20])


def busy(net, p, tasks, starts, seq, res=None, clr=False):
    """Share of [0, completion] in which each resource is occupied (at least one vehicle on it); without res the five highest."""
    cap, size = net.capacities(p)
    iv, M = collections.defaultdict(list), 0.0
    for k, lst in seq.items():
        loc = starts[k]
        for j, t1, t2 in lst:
            hs, TE, DL = task_holds(net, cap, size, p, tasks[j], loc, clr)
            T = {"e": t1, "l": t2}
            for r, w, a1, a, b1, b, cr in hs:
                iv[r].append((T[a1] + a, T[b1] + b))
            M = max(M, t2 + DL)
            loc = net.stops[tasks[j]["d"]]["node"]
    out = {}
    for r, xs in iv.items():
        tot, end = 0.0, -math.inf
        for s, e in sorted(xs):
            if e > end:
                tot += e - max(s, end)
                end = e
        out[r] = tot / M
    keep = res if res is not None else sorted(out, key=lambda r: -out[r])[:5]
    return {r: out.get(r, 0.0) for r in keep}


# ---------------------------------------------------------------- exhaustive search on small instances (pairwise conflicting)
def routes(N, K):
    """All ways to assign N tasks to K (distinguishable) vehicles and order them on each vehicle: [[task sequence of vehicle
    0], ...]."""
    for assign in itertools.product(range(K), repeat=N):
        groups = [[j for j in range(N) if assign[j] == k] for k in range(K)]
        for perms in itertools.product(*(itertools.permutations(g) for g in groups)):
            yield [list(x) for x in perms]


def brute_force(net, p, tasks, starts, K, rounded=False):
    """Optimal completion time of schedules that wait only at the origin (exhaustive): enumerates dispatch and order, then for
    every conflicting pair of holds of different vehicles the precedence (branch and bound), and under each precedence takes
    the earliest times by the longest path of the difference constraints. Only for instances in which the holds of any two
    vehicles on the same resource do not fit together (pairwise exclusion is then the capacity constraint; otherwise an error
    is raised).
    rounded: integer optimum with the rounding of exact.model (starts down, ends up, durations up), for a digit-by-digit
    comparison with CP-SAT.
    Returns (optimal completion time, {vehicle: [(task, t1, t2)]})."""
    cap, size = net.capacities(p)
    fl = math.floor if rounded else (lambda x: x)
    ce = math.ceil if rounded else (lambda x: x)
    best = [math.inf, None]
    for rt in routes(len(tasks), K):
        var, cons, wins, ends = {}, [], [], []          # variables: (task, 'e'|'l'); constraints (u, v, c): v - u >= c
        for k, lst in enumerate(rt):
            loc, prev = starts[k], None
            for j in lst:
                hs, TE, DL = task_holds(net, cap, size, p, tasks[j], loc, False)
                cons.append(((j, "e"), (j, "l"), ce(TE + p["load_s"])))
                if prev is not None:
                    cons.append(((prev[0], "l"), (j, "e"), prev[1]))
                prev = (j, ce(DL))
                ends.append(((j, "l"), ce(DL)))
                for r, w, a1, a, b1, b, cr in hs:
                    wins.append((r, w, k, (j, a1), fl(a), (j, b1), ce(b)))
                loc = net.stops[tasks[j]["d"]]["node"]
        pairs = []
        for u, v in itertools.combinations(wins, 2):
            if u[0] == v[0] and (u[2] != v[2] or u[0][0] == "c"):
                if u[1] + v[1] <= cap[u[0]] + TOL:
                    raise ValueError(f"{u[0]}: two vehicles fit, not a pairwise exclusive instance")
                pairs.append((u, v))
        nodes = sorted({x for c in cons for x in c[:2]} | {x for _, _, _, s, _, e, _ in wins for x in (s, e)})
        _branch(nodes, cons, pairs, 0, ends, best, rt)
    return best[0], best[1]


def _earliest(nodes, cons):
    """Least solution of the difference constraints v - u >= c (variables >= 0) (longest path, Bellman-Ford); None if there is
    a positive cycle."""
    t = {n: 0.0 for n in nodes}
    for _ in range(len(nodes) + 1):
        ch = False
        for u, v, c in cons:
            if t[u] + c > t[v] + 1e-9:
                t[v] = t[u] + c
                ch = True
        if not ch:
            return t
    return None


def _branch(nodes, cons, pairs, i, ends, best, rt):
    t = _earliest(nodes, cons)
    if t is None:
        return
    M = max(t[v] + d for v, d in ends)
    if M >= best[0] - 1e-9:
        return
    while i < len(pairs):
        u, v = pairs[i]
        if t[u[5]] + u[6] <= t[v[3]] + v[4] + 1e-9 or t[v[5]] + v[6] <= t[u[3]] + u[4] + 1e-9:
            i += 1              # the pair does not overlap at the earliest times: no branch yet (if later constraints make it overlap, the recursion checks it again)
            continue
        for a, b in ((u, v), (v, u)):     # a before b: start of b - end of a >= 0
            _branch(nodes, cons + [(a[5], b[3], a[6] - b[4])], pairs, 0, ends, best, rt)
        return
    best[0] = M
    best[1] = {k: [(j, t[(j, "e")], t[(j, "l")]) for j in lst] for k, lst in enumerate(rt) if lst}

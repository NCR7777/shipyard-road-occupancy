"""Synthetic dock mouth (paper, Section 4.5, Supplementary S2.2 and Fig. 8): a crane, a single-lane dock road and one adjacent stop.
The passable margin beside a standing loaded vehicle is a parameter; the crane utilisation rho is swept, and the threshold
rho_10 at which throughput collapses is read under two kinds of rule: waiting on the road (stepwise requests with safe
release, whole or sectioned dock road) and waiting off the road (full-path reservation; safe release with a dock-side off-road
bay, where a vehicle waits for the crane in the bay).

Network: A (vehicle depot entrance, where vehicles start) - r0 (20 m) - J0 - rs (road of the origin area, 120 m) - J - r1
(dock road, 300 m, width W) - dead end. r0 and rs have no width limit; rs has 6 drive-in loading bays (the origin area, never a
bottleneck). At 150 m on r1 is the roadside stop of the dock (one crane, each erection holds it 10 + 180 min), at 250 m the
drive-in bay of the adjacent area (one place): vehicles to the adjacent area pass beside a vehicle standing at the dock.
Tasks: P6 (origin area -> dock) released at the crane pace at equal intervals (day_s / n6, as occupancy.tasks.crane_tasks)
and dispatched first; background tasks (origin area -> adjacent area) saturated (a vehicle gets one whenever it is free and no
P6 waits; random loading bay). Blocks of width B = 7 m on vehicles 20 m long: a loaded vehicle takes 8 m, an empty one 6.5 m.
Passable margin m = dock road capacity W + 1 - 8 = W - 7, in steps of the empty vehicles that fit beside a standing loaded
one: one if m >= 6.5 m, two if m >= 13 m (under this rule loaded vehicles never pass a standing one).
rho = n6 x 190 min / 960 min (one crane, 16-h day).
rho_10: the rho at which the total throughput of the statistics window (days 2-8) has fallen 10 % below that at rho = 0.3
(linear between adjacent levels). The crane side always keeps up (P6 completed / released >= 0.95, the secondary criterion,
never fails), so the loss falls on the background tasks.

Run:   python experiments/dock_mouth.py --out <dir> [--procs N] [--nbar]
Writes <dir>/points/<key>.json (cache), points.csv and rho_star.csv; results/synthetic_dock/ holds the stored results and
their configuration. --nbar computes only NBAR_KEYS with the occupancy log and adds the waiting term nbar
(results/runs/synthetic_dock_waiting/).
"""
import argparse
import collections
import csv
import io
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import runner as R                         # noqa: E402
from occupancy import params as P         # noqa: E402
from occupancy.mapgraph import to_graph   # noqa: E402
from occupancy.network import Net         # noqa: E402

SC, RS = "org.shipyard.spatial_classification", "org.shipyard.research_semantics"
L1, DOCK_S, YARD_S = 300.0, 150.0, 250.0
WIDTHS = (9.0, 12.0, 14.0, 15.0, 17.0, 19.0, 20.0, 21.0, 23.0)      # margins 2, 5, 7, 8, 10, 12, 13, 14, 16 m
RHOS = (0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99)
KS = (4, 8, 12, 16, 20, 24, 32)
SRC = (10.0, 30.0, 50.0, 70.0, 90.0, 110.0)       # loading bays of the origin area (chainage on rs)
RULES = {"seg": ("segment_safe", {}), "seg_s": ("segment_safe", {"section": "stops"}),
         "seg_bay": ("segment_safe", {"retrofit": [{"kind": "dock_bay", "cap": 1}]}), "res": ("reserve", {})}
RULE_NAME = {"seg": "stepwise + safe release (waiting on the road, whole road)",
             "seg_s": "stepwise + safe release (waiting on the road, dock road sectioned)",
             "seg_bay": "stepwise + safe release + dock-side off-road bay (waiting off the road for the crane)",
             "res": "full-path reservation (waiting off the road, waiting at the origin)"}
SEEDS = tuple(range(101, 111))
DAYS = 8
FAIL = 0.95


def dock_map(W, L=L1, yard_s=(YARD_S,)):
    """Synthetic map (map format of occupancy.mapgraph; the junctions allow every turn except U-turns). yard_s: chainages of
    the drive-in bays of the adjacent area (default one, at 250 m)."""
    node = lambda x, y, kind="ordinary": dict(kind=kind, position=[x, y, 0], name="")
    road = lambda a, b, w: dict(fromNodeId=a, toNodeId=b, name=f"{a}-{b}", direction="both",
                                geometry=dict(kind="path", anchors=[], spans=[dict(kind="line")]),
                                widthM=dict(state="known", value=w) if w else dict(state="unrestricted"), extensions={})
    nodes = dict(A=node(-140.0, 0.0, "access"), J0=node(-120.0, 0.0), J=node(0.0, 0.0), E=node(L, 0.0))
    roads = dict(r0=road("A", "J0", None), rs=road("J0", "J", None), r1=road("J", "E", W))
    mv = lambda i, j, a, b: (f"m{i}", dict(allowed=True, junctionId=j, incomingArc=dict(roadId=a[0], direction=a[1]),
                                            outgoingArc=dict(roadId=b[0], direction=b[1])))
    movements = dict([mv(0, "jJ0", ("r0", "forward"), ("rs", "forward")), mv(1, "jJ0", ("rs", "backward"), ("r0", "backward")),
                      mv(2, "jJ", ("rs", "forward"), ("r1", "forward")), mv(3, "jJ", ("r1", "backward"), ("rs", "backward"))])
    y0 = W / 2 + 2.0
    dock = dict(name="dock", boundary=dict(outer=[[DOCK_S - 10, y0, 0], [DOCK_S + 10, y0, 0], [DOCK_S + 10, y0 + 60, 0], [DOCK_S - 10, y0 + 60, 0]]),
                extensions={SC: {"classId": "dry_dock"}, RS: {"cranes": [{"count": 1, "countsForErection": True}]}})
    fr = lambda rid, ss: {RS: {"frontage": [dict(roadId=rid, sStartM=s - 5, sEndM=s + 5) for s in ss]}}
    yard = dict(name="adjacent area", extensions={SC: {"classId": "yard"}, **fr("r1", list(yard_s))})
    src = dict(name="origin area", extensions={SC: {"classId": "assembly_yard"}, **fr("rs", SRC)})
    return dict(nodes=nodes, roads=roads, movements=movements,
                junctions=dict(jJ0=dict(nodeIds=["J0"], model="explicit_movements"), jJ=dict(nodeIds=["J"], model="explicit_movements")),
                zones=dict(zdock=dock, zyard=yard, zsrc=src), accessPoints=dict(aA=dict(nodeId="A")),
                facilities=dict(fA=dict(name="depot", accessPointIds=["aA"], extensions={SC: {"classId": "assembly_workshop"}})))


def params(rule, days=DAYS):
    """Simulation parameters: crane pacing (cranes from the map), P6 not drawn by the task stream (the tasks are given);
    the extra parameters of the rule (sectioning, bay)."""
    mode, extra = RULES[rule]
    return mode, dict(P.DEFAULTS, **P.SCENARIOS["crane"], p6_per_day=1.0, days=days, **extra)


NETS = {}


def make_net(W, p):
    k = (W, json.dumps(p.get("retrofit")), p.get("section"))
    if k not in NETS:
        NETS[k] = Net(to_graph(dock_map(W)), p)
    return NETS[k]


SRC_IDS = [f"zsrc#{k}" for k in range(len(SRC))]


def p6_tasks(seed, rho, days=DAYS, day_s=16 * 3600.0):
    """P6 (origin area -> dock) released at the crane pace at equal intervals (as crane_tasks): n6 = rho x 960 / 190 a day,
    random loading bay; B = 7 m, vehicle length 20 m."""
    rng = random.Random(seed * 100 + 6)
    n6 = rho * day_s / (190 * 60.0)
    return [dict(id=10 ** 6 + j, flow="P6", o=rng.choice(SRC_IDS), d="zdock#0", B=7.0, Lv=20.0, release=(j + 0.5) * day_s / n6)
            for j in range(int(days * n6))]


class Background:
    """Background task stream (origin area -> adjacent area, saturated: a vehicle gets one whenever it is free and no P6
    waits, like the saturated level of TaskStream); random origin; the sequence of a seed does not depend on rho or the rule
    (common random numbers)."""

    def __init__(self, seed):
        self.rng, self.n = random.Random(seed), 0

    def next(self, release):
        self.n += 1
        return dict(id=self.n, flow="BG", o=self.rng.choice(SRC_IDS), d="zyard#0", B=7.0, Lv=20.0, release=release)


def compute(key, nbar=False):
    """One point: key = rule|W|rho|K|seed. nbar: also the waiting term from the occupancy log (nbar_cols; the simulation
    itself is unchanged)."""
    from occupancy.sim import Sim
    rule, W, rho, K, seed = key.split("|")
    W, rho, K, seed = float(W), float(rho), int(K), int(seed)
    mode, p = params(rule)
    net = make_net(W, p)
    ts = p6_tasks(seed, rho)
    t0 = time.time()
    sim = Sim(net, p, mode, K, seed, None, record=nbar, tasks=ts)
    sim.saturated, sim.stream = True, Background(seed)        # background saturated; released P6 go first
    sim.run()
    m = sim.metrics()
    lo, hi = sim.t0m, sim.H
    nd = (hi - lo) / p["day_s"]
    rel6 = sum(lo <= t["release"] < hi for t in ts) / nd
    done = {f: sum(lo <= r["done"] <= hi for r in sim.done if r["flow"] == f) / nd for f in ("P6", "BG")}
    return dict(key=key, rule=rule, W=W, margin=W + p["lat_clear"] - (7.0 + p["lat_clear"]), rho=rho, K=K, seed=seed,
                thr=done["P6"] + done["BG"], rel_P6=rel6, done_P6=done["P6"], done_BG=done["BG"], ratio_P6=done["P6"] / rel6 if rel6 else None,
                starve=(m.get("starve") or {}).get("starve_max"), starve_blk=(m.get("starve") or {}).get("blocked_max"),
                wait_mean=m["wait_mean"], waiting_end=m["waiting_end"], dock0_busy=m["dock"]["ring0"]["busy_max"],
                p6_backlog_end=sum(1 for t in sim.queue if t["flow"] == "P6"), **(nbar_cols(sim, p) if nbar else {}),
                secs=round(time.time() - t0, 2))


# waiting term of Proposition 5 (nbar, the time-mean number of vehicles waiting on the road for the crane): safe release,
# whole road, margins below 8 m (W = 9, 12, 14), rho = 0.99, K = 16 and 32; also rho = 0.6, margin 2 m; ten seeds. The runs are
# those of the dock-mouth points with the same keys (only the occupancy log is added).
NBAR_KEYS = tuple(f"seg|{W:g}|{rho}|{K}|{s}" for rho, Ws in ((0.99, (9.0, 12.0, 14.0)), (0.6, (9.0,))) for W in Ws for K in (16, 32)
                  for s in SEEDS)
LOADED_W = 8.0             # width of a loaded vehicle B + c (B = 7 m)


def union_len(iv, lo, hi):
    """Length of the union of intervals within [lo, hi]."""
    iv = sorted((max(s, lo), min(e, hi)) for s, e in iv if min(e, hi) > max(s, lo))
    tot, cs, ce = 0.0, None, None
    for s, e in iv:
        if cs is None or s > ce:
            if cs is not None:
                tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return tot + (ce - cs if cs is not None else 0.0)


def nbar_cols(sim, p):
    """nbar: for every completed P6, the interval from its arrival at the dock (start of its loaded occupancy of r1 + 150 m /
    v_loaded) to the start of the erection; the length of their union within the statistics window / window length. With a
    margin below 8 m at most one loaded vehicle is on r1, so the union is the sum and nbar the time-mean number of vehicles
    waiting on the road for the crane. nbar_waitby: the crane part of Sim.wait_by / window length (waits ending in the window,
    not clipped), for comparison."""
    lo, hi, v1 = sim.t0m, sim.H, p["v_loaded"]
    loaded = [(s, e, v) for res, s, e, d, v in sim.log if res == ("r", "r1") and e > s and d >= LOADED_W - 1e-6]
    waits = []
    for r in sim.done:
        if r["flow"] == "P6":
            cand = [x for x in loaded if x[0] <= r["cs"] - DOCK_S / v1 + 1e-6 and x[1] >= r["cs"] - 1e-6]
            if cand:
                waits.append((max(cand)[0] + DOCK_S / v1, r["cs"]))
    return dict(nbar=union_len(waits, lo, hi) / (hi - lo), n_waits=len(waits),
                nbar_waitby=sum(x for r, x in sim.wait_by.items() if r[0] == "c") / (hi - lo))


NBAR = False


def init(nbar=False):
    global NBAR
    NBAR = nbar
    R.pin(list(range(1, os.cpu_count())) or [0])


def run_one(key):
    try:
        return key, compute(key, NBAR), None
    except Exception as e:                       # noqa: BLE001
        return key, None, repr(e)


# total throughput 10 % below the lowest rho level of the group (main criterion); P6 completed / released < 0.95 (secondary)
CRIT = {"thr": 0.9, "ratio_P6": FAIL}


def rho_star(rows, crit="ratio_P6"):
    """{(rule, W, K): rho*}: the rho at which the ten-seed mean first falls below the criterion (linear with the previous
    level); thr is relative to the lowest rho of the group. None if it never does (beyond the highest level)."""
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["rule"], r["W"], r["K"], r["rho"])].append(r[crit])
    out = {}
    for rule in RULES:
        for W in WIDTHS:
            for K in KS:
                xs = [(rho, sum(g[(rule, W, K, rho)]) / len(g[(rule, W, K, rho)])) for rho in RHOS if g.get((rule, W, K, rho))]
                if crit == "thr" and xs:
                    xs = [(rho, v / xs[0][1]) for rho, v in xs]
                star, lim = None, CRIT[crit]
                for (r0, v0), (r1, v1) in zip(xs, xs[1:]):
                    if v0 >= lim > v1:
                        star = r0 + (r1 - r0) * (v0 - lim) / (v0 - v1)
                        break
                if xs and xs[0][1] < lim:
                    star = xs[0][0]
                out[(rule, W, K)] = dict(star=star, curve=xs)
    return out


def keys_of(nbar):
    return list(NBAR_KEYS) if nbar else [f"{r}|{W:g}|{rho}|{K}|{s}" for r in RULES for W in WIDTHS for rho in RHOS for K in KS for s in SEEDS]


def point_row(r, columns):
    """One point as the string cells of points.csv, restricted to `columns`."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, list(r))
    w.writeheader()
    w.writerow(r)
    rd = list(csv.DictReader(io.StringIO(buf.getvalue())))[0]
    return {c: rd.get(c, "") for c in columns}


def rerun(cfg, keys, procs):
    """Recompute stored points (experiments/rerun.py): {key: result}; nbar columns where the configuration logs occupancy."""
    import multiprocessing as mp
    out = {}
    with mp.Pool(min(procs, len(keys)), initializer=init, initargs=(bool(cfg.get("record")),)) as pool:
        for k, r, err in pool.imap_unordered(run_one, keys):
            if err:
                raise RuntimeError(f"{k}: {err}")
            out[k] = r
    return out


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--nbar", action="store_true", help="only NBAR_KEYS, with the waiting term nbar")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "points").mkdir(parents=True, exist_ok=True)
    keys = keys_of(a.nbar)
    todo = [k for k in keys if not (out / "points" / (k.replace("|", "_") + ".json")).exists()]
    todo.sort(key=lambda k: (-int(k.split("|")[3]), -float(k.split("|")[2])))
    print(f"{len(todo)} of {len(keys)} points to compute in {out}", flush=True)
    with mp.Pool(a.procs, initializer=init, initargs=(a.nbar,)) as pool:
        for i, (k, r, err) in enumerate(pool.imap_unordered(run_one, todo), 1):
            if err:
                sys.exit(f"failed {k}: {err}")
            R.write_json(out / "points" / (k.replace("|", "_") + ".json"), r)
            print(f"{i}/{len(todo)} {k}", flush=True)
    rows = [json.loads(f.read_text(encoding="utf-8")) for f in (out / "points").glob("*.json")]
    rows.sort(key=lambda r: (r["rule"], r["W"], r["K"], r["rho"], r["seed"]))
    with open(out / "points.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if a.nbar:
        return
    stars = {c: rho_star(rows, c) for c in CRIT}
    with open(out / "rho_star.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["rule", "rule_name", "W", "margin", "K", "rho_star_thr10", "rho_star_P6"] + [f"thr_rel@{r}" for r in RHOS]
                   + [f"P6ratio@{r}" for r in RHOS])
        for (rule, W, K), x in stars["thr"].items():
            st = [stars[c][(rule, W, K)]["star"] for c in CRIT]
            w.writerow([rule, RULE_NAME[rule], W, W - 7.0, K] + [None if s is None else round(s, 3) for s in st]
                       + [round(v, 3) for _, v in x["curve"]] + [round(v, 3) for _, v in stars["ratio_P6"][(rule, W, K)]["curve"]])


if __name__ == "__main__":
    main()

"""Why the binding road with the corridor widened to 14 m, road099-4, has no sub-network instance comparable with those of the
exact sub-network study (and why the holding comparison of the interventions reports only the whole network for it).
In the task stream of the main case (conserved_mapping.stream, seeds 101-103, the first 20,000 tasks each), the tasks whose
loaded shortest path passes the road: their number per thousand tasks and their flows, the distribution of the network
distances from the origin to the nearer end of the road and from the nearer end of the road to the destination, and the number
of tasks with both distances within R; compared with the through corridor road010-6 (current layout). The ball sub-network of
road099-4 is also enlarged to a radius of 1,500 m (the first radius at which some passing task has both ends within R): size of
the sub-network, the binding elements of lambda_W^N on the 60 tasks of seed 101, the fleet bound lambda^F with K = 8 and
full-path reservation, compared with lambda^F / lambda_W^N of the sub-networks of the exact study (which takes K = 8 so that
every sub-network is network-bound).

Run:   python experiments/intervention_diag.py --out <dir> --subnet <points.csv of experiments/subnet_exact.py>
       with the released table: --subnet results/subnet/points.csv (columns inst, seed, N, lamF_N)
Writes <dir>/diag.json, released as results/interventions/diag.json.
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import runner as F                        # noqa: E402
import intervention_net as N              # noqa: E402

DRAWS, RS, R_BIG = 20000, (400, 600, 800, 1000, 1500, 2000), 1500.0


def via_stats(scen, road):
    import conserved_mapping
    from exact import model as X
    p = N.params("yupu", scen)
    net = N.net("yupu", p)
    rid = next(k for k, r in net.road.items() if r["name"] == road)
    a, b = net.road[rid]["a"], net.road[rid]["b"]
    ball = X.ball_roads(net, {a, b}, 400.0)
    stops = X.sub_stops(net, ball)
    d_o, d_d, flows, n = [], [], {}, 0
    for sd in (101, 102, 103):
        st = conserved_mapping.stream(net, sd, p)
        for _ in range(DRAWS):
            t = st.next(0.0)
            o, d = net.stops[t["o"]]["node"], net.stops[t["d"]]["node"]
            if rid not in X.path_roads(net, o, d):
                continue
            n += 1
            flows[t["flow"]] = flows.get(t["flow"], 0) + 1
            d_o.append(min(net.dist(o, a), net.dist(o, b)))
            d_d.append(min(net.dist(a, d), net.dist(b, d)))
    q = lambda xs, f: sorted(xs)[int(f * (len(xs) - 1))]
    both = [max(u, v) for u, v in zip(d_o, d_d)]
    return dict(road=road, scen=scen, length_m=net.road[rid]["L"], ball_roads=len(ball), ball_stops=len(stops), draws=3 * DRAWS, via=n,
                via_per_1000=1000 * n / (3 * DRAWS), flows=flows,
                o_dist=dict(median=statistics.median(d_o), p10=q(d_o, 0.1), p90=q(d_o, 0.9), min=min(d_o)),
                d_dist=dict(median=statistics.median(d_d), p10=q(d_d, 0.1), p90=q(d_d, 0.9), min=min(d_d)),
                both_within={R: sum(1 for x in both if x <= R) for R in RS})


def big_ball(subnet_points):
    """The enlarged sub-network of road099-4; subnet_points: points.csv of the exact sub-network study (lamF_N of seed 101,
    N = 60)."""
    import intervention_subnet as S
    from exact import model as X
    p = N.params("yupu", "w14")
    net = N.net("yupu", p)
    rid = next(k for k, r in net.road.items() if r["name"] == "road099-4")
    roads = X.ball_roads(net, {net.road[rid]["a"], net.road[rid]["b"]}, R_BIG)
    tasks = S.sample_tasks(net, p, 101, roads, 60, None, [rid])
    inst = X.Instance("intervention_diag", net, p, tasks, X.vehicle_starts(net, tasks, 101, 8), roads)
    mk, _, _ = X.run_fpr(inst, 8)
    lam, bind = X.lam_window(inst, mk)
    sub = {r["inst"]: r for r in csv.DictReader(open(subnet_points, encoding="utf-8-sig"))
           if r["seed"] == "101" and r["N"] == "60"}
    return dict(R=R_BIG, roads=len(roads), roads_yard=len(net.road), stops=len(X.sub_stops(net, roads)), lamN=lam,
                binding=[(x["name"], x["util"]) for x in bind[:5]], fpr=60 * p["day_s"] / mk,
                lamF=X.lam_fleet(inst, 8), loaded_median_m=statistics.median(net.dist(net.stops[x["o"]]["node"], net.stops[x["d"]]["node"]) for x in tasks),
                subnet_lamF_over_lamN={k: float(sub[k]["lamF_N"]) for k in ("yupu_corridor", "yantai_paint", "yupu_dock1")})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--subnet", required=True, help="points.csv of experiments/subnet_exact.py (released: results/subnet/points.csv)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = dict(script="experiments/intervention_diag.py", r0994=via_stats("w14", "road099-4"), corridor=via_stats("base", "road010-6"),
               big_ball=big_ball(a.subnet))
    F.write_json(out / "diag.json", res)
    print(json.dumps(res, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()

"""Roadside stops within a clearing length of a resource boundary behind them (paper, Appendix A.1).

A roadside stop is counted when, with the vehicle's front at the stop, at least one entering direction has a resource
boundary within the clearing length behind it: a junction or entrance node n at D_n + L + s, a section boundary or the
joint of two roads at L + s (s = longitudinal clearance). There the body of a vehicle that loads or unloads at the stop can
still stand on the resource behind when its timer runs out. L is the platform length L0 (within_platform_clearing) or a
30-m block (within_block_clearing); all roadside stops are counted as well. The walk back from the stop is the model's
own, tail_booking.behind (straight on along the graph road, at junctions the incoming road with the smallest turn).

  python experiments/roadside_stops.py --out DIR [--configs CFG ...]
  -> DIR/roadside_stops.json (whole roads, the keys of results/tail/roadside_stops.json) and
     DIR/roadside_stops_sectioned.json (the same counts with the roads that carry roadside stops sectioned at the stops).
The networks and parameters are those of the stored configurations (default: the saturation runs of the main case of
Okpo and Yantai in results/runs)."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from occupancy.network import Net               # noqa: E402
from tail_booking import TOL, behind            # noqa: E402

BLOCK_L = 30.0
CONFIGS = [ROOT / "results" / "runs" / f"{y}_main_saturation" / "config.json" for y in ("okpo", "yantai")]
ABOUT = ("Roadside stops of the {net}: all of them (roadside_stops), and those less than a clearing length beyond the node "
         "behind them, where a vehicle's body can still stand on the resource behind it when its timer runs out: with the "
         "platform length (within_platform_clearing) and with a 30-m block (within_block_clearing).")


def within(net, cap, size, stop, L, s):
    """Is there, for some entering direction, a resource boundary behind the front at the stop within the clearing length?"""
    reach = L + s + max(size.values())
    return any((size[r[1]] if r[0] == "n" else 0.0) + L + s - dist > TOL
               for fw in (True, False) for r, dist in behind(net, cap, stop["node"], stop["res"][1], fw, 0.0, reach))


def counts(cfg, section):
    p = dict(cfg["params"], section=section)
    net = Net(cfg["network"], p, cfg.get("roles"))
    cap, size = net.capacities(p)
    road = [st for st in net.stops.values() if st["mode"] == "on_road"]
    n = lambda L: sum(within(net, cap, size, st, L, p["long_clear"]) for st in road)
    return dict(within_platform_clearing=n(p["plat_len"]), within_block_clearing=n(BLOCK_L), roadside_stops=len(road))


def main():
    ap = argparse.ArgumentParser(description="Roadside stops within a clearing length of a resource boundary behind them.")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--configs", nargs="+", default=[str(c) for c in CONFIGS], help="stored run configurations (one per yard)")
    a = ap.parse_args()
    cfgs = [json.loads(Path(c).read_text(encoding="utf-8")) for c in a.configs]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, section, net in (("roadside_stops.json", None, "whole-road networks"),
                               ("roadside_stops_sectioned.json", "stops", "networks sectioned at the stops")):
        res = dict(about=ABOUT.format(net=net), block_length_m=int(BLOCK_L), **{c["yard"]: counts(c, section) for c in cfgs})
        (out / name).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
        print(name, json.dumps({k: v for k, v in res.items() if k != "about"}))


if __name__ == "__main__":
    main()

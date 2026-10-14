"""Shipyard road networks: topology, road lengths, turn permissions, stops of origins and destinations, exclusive resources
and the resources one trip occupies.

A network is a derived graph (see data/networks/README.md and read_graph): nodes, roads with their centre-line lengths and
widths, junctions with the allowed movements and the turn angle of each movement, entrances of buildings, zones with their
frontage stretches along roads, the cranes of each dock and the share of each assembly area inside a crane span. It holds no
coordinates; occupancy.mapgraph derives such a graph from a map with geometry.

Resources (keys are tuples):
  ('r', road)      every road; capacity = width W + c (a width container: each vehicle takes its outer width + c). With
                   sectioning (p["section"]) a long road is cut into sections, one resource each, named <road>@<k>
                   (Net._section);
  ('n', node)      junctions of degree >= 3 and all entrance nodes; capacity = junction size D + c, D = widest adjacent road
                   with a known width;
  ('b', zone, k)   drive-in bays, one vehicle each; dock-side off-road bays and in-door bays (road changes) have capacity =
                   number of places, each vehicle takes 1;
  ('w', zone, k)   dock-side waiting places (road change dock_wait), capacity = number of places, each vehicle takes 1.
A loaded vehicle holds a junction alone; on a road vehicles share by width: two can be on one road iff the sum of their outer
widths + c <= W.
Road changes (p["retrofit"] = [change, ...], applied in order and combinable; without them the network is unchanged):
  widen     {"kind": "widen", "roads": [road], "add": m or "to": m}: add m to the width, or raise it to at least m (known
            widths only; the size of the end junctions follows the junction rule);
  entrance  {"kind": "entrance", "facility": f, "n": 1}: n more entrances for a building, copies of its first entrance (a new
            entrance node at the same place, an access road of the same length to the same junction, the same movements);
  dock_bay  {"kind": "dock_bay", "cap": c[, "docks": [zone]]}: dock-side off-road bays of capacity c: the roadside stop of a
            dock becomes a drive-in bay beside the dock where vehicles unload and wait for the crane, so the dock road is held
            only while passing. Without crane resources the c places are c parallel unloading places; with cranes, unloading
            (hooking) is limited by the number of cranes and extra places can only wait. All docks if no list is given;
  dock_wait {"kind": "dock_wait", "cap": c[, "docks": [zone]]}: dock-side waiting places: the roadside stop stays on the road,
            one vehicle unloads at a time, and c off-road places beside it let reservation rules wait there until the stop
            (and the crane) is free (Sim._book_dest); occupancy without waiting is unchanged, and so are the bounds. Stepwise
            rules do not use the waiting places;
  passing   {"kind": "passing", "road": r[, "len": 40, "w": width, "at": chainage]}: a passing place on a long single-lane
            road (length len, width w, by default the equivalent width of a road without width limit); centred at `at`,
            or at the middle of the longest stretch between the stops and the ends of the road; its ends are section
            boundaries;
  blocks    {"kind": "blocks", "road": r, "len": m}: cut the whole road into blocks of length <= m (forced boundaries, width
            unchanged; no boundary within 1 m of a stop);
  door      {"kind": "door", "facilities": [f]}: door granularity: a door (entrance node) is held only while a vehicle passes,
            (D + vehicle length + longitudinal clearance)/v; loading and unloading take place at an in-door bay (one per
            building, capacity = number of doors); the access roads in front of the doors are shared by width as before.
"""
import collections
import copy
import csv
import heapq
import json
import math
from pathlib import Path

from .params import DOCK_CLASS, FLOW, STOP_MODE

EPS = 1e-6
MAP_KINDS = ("widen", "entrance")          # road changes that alter the graph itself (retrofit_map); the others alter stops and sections
FRONTAGE_KEYS = ("frontage_gap", "frontage_min_len", "frontage_merge")
ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- derived graph: files
def read_graph(path):
    """Derived graph of a yard from the directory `path` (data/networks/<yard>/<revision>), or a path relative to
    data/networks. Returns the in-memory graph that Net takes (see write_graph for the layout)."""
    d = Path(path)
    if not d.is_absolute() and not d.exists():
        d = ROOT / "data" / "networks" / d
    rd = lambda name: list(csv.DictReader(open(d / name, encoding="utf-8", newline="")))
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    G = dict(yard=meta["yard"], revision=meta["revision"], frontage_rule=meta["frontage_rule"])
    G["nodes"] = {r["node_id"]: dict(name=r["name"], kind=r["kind"]) for r in rd("nodes.csv")}
    G["roads"] = {r["road_id"]: dict(fromNodeId=r["from_node"], toNodeId=r["to_node"], name=r["name"], lengthM=float(r["length_m"]),
                                     widthM=dict(state="known", value=float(r["width_m"])) if r["width_state"] == "known"
                                     else dict(state=r["width_state"]))
                  for r in rd("roads.csv")}
    G["junctions"] = {r["junction_id"]: dict(nodeIds=[r["node_id"]]) for r in rd("junctions.csv")}
    G["movements"] = {r["movement_id"]: dict(junctionId=r["junction_id"], allowed=r["allowed"] == "true", turnDeg=float(r["turn_deg"]),
                                             incomingArc=dict(roadId=r["in_road"], direction=r["in_dir"]),
                                             outgoingArc=dict(roadId=r["out_road"], direction=r["out_dir"]))
                      for r in rd("movements.csv")}
    G["accessPoints"] = {r["access_id"]: dict(nodeId=r["node_id"], facilityId=r["facility_id"], name=r["name"])
                         for r in rd("access_points.csv")}
    G["facilities"] = {r["facility_id"]: dict(name=r["name"], classId=r["class"] or None, accessPointIds=r["access_points"].split())
                       for r in rd("facilities.csv")}
    G["zones"] = {r["zone_id"]: dict(name=r["name"], classId=r["class"] or None, frontage=[], cranes=[], span=None)
                  for r in rd("zones.csv")}
    for r in rd("frontage.csv"):
        G["zones"][r["zone_id"]]["frontage"].append((r["road_id"], float(r["s_start_m"]), float(r["s_end_m"])))
    for r in rd("cranes.csv"):
        G["zones"][r["dock_zone_id"]]["cranes"].append(dict(type=r["type"], count=int(r["count"]),
                                                            countsForErection=r["counts_for_erection"] == "true"))
    for r in rd("crane_spans.csv"):
        s = G["zones"][r["span_zone_id"]]["span"] = G["zones"][r["span_zone_id"]]["span"] or dict(dock=r["dock_zone_id"], covers={})
        if r["covered_zone_id"]:
            s["covers"][r["covered_zone_id"]] = float(r["share"])
    return G


def write_graph(G, path):
    """Write a derived graph as CSV tables and meta.json (field descriptions in data/networks/README.md)."""
    d = Path(path)
    d.mkdir(parents=True, exist_ok=True)

    def wr(name, head, rows):
        with open(d / name, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(head)
            w.writerows(rows)
    tf = lambda x: "true" if x else "false"
    (d / "meta.json").write_text(json.dumps(dict(yard=G["yard"], revision=G["revision"], frontage_rule=G["frontage_rule"]),
                                            indent=1) + "\n", encoding="utf-8")
    wr("nodes.csv", ["node_id", "name", "kind"], [[k, v["name"], v["kind"]] for k, v in G["nodes"].items()])
    wr("roads.csv", ["road_id", "name", "from_node", "to_node", "length_m", "width_m", "width_state"],
       [[k, v["name"], v["fromNodeId"], v["toNodeId"], repr(float(v["lengthM"])),
         repr(float(v["widthM"]["value"])) if v["widthM"]["state"] == "known" else "", v["widthM"]["state"]] for k, v in G["roads"].items()])
    wr("junctions.csv", ["junction_id", "node_id"], [[k, v["nodeIds"][0]] for k, v in G["junctions"].items()])
    wr("movements.csv", ["movement_id", "junction_id", "in_road", "in_dir", "out_road", "out_dir", "allowed", "turn_deg"],
       [[k, v["junctionId"], v["incomingArc"]["roadId"], v["incomingArc"]["direction"], v["outgoingArc"]["roadId"],
         v["outgoingArc"]["direction"], tf(v["allowed"]), repr(float(v["turnDeg"]))] for k, v in G["movements"].items()])
    wr("access_points.csv", ["access_id", "name", "node_id", "facility_id"],
       [[k, v["name"], v["nodeId"], v["facilityId"]] for k, v in G["accessPoints"].items()])
    wr("facilities.csv", ["facility_id", "name", "class", "access_points"],
       [[k, v["name"], v["classId"] or "", " ".join(v["accessPointIds"])] for k, v in G["facilities"].items()])
    wr("zones.csv", ["zone_id", "name", "class"], [[k, v["name"], v["classId"] or ""] for k, v in G["zones"].items()])
    wr("frontage.csv", ["zone_id", "k", "road_id", "s_start_m", "s_end_m"],
       [[z, k, r, repr(float(a)), repr(float(b))] for z, v in G["zones"].items() for k, (r, a, b) in enumerate(v["frontage"])])
    wr("cranes.csv", ["dock_zone_id", "type", "count", "counts_for_erection"],
       [[z, c["type"], c["count"], tf(c["countsForErection"])] for z, v in G["zones"].items() for c in v["cranes"]])
    wr("crane_spans.csv", ["span_zone_id", "dock_zone_id", "covered_zone_id", "share"],
       [[z, v["span"]["dock"], e, repr(float(f))] for z, v in G["zones"].items() if v["span"]
        for e, f in (v["span"]["covers"].items() or [("", 0.0)])])


def retrofit_map(D, specs):
    """Road changes on the graph itself (widen, entrance; see the module docstring): returns a changed deep copy, or D itself
    when there are none of these two kinds."""
    if not any(s["kind"] in MAP_KINDS for s in specs):
        return D
    D = copy.deepcopy(D)
    for s in specs:
        if s["kind"] == "widen":
            for rid in s["roads"]:
                w = D["roads"][rid]["widthM"]
                if w["state"] == "known":
                    w["value"] = w["value"] + s["add"] if "add" in s else max(w["value"], s["to"])
        elif s["kind"] == "entrance":
            f = D["facilities"][s["facility"]]
            ap0 = f["accessPointIds"][0]
            n0 = D["accessPoints"][ap0]["nodeId"]
            (rid0, r0), = [(k, r) for k, r in D["roads"].items() if n0 in (r["fromNodeId"], r["toNodeId"])]
            js = [jid for jid, j in D["junctions"].items() if n0 in j["nodeIds"]]
            for k in range(1, s.get("n", 1) + 1):
                tag = f"~{k}"
                n1, rid1, ap1 = n0 + tag, rid0 + tag, ap0 + tag
                D["nodes"][n1] = copy.deepcopy(D["nodes"][n0])
                D["roads"][rid1] = dict(copy.deepcopy(r0), **{e: n1 for e in ("fromNodeId", "toNodeId") if r0[e] == n0})
                D["accessPoints"][ap1] = dict(copy.deepcopy(D["accessPoints"][ap0]), nodeId=n1)
                f["accessPointIds"].append(ap1)
                for jid in js:
                    D["junctions"][jid + tag] = dict(copy.deepcopy(D["junctions"][jid]), nodeIds=[n1])
                for mid, m in list(D["movements"].items()):
                    if rid0 in (m["incomingArc"]["roadId"], m["outgoingArc"]["roadId"]):
                        m1 = copy.deepcopy(m)
                        for a in (m1["incomingArc"], m1["outgoingArc"]):
                            if a["roadId"] == rid0:
                                a["roadId"] = rid1
                        if m["junctionId"] in js:
                            m1["junctionId"] += tag
                        D["movements"][mid + tag] = m1
    return D


class Net:
    def __init__(self, G, p, roles=None):
        """G: derived graph (read_graph, or occupancy.mapgraph.to_graph for a map with geometry); a path is read with
        read_graph. roles = {role in FLOW: class in the graph}: entities of that class take part in the flows in that role
        (for instance a steel processing shop as the assembly workshop)."""
        if not isinstance(G, dict):
            G = read_graph(G)
        rule = G["frontage_rule"]
        if any(p[k] != rule[k.split("_", 1)[1]] for k in FRONTAGE_KEYS):
            raise ValueError(f"the dock frontage of this graph was derived with {rule}; other frontage parameters need the map")
        self.retro = p.get("retrofit") or []
        docked = {r for z in G["zones"].values() if z["classId"] == DOCK_CLASS for r, *_ in z["frontage"]}
        for s in self.retro:
            if s["kind"] == "widen" and docked & set(s["roads"]):
                raise NotImplementedError("widening a road along a dock moves the dock stop, which needs the dock outline; "
                                          "the derived graph holds the frontage at the mapped widths")
        D = retrofit_map(G, self.retro)
        self.D = D
        self.role_of = {c: r for r, c in (roles or {}).items()}
        self.road = {}
        for rid, r in D["roads"].items():
            L = r["lengthM"]
            w = r["widthM"]
            # orig, s0, s1: the graph road this one belongs to and its chainage on it (without sectioning: itself and [0, L])
            self.road[rid] = dict(a=r["fromNodeId"], b=r["toNodeId"], L=L, name=r["name"],
                                  W=float(w["value"]) if w["state"] == "known" else None, orig=rid, s0=0.0, s1=L)
        deg = collections.Counter(x for r in self.road.values() for x in (r["a"], r["b"]))
        self.access = {a["nodeId"] for a in D["accessPoints"].values()}
        self.node_res = {n for n in D["nodes"] if deg[n] >= 3 or n in self.access}
        self.incident = collections.defaultdict(list)
        for rid, r in self.road.items():
            self.incident[r["a"]].append(rid)
            self.incident[r["b"]].append(rid)
        self._stops(D, p)
        self._retrofit()
        self._section(p)
        self._graph(D)
        self._trees = {}

    def _retrofit(self):
        """Road changes of the stops (dock_bay, dock_wait, door; see the module docstring). bay_cap = {off-road bay / in-door
        bay / waiting place: number of places}; dock_bays = dock-side off-road bays; wait_at = {dock stop node: waiting place};
        doors = door nodes."""
        self.bay_cap, self.dock_bays, self.doors, self.wait_at = {}, set(), set(), {}
        for s in self.retro:
            if s["kind"] == "dock_wait":
                for z in s.get("docks") or self.by_cls[DOCK_CLASS]:
                    for k, sid in enumerate(self.ents[z]["stops"]):
                        if self.stops[sid]["mode"] == "on_road":
                            self.wait_at[self.stops[sid]["node"]] = ("w", z, k)
                            self.bay_cap[("w", z, k)] = float(s["cap"])
            elif s["kind"] == "dock_bay":
                for z in s.get("docks") or self.by_cls[DOCK_CLASS]:
                    for k, sid in enumerate(self.ents[z]["stops"]):
                        st = self.stops[sid]
                        if st["mode"] == "on_road":
                            st.update(mode="pull_in", res=("b", z, k), bay=True)
                            self.bay_cap[st["res"]] = float(s["cap"])
                            self.dock_bays.add(st["res"])
            elif s["kind"] == "door":
                for f in s["facilities"]:
                    res = ("b", f, "in")
                    for sid in self.ents[f]["stops"]:
                        assert self.stops[sid]["mode"] == "node", sid
                        self.stops[sid].update(mode="door", res=res)
                        self.doors.add(self.stops[sid]["node"])
                    self.bay_cap[res] = float(len(self.ents[f]["stops"]))

    # ---------------------------------------------------------------- entities of origins and destinations, and their stops
    def _stops(self, D, p):
        """stops[id] = dict(node, res, mode, ent); ents[entity] = dict(cls, stops)."""
        classes = {c for f in FLOW for x in f[1:3] for c in x.split("|")}
        self.ents, self.stops = {}, {}
        cuts = collections.defaultdict(set)
        for fid, f in D["facilities"].items():
            mc = f["classId"]
            c = self.role_of.get(mc, mc)
            nodes = [D["accessPoints"][a]["nodeId"] for a in f.get("accessPointIds", []) if a in D["accessPoints"]]
            if c in classes and nodes:
                ids = []
                for n in nodes:
                    self.stops[n] = dict(node=n, res=("n", n), mode="node", ent=fid)
                    ids.append(n)
                self.ents[fid] = dict(cls=c, map_cls=mc, stops=ids, name=f["name"])
        for zid, z in D["zones"].items():
            mc = z["classId"]
            c = self.role_of.get(mc, mc)
            if c not in classes:
                continue
            fr = self._dock_frontage(z, p) if c == "dry_dock" else list(z["frontage"])
            mode = STOP_MODE[c]
            ids = []
            for k, (rid, s0, s1) in enumerate(fr):
                L = self.road[rid]["L"]
                s = round(min(max((s0 + s1) / 2, min(1.0, L / 2)), max(L - 1.0, L / 2)), 1)
                node = f"S|{rid}|{s}"
                cuts[rid].add(s)
                sid = f"{zid}#{k}"
                self.stops[sid] = dict(node=node, res=("r", rid) if mode == "on_road" else ("b", zid, k), mode=mode, ent=zid)
                ids.append(sid)
            if ids:
                self.ents[zid] = dict(cls=c, map_cls=mc, stops=ids, name=z["name"])
        self.cuts = {rid: sorted(v) for rid, v in cuts.items()}
        self.by_cls = collections.defaultdict(list)
        for e, v in sorted(self.ents.items()):
            self.by_cls[v["cls"]].append(e)

    def _dock_frontage(self, z, p):
        """Frontage of a dock: the stretches of road whose centre line lies within half the road width + frontage_gap of the
        dock outline (gaps <= frontage_merge closed, stretches shorter than frontage_min_len dropped; access roads of entrances
        excluded), derived from the map (occupancy.mapgraph.dock_frontage) and stored in the graph. One roadside stop at the
        middle of each stretch; p["dock_stops"] = n keeps only the n longest (ties by road id and chainage), in their order."""
        out = list(z["frontage"])
        n = p.get("dock_stops")
        if n is not None:
            keep = set(sorted(out, key=lambda x: (-(x[2] - x[1]), x[0], x[1]))[:n])
            out = [x for x in out if x in keep]
        return out

    def near(self, cls):
        """Resources around the zones of class cls, [ring 0, ring 1] (disjoint). Ring 0 = the stop resources of its entities
        (for roadside stops the whole road, or with sectioning the section of the stop); ring 1 = the junctions at both ends of
        the graph road of each stop (or the entrance node), the roads at these ends, and the graph road of the stop itself
        (its sections when sectioned; if not in ring 0)."""
        ring0 = {self.stops[s]["res"] for e in self.by_cls[cls] for s in self.ents[e]["stops"]}
        roads, ends = set(), set()
        for e in self.by_cls[cls]:
            for s in self.ents[e]["stops"]:
                n = self.stops[s]["node"]
                if n.startswith("S|"):
                    secs = self.sec.get(n.split("|")[1], [n.split("|")[1]])
                    roads |= set(secs)
                    ends |= {self.road[secs[0]]["a"], self.road[secs[-1]]["b"]}
                else:
                    ends.add(n)
        ring1 = {("n", x) for x in ends if x in self.node_res} | {("r", r) for x in ends for r in self.incident[x]} \
            | {("r", r) for r in roads}
        return [ring0, ring1 - ring0]

    # ---------------------------------------------------------------- sectioning
    def _section(self, p, min_gap=1.0):
        """p["section"] = "stops": a road longer than section_min_m with roadside stops (on_road) gets section boundaries
        section_gap_m before and after each stop on it (roadside stops and drive-in points) and is cut into sections, one
        resource each (same width); with block_m each section is further cut into blocks of length <= block_m.
        No boundary within min_gap of a stop, a road end or another boundary (no slivers, no stop on a boundary).
        Sections are named <road>@<k> (names get @k); boundaries are new cut points S|<road>|<chainage> (named by the chainage
        on the graph road, like stops) that can only be passed straight on; a roadside stop's resource becomes its section.
        sec = {graph road: [sections]}, edges = {graph road: [boundary chainages]}; both empty without sectioning.
        Stops of dock-side bays (dock_bay) still set boundaries; the ends of a passing place (passing) are forced boundaries
        (also without sectioning), with the width of the passing place."""
        self.sec, self.edges = {}, {}
        force, wide = self._passing(p, min_gap)
        if not p.get("section") and not force:
            return
        if p.get("section") not in (None, "stops"):
            raise ValueError(f"unknown sectioning: {p['section']}")
        on_road = {st["res"][1] for st in self.stops.values() if st["mode"] == "on_road"} | \
            {st["node"].split("|")[1] for st in self.stops.values() if st.get("bay")} if p.get("section") else set()
        gap, B = p["section_gap_m"], p.get("block_m")
        for rid in sorted(on_road | set(force)):
            r = self.road[rid]
            L, cuts = r["L"], self.cuts.get(rid, [])
            edges = list(force.get(rid, []))

            def add(e):
                if min_gap < e < L - min_gap and all(abs(e - x) > min_gap for x in cuts + edges):
                    edges.append(e)
            if rid in on_road and L > p["section_min_m"]:
                for s in cuts:
                    add(round(s - gap, 1))
                    add(round(s + gap, 1))
                if B:
                    pts = [0.0] + sorted(edges) + [L]
                    for a, b in zip(pts, pts[1:]):
                        n = math.ceil((b - a) / B - EPS)
                        for j in range(1, n):
                            add(round(a + j * (b - a) / n, 1))
            if not edges:
                continue
            pts = [0.0] + sorted(edges) + [L]
            node_at = lambda x: r["a"] if x == 0.0 else (r["b"] if x == L else f"S|{rid}|{x}")
            secs = []
            for k, (a, b) in enumerate(zip(pts, pts[1:])):
                sid = f"{rid}@{k}"
                self.road[sid] = dict(r, a=node_at(a), b=node_at(b), L=b - a, s0=a, s1=b, name=f"{r['name']}@{k}",
                                      **{"W": w for wa, wb, w in wide.get(rid, ()) if wa - EPS <= a and b <= wb + EPS})
                secs.append(sid)
            del self.road[rid]
            self.sec[rid], self.edges[rid] = secs, pts[1:-1]
            for end, s in ((r["a"], secs[0]), (r["b"], secs[-1])):
                self.incident[end] = [s if x == rid else x for x in self.incident[end]]
            for st in self.stops.values():
                if st["res"] == ("r", rid):
                    at = float(st["node"].split("|")[2])
                    st["res"] = ("r", next(s for s in secs if self.road[s]["s0"] < at < self.road[s]["s1"]))

    def _passing(self, p, min_gap):
        """Passing places and blocks (road changes): {road: [chainages of the forced boundaries]} and {road: [(start, end,
        width)]}. A passing place sits at the middle of the longest stretch between the stops and the ends of the road (the
        middle of a long single-lane road), length len (default 40 m: the longest loaded vehicle, 30 m, plus the longitudinal
        clearance of 5 m, with room to spare), width w (default the equivalent width without width limit, where two loaded
        vehicles pass)."""
        force, wide = {}, {}
        for s in self.retro:
            if s["kind"] not in ("passing", "blocks"):
                continue
            rid = s["road"]
            L = self.road[rid]["L"]
            cuts = self.cuts.get(rid, [])
            if s["kind"] == "blocks":     # the whole road in blocks of length <= len; no boundary within min_gap of a stop
                n = math.ceil(L / s["len"] - EPS)
                e = [round(j * L / n, 1) for j in range(1, n)]
                force.setdefault(rid, []).extend(x for x in e if all(abs(x - c) > min_gap for c in cuts))
                continue
            ln = s.get("len", 40.0)
            if "at" in s:
                a = b = s["at"]
            else:
                pts = [0.0] + cuts + [L]
                a, b = max(zip(pts, pts[1:]), key=lambda x: x[1] - x[0])
            e = [round((a + b - ln) / 2, 1), round((a + b + ln) / 2, 1)]
            assert min_gap < e[0] and e[1] < L - min_gap and all(not (e[0] - min_gap <= x <= e[1] + min_gap) for x in cuts), (rid, e)
            force.setdefault(rid, []).extend(e)
            wide.setdefault(rid, []).append((e[0], e[1], s.get("w", p["unrestricted_w"])))
        return force, wide

    def runs(self, src, dst):
        """The parts of a trip merged by (road, direction), one per decision point of leg: part k is entered at decision point
        k (used by the direction lock)."""
        out = []
        for a in (self.path(src, dst) if src != dst else []):
            x = (self.arc[a][3], self.arc[a][4])
            if not out or out[-1] != x:
                out.append(x)
        return out

    # ---------------------------------------------------------------- directed arcs (stops and section boundaries cut roads into arcs)
    def _graph(self, D):
        self.arc = []        # (tail, head, length, road, forward)
        for rid, r in sorted(self.road.items()):
            o = r["orig"]
            if o == rid:                  # road not sectioned
                marks, pts = self.cuts.get(rid, []), [0.0] + self.cuts.get(rid, []) + [r["L"]]
            else:                         # section: the stops inside it
                marks = [c for c in self.cuts.get(o, []) if r["s0"] < c < r["s1"]]
                pts = [r["s0"]] + marks + [r["s1"]]
            nodes = [r["a"]] + [f"S|{o}|{s}" for s in marks] + [r["b"]]
            for i in range(len(pts) - 1):
                self.arc.append((nodes[i], nodes[i + 1], pts[i + 1] - pts[i], rid, True))
                self.arc.append((nodes[i + 1], nodes[i], pts[i + 1] - pts[i], rid, False))
        self.out = collections.defaultdict(list)
        for i, a in enumerate(self.arc):
            self.out[a[0]].append(i)
        allowed = collections.defaultdict(set)
        self.turn_at = {}                 # (node, graph road in, forward?, graph road out, forward?) -> turn angle (deg)
        jn = {jid: j["nodeIds"][0] for jid, j in D["junctions"].items()}
        for m in D["movements"].values():
            ia, oa = m["incomingArc"], m["outgoingArc"]
            key = ((ia["roadId"], ia["direction"] == "forward"), (oa["roadId"], oa["direction"] == "forward"))
            self.turn_at[(jn[m["junctionId"]],) + key[0] + key[1]] = m["turnDeg"]
            if m["allowed"]:
                allowed[jn[m["junctionId"]]].add(key)
        orig = {rid: r["orig"] for rid, r in self.road.items()}       # turn permissions refer to graph roads
        self.succ = []
        for t, h, _, rid, fw in self.arc:
            if h.startswith("S|"):        # stops and section boundaries: only straight on along the same graph road
                self.succ.append([j for j in self.out[h] if orig[self.arc[j][3]] == orig[rid] and self.arc[j][4] == fw])
            else:
                self.succ.append([j for j in self.out[h] if ((orig[rid], fw), (orig[self.arc[j][3]], self.arc[j][4])) in allowed[h]])

    def tree(self, src):
        """Shortest paths (by distance) from src that respect the turn permissions: best[node] = (distance, arriving arc),
        pred[arc] = previous arc."""
        if src in self._trees:
            return self._trees[src]
        dist, pred = {}, {}
        heap = [(self.arc[a][2], a, -1) for a in self.out[src]]
        heapq.heapify(heap)
        while heap:
            d, a, p = heapq.heappop(heap)
            if a in dist:
                continue
            dist[a], pred[a] = d, p
            for b in self.succ[a]:
                if b not in dist:
                    heapq.heappush(heap, (d + self.arc[b][2], b, a))
        best = {src: (0.0, -1)}
        for a, d in dist.items():
            h = self.arc[a][1]
            if d < best.get(h, (math.inf,))[0]:
                best[h] = (d, a)
        self._trees[src] = (best, pred)
        return best, pred

    def dist(self, src, dst):
        return self.tree(src)[0].get(dst, (math.inf,))[0]

    def path(self, src, dst):
        best, pred = self.tree(src)
        if dst not in best:
            raise RuntimeError(f"unreachable: {src} -> {dst}")
        out, a = [], best[dst][1]
        while a != -1:
            out.append(a)
            a = pred[a]
        return out[::-1]

    # ---------------------------------------------------------------- capacities and occupancy
    def capacities(self, p):
        """cap[resource] and size[node resource] (junction size D)."""
        c = p["lat_clear"]
        W = {rid: (r["W"] + p["width_delta"]) if r["W"] is not None else p["unrestricted_w"] for rid, r in self.road.items()}
        cap = {("r", rid): w + c for rid, w in W.items()}
        size = {}
        for n in self.node_res:
            known = [W[rid] for rid in self.incident[n] if self.road[rid]["W"] is not None]
            size[n] = max(known) if known else p["unrestricted_w"]
            cap[("n", n)] = size[n] + c
        for s in self.stops.values():
            if s["res"][0] == "b":
                cap[s["res"]] = self.bay_cap.get(s["res"], 1.0)
        for w in self.wait_at.values():
            cap[w] = self.bay_cap[w]
        if p.get("crane_n"):          # crane-paced erection: the cranes of each dock are a resource ('c', dock), capacity = number
            for z, n in self.crane_counts(p).items():
                cap[("c", z)] = float(n)
        return cap, size

    def crane_counts(self, p):
        """Cranes per dock: p["crane_n"] = "map" counts the cranes of the dock in the graph that count for erection; an integer
        (the same for every dock) or {dock: number} gives them directly."""
        n = p["crane_n"]
        if n == "map":
            return {z: sum(c["count"] for c in self.D["zones"][z]["cranes"] if c["countsForErection"]) for z in self.by_cls[DOCK_CLASS]}
        return {z: (n[z] if isinstance(n, dict) else n) for z in self.by_cls[DOCK_CLASS]}

    def spans(self):
        """Gantry crane spans: {span zone: (dock, {assembly area: share of its area inside the span})}."""
        return {zid: (z["span"]["dock"], z["span"]["covers"]) for zid, z in self.D["zones"].items() if z["span"]}

    def lift_pairs(self, p):
        """P6 (assembly area, dock) pairs inside a crane span: the span of the dock covers at least p["span_min"] of the area,
        so the gantry crane lifts the block directly and no road is used. span_min None (or a scenario without crane-paced
        erection): no lifts."""
        m = p.get("span_min") if p.get("crane_n") else None
        if m is None:
            return set()
        return {(e, dock) for dock, cov in self.spans().values() for e, f in cov.items() if f >= m}

    @staticmethod
    def demand(res, b, loaded, p, cap):
        """The "width" a vehicle takes on a resource. A loaded vehicle holds a junction alone; a bay takes one place per vehicle
        (a drive-in bay has one place, so it is held alone); an empty vehicle follows empty_rule."""
        C = cap[res]
        if res[0] == "b":
            return 1.0
        if not loaded:
            if p["empty_rule"] == "free":
                return 0.0
            if p["empty_rule"] == "exclusive":
                return C
        elif res[0] == "n":
            return C
        return min(C, b + p["lat_clear"])

    def turn_deg(self, a, b):
        """Turn angle (degrees) entering by arc a and leaving by arc b at the node between them (stored per movement)."""
        _, h, _, ra, fa = self.arc[a]
        _, _, _, rb, fb = self.arc[b]
        return self.turn_at[(h, self.road[ra]["orig"], fa, self.road[rb]["orig"], fb)]

    def leg(self, src, dst, v, Lv, b, loaded, p, cap, size, end_res, end_d, start_node=False):
        """Resources occupied by one trip (src -> dst, without waiting).
        Returns (T, st, uses, arr): T travel time; st[k] time of decision point k (from departure); arr index of the arrival;
        uses = [(resource, width taken, acquired at decision point, released at decision point or None, release delay)], None
        meaning released by the task logic (the destination stop).
        Decision points: 0 = departure, k = entering part k, arr = arrival. A road is held until the vehicle's tail has left
        the next junction plus the longitudinal clearance; a junction for (D + vehicle length + longitudinal clearance)/v plus
        the turn sweep; the destination stop is requested together with the last road (no access road is held without it).
        Doors (door): requested at decision point 0 when leaving a door and at arr when arriving at one, released after
        (D + vehicle length + longitudinal clearance)/v.
        Dock-side off-road bays (dock_bay): the last road is released at entry + its travel time + the tail clearance, so a
        vehicle waiting for the crane in the bay no longer holds it.
        Dock-side waiting places (dock_wait): the last road is released as usual at arrival + tail clearance, and the roadside
        stop is acquired at decision point arr (start of unloading); without waiting their union is unchanged, and reservation
        rules can insert a wait in the waiting place between the two (Sim._book_dest)."""
        tail = (Lv + p["long_clear"]) / v
        if src == dst:
            return 0.0, [0.0], [(end_res, end_d, 0, None, 0.0)], 0
        door = lambda n: (("n", n), self.demand(("n", n), b, loaded, p, cap), (size[n] + Lv + p["long_clear"]) / v)
        arcs = self.path(src, dst)
        runs = []                         # [road, chainage at entry, first arc, last arc]
        D = 0.0
        for a in arcs:
            t, h, L, rid, fw = self.arc[a]
            if runs and runs[-1][0] == rid and self.arc[runs[-1][3]][4] == fw:
                runs[-1][3] = a
            else:
                runs.append([rid, D, a, a])
            D += L
        m = len(runs) - 1
        st = [r[1] / v for r in runs] + [D / v]
        uses = []
        if src in self.doors:
            r_, d_, dl = door(src)
            uses.append((r_, d_, 0, 0, dl))
        elif start_node and ("n", src) in cap:
            uses.append((("n", src), self.demand(("n", src), b, loaded, p, cap), 0, 0, tail))
        for i, (rid, _, a0, a1) in enumerate(runs):
            res = ("r", rid)
            if i > 0:
                n = self.arc[a0][0]
                if n in self.node_res:
                    sweep = p["turn_sweep_s"] if self.turn_deg(runs[i - 1][3], a0) >= p["turn_deg"] else 0.0
                    uses.append((("n", n), self.demand(("n", n), b, loaded, p, cap), i, i, (size[n] + Lv + p["long_clear"]) / v + sweep))
            if i == m and res == end_res and dst in self.wait_at:     # roadside stop with a waiting place: the stop is acquired when unloading starts
                uses.append((res, self.demand(res, b, loaded, p, cap), i, i + 1, tail))
                uses.append((end_res, max(end_d, self.demand(res, b, loaded, p, cap)), m + 1, None, 0.0))
            elif i == m and res == end_res:          # roadside stop: this road is the stop resource
                uses.append((res, max(end_d, self.demand(res, b, loaded, p, cap)), i, None, 0.0))
            elif i == m and end_res in self.dock_bays:     # into a dock-side bay: released on entry, not held while waiting for the crane
                uses.append((res, self.demand(res, b, loaded, p, cap), i, i, st[i + 1] - st[i] + tail))
            else:
                uses.append((res, self.demand(res, b, loaded, p, cap), i, i + 1, tail))
        if end_res != ("r", runs[-1][0]):
            uses.append((end_res, end_d, m, None, 0.0))
        if dst in self.doors:
            r_, d_, dl = door(dst)
            uses.append((r_, d_, m + 1, m + 1, dl))
        return D / v, st, uses, m + 1

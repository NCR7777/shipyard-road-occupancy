"""Derived graph from a map with geometry.

The yards of the paper are released only as derived graphs (data/networks); their maps, digitised from satellite imagery,
are not. This module is how those graphs were derived, and it builds the graphs of the synthetic networks in experiments/ and
tests/ from small maps written in code. A map is a dict in the layout of the map editor used for the yards: nodes with
positions, roads between two nodes with a centre line (anchors and line or cubic spans) and a width, junctions with their
movements, entrances (accessPoints) of buildings (facilities), and zones with an outline; classes and the research attributes
(frontage stretches, cranes, crane spans) sit in two extension namespaces.

to_graph(D) keeps the topology and replaces the geometry by what the model uses:
  road length          length of the centre line, cubic spans flattened at about 0.5 m;
  turn angle           of every movement, from the headings of the two roads at the junction (3 m from the end);
  dock frontage        stretches of road whose centre line lies within half the road width (12 m if unknown) + gap of the
                       dock outline, sampled every metre, gaps <= merge closed, stretches shorter than min_len dropped, access
                       roads of entrances excluded;
  other frontage       the frontage stretches recorded in the map;
  cranes, crane spans  the cranes of each dock and the share of each area inside the span of the dock's gantry crane.
"""
import math

SC = "org.shipyard.spatial_classification"
RS = "org.shipyard.research_semantics"
FRONTAGE = dict(gap=8.0, min_len=10.0, merge=3.0)


def polyline(D, r):
    """Centre line of a road; cubic spans flattened at about 0.5 m."""
    P = lambda n: tuple(D["nodes"][n]["position"][:2])
    pts = [P(r["fromNodeId"])] + [tuple(a[:2]) for a in r["geometry"]["anchors"]] + [P(r["toNodeId"])]
    out = [pts[0]]
    for i, sp in enumerate(r["geometry"]["spans"]):
        a, b = pts[i], pts[i + 1]
        if sp["kind"] == "line":
            out.append(b)
            continue
        c1, c2 = sp["control1"][:2], sp["control2"][:2]
        n = max(8, int(math.dist(a, b) / 0.5) + 1)
        for k in range(1, n + 1):
            t = k / n
            u = 1 - t
            out.append(tuple(u ** 3 * a[j] + 3 * u * u * t * c1[j] + 3 * u * t * t * c2[j] + t ** 3 * b[j] for j in (0, 1)))
    return out


def samples(poly, step):
    """A point every `step` metres along a polyline [(s, x, y)]; the end point is always included."""
    out, s = [(0.0, *poly[0])], 0.0
    nxt = step
    for (x1, y1), (x2, y2) in zip(poly, poly[1:]):
        L = math.hypot(x2 - x1, y2 - y1)
        while L > 0 and nxt <= s + L:
            f = (nxt - s) / L
            out.append((nxt, x1 + f * (x2 - x1), y1 + f * (y2 - y1)))
            nxt += step
        s += L
    if out[-1][0] < s - 1e-6:
        out.append((s, *poly[-1]))
    return out


def heading(poly, at_end, look=3.0):
    """Heading (rad) of a road at one end, in the from -> to direction."""
    pts = poly[::-1] if at_end else poly
    p0 = pts[0]
    for q in pts[1:]:
        if math.dist(p0, q) >= look:
            break
    a = math.atan2(q[1] - p0[1], q[0] - p0[0])
    return a + math.pi if at_end else a


def poly_dist(p, poly):
    """Distance from a point to a polygon, 0 inside."""
    x, y = p
    inside, best = False, float("inf")
    for (x1, y1), (x2, y2) in zip(poly, poly[1:]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
        dx, dy = x2 - x1, y2 - y1
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / L2))
        best = min(best, math.hypot(x - x1 - t * dx, y - y1 - t * dy))
    return 0.0 if inside else best


def _area(pts):
    """Polygon area (shoelace)."""
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]))) / 2 if pts else 0.0


def cover(poly, clip):
    """Share of the area of polygon poly inside the convex polygon clip (edge by edge clipping); checks coversZones."""
    clip = clip[:-1] if clip[0] == clip[-1] else clip
    sgn = 1 if sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(clip, clip[1:] + clip[:1])) > 0 else -1
    side = lambda p, a, b: sgn * ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) >= 0
    pts = poly
    for a, b in zip(clip, clip[1:] + clip[:1]):
        out = []
        for p, q in zip(pts, pts[1:] + pts[:1]):
            if side(p, a, b) != side(q, a, b):
                d = (b[0] - a[0]) * (q[1] - p[1]) - (b[1] - a[1]) * (q[0] - p[0])
                t = ((b[0] - a[0]) * (a[1] - p[1]) - (b[1] - a[1]) * (a[0] - p[0])) / d
                out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
            if side(q, a, b):
                out.append(q)
        pts = out
        if not pts:
            break
    return _area(pts) / _area(poly)


def dock_frontage(D, z, roads, access, rule=FRONTAGE):
    """Frontage stretches [(road, s0, s1)] of a dock z: see the module docstring. roads = {road: (polyline, width or None)},
    access = entrance nodes; roads in id order, stretches in chainage order."""
    poly = [tuple(q[:2]) for q in z["boundary"]["outer"]]
    out = []
    for rid, (pl, W) in sorted(roads.items()):
        r = D["roads"][rid]
        if r["fromNodeId"] in access or r["toNodeId"] in access:
            continue
        lim = (W if W is not None else 12.0) / 2 + rule["gap"]
        hits = [s for s, x, y in samples(pl, 1.0) if poly_dist((x, y), poly) <= lim]
        runs = []
        for s in hits:
            if runs and s - runs[-1][1] <= rule["merge"]:
                runs[-1][1] = s
            else:
                runs.append([s, s])
        out += [(rid, s0, s1) for s0, s1 in runs if s1 - s0 >= rule["min_len"]]
    return out


def to_graph(D, yard="", name=lambda s: s, rule=FRONTAGE):
    """Derived graph (the layout Net and network.write_graph take) of the map D. name: applied to every name (the yards'
    names are translated from the map's language)."""
    cls = lambda o: ((o.get("extensions") or {}).get(SC) or {}).get("classId")
    rs = lambda o: (o.get("extensions") or {}).get(RS) or {}
    nm = lambda o: name(o.get("name") or "")
    roads, head = {}, {}
    for rid, r in D["roads"].items():
        pl = polyline(D, r)
        w = r["widthM"]
        roads[rid] = (pl, float(w["value"]) if w["state"] == "known" else None)
        head[rid] = (heading(pl, False), heading(pl, True))
    access = {a["nodeId"] for a in D["accessPoints"].values()}
    G = dict(yard=yard, revision=D.get("revision"), frontage_rule=dict(rule))
    G["nodes"] = {k: dict(name=nm(n), kind=n.get("kind", "")) for k, n in D["nodes"].items()}
    G["roads"] = {k: dict(fromNodeId=r["fromNodeId"], toNodeId=r["toNodeId"], name=nm(r),
                          lengthM=sum(math.dist(a, b) for a, b in zip(roads[k][0], roads[k][0][1:])),
                          widthM=dict(state="known", value=r["widthM"]["value"]) if r["widthM"]["state"] == "known"
                          else dict(state=r["widthM"]["state"]))
                  for k, r in D["roads"].items()}
    G["junctions"] = {k: dict(nodeIds=list(j["nodeIds"])) for k, j in D["junctions"].items()}

    def turn(ia, oa):
        ra, fa = ia["roadId"], ia["direction"] == "forward"
        rb, fb = oa["roadId"], oa["direction"] == "forward"
        hin = head[ra][1] if fa else head[ra][0] + math.pi
        hout = head[rb][0] if fb else head[rb][1] + math.pi
        d = (hout - hin) % (2 * math.pi)
        return math.degrees(min(d, 2 * math.pi - d))
    G["movements"] = {k: dict(junctionId=m["junctionId"], allowed=bool(m["allowed"]), turnDeg=turn(m["incomingArc"], m["outgoingArc"]),
                              incomingArc=dict(roadId=m["incomingArc"]["roadId"], direction=m["incomingArc"]["direction"]),
                              outgoingArc=dict(roadId=m["outgoingArc"]["roadId"], direction=m["outgoingArc"]["direction"]))
                      for k, m in D["movements"].items()}
    G["accessPoints"] = {k: dict(nodeId=a["nodeId"], facilityId=a.get("facilityId") or "", name=nm(a))
                         for k, a in D["accessPoints"].items()}
    G["facilities"] = {k: dict(name=nm(f), classId=cls(f), accessPointIds=list(f.get("accessPointIds", [])))
                       for k, f in D["facilities"].items()}
    G["zones"] = {}
    for k, z in D["zones"].items():
        r = rs(z)
        if cls(z) == "dry_dock":
            fr = dock_frontage(D, z, roads, access, rule)
        else:
            fr = [(x["roadId"], x["sStartM"], x["sEndM"]) for x in r.get("frontage") or []]
        G["zones"][k] = dict(name=nm(z), classId=cls(z), frontage=fr,
                             cranes=[dict(type=c.get("type", ""), count=c["count"], countsForErection=bool(c["countsForErection"]))
                                     for c in r.get("cranes", [])],
                             span=dict(dock=r["dockZoneId"], covers=dict(r["coversZones"])) if r.get("role") == "gantry_crane_span" else None)
    return G

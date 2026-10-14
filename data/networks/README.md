# Road networks of the two yards (derived graphs)

The road networks of the two shipyards are released as derived graphs: the topology and the quantities the model uses, with
no coordinates. They were derived from vector maps digitised from Google Earth imagery (Google Earth Pro; Okpo dated
15 November 2023: Google, Airbus, TMap Mobility; Yantai dated 11 March 2025: Google, Airbus). Neither the imagery nor the
centre-line geometry traced from it is released; `occupancy/mapgraph.py` documents how the derived quantities were computed
from the geometry.

| Directory | Yard | Revision | Used for |
|---|---|---|---|
| `okpo/rev1833` | Okpo (Hanwha Ocean, Geoje, Korea; yard key `yupu`) | 1833 | the paper's Okpo network, with three floating docks and their spur roads |
| `okpo/rev1825` | Okpo (yard key `yupu_r1825`) | 1825 | runs stored from the earlier revision: three spur roads fewer and the end points of three roads moved; same stops, junctions and cranes (see `results/network/okpo_revisions.json`); its road network equals that of revision 1824, which some runs name |
| `yantai/rev396` | Yantai (CIMC Raffles; yard key `yantai`) | 396 | the paper's Yantai network; revision 395, named by some runs, has the same road network without the crane data |

Read a graph with `occupancy.network.read_graph("okpo/rev1833")`, or pass the path to `occupancy.network.Net` directly:

```python
from occupancy import params as P
from occupancy.network import Net
net = Net("okpo/rev1833", P.scenario("main"))
```

Names (of roads, nodes, buildings, zones) are the map's names translated from Chinese; identifiers (`road_…`, `node_…`,
`facility_…`, `zone_…`) are those of the maps and appear in the result files (`results/`). All files are UTF-8 CSV with a
header row; floating-point values are written in full precision (Python `repr`), so they read back exactly.

## Files

**`meta.json`** — `yard`, `revision`, and `frontage_rule` (`gap`, `min_len`, `merge`, in m): the rule with which the dock
frontage below was derived (equal to the model parameters `frontage_gap`, `frontage_min_len`, `frontage_merge`).

**`nodes.csv`** — one row per node.

| Field | Meaning |
|---|---|
| `node_id` | identifier |
| `name` | name on the map (e.g. `entrance001 node`, `road023 / split point`, `road end`) |
| `kind` | `access` (an entrance node of a building) or `ordinary` |

**`roads.csv`** — one row per road (an undirected edge between two nodes; the model forms two arcs from it, one per
direction, and splits it further at stops and section boundaries).

| Field | Meaning |
|---|---|
| `road_id` | identifier |
| `name` | name on the map (e.g. `road161`, `road010-6`) |
| `from_node`, `to_node` | end nodes; "forward" runs from `from_node` to `to_node` |
| `length_m` | length of the centre line (m) |
| `width_m` | road width (m) when `width_state` is `known` |
| `width_state` | `known`, or `unrestricted` for access roads without a width limit (the model gives these the equivalent width `unrestricted_w`, two loaded vehicles side by side) |

A road is a resource with capacity `width_m + c` (lateral clearance `c`). Junctions of degree three or more and all entrance
nodes are resources too; their size `D` is the largest width of the adjacent roads with a known width (computed by the model).

**`junctions.csv`** — `junction_id`, `node_id`: the node at which the movements of a junction take place.

**`movements.csv`** — turn permissions: one row per possible movement through a junction.

| Field | Meaning |
|---|---|
| `movement_id` | identifier |
| `junction_id` | junction |
| `in_road`, `in_dir` | road and direction (`forward`, `backward`) of the vehicle entering the junction |
| `out_road`, `out_dir` | road and direction of the vehicle leaving it |
| `allowed` | `true` if the movement is permitted (shortest paths use only permitted movements) |
| `turn_deg` | turn angle of the movement (degrees, 0 = straight on, 180 = U-turn), from the headings of the two roads 3 m from the junction; a turn of at least `turn_deg` (60°) adds the turn sweep `turn_sweep_s` to the junction's occupancy |

**`access_points.csv`** — entrances of buildings: `access_id`, `name` (e.g. `building014-entrance1`), `node_id` (the
entrance node), `facility_id` (the building).

**`facilities.csv`** — buildings: `facility_id`, `name` (e.g. `building014·paint shop (inferred)`), `class` (functional
class, e.g. `paint_workshop`, `assembly_workshop`, `outfitting_workshop`; `(inferred)` in a name marks a class read from the
imagery without a public source), `access_points` (the building's entrances, in order, separated by spaces). The entrances of
a building are its stops.

**`zones.csv`** — open areas: `zone_id`, `name`, `class` (`yard` stockyard, `buffer`, `assembly_yard` assembly area,
`dry_dock`, `floating_dock`, `quay`, …; empty for the gantry-crane span zones).

**`frontage.csv`** — stops of the zones: the stretches of road along a zone where vehicles stop.

| Field | Meaning |
|---|---|
| `zone_id` | zone |
| `k` | number of the stretch within the zone (its stop is `<zone_id>#<k>`) |
| `road_id` | road |
| `s_start_m`, `s_end_m` | chainage of the stretch on the road (m from `from_node`) |

One stop sits at the middle of each stretch: a roadside stop on the road for buffers and docks, a drive-in bay for
stockyards and assembly areas. The stretches of the docks were derived from the dock outlines with the rule of `meta.json`
(the stretches of road whose centre line lies within half the road width + `gap` of the dock, gaps of at most `merge` closed,
stretches shorter than `min_len` dropped); the model parameter `dock_stops` keeps the longest ones.

**`cranes.csv`** — erection cranes of the docks: `dock_zone_id`, `type` (gantry, jib, tower, bridge), `count`,
`counts_for_erection` (`true` if the crane lifts blocks into the dock; the model's crane resource of a dock has as many units
as these cranes).

**`crane_spans.csv`** — gantry-crane spans: `span_zone_id`, `dock_zone_id` (the dock of the gantry crane),
`covered_zone_id`, `share` (share of the area of the covered zone that lies inside the span). An erection from an assembly area
whose share is at least `span_min` (0.5) is lifted by the gantry crane directly, without a vehicle.

## Checks

The derived graphs reproduce the networks built from the maps exactly: roads, arcs, turn permissions, stops, sections,
capacities, cranes, lifts, turn decisions and trip occupancies are identical, and rerunning stored simulation points on these
graphs reproduces them to the last bit (`experiments/rerun.py`).

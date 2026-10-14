# Result data

Every number, table of results and data figure of the paper is generated from the files of this directory by
`paper/reproduce.py`. `MANIFEST.csv` lists every file with its SHA-256, its size and where the paper uses it.

## Layout

| Directory | Contents | Paper |
|---|---|---|
| `network/` | counts of the two networks, cranes and crane spans, Okpo revisions, sectioned dock roads, short roads | Section 4.1, Appendix B.1 |
| `saturation/` | saturation curves and plateaus of the traffic rules, onset of saturation K*, refusals of safe release, teleports of the reference rule, width sensitivity | Fig. 4, Sections 4.2-4.3, Appendix C |
| `bounds/` | system bound, route-free bound and route margin, network-bound samples, convergence, read-out of the binding resource, dual prices (`okpo/`, `yantai/`); system bounds by granularity; section and block length; bound sensitivity; sampling of origins and destinations | Table 6, Section 4.2, Appendix C |
| `interval/` | capacity interval (lower end: full-path reservation; upper end: system bound), crane-paced erection, 30-min loading and unloading | Fig. 5, Section 4.3, Appendix C.3 |
| `fleet/` | required fleets (all levels, both working days, four rules), eight-day checks, 30-min handling, crane pacing, the screen of demand over the bound, the bootstrap size | Table 7, Figs. 6-7, Appendix C.4 |
| `dock/` | dock-mouth thresholds rho_10 at both yards (`thresholds.csv`, one row per yard, granularity, rule and fleet), crane hours, utilisation levels, one crane at Dock 1 | Fig. 8, Section 4.4, Appendix D.1 |
| `synthetic_dock/` | synthetic dock mouth: configuration, every simulated point, thresholds, waiting term | Fig. 9, Appendix D.2 |
| `anchors/` | external anchor against handling time, blocks per vehicle-day | Fig. 10, Section 4.5, Appendix C.5 |
| `runs/<run>/` | one simulation or bound batch each: `config.json`, and where released `points*.csv` (one row per simulated point) and `check.json` (the per-point check against the network bound) | Section 4.5 and the sections of the run |

## Per-point tables of the dock-mouth thresholds

The thresholds of `dock/thresholds.csv` (Fig. 8) are read from these runs; their tables hold the points of the cells of
`thresholds.csv` (scenarios `rho<N>`, `p6lo`, `p6hi` and `crane`, with `_8d` / `_16d` for eight or 16 simulated days and `_s`
for dock roads sectioned at the stops). Column `src8` of `thresholds.csv` names the eight-day run of each cell; the cells
with a 16-day reading (`reading_16d`) come from the 16-day run of the yard.

| Run | Yard, days | Files |
|---|---|---|
| `runs/okpo_dock_8day` | Okpo (revision 1833), 8 days | `points_whole.csv`, `points_stops.csv` |
| `runs/okpo_dock_16day` | Okpo (revision 1833), 16 days | `points_whole.csv`, `points_stops_K40-100.csv`, `points_stops_K150.csv` |
| `runs/yantai_dock_safe` | Yantai, 8 days: safe release; full-path reservation sectioned | `points.csv` |
| `runs/yantai_crane_curves` | Yantai, 8 days: full-path reservation and reference rule, whole roads | `points.csv` |
| `runs/yantai_rules_dock` | Yantai, 8 days: reference rule sectioned, K = 20 | `points.csv` |
| `runs/yantai_dock_8day` | Yantai, 8 days: reference rule sectioned, the other fleets | `points.csv` |
| `runs/yantai_dock_16day` | Yantai, 16 days | `points_whole.csv`, `points_stops.csv` |

A table larger than 45 MB is split into parts with the same columns (by granularity and, where still too large, by fleet);
concatenating the parts with the header line once gives the whole table, and `experiments/rerun.py` reads them together.

## Conventions

- Yard keys: `yupu` is Okpo (its Chinese name), `yantai` is Yantai; `yupu_r1825` is Okpo's earlier network revision.
  Network revisions are named in each `config.json` (`network`, a directory of `data/networks/`).
- Traffic rules (`model`): `free` free flow; `reserve` full-path reservation; `segment_safe` safe release; `segment` the
  reference rule (stepwise requests with teleport recovery). Granularity: `whole` roads, or `stops` (`_s` in scenario
  names) = dock roads sectioned at the stops.
- Demand levels (`level`): `saturated` (an endless supply of tasks), `n<N>` (N tasks a day), `regular`, `peak`.
- Throughput is in blocks (tasks) per working day, read over days 2-4 of four simulated days (or days 2-8 / 2-16 of eight or
  16 days); bounds are in tasks per day; times in seconds unless a column says min or h.
- `T1` is the network bound, `T1′` (`T1_tight`, `T1_sys`, `T1_system`) its tightened form with width packing (the system
  bound when computed on the exact task mix), `T1″` (`T1pp`) the route-free bound.
- Resource names: `road161 (W=9.0, 1095 m)` is a road with its width and length; `junction <node> [<building>·<entrance>,
  <short id>]` a junction or entrance node; `crane <dock>` the cranes of a dock; `stop <zone>#<k>` a stop. Tables written
  by earlier versions of the code (the bound tables in `bounds/okpo`, `bounds/yantai` and the older runs) give junction
  names without the bracketed tag, or with the building only.
- In `interval/crane_interval.csv` the rows of kind `best` give the bound of the best rule (`T1`, `top1`, `top2`); its other
  columns are empty there.
- CSV files are UTF-8 with a header row; empty cells are missing values; floats are written in full precision.
- JSON summaries (`*summary*.json`, `check.json`, `network/*.json`) are excerpts of larger summaries of the runs: the parts
  the paper's scripts read. The per-point checks (`check.json`) count every point of their batch.

## `points.csv` of a run

One row per simulated point: `scen`, `model`, `level`, `K` (fleet size), `seed`; `throughput` (tasks/day), `completed`,
`flow_mean`, `flow_p95` (task flow time, s), `wait_mean`, `wait_p95`, deadlocks and near-deadlocks (stepwise rules; the
reference rule's `deadlock_times_h` lists the time of every deadlock), `backlog`, `T1_tight_mix` (network bound T1′ of the
tasks completed in the point's own statistics window; the throughput of every occupancy-rule point is at most this value),
P6 (erection) flow, dock-mouth readings (`dock0_*`: the dock stop roads; `dock_*`: the dock-mouth zone), the most occupied
resources (`util_top1*`, `busy_top1*`), steady-state readings of saturated points (`mix_shift`, `drift`, `longrun_est`,
`thr_by_day`), crane and lift readings, crane starvation, refusals of safe release and cycle parts (`cyc_*`, min per task).
The columns are those of `experiments/runner.py` (`FIELDS`); earlier runs have a prefix of them.

Rerun stored points with `experiments/rerun.py` (see the main README).

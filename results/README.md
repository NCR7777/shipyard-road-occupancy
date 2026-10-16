# Result data

Every number, table of results and data figure of the paper and of its supplementary material is generated from the files
of this directory by `paper/reproduce.py`. `MANIFEST.csv` lists every file with its SHA-256, its size, the commit of the
source data it was exported from (`source_commit`) and where the paper uses it. `RECOMPUTE.md` (repository root) names the
script and configuration that recompute each class of files.

## Layout

| Directory | Contents | Paper |
|---|---|---|
| `main_case/` | the main case (erection flow conserved, every other move between areas): system bounds of the three mappings with demand ratios and binding elements (`bounds.csv`, `binding.csv`), with sectioned dock roads and 30-min handling (`bounds_sections.csv`), saturation curves and plateaus (`saturation_curves.csv`, `saturation.csv`), capacity interval (`interval.csv`), fleets K_req and K_stable of the four rules (`fleet.csv`, `fleet_rules.csv`, `fleet_30min.csv`), the scan of moves within areas (`scan.csv`, `scan_switch.csv`), crane load by dock, loaded occupancy and peak waits, per-point checks (`check*.csv`) | Table 7, Figs. 4-5, Table 9, Fig. S1, Sections 4.1-4.4 |
| `intra_area/` | moves within areas: bounds, demand ratios, sampled bounds, placement variants, fleets, screen, saturation, and the 30-day fleets without growing backlog of both inter-area mappings and this one (`stable.csv`) | Tables 7, 9, S2, Appendix B, Supplementary S1.4 |
| `ablation/` | ablation of the occupancy representation, main case and every move scaled: system bounds, fleets, checks | Tables 8, S3 |
| `retiming/` | whole-network retiming of full-path reservation's schedules and offline dispatch (main case, every move scaled) | Table S5, Section 4.3 |
| `subnet/` | sub-network instances: rule, fixed-order and best schedules, bounds (`points.csv`), the through corridor (`corridor*.csv`) | Table S4, Section 4.3 |
| `screen/` | the screen in two levels: extra fleet against the demand ratio and the chain load (`series*.csv`, `cross*.csv`), 30-seed reruns of the low levels, waits and the widened roads, across yards and road widths (`widths*.csv`, `leave_one_out.csv`), densified low levels (`low_*.csv`) | Fig. 6, Section 4.4, Supplementary S1.4 |
| `stochastic/` | random handling and driving times, late rebooking, full-path reservation executed in booked order (`ordered_*.csv`) | Table S6, Section 4.3 |
| `width_ban/` | ban on blocks wider than the passable width: bounds and fleets (`summary.json`), passable widths read on the imagery (`clear_width.json`) | Section 4.1, Supplementary S1.1 |
| `tail/` | release at tail clearance: roadside stops within a clearing length, plateaus and fleets when the tail is booked | Appendix A.1, Supplementary S1.1 |
| `hand/` | the three hand-computed instances: hand values and simulated throughput | Appendix A.6, Supplementary S4 |
| `network/` | counts of the two networks, cranes and crane spans, Okpo revisions, sectioned dock roads, roads shorter than a vehicle | Section 4.1, Appendices A.1, B.1 |
| `saturation/` | every move scaled: saturation curves and plateaus of the traffic rules, onset of saturation K*, refusals of safe release, teleports of the reference rule, width sensitivity | Section 4.2, Supplementary S1.3 |
| `bounds/` | every move scaled: system bound, route-free bound and route margin, network-bound samples, convergence, read-out of the binding resource, dual prices (`okpo/`, `yantai/`); system bounds by granularity; section and block length; bound sensitivity; sampling of origins and destinations. Main case: route-free bounds and certificates (`route_free_*.csv`, `interval_route_free.csv`), the bound with the utilisation rows only (`packing_rows.csv`), platform size (`platform_size.csv`) | Table 7, Fig. 5, Section 4.2, Supplementary S1.1-S1.2 |
| `interval/` | every move scaled: capacity interval, crane-paced erection, 30-min loading and unloading | Fig. 5, Supplementary S1.3 |
| `fleet/` | every move scaled: required fleets (all levels, both working days, four rules), eight-day checks, 30-min handling, crane pacing, the 10 % screen, the bootstrap size | Table S2, Fig. S2, Supplementary S1.4, Appendix B.4 |
| `dock/` | dock-mouth thresholds rho_10 at both yards (`thresholds.csv`, one row per yard, granularity, rule and fleet), crane hours, utilisation levels, one crane at Dock 1, off-road unloading bays (`bays.csv`) | Fig. 7, Table S7, Section 4.5, Supplementary S2.1 |
| `synthetic_dock/` | synthetic dock mouth: configuration, every simulated point, thresholds, waiting term | Fig. 8, Supplementary S2.2 |
| `anchors/` | external anchor against handling time, moves per vehicle-day | Fig. 9, Sections 4.1, 4.6 |
| `runs/<run>/` | one simulation or bound batch each: `config.json`, and where kept `points*.csv` (one row per simulated point), `demand.csv` (the required-fleet search), `daily.csv` and `stable.json` (30-day runs: day-by-day readings and the K_stable search), `bound_points.json` (sampled bounds per seed) and `check.json` (the per-point check against the network bound) | the sections of the run |

The runs of the main case are `runs/{okpo,yantai}_main_*` (saturation, interval rows, fleets, fleets of safe release and
the reference rule, 30-min handling, 30-day runs, screen, 30-seed screen, ablation); the intra-area mapping
`runs/okpo_intra_*`; the 30-day runs of the proportional mapping `runs/okpo_stable_30day`; random times `runs/*_stoch_*`;
execution in booked order `runs/*_ordered_*`; the ban `runs/okpo_ban_fleet`; the off-road bays `runs/*_dock_bay*`; the
screen across road widths and its low levels `runs/*_screen_widths`, `runs/*_screen_low`, `runs/okpo_*screen_30seeds`;
tail booking `runs/*_tail_*`. Five batches kept no per-point table: `runs/{okpo,yantai}_intra_saturation`,
`runs/okpo_stable_30day`, `runs/okpo_ablation_stable` and `runs/okpo_main_ablation_stable`; their configuration (and the
K_stable search, `stable.json`) is released, and their readings are the per-cell tables `intra_area/saturation.csv`,
`intra_area/stable.csv` and `ablation/summary_*.json`.

## Per-point tables of the dock-mouth thresholds

The thresholds of `dock/thresholds.csv` (Fig. 7) are read from these runs; their tables hold the points of the cells of
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
- Mappings of demand to moves (`map`): `conserved` the main case (erection flow at its weight, the other moves between
  areas); `main`, `proportional` or `current` every move scaled with the flows; `intra` moves within areas
  (`inter_only`: the inter-area moves alone). In labels, "proportional" names the same mapping.
- Traffic rules (`model`): `free` free flow; `reserve` full-path reservation; `segment_safe` safe release; `segment` the
  reference rule (stepwise requests with teleport recovery); `reserve_rb` full-path reservation rebooked when late
  (random times); `reserve_op`, `reserve_op2`, `reserve_op3` full-path reservation executed in booked order, booked at nominal
  times, on projected occupancy, and on projected occupancy at mean durations; `reserve_tlw` full-path reservation booking
  the tail. Granularity: `whole` roads, or `stops` (`_s` in scenario names) = dock roads sectioned at the stops.
- Demand levels (`level`): `saturated` (an endless supply of tasks), `n<N>` (N tasks a day), `regular`, `peak`.
- Throughput is in moves (tasks) per working day, read over days 2-4 of four simulated days (or days 2-8 / 2-16 of eight
  or 16 days); in the 30-day runs the end-of-day backlog is regressed on the day over days 6-30; bounds are in tasks per
  day; times in seconds unless a column says min or h.
- `T1` is the network bound, `T1′` (`T1_tight`, `T1_sys`, `T1_system`, `lam16`, `lam_S`) its tightened form with width
  packing (the system bound when computed on the exact task mix), `T1″` (`T1pp`, `certificate`) the route-free bound.
- `K_req` is the required fleet (the smallest fleet whose mean throughput over the ten seeds reaches 99 % of the demand),
  `K_stable` the fleet without a growing backlog over 30 days (criterion in `experiments/daily.py`; paper, Appendix B.4); a superscript `a` (`ᵃ`) marks a fleet above which a larger tested fleet fails
  again.
- Resource names: `road161 (W=9.0, 1095 m)` is a road with its width and length; `junction <node> [<building>·<entrance>,
  <short id>]` a junction or entrance node; `crane <dock>` the cranes of a dock; `stop <zone>#<k>` a stop. Tables written
  by earlier versions of the code (the bound tables in `bounds/okpo`, `bounds/yantai` and the older runs) give junction
  names without the bracketed tag, or with the building only.
- Run and batch names in columns (`batch`, `dir`, `src`, `source`) are the names of this directory (`runs/<run>`,
  `main_saturation`, ...); "unreleased" marks a run whose files are not part of the release, "earlier" or "low_demand" an
  earlier screen of the paper.
- In `interval/crane_interval.csv` the rows of kind `best` give the bound of the best rule (`T1`, `top1`, `top2`); its other
  columns are empty there.
- CSV files are UTF-8 with a header row; empty cells are missing values; floats are written in full precision.
- JSON summaries (`*summary*.json`, `check.json`, `network/*.json`, `screen/wide_config.json`) are excerpts of larger
  summaries of the runs: the parts the paper's scripts read. The per-point checks (`check.json`, `check*.csv`) count every
  point of their batch.

## `points.csv` of a run

One row per simulated point: `scen`, `model`, `level`, `K` (fleet size), `seed`; `throughput` (tasks/day), `completed`,
`flow_mean`, `flow_p95` (task flow time, s), `wait_mean`, `wait_p95`, deadlocks and near-deadlocks (stepwise rules; the
reference rule's `deadlock_times_h` lists the time of every deadlock), `backlog`, `T1_tight_mix` (network bound T1′ of the
tasks completed in the point's own statistics window; the throughput of every occupancy-rule point is at most this value),
P6 (erection) flow, dock-mouth readings (`dock0_*`: the dock stop roads; `dock_*`: the dock-mouth zone), the most occupied
resources (`util_top1*`, `busy_top1*`), steady-state readings of saturated points (`mix_shift`, `drift`, `longrun_est`,
`thr_by_day`), crane and lift readings, crane starvation, refusals of safe release, cycle parts (`cyc_*`, min per task) and
the rebooking readings of `reserve_rb` (`rb_late`, `rb_rebook`, `rb_late_mean_s`). The columns are those of
`experiments/runner.py` (`FIELDS`); earlier runs have a prefix of them.

Rerun stored points with `experiments/rerun.py` (see the main README).

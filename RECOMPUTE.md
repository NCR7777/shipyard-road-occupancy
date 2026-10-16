# Recomputing the results

`paper/reproduce.py` writes the paper's numbers, tables and figures from the result files of `results/` without simulating.
This page names, for each class of those files, the script and the stored configuration or arguments that recompute it
from the derived graphs (`data/networks/`) and the parameters (`occupancy/params.py`, and the `params` and `scenarios` of
every stored configuration). Every command writes into the directory given by `--out` (a cache of finished points under
`<out>/points/` lets an interrupted run resume) and accepts `--procs N`; `--slots DIR` shares a machine with other jobs
through a directory of CPU slots, and without it an ordinary process pool is used. The definitions by which the tables
read the point tables (the plateau, the onset of saturation K*, the required fleet K_req, the fleet without growing
backlog K_stable, the first 10 % crossing of the screen, the paired-seed bootstrap) are those of Appendix B.4 of the paper.

## Point tables of the batches: `results/runs/<run>/`

| Files | Command |
|---|---|
| `points.csv` of a batch of simulated points (saturation curves, interval rows, crane pacing, 30-min handling, random times, execution in booked order, tail booking, off-road bays, dock-mouth runs, 30-seed reruns) and its sampled bounds (`bound_points.json`, `check.json`: the per-point check) | `python experiments/runner.py results/runs/<run>/config.json --out out/<run>` |
| `demand.csv` and `points.csv` of a required-fleet search (configurations with demand levels `n<N>`) | `python experiments/required_fleet.py results/runs/<run>/config.json --out out/<run>` |
| `stable.json`, `daily.csv`, `points.csv` of a 30-day search for K_stable, starting from K_req of the search named | every move scaled and moves within areas: `python experiments/stable_fleet.py results/runs/okpo_stable_30day/config.json --out out/stable --kreq results/fleet/required_fleet.csv results/runs/okpo_intra_fleet/demand.csv`; main case: `python experiments/stable_fleet_conserved.py results/runs/okpo_main_stable/config.json --out out/main_stable --kreq results/runs/okpo_main_fleet/demand.csv` (30-min handling: `okpo_main_stable_30min` with `okpo_main_fleet_30min`); ablation: `python experiments/stable_fleet_ablation.py` (see its docstring) with `okpo_ablation_stable` / `okpo_main_ablation_stable` and the searches `okpo_ablation_fleet` / `okpo_main_ablation_fleet`; tail booking: `python experiments/tail_runs.py` (see its docstring) with `okpo_tail_stable` and `okpo_tail_fleet` |
| a single stored point, compared with its row | `python experiments/rerun.py results/runs/<run> --pick scen=...,model=...,K=...,seed=...` |
| the synthetic dock mouth (`results/synthetic_dock/`) | `python experiments/dock_mouth.py --out out/dock` |

`runner.compute` hands a point to the module of its rule or scenario: `reserve_rb` and random times (`stoch`) to
`stochastic.py`; `reserve_op`, `reserve_op2`, `reserve_op3` (execution in booked order) to `ordered.py` and
`ordered_proj.py`; `reserve_tlw`, `reserve_tlw_timer` (booking the tail) to `tail_booking.py`; the demand mappings
(`demand_map`) to `demand_mapping.py`, `demand_mapping_var.py` and `conserved_mapping.py`; `width_ban` to `width_ban.py`;
`bay_spots` to `dock_spots.py`; the ablation (`occ_rep`, `empty_rule`) to `ablation.py` and `ablation_conserved.py`; the
day-by-day readings of the 30-day runs (`daily_stats`) to `daily.py`.

## Tables by directory

| Files | Recomputed by | From |
|---|---|---|
| `bounds/okpo/`, `bounds/yantai/` | `experiments/capacity_bounds.py` | their `config.json` (README, "Recomputing the results") |
| `bounds/system_by_granularity.*`, `bounds/sensitivity.csv`, `bounds/block_length.csv`, `bounds/od_sampling.csv` | `experiments/capacity_bounds.py` and its functions (`item_system`, `keep_rows`) | the configurations of `bounds/*/config.json` with the scenario changes named in the table rows; `runs/*_od_sampling/config.json` |
| `main_case/bounds.csv`, `binding.csv`, `scan.csv`, `scan_switch.csv` | `experiments/conserved_bounds.py --mapping-bounds results/intra_area/bounds.csv` (the scan of moves within areas, q = 0 the main case, with the binding elements and dual prices) and `experiments/mapping_bounds.py --sensitivity results/bounds/sensitivity.csv` (every move scaled and moves within areas); the demand ratios are the levels over these bounds | the derived graphs okpo/rev1833, yantai/rev396 |
| `main_case/bounds_sections.csv` | `experiments/conserved_bounds_sections.py --conserved-scan results/main_case/scan.csv` | the same |
| `main_case/saturation_curves.csv`, `saturation.csv` | `experiments/runner.py` (seed mean, minimum and maximum per K; plateau, K*, crossing of free flow; sampled bounds from `bound_points.json`) | `runs/okpo_main_saturation`, `runs/yantai_main_saturation` |
| `main_case/interval.csv` | `experiments/runner.py`, with the bounds of `conserved_bounds_sections.py` | `runs/*_main_interval`, `runs/*_main_saturation` |
| `main_case/fleet.csv`, `fleet_rules.csv`, `fleet_30min.csv` | `experiments/required_fleet.py` (K_req) and `experiments/stable_fleet_conserved.py` (K_stable) | `runs/okpo_main_fleet`, `okpo_main_fleet_rules`, `okpo_main_fleet_30min`; `runs/okpo_main_stable`, `okpo_main_stable_30min` |
| `main_case/crane_by_dock.csv`, `loaded_occupancy.csv`, `peak_waits.csv` | `experiments/mechanisms.py` (its docstring names the runs and arguments; the peak waits read the per-point files of reruns of the peak cells) | `runs/okpo_main_fleet`, `runs/okpo_rules`, `runs/*_main_saturation`, `intra_area/bounds.csv`, `main_case/scan.csv` |
| `main_case/check*.csv`, `intra_area/check*.csv`, `stochastic/*check.csv` | the column `T1_tight_mix` that `runner.py` writes for every point of an occupancy rule: points checked, points above their bound, largest ratio | the runs named in each row |
| `intra_area/bounds.csv`, `variant_bounds.csv`, `sampled.csv`, `ratio.csv` | `experiments/mapping_bounds.py` | the derived graphs |
| `intra_area/fleet.csv`, `fleet_30min.csv`, `variants.csv`, `screen.csv`, `screen_cross.csv`, `marks.csv` | `experiments/required_fleet.py` | `runs/okpo_intra_fleet`, `okpo_intra_fleet_30min`, `okpo_intra_fleet_8day`, `okpo_intra_variants`, `okpo_intra_screen` |
| `intra_area/saturation.csv` | `experiments/runner.py` | `runs/okpo_intra_saturation`, `runs/yantai_intra_saturation` |
| `intra_area/stable.csv` | `experiments/stable_fleet.py` | `runs/okpo_stable_30day` |
| `ablation/bounds_*.csv` | `ablation.system_bound` and `ablation_conserved.system_bound` for each yard, scenario and representation | the derived graphs |
| `ablation/summary_*.json` | `experiments/required_fleet.py` (K_req, per-point checks), `experiments/stable_fleet_ablation.py` (K_stable) | `runs/okpo_ablation_fleet`, `okpo_main_ablation_fleet`, `okpo_ablation_stable`, `okpo_main_ablation_stable` |
| `retiming/main_case.csv`, `proportional.csv` | `experiments/retime.py` (`--map conserved` for the main case; its docstring names `--system results/bounds/system_by_granularity.csv`, `--plateau-okpo results/saturation/okpo_readings.csv`, `--plateau-yantai results/saturation/rules_summary.json`, `--conserved-report results/main_case`, `--dispatch results/retiming/offline_dispatch.csv`) | the derived graphs; schedules of full-path reservation simulated afresh |
| `retiming/offline_dispatch.csv`, `offline_dispatch_proportional.csv` | `experiments/offline_dispatch.py` (`--map conserved` for the main case; the same reference arguments as `retime.py`) | the same |
| `subnet/points.csv`, `config.json` | `experiments/subnet_exact.py --out out/subnet` (CP-SAT) | the derived graphs and the synthetic dock map of `dock_mouth.py` |
| `subnet/corridor.csv`, `corridor_split.csv` | `experiments/subnet_bounds.py`, `subnet_gap.py` and `corridor.py`, each reading the `--out` directory of the `subnet_exact.py` run (`--subnet out/subnet`; the per-point schedules it writes are not part of the release) | the same |
| `screen/series.csv`, `cross.csv` | `experiments/chain_load.py --sensitivity results/bounds/sensitivity.csv --mapping-bounds results/intra_area/bounds.csv` (chain loads S_d of each series) with the fleets of `fleet/screen.csv`, `screen/widths.csv` and `intra_area/screen.csv` | the derived graphs |
| `screen/seeds_levels.csv`, `waits_pairs.csv`, `wide.csv`, `wide_config.json`, `summary.json` | `experiments/screen_runs.py` (`--part seeds`, `intra`: 30-seed reruns of the low levels; `--part waits`: waiting readings at the cells of `--screen results/fleet/screen.csv` with `--wait-configs` the configurations of `runs/okpo_screen_low`, `runs/yantai_screen_low`; `--part wide`: Okpo's roads of at most 15 m widened to 18 m, whose widening of the roads along the docks needs the dock outlines of the yard maps, which are not part of the derived graphs) | `runs/okpo_screen_30seeds`, `runs/okpo_intra_screen_30seeds`, `fleet/screen.csv`, `screen/wide_config.json` |
| `screen/*_main_case.*` | `experiments/screen_conserved.py --part screen|seeds --yard yupu|yantai --conserved-scan results/main_case/scan.csv` | `runs/*_main_screen`, `runs/*_main_screen_30seeds` |
| `screen/widths.csv`, `leave_one_out.csv`, `widths_summary.json`, `low_*.csv`, `low_summary.json` | `experiments/required_fleet.py` | `runs/*_screen_widths`, `runs/*_screen_low` |
| `stochastic/curves.csv`, `plateau.csv`, `deadlocks.csv`, `eight_day.csv`, `summary.json` | `experiments/runner.py` | `runs/*_stoch_curves`, `*_stoch_variants`, `*_stoch_8day` |
| `stochastic/fleet.csv` | `experiments/required_fleet.py` | `runs/okpo_stoch_fleet` |
| `stochastic/ordered_*.csv` | `experiments/runner.py`, `experiments/required_fleet.py` | `runs/*_ordered_*` |
| the hand cases and adversarial checks of execution in booked order, reversing and the release at tail clearance (not tables of `results/`) | `experiments/ordered.py`, `ordered_check.py`, `reverse_cover.py`, `tail_release.py` | instances built in the code |
| `width_ban/summary.json` | `experiments/width_ban_bounds.py --part roads`, then `--part bounds --earlier results/bounds/sensitivity.csv` (system bounds under the ban), and `experiments/required_fleet.py` (fleets) | `width_ban/clear_width.json`; `runs/okpo_ban_fleet` |
| `dock/bays.csv` | `experiments/runner.py` | `runs/*_dock_bay`, `*_dock_bay2`, `*_dock_bay_16day` |
| `bounds/route_free_mappings.csv`, `route_free_main_case.csv`, `interval_route_free.csv` | `experiments/route_free_mappings.py --conserved-scan results/main_case/scan.csv` | the derived graphs |
| `bounds/packing_rows.csv` | `experiments/packing_rows.py --sensitivity results/bounds/sensitivity.csv --route-free results/bounds/route_free_mappings.csv` | the same |
| `bounds/platform_size.csv` | `experiments/platform_size.py` | the same |
| `tail/saturation.csv`, `fleet.csv` | `experiments/tail_runs.py` (`--part sat`, `demand`, `stable`; `--kreq results/runs/okpo_tail_fleet/demand.csv` for the 30-day search), with the release by timer from `main_case/` | `runs/*_tail_saturation`, `okpo_tail_fleet`, `okpo_tail_stable` |
| `tail/roadside_stops.json` | `experiments/roadside_stops.py` | the derived graphs |
| `hand/instances.json` | the instances of `tests/test_occupancy.py` (`test_T1_tight_corridor_hand_value`, `test_T1_tight_crossroads_hand_value`, `test_T1_tight_reuse_regression`) simulated at the fleets of the file | instances built in the tests |
| the files of `saturation/`, `interval/`, `fleet/`, `dock/`, `anchors/`, `network/` (every move scaled with the flows) | `experiments/runner.py` and `experiments/required_fleet.py` | the runs named in `MANIFEST.csv` and in `results/README.md` |

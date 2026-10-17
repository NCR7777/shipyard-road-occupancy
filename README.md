# shipyard-road-occupancy

Experiment code, input road networks and result data of the paper

> **Capacity bounds for shipyard block transport with load-dependent road occupancy**
> Chunrui Na, Qing Wang, Jingzheng Yao, Xiao Luo
> College of Shipbuilding Engineering, Harbin Engineering University; School of Civil and Environmental Engineering, Nanyang
> Technological University; Yantai Research Institute, Harbin Engineering University; School of Marine Science and
> Technology, Northwestern Polytechnical University.
> Submitted to *Applied Mathematical Modelling*.

Self-propelled transporters move ship blocks between workshops, stockyards and the docks over the yard roads. The code
models roads, junctions, entrances and stops as resources that a vehicle holds for blocking times at load-dependent widths;
simulates traffic rules on the road networks of two yards, Okpo (Hanwha Ocean, Korea) and Yantai (CIMC Raffles, China); and
computes linear-programming capacity bounds for the rules that respect road widths and release resources only by timer (a
system bound, and a route-free bound with a solver-independent certificate), with constraint programmes of small instances
and an independent schedule checker.

The repository holds

- the code: the occupancy simulation and its traffic rules, the capacity bounds and certificates, the constraint programmes,
  and the script of every experiment of the paper;
- the input road networks of both yards as derived graphs (nodes and roads with their lengths, resources with their widths,
  stops, turn permissions and the turn angle at each junction, cranes and crane spans), without coordinates;
- the result data of every experiment (configurations, per-point tables, per-cell tables), so that each reading can be
  checked without rerunning, and any stored point can be rerun and compared.

## Code

```
occupancy/            simulation model: network (derived graphs), tasks, traffic rules (discrete-event simulation),
                      network bound T1 / T1' (bound.py), parameters and scenarios (params.py), map -> graph (mapgraph.py)
bounds/               system bound (system.py), route-free bound with Lagrangian certificate (route_free.py)
exact/                constraint programme (CP-SAT) of small instances (model.py) and the independent schedule checker
                      (check.py)
experiments/          batch runner and searches: runner.py (simulated points and sampled bounds of a stored configuration),
                      rerun.py (rerun stored points), required_fleet.py (required-fleet search K_req), stable_fleet*.py
                      (30-day fleets without growing backlog K_stable, criterion in daily.py);
                      demand mappings: demand_mapping.py, demand_mapping_var.py, conserved_mapping.py (the main case);
                      bounds: capacity_bounds.py, mapping_bounds.py, conserved_bounds*.py, route_free_mappings.py,
                      packing_rows.py, platform_size.py, chain_load.py (chain load of the screen);
                      traffic rules beyond occupancy/: stochastic.py (random times, late rebooking), ordered.py,
                      ordered_proj.py, ordered_check.py (execution in booked order), reverse_cover.py (reversing),
                      tail_booking.py, tail_release.py, tail_runs.py, roadside_stops.py (release at tail clearance),
                      width_ban*.py (ban on over-wide blocks), dock_spots.py (off-road bays at the dock), ablation*.py
                      (ablation of the occupancy representation);
                      sub-networks and the whole network: subnet_exact.py, subnet_bounds.py, subnet_gap.py, corridor.py,
                      retime.py, offline_dispatch.py;
                      interventions on the binding element: intervention_net.py, intervention_bounds.py,
                      intervention_runs.py, intervention_dispatch.py, intervention_subnet*.py, intervention_diag.py,
                      hold_ratio_matched*.py (the holding ratio with the same method on both sides);
                      the screen: screen_conserved.py, screen_runs.py, screen_stable.py, chain_statistics.py;
                      random times in the main case: stochastic_main_stable.py; mechanisms.py; dock_mouth.py (synthetic
                      dock mouth)
tests/                pytest suite of the model, the bounds and the experiment code, on the released data
data/networks/        input road networks (derived graphs): okpo/rev1833, okpo/rev1825, yantai/rev396 (fields in its README.md)
results/              result data by experiment, runs/ (configurations and per-point tables of the batches), MANIFEST.csv
                      (SHA-256, source commit and manuscript location of every file), README.md (layout and fields)
RECOMPUTE.md          for every result file: the script, the stored configuration and the arguments that recompute it
requirements.txt      pinned package versions (Python 3.13.5, pip)
environment.yml       the same versions as conda builds
LICENSE, LICENSE-data MIT (code), CC BY 4.0 (data)
CITATION.cff          how to cite
```

## Installation

Python 3.13.5 (64-bit); the experiments ran on Windows 11 (24 logical cores).

```
python -m venv .venv
.venv\Scripts\activate            (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python -m pytest tests            (about 4 minutes)
```

`conda env create -f environment.yml` gives the same versions as conda builds, the environment in which the results were
computed. In both, a stored point reruns to the last bit, and the per-cell tables that the code rebuilds from the released
point tables (the screen's lower end, the holding ratio with the same method, the main case's random-time plateaus) equal
the released ones byte for byte; the one exception is the chain statistics of the screen (`screen/chain_stats.json`),
written with numpy on MKL (conda): numpy's pip wheel (OpenBLAS) sums their dot products in another order, so its values
differ in the last bits (the r2 of its isotonic fit by about 1e-7). OR-Tools (CP-SAT) is needed only for the
sub-network instances, the whole-network retiming and the interventions' sub-networks.

## Experiments

Every experiment reads the derived graphs of `data/networks/`, the parameters of `occupancy/params.py` and those of its
stored configuration, and writes into the directory given by `--out` (a cache of finished points under `<out>/points/`
lets an interrupted run resume). The commands accept `--procs N`; worker processes are pinned to the cores other than core
0; `--slots DIR` (optional) shares a machine with other jobs through a directory of CPU slots, and without it an ordinary
process pool is used. Run times are those of the stored runs: the summed run time of their points (CPU hours, the column
`secs` of the point tables) and the typical time of one point. `RECOMPUTE.md` gives the full arguments for every result
file; the definitions by which the per-cell tables read the point tables (plateau, onset of saturation K*, required fleet
K_req, fleet without growing backlog K_stable, 10 % crossing of the screen, paired-seed bootstrap) are those of Appendix B.4
of the paper. In the table, `<run>` is a directory of `results/runs/`.

| Experiment | Module and command | Results | Run time | Manuscript |
|---|---|---|---|---|
| System and route-free bounds, every move scaled with the flows | `capacity_bounds.py results/bounds/okpo/config.json --out out/bounds_okpo --scens main --items system route_free seeds` (Yantai: `bounds/yantai`) | `bounds/okpo/`, `bounds/yantai/`, `bounds/system_by_granularity.*`, `bounds/sensitivity.csv` | seconds per scenario; the route-free programme of Okpo's crane-paced scenario minutes | Table 7; Section 4.2; Supplementary Section S1.2 |
| Bounds of the main case (erection conserved) and of the other mappings | `conserved_bounds.py`, `conserved_bounds_sections.py`, `mapping_bounds.py`, `route_free_mappings.py` | `main_case/bounds*.csv`, `binding.csv`, `scan*.csv`; `intra_area/bounds.csv`; `bounds/route_free_*.csv` | 6-72 s per system bound (`interventions/computing_times.csv`) | Table 7; Fig. 4; Section 4.2; Appendix B.2; Supplementary Fig. S1 |
| Saturation curves of the four rules, main case | `runner.py results/runs/<run>/config.json --out out/<run>` with `okpo_main_saturation`, `yantai_main_saturation` | `main_case/saturation_curves.csv`, `saturation.csv` | 4,800 points, 23 CPU h; 5-6 s per point (at most 3 min) | Fig. 4; Table 7; Section 4.2 |
| Capacity interval, main case | `runner.py` with `*_main_interval` | `main_case/interval.csv` | 6,000 points, 31 CPU h | Fig. 5; Section 4.3 |
| Required fleets K_req and fleets without growing backlog K_stable, main case | `required_fleet.py` with `okpo_main_fleet`, `okpo_main_fleet_rules`, `okpo_main_fleet_30min`; `stable_fleet_conserved.py` with `okpo_main_stable` (30 days) | `main_case/fleet*.csv`; `runs/okpo_main_stable/` | 7,000 points of 4 days, 24 CPU h; 1,700 points of 30 days, 25 CPU h (15 s per point, at most 10 min) | Table 9; Section 4.4; Appendix B.4 |
| Rules, fleets and onset of saturation, every move scaled; moves within areas | `runner.py` with `okpo_rules`, `okpo_rules_erection`, `yantai_rev395_saturation`, `*_crane*`, `*_handling_30min`; `required_fleet.py` with `okpo_rev1825_fleet_*`, `yantai_fleet_plant`, `okpo_intra_*`; `stable_fleet.py` with `okpo_stable_30day` | `saturation/`, `interval/`, `fleet/`, `intra_area/` | 22,000 points of saturation, 91 CPU h; 34,000 points of fleet searches, 112 CPU h; 12,600 points within areas, 3 CPU h | Table 7; Table 9; Supplementary Table S2; Supplementary Fig. S2; Supplementary Sections S1.3-S1.4 |
| Ablation of the occupancy representation | `ablation.system_bound`, `ablation_conserved.system_bound`; `required_fleet.py` with `*_ablation_fleet`; `stable_fleet_ablation.py` | `ablation/` | 2,700 points, 2 CPU h | Table 8; Supplementary Table S3; Supplementary Section S1.5 |
| Sub-network instances (CP-SAT) | `subnet_exact.py --out out/subnet`, then `subnet_bounds.py`, `subnet_gap.py`, `corridor.py` | `subnet/` | 36 points, 11 CPU h; 15 min per point (at most 35 min) | Supplementary Table S4; Supplementary Section S1.6; Section 4.3 |
| Whole-network retiming and offline dispatch | `retime.py`, `offline_dispatch.py` (`--map conserved` for the main case) | `retiming/` | retiming: CP-SAT 25 min per yard and seed (one thread); offline dispatch: 1-3 min per yard and seed | Supplementary Table S5; Supplementary Section S1.6; Section 4.3 |
| The screen: extra fleet against the demand ratio and the chain load | `required_fleet.py` with `*_screen*`, `*_main_screen*`; `screen_conserved.py`, `screen_runs.py`, `chain_load.py`; `screen_stable.py` (30 days); `chain_statistics.py --out out/chain_stats` (from the released files, seconds) | `screen/`, `fleet/screen.csv`, `intra_area/screen*.csv` | 44,500 points, 4 CPU h (0.1-0.4 s per point); 640 points of 30 days, 0.3 CPU h | Fig. 6; Section 4.4; Supplementary Section S1.4 |
| Dock mouth at the two yards, off-road bays | `runner.py` with `*_dock_8day`, `*_dock_16day`, `yantai_dock_safe`, `*_dock_bay*` | `dock/` | 9,700 points, 200 CPU h; 0.5-2 min per point (at most 8 min) | Fig. 7; Section 4.5; Supplementary Table S12; Supplementary Section S2.1 |
| Synthetic dock mouth | `dock_mouth.py --out out/dock` | `synthetic_dock/`, `runs/synthetic_dock_waiting/` | 22,760 points, 5 CPU h | Fig. 8; Section 4.5; Supplementary Section S2.2 |
| External anchors, main case | `runner.py` with `okpo_main_anchor`; `required_fleet.py` with `okpo_main_anchor_fleet` | `anchors/` | 590 points, 0.6 CPU h | Fig. 9; Sections 4.1, 4.6 |
| Interventions on the binding element | `intervention_bounds.py`; `intervention_runs.py --part sat\|demand\|stable` with `*_interv_*`; `intervention_dispatch.py`, `intervention_subnet_run.py` (CP-SAT), `intervention_diag.py` | `interventions/` (all but the files below) | bounds 6-72 s per scenario; 5,100 simulated points, 45 CPU h; whole-network dispatch 5-9 min per scenario and seed; best sub-network schedules 36-47 min per point (45 points) | Supplementary Tables S6-S7; Supplementary Section S1.7; Section 4.3 |
| Holding ratio with the same method on both sides | `hold_ratio_matched.py --out out/hold_matched --bounds results/interventions/bound_points.csv`; tables: `hold_ratio_matched_report.py --out out/hold_matched_report` | `interventions/hold_ratio_matched*.csv`, `computing_times.csv` | under 1 s per sub-network point (36 points); 3-6 min per whole-network point (12 points) | Supplementary Table S8; Supplementary Section S1.7; Section 4.3 |
| Random handling and driving times, booked order, every move scaled | `runner.py` with `*_stoch_*`, `*_ordered_*`; `required_fleet.py` with `okpo_stoch_fleet`, `okpo_ordered_*fleet` | `stochastic/` (all but `main_case_*`) | 21,000 points, 93 CPU h | Supplementary Table S11; Supplementary Section S1.8 |
| Random times in the main case | `runner.py` with `*_stoch_main_curves`, `*_stoch_main_8day`; `required_fleet.py` with `okpo_stoch_main_fleet`; `stochastic_main_stable.py` with `okpo_stoch_main_stable` (30 days); plateaus: `stochastic_main_stable.py --plateau --out out/stoch_main_plateau` | `stochastic/main_case_*.csv` | 18,400 points, 129 CPU h (30-day points 20 s, at most 15 min) | Supplementary Tables S9-S10; Table 9; Fig. 5; Sections 4.3-4.4 |
| Release at tail clearance | `tail_runs.py --part sat\|demand\|stable`; `roadside_stops.py` | `tail/` | 2,900 points, 13 CPU h | Appendix A.1; Supplementary Section S1.1 |
| Ban on over-wide blocks; packing rows; platform size | `width_ban_bounds.py`, `required_fleet.py` with `okpo_ban_fleet`; `packing_rows.py`; `platform_size.py` | `width_ban/`, `bounds/packing_rows.csv`, `bounds/platform_size.csv` | 1,800 points, 0.8 CPU h; the programmes seconds to minutes | Sections 4.1-4.2; Supplementary Section S1.1 |
| Per-point check against the network bound | the column `T1_tight_mix` that `runner.py` writes for every point | `main_case/check*.csv`, `intra_area/check*.csv`, `stochastic/*check.csv`, `runs/*/check.json` | with the points | Section 4.6; Supplementary Section S1.3 |
| Hand-computed instances | the instances of `tests/test_occupancy.py` | `hand/instances.json` | seconds | Section 4.6; Supplementary Section S4; Supplementary Table S15 |

`MANIFEST.csv` gives the manuscript location of every file of `results/`; `results/README.md` describes the layout and the
fields of the files.

## Rerunning stored points and comparing them with results/

Every simulated point is determined by its configuration, network and seed. Rerunning a stored point with this code at its
fixed seed reproduces its row of `points.csv` in every column (except the run time `secs`), floats to the last bit. One
label differs for runs stored by an earlier version of the code: their most-occupied-resource columns (`util_top1_name`,
`busy_top1_name`) name junctions without the bracketed tag `[building·entrance, short id]` that the code now appends, and
`rerun.py` compares these two columns up to the tag (and says so in its output):

```
python experiments/rerun.py results/runs/okpo_rules --pick scen=main,model=reserve,K=30,seed=101
python experiments/rerun.py results/runs/okpo_main_saturation --sample 3 --max-k 40
python experiments/rerun.py results/synthetic_dock --pick rule=seg,W=9.0,rho=0.9,K=24,seed=101
python experiments/rerun.py results/runs/okpo_dock_8day --pick scen=rho8_8d,model=reserve,K=20,seed=101
```

`tests/test_reproduction.py` reruns two stored points this way and rebuilds the per-cell tables of the screen from the
released files. The per-cell tables of the holding ratio and of the main case's random-time plateaus are rebuilt from the
released point tables by `hold_ratio_matched_report.py` and `stochastic_main_stable.py --plateau`; one seed of the holding
ratio's schedules is rerun by `hold_ratio_matched.py ... --only-seed 101`.

The point tables of the yard dock-mouth runs are larger than 50 MB and are split into files of at most 45 MB with the same
columns: by granularity (`points_whole.csv`, whole dock roads; `points_stops.csv`, dock roads sectioned at the stops) and,
for Okpo's 16-day runs, the sectioned part by fleet (`points_stops_K40-100.csv`, `points_stops_K150.csv`). `rerun.py` reads
all `points*.csv` of a run together; to get one table, concatenate them and keep the header line once, for example

```python
import csv, glob
rows = [r for f in sorted(glob.glob("results/runs/okpo_dock_16day/points*.csv")) for r in csv.DictReader(open(f))]
```

Single bounds can be computed directly:

```python
import json, sys
sys.path.insert(0, "experiments")
import capacity_bounds as B
cfg = json.load(open("results/bounds/yantai/config.json"))
print(B.item_system(cfg, "main")["T1"])               # system bound, tasks per 16-h day
```

The network bound of a task sample, the fleet bound and the dual prices are in `occupancy/bound.py`; the per-point check of
a simulated point (`T1_tight_mix`) is `experiments/runner.t1_window`.

## Data sources and what is not released

The road networks were digitised from Google Earth imagery (Google Earth Pro; Okpo dated 15 November 2023: Google, Airbus,
TMap Mobility; Yantai dated 11 March 2025: Google, Airbus). Neither the imagery nor the centre-line geometry traced from it is
released: the networks are released as derived graphs only (`data/networks/`), which hold everything the model and the
bounds use. The passable widths of `results/width_ban/clear_width.json` were read on the same imagery. Map lines delineate
study areas and do not necessarily depict accepted national boundaries. The other inputs (equipment data, handling times,
demand, erection volumes, platform sizes) come from published sources listed in the paper (Appendix B); the parameters are
in `occupancy/params.py` and in every stored configuration.

Okpo's network is released in two revisions. Revision 1833, with the three floating docks, is the main one. Revision 1825 is
the network of the bound tables with every move scaled and of the runs made before the floating docks were digitised: it was
rebuilt from the earlier map (revision 1824, whose road network it shares) with the crane lists and gantry-crane spans of the
two dry docks taken from revision 1833, where they are unchanged. Built this way it equals, element by element, the network
on which those runs and bounds were computed, and the system, network and route-free bounds recomputed from it equal the
stored ones.

Not released: the imagery, the yard maps and the centre-line geometry; the per-point caches of the runs (`<out>/points/`,
with the full schedules of the CP-SAT runs); the per-point tables of the few batches named in `results/README.md`, whose
configurations and per-cell tables are released; and the scripts that typeset the paper's text, tables and figures.

## License and citation

Code: MIT License (`LICENSE`). Data (the derived road networks in `data/networks/` and the result data in `results/`):
Creative Commons Attribution 4.0 International, full legal text in `LICENSE-data`. Copyright 2026 the four authors.
Please cite the paper (`CITATION.cff`).

Repository: https://github.com/NCR7777/shipyard-road-occupancy. On acceptance of the paper the accompanying version will be
archived in Zenodo; its DOI will be given here.

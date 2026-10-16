# shipyard-road-occupancy

Code and data of the paper

> **Capacity bounds for shipyard block transport with load-dependent road occupancy**
> Chunrui Na, Qing Wang, Jingzheng Yao, Xiao Luo
> College of Shipbuilding Engineering, Harbin Engineering University; School of Civil and Environmental Engineering, Nanyang
> Technological University; Yantai Research Institute, Harbin Engineering University; School of Marine Science and
> Technology, Northwestern Polytechnical University.
> Submitted to *Applied Mathematical Modelling*.

Self-propelled transporters move ship blocks between workshops, stockyards and the docks over the yard roads. The paper models
roads, junctions, entrances and stops as resources that a vehicle holds for blocking times at load-dependent widths, and
derives linear-programming capacity bounds for every traffic rule that respects road widths and releases resources only by
timer: a system bound for rules that drive loaded vehicles on shortest routes, and a route-free bound with a
solver-independent certificate for any routing. The capacity of a yard is read as an interval from a deadlock-free rule,
full-path reservation, to the bound. On the public layouts of two yards, Okpo (Hanwha Ocean, Korea) and Yantai (CIMC
Raffles, China), the paper reads the bounds, the saturation of four traffic rules, the fleets they need (in a main case with
the erection flow conserved and under two other mappings of demand to moves), a screen of the fleet by the ratio of demand
to the bound, the dock mouth, and the robustness of these readings to random handling and driving times, the occupancy
representation, road widths, platform size and the release at tail clearance.

This repository holds

- the code of the occupancy simulation, of the traffic rules (including full-path reservation executed in booked order,
  random times with late rebooking and booking the tail), of the capacity bounds and certificates, of the constraint
  programmes of the sub-network instances and the whole-network retiming with their independent schedule checker, and
  the experiment scripts that produced the results;
- the road networks of both yards as derived graphs (nodes and roads with their lengths, resources with their widths, stops,
  turn permissions and the turn angle at each junction, cranes and crane spans), without coordinates;
- the result data from which every number, table of results and data figure of the paper and of its supplementary
  material is generated, with the detailed per-point tables of the scenarios, and the scripts that generate them.

## Contents

```
occupancy/            simulation model: network (derived graphs), tasks, traffic rules (discrete-event simulation),
                      network bound T1 / T1' (bound.py), parameters and scenarios (params.py), map -> graph (mapgraph.py)
bounds/               system bound (system.py), route-free bound with Lagrangian certificate (route_free.py)
exact/                constraint programme (CP-SAT) of small instances and the independent schedule checker (check.py)
experiments/          batch runner and searches: runner.py (simulated points and sampled bounds of a stored configuration),
                      rerun.py (rerun stored points), required_fleet.py (required-fleet search K_req), stable_fleet*.py
                      (30-day fleets without growing backlog K_stable, criterion in daily.py);
                      demand mappings: demand_mapping.py, demand_mapping_var.py, conserved_mapping.py (the main case);
                      bounds: capacity_bounds.py, mapping_bounds.py, conserved_bounds*.py, route_free_mappings.py,
                      packing_rows.py, platform_size.py, chain_load.py (chain load of the screen);
                      traffic rules beyond occupancy/: stochastic.py (random times, late rebooking), ordered.py,
                      ordered_proj.py, ordered_check.py (execution in booked order), reverse_cover.py (reversing),
                      tail_booking.py, tail_release.py, tail_runs.py (release at tail clearance), width_ban*.py (ban on
                      over-wide blocks), dock_spots.py (off-road bays at the dock), ablation*.py (ablation of the
                      occupancy representation);
                      sub-networks and the whole network: subnet_exact.py, subnet_bounds.py, subnet_gap.py, corridor.py,
                      retime.py, offline_dispatch.py; the screen: screen_conserved.py, screen_runs.py; mechanisms.py;
                      dock_mouth.py (synthetic dock mouth)
data/networks/        derived graphs: okpo/rev1833, okpo/rev1825, yantai/rev396 (field descriptions in its README.md)
results/              result data by use in the paper, runs/ (configurations and per-point tables of the batches),
                      MANIFEST.csv (SHA-256, source commit and paper location of every file), README.md (layout)
paper/                reproduce.py, scripts/ (make_numbers.py, make_figures.py, make_highlights.py, src.py), highlights.tex
tests/                pytest suite of the model, the bounds and the experiment code, on the released data
RECOMPUTE.md          which script and which configuration recompute each class of results from the derived graphs
requirements.txt      pinned package versions (Python 3.13.5, pip)
environment.yml       the same versions as conda builds (the environment of the paper's figures)
LICENSE, LICENSE-data MIT (code), CC BY 4.0 (data)
CITATION.cff          how to cite
```

## Installation

Python 3.13.5 (64-bit); the paper's results were computed on Windows 11.

```
python -m venv .venv
.venv\Scripts\activate            (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python -m pytest tests
```

The pinned pip packages reproduce the numbers, tables and simulation results byte for byte. The figures of the paper were
drawn with the conda builds of the same versions (Anaconda, matplotlib linked to FreeType 2.13.3; `environment.yml`); the
matplotlib wheel on PyPI bundles FreeType 2.6.1, which changes glyph metrics, so with pip the figures draw the same numbers
but their PDF bytes differ. For byte-identical figures use `conda env create -f environment.yml`. OR-Tools (CP-SAT) is needed
only for the sub-network instances and the whole-network retiming.

## Reproducing the paper

One command, from the repository root:

```
python paper/reproduce.py
```

It writes `paper/numbers.tex` (one LaTeX macro per number of the text and of the supplementary material),
`paper/tables/*.tex`, `paper/highlights.txt` and `paper/figures/*.pdf` from `results/` in a few minutes; nothing is
simulated. The scripts check their readings with assertions throughout (for example that every simulated point lies below
its bound, and that the figures draw the numbers the text quotes). The PDFs carry no time stamps, so they are
byte-identical across runs in the same environment (the paper's: `environment.yml`, with the fonts Times New Roman and STIX;
on systems without Times New Roman matplotlib substitutes a font).

Not drawn here: Figure 3 (the two yards) draws the road centre lines of the yard maps, which are not released; the binding
elements it marks are in `results/main_case/binding.csv` (main case) and `results/bounds/*/prices.csv` (the other
scenarios). Figures 1, 2, 8(a) and 10 are diagrams and schematics, not data figures.

| Paper | Generated file | Script | Data (under `results/`) |
|---|---|---|---|
| Numbers in the text | `numbers.tex` | `make_numbers.py` | all files; `MANIFEST.csv` gives the location of each |
| Highlights | `highlights.txt` | `make_highlights.py` | `numbers.tex` |
| Table 7 (bounds of five scenarios) | `tables/tab_bounds.tex` | `make_numbers.py` | `main_case/bounds.csv`, `main_case/saturation.csv`, `intra_area/sampled.csv`, `bounds/okpo/`, `bounds/yantai/` (`system.csv`, `route_free.csv`, `readout.csv`), `bounds/route_free_main_case.csv` |
| Table 8 (ablation, main case) | `tables/tab_ablation.tex` | `make_numbers.py` | `ablation/bounds_main_case.csv`, `ablation/summary_main_case.json` |
| Table 9 (fleets K_req and K_stable) | `tables/tab_fleethead.tex` | `make_numbers.py` | `main_case/fleet.csv`, `fleet_rules.csv`, `fleet_30min.csv`; `intra_area/fleet.csv`, `fleet_30min.csv`, `stable.csv`; `fleet/crane_fleet.csv`, `runs/*_crane_fleet/demand.csv` |
| Table B.1 (task-flow rows) | `tables/tab_flows.tex` | `make_numbers.py` | `occupancy/params.py` (`FLOW`) |
| Table S2 (fleets, every move scaled) | `tables/tab_fleethead_prop.tex` | `make_numbers.py` | `fleet/required_fleet.csv`, `fleet_8day.csv`, `kstar_marks.csv`, `fleet_30min.csv`; `intra_area/stable.csv`; `saturation/kstar.csv` |
| Table S3 (ablation, every move scaled) | `tables/tab_ablation_prop.tex` | `make_numbers.py` | `ablation/bounds_proportional.csv`, `ablation/summary_proportional.json` |
| Table S4 (sub-networks) | `tables/tab_subnet.tex` | `make_numbers.py` | `subnet/points.csv`, `subnet/corridor.csv` |
| Table S5 (whole network) | `tables/tab_whole.tex` | `make_numbers.py` | `retiming/main_case.csv` |
| Table S6 (random times, booked order) | `tables/tab_stoch.tex` | `make_numbers.py` | `stochastic/plateau.csv`, `curves.csv`, `ordered_plateau.csv` |
| Table S7 (off-road bays) | `tables/tab_dockbay.tex` | `make_numbers.py` | `dock/bays.csv` |
| Table S10 (hand-computed instances) | `numbers.tex` (macros `Hand*`) | `make_numbers.py` | `hand/instances.json`, the parameters of `runs/okpo_rev1825_safe_sectioned/config.json` |
| Fig. 3 (the two yards) | not generated | - | geometry not released; `main_case/binding.csv`, `bounds/*/prices.csv` |
| Fig. 4 (saturation curves, main case) | `figures/fig_flag.pdf` | `make_figures.py` | `main_case/saturation_curves.csv`, `main_case/saturation.csv`, `saturation/kstar.csv` |
| Fig. 5 (capacity interval) | `figures/fig_interval.pdf` | `make_figures.py` | `main_case/interval.csv`, `bounds/interval_route_free.csv`, `interval/interval.csv`, `interval/crane_interval.csv`, `interval/crane_16day_plateau.csv`, `saturation/rules.csv`, `bounds/*/route_free.csv` |
| Fig. 6 (the screen in two levels) | `figures/fig_screen.pdf` | `make_figures.py` | `screen/series.csv`, `screen/cross.csv`, `screen/series_main_case.csv`, `screen/cross_main_case.csv` |
| Fig. 7 (dock-mouth thresholds) | `figures/fig_dock.pdf` | `make_figures.py` | `dock/thresholds.csv`; per-point tables in `runs/*_dock_*` (see `results/README.md`) |
| Fig. 8(b, c) (synthetic dock mouth) | `figures/fig_dockmech_bc.pdf` | `make_figures.py` | `synthetic_dock/rho_star.csv`, `synthetic_dock/config.json`, `experiments/dock_mouth.py` |
| Fig. 9 (anchors) | `figures/fig_anchors.pdf` | `make_figures.py` | `anchors/handling_time.csv`, `anchors/per_vehicle_rates.csv` |
| Fig. S1 (moves within areas) | `figures/fig_scan.pdf` | `make_figures.py` | `main_case/scan.csv`, `main_case/scan_switch.csv` |
| Fig. S2 (required fleet, every move scaled) | `figures/fig_workzone.pdf` | `make_figures.py` | `fleet/required_fleet.csv`, `fleet/fleet_8day.csv`, `runs/yantai_crane_fleet/demand.csv`, `saturation/kstar.csv` |
| Section 4.1, Appendix B | `numbers.tex` | `make_numbers.py` | `network/`, `main_case/`, `intra_area/`, run configurations (`runs/*/config.json`), `occupancy/params.py`, `fleet/bootstrap.json`, `experiments/daily.py` |
| Sections 4.2-4.3 | `numbers.tex` | `make_numbers.py` | `main_case/`, `bounds/`, `ablation/`, `saturation/`, `interval/`, `retiming/`, `subnet/`, `stochastic/` |
| Section 4.4 | `numbers.tex` | `make_numbers.py` | `main_case/fleet*.csv`, `intra_area/`, `screen/`, `fleet/` |
| Section 4.5 | `numbers.tex` | `make_numbers.py` | `dock/`, `synthetic_dock/`, `runs/synthetic_dock_waiting/` |
| Section 4.6 (verification and validation) | `numbers.tex` | `make_numbers.py` | `runs/*/check.json`, `runs/*/points.csv` (column `T1_tight_mix`), the `check*.csv` files of `main_case/`, `intra_area/`, `stochastic/`, `saturation/okpo_curves_check.json`, `saturation/refusal_summary.json`, `fleet/*summary.json`; `anchors/` |
| Appendix A | `numbers.tex` | `make_numbers.py` | `tail/`, `hand/`, `network/short_roads*.csv`, `saturation/teleport*.csv`, `bounds/sensitivity.csv` |
| Supplementary S1-S2, S4 | `numbers.tex` | `make_numbers.py` | `width_ban/`, `bounds/`, `screen/`, `stochastic/`, `subnet/`, `retiming/`, `ablation/`, `dock/`, `hand/` |

Every scenario of the paper has its detailed table: the per-point tables (`points.csv`, one row per fleet, scenario, rule
and seed) of the saturation curves, the required-fleet searches, the 30-day runs (with their day-by-day table `daily.csv`),
the random-time and booked-order runs, the dock-mouth runs, the tail-booking runs and the screens under `results/runs/`, the
synthetic dock mouth under `results/synthetic_dock/`, and the per-cell tables (plateaus, required fleets, thresholds,
bounds, schedules) in the other directories; for the few batches whose per-point table was not kept (named in
`results/README.md`) the configuration and the per-cell table are released. `results/README.md` lists the runs and how the
tables are laid out.

## Recomputing the results from the derived graphs

`RECOMPUTE.md` names, for each class of results, the script, the stored configuration and the arguments that recompute it
from the derived graphs and the parameters. Three common cases:

The bound tables of a yard with every move scaled with the flows (Table 7 and Supplementary S1.2) come from
`experiments/capacity_bounds.py` with the configuration stored with them:

```
python experiments/capacity_bounds.py results/bounds/okpo/config.json --out out/bounds_okpo --scens main --items system route_free seeds
```

writes `system.csv` (system bound and the range of the ten sampled network bounds), `route_free.csv` (route-free bound
T1'', route-fixed value R, route margin), `seeds.csv` (network bound and route-free bound of each task sample), and with
all items also `convergence.csv`, `readout.csv`, `prices.csv` and `attribution.csv`. Okpo's configuration runs on network
revision 1825; the system bound of the main mix is the same on revision 1833 (`results/bounds/system_by_granularity.json`,
which lists the system bounds of both yards with whole and sectioned dock roads). The route-free programme of Okpo's
crane-paced scenario takes minutes. Single values can be computed directly:

```python
import json
from experiments import capacity_bounds as B          # or: sys.path.insert(0, "experiments"); import capacity_bounds as B
cfg = json.load(open("results/bounds/yantai/config.json"))
print(B.item_system(cfg, "main")["T1"])               # system bound, tasks per 16-h day
```

A whole batch of simulated points is run with `python experiments/runner.py results/runs/<run>/config.json --out out/<run>`,
a required-fleet search with `python experiments/required_fleet.py results/runs/<run>/config.json --out out/<run>`, and the
synthetic dock mouth with `python experiments/dock_mouth.py --out out/dock`. Processes are pinned to the cores other than
core 0; a point takes from about a second (small fleets, four days) to a few minutes (large fleets, 16 or 30 days). The
option `--slots DIR` shares a machine with other jobs through a directory of CPU slots; without it an ordinary process pool
is used.

The network bound of a task sample, the fleet bound and the dual prices are in `occupancy/bound.py`; the per-point check of
a simulated point (`T1_tight_mix`) is `experiments/runner.t1_window`.

## Rerunning stored points

Every simulated point is determined by its configuration, network and seed. Rerunning a stored point with this code at its
fixed seed reproduces its row of `points.csv` in every column (except the run time `secs`), floats to the last bit. One
label differs for runs stored by an earlier version of the code: their most-occupied-resource columns (`util_top1_name`,
`busy_top1_name`) name junctions without the bracketed tag `[building·entrance, short id]` that the code now appends, and
`rerun.py` compares these two columns up to the tag (and says so in its output):

```
python experiments/rerun.py results/runs/okpo_rules --pick scen=main,model=reserve,K=30,seed=101
python experiments/rerun.py results/runs/okpo_fleet_sectioned --sample 3 --max-k 40
python experiments/rerun.py results/runs/okpo_main_saturation --sample 3 --max-k 40
python experiments/rerun.py results/synthetic_dock --pick rule=seg,W=9.0,rho=0.9,K=24,seed=101
python experiments/rerun.py results/runs/okpo_dock_8day --pick scen=rho8_8d,model=reserve,K=20,seed=101
```

The point tables of the yard dock-mouth runs are larger than 50 MB and are split into files of at most 45 MB with the same
columns: by granularity (`points_whole.csv`, whole dock roads; `points_stops.csv`, dock roads sectioned at the stops) and,
for Okpo's 16-day runs, the sectioned part by fleet (`points_stops_K40-100.csv`, `points_stops_K150.csv`). `rerun.py` reads
all `points*.csv` of a run together; to get one table, concatenate them and keep the header line once, for example

```python
import csv, glob
rows = [r for f in sorted(glob.glob("results/runs/okpo_dock_16day/points*.csv")) for r in csv.DictReader(open(f))]
```

## Data sources and what is not released

The road networks were digitised from Google Earth imagery (Google Earth Pro; Okpo dated 15 November 2023: Google, Airbus,
TMap Mobility; Yantai dated 11 March 2025: Google, Airbus). Neither the imagery nor the centre-line geometry traced from it is
released: the networks are released as derived graphs only (`data/networks/`), which hold everything the model and the
bounds use. The passable widths of `results/width_ban/clear_width.json` were read on the same imagery. Map lines delineate
study areas and do not necessarily depict accepted national boundaries. The other inputs (equipment data, handling times,
demand, erection volumes, platform sizes) come from published sources listed in the paper (Appendix B); the parameters are
in `occupancy/params.py` and in every stored configuration.

Okpo's network is released in two revisions. Revision 1833, with the three floating docks, is the paper's. Revision 1825 is
the network of the bound tables with every move scaled and of the runs made before the floating docks were digitised: it was
rebuilt from the earlier map (revision 1824, whose road network it shares) with the crane lists and gantry-crane spans of the
two dry docks taken from revision 1833, where they are unchanged. Built this way it equals, element by element, the network
on which those runs and bounds were computed, and the system, network and route-free bounds recomputed from it equal the
stored ones.

## License and citation

Code: MIT License (`LICENSE`). Data (the derived road networks in `data/networks/` and the result data in `results/`):
Creative Commons Attribution 4.0 International, full legal text in `LICENSE-data`. Copyright 2026 the four authors.
Please cite the paper (`CITATION.cff`).

Repository: https://github.com/NCR7777/shipyard-road-occupancy. On acceptance of the paper the accompanying version will be
archived in Zenodo; its DOI will be given here.

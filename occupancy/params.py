"""Model parameters, scenarios and the yard registry.

Every parameter is a configuration item; the values and their sources (equipment specifications, literature, public company
data, scenario settings) are listed in the paper, Appendix B (parameter table). Each batch of results stores the full
parameter set it ran with (results/**/config.json), so a stored point can be rerun from its config alone.
"""

# Daily demand: annual steel throughput (t) / mean block weight (t) x transfers per block / working days. Defaults: blocks of
# 100 t, 10 transfers per block, 300 working days (paper, Appendix B.3).
def daily_tasks(steel_t, block_t=100.0, transfers=10.0, work_days=300.0):
    """Daily transport tasks (blocks/day) from the annual steel throughput."""
    return steel_t / block_t * transfers / work_days


# Annual steel throughput (t): Yantai 68,512 t (cleaner-production audit notice, 2023); Okpo 1,018,067 t (mean of 2022-2025,
# Hanwha Ocean sustainability report 2025, p. 110), which gives 339 blocks/day
STEEL_T = {"yantai": 68512, "yupu": 1018067}
YUPU_DEMAND = round(daily_tasks(STEEL_T["yupu"]))      # 339
# Yantai: daily_tasks(68,512) = 22.8 -> 23 blocks/day; the peak is twice the regular level, as for Okpo
YANTAI_DEMAND = {"regular": round(daily_tasks(STEEL_T["yantai"])), "peak": 2 * round(daily_tasks(STEEL_T["yantai"])), "saturated": None}
# Plant-level daily demand: regular 300, 450 and 600 blocks/day (Yim et al. 2008: more than 300 blocks a day; Shen et al. 2018:
# about 600 blocks a day), peaks twice the regular levels; working days of 16 h and 24 h
PLANT_REGULAR = (300, 450, 600)
PLANT_LEVELS = tuple(sorted(set(PLANT_REGULAR) | {2 * x for x in PLANT_REGULAR}))     # 300, 450, 600, 900, 1200
DAY_HOURS = (16, 24)

# Yards: key -> derived graph (data/networks/<yard>/<revision>), roles = {role in FLOW: class in the graph} (empty: by class
# name) and the daily demand of the yard where it differs from DEMAND. The yard key "yupu" is Okpo (Chinese name Yupu), the
# yard of Hanwha Ocean on Geoje; "yantai" is the CIMC Raffles yard at Yantai. base: the key of the same yard for the crane
# pacing, rho levels and peak demand.
#   yupu        Okpo, revision 1833 (the network of the paper, with three floating docks and their spur roads);
#   yupu_r1825  Okpo, revision 1825 (before the floating docks were added; runs stored from that revision);
#   yantai      Yantai, revision 396 (the road network of revision 395 plus the crane data).
YARDS = {
    "yupu": dict(name="Okpo", network="okpo/rev1833", roles={}),
    "yupu_r1825": dict(name="Okpo", network="okpo/rev1825", roles={}, base="yupu"),
    "yantai": dict(name="Yantai", network="yantai/rev396", roles={}, demand=YANTAI_DEMAND),
}


def base(yard):
    """Yard key -> the current key of the same yard (yupu_r1825 -> yupu); crane pacing and rho levels are taken from it."""
    return YARDS[yard].get("base", yard)


DEFAULTS = dict(
    v_empty=12.0 / 3.6,        # m/s
    v_loaded=6.0 / 3.6,        # m/s
    plat_len=16.25,            # m, length of an empty vehicle; a loaded vehicle is max(block length, platform length) long
    plat_w=5.5,                # m, width of an empty vehicle w
    lat_clear=1.0,             # m, lateral clearance c
    long_clear=5.0,            # m, longitudinal clearance (as a time: clearance / speed)
    turn_sweep_s=30.0,         # s, extra time a junction is swept when a vehicle turns by at least turn_deg
    turn_deg=60.0,
    load_s=600.0, unload_s=600.0,
    block_w=(5, 15), block_l=(10, 30),
    width_delta=0.0,           # m, added to every mapped road width (sensitivity +-1.5)
    unrestricted_w=31.0,       # m, equivalent width of a road without width limit = 2 B_max + c: two loaded vehicles pass
    empty_rule="width",        # width: empty vehicles pass by road width; exclusive: empty vehicles hold every road alone; free: always pass
    day_s=16 * 3600.0, days=4, warmup_days=1, batches=8,
    deadlock_recovery_s=900.0,  # stepwise requests: time to clear a deadlock (one vehicle backs onto open ground); None = no recovery
    near_deadlock_s=1800.0,     # a request waiting longer than this counts as a near-deadlock
    frontage_gap=8.0, frontage_min_len=10.0, frontage_merge=3.0,   # frontage rule of docks
    dock_stops=None,           # roadside stops per dock: None = one per frontage stretch; n = only the n longest
    reserve_jit=True,          # full-path reservation: once the loaded run is booked, the empty run is delayed to arrive just in time
    flow_share=None,           # {flow: share}: erection-heavy mix sets the share of P6, the other flows scale; None = base weights
    erect_batches=None,        # P6 tasks are released only in these batches (from 0); None = spread like the other tasks
    od_sampling="entity",      # entity: entities of a role equally likely, stops of an entity equally likely (main model);
                               # stop: all stops of a role equally likely (destination not in the origin's entity)
    # Sectioning of long dock roads (Net._section): None = one resource per road; "stops" = roads longer than section_min_m
    # with roadside stops get section boundaries section_gap_m before and after each stop; with block_m each section is cut
    # further into blocks of length <= block_m
    section=None, section_min_m=200.0, section_gap_m=25.0, block_m=None,
)

# Daily tasks per 16-h day; None = an endless supply of tasks (saturation). 97 / 194 are a four-vehicle subset of a large
# yard (Yim et al. 2008), used only for comparison; plant-level demand: PLANT_LEVELS
DEMAND = {"regular": 97, "peak": 194, "saturated": None}

# Flows of blocks and their weights (mean number of times a block makes the move)
FLOW = [
    ("P1", "assembly_workshop", "outfitting_workshop", 0.5),
    ("P2", "assembly_workshop", "paint_workshop", 0.5),
    ("P3", "outfitting_workshop", "paint_workshop", 0.5),
    ("P4", "paint_workshop", "yard|buffer", 0.7),
    ("P5", "yard|buffer", "assembly_yard", 0.7),
    ("P6", "assembly_yard", "dry_dock", 0.2),
    ("P7", "assembly_workshop", "assembly_yard", 0.05),
    ("P8", "paint_workshop", "assembly_yard", 0.3),
]
START_CLASSES = ("assembly_workshop", "paint_workshop", "outfitting_workshop")
STOP_MODE = {"yard": "pull_in", "assembly_yard": "pull_in", "buffer": "on_road", "dry_dock": "on_road"}

SCENARIOS = {
    "main": {},
    "w+1.5": {"width_delta": 1.5},
    "w-1.5": {"width_delta": -1.5},
    "empty_excl": {"empty_rule": "exclusive"},
    "empty_free": {"empty_rule": "free"},
    "jit_off": {"reserve_jit": False},                                   # earlier reservation heuristic (earliest empty departure)
    "dock1": {"dock_stops": 1},          # one roadside stop per dock (the longest frontage stretch)
}
# Erection-heavy mix: P6 takes 15 % of the tasks and is released in the day-shift batches 1-4 (8:00-16:00); daily demand 194
ERECTION = {"flow_share": {"P6": 0.15}, "erect_batches": [0, 1, 2, 3]}
SCENARIOS["erection"] = ERECTION
SCENARIOS["erection_even"] = {"flow_share": {"P6": 0.15}}               # heavier share only, not concentrated
# Sensitivity: sampling of origins and destinations by stop (with widths), Yantai widths +-3 m, P6 15 % with one stop per
# dock (bounds only), eight-day steady-state checks
SCENARIOS.update({
    "od_stop": {"od_sampling": "stop"},
    "od_stop_w+1.5": {"od_sampling": "stop", "width_delta": 1.5},
    "od_stop_w-1.5": {"od_sampling": "stop", "width_delta": -1.5},
    "w+3": {"width_delta": 3.0},
    "w-3": {"width_delta": -3.0},
    "erection_dock1": dict(ERECTION, dock_stops=1),
    "main_8d": {"days": 8},
    "empty_free_8d": {"empty_rule": "free", "days": 8},
})
# Crane-paced erection (main erection scenario): each erection holds a crane for 10 min (hooking, lifting) + 180 min
# (positioning, releasing) (Zhang 2025). P6 is spread over the docks in proportion to their cranes and released at equal
# intervals within a dock (occupancy.tasks.crane_tasks); the other flows do not draw P6 (flow_share P6 = 0). Cranes per dock
# are those of the dock that count for erection (crane_n = "map"); a P6 whose assembly area lies inside the gantry span of
# its dock (share of the area >= span_min) is lifted by the gantry crane directly, without a vehicle or a road, and still holds
# the crane for crane_cycle_s; the lift counts as done crane_hook_s after it starts. span_min = None: every P6 by road.
CRANE = {"crane_cycle_s": (10 + 180) * 60.0, "crane_hook_s": 10 * 60.0, "flow_share": {"P6": 0.0}, "crane_n": "map", "span_min": 0.5}
SCENARIOS["crane"] = CRANE
# Erection units on a peak day: p6_peak = 2 x ships a year x erection units per ship / 300 working days (peak = twice the
# regular level, as for demand).
#   Okpo: the LNG carriers of docks 1 and 2, 20 in 2025 (derived from the order book); 89-110 erection units per ship
#     (Roh and Lee 2006: 89 for a 300K VLCC; a Daewoo patent: usually 90-110), main value 100.
#   Yantai: 685 blocks a year (68,512 t / 100 t) x 89/200 (89 erection units of about 200 blocks of a 300K VLCC).
P6_UNITS = (89, 100, 110)
CRANE_YARD = {
    "yupu": dict(p6_peak=2 * 20 * P6_UNITS[1] / 300, p6_range=(2 * 20 * P6_UNITS[0] / 300, 2 * 20 * P6_UNITS[2] / 300)),
    "yantai": dict(p6_peak=round(2 * 68512 / 100 * 89 / 200 / 300)),
}
DOCK_CLASS = "dry_dock"


def scenario(name, **over):
    p = dict(DEFAULTS)
    p.update(SCENARIOS[name])
    p.update(over)
    return p

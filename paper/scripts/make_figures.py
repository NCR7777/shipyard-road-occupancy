"""Draw the data figures of the paper, figures/*.pdf, and write figures/palette.json, from the result data of the
repository (results/, paths relative to the repository root, read through src.py; the src argument of each save call
names the sources, and the PDFs carry a neutral Subject). PNG previews go to build/.

Figures of this paper, in their order:
  Figure 3          the two yards with the binding elements of the system bound: not drawn here; it needs the centre-line
                    geometry of the yard maps, which is not released (the binding elements are in
                    results/main_case/binding.csv for the main case and results/bounds/*/prices.csv for the other
                    scenarios, resources with a positive dual price)
  fig_flag.pdf      Figure 4, fleet size vs throughput in the main case (erection conserved), three rules + reference rule,
                    sampled and system bounds, seed range shaded; results/main_case/{saturation_curves.csv, saturation.csv}
  fig_interval.pdf  Figure 5, capacity interval [full-path reservation, system bound] as a share of each row's system
                    bound, with both ends at 30-min handling on the same scale, moves/day and the share in two columns;
                    main case from results/main_case/interval.csv, its route-free bound from
                    results/bounds/interval_route_free.csv; erection-heavy mix from results/interval/interval.csv and
                    results/saturation/rules.csv; crane-paced erection from results/interval/crane_interval.csv, Okpo's
                    crane-paced lower end over 16 days from results/interval/crane_16day_plateau.csv; the route-free bound
                    of the other rows from results/bounds/*/route_free.csv; the two-tier lower end of the main case
                    (offline dispatch) from results/interventions/lower_tier.csv and its random-time markers from
                    results/stochastic/main_case_plateau.csv
  fig_screen.pdf    Figure 6, the screen in two levels, full-path reservation against free flow, ten series and the main
                    case on 16-h days: (a) every tested level as a cell against demand over the system bound, classed
                    below the lower end in whole vehicles (same fleet, one more) and from it by the extra fleet with
                    fractional fleets (at most or over 10 %), with the lower end and the range of the 16-h upper ends;
                    (b) the extra fleet against the chain load S_d, main case (thick) and road widths, with each yard's
                    upper end; results/screen/{series.csv, cross.csv}; the main case on 16-h days from
                    results/screen/{series_main_case.csv, cross_main_case.csv}
  fig_dock.pdf      Figure 7, dock-mouth threshold (10 % throughput drop) vs fleet size, whole and sectioned dock roads;
                    results/dock/thresholds.csv (eight days, or 16 where the eight-day runs were not steady)
  fig_dockmech_bc.pdf  Figure 8(b, c), synthetic dock mouth, at two thirds of the text width beside the schematic (a):
                    (b) throughput relative to the lowest utilisation against rho, (c) rho_10 against the fleet;
                    results/synthetic_dock/rho_star.csv; the geometry printed in (a) is checked against
                    experiments/dock_mouth.py
  fig_anchors.pdf   Figure 9, external anchors and the Okpo model at the same scale (moves per vehicle-day against the
                    handling time; 16-h values at the main handling time), main case; results/anchors/
                    {main_case_handling_time.csv, main_case_per_vehicle.csv}
  fig_scan.pdf      Supplementary Figure S1, system bound against the share of moves within an area, erection conserved,
                    with the switch of the binding element; results/main_case/{scan.csv, scan_switch.csv}
  fig_workzone.pdf  Supplementary Figure S2, plant-level daily demand vs required fleet with every move scaled with the
                    flows; results/fleet/required_fleet.csv with one Yantai cell from
                    results/runs/yantai_crane_fleet/demand.csv and Okpo's own-volume cells over eight days from
                    results/fleet/fleet_8day.csv (src.demand); K* from results/saturation/kstar.csv
Figures 1, 2, 8(a) and 10 are diagrams and schematics, not drawn by this code; they use the colours of
figures/palette.json, which scripts/check_palette.py checks.

Style of this paper's figures: drawn at the text width of the CAS single-column layout (164.6 mm) and inserted at
1:1; Times New Roman with STIX mathematics, TrueType embedded (pdf.fonttype 42), 9-pt text and nothing below 8 pt
except mathematical sub- and superscripts (FS_TEXT, FS_NOTE). Colours come only from PALETTE in the style section below, one colour per meaning in every figure, and are
written to figures/palette.json for the schematics; every traffic rule keeps one colour, marker and line style (COL, MK,
MFC, LS), so the rules stay apart in black and white and for colour-blind readers.
Run from paper/ after make_numbers.py (the figures check their readings against numbers.tex):
python scripts/make_figures.py (or, from the repository root, python paper/reproduce.py).
"""
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from src import BOUNDS_DIR, SYN_DOCK, demand, interval, res_name, rows, show

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
MACRO = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\\xspace\}", (ROOT / "numbers.tex").read_text(encoding="utf-8")))
num = lambda name: float(MACRO[name].replace("{,}", "").replace(r"\%", ""))

# ================================================================== style: the only place with colours and sizes
FIGW, MM = 164.6 / 25.4, 1 / 25.4          # text width of the CAS single-column layout (in); one millimetre (in)
# One colour per meaning in every figure of the paper. Five hues and neutrals: blue for full-path reservation, the main
# colour (and, pale, docks); orange for safe release and its margin steps (and, pale, assembly areas); teal for the
# reference rule (and, pale, stockyards and buffers); plum for the two yards; red only for the elements that bind the
# system bound. Neutrals: free flow, the bounds, external anchors, roads, text. The rules differ also in lightness (the
# main colour darkest) and keep their markers and line styles, for black-and-white print and colour-blind readers. The
# cells of the screen (Figure 6a) run from light to dark: neutrals in whole vehicles, the main colour's blues with
# fractional fleets; scripts/check_palette.py checks this under simulated protanopia, deuteranopia and tritanopia. The
# hex values are written to figures/palette.json, which the schematics (Figures 2 and 9a, not drawn by this code) use;
# their resource
# classes (res_*, numbered 1-6 in Figure 2) take three further hues, pale for their areas, and three of them share the
# value of road_major, stop and crane, which draw the same things.
PALETTE = {   # role: (hex, use, figures)
    "rule_reserve": ("#1F5A99", "full-path reservation, the main colour: solid line, filled circles", "4-6, 8-10"),
    "rule_safe": ("#E07A1F", "safe release: solid line, filled squares", "4-6, 8-10"),
    "rule_reference": ("#2A9285", "reference rule: dotted line, open triangles", "4-6, 8, 10"),
    "rule_free": ("#B0B0B0", "free flow: dashed line, open circles", "4, 6, 10"),
    "margin_0": ("#7A3A0C", "safe release on a whole dock road, no empty vehicle fits beside a standing loaded one", "9"),
    "margin_1": ("#B0560F", "as margin_0, one empty vehicle fits", "9"),
    "margin_2": ("#EBA563", "as margin_0, two empty vehicles fit", "9"),
    "interval_fill": ("#D6E2F1", "capacity interval bar (pale main colour)", "5"),
    "interval_fill_light": ("#EAF0F8", "capacity interval with 30-min handling, thin bar (paler interval_fill)", "5"),
    "offline_dispatch": ("#C2185B", "offline dispatch of full-path reservation's tasks, all tasks known: range over "
                                    "seeds, the second lower marker", "5"),
    "booked_random": ("#8C6D1F", "full-path reservation executed in booked order under random times: down triangles, "
                                 "filled for CV 0.3, open for CV 0.6, the third lower marker", "5"),
    "extra_fleet_fill": ("#E6EEF8", "extra fleet of full-path reservation over free flow, between the two lines "
                                    "(paler main colour)", "6"),
    "cell_same": ("#E8E6E0", "screen cell below the lower end, in whole vehicles: same fleet as free flow", "6"),
    "cell_frac_low": ("#B3C8E4", "screen cell from the lower end, fractional fleets: at most 10 % more", "6"),
    "cell_one_more": ("#96928A", "screen cell below the lower end, in whole vehicles: one vehicle more", "6"),
    "cell_frac_high": ("#1F3F73", "screen cell from the lower end, fractional fleets: over 10 % more", "6"),
    "yard_okpo": ("#6B3A7D", "Okpo as a series: diamonds", "7"),
    "yard_yantai": ("#B48AC4", "Yantai as a series: down triangles", "7"),
    "bound_system": ("#1A1A1A", "system bound: solid line; upper end of the capacity interval", "4, 5"),
    "bound_network": ("#5C5C5C", "network bound, mean of the task samples: dash-dot line", "4"),
    "bound_range": ("#DCE3EB", "network bound, range of the task samples: band", "4"),
    "anchor": ("#1A1A1A", "external anchors (published rates and fleets): stars, solid or dashed lines", "6, 10"),
    "binding": ("#D7263D", "elements that bind the system bound, and only these; in Figure 2, a vehicle that holds a road "
                           "alone", "2, 3"),
    "road": ("#ABA79F", "road narrower than 18 m on a map: line width = road width; access links dotted; kerb lines in "
                        "the schematics", "2, 3, 9a"),
    "road_major": ("#8E8A82", "edge of a road 18 m or wider on a map (white inside)", "3"),
    "road_surface": ("#E3E0D9", "road drawn as an area in a schematic", "2, 9a"),
    "zone_building": ("#D3CCC1", "workshops and other buildings", "2, 3"),
    "zone_storage": ("#E2F1EB", "stockyards, buffers and quays", "2, 3, 9a"),
    "zone_assembly": ("#F7E2B8", "assembly areas", "3"),
    "zone_dock": ("#B3CDE8", "docks", "2, 3, 9a"),
    "stop": ("#3A3A3A", "stops (access points, roadside stops)", "2, 3, 9a"),
    "vehicle_loaded": ("#34495E", "loaded transporter", "2, 9a"),
    "block": ("#C9A77C", "block on a loaded transporter", "2, 9a"),
    "block_edge": ("#977D5D", "edge of a block (darker block)", "2, 9a"),
    "vehicle_empty": ("#C3CDD8", "empty transporter (edge: vehicle_loaded)", "2, 9a"),
    "crane": ("#4A4A4A", "crane and its rail", "2, 9a"),
    "res_segment": ("#8E8A82", "resource class 1, road segment: number badge and occupancy bar (the value of road_major)",
                    "2"),
    "res_junction": ("#8C6BA0", "resource class 2, junction: dashed outline, badge, bar", "2"),
    "res_junction_fill": ("#EAE4EE", "junction area (pale res_junction)", "2"),
    "res_entrance": ("#94704C", "resource class 3, workshop entrance (door): door mark, badge, bar", "2"),
    "res_bay": ("#4E9E80", "resource class 4, drive-in bay: outline, badge, bar", "2, 9a"),
    "res_bay_fill": ("#D8EAE3", "drive-in bay area (pale res_bay)", "2, 9a"),
    "res_stop": ("#3A3A3A", "resource class 5, roadside stop: dashed outline, badge, bar (the value of stop)", "2, 9a"),
    "res_stop_fill": ("#C4C4C4", "roadside stop area (pale res_stop)", "2, 9a"),
    "res_crane": ("#4A4A4A", "resource class 6, erection crane: crane, badge (the value of crane)", "2, 9a"),
    "ink": ("#0B0B0B", "text, scale bars, leaders", "all"),
    "ink_2": ("#52514E", "secondary text, axis labels and ticks", "all"),
    "muted": ("#898781", "reference lines: 10 % loss, own volumes, onset of saturation; extension lines in the schematics",
              "2, 6, 7, 9"),
    "axis": ("#A9A8A0", "axis lines", "2, 4-10"),
    "grid": ("#E1E0D9", "grid lines", "2, 4-10"),
    "band": ("#EFEEE8", "reading bands: onset of saturation K*, first crossings of 10 %", "4, 6, 7"),
    "strip": ("#F6F6F4", "top strip for 'not reached' and 'no 10% loss'", "6, 8, 9"),
    "paper": ("#FFFFFF", "background; open markers; label boxes", "all"),
}
C = {k: v[0] for k, v in PALETTE.items()}
COL = {"free": C["rule_free"], "reserve": C["rule_reserve"], "segment_safe": C["rule_safe"], "segment": C["rule_reference"]}
MK = {"free": "o", "reserve": "o", "segment_safe": "s", "segment": "^"}
MFC = {"free": C["paper"], "reserve": None, "segment_safe": None, "segment": C["paper"]}   # None: filled
LS = {"free": "--", "reserve": "-", "segment_safe": "-", "segment": ":"}
LW = {"free": 1.0, "reserve": 1.6, "segment_safe": 1.3, "segment": 1.0}
ORDER = ("free", "reserve", "segment_safe", "segment")                                   # as in the rule table
INK, INK2, MUTED, GRID, BAND, PAPER = C["ink"], C["ink_2"], C["muted"], C["grid"], C["band"], C["paper"]
YCOL = {"yupu": C["yard_okpo"], "yantai": C["yard_yantai"]}
YMK = {"yupu": "D", "yantai": "v"}
MARGIN = {0: C["margin_0"], 1: C["margin_1"], 2: C["margin_2"]}
MS, MS_PT, MS_ANCHOR = 3.8, 5.0, 7.5        # markers on lines, markers alone, external anchors (stars)
LW_BOUND, LW_REF, LW_THIN = 1.2, 0.8, 0.6   # bounds, reference lines, leaders and frames
LABEL = {"free": "Free flow", "reserve": "Full-path reservation", "segment_safe": "Safe release",
         "segment": "Reference rule (teleport recovery)"}
NAME = {r: LABEL[r].replace(" (teleport recovery)", "").lower() for r in LABEL}    # direct labels on the lines
YNAME = {"yupu": "Okpo", "yantai": "Yantai"}
YARDS = ("yupu", "yantai")
# Times New Roman (system font) with STIX mathematics; Times reads smaller than a sans serif at the
# same size, so every size is one step up from the Arial set (8 / 7.5 / 7 pt): 9 pt text and axis titles, 8.5 pt ticks,
# legends and row labels (FS_TEXT), 8 pt notes and direct labels (FS_NOTE), 9.5 pt panel titles
FS_TEXT, FS_NOTE = 8.5, 8
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix",
                     "pdf.fonttype": 42, "ps.fonttype": 42,
                     "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": FS_TEXT,
                     "ytick.labelsize": FS_TEXT, "legend.fontsize": FS_TEXT, "axes.titlelocation": "left",
                     "axes.edgecolor": C["axis"], "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": PAPER, "axes.facecolor": PAPER,
                     "savefig.facecolor": PAPER, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "lines.linewidth": 1.4})
grp = lambda v: f"{v:,.0f}" if abs(round(v)) >= 10000 else f"{v:.0f}"   # AMM: separator from five digits on, as fmt
THOUSANDS = FuncFormatter(lambda v, _: grp(v))
(OUT / "palette.json").write_text(json.dumps(
    {"about": "Colours of the first paper's figures, one per meaning; written by scripts/make_figures.py (PALETTE), "
              "checked by scripts/check_palette.py. Figures: numbers in the paper; 2 and 9a are the schematics drawn "
              "in figure-studio with these values.",
     "font": {"family": "Times New Roman", "maths": "STIX", "size_pt": 9, "min_pt": 8},
     "colours": {k: {"hex": h, "use": u, "figures": f} for k, (h, u, f) in PALETTE.items()}},
    ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def save(fig, name, src, tight=False, width=FIGW):
    """Write the PDF (and a PNG preview in build/); this paper's figures keep the text width (or the given share of it,
    for panels set side by side), so they are inserted at 1:1. src records the inputs here in the code; the PDF carries a
    neutral Subject."""
    assert tight or abs(fig.get_figwidth() - width) < 1e-9, name
    kw = {"bbox_inches": "tight"} if tight else {}
    # no time stamps, so the same inputs give the same bytes
    fig.savefig(OUT / name, metadata={"Subject": "generated by make_figures.py from the result files",
                                      "Creator": "scripts/make_figures.py", "CreationDate": None, "ModDate": None}, **kw)
    (ROOT / "build").mkdir(exist_ok=True)
    fig.savefig(ROOT / "build" / name.replace(".pdf", ".png"), dpi=150, **kw)   # preview only
    plt.close(fig)


def grid(ax):
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def rk(r, **kw):
    """Colour, marker and line style of traffic rule r, the same in every figure."""
    d = dict(color=COL[r], linestyle=LS[r], linewidth=LW[r], marker=MK[r], markersize=MS,
             markerfacecolor=MFC[r] or COL[r], markeredgecolor=COL[r], markeredgewidth=0.9)
    d.update(kw)
    return d


def rule_handle(r, **kw):
    return Line2D([], [], label=kw.pop("label", LABEL[r]), **rk(r, **kw))


def break_marks(ax, y0, y1):
    """Two short slashes on the left spine between y0 and y1 (data units): the axis is broken there."""
    t = ax.get_yaxis_transform()
    for yy in (y0, y1):
        ax.plot([-0.012, 0.012], [yy - 0.006, yy + 0.006], transform=t, color=INK2, lw=0.8, clip_on=False)


# onset of saturation at 20 % (kstar.csv; Okpo on the curve of rev 1833), as interval (lo, hi)
KS = {r["curve"]: (int(r["lo"]), int(r["hi"])) for r in rows("results/saturation/kstar.csv")
      if r["curve"] in YARDS and float(r["theta"]) == 0.2}
assert all(f"{KS[y][0]}" == MACRO[{"yupu": "Yu", "yantai": "Ya"}[y] + "KStarLo"] for y in YARDS)
SYS = {y: {r["scen"]: r for r in rows(BOUNDS_DIR[y] + "/system.csv")} for y in YARDS}


# ================================================================== Figure 3 (the two yards): not drawn here
# It draws the road centre lines of the yard maps, which are not released (only the derived graphs in
# data/networks are); the binding elements it marks are in results/bounds/*/prices.csv.

# ================================================================== Figure 4: saturation curves (main case)
# erection conserved: seed means with seed minimum and maximum per K (saturation_curves.csv), system bound, sampled
# bounds of the ten seeds and the onset of saturation (saturation.csv)
CUR61 = defaultdict(dict)                       # (yard, model) -> K -> row
for r in rows("results/main_case/saturation_curves.csv"):
    CUR61[(r["yard"], r["model"])][int(r["K"])] = r
SAT61 = {(r["yard"], r["model"]): r for r in rows("results/main_case/saturation.csv") if r["map"] == "conserved"}
KS61 = {y: (int(SAT61[(y, "reserve")]["K_star_20_lo"]), int(SAT61[(y, "reserve")]["K_star_20_hi"])) for y in YARDS}
assert all(grp(float(SAT61[(y, "reserve")]["lam_S"])) == MACRO[{"yupu": "Yu", "yantai": "Ya"}[y] + "SysCons"] for y in YARDS)
BIND_MAIN = {"yupu": "through corridor", "yantai": "paint-shop doors"}    # Table 7, main case


def every(ks, at=(20, 55, 90)):
    return sorted({min(range(len(ks)), key=lambda i: abs(ks[i] - a)) for a in at})


def end_label(ax, x, y, text, color, dy, **kw):
    """Name a line at its end, in the line's colour (free flow's pale grey would not read: secondary ink)."""
    ax.text(x, y + dy, text, fontsize=FS_NOTE, color=INK2 if color == COL["free"] else color, **kw)


fig, axes = plt.subplots(1, 2, figsize=(FIGW, 74 * MM), layout="constrained")
for i, (ax, y) in enumerate(zip(axes, YARDS)):
    s61 = SAT61[(y, "reserve")]
    sysb = float(s61["lam_S"])
    lo, hi = float(s61["sampled_lo"]), float(s61["sampled_hi"])
    mean_nb = float(s61["sampled"])
    k0, k1 = KS61[y]
    ax.axvspan(k0 - 1.5, k1 + 1.5, color=BAND, linewidth=0, zorder=0)
    ax.axhspan(lo, hi, color=C["bound_range"], linewidth=0, zorder=0.5, label="Sampled bound, range")
    ax.axhline(mean_nb, color=C["bound_network"], linestyle="-.", linewidth=0.9, label="Sampled bound, mean")
    ax.axhline(sysb, color=C["bound_system"], linewidth=LW_BOUND, label="System bound")
    curves = {}
    for r in ORDER:
        cr = CUR61[(y, r)]
        ks = sorted(cr)
        v = [float(cr[k]["mean"]) for k in ks]
        lo_, hi_ = [float(cr[k]["lo"]) for k in ks], [float(cr[k]["hi"]) for k in ks]
        assert all(cr[k]["n"] == MACRO["Seeds"] for k in ks)
        curves[r] = dict(zip(ks, v))
        if r != "segment":
            ax.fill_between(ks, lo_, hi_, color=COL[r], alpha=0.15, linewidth=0)
        ax.plot(ks, v, label=LABEL[r].replace(" (teleport recovery)", ""), markevery=every(ks), **rk(r))
    # every line named where it ends (no legend), clear of its own line: the two rising rules above their last points,
    # safe release, which declines, below; free flow beside its steep line; the bounds by their labels
    for r, dy, va in (("reserve", 0.03, "bottom"), ("segment", 0.03, "bottom"), ("segment_safe", -0.03, "top")):
        end_label(ax, 148, curves[r][150], NAME[r], COL[r], dy * hi, ha="right", va=va)
    yf = 0.85 * 1.3 * hi
    kf = next(k for k in sorted(curves["free"]) if curves["free"][k] >= yf)
    end_label(ax, kf + 4, yf, NAME["free"], COL["free"], 0, ha="left", va="center")
    # free flow passes the system bound (ringed, with its fleet); K* band and the plateau window are readings of
    # Sections 4.2 and 4.3
    kx = min(k for k, v in curves["free"].items() if v > sysb)
    assert kx == num(("Yu" if y == "yupu" else "Ya") + "ConsFreeCross")
    ax.plot([kx], [sysb], marker="o", markersize=MS_PT + 1, markerfacecolor="none", markeredgecolor=INK,
            markeredgewidth=0.8, linestyle="", zorder=5)
    ax.text(kx + 4, sysb - 0.03 * hi, f"crosses at $K$ = {kx}", fontsize=FS_NOTE, color=INK, ha="left", va="top",
            bbox={"facecolor": PAPER, "edgecolor": "none", "pad": 0.5, "alpha": 0.8})     # over the sampled-bound line
    name = BIND_MAIN[y]
    plat = statistics.mean(curves["reserve"][k] for k in curves["reserve"] if k >= 100)
    assert abs(plat - float(s61["plateau"])) < 1e-6 and grp(plat) == MACRO[{"yupu": "Yu", "yantai": "Ya"}[y] + "ConsResPlateau"]
    lx, lha = (3, "left") if kx > 75 else (148, "right")        # on the side away from the crossing
    ax.text(lx, sysb + 0.03 * hi, f"system bound {grp(sysb)}:\n{name}", fontsize=FS_NOTE, color=C["bound_system"],
            ha=lha, va="bottom", bbox={"facecolor": PAPER, "edgecolor": "none", "pad": 0.5, "alpha": 0.8})
    ax.text(148, lo - 0.015 * hi, "sampled bound, ten samples:\nmean (dash-dot), range (shaded)", fontsize=FS_NOTE, color=C["bound_network"], ha="right",
            va="top")
    ax.text((k0 + k1) / 2, 0.04 * hi, "$K^{*}$", fontsize=FS_TEXT, color=INK2, ha="center", va="bottom")
    yb = 0.97 * 1.25 * hi
    ax.plot([100, 100, 150, 150], [yb - 0.02 * hi, yb, yb, yb - 0.02 * hi], color=INK2, lw=LW_THIN)
    ax.text(125, yb + 0.01 * hi, "plateau", fontsize=FS_NOTE, color=INK2, ha="center", va="bottom")
    ax.set_ylim(0, 1.3 * hi)
    ax.set_xlim(0, 150)
    ax.yaxis.set_major_formatter(THOUSANDS)
    ax.set_title(f"({'ab'[i]}) {YNAME[y]}", color=INK)
    ax.set_xlabel("Fleet size $K$ (vehicles)")
    grid(ax)
axes[0].set_ylabel("Throughput (moves/day)")
save(fig, "fig_flag.pdf", "results/main_case/saturation_curves.csv, saturation.csv (map conserved)")

# ================================================================== Figure 5: capacity interval
I27 = interval()
R27E = {r["case"] + "|" + r["combo"]: float(r["plateau"]) for r in rows("results/saturation/rules.csv")}
CIV = {(r["yard"], r["gran"], r["mix"], r["kind"]): r for r in rows("results/interval/crane_interval.csv")}
# Okpo's crane-paced whole-road lower end over 16 days (crane_16day_plateau.csv; the four-day runs are not steady
# there), Yantai's over four
CP16 = {r["yard"]: r for r in rows("results/interval/crane_16day_plateau.csv")}
assert list(CP16) == ["yupu"] and CP16["yupu"]["gran"] == "whole" and CP16["yupu"]["steady_K"] == "11"
assert abs(float(CP16["yupu"]["plat_4d"]) - float(CIV[("yupu", "whole", "crane", "reserve")]["plat"])) < 1e-9
RFB = {y: {r["scen"]: r for r in rows(BOUNDS_DIR[y] + "/route_free.csv")} for y in YARDS}
# the main case (erection conserved): both ends, the other two rules and both ends with 30-min handling, whole and
# sectioned dock roads (main_case/interval.csv); its route-free bound (bounds/interval_route_free.csv: drawn where it exceeds the
# system bound, Okpo with whole and sectioned roads; at Yantai it equals the system bound)
IV61 = {(r["yard"], r["section"], int(r["load_min"])): r for r in rows("results/main_case/interval.csv")}
IRF = {(r["yard"], r["gran"]): r for r in rows("results/bounds/interval_route_free.csv")}
groups =[("Main case", [("yupu", "main", "whole"), ("yupu", "main", "stops"), ("yantai", "main", "whole"),
                        ("yantai", "main", "stops")]),
          ("Erection-heavy mix", [("yupu", "erection", "whole"), ("yupu", "erection", "stops")]),
          ("Crane-paced erection", [("yupu", "crane", "whole"), ("yantai", "crane", "whole")])]
R5 = []                 # one row each: lower end, upper end, the other two rules, both ends with 30-min handling
for gname, cases in groups:
    for (y, kind, sec) in cases:
        lo3 = up3 = rf = None
        if kind == "crane":
            r_ = CIV[(y, "whole", "crane", "reserve")]
            low = float(CP16[y]["plat_16d"]) if y in CP16 else float(r_["plat"])
            up, safe, ref = float(r_["T1"]), None, None
        elif kind == "main":
            i10, i30 = IV61[(y, sec, 10)], IV61[(y, sec, 30)]
            up, low = float(i10["lam_S"]), float(i10["plat_reserve"])
            safe, ref = float(i10["plat_segment_safe"]), float(i10["plat_segment"])
            lo3, up3 = float(i30["plat_reserve"]), float(i30["lam_S"])
            assert grp(up) == MACRO[{"yupu": "Yu", "yantai": "Ya"}[y] + "SysCons"]
            ir = IRF[(y, sec)]
            assert abs(float(ir["system"]) - up) < 1e-6 and (ir["draw"] == "True") == (float(ir["route_free"]) > up + 0.5)
            rf = float(ir["route_free"]) if ir["draw"] == "True" else None
        else:
            up, low = float(I27[(y, kind, sec)]["T1_sys"]), R27E[f"{y}|{kind}|{sec}|reserve"]
            safe, ref = R27E[f"{y}|{kind}|{sec}|segment_safe"], R27E[f"{y}|{kind}|{sec}|segment"]
        # route-free bound (certificate of bounds/<yard>/route_free.csv, whole roads only): the upper end for rules that
        # may reroute, drawn where it exceeds the system bound (Table 7, route margin)
        if sec == "whole" and kind != "main":
            assert abs(up - float(SYS[y][kind]["T1_system"])) < 0.5
            rf = float(RFB[y][kind]["T1pp"])
            rf = rf if rf > up + 0.5 else None
        R5.append(dict(group=gname, label=f"{YNAME[y]}, {'whole road' if sec == 'whole' else 'sectioned'}", low=low,
                       up=up, safe=safe, ref=ref, lo3=lo3, up3=up3, rf=rf, yard=y, kind=kind, sec=sec))
# two-tier lower end of the main-case whole-road rows (interventions/lower_tier.csv): full-path reservation's plateau, guaranteed
# online (the lower end of the bar), and offline dispatch of the same tasks, reachable when all are known (range over
# seeds); and the plateau of full-path reservation executed in booked order under random times (stochastic/main_case_plateau.csv,
# booked on projected occupancy at mean durations), {yard: (CV 0.3, CV 0.6)} in moves/day.
LT5 = {r["yard"]: r for r in rows("results/interventions/lower_tier.csv")}
PL5 = {(r["yard"], r["level"], r["model"]): float(r["plateau"]) for r in rows("results/stochastic/main_case_plateau.csv")}
LOWER_RANDOM = {y: (PL5[(y, "cv3", "reserve_op3")], PL5[(y, "cv6", "reserve_op3")]) for y in ("yupu", "yantai")}
for r in R5:
    if r["kind"] == "main" and r["sec"] == "whole":
        lt = LT5[r["yard"]]
        assert abs(float(lt["online"]) - r["low"]) < 1e-6 and abs(float(lt["lam_S"]) - r["up"]) < 0.01
        r["offline"] = (float(lt["offline_lo"]), float(lt["offline_hi"]))
        r["random"] = LOWER_RANDOM.get(r["yard"])
# each row as a share of its own system bound with 10-min handling (= 100 %), so the length of a bar is the width of the
# interval; moves/day and the share in two columns on the right; the rows in a label column, grouped by the mix
share = lambda r, v: 100 * v / r["up"]
RF_MK = dict(marker="D", markersize=MS, color=C["bound_system"], markerfacecolor=PAPER, markeredgewidth=0.9,
             linestyle="", zorder=4)
yv, YP5, SEP5, GRP5 = 0.0, [], [], []
for gi, (gname, cases) in enumerate(groups):
    if gi:
        yv -= 0.45
        SEP5.append(yv + 0.72)
    ys = [yv - k for k in range(len(cases))]
    YP5 += ys
    GRP5.append((gname, statistics.mean(ys)))
    yv -= len(cases)
YTOP5 = 0.95                                        # room above the first row for its offline-dispatch marker
fig = plt.figure(figsize=(FIGW, 96 * MM), layout="constrained")   # 4 mm of it for the fifth legend row
gs = fig.add_gridspec(1, 3, width_ratios=[40, 96, 26])
axl = fig.add_subplot(gs[0])
ax = fig.add_subplot(gs[1], sharey=axl)
axt = fig.add_subplot(gs[2], sharey=axl)
GLAB = {"Main case": "Main case", "Erection-heavy mix": "Erection-\nheavy mix", "Crane-paced erection": "Crane-paced\nerection"}
for gname, yc in GRP5:
    axl.text(0, yc, GLAB[gname], fontsize=FS_TEXT, color=INK, ha="left", va="center")
for r, y_ in zip(R5, YP5):
    axl.text(1, y_, r["label"], fontsize=FS_TEXT, color=INK2, ha="right", va="center")
    ax.barh(y_, 100 - share(r, r["low"]), left=share(r, r["low"]), height=0.36, color=C["interval_fill"], linewidth=0)
    ax.plot([100], [y_], marker="|", color=C["bound_system"], markersize=9, markeredgewidth=1.8, linestyle="")
    if r["rf"]:                                     # rules that may reroute: on to the route-free bound
        ax.plot([100, share(r, r["rf"])], [y_, y_], color=C["bound_system"], linewidth=LW_THIN, linestyle=":")
        ax.plot([share(r, r["rf"])], [y_], **RF_MK)
        ax.text(share(r, r["rf"]) + 2.5, y_, grp(r["rf"]), fontsize=FS_NOTE, color=INK2, va="center")
    ax.plot([share(r, r["low"])], [y_], **rk("reserve", linestyle="", markersize=MS_PT, zorder=4))
    if r["safe"] is not None:
        ax.plot([share(r, r["safe"])], [y_], **rk("segment_safe", linestyle=""))
        ax.plot([share(r, r["ref"])], [y_], **rk("segment", linestyle=""))
    if r.get("offline"):                            # offline dispatch, above the bar: a range with end ticks
        yo = y_ + 0.27
        ax.plot([share(r, v) for v in r["offline"]], [yo, yo], color=C["offline_dispatch"], linewidth=1.4,
                solid_capstyle="butt", zorder=4)
        ax.plot([share(r, v) for v in r["offline"]], [yo, yo], marker="|", color=C["offline_dispatch"], markersize=5,
                markeredgewidth=1.2, linestyle="", zorder=4)
    if r.get("random") is not None:                 # random times, executed in booked order, on the offline line
        yo = y_ + 0.27
        for v, face in zip(r["random"], (C["booked_random"], PAPER)):
            ax.plot([share(r, v)], [yo], marker="v", markersize=MS, color=C["booked_random"], markerfacecolor=face,
                    markeredgewidth=0.9, linestyle="", zorder=4)
    if r["lo3"] is not None:                        # 30-min handling, scaled to the same 10-min bound, under the row
        y3 = y_ - 0.32
        ax.barh(y3, share(r, r["up3"]) - share(r, r["lo3"]), left=share(r, r["lo3"]), height=0.13,
                color=C["interval_fill_light"], linewidth=0)
        ax.plot([share(r, r["up3"])], [y3], marker="|", color=C["bound_system"], markersize=4.5, markeredgewidth=1.1,
                linestyle="")
        ax.plot([share(r, r["lo3"])], [y3], **rk("reserve", linestyle="", markersize=MS - 0.8, markerfacecolor=PAPER,
                                                 zorder=4))
    axt.text(0.66, y_, f"{grp(r['low'])}–{grp(r['up'])}", fontsize=FS_TEXT, color=INK2, ha="right", va="center")
    axt.text(1.0, y_, f"{share(r, r['low']):.0f}%", fontsize=FS_TEXT, color=INK, ha="right", va="center")
r3, y3 = next((r, y_ - 0.32) for r, y_ in zip(R5, YP5) if r["lo3"] is not None)
ax.text(share(r3, r3["up3"]) + 1.5, y3, "30-min handling", fontsize=FS_NOTE, color=INK2, va="center")
axt.text(0.66, YTOP5, "moves/day", fontsize=FS_NOTE, color=INK2, ha="right", va="bottom")
axt.text(1.0, YTOP5, "share", fontsize=FS_NOTE, color=INK2, ha="right", va="bottom")
# the reading of the text, without saying what the gap is: a lower end far below the bound, a wide interval; close to it,
# a narrow one
for x0, x1, x2, s, ha in ((1, 15, 16, "wide interval", "left"), (100, 86, 85, "narrow interval", "right")):
    ax.annotate("", xy=(x0, YTOP5 - 0.3), xytext=(x1, YTOP5 - 0.3),
                arrowprops={"arrowstyle": "-|>", "lw": LW_THIN, "color": INK2, "shrinkA": 0, "shrinkB": 0})
    ax.text(x2, YTOP5 - 0.3, s, fontsize=FS_NOTE, color=INK2, ha=ha, va="center")
for a in (axl, ax, axt):
    for s in SEP5:
        a.axhline(s, color=GRID, linewidth=LW_REF, clip_on=False)
axl.set_xlim(0, 1)
axl.set_ylim(yv + 0.45, YTOP5)
axl.axis("off")
axt.set_xlim(0, 1)
axt.axis("off")
ax.set_xlim(0, 160)
ax.set_xticks([0, 25, 50, 75, 100, 125, 150])
ax.set_xlabel("Share of the system bound (%)")
ax.tick_params(axis="y", left=False, labelleft=False)
ax.spines["left"].set_visible(False)
ax.grid(True, axis="x", color=GRID, linewidth=0.5)
ax.set_axisbelow(True)
# every mark named, column by column: the interval with 10-min handling, the other two rules, 30-min handling (the open
# circle is its lower end, the full-path reservation plateau; the thin bar runs up to its system bound, the short tick)
fig.legend(handles=[Line2D([], [], **rk("reserve", linestyle="", markersize=MS_PT)),
                    (Line2D([], [], color=C["interval_fill"], linewidth=6, solid_capstyle="butt"),
                     Line2D([], [], marker="|", color=C["bound_system"], markersize=9, markeredgewidth=1.8, linestyle="")),
                    Line2D([], [], **RF_MK),
                    Line2D([], [], **rk("segment_safe", linestyle="")),
                    Line2D([], [], **rk("segment", linestyle="")),
                    (Line2D([], [], color=C["interval_fill_light"], linewidth=2.5, solid_capstyle="butt"),
                     Line2D([], [], **rk("reserve", linestyle="", markersize=MS - 0.8, markerfacecolor=PAPER))),
                    Line2D([], [], color=C["offline_dispatch"], linewidth=1.4, marker="|", markersize=5,
                           markeredgewidth=1.2),
                    Line2D([], [], marker="v", markersize=MS, color=C["booked_random"], markeredgewidth=0.9,
                           linestyle=""),
                    Line2D([], [], marker="v", markersize=MS, color=C["booked_random"], markerfacecolor=PAPER,
                           markeredgewidth=0.9, linestyle="")],
           labels=["lower end: full-path reservation's plateau", "capacity interval up to the system bound",
                   "route-free bound", "safe release", "reference rule", "30-min handling (thin bar, open circle)",
                   "offline dispatch (all tasks known)", "random times, CV 0.3 (booked order)",
                   "random times, CV 0.6 (booked order)"],
           handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)}, loc="outside upper center", ncol=2,
           fontsize=FS_NOTE, handlelength=1.8, columnspacing=3.0, handletextpad=0.5)
# the numbers drawn are those of the rows (read back from the figure): bar ends, the other two rules, both ends with
# 30-min handling, the two columns; and those the text quotes (numbers.tex)
xs = {round(float(x), 6) for ln in ax.lines for x in ln.get_xdata()}
bars = sorted((round(p.get_x(), 6), round(p.get_x() + p.get_width(), 6)) for p in ax.patches)
assert bars == sorted([(round(share(r, r["low"]), 6), 100.0) for r in R5]
                      + [(round(share(r, r["lo3"]), 6), round(share(r, r["up3"]), 6)) for r in R5 if r["lo3"]]), bars
assert all(round(share(r, r[k]), 6) in xs for r in R5 for k in ("low", "safe", "ref", "lo3", "up3", "rf") if r[k]) and 100 in xs
assert [r["label"] for r in R5 if r.get("offline")] == ["Okpo, whole road", "Yantai, whole road"]
assert all(round(share(r, v), 6) in xs for r in R5 if r.get("offline") for v in r["offline"])
assert all(round(share(r, v), 6) in xs for r in R5 if r.get("random") for v in r["random"])
for r in R5:                                        # the random-time markers as the text quotes them
    if r.get("random"):
        k = {"yupu": "Yu", "yantai": "Ya"}[r["yard"]]
        assert f"{share(r, r['random'][0]):.1f}\\%" == MACRO["StochMainLo" + k]
        assert f"{share(r, r['random'][1]):.1f}\\%" == MACRO["StochMainHi" + k]
for r in R5:                                        # the offline ranges as Fig. 5's caption quotes them
    if r.get("offline"):
        k = {"yupu": "Yu", "yantai": "Ya"}[r["yard"]]
        assert f"{share(r, r['offline'][0]):.0f}--{share(r, r['offline'][1]):.0f}\\%" == MACRO["LowOffline" + k]
# route-free markers: Okpo's main-case rows (whole and sectioned) and Okpo's whole-road rows of the other scenarios
# (Yantai's route margins are zero), the main-case and crane-paced ones as the text quotes them
assert [r["label"] for r in R5 if r["rf"]] == ["Okpo, whole road", "Okpo, sectioned", "Okpo, whole road",
                                               "Okpo, whole road"] and grp(R5[6]["rf"]) == MACRO["YuRFCrane"]
cols = [t.get_text() for t in axt.texts]
assert all(f"{grp(r['low'])}–{grp(r['up'])}" in cols and f"{share(r, r['low']):.0f}%" in cols for r in R5)
pc = lambda m: MACRO[m].replace("\\%", "%")
assert grp(R5[0]["rf"]) == grp(R5[1]["rf"]) == MACRO["YuRFCons"] and f"{share(R5[0], R5[0]['rf']):.0f}%" == pc("YuRFConsShare")
for r, low_m, pct_m, pct30_m in zip(R5, ("YuConsResLowWhole", "YuConsResLowSec", "YaConsResLowWhole", "YaConsResLowSec",
                                         "YuErResLowWhole", "YuErResLowSec", "YuCraneSixteenPlat", None),
                                    ("YuConsResLowWholeRatio", "YuConsResLowSecRatio", "YaConsResLowWholeRatio",
                                     "YaConsResLowSecRatio", "YuErResLowWholeRatio", "YuErResLowSecRatio", "YuCraneResRatio",
                                     "YaCraneResRatio"),
                                    ("YuConsLoadThirtyWholeRatio", "YuConsLoadThirtySecRatio", "YaConsLoadThirtyWholeRatio",
                                     "YaConsLoadThirtySecRatio", None, None, None, None)):
    assert (low_m is None or grp(r['low']).replace(",", "{,}") == MACRO[low_m]) and f"{share(r, r['low']):.0f}%" == pc(pct_m), r
    assert pct30_m is None or f"{100 * r['lo3'] / r['up3']:.0f}%" == pc(pct30_m), r
SRC_INTERVAL = "results/main_case/interval.csv (main case); results/interventions/lower_tier.csv; results/stochastic/main_case_plateau.csv; results/interval/interval.csv, crane_interval.csv, crane_16day_plateau.csv; results/saturation/rules.csv; results/bounds/*/route_free.csv, results/bounds/interval_route_free.csv"
save(fig, "fig_interval.pdf", SRC_INTERVAL)

# ================================================================== Supplementary Figure S2: required fleet (every move scaled)
DEM = demand()
# each yard its own range (Yantai's fleets stay below 70), the rows as high as their ranges, so both keep one scale
TOPS = {"yupu": 160, "yantai": 100}
OFFS = {"free": -12, "reserve": -4, "segment_safe": 4, "segment": 12}
fig, axes = plt.subplots(2, 2, figsize=(FIGW, 108 * MM), layout="constrained", sharey="row",
                         gridspec_kw={"height_ratios": [TOPS[y] for y in YARDS]})
for i, y in enumerate(YARDS):
    TOP = TOPS[y]
    NR = TOP - 4                 # plotting row for "not reached within the fleets simulated"
    for j, h in enumerate((16.0, 24.0)):
        ax = axes[i][j]
        lo, hi = KS[y]
        ax.axhspan(lo - 1.5, hi + 1.5, color=BAND, linewidth=0, zorder=0)
        ax.axhline((lo + hi) / 2, color=MUTED, linestyle="-.", linewidth=LW_REF, zorder=0)
        ax.axhspan(NR - 8, TOP, color=C["strip"], linewidth=0, zorder=0)            # the "not reached" row
        # K* label above the band at the left, except in the 16-h panels, where safe release (Okpo) or full-path
        # reservation (Yantai) climbs through that corner: there it sits below the band at the right, clear of every curve
        kx, ky, kha, kva = (1240, lo - 3, "right", "top") if h == 16.0 else (258, hi + 3, "left", "bottom")
        # the point estimate (the dash-dot line) and its interval, as the text gives them (e.g. "50 (45–55)")
        kst = MACRO[{"yupu": "Yu", "yantai": "Ya"}[y] + "KStarTwenty"].replace("--", "–")
        assert float(kst.split()[0]) == (lo + hi) / 2
        if h == 24.0:                     # two lines at the left, clear of safe release rising past 450 moves/day
            kst = kst.replace(" (", "\n(")
        ax.text(kx, ky, f"$K^{{*}}$ = {kst}", color=INK2, fontsize=FS_TEXT, ha=kha, va=kva,
                bbox={"facecolor": PAPER, "edgecolor": "none", "pad": 0.5, "alpha": 0.8})   # over the own-volume line
        # the comparison of the text, full-path reservation against free flow, in front, with the extra fleet between
        # them shaded; safe release and the reference rule thin and pale behind (their markers kept, small)
        req = {}
        for m in ORDER:
            pts = sorted((int(float(r["demand"])), r["K_req"]) for r in DEM
                         if r["yard"] == y and float(r["day_h"]) == h and r["section"] == "whole" and r["model"] == m)
            xs = [n for n, _ in pts]
            ks = [int(k) if k.isdigit() else None for _, k in pts]
            req[m] = dict(zip(xs, ks))
            back = {} if m in ("free", "reserve") else dict(linewidth=0.8, markersize=2.8, alpha=0.6, zorder=1.8)
            ax.plot([x for x, k in zip(xs, ks) if k is not None], [k for k in ks if k is not None], label=LABEL[m],
                    **rk(m, **back))
            over = [x for x, k in zip(xs, ks) if k is None]
            ax.plot([x + OFFS[m] for x in over], [NR] * len(over), **rk(m, linestyle="", markerfacecolor=PAPER))
        both = sorted(x for x in req["reserve"] if req["reserve"][x] is not None and req["free"].get(x) is not None)
        ax.fill_between(both, [req["free"][x] for x in both], [req["reserve"][x] for x in both],
                        color=C["extra_fleet_fill"], linewidth=0, zorder=0.8)
        if (i, j) == (0, 0):                                    # the lines named once, in (a)
            for m, x, dx, dy, ha in (("free", 1200, 0, 4, "right"), ("reserve", 700, 14, 0, "left"),
                                     ("segment", 900, 14, 0, "left"), ("segment_safe", 350, 0, 5, "center")):
                end_label(ax, x + dx, req[m][x], NAME[m], COL[m], dy, ha=ha, va="bottom" if dy else "center",
                          bbox={"facecolor": PAPER, "edgecolor": "none", "pad": 0.5, "alpha": 0.8})
            ax.text(650, 80, "extra\nfleet", fontsize=FS_NOTE, color=COL["reserve"], ha="center", va="center", zorder=3)
        if y == "yupu":
            own = [int(MACRO["OkOwn"]), int(MACRO["OkDirect"])]
            for n, lab in ((own[0], "own volume"), (2 * own[0], "own peak")):
                ax.axvline(n, color=MUTED, linewidth=LW_REF, linestyle=":")
                ax.text(n + 6, 141, lab, color=INK2, fontsize=FS_NOTE, ha="left",
                        bbox={"facecolor": PAPER, "edgecolor": "none", "pad": 1})
            # fleets at the own volume, a short label under the "own volume" header, in the blank area right of its
            # line, each in the colour of its rule (free flow's pale grey would not read as text: secondary ink)
            # (the numbers only, in the colours of the line labels: full-path reservation above, free flow below)
            rng_ = lambda m: (lambda a, b: f"{a}" if a == b else f"{a}–{b}")(*sorted(req[m][n] for n in own))
            ax.text(own[1] + 8, 132, rng_("reserve"), fontsize=FS_NOTE, color=COL["reserve"], ha="left", va="top",
                    fontweight="bold")
            ax.text(own[1] + 8, 123, rng_("free"), fontsize=FS_NOTE, color=INK2, ha="left", va="top")
            star = dict(marker="*", color=C["anchor"], markersize=MS_ANCHOR, markerfacecolor=PAPER, markeredgewidth=0.8,
                        linestyle="", zorder=5)
            if h == 24.0:
                ax.plot([500], [24], **star)
                ax.text(490, 30, "A", fontsize=FS_TEXT, color=C["anchor"], ha="right", va="bottom")
            ax.plot([600], [30], **star)
            ax.text(590, 36, "B", fontsize=FS_TEXT, color=C["anchor"], ha="right", va="bottom")
        ax.set_ylim(0, TOP)
        ax.set_xlim(250, 1250)
        ticks = list(range(0, NR - 20, 25))
        ax.set_yticks(ticks + [NR])
        ax.set_yticklabels([str(t) for t in ticks] + ["not reached"])
        break_marks(ax, NR - 18.5, NR - 14.5) if j == 0 else None
        ax.xaxis.set_major_formatter(THOUSANDS)
        ax.set_title(f"({'abcd'[2 * i + j]}) {YNAME[y]}, {int(h)}-h day", color=INK)
        grid(ax)
        if i == 1:
            ax.set_xlabel("Daily demand (moves/day)")
        if j == 0:
            ax.set_ylabel("Required fleet (vehicles)")
save(fig, "fig_workzone.pdf", "results/fleet/required_fleet.csv, fleet_8day.csv; results/runs/yantai_crane_fleet/demand.csv "
     "(one Yantai cell); results/saturation/kstar.csv")

# ================================================================== Figure 6: the screen in two levels
# (series.csv, cross.csv): the extra fleet of full-path reservation over free flow with fractional fleets (interpolated
# between K_req - 1 and K_req) at every tested level of ten series: two yards x 16- and 24-h days with every move between
# areas, all roads 1.5 m narrower or wider (16-h days), and Okpo under the intra-area mapping (both days). (a) against the
# demand ratio, the utilisation of the binding element: below the lower end no series exceeds 10 %, the lowest first
# crossing; (b) the road-width series against the chain load S_d, on which the three widths of a yard coincide
F7 = rows("results/screen/series.csv")
CR7 = {(r["series"], r["axis"], r["measure"]): r for r in rows("results/screen/cross.csv")}
SER7 = {"okpo_16": ("yupu", 16, "-"), "okpo_24": ("yupu", 24, "-"), "yantai_16": ("yantai", 16, "-"),
        "yantai_24": ("yantai", 24, "-"), "okpo_w-1.5": ("yupu", 16, "--"), "okpo_w+1.5": ("yupu", 16, ":"),
        "yantai_w-1.5": ("yantai", 16, "--"), "yantai_w+1.5": ("yantai", 16, ":"), "intra_16": ("yupu", 16, "-."),
        "intra_24": ("yupu", 24, "-.")}
assert set(SER7) == {r["series"] for r in F7}
# (series_main_case.csv, cross_main_case.csv): the main case on 16-h days at both yards, in the top rows of (a) and thick
# in (b); its 24-h series stay in the supplementary material. The lower end stays that of the ten series (the main case
# needs at most one
# vehicle more below it, in one cell); the upper ends include the main case
F7C = [r for r in rows("results/screen/series_main_case.csv") if r["series"] in ("okpo_c16", "yantai_c16")]
CR7C = {(r["series"], r["axis"], r["measure"]): r for r in rows("results/screen/cross_main_case.csv")}
SERC = {"okpo_c16": ("yupu", 16, "-"), "yantai_c16": ("yantai", 16, "-")}
LOW7 = 100 * min(float(CR7[(s, "ratio", "gap_frac")]["first"]) for s in SER7)
UP7 = [100 * float(CR7[(s, "ratio", "gap_frac")]["last"]) for s in SER7 if SER7[s][1] == 16] + \
      [100 * float(CR7C[(s, "ratio", "gap_frac")]["last"]) for s in SERC]
assert f"{LOW7:.1f}%" == pc("ScreenLowEnd") and f"{min(UP7):.1f}%" == pc("ScreenUpLo") and f"{max(UP7):.1f}%" == pc("ScreenUpHi")
XMAX7 = 25.0                                    # levels up to a quarter of the bound; all beyond exceed 10 % by far
assert all(float(r["gap_frac"]) > 0.10 for r in F7 + F7C if 100 * float(r["ratio"]) > XMAX7)
# (a) every tested level of each series as a cell, classed as Section 4.4 reads the screen: below the lower end in whole
# vehicles (same fleet, one vehicle more), from it by the extra fleet with fractional fleets (at most or over 10 %); the
# main case in the top rows; each cell reaches halfway (log) to its neighbours and stops at the lower end. (b) the extra
# fleet against the chain load: the main case thick, marked and labelled, the road widths a faded background in the
# yard's colour; the y axis stops at 55 %, the lines leaving through the top edge
CLS7 = ("same fleet", "one vehicle more", "at most 10% more", "over 10% more")
CCOL7 = (C["cell_same"], C["cell_one_more"], C["cell_frac_low"], C["cell_frac_high"])


def cls7(r):
    if 100 * float(r["ratio"]) < LOW7:                 # below the lower end: whole vehicles, at most one more
        dk = int(r["K_res"]) - int(r["K_free"])
        assert dk in (0, 1), r
        return dk
    return 2 + (float(r["gap_frac"]) > 0.10)           # from it: fractional fleets


ROW7 = [("okpo_c16", "Okpo, 16 h"), ("yantai_c16", "Yantai, 16 h"), ("okpo_16", "Okpo, 16 h"), ("okpo_24", "Okpo, 24 h"),
        ("yantai_16", "Yantai, 16 h"), ("yantai_24", "Yantai, 24 h"), ("okpo_w-1.5", "Okpo, \u22121.5 m"),
        ("okpo_w+1.5", "Okpo, +1.5 m"), ("yantai_w-1.5", "Yantai, \u22121.5 m"), ("yantai_w+1.5", "Yantai, +1.5 m"),
        ("intra_16", "Okpo, 16 h"), ("intra_24", "Okpo, 24 h")]
GRP7 = (("Main\ncase", 0, 2), ("Every\nmove\nscaled", 2, 6), ("All\nroads,\n16 h", 6, 10), ("Intra-\narea", 10, 12))
assert [s for s, _ in ROW7] == list(SERC) + list(SER7)
DATA7 = {s: sorted((r for r in (F7C if s in SERC else F7) if r["series"] == s and 100 * float(r["ratio"]) <= XMAX7),
                   key=lambda r: float(r["ratio"])) for s, _ in ROW7}
# below the lower end, ten seeds: one vehicle more in three cells; with 30 seeds the first two need equal fleets (open
# circles), the main case's still one more (S1.4)
ODD7 = {("intra_16", "150"), ("okpo_w-1.5", "75")}
assert {(s, r["n"]) for s in DATA7 for r in DATA7[s] if 100 * float(r["ratio"]) < LOW7 and cls7(r) == 1} == \
       ODD7 | {("okpo_c16", "75")}
assert f"{100 * float(next(r for r in DATA7['okpo_c16'] if r['n'] == '75')['ratio']):.1f}%" == pc("ScreenOddRatio")
fig = plt.figure(figsize=(FIGW, 84 * MM), layout="constrained")
gs = fig.add_gridspec(1, 3, width_ratios=[33, 68, 60])
axa = fig.add_subplot(gs[1])
axl = fig.add_subplot(gs[0], sharey=axa)                # group and row names
axb = fig.add_subplot(gs[2])
for i, (s, lab) in enumerate(ROW7):
    xs7 = [100 * float(r["ratio"]) for r in DATA7[s]]
    mid = [math.sqrt(a * b) for a, b in zip(xs7, xs7[1:])]
    for r, x, x0, x1 in zip(DATA7[s], xs7, [xs7[0] ** 2 / mid[0]] + mid, mid + [xs7[-1] ** 2 / mid[-1]]):
        x0, x1 = (x0, min(x1, LOW7)) if x < LOW7 else (max(x0, LOW7), x1)
        axa.barh(-i, x1 - x0, left=x0, height=0.62, color=CCOL7[cls7(r)], edgecolor=PAPER, linewidth=0.6, zorder=2,
                 gid=f"{s}|{r['n']}")
        if (s, r["n"]) in ODD7:
            axa.plot([x], [-i], linestyle="", marker="o", markersize=2.8, markerfacecolor=PAPER, markeredgecolor=INK,
                     markeredgewidth=0.6, zorder=3, gid=f"odd|{s}|{r['n']}")
    axl.text(1, -i, lab, fontsize=FS_TEXT, color=INK if i < 2 else INK2, ha="right", va="center",
             fontweight="bold" if i < 2 else "normal")
for g, a, b in GRP7:
    axl.text(0, -(a + b - 1) / 2, g, fontsize=FS_NOTE, color=INK, ha="left", va="center")
    if a:
        for ax in (axl, axa):
            ax.axhline(-a + 0.5, color=GRID, linewidth=LW_REF, clip_on=False)
axa.axvspan(min(UP7), max(UP7), color=C["extra_fleet_fill"], linewidth=0, zorder=0)
axa.axvline(LOW7, color=INK, linewidth=0.9, zorder=4)
axa.text((1.0 * LOW7) ** 0.5, 0.6, f"below {LOW7:.1f}%:\nat most one more", fontsize=FS_NOTE, color=INK, ha="center",
         va="bottom")
axa.text((min(UP7) * max(UP7)) ** 0.5, 0.6, "upper ends,\n16-h days", fontsize=FS_NOTE, color=INK, ha="center", va="bottom")
axa.set_xscale("log")
axa.xaxis.set_minor_locator(NullLocator())
axa.set_xlim(1.0, XMAX7)
axa.xaxis.set_major_locator(FixedLocator([1, 2, LOW7, 10, 20]))
axa.set_xticklabels(["1", "2", f"{LOW7:.1f}", "10", "20"])
axa.set_ylim(-len(ROW7) + 0.4, 2.2)
axa.tick_params(axis="y", left=False, labelleft=False)
axa.spines["left"].set_visible(False)
axa.grid(True, axis="x", color=GRID, linewidth=0.5)
axa.set_axisbelow(True)
axa.set_xlabel("Daily demand / system bound (%, logarithmic)")
axa.set_title("(a) Demand ratio, every tested level", color=INK)
axl.set_xlim(0, 1)
axl.axis("off")
style7 = lambda y: dict(color=YCOL[y], linestyle="-", linewidth=0.8, alpha=0.45)
LW7C = 2.0                                                     # the main case, thick
styleC = lambda y: dict(color=YCOL[y], linestyle="-", linewidth=LW7C, marker=YMK[y], markersize=MS - 0.4,
                        markerfacecolor=YCOL[y], markeredgewidth=0.8)
LAB7 = {"yupu": ("Okpo, main case", "right"), "yantai": ("Yantai, main case", "left")}
YTOP7 = 55
pts7 = lambda s, src: [q for q in sorted((100 * float(r["ratio"]), float(r["S_d"]), 100 * float(r["gap_frac"]))
                                         for r in src if r["series"] == s) if q[0] <= XMAX7]
for s, (y, h, ls) in SER7.items():                             # (b): a yard's three road widths
    if h == 16 and not s.startswith("intra"):
        pts = pts7(s, F7)
        axb.plot([q[1] for q in pts], [q[2] for q in pts], gid=f"{s}|b", **style7(y))
for s, (y, h, ls) in SERC.items():
    pts = pts7(s, F7C)
    axb.plot([q[1] for q in pts], [q[2] for q in pts], gid=f"{s}|b", **styleC(y), zorder=3)
    q = next(q for q in pts if q[2] >= (25 if y == "yupu" else 10))   # Okpo labelled at 25 %, Yantai at 10 %
    lab, ha = LAB7[y]
    axb.annotate(lab, xy=(q[1], q[2]), xytext=(-6, 4) if ha == "right" else (8, -2), textcoords="offset points", ha=ha,
                 va="bottom" if ha == "right" else "top", fontsize=FS_NOTE, color=YCOL[y], fontweight="bold",
                 bbox={"facecolor": PAPER, "edgecolor": "none", "pad": 0.6, "alpha": 0.85}, zorder=4)
axb.axhline(10, color=MUTED, linewidth=LW_REF, linestyle=(0, (4, 2)), zorder=0.6)   # above the grid lines (0.5)
axb.text(0.42, 11, "10%", fontsize=FS_NOTE, color=INK2, ha="left", va="bottom")
# (b) the chain load at each yard's upper end (last crossing with fractional fleets, three road widths)
SDU = {y: [float(CR7[(s, "S_d", "gap_frac")]["last"]) for s in SER7 if SER7[s][0] == y and SER7[s][1] == 16
           and not s.startswith("intra") and (s.endswith("16") or "1.5" in s)] for y in YARDS}
for y in YARDS:
    assert f"{min(SDU[y]):.2f}--{max(SDU[y]):.2f}" == MACRO[{"yupu": "Yu", "yantai": "Ya"}[y] + "ChainUp"]
    axb.axvspan(min(SDU[y]), max(SDU[y]), color=YCOL[y], alpha=0.25, linewidth=0, zorder=0)
    left = y == "yupu"                                  # Okpo's label left of its band, Yantai's right of its own
    axb.text(min(SDU[y]) * 0.96 if left else max(SDU[y]) * 1.04, YTOP7 - 3,
             f"{YNAME[y]}'s{' ' if left else chr(10)}upper end\n{min(SDU[y]):.2f}\u2013{max(SDU[y]):.2f}", fontsize=FS_NOTE,
             color=YCOL[y], bbox={"facecolor": PAPER, "edgecolor": "none", "pad": 0.6, "alpha": 0.85}, zorder=4,
             ha="right" if left else "left", va="top")
axb.set_xscale("log")
axb.xaxis.set_minor_locator(NullLocator())
grid(axb)
axb.set_xlim(0.4, 6.0)
axb.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 4]))
axb.set_xticklabels(["0.5", "1", "2", "4"])
axb.set_xlabel("Chain load $S$ (logarithmic)")
axb.set_ylabel("Extra fleet of full-path reservation (%)")
axb.set_ylim(-3, YTOP7)
axb.set_title("(b) Chain load, 16-h days", color=INK)
# legends: the classes in their two regions (a heading without a mark above each; the open circles in the caption), then
# the lines of (b)
fig.legend(handles=[Line2D([], [], linestyle=""), Patch(color=CCOL7[0]), Patch(color=CCOL7[1]),
                    Line2D([], [], linestyle=""), Patch(color=CCOL7[2]), Patch(color=CCOL7[3])],
           labels=[f"below {LOW7:.1f}% (whole vehicles):", CLS7[0], CLS7[1],
                   f"from {LOW7:.1f}% (fractional fleets):", CLS7[2], CLS7[3]],
           loc="outside upper left", ncol=2, fontsize=FS_NOTE, handlelength=1.6, columnspacing=1.6)
fig.legend(handles=[(Line2D([], [], **styleC("yupu")), Line2D([], [], **styleC("yantai"))),
                    (Line2D([], [], **style7("yupu")), Line2D([], [], **style7("yantai")))],
           labels=["(b) main case, 16-h days", "(b) every move scaled; all\nroads 1.5 m narrower or wider"],
           handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)}, loc="outside upper right", fontsize=FS_NOTE,
           handlelength=2.4)
# the cells drawn are those of the series, in the colour of their class and on the right side of the lower end; the
# open circles; the lines of (b) (read back from the figure)
cells = {pa.get_gid(): pa for pa in axa.patches if pa.get_gid()}
assert len(cells) == sum(len(v) for v in DATA7.values())
for s, rs in DATA7.items():
    for r in rs:
        pa, x = cells[f"{s}|{r['n']}"], 100 * float(r["ratio"])
        assert pa.get_x() <= x <= pa.get_x() + pa.get_width() and pa.get_facecolor() == to_rgba(CCOL7[cls7(r)])
        assert pa.get_x() + pa.get_width() <= LOW7 if x < LOW7 else pa.get_x() >= LOW7
assert {tuple(ln.get_gid().split("|")[1:]) for ln in axa.lines if (ln.get_gid() or "").startswith("odd|")} == ODD7
got = {ln.get_gid(): (list(ln.get_xdata()), list(ln.get_ydata())) for ln in axb.lines if ln.get_gid()}
for s, (y, h, ls) in SER7.items():
    assert (f"{s}|b" in got) == (h == 16 and not s.startswith("intra"))
    if f"{s}|b" in got:
        assert got[f"{s}|b"] == ([q[1] for q in pts7(s, F7)], [q[2] for q in pts7(s, F7)])
for s in SERC:                                  # the main case; its upper end inside its yard's band in (b)
    assert got[f"{s}|b"] == ([q[1] for q in pts7(s, F7C)], [q[2] for q in pts7(s, F7C)])
    assert min(SDU[SERC[s][0]]) <= float(CR7C[(s, "S_d", "gap_frac")]["last"]) <= max(SDU[SERC[s][0]])
SRC_SCREEN = "results/screen/series.csv, cross.csv (K_free, K_res, gap_frac; axes ratio and S_d); results/screen/series_main_case.csv, cross_main_case.csv (main case, 16-h days)"
save(fig, "fig_screen.pdf", SRC_SCREEN)

# ================================================================== Supplementary Figure S1: rehandling within areas
# scan.csv, scan_switch.csv: the system bound as the share q of moves made within an area grows, erection conserved; one
# switch of the binding element per yard, where 1/lambda^S, convex and piecewise linear in q, has its breakpoint
SCAN = rows("results/main_case/scan.csv")
SW = {r["yard"]: r for r in rows("results/main_case/scan_switch.csv")}
BEFORE = {"yupu": "through corridor", "yantai": "paint-shop doors"}
AFTER = {"yupu": "assembly-area bays", "yantai": "panel-line hall doors"}
fig, ax = plt.subplots(figsize=(FIGW, 62 * MM), layout="constrained")
for y in YARDS:
    pts = sorted((100 * float(r["q"]), float(r["lam16"]), r["top1"]) for r in SCAN if r["yard"] == y)
    assert all(res_name(t) == (BEFORE[y] if q < 100 * float(SW[y]["q_star"]) else AFTER[y]) for q, _, t in pts)
    qx, lx = 100 * float(SW[y]["q_star"]), float(SW[y]["lam_star"])
    # 1/lambda^S is linear in q on either side of the switch (scan points within 1e-13 of two lines), so the switch,
    # where the two lines meet, is a point of the curve: drawn between the scan points
    qs = [p[0] for p in pts if p[0] < qx] + [qx] + [p[0] for p in pts if p[0] > qx]
    lam = [p[1] for p in pts if p[0] < qx] + [lx] + [p[1] for p in pts if p[0] > qx]
    ax.plot(qs, lam, color=YCOL[y], linewidth=1.2, gid=f"scan|{y}")
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=YCOL[y], linestyle="", marker=YMK[y], markersize=MS - 1.2)
    ax.plot([qx, qx], [0, lx], color=YCOL[y], linewidth=LW_THIN, linestyle=":")
    assert f"{qx:.0f}%" == pc(("Yu" if y == "yupu" else "Ya") + "ScanSwitch") and grp(lx) == MACRO[("Yu" if y == "yupu" else "Ya") + "ScanMax"]
    dy = 90 if y == "yupu" else -110
    ax.text(qx / 2, float(next(p[1] for p in pts if p[0] >= qx / 2)) + dy, BEFORE[y], fontsize=FS_NOTE, color=YCOL[y],
            ha="center", va="bottom" if dy > 0 else "top")
    ax.text((qx + 80) / 2, float(next(p[1] for p in pts if p[0] >= (qx + 80) / 2)) + dy, AFTER[y], fontsize=FS_NOTE,
            color=YCOL[y], ha="center", va="bottom" if dy > 0 else "top")
    for q0, lab in ((0.0, "main case"), (65.5, "intra-area\nmapping")):
        v = next(p[1] for p in pts if abs(p[0] - q0) < 1e-6)
        ax.plot([q0], [v], marker="o", markersize=MS_PT + 1, markerfacecolor="none", markeredgecolor=INK,
                markeredgewidth=0.8, linestyle="", zorder=5)
        if y == "yupu":
            ax.annotate(lab, xy=(q0, v), xytext=(q0 + 4, v + 420) if q0 == 0 else (q0 - 4, v - 520), fontsize=FS_NOTE,
                        color=INK, ha="left" if q0 == 0 else "right", va="bottom" if q0 == 0 else "top",
                        arrowprops={"arrowstyle": "-", "lw": LW_THIN, "color": INK2, "shrinkA": 1, "shrinkB": 3})
    ax.text(81, lam[-1], YNAME[y], fontsize=FS_TEXT, color=YCOL[y], ha="left", va="center")
ax.set_xlim(0, 88)
ax.set_ylim(0, 3700)
ax.set_xticks([0, 20, 40, 60, 80])
ax.yaxis.set_major_formatter(THOUSANDS)
ax.set_xlabel("Moves within an area (% of all moves; erection flow kept at its weight)")
ax.set_ylabel("System bound (moves per 16-h day)")
grid(ax)
save(fig, "fig_scan.pdf", "results/main_case/scan.csv, scan_switch.csv")

# ================================================================== Figure 7: dock-mouth thresholds at the two yards
# eight days, or 16 where the eight-day runs were not steady (thresholds.csv: thr_table, steady_table; Okpo rev 1833)
RK = {r["key"]: r for r in rows("results/dock/thresholds.csv")}
assert all(f"{float(RK[k]['thr_table']):.2f}" == MACRO[m] for k, m in (("yupu|whole|segment_safe|20", "YuSafeThrTwenty"),
                                                                     ("yantai|whole|segment_safe|150", "YaSafeThrOneFifty")))
KK = (20, 40, 60, 100, 150)
NOD = 1.03          # plotting row for "no 10 % loss up to the top utilisation"
fig, axes = plt.subplots(2, 2, figsize=(FIGW, 100 * MM), layout="constrained", sharey=True, sharex=True)
for i, sec in enumerate(("whole", "stops")):
    for j, y in enumerate(YARDS):
        ax = axes[i][j]
        for m in ("reserve", "segment_safe", "segment"):
            xs, vs, st = [], [], []
            for K in KK:
                r = RK[f"{y}|{sec}|{m}|{K}"]
                xs.append(K + {"reserve": -2.5, "segment_safe": 0, "segment": 2.5}[m])
                vs.append(float(r["thr_table"]) if r["thr_table"] else NOD)
                st.append(r["steady_table"] == "True" or not r["thr_table"])
            ax.plot(xs, vs, **rk(m, marker=""))
            for x, v, s in zip(xs, vs, st):
                ax.plot([x], [v], **rk(m, linestyle="", markerfacecolor=COL[m] if s else PAPER))
            if (i, j) == (0, 0):          # the lines named once, in (a) (no legend; open markers in the caption)
                end_label(ax, xs[-1] + 2, 1.075 if m == "reserve" else vs[-1] - 0.025, NAME[m], COL[m], 0, ha="right",
                          va="bottom" if m == "reserve" else "top", clip_on=False)
        ax.axhspan(1.015, 1.07, color=C["strip"], linewidth=0, zorder=0)            # the "no 10% loss" row
        ax.axhline(1.0, color=GRID, linewidth=LW_THIN)
        if (i, j) == (1, 0):              # the open markers explained once, in an empty corner of (c)
            ax.text(17, 0.33, "open markers: runs not steady\nover 16 days", fontsize=FS_NOTE, color=INK2, ha="left",
                    va="bottom")
        ax.set_title(f"({'abcd'[2 * i + j]}) {YNAME[y]}, {'whole' if sec == 'whole' else 'sectioned'} dock road", color=INK)
        ax.set_xticks(KK)
        ax.set_ylim(0.3, 1.07)
        grid(ax)
        if i == 1:
            ax.set_xlabel("Fleet size $K$ (vehicles)")
        if j == 0:
            ax.set_ylabel(r"Threshold $\rho_{10}$")
            ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, NOD])
            ax.set_yticklabels(["0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "no 10% loss"])
            break_marks(ax, 0.995, 1.015)
save(fig, "fig_dock.pdf", "results/dock/thresholds.csv (thr_table, steady_table)")

# ================================================================== Figure 8(b, c): the dock-mouth mechanism (synthetic network)
# (a), the network, is a schematic not drawn by this code; the numbers it prints are these macros, which must be the
# geometry of the experiment (experiments/dock_mouth.py)
DOCK_PY = show("experiments/dock_mouth.py")
L1_, DOCK_S_, YARD_S_ = map(float, re.search(r"^L1, DOCK_S, YARD_S = ([\d.]+), ([\d.]+), ([\d.]+)", DOCK_PY, re.M).groups())
SRC_S = [float(s) for s in re.search(r"^SRC = \(([^)]*)\)", DOCK_PY, re.M).group(1).split(",")]
assert (L1_, DOCK_S_, YARD_S_, len(SRC_S)) == (num("NetLen"), num("NetAt"), num("YardAt"), num("NetSrcBays"))
D0 = num("NetDzero")
RS25 = rows(SYN_DOCK + "/rho_star.csv")
RHOS = [float(c.split("@")[1]) for c in RS25[0] if c.startswith("thr_rel@")]
KD = sorted({int(r["K"]) for r in RS25})
cls = lambda g: int(g // D0)
CURV, THR = defaultdict(list), {}
for r in RS25:
    g, K = float(r["margin"]), int(r["K"])
    c = [float(r[f"thr_rel@{x:g}"]) for x in RHOS]
    if r["rule"] == "seg":
        key = (cls(g), K)
        assert key not in THR or THR[key] == r["rho_star_thr10"]          # every margin of a class: same threshold
        THR[key] = r["rho_star_thr10"]
        CURV[key].append(c)
    elif r["rule"] in ("seg_bay", "res"):
        assert not r["rho_star_thr10"]                                     # waiting off the road: no 10 % loss
        CURV[(r["rule"], K)].append(c)
KM = int(num("NetMechK"))
assert all(c == CURV[(k, KM)][0] for k in (0, 1, 2) for c in CURV[(k, KM)])   # at the fleet drawn: one curve per class
# (b) and (c) at two thirds of the text width and the height of (a), which is set beside them at one third (54.87 mm)
DMH = 76 * MM
# (b) throughput against crane utilisation at one fleet size
fig = plt.figure(figsize=(2 * FIGW / 3, DMH), layout="constrained")
gs = fig.add_gridspec(1, 2)
ax = fig.add_subplot(gs[0])
ax.axhline(0.9, color=MUTED, linestyle="--", linewidth=LW_REF)
ax.text(RHOS[0] + 0.005, 0.905, "10% loss", fontsize=FS_NOTE, color=INK2, va="bottom")
# waiting off the road: two thin lines at the top, named where they start; waiting on the road: the three margin steps
# as plain lines, each with its threshold rho_10 marked where it crosses the 10 % line (the values of (c) at this fleet)
for rule, kw, lab, yl in (("res", rk("reserve", marker="", linewidth=1.3), NAME["reserve"], 1.052),
                          ("seg_bay", rk("segment_safe", linestyle="-.", marker="", linewidth=1.0),
                           "safe release, off-road unloading bay", 1.085)):
    cs = CURV[(rule, KM)]
    lo_ = [min(c[i] for c in cs) for i in range(len(RHOS))]
    hi_ = [max(c[i] for c in cs) for i in range(len(RHOS))]
    ax.fill_between(RHOS, lo_, hi_, color=kw["color"], alpha=0.15, linewidth=0)
    ax.plot(RHOS, [statistics.mean(c[i] for c in cs) for i in range(len(RHOS))], **kw)
    ax.text(RHOS[0], yl, lab, fontsize=FS_NOTE, color=kw["color"], ha="left", va="bottom")
for c in (0, 1, 2):
    v = CURV[(c, KM)][0]
    ax.plot(RHOS, v, **rk("segment_safe", color=MARGIN[c], marker="", linewidth=1.6))
    t = float(THR[(c, KM)])
    i = next(i for i in range(len(RHOS) - 1) if v[i] >= 0.9 > v[i + 1])      # the threshold is this crossing
    assert abs(RHOS[i] + (0.9 - v[i]) / (v[i + 1] - v[i]) * (RHOS[i + 1] - RHOS[i]) - t) < 1e-3
    ax.plot([t], [0.9], **rk("segment_safe", color=MARGIN[c], markeredgecolor=INK, markerfacecolor=MARGIN[c],
                             markersize=MS_PT, markeredgewidth=0.6, linestyle="", zorder=4))
    ax.text(RHOS[-1] + 0.012, v[-1], f"{c}", fontsize=FS_TEXT, color=INK2, va="center")
ax.text(RHOS[-1] + 0.012, 0.95, r"$\lfloor g/d_0 \rfloor$", fontsize=FS_TEXT, color=INK2, va="center")
ax.text(RHOS[0] + 0.005, 0.80, r"squares: $\rho_{10}$" "\nsafe release, whole\ndock road, one line\n"
        r"per $\lfloor g/d_0 \rfloor$" "\noff the road: mean\nover the margins,\nrange shaded",
        fontsize=FS_NOTE, color=INK2, ha="left", va="top")      # below 0.78 the lines are right of rho = 0.85
ax.set_xlim(RHOS[0] - 0.02, 1.08)
ax.set_xticks([0.4, 0.6, 0.8, 1.0])
ax.set_ylim(0.48, 1.13)
ax.set_xlabel(r"Crane utilisation $\rho$")
ax.set_ylabel("Throughput relative to the lowest $\\rho$")
ax.set_title(f"(b) $K$ = {KM}", color=INK)
grid(ax)
# (c) threshold against fleet size, by the number of empty vehicles that fit beside a standing loaded one
ax = fig.add_subplot(gs[1])
ax.axhspan(1.015, 1.07, color=C["strip"], linewidth=0, zorder=0)                # the "no 10% loss" row
ax.axvspan(KM - 1, KM + 1, color=BAND, linewidth=0, zorder=0)                     # the fleet of (b)
ax.text(KM, 0.99, "(b)", fontsize=FS_NOTE, color=INK2, ha="center", va="center")
for c in (0, 1, 2):
    vs = [float(THR[(c, K)]) if THR[(c, K)] else NOD for K in KD]
    ax.plot(KD, vs, **rk("segment_safe", color=MARGIN[c], markeredgecolor=MARGIN[c], marker=""))
    ax.plot([K for K, v in zip(KD, vs) if v != NOD], [v for v in vs if v != NOD],
            **rk("segment_safe", color=MARGIN[c], markeredgecolor=MARGIN[c], markerfacecolor=MARGIN[c], linestyle=""))
    ax.plot([K for K, v in zip(KD, vs) if v == NOD], [NOD + 0.012 * (c - 1)] * sum(v == NOD for v in vs),
            **rk("segment_safe", color=MARGIN[c], markeredgecolor=MARGIN[c], markerfacecolor=PAPER, linestyle=""))
    ax.text(KD[-1] + 0.8, vs[-1], f"{c}", fontsize=FS_TEXT, color=INK2, va="center")
ax.text(KD[-1] + 0.8, 0.995, r"$\lfloor g/d_0 \rfloor$", fontsize=FS_TEXT, color=INK2, va="center")
ax.text(KD[0], 0.515, "waiting off the road:\nno 10% loss at any $K$\nand margin",
        fontsize=FS_NOTE, color=INK2, va="bottom")
ax.set_xticks(KD)
ax.set_xlim(KD[0] - 1.5, KD[-1] + 5)
ax.set_ylim(0.5, 1.07)
ax.set_yticks([0.6, 0.7, 0.8, 0.9, NOD])
ax.set_yticklabels(["0.6", "0.7", "0.8", "0.9", "no 10%\nloss"])
break_marks(ax, 0.995, 1.015)
ax.set_xlabel("Fleet size $K$ (vehicles)")
ax.set_ylabel(r"Threshold $\rho_{10}$")
ax.set_title("(c) Whole dock road", color=INK)
grid(ax)
save(fig, "fig_dockmech_bc.pdf", "results/synthetic_dock/rho_star.csv, config.json", width=2 * FIGW / 3)

# ================================================================== Figure 9: external anchors against handling time
# the numbers of the anchor table (make_numbers.py tab_anchor): Okpo with 30 vehicles at saturation, moves per
# vehicle-day with t min each of loading and unloading on 24-h days (main_case_handling_time.csv), the four rules at the
# main handling time on 16-h days (main_case_per_vehicle.csv); safe release was run at the main handling time only
AF = [r for r in rows("results/anchors/main_case_handling_time.csv") if r["map"] == "conserved"]       # main case
F9 = {(r["map"], r["panel"], int(r["load_min"]), r["model"]): float(r["per_truck"]) for r in rows("results/anchors/main_case_per_vehicle.csv")}
S72 = {f"YuPerTruck{nm}{d}": F9[("conserved", p_, 10, mo)] for nm, mo in (("Free", "free"), ("Res", "reserve"),
       ("Ref", "segment"), ("Safe", "segment_safe")) for d, p_ in (("", "b"), ("Day", "a"))}
tt = [float(r["load_min"]) for r in AF]
hhi, shen = num("HHIRate"), num("ShenRate")
assert all(abs(float(r["anchor"]) - 500 / 24) < 1e-9 for r in AF) and f"{hhi:.1f}" == f"{500 / 24:.1f}"
assert abs(float(AF[0]["free"]) - S72["YuPerTruckFreeDay"]) < 1e-9 and tt[0] == num("LoadMin")
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(FIGW, 72 * MM), layout="constrained", width_ratios=[3.2, 1], sharey=True)
for m, col in (("free", "free"), ("reserve", "reserve"), ("segment", "reference")):
    ax.plot(tt, [float(r[col]) for r in AF], label=LABEL[m].replace(" (teleport recovery)", ""), **rk(m))
ax.plot([tt[0]], [S72["YuPerTruckSafeDay"]], label=LABEL["segment_safe"], **rk("segment_safe", linestyle=""))
ax.text(tt[0] + 1.2, S72["YuPerTruckSafeDay"] - 1.0, f"safe release: run at\n$t$ = {tt[0]:.0f} only", fontsize=FS_NOTE,
        color=COL["segment_safe"], ha="left", va="top")
ax.axhline(hhi, color=C["anchor"], linewidth=LW_BOUND)
ax.text(tt[-1], hhi + 0.6, f"Hyundai, {hhi:.1f} (24-h day)", fontsize=FS_NOTE, color=C["anchor"], ha="right", va="bottom")
ax.axhline(shen, color=C["anchor"], linewidth=1.0, linestyle="--")
ax.text(tt[-1], shen - 0.6, f"unnamed Korean yard, about {shen:.0f} (shift not reported)", fontsize=FS_NOTE, color=INK2,
        ha="right", va="top")
ax.set_xticks(tt)
ax.set_xlim(tt[0] - 3, tt[-1] + 3)
ax.set_ylim(0, 1.12 * max(float(r["free"]) for r in AF))
ax.set_xlabel("Loading and unloading time $t$, each (min)")
ax.set_ylabel("Throughput per vehicle (moves/day)")
ax.set_title("(a) 24-h day", color=INK)
grid(ax)
# (b): the four rules as stems on the scale of (a), each with its value (the numbers of the text)
for i, (m, mn) in enumerate((("free", "Free"), ("reserve", "Res"), ("segment_safe", "Safe"), ("segment", "Ref"))):
    v = S72["YuPerTruck" + mn]
    assert f"{v:.1f}" == MACRO["YuPerTruck" + mn]
    ax2.plot([i, i], [0, v], color=COL[m], linewidth=LW[m], solid_capstyle="butt")
    ax2.plot([i], [v], **rk(m, linestyle="", markersize=MS_PT))
    ax2.text(i, v + 1.0, f"{v:.1f}", fontsize=FS_NOTE, color=INK2, ha="center", va="bottom")
ax2.set_xticks([])                     # the rules by colour and marker, as in the legend (no abbreviations)
ax2.set_xlim(-0.6, 3.6)
ax2.tick_params(labelleft=False)
ax2.set_title(f"(b) 16-h day, $t$ = {tt[0]:.0f}", color=INK)
grid(ax2)
ax2.grid(False, axis="x")
h_, l_ = ax.get_legend_handles_labels()
order = [l_.index(LABEL[m].replace(" (teleport recovery)", "")) for m in ORDER]           # rule order of the paper
fig.legend([h_[k] for k in order], [l_[k] for k in order], loc="outside upper center", ncol=4, fontsize=FS_TEXT)
assert all(f"{S72[k]:.1f}" == MACRO[k] for k in S72)
save(fig, "fig_anchors.pdf", "results/anchors/main_case_handling_time.csv, main_case_per_vehicle.csv (main case)")

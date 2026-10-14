"""Draw the data figures of the paper, figures/*.pdf, and write figures/palette.json, from the result data of the
repository (results/, paths relative to the repository root, read through src.py; the src argument of each save call
names the sources, and the PDFs carry a neutral Subject). PNG previews go to build/.

Figures of this paper, in their order:
  Figure 3          the two yards with the binding elements of the system bound: not drawn here; it needs the centre-line
                    geometry of the yard maps, which is not released (the binding elements are in
                    results/bounds/*/prices.csv, resources with a positive dual price)
  fig_flag.pdf      Figure 4, fleet size vs throughput, three rules + reference rule, sampled and system bounds, seed
                    range shaded; Okpo (map with floating docks): results/saturation/okpo_curves.csv; Yantai:
                    results/saturation/rules_summary.json (sat), seed ranges from
                    results/runs/{yantai_rev395_saturation,yantai_safe_sectioned}/points.csv; results/bounds/*/system.csv
  fig_interval.pdf  Figure 5, capacity interval [full-path reservation, system bound] as a share of each row's system
                    bound, with both ends at 30-min handling on the same scale, blocks/day and the share in two columns;
                    results/interval/{interval.csv, handling_30min.csv, crane_interval.csv},
                    results/saturation/rules.csv; Okpo's crane-paced lower end over 16 days from
                    results/interval/crane_16day_plateau.csv
  fig_workzone.pdf  Figure 6, plant-level daily demand vs required fleet; results/fleet/required_fleet.csv with one Yantai
                    cell from results/runs/yantai_crane_fleet/demand.csv and Okpo's own-volume cells over eight days from
                    results/fleet/fleet_8day.csv (src.demand); K* from results/saturation/kstar.csv
  fig_screen.pdf    Figure 7, the screen as a signal band: each tested level of each series marked "same fleet as free
                    flow" or "full-path reservation needs more than 10 % more", against demand over the system bound (day
                    length scaled, logarithmic), in three zones around the band of first 10 % crossings, with the first
                    crossings of the fractional fleets (gap_frac) of the four-day runs and the own volumes;
                    results/fleet/{screen.csv, screen_cross.csv}, its table cells as in fig_workzone (checked against
                    screen.csv)
  fig_dock.pdf      Figure 8, dock-mouth threshold (10 % throughput drop) vs fleet size, whole and sectioned dock roads;
                    results/dock/thresholds.csv (eight days, or 16 where the eight-day runs were not steady)
  fig_dockmech_bc.pdf  Figure 9(b, c), synthetic dock mouth, at two thirds of the text width beside the schematic (a):
                    (b) throughput relative to the lowest utilisation against rho, (c) rho_10 against the fleet;
                    results/synthetic_dock/rho_star.csv; the geometry printed in (a) is checked against
                    experiments/dock_mouth.py
  fig_anchors.pdf   Figure 10, external anchors and the Okpo model at the same scale (blocks per vehicle-day against the
                    handling time; 16-h values at the main handling time); results/anchors/{handling_time.csv,
                    per_vehicle_rates.csv}
Figures 1, 2, 9(a) and 11 are diagrams and schematics, not drawn by this code; they use the colours of
figures/palette.json.

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
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from src import BOUNDS_DIR, SYN_DOCK, demand, interval, js, res_name, rows, show

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
# main colour darkest) and keep their markers and line styles, for black-and-white print and colour-blind readers
# (chosen to stay apart under simulated protanopia, deuteranopia and tritanopia). The hex values are written to
# figures/palette.json, which the schematics (Figures 2 and 9a, not drawn by this code) use; their resource
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
    "extra_fleet_fill": ("#E6EEF8", "extra fleet of full-path reservation over free flow, between the two lines "
                                    "(paler main colour)", "6"),
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


S = js("results/saturation/rules_summary.json")
SAT = S["sat"]
# onset of saturation at 20 % (kstar.csv; Okpo on the curve of rev 1833), as interval (lo, hi)
KS = {r["curve"]: (int(r["lo"]), int(r["hi"])) for r in rows("results/saturation/kstar.csv")
      if r["curve"] in YARDS and float(r["theta"]) == 0.2}
assert all(f"{KS[y][0]}" == MACRO[{"yupu": "Yu", "yantai": "Ya"}[y] + "KStarLo"] for y in YARDS)
SYS = {y: {r["scen"]: r for r in rows(BOUNDS_DIR[y] + "/system.csv")} for y in YARDS}


# ================================================================== Figure 3 (the two yards): not drawn here
# It draws the road centre lines of the yard maps, which are not released (only the derived graphs in
# data/networks are); the binding elements it marks are in results/bounds/*/prices.csv.

# ================================================================== Figure 4: saturation curves
F1 = defaultdict(dict)                          # Okpo, map with floating docks: model -> K -> row
for r in rows("results/saturation/okpo_curves.csv"):
    F1[r["model"]][int(r["K"])] = r
f1 = next(iter(F1["free"].values()))
assert abs(float(f1["system_bound"]) - float(SYS["yupu"]["main"]["T1_system"])) < 0.05


def seed_range(batch, scen, model, cur):
    """Seed minimum and maximum per K of a batch in results/runs; its seed means must reproduce the curve."""
    g = defaultdict(list)
    for r in rows(f"results/runs/{batch}/points.csv"):
        if (r["scen"], r["model"], r["level"]) == (scen, model, "saturated"):
            g[int(r["K"])].append(float(r["throughput"]))
    assert all(len(g[int(k)]) == 10 and abs(sum(g[int(k)]) / 10 - v) < 0.05 for k, v in cur.items())
    ks = sorted(int(k) for k in cur)
    return ks, [min(g[k]) for k in ks], [max(g[k]) for k in ks]


def every(ks, at=(20, 55, 90)):
    return sorted({min(range(len(ks)), key=lambda i: abs(ks[i] - a)) for a in at})


def end_label(ax, x, y, text, color, dy, **kw):
    """Name a line at its end, in the line's colour (free flow's pale grey would not read: secondary ink)."""
    ax.text(x, y + dy, text, fontsize=FS_NOTE, color=INK2 if color == COL["free"] else color, **kw)


YA_SEED = {"free": "yantai_rev395_saturation", "reserve": "yantai_rev395_saturation", "segment": "yantai_rev395_saturation",
           "segment_safe": "yantai_safe_sectioned"}
fig, axes = plt.subplots(1, 2, figsize=(FIGW, 74 * MM), layout="constrained")
for i, (ax, y) in enumerate(zip(axes, YARDS)):
    b = f"{y}|main"
    sysb = float(SYS[y]["main"]["T1_system"])
    lo, hi = float(SYS[y]["main"]["ref_min"]), float(SYS[y]["main"]["ref_max"])
    mean_nb = float(f1["sample_bound_mean"]) if y == "yupu" else SAT[b]["T1_whole"]
    if y == "yupu":           # the ten task samples are the same on both maps
        assert abs(lo - float(f1["sample_bound_lo"])) < 0.1 and abs(hi - float(f1["sample_bound_hi"])) < 0.1
    k0, k1 = KS[y]
    ax.axvspan(k0 - 1.5, k1 + 1.5, color=BAND, linewidth=0, zorder=0)
    ax.axhspan(lo, hi, color=C["bound_range"], linewidth=0, zorder=0.5, label="Sampled bound, range")
    ax.axhline(mean_nb, color=C["bound_network"], linestyle="-.", linewidth=0.9, label="Sampled bound, mean")
    ax.axhline(sysb, color=C["bound_system"], linewidth=LW_BOUND, label="System bound")
    curves = {}
    for r in ORDER:
        if y == "yupu":
            ks = sorted(F1[r])
            v = [float(F1[r][k]["mean"]) for k in ks]
            lo_, hi_ = [float(F1[r][k]["lo"]) for k in ks], [float(F1[r][k]["hi"]) for k in ks]
        else:
            cur = SAT[b]["rules"][f"whole|{r}"]["curve"]
            ks, lo_, hi_ = seed_range(YA_SEED[r], "main", r, cur)
            v = [cur[str(k)] for k in ks]
        curves[r] = dict(zip(ks, v))
        if r != "segment":
            ax.fill_between(ks, lo_, hi_, color=COL[r], alpha=0.15, linewidth=0)
        ax.plot(ks, v, label=LABEL[r].replace(" (teleport recovery)", ""), markevery=every(ks), **rk(r))
    # every line named where it ends (no legend), clear of its own line: the two rising rules above their last points,
    # safe release, which declines, below; free flow beside its steep line; the bounds by their labels
    # at Okpo the reference rule is named below its line, clear of the two-line label of the sampled bound
    seg_dy, seg_va = (-0.03, "top") if y == "yupu" else (0.03, "bottom")
    for r, dy, va in (("reserve", 0.03, "bottom"), ("segment", seg_dy, seg_va), ("segment_safe", -0.03, "top")):
        end_label(ax, 148, curves[r][150], NAME[r], COL[r], dy * hi, ha="right", va=va)
    yf = 0.85 * 1.3 * hi
    kf = next(k for k in sorted(curves["free"]) if curves["free"][k] >= yf)
    end_label(ax, kf + 4, yf, NAME["free"], COL["free"], 0, ha="left", va="center")
    # free flow passes the system bound (ringed, with its fleet); K* band and the plateau window are readings of
    # Sections 4.2 and 4.3
    kx = min(k for k, v in curves["free"].items() if v > sysb)
    assert kx == num(("Yu" if y == "yupu" else "Ya") + "FreeCross")
    ax.plot([kx], [sysb], marker="o", markersize=MS_PT + 1, markerfacecolor="none", markeredgecolor=INK,
            markeredgewidth=0.8, linestyle="", zorder=5)
    ax.text(kx + 4, sysb - 0.03 * hi, f"crosses at $K$ = {kx}", fontsize=FS_NOTE, color=INK, ha="left", va="top",
            bbox={"facecolor": PAPER, "edgecolor": "none", "pad": 0.5, "alpha": 0.8})     # over the sampled-bound line
    name = res_name(SYS[y]["main"]["top1"])
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
axes[0].set_ylabel("Throughput (blocks/day)")
save(fig, "fig_flag.pdf", "results/saturation/okpo_curves.csv, rules_summary.json; results/runs/yantai_rev395_saturation, "
     "yantai_safe_sectioned points.csv; results/bounds/*/system.csv")

# ================================================================== Figure 5: capacity interval
I27 = interval()
R27E = {r["case"] + "|" + r["combo"]: float(r["plateau"]) for r in rows("results/saturation/rules.csv")}
L30 = {(r["yard"], r["gran"], r["kind"]): r for r in rows("results/interval/handling_30min.csv")}
CIV = {(r["yard"], r["gran"], r["mix"], r["kind"]): r for r in rows("results/interval/crane_interval.csv")}
# Okpo's crane-paced whole-road lower end over 16 days (crane_16day_plateau.csv; the four-day runs are not steady
# there), Yantai's over four
CP16 = {r["yard"]: r for r in rows("results/interval/crane_16day_plateau.csv")}
assert list(CP16) == ["yupu"] and CP16["yupu"]["gran"] == "whole" and CP16["yupu"]["steady_K"] == "11"
assert abs(float(CP16["yupu"]["plat_4d"]) - float(CIV[("yupu", "whole", "crane", "reserve")]["plat"])) < 1e-9
groups =[("Main mix", [("yupu", "main", "whole"), ("yupu", "main", "stops"), ("yantai", "main", "whole"),
                        ("yantai", "main", "stops")]),
          ("Erection-heavy mix", [("yupu", "erection", "whole"), ("yupu", "erection", "stops")]),
          ("Crane-paced erection", [("yupu", "crane", "whole"), ("yantai", "crane", "whole")])]
R5 = []                 # one row each: lower end, upper end, the other two rules, both ends with 30-min handling
for gname, cases in groups:
    for (y, kind, sec) in cases:
        if kind == "crane":
            r_ = CIV[(y, "whole", "crane", "reserve")]
            low = float(CP16[y]["plat_16d"]) if y in CP16 else float(r_["plat"])
            up, safe, ref = float(r_["T1"]), None, None
        else:
            up, low = float(I27[(y, kind, sec)]["T1_sys"]), R27E[f"{y}|{kind}|{sec}|reserve"]
            safe, ref = R27E[f"{y}|{kind}|{sec}|segment_safe"], R27E[f"{y}|{kind}|{sec}|segment"]
        l30 = L30.get((y, sec, "reserve")) if kind == "main" else None
        if l30:
            assert abs(float(l30["plat_10"]) - low) < 0.5 and abs(float(l30["T1_10"]) - up) < 0.5
        R5.append(dict(group=gname, label=f"{YNAME[y]}, {'whole road' if sec == 'whole' else 'sectioned'}", low=low,
                       up=up, safe=safe, ref=ref, lo3=float(l30["plat_30"]) if l30 else None,
                       up3=float(l30["T1_30"]) if l30 else None))
# each row as a share of its own system bound with 10-min handling (= 100 %), so the length of a bar is the width of the
# interval; blocks/day and the share in two columns on the right; the rows in a label column, grouped by the mix
share = lambda r, v: 100 * v / r["up"]
yv, YP5, SEP5, GRP5 = 0.0, [], [], []
for gi, (gname, cases) in enumerate(groups):
    if gi:
        yv -= 0.45
        SEP5.append(yv + 0.72)
    ys = [yv - k for k in range(len(cases))]
    YP5 += ys
    GRP5.append((gname, statistics.mean(ys)))
    yv -= len(cases)
YTOP5 = 0.75
fig = plt.figure(figsize=(FIGW, 92 * MM), layout="constrained")
gs = fig.add_gridspec(1, 3, width_ratios=[40, 96, 26])
axl = fig.add_subplot(gs[0])
ax = fig.add_subplot(gs[1], sharey=axl)
axt = fig.add_subplot(gs[2], sharey=axl)
GLAB = {"Main mix": "Main mix", "Erection-heavy mix": "Erection-\nheavy mix", "Crane-paced erection": "Crane-paced\nerection"}
for gname, yc in GRP5:
    axl.text(0, yc, GLAB[gname], fontsize=FS_TEXT, color=INK, ha="left", va="center")
for r, y_ in zip(R5, YP5):
    axl.text(1, y_, r["label"], fontsize=FS_TEXT, color=INK2, ha="right", va="center")
    ax.barh(y_, 100 - share(r, r["low"]), left=share(r, r["low"]), height=0.36, color=C["interval_fill"], linewidth=0)
    ax.plot([100], [y_], marker="|", color=C["bound_system"], markersize=9, markeredgewidth=1.8, linestyle="")
    ax.plot([share(r, r["low"])], [y_], **rk("reserve", linestyle="", markersize=MS_PT, zorder=4))
    if r["safe"] is not None:
        ax.plot([share(r, r["safe"])], [y_], **rk("segment_safe", linestyle=""))
        ax.plot([share(r, r["ref"])], [y_], **rk("segment", linestyle=""))
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
axt.text(0.66, YTOP5, "blocks/day", fontsize=FS_NOTE, color=INK2, ha="right", va="bottom")
axt.text(1.0, YTOP5, "share", fontsize=FS_NOTE, color=INK2, ha="right", va="bottom")
# the reading of the text: a lower end close to the bound, the roads limit throughput; far below it, the rule does
for x0, x1, x2, s, ha in ((1, 23, 24, "rule limits", "left"), (100, 78, 77, "roads limit", "right")):
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
ax.set_xlim(0, 104)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("Share of the system bound (%)")
ax.tick_params(axis="y", left=False, labelleft=False)
ax.spines["left"].set_visible(False)
ax.grid(True, axis="x", color=GRID, linewidth=0.5)
ax.set_axisbelow(True)
# every mark named, column by column: the interval with 10-min handling, the other two rules, 30-min handling (the open
# circle is its lower end, the full-path reservation plateau; the thin bar runs up to its system bound, the short tick)
fig.legend(handles=[Line2D([], [], **rk("reserve", linestyle="", markersize=MS_PT)),
                    Line2D([], [], marker="|", color=C["bound_system"], markersize=9, markeredgewidth=1.8, linestyle=""),
                    Line2D([], [], color=C["interval_fill"], linewidth=6, solid_capstyle="butt"),
                    Line2D([], [], **rk("segment_safe", linestyle="")),
                    Line2D([], [], **rk("segment", linestyle="")),
                    Line2D([], [], **rk("reserve", linestyle="", markersize=MS - 0.8, markerfacecolor=PAPER)),
                    Line2D([], [], color=C["interval_fill_light"], linewidth=2.5, solid_capstyle="butt")],
           labels=["lower end: full-path reservation", "upper end: system bound", "capacity interval", "safe release",
                   "reference rule", "lower end, 30-min handling", "30-min handling: interval"],
           loc="outside upper center", ncol=3, fontsize=FS_NOTE, handlelength=1.6, columnspacing=1.2, handletextpad=0.5)
# the numbers drawn are those of the rows (read back from the figure): bar ends, the other two rules, both ends with
# 30-min handling, the two columns; and those the text quotes (numbers.tex)
xs = {round(float(x), 6) for ln in ax.lines for x in ln.get_xdata()}
bars = sorted((round(p.get_x(), 6), round(p.get_x() + p.get_width(), 6)) for p in ax.patches)
assert bars == sorted([(round(share(r, r["low"]), 6), 100.0) for r in R5]
                      + [(round(share(r, r["lo3"]), 6), round(share(r, r["up3"]), 6)) for r in R5 if r["lo3"]]), bars
assert all(round(share(r, r[k]), 6) in xs for r in R5 for k in ("low", "safe", "ref", "lo3", "up3") if r[k]) and 100 in xs
cols = [t.get_text() for t in axt.texts]
assert all(f"{grp(r['low'])}–{grp(r['up'])}" in cols and f"{share(r, r['low']):.0f}%" in cols for r in R5)
pc = lambda m: MACRO[m].replace("\\%", "%")
for r, low_m, pct_m, pct30_m in zip(R5, ("YuResLowWhole", "YuResLowSec", "YaResLowWhole", "YaResLowSec", "YuErResLowWhole",
                                         "YuErResLowSec", "YuCraneSixteenPlat", None),
                                    ("YuResLowWholeRatio", "YuResLowSecRatio", "YaResLowWholeRatio", "YaResLowSecRatio",
                                     "YuErResLowWholeRatio", "YuErResLowSecRatio", "YuCraneResRatio", "YaCraneResRatio"),
                                    ("YuLoadThirtyWholeRatio", "YuLoadThirtySecRatio", "YaLoadThirtyWholeRatio",
                                     "YaLoadThirtySecRatio", None, None, None, None)):
    assert (low_m is None or grp(r['low']).replace(",", "{,}") == MACRO[low_m]) and f"{share(r, r['low']):.0f}%" == pc(pct_m), r
    assert pct30_m is None or f"{100 * r['lo3'] / r['up3']:.0f}%" == pc(pct30_m), r
SRC_INTERVAL = "results/interval/interval.csv, handling_30min.csv, crane_interval.csv, crane_16day_plateau.csv; results/saturation/rules.csv"
save(fig, "fig_interval.pdf", SRC_INTERVAL)

# ================================================================== Figure 6: required fleet
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
        if h == 24.0:                     # two lines at the left, clear of safe release rising past 450 blocks/day
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
            ax.set_xlabel("Daily demand (blocks/day)")
        if j == 0:
            ax.set_ylabel("Required fleet (vehicles)")
save(fig, "fig_workzone.pdf", "results/fleet/required_fleet.csv, fleet_8day.csv; results/runs/yantai_crane_fleet/demand.csv "
     "(one Yantai cell); results/saturation/kstar.csv")

# ================================================================== Figure 7: screening by demand over the bound
SCRN = [r for r in rows("results/fleet/screen.csv") if r["gap"]]      # fleet-table cells and the low-demand cells
CRS = [r for r in rows("results/fleet/screen_cross.csv") if r["measure"] == "gap" and r["yard"] != "all"]
# first 10 % crossings: the band from the integer fleets (gap), the ticks from the fractional fleets (gap_frac)
FRX ={(r["yard"], float(r["day_h"])): 100 * float(r["first"]) for r in rows("results/fleet/screen_cross.csv")
       if r["measure"] == "gap_frac" and r["yard"] != "all"}
assert (f"{FRX[('yupu', 16.0)]:.1f}%" == f"{FRX[('yupu', 24.0)]:.1f}%" == pc("ScreenFracYu")
        and f"{min(FRX[('yantai', h)] for h in (16.0, 24.0)):.1f}%" == pc("ScreenFracYaLo")
        and f"{max(FRX[('yantai', h)] for h in (16.0, 24.0)):.1f}%" == pc("ScreenFracYaHi"))
band_ = (100 * min(float(r["first"]) for r in CRS), 100 * max(float(r["last"]) for r in CRS))
assert f"{band_[0]:.1f}%" == pc("ScreenFirstLo") and f"{band_[1]:.1f}%" == pc("ScreenFirstHiOne")
assert f"{band_[1]:.0f}%" == pc("ScreenFirstHi") == pc("ScreenAllAbove")
SERIES = [(y, h) for y in YARDS for h in (16.0, 24.0)]                   # top to bottom
PTS = {}                                    # (yard, day) -> [(demand / bound in %, extra fleet as a fraction)], by ratio
for y, h in SERIES:
    sysb = float(SYS[y]["main"]["T1_system"])
    idx = {(r["model"], int(float(r["demand"]))): r["K_req"] for r in DEM
           if r["yard"] == y and float(r["day_h"]) == h and r["section"] == "whole"}
    t5 = {n: int(idx[("reserve", n)]) / int(idx[("free", n)]) - 1
          for (m, n) in idx if m == "reserve" and idx[("reserve", n)].isdigit() and idx[("free", n)].isdigit()}
    rs = [r for r in SCRN if r["yard"] == y and float(r["day_h"]) == h]
    # integer fleets: the cells of the required-fleet table as in the workzone figure (Okpo's headline cells are
    # eight-day values; only 350 blocks/day on 24-h days differs from the four-day row of screen.csv), and the
    # low-demand levels
    odd = [r for r in rs if r["src"] == "table5" and abs(t5[int(r["demand"])] - float(r["gap"])) > 1e-9]
    assert [(y, h, r["demand"]) for r in odd] == ([("yupu", 24.0, "350")] if (y, h) == ("yupu", 24.0) else [])
    assert all(abs(float(r["ratio"]) - n / (sysb * h / 16)) < 1e-6 for r in rs for n in [int(r["demand"])])
    PTS[(y, h)] = sorted((100 * float(r["ratio"]), t5[int(r["demand"])] if r["src"] == "table5" else float(r["gap"]))
                         for r in rs)
# the screen as the text states it: below the band every tested level needs the same fleet, above it every level more
# than 10 % more; integer fleets take no value in (0, 10 %] at any of the 54 tested levels
allp = [p for s in SERIES for p in PTS[s]]
assert len(allp) == 54 and all(g == 0 or g > 0.1 for _, g in allp)
assert all(g == 0 for x, g in allp if x < band_[0]) and all(g > 0.1 for x, g in allp if x > band_[1])
own = {"yupu": 100 * num("OkOwn") / float(SYS["yupu"]["main"]["T1_system"]),
       "yantai": 100 * num("YaOwn") / float(SYS["yantai"]["main"]["T1_system"])}
assert f"{own['yupu']:.0f}%" == pc("YuDensOwn") and f"{own['yantai']:.1f}%" == pc("YaDensOwn")
# the screen as a signal band: one row per series, each tested level marked "same fleet as free flow" (grey circle) or
# "more than 10 % more" (series marker), against the ratio on a logarithmic axis, which spreads the low levels; three
# zones (below the band of first 10 % crossings, the band, above it), the first crossing with fractional fleets as a
# tick, each yard's own volume on its 16-h row, and above the zones the planning reading of the discussion
LO, HI, XLO, XHI = *band_, 1.0, 50.0
ZONES = [(XLO, LO, PAPER, "same fleet\nas free flow"), (LO, HI, BAND, "first crossings\nof 10%"),
         (HI, XHI, C["extra_fleet_fill"], "more than 10% more\nat every tested level")]
ROW = {s: len(SERIES) - 1 - i for i, s in enumerate(SERIES)}
fig, ax = plt.subplots(figsize=(FIGW, 74 * MM), layout="constrained")
for x0, x1, col, lab in ZONES:
    ax.axvspan(x0, x1, color=col, linewidth=0, zorder=0)
    ax.text((x0 * x1) ** 0.5, 4.42, lab, fontsize=FS_NOTE, color=INK, ha="center", va="top", zorder=5)
for x in (LO, HI):
    ax.axvline(x, color=MUTED, linewidth=LW_THIN, zorder=0.5)
tr = ax.get_xaxis_transform()
ax.text((XLO * LO) ** 0.5, 1.06, "free-flow sizing suffices", fontsize=FS_NOTE, color=INK2, ha="center", va="center",
        style="italic", transform=tr, clip_on=False)
ax.annotate("", xy=(LO * 1.01, 1.06), xytext=(XHI, 1.06), xycoords=tr, textcoords=tr,
            arrowprops={"arrowstyle": "|-|", "lw": LW_THIN, "color": INK2, "shrinkA": 0, "shrinkB": 0, "mutation_scale": 3})
ax.text((LO * XHI) ** 0.5, 1.06, " simulate the traffic rule with the fleet ", fontsize=FS_NOTE, color=INK2, ha="center",
        va="center", style="italic", transform=tr, clip_on=False, bbox={"facecolor": PAPER, "edgecolor": "none", "pad": 0.3})
for (y, h), yv in ROW.items():
    same = [x for x, g in PTS[(y, h)] if g == 0]
    more = [x for x, g in PTS[(y, h)] if g > 0]
    ax.plot(same, [yv] * len(same), linestyle="", marker="o", markersize=MS, color=MUTED, markerfacecolor=PAPER,
            markeredgewidth=0.9, zorder=3, gid=f"{y}|{h}|same")
    ax.plot(more, [yv] * len(more), linestyle="", marker=YMK[y], markersize=MS_PT, color=YCOL[y],
            markerfacecolor=YCOL[y] if h == 16.0 else PAPER, markeredgewidth=0.9, zorder=3, gid=f"{y}|{h}|more")
    ax.plot([FRX[(y, h)]] * 2, [yv - 0.3, yv + 0.3], color=INK, linewidth=1.1, solid_capstyle="butt", zorder=4,
            gid=f"{y}|{h}|frac")
    if h == 16.0:                                        # own volume, as a ratio on 16-h days, on that row
        ax.plot([own[y]] * 2, [yv - 0.42, yv + 0.42], color=YCOL[y], linewidth=1.2, linestyle=(0, (1.5, 1)), zorder=4)
        ax.text(own[y] * 1.05, yv - 0.5, f"{YNAME[y]}'s own volume, {pc({'yupu': 'YuDensOwn', 'yantai': 'YaDensOwn'}[y])}",
                fontsize=FS_NOTE, color=YCOL[y], ha="left", va="center", zorder=5,
                bbox={"facecolor": PAPER, "edgecolor": "none", "pad": 0.3, "alpha": 0.85})
s0 = SERIES[0]                                           # the fractional crossing named once, on the top row
ax.annotate("first crossing,\nfractional fleets", xy=(FRX[s0], ROW[s0] - 0.3), xytext=(FRX[s0] * 0.62, ROW[s0] - 0.62),
            fontsize=FS_NOTE, color=INK2, ha="right", va="center",
            arrowprops={"arrowstyle": "-", "lw": LW_THIN, "color": INK2, "shrinkA": 1, "shrinkB": 1})
ax.set_xscale("log")
ax.set_xlim(XLO, XHI)
ax.xaxis.set_major_locator(FixedLocator([1, 2, LO, HI, 20, 40]))
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xticklabels(["1", "2", f"{LO:.1f}", f"{HI:.0f}", "20", "40"])
ax.set_xlabel("Daily demand / system bound (%, logarithmic)")
ax.set_yticks(list(ROW.values()))
ax.set_yticklabels([f"{YNAME[y]}, {int(h)}-h day" for y, h in ROW])
ax.tick_params(axis="y", length=0)
ax.set_ylim(-0.6, 4.5)
ax.spines["left"].set_visible(False)
more_h = tuple(Line2D([], [], linestyle="", marker=YMK[y], markersize=MS_PT, color=YCOL[y]) for y in YARDS)
fig.legend(handles=[Line2D([], [], linestyle="", marker="o", markersize=MS, color=MUTED, markerfacecolor=PAPER,
                           markeredgewidth=0.9), more_h],
           labels=["same fleet as free flow", "full-path reservation needs more than 10% more (open: 24-h day)"],
           handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)}, loc="outside upper center", ncol=2, fontsize=FS_TEXT)
# the numbers drawn are those of the series (read back from the figure)
got = {ln.get_gid(): sorted(ln.get_xdata()) for ln in ax.lines if ln.get_gid()}
for y, h in SERIES:
    assert got[f"{y}|{h}|same"] == [x for x, g in PTS[(y, h)] if g == 0]
    assert got[f"{y}|{h}|more"] == [x for x, g in PTS[(y, h)] if g > 0] and got[f"{y}|{h}|frac"] == [FRX[(y, h)]] * 2
SRC_SCREEN = "results/fleet/screen.csv, screen_cross.csv (gap, gap_frac); table cells from results/fleet/required_fleet.csv, results/runs/yantai_crane_fleet/demand.csv (one Yantai cell), results/fleet/fleet_8day.csv (Okpo's own volume, eight days) and results/bounds/*/system.csv"
save(fig, "fig_screen.pdf", SRC_SCREEN)

# ================================================================== Figure 8: dock-mouth thresholds at the two yards
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

# ================================================================== Figure 9(b, c): the dock-mouth mechanism (synthetic network)
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

# ================================================================== Figure 10: external anchors against handling time
# the numbers of the anchor table (make_numbers.py tab_anchor): Okpo with 30 vehicles at saturation, blocks per
# vehicle-day with t min each of loading and unloading on 24-h days (handling_time.csv), the four rules at the main
# handling time on 16-h days (per_vehicle_rates.csv, column r1833); safe release was run at the main handling time only
AF = rows("results/anchors/handling_time.csv")
S72 = {r["macro"]: float(r["r1833"]) for r in rows("results/anchors/per_vehicle_rates.csv")}
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
ax.set_ylabel("Throughput per vehicle (blocks/day)")
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
save(fig, "fig_anchors.pdf", "results/anchors/handling_time.csv, per_vehicle_rates.csv (column r1833)")

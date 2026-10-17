"""The two ends of the screen of demand over the bound. Upper end: the chain statistics, i.e. which load of the booked chain
the fleet gap of full-path reservation over free flow follows. Lower end: representative low-load cells checked without a
long-run backlog (K_stable of the 30-day runs of experiments/screen_stable.py against the four-day K_req), and the own volume
of Okpo under the intra-area mapping. Reads released tables only; no simulation, no linear programme.

Upper end. Curves: the 14 curves of the screen, 10 of the proportional mapping (Okpo and Yantai at 16 h and 24 h, road width -1.5 m and
+1.5 m at 16 h in both yards, intra-area moves at Okpo at 16 h and 24 h) and 4 of the conserved mapping of the main case (both
yards, 16 h and 24 h); each level of a curve has its demand ratio x = n / lambda^S and its fractional fleet gap
(K_res_frac / K_free_frac - 1). Statistics of the chain on x, with u_r = x beta_r (chain_load):
  chain   loaded chain (weights pi; empty part of u by the minimum empty flow) / full booked chain (weights pi + eps_d;
          dispatch-like empty runs);
  row     k0 width-weighted / k1 exclusive (packing row k = 1: a vehicle wider than half the road counts as the whole road,
          i.e. the time a narrow road, a door or a junction is held);
  form    S = sum w u; L = -sum w ln(1 - u); Q = sum w u / (1 - u) (queueing delay); TS, TL, TQ: the same three forms weighted
          by the occupancy time of a task on r and divided by the cycle without waiting.
The width-weighted S, L, S_d, L_d, T_d and T of the curve tables are recomputed and must be equal (assertion).
Upper end = the last crossing of 10 % by the fractional fleet gap (every level above it exceeds 10 %; cross10), with a
bootstrap over the seeds (2,000 draws, Random(7), the same draws for every curve and level). Alignment across the yards at
16 h (each yard: proportional, -1.5 m, +1.5 m, main case): ratio of the medians of the upper ends (Yantai / Okpo), whether the
two ranges overlap, the relative spread within each yard, intra-area moves at 16 h against the Okpo median, the pairwise
differences with their bootstrap intervals, and R^2 of a pooled isotonic fit of the fleet gap on the statistic.

Lower end (PARTS: the three 30-day runs results/runs/okpo_screen_stable_main, yantai_screen_stable_main,
okpo_screen_stable_proportional). For every cell (30-day scenario x level): K_stable of free flow and full-path reservation
recomputed from daily.csv with the search of screen_stable (stable_fleet.search from stable_fleet_conserved.k0_of) and
checked against stable.json; the backlog slope at K_req and at K_stable, the mean end-of-day backlog of day 30; the
four-day K_req and fractional fleet of the same cell with 10 seeds (the screen runs okpo_main_screen, yantai_main_screen,
okpo_screen_widths, okpo_screen_low, okpo_intra_screen) and with 30 seeds (okpo_main_screen_30seeds,
yantai_main_screen_30seeds, okpo_screen_30seeds, okpo_intra_screen_30seeds); the demand ratio n / lambda^S (main case:
lambda^S of results/main_case/scan.csv at q = 0; proportional mapping: lam16 of results/screen/chain_load.json). Own volume of
Okpo under the intra-area mapping (339, 350 tasks a day, 16 h and 24 h): K_req and K_stable of results/runs/okpo_stable_30day
(stable.json) and the fractional fleets of results/intra_area/screen.csv.

Run:   python experiments/chain_statistics.py --out <dir> [--series <curve tables>] [--chain-load <file or dir>]
                [--chain-conserved <file or dir>] [--points yupu=<points.csv> ... yantai=<points.csv> ...]
Without options other than --out everything is read from the release (paths relative to the repository root):
--series: the curve tables (one row per curve and level: series, chain, day_h, n, ratio, gap_frac and the width-weighted
  statistics); default results/screen/series.csv and results/screen/series_main_case.csv.
--chain-load: the per-resource chain loads of the nine cases of experiments/chain_load.py; default the released
  results/screen/chain_load.json (the fields Chain reads), or the output directory of a rerun `python
  experiments/chain_load.py --out <dir> --sensitivity results/bounds/sensitivity.csv --mapping-bounds
  results/intra_area/bounds.csv` (parts/<case>.json, combined into <dir>/chain.json on reading).
--chain-conserved: the chain loads of the conserved mapping (cases yupu_cons, yantai_cons); default the released
  results/screen/chain_conserved.json, or the output directory of `python experiments/route_free_mappings.py --out <dir>
  --conserved-scan results/main_case/scan.csv` (chain.json).
--points (for the bootstrap; without it the intervals are left empty): the per-point tables of the four-day screen runs,
  yard=FILE (yupu or yantai; several files per yard; the same point in two files must agree); every level of every curve
  needs its points. Released: Okpo results/runs/okpo_screen_low, okpo_screen_widths, okpo_intra_screen, okpo_intra_fleet,
  okpo_main_screen, okpo_main_fleet; Yantai results/runs/yantai_screen_low, yantai_screen_widths, yantai_main_screen,
  yantai_fleet_plant (points.csv). Not released are the per-point tables behind the four proportional-mapping curves at 16 h
  and 24 h: the low-demand screen runs of both yards, the Okpo plant-level fleet searches (revision 1833) and the Yantai
  fleet searches of results/runs/yantai_crane_fleet (configuration and demand.csv only).
Writes <dir>/series_axes.csv (every level with every statistic), cross.csv (crossings per curve and statistic), align.csv,
pairs.csv, chain_stats.json (n_rows, n_boot, best: the statistic whose ratio of medians is closest to 1, align; the released
results/screen/chain_stats.json holds best and align), stable_low.csv and intra_own.csv (as the released
results/screen/stable_low.csv, intra_own.csv). The alignment readings (axis, ratio, overlap, r2), best and the two lower-end
tables do not depend on the bootstrap.
"""
import argparse
import collections
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
csv.field_size_limit(1 << 30)
import runner as F                 # noqa: E402
import chain_load as C             # noqa: E402
import route_free_mappings as XB   # noqa: E402
import stable_fleet as R49         # noqa: E402
import stable_fleet_conserved as R61   # noqa: E402
import conserved_bounds as TB      # noqa: E402
from screen_stable import cells    # noqa: E402

OUT = None
RUNS = ROOT / "results" / "runs"
CHAIN_LOAD = ROOT / "results" / "screen" / "chain_load.json"
SEEDS30 = tuple(range(101, 131))
# part -> (released 30-day run, yard, {30-day scenario: (scenario it was derived from, added parameters, levels,
#          (released four-day screen run with 10 seeds, its scenario))})
PARTS = {
    "cons_yupu": ("okpo_screen_stable_main", "yupu", {"d16_c_30d": ("d16_c_30d", {}, [75, 85, 100], ("okpo_main_screen", "d16_c"))}),
    "cons_yantai": ("yantai_screen_stable_main", "yantai", {"d16_c_30d": ("d16_c_30d", {}, [60, 70], ("yantai_main_screen", "d16_c"))}),
    "prop": ("okpo_screen_stable_proportional", "yupu",
             {"d16_w-1.5_30d": ("d16_30d", {"width_delta": -1.5}, [75], ("okpo_screen_widths", "d16_w-1.5")),
              "d16_w+1.5_30d": ("d16_30d", {"width_delta": 1.5}, [60], ("okpo_screen_low", "d16_w+1.5")),
              "d16_in_30d": ("d16_in_30d", {}, [150], ("okpo_intra_screen", "d16_in"))}),
}
NBOOT = 2000
SEEDS10 = list(F.SEEDS)
CHAINS = {"load": "loaded chain", "full": "full booked chain"}
ROWS = {"k0": "width-weighted", "k1": "exclusive (k = 1)"}
FORMS = {"S": "Σu", "L": "−Σln(1−u)", "Q": "Σu/(1−u)", "TS": "time-weighted Σu", "TL": "time-weighted −Σln(1−u)",
         "TQ": "time-weighted Σu/(1−u)"}
AXES = ["ratio"] + [f"{c}_{r}_{f}" for c in CHAINS for r in ROWS for f in FORMS]
OLD = {"load_k0_S": "S", "load_k0_L": "L", "full_k0_S": "S_d", "full_k0_L": "L_d", "full_k0_TS": "T_d", "load_k0_TS": "T"}   # curve tables
LABEL = {"okpo_16": "Okpo·proportional·16 h", "okpo_24": "Okpo·proportional·24 h", "yantai_16": "Yantai·proportional·16 h",
         "yantai_24": "Yantai·proportional·24 h", "okpo_w-1.5": "Okpo·−1.5 m", "okpo_w+1.5": "Okpo·+1.5 m", "yantai_w-1.5": "Yantai·−1.5 m",
         "yantai_w+1.5": "Yantai·+1.5 m", "intra_16": "Okpo·intra-area·16 h", "intra_24": "Okpo·intra-area·24 h",
         "okpo_c16": "Okpo·main case·16 h", "okpo_c24": "Okpo·main case·24 h", "yantai_c16": "Yantai·main case·16 h",
         "yantai_c24": "Yantai·main case·24 h"}
YARD16 = {"yupu": ["okpo_16", "okpo_w-1.5", "okpo_w+1.5", "okpo_c16"], "yantai": ["yantai_16", "yantai_w-1.5", "yantai_w+1.5", "yantai_c16"]}
PAIRS = [("okpo_16", "yantai_16", "proportional·16 h"), ("okpo_c16", "yantai_c16", "main case·16 h"), ("okpo_24", "yantai_24", "proportional·24 h"),
         ("okpo_c24", "yantai_c24", "main case·24 h")]
# scenario of the four-day runs behind each curve (per-point tables, --points)
SCEN = {"okpo_16": "d16", "okpo_24": "d24", "yantai_16": "d16", "yantai_24": "d24", "okpo_w-1.5": "d16_w-1.5", "okpo_w+1.5": "d16_w+1.5",
        "yantai_w-1.5": "d16_w-1.5", "yantai_w+1.5": "d16_w+1.5", "intra_16": "d16_in", "intra_24": "d24_in", "okpo_c16": "d16_c",
        "okpo_c24": "d24_c", "yantai_c16": "d16_c", "yantai_c24": "d24_c"}


def ax_label(ax):
    if ax == "ratio":
        return "demand ratio x"
    c, r, f = ax.split("_")
    return f"{CHAINS[c]}·{ROWS[r]}·{FORMS[f]}"


def write(name, rows):
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:          # UTF-8 without BOM, LF: as the released tables
        w = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r)), lineterminator="\n")
        w.writeheader()
        w.writerows([{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, tuple, dict)) else v) for k, v in r.items()} for r in rows])


def read(f):
    return list(csv.DictReader(open(f, encoding="utf-8-sig")))


# ---------------------------------------------------------------- linear-programme outputs of the chain load
class Chain:
    def __init__(self, part):
        self.d = part
        self.lam16 = part["lam"] * 57600.0 / part["day_s"]
        b, bd = np.array(part["beta"]), np.array(part["beta_d"])
        self.b0, self.bp, self.bd, self.bdp = b[0], b.max(axis=0), bd[0], bd.max(axis=0)
        self.pi, self.eps_d = np.array(part["pi"]), np.array(part["eps_d"])
        self.dur, self.dur_d = np.array(part["dur"]), np.array(part["dur_d"])
        self.tau = np.where(self.pi > 0, self.dur / np.maximum(self.pi, 1e-300), 0.0)
        self.C0, self.C0d = part["cycle0_s"], part["cycle0_d_s"]

    def stat(self, ax, x):
        if ax == "ratio":
            return x
        d, pk = ax.endswith(("_d", "_dp")), ax.endswith("p")
        w = self.pi + self.eps_d if d else self.pi
        b = (self.bdp if pk else self.bd) if d else (self.bp if pk else self.b0)
        u = x * b
        assert u.max() < 1.0, (ax, x, u.max())
        if ax[0] == "S":
            return float(w @ u)
        if ax[0] == "L":
            return float(w @ -np.log1p(-u))
        return float(u @ (self.dur + self.dur_d) / self.C0d) if d else float(u @ self.dur / self.C0)

    def inverse(self, ax, z):
        """Inverse of stat(ax, .) (increasing; bisection to 1e-12)."""
        lo, hi = 0.0, 0.999 / max(self.bdp.max(), self.bp.max())
        for _ in range(100):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if self.stat(ax, mid) < z else (lo, mid)
        return (lo + hi) / 2


def packed(d, name, build):
    """d/name: the combined file (compact JSON). Where the resumable cache parts/ exists it is rewritten from parts (left as
    it is when the content is the same), in the listing order of the parts directory, so the order of floating-point sums
    is that of reading the parts one by one; without parts the combined file is read."""
    f, ps = d / name, list((d / "parts").glob("*.json"))
    if ps:
        txt = json.dumps(build(ps), ensure_ascii=False, separators=(",", ":"))
        if not f.exists() or f.read_text(encoding="utf-8") != txt:
            f.write_text(txt, encoding="utf-8")
    return json.loads(f.read_text(encoding="utf-8"))


def chain_parts(d):
    """The linear-programme output of every case of chain_load {case: part}: the file d (the released
    results/screen/chain_load.json), or d/chain.json of a run directory (see packed)."""
    if d.is_file():
        return json.loads(d.read_text(encoding="utf-8"))
    return packed(d, "chain.json", lambda ps: {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in ps})


def chains(d):
    parts = chain_parts(d)
    assert set(parts) == set(C.SCENS), sorted(parts)
    return {k: Chain(v) for k, v in parts.items()}, d


def conserved_chains(d):
    """The two chain-load cases of the conserved mapping: the file d (the released results/screen/chain_conserved.json), or
    d/chain.json of a run directory of route_free_mappings."""
    parts = json.loads((d / "chain.json" if d.is_dir() else d).read_text(encoding="utf-8"))
    return {y: Chain(parts[f"{y}_cons"]) for y in XB.YARDS}, parts


# ---------------------------------------------------------------- fleets of a curve level: criterion, bootstrap, crossings
def need_line(n, lift):
    """Target line (a function of the drawn seeds): 0.99 x the vehicles' daily demand = 0.99 x (n - lifts by gantry crane
    released per day in the statistics window of the drawn seeds), as required_fleet.evaluate. lift = {seed: released per
    day}; empty for scenarios without lifts, i.e. 0.99 x n."""
    return lambda ss: 0.99 * sum(n - lift.get(s, 0.0) for s in ss) / len(ss)


def kreq(byk, seeds, need):
    """The smallest computed K (all of `seeds` computed at it) whose mean throughput reaches the target line in {K: {seed:
    throughput}} (the definition of required_fleet; seeds may repeat, for the bootstrap). need is the line, or a function
    of the seeds giving it. None (censored) when no computed K reaches it."""
    line = need(seeds) if callable(need) else need
    for k in sorted(byk):
        v = byk[k]
        if all(s in v for s in seeds) and sum(v[s] for s in seeds) / len(seeds) >= line:
            return k
    return None


def kfrac(byk, seeds, need):
    """Fractional required fleet: the mean throughput over the seeds interpolated linearly between K_req - 1 and K_req to the
    target line (for K_req = 1 the lower end is K = 0, throughput 0). The integer K makes the percentages jump at small
    fleets (1 vehicle = 10 %-50 %); the fractional fleet smooths them. None without K_req or with K_req - 1 not complete."""
    k = kreq(byk, seeds, need)
    if k is None:
        return None
    line = need(seeds) if callable(need) else need
    mean = lambda K: sum(byk[K][s] for s in seeds) / len(seeds) if K in byk and all(s in byk[K] for s in seeds) else None
    lo, hi = (0.0 if k == 1 else mean(k - 1)), mean(k)
    return None if lo is None else k - 1 + (line - lo) / (hi - lo)


def num(x):
    return None if x in ("", None, "None") else float(x)


_PTS = {}


def load(d):
    """points.csv of a run -> {(scenario, rule, level): {K: {seed: (throughput, waiting, cycle, empty run)}}} (min per task);
    a prefix main_ of a scenario name is dropped (main_d16 is d16)."""
    if d not in _PTS:
        out = collections.defaultdict(lambda: collections.defaultdict(dict))
        for r in read(d / "points.csv"):
            out[(r["scen"].replace("main_", ""), r["model"], r["level"])][int(r["K"])][int(r["seed"])] = \
                (float(r["throughput"]), num(r.get("cyc_wait")), num(r.get("cyc_total")), num(r.get("cyc_empty")))
        _PTS[d] = out
    return _PTS[d]


def thr(byk):
    return {K: {s: x[0] for s, x in v.items()} for K, v in byk.items()}


_DRAWS = {}


def draws(seeds, n=2000):
    """Bootstrap draws of the seeds (Random(7); the same draws for every cell, so that the cells are paired)."""
    key = (tuple(seeds), n)
    if key not in _DRAWS:
        rng = random.Random(7)
        _DRAWS[key] = [[rng.choice(seeds) for _ in seeds] for _ in range(n)]
    return _DRAWS[key]


def boot_fleet(byk, need, D, seeds=None):
    """D: draws x seeds count matrix (seeds in the order of `seeds`, default SEEDS). Returns two arrays (K_req, fractional
    fleet), NaN where a draw is censored."""
    seeds = seeds or SEEDS
    Ks = sorted(byk)
    M = np.array([[byk[K].get(s, np.nan) for s in seeds] for K in Ks])
    miss = np.isnan(M).astype(float)
    ns = D.sum(axis=1, keepdims=True)
    present = (D @ miss.T) == 0
    means = (D @ np.nan_to_num(M).T) / ns
    line = need(seeds)
    ok = present & (means >= line)
    has = ok.any(axis=1)
    j = ok.argmax(axis=1)
    K = np.where(has, np.array(Ks, dtype=float)[j], np.nan)
    prev = np.array([Ks.index(k - 1) if k - 1 in Ks else -1 for k in Ks])
    pj = prev[j]
    rows = np.arange(len(D))
    lo = np.where(np.array(Ks)[j] == 1, 0.0, np.where((pj >= 0) & present[rows, np.maximum(pj, 0)], means[rows, np.maximum(pj, 0)], np.nan))
    hi = means[rows, j]
    frac = np.where(has, np.array(Ks, dtype=float)[j] - 1 + (line - lo) / (hi - lo), np.nan)
    return K, frac


def ci(xs):
    xs = sorted(x for x in xs if x is not None)
    return (xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]) if len(xs) >= 20 else (None, None)


SEEDS = None
SCREEN_THR = 0.10          # screen threshold: fleet gap (full-path reservation / free flow - 1) above 10 %


def cross10(pts, thr=SCREEN_THR):
    """pts = [(ratio, fleet gap)] sorted by ratio. first = the ratio where the gap first exceeds thr, last = the ratio where
    it last goes from <= thr to > thr (every level after it exceeds), both interpolated linearly between adjacent levels.
    Already above at the lowest level: kind = "below" (threshold below the measured range); never above: kind = "above";
    back to <= thr at the last level: last = None, kind = "ends_below"."""
    above = [g > thr for _, g in pts]
    if not any(above):
        return dict(first=None, last=None, kind="above")
    at = lambda i: pts[i - 1][0] + (thr - pts[i - 1][1]) / (pts[i][1] - pts[i - 1][1]) * (pts[i][0] - pts[i - 1][0])
    i = above.index(True)
    j = max((k for k, a in enumerate(above) if not a), default=None)
    if i == 0:
        return dict(first=None, last=None if j is None else (at(j + 1) if j + 1 < len(pts) else None), kind="below")
    if j == len(pts) - 1:
        return dict(first=at(i), last=None, kind="ends_below")
    return dict(first=at(i), last=at(j + 1), kind="cross")


# ---------------------------------------------------------------- the statistics of the upper end
class Arr:
    """The statistics on the linear-programme output of one case (Chain). Written as Chain.stat term by term (the
    width-weighted columns are equal bit for bit)."""

    def __init__(self, ch):
        self.b = {"load": np.array(ch.d["beta"]), "full": np.array(ch.d["beta_d"])}
        self.w = {"load": ch.pi, "full": ch.pi + ch.eps_d}
        self.dur = {"load": ch.dur, "full": ch.dur + ch.dur_d}
        self.C0 = {"load": ch.C0, "full": ch.C0d}

    def stat(self, ax, x):
        if ax == "ratio":
            return x
        c, r, f = ax.split("_")
        u = x * self.b[c][int(r[1])]
        assert u.max() < 1.0, (ax, x, u.max())
        g = u if f[-1] == "S" else (-np.log1p(-u) if f[-1] == "L" else u / (1 - u))
        return float(self.w[c] @ g) if len(f) == 1 else float(g @ self.dur[c] / self.C0[c])


def points(specs):
    """Per-seed throughputs {yard: {(scenario, rule, level): {K: {seed: throughput}}}} of the points.csv files given as
    yard=FILE (relative to the repository root); a prefix main_ of a scenario name is dropped (main_d16 is d16); a point in two
    files must agree."""
    out = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict)))
    for spec in specs:
        y, f = spec.split("=", 1)
        for r in read(ROOT / f):
            c, x = out[y][(r["scen"].replace("main_", ""), r["model"], r["level"])][int(r["K"])], float(r["throughput"])
            assert c.setdefault(int(r["seed"]), x) == x, (f, r["scen"], r["model"], r["level"], r["K"], r["seed"])
    return out


def series_rows(tables, CH, pts=None):
    """The levels of the curves from the curve tables (strings but the numbers used here), each with the per-seed
    throughputs of free flow and full-path reservation by fleet (f, v) and its target line where pts is given. The demand
    ratio of a level must be n / lambda^S of its chain."""
    rows = []
    for t in tables:
        for r in read(t):
            r.update(n=int(r["n"]), day_h=int(r["day_h"]), **{k: float(r[k]) for k in ("ratio", "gap_frac", *OLD.values())})
            ch = CH[r["chain"]]
            assert abs(r["ratio"] / (r["n"] / (ch.lam16 * r["day_h"] / 16)) - 1) < 1e-9, (r["series"], r["n"])
            if pts is not None:
                p = pts["yantai" if r["series"].startswith("yantai") else "yupu"]
                f, v = (p.get((SCEN[r["series"]], m, f"n{r['n']}")) for m in ("free", "reserve"))
                assert f and v, ("no points", r["series"], r["n"])
                r.update(seeds=sorted({s for k in f for s in f[k]}), f=f, v=v, need=need_line(r["n"], {}))
            rows.append(r)
    return rows


def series(chain_dir, cons_dir, tables, pts=None):
    """The levels of the 14 curves with every statistic; the width-weighted columns of the tables are checked bit for bit."""
    CH64, _ = chains(chain_dir)
    CH69, _ = conserved_chains(cons_dir)
    arr = {k: Arr(v) for k, v in CH64.items()} | {f"{y}_cons": Arr(v) for y, v in CH69.items()}
    rows = series_rows(tables, CH64 | {f"{y}_cons": v for y, v in CH69.items()}, pts)
    for r in rows:
        a = arr[r["chain"]]
        for ax in AXES[1:]:
            r[ax] = a.stat(ax, r["ratio"])
            if ax in OLD:      # bit for bit with numpy on MKL (the conda build); OpenBLAS (the pip wheel) differs in the last bits
                assert math.isclose(r[ax], r[OLD[ax]], rel_tol=1e-12), (r["series"], r["n"], ax)
    return rows


def bootstrap(rows):
    """The fractional fleet gap of every level of every curve per draw (the same Random(7) draws, seeds 101-110), and the
    upper end (last) of every draw on every statistic."""
    D = np.array([[d.count(x) for x in SEEDS10] for d in draws(SEEDS10, NBOOT)], dtype=float)
    one = np.ones((1, len(SEEDS10)))
    gaps = {}
    for r in rows:
        assert r["seeds"] == SEEDS10 if "seeds" in r else True
        (k0f,), (f0f,) = boot_fleet(r["f"], r["need"], one, SEEDS10)
        (k0r,), (f0r,) = boot_fleet(r["v"], r["need"], one, SEEDS10)
        assert abs(f0r / f0f - 1 - r["gap_frac"]) < 1e-12           # full sample = the stored point estimate
        kf, ff = boot_fleet(r["f"], r["need"], D, SEEDS10)
        kr, fr = boot_fleet(r["v"], r["need"], D, SEEDS10)
        gaps[(r["series"], r["n"])] = fr / ff - 1
    out = {}
    for s in LABEL:
        rs = [r for r in rows if r["series"] == s]
        G = np.array([gaps[(s, r["n"])] for r in rs])
        for ax in AXES:
            z = [r[ax] for r in rs]
            out[(s, ax)] = [cross10([(zz, g) for zz, g in zip(z, G[:, i]) if not math.isnan(g)])["last"] for i in range(NBOOT)]
    return out, gaps


def crosses(rows, boot):
    out = []
    for s in LABEL:
        rs = [r for r in rows if r["series"] == s]
        for ax in AXES:
            c = cross10([(r[ax], r["gap_frac"]) for r in rs])
            lo, hi = ci(boot[(s, ax)])
            out.append(dict(series=s, label=LABEL[s], axis=ax, first=c["first"], last=c["last"], kind=c["kind"], last_lo=lo, last_hi=hi,
                            n_boot=sum(b is not None for b in boot[(s, ax)])))
    return out


def align(rows, cx, boot):
    """Per statistic: ratio of the medians of the upper ends of the four curves of each yard at 16 h (Yantai / Okpo, 1 =
    aligned), the two ranges, the relative spread within each yard; intra-area moves at 16 h against the Okpo median; the
    pairwise differences with bootstrap intervals; R^2 of the pooled isotonic fit (every level of the 14 curves, fractional
    fleet gap on the statistic)."""
    from sklearn.isotonic import IsotonicRegression
    get = {(r["series"], r["axis"]): r for r in cx}
    y = np.array([r["gap_frac"] for r in rows])
    out, prs = [], []
    for ax in AXES:
        lv = {k: [get[(s, ax)]["last"] for s in v] for k, v in YARD16.items()}
        mo, my = float(np.median(lv["yupu"])), float(np.median(lv["yantai"]))
        x = np.array([r[ax] for r in rows])
        yh = IsotonicRegression().fit(x, y).predict(x)
        out.append(dict(axis=ax, label=ax_label(ax), okpo_med=mo, yantai_med=my, ratio=my / mo, okpo_lo=min(lv["yupu"]), okpo_hi=max(lv["yupu"]),
                        yantai_lo=min(lv["yantai"]), yantai_hi=max(lv["yantai"]), overlap=max(lv["yupu"]) >= min(lv["yantai"]),
                        okpo_range=(max(lv["yupu"]) - min(lv["yupu"])) / mo, yantai_range=(max(lv["yantai"]) - min(lv["yantai"])) / my,
                        intra16_vs_okpo=get[("intra_16", ax)]["last"] / mo, r2=1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()))
        for a, b, lab in PAIRS:
            pa, pb = get[(a, ax)]["last"], get[(b, ax)]["last"]
            xs = [(q - p) / ((q + p) / 2) for p, q in zip(boot[(a, ax)], boot[(b, ax)]) if p is not None and q is not None]
            lo, hi = ci(xs)
            prs.append(dict(pair=lab, axis=ax, a=pa, b=pb, rel=(pb - pa) / ((pa + pb) / 2), lo=lo, hi=hi, aligned=None if lo is None else bool(lo <= 0 <= hi)))
    return out, prs


# ---------------------------------------------------------------- the lower end without a long-run backlog
def config(part):
    """The stored configuration of a part (config.json of its released 30-day run)."""
    return json.loads((RUNS / PARTS[part][0] / "config.json").read_text(encoding="utf-8"))


def kreq_dir(batch, y):
    """The released four-day screen run (10 seeds) of a cell (PARTS)."""
    return RUNS / batch


def daily_pts(d):
    """daily.csv of a 30-day run -> {(scenario, rule, level, K, seed): {"daily": {"backlog": [...]}}}, for
    stable_fleet.evaluate."""
    return {(r["scen"], r["model"], r["level"], int(r["K"]), int(r["seed"])): {"daily": {"backlog": [int(v) for v in r["backlog"].split()]}}
            for r in read(d / "daily.csv")}


def kreq30(part, s30, n):
    """Four-day required fleets (free flow, full-path reservation) and fractional fleets of the same cell with 30 seeds: the
    reruns with 30 seeds (main case: <yard>_main_screen_30seeds; proportional mapping: okpo_screen_30seeds; intra-area moves:
    okpo_intra_screen_30seeds)."""
    _, y, scen = PARTS[part]
    s4 = scen[s30][3][1]
    d = RUNS / (f"{'okpo' if y == 'yupu' else 'yantai'}_main_screen_30seeds" if part.startswith("cons") else
                ("okpo_intra_screen_30seeds" if s4 == "d16_in" else "okpo_screen_30seeds"))
    pts, need, ss = load(d), need_line(n, {}), list(SEEDS30)
    f, v = thr(pts[(s4, "free", f"n{n}")]), thr(pts[(s4, "reserve", f"n{n}")])
    return dict(dir30=f"runs/{d.name}", K_free30=kreq(f, ss, need), K_res30=kreq(v, ss, need), frac_free30=kfrac(f, ss, need), frac_res30=kfrac(v, ss, need))


def kreq10(part, s30, n):
    """Four-day fractional fleets of the same cell with 10 seeds (the measure of the screen): the four-day screen run."""
    _, y, scen = PARTS[part]
    b, s4 = scen[s30][3]
    pts, need = load(kreq_dir(b, y)), need_line(n, {})
    f, v = thr(pts[(s4, "free", f"n{n}")]), thr(pts[(s4, "reserve", f"n{n}")])
    return dict(frac_free10=kfrac(f, SEEDS10, need), frac_res10=kfrac(v, SEEDS10, need), K_free10=kreq(f, SEEDS10, need), K_res10=kreq(v, SEEDS10, need))


LAM = {"cons_yupu": ("yupu", 16), "cons_yantai": ("yantai", 16)}


def lam(y, h):
    """lambda^S of the conserved mapping (tasks a day, h-hour working day; 24 h is 1.5 x 16 h): q = 0 of the released
    results/main_case/scan.csv."""
    return TB.q0_lam(ROOT / "results" / "main_case" / "scan.csv")[y] * h / 16


def ratio_of(part, s30, n):
    if part in LAM:
        return n / lam(*LAM[part])
    ch, _ = chains(CHAIN_LOAD)
    return n / ch[{"d16_w-1.5_30d": "yupu_w-1.5", "d16_w+1.5_30d": "yupu_w+1.5", "d16_in_30d": "yupu_intra"}[s30]].lam16


def stable_part():
    rows = []
    for part in PARTS:
        cfg = config(part)
        d = RUNS / PARTS[part][0]
        st = json.loads((d / "stable.json").read_text(encoding="utf-8"))
        pts = daily_pts(d)
        cell = collections.defaultdict(dict)
        for s, m, lvl in cells(cfg):
            key = f"{s}|{m}|{lvl}"
            ev = R49.evaluate(cfg, pts, s, m, lvl)
            k0 = R61.k0_of(ev, st[key]["K_req"])
            res = R49.search(ev, k0)[1]
            assert k0 == st[key]["k0"] and all(res[c] == st[key][c] for c in res), (part, key)     # daily.csv recomputed = stable.json
            Ks = st[key]["stable_tol"]
            day30 = lambda K: float(np.mean([pts[(s, m, lvl, K, sd)]["daily"]["backlog"][-1] for sd in cfg["seeds"]]))
            cell[(s, lvl)][m] = dict(K_req=st[key]["K_req"], K_stable=Ks, K_stable_int=st[key]["stable"], k0=k0, slope_req=ev[st[key]["K_req"]]["slope"],
                                     slope_req_lo=ev[st[key]["K_req"]]["lo"], slope_req_hi=ev[st[key]["K_req"]]["hi"], slope_st=ev[Ks]["slope"],
                                     rel_req=ev[st[key]["K_req"]]["rel"], day30_req=day30(st[key]["K_req"]), day30_st=day30(Ks), Ks_tested=sorted(K for K in ev))
        for (s, lvl), c in cell.items():
            n = cfg["demand"][lvl]
            rows.append(dict(part=part, dir=f"runs/{d.name}", scen=s, n=n, ratio=ratio_of(part, s, n), **kreq10(part, s, n), **kreq30(part, s, n),
                             **{f"{k}_{mm}": v for mm, cc in (("free", c["free"]), ("res", c["reserve"])) for k, v in cc.items()}))
    for r in rows:
        r.update(dK_req10=r["K_res10"] - r["K_free10"], dK_req30=r["K_res30"] - r["K_free30"], dK_stable=r["K_stable_res"] - r["K_stable_free"],
                 dfrac10=r["frac_res10"] - r["frac_free10"], dfrac30=r["frac_res30"] - r["frac_free30"])
        assert (r["K_free10"], r["K_res10"]) == (r["K_req_free"], r["K_req_res"])       # K_req of the four-day points = the start of the search
    return rows


def intra_own():
    """Own volume of Okpo under the intra-area mapping: K_req, K_stable (tolerance criterion) of the 30-day run
    okpo_stable_30day and the fractional fleets of the intra-area screen (16 h, 24 h)."""
    st = json.loads((RUNS / "okpo_stable_30day" / "stable.json").read_text(encoding="utf-8"))
    scr = {(int(r["day_h"]), int(r["demand"])): r for r in read(ROOT / "results" / "intra_area" / "screen.csv")}
    out = []
    for h in (16, 24):
        for n in (339, 350):
            f, v = st[f"d{h}_in_30d|free|n{n}"], st[f"d{h}_in_30d|reserve|n{n}"]
            s = scr[(h, n)]
            out.append(dict(day_h=h, n=n, ratio=float(s["ratio"]), K_req_free=f["K_req"], K_req_res=v["K_req"], K_stable_free=f["stable_tol"],
                            K_stable_res=v["stable_tol"], frac_free=float(s["K_free_frac"]), frac_res=float(s["K_res_frac"]), gap_frac=float(s["gap_frac"]),
                            gap_req=v["K_req"] / f["K_req"] - 1, gap_stable=v["stable_tol"] / f["stable_tol"] - 1))
    return out


def main():
    global OUT
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--series", nargs="+", default=["results/screen/series.csv", "results/screen/series_main_case.csv"],
                    help="curve tables (default: the released results/screen/series.csv, series_main_case.csv)")
    ap.add_argument("--chain-load", default="results/screen/chain_load.json",
                    help="chain loads of chain_load.py: a file, or a run directory (parts/ or chain.json)")
    ap.add_argument("--chain-conserved", default="results/screen/chain_conserved.json",
                    help="chain loads of the conserved mapping: a file, or a run directory of route_free_mappings.py (chain.json)")
    ap.add_argument("--points", nargs="+", default=None, help="yard=points.csv of the screen runs, for the bootstrap")
    args = ap.parse_args()
    np.seterr(divide="ignore", invalid="ignore")
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = series(ROOT / args.chain_load, ROOT / args.chain_conserved, [ROOT / t for t in args.series],
                  points(args.points) if args.points else None)
    if args.points:
        boot, gaps = bootstrap(rows)
        for r in rows:
            g = gaps[(r["series"], r["n"])]
            r["gap_frac_ci"] = ci(list(g[~np.isnan(g)]))
    else:
        boot = collections.defaultdict(list)            # no per-seed throughputs: crossings without intervals
    cx = crosses(rows, boot)
    al, prs = align(rows, cx, boot)
    best = min((a for a in al), key=lambda a: abs(math.log(a["ratio"])))
    st = stable_part()
    own = intra_own()
    keep = [k for k in rows[0] if k not in ("f", "v", "need", "seeds")]
    write("series_axes.csv", [dict({k: r.get(k) for k in keep}, curve_label=LABEL[r["series"]], gap_frac_ci=r.get("gap_frac_ci"))
                              for r in rows])
    write("cross.csv", cx)
    write("align.csv", al)
    write("pairs.csv", prs)
    write("stable_low.csv", st)
    write("intra_own.csv", own)
    F.write_json(OUT / "chain_stats.json", dict(n_rows=len(rows), n_boot=NBOOT if args.points else 0, best=best, align=al))
    print(f"{len(rows)} levels of {len(LABEL)} curves; best statistic {best['axis']} (ratio {best['ratio']:.3f}); {len(st)} lower-end "
          f"cells; tables in {OUT}")


if __name__ == "__main__":
    main()

"""Check figures/palette.json (written by make_figures.py) for colour-vision deficiencies and greyscale printing.

Each colour group that appears together in a figure is simulated with the matrices of Machado, Oliveira & Fernandes (2009,
IEEE TVCG 15(6):1291-1298) at severity 1.0 for protanopia, deuteranopia and tritanopia (applied to linear RGB), and the
smallest pairwise CIEDE2000 difference is reported for normal vision and each deficiency; greyscale is CIE L*. Asserts:
the four rules, the two yards, the lower markers of Fig. 5 and the four cell classes of Fig. 6(a) stay apart (min
dE00 >= 10 in every vision); the cell classes run from light to dark in steps of at least 10 L*; and the main colour
(full-path reservation) is at least 15 L* darker than every other rule.
Run after make_figures.py, from the repository root: python paper/scripts/check_palette.py [palette.json]
(paper/reproduce.py runs it).
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAL = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "figures" / "palette.json").read_text(encoding="utf-8"))
HEX = {k: v["hex"] for k, v in PAL["colours"].items()}
MACHADO = {"protan": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
           "deutan": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
           "tritan": [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]]}
GROUPS = {"rules": ["rule_free", "rule_reserve", "rule_safe", "rule_reference"],
          "yards": ["yard_okpo", "yard_yantai"],
          "rules and yards": ["rule_free", "rule_reserve", "rule_safe", "rule_reference", "yard_okpo", "yard_yantai"],
          "Fig. 9(b): rules and margin steps": ["rule_reserve", "rule_safe", "margin_0", "margin_1", "margin_2"],
          "Fig. 5: rules and offline dispatch": ["rule_reserve", "rule_safe", "rule_reference", "offline_dispatch"],
          "Fig. 5: rules and lower markers": ["rule_reserve", "rule_safe", "rule_reference", "offline_dispatch",
                                              "booked_random"],
          "Fig. 6: shading": ["extra_fleet_fill", "band", "strip"],
          "Fig. 6: cell classes": ["cell_same", "cell_frac_low", "cell_one_more", "cell_frac_high"],   # light to dark
          "Fig. 3: map": ["road", "binding", "stop", "zone_building", "zone_storage", "zone_assembly", "zone_dock"],
          "Fig. 3: binding vs map": ["binding", "road", "road_major", "zone_building", "zone_storage", "zone_assembly",
                                     "zone_dock"]}


def lin(h):
    c = np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lab(rgb_lin):
    xyz = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                    [0.0193339, 0.1191920, 0.9503041]]) @ rgb_lin / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def de2000(a, b):
    """CIEDE2000 (Sharma, Wu & Dalal 2005), kL = kC = kH = 1."""
    L1, a1, b1 = a
    L2, a2, b2 = b
    Cb = (np.hypot(a1, b1) + np.hypot(a2, b2)) / 2
    G = 0.5 * (1 - np.sqrt(Cb ** 7 / (Cb ** 7 + 25 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p, h2p = np.degrees(np.arctan2(b1, a1p)) % 360, np.degrees(np.arctan2(b2, a2p)) % 360
    dLp, dCp = L2 - L1, C2p - C1p
    dh = h2p - h1p if C1p * C2p else 0.0
    dh = dh - 360 if dh > 180 else dh + 360 if dh < -180 else dh
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh / 2))
    Lbp, Cbp = (L1 + L2) / 2, (C1p + C2p) / 2
    hs = h1p + h2p
    hbp = hs if C1p * C2p == 0 else (hs / 2 if abs(h1p - h2p) <= 180 else (hs + 360) / 2 if hs < 360 else (hs - 360) / 2)
    T = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    SL = 1 + 0.015 * (Lbp - 50) ** 2 / np.sqrt(20 + (Lbp - 50) ** 2)
    SC, SH = 1 + 0.045 * Cbp, 1 + 0.015 * Cbp * T
    RT = (-2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25 ** 7))
          * np.sin(np.radians(60 * np.exp(-(((hbp - 275) / 25) ** 2)))))
    return float(np.sqrt((dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2 + RT * (dCp / SC) * (dHp / SH)))


# self-check against Sharma et al. (2005), Table 1, pairs 1 and 17
assert abs(de2000((50, 2.6772, -79.7751), (50, 0, -82.7485)) - 2.0425) < 1e-4
assert abs(de2000((50, 2.5, 0), (73, 25, -18)) - 27.1492) < 1e-4

VISION = {"normal": lambda h: lab(lin(h))} | {k: (lambda m: lambda h: lab(np.clip(np.array(m) @ lin(h), 0, 1)))(m)
                                              for k, m in MACHADO.items()}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    worst = {}
    for g, keys in GROUPS.items():
        out = []
        for v, f in VISION.items():
            d, pair = min((de2000(f(HEX[a]), f(HEX[b])), (a, b)) for a, b in itertools.combinations(keys, 2))
            out.append(f"{v} {d:5.1f} ({pair[0]}/{pair[1]})")
            worst[(g, v)] = d
        print(f"{g}: min dE00 " + "; ".join(out))
    L = {k: lab(lin(h))[0] for k, h in HEX.items()}
    print("greyscale L*: " + ", ".join(f"{k} {L[k]:.0f}" for k in GROUPS["rules and yards"] + GROUPS["Fig. 9(b): rules and margin steps"][2:]))
    gap = min(L[k] - L["rule_reserve"] for k in GROUPS["rules"] if k != "rule_reserve")
    print(f"main colour vs other rules: L* gap >= {gap:.0f}")
    Lc = [L[k] for k in GROUPS["Fig. 6: cell classes"]]
    print("Fig. 6 cell classes, light to dark: L* " + ", ".join(f"{x:.0f}" for x in Lc))
    assert all(worst[(g, v)] >= 10 for g in ("rules", "yards", "Fig. 5: rules and offline dispatch", "Fig. 5: rules and lower markers",
                                             "Fig. 6: cell classes") for v in VISION), worst
    assert gap >= 15, gap
    assert all(a - b >= 10 for a, b in zip(Lc, Lc[1:])), Lc


if __name__ == "__main__":
    main()

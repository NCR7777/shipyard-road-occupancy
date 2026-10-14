"""Write highlights.txt: the highlights of highlights.tex as displayed, with the number macros of numbers.tex expanded.

Run from paper/ after make_numbers.py: python scripts/make_highlights.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MACROS = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\\xspace\}", (ROOT / "numbers.tex").read_text(encoding="utf-8")))


def shown(s):
    """Displayed text of a plain LaTeX fragment: number macros expanded, a few typographic commands resolved."""
    s = re.sub(r"\\(\w+)", lambda m: MACROS.get(m.group(1), m.group(0)), s)
    for a, b in (("{,}", ","), ("--", "\u2013"), ("\\%", "%"), ("~", " "), ("\\sep", ";"), ("$>$", ">")):
        s = s.replace(a, b)
    return " ".join(s.split())


if __name__ == "__main__":
    hl = [shown(x) for x in re.findall(r"\\item\s+(.*)", (ROOT / "highlights.tex").read_text(encoding="utf-8"))]
    assert 3 <= len(hl) <= 5 and all(len(h) <= 85 for h in hl), hl     # 3-5 highlights of at most 85 characters
    (ROOT / "highlights.txt").write_text("Highlights\n" + "".join(f"\u2022 {h}\n" for h in hl), encoding="utf-8")
    print("\n".join(hl))

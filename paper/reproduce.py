"""Reproduce the paper's numbers, tables, highlights and data figures from results/ (one command):

    python paper/reproduce.py

Writes paper/numbers.tex (one LaTeX macro per number of the text), paper/tables/*.tex, paper/highlights.txt and
paper/figures/*.pdf (with paper/figures/palette.json); PNG previews of the figures go to paper/build/. Every script checks its
readings with assertions as it goes. Figure 3 (the two yard maps) and the two schematics (Figs. 2 and 9a) are not drawn here:
see README.md, "Reproducing the paper".
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for s in ("make_numbers.py", "make_highlights.py", "make_figures.py"):
    print(f"== {s}", flush=True)
    subprocess.run([sys.executable, str(HERE / "scripts" / s)], cwd=HERE, check=True)

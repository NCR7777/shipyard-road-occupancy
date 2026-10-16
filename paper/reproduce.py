"""Reproduce the paper's numbers, tables, highlights and data figures from results/ (one command):

    python paper/reproduce.py

Writes paper/numbers.tex (one LaTeX macro per number of the text, the supplementary material included, e.g. the
hand-computed instances of Section S4), paper/tables/*.tex (the tables of results of the paper and of the supplementary
material), paper/highlights.txt and paper/figures/*.pdf (with paper/figures/palette.json); PNG previews of the figures go
to paper/build/. Every script checks its readings with assertions as it goes; check_palette.py last checks the
colours of palette.json for colour-vision deficiencies and greyscale print. Figure 3 (the two yards) and the diagrams
and schematics (Figs. 1, 2, 8(a) and 10) are not drawn here: see README.md, "Reproducing the paper".
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for s in ("make_numbers.py", "make_highlights.py", "make_figures.py", "check_palette.py"):
    print(f"== {s}", flush=True)
    subprocess.run([sys.executable, str(HERE / "scripts" / s)], cwd=HERE, check=True)

#!/usr/bin/env bash
# Deterministic build of the deck: figures + numbers from the data tables, then lualatex (deck and notes edition), then PNG renders for QC.
set -euo pipefail
cd "$(dirname "$0")"
python3 make_slide_figures.py 2>&1 | grep -v "findfont\|Pandas4Warning\|acc = np.sqrt" || true
# lualatex + fontspec gives Lato (matches the figures); without texlive-luatex the deck falls back to pdflatex + Helvetica
if kpsewhich luaotfload-main.lua >/dev/null 2>&1; then TEX=lualatex; else TEX=pdflatex; echo "texlive-luatex missing: building with pdflatex (Helvetica fallback); install with: sudo apt install texlive-luatex"; fi
for i in 1 2; do $TEX -interaction=nonstopmode -halt-on-error slides.tex >/dev/null; done
for i in 1 2; do $TEX -interaction=nonstopmode -halt-on-error -jobname=slides_notes "\def\shownotes{1}\input{slides.tex}" >/dev/null; done
python3 extract_notes.py
rm -rf render && mkdir -p render && pdftoppm -r 80 -png slides.pdf render/slide
echo "slides.pdf: $(pdfinfo slides.pdf | awk '/Pages/{print $2}') pages; slides_notes.pdf: $(pdfinfo slides_notes.pdf | awk '/Pages/{print $2}') pages; renders in render/"

# Deck: FIVE SENSORS. ONE DECISION.

The presentation is built, not described: `docs/slides/slides.tex` (13 core slides + 8 backup, 16:9), `style.tex`
(theme and components), figures from `make_slide_figures.py` (data: `docs/report/figures_src/*.json` and the real
frames of the jumping-jacks clip via `docs/poster/clip_assets.py`), numbers injected from `gen/numbers.tex`.

    bash docs/slides/build.sh      # figures + numbers, slides.pdf, slides_notes.pdf (notes on the right), render/*.png, SPEAKER_NOTES.md

Speaker notes: `SPEAKER_NOTES.md` (generated from the `\talk` blocks) or `slides_notes.pdf`.
Visual audit, one row per slide with sources and risks: `SLIDE_AUDIT.md`.
Typeface: Lato via LuaLaTeX (`sudo apt install texlive-luatex`); without it the build falls back to pdflatex + Helvetica and says so.

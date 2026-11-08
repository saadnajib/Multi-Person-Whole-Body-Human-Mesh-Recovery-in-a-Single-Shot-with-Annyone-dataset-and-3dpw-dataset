# Thesis report

## Current revision: 8 November 2026

The current editable report is [thesis.tex](thesis.tex), with its text in
[chapters/](chapters/). The revised PDF is
[thesis_revised_2026-11-08.pdf](thesis_revised_2026-11-08.pdf).

This revision uses shorter sentences, direct explanations and first person
for the author's decisions. It retains the technical method, source-backed
results, figures, references and AI-use disclosure. It also updates the video
method to cover temporal filtering, landmark alignment and aligned-depth
smoothing. The video extensions do not add new reconstruction scores.

The previous compiled supervisor revision had **102 pages**, including
appendices. There was no need to remove useful results to meet an approximate
105-page target. See [EDITORIAL_REVISION_2026-11-08.md](EDITORIAL_REVISION_2026-11-08.md)
for the changes, checks and remaining author-review items. The finished report
is **100 pages**, including front matter, appendices and bibliography.

## Build

From this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error thesis.tex
```

Requires pdflatex, biber, biblatex, TikZ/pgfplots, siunitx, KOMA-Script and the
other packages declared in `thesis.tex`. Run until references and citations
resolve. No shell escape is needed. `hardcover_titlepage.tex` is a separate
title-page source.

## Contents

The seven chapters cover the introduction, background, data conversion,
methodology, results, discussion and conclusion. Appendices A–G retain run
provenance, joint correspondence, flags, extra figures, pilot results,
diagnostics and converter checks.

- `numbers_*.tex`: measured values, with source comments; unchanged in this revision.
- `tables/`, `figures/`, `data/`: tables, illustrations and data behind the plots.
- `references.bib`: bibliography used by the report.
- [REPRODUCE.md](REPRODUCE.md): commands behind the recorded experiments.
- [SUPERVISOR_REVISION_2026-10-22.md](SUPERVISOR_REVISION_2026-10-22.md): historical revision notes.
- `draft_2026-10_original/` and dated older PDFs: previous versions, not the current report.

## Model names and measurement conventions

**Student** is the verified 30-epoch ViT-B model. **Final** is its ViT-L teacher.
**Base** is the earlier synthetic-only teacher checkpoint. **Multi-HMR-Anny**
is NAVER's released baseline. Full run mappings remain in Appendix A.

Keep V0 and V1 synthetic results separate. Report 3DPW results as errors against
converted Anny labels, not published SMPL-space scores. The selected synthetic
holdout was excluded from thesis training but may have been seen by the released
baseline. Video tracking counts, smoothness, overlay alignment and 3D accuracy
measure different things.

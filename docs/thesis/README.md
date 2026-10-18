# Master thesis: Multi-Person Human Mesh Recovery with the Anny Body Model

LaTeX source of the rewritten thesis. The plan behind it (what each chapter, table and figure holds, the writing
rules, the work split) is `REWRITE_PLAN.md`; the old draft is kept read-only in `draft_2026-10_original/`.

## Build

```
cd docs/thesis && latexmk -pdf -interaction=nonstopmode thesis.tex
latexmk -c          # remove the aux files, keep thesis.pdf
```

Needs pdflatex, biber, and the packages TikZ, pgfplots (`compat=1.18`), booktabs, siunitx, subcaption, longtable,
multirow, biblatex. No `-shell-escape` (TikZ externalisation is off). The build never breaks on a missing chapter:
`thesis.tex` reads chapters 2-5 through `\InputIfExists{file}{title}`, which prints a placeholder chapter while the
file does not exist, and `numbers.tex` reads each `numbers_*.tex` only if it exists. Undefined-reference warnings
are expected while chapters are being written.

## Layout

```
thesis.tex               main file: preamble (from the draft + pgfplots, subcaption, siunitx, longtable, \graphicspath),
                         title page, Eigenstaendigkeitserklaerung, chapters, lists, appendices, bibliography
numbers.tex              convention + \input of the five numbers_*.tex files
numbers_<chapter>.tex    headline numbers per chapter (data, method, diagnostics, results, background)
references.bib           the draft's 16 entries + plan 1.8 additions + the Background writer's entries (merged)
chapters/                00_abstract ... 08_conclusion, appendix_a ... appendix_e
tables/                  T01-T18 (one file per table of plan section 4); draft_*.tex = the draft's tables, verbatim,
                         each with the plan's verdict in its header (several are marked DELETE: reference only)
figures/img/             raster figures (docs/figures/*.jpg, demo montages, files sent from the cluster)
figures/plots/           pgfplots figures F15, F16, F18, F19, F20, F22 (placeholders say "data pending")
figures/tikz/            TikZ drawings F1-F5
figures/RPTU-*.pdf       title-page logos
data/                    CSVs read by the plots; data/README.md lists what is expected, CLUSTER_ANSWERS.md the column names
```

## Who owns which file (plan section 6)

| owner | files |
|---|---|
| Background writer | `chapters/02_background.tex`, `numbers_background.tex` (bib entries go to `references.bib`; `references_background.bib` was merged into it on 18 Oct and is not read by the build) |
| Data writer | `chapters/03_data.tex`, `numbers_data.tex`, `tables/T03`-`T06`, `figures/tikz/F02_pipeline.tex` |
| Method writer | `chapters/04_method.tex`, `numbers_method.tex`, `tables/T02`, `T07`, `T11`, `figures/tikz/F01`, `F03`, `F04`, `F05` |
| Diagnostics writer | `chapters/05_diagnostics.tex`, `numbers_diagnostics.tex`, `tables/T08`-`T10`, `T14`, `T18` |
| Results writer (step 2) | `chapters/06_results.tex`, `numbers_results.tex`, `tables/T12`, `T15`, `T16`, `T17`, `figures/plots/*`, `data/`; owns `numbers.tex` and merges additions |
| Discussion writer (step 3) | `chapters/07_discussion.tex`, `tables/T01_requirements.tex` |
| Appendices (step 3) | `chapters/appendix_a`-`appendix_e`, `tables/T13_runs.tex` |
| Last (step 4) | `chapters/00_abstract.tex`, `01_introduction.tex`, `08_conclusion.tex` |
| Skeleton | `thesis.tex`, `references.bib`, `README.md`, `.gitignore`, `tables/draft_*.tex` |

The placeholder chapters carry, under each section heading, `% absorbs draft main.tex lines a-b` (the draft text
that moves there), `% needs: ...` (the new material, from plan section 2) and one `\TODO{...}` line.

## The numbers.tex convention

- One macro per headline number, with its source in a comment on the same line: `job:line` of the Slurm log, a
  handover section (`H 0b`, `H 3.1`, ...) or `file:line` of the repository. Example:
  `\newcommand{\FinalPVE}{73.7} % 3611336:191 (V1); H 0b`.
- Job ids never appear in the running text or captions. Appendix A (`tables/T13_runs.tex`) maps run names to job ids,
  logs and checkpoints, so any number can be traced in two steps.
- Macro names are letters only (`\FinalThreeDPW`, not `\Final3DPW`); values are plain numbers (`1927.4`), so
  `\num{\FinalGFLOPs}` can format them.
- Define each macro once, in the numbers file of the chapter that owns it. A second `\newcommand` of the same name
  in another numbers file stops the build: grep the other `numbers_*.tex` first.
- Run names in the text: Base, Control, Final, Final-masked, Final-masked-7, Student-S (history), Student,
  Multi-HMR-Anny (NAVER). Settings in every caption: V1 / V0 / 3DPW test (v23 labels).

## Labels defined by the skeleton chapters

Chapters `ch:intro`, `ch:results`, `ch:discussion`, `ch:conclusion`; appendices `app:reproduction`, `app:joints`,
`app:flags`, `app:figures`, `app:pilots`; results sections `sec:res-setup`, `sec:res-depth`, `sec:res-bone`,
`sec:res-fingers`, `sec:res-final`, `sec:res-3dpw`, `sec:res-rh`, `sec:res-mask`, `sec:res-baseline`,
`sec:res-student`, `sec:res-history`; tables `tab:requirements`, `tab:relhuman`, `tab:runs`, `tab:final`
(`tab:final-annyone`, `tab:final-3dpw`), `tab:baseline`, `tab:student`, `tab:mapping`; plots `fig:depth-scatter`,
`fig:rh-age`, `fig:cost-pve`, `fig:student-curves`, `fig:val-curves`, `fig:vits-curves`.

## Before submission

Remove `\nocite{*}` in `thesis.tex` (it lists every bib entry for checking); resolve every `% TODO verify` in
`references.bib` and `chapters/appendix_b_joints.tex`; search for `\TODO`.

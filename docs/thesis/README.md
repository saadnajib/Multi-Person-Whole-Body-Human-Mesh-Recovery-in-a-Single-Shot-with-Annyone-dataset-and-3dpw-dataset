# Master thesis: Multi-Person Human Mesh Recovery with the Anny Body Model

LaTeX source of the rewritten thesis (full draft since 18 Oct 2026; 132 pages with appendices on 19 Oct, after the partial-mask result). The plan behind it
(what each chapter, table and figure holds, the writing rules) is `REWRITE_PLAN.md`; the review pass that made the
chapters one document is logged in `REVIEW_LOG.md`; the old draft is kept read-only in `draft_2026-10_original/`.

`hardcover_titlepage.tex` is the separate one-page title for the bound copy (same fields as `thesis.tex`); build it alone with `latexmk -pdf hardcover_titlepage.tex`.

## Build

```
cd docs/thesis && latexmk -pdf -interaction=nonstopmode thesis.tex
latexmk -c          # remove the aux files, keep thesis.pdf
```

Needs pdflatex, biber, and the packages TikZ, pgfplots (`compat=1.18`), booktabs, siunitx, subcaption, longtable,
multirow, biblatex, listings. No `-shell-escape`. The build must end with zero errors, zero undefined references and
zero undefined citations (state on 18 Oct: it does); the only expected warnings are a few overfull boxes below 10 pt.
Figures that have not been delivered yet (F14 label check, F19 student curves) are wrapped in `\IfFileExists` or
drawn as "pending" boxes, so the build never breaks on a missing image. F11 (predicted fingers), F17 (rest-pose
line-up), F24 (ground-truth decode) and the V1 holdout montages of F12 were placed on 19 Oct (zip batch 2).

## Layout

```
thesis.tex               preamble, title page, Eigenstaendigkeitserklaerung + "Use of AI tools", abstract, TOC,
                         chapters 1-8, lists of figures/tables, appendices A-E, bibliography
numbers.tex              \input of the numbers_*.tex files (one macro per headline number, source in a comment)
numbers_<chapter>.tex    data, method, diagnostics, results, background, discussion
references.bib           every entry is cited; "% TODO verify" marks fields the student must check against the paper
chapters/                00_abstract, 01_introduction ... 08_conclusion, appendix_a_reproduction ... appendix_e_pilots
tables/                  T01-T21 (one file per table; T21 = the three shape-mask arms, added 19 Oct); draft_*.tex = the old draft's tables, reference only, not input
figures/img/             raster figures (docs/figures/*.jpg, demo montages, files sent from the cluster)
figures/plots/           pgfplots figures F15, F16, F18, F19 (placeholder), F20, F22
figures/tikz/            TikZ drawings F1-F5
figures/RPTU-*.pdf       title-page logos
data/                    CSVs behind the plots; plot_stats.md = the statistics computed from them; data/README.md
```

Pages per part (19 Oct, after the trimming pass): front matter 8, ch. 1: 4, ch. 2: 6, ch. 3: 8, ch. 4: 12, ch. 5: 8,
ch. 6: 16, ch. 7: 4, ch. 8: 2 (printed pages 1-60; chapters open on odd pages, so some spans include one blank page),
lists 4, App. A: 6, B: 2, C: 2, D: 4, E: 2, bibliography; 92 pages in all. The long 132-page version is commit
76aa1a4 and `thesis_long_2026-10-19.pdf`; scripts and commands cut from Appendix A are in `REPRODUCE.md`.

## The naming rule

The thesis uses only the model names of `docs/model_names.md`: Multi-HMR-Anny, L-stage 1 to 4, Base, Mixed-L,
Fresh-heads-L, Control, Final, Final-masked, Final-partial-mask, Student-S/1 to /6c, Student-S/mixed,
Student-S/final, Student. The short definition is Table 4.3 (`chapters/04_method.tex`, `tab:model-names`); the full
definition with old run names, starting weights, data, schedules, flags and checkpoint paths is Table A.1
(`tables/T19_model_names.tex`, `tab:names`). Old run names (`v5`, `v9`, `exp7`, `anny_s2_shape_v5`, `shpmask`, ...)
and job ids appear only in Appendix A (T13, T19), in the file paths of Appendix C and in LaTeX comments. Pilot runs
are named by what they are ("depth pilot", "neck pilot"), never by their job.

## The numbers rule

- Every headline number in the text and the tables is a macro of a `numbers_*.tex` file, with its source in a comment
  on the same line: `job:line` of the Slurm log, a handover section (`H 0b`, `H 3.1`, ...) or `file:line` of the
  repository. Example: `\newcommand{\FinalPVE}{73.7} % 3611336:191 (V1); H 0b`.
- Job ids never appear in the running text or captions. Appendix A (T13) maps run names to job ids, logs and
  checkpoints, so any number can be traced in two steps.
- Macro names are letters only (`\FinalThreeDPW`, not `\Final3DPW`); values are plain numbers so `\num{}` can format
  them. Define each macro once; grep the other `numbers_*.tex` first (a second `\newcommand` stops the build).
- Every caption names the set, the setting (V1 / V0 / 3DPW test, v23 labels), the models and the caveat (one run, no
  seed; synthetic; by eye; a good case).
- Nothing in `docs/handover.md` section 6 (corrections log) may reappear as a claim. Old-label 3DPW numbers (119.1,
  102.5, 111.5, 139.5, 123.6) never stand next to v23 numbers.
- The fixed NAVER sentence (plan section 5) accompanies every comparison of 60.3 with 73.7 mm.

## Open TODO list (grep -n TODO chapters tables figures thesis.tex references.bib)

Pending results (fill the macro or the cell, then delete the `\TODO`). Final-partial-mask (jobs 3614042, 3616821) is in
since 19 Oct (6.8 with T21, T12, 7.1, T01, 8.2, 4.3, T13, T19); no TODO left for it.
- Student (job 3614191): `numbers_results.tex` (`\StudentPending`, `\StudentCell`), `tables/T17_student.tex`,
  `tables/T16_baseline_cost.tex`, `figures/plots/F19_student.tex` (+ `data/F19_student.csv`), `figures/plots/F18_cost_pve.tex`
  (replace the dashed line by a point), chapter 6.10, chapter 7 (1b paragraph, T01 status word), chapter 8.
- Figures: F14 `figures/img/F14_label_check.jpg`.

Pending checks by the student:
- `thesis.tex`: the "Use of AI tools" wording against RPTU's rules; the title-page date (`\today`).
- `references.bib`: every `% TODO verify` field (pages, author lists, the Anny-fit venue, the MakeHuman title).
- `chapters/appendix_b_joints.tex`: the spine rows, the neck offset and the scale factor against a converter log
  (TODO in the file header), and the missing paragraph on how the correspondence was chosen.
- `tables/T11_student_recipe.tex`: Final's V1 MPJPE as re-measured by the student job (77.9 in T17 is from the CSV).
- Supervisor's answers (which model sets the 95 % bar; whether a smaller student counts for 1b; whether a refiner on
  frozen features counts as temporal; children path A or B): chapter 6.10, 7.1, 8.2.

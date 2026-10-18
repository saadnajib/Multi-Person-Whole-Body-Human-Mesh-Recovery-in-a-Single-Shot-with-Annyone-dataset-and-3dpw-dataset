# Review log: integration pass over the rewritten thesis (18 Oct 2026, late evening)

Scope: the eight chapters and five appendices written in parallel from `REWRITE_PLAN.md`, made one document and
checked against the project record (`docs/handover.md` 0a / 0b / 3.x / 6, `docs/model_names.md`,
`data/plot_stats.md`, the `numbers_*.tex` source comments). Build after the pass: 126 pages, zero errors, zero
undefined references, zero undefined citations. Nothing was committed.

## Summary by category (counts)

| category | changes |
|---|---|
| 1 Naming consistency | 61 replacements in 13 files + 1 table rewritten + 1 table relabelled |
| 2 Duplicates and dangling references | 1 duplicate figure removed, 1 undefined label resolved, Appendix D rewritten (5 figures), `\nocite{*}` removed, 1 placeholder renamed, 1 internal label renamed |
| 3 Numbers audit | 118 numbers checked; 1 text reading corrected (F22); 0 value changes needed; 3 numbers only partly traceable (listed below) |
| 4 Writing rules | 0 banned words, 0 confidence tags, 0 job ids outside Appendix A found; 1 NAVER fixed sentence added (Chapter 1); 1 results section without an opening table noted (6.8) |
| 5 Front matter and housekeeping | AI-use paragraph added; 18 short captions added; T01 and T20 re-laid out (caption was running into the footer, two rows wider than the text); README rewritten; handover 0a and requirements-plan entries added |
| 6 Build and render | 3 builds; 17 pages rendered and inspected; 2 layout faults found and fixed (T01, T20) |

## 1 Naming consistency

Rule: the names of `docs/model_names.md` / Table A.1 (`tables/T19_model_names.tex`) only; old run names and job ids
in Appendix A (T13, T19), Appendix C paths and LaTeX comments only.

- `Base+3DPW (old labels)` / `Base+3DPW` -> `Mixed-L`: chapters/04 (2), 05 (1), 06 (4), tables/T08 (2), T15 (3),
  T20 (3); comments in numbers_results (4), numbers_method (2).
- `Final-masked-7` -> `Final-partial-mask`: chapters/04 (2), 06 (2), 07 (2), 08 (1); numbers_method comment (1); README (1).
- `fresh-head run` -> `Fresh-heads-L`: chapters/05 (1, sentence rewritten), tables/T08 (2), T10 (2).
- `S1`..`S8`, `S6b`, `S6c` -> `Student-S/1`..`/6c`, `/mixed` (S7), `/final` (S8): chapters/04 (11), 06 (7),
  tables/T20 (10, row labels shortened to fit), figures/plots/F22 legend and caption (9).
- `the three stages before Base` -> `L-stage 2 to L-stage 4` (chapters/05, the eight flipped checkpoints).
- Chapter 4, `tab:model-names`: rewritten as the short definition (thesis name + one-line meaning, no old names, no
  epochs of old runs), covering every name of model_names.md (Multi-HMR-Anny, L-stage 1-4, Base, Mixed-L,
  Fresh-heads-L, Control, Final, Final-masked, Final-partial-mask, Student-S/1-/6c, /mixed, /final, Student), with a
  pointer to Table A.1 as the full definition. The "Base lineage" paragraph now names L-stage 1-4 and says that
  L-stage 1 is a side branch and that Base is the run after L-stage 4 (the chapter had called Base "the fourth
  stage", off by one against model_names.md).
- `tables/T20_history.tex` part (a): rows renamed Stage 1-4 -> L-stage 1 (side branch, "--"), L-stage 2, L-stage 3,
  L-stage 4, Base, so the history table and Table A.1 agree.
- Chapter 4 history paragraph: "(v1 to v8 in the code)" -> "(the number after the slash is the old version number,
  Table A.1)".
- `chapters/04`: L-stage 2 learning rate now "1e-4, then 3e-5 after a resume" (was only 3e-5), as in model_names.md.
- Internal label `fig:diag-neck-v5` -> `fig:diag-neck-base` (chapters/01, 05, 06, appendix D); not visible in the
  PDF, renamed so the grep for `v5` is clean.
- Converter versions (v1, v2, v2.1, v2.2, v2.3, "v23 labels") stay: they name the converter and the label set, not a
  model, and the plan uses them.
- Final grep (`masked-7|Base+3DPW|S1..S8|v5|v9|v10|exp7|exp8|anny_s2|shpmask`) over chapters, tables and figures,
  excluding comments, Appendix A, Appendix C paths and the draft tables: no hit.

## 2 Duplicates and dangling references

- F21 (Anny age sweep) stood in Chapter 2 (`fig:bg-age-sweep`) and Chapter 3 (`fig:data-age-sweep`). Kept Chapter 2's
  (the plan places F21 in 2.1); the Chapter 3 figure environment was removed and its reference now points to
  `fig:bg-age-sweep`.
- F24 (gt-decode) and F6 (neck before/after) appear only in Chapter 5; Appendix D had no copies, so nothing to drop.
  The F24 placeholder now waits for `figures/img/F24_gt_decode_1.jpg` (was `gt_decode_overlay.jpg`), inside
  `\IfFileExists`.
- `fig:exp6-compare` (undefined before the pass): now Figure D.2 in Appendix D, `exp6_compare_579864.jpg` and
  `exp6_compare_579866.jpg` side by side, caption with images, model, setting and caveat.
- Appendix D rewritten: intro paragraph, F8 (579860), F23 (exp6 pairs), F9 office pair, the four remaining 3DPW
  montages (walking, stairs, runforbus, fencing; new label `fig:3dpw-overlays-more`, referenced from the caption of
  Figure 6.7) and the four photo montages. The appendix-level `\TODO` was removed. Every caption names images,
  models, setting and caveat.
- `\nocite{*}` removed from thesis.tex. All 34 entries of `references.bib` are cited (checked against the `\cite`
  keys of every .tex file), so no uncited entry had to be removed; the 11 `% TODO verify` comments all sit on cited
  entries and stay for the student.
- Build: zero undefined references, zero undefined citations (thesis.log).

## 3 Numbers audit (chapters 1, 6, 7, 8, abstract, and the tables they use)

Checked against handover 0b / 0a / 3.1-3.6 / 5 item 7, `data/plot_stats.md`, the CSVs in `data/` and the
`numbers_*.tex` comments. All of the following agree with the record:

- AnnyOne V1: PVE 73.7 / 77.4 / 70.5; PA-PVE 50.3 / 56.4 / 53.3; recall 87.0 / 89.0 / 90.2; root error 728 / 1402 /
  1208 mm (plot_stats.md and the CSV means reproduce 728.4 / 1401.9 / 1208.4); size 0.78 / 0.59 / 0.63; slope
  0.42 / 0.28 / 0.33 (CSV fits 0.419 / 0.276 / 0.333); intercepts 1.75 / 2.66 / 2.36; neck02 4.8 / 175.8 / 175.6;
  girth 0.996 / 0.464 / 0.461 (text says 1.00 vs 0.46); fingers 8.3 / 17.4 / 17.3; head bone 8.9 / 177.7; PVE_neck
  11.0 / 27.7; head direction 13.1 / 13.0 / 12.0; neck region 5.5 / 62.6, head region 5.0 / 89.7 (H 3.2); matched
  people 214 / 219 / 222; too-far shares 94.4 / 99.1 %; identity crossing 3.01 / 3.67 m; Pearson r 0.56 / 0.46.
- 3DPW test v23: PVE 89.0 / 91.5 / 121.8 / 126.9 / 95.5 / 120.2; PA-PVE 57.6 / 59.7 / 72.0 / 68.8 / 62.9 / 68.7;
  MPJPE 88.3 / 90.9 / 126.9 / 129.4 / 94.8 / 124.8; precision 64 / 62 / 54 / 75 / 73 / 53; recall 99 / 99 / 100 /
  97 / 94 / 100; AnnyOne test-100 V0 77.5 / 81.2 / 74.3 / 78.5 / 72.7; gains 30.3 / 2.5 / 2.1 / 6.5 / 32.8 / 37.9.
- Multi-HMR-Anny: 60.3 / 44.6 / 89.0 on AnnyOne (V1 by design).
- Relative Human: recall 47.1 / 64.7 / 57.2; gender 46.5 / 64.2 / 49.3; AUC 0.606 / 0.719 / 0.708; ages adult / kid /
  baby 0.631 / 0.613 / 0.624, 0.536 / 0.500 / 0.438, 0.638 / 0.611 / 0.524; teen 0.597 / 0.524 / 0.616; heights
  (all three models, all groups) and the matched counts 948 / 1,305 / 1,157: recomputed from
  `data/rh_per_person_*.csv`, all reproduce to the digits shown; babies below 0.45: 2.6 / 49.3 / 16.9 %; kids 16.8 %.
- Cost: params 318.38 / 380.94 / 101.38, heads 14.01 / 76.57, backbone 304.37; GFLOPs 1927.4 / 1917.2 / 603.3; ms
  77.9 (A100), 110 / 51.4 (A40); ratios 3.14 / 3.19 / 2.14 / 3.76; 16 % fewer parameters.
- Bars: 77.6 / 93.7 against Final, 63.5 / 133.6 against NAVER; T17's derived bars 52.9, 82.0, 82.7, 81.6, 57.3,
  60.6, 92.9, 94.1 recomputed (teacher / 0.95 or x 0.95) and correct; Final V1 MPJPE 77.9 = CSV mean 77.95.
- Labels: 5,882 / 7,584; 2,622 / 3,311; 8,190 / 11,854; 16,694 / 22,749; 87 person-sequences; label check 14.1
  (was 18.7) mm median, 21.1 / 24.5 p95; 407 of 960 -> 0 of 1,080; men 19.0 -> 14.7 mm.
- Noise: 7.6 mm / 1.9 deg / 0.8 measured, "about 2" recall points resolvable; restart cost 2.4 mm; RH 635 images /
  2,130 people; 55 manifest images; Final 4,292 s, 10.9 images/s, 16 NaN steps.
- Pilot cells of T15 (depth pilot 74.8 / 54.0 / 87.8 / 544 / 0.38 / 1.59 / 0.84 / 0.457 / 27.7 / 12.7 / 17.4 /
  175.5 / 178.0; neck pilot 68.8 / 50.9 / 89.4 / 1429 / 0.34 / 2.56 / 0.58 / 0.997 / 10.7 / 14.4 / 8.3 / 4.7 /
  11.2): handover section 5, item 7 table and 0b.
- Arithmetic in chapter 7: neck-skin share 171 / 13,718 = 1.2 %; (27.7 - 11.0) x 171 / 13,718 = 0.21 mm; 1 -
  318.38 / 380.94 = 16.4 %; 73.7 - 60.3 = 13.4; 77.4 - 70.5 = 6.9; 53.3 - 50.3 = 3.0.

One reading corrected: Section 6.11 said the head-transfer students "overtake the random-heads student after about
20 epochs and keep their order to the end"; the F22 data (draft `tab:kd_epochs`) show Student-S/4-/6 crossing the
193 mm line between epochs 60 and 100, Student-S/3 never, and Student-S/4 and /5 swapping places. Rewritten to what
the curves show.

Numbers that could not be traced to a log line or a handover table (left as they are, flagged here):

1. `tables/T16_baseline_cost.tex`: DINOv2 ViT-S/14 backbone 22.06 M / 196.9 GFLOPs. Source is the docstring of
   `scripts/analysis/model_cost.py:28` and `REWRITE_PLAN.md` T16, not a job log (ViT-B 86.58 / 589.5 and ViT-L
   304.37 / 1917.2 are in handover 0a).
2. `numbers_method.tex` `\MethodNumVerts` 13,718 (Anny vertex count, used in Table 4.1 and the chapter 7 estimate):
   from the draft's I/O table, marked "NOT re-verified against anny 0.6.1" in the source comment.
3. Chapter 8, "about two GPU hours" for a repeat of Final: Final's training took 4,292 s on an A100 (3611336) plus
   the trace and the 3DPW test; `docs/thesis_requirements_plan.md` gives "~2 h" for an exp7 run. An estimate, not a
   measured total.

Numbers already flagged by their writers and kept as `\TODO`: every Student cell (job 3614191), the
Final-partial-mask result (job 3614042), Student-S/6c and /final on v23 (job 3615020), Final's V1 MPJPE as the student
job re-measures it (T11).

## 4 Writing rules (plan section 5)

- grep over chapters, tables and figures (comments excluded) for delve, leverage, robust, novel, seamless,
  cutting-edge, state-of-the-art, significantly, worth noting, crucially, importantly, notably, comprehensive,
  holistic, paving the way, unlock, showcase, plethora, solved, fixed completely, perfect, sentence-initial
  Furthermore/Moreover, `[Certain]`/`[Likely]`/`[Guessing]`: the only hits are "significantly smaller" inside the
  supervisor's quoted wording (chapters 6, 7, T01) and "not solved" (abstract, 6.7, 7.3, 8.1). No em-dash asides, no
  rhetorical questions.
- Job ids (35xxxxx / 36xxxxx) outside Appendix A: only in LaTeX comments (appendix E source lines, table headers);
  none in the text.
- Chapter 1, contribution 7: the comparison 60.3 vs 73.7 now carries the plan's fixed NAVER sentence (it had a
  shorter paraphrase). The sentence was already present in 4.7.2, 6.9, 6.10, 7.1 (1b), T15 and T16 footnotes, T01.
- Evidence first: Sections 6.2-6.7 and 6.9-6.11 open with their figure or table. Section 6.8 (shape-mask trade-off)
  opens with text and reads the third column of Table 6.2 and the numbers of 6.5-6.6; it has no table of its own.
  Left as is; a one-row table would repeat T12. Noted for the student.
- Captions: every figure caption of chapters 5, 6 and Appendix D names the images, the models, the setting and the
  caveat; checked one by one while adding the short forms.

## 5 Front matter and housekeeping

- `thesis.tex`: the Eigenstaendigkeitserklaerung stays the official German text; after it, on the same page, an
  English "Use of AI tools" paragraph (assistant used for code, analysis scripts and drafting under the author's
  direction; every number checked by the author against the logs; author takes responsibility), with a one-line
  `\TODO` for the student to confirm the wording against RPTU's rules. Title-page fields unchanged (title, author,
  `\today`, examiner and supervisor); the date is the student's to fix.
- Short captions (`\caption[short]{long}`) added where the long caption exceeded two lines and no short form existed:
  chapters/03 (5), 04 (5), appendix B (1), T02, T03, T04, T05, T06, T07, T11 (7), F19 (1; its `\TODO` no longer
  appears in the list of figures). Total 18.
- Page headers: scrlayer-scrpage with chapter / section marks; checked on the rendered pages (correct on every page
  rendered, including the appendices).
- `README.md` rewritten: final layout, build, naming rule, numbers rule, page budget and the open TODO list (file
  ownership is over).
- `docs/handover.md` 0a: a "docs/thesis, 18 Oct (late evening)" entry (folder, build, page count, pending items).
- `docs/thesis_requirements_plan.md` section 2, row 3-report: "partly (full draft exists)", 126 pages / 82 in the
  chapters, pending items.

## 6 Build and render

- Three full builds (latexmk, pdflatex + biber). Final state: 126 pages, exit 0, no `!` errors, no undefined
  references or citations, no multiply-defined labels. `latexmk -c` run at the end; `thesis.pdf` kept.
- Overfull boxes > 20 pt: none. Remaining (all below 10 pt): thesis.toc (five entries, 1.2 pt each, the long
  appendix titles in the table of contents) and `chapters/02_background.tex` lines 234-249 (8.5 pt, a paragraph with
  a long citation key). Fixed during the pass: T01 (the table ran into the footer; arraystretch 1.0, columns
  re-widened, now fits with the caption above the footer) and T20 (two rows 17 and 55 pt too wide after the longer
  Student-S names; rows shortened, column gap 2.5 pt).
- Rendered with `pdftoppm -jpeg -r 50` into the scratchpad folder `thesis_render/` and inspected (pdf page / content /
  verdict):
  1 title page: OK (logo, title, author, date, examiner). 2 Eigenstaendigkeitserklaerung + Use of AI tools: OK, both
  on one page, TODO visible in red. 3 abstract: OK, one page. 11 chapter 1 with Table 1.1: OK. 17 chapter 2 with
  Figure 2.1 (age sweep): OK. 33 chapter 3 with Table 3.7 and the F14 placeholder: OK (placeholder box as designed).
  43 chapter 4 with Table 4.3 (model names): OK, float page. 56 chapter 5 with Table 5.3 (depth runs): OK. 68
  chapter 6 with Table 6.1: OK. 69 Figure 6.6 (six holdout montages): OK. 71 Figure 6.8 (RH box plot): OK. 75 Table
  6.3 + Figure 6.11: OK (table resized to the text width, small but legible; Student cells TODO). 79 Table 6.5 +
  Figure 6.13: OK after the fix. 82 Table 7.1 (requirements): FIXED (caption had overlapped the footer). 89 chapter 8:
  OK. 95 Appendix A start with the T19 longtable head: OK (the first longtable begins at the foot of the page with
  its header and group line only; cosmetic). 116-118 Appendix D: OK (new figures D.1-D.5 inside the margins; D.4's
  four landscape montages are small, the same size as Figure 6.7).

## 7 Length

Pages per part (final build): front matter 8 (title, declaration, abstract, contents), chapter 1: 6, chapter 2: 10,
chapter 3: 12, chapter 4: 16, chapter 5: 10, chapter 6: 18, chapter 7: 8, chapter 8: 2, lists of figures and tables
4, Appendix A: 14, B: 2, C: 4, D: 4, E: 4, bibliography 4. Chapters 1-8: 82 pages (plan: 60-80 with figures);
whole document 126.

Ten biggest trimming candidates (page cost each), for the student to decide; nothing was cut:

1. Chapter 6.11 + Table 6.5 + Figure 6.13 (ViT-S history, V0, old labels): 2 pages. Could shrink to one paragraph
   and the table, with Figure 6.13 dropped or moved to Appendix D.
2. Chapter 4.7.1 (history of the ViT-S students): 1 page. Repeats 6.11 in prose; keep the three lessons only.
3. Chapter 3.4 (converter development, three tables 3.3-3.5): 2 pages. One table (the fencing-scene versions) and
   half a page of text would carry the argument.
4. Appendix A, Section A.4 scripts table and A.5 commands: 5 pages together. The scripts table could list the
   experiment and evaluation scripts only (1.5 pages saved), the commands could move to the repository README
   (1.5 pages saved).
5. Chapter 2.6 (temporal models for video): 1.5 pages. Until 2b exists, half a page naming the model family and the
   planned refiner suffices.
6. Chapter 5.8 + Table 5.8 (what did not work): 1 page. Appendix E holds the same pilots in full; the chapter needs
   one paragraph and a pointer.
7. Chapter 7.1, the seven requirement paragraphs: 1 page. Table 7.1 carries the status and the evidence; the
   paragraphs could keep only "what is missing" and the plan pointer.
8. Figure 6.6 (six holdout montages) and Figure 6.5 (in-training curves): 1.5 pages. Four montages and the curves
   in Appendix D would do; the text reads both in two sentences each.
9. Appendix E, the six pilot paragraphs after Table E.1: 1.5 pages. The table and the noise paragraph suffice; the
   paragraphs repeat chapter 5.
10. Chapter 4.1, the I/O table (Table 4.1) and the HPH dimension paragraph: 0.5 page. The architecture figure and
    two sentences carry it.

Cutting 1-10 in full would save about 17 pages of the main text and 8 of the appendices.

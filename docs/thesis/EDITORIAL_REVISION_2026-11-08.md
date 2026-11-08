# Thesis merge and editorial revision — 8 November 2026

The plain-language branch was drafted from an older local checkout. Before merging,
fetching main revealed a newer 6–7 November thesis rewrite, verified 60-epoch
results, phenotype and gender analyses, recorded video timing and motion summaries,
new figures, and a temperature-sweep progress record.

## Merge resolution

The newer main chapters and their plain-language style were retained. The merge
applies the compatible editorial corrections from `codex/thesis-plain-language`:

- Explain the projector as learned weighted combinations, not added information.
- Clarify what the real-image hand mask removes, including remaining mesh and
  teacher-output supervision.
- Distinguish PVE's sensitivity to body size from PA-PVE's fitted scale.
- Describe how aligned-depth filtering preserves the pelvis image position and
  why its final projection error must be recomputed.
- Qualify baseline holdout exposure and remove unsupported general noise thresholds.
- Retain the updated pilot table and MediaPipe citation.
- Widen contents/list-of-figures number columns and fit the three conversion-check
  images within their page, resolving layout warnings from the expanded draft.
- Keep the AI-use disclosure; move its editorial reminder into these notes.

No incoming main code, measured values, figures or experimental result tables were
discarded. All `numbers*.tex` files match pre-merge main (`5d61a7b`). The completed
60-epoch result, verified five-panel video run, and temperature-sweep progress in
`data/CLUSTER_ANSWERS.md` remain available. The complete sweep table is still pending;
missing values have not been invented. The reported default remains the 30-epoch
student, with the 60-epoch run as a training-length comparison.

## PDF and length

`thesis_revised_2026-11-08.pdf` is rebuilt from the merged source. Its page count
and checks are in `validation_2026-11-08.txt`. The previously delivered 100-page PDF
on the editing branch did not contain the newer main experiments. The preceding
main PDF had 120 pages. The merge preserves those experiments instead of deleting
them to reproduce the shorter draft. Font size and page geometry are unchanged.

## Author review before submission

Read the merged text and check the title-page details, bibliography fields marked
`% TODO verify`, production body-model metadata and AI-use disclosure. The existing
temperature-sweep TODOs remain explicit. A successful build verifies references
and layout, not the factual completeness of pending experiments or institutional
approval of the declaration. No new training or cluster inference ran for this merge.

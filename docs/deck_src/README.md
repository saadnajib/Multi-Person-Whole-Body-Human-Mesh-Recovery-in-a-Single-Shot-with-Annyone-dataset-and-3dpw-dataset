# Deck source: supervisor update, 19 October 2026

`build.js` builds the 30-slide deck `docs/supervisor_update_2026-10-19.pptx` and its speaker script
`docs/supervisor_talk_notes_2026-10-19.md` for the supervisor meeting on Monday 19 October 2026, 16:00. The 19 Oct
version (built on the evening of 18 Oct) tells the whole story in order, in plain words: overview, data, training,
findings and fixes, the final model, requirements and plan, appendix. It replaces the 21-slide deck of 16-18 Oct
(`supervisor_update_2026-10-16.pptx`, removed); the slides kept from it are reused unchanged apart from renumbered
cross-references and a few titles. Every number on a slide comes from `docs/handover.md` (0b, 0a, 3.x), `docs/thesis_requirements_plan.md`,
`docs/experiments_plan.md`, `loss.py`, `slurm/experiments/exp7_final_model.sh`, `slurm/experiments/exp8_distill_vitb.sh`
or `scripts/analysis/model_cost.py`; slide 30 lists the jobs and logs. The script runs outside the machine it was first
built on (line 2 loads `./apply_theme.js`; the figures are found through `__dirname`).

Slides (section, number, title):
- Overview: 1 Multi-HMR with Anny: what we changed and what we measured (title, DARK); 2 The goal and your requirements;
  3 The model in one picture.
- Data: 4 The datasets; 5 3DPW to Anny: how the labels are made; 6 3DPW labels: three defects found and fixed.
- Training: 7 What the model learns: the losses in plain words; 8 What we changed in training: v5 -> final model;
  9 How we measure.
- Findings and fixes: 10 Where we started: three visible failures; 11 Checked first: what is NOT the cause;
  12 The hidden depth bug: anchor vs target; 13 Depth fix: before and after; 14 Depth fix: the numbers;
  15 Neck: two half-turned bones, and a pilot fix; 16 Neck: before and after the fix; 17 Fingers: about half the
  error was two flipped bones; 18 Children and gender on real photos; 19 Masked shape labels: a trade-off.
- Final model: 20 Final model: all fixes together, with a control; 21 Final model on real photos: v5 vs final (by eye).
- Requirements and plan: 22 Pilots: four results and the noise floor; 23 Thesis requirements: where we stand;
  24 Plan for 1b: a smaller model by distillation; 25 Plan: the next four weeks; 26 What I claim, what I do not,
  and my questions (DARK_CLOSE).
- Appendix: 27 Still open, and what I test next; 28 Corrections to the draft and the repo; 29 Backup: before and
  after, holdout image 579860; 30 Jobs and checkpoints.

New on 19 Oct: slides 1-4, 7-9, 11 (merges the old slides 3 and 4), 18, 19, 24, 25 and 30. Old -> new numbering of
the reused slides: 2->10, 5->5, 6->6, 7->15, 8->16, 9->17, 11->12, 12->13, 13->14, 14->22, 15->28, 16->20, 17->21,
18->27, 19->26, 20->23, 21->29.

Files here:
- `build.js`: the whole deck (layouts, slides, speaker notes) written with pptxgenjs, plus two post-processing steps.
- `apply_theme.js`: writes the deck's theme colours into the pptx (step 2 below).
- `package.json`, `package-lock.json`: pptxgenjs, locked at 4.0.1. Built with node v22.22.0; any node >= 18 should do.
- `.gitignore`: keeps `node_modules/` and the build outputs out of git.

## Build

Run on a machine with node >= 18 and internet access (`npm ci` downloads pptxgenjs); probably a laptop, since the
cluster may not have node. LibreOffice (`soffice`) and `pdftoppm` are needed only for the optional render check.

```bash
cd docs/deck_src
npm ci          # installs exactly the locked pptxgenjs 4.0.1
node build.js   # writes supervisor_update.pptx and supervisor_talk_notes.md here
```

Run `node build.js` from this folder: it writes its outputs into the current folder.

Checked on 15 Oct 2026 (node 22, `npm ci`, a fresh copy of this folder next to `docs/figures/`): `node build.js` gives
the same slides, theme, media and layouts as the then-published `docs/supervisor_update_2026-10-16.pptx` (only the timestamps in
`docProps/core.xml` and the random section IDs in `ppt/presentation.xml` differ), and a speaker-notes `.md`
byte-identical to the then-published `docs/supervisor_talk_notes_2026-10-16.md`. Checked again on the afternoon of 18 Oct for the
21-slide version (this folder's `build.js` and `apply_theme.js` copied next to `docs/figures/`, pptxgenjs 4.0.1 from
an existing `node_modules` instead of a fresh `npm ci`): the same result. The 30-slide version was built the same way
on 18 Oct (evening) and rendered with LibreOffice for the layout check.

## What `build.js` does when it runs

1. Writes `supervisor_update.pptx` into the current folder.
2. Calls `applyTheme("supervisor_update.pptx", THEME)` (`build.js:670`, from `apply_theme.js`): writes the deck's 12
   theme colours (theme "MeshDiag": dark `1B2233`, accent `0E7C86`, orange `E8792B`, green `3BA55D`, red `C0392B`,
   grey `7B8794`, yellow `F4C95D`; fonts Calibri; the object is `THEME` at `build.js:4-5`) into
   `ppt/theme/theme1.xml`, replacing the Office palette that pptxgenjs always writes there. Many shapes use scheme
   colours (`pres.SchemeColor`), so **without this step the colours fall back to Office blue/orange**.
3. `keepFirstParaProps(...)` (inside `build.js`): removes duplicate paragraph properties and splits each slide's notes
   into one paragraph per line.
4. Writes `supervisor_talk_notes.md` from the `N(slide, title, text)` calls. **Edit the notes in `build.js`, never in
   the `.md`**; the `.md` is regenerated on every build.

## Render check (optional)

```bash
soffice --headless --convert-to pdf supervisor_update.pptx
pdftoppm -jpeg -r 80 supervisor_update.pdf slide
```

LibreOffice wraps text slightly differently from PowerPoint. The table slides 7, 8, 9, 23, 24, 25 and 30, slide 13's
footnote and slide 26 (the claims slide) are the tightest; check them in PowerPoint before the meeting.

## Publish

```bash
cp supervisor_update.pptx ../supervisor_update_2026-10-19.pptx
cp supervisor_talk_notes.md ../supervisor_talk_notes_2026-10-19.md
```

Small edits (a number, a word) are faster directly in PowerPoint, e.g. a result that arrives just before the meeting
(`docs/handover.md` section 9 lists which slides change for which result). Then the `.md` notes and `build.js` drift
from the pptx: port the same edits into `build.js` afterwards, rebuild, and say so in the commit message.

## History: how it was built the first time

On the build machine the deck was built in a scratch folder with a longer version of the theme helper (86 lines; same
job as `apply_theme.js` plus input checks) and checked with two helper scripts from the same folder: an OOXML schema
check (`validate.py`) and a LibreOffice wrapper (`soffice.py`). None of these exist outside that machine and none is
needed. The original line 2 loaded that helper from
`/root/.claude/skills/synced/9d62f05e-4200-4cf4-8106-de589de8d052_d2acd300-8cd3-4e6a-8972-a7838572b8ef/pptx/scripts/apply_theme.js`,
and line 70 set `FIG_DIR` to the absolute path
`/home/user/Multi-Person-Whole-Body-Human-Mesh-Recovery-in-a-Single-Shot-with-Annyone-dataset-and-3dpw-dataset/docs/figures/`.

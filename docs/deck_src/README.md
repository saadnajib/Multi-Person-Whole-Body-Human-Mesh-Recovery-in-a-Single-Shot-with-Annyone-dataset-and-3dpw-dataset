# Deck source: supervisor update, 16 October 2026

`build.js` builds the 18-slide deck `docs/supervisor_update_2026-10-16.pptx` and its speaker notes
`docs/supervisor_talk_notes_2026-10-16.md`. It is the script that made the published files (on 15 Oct 2026, and the
18-slide update on 16 Oct), with two lines changed so that it runs outside the machine it was first built on (line 2
loads `./apply_theme.js`; line 70 finds the figures through `__dirname`).

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
the same slides, theme, media and layouts as `docs/supervisor_update_2026-10-16.pptx` (only the timestamps in
`docProps/core.xml` and the random section IDs in `ppt/presentation.xml` differ), and a speaker-notes `.md`
byte-identical to `docs/supervisor_talk_notes_2026-10-16.md`.

## What `build.js` does when it runs

1. Writes `supervisor_update.pptx` into the current folder.
2. Calls `applyTheme("supervisor_update.pptx", THEME)` (`build.js:522`, from `apply_theme.js`): writes the deck's 12
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

LibreOffice wraps text slightly differently from PowerPoint. Slides 5, 6 and 17 are the tightest; check them in
PowerPoint (`docs/handover.md` section 9 lists every slide to check).

## Publish

```bash
cp supervisor_update.pptx ../supervisor_update_2026-10-16.pptx
cp supervisor_talk_notes.md ../supervisor_talk_notes_2026-10-16.md
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

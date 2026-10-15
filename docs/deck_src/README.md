# Deck source: supervisor update, 16 October 2026

`build.js` builds the 17-slide deck `docs/supervisor_update_2026-10-16.pptx` and its speaker notes
`docs/supervisor_talk_notes_2026-10-16.md`. It is a copy of the script that made the published files on 15 Oct 2026
(the copied outputs were byte-identical to the files in `docs/`). It is unchanged; the two absolute paths below are
still the ones from the machine it was built on.

Files here:
- `build.js`: the whole deck (layouts, slides, speaker notes) written with pptxgenjs, plus two post-processing steps.
- `package.json`, `package-lock.json`: pptxgenjs `^4.0.1` (4.0.1 was installed). Built with node v22.22.0.
- `.gitignore`: keeps `node_modules/` and the build outputs out of git.

## What `build.js` does when it runs

1. Writes `supervisor_update.pptx` into the current folder.
2. Calls `applyTheme(...)`: writes the deck's own 12 theme colours (theme "MeshDiag": dark `1B2233`, accent `0E7C86`,
   orange `E8792B`, green `3BA55D`, red `C0392B`, grey `7B8794`, yellow `F4C95D`; fonts Calibri) into
   `ppt/theme/theme1.xml`. Many shapes use scheme colours (`pres.SchemeColor`), so **without this step the colours
   fall back to Office blue/orange**.
3. `keepFirstParaProps(...)` (inside `build.js`): removes duplicate paragraph properties and splits each slide's notes
   into one paragraph per line.
4. Writes `supervisor_talk_notes.md` from the `N(slide, title, text)` calls. **Edit the notes in `build.js`, never in
   the `.md`**; the `.md` is regenerated on every build.

## Two paths that are NOT in this repo

| line | what | what to do |
|---|---|---|
| `build.js:2` | `require("/root/.claude/skills/synced/9d62f05e-4200-4cf4-8106-de589de8d052_d2acd300-8cd3-4e6a-8972-a7838572b8ef/pptx/scripts/apply_theme.js")`: an 86-line helper from the build machine. It opens the pptx with the jszip that ships with pptxgenjs, replaces the `<a:clrScheme>` in `ppt/theme/theme1.xml` with the 12 colours, and renames the theme. | Not copied. Either write a small `apply_theme.js` that does exactly that (about 20 lines with jszip) and change the path, or delete line 2 and the `applyTheme` call (then fix the colours in PowerPoint: Design > Variants > Colors). |
| `build.js:70` | `FIG_DIR = "/home/user/Multi-Person-Whole-Body-Human-Mesh-Recovery-in-a-Single-Shot-with-Annyone-dataset-and-3dpw-dataset/docs/figures/"`: the two before/after figures `depth_before_after_579860.jpg` and `depth_before_after_579862.jpg`. | Point it at your own checkout's `docs/figures/`. The files are in the repo. |

The build machine also used two checking scripts from the same helper folder, `.../pptx/scripts/office/validate.py`
(OOXML schema check) and `.../pptx/scripts/office/soffice.py` (a LibreOffice wrapper). They are not needed: use plain
`soffice` and `pdftoppm` (below), and open the file in PowerPoint once.

## Build

```bash
cd docs/deck_src
npm install pptxgenjs@4.0.1
# after fixing the two paths above:
NODE_PATH=$PWD/node_modules node build.js
```

`NODE_PATH` is needed when the theme helper lives outside this folder, because it loads jszip from beside pptxgenjs.

The exact command used on the build machine (build, validate, render to JPG; the paths only exist there):

```bash
cd /tmp/claude-0/-home-user-Multi-Person-Whole-Body-Human-Mesh-Recovery-in-a-Single-Shot-with-Annyone-dataset-and-3dpw-dataset/ebfb21e0-1d18-5a3b-93e8-48ceb0b0c30c/scratchpad/deck && NODE_PATH=$PWD/node_modules node build.js && python3 /root/.claude/skills/synced/9d62f05e-4200-4cf4-8106-de589de8d052_d2acd300-8cd3-4e6a-8972-a7838572b8ef/pptx/scripts/office/validate.py supervisor_update.pptx | tail -4 && python3 /root/.claude/skills/synced/9d62f05e-4200-4cf4-8106-de589de8d052_d2acd300-8cd3-4e6a-8972-a7838572b8ef/pptx/scripts/office/soffice.py --headless --convert-to pdf supervisor_update.pptx >/dev/null 2>&1; rm -f slide-*.jpg; pdftoppm -jpeg -r 80 supervisor_update.pdf slide && ls "$PWD"/slide-*.jpg
```

Render check anywhere else:

```bash
soffice --headless --convert-to pdf supervisor_update.pptx
pdftoppm -jpeg -r 80 supervisor_update.pdf slide
```

LibreOffice wraps text slightly differently from PowerPoint. Slides 5, 6 and 16 are the tightest; check them in
PowerPoint.

## Publish

```bash
cp supervisor_update.pptx ../supervisor_update_2026-10-16.pptx
cp supervisor_talk_notes.md ../supervisor_talk_notes_2026-10-16.md
```

Small edits (a number, a word) are faster directly in PowerPoint. Then the `.md` notes and `build.js` drift from the
pptx; say so in the commit message.

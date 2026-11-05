# Supervisor presentation — 22 October 2026

21 main slides and 3 technical backup slides, based on the supplied seven-topic outline and thesis revision `3e2f21e`.

Order, revised after the supervisor's feedback (introduce the body models and both datasets before the method):

| Slides | Block |
|---|---|
| 1–3 | Title, problem statement, Multi-HMR baseline |
| 4 | Two body models: SMPL (3DPW labels) and Anny (our output) |
| 5–8 | Datasets: AnnyOne and 3DPW, how the 3DPW–Anny labels were created, virtual joints, label checks |
| 9–16 | Method, losses, distillation, training recipe, metrics, reconstruction and efficiency results |
| 17–21 | Pictures, video placeholder, open points, conclusion |
| 22–24 | Technical backup |

- [Editable PowerPoint](Najib_supervisor_presentation.pptx): diagrams, tables, pictures and detailed speaker notes.
- [Slide PDF](Najib_supervisor_presentation.pdf): preview of all 24 slides.
- Speaking script: [PDF](Speaker_script.pdf), [Word](Speaker_script.docx), or [Markdown](Speaker_script.md).
- [Sources and presenting notes](Sources_and_presenting_notes.md): slide provenance and interpretation limits.
- [Validation](validation.txt): export and layout checks.

**Slide 19 contains the video placeholder.** Insert the walking comparison video yourself using PowerPoint's Insert → Video → This Device. The video is not included in this folder.

## Rebuilding

The PowerPoint can be edited directly. To regenerate it and the scripts from the Python source, install the document-generation dependencies in a separate Python environment and run from the repository root:

```bash
python -m pip install -r docs/presentations/2026-10-22_supervisor/requirements.txt
python docs/presentations/2026-10-22_supervisor/build_presentation.py
```

The generator reads existing images from `docs/thesis/figures/img/` and writes outputs beside itself. It recreates the deck, speaker scripts, source notes and slide-content JSON; direct edits to those generated files will be replaced. Equation images and the slide 4 skeleton drawings are generated in ignored `assets/`.

Slide 4 also reads the committed `figures/` folder:

- `smpl_mesh_crop.jpg`, `anny_mesh_crop.jpg`: crops of thesis Figure F14 (3DPW `courtyard_basketball_01`, image 570), rewritten by the generator.
- `anny_rest_skeleton.json`: the 163 Anny bones (names, parents, rest-pose bone heads, body region, the SMPL→Anny name mapping), written by `python scripts/analysis/render_smpl_vs_anny.py --mode anny_skeleton` (needs only `anny` and `torch`).
- `smpl_vs_anny_render.png` (optional): the SMPL and Anny rest meshes with their joints, rendered on the cluster with `scripts/analysis/render_smpl_vs_anny.py` (the command is in its docstring). When this file exists, it replaces the two F14 crops.

To refresh the slide PDF, export the PowerPoint as PDF, or use LibreOffice:

```bash
soffice --headless --convert-to pdf --outdir docs/presentations/2026-10-22_supervisor docs/presentations/2026-10-22_supervisor/Najib_supervisor_presentation.pptx
```

Review layout after rebuilding, since fonts and application versions can affect rendering. Refresh validation notes after reviewing any changes.

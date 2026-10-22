# Supervisor presentation — 22 October 2026

20 main slides and 3 technical backup slides, based on the supplied seven-topic outline and thesis revision `3e2f21e`.

- [Editable PowerPoint](Najib_supervisor_presentation.pptx): diagrams, tables, pictures and detailed speaker notes.
- [Slide PDF](Najib_supervisor_presentation.pdf): preview of all 23 slides.
- Speaking script: [PDF](Speaker_script.pdf), [Word](Speaker_script.docx), or [Markdown](Speaker_script.md).
- [Sources and presenting notes](Sources_and_presenting_notes.md): slide provenance and interpretation limits.
- [Validation](validation.txt): export and layout checks.

**Slide 18 contains the video placeholder.** Insert the walking comparison video yourself using PowerPoint's Insert → Video → This Device. The video is not included in this folder.

## Rebuilding

The PowerPoint can be edited directly. To regenerate it and the scripts from the Python source, install the document-generation dependencies in a separate Python environment and run from the repository root:

```bash
python -m pip install -r docs/presentations/2026-10-22_supervisor/requirements.txt
python docs/presentations/2026-10-22_supervisor/build_presentation.py
```

The generator reads existing images from `docs/thesis/figures/img/` and writes outputs beside itself. It recreates the deck, speaker scripts, source notes and slide-content JSON; direct edits to those generated files will be replaced. Equation images are generated in ignored `assets/`.

To refresh the slide PDF, export the PowerPoint as PDF, or use LibreOffice:

```bash
soffice --headless --convert-to pdf --outdir docs/presentations/2026-10-22_supervisor docs/presentations/2026-10-22_supervisor/Najib_supervisor_presentation.pptx
```

Review layout after rebuilding, since fonts and application versions can affect rendering. Refresh validation notes after reviewing any changes.

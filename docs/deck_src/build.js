const pptxgen = require("pptxgenjs");
const { applyTheme } = require("./apply_theme.js");

const THEME = { name: "MeshDiag", headFontFace: "Calibri", bodyFontFace: "Calibri",
  colors: { dk1: "1B2233", lt1: "FFFFFF", dk2: "4A5568", lt2: "EEF2F7", accent1: "0E7C86", accent2: "E8792B", accent3: "3BA55D", accent4: "C0392B", accent5: "7B8794", accent6: "F4C95D", hlink: "0E7C86", folHlink: "7B8794" } };

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";               // 10 x 5.625 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.author = "Saad Najib"; pres.title = "Multi-HMR with Anny: what we changed and what we measured";
const C = pres.SchemeColor;

pres.defineSlideMaster({ title: "DARK", background: { color: THEME.colors.dk1 },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.6, w: 8.8, h: 1.4, fontSize: 40, bold: true, color: C.background1, align: "left", margin: 0 } } },
            { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 3.1, w: 8.8, h: 1.8, fontSize: 18, color: C.background2, align: "left", margin: 0 } } }] });
// closing slide: same look, title higher; the body holds the claims, the question gets its own box below
pres.defineSlideMaster({ title: "DARK_CLOSE", background: { color: THEME.colors.dk1 },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 8.8, h: 0.8, fontSize: 36, bold: true, color: C.background1, align: "left", margin: 0 } } },
            { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 1.25, w: 8.8, h: 2.35, fontSize: 15, color: C.background2, align: "left", valign: "top", margin: 0 } } }],
  slideNumber: { x: 9.2, y: 5.2, w: 0.5, h: 0.3, fontSize: 10, color: THEME.colors.accent5, margin: [0, 0, 0, 7.2], valign: "middle" } });
pres.defineSlideMaster({ title: "LIGHT", background: { color: THEME.colors.lt1 },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.35, w: 9.0, h: 0.8, fontSize: 27, bold: true, color: C.text1, align: "left", margin: 0, valign: "middle" } } },
            { text: { text: "Supervisor update, 19 Oct 2026", options: { x: 0.5, y: 5.2, w: 5, h: 0.3, fontSize: 10, color: C.accent5, margin: 0 } } }],
  slideNumber: { x: 9.2, y: 5.2, w: 0.5, h: 0.3, fontSize: 10, color: THEME.colors.accent5, margin: [0, 0, 0, 7.2], valign: "middle" } });   // insets / anchor as the footer: one baseline

const notes = [];
function N(slide, title, text) { slide.addNotes(text); notes.push({ title, text }); }

// helpers
function card(slide, x, y, w, h, head, body, color, opts = {}) {
  slide.addShape(pres.ShapeType.roundRect, { x, y, w, h, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.08, objectName: head });
  slide.addShape(pres.ShapeType.ellipse, { x: x + 0.2, y: y + 0.2, w: 0.42, h: 0.42, fill: { color }, line: { color }, objectName: head + " dot" });
  slide.addText(head, { x: x + 0.75, y: y + 0.15, w: w - 0.9, h: 0.5, fontSize: opts.hs || 16, bold: true, color: C.text1, margin: 0, isTextBox: true, valign: "middle", objectName: head + " head" });
  slide.addText(body, { x: x + 0.2, y: y + 0.75, w: w - 0.4, h: opts.bh || h - 0.9, fontSize: opts.bs || 13, color: C.text2, margin: 0, isTextBox: true, valign: "top", objectName: head + " body" });
}
// big number + label; the label is top-aligned so one-line and two-line labels start on the same line
function stat(slide, x, y, w, big, label, color, o = {}) {
  const bh = o.bh || 0.9;
  slide.addText(big, { x, y, w, h: bh, fontSize: o.bfs || 40, bold: true, color, margin: 0, isTextBox: true, align: "left", objectName: "stat " + label });
  slide.addText(label, { x, y: y + bh, w, h: o.lh || 0.7, fontSize: o.lfs || 13, color: C.text2, margin: 0, isTextBox: true, align: "left", valign: "top", objectName: "label " + label });
}
function table(slide, rows, x, y, w, colW, opts = {}) {
  const data = rows.map((r, i) => r.map((c, j) => ({ text: String(c), options: { bold: i === 0 || j === 0, color: i === 0 ? THEME.colors.lt1 : THEME.colors.dk1, fill: { color: i === 0 ? THEME.colors.dk1 : (i % 2 ? THEME.colors.lt1 : THEME.colors.lt2) }, fontSize: opts.fs || 13, align: j === 0 ? "left" : (opts.alignBody || "center"), valign: "middle" } })));
  slide.addTable(data, { x, y, w, colW, border: { type: "solid", color: "FFFFFF", pt: 1 }, rowH: opts.rowH || 0.42, margin: opts.margin === undefined ? 0.06 : opts.margin, objectName: opts.name || "table" });
}
// bulleted runs: items are strings or [boldLead, rest]
function bulletRuns(items, opts = {}) {
  const runs = [];
  items.forEach((it, i) => {
    const last = i === items.length - 1;
    const para = { bullet: { indent: 14 }, paraSpaceAfter: opts.space === undefined ? 4 : opts.space };
    if (Array.isArray(it)) {
      runs.push({ text: it[0], options: { ...para, bold: true } });
      runs.push({ text: it[1], options: { breakLine: !last } });
    } else {
      runs.push({ text: it, options: { ...para, breakLine: !last } });
    }
  });
  return runs;
}
// numbered item (circle + bold head + body), same motif as the "what I did" slide
function numberedItem(slide, x, y, w, num, head, body, color, bodyH) {
  slide.addShape(pres.ShapeType.ellipse, { x, y: y + 0.05, w: 0.5, h: 0.5, fill: { color }, line: { color }, objectName: "open " + num });
  slide.addText(num, { x, y: y + 0.05, w: 0.5, h: 0.5, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "open num " + num });
  slide.addText(head, { x: x + 0.7, y, w: w - 0.7, h: 0.33, fontSize: 15, bold: true, color: C.text1, margin: 0, isTextBox: true, objectName: "open head " + num });
  slide.addText(body, { x: x + 0.7, y: y + 0.36, w: w - 0.7, h: bodyH, fontSize: 12, color: C.text2, margin: 0, isTextBox: true, valign: "top", objectName: "open body " + num });
}

// before/after figures (pixel size measured from the files; aspect ratio preserved)
const FIG_DIR = require("path").join(__dirname, "../figures/");
const FIG_PX = { w: 2394, h: 872 };            // 3 panels of 790 px + 2 gaps of 12 px
function figureSlide(slide, file, x, y, w, captions, capH = 0.4) {
  const h = w * FIG_PX.h / FIG_PX.w;
  slide.addImage({ path: FIG_DIR + file, x, y, w, h, altText: "Ground truth, before (v5) and after (fix) meshes on a synthetic holdout image", objectName: "figure " + file });
  const s = w / FIG_PX.w;
  captions.forEach((cap, i) => {
    slide.addText(cap, { x: x + i * 802 * s, y: y + h + 0.06, w: 790 * s, h: capH, fontSize: 11, color: C.text1, margin: 0, isTextBox: true, valign: "top", objectName: "caption " + i });
  });
  return y + h;
}
// same left edge and width as the figure (x 1.0, w 8.0)
function footnote(slide, lead, text, y, h) {
  slide.addText([{ text: lead, options: { bold: true, color: THEME.colors.dk1 } }, { text }],
    { x: 1.0, y, w: 8.0, h, fontSize: 11, color: C.text2, margin: 0, isTextBox: true, valign: "top", objectName: "figure footnote" });
}

// status colours used on slides 2 and 23: green = met, orange = partly, red = not met, grey = not started
const STATUS = { met: THEME.colors.accent3, partly: THEME.colors.accent2, "not met": THEME.colors.accent4, "not started": THEME.colors.accent5 };
// one-line caption under a table: "How to read: ..."
function howToRead(slide, text, y, h = 0.3, w = 9.0, x = 0.5) {
  slide.addText([{ text: "How to read: ", options: { bold: true, color: THEME.colors.dk1 } }, { text }],
    { x, y, w, h, fontSize: 11, color: C.text2, margin: 0, isTextBox: true, valign: "top", objectName: "how to read " + y });
}

pres.addSection({ title: "Overview" });
pres.addSection({ title: "Data" });
pres.addSection({ title: "Training" });
pres.addSection({ title: "Findings and fixes" });
pres.addSection({ title: "Final model" });
pres.addSection({ title: "Requirements and plan" });
pres.addSection({ title: "Appendix" });

// 1 title (19 Oct version: the whole story, not one week's findings)
let s = pres.addSlide({ masterName: "DARK", sectionTitle: "Overview" });
s.addText([{ text: "Multi-HMR with Anny:", options: { fontSize: 34, breakLine: true } }, { text: "what we changed and what we measured", options: { fontSize: 34 } }], { placeholder: "title" });
// five one-line results (16 pt) and the byline (14 pt, muted grey 9AA5B4: 6.4:1 on the dark background)
// numbers: handover 0b (size 0.63 -> 0.78, neck02 175.6 -> 4.8, fingers 17.3 -> 8.3, 3DPW test 121.8 -> 89.0; Relative Human gender 46.5 -> 64.2%)
s.addText(["A hidden depth bug: bodies were drawn at 0.63 of their true size; now 0.78.",
           "Neck and finger bones were half a turn off; a 0-parameter fix turns them back.",
           "3DPW converted to Anny, three label defects fixed, and 30% of every training batch.",
           "Final model: best on 3DPW test, 89.0 mm (v5 121.8); real photos: gender 46 → 64%.",
           "Not yet: a model that is not Multi-HMR (ViT-B student running) and temporal modelling."].map(t => ({ text: t, options: { fontSize: 16, breakLine: true } }))
  .concat([{ text: "Saad Najib, supervisor update, 19 October 2026", options: { fontSize: 14, paraSpaceBefore: 6, color: "9AA5B4" } }]), { placeholder: "body" });
N(s, "Multi-HMR with Anny: what we changed and what we measured", "Good afternoon. This talk tells the whole story in order: the goal and your requirements, the model, the data, how it is trained and measured, what went wrong and how I fixed it, the final model, and the plan for the next four weeks. The five lines on this slide are the results. A hidden bug in the training targets made every body too small in the image; it is fixed for size, not yet for near-versus-far depth. Two neck bones and two hand bones were turned half a turn; a fixed correction at the output turns them back with no new parameters. 3DPW is converted to Anny, three label defects are fixed, and it is in training. The final model is our best on the 3DPW test, and gender on real photos goes from chance to 64 percent. What is not done: a model that is not Multi-HMR, and temporal modelling. I will say for every number whether it is measured or only seen by eye.");

// 2 the goal and the requirements (status words = slide 23; dot colours = STATUS)
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Overview" });
s.addText("The goal and your requirements", { placeholder: "title" });
{
  s.addText([{ text: "The task. ", options: { bold: true, color: THEME.colors.dk1 } },
             { text: "From one photo, find every person and predict their Anny body: the pose (how each bone is turned), the shape (11 values such as gender, age, height and weight, each between 0 and 1) and the distance from the camera." }],
    { x: 0.5, y: 1.25, w: 4.4, h: 0.95, fontSize: 12, color: C.text1, margin: 0, isTextBox: true, valign: "top", objectName: "task sentence" });
  const steps = [["Photo", "one image, 672 px"], ["Find people", "a score per cell of a 48×48 grid"], ["Predict", "pose, shape, depth per person"], ["Anny body", "a mesh you can draw and measure"]];
  const bw = 0.98, gap = 0.16, by = 2.35, bh = 0.95;
  steps.forEach((st, i) => {
    const x = 0.5 + i * (bw + gap);
    s.addShape(pres.ShapeType.roundRect, { x, y: by, w: bw, h: bh, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.08, objectName: "pipe box " + i });
    s.addText([{ text: st[0], options: { bold: true, fontSize: 11.5, breakLine: true } }, { text: st[1], options: { fontSize: 9.5, color: THEME.colors.dk2 } }],
      { x: x + 0.05, y: by + 0.05, w: bw - 0.1, h: bh - 0.1, color: C.text1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "pipe text " + i });
    if (i < steps.length - 1) s.addShape(pres.ShapeType.line, { x: x + bw + 0.02, y: by + bh / 2, w: gap - 0.04, h: 0, line: { color: C.text2, width: 1.5, endArrowType: "triangle" }, objectName: "pipe arrow " + i });
  });
  s.addText([{ text: "Where we test it. ", options: { bold: true, color: THEME.colors.dk1 } },
             { text: "AnnyOne holdout (synthetic, exact labels), 3DPW test (real video, labels converted to Anny), Relative Human (real photos with age and gender tags) and demo photos (judged by eye). Slide 4 lists them." }],
    { x: 0.5, y: 3.5, w: 4.4, h: 0.95, fontSize: 12, color: C.text1, margin: 0, isTextBox: true, valign: "top", objectName: "where we test" });
  s.addText([{ text: "Status dots: ", options: { bold: true, color: THEME.colors.dk1 } }, { text: "green met · orange partly · red not met · grey not started (details: slide 23)." }],
    { x: 0.5, y: 4.6, w: 4.4, h: 0.45, fontSize: 11, color: C.text2, margin: 0, isTextBox: true, valign: "top", objectName: "status legend" });
  // the six requirements of the e-mail of 18 Oct, in plain words (docs/thesis_requirements_plan.md section 1 and 2)
  const reqs = [["1a", "Correct pose, visibly varying body shape", "partly", "pose: neck and fingers fixed, best on 3DPW test; shape: gender yes, muscle and weight not yet"],
                ["1b", "A model that is not exactly Multi-HMR", "not met", "new architecture that beats it, or a much smaller one within 95%; a ViT-B student is running (slide 24)"],
                ["1c", "3DPW evaluated in Anny space", "met", "all 8,190 test images have Anny labels; limits stated on slide 23"],
                ["2a", "Train on 3DPW converted to Anny", "met", "30% of every batch of the final model"],
                ["2b", "Temporal modelling (video, not frame by frame)", "not started", "designed, planned for weeks 5–6"],
                ["3", "Well-written report; clean, verifiable code", "partly", "draft exists; code clean-up and model release to do"]];
  s.addText("Your requirements (e-mail of 18 Oct)", { x: 5.2, y: 1.2, w: 4.3, h: 0.3, fontSize: 13, bold: true, color: C.text1, margin: 0, isTextBox: true, valign: "middle", objectName: "req head" });
  reqs.forEach((r, i) => {
    const y = 1.55 + i * 0.59, col = STATUS[r[2]];
    s.addShape(pres.ShapeType.ellipse, { x: 5.2, y: y + 0.07, w: 0.17, h: 0.17, fill: { color: col }, line: { color: col }, objectName: "req dot " + r[0] });
    s.addText([{ text: r[0] + "  " + r[1], options: { bold: true, color: THEME.colors.dk1 } }, { text: "  " + r[2], options: { bold: true, color: col } }],
      { x: 5.47, y, w: 4.05, h: 0.3, fontSize: 11, margin: 0, isTextBox: true, valign: "top", objectName: "req head " + r[0] });
    s.addText(r[3], { x: 5.47, y: y + 0.27, w: 4.05, h: 0.3, fontSize: 9.5, color: C.text2, margin: 0, isTextBox: true, valign: "top", objectName: "req body " + r[0] });
  });
}
N(s, "The goal and your requirements", "First the goal in one sentence. From one photo the model finds every person and predicts their Anny body: the pose, which is how each bone is turned; the shape, eleven values between 0 and 1 such as gender, age, height and weight; and the distance from the camera. The strip shows the four steps: photo, find people on a grid, predict pose, shape and depth per person, and the Anny body model turns that into a mesh. On the right are your six requirements from the e-mail of 18 October, each with a status dot: green met, orange partly, red not met, grey not started. Two are met: 3DPW is evaluated in Anny space and used in training. Pose and shape are partly there. The model is still Multi-HMR, so 1b is not met; the smaller model is my route. Temporal modelling has not started. The report and the code are partly done. Slide 23 has the evidence for each dot.");

// 3 the model in one picture (facts: model.py: img_size 672, patch 14 -> 48x48; HPH head; 163 bones x 6D; 11 phenotypes;
// 88 helper bones masked -> 75 active; handover 1; bone fix 4 bones at the output; backbone weights from NAVER's Anny checkpoint)
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Overview" });
s.addText("The model in one picture", { placeholder: "title" });
{
  const boxes = [["Photo", "resized to 672 × 672 px"],
                 ["Backbone", "DINOv2 ViT-L: 14-px patches, a 48 × 48 grid, one feature per cell (304 M parameters)"],
                 ["Detection", "per cell: is a person's head here? plus a small 2D offset"],
                 ["Per-person head", "cross-attention (HPH): pose = 163 bone rotations (75 trained, 88 helper bones fixed), shape = 11 Anny values 0–1, distance"],
                 ["Anny body", "the body model turns pose + shape into a mesh; placed at the predicted distance"]];
  const bw = 1.62, gap = 0.225, by = 1.25, bh = 1.6;
  boxes.forEach((b, i) => {
    const x = 0.5 + i * (bw + gap);
    s.addShape(pres.ShapeType.roundRect, { x, y: by, w: bw, h: bh, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.08, objectName: "arch box " + i });
    s.addText(b[0], { x: x + 0.08, y: by + 0.08, w: bw - 0.16, h: 0.32, fontSize: 13, bold: true, color: C.text1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "arch head " + i });
    s.addText(b[1], { x: x + 0.08, y: by + 0.42, w: bw - 0.16, h: bh - 0.5, fontSize: 10, color: C.text2, align: "center", valign: "top", margin: 0, isTextBox: true, objectName: "arch body " + i });
    if (i < boxes.length - 1) s.addShape(pres.ShapeType.line, { x: x + bw + 0.03, y: by + bh / 2, w: gap - 0.06, h: 0, line: { color: C.text2, width: 1.5, endArrowType: "triangle" }, objectName: "arch arrow " + i });
  });
  // teal markers under the boxes that carry our changes
  [[2, "ours: head anchor fix"], [3, "ours: Anny output, bone fix"], [4, "ours: 3DPW → Anny labels"]].forEach(([i, t], k) => {
    const x = 0.5 + i * (bw + gap);
    s.addText(t, { x, y: by + bh + 0.05, w: bw, h: 0.24, fontSize: 9.5, bold: true, color: THEME.colors.lt1, fill: { color: THEME.colors.accent1 }, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "ours tag " + k });
  });
  s.addShape(pres.ShapeType.roundRect, { x: 0.5, y: 3.3, w: 9.0, h: 1.25, fill: { color: "D3ECEE" }, line: { color: "D3ECEE" }, rectRadius: 0.08, objectName: "ours box" });
  s.addText(bulletRuns([
      ["Output layer sized for Anny: ", "163 bones × 6 rotation numbers and 11 shape values (upstream Multi-HMR predicts SMPL-X)."],
      ["Head-anchor fix (training code): ", "the targets for position, distance and detection are built at the head, the joint the model anchors (slide 12)."],
      ["Bone fix (0 parameters): ", "one constant rotation each on neck02, head and two hand bones, applied at the output (slide 15)."],
      ["3DPW → Anny conversion and training recipe: ", "our fitting script, three label defects fixed (slides 5–6), 30% real frames per batch (slide 8)."]], { space: 3 }),
    { x: 0.7, y: 3.35, w: 8.6, h: 1.15, fontSize: 11.5, color: C.text1, margin: 0, isTextBox: true, valign: "middle", objectName: "ours text" });
  s.addText([{ text: "Honest for 1b: ", options: { bold: true, color: THEME.colors.accent4 } },
             { text: "this is still Multi-HMR's architecture, and the backbone weights start from NAVER's Anny checkpoint. Everything marked “ours” is an output layer, a training-side fix or data. The route to 1b is a much smaller model (slide 24)." }],
    { x: 0.5, y: 4.62, w: 9.0, h: 0.5, fontSize: 11, color: C.text1, margin: 0, isTextBox: true, valign: "top", objectName: "honest 1b" });
}
N(s, "The model in one picture", "This is the model, left to right. The photo is resized to 672 pixels. A DINOv2 ViT-L backbone cuts it into 14-pixel patches, a 48 by 48 grid, and gives one feature per cell. A detection layer says, per cell, whether a person's head is there, with a small offset to place it exactly. For each detected person a cross-attention head predicts the pose, 163 bone rotations of which 75 are trained and 88 helper bones are held fixed, the shape, 11 Anny values between 0 and 1, and the distance from the camera. The Anny body model turns pose and shape into a mesh. What is ours is marked in teal: the output layer sized for Anny, the head-anchor fix in the training code, the bone fix at the output, and the 3DPW conversion and recipe. I want to be honest about requirement 1b: this is still Multi-HMR's architecture, and the backbone weights come from NAVER's Anny checkpoint. The smaller model on slide 24 is my route.");

// 4 the datasets (handover 1 and 0b: AnnyOne 579,960 renders, last 100 = holdout; v23 labels train 5,882 / 7,584, val 2,622 / 3,311,
// test 8,190 / 11,854; Relative Human val 635 images / 2,130 people; demo 4 photos + 8 3DPW-test frames)
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Data" });
s.addText("The datasets", { placeholder: "title" });
{
  const rows = [["dataset", "real or synthetic", "what it has", "used for", "size"],
                ["AnnyOne", "synthetic renders, 1280 × 1280 px", "exact Anny pose, all 11 shape values, camera; babies to old people", "training (70% of each batch); the last 100 images = holdout, never trained on", "579,960 images"],
                ["3DPW → Anny (v23 labels)", "real video frames, 60 sequences", "SMPL joints fitted to Anny: pose, gender, height, proportions; 8 shape values constant; hands and head at rest", "training (30% of each batch, train split); test split = our real-video test", "train 5,882 images / 7,584 people; val 2,622 / 3,311; test 8,190 / 11,854"],
                ["Relative Human (val)", "real photos, crowded scenes", "per person: age group (adult, teen, kid, baby), gender, a box; no 3D", "evaluation only: children, gender, recall", "635 images, 2,130 labelled people"],
                ["Demo photos", "real photos + 3DPW test frames", "no labels", "judged by eye only", "4 photos + 8 frames"]];
  table(s, rows, 0.5, 1.2, 9.0, [1.55, 1.45, 2.65, 2.0, 1.35], { fs: 10, rowH: [0.3, 0.6, 0.72, 0.56, 0.42], alignBody: "left", name: "datasets table" });
  howToRead(s, "every number in this talk names its dataset. AnnyOne numbers are exact but synthetic; 3DPW numbers are against our converted labels (not comparable with published 3DPW numbers); Relative Human counts detected people only.", 3.88, 0.42);
  stat(s, 0.5, 4.35, 2.9, "100 images", "AnnyOne holdout: synthetic, exact labels, helper bones masked (indices 579,860–959)", C.accent1, { bfs: 22, bh: 0.38, lh: 0.4, lfs: 10 });
  stat(s, 3.55, 4.35, 2.9, "8,190 images", "3DPW test, 11,854 people, Anny labels (v23); one sequence yields no labels", C.accent1, { bfs: 22, bh: 0.38, lh: 0.4, lfs: 10 });
  stat(s, 6.6, 4.35, 2.9, "2,130 people", "Relative Human validation, 635 real photos with age-group and gender tags", C.accent1, { bfs: 22, bh: 0.38, lh: 0.4, lfs: 10 });
}
N(s, "The datasets", "Four datasets, and every number later names one of them. AnnyOne is synthetic: 579,960 renders with exact Anny pose and all eleven shape values, from babies to old people. It is 70 percent of every training batch, and its last 100 images are the holdout that no model trained on. 3DPW is real video, 60 sequences, with SMPL ground truth that I fitted to Anny. Its labels carry pose, gender, height and proportions, but eight shape values are constants and the hands and head stay at rest, so it tests pose more than shape. The train split is 30 percent of each batch; the test split, 8,190 images with 11,854 people, is our real-video test. Relative Human is real photos with an age group and a gender per person and no 3D, so we use it only to measure children, gender and recall: 635 images, 2,130 people. The demo photos have no labels and are judged by eye.");

// 5-6: 3DPW labels (reused)
// 5 3DPW conversion: pipeline, native shapes only (boxes in code order: the camera step runs after the fit)
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Data" });
s.addText("3DPW to Anny: how the labels are made", { placeholder: "title" });
{
  const flow = [["1", "3DPW ground truth", "24 SMPL joints, camera, gender"],
                ["2", "Pick frames", "valid frames, then every 3rd per person"],
                ["3", "Fit Anny per frame", "match the 24 joints in 3D (9 torso joints via fixed offsets)"],
                ["4", "One body per person", "median shape, then fit the pose again"],
                ["5", "World → camera", "only the root bone changes"],
                ["6", "Label + loader", "drop fits > 80 mm; one file per image; 30% of each batch"]];
  const bw = 1.25, gap = 0.3, by = 1.3, bh = 1.1;
  flow.forEach((st, i) => {
    const x = 0.5 + i * (bw + gap);
    s.addShape(pres.ShapeType.roundRect, { x, y: by, w: bw, h: bh, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.08, objectName: "flow box " + st[0] });
    s.addShape(pres.ShapeType.ellipse, { x: x + bw / 2 - 0.2, y: by + 0.1, w: 0.4, h: 0.4, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: "flow dot " + st[0] });
    s.addText(st[0], { x: x + bw / 2 - 0.2, y: by + 0.1, w: 0.4, h: 0.4, fontSize: 14, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "flow num " + st[0] });
    s.addText(st[1], { x: x + 0.06, y: by + 0.55, w: bw - 0.12, h: 0.5, fontSize: 13, bold: true, color: C.text1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "flow head " + st[0] });
    s.addText(st[2], { x: x - 0.04, y: by + bh + 0.08, w: bw + 0.08, h: 0.58, fontSize: 11, color: C.text2, align: "center", valign: "top", margin: 0, isTextBox: true, objectName: "flow line " + st[0] });
    if (i < flow.length - 1) s.addShape(pres.ShapeType.line, { x: x + bw + 0.05, y: by + bh / 2, w: gap - 0.1, h: 0, line: { color: C.text2, width: 1.5, endArrowType: "triangle" }, objectName: "flow arrow " + st[0] });
  });
  // what the fit matches, what it may change, what it never changes
  const what = [["Matched", "24 joint positions per frame, from 3DPW's own male or female SMPL. SMPL rotations, body-shape values and the mesh surface are not used.", C.accent1],
                ["Free", "Root position and rotation; 18 of the 23 mapped bones (e.g. SMPL neck → neck01). The 5 end bones get no signal: head, finger2-1, toe1-1 (L/R). Height and proportions, one per person.", C.accent3],
                ["Fixed", "Gender from 3DPW (reversed: next slide); age 0.55; the other 7 shape values 0.5. By the code, 144 of 163 bones stay at rest: neck02/03, head, jaw, fingers, twist bones.", C.accent2]];
  what.forEach((w, i) => {
    const x = 0.5 + i * 3.05, y = 3.12, cw = 2.9, ch = 1.66;
    s.addShape(pres.ShapeType.roundRect, { x, y, w: cw, h: ch, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.08, objectName: "fit " + w[0] });
    s.addShape(pres.ShapeType.ellipse, { x: x + 0.17, y: y + 0.15, w: 0.3, h: 0.3, fill: { color: w[2] }, line: { color: w[2] }, objectName: "fit dot " + w[0] });
    s.addText(w[0], { x: x + 0.6, y: y + 0.1, w: cw - 0.75, h: 0.4, fontSize: 15, bold: true, color: C.text1, margin: 0, isTextBox: true, valign: "middle", objectName: "fit head " + w[0] });
    s.addText(w[1], { x: x + 0.17, y: y + 0.56, w: cw - 0.34, h: ch - 0.66, fontSize: 11, color: C.text2, margin: 0, isTextBox: true, valign: "top", objectName: "fit body " + w[0] });
  });
}
s.addText([{ text: "Code: ", options: { bold: true, color: THEME.colors.dk1 } }, { text: "scripts/data/smpl_to_anny.py v2.2 (steps 1–4), build_threedpw_anny_labels.py (5–6), datasets/threedpw_anny.py + train.py (6)." }],
  { x: 0.5, y: 4.86, w: 9.0, h: 0.26, fontSize: 11, color: C.text2, margin: 0, isTextBox: true, valign: "top", objectName: "flow code" });
N(s, "3DPW to Anny: how the labels are made", "One more ground-truth check, now for 3DPW. 3DPW gives, per frame, 24 SMPL joint positions, the camera and the gender. I keep the valid frames and take every third one per person. Then Anny is fitted frame by frame with gradient descent, so that its 24 joints land on the SMPL joints. The two skeletons are different, so first each SMPL joint is assigned one Anny bone; the hip and spine choice is checked on the rest pose, and for nine torso joints, where the rigs differ, the fit aims at a fixed offset from the bone, not at the bone itself. Only positions are matched: SMPL rotations, shape values and the surface are not used. Each person then gets one body, the median over the frames, and the pose is fitted again. Moving into the camera changes only the root bone. When the label files are written, fits with more than 80 millimetres joint error, people less than 30 percent visible and boxes under 40 pixels are dropped. Only the root and 18 bones move. The head, finger and toe end bones are mapped, but nothing below them is matched, so they get no signal. Age is fixed at 0.55. By the code, 144 of the 163 bones stay at rest, including neck02, neck03, the head and the fingers. The check on the next slide confirms this for every person it sampled. So in these labels the head turns only through neck01.");

// 6 3DPW conversion: measured result of slurm/diagnostics/check_3dpw_conversion.sh, cluster job 3575369
// (log: check_3dpw_conversion_3575369.out). Every number below is from that log: V1 body joints, V2 surface, V3 2D,
// V4a/b, V5 loader, I4 all fits; I1 gender refit, I3 two-person coverage, I4 expected vs indexed label images; I2 head.
// 18 Oct: the cards show the same check on the re-fit (v2.3) labels, job 3611284 (check_3dpw_conversion_3611284.out,
// results/diagnostics/check_3dpw_conversion_v23); the 3575369 numbers are given as "was" and in the notes.
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Data" });
s.addText("3DPW labels: three defects found and fixed", { placeholder: "title" });
{
  const FIXED = "E6F4EA";                      // light green tint: the defects card, fixed by the re-fit (18 Oct)
  const cols = [["Measured: pass", C.accent3, C.background2, [
                   ["Body joints: ", "median 14.1\u00A0mm (p95 21.1; was 18.7), below the ~26 mm cited for 3DPW's own joints."],
                   ["Surface vs SMPL: ", "median 18.9 / 16.1\u00A0mm (was 18.9 / 18.3), nearest point, so a lower bound."],
                   ["2D vs OpenPose: ", "11.5 px median, only +0.4 px over raw 3DPW (11.1 px); was +1.1."],
                   ["Left/right, mirror: ", "100% pass. The loader passes labels unchanged."],
                   ["All 87 fits: ", "22,912 frames; mean joint error per frame: median 14.0 mm, max 79.8 mm."]]],
                ["Defects: fixed", C.accent3, FIXED, [
                   ["Gender: ", "was reversed (men 1, women 0); now men 0, women 1, Anny's convention. Flipping it now makes the fit worse: men 13.9\u00A0\u2192\u00A020.0 mm (was better for 72 of 73)."],
                   ["Two-person images: ", "one actor unlabelled in 407 of 960 \u2192 0 of 1,080: both people now share one frame grid."],
                   ["Labels missing: ", "image files missing; re-extracted 16 Oct. New labels: 16,694 images, 22,749 people (fewer images, same people)."]]],
                ["Limits and grades", C.accent5, C.background2, [
                   ["Head at rest: ", "neck02/03, head, jaw at rest in every label; only neck01 turns."],
                   ["Head turn vs chest: ", "19.3\u00B0 in SMPL, 19.6\u00B0 in the labels (median); was 9.5\u00B0."],
                   ["Overall WARN (was FAIL): ", "limb-length ratios, forearm 0.948, thigh 1.064 Anny/SMPL; FAIL in validation only (forearm 0.877)."],
                   ["WARN: ", "3.2% of people have a joint >\u00A0100\u00A0mm (worst: left hip, climbing_02). Train alone: 5.5%, FAIL."]]]];
  cols.forEach((c, i) => {
    const x = 0.5 + i * 3.05, y = 1.25, cw = 2.9, ch = 3.5;   // same card grid as slide 5
    s.addShape(pres.ShapeType.roundRect, { x, y, w: cw, h: ch, fill: { color: c[2] }, line: { color: c[2] }, rectRadius: 0.08, objectName: "ev " + c[0] });
    s.addShape(pres.ShapeType.ellipse, { x: x + 0.17, y: y + 0.15, w: 0.3, h: 0.3, fill: { color: c[1] }, line: { color: c[1] }, objectName: "ev dot " + c[0] });
    s.addText(c[0], { x: x + 0.6, y: y + 0.1, w: cw - 0.75, h: 0.4, fontSize: 15, bold: true, color: C.text1, margin: 0, isTextBox: true, valign: "middle", objectName: "ev head " + c[0] });
    s.addText(bulletRuns(c[3], { space: 4 }), { x: x + 0.15, y: y + 0.55, w: cw - 0.3, h: ch - 0.62, fontSize: 11, color: C.text1, margin: 0, isTextBox: true, valign: "top", objectName: "ev body " + c[0] });
  });
}
s.addText([{ text: "Source: ", options: { bold: true, color: THEME.colors.dk1 } }, { text: "check_3dpw_conversion.sh, job 3611284 on the re-fit labels (old labels: 3575369); 3,314 people, 2,360 images, 59 sequences" }],
  { x: 0.5, y: 4.86, w: 9.0, h: 0.26, fontSize: 11, color: C.text2, margin: 0, isTextBox: true, valign: "top", objectName: "check source" });
N(s, "3DPW labels: three defects found and fixed", "The finished labels had never been checked end to end against 3DPW. A first cluster check, on 15 October, found that joints, surface and 2D pass, but also three defects: gender was reversed, in 407 of 960 two-person images one person had no label, and eight sequences had no labels or too few. The missing labels came from missing image files; I re-extracted them on 16 October, and a second rebuild brought them back. Gender and the missing person are decided inside the fit, so I re-fitted all 87 people with two fixes: gender by Anny's convention, men 0 and women 1, and one shared frame grid for both people. That finished on 18 October, and the fit's own check passed on all 60 sequences: the men's mean joint error went from 19.0 to 14.7 millimetres, the women stayed at about 15, and the frames where one of two people was missing went from 54 percent to zero. The new labels have 16,694 images with 22,749 people: fewer images than the 18,936 before, with the same people, because both people now share each frame. Then I ran the same conversion check on the new labels, and this slide shows it. Body joints are 14.1 millimetres off at the median, was 18.7, below the about 26 millimetres cited for 3DPW's own joints. The surface is 18.9 and 16.1 millimetres off, was 18.9 and 18.3; that is a nearest-point distance, so a lower bound. In 2D the labels are only 0.4 pixels worse than raw 3DPW, was 1.1. Gender is now right: flipping it makes the fit worse, for the men 13.9 against 20.0 millimetres, where on 15 October the flip was better for 72 of 73. No two-person image misses a person now, 0 of 1,080. The overall grade went from FAIL to WARN. What is left are known differences between the Anny and SMPL bodies, the limb proportions and the hip joint, not label bugs. So the final model can train on these labels.\n\nIf asked:\n- Right column: by design, the head does not turn on its own: neck02, neck03, head and jaw stay at rest in every label, and only neck01 turns. The head turn against the chest is now 19.6 degrees in the labels against 19.3 in SMPL, at the median. In the old labels it was 9.5 against 20, lost mainly in the test split. I have not checked why it came back; my guess is the gender fix, because every test person is a man and was fitted with the female body before.\n- Grades: the WARN comes from two checks. Limb-length ratios: forearm 0.948 and thigh 1.064, Anny over SMPL; in the validation split alone the forearm is 0.877, so FAIL there. And 3.2 percent of people have a joint more than 100 millimetres off, mostly the left hip in outdoors_climbing_02, plus a few capoeira, freestyle and dancing frames; that is 5.5 percent in the train split, so FAIL there. Before the re-fit the limb ratios failed in test only, forearm 0.879 and thigh 1.119; the gender fix most likely removed that, but I have not shown it.\n- Gender: Anny's male end is 0: at gender 0 the rest body is 1.753 metres tall, at 1 it is 1.632 metres. In the new check every man is labelled 0.000 and every woman 1.000, 2,386 and 928 people in the checked images; these count people in images, not different persons. The refit test used 118 people, 2 per sequence: men 13.9 millimetres with their label and 20.0 with the gender flipped, the flip better for 1 percent of them; women 14.8 against 15.6, the flip better for 38 percent on the joints but only 9 percent on the surface. On 15 October the flip lowered the men's median joint error from 19.2 to 15.3 millimetres, better for 72 of 73. The surface test alone cannot see a wrong gender; that is why there is a separate refit test.\n- Missing labels: train had 4,451 of 6,437 expected label images, validation 2,458 of 2,777. The first rebuild, job 3579266, gave the same counts. The label builder skips every frame whose image file is missing, without a warning, so the likely cause was an incomplete unzip of the 3DPW images on the cluster. On 16 October I re-extracted the image folders of the eight sequences; captureSelfies_00 had 480 of 750 files. The second rebuild, job 3581320, gave 18,936 label images, exactly what the old fits imply. My earlier guess, that the builder ran before the fits finished, was wrong.\n- Re-fit: all 87 fits ran on free GPU nodes through a small environment with a newer PyTorch, so they did not wait in the queue; 36 tasks first failed on two V100 nodes and were rerun elsewhere. New label counts: train 5,882 images, validation 2,622, test 8,190.\n- outdoors_crosscountry_00 has no labels, now or before: with 3DPW's own camera its people sit behind the camera, so nothing projects into the image. The fit itself is fine, 17.8 millimetres. So 3DPW validation has 11 usable sequences.\n- Joints: the 14.1 millimetres are measured at the points the fit aims at. The bare bone heads that train.py uses are 30.0 millimetres off, was 32.5, mostly a fixed torso rig gap, not a fit error. The script marks the 26 millimetres for 3DPW as likely, from von Marcard et al., 2018; I have not checked the paper.\n- Still fixed by design: age 0.55 for everyone and seven more shape values at 0.5; only height and proportions are fitted, and gender comes from 3DPW.\n- 2D and left/right: left and right were correct in every checked person, and the mirror test passes.");

// 7 the losses in plain words. Terms and defaults: loss.py forward() (total = weighted sum, :409-420) and
// add_specific_args (:447-463: bce 10, offset 1, rotmat 0.1, shape 1, dist 1, transl 1, j3d 100, v3d 100, j2d 1, v2d 1).
// Final model overrides (slurm/experiments/exp7_final_model.sh:762-766, fix arm :211-213): alpha_shape 20, j2d/v2d 0.1,
// rotmat l1 0.1 with neck and hand bones boosted x4, alpha_dist 5, alpha_transl 5, start_2d_epoch 0, mask_3dpw_hands 1
// (38 hand bones, handover 0a); the control arm keeps dist/transl at 1 (:217-219).
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Training" });
s.addText("What the model learns: the losses in plain words", { placeholder: "title" });
{
  const rows = [["loss", "what it compares", "plain meaning", "weight (final model)"],
                ["detection (BCE)", "per cell: predicted vs true “a person's head is here”", "find every person", "10"],
                ["2D offset", "predicted vs true shift of the head inside its cell", "place the anchor exactly", "1"],
                ["bone rotation (L1)", "163 bone rotations vs the labels (relative to parent)", "get the pose right", "0.1 (neck, hands ×4)"],
                ["shape (L1)", "11 Anny values vs the labels", "get the body shape right", "20 (default 1)"],
                ["distance", "predicted vs true distance from the camera", "how far away the body is", "5 (default 1)"],
                ["position", "predicted vs true 3D position of the anchor joint", "where the body sits in 3D", "5 (default 1)"],
                ["3D joints", "joint positions vs truth, both centred on the pelvis", "the skeleton in 3D", "100"],
                ["3D mesh points", "mesh vertices vs truth, both centred on the pelvis", "the body surface in 3D", "100"],
                ["2D joints", "joints projected into the image vs the 2D labels", "the skeleton on the photo", "0.1 (default 1)"],
                ["2D mesh points", "vertices projected into the image vs the 2D labels", "the surface on the photo", "0.1 (default 1)"]];
  table(s, rows, 0.5, 1.2, 9.0, [1.6, 4.0, 1.75, 1.65], { fs: 9.5, rowH: [0.28].concat(rows.slice(1).map(() => 0.28)), alignBody: "left", name: "loss table" });
  howToRead(s, "the total loss is the weighted sum of the rows; the 3D rows dominate, the 2D rows are small. “Default” is Multi-HMR's value where ours differs; the control keeps distance and position at 1. All distances are L1 (absolute differences).", 4.4, 0.4);
  s.addText([{ text: "Masked for 3DPW people: ", options: { bold: true, color: THEME.colors.dk1 } },
             { text: "the 38 hand bones in the bone-rotation loss (their labels are the rest pose, not real fingers). In the masked arm (slide 19) also the 8 constant shape values (age 0.55, seven at 0.5)." }],
    { x: 0.5, y: 4.78, w: 9.0, h: 0.36, fontSize: 11, color: C.text2, margin: 0, isTextBox: true, valign: "top", objectName: "loss mask line" });
}
N(s, "What the model learns: the losses in plain words", "Training minimises a weighted sum of ten terms, and this table says what each one compares in plain words. Two find the people: a detection score per grid cell, and a small offset that places the head exactly. Four describe the person: bone rotations, the eleven shape values, the distance from the camera, and the 3D position. Four compare the resulting body with the truth: joints and mesh points in 3D, both centred on the pelvis, and the same projected onto the photo. The last column is the weight in the final model, with Multi-HMR's default where ours differs: shape is twenty instead of one, the two depth terms are five instead of one, that is the depth fix, and the 2D terms are small. For 3DPW people the hand bones are not scored, because their labels are a rest pose, not real fingers. The masked arm on slide 19 also drops the eight constant shape values.");

// 8 what changed in training, v5 -> final model (exp7_final_model.sh header and :760-771; numbers: handover 0b final-model bullet
// and the 3DPW-test table: control 91.5, final 89.0, v5 121.8; training time 4,292 s on an A100, handover section 5 item 7)
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Training" });
s.addText("What we changed in training: v5 → final model", { placeholder: "title" });
{
  const rows = [["change", "why", "what it does (final vs control)"],
                ["Depth targets at the person centre (the head) + distance and position losses ×5", "the model anchors each body at its head, but the targets were built at the pelvis (slide 12)", "bodies drawn at 0.78 of their true size (control 0.59); root error 0.73 m (control 1.40 m)"],
                ["Bone fix on 4 bones at the output (0 parameters)", "neck02, head and two hand bones sit half a turn off in every v5-line checkpoint (slide 15)", "neck02 4.8° (control 175.8°); neck thickness 1.00 (0.46); fingers 8.3° (17.4°)"],
                ["30% real 3DPW-train frames in every batch (v23 labels)", "synthetic-only training gave men female-looking bodies on photos; requirement 2a", "3DPW test PVE 89.0 mm; the control, with the same mix, 91.5; v5 121.8"],
                ["3DPW hand losses masked", "the converted labels keep the fingers at rest: no real finger labels", "fingers learned from AnnyOne only"],
                ["Fine-tune 10 epochs from v5: 40,000 images (batch 4), lr 1e-5; last 4 backbone blocks + heads trained", "short and cheap (72 min on an A100), comparable with the 10-epoch pilots", "both arms end above v5's 70.5 mm on the holdout: the restart cost seen in every fine-tune"],
                ["A control with the same recipe minus the depth and bone fixes", "isolates what the two fixes add on top of training on 3DPW", "the “control” column on slide 20"]];
  table(s, rows, 0.5, 1.2, 9.0, [3.0, 3.1, 2.9], { fs: 9.5, rowH: [0.28, 0.5, 0.46, 0.46, 0.36, 0.5, 0.4], alignBody: "left", name: "changes table" });
  howToRead(s, "each row is one difference between v5 and the final model (job 3611336); the right column compares with the control (job 3611339), which shares everything except rows 1 and 2. One run each, no seed: differences under ~8 mm PVE or ~2 recall points are noise.", 4.7, 0.42);
}
N(s, "What we changed in training: v5 → final model", "The final model is v5 fine-tuned for ten epochs with five changes. First, the depth fix: the training targets for position and distance are built at the head, which is where the model anchors each body, and the two depth losses are five times stronger. Second, the bone fix: four bones are turned back by a constant at the output, with no new parameters. Third, thirty percent of every batch are real 3DPW frames with the corrected labels. Fourth, the hand losses are not scored for 3DPW people, because their finger labels are a rest pose. Fifth, the schedule: forty thousand images, about seventy minutes on one GPU. The control shares everything except the depth and bone fixes, so the right column shows what those two add: bodies drawn at 78 instead of 59 percent of their size, a normal neck, half the finger error, and about two millimetres on the 3DPW test. Each arm ran once, so small differences are noise.");

// 9 how we measure (definitions as used in the notes of slides 12-20; noise floor: handover 3.6, 7.6 mm PVE / 1.9 deg / ~2 recall points)
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Training" });
s.addText("How we measure", { placeholder: "title" });
{
  const rows = [["metric", "what it measures", "how to read it"],
                ["PVE (mm)", "mean distance between predicted and true mesh points, pelvis aligned", "lower is better; our headline number"],
                ["PA-PVE (mm)", "the same after also matching rotation and scale", "pose and shape only; ignores placement"],
                ["recall / precision", "true people found / detections that match a true person", "100% = all found / no ghost bodies"],
                ["root error (m)", "3D distance between the predicted and the true pelvis", "placement, mostly depth; 0 = perfect"],
                ["drawn size ratio", "predicted head-to-pelvis length in pixels over the true one", "1 = right size; 0.63 = too small"],
                ["depth slope", "slope of predicted against true depth over the holdout", "1 = perfect; 0 = same depth for all"],
                ["bone angle error (°)", "median angle error per bone vs its parent (neck02, head, fingers)", "~180° = half a turn (flipped bone)"],
                ["neck girth ratio", "predicted neck-skin radius over the true one", "1 = right; 0.46 = half as thick"],
                ["children AUC", "how well predicted age ranks children below adults (Relative Human)", "0.5 = chance, 1 = perfect"],
                ["gender accuracy", "share of people whose predicted gender is on the right side of 0.5", "50% = chance (Relative Human)"]];
  table(s, rows, 0.5, 1.2, 9.0, [1.7, 4.8, 2.5], { fs: 9.5, rowH: [0.28].concat(rows.slice(1).map(() => 0.29)), alignBody: "left", name: "metrics table" });
  howToRead(s, "AnnyOne: 100-image holdout, helper bones masked, matched people; 3DPW: all 8,190 test images against our labels.", 4.5, 0.25);
  s.addText([{ text: "Noise floor: ", options: { bold: true, color: THEME.colors.dk1 } },
             { text: "no fixed random seed; two identical runs differed by 7.6 mm PVE and 1.9° head error. One run only detects about 10 mm PVE, 3° or 2 recall points. Root error has no measured floor." }],
    { x: 0.5, y: 4.76, w: 9.0, h: 0.4, fontSize: 11, color: C.text2, margin: 0, isTextBox: true, valign: "top", objectName: "noise line" });
}
N(s, "How we measure", "Before the findings, the metrics, in plain words. PVE is the mean distance between predicted and true mesh points after lining up the pelvis; it is our headline number, and lower is better. PA-PVE also matches rotation and scale, so it measures pose and shape alone. Recall is the share of true people found, precision the share of detections that match a real person. The root error is the 3D distance between the predicted and the true pelvis, mostly a depth error. The drawn size ratio says how large the body is on the photo compared with the truth; one is right. The depth slope says whether near and far people are told apart; one is perfect, zero means everyone at the same depth. Bone angle errors near 180 degrees mean a flipped bone. The neck girth ratio compares neck thickness. On Relative Human we report the children AUC and gender accuracy, where 0.5 and 50 percent are chance. And the noise floor: two identical runs differed by 7.6 millimetres, so one run only detects about ten.");

// 10 where we started (reused)
// 2 where we started
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Findings and fixes" });
s.addText("Where we started: three visible failures", { placeholder: "title" });
const weekRuns = (before, week) => [{ text: before + "\n\n" }, { text: "This week: ", options: { bold: true, color: THEME.colors.dk1 } }, { text: week }];
card(s, 0.5, 1.4, 2.9, 3.3, "Neck", weekRuns("Necks looked too long or bent wrong in the overlays.\n\nThesis assumption so far: a rig or body-model problem.", "the rig is fine. v5's neck02 and head bone are half a turn off; neck skin half as thick. A pilot fix restores it (one run; on real images by eye only)."), THEME.colors.accent1);
card(s, 3.55, 1.4, 2.9, 3.3, "Fingers", weekRuns("Fingers came out claw-like; demos already zero them.\n\nAssumption so far: a hand-resolution or hand-branch problem.", "test-time zoom and a loss rebalance did not help; half the error was two flipped bones."), THEME.colors.accent2);
card(s, 6.6, 1.4, 2.9, 3.3, "Children", weekRuns("Children reconstructed with adult proportions.\n\nAssumption so far: age is not learned.", "child-like on training images, adult on real photos."), THEME.colors.accent6);
N(s, "Where we started: three visible failures", "These are the three failures we discussed last time. Until this week they were explained by guesses: a rig problem, too few pixels on the hands, or a shape head that does not learn age. None of these guesses had been tested on its own. That is what I did first. The last line on each card is what this week showed; the details follow. For the neck, two bones in the v5 models, neck02 and the head bone, are turned half a turn, and the neck skin comes out about half as thick. A first fix, in one run, restores it; holdout renders of the correction alone show normal necks, and on real images the fine-tuned fixed model's necks look normal, judged by eye only; there is no neck metric for real images. For the fingers, about half of the error came from two flipped hand bones.");

// 11 checked first: what is NOT the cause (merge of the old slides 3 and 4: the six checks as one strip, the four results below)
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Findings and fixes" });
s.addText("Checked first: what is NOT the cause", { placeholder: "title" });
{
  const checks = [["1", "Ground truth", "render the labels, no network"], ["2", "Body model", "move one joint; sweep age"], ["3", "Training signal", "do all outputs get gradients?"],
                  ["4", "Predictions vs truth", "100-image holdout + 55-image set"], ["5", "Shape swap", "true shape into the predicted pose"], ["6", "Overfit test", "memorise 40 images (optimiser only)"]];
  const bw = 1.35, gap = 0.18, by = 1.2, bh = 0.95;
  checks.forEach((c, i) => {
    const x = 0.5 + i * (bw + gap);
    s.addShape(pres.ShapeType.roundRect, { x, y: by, w: bw, h: bh, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.08, objectName: "check box " + c[0] });
    s.addShape(pres.ShapeType.ellipse, { x: x + 0.08, y: by + 0.08, w: 0.3, h: 0.3, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: "check dot " + c[0] });
    s.addText(c[0], { x: x + 0.08, y: by + 0.08, w: 0.3, h: 0.3, fontSize: 11, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "check num " + c[0] });
    s.addText(c[1], { x: x + 0.42, y: by + 0.06, w: bw - 0.48, h: 0.34, fontSize: 10.5, bold: true, color: C.text1, valign: "middle", margin: 0, isTextBox: true, objectName: "check head " + c[0] });
    s.addText(c[2], { x: x + 0.08, y: by + 0.44, w: bw - 0.16, h: 0.48, fontSize: 9.5, color: C.text2, valign: "top", margin: 0, isTextBox: true, objectName: "check body " + c[0] });
  });
  // results: three pass, one pass with warnings (amber "!")
  const oks = [["Ground truth decodes correctly", "AnnyOne labels render onto the people; camera matches on 100% of images. 3DPW labels: slides 5–6.", "ok"],
               ["Body model is fine", "All 16 single-joint moves hit the intended part. Age 0 → 1: skeletal height 0.53 m → 1.54 m.", "ok"],
               ["Gradients reach every bone and shape output", "All 75 active bones and all 11 shape outputs get gradient; the mesh responds. (Two of three camera outputs are unused.)", "ok"],
               ["Optimiser works, with two warnings", "40 training images: error falls 36%. But the position and 2D losses rose while memorising (1.67 → 1.89; 58 → 75): possibly the first sign of the depth bug (slide 12). The summed rotation loss barely moved (44.6 → 40.8).", "warn"]];
  oks.forEach((o, i) => { const y = 2.32 + i * 0.63; const warn = o[2] === "warn"; const col = warn ? C.accent2 : C.accent3;
    s.addShape(pres.ShapeType.ellipse, { x: 0.5, y: y + 0.05, w: 0.4, h: 0.4, fill: { color: col }, line: { color: col }, objectName: "ok " + i });
    s.addText(warn ? "!" : "✓", { x: 0.5, y: y + 0.05, w: 0.4, h: 0.4, fontSize: 15, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "tick " + i });
    s.addText(o[0], { x: 1.05, y, w: 8.45, h: 0.28, fontSize: 13, bold: true, color: C.text1, margin: 0, isTextBox: true, objectName: "okh " + i });
    s.addText(o[1], { x: 1.05, y: y + 0.28, w: 8.45, h: warn ? 0.44 : 0.3, fontSize: 10.5, color: C.text2, margin: 0, isTextBox: true, valign: "top", objectName: "okb " + i }); });
  s.addText("So the failures are not rig, decoding or body-model problems, as the draft said. The cause had to be in training: next slides.",
    { x: 0.5, y: 4.95, w: 9.0, h: 0.2, fontSize: 10.5, bold: true, color: THEME.colors.dk1, margin: 0, isTextBox: true, valign: "top", objectName: "checks conclusion" });
}
N(s, "Checked first: what is NOT the cause", "Before changing anything I tested each stage on its own, with six checks. Render the labels without the network: do they sit on the people? Move one joint of the body model at a time: does the mesh react correctly? Do all outputs receive gradients? Compare predictions with the truth on a fixed holdout of 100 images and a fixed set of 55 adults, children, hands and hard necks. Put the true shape into the predicted pose. And an overfit test: can the optimiser memorise 40 images? The results below: the ground truth decodes correctly, the body model is fine, and gradients reach every bone and every shape output. The optimiser works, with two warnings: while the images were memorised, the position and 2D losses went up instead of down, possibly the first sign of the depth bug, although that is a guess, and the summed rotation loss barely moved. So the failures are not rig or decoding problems, which is how the thesis described them until now. The cause had to be in training.");

// 12-14 depth bug (reused)
// 11 the hidden bug
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Findings and fixes" });
s.addText("The hidden depth bug: anchor vs target", { placeholder: "title" });
// diagram: two stick figures
function figure(slide, x, y, label, anchorY, color, note) {
  slide.addShape(pres.ShapeType.ellipse, { x: x + 0.45, y, w: 0.5, h: 0.5, fill: { color: C.background2 }, line: { color: C.text2, width: 1.5 }, objectName: label + " head" });
  slide.addShape(pres.ShapeType.line, { x: x + 0.7, y: y + 0.5, w: 0, h: 1.3, line: { color: C.text2, width: 2 }, objectName: label + " body" });
  slide.addShape(pres.ShapeType.rect, { x: x + 0.2, y: y + 0.78, w: 1.0, h: 0.03, fill: { color: C.text2 }, line: { color: C.text2 }, objectName: label + " arms" });
  slide.addShape(pres.ShapeType.line, { x: x + 0.25, y: y + 1.8, w: 0.45, h: 0.9, line: { color: C.text2, width: 2 }, flipH: true, objectName: label + " legL" });
  slide.addShape(pres.ShapeType.line, { x: x + 0.7, y: y + 1.8, w: 0.45, h: 0.9, line: { color: C.text2, width: 2 }, objectName: label + " legR" });
  slide.addShape(pres.ShapeType.ellipse, { x: x + 0.58, y: anchorY - 0.12, w: 0.24, h: 0.24, fill: { color }, line: { color }, objectName: label + " anchor" });
  slide.addText(label, { x: x - 0.4, y: y + 2.8, w: 2.2, h: 0.3, fontSize: 12, bold: true, color: C.text1, align: "center", margin: 0, isTextBox: true, objectName: label + " cap" });
  slide.addText(note, { x: x - 0.4, y: y + 3.1, w: 2.2, h: 0.5, fontSize: 11, color: C.text2, align: "center", margin: 0, isTextBox: true, objectName: label + " note" });
}
figure(s, 0.9, 1.35, "Model: anchor = head", 1.6, THEME.colors.accent1, "model.py places the mesh by its head joint");
figure(s, 3.6, 1.35, "Training target = pelvis", 3.15, THEME.colors.accent2, "train.py built transl / dist / heatmap / offset at joint 0");
s.addText([{ text: "predicted depth =", options: { fontSize: 16, breakLine: true } }, { text: "0.33 × true + 2.36 m", options: { fontSize: 24 } }],
  { x: 6.5, y: 1.4, w: 3.0, h: 0.8, bold: true, color: C.accent4, margin: 0, isTextBox: true, valign: "top", objectName: "stat depth" });
s.addText("v5: predicted vs true depth; all 222 holdout people too far", { x: 6.5, y: 2.2, w: 3.0, h: 0.6, fontSize: 13, color: C.text2, margin: 0, isTextBox: true, valign: "top", objectName: "label depth" });
s.addText(bulletRuns([
    ["Confirmed: ", "with targets at the head, bodies are drawn much closer to the right size; a same-schedule control does not improve (10 epochs: size 0.59, root error 1.44 m)."],
    ["Size: ", "3D height right, but drawn at 0.63 of true size; head 64 px off."],
    ["Hidden ", "from pelvis-centred PVE and from the box-placed demos. v9 and v10 have it too."]], { space: 4 }),
  { x: 6.5, y: 2.95, w: 3.0, h: 2.15, fontSize: 12, color: C.text1, margin: 0, isTextBox: true, valign: "top", objectName: "bug text" });
N(s, "The hidden depth bug: anchor vs target", "This is the main result of the week. The model attaches the mesh to the detected pixel by its head joint and reports the head as its position. The training code built the position, distance, heat-map and offset targets at joint 0, the root joint at the pelvis. Those two can never agree. The cause is confirmed: with targets at the head the bodies are drawn much closer to the right size, and a control run with the same schedule does not improve; after 10 epochs its drawn size is 0.59 and its root error 1.44 metres. The likely mechanism is that the 2D losses solve the conflict by pushing the body farther away, so it is drawn smaller. Measured on the holdout for v5: predicted depth is 0.33 times the true depth plus 2.36 metres, all 222 people too far, and the bodies are drawn at about two thirds of their true size in the image, a size ratio of 0.63. Their 3D height is right, 1.405 against 1.409 metres; they are only placed too far away. Our standard error is pelvis-centred, so it never saw this. That is why the thesis numbers looked normal. The real-photo demos did not show it either, because they place each body from its detection box. The synthetic demo and the trace panels use the model's own depth, and there you can see it, as on the next slide. The bug is in the training code, so v9 and v10 have it too: the same kind of depth fit and size ratio, and v10 is slightly worse.");

// 12 before / after figure (8.0 in wide so the title and the footer keep their usual spacing)
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Findings and fixes" });
s.addText("Depth fix: before and after", { placeholder: "title" });
let figBottom = figureSlide(s, "depth_before_after_579862.jpg", 1.0, 1.2, 8.0, [
  "Synthetic holdout image 579862, model's own depth (no box placement).",
  "v5, whole holdout: root error 1.21 m, median drawn size 0.63",
  "Fix + depth ×5, 10 epochs, whole holdout: root error 0.54 m, median drawn size 0.84"], 0.4);
footnote(s, "Not fully fixed: ", "near and far people still get similar depths (depth slope 0.38; 1 = perfect, 0 = same depth for everyone), and bodies are still drawn about 16% too small (median) in this pilot; the final model reaches 0.78, 22% too small (slide 20). All 12 rendered images show the same change. After the fix the blue mesh is 0.95 of its true height (before: 0.76).", figBottom + 0.5, 0.56);
N(s, "Depth fix: before and after", "This is the bug on a synthetic holdout image, using the model's own depth, not the box placement of the real-photo demos. Left is the truth. In the middle, the old model: the bodies are drawn too small, the real man's head sticks out above the mesh, and the mesh floats. On the right, the fix with the depth loss five times stronger, trained for 10 epochs. The blue mesh is now 95 percent of the true height; the woman's mesh is closer to her size but still narrower. I chose this image because no person with a ground-truth mesh is missed, so it is a good case, not a typical one; the arm at the bottom left has no ground-truth mesh, so it is not counted as a miss. On the whole holdout the median drawn size is 84 percent of the truth, so bodies are still about 16 percent too small in this pilot run, 22 percent in the final model, and near and far people still get similar depths: in the slope, 1 means perfect and 0 means the same depth for everyone. All 12 holdout images we rendered, in all three prediction settings, show the same change. In 4 of them both models miss people very close to the camera; since both miss them, the fix did not cause it. It is likely a detection problem, but not yet counted. Mesh colours follow detection order. The final model on slide 20 uses the same depth fix: root error 0.73 metres, against 1.40 for its own control and 1.21 for v5.");

// 13 depth fix results: all seven runs. A group band separates the 10-epoch and the 2-epoch fine-tunes;
// the column shown on slide 13 (fix + x5, 10 epochs) is tinted teal
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Findings and fixes" });
s.addText("Depth fix: the numbers", { placeholder: "title" });
{
  const thin = { type: "solid", pt: 1, color: "FFFFFF" }, none = { type: "none" };
  const sepBody = { type: "solid", pt: 3, color: THEME.colors.dk2 }, sepHead = { type: "solid", pt: 3, color: "FFFFFF" };
  const HI = "D3ECEE";                         // light teal tint for the slide 13 column
  const left = c => c === 2 || c === 5, right = c => c === 1 || c === 4;
  const head = ["holdout, 100 images", "v5", "control", "fix", "fix + ×5", "control", "fix", "fix + ×5", "fix + ×20"];
  const body = [["root error, 3D (pelvis)", "1.21 m", "1.44 m", "0.73 m", "0.54 m", "1.11 m", "0.69 m", "0.66 m", "0.53 m"],
                ["depth slope (1 = perfect)", "0.33", "0.34", "0.30", "0.38", "0.33", "0.30", "0.44", "0.43"],
                ["drawn size (1 = right)", "0.63", "0.59", "0.78", "0.84", "0.66", "0.78", "0.80", "0.87"],
                ["head pixel error", "64.5 px", "64.0 px", "7.6 px", "7.2 px", "58.1 px", "9.6 px", "8.5 px", "8.8 px"],
                ["pelvis pixel error", "23.4 px", "24.3 px", "38.5 px", "33.1 px", "23.3 px", "37.3 px", "38.1 px", "31.4 px"],
                ["PVE (pelvis-centred)", "70.5 mm", "72.2 mm", "75.5 mm", "74.8 mm", "72.3 mm", "77.3 mm", "76.8 mm", "75.7 mm"],
                ["recall", "90.2%", "89.4%", "87.0%", "87.8%", "89.8%", "87.8%", "87.8%", "87.4%"]];
  const base = { fontSize: 11, valign: "middle", margin: [0.02, 0.05, 0.02, 0.05] };
  const band = (text, colspan, sepL) => ({ text, options: { ...base, colspan, bold: true, color: THEME.colors.lt1, fill: { color: THEME.colors.dk2 }, align: "center", border: [thin, sepHead, thin, sepL ? sepHead : thin] } });
  const rows = [[{ text: "", options: { ...base, colspan: 2, fill: { color: THEME.colors.lt1 }, border: [none, sepHead, thin, none] } },
                 band("10-epoch fine-tunes", 3, true), band("2-epoch fine-tunes", 4, true)],
                head.map((t, c) => ({ text: t, options: { ...base, bold: true, color: THEME.colors.lt1, fill: { color: c === 4 ? THEME.colors.accent1 : THEME.colors.dk1 }, align: c === 0 ? "left" : "center",
                                                          border: [thin, right(c) ? sepHead : thin, thin, left(c) ? sepHead : thin] } }))];
  body.forEach((r, i) => rows.push(r.map((t, c) => ({ text: t, options: { ...base, bold: c === 0, color: THEME.colors.dk1, align: c === 0 ? "left" : "center",
      fill: { color: c === 4 ? HI : (i % 2 ? THEME.colors.lt2 : THEME.colors.lt1) }, border: [thin, right(c) ? sepBody : thin, thin, left(c) ? sepBody : thin] } }))));
  const cw = 6.2 / 7;
  s.addTable(rows, { x: 0.5, y: 1.2, w: 9.0, colW: [2.0, 0.8, cw, cw, cw, cw, cw, cw, cw], rowH: [0.25, 0.27, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24], margin: [0.02, 0.05, 0.02, 0.05], objectName: "fix table" });
}
s.addText("Fix: targets at the head; ×5 / ×20 = depth loss weight. Controls: root targets, depth ×1, same schedule. No control was run at ×5 or ×20.",
  { x: 0.5, y: 3.45, w: 9.0, h: 0.22, fontSize: 11, color: C.text2, margin: 0, isTextBox: true, valign: "top", objectName: "fix table note" });
s.addText(bulletRuns([
    ["Not fixed: ", "depth slope 0.30 with the switch alone, 0.38–0.44 with a stronger depth loss; predictions are pulled toward 2.5–2.9 m, cause unknown. With ×5, 10 epochs gave a lower root error than 2 epochs (0.66 → 0.54 m) but no better slope (0.44 → 0.38)."],
    ["Likely cost: ", "every fix run is worse than its control: PVE +2.6 to +4.9 mm (inside noise, ~10 mm), recall −1.6 to −2.4 points (about the 2-point noise limit). The five runs share two controls."],
    // 18 Oct evening: exp7, jobs 3611336 / 3611339, compare table (V1): root 728 / 1402 mm (v5 1208), size 0.78 / 0.59, slope 0.42 / 0.28
    ["Final model (slide 20): ", "×5 plus the bone fix, 30% real 3DPW, 10 epochs: root error 0.73 m vs 1.40 m for its own control (v5 1.21 m), size 0.78 vs 0.59, slope 0.42 vs 0.28."]], { space: 3 }),
  { x: 0.5, y: 3.78, w: 9.0, h: 1.34, fontSize: 12, color: C.text1, margin: 0, isTextBox: true, valign: "top", objectName: "fix text" });
N(s, "Depth fix: the numbers", "The fix is one switch: build the training targets at the head, the joint the model anchors. Each fix run has a control with the same schedule and the switch off. The root error is the 3D distance between the predicted and the true root joint, at the pelvis. Drawn size is the predicted head-to-pelvis distance in pixels over the true one; 1 is right. With the depth loss five times stronger for 10 epochs, the root error falls from 1.21 to 0.54 metres, and bodies are drawn at 84 percent of their true size. The controls stay at 1.11 to 1.44 metres. But near and far are still not separated: the slope is 0.30 with the switch alone and 0.38 to 0.44 with the stronger depth loss. Predictions are pulled toward about 2.5 to 2.9 metres, and I do not know why. The fix likely has a cost: every fix run is worse than its control, by 2.6 to 4.9 millimetres PVE and 1.6 to 2.4 recall points. That is inside or at the edge of pilot noise, so I call it a likely cost, not a proven one. The twenty-times 10-epoch run has no result yet. The final model on slide 20 uses the five-times fix together with the bone fix and real photos: root error 0.73 metres against 1.40 for its own control and 1.21 for v5, size 0.78 against 0.59, slope 0.42 against 0.28. So its slope is about as good as the best pilots, but still far from 1, and its measured cost is 2 recall points, with PVE inside the noise.\n\nIf asked: The switch is gt_center_joint person_center. The stronger-loss runs have no control at the same weight. The mean signed depth offset fell from 1.08 to 0.39 metres, but after the fix near people are too far and far people too near, and these cancel in that mean, so it flatters the fix. The head now lands within about 7 to 10 pixels, as expected, because the target moved to the head. Twenty times for 2 epochs ties with five times for 10 epochs. The pull toward 2.5 to 2.9 metres is close to the estimated training average; holdout people are nearer, 1.9 metres on average, so this pull becomes a large error here. Longer training is not clearly better: with five times, going from 2 to 10 epochs lowered the root error from 0.66 to 0.54 metres but did not improve the slope; with the switch alone neither improved, the root error went from 0.69 to 0.73 metres, and the control also got worse, from 1.11 to 1.44 metres. Root error has no measured noise floor, so I do not read a trend into this. The PVE gaps, 2.6 to 4.9 millimetres, are inside pilot noise; the recall drops, 1.6 to 2.4 points, are at the edge of what one pilot can detect; and the five fix runs share only two controls, so they are not five independent tests. The pelvis 2D error clearly got worse, from about 23 pixels to between 31 and 39 pixels; probably because the body is still drawn too small while the head is now pinned, but I have not checked that.");

// 15-17 neck and fingers (reused)
// 7 neck: the cause (per-bone dumps: diag_neck_bones_3573518 / _3573519, exp5_neck_fix_3573520 / _3573521; "v5 line" =
// 8 checkpoints: frozen_bb_v2, v3, v4, v5, v9, long_pc_dist5, exp5 L1, exp5 geodesic) and the exp6 pilot fix
// (exp6_bone_fix_3579014, run anny_s2_shape_v5_00099_n200_s0, final SUMMARY table: baseline | zero-training |
// fine-tuned | control = exp5 L1 recipe without the fix). All V1_eval_masked, 100-image holdout, matched people.
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Findings and fixes" });
s.addText("Neck: two half-turned bones, and a pilot fix", { placeholder: "title" });
{
  const BAD = THEME.colors.accent4, HI = "D3ECEE";   // red = flipped or failed; teal = the fine-tuned fix (as on slide 14)
  // [label, v5 before, fix with no training, fix + 2-epoch fine-tune, control, columns shown in red]
  const rows = [["neck02", "175.6°", "4.5°", "4.7°", "175.7°", [1, 4]],
                ["head", "177.9°", "12.4°", "11.2°", "177.9°", [1, 4]],
                ["neck thickness", "0.46", "1.00", "1.00", "0.46", [1, 4]],
                ["neck-skin error", "27.3 mm", "10.8 mm", "10.7 mm", "27.9 mm", [1, 4]],
                ["body PVE", "70.5 mm", "71.3 mm", "68.8 mm", "76.9 mm", []],
                ["head direction", "12.0°", "19.1°", "14.4°", "13.8°", [2]]];
  const base = { fontSize: 12, valign: "middle", margin: [0.02, 0.06, 0.02, 0.06] };
  const head = ["holdout, 100 images", "v5 before", "fix, no training", "fix + 2-epoch fine-tune", "control (no\u00A0fix)"].map((t, c) => ({ text: t, options: { ...base, fontSize: 11, bold: true,
    color: THEME.colors.lt1, fill: { color: c === 3 ? THEME.colors.accent1 : THEME.colors.dk1 }, align: c === 0 ? "left" : "center" } }));
  const data = [head].concat(rows.map((r, i) => [0, 1, 2, 3, 4].map(c => ({ text: r[c], options: { ...base,
    bold: c === 0 || r[5].includes(c), color: r[5].includes(c) ? BAD : THEME.colors.dk1, align: c === 0 ? "left" : "center",
    fill: { color: c === 3 ? HI : (i % 2 ? THEME.colors.lt2 : THEME.colors.lt1) } } }))));
  s.addTable(data, { x: 0.5, y: 1.3, w: 5.3, colW: [1.45, 0.9, 0.95, 1.05, 0.95], rowH: [0.42, 0.27, 0.27, 0.27, 0.27, 0.27, 0.27], border: { type: "solid", color: "FFFFFF", pt: 1 }, objectName: "neck table" });
  s.addText("Bones: median error vs truth, relative to the parent bone. Thickness: neck-skin radius, predicted / true. Neck-skin error: centred at neck01. Head direction: error of the whole head's direction.",
    { x: 0.5, y: 3.42, w: 5.3, h: 0.6, fontSize: 11, color: C.text2, margin: 0, isTextBox: true, valign: "top", objectName: "neck table note" });
  s.addText(bulletRuns([
      ["Cause, measured: ", "in all 8 v5-line checkpoints, neck02 and head sit half a turn off. They cancel, so PVE hid it, but the neck skin is half as thick."],
      ["Fix (exp6): ", "one constant turn of about 179° per flipped bone at the network output, estimated on 200 training images. It also turns back two hand bones (slide 17)."],
      ["Fix, no training: ", "head direction fails its 15° limit; likely a per-person head offset."],
      ["Fix + 2-epoch fine-tune: ", "passes the fine-tune checks, each within a tolerance of its control."]], { space: 5 }),
    { x: 6.05, y: 1.3, w: 3.45, h: 2.92, fontSize: 12, color: C.text1, margin: 0, isTextBox: true, valign: "top", objectName: "neck text" });
  // caveat, open, final model: one line each, dot colour = status (amber = caveat, grey = unknown, teal = the final model;
  // until 18 Oct the teal line was "Next: a longer fine-tune with the fix"). Final-model numbers: exp7 fix arm, job
  // 3611336, compare table (V1): neck02 4.8, head 8.9, girth 0.996, PVE_neck 11.0 mm, global head 13.1 deg
  s.addShape(pres.ShapeType.roundRect, { x: 0.5, y: 4.3, w: 9.0, h: 0.82, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.08, objectName: "neck cause box" });
  [["Caveat: ", "one run, no fixed seed (identical runs differ by up to 7.6 mm PVE), so no PVE gain is claimed.", C.accent2],
   ["Open: ", "why the bones flipped. Next slide: the fix alone on a holdout image; fix + fine-tune on a real 3DPW frame.", C.accent5],
   ["Final model, 10 epochs (slide\u00A020): ", "neck02 4.8°, head 8.9°, thickness 1.00, neck-skin error 11.0 mm, head direction 13.1°.", C.accent1]].forEach((l, i) => {
    const y = 4.35 + i * 0.25;
    s.addShape(pres.ShapeType.ellipse, { x: 0.68, y: y + 0.06, w: 0.13, h: 0.13, fill: { color: l[2] }, line: { color: l[2] }, objectName: "neck cause dot " + i });
    s.addText([{ text: l[0], options: { bold: true, color: THEME.colors.dk1 } }, { text: l[1] }],
      { x: 0.92, y, w: 8.45, h: 0.25, fontSize: 12, color: C.text1, margin: 0, isTextBox: true, valign: "middle", objectName: "neck cause " + i });
  });
}
N(s, "Neck: two half-turned bones, and a pilot fix", "In all eight v5-line checkpoints, neck02 and the head bone are about half a turn off. The two half-turns cancel, so the head still points the right way and PVE never showed it, but the neck skin comes out half as thick. I tested a fix: one constant turn per flipped bone at the network output, estimated on 200 training images. On the holdout, the fix alone turns both bones back; neck thickness goes from 0.46 to 1.00. But head direction gets worse, 19.1 degrees, likely because the head's offset differs per person. After a two-epoch fine-tune, it passes the fine-tune checks, each within a tolerance of its control; head direction is 14.4, still worse than 12.0 before. One run, no fixed seed: I claim no PVE gain, only no sign of a PVE cost, 68.8 against 76.9 for the control. The final model, 10 epochs with the fix and with real photos in training, keeps it: neck02 4.8 degrees, thickness 1.00, head direction 13.1 (slide 20). The next slide shows the correction alone on a holdout image, and the fine-tuned model on a real 3DPW test frame.\n\nIf asked: After the fine-tune, neck02 is 4.7 degrees and the thickness 1.00. The flip is there from the first training stage, frozen_bb_v2, and in v3, v4, v5, v9 and every pilot checked. A two-epoch geodesic loss barely moved it, 175.6 to 175.0 degrees. Its origin is not known: the singular start value of the rotation output would start a bone with a tilt, like v5's left index base, not this twist. v10 has no neck flip. The constants are 178.8 degrees for neck02 and 179.2 for the head. Before the test, the script's guard predicted the head-direction rise, 8.4 to 17.9 degrees on the training sample: a compensated chain, since the head's offset differs per person and one constant cannot undo it. Without training, the head-and-neck surface error also rose, 52.3 to 63.1 millimetres; after the fine-tune it is 51.8. The fine-tune checks, against the control: corrected bones under 15 degrees, thickness at least 0.9, PVE at most 8 millimetres higher, recall at most 2 points lower, head direction at most 3 degrees worse. Recall passed by only 0.3 points, 89.4 against a limit of 89.1. Against the control: PVE 68.8 against 76.9, PA-PVE 50.9 against 55.2, recall 89.4 against 91.1, head direction 14.4 against 13.8. Against v5 before: PVE 68.8 against 70.5, PA-PVE 50.9 against 53.3, recall 89.4 against 90.2. Identical configs differ by up to 7.6 millimetres PVE, so none of these PVE differences is certain. Last week's group errors, 63 degrees for the neck and 90 for the head, are now 5.9 and 6.7. The script wrote 12 side-by-side panels, without and with the correction alone; the next slide shows one of them, next to a real-image demo of the fine-tuned checkpoint, which carries the fix. The longer run is the final model: neck02 4.8 degrees, head bone 8.9, thickness 1.00, neck-skin error 11.0 millimetres, head direction 13.1 against 13.0 for its own control (slide 20).");

// 8 neck, visual check: one synthetic pair and one real pair, before | after each.
// Left pair: 1 of the 12 exp6 compare panels (results/experiments/exp6_bone_fix/anny_s2_shape_v5_00099_n200_s0/compare/,
// holdout 579860-579871; no fix vs fix with zero training, job 3579014), regrouped by the student into
// docs/figures/neck_before_after_v5.jpg (1254 x 1118 px): left column v5 as trained, right column with the correction;
// top row holdout 579862 (the bottom row, 579863, was on this slide until the real pair replaced it).
// Right pair: real-image demo, job 3581312 (settings of slurm/demo/run_demo_compare.sh: person detector, anchor_to_box 1,
// relax_hands 1, mask_helper_joints 1): v5 (anny_s2_shape_v5_00099) vs exp6_v5_bonefix_00001, the exp6 checkpoint
// fine-tuned 2 epochs with the bone fix (neck02, head, metacarpal1.L, finger4-1.R; re-applied on load from the checkpoint
// args, log: "[bone_fix] ON, 4 bone(s)"), on 3DPW TEST downtown_bar_00 frame 701 (never trained on); the student's
// docs/figures/neck_before_after_3dpw_bar.jpg (1134 x 570 px: before | after). Judged by eye only, no metric.
// The four panels are cropped out of the two files (srcRect, so the pptx holds each file once and unchanged) and laid
// out in one row; all four are 518 px tall in their files, so one scale fits all. The baked-in label strips (y 0-44 in
// both files) are cropped off; slide-level header bars replace them, in the red / teal of slides 13 and 29. Second line
// of each bar: the left After panel is the correction with no training; the right After panel is the fine-tuned fixed
// model, the teal column of slide 15, so it differs from v5 by the correction and by 2 epochs of training.
// Panel pixel boxes, measured from the files, inset 1 px against JPEG fringe: neck_before_after_v5.jpg top row
// x 0-619 / 634-1253, y 50-569; neck_before_after_3dpw_bar.jpg x 0-559 / 574-1133, y 50-569 (x 560-573, y 45-49 white).
// Yellow rings mark the necks (centres measured from the files in the Before panel; After panel = the same + dx in x,
// checked by eye in both): 579862 blue (155, 232), orange (480, 335), dx 634; 3DPW bar blue (143, 172), light (334, 172), dx 574.
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Findings and fixes" });
s.addText("Neck: before and after the fix", { placeholder: "title" });
{
  const FIGS = { holdout: { file: "neck_before_after_v5.jpg", px: { w: 1254, h: 1118 }, dx: 634, necks: [[155, 232], [480, 335]],
                            cap: "Synthetic holdout 579862 (correction only, no training)" },
                 bar: { file: "neck_before_after_3dpw_bar.jpg", px: { w: 1134, h: 570 }, dx: 574, necks: [[143, 172], [334, 172]],
                        cap: "3DPW test frame (real, never trained on), downtown_bar_00" } };
  const panels = [{ fig: "holdout", after: false, box: [1, 51, 618, 518], head: ["Before (v5)", "as trained"],
                    alt: "Synthetic holdout image 579862, v5 as trained: thin necks" },
                  { fig: "holdout", after: true, box: [635, 51, 618, 518], head: ["After (correction)", "no training"],
                    alt: "Synthetic holdout image 579862, v5 with the correction on neck02 and the head, no training: normal neck width" },
                  { fig: "bar", after: false, box: [1, 51, 558, 518], head: ["Before (v5)", "real image"],
                    alt: "3DPW test image downtown_bar_00 frame 701, v5: thin necks" },
                  { fig: "bar", after: true, box: [575, 51, 558, 518], head: ["After (fixed model)", "fix + 2-epoch fine-tune"],
                    alt: "3DPW test image downtown_bar_00 frame 701, fixed model after the 2-epoch fine-tune: normal neck width" }];
  const gIn = 0.06, gOut = 0.3, barH = 0.5, x0 = 0.5, barY = 1.2;       // gap inside a pair, gap between the pairs
  const sc = (9.0 - 2 * gIn - gOut) / panels.reduce((a, p) => a + p.box[2], 0);   // in per px: the row is 9.0 in wide
  const ph = 518 * sc, imgY = barY + barH;                               // 1.89 in tall, ends at 3.59
  const ringD = 0.26;
  let x = x0;
  const pairX = {}, pairW = {};
  panels.forEach((p, i) => {
    const F = FIGS[p.fig], [bx, by, bw, bh] = p.box, w = bw * sc;
    if (!p.after) { pairX[p.fig] = x; pairW[p.fig] = 2 * w + gIn; }
    s.addText([{ text: p.head[0], options: { fontSize: 16, bold: true, breakLine: true } },
               { text: p.head[1], options: { fontSize: 11 } }],     // 11 pt: "fix + 2-epoch fine-tune" fits the 2.04 in bar on one line, with room for a wider fallback font
      { x, y: barY, w, h: barH, color: THEME.colors.lt1, fill: { color: p.after ? THEME.colors.accent1 : THEME.colors.accent4 }, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "neck header " + i });
    s.addImage({ path: FIG_DIR + F.file, x, y: imgY, w: F.px.w * sc, h: F.px.h * sc,
      sizing: { type: "crop", x: bx * sc, y: by * sc, w, h: bh * sc },
      altText: p.alt, objectName: "figure " + F.file + (p.after ? " after" : " before") });
    F.necks.forEach(([nx, ny], k) => s.addShape(pres.ShapeType.ellipse, {
      x: x + (nx + (p.after ? F.dx : 0) - bx) * sc - ringD / 2, y: imgY + (ny - by) * sc - ringD / 2, w: ringD, h: ringD,
      line: { color: THEME.colors.accent6, width: 2 }, objectName: "neck ring " + p.fig + (p.after ? " after " : " before ") + k }));
    x += w + (p.after ? gOut : gIn);
  });
  Object.keys(pairX).forEach(f => s.addText(FIGS[f].cap, { x: pairX[f], y: imgY + ph + 0.06, w: pairW[f], h: 0.22, fontSize: 11, color: C.text1, margin: 0, isTextBox: true, valign: "top", objectName: "neck caption " + f }));
  // explicit break per line, so each line ends at a sentence; left-aligned to the row's left edge, like slides 13 and 29
  const lines = (arr, o = {}) => arr.map((t, i) => ({ text: t, options: { ...o, breakLine: i < arr.length - 1 } }));
  // the correction also turns back the two hand bones (metacarpal1.L, finger4-1.R), hence "on the neck";
  // the thickness is the whole 100-image holdout (girth 0.461 -> 1.004), not these images
  s.addText(lines(["Left: same model, no training; on the neck, only neck02 and the head are corrected. Right: fixed model, fine-tuned 2 epochs.",
                   "Neck skin thickness on the 100-image holdout: 0.46 → 1.00 of the true value (job 3579014). Real-image demo: job 3581312."]),
    { x: x0, y: imgY + ph + 0.36, w: 9.0, h: 0.4, fontSize: 11, color: C.text1, margin: 0, isTextBox: true, valign: "top", objectName: "neck figure caption" });
  s.addText([{ text: "Limits: ", options: { bold: true, color: THEME.colors.dk1 } },
             { text: "without fine-tuning the head points 19° off (12° before); the 2-epoch fine-tune brings it to 14°.", options: { breakLine: true } },
             { text: "Holdout bodies are still too small (separate depth bug). Real images judged by eye only: no neck ground truth, no metric." }],
    { x: x0, y: imgY + ph + 0.84, w: 9.0, h: 0.4, fontSize: 11, color: C.text2, margin: 0, isTextBox: true, valign: "top", objectName: "neck figure footnote" });
}
N(s, "Neck: before and after the fix", "Two pairs. On the left, a synthetic holdout image: v5 as trained, then the same model with the correction alone, no training. On the right, a real 3DPW test frame, never trained on: v5, then the fine-tuned fixed model. Look at the necks, ringed in yellow. With v5 they are thin stalks, pinched under the head; with the fix they have normal width. On the holdout this is measured: thickness 0.46 to 1.00 over 100 images. On real images I can only judge by eye. The change is clearest here and on an office frame, and also visible on two more 3DPW sequences and three photos. Head direction stays a limit: 14 degrees after the fine-tune, against 12.\n\nIf asked:\n- The right column is the fine-tuned checkpoint, so it differs from v5 by the correction and by two epochs of training; this pair cannot separate the two. On the holdout, the same 2-epoch fine-tune without the fix keeps thickness 0.46, so the correction is the likely cause; not tested on real images. The demo log shows the fix on for four bones. Demo: job 3581312, person detector, each body placed from its detection box, 4 example photos and 8 3DPW test frames. I have looked at 8 of these 12 images so far; the other four, the runForBus, sitOnStairs and fencing frames and one photo, not yet.\n- Men: in both models, adult men on real photos get female-looking bodies, with breasts and narrow shoulders, for example the male tennis players. v5 never saw 3DPW, so this is not the gender defect in our 3DPW labels. The demo log shows predicted gender near the training median, about 0.6, while Anny's male end is 0: the same pull to the median as for age (slide 18). On Relative Human, measured on 932 real people, men get 0.535 on average and women 0.551: no gender signal. Whether the gap between synthetic and real images or the loss causes it is not settled. The final model, trained with the corrected 3DPW gender labels, gives most men male bodies, judged by eye (slide 21); on Relative Human its gender accuracy is 64 percent, against 46.5 for v5 (slide 18).\n- Also in both models: some bodies are still small or misplaced in depth, although the demo places them from the detection box, and a figure in a painting is detected as a person.\n- The correction also turns back two hand bones, so the hands differ too; I have not judged them (slide 17).\n- All twelve holdout panels show the thin neck and its fix wherever a neck is visible.");

// 9 fingers. Before: 3571960 (predicted 17.3, rest pose 15.1, crop 17.2), 3571959 (region-norm 17.5 vs 17.4).
// Bone fix: exp6_bone_fix_3579014 SUMMARY, row "rot fingers": baseline 17.3, zero-training 8.2, fine-tuned 8.3
// (corrected hand bones metacarpal1.L, finger4-1.R). The group error is a per-person mean over the active hand bones.
// 18 Oct evening: final model (exp7 fix arm, job 3611336) "rot fingers" 8.3, its control (job 3611339) 17.4.
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Findings and fixes" });
s.addText("Fingers: about half the error was two flipped bones", { placeholder: "title" });
stat(s, 0.5, 1.4, 2.8, "17.3°", "finger error, v5 with predicted fingers", C.accent4);
stat(s, 3.5, 1.4, 2.8, "8.3°", "with the bone fix: 2-epoch pilot and 10-epoch final model (controls 17.4°)", C.accent3);
stat(s, 6.6, 1.4, 2.9, "15.1°", "fingers set to rest pose, before the fix (the index-base flip stays)", C.accent5);
table(s, [["hand size at the network input", "median", "under 2 tokens"], ["full image, 672 px", "2.3 tokens", "41%"], ["person crop (box ×1.3), test time", "4.2 tokens", "0%"], ["children's hands, full image", "0.7 tokens", "80%"]], 0.5, 3.25, 5.6, [2.8, 1.4, 1.4], { fs: 12, rowH: 0.36, name: "hand table" });
s.addText("Table: median hand size (2.3 → 4.2 tokens). Per person the median gain is 1.6×; 26% of people get no magnification, a quarter 2.5× or more.", { x: 0.5, y: 4.74, w: 5.6, h: 0.4, fontSize: 11, color: C.text2, margin: 0, isTextBox: true, valign: "top", objectName: "hand table note" });
s.addText([{ text: "Two flipped bones: ", options: { bold: true, paraSpaceAfter: 6 } }, { text: "left index base and finger4-1.R, each about half a turn off. Turning them back halves the error; not yet checked by eye.", options: { breakLine: true } },
             { text: "Test-time zoom: ", options: { bold: true, paraSpaceAfter: 6 } }, { text: "17.2° vs 17.3°; body error nearly doubles.", options: { breakLine: true } },
             { text: "Loss rebalance ", options: { bold: true } }, { text: "(hands 33% of the rotation loss): 17.5° vs 17.4°." }],
  { x: 6.4, y: 3.25, w: 3.1, h: 1.9, fontSize: 12, color: C.text1, margin: 0, isTextBox: true, valign: "top", objectName: "finger text" });
N(s, "Fingers: about half the error was two flipped bones", "With predicted fingers the finger error is 17.3 degrees; with the fingers left at rest it is 15.1. The new result: about half of the 17.3 came from two hand bones turned half a turn, the left index base and one right finger bone. The same correction as for the neck turns them back: the error falls to 8.2 degrees with no training and 8.3 after the fine-tune. The final model, 10 epochs with the fix and with real photos in training, keeps it: 8.3 against 17.4 for its own control. I have not yet looked at the hands in the renders. The rest-pose number still contains the index-base flip, because rest pose does not reset that bone, and I have not measured it with the fix. What I tested before did not help: magnifying the person at test time gave 17.2 against 17.3, and a rebalanced loss gave 17.5 against 17.4 for its control.\n\nIf asked: The finger error is a mean over the hand bones of each person, so two bones at half a turn raise it a lot. The left index base is 180 degrees off on the holdout. The other bone, finger4-1.R, at 178.8 degrees, was seen only on the 200 training images, because the earlier holdout dump did not cover finger bones; with the fix it is 10.2 degrees on the holdout. Hands are small: about two tokens at 672 pixels. Cropping around the person at test time, without retraining, magnifies the hands by 1.6 times on median per person, and by 2.5 times or more for a quarter; 26 percent are not magnified. The table compares median hand sizes, so it shows a slightly larger factor. For the most magnified quarter the finger error went from 17.7 to 17.5. The body error nearly doubles on these crops, because the model never saw such a camera in training. The rebalanced loss cut the hands' share of the rotation loss from about 70 percent, which includes the four-times hand boost, to 33 percent; without the boost it would be 40. So it does not test giving the fingers more weight; more weight and training on larger hand crops are untested. v10 has at least five flipped hand bones. Correcting three of them, with no training, took its finger error from 32.0 to 17.8 degrees and PVE from 74.1 to 69.9 millimetres; PA-PVE went from 55.6 to 51.1, while head direction and recall did not change. Two more, finger2-1.L and finger5-2.L, seen on training images, are not corrected yet; they may be part of why v10's fingers are still at 17.8, but that is a guess. The demos zero the fingers, but rest pose does not reset the hand bases, so the bent index base is likely visible there.");

// 18 children and gender on real photos: v5 / final / masked arm on Relative Human val (635 images, 2,130 labelled people, IoU 0.3)
// v5: job 3611285_0 (recall 0.471, gender 0.465, AUC 0.606, age adult 0.631 / kid 0.613 / baby 0.624, babies < 0.45: 1 of 38 = 2.6%)
// final: job 3612131 (0.647, 0.642, 0.719, 0.536 / 0.500 / 0.438, 49.3%); masked arm: job 3613616 (0.572, 0.493, 0.708,
// 0.638 / 0.611 / 0.524, 16.9%). Sources: handover 0a (18 Oct 15:30 block) and 0b; docs/thesis_requirements_plan.md 3.1 item 7.
// The three stat tiles are the old slide 10's, trimmed (sampler 2% -> 10%, job 3571601; training babies 0.10 vs 0.015, 3569950:61).
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Findings and fixes" });
s.addText("Children and gender on real photos", { placeholder: "title" });
{
  const HI = "D3ECEE", BAD = THEME.colors.accent4;
  const rows = [["recall (people found)", "47%", "65%", "57%", []],
                ["gender accuracy", "46.5%", "64.2%", "49.3%", [1, 3]],
                ["children AUC (0.5 = chance)", "0.61", "0.72", "0.71", []],
                ["mean predicted age, adults", "0.63", "0.54", "0.64", []],
                ["mean predicted age, kids", "0.61", "0.50", "0.61", []],
                ["mean predicted age, babies", "0.62", "0.44", "0.52", []],
                ["babies below the child line 0.45", "3%", "49%", "17%", []]];
  const base = { fontSize: 11, valign: "middle", margin: [0.02, 0.06, 0.02, 0.06] };
  const head = ["Relative Human val, 2,130 people", "v5", "final model", "masked arm"].map((t, c) => ({ text: t, options: { ...base, fontSize: 10.5, bold: true,
    color: THEME.colors.lt1, fill: { color: c === 2 ? THEME.colors.accent1 : THEME.colors.dk1 }, align: c === 0 ? "left" : "center" } }));
  const data = [head].concat(rows.map((r, i) => [0, 1, 2, 3].map(c => ({ text: r[c], options: { ...base, bold: c === 0 || c === 2 || r[4].includes(c),
    color: r[4].includes(c) ? BAD : THEME.colors.dk1, align: c === 0 ? "left" : "center", fill: { color: c === 2 ? HI : (i % 2 ? THEME.colors.lt2 : THEME.colors.lt1) } } }))));
  s.addTable(data, { x: 0.5, y: 1.25, w: 5.5, colW: [2.5, 0.9, 1.1, 1.0], rowH: [0.34].concat(rows.map(() => 0.29)), border: { type: "solid", color: "FFFFFF", pt: 1 }, objectName: "rh table" });
  howToRead(s, "real photos with age and gender tags; detected people only (the columns cover different people). Age is Anny's 0–1 value. Red = chance. One run each; eval jobs 3611285 / 3612131 / 3613616.", 3.68, 0.62, 5.5);
  stat(s, 6.3, 1.25, 3.2, "2% → 10%", "children per training batch with a balanced sampler: unproven after 2 epochs", C.accent5, { bfs: 26, bh: 0.45, lh: 0.45, lfs: 10.5 });
  stat(s, 6.3, 2.2, 3.2, "0.10", "predicted age of training babies; true 0.015 (child-like, but too old)", C.text2, { bfs: 26, bh: 0.45, lh: 0.45, lfs: 10.5 });
  stat(s, 6.3, 3.15, 3.2, "0.61 → 0.72", "children AUC, v5 → final model: measured improvement, not solved", C.accent1, { bfs: 26, bh: 0.45, lh: 0.45, lfs: 10.5 });
  s.addShape(pres.ShapeType.roundRect, { x: 0.5, y: 4.4, w: 9.0, h: 0.72, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.08, objectName: "child box" });
  s.addText(bulletRuns([
      ["v5: ", "children read as short adults (same age, lower height); gender at chance: the shape head gives the training median."],
      ["Final: ", "women recognised (0.86), men mid-range (0.55); babies 0.44 vs adults 0.54, likely pulled down by the 3DPW age label 0.55."],
      ["Not solved: ", "kids near adult age; our demo child still gets an adult body. Limitation, or weak age labels from Relative Human (1–2 weeks)?"]], { space: 2 }),
    { x: 0.7, y: 4.43, w: 8.6, h: 0.66, fontSize: 10, color: C.text1, margin: 0, isTextBox: true, valign: "middle", objectName: "child text" });
}
N(s, "Children and gender on real photos", "Children and gender are measured on Relative Human, real photos with an age group and a gender per person: 635 images, 2,130 people. The table has three models: v5, the final model, and the masked arm from the next slide. v5 reads children as short adults: kids, babies and adults all get an age of about 0.6, only the height drops, and gender is at chance, 46 percent, because the shape head gives the training median for everyone. The final model is better on every line: it finds 65 percent of the people instead of 47, gets gender right for 64 percent, women clearly and men still in the middle, and babies now come out at 0.44 against 0.54 for adults, so the children AUC rises from 0.61 to 0.72 and half of the babies fall below the child line. Part of that age shift is likely the 3DPW age label of 0.55, which pulls every real person down. I call this a measured improvement, not a solution: kids are still near adult age, and the one child in our demo photos still gets an adult body. The tiles on the right are the synthetic side: a balanced sampler was inconclusive, and training babies come out too old.");

// 19 masked shape labels: a trade-off. Mask: loss.py THREEDPW_CONSTANT_PHENOTYPES (age, muscle, weight, cupsize, firmness, african,
// asian, caucasian; converter constants age 0.55, the rest 0.5), --mask_3dpw_shape 1 zeroes their shape L1 for 3DPW people.
// Masked arm job 3612325 (handover 0a): holdout PVE 71.7 / PA-PVE 50.7 / recall 86.2 (unmasked 73.7 / 50.3 / 87.0); 3DPW test
// 91.1 / 59.0 (89.0 / 57.6). Demo 3613617 [shape] std over 12 images: muscle 0.04 -> 0.06, weight 0.04 -> 0.11, cup size 0.05 -> 0.11,
// firmness 0.04 -> 0.11, age 0.03 -> 0.08. Relative Human 3613616: slide 18. Partial mask (keep age): job 3614042, queued.
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Findings and fixes" });
s.addText("Masked shape labels: a trade-off", { placeholder: "title" });
{
  card(s, 0.5, 1.25, 4.4, 2.95, "What the mask does", bulletRuns([
      ["The problem: ", "3DPW labels carry 8 constant shape values for everyone (age 0.55; muscle, weight, cup size, firmness and three ethnicity values 0.5): the fitter never estimates them."],
      ["Trained at full weight, ", "the final model learned them: on real photos muscle, weight, cup size and firmness all sit near 0.5 (spread ≈ 0.04)."],
      ["The mask: ", "do not learn these 8 values from 3DPW people; keep gender, height and proportions. AnnyOne people unchanged. Everything else as the final model (training job 3612325)."]], { space: 4 }), THEME.colors.accent1, { bs: 11 });
  card(s, 5.1, 1.25, 4.4, 2.95, "What happened (one run each)", bulletRuns([
      ["Variety comes back: ", "spread of muscle 0.04 → 0.06, weight 0.04 → 0.11, cup size 0.05 → 0.11, firmness 0.04 → 0.11 over 12 demo photos (demo job 3613617). Requirement 1a-shape wants this."],
      ["Gender and children fall back: ", "Relative Human gender 64 → 49% (chance), recall 65 → 57%, babies 0.44 → 0.52 (adults 0.64); AUC 0.72 → 0.71 (eval job 3613616, slide 18)."],
      ["Error unchanged within noise: ", "holdout PVE 71.7 (final 73.7), 3DPW test 91.1 (89.0): the 3DPW labels keep the neutral body, so a small loss there is expected."]], { space: 4 }), THEME.colors.accent2, { bs: 11 });
  s.addShape(pres.ShapeType.roundRect, { x: 0.5, y: 4.35, w: 9.0, h: 0.77, fill: { color: "D3ECEE" }, line: { color: "D3ECEE" }, rectRadius: 0.08, objectName: "mask next box" });
  s.addText([{ text: "Reading and next step. ", options: { bold: true, color: THEME.colors.dk1 } },
             { text: "The age label 0.55 likely does the useful work (pulls real-photo ages down, helps gender); the seven 0.5 constants cause the collapse. Decided 18 Oct: the unmasked model stays the final model and the student's teacher. Next: mask the seven constants, keep age supervised (job 3614042, queued; result after this meeting)." }],
    { x: 0.7, y: 4.4, w: 8.6, h: 0.68, fontSize: 11, color: C.text1, margin: 0, isTextBox: true, valign: "middle", objectName: "mask next text" });
}
N(s, "Masked shape labels: a trade-off", "This slide is about requirement 1a-shape, that body shape should visibly change per person. The 3DPW labels carry eight constant shape values for everyone, because the fitter never estimates them: age 0.55 and seven values at 0.5. The final model learned those constants, so on real photos muscle, weight, cup size and firmness all sit near 0.5 with almost no spread. The mask says: do not learn these eight values from 3DPW people. One run with the mask, otherwise identical, shows a trade-off. On the left side of the trade, variety comes back: the spread of weight and cup size nearly triples on the demo photos. On the right side, gender accuracy falls back to chance, recall drops, and babies move back toward adult age. The error numbers do not change beyond noise. My reading: the age label does useful work, and the seven 0.5 constants cause the collapse. So the unmasked model stays the final model, and the next run masks the seven constants but keeps age. It is queued; the result comes after this meeting.");

// 20-21 final model (reused)
// 16 final model (exp7, trained 18 Oct). Fix arm: job 3611336, run v5_pc_dist5_bonefix_v23_e10 (depth fix person_center
// x5 and v5's bone fix); control arm: job 3611339, run v5_root_dist1_nofix_v23_e10 (root targets x1, no bone fix). Both start
// from v5 00099: 10 epochs x 1000 iterations x batch 4, lr 1e-5, 30% 3DPW-train on the v23 labels, mask_3dpw_hands 1,
// shape mask off. Holdout rows: the compare table of each log (V1, 100 images, matched people; v5 = its "v5 00099"
// column). 3DPW-test row: section 4 of each log (run_eval_3dpw_test.sh, checkpoint 00009, v23 labels, 8,190 images);
// v5's cell: the same script on v23, job 3612073 (eval-test3dpw-3612073_0.out:117-124, 18 Oct; was "pending").
// The line under the table adds v9 e62 (job 3612074, :117-124) and the exp6 neck pilot (job 3612075, :118-125);
// "finds 99% / 94%" = the recall of those same lines. NAVER 126.9: multiHMR_672_L_anny, --model_class multi_hmr_original,
// job 3614192 (logs/eval-test3dpw-3614192_0.out, 18 Oct evening; 518/518 tensors loaded). Red = flipped (as on slide 15);
// orange = the cost (recall); teal = the final model (as on slides 15 and 14).
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Final model" });
s.addText("Final model: all fixes together, with a control", { placeholder: "title" });
{
  const BAD = THEME.colors.accent4, COST = THEME.colors.accent2, HI = "D3ECEE";
  // [label, v5, control, final model, columns in red, columns in orange]
  const rows = [["PVE (mm)", "70.5", "77.4", "73.7", [], []],
                ["PA-PVE (mm)", "53.3", "56.4", "50.3", [], []],
                ["recall", "90.2%", "89.0%", "87.0%", [], [3]],
                ["root error (mm)", "1208", "1402", "728", [], []],
                ["size ratio (1 = right)", "0.63", "0.59", "0.78", [], []],
                ["neck02 (deg)", "175.6°", "175.8°", "4.8°", [1, 2], []],
                ["neck girth (pred / true)", "0.46", "0.46", "1.00", [1, 2], []],
                ["fingers (deg)", "17.3°", "17.4°", "8.3°", [], []],
                ["3DPW test PVE (mm)", "121.8", "91.5", "89.0", [], []]];
  const base = { fontSize: 12, valign: "middle", margin: [0.02, 0.06, 0.02, 0.06] };
  const head = ["AnnyOne holdout, 100 images", "v5", "control", "final model"].map((t, c) => ({ text: t, options: { ...base, fontSize: 11, bold: true,
    color: THEME.colors.lt1, fill: { color: c === 3 ? THEME.colors.accent1 : THEME.colors.dk1 }, align: c === 0 ? "left" : "center" } }));
  const data = [head].concat(rows.map((r, i) => [0, 1, 2, 3].map(c => {
    const pending = r[c] === "pending";
    return { text: r[c], options: { ...base, bold: c === 0 || c === 3 || r[4].includes(c), italic: pending, align: c === 0 ? "left" : "center",
      color: r[4].includes(c) ? BAD : (r[5].includes(c) ? COST : (pending ? THEME.colors.accent5 : THEME.colors.dk1)),
      fill: { color: c === 3 ? HI : (i % 2 ? THEME.colors.lt2 : THEME.colors.lt1) } } };
  })));
  s.addTable(data, { x: 0.5, y: 1.25, w: 5.4, colW: [2.45, 0.9, 0.95, 1.1], rowH: [0.34].concat(rows.map(() => 0.26)), border: { type: "solid", color: "FFFFFF", pt: 1 }, objectName: "final table" });
  s.addText(bulletRuns([
      ["Fixes hold together: ", "neck02 4.8°, neck girth back to 1.00; root error −48%; finger error halved; PA-PVE 50.3 mm, the best of every model so far."],
      ["The cost: ", "recall −2 points vs the control, −3 vs v5. PVE −3.7 mm vs the control: inside the noise. Depth slope still 0.42 (1 = perfect)."]], { space: 10 }),
    { x: 6.15, y: 1.25, w: 3.35, h: 2.95, fontSize: 14, color: C.text1, margin: 0, isTextBox: true, valign: "top", objectName: "final text" });
}
// 18 Oct: all five models on the v23 3DPW test (two lines at 11 pt)
s.addText([{ text: "3DPW test, corrected labels, 8,190 images: ", options: { bold: true, color: THEME.colors.dk1 } },
           { text: "NAVER 126.9 · v5 121.8 · v9 (old best) 95.5 · neck pilot 120.2 · control 91.5 · final 89.0 mm; final model finds 99% of people, v9 94%." }],
  { x: 0.5, y: 4.03, w: 9.0, h: 0.4, fontSize: 11, color: C.text1, margin: 0, isTextBox: true, valign: "top", objectName: "final 3dpw line" });
s.addText("10 epochs, 40,000 images, 70% AnnyOne + 30% 3DPW-train (corrected labels); same recipe for both arms; one run each, no seed (noise ~7.6 mm PVE, ~2 recall points).",
  { x: 0.5, y: 4.46, w: 9.0, h: 0.4, fontSize: 11, color: C.text1, margin: 0, isTextBox: true, valign: "top", objectName: "final recipe" });
s.addText([{ text: "Source: ", options: { bold: true, color: THEME.colors.dk1 } }, { text: "exp7 jobs 3611336 (final model), 3611339 (control); v5, v9, neck pilot: 3612073–75; NAVER: 3614192. Holdout: helper bones masked." }],
  { x: 0.5, y: 4.88, w: 9.0, h: 0.24, fontSize: 11, color: C.text2, margin: 0, isTextBox: true, valign: "top", objectName: "final source" });
N(s, "Final model: all fixes together, with a control", "This is the final model: v5 fine-tuned for 10 epochs with all fixes at once, the depth fix, the bone fix and the corrected 3DPW labels, next to a control with the same recipe but without the depth and bone fixes. All three fixes hold together after 10 epochs with real photos in the mix: the neck bone goes from 176 to 5 degrees and the neck thickness from 0.46 to 1.00, the finger error halves from 17.4 to 8.3 degrees, and the root error drops by 48 percent, from 1.40 to 0.73 metres. PA-PVE is 50.3 millimetres, 6.1 better than the control and the best of every model so far; that gap is beyond the pilot noise. PVE is 3.7 millimetres better than the control, which is inside the noise. Recall is 2 points lower than the control and 3.2 lower than v5; that is a real cost, and I state it. On the 3DPW test split with the corrected labels, the final model also beats its control: PVE 89.0 against 91.5, PA-PVE 57.6 against 59.7. And it is now the best of our models on real photos: on the same labels v9, the previous best, gets 95.5 and v5 121.8, so the final model is 6.5 millimetres better than v9, and it finds more of the 3DPW people, 99 percent against 94. NAVER's own Anny model, run with our code on the same labels, gets 126.9 on the 3DPW test, about v5's level and 37.9 millimetres behind the final model. Most of the gain over v5 comes from training on 3DPW at all: the control, without the two fixes, already gets 91.5. The fixes add 2.5 millimetres PVE and 2.1 PA-PVE on top, plus the neck, the fingers and the depth, which a pelvis-aligned PVE barely sees. Both arms sit above v5's 70.5 on the holdout, the restart cost we saw in every fine-tune; the final model recovers more of it than the control. What stays open: the depth slope is 0.42, as good as the best pilots, 0.43 to 0.44, but far from 1, so predicted depth is still compressed; the fix moved the offset and the body size, not the slope.\n\nIf asked:\n- 3DPW test, all five models on the corrected labels with the same script and the same 8,190 images (jobs 3612073 to 3612075 and the two exp7 jobs): PVE v5 121.8, v9 epoch 62 95.5, the 2-epoch neck pilot 120.2, control 91.5, final model 89.0; PA-PVE 72.0, 62.9, 68.7, 59.7, 57.6; MPJPE 126.9, 94.8, 124.8, 90.9, 88.3. NAVER's Anny model, job 3614192, same script and images: PVE 126.9, PA-PVE 68.8, MPJPE 129.4, PA-MPJPE 66.4, precision 75 percent, recall 97. v9 trained on the old labels with the reversed gender; like every model it is tested on the corrected ones. The old numbers, v9 102.5 and v5 119.1, are retired: only numbers on the corrected labels are compared now.\n- Detection on 3DPW: the final model finds 99 percent of the labelled people, v9 94, v5 and the neck pilot 100. Precision is 64 percent for the final model, 73 for v9 and 54 for v5; 3DPW labels only the one or two actors, not the people walking past, so a low precision partly counts real bystanders, and I do not read much into it.\n- Head direction: 13.1 degrees, against 12.0 for v5 and 13.0 for the control: within noise. The 14.4 of the 2-epoch pilot did not repeat. The head bone is 8.9 degrees off, against 177.7 in the control, and the neck-skin error is 11.0 millimetres against 27.7.\n- During training, the 3DPW validation PVE went from 79.9 to 74.5 millimetres over the 10 epochs, and from 80.6 to 75.9 for the control. That is the training script's own evaluation, without the helper mask, on 300 images.\n- NaN steps: both arms skipped a few gradient steps with NaN or infinite values, 16 and 10 of 10,000. At that rate it is harmless, but I note it.\n- Training time: the final model trained in 4,292 seconds, about 72 minutes, on an A100, at 10.9 images per second. The control took 6,753 seconds on an A40, at 6.7 images per second. They ran on different GPUs, so the times do not compare the recipes.\n- Children and gender: on Relative Human the final model separates children from adults better than v5, AUC 0.72 against 0.61, and gets gender right for 64 percent of the people against 46.5 (slide 18). Part of the age shift is likely the 3DPW age label of 0.55, and training has no child signal of its own, so I do not call children solved. The shape mask was off, so the corrected 3DPW gender labels train the shape head; on real photos most men now look male, judged by eye (next slide).\n- Checkpoint: results/experiments/exp7_final_model/v5_pc_dist5_bonefix_v23_e10/train/checkpoints/00009.pt. The bone fix is saved in its settings and applied on load.");

// 17 final model on real photos, by eye. Demo job 3612078 (18 Oct; slurm/demo/run_demo_compare.sh with
// DEMO_MODELS="anny_s2_shape_v5|00099 exp7_v5_final|00009"; the final checkpoint copied to
// models/multiHMR/exp7_v5_final_00009.pt; its log prints "[load_model] bone_fix: checkpoint args (bone_fix_spec)" and
// "[bone_fix] ON, 4 bone(s)" for the second model). 4 photos (example_data) + 8 3DPW-test frames, never trained on.
// Figure docs/figures/final_model_real_photos.jpg (1008 x 1000 px, made by the student from the montages): rows =
// tennis photo, downtown_walking_00, downtown_runForBus_00; left v5, right final model; the column labels are baked in,
// so no header bars here (slide 16 crops its labels off). Placed like the other figures: left edge 0.5, top 1.2, aspect
// kept; 3.9 in tall is the most that fits between the title and the footer (5.2).
// [shape] lines over all 12 images (148 for v5, 127 for the final model; one line per person per crop): gender v5
// 0.32-0.76 (median 0.59), final 0.05-0.99 (median 0.90); age v5 0.50-0.77 (0.61), final 0.42-0.65 (0.56). Size ratio:
// holdout V1 (slide 20). Relative Human gender accuracy of v5: 0.465 (eval_rh_3611285_0.out:30); of the final model
// 0.642, children AUC 0.719, recall 0.647 (job 3612131, eval_rh_3612131_0.out:31-33; 18 Oct afternoon; the last
// bullet said "Gender on Relative Human: pending (v5: 46.5%, chance)." until then). The pictures stay "by eye", so
// the title keeps "(by eye)".
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Final model" });
s.addText("Final model on real photos: v5 vs final (by eye)", { placeholder: "title" });
{
  const PX = { w: 1008, h: 1000 }, fh = 3.9, fw = fh * PX.w / PX.h;     // 3.93 x 3.9 in
  s.addImage({ path: FIG_DIR + "final_model_real_photos.jpg", x: 0.5, y: 1.2, w: fw, h: fh,
    altText: "Real photos, v5 (left) and the final model (right): a tennis photo and two 3DPW test frames; men get male bodies in the final model",
    objectName: "figure final_model_real_photos.jpg" });
  const tx = 0.5 + fw + 0.32, tw = 9.5 - tx;                            // text column right of the figure
  s.addText(bulletRuns([
      ["Men look male: ", "tennis (3 of 4 men), walking, office, guitar: flat chest, broader shoulders. v5 gave them breasts."],
      ["Women stay female: ", "stage photo, weekly market."],
      ["Bodies a little larger: ", "size ratio 0.63 → 0.78 on the synthetic holdout (slide 20)."],
      ["Unchanged: ", "the boy is still an adult body; placement offsets; ghost bodies on half-visible people; hard poses."],
      ["Relative Human: ", "gender 46 → 64%, children AUC 0.61 → 0.72, recall 47 → 65% (job 3612131)."]], { space: 8 }),
    { x: tx, y: 1.25, w: tw, h: 2.95, fontSize: 14, color: C.text1, margin: 0, isTextBox: true, valign: "top", objectName: "real photos text" });
  s.addText([{ text: "Source: ", options: { bold: true, color: THEME.colors.dk1 } },
             { text: "demo job 3612078, 4 photos + 8 3DPW-test frames, never trained on. Shown: the tennis photo, downtown_walking_00, downtown_runForBus_00. Pictures judged by eye; the Relative Human line is measured (val split)." }],
    { x: tx, y: 4.35, w: tw, h: 0.75, fontSize: 11, color: C.text2, margin: 0, isTextBox: true, valign: "bottom", objectName: "real photos source" });   // bottom at 5.1, level with the figure
}
N(s, "Final model on real photos: v5 vs final (by eye)", "This is the final model on real photos, next to v5, judged by eye. These are three of the twelve demo images: a photo from our example set and two 3DPW test frames that no model trained on; v5 on the left, the final model on the right. The clearest change is the body shape of the men. v5 gave them breasts and wide hips; the final model gives most of them a flat chest and broader shoulders: three of the four men on the tennis photo, both men in the walking frame, the man on the phone, the guitar player and the seated man in the WhatsApp photo. Women stay female, and the bodies look a little larger, which fits the size ratio going from 0.63 to 0.78 on the synthetic holdout. What did not change: the boy in the garden photo still gets an adult body, some bodies still sit off their person, half-visible people still get ghost bodies, and hard poses like the guitar player or the seated man are poor in both models. The pictures are judged by eye, but the gender change is now also measured: on Relative Human, gender accuracy goes from 46.5 percent for v5, which is chance, to 64 percent for the final model; women are recognised, men are still in the middle. The same run separates children from adults better, an AUC of 0.72 against 0.61, and finds 65 percent of the people instead of 47 (job 3612131, slide 18).\n\nIf asked:\n- Source: demo job 3612078, the same demo script as slide 16, v5 epoch 99 next to the final model, on 4 example photos and 8 3DPW test frames, 12 images in all. The final model carries the bone fix in its saved settings, and the log shows it applied on load.\n- Why gender changed: the shape mask was off, so the corrected 3DPW gender labels train the shape head; v5 never saw 3DPW. That is my reading; I have not tested it.\n- The log's predicted gender, over all 12 images: v5 0.32 to 0.76, median 0.59, close to the training median; the final model 0.05 to 0.99, median 0.90. So the shape head now gives clear values instead of the middle. The demo log alone does not settle the direction; Relative Human does, on average (next point): zero is male in Anny, and in the frame with the man on the phone every value is below 0.5, but on the tennis photo, where the men look male, most values are above 0.7, and the log does not say which value belongs to which body. Other shape values moved too and likely add to the male look: muscle, median 0.28 to 0.48, and proportions, 0.81 to 0.57. Age: v5 0.50 to 0.77, the final model 0.42 to 0.65.\n- Relative Human, job 3612131: the final model's checkpoint, the same evaluation as slide 18. Gender accuracy is 0.642 over 1,308 people of known gender; the mean predicted gender is 0.86 for women and 0.55 for men. So on average the direction is right, high values are female as in Anny, and most of the errors are likely men in the middle. The pictures themselves stay a by-eye judgement.\n- Ghost bodies: in the walking frame the final model draws a green body on a person cut off at the image edge; in the stage photo both models draw a yellow one. Placement offsets: the bar and fencing frames, in both models.");

// 22-23 pilots and requirements (reused)
// 14 experiments: results
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Requirements and plan" });
s.addText("Pilots: four results and the noise floor", { placeholder: "title" });
table(s, [["experiment", "what it changes", "success rule", "result"],
          ["1  depth anchor", "targets at the head joint (+ depth loss ×5 or ×20)", "depth slope toward 1; PVE ≤\u00A071.5\u00A0mm, recall\u00A0≥\u00A089.2%", "Rule not met: PVE and recall miss in every run, controls too (×5, 10 epochs: 74.8 mm, 87.8%); slope 0.30–0.44, far from 1. Outside the rule, clearly better: root error 1.21 → 0.54 m, size 0.63 → 0.84"],
          ["2  children", "age-balanced sampler", "child error on the manifest, adults unchanged", "Inconclusive after 2 epochs: 10 training children 48.5 vs 53.1 mm (both arms: age loss ×3, height ×2). v5 on 127 real children (Relative Human): adult age, AUC 0.61 (slide 18)"],
          ["3  neck", "loss on the global head orientation", "global head error below 12°", "11.3° passes, but the controls give 11.7–13.6° (inside the ~3° noise). Safety limit (PVE ≤ 71.5 mm, recall ≥ 89.2%) just missed: 71.8 mm, 89.0%. All five control runs miss it too."],
          ["4  fingers", "region-normalised rotation loss", "finger error vs rest pose (about 15°)", "No effect: 17.5° vs 17.4° for the control"]],
      0.5, 1.2, 9.0, [1.2, 1.75, 1.75, 4.3], { fs: 11, rowH: [0.3, 0.62, 0.45, 0.45, 0.45], alignBody: "left", name: "exp table" });
s.addShape(pres.ShapeType.roundRect, { x: 0.5, y: 4.17, w: 9.0, h: 0.95, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.08, objectName: "noise box" });
s.addText([{ text: "Pilot noise. ", options: { bold: true } }, { text: "No fixed random seed: two identical runs differed by 7.6 mm PVE and 1.9° head error, so one pilot only detects about 10 mm PVE, 3° head error or 2 recall points. Restarting the fine-tune alone costs about 2.4 mm (controls average 72.9 vs 70.5 mm). Root error has no measured noise floor, but every fix run (0.53–0.73 m) is below every control (1.11–1.44 m). All runs: same holdout and manifest, helper bones masked. All fixes together, against a control: slide 20." }],
  { x: 0.7, y: 4.2, w: 8.6, h: 0.89, fontSize: 12, color: C.text1, margin: 0, isTextBox: true, valign: "middle", objectName: "noise text" });
N(s, "Pilots: four results and the noise floor", "All four experiments are done, each next to a control with the main switch off. Depth: by its own success rule the experiment failed, because every run, the controls too, misses the limit of 71.5 millimetres PVE and 89.2 percent recall. The 10-epoch run with the stronger loss has 74.8 millimetres and 87.8 percent. The slope is 0.30 with the switch alone and 0.38 to 0.44 with the stronger depth loss, against 0.33 for v5: far from 1. What clearly improved is outside the rule: the root error went from 1.21 to 0.54 metres and the drawn size from 0.63 to 0.84. I say that openly instead of changing the rule afterwards. Neck and fingers: no measurable effect. Children: inconclusive after two epochs, 48.5 against 53.1 millimetres on ten training children; on real photos, v5 still gives children an adult age (slide 18). The neck loss gave 11.3 degrees of head error, which passes the rule of below 12, but the two controls gave 11.7 and 13.6, so that is inside the noise. With no fixed seed, one pilot only detects about 10 millimetres PVE, 3 degrees or 2 recall points. No new architecture, because the diagnostics do not justify one. The final model, with all fixes together and its own control, is on slide 20.\n\nIf asked: The stronger age loss and the stronger depth loss were not separately controlled: both children arms used the same age and height weights, and the depth runs at five and twenty times have no control at the same weight. Depth PVE: 74.8 for the 10-epoch run with the stronger loss, 72.2 for the 10-epoch control, 70.5 for v5. Two identical runs differed by 7.6 millimetres PVE and 1.9 degrees of head error. The slope part of the depth rule only asks for any move toward 1 with a smaller intercept: it passed for the three stronger-loss runs and even for the 2-epoch control, and failed with the switch alone. All runs used the same 100-image holdout and manifest, with helper bones masked. The neck experiment has the same safety limit, PVE at most 71.5 millimetres and recall at least 89.2 percent. The neck run just missed it, with 71.8 millimetres and 89.0 percent, but all five control runs miss it too, mostly because of the restart cost; its PVE is even 4.5 millimetres better than its own control, which is also inside the noise. Its neck bones stayed at 62.5 degrees. For the fingers, the rest-pose floor is 15.2 degrees in this evaluation and 15.1 on slide 17; the difference is only which people are matched. Restarting the fine-tune alone costs about 2.4 millimetres. A 2-epoch pilot trains in about 12 minutes, a 10-epoch run in about one hour.");

// 20 thesis requirements (new on the afternoon of 18 Oct): the supervisor's e-mail of 18 Oct against what exists; the
// status matrix of docs/thesis_requirements_plan.md section 2, with the final model's Relative Human result (job 3612131).
// Status words in the deck's status colours: green = met (the done dots of slides 11 and 28), orange = partly (the to-do
// dots of slide 28, the cost on slide 20), red = not met (the flips on slides 15 and 20), grey = not started.
// Evidence: 1c post-check logs/check_refit_all_v23.log (RESULT: PASS, 60 sequences) and job 3611284 (OVERALL WARN);
// 2a exp7_final_model.sh (30% 3DPW-train, v23 labels); 1a pose 3611336:191-209 and :224-229 (3DPW test 89.0); demos
// force relaxed fingers (run_demo_compare.sh --relax_hands 1); 1a shape eval_rh_3612131_0.out:26-32 and the [shape]
// lines of demo 3612078; 1b model.py = upstream Multi-HMR with the output resized for Anny, plus the 0-parameter bone
// fix; NAVER multiHMR_672_L_anny evaluated 18 Oct, job 3614192 (logs/eval-test3dpw-3614192_0.out): AnnyOne test-100 PVE 60.3
// (likely in-sample for NAVER), 3DPW test 126.9; its cost: job 3614947 (logs/model_cost_3614947.out, results/diagnostics/model_cost_naver_vs_final.json,
// A100): 380.94 M params (backbone 304.37 + head 76.57) vs our 318.38 M (head 14.01); 3 the plan's 6.1-6.2 (98 tracked files with
// /netscratch, nothing committed under results/ or models/). Route: scripts/analysis/model_cost.py, analytic, 672 px,
// 1 multiply-add = 2 FLOPs: ViT-B student ~101 M params / 603 GFLOPs vs our ViT-L 318 M / 1,927 (3.1x / 3.2x);
// exp8 launched 18 Oct as job 3614191 (30 epochs, 20 h limit): result this week once the queue runs it.
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Requirements and plan" });
s.addText("Thesis requirements: where we stand", { placeholder: "title" });
{
  const ST = { "met": THEME.colors.accent3, "partly": THEME.colors.accent2, "partly, at risk": THEME.colors.accent2,
               "not met": THEME.colors.accent4, "not started": THEME.colors.accent5 };
  // [requirement, status, evidence, next]
  const rows = [["1c  3DPW in Anny space", "met", "v23 labels: post-check PASS, 60 sequences; independent check WARN (body-model limits)", "state the limits in the thesis"],
                ["2a  3DPW-train used in training", "met", "final model: 30% of every batch is 3DPW-train", "–"],
                ["1a  pose correct, 3DPW test and wild", "partly", "neck 4.8°, fingers 8.3°; 3DPW test 89.0 mm, our best so far", "depth slope 0.42, placement, hard poses; real-photo hands unshown"],
                ["1a  shape changes per person", "partly, at risk", "gender yes (RH 64%); age partly (baby 0.44, adult 0.54); muscle, weight, cup, firmness ≈ 0.5; masked run: variety back, gender lost (slide 19)", "partial mask, age kept (job 3614042); rest-pose line-up"],
                ["1b  not the same model as Multi-HMR", "not met", "upstream Multi-HMR + Anny output + 0-param bone fix; NAVER's model: 60.3 mm AnnyOne (likely in-sample) / 126.9 mm 3DPW vs ours 73.7 / 89.0; 380.9 M vs our 318.4 M params", "ViT-B student launched 18 Oct (job 3614191, slide 24); needs your yes"],
                ["2b  temporal (motion)", "not started", "per-frame model only", "refiner on frozen per-frame tokens, 16-frame windows; weeks 5–6 (slide 25)"],
                ["3   report and code", "partly", "draft exists (74 mm → 70.5); 98 files hard-code /netscratch; no results or models committed", "clean-up list; release models with checksums (slide 25)"]];
  const base = { fontSize: 10, valign: "middle", margin: [0.02, 0.06, 0.02, 0.06] };
  const head = ["requirement", "status", "evidence (18 Oct)", "next"].map(t => ({ text: t, options: { ...base, bold: true,
    color: THEME.colors.lt1, fill: { color: THEME.colors.dk1 }, align: "left" } }));
  const data = [head].concat(rows.map((r, i) => r.map((t, c) => ({ text: t, options: { ...base, align: "left",
    bold: c <= 1, color: c === 1 ? ST[t] : THEME.colors.dk1, fill: { color: i % 2 ? THEME.colors.lt2 : THEME.colors.lt1 } } }))));
  s.addTable(data, { x: 0.5, y: 1.15, w: 9.0, colW: [1.85, 0.95, 3.7, 2.5], rowH: [0.28].concat(rows.map(() => 0.4)),
    border: { type: "solid", color: "FFFFFF", pt: 1 }, objectName: "requirements table" });
}
howToRead(s, "status colours as on slide 2 (green met, orange partly, red not met, grey not started). Each evidence cell is a measured number or a fact read in the code; \u201cnext\u201d is the step that would move the dot. The route for 1b is on the next slide.", 4.72, 0.4);
N(s, "Thesis requirements: where we stand", "This slide puts your list of requirements next to what we have today. Two are met: 3DPW is converted to Anny space with the corrected labels, which pass our own check on all 60 sequences and the independent check with a warning that comes from the differences between the two body models; and the 3DPW training set is used in training, as 30 percent of every batch of the final model. Pose is partly there: the neck and finger flips are fixed, and the final model is our best on the 3DPW test with 89.0 millimetres, but the depth slope is 0.42, some bodies sit off their person, hard poses are poor, and I have not yet shown predicted hands on real photos, because every real-photo demo so far relaxes the fingers. Shape is partly there and at risk: gender now changes per person, 64 percent correct on Relative Human, and age partly, but muscle, weight, cup size and firmness sit near 0.5, the values the 3DPW labels fix; the masked run tested that: variety comes back, but gender and children fall back (slide 19), so the next run keeps age supervised. Requirement 1b is not met, and I want to be clear about it: our model is the upstream Multi-HMR with its output layer resized for Anny and a bone correction that adds no parameters, and its backbone weights come from NAVER's Anny checkpoint. My proposed route is the smaller-model option: a ViT-B student distilled from the final model, about 101 million parameters and 603 GFLOPs against 318 million and 1,927 for our ViT-L, so 3.1 times fewer parameters and 3.2 times fewer FLOPs, with the milliseconds per image measured. The bar is the one the script computes: the student's error may be at most the teacher's divided by 0.95 on both AnnyOne and 3DPW, reported against our ViT-L and against NAVER's ViT-L. NAVER's model is now measured with our code on the same labels: 60.3 millimetres on the AnnyOne holdout, against our 73.7, but likely in-sample for NAVER, which trained on the whole AnnyOne release; and 126.9 on the 3DPW test, far worse than our 89.0. So our model already beats NAVER's on real photos, and the AnnyOne gap of 13.4 millimetres is likely an in-sample advantage. The risk: NAVER's own ViT-B is about 15 percent worse than its ViT-L on synthetic BEDLAM data, so AnnyOne may miss the bar; it is one long job, launched on 18 October, 30 epochs with a 20-hour limit, so the result comes this week once the queue runs it; the details are on the next slide. Temporal modelling has not started; my plan is a small refiner on top of the frozen per-frame model, in month two. The report has a draft, whose 74 millimetre headline must become 70.5, and the code needs a clean-up before you can check it: paths hard-coded to our cluster in 98 files, and no results or checkpoints committed.\n\nIf asked:\n- NAVER's checkpoint: multiHMR_672_L_anny has a different head, with eight layers, so it is a real baseline and not our weights under another name. Evaluated with our code on the same labels on 18 October (job 3614192): AnnyOne holdout PVE 60.3, PA-PVE 44.6, recall 89 percent; 3DPW test PVE 126.9, PA-PVE 68.8, recall 97 percent. Our final model: 73.7 and 50.3 on AnnyOne, 89.0 and 57.6 on 3DPW. NAVER's model was likely trained on all of AnnyOne, so our 100 holdout images are probably in its training set: held out for us, not for it. Its parameters are now measured too: 380.9 million, so our final model is already 16 percent smaller, which by itself is not the significantly smaller model 1b asks for.\n- FLOPs: counted analytically from the layer shapes at 672 pixels, with one multiply-add counted as two FLOPs; the ViT-L backbone alone is about 1,917 of the 1,927 GFLOPs. NAVER's model is measured too (job 3614947): 380.9 million parameters against our 318.4 million; the backbone is the same, so the difference is its eight-layer decoder head, 76.6 million parameters against our 14.0 million. Its head FLOPs and milliseconds are not measured, because the cost script only times our model class. The milliseconds: teacher 110 and student 51.4 on an A40 (job 3613926); the teacher alone 77.9 on an A100 (job 3614947).\n- Why ViT-B first: it is the smallest step with a fair chance at the bar; it reuses the distillation code that already ran four times; and NAVER's own ViT-B equals its ViT-L on 3DPW, 94.0 against 94.1. Our old ViT-S students had 1.17 to 1.54 times their teacher's error, on the old labels. A completely new architecture that beats Multi-HMR on both test sets is not realistic in four weeks.\n- Whether a smaller Multi-HMR-style student counts as not exactly Multi-HMR is your call, because NAVER ships its own ViT-B; that is my first question on slide 26.\n- Temporal design, in one sentence: a two-layer transformer over 16-frame windows of the frozen per-person tokens, with the shape pooled over each track, trained on 3DPW-train tracks and judged by acceleration error and PVE against simple smoothing baselines.\n- Timeline: part one closes around 15 November, with no buffer; temporal in weeks five and six; the last two weeks for writing, the release and a buffer.\n- 3DPW in Anny space has limits I will state: hands, neck and head stay at rest in the converted labels, and eight shape values are constants, so 3DPW tests pose plus gender, height and proportions, not full shape; and our MPJPE averages all 163 Anny joints, so it does not compare with published 3DPW numbers.");

// 24 plan for 1b: a smaller model by distillation. Cost: exp8 smoke test job 3613926 (handover 0a, 18 Oct 17:00): student ViT-B
// 101.38 M params / 603.3 GFLOPs / 51.4 ms vs teacher ViT-L 318.38 M / 1927.4 GFLOPs / 110 ms on an A40 (3.14x / 3.19x / 2.14x);
// ViT-L backbone alone 304.37 M / 1917.2 GFLOPs (model_cost.py --no-build). Teacher errors: handover 0b (holdout V1 73.7, 3DPW 89.0).
// 95% rule: exp8_distill_vitb.sh header (student error <= teacher / 0.95: targets 77.6 and 93.7). Recipe: exp8 header (ViT-B,
// head_dim 1024, 30 epochs x 1000 x batch 8, lr 3e-5, KD lambda 1.0 / T 4). Full run job 3614191 (submitted 18 Oct, 20 h limit);
// NAVER's multiHMR_672_L_anny evaluated on the same labels: job 3614192 (logs/eval-test3dpw-3614192_0.out, 18 Oct evening,
// --model_class multi_hmr_original, 518/518 tensors): AnnyOne test-100 PVE 60.3 (likely in-sample for NAVER), 3DPW test 126.9.
// 95% bars against NAVER: 60.3/0.95 = 63.5, 126.9/0.95 = 133.6. Its cost: job 3614947 (logs/model_cost_3614947.out,
// results/diagnostics/model_cost_naver_vs_final.json; A100-PCIE-40GB, fp32 + TF32, batch 1, 20 warm-up + 50 timed): NAVER 380.94 M params
// (backbone 304.37 + head 76.57), backbone 1,917.2 GFLOPs, head FLOPs and ms not counted for class Multi_HMR; our final model
// 318.38 M (head 14.01), 1,927.4 GFLOPs, 77.9 ms median full forward (backbone 60.8 ms), 1.54 GB peak. Student 101.38 M:
// 3.76x fewer params than NAVER's, 3.14x fewer than ours. The ms column stays the A40 pair of job 3613926. Student result: this week, once the queue runs the 20-h job.
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Requirements and plan" });
s.addText("Plan for 1b: a smaller model by distillation", { placeholder: "title" });
{
  const HI = "D3ECEE", PEND = THEME.colors.accent5;
  const rows = [["model", "parameters", "GFLOPs", "ms / image (A40)", "AnnyOne holdout PVE", "3DPW test PVE"],
                ["our ViT-L (final model, the teacher)", "318.4 M", "1,927", "110", "73.7 mm", "89.0 mm"],
                ["ViT-B student (exp8, launched 18 Oct)", "101.4 M", "603", "51.4", "pending (this week)", "pending (this week)"],
                ["NAVER's ViT-L Anny model (baseline)†", "380.9 M", "1,917 (backbone)", "not measured", "60.3 mm*", "126.9 mm"],
                ["student vs teacher", "3.14× fewer", "3.19× fewer", "2.14× faster", "bar: ≤ 77.6 mm", "bar: ≤ 93.7 mm"]];
  const base = { fontSize: 10.5, valign: "middle", margin: [0.02, 0.06, 0.02, 0.06] };
  const data = rows.map((r, i) => r.map((t, c) => {
    const pend = /pending|not measured/.test(t);
    return { text: t, options: { ...base, bold: i === 0 || c === 0 || i === 4, italic: pend, align: c === 0 ? "left" : "center",
      color: i === 0 ? THEME.colors.lt1 : (pend ? PEND : THEME.colors.dk1),
      fill: { color: i === 0 ? THEME.colors.dk1 : (i === 2 ? HI : (i % 2 ? THEME.colors.lt1 : THEME.colors.lt2)) } } };
  }));
  s.addTable(data, { x: 0.5, y: 1.2, w: 9.0, colW: [2.5, 1.35, 1.2, 1.15, 1.45, 1.35], rowH: [0.32, 0.3, 0.3, 0.3, 0.3], border: { type: "solid", color: "FFFFFF", pt: 1 }, objectName: "distill table" });
  howToRead(s, "params and GFLOPs measured (1 multiply-add = 2 FLOPs, 672 px, one person); ms on an A40 for the teacher/student pair (job 3613926); the teacher runs in 77.9 ms on an A100 (job 3614947). Errors: NAVER job 3614192, student job 3614191. *likely in-sample for NAVER: trained on all of AnnyOne. †same ViT-L backbone; its head is 76.6 M params vs our 14.0 M; head FLOPs and ms not measured for NAVER's class (job 3614947).", 2.78, 0.7);
  s.addText([{ text: "The 95% rule (your 1b-ii): ", options: { bold: true, color: THEME.colors.dk1 } },
             { text: "the student's PVE and MPJPE may be at most the teacher's divided by 0.95, on AnnyOne test and 3DPW test, with fewer params and FLOPs and faster inference. Bars vs our ViT-L: 77.6 / 93.7 mm; vs NAVER's: 63.5 / 133.6 mm (which counts: slide 26, Q1)." }],
    { x: 0.5, y: 3.52, w: 9.0, h: 0.48, fontSize: 11, color: C.text1, margin: 0, isTextBox: true, valign: "top", objectName: "rule line" });
  s.addText(bulletRuns([
      ["Recipe: ", "DINOv2 ViT-B backbone (NAVER's HMR-trained weights), heads copied from the teacher through a 768 → 1024 projector, the teacher's bone fix and depth targets; feature + output distillation (weight 1.0, T 4); 30 × 1,000 × batch 8 = 240,000 images, lr 3e-5; vs NAVER: 3.76× fewer params."],
      ["Timeline: ", "smoke test passed end to end 18 Oct (job 3613926); full run launched 18 Oct (job 3614191, 30 epochs, 20 h limit): result this week once the queue runs it. Misses the bar with a still-falling curve: train longer, report the ratio as measured."],
      ["Risk: ", "NAVER's own ViT-B is ~15% worse than its ViT-L on synthetic BEDLAM but equal on 3DPW, so AnnyOne may miss. Your call: does a smaller Multi-HMR-style student count as “not exactly Multi-HMR”?"]], { space: 2 }),
    { x: 0.5, y: 4.04, w: 9.0, h: 1.06, fontSize: 10, color: C.text1, margin: 0, isTextBox: true, valign: "top", objectName: "distill text" });
}
N(s, "Plan for 1b: a smaller model by distillation", "Requirement 1b is not met today, and this is my route: the smaller-model option. The student is a ViT-B version of the same model, trained to copy the final model, which is the teacher. The cost side is already measured in the smoke test: 101 million parameters against 318, 603 GFLOPs against 1,927, and 51 against 110 milliseconds per image on the same A40 GPU; that is about three times smaller and twice as fast. The error side is pending. Your rule says the student's errors may be at most the teacher's divided by 0.95 on both test sets, so at most 77.6 millimetres on AnnyOne and 93.7 on 3DPW. NAVER's own Anny model is now measured with our code on the same labels: 60.3 millimetres PVE on the AnnyOne holdout and 126.9 on the 3DPW test, against 73.7 and 89.0 for our final model. Its cost is now measured too (job 3614947, on an A100): 380.9 million parameters against our 318.4 million. The backbone is the same ViT-L, so the whole difference is the head: NAVER's eight-layer cross-attention decoder has 76.6 million parameters, ours 14.0 million; so our final model is already 16 percent smaller than NAVER's, which by itself is not the significantly smaller model you asked for, and the student is 3.76 times smaller than NAVER's model and 3.14 times smaller than ours. On that A100 our teacher runs in 77.9 milliseconds per image; NAVER's milliseconds are not measured, because the timing code supports only our model class. The AnnyOne number is likely in-sample for NAVER: it was trained on the whole AnnyOne release, so our 100 holdout images are probably in its training set, held out only for us. Against NAVER the 95 percent bar would be 63.5 millimetres on AnnyOne, which no student of a 73.7 millimetre teacher can reach, and 133.6 on 3DPW, which is trivially met; so which baseline counts is my first question on slide 26. The full run was launched yesterday with a twenty-hour limit, so the result comes this week once the queue runs it. The risk: NAVER's own ViT-B is 15 percent worse than its ViT-L on synthetic data, so AnnyOne may miss. And I need your answer on whether this counts as not exactly Multi-HMR.");

// 25 the next four weeks: docs/thesis_requirements_plan.md section 4 (week 1-4, weeks 5-6, 7-8) and 3.6 ((1) closes 15 Nov);
// student job 3614191 launched 18 Oct (30 epochs, 20 h limit: result this week, earlier than the plan file assumed); partial-mask run job 3614042 (handover 0a)
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Requirements and plan" });
s.addText("Plan: the next four weeks", { placeholder: "title" });
{
  const rows = [["week", "dates", "what happens", "for"],
                ["1", "19–25 Oct", "NAVER baseline: done 18 Oct (job 3614192, slide 24). Partial-mask run (job 3614042) + its Relative Human; decide the final ViT-L (Sun 25). Student result (job 3614191) once the queue runs it: holdout, 3DPW test, the 95% table. Code: masked-evaluation flag; the student's shape-distillation mask. Phenotype statistics on AnnyOne (r, slope, R²).", "1a shape, 1b"],
                ["2", "26 Oct – 1 Nov", "Rest-pose line-up of predicted bodies (the “visibly changes” figure). Predicted fingers on real photos (every demo so far forces rest-pose hands). Old ViT-S students on the new labels. Body-joint MPJPE. Writing: Method and Data chapters.", "1a, 3"],
                ["3", "2–8 Nov", "If the student missed the bar and its curve still fell: train it longer from its last checkpoint. The 95% table against our ViT-L and NAVER's; paired bootstrap confidence intervals. Writing: Experiments chapter (depth, neck, shape mask, Relative Human).", "1b, 3"],
                ["4", "9–15 Nov", "One final table, one protocol, every model. One more seed of the final ViT-L if a slot is free. Architecture figure and module table. Freeze the numbers; start the model release with checksums.", "1b, 3"],
                ["5–6", "16–29 Nov", "Temporal (2b): a 2-layer refiner over 16-frame windows of frozen per-frame features, shape pooled per track; trained on 3DPW-train tracks; judged by acceleration error and PVE against smoothing baselines.", "2b"],
                ["7–8", "30 Nov – 13 Dec", "Introduction, related work, conclusion, limitations, full revision; code clean-up (paths, environment, tests, README); presentation. Week 8 is buffer.", "3"]];
  table(s, rows, 0.5, 1.2, 9.0, [0.6, 1.15, 6.25, 1.0], { fs: 9, rowH: [0.28, 0.6, 0.5, 0.5, 0.5, 0.5, 0.42], alignBody: "left", name: "plan table" });
  howToRead(s, "part (1) of your list closes around 15 Nov. Dropped: ViT-S student, children sampler runs, the depth-slope investigation (slope 0.42 becomes a stated limitation).", 4.72, 0.4);
}
N(s, "Plan: the next four weeks", "Here is the plan, week by week, from the requirements document. NAVER's baseline result came in on 18 October, so it is done. Week one: run the partial-mask arm with its Relative Human check, and decide on Sunday which ViT-L is the final one; measure the shape values on AnnyOne properly. Week two: the figures that answer your shape requirement, a rest-pose line-up and real photos with predicted fingers, which no demo has shown yet; and the Method and Data chapters. The student was launched yesterday, so its result comes this week once the queue runs it; the 95 percent table is built against our ViT-L and NAVER's. Week three: if the student missed the bar, train it longer; confidence intervals; the Experiments chapter. Week four: one final table with one protocol for every model, and the start of the model release. That closes part one of your list around 15 November. Weeks five and six are the temporal refiner, weeks seven and eight writing, clean-up and the presentation. I have dropped the ViT-S student, the children sampler and the depth-slope investigation.");

// 26 claims (reused, DARK_CLOSE)
// 19 claims vs not (the question gets its own tinted box with an accent bar); 18 Oct evening: the final model (slide 20);
// 18 Oct afternoon: the Relative Human result of the final model (job 3612131, slide 18) in the claims, and the question
// box holds the four questions of docs/thesis_requirements_plan.md section 5 (was: "Chosen: fix gender and frame grid,
// re-fit all 3DPW tasks, then fine-tune v5 (done 18 Oct: slide 20). Next: write up, or also add weak age labels for
// children (1–2 weeks)?" and the v9 / v10 flip line; both stay in the notes)
s = pres.addSlide({ masterName: "DARK_CLOSE", sectionTitle: "Requirements and plan" });
s.addText([{ text: "What I claim, what I do not, and my questions", options: { fontSize: 26 } }], { placeholder: "title" });
s.addText([{ text: "Claimed, measured: ", options: { bold: true, color: THEME.colors.accent6 } }, { text: "the failures are not rig or decoding problems. A training-target bug puts people too far away in v5, v9 and v10. In the v5 line, neck02 and the head bone sit half a turn off; two flipped hand bones were half the finger error. 3DPW labels: three defects fixed by a re-fit (slide 6). The final model keeps every fix against its control: root error −48%, neck02 4.8°, fingers 8.3°, PA-PVE 50.3 vs 56.4 mm; 3DPW test 89.0 mm, our best (v9 95.5, v5 121.8; slide 20). Relative Human, v5 → final: gender 46 → 64%, children AUC 0.61 → 0.72 (slide 18).", options: { breakLine: true, paraSpaceAfter: 8 } },
           { text: "Not claimed: ", options: { bold: true, color: THEME.colors.accent6 } }, { text: "results proven over seeds (one run each) or why the bones flipped; children or gender solved (men still mid-range); correct near/far depths (slope 0.42); fixes without cost (recall −2 points vs the control); that the fixes made most of the 3DPW gain; a model that is not Multi-HMR (slide 23)." }],
  { placeholder: "body" });
s.addShape(pres.ShapeType.rect, { x: 0.6, y: 3.62, w: 8.8, h: 1.5, fill: { color: "2B3550" }, line: { color: "2B3550" }, objectName: "question box" });
s.addShape(pres.ShapeType.rect, { x: 0.6, y: 3.62, w: 0.08, h: 1.5, fill: { color: THEME.colors.accent6 }, line: { color: THEME.colors.accent6 }, objectName: "question bar" });
s.addText([{ text: "Questions for you", options: { bold: true, color: THEME.colors.accent6, breakLine: true, paraSpaceAfter: 3 } },
           { text: "1. (1b) Is the 95% bar against our ViT-L, NAVER's, or both? (NAVER: 60.3 mm on AnnyOne, likely in-sample; 126.9 on 3DPW; ours 73.7 / 89.0) Does a smaller Multi-HMR-style student count as “not exactly Multi-HMR”?", options: { breakLine: true, paraSpaceAfter: 2 } },
           { text: "2. (2b) Does a temporal refiner on frozen per-frame features count as “understands motion”?", options: { breakLine: true, paraSpaceAfter: 2 } },
           { text: "3. Children: a stated limitation, or weak age labels from Relative Human (1–2 weeks)?", options: { breakLine: true, paraSpaceAfter: 2 } },
           { text: "4. When is the submission date?" }],
  { x: 0.9, y: 3.67, w: 8.35, h: 1.4, fontSize: 12, color: THEME.colors.lt2, margin: 0, isTextBox: true, valign: "middle", objectName: "question text" });
N(s, "What I claim, what I do not, and my questions", "What I claim, measured. The failures are not rig or decoding problems. A training-target bug puts people too far away in v5, v9 and v10; the fix draws bodies much closer to their true size. In the v5 line, neck02 and the head bone sit half a turn off, and two flipped hand bones were about half the finger error; a fixed correction turns them back. The 3DPW labels had three defects; a re-fit fixed them, and the new labels pass the check with a warning (slide 6). And the final model, with all fixes together, keeps every fix against its own control: the root error is 48 percent lower, neck02 is 4.8 degrees, the fingers 8.3, PA-PVE 50.3 against 56.4 millimetres, and on the 3DPW test the PVE is 89.0 against 91.5, against 95.5 for v9, the previous best, and 121.8 for v5 (slide 20). On Relative Human, the final model is better than v5 on every line: gender accuracy 46 to 64 percent, child-vs-adult AUC 0.61 to 0.72, recall 47 to 65 percent (slide 18). What I do not claim: results proven over several seeds, since each arm ran once; why the bones flipped; that children or gender are solved, since men are still in the middle and children still get near-adult ages; correct depth for near and far people, since the slope is still 0.42; fixes without a cost, since recall is 2 points below the control and 3 below v5; that the fixes made most of the 3DPW gain, because training on 3DPW at all did most of it and the fixes add 2.5 millimetres; and a model that is not Multi-HMR: that requirement is still open (slide 23). On last week's question, v9 now or the labels first: I went ahead with the labels first, then fine-tuned v5; that is the final model. I have four questions now. First, for the smaller model: is the 95 percent bar against our ViT-L, against NAVER's, or both, and does a smaller Multi-HMR-style student count as not exactly Multi-HMR? NAVER's model gets 60.3 millimetres on AnnyOne, likely in-sample for it, and 126.9 on 3DPW, against our 73.7 and 89.0: against NAVER the bar would be 63.5 on AnnyOne, out of reach for any student of our teacher, and 133.6 on 3DPW, already met. Second, does a temporal refiner on frozen per-frame features count as understanding motion? Third, children: a stated limitation, or weak age labels from Relative Human, one to two weeks? Fourth, what is the submission date?\n\nIf asked:\n- NAVER baseline, job 3614192, its own multiHMR_672_L_anny run with our code, labels and settings: AnnyOne holdout PVE 60.3, PA-PVE 44.6, MPJPE 63.3, PA-MPJPE 45.0, recall 89 percent; 3DPW test PVE 126.9, PA-PVE 68.8, MPJPE 129.4, PA-MPJPE 66.4, recall 97 percent. It was likely trained on all of AnnyOne, so its AnnyOne number is probably in-sample and not a fair test number for it. Our final model already beats it on 3DPW, 89.0 against 126.9, but not on AnnyOne, so a new-architecture route to 1b would hinge on AnnyOne.\n- Depth fix: it does not separate near and far people well; the slope is 0.42 in the final model, against 0.28 for its control and 0.33 for v5. Its cost is now measured in the final model against its control: PVE 3.7 millimetres better, inside the noise, and recall 2 points lower, at the noise limit; against v5, recall is 3.2 points lower, and I state that as a real cost. In the pilots, every fix run was worse than its control, by 2.6 to 4.9 millimetres PVE and 1.6 to 2.4 recall points. In v5 the depth bug affects all 222 holdout people. The ViT-S students use the same training code and are probably affected too; not measured.\n- Neck: the two half-turned bones cancel, so the head points the right way. A 2-epoch geodesic loss did not undo it. The fixed correction with a 2-epoch fine-tune did, in one run: neck02 4.7 degrees, thickness 1.00, head direction 14.4 against 13.8 for its control and 12.0 before. In the 10-epoch final model it holds: neck02 4.8 degrees, thickness 1.00, head direction 13.1 against 13.0 for its control, so the 14.4 did not repeat. One run per arm without a seed is not proof; the holdout renders of the correction without training show normal neck width, and in a first real-image demo the 2-epoch model's necks look normal in the 8 of its 12 images I have seen, judged by eye only; there is no neck metric on real images. The origin of the flip is not known; the singular start value of the rotation output would start a bone turned about its sideways axis, a tilt like the one in v5's left index base, not the twist in neck02 and the head.\n- Fingers: two flipped hand bones were about half the finger number, 17.3 to 8.3 degrees, and the final model keeps 8.3 against 17.4 for its control; I have not looked at the hands, so I do not claim better-looking fingers.\n- 3DPW labels: 6 train sequences, part of a seventh, and one validation sequence had no labels, because their image files were missing on the cluster. On 16 October I re-extracted them, and a second rebuild, job 3581320, brought the labels back. The re-fit fixed gender and the missing second person on 18 October; the check on the new labels confirms both: flipping the gender now makes the fit worse, and no two-person image misses a person. In the 3DPW labels the head, neck02 and neck03 stay at rest; only neck01 turns. Age is fixed at 0.55. The remaining warnings are the limb proportions and the hip joint, where Anny and SMPL differ.\n- 3DPW test, all on the corrected labels: PVE v5 121.8, the 2-epoch neck model 120.2, v9 epoch 62 95.5, control 91.5, final model 89.0; the final model also finds more 3DPW people than v9, 99 percent against 94. The old numbers, v9 102.5 and v5 119.1, measured on the old labels, are retired.\n- Option 1, v9 now, which I did not choose: the fine-tune is about 1 to 2 hours of training; that is not measured with 3DPW batches, and the v9 script expects 8 to 10 minutes per epoch. From the v5 pilot I expected a drawn size of about 0.84 and a root error of about half a metre; the v5 final model reached 0.78 and 0.73 metres, at a cost of 2 recall points against its control. Five, not twenty, because the twenty-times 10-epoch run has no result yet. v9 is best on our old 3DPW-Anny labels, but those labels had the gender and two-person defects, so that score is not a clean measure; on the corrected labels v9 gets 95.5, behind the final model's 89.0. On the AnnyOne holdout v9 is not the best: 75.0 at epoch 62, the planned start, and 77.0 at epoch 149, against 70.5.\n- Option 2, labels first, which I chose and did: the re-fit is done. Men to 0, women to 1 by Anny's convention, one shared frame grid, all 87 conversion tasks fitted again into a new folder, about 28 GPU hours at the measured 4.4 seconds per frame, finished on 18 October; the labels are rebuilt, 16,694 images, and the check on them passes with a warning (slide 6). Rebuilding the labels alone would not have been enough, because the gender and the frame choice are fixed inside the fit. The fine-tune after it started from v5 with v5's own bone fix and the depth fix, ran 10 epochs, and had a control run without the two fixes; both trained on 18 October, after a short smoke test of the whole recipe passed; the result is on slide 20. I chose this option because 3DPW is 30 percent of each batch, so a fine-tune on the old labels would keep learning the wrong body for the men.\n- Bone flips in v9 and v10: a v9 fine-tune keeps the neck flip unless it carries the bone fix, estimated on v9 first; the v5 constants belong to v5. The flip was measured on v9 epoch 149; epoch 62, the planned start, was never measured per bone, so there it is likely, not shown. v10 has no neck flip, but the same kind of flip in the left shoulder and upper arm, which is likely most of why its shoulders and arms are worse; not shown.\n- Both options: either fine-tune should also mask the 3DPW finger labels, which are set to the rest pose, with mask_3dpw_hands; the final model does. A new flag can also leave out the eight 3DPW shape values that are constants, age 0.55 and seven at 0.5; it is off by default and off in the final model; a run with it on is done (slide 19): body variety on photos opens, gender and children fall back; a partial mask that keeps age is queued (job 3614042). With it on, the 3DPW error can change for a reason unrelated to the two fixes.");

// 27-29 appendix (reused)
// 18 still open (two columns, row-major like slide 11; item 7, the 3DPW label fix, spans both columns, red like the slide 6 bug band)
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Appendix" });
s.addText("Still open, and what I test next", { placeholder: "title" });
// 18 Oct evening: items 1-3 and 7 updated with the final model (slide 20); 18 Oct afternoon: item 6 with the final
// model's Relative Human AUC (job 3612131; was "(AUC 0.61, read as short adults); decision: limitation or weak labels.")
const open = [["1", "Bone flips: fix confirmed at 10 epochs", "Final model, real photos in training: neck02 4.8° (slide\u00A020). Real images by eye only (slide\u00A016). v10's left arm: two-sided fix, not built yet."],
              ["2", "Depth slope: 0.42 in the final model", "Still far from 1; cause unknown. Next: measure training depths properly, and the slope on training images."],
              ["3", "Depth fix cost: now measured", "Final model vs its control: PVE within noise, recall −2 points (−3 vs v5). Paired PVE not done; no ×20 result at 10 epochs."],
              ["4", "Pilot noise", "No fixed seed. Fix it, or repeat the key run, before trusting differences under about 10 mm."],
              ["5", "People very close to the camera", "Both v5 and the depth fix miss them (4 of 12 holdout renders): not the fix. Likely detection; not counted."],
              ["6", "Children and fingers", "Children: Relative Human AUC 0.61 (v5) → 0.72 (final), not solved; limitation or weak labels? Fingers: bone-fix hands, then larger crops."]];
const openY = [1.25, 2.25, 3.25];
open.forEach((o, i) => numberedItem(s, 0.5 + (i % 2) * 4.65, openY[Math.floor(i / 2)], 4.35, o[0], o[1], o[2], C.accent1, 0.6));
numberedItem(s, 0.5, 4.25, 9.0, "7", "3DPW labels: three defects fixed (slide 6)", "Missing labels fixed 16 Oct (images re-extracted). Re-fit of all 87 tasks (gender m\u00A0→\u00A00,\u00A0f\u00A0→\u00A01; one shared frame grid) and re-check done 18 Oct: WARN (was FAIL), no actor missing. In use: the final model trained on them (slide\u00A020).", C.accent3, 0.55);
N(s, "Still open, and what I test next", "Seven things are still open. First, the bone flips. The fix now holds in the 10-epoch final model, with real photos in training: neck02 4.8 degrees and fingers 8.3 (slide 20). The holdout renders of the correction alone show normal necks, and so, by eye, do the 8 real images I have seen so far from a first demo of the 2-epoch model; real images have no neck metric. v10's left arm has the same kind of flip, but a bone further down the arm undoes it, so it needs a two-sided correction, not built yet. Second, the depth slope: 0.42 in the final model, about as good as the best pilots but far from 1; cause unknown. Third, the cost of the depth fix: now measured in the final model against its control, PVE within the noise and recall 2 points lower; a comparison paired on the same people is still to do. Fourth, pilot noise: fix the seed or repeat key runs. Fifth, people very close to the camera. Sixth, children and fingers: children are now measured on real photos with age labels, the Relative Human dataset: v5 reads them as short adults, and age separates them from adults with an AUC of only 0.61; the final model does better, 0.72, but children still get near-adult ages (slide 18); the decision is a limitation or weak age labels. For fingers, a look at the hands with the bone fix. Seventh, 3DPW: the labels had three defects. The missing labels came back on 16 October, when I re-extracted the images of eight sequences; train went from 4,451 to 6,437 label images. The re-fit with the gender and frame-grid fixes finished on 18 October, and the check on the new labels passes with a warning, with no defect left (slide 6). They are in use: the final model and its control trained on them on 18 October, after a short smoke test of the whole recipe passed (slide 20).\n\nIf asked: Neck: the fine-tuned checkpoint carries the fix in its saved settings, so the demo applies it on load; the demo comparison, job 3581312, ran it next to v5 on 4 example photos and 8 3DPW test frames (slide 16); I have looked at 8 of these 12 images so far. In the 12 side-by-side holdout panels, with the correction alone and no training, the necks have normal width wherever a neck is visible (slide 16). v10: correcting three of its flipped hand bones passed without training; two more finger bones, seen on training images, are not corrected yet; shoulder01.L at 166 degrees and upperarm02.L at 180 stay flipped, because one constant per bone would break the chain that compensates them. Depth: the training depths so far come from small samples, batch means of about 2.5 to 3.2 metres. I will also measure the fix run's slope on training images; for v5 it was just as low there. Cost: I will report the root error next to PVE in the thesis. On the 3DPW test split, with the new labels, the final model beats its control, 89.0 against 91.5 millimetres PVE, and also v9, 95.5, and v5, 121.8, on the same labels; most of that gain comes from the 3DPW training itself (slide 20). The twenty-times 10-epoch run has no result yet. Close people: v5 and the depth-fix model both miss them in 4 of the 12 rendered holdout images (slide 13). Children and fingers: one 2-epoch test is not enough to close them; for children it is inconclusive, not a proof that more synthetic children cannot help, and a 10-epoch sampler run stays a low-priority option. 3DPW labels: the rebuild, into a new folder, gave the same counts, so my old guess that the builder ran before the fits finished is wrong. The builder skips every frame without an image file. So I re-extracted the image folders of the eight affected sequences on 16 October: captureSelfies_00 had 480 of 750 files; for the other seven the counts before were not recorded. The second rebuild, job 3581320, confirmed it: train rose from 4,451 to 6,437 label images and validation from 2,458 to 2,777, exactly what the fits imply; test stayed at 9,722. Those labels still had the reversed gender and the missing second person. Gender and the frame grid are different: men to 0 and, by Anny's convention, women to 1, and both people's frames on one shared grid. Both happen inside the fit, so all 87 conversion tasks were fitted again, into a new folder, because the task script skips every person that already has a file; the task script now refuses to mix old and new fits. That was an 87-task array job: about 28 GPU hours at the measured 4.4 seconds per frame (the 17 October test, on an H100), done on 18 October. The rebuilt labels have 16,694 images with 22,749 people, and the check on them confirms both fixes: flipping the gender now makes the fit worse, and no two-person image misses a person.");

// 15 housekeeping: each card body says "Done:" / "To do:"; dot green = all done, amber = something still to do
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Appendix" });
s.addText("Corrections to the draft and the repo", { placeholder: "title" });
const status = (...parts) => parts.flatMap(([lead, text], i) => [
  { text: lead, options: { bold: true, color: THEME.colors.dk1 } }, { text, options: { breakLine: i < parts.length - 1 } }]);
card(s, 0.5, 1.35, 4.4, 1.75, "Headline number", status(["Done: ", "the README gives AnnyOne holdout PVE 70.5 mm, helper bones masked, as in the demos."], ["To do: ", "replace the thesis draft's 74 mm, the unmasked evaluation (+3.8 mm on the same people)."]), THEME.colors.accent2, { bs: 12 });
card(s, 5.1, 1.35, 4.4, 1.75, "Demos", status(["Done: ", "all demo scripts zero the fingers and the README says so: finger renders are the rest pose, not predictions."]), THEME.colors.accent3, { bs: 12 });
card(s, 0.5, 3.28, 4.4, 1.82, "Wording", status(["Done: ", "neck, fingers, children no longer called rig or decoding problems; the README's Known-defects section documents the depth bug."], ["To do: ", "README and findings still carry last week's neck story; I will fix them next week."]), THEME.colors.accent2, { bs: 12, bh: 1.0 });
card(s, 5.1, 3.28, 4.4, 1.82, "Repository", status(["Done: ", "reorganised, one shared cluster setup, diagnostics and experiments documented so the work can be rerun."]), THEME.colors.accent3, { bs: 12 });
N(s, "Corrections to the draft and the repo", "Some corrections are already in. The headline AnnyOne number is 70.5 millimetres, measured with the helper bones masked, as in the demos. That is the evaluation setting V1, not the demo's own number: the demo setting gives 68.5 at a lower recall. The 74 millimetres in the thesis draft came from the unmasked evaluation; on the same people that adds 3.8 millimetres. The README already uses 70.5; the thesis draft I still have to change. The demos now say that the fingers are the rest pose. The text no longer blames the rig. The Known-defects section in the README documents the depth bug. But the README and the findings document still carry last week's neck story, that the long neck comes from weak supervision and from the depth bug, and some old depth wording, for example that the overlays look wrong and that no improvement is claimed yet. I will correct them next week. The repository is reorganised with everything scripted, so any of these numbers can be produced again.");

// 21 appendix: second before / after example (same geometry as slide 13)
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Appendix" });
s.addText("Backup: before and after, holdout image 579860", { placeholder: "title" });
figBottom = figureSlide(s, "depth_before_after_579860.jpg", 1.0, 1.2, 8.0, [
  "Synthetic holdout image 579860. Predictions use the model's own depth.",
  "v5: the real head sticks out above a shrunken mesh",
  "Fix + depth ×5, 10 epochs: head level with the truth; mesh cut off at the image edge"], 0.4);
footnote(s, "Not an exact fit: ", "here the fixed mesh is a little larger and wider than the ground truth. The two people in front have no ground-truth mesh, so they are not counted as misses.", figBottom + 0.5, 0.4);
N(s, "Backup: before and after, holdout image 579860", "A second holdout example, in case there is a question. Before the fix the boy's real head sticks out above a shrunken mesh that floats. After the fix the top of the head is level with the truth; the after mesh is cut off at the bottom of the image, so I cannot measure its full height. Here the fixed mesh is a little larger and wider than the ground truth, so I do not call it an exact fit. The two large people in front have no ground-truth mesh, so they are not counted as misses.");

// 30 jobs and checkpoints (paths relative to /netscratch/najib/multi-hmr; handover 1 "Exact checkpoint paths", 0a, 0b and section 8)
// 3614947 (model cost, A100; json results/diagnostics/model_cost_naver_vs_final.json) is folded into the 3614192 row: the 13 body
// rows already render down to ~4.97 in at font 8.5 (~4.85 at 8), so a 14th would cross the how-to-read line at 5.0. Font 8 (was 8.5) so that row's path stays on one line.
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Appendix" });
s.addText("Jobs and checkpoints", { placeholder: "title" });
{
  const rows = [["job", "what", "path under /netscratch/najib/multi-hmr (EXP7 = results/experiments/exp7_final_model)"],
                ["–", "v5 epoch 99: start point and reference", "logs/anny_model/anny_s2_shape_v5/checkpoints/00099.pt"],
                ["3611336", "final model (exp7 fix arm, 10 epochs)", "EXP7/v5_pc_dist5_bonefix_v23_e10/train/checkpoints/00009.pt (3DPW test in its log)"],
                ["3611339", "control: no depth or bone fix", "EXP7/v5_root_dist1_nofix_v23_e10/ (3DPW test in its log)"],
                ["3612325", "masked-shape arm (training run)", "EXP7/v5_pc_dist5_bonefix_v23_e10_shpmask/"],
                ["3614042", "partial mask, age kept (queued)", "EXP7/v5_pc_dist5_bonefix_v23_e10_shpmask_noage/"],
                ["3614191", "ViT-B student, full run (smoke test 3613926)", "logs/exp8_distill_vitb_3614191.out"],
                ["3614192 / 3614947", "NAVER: 60.3 / 126.9 mm; 380.9 M vs our 318.4 M", "logs/{eval-test3dpw-3614192_0,model_cost_3614947}.out; models/multiHMR/multiHMR_672_L_anny.pt"],
                ["3611285", "Relative Human: v5 (_0), neck pilot (_1)", "logs/eval_rh_3611285_0.out, logs/eval_rh_3611285_1.out"],
                ["3612131 / 3613616", "Relative Human: final model / masked arm", "logs/eval_rh_3612131_0.out, logs/eval_rh_3613616_0.out"],
                ["3612078 / 3613617", "demos, 12 images: v5 vs final; vs masked arm", "logs/demo-compare-3612078.out, logs/demo-compare-3613617.out"],
                ["3612073–75", "3DPW test, v23 labels: v5, v9 e62, neck pilot", "logs/eval-test3dpw-361207{3,4,5}_0.out"],
                ["3611284", "v23 label check (WARN; old labels 3575369 FAIL)", "results/diagnostics/check_3dpw_conversion_v23/; labels: threedpw_anny_v23/labels"],
                ["3579014", "exp6 bone fix on v5 (estimate + 2-epoch fine-tune)", "results/experiments/exp6_bone_fix/anny_s2_shape_v5_00099_n200_s0/ (bone_fix.json)"]];
  table(s, rows, 0.5, 1.2, 9.0, [1.15, 2.85, 5.0], { fs: 8, rowH: [0.26].concat(rows.slice(1).map(() => 0.25)), alignBody: "left", name: "jobs table" });
  howToRead(s, "every number in the deck comes from these logs or run folders; checkpoints are not in git (release: week 4).", 5.0, 0.2);
}
N(s, "Jobs and checkpoints", "This appendix slide is the provenance table: every number in the deck comes from one of these jobs. v5 epoch 99 is the starting point. The final model is job 3611336 and its control 3611339; the masked-shape arm is 3612325 and the partial mask, queued, 3614042. The ViT-B student is job 3614191 after the smoke test 3613926, and NAVER's Anny model was evaluated in job 3614192: 60.3 millimetres on the AnnyOne holdout, likely in-sample for it, and 126.9 on the 3DPW test. Its cost was measured in job 3614947, on an A100: 380.9 million parameters, of which the head is 76.6 million, against 318.4 million for our final model, which runs in 77.9 milliseconds there; the numbers are in results/diagnostics/model_cost_naver_vs_final.json. The Relative Human evaluations are 3611285 for v5, 3612131 for the final model and 3613616 for the masked arm; the demos 3612078 and 3613617; the 3DPW test evaluations 3612073 to 75 for the older models, while the final model and the control have theirs inside their own logs. The label check is 3611284, and the bone fix comes from exp6, job 3579014. Checkpoints are not in git yet; the release with checksums is planned for week four.");

// pptxgenjs writes one <a:pPr> per run, so a paragraph with a bold lead run gets a second pPr after the
// first run (schema-invalid; LibreOffice then drops the bullet). Keep only the first pPr of each paragraph.
async function keepFirstParaProps(file) {
  const fs = require("fs");
  const JSZip = require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  let removed = 0;
  for (const name of Object.keys(zip.files).filter(n => /^ppt\/slides\/slide\d+\.xml$/.test(n))) {
    const xml = await zip.file(name).async("string");
    const out = xml.replace(/<a:p>([\s\S]*?)<\/a:p>/g, (m, inner) => {
      let seen = false;
      const fixed = inner.replace(/<a:pPr\b[^>]*\/>|<a:pPr\b[^>]*>[\s\S]*?<\/a:pPr>/g, pp => { if (!seen) { seen = true; return pp; } removed++; return ""; });
      return "<a:p>" + fixed + "</a:p>";
    });
    zip.file(name, out);
  }
  // notes: pptxgenjs puts all notes text in one <a:t>, line breaks included; split it into one paragraph per line
  let split = 0;
  for (const name of Object.keys(zip.files).filter(n => /^ppt\/notesSlides\/notesSlide\d+\.xml$/.test(n))) {
    const xml = await zip.file(name).async("string");
    const out = xml.replace(/<a:p><a:r><a:rPr lang="en-US" dirty="0"\/><a:t>([^<]*)<\/a:t><\/a:r><a:endParaRPr lang="en-US" dirty="0"\/><\/a:p>/, (m, t) => {
      const lines = t.split(/\r?\n/);
      if (lines.length > 1) split++;
      return lines.map(l => l ? `<a:p><a:r><a:rPr lang="en-US" dirty="0"/><a:t>${l}</a:t></a:r></a:p>` : `<a:p><a:endParaRPr lang="en-US" dirty="0"/></a:p>`).join("");
    });
    zip.file(name, out);
  }
  fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
  console.log("removed duplicate pPr:", removed, "| notes split into paragraphs:", split);
}

(async () => {
  await pres.writeFile({ fileName: "supervisor_update.pptx" });
  await applyTheme("supervisor_update.pptx", THEME);
  await keepFirstParaProps("supervisor_update.pptx");
  let md = "# Speaker notes: supervisor update, 19 October 2026\n\n";
  notes.forEach((n, i) => { md += `## Slide ${i + 1}: ${n.title}\n\n${n.text}\n\n`; });
  require("fs").writeFileSync("supervisor_talk_notes.md", md);
  console.log("written");
})();

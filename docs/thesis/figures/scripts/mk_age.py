import sys
from PIL import Image, ImageFilter
import numpy as np
src, out = sys.argv[1], sys.argv[2]
im = Image.open(src).convert('RGB'); a = np.asarray(im).astype(int)
H, Wtot = a.shape[:2]; W = Wtot / 9
AGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]   # diag_bodymodel_probe.py:59, panel order
sel = [0, 2, 4, 6, 8]
bg = (235, 235, 235)
crops = []
for i in sel:
    x0, x1 = int(round(i * W)), int(round((i + 1) * W))
    p = a[int(0.06 * H):, x0:x1]
    m = (p[..., 2] - p[..., 0]) > 12
    ys, xs = np.nonzero(m)
    pad = 4
    box = (x0 + xs.min() - pad, int(0.06 * H) + ys.min() - pad, x0 + xs.max() + pad, int(0.06 * H) + ys.max() + pad)
    crops.append(im.crop(box)); print(AGES[i], box, box[2]-box[0], box[3]-box[1])
TH = 1100  # target body height in px (all bodies scaled to the same displayed height)
cellw = 780
canvas = Image.new('RGB', (cellw * len(crops), TH + 40), bg)
for k, c in enumerate(crops):
    s = TH / c.size[1]
    c2 = c.resize((int(round(c.size[0] * s)), TH), Image.LANCZOS).filter(ImageFilter.UnsharpMask(2, 60, 2))
    canvas.paste(c2, (k * cellw + (cellw - c2.size[0]) // 2, 20))
canvas.save(out, quality=92)
print(canvas.size)

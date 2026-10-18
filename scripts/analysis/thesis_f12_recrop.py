"""Re-crop the delivered F12 montages into consistent panels.

The delivered files (cluster command of REWRITE_PLAN.md section 3) assumed 4 trace panels; the trace writes 5
(image | GT | V1_eval_masked | V2_demo | V1 pose + GT shape, each 672 px, 4-px white pads; diag_holdout_trace.py:894-904,
diag_common.py:690). So each file holds: Base image 0-672, GT 676-1348, Base V1 1352-2024, Base V2 2028-2532 (left
504 px), then Final panel cols 1688-2532 = right half of Final V1 (2532-2868) and left 504 px of Final V2 (2872-3376).
The only setting present in full for both models is the left 504 px of the V2_demo panel. This script writes
image | GT | Base V2 | Final V2, all cropped to the same left 504 px, with new labels.
"""
import sys, os
from PIL import Image, ImageDraw, ImageFont
src, dst = sys.argv[1], sys.argv[2]
W = 504
cols = {'image': 0, 'ground truth': 676, 'Base': 2028, 'Final': 2872}
font = ImageFont.load_default(size=44)
for idx in [579860, 579862, 579863, 579865, 579868, 579871]:
    a = Image.open(os.path.join(src, f'F12_holdout_{idx}_img_gt_v5_final.jpg')).convert('RGB')
    assert a.size == (3376, 672), a.size
    out = Image.new('RGB', (4 * W + 3 * 6, 672), 'white')
    for k, (name, x0) in enumerate(cols.items()):
        p = a.crop((x0, 0, x0 + W, 672))
        d = ImageDraw.Draw(p)
        d.rectangle([0, 0, 200, 18], fill=(0, 0, 0))            # hide the baked-in trace label (max 169 x 16 px)
        tw = d.textlength(name, font=font)
        d.rectangle([0, 0, tw + 16, 56], fill=(0, 0, 0))
        d.text((8, 4), name, fill=(255, 255, 255), font=font)
        out.paste(p, (k * (W + 6), 0))
    out.save(os.path.join(dst, f'F12_holdout_{idx}_demo_crop.jpg'), quality=92)
    print('wrote', idx)

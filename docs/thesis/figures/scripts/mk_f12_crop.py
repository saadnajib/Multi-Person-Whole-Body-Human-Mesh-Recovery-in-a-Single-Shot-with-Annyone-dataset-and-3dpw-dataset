import sys
from PIL import Image, ImageEnhance
src, out = sys.argv[1], sys.argv[2]
im = Image.open(src).convert("RGB")
W, H, X0, Y0 = 508, 427, 90, 170
strip = Image.new("RGB", (4 * W + 3 * 4, H), "white")
for k in range(4):
    p = im.crop((600 * k + X0, Y0, 600 * k + X0 + W, Y0 + H))
    p = Image.eval(p, lambda v: int(255 * (v / 255) ** 0.7))  # brighten the dark scene (gamma 0.7)
    p = ImageEnhance.Contrast(p).enhance(1.1)
    strip.paste(p, (k * (W + 4), 0))
strip.save(out, quality=90)

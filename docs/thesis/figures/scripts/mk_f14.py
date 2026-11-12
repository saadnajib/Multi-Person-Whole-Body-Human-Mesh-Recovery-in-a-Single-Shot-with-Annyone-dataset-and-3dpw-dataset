import sys
from PIL import Image
im = Image.open(sys.argv[1]).convert('RGB'); W, H = im.size; pw = W / 3
x0, x1, y0 = 0, 690, 22           # crop around the two actors, below the printed title strip
P = [im.crop((int(round(k * pw)) + x0, y0, int(round(k * pw)) + x1, H)) for k in range(3)]
w, h = P[0].size; g = 16
c = Image.new('RGB', (2 * w + g, 2 * h + g), (255, 255, 255))
c.paste(P[0], ((2 * w + g - w) // 2, 0)); c.paste(P[1], (0, h + g)); c.paste(P[2], (w + g, h + g))
c.save(sys.argv[2], quality=92); print(c.size)

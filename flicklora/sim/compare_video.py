"""Side-by-side comparison video: reference scene vs our restyled scene (driven) vs our contact-driven flick."""
import argparse
import os

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--clips", default="original:Reference scene (driven slide);ours_driven:Ours - restyled (driven slide);"
                                   "ours_passive:Ours - no drive - finger contact only")
ap.add_argument("--name", default="comparison.mp4")
a = ap.parse_args()

clips = []
for item in a.clips.split(";"):
    key, label = item.split(":", 1)
    p = os.path.join(a.out, key + ".mp4")
    if os.path.exists(p):
        clips.append((label, [f for f in iio.get_reader(p)]))
if not clips:
    raise SystemExit("no clips found")
h = 360
n = max(len(f) for _, f in clips)
writer = iio.get_writer(os.path.join(a.out, a.name), fps=30, codec="libx264", quality=8, macro_block_size=1)
for i in range(n):
    tiles = []
    for label, fr in clips:
        im = Image.fromarray(fr[min(i, len(fr) - 1)])
        im = im.resize((int(im.width * h / im.height), h))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, im.width, 26], fill=(0, 0, 0))
        d.text((8, 7), label, fill=(255, 255, 255))
        tiles.append(np.asarray(im))
    frame = np.concatenate(tiles, axis=1)
    writer.append_data(frame[:, : frame.shape[1] // 2 * 2])
writer.close()
print("wrote", a.name, n, "frames,", len(clips), "clips")

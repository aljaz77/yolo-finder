"""Draws the app icon (magnifier over a detection box) -> icon.png + icon.ico."""
import sys
from pathlib import Path

from PIL import Image, ImageDraw


def draw(size=512):
    s = size / 512
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([16 * s, 16 * s, 496 * s, 496 * s], radius=96 * s, fill=(30, 34, 44, 255))
    # detection box with corner label
    d.rectangle([70 * s, 90 * s, 330 * s, 350 * s], outline=(255, 196, 0, 255), width=int(18 * s))
    d.rectangle([70 * s, 60 * s, 200 * s, 100 * s], fill=(255, 196, 0, 255))
    # person silhouette in the box
    d.ellipse([170 * s, 140 * s, 220 * s, 190 * s], fill=(230, 233, 240, 255))
    d.rounded_rectangle([160 * s, 200 * s, 230 * s, 320 * s], radius=22 * s, fill=(230, 233, 240, 255))
    # magnifier
    cx, cy, r = 290 * s, 280 * s, 110 * s
    d.line([cx + r * 0.68, cy + r * 0.68, 450 * s, 440 * s], fill=(61, 139, 253, 255), width=int(46 * s))
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(61, 139, 253, 255), width=int(34 * s))
    lens = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ImageDraw.Draw(lens).ellipse([cx - r + 17 * s, cy - r + 17 * s, cx + r - 17 * s, cy + r - 17 * s],
                                 fill=(61, 139, 253, 70))
    return Image.alpha_composite(im, lens)


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out.mkdir(parents=True, exist_ok=True)
    im = draw(512)
    im.save(out / "icon.png")
    im.save(out / "icon.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("icon written to", out)

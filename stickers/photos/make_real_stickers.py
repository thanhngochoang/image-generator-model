"""Lumen face stickers cut from real CC0 / public-domain photos (sources and credits in CREDITS.json).

usage: make_real_stickers.py SRC_DIR ISNET_ONNX OUT_DIR
Every picture is cut from a photo with ISNet (DIS, general use, Apache-2.0) plus a hand-drawn region for the part
(an ear, a nose); shapes are given in the units of the 900-px preview of each photo (scaled to the original here).
Writes OUT_DIR/{*.png, stickers.json, CREDITS.json, face_stickers.zip}.
"""
import json, math, os, sys, zipfile
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageChops, ImageDraw, ImageFilter

SRC, MODEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])


def load(name):
    im = Image.open(os.path.join(SRC, name + ".jpg")).convert("RGB")
    return im, max(im.size) / 900.0


def matte(im):
    """ISNet subject alpha at the photo's size."""
    x = (np.asarray(im.resize((1024, 1024), Image.BILINEAR), np.float32) / 255.0 - 0.5)
    o = sess.run(None, {sess.get_inputs()[0].name: x.transpose(2, 0, 1)[None]})[0][0, 0]
    o = (o - o.min()) / max(1e-6, o.max() - o.min())
    return Image.fromarray((o * 255).astype(np.uint8)).resize(im.size, Image.BILINEAR)


def region(size, s, polys=(), ellipses=(), feather=6):
    """Mask from polygons / ellipses in preview units (× s = original pixels), softened by [feather] px."""
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    for p in polys:
        d.polygon([(x * s, y * s) for x, y in p], fill=255)
    for cx, cy, rx, ry in ellipses:
        d.ellipse([(cx - rx) * s, (cy - ry) * s, (cx + rx) * s, (cy + ry) * s], fill=255)
    return m.filter(ImageFilter.GaussianBlur(feather)) if feather else m


def cut(im, alpha, upscale=1.0, pad=8):
    """RGBA of [im] with [alpha], trimmed to what is visible, optionally enlarged (Lanczos)."""
    rgba = im.copy()
    rgba.putalpha(alpha)
    box = alpha.point(lambda v: 255 if v > 12 else 0).getbbox()
    rgba = rgba.crop((max(0, box[0] - pad), max(0, box[1] - pad), min(im.width, box[2] + pad), min(im.height, box[3] + pad)))
    if upscale != 1.0:
        rgba = rgba.resize((round(rgba.width * upscale), round(rgba.height * upscale)), Image.LANCZOS)
    return rgba


def fade_bottom(rgba, share):
    """Alpha fades out over the bottom [share] of the picture (where an ear meets the head)."""
    a = np.asarray(rgba.getchannel("A"), np.float32)
    h = a.shape[0]
    start = int(h * (1 - share))
    ramp = np.ones(h, np.float32)
    ramp[start:] = np.linspace(1, 0, h - start) ** 1.5
    rgba.putalpha(Image.fromarray((a * ramp[:, None]).astype(np.uint8)))
    return rgba


def save(img, name):
    img.save(os.path.join(OUT, name), optimize=True)
    return name


def side_by_side(left, right, gap):
    w = left.width + gap + right.width
    h = max(left.height, right.height)
    c = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    c.alpha_composite(left, (0, h - left.height))
    c.alpha_composite(right, (left.width + gap, h - right.height))
    return c


pics = {}

# ---- cat: sphynx ears (head dome between them removed) + a kitten's pink nose ----
im, s = load("cat_2")
a = ImageChops.multiply(matte(im), region(im.size, s, polys=[[(0, 0), (900, 0), (900, 300), (0, 300)]], feather=4))
dome = region(im.size, s, ellipses=[(440, 335, 172, 160)], feather=10)
a = ImageChops.subtract(a, dome)
pics["cat_ears"] = save(cut(im, a, 1.2), "cat_ears.png")
im, s = load("cat_19")
pics["cat_nose"] = save(cut(im, region(im.size, s, ellipses=[(283, 422, 58, 42)], feather=5), 1.5), "cat_nose.png")

# ---- bunny: one real ear, mirrored and turned for the other ----
im, s = load("bunny_5")
ear = ImageChops.multiply(matte(im), region(im.size, s, polys=[[(287, 360), (292, 120), (322, 55), (382, 36), (452, 52), (497, 112), (515, 235), (505, 360)]], feather=3))
ear = fade_bottom(cut(im, ear, 1.2), 0.18)
l = ear.rotate(12, resample=Image.BICUBIC, expand=True)
r = ear.transpose(Image.FLIP_LEFT_RIGHT).rotate(-12, resample=Image.BICUBIC, expand=True)
pics["bunny_ears"] = save(side_by_side(l, r, int(ear.width * 0.55)), "bunny_ears.png")

# ---- puppy: a boxer's floppy ear (mirrored for the pair) and a dog's nose ----
im, s = load("dog_10")
ear = ImageChops.multiply(matte(im), region(im.size, s, polys=[[(560, 12), (700, 4), (790, 40), (812, 110), (802, 200), (782, 248), (748, 185), (692, 98), (598, 62)]], feather=3))
ear = cut(im, ear, 1.2)
r = ear.rotate(-8, resample=Image.BICUBIC, expand=True)
l = ear.transpose(Image.FLIP_LEFT_RIGHT).rotate(8, resample=Image.BICUBIC, expand=True)
pics["puppy_ears"] = save(side_by_side(l, r, int(ear.width * 1.5)), "puppy_ears.png")
im, s = load("dog_20")
pics["puppy_nose"] = save(cut(im, region(im.size, s, ellipses=[(441, 246, 30, 21)], feather=3), 2.5), "puppy_nose.png")


# ---- tiara: a crystal tiara photographed on white ----
im, s = load("crown_15")
pics["tiara"] = save(cut(im, matte(im), 1.8), "tiara.png")

# ---- yellow sunglasses ----
im, s = load("sunglasses_0")
pics["sunglasses"] = save(cut(im, matte(im), 2.6), "sunglasses.png")

# ---- flower crown: real flowers (one photographed cluster) set along an arc ----
im, s = load("flower_crown_0")
cluster = cut(im, matte(im), 0.36)
W, H = 1500, 560
crown = Image.new("RGBA", (W, H), (0, 0, 0, 0))
n = 7
for i in range(n):
    t = (i / (n - 1)) * 2 - 1  # -1..1 across the forehead
    ang = t * 62
    x = W / 2 + math.sin(math.radians(ang)) * 640
    y = H - 120 - math.cos(math.radians(ang)) * 300 + 300 - 150
    k = 0.8 + 0.25 * (1 - abs(t))
    c = cluster.resize((int(cluster.width * k), int(cluster.height * k)), Image.LANCZOS).rotate(-ang + (17 if i % 2 else -11), resample=Image.BICUBIC, expand=True)
    crown.alpha_composite(c, (int(x - c.width / 2), int(y - c.height / 2)))
box = crown.getchannel("A").getbbox()
pics["flower_crown"] = save(crown.crop(box), "flower_crown.png")

# ---- blush: a soft rosy patch (makeup, not a picture) ----
b = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
yy, xx = np.mgrid[0:256, 0:256]
d = np.sqrt(((xx - 128) / 120.0) ** 2 + ((yy - 128) / 90.0) ** 2)
al = np.clip(1 - d, 0, 1) ** 1.8 * 150
arr = np.zeros((256, 256, 4), np.uint8)
arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3] = 240, 120, 140, al.astype(np.uint8)
pics["blush"] = save(Image.fromarray(arr, "RGBA"), "blush.png")


def part(f, anchor, width, dx=0.0, dy=0.0):
    return {"file": pics[f], "anchor": anchor, "width": width, "dx": dx, "dy": dy}


stickers = [
    {"id": "cat", "name": {"vi": "Tai mèo", "en": "Cat"}, "thumb": pics["cat_ears"], "parts": [part("cat_ears", "forehead", 1.25, 0, -0.42), part("cat_nose", "nose", 0.3, 0, 0.03)]},
    {"id": "bunny", "name": {"vi": "Tai thỏ", "en": "Bunny"}, "thumb": pics["bunny_ears"], "parts": [part("bunny_ears", "forehead", 1.2, 0, -0.55)]},
    {"id": "puppy", "name": {"vi": "Cún con", "en": "Puppy"}, "thumb": pics["puppy_ears"], "parts": [part("puppy_ears", "forehead", 1.5, 0, -0.3), part("puppy_nose", "nose", 0.36, 0, 0.0)]},
    {"id": "tiara", "name": {"vi": "Vương miện", "en": "Tiara"}, "thumb": pics["tiara"], "parts": [part("tiara", "forehead", 0.78, 0, -0.27)]},
    {"id": "shades", "name": {"vi": "Kính râm", "en": "Sunglasses"}, "thumb": pics["sunglasses"], "parts": [part("sunglasses", "eyes", 1.08, 0, 0.03)]},
    {"id": "flowers", "name": {"vi": "Vòng hoa", "en": "Flower crown"}, "thumb": pics["flower_crown"], "parts": [part("flower_crown", "forehead", 1.4, 0, -0.3)]},
    {"id": "blush", "name": {"vi": "Má hồng", "en": "Blush"}, "thumb": pics["blush"], "parts": [part("blush", "cheek_left", 0.32), part("blush", "cheek_right", 0.32)]},
]
manifest = {"version": 1, "license": "Cut from CC0 / public-domain photos (see CREDITS.json); this pack: CC0-1.0", "stickers": stickers}
json.dump(manifest, open(os.path.join(OUT, "stickers.json"), "w"), ensure_ascii=False, indent=1)

credits = json.load(open(os.path.join(SRC, "credits.json")))
used = {"cat_ears": "cat_2", "cat_nose": "cat_19", "bunny_ears": "bunny_5", "puppy_ears": "dog_10", "puppy_nose": "dog_20",
        "tiara": "crown_15", "sunglasses": "sunglasses_0", "flower_crown": "flower_crown_0"}
json.dump({pics[k]: credits[v] for k, v in used.items()} | {"blush.png": "drawn by the script"}, open(os.path.join(OUT, "CREDITS.json"), "w"), ensure_ascii=False, indent=1)

files = sorted(set(pics.values())) + ["stickers.json"]
with zipfile.ZipFile(os.path.join(OUT, "face_stickers.zip"), "w", zipfile.ZIP_STORED) as z:
    for f in files:
        z.write(os.path.join(OUT, f), f)
for f in files:
    print(f, os.path.getsize(os.path.join(OUT, f)))

"""Draws Lumen's face stickers (original art, CC0) and writes stickers.json + face_stickers.zip.

usage: python make_stickers.py <out_dir>
Every picture is drawn at 4x and scaled down for smooth edges. Parts are placed on the face by the app from the
manifest: anchor landmark, width in face widths, offset (dx, dy) in face widths along the face's own axes.
"""
import json, math, os, sys, zipfile
from PIL import Image, ImageDraw, ImageFilter

SS = 4  # supersampling


def canvas(w, h):
    img = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)


def done(img, w, h):
    return img.resize((w, h), Image.LANCZOS)


def S(*v):
    return [x * SS for x in v]


def ellipse(d, box, **kw):
    d.ellipse(S(*box), **kw)


def poly(d, pts, **kw):
    d.polygon([(x * SS, y * SS) for x, y in pts], **kw)


def outline_kw(width, color=(40, 30, 40, 255)):
    return dict(outline=color, width=width * SS)


def cat_ears():
    img, d = canvas(660, 260)
    for sx in (1, -1):
        cx = 330 + sx * 200
        outer = [(cx - 95 * sx, 250), (cx + 20 * sx, 20), (cx + 115 * sx, 250)]
        inner = [(cx - 55 * sx, 235), (cx + 18 * sx, 75), (cx + 80 * sx, 235)]
        poly(d, outer, fill=(70, 62, 72, 255), **outline_kw(6, (30, 25, 32, 255)))
        poly(d, inner, fill=(255, 158, 187, 255))
    return done(img, 660, 260)


def cat_nose():
    img, d = canvas(360, 160)
    for sy, ang in ((-1, -8), (0, 0), (1, 8)):
        y = 80 + sy * 22
        d.line(S(30, y - ang, 135, 80 + sy * 8), fill=(40, 35, 40, 230), width=5 * SS)
        d.line(S(225, 80 + sy * 8, 330, y - ang), fill=(40, 35, 40, 230), width=5 * SS)
    poly(d, [(150, 58), (210, 58), (180, 98)], fill=(255, 120, 160, 255), **outline_kw(4, (120, 40, 70, 255)))
    return done(img, 360, 160)


def bunny_ears():
    img, d = canvas(460, 420)
    for sx, tilt in ((-1, -12), (1, 12)):
        ear, ed = canvas(150, 400)
        ellipse(ed, (10, 10, 140, 395), fill=(250, 250, 252, 255), **outline_kw(6, (90, 80, 95, 255)))
        ellipse(ed, (45, 60, 105, 360), fill=(255, 170, 195, 255))
        ear = ear.rotate(-tilt, resample=Image.BICUBIC, expand=True)
        x = 230 + sx * 95 - ear.width // (2 * SS) // 1
        img.alpha_composite(ear, (int((230 + sx * 95) * SS - ear.width / 2), int(10 * SS)))
    return done(img, 460, 420)


def crown():
    img, d = canvas(520, 300)
    pts = [(20, 280), (40, 90), (140, 190), (260, 30), (380, 190), (480, 90), (500, 280)]
    poly(d, pts, fill=(255, 200, 40, 255), **outline_kw(7, (170, 110, 0, 255)))
    d.rectangle(S(20, 250, 500, 290), fill=(240, 180, 30, 255), outline=(170, 110, 0, 255), width=6 * SS)
    for (x, y, c) in ((260, 270, (220, 40, 60)), (140, 270, (40, 120, 230)), (380, 270, (40, 170, 90)), (260, 120, (220, 40, 60))):
        ellipse(d, (x - 20, y - 20, x + 20, y + 20), fill=c + (255,), **outline_kw(4, (255, 255, 255, 255)))
    for (x, y) in ((40, 90), (260, 30), (480, 90)):
        ellipse(d, (x - 16, y - 16, x + 16, y + 16), fill=(255, 240, 200, 255), **outline_kw(4, (170, 110, 0, 255)))
    return done(img, 520, 300)


def heart(d, cx, cy, r, fill, ow, oc):
    pts = []
    for i in range(200):
        t = 2 * math.pi * i / 200
        x = 16 * math.sin(t) ** 3
        y = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
        pts.append((cx + x * r / 17, cy + y * r / 17))
    poly(d, pts, fill=fill, outline=oc, width=ow * SS)


def heart_glasses():
    img, d = canvas(640, 260)
    d.line(S(270, 105, 370, 105), fill=(200, 20, 70, 255), width=12 * SS)
    for cx in (170, 470):
        heart(d, cx, 140, 120, (255, 60, 120, 200), 10, (190, 10, 70, 255))
    d.line(S(30, 95, 0, 80), fill=(190, 10, 70, 255), width=10 * SS)
    d.line(S(610, 95, 640, 80), fill=(190, 10, 70, 255), width=10 * SS)
    ellipse(d, (100, 80, 140, 110), fill=(255, 255, 255, 170))
    ellipse(d, (400, 80, 440, 110), fill=(255, 255, 255, 170))
    return done(img, 640, 260)


def sunglasses():
    img, d = canvas(640, 220)
    d.line(S(270, 70, 370, 70), fill=(25, 25, 30, 255), width=14 * SS)
    for x0 in (40, 370):
        d.rounded_rectangle(S(x0, 40, x0 + 230, 190), radius=60 * SS, fill=(20, 22, 30, 245), outline=(5, 5, 8, 255), width=10 * SS)
        d.line(S(x0 + 40, 75, x0 + 100, 60), fill=(255, 255, 255, 140), width=10 * SS)
    d.line(S(40, 70, 0, 55), fill=(20, 20, 25, 255), width=12 * SS)
    d.line(S(600, 70, 640, 55), fill=(20, 20, 25, 255), width=12 * SS)
    return done(img, 640, 220)


def blush():
    img, d = canvas(300, 200)
    ellipse(d, (30, 50, 270, 170), fill=(255, 110, 140, 150))
    img = img.filter(ImageFilter.GaussianBlur(14 * SS))
    d = ImageDraw.Draw(img)
    for (x, y, r) in ((200, 50, 18), (240, 85, 10), (70, 150, 12)):
        poly(d, [(x, y - r), (x + r * 0.3, y - r * 0.3), (x + r, y), (x + r * 0.3, y + r * 0.3), (x, y + r), (x - r * 0.3, y + r * 0.3), (x - r, y), (x - r * 0.3, y - r * 0.3)], fill=(255, 255, 255, 235))
    return done(img, 300, 200)


def flower_crown():
    img, d = canvas(700, 240)
    pts = [(70 + i * 560 / 8, 165 - 70 * math.sin(math.pi * i / 8)) for i in range(9)]
    for (x, y) in pts[::2]:
        for a in (-0.6, 0.6):
            poly(d, [(x, y), (x + 60 * math.cos(a + 1.2), y - 30 * math.sin(a + 1.2)), (x + 70 * math.cos(a), y + 25 * math.sin(a))], fill=(90, 170, 90, 255))
    colours = [(255, 140, 180), (255, 220, 90), (255, 255, 255), (190, 150, 255)]
    for i, (x, y) in enumerate(pts):
        c = colours[i % len(colours)]
        for k in range(5):
            a = 2 * math.pi * k / 5
            ellipse(d, (x + 26 * math.cos(a) - 22, y + 26 * math.sin(a) - 22, x + 26 * math.cos(a) + 22, y + 26 * math.sin(a) + 22), fill=c + (255,), **outline_kw(3, (200, 120, 150, 255)))
        ellipse(d, (x - 14, y - 14, x + 14, y + 14), fill=(255, 190, 60, 255))
    return done(img, 700, 240)


def mustache():
    img, d = canvas(500, 150)
    col = (70, 45, 30, 255)
    for sx in (1, -1):
        cx = 250 + sx * 80
        ellipse(d, (cx - 85, 45, cx + 85, 120), fill=col)
        x = 250 + sx * 165
        d.arc(S(x - 35, 20, x + 35, 90), 90 if sx > 0 else 0, 270 if sx > 0 else 180, fill=col, width=22 * SS)
    ellipse(d, (230, 50, 270, 80), fill=(70, 45, 30, 255))
    return done(img, 500, 150)


def party_hat():
    img, d = canvas(300, 400)
    poly(d, [(30, 380), (150, 60), (270, 380)], fill=(80, 170, 255, 255), **outline_kw(6, (20, 70, 160, 255)))
    for i, y in enumerate((150, 230, 310)):
        half = (y - 60) / 320 * 120
        d.line(S(150 - half, y, 150 + half, y + 20), fill=(255, 220, 60, 255), width=14 * SS)
    for (x, y, c) in ((120, 200, (255, 90, 130)), (180, 270, (120, 220, 120)), (100, 330, (255, 255, 255))):
        ellipse(d, (x - 10, y - 10, x + 10, y + 10), fill=c + (255,))
    ellipse(d, (115, 20, 185, 90), fill=(255, 90, 130, 255), **outline_kw(4, (190, 40, 80, 255)))
    return done(img, 300, 400)


def devil_horns():
    img, d = canvas(560, 240)
    for sx in (1, -1):
        base = 280 + sx * 170
        pts = []
        for i in range(31):
            t = i / 30
            x = base + sx * (40 * t - 10 * math.sin(math.pi * t))
            y = 230 - 200 * t
            w = 45 * (1 - t) + 2
            pts.append((x - w, y)); pts.insert(0, (x + w, y))
        poly(d, pts, fill=(220, 30, 40, 255), **outline_kw(6, (120, 0, 10, 255)))
    return done(img, 560, 240)


def halo():
    img, d = canvas(460, 160)
    ellipse(d, (20, 30, 440, 130), outline=(255, 215, 80, 255), width=24 * SS)
    glow = img.filter(ImageFilter.GaussianBlur(10 * SS))
    glow.alpha_composite(img)
    return done(glow, 460, 160)


def puppy_ears():
    img, d = canvas(820, 380)
    for sx in (1, -1):
        cx = 410 + sx * 285
        ear, ed = canvas(200, 340)
        ellipse(ed, (10, 10, 190, 330), fill=(150, 95, 55, 255), **outline_kw(6, (90, 55, 30, 255)))
        ellipse(ed, (55, 80, 145, 300), fill=(200, 140, 95, 255))
        ear = ear.rotate(sx * 18, resample=Image.BICUBIC, expand=True)
        img.alpha_composite(ear, (int(cx * SS - ear.width / 2), int(10 * SS)))
    return done(img, 820, 380)


def puppy_nose():
    img, d = canvas(220, 150)
    ellipse(d, (20, 20, 200, 130), fill=(30, 25, 28, 255))
    ellipse(d, (60, 35, 110, 65), fill=(255, 255, 255, 150))
    return done(img, 220, 150)


def puppy_tongue():
    img, d = canvas(140, 180)
    ellipse(d, (15, 10, 125, 170), fill=(255, 120, 150, 255), **outline_kw(5, (190, 60, 90, 255)))
    d.line(S(70, 30, 70, 120), fill=(200, 70, 100, 255), width=5 * SS)
    return done(img, 140, 180)


PICTURES = {
    "cat_ears.png": cat_ears, "cat_nose.png": cat_nose, "bunny_ears.png": bunny_ears, "crown.png": crown,
    "heart_glasses.png": heart_glasses, "sunglasses.png": sunglasses, "blush.png": blush, "flower_crown.png": flower_crown,
    "mustache.png": mustache, "party_hat.png": party_hat, "devil_horns.png": devil_horns, "halo.png": halo,
    "puppy_ears.png": puppy_ears, "puppy_nose.png": puppy_nose, "puppy_tongue.png": puppy_tongue,
}

STICKERS = [
    ("cat", "Tai mèo", "Cat", "cat_ears.png", [("cat_ears.png", "forehead", 1.3, 0, -0.34), ("cat_nose.png", "nose", 0.7, 0, 0.04)]),
    ("bunny", "Tai thỏ", "Bunny", "bunny_ears.png", [("bunny_ears.png", "forehead", 0.95, 0, -0.55)]),
    ("crown", "Vương miện", "Crown", "crown.png", [("crown.png", "forehead", 0.85, 0, -0.32)]),
    ("hearts", "Kính tim", "Heart glasses", "heart_glasses.png", [("heart_glasses.png", "eyes", 1.1, 0, 0.02)]),
    ("shades", "Kính râm", "Sunglasses", "sunglasses.png", [("sunglasses.png", "eyes", 1.05, 0, 0.03)]),
    ("blush", "Má hồng", "Blush", "blush.png", [("blush.png", "cheek_left", 0.32, 0, 0), ("blush.png", "cheek_right", 0.32, 0, 0)]),
    ("flowers", "Vòng hoa", "Flower crown", "flower_crown.png", [("flower_crown.png", "forehead", 1.25, 0, -0.18)]),
    ("mustache", "Ria mép", "Mustache", "mustache.png", [("mustache.png", "mouth", 0.6, 0, -0.1)]),
    ("party", "Mũ tiệc", "Party hat", "party_hat.png", [("party_hat.png", "forehead", 0.7, 0.12, -0.72)]),
    ("devil", "Sừng quỷ", "Devil horns", "devil_horns.png", [("devil_horns.png", "forehead", 1.0, 0, -0.2)]),
    ("halo", "Hào quang", "Halo", "halo.png", [("halo.png", "forehead", 0.9, 0, -0.5)]),
    ("puppy", "Cún con", "Puppy", "puppy_ears.png", [("puppy_ears.png", "forehead", 1.5, 0, -0.16), ("puppy_nose.png", "nose", 0.3, 0, 0.0), ("puppy_tongue.png", "mouth", 0.16, 0, 0.12)]),
]


def main(out):
    os.makedirs(out, exist_ok=True)
    for name, fn in PICTURES.items():
        fn().save(os.path.join(out, name), optimize=True)
    manifest = {"version": 1, "license": "CC0-1.0, original art made for Lumen", "stickers": [
        {"id": i, "name": {"vi": vi, "en": en}, "thumb": thumb,
         "parts": [{"file": f, "anchor": a, "width": w, "dx": dx, "dy": dy} for f, a, w, dx, dy in parts]}
        for i, vi, en, thumb, parts in STICKERS]}
    with open(os.path.join(out, "stickers.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    names = ["stickers.json"] + sorted(PICTURES)
    with zipfile.ZipFile(os.path.join(out, "face_stickers.zip"), "w", zipfile.ZIP_STORED) as z:
        for n in names:
            zi = zipfile.ZipInfo(n, date_time=(2026, 11, 23, 0, 0, 0))
            with open(os.path.join(out, n), "rb") as f:
                z.writestr(zi, f.read())
    print("wrote", len(names), "files")


if __name__ == "__main__":
    main(sys.argv[1])

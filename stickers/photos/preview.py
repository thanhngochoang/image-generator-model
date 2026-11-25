"""Places every sticker of OUT_DIR/stickers.json on faces (MediaPipe landmarks, same maths as FaceStickers.place)."""
import json, math, os, sys
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from PIL import Image
OUT, TASK, sheet = sys.argv[1], sys.argv[2], sys.argv[3]
photos = sys.argv[4:]
lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=TASK), num_faces=1))
man = json.load(open(os.path.join(OUT, "stickers.json")))
ANCH = {"forehead": [10], "eyes": [33, 263], "nose": [1], "mouth": [13, 14], "chin": [152], "cheek_left": [50], "cheek_right": [280]}
tiles = []
for ph in photos:
    base = Image.open(ph).convert("RGB")
    res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=__import__("numpy").asarray(base)))
    pts = res.face_landmarks[0]
    w, h = base.size
    P = lambda i: (pts[i].x * w, pts[i].y * h)
    fw = math.dist(P(454), P(234)); ang = math.atan2(P(263)[1] - P(33)[1], P(263)[0] - P(33)[0])
    for st in man["stickers"]:
        im = base.convert("RGBA")
        for p in st["parts"]:
            ax = sum(P(i)[0] for i in ANCH[p["anchor"]]) / len(ANCH[p["anchor"]]); ay = sum(P(i)[1] for i in ANCH[p["anchor"]]) / len(ANCH[p["anchor"]])
            ox, oy = p["dx"] * fw, p["dy"] * fw
            cx = ax + ox * math.cos(ang) - oy * math.sin(ang); cy = ay + ox * math.sin(ang) + oy * math.cos(ang)
            pic = Image.open(os.path.join(OUT, p["file"])).convert("RGBA")
            tw = p["width"] * fw; th = tw * pic.height / pic.width
            pic = pic.resize((max(1, int(tw)), max(1, int(th))), Image.LANCZOS).rotate(-math.degrees(ang), resample=Image.BICUBIC, expand=True)
            im.alpha_composite(pic, (int(cx - pic.width / 2), int(cy - pic.height / 2)))
        cx0, cy0 = P(1); half = 1.25 * fw
        t = im.convert("RGB").crop((int(cx0 - half), int(cy0 - 1.45 * half), int(cx0 + half), int(cy0 + 1.05 * half))); t = t.resize((360, 360)); tiles.append(t)
cols = len(man["stickers"])
S = Image.new("RGB", (cols * 360, (len(tiles) // cols) * 360), "white")
for i, t in enumerate(tiles): S.paste(t, ((i % cols) * 360, (i // cols) * 360))
S.save(sheet, quality=88)

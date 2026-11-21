"""Model inputs built with the app's own pre-processing (see lumen MODELS.md and engine/*Math.kt), for evaluation and
calibration. Each builder returns a list of (name, feeds) pairs."""
import os, sys, math
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image
from common import load_rgb, rgb2lab, lab2rgb, EMU, SAMPLES, ROOT

SK = f"{ROOT}/inputs/skimage"


# ---------------------------------------------------------------- GFPGAN (FaceRestoreMath)
TEMPLATE = np.array([192.98138, 239.94708, 318.90277, 240.1936, 256.63416, 314.01935, 201.26117, 371.41043, 313.08905, 371.15118]).reshape(5, 2)


def similarity(src, dst):
    """FaceRestoreMath.similarity: (a, b, tx, ty) for x' = a x - b y + tx, y' = b x + a y + ty."""
    s = src.mean(0); d = dst.mean(0)
    x = src - s; u = dst - d
    den = (x ** 2).sum()
    a = (x[:, 0] * u[:, 0] + x[:, 1] * u[:, 1]).sum() / den
    b = (x[:, 0] * u[:, 1] - x[:, 1] * u[:, 0]).sum() / den
    return np.array([a, b, d[0] - (a * s[0] - b * s[1]), d[1] - (b * s[0] + a * s[1])])


def invert(m):
    a, b, tx, ty = m; s = a * a + b * b; ia, ib = a / s, -b / s
    return np.array([ia, ib, -(ia * tx - ib * ty), -(ib * tx + ia * ty)])


def face_crop(rgb, pts5):
    """FaceRestoreMath.crop: aligned 512 crop, bilinear, edge clamp, NCHW in [-1, 1]."""
    h, w = rgb.shape[:2]
    inv = invert(similarity(pts5, TEMPLATE))
    cy, cx = np.mgrid[0:512, 0:512].astype(np.float64) + 0.5
    px = inv[0] * cx - inv[1] * cy + inv[2] - 0.5
    py = inv[1] * cx + inv[0] * cy + inv[3] - 0.5
    px = np.clip(px, 0, w - 1); py = np.clip(py, 0, h - 1)
    x0 = px.astype(int); y0 = py.astype(int); x1 = np.minimum(x0 + 1, w - 1); y1 = np.minimum(y0 + 1, h - 1)
    fx = (px - x0)[..., None]; fy = (py - y0)[..., None]
    f = rgb.astype(np.float64)
    c = (f[y0, x0] * (1 - fx) + f[y0, x1] * fx) * (1 - fy) + (f[y1, x0] * (1 - fx) + f[y1, x1] * fx) * fy
    return (c / 127.5 - 1).transpose(2, 0, 1)[None].astype(np.float32)


# Five landmarks (left eye, right eye, nose tip, left / right mouth corner) marked by hand on gridded zooms, in pixels.
# Only one real identity is available on this Mac (the NASA astronaut portrait in several app test photos), so the set
# varies scale, colour, rotation and degradation instead: full-res, grey, red-eye, ID-sheet tile, tiny bystander face, rotated.
ASTRO = [[202.5, 101], [246.5, 103.5], [224, 126], [204, 141], [243, 141]]
FACES = {
    "astro": (f"{EMU}/astro.jpg", ASTRO),
    "astro_gray": (f"{EMU}/lumen_gray.jpg", ASTRO),
    "astro_redeye": (f"{EMU}/lumen_redeye.jpg", ASTRO),
    "sheet_id": (f"{EMU}/sheet.jpg", [[190, 362], [268, 360], [230, 412], [188, 450], [256, 448]]),
    "bystander_tiny": (f"{EMU}/lumen_bystander.jpg", [[614.5, 53.5], [633.5, 53.5], [624, 62.5], [617, 72.5], [632, 72.5]]),
    "astro_rot20": (f"{EMU}/astro.jpg", "rot20"),
}


def gfpgan_feeds(extra=False):
    out = []
    for name, (path, pts) in FACES.items():
        rgb = load_rgb(path)
        if pts == "rot20":
            im = Image.fromarray(rgb); c = (224, 120); ang = 20.0
            rgb = np.asarray(im.rotate(ang, Image.BILINEAR, center=c))  # counter-clockwise
            t = math.radians(ang); P = np.array(ASTRO, float) - c
            # PIL rotates the image counter-clockwise; in image coords (y down) a point moves by angle -t
            pts = np.stack([P[:, 0] * math.cos(t) + P[:, 1] * math.sin(t), -P[:, 0] * math.sin(t) + P[:, 1] * math.cos(t)], -1) + c
        out.append((name, {"input": face_crop(rgb, np.array(pts, float))}))
    if extra:  # calibration only: flips and brightness changes of the same faces
        aug = []
        for name, f in out:
            x = f["input"]
            aug.append((name + "_flip", {"input": x[:, :, :, ::-1].copy()}))
            aug.append((name + "_dark", {"input": np.clip((x + 1) * 0.7 - 1, -1, 1).astype(np.float32)}))
        out += aug
    return out


# ---------------------------------------------------------------- MI-GAN (RemovalMath / ObjectRemoval)
def dilate(hole, r):
    from scipy.ndimage import maximum_filter
    return maximum_filter(hole, size=2 * r + 1)


def migan_case(path, rects, ellipse=False):
    rgb = load_rgb(path)
    h, w = rgb.shape[:2]
    hole = np.zeros((h, w), np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    for (u0, v0, u1, v1) in rects:
        x0, y0, x1, y1 = int(u0 * w), int(v0 * h), int(u1 * w), int(v1 * h)
        if ellipse:
            cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
            hole[((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1] = 255
        else:
            hole[y0:y1, x0:x1] = 255
    r = max(1, round(min(w, h) * 0.01))
    grown = dilate(hole, r)
    mask = np.where(grown != 0, 0, 255).astype(np.uint8)
    feeds = {"image": rgb.transpose(2, 0, 1)[None].copy(), "mask": mask[None, None]}
    return feeds, grown != 0


MIGAN_CASES = {
    "astro_helmet_small": (f"{EMU}/astro.jpg", [(0.70, 0.80, 0.80, 0.90)], True),
    "coffee_cup_medium": (f"{SK}/coffee.png", [(0.30, 0.25, 0.60, 0.65)], True),
    "chelsea_large": (f"{SK}/chelsea.png", [(0.35, 0.15, 0.85, 0.85)], False),
    "motorcycle_two_holes": (f"{SK}/motorcycle_left.png", [(0.10, 0.10, 0.25, 0.30), (0.55, 0.50, 0.75, 0.70)], True),
    "rocket_tower": (f"{SK}/rocket.jpg", [(0.40, 0.20, 0.55, 0.80)], False),
}


def migan_feeds():
    return [(k, migan_case(*v)[0]) for k, v in MIGAN_CASES.items()]


def migan_calib():
    out = migan_feeds()
    extra = [(f"{SK}/astronaut.png", [(0.6, 0.6, 0.9, 0.95)]), (f"{SK}/hubble_deep_field.jpg", [(0.2, 0.2, 0.4, 0.5)]),
             (f"{SK}/grass.png", [(0.3, 0.3, 0.6, 0.6)]), (f"{SK}/brick.png", [(0.1, 0.5, 0.4, 0.8)]),
             (f"{SAMPLES}/real_Owl.jpg", [(0.1, 0.1, 0.3, 0.3)]), (f"{SAMPLES}/real_Parrot.jpg", [(0.5, 0.2, 0.8, 0.5)]),
             (f"{SAMPLES}/real_two.jpg", [(0.42, 0.1, 0.58, 0.9)]), (f"{SK}/ihc.png", [(0.4, 0.4, 0.7, 0.7)]),
             (f"{SK}/retina.jpg", [(0.2, 0.3, 0.4, 0.6)]), (f"{SK}/gravel.png", [(0.5, 0.1, 0.9, 0.3)])]
    for i, (p, r) in enumerate(extra):
        out.append((f"cal{i}", migan_case(p, r, True)[0]))
    return out


# ---------------------------------------------------------------- MODNet (SegmenterMath PORTRAIT spec)
def modnet_size(w, h, short=512, mult=32, max_side=2048):
    scale = min(1.0, short / min(w, h), max_side / max(w, h))
    fit = lambda v: max(mult, (math.floor(v * scale) // mult) * mult)
    return fit(w), fit(h)


def modnet_input(path):
    rgb = load_rgb(path)
    iw, ih = modnet_size(rgb.shape[1], rgb.shape[0])
    small = np.asarray(Image.fromarray(rgb).resize((iw, ih), Image.BILINEAR)).astype(np.float32)
    return {"input": ((small / 255 - 0.5) / 0.5).transpose(2, 0, 1)[None].astype(np.float32)}


MODNET_EVAL = [f"{EMU}/astro.jpg", f"{SAMPLES}/real_two.jpg", f"{EMU}/sheet.jpg", f"{SK}/camera.png", f"{EMU}/lumen_gray.jpg", f"{EMU}/lumen_bystander.jpg"]
MODNET_CAL = MODNET_EVAL + [f"{SK}/astronaut.png", f"{SK}/chelsea.png", f"{SK}/coffee.png", f"{SK}/motorcycle_left.png",
                            f"{SK}/rocket.jpg", f"{SAMPLES}/real_Owl.jpg", f"{SAMPLES}/real_Parrot.jpg", f"{SAMPLES}/lumen_sample_port.jpg"]


def modnet_feeds(cal=False):
    return [(os.path.basename(p).split(".")[0], modnet_input(p)) for p in (MODNET_CAL if cal else MODNET_EVAL)]


# ---------------------------------------------------------------- MobileSAM (SamMath)
def sam_input(rgb):
    h, w = rgb.shape[:2]
    s = 1024 / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    im = Image.fromarray(rgb).resize((nw, nh), Image.BOX if s < 1 else Image.BILINEAR)
    return np.asarray(im).astype(np.float32), s


# (photo, list of tap prompts as (u, v, label) triples)
SAM_CASES = {
    "astro_person": (f"{EMU}/astro.jpg", [0.40, 0.25, 1, 0.40, 0.70, 1]),
    "astro_helmet": (f"{EMU}/astro.jpg", [0.80, 0.85, 1]),
    "chelsea_cat": (f"{SK}/chelsea.png", [0.45, 0.55, 1]),
    "coffee_cup": (f"{SK}/coffee.png", [0.45, 0.45, 1]),
    "motorcycle": (f"{SK}/motorcycle_left.png", [0.50, 0.55, 1]),
    "rocket": (f"{SK}/rocket.jpg", [0.48, 0.40, 1]),
    "two_right_person": (f"{SAMPLES}/real_two.jpg", [0.85, 0.25, 1, 0.85, 0.75, 1, 0.15, 0.4, 0]),
}
SAM_CAL_EXTRA = [f"{SK}/astronaut.png", f"{SAMPLES}/real_Owl.jpg", f"{SAMPLES}/real_Parrot.jpg", f"{SK}/hubble_deep_field.jpg",
                 f"{SK}/grass.png", f"{SK}/camera.png", f"{EMU}/sheet.jpg", f"{SK}/ihc.png"]


def sam_prompts(points, w, h, s):
    xs, ys, ls = [], [], []
    for i in range(0, len(points) - 2, 3):
        xs.append(points[i] * w * s); ys.append(points[i + 1] * h * s); ls.append(points[i + 2])
    xs.append(0.0); ys.append(0.0); ls.append(-1.0)
    return np.stack([xs, ys], -1)[None].astype(np.float32), np.array(ls, np.float32)[None]


def sam_dec_feeds(emb, points, w, h, s):
    c, l = sam_prompts(points, w, h, s)
    return {"image_embeddings": emb, "point_coords": c, "point_labels": l,
            "mask_input": np.zeros((1, 1, 256, 256), np.float32), "has_mask_input": np.zeros(1, np.float32),
            "orig_im_size": np.array([h, w], np.float32)}


def mask_from_low_res(low, w, h, max_side=1024):
    """SamMath.maskFromLowRes -> soft alpha 0..1 at the photo size (long side <= 1024)."""
    low = low.reshape(256, 256)
    k = min(1.0, max_side / max(w, h))
    mw, mh = max(1, round(w * k)), max(1, round(h * k))
    s = 1024 / max(w, h); rw, rh = max(1, round(w * s)), max(1, round(h * s))
    gy = np.clip((np.arange(mh) + 0.5) * rh / mh / 4 - 0.5, 0, 255)
    gx = np.clip((np.arange(mw) + 0.5) * rw / mw / 4 - 0.5, 0, 255)
    y0 = gy.astype(int); x0 = gx.astype(int); y1 = np.minimum(255, y0 + 1); x1 = np.minimum(255, x0 + 1)
    ty = (gy - y0)[:, None]; tx = (gx - x0)[None, :]
    l = (low[y0][:, x0] * (1 - tx) + low[y0][:, x1] * tx) * (1 - ty) + (low[y1][:, x0] * (1 - tx) + low[y1][:, x1] * tx) * ty
    return np.clip(0.5 + l / 4, 0, 1)


# ---------------------------------------------------------------- DDColor (LabMath.greyTensor)
def ddcolor_input(path):
    rgb = load_rgb(path)
    small = np.asarray(Image.fromarray(rgb).resize((512, 512), Image.BILINEAR)).astype(np.float64) / 255
    L = rgb2lab(small)[..., :1]
    grey = lab2rgb(np.concatenate([L, np.zeros_like(L), np.zeros_like(L)], -1))
    grey = np.round(grey * 255) / 255  # the app goes through an 8-bit ARGB int
    return {"input": grey.transpose(2, 0, 1)[None].astype(np.float32)}, rgb


DD_EVAL = [f"{ROOT}/inputs/gray_in.bmp", f"{EMU}/lumen_gray.jpg", f"{SK}/camera.png", f"{SK}/chelsea.png", f"{SK}/coffee.png", f"{SK}/rocket.jpg"]
DD_CAL = DD_EVAL + [f"{SK}/astronaut.png", f"{SK}/motorcycle_left.png", f"{SAMPLES}/real_Owl.jpg", f"{SAMPLES}/real_Parrot.jpg",
                    f"{SAMPLES}/real_two.jpg", f"{SK}/hubble_deep_field.jpg", f"{SK}/grass.png", f"{SK}/retina.jpg"]


def ddcolor_feeds(cal=False):
    return [(os.path.basename(p).split(".")[0], ddcolor_input(p)[0]) for p in (DD_CAL if cal else DD_EVAL)]

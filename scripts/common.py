"""Shared helpers for the int8 quantization study (image IO, Lab maths copied from try_models.py, metrics, timing)."""
import os, time, hashlib
import numpy as np
from PIL import Image

ROOT = os.environ.get("INT8_ROOT", os.path.abspath("work"))
PACKS = os.environ.get("MODEL_PACKS", os.path.abspath("originals"))  # original fp32/fp16 models
SCRATCH = os.environ.get("TEST_IMAGES", os.path.abspath("images"))  # test photos
EMU = f"{SCRATCH}/emu"
SAMPLES = os.environ.get("SAMPLE_IMAGES", os.path.abspath("images/samples"))
MODELS = f"{ROOT}/models"
REPORT = f"{ROOT}/report"

ORIG = {
    "gfpgan": f"{PACKS}/GFPGANv1.4.onnx",
    "migan": f"{PACKS}/migan_pipeline_v2.onnx",
    "modnet": f"{PACKS}/modnet.onnx",
    "sam_enc": f"{PACKS}/mobile_sam_encoder.onnx",
    "sam_dec": f"{PACKS}/mobile_sam_decoder.onnx",
    "ddcolor": f"{SCRATCH}/ddcolor_fp32_test.onnx",
}

# Photos used for evaluation / calibration (all real photos available on this Mac).
PHOTOS = [
    f"{EMU}/astro.jpg", f"{EMU}/lumen_bystander.jpg", f"{EMU}/lumen_gray.jpg", f"{EMU}/lumen_redeye.jpg",
    f"{EMU}/devsrc.jpg", f"{EMU}/export.jpg", f"{EMU}/sheet.jpg", f"{EMU}/lumen_text.jpg", f"{EMU}/home.png",
    f"{SAMPLES}/real_Owl.jpg", f"{SAMPLES}/real_Parrot.jpg", f"{SAMPLES}/real_astronaut.jpg", f"{SAMPLES}/real_two.jpg",
    f"{SAMPLES}/lumen_sample_land.jpg", f"{SAMPLES}/lumen_sample_lights.jpg", f"{SAMPLES}/lumen_sample_port.jpg",
    f"{SAMPLES}/lumen_sample_square.jpg",
]


def load_rgb(p):
    """uint8 HxWx3 RGB."""
    return np.asarray(Image.open(p).convert("RGB"))


def save_png(p, rgb_u8):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    Image.fromarray(rgb_u8).save(p)


def resize_u8(rgb, w, h, bilinear=True):
    return np.asarray(Image.fromarray(rgb).resize((w, h), Image.BILINEAR if bilinear else Image.NEAREST))


def side_by_side(a, b, path, gain=4):
    """original | int8 | |diff| x gain, all uint8 HxWx3 (or HxW)."""
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, -1); b = np.repeat(b[..., None], 3, -1)
    d = np.clip(np.abs(a.astype(np.int32) - b.astype(np.int32)) * gain, 0, 255).astype(np.uint8)
    sep = np.full((a.shape[0], 6, 3), 255, np.uint8)
    save_png(path, np.concatenate([a, sep, b, sep, d], 1))


# sRGB <-> Lab (D65), OpenCV float convention 
def srgb2lin(c): return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
def lin2srgb(c): return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.clip(c, 0, None), 1 / 2.4) - 0.055)
MX = np.array([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]])
WP = np.array([0.950456, 1.0, 1.088754])


def rgb2lab(rgb):
    xyz = srgb2lin(rgb) @ MX.T / WP
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab2rgb(lab):
    fy = (lab[..., 0] + 16) / 116; fx = fy + lab[..., 1] / 500; fz = fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], -1)
    xyz = np.where(f > 0.206893, f ** 3, (f - 16 / 116) / 7.787) * WP
    return np.clip(lin2srgb(xyz @ np.linalg.inv(MX).T), 0, 1)


def psnr(a, b, peak=255.0, mask=None):
    a = a.astype(np.float64); b = b.astype(np.float64)
    d = (a - b) ** 2
    if mask is not None:
        d = d[mask]
    mse = d.mean()
    return float("inf") if mse == 0 else 10 * np.log10(peak * peak / mse)


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else float(np.logical_and(a, b).sum() / u)


def sess(path, threads=4):
    import onnxruntime as ort
    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    o.inter_op_num_threads = 1
    return ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])


def median_time(s, feeds, outs=None, n=5):
    s.run(outs, feeds)  # warm-up
    ts = []
    for _ in range(n):
        t = time.perf_counter(); s.run(outs, feeds); ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

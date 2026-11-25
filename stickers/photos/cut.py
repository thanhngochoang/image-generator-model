"""Cut the subject out of each source photo with ISNet (DIS, general use; Apache-2.0), run here only to make assets.
Mask: model at 1024², min-max normalised like rembg, resized back, softened edge -> cut/<name>.png (RGBA, trimmed)."""
import sys, os
import numpy as np, onnxruntime as ort
from PIL import Image, ImageFilter
MODEL = sys.argv[1]
sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
os.makedirs("cut", exist_ok=True)
for p in sys.argv[2:]:
    im = Image.open(p).convert("RGB")
    x = np.asarray(im.resize((1024, 1024), Image.BILINEAR), np.float32) / 255.0
    x = (x - 0.5) / 1.0
    out = sess.run(None, {sess.get_inputs()[0].name: x.transpose(2, 0, 1)[None]})[0][0, 0]
    out = (out - out.min()) / max(1e-6, out.max() - out.min())
    m = Image.fromarray((out * 255).astype(np.uint8)).resize(im.size, Image.BILINEAR).filter(ImageFilter.GaussianBlur(0.8))
    rgba = im.copy(); rgba.putalpha(m)
    bbox = m.point(lambda v: 255 if v > 20 else 0).getbbox()
    if bbox: rgba = rgba.crop(bbox)
    rgba.save("cut/" + os.path.basename(p).rsplit(".", 1)[0] + ".png")
    print("cut", p, rgba.size)

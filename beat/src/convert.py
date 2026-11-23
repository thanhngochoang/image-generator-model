"""Beat This! small0 (CPJKU, MIT) -> ONNX, checked against PyTorch; writes reference data for the app's Kotlin tests."""
import json, math, sys
import numpy as np, torch, torchaudio
from beat_this.inference import load_model, split_predict_aggregate
from beat_this.preprocessing import LogMelSpect
from beat_this.model.postprocessor import Postprocessor
import onnxruntime as ort

SR = 22050
torch.manual_seed(0)
model = load_model("small0.ckpt", "cpu")

# Synthetic test music: 120 BPM kick + hi-hat + a bass tone, 4/4 with an accented downbeat, 20 s.
def music(bpm=120.0, secs=20.0):
    t = np.arange(int(secs * SR)) / SR
    y = 0.15 * np.sin(2 * np.pi * 55 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.25 * t))
    beat = 60.0 / bpm
    rng = np.random.default_rng(1)
    for k in range(int(secs / beat)):
        s = int(k * beat * SR)
        n = np.arange(int(0.25 * SR))
        kick = np.sin(2 * np.pi * (120 * np.exp(-n / 2000.0)) * n / SR) * np.exp(-n / 3000.0)
        amp = 1.0 if k % 4 == 0 else 0.6
        y[s:s + len(n)] += amp * kick[: len(y) - s]
        hs = s + int(beat * SR / 2)
        hn = np.arange(int(0.05 * SR))
        hat = rng.standard_normal(len(hn)) * np.exp(-hn / 300.0) * 0.2
        y[hs:hs + len(hn)] += hat[: max(0, len(y) - hs)]
    return (y / np.abs(y).max() * 0.8).astype(np.float32)

wave = music()
spect = LogMelSpect()(torch.from_numpy(wave.astype(np.float64)).float())
print("spect", tuple(spect.shape))

class Wrap(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x):
        o = self.m(x)
        return o["beat"], o["downbeat"]

w = Wrap(model).eval()
dummy = spect[:1500].unsqueeze(0)
torch.onnx.export(w, (dummy,), "beat_this_small0.onnx", input_names=["spect"], output_names=["beat", "downbeat"],
                  dynamic_axes={"spect": {1: "frames"}, "beat": {1: "frames"}, "downbeat": {1: "frames"}}, opset_version=17, dynamo=False)

sess = ort.InferenceSession("beat_this_small0.onnx")
with torch.inference_mode():
    ref = split_predict_aggregate(spect, 1500, 6, "keep_first", model)
# ONNX with the same chunking
def onnx_model(x):
    b, d = sess.run(None, {"spect": x.numpy()})
    return {"beat": torch.from_numpy(b), "downbeat": torch.from_numpy(d)}
got = split_predict_aggregate(spect, 1500, 6, "keep_first", onnx_model)
diff = max((ref["beat"] - got["beat"]).abs().max().item(), (ref["downbeat"] - got["downbeat"]).abs().max().item())
print("max |logit diff| onnx vs torch:", diff)
pp = Postprocessor("minimal")
beats, downs = pp(ref["beat"], ref["downbeat"])
print("beats", len(beats), "first", np.round(beats[:6], 3), "downbeats", np.round(downs[:4], 3))
ibi = np.diff(beats)
print("median interval", np.median(ibi), "-> bpm", 60 / np.median(ibi))

# Reference for Kotlin tests: a short 2 s 440 Hz + click wave (deterministic), its log-mel, and the full logits+beats.
t = np.arange(int(2.0 * SR)) / SR
small = (0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 1250 * t) * (t % 0.5 < 0.02)).astype(np.float32)
smel = LogMelSpect()(torch.from_numpy(small.astype(np.float64)).float()).numpy()
json.dump({"sr": SR, "wave": [round(float(v), 6) for v in small], "mel_frames": smel.shape[0],
           "mel_rows": {str(i): [round(float(v), 5) for v in smel[i]] for i in (0, 1, 50, smel.shape[0] - 1)},
           "logits_beat": [round(float(v), 4) for v in ref["beat"].numpy()],
           "logits_downbeat": [round(float(v), 4) for v in ref["downbeat"].numpy()],
           "beats": [round(float(v), 4) for v in beats], "downbeats": [round(float(v), 4) for v in downs]},
          open("reference.json", "w"))
import os; print("onnx bytes", os.path.getsize("beat_this_small0.onnx"))

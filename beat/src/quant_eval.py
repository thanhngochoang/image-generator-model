"""Int8 weights for Beat This! small0, gated against the fp32 ONNX on varied synthetic music.

Pass (fixed before measuring): on EVERY piece, int8 beats and downbeats match fp32 within +-20 ms (one frame)
with F-measure >= 0.98, and the same median BPM.
"""
import os, sys, json
import numpy as np, torch
import onnxruntime as ort
from beat_this.inference import split_predict_aggregate
from beat_this.preprocessing import LogMelSpect
from beat_this.model.postprocessor import Postprocessor

SKILL = sys.argv[1]  # quantize-onnx-int8/scripts
os.environ["INT8_ROOT"] = os.path.abspath("int8")
os.makedirs("int8/models", exist_ok=True)
sys.path.insert(0, SKILL)
import common, quantize  # noqa: E402
common.ORIG["beat"] = quantize.ORIG["beat"] = os.path.abspath("beat_this_small0.onnx")
quantize.MODELS = common.MODELS

SR = 22050
rng = np.random.default_rng(7)


def kick(n_s=0.25):
    n = np.arange(int(n_s * SR))
    return np.sin(2 * np.pi * (120 * np.exp(-n / 2000.0)) * n / SR) * np.exp(-n / 3000.0)


def hat(n_s=0.05):
    n = np.arange(int(n_s * SR))
    return rng.standard_normal(len(n)) * np.exp(-n / 300.0)


def tone(f, n_s, decay):
    n = np.arange(int(n_s * SR))
    return (np.sin(2 * np.pi * f * n / SR) + 0.4 * np.sin(4 * np.pi * f * n / SR)) * np.exp(-n / (decay * SR))


def add(y, s, x, a):
    s = int(s)
    if s >= len(y): return
    m = min(len(x), len(y) - s)
    y[s:s + m] += a * x[:m]


def piece(times, meter=4, secs=45.0, style="kick", noise=0.0):
    y = np.zeros(int(secs * SR))
    chords = [220, 261.6, 196, 246.9]
    for k, t in enumerate(times):
        if t * SR >= len(y): break
        down = k % meter == 0
        if style == "kick":
            add(y, t * SR, kick(), 1.0 if down else 0.6)
        elif style == "piano":
            add(y, t * SR, tone(chords[(k // meter) % 4] * (1 if down else 1.5), 0.6, 0.25), 0.7 if down else 0.4)
        nxt = times[k + 1] if k + 1 < len(times) else t + 0.5
        add(y, (t + (nxt - t) * (0.62 if style == "swing" else 0.5)) * SR, hat(), 0.2)
        if style == "swing":
            add(y, t * SR, kick() if down or k % meter == 2 else tone(98, 0.3, 0.1), 0.8)
        if style == "sync":  # four on the floor plus off-beat bass
            add(y, (t + (nxt - t) * 0.5) * SR, tone(55, 0.2, 0.08), 0.5)
    t = np.arange(len(y)) / SR
    y += 0.08 * np.sin(2 * np.pi * 110 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t))
    y += noise * rng.standard_normal(len(y))
    return (y / np.abs(y).max() * 0.8).astype(np.float32)


def grid(bpm, secs=45.0, start=0.3):
    return list(start + np.arange(0, secs * bpm / 60.0) * 60.0 / bpm)


def ramp(b0, b1, secs=45.0):
    t, out = 0.3, []
    while t < secs:
        out.append(t); t += 60.0 / (b0 + (b1 - b0) * t / secs)
    return out


PIECES = {
    "120_kick": piece(grid(120)),
    "90_swing": piece(grid(90), style="swing"),
    "140_sync": piece(grid(140), style="sync"),
    "75_piano": piece(grid(75), style="piano"),
    "ramp_100_130": piece(ramp(100, 130)),
    "waltz_150": piece(grid(150), meter=3),
    "128_noisy": piece(grid(128), noise=0.15),
}

pp = Postprocessor("minimal")


def run(path, wave):
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    spect = LogMelSpect()(torch.from_numpy(wave.astype(np.float64)).float())

    def m(x):
        b, d = s.run(None, {"spect": x.numpy()})
        return {"beat": torch.from_numpy(b), "downbeat": torch.from_numpy(d)}

    out = split_predict_aggregate(spect, 1500, 6, "keep_first", m)
    beats, downs = pp(out["beat"], out["downbeat"])
    return out, np.asarray(beats), np.asarray(downs)


def fmeasure(ref, est, tol=0.02):
    if len(ref) == 0 and len(est) == 0: return 1.0
    used, hit = set(), 0
    for r in ref:
        j = np.argmin(np.abs(est - r)) if len(est) else None
        if j is not None and j not in used and abs(est[j] - r) <= tol:
            used.add(j); hit += 1
    p = hit / max(len(est), 1); rc = hit / max(len(ref), 1)
    return 0.0 if hit == 0 else 2 * p * rc / (p + rc)


def bpm(b):
    return round(60 / float(np.median(np.diff(b)))) if len(b) > 1 else None


def evaluate(qpath):
    rows, ok = [], True
    for name, w in PIECES.items():
        o32, b32, d32 = run("beat_this_small0.onnx", w)
        o8, b8, d8 = run(qpath, w)
        logit = max(float((o32["beat"] - o8["beat"]).abs().max()), float((o32["downbeat"] - o8["downbeat"]).abs().max()))
        fb, fd = fmeasure(b32, b8), fmeasure(d32, d8)
        same = bpm(b32) == bpm(b8)
        good = fb >= 0.98 and fd >= 0.98 and same
        ok &= good
        rows.append(dict(piece=name, beats=len(b32), bpm32=bpm(b32), bpm8=bpm(b8), f_beat=round(fb, 4), f_down=round(fd, 4),
                         max_logit_diff=round(logit, 4), passed=good))
        print(rows[-1], flush=True)
    return ok, rows


if __name__ == "__main__":
    method = sys.argv[2] if len(sys.argv) > 2 else "wo"
    min_elems = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    q = quantize.weight_only("beat", min_elems=min_elems, block=32 if method == "wob" else 0)
    method = f"{method}_{min_elems}"
    print("int8 file", q, os.path.getsize(q), "B")
    ok, rows = evaluate(q)
    json.dump({"method": method, "bytes": os.path.getsize(q), "passed": ok, "rows": rows}, open(f"int8/results_{method}.json", "w"), indent=1)
    print("PASS" if ok else "FAIL")

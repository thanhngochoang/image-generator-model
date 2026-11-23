"""Gate int8 vs fp32 on real music: five 60 s excerpts from each MP3 in a folder (usage: real_eval.py SKILL_SCRIPTS MUSIC_DIR)."""
import sys, glob, os, numpy as np, av
MUSIC_DIR = sys.argv[2]
sys.argv = [sys.argv[0], sys.argv[1]]
import quant_eval as q

def excerpt(path, start, secs=60.0, sr=q.SR):
    c = av.open(path)
    s = c.streams.audio[0]
    c.seek(int(start / s.time_base), stream=s)
    r = av.AudioResampler(format="flt", layout="mono", rate=sr)
    out, n = [], 0
    for fr in c.decode(s):
        for x in r.resample(fr):
            a = x.to_ndarray().reshape(-1); out.append(a); n += len(a)
        if n >= secs * sr: break
    c.close()
    return np.concatenate(out)[: int(secs * sr)].astype(np.float32)

files = sorted(glob.glob(os.path.join(MUSIC_DIR, "*.mp3")))
pieces = {}
for fi, f in enumerate(files):
    dur = av.open(f).duration / 1e6
    for k in range(5):
        start = dur * (k + 0.5) / 5
        pieces[f"file{fi + 1}@{int(start // 60)}m"] = excerpt(f, start)
q.PIECES = pieces
print("excerpts", len(pieces), "x 60 s")
for name in ("beat_this_small0_int8w.onnx",):
    print("===", name, os.path.getsize(name), "B")
    ok, rows = q.evaluate(name)
    print("PASS" if ok else "FAIL")

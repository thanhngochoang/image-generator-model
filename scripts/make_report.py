"""Summarise report/results_*.json: choose per model the smallest passing int8 candidate, copy it to final/<app file name>,
print markdown tables (summary + every candidate) and SHA-256 lines for REPORT.md."""
import os, sys, json, shutil
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, REPORT, MODELS, ORIG, sha256

APP_NAME = {"gfpgan": "GFPGANv1.4.onnx", "migan": "migan_pipeline_v2.onnx", "modnet": "modnet.onnx",
            "sam_enc": "mobile_sam_encoder.onnx", "sam_dec": "mobile_sam_decoder.onnx", "ddcolor": "ddcolor_tiny_fp16.onnx"}
LABEL = {"gfpgan": "GFPGAN v1.4", "migan": "MI-GAN pipeline v2", "modnet": "MODNet", "sam_enc": "MobileSAM encoder",
         "sam_dec": "MobileSAM decoder", "ddcolor": "DDColor tiny (vs fp32)"}
METHOD = {"dyn": "dynamic (MatMul/Gemm)", "dynconv": "dynamic (+Conv)", "dynconv_pt": "dynamic (+Conv, per-tensor)",
          "wo": "weight-only int8, per-channel", "wob": "weight-only int8, block 32", "qdq": "static QDQ, MinMax",
          "qdqp": "static QDQ, Percentile", "f16w": "fp16 weights (reference, not int8)", "fp16_shipped": "shipped fp16 (reference)"}
NOT_INT8 = {"f16w", "fp16_shipped"}
MB = lambda b: f"{b / 1e6:.1f} MB"


def quality(model, r):
    rows = r["rows"]
    if model in ("gfpgan", "migan"):
        p = min(x["psnr_db"] for x in rows); m = max(x["mad_levels"] for x in rows)
        where = "face" if model == "gfpgan" else "hole"
        return f"{where} PSNR min {p:.1f} dB, MAD max {m:.2f}/255"
    if model == "ddcolor":
        return (f"mean dAB max {max(x['dab_mean'] for x in rows):.2f} (p99 {max(x['dab_p99'] for x in rows):.1f}); "
                f"chroma gray_in {rows[0]['chroma_ref']:.1f} -> {rows[0]['chroma_int8']:.1f}")
    return f"IoU min {min(x['iou'] for x in rows):.4f}, alpha MAD max {max(x['mad_alpha'] for x in rows):.4f}"


def main():
    res = {k: json.load(open(f"{REPORT}/results_{k}.json")) for k in APP_NAME if os.path.exists(f"{REPORT}/results_{k}.json")}
    pair = json.load(open(f"{REPORT}/results_sam_pair.json")) if os.path.exists(f"{REPORT}/results_sam_pair.json") else {}
    os.makedirs(f"{ROOT}/final", exist_ok=True)
    summary, full, shas, chosen = [], [], [], {}
    for k, r in sorted(res.items(), key=lambda kv: kv[0] == "sam_enc"):  # decoder first, see the pair rule
        o = r["original"]
        cands = {m: c for m, c in r.items() if m != "original"}
        ok = [(c["size"], m) for m, c in cands.items() if c.get("pass") and m not in NOT_INT8]
        for m, c in sorted(cands.items(), key=lambda x: x[1].get("size", 0)):
            if "rows" in c:
                full.append(f"| {LABEL[k]} | {m} | {METHOD.get(m, m)} | {MB(c['size'])} ({100 * c['size'] / o['size']:.0f} %) | {quality(k, c)} | "
                            f"{c['time_s'] * 1000:.0f} ms vs {o['time_s'] * 1000:.0f} ms | {'yes' if c['same_io'] else 'NO'} | {'pass' if c['pass'] else 'fail'} |")
            else:
                full.append(f"| {LABEL[k]} | {m} | {METHOD.get(m, m)} | {MB(c['size'])} | does not load: {c['error'][:120]}… | – | – | fail |")
        if ok and k == "sam_enc" and "sam_dec" in chosen:
            # encoder and decoder run together: keep only encoders whose pair with the chosen decoder passes
            ok = [(sz, m) for sz, m in ok if pair.get(f"{m}+{chosen['sam_dec']}", {}).get("pass")]
        if ok:
            m = min(ok)[1]; c = cands[m]; chosen[k] = m
            dst = f"{ROOT}/final/{APP_NAME[k]}"
            shutil.copyfile(f"{MODELS}/{k}.{m}.onnx", dst)
            summary.append(f"| {LABEL[k]} | {MB(o['size'])} | {MB(c['size'])} ({100 * c['size'] / o['size']:.0f} %) | {METHOD[m]} | {quality(k, c)} | "
                           f"{o['time_s'] * 1000:.0f} → {c['time_s'] * 1000:.0f} ms | **pass** → `final/{APP_NAME[k]}` |")
        else:
            # report the best-quality int8 candidate that loads
            loads = {m: c for m, c in cands.items() if "rows" in c and m not in NOT_INT8}
            def score(c):
                rows = c["rows"]
                if k in ("gfpgan", "migan"): return min(x["psnr_db"] for x in rows)
                if k == "ddcolor": return -max(x["dab_mean"] for x in rows)
                return min(x["iou"] for x in rows)
            m = max(loads, key=lambda m: score(loads[m])); c = loads[m]
            summary.append(f"| {LABEL[k]} | {MB(o['size'])} | best: {MB(c['size'])} ({100 * c['size'] / o['size']:.0f} %) | {METHOD[m]} | {quality(k, c)} | "
                           f"{o['time_s'] * 1000:.0f} → {c['time_s'] * 1000:.0f} ms | **fail** – keep original |")
    print("## Summary\n")
    print("| Model | Original | int8 | Method | Quality vs original (worst input) | CPU median, 4 thr | Verdict |")
    print("|---|---|---|---|---|---|---|")
    print("\n".join(summary))
    if pair:
        print("\nMobileSAM encoder + decoder together (both int8, vs both original):\n")
        for t, p in pair.items():
            print(f"- `{t}`: IoU min {min(x['iou'] for x in p['rows']):.4f}, alpha MAD max {max(x['mad_alpha'] for x in p['rows']):.4f} → {'pass' if p['pass'] else 'fail'}")
    print("\n## Every candidate\n")
    print("| Model | Candidate | Method | Size | Quality (worst input) | Time int8 vs orig | Same IO | Verdict |")
    print("|---|---|---|---|---|---|---|---|")
    print("\n".join(full))
    print("\n## SHA-256\n")
    print("Files in `final/` (passing models under the app's file name):\n")
    for k, m in chosen.items():
        p = f"{ROOT}/final/{APP_NAME[k]}"
        print(f"- `final/{APP_NAME[k]}` ({os.path.getsize(p)} B, from `models/{k}.{m}.onnx`): `{sha256(p)}`")
    print("\nAll candidates in `models/`:\n")
    for f in sorted(os.listdir(MODELS)):
        if f.endswith(".onnx") and ".pre." not in f:
            print(f"- `models/{f}` ({os.path.getsize(f'{MODELS}/{f}')} B): `{sha256(f'{MODELS}/{f}')}`")
    json.dump(chosen, open(f"{REPORT}/chosen.json", "w"))


if __name__ == "__main__":
    main()

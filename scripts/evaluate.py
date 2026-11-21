"""Compare every int8 candidate of a model with the original: size, IO signature, quality on app-preprocessed inputs,
median CPU time (4 threads, 5 runs after a warm-up). Writes report/results_<model>.json and side-by-side PNGs
(original | int8 | |diff| x4) under report/<model>/.

usage: evaluate.py <model> [method ...]   (default: every models/<model>.<method>.onnx except .pre)
"""
import os, sys, glob, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import onnx
from common import ORIG, PACKS, MODELS, REPORT, sess, median_time, psnr, iou, side_by_side, save_png, lab2rgb, rgb2lab, load_rgb
import feeds as F

PASS = {"psnr_db": 35.0, "mad_levels": 2.0, "iou": 0.98, "dab": 2.0}


def signature(path):
    m = onnx.load(path, load_external_data=False)
    def sig(vs):
        return [(v.name, v.type.tensor_type.elem_type, [d.dim_value if d.dim_value else "?" for d in v.type.tensor_type.shape.dim]) for v in vs]
    return sig(m.graph.input), sig(m.graph.output)


def same_signature(a, b):
    """Names, element types and ranks equal; fixed dims of the original kept (symbolic dims may be renamed)."""
    for x, y in zip(signature(a), signature(b)):
        if len(x) != len(y): return False
        for (n1, t1, s1), (n2, t2, s2) in zip(x, y):
            if n1 != n2 or t1 != t2 or len(s1) != len(s2): return False
            if any(d1 != "?" and d1 != d2 for d1, d2 in zip(s1, s2)): return False
    return True


def candidates(model, methods):
    if methods:
        return {m: f"{MODELS}/{model}.{m}.onnx" for m in methods}
    out = {}
    for p in sorted(glob.glob(f"{MODELS}/{model}.*.onnx")):
        m = os.path.basename(p)[len(model) + 1:-5]
        if m != "pre":
            out[m] = p
    return out


def try_sess(path):
    try:
        return sess(path), None
    except Exception as e:
        return None, repr(e)[:300]


def rgb_u8_from_gfpgan(x):
    return np.round((np.clip(x[0], -1, 1) + 1) * 127.5).astype(np.uint8).transpose(1, 2, 0)


def face_mask():
    cy, cx = np.mgrid[0:512, 0:512] + 0.5
    r = np.sqrt(((cx - 256) / 226) ** 2 + ((cy - 276) / 262) ** 2)
    t = np.clip((1 - r) / 0.22, 0, 1)
    return t * t * (3 - 2 * t) > 0.5  # FaceRestoreMath.maskAt > 0.5: the part of the crop that is pasted back


def image_verdict(rows):
    return all(r["psnr_db"] >= PASS["psnr_db"] or r["mad_levels"] <= PASS["mad_levels"] for r in rows)


# ------------------------------------------------------------------------------------------------------------------
def eval_gfpgan(cands):
    fs = F.gfpgan_feeds()
    fm = face_mask()
    s0 = sess(ORIG["gfpgan"])
    ref = {n: rgb_u8_from_gfpgan(s0.run(["output"], f)[0]) for n, f in fs}
    t0 = median_time(s0, fs[0][1], ["output"])
    del s0
    res = {"original": {"time_s": t0}}
    for m, p in cands.items():
        s, err = try_sess(p)
        if s is None:
            res[m] = {"error": err}; continue
        rows = []
        for n, f in fs:
            o = rgb_u8_from_gfpgan(s.run(["output"], f)[0])
            a = ref[n]
            rows.append({"input": n, "psnr_db": psnr(a, o, mask=fm), "mad_levels": float(np.abs(a.astype(int) - o)[fm].mean()),
                         "psnr_full_db": psnr(a, o), "max_abs": int(np.abs(a.astype(int) - o)[fm].max())})
            side_by_side(a, o, f"{REPORT}/gfpgan/{m}_{n}.png")
        res[m] = {"rows": rows, "time_s": median_time(s, fs[0][1], ["output"]), "pass": image_verdict(rows)}
        del s
    return res


def eval_migan(cands):
    cases = {k: F.migan_case(*v) for k, v in F.MIGAN_CASES.items()}
    s0 = sess(ORIG["migan"])
    ref = {k: s0.run(["result"], f)[0] for k, (f, _) in cases.items()}
    tk = "coffee_cup_medium"
    t0 = median_time(s0, cases[tk][0], ["result"])
    del s0
    res = {"original": {"time_s": t0}}
    for m, p in cands.items():
        s, err = try_sess(p)
        if s is None:
            res[m] = {"error": err}; continue
        rows = []
        for k, (f, hole) in cases.items():
            o = s.run(["result"], f)[0]
            a = ref[k]
            A = a[0].transpose(1, 2, 0); O = o[0].transpose(1, 2, 0)
            d = np.abs(A.astype(int) - O)
            rows.append({"input": k, "hole_px": int(hole.sum()), "psnr_db": psnr(A, O, mask=hole), "mad_levels": float(d[hole].mean()),
                         "outside_hole_max_abs": int(d[~hole].max()) if (~hole).any() else 0, "dtype": str(o.dtype), "shape": list(o.shape)})
            ys, xs = np.nonzero(hole); pad = 24
            y0, y1, x0, x1 = max(0, ys.min() - pad), ys.max() + pad, max(0, xs.min() - pad), xs.max() + pad
            side_by_side(A[y0:y1, x0:x1], O[y0:y1, x0:x1], f"{REPORT}/migan/{m}_{k}.png")
        res[m] = {"rows": rows, "time_s": median_time(s, cases[tk][0], ["result"]), "pass": image_verdict(rows)}
        del s
    return res


def eval_modnet(cands):
    fs = F.modnet_feeds()
    s0 = sess(ORIG["modnet"])
    ref = {n: s0.run(["output"], f)[0][0, 0] for n, f in fs}
    t0 = median_time(s0, fs[0][1], ["output"])
    del s0
    res = {"original": {"time_s": t0}}
    for m, p in cands.items():
        s, err = try_sess(p)
        if s is None:
            res[m] = {"error": err}; continue
        rows = []
        for n, f in fs:
            o = s.run(["output"], f)[0][0, 0]; a = ref[n]
            rows.append({"input": n, "size": list(a.shape), "iou": iou(a >= 0.5, o >= 0.5), "mad_alpha": float(np.abs(a - o).mean()),
                         "max_abs_alpha": float(np.abs(a - o).max())})
            side_by_side((a * 255 + 0.5).astype(np.uint8), (np.clip(o, 0, 1) * 255 + 0.5).astype(np.uint8), f"{REPORT}/modnet/{m}_{n}.png")
        res[m] = {"rows": rows, "time_s": median_time(s, fs[0][1], ["output"]), "pass": all(r["iou"] >= PASS["iou"] for r in rows)}
        del s
    return res


def sam_case_inputs():
    out = {}
    for name, (p, pts) in F.SAM_CASES.items():
        rgb = load_rgb(p); x, s = F.sam_input(rgb)
        out[name] = (rgb, x, s, pts)
    return out


def sam_masks(enc, dec, cases):
    embs, masks = {}, {}
    for name, (rgb, x, s, pts) in cases.items():
        key = id(x)
        if key not in embs:
            embs[key] = enc.run(["image_embeddings"], {"input_image": x})[0]
        low = dec.run(["low_res_masks"], F.sam_dec_feeds(embs[key], pts, rgb.shape[1], rgb.shape[0], s))[0]
        masks[name] = F.mask_from_low_res(low, rgb.shape[1], rgb.shape[0])
    return masks


def eval_sam(enc_cands, dec_cands):
    cases = sam_case_inputs()
    e0 = sess(ORIG["sam_enc"]); d0 = sess(ORIG["sam_dec"])
    ref = sam_masks(e0, d0, cases)
    rgb, x, s, pts = cases["astro_person"]
    emb = e0.run(["image_embeddings"], {"input_image": x})[0]
    dfeed = F.sam_dec_feeds(emb, pts, rgb.shape[1], rgb.shape[0], s)
    res = {"sam_enc": {"original": {"time_s": median_time(e0, {"input_image": x}, ["image_embeddings"])}},
           "sam_dec": {"original": {"time_s": median_time(d0, dfeed, ["low_res_masks"])}}}

    def rows_for(masks, tag):
        rows = []
        for name, a in ref.items():
            o = masks[name]
            rows.append({"input": name, "iou": iou(a >= 0.5, o >= 0.5), "mad_alpha": float(np.abs(a - o).mean())})
            side_by_side((a * 255 + 0.5).astype(np.uint8), (o * 255 + 0.5).astype(np.uint8), f"{REPORT}/sam/{tag}_{name}.png")
        return rows

    for m, p in enc_cands.items():
        e, err = try_sess(p)
        if e is None:
            res["sam_enc"][m] = {"error": err}; continue
        rows = rows_for(sam_masks(e, d0, cases), f"enc-{m}")
        res["sam_enc"][m] = {"rows": rows, "time_s": median_time(e, {"input_image": x}, ["image_embeddings"]),
                             "pass": all(r["iou"] >= PASS["iou"] for r in rows), "note": "with the original decoder"}
        del e
    for m, p in dec_cands.items():
        d, err = try_sess(p)
        if d is None:
            res["sam_dec"][m] = {"error": err}; continue
        rows = rows_for(sam_masks(e0, d, cases), f"dec-{m}")
        res["sam_dec"][m] = {"rows": rows, "time_s": median_time(d, dfeed, ["low_res_masks"]),
                             "pass": all(r["iou"] >= PASS["iou"] for r in rows), "note": "with the original encoder"}
        del d
    return res


def eval_sam_pair(enc_path, dec_path, tag):
    cases = sam_case_inputs()
    ref = sam_masks(sess(ORIG["sam_enc"]), sess(ORIG["sam_dec"]), cases)
    got = sam_masks(sess(enc_path), sess(dec_path), cases)
    rows = []
    for name, a in ref.items():
        o = got[name]
        rows.append({"input": name, "iou": iou(a >= 0.5, o >= 0.5), "mad_alpha": float(np.abs(a - o).mean())})
        side_by_side((a * 255 + 0.5).astype(np.uint8), (o * 255 + 0.5).astype(np.uint8), f"{REPORT}/sam/pair-{tag}_{name}.png")
    return {"rows": rows, "pass": all(r["iou"] >= PASS["iou"] for r in rows)}


def eval_ddcolor(cands):
    fs = F.ddcolor_feeds()
    s0 = sess(ORIG["ddcolor"])
    ref = {n: s0.run(None, f)[0][0].transpose(1, 2, 0) for n, f in fs}
    t0 = median_time(s0, fs[0][1])
    del s0
    res = {"original": {"time_s": t0, "chroma": {n: float(np.hypot(a[..., 0], a[..., 1]).mean()) for n, a in ref.items()}}}
    for m, p in cands.items():
        s, err = try_sess(p)
        if s is None:
            res[m] = {"error": err}; continue
        rows = []
        for n, f in fs:
            o = s.run(None, f)[0][0].transpose(1, 2, 0); a = ref[n]
            dab = np.hypot(*(a - o).transpose(2, 0, 1))
            rows.append({"input": n, "dab_mean": float(dab.mean()), "dab_p99": float(np.percentile(dab, 99)),
                         "chroma_ref": float(np.hypot(a[..., 0], a[..., 1]).mean()), "chroma_int8": float(np.hypot(o[..., 0], o[..., 1]).mean())})
            L = rgb2lab(f["input"][0].transpose(1, 2, 0).astype(np.float64))[..., :1]
            ca = (lab2rgb(np.concatenate([L, a], -1)) * 255 + 0.5).astype(np.uint8)
            co = (lab2rgb(np.concatenate([L, o], -1)) * 255 + 0.5).astype(np.uint8)
            side_by_side(ca, co, f"{REPORT}/ddcolor/{m}_{n}.png")
        res[m] = {"rows": rows, "time_s": median_time(s, fs[0][1]), "pass": all(r["dab_mean"] <= PASS["dab"] for r in rows)}
        del s
    return res


def main():
    model, methods = sys.argv[1], sys.argv[2:]
    if model == "sam":
        enc = candidates("sam_enc", methods); dec = candidates("sam_dec", methods)
        res = eval_sam(enc, dec)
        for k, cs in (("sam_enc", enc), ("sam_dec", dec)):
            for m, p in cs.items():
                res[k][m]["size"] = os.path.getsize(p); res[k][m]["same_io"] = same_signature(ORIG[k], p)
            res[k]["original"]["size"] = os.path.getsize(ORIG[k])
        for k in ("sam_enc", "sam_dec"):
            merge(k, res[k])
        return
    if model == "sam_pair":
        enc, dec = methods
        r = eval_sam_pair(f"{MODELS}/sam_enc.{enc}.onnx", f"{MODELS}/sam_dec.{dec}.onnx", f"{enc}+{dec}")
        merge("sam_pair", {f"{enc}+{dec}": r}); return
    cs = candidates(model, methods)
    if model == "ddcolor":
        cs.setdefault("fp16_shipped", os.path.join(PACKS, "ddcolor-tiny-fp16.onnx"))
    res = {"gfpgan": eval_gfpgan, "migan": eval_migan, "modnet": eval_modnet, "ddcolor": eval_ddcolor}[model](cs)
    res["original"]["size"] = os.path.getsize(ORIG[model])
    for m, p in cs.items():
        res[m]["size"] = os.path.getsize(p); res[m]["same_io"] = same_signature(ORIG[model], p)
    merge(model, res)


def merge(model, res):
    path = f"{REPORT}/results_{model}.json"
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(res)
    json.dump(old, open(path, "w"), indent=1)
    for m, r in res.items():
        if "rows" in r:
            keys = [k for k in r["rows"][0] if k not in ("input", "size", "dtype", "shape")]
            agg = {k: (min if k in ("psnr_db", "iou", "psnr_full_db") else max)(x[k] for x in r["rows"]) for k in keys if isinstance(r["rows"][0][k], (int, float))}
            print(f"{model:8s} {m:14s} size={r.get('size')} time={r.get('time_s', 0):.3f}s pass={r.get('pass')} io={r.get('same_io')} worst={ {k: round(v, 4) for k, v in agg.items()} }")
        else:
            print(f"{model:8s} {m:14s} {r}")


if __name__ == "__main__":
    main()

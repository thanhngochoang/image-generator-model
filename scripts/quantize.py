"""Make int8 candidates of one model.

usage: quantize.py <model> <method> [...]
  model : gfpgan | migan | modnet | sam_enc | sam_dec | ddcolor
  method: dyn      quantize_dynamic, per-channel QInt8 weights, MatMul/Gemm only
          dynconv  quantize_dynamic, per-channel QInt8 weights, MatMul/Gemm + Conv (ConvInteger)
          wo       weight-only int8: every float initializer with >= 1024 elements stored as int8 + per-channel fp32
                   scale, dequantised in-graph by Cast -> Mul (ORT constant-folds it at load, so compute stays fp32)
          wob      as wo but one scale per block of 32 weights along the reduction dims (finer, ~+12 % size)
          f16w     reference only (not int8): weights stored fp16, cast to fp32 in-graph
          dynconv_pt  as dynconv but per-tensor weights
          qdq      quantize_static, QDQ format, U8 activations / S8 per-channel weights, MinMax calibration on images
                   built with the app's pre-processing
          qdqp     as qdq but Percentile (99.999) calibration
Writes models/<model>.<method>.onnx (and models/<model>.pre.onnx, the quant_pre_process output).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import onnx
from onnx import numpy_helper, helper, TensorProto
from common import ORIG, MODELS

os.makedirs(MODELS, exist_ok=True)


def pre(model):
    """quant_pre_process (symbolic shape inference + ORT basic optimisation + onnx shape inference); falls back
    to skipping symbolic shape inference when it fails."""
    from onnxruntime.quantization.shape_inference import quant_pre_process
    out = f"{MODELS}/{model}.pre.onnx"
    if os.path.exists(out):
        return out
    try:
        quant_pre_process(ORIG[model], out, skip_symbolic_shape=False)
        print("pre-process: full")
    except Exception as e:
        print("pre-process: symbolic shape inference failed:", (repr(e)[:200]), "-> retry without")
        quant_pre_process(ORIG[model], out, skip_symbolic_shape=True)
    m = onnx.load(out)
    if next(o.version for o in m.opset_import if o.domain in ("", "ai.onnx")) < 13:
        # per-channel DequantizeLinear (axis attribute) needs opset 13 (MODNet is opset 11)
        from onnx import version_converter
        onnx.save(version_converter.convert_version(m, 13), out)
        print("pre-process: opset converted to 13")
    return out


def weight_only(model, min_elems=1024, block=0):
    m = onnx.load(ORIG[model])
    g = m.graph
    # Large Constant nodes (e.g. position tables) become initializers first so they are quantised too.
    for n in list(g.node):
        if n.op_type == "Constant" and n.attribute and n.attribute[0].name == "value" and n.attribute[0].t.data_type == TensorProto.FLOAT:
            a = numpy_helper.to_array(n.attribute[0].t)
            if a.size >= min_elems:
                g.initializer.append(numpy_helper.from_array(a, n.output[0])); g.node.remove(n)
    consumers = {}
    for n in g.node:
        for i, name in enumerate(n.input):
            consumers.setdefault(name, []).append((n, i))
    new_inits, new_nodes, done = [], [], 0
    keep = []
    for t in g.initializer:
        a = numpy_helper.to_array(t)
        if a.dtype != np.float32 or a.size < min_elems:
            keep.append(t); continue
        # Pick the channel axis from the consumer: Conv/ConvTranspose -> output channels; MatMul -> last; Gemm -> by transB.
        axis = None
        for n, i in consumers.get(t.name, []):
            if n.op_type == "Conv" and i == 1: axis = 0
            elif n.op_type == "ConvTranspose" and i == 1: axis = 1
            elif n.op_type == "MatMul" and i == 1: axis = a.ndim - 1
            elif n.op_type == "Gemm" and i == 1:
                tb = next((x.i for x in n.attribute if x.name == "transB"), 0); axis = 0 if tb else 1
        if axis is None:
            axis = next((k for k, d in enumerate(a.shape) if d > 1), 0) if a.ndim >= 2 else None
        qn, sn, cn = t.name + "_q8", t.name + "_q8_scale", t.name + "_q8_f"
        if block and a.ndim >= 2 and axis is not None:
            # Block-wise: channel axis first, the rest flattened (K) and cut into blocks of `block` values, one scale each.
            perm = [axis] + [k for k in range(a.ndim) if k != axis]
            p_shape = [a.shape[k] for k in perm]
            C, K = p_shape[0], int(np.prod(p_shape[1:]))
            nb = -(-K // block); Kp = nb * block
            w = np.zeros((C, Kp), np.float32); w[:, :K] = a.transpose(perm).reshape(C, K)
            w = w.reshape(C, nb, block)
            amax = np.abs(w).max(axis=2, keepdims=True)
            scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
            q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
            names = [t.name + suf for suf in ("_q8_m", "_q8_r", "_q8_s", "_q8_p")]
            new_inits += [numpy_helper.from_array(q, qn), numpy_helper.from_array(scale, sn),
                          numpy_helper.from_array(np.array([C, Kp], np.int64), t.name + "_q8_shape2"),
                          numpy_helper.from_array(np.array(p_shape, np.int64), t.name + "_q8_shapep")]
            new_nodes += [helper.make_node("Cast", [qn], [cn], to=TensorProto.FLOAT, name=cn),
                          helper.make_node("Mul", [cn, sn], [names[0]], name=names[0]),
                          helper.make_node("Reshape", [names[0], t.name + "_q8_shape2"], [names[1]], name=names[1])]
            cur = names[1]
            if Kp != K:
                new_inits += [numpy_helper.from_array(np.array([0], np.int64), t.name + "_q8_st"),
                              numpy_helper.from_array(np.array([K], np.int64), t.name + "_q8_en"),
                              numpy_helper.from_array(np.array([1], np.int64), t.name + "_q8_ax")]
                new_nodes.append(helper.make_node("Slice", [cur, t.name + "_q8_st", t.name + "_q8_en", t.name + "_q8_ax"], [names[2]], name=names[2]))
                cur = names[2]
            if perm == list(range(a.ndim)):
                new_nodes.append(helper.make_node("Reshape", [cur, t.name + "_q8_shapep"], [t.name], name=names[3]))
            else:
                new_nodes.append(helper.make_node("Reshape", [cur, t.name + "_q8_shapep"], [names[3]], name=names[3]))
                new_nodes.append(helper.make_node("Transpose", [names[3]], [t.name], perm=list(np.argsort(perm)), name=t.name + "_q8_t"))
            done += 1
            continue
        if axis is None:  # 1-D: per-tensor
            amax = np.abs(a).max(keepdims=True)
        else:
            red = tuple(k for k in range(a.ndim) if k != axis)
            amax = np.abs(a).max(axis=red, keepdims=True)
        scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
        q = np.clip(np.round(a / scale), -127, 127).astype(np.int8)
        new_inits += [numpy_helper.from_array(q, qn), numpy_helper.from_array(scale, sn)]
        new_nodes += [helper.make_node("Cast", [qn], [cn], to=TensorProto.FLOAT, name=cn),
                      helper.make_node("Mul", [cn, sn], [t.name], name=t.name + "_q8_mul")]
        done += 1
    del g.initializer[:]
    g.initializer.extend(keep + new_inits)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(new_nodes + nodes)
    out = f"{MODELS}/{model}.{'wob' if block else 'wo'}.onnx"
    onnx.save(m, out)
    print(f"weight-only: {done} tensors quantised")
    return out


def fp16_weights(model, min_elems=1024):
    """Reference point only (not int8): float initializers stored as fp16 + Cast to fp32 (folded at load)."""
    m = onnx.load(ORIG[model]); g = m.graph
    keep, inits, nodes = [], [], []
    for t in g.initializer:
        a = numpy_helper.to_array(t)
        if a.dtype != np.float32 or a.size < min_elems:
            keep.append(t); continue
        inits.append(numpy_helper.from_array(a.astype(np.float16), t.name + "_f16"))
        nodes.append(helper.make_node("Cast", [t.name + "_f16"], [t.name], to=TensorProto.FLOAT, name=t.name + "_f16c"))
    del g.initializer[:]; g.initializer.extend(keep + inits)
    old = list(g.node); del g.node[:]; g.node.extend(nodes + old)
    onnx.save(m, f"{MODELS}/{model}.f16w.onnx")


def calib_feeds(model):
    import feeds as F
    if model == "gfpgan": return F.gfpgan_feeds(extra=True)
    if model == "migan": return F.migan_calib()
    if model == "modnet": return F.modnet_feeds(cal=True)
    if model == "ddcolor": return F.ddcolor_feeds(cal=True)
    if model == "sam_enc":
        out = []
        for p in sorted({v[0] for v in F.SAM_CASES.values()}) + F.SAM_CAL_EXTRA:
            x, _ = F.sam_input(F.load_rgb(p)); out.append((p, {"input_image": x}))
        return out
    if model == "sam_dec":
        from common import sess
        enc = sess(ORIG["sam_enc"])
        out = []
        for name, (p, pts) in F.SAM_CASES.items():
            rgb = F.load_rgb(p); x, s = F.sam_input(rgb)
            emb = enc.run(["image_embeddings"], {"input_image": x})[0]
            out.append((name, F.sam_dec_feeds(emb, pts, rgb.shape[1], rgb.shape[0], s)))
        return out
    raise ValueError(model)


class Reader:
    def __init__(self, feeds):
        self.it = iter([f for _, f in feeds])

    def get_next(self):
        return next(self.it, None)


STATIC_OPS = {
    "gfpgan": ["Conv", "Gemm", "MatMul"],
    "migan": ["Conv"],
    "modnet": ["Conv", "MatMul"],
    "sam_enc": ["Conv", "MatMul"],
    "sam_dec": ["MatMul", "Gemm", "Conv", "ConvTranspose"],
    "ddcolor": ["Conv", "MatMul", "Gemm"],
}


def main():
    from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType, QuantFormat, CalibrationMethod
    model, methods = sys.argv[1], sys.argv[2:]
    for method in methods:
        t0 = time.time()
        out = f"{MODELS}/{model}.{method}.onnx"
        print(f"--- {model} {method}")
        if method == "wo":
            weight_only(model)
        elif method == "f16w":
            fp16_weights(model)
        elif method == "wob":
            weight_only(model, block=32)
        elif method == "dynconv_pt":
            quantize_dynamic(pre(model), out, weight_type=QuantType.QInt8, per_channel=False,
                             op_types_to_quantize=["MatMul", "Gemm", "Conv"], extra_options={"MatMulConstBOnly": True})
        elif method in ("dyn", "dynconv"):
            ops = ["MatMul", "Gemm"] + (["Conv"] if method == "dynconv" else [])
            quantize_dynamic(pre(model), out, weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=ops,
                             extra_options={"MatMulConstBOnly": True})
        elif method in ("qdq", "qdqp"):
            cm = CalibrationMethod.MinMax if method == "qdq" else CalibrationMethod.Percentile
            ex = {"ActivationSymmetric": False, "WeightSymmetric": True}
            if method == "qdqp":
                ex["CalibPercentile"] = 99.999
            quantize_static(pre(model), out, Reader(calib_feeds(model)), quant_format=QuantFormat.QDQ, per_channel=True,
                            activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, op_types_to_quantize=STATIC_OPS[model],
                            calibrate_method=cm, extra_options=ex)
        else:
            raise ValueError(method)
        print(f"    -> {out} {os.path.getsize(out)} B in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

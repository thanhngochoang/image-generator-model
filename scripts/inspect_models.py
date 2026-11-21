"""Print opset, IO, op-type histogram and external-data status of each original model."""
import sys, collections, os
sys.path.insert(0, os.path.dirname(__file__))
import onnx
import onnxruntime as ort
from common import ORIG

print("onnx", onnx.__version__, "onnxruntime", ort.__version__)
for k, p in ORIG.items():
    m = onnx.load(p, load_external_data=False)
    ext = any(t.data_location == onnx.TensorProto.EXTERNAL for t in m.graph.initializer)
    ops = collections.Counter(n.op_type for n in m.graph.node)
    dts = collections.Counter(onnx.TensorProto.DataType.Name(t.data_type) for t in m.graph.initializer)
    def io(vs):
        return [(v.name, onnx.TensorProto.DataType.Name(v.type.tensor_type.elem_type),
                 [d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim]) for v in vs]
    print(f"\n== {k}: {p} ({os.path.getsize(p)} B)")
    print(" opset", [(o.domain or "ai.onnx", o.version) for o in m.opset_import], "ir", m.ir_version, "external", ext)
    print(" inputs", io(m.graph.input))
    print(" outputs", io(m.graph.output))
    print(" initializer dtypes", dict(dts))
    print(" nodes", sum(ops.values()), dict(ops.most_common()))

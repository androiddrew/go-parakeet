"""Inspect ONNX export metadata without loading external encoder weights."""

import argparse
import json
from pathlib import Path

import onnx


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("assets", type=Path)
args = parser.parse_args()
for path in sorted(args.assets.glob("*.onnx")):
    model = onnx.load(path, load_external_data=False)
    def describe(value):
        tensor = value.type.tensor_type
        return {
            "name": value.name,
            "type": onnx.TensorProto.DataType.Name(tensor.elem_type),
            "shape": [d.dim_param or d.dim_value for d in tensor.shape.dim],
        }
    print(json.dumps({
        "file": path.name,
        "ir": model.ir_version,
        "opsets": {o.domain: o.version for o in model.opset_import},
        "metadata": {p.key: p.value for p in model.metadata_props},
        "position_tables": [
            {"node": node.name, "shape": list(attribute.t.dims)}
            for node in model.graph.node for attribute in node.attribute
            if attribute.HasField("t") and len(attribute.t.dims) == 3
            and attribute.t.dims[0] == 1 and attribute.t.dims[1] >= 5000
            and attribute.t.dims[2] == 1024
        ],
        "inputs": [describe(v) for v in model.graph.input],
        "outputs": [describe(v) for v in model.graph.output],
    }, indent=2))

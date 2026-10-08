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
        "inputs": [describe(v) for v in model.graph.input],
        "outputs": [describe(v) for v in model.graph.output],
    }, indent=2))

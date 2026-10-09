# Parakeet assets

The engine needs the pinned
[istupakov/parakeet-tdt-0.6b-v3-onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/tree/8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce)
export at revision `8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce`. Download each file from
`https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce/<file>`.

| File | Size | Use |
|---|---:|---|
| `nemo128.onnx` | 140 KB | 16 kHz audio to normalized 128-bin mel features (both precisions) |
| `encoder-model.onnx` | 42 MB | FP32 encoder graph |
| `encoder-model.onnx.data` | 2.44 GB | FP32 encoder external weights; must be beside the graph |
| `decoder_joint-model.onnx` | 73 MB | FP32 decoder and joint network |
| `encoder-model.int8.onnx` | 652 MB | INT8 encoder |
| `decoder_joint-model.int8.onnx` | 18 MB | INT8 decoder and joint network |
| `README.md` | — | Conversion model card; retain when redistributing |

Choose FP32 or INT8; you need `nemo128.onnx` plus that precision's encoder and
decoder. [`SHA256SUMS`](SHA256SUMS) pins each file's exact contents. Its paths
assume a `parakeet/` directory under an asset root:

```bash
cd /path/to/asset-root && sha256sum -c --ignore-missing /path/to/go-parakeet/SHA256SUMS
```

`--ignore-missing` accommodates the precision you did not download; make sure
every file you need is listed as `OK`.

The vocabulary is embedded in the library (`internal/assets/vocab.txt`, the
export's unmodified `vocab.txt`; see THIRD_PARTY.md). The model files are not in
Git. Expected tensors are validated when `New` loads each graph; unsupported
exports fail at load time.

The model weights declare CC-BY-4.0. Retain the model card and source/conversion
attribution when redistributing them.

Supply an independently installed ONNX Runtime through ortenv by path or loader
name. The current binding requests C API 22; native 1.22.0 and 1.22.2 are tested
baselines, not exact-version restrictions. GPU use requires the dependencies
appropriate to the selected native runtime.

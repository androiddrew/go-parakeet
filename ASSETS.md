# Parakeet assets

The supported export is
[istupakov/parakeet-tdt-0.6b-v3-onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/tree/8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce)
at revision `8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce`.

| File | Approximate size | Use |
|---|---:|---|
| `config.json`, `vocab.txt`, `README.md` | 95 KB | Model contract, vocabulary, retained model card |
| `nemo128.onnx` | 140 KB | 16 kHz audio to normalized 128-bin mel features |
| `encoder-model.onnx` | 42 MB | FP32 encoder graph |
| `encoder-model.onnx.data` | 2.44 GB | FP32 encoder external weights; must be beside the graph |
| `decoder_joint-model.onnx` | 73 MB | FP32 decoder and joint network |
| `encoder-model.int8.onnx` | 652 MB | Quantized CPU encoder |
| `decoder_joint-model.int8.onnx` | 18 MB | Quantized CPU decoder and joint network |

The model weights declare CC-BY-4.0. Retain the model card and source/conversion
attribution when redistributing them. Preprocessing provenance and code licenses
are documented in [THIRD_PARTY.md](THIRD_PARTY.md).

## Download

The optional developer utility uses Python 3.11+ (standard library only). Python
is not needed by the inference library:

```bash
python3 tools/download-assets.py --root runtime-assets --precision fp32
# CPU quantized assets, or use --precision all for both sets:
python3 tools/download-assets.py --root runtime-assets --precision int8
```

Downloads use immutable revision URLs and verify [SHA256SUMS](SHA256SUMS) before
installing files. Existing files are verified, never silently replaced on a
checksum mismatch. All assets reside in `runtime-assets/parakeet/`, outside Git.
To download manually, use:

```
https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce/<filename>
```

Verify from your asset root with `sha256sum -c --ignore-missing
/path/to/go-parakeet/SHA256SUMS`. Ensure every file for the selected precision is
present; `--ignore-missing` accommodates the unselected precision.

Install ONNX Runtime independently. The binding requests C API 22; 1.22.2 is the
release validation target. CUDA/cuDNN dependencies follow the native runtime
build. Downloading a model does not install or configure that runtime.

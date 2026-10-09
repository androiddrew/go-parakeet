# Attribution and asset terms

Original code is Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). Retained
third-party notices are under `third_party/licenses/`.

| Component | Source/version | Terms and use |
|---|---|---|
| TDT decisions, whitespace cleanup, Python reference harness | [istupakov/onnx-asr](https://github.com/istupakov/onnx-asr/tree/675f0e68c24d846ee1775743e92d9b4ed452380e), `675f0e68c24d846ee1775743e92d9b4ed452380e` | MIT; retained `onnx-asr.txt` |
| Runtime ownership and session conventions | [androiddrew/go-kokoro](https://github.com/androiddrew/go-kokoro/tree/8153f29e65cd2a692a98fee9427dd3c049ed8b8d), `8153f29e65cd2a692a98fee9427dd3c049ed8b8d` | Apache-2.0, Drew Bednar; patterns adapted to three-model ASR |
| ortenv | `github.com/androiddrew/ortenv v0.2.0` | External Go dependency, MIT, Copyright (c) 2026 Drew Bednar |
| ONNX Runtime Go binding | `github.com/yalue/onnxruntime_go v1.22.0` | External Go dependency, MIT |
| Original Parakeet TDT 0.6B v3 weights | [NVIDIA model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | CC-BY-4.0; external model assets |
| ONNX weights, configuration, vocabulary | [istupakov conversion](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/tree/8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce), `8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce` | CC-BY-4.0 as declared by the pinned conversion card; converted/quantized from NVIDIA's weights |
| `nemo128.onnx` | Same pinned asset bundle; hash in SHA256SUMS | Graph metadata identifies Ilya Stupakov and MIT License; preprocessing source is onnx-asr |

The decoder in `internal/decoder/greedy.go` adapts the decisions in
`src/onnx_asr/asr.py` and `src/onnx_asr/models/nemo.py`. Changes separate native
execution from state commits, add context/error checks, and validate outputs.
`internal/assets/assets.go` implements the reference's SentencePiece whitespace
cleanup in Go, extended to preserve cleanup state across finalized callbacks.
Window planning and the word/time-alignment seam implementation are original
Go code. `tools/fixtures/reference.py` adapts the Python loop into a
single-utterance fixture exporter with explicit tensor types and stage dumps.

[achetronic/parakeet](https://github.com/achetronic/parakeet/tree/436daa8a75fa8c6d115a3188e18ef046444edccf)
at `436daa8a75fa8c6d115a3188e18ef046444edccf` was consulted for the model/session
architecture and decoder tensor reuse. Its LICENSE and inference-file headers
declare Apache-2.0, Copyright 2026 Alby Hernández; its README's MIT claim is
inconsistent. The current implementation follows onnx-asr's decoding rules and
does not include the server, audio conversion, handwritten mel extractor, or
long-audio implementation from that repository.

Model binaries are downloaded independently and are not relicensed by this
repository. The download tool retains and checksums the pinned conversion card;
redistributors should retain NVIDIA and conversion attribution, the CC-BY-4.0
license reference, and descriptions of conversion/quantization modifications.

The native ONNX Runtime and optional CUDA/cuDNN libraries are user-installed.
Python ONNX tooling and FFmpeg are optional development tools. FFmpeg's terms
depend on the installed build; this repository neither embeds nor links it.
Fixture prompts are original test material. Generated audio records the source
service, voice, request, converter version, and content checksums in its manifest.

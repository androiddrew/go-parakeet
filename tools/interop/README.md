# Same-process interoperability check

This separate developer module pins go-kokoro and go-g2p v0.2.0 and uses the local
go-parakeet checkout. It creates all three engines using one ORT selector, runs
G2P/TTS/ASR, closes individual engines while others remain usable, closes the last
engine, and reopens Parakeet. The TTS and ASR checks use independent audio inputs;
this is a runtime-ownership test, not an implicit resampling pipeline.

From the repository root:

```bash
python3 tools/interop/download.py --root runtime-assets
```

Then, from `tools/interop`:

```bash
go run . --ort /path/to/ort/Release/libonnxruntime.so.1.22.2 \
    --assets /absolute/path/to/runtime-assets \
    --input /absolute/path/to/16k-mono.samples.f32
# Add --cuda to put Kokoro and Parakeet on CUDA. G2P remains on CPU.
```

The Kokoro downloads are pinned to the model-files-v1.0 release and verified
against go-kokoro v0.2.0's SHA256SUMS. Model/voice terms and acknowledgments are
documented in that library's ASSETS.md and THIRD_PARTY.md. These optional tool
dependencies are not added to the inference module.

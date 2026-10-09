# Reference fixtures

Unit tests use synthetic logits, vocabularies, and metadata. Native parity tests
consume recorded audio/features/encoder outputs and token decisions generated
by a separate Python ONNX execution harness. No test contacts a live TTS service.

## Provision development inputs

```bash
python3 tools/download-assets.py --root runtime-assets --precision all
python3 tools/fixtures/generate.py --base-url http://localhost:8880 \
    --output artifacts/fixtures
```

The second command explicitly calls the developer's Kokoro service. Four original
prompts exercise US/UK voices, punctuation, and genuine repetition. It retains
source WAV, resampled 16 kHz PCM16 WAV, exact API requests, converter version,
and checksums. FFmpeg must support the soxr resampler. Existing fixtures are not
overwritten; choose a new directory to regenerate. Repeated TTS generation is
not assumed byte-identical: saved audio and its checksum define each test input.

Install `tools/fixtures/requirements.txt` in a development virtual environment
(tested Python 3.12). Python, NumPy, and FFmpeg are not runtime dependencies of
the Go library. For example:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r tools/fixtures/requirements.txt
.venv/bin/python tools/fixtures/reference.py --assets runtime-assets/parakeet \
    --output artifacts/reference-fp32 \
    artifacts/fixtures/clean-us.wav artifacts/fixtures/clean-gb.wav \
    artifacts/fixtures/repetition-us.wav artifacts/fixtures/repetition-gb.wav
.venv/bin/python tools/fixtures/reference.py --assets runtime-assets/parakeet \
    --precision int8 --output artifacts/reference-int8 \
    artifacts/fixtures/clean-us.wav artifacts/fixtures/clean-gb.wav \
    artifacts/fixtures/repetition-us.wav artifacts/fixtures/repetition-gb.wav
```

The reference harness uses ORT 1.22.0 on CPU, fixed one-thread settings, and the
same pinned graphs. Its TDT rules come from the pinned onnx-asr source identified
in THIRD_PARTY.md. Each manifest records actual valid lengths, token IDs, frame
positions/advances, text, and checksummed little-endian float32 stage files.
The graph's valid length can differ from its physical padded tensor width.

## CPU checks

```bash
PARAKEET_TEST_ORT=libonnxruntime.so \
PARAKEET_TEST_ASSETS=runtime-assets/parakeet \
PARAKEET_TEST_REFERENCE=artifacts/reference-fp32 \
    go test -race -count=1 ./...

PARAKEET_TEST_ORT=libonnxruntime.so \
PARAKEET_TEST_ASSETS=runtime-assets/parakeet \
PARAKEET_TEST_PRECISION=int8 \
PARAKEET_TEST_REFERENCE=artifacts/reference-int8 \
    go test -race -count=1 -run '^(TestNativeParity|TestNativeChunking)$' .
```

Parity checks compare features/encoder arrays with absolute and relative
tolerances of 1e-4, then require exact text and token decisions against each
precision's own baseline. Each utterance runs twice to catch stale recurrent
state. Fixture hashes are verified before use.

## CUDA checks

```bash
PARAKEET_TEST_ORT=libonnxruntime.so \
PARAKEET_TEST_CUDA_ORT=libonnxruntime.so \
PARAKEET_TEST_ASSETS=runtime-assets/parakeet \
PARAKEET_TEST_REFERENCE=artifacts/reference-fp32 \
    go test -race -tags=cuda -count=1 -run '^(TestCUDAInferenceCycles|TestCUDAChunking)$' .
```

This runs the four fixtures across two engine create/close cycles and compares
text and token IDs with CPU FP32. Placement can be inspected using the probe's
`--provider cuda --verbose` flags. Provider registration alone does not establish
that the model is executing on the GPU.

## Long-audio and seam checks

`TestNativeChunking` / `TestCUDAChunking` reconstruct a 19.267-second recording
from the saved stage inputs, joining them with 160 ms gaps and shifting seam
positions. They check the spoken sequence, stream/aggregate equality, timings,
source coverage, cancellation, callback failure, result ownership, and Close
serialization. FP32 uses 8–12-second stress windows; INT8 uses 10.4–16 seconds.
See VALIDATION.md for the quantized model's 8-second-window limitation.

Enable a repeated recording exceeding 300 seconds separately:

```bash
PARAKEET_TEST_ORT=libonnxruntime.so \
PARAKEET_TEST_ASSETS=runtime-assets/parakeet \
PARAKEET_TEST_REFERENCE=artifacts/reference-fp32 \
PARAKEET_TEST_LONG=1 \
    go test -race -count=1 -run '^TestNativeLongRecording$' -v .

# For CPU INT8, also set PARAKEET_TEST_PRECISION=int8 and use reference-int8.
# For CUDA, use its explicit opt-in library and build tag:
PARAKEET_TEST_ORT=libonnxruntime.so \
PARAKEET_TEST_CUDA_ORT=libonnxruntime.so \
PARAKEET_TEST_ASSETS=runtime-assets/parakeet \
PARAKEET_TEST_REFERENCE=artifacts/reference-fp32 \
PARAKEET_TEST_LONG=1 \
    go test -race -tags=cuda -count=1 -run '^TestCUDALongRecording$' -v .
```

The long tests use 30-second windows with 4-second overlap, compare the full
normalized spoken sequence, and report first-chunk latency and RTF. They prove
that automatic windowing handles a recording beyond the explicit SinglePass
limit. To produce the same input as a reusable probe fixture:

```bash
python3 tools/fixtures/concatenate.py --references artifacts/reference-fp32 \
    --repeat 16 --output artifacts/long
```

Unit seam tests include the INT8-derived failure cases and run without native
models. Ordinary CI therefore covers the planner/merger/streaming contracts even
before a redistributable native corpus is frozen.

## Corpus growth before v0.1.0

Add frozen redistributable fixtures for native CI, licensed natural/multilingual
speech, longer continuous passages, shifted seam locations, pauses, and repeated
words at overlaps. The present synthetic English recordings establish execution
parity and controlled seam/long-audio behavior, not general ASR accuracy.

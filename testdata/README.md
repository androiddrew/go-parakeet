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
    go test -race -count=1 -run '^TestNativeParity$' .
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
    go test -race -tags=cuda -count=1 -run '^TestCUDAInferenceCycles$' .
```

This runs the four fixtures across two engine create/close cycles and compares
text and token IDs with CPU FP32. Placement can be inspected using the probe's
`--provider cuda --verbose` flags. Provider registration alone does not establish
that the model is executing on the GPU.

## Corpus growth before v0.1.0

Add frozen redistributable fixtures for native CI, licensed natural/multilingual
speech, longer continuous passages, shifted seam locations, pauses, and repeated
words at overlaps. The present four short synthetic English recordings establish
single-pass execution parity, not general ASR accuracy or long-audio quality.

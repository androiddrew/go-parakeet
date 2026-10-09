# Test fixtures

Unit tests use synthetic logits, vocabularies, and metadata and need no fixtures
beyond the checksum check. `native/` holds the inputs and expected outputs for
the native parity tests.

`native/` contains four synthetic recordings (US/UK English voices, clean speech
and genuine repetition) as checksummed 16 kHz float32 PCM, together with the
preprocessor and encoder tensors and the token decisions produced by a Python
ONNX Runtime 1.22.0 reference using onnx-asr's TDT rules (see THIRD_PARTY.md).
`fp32/` and `int8/` hold each precision's own expected decisions. The INT8
baselines were recorded on an AVX2 CPU without VNNI; ONNX Runtime's quantized
kernels differ by CPU feature set, so INT8 encoder parity is only expected on
comparable hardware. FP32 parity is hardware-independent within tolerance; tensors shared
between them are deduplicated into gzip-compressed `blobs/`. `PROVENANCE.json`
records the original TTS requests, voices, resampling command and hashes.

`TestFrozenFixtures` verifies `native/SHA256SUMS` in ordinary unit runs. Native
tests check features and encoder outputs with absolute and relative tolerances of
1e-4, then require exact text and token decisions against the selected
precision's baseline. Each utterance runs twice to catch stale decoder state.
The chunking tests join the four recordings, with 160 ms gaps, into a longer
recording to exercise seams.

These fixtures establish Go/Python implementation parity and controlled seam
behavior, not ASR accuracy.

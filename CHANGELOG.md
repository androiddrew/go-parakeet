# Changelog

## [Unreleased]

### Added

- Parakeet TDT 0.6B v3 inference with pinned ONNX assets, CPU FP32/INT8 and CUDA FP32.
- Process-lifetime ORT coordination through ortenv, with caller-owned mode.
- 16 kHz mono PCM input, greedy TDT decoding, text and recording-relative token timing.
- Automatic long-audio windows, whole-word seam reconciliation and finalized-chunk callbacks.
- Owned results, synchronous callback backpressure, cancellation checks, explicit errors and stage timings.
- Frozen synthetic reference tensors, six-language FLEURS fixtures, native CI and race/lifecycle tests.
- ORT 1.22.2 CPU/CUDA source-build validation and same-process Kokoro/G2P interoperability checks.
- Inference probe, reproducible fixture tools, warm-inference/RSS/VRAM benchmark tooling and release evidence.

### Known limits

- Native Run cannot be interrupted mid-call. Streaming accepts complete PCM input.
- Token positions are model emissions, not exact word boundaries.
- Very short INT8 windows can omit speech. Greek accuracy is lower on the small natural-speech sample; see VALIDATION.md.
- Linux/amd64 is the validated platform; other architectures require their own native validation.

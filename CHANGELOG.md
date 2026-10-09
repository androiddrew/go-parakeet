# Changelog

## [Unreleased]

## [0.1.0] - 2026-10-09

### Added

- Parakeet TDT 0.6B v3 inference with pinned ONNX assets, CPU FP32/INT8 and CUDA FP32.
- Embedded vocabulary; `VocabPath` optionally overrides it.
- Process-lifetime ORT coordination through ortenv, with caller-owned mode.
- 16 kHz mono PCM input, greedy TDT decoding, text and recording-relative token timing.
- Automatic long-audio windows, whole-word seam reconciliation and finalized-chunk callbacks.
- Owned results, synchronous callback backpressure, cancellation checks, explicit errors and stage timings.
- `cmd/inference-probe` diagnostic tool.

### Known limits

- Native Run cannot be interrupted mid-call. Streaming accepts complete PCM input.
- Token positions are model emissions, not exact word boundaries.
- Very short INT8 windows can omit speech; keep the default windows for INT8.
- Linux/amd64 is the tested platform; other architectures require their own native validation.

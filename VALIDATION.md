# Implementation and validation status

Initial checkpoint: 2026-10-08. This is development evidence, not a v0.1.0 release.

## Implemented

- Pinned, checksummed FP32/INT8 model bundle and retained conversion model card.
- Explicit paths, configuration/vocabulary validation, and tensor-contract checks.
- ortenv v0.2.0 process ownership and caller-managed environment mode.
- CPU preprocessing, CPU/CUDA encoder and decoder sessions.
- Single-pass greedy TDT decoding with reusable tensors, state reset,
  cancellation checks, owned text/tokens, emission timestamps, and stage timings.
- CPU reference exporter, controlled Kokoro fixture generator, and Go probe.
- Unit, parity, concurrency/lifetime, and opt-in CUDA regression tests.

## Executed checks

Host: Linux/amd64, Go 1.26.4, native ONNX Runtime **1.22.0**, RTX 4090 for CUDA.
Python reference: Python 3.12, onnxruntime 1.22.0, numpy 2.2.6.
Assets: `8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce`; all SHA256SUMS verified.

- `go test ./...`, `go test -race ./...`, `go vet ./...`, probe build: passed.
- Minimum-toolchain check: `GOTOOLCHAIN=go1.25.0 go test ./...` passed.
- CPU FP32 and INT8: four fixtures each, repeated calls; features/encoder values
  matched reference tolerances, text and all token decisions matched exactly.
- Native lifecycle: shared environment, independent engines, idempotent closure,
  use-after-close, concurrent requests/Close, conflicting selectors, reopening
  after the last engine closes, and caller-owned environment: passed under race.
- Minimum-duration and subsampling-edge inputs (320, 321, 1279, 1280, and 1281
  samples) completed native inference successfully.
- CUDA FP32: four fixtures across two engine cycles; text/token IDs matched
  CPU FP32. Native initialization failures would fail this opt-in test.
- Verbose native placement during real CUDA inference: preprocessor 35 CPU
  nodes, encoder 1515 CUDA and 482 CPU nodes, decoder 23 CUDA nodes. Inspection
  and retained inference sessions each reported these placements.

The four recordings came from the developer-provided Kokoro API, using af_heart,
bf_emma, am_adam, and bm_george. Manifests and stage files are stored locally under
`artifacts/fixtures` and `artifacts/reference-{fp32,int8}`. Generation instructions
are in testdata/README.md. Recordings and reference files are not yet distributed
as a frozen test corpus.

FP32 and INT8 differ in punctuation on one repeated-phrase fixture. Both preserve
the spoken words. Tests use precision-specific references rather than assuming
quantization preserves every token. No broad accuracy or performance claim is
based on this small synthetic corpus.

## Required before v0.1.0

- [ ] Exercise native ORT **1.22.2** on CPU/CUDA. The installed 1.22.0 baseline
  does not complete this gate.
- [ ] Verify exact exported-model length boundaries and benchmark safe window
  sizes. The initial single-pass API uses a conservative 300-second cap.
- [ ] Implement overlapping-window planning, seam reconciliation, and absolute
  token timing for long recordings.
- [ ] Implement finalized-chunk callbacks and streaming/aggregate equivalence,
  backpressure, cancellation, and partial-delivery failure tests.
- [ ] Freeze and distribute native-CI fixtures; add natural/multilingual audio.
- [ ] Benchmark first-result latency, warm throughput, native memory, and GPU
  placement across representative durations.
- [ ] Exercise composition with go-kokoro/go-g2p in the same application.

Automatic long-audio windowing and TranscribeStream are not exposed by the
current API. They remain part of the agreed initial release scope.

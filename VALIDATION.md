# Implementation and validation status

Checkpoint: 2026-10-08, including long-audio and streaming implementation.
This is development evidence, not a v0.1.0 release.

## Implemented

- Pinned, checksummed FP32/INT8 model bundle and retained conversion model card.
- Explicit paths, configuration/vocabulary validation, and tensor-contract checks.
- ortenv v0.2.0 process ownership and caller-managed environment mode.
- CPU preprocessing, CPU/CUDA encoder and decoder sessions.
- Single-pass greedy TDT decoding with reusable tensors, state reset,
  cancellation checks, owned text/tokens, emission timestamps, and stage timings.
- CPU reference exporter, controlled Kokoro fixture generator, and Go probe.
- Unit, parity, concurrency/lifetime, and opt-in CUDA regression tests.
- Constant-space, frame-aligned window planning, bounded word alignment,
  quiet-region/timestamp fallback, whole-word holdback, and absolute token times.
- Finalized-chunk callbacks and aggregate collection through one shared path.
  Context/error propagation, owned delivery, synchronous backpressure, and
  callback-excluding timings are covered by unit and native tests.

## Executed checks

Host: Linux/amd64, Go 1.26.4, native ONNX Runtime **1.22.0**, RTX 4090 for CUDA.
Python reference: Python 3.12, onnxruntime 1.22.0, numpy 2.2.6.
Assets: `8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce`; all SHA256SUMS verified.

- `go test ./...`, `go test -race ./...`, `go vet ./...`, probe build: passed.
- Minimum-toolchain check: `GOTOOLCHAIN=go1.25.0 go test -race ./...` passed.
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

## Long audio and streaming checks

Native seam tests concatenate the four saved inputs with 160 ms gaps. They also
shift the recording by 400 ms so boundaries cross different speech:

- CPU/CUDA FP32: windows of 8, 10.4, and 12 seconds; overlaps of 3.2 or 4 seconds.
- CPU INT8: windows of 10.4, 12, and 16 seconds; 4-second overlaps.
- All listed configurations preserve the normalized spoken sequence, including
  `very, very` and both occurrences of `one more time`.
- Normalization removes case/punctuation and explicitly treats `window sill`
  and `windowsill` as the same spoken word. Raw text and token IDs are still
  required to agree exactly between stream and aggregate with identical settings.
- Tests check complete contiguous source coverage, monotonic token timing,
  cancellation after partial delivery, callback error identity, subsequent engine
  reuse, delivered-buffer lifetime, and Close waiting for a callback to finish.
- Unit tests additionally cover final-callback cancellation, invalid input before
  any native run/delivery, partial native-failure timings, empty windows, short
  tails, whole-word seams, repeated words, independent tokenizations, and
  near-integer-limit lazy plans. Pathological unbroken hypotheses fail explicitly
  at the retained-token budget instead of growing with recording length.

Two INT8 stress failures became asset-free seam regressions: retaining speech
recovered before an alignment anchor, and retaining disjoint decoded spans when
one window has no hypothesis for part of the overlap. An additional omission
remains in the quantized model's raw predictions with 8-second windows, so that
configuration is not counted as passing the accuracy check. The 30-second
default and the longer tested INT8 contexts retain the source speech. This is
a model/context sensitivity, not a general guarantee for every recording.

### Recording beyond the original single-pass limit

Repeating the 19.267-second concatenation 16 times produces **308.268 seconds**.
It is rejected in explicit SinglePass mode and succeeds in normal streaming
mode with 12 windows (30 seconds, 4-second overlap). CPU FP32, CPU INT8, and CUDA
FP32 all preserve the complete normalized spoken sequence, including repetitions.

The generated probe fixture has SHA-256
`faf603f710ceb33030e88df29d9edcacc258ff83ceae6d05bfa52b6136bf0414`.
The actual CUDA probe's aggregate and JSON-lines modes produced identical text,
tokens, and all 12 finalized chunks on this file. Explicit `--single-pass`
rejected it, exited nonzero, and emitted no success event.

Observed development-host timings with the race detector enabled:

| Execution | First finalized chunk | Total transcription | RTF |
|---|---:|---:|---:|
| CPU FP32, one intra-op thread | 5.16 s | 57.24 s | 0.186 |
| CPU INT8, one intra-op thread | 3.83 s | 42.52 s | 0.138 |
| CUDA FP32, RTX 4090 | 0.82 s | 4.58 s | 0.015 |

These are smoke measurements on one repeated synthetic corpus, with the engine
already loaded and the first native request included. They exclude engine
initialization/audio preparation and are not a controlled performance benchmark.
CPU checks ran concurrently on the development host. Native memory and broader
accuracy/performance measurements remain release work.

## Required before v0.1.0

- [ ] Exercise native ORT **1.22.2** on CPU/CUDA. The installed 1.22.0 baseline
  does not complete this gate. Testing requires a suitable Linux native build;
  the [published 1.22.2 release assets](https://github.com/microsoft/onnxruntime/releases/tag/v1.22.2)
  are NuGet packages.
- [ ] Verify exact exported-model length boundaries and benchmark safe window
  sizes across hardware. Individual passes retain a conservative 300-second cap.
- [x] Implement overlapping-window planning, seam reconciliation, and absolute
  token timing for long recordings.
- [x] Implement finalized-chunk callbacks and streaming/aggregate equivalence,
  backpressure, cancellation, and partial-delivery failure tests.
- [ ] Freeze and distribute native-CI fixtures; add natural/multilingual audio.
- [ ] Benchmark first-result latency, warm throughput, native memory, and GPU
  placement across representative durations.
- [ ] Exercise composition with go-kokoro/go-g2p in the same application.

Long-audio windowing and TranscribeStream are now exposed by the API. Validation
on natural/multilingual speech is still needed before the initial release.

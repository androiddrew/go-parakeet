# Implementation and validation status

Release-candidate checkpoint: 2026-10-09. Local acceptance and final review are
complete on the declared Linux/amd64 CPU/CUDA configurations, including the
post-reboot checks below. Hosted CI must pass on the candidate commit before
publication; the tag has not been published.

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

Host: Linux/amd64, Ryzen 5 5600X, Go 1.25/1.26.4, RTX 4090 for CUDA.
Native **ONNX Runtime 1.22.2** CPU and CUDA builds passed release acceptance;
1.22.0 passed the earlier development baseline.
Python reference: Python 3.12, onnxruntime 1.22.0, numpy 2.2.6.
Assets: `8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce`; all SHA256SUMS verified.

- `go test ./...`, `go test -race ./...`, `go vet ./...`, probe/benchmark builds: passed.
- Minimum-toolchain check: `GOTOOLCHAIN=go1.25.0 go test -race ./...` passed.
- Final Go 1.25.0 CPU INT8 sweep with ORT 1.22.2, `-race -count=1 -timeout=15m`
  and `PARAKEET_TEST_LONG=1`: passed, including parity, natural speech, streaming,
  lifecycle, and the 308.268-second recording.
- CPU FP32 and INT8: four fixtures each, repeated calls; features/encoder values
  matched reference tolerances, text and all token decisions matched exactly.
- Native lifecycle: shared environment, independent engines, idempotent closure,
  use-after-close, concurrent requests/Close, conflicting selectors, reopening
  after the last engine closes, and caller-owned environment: passed under race.
- Minimum-duration and subsampling-edge inputs (320, 321, 1279, 1280, and 1281
  samples) completed native inference successfully.
- CUDA FP32: four fixtures across two engine cycles; text/token IDs matched
  CPU FP32. Native initialization failures would fail this opt-in test.
- Verbose native placement during real ORT 1.22.2 CUDA inference: preprocessor 35 CPU
  nodes, encoder 1515 CUDA and 482 CPU nodes, decoder 23 CUDA nodes. Inspection
  and retained inference sessions each reported these placements. These counts
  were rechecked after reboot and match the earlier 1.22.0 development baseline.

After the system update/reboot, the NVIDIA driver and system NVML both report
595.99.02. All 34 preserved runtime files passed SHA-256 verification, and CUDA
inference with the preserved cuDNN libraries matched the frozen `clean-us` FP32
text and token IDs without a legacy NVML preload. Probe evidence is recorded in
`validation/runtime-builds.json`; the benchmark measurements below retain their
original pre-reboot driver provenance.

The four recordings came from the developer-provided Kokoro API, using af_heart,
bf_emma, am_adam, and bm_george. Manifests and stage files are stored locally under
`artifacts/fixtures` and `artifacts/reference-{fp32,int8}`. Generation instructions
are in testdata/README.md. The frozen, checksummed synthetic corpus is now
distributed under `testdata/native`; the local artifact directories are only
regeneration intermediates. Native tests default to the frozen files.

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

Earlier ORT 1.22.0 development-host timings with the race detector enabled:

| Execution | First finalized chunk | Total transcription | RTF |
|---|---:|---:|---:|
| CPU FP32, one intra-op thread | 5.16 s | 57.24 s | 0.186 |
| CPU INT8, one intra-op thread | 3.83 s | 42.52 s | 0.138 |
| CUDA FP32, RTX 4090 | 0.82 s | 4.58 s | 0.015 |

These are smoke measurements on one repeated synthetic corpus, with the engine
already loaded and the first native request included. They exclude engine
initialization/audio preparation and are not a controlled performance benchmark.
CPU checks ran concurrently on the development host. The release measurements
below use separate, sequential, non-race benchmark processes.

## ORT 1.22.2 and composition

Both builds use upstream commit `5630b081cd25e4eccc7516a652ff956e51676794`, GCC
12.4 and the standard operator set. CUDA was compiled for SM89 using nvcc 12.0
and cuDNN 9.8. Source/build provenance and tested core-library SHA-256 values are
in [validation/runtime-builds.json](validation/runtime-builds.json).

CPU FP32/INT8 reference parity and natural-speech tests pass against the frozen
Python ORT 1.22.0 outputs. CPU lifetime/race/caller-owned checks and CUDA engine
cycles, chunking, natural speech and long recording checks pass on 1.22.2.
The optional `tools/interop` module also passed on CPU and CUDA with go-kokoro
and go-g2p v0.2.0: all three engines coexist, closing one leaves the others usable,
and Parakeet reopens successfully after the final engine is closed.

### Pinned export boundary

Inspection finds a constant positional table `[1,9999,1024]` in both encoder
precisions. Its intended relative-position span covers **5,000 encoder frames**.
Direct encoder boundary tests on ORT 1.22.2 CPU and CUDA found:

- 40,000 mel frames -> 5,000 encoder frames: succeeds.
- 40,001 mel frames -> 5,001 encoder frames: also runs, despite exceeding the
  intended table span. Successful execution here is not a valid capacity claim.
- 40,009 mel frames -> 5,002 encoder frames: fails at the first relative-attention
  Add with incompatible shapes ending in `5002` and `2`.
- A valid boundary run succeeds after that rejected run.

This matches the negative-slice/broadcast edge of the fixed relative-position
table: the next frame can degenerate to a singleton broadcast before subsequent
lengths fail. The public **300-second per-pass cap remains well below this edge**;
normal long recordings use 30-second windows. Direct boundary tests bypass the
cap only under explicit `PARAKEET_TEST_LIMIT=1`.

## Natural/multilingual pilot

The frozen corpus adds 18 FLEURS clips, three per language, about 226 seconds in
total. They are the first WAV members of each selected pinned test archive,
chosen before inference. Provenance, original/converted checksums and human
transcripts are in `testdata/natural/manifest.json`; CC-BY-4.0 attribution is
retained under `third_party/licenses/fleurs.txt`.

Go CPU outputs match the Python reference text and token decisions exactly;
CUDA text matches the FP32 reference. Human-reference word errors are a separate
measurement. Normalization lowercases and splits on non-letter/number/mark
characters, without number expansion or spelling correction:

| Language | Reference words | FP32 edits / WER | INT8 edits / WER |
|---|---:|---:|---:|
| English | 73 | 3 / 4.11% | 3 / 4.11% |
| French | 109 | 2 / 1.83% | 6 / 5.50% |
| German | 62 | 3 / 4.84% | 2 / 3.23% |
| Spanish | 102 | 3 / 2.94% | 6 / 5.88% |
| Russian | 45 | 0 / 0% | 0 / 0% |
| Greek | 74 | 27 / 36.49% | 33 / 44.59% |

The higher Greek error is present in the reference model output too; those
examples remain in the corpus. Three clips per language establish coverage,
not a language-wide accuracy estimate. This is a parity-oriented release test,
not a competitive ASR benchmark or validation of all 25 model languages.

## Release performance and memory

Raw local reports are summarized in
[validation/benchmarks.json](validation/benchmarks.json). Each provider ran in a
separate sequential process, using one intra-op thread, with a first request,
one warmup, and three measured warm requests at each duration. Audio preparation
and model loading are outside request timing; the consumer hashes chunk text
without retaining aggregate output. Repeated output hashes matched.

| Provider / precision | 5 s warm median | 30 s warm median | 308 s warm median | 308 s warm first chunk | Peak process RSS |
|---|---:|---:|---:|---:|---:|
| CPU FP32 | 0.786 s | 4.547 s | 54.251 s | 4.635 s | 2821 MiB |
| CPU INT8 | 0.527 s | 3.445 s | 39.265 s | 3.291 s | 1571 MiB |
| CUDA FP32 | 0.147 s | 0.226 s | 2.869 s | 0.264 s | 1092 MiB |

CUDA's sampled per-process VRAM peak was **3422 MiB**. RSS includes native
allocations, mapped pages, Go heap and input storage; it is not a native-only
allocation counter. Go heap observations are separately recorded. From the
30-second to 308-second case, RSS high-water increased by about 33 MiB (CPU FP32),
35 MiB (INT8), and 80 MiB (CUDA host memory), rather than scaling with the number
of encoder windows. The full input slice still grows with recording duration.

VRAM was sampled every 100 ms and may miss short peaks. The host's installed
NVML had been upgraded beyond its loaded kernel version; only the sampler used
an extracted matching NVML 595.91.07 library. The inference process retained its
normal loader environment. These measurements describe this hardware, seeded
synthetic audio and window configuration; raw ranges and methodology are retained.

## Native CI

The workflow builds pinned CPU ORT 1.22.2 (cached), downloads and checksum-verifies
INT8 models, and runs native parity, natural-speech, streaming and lifetime checks.
Ordinary CI checks both Go 1.25 and stable. Fixtures ship in the repo; native CI
uses neither a live TTS API nor Python inference. GPU, long-recording and boundary
checks are explicit local acceptance steps. Hosted execution awaits the candidate
commit being pushed; local equivalents are recorded above.

`actionlint` v1.7.7 passes. The pinned CPU CMake configuration and ONNX code
generator also passed in a fresh Python 3.12 environment without pip or third-party
packages; ORT symbol generation passed with Python site packages disabled. The
workflow explicitly installs GCC 12, CMake, Ninja, git, patch and Python. A full
clean hosted build remains part of the pending hosted CI gate.

The final fixture review verified complete checksum coverage: 26 synthetic files
(3,344,014 bytes) and 22 natural-corpus files (6,180,011 bytes), including manifests
and checksums. Human transcripts, predictions, and retained FLEURS attribution
were reviewed separately. Both Go modules pass `go mod verify`; the inference
module still depends only on ortenv v0.2.0 and onnxruntime_go v1.22.0.

## Required before v0.1.0

- [x] Exercise native ORT **1.22.2** on CPU/CUDA using pinned source builds.
- [x] Verify the pinned export's positional boundary and measure default windows
  on the declared CPU/CUDA test hardware. Individual passes retain a 300-second cap.
- [x] Implement overlapping-window planning, seam reconciliation, and absolute
  token timing for long recordings.
- [x] Implement finalized-chunk callbacks and streaming/aggregate equivalence,
  backpressure, cancellation, and partial-delivery failure tests.
- [x] Freeze and distribute native-CI fixtures; add natural/multilingual audio.
- [x] Measure first-result latency, warm throughput, process RSS and GPU memory
  across representative durations; verify GPU placement.
- [x] Exercise composition with go-kokoro/go-g2p in the same application.
- [ ] Confirm hosted CI on the candidate commit and publish the v0.1.0 tag.

Release scope is Linux/amd64 CPU and CUDA with the pinned model family. Other
architectures and broader audio populations need their own validation.

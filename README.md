# go-parakeet

`github.com/androiddrew/go-parakeet` (package `parakeet`) is an inference library
for NVIDIA Parakeet TDT 0.6B v3 through ONNX Runtime. It accepts 16 kHz mono
float32 PCM and produces text and token emission timestamps. Applications supply
audio decoding/resampling and their own CLI or serving layer.

## Development status

The **v0.1.0 release candidate** has passed local acceptance and final review.
Hosted CI on the candidate commit and publication remain. CPU/CUDA inference,
**long-audio windowing**, and **finalized-chunk callbacks** are implemented.
Both transcription methods accept complete audio of at least 20 ms and use the
same pipeline. A 308-second recording has been checked on CPU FP32/INT8 and CUDA
FP32. Frozen native fixtures also cover 18 natural-speech clips in six languages.
See [VALIDATION.md](VALIDATION.md) for evidence, accuracy limits, and release scope,
and [RELEASING.md](RELEASING.md) for the acceptance/publishing procedure.

## Usage

Download the model files as described in [ASSETS.md](ASSETS.md), then supply
their paths and an independently installed ONNX Runtime:

```go
import (
    "context"
    "errors"
    parakeet "github.com/androiddrew/go-parakeet"
)

func transcribe(ctx context.Context, samples []float32) (result parakeet.Result, err error) {
    engine, err := parakeet.New(parakeet.Config{
        ORTLibrary:       "libonnxruntime.so",
        PreprocessorPath: "/assets/parakeet/nemo128.onnx",
        EncoderPath:      "/assets/parakeet/encoder-model.onnx",
        DecoderPath:      "/assets/parakeet/decoder_joint-model.onnx",
        ConfigPath:       "/assets/parakeet/config.json",
        VocabPath:        "/assets/parakeet/vocab.txt",
        Provider:         parakeet.CPU, // or parakeet.CUDA
        Threads:          1,
    })
    if err != nil { return result, err }
    defer func() { err = errors.Join(err, engine.Close()) }()
    return engine.Transcribe(ctx, parakeet.Request{
        Samples: samples, SampleRate: parakeet.SampleRate,
    })
}
```

Reuse an initialized engine across requests in an application. For INT8 CPU
inference, select `encoder-model.int8.onnx` and `decoder_joint-model.int8.onnx`.
FP32's `encoder-model.onnx.data` must be beside its graph.

`Result.Tokens` retains raw SentencePiece IDs/pieces, recording-relative emission
frames/timestamps, and predicted frame advances. `Result.Chunks` contains the
finalized chunks. Its embedded `Summary` reports duration, decoded windows,
delivered chunks, finalized source coverage, effective window sizes, and timings.
`ModelInfo()` returns owned tensor metadata and
`Initialization()` reports loading times and the actual native runtime version.
`Transcribe` returns no partial transcript on error. Invalid/nonfinite input fails before
inference; samples are not rescaled or clipped.

## Long audio

`ChunkOptions{}` automatically uses **30-second windows with 4-second overlap**.
Audio fitting one window follows the original single-pass path. Customize the
latency/context tradeoff per request:

```go
// Import "time" for durations.
result, err := engine.Transcribe(ctx, parakeet.Request{
    Samples: samples, SampleRate: parakeet.SampleRate,
    Chunking: parakeet.ChunkOptions{
        Window:  30 * time.Second,
        Overlap: 4 * time.Second,
    },
})
```

- `Window`: 2–300 seconds; zero selects 30 seconds.
- `Overlap`: positive, at most half the window; zero selects
  `min(4 seconds, Window/4)`.
- Durations round down to the 80 ms encoder grid. A positive overlap must round
  to at least one frame. Effective sizes are reported as sample counts in Summary.
- `SinglePass: true`: disables windowing and rejects input exceeding
  `MaxWindowSamples` (300 seconds). Window/Overlap must then be zero.
- `MaxSamples` remains an alias for the per-pass limit; it no longer limits a
  normally windowed recording. The cap is not a memory guarantee on every GPU.

Each window is preprocessed independently and starts with fresh decoder state.
Seam reconciliation aligns whole SentencePiece words using timestamps and a
bounded word-alignment search. It prefers quiet regions for boundaries and uses
timestamp ownership when alignment is ambiguous. The unresolved overlap and a
640 ms timestamp-drift margin are retained until the next window is decoded.
Small backwards timestamp differences at a splice are clamped, retaining words
and nondecreasing times. `ChunkResult.Seam` identifies `aligned` or `timestamp`
joins; its first chunk has an empty seam value.

A pathological unbroken hypothesis that cannot be finalized within two windows'
maximum token budget returns `ErrUnstableSeam`. This keeps stream memory bounded
and avoids silently cutting an unresolved word; already delivered chunks remain
valid. Increasing window context or splitting the input at pauses can help.

Windowing bounds intermediate inference memory. The caller still holds the
complete PCM input. `Transcribe` additionally retains the complete output.
Smaller windows change acoustic context and per-window normalization and can
change recognition. Keep the defaults for INT8 unless shorter windows are
validated on your inputs: an 8-second stress configuration omitted speech in
the quantized model's raw predictions on our small corpus.

## Incremental finalized text

Use `TranscribeStream` to consume stable text before the complete recording
finishes. Reuse the initialized engine from the example above:

```go
// output is an application-supplied io.Writer.
summary, err := engine.TranscribeStream(ctx, parakeet.Request{
    Samples: samples, SampleRate: parakeet.SampleRate,
}, func(chunk parakeet.ChunkResult) error {
    // Forward, display, or enqueue the chunk. Text may begin with a space.
    _, err := io.WriteString(output, chunk.Text)
    return err
})
```

Concatenating `chunk.Text` **verbatim** equals a successful `Transcribe` result
with the same settings. Text is never revised. Each callback owns its token
slice, which remains valid after further inference, errors, and engine closure.
Chunks describe contiguous finalized source spans in `StartSample`/`EndSample`;
these are progress boundaries, not precise word alignment. Empty text chunks
can report progress through silence.

The callback is synchronous and applies backpressure. The library starts no
background goroutines and retains no aggregate output in streaming mode.
It waits for a complete encoder window before decoding/delivering that window's
stable prefix; this is incremental output from complete audio, rather than a
live microphone input API.

**Failure and cancellation:** callback errors stop processing, are wrapped with
the chunk index, and support `errors.Is`/`errors.As`. Previously delivered data
remains usable. Summary includes completed work and timings on error;
`DeliveredChunks` counts callbacks that returned nil, and `FinalizedSeconds`
ends at the last such chunk. A callback that returns an error may already have
consumed some of its chunk. Cancellation is checked between native calls and
before/after callbacks, including the final callback. Blocking callbacks must
observe context themselves. Timings exclude consumer callback time.

Requests and Close stay serialized for the whole stream. **Do not call
Transcribe, TranscribeStream, or Close on the same engine from its callback**;
that can deadlock. Close the engine after transcription returns.

## Runtime and assets

Requires Go 1.25 or newer, CGO, an installed ONNX Runtime, and separately
downloaded model files. The Go binding targets C API 22; native ONNX Runtime
1.22.2 is the validated release baseline: Linux/amd64, Go 1.25/1.26.4,
CPU FP32/INT8, and CUDA FP32 on an RTX 4090. Native **ORT 1.22.0** was also tested
during development. See [VALIDATION.md](VALIDATION.md) for evidence. These are
tested configurations, not exact version-string
restrictions. Check the resolved binding with
`go list -m github.com/yalue/onnxruntime_go`; a consuming application's dependency
graph can select a newer binding through Go's minimal version selection.

The inference library operates on local assets and performs no downloads.
Preprocessing runs on CPU. Encoder/decoder sessions use the requested provider;
CUDA setup errors propagate. Some encoder operators remain on CPU under CUDA.
`Verbose` enables placement diagnostics. Threads defaults to one intra-op thread
per session; inter-op threads are one. CUDA uses explicit device selection,
`cudnn_conv_algo_search=DEFAULT`, `use_tf32=0`, and exact-request arena growth.

Metadata inspection creates temporary sessions with the pinned binding, so
initialization includes two graph loads per model. Requests reuse their sessions
and decoder tensors.

## ONNX Runtime ownership

ONNX Runtime has one environment per process, shared by all models.

- Set `Config.ORTLibrary` to a shared-library path or OS loader name to initialize
  through [ortenv](https://github.com/androiddrew/ortenv). All ortenv-backed
  components must use the same selector. The environment stays loaded until
  process exit.
- Leave `ORTLibrary` empty when the application has already initialized the
  environment. Keep it alive until all engines have closed.
- Close each engine to release its sessions and tensors. Engine closure never
  unloads the process-global environment.

## API contracts

The library follows [go-kokoro](https://github.com/androiddrew/go-kokoro)'s
`Config`, `New`, `Engine`, `Request`, `Result`, and `Close` pattern. Requests and
Close are serialized per engine; applications can create separate engines for
parallel work. Results own their Go buffers. Input samples must not be mutated
during a call. Cancellation is checked between native calls; an in-flight
native Run cannot be interrupted by this binding.

Parakeet v3 supports 25 languages with punctuation and capitalization. Token
timestamps describe model emission positions, not exact word boundaries.
Incremental delivery finalizes text, not recognition accuracy. Model emission
timestamps are approximate and can change with chunk settings.

## Development

```bash
go test ./...
go vet ./...
go test -race ./...
go build ./cmd/inference-probe
```

Unit tests require neither model assets nor a native runtime. Native lifecycle
tests opt in using `PARAKEET_TEST_ORT` and `PARAKEET_TEST_ASSETS` and default to
the frozen fixtures in `testdata/`; `PARAKEET_TEST_REFERENCE` is an optional
override. They run in subprocesses because ortenv
retains its environment. See [testdata/README.md](testdata/README.md) for fixture
generation and CPU/CUDA commands. CUDA needs both the `cuda` build tag and
`PARAKEET_TEST_CUDA_ORT`.

CI builds pinned ORT 1.22.2 on Linux and runs native INT8 parity, natural-speech,
streaming and lifecycle tests. Fixture checksums are verified in ordinary unit
tests; native CI requires no live TTS service or Python inference stack.
Natural-speech checks distinguish Go/Python implementation parity from model
accuracy. The three Greek examples have materially higher word error than the
other pilot languages, especially with INT8; the full results are recorded.

`cmd/inference-probe` accepts already-resampled little-endian float32 PCM and
prints JSON metadata, timings, text, and tokens. It accepts long recordings;
`--window`, `--overlap`, and `--single-pass` control windowing. `--stream` writes
`type: "chunk"` JSON lines followed by a final `type: "done"` metadata/summary
object. Failure after partial output exits nonzero without a success summary:

```bash
go run ./cmd/inference-probe --assets runtime-assets/parakeet \
    --ort libonnxruntime.so --provider cpu \
    --input artifacts/reference-fp32/clean-us.samples.f32

go run ./cmd/inference-probe --assets runtime-assets/parakeet \
    --ort libonnxruntime.so --provider cuda --stream \
    --window 30s --overlap 4s --input artifacts/long/recording.samples.f32
```

Original code is [Apache-2.0](LICENSE). See [THIRD_PARTY.md](THIRD_PARTY.md),
[NOTICE](NOTICE), and [ASSETS.md](ASSETS.md) for code/model attribution.

`cmd/inference-bench` and `tools/benchmark.py` record first/warm request timings,
RSS (including native allocations), Go heap, and sampled VRAM. Reproduction and
interpretation are in RELEASING.md; measured data is in `validation/benchmarks.json`.

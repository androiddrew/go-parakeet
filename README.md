# go-parakeet

`github.com/androiddrew/go-parakeet` (package `parakeet`) transcribes speech with
NVIDIA Parakeet TDT 0.6B v3 through ONNX Runtime. It accepts 16 kHz mono float32
PCM and produces text and token emission timestamps in 25 languages, with
punctuation and capitalization. Audio decoding, resampling and any CLI or
serving layer belong to the application.

You need an installed ONNX Runtime and three model downloads: the preprocessor
plus an FP32 (2.5 GB) or INT8 (670 MB) encoder and decoder; see
[ASSETS.md](ASSETS.md) for links and checksums. The vocabulary is built in.

```go
import (
    "context"
    "errors"
    parakeet "github.com/androiddrew/go-parakeet"
)

func transcribe(ctx context.Context, samples []float32) (result parakeet.Result, err error) {
    engine, err := parakeet.New(parakeet.Config{
        ORTLibrary:       "libonnxruntime.so", // or an explicit installed-library path
        PreprocessorPath: "/assets/parakeet/nemo128.onnx",
        EncoderPath:      "/assets/parakeet/encoder-model.onnx",
        DecoderPath:      "/assets/parakeet/decoder_joint-model.onnx",
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

Reuse an initialized engine across requests. For INT8, use
`encoder-model.int8.onnx` and `decoder_joint-model.int8.onnx`. FP32's
`encoder-model.onnx.data` must be beside its graph.

## API and guarantees

- `Transcribe` returns the complete text, tokens and finalized chunks. It
  returns no partial transcript on error.
- `TranscribeStream` is an alternative to `Transcribe`: it delivers finalized
  chunks to a callback without accumulating a whole-recording result.
- `Result.Tokens` retains raw SentencePiece IDs/pieces, recording-relative
  emission frames/timestamps, and predicted frame advances. These are model
  emission positions, not exact word boundaries, and can change with chunk settings.
- The embedded `Summary` reports duration, decoded windows, delivered chunks,
  finalized source coverage, effective window sizes and stage `Timings`.
- `ModelInfo` returns inspected tensor metadata for all three graphs.
  `Initialization` reports loading times and the loaded native runtime version.
- Results own their Go buffers. Input samples must not be mutated during a call.
- Input must be at least 20 ms of finite samples at `SampleRate`; otherwise
  `ErrInvalidAudio`. Samples are not rescaled, clipped, mixed or resampled.
- `VocabPath` overrides the embedded vocabulary only for a custom export.

## Long audio

`ChunkOptions{}` automatically uses **30-second windows with 4-second overlap**.
Audio fitting one window runs as a single pass. Customize the latency/context
tradeoff per request:

```go
result, err := engine.Transcribe(ctx, parakeet.Request{
    Samples: samples, SampleRate: parakeet.SampleRate,
    Chunking: parakeet.ChunkOptions{
        Window:  30 * time.Second,
        Overlap: 4 * time.Second,
    },
})
```

- `Window`: 2–300 seconds; zero selects `DefaultWindow` (30 s).
- `Overlap`: positive, at most half the window; zero selects
  `min(DefaultOverlap, Window/4)`.
- Durations round down to the 80 ms encoder grid. A positive overlap must round
  to at least one frame. Effective sizes are reported as sample counts in `Summary`.
- `SinglePass: true` disables windowing and rejects input exceeding
  `MaxWindowSamples` (300 seconds) with `ErrAudioTooLong`. Window/Overlap must
  then be zero. The cap is not a memory guarantee on every GPU.
- Invalid options return `ErrInvalidChunking`.

Each window is preprocessed independently and starts with fresh decoder state.
Seam reconciliation aligns whole SentencePiece words using timestamps and a
bounded word-alignment search. It prefers quiet regions for boundaries and falls
back to timestamp ownership when alignment is ambiguous; `ChunkResult.Seam`
reports `SeamAligned` or `SeamTimestamp` (empty for the first chunk). The
unresolved overlap and a 640 ms timestamp-drift margin are retained until the
next window is decoded. Small backwards timestamp differences at a splice are
clamped, retaining words and nondecreasing times.

A pathological unbroken hypothesis that cannot be finalized within two windows'
maximum token budget returns `ErrUnstableSeam`. This keeps memory bounded and
avoids silently cutting an unresolved word; already delivered chunks remain
valid. Increasing window context or splitting the input at pauses can help.

Windowing bounds intermediate inference memory. The caller still holds the
complete PCM input, and `Transcribe` retains the complete output. Smaller windows
change acoustic context and per-window normalization and can change recognition.
**Keep the default windows for INT8** unless shorter windows are validated on
your inputs: with 8-second windows the quantized model omitted speech in testing.

## Streaming text

Use `TranscribeStream` to consume stable text before the complete recording
finishes. Each callback receives one finalized chunk; text is never revised.

```go
// output is an application-supplied io.Writer.
summary, err := engine.TranscribeStream(ctx, parakeet.Request{
    Samples: samples, SampleRate: parakeet.SampleRate,
}, func(chunk parakeet.ChunkResult) error {
    // Text may begin with a space; concatenate verbatim.
    _, err := io.WriteString(output, chunk.Text)
    return err
})
// Handle err even if some chunks were already delivered. summary excludes callback time.
```

- Concatenating `chunk.Text` verbatim equals a successful `Transcribe` result
  with the same settings.
- Each chunk owns its token slice, which remains valid after further inference,
  errors and engine closure.
- `StartSample`/`EndSample` describe contiguous finalized source spans. They are
  progress boundaries, not word alignment. Empty-text chunks can report progress
  through silence.
- Streaming waits for a complete encoder window before delivering that window's
  stable prefix. Audio is supplied up front; this is not a live microphone API.

The callback is synchronous and applies backpressure. Queue text for a separate
consumer to overlap it with the next window. The engine creates no background
goroutines and retains no aggregate output in streaming mode.

**Errors and lifetime:** a callback error stops further inference; it is wrapped
with the chunk index and supports `errors.Is`/`errors.As`. Previously delivered
data remains valid on any error. `Summary` includes completed work and timings
on error; `DeliveredChunks` counts callbacks that returned nil and
`FinalizedSeconds` ends at the last such chunk. Context cancellation is checked
between native runs and around callbacks, including after the final callback.
Native inference cannot be interrupted mid-run. A blocking callback must observe
the context itself.

Requests and `Close` remain serialized for the entire stream, including callbacks.
**Do not call back into the same engine from a callback**; that can deadlock.
Close the engine after streaming returns. See `ExampleEngine_TranscribeStream`
for a complete producer/consumer example.

## ONNX Runtime ownership

ONNX Runtime has one environment per process, shared by every model in it.

- **Set `ORTLibrary`** (simplest): the engine initializes the environment through
  [`github.com/androiddrew/ortenv`](https://github.com/androiddrew/ortenv) on first
  use. The environment stays loaded until process exit, so closing the last engine
  never unloads the runtime. Every component using ortenv in the process must use
  the same library selector, and nothing may destroy the environment ortenv owns.
- **Leave `ORTLibrary` empty** when your application already manages ONNX Runtime
  itself (directly through `onnxruntime_go` or through ortenv). Initialize the
  environment before `New` and destroy it only after `Close`; otherwise `New`
  returns `ErrNotInitialized`:

```go
ort.SetSharedLibraryPath("libonnxruntime.so")
if err := ort.InitializeEnvironment(); err != nil { return err }
defer ort.DestroyEnvironment() // runs after engine.Close

engine, err := parakeet.New(parakeet.Config{
    PreprocessorPath: preprocessorPath, EncoderPath: encoderPath, DecoderPath: decoderPath,
})
if err != nil { return err }
defer engine.Close()
```

This makes go-parakeet usable in the same process as
[go-kokoro](https://github.com/androiddrew/go-kokoro) and
[go-g2p](https://github.com/androiddrew/go-g2p).

## Providers and lifetime

Requires Go 1.25 or newer with CGO. Tested on Linux/amd64, Go 1.25.0 and 1.26.4,
with ORT 1.22.0 (official release) and 1.22.2 (source build) on CPU, and ORT
1.22.2 with CUDA 12.0/cuDNN 9.8 for GPU tests. These describe the known tested
configuration. Native runtime installation is user-managed and must provide the
resolved binding's C API; there is no exact version-string restriction.
CUDA/cuDNN requirements follow the selected native runtime/provider build.

Preprocessing always runs on CPU. The encoder and decoder use the requested
provider; CUDA setup errors propagate, with no CPU fallback. `Provider:
parakeet.CUDA` uses an explicit device ID, `cudnn_conv_algo_search=DEFAULT`,
`use_tf32=0` and exact-request arena growth. Some encoder operators remain on
CPU under CUDA; `Verbose` exposes node placement. CUDA is tested with FP32
assets. Each session uses one intra/inter-op thread by default, sequential graph
execution and full optimization. Set `Threads` for intra-op parallelism.

Close every engine. Requests and Close on an engine are serialized; use separate
engines for parallel work. Metadata inspection currently creates a temporary
session before each inference session, so loading timings include **two graph
loads per model**. Requests reuse their sessions and decoder tensors.

On a small FLEURS sample, Greek had materially higher word error than English,
French, German, Spanish and Russian, especially with INT8. Validate accuracy on
your own audio.

## Development

```bash
go test ./...
go vet ./...
go build ./cmd/inference-probe
```

Set `PARAKEET_TEST_ASSETS` to the directory holding the model files, and
`PARAKEET_TEST_ORT` to an ONNX Runtime library, then run `go test -race ./...`.
Unit tests need neither. `PARAKEET_TEST_PRECISION=int8` selects the INT8 assets
and baselines (default `fp32`). Once native testing is enabled, incomplete
configuration fails instead of skipping. Tests that load ONNX Runtime each run
in a fresh subprocess because ortenv keeps the environment loaded until process
exit. The fixtures in `testdata/` hold the exact audio, tensors and token
decisions expected from a Python reference; see
[testdata/README.md](testdata/README.md). CI runs everything against the
official ONNX Runtime 1.22.0 release with FP32 assets. INT8 parity baselines
were recorded on an AVX2 CPU; quantized kernels differ by CPU feature set, so
INT8 tensor parity can fail on other hardware (for example AVX-512 VNNI) even
when transcription is unaffected.

`PARAKEET_TEST_LONG=1` additionally transcribes a recording over 300 seconds.
CUDA tests need both the `cuda` build tag and `PARAKEET_TEST_CUDA_ORT` (FP32):

```bash
PARAKEET_TEST_CUDA_ORT=/path/to/cuda/libonnxruntime.so \
PARAKEET_TEST_ASSETS=/assets/parakeet \
    go test -race -tags=cuda -count=1 -run '^TestCUDA' .
```

`cmd/inference-probe` reads little-endian 16 kHz mono float32 PCM and prints JSON
metadata, timings, text and tokens. `--window`, `--overlap` and `--single-pass`
control windowing; `--stream` writes `type: "chunk"` JSON lines followed by a
final `type: "done"` summary. It is an optional diagnostic tool.

See [ASSETS.md](ASSETS.md), [THIRD_PARTY.md](THIRD_PARTY.md), LICENSE and NOTICE.

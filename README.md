# go-parakeet

`github.com/androiddrew/go-parakeet` (package `parakeet`) is an inference library
for NVIDIA Parakeet TDT 0.6B v3 through ONNX Runtime. It accepts 16 kHz mono
float32 PCM and produces text and token emission timestamps. Applications supply
audio decoding/resampling and their own CLI or serving layer.

## Development status

Implementation is in progress; v0.1.0 has not been released. **Single-pass
transcription is implemented**, including CPU/CUDA sessions, token timing,
metadata, initialization/request timings, and native lifecycle tests.
Long-audio windowing and finalized-chunk callbacks remain required v0.1.0
milestones. The current API accepts 20 ms to 300 seconds of audio and rejects
longer requests; 300 seconds is a conservative API cap, not a measured memory
guarantee on every machine.

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

`Result.Tokens` retains raw SentencePiece IDs/pieces, emission frames, timestamps,
and predicted frame advances. `ModelInfo()` returns owned tensor metadata and
`Initialization()` reports loading times and the actual native runtime version.
An error returns no partial transcript. Invalid/nonfinite input fails before
inference; samples are not rescaled or clipped.

## Runtime and assets

Requires Go 1.25 or newer, CGO, an installed ONNX Runtime, and separately
downloaded model files. The Go binding targets C API 22; native ONNX Runtime
1.22.2 is the release compatibility target. Tested so far: Linux/amd64, Go
1.26.4, native ORT **1.22.0**, CPU FP32/INT8, and CUDA FP32 on an RTX 4090. See
[VALIDATION.md](VALIDATION.md) for evidence and outstanding checks; 1.22.2 has
not yet been exercised. These are tested configurations, not exact version-string
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
The planned incremental API delivers finalized text from complete input audio.

## Development

```bash
go test ./...
go vet ./...
go test -race ./...
go build ./cmd/inference-probe
```

Unit tests require neither model assets nor a native runtime. Native lifecycle
tests opt in using `PARAKEET_TEST_ORT` and `PARAKEET_TEST_ASSETS`; parity tests
also require `PARAKEET_TEST_REFERENCE`. They run in subprocesses because ortenv
retains its environment. See [testdata/README.md](testdata/README.md) for fixture
generation and CPU/CUDA commands. CUDA needs both the `cuda` build tag and
`PARAKEET_TEST_CUDA_ORT`.

`cmd/inference-probe` accepts already-resampled little-endian float32 PCM and
prints JSON metadata, timings, text, and tokens:

```bash
go run ./cmd/inference-probe --assets runtime-assets/parakeet \
    --ort libonnxruntime.so --provider cpu \
    --input artifacts/reference-fp32/clean-us.samples.f32
```

Original code is [Apache-2.0](LICENSE). See [THIRD_PARTY.md](THIRD_PARTY.md),
[NOTICE](NOTICE), and [ASSETS.md](ASSETS.md) for code/model attribution.

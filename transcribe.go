package parakeet

import (
	"context"
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/androiddrew/go-parakeet/internal/assets"
	"github.com/androiddrew/go-parakeet/internal/chunking"
	"github.com/androiddrew/go-parakeet/internal/decoder"
	ort "github.com/yalue/onnxruntime_go"
)

func validateFinite(samples []float32) error {
	for i, value := range samples {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return fmt.Errorf("nonfinite value at index %d", i)
		}
	}
	return nil
}

func validateAudio(ctx context.Context, request Request) error {
	if request.SampleRate != SampleRate {
		return fmt.Errorf("%w: require %d Hz mono PCM", ErrInvalidAudio, SampleRate)
	}
	// The preprocessor normalizes using sample variance over mel frames. At
	// least two valid frames are necessary; shorter input is not silently padded.
	if len(request.Samples) < 320 {
		return fmt.Errorf("%w: require at least 320 samples (20 ms)", ErrInvalidAudio)
	}
	if request.Chunking.SinglePass && len(request.Samples) > MaxWindowSamples {
		return fmt.Errorf("%w: maximum %d samples", ErrAudioTooLong, MaxSamples)
	}
	for i, sample := range request.Samples {
		if i%16384 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		if math.IsNaN(float64(sample)) || math.IsInf(float64(sample), 0) {
			return fmt.Errorf("%w: nonfinite value at index %d", ErrInvalidAudio, i)
		}
	}
	return nil
}

// Transcribe collects finalized chunks into an owned aggregate result. Long
// recordings are windowed by default. On error it returns no partial transcript.
func (e *Engine) Transcribe(ctx context.Context, request Request) (Result, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed || e.decoder == nil {
		return Result{}, ErrClosed
	}
	return collectTranscript(ctx, request, e.runWindow)
}

// runWindow is called with the engine lock held and a validated, bounded input.
// Each independent window resets the decoder; overlapping state is never reused.
func (e *Engine) runWindow(ctx context.Context, samples []float32, offset int) ([]Token, Timings, error) {
	var timings Timings
	if err := ctx.Err(); err != nil {
		return nil, timings, err
	}
	start := time.Now()
	features, err := runSequence(e.preprocessor, samples, ort.NewShape(1, int64(len(samples))), len(samples), assets.Features)
	timings.PreprocessingSeconds = time.Since(start).Seconds()
	if err != nil {
		return nil, timings, fmt.Errorf("preprocess: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return nil, timings, err
	}
	start = time.Now()
	encoded, err := runSequence(e.encoder, features.data, ort.NewShape(1, assets.Features, int64(features.stride)), features.length, assets.EncoderDim)
	timings.EncoderSeconds = time.Since(start).Seconds()
	if err != nil {
		return nil, timings, fmt.Errorf("encode: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return nil, timings, err
	}
	start = time.Now()
	e.decoder.reset()
	decoded, err := decoder.Greedy(ctx, encoded.length, assets.VocabSize, e.vocab.Blank, assets.Durations,
		func(frame, previous int) ([]float32, error) { return e.decoder.step(encoded, frame, previous) }, e.decoder.commit)
	timings.DecoderSeconds = time.Since(start).Seconds()
	if err != nil {
		return nil, timings, err
	}
	start = time.Now()
	tokens := make([]Token, len(decoded))
	for i, token := range decoded {
		piece := e.vocab.Pieces[token.ID]
		frame := offset/chunking.FrameSamples + token.Frame
		tokens[i] = Token{token.ID, piece, frame, float64(frame) * 0.08, token.Advance}
	}
	timings.AssemblySeconds = time.Since(start).Seconds()
	if err := ctx.Err(); err != nil {
		return nil, timings, err
	}
	return tokens, timings, nil
}

type sequence struct {
	data           []float32
	stride, length int
}

// runSequence keeps physical tensor width and valid length distinct. Returned
// data is copied before native outputs are destroyed and contains no native views.
func runSequence(session *ort.DynamicAdvancedSession, data []float32, shape ort.Shape, length, channels int) (_ sequence, err error) {
	input, err := ort.NewTensor(shape, data)
	if err != nil {
		return sequence{}, err
	}
	defer func() { err = errors.Join(err, input.Destroy()) }()
	lens, err := ort.NewTensor(ort.NewShape(1), []int64{int64(length)})
	if err != nil {
		return sequence{}, err
	}
	defer func() { err = errors.Join(err, lens.Destroy()) }()
	outputs := []ort.Value{nil, nil}
	defer func() {
		for _, value := range outputs {
			if value != nil {
				err = errors.Join(err, value.Destroy())
			}
		}
	}()
	if err = session.Run([]ort.Value{input, lens}, outputs); err != nil {
		return sequence{}, err
	}
	values, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return sequence{}, errors.New("expected float32 sequence")
	}
	valid, ok := outputs[1].(*ort.Tensor[int64])
	if !ok || len(valid.GetShape()) != 1 || valid.GetShape()[0] != 1 {
		return sequence{}, errors.New("expected one int64 sequence length")
	}
	dims := values.GetShape()
	if len(dims) != 3 || dims[0] != 1 || dims[1] != int64(channels) || dims[2] <= 0 || dims[2] > int64(MaxSamples) {
		return sequence{}, fmt.Errorf("invalid sequence shape %v", dims)
	}
	count := valid.GetData()[0]
	if count <= 0 || count > dims[2] {
		return sequence{}, fmt.Errorf("invalid sequence length %d for %v", count, dims)
	}
	if err = validateFinite(values.GetData()); err != nil {
		return sequence{}, err
	}
	return sequence{append([]float32(nil), values.GetData()...), int(dims[2]), int(count)}, nil
}

package parakeet

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/androiddrew/go-parakeet/internal/assets"
	"github.com/androiddrew/go-parakeet/internal/chunking"
	"github.com/androiddrew/go-parakeet/internal/decoder"
)

// TranscribeStream delivers finalized, caller-owned chunks synchronously. Text
// is final and concatenates verbatim to Transcribe's result. Empty text chunks
// can report progress through silence. The library retains no aggregate output.
//
// A callback error stops inference; errors.Is/errors.As preserve its identity.
// Previously delivered chunks survive all errors and engine closure. Summary
// counts callbacks that returned nil, even if cancellation follows the last one.
// Callbacks must observe ctx themselves when blocking, and must not call this
// engine's inference methods or Close. Requests and Close stay serialized during callbacks.
// Native Run cannot be interrupted mid-call. The complete input audio is supplied
// up front and must remain unmodified until this method returns.
func (e *Engine) TranscribeStream(ctx context.Context, request Request, emit func(ChunkResult) error) (Summary, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed || e.decoder == nil {
		return Summary{}, ErrClosed
	}
	return streamTranscript(ctx, request, emit, e.runWindow)
}

type windowRunner func(context.Context, []float32, int) ([]Token, Timings, error)

func planAudio(samples int, options ChunkOptions) (chunking.Plan, error) {
	if options.SinglePass {
		if options.Window != 0 || options.Overlap != 0 {
			return chunking.Plan{}, fmt.Errorf("%w: SinglePass requires zero durations", ErrInvalidChunking)
		}
		if samples > MaxWindowSamples {
			return chunking.Plan{}, ErrAudioTooLong
		}
		return chunking.New(samples, MaxWindowSamples, 0)
	}
	window, overlap := options.Window, options.Overlap
	if window == 0 {
		window = DefaultWindow
	}
	if window < 2*time.Second || window > 300*time.Second || overlap < 0 {
		return chunking.Plan{}, fmt.Errorf("%w: window must be 2..300 seconds and overlap nonnegative", ErrInvalidChunking)
	}
	const frameTime = 80 * time.Millisecond
	size := int(window/frameTime) * chunking.FrameSamples
	if overlap == 0 {
		overlap = min(DefaultOverlap, window/4)
	}
	// Check duration bounds before integer conversion/multiplication so even
	// enormous duration values cannot overflow sample arithmetic on 32-bit Go.
	if overlap < frameTime || overlap > window/2 {
		return chunking.Plan{}, fmt.Errorf("%w: overlap must be 80 ms..half the window", ErrInvalidChunking)
	}
	overlapSamples := int(overlap / frameTime)
	if overlapSamples > size/chunking.FrameSamples/2 {
		return chunking.Plan{}, fmt.Errorf("%w: rounded overlap exceeds half the rounded window", ErrInvalidChunking)
	}
	p, err := chunking.New(samples, size, overlapSamples*chunking.FrameSamples)
	if err != nil {
		return chunking.Plan{}, fmt.Errorf("%w: %v", ErrInvalidChunking, err)
	}
	return p, nil
}

func (t *Timings) add(other Timings) {
	t.ValidationSeconds += other.ValidationSeconds
	t.PreprocessingSeconds += other.PreprocessingSeconds
	t.EncoderSeconds += other.EncoderSeconds
	t.DecoderSeconds += other.DecoderSeconds
	t.AssemblySeconds += other.AssemblySeconds
}

func streamTranscript(ctx context.Context, request Request, emit func(ChunkResult) error, run windowRunner) (Summary, error) {
	var summary Summary
	if emit == nil {
		return summary, errors.New("transcription callback is required")
	}
	if err := ctx.Err(); err != nil {
		return summary, err
	}
	start := time.Now()
	err := validateAudio(ctx, request)
	var plan chunking.Plan
	if err == nil {
		plan, err = planAudio(len(request.Samples), request.Chunking)
	}
	summary.Timings.ValidationSeconds = time.Since(start).Seconds()
	if err != nil {
		return summary, err
	}
	summary.AudioSeconds = float64(plan.Total) / SampleRate
	summary.WindowSamples, summary.OverlapSamples = plan.Size, plan.Overlap
	var pending []Token
	var text assets.TextDecoder
	endSample, lastFrame := 0, 0
	for index := range plan.Count() {
		if err := ctx.Err(); err != nil {
			return summary, err
		}
		window := plan.At(index)
		tokens, timings, err := run(ctx, request.Samples[window.Start:window.End], window.Start)
		summary.Timings.add(timings)
		if err != nil {
			return summary, fmt.Errorf("window %d: %w", index, err)
		}
		summary.Windows++
		summary.ProcessedSeconds = float64(window.End) / SampleRate
		if err := ctx.Err(); err != nil {
			return summary, err
		}
		start = time.Now()
		method := ""
		if index == 0 {
			pending = tokens
		} else {
			from, to := window.Start/chunking.FrameSamples, plan.At(index-1).End/chunking.FrameSamples
			boundary := seamBoundary(request.Samples, from, to)
			pending, method = reconcile(pending, tokens, from, to, boundary)
		}
		// Window-local alignments can differ slightly. Keep every selected word,
		// clamping backwards timestamp drift rather than deleting its tokens.
		frame := lastFrame
		for i := range pending {
			pending[i].Frame = max(frame, pending[i].Frame)
			pending[i].StartSeconds = float64(pending[i].Frame) * 0.08
			frame = pending[i].Frame
		}
		cut, finalizedEnd := len(pending), plan.Total
		if index+1 < plan.Count() {
			nextStart := plan.At(index + 1).Start
			// Keep a drift margin as well as the acoustic overlap so small
			// cross-window timestamp shifts cannot revise an emitted word.
			cut = safePrefix(pending, nextStart/chunking.FrameSamples-seamDriftFrames)
			finalizedEnd = nextStart
			if cut < len(pending) {
				finalizedEnd = min(finalizedEnd, pending[cut].Frame*chunking.FrameSamples)
			}
			finalizedEnd = max(endSample, finalizedEnd)
		}
		// A pathological unbroken word can otherwise concatenate across windows
		// indefinitely. Bound retained hypotheses to two worst-case decoder
		// windows and fail explicitly rather than retaining the entire recording
		// or silently chopping an unresolved word. One extra frame covers padding.
		maxPending := 2 * (plan.Size/chunking.FrameSamples + 1) * decoder.MaxSymbolsPerFrame
		if len(pending)-cut > maxPending {
			summary.Timings.AssemblySeconds += time.Since(start).Seconds()
			return summary, fmt.Errorf("window %d: %w", index, ErrUnstableSeam)
		}
		owned := append([]Token{}, pending[:cut]...)
		pieces := make([]string, len(owned))
		for i, token := range owned {
			pieces[i] = token.Piece
		}
		chunk := ChunkResult{Index: index, StartSample: endSample, EndSample: finalizedEnd, Tokens: owned, Text: text.Append(pieces), Seam: method}
		if cut > 0 {
			lastFrame = pending[cut-1].Frame
		}
		// Copy the retained tail too: callbacks may mutate their owned tokens and
		// we don't retain a backing array containing an already-delivered prefix.
		pending = append([]Token(nil), pending[cut:]...)
		summary.Timings.AssemblySeconds += time.Since(start).Seconds()
		if err := ctx.Err(); err != nil {
			return summary, err
		}
		if err := emit(chunk); err != nil {
			return summary, fmt.Errorf("deliver chunk %d: %w", index, err)
		}
		summary.DeliveredChunks++
		endSample = finalizedEnd
		summary.FinalizedSeconds = float64(endSample) / SampleRate
		if err := ctx.Err(); err != nil {
			return summary, err
		}
	}
	return summary, nil
}

func collectTranscript(ctx context.Context, request Request, run windowRunner) (Result, error) {
	result := Result{Tokens: []Token{}, Chunks: []ChunkResult{}}
	var text strings.Builder
	var collectionSeconds float64
	summary, err := streamTranscript(ctx, request, func(chunk ChunkResult) error {
		start := time.Now()
		text.WriteString(chunk.Text)
		result.Tokens = append(result.Tokens, chunk.Tokens...)
		result.Chunks = append(result.Chunks, chunk)
		collectionSeconds += time.Since(start).Seconds()
		return nil
	}, run)
	if err != nil {
		return Result{}, err
	}
	summary.Timings.AssemblySeconds += collectionSeconds
	result.Text, result.Summary = text.String(), summary
	return result, nil
}

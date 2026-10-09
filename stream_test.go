package parakeet

import (
	"context"
	"errors"
	"math"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/androiddrew/go-parakeet/internal/assets"
	"github.com/androiddrew/go-parakeet/internal/chunking"
)

func testRecording() (Request, []Token) {
	request := Request{Samples: make([]float32, 64*chunking.FrameSamples), SampleRate: SampleRate,
		Chunking: ChunkOptions{Window: 2 * time.Second, Overlap: 800 * time.Millisecond}}
	var tokens []Token
	for i, piece := range []string{"One", "very,", "very", "important", "bird", "opened", "the", "blue", "door", "again", "today."} {
		frame := 2 + i*6
		tokens = append(tokens, Token{ID: i, Piece: "▁" + piece, Frame: frame, StartSeconds: float64(frame) * 0.08, Advance: 1})
	}
	return request, tokens
}

func tapeRunner(tape []Token, calls *int) windowRunner {
	return func(_ context.Context, samples []float32, offset int) ([]Token, Timings, error) {
		if calls != nil {
			*calls++
		}
		var result []Token
		for _, token := range tape {
			position := token.Frame * chunking.FrameSamples
			if position >= offset && position < offset+len(samples) {
				result = append(result, token)
			}
		}
		return result, Timings{}, nil
	}
}

func TestStreamingMatchesAggregate(t *testing.T) {
	request, tape := testRecording()
	got, err := collectTranscript(context.Background(), request, tapeRunner(tape, nil))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.Tokens, tape) {
		t.Fatalf("seam altered the utterance: %+v", got.Tokens)
	}
	var pieces []string
	for _, token := range tape {
		pieces = append(pieces, token.Piece)
	}
	if got.Text != assets.Text(pieces) {
		t.Fatal(got.Text)
	}
	var chunks []ChunkResult
	var text strings.Builder
	end, lastFrame := 0, -1
	summary, err := streamTranscript(context.Background(), request, func(chunk ChunkResult) error {
		if chunk.Index != len(chunks) || chunk.StartSample != end || chunk.EndSample < end {
			t.Fatalf("noncontiguous delivery %+v", chunk)
		}
		for _, token := range chunk.Tokens {
			if token.Frame < lastFrame {
				t.Fatal("backwards token time")
			}
			lastFrame = token.Frame
		}
		end = chunk.EndSample
		chunks = append(chunks, chunk)
		text.WriteString(chunk.Text)
		return nil
	}, tapeRunner(tape, nil))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(chunks, got.Chunks) || text.String() != got.Text || summary.DeliveredChunks != len(chunks) || summary.Windows != len(chunks) || end != len(request.Samples) {
		t.Fatalf("inconsistent stream: %+v", summary)
	}
}

func TestChunkOptions(t *testing.T) {
	for _, bad := range []ChunkOptions{
		{Window: -1}, {Window: time.Second}, {Window: 301 * time.Second}, {Overlap: -1},
		{Overlap: time.Millisecond}, {Overlap: time.Hour},
		{Window: 2 * time.Second, Overlap: 1100 * time.Millisecond},
		{SinglePass: true, Window: time.Second}, {SinglePass: true, Overlap: time.Second},
	} {
		if _, err := planAudio(320, bad); !errors.Is(err, ErrInvalidChunking) {
			t.Fatalf("%+v: %v", bad, err)
		}
	}
	p, err := planAudio(MaxSamples+1, ChunkOptions{})
	if err != nil || p.Size != 30*SampleRate || p.Overlap != 4*SampleRate || p.Count() < 2 {
		t.Fatalf("%+v %v", p, err)
	}
	p, err = planAudio(320, ChunkOptions{Window: 5 * time.Second})
	if err != nil || p.Size != 62*chunking.FrameSamples || p.Overlap != 15*chunking.FrameSamples {
		t.Fatalf("rounding: %+v %v", p, err)
	}
	if _, err := planAudio(MaxSamples+1, ChunkOptions{SinglePass: true}); !errors.Is(err, ErrAudioTooLong) {
		t.Fatal(err)
	}
}

func TestValidatesBeforeAnyDelivery(t *testing.T) {
	request, tape := testRecording()
	request.Samples[len(request.Samples)-1] = float32(math.NaN())
	calls := 0
	_, err := streamTranscript(context.Background(), request, func(ChunkResult) error { t.Fatal("delivered invalid request"); return nil }, tapeRunner(tape, &calls))
	if !errors.Is(err, ErrInvalidAudio) || calls != 0 {
		t.Fatalf("%v, calls %d", err, calls)
	}
	request.Samples[len(request.Samples)-1] = 0
	if _, err := streamTranscript(context.Background(), request, nil, tapeRunner(tape, &calls)); err == nil || calls != 0 {
		t.Fatal("accepted nil callback")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := streamTranscript(ctx, request, func(ChunkResult) error { return nil }, tapeRunner(tape, &calls)); !errors.Is(err, context.Canceled) || calls != 0 {
		t.Fatal(err)
	}
}

type callbackFailure struct{}

func (*callbackFailure) Error() string { return "consumer failed" }

func TestStreamingFailures(t *testing.T) {
	request, tape := testRecording()
	for _, failAt := range []int{0, 1, 3} {
		calls := 0
		want := &callbackFailure{}
		var delivered []ChunkResult
		summary, err := streamTranscript(context.Background(), request, func(chunk ChunkResult) error {
			delivered = append(delivered, chunk)
			if chunk.Index == failAt {
				return want
			}
			return nil
		}, tapeRunner(tape, &calls))
		var typed *callbackFailure
		if !errors.Is(err, want) || !errors.As(err, &typed) || calls != failAt+1 || summary.DeliveredChunks != failAt || summary.Windows != failAt+1 {
			t.Fatalf("callback failure: %+v %v calls=%d", summary, err, calls)
		}
		if len(delivered) != failAt+1 {
			t.Fatal("lost delivered data")
		}
	}
	want := errors.New("encoder failed")
	calls := 0
	run := tapeRunner(tape, nil)
	summary, err := streamTranscript(context.Background(), request, func(ChunkResult) error { return nil }, func(ctx context.Context, samples []float32, offset int) ([]Token, Timings, error) {
		calls++
		if calls == 2 {
			return nil, Timings{EncoderSeconds: 0.25}, want
		}
		return run(ctx, samples, offset)
	})
	if !errors.Is(err, want) || summary.Windows != 1 || summary.DeliveredChunks != 1 || summary.Timings.EncoderSeconds != 0.25 {
		t.Fatalf("native failure: %+v %v", summary, err)
	}
}

func TestCancellationAroundCallbacks(t *testing.T) {
	request, tape := testRecording()
	for _, cancelAt := range []int{0, 3} { // including cancellation after final delivery
		ctx, cancel := context.WithCancel(context.Background())
		calls := 0
		summary, err := streamTranscript(ctx, request, func(chunk ChunkResult) error {
			if chunk.Index == cancelAt {
				cancel()
			}
			return nil
		}, tapeRunner(tape, &calls))
		cancel()
		if !errors.Is(err, context.Canceled) || calls != cancelAt+1 || summary.DeliveredChunks != cancelAt+1 {
			t.Fatalf("%+v %v calls=%d", summary, err, calls)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	result, err := collectTranscript(ctx, request, func(context.Context, []float32, int) ([]Token, Timings, error) { cancel(); return tape, Timings{}, nil })
	if !errors.Is(err, context.Canceled) || result.Text != "" || result.Tokens != nil {
		t.Fatalf("aggregate leaked partial result: %+v %v", result, err)
	}
}

func TestCallbackOwnershipAndBackpressure(t *testing.T) {
	request, tape := testRecording()
	var calls atomic.Int32
	entered, release, done := make(chan struct{}), make(chan struct{}), make(chan error, 1)
	run := tapeRunner(tape, nil)
	go func() {
		start := time.Now()
		var callbackTime time.Duration
		summary, err := streamTranscript(context.Background(), request, func(chunk ChunkResult) error {
			begin := time.Now()
			if chunk.Index == 0 {
				close(entered)
				<-release
			}
			for i := range chunk.Tokens {
				chunk.Tokens[i].Piece = "corrupt"
				chunk.Tokens[i].Frame = -100
			}
			callbackTime += time.Since(begin)
			return nil
		}, func(ctx context.Context, samples []float32, offset int) ([]Token, Timings, error) {
			calls.Add(1)
			return run(ctx, samples, offset)
		})
		if measured := summary.Timings.ValidationSeconds + summary.Timings.AssemblySeconds; measured > (time.Since(start)-callbackTime).Seconds()+0.001 {
			done <- errors.New("callback time included in inference timings")
			return
		}
		done <- err
	}()
	<-entered
	if calls.Load() != 1 {
		t.Error("inference ran ahead of blocked consumer")
	}
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if tape[0].Piece == "corrupt" {
		t.Fatal("callback aliased source tokens")
	}
	got, err := collectTranscript(context.Background(), request, tapeRunner(tape, nil))
	if err != nil || strings.Contains(got.Text, "corrupt") {
		t.Fatalf("ownership: %v %+v", err, got)
	}
}

func TestSilenceAndShortTail(t *testing.T) {
	request, _ := testRecording()
	request.Samples = make([]float32, 2*SampleRate+1)
	got, err := collectTranscript(context.Background(), request, tapeRunner(nil, nil))
	if err != nil || got.Text != "" || len(got.Chunks) != 2 || got.FinalizedSeconds != got.AudioSeconds {
		t.Fatalf("%+v %v", got, err)
	}
	for _, chunk := range got.Chunks {
		if len(chunk.Tokens) != 0 {
			t.Fatal(chunk)
		}
	}
}

func TestUnbrokenHypothesisIsBounded(t *testing.T) {
	request, _ := testRecording()
	request.Samples = make([]float32, 12*SampleRate)
	calls := 0
	summary, err := streamTranscript(context.Background(), request, func(ChunkResult) error { return nil }, func(_ context.Context, samples []float32, offset int) ([]Token, Timings, error) {
		calls++
		var tokens []Token
		for frame := range len(samples) / chunking.FrameSamples {
			for range 10 {
				tokens = append(tokens, Token{Piece: "x", Frame: offset/chunking.FrameSamples + frame})
			}
		}
		return tokens, Timings{}, nil
	})
	if !errors.Is(err, ErrUnstableSeam) || calls > 3 || summary.DeliveredChunks != calls-1 {
		t.Fatalf("unbounded hypothesis: calls=%d summary=%+v err=%v", calls, summary, err)
	}
}

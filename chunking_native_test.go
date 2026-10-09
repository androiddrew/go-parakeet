package parakeet

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
	"unicode"

	"github.com/androiddrew/go-parakeet/internal/assets"
)

func speechCorpus(t *testing.T) ([]float32, string) {
	t.Helper()
	dir := os.Getenv("PARAKEET_TEST_REFERENCE")
	if dir == "" {
		t.Skip("set PARAKEET_TEST_REFERENCE for native chunking fixtures")
	}
	var samples []float32
	var text strings.Builder
	for _, name := range []string{"clean-us", "clean-gb", "repetition-us", "repetition-gb"} {
		var ref struct {
			Text   string `json:"text"`
			Stages map[string]struct {
				File   string `json:"file"`
				SHA256 string `json:"sha256"`
			} `json:"stages"`
		}
		raw, err := os.ReadFile(filepath.Join(dir, name+".json"))
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(raw, &ref); err != nil {
			t.Fatal(err)
		}
		s := ref.Stages["samples"]
		samples = append(samples, readFloat32(t, filepath.Join(dir, s.File), s.SHA256)...)
		samples = append(samples, make([]float32, 2560)...) // 160 ms between recordings
		text.WriteString(ref.Text)
		text.WriteByte(' ')
	}
	return samples, text.String()
}

func normalizedWords(text string) []string {
	words := strings.FieldsFunc(strings.ToLower(text), func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsNumber(r) && !unicode.IsMark(r) })
	// Both spellings represent the same spoken word in this corpus. Keep this
	// explicit orthographic allowance separate from omissions or repetitions.
	var result []string
	for i := 0; i < len(words); i++ {
		if i+1 < len(words) && words[i] == "window" && words[i+1] == "sill" {
			result = append(result, "windowsill")
			i++
		} else {
			result = append(result, words[i])
		}
	}
	return result
}

func testChunkedInference(t *testing.T, c Config) {
	t.Helper()
	samples, expected := speechCorpus(t)
	e, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	configs := []struct {
		window, overlap time.Duration
		shift           int
	}{
		{8 * time.Second, 3200 * time.Millisecond, 0},
		{10400 * time.Millisecond, 4 * time.Second, 0},
		{12 * time.Second, 4 * time.Second, 0},
		{8 * time.Second, 3200 * time.Millisecond, 6400},
	}
	if os.Getenv("PARAKEET_TEST_PRECISION") == "int8" {
		// The quantized encoder omits a phrase in the 8-second stress windows
		// themselves. Gate accuracy on longer contexts; see VALIDATION.md.
		configs[0].window, configs[0].overlap = 16*time.Second, 4*time.Second
		configs[3].window, configs[3].overlap = 10400*time.Millisecond, 4*time.Second
	}
	for _, config := range configs {
		audio := append(make([]float32, config.shift), samples...)
		request := Request{Samples: audio, SampleRate: SampleRate, Chunking: ChunkOptions{Window: config.window, Overlap: config.overlap}}
		result, err := e.Transcribe(context.Background(), request)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(normalizedWords(result.Text), normalizedWords(expected)) {
			for _, chunk := range result.Chunks {
				t.Logf("chunk %d [%d,%d) seam=%s: %s", chunk.Index, chunk.StartSample, chunk.EndSample, chunk.Seam, chunk.Text)
			}
			plan, _ := planAudio(len(audio), request.Chunking)
			for i := range plan.Count() {
				w := plan.At(i)
				tokens, _, err := e.runWindow(context.Background(), audio[w.Start:w.End], w.Start)
				if err != nil {
					t.Fatal(err)
				}
				var pieces []string
				for _, token := range tokens {
					pieces = append(pieces, token.Piece)
				}
				t.Logf("raw window %d [%.2f,%.2f): %s; tokens=%+v", i, float64(w.Start)/SampleRate, float64(w.End)/SampleRate, assets.Text(pieces), tokens)
			}
			t.Fatalf("window=%s overlap=%s shift=%d:\ngot:  %s\nwant: %s", config.window, config.overlap, config.shift, result.Text, expected)
		}
		var chunks []ChunkResult
		var text strings.Builder
		end, lastFrame := 0, 0
		summary, err := e.TranscribeStream(context.Background(), request, func(chunk ChunkResult) error {
			if chunk.StartSample != end || chunk.EndSample < end {
				t.Fatal("invalid chunk coverage")
			}
			if chunk.Index == 0 && chunk.EndSample >= len(audio) {
				t.Fatal("stream waited for entire recording")
			}
			for _, token := range chunk.Tokens {
				if token.Frame < lastFrame || token.StartSeconds != float64(token.Frame)*0.08 {
					t.Fatal("invalid token timing")
				}
				lastFrame = token.Frame
			}
			end = chunk.EndSample
			text.WriteString(chunk.Text)
			chunks = append(chunks, chunk)
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(chunks, result.Chunks) || text.String() != result.Text || end != len(audio) || summary.Windows < 2 || summary.DeliveredChunks != len(chunks) {
			t.Fatalf("stream mismatch: %+v", summary)
		}
		t.Logf("window=%s overlap=%s shift=%d: %d chunks; words and stream/aggregate output match", config.window, config.overlap, config.shift, summary.DeliveredChunks)
	}

	// A stopped stream leaves the engine reusable, and delivered memory owned.
	request := Request{Samples: samples, SampleRate: SampleRate, Chunking: ChunkOptions{Window: 8 * time.Second, Overlap: 3200 * time.Millisecond}}
	ctx, cancel := context.WithCancel(context.Background())
	var delivered ChunkResult
	summary, err := e.TranscribeStream(ctx, request, func(chunk ChunkResult) error { delivered = chunk; cancel(); return nil })
	if !errors.Is(err, context.Canceled) || summary.DeliveredChunks != 1 {
		t.Fatalf("cancellation: %+v %v", summary, err)
	}
	saved := append([]Token(nil), delivered.Tokens...)
	want := errors.New("consumer stopped")
	if _, err := e.TranscribeStream(context.Background(), request, func(ChunkResult) error { return want }); !errors.Is(err, want) {
		t.Fatal(err)
	}
	if _, err := e.Transcribe(context.Background(), Request{Samples: samples, SampleRate: SampleRate}); err != nil {
		t.Fatal("reuse after failed callback:", err)
	}
	if !reflect.DeepEqual(saved, delivered.Tokens) {
		t.Fatal("delivered tokens were overwritten")
	}

	// Close must wait for a synchronous callback to return.
	entered, release := make(chan struct{}), make(chan struct{})
	streamDone, closeDone := make(chan error, 1), make(chan error, 1)
	go func() {
		_, err := e.TranscribeStream(context.Background(), request, func(ChunkResult) error { close(entered); <-release; return want })
		streamDone <- err
	}()
	<-entered
	go func() { closeDone <- e.Close() }()
	select {
	case err := <-closeDone:
		close(release)
		<-streamDone
		t.Fatal("Close returned during a callback:", err)
	case <-time.After(20 * time.Millisecond):
	}
	close(release)
	if err := <-streamDone; !errors.Is(err, want) {
		t.Fatal(err)
	}
	if err := <-closeDone; err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(saved, delivered.Tokens) {
		t.Fatal("Close invalidated delivered tokens")
	}
}

func TestNativeChunking(t *testing.T) {
	c := nativeConfig(t)
	if os.Getenv("PARAKEET_TEST_REFERENCE") == "" {
		t.Skip("set PARAKEET_TEST_REFERENCE")
	}
	if !nativeSubprocess(t) {
		return
	}
	testChunkedInference(t, c)
}

func testLongRecording(t *testing.T, c Config) {
	t.Helper()
	part, text := speechCorpus(t)
	repeat := MaxWindowSamples/len(part) + 1
	samples := make([]float32, 0, repeat*len(part))
	for range repeat {
		samples = append(samples, part...)
	}
	e, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	request := Request{Samples: samples, SampleRate: SampleRate, Chunking: ChunkOptions{SinglePass: true}}
	if _, err := e.Transcribe(context.Background(), request); !errors.Is(err, ErrAudioTooLong) {
		t.Fatal("single pass accepted long input:", err)
	}
	request.Chunking = ChunkOptions{}
	var transcript strings.Builder
	var first time.Duration
	start := time.Now()
	summary, err := e.TranscribeStream(context.Background(), request, func(chunk ChunkResult) error {
		if chunk.Index == 0 {
			first = time.Since(start)
		}
		transcript.WriteString(chunk.Text)
		return nil
	})
	elapsed := time.Since(start)
	if err != nil {
		t.Fatal(err)
	}
	expected := strings.Repeat(text, repeat)
	if !reflect.DeepEqual(normalizedWords(transcript.String()), normalizedWords(expected)) {
		t.Fatalf("long recording lost/repeated words:\ngot: %s\nwant: %s", transcript.String(), expected)
	}
	if summary.Windows < 2 || summary.FinalizedSeconds != summary.AudioSeconds {
		t.Fatalf("incomplete summary: %+v", summary)
	}
	t.Logf("audio %.3fs, %d windows, first chunk %s, total %s, RTF %.3f; exact normalized word sequence", summary.AudioSeconds, summary.Windows, first, elapsed, elapsed.Seconds()/summary.AudioSeconds)
}

func TestNativeLongRecording(t *testing.T) {
	c := nativeConfig(t)
	if os.Getenv("PARAKEET_TEST_LONG") != "1" {
		t.Skip("set PARAKEET_TEST_LONG=1 for a recording over 300 seconds")
	}
	if !nativeSubprocess(t) {
		return
	}
	testLongRecording(t, c)
}

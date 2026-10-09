package parakeet_test

import (
	"context"
	"fmt"
	"log"

	parakeet "github.com/androiddrew/go-parakeet"
	ort "github.com/yalue/onnxruntime_go"
)

// loadSamples stands in for the application's audio decoding. The engine
// requires 16 kHz mono float32 PCM and performs no resampling.
func loadSamples() []float32 { return make([]float32, parakeet.SampleRate) }

// Transcribe a recording. The model files are downloads (see ASSETS.md); the
// vocabulary is built in. Long recordings are windowed automatically.
func Example() {
	engine, err := parakeet.New(parakeet.Config{
		ORTLibrary:       "libonnxruntime.so", // OS loader name or a path to the library
		PreprocessorPath: "assets/parakeet/nemo128.onnx",
		EncoderPath:      "assets/parakeet/encoder-model.onnx",
		DecoderPath:      "assets/parakeet/decoder_joint-model.onnx",
	})
	if err != nil {
		log.Fatal(err)
	}
	defer engine.Close()

	result, err := engine.Transcribe(context.Background(), parakeet.Request{
		Samples: loadSamples(), SampleRate: parakeet.SampleRate,
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Text)
}

// An application that already initializes ONNX Runtime itself leaves
// ORTLibrary empty. The engine then uses that environment and never destroys it.
func Example_callerOwnedEnvironment() {
	ort.SetSharedLibraryPath("libonnxruntime.so")
	if err := ort.InitializeEnvironment(); err != nil {
		log.Fatal(err)
	}
	defer ort.DestroyEnvironment() // runs after engine.Close

	engine, err := parakeet.New(parakeet.Config{
		PreprocessorPath: "assets/parakeet/nemo128.onnx",
		EncoderPath:      "assets/parakeet/encoder-model.onnx",
		DecoderPath:      "assets/parakeet/decoder_joint-model.onnx",
		Provider:         parakeet.CUDA, // requires an ONNX Runtime build with CUDA
	})
	if err != nil {
		log.Fatal(err)
	}
	defer engine.Close()
	fmt.Println(engine.Initialization().RuntimeVersion)
}

// Stream finalized text into a bounded queue so a consumer can forward it while
// the next window is decoded.
func ExampleEngine_TranscribeStream() {
	engine, err := parakeet.New(parakeet.Config{
		ORTLibrary:       "libonnxruntime.so",
		PreprocessorPath: "assets/parakeet/nemo128.onnx",
		EncoderPath:      "assets/parakeet/encoder-model.int8.onnx",
		DecoderPath:      "assets/parakeet/decoder_joint-model.int8.onnx",
	})
	if err != nil {
		log.Fatal(err)
	}
	defer engine.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	text := make(chan string, 2)
	done := make(chan error, 1)
	go func() {
		_, err := engine.TranscribeStream(ctx, parakeet.Request{
			Samples: loadSamples(), SampleRate: parakeet.SampleRate,
		}, func(chunk parakeet.ChunkResult) error {
			select {
			case text <- chunk.Text:
				return nil
			case <-ctx.Done():
				return ctx.Err()
			}
		})
		close(text)
		done <- err
	}()
	for t := range text {
		// Forward to the application's display or transport. Text may begin
		// with a space; concatenate chunks verbatim. On a consumer failure,
		// cancel() and wait for done before closing the engine.
		fmt.Print(t)
	}
	if err := <-done; err != nil {
		log.Print(err) // earlier chunks may already have been delivered
	}
}

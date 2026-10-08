package parakeet_test

import (
	"context"
	"errors"
	"fmt"

	parakeet "github.com/androiddrew/go-parakeet"
)

func ExampleEngine_Transcribe() {
	// Applications decode and resample their audio before calling this function.
	transcribe := func(ctx context.Context, samples []float32) (err error) {
		e, err := parakeet.New(parakeet.Config{
			ORTLibrary:       "libonnxruntime.so",
			PreprocessorPath: "/assets/parakeet/nemo128.onnx",
			EncoderPath:      "/assets/parakeet/encoder-model.onnx",
			DecoderPath:      "/assets/parakeet/decoder_joint-model.onnx",
			ConfigPath:       "/assets/parakeet/config.json",
			VocabPath:        "/assets/parakeet/vocab.txt",
			Provider:         parakeet.CPU,
		})
		if err != nil {
			return err
		}
		defer func() { err = errors.Join(err, e.Close()) }()
		result, err := e.Transcribe(ctx, parakeet.Request{Samples: samples, SampleRate: parakeet.SampleRate})
		if err != nil {
			return err
		}
		fmt.Println(result.Text)
		return nil
	}
	_ = transcribe
}

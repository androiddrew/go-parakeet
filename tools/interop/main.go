// interop verifies same-process runtime ownership across the three libraries.
package main

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"

	g2p "github.com/androiddrew/go-g2p"
	kokoro "github.com/androiddrew/go-kokoro"
	parakeet "github.com/androiddrew/go-parakeet"
	ort "github.com/yalue/onnxruntime_go"
)

func run() (err error) {
	library := flag.String("ort", "", "ORT library selector shared by all components")
	assets := flag.String("assets", "", "asset root containing kokoro/ and parakeet/")
	input := flag.String("input", "", "16 kHz mono float32 PCM")
	cuda := flag.Bool("cuda", false, "CUDA for Kokoro and Parakeet, CPU for G2P")
	flag.Parse()
	if *library == "" || *assets == "" || *input == "" {
		return errors.New("--ort, --assets, --input are required")
	}
	frontend, err := g2p.New(g2p.Config{ORTLibrary: *library})
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, frontend.Close()) }()
	kprovider, pprovider := kokoro.CPU, parakeet.CPU
	if *cuda {
		kprovider, pprovider = kokoro.CUDA, parakeet.CUDA
	}
	tts, err := kokoro.New(kokoro.Config{ORTLibrary: *library, ModelPath: filepath.Join(*assets, "kokoro/kokoro-v1.0.onnx"), VoicesPath: filepath.Join(*assets, "kokoro/voices-v1.0.bin"), Provider: kprovider})
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, tts.Close()) }()
	path := func(name string) string { return filepath.Join(*assets, "parakeet", name) }
	config := parakeet.Config{ORTLibrary: *library, PreprocessorPath: path("nemo128.onnx"), EncoderPath: path("encoder-model.onnx"), DecoderPath: path("decoder_joint-model.onnx"), VocabPath: path("vocab.txt"), ConfigPath: path("config.json"), Provider: pprovider}
	asr, err := parakeet.New(config)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, asr.Close()) }()
	ctx := context.Background()
	phones, err := frontend.Phonemize(ctx, g2p.Request{Text: "Hello, world!", Dialect: g2p.US})
	if err != nil {
		return err
	}
	audio, err := tts.Synthesize(ctx, kokoro.Request{Phonemes: phones.Phonemes, Voice: "af_heart", Speed: 1, Trim: true})
	if err != nil {
		return fmt.Errorf("TTS failed: %w", err)
	}
	if len(audio.Samples) == 0 {
		return errors.New("TTS returned empty audio")
	}
	if err := tts.Close(); err != nil {
		return err
	}
	raw, err := os.ReadFile(*input)
	if err != nil {
		return err
	}
	if len(raw)%4 != 0 {
		return errors.New("invalid float32 input")
	}
	samples := make([]float32, len(raw)/4)
	for i := range samples {
		samples[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	result, err := asr.Transcribe(ctx, parakeet.Request{Samples: samples, SampleRate: parakeet.SampleRate})
	if err != nil {
		return err
	}
	if _, err := frontend.Phonemize(ctx, g2p.Request{Text: "Hello again.", Dialect: g2p.US}); err != nil {
		return err
	}
	if err := frontend.Close(); err != nil {
		return err
	}
	if err := asr.Close(); err != nil {
		return err
	}
	if !ort.IsInitialized() {
		return errors.New("last engine close destroyed the runtime")
	}
	again, err := parakeet.New(config)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, again.Close()) }()
	second, err := again.Transcribe(ctx, parakeet.Request{Samples: samples, SampleRate: parakeet.SampleRate})
	if err != nil {
		return err
	}
	if second.Text != result.Text {
		return errors.New("reopening changed ASR output")
	}
	return json.NewEncoder(os.Stdout).Encode(map[string]any{"runtime": ort.GetVersion(), "cuda": *cuda, "phonemes": phones.Phonemes, "tts_samples": len(audio.Samples), "text": result.Text, "reopen_verified": true})
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

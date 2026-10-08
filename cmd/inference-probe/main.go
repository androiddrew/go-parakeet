// inference-probe runs a local model on little-endian float32 mono PCM.
package main

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"os/signal"
	"path/filepath"

	parakeet "github.com/androiddrew/go-parakeet"
)

func run() (err error) {
	dir := flag.String("assets", "", "directory containing the pinned model bundle")
	lib := flag.String("ort", "", "installed ONNX Runtime path or loader name (required)")
	precision := flag.String("precision", "fp32", "fp32 or int8")
	provider := flag.String("provider", "cpu", "cpu or cuda")
	device := flag.Int("device", 0, "CUDA device ID")
	threads := flag.Int("threads", 1, "intra-op threads")
	input := flag.String("input", "", "16 kHz mono little-endian float32 PCM file")
	verbose := flag.Bool("verbose", false, "ORT node-placement diagnostics to stderr")
	metadata := flag.Bool("metadata-only", false, "inspect models without transcribing")
	flag.Parse()
	if *dir == "" || *lib == "" {
		return errors.New("--assets and --ort are required")
	}
	suffix := ""
	switch *precision {
	case "fp32":
	case "int8":
		suffix = ".int8"
	default:
		return errors.New("--precision must be fp32 or int8")
	}
	var samples []float32
	if !*metadata {
		if *input == "" {
			return errors.New("--input is required")
		}
		f, err := os.Open(*input)
		if err != nil {
			return err
		}
		raw, readErr := io.ReadAll(io.LimitReader(f, int64(parakeet.MaxSamples+1)*4))
		if err := errors.Join(readErr, f.Close()); err != nil {
			return err
		}
		if len(raw)%4 != 0 {
			return errors.New("PCM byte length must be a multiple of four")
		}
		if len(raw)/4 > parakeet.MaxSamples {
			return parakeet.ErrAudioTooLong
		}
		samples = make([]float32, len(raw)/4)
		for i := range samples {
			samples[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
	}
	e, err := parakeet.New(parakeet.Config{
		ORTLibrary: *lib, PreprocessorPath: filepath.Join(*dir, "nemo128.onnx"),
		EncoderPath: filepath.Join(*dir, "encoder-model"+suffix+".onnx"), DecoderPath: filepath.Join(*dir, "decoder_joint-model"+suffix+".onnx"),
		ConfigPath: filepath.Join(*dir, "config.json"), VocabPath: filepath.Join(*dir, "vocab.txt"),
		Provider: parakeet.Provider(*provider), DeviceID: *device, Threads: *threads, Verbose: *verbose,
	})
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, e.Close()) }()
	report := struct {
		Initialization parakeet.Initialization `json:"initialization"`
		Models         parakeet.ModelInfo      `json:"models"`
		Result         *parakeet.Result        `json:"result,omitempty"`
	}{Initialization: e.Initialization(), Models: e.ModelInfo()}
	if !*metadata {
		ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
		defer cancel()
		result, err := e.Transcribe(ctx, parakeet.Request{Samples: samples, SampleRate: parakeet.SampleRate})
		if err != nil {
			return err
		}
		report.Result = &result
	}
	return json.NewEncoder(os.Stdout).Encode(report)
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

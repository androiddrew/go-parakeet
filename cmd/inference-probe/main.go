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
	vocab := flag.String("vocab", "", "vocabulary file (default: embedded Parakeet TDT v3)")
	verbose := flag.Bool("verbose", false, "ORT node-placement diagnostics to stderr")
	metadata := flag.Bool("metadata-only", false, "inspect models without transcribing")
	stream := flag.Bool("stream", false, "emit finalized chunk events and a final summary as JSON lines")
	window := flag.Duration("window", 0, "audio window duration (default 30s; rounds down to 80ms)")
	overlap := flag.Duration("overlap", 0, "window overlap (default min(4s, window/4))")
	singlePass := flag.Bool("single-pass", false, "disable windowing; reject audio over 300s")
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
		stat, statErr := f.Stat()
		if statErr != nil || !stat.Mode().IsRegular() {
			return errors.Join(statErr, errors.New("--input must be a regular PCM file"), f.Close())
		}
		raw, readErr := io.ReadAll(f)
		if err := errors.Join(readErr, f.Close()); err != nil {
			return err
		}
		if len(raw)%4 != 0 {
			return errors.New("PCM byte length must be a multiple of four")
		}
		samples = make([]float32, len(raw)/4)
		for i := range samples {
			samples[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
	}
	e, err := parakeet.New(parakeet.Config{
		ORTLibrary: *lib, PreprocessorPath: filepath.Join(*dir, "nemo128.onnx"),
		EncoderPath: filepath.Join(*dir, "encoder-model"+suffix+".onnx"), DecoderPath: filepath.Join(*dir, "decoder_joint-model"+suffix+".onnx"),
		VocabPath: *vocab, Provider: parakeet.Provider(*provider), DeviceID: *device, Threads: *threads, Verbose: *verbose,
	})
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, e.Close()) }()
	report := struct {
		Type           string                  `json:"type,omitempty"`
		Initialization parakeet.Initialization `json:"initialization"`
		Models         parakeet.ModelInfo      `json:"models"`
		Result         *parakeet.Result        `json:"result,omitempty"`
		Summary        *parakeet.Summary       `json:"summary,omitempty"`
	}{Initialization: e.Initialization(), Models: e.ModelInfo()}
	if !*metadata {
		ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
		defer cancel()
		request := parakeet.Request{Samples: samples, SampleRate: parakeet.SampleRate,
			Chunking: parakeet.ChunkOptions{Window: *window, Overlap: *overlap, SinglePass: *singlePass}}
		if *stream {
			encoder := json.NewEncoder(os.Stdout)
			summary, err := e.TranscribeStream(ctx, request, func(chunk parakeet.ChunkResult) error {
				return encoder.Encode(struct {
					Type  string               `json:"type"`
					Chunk parakeet.ChunkResult `json:"chunk"`
				}{"chunk", chunk})
			})
			if err != nil {
				return err
			}
			report.Summary = &summary
			report.Type = "done"
			return encoder.Encode(report)
		}
		result, err := e.Transcribe(ctx, request)
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

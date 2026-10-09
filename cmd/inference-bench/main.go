// inference-bench measures a loaded engine, with separate first/warm requests.
package main

import (
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	parakeet "github.com/androiddrew/go-parakeet"
)

type memory struct {
	RSSBytes     uint64 `json:"rss_bytes"`
	PeakRSSBytes uint64 `json:"peak_rss_bytes"`
	GoHeapBytes  uint64 `json:"go_heap_bytes"`
}

// Linux process RSS includes native memory and mapped pages; it is not Go heap
// usage or a precise native-only allocation counter. GPU memory is separate.
func memoryInfo() (memory, error) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	result := memory{GoHeapBytes: m.HeapAlloc}
	data, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return result, err
	}
	for _, line := range strings.Split(string(data), "\n") {
		fields := strings.Fields(line)
		if len(fields) == 3 && (fields[0] == "VmRSS:" || fields[0] == "VmHWM:") {
			value, err := strconv.ParseUint(fields[1], 10, 64)
			if err != nil {
				return result, err
			}
			if fields[0] == "VmRSS:" {
				result.RSSBytes = value * 1024
			} else {
				result.PeakRSSBytes = value * 1024
			}
		}
	}
	return result, nil
}

type measurement struct {
	Phase             string           `json:"phase"`
	WallSeconds       float64          `json:"wall_seconds"`
	FirstChunkSeconds float64          `json:"first_chunk_seconds"`
	RTF               float64          `json:"rtf"`
	TextSHA256        string           `json:"text_sha256"`
	Summary           parakeet.Summary `json:"summary"`
	Memory            memory           `json:"memory_after_request"`
}

func run() (err error) {
	assets := flag.String("assets", "", "model directory")
	library := flag.String("ort", "", "installed ORT selector")
	input := flag.String("input", "", "seed little-endian float32 PCM (optional gzip)")
	provider := flag.String("provider", "cpu", "cpu or cuda")
	precision := flag.String("precision", "fp32", "fp32 or int8")
	lengths := flag.String("seconds", "5,30,308", "comma-separated test durations; seed repeats to fill each")
	repeat := flag.Int("repeat", 3, "warm measured requests per duration")
	warmup := flag.Int("warmup", 1, "unreported warmups after first request")
	threads := flag.Int("threads", 1, "intra-op threads")
	window := flag.Duration("window", 30*time.Second, "window duration")
	overlap := flag.Duration("overlap", 4*time.Second, "overlap duration")
	flag.Parse()
	if *assets == "" || *library == "" || *input == "" || *repeat < 1 || *warmup < 0 {
		return errors.New("require assets, ort, input, positive repeat and nonnegative warmup")
	}
	suffix := ""
	if *precision == "int8" {
		suffix = ".int8"
	} else if *precision != "fp32" {
		return errors.New("invalid precision")
	}
	if *provider == "cuda" && *precision != "fp32" {
		return errors.New("CUDA benchmarks require FP32")
	}
	file, err := os.Open(*input)
	if err != nil {
		return err
	}
	var reader io.Reader = file
	if strings.HasSuffix(*input, ".gz") {
		z, err := gzip.NewReader(file)
		if err != nil {
			file.Close()
			return err
		}
		defer z.Close()
		reader = z
	}
	data, readErr := io.ReadAll(reader)
	if err := errors.Join(readErr, file.Close()); err != nil {
		return err
	}
	if len(data) == 0 || len(data)%4 != 0 {
		return errors.New("invalid seed PCM")
	}
	seed := make([]float32, len(data)/4)
	for i := range seed {
		seed[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	seedHash := fmt.Sprintf("%x", sha256.Sum256(data))
	data = nil
	path := func(name string) string { return filepath.Join(*assets, name) }
	e, err := parakeet.New(parakeet.Config{ORTLibrary: *library, PreprocessorPath: path("nemo128.onnx"), EncoderPath: path("encoder-model" + suffix + ".onnx"), DecoderPath: path("decoder_joint-model" + suffix + ".onnx"), ConfigPath: path("config.json"), VocabPath: path("vocab.txt"), Provider: parakeet.Provider(*provider), Threads: *threads})
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, e.Close()) }()
	runtime.GC()
	initialMemory, err := memoryInfo()
	if err != nil {
		return err
	}
	var results []measurement
	for _, value := range strings.Split(*lengths, ",") {
		seconds, err := strconv.ParseFloat(value, 64)
		if err != nil || math.IsNaN(seconds) || math.IsInf(seconds, 0) || seconds < 0.02 || seconds > 3600 {
			return fmt.Errorf("invalid benchmark seconds %q (20ms..1h)", value)
		}
		samples := make([]float32, int(seconds*parakeet.SampleRate))
		for offset := 0; offset < len(samples); {
			offset += copy(samples[offset:], seed)
		}
		request := parakeet.Request{Samples: samples, SampleRate: parakeet.SampleRate, Chunking: parakeet.ChunkOptions{Window: *window, Overlap: *overlap}}
		var expected string
		for iteration := 0; iteration < 1+*warmup+*repeat; iteration++ {
			hash := sha256.New()
			first := 0.0
			start := time.Now()
			summary, err := e.TranscribeStream(context.Background(), request, func(chunk parakeet.ChunkResult) error {
				if chunk.Index == 0 {
					first = time.Since(start).Seconds()
				}
				_, err := io.WriteString(hash, chunk.Text)
				return err
			})
			elapsed := time.Since(start).Seconds()
			if err != nil {
				return err
			}
			digest := fmt.Sprintf("%x", hash.Sum(nil))
			if iteration == 0 {
				expected = digest
			} else if digest != expected {
				return errors.New("repeated request changed transcript")
			}
			if iteration > 0 && iteration <= *warmup {
				continue
			}
			mem, err := memoryInfo()
			if err != nil {
				return err
			}
			phase := "warm"
			if iteration == 0 {
				phase = "first"
			}
			results = append(results, measurement{phase, elapsed, first, elapsed / summary.AudioSeconds, digest, summary, mem})
		}
	}
	return json.NewEncoder(os.Stdout).Encode(map[string]any{"runtime": e.Initialization(), "go_version": runtime.Version(), "pid": os.Getpid(), "provider": *provider, "precision": *precision, "threads": *threads, "warmup": *warmup, "repeat": *repeat, "seed_sha256": seedHash, "memory_after_loading": initialMemory, "runs": results, "consumer": "SHA-256 of chunk text; no aggregate retained"})
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

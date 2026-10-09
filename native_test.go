package parakeet

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"sync"
	"testing"

	ort "github.com/yalue/onnxruntime_go"
)

func nativeConfig(t *testing.T) Config {
	t.Helper()
	dir, lib := os.Getenv("PARAKEET_TEST_ASSETS"), os.Getenv("PARAKEET_TEST_ORT")
	if dir == "" || lib == "" {
		t.Skip("set PARAKEET_TEST_ASSETS and PARAKEET_TEST_ORT for native tests")
	}
	suffix := ""
	if os.Getenv("PARAKEET_TEST_PRECISION") == "int8" {
		suffix = ".int8"
	}
	return Config{ORTLibrary: lib, PreprocessorPath: filepath.Join(dir, "nemo128.onnx"), EncoderPath: filepath.Join(dir, "encoder-model"+suffix+".onnx"), DecoderPath: filepath.Join(dir, "decoder_joint-model"+suffix+".onnx"), ConfigPath: filepath.Join(dir, "config.json"), VocabPath: filepath.Join(dir, "vocab.txt")}
}

func nativeSubprocess(t *testing.T) bool {
	t.Helper()
	if os.Getenv("PARAKEET_NATIVE_CHILD") == t.Name() {
		return true
	}
	cmd := exec.Command(os.Args[0], "-test.run=^"+t.Name()+"$", "-test.v")
	cmd.Env = append(os.Environ(), "PARAKEET_NATIVE_CHILD="+t.Name())
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("native subprocess: %v\n%s", err, output)
	}
	t.Log(string(output))
	return false
}

func readFloat32(t *testing.T, path, expected string) []float32 {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if expected != "" && fmt.Sprintf("%x", sha256.Sum256(raw)) != expected {
		t.Fatal("fixture checksum mismatch:", path)
	}
	if len(raw)%4 != 0 {
		t.Fatal("invalid float32 fixture")
	}
	data := make([]float32, len(raw)/4)
	for i := range data {
		data[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return data
}

func compareFloats(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("stage length %d != %d", len(got), len(want))
	}
	for i, a := range got {
		b := want[i]
		if math.IsNaN(float64(a)) || math.Abs(float64(a-b)) > 1e-4+1e-4*math.Abs(float64(b)) {
			t.Fatalf("stage[%d] = %g != %g", i, a, b)
		}
	}
}

func TestNativeParity(t *testing.T) {
	c := nativeConfig(t)
	dir := os.Getenv("PARAKEET_TEST_REFERENCE")
	if dir == "" {
		t.Skip("set PARAKEET_TEST_REFERENCE to generated Python stage fixtures")
	}
	if !nativeSubprocess(t) {
		return
	}
	e, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := e.Close(); err != nil {
			t.Error(err)
		}
	}()
	files, err := filepath.Glob(filepath.Join(dir, "*.json"))
	if err != nil || len(files) == 0 {
		t.Fatal("no reference fixtures", err)
	}
	for _, file := range files {
		t.Run(filepath.Base(file), func(t *testing.T) {
			var reference struct {
				Text          string  `json:"text"`
				Tokens        []Token `json:"tokens"`
				FeatureLength int     `json:"feature_length"`
				EncodedLength int     `json:"encoded_length"`
				Stages        map[string]struct {
					File   string  `json:"file"`
					SHA256 string  `json:"sha256"`
					Shape  []int64 `json:"shape"`
				} `json:"stages"`
			}
			raw, err := os.ReadFile(file)
			if err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(raw, &reference); err != nil {
				t.Fatal(err)
			}
			stage := func(name string) []float32 {
				s := reference.Stages[name]
				return readFloat32(t, filepath.Join(dir, s.File), s.SHA256)
			}
			samples := stage("samples")
			features, err := runSequence(e.preprocessor, samples, ort.NewShape(1, int64(len(samples))), len(samples), 128)
			if err != nil {
				t.Fatal(err)
			}
			compareFloats(t, features.data, stage("features"))
			if features.length != reference.FeatureLength {
				t.Fatal("feature length mismatch")
			}
			encoded, err := runSequence(e.encoder, features.data, ort.NewShape(1, 128, int64(features.stride)), features.length, 1024)
			if err != nil {
				t.Fatal(err)
			}
			compareFloats(t, encoded.data, stage("encoder"))
			if encoded.length != reference.EncodedLength {
				t.Fatal("encoder length mismatch")
			}
			for range 2 { // catches decoder state leaking across requests
				result, err := e.Transcribe(context.Background(), Request{Samples: samples, SampleRate: SampleRate})
				if err != nil {
					t.Fatal(err)
				}
				if result.Text != reference.Text || !reflect.DeepEqual(result.Tokens, reference.Tokens) {
					t.Fatalf("transcript/decisions differ:\ngot %+v\nwant %+v", result, reference.Tokens)
				}
			}
		})
	}
	t.Log("native runtime:", e.Initialization().RuntimeVersion)
}

func TestNativeLifecycle(t *testing.T) {
	c := nativeConfig(t)
	if !nativeSubprocess(t) {
		return
	}
	caller := c
	caller.ORTLibrary = ""
	if _, err := New(caller); !errors.Is(err, ErrNotInitialized) {
		t.Fatal("uninitialized environment:", err)
	}
	e, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	other, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	defer other.Close()
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := e.Transcribe(context.Background(), Request{}); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := other.Transcribe(ctx, Request{}); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if _, err := other.Transcribe(context.Background(), Request{Samples: make([]float32, 1600), SampleRate: SampleRate}); err != nil {
		t.Fatal("closing one engine invalidated another:", err)
	}
	// Exercise the minimum accepted duration and the 8x subsampling boundary,
	// where physical encoder width and valid sequence length can differ.
	for _, n := range []int{320, 321, 1279, 1280, 1281} {
		if _, err := other.Transcribe(context.Background(), Request{Samples: make([]float32, n), SampleRate: SampleRate}); err != nil {
			t.Fatalf("short input (%d samples): %v", n, err)
		}
	}
	info := other.ModelInfo()
	info.Encoder.Inputs[0].Shape[1] = 999
	if other.ModelInfo().Encoder.Inputs[0].Shape[1] != 128 {
		t.Fatal("metadata aliases engine state")
	}
	conflict := c
	conflict.ORTLibrary += ".other"
	if _, err := New(conflict); err == nil {
		t.Fatal("accepted conflicting library selector")
	}
	var wg sync.WaitGroup
	for range 2 {
		wg.Go(func() {
			if _, err := other.Transcribe(context.Background(), Request{Samples: make([]float32, 1600), SampleRate: SampleRate}); err != nil && !errors.Is(err, ErrClosed) {
				t.Error(err)
			}
		})
	}
	if err := other.Close(); err != nil {
		t.Fatal(err)
	}
	wg.Wait() // requests that race with Close may validly return ErrClosed
	if !ort.IsInitialized() {
		t.Fatal("Close unloaded the runtime")
	}
	again, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	if err := again.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestCallerOwnedEnvironment(t *testing.T) {
	c := nativeConfig(t)
	if !nativeSubprocess(t) {
		return
	}
	ort.SetSharedLibraryPath(c.ORTLibrary)
	if err := ort.InitializeEnvironment(); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := ort.DestroyEnvironment(); err != nil {
			t.Error(err)
		}
	}()
	c.ORTLibrary = ""
	e, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	if _, err := e.Transcribe(context.Background(), Request{Samples: make([]float32, 1600), SampleRate: SampleRate}); err != nil {
		t.Fatal(err)
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	if !ort.IsInitialized() {
		t.Fatal("engine destroyed caller's environment")
	}
}

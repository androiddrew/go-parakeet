//go:build cuda

package parakeet

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// CUDA access requires both the build tag and a dedicated opt-in selector.
func TestCUDAInferenceCycles(t *testing.T) {
	lib := os.Getenv("PARAKEET_TEST_CUDA_ORT")
	if lib == "" {
		t.Skip("set PARAKEET_TEST_CUDA_ORT to opt into GPU inference")
	}
	c := nativeConfigFor(t, lib)
	c.ORTLibrary, c.Provider = lib, CUDA
	dir := referenceDir(t)
	if os.Getenv("PARAKEET_TEST_PRECISION") == "int8" {
		t.Fatal("CUDA validation requires FP32 assets")
	}
	if !nativeSubprocess(t) {
		return
	}
	files, err := filepath.Glob(filepath.Join(dir, "*.json"))
	if err != nil || len(files) == 0 {
		t.Fatal("missing references:", err)
	}
	for range 2 {
		e, err := New(c)
		if err != nil {
			t.Fatal(err)
		}
		defer e.Close()
		for _, path := range files {
			var ref struct {
				Text   string  `json:"text"`
				Tokens []Token `json:"tokens"`
				Stages map[string]struct {
					File   string `json:"file"`
					SHA256 string `json:"sha256"`
				} `json:"stages"`
			}
			raw, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(raw, &ref); err != nil {
				t.Fatal(err)
			}
			s := ref.Stages["samples"]
			samples := readFloat32(t, filepath.Join(dir, s.File), s.SHA256)
			got, err := e.Transcribe(context.Background(), Request{Samples: samples, SampleRate: SampleRate})
			if err != nil {
				t.Fatal(err)
			}
			if got.Text != ref.Text {
				t.Fatalf("%s: %q != %q", path, got.Text, ref.Text)
			}
			if len(got.Tokens) != len(ref.Tokens) {
				t.Fatal("token count differs")
			}
			for i, token := range got.Tokens {
				if token.ID != ref.Tokens[i].ID {
					t.Fatal("token ID differs")
				}
			}
		}
		if err := e.Close(); err != nil {
			t.Fatal(err)
		}
	}
}

func TestCUDAChunking(t *testing.T) {
	lib := os.Getenv("PARAKEET_TEST_CUDA_ORT")
	if lib == "" {
		t.Skip("set PARAKEET_TEST_CUDA_ORT to opt into GPU inference")
	}
	c := nativeConfigFor(t, lib)
	c.ORTLibrary, c.Provider = lib, CUDA
	if os.Getenv("PARAKEET_TEST_PRECISION") == "int8" {
		t.Fatal("CUDA validation requires FP32")
	}
	if !nativeSubprocess(t) {
		return
	}
	testChunkedInference(t, c)
}

func TestCUDALongRecording(t *testing.T) {
	lib := os.Getenv("PARAKEET_TEST_CUDA_ORT")
	if lib == "" || os.Getenv("PARAKEET_TEST_LONG") != "1" {
		t.Skip("set PARAKEET_TEST_CUDA_ORT and PARAKEET_TEST_LONG=1")
	}
	c := nativeConfigFor(t, lib)
	c.ORTLibrary, c.Provider = lib, CUDA
	if os.Getenv("PARAKEET_TEST_PRECISION") == "int8" {
		t.Fatal("CUDA validation requires FP32")
	}
	if !nativeSubprocess(t) {
		return
	}
	testLongRecording(t, c)
}

func TestCUDANaturalSpeech(t *testing.T) {
	lib := os.Getenv("PARAKEET_TEST_CUDA_ORT")
	if lib == "" {
		t.Skip("set PARAKEET_TEST_CUDA_ORT")
	}
	c := nativeConfigFor(t, lib)
	c.Provider = CUDA
	if testPrecision(t) != "fp32" {
		t.Fatal("CUDA requires FP32")
	}
	if !nativeSubprocess(t) {
		return
	}
	testNaturalSpeech(t, c)
}

func TestCUDAEncoderLimit(t *testing.T) {
	lib := os.Getenv("PARAKEET_TEST_CUDA_ORT")
	if lib == "" || os.Getenv("PARAKEET_TEST_LIMIT") != "1" {
		t.Skip("set PARAKEET_TEST_CUDA_ORT and PARAKEET_TEST_LIMIT=1")
	}
	c := nativeConfigFor(t, lib)
	c.Provider = CUDA
	if testPrecision(t) != "fp32" {
		t.Fatal("CUDA requires FP32")
	}
	if !nativeSubprocess(t) {
		return
	}
	testEncoderLimit(t, c)
}

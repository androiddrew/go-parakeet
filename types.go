// Package parakeet transcribes 16 kHz mono PCM using Parakeet TDT v3 and ONNX Runtime.
package parakeet

import (
	"errors"

	"github.com/androiddrew/go-parakeet/internal/ortruntime"
)

const SampleRate = 16000

// MaxSamples is the initial single-pass API's conservative 300-second bound.
// It is not a claim about the maximum supported by every ONNX export or GPU.
const MaxSamples = 300 * SampleRate

var (
	ErrNotInitialized = ortruntime.ErrNotInitialized
	ErrClosed         = errors.New("parakeet engine is closed")
	ErrInvalidAudio   = errors.New("invalid audio")
	ErrAudioTooLong   = errors.New("audio exceeds the single-pass limit")
)

type Provider string

const (
	CPU  Provider = "cpu"
	CUDA Provider = "cuda"
)

// Config selects local assets and session settings. Zero Provider selects CPU;
// zero Threads selects one intra-op thread per session, with one inter-op thread.
type Config struct {
	// ORTLibrary is an installed library path or OS-loader name. Nonempty uses
	// ortenv's process-lifetime environment. Empty requires prior initialization.
	ORTLibrary       string
	PreprocessorPath string
	EncoderPath      string
	DecoderPath      string
	ConfigPath       string
	VocabPath        string
	Provider         Provider
	DeviceID         int
	Threads          int
	Verbose          bool
}

// Request supplies already-decoded mono PCM. Samples must remain unmodified
// until Transcribe returns. Values must be finite, conventionally in [-1, 1].
// No implicit gain, clipping, channel conversion, or resampling is performed.
type Request struct {
	Samples    []float32 `json:"-"`
	SampleRate int       `json:"sample_rate"`
}

// Token records a nonblank decoder emission. Piece is the raw SentencePiece
// entry. StartSeconds is an emission position, not an aligned word boundary.
type Token struct {
	ID           int     `json:"id"`
	Piece        string  `json:"piece"`
	Frame        int     `json:"frame"`
	StartSeconds float64 `json:"start_seconds"`
	Advance      int     `json:"advance"` // predicted encoder-frame advance, not word duration
}

type Timings struct {
	ValidationSeconds    float64 `json:"validation_seconds"`
	PreprocessingSeconds float64 `json:"preprocessing_seconds"`
	EncoderSeconds       float64 `json:"encoder_seconds"`
	DecoderSeconds       float64 `json:"decoder_seconds"`
	AssemblySeconds      float64 `json:"assembly_seconds"`
}

type Result struct {
	Text         string  `json:"text"`
	Tokens       []Token `json:"tokens"`
	AudioSeconds float64 `json:"audio_seconds"`
	Timings      Timings `json:"timings"`
}

// Initialization includes metadata inspection, which loads temporary sessions
// with this binding version, as well as the retained inference sessions.
type Initialization struct {
	RuntimeVersion string  `json:"runtime_version"`
	RuntimeSeconds float64 `json:"runtime_seconds"`
	AssetsSeconds  float64 `json:"assets_seconds"`
	ModelSeconds   float64 `json:"model_seconds"`
}

type TensorInfo struct {
	Name  string  `json:"name"`
	Shape []int64 `json:"shape"`
	Type  string  `json:"type"`
}

type GraphInfo struct {
	Inputs  []TensorInfo `json:"inputs"`
	Outputs []TensorInfo `json:"outputs"`
}

type ModelInfo struct {
	Preprocessor GraphInfo `json:"preprocessor"`
	Encoder      GraphInfo `json:"encoder"`
	Decoder      GraphInfo `json:"decoder"`
}

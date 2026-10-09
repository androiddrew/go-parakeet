// Package parakeet transcribes 16 kHz mono PCM using Parakeet TDT v3 and ONNX Runtime.
package parakeet

import (
	"errors"
	"time"

	"github.com/androiddrew/go-parakeet/internal/ortruntime"
)

const SampleRate = 16000

// MaxWindowSamples bounds each native pass, not the complete recording.
// It is not a memory guarantee on every GPU or an exact exported-model limit.
const MaxWindowSamples = 300 * SampleRate

// MaxSamples is retained as an alias for the single-pass bound.
const MaxSamples = MaxWindowSamples

const (
	DefaultWindow  = 30 * time.Second
	DefaultOverlap = 4 * time.Second
)

var (
	ErrNotInitialized  = ortruntime.ErrNotInitialized
	ErrClosed          = errors.New("parakeet engine is closed")
	ErrInvalidAudio    = errors.New("invalid audio")
	ErrAudioTooLong    = errors.New("audio exceeds the single-pass limit")
	ErrInvalidChunking = errors.New("invalid chunking options")
	ErrUnstableSeam    = errors.New("overlap cannot be finalized within bounded context")
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
// until transcription returns. Values must be finite, conventionally in [-1, 1].
// No implicit gain, clipping, channel conversion, or resampling is performed.
type Request struct {
	Samples    []float32    `json:"-"`
	SampleRate int          `json:"sample_rate"`
	Chunking   ChunkOptions `json:"chunking"`
}

// ChunkOptions controls complete-audio windowing. Zero values enable automatic
// chunking with 30-second windows and up to 4 seconds of overlap. Window is
// 2..300 seconds; Overlap is at most half the window. Durations round down to
// the 80 ms encoder grid. Zero Overlap selects min(4 seconds, Window/4).
// A positive overlap must round to at least one frame. SinglePass disables
// windowing and rejects audio beyond MaxWindowSamples; its durations must be zero.
type ChunkOptions struct {
	Window     time.Duration `json:"window"`
	Overlap    time.Duration `json:"overlap"`
	SinglePass bool          `json:"single_pass"`
}

// Token records a nonblank decoder emission. Piece is the raw SentencePiece
// entry. Frame and StartSeconds are recording-relative emission positions, not
// aligned word boundaries. Small backwards shifts at seams are clamped to keep
// the assembled sequence nondecreasing.
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

// ChunkResult is finalized text and owned tokens. Text may start with a space;
// concatenate it verbatim. The half-open sample span records finalized source
// coverage, not precise word alignment or the encoder's overlapping window.
// Seam is empty for the first window, then "aligned" or "timestamp".
type ChunkResult struct {
	Index       int     `json:"index"`
	StartSample int     `json:"start_sample"`
	EndSample   int     `json:"end_sample"`
	Text        string  `json:"text"`
	Tokens      []Token `json:"tokens"`
	Seam        string  `json:"seam,omitempty"`
}

// Summary has constant size. Windows counts completed decodes; DeliveredChunks
// counts callbacks that returned nil. On error it includes completed work and
// timings (including work in a failed window), excluding time in callbacks.
// ProcessedSeconds is the furthest decoded sample; FinalizedSeconds is the end
// of the last successfully delivered chunk. Neither implies recognition accuracy.
type Summary struct {
	AudioSeconds     float64 `json:"audio_seconds"`
	ProcessedSeconds float64 `json:"processed_seconds"`
	FinalizedSeconds float64 `json:"finalized_seconds"`
	Windows          int     `json:"windows"`
	DeliveredChunks  int     `json:"delivered_chunks"`
	WindowSamples    int     `json:"window_samples"`
	OverlapSamples   int     `json:"overlap_samples"`
	Timings          Timings `json:"timings"`
}

// Result collects the same finalized chunks emitted by TranscribeStream.
type Result struct {
	Text   string        `json:"text"`
	Tokens []Token       `json:"tokens"`
	Chunks []ChunkResult `json:"chunks"`
	Summary
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

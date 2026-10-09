// Package parakeet transcribes 16 kHz mono PCM using Parakeet TDT v3 and ONNX Runtime.
package parakeet

import (
	"errors"
	"time"

	"github.com/androiddrew/go-parakeet/internal/ortruntime"
)

// SampleRate is the required input rate in Hz. Input is mono.
const SampleRate = 16000

// MaxWindowSamples bounds each native pass, not the complete recording.
// It is not a memory guarantee on every GPU or an exact exported-model limit.
const MaxWindowSamples = 300 * SampleRate

// Default long-audio windowing, used when ChunkOptions durations are zero.
const (
	DefaultWindow  = 30 * time.Second
	DefaultOverlap = 4 * time.Second
)

var (
	// ErrNotInitialized is returned by New when ORTLibrary is empty and the
	// caller has not initialized the ONNX Runtime environment.
	ErrNotInitialized = ortruntime.ErrNotInitialized
	// ErrClosed is returned by requests on a closed engine.
	ErrClosed = errors.New("parakeet engine is closed")
	// ErrInvalidAudio reports a wrong sample rate, too few samples or a
	// nonfinite sample.
	ErrInvalidAudio = errors.New("invalid audio")
	// ErrAudioTooLong is returned when SinglePass input exceeds MaxWindowSamples.
	ErrAudioTooLong = errors.New("audio exceeds the single-pass limit")
	// ErrInvalidChunking reports ChunkOptions outside their documented bounds.
	ErrInvalidChunking = errors.New("invalid chunking options")
	// ErrUnstableSeam is returned when an unbroken hypothesis cannot be
	// finalized within two windows' token budget. Delivered chunks remain valid.
	ErrUnstableSeam = errors.New("overlap cannot be finalized within bounded context")
)

// Provider selects the ONNX Runtime execution provider for the encoder and
// decoder. Preprocessing always runs on CPU.
type Provider string

// Supported providers.
const (
	CPU  Provider = "cpu"
	CUDA Provider = "cuda" // appended explicitly; there is no CPU retry on failure
)

// Config locates the model assets and configures the sessions.
type Config struct {
	// ORTLibrary is an ONNX Runtime library path or OS-loader name. When set, the
	// engine initializes the environment through ortenv, which retains it until
	// process exit. When empty, the caller owns the environment and must
	// initialize it before New and destroy it only after Close.
	ORTLibrary       string
	PreprocessorPath string // nemo128.onnx
	EncoderPath      string // encoder-model.onnx (with .onnx.data beside it) or encoder-model.int8.onnx
	DecoderPath      string // decoder_joint-model.onnx or decoder_joint-model.int8.onnx
	// VocabPath overrides the embedded Parakeet TDT v3 vocabulary. Usually empty.
	VocabPath string
	Provider  Provider // empty selects CPU
	DeviceID  int      // CUDA device
	Threads   int      // intra-op threads per session; zero selects 1
	Verbose   bool     // write ORT node-placement diagnostics to stderr
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

// Timings is a diagnostic breakdown of one request into disjoint stages,
// summed over all windows. TranscribeStream excludes time spent in the callback.
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
// Seam is empty for the first window.
type ChunkResult struct {
	Index       int     `json:"index"`
	StartSample int     `json:"start_sample"`
	EndSample   int     `json:"end_sample"`
	Text        string  `json:"text"`
	Tokens      []Token `json:"tokens"`
	Seam        Seam    `json:"seam,omitempty"`
}

// Seam reports how a chunk was joined to the previous window's hypothesis.
type Seam string

// Seam methods.
const (
	SeamAligned   Seam = "aligned"   // matching words anchored the splice
	SeamTimestamp Seam = "timestamp" // no confident match; emission time decided ownership
)

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

// Initialization is a diagnostic breakdown of New's loading time. ModelSeconds
// includes metadata inspection, which loads temporary sessions with this binding
// version, as well as the retained inference sessions. RuntimeVersion is the
// loaded native ONNX Runtime version.
type Initialization struct {
	RuntimeVersion string  `json:"runtime_version"`
	RuntimeSeconds float64 `json:"runtime_initialization_seconds"`
	VocabSeconds   float64 `json:"vocabulary_loading_seconds"`
	ModelSeconds   float64 `json:"model_load_seconds"`
}

// TensorInfo describes one model input or output.
type TensorInfo struct {
	Name  string  `json:"name"`
	Shape []int64 `json:"shape"`
	Type  string  `json:"type"`
}

// GraphInfo describes one ONNX graph's inputs and outputs.
type GraphInfo struct {
	Inputs  []TensorInfo `json:"inputs"`
	Outputs []TensorInfo `json:"outputs"`
}

// ModelInfo describes the three loaded graphs.
type ModelInfo struct {
	Preprocessor GraphInfo `json:"preprocessor"`
	Encoder      GraphInfo `json:"encoder"`
	Decoder      GraphInfo `json:"decoder"`
}

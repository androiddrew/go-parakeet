package parakeet

import (
	"errors"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/androiddrew/go-parakeet/internal/assets"
	"github.com/androiddrew/go-parakeet/internal/ortruntime"
	ort "github.com/yalue/onnxruntime_go"
)

// Engine owns three model sessions. Requests and Close are serialized.
// Construct with New; the zero value is not an initialized engine.
type Engine struct {
	mu             sync.Mutex
	closed         bool
	preprocessor   *ort.DynamicAdvancedSession
	encoder        *ort.DynamicAdvancedSession
	decoder        *nativeDecoder
	vocab          assets.Vocabulary
	info           ModelInfo
	initialization Initialization
}

// New validates assets and tensor metadata and eagerly creates the sessions.
// Call Close on every successfully constructed engine.
func New(c Config) (_ *Engine, err error) {
	if c.Provider != "" && c.Provider != CPU && c.Provider != CUDA {
		return nil, fmt.Errorf("unsupported provider %q", c.Provider)
	}
	if c.Threads < 0 || c.DeviceID < 0 {
		return nil, errors.New("threads and device ID must be nonnegative")
	}
	for _, file := range []struct{ name, path string }{
		{"PreprocessorPath", c.PreprocessorPath}, {"EncoderPath", c.EncoderPath},
		{"DecoderPath", c.DecoderPath}, {"ConfigPath", c.ConfigPath}, {"VocabPath", c.VocabPath},
	} {
		if file.path == "" {
			return nil, fmt.Errorf("%s is required", file.name)
		}
		stat, e := os.Stat(file.path)
		if e != nil {
			return nil, fmt.Errorf("%s: %w", file.name, e)
		}
		if !stat.Mode().IsRegular() {
			return nil, fmt.Errorf("%s must be a regular file", file.name)
		}
	}
	e := &Engine{}
	defer func() {
		if err != nil {
			err = errors.Join(err, e.Close())
		}
	}()
	start := time.Now()
	if err = assets.LoadConfig(c.ConfigPath); err != nil {
		return nil, err
	}
	e.vocab, err = assets.LoadVocabulary(c.VocabPath)
	if err != nil {
		return nil, err
	}
	e.initialization.AssetsSeconds = time.Since(start).Seconds()
	start = time.Now()
	if err = ortruntime.Init(c.ORTLibrary); err != nil {
		return nil, err
	}
	e.initialization.RuntimeSeconds = time.Since(start).Seconds()
	e.initialization.RuntimeVersion = ort.GetVersion()
	start = time.Now()
	defer func() { e.initialization.ModelSeconds = time.Since(start).Seconds() }()
	cpu, err := sessionOptions(c, CPU)
	if err != nil {
		return nil, err
	}
	defer func() { err = errors.Join(err, cpu.Destroy()) }()
	e.preprocessor, e.info.Preprocessor, err = loadDynamic(c.PreprocessorPath, cpu, preInputs, preOutputs)
	if err != nil {
		return nil, err
	}
	options, err := sessionOptions(c, c.Provider)
	if err != nil {
		return nil, err
	}
	defer func() { err = errors.Join(err, options.Destroy()) }()
	e.encoder, e.info.Encoder, err = loadDynamic(c.EncoderPath, options, encInputs, encOutputs)
	if err != nil {
		return nil, err
	}
	e.info.Decoder, err = inspect(c.DecoderPath, options, decInputs, decOutputs)
	if err != nil {
		return nil, err
	}
	e.decoder, err = newNativeDecoder(c.DecoderPath, options)
	if err != nil {
		return nil, err
	}
	return e, nil
}

// Close releases all owned native objects and leaves the environment loaded.
func (e *Engine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil
	}
	e.closed = true
	var err error
	if e.decoder != nil {
		err = errors.Join(err, e.decoder.close())
		e.decoder = nil
	}
	if e.encoder != nil {
		err = errors.Join(err, e.encoder.Destroy())
		e.encoder = nil
	}
	if e.preprocessor != nil {
		err = errors.Join(err, e.preprocessor.Destroy())
		e.preprocessor = nil
	}
	e.vocab = assets.Vocabulary{}
	return err
}

func (e *Engine) Initialization() Initialization { return e.initialization }

// ModelInfo returns an owned metadata copy. Negative dimensions are dynamic.
func (e *Engine) ModelInfo() ModelInfo {
	clone := func(group []TensorInfo) []TensorInfo {
		out := append([]TensorInfo(nil), group...)
		for i := range out {
			out[i].Shape = append([]int64(nil), out[i].Shape...)
		}
		return out
	}
	graph := func(g GraphInfo) GraphInfo { return GraphInfo{clone(g.Inputs), clone(g.Outputs)} }
	return ModelInfo{graph(e.info.Preprocessor), graph(e.info.Encoder), graph(e.info.Decoder)}
}

// Package ortruntime selects process-global ONNX Runtime ownership.
package ortruntime

import (
	"errors"

	"github.com/androiddrew/ortenv"
	ort "github.com/yalue/onnxruntime_go"
)

var ErrNotInitialized = errors.New("ONNX Runtime is not initialized: set ORTLibrary or initialize the environment before New")

func Init(library string) error {
	if library != "" {
		return ortenv.Init(library)
	}
	if !ort.IsInitialized() {
		return ErrNotInitialized
	}
	return nil
}

// Package chunking plans frame-aligned, overlapping PCM windows without
// retaining a per-recording list of windows.
package chunking

import "fmt"

// FrameSamples is the 10 ms feature hop times the encoder's 8x subsampling.
const FrameSamples = 1280

const MinSamples = 320

// Plan uses constant space regardless of the recording's length.
type Plan struct {
	Total, Size, Overlap int
}

type Window struct {
	Start, End int // absolute sample positions, half-open
}

func New(total, size, overlap int) (Plan, error) {
	if total < MinSamples || size < MinSamples || size%FrameSamples != 0 {
		return Plan{}, fmt.Errorf("invalid recording/window size")
	}
	if overlap < 0 || overlap > size/2 || overlap%FrameSamples != 0 {
		return Plan{}, fmt.Errorf("overlap must be frame-aligned and at most half the window")
	}
	if total > size && overlap < FrameSamples {
		return Plan{}, fmt.Errorf("multiple windows require at least one frame of overlap")
	}
	return Plan{total, size, overlap}, nil
}

func (p Plan) Count() int {
	if p.Total <= p.Size {
		return 1
	}
	return 2 + (p.Total-p.Size-1)/(p.Size-p.Overlap)
}

// At requires 0 <= index < Count(). Window starts stay on the global encoder
// frame grid; the final window is clipped to the recording without padding.
func (p Plan) At(index int) Window {
	start := index * (p.Size - p.Overlap)
	return Window{start, start + min(p.Size, p.Total-start)}
}

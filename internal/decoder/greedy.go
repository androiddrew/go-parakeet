// Package decoder implements Parakeet's greedy token-and-duration state machine.
// Decision rules adapt istupakov/onnx-asr (MIT); see THIRD_PARTY.md. This version
// separates native execution/state commits and adds cancellation and validation.
package decoder

import (
	"context"
	"fmt"
	"math"
)

const MaxSymbolsPerFrame = 10

type Token struct{ ID, Frame, Advance int }

// Step evaluates a frame using the previously committed recurrent state and
// previous token. Commit accepts the newly produced state only for nonblank output.
type Step func(frame, previous int) (logits []float32, err error)

func Greedy(ctx context.Context, frames, vocabSize, blank, durations int, step Step, commit func()) ([]Token, error) {
	if frames <= 0 || vocabSize < 2 || blank < 0 || blank >= vocabSize || durations <= 0 {
		return nil, fmt.Errorf("invalid decoder dimensions")
	}
	var tokens []Token
	previous, emitted := blank, 0
	for frame := 0; frame < frames; {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		logits, err := step(frame, previous)
		if err != nil {
			return nil, fmt.Errorf("decode frame %d: %w", frame, err)
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if len(logits) != vocabSize+durations {
			return nil, fmt.Errorf("invalid decoder logits length %d", len(logits))
		}
		id, err := argmax(logits[:vocabSize])
		if err != nil {
			return nil, err
		}
		advance, err := argmax(logits[vocabSize:])
		if err != nil {
			return nil, err
		}
		if id != blank {
			commit()
			previous = id
			emitted++
			tokens = append(tokens, Token{id, frame, advance})
		}
		if advance > 0 {
			frame += advance
			emitted = 0
		} else if id == blank || emitted == MaxSymbolsPerFrame {
			frame++
			emitted = 0
		}
	}
	return tokens, nil
}

func argmax(values []float32) (int, error) {
	best := 0
	for i, v := range values {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return 0, fmt.Errorf("nonfinite decoder logit")
		}
		if v > values[best] {
			best = i
		}
	}
	return best, nil
}

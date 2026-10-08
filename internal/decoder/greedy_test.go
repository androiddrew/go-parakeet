package decoder

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"
)

func scores(token, advance int) []float32 {
	x := make([]float32, 8) // 3 tokens (blank = 2), 5 duration classes
	x[token], x[3+advance] = 1, 1
	return x
}

func TestStateAndFrameTransitions(t *testing.T) {
	// Blank leaves previous token and state unchanged. Nonblank zero duration
	// emits again at the same frame. Positive durations skip encoder frames.
	steps := []struct{ frame, previous, token, advance int }{
		{0, 2, 0, 0}, {0, 0, 2, 0}, {1, 0, 1, 2}, {3, 1, 2, 1},
	}
	calls, commits := 0, 0
	got, err := Greedy(context.Background(), 4, 3, 2, 5, func(frame, previous int) ([]float32, error) {
		if calls >= len(steps) {
			t.Fatal("unexpected extra step")
		}
		s := steps[calls]
		if frame != s.frame || previous != s.previous {
			t.Fatalf("step %d: frame=%d previous=%d", calls, frame, previous)
		}
		calls++
		return scores(s.token, s.advance), nil
	}, func() { commits++ })
	want := []Token{{0, 0, 0}, {1, 1, 2}}
	if err != nil || !reflect.DeepEqual(got, want) || calls != 4 || commits != 2 {
		t.Fatalf("%v, %v, calls=%d commits=%d", got, err, calls, commits)
	}
}

func TestNonProgressIsBounded(t *testing.T) {
	got, err := Greedy(context.Background(), 2, 3, 2, 5, func(int, int) ([]float32, error) { return scores(0, 0), nil }, func() {})
	if err != nil || len(got) != 2*MaxSymbolsPerFrame || got[MaxSymbolsPerFrame].Frame != 1 {
		t.Fatalf("%v, %v", got, err)
	}
}

func TestCancellationAfterNativeStep(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	committed := false
	_, err := Greedy(ctx, 2, 3, 2, 5, func(int, int) ([]float32, error) { cancel(); return scores(0, 1), nil }, func() { committed = true })
	if !errors.Is(err, context.Canceled) || committed {
		t.Fatalf("err=%v committed=%v", err, committed)
	}
}

func TestInvalidNativeOutput(t *testing.T) {
	for _, values := range [][]float32{nil, {0, 0}, {1, 0, 0, 0, 0, 0, 0, float32(math.NaN())}, {float32(math.Inf(1)), 0, 0, 0, 0, 0, 0, 0}} {
		_, err := Greedy(context.Background(), 1, 3, 2, 5, func(int, int) ([]float32, error) { return values, nil }, func() { t.Fatal("invalid output committed") })
		if err == nil {
			t.Fatal("accepted invalid logits")
		}
	}
	want := errors.New("native failure")
	_, err := Greedy(context.Background(), 1, 3, 2, 5, func(int, int) ([]float32, error) { return nil, want }, func() {})
	if !errors.Is(err, want) {
		t.Fatal(err)
	}
}

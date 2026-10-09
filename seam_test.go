package parakeet

import (
	"strings"
	"testing"

	"github.com/androiddrew/go-parakeet/internal/chunking"
)

func wordToken(piece string, frame int) Token { return Token{Piece: "▁" + piece, Frame: frame} }

func tokenText(tokens []Token) string {
	var b strings.Builder
	for _, token := range tokens {
		b.WriteString(strings.ReplaceAll(token.Piece, "▁", " "))
	}
	return strings.TrimSpace(b.String())
}

func TestSeamAlignment(t *testing.T) {
	for _, test := range []struct {
		name         string
		left, right  []Token
		boundary     int
		want, method string
	}{
		{"overlap", []Token{wordToken("open", 98), wordToken("the", 105), wordToken("blue", 110), wordToken("door.", 116)}, []Token{wordToken("the", 106), wordToken("blue", 111), wordToken("door", 117), wordToken("now", 125)}, 110, "open the blue door now", "aligned"},
		{"repetition", []Token{wordToken("very,", 100), wordToken("very", 106), wordToken("important.", 112)}, []Token{wordToken("very", 101), wordToken("very", 107), wordToken("important", 113), wordToken("today", 125)}, 108, "very, very important today", "aligned"},
		{"separate occurrences", []Token{wordToken("again", 98)}, []Token{wordToken("again", 122)}, 110, "again again", "timestamp"},
		{"tokenization", []Token{{Piece: "▁im", Frame: 104}, {Piece: "port", Frame: 105}, {Piece: "ant", Frame: 106}}, []Token{wordToken("important", 106), wordToken("work", 125)}, 105, "important work", "aligned"},
		{"no agreement", []Token{wordToken("red", 100), wordToken("boat", 116)}, []Token{wordToken("blue", 104), wordToken("boat", 116), wordToken("arrived", 125)}, 110, "red boat arrived", "aligned"},
		{"empty", nil, nil, 110, "", "timestamp"},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, method := reconcile(test.left, test.right, 96, 124, test.boundary)
			if text := tokenText(got); text != test.want || method != test.method {
				t.Fatalf("%q (%s), want %q (%s)", text, method, test.want, test.method)
			}
		})
	}
}

func TestWholeWordHoldback(t *testing.T) {
	tokens := []Token{wordToken("the", 5), wordToken("window", 10), {Piece: "sill", Frame: 14}, {Piece: ".", Frame: 15}}
	if n := safePrefix(tokens, 12); n != 1 {
		t.Fatalf("split a word: %d", n)
	}
	if n := safePrefix(tokens, 16); n != 4 {
		t.Fatal(n)
	}
}

func TestSeamPreservesRecoveredPrefix(t *testing.T) {
	left := []Token{wordToken("The", 198), wordToken("red", 200), wordToken("boat", 205)}
	right := []Token{wordToken("One", 183), wordToken("more", 188), wordToken("time,", 192), wordToken("the", 198), wordToken("red", 201), wordToken("boat", 206), wordToken("passed", 210)}
	got, method := reconcile(left, right, 180, 220, 200)
	if text := tokenText(got); text != "One more time, The red boat passed" || method != "aligned" {
		t.Fatalf("lost recovered speech: %q (%s)", text, method)
	}
}

func TestSeamPreservesDisjointRecoveredSpeech(t *testing.T) {
	left := []Token{wordToken("leave.", 159)}
	right := []Token{wordToken("One", 166), wordToken("more", 171), wordToken("time,", 175), wordToken("one", 183), wordToken("more", 187), wordToken("time.", 191)}
	got, method := reconcile(left, right, 160, 210, 185)
	if text := tokenText(got); text != "leave. One more time, one more time." || method != "timestamp" {
		t.Fatalf("discarded disjoint speech: %q (%s)", text, method)
	}
	if got, _ := reconcile(nil, right, 160, 210, 185); tokenText(got) != tokenText(right) {
		t.Fatal("empty left hypothesis discarded right speech")
	}
	if got, _ := reconcile(right, nil, 160, 210, 185); tokenText(got) != tokenText(right) {
		t.Fatal("empty right hypothesis discarded left speech")
	}
}

func TestQuietBoundary(t *testing.T) {
	samples := make([]float32, 60*chunking.FrameSamples)
	for i := range samples {
		samples[i] = 0.1
	}
	clear(samples[23*chunking.FrameSamples : 27*chunking.FrameSamples])
	if got := seamBoundary(samples, 10, 50); got != 25 {
		t.Fatalf("quiet boundary %d", got)
	}
	for i := range samples {
		samples[i] = 0.1
	}
	if got := seamBoundary(samples, 10, 50); got != 30 {
		t.Fatalf("midpoint %d", got)
	}
}

// Package assets validates the supported Parakeet export's vocabulary.
package assets

import (
	"bufio"
	"bytes"
	_ "embed"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

const (
	Features    = 128
	EncoderDim  = 1024
	StateLayers = 2
	StateDim    = 640
	VocabSize   = 8193
	Durations   = 5
)

// Vocabulary preserves raw SentencePiece entries and their explicit IDs.
type Vocabulary struct {
	Pieces []string
	Blank  int
}

// defaultVocab is vocab.txt from istupakov/parakeet-tdt-0.6b-v3-onnx at
// revision 8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce (CC-BY-4.0), converted
// from NVIDIA Parakeet TDT 0.6B v3. See THIRD_PARTY.md.
//
//go:embed vocab.txt
var defaultVocab []byte

// LoadVocabulary reads a SentencePiece "piece id" file. An empty path selects
// the embedded Parakeet TDT v3 vocabulary.
func LoadVocabulary(path string) (Vocabulary, error) {
	if path == "" {
		return parseVocabulary(bytes.NewReader(defaultVocab))
	}
	f, err := os.Open(path)
	if err != nil {
		return Vocabulary{}, fmt.Errorf("open vocabulary: %w", err)
	}
	defer f.Close()
	return parseVocabulary(f)
}

func parseVocabulary(r io.Reader) (Vocabulary, error) {
	v := Vocabulary{Pieces: make([]string, VocabSize), Blank: -1}
	seen := make(map[string]bool)
	s := bufio.NewScanner(r)
	count := 0
	for s.Scan() {
		line := s.Text()
		piece, number, ok := strings.Cut(line, " ")
		id, err := strconv.Atoi(number)
		if !ok || err != nil || piece == "" || !utf8.ValidString(piece) || id < 0 || id >= VocabSize || v.Pieces[id] != "" || seen[piece] {
			return Vocabulary{}, fmt.Errorf("malformed or duplicate vocabulary entry at line %d", count+1)
		}
		v.Pieces[id], seen[piece] = piece, true
		if piece == "<blk>" {
			v.Blank = id
		}
		count++
	}
	if err := s.Err(); err != nil {
		return Vocabulary{}, fmt.Errorf("read vocabulary: %w", err)
	}
	if count != VocabSize || v.Blank != VocabSize-1 {
		return Vocabulary{}, fmt.Errorf("require %d vocabulary entries with final <blk> token", VocabSize)
	}
	return v, nil
}

// Text reproduces onnx-asr's SentencePiece whitespace cleanup. Raw token pieces
// remain available separately. It preserves Unicode letters/numbers and punctuation.
func Text(pieces []string) string {
	var decoder TextDecoder
	return decoder.Append(pieces)
}

// TextDecoder preserves whitespace decisions across finalized chunks without
// retaining earlier text. A pending space is emitted only when a later word
// needs it; trailing whitespace requires no final flush.
type TextDecoder struct {
	space, haveText bool
}

func (d *TextDecoder) Append(pieces []string) string {
	var b strings.Builder
	for _, piece := range pieces {
		for _, r := range piece {
			if r == '▁' || unicode.IsSpace(r) {
				d.space = true
				continue
			}
			if d.space && d.haveText && (unicode.IsLetter(r) || unicode.IsNumber(r) || r == '_') {
				b.WriteByte(' ')
			}
			d.space = false
			d.haveText = true
			b.WriteRune(r)
		}
	}
	return b.String()
}

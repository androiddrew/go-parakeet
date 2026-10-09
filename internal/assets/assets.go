// Package assets validates the supported Parakeet export's configuration and vocabulary.
package assets

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

const (
	Features    = 128
	Subsampling = 8
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

func LoadConfig(path string) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read model config: %w", err)
	}
	var c struct {
		ModelType   string `json:"model_type"`
		Features    int    `json:"features_size"`
		Subsampling int    `json:"subsampling_factor"`
	}
	if err := json.Unmarshal(raw, &c); err != nil {
		return fmt.Errorf("parse model config: %w", err)
	}
	if c.ModelType != "nemo-conformer-tdt" || c.Features != Features || c.Subsampling != Subsampling {
		return fmt.Errorf("unsupported model config: require nemo-conformer-tdt, 128 features, subsampling 8")
	}
	return nil
}

func LoadVocabulary(path string) (Vocabulary, error) {
	f, err := os.Open(path)
	if err != nil {
		return Vocabulary{}, fmt.Errorf("open vocabulary: %w", err)
	}
	defer f.Close()
	v := Vocabulary{Pieces: make([]string, VocabSize), Blank: -1}
	seen := make(map[string]bool)
	s := bufio.NewScanner(f)
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

package assets

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestVocabularyValidation(t *testing.T) {
	var b strings.Builder
	for i := range VocabSize - 1 {
		fmt.Fprintf(&b, "piece%d %d\n", i, i)
	}
	fmt.Fprintf(&b, "<blk> %d\n", VocabSize-1)
	valid := b.String()
	for _, test := range []struct {
		name, data string
		ok         bool
	}{
		{"valid", valid, true},
		{"missing", "<blk> 8192\n", false},
		{"duplicate-id", strings.Replace(valid, "piece1 1\n", "piece1 0\n", 1), false},
		{"duplicate-piece", strings.Replace(valid, "piece1 1\n", "piece0 1\n", 1), false},
		{"negative-id", strings.Replace(valid, "piece1 1\n", "piece1 -1\n", 1), false},
		{"blank", strings.Replace(valid, "<blk>", "wrong", 1), false},
		{"malformed", valid + "garbage\n", false},
	} {
		t.Run(test.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "vocab.txt")
			if err := os.WriteFile(path, []byte(test.data), 0600); err != nil {
				t.Fatal(err)
			}
			v, err := LoadVocabulary(path)
			if (err == nil) != test.ok {
				t.Fatalf("%v", err)
			}
			if test.ok && (len(v.Pieces) != VocabSize || v.Blank != 8192) {
				t.Fatal("incorrect vocabulary")
			}
		})
	}
}

func TestTextAssembly(t *testing.T) {
	for _, test := range []struct {
		pieces []string
		want   string
	}{
		{[]string{"▁Hello", ",", "▁world", "!"}, "Hello, world!"},
		{[]string{"▁very", ",", "▁very", "▁important", "."}, "very, very important."},
		{[]string{"▁Καλημέρα", "▁κόσμε", "!"}, "Καλημέρα κόσμε!"},
		{[]string{"▁Привет", "▁мир", "▁!", "▁"}, "Привет мир!"},
		{[]string{"▁", "▁Bonjour", "▁", "▁à", "▁tous"}, "Bonjour à tous"},
	} {
		if got := Text(test.pieces); got != test.want {
			t.Fatalf("%q != %q", got, test.want)
		}
	}
}

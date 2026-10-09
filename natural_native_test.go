package parakeet

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"unicode"
)

func naturalWords(text string) []string {
	return strings.FieldsFunc(strings.ToLower(text), func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsNumber(r) && !unicode.IsMark(r) })
}

func wordEdits(want, got []string) int {
	row := make([]int, len(got)+1)
	for i := range row {
		row[i] = i
	}
	for i, a := range want {
		previous := row[0]
		row[0] = i + 1
		for j, b := range got {
			old := row[j+1]
			cost := 0
			if a != b {
				cost = 1
			}
			row[j+1] = min(row[j]+1, row[j+1]+1, previous+cost)
			previous = old
		}
	}
	return row[len(got)]
}

func testNaturalSpeech(t *testing.T, c Config) {
	t.Helper()
	var manifest struct {
		Entries []struct {
			Name, Language, File, SHA256, Transcript string
			Samples                                  int
		}
	}
	raw, err := os.ReadFile("testdata/natural/manifest.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &manifest); err != nil {
		t.Fatal(err)
	}
	var baseline map[string]struct {
		Text   string
		Tokens []Token
	}
	raw, err = os.ReadFile(filepath.Join("testdata/natural", testPrecision(t)+".json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &baseline); err != nil {
		t.Fatal(err)
	}
	e, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	totals := map[string][2]int{}
	for _, entry := range manifest.Entries {
		samples := readFloat32(t, filepath.Join("testdata/natural", entry.File), entry.SHA256)
		if len(samples) != entry.Samples {
			t.Fatal("natural input length mismatch")
		}
		result, err := e.Transcribe(context.Background(), Request{Samples: samples, SampleRate: SampleRate, Chunking: ChunkOptions{SinglePass: true}})
		if err != nil {
			t.Fatal(entry.Name, err)
		}
		ref, ok := baseline[entry.Name]
		if !ok || result.Text != ref.Text {
			t.Errorf("%s reference mismatch:\ngot %s\nwant %s", entry.Name, result.Text, ref.Text)
		}
		// CPU reference decisions should match exactly in the pinned native
		// configuration. CUDA is checked for text, allowing emission-time drift.
		if c.Provider != CUDA && !reflect.DeepEqual(result.Tokens, ref.Tokens) {
			t.Errorf("%s token decisions differ from Python", entry.Name)
		}
		want, got := naturalWords(entry.Transcript), naturalWords(result.Text)
		count := totals[entry.Language]
		count[0] += wordEdits(want, got)
		count[1] += len(want)
		totals[entry.Language] = count
	}
	for _, language := range []string{"en_us", "fr_fr", "de_de", "es_419", "ru_ru", "el_gr"} {
		count := totals[language]
		if count[1] == 0 {
			t.Fatal("missing language", language)
		}
		t.Logf("%s: %d/%d word edits (%.2f%%); model accuracy, separate from Go/Python parity", language, count[0], count[1], 100*float64(count[0])/float64(count[1]))
	}
}

func TestNativeNaturalSpeech(t *testing.T) {
	c := nativeConfig(t)
	if !nativeSubprocess(t) {
		return
	}
	testNaturalSpeech(t, c)
}

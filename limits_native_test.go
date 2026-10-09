package parakeet

import (
	"os"
	"strings"
	"testing"

	ort "github.com/yalue/onnxruntime_go"
)

// This bypasses the public 300-second pass cap to validate the pinned export's
// relative-position table boundary. It is an explicit, memory-intensive opt-in.
func testEncoderLimit(t *testing.T, c Config) {
	t.Helper()
	e, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	for _, frames := range []int{40009, 40000, 40001} {
		encoded, err := runSequence(e.encoder, make([]float32, 128*frames), ort.NewShape(1, 128, int64(frames)), frames, 1024)
		if frames == 40009 {
			if err == nil || !strings.Contains(strings.ToLower(err.Error()), "broadcast") {
				t.Fatalf("expected positional-table broadcast rejection, got %v", err)
			}
			t.Log("40009 mel frames (5002 encoder frames) rejected by positional-table broadcast check")
		} else {
			want := (frames + 7) / 8
			if err != nil || encoded.length != want {
				t.Fatalf("encoder boundary: wanted %d, length=%d err=%v", want, encoded.length, err)
			}
			t.Logf("%d mel frames -> %d encoder frames accepted", frames, encoded.length)
		}
	}
}

func TestNativeEncoderLimit(t *testing.T) {
	c := nativeConfig(t)
	if os.Getenv("PARAKEET_TEST_LIMIT") != "1" {
		t.Skip("set PARAKEET_TEST_LIMIT=1 for exported-model boundary validation")
	}
	if !nativeSubprocess(t) {
		return
	}
	testEncoderLimit(t, c)
}

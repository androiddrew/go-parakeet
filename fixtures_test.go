package parakeet

import (
	"bufio"
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFrozenFixtures(t *testing.T) {
	for _, dir := range []string{"testdata/native"} {
		file, err := os.Open(filepath.Join(dir, "SHA256SUMS"))
		if err != nil {
			t.Fatal(err)
		}
		scanner := bufio.NewScanner(file)
		count := 0
		for scanner.Scan() {
			fields := strings.Fields(scanner.Text())
			if len(fields) != 2 {
				t.Fatal("malformed fixture checksum")
			}
			raw, err := os.ReadFile(filepath.Join(dir, fields[1]))
			if err != nil {
				t.Fatal(err)
			}
			if fmt.Sprintf("%x", sha256.Sum256(raw)) != fields[0] {
				t.Fatal("fixture changed:", fields[1])
			}
			count++
		}
		if err := scanner.Err(); err != nil {
			t.Fatal(err)
		}
		if err := file.Close(); err != nil {
			t.Fatal(err)
		}
		if count == 0 {
			t.Fatal("empty fixture manifest")
		}
	}
}

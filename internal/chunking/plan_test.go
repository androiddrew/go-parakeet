package chunking

import "testing"

func TestPlanCoverage(t *testing.T) {
	for _, size := range []int{25 * FrameSamples, 100 * FrameSamples, 375 * FrameSamples} {
		for _, overlap := range []int{FrameSamples, 4 * FrameSamples, size / 2 / FrameSamples * FrameSamples} {
			for _, total := range []int{320, size - 1, size, size + 1, size * 2, size*3 + 17} {
				p, err := New(total, size, overlap)
				if err != nil {
					t.Fatal(err)
				}
				end := 0
				for i := range p.Count() {
					w := p.At(i)
					if w.Start%FrameSamples != 0 || w.End > total || w.End-w.Start > size || w.End-w.Start < MinSamples {
						t.Fatalf("invalid window %+v", w)
					}
					if i == 0 && w.Start != 0 || i > 0 && w.Start != end-overlap {
						t.Fatalf("coverage gap: %+v after %d", w, end)
					}
					end = w.End
				}
				if end != total {
					t.Fatalf("lost tail: %d != %d", end, total)
				}
			}
		}
	}
}

func TestLargePlanAndValidation(t *testing.T) {
	// Arithmetic remains bounded even when a plan is too large to materialize.
	maximum := int(^uint(0) >> 1)
	p, err := New(maximum, 375*FrameSamples, 50*FrameSamples)
	if err != nil {
		t.Fatal(err)
	}
	if p.At(p.Count()-1).End != maximum {
		t.Fatal("overflow in final window")
	}
	for _, v := range [][3]int{{0, 32000, 1280}, {320, 320, 0}, {64000, 32000, 0}, {64000, 32000, -1280}, {64000, 32000, 16640}, {64000, 32000, 100}} {
		if _, err := New(v[0], v[1], v[2]); err == nil {
			t.Fatalf("accepted %v", v)
		}
	}
}

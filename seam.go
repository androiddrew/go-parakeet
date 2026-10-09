package parakeet

import (
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/androiddrew/go-parakeet/internal/chunking"
)

// A seam word groups whole SentencePiece words (including trailing punctuation)
// so a splice never combines the first half of one hypothesis with another's
// second half. Keys ignore case/punctuation only for alignment, not output.
type seamWord struct {
	first, end int // token indexes
	frame      int // midpoint of first/last token emissions
	key        string
}

const seamDriftFrames = 8 // 640 ms of disagreement between overlapping hypotheses

func words(tokens []Token) []seamWord {
	var result []seamWord
	var key strings.Builder
	start := 0
	finish := func(end int) {
		result = append(result, seamWord{start, end, tokens[start].Frame + (tokens[end-1].Frame-tokens[start].Frame)/2, key.String()})
		key.Reset()
		start = end
	}
	for i, token := range tokens {
		first, _ := utf8.DecodeRuneInString(token.Piece)
		if i > start && (first == '▁' || unicode.IsSpace(first)) {
			finish(i)
		}
		for _, r := range token.Piece {
			if unicode.IsLetter(r) || unicode.IsNumber(r) || unicode.IsMark(r) {
				key.WriteRune(unicode.ToLower(r))
			}
		}
	}
	if start < len(tokens) {
		finish(len(tokens))
	}
	return result
}

// safePrefix holds a whole word if any of its pieces touches the next overlap.
func safePrefix(tokens []Token, overlapStartFrame int) int {
	for _, word := range words(tokens) {
		if tokens[word.end-1].Frame >= overlapStartFrame {
			return word.first
		}
	}
	return len(tokens)
}

func distance(a, b int) int {
	if a < b {
		return b - a
	}
	return a - b
}

// candidates bounds the alignment matrix even for dense/pathological decoder
// output and large overlaps. Only words near the proposed acoustic seam matter.
func candidates(all []seamWord, from, to, boundary int) []seamWord {
	var result []seamWord
	for _, word := range all {
		if word.key != "" && word.frame >= from && word.frame < to && distance(word.frame, boundary) <= 25 {
			result = append(result, word)
		}
	}
	if len(result) > 64 {
		middle := sort.Search(len(result), func(i int) bool { return result[i].frame >= boundary })
		start := max(0, min(middle-32, len(result)-64))
		result = result[start : start+64]
	}
	return result
}

type wordMatch struct{ left, right seamWord }

// align finds a monotonic, time-constrained word alignment. Time scoring avoids
// confusing adjacent occurrences of repeated words. Different tokenizations of
// the same word can match; raw spelling/punctuation from the chosen side wins.
func align(left, right []seamWord) []wordMatch {
	width := len(right) + 1
	dp := make([]int, (len(left)+1)*width)
	score := func(a, b seamWord) int {
		d := distance(a.frame, b.frame)
		if a.key != b.key || d > seamDriftFrames {
			return 0
		} // at most 640 ms of emission drift
		return 100 - 8*d
	}
	for i := len(left) - 1; i >= 0; i-- {
		for j := len(right) - 1; j >= 0; j-- {
			dp[i*width+j] = max(dp[(i+1)*width+j], dp[i*width+j+1])
			if s := score(left[i], right[j]); s > 0 {
				dp[i*width+j] = max(dp[i*width+j], s+dp[(i+1)*width+j+1])
			}
		}
	}
	var matches []wordMatch
	for i, j := 0, 0; i < len(left) && j < len(right); {
		if s := score(left[i], right[j]); s > 0 && dp[i*width+j] == s+dp[(i+1)*width+j+1] {
			matches = append(matches, wordMatch{left[i], right[j]})
			i++
			j++
		} else if dp[(i+1)*width+j] >= dp[i*width+j+1] {
			i++
		} else {
			j++
		}
	}
	return matches
}

// reconcile owns the unresolved left tail and a complete right window. It
// chooses a word-aligned splice near the acoustic boundary. Ambiguous/missing
// matches fall back to timestamp ownership rather than lexical deduplication.
func reconcile(left, right []Token, from, to, boundary int) ([]Token, Seam) {
	lw, rw := words(left), words(right)
	lc, rc := candidates(lw, from, to, boundary), candidates(rw, from, to, boundary)
	matches := align(lc, rc)
	confident := len(matches) >= 2
	if len(matches) == 1 {
		m := matches[0]
		occurrences := func(ws []seamWord) int {
			n := 0
			for _, w := range ws {
				if w.key == m.left.key {
					n++
				}
			}
			return n
		}
		confident = utf8.RuneCountInString(m.left.key) >= 4 && distance(m.left.frame, m.right.frame) <= 2 && occurrences(lc) == 1 && occurrences(rc) == 1
	}
	leftEnd, rightStart := 0, len(right)
	method := SeamTimestamp
	if confident {
		best, bestDistance := matches[0], int(^uint(0)>>1)
		for _, m := range matches {
			if d := distance(m.left.frame+(m.right.frame-m.left.frame)/2, boundary); d < bestDistance {
				best, bestDistance = m, d
			}
		}
		leftEnd, rightStart = best.left.end, best.right.end
		method = SeamAligned
	} else if len(left) == 0 || len(right) == 0 || left[len(left)-1].Frame+2 < right[0].Frame {
		// Disjoint emission spans need no midpoint cut: doing so can discard
		// speech recovered by the other hypothesis inside an otherwise empty
		// part of the overlap. Retain both spans in chronological order.
		leftEnd, rightStart = len(left), 0
	} else {
		for _, word := range lw {
			if word.frame >= boundary {
				break
			}
			leftEnd = word.end
		}
		for _, word := range rw {
			if word.frame >= boundary {
				rightStart = word.first
				break
			}
		}
	}
	// A left hypothesis may contain no emissions in the beginning of the
	// overlap while the right one recovers speech there. Preserve complete
	// right words strictly before the first left emission, rather than losing
	// them merely because an alignment anchor occurs later. Require a temporal
	// gap and different keys to avoid duplicating a drifted first word.
	leadingEnd := 0
	if confident && leftEnd > 0 {
		for _, word := range rw {
			if word.end > rightStart || right[word.end-1].Frame+2 > left[0].Frame || word.key == lw[0].key {
				break
			}
			leadingEnd = word.end
		}
	}
	merged := make([]Token, 0, leadingEnd+leftEnd+len(right)-rightStart)
	merged = append(merged, right[:leadingEnd]...)
	merged = append(merged, left[:leftEnd]...)
	merged = append(merged, right[rightStart:]...)
	return merged, method
}

// seamBoundary prefers the centre of a sustained quiet run in the inner half
// of an overlap. If none exists, use its midpoint. No additional VAD model is
// required. Returned positions are absolute encoder frames.
func seamBoundary(samples []float32, from, to int) int {
	middle := from + (to-from)/2
	lo, hi := from+(to-from)/4, to-(to-from)/4
	power := make([]float64, hi-lo)
	var peak float64
	for frame := lo; frame < hi; frame++ {
		start := frame * chunking.FrameSamples
		end := min(len(samples), start+chunking.FrameSamples)
		for _, sample := range samples[start:end] {
			power[frame-lo] += float64(sample) * float64(sample)
		}
		power[frame-lo] /= float64(end - start)
		peak = max(peak, power[frame-lo])
	}
	threshold := max(1e-8, peak*0.001)
	best, bestLength := middle, 1
	for i := 0; i < len(power); {
		if power[i] > threshold {
			i++
			continue
		}
		end := i + 1
		for end < len(power) && power[end] <= threshold {
			end++
		}
		centre := lo + i + (end-i)/2
		if end-i > bestLength || (end-i == bestLength && distance(centre, middle) < distance(best, middle)) {
			best, bestLength = centre, end-i
		}
		i = end
	}
	return best
}

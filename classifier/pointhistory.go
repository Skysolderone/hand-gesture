package classifier

import "v1/cbridge"

const HistoryLen = 16

// PointHistory keeps a ring-buffer of the last HistoryLen (x,y) coordinates.
type PointHistory struct {
	buf  [HistoryLen][2]int
	head int
	size int
}

// Push adds a point to the history.
func (ph *PointHistory) Push(p [2]int) {
	ph.buf[ph.head] = p
	ph.head = (ph.head + 1) % HistoryLen
	if ph.size < HistoryLen {
		ph.size++
	}
}

// Clear resets the history.
func (ph *PointHistory) Clear() {
	ph.size = 0
	ph.head = 0
}

// Full returns true when HistoryLen frames have been collected.
func (ph *PointHistory) Full() bool { return ph.size == HistoryLen }

// Len returns the current number of stored points.
func (ph *PointHistory) Len() int { return ph.size }

// Slice returns a slice of size points in chronological order.
func (ph *PointHistory) Slice() [][2]int {
	out := make([][2]int, ph.size)
	start := (ph.head - ph.size + HistoryLen*2) % HistoryLen
	for i := 0; i < ph.size; i++ {
		out[i] = ph.buf[(start+i)%HistoryLen]
	}
	return out
}

// ClassifyPointHistory classifies the 32-float pre-processed point history.
func ClassifyPointHistory(input []float32) int {
	if len(input) < 32 {
		return 0
	}
	return cbridge.RunPointHistoryClassifier(input[:32])
}

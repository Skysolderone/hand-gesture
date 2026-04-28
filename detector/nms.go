package detector

import "sort"

// iou computes Intersection-over-Union for two boxes in (cx,cy,w,h) format.
func iou(a, b Detection) float32 {
	ax1, ay1 := a.CX-a.W/2, a.CY-a.H/2
	ax2, ay2 := a.CX+a.W/2, a.CY+a.H/2
	bx1, by1 := b.CX-b.W/2, b.CY-b.H/2
	bx2, by2 := b.CX+b.W/2, b.CY+b.H/2

	ix1 := max32(ax1, bx1)
	iy1 := max32(ay1, by1)
	ix2 := min32(ax2, bx2)
	iy2 := min32(ay2, by2)

	iw := ix2 - ix1
	ih := iy2 - iy1
	if iw <= 0 || ih <= 0 {
		return 0
	}
	inter := iw * ih
	aArea := a.W * a.H
	bArea := b.W * b.H
	return inter / (aArea + bArea - inter)
}

func max32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}
func min32(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

// NonMaxSuppression applies greedy NMS.
func NonMaxSuppression(dets []Detection, iouThresh float32) []Detection {
	if len(dets) == 0 {
		return nil
	}
	sort.Slice(dets, func(i, j int) bool { return dets[i].Score > dets[j].Score })
	suppressed := make([]bool, len(dets))
	var kept []Detection
	for i := range dets {
		if suppressed[i] {
			continue
		}
		kept = append(kept, dets[i])
		for j := i + 1; j < len(dets); j++ {
			if !suppressed[j] && iou(dets[i], dets[j]) > iouThresh {
				suppressed[j] = true
			}
		}
	}
	return kept
}

package preprocess

import "math"

// Landmark converts 21 pixel-coordinate keypoints into the 42-float
// normalised vector expected by KeypointClassifier.
//
// Algorithm (direct port of app.py pre_process_landmark):
//  1. Make all coords relative to kp[0] (wrist).
//  2. Flatten to 1-D slice of length 42.
//  3. Divide by max(abs) to normalise to [-1, 1].
func Landmark(pts [21][2]int) []float32 {
	baseX, baseY := pts[0][0], pts[0][1]

	rel := make([]float32, 42)
	for i, p := range pts {
		rel[i*2+0] = float32(p[0] - baseX)
		rel[i*2+1] = float32(p[1] - baseY)
	}

	maxAbs := float32(0)
	for _, v := range rel {
		if a := float32(math.Abs(float64(v))); a > maxAbs {
			maxAbs = a
		}
	}
	if maxAbs > 0 {
		for i := range rel {
			rel[i] /= maxAbs
		}
	}
	return rel
}

// PointHistory converts HistoryLen (x,y) pixel points into the 32-float
// normalised vector expected by PointHistoryClassifier.
//
// Algorithm (port of app.py pre_process_point_history):
//  1. Make all coords relative to the first point.
//  2. Normalise by image width / height.
func PointHistory(pts [][2]int, imgW, imgH int) []float32 {
	if len(pts) == 0 {
		return nil
	}
	baseX, baseY := pts[0][0], pts[0][1]
	out := make([]float32, len(pts)*2)
	for i, p := range pts {
		out[i*2+0] = float32(p[0]-baseX) / float32(imgW)
		out[i*2+1] = float32(p[1]-baseY) / float32(imgH)
	}
	return out
}

// PalmCenter computes the average of a subset of landmark indices used to
// track overall hand movement (port of app.py calc_palm_center).
func PalmCenter(pts [21][2]int) [2]int {
	indices := [5]int{0, 5, 9, 13, 17}
	sumX, sumY := 0, 0
	for _, idx := range indices {
		sumX += pts[idx][0]
		sumY += pts[idx][1]
	}
	n := len(indices)
	return [2]int{sumX / n, sumY / n}
}

// BoundingRect returns [x1,y1,x2,y2] bounding rect of landmarks in pixels.
func BoundingRect(pts [21][2]int) [4]int {
	x1, y1 := pts[0][0], pts[0][1]
	x2, y2 := x1, y1
	for _, p := range pts[1:] {
		if p[0] < x1 {
			x1 = p[0]
		}
		if p[1] < y1 {
			y1 = p[1]
		}
		if p[0] > x2 {
			x2 = p[0]
		}
		if p[1] > y2 {
			y2 = p[1]
		}
	}
	return [4]int{x1, y1, x2, y2}
}

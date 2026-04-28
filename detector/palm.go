package detector

import (
	"fmt"
	"math"

	"v1/cbridge"
	"gocv.io/x/gocv"
)

const (
	PalmInputSize   = 192
	PalmScoreThresh = 0.35 // lowered for easier detection
	PalmNMSThresh   = 0.3
)

// Detection holds one decoded palm detection in normalised [0,1] coordinates.
type Detection struct {
	Score float32
	CX, CY, W, H float32      // bounding box centre + size
	Keypoints     [7][2]float32 // 7 palm keypoints (cx,cy) normalised
}

// Detector wraps anchor state for palm detection.
type Detector struct {
	anchors    []Anchor
	numAnchors int
}

// NewDetector creates a Detector. numAnchors must equal the model's anchor count.
func NewDetector(numAnchors int) *Detector {
	anchors := GeneratePalmAnchors()
	if len(anchors) != numAnchors {
		// Fallback: truncate or warn
		println("[detector] WARNING: generated", len(anchors), "anchors but model expects", numAnchors)
		if len(anchors) > numAnchors {
			anchors = anchors[:numAnchors]
		}
	}
	return &Detector{anchors: anchors, numAnchors: len(anchors)}
}

var debugFrameCount int

// Detect runs palm detection on frame and returns filtered detections.
func (d *Detector) Detect(frame gocv.Mat) []Detection {
	bytes := frame.ToBytes()
	w, h := frame.Cols(), frame.Rows()

	scores, boxes := cbridge.RunPalmDetection(bytes, w, h, d.numAnchors)

	// Debug: print max sigmoid score for first 10 frames then every 60
	debugFrameCount++
	if debugFrameCount <= 10 || debugFrameCount%60 == 0 {
		maxS := float32(0)
		for _, s := range scores {
			if v := sigmoid(s); v > maxS {
				maxS = v
			}
		}
		fmt.Printf("[detect] frame=%d maxSigmoidScore=%.3f\n", debugFrameCount, maxS)
	}

	raw := decodeBoxes(scores, boxes, d.anchors)
	if debugFrameCount <= 10 || debugFrameCount%60 == 0 {
		fmt.Printf("[detect] raw=%d after_nms=%d\n", len(raw),
			len(NonMaxSuppression(append([]Detection(nil), raw...), PalmNMSThresh)))
	}
	return NonMaxSuppression(raw, PalmNMSThresh)
}

// sigmoid applies the logistic function.
func sigmoid(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(-float64(x))))
}

// decodeBoxes decodes raw model outputs into Detections.
func decodeBoxes(scores, boxes []float32, anchors []Anchor) []Detection {
	var dets []Detection
	n := len(anchors)
	for i := 0; i < n && i < len(scores); i++ {
		s := sigmoid(scores[i])
		if s < PalmScoreThresh {
			continue
		}
		a := anchors[i]
		b := boxes[i*18:]

		cx := b[0]/PalmInputSize + a.CX
		cy := b[1]/PalmInputSize + a.CY
		bw := b[2] / PalmInputSize
		bh := b[3] / PalmInputSize

		var kps [7][2]float32
		for k := 0; k < 7; k++ {
			kps[k][0] = b[4+k*2]/PalmInputSize + a.CX
			kps[k][1] = b[5+k*2]/PalmInputSize + a.CY
		}
		dets = append(dets, Detection{
			Score: s,
			CX: cx, CY: cy,
			W: bw, H: bh,
			Keypoints: kps,
		})
	}
	return dets
}

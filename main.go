package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"time"

	"v1/cbridge"
	"v1/classifier"
	"v1/detector"
	"v1/preprocess"
	"v1/render"

	"gocv.io/x/gocv"
)

const (
	leftGestureID  = 4
	rightGestureID = 5

	directionValidPointRatio   = 0.75
	directionMinDistanceRatio  = 0.05
	directionDominanceRatio    = 1.8
	directionMinProgressRatio  = 0.6
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("usage: hand-gestures <camera-id>")
		return
	}
	deviceID := os.Args[1]

	// ---- load labels -------------------------------------------------------
	kpLabels := loadCSVLabels("labels/keypoint_classifier_label.csv")
	phLabels := loadCSVLabels("labels/point_history_classifier_label.csv")

	// ---- init TFLite models ------------------------------------------------
	fmt.Println("Loading TFLite models …")
	info, err := cbridge.Init(
		"models/palm_detection_lite.tflite",
		"models/hand_landmark_lite.tflite",
		"models/keypoint_classifier.tflite",
		"models/point_history_classifier.tflite",
	)
	if err != nil {
		fmt.Println("Error loading models:", err)
		return
	}
	fmt.Printf("Palm anchors: %d\n", info.NumPalmAnchors)

	// ---- detector ----------------------------------------------------------
	det := detector.NewDetector(info.NumPalmAnchors)

	// ---- camera ------------------------------------------------------------
	webcam, err := gocv.OpenVideoCapture(deviceID)
	if err != nil {
		fmt.Println("Error opening camera:", deviceID)
		return
	}
	defer webcam.Close()
	webcam.Set(gocv.VideoCaptureFrameWidth, 960)
	webcam.Set(gocv.VideoCaptureFrameHeight, 540)

	window := gocv.NewWindow("Hand Gesture Recognition")
	defer window.Close()

	frame := gocv.NewMat()
	defer frame.Close()

	// ---- history buffers ---------------------------------------------------
	var pointHistory classifier.PointHistory
	var handDirHistory [classifier.HistoryLen][2]int
	handDirLen := 0
	handDirHead := 0

	var fingerGestureHistory [classifier.HistoryLen]int
	fgHead, fgLen := 0, 0

	// ---- FPS ---------------------------------------------------------------
	var fpsBuf [10]float64
	fpsIdx := 0
	fpsFull := false
	lastTime := time.Now()

	fmt.Printf("Capturing from device: %v  (press ESC to quit)\n", deviceID)

	for {
		if ok := webcam.Read(&frame); !ok {
			fmt.Println("camera closed")
			return
		}
		if frame.Empty() {
			continue
		}

		// FPS
		now := time.Now()
		dt := now.Sub(lastTime).Seconds()
		lastTime = now
		if dt > 0 {
			fpsBuf[fpsIdx] = 1.0 / dt
			fpsIdx = (fpsIdx + 1) % 10
			if fpsIdx == 0 {
				fpsFull = true
			}
		}
		n := 10
		if !fpsFull {
			n = fpsIdx
		}
		fps := 0.0
		for i := 0; i < n; i++ {
			fps += fpsBuf[i]
		}
		if n > 0 {
			fps /= float64(n)
		}

		// Mirror
		gocv.Flip(frame, &frame, 1)

		// ---- palm detection ------------------------------------------------
		palms := det.Detect(frame)

		handSignLabel := ""
		gestureLabel := ""

		if len(palms) > 0 {
			d := palms[0]

			// Hand patch + landmarks
			patch, rotMat := detector.GetHandPatch(frame, d)
			lmResult := detector.DetectLandmarks(patch, rotMat)
			patch.Close()
			rotMat.Close()

			pts := lmResult.Points

			// Track palm centre for horizontal gesture
			center := preprocess.PalmCenter(pts)
			handDirHistory[handDirHead] = center
			handDirHead = (handDirHead + 1) % classifier.HistoryLen
			if handDirLen < classifier.HistoryLen {
				handDirLen++
			}
			dirSlice := ringSlice(handDirHistory[:], handDirHead, handDirLen)

			// Keypoint classification
			ppLandmark := preprocess.Landmark(pts)
			handSignID := classifier.ClassifyKeypoint(ppLandmark)
			if handSignID < len(kpLabels) {
				handSignLabel = kpLabels[handSignID]
			}

			// Update point history
			const pointerID = 2
			if handSignID == pointerID {
				pointHistory.Push(pts[8]) // index fingertip
			} else {
				pointHistory.Push([2]int{0, 0})
			}

			// Dynamic gesture
			fingerGestureID := 0
			dirID := classifyHorizontalGesture(dirSlice, frame.Cols(), frame.Rows())
			if dirID >= 0 {
				fingerGestureID = dirID
				fgLen = 0
				fgHead = 0
			} else if pointHistory.Full() {
				ppHist := preprocess.PointHistory(pointHistory.Slice(), frame.Cols(), frame.Rows())
				fingerGestureID = classifier.ClassifyPointHistory(ppHist)
			}

			fingerGestureHistory[fgHead] = fingerGestureID
			fgHead = (fgHead + 1) % classifier.HistoryLen
			if fgLen < classifier.HistoryLen {
				fgLen++
			}
			mostCommonID := mostCommon(fingerGestureHistory[:fgLen])
			if mostCommonID < len(phLabels) {
				gestureLabel = phLabels[mostCommonID]
			}

			// Render
			brect := preprocess.BoundingRect(pts)
			render.DrawBoundingRect(&frame, brect)
			render.DrawLandmarks(&frame, pts)

			handLabel := "L"
			if lmResult.Handedness > 0.5 {
				handLabel = "R"
			}
			render.DrawInfoText(&frame, brect, handLabel, handSignLabel, gestureLabel)
		} else {
			pointHistory.Push([2]int{0, 0})
			handDirHistory[handDirHead] = [2]int{0, 0}
			handDirHead = (handDirHead + 1) % classifier.HistoryLen
			if handDirLen < classifier.HistoryLen {
				handDirLen++
			}
		}

		render.DrawPointHistory(&frame, pointHistory.Slice())
		render.DrawFPS(&frame, fps)

		window.IMShow(frame)
		if window.WaitKey(1) == 27 {
			break
		}
	}
}

// ringSlice returns elements from a ring buffer in chronological order.
func ringSlice(buf [][2]int, head, length int) [][2]int {
	n := len(buf)
	out := make([][2]int, length)
	start := (head - length + n*2) % n
	for i := 0; i < length; i++ {
		out[i] = buf[(start+i)%n]
	}
	return out
}

// mostCommon returns the most frequent value in the slice.
func mostCommon(s []int) int {
	counts := make(map[int]int)
	for _, v := range s {
		counts[v]++
	}
	best, bestC := 0, 0
	for k, c := range counts {
		if c > bestC {
			best, bestC = k, c
		}
	}
	return best
}

// classifyHorizontalGesture detects left/right swipe gestures.
// Returns leftGestureID, rightGestureID, or -1 if no gesture.
func classifyHorizontalGesture(pts [][2]int, imgW, imgH int) int {
	if len(pts) == 0 {
		return -1
	}
	var valid [][2]int
	for _, p := range pts {
		if p[0] != 0 || p[1] != 0 {
			valid = append(valid, p)
		}
	}
	if len(valid) < 2 {
		return -1
	}
	if float64(len(valid)) < float64(len(pts))*directionValidPointRatio {
		return -1
	}
	dx := valid[len(valid)-1][0] - valid[0][0]
	dy := valid[len(valid)-1][1] - valid[0][1]
	ndx := float64(dx) / float64(imgW)
	ndy := float64(dy) / float64(imgH)

	if math.Abs(ndx) < directionMinDistanceRatio {
		return -1
	}
	if math.Abs(ndx) < math.Abs(ndy)*directionDominanceRatio {
		return -1
	}

	dir := 1
	if dx < 0 {
		dir = -1
	}
	advancing := 0
	for i := 0; i < len(valid)-1; i++ {
		step := valid[i+1][0] - valid[i][0]
		if step*dir > 0 {
			advancing++
		}
	}
	if float64(advancing)/float64(len(valid)-1) < directionMinProgressRatio {
		return -1
	}
	if dx < 0 {
		return leftGestureID
	}
	return rightGestureID
}

// loadCSVLabels reads a single-column CSV and returns label strings.
func loadCSVLabels(path string) []string {
	f, err := os.Open(path)
	if err != nil {
		fmt.Println("Warning: cannot open", path, err)
		return nil
	}
	defer f.Close()
	r := csv.NewReader(bufio.NewReader(f))
	rows, _ := r.ReadAll()
	var labels []string
	for _, row := range rows {
		if len(row) > 0 {
			labels = append(labels, row[0])
		}
	}
	return labels
}

package render

import (
	"fmt"
	"image"
	"image/color"

	"gocv.io/x/gocv"
)

var (
	black = color.RGBA{0, 0, 0, 255}
	white = color.RGBA{255, 255, 255, 255}
	green = color.RGBA{152, 251, 152, 255} // pale green for trail
)

// hand bone connections: pairs of landmark indices
var connections = [][2]int{
	// thumb
	{2, 3}, {3, 4},
	// index
	{5, 6}, {6, 7}, {7, 8},
	// middle
	{9, 10}, {10, 11}, {11, 12},
	// ring
	{13, 14}, {14, 15}, {15, 16},
	// pinky
	{17, 18}, {18, 19}, {19, 20},
	// palm
	{0, 1}, {1, 2}, {2, 5}, {5, 9}, {9, 13}, {13, 17}, {17, 0},
}

// tipIndices are fingertip landmark indices (larger circles).
var tipIndices = map[int]bool{4: true, 8: true, 12: true, 16: true, 20: true}

// DrawLandmarks draws bones and keypoint circles onto img.
func DrawLandmarks(img *gocv.Mat, pts [21][2]int) {
	for _, c := range connections {
		a := image.Pt(pts[c[0]][0], pts[c[0]][1])
		b := image.Pt(pts[c[1]][0], pts[c[1]][1])
		gocv.Line(img, a, b, black, 6)
		gocv.Line(img, a, b, white, 2)
	}
	for i, p := range pts {
		pt := image.Pt(p[0], p[1])
		r := 5
		if tipIndices[i] {
			r = 8
		}
		gocv.Circle(img, pt, r, white, -1)
		gocv.Circle(img, pt, r, black, 1)
	}
}

// DrawBoundingRect draws a bounding rectangle from [x1,y1,x2,y2].
func DrawBoundingRect(img *gocv.Mat, brect [4]int) {
	r := image.Rect(brect[0], brect[1], brect[2], brect[3])
	gocv.Rectangle(img, r, black, 1)
}

// DrawInfoText renders hand sign and gesture labels near the bounding rect.
func DrawInfoText(img *gocv.Mat, brect [4]int, handLabel, signText, gestureText string) {
	// Dark background strip above bounding rect
	strip := image.Rect(brect[0], brect[1]-22, brect[2], brect[1])
	gocv.Rectangle(img, strip, black, -1)

	var infoText string
	if gestureText == "Left" || gestureText == "Right" {
		infoText = "Move:" + gestureText
	} else {
		infoText = "Hand:" + handLabel
		if signText != "" {
			infoText += " Sign:" + signText
		}
	}
	gocv.PutText(img, infoText,
		image.Pt(brect[0]+5, brect[1]-4),
		gocv.FontHersheySimplex, 0.6, white, 1)

	if gestureText != "" {
		label := "Finger Gesture:" + gestureText
		gocv.PutText(img, label, image.Pt(10, 60), gocv.FontHersheySimplex, 1.0, black, 4)
		gocv.PutText(img, label, image.Pt(10, 60), gocv.FontHersheySimplex, 1.0, white, 2)
	}
}

// DrawPointHistory draws the fading pointer trail.
func DrawPointHistory(img *gocv.Mat, pts [][2]int) {
	for i, p := range pts {
		if p[0] == 0 && p[1] == 0 {
			continue
		}
		r := 1 + i/2
		gocv.Circle(img, image.Pt(p[0], p[1]), r, green, 2)
	}
}

// DrawFPS renders the FPS counter in the top-left corner.
func DrawFPS(img *gocv.Mat, fps float64) {
	text := fmt.Sprintf("FPS:%.0f", fps)
	gocv.PutText(img, text, image.Pt(10, 30), gocv.FontHersheySimplex, 1.0, black, 4)
	gocv.PutText(img, text, image.Pt(10, 30), gocv.FontHersheySimplex, 1.0, white, 2)
}
